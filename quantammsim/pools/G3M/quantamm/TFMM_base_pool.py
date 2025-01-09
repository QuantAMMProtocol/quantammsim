# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
from jax.lib.xla_bridge import default_backend
from jax import local_device_count, devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import jit, vmap
from jax import devices, device_put
from jax.lax import stop_gradient, dynamic_slice
from jax.nn import softmax

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.G3M.quantamm.quantamm_reserves import (
    _jax_calc_quantAMM_reserve_ratios,
    _jax_calc_quantAMM_reserves_with_fees_using_precalcs,
    _jax_calc_quantAMM_reserves_with_dynamic_inputs,
)
from quantammsim.core_simulator.param_utils import make_vmap_in_axes_dict

import numpy as np

from typing import Dict, Any, Optional
from functools import partial
from abc import abstractmethod


class TFMMBasePool(AbstractPool):
    """
    TFMMBasePool is an abstract base class for implementing TFMM (Temporal Function Market Making) liquidity pools.

    This class extends the AbstractPool class and provides a foundation for specific TFMM pool implementations.
    It defines additional abstract methods that are specific to TFMM pools, such as weight calculation.

    Abstract Methods:
        calculate_weights: Calculate the weights of assets in the pool based on prices and parameters.

    In addition to the methods from AbstractPool, subclasses of TFMMBasePool must implement these
    TFMM-specific methods to define the behavior of the pool.

    Note:
        This class is designed to be subclassed, not instantiated directly. Concrete implementations
        should provide specific logic for weight calculation and slippage estimation. It is reccomended
        to implement the functions used within implementations of these methods as external JAX functions
        that are jitted and then used within pool methods. This separation of concerns comes from that JAX
        is a functional programming language and we want to keep the pool methods pure. Finally, note that due
        to this separation of concerns this class does not hold any state, for example pool parameters.
    """

    def __init__(self):
        """
        Initialize a new TFMMBasePool instance.
        """
        super().__init__()

    @partial(jit, static_argnums=(2))
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        weights = self.calculate_weights(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )
        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_weights = weights[:: run_fingerprint["arb_frequency"]]
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_weights = weights
            arb_acted_upon_local_prices = local_prices

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = arb_acted_upon_weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]
        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_quantAMM_reserves_with_fees_using_precalcs(
                initial_reserves,
                arb_acted_upon_weights,
                arb_acted_upon_local_prices,
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves, arb_acted_upon_local_prices.shape
            )

        return reserves

    @partial(jit, static_argnums=(2))
    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        weights = self.calculate_weights(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        if run_fingerprint["do_arb"]:
            if run_fingerprint["arb_frequency"] != 1:
                arb_acted_upon_weights = weights[:: run_fingerprint["arb_frequency"]]
                arb_acted_upon_local_prices = local_prices[:: run_fingerprint["arb_frequency"]]
            else:
                arb_acted_upon_weights = weights
                arb_acted_upon_local_prices = local_prices

            reserve_ratios = _jax_calc_quantAMM_reserve_ratios(
                arb_acted_upon_weights[:-1],
                arb_acted_upon_local_prices[:-1],
                arb_acted_upon_weights[1:],
                arb_acted_upon_local_prices[1:],
            )

            # calculate the reserves by cumprod of reserve ratios
            reserves = jnp.vstack(
                [
                    initial_reserves,
                    initial_reserves * jnp.cumprod(reserve_ratios, axis=0),
                ]
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves, arb_acted_upon_local_prices.shape
            )

        return reserves

    @partial(jit, static_argnums=(2))
    def calculate_reserves_with_dynamic_inputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        fees_array: jnp.ndarray,
        arb_thresh_array: jnp.ndarray,
        arb_fees_array: jnp.ndarray,
        trade_array: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        weights = self.calculate_weights(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )
        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_weights = weights[:: run_fingerprint["arb_frequency"]]
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_weights = weights
            arb_acted_upon_local_prices = local_prices

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = arb_acted_upon_weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]

        # any of fees_array, arb_thresh_array, arb_fees_array, trade_array
        # can be singletons, in which case we repeat them for the length of the bout

        # Determine the maximum leading dimension
        max_len = bout_length - 1
        if run_fingerprint["arb_frequency"] != 1:
            max_len = max_len // run_fingerprint["arb_frequency"]
        # Broadcast input arrays to match the maximum leading dimension.
        # If they are singletons, this will just repeat them for the length of the bout.
        # If they are arrays of length bout_length, this will cause no change.
        fees_array_broadcast = jnp.broadcast_to(
            fees_array, (max_len,) + fees_array.shape[1:]
        )
        arb_thresh_array_broadcast = jnp.broadcast_to(
            arb_thresh_array, (max_len,) + arb_thresh_array.shape[1:]
        )
        arb_fees_array_broadcast = jnp.broadcast_to(
            arb_fees_array, (max_len,) + arb_fees_array.shape[1:]
        )
        # if we are doing trades, the trades array must be of the same length as the other arrays
        if run_fingerprint["do_trades"]:
            assert trade_array.shape[0] == max_len
        reserves = _jax_calc_quantAMM_reserves_with_dynamic_inputs(
            initial_reserves,
            arb_acted_upon_weights,
            arb_acted_upon_local_prices,
            fees_array_broadcast,
            arb_thresh_array_broadcast,
            arb_fees_array_broadcast,
            jnp.array(run_fingerprint["all_sig_variations"]),
            trade_array,
            run_fingerprint["do_trades"],
            run_fingerprint["do_arb"],
        )
        return reserves

    @abstractmethod
    def calculate_raw_weights_outputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def fine_weight_output(
        self,
        raw_weight_output: jnp.ndarray,
        initial_weights: jnp.ndarray,
        run_fingerprint: Dict[str, Any],
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=(2, 5))
    def calculate_weights(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Calculate the weights of assets in the pool.

        This method should be implemented by subclasses to define how weights are calculated
        based on current prices, pool parameters, and optional additional oracle input.

        Args:
            prices (jnp.ndarray): Current prices of the assets.
            params (Dict[str, Any]): Pool parameters.
            additional_oracle_input (Optional[jnp.ndarray], optional): Additional input from an oracle. Defaults to None.

        Returns:
            jnp.ndarray: Calculated weights for each asset in the pool.
        """
        chunk_period = run_fingerprint["chunk_period"]
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        raw_weight_outputs = self.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, additional_oracle_input
        )

        initial_weights_logits = params.get("initial_weights_logits")
        # we dont't want to change the initial weights during any training
        # so wrap them in a stop_grad
        initial_weights = softmax(stop_gradient(initial_weights_logits))

        # we have a sequence now of weight changes, but if we are doing
        # a burnin operation, we need to cut off the changes associated
        # with the burnin period, ie everything before the start of the sequence

        start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)
        raw_weight_outputs = dynamic_slice(
            raw_weight_outputs,
            start_index_coarse,
            (int((bout_length) / chunk_period), n_assets),
        )
        raw_weight_outputs_cpu = device_put(raw_weight_outputs, CPU_DEVICE)
        initial_weights_cpu = device_put(initial_weights, CPU_DEVICE)

        weights = self.fine_weight_output(
            raw_weight_outputs_cpu,
            initial_weights_cpu,
            run_fingerprint,
            params,
        )
        weights = dynamic_slice(weights, (0, 0), (bout_length - 1, n_assets))
        # initial_value_per_token = initial_weights * initial_pool_value
        # initial_reserves = initial_value_per_token / prices[start_index]

        return weights

    def calculate_all_signature_variations(self, params: Dict[str, Any]) -> jnp.ndarray:
        raise NotImplementedError

    def make_vmap_in_axes(self, params: Dict[str, Any], n_repeats_of_recurred: int = 0):
        return make_vmap_in_axes_dict(
            params, 0, [], ["subsidary_params"], n_repeats_of_recurred
        )

    def is_trainable(self):
        return True
