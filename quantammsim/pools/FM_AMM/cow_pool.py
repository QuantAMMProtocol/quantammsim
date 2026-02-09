"""CoW AMM (Function-Maximising AMM) pool implementation.

Implements the CoW Protocol's FM-AMM, which replaces the continuous
arbitrage of traditional AMMs with batch-auction-based rebalancing.
The pool computes reserves under both perfect and single-arbitrageur
models, blended by an ``arb_quality`` parameter, and supports dynamic
fees, external trades, and noise traders.
"""
from typing import Dict, Any, Optional

# again, this only works on startup!
from jax import config

from jax import default_backend
from jax import devices, tree_util

config.update("jax_enable_x64", True)

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
from jax import device_put
from jax import tree_util
from jax.lax import stop_gradient, dynamic_slice

from typing import Dict, Any, Optional, Callable
from functools import partial
import numpy as np

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.FM_AMM.cow_reserves import (
    _jax_calc_cowamm_reserve_ratio_vmapped,
    _jax_calc_cowamm_reserves_with_fees,
    _jax_calc_cowamm_reserves_one_arb_zero_fees,
    _jax_calc_cowamm_reserves_one_arb_with_fees,
    _jax_calc_cowamm_reserves_with_dynamic_inputs,
)

class CowPool(AbstractPool):
    """
    A class representing a CowPool, which is a type of automated market maker (AMM) pool
     with specific characteristics.

    Methods
    -------
    __init__():
        Initializes the CowPool instance.

    calculate_reserves_with_fees(params, run_fingerprint, prices, 
    start_index, additional_oracle_input=None) -> jnp.ndarray:
        Calculates the reserves of the pool considering fees.

    calculate_reserves_zero_fees(params, run_fingerprint, prices, 
    start_index, additional_oracle_input=None) -> jnp.ndarray:
        Calculates the reserves of the pool without considering fees.

    calculate_reserves_with_dynamic_inputs(params, run_fingerprint, prices, 
    start_index, fees_array, arb_thresh_array, arb_fees_array, trade_array, 
    additional_oracle_input=None) -> jnp.ndarray:
        Calculates the reserves of the pool with dynamic inputs for fees, 
        arbitrage thresholds, arbitrage fees, and trades.

    init_base_parameters(initial_values_dict, run_fingerprint, n_assets, 
    n_parameter_sets=1, noise="gaussian") -> Dict[str, Any]:
        Initializes the base parameters for the pool. Cow pools have no parameters.

    calculate_initial_weights(params, *args, **kwargs) -> jnp.ndarray:
        Calculates the weights for the assets in the pool. For CowPool, 
        the weights are always [0.5, 0.5].

    make_vmap_in_axes(params, n_repeats_of_recurred=0):
        Creates the vmap in_axes for the parameters.

    is_trainable() -> bool:
        Indicates whether the pool is trainable. For CowPool, this is always False.
    """

    def __init__(self):
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
        weights = self.calculate_initial_weights(params)
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[:: run_fingerprint["arb_frequency"]]
        else:
            arb_acted_upon_local_prices = local_prices

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        if run_fingerprint["do_arb"]:
            reserves_with_perfect_arbs = _jax_calc_cowamm_reserves_with_fees(
                initial_reserves,
                arb_acted_upon_local_prices,
                weight=weights[0],
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                noise_trader_ratio=run_fingerprint["noise_trader_ratio"],
            )
            # now we need to calculate the reserves with imperfect arbs
            # we do this by taking the perfect arbs and then applying a small
            # amount of noise to the weights
            reserves_with_one_arb = _jax_calc_cowamm_reserves_one_arb_with_fees(
                initial_reserves,
                arb_acted_upon_local_prices,
                weight=weights[0],
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                noise_trader_ratio=run_fingerprint["noise_trader_ratio"],
            )
            reserves = (
                run_fingerprint["arb_quality"] * reserves_with_perfect_arbs
                + (1.0 - run_fingerprint["arb_quality"]) * reserves_with_one_arb
            )
        else:
            reserves = jnp.broadcast_to(initial_reserves, local_prices.shape)

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
        weights = self.calculate_initial_weights(params)
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        if run_fingerprint["do_arb"]:
            reserves_with_perfect_arbs = (
                _jax_calc_cowamm_reserves_with_fees(
                    initial_reserves,
                    arb_acted_upon_local_prices,
                    weight=weights[0],
                    fees=0.0,
                    arb_thresh=run_fingerprint["gas_cost"],
                    arb_fees=run_fingerprint["arb_fees"],
                )
            )
            reserves_with_one_arb = _jax_calc_cowamm_reserves_one_arb_zero_fees(
                initial_reserves,
                arb_acted_upon_local_prices,
                weight=weights[0],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
            )
            reserves = (
                run_fingerprint["arb_quality"] * reserves_with_perfect_arbs
                + (1.0 - run_fingerprint["arb_quality"]) * reserves_with_one_arb
            )
        else:
            reserves = jnp.broadcast_to(initial_reserves, local_prices.shape)
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
        lp_supply_array: jnp.ndarray = None,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        weights = self.calculate_initial_weights(params)

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
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

        reserves = _jax_calc_cowamm_reserves_with_dynamic_inputs(
            initial_reserves,
            arb_acted_upon_local_prices,
            fees_array_broadcast,
            arb_thresh_array_broadcast,
            arb_fees_array_broadcast,
            weights,
            run_fingerprint["arb_quality"],
            trade_array,
            run_fingerprint["do_trades"],
            run_fingerprint["do_arb"],
            noise_trader_ratio=run_fingerprint["noise_trader_ratio"],
        )
        return reserves

    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        np.random.seed(0)

        # We need to initialise the weights for each parameter set
        # If a vector is provided in the inital values dict, we use
        # that, if only a singleton array is provided we expand it
        # to n_assets and use that vlaue for all assets.
        def process_initial_values(
            initial_values_dict, key, n_assets, n_parameter_sets
        ):
            if key in initial_values_dict:
                initial_value = initial_values_dict[key]
                if isinstance(initial_value, (np.ndarray, jnp.ndarray, list)):
                    initial_value = np.array(initial_value)
                    if initial_value.size == n_assets:
                        return np.array([initial_value] * n_parameter_sets)
                    elif initial_value.size == 1:
                        return np.array([[initial_value] * n_assets] * n_parameter_sets)
                    elif initial_value.shape == (n_parameter_sets, n_assets):
                        return initial_value
                    else:
                        raise ValueError(
                            f"{key} must be a singleton or a vector of length n_assets or a matrix of shape (n_parameter_sets, n_assets)"
                        )
                else:
                    return np.array([[initial_value] * n_assets] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        initial_weights_logits = process_initial_values(
            initial_values_dict, "initial_weights_logits", n_assets, n_parameter_sets
        )
        params = {
            "initial_weights_logits": initial_weights_logits,
        }
        params = self.add_noise(params, noise, n_parameter_sets)
        return params

    def is_trainable(self):
        return False


tree_util.register_pytree_node(
    CowPool,
    CowPool._tree_flatten,
    CowPool._tree_unflatten,
)
