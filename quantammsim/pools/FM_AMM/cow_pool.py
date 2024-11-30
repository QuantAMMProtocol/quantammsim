
from typing import Dict, Any, Optional

# again, this only works on startup!
from jax import config

from jax.lib.xla_bridge import default_backend
from jax import devices, tree_util

import jax.numpy as jnp
from jax.lax import dynamic_slice

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.FM_AMM.cow_reserves import (
    _jax_calc_cowamm_reserve_ratio_vmapped,
    _jax_calc_cowamm_reserves_with_fees,
    _jax_calc_cowamm_reserves_with_dynamic_fees_and_trades,
)
from quantammsim.core_simulator.param_utils import make_vmap_in_axes_dict

config.update("jax_enable_x64", True)

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")


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

    _init_base_parameters(initial_values_dict, run_fingerprint, n_assets, 
    n_parameter_sets=1, noise="gaussian") -> Dict[str, Any]:
        Initializes the base parameters for the pool. Cow pools have no parameters.

    calculate_weights(params, *args, **kwargs) -> jnp.ndarray:
        Calculates the weights for the assets in the pool. For CowPool, 
        the weights are always [0.5, 0.5].

    make_vmap_in_axes(params, n_repeats_of_recurred=0):
        Creates the vmap in_axes for the parameters.

    is_trainable() -> bool:
        Indicates whether the pool is trainable. For CowPool, this is always False.
    """

    def __init__(self):
        super().__init__()

    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Cow pools have no parameters and are only defined for 2 assets
        assert run_fingerprint["n_assets"] == 2
        weights = self.calculate_weights(params)
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

        reserves = _jax_calc_cowamm_reserves_with_fees(
            initial_reserves,
            arb_acted_upon_local_prices,
            fees=run_fingerprint["fees"],
            arb_thresh=run_fingerprint["gas_cost"],
            arb_fees=run_fingerprint["arb_fees"],
        )
        return reserves

    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Cow pools have no parameters and are only defined for 2 assets
        assert run_fingerprint["n_assets"] == 2
        weights = self.calculate_weights(params)
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

        reserve_ratios = _jax_calc_cowamm_reserve_ratio_vmapped(
            arb_acted_upon_local_prices[:-1], arb_acted_upon_local_prices[1:]
        )
        reserves = jnp.vstack(
            [
                initial_reserves,
                initial_reserves * jnp.cumprod(reserve_ratios, axis=0),
            ]
        )

        return reserves

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
        # Cow pools have no parameters and are only defined for 2 assets
        assert run_fingerprint["n_assets"] == 2
        weights = self.calculate_weights(params)
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

        reserves = _jax_calc_cowamm_reserves_with_dynamic_fees_and_trades(
            initial_reserves,
            arb_acted_upon_local_prices,
            fees=fees_array_broadcast,
            arb_thresh=arb_thresh_array_broadcast,
            arb_fees=arb_fees_array_broadcast,
            trades=trade_array,
        )
        return reserves

    def _init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        # Cow pools have no parameters
        return {}

    def calculate_weights(
        self, params: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        return jnp.array([0.5, 0.5])

    def make_vmap_in_axes(self, params: Dict[str, Any], n_repeats_of_recurred: int = 0):
        return make_vmap_in_axes_dict(params, 0, [], [], n_repeats_of_recurred)

    def is_trainable(self):
        return False


tree_util.register_pytree_node(
    CowPool,
    CowPool._tree_flatten,
    CowPool._tree_unflatten,
)
