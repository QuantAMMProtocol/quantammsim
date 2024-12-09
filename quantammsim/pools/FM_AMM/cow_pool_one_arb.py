from typing import Dict, Any, Optional

# again, this only works on startup!
from jax import config

from jax.lib.xla_bridge import default_backend
from jax import devices

import jax.numpy as jnp
from jax import tree_util
from jax.lax import dynamic_slice

from quantammsim.pools.FM_AMM.cow_pool import CowPool
from quantammsim.pools.FM_AMM.cow_reserves import (
    _jax_calc_cowamm_reserves_under_attack_zero_fees,
    _jax_calc_cowamm_reserves_under_attack_with_fees,
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

class CowPoolOneArb(CowPool):
    """
    CowPoolOneArb is a subclass of CowPool that represents a pool with a single arbitrageur.
    
    Methods
    -------
    __init__():
        Initializes the CowPoolOneArb instance.
    
    calculate_reserves_with_fees(params, run_fingerprint, prices, start_index, 
    additional_oracle_input=None):
        Calculates the reserves of the pool considering fees.
        
        Parameters:
        - params (Dict[str, Any]): Parameters for the pool.
        - run_fingerprint (Dict[str, Any]): Run-specific parameters.
        - prices (jnp.ndarray): Array of prices.
        - start_index (jnp.ndarray): Starting index for slicing prices.
        - additional_oracle_input (Optional[jnp.ndarray]): Additional input for the oracle.
        
        Returns:
        - jnp.ndarray: Calculated reserves.
    
    calculate_reserves_zero_fees(params, run_fingerprint, prices, start_index, 
    additional_oracle_input=None):
        Calculates the reserves of the pool without considering fees.
        
        Parameters:
        - params (Dict[str, Any]): Parameters for the pool.
        - run_fingerprint (Dict[str, Any]): Run-specific parameters.
        - prices (jnp.ndarray): Array of prices.
        - start_index (jnp.ndarray): Starting index for slicing prices.
        - additional_oracle_input (Optional[jnp.ndarray]): Additional input for the oracle.
        
        Returns:
        - jnp.ndarray: Calculated reserves.
    
    calculate_reserves_with_dynamic_inputs(params, run_fingerprint, prices, start_index, 
    fees_array, arb_thresh_array, arb_fees_array, trade_array, additional_oracle_input=None):
        Raises NotImplementedError as dynamic inputs are not implemented 
        for COW pools with only a single arbitrageur.
        
        Parameters:
        - params (Dict[str, Any]): Parameters for the pool.
        - run_fingerprint (Dict[str, Any]): Run-specific parameters.
        - prices (jnp.ndarray): Array of prices.
        - start_index (jnp.ndarray): Starting index for slicing prices.
        - fees_array (jnp.ndarray): Array of fees.
        - arb_thresh_array (jnp.ndarray): Array of arbitrage thresholds.
        - arb_fees_array (jnp.ndarray): Array of arbitrage fees.
        - trade_array (jnp.ndarray): Array of trades.
        - additional_oracle_input (Optional[jnp.ndarray]): Additional input for the oracle.
        
        Returns:
        - jnp.ndarray: Calculated reserves.
    
    make_vmap_in_axes(params, n_repeats_of_recurred=0):
        Creates a dictionary for vectorized mapping of input axes.
        
        Parameters:
        - params (Dict[str, Any]): Parameters for the pool.
        - n_repeats_of_recurred (int): Number of repeats of recurred.
        
        Returns:
        - dict: Dictionary for vectorized mapping of input axes.
    
    is_trainable():
        Indicates whether the pool is trainable.
        
        Returns:
        - bool: False, as the pool is not trainable.
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

        reserves = _jax_calc_cowamm_reserves_under_attack_with_fees(
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

        # calculate the reserves by cumprod of reserve ratios
        reserves = _jax_calc_cowamm_reserves_under_attack_zero_fees(
            initial_reserves,
            arb_acted_upon_local_prices,
            arb_thresh=run_fingerprint["gas_cost"],
            arb_fees=run_fingerprint["arb_fees"],
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
        raise NotImplementedError(
            "Dynamic inputs not implemented for COW pools with only a single arbitrageur."
        )

    def make_vmap_in_axes(self, params: Dict[str, Any], n_repeats_of_recurred: int = 0):
        return make_vmap_in_axes_dict(params, 0, [], [], n_repeats_of_recurred)

    def is_trainable(self):
        return False


tree_util.register_pytree_node(
    CowPoolOneArb,
    CowPoolOneArb._tree_flatten,
    CowPoolOneArb._tree_unflatten,
)
