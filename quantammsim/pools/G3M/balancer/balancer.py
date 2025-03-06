from typing import Dict, Any, Optional
from functools import partial
import numpy as np

# again, this only works on startup!
from jax import config, devices, jit, tree_util
from jax import default_backend
import jax.numpy as jnp
from jax.lax import stop_gradient, dynamic_slice

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.G3M.balancer.balancer_reserves import (
    _jax_calc_balancer_reserve_ratios,
    _jax_calc_balancer_reserves_with_fees_using_precalcs,
    _jax_calc_balancer_reserves_with_dynamic_inputs,
)

config.update("jax_enable_x64", True)

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")


class BalancerPool(AbstractPool):
    """
    Implementation of Balancer's constant-weight liquidity pool.

    Unlike TFMM pools that can adjust weights dynamically, Balancer pools maintain fixed weight 
    ratios between tokens. These weights are determined at initialization and remain constant,
    making the pool non-trainable.

    Core Features:
    --------------
    - Fixed weights (unlike TFMM's dynamic weights)
    - Simple initial weight calculation
    - No parameter processing from web interface needed
    - Non-trainable design

    Calculation Modes:
    ------------------
    1. Standard trading with fees (calculate_reserves_with_fees)
       - Handles regular trading with configurable fees
       - Supports arbitrage simulation with gas costs
       - Uses _jax_calc_balancer_reserves_with_fees_using_precalcs

    2. Zero-fee trading (calculate_reserves_zero_fees)
       - Special case for theoretical analysis
       - Perfect arbitrage simulation
       - Uses _jax_calc_balancer_reserve_ratios

    3. Dynamic input trading (calculate_reserves_with_dynamic_inputs)
       - Supports time-varying fees and parameters
       - Handles custom trade sequences
       - Uses _jax_calc_balancer_reserves_with_dynamic_inputs

    Parameters
    ----------
    params : Dict[str, Any]
        Pool parameters including:
        - initial_weights_logits: Determines fixed token weight ratios

    run_fingerprint : Dict[str, Any]
        Simulation parameters including:
        - initial_pool_value: Starting total value
        - fees: Trading fee percentages
        - gas_cost: Arbitrage threshold
        - arb_fees: Arbitrage-specific fees
        - bout_length: Simulation length
        - n_assets: Number of tokens
        - do_arb: Enable/disable arbitrage
        - arb_frequency: Frequency of arbitrage checks

    Notes
    -----
    - Unlike TFMM pools, no raw_weights_outputs or fine_weight_output methods
    - Simple weight calculation using softmax(initial_weights_logits) if provided
    - Non-trainable by design (is_trainable() returns False)
    - No web interface parameter processing needed (as it has no parameters other than initial_weights_logits)
    - JAX-accelerated calculations for efficiency
    """
    def __init__(self):
        super().__init__()

    @partial(jit, static_argnums=2)
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate reserves considering trading fees and arbitrage costs.

        Uses JAX-accelerated function _jax_calc_balancer_reserves_with_fees_using_precalcs
        for efficient computation. Unlike TFMM pools, this uses constant weights and
        doesn't require raw weight calculations.

        Implementation Notes:
        ---------------------
        1. Extracts local price window using dynamic_slice
        2. Uses constant weights from calculate_initial_weights
        3. Handles arbitrage frequency adjustments
        4. Computes initial reserves based on pool value
        5. Delegates core calculation to jitted external function

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters containing initial_weights_logits or initial_weights
        run_fingerprint : Dict[str, Any]
            Simulation parameters including:
            - bout_length: Length of simulation window
            - n_assets: Number of tokens
            - arb_frequency: How often arbitrage is checked
            - initial_pool_value: Starting pool value
            - fees: Trading fee percentages
            - gas_cost: Arbitrage threshold
            - arb_fees: Arbitrage-specific fees
            - do_arb: Enable/disable arbitrage
        prices : jnp.ndarray
            Price history array
        start_index : jnp.ndarray
            Starting index for the calculation window
        additional_oracle_input : Optional[jnp.ndarray]
            Not used in BalancerPool, kept for interface compatibility

        Returns
        -------
        jnp.ndarray
            Calculated reserves over time
        """
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
            reserves = _jax_calc_balancer_reserves_with_fees_using_precalcs(
                initial_reserves,
                weights,
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

    @partial(jit, static_argnums=2)
    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate reserves assuming zero fees and perfect arbitrage.

        Uses JAX-accelerated function _jax_calc_balancer_reserve_ratios for efficient
        computation in the theoretical zero-fee case. Simpler than TFMM implementation
        due to constant weights.

        Implementation Notes:
        ---------------------
        1. Uses dynamic_slice for price window
        2. Applies constant weights from calculate_initial_weights
        3. Computes reserve ratios directly
        4. Uses cumprod for reserve calculation
        5. Handles no-arbitrage case via broadcasting

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters containing initial_weights_logits or initial_weights
        run_fingerprint : Dict[str, Any]
            Simulation parameters
        prices : jnp.ndarray
            Price history array
        start_index : jnp.ndarray
            Starting index for the calculation window
        additional_oracle_input : Optional[jnp.ndarray]
            Not used in BalancerPool, kept for interface compatibility

        Returns
        -------
        jnp.ndarray
            Calculated reserves over time
        """
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
            reserve_ratios = _jax_calc_balancer_reserve_ratios(
                arb_acted_upon_local_prices[:-1],
                weights,
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

    @partial(jit, static_argnums=2)
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
        """
        Calculate reserves with time-varying fees and parameters.

        Uses JAX-accelerated function _jax_calc_balancer_reserves_with_dynamic_inputs.
        Simpler than TFMM version due to constant weights, but handles dynamic
        parameters for fees and arbitrage thresholds.

        Implementation Notes:
        ---------------------
        1. Handles time-varying parameters via broadcasting
        2. Uses constant weights throughout
        3. Supports custom trade sequences
        4. Maintains arbitrage frequency adjustments
        5. Validates trade array dimensions

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters containing initial_weights_logits or initial_weights
        run_fingerprint : Dict[str, Any]
            Simulation parameters
        prices : jnp.ndarray
            Price history array
        start_index : jnp.ndarray
            Starting index for the calculation window
        fees_array : jnp.ndarray
            Time-varying trading fees
        arb_thresh_array : jnp.ndarray
            Time-varying arbitrage thresholds
        arb_fees_array : jnp.ndarray
            Time-varying arbitrage fees
        trade_array : jnp.ndarray
            Custom trade sequence

        Returns
        -------
        jnp.ndarray
            Calculated reserves over time
        """
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
        reserves = _jax_calc_balancer_reserves_with_dynamic_inputs(
            initial_reserves,
            weights,
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
                            f"{key} must be a singleton or a vector of length n_assets"
                             +  "or a matrix of shape (n_parameter_sets, n_assets)"
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
        """
        Indicate if pool weights can be trained.

        Returns
        -------
        bool
            Always False for BalancerPool as weights are fixed
        """
        return False


tree_util.register_pytree_node(
    BalancerPool,
    BalancerPool._tree_flatten,
    BalancerPool._tree_unflatten,
)
