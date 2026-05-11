from typing import Dict, Any, Optional
from functools import partial
import numpy as np

# again, this only works on startup!
from jax import config, devices, jit, tree_util
from jax import default_backend
import jax.numpy as jnp
from jax.lax import dynamic_slice

from quantammsim.core_simulator.dynamic_inputs import materialize_dynamic_inputs
from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.G3M.balancer.balancer_reserves import (
    _jax_calc_balancer_reserve_ratios,
    _jax_calc_balancer_reserves_with_fees_using_precalcs,
    _jax_calc_balancer_reserves_with_dynamic_inputs,
)
from quantammsim.pools.reCLAMM.reclamm import _prepare_dynamic_array

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

        noise_model = run_fingerprint.get("noise_model", "ratio")
        noise_params = run_fingerprint.get("reclamm_noise_params", None)
        if noise_params is not None and type(noise_params) is not dict:
            noise_params = dict(noise_params)

        arb_freq = run_fingerprint["arb_frequency"]
        max_len = arb_acted_upon_local_prices.shape[0]
        _na = self._prepare_noise_arrays(
            prices, run_fingerprint, start_index,
            bout_length, arb_freq, max_len,
        )

        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_balancer_reserves_with_fees_using_precalcs(
                initial_reserves,
                weights,
                arb_acted_upon_local_prices,
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
                noise_model=noise_model,
                noise_trader_ratio=run_fingerprint.get("noise_trader_ratio", 0.0),
                protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
                seconds_per_step=float(arb_freq) * 60.0,
                noise_params=noise_params,
                volatility_array=_na.get("volatility"),
                dow_sin_array=_na.get("dow_sin"),
                dow_cos_array=_na.get("dow_cos"),
                noise_base_array=_na.get("noise_base"),
                noise_tvl_coeff_array=_na.get("noise_tvl_coeff"),
                competitor_tvl_array=_na.get("competitor_tvl"),
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
        dynamic_inputs,
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
        dynamic_inputs : DynamicInputArrays
            Fixed-structure bundle of dynamic inputs.

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

        max_len = bout_length - 1
        if run_fingerprint["arb_frequency"] != 1:
            max_len = max_len // run_fingerprint["arb_frequency"]
        materialized_inputs = materialize_dynamic_inputs(
            dynamic_inputs,
            run_fingerprint.get("dynamic_input_flags"),
            run_fingerprint,
            scan_len=max_len,
            do_trades=run_fingerprint["do_trades"],
            dtype=arb_acted_upon_local_prices.dtype,
        )
        noise_model = run_fingerprint.get("noise_model", "ratio")
        noise_arrays = self._prepare_noise_arrays(
            prices, run_fingerprint, start_index,
            bout_length, run_fingerprint["arb_frequency"], max_len,
        )

        noise_params = run_fingerprint.get("reclamm_noise_params", None)
        if noise_params is not None and type(noise_params) is not dict:
            noise_params = dict(noise_params)

        reserves = _jax_calc_balancer_reserves_with_dynamic_inputs(
            initial_reserves,
            weights,
            arb_acted_upon_local_prices,
            materialized_inputs.fees,
            materialized_inputs.gas_cost,
            materialized_inputs.arb_fees,
            jnp.array(run_fingerprint["all_sig_variations"]),
            materialized_inputs.trades,
            run_fingerprint["do_trades"],
            run_fingerprint["do_arb"],
            noise_model,
            materialized_inputs.lp_supply,
            protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
            noise_trader_ratio=run_fingerprint.get("noise_trader_ratio", 0.0),
            seconds_per_step=float(run_fingerprint.get("arb_frequency", 1)) * 60.0,
            noise_params=noise_params,
            volatility_array=noise_arrays.get("volatility"),
            dow_sin_array=noise_arrays.get("dow_sin"),
            dow_cos_array=noise_arrays.get("dow_cos"),
            noise_base_array=noise_arrays.get("noise_base"),
            noise_tvl_coeff_array=noise_arrays.get("noise_tvl_coeff"),
            competitor_tvl_array=noise_arrays.get("competitor_tvl"),
        )
        return reserves

    def _prepare_noise_arrays(self, prices, run_fingerprint, start_index,
                              bout_length, arb_freq, max_len):
        """Prepare noise arrays — identical to ReClammPool._prepare_noise_arrays.

        The noise model describes market-level organic volume, not pool
        mechanics, so the same implementation applies to any pool type.
        """
        import numpy as np
        noise_model = run_fingerprint.get("noise_model", "ratio")
        result = {"volatility": None, "dow_sin": None, "dow_cos": None,
                  "noise_base": None, "noise_tvl_coeff": None,
                  "competitor_tvl": None}

        if noise_model == "mm_observed":
            nb = run_fingerprint.get("noise_base_array")
            ct = run_fingerprint.get("competitor_tvl_array")
            if nb is None and "noise_arrays_path" in run_fingerprint:
                path = run_fingerprint["noise_arrays_path"]
                if not hasattr(self, "_mm_observed_cache") or self._mm_observed_cache[0] != path:
                    arrays = np.load(path)
                    self._mm_observed_cache = (
                        path, arrays["noise_base"], arrays["competitor_tvl"])
                nb = self._mm_observed_cache[1]
                ct = self._mm_observed_cache[2]
            if nb is not None:
                result["noise_base"] = _prepare_dynamic_array(
                    jnp.array(nb), start_index, bout_length, arb_freq, max_len)
            if ct is not None:
                result["competitor_tvl"] = _prepare_dynamic_array(
                    jnp.array(ct), start_index, bout_length, arb_freq, max_len)
            return result

        if noise_model == "market_linear":
            nb = run_fingerprint.get("noise_base_array")
            ntc = run_fingerprint.get("noise_tvl_coeff_array")
            if nb is None and "noise_arrays_path" in run_fingerprint:
                path = run_fingerprint["noise_arrays_path"]
                if not hasattr(self, "_market_linear_cache") or self._market_linear_cache[0] != path:
                    arrays = np.load(path)
                    self._market_linear_cache = (path, arrays["noise_base"], arrays["noise_tvl_coeff"])
                nb = self._market_linear_cache[1]
                ntc = self._market_linear_cache[2]
            if nb is not None:
                result["noise_base"] = _prepare_dynamic_array(
                    jnp.array(nb), start_index, bout_length, arb_freq, max_len)
            if ntc is not None:
                result["noise_tvl_coeff"] = _prepare_dynamic_array(
                    jnp.array(ntc), start_index, bout_length, arb_freq, max_len)
            return result

        needs_vol = noise_model in (
            "tsoukalas_sqrt", "tsoukalas_log", "loglinear", "calibrated",
        )
        if not needs_vol:
            return result

        volatility_array = self.calculate_volatility_array(
            prices, run_fingerprint,
        )
        result["volatility"] = _prepare_dynamic_array(
            volatility_array, start_index, bout_length, arb_freq, max_len,
        )

        if noise_model != "calibrated":
            return result

        # Day-of-week sin/cos arrays for the calibrated noise model.
        import pandas as pd
        start_dt = pd.Timestamp(run_fingerprint["startDateString"])
        n_minutes = prices.shape[0]
        day_indices = np.arange(n_minutes) // 1440
        start_weekday = start_dt.weekday()
        weekdays = ((start_weekday + day_indices) % 7).astype(np.float64)
        dow_sin_full = jnp.array(np.sin(2.0 * np.pi * weekdays / 7.0))
        dow_cos_full = jnp.array(np.cos(2.0 * np.pi * weekdays / 7.0))
        result["dow_sin"] = _prepare_dynamic_array(
            dow_sin_full, start_index, bout_length, arb_freq, max_len,
        )
        result["dow_cos"] = _prepare_dynamic_array(
            dow_cos_full, start_index, bout_length, arb_freq, max_len,
        )
        return result

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
