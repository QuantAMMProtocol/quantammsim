# again, this only works on startup!
from jax import config

from jax import default_backend

DEFAULT_BACKEND = default_backend()

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import stop_gradient, dynamic_slice, scan, fori_loop
from jax.tree_util import Partial

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.G3M.quantamm.quantamm_reserves import (
    _jax_calc_quantAMM_reserve_ratios,
    _jax_calc_quantAMM_reserves_with_fees_using_precalcs,
    _jax_calc_quantAMM_reserves_with_dynamic_inputs,
    _fused_chunked_reserves,
)
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    _jax_calc_coarse_weights,
    _jax_calc_coarse_weight_scan_function,
    calc_coarse_weight_output_from_weight_changes,
    calc_coarse_weight_output_from_weights,
    scale_diff,
    ste,
)
from quantammsim.pools.G3M.quantamm.weight_calculations.linear_interpolation import (
    _jax_calc_linear_interpolation_block,
)
from quantammsim.pools.G3M.quantamm.weight_calculations.non_linear_interpolation import (
    _jax_calc_approx_optimal_interpolation_block,
)
from quantammsim.core_simulator.param_utils import make_vmap_in_axes_dict
from quantammsim.core_simulator.param_utils import memory_days_to_lamb
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
        calculate_rule_outputs: Calculate the raw weight outputs of assets in the pool based on oracle values and parameters.
        calculate_fine_weights: Function to handle how raw weights get mapped to per-block/per-minute weights. Two standard methods
        are provided, for when 1) rules output raw weight _changes_ and 2) when rule output raw _weights_ themselves. See MomentumPool
        and MinVariancePool as prototypical examples of each respectively.

    In addition to the methods from AbstractPool, subclasses of TFMMBasePool must implement these
    TFMM-specific methods to define the behavior of the pool.

    Note:
        This class is designed to be subclassed, not instantiated directly. Concrete implementations
        should provide specific logic for weight calculation and slippage estimation. It is recommended
        to implement the functions used within implementations of these methods as external JAX functions
        that are jitted and then used within pool methods. This separation of concerns comes from that JAX
        is a functional programming language and we want to keep the pool methods pure. Finally, note that due
        to this separation of concerns this class does not hold any state, for example pool parameters.
    """

    # Subclasses must set this: True if calculate_fine_weights uses
    # calc_fine_weight_output_from_weights (target-weight rules like min_variance),
    # False if it uses calc_fine_weight_output_from_weight_changes (delta rules
    # like momentum). Needed by the fused reserve path to handle the
    # initial-weight block prepended by delta-based pools.
    _rule_outputs_are_weights = False  # default; overridden in weight-based subclasses

    @property
    def supports_fused_reserves(self) -> bool:
        """Whether this pool supports the fused chunked reserve computation path."""
        return True

    def __init__(self):
        """
        Initialize a new TFMMBasePool instance.
        """
        super().__init__()

    @partial(jit, static_argnums=(2, 6, 7, 8))
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: Optional[jnp.ndarray] = None,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        weights: Optional[jnp.ndarray] = None,
        local_prices: Optional[jnp.ndarray] = None,
        initial_reserves: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate reserves with fees and dynamic weights.

        TFMM pools calculate weights dynamically based on price history.
        This method handles the full complexity of weight adjustments, fees, and arbitrage.

        Implementation Steps:
        ---------------------
        1. Extract local price window
        2. Calculate dynamic weights based on price history
        3. Apply arbitrage frequency adjustments
        4. Initialize reserves based on pool value
        5. Calculate reserve changes using quantAMM precalcs

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters including weight calculation parameters
        run_fingerprint : Dict[str, Any]
            Simulation settings including:
            - bout_length: Simulation window length
            - n_assets: Number of tokens
            - arb_frequency: Arbitrage check frequency
            - initial_pool_value: Starting pool value
            - fees: Trading fees
            - gas_cost: Arbitrage threshold
            - arb_fees: Arbitrage fees
            - do_arb: Enable arbitrage
            - all_sig_variations: Valid trade combinations
        prices : jnp.ndarray
            Historical price data
        start_index : jnp.ndarray
            Window start position
        additional_oracle_input : Optional[jnp.ndarray]
            Extra data for weight calculation

        Returns
        -------
        jnp.ndarray
            Time series of pool reserves
        """
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        if local_prices is None:
            local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        else:
            local_prices = local_prices.val

        if weights is None:
            weights = self.calculate_weights(
                params, run_fingerprint, prices, start_index, additional_oracle_input
            )
        else:
            weights = weights.val
        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_weights = weights[:: run_fingerprint["arb_frequency"]]
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_weights = weights
            arb_acted_upon_local_prices = local_prices

        if initial_reserves is None:
            initial_pool_value = run_fingerprint["initial_pool_value"]
            initial_value_per_token = arb_acted_upon_weights[0] * initial_pool_value
            initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]
        else:
            initial_reserves = initial_reserves.val

        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_quantAMM_reserves_with_fees_using_precalcs(
                initial_reserves,
                arb_acted_upon_weights,
                arb_acted_upon_local_prices,
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
                noise_trader_ratio=run_fingerprint["noise_trader_ratio"],
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

    def calculate_fused_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Compute metric-cadence boundary values via the fused chunked path.

        This method avoids materialising the full ``(T_fine, n_assets)`` weight
        and reserve arrays by computing per-chunk interpolation + ratio products
        inline, then aggregating to metric-period (e.g. daily) granularity.

        Parameters
        ----------
        params, run_fingerprint, prices, start_index, additional_oracle_input
            Same as :meth:`calculate_reserves_zero_fees`.

        Returns
        -------
        dict with keys:
            ``boundary_values`` : (n_metric_periods + 1,)
                Pool values at metric-period boundaries (e.g. daily).
                ``boundary_values[0]`` = initial value, ``boundary_values[k]``
                = value at end of metric period k.
            ``final_reserves`` : (n_assets,)
            ``initial_reserves`` : (n_assets,)
            ``boundary_prices`` : (n_metric_periods + 1, n_assets)
        """
        chunk_period = run_fingerprint["chunk_period"]
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        weight_interpolation_method = run_fingerprint.get(
            "weight_interpolation_method", "linear"
        )
        metric_period = 1440  # always daily for fused path
        interpol_num = run_fingerprint["weight_interpolation_period"] + 1
        chunks_per_metric = metric_period // chunk_period

        rule_outputs_are_weights = self._rule_outputs_are_weights

        # --- How many daily values / metric periods? ---
        # Full-resolution values has (bout_length - 1) entries.
        # daily_values = values[::metric_period] samples at indices
        # 0, metric_period, 2*metric_period, ... up to bout_length - 2.
        n_daily_values = (bout_length - 2) // metric_period + 1
        n_metric_periods = n_daily_values - 1

        # --- How many coarse chunks do we need? ---
        # n_chunks_total: chunks that cover the metric periods
        #   (includes virtual block for delta pools).
        n_chunks_total = n_metric_periods * chunks_per_metric

        # --- Rule outputs → coarse weights ---
        # CRITICAL: slice rule_outputs with the SAME size as
        # calculate_weights_vectorized to get identical dynamic_slice
        # clipping behaviour.  JAX's dynamic_slice clips the start index
        # when the requested window would exceed the array bounds, so
        # requesting a different size from the same start can yield a
        # different effective start.
        raw_weight_additional_offset = 0 if bout_length % chunk_period == 0 else 1
        n_coarse_for_slice = int(bout_length / chunk_period) + raw_weight_additional_offset

        rule_outputs = self.calculate_rule_outputs(
            params, run_fingerprint, prices, additional_oracle_input
        )
        initial_weights = self.calculate_initial_weights(params)

        start_index_coarse = (start_index[0] / chunk_period).astype("int64")
        rule_outputs = dynamic_slice(
            rule_outputs,
            (start_index_coarse, 0),
            (n_coarse_for_slice, n_assets),
        )

        # Get coarse weights
        if rule_outputs_are_weights:
            actual_starts, scaled_diffs = calc_coarse_weight_output_from_weights(
                rule_outputs, initial_weights, run_fingerprint, params,
            )
        else:
            actual_starts, scaled_diffs = calc_coarse_weight_output_from_weight_changes(
                rule_outputs, initial_weights, run_fingerprint, params,
            )

        # --- Local prices for the bout ---
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        # Initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = initial_weights * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        # --- Select interpolation function ---
        if weight_interpolation_method == "linear":
            interpolation_fn = _jax_calc_linear_interpolation_block
        elif weight_interpolation_method == "approx_optimal":
            interpolation_fn = _jax_calc_approx_optimal_interpolation_block
        else:
            raise ValueError(
                f"Invalid interpolation method: {weight_interpolation_method}"
            )

        checkpoint_mode = run_fingerprint.get("checkpoint_fused", "none")

        boundary_values, final_reserves = _fused_chunked_reserves(
            actual_starts, scaled_diffs, local_prices, initial_reserves,
            initial_weights,
            chunk_period, interpol_num, metric_period,
            interpolation_fn, rule_outputs_are_weights,
            n_chunks_total, n_metric_periods,
            checkpoint_mode,
        )

        return {
            "boundary_values": boundary_values,
            "final_reserves": final_reserves,
            "initial_reserves": initial_reserves,
        }

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

        # any of fees_array, arb_thresh_array, arb_fees_array, trade_array, and lp_supply_array
        # can be singletons, in which case we repeat them for the length of the bout.

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
        # if lp_supply_array is not provided, we set it to a constant of 1.0
        if lp_supply_array is None:
            lp_supply_array = jnp.array(1.0)

        lp_supply_array_broadcast = jnp.broadcast_to(
            lp_supply_array, (max_len,) + lp_supply_array.shape[1:]
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
            run_fingerprint["noise_trader_ratio"],
            lp_supply_array_broadcast,
        )
        return reserves

    def calculate_rule_outputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate raw weight adjustments based on price history (vectorized path).

        This is the first step in TFMM's two-step weight calculation process.
        Subclasses should implement either this method OR calculate_rule_output_step.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters for weight calculation
        run_fingerprint : Dict[str, Any]
            Simulation settings
        prices : jnp.ndarray
            Historical price data
        additional_oracle_input : Optional[jnp.ndarray]
            Extra data for weight calculation

        Returns
        -------
        jnp.ndarray
            Raw weight adjustment values
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement either calculate_rule_outputs() "
            "or calculate_rule_output_step()"
        )

    def calculate_rule_output_step(
        self,
        carry: Dict[str, jnp.ndarray],
        price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> tuple:
        """
        Calculate a single step of weight update (scan-based path).

        This method represents how the strategy would run in production, where we are
        given current state, receive a new price observation, and output new state
        along with the weight update for this timestep.

        This is the core primitive that enables causality-preserving simulation.
        The state (carry) contains all information needed to compute the next step
        without any look-ahead bias.

        Subclasses should implement either this method OR calculate_rule_outputs.

        Parameters
        ----------
        carry : Dict[str, jnp.ndarray]
            Current state containing estimator variables. Typical keys include:
            - 'ewma': Exponentially weighted moving average of prices (shape: n_assets,)
            - 'running_a': Running accumulator for gradient estimation (shape: n_assets,)
            Additional keys may be present depending on the pool implementation.
        price : jnp.ndarray
            Current price observation (shape: n_assets,)
        params : Dict[str, Any]
            Pool parameters (k, lamb, etc.)
        run_fingerprint : Dict[str, Any]
            Simulation settings (chunk_period, max_memory_days, etc.)

        Returns
        -------
        tuple
            (new_carry, rule_output) where:
            - new_carry: Updated state dict with same structure as input carry
            - rule_output: Weight update/output for this timestep (shape: n_assets,)
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement either calculate_rule_outputs() "
            "or calculate_rule_output_step()"
        )

    def get_initial_rule_state(
        self,
        initial_price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> Dict[str, jnp.ndarray]:
        """
        Initialize the carry state for scanning.

        This creates the initial state needed to begin the scan-based
        weight calculation. The initial state is typically derived from
        the first price observation.

        Required if using scan-based path (calculate_rule_output_step).

        Parameters
        ----------
        initial_price : jnp.ndarray
            First price observation (shape: n_assets,)
        params : Dict[str, Any]
            Pool parameters
        run_fingerprint : Dict[str, Any]
            Simulation settings

        Returns
        -------
        Dict[str, jnp.ndarray]
            Initial carry state with keys appropriate for this pool type.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_initial_rule_state() for scan-based calculation"
        )

    def supports_vectorized_path(self) -> bool:
        """
        Check if pool implements vectorized weight calculation.

        Returns True if this pool class overrides calculate_rule_outputs,
        indicating it supports the vectorized (convolution-based) path.

        Returns
        -------
        bool
            True if vectorized path is supported.
        """
        return type(self).calculate_rule_outputs is not TFMMBasePool.calculate_rule_outputs

    def supports_scan_path(self) -> bool:
        """
        Check if pool implements scan-based weight calculation.

        Returns True if this pool class overrides calculate_rule_output_step,
        indicating it supports the scan-based path.

        Returns
        -------
        bool
            True if scan path is supported.
        """
        return type(self).calculate_rule_output_step is not TFMMBasePool.calculate_rule_output_step

    def calculate_rule_outputs_scan(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate raw weight outputs using jax.lax.scan over single-step updates.

        This method produces the same outputs as calculate_rule_outputs,
        but uses an explicit scan loop over the single-step update method.
        This mirrors how the strategy would be executed in production, where
        we process one price at a time.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters
        run_fingerprint : Dict[str, Any]
            Simulation settings
        prices : jnp.ndarray
            Historical price data (shape: time_steps, n_assets)
        additional_oracle_input : Optional[jnp.ndarray]
            Extra data for weight calculation (not used in scan-based approach)

        Returns
        -------
        jnp.ndarray
            Raw weight outputs with same shape and values as calculate_rule_outputs
        """
        chunkwise_price_values = prices[:: run_fingerprint["chunk_period"]]
        n_assets = chunkwise_price_values.shape[1]

        # Initialize carry from first price
        initial_carry = self.get_initial_rule_state(
            chunkwise_price_values[0], params, run_fingerprint
        )

        # Create scan function with params/fingerprint bound
        scan_fn = Partial(
            self.calculate_rule_output_step,
            params=params,
            run_fingerprint=run_fingerprint,
        )

        # Run scan over remaining prices (starting from index 1)
        final_carry, rule_outputs = scan(
            scan_fn, initial_carry, chunkwise_price_values[1:]
        )

        # Note: The scan produces outputs for prices[1:], which gives (n-1) outputs.
        # This matches calc_gradients which returns gradients[1:] (dropping first zero row).
        return rule_outputs

    def get_initial_guardrail_state(
        self,
        initial_weights: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """
        Initialize the weight carry state for scanning with guardrails.

        Parameters
        ----------
        initial_weights : jnp.ndarray
            Initial portfolio weights (shape: n_assets,)

        Returns
        -------
        Dict[str, jnp.ndarray]
            Initial weight carry state with 'prev_actual_weight'
        """
        return {"prev_actual_weight": initial_weights}

    def calculate_coarse_weight_step(
        self,
        estimator_carry: Dict[str, jnp.ndarray],
        weight_carry: Dict[str, jnp.ndarray],
        price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> tuple:
        """
        Compute raw weight update and apply guardrails for a single step.

        This method calls calculate_rule_output_step to get the raw
        weight output, then applies guardrails (normalization, min/max constraints,
        max change limits).

        Parameters
        ----------
        estimator_carry : Dict[str, jnp.ndarray]
            Current estimator state (ewma, running_a, etc.)
        weight_carry : Dict[str, jnp.ndarray]
            Current weight state with 'prev_actual_weight'
        price : jnp.ndarray
            Current price observation (shape: n_assets,)
        params : Dict[str, Any]
            Pool parameters
        run_fingerprint : Dict[str, Any]
            Simulation settings

        Returns
        -------
        tuple
            (new_estimator_carry, new_weight_carry, step_output) where:
            - new_estimator_carry: Updated estimator state
            - new_weight_carry: Updated weight state with 'prev_actual_weight'
            - step_output: Dict with 'actual_start', 'scaled_diff', 'target_weight'
        """
        # Step 1: Get raw weight output from the pool-specific calculation
        new_estimator_carry, rule_output = self.calculate_rule_output_step(
            estimator_carry, price, params, run_fingerprint
        )

        # Step 2: Apply guardrails using the existing low-level function
        n_assets = run_fingerprint["n_assets"]
        minimum_weight = run_fingerprint.get("minimum_weight")
        if minimum_weight is None:
            minimum_weight = 0.1 / n_assets
        maximum_change = run_fingerprint["maximum_change"]
        weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
        interpol_num = weight_interpolation_period + 1
        ste_max_change = run_fingerprint.get("ste_max_change", False)
        ste_min_max_weight = run_fingerprint.get("ste_min_max_weight", False)

        asset_arange = jnp.arange(n_assets)

        carry_list = [weight_carry["prev_actual_weight"]]
        new_carry_list, (actual_start, scaled_diff, target_weight) = _jax_calc_coarse_weight_scan_function(
            carry_list,
            rule_output,
            minimum_weight=minimum_weight,
            asset_arange=asset_arange,
            n_assets=n_assets,
            alt_lamb=None,
            interpol_num=interpol_num,
            maximum_change=maximum_change,
            rule_outputs_are_weights=False,
            ste_max_change=ste_max_change,
            ste_min_max_weight=ste_min_max_weight,
            max_weights_per_asset=None,
            min_weights_per_asset=None,
            use_per_asset_bounds=False,
        )

        new_weight_carry = {"prev_actual_weight": new_carry_list[0]}
        step_output = {
            "actual_start": actual_start,
            "scaled_diff": scaled_diff,
            "target_weight": target_weight,
        }

        return new_estimator_carry, new_weight_carry, step_output

    def calculate_fine_weights_step(
        self,
        estimator_carry: Dict[str, jnp.ndarray],
        weight_carry: Dict[str, jnp.ndarray],
        price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> tuple:
        """
        Compute a single interpolation block of fine weights for one price step.

        This method calls calculate_coarse_weight_step to get the
        guardrailed weight outputs, then generates the full interpolation block.

        Parameters
        ----------
        estimator_carry : Dict[str, jnp.ndarray]
            Current estimator state
        weight_carry : Dict[str, jnp.ndarray]
            Current weight state with 'prev_actual_weight'
        price : jnp.ndarray
            Current price observation
        params : Dict[str, Any]
            Pool parameters
        run_fingerprint : Dict[str, Any]
            Simulation settings

        Returns
        -------
        tuple
            (new_estimator_carry, new_weight_carry, interpolation_block) where:
            - new_estimator_carry: Updated estimator state
            - new_weight_carry: Updated weight state
            - interpolation_block: Array of shape (chunk_period, n_assets)
        """
        # Get guardrailed weight outputs
        new_estimator_carry, new_weight_carry, step_output = self.calculate_coarse_weight_step(
            estimator_carry, weight_carry, price, params, run_fingerprint
        )

        actual_start = step_output["actual_start"]
        scaled_diff = step_output["scaled_diff"]

        n_assets = run_fingerprint["n_assets"]
        weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
        chunk_period = run_fingerprint["chunk_period"]
        weight_interpolation_method = run_fingerprint.get("weight_interpolation_method", "linear")

        interpol_num = weight_interpolation_period + 1
        num = chunk_period + 1

        # Create interpolation arrays
        interpol_arange = jnp.expand_dims(jnp.arange(start=0, stop=interpol_num), 1)
        fine_ones = jnp.ones((num - 1, n_assets))

        # Generate interpolation block
        if weight_interpolation_method == "linear":
            interpolation_block = _jax_calc_linear_interpolation_block(
                actual_start, scaled_diff, interpol_arange, fine_ones, interpol_num
            )
        elif weight_interpolation_method == "approx_optimal":
            interpolation_block = _jax_calc_approx_optimal_interpolation_block(
                actual_start, scaled_diff, interpol_arange, fine_ones, interpol_num
            )
        else:
            raise ValueError(f"Invalid interpolation method: {weight_interpolation_method}")

        return new_estimator_carry, new_weight_carry, interpolation_block

    def calculate_weights_scan(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate fine weights using sequential single-step interpolation blocks.

        This method produces the same outputs as calculate_weights, but uses
        a truly sequential approach:
        1. Warm up the estimator over the burn-in period (single-step updates)
        2. Reset weight state to initial_weights at bout start
        3. Scan over bout prices using calculate_fine_weights_step
        4. Concatenate interpolation blocks

        This mirrors how weights would be computed in a production system
        processing prices one step at a time.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters
        run_fingerprint : Dict[str, Any]
            Simulation settings including chunk_period, bout_length, n_assets
        prices : jnp.ndarray
            Full price history including burn-in period
        start_index : jnp.ndarray
            Start index for the bout period (after burn-in)
        additional_oracle_input : Optional[jnp.ndarray]
            Extra data for weight calculation

        Returns
        -------
        jnp.ndarray
            Fine weights matching calculate_weights output
        """
        chunk_period = run_fingerprint["chunk_period"]
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        # Get initial weights
        initial_weights = self.calculate_initial_weights(params)

        # Chunk prices at chunk_period intervals
        chunkwise_price_values = prices[::chunk_period]

        # Calculate start chunk index (coarse level)
        start_chunk_idx = (start_index[0] / chunk_period).astype("int64")

        # Handle bout_length not divisible by chunk_period
        if bout_length % chunk_period != 0:
            n_bout_chunks = int(bout_length / chunk_period) + 1
        else:
            n_bout_chunks = int(bout_length / chunk_period)

        # Phase 1: Warm up estimator over burn-in period
        # Initialize estimator from first price
        estimator_carry = self.get_initial_rule_state(
            chunkwise_price_values[0], params, run_fingerprint
        )

        # Warm-up using fori_loop: supports traced (dynamic) bounds unlike scan.
        # Process burn-in chunks (index 1 through start_chunk_idx inclusive).
        #
        # PERFORMANCE NOTE: An alternative approach would use a fixed-size scan
        # which XLA can optimize better (unrolling, vectorization). The maximum
        # burn-in size is bounded by (max_memory_days * 1440 + bout_offset) / chunk_period
        # (maybe with some off by one indexing too) because:
        #   - Pre-slicing loads data starting at original_start - max_memory_days
        #   - start_idx can vary within bout_offset range during training
        #   - So max start_chunk_idx = (max_memory_days * 1440 + bout_offset) / chunk_period
        #
        # A fixed-size scan would:
        #   1. Compute max_burn_in_chunks from max_memory_days and bout_offset
        #   2. Always scan over max_burn_in_chunks prices (wasting iterations when
        #      actual burn-in is shorter)
        #   3. Benefit from better XLA optimization of scan vs fori_loop
        #
        # We use fori_loop here for clarity - it runs exactly the needed iterations.
        # If profiling shows this is a bottleneck, consider switching to fixed-size scan.
        def warm_up_body(i, est_carry):
            price = chunkwise_price_values[i]
            new_est_carry, _ = self.calculate_rule_output_step(
                est_carry, price, params, run_fingerprint
            )
            return new_est_carry

        # fori_loop upper bound is exclusive, so use start_chunk_idx + 1
        warmed_estimator_carry = fori_loop(
            1,  # start from index 1 (index 0 used for initialization)
            start_chunk_idx + 1,  # end exclusive (process up to start_chunk_idx)
            warm_up_body,
            estimator_carry,
        )

        # Phase 2: Compute fine weights for bout period
        # Reset weight carry to initial_weights (fresh start for bout)
        weight_carry = self.get_initial_guardrail_state(initial_weights)

        # Bout scan: process bout prices, output interpolation blocks
        def bout_scan_fn(carry, price):
            est_carry, wt_carry = carry
            new_est_carry, new_wt_carry, interpolation_block = self.calculate_fine_weights_step(
                est_carry, wt_carry, price, params, run_fingerprint
            )
            return (new_est_carry, new_wt_carry), interpolation_block

        # Get bout prices (from chunk start_chunk_idx+1 onwards)
        bout_prices = dynamic_slice(
            chunkwise_price_values,
            (start_chunk_idx + 1, 0),
            (n_bout_chunks, n_assets),
        )

        initial_bout_carry = (warmed_estimator_carry, weight_carry)
        _, interpolation_blocks = scan(
            bout_scan_fn, initial_bout_carry, bout_prices
        )

        # Reshape blocks: (n_bout_chunks, chunk_period, n_assets) -> flat
        fine_weights = interpolation_blocks.reshape(-1, n_assets)

        # Prepend initial weights for first chunk (matching calculate_fine_weights)
        fine_weights = jnp.vstack([
            jnp.ones((chunk_period, n_assets), dtype=jnp.float64) * initial_weights,
            fine_weights,
        ])

        # Final slice to exact bout_length - 1
        weights = dynamic_slice(fine_weights, (0, 0), (bout_length - 1, n_assets))

        return weights

    def calculate_weights_hybrid(
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
        Calculate the weights of assets in the pool using scan-based raw weight calculation.

        This method produces the same outputs as calculate_weights, but uses
        calculate_rule_outputs_scan instead of calculate_rule_outputs.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters.
        run_fingerprint : Dict[str, Any]
            Simulation settings.
        prices : jnp.ndarray
            Current prices of the assets.
        start_index : jnp.ndarray
            Start index for slicing
        additional_oracle_input : Optional[jnp.ndarray], optional
            Additional input from an oracle. Defaults to None.

        Returns
        -------
        jnp.ndarray
            Calculated weights for each asset in the pool.
        """
        chunk_period = run_fingerprint["chunk_period"]
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        rule_outputs = self.calculate_rule_outputs_scan(
            params, run_fingerprint, prices, additional_oracle_input
        )
        # we don't want to change the initial weights during any training
        # so wrap them in a stop_grad
        initial_weights = self.calculate_initial_weights(params)

        # we have a sequence now of weight changes, but if we are doing
        # a burnin operation, we need to cut off the changes associated
        # with the burnin period, ie everything before the start of the sequence

        start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)

        # if the chunk period is not a divisor of bout_length, we need to pad the rule_outputs.
        # this can require more data to be available, potentially beyond the end of the bout.
        if bout_length % chunk_period != 0:
            raw_weight_additional_offset = 1
        else:
            raw_weight_additional_offset = 0
        rule_outputs = dynamic_slice(
            rule_outputs,
            start_index_coarse,
            (
                int((bout_length) / chunk_period) + raw_weight_additional_offset,
                n_assets,
            ),
        )
        weights = self.calculate_fine_weights(
            rule_outputs,
            initial_weights,
            run_fingerprint,
            params,
        )
        weights = dynamic_slice(
            weights, (0, 0), (bout_length - 1, n_assets)
        )
        return weights

    def calculate_readouts(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate readouts (internal estimator variables, other running variables) for the pool,
        based on price history.

        This method can potentially have some overlap with calculate_rule_outputs, but
        for most TFMM pools it will simply correspond to the readout values for the
        gradient estimator (the ewma of prices and running a), sliced in the same way that
        the raw weight outputs are sliced.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters for weight calculation
        run_fingerprint : Dict[str, Any]
            Simulation settings
        prices : jnp.ndarray
            Historical price data
        start_index : jnp.ndarray
            Start index for slicing
        additional_oracle_input : Optional[jnp.ndarray]
            Extra data for weight calculation

        Returns
        -------
        dict
            Dict containing readout values for the pool
        """
        pass

    @abstractmethod
    def calculate_fine_weights(
        self,
        rule_output: jnp.ndarray,
        initial_weights: jnp.ndarray,
        run_fingerprint: Dict[str, Any],
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Refine raw weight outputs into final weights.

        Second step of TFMM's weight calculation process. Converts raw weight
        adjustments into valid pool weights.

        Parameters
        ----------
        rule_output : jnp.ndarray
            Output from calculate_rule_outputs
        initial_weights : jnp.ndarray
            Starting weights
        run_fingerprint : Dict[str, Any]
            Simulation settings
        params : Dict[str, Any]
            Pool parameters

        Returns
        -------
        jnp.ndarray
            Final refined weights
        """
        pass

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

        Routes to either vectorized or scan-based weight calculation
        based on the `weight_calculation_method` in run_fingerprint:
        - "auto" (default): Use vectorized if available, else scan
        - "vectorized": Force vectorized path (errors if not supported)
        - "scan": Force scan path (errors if not supported)

        Parameters
        ----------
            params (Dict[str, Any]): Pool parameters.
            run_fingerprint (Dict[str, Any]): Simulation settings.
            prices (jnp.ndarray): Current prices of the assets.
            start_index (jnp.ndarray): Start index for slicing
            additional_oracle_input (Optional[jnp.ndarray], optional): Additional input from an oracle. Defaults to None.

        Returns
        -------
            jnp.ndarray: Calculated weights for each asset in the pool.
        """
        method = run_fingerprint.get("weight_calculation_method", "auto")

        if method == "scan":
            if not self.supports_scan_path():
                raise NotImplementedError(
                    f"{type(self).__name__} does not support scan-based weight calculation"
                )
            return self.calculate_weights_scan(
                params, run_fingerprint, prices, start_index, additional_oracle_input
            )

        if method == "vectorized":
            if not self.supports_vectorized_path():
                raise NotImplementedError(
                    f"{type(self).__name__} does not support vectorized weight calculation"
                )
            return self.calculate_weights_vectorized(
                params, run_fingerprint, prices, start_index, additional_oracle_input
            )

        if method == "auto":
            if self.supports_vectorized_path():
                return self.calculate_weights_vectorized(
                    params, run_fingerprint, prices, start_index, additional_oracle_input
                )
            if self.supports_scan_path():
                return self.calculate_weights_scan(
                    params, run_fingerprint, prices, start_index, additional_oracle_input
                )
            raise NotImplementedError(
                f"{type(self).__name__} must implement either calculate_rule_outputs() "
                "or calculate_rule_output_step()"
            )

        raise ValueError(f"Unknown weight_calculation_method: {method}")

    @partial(jit, static_argnums=(2, 5))
    def calculate_weights_vectorized(
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
        Calculate weights using the vectorized path (calculate_rule_outputs).

        Parameters
        ----------
            params (Dict[str, Any]): Pool parameters.
            run_fingerprint (Dict[str, Any]): Simulation settings.
            prices (jnp.ndarray): Current prices of the assets.
            start_index (jnp.ndarray): Start index for slicing
            additional_oracle_input (Optional[jnp.ndarray], optional): Additional input from an oracle. Defaults to None.

        Returns
        -------
            jnp.ndarray: Calculated weights for each asset in the pool.
        """
        chunk_period = run_fingerprint["chunk_period"]
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        rule_outputs = self.calculate_rule_outputs(
            params, run_fingerprint, prices, additional_oracle_input
        )
        # we don't want to change the initial weights during any training
        # so wrap them in a stop_grad
        initial_weights = self.calculate_initial_weights(params)

        # we have a sequence now of weight changes, but if we are doing
        # a burnin operation, we need to cut off the changes associated
        # with the burnin period, ie everything before the start of the sequence

        start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)

        # if the chunk period is not a divisor of bout_length, we need to pad the rule_outputs.
        # this can require more data to be available, potentially beyond the end of the bout.
        if bout_length % chunk_period != 0:
            raw_weight_additional_offset = 1
        else:
            raw_weight_additional_offset = 0
        rule_outputs = dynamic_slice(
            rule_outputs,
            start_index_coarse,
            (
                int((bout_length) / chunk_period) + raw_weight_additional_offset,
                n_assets,
            ),
        )
        weights = self.calculate_fine_weights(
            rule_outputs,
            initial_weights,
            run_fingerprint,
            params,
        )
        weights = dynamic_slice(
            weights, (0, 0), (bout_length - 1, n_assets)
        )
        return weights

    @partial(jit, static_argnums=(2, 3, 5))
    def calculate_final_weights(
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

        Parameters
        ----------
            params (Dict[str, Any]): Pool parameters.
            run_fingerprint (Dict[str, Any]): Simulation settings.
            prices (jnp.ndarray): Current prices of the assets.
            start_index (jnp.ndarray): Start index for slicing
            additional_oracle_input (Optional[jnp.ndarray], optional): Additional input from an oracle. Defaults to None.

        Returns
        -------
            jnp.ndarray: Calculated weights for each asset in the pool.
        """
        chunk_period = run_fingerprint["chunk_period"]
        bout_length = len(prices) - start_index[0]
        n_assets = run_fingerprint["n_assets"]
        rule_outputs = self.calculate_rule_outputs(
            params, run_fingerprint, prices, additional_oracle_input
        )
        # we don't want to change the initial weights during any training
        # so wrap them in a stop_grad
        initial_weights = self.calculate_initial_weights(params)

        # we have a sequence now of weight changes, but if we are doing
        # a burnin operation, we need to cut off the changes associated
        # with the burnin period, ie everything before the start of the sequence

        start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)

        # if the chunk period is not a divisor of bout_length, we need to pad the rule_outputs.
        # this can require more data to be available, potentially beyond the end of the bout.
        raw_weight_additional_offset = jnp.where(bout_length % chunk_period != 0, 1, 0).astype("int64")
        from jax.lax import slice as jax_slice
        alt_slice = jax_slice(rule_outputs, start_index_coarse, int((len(prices)/chunk_period), n_assets))

        rule_outputs = dynamic_slice(
            rule_outputs,
            start_index_coarse,
            (
                int((bout_length) / chunk_period) + raw_weight_additional_offset,
                n_assets,
            ),
        )
        weights = self.calculate_fine_weights(
            rule_outputs,
            initial_weights,
            run_fingerprint,
            params,
        )
        raise Exception("Not implemented")
        return weights

    def calculate_all_signature_variations(self, params: Dict[str, Any]) -> jnp.ndarray:
        """
        Calculate all valid trading signature combinations.

        Abstract method that subclasses may implement to define valid trading patterns.
        Can be used by reserve calculation methods to determine possible arbitrage opportunities.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters that may influence valid trade combinations

        Returns
        -------
        jnp.ndarray
            Array of valid trading signature combinations

        Raises
        ------
        NotImplementedError
            Base class does not implement this method
        """
        raise NotImplementedError

    def make_vmap_in_axes(self, params: Dict[str, Any], n_repeats_of_recurred: int = 0):
        """
        Configure JAX vectorization axes for pool parameters.

        FMM pools handle subsidiary parameters differently
        for vectorization due to their potentially more complex parameter structure.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters to vectorize
        n_repeats_of_recurred : int, optional
            Number of times to repeat recurrent parameters, by default 0

        Returns
        -------
        Dict[str, Any]
            vmap axes configuration with subsidiary_params handled separately
        """
        return make_vmap_in_axes_dict(
            params, 0, [], ["subsidary_params"], n_repeats_of_recurred
        )

    def is_trainable(self):
        """
        Indicate if pool weights can be trained.

        TFMM pools are trainable by default, as their weights
        change based on market conditions.

        Returns
        -------
        bool
            Always True for TFMM pools as weights are trainable
        """
        return True

    @classmethod
    def process_parameters(cls, update_rule_parameters, run_fingerprint):
        """
        Process TFMM pool parameters from web interface input.

        Handles common TFMM parameters and delegates pool-specific processing
        to subclasses. Supports both per-token and global parameters.

        Parameters
        ----------
        update_rule_parameters : Dict[str, Any]
            Raw parameters from web interface, each containing:
            - name: Parameter identifier
            - value: Parameter values per token
        run_fingerprint : Dict[str, Any]
            Run fingerprint dictionary

        Returns
        -------
        Dict[str, np.ndarray]
            Processed parameters including:
            - logit_lamb: Memory parameter
            - k: Update rate parameter
            - Additional pool-specific parameters

        Notes
        -----
        - Handles parameter broadcasting for single values
        - Validates parameter dimensions
        - Processes memory_days and k_per_day specially
        - Allows subclasses to add specific parameters
        """
        result = {}
        processed_params = set()
        n_assets = len(run_fingerprint["tokens"])
        # Process TFMM common parameters
        memory_days_values = cls._process_memory_days(update_rule_parameters, n_assets, run_fingerprint["chunk_period"])
        if memory_days_values is not None:
            result.update(memory_days_values)
            processed_params.add("memory_days")

        k_values = cls._process_k_per_day(update_rule_parameters, n_assets)
        if k_values is not None:
            result.update(k_values)
            processed_params.add("k_per_day")

        # Let specific pools process their parameters
        specific_params = cls._process_specific_parameters(
            update_rule_parameters, run_fingerprint
        )
        if specific_params is not None:
            result.update(specific_params)
            # Assume any parameters returned by specific processing are handled
            processed_params.update(specific_params.keys())

        # Process any remaining parameters in a default way
        for urp in update_rule_parameters:
            if urp.name not in processed_params:
                value = []
                for tokenValue in urp.value:
                    value.append(tokenValue)
                if len(value) != n_assets:
                    value = [value[0]] * n_assets
                result[urp.name] = np.array(value)

        return result

    @classmethod
    def _process_memory_days(cls, update_rule_parameters, n_assets, chunk_period):
        """
        Process memory_days parameter into logit_lamb values.

        Converts memory_days into a logit-transformed lambda parameter
        that determines how quickly the pool forgets past price information.

        Parameters
        ----------
        update_rule_parameters : List[Parameter]
            Raw parameters containing memory_days values
        n_assets : int
            Number of tokens in pool

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'logit_lamb' key containing transformed values,
            or None if memory_days not found

        Notes
        -----
        - Converts memory days to lambda using memory_days_to_lamb
        - Applies logit transform for numerical stability
        - Broadcasts single values to match n_assets
        """
        for urp in update_rule_parameters:
            if urp.name == "memory_days":
                logit_lamb_vals = []
                memory_days_values = urp.value
                for tokenValue in urp.value:
                    initial_lamb = memory_days_to_lamb(tokenValue, chunk_period)
                    logit_lamb = np.log(initial_lamb / (1.0 - initial_lamb))
                    logit_lamb_vals.append(logit_lamb)
                if len(logit_lamb_vals) != n_assets:
                    logit_lamb_vals = [logit_lamb_vals[0]] * n_assets
                return {"logit_lamb": np.array(logit_lamb_vals)}
        return None

    @classmethod
    def _process_k_per_day(cls, update_rule_parameters, n_assets):
        """
        Process k_per_day parameter into update rate values.

        The k parameter determines how quickly weights adjust to new prices.
        Higher values mean faster adjustments.

        Parameters
        ----------
        update_rule_parameters : List[Parameter]
            Raw parameters containing k_per_day values
        n_assets : int
            Number of tokens in pool

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'k' key containing update rates,
            or None if k_per_day not found

        Notes
        -----
        - Uses raw k values without transformation
        - Broadcasts single values to match n_assets
        """
        for urp in update_rule_parameters:
            if urp.name == "k_per_day":
                k_vals = []
                for tokenValue in urp.value:
                    k_vals.append(tokenValue)
                if len(k_vals) != n_assets:
                    k_vals = [k_vals[0]] * n_assets
                return {"k": np.array(k_vals)}
        return None

    @classmethod
    def _process_specific_parameters(cls, update_rule_parameters, run_fingerprint):
        """
        Process pool-specific parameters.

        Abstract method that subclasses should override to handle any
        parameters specific to their implementation.

        Parameters
        ----------
        update_rule_parameters : Dict[str, Any]
            Raw parameters to process
        run_fingerprint : Dict[str, Any]
            Run fingerprint dictionary

        Returns
        -------
        Dict[str, np.ndarray] or None
            Processed pool-specific parameters if any,
            None if no specific parameters needed
        """
        return None

    @partial(jit, static_argnums=(2, 5))
    def calculate_weights_direct(
        self,
        params: Dict[str, Any],
        prices: jnp.ndarray,
        maximum_change: float = 3e-4,
        minimum_weight: float = 0.03,
        initial_weights: Optional[jnp.ndarray] = None,
        initial_running_a: Optional[jnp.ndarray] = None,
        initial_ewma: Optional[jnp.ndarray] = None,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Calculate the weights of assets in the pool, directly from the prices.
        This is used to quickly calculate the weights from any given price array, without
        doing any chunking or fine-weighting.


        Parameters
        ----------
            params (Dict[str, Any]): Pool parameters.
            prices (jnp.ndarray): Current prices of the assets
            initial_weights (jnp.ndarray, optional): Initial weights of the assets
            initial_running_a (jnp.ndarray, optional): Initial running_a value of the gradient estimator
            initial_ewma (jnp.ndarray, optional): Initial ewma value of the gradient estimator

        Returns
        -------
            jnp.ndarray: Calculated weights for each asset in the pool.
        """
        n_assets = prices.shape[1]
        local_fingerprint = {
            "chunk_period": 1,
            "weight_interpolation_period": 1,
            "max_memory_days": 365.0,
            "use_alt_lamb": False,
        }

        rule_outputs = self.calculate_rule_outputs(
            params, local_fingerprint, prices, None
        )
        # we dont't want to change the initial weights during any training
        # so wrap them in a stop_grad
        if initial_weights is None:
            initial_weights = self.calculate_initial_weights(params)

        actual_starts_cpu, scaled_diffs_cpu, target_weights_cpu = _jax_calc_coarse_weights(
            rule_outputs,
            initial_weights,
            minimum_weight,
            params,
            local_fingerprint["max_memory_days"],
            local_fingerprint["chunk_period"],
            local_fingerprint["weight_interpolation_period"],
            maximum_change,
            False,
        )

        return weights
