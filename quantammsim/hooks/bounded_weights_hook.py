"""Bounded weights hook for per-asset min/max weight constraints.

This module provides a mixin class that adds per-asset minimum and maximum
weight constraints to pool strategies. The bounds are applied as a pre-processing
step before the standard uniform guardrails in the weight calculation pipeline.

The actual bounded weight logic is implemented in fine_weights.py via the
_apply_per_asset_bounds function, which is called when use_per_asset_bounds=True.
"""
from typing import Dict, Any, Tuple
from functools import partial
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
from jax import jit
from jax.lax import stop_gradient

from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output_bounded_from_weight_changes,
    _jax_calc_coarse_weight_scan_function,
)


# =============================================================================
# Learnable Bounds Reparameterization
# =============================================================================
# These functions allow min/max weight bounds to be learned via gradient descent
# while guaranteeing all feasibility constraints are satisfied by construction.
#
# Constraints:
#   1. sum(min_i) <= 1
#   2. sum(max_i) >= 1
#   3. min_i < max_i for all i
#   4. min_i >= 0 for all i
#   5. max_i <= 1 for all i
#
# The reparameterization uses:
#   - Sigmoid for budget allocation (keeps sum bounded)
#   - Softmax for distribution across assets
#   - Gap-based max construction (ensures max > min and max <= 1)
#   - Scale-up-only to ensure sum(max) >= 1
# =============================================================================


def reparameterize_bounds(
    raw_min_budget: jnp.ndarray,
    raw_min_logits: jnp.ndarray,
    raw_gap_logits: jnp.ndarray,
    n_assets: int,
    eps: float = 1e-6,
    freeze_bounds: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transform unconstrained raw parameters into valid min/max weight bounds.

    This reparameterization guarantees all feasibility constraints are satisfied
    by construction, making it safe for gradient-based optimization.

    Algorithm
    ---------
    1. **Min weights** (sum < 1, each >= 0):
       - min_budget = sigmoid(raw_min_budget) controls total allocation to mins
       - min_weights = min_budget * softmax(raw_min_logits) distributes budget

    2. **Gaps** (ensures max > min and initial_max <= 1):
       - available = 1 - min_weights (room for gap per asset)
       - gap = available * sigmoid(raw_gap_logits) per asset
       - initial_max = min_weights + gap (each in (min_i, 1])

    3. **Scale up if needed** (ensures sum(max) >= 1):
       - scale = max(1 / sum(initial_max), 1)
       - max_weights = initial_max * scale

    Why max_i <= 1 is guaranteed:
       - Each initial_max_i <= 1 by construction (gap <= available = 1 - min)
       - scale = max(1/sum, 1), and sum >= max(initial_max) always
       - So scale <= 1/max(initial_max), meaning largest scaled value <= 1

    Numerical Note:
       Due to the eps term for stability, sum(max) may be slightly below 1
       (e.g., 0.9999985 instead of 1.0). This is within floating point
       tolerance and acceptable for practical use.

    Gradient Properties
    -------------------
    - Sigmoids: smooth, but can saturate at extremes (init near 0)
    - Softmax: smooth, gradients flow to all assets
    - max(1/sum, 1): piecewise linear (like ReLU), generally fine
    - No jnp.max over assets, so gradients flow uniformly

    Args:
        raw_min_budget: Scalar or shape (n_parameter_sets,), unconstrained.
            Controls total allocation to minimums. sigmoid(0) = 0.5.
        raw_min_logits: Shape (n_parameter_sets, n_assets), unconstrained.
            Controls distribution of min budget across assets.
        raw_gap_logits: Shape (n_parameter_sets, n_assets), unconstrained.
            Controls gap between min and max per asset. sigmoid(0) = 0.5.
        n_assets: Number of assets.
        eps: Small constant for numerical stability.
        freeze_bounds: If True, wrap inputs in stop_gradient so bounds act
            as hyperparameters (not learned). Default False (bounds are learned).

    Returns:
        Tuple of (min_weights_per_asset, max_weights_per_asset), each with
        shape (n_parameter_sets, n_assets).

    Example
    -------
    >>> # Initialise for ~uniform bounds
    >>> raw_min_budget = jnp.array([-1.0])  # sigmoid(-1) ≈ 0.27, so sum(min) ≈ 0.27
    >>> raw_min_logits = jnp.zeros((1, 4))  # uniform distribution
    >>> raw_gap_logits = jnp.zeros((1, 4))  # sigmoid(0) = 0.5 of available
    >>> min_w, max_w = reparameterize_bounds(
    ...     raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
    ... )
    >>> # min_w ≈ [0.068, 0.068, 0.068, 0.068], sum ≈ 0.27
    >>> # max_w: each initial_max ≈ 0.068 + 0.5*(1-0.068) ≈ 0.534
    >>> #        sum(initial_max) ≈ 2.14 > 1, so scale = 1, max_w ≈ [0.534, ...]

    >>> # Freeze bounds as hyperparameters (no gradients)
    >>> min_w, max_w = reparameterize_bounds(
    ...     raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4,
    ...     freeze_bounds=True
    ... )
    """
    # If freeze_bounds is True, wrap inputs in stop_gradient so no gradients
    # flow through the bound parameters (treat them as hyperparameters)
    if freeze_bounds:
        raw_min_budget = stop_gradient(raw_min_budget)
        raw_min_logits = stop_gradient(raw_min_logits)
        raw_gap_logits = stop_gradient(raw_gap_logits)

    # This function works with simple (non-batched) params:
    # - raw_min_budget: scalar
    # - raw_min_logits: (n_assets,)
    # - raw_gap_logits: (n_assets,)
    # Batching over n_parameter_sets is handled externally by vmap.

    # Step 1: Minimum weights
    # min_budget in (0, 1), controls sum of minimums
    min_budget = jnn.sigmoid(raw_min_budget)  # scalar
    # Distribute budget across assets using softmax
    min_fractions = jnn.softmax(raw_min_logits, axis=-1)  # (n_assets,)
    min_weights = min_budget * min_fractions  # (n_assets,)
    # Now: sum(min_weights) = min_budget < 1, each min_weight >= 0

    # Step 2: Gaps between min and max
    # available_i = 1 - min_i (maximum possible gap per asset)
    available = 1.0 - min_weights  # (n_assets,)
    # gap_i in (0, available_i), so initial_max_i in (min_i, 1]
    gap = available * jnn.sigmoid(raw_gap_logits)  # (n_assets,)

    # Step 3: Initial max weights
    initial_max = min_weights + gap  # (n_assets,)
    # Each initial_max_i in (min_i, 1] by construction

    # Step 4: Scale up if sum < 1, otherwise leave alone
    sum_initial = jnp.sum(initial_max)  # scalar
    scale = jnp.maximum(1.0 / (sum_initial + eps), 1.0)  # scalar

    # Final max weights
    max_weights = initial_max * scale  # (n_assets,)
    # sum(max) = sum(initial_max) * scale >= 1
    # Each max_i <= 1 because scale <= 1/max(initial_max) (since sum >= max)

    return min_weights, max_weights


def init_learnable_bounds_params(
    n_assets: int,
    n_parameter_sets: int = 1,
    target_min_sum: float = 0.25,
    target_gap_fraction: float = 0.5,
) -> Dict[str, jnp.ndarray]:
    """
    Initialize raw parameters for learnable bounds with sensible defaults.

    The defaults create roughly uniform bounds across assets with:
    - sum(min) ≈ target_min_sum (default 0.25)
    - gap ≈ target_gap_fraction * available space (default 0.5)

    Args:
        n_assets: Number of assets.
        n_parameter_sets: Number of parallel parameter sets.
        target_min_sum: Target sum of minimum weights.
        target_gap_fraction: Target fraction of available space for gaps.

    Returns:
        Dict with raw parameter arrays ready for optimization.
    """
    # sigmoid(x) = target_min_sum => x = logit(target_min_sum)
    # logit(p) = log(p / (1-p))
    raw_min_budget_val = np.log(target_min_sum / (1 - target_min_sum + 1e-8))
    raw_min_budget = jnp.full((n_parameter_sets,), raw_min_budget_val)

    # Uniform distribution across assets
    raw_min_logits = jnp.zeros((n_parameter_sets, n_assets))

    # Gaps: sigmoid(x) = target_gap_fraction => x = logit(target_gap_fraction)
    raw_gap_val = np.log(target_gap_fraction / (1 - target_gap_fraction + 1e-8))
    raw_gap_logits = jnp.full((n_parameter_sets, n_assets), raw_gap_val)

    return {
        "raw_min_budget": raw_min_budget,
        "raw_min_logits": raw_min_logits,
        "raw_gap_logits": raw_gap_logits,
    }


class BoundedWeightsHook:
    """
    Mixin class to add per-asset weight bounds to pool strategies.

    This hook provides parameter initialisation for per-asset minimum and
    maximum weight constraints. When these parameters are present in the
    params dict, the weight calculation pipeline (in fine_weights.py)
    will apply them as a pre-processing step before the uniform guardrails.

    The bounds are applied BEFORE the standard minimum_weight/maximum_weight
    guardrails, so both constraints must be satisfied.

    Algorithm
    ---------
    The approach uses a clip-and-redistribute method:

    **Step 1: Initial Clip**
    ::

        w'_i = clip(w_i, min_i, max_i)

    **Step 2: Calculate Slack**
    ::

        slack_up_i   = max_i - w'_i    (room to grow)
        slack_down_i = w'_i - min_i    (room to shrink)

    **Step 3: Redistribute Proportionally**

    If sum(w'_i) < 1 (deficit)::

        adjustment_i = (1 - sum(w'_i)) * slack_up_i / sum(slack_up_j)

    If sum(w'_i) > 1 (surplus)::

        adjustment_i = -(sum(w'_i) - 1) * slack_down_i / sum(slack_down_j)

    **Step 4: Final Weights**
    ::

        w''_i = clip(w'_i + adjustment_i, min_i, max_i)
        w_final_i = w''_i / sum(w''_j)

    Restrictions on Bounds for N Assets
    ------------------------------------
    For a feasible solution to exist, the bounds must satisfy:

    +------------------------+------------------+--------------------------------------+
    | Constraint             | Formula          | Meaning                              |
    +========================+==================+======================================+
    | Sum of minimums        | sum(min_i) <= 1  | Must be possible to satisfy all mins |
    +------------------------+------------------+--------------------------------------+
    | Sum of maximums        | sum(max_i) >= 1  | Must be possible to reach total of 1 |
    +------------------------+------------------+--------------------------------------+
    | Per-asset ordering     | min_i < max_i    | Each asset must have a valid range   |
    +------------------------+------------------+--------------------------------------+
    | Non-negative           | min_i >= 0       | Weights cannot be negative           |
    +------------------------+------------------+--------------------------------------+
    | Upper bound            | max_i <= 1       | No single asset can exceed 100%      |
    +------------------------+------------------+--------------------------------------+

    Key Properties
    --------------
    1. **Guaranteed feasibility**: If the bounds satisfy the constraints above,
       a valid weight vector always exists.

    2. **Proportional redistribution**: Slack is redistributed proportionally,
       so assets with more room to adjust absorb more of the deficit/surplus.

    3. **Preserves relative ordering**: Assets closer to their bounds move less
       than those with more slack.

    4. **Layered on existing guardrails**: This runs BEFORE the uniform
       `minimum_weight` guardrail, so both constraints must be satisfied.

    Example: Why Sum Constraints Matter
    -----------------------------------
    For 3 assets with bounds:

    - Valid: min = [0.2, 0.2, 0.2], max = [0.5, 0.5, 0.5]
      sum(min)=0.6 <= 1, sum(max)=1.5 >= 1  (feasible)

    - Invalid: min = [0.4, 0.4, 0.4]
      sum(min)=1.2 > 1, impossible to satisfy all minimums

    - Invalid: max = [0.3, 0.3, 0.3]
      sum(max)=0.9 < 1, impossible to reach total weight of 1

    Usage
    -----
    Create a bounded pool via the pool creator with the ``bounded__`` prefix::

        from quantammsim.pools.creator import create_pool

        pool = create_pool("bounded__momentum")
        pool = create_pool("bounded__mean_reversion_channel")

    Or mix this hook with a base pool directly::

        class BoundedMomentumPool(BoundedWeightsHook, MomentumPool):
            def init_base_parameters(self, initial_values_dict, run_fingerprint,
                                     n_assets, n_parameter_sets=1, noise="gaussian"):
                base_params = super().init_base_parameters(
                    initial_values_dict, run_fingerprint, n_assets,
                    n_parameter_sets, noise
                )
                return self.extend_parameters(
                    base_params, initial_values_dict, run_fingerprint,
                    n_assets, n_parameter_sets
                )

    Parameters (in params dict)
    ---------------------------
    min_weights_per_asset : np.ndarray
        Minimum weight for each asset, shape (n_parameter_sets, n_assets).
        Must sum to <= 1 for each set.
    max_weights_per_asset : np.ndarray
        Maximum weight for each asset, shape (n_parameter_sets, n_assets).
        Must sum to >= 1 for each set.

    Notes
    -----
    - The per-asset bounds are applied before run_fingerprint["minimum_weight"]
    - All standard guardrails (maximum_change, weight interpolation, etc.) are preserved
    - The mixin must appear before the base pool class in the inheritance order
    """

    @staticmethod
    def validate_bounds(
        min_weights: np.ndarray,
        max_weights: np.ndarray,
        n_assets: int = None,
    ) -> None:
        """
        Validate that per-asset bounds satisfy feasibility requirements.

        For a valid weight vector to exist, the bounds must satisfy:
        1. sum(min_i) <= 1  (must be possible to satisfy all minimums)
        2. sum(max_i) >= 1  (must be possible to reach total weight of 1)
        3. min_i < max_i    (each asset must have a valid range)
        4. min_i >= 0       (weights cannot be negative)
        5. max_i <= 1       (no single asset can exceed 100%)

        Args:
            min_weights: Minimum weight for each asset, shape (n_assets,).
            max_weights: Maximum weight for each asset, shape (n_assets,).
            n_assets: Expected number of assets (optional, for length validation).

        Raises:
            ValueError: If any constraint is violated, with detailed message.
        """
        min_w = np.asarray(min_weights)
        max_w = np.asarray(max_weights)

        errors = []

        # Check array lengths match - must match to proceed with other checks
        if min_w.shape != max_w.shape:
            raise ValueError(
                f"Invalid per-asset weight bounds:\n  - "
                f"min_weights shape {min_w.shape} != max_weights shape {max_w.shape}"
            )

        if n_assets is not None:
            if len(min_w) != n_assets:
                errors.append(
                    f"min_weights length {len(min_w)} != n_assets {n_assets}"
                )
            if len(max_w) != n_assets:
                errors.append(
                    f"max_weights length {len(max_w)} != n_assets {n_assets}"
                )

        # Check non-negative minimums
        negative_mins = np.where(min_w < 0)[0]
        if len(negative_mins) > 0:
            errors.append(
                f"min_weights must be non-negative. "
                f"Violations at indices {negative_mins.tolist()}: "
                f"values {min_w[negative_mins].tolist()}"
            )

        # Check maximums don't exceed 1
        over_one_maxs = np.where(max_w > 1)[0]
        if len(over_one_maxs) > 0:
            errors.append(
                f"max_weights must not exceed 1.0. "
                f"Violations at indices {over_one_maxs.tolist()}: "
                f"values {max_w[over_one_maxs].tolist()}"
            )

        # Check min < max for each asset
        invalid_ranges = np.where(min_w >= max_w)[0]
        if len(invalid_ranges) > 0:
            errors.append(
                f"min_weights must be strictly less than max_weights. "
                f"Violations at indices {invalid_ranges.tolist()}: "
                f"min={min_w[invalid_ranges].tolist()}, "
                f"max={max_w[invalid_ranges].tolist()}"
            )

        # Check sum of minimums <= 1
        sum_min = np.sum(min_w)
        if sum_min > 1.0:
            errors.append(
                f"Sum of min_weights ({sum_min:.4f}) exceeds 1.0. "
                f"Impossible to satisfy all minimum constraints simultaneously. "
                f"min_weights={min_w.tolist()}"
            )

        # Check sum of maximums >= 1
        sum_max = np.sum(max_w)
        if sum_max < 1.0:
            errors.append(
                f"Sum of max_weights ({sum_max:.4f}) is less than 1.0. "
                f"Impossible to reach total weight of 1. "
                f"max_weights={max_w.tolist()}"
            )

        if errors:
            raise ValueError(
                "Invalid per-asset weight bounds:\n  - " + "\n  - ".join(errors)
            )

    @partial(jit, static_argnums=(3,))
    def calculate_fine_weights(
        self,
        rule_output: jnp.ndarray,
        initial_weights: jnp.ndarray,
        run_fingerprint: Dict[str, Any],
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Calculate fine weights with per-asset bounds (vectorized path).

        This method overrides the base pool's calculate_fine_weights to use
        per-asset min/max constraints before the uniform bounds.

        Args:
            rule_output: Raw weight changes from the strategy.
            initial_weights: Initial weights.
            run_fingerprint: Run configuration.
            params: Parameters containing either:
                - Learnable: raw_min_budget, raw_min_logits, raw_gap_logits
                - Direct: min_weights_per_asset, max_weights_per_asset

        Returns:
            Fine weights satisfying per-asset bounds and uniform guardrails.
        """
        # Convert learnable raw params to actual bounds if present
        # This check happens at trace time (dict structure is static)
        if "raw_min_budget" in params:
            n_assets = run_fingerprint["n_assets"]
            freeze_bounds = run_fingerprint.get("learnable_bounds_settings", {}).get(
                "freeze_bounds", False
            )
            min_weights_per_asset, max_weights_per_asset = reparameterize_bounds(
                params["raw_min_budget"],
                params["raw_min_logits"],
                params["raw_gap_logits"],
                n_assets,
                freeze_bounds=freeze_bounds,
            )
            # Shallow copy with actual bounds added
            params = {
                **params,
                "min_weights_per_asset": min_weights_per_asset,
                "max_weights_per_asset": max_weights_per_asset,
            }

        return calc_fine_weight_output_bounded_from_weight_changes(
            rule_output, initial_weights, run_fingerprint, params
        )

    def calculate_coarse_weight_step(
        self,
        estimator_carry: Dict[str, jnp.ndarray],
        weight_carry: Dict[str, jnp.ndarray],
        price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> tuple:
        """
        Compute raw weight update and apply guardrails with per-asset bounds (scan path).

        This method overrides the base pool's calculate_coarse_weight_step to use
        per-asset min/max constraints. The per-asset bounds are applied before
        the standard uniform guardrails.

        Parameters
        ----------
        estimator_carry : Dict[str, jnp.ndarray]
            Current estimator state (ewma, running_a, etc.)
        weight_carry : Dict[str, jnp.ndarray]
            Current weight state with 'prev_actual_weight'
        price : jnp.ndarray
            Current price observation (shape: n_assets,)
        params : Dict[str, Any]
            Pool parameters including per-asset bounds
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

        # Step 2: Apply guardrails with per-asset bounds
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

        # Extract per-asset bounds from params
        # Check for learnable raw params first, then fall back to direct bounds
        raw_min_budget = params.get("raw_min_budget")
        if raw_min_budget is not None:
            # Convert raw learnable params to actual bounds
            freeze_bounds = run_fingerprint.get("learnable_bounds_settings", {}).get(
                "freeze_bounds", False
            )
            min_weights_per_asset, max_weights_per_asset = reparameterize_bounds(
                params["raw_min_budget"],
                params["raw_min_logits"],
                params["raw_gap_logits"],
                n_assets,
                freeze_bounds=freeze_bounds,
            )
        else:
            # Use direct bounds from params (e.g., for forward pass with fixed bounds)
            min_weights_per_asset = params.get("min_weights_per_asset")
            max_weights_per_asset = params.get("max_weights_per_asset")
            # For scan path, handle the parameter set dimension
            if min_weights_per_asset is not None and min_weights_per_asset.ndim > 1:
                min_weights_per_asset = min_weights_per_asset[0]
            if max_weights_per_asset is not None and max_weights_per_asset.ndim > 1:
                max_weights_per_asset = max_weights_per_asset[0]

        use_per_asset_bounds = (
            min_weights_per_asset is not None and max_weights_per_asset is not None
        )

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
            max_weights_per_asset=max_weights_per_asset,
            min_weights_per_asset=min_weights_per_asset,
            use_per_asset_bounds=use_per_asset_bounds,
        )

        new_weight_carry = {"prev_actual_weight": new_carry_list[0]}
        step_output = {
            "actual_start": actual_start,
            "scaled_diff": scaled_diff,
            "target_weight": target_weight,
        }

        return new_estimator_carry, new_weight_carry, step_output

    def extend_parameters(
        self,
        base_params: Dict[str, Any],
        initial_values_dict: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
    ) -> Dict[str, Any]:
        """
        Extend base pool parameters with learnable bounded weight parameters.

        The bounds are stored as raw learnable parameters (raw_min_budget,
        raw_min_logits, raw_gap_logits) that are converted to actual bounds
        during the forward pass via reparameterize_bounds.

        Args:
            base_params: Base parameters from the underlying pool.
            initial_values_dict: Initial values for all parameters. Must contain:
                - min_weights_per_asset: Target minimum weights (scalar or array)
                - max_weights_per_asset: Target maximum weights (scalar or array)
            n_assets: Number of assets.
            n_parameter_sets: Number of parameter sets.

        Returns:
            Combined parameters with learnable bound raw params.

        Raises:
            ValueError: If min_weights_per_asset or max_weights_per_asset
                are not provided in initial_values_dict.
        """
        # Validate that bounds are provided
        min_w = initial_values_dict.get("min_weights_per_asset")
        max_w = initial_values_dict.get("max_weights_per_asset")

        if min_w is None or max_w is None:
            raise ValueError(
                "Bounded pools require min_weights_per_asset and max_weights_per_asset "
                "to be specified in initial_params or run_fingerprint['learnable_bounds_settings']. "
                f"Got min_weights_per_asset={min_w}, max_weights_per_asset={max_w}"
            )

        # Use learnable parameters (raw params that get converted to bounds)
        learnable_params = self.init_learnable_bounded_weight_parameters(
            initial_values_dict, n_assets, n_parameter_sets
        )
        return {**base_params, **learnable_params}

    def init_learnable_bounded_weight_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
    ) -> Dict[str, Any]:
        """
        Initialise learnable parameters for bounded weights.

        Unlike init_bounded_weight_parameters which creates fixed bounds,
        this creates raw parameters that can be optimized via gradient descent.
        The reparameterization guarantees all constraints are satisfied by
        construction at every step of optimization.

        Args:
            initial_values_dict: Initial values. Can contain:
                - target_min_sum: Target sum of minimums (default: 0.25)
                - target_gap_fraction: Target gap as fraction of available (default: 0.5)
                Or, if providing explicit starting bounds:
                - min_weights_per_asset: array or scalar (default: 0.05)
                - max_weights_per_asset: array or scalar (default: 0.95)
            n_assets: Number of assets.
            n_parameter_sets: Number of parameter sets for parallel optimisation.

        Returns:
            Parameters dict with raw learnable parameters.
        """
        # Check if user provided explicit target sums
        target_min_sum = initial_values_dict.get("target_min_sum", None)
        target_gap_fraction = initial_values_dict.get("target_gap_fraction", None)

        if target_min_sum is not None or target_gap_fraction is not None:
            # Use targets to initialize
            target_min_sum = target_min_sum if target_min_sum is not None else 0.25
            target_gap_fraction = target_gap_fraction if target_gap_fraction is not None else 0.5
            return init_learnable_bounds_params(
                n_assets, n_parameter_sets, target_min_sum, target_gap_fraction
            )

        # Otherwise, check for explicit bounds and convert them
        min_w = initial_values_dict.get("min_weights_per_asset", None)
        max_w = initial_values_dict.get("max_weights_per_asset", None)

        if min_w is not None or max_w is not None:
            # Convert explicit bounds to raw parameters (approximately)
            min_w = min_w if min_w is not None else 0.05
            max_w = max_w if max_w is not None else 0.95

            if isinstance(min_w, (int, float)):
                min_w = np.array([min_w] * n_assets)
            else:
                min_w = np.asarray(min_w)
            if isinstance(max_w, (int, float)):
                max_w = np.array([max_w] * n_assets)
            else:
                max_w = np.asarray(max_w)

            # Validate the target bounds
            self.validate_bounds(min_w, max_w, n_assets)

            # Convert to raw parameters (this is approximate - the reparameterization
            # may not exactly reproduce the target bounds, but will be close)
            return self._bounds_to_raw_params(min_w, max_w, n_parameter_sets)

        # Default: use init_learnable_bounds_params with defaults
        return init_learnable_bounds_params(n_assets, n_parameter_sets)

    @staticmethod
    def _bounds_to_raw_params(
        min_weights: np.ndarray,
        max_weights: np.ndarray,
        n_parameter_sets: int = 1,
    ) -> Dict[str, jnp.ndarray]:
        """
        Convert explicit bounds to approximate raw parameters.

        This is an inverse of the reparameterization, used to initialize
        raw parameters from user-specified target bounds. The actual bounds
        after reparameterization may differ slightly.

        Args:
            min_weights: Target minimum weights, shape (n_assets,).
            max_weights: Target maximum weights, shape (n_assets,).
            n_parameter_sets: Number of parallel parameter sets.

        Returns:
            Dict with raw parameter arrays.
        """
        min_w = np.asarray(min_weights)
        max_w = np.asarray(max_weights)

        # min_budget: sum of minimums
        min_sum = np.sum(min_w)
        # Clamp to avoid log(0) or log(inf)
        min_sum_clamped = np.clip(min_sum, 1e-6, 1 - 1e-6)
        raw_min_budget_val = np.log(min_sum_clamped / (1 - min_sum_clamped))

        # min_logits: softmax inverse (log of normalized weights)
        # softmax(x)_i = exp(x_i) / sum(exp(x_j))
        # If we want softmax(x) = p, then x_i = log(p_i) + c for any c
        # We'll use c = 0 for simplicity
        min_fractions = min_w / (min_sum + 1e-8)
        raw_min_logits = np.log(min_fractions + 1e-8)

        # gaps: max - min
        gaps = max_w - min_w
        available = 1.0 - min_w + 1e-8
        # gap = available * sigmoid(raw_gap_logits)
        # sigmoid(x) = gap / available
        gap_fractions = np.clip(gaps / available, 1e-6, 1 - 1e-6)
        raw_gap_logits = np.log(gap_fractions / (1 - gap_fractions))

        return {
            "raw_min_budget": jnp.full((n_parameter_sets,), raw_min_budget_val),
            "raw_min_logits": jnp.tile(raw_min_logits, (n_parameter_sets, 1)),
            "raw_gap_logits": jnp.tile(raw_gap_logits, (n_parameter_sets, 1)),
        }

    @staticmethod
    def raw_params_to_bounds(
        params: Dict[str, Any],
        n_assets: int,
        freeze_bounds: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Convert raw learnable parameters to actual min/max bounds.

        This is the forward pass of the reparameterization, used during
        the forward pass to get the actual bounds from raw parameters.

        Args:
            params: Dict containing raw_min_budget, raw_min_logits,
                raw_gap_logits.
            n_assets: Number of assets.
            freeze_bounds: If True, wrap inputs in stop_gradient so bounds
                act as hyperparameters (not learned). Default False.

        Returns:
            Tuple of (min_weights_per_asset, max_weights_per_asset).
        """
        return reparameterize_bounds(
            params["raw_min_budget"],
            params["raw_min_logits"],
            params["raw_gap_logits"],
            n_assets,
            freeze_bounds=freeze_bounds,
        )

    def extend_parameters_learnable(
        self,
        base_params: Dict[str, Any],
        initial_values_dict: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
    ) -> Dict[str, Any]:
        """
        Extend base pool parameters with learnable bounded weight parameters.

        Unlike extend_parameters which creates fixed bounds, this creates
        raw parameters that can be optimized via gradient descent.

        Args:
            base_params: Base parameters from the underlying pool.
            initial_values_dict: Initial values for all parameters.
            n_assets: Number of assets.
            n_parameter_sets: Number of parameter sets.

        Returns:
            Combined parameters with learnable bound raw params.
        """
        learnable_params = self.init_learnable_bounded_weight_parameters(
            initial_values_dict, n_assets, n_parameter_sets
        )
        return {**base_params, **learnable_params}

    def _tree_flatten(self):
        children = ()
        aux_data = dict()  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
