"""Ensemble averaging hook for training multiple parameter sets jointly.

This hook modifies pool behavior to average rule outputs across ensemble members,
allowing multiple models to be trained together with shared gradients.

The key insight is that by averaging rule_outputs before computing fine weights,
gradients flow back to all ensemble members through the averaging operation.
This creates an implicit regularization effect where ensemble members must
agree on the directional weight changes.

Architecture
------------
With this hook, params have shape: (n_parameter_sets, n_ensemble_members, ...)
Exception: initial_weights_logits has shape (n_parameter_sets, n_assets)
           because initial weights are shared across ensemble members.

- Outer vmap (in jax_runners): over n_parameter_sets (axis 0)
- After slicing: hook receives (n_ensemble_members, ...) for rule params
- Hook vmaps over n_ensemble_members, averages rule outputs
- Returns single output per parameter set

The ensemble is about the *strategy* (rule_outputs), not the starting allocation.
Each ensemble member has different rule parameters (e.g., memory_length, k_per_day)
but they all share the same initial_weights_logits.

Ensemble Initialization Methods
-------------------------------
The hook supports several methods for initializing ensemble members across
parameter space (set via run_fingerprint["ensemble_init_method"]):

- "gaussian" (default): Random Gaussian noise around initial values
- "lhs": Latin Hypercube Sampling - ensures even coverage of each dimension
- "sobol": Sobol quasi-random sequences - low-discrepancy space filling
- "grid": Regular grid (for small ensembles) - deterministic coverage
- "centered_lhs": LHS centered in each stratum (more uniform than random LHS)

The spread of ensemble members is controlled by:
- run_fingerprint["ensemble_init_scale"]: Multiplier for spread (default 0.5)

Backwards compatible: when n_ensemble_members=1 (default), no extra dimension
is added and behavior is unchanged.

Usage:
    from quantammsim.pools.creator import create_pool

    # Create an ensemble-averaged pool
    pool = create_pool("ensemble__momentum")

    # In run_fingerprint, set n_ensemble_members > 1
    run_fingerprint["n_ensemble_members"] = 4
    run_fingerprint["ensemble_init_method"] = "lhs"  # Latin Hypercube
    run_fingerprint["ensemble_init_scale"] = 0.5     # Spread scale
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import jax.numpy as jnp
from jax import vmap

from quantammsim.core_simulator.param_utils import make_vmap_in_axes_dict
from quantammsim.utils.sampling import generate_ensemble_samples, generate_param_space_samples


class EnsembleAveragingHook:
    """
    Hook that averages rule outputs across ensemble members.

    This creates an ensemble where multiple parameter sets (members) contribute
    to a single averaged output. During training, gradients flow back to all
    members through the averaging operation.

    The hook is fully self-contained - no changes to jax_runners are needed.
    It achieves this by:
    1. Overriding init_base_parameters to create (n_parameter_sets, n_ensemble_members, ...) params
    2. Overriding calculate_rule_outputs to vmap over members and average
    3. Returning standard vmap axes so outer vmap works unchanged

    Gradient Flow
    -------------
    Because we use jnp.mean (not stop_gradient), gradients flow back:

        d_metric/d_params[i] = d_metric/d_mean * (1/n) * d_rule_output[i]/d_params[i]

    Each ensemble member receives gradients proportional to how its rule_output
    affected the average.

    Parameters
    ----------
    n_ensemble_members : int (in run_fingerprint)
        Number of ensemble members per parameter set. Default is 1 (no ensembling).
        When > 1, params get an extra dimension and outputs are averaged.

    Example
    -------
    >>> from quantammsim.pools.creator import create_pool
    >>> pool = create_pool("ensemble__momentum")
    >>>
    >>> run_fingerprint["n_ensemble_members"] = 4  # 4 members per ensemble
    >>> run_fingerprint["n_parameter_sets"] = 2   # 2 independent ensembles
    >>>
    >>> # params shape will be (2, 4, ...)
    >>> # Each of the 2 ensembles averages its 4 members
    >>> params = pool.init_base_parameters(
    ...     initial_values_dict, run_fingerprint, n_assets=3, n_parameter_sets=2
    ... )

    Notes
    -----
    - The hook must appear before the base pool in the inheritance order
    - Compatible with other hooks like BoundedWeightsHook
    - n_ensemble_members=1 is a no-op (standard behavior)
    """

    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        """
        Initialize parameters with ensemble member dimension if needed.

        When n_ensemble_members > 1, creates params with shape
        (n_parameter_sets, n_ensemble_members, ...) using structured
        sampling methods for better coverage of parameter space.

        Ensemble initialization methods (set via run_fingerprint["ensemble_init_method"]):
        - "gaussian": Random Gaussian noise (default, backwards compatible)
        - "lhs": Latin Hypercube Sampling
        - "centered_lhs": Centered LHS (more uniform)
        - "sobol": Sobol quasi-random sequences
        - "grid": Regular grid sampling

        The spread is controlled by run_fingerprint["ensemble_init_scale"] (default 0.5).
        """
        n_ensemble_members = run_fingerprint.get("n_ensemble_members", 1)

        if n_ensemble_members <= 1:
            # Standard path - no change
            return super().init_base_parameters(
                initial_values_dict,
                run_fingerprint,
                n_assets,
                n_parameter_sets=n_parameter_sets,
                noise=noise,
            )

        # Get ensemble initialization settings
        ensemble_init_method = run_fingerprint.get("ensemble_init_method", "gaussian")
        ensemble_init_scale = run_fingerprint.get("ensemble_init_scale", 0.5)
        ensemble_init_seed = run_fingerprint.get("ensemble_init_seed", 42)

        if ensemble_init_method == "gaussian":
            # Use the original approach - let base class handle noise
            total = n_parameter_sets * n_ensemble_members
            params = super().init_base_parameters(
                initial_values_dict,
                run_fingerprint,
                n_assets,
                n_parameter_sets=total,
                noise=noise,
            )
        else:
            # Structured sampling approach using shared utility
            # 1. Get base params without noise (single set)
            base_params = super().init_base_parameters(
                initial_values_dict,
                run_fingerprint,
                n_assets,
                n_parameter_sets=1,
                noise="none",  # No noise for base
            )

            # 2. Generate structured samples via shared utility
            total_samples = n_parameter_sets * n_ensemble_members
            samples, ensembled_keys, dim_map = generate_param_space_samples(
                base_params, total_samples, ensemble_init_method, ensemble_init_seed,
            )

            # 3. Transform [0, 1] → [-scale, +scale] centered around base value
            offsets = (samples - 0.5) * 2 * ensemble_init_scale

            # 4. Build params dict with ensemble dimension
            params = {}
            for k, v in base_params.items():
                if k == "subsidary_params":
                    params[k] = v
                elif k == "initial_weights_logits":
                    # Shared across members, just tile for n_parameter_sets
                    params[k] = jnp.tile(v, (n_parameter_sets, 1))
                elif k in dim_map:
                    start_col, n_dims, shape_after = dim_map[k]
                    base_val = v[0]  # Remove the (1,) prefix, get base value

                    param_offsets = offsets[:, start_col:start_col + n_dims]
                    if shape_after:
                        param_offsets = param_offsets.reshape(
                            (total_samples,) + shape_after
                        )

                    all_vals = base_val * (1 + param_offsets)

                    final_shape = (n_parameter_sets, n_ensemble_members) + shape_after
                    params[k] = all_vals.reshape(final_shape)
                else:
                    params[k] = v

            return params

        # Reshape from (total, ...) to (n_parameter_sets, n_ensemble_members, ...)
        # (Only used for gaussian path)
        total = n_parameter_sets * n_ensemble_members
        reshaped = {}
        for k, v in params.items():
            if k == "subsidary_params":
                reshaped[k] = v
            elif k == "initial_weights_logits":
                # Initial weights are shared across ensemble members
                if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == total:
                    temp_shape = (n_parameter_sets, n_ensemble_members) + v.shape[1:]
                    temp = v.reshape(temp_shape)
                    reshaped[k] = temp[:, 0, ...]
                else:
                    reshaped[k] = v
            elif hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == total:
                new_shape = (n_parameter_sets, n_ensemble_members) + v.shape[1:]
                reshaped[k] = v.reshape(new_shape)
            else:
                reshaped[k] = v
        return reshaped

    def make_vmap_in_axes(
        self, params: Dict[str, Any], n_repeats_of_recurred: int = 0
    ) -> Dict[str, Any]:
        """
        Return standard vmap axes - outer vmap handles n_parameter_sets.

        With params shape (n_parameter_sets, n_ensemble_members, ...),
        returning axis 0 means outer vmap slices n_parameter_sets,
        leaving (n_ensemble_members, ...) for the hook to handle internally.

        This requires no changes to jax_runners.
        """
        return super().make_vmap_in_axes(params, n_repeats_of_recurred)

    def calculate_rule_outputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate averaged rule outputs across ensemble members.

        After outer vmap slices, params has shape (n_ensemble_members, ...).
        This method vmaps the base calculation over members and averages.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters with shape (n_ensemble_members, ...) per key
        run_fingerprint : Dict[str, Any]
            Simulation settings including n_ensemble_members
        prices : jnp.ndarray
            Price data
        additional_oracle_input : Optional[jnp.ndarray]
            Extra oracle data

        Returns
        -------
        jnp.ndarray
            Averaged rule outputs, shape (time, n_assets)
        """
        n_ensemble_members = run_fingerprint.get("n_ensemble_members", 1)

        if n_ensemble_members <= 1:
            # No ensembling - standard path
            return super().calculate_rule_outputs(
                params, run_fingerprint, prices, additional_oracle_input
            )

        # Build in_axes for vmapping over ensemble members (axis 0)
        # initial_weights_logits is shared (no ensemble dim), so don't vmap over it
        member_in_axes = {}
        for k, v in params.items():
            if k == "subsidary_params":
                member_in_axes[k] = None
            elif k == "initial_weights_logits":
                # Shared across ensemble members - don't vmap
                member_in_axes[k] = None
            elif hasattr(v, "shape") and len(v.shape) > 0:
                member_in_axes[k] = 0
            else:
                member_in_axes[k] = None

        # Get reference to base class method for use in vmap
        base_calculate = super().calculate_rule_outputs

        # vmap base calculation over ensemble members
        vmapped_outputs = vmap(
            lambda p: base_calculate(p, run_fingerprint, prices, additional_oracle_input),
            in_axes=[member_in_axes],
        )(params)
        # vmapped_outputs shape: (n_ensemble_members, time, n_assets)

        # Average across ensemble members
        averaged = jnp.mean(vmapped_outputs, axis=0)
        # averaged shape: (time, n_assets)

        return averaged

    def calculate_rule_output_step(
        self,
        estimator_carry: Dict[str, jnp.ndarray],
        price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Calculate averaged rule output for a single step (scan path).

        For scan-based weight calculation, this method averages the rule
        outputs across ensemble members for each step.

        Parameters
        ----------
        estimator_carry : Dict[str, jnp.ndarray]
            Estimator state for each member, shape (n_ensemble_members, ...)
        price : jnp.ndarray
            Current price, shape (n_assets,)
        params : Dict[str, Any]
            Parameters with shape (n_ensemble_members, ...)
        run_fingerprint : Dict[str, Any]
            Simulation settings

        Returns
        -------
        Tuple[Dict, jnp.ndarray]
            (updated_estimator_carry, averaged_rule_output)
        """
        n_ensemble_members = run_fingerprint.get("n_ensemble_members", 1)

        if n_ensemble_members <= 1:
            # No ensembling - standard path
            return super().calculate_rule_output_step(
                estimator_carry, price, params, run_fingerprint
            )

        # Build in_axes for params
        # initial_weights_logits is shared (no ensemble dim), so don't vmap over it
        params_in_axes = {}
        for k, v in params.items():
            if k == "subsidary_params":
                params_in_axes[k] = None
            elif k == "initial_weights_logits":
                # Shared across ensemble members - don't vmap
                params_in_axes[k] = None
            elif hasattr(v, "shape") and len(v.shape) > 0:
                params_in_axes[k] = 0
            else:
                params_in_axes[k] = None

        # Build in_axes for estimator_carry
        carry_in_axes = {k: 0 for k in estimator_carry.keys()}

        # Get reference to base class method
        base_step = super().calculate_rule_output_step

        def step_fn(carry, p):
            return base_step(carry, price, p, run_fingerprint)

        # vmap over ensemble members
        new_carries, rule_outputs = vmap(
            step_fn, in_axes=[carry_in_axes, params_in_axes]
        )(estimator_carry, params)
        # rule_outputs shape: (n_ensemble_members, n_assets)

        # Average rule outputs
        averaged_rule_output = jnp.mean(rule_outputs, axis=0)
        # averaged_rule_output shape: (n_assets,)

        # Note: new_carries still has shape (n_ensemble_members, ...) per key
        # This is correct - each member maintains its own estimator state

        return new_carries, averaged_rule_output

    def get_initial_carry(
        self,
        initial_price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
        """
        Get initial carry state for all ensemble members.

        For ensemble averaging in scan path, we need initial carry
        for each ensemble member.

        Parameters
        ----------
        initial_price : jnp.ndarray
            Initial price observation
        params : Dict[str, Any]
            Parameters with shape (n_ensemble_members, ...)
        run_fingerprint : Dict[str, Any]
            Simulation settings

        Returns
        -------
        Tuple[Dict, Dict]
            (estimator_carry, weight_carry) for all members
        """
        n_ensemble_members = run_fingerprint.get("n_ensemble_members", 1)

        if n_ensemble_members <= 1:
            # No ensembling - standard path
            return super().get_initial_carry(initial_price, params, run_fingerprint)

        # Build in_axes for params
        # initial_weights_logits is shared (no ensemble dim), so don't vmap over it
        params_in_axes = {}
        for k, v in params.items():
            if k == "subsidary_params":
                params_in_axes[k] = None
            elif k == "initial_weights_logits":
                # Shared across ensemble members - don't vmap
                params_in_axes[k] = None
            elif hasattr(v, "shape") and len(v.shape) > 0:
                params_in_axes[k] = 0
            else:
                params_in_axes[k] = None

        # Get reference to base class method
        base_init = super().get_initial_carry

        # vmap initialization over ensemble members
        vmapped_init = vmap(
            lambda p: base_init(initial_price, p, run_fingerprint),
            in_axes=[params_in_axes],
        )

        return vmapped_init(params)

    def calculate_readouts(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> Optional[Dict[str, jnp.ndarray]]:
        """
        Calculate readouts with ensemble averaging.

        For ensemble mode, readouts need special handling since they're computed
        per ensemble member. For now, we skip readouts in ensemble mode.

        TODO: Implement proper readout averaging for ensemble mode if needed.
        """
        n_ensemble_members = run_fingerprint.get("n_ensemble_members", 1)

        if n_ensemble_members <= 1:
            # No ensembling - standard path
            return super().calculate_readouts(
                params, run_fingerprint, prices, start_index, additional_oracle_input
            )

        # For ensemble mode, skip readouts for now
        # This avoids shape mismatch issues in the scan function
        return None

    def _tree_flatten(self):
        children = ()
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
