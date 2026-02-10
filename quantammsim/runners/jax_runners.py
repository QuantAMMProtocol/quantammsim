"""Core training and simulation runners for quantammsim.

This module provides the two primary entry points for using quantammsim:

:func:`train_on_historic_data`
    Optimise strategy parameters on historical price data using either
    gradient descent (Adam/AdamW/SGD via Optax) or gradient-free search
    (Optuna).  Supports ensemble training, early stopping with validation
    holdout, warm-starting from previous walk-forward cycles, checkpointing
    for Rademacher complexity analysis, and Stochastic Weight Averaging.

:func:`do_run_on_historic_data`
    Execute a single forward pass (simulation) with fixed parameters and
    return the full results dict.  Used for post-training evaluation,
    walk-forward OOS testing, and visualisation.  Supports injecting
    real trade data, time-varying fees/gas costs, and LP supply changes.

Both functions accept a ``run_fingerprint`` dict as their primary
configuration.  See :doc:`/user_guide/run_fingerprints` for the complete
reference of available settings.
"""

import numpy as np
from copy import deepcopy

from tqdm import tqdm
import math
import gc
import os
import optuna
from jax.tree_util import Partial
from jax import jit, vmap, random, lax
from jax import clear_caches
from jax.tree_util import tree_map

from quantammsim.utils.data_processing.historic_data_utils import (
    get_data_dict,
)

from quantammsim.core_simulator.forward_pass import (
    forward_pass,
    forward_pass_nograd,
    _calculate_return_value,
)
from quantammsim.core_simulator.windowing_utils import get_indices, filter_coarse_weights_by_data_indices

from quantammsim.training.backpropagation import (
    update_from_partial_training_step_factory,
    update_from_partial_training_step_factory_with_optax,
    create_opt_state_in_axes_dict,
    create_optimizer_chain,
)
from quantammsim.core_simulator.param_utils import (
    recursive_default_set,
    check_run_fingerprint,
    memory_days_to_logit_lamb,
    retrieve_best,
    process_initial_values,
    get_run_location,
)

from quantammsim.core_simulator.result_exporter import (
    save_multi_params,
    save_optuna_results_sgd_format,
)

from quantammsim.runners.jax_runner_utils import (
    nan_param_reinit,
    nan_param_reinit_vectorized,
    has_nan_grads,
    Hashabledict,
    NestedHashabledict,
    HashableArrayWrapper,
    get_trades_and_fees,
    get_unique_tokens,
    OptunaManager,
    generate_evaluation_points,
    create_trial_params,
    create_static_dict,
    get_sig_variations,
    BestParamsTracker,
    SELECTION_METHODS,
    init_tracker_state,
    update_tracker_state,
    generate_nan_bank,
    nan_reinit_from_bank,
)

from quantammsim.pools.creator import create_pool

from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.utils.post_train_analysis import (
    calculate_period_metrics,
    calculate_continuous_test_metrics,
    _compute_all_metrics_batched,
    _METRIC_KEYS,
    metrics_arr_to_dicts,
)
import jax.numpy as jnp


def train_on_historic_data(
    run_fingerprint,
    root=None,
    iterations_per_print=1,
    force_init=False,
    price_data=None,
    verbose=True,
    run_location=None,
    return_training_metadata=False,
    warm_start_params=None,
    warm_start_weights=None,
):
    """Optimise strategy parameters on historical price data.

    This is the primary training entry point for quantammsim.  It loads (or
    accepts) price data, constructs the JAX computation graph, and runs either
    gradient-based (Adam/AdamW/SGD) or gradient-free (Optuna) optimisation
    according to ``run_fingerprint["optimisation_settings"]["method"]``.

    Parameters
    ----------
    run_fingerprint : dict
        Master configuration dict.  Key fields consumed here:

        - ``tokens``, ``startDateString``, ``endDateString``,
          ``endTestDateString`` — data selection
        - ``rule`` — pool/strategy type (e.g. ``"momentum"``,
          ``"mean_reversion_channel"``)
        - ``return_val`` — objective metric (default ``"daily_log_sharpe"``)
        - ``optimisation_settings.method`` — ``"gradient_descent"`` or
          ``"optuna"``
        - ``optimisation_settings.optimiser`` — ``"adam"``, ``"adamw"``,
          or ``"sgd"``
        - ``optimisation_settings.n_iterations`` — training epochs
        - ``optimisation_settings.val_fraction`` — fraction of training
          window held out for early-stopping validation (0 = disabled)
        - ``optimisation_settings.use_swa`` — enable Stochastic Weight
          Averaging
        - ``optimisation_settings.track_checkpoints`` — save periodic
          parameter snapshots for Rademacher complexity analysis

        See :doc:`/user_guide/run_fingerprints` for the full reference.
    root : str, optional
        Root directory for data files and saved results.
    iterations_per_print : int, optional
        Print training progress every *N* iterations (default 1).
    force_init : bool, optional
        If True, ignore cached results and re-initialise parameters.
    price_data : array-like or DataFrame, optional
        Pre-loaded price data.  When None, data is loaded from parquet
        files based on ``run_fingerprint`` date/token settings.
    verbose : bool, optional
        Print detailed progress information (default True).
    run_location : str, optional
        Path to a previously-saved run to resume from.  When None, a new
        run is initialised (or auto-detected from the fingerprint hash).
    return_training_metadata : bool, optional
        If True, return ``(params, metadata)`` where *metadata* contains
        ``epochs_trained``, ``final_objective``, and ``checkpoint_returns``
        (a ``(n_checkpoints, T-1)`` array for Rademacher complexity, or
        None if checkpointing was disabled).
    warm_start_params : dict, optional
        Strategy parameters from a previous walk-forward cycle.  Each
        value is expanded to ``(n_parameter_sets, ...)`` shape with
        added Gaussian noise (scale controlled by
        ``optimisation_settings.noise_scale``).
    warm_start_weights : array-like, optional
        Final portfolio weights from a previous cycle, shape
        ``(n_assets,)``.  The pool starts with a fresh
        ``initial_pool_value`` distributed according to these weights.

    Returns
    -------
    dict or tuple or list or None
        - **Gradient descent**, ``return_training_metadata=False``:
          best params dict.
        - **Gradient descent**, ``return_training_metadata=True``:
          ``(params, metadata)`` tuple.
        - **Optuna**: list of best trials, or None if none completed.
    """

    recursive_default_set(run_fingerprint, run_fingerprint_defaults)
    check_run_fingerprint(run_fingerprint)
    if verbose:
        print("Run Fingerprint: ", run_fingerprint)
    rule = run_fingerprint["rule"]
    chunk_period = run_fingerprint["chunk_period"]
    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    use_alt_lamb = run_fingerprint["use_alt_lamb"]
    use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
    fees = run_fingerprint["fees"]
    arb_fees = run_fingerprint["arb_fees"]
    gas_cost = run_fingerprint["gas_cost"]
    n_parameter_sets = run_fingerprint["optimisation_settings"]["n_parameter_sets"]
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    arb_frequency = run_fingerprint["arb_frequency"]
    random_key = random.key(
        run_fingerprint["optimisation_settings"]["initial_random_key"]
    )

    learnable_bounds = run_fingerprint.get("learnable_bounds_settings", {})
    initial_params = {
        "initial_memory_length": run_fingerprint["initial_memory_length"],
        "initial_memory_length_delta": run_fingerprint["initial_memory_length_delta"],
        "initial_k_per_day": run_fingerprint["initial_k_per_day"],
        "initial_weights_logits": run_fingerprint["initial_weights_logits"],
        "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
        "initial_raw_width": run_fingerprint["initial_raw_width"],
        "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
        "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
        "min_weights_per_asset": learnable_bounds.get("min_weights_per_asset"),
        "max_weights_per_asset": learnable_bounds.get("max_weights_per_asset"),
    }

    unique_tokens = get_unique_tokens(run_fingerprint)
    n_tokens = len(unique_tokens)
    n_assets = n_tokens

    all_sig_variations = get_sig_variations(n_assets)

    np.random.seed(0)

    max_memory_days = run_fingerprint["max_memory_days"]

    if price_data is None and verbose:
        print(f"[Data] Loading {run_fingerprint['optimisation_settings']['training_data_kind']} data...")
    data_dict = get_data_dict(
        unique_tokens,
        run_fingerprint,
        data_kind=run_fingerprint["optimisation_settings"]["training_data_kind"],
        root=root,
        max_memory_days=max_memory_days,
        start_date_string=run_fingerprint["startDateString"],
        end_time_string=run_fingerprint["endDateString"],
        start_time_test_string=run_fingerprint["endDateString"],
        end_time_test_string=run_fingerprint["endTestDateString"],
        max_mc_version=run_fingerprint["optimisation_settings"]["max_mc_version"],
        price_data=price_data,
        do_test_period=True,
    )
    max_memory_days = data_dict["max_memory_days"]

    # Validation holdout setup
    # If val_fraction > 0, carve out validation window from end of training
    val_fraction = run_fingerprint["optimisation_settings"].get("val_fraction", 0.0)

    # Validate val_fraction
    if val_fraction < 0 or val_fraction >= 1.0:
        raise ValueError(
            f"val_fraction must be in [0, 1), got {val_fraction}. "
            f"Use 0 for no validation holdout, or a value like 0.2 for 20% validation."
        )

    if val_fraction > 0:
        # Store original bout_length for reference (used for continuous forward pass and test slicing)
        original_bout_length = data_dict["bout_length"]

        # Calculate validation and effective training lengths
        val_length = int(original_bout_length * val_fraction)
        effective_train_length = original_bout_length - val_length

        # Ensure validation window is meaningful (at least 1 day of data for minute frequency)
        min_val_length = run_fingerprint.get("chunk_period", 1440)  # Default 1 day
        if val_length < min_val_length:
            raise ValueError(
                f"val_fraction={val_fraction} results in val_length={val_length} steps, "
                f"which is less than minimum {min_val_length} steps (1 chunk_period). "
                f"Increase val_fraction or use a longer training period."
            )

        # Override data_dict["bout_length"] to be the effective training length
        # This ensures training sampling and forward passes use the correct (reduced) length
        data_dict["bout_length"] = effective_train_length

        val_start_idx = data_dict["start_idx"] + effective_train_length

        # Ensure we have room for random sampling in the training region
        bout_length_window = effective_train_length - run_fingerprint["bout_offset"]

        if bout_length_window <= 0:
            raise ValueError(
                f"val_fraction={val_fraction} is too large. "
                f"effective_train_length ({effective_train_length}) must be > bout_offset ({run_fingerprint['bout_offset']}). "
                f"Either reduce val_fraction or increase bout_length or reduce bout_offset."
            )

        if verbose:
            # Convert steps to days for readability (assuming minute data)
            steps_per_day = 1440
            print(f"[Setup] Validation holdout: {val_fraction*100:.0f}%")
            print(f"  Train: {effective_train_length:,} steps (~{effective_train_length/steps_per_day:.1f} days)")
            print(f"  Val:   {val_length:,} steps (~{val_length/steps_per_day:.1f} days)")
            print(f"  Test:  {data_dict.get('bout_length_test', 0):,} steps (~{data_dict.get('bout_length_test', 0)/steps_per_day:.1f} days)")
    else:
        # No validation holdout - use full training window
        # Early stopping will use test data (not recommended but backwards compatible)
        original_bout_length = data_dict["bout_length"]  # No difference when no validation
        bout_length_window = data_dict["bout_length"] - run_fingerprint["bout_offset"]
        val_length = 0
        val_start_idx = None

    assert bout_length_window > 0

    # Determine the end index for sampling (must not overlap with validation)
    if val_fraction > 0:
        # Sampling must stay within effective training region
        sampling_end_idx = val_start_idx
    else:
        # No validation - use original behavior
        sampling_end_idx = data_dict["end_idx"]

    if run_location is None:
        run_location = './results/' + get_run_location(run_fingerprint) + ".json"

    # Check for cached results (skip if force_init=True)
    if not force_init and os.path.isfile(run_location):
        if verbose:
            print(f"[Cache] Loading cached results from: {run_location}")
        params, step = retrieve_best(run_location, "best_train_objective", False, None)
        loaded = True
    else:
        if force_init and os.path.isfile(run_location) and verbose:
            print(f"[Cache] force_init=True, ignoring cached file")
        loaded = False
    # Create pool
    pool = create_pool(rule)

    # pool must be trainable
    assert pool.is_trainable(), "The selected pool must be trainable for this operation"

    if not loaded:
        # Check if we should warm-start from previous cycle params
        if warm_start_params is not None:
            # Use warm_start_params as initialization for strategy parameters
            # (lamb, k, etc.). Pool starts with fresh initial_pool_value but
            # distributed according to warm_start_weights if provided.
            params = {}
            for key, value in warm_start_params.items():
                if key == "subsidary_params":
                    params[key] = value if value is not None else []
                    continue
                # Skip initial_reserves - we compute fresh reserves below
                if key == "initial_reserves":
                    continue
                if hasattr(value, 'copy'):
                    params[key] = jnp.array(value.copy())
                else:
                    params[key] = jnp.array(value) if not isinstance(value, (list, type(None))) else value

            # Ensure params have correct shape for n_parameter_sets
            # warm_start_params are single param set (shape: (n_assets,) or scalar)
            # need to expand to (n_parameter_sets, ...) format
            # Step 1: Stack to (n_parameter_sets, ...) shape
            for key, value in list(params.items()):
                if key == "subsidary_params" or value is None:
                    continue
                # Convert to array if not already (handles scalars from optuna make_scalar=True)
                arr_value = np.array(value)
                if arr_value.ndim == 0:
                    # Scalar: expand to (n_parameter_sets, 1)
                    params[key] = np.stack([arr_value.reshape(1)] * n_parameter_sets, axis=0)
                else:
                    # Array: expand to (n_parameter_sets, ...)
                    params[key] = np.stack([arr_value] * n_parameter_sets, axis=0)

            # Step 2: Add noise using existing pool method (reuse single source of truth)
            noise_scale = run_fingerprint["optimisation_settings"].get("noise_scale", 0.1)
            per_param_noise_scale = run_fingerprint["optimisation_settings"].get("per_param_noise_scale")
            params = pool.add_noise(params, "gaussian", n_parameter_sets, noise_scale, per_param_noise_scale)

            # Initialize reserves with fresh initial_pool_value
            # If warm_start_weights provided, distribute according to those weights
            # Otherwise use equal weights
            initial_pool_value = run_fingerprint["initial_pool_value"]
            start_prices = data_dict["prices"][data_dict["start_idx"]]
            n_assets_local = len(start_prices)

            if warm_start_weights is not None:
                # Validate warm_start_weights before using
                weights = jnp.array(warm_start_weights)
                weights_sum = jnp.sum(weights)

                if jnp.any(jnp.isnan(weights)):
                    if verbose:
                        print("[Warm-start] Warning: weights contain NaN, using equal weights")
                    warm_start_weights = None
                elif weights_sum <= 0:
                    if verbose:
                        print("[Warm-start] Warning: weights sum <= 0, using equal weights")
                    warm_start_weights = None

            if warm_start_weights is not None:
                # Use previous cycle's ending weights to distribute fresh pool value
                weights = jnp.array(warm_start_weights)
                # Normalize weights to sum to 1 (safety check)
                weights = weights / (jnp.sum(weights) + 1e-10)
                # Compute reserves: value_per_asset = weight * total_value, reserves = value / price
                value_per_asset = weights * initial_pool_value
                fresh_reserves = value_per_asset / start_prices
                if verbose:
                    weights_str = ", ".join([f"{w:.2%}" for w in np.array(weights)])
                    print(f"[Warm-start] Using previous params + weights [{weights_str}]")
            else:
                # Equal weight initial reserves
                value_per_asset = initial_pool_value / n_assets_local
                fresh_reserves = value_per_asset / start_prices
                if verbose:
                    print(f"[Warm-start] Using previous params with equal weights")

            params["initial_reserves"] = jnp.stack([fresh_reserves] * n_parameter_sets, axis=0)

            offset = 0
        else:
            parameter_init_method = run_fingerprint["optimisation_settings"].get(
                "parameter_init_method", "gaussian"
            )
            params = pool.init_parameters(
                initial_params, run_fingerprint, n_tokens, n_parameter_sets,
                noise=parameter_init_method,
            )
            offset = 0
    else:
        offset = step + 1
        if verbose:
            print(f"[Cache] Resuming from step {offset}")
        for key in ["step", "test_objective", "train_objective", "hessian_trace", "local_learning_rate", "iterations_since_improvement", "objective", "continuous_test_metrics", "validation_metrics"]:
            if key in params:
                params.pop(key)
        if run_fingerprint["optimisation_settings"]["method"] == "optuna":
            n_parameter_sets = 1
        for key, value in params.items():
            params[key] = process_initial_values(
                params, key, n_assets, n_parameter_sets, force_scalar=True
            )
        params["subsidary_params"] = []
        # noise_scale controls initialization diversity for param sets 1+
        # Default 0.1 maintains backward compatibility
        noise_scale = run_fingerprint["optimisation_settings"].get("noise_scale", 0.1)
        per_param_noise_scale = run_fingerprint["optimisation_settings"].get("per_param_noise_scale")
        params = pool.add_noise(params, "gaussian", n_parameter_sets, noise_scale=noise_scale, per_param_noise_scale=per_param_noise_scale)

    params_in_axes_dict = pool.make_vmap_in_axes(params)

    # Create static dict using helper - overrides for training-specific values
    base_static_dict = create_static_dict(
        run_fingerprint,
        bout_length=bout_length_window,
        all_sig_variations=all_sig_variations,
        overrides={
            "n_assets": n_assets,
            "training_data_kind": run_fingerprint["optimisation_settings"]["training_data_kind"],
            "do_trades": False,
        },
    )

    partial_training_step = Partial(
        forward_pass,
        prices=data_dict["prices"],
        static_dict=Hashabledict(base_static_dict),
        pool=pool,
    )
    partial_forward_pass_nograd_batch = Partial(
        forward_pass_nograd,
        prices=data_dict["prices"],
        static_dict=Hashabledict(base_static_dict),
        pool=pool,
    )

    # Note: Validation and test metrics are now computed by slicing from the continuous
    # forward pass (which covers train + validation + test) rather than running separate
    # passes. This ensures metrics reflect continuous simulation state.

    returns_train_static_dict = base_static_dict.copy()
    returns_train_static_dict["return_val"] = "returns"
    returns_train_static_dict["bout_length"] = data_dict["bout_length"]
    partial_forward_pass_nograd_batch_returns_train = Partial(
        forward_pass_nograd,
        static_dict=Hashabledict(returns_train_static_dict),
        pool=pool,
    )

    # Create continuous forward pass that covers train + validation + test period
    # Use original_bout_length to include validation period when val_fraction > 0
    continuous_static_dict = base_static_dict.copy()
    continuous_static_dict["return_val"] = "reserves_and_values"
    continuous_static_dict["bout_length"] = original_bout_length + data_dict["bout_length_test"]
    partial_forward_pass_nograd_batch_continuous = Partial(
        forward_pass_nograd,
        static_dict=Hashabledict(continuous_static_dict),
        pool=pool,
    )

    nograd_in_axes = [params_in_axes_dict, None, None]

    partial_forward_pass_nograd_returns_train = jit(
        vmap(
            partial_forward_pass_nograd_batch_returns_train,
            in_axes=nograd_in_axes,
        )
    )
    partial_forward_pass_nograd_continuous = jit(
        vmap(
            partial_forward_pass_nograd_batch_continuous,
            in_axes=nograd_in_axes,
        )
    )

    partial_fixed_training_step = Partial(
        partial_training_step, start_index=(data_dict["start_idx"], 0)
    )

    local_learning_rate = run_fingerprint["optimisation_settings"]["base_lr"]
    iterations_since_improvement = 0

    max_iterations_with_no_improvement = run_fingerprint["optimisation_settings"][
        "decay_lr_plateau"
    ]
    decay_lr_ratio = run_fingerprint["optimisation_settings"]["decay_lr_ratio"]
    min_lr = run_fingerprint["optimisation_settings"]["min_lr"]

    # Early stopping settings
    # If val_fraction > 0, early stopping uses validation metrics (recommended)
    # If val_fraction == 0, early stopping uses test metrics (data leakage - not recommended)
    use_early_stopping = run_fingerprint["optimisation_settings"].get("early_stopping", False)
    early_stopping_patience = run_fingerprint["optimisation_settings"].get("early_stopping_patience", 200)

    # This metric is used for TWO purposes:
    # 1. Early stopping: determines when to stop training (if use_early_stopping=True)
    # 2. Param selection: determines which params to return (if val_fraction > 0)
    # The name "early_stopping_metric" is historical - it's really a "selection_metric"
    selection_metric = run_fingerprint["optimisation_settings"].get("early_stopping_metric", "sharpe")

    # Validate selection metric
    # All metrics are normalized so higher = better (see forward_pass.py _calculate_* functions)
    # These must match keys returned by calculate_period_metrics in post_train_analysis.py
    valid_metrics = [
        "sharpe", "daily_log_sharpe", "return", "returns_over_hodl",
        "returns_over_uniform_hodl", "calmar", "sterling", "ulcer",
    ]
    if (use_early_stopping or val_fraction > 0) and selection_metric not in valid_metrics:
        raise ValueError(
            f"early_stopping_metric '{selection_metric}' is not valid. "
            f"Must be one of: {valid_metrics}"
        )
    metric_direction = 1  # All metrics: higher = better

    # Early stopping state (only used when use_early_stopping=True)
    # Early stopping only controls WHEN to stop, not WHAT params to return.
    # Final param selection is handled by BestParamsTracker.
    best_early_stopping_metric = jnp.where(metric_direction == -1, jnp.inf, -jnp.inf)
    iterations_since_early_stopping_improvement = jnp.int32(0)
    use_validation_for_early_stopping = val_fraction > 0
    warned_about_nan = False  # Track if we've already warned about NaN metrics

    # Initialize BestParamsTracker for unified param selection
    # Selection method depends on whether validation is enabled
    tracker_selection_method = "best_val" if val_fraction > 0 else "best_train"
    params_tracker = BestParamsTracker(
        selection_method=tracker_selection_method,
        metric=selection_metric,
        min_threshold=0.0,
    )

    # SWA settings
    use_swa = run_fingerprint["optimisation_settings"].get("use_swa", False)
    swa_start_frac = run_fingerprint["optimisation_settings"].get("swa_start_frac", 0.75)
    swa_freq = run_fingerprint["optimisation_settings"].get("swa_freq", 10)
    swa_params_list = []  # Will collect parameters for averaging
    n_iterations = run_fingerprint["optimisation_settings"]["n_iterations"]

    # Checkpoint tracking for Rademacher complexity
    track_checkpoints = run_fingerprint["optimisation_settings"].get("track_checkpoints", False)
    checkpoint_interval = run_fingerprint["optimisation_settings"].get("checkpoint_interval", 10)
    checkpoint_returns_list = []  # Will collect returns at each checkpoint for Rademacher

    # Warn about SWA + validation conflict
    if use_swa and val_fraction > 0:
        import warnings
        warnings.warn(
            "Both SWA and validation holdout are enabled. "
            "Validation-based param selection will take precedence over SWA. "
            "To use SWA, set val_fraction=0.",
            UserWarning
        )

    if run_fingerprint["optimisation_settings"]["method"] == "gradient_descent":
        if run_fingerprint["optimisation_settings"]["optimiser"] in ["adam", "adamw"]:
            import optax

            # Create Adam optimizer with the specified learning rate
            optimizer = create_optimizer_chain(run_fingerprint)

            # Initialize optimizer state for each parameter set
            # For multiple parameter sets, each needs its own optimizer state
            # if n_parameter_sets > 1:
            # Use vmap to vectorize optimizer initialization over parameter sets
            init_optimizer = lambda params: optimizer.init(params)
            batched_init = vmap(init_optimizer, in_axes=[params_in_axes_dict])
            opt_state = batched_init(params)
            # else:
            # opt_state = optimizer.init(params)

            opt_state_in_axes_dict = create_opt_state_in_axes_dict(opt_state)
            # Use optax-based update function
            update_batch = update_from_partial_training_step_factory_with_optax(
                partial_training_step,
                optimizer,
                run_fingerprint["optimisation_settings"]["train_on_hessian_trace"],
                partial_fixed_training_step,
            )
            update = jit(
                vmap(
                    update_batch,
                    in_axes=[params_in_axes_dict, None, None, opt_state_in_axes_dict],
                )
            )

        elif run_fingerprint["optimisation_settings"]["optimiser"] == "sgd":

            update_batch = update_from_partial_training_step_factory(
                partial_training_step,
                run_fingerprint["optimisation_settings"]["train_on_hessian_trace"],
                partial_fixed_training_step,
            )

            update = jit(
                vmap(
                    update_batch,
                    in_axes=[params_in_axes_dict, None, None],
                )
            )

        elif run_fingerprint["optimisation_settings"]["optimiser"] != "sgd":
            raise NotImplementedError

        train_prices = data_dict["prices"][data_dict["start_idx"]:data_dict["start_idx"] + data_dict["bout_length"]]
        continuous_prices = data_dict["prices"][data_dict["start_idx"]:data_dict["start_idx"] + original_bout_length + data_dict["bout_length_test"]]
        val_prices = data_dict["prices"][
            data_dict["start_idx"] + data_dict["bout_length"]:
            data_dict["start_idx"] + original_bout_length
        ]

        # ── Pre-generate NaN replacement bank for scan-compatible reinit ──
        nan_bank = generate_nan_bank(
            pool, initial_params, run_fingerprint, n_tokens,
            n_parameter_sets,
        )

        # ── Selection metric index (Python int, baked into closures) ──
        sel_metric_idx = _METRIC_KEYS.index(selection_metric)
        use_val_for_selection = val_fraction > 0
        # Early stopping metric source (resolved at Python time)
        if use_early_stopping:
            if use_validation_for_early_stopping:
                metric_source = "validation"
            else:
                metric_source = "continuous_test"

        # ── Scan carry: static indices for slicing ──
        _start_idx = data_dict["start_idx"]
        _bout_length = data_dict["bout_length"]
        _bout_length_test = data_dict["bout_length_test"]
        _test_start = original_bout_length
        _test_end = original_bout_length + _bout_length_test

        # ── Build the scan body ──────────────────────────────────────────
        is_optax = run_fingerprint["optimisation_settings"]["optimiser"] in ["adam", "adamw"]
        _optim_settings = run_fingerprint["optimisation_settings"]
        _swa_start_step = int(n_iterations * swa_start_frac)

        def scan_body(carry, _):
            """One training iteration — pure JAX, no Python side effects."""
            # 1. Sample indices — get_indices splits the key internally
            #    and returns the advanced key, matching the original loop exactly.
            start_indexes, key = get_indices(
                start_index=_start_idx,
                bout_length=bout_length_window,
                len_prices=sampling_end_idx,
                key=carry["random_key"],
                optimisation_settings=_optim_settings,
            )

            # 2. Gradient update
            if is_optax:
                new_params, obj, old_params, grads, new_opt_state, has_nan = update(
                    carry["params"], start_indexes, carry["lr"], carry["opt_state"],
                )
            else:
                new_params, obj, old_params, grads, has_nan = update(
                    carry["params"], start_indexes, carry["lr"],
                )
                new_opt_state = None

            # 3. NaN reinit from bank
            new_params, nan_count = nan_reinit_from_bank(
                new_params, has_nan, nan_bank, carry["nan_count"],
            )

            # 4. Continuous forward pass (full eval every iteration)
            continuous_outputs = partial_forward_pass_nograd_continuous(
                new_params, (_start_idx, 0), data_dict["prices"],
            )

            # 5. Batched metrics — train
            train_values = continuous_outputs["value"][:, :_bout_length]
            train_reserves = continuous_outputs["reserves"][:, :_bout_length, :]
            train_metrics_arr, _ = _compute_all_metrics_batched(
                train_values, train_reserves, train_prices,
            )

            # 5b. Batched metrics — test
            test_values = continuous_outputs["value"][:, _test_start:_test_end]
            test_reserves = continuous_outputs["reserves"][:, _test_start:_test_end, :]
            test_prices_slice = continuous_prices[_test_start:_test_end]
            cont_test_metrics_arr, _ = _compute_all_metrics_batched(
                test_values, test_reserves, test_prices_slice,
            )

            # 5c. Batched metrics — val (always compute; zeros if unused)
            if val_fraction > 0:
                val_values = continuous_outputs["value"][:, _bout_length:original_bout_length]
                val_reserves = continuous_outputs["reserves"][:, _bout_length:original_bout_length, :]
                val_metrics_arr, _ = _compute_all_metrics_batched(
                    val_values, val_reserves, val_prices,
                )
            else:
                val_metrics_arr = jnp.zeros_like(train_metrics_arr)

            # 6. Tracker update (pure function, small carry)
            tracker, improved = update_tracker_state(
                carry["tracker"], carry["step"], new_params,
                train_metrics_arr, val_metrics_arr, cont_test_metrics_arr,
                sel_metric_idx=sel_metric_idx,
                use_val=use_val_for_selection,
                use_test=False,
            )

            # 7. LR decay (branchless)
            iters_since = jnp.where(improved, jnp.int32(0), carry["iters_since_improvement"] + 1)
            should_decay = iters_since > max_iterations_with_no_improvement
            lr = jnp.where(
                should_decay,
                jnp.maximum(carry["lr"] * decay_lr_ratio, min_lr),
                carry["lr"],
            )
            iters_since = jnp.where(should_decay, jnp.int32(0), iters_since)

            # 8. Early stopping state (branchless)
            if use_early_stopping:
                if use_validation_for_early_stopping:
                    es_metric = jnp.nanmean(val_metrics_arr[:, sel_metric_idx])
                else:
                    es_metric = jnp.nanmean(cont_test_metrics_arr[:, sel_metric_idx])
                es_improved = es_metric > carry["es_metric"]
                new_es_metric = jnp.where(es_improved, es_metric, carry["es_metric"])
                new_es_counter = jnp.where(es_improved, jnp.int32(0), carry["es_counter"] + 1)
            else:
                new_es_metric = carry["es_metric"]
                new_es_counter = carry["es_counter"]

            # 9. SWA accumulator
            if use_swa:
                should_swa = (carry["step"] >= _swa_start_step) & (carry["step"] % swa_freq == 0)
                swa_sum = tree_map(
                    lambda s, p: jnp.where(should_swa, s + p, s),
                    carry["swa_sum"], new_params,
                )
                swa_count = carry["swa_count"] + should_swa.astype(jnp.int32)
            else:
                swa_sum = carry["swa_sum"]
                swa_count = carry["swa_count"]

            new_carry = {
                "params": new_params,
                "random_key": key,
                "lr": lr,
                "iters_since_improvement": iters_since,
                "tracker": tracker,
                "es_metric": new_es_metric,
                "es_counter": new_es_counter,
                "nan_count": nan_count,
                "swa_sum": swa_sum,
                "swa_count": swa_count,
                "step": carry["step"] + 1,
            }
            if is_optax:
                new_carry["opt_state"] = new_opt_state

            per_step = {
                "objective": obj,
                "train_metrics": train_metrics_arr,
                "test_metrics": cont_test_metrics_arr,
                "val_metrics": val_metrics_arr,
                "lr": lr,
                "iters_since_improvement": iters_since,
                "params": new_params,
            }
            return new_carry, per_step

        # ── Initialize carry ─────────────────────────────────────────────
        carry = {
            "params": params,
            "random_key": random_key,
            "lr": jnp.array(local_learning_rate),
            "iters_since_improvement": jnp.int32(0),
            "tracker": init_tracker_state(params, n_parameter_sets),
            "es_metric": jnp.array(-jnp.inf),
            "es_counter": jnp.int32(0),
            "nan_count": jnp.int32(0),
            "swa_sum": tree_map(jnp.zeros_like, params),
            "swa_count": jnp.int32(0),
            "step": jnp.int32(offset),
        }
        if is_optax:
            carry["opt_state"] = opt_state

        # ── Outer loop: scan in chunks, Python between chunks ────────────
        total_iterations = n_iterations + 1  # 0..n_iterations inclusive
        chunk_size = max(iterations_per_print, 1)
        remaining = total_iterations
        completed = 0
        last_objective_value = None

        # Accumulators for save_multi_params (filled from per-step outputs)
        paramSteps = []
        trainingSteps = []
        continuousTestSteps = []
        validationSteps = []
        objectiveSteps = []
        learningRateSteps = []
        interationsSinceImprovementSteps = []
        stepSteps = []

        while remaining > 0:
            actual = min(chunk_size, remaining)

            if actual < chunk_size:
                # Partial last chunk: run Python loop to avoid recompilation
                # at a different scan length
                all_per_steps = {
                    "objective": [],
                    "train_metrics": [],
                    "test_metrics": [],
                    "val_metrics": [],
                    "lr": [],
                    "iters_since_improvement": [],
                    "params": {k: [] for k in carry["params"]},
                }
                for _ in range(actual):
                    carry, step_out = scan_body(carry, None)
                    all_per_steps["objective"].append(step_out["objective"])
                    all_per_steps["train_metrics"].append(step_out["train_metrics"])
                    all_per_steps["test_metrics"].append(step_out["test_metrics"])
                    all_per_steps["val_metrics"].append(step_out["val_metrics"])
                    all_per_steps["lr"].append(step_out["lr"])
                    all_per_steps["iters_since_improvement"].append(step_out["iters_since_improvement"])
                    for k in step_out["params"]:
                        all_per_steps["params"][k].append(step_out["params"][k])
                # Stack into arrays with leading chunk dim
                history = {
                    "objective": jnp.stack(all_per_steps["objective"]),
                    "train_metrics": jnp.stack(all_per_steps["train_metrics"]),
                    "test_metrics": jnp.stack(all_per_steps["test_metrics"]),
                    "val_metrics": jnp.stack(all_per_steps["val_metrics"]),
                    "lr": jnp.stack(all_per_steps["lr"]),
                    "iters_since_improvement": jnp.stack(all_per_steps["iters_since_improvement"]),
                    "params": {
                        k: jnp.stack(v) if v and hasattr(v[0], 'shape') else v
                        for k, v in all_per_steps["params"].items()
                    },
                }
            else:
                carry, history = lax.scan(scan_body, carry, None, length=actual)

            remaining -= actual
            completed += actual
            last_objective_value = history["objective"][-1]

            # ── Python-side: accumulate save data ────────────────────
            def _extract_params_at(params_tree, j):
                """Extract params at step j, handling empty-list pytree leaves."""
                return {
                    k: v[j] if hasattr(v, '__getitem__') and not isinstance(v, list) else v
                    for k, v in params_tree.items()
                }

            for j in range(actual):
                step_j = int(carry["step"]) - actual + j
                p_j = _extract_params_at(history["params"], j)
                paramSteps.append(p_j)
                trainingSteps.append(
                    metrics_arr_to_dicts(history["train_metrics"][j])
                )
                continuousTestSteps.append(
                    metrics_arr_to_dicts(history["test_metrics"][j])
                )
                objectiveSteps.append(history["objective"][j])
                learningRateSteps.append(history["lr"][j])
                interationsSinceImprovementSteps.append(
                    history["iters_since_improvement"][j]
                )
                stepSteps.append(step_j)
                if val_fraction > 0:
                    validationSteps.append(
                        metrics_arr_to_dicts(history["val_metrics"][j])
                    )

            # ── Python-side: checkpoint tracking ─────────────────────
            if track_checkpoints:
                for j in range(actual):
                    global_step = int(carry["step"]) - actual + j
                    if global_step % checkpoint_interval == 0:
                        ckpt_params = _extract_params_at(history["params"], j)
                        ckpt_outputs = partial_forward_pass_nograd_continuous(
                            ckpt_params, (_start_idx, 0), data_dict["prices"],
                        )
                        ckpt_train_values = ckpt_outputs["value"][:, :_bout_length]
                        ckpt_train_prices = data_dict["prices"][_start_idx:_start_idx + _bout_length]
                        price_ratios = ckpt_train_prices / (ckpt_train_prices[0:1] + 1e-10)
                        uniform_hodl_value = price_ratios.mean(axis=-1)
                        model_log_returns = jnp.diff(jnp.log(ckpt_train_values + 1e-10), axis=-1)
                        hodl_log_returns = jnp.diff(jnp.log(uniform_hodl_value + 1e-10))
                        excess_returns = model_log_returns - hodl_log_returns[None, :]
                        checkpoint_excess_returns = np.array(excess_returns.mean(axis=0))
                        minutes_per_day = 1440
                        n_full_days = len(checkpoint_excess_returns) // minutes_per_day
                        if n_full_days > 0:
                            daily_excess = checkpoint_excess_returns[:n_full_days * minutes_per_day]
                            daily_excess = daily_excess.reshape(n_full_days, minutes_per_day).sum(axis=1)
                            if not np.isnan(daily_excess).any():
                                checkpoint_returns_list.append(daily_excess)

            # ── Python-side: display ─────────────────────────────────
            if verbose:
                last_train = history["train_metrics"][-1]
                last_test = history["test_metrics"][-1]
                obj_val = float(np.mean(np.array(last_objective_value)))
                current_step = int(carry["step"]) - 1
                print(f"\n[Iter {current_step}] objective={obj_val:.4f}")

                sharpe_idx = _METRIC_KEYS.index("sharpe")
                roh_idx = _METRIC_KEYS.index("returns_over_uniform_hodl")
                print(f"  Train (IS):  sharpe={float(jnp.nanmean(last_train[:, sharpe_idx])):+.4f}"
                      f"  ret_over_hodl={float(jnp.nanmean(last_train[:, roh_idx])):+.4f}")

                if val_fraction > 0:
                    last_val = history["val_metrics"][-1]
                    print(f"  Val:         sharpe={float(jnp.nanmean(last_val[:, sharpe_idx])):+.4f}"
                          f"  ret_over_hodl={float(jnp.nanmean(last_val[:, roh_idx])):+.4f}")
                    if use_early_stopping:
                        print(f"  Early stop:  {selection_metric}={float(carry['es_metric']):+.4f} "
                              f"(wait={int(carry['es_counter'])}/{early_stopping_patience})")

                print(f"  Test (OOS):  sharpe={float(jnp.nanmean(last_test[:, sharpe_idx])):+.4f}"
                      f"  ret_over_hodl={float(jnp.nanmean(last_test[:, roh_idx])):+.4f}")

            # ── Python-side: save checkpoint ─────────────────────────
            save_multi_params(
                deepcopy(run_fingerprint),
                paramSteps,
                continuousTestSteps,
                trainingSteps,
                objectiveSteps,
                learningRateSteps,
                interationsSinceImprovementSteps,
                stepSteps,
                continuousTestSteps,
                validation_metrics=validationSteps if validationSteps else None,
                sorted_tokens=True,
            )
            paramSteps = []
            trainingSteps = []
            continuousTestSteps = []
            validationSteps = []
            objectiveSteps = []
            learningRateSteps = []
            interationsSinceImprovementSteps = []
            stepSteps = []

            # ── Python-side: early stopping break ────────────────────
            if use_early_stopping and bool(carry["es_counter"] >= early_stopping_patience):
                if verbose:
                    print(f"\n[Early stopping] No {metric_source} {selection_metric} improvement for {early_stopping_patience} iterations")
                    print(f"  Stopped at iteration {int(carry['step']) - 1}, best {selection_metric}={float(carry['es_metric']):+.4f}")
                break

        # ── Post-loop: reconstruct results via BestParamsTracker ─────
        # Reconstruct continuous_outputs for best and last params
        # (not stored in carry to keep carry small)
        best_continuous_outputs = partial_forward_pass_nograd_continuous(
            carry["tracker"]["best_params"], (_start_idx, 0), data_dict["prices"],
        )
        last_continuous_outputs = partial_forward_pass_nograd_continuous(
            carry["params"], (_start_idx, 0), data_dict["prices"],
        )

        # Convert tracker metric arrays to list-of-dicts for compatibility
        best_train_metrics_list = metrics_arr_to_dicts(carry["tracker"]["best_train_metrics"])
        best_test_metrics_list = metrics_arr_to_dicts(carry["tracker"]["best_test_metrics"])
        best_val_metrics_list = metrics_arr_to_dicts(carry["tracker"]["best_val_metrics"]) if val_fraction > 0 else None

        # Compute last-iteration metrics from one more forward pass
        last_train_values = last_continuous_outputs["value"][:, :_bout_length]
        last_train_reserves = last_continuous_outputs["reserves"][:, :_bout_length, :]
        last_train_arr, last_train_dr = _compute_all_metrics_batched(last_train_values, last_train_reserves, train_prices)
        last_train_metrics_list = metrics_arr_to_dicts(last_train_arr, last_train_dr)

        last_test_values = last_continuous_outputs["value"][:, _test_start:_test_end]
        last_test_reserves = last_continuous_outputs["reserves"][:, _test_start:_test_end, :]
        last_test_arr, last_test_dr = _compute_all_metrics_batched(last_test_values, last_test_reserves, continuous_prices[_test_start:_test_end])
        last_test_metrics_list = metrics_arr_to_dicts(last_test_arr, last_test_dr)

        if val_fraction > 0:
            last_val_values = last_continuous_outputs["value"][:, _bout_length:original_bout_length]
            last_val_reserves = last_continuous_outputs["reserves"][:, _bout_length:original_bout_length, :]
            last_val_arr, last_val_dr = _compute_all_metrics_batched(last_val_values, last_val_reserves, val_prices)
            last_val_metrics_list = metrics_arr_to_dicts(last_val_arr, last_val_dr)
        else:
            last_val_metrics_list = None

        # Populate BestParamsTracker for get_results() compatibility
        params_tracker.best_params = carry["tracker"]["best_params"]
        params_tracker.best_metric_value = carry["tracker"]["best_metric_value"]
        params_tracker.best_iteration = carry["tracker"]["best_iteration"]
        params_tracker.best_param_idx = carry["tracker"]["best_param_idx"]
        params_tracker.best_train_metrics = best_train_metrics_list
        params_tracker.best_continuous_test_metrics = best_test_metrics_list
        params_tracker.best_val_metrics = best_val_metrics_list
        params_tracker.best_continuous_outputs = {
            "reserves": best_continuous_outputs["reserves"],
            "weights": best_continuous_outputs["weights"],
        }

        params_tracker.last_params = carry["params"]
        params_tracker.last_iteration = int(carry["step"]) - 1
        params_tracker.last_param_idx = carry["tracker"]["best_param_idx"]  # Use best as proxy for last selection
        params_tracker.last_train_metrics = last_train_metrics_list
        params_tracker.last_continuous_test_metrics = last_test_metrics_list
        params_tracker.last_val_metrics = last_val_metrics_list
        params_tracker.last_continuous_outputs = {
            "reserves": last_continuous_outputs["reserves"],
            "weights": last_continuous_outputs["weights"],
        }

        # Compute last_param_idx properly for last iteration
        from quantammsim.runners.jax_runner_utils import compute_selection_metric
        _, last_param_idx = compute_selection_metric(
            last_train_metrics_list,
            last_val_metrics_list,
            last_test_metrics_list,
            method=tracker_selection_method,
            metric=selection_metric,
        )
        params_tracker.last_param_idx = last_param_idx

        tracker_results = params_tracker.get_results(n_parameter_sets, original_bout_length)

        # Update objective_value and i for post-loop code
        objective_value = last_objective_value
        i = completed - 1

        if verbose:
            obj_val = float(np.mean(objective_value)) if hasattr(objective_value, '__len__') else float(objective_value)
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE - {i + 1} iterations")
            print(f"{'='*60}")
            print(f"Final objective: {obj_val:.4f}")
            print(f"Selection: method={tracker_results['selection_method']}, metric={tracker_results['selection_metric']}")

        # Build training metadata for analysis and evaluation
        # Includes both "last" (final iteration) and "best" (by selection method) results
        training_metadata = {
            "method": "gradient_descent",
            "epochs_trained": i + 1,  # Actual iterations completed
            "final_objective": float(np.array(objective_value).mean()),

            # Last iteration metrics (for all param sets)
            "last_train_metrics": tracker_results["last_train_metrics"],
            "last_continuous_test_metrics": tracker_results["last_continuous_test_metrics"],
            "last_val_metrics": tracker_results["last_val_metrics"],
            "last_param_idx": tracker_results["last_param_idx"],
            "last_final_reserves": tracker_results["last_final_reserves"][tracker_results["last_param_idx"]] if tracker_results["last_final_reserves"] is not None else None,
            "last_final_weights": tracker_results["last_final_weights"][tracker_results["last_param_idx"]] if tracker_results["last_final_weights"] is not None else None,

            # Best iteration metrics (by selection method)
            "best_train_metrics": tracker_results["best_train_metrics"],
            "best_continuous_test_metrics": tracker_results["best_continuous_test_metrics"],
            "best_val_metrics": tracker_results["best_val_metrics"],
            "best_param_idx": tracker_results["best_param_idx"],
            "best_iteration": tracker_results["best_iteration"],
            "best_metric_value": tracker_results["best_metric_value"],
            "best_final_reserves": tracker_results["best_final_reserves"][tracker_results["best_param_idx"]] if tracker_results["best_final_reserves"] is not None else None,
            "best_final_weights": tracker_results["best_final_weights"][tracker_results["best_param_idx"]] if tracker_results["best_final_weights"] is not None else None,

            # Selection info
            "selection_method": tracker_results["selection_method"],
            "selection_metric": tracker_results["selection_metric"],

            # Legacy field names (for backward compatibility)
            # TODO: Deprecate these in favor of best_* fields
            "final_train_metrics": tracker_results["best_train_metrics"],
            "final_continuous_test_metrics": tracker_results["best_continuous_test_metrics"],
            "final_weights": tracker_results["best_final_weights"][tracker_results["best_param_idx"]] if tracker_results["best_final_weights"] is not None else None,
            "final_reserves": tracker_results["best_final_reserves"][tracker_results["best_param_idx"]] if tracker_results["best_final_reserves"] is not None else None,

            # Provenance
            "run_location": run_location,
            "run_fingerprint": deepcopy(run_fingerprint),
        }

        if track_checkpoints and checkpoint_returns_list:
            training_metadata["checkpoint_returns"] = np.stack(checkpoint_returns_list, axis=0)
        else:
            training_metadata["checkpoint_returns"] = None

        # SWA: Stochastic Weight Averaging (only if no validation data)
        # SWA averages params across TIME (different training iterations), not across param sets.
        # After SWA averaging, we still have n_parameter_sets param sets - we then select the
        # best one based on the tracker's best_param_idx.
        if use_swa and int(carry["swa_count"]) > 0 and val_fraction == 0:
            if verbose:
                print(f"Applying SWA: averaged {int(carry['swa_count'])} parameter snapshots")
            swa_params = tree_map(
                lambda s: s / carry["swa_count"],
                carry["swa_sum"],
            )
            # Handle scalars: use last params for scalar keys
            for key in swa_params:
                if hasattr(swa_params[key], 'shape') and swa_params[key].ndim == 0:
                    swa_params[key] = carry["params"][key]
            # Select param set using tracker's best_param_idx
            selected_params = params_tracker.select_param_set(swa_params, tracker_results["best_param_idx"], n_parameter_sets)
            if return_training_metadata:
                return selected_params, training_metadata
            return selected_params

        # Return best params from tracker
        best_params = tracker_results["best_params"]
        best_idx = tracker_results["best_param_idx"]

        if verbose:
            # Print best iteration results
            print(f"\nBest iteration: {tracker_results['best_iteration']} (param_set={best_idx})")
            print(f"  Selection {tracker_results['selection_metric']}: {tracker_results['best_metric_value']:+.4f}")

            # Best train metrics
            if tracker_results["best_train_metrics"]:
                best_train = tracker_results["best_train_metrics"][best_idx]
                print(f"  Train (IS):  sharpe={best_train.get('sharpe', np.nan):+.4f}  "
                      f"ret_over_hodl={best_train.get('returns_over_uniform_hodl', np.nan):+.4f}")

            # Best validation metrics (if used)
            if tracker_results["best_val_metrics"] and tracker_results["best_val_metrics"][best_idx]:
                best_val = tracker_results["best_val_metrics"][best_idx]
                print(f"  Val:         sharpe={best_val.get('sharpe', np.nan):+.4f}  "
                      f"ret_over_hodl={best_val.get('returns_over_uniform_hodl', np.nan):+.4f}")

            # Best continuous test metrics (OOS)
            if tracker_results["best_continuous_test_metrics"]:
                best_test = tracker_results["best_continuous_test_metrics"][best_idx]
                print(f"  Test (OOS):  sharpe={best_test.get('sharpe', np.nan):+.4f}  "
                      f"ret_over_hodl={best_test.get('returns_over_uniform_hodl', np.nan):+.4f}")

            # Compare with last iteration if different
            if tracker_results["best_iteration"] != i:
                print(f"\nLast iteration: {i}")
                if tracker_results["last_train_metrics"]:
                    last_train = tracker_results["last_train_metrics"][tracker_results["last_param_idx"]]
                    print(f"  Train (IS):  sharpe={last_train.get('sharpe', np.nan):+.4f}  "
                          f"ret_over_hodl={last_train.get('returns_over_uniform_hodl', np.nan):+.4f}")
                if tracker_results["last_continuous_test_metrics"]:
                    last_test = tracker_results["last_continuous_test_metrics"][tracker_results["last_param_idx"]]
                    print(f"  Test (OOS):  sharpe={last_test.get('sharpe', np.nan):+.4f}  "
                          f"ret_over_hodl={last_test.get('returns_over_uniform_hodl', np.nan):+.4f}")

            print(f"{'='*60}")

        selected_params = params_tracker.select_param_set(best_params, best_idx, n_parameter_sets)

        if return_training_metadata:
            return selected_params, training_metadata
        return selected_params
    elif run_fingerprint["optimisation_settings"]["method"] == "optuna":

        n_evaluation_points = 20
        min_spacing = data_dict["bout_length"] // 2  # E

        run_fingerprint["optimisation_settings"]["n_parameter_sets"] = 1
        # assert run_fingerprint["optimisation_settings"]["n_parameter_sets"] == 1, \
        #     "Optuna only supports single parameter sets"

        # Generate and store evaluation points
        if "evaluation_starts" not in run_fingerprint:
            evaluation_starts = generate_evaluation_points(
                data_dict["start_idx"],
                data_dict["end_idx"],
                bout_length_window,
                n_evaluation_points,
                min_spacing,
                run_fingerprint["optimisation_settings"]["initial_random_key"],
            )
            run_fingerprint["evaluation_starts"] = [int(e) for e in evaluation_starts]
        else:
            evaluation_starts = run_fingerprint["evaluation_starts"]

        reserves_values_train_static_dict = base_static_dict.copy()
        reserves_values_train_static_dict["return_val"] = "reserves_and_values"
        reserves_values_train_static_dict["bout_length"] = data_dict["bout_length"]
        partial_forward_pass_nograd_batch_reserves_values_train = jit(
            Partial(
                forward_pass_nograd,
                static_dict=Hashabledict(reserves_values_train_static_dict),
                pool=pool,
            )
        )

        reserves_values_test_static_dict = base_static_dict.copy()
        reserves_values_test_static_dict["return_val"] = "reserves_and_values"
        reserves_values_test_static_dict["bout_length"] = data_dict["bout_length_test"]
        partial_forward_pass_nograd_batch_reserves_values_test = jit(
            Partial(
                forward_pass_nograd,
                static_dict=Hashabledict(reserves_values_test_static_dict),
                pool=pool,
            )
        )

        # Continuous forward pass covering train + test for proper continuous metrics
        continuous_optuna_static_dict = base_static_dict.copy()
        continuous_optuna_static_dict["return_val"] = "reserves_and_values"
        continuous_optuna_static_dict["bout_length"] = original_bout_length + data_dict["bout_length_test"]
        partial_forward_pass_continuous_optuna = jit(
            Partial(
                forward_pass_nograd,
                static_dict=Hashabledict(continuous_optuna_static_dict),
                pool=pool,
            )
        )

        # Initialize Optuna manager
        optuna_manager = OptunaManager(run_fingerprint)
        optuna_manager.setup_study(
            multi_objective=run_fingerprint["optimisation_settings"]["optuna_settings"][
                "multi_objective"
            ]
        )

        run_fingerprint["optimisation_settings"]["optuna_settings"]["parameter_config"][
            "logit_lamb"
        ] = {
            "low": float(
                memory_days_to_logit_lamb(
                    0.5, chunk_period=base_static_dict["chunk_period"]
                )
            ),
            "high": float(
                memory_days_to_logit_lamb(
                    base_static_dict["max_memory_days"],
                    chunk_period=base_static_dict["chunk_period"],
                )
            ),
            "log_scale": False,
        }

        # Get optuna-specific settings
        optuna_settings = run_fingerprint["optimisation_settings"]["optuna_settings"]
        expand_around = optuna_settings.get("expand_around", True)
        overfitting_penalty = optuna_settings.get("overfitting_penalty", 0.0)

        # Create objective with parameter configuration and validation
        def objective(trial):
            try:
                param_config = run_fingerprint["optimisation_settings"][
                    "optuna_settings"
                ]["parameter_config"]

                if run_fingerprint["optimisation_settings"]["optuna_settings"][
                    "make_scalar"
                ]:
                    # Set scalar=True for all parameter configurations
                    for param_key in param_config:
                        param_config[param_key]["scalar"] = True

                trial_params = create_trial_params(
                    trial, param_config, params, run_fingerprint, n_assets, expand_around=expand_around
                )
                # Training evaluation
                train_outputs = partial_forward_pass_nograd_batch_reserves_values_train(
                    trial_params,
                    (data_dict["start_idx"], 0),
                    data_dict["prices"],
                )
                # Calculate objectives for each evaluation point through slicing
                train_objectives = []
                for start_offset in evaluation_starts:
                    # Calculate relative indices for slicing
                    start_idx = start_offset - data_dict["start_idx"]
                    end_idx = start_idx + data_dict["bout_length"]

                    # Slice the relevant portions of the full trajectory
                    train_value = _calculate_return_value(
                        run_fingerprint["return_val"],
                        train_outputs["reserves"][start_idx:end_idx],
                        data_dict["prices"][start_idx:end_idx],
                        train_outputs["value"][start_idx:end_idx],
                        initial_reserves=train_outputs["reserves"][start_idx],
                    )
                    train_objectives.append(train_value)

                mean_train_value = jnp.sum(jnp.array(train_objectives)) / len(train_objectives)
                train_value = _calculate_return_value(
                    run_fingerprint["return_val"],
                    train_outputs["reserves"],
                    train_outputs["prices"],
                    train_outputs["value"],
                    initial_reserves=train_outputs["reserves"][0],
                )

                train_sharpe = _calculate_return_value(
                    "sharpe",
                    train_outputs["reserves"],
                    train_outputs["prices"],
                    train_outputs["value"],
                )

                train_return = (
                    train_outputs["value"][-1] / train_outputs["value"][0] - 1.0
                )

                train_returns_over_hodl = _calculate_return_value(
                    "returns_over_hodl",
                    train_outputs["reserves"],
                    train_outputs["prices"],
                    train_outputs["value"],
                    initial_reserves=train_outputs["reserves"][0],
                )

                train_returns_over_uniform_hodl = _calculate_return_value(
                    "returns_over_uniform_hodl",
                    train_outputs["reserves"],
                    train_outputs["prices"],
                    train_outputs["value"],
                    initial_reserves=train_outputs["reserves"][0],
                )

                # Test period evaluation using continuous forward pass
                # This ensures test metrics reflect continuous simulation from training
                continuous_outputs = partial_forward_pass_continuous_optuna(
                    trial_params,
                    (data_dict["start_idx"], 0),
                    data_dict["prices"],
                )

                # Calculate continuous test metrics first (always needed)
                continuous_prices = data_dict["prices"][
                    data_dict["start_idx"]:data_dict["start_idx"] + original_bout_length + data_dict["bout_length_test"]
                ]
                continuous_test_dict = {
                    "value": continuous_outputs["value"],
                    "reserves": continuous_outputs["reserves"],
                }
                continuous_test_metrics = calculate_continuous_test_metrics(
                    continuous_test_dict,
                    original_bout_length,
                    data_dict["bout_length_test"],
                    continuous_prices,
                )

                # Calculate validation metrics
                train_length = data_dict["bout_length"]
                if val_fraction > 0:
                    # Validation period exists between train and test
                    validation_reserves = continuous_outputs["reserves"][train_length:original_bout_length]
                    validation_value_arr = continuous_outputs["value"][train_length:original_bout_length]
                    validation_prices = continuous_outputs["prices"][train_length:original_bout_length]

                    validation_value = _calculate_return_value(
                        run_fingerprint["return_val"],
                        validation_reserves,
                        validation_prices,
                        validation_value_arr,
                        initial_reserves=validation_reserves[0],
                    )

                    validation_sharpe = _calculate_return_value(
                        "sharpe",
                        validation_reserves,
                        validation_prices,
                        validation_value_arr,
                    )

                    validation_return = (
                        validation_value_arr[-1] / validation_value_arr[0]
                        - 1.0
                    )

                    validation_returns_over_hodl = _calculate_return_value(
                        "returns_over_hodl",
                        validation_reserves,
                        validation_prices,
                        validation_value_arr,
                        initial_reserves=validation_reserves[0],
                    )

                    validation_returns_over_uniform_hodl = _calculate_return_value(
                        "returns_over_uniform_hodl",
                        validation_reserves,
                        validation_prices,
                        validation_value_arr,
                        initial_reserves=validation_reserves[0],
                    )
                else:
                    # No validation period - use continuous test metrics
                    validation_value = continuous_test_metrics.get(run_fingerprint["return_val"], continuous_test_metrics["sharpe"])
                    validation_sharpe = continuous_test_metrics["sharpe"]
                    validation_return = continuous_test_metrics["return"]
                    validation_returns_over_hodl = continuous_test_metrics["returns_over_hodl"]
                    validation_returns_over_uniform_hodl = continuous_test_metrics["returns_over_uniform_hodl"]

                # Log both training and validation metrics
                # optuna_manager.logger.info(f"Trial {trial.number}:")
                optuna_manager.logger.info(
                    f"Training {trial.number}, Return over HODL: {train_returns_over_hodl}"
                )
                optuna_manager.logger.info(
                    f"Training {trial.number}, Return: {train_return}"
                )
                optuna_manager.logger.info(
                    f"Training {trial.number}, Sharpe: {train_sharpe}"
                )
                optuna_manager.logger.info(
                    f"Training {trial.number}, {run_fingerprint['return_val']}: {train_value}"
                )
                optuna_manager.logger.info(
                    f"Validation {trial.number}, Return over HODL: {validation_returns_over_hodl}"
                )
                optuna_manager.logger.info(
                    f"Validation {trial.number}, Return: {validation_return}"
                )
                optuna_manager.logger.info(
                    f"Validation {trial.number}, Sharpe: {validation_sharpe}"
                )
                optuna_manager.logger.info(
                    f"Validation {trial.number}, {run_fingerprint['return_val']}: {validation_value}"
                )
                for i, value in enumerate(train_objectives):
                    optuna_manager.logger.info(
                        f"Training {trial.number},  Evaluation point {i}: {value}"
                    )
                optuna_manager.logger.info(
                    f"Training {trial.number},  Mean value: {mean_train_value}"
                )
                # Store validation value as a trial attribute
                trial.set_user_attr("validation_value", validation_value)
                trial.set_user_attr(
                    "validation_returns_over_hodl", validation_returns_over_hodl
                )
                trial.set_user_attr("validation_returns_over_uniform_hodl", validation_returns_over_uniform_hodl)
                trial.set_user_attr("validation_sharpe", validation_sharpe)
                trial.set_user_attr("validation_return", validation_return)
                trial.set_user_attr("train_value", train_value)
                trial.set_user_attr("train_returns_over_hodl", train_returns_over_hodl)
                trial.set_user_attr("train_returns_over_uniform_hodl", train_returns_over_uniform_hodl)
                trial.set_user_attr("train_sharpe", train_sharpe)
                trial.set_user_attr("train_return", train_return)
                trial.set_user_attr("train_objectives", train_objectives)
                trial.set_user_attr("mean_train_value", mean_train_value)
                # Store continuous test metrics (same ones as train/val)
                trial.set_user_attr("continuous_test_sharpe", continuous_test_metrics["sharpe"])
                trial.set_user_attr("continuous_test_return", continuous_test_metrics["return"])
                trial.set_user_attr("continuous_test_returns_over_hodl", continuous_test_metrics["returns_over_hodl"])
                trial.set_user_attr("continuous_test_returns_over_uniform_hodl", continuous_test_metrics["returns_over_uniform_hodl"])

                if run_fingerprint["optimisation_settings"]["optuna_settings"][
                    "multi_objective"
                ]:
                    return (
                        np.mean(train_objectives),  # mean_return
                        np.min(train_objectives),  # worst_case
                        -np.std(train_objectives),  # stability
                    )
                else:
                    # Apply overfitting penalty if configured
                    # Penalty is proportional to (train - validation) gap when train > validation
                    if overfitting_penalty > 0:
                        train_val_gap = float(mean_train_value) - float(validation_value)
                        if train_val_gap > 0:  # Only penalize if training better than validation
                            penalty = overfitting_penalty * train_val_gap
                            penalized_value = float(mean_train_value) - penalty
                            trial.set_user_attr("overfitting_penalty_applied", float(penalty))
                            return penalized_value
                    return mean_train_value  # Optimize on training value

            except Exception as e:
                import traceback
                optuna_manager.logger.error(f"Trial {trial.number} failed: {str(e)}")
                optuna_manager.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise e

        # Run optimization
        optuna_manager.optimize(objective)

        # Check if any trials completed successfully
        completed_trials = [
            t for t in optuna_manager.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        # Save results in SGD-compatible format for unified downstream analysis
        if completed_trials:
            sgd_format_path = save_optuna_results_sgd_format(
                run_fingerprint=run_fingerprint,
                study=optuna_manager.study,
                n_assets=n_assets,
                sorted_tokens=True,
            )
            if verbose:
                print(f"Saved SGD-compatible results to: {sgd_format_path}")

        if verbose:
            n_total = len(optuna_manager.study.trials)
            n_completed = len(completed_trials)
            n_pruned = len([t for t in optuna_manager.study.trials if t.state == optuna.trial.TrialState.PRUNED])
            n_failed = n_total - n_completed - n_pruned

            print(f"\n{'='*60}")
            print(f"OPTUNA OPTIMIZATION COMPLETE")
            print(f"{'='*60}")
            print(f"Trials: {n_completed} completed, {n_pruned} pruned, {n_failed} failed (of {n_total} total)")

            if not completed_trials:
                print("\nWARNING: No trials completed successfully!")
            elif run_fingerprint["optimisation_settings"]["optuna_settings"]["multi_objective"]:
                print(f"\nPareto front ({len(optuna_manager.study.best_trials)} trials):")
                for i, trial in enumerate(optuna_manager.study.best_trials[:5]):  # Show top 5
                    train_val = trial.values[0] if trial.values else 0
                    test_val = trial.user_attrs.get('validation_value', 0)
                    print(f"  [{i+1}] Train={train_val:+.4f}  Test={test_val:+.4f}  (trial #{trial.number})")
                if len(optuna_manager.study.best_trials) > 5:
                    print(f"  ... and {len(optuna_manager.study.best_trials) - 5} more")
            else:
                best = optuna_manager.study.best_trial
                train_sharpe = best.user_attrs.get('train_sharpe', best.value)
                test_sharpe = best.user_attrs.get('validation_value', 0)
                train_roh = best.user_attrs.get('train_returns_over_hodl', 0)
                print(f"\nBest trial: #{best.number}")
                print(f"  Train (IS):  sharpe={train_sharpe:+.4f}  ret_over_hodl={train_roh:+.4f}")
                print(f"  Test (OOS):  sharpe={test_sharpe:+.4f}")
            print(f"{'='*60}")

        if completed_trials:
            # Convert best trial params to dict format like gradient descent returns
            from quantammsim.core_simulator.result_exporter import _optuna_params_to_arrays
            best_trial = optuna_manager.study.best_trial
            last_trial = completed_trials[-1]  # Most recent trial

            best_params = _optuna_params_to_arrays(best_trial.params, n_assets)
            best_params["subsidary_params"] = []
            if "initial_weights_logits" not in best_params:
                best_params["initial_weights_logits"] = jnp.zeros(n_assets)

            last_params = _optuna_params_to_arrays(last_trial.params, n_assets)
            last_params["subsidary_params"] = []
            if "initial_weights_logits" not in last_params:
                last_params["initial_weights_logits"] = jnp.zeros(n_assets)

            if return_training_metadata:
                # Run continuous forward passes for both best and last trials
                best_continuous_outputs = partial_forward_pass_continuous_optuna(
                    best_params,
                    (data_dict["start_idx"], 0),
                    data_dict["prices"],
                )
                last_continuous_outputs = partial_forward_pass_continuous_optuna(
                    last_params,
                    (data_dict["start_idx"], 0),
                    data_dict["prices"],
                )

                # Extract final state at end of TRAINING period (for warm-starting)
                # Use bout_length - 1 to get state at end of training
                train_length = data_dict["bout_length"]
                best_final_reserves = np.array(best_continuous_outputs["reserves"][train_length - 1])
                best_final_weights = np.array(best_continuous_outputs["weights"][train_length - 1])
                last_final_reserves = np.array(last_continuous_outputs["reserves"][train_length - 1])
                last_final_weights = np.array(last_continuous_outputs["weights"][train_length - 1])

                # Build train metrics for best trial
                best_train_metrics = {
                    "sharpe": float(best_trial.user_attrs.get("train_sharpe", 0)),
                    "returns": float(best_trial.user_attrs.get("train_return", 0)),
                    "returns_over_hodl": float(best_trial.user_attrs.get("train_returns_over_hodl", 0)),
                    "returns_over_uniform_hodl": float(best_trial.user_attrs.get("train_returns_over_uniform_hodl", 0)),
                    run_fingerprint["return_val"]: float(best_trial.user_attrs.get("train_value", 0)),
                }

                # Build train metrics for last trial
                last_train_metrics = {
                    "sharpe": float(last_trial.user_attrs.get("train_sharpe", 0)),
                    "returns": float(last_trial.user_attrs.get("train_return", 0)),
                    "returns_over_hodl": float(last_trial.user_attrs.get("train_returns_over_hodl", 0)),
                    "returns_over_uniform_hodl": float(last_trial.user_attrs.get("train_returns_over_uniform_hodl", 0)),
                    run_fingerprint["return_val"]: float(last_trial.user_attrs.get("train_value", 0)),
                }

                # Compute continuous_test_metrics for best trial
                continuous_prices = data_dict["prices"][
                    data_dict["start_idx"]:data_dict["start_idx"] + original_bout_length + data_dict["bout_length_test"]
                ]
                best_continuous_dict = {
                    "value": best_continuous_outputs["value"],
                    "reserves": best_continuous_outputs["reserves"],
                }
                best_continuous_test_metrics = calculate_continuous_test_metrics(
                    best_continuous_dict,
                    original_bout_length,
                    data_dict["bout_length_test"],
                    continuous_prices
                )

                # Compute continuous_test_metrics for last trial
                last_continuous_dict = {
                    "value": last_continuous_outputs["value"],
                    "reserves": last_continuous_outputs["reserves"],
                }
                last_continuous_test_metrics = calculate_continuous_test_metrics(
                    last_continuous_dict,
                    original_bout_length,
                    data_dict["bout_length_test"],
                    continuous_prices
                )

                # Return unified metadata matching gradient_descent format
                metadata = {
                    "method": "optuna",
                    "epochs_trained": len(completed_trials),

                    # Last trial metrics
                    "last_train_metrics": [last_train_metrics],
                    "last_continuous_test_metrics": [last_continuous_test_metrics],
                    "last_val_metrics": None,  # Optuna doesn't have validation holdout
                    "last_param_idx": 0,
                    "last_final_reserves": last_final_reserves,
                    "last_final_weights": last_final_weights,

                    # Best trial metrics
                    "best_train_metrics": [best_train_metrics],
                    "best_continuous_test_metrics": [best_continuous_test_metrics],
                    "best_val_metrics": None,  # Optuna doesn't have validation holdout
                    "best_param_idx": 0,
                    "best_iteration": best_trial.number,
                    "best_metric_value": float(best_trial.value) if best_trial.value is not None else 0.0,
                    "best_final_reserves": best_final_reserves,
                    "best_final_weights": best_final_weights,

                    # Selection info
                    "selection_method": "best_train",  # Optuna optimizes on training objective
                    "selection_metric": run_fingerprint["return_val"],

                    # Legacy fields (for backward compat)
                    "final_train_metrics": [best_train_metrics],
                    "final_continuous_test_metrics": [best_continuous_test_metrics],
                    "final_objective": float(best_trial.value) if best_trial.value is not None else 0.0,
                    "final_weights": best_final_weights,
                    "final_reserves": best_final_reserves,

                    # Provenance
                    "run_location": run_location,
                    "run_fingerprint": deepcopy(run_fingerprint),
                    "checkpoint_returns": None,

                    # Optuna-specific extras
                    "n_trials": len(completed_trials),
                    "best_value": float(best_trial.value) if best_trial.value is not None else None,
                }

                if verbose:
                    # Print continuous test metrics (computed from actual forward pass)
                    print(f"\nContinuous test metrics (from forward pass):")
                    print(f"  Best trial #{best_trial.number}:")
                    print(f"    Train (IS):  sharpe={best_train_metrics.get('sharpe', 0):+.4f}  "
                          f"ret_over_hodl={best_train_metrics.get('returns_over_hodl', 0):+.4f}")
                    print(f"    Test (OOS):  sharpe={best_continuous_test_metrics.get('sharpe', 0):+.4f}  "
                          f"ret_over_hodl={best_continuous_test_metrics.get('returns_over_uniform_hodl', 0):+.4f}")
                    if best_trial.number != last_trial.number:
                        print(f"  Last trial #{last_trial.number}:")
                        print(f"    Train (IS):  sharpe={last_train_metrics.get('sharpe', 0):+.4f}  "
                              f"ret_over_hodl={last_train_metrics.get('returns_over_hodl', 0):+.4f}")
                        print(f"    Test (OOS):  sharpe={last_continuous_test_metrics.get('sharpe', 0):+.4f}  "
                              f"ret_over_hodl={last_continuous_test_metrics.get('returns_over_uniform_hodl', 0):+.4f}")

                return best_params, metadata
            return best_params
        else:
            if return_training_metadata:
                return None, {
                    "method": "optuna",
                    "n_trials": 0,
                    "error": "No trials completed",
                    "epochs_trained": 0,

                    # Last trial metrics (none available)
                    "last_train_metrics": None,
                    "last_continuous_test_metrics": None,
                    "last_final_reserves": None,
                    "last_final_weights": None,

                    # Best trial metrics (none available)
                    "best_train_metrics": None,
                    "best_continuous_test_metrics": None,
                    "best_final_reserves": None,
                    "best_final_weights": None,

                    # Selection info
                    "selection_method": "best_train",
                    "selection_metric": run_fingerprint.get("return_val", "sharpe"),
                    "best_param_idx": 0,

                    # Legacy fields (for backward compat)
                    "final_objective": float("-inf"),
                    "final_train_metrics": None,
                    "final_continuous_test_metrics": None,
                    "final_reserves": None,
                    "final_weights": None,

                    # Provenance
                    "run_location": run_location,
                    "run_fingerprint": deepcopy(run_fingerprint),
                    "checkpoint_returns": None,
                }
            return None
    else:
        raise NotImplementedError


def do_run_on_historic_data(
    run_fingerprint,
    params={},
    root=None,
    price_data=None,
    verbose=False,
    raw_trades=None,
    fees=None,
    gas_cost=None,
    arb_fees=None,
    fees_df=None,
    gas_cost_df=None,
    arb_fees_df=None,
    lp_supply_df=None,
    do_test_period=False,
    low_data_mode=False,
    preslice_burnin=True,
):
    """Execute a forward-pass simulation with fixed parameters.

    Runs the full simulation pipeline — price loading, weight calculation,
    arbitrage, and metric computation — using pre-trained (or manually
    specified) strategy parameters.  This is the primary entry point for
    post-training evaluation, walk-forward OOS testing, and visualisation.

    Parameters
    ----------
    run_fingerprint : dict
        Master configuration dict (same structure as
        :func:`train_on_historic_data`).
    params : dict or list of dict
        Strategy parameters.  A single dict runs one simulation; a list
        of dicts runs multiple parameter sets in parallel via ``vmap``.
    root : str, optional
        Root directory for data files.
    price_data : array-like or DataFrame, optional
        Pre-loaded price data.  When None, loaded from parquet files.
    verbose : bool, optional
        Print progress information (default False).
    raw_trades : DataFrame, optional
        Real trade data to inject.  Columns: unix timestamp (minute),
        token_in, token_out, amount_in.
    fees : float, optional
        Swap fee override (e.g. 0.003 for 30 bps).
    gas_cost : float, optional
        Gas cost override per transaction.
    arb_fees : float, optional
        Arbitrageur fee override.
    fees_df : DataFrame, optional
        Time-varying swap fees (columns: unix, fee).
    gas_cost_df : DataFrame, optional
        Time-varying gas costs (columns: unix, gas_cost).
    arb_fees_df : DataFrame, optional
        Time-varying arb fees (columns: unix, arb_fee).
    lp_supply_df : DataFrame, optional
        Time-varying LP supply changes.
    do_test_period : bool, optional
        If True, also run the OOS test period defined by
        ``endDateString`` to ``endTestDateString`` (default False).
    low_data_mode : bool, optional
        If True, drop raw price arrays from the output dict to reduce
        memory usage (default False).
    preslice_burnin : bool, optional
        If True, pre-slice data to ``max_memory_days`` of burn-in plus
        the simulation period (default True).  Set False to load all
        available history.

    Returns
    -------
    dict or tuple[dict, dict]
        When ``do_test_period=False``: a single results dict with keys
        including ``values``, ``reserves``, ``weights``,
        ``coarse_weights``, ``objective``, and per-asset breakdowns.

        When ``do_test_period=True``: ``(train_results, test_results)``.

        For multiple parameter sets, each value in the dict is a list
        (one entry per parameter set).
    """

    # Set default values for run_fingerprint and its optimisation_settings
    recursive_default_set(run_fingerprint, run_fingerprint_defaults)
    # Extract various settings from run_fingerprint
    chunk_period = run_fingerprint["chunk_period"]
    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    use_alt_lamb = run_fingerprint["use_alt_lamb"]
    use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    arb_frequency = run_fingerprint["arb_frequency"]
    rule = run_fingerprint["rule"]

    # Create a list of unique tokens
    unique_tokens = get_unique_tokens(run_fingerprint)

    n_tokens = len(run_fingerprint["tokens"])
    n_assets = n_tokens

    # Generate all possible signature variations
    all_sig_variations = get_sig_variations(n_assets)

    max_memory_days = run_fingerprint["max_memory_days"]

    np.random.seed(0)

    dynamic_inputs_dict = get_trades_and_fees(
        run_fingerprint,
        raw_trades,
        fees_df,
        gas_cost_df,
        arb_fees_df,
        lp_supply_df,
        do_test_period=do_test_period,
    )

    # Load price data if not provided
    if price_data is None:
        if verbose:
            print("loading data")
    data_dict = get_data_dict(
        unique_tokens,
        run_fingerprint,
        data_kind=run_fingerprint["optimisation_settings"]["training_data_kind"],
        root=root,
        max_memory_days=max_memory_days,
        start_date_string=run_fingerprint["startDateString"],
        end_time_string=run_fingerprint["endDateString"],
        start_time_test_string=run_fingerprint["endDateString"],
        end_time_test_string=run_fingerprint["endTestDateString"],
        max_mc_version=run_fingerprint["optimisation_settings"]["max_mc_version"],
        price_data=price_data,
        do_test_period=do_test_period,
        preslice_burnin=preslice_burnin,
    )
    max_memory_days = data_dict["max_memory_days"]
    if verbose:
        print("max_memory_days: ", max_memory_days)

    if run_fingerprint["optimisation_settings"]["training_data_kind"] == "mc":
        # TODO: Handle MC data for post-training analysis
        raise NotImplementedError

    # create pool
    pool = create_pool(rule)

    # Create static dict using helper - with run-specific overrides
    base_static_dict = create_static_dict(
        run_fingerprint,
        bout_length=data_dict["bout_length"],
        all_sig_variations=all_sig_variations,
        overrides={
            "n_assets": n_assets,
            "training_data_kind": run_fingerprint["optimisation_settings"]["training_data_kind"],
            # Override fees if provided as function args
            "fees": fees if fees is not None else run_fingerprint["fees"],
            "arb_fees": arb_fees if arb_fees is not None else run_fingerprint["arb_fees"],
            "gas_cost": gas_cost if gas_cost is not None else run_fingerprint["gas_cost"],
            "do_trades": False if raw_trades is None else run_fingerprint["do_trades"],
            # Include date strings for run-time use
            "startDateString": run_fingerprint["startDateString"],
            "endDateString": run_fingerprint["endDateString"],
            "endTestDateString": run_fingerprint["endTestDateString"],
        },
    )

    # Create static dictionaries for training and testing
    reserves_values_train_static_dict = base_static_dict.copy()
    reserves_values_train_static_dict["return_val"] = "reserves_and_values"
    reserves_values_train_static_dict["bout_length"] = data_dict["bout_length"]
    partial_forward_pass_nograd_batch_reserves_values_train = jit(
        Partial(
            forward_pass_nograd,
            static_dict=Hashabledict(reserves_values_train_static_dict),
            pool=pool,
        )
    )

    if do_test_period:
        reserves_values_test_static_dict = base_static_dict.copy()
        reserves_values_test_static_dict["return_val"] = "reserves_and_values"
        reserves_values_test_static_dict["bout_length"] = data_dict["bout_length_test"]
        partial_forward_pass_nograd_batch_reserves_values_test = jit(
            Partial(
                forward_pass_nograd,
                static_dict=Hashabledict(reserves_values_test_static_dict),
                pool=pool,
            )
        )

    # Ensure params is a list
    if isinstance(params, dict):
        params = [params]

    total_params = len(params)
    update_every = max(
        math.floor(total_params / 10), 1
    )  # Update every 10% of the way through the number of param sets
    output_dicts = []
    if do_test_period:
        output_dicts_test = []

    # Process each set of parameters
    for i in range(total_params):
        param = params[i]
        if i % update_every == 0:
            if verbose:
                tqdm.write(f"Processed {i+1} out of {total_params} parameters.")

        # Run forward pass for training data
        output_dict = partial_forward_pass_nograd_batch_reserves_values_train(
            param,
            (data_dict["start_idx"], 0),
            data_dict["prices"],
            dynamic_inputs_dict["train_period_trades"],
            dynamic_inputs_dict["fees_array"],
            dynamic_inputs_dict["gas_cost_array"],
            dynamic_inputs_dict["arb_fees_array"],
        )
        if low_data_mode:
            output_dict["final_prices"] = output_dict["prices"][-1]
            output_dict["initial_reserves"] = output_dict["reserves"][0]
            output_dict["initial_prices"] = output_dict["prices"][0]
            del output_dict["prices"]
            del output_dict["reserves"]
            del output_dict["value"]
        output_dicts.append(output_dict)
        # Run forward pass for test data if required
        if do_test_period:
            output_dict_test = partial_forward_pass_nograd_batch_reserves_values_test(
                param,
                (data_dict["start_idx_test"], 0),
                data_dict["prices"],
                dynamic_inputs_dict["test_period_trades"],
                dynamic_inputs_dict["test_fees_array"],
                dynamic_inputs_dict["test_gas_cost_array"],
                dynamic_inputs_dict["test_arb_fees_array"],
            )
            if low_data_mode:
                output_dict_test["final_prices"] = output_dict_test["prices"][-1]
                output_dict_test["initial_reserves"] = output_dict_test["reserves"][0]
                output_dict_test["initial_prices"] = output_dict_test["prices"][0]
                del output_dict_test["prices"]
                del output_dict_test["reserves"]
                del output_dict_test["value"]
            output_dicts_test.append(output_dict_test)

    # out = partial_forward_pass_nograd_batch(
    #     params[0],
    #     (data_dict["start_idx"], 0),
    # )
    # raise Exception("stop")
    # If only one set of parameters, return as single dict instead of list
    if len(output_dicts) == 1:
        output_dicts = output_dicts[0]
        output_dicts["data_dict"] = data_dict
        if do_test_period:
            output_dicts_test = output_dicts_test[0]
    # Return results
    gc.collect()
    gc.collect()
    # Clear any cached JAX computations to free memory
    clear_caches()
    if do_test_period:
        return output_dicts, output_dicts_test
    else:
        return output_dicts

def do_run_on_historic_data_with_provided_coarse_weights(
    run_fingerprint,
    coarse_weights,
    params={},
    root=None,
    price_data=None,
    verbose=False,
    raw_trades=None,
    fees=None,
    gas_cost=None,
    arb_fees=None,
    fees_df=None,
    gas_cost_df=None,
    arb_fees_df=None,
    lp_supply_df=None,
    do_test_period=False,
    low_data_mode=False,
):
    """Execute a simulation using pre-computed coarse weights.

    Like :func:`do_run_on_historic_data`, but bypasses the weight-calculation
    step entirely.  The caller provides ``coarse_weights`` directly, and
    this function performs only fine-weight interpolation, arbitrage
    simulation, and metric computation.

    This is useful for replaying a trained strategy with externally-computed
    or manually-specified weight trajectories, or for separating the weight
    computation from the simulation for profiling or debugging.

    Parameters
    ----------
    run_fingerprint : dict
        Master configuration dict.
    coarse_weights : jnp.ndarray
        Pre-computed coarse weights, shape ``(n_coarse_steps, n_assets)``.
    params : dict or list of dict, optional
        Strategy parameters (used only for ``initial_reserves`` and any
        subsidiary parameters, not for weight computation).
    root : str, optional
        Root directory for data files.
    price_data : array-like or DataFrame, optional
        Pre-loaded price data.
    verbose : bool, optional
        Print progress (default False).
    raw_trades : DataFrame, optional
        Real trade data to inject.
    fees : float, optional
        Swap fee override.
    gas_cost : float, optional
        Gas cost override.
    arb_fees : float, optional
        Arbitrageur fee override.
    fees_df : DataFrame, optional
        Time-varying swap fees.
    gas_cost_df : DataFrame, optional
        Time-varying gas costs.
    arb_fees_df : DataFrame, optional
        Time-varying arb fees.
    lp_supply_df : DataFrame, optional
        Time-varying LP supply changes.
    do_test_period : bool, optional
        Run OOS test period (default False).
    low_data_mode : bool, optional
        Drop raw arrays from output to save memory (default False).

    Returns
    -------
    dict or tuple[dict, dict]
        Same structure as :func:`do_run_on_historic_data`.
    """
    from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
        _jax_calc_coarse_weights,
        _jax_fine_weights_from_actual_starts_and_diffs,
    )
    from quantammsim.pools.G3M.quantamm.quantamm_reserves import (
        _jax_calc_quantAMM_reserves_with_dynamic_inputs,
    )

    # Set default values for run_fingerprint and its optimisation_settings
    recursive_default_set(run_fingerprint, run_fingerprint_defaults)
    # Extract various settings from run_fingerprint
    chunk_period = run_fingerprint["chunk_period"]
    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    use_alt_lamb = run_fingerprint["use_alt_lamb"]
    use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    arb_frequency = run_fingerprint["arb_frequency"]
    rule = run_fingerprint["rule"]

    # Create a list of unique tokens
    unique_tokens = get_unique_tokens(run_fingerprint)

    n_tokens = len(run_fingerprint["tokens"])
    n_assets = n_tokens

    # Generate all possible signature variations
    all_sig_variations = get_sig_variations(n_assets)

    max_memory_days = run_fingerprint["max_memory_days"]

    np.random.seed(0)

    dynamic_inputs_dict = get_trades_and_fees(
        run_fingerprint,
        raw_trades,
        fees_df,
        gas_cost_df,
        arb_fees_df,
        lp_supply_df,
        do_test_period=do_test_period,
    )

    # Load price data if not provided
    if price_data is None:
        if verbose:
            print("loading data")
    data_dict = get_data_dict(
        unique_tokens,
        run_fingerprint,
        data_kind=run_fingerprint["optimisation_settings"]["training_data_kind"],
        root=root,
        max_memory_days=max_memory_days,
        start_date_string=run_fingerprint["startDateString"],
        end_time_string=run_fingerprint["endDateString"],
        start_time_test_string=run_fingerprint["endDateString"],
        end_time_test_string=run_fingerprint["endTestDateString"],
        max_mc_version=run_fingerprint["optimisation_settings"]["max_mc_version"],
        price_data=price_data,
        do_test_period=do_test_period,
    )

    max_memory_days = data_dict["max_memory_days"]
    if verbose:
        print("max_memory_days: ", max_memory_days)

    if run_fingerprint["optimisation_settings"]["training_data_kind"] == "mc":
        # TODO: Handle MC data for post-training analysis
        raise NotImplementedError

    # create pool
    pool = create_pool(rule)

    # Create static dict using helper - with run-specific overrides
    base_static_dict = create_static_dict(
        run_fingerprint,
        bout_length=data_dict["bout_length"],
        all_sig_variations=all_sig_variations,
        overrides={
            "n_assets": n_assets,
            "training_data_kind": run_fingerprint["optimisation_settings"]["training_data_kind"],
            # Override fees if provided as function args
            "fees": fees if fees is not None else run_fingerprint["fees"],
            "arb_fees": arb_fees if arb_fees is not None else run_fingerprint["arb_fees"],
            "gas_cost": gas_cost if gas_cost is not None else run_fingerprint["gas_cost"],
            "do_trades": False if raw_trades is None else run_fingerprint["do_trades"],
            # Include date strings for run-time use
            "startDateString": run_fingerprint["startDateString"],
            "endDateString": run_fingerprint["endDateString"],
            "endTestDateString": run_fingerprint["endTestDateString"],
        },
    )

    # Create static dictionaries for training and testing
    static_dict = base_static_dict.copy()
    static_dict["return_val"] = "reserves_and_values"
    static_dict["bout_length"] = data_dict["bout_length"]

    training_data_kind = static_dict["training_data_kind"]
    minimum_weight = static_dict.get("minimum_weight")
    n_assets = static_dict["n_assets"]
    return_val = static_dict["return_val"]
    bout_length = static_dict["bout_length"]

    # filter coarse weights using the start and end indices
    coarse_weights = filter_coarse_weights_by_data_indices(coarse_weights, data_dict)
    # take coarse weights and convert to array of fine weights
    initial_weights = coarse_weights["weights"][0]
    # Repeat the last row of coarse weights
    coarse_weights_padded = jnp.vstack(
        [coarse_weights["weights"], coarse_weights["weights"][-1]]
    )
    coarse_weight_changes = jnp.diff(coarse_weights_padded, axis=0)
    actual_starts_cpu, scaled_diffs_cpu, target_weights_cpu = _jax_calc_coarse_weights(
        coarse_weight_changes,
        initial_weights,
        minimum_weight,
        params,
        run_fingerprint["max_memory_days"],
        chunk_period,
        chunk_period,
        1.0,
        False,
    )

    weights = _jax_fine_weights_from_actual_starts_and_diffs(
        actual_starts_cpu,
        scaled_diffs_cpu,
        initial_weights,
        interpol_num=chunk_period + 1,
        num=chunk_period + 1,
        maximum_change=1.0,
        method="linear",
    )
    # undo padding
    weights = weights[: (-1 * chunk_period + 1)]

    # Check that weights[::chunk_period] matches coarse_weights["weights"]
    # Get weights at coarse timesteps
    coarse_timestep_weights = weights[::chunk_period]
    weights = weights[:-1]

    # Compare with original coarse weights
    weights_match = jnp.allclose(
        coarse_timestep_weights, coarse_weights["weights"], rtol=1e-10
    )

    start_index = data_dict["start_idx"]
    end_index = data_dict["end_idx"] - 1

    local_prices = data_dict["prices"][start_index:end_index]
    local_unix_values = data_dict["unix_values"][start_index:end_index]

    # reserves = pool.calculate_reserves_with_fees(
    #     params,
    #     NestedHashabledict(static_dict),
    #     data_dict["prices"],
    #     start_index=None,
    #     local_prices=HashableArrayWrapper(local_prices),
    #     weights=HashableArrayWrapper(weights),
    #     initial_reserves=HashableArrayWrapper(params["initial_reserves"]),
    # )
    fees_array = dynamic_inputs_dict.get("fees_array")
    arb_thresh_array = dynamic_inputs_dict.get("gas_cost_array")
    arb_fees_array = dynamic_inputs_dict.get("arb_fees_array")
    trade_array = dynamic_inputs_dict.get("trades")
    lp_supply_array = dynamic_inputs_dict.get("lp_supply_array")

    if fees_array is None:
        fees_array = jnp.array([static_dict["fees"]])
    if arb_thresh_array is None:
        arb_thresh_array = jnp.array([static_dict["gas_cost"]])
    if arb_fees_array is None:
        arb_fees_array = jnp.array([static_dict["arb_fees"]])

        # initial_pool_value = run_fingerprint["initial_pool_value"]
        # initial_value_per_token = arb_acted_upon_weights[0] * initial_pool_value
        # initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]

    initial_reserves = params["initial_reserves"]

    # any of fees_array, arb_thresh_array, arb_fees_array, trade_array, and lp_supply_array
    # can be singletons, in which case we repeat them for the length of the bout.

    # Determine the maximum leading dimension
    max_len = bout_length - 1

    if run_fingerprint["arb_frequency"] != 1:
        max_len = max_len // run_fingerprint["arb_frequency"]

    fees_array = fees_array[:max_len]
    arb_thresh_array = arb_thresh_array[:max_len]
    arb_thresh_array = arb_thresh_array * 0.0
    arb_fees_array = arb_fees_array[:max_len]
    if lp_supply_array is not None:
        lp_supply_array = lp_supply_array[:max_len]
    if trade_array is not None:
        trade_array = trade_array[:max_len]
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
        weights,
        local_prices,
        fees_array_broadcast,
        arb_thresh_array_broadcast,
        arb_fees_array_broadcast,
        jnp.array(static_dict["all_sig_variations"]),
        None,
        run_fingerprint["do_trades"],
        run_fingerprint["do_arb"],
        run_fingerprint["noise_trader_ratio"],
        lp_supply_array_broadcast,
    )

    value_over_time = jnp.sum(jnp.multiply(reserves, local_prices), axis=-1)
    return_dict = {
        "final_reserves": reserves[-1],
        "final_value": (reserves[-1] * local_prices[-1]).sum(),
        "value": value_over_time,
        "prices": local_prices,
        "reserves": reserves,
        "weights": weights,
        "coarse_weight_changes": coarse_weight_changes,
        "data_dict": data_dict,
        "unix_values": local_unix_values,
    }
    return return_dict
