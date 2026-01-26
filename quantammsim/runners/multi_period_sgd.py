"""
Multi-Period SGD Training for Financial Strategies

This module implements multi-period robust training where we optimize
parameters across multiple temporal windows simultaneously with a single
forward pass and continuous pool state.

Key Design:
- ONE forward pass spanning the entire data period
- Dynamic slice out evaluation windows for each "period"
- Aggregate losses across periods -> single backward pass
- Pool state continuity is automatic (one continuous simulation)

This is NOT walk-forward (no retraining per period), but rather finds
ONE set of params that performs well across all temporal windows.

Benefits:
- Automatic pool state continuity through continuous forward pass
- Single JIT compilation (no recompilation for different bout lengths)
- Efficient: one forward pass, one backward pass per update step
- Encourages robust parameters that work across market regimes
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.lax import dynamic_slice
from jax.tree_util import Partial
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from copy import deepcopy
from itertools import product

from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set
from quantammsim.runners.jax_runner_utils import (
    Hashabledict,
    get_unique_tokens,
    create_static_dict,
)
from quantammsim.runners.jax_runners import nan_param_reinit
from jax.nn import softmax
from quantammsim.training.backpropagation import (
    update_from_partial_training_step_factory_with_optax,
    create_optimizer_chain,
)
from quantammsim.utils.post_train_analysis import calculate_period_metrics
from quantammsim.utils.data_processing.historic_data_utils import get_data_dict
from quantammsim.pools.creator import create_pool
from quantammsim.core_simulator.forward_pass import (
    forward_pass,
    forward_pass_nograd,
    _calculate_return_value,
)


@dataclass
class PeriodSpec:
    """Specification for a single evaluation period within the forward pass."""
    period_id: int
    rel_start: int  # Start index relative to forward pass output
    rel_end: int    # End index relative to forward pass output

    @property
    def length(self) -> int:
        return self.rel_end - self.rel_start


@dataclass
class MultiPeriodResult:
    """Results from multi-period training."""
    period_sharpes: List[float]
    period_returns: List[float]
    period_returns_over_hodl: List[float]
    mean_sharpe: float
    std_sharpe: float
    worst_sharpe: float
    mean_returns_over_hodl: float
    epochs_trained: int
    final_objective: float
    best_params: Dict[str, Any] = field(default_factory=dict)


def create_multi_period_training_step(
    base_forward_pass,
    prices: jnp.ndarray,
    period_specs: Tuple[Tuple[int, int], ...],
    n_assets: int,
    return_val: str,
    aggregation: str = "mean",
    softmin_temperature: float = 1.0,
):
    """
    Create a training step function that computes aggregate loss across periods.

    This returns a function with signature (params, start_index) -> scalar,
    compatible with the existing backpropagation factories.

    Parameters
    ----------
    base_forward_pass : callable
        Partial forward_pass with full bout_length static_dict
    prices : jnp.ndarray
        Full price array
    period_specs : tuple of (rel_start, slice_len)
        For each period: relative start index and length within forward pass output.
        Must be tuple of tuples (static) so loop unrolls at trace time.
    n_assets : int
        Number of assets
    return_val : str
        Metric to compute per period ("sharpe", "returns", etc.)
    aggregation : str
        How to combine period metrics:
        - "mean": Simple average (all periods contribute equally)
        - "min": Hard minimum (CAUTION: only minimum element gets gradients)
        - "softmin": Soft minimum via negative softmax (recommended for worst-case)
        - "sum": Sum of all metrics
    softmin_temperature : float
        Temperature for softmin aggregation. Lower = closer to hard min.
        Default 1.0 gives moderate smoothing. Use 0.1-0.5 for sharper focus on worst.

    Returns
    -------
    callable
        Function (params, start_index) -> scalar

    Notes
    -----
    IMPORTANT: Using aggregation="min" has a gradient flow problem!

    With hard min, gradients only flow through the single minimum element.
    This means:
    - Only 1 of N periods contributes to parameter updates
    - Gradients are sparse and noisy
    - Training can be unstable

    Solution: Use "softmin" instead, which computes a soft minimum:
        softmin(x) = sum(x * softmax(-x / temperature))

    This gives more weight to lower-performing periods while still allowing
    gradients to flow from all periods. As temperature → 0, softmin → hard min.
    """
    def multi_period_training_step(params, start_index):
        # One forward pass for entire bout
        output = base_forward_pass(params, start_index)

        full_value = output["value"]
        full_reserves = output["reserves"]
        time_idx = start_index[0]

        # Compute metric for each period (loop unrolls at trace time)
        period_metrics = []
        for (rel_start, slice_len) in period_specs:
            sliced_value = dynamic_slice(full_value, (rel_start,), (slice_len,))
            sliced_reserves = dynamic_slice(
                full_reserves, (rel_start, 0), (slice_len, n_assets)
            )
            sliced_prices = dynamic_slice(
                prices, (time_idx + rel_start, 0), (slice_len, n_assets)
            )

            metric = _calculate_return_value(
                return_val,
                sliced_reserves,
                sliced_prices,
                sliced_value,
                initial_reserves=sliced_reserves[0],
            )
            period_metrics.append(metric)

        stacked = jnp.stack(period_metrics)

        if aggregation == "mean":
            return jnp.mean(stacked)
        elif aggregation == "min":
            # WARNING: Hard min only passes gradients through minimum element!
            return jnp.min(stacked)
        elif aggregation == "softmin":
            # Soft minimum: weighted average with weights from softmax(-x/temp)
            # This gives more weight to lower values while maintaining gradient flow
            weights = softmax(-stacked / softmin_temperature)
            return jnp.sum(stacked * weights)
        elif aggregation == "sum":
            return jnp.sum(stacked)
        else:
            return jnp.mean(stacked)

    return multi_period_training_step


def generate_period_specs(
    n_periods: int,
    total_length: int,
    overlap_fraction: float = 0.0,
) -> List[PeriodSpec]:
    """Generate period specifications for multi-period training."""
    if overlap_fraction > 0:
        base_period_len = total_length // n_periods
        overlap = int(base_period_len * overlap_fraction)
        effective_step = base_period_len - overlap

        specs = []
        for i in range(n_periods):
            rel_start = i * effective_step
            rel_end = min(rel_start + base_period_len, total_length)
            specs.append(PeriodSpec(period_id=i, rel_start=rel_start, rel_end=rel_end))
    else:
        period_len = total_length // n_periods
        specs = []
        for i in range(n_periods):
            rel_start = i * period_len
            rel_end = (i + 1) * period_len if i < n_periods - 1 else total_length
            specs.append(PeriodSpec(period_id=i, rel_start=rel_start, rel_end=rel_end))

    return specs


def multi_period_sgd_training(
    run_fingerprint: dict,
    n_periods: int = 4,
    overlap_fraction: float = 0.0,
    max_epochs: int = 500,
    aggregation: str = "mean",
    softmin_temperature: float = 1.0,
    verbose: bool = True,
) -> Tuple[MultiPeriodResult, dict]:
    """
    Run multi-period SGD training.

    Trains ONE set of parameters that performs well across multiple
    temporal windows simultaneously.

    Parameters
    ----------
    run_fingerprint : dict
        Run configuration
    n_periods : int
        Number of evaluation periods
    overlap_fraction : float
        Fraction of overlap between periods (0.0 = no overlap)
    max_epochs : int
        Maximum training epochs
    aggregation : str
        How to combine period metrics:
        - "mean": Simple average (default, all periods equal)
        - "softmin": Soft minimum (recommended for worst-case optimization)
        - "min": Hard minimum (NOT recommended - gradient flow issues)
        - "sum": Sum of metrics
    softmin_temperature : float
        Temperature for softmin aggregation. Lower = closer to hard min.
        Default 1.0. Use 0.1-0.5 for sharper worst-case focus.
    verbose : bool
        Print progress

    Returns
    -------
    Tuple[MultiPeriodResult, dict]
        Training result and summary statistics
    """
    recursive_default_set(run_fingerprint, run_fingerprint_defaults)

    if verbose:
        print("=" * 70)
        print("MULTI-PERIOD SGD TRAINING")
        print("=" * 70)
        print(f"Periods: {n_periods}")
        print(f"Overlap: {overlap_fraction:.1%}")
        print(f"Aggregation: {aggregation}", end="")
        if aggregation == "softmin":
            print(f" (temperature={softmin_temperature})")
        elif aggregation == "min":
            print(" (WARNING: gradient flow issues - consider 'softmin' instead)")
        else:
            print()
        print("=" * 70)

    # Setup
    unique_tokens = get_unique_tokens(run_fingerprint)
    n_assets = len(unique_tokens)

    all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    all_sig_variations = all_sig_variations[(all_sig_variations == 1).sum(-1) == 1]
    all_sig_variations = all_sig_variations[(all_sig_variations == -1).sum(-1) == 1]
    all_sig_variations = tuple(map(tuple, all_sig_variations))

    pool = create_pool(run_fingerprint["rule"])
    assert pool.is_trainable(), "Pool must be trainable"

    initial_params = {
        "initial_memory_length": run_fingerprint["initial_memory_length"],
        "initial_memory_length_delta": run_fingerprint["initial_memory_length_delta"],
        "initial_k_per_day": run_fingerprint["initial_k_per_day"],
        "initial_weights_logits": run_fingerprint["initial_weights_logits"],
        "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
        "initial_raw_width": run_fingerprint["initial_raw_width"],
        "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
        "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
    }

    if verbose:
        print(f"\nLoading data...")

    data_dict = get_data_dict(
        unique_tokens,
        run_fingerprint,
        data_kind=run_fingerprint["optimisation_settings"]["training_data_kind"],
        max_memory_days=run_fingerprint["max_memory_days"],
        start_date_string=run_fingerprint["startDateString"],
        end_time_string=run_fingerprint["endDateString"],
        do_test_period=False,
    )

    bout_length = data_dict["end_idx"] - data_dict["start_idx"]
    output_length = bout_length - 1

    if verbose:
        print(f"Data loaded: {data_dict['prices'].shape[0]} timesteps")
        print(f"Training bout_length: {bout_length}")

    # Generate period specifications
    period_specs = generate_period_specs(n_periods, output_length, overlap_fraction)

    if verbose:
        print(f"\nPeriod breakdown:")
        for spec in period_specs:
            print(f"  Period {spec.period_id}: [{spec.rel_start}:{spec.rel_end}] (len={spec.length})")

    # Convert to tuple of tuples for static handling
    period_specs_tuple = tuple((spec.rel_start, spec.length) for spec in period_specs)

    # Create static dict with full bout_length
    static_dict = create_static_dict(
        run_fingerprint,
        bout_length,
        all_sig_variations,
        overrides={
            "n_assets": n_assets,
            "return_val": "reserves_and_values",
            "training_data_kind": run_fingerprint["optimisation_settings"]["training_data_kind"],
        }
    )

    # Create base forward pass
    base_forward_pass = Partial(
        forward_pass,
        prices=data_dict["prices"],
        static_dict=Hashabledict(static_dict),
        pool=pool,
    )

    # Create multi-period training step
    partial_training_step = create_multi_period_training_step(
        base_forward_pass,
        data_dict["prices"],
        period_specs_tuple,
        n_assets,
        run_fingerprint["return_val"],
        aggregation,
        softmin_temperature,
    )

    # Initialize params
    n_parameter_sets = 1
    params = pool.init_parameters(
        initial_params, run_fingerprint, n_assets, n_parameter_sets
    )
    params = {k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
              for k, v in params.items()}

    if verbose:
        print("\nParam shapes:")
        for k, v in params.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")

    # Create optimizer and update function using existing factory
    optimizer = create_optimizer_chain(run_fingerprint)
    opt_state = optimizer.init(params)

    # Use existing factory - it handles batching, gradients, optimizer application
    update_fn = update_from_partial_training_step_factory_with_optax(
        partial_training_step,
        optimizer,
        run_fingerprint["optimisation_settings"]["train_on_hessian_trace"],
        Partial(partial_training_step, start_index=(data_dict["start_idx"], 0)),
    )

    # Training loop
    best_objective = -np.inf
    best_params = deepcopy(params)
    start_indexes = jnp.array([[data_dict["start_idx"], 0]])  # Batch of 1
    local_lr = run_fingerprint["optimisation_settings"]["base_lr"]

    for epoch in range(max_epochs):
        params, objective_value, old_params, grads, opt_state = update_fn(
            params, start_indexes, local_lr, opt_state
        )

        # Handle NaN gradients
        params = nan_param_reinit(
            params, grads, pool, initial_params,
            run_fingerprint, n_assets, n_parameter_sets
        )

        objective_value = float(objective_value)
        if objective_value > best_objective:
            best_objective = objective_value
            best_params = deepcopy(params)

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}: objective={objective_value:.4f}")

    epochs_trained = epoch + 1
    params = best_params

    # Final evaluation
    if verbose:
        print("\nFinal evaluation...")

    partial_nograd = jit(Partial(
        forward_pass_nograd,
        prices=data_dict["prices"],
        static_dict=Hashabledict(static_dict),
        pool=pool,
    ))

    output = partial_nograd(params, (data_dict["start_idx"], 0))

    period_sharpes = []
    period_returns = []
    period_returns_over_hodl = []

    for spec in period_specs:
        period_value = output["value"][spec.rel_start:spec.rel_end]
        period_reserves = output["reserves"][spec.rel_start:spec.rel_end]
        period_prices = data_dict["prices"][
            data_dict["start_idx"] + spec.rel_start:
            data_dict["start_idx"] + spec.rel_end
        ]

        metrics = calculate_period_metrics(
            {"value": period_value, "reserves": period_reserves},
            period_prices
        )

        period_sharpes.append(metrics["sharpe"])
        period_returns.append(metrics["return"])
        period_returns_over_hodl.append(metrics["returns_over_uniform_hodl"])

    if verbose:
        print("\nPer-period results:")
        for i, spec in enumerate(period_specs):
            print(f"  Period {spec.period_id}: sharpe={period_sharpes[i]:.4f}, "
                  f"ret_over_hodl={period_returns_over_hodl[i]:.4f}")

    result = MultiPeriodResult(
        period_sharpes=period_sharpes,
        period_returns=period_returns,
        period_returns_over_hodl=period_returns_over_hodl,
        mean_sharpe=np.mean(period_sharpes),
        std_sharpe=np.std(period_sharpes),
        worst_sharpe=np.min(period_sharpes),
        mean_returns_over_hodl=np.mean(period_returns_over_hodl),
        epochs_trained=epochs_trained,
        final_objective=best_objective,
        best_params={k: np.array(v) for k, v in params.items()},
    )

    summary = {
        "n_periods": n_periods,
        "aggregation": aggregation,
        "softmin_temperature": softmin_temperature if aggregation == "softmin" else None,
        "mean_sharpe": result.mean_sharpe,
        "std_sharpe": result.std_sharpe,
        "worst_sharpe": result.worst_sharpe,
        "mean_returns_over_hodl": result.mean_returns_over_hodl,
        "epochs_trained": epochs_trained,
        "final_objective": best_objective,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("MULTI-PERIOD SUMMARY")
        print("=" * 70)
        print(f"Mean Sharpe:     {summary['mean_sharpe']:.4f} +/- {summary['std_sharpe']:.4f}")
        print(f"Worst Sharpe:    {summary['worst_sharpe']:.4f}")
        print(f"Mean Ret/Hodl:   {summary['mean_returns_over_hodl']:.4f}")
        print(f"Epochs:          {summary['epochs_trained']}")
        print(f"Final Objective: {summary['final_objective']:.4f}")
        print("=" * 70)

    return result, summary


if __name__ == "__main__":
    run_fingerprint = {
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-06-01 00:00:00",
        "tokens": ["BTC", "ETH"],
        "rule": "momentum",
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1000000.0,
        "fees": 0.003,
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "maximum_change": 0.001,
        "return_val": "sharpe",
        "optimisation_settings": {
            "n_parameter_sets": 1,
            "training_data_kind": "historic",
            "optimiser": "adam",
            "base_lr": 0.1,
            "decay_lr_plateau": 50,
            "decay_lr_ratio": 0.5,
            "min_lr": 1e-5,
            "initial_random_key": 42,
            "batch_size": 8,
            "sample_method": "uniform",
            "train_on_hessian_trace": False,
        },
    }

    # Example 1: Mean aggregation (default)
    result, summary = multi_period_sgd_training(
        run_fingerprint,
        n_periods=4,
        overlap_fraction=0.0,
        max_epochs=200,
        aggregation="mean",
        verbose=True,
    )

    # Example 2: Softmin for worst-case optimization (recommended over "min")
    # result, summary = multi_period_sgd_training(
    #     run_fingerprint,
    #     n_periods=4,
    #     overlap_fraction=0.0,
    #     max_epochs=200,
    #     aggregation="softmin",
    #     softmin_temperature=0.5,  # Lower = more focus on worst period
    #     verbose=True,
    # )
