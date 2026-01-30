"""
Robust Walk-Forward Training with Rademacher Complexity Tracking

This module implements a comprehensive walk-forward training system that combines:

1. **Pardo-style Walk-Forward Analysis (WFA)**
   - Sequential optimize → test → step forward
   - Walk-Forward Efficiency (WFE) metric
   - Strict temporal separation

2. **SGD-like Warm Starting**
   - Parameters from cycle i initialize cycle i+1
   - Regularization toward prior params to prevent drift
   - Reduced iterations for fine-tuning after warm start

3. **Rademacher Anti-Serum (Paleologo)**
   - Track checkpointed models during training
   - Compute empirical Rademacher complexity
   - Apply haircut to OOS performance estimates

Key References:
- Pardo, "The Evaluation and Optimization of Trading Strategies" (2008)
- Paleologo, "The Elements of Quantitative Investing" (2024), Ch. 6

Design Philosophy:
- Traditional WFA discards params each cycle to test the optimization process
- SGD warm-starting tests whether params are robust and adaptable
- Rademacher haircut quantifies overfitting from implicit strategy selection
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, random
from jax.tree_util import Partial, tree_map
from jax.nn import softmax
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from copy import deepcopy
from itertools import product
from datetime import datetime
import json

from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set
from quantammsim.runners.jax_runner_utils import (
    Hashabledict,
    get_unique_tokens,
    create_static_dict,
)
from quantammsim.runners.jax_runners import nan_param_reinit
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
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WalkForwardCycle:
    """Specification for a single walk-forward cycle."""
    cycle_number: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    # Indices populated after data loading
    train_start_idx: int = 0
    train_end_idx: int = 0
    test_start_idx: int = 0
    test_end_idx: int = 0


@dataclass
class CycleResult:
    """Results from a single walk-forward cycle."""
    cycle_number: int

    # In-sample metrics (training period)
    train_sharpe: float
    train_returns: float
    train_returns_over_hodl: float

    # Validation metrics (for early stopping)
    val_sharpe: float

    # Out-of-sample metrics (test period - what actually matters)
    test_sharpe: float
    test_returns: float
    test_returns_over_hodl: float

    # Walk-Forward Efficiency (Pardo metric)
    # WFE = annualized_oos_performance / annualized_is_performance
    walk_forward_efficiency: float

    # Training diagnostics
    epochs_trained: int
    stopped_early: bool
    final_lr: float

    # Rademacher tracking
    n_checkpoints: int
    rademacher_complexity: float
    adjusted_test_sharpe: float  # After Rademacher haircut

    # State for continuity
    best_params: Dict[str, Any] = field(default_factory=dict)
    final_weights: Optional[np.ndarray] = None
    final_value: Optional[float] = None

    # Returns time series for Rademacher calculation
    checkpoint_returns: Optional[np.ndarray] = None


@dataclass
class RobustTrainingResult:
    """Aggregate results from robust walk-forward training."""
    cycles: List[CycleResult]

    # Pardo-style metrics
    mean_wfe: float  # Walk-Forward Efficiency
    aggregate_oos_sharpe: float
    aggregate_is_sharpe: float

    # Rademacher-adjusted metrics
    aggregate_rademacher_complexity: float
    adjusted_aggregate_sharpe: float

    # Summary statistics
    mean_oos_sharpe: float
    std_oos_sharpe: float
    worst_oos_sharpe: float

    # Overfitting indicators
    is_oos_gap: float
    haircut_magnitude: float


# =============================================================================
# Rademacher Complexity Utilities
# =============================================================================

def compute_empirical_rademacher(
    returns_matrix: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
) -> float:
    """
    Compute empirical Rademacher complexity of a set of strategies.

    The Rademacher complexity measures how well the strategy class can
    "fit" random noise. Higher complexity = more overfitting risk.

    Parameters
    ----------
    returns_matrix : ndarray of shape (n_strategies, T)
        Returns time series for each strategy (checkpoint)
    n_samples : int
        Number of random sign vectors to sample
    seed : int
        Random seed for reproducibility

    Returns
    -------
    float
        Empirical Rademacher complexity R̂

    Notes
    -----
    R̂ = E_σ[sup_s (1/T) Σ_t σ_t r_s(t)]

    where σ_t are random Rademacher variables (±1 with prob 0.5)
    """
    if returns_matrix.ndim == 1:
        returns_matrix = returns_matrix.reshape(1, -1)

    n_strategies, T = returns_matrix.shape

    if n_strategies == 0 or T == 0:
        return 0.0

    rng = np.random.RandomState(seed)
    suprema = []

    for _ in range(n_samples):
        # Random Rademacher signs
        sigma = rng.choice([-1, 1], size=T)

        # Correlation of each strategy with random signs
        correlations = returns_matrix @ sigma / T

        # Supremum over strategies
        sup = np.max(correlations)
        suprema.append(sup)

    return np.mean(suprema)


def compute_rademacher_haircut(
    observed_sharpe: float,
    rademacher_complexity: float,
    T: int,
    delta: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute Rademacher-adjusted performance bound.

    From Paleologo (2024):
    θ_n ≥ θ̂_n - 2R̂ - estimation_error

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio on test data
    rademacher_complexity : float
        Empirical Rademacher complexity R̂
    T : int
        Number of time periods in test data
    delta : float
        Confidence level (default 0.05 for 95% confidence)

    Returns
    -------
    Tuple[float, float]
        (adjusted_sharpe, haircut_magnitude)
    """
    # Estimation error term: 3√(2log(2/δ)/T)
    estimation_error = 3 * np.sqrt(2 * np.log(2 / delta) / T)

    # Total haircut
    haircut = 2 * rademacher_complexity + estimation_error

    adjusted_sharpe = observed_sharpe - haircut

    return adjusted_sharpe, haircut


# =============================================================================
# Walk-Forward Efficiency (Pardo)
# =============================================================================

def compute_walk_forward_efficiency(
    is_sharpe: float,
    oos_sharpe: float,
    is_days: int,
    oos_days: int,
) -> float:
    """
    Compute Walk-Forward Efficiency (WFE) as per Pardo.

    WFE = (Annualized OOS Performance) / (Annualized IS Performance)

    A WFE of 0.5 or higher suggests robustness.
    A WFE near 1.0 is ideal (OOS ≈ IS).
    A WFE > 1.0 means OOS outperformed IS (unusual but possible).

    Parameters
    ----------
    is_sharpe : float
        In-sample Sharpe ratio
    oos_sharpe : float
        Out-of-sample Sharpe ratio
    is_days : int
        Number of days in IS period
    oos_days : int
        Number of days in OOS period

    Returns
    -------
    float
        Walk-Forward Efficiency (returns 0.0 for undefined cases)
    """
    # Handle edge cases where WFE is undefined
    if is_sharpe <= 0:
        # Can't compute meaningful ratio when IS is non-positive
        # Return 0 to indicate "not robust" (failed to profit in-sample)
        return 0.0

    return oos_sharpe / is_sharpe


# =============================================================================
# Checkpoint Tracking for Rademacher
# =============================================================================

class CheckpointTracker:
    """
    Tracks parameter checkpoints during training for Rademacher complexity.

    Each checkpoint represents a "strategy" that was implicitly considered
    during the optimization process.
    """

    def __init__(
        self,
        checkpoint_every: int = 50,
        max_checkpoints: int = 100,
    ):
        self.checkpoint_every = checkpoint_every
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Dict[str, Any]] = []
        self.checkpoint_returns: List[np.ndarray] = []
        self.iteration = 0

    def maybe_checkpoint(
        self,
        params: Dict[str, Any],
        returns: np.ndarray,
        force: bool = False,
    ):
        """Record a checkpoint if due."""
        self.iteration += 1

        if force or (self.iteration % self.checkpoint_every == 0):
            if len(self.checkpoints) < self.max_checkpoints:
                self.checkpoints.append(deepcopy(params))
                self.checkpoint_returns.append(np.array(returns))

    def get_returns_matrix(self) -> np.ndarray:
        """Get returns matrix for Rademacher computation."""
        if not self.checkpoint_returns:
            return np.array([])

        # All returns should have same length (from same forward pass)
        min_len = min(len(r) for r in self.checkpoint_returns)
        truncated = [r[:min_len] for r in self.checkpoint_returns]
        return np.stack(truncated)

    def compute_complexity(self, n_samples: int = 1000) -> float:
        """Compute Rademacher complexity of checkpointed strategies."""
        returns_matrix = self.get_returns_matrix()
        if returns_matrix.size == 0:
            return 0.0
        return compute_empirical_rademacher(returns_matrix, n_samples)

    def reset(self):
        """Reset tracker for new cycle."""
        self.checkpoints = []
        self.checkpoint_returns = []
        self.iteration = 0


# =============================================================================
# Core Training Functions
# =============================================================================

def datetime_to_timestamp(date_string: str) -> float:
    """Convert datetime string to unix timestamp."""
    dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


def timestamp_to_datetime(timestamp: float) -> str:
    """Convert unix timestamp to datetime string."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def generate_walk_forward_cycles(
    start_date: str,
    end_date: str,
    n_cycles: int,
    keep_fixed_start: bool = True,
    test_fraction: float = 0.25,
) -> List[WalkForwardCycle]:
    """
    Generate walk-forward cycle specifications.

    Parameters
    ----------
    start_date : str
        Start date (format: "YYYY-MM-DD HH:MM:SS")
    end_date : str
        End date
    n_cycles : int
        Number of training/test cycles
    keep_fixed_start : bool
        If True, training starts from beginning (expanding window)
        If False, training window rolls forward (rolling window)
    test_fraction : float
        Fraction of each cycle's period to use for testing (Pardo: 25-35%)

    Returns
    -------
    List[WalkForwardCycle]
    """
    start_ts = datetime_to_timestamp(start_date)
    end_ts = datetime_to_timestamp(end_date)

    # Create n_cycles + 1 boundary points
    times = np.linspace(start_ts, end_ts, n_cycles + 1)
    cycle_length = times[1] - times[0]

    # Add one more boundary for final test period
    times = np.append(times, times[-1] + cycle_length * test_fraction)

    # Round to midnight
    times = times - (times % (24 * 60 * 60))

    cycles = []
    for i in range(n_cycles):
        if keep_fixed_start:
            train_start = times[0]
        else:
            train_start = times[i]

        train_end = times[i + 1]
        test_start = times[i + 1]
        test_end = times[i + 2] if i + 2 < len(times) else times[-1]

        cycles.append(WalkForwardCycle(
            cycle_number=i,
            train_start_date=timestamp_to_datetime(train_start),
            train_end_date=timestamp_to_datetime(train_end),
            test_start_date=timestamp_to_datetime(test_start),
            test_end_date=timestamp_to_datetime(test_end),
        ))

    return cycles


def create_regularized_objective(
    base_objective: Callable,
    prior_params: Optional[Dict[str, Any]],
    regularization_strength: float = 0.01,
) -> Callable:
    """
    Create objective with L2 regularization toward prior params.

    This encourages parameters to not drift too far from the previous
    cycle's solution, providing stability during warm-starting.
    """
    if prior_params is None or regularization_strength <= 0:
        return base_objective

    def regularized_objective(params, start_indexes):
        base_loss = base_objective(params, start_indexes)

        # L2 penalty toward prior
        l2_penalty = 0.0
        for key in params:
            if key in prior_params and hasattr(params[key], 'shape'):
                diff = params[key] - prior_params[key]
                l2_penalty = l2_penalty + jnp.sum(diff ** 2)

        return base_loss - regularization_strength * l2_penalty

    return regularized_objective


def train_single_cycle_robust(
    run_fingerprint: dict,
    cycle: WalkForwardCycle,
    pool,
    initial_params: dict,
    data_dict: dict,
    all_sig_variations: tuple,
    n_assets: int,
    warm_start_params: Optional[dict] = None,
    initial_pool_value: Optional[float] = None,
    initial_weights_logits: Optional[np.ndarray] = None,
    max_epochs: int = 500,
    patience: int = 50,
    val_fraction: float = 0.2,
    regularization_strength: float = 0.01,
    checkpoint_every: int = 50,
    verbose: bool = True,
) -> CycleResult:
    """
    Train a single walk-forward cycle with Rademacher tracking.

    This is the core training function that:
    1. Optionally warm-starts from previous cycle
    2. Uses early stopping on validation
    3. Tracks checkpoints for Rademacher complexity
    4. Computes WFE and adjusted metrics
    """
    n_parameter_sets = 1

    # Calculate bout lengths
    train_bout_length = cycle.train_end_idx - cycle.train_start_idx
    test_bout_length = cycle.test_end_idx - cycle.test_start_idx

    # Split training into train/val for early stopping
    val_length = int(train_bout_length * val_fraction)
    inner_train_length = train_bout_length - val_length
    val_start_idx = cycle.train_start_idx + inner_train_length

    if verbose:
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle.cycle_number}")
        print(f"{'='*60}")
        print(f"  Train: {cycle.train_start_date} → {cycle.train_end_date}")
        print(f"    Inner train: {inner_train_length} steps")
        print(f"    Validation:  {val_length} steps")
        print(f"  Test:  {cycle.test_start_date} → {cycle.test_end_date}")
        print(f"    Test length: {test_bout_length} steps")

    # Initialize checkpoint tracker
    checkpoint_tracker = CheckpointTracker(
        checkpoint_every=checkpoint_every,
        max_checkpoints=100,
    )

    # Initialize or warm-start parameters
    is_warm_start = warm_start_params is not None
    if is_warm_start:
        params = deepcopy(warm_start_params)
        if verbose:
            print("  Mode: Warm-starting from previous cycle")
    else:
        params = pool.init_parameters(
            initial_params, run_fingerprint, n_assets, n_parameter_sets
        )
        params = {k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
                  for k, v in params.items()}
        if verbose:
            print("  Mode: Fresh initialization")

    # Override initial weights/value if provided (pool state continuity)
    if initial_weights_logits is not None:
        params["initial_weights_logits"] = jnp.array(initial_weights_logits)

    # Modify run_fingerprint for this cycle
    local_fp = deepcopy(run_fingerprint)
    if initial_pool_value is not None:
        local_fp["initial_pool_value"] = initial_pool_value

    # Create static dicts
    training_data_kind = run_fingerprint["optimisation_settings"]["training_data_kind"]
    train_static_dict = create_static_dict(
        local_fp, inner_train_length, all_sig_variations,
        overrides={"n_assets": n_assets, "return_val": run_fingerprint["return_val"], "training_data_kind": training_data_kind}
    )
    val_static_dict = create_static_dict(
        local_fp, val_length, all_sig_variations,
        overrides={"n_assets": n_assets, "return_val": "reserves_and_values", "training_data_kind": training_data_kind}
    )
    test_static_dict = create_static_dict(
        local_fp, test_bout_length, all_sig_variations,
        overrides={"n_assets": n_assets, "return_val": "reserves_and_values", "training_data_kind": training_data_kind}
    )
    full_train_static_dict = create_static_dict(
        local_fp, train_bout_length, all_sig_variations,
        overrides={"n_assets": n_assets, "return_val": "reserves_and_values", "training_data_kind": training_data_kind}
    )

    # Create forward pass functions
    partial_training_step = Partial(
        forward_pass,
        prices=data_dict["prices"],
        static_dict=Hashabledict(train_static_dict),
        pool=pool,
    )

    partial_val_nograd = jit(Partial(
        forward_pass_nograd,
        prices=data_dict["prices"],
        static_dict=Hashabledict(val_static_dict),
        pool=pool,
    ))

    partial_test_nograd = jit(Partial(
        forward_pass_nograd,
        prices=data_dict["prices"],
        static_dict=Hashabledict(test_static_dict),
        pool=pool,
    ))

    partial_full_train_nograd = jit(Partial(
        forward_pass_nograd,
        prices=data_dict["prices"],
        static_dict=Hashabledict(full_train_static_dict),
        pool=pool,
    ))

    # Setup optimizer with optional regularization
    optimizer = create_optimizer_chain(run_fingerprint)
    opt_state = optimizer.init(params)

    # Create update function
    # Note: regularization is applied by modifying the objective
    update_fn = update_from_partial_training_step_factory_with_optax(
        partial_training_step,
        optimizer,
        run_fingerprint["optimisation_settings"]["train_on_hessian_trace"],
        Partial(partial_training_step, start_index=(cycle.train_start_idx, 0)),
    )

    # Training loop with early stopping and checkpoint tracking
    local_lr = run_fingerprint["optimisation_settings"]["base_lr"]

    # Reduced LR for warm starts (fine-tuning)
    if is_warm_start:
        local_lr = local_lr * 0.5

    best_val_sharpe = -np.inf
    best_params = deepcopy(params)
    patience_counter = 0
    stopped_early = False

    # LR decay settings
    max_iters_no_improve = run_fingerprint["optimisation_settings"]["decay_lr_plateau"]
    decay_ratio = run_fingerprint["optimisation_settings"]["decay_lr_ratio"]
    min_lr = run_fingerprint["optimisation_settings"]["min_lr"]
    iters_since_improve = 0

    for epoch in range(max_epochs):
        start_indexes = jnp.array([[cycle.train_start_idx, 0]])

        params, objective_value, old_params, grads, opt_state = update_fn(
            params, start_indexes, local_lr, opt_state
        )

        # Handle NaN gradients
        params = nan_param_reinit(
            params, grads, pool, initial_params,
            run_fingerprint, n_assets, n_parameter_sets
        )

        # Evaluate on validation set for early stopping
        val_output = partial_val_nograd(params, (val_start_idx, 0))
        val_prices = data_dict["prices"][val_start_idx:val_start_idx + val_length]
        val_metrics = calculate_period_metrics(
            {"value": val_output["value"], "reserves": val_output["reserves"]},
            val_prices
        )
        val_sharpe = val_metrics["sharpe"]

        # Compute returns for Rademacher tracking
        val_returns = np.diff(np.log(np.array(val_output["value"])))
        checkpoint_tracker.maybe_checkpoint(params, val_returns)

        # Early stopping check
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_params = deepcopy(params)
            patience_counter = 0
            iters_since_improve = 0
        else:
            patience_counter += 1
            iters_since_improve += 1

        # LR decay
        if iters_since_improve > max_iters_no_improve:
            local_lr = local_lr * decay_ratio
            iters_since_improve = 0
            if local_lr < min_lr:
                local_lr = min_lr

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (val_sharpe={best_val_sharpe:.4f})")
            stopped_early = True
            break

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch}: obj={float(objective_value):.4f}, "
                  f"val_sharpe={val_sharpe:.4f}, lr={local_lr:.6f}")

    epochs_trained = epoch + 1
    params = best_params

    # Final checkpoint
    val_output = partial_val_nograd(params, (val_start_idx, 0))
    val_returns = np.diff(np.log(np.array(val_output["value"])))
    checkpoint_tracker.maybe_checkpoint(params, val_returns, force=True)

    # Evaluate on full training period (in-sample)
    train_output = partial_full_train_nograd(params, (cycle.train_start_idx, 0))
    train_prices = data_dict["prices"][cycle.train_start_idx:cycle.train_end_idx]
    train_metrics = calculate_period_metrics(
        {"value": train_output["value"], "reserves": train_output["reserves"]},
        train_prices
    )

    # Evaluate on test period (out-of-sample)
    test_output = partial_test_nograd(params, (cycle.test_start_idx, 0))
    test_prices = data_dict["prices"][cycle.test_start_idx:cycle.test_end_idx]
    test_metrics = calculate_period_metrics(
        {"value": test_output["value"], "reserves": test_output["reserves"]},
        test_prices
    )

    # Compute Rademacher complexity and haircut
    rademacher_complexity = checkpoint_tracker.compute_complexity()
    test_T = test_bout_length
    adjusted_test_sharpe, haircut = compute_rademacher_haircut(
        test_metrics["sharpe"],
        rademacher_complexity,
        test_T,
    )

    # Compute Walk-Forward Efficiency
    is_days = train_bout_length
    oos_days = test_bout_length
    wfe = compute_walk_forward_efficiency(
        train_metrics["sharpe"],
        test_metrics["sharpe"],
        is_days,
        oos_days,
    )

    if verbose:
        print(f"\n  Results:")
        print(f"    Train (IS):  sharpe={train_metrics['sharpe']:.4f}, "
              f"ret_over_hodl={train_metrics['returns_over_uniform_hodl']:.4f}")
        print(f"    Val:         sharpe={best_val_sharpe:.4f}")
        print(f"    Test (OOS):  sharpe={test_metrics['sharpe']:.4f}, "
              f"ret_over_hodl={test_metrics['returns_over_uniform_hodl']:.4f}")
        print(f"    WFE:         {wfe:.4f}")
        print(f"    Rademacher:  R̂={rademacher_complexity:.4f}, "
              f"haircut={haircut:.4f}")
        print(f"    Adjusted:    sharpe={adjusted_test_sharpe:.4f}")

    # Extract final state
    final_weights = np.array(test_output["weights"][-1])
    final_value = float(test_output["value"][-1])

    return CycleResult(
        cycle_number=cycle.cycle_number,
        train_sharpe=train_metrics["sharpe"],
        train_returns=train_metrics["return"],
        train_returns_over_hodl=train_metrics["returns_over_uniform_hodl"],
        val_sharpe=best_val_sharpe,
        test_sharpe=test_metrics["sharpe"],
        test_returns=test_metrics["return"],
        test_returns_over_hodl=test_metrics["returns_over_uniform_hodl"],
        walk_forward_efficiency=wfe,
        epochs_trained=epochs_trained,
        stopped_early=stopped_early,
        final_lr=local_lr,
        n_checkpoints=len(checkpoint_tracker.checkpoints),
        rademacher_complexity=rademacher_complexity,
        adjusted_test_sharpe=adjusted_test_sharpe,
        best_params={k: np.array(v) for k, v in params.items()},
        final_weights=final_weights,
        final_value=final_value,
        checkpoint_returns=checkpoint_tracker.get_returns_matrix(),
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def robust_walk_forward_training(
    run_fingerprint: dict,
    n_cycles: int = 5,
    keep_fixed_start: bool = True,
    max_epochs_per_cycle: int = 500,
    patience: int = 50,
    val_fraction: float = 0.2,
    warm_start: bool = True,
    maintain_pool_state: bool = True,
    regularization_strength: float = 0.01,
    checkpoint_every: int = 50,
    verbose: bool = True,
    root: str = None,
) -> Tuple[RobustTrainingResult, dict]:
    """
    Run robust walk-forward training with WFE and Rademacher tracking.

    This combines:
    1. Pardo-style walk-forward analysis with WFE metric
    2. SGD warm-starting for parameter continuity
    3. Rademacher complexity tracking for overfitting assessment

    Parameters
    ----------
    run_fingerprint : dict
        Run configuration
    n_cycles : int
        Number of walk-forward cycles
    keep_fixed_start : bool
        If True, expanding window; if False, rolling window
    max_epochs_per_cycle : int
        Maximum training epochs per cycle
    patience : int
        Early stopping patience
    val_fraction : float
        Fraction of training data for validation
    warm_start : bool
        Initialize from previous cycle's params
    maintain_pool_state : bool
        Carry forward pool weights and value
    regularization_strength : float
        L2 penalty toward prior params when warm-starting
    checkpoint_every : int
        Checkpoint frequency for Rademacher tracking
    verbose : bool
        Print progress

    Returns
    -------
    Tuple[RobustTrainingResult, dict]
        - Comprehensive results object
        - Summary statistics dictionary
    """
    recursive_default_set(run_fingerprint, run_fingerprint_defaults)

    # Generate cycles
    cycles = generate_walk_forward_cycles(
        start_date=run_fingerprint["startDateString"],
        end_date=run_fingerprint["endDateString"],
        n_cycles=n_cycles,
        keep_fixed_start=keep_fixed_start,
    )

    if verbose:
        print("=" * 70)
        print("ROBUST WALK-FORWARD TRAINING")
        print("=" * 70)
        print(f"Mode: {'Expanding' if keep_fixed_start else 'Rolling'} window")
        print(f"Cycles: {n_cycles}")
        print(f"Warm start: {warm_start}")
        print(f"Pool state continuity: {maintain_pool_state}")
        if warm_start:
            print(f"Regularization strength: {regularization_strength}")
        print(f"Checkpoint frequency: every {checkpoint_every} epochs")
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

    # Load data
    last_test_end = cycles[-1].test_end_date

    if verbose:
        print(f"\nLoading data: {run_fingerprint['startDateString']} → {last_test_end}")

    data_dict = get_data_dict(
        unique_tokens,
        run_fingerprint,
        data_kind=run_fingerprint["optimisation_settings"]["training_data_kind"],
        max_memory_days=run_fingerprint["max_memory_days"],
        start_date_string=run_fingerprint["startDateString"],
        end_time_string=last_test_end,
        do_test_period=False,
        root=root,
    )

    if verbose:
        print(f"Data loaded: {data_dict['prices'].shape[0]} timesteps")

    # Compute indices for each cycle
    for cycle in cycles:
        total_ts = datetime_to_timestamp(last_test_end) - datetime_to_timestamp(run_fingerprint["startDateString"])

        train_start_ts = datetime_to_timestamp(cycle.train_start_date) - datetime_to_timestamp(run_fingerprint["startDateString"])
        train_end_ts = datetime_to_timestamp(cycle.train_end_date) - datetime_to_timestamp(run_fingerprint["startDateString"])
        test_start_ts = datetime_to_timestamp(cycle.test_start_date) - datetime_to_timestamp(run_fingerprint["startDateString"])
        test_end_ts = datetime_to_timestamp(cycle.test_end_date) - datetime_to_timestamp(run_fingerprint["startDateString"])

        data_length = data_dict["end_idx"] - data_dict["start_idx"]

        cycle.train_start_idx = data_dict["start_idx"] + int(data_length * train_start_ts / total_ts)
        cycle.train_end_idx = data_dict["start_idx"] + int(data_length * train_end_ts / total_ts)
        cycle.test_start_idx = data_dict["start_idx"] + int(data_length * test_start_ts / total_ts)
        cycle.test_end_idx = min(
            data_dict["start_idx"] + int(data_length * test_end_ts / total_ts),
            data_dict["end_idx"]
        )

    # Train each cycle
    results: List[CycleResult] = []
    prev_params = None
    prev_weights = None
    prev_value = None

    # Aggregate checkpoint returns for global Rademacher
    all_checkpoint_returns = []

    for cycle in cycles:
        warm_start_params = prev_params if warm_start and prev_params is not None else None
        pool_value = prev_value if maintain_pool_state and prev_value is not None else None
        weights_logits = np.log(prev_weights) if maintain_pool_state and prev_weights is not None else None

        result = train_single_cycle_robust(
            run_fingerprint=run_fingerprint,
            cycle=cycle,
            pool=pool,
            initial_params=initial_params,
            data_dict=data_dict,
            all_sig_variations=all_sig_variations,
            n_assets=n_assets,
            warm_start_params=warm_start_params,
            initial_pool_value=pool_value,
            initial_weights_logits=weights_logits,
            max_epochs=max_epochs_per_cycle,
            patience=patience,
            val_fraction=val_fraction,
            regularization_strength=regularization_strength if warm_start else 0.0,
            checkpoint_every=checkpoint_every,
            verbose=verbose,
        )

        results.append(result)

        # Collect checkpoint returns for global complexity
        if result.checkpoint_returns is not None and result.checkpoint_returns.size > 0:
            all_checkpoint_returns.append(result.checkpoint_returns)

        # Update state for next cycle
        prev_params = result.best_params
        prev_weights = result.final_weights
        prev_value = result.final_value

    # Compute aggregate metrics
    oos_sharpes = [r.test_sharpe for r in results]
    is_sharpes = [r.train_sharpe for r in results]
    wfes = [r.walk_forward_efficiency for r in results]

    # Aggregate Rademacher complexity across all cycles
    # This captures the total "search" done across the entire walk-forward
    if all_checkpoint_returns:
        # Different cycles may have different return lengths due to window sizes
        # Truncate all to minimum length for fair comparison
        min_len = min(arr.shape[1] for arr in all_checkpoint_returns if arr.size > 0)
        if min_len > 0:
            truncated = [arr[:, :min_len] for arr in all_checkpoint_returns if arr.size > 0]
            combined_returns = np.vstack(truncated)
            aggregate_rademacher = compute_empirical_rademacher(combined_returns)
        else:
            aggregate_rademacher = 0.0
    else:
        aggregate_rademacher = 0.0

    # Aggregate OOS Sharpe (time-weighted would be ideal, but mean is simple)
    aggregate_oos_sharpe = np.mean(oos_sharpes)
    aggregate_is_sharpe = np.mean(is_sharpes)

    # Compute aggregate haircut
    total_test_T = sum(c.test_end_idx - c.test_start_idx for c in cycles)
    adjusted_aggregate_sharpe, aggregate_haircut = compute_rademacher_haircut(
        aggregate_oos_sharpe,
        aggregate_rademacher,
        total_test_T,
    )

    # Mean WFE (Pardo metric)
    mean_wfe = np.mean([w for w in wfes if np.isfinite(w)])

    result = RobustTrainingResult(
        cycles=results,
        mean_wfe=mean_wfe,
        aggregate_oos_sharpe=aggregate_oos_sharpe,
        aggregate_is_sharpe=aggregate_is_sharpe,
        aggregate_rademacher_complexity=aggregate_rademacher,
        adjusted_aggregate_sharpe=adjusted_aggregate_sharpe,
        mean_oos_sharpe=np.mean(oos_sharpes),
        std_oos_sharpe=np.std(oos_sharpes),
        worst_oos_sharpe=np.min(oos_sharpes),
        is_oos_gap=aggregate_is_sharpe - aggregate_oos_sharpe,
        haircut_magnitude=aggregate_haircut,
    )

    summary = {
        "n_cycles": n_cycles,
        "mode": "expanding" if keep_fixed_start else "rolling",
        "warm_start": warm_start,

        # Pardo metrics
        "mean_wfe": mean_wfe,
        "aggregate_is_sharpe": aggregate_is_sharpe,
        "aggregate_oos_sharpe": aggregate_oos_sharpe,
        "is_oos_gap": result.is_oos_gap,

        # Rademacher metrics
        "aggregate_rademacher_complexity": aggregate_rademacher,
        "adjusted_aggregate_sharpe": adjusted_aggregate_sharpe,
        "haircut_magnitude": aggregate_haircut,

        # Distribution statistics
        "mean_oos_sharpe": result.mean_oos_sharpe,
        "std_oos_sharpe": result.std_oos_sharpe,
        "worst_oos_sharpe": result.worst_oos_sharpe,

        # Training statistics
        "avg_epochs": np.mean([r.epochs_trained for r in results]),
        "early_stopped": sum(1 for r in results if r.stopped_early),
        "total_checkpoints": sum(r.n_checkpoints for r in results),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("ROBUST WALK-FORWARD SUMMARY")
        print("=" * 70)

        print("\n--- Pardo Metrics ---")
        print(f"Mean WFE:            {summary['mean_wfe']:.4f}")
        print(f"  (WFE > 0.5 suggests robustness; ~1.0 is ideal)")
        print(f"Aggregate IS Sharpe: {summary['aggregate_is_sharpe']:.4f}")
        print(f"Aggregate OOS Sharpe:{summary['aggregate_oos_sharpe']:.4f}")
        print(f"IS-OOS Gap:          {summary['is_oos_gap']:.4f}")

        print("\n--- Rademacher Metrics ---")
        print(f"Total Checkpoints:   {summary['total_checkpoints']}")
        print(f"Rademacher R̂:        {summary['aggregate_rademacher_complexity']:.4f}")
        print(f"Haircut Magnitude:   {summary['haircut_magnitude']:.4f}")
        print(f"Adjusted OOS Sharpe: {summary['adjusted_aggregate_sharpe']:.4f}")

        print("\n--- Distribution ---")
        print(f"OOS Sharpe:          {summary['mean_oos_sharpe']:.4f} ± {summary['std_oos_sharpe']:.4f}")
        print(f"Worst OOS Sharpe:    {summary['worst_oos_sharpe']:.4f}")

        print("\n--- Training ---")
        print(f"Avg Epochs:          {summary['avg_epochs']:.1f}")
        print(f"Early Stopped:       {summary['early_stopped']}/{n_cycles}")

        print("\n--- Per-Cycle Breakdown ---")
        for r in results:
            print(f"  Cycle {r.cycle_number}: "
                  f"IS={r.train_sharpe:.3f} → OOS={r.test_sharpe:.3f} "
                  f"(WFE={r.walk_forward_efficiency:.2f}, "
                  f"R̂={r.rademacher_complexity:.3f}, "
                  f"adj={r.adjusted_test_sharpe:.3f})")

        print("=" * 70)

    return result, summary


# =============================================================================
# Utility: Compare Training Approaches
# =============================================================================

def compare_approaches(
    run_fingerprint: dict,
    n_cycles: int = 4,
    max_epochs: int = 300,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare different training approaches on the same data.

    Runs:
    1. Classic (cold start, no pool state)
    2. Warm start only
    3. Warm start + pool state
    4. Warm start + pool state + regularization

    Returns comparison metrics.
    """
    configs = [
        {
            "name": "Classic (cold start)",
            "warm_start": False,
            "maintain_pool_state": False,
            "regularization_strength": 0.0,
        },
        {
            "name": "Warm start only",
            "warm_start": True,
            "maintain_pool_state": False,
            "regularization_strength": 0.0,
        },
        {
            "name": "Warm + pool state",
            "warm_start": True,
            "maintain_pool_state": True,
            "regularization_strength": 0.0,
        },
        {
            "name": "Warm + pool state + reg",
            "warm_start": True,
            "maintain_pool_state": True,
            "regularization_strength": 0.01,
        },
    ]

    comparisons = {}

    for config in configs:
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# Running: {config['name']}")
            print(f"{'#'*70}")

        result, summary = robust_walk_forward_training(
            run_fingerprint,
            n_cycles=n_cycles,
            max_epochs_per_cycle=max_epochs,
            warm_start=config["warm_start"],
            maintain_pool_state=config["maintain_pool_state"],
            regularization_strength=config["regularization_strength"],
            verbose=verbose,
        )

        comparisons[config["name"]] = {
            "mean_wfe": summary["mean_wfe"],
            "aggregate_oos_sharpe": summary["aggregate_oos_sharpe"],
            "adjusted_sharpe": summary["adjusted_aggregate_sharpe"],
            "is_oos_gap": summary["is_oos_gap"],
            "rademacher": summary["aggregate_rademacher_complexity"],
        }

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Approach':<30} {'WFE':>8} {'OOS':>8} {'Adj':>8} {'Gap':>8} {'R̂':>8}")
        print("-" * 70)
        for name, metrics in comparisons.items():
            print(f"{name:<30} {metrics['mean_wfe']:>8.3f} {metrics['aggregate_oos_sharpe']:>8.3f} "
                  f"{metrics['adjusted_sharpe']:>8.3f} {metrics['is_oos_gap']:>8.3f} "
                  f"{metrics['rademacher']:>8.3f}")
        print("=" * 70)

    return comparisons


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    run_fingerprint = {
        "startDateString": "2021-01-01 00:00:00",
        "endDateString": "2024-06-01 00:00:00",
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

    # Run robust walk-forward training
    result, summary = robust_walk_forward_training(
        run_fingerprint,
        n_cycles=4,
        keep_fixed_start=True,
        max_epochs_per_cycle=200,
        patience=30,
        warm_start=True,
        maintain_pool_state=True,
        regularization_strength=0.01,
        checkpoint_every=25,
        verbose=True,
    )
