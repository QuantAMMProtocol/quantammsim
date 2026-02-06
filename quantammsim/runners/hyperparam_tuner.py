"""
Hyperparameter Tuner: Optuna/TPE-based optimization of training hyperparameters.

This module provides meta-optimization for training hyperparameters using
walk-forward evaluation as the objective. Instead of optimizing for in-sample
performance (which leads to overfitting), we optimize for OOS metrics like:
- Mean OOS Sharpe
- Walk-Forward Efficiency (WFE)
- Rademacher-adjusted Sharpe

Architecture:
------------
Level 3: HyperparamTuner (this module)
    ↓ tries different (lr, bs, bout_offset, ...)
Level 2: TrainingEvaluator
    ↓ runs walk-forward cycles, computes WFE/Rademacher
Level 1: Trainer (train_on_historic_data, multi_period_sgd)
    ↓ optimizes strategy params (lamb, k, weights)
Level 0: Forward pass

Usage:
------
```python
from quantammsim.runners.hyperparam_tuner import HyperparamTuner

# Basic usage - tune training hyperparameters
tuner = HyperparamTuner(
    runner_name="train_on_historic_data",
    n_trials=50,
    n_wfa_cycles=3,  # WFA cycles per trial
)
result = tuner.tune(run_fingerprint)

# Use best params for final training
run_fingerprint["optimisation_settings"].update(result.best_params)

# Multi-objective: optimize OOS Sharpe AND WFE
tuner = HyperparamTuner(
    runner_name="multi_period_sgd",
    objective="multi",  # Pareto front of OOS Sharpe vs WFE
    n_trials=30,
)
```
"""

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner, SuccessiveHalvingPruner
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from copy import deepcopy
from datetime import datetime
import json
import warnings
import traceback

from quantammsim.runners.training_evaluator import (
    TrainingEvaluator,
    EvaluationResult,
    ExistingRunnerWrapper,
)
from quantammsim.core_simulator.param_utils import recursive_default_set
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.runners.metric_extraction import extract_cycle_metric


# =============================================================================
# Constants
# =============================================================================

# Maps outer Optuna objective to inner training metric (return_val / early_stopping_metric)
# Used in HyperparamSpace.create() to decide if training_objective choice is meaningful,
# and in create_objective() to resolve "aligned" to the actual metric.
#
# Valid inner metrics (from calculate_period_metrics / forward_pass.py):
#   sharpe, return, returns_over_hodl, returns_over_uniform_hodl, calmar, sterling, ulcer
# All metrics are normalized so higher = better.
OUTER_TO_INNER_METRIC = {
    "mean_oos_sharpe": "sharpe",
    "worst_oos_sharpe": "sharpe",
    "mean_oos_daily_log_sharpe": "daily_log_sharpe",
    "worst_oos_daily_log_sharpe": "daily_log_sharpe",
    "mean_oos_calmar": "calmar",
    "worst_oos_calmar": "calmar",
    "mean_oos_sterling": "sterling",
    "worst_oos_sterling": "sterling",
    "mean_oos_ulcer": "ulcer",
    "worst_oos_ulcer": "ulcer",
    "mean_oos_returns_over_hodl": "returns_over_uniform_hodl",
    "worst_oos_returns_over_hodl": "returns_over_uniform_hodl",
    "mean_wfe": "sharpe",  # WFE uses sharpe internally
    "worst_wfe": "sharpe",
}


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""
    best_params: Dict[str, Any]
    best_value: float
    best_evaluation: Optional[EvaluationResult]

    # Study metadata
    n_trials: int
    n_completed: int
    n_pruned: int
    n_failed: int = 0

    # All trials for analysis
    all_trials: List[Dict[str, Any]] = field(default_factory=list)

    # Multi-objective results (if applicable)
    pareto_front: Optional[List[Dict[str, Any]]] = None

    # Timing
    total_time_seconds: float = 0.0


@dataclass
class HyperparamSpace:
    """
    Defines the hyperparameter search space with conditional sampling support.

    Each parameter can be:
    - float range: {"low": 0.001, "high": 1.0, "log": True}
    - int range: {"low": 1, "high": 100, "log": False, "type": "int"}
    - categorical: {"choices": ["adam", "sgd"]}
    - conditional: {"conditional_on": "parent_param", "conditional_value": "value", ...}

    Conditional Parameters:
    - softmin_temperature: only sampled when aggregation="softmin"
    - weight_decay: only sampled when use_weight_decay=True (and triggers adamw)
    - lr_decay_ratio: only sampled when lr_schedule_type != "constant"
    - warmup_fraction: only sampled when lr_schedule_type == "warmup_cosine"
      (converted to warmup_steps = warmup_fraction * n_iterations)

    Note on bout_offset:
    - bout_offset is in MINUTES, always multiples of 1440 (whole days)
    - Internally we tune bout_offset_days (1 to ~90% of cycle in days)
    - Then multiply by 1440 to get minutes
    """
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        runner: str = "train_on_historic_data",
        cycle_days: int = 180,
        optimizer: str = "adam",
        include_lr_schedule: bool = True,
        include_early_stopping: bool = True,
        include_weight_decay: bool = True,
        minimal: bool = False,
        objective_metric: str = "mean_oos_sharpe",
    ) -> "HyperparamSpace":
        """
        Unified factory method for creating hyperparameter search spaces.

        Parameters
        ----------
        runner : str
            Which runner to create space for: "train_on_historic_data" or "multi_period_sgd"
        cycle_days : int
            Approximate duration of one WFA cycle in days. Used to set bout_offset upper bound.
        optimizer : str
            Optimizer type: "adam", "adamw", or "sgd". Affects learning rate ranges.
        include_lr_schedule : bool
            Include lr_schedule_type and warmup_fraction (conditional).
        include_early_stopping : bool
            Include early_stopping_patience.
        include_weight_decay : bool
            Include use_weight_decay and weight_decay (conditional).
        minimal : bool
            If True, return minimal space with just lr and iterations.
        objective_metric : str
            Outer Optuna objective (e.g., "mean_oos_sharpe", "mean_oos_calmar").
            Used to determine if training_objective choice is meaningful.

        Returns
        -------
        HyperparamSpace
            Configured search space.

        Example
        -------
        >>> space = HyperparamSpace.create(cycle_days=180, optimizer="adam")
        >>> space = HyperparamSpace.create(runner="multi_period_sgd", cycle_days=90)
        >>> space = HyperparamSpace.create(minimal=True)  # Quick tuning
        """
        if minimal:
            return cls(params={
                "base_lr": {"low": 0.01, "high": 0.5, "log": True},
                "n_iterations": {"low": 50, "high": 200, "log": True, "type": "int"},
            })

        max_bout_days = max(1, int(cycle_days * 0.9))  # Ensure at least 1 day
        # LR ranges calibrated for each optimizer:
        # - SGD: typically needs higher LR (1e-3 to 1.0)
        # - Adam/AdamW: typically needs lower LR (1e-5 to 1e-1), with 3e-4 being common default
        lr_range = (
            {"low": 1e-3, "high": 1.0, "log": True}
            if optimizer == "sgd"
            else {"low": 1e-5, "high": 1e-1, "log": True}
        )

        if runner == "multi_period_sgd":
            params = {
                "base_lr": lr_range,
                "n_periods": {"low": 2, "high": 8, "log": False, "type": "int"},
                "max_epochs": {"low": 50, "high": 300, "log": True, "type": "int"},
                "aggregation": {"choices": ["mean", "worst", "softmin"]},
                "softmin_temperature": {
                    "low": 0.1, "high": 10.0, "log": True,
                    "conditional_on": "aggregation", "conditional_value": "softmin"
                },
                "bout_offset_days": {"low": 1, "high": max_bout_days, "log": True, "type": "int"},
            }
        else:
            # Ensure bout_offset_days bounds are valid (low <= high)
            # For short cycles, allow smaller bout offsets
            bout_offset_low = min(7, max_bout_days)  # Use 7 or max if max is smaller
            params = {
                "base_lr": lr_range,
                "batch_size": {"low": 2, "high": 64, "log": True, "type": "int"},
                "n_iterations": {"low": 50, "high": 5000, "log": True, "type": "int"},
                "bout_offset_days": {"low": bout_offset_low, "high": max_bout_days, "log": True, "type": "int"},
                "clip_norm": {"low": 0.5, "high": 50.0, "log": True},
            }

        if include_weight_decay:
            params["use_weight_decay"] = {"choices": [True, False]}
            params["weight_decay"] = {
                "low": 0.0001, "high": 0.1, "log": True,
                "conditional_on": "use_weight_decay", "conditional_value": True
            }

        if include_lr_schedule:
            # Available schedules in backpropagation._create_lr_schedule:
            # constant, cosine, exponential, warmup_cosine
            params["lr_schedule_type"] = {"choices": ["constant", "cosine", "warmup_cosine", "exponential"]}
            # lr_decay_ratio: min_lr = base_lr / lr_decay_ratio (only for decay schedules)
            params["lr_decay_ratio"] = {
                "low": 10, "high": 10000, "log": True,
                "conditional_on": "lr_schedule_type", "conditional_value_not": "constant"
            }
            # Only warmup_cosine uses warmup_fraction (converted to warmup_steps later)
            # Sample as fraction of n_iterations to avoid warmup_steps > n_iterations
            params["warmup_fraction"] = {
                "low": 0.05, "high": 0.3, "log": False,
                "conditional_on": "lr_schedule_type", "conditional_value": "warmup_cosine"
            }

        if include_early_stopping:
            params["use_early_stopping"] = {"choices": [True, False]}
            params["early_stopping_patience"] = {
                "low": 30, "high": 300, "log": True, "type": "int",
                "conditional_on": "use_early_stopping", "conditional_value": True
            }
            # Validation fraction - how much of training to hold out for early stopping
            # Larger = more robust validation signal but less training data
            params["val_fraction"] = {
                "low": 0.15, "high": 0.4, "log": False,
                "conditional_on": "use_early_stopping", "conditional_value": True
            }

        # Training objective: controls BOTH return_val (what gradients optimize) AND
        # early_stopping_metric (what decides when to stop / which params to select)
        # - "aligned": match the outer Optuna objective (sharpe→sharpe, calmar→calmar, etc.)
        # - "returns_over_uniform_hodl": always use this robust proxy metric
        # If "aligned" performs poorly (e.g., calmar has bad gradients), Optuna will learn
        # to favor "returns_over_uniform_hodl" instead.
        #
        # Only include this choice if it's meaningful (i.e., aligned would differ from
        # returns_over_uniform_hodl). If outer objective already maps to returns_over_hodl,
        # both choices would be identical, so we skip it.
        aligned_metric = OUTER_TO_INNER_METRIC.get(objective_metric, "returns_over_uniform_hodl")
        if aligned_metric != "returns_over_uniform_hodl":
            # Choice is meaningful - include it
            params["training_objective"] = {"choices": ["aligned", "returns_over_uniform_hodl"]}

        # noise_scale: controls initialization diversity for n_parameter_sets > 1
        # Larger noise = more diverse initializations = better exploration but more variance
        # Only relevant when n_parameter_sets > 1 (set in run_fingerprint)
        params["noise_scale"] = {"low": 0.01, "high": 0.5, "log": True}

        # maximum_change: max weight change per time step (controls trading speed limit)
        # Lower = more constrained/slower rebalancing, higher = more aggressive
        # Default is 3e-4, range from very constrained (1e-5) to effectively unconstrained (2.0)
        params["maximum_change"] = {"low": 1e-5, "high": 2.0, "log": True}

        # turnover_penalty: penalize weight turnover in loss function
        # Higher values discourage frequent rebalancing, improving out-of-sample robustness
        params["turnover_penalty"] = {"low": 1e-4, "high": 1.0, "log": True}

        # price_noise_sigma: multiplicative noise on prices during training
        # Acts as data augmentation to improve out-of-sample robustness
        params["price_noise_sigma"] = {"low": 0.0001, "high": 0.01, "log": True}

        return cls(params=params)

    @classmethod
    def default_sgd_space(cls, cycle_days: int = 180) -> "HyperparamSpace":
        """Default search space for SGD-based training. Wrapper around create()."""
        return cls.create(cycle_days=cycle_days, optimizer="sgd")

    @classmethod
    def default_adam_space(cls, cycle_days: int = 180) -> "HyperparamSpace":
        """Default search space for Adam-based training. Wrapper around create()."""
        return cls.create(cycle_days=cycle_days, optimizer="adam")

    @classmethod
    def default_multi_period_space(cls, cycle_days: int = 180) -> "HyperparamSpace":
        """Default search space for multi_period_sgd. Wrapper around create()."""
        return cls.create(runner="multi_period_sgd", cycle_days=cycle_days)

    @classmethod
    def minimal_space(cls) -> "HyperparamSpace":
        """Minimal space for quick tuning. Wrapper around create()."""
        return cls.create(minimal=True)

    @classmethod
    def for_cycle_duration(
        cls,
        cycle_days: int,
        runner: str = "train_on_historic_data",
        include_lr_schedule: bool = True,
        include_early_stopping: bool = True,
        include_weight_decay: bool = True,
        **kwargs,
    ) -> "HyperparamSpace":
        """Create search space with bout_offset scaled to cycle duration. Wrapper around create()."""
        return cls.create(
            runner=runner,
            cycle_days=cycle_days,
            include_lr_schedule=include_lr_schedule,
            include_early_stopping=include_early_stopping,
            include_weight_decay=include_weight_decay,
            **kwargs,
        )

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial with conditional sampling.

        Conditional parameters are only sampled when their parent condition is met.
        This allows Optuna's TPE sampler to properly model the conditional structure.

        Supported conditionals:
        - conditional_on + conditional_value: sample only when parent == value
        - conditional_on + conditional_value_not: sample only when parent != value
        """
        suggested = {}

        # First pass: sample all non-conditional params
        for name, spec in self.params.items():
            if "conditional_on" in spec:
                continue  # Handle in second pass
            suggested[name] = self._suggest_param(trial, name, spec)

        # Second pass: sample conditional params based on parent values
        for name, spec in self.params.items():
            if "conditional_on" not in spec:
                continue

            parent_name = spec["conditional_on"]
            parent_value = suggested.get(parent_name)

            # Check if condition is met
            should_sample = False
            if "conditional_value" in spec:
                should_sample = (parent_value == spec["conditional_value"])
            elif "conditional_value_not" in spec:
                should_sample = (parent_value != spec["conditional_value_not"])

            if should_sample:
                suggested[name] = self._suggest_param(trial, name, spec)
            # If condition not met, param is not suggested (not in dict)

        return suggested

    def _suggest_param(self, trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
        """Suggest a single parameter value."""
        if "choices" in spec:
            return trial.suggest_categorical(name, spec["choices"])
        elif spec.get("type") == "int":
            return trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        else:
            return trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )


# =============================================================================
# Objective Functions
# =============================================================================

def create_objective(
    run_fingerprint: dict,
    runner_name: str,
    runner_kwargs: Dict[str, Any],
    hyperparam_space: HyperparamSpace,
    n_wfa_cycles: int,
    objective_metric: str,
    verbose: bool,
    enable_pruning: bool = True,
    root: str = None,
) -> Callable[[optuna.Trial], float]:
    """
    Create an Optuna objective function with pruning support.

    The objective runs TrainingEvaluator with suggested hyperparameters
    and returns the specified metric. Reports intermediate values after
    each WFA cycle to enable early pruning of unpromising trials.

    Parameters
    ----------
    run_fingerprint : dict
        Base run configuration
    runner_name : str
        Which runner to use
    runner_kwargs : dict
        Extra kwargs for the runner
    hyperparam_space : HyperparamSpace
        Search space
    n_wfa_cycles : int
        Number of WFA cycles per trial
    objective_metric : str
        Metric to optimize
    verbose : bool
        Print progress
    enable_pruning : bool
        If True, report intermediate values and check for pruning after each cycle.
        If False, run all cycles without pruning checks (default True).
    """
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters (with conditional sampling)
        suggested = hyperparam_space.suggest(trial)

        # Build fingerprint with suggested params
        fp = deepcopy(run_fingerprint)
        recursive_default_set(fp, run_fingerprint_defaults)

        # Handle weight_decay conditional logic:
        # If use_weight_decay=True and weight_decay was sampled, use AdamW
        # If use_weight_decay=False or not present, use Adam (no weight decay)
        use_weight_decay = suggested.get("use_weight_decay", False)
        if use_weight_decay and "weight_decay" in suggested:
            fp["optimisation_settings"]["optimiser"] = "adamw"
            fp["optimisation_settings"]["weight_decay"] = suggested["weight_decay"]
        else:
            # Ensure no weight decay is applied
            fp["optimisation_settings"]["weight_decay"] = 0.0

        # Handle val_fraction and early_stopping
        # For Optuna: val_fraction always applies (controls validation holdout)
        # For SGD: val_fraction is tied to early_stopping
        is_optuna = fp["optimisation_settings"].get("method") == "optuna"
        use_early_stopping = suggested.get("use_early_stopping", True)

        if is_optuna:
            # Optuna always uses val_fraction (not tied to early_stopping)
            if "val_fraction" in suggested:
                fp["optimisation_settings"]["val_fraction"] = suggested["val_fraction"]
        else:
            # SGD: val_fraction tied to early_stopping
            fp["optimisation_settings"]["early_stopping"] = use_early_stopping
            if use_early_stopping:
                if "val_fraction" in suggested:
                    fp["optimisation_settings"]["val_fraction"] = suggested["val_fraction"]
            else:
                # Set a very high patience so it effectively never triggers
                fp["optimisation_settings"]["early_stopping_patience"] = 999999
                # Set val_fraction to 0 when early stopping is disabled
                fp["optimisation_settings"]["val_fraction"] = 0.0

        # Handle training_objective: controls BOTH return_val AND early_stopping_metric
        training_obj = suggested.get("training_objective", "returns_over_uniform_hodl")
        if training_obj == "aligned":
            # Align with outer objective
            inner_metric = OUTER_TO_INNER_METRIC.get(objective_metric, "returns_over_uniform_hodl")
        else:
            # Use robust proxy
            inner_metric = "returns_over_uniform_hodl"
        fp["return_val"] = inner_metric
        fp["optimisation_settings"]["early_stopping_metric"] = inner_metric

        # Apply suggested hyperparameters
        # These go in optimisation_settings
        opt_settings_keys = [
            "base_lr", "batch_size", "n_iterations",
            "clip_norm", "n_cycles", "lr_schedule_type", "lr_decay_ratio",
            "early_stopping_patience", "noise_scale",
        ]

        # Parameters that go directly in run_fingerprint (not optimisation_settings)
        fingerprint_root_keys = [
            # Initial strategy params (tunable)
            "initial_memory_length",
            "initial_k_per_day", "initial_log_amplitude",
            "initial_raw_width", "initial_raw_exponents",
            "initial_pre_exp_scaling",
            # Strategy constraints
            "maximum_change",
            "minimum_weight",
            # Training loss modifiers
            "turnover_penalty",
            # Data augmentation
            "price_noise_sigma",
        ]

        for key, value in suggested.items():
            if key in opt_settings_keys:
                fp["optimisation_settings"][key] = value
            elif key in fingerprint_root_keys:
                # Initial strategy params go in fingerprint root
                fp[key] = value
            elif key == "bout_offset_days":
                # Convert days to minutes (bout_offset is in minutes)
                fp["bout_offset"] = value * 1440
            elif key == "bout_offset":
                # Legacy: direct minutes value
                fp["bout_offset"] = value
            elif key == "warmup_fraction":
                # Convert warmup_fraction to warmup_steps based on n_iterations
                n_iterations = suggested.get("n_iterations", fp["optimisation_settings"].get("n_iterations", 1000))
                fp["optimisation_settings"]["warmup_steps"] = int(value * n_iterations)
            # Inner Optuna settings (for method="optuna")
            elif key == "optuna_overfitting_penalty":
                if "optuna_settings" not in fp["optimisation_settings"]:
                    fp["optimisation_settings"]["optuna_settings"] = {}
                fp["optimisation_settings"]["optuna_settings"]["overfitting_penalty"] = value
            elif key == "optuna_n_startup_trials":
                if "optuna_settings" not in fp["optimisation_settings"]:
                    fp["optimisation_settings"]["optuna_settings"] = {}
                fp["optimisation_settings"]["optuna_settings"]["n_startup_trials"] = int(value)
            elif key == "optuna_n_trials":
                if "optuna_settings" not in fp["optimisation_settings"]:
                    fp["optimisation_settings"]["optuna_settings"] = {}
                fp["optimisation_settings"]["optuna_settings"]["n_trials"] = int(value)
            # Skip control params that aren't real hyperparams (handled above)
            elif key in ["use_weight_decay", "weight_decay", "use_early_stopping",
                         "val_fraction", "training_objective"]:
                pass  # Already handled above
            # multi_period_sgd specific params handled in runner_kwargs

        # Build runner kwargs with suggested params
        # Only include params that were actually sampled (conditional params may be absent)
        local_runner_kwargs = deepcopy(runner_kwargs)
        for key in ["n_periods", "max_epochs", "aggregation", "softmin_temperature"]:
            if key in suggested:
                local_runner_kwargs[key] = suggested[key]

        # Determine WFE metric from outer objective (e.g., "mean_oos_calmar" → "calmar")
        wfe_metric = OUTER_TO_INNER_METRIC.get(objective_metric, "sharpe")

        # Create evaluator
        evaluator = TrainingEvaluator.from_runner(
            runner_name,
            n_cycles=n_wfa_cycles,
            verbose=verbose,
            root=root,
            wfe_metric=wfe_metric,
            **local_runner_kwargs,
        )

        # Run evaluation with pruning support
        try:
            cycle_evals = []
            gen = evaluator.evaluate_iter(fp)

            # Manually iterate to capture the return value from StopIteration
            # (for loops consume StopIteration without giving access to .value)
            result = None
            while True:
                try:
                    cycle_eval = next(gen)
                except StopIteration as e:
                    result = e.value
                    break

                cycle_evals.append(cycle_eval)

                # Compute running metric for intermediate reporting using unified extraction
                # Ensure Python float (not np.float64) for Optuna storage compatibility
                intermediate_value = float(extract_cycle_metric(cycle_evals, objective_metric))

                # Report intermediate value BEFORE pruning checks
                # This ensures all pruned trials have their intermediate values stored for analysis
                if enable_pruning:
                    trial.report(intermediate_value, step=cycle_eval.cycle_number)

                    # Aggressive pruning: prune if oos_returns_over_hodl is non-positive or NaN
                    # This catches obviously broken training early without waiting for Optuna's pruner
                    oos_roh = cycle_eval.oos_returns_over_hodl
                    if oos_roh is None or (isinstance(oos_roh, float) and np.isnan(oos_roh)) or oos_roh <= 0:
                        if verbose:
                            print(f"Trial {trial.number} pruned at cycle {cycle_eval.cycle_number}: "
                                  f"non-positive OOS metrics (sharpe={cycle_eval.oos_sharpe:.4f}, "
                                  f"returns_over_hodl={oos_roh}, intermediate={intermediate_value:.4f})")
                        raise optuna.TrialPruned()

                    # Check if trial should be pruned (Optuna's percentile/median pruner)
                    if trial.should_prune():
                        if verbose:
                            print(f"Trial {trial.number} pruned at cycle {cycle_eval.cycle_number} "
                                  f"by Optuna pruner (intermediate={intermediate_value:.4f})")
                        raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise  # Re-raise pruning exception
        except ValueError as e:
            # Re-raise ValueErrors (including NaN detection) - these should FAIL the trial
            # not silently return -inf. NaN metrics indicate training collapsed and
            # Optuna should mark this as a failed trial, not a completed one.
            if verbose:
                print(f"Trial {trial.number} failed with ValueError: {e}")
                traceback.print_exc()
            raise
        except Exception as e:
            if verbose:
                print(f"Trial {trial.number} failed: {e}")
                traceback.print_exc()
            # Return bad value for other failures (e.g., data loading issues)
            # Metrics we MAXIMIZE (higher is better): sharpe, wfe, calmar, sterling, returns, ulcer
            # Note: ulcer is negated (higher = less pain), so we maximize
            # Metrics we MINIMIZE (lower is better): is_oos_gap
            maximize_metrics = [
                "mean_oos_sharpe", "worst_oos_sharpe",
                "mean_wfe", "worst_wfe",
                "adjusted_mean_oos_sharpe",
                "mean_oos_calmar", "worst_oos_calmar",
                "mean_oos_sterling", "worst_oos_sterling",
                "mean_oos_returns", "worst_oos_returns",
                "mean_oos_returns_over_hodl", "worst_oos_returns_over_hodl",
                "mean_oos_ulcer", "worst_oos_ulcer",
            ]
            if objective_metric in maximize_metrics:
                return float("-inf")  # Worst possible for maximization
            else:
                return float("inf")  # Worst possible for minimization

        # Store full result for later analysis
        # Include per-cycle metrics for detailed inspection
        per_cycle_metrics = []
        for c in result.cycles:
            per_cycle_metrics.append({
                "cycle": c.cycle_number,
                # Date ranges
                "train_start_date": c.train_start_date,
                "train_end_date": c.train_end_date,
                "test_start_date": c.test_start_date,
                "test_end_date": c.test_end_date,
                # Metrics
                "is_sharpe": c.is_sharpe,
                "oos_sharpe": c.oos_sharpe,
                "is_calmar": c.is_calmar,
                "oos_calmar": c.oos_calmar,
                "is_sterling": c.is_sterling,
                "oos_sterling": c.oos_sterling,
                "is_ulcer": c.is_ulcer,
                "oos_ulcer": c.oos_ulcer,
                "is_returns_over_hodl": c.is_returns_over_hodl,
                "oos_returns_over_hodl": c.oos_returns_over_hodl,
                "wfe": c.walk_forward_efficiency,
                "is_oos_gap": c.is_oos_gap,
                # Trained strategy parameters
                "trained_params": c.trained_params,
                # Provenance: for debugging and linking to output files
                "run_location": c.run_location,
                "run_fingerprint": c.run_fingerprint,
            })

        try:
            trial.set_user_attr("evaluation_result", {
                "mean_oos_sharpe": result.mean_oos_sharpe,
                "mean_wfe": result.mean_wfe,
                "worst_oos_sharpe": result.worst_oos_sharpe,
                "mean_is_oos_gap": result.mean_is_oos_gap,
                "aggregate_rademacher": result.aggregate_rademacher,
                "adjusted_mean_oos_sharpe": result.adjusted_mean_oos_sharpe,
                "is_effective": result.is_effective,
                "cycles": per_cycle_metrics,
            })
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to store evaluation_result for trial {trial.number}: {e}")

        # Return requested metric using unified extraction from cycles
        # Ensure Python float (not np.float64) for Optuna storage compatibility
        final_value = extract_cycle_metric(result.cycles, objective_metric)
        final_value = float(final_value)  # Convert np.float64 -> Python float

        if verbose:
            print(f"Trial {trial.number} returning final value: {final_value}")

        return final_value

    return objective


def create_multi_objective(
    run_fingerprint: dict,
    runner_name: str,
    runner_kwargs: Dict[str, Any],
    hyperparam_space: HyperparamSpace,
    n_wfa_cycles: int,
    objectives: List[str],
    verbose: bool,
    enable_pruning: bool = True,
    root: str = None,
) -> Callable[[optuna.Trial], Tuple[float, ...]]:
    """
    Create a multi-objective function for Pareto optimization.

    Common combinations:
    - ["mean_oos_sharpe", "mean_wfe"]: Maximize both OOS performance and efficiency
    - ["mean_oos_sharpe", "neg_is_oos_gap"]: Maximize OOS while minimizing overfitting

    Note: Pruning in multi-objective is based on the first objective only.
    """
    single_objective = create_objective(
        run_fingerprint, runner_name, runner_kwargs,
        hyperparam_space, n_wfa_cycles, objectives[0], verbose,
        enable_pruning=enable_pruning,
        root=root,
    )

    def multi_objective(trial: optuna.Trial) -> Tuple[float, ...]:
        # Run evaluation once (with pruning on first objective)
        try:
            _ = single_objective(trial)
        except optuna.TrialPruned:
            raise  # Re-raise pruning exception
        except ValueError:
            raise  # Re-raise ValueError (e.g., NaN detection) to fail the trial
        except Exception as e:
            # For other exceptions, log and return worst values for all objectives
            if verbose:
                print(f"Trial {trial.number} multi-objective failed: {e}")
            return tuple(float("-inf") for _ in objectives)

        # Get stored results
        eval_result = trial.user_attrs.get("evaluation_result", {})

        # Check if evaluation_result is empty (shouldn't happen if single_objective succeeded)
        if not eval_result:
            if verbose:
                print(f"Trial {trial.number}: evaluation_result is empty after single_objective succeeded")
            return tuple(float("-inf") for _ in objectives)

        values = []
        for metric in objectives:
            if metric == "mean_oos_sharpe":
                values.append(eval_result.get("mean_oos_sharpe", float("-inf")))
            elif metric == "mean_wfe":
                values.append(eval_result.get("mean_wfe", float("-inf")))
            elif metric == "worst_oos_sharpe":
                values.append(eval_result.get("worst_oos_sharpe", float("-inf")))
            elif metric == "neg_is_oos_gap":
                gap = eval_result.get("mean_is_oos_gap", float("inf"))
                values.append(-gap)  # Negative because we want to minimize gap
            elif metric == "adjusted_mean_oos_sharpe":
                adj = eval_result.get("adjusted_mean_oos_sharpe")
                if adj is None:
                    adj = eval_result.get("mean_oos_sharpe", float("-inf"))
                values.append(adj)
            else:
                values.append(float("-inf"))

        return tuple(values)

    return multi_objective


# =============================================================================
# Main Tuner Class
# =============================================================================

class HyperparamTuner:
    """
    Tunes training hyperparameters using Optuna/TPE.

    Uses walk-forward evaluation as the objective, optimizing for
    OOS performance rather than in-sample fit.

    Parameters
    ----------
    runner_name : str
        Which runner to tune: "train_on_historic_data" or "multi_period_sgd"
    n_trials : int
        Number of Optuna trials to run
    n_wfa_cycles : int
        Number of walk-forward cycles per evaluation (more = more robust but slower)
    objective : str
        What to optimize:
        - "mean_oos_sharpe": Maximize average OOS Sharpe ratio
        - "mean_wfe": Maximize Walk-Forward Efficiency
        - "worst_oos_sharpe": Maximize worst-case OOS Sharpe
        - "adjusted_mean_oos_sharpe": Maximize Rademacher-adjusted Sharpe
        - "multi": Multi-objective (returns Pareto front)
    multi_objectives : List[str]
        If objective="multi", which metrics to jointly optimize
    hyperparam_space : HyperparamSpace
        Search space (uses sensible defaults if not provided)
    sampler : optuna.samplers.BaseSampler
        Optuna sampler (defaults to TPE)
    pruner : optuna.pruners.BasePruner
        Optuna pruner for early stopping unpromising trials.
        Defaults to MedianPruner. Set to None to disable pruning.
    enable_pruning : bool
        Whether to enable intermediate value reporting and pruning (default True)
    timeout_per_trial : Optional[float]
        Maximum seconds per trial. If None, no per-trial timeout (default None).
        Note: This is approximate - enforced via study.optimize timeout.
    total_timeout : Optional[float]
        Maximum total seconds for all trials. If None, no total timeout (default None).
    verbose : bool
        Print progress
    runner_kwargs : dict
        Extra kwargs passed to the runner

    Example
    -------
    >>> tuner = HyperparamTuner(
    ...     runner_name="train_on_historic_data",
    ...     n_trials=30,
    ...     objective="mean_oos_sharpe",
    ...     enable_pruning=True,  # Prune slow/bad trials early
    ...     total_timeout=3600,   # Stop after 1 hour
    ... )
    >>> result = tuner.tune(run_fingerprint)
    >>> print(f"Best LR: {result.best_params['base_lr']}")
    >>> print(f"Best OOS Sharpe: {result.best_value}")
    """

    def __init__(
        self,
        runner_name: str = "train_on_historic_data",
        n_trials: int = 50,
        n_wfa_cycles: int = 3,
        objective: str = "mean_oos_sharpe",
        multi_objectives: Optional[List[str]] = None,
        hyperparam_space: Optional[HyperparamSpace] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = "default",
        enable_pruning: bool = True,
        timeout_per_trial: Optional[float] = None,
        total_timeout: Optional[float] = None,
        verbose: bool = True,
        runner_kwargs: Optional[Dict[str, Any]] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        root: str = None,
    ):
        self.runner_name = runner_name
        self.n_trials = n_trials
        self.n_wfa_cycles = n_wfa_cycles
        self.objective = objective
        self.multi_objectives = multi_objectives or ["mean_oos_sharpe", "mean_wfe"]
        self.enable_pruning = enable_pruning
        self.timeout_per_trial = timeout_per_trial
        self.total_timeout = total_timeout
        self.verbose = verbose
        self.runner_kwargs = runner_kwargs or {}
        self.study_name = study_name
        self.storage = storage
        self.root = root

        # Set default search space based on runner
        # IMPORTANT: Pass objective so training_objective is conditionally included correctly
        if hyperparam_space is not None:
            self.hyperparam_space = hyperparam_space
        elif runner_name == "multi_period_sgd":
            self.hyperparam_space = HyperparamSpace.create(
                runner="multi_period_sgd",
                objective_metric=objective,
            )
        else:
            self.hyperparam_space = HyperparamSpace.create(
                optimizer="adam",
                objective_metric=objective,
            )

        # Set sampler (TPE is good for expensive evaluations)
        self.sampler = sampler or TPESampler(
            n_startup_trials=min(10, n_trials // 3),
            multivariate=True,
        )

        # Set pruner for early stopping unpromising trials
        # Note: WFA cycles are NOT true multi-fidelity (cycle 1 doesn't predict cycles 2-4,
        # they're different market regimes). So Hyperband/ASHA are overkill - their
        # sophisticated logic assumes correlation between fidelities we don't have.
        # PercentilePruner is better: just filter obvious disasters without predicting.
        if not enable_pruning or pruner is None or pruner == "none":
            self.pruner = optuna.pruners.NopPruner()
        elif pruner == "default" or pruner == "percentile":
            # PercentilePruner with 25%: prune bottom 25% after each cycle.
            # This is appropriate for WFA where cycles are independent regimes.
            # We're not predicting future cycles, just filtering disasters.
            self.pruner = PercentilePruner(
                percentile=25.0,
                n_startup_trials=max(5, n_trials // 5),
                n_warmup_steps=0,
                interval_steps=1,
            )
        elif pruner == "median":
            # MedianPruner: prune if below median of completed trials at same step
            self.pruner = MedianPruner(
                n_startup_trials=max(3, n_trials // 5),
                n_warmup_steps=0,
                interval_steps=1,
            )
        elif pruner == "hyperband":
            # HyperbandPruner: structured successive halving with multiple brackets
            # Note: Designed for true multi-fidelity where cheap evals predict expensive ones.
            # Use cautiously with WFA - cycles are different regimes, not fidelity levels.
            self.pruner = HyperbandPruner(
                min_resource=1,
                max_resource=n_wfa_cycles,
                reduction_factor=3,
            )
        elif pruner == "successive_halving":
            # SuccessiveHalvingPruner: single bracket successive halving
            self.pruner = SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=3,
            )
        else:
            # Custom pruner instance
            self.pruner = pruner

    def tune(self, run_fingerprint: dict) -> TuningResult:
        """
        Run hyperparameter tuning.

        Parameters
        ----------
        run_fingerprint : dict
            Base run configuration. Hyperparameters will be varied around this.

        Returns
        -------
        TuningResult
            Contains best parameters, best value, and all trial data.
        """
        start_time = datetime.now()

        # Create study
        study_name = self.study_name or f"hyperparam_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if self.objective == "multi":
            # Multi-objective optimization
            # Note: Multi-objective doesn't support pruning directly in Optuna,
            # but we still report intermediate values for monitoring
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage,
                directions=["maximize"] * len(self.multi_objectives),
                sampler=self.sampler,
                load_if_exists=True,
            )

            objective_fn = create_multi_objective(
                run_fingerprint,
                self.runner_name,
                self.runner_kwargs,
                self.hyperparam_space,
                self.n_wfa_cycles,
                self.multi_objectives,
                self.verbose,
                enable_pruning=False,  # Multi-objective doesn't support pruning
                root=self.root,
            )
        else:
            # Single objective optimization with pruning support
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage,
                direction="maximize",
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True,
            )

            objective_fn = create_objective(
                run_fingerprint,
                self.runner_name,
                self.runner_kwargs,
                self.hyperparam_space,
                self.n_wfa_cycles,
                self.objective,
                self.verbose,
                enable_pruning=self.enable_pruning,
                root=self.root,
            )

        # Run optimization
        if self.verbose:
            print("=" * 70)
            print(f"HYPERPARAMETER TUNING: {self.runner_name}")
            print("=" * 70)
            print(f"Objective: {self.objective}")
            print(f"Trials: {self.n_trials}")
            print(f"WFA cycles per trial: {self.n_wfa_cycles}")
            print(f"Search space: {list(self.hyperparam_space.params.keys())}")
            print(f"Pruning: {'enabled' if self.enable_pruning and self.objective != 'multi' else 'disabled'}")
            if self.total_timeout:
                print(f"Total timeout: {self.total_timeout}s")
            print("=" * 70)

        # Suppress Optuna's verbose logging unless we want it
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective_fn,
            n_trials=self.n_trials,
            timeout=self.total_timeout,
            show_progress_bar=self.verbose,
            catch=(Exception,),  # Catch exceptions and continue with other trials
        )

        # Collect results
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Get all trials data
        all_trials = []
        for trial in study.trials:
            trial_data = {
                "number": trial.number,
                "params": trial.params,
                "value": trial.value if self.objective != "multi" else trial.values,
                "state": str(trial.state),
                "evaluation_result": trial.user_attrs.get("evaluation_result"),
            }
            all_trials.append(trial_data)

        # Count trial states
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        # Build result
        if self.objective == "multi":
            # Multi-objective: return Pareto front
            pareto_trials = study.best_trials
            pareto_front = [
                {
                    "params": t.params,
                    "values": t.values,
                    "evaluation_result": t.user_attrs.get("evaluation_result"),
                }
                for t in pareto_trials
            ]

            # Pick one "best" for convenience (highest first objective)
            if pareto_trials:
                best_trial = max(pareto_trials, key=lambda t: t.values[0])
                best_params = best_trial.params
                best_value = best_trial.values[0]
            else:
                best_params = {}
                best_value = float("-inf")

            result = TuningResult(
                best_params=best_params,
                best_value=best_value,
                best_evaluation=None,  # Would need to re-run to get full result
                n_trials=self.n_trials,
                n_completed=n_completed,
                n_pruned=n_pruned,
                n_failed=n_failed,
                all_trials=all_trials,
                pareto_front=pareto_front,
                total_time_seconds=total_time,
            )
        else:
            # Single objective
            if n_completed > 0:
                best_trial = study.best_trial
                best_params = best_trial.params
                best_value = best_trial.value
            else:
                # No completed trials - return empty result
                best_params = {}
                best_value = float("-inf")

            result = TuningResult(
                best_params=best_params,
                best_value=best_value,
                best_evaluation=None,
                n_trials=self.n_trials,
                n_completed=n_completed,
                n_pruned=n_pruned,
                n_failed=n_failed,
                all_trials=all_trials,
                pareto_front=None,
                total_time_seconds=total_time,
            )

        if self.verbose:
            self._print_report(result, study)

        return result

    def _print_report(self, result: TuningResult, study: optuna.Study):
        """Print tuning results."""
        print("\n" + "=" * 70)
        print("TUNING COMPLETE")
        print("=" * 70)

        # Count trial states
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        n_running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])

        print(f"Trials: {len(study.trials)} total")
        print(f"  Completed: {result.n_completed}")
        print(f"  Pruned:    {result.n_pruned}")
        if n_failed > 0:
            print(f"  Failed:    {n_failed}")
        if n_running > 0:
            print(f"  Running:   {n_running}")

        print(f"\nTotal time: {result.total_time_seconds:.1f}s")
        if result.n_completed > 0:
            print(f"Time per completed trial: {result.total_time_seconds / result.n_completed:.1f}s")
        if result.n_pruned > 0:
            # Estimate time saved by pruning
            avg_completed_time = result.total_time_seconds / max(1, result.n_completed + result.n_pruned)
            print(f"Estimated time saved by pruning: {result.n_pruned * avg_completed_time * 0.5:.1f}s")

        print("\n--- Best Parameters ---")
        for key, value in result.best_params.items():
            print(f"  {key}: {value}")

        print(f"\n--- Best {self.objective}: {result.best_value:.4f} ---")

        if result.pareto_front:
            print(f"\n--- Pareto Front ({len(result.pareto_front)} solutions) ---")
            for i, sol in enumerate(result.pareto_front[:5]):  # Show top 5
                values_str = ", ".join(f"{v:.3f}" for v in sol["values"])
                print(f"  {i+1}. [{values_str}]")
                print(f"     params: {sol['params']}")

        print("=" * 70)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_tune(
    run_fingerprint: dict,
    runner_name: str = "train_on_historic_data",
    n_trials: int = 20,
) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning with minimal configuration.

    Returns the best hyperparameters found.

    Example
    -------
    >>> best_params = quick_tune(run_fingerprint, n_trials=20)
    >>> run_fingerprint["optimisation_settings"]["base_lr"] = best_params["base_lr"]
    """
    tuner = HyperparamTuner(
        runner_name=runner_name,
        n_trials=n_trials,
        n_wfa_cycles=2,  # Fast evaluation
        hyperparam_space=HyperparamSpace.minimal_space(),
        verbose=True,
    )
    result = tuner.tune(run_fingerprint)
    return result.best_params


def tune_for_robustness(
    run_fingerprint: dict,
    runner_name: str = "train_on_historic_data",
    n_trials: int = 50,
) -> TuningResult:
    """
    Tune hyperparameters with emphasis on robustness (WFE + OOS Sharpe).

    Uses multi-objective optimization to find the Pareto front of
    OOS performance vs walk-forward efficiency.
    """
    tuner = HyperparamTuner(
        runner_name=runner_name,
        n_trials=n_trials,
        n_wfa_cycles=4,  # More cycles for robust estimate
        objective="multi",
        multi_objectives=["mean_oos_sharpe", "mean_wfe"],
        verbose=True,
    )
    return tuner.tune(run_fingerprint)


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    # Example usage
    run_fingerprint = {
        "tokens": ["BTC", "ETH"],
        "rule": "momentum",
        "startDateString": "2021-01-01 00:00:00",
        "endDateString": "2023-06-01 00:00:00",
        "endTestDateString": "2024-01-01 00:00:00",
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1000000.0,
        "fees": 0.003,
        "optimisation_settings": {
            "training_data_kind": "historic",
            "optimiser": "adam",
        },
    }

    # Quick tune
    best_params = quick_tune(run_fingerprint, n_trials=10)
    print(f"\nBest params: {best_params}")
