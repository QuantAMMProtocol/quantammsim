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
best_params, study = tuner.tune(run_fingerprint)

# Use best params for final training
run_fingerprint["optimisation_settings"].update(best_params)

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
    - warmup_steps: only sampled when lr_schedule_type == "warmup_cosine"

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
            Include lr_schedule_type and warmup_steps (conditional).
        include_early_stopping : bool
            Include early_stopping_patience.
        include_weight_decay : bool
            Include use_weight_decay and weight_decay (conditional).
        minimal : bool
            If True, return minimal space with just lr and iterations.

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

        max_bout_days = int(cycle_days * 0.9)
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
            params = {
                "base_lr": lr_range,
                "batch_size": {"low": 1, "high": 16, "log": False, "type": "int"},
                "n_iterations": {"low": 50, "high": 500, "log": True, "type": "int"},
                "bout_offset_days": {"low": 1, "high": max_bout_days, "log": True, "type": "int"},
                "clip_norm": {"low": 1.0, "high": 100.0, "log": True},
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
            # Only warmup_cosine uses warmup_steps
            params["warmup_steps"] = {
                "low": 10, "high": 500, "log": True, "type": "int",
                "conditional_on": "lr_schedule_type", "conditional_value": "warmup_cosine"
            }

        if include_early_stopping:
            params["early_stopping_patience"] = {"low": 20, "high": 200, "log": True, "type": "int"}

        # noise_scale: controls initialization diversity for n_parameter_sets > 1
        # Larger noise = more diverse initializations = better exploration but more variance
        # Only relevant when n_parameter_sets > 1 (set in run_fingerprint)
        params["noise_scale"] = {"low": 0.01, "high": 0.5, "log": True}

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

        # Apply suggested hyperparameters
        # These go in optimisation_settings
        opt_settings_keys = [
            "base_lr", "batch_size", "n_iterations",
            "clip_norm", "n_cycles", "lr_schedule_type", "lr_decay_ratio",
            "warmup_steps", "early_stopping_patience", "noise_scale",
        ]

        # Initial strategy parameter values go directly in run_fingerprint
        initial_param_keys = [
            "initial_memory_length", "initial_memory_length_delta",
            "initial_k_per_day", "initial_log_amplitude",
            "initial_raw_width", "initial_raw_exponents",
            "initial_pre_exp_scaling", "initial_weights_logits",
        ]

        for key, value in suggested.items():
            if key in opt_settings_keys:
                fp["optimisation_settings"][key] = value
            elif key in initial_param_keys:
                # Initial strategy params go in fingerprint root
                fp[key] = value
            elif key == "bout_offset_days":
                # Convert days to minutes (bout_offset is in minutes)
                fp["bout_offset"] = value * 1440
            elif key == "bout_offset":
                # Legacy: direct minutes value
                fp["bout_offset"] = value
            # Skip control params that aren't real hyperparams
            elif key in ["use_weight_decay", "weight_decay"]:
                pass  # Already handled above
            # multi_period_sgd specific params handled in runner_kwargs

        # Build runner kwargs with suggested params
        # Only include params that were actually sampled (conditional params may be absent)
        local_runner_kwargs = deepcopy(runner_kwargs)
        for key in ["n_periods", "max_epochs", "aggregation", "softmin_temperature"]:
            if key in suggested:
                local_runner_kwargs[key] = suggested[key]

        # Create evaluator
        evaluator = TrainingEvaluator.from_runner(
            runner_name,
            n_cycles=n_wfa_cycles,
            verbose=verbose,
            root=root,
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
                intermediate_value = extract_cycle_metric(cycle_evals, objective_metric)

                # Report intermediate value for pruning
                if enable_pruning:
                    trial.report(intermediate_value, step=cycle_eval.cycle_number)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        if verbose:
                            print(f"Trial {trial.number} pruned at cycle {cycle_eval.cycle_number} "
                                  f"(intermediate={intermediate_value:.4f})")
                        raise optuna.TrialPruned()

        except optuna.TrialPruned:
            raise  # Re-raise pruning exception
        except Exception as e:
            if verbose:
                print(f"Trial {trial.number} failed: {e}")
                traceback.print_exc()
            # Return bad value for failed trials
            # Metrics we MAXIMIZE (higher is better): sharpe, wfe, calmar, sterling, returns
            # Metrics we MINIMIZE (lower is better): ulcer, is_oos_gap
            maximize_metrics = [
                "mean_oos_sharpe", "worst_oos_sharpe",
                "mean_wfe", "worst_wfe",
                "adjusted_mean_oos_sharpe",
                "mean_oos_calmar", "worst_oos_calmar",
                "mean_oos_sterling", "worst_oos_sterling",
                "mean_oos_returns", "worst_oos_returns",
                "mean_oos_returns_over_hodl", "worst_oos_returns_over_hodl",
            ]
            if objective_metric in maximize_metrics:
                return float("-inf")  # Worst possible for maximization
            else:
                return float("inf")  # Worst possible for minimization

        # Store full result for later analysis
        trial.set_user_attr("evaluation_result", {
            "mean_oos_sharpe": result.mean_oos_sharpe,
            "mean_wfe": result.mean_wfe,
            "worst_oos_sharpe": result.worst_oos_sharpe,
            "mean_is_oos_gap": result.mean_is_oos_gap,
            "aggregate_rademacher": result.aggregate_rademacher,
            "adjusted_mean_oos_sharpe": result.adjusted_mean_oos_sharpe,
            "is_effective": result.is_effective,
        })

        # Return requested metric using unified extraction from cycles
        return extract_cycle_metric(result.cycles, objective_metric)

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
        _ = single_objective(trial)

        # Get stored results
        eval_result = trial.user_attrs.get("evaluation_result", {})

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
        if hyperparam_space is not None:
            self.hyperparam_space = hyperparam_space
        elif runner_name == "multi_period_sgd":
            self.hyperparam_space = HyperparamSpace.default_multi_period_space()
        else:
            self.hyperparam_space = HyperparamSpace.default_adam_space()

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
