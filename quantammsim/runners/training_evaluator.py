"""
Training Evaluator: A Meta-Runner for Assessing Training Effectiveness

Wrap any training approach and evaluate whether it's effective using:
- Walk-Forward Efficiency (Pardo)
- Rademacher Complexity (Paleologo) - requires checkpoint tracking, see below
- OOS performance metrics

Usage:
------
```python
from quantammsim.runners.training_evaluator import TrainingEvaluator, compare_trainers

# Option 1: Wrap existing runner
evaluator = TrainingEvaluator.from_runner("train_on_historic_data", max_iterations=500)
results = evaluator.evaluate(run_fingerprint, n_cycles=5)

# Option 2: Wrap custom function
def my_trainer(data_dict, train_start_idx, train_end_idx, pool, run_fp, warm_start=None):
    # ... your logic ...
    return params, {"epochs": n}

evaluator = TrainingEvaluator.from_function(my_trainer)

# Option 3: Compare approaches
comparison = compare_trainers(
    run_fingerprint,
    trainers={
        "sgd": TrainingEvaluator.from_runner("train_on_historic_data"),
        "random": TrainingEvaluator.random_baseline(),
    },
)
```

Rademacher Complexity
---------------------
Rademacher complexity measures overfitting risk by tracking the "search space" explored
during optimization. To compute Rademacher complexity, the trainer must return
checkpoint_returns in metadata:

```python
def my_trainer_with_checkpoints(...):
    checkpoint_returns = []
    for epoch in range(n_epochs):
        params = update(params)
        if epoch % checkpoint_interval == 0:
            returns = evaluate(params)  # Returns array of shape (T,)
            checkpoint_returns.append(returns)

    return params, {
        "epochs_trained": n_epochs,
        "checkpoint_returns": np.stack(checkpoint_returns),  # Shape: (n_checkpoints, T)
    }

evaluator = TrainingEvaluator.from_function(
    my_trainer_with_checkpoints,
    compute_rademacher=True,  # Enable Rademacher computation
)
```

The built-in wrapper for train_on_historic_data now supports checkpoint tracking.
Enable it by passing compute_rademacher=True to from_runner():

```python
evaluator = TrainingEvaluator.from_runner(
    "train_on_historic_data",
    compute_rademacher=True,  # Enable checkpoint tracking
    checkpoint_interval=10,   # Optional: checkpoint every N iterations
)
```

For multi_period_sgd or custom trainers, you can implement checkpoint tracking manually
by returning checkpoint_returns in metadata (as shown above).
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.tree_util import Partial
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable, Union, Generator
from copy import deepcopy
from datetime import datetime
from functools import partial

from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set
from quantammsim.runners.jax_runner_utils import (
    Hashabledict,
    get_unique_tokens,
    create_static_dict,
    get_sig_variations,
)
from quantammsim.utils.post_train_analysis import calculate_period_metrics
from quantammsim.utils.data_processing.historic_data_utils import get_data_dict
from quantammsim.pools.creator import create_pool
from quantammsim.core_simulator.forward_pass import forward_pass_nograd

# Import utilities from robust_walk_forward
from quantammsim.runners.robust_walk_forward import (
    compute_empirical_rademacher,
    compute_rademacher_haircut,
    compute_walk_forward_efficiency,
    WalkForwardCycle,
    generate_walk_forward_cycles,
)


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class CycleEvaluation:
    """Evaluation results for a single walk-forward cycle."""
    cycle_number: int
    is_sharpe: float
    is_returns_over_hodl: float
    oos_sharpe: float
    oos_returns_over_hodl: float
    walk_forward_efficiency: float
    is_oos_gap: float
    epochs_trained: int = 0
    rademacher_complexity: Optional[float] = None
    adjusted_oos_sharpe: Optional[float] = None
    # Additional risk metrics (from calculate_period_metrics)
    is_calmar: Optional[float] = None
    oos_calmar: Optional[float] = None
    is_sterling: Optional[float] = None
    oos_sterling: Optional[float] = None
    is_ulcer: Optional[float] = None
    oos_ulcer: Optional[float] = None
    is_returns: Optional[float] = None
    oos_returns: Optional[float] = None


@dataclass
class EvaluationResult:
    """Complete evaluation results."""
    trainer_name: str
    trainer_config: Dict[str, Any]
    cycles: List[CycleEvaluation]

    # Aggregate metrics
    mean_wfe: float
    mean_oos_sharpe: float
    std_oos_sharpe: float
    worst_oos_sharpe: float
    mean_is_oos_gap: float

    # Rademacher-adjusted
    aggregate_rademacher: Optional[float] = None
    adjusted_mean_oos_sharpe: Optional[float] = None

    # Verdict
    is_effective: bool = False
    effectiveness_reasons: List[str] = field(default_factory=list)


# =============================================================================
# Trainer Wrappers
# =============================================================================

class TrainerWrapper:
    """
    Base class for wrapping training functions.

    A trainer must implement:
        train(data_dict, train_start_idx, train_end_idx, pool, run_fp, warm_start, ...)
            -> (params, metadata)
    """

    def __init__(self, name: str = "trainer", config: Optional[Dict] = None):
        self._name = name
        self._config = config or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def train(
        self,
        data_dict: dict,
        train_start_idx: int,
        train_end_idx: int,
        pool: Any,
        run_fingerprint: dict,
        n_assets: int,
        warm_start_params: Optional[Dict] = None,
        train_start_date: Optional[str] = None,
        train_end_date: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Train and return (params, metadata)."""
        raise NotImplementedError


class FunctionWrapper(TrainerWrapper):
    """Wrap a plain function as a trainer."""

    def __init__(
        self,
        fn: Callable,
        name: str = "custom",
        config: Optional[Dict] = None,
    ):
        super().__init__(name, config)
        self.fn = fn

    def train(
        self,
        data_dict: dict,
        train_start_idx: int,
        train_end_idx: int,
        pool: Any,
        run_fingerprint: dict,
        n_assets: int,
        warm_start_params: Optional[Dict] = None,
        train_start_date: Optional[str] = None,
        train_end_date: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.fn(
            data_dict=data_dict,
            train_start_idx=train_start_idx,
            train_end_idx=train_end_idx,
            pool=pool,
            run_fingerprint=run_fingerprint,
            n_assets=n_assets,
            warm_start_params=warm_start_params,
        )


class ExistingRunnerWrapper(TrainerWrapper):
    """
    Wrap an existing runner (train_on_historic_data, etc).

    This creates a thin adapter that calls the existing runner
    with appropriate parameters.
    """

    def __init__(
        self,
        runner_name: str,
        runner_kwargs: Optional[Dict] = None,
        compute_rademacher: bool = False,
        root: str = None,
    ):
        self.runner_name = runner_name
        self.runner_kwargs = runner_kwargs or {}
        self.compute_rademacher = compute_rademacher
        self.root = root
        super().__init__(
            name=f"{runner_name}",
            config=self.runner_kwargs,
        )

    def train(
        self,
        data_dict: dict,
        train_start_idx: int,
        train_end_idx: int,
        pool: Any,
        run_fingerprint: dict,
        n_assets: int,
        warm_start_params: Optional[Dict] = None,
        train_start_date: Optional[str] = None,
        train_end_date: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call the existing runner.

        Note: This adapts the cycle-based interface to the existing runners
        which expect full run_fingerprint with date strings. The date strings
        are used to modify the fingerprint so each cycle trains on different data.
        """
        if self.runner_name == "train_on_historic_data":
            return self._run_train_on_historic_data(
                data_dict, train_start_idx, train_end_idx,
                pool, run_fingerprint, n_assets, warm_start_params,
                train_start_date, train_end_date,
            )
        elif self.runner_name == "multi_period_sgd":
            return self._run_multi_period_sgd(
                data_dict, train_start_idx, train_end_idx,
                pool, run_fingerprint, n_assets, warm_start_params,
                train_start_date, train_end_date,
            )
        else:
            raise ValueError(f"Unknown runner: {self.runner_name}")

    def _run_train_on_historic_data(
        self,
        data_dict: dict,
        train_start_idx: int,
        train_end_idx: int,
        pool: Any,
        run_fingerprint: dict,
        n_assets: int,
        warm_start_params: Optional[Dict],
        train_start_date: Optional[str],
        train_end_date: Optional[str],
    ) -> Tuple[Dict, Dict]:
        """Adapter for train_on_historic_data."""
        from datetime import datetime, timedelta
        from quantammsim.runners.jax_runners import train_on_historic_data

        # Create a local fingerprint for this cycle
        local_fp = deepcopy(run_fingerprint)

        # Update date strings for this cycle's training window
        if train_start_date is not None:
            local_fp["startDateString"] = train_start_date
        if train_end_date is not None:
            local_fp["endDateString"] = train_end_date
            # train_on_historic_data requires a test period, so set endTestDateString
            # to 1 day after training end (we won't use the test results)
            train_end_dt = datetime.strptime(train_end_date, "%Y-%m-%d %H:%M:%S")
            test_end_dt = train_end_dt + timedelta(days=1)
            local_fp["endTestDateString"] = test_end_dt.strftime("%Y-%m-%d %H:%M:%S")

        # Override iterations if specified
        if "max_iterations" in self.runner_kwargs:
            local_fp["optimisation_settings"]["n_iterations"] = self.runner_kwargs["max_iterations"]

        # Enable checkpoint tracking if computing Rademacher
        if self.compute_rademacher:
            local_fp["optimisation_settings"]["track_checkpoints"] = True
            local_fp["optimisation_settings"]["checkpoint_interval"] = self.runner_kwargs.get(
                "checkpoint_interval", 10
            )

        # Run training with metadata return if computing Rademacher
        result = train_on_historic_data(
            local_fp,
            iterations_per_print=self.runner_kwargs.get("iterations_per_print", 10000),
            return_training_metadata=self.compute_rademacher,
            root=self.root,
        )

        if self.compute_rademacher:
            # Unpack (params, metadata) tuple
            params, metadata = result
        else:
            # Just params returned
            params = result
            metadata = {
                "epochs_trained": local_fp["optimisation_settings"].get("n_iterations", 0),
                "final_objective": 0.0,
            }

        # train_on_historic_data now returns properly shaped params
        # (n_ensemble_members, ...) not (n_parameter_sets, n_ensemble_members, ...)
        # No squeeze needed - selection happens in train_on_historic_data

        return params, metadata

    def _run_multi_period_sgd(
        self,
        data_dict: dict,
        train_start_idx: int,
        train_end_idx: int,
        pool: Any,
        run_fingerprint: dict,
        n_assets: int,
        warm_start_params: Optional[Dict],
        train_start_date: Optional[str],
        train_end_date: Optional[str],
    ) -> Tuple[Dict, Dict]:
        """Adapter for multi_period_sgd_training."""
        from datetime import datetime, timedelta
        from quantammsim.runners.multi_period_sgd import multi_period_sgd_training

        local_fp = deepcopy(run_fingerprint)

        # Update date strings for this cycle's training window
        if train_start_date is not None:
            local_fp["startDateString"] = train_start_date
        if train_end_date is not None:
            local_fp["endDateString"] = train_end_date
            # Set a test period just after training end for consistency
            train_end_dt = datetime.strptime(train_end_date, "%Y-%m-%d %H:%M:%S")
            test_end_dt = train_end_dt + timedelta(days=1)
            local_fp["endTestDateString"] = test_end_dt.strftime("%Y-%m-%d %H:%M:%S")

        result, summary = multi_period_sgd_training(
            local_fp,
            n_periods=self.runner_kwargs.get("n_periods", 4),
            max_epochs=self.runner_kwargs.get("max_epochs", 200),
            aggregation=self.runner_kwargs.get("aggregation", "mean"),
            verbose=False,
            root=self.root,
        )

        params = result.best_params
        metadata = {
            "epochs_trained": result.epochs_trained,
            "final_objective": result.final_objective,
        }

        return params, metadata


class RandomBaselineWrapper(TrainerWrapper):
    """
    Baseline: Random parameters.

    Use to check if your trainer beats random chance.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="random_baseline", config={"seed": seed})
        self.seed = seed
        self._call_count = 0

    def train(
        self,
        data_dict: dict,
        train_start_idx: int,
        train_end_idx: int,
        pool: Any,
        run_fingerprint: dict,
        n_assets: int,
        warm_start_params: Optional[Dict] = None,
        train_start_date: Optional[str] = None,
        train_end_date: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return random parameters (ignores date strings)."""
        rng = np.random.RandomState(self.seed + self._call_count)
        self._call_count += 1

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

        n_parameter_sets = 1
        params = pool.init_parameters(
            initial_params, run_fingerprint, n_assets, n_parameter_sets
        )

        # Add random noise
        for key in params:
            if hasattr(params[key], 'shape') and params[key].size > 0:
                noise = rng.randn(*params[key].shape) * 0.5
                params[key] = params[key] + noise

        # Squeeze out parameter set dimension
        params = {
            k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
            for k, v in params.items()
        }

        metadata = {"epochs_trained": 0, "final_objective": 0.0}
        return params, metadata


# =============================================================================
# Main Evaluator
# =============================================================================

class TrainingEvaluator:
    """
    Evaluates whether a training approach is effective.

    Wraps any trainer and runs walk-forward evaluation to assess
    effectiveness using WFE and Rademacher metrics.
    """

    def __init__(
        self,
        trainer: TrainerWrapper,
        n_cycles: int = 5,
        keep_fixed_start: bool = True,
        compute_rademacher: bool = False,  # Off by default (needs checkpoint tracking)
        verbose: bool = True,
        root: str = None,
    ):
        self.trainer = trainer
        self.n_cycles = n_cycles
        self.keep_fixed_start = keep_fixed_start
        self.compute_rademacher = compute_rademacher
        self.verbose = verbose
        self.root = root

    # -------------------------------------------------------------------------
    # Convenience Constructors
    # -------------------------------------------------------------------------

    @classmethod
    def from_runner(
        cls,
        runner_name: str,
        n_cycles: int = 5,
        verbose: bool = True,
        compute_rademacher: bool = False,
        root: str = None,
        **runner_kwargs,
    ) -> "TrainingEvaluator":
        """
        Create evaluator from an existing runner.

        Parameters
        ----------
        runner_name : str
            One of: "train_on_historic_data", "multi_period_sgd"
        n_cycles : int
            Number of walk-forward cycles
        verbose : bool
            Print progress
        compute_rademacher : bool
            Enable Rademacher complexity computation. This enables checkpoint
            tracking in the trainer, which saves intermediate returns during
            training for Rademacher estimation. Default False.
        root : str, optional
            Root directory for data files. If None, uses default data location.
        **runner_kwargs
            Arguments passed to the runner (e.g., max_iterations=500)

        Example
        -------
        >>> evaluator = TrainingEvaluator.from_runner(
        ...     "train_on_historic_data",
        ...     max_iterations=500,
        ...     compute_rademacher=True,  # Enable Rademacher complexity
        ... )
        """
        wrapper = ExistingRunnerWrapper(
            runner_name, runner_kwargs, compute_rademacher=compute_rademacher, root=root
        )
        return cls(
            trainer=wrapper,
            n_cycles=n_cycles,
            verbose=verbose,
            compute_rademacher=compute_rademacher,
            root=root,
        )

    @classmethod
    def from_function(
        cls,
        fn: Callable,
        name: str = "custom",
        n_cycles: int = 5,
        verbose: bool = True,
        root: str = None,
        **config,
    ) -> "TrainingEvaluator":
        """
        Create evaluator from a custom training function.

        Parameters
        ----------
        fn : Callable
            Function with signature:
            fn(data_dict, train_start_idx, train_end_idx, pool, run_fingerprint,
               n_assets, warm_start_params) -> (params, metadata)
        name : str
            Name for this trainer
        n_cycles : int
            Number of walk-forward cycles
        root : str, optional
            Root directory for data files. If None, uses default data location.
        **config
            Config dict for reporting

        Example
        -------
        >>> def my_trainer(data_dict, train_start_idx, train_end_idx, pool,
        ...                run_fingerprint, n_assets, warm_start_params=None):
        ...     # Your training logic
        ...     return params, {"epochs": 100}
        >>>
        >>> evaluator = TrainingEvaluator.from_function(my_trainer)
        """
        wrapper = FunctionWrapper(fn, name=name, config=config)
        return cls(trainer=wrapper, n_cycles=n_cycles, verbose=verbose, root=root)

    @classmethod
    def random_baseline(
        cls,
        seed: int = 42,
        n_cycles: int = 5,
        verbose: bool = True,
        root: str = None,
    ) -> "TrainingEvaluator":
        """
        Create evaluator that uses random parameters.

        Use this as a baseline to verify your trainer beats random chance.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        n_cycles : int
            Number of walk-forward cycles
        verbose : bool
            Print progress
        root : str, optional
            Root directory for data files. If None, uses default data location.
        """
        wrapper = RandomBaselineWrapper(seed=seed)
        return cls(trainer=wrapper, n_cycles=n_cycles, verbose=verbose, root=root)

    # -------------------------------------------------------------------------
    # Core Evaluation
    # -------------------------------------------------------------------------

    def evaluate_iter(
        self, run_fingerprint: dict
    ) -> "Generator[CycleEvaluation, None, EvaluationResult]":
        """
        Generator that yields CycleEvaluation after each cycle completes.

        This allows callers to inspect intermediate results and potentially
        stop early (e.g., for Optuna pruning).

        Yields
        ------
        CycleEvaluation
            Results from each completed cycle

        Returns
        -------
        EvaluationResult
            Final aggregated results (accessible via generator.value after StopIteration)

        Example
        -------
        >>> evaluator = TrainingEvaluator.from_runner("train_on_historic_data")
        >>> gen = evaluator.evaluate_iter(run_fingerprint)
        >>> for cycle_eval in gen:
        ...     print(f"Cycle {cycle_eval.cycle_number}: OOS Sharpe = {cycle_eval.oos_sharpe}")
        ...     if cycle_eval.oos_sharpe < -1.0:
        ...         break  # Stop early if terrible
        >>> # If completed, get final result
        >>> # final_result = gen.value  # Only available after StopIteration
        """
        recursive_default_set(run_fingerprint, run_fingerprint_defaults)

        if self.verbose:
            print("=" * 70)
            print(f"EVALUATING: {self.trainer.name}")
            print("=" * 70)
            print(f"Config: {self.trainer.config}")
            print(f"Cycles: {self.n_cycles}")
            print(f"Mode: {'Expanding' if self.keep_fixed_start else 'Rolling'}")
            print("=" * 70)

        # Setup
        unique_tokens = get_unique_tokens(run_fingerprint)
        n_assets = len(unique_tokens)

        pool = create_pool(run_fingerprint["rule"])
        assert pool.is_trainable(), "Pool must be trainable"

        # Generate cycles (reuse from robust_walk_forward)
        cycles = generate_walk_forward_cycles(
            start_date=run_fingerprint["startDateString"],
            end_date=run_fingerprint["endDateString"],
            n_cycles=self.n_cycles,
            keep_fixed_start=self.keep_fixed_start,
        )

        # Load data for full period
        last_test_end = cycles[-1].test_end_date

        if self.verbose:
            print(f"\nLoading data: {run_fingerprint['startDateString']} → {last_test_end}")

        data_dict = get_data_dict(
            unique_tokens,
            run_fingerprint,
            data_kind=run_fingerprint["optimisation_settings"]["training_data_kind"],
            max_memory_days=run_fingerprint["max_memory_days"],
            start_date_string=run_fingerprint["startDateString"],
            end_time_string=last_test_end,
            do_test_period=False,
            root=self.root,
        )

        if self.verbose:
            print(f"Data loaded: {data_dict['prices'].shape[0]} timesteps")

        # Convert cycle dates to indices
        self._compute_cycle_indices(cycles, run_fingerprint, data_dict, last_test_end)

        # Run evaluation
        cycle_results = []
        prev_params = None
        all_checkpoint_returns = []  # For aggregate Rademacher

        for cycle in cycles:
            if self.verbose:
                print(f"\n--- Cycle {cycle.cycle_number} ---")

            # Train
            params, metadata = self.trainer.train(
                data_dict=data_dict,
                train_start_idx=cycle.train_start_idx,
                train_end_idx=cycle.train_end_idx,
                pool=pool,
                run_fingerprint=run_fingerprint,
                n_assets=n_assets,
                warm_start_params=prev_params,
                train_start_date=cycle.train_start_date,
                train_end_date=cycle.train_end_date,
            )

            # Evaluate on IS
            is_metrics = self._evaluate_params(
                params, data_dict,
                cycle.train_start_idx, cycle.train_end_idx,
                pool, n_assets, run_fingerprint,
            )

            # Evaluate on OOS
            oos_metrics = self._evaluate_params(
                params, data_dict,
                cycle.test_start_idx, cycle.test_end_idx,
                pool, n_assets, run_fingerprint,
            )

            # Compute WFE
            wfe = compute_walk_forward_efficiency(
                is_metrics["sharpe"],
                oos_metrics["sharpe"],
                cycle.train_end_idx - cycle.train_start_idx,
                cycle.test_end_idx - cycle.test_start_idx,
            )

            # Compute Rademacher if checkpoint data available
            rademacher_complexity = None
            adjusted_oos_sharpe = None
            checkpoint_returns = metadata.get("checkpoint_returns")

            if self.compute_rademacher and checkpoint_returns is not None:
                checkpoint_returns = np.array(checkpoint_returns)
                if checkpoint_returns.size > 0:
                    rademacher_complexity = compute_empirical_rademacher(checkpoint_returns)
                    test_T = cycle.test_end_idx - cycle.test_start_idx
                    adjusted_oos_sharpe, _ = compute_rademacher_haircut(
                        oos_metrics["sharpe"],
                        rademacher_complexity,
                        test_T,
                    )
                    all_checkpoint_returns.append(checkpoint_returns)

            cycle_eval = CycleEvaluation(
                cycle_number=cycle.cycle_number,
                is_sharpe=is_metrics["sharpe"],
                is_returns_over_hodl=is_metrics["returns_over_uniform_hodl"],
                oos_sharpe=oos_metrics["sharpe"],
                oos_returns_over_hodl=oos_metrics["returns_over_uniform_hodl"],
                walk_forward_efficiency=wfe,
                is_oos_gap=is_metrics["sharpe"] - oos_metrics["sharpe"],
                epochs_trained=metadata.get("epochs_trained", 0),
                rademacher_complexity=rademacher_complexity,
                adjusted_oos_sharpe=adjusted_oos_sharpe,
                # Additional risk metrics
                is_calmar=is_metrics.get("calmar"),
                oos_calmar=oos_metrics.get("calmar"),
                is_sterling=is_metrics.get("sterling"),
                oos_sterling=oos_metrics.get("sterling"),
                is_ulcer=is_metrics.get("ulcer"),
                oos_ulcer=oos_metrics.get("ulcer"),
                is_returns=is_metrics.get("return"),
                oos_returns=oos_metrics.get("return"),
            )

            cycle_results.append(cycle_eval)
            prev_params = params

            if self.verbose:
                print(f"  IS:  sharpe={is_metrics['sharpe']:.4f}")
                print(f"  OOS: sharpe={oos_metrics['sharpe']:.4f}")
                print(f"  WFE: {wfe:.4f}")
                if rademacher_complexity is not None:
                    print(f"  Rademacher: R̂={rademacher_complexity:.4f}, adj_sharpe={adjusted_oos_sharpe:.4f}")

            # Yield intermediate result for pruning decisions
            yield cycle_eval

        # Aggregate results
        result = self._aggregate_results(cycle_results, cycles, all_checkpoint_returns)

        if self.verbose:
            self.print_report(result)

        return result

    def evaluate(self, run_fingerprint: dict) -> EvaluationResult:
        """
        Run walk-forward evaluation.

        Parameters
        ----------
        run_fingerprint : dict
            Run configuration

        Returns
        -------
        EvaluationResult
            Comprehensive evaluation results
        """
        # Use the generator, consuming all cycles
        gen = self.evaluate_iter(run_fingerprint)
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value
        return result

    def _compute_cycle_indices(
        self,
        cycles: List[WalkForwardCycle],
        run_fingerprint: dict,
        data_dict: dict,
        last_test_end: str,
    ):
        """Convert cycle dates to data indices."""
        def to_ts(date_str):
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp()

        total_ts = to_ts(last_test_end) - to_ts(run_fingerprint["startDateString"])
        data_length = data_dict["end_idx"] - data_dict["start_idx"]
        start_ts = to_ts(run_fingerprint["startDateString"])

        for cycle in cycles:
            cycle.train_start_idx = data_dict["start_idx"] + int(
                data_length * (to_ts(cycle.train_start_date) - start_ts) / total_ts
            )
            cycle.train_end_idx = data_dict["start_idx"] + int(
                data_length * (to_ts(cycle.train_end_date) - start_ts) / total_ts
            )
            cycle.test_start_idx = data_dict["start_idx"] + int(
                data_length * (to_ts(cycle.test_start_date) - start_ts) / total_ts
            )
            cycle.test_end_idx = min(
                data_dict["start_idx"] + int(
                    data_length * (to_ts(cycle.test_end_date) - start_ts) / total_ts
                ),
                data_dict["end_idx"]
            )

    def _evaluate_params(
        self,
        params: Dict[str, Any],
        data_dict: dict,
        start_idx: int,
        end_idx: int,
        pool: Any,
        n_assets: int,
        run_fingerprint: dict,
    ) -> Dict[str, float]:
        """Evaluate params on a data window."""
        bout_length = end_idx - start_idx

        all_sig_variations = get_sig_variations(n_assets)

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

        eval_fn = jit(Partial(
            forward_pass_nograd,
            prices=data_dict["prices"],
            static_dict=Hashabledict(static_dict),
            pool=pool,
        ))

        output = eval_fn(params, (start_idx, 0))
        prices = data_dict["prices"][start_idx:end_idx]

        metrics = calculate_period_metrics(
            {"value": output["value"], "reserves": output["reserves"]},
            prices
        )

        return metrics

    def _aggregate_results(
        self,
        cycle_results: List[CycleEvaluation],
        cycles: List[WalkForwardCycle],
        all_checkpoint_returns: List[np.ndarray],
    ) -> EvaluationResult:
        """Aggregate cycle results into final evaluation."""
        oos_sharpes = [c.oos_sharpe for c in cycle_results]
        wfes = [c.walk_forward_efficiency for c in cycle_results]
        gaps = [c.is_oos_gap for c in cycle_results]

        mean_wfe = np.mean([w for w in wfes if np.isfinite(w)])
        mean_oos_sharpe = np.mean(oos_sharpes)
        std_oos_sharpe = np.std(oos_sharpes)
        worst_oos_sharpe = np.min(oos_sharpes)
        mean_gap = np.mean(gaps)

        # Compute aggregate Rademacher if checkpoint data available
        aggregate_rademacher = None
        adjusted_mean_oos_sharpe = None

        if self.compute_rademacher and all_checkpoint_returns:
            # Different cycles may have different return lengths
            min_len = min(arr.shape[-1] for arr in all_checkpoint_returns if arr.size > 0)
            if min_len > 0:
                # Truncate and stack
                truncated = []
                for arr in all_checkpoint_returns:
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    truncated.append(arr[:, :min_len])
                combined_returns = np.vstack(truncated)
                aggregate_rademacher = compute_empirical_rademacher(combined_returns)

                # Compute haircut on aggregate OOS sharpe
                total_test_T = sum(c.test_end_idx - c.test_start_idx for c in cycles)
                adjusted_mean_oos_sharpe, _ = compute_rademacher_haircut(
                    mean_oos_sharpe,
                    aggregate_rademacher,
                    total_test_T,
                )

        # Effectiveness verdict
        is_effective = False
        reasons = []

        if mean_wfe >= 0.5:
            reasons.append(f"WFE {mean_wfe:.2f} >= 0.5 (good IS→OOS transfer)")
            is_effective = True
        else:
            reasons.append(f"WFE {mean_wfe:.2f} < 0.5 (poor IS→OOS transfer)")

        if worst_oos_sharpe > 0:
            reasons.append(f"Worst OOS Sharpe {worst_oos_sharpe:.2f} > 0")
        else:
            reasons.append(f"Worst OOS Sharpe {worst_oos_sharpe:.2f} <= 0")
            is_effective = False

        if mean_gap < 0.5:
            reasons.append(f"IS-OOS gap {mean_gap:.2f} < 0.5 (not overfitting badly)")
        else:
            reasons.append(f"IS-OOS gap {mean_gap:.2f} >= 0.5 (significant overfitting)")

        if aggregate_rademacher is not None:
            reasons.append(f"Rademacher R̂={aggregate_rademacher:.3f}")
            if adjusted_mean_oos_sharpe is not None and adjusted_mean_oos_sharpe > 0:
                reasons.append(f"Adjusted OOS Sharpe {adjusted_mean_oos_sharpe:.2f} > 0")
            elif adjusted_mean_oos_sharpe is not None:
                reasons.append(f"Adjusted OOS Sharpe {adjusted_mean_oos_sharpe:.2f} <= 0")
                is_effective = False

        return EvaluationResult(
            trainer_name=self.trainer.name,
            trainer_config=self.trainer.config,
            cycles=cycle_results,
            mean_wfe=mean_wfe,
            mean_oos_sharpe=mean_oos_sharpe,
            std_oos_sharpe=std_oos_sharpe,
            worst_oos_sharpe=worst_oos_sharpe,
            mean_is_oos_gap=mean_gap,
            aggregate_rademacher=aggregate_rademacher,
            adjusted_mean_oos_sharpe=adjusted_mean_oos_sharpe,
            is_effective=is_effective,
            effectiveness_reasons=reasons,
        )

    def print_report(self, result: EvaluationResult):
        """Print formatted evaluation report."""
        print("\n" + "=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)
        print(f"Trainer: {result.trainer_name}")

        print("\n--- Aggregate Metrics ---")
        print(f"Mean WFE:         {result.mean_wfe:.4f}")
        print(f"Mean OOS Sharpe:  {result.mean_oos_sharpe:.4f} ± {result.std_oos_sharpe:.4f}")
        print(f"Worst OOS Sharpe: {result.worst_oos_sharpe:.4f}")
        print(f"IS-OOS Gap:       {result.mean_is_oos_gap:.4f}")

        if result.aggregate_rademacher is not None:
            print("\n--- Rademacher Metrics ---")
            print(f"Aggregate R̂:     {result.aggregate_rademacher:.4f}")
            if result.adjusted_mean_oos_sharpe is not None:
                print(f"Adjusted Sharpe:  {result.adjusted_mean_oos_sharpe:.4f}")

        print(f"\n--- Verdict ---")
        print(f"Effective: {'YES' if result.is_effective else 'NO'}")
        for reason in result.effectiveness_reasons:
            print(f"  • {reason}")

        print("\n--- Per-Cycle ---")
        for c in result.cycles:
            rademacher_str = ""
            if c.rademacher_complexity is not None:
                rademacher_str = f", R̂={c.rademacher_complexity:.3f}"
            print(f"  Cycle {c.cycle_number}: "
                  f"IS={c.is_sharpe:.3f} → OOS={c.oos_sharpe:.3f} "
                  f"(WFE={c.walk_forward_efficiency:.2f}{rademacher_str})")

        print("=" * 70)


# =============================================================================
# Comparison Utility
# =============================================================================

def compare_trainers(
    run_fingerprint: dict,
    trainers: Dict[str, TrainingEvaluator],
    verbose: bool = True,
) -> Dict[str, EvaluationResult]:
    """
    Compare multiple trainers on the same data.

    Parameters
    ----------
    run_fingerprint : dict
        Run configuration
    trainers : Dict[str, TrainingEvaluator]
        Dictionary of name -> evaluator
    verbose : bool
        Print progress and summary

    Returns
    -------
    Dict[str, EvaluationResult]
        Results keyed by trainer name

    Example
    -------
    >>> results = compare_trainers(
    ...     run_fingerprint,
    ...     trainers={
    ...         "sgd_500": TrainingEvaluator.from_runner(
    ...             "train_on_historic_data", max_iterations=500
    ...         ),
    ...         "sgd_100": TrainingEvaluator.from_runner(
    ...             "train_on_historic_data", max_iterations=100
    ...         ),
    ...         "random": TrainingEvaluator.random_baseline(),
    ...     },
    ... )
    """
    results = {}

    for name, evaluator in trainers.items():
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# Evaluating: {name}")
            print(f"{'#' * 70}")

        results[name] = evaluator.evaluate(run_fingerprint)

    if verbose:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Trainer':<30} {'WFE':>8} {'OOS':>8} {'Worst':>8} {'Gap':>8} {'Eff?':>6}")
        print("-" * 70)

        for name, r in results.items():
            eff = "YES" if r.is_effective else "NO"
            print(f"{name:<30} {r.mean_wfe:>8.3f} {r.mean_oos_sharpe:>8.3f} "
                  f"{r.worst_oos_sharpe:>8.3f} {r.mean_is_oos_gap:>8.3f} {eff:>6}")

        print("=" * 70)

    return results


# =============================================================================
# Example
# =============================================================================

if __name__ == "__main__":
    run_fingerprint = {
        "startDateString": "2022-01-01 00:00:00",
        "endDateString": "2023-06-01 00:00:00",
        "endTestDateString": "2024-01-01 00:00:00",
        "tokens": ["BTC", "ETH"],
        "rule": "momentum",
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1000000.0,
        "fees": 0.003,
        "return_val": "sharpe",
        "optimisation_settings": {
            "training_data_kind": "historic",
            "optimiser": "adam",
            "base_lr": 0.1,
            "n_iterations": 200,
        },
    }

    # Simple usage
    evaluator = TrainingEvaluator.random_baseline(n_cycles=3)
    result = evaluator.evaluate(run_fingerprint)
