"""
Runners module for quantammsim.

This module provides simulation runners and utilities for running AMM simulations.
"""

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError(
        "JAX is required for runners. Please install jax and jaxlib."
    ) from e

try:
    import numpy as np
except ImportError as e:
    raise ImportError("NumPy is required for runners.") from e

from .jax_runners import (
    do_run_on_historic_data,
    train_on_historic_data,
)

from .jax_runner_utils import (
    nan_rollback,
    Hashabledict,
    get_trades_and_fees,
    get_unique_tokens,
    OptunaManager,
    generate_evaluation_points,
    create_trial_params,
    create_static_dict,
    get_sig_variations,
    probe_max_n_parameter_sets,
    allocate_memory_budget,
    apply_memory_allocation,
    auto_configure_memory_params,
)

from .multi_period_sgd import (
    multi_period_sgd_training,
    MultiPeriodResult,
    PeriodSpec,
    generate_period_specs,
)

from .robust_walk_forward import (
    WalkForwardCycle,
    compute_empirical_rademacher,
    compute_rademacher_haircut,
    compute_walk_forward_efficiency,
    generate_walk_forward_cycles,
)

from .training_evaluator import (
    TrainingEvaluator,
    EvaluationResult,
    compare_trainers,
    TrainerWrapper,
    FunctionWrapper,
    RandomBaselineWrapper,
)

from .hyperparam_tuner import (
    HyperparamTuner,
    HyperparamSpace,
    TuningResult,
    quick_tune,
    tune_for_robustness,
)

__all__ = [
    # Core runners
    "do_run_on_historic_data",
    "train_on_historic_data",
    # Utilities
    "nan_rollback",
    "Hashabledict",
    "get_trades_and_fees",
    "get_unique_tokens",
    "OptunaManager",
    "generate_evaluation_points",
    "create_trial_params",
    "create_static_dict",
    "get_sig_variations",
    "probe_max_n_parameter_sets",
    "allocate_memory_budget",
    "apply_memory_allocation",
    "auto_configure_memory_params",
    # Multi-period SGD
    "multi_period_sgd_training",
    "MultiPeriodResult",
    "PeriodSpec",
    "generate_period_specs",
    # Robust walk-forward utilities
    "WalkForwardCycle",
    "compute_empirical_rademacher",
    "compute_rademacher_haircut",
    "compute_walk_forward_efficiency",
    "generate_walk_forward_cycles",
    # Training evaluator (meta-runner)
    "TrainingEvaluator",
    "EvaluationResult",
    "compare_trainers",
    "TrainerWrapper",
    "FunctionWrapper",
    "RandomBaselineWrapper",
    # Hyperparameter tuner (meta-meta-runner)
    "HyperparamTuner",
    "HyperparamSpace",
    "TuningResult",
    "quick_tune",
    "tune_for_robustness",
]
