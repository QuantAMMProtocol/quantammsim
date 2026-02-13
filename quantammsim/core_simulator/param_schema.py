"""Parameter schema utilities for pool-owned parameter definitions.

This module provides the infrastructure for pools to declare their own parameters
— including default values, Optuna search ranges, and metadata — in a single
location (the pool class's ``PARAM_SCHEMA`` dict). This eliminates the need for
scattered parameter definitions across ``run_fingerprint_defaults``, separate
``optuna_settings`` dicts, and ad-hoc initialization code.

The schema system supports a three-level priority for parameter resolution:

1. ``initial_values_dict`` (user-provided at runtime — highest priority)
2. ``run_fingerprint`` (experiment configuration)
3. ``PARAM_SCHEMA`` default (pool class definition — lowest priority)

Key classes:

- :class:`OptunaRange`: Defines search bounds and scale for hyperparameter tuning.
- :class:`ParamSpec`: Full specification for a single parameter (default, tuning range,
  transform, trainability).
- :data:`COMMON_PARAM_SCHEMA`: Shared parameter definitions used across multiple pool types.

Key functions:

- :func:`get_param_value`: Resolves a parameter value through the priority chain.
- :func:`get_optuna_range`: Retrieves tuning ranges with optional run_fingerprint overrides.
- :func:`sample_in_range`: Maps [0, 1] samples to parameter ranges (for ensemble init).

Example usage in a pool class::

    class MomentumPool(TFMMBasePool):
        PARAM_SCHEMA = {
            "memory_length": ParamSpec(
                initial=10.0,
                optuna=OptunaRange(low=1, high=200, log_scale=True),
            ),
            "k_per_day": ParamSpec(
                initial=20,
                optuna=OptunaRange(low=0.1, high=1000, log_scale=True),
            ),
        }
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List
import numpy as np


@dataclass
class OptunaRange:
    """Defines the search range for Optuna hyperparameter optimization.

    Attributes
    ----------
    low : float
        Lower bound of the search range
    high : float
        Upper bound of the search range
    log_scale : bool
        Whether to use log scale for sampling (useful for parameters
        that span multiple orders of magnitude)
    scalar : bool
        If True, use same value for all assets. If False, tune per-asset.
    """
    low: float
    high: float
    log_scale: bool = False
    scalar: bool = False


@dataclass
class ParamSpec:
    """Specification for a single pool parameter.

    Attributes
    ----------
    initial : Union[float, List[float]]
        Default initial value. Can be a scalar (applied to all assets)
        or a list (per-asset values).
    optuna : Optional[OptunaRange]
        Search range for Optuna optimization. If None, parameter is not tuned.
    transform : Optional[str]
        Transformation to apply: "log2" for log_k, "logit_lamb" for memory params.
        If None, no transformation is applied.
    description : str
        Human-readable description of the parameter.
    trainable : bool
        Whether this parameter should receive gradients during training.
    """
    initial: Union[float, List[float]]
    optuna: Optional[OptunaRange] = None
    transform: Optional[str] = None
    description: str = ""
    trainable: bool = True


#: Parameter definitions shared across multiple pool types.
#:
#: Pool classes can merge this into their own ``PARAM_SCHEMA`` to inherit
#: common defaults without duplication. Currently contains:
#:
#: - ``initial_weights_logits``: Logit-space initial portfolio weights.
#:   These are passed through softmax to produce the initial weight vector.
#:   Typically not trained (optimized via Optuna instead), since gradient
#:   descent on initial weights is poorly conditioned.
COMMON_PARAM_SCHEMA = {
    "initial_weights_logits": ParamSpec(
        initial=1.0,
        optuna=OptunaRange(low=-10, high=10, log_scale=False, scalar=False),
        description="Logit-space initial portfolio weights",
        trainable=False,  # Usually not trained
    ),
}


def get_param_value(
    param_name: str,
    schema: Dict[str, ParamSpec],
    initial_values_dict: Dict[str, Any],
    run_fingerprint: Dict[str, Any],
) -> Any:
    """Get parameter value with priority: initial_values_dict > run_fingerprint > schema.

    Parameters
    ----------
    param_name : str
        Name of the parameter (e.g., "memory_length")
    schema : Dict[str, ParamSpec]
        Parameter schema from the pool class
    initial_values_dict : Dict[str, Any]
        User-provided initial values
    run_fingerprint : Dict[str, Any]
        Run configuration dict

    Returns
    -------
    Any
        The parameter value to use
    """
    # Check initial_values_dict first (highest priority)
    # Try both "initial_X" and "X" formats
    if f"initial_{param_name}" in initial_values_dict:
        return initial_values_dict[f"initial_{param_name}"]
    if param_name in initial_values_dict:
        return initial_values_dict[param_name]

    # Check run_fingerprint (second priority)
    if f"initial_{param_name}" in run_fingerprint:
        return run_fingerprint[f"initial_{param_name}"]
    if param_name in run_fingerprint:
        return run_fingerprint[param_name]

    # Fall back to schema default
    if param_name in schema:
        return schema[param_name].initial

    raise KeyError(f"Parameter '{param_name}' not found in initial_values_dict, "
                   f"run_fingerprint, or schema")


def get_optuna_range(
    param_name: str,
    schema: Dict[str, ParamSpec],
    run_fingerprint: Optional[Dict[str, Any]] = None,
) -> Optional[OptunaRange]:
    """Get Optuna search range for a parameter.

    Parameters
    ----------
    param_name : str
        Name of the parameter
    schema : Dict[str, ParamSpec]
        Parameter schema from the pool class
    run_fingerprint : Optional[Dict[str, Any]]
        Run configuration (may contain overrides)

    Returns
    -------
    Optional[OptunaRange]
        The Optuna range, or None if parameter shouldn't be tuned
    """
    if param_name not in schema:
        return None

    spec = schema[param_name]
    if spec.optuna is None:
        return None

    # Check for overrides in run_fingerprint
    if run_fingerprint:
        optuna_settings = run_fingerprint.get("optimisation_settings", {}).get("optuna_settings", {})
        param_config = optuna_settings.get("parameter_config", {})
        if param_name in param_config:
            cfg = param_config[param_name]
            return OptunaRange(
                low=cfg.get("low", spec.optuna.low),
                high=cfg.get("high", spec.optuna.high),
                log_scale=cfg.get("log_scale", spec.optuna.log_scale),
                scalar=cfg.get("scalar", spec.optuna.scalar),
            )

    return spec.optuna


def sample_in_range(
    samples: np.ndarray,
    optuna_range: OptunaRange,
) -> np.ndarray:
    """Map [0, 1] samples to parameter range.

    Parameters
    ----------
    samples : np.ndarray
        Samples in [0, 1] range
    optuna_range : OptunaRange
        The parameter's valid range

    Returns
    -------
    np.ndarray
        Samples mapped to [low, high] (or log-space if log_scale=True)
    """
    if optuna_range.log_scale:
        # Map [0, 1] to [log(low), log(high)] then exponentiate
        log_low = np.log(optuna_range.low)
        log_high = np.log(optuna_range.high)
        return np.exp(log_low + samples * (log_high - log_low))
    else:
        # Linear mapping
        return optuna_range.low + samples * (optuna_range.high - optuna_range.low)
