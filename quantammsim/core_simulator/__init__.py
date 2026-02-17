"""
Core simulator module for quantammsim.

This module provides the core simulation functionality including forward passes,
parameter utilities, and result handling.
"""

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "NumPy is required for core simulator. Please install numpy."
    ) from e

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError(
        "JAX is required for core simulator. Please install jax and jaxlib."
    ) from e

from .forward_pass import (
    forward_pass,
    forward_pass_nograd,
    _calculate_return_value,
)

from .windowing_utils import (
    get_indices,
)

from .param_utils import (
    recursive_default_set,
    check_run_fingerprint,
    memory_days_to_logit_lamb,
)

from .result_exporter import (
    save_multi_params,
)

__all__ = [
    # Forward pass functions
    'forward_pass',
    'forward_pass_nograd',
    '_calculate_return_value',
    
    # Windowing utilities
    'get_indices',
    
    # Parameter utilities
    'recursive_default_set',
    'check_run_fingerprint',
    'memory_days_to_logit_lamb',
    
    # Result handling
    'save_multi_params',
]
