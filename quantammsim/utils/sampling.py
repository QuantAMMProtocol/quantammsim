"""Structured sampling utilities for parameter space exploration.

Provides low-discrepancy and quasi-random sampling methods used by both
parameter initialization (base_pool.add_noise) and ensemble averaging
(EnsembleAveragingHook.init_base_parameters).
"""
from typing import Dict, Any, Tuple, List
import numpy as np


# =============================================================================
# Primitive sampling methods in [0, 1]^d
# =============================================================================

def _latin_hypercube_samples(n_samples: int, n_dims: int, seed: int = 0) -> np.ndarray:
    """Generate Latin Hypercube samples in [0, 1]^n_dims.

    Each dimension is divided into ``n_samples`` equal intervals and one
    random point is placed in each interval, with a random permutation
    across dimensions to ensure space-filling coverage.

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    n_dims : int
        Dimensionality of the sample space.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_samples, n_dims)
        Samples in [0, 1]^n_dims with stratified marginals.
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        intervals = np.arange(n_samples)
        rng.shuffle(intervals)
        samples[:, dim] = (intervals + rng.random(n_samples)) / n_samples
    return samples


def _centered_lhs_samples(n_samples: int, n_dims: int, seed: int = 0) -> np.ndarray:
    """Generate centered Latin Hypercube samples in [0, 1]^n_dims.

    Like :func:`_latin_hypercube_samples` but places each point at the
    *centre* of its stratum (interval midpoint) rather than a random
    position within the interval.  This produces a more deterministic,
    evenly-spaced layout at the cost of reduced randomness.

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    n_dims : int
        Dimensionality of the sample space.
    seed : int
        Random seed (controls the stratum permutation only).

    Returns
    -------
    np.ndarray, shape (n_samples, n_dims)
        Centred LHS samples in [0, 1]^n_dims.
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, n_dims))
    for dim in range(n_dims):
        intervals = np.arange(n_samples)
        rng.shuffle(intervals)
        samples[:, dim] = (intervals + 0.5) / n_samples
    return samples


def _sobol_samples(n_samples: int, n_dims: int, seed: int = 0) -> np.ndarray:
    """Generate scrambled Sobol quasi-random samples in [0, 1]^n_dims.

    Uses ``scipy.stats.qmc.Sobol`` with Owen scrambling for a
    low-discrepancy sequence.  Falls back to LHS if scipy is unavailable.

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    n_dims : int
        Dimensionality of the sample space.
    seed : int
        Scrambling seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_samples, n_dims)
        Sobol samples in [0, 1]^n_dims (or LHS fallback).
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        samples = sampler.random(n_samples + 1)[1:]
        return samples
    except ImportError:
        print("Warning: scipy.stats.qmc not available, falling back to LHS")
        return _latin_hypercube_samples(n_samples, n_dims, seed)


def _grid_samples(n_samples: int, n_dims: int, seed: int = 0) -> np.ndarray:
    """Generate grid samples in [0, 1]^n_dims.

    Creates a regular grid with ``ceil(n_samples^(1/n_dims))`` points per
    dimension, then randomly subsamples to exactly ``n_samples`` points
    if the full grid is larger.  Grid coordinates are spaced in [0.1, 0.9]
    to avoid boundary effects.

    Parameters
    ----------
    n_samples : int
        Desired number of sample points.
    n_dims : int
        Dimensionality of the sample space.
    seed : int
        Random seed for the subsampling step.

    Returns
    -------
    np.ndarray, shape (n_samples, n_dims)
        Grid samples in [0.1, 0.9]^n_dims.
    """
    points_per_dim = max(2, int(np.ceil(n_samples ** (1.0 / n_dims))))
    coords = [np.linspace(0.1, 0.9, points_per_dim) for _ in range(n_dims)]
    grid = np.meshgrid(*coords, indexing='ij')
    samples = np.stack([g.flatten() for g in grid], axis=-1)
    if len(samples) > n_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(samples), n_samples, replace=False)
        samples = samples[indices]
    return samples


def generate_ensemble_samples(
    n_samples: int,
    n_dims: int,
    method: str = "lhs",
    seed: int = 0,
) -> np.ndarray:
    """
    Generate samples for parameter space exploration.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_dims : int
        Number of dimensions
    method : str
        Sampling method: "gaussian", "lhs", "centered_lhs", "sobol", "grid"
    seed : int
        Random seed

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_dims) with values in [0, 1] for structured methods,
        or standard normal for "gaussian"
    """
    if method == "gaussian":
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_samples, n_dims))
    elif method == "lhs":
        return _latin_hypercube_samples(n_samples, n_dims, seed)
    elif method == "centered_lhs":
        return _centered_lhs_samples(n_samples, n_dims, seed)
    elif method == "sobol":
        return _sobol_samples(n_samples, n_dims, seed)
    elif method == "grid":
        return _grid_samples(n_samples, n_dims, seed)
    else:
        raise ValueError(f"Unknown sampling method: {method}. "
                        f"Choose from: gaussian, lhs, centered_lhs, sobol, grid")


# =============================================================================
# Shared parameter-space sampling utility
# =============================================================================

#: Parameter keys excluded from parameter-space sampling by default.
#: ``subsidary_params`` are subsidiary pool parameters (not directly tunable),
#: and ``initial_weights_logits`` are typically handled separately by Optuna.
_DEFAULT_EXCLUDE_KEYS = ("subsidary_params", "initial_weights_logits")


def generate_param_space_samples(
    params: Dict[str, Any],
    n_samples: int,
    method: str,
    seed: int = 0,
    exclude_keys: tuple = _DEFAULT_EXCLUDE_KEYS,
) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[int, int, tuple]]]:
    """
    Generate structured samples in the parameter space defined by a params dict.

    Identifies the trainable dimensions across all parameter arrays, generates
    low-discrepancy samples in that joint space, and returns a mapping so callers
    can distribute columns back to individual parameters.

    Parameters
    ----------
    params : Dict[str, Any]
        Parameter dictionary. Values are arrays with shape (n_sets, ...).
        The first dimension is the "set" dimension.
    n_samples : int
        Number of sample points to generate
    method : str
        Sampling method passed to generate_ensemble_samples
    seed : int
        Random seed for reproducibility
    exclude_keys : tuple
        Keys to skip (not perturbed)

    Returns
    -------
    samples : np.ndarray
        Shape (n_samples, total_dims). Values in [0, 1] for structured
        methods, or N(0, 1) for "gaussian".
    trainable_keys : List[str]
        Ordered list of keys that were included
    dim_map : Dict[str, Tuple[int, int, tuple]]
        key -> (start_col, n_dims, shape_per_sample) so callers can slice
        ``samples[:, start_col:start_col + n_dims].reshape((n_samples,) + shape_per_sample)``
    """
    trainable_keys = [
        k for k in params.keys()
        if k not in exclude_keys
        and hasattr(params[k], "shape")
        and len(params[k].shape) > 0
    ]

    dim_map = {}
    col = 0
    for k in trainable_keys:
        shape_per_sample = params[k].shape[1:]  # skip the sets dimension
        n_dims = int(np.prod(shape_per_sample)) if shape_per_sample else 1
        dim_map[k] = (col, n_dims, shape_per_sample)
        col += n_dims

    total_dims = col
    samples = generate_ensemble_samples(n_samples, total_dims, method, seed)

    return samples, trainable_keys, dim_map
