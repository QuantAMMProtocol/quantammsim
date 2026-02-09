"""Path augmentations for signature computation in Sig-W1 training.

Applied to windows of log-returns before computing their path signature.
All functions operate on a single path (L, d) and are designed to be
vmap'd over batches. Pure JAX, JIT-compatible.

Reference: Ni, Szpruch, Wiese, Liao (2021) — "Sig-Wasserstein GANs for
Time Series Generation", Section 3.2 on augmentations.

Key insight: the path signature is invariant to time-reparameterisation,
so augmentations like lead-lag are needed to break this invariance and
capture temporal structure (autocorrelations, vol clustering).
"""

from typing import Callable

import jax.numpy as jnp


def scale(path: jnp.ndarray, scale_factor: float = 1.0) -> jnp.ndarray:
    """Scale path values by a constant factor.

    Useful for normalising paths to a common scale before signature computation,
    preventing numerical issues when signature terms grow exponentially with depth.

    Args:
        path: (L, d) array.
        scale_factor: Multiplicative constant.

    Returns:
        (L, d) scaled path.
    """
    return path * scale_factor


def cumsum(path: jnp.ndarray) -> jnp.ndarray:
    """Cumulative sum along the time axis.

    Converts increments (e.g. log-returns) into levels (e.g. cumulative log-returns).
    The signature of the cumsum'd path captures different information than
    the signature of the raw increments.

    Args:
        path: (L, d) array of increments.

    Returns:
        (L, d) cumulative sum.
    """
    return jnp.cumsum(path, axis=0)


def add_time(path: jnp.ndarray) -> jnp.ndarray:
    """Append a normalised time channel.

    Breaks time-reparameterisation invariance of the signature by making
    the path explicitly time-parameterised. The time channel runs linearly
    from 0 to 1.

    Args:
        path: (L, d) array.

    Returns:
        (L, d+1) array with time channel appended.
    """
    L = path.shape[0]
    t = jnp.linspace(0.0, 1.0, L)[:, None]  # (L, 1)
    return jnp.concatenate([path, t], axis=1)


def lead_lag(path: jnp.ndarray) -> jnp.ndarray:
    """Lead-lag transform.

    The lead-lag embedding doubles the dimension and produces a path of
    length 2L-1. It captures quadratic variation (and hence volatility
    information) in the first-level signature, which the raw path cannot.

    Construction:
        - Repeat each row twice: [x_0, x_0, x_1, x_1, ..., x_{L-1}, x_{L-1}]
        - Lead = repeated[1:]  (shifts forward by one position)
        - Lag  = repeated[:-1] (original timing)
        - Concatenate along feature axis: (2L-1, 2d)

    Args:
        path: (L, d) array.

    Returns:
        (2L-1, 2d) lead-lag transformed path.
    """
    # (L, d) -> (2L, d) by repeating each row
    repeated = jnp.repeat(path, 2, axis=0)
    lead = repeated[1:]   # (2L-1, d)
    lag = repeated[:-1]   # (2L-1, d)
    return jnp.concatenate([lag, lead], axis=1)


def compose_augmentations(*fns: Callable) -> Callable:
    """Compose augmentation functions left-to-right.

    compose_augmentations(f, g, h)(path) == h(g(f(path)))

    Args:
        *fns: Augmentation functions (path -> path).

    Returns:
        Composed function.
    """
    def composed(path: jnp.ndarray) -> jnp.ndarray:
        for fn in fns:
            path = fn(path)
        return path
    return composed


def get_minimal_augmentation(s: float = 1.0) -> Callable:
    """Lead-lag only (with optional pre-scaling).

    Minimal augmentation that captures quadratic variation. Sufficient
    for most use cases where vol structure is the primary target.

    Args:
        s: Scale factor applied before lead-lag.

    Returns:
        Augmentation function (L, d) -> (2L-1, 2d).
    """
    if s != 1.0:
        return compose_augmentations(lambda p: scale(p, s), lead_lag)
    return lead_lag


def get_standard_augmentation(s: float = 1.0) -> Callable:
    """Scale -> cumsum -> add_time -> lead_lag.

    Richer augmentation that captures both level and increment structure,
    plus explicit time dependence.

    Args:
        s: Scale factor applied first.

    Returns:
        Augmentation function (L, d) -> (2L-1, 2(d+1)).
    """
    return compose_augmentations(
        lambda p: scale(p, s),
        cumsum,
        add_time,
        lead_lag,
    )
