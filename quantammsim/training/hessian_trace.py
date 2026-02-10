import jax.numpy as jnp

# from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
from jax import hessian


def make_flat_fn(params_dict, func):
    """Create a flat-parameter wrapper around a dict-parameterised function.

    Uses :func:`jax.flatten_util.ravel_pytree` to flatten ``params_dict``
    into a 1-D array and returns a closure that unflattens before calling
    ``func``.  This is needed because :func:`jax.hessian` operates on a
    single array argument.

    Parameters
    ----------
    params_dict : dict
        Parameter pytree (e.g. ``{"logit_lamb": ..., "k": ...}``).
    func : callable
        Scalar-valued function that takes a parameter dict.

    Returns
    -------
    callable
        ``flat_fn(flat_params) -> scalar``, suitable for passing to
        :func:`jax.hessian`.
    """
    flat_params, tree_structure = ravel_pytree(params_dict)

    def flat_fn(flat_params_dict):
        params_dict = tree_structure(flat_params_dict)
        return func(params_dict)

    return flat_fn


def hessian_trace(params_dict, func):
    """Compute the trace of the Hessian of ``func`` at ``params_dict``.

    Flattens the parameter dict, computes the full Hessian matrix via
    :func:`jax.hessian`, and returns its trace.  The trace of the Hessian
    measures the total curvature of the loss landscape and can be used as
    a regulariser or as a diagnostic for training stability.

    Parameters
    ----------
    params_dict : dict
        Parameter pytree to evaluate at.
    func : callable
        Scalar-valued function that takes a parameter dict.

    Returns
    -------
    jnp.ndarray
        Scalar trace of the Hessian.
    """
    flat_params, tree_structure = ravel_pytree(params_dict)
    flat_fn = make_flat_fn(params_dict, func)
    hess = hessian(flat_fn)(flat_params)
    trace = jnp.trace(hess)
    return trace
