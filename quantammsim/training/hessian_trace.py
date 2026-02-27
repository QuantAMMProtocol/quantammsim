import jax.numpy as jnp
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


def flat_hessian(params_dict, func, exclude_params=None):
    """Compute the full Hessian matrix of ``func`` w.r.t. flattened parameters.

    Flattens ``params_dict`` via :func:`jax.flatten_util.ravel_pytree` and
    calls :func:`jax.hessian` on the resulting 1-D array.  When
    ``exclude_params`` is provided, excluded keys are held constant at their
    values in ``params_dict`` and the Hessian is computed only over the
    remaining (non-excluded) parameters.

    Parameters
    ----------
    params_dict : dict
        Parameter pytree to evaluate at.
    func : callable
        Scalar-valued function that takes a parameter dict.
    exclude_params : list of str, optional
        Parameter keys to hold fixed.  These are stitched back into the
        dict before calling ``func`` but are not differentiated through.

    Returns
    -------
    jnp.ndarray
        Square Hessian matrix of shape ``(D, D)`` where *D* is the total
        number of scalar entries in the non-excluded parameters.
    """
    if exclude_params is None:
        flat_params, _ = ravel_pytree(params_dict)
        flat_fn = make_flat_fn(params_dict, func)
        return hessian(flat_fn)(flat_params)

    filtered_params_dict = {
        k: v for k, v in params_dict.items() if k not in exclude_params
    }
    flat_filtered_params, filtered_tree_structure = ravel_pytree(
        filtered_params_dict
    )
    flat_fn = make_flat_fn(params_dict, func)

    def refill_function(flat_filtered):
        refilled_params_dict = filtered_tree_structure(flat_filtered)
        for param_key in exclude_params:
            refilled_params_dict[param_key] = params_dict[param_key]
        return ravel_pytree(refilled_params_dict)[0]

    return hessian(lambda flat_filtered: flat_fn(refill_function(flat_filtered)))(
        flat_filtered_params
    )


def hessian_trace(params_dict, func, exclude_params=None):
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
    exclude_params : list of str, optional
        Parameter keys to hold fixed (excluded from Hessian computation).

    Returns
    -------
    jnp.ndarray
        Scalar trace of the Hessian.
    """
    hess = flat_hessian(params_dict, func, exclude_params=exclude_params)
    return jnp.trace(hess)


def hessian_frobenius(params_dict, func, exclude_params=None):
    """Compute the Frobenius norm of the Hessian of ``func`` at ``params_dict``.

    Parameters
    ----------
    params_dict : dict
        Parameter pytree to evaluate at.
    func : callable
        Scalar-valued function that takes a parameter dict.
    exclude_params : list of str, optional
        Parameter keys to hold fixed (excluded from Hessian computation).

    Returns
    -------
    jnp.ndarray
        Scalar Frobenius norm of the Hessian.
    """
    hess = flat_hessian(params_dict, func, exclude_params=exclude_params)
    return jnp.linalg.norm(hess, ord="fro")
