import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import hessian


def make_flat_fn(params_dict, func):
    flat_params, tree_structure = ravel_pytree(params_dict)

    def flat_fn(flat_params_dict):
        params_dict = tree_structure(flat_params_dict)
        return func(params_dict)

    return flat_fn


def flat_hessian(params_dict, func, exclude_params=None):
    """Compute the Hessian of func w.r.t. flattened params.

    When exclude_params is provided, the Hessian is computed only over the
    non-excluded parameters, with excluded parameters held fixed at their
    values in params_dict.
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
    hess = flat_hessian(params_dict, func, exclude_params=exclude_params)
    return jnp.trace(hess)


def hessian_frobenius(params_dict, func, exclude_params=None):
    hess = flat_hessian(params_dict, func, exclude_params=exclude_params)
    return jnp.linalg.norm(hess, ord="fro")
