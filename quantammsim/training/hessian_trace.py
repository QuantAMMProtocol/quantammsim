import jax.numpy as jnp

# from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree
from jax import hessian


def make_flat_fn(params_dict, func):

    flat_params, tree_structure = ravel_pytree(params_dict)

    def flat_fn(flat_params_dict):
        params_dict = tree_structure(flat_params_dict)
        return func(params_dict)

    return flat_fn


def hessian_trace(params_dict, func):

    flat_params, tree_structure = ravel_pytree(params_dict)
    flat_fn = make_flat_fn(params_dict, func)
    hess = hessian(flat_fn)(flat_params)
    # raise Exception
    trace = jnp.trace(hess)
    return trace
