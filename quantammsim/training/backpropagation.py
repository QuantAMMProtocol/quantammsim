import os
import glob

# BATCH_SIZE=32
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count='+str(BATCH_SIZE)

# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
from jax.lib.xla_bridge import default_backend
from jax import local_device_count, devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

# jax.set_cpu_device_count(n)
# print(devices("cpu"))

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import tree_map
from jax import jacfwd, jacrev, jvp
from jax import devices

import sys

import numpy as np

from quantammsim.training.hessian_trace import hessian_trace
from functools import partial


np.seterr(all="raise")
np.seterr(under="print")

# TODO above is all from jax utils, tidy up required


def objective_factor(partial_training_step):
    """Creates a JIT-compiled objective function from a partial training step.

    This function wraps a partial training step into a simple objective function that can be
    used for optimization. The resulting function is JIT-compiled for performance.

    Parameters
    ----------
    partial_training_step : callable
        A function that takes parameters and start indexes as input and returns some output
        to be optimized.

    Returns
    -------
    callable
        A JIT-compiled objective function that takes parameters and start indexes as input
        and returns the output of the partial training step.
    """

    @jit
    def objective(params, start_indexes):
        output = partial_training_step(params, start_indexes)
        return output

    return objective


def batched_partial_training_step_factory(partial_training_step):
    """Creates a batched version of a partial training step using JAX's vmap.

    This function vectorizes the partial training step to operate on batches of start indexes
    while sharing the same parameters across the batch. The resulting function is JIT-compiled
    for performance.

    Parameters
    ----------
    partial_training_step : callable
        A function that takes parameters and a single start index as input.

    Returns
    -------
    callable
        A JIT-compiled vectorized function that can process batches of start indexes in parallel.
        The parameters are shared across the batch (in_axes=None) while start_indexes are batched
        (in_axes=0).
    """
    batched_partial_training_step = jit(vmap(partial_training_step, in_axes=(None, 0)))
    return batched_partial_training_step


def batched_objective_factory(batched_partial_training_step):
    """Creates an objective function that operates on batched inputs and returns their mean.

    This function wraps a batched partial training step into an objective function that
    computes the mean output across the batch. The resulting function is JIT-compiled
    for performance.

    Parameters
    ----------
    batched_partial_training_step : callable
        A vectorized function that can process batches of inputs in parallel.

    Returns
    -------
    callable
        A JIT-compiled objective function that takes parameters and start indexes as input
        and returns the mean output across the batch.
    """

    @jit
    def batched_objective(params, start_indexes):
        output = batched_partial_training_step(params, start_indexes)
        # print('output shape ', output.shape)
        return jnp.mean(output)

    return batched_objective


def batched_objective_with_hessian_factory(
    batched_partial_training_step, partial_fixed_training_step
):
    """Creates an objective function that combines batched outputs with a Hessian trace regularization term.

    This function creates an objective that adds a weighted Hessian trace term to the mean output
    across the batch. The Hessian trace acts as a regularization term. The weighting parameter
    is treated as a static argument for JIT compilation optimization.

    Parameters
    ----------
    batched_partial_training_step : callable
        A vectorized function that can process batches of inputs in parallel.
    partial_fixed_training_step : callable
        A function used to compute the Hessian trace for regularization.

    Returns
    -------
    callable
        A JIT-compiled objective function that returns the mean batch output plus a weighted
        Hessian trace term. Takes parameters, start indexes, and an optional weighting factor
        (default 1e-4) as input.

    References
    ----------
    For Hessian trace calculation details, see:
    ```python:quantammsim/training/hessian_trace.py
    startLine: 1
    endLine: 26
    ```
    """

    @partial(jit, static_argnums=(2,))
    def batched_objective_with_hessian(params, start_indexes, weighting=1e-4):
        output = batched_partial_training_step(params, start_indexes)
        hessian_trace_fixed = hessian_trace(params, partial_fixed_training_step)
        return jnp.mean(output) + weighting * hessian_trace_fixed
        # return weighting * hessian_trace_fixed

    return batched_objective_with_hessian


def update_factory(batched_objective):
    """Creates an update function for gradient-based optimization.

    This function creates a JIT-compiled update function that performs one step of gradient
    descent optimization. It computes gradients of the objective with respect to parameters
    and updates them using a learning rate.

    Parameters
    ----------
    batched_objective : callable
        The objective function to be optimized.

    Returns
    -------
    callable
        A JIT-compiled update function that takes parameters, start indexes, and learning rate
        as input and returns a tuple containing:
        - Updated parameters after one gradient step
        - Current objective value
        - Original parameters (before update)
        - Computed gradients
    """

    @jit
    def update(params, start_indexes, learning_rate):
        grads = grad(batched_objective)(params, start_indexes)
        return (
            tree_map(lambda p, g: p + learning_rate * g, params, grads),
            batched_objective(params, start_indexes),
            params,
            grads,
        )

    return update


def update_with_hessian_factory(batched_objective_with_hessian):
    """Creates an update function for gradient-based optimization with Hessian regularization.

    Similar to update_factory, but works with an objective function that includes Hessian
    trace regularization. The function is JIT-compiled for performance.

    Parameters
    ----------
    batched_objective_with_hessian : callable
        The objective function with Hessian regularization to be optimized.

    Returns
    -------
    callable
        A JIT-compiled update function that takes parameters, start indexes, and learning rate
        as input and returns a tuple containing:
        - Updated parameters after one gradient step
        - Current objective value (including Hessian term)
        - Original parameters (before update)
        - Computed gradients
    """

    @jit
    def update_with_hessian(params, start_indexes, learning_rate):
        grads = grad(batched_objective_with_hessian)(params, start_indexes)
        return (
            tree_map(lambda p, g: p + learning_rate * g, params, grads),
            batched_objective_with_hessian(params, start_indexes),
            params,
            grads,
        )

    return update_with_hessian


def update_singleton_factory(objective):
    """Creates an update function for non-batched (singleton) gradient-based optimization.

    This function creates a JIT-compiled update function for when batching is not needed
    or desired. It performs one step of gradient descent optimization on a single input.

    Parameters
    ----------
    objective : callable
        The objective function to be optimized.

    Returns
    -------
    callable
        A JIT-compiled update function that takes parameters, start indexes, and learning rate
        as input and returns a tuple containing:
        - Updated parameters after one gradient step
        - Current objective value
        - Original parameters (before update)
        - Computed gradients
    """

    @jit
    def update_singleton(params, start_indexes, learning_rate):
        grads = grad(objective)(params, start_indexes)
        return (
            tree_map(lambda p, g: p + learning_rate * g, params, grads),
            objective(params, start_indexes),
            params,
            grads,
        )

    return update_singleton


def update_from_partial_training_step_factory(
    partial_training_step,
    train_on_hessian_trace=False,
    partial_fixed_training_step=None,
):
    """Creates a complete update function from a partial training step.

    This is a high-level factory function that combines the other factories to create
    a complete update function. It handles both regular training and training with
    Hessian trace regularization.

    Parameters
    ----------
    partial_training_step : callable
        The base training step function to be wrapped.
    train_on_hessian_trace : bool, optional
        Whether to include Hessian trace regularization, by default False.
    partial_fixed_training_step : callable, optional
        The function used to compute Hessian trace when train_on_hessian_trace is True.
        Required if train_on_hessian_trace is True.

    Returns
    -------
    callable
        A JIT-compiled update function that implements the complete training step,
        either with or without Hessian regularization.
    """
    batched_partial_training_step = batched_partial_training_step_factory(
        partial_training_step
    )

    if train_on_hessian_trace:
        batched_objective_with_hessian = batched_objective_with_hessian_factory(
            batched_partial_training_step, partial_fixed_training_step
        )
        update = update_with_hessian_factory(batched_objective_with_hessian)
    else:
        batched_objective = batched_objective_factory(batched_partial_training_step)
        update = update_factory(batched_objective)
    return update


def hessian(fun):
    """Creates a JIT-compiled function to compute the Hessian matrix.

    Uses JAX's forward-over-reverse automatic differentiation to compute
    the Hessian matrix efficiently.

    Parameters
    ----------
    fun : callable
        The function whose Hessian is to be computed.

    Returns
    -------
    callable
        A JIT-compiled function that computes the Hessian matrix of the input function.
    """
    return jit(jacfwd(jacrev(fun)))


@jit
def hvp(f, primals, tangents):
    """Computes a Hessian-vector product efficiently using JAX's JVP of gradients.

    This function implements the Hessian-vector product without explicitly constructing
    the full Hessian matrix, which can be more efficient for large-scale problems.

    Parameters
    ----------
    f : callable
        The function whose Hessian-vector product is to be computed.
    primals : array_like
        The point at which to evaluate the Hessian-vector product.
    tangents : array_like
        The vector to multiply with the Hessian.

    Returns
    -------
    array_like
        The Hessian-vector product at the specified point.
    """
    return jvp(grad(f), primals, tangents)[1]
