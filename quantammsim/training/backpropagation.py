import os
import glob

# BATCH_SIZE=32
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count='+str(BATCH_SIZE)

# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
from jax import default_backend
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
from jax import grad, value_and_grad, jit, vmap
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax import jacfwd, jacrev, jvp
from jax import devices

import sys

import numpy as np

from quantammsim.training.hessian_trace import hessian_trace
from functools import partial

import optax


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
        objective_value, grads = value_and_grad(batched_objective)(params, start_indexes)
        return (
            tree_map(lambda p, g: p + learning_rate * g, params, grads),
            objective_value,
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
        objective_value, grads = value_and_grad(batched_objective_with_hessian)(params, start_indexes)
        return (
            tree_map(lambda p, g: p + learning_rate * g, params, grads),
            objective_value,
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
        objective_value, grads = value_and_grad(objective)(params, start_indexes)
        return (
            tree_map(lambda p, g: p + learning_rate * g, params, grads),
            objective_value,
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


def update_factory_with_optax(batched_objective, optimizer):
    """Creates an update function using optax optimizer.

    This function creates a JIT-compiled update function that uses an optax optimizer
    while maintaining the same interface as the other update functions.

    Parameters
    ----------
    batched_objective : callable
        The objective function to be optimized.
    optimizer : optax.GradientTransformation
        The optax optimizer to use.

    Returns
    -------
    callable
        A JIT-compiled update function that takes parameters, start indexes, and learning rate
        as input and returns a tuple containing:
        - Updated parameters after one optimizer step
        - Current objective value
        - Original parameters (before update)
        - Computed gradients
        - Optimizer state (for maintaining across iterations)
    """

    @jit
    def update(params, start_indexes, learning_rate, opt_state=None):
        objective_value, grads = value_and_grad(batched_objective)(params, start_indexes)        
        # Initialize optimizer state if not provided
        if opt_state is None:
            opt_state = optimizer.init(params)

        neg_grads = tree_map(lambda g: -g, grads)

        # Apply optimizer update, cast to float32 to avoid type errors as optax doesn't use float64 internally for state
        updates, new_opt_state = optimizer.update(
            neg_grads,
            opt_state,
            params,
            value=jnp.array(-objective_value, dtype=jnp.float32),
        )
        new_params = optax.apply_updates(params, updates)

        return (
            new_params,
            objective_value,
            params,
            grads,
            new_opt_state,
        )

    return update


def update_with_hessian_factory_with_optax(batched_objective_with_hessian, optimizer):
    """Creates an update function using optax optimizer with Hessian regularization.

    Similar to update_factory_with_optax, but works with an objective function that includes Hessian
    trace regularization.

    Parameters
    ----------
    batched_objective_with_hessian : callable
        The objective function with Hessian regularization to be optimized.
    optimizer : optax.GradientTransformation
        The optax optimizer to use.

    Returns
    -------
    callable
        A JIT-compiled update function that takes parameters, start indexes, and learning rate
        as input and returns a tuple containing:
        - Updated parameters after one optimizer step
        - Current objective value (including Hessian term)
        - Original parameters (before update)
        - Computed gradients
        - Optimizer state (for maintaining across iterations)
    """

    @jit
    def update_with_hessian(params, start_indexes, learning_rate, opt_state=None):
        objective_value, grads = value_and_grad(batched_objective_with_hessian)(params, start_indexes)
        # Initialize optimizer state if not provided
        if opt_state is None:
            opt_state = optimizer.init(params)

        neg_grads = tree_map(lambda g: -g, grads)

        # Apply optimizer update, cast to float32 to avoid type errors as optax doesn't use float64 internally for state
        updates, new_opt_state = optimizer.update(
            neg_grads,
            opt_state,
            params,
            value=jnp.array(-objective_value, dtype=jnp.float32),
        )
        new_params = optax.apply_updates(params, updates)

        return (
            new_params,
            objective_value,
            params,
            grads,
            new_opt_state,
        )

    return update_with_hessian


def update_from_partial_training_step_factory_with_optax(
    partial_training_step,
    optimizer,
    train_on_hessian_trace=False,
    partial_fixed_training_step=None,
):
    """Creates a complete update function from a partial training step using optax optimizer.

    This is a high-level factory function that combines the other factories to create
    a complete update function using optax optimizers.

    Parameters
    ----------
    partial_training_step : callable
        The base training step function to be wrapped.
    optimizer : optax.GradientTransformation
        The optax optimizer to use.
    train_on_hessian_trace : bool, optional
        Whether to include Hessian trace regularization, by default False.
    partial_fixed_training_step : callable, optional
        The function used to compute Hessian trace when train_on_hessian_trace is True.
        Required if train_on_hessian_trace is True.

    Returns
    -------
    callable
        A JIT-compiled update function that implements the complete training step,
        either with or without Hessian regularization, using the specified optax optimizer.
    """
    batched_partial_training_step = batched_partial_training_step_factory(
        partial_training_step
    )

    if train_on_hessian_trace:
        batched_objective_with_hessian = batched_objective_with_hessian_factory(
            batched_partial_training_step, partial_fixed_training_step
        )
        update = update_with_hessian_factory_with_optax(batched_objective_with_hessian, optimizer)
    else:
        batched_objective = batched_objective_factory(batched_partial_training_step)
        update = update_factory_with_optax(batched_objective, optimizer)
    return update


def create_opt_state_in_axes_dict(opt_state):
    """Create in_axes dict for optimizer state based on its actual structure."""

    def _create_axes_for_leaf(leaf):
        # Handle empty lists specifically - they should not be vmapped over
        if isinstance(leaf, list) and len(leaf) == 0:
            return None
        elif hasattr(leaf, "shape") and len(leaf.shape) > 0:
            # If first dimension >= 1, it's batched (map over first dimension)
            if leaf.shape[0] >= 1:  # Changed from > 1 to >= 1
                return 0
            else:
                return None
        elif hasattr(leaf, "__len__") and len(leaf) == 0:
            # Any empty sequence - don't map over
            return None
        else:
            # Other types (like EmptyState) - don't map over
            return None

    return tree_map(_create_axes_for_leaf, opt_state)


def _create_base_optimizer(optimizer_type, learning_rate, weight_decay=0.0):
    """Create a base optimizer with the given learning rate.

    Parameters
    ----------
    optimizer_type : str
        One of "adam", "adamw", or "sgd"
    learning_rate : float or optax schedule
        Learning rate or schedule
    weight_decay : float
        Weight decay coefficient for adamw (default 0.0)
    """
    if optimizer_type == "adam":
        return optax.adam(learning_rate=learning_rate)
    elif optimizer_type == "adamw":
        # AdamW applies weight decay directly to weights, not through gradients
        # This is more principled than L2 reg with Adam
        return optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        return optax.sgd(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def _create_lr_schedule(settings):
    """Create a learning rate schedule based on settings.

    Supports two ways to specify the minimum LR for decay schedules:
    - lr_decay_ratio: min_lr = base_lr / lr_decay_ratio (preferred, scale-invariant)
    - min_lr: absolute minimum LR (fallback for backwards compatibility)

    If both are provided, lr_decay_ratio takes precedence.
    """
    base_lr = settings.get("base_lr", 0.001)
    n_iterations = settings.get("n_iterations", 1000)
    schedule_type = settings.get("lr_schedule_type", "constant")

    # Compute min_lr: prefer lr_decay_ratio if provided, else use min_lr directly
    if "lr_decay_ratio" in settings:
        min_lr = base_lr / settings["lr_decay_ratio"]
    else:
        min_lr = settings.get("min_lr", 1e-6)
        # Safety check: ensure min_lr < base_lr for decay schedules
        if schedule_type != "constant" and min_lr >= base_lr:
            min_lr = base_lr / 100  # Fallback to 100:1 ratio

    if schedule_type == "constant":
        return optax.constant_schedule(base_lr)

    elif schedule_type == "cosine":
        if n_iterations <= 0:
            raise ValueError(f"cosine schedule requires positive n_iterations, got {n_iterations}")
        return optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=n_iterations,  # Use n_iterations
            alpha=min_lr / base_lr,
        )

    elif schedule_type == "exponential":
        if n_iterations <= 0:
            raise ValueError(f"exponential schedule requires positive n_iterations, got {n_iterations}")
        # Decay from base_lr to min_lr over n_iterations steps.
        # Formula: LR(step) = base_lr * decay_rate^step
        # At step=n_iterations: min_lr = base_lr * decay_rate^n_iterations
        # So: decay_rate = (min_lr / base_lr)^(1/n_iterations)
        decay_rate = (min_lr / base_lr) ** (1.0 / n_iterations)
        return optax.exponential_decay(
            init_value=base_lr,
            transition_steps=1,  # Apply decay at every step
            decay_rate=decay_rate
        )

    elif schedule_type == "warmup_cosine":
        if n_iterations <= 0:
            raise ValueError(f"warmup_cosine schedule requires positive n_iterations, got {n_iterations}")
        warmup_steps = settings["warmup_steps"]
        if warmup_steps >= n_iterations:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than n_iterations ({n_iterations}). "
                f"Use warmup_fraction in HyperparamSpace to avoid this."
            )
        return optax.warmup_cosine_decay_schedule(
            init_value=min_lr,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=n_iterations,
            end_value=min_lr,
        )

    else:
        raise ValueError(f"Unknown learning rate schedule type: {schedule_type}")


def create_optimizer_chain(run_fingerprint):
    settings = run_fingerprint["optimisation_settings"]
    weight_decay = settings.get("weight_decay", 0.0)  # Default to no weight decay

    # Create base optimizer with lr=1.0 - the schedule will control the actual LR
    base_optimizer = _create_base_optimizer(settings["optimiser"], 1.0, weight_decay)

    # Create vanilla LR schedule
    lr_schedule = _create_lr_schedule(settings)

    # Build base optimizer chain
    optimizer_chain = optax.chain(base_optimizer, optax.scale_by_schedule(lr_schedule))

    # Add plateau reduction if enabled
    if settings["use_plateau_decay"]:
        # Use atol (absolute tolerance) instead of default rtol (relative tolerance)
        # because we pass -objective_value (negative values) for maximization.
        # rtol compares value < best * (1 - rtol) which misbehaves for negative values.
        plateau_reduction = optax.contrib.reduce_on_plateau(
            factor=settings["decay_lr_ratio"],
            patience=settings["decay_lr_plateau"],
            rtol=0.0,
            atol=1e-4,
        )
        optimizer_chain = optax.chain(optimizer_chain, plateau_reduction)

    # Add gradient clipping if enabled
    if settings["use_gradient_clipping"]:
        optimizer_chain = optax.chain(
            optax.clip_by_global_norm(settings["clip_norm"]), optimizer_chain
        )

    return optimizer_chain
