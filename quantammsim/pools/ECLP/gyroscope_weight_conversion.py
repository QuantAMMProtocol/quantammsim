"""Conversion between ECLP parameters and G3M-style weights.

Provides routines to find the ECLP parameters (lambda, tan_phi) that
achieve a target token weight, bridging the Gyroscope ECLP parameterisation
with the weight-based interface used by the rest of the simulator. Uses
grid search followed by gradient descent in an unconstrained (softplus)
parameter space.
"""
from jax import numpy as jnp
from jax import jit, grad, value_and_grad
from typing import Tuple
from quantammsim.pools.ECLP.gyroscope_reserves import initialise_gyroscope_reserves_given_value
import jax
from functools import partial


@jit
def safe_softplus(x: float) -> float:
    """
    Numerically stable softplus.
    For small x: log(1 + exp(x))
    For large x: approximately x
    """
    threshold = 30.0
    return jnp.where(
        x > threshold,
        x,  # identity for large values
        jax.nn.softplus(x),  # softplus for small values
    )


@jit
def safe_inverse_softplus(x: float) -> float:
    """
    Numerically stable inverse softplus.
    For small x: log(exp(x) - 1)
    For large x: approximately x
    """
    threshold = 30.0
    return jnp.where(
        x > threshold,
        x,  # identity for large values
        jnp.log(jnp.exp(x) - 1.0),  # inverse softplus for small values
    )


@jit
def transform_params(unconstrained_params: jnp.ndarray) -> jnp.ndarray:
    """Transform unconstrained parameters to constrained space."""
    x1, x2 = unconstrained_params
    lambda_constrained = 1.0 + safe_softplus(x1)
    tan_phi_constrained = safe_softplus(x2)
    return jnp.array([lambda_constrained, tan_phi_constrained])


@jit
def inverse_transform_params(constrained_params: jnp.ndarray) -> jnp.ndarray:
    """Transform constrained parameters to unconstrained space."""
    lam, tan_phi = constrained_params
    x1 = safe_inverse_softplus(lam - 1.0)
    x2 = safe_inverse_softplus(tan_phi)
    return jnp.array([x1, x2])


@jit
def calculate_weight(
    params: jnp.ndarray,  # [lambda, tan_phi]
    initial_pool_value: float,
    initial_prices: jnp.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Calculate first token weight for given parameters."""
    lam, tan_phi = params
    phi = jnp.arctan(tan_phi)

    reserves = initialise_gyroscope_reserves_given_value(
        initial_pool_value, initial_prices, alpha, beta, lam, jnp.sin(phi), jnp.cos(phi)
    )

    value = reserves * initial_prices
    return value[0] / jnp.sum(value)


@partial(jit, static_argnums=(4,))
def explore_weight_space(
    initial_pool_value: float,
    initial_prices: jnp.ndarray,
    alpha: float,
    beta: float,
    n_points: int = 50,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Calculate first token weight for a grid of (lam, tan_phi) values.

    Parameters
    ----------
    initial_pool_value : float
        Initial pool value
    initial_prices : jnp.ndarray
        Initial prices [p1, p2]
    alpha : float
        Lower price bound
    beta : float
        Upper price bound
    n_points : int
        Number of points in each dimension

    Returns
    -------
    lam_grid : jnp.ndarray
        2D array of lambda values
    tan_phi_grid : jnp.ndarray
        2D array of tan(phi) values
    weights : jnp.ndarray
        2D array of first token weights
    """
    # Create grids for lam and tan_phi
    lam_values = jnp.geomspace(1.1, beta**2.0, n_points)  # lam > 1

    # tan_phi represents peg price, so let's center around current price
    current_price = initial_prices[0] / initial_prices[1]
    tan_phi_min = alpha * 0.5  # 50% of low price
    tan_phi_max = beta * 1.5  # 150% of high price
    tan_phi_values = jnp.geomspace(tan_phi_min, tan_phi_max, n_points)

    # Calculate weights for each point

    # Create a function that takes both parameters
    @jit
    def single_weight(lam, tan_phi):
        return calculate_weight(
            jnp.array([lam, tan_phi]), initial_pool_value, initial_prices, alpha, beta
        )

    # Vectorize over both parameters
    vectorized_weight = jax.vmap(
        jax.vmap(single_weight, in_axes=(None, 0)),  # Inner vmap over tan_phi
        in_axes=(0, None),  # Outer vmap over lambda
    )

    # Calculate all weights at once
    weights = vectorized_weight(lam_values, tan_phi_values)

    return lam_values, tan_phi_values, weights


@jit
def objective(
    params: jnp.ndarray,  # [lambda, tan_phi]
    target_weight: float,
    initial_pool_value: float,
    initial_prices: jnp.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Objective function to minimize."""
    lam, tan_phi = params

    # L2 norm
    l2_norm = jnp.sqrt(lam**2 + tan_phi**2)

    # Weight constraint
    weight = calculate_weight(params, initial_pool_value, initial_prices, alpha, beta)
    weight_error = (weight - target_weight) ** 2

    return 1000000.0 * weight_error + l2_norm

@jit
def loss_fn(
    unconstrained_params: jnp.ndarray,  # [lambda, tan_phi]
    target_weight: float,
    initial_pool_value: float,
    initial_prices: jnp.ndarray,
    alpha: float,
    beta: float,
) -> float:
    params = transform_params(unconstrained_params)
    return objective(
        params, target_weight, initial_pool_value, initial_prices, alpha, beta
    )

@jit
def update_params(
    params: jnp.ndarray, grads: jnp.ndarray, step_size: float
) -> jnp.ndarray:
    """Simple gradient descent update."""
    return params - step_size * grads


@jit
def optimization_step(
    i: int,  # unused but required by fori_loop
    carry: Tuple[jnp.ndarray, float],
    target_weight: float,
    initial_pool_value: float,
    initial_prices: jnp.ndarray,
    alpha: float,
    beta: float,
    step_size: float,
) -> Tuple[jnp.ndarray, float]:
    """Single optimization step returning (params, loss)."""
    params, _ = carry
    loss, grads = value_and_grad(loss_fn)(
        params, target_weight, initial_pool_value, initial_prices, alpha, beta
    )
    new_params = params - step_size * grads
    return (new_params, loss)


@jit
def optimize_lambda_and_tan_phi(
    target_weight: float,
    initial_pool_value: float,
    initial_prices: jnp.ndarray,
    alpha: float,
    beta: float,
    n_steps: int = 10000,
    step_size: float = 0.001,
) -> Tuple[float, float]:
    """Find optimal lambda and tan_phi parameters for a Gyroscope ECLP pool.

    This function uses a two-step optimization process to find parameters that achieve a target weight:

    1. Grid Search: First explores a coarse grid of (lambda, tan_phi) values to find a good initial guess
    2. Gradient Descent: Then refines this guess using gradient descent via fori_loop

    The optimization minimizes an objective function that balances:
    - Weight error: How close the achieved weight is to the target weight
    - Parameter magnitude: Prefers smaller parameter values via L2 regularization

    Parameters
    ----------
    target_weight : float
        Desired weight of the first token (between 0 and 1)
    initial_pool_value : float
        Total initial value of the pool in terms of token2
    initial_prices : jnp.ndarray
        Initial prices [p1, p2] of both tokens
    alpha : float
        Lower price bound for the pool
    beta : float
        Upper price bound for the pool
    n_steps : int, optional
        Number of gradient descent steps, by default 10000
    step_size : float, optional
        Learning rate for gradient descent, by default 0.001

    Returns
    -------
    Tuple[float, float]
        Optimal (lambda, tan_phi) parameters that achieve the target weight
    """
    # Initial guess
    current_price = initial_prices[0] / initial_prices[1]
    initial_params = jnp.array([0.5 * (alpha + beta), 0.5 * (alpha + beta)])
    initial_loss = 0.0

    # first calculate the weights over a grid of (lam, tan_phi) values
    lam_values, tan_phi_values, weights = explore_weight_space(
        initial_pool_value, initial_prices, alpha, beta, n_points=10
    )

    # find the optimal lam and tan_phi
    optimal_lam, optimal_tan_phi = jnp.unravel_index(
        jnp.argmin(jnp.abs(weights - target_weight)), weights.shape
    )
    initial_params = jnp.array(
        [lam_values[optimal_lam], tan_phi_values[optimal_tan_phi]]
    )
    initial_unconstrained_params = inverse_transform_params(initial_params)
    initial_weight = weights[optimal_lam, optimal_tan_phi]
    # Run optimization using fori_loop
    final_unconstrained_params, final_loss = jax.lax.fori_loop(
        0,  # lower
        n_steps,  # upper
        lambda i, carry: optimization_step(
            i,
            carry,
            target_weight,
            initial_pool_value,
            initial_prices,
            alpha,
            beta,
            step_size,
        ),
        (initial_unconstrained_params, initial_loss),  # initial carry
    )

    final_params = transform_params(final_unconstrained_params)
    # Calculate the final weight using the optimized parameters
    final_weight = calculate_weight(
        final_params,
        initial_pool_value,
        initial_prices,
        alpha,
        beta,
    )

    # Check if the final weight is closer to the target weight than the initial weight.
    # This is insurance against the gradient descent getting stuck in a local minimum
    # or otherwise going off the rails.
    optimal_params = jnp.where(
        jnp.abs(final_weight - target_weight) < jnp.abs(initial_weight - target_weight),
        final_params,
        initial_params,
    )
    return optimal_params[0], optimal_params[1]
