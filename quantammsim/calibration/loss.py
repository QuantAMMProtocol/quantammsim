"""Per-pool loss function for the direct calibration pipeline.

JAX-differentiable loss: sum((log(V_arb_i + V_noise_i) - log(V_obs_i))^2)
where V_arb comes from per-day grid interpolation and V_noise from
log-linear regression on observation covariates.
"""

from typing import Tuple

import jax.numpy as jnp

from quantammsim.calibration.grid_interpolation import (
    PoolCoeffsDaily,
    interpolate_pool_daily,
)

K_OBS = 8  # observation-level covariates


def noise_volume(
    noise_coeffs: jnp.ndarray, x_obs: jnp.ndarray
) -> jnp.ndarray:
    """V_noise = exp(x_obs @ noise_coeffs). Shape: (n_obs,)."""
    return jnp.exp(x_obs @ noise_coeffs)


def pack_params(
    log_cadence: float, log_gas: float, noise_coeffs: jnp.ndarray
) -> jnp.ndarray:
    """Pack into flat array: [log_cadence, log_gas, noise_coeffs...]."""
    return jnp.concatenate([
        jnp.array([log_cadence, log_gas]),
        jnp.asarray(noise_coeffs),
    ])


def unpack_params(
    flat: jnp.ndarray,
) -> Tuple[float, float, jnp.ndarray]:
    """Unpack flat array to (log_cadence, log_gas, noise_coeffs)."""
    return flat[0], flat[1], flat[2:]


def pool_loss(
    params_flat: jnp.ndarray,
    coeffs: PoolCoeffsDaily,
    x_obs: jnp.ndarray,
    y_obs: jnp.ndarray,
    day_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Per-pool log-space L2 loss with per-day V_arb.

    Args:
        params_flat: [log_cadence, log_gas, noise_coeffs...] from pack_params
        coeffs: PoolCoeffsDaily with per-day grid values
        x_obs: (n_obs, K_OBS) observation covariates
        y_obs: (n_obs,) log(V_obs) — observed log volume
        day_indices: (n_obs,) int indices mapping panel rows to grid days

    Returns:
        Scalar mean squared error in log space.
    """
    log_cadence, log_gas, noise_coeffs = unpack_params(params_flat)

    # Per-day V_arb from grid interpolation
    v_arb_all = interpolate_pool_daily(coeffs, log_cadence, jnp.exp(log_gas))  # (n_days,)
    v_arb = v_arb_all[day_indices]  # (n_obs,)

    # Per-day V_noise from covariates
    v_noise = noise_volume(noise_coeffs, x_obs)  # (n_obs,)

    # Log-space L2 loss
    log_v_pred = jnp.log(jnp.maximum(v_arb + v_noise, 1e-6))
    return jnp.mean((log_v_pred - y_obs) ** 2)
