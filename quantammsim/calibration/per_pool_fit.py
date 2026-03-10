"""Per-pool fitting via L-BFGS-B for the direct calibration pipeline.

Fits (log_cadence, log_gas, noise_coeffs) per pool by minimizing
the log-space L2 loss using scipy.optimize.minimize with JAX gradients.
"""

from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from quantammsim.calibration.grid_interpolation import PoolCoeffsDaily
from quantammsim.calibration.loss import K_OBS, pack_params, pool_loss
from quantammsim.calibration.pool_data import build_x_obs


def make_initial_guess(x_obs: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    """Initial params: cadence=12min, gas=$1, noise_coeffs from OLS.

    OLS: noise_coeffs = lstsq(x_obs, y_obs) — assumes all volume is noise.
    This overestimates noise but gives a reasonable starting point.
    """
    noise_coeffs, _, _, _ = np.linalg.lstsq(x_obs, y_obs, rcond=None)
    init = np.zeros(2 + K_OBS)
    init[0] = np.log(12.0)   # log_cadence
    init[1] = np.log(1.0)    # log_gas (= 0.0)
    init[2:] = noise_coeffs
    return init


def fit_single_pool(
    coeffs: PoolCoeffsDaily,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    day_indices: np.ndarray,
    init: Optional[np.ndarray] = None,
    bounds: Optional[dict] = None,
) -> dict:
    """Fit (log_cadence, log_gas, noise_coeffs) for one pool via L-BFGS-B.

    Returns dict with fitted params, loss, and convergence status.
    """
    if init is None:
        init = make_initial_guess(x_obs, y_obs)

    # Default bounds
    if bounds is None:
        bounds = {}
    log_cad_bounds = bounds.get("log_cadence", (np.log(1.0), np.log(60.0)))
    log_gas_bounds = bounds.get("log_gas", (np.log(0.001), np.log(50.0)))
    noise_bounds = bounds.get("noise_coeffs", (-20.0, 20.0))

    scipy_bounds = [
        log_cad_bounds,
        log_gas_bounds,
    ] + [(noise_bounds[0], noise_bounds[1])] * K_OBS

    # Convert to JAX arrays
    x_obs_j = jnp.array(x_obs)
    y_obs_j = jnp.array(y_obs)
    day_idx_j = jnp.array(day_indices)

    # Value and gradient function
    @jax.jit
    def loss_and_grad(params_flat):
        loss = pool_loss(params_flat, coeffs, x_obs_j, y_obs_j, day_idx_j)
        grad = jax.grad(pool_loss, argnums=0)(
            params_flat, coeffs, x_obs_j, y_obs_j, day_idx_j
        )
        return loss, grad

    def scipy_wrapper(params_np):
        params_j = jnp.array(params_np)
        loss, grad = loss_and_grad(params_j)
        return float(loss), np.array(grad, dtype=np.float64)

    result = scipy.optimize.minimize(
        scipy_wrapper,
        init,
        method="L-BFGS-B",
        jac=True,
        bounds=scipy_bounds,
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8},
    )

    log_cadence = float(result.x[0])
    log_gas = float(result.x[1])
    noise_coeffs = np.array(result.x[2:])

    return {
        "log_cadence": log_cadence,
        "log_gas": log_gas,
        "noise_coeffs": noise_coeffs,
        "loss": float(result.fun),
        "converged": result.success,
        "cadence_minutes": float(np.exp(log_cadence)),
        "gas_usd": float(np.exp(log_gas)),
    }


def fit_all_pools(
    matched: Dict[str, dict],
    n_workers: int = 1,
) -> Dict[str, dict]:
    """Fit all matched pools. Returns prefix -> fit_result with metadata."""
    results = {}

    for prefix, entry in matched.items():
        panel = entry["panel"]
        coeffs = entry["coeffs"]
        day_indices = entry["day_indices"]

        x_obs = build_x_obs(panel)
        y_obs = panel["log_volume"].values.astype(float)

        result = fit_single_pool(coeffs, x_obs, y_obs, day_indices)

        # Add metadata
        result["chain"] = entry["chain"]
        result["fee"] = entry["fee"]
        result["tokens"] = entry["tokens"]

        results[prefix] = result

    return results
