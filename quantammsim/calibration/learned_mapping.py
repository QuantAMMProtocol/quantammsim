"""Learned mapping: pool attributes -> (cadence, gas, noise_coeffs).

Ridge regression from pool-level features to per-pool fitted parameters.
Trained on per-pool fit results from per_pool_fit.py; used to predict
simulator settings for new/hypothetical pools.
"""

from typing import Dict, List

import numpy as np

from quantammsim.calibration.loss import K_OBS


def build_targets(
    fit_results: Dict[str, dict],
    pool_order: List[str],
) -> np.ndarray:
    """Stack per-pool fitted params into (n_pools, 2+K_OBS) target matrix.

    Columns: [log_cadence, log_gas, noise_coeffs...]
    Row ordering matches pool_order.
    """
    n_pools = len(pool_order)
    Y = np.zeros((n_pools, 2 + K_OBS))
    for i, pid in enumerate(pool_order):
        r = fit_results[pid]
        Y[i, 0] = r["log_cadence"]
        Y[i, 1] = r["log_gas"]
        Y[i, 2:] = r["noise_coeffs"]
    return Y


def fit_mapping(
    X_attr: np.ndarray,
    Y_target: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Fit Ridge regression: X_attr -> Y_target.

    Multi-output Ridge: one shared regularization strength across all
    target columns.

    Returns dict with weights, intercept, and diagnostics.
    """
    # Ridge with intercept: center Y, solve on centered data
    n, k = X_attr.shape
    Y_mean = Y_target.mean(axis=0)
    X_mean = X_attr.mean(axis=0)
    Xc = X_attr - X_mean
    Yc = Y_target - Y_mean

    # W = (Xc^T Xc + alpha * I)^{-1} Xc^T Yc
    A = Xc.T @ Xc + alpha * np.eye(k)
    W = np.linalg.solve(A, Xc.T @ Yc)  # (K_attr, K_target)
    intercept = Y_mean - X_mean @ W  # (K_target,)

    Y_pred = X_attr @ W + intercept
    ss_res = np.sum((Y_target - Y_pred) ** 2)
    ss_tot = np.sum((Y_target - Y_target.mean(axis=0)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return {
        "weights": W,  # (K_attr, K_target)
        "intercept": intercept,  # (K_target,)
        "alpha": alpha,
        "r2_train": float(r2),
    }


def predict_pool(
    mapping: dict,
    x_attr: np.ndarray,
) -> dict:
    """Predict simulator settings for a new pool.

    Args:
        mapping: dict from fit_mapping
        x_attr: (K_attr,) single pool attribute vector

    Returns dict with cadence_minutes, gas_usd, noise_coeffs, etc.
    """
    y = x_attr @ mapping["weights"] + mapping["intercept"]

    log_cadence = float(y[0])
    log_gas = float(y[1])
    noise_coeffs = np.array(y[2:])

    return {
        "log_cadence": log_cadence,
        "log_gas": log_gas,
        "cadence_minutes": float(np.exp(log_cadence)),
        "gas_usd": float(np.exp(log_gas)),
        "noise_coeffs": noise_coeffs,
    }


def cross_validate_loo(
    X_attr: np.ndarray,
    Y_target: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """Leave-one-out cross-validation.

    Returns per-pool prediction errors and summary statistics.
    """
    n = X_attr.shape[0]
    errors = np.zeros((n, Y_target.shape[1]))

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X_attr[mask]
        Y_train = Y_target[mask]

        model = fit_mapping(X_train, Y_train, alpha=alpha)
        y_pred = X_attr[i] @ model["weights"] + model["intercept"]
        errors[i] = Y_target[i] - y_pred

    mse = np.mean(errors ** 2, axis=0)
    return {
        "per_pool_errors": errors,
        "mse_per_target": mse,
        "rmse_per_target": np.sqrt(mse),
        "mean_rmse": float(np.mean(np.sqrt(mse))),
    }
