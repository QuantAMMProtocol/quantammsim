"""Precompute noise_base and noise_tvl_coeff arrays for the simulator.

Takes a trained per-pool noise model artifact and produces the two daily
arrays needed by reclamm_market_linear_noise_volume():

    log(V_daily_noise) = noise_base_t + noise_tvl_coeff_t * log(effective_TVL)

The arrays are at daily resolution and need to be expanded to minute-level
(by repeating each day's value 1440 times) before passing to the simulator.

Usage:
    from quantammsim.calibration.noise_model_arrays import build_simulator_arrays

    arrays = build_simulator_arrays(
        pool_id="0x0b09dea16768f0",
        start_date="2025-06-01",
        end_date="2026-03-01",
        artifact_dir="results/linear_market_noise",
    )
    # arrays["noise_base"]       — (n_minutes,) float64
    # arrays["noise_tvl_coeff"]  — (n_minutes,) float64
"""

import json
import os
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def load_artifact(artifact_dir: str) -> Tuple[dict, dict]:
    """Load model.npz + meta.json from artifact directory."""
    art = np.load(os.path.join(artifact_dir, "model.npz"), allow_pickle=True)
    with open(os.path.join(artifact_dir, "meta.json")) as f:
        meta = json.load(f)
    return dict(art), meta


def _find_pool_index(pool_id: str, pool_ids: list) -> int:
    """Match pool_id (full or prefix) to calibration pool list."""
    for i, cid in enumerate(pool_ids):
        if pool_id.startswith(cid) or cid.startswith(pool_id):
            return i
    return -1


def _identify_tvl_columns(feat_names: list) -> Tuple[int, list]:
    """Identify which feature columns involve TVL.

    Returns:
        tvl_col: index of the pure log_tvl feature (xobs_1)
        tvl_interaction_cols: list of (col_idx, paired_col_idx) for
            interaction terms that multiply TVL with another feature
    """
    tvl_col = None
    tvl_interaction_cols = []

    for i, name in enumerate(feat_names):
        if name == "xobs_1":
            tvl_col = i
        elif "xobs_1×" in name:
            # e.g. "xobs_1×btc_realized_vol_7d" — find the paired feature
            paired_name = name.split("×")[1]
            for j, n2 in enumerate(feat_names):
                if n2 == paired_name:
                    tvl_interaction_cols.append((i, j))
                    break

    if tvl_col is None:
        raise ValueError("xobs_1 (log_tvl) not found in feature names")

    return tvl_col, tvl_interaction_cols


def build_daily_features(
    pool_id: str,
    matched_clean: dict,
    start_date: str,
    end_date: str,
    feat_names: list,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    trend_windows: tuple = (7,),
) -> Tuple[np.ndarray, list]:
    """Build the full standardized feature matrix for a pool over a date range.

    Returns (x_daily, dates) where x_daily is (n_days, n_feat) and dates
    is the list of dates. TVL column (xobs_1) is filled with the pool's
    observed log_tvl_lag1 where available, 0 otherwise.
    """
    from quantammsim.calibration.pool_data import (
        build_x_obs, build_cross_pool_x_obs, K_OBS_CROSS,
    )
    from quantammsim.calibration.market_features import (
        build_pool_market_features,
    )

    # Find the pool
    pid_match = None
    for pid in matched_clean:
        if pool_id.startswith(pid) or pid.startswith(pool_id):
            pid_match = pid
            break
    if pid_match is None:
        raise ValueError(f"Pool {pool_id} not found in matched_clean")

    entry = matched_clean[pid_match]
    panel = entry["panel"]

    # Filter panel to date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    panel_dates = pd.to_datetime(panel["date"])
    mask = (panel_dates >= start) & (panel_dates <= end)
    panel_sub = panel[mask.values].copy()
    n_days = len(panel_sub)

    if n_days < 2:
        raise ValueError(f"Only {n_days} days in range for pool {pool_id}")

    dates = panel_sub["date"].values

    # x_obs (cross-pool, 7 features) — need at least 1 lag
    xc = build_cross_pool_x_obs(panel_sub, matched_clean, pid_match)
    # xc drops first row; align
    if len(xc) < n_days:
        # Pad first row with zeros
        xc = np.vstack([np.zeros((1, xc.shape[1])), xc])

    # Market features
    pool_feat = build_pool_market_features(
        matched_clean, trend_windows=list(trend_windows))
    pf = pool_feat.get(pid_match)
    if pf is None:
        raise ValueError(f"No market features for {pool_id}")

    # Align market features to panel dates
    n_base = K_OBS_CROSS
    market_cols = [c for c in sorted(pf.columns)]
    n_market = len(market_cols)

    x_base = np.zeros((n_days, n_base + n_market), dtype=np.float32)
    x_base[:, :n_base] = xc[:n_days]

    for k, d in enumerate(dates):
        day = pd.Timestamp(d).normalize()
        if day in pf.index:
            for m, col in enumerate(market_cols):
                val = pf.loc[day, col]
                if np.isfinite(val):
                    x_base[k, n_base + m] = val

    # Standardize using saved stats (base features only)
    n_base_total = n_base + n_market
    x_base = ((x_base - x_mean[:n_base_total]) / x_std[:n_base_total]).astype(np.float32)

    # Interaction terms
    base_names = [f"xobs_{i}" for i in range(n_base)] + market_cols
    col_idx = {name: i for i, name in enumerate(base_names)}

    interactions = []
    for fname in feat_names[n_base_total:]:
        if "×" in fname:
            parts = fname.split("×")
            if parts[0] in col_idx and parts[1] in col_idx:
                interactions.append(
                    x_base[:, col_idx[parts[0]]] * x_base[:, col_idx[parts[1]]])
            else:
                interactions.append(np.zeros(n_days, dtype=np.float32))
        else:
            interactions.append(np.zeros(n_days, dtype=np.float32))

    if interactions:
        x_all = np.concatenate(
            [x_base, np.column_stack(interactions)], axis=1).astype(np.float32)
    else:
        x_all = x_base

    return x_all, dates.tolist()


def build_simulator_arrays(
    pool_id: str,
    start_date: str,
    end_date: str,
    artifact_dir: str = "results/linear_market_noise",
    matched_clean: Optional[dict] = None,
    arb_frequency: int = 1,
) -> Dict[str, np.ndarray]:
    """Build noise_base and noise_tvl_coeff arrays for the simulator.

    Parameters
    ----------
    pool_id : str
        Pool ID (full or prefix).
    start_date, end_date : str
        Date range (inclusive).
    artifact_dir : str
        Directory containing model.npz and meta.json.
    matched_clean : dict, optional
        Pre-loaded matched_clean dict. If None, loads from stage1.pkl.
    arb_frequency : int
        Arb frequency in minutes. Arrays are at minute resolution,
        repeated from daily values.

    Returns
    -------
    dict with:
        noise_base : (n_minutes,) array
        noise_tvl_coeff : (n_minutes,) array
        dates : list of dates
        pool_index : int (index in calibration set, or -1)
    """
    art, meta = load_artifact(artifact_dir)
    noise_coeffs = art["noise_coeffs"]
    feat_names = meta["feat_names"]
    pool_ids = meta["pool_ids"]
    x_mean = art["x_mean"]
    x_std = art["x_std"]
    per_pool = noise_coeffs.ndim == 2
    trend_windows = tuple(meta["hparams"]["trend_windows"])

    # Load matched_clean if needed
    if matched_clean is None:
        import pickle
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))),
            "results", "token_factored_calibration", "_cache",
        )
        with open(os.path.join(cache_dir, "stage1.pkl"), "rb") as f:
            data = pickle.load(f)
        matched_clean = data["matched_clean"]

    # Find pool coefficients
    pool_idx = _find_pool_index(pool_id, pool_ids)
    if pool_idx >= 0 and per_pool:
        coeffs = noise_coeffs[pool_idx]
    elif per_pool:
        print(f"  Warning: pool {pool_id} not in calibration set, using median coeffs")
        coeffs = np.median(noise_coeffs, axis=0)
    else:
        coeffs = noise_coeffs

    # Build daily features
    x_daily, dates = build_daily_features(
        pool_id, matched_clean, start_date, end_date,
        feat_names, x_mean, x_std, trend_windows,
    )
    n_days = len(dates)

    # Decompose into base (non-TVL) and tvl_coeff
    tvl_col, tvl_interactions = _identify_tvl_columns(feat_names)

    # tvl_coeff_t = coeffs[tvl_col] + sum(coeffs[inter_col] * x[paired_col])
    tvl_coeff_daily = np.full(n_days, coeffs[tvl_col], dtype=np.float64)
    for inter_col, paired_col in tvl_interactions:
        tvl_coeff_daily += coeffs[inter_col] * x_daily[:, paired_col]

    # base_t = sum(coeffs[j] * x[j]) for j not in {tvl_col, interaction_cols}
    tvl_related = {tvl_col} | {ic for ic, _ in tvl_interactions}
    base_daily = np.zeros(n_days, dtype=np.float64)
    for j in range(len(feat_names)):
        if j not in tvl_related:
            base_daily += coeffs[j] * x_daily[:, j]

    # Expand to minute resolution: each day's value repeats 1440 times
    n_minutes = n_days * 1440
    noise_base = np.repeat(base_daily, 1440)
    noise_tvl_coeff = np.repeat(tvl_coeff_daily, 1440)

    return {
        "noise_base": noise_base,
        "noise_tvl_coeff": noise_tvl_coeff,
        "tvl_mean": float(x_mean[tvl_col]),
        "tvl_std": float(x_std[tvl_col]),
        "dates": dates,
        "pool_index": pool_idx,
        "n_days": n_days,
        "n_minutes": n_minutes,
        "coeffs": coeffs,
        "tvl_col": tvl_col,
    }
