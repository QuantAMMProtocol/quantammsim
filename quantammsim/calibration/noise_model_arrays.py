"""Precompute noise_base and noise_tvl_coeff arrays for the simulator.

Builds daily feature vectors from Binance price data only — no panel/API
dependency. Works for any date range covered by Binance parquets.

Produces the two arrays needed by reclamm_market_linear_noise_volume():

    log(V_daily_noise) = noise_base_t + noise_tvl_coeff_t * log(effective_TVL)

Usage:
    from quantammsim.calibration.noise_model_arrays import build_simulator_arrays

    arrays = build_simulator_arrays(
        token_a="AAVE", token_b="ETH",
        start_date="2024-06-01",
        end_date="2026-03-01",
        artifact_dir="results/linear_market_noise",
        pool_id="0x9d1fcf346ea1b0",  # for per-pool coeffs
    )
"""

import json
import os
from typing import Dict, List, Optional, Tuple

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
        tvl_interaction_cols: list of (col_idx, paired_col_idx)
    """
    tvl_col = None
    tvl_interaction_cols = []

    for i, name in enumerate(feat_names):
        if name == "xobs_1":
            tvl_col = i
        elif "xobs_1\u00d7" in name:
            paired_name = name.split("\u00d7")[1]
            for j, n2 in enumerate(feat_names):
                if n2 == paired_name:
                    tvl_interaction_cols.append((i, j))
                    break

    if tvl_col is None:
        raise ValueError("xobs_1 (log_tvl) not found in feature names")

    return tvl_col, tvl_interaction_cols


def build_daily_features_from_binance(
    token_a: str,
    token_b: str,
    start_date: str,
    end_date: str,
    feat_names: List[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    trend_windows: tuple = (7,),
) -> Tuple[np.ndarray, list]:
    """Build daily feature matrix from Binance data only.

    No panel or API dependency. Features:
      - xobs_0 (intercept), xobs_1 (log_tvl — filled with 0, handled at runtime),
        xobs_2/3 (dow_sin/cos)
      - BTC: log_price, log_return, realized_vol_7d, trend, volume_zscore
      - Token A/B: log_return, realized_vol_7d, trend, volume_zscore
      - Pair realized_vol_7d
      - Interaction terms
    """
    from quantammsim.calibration.market_features import (
        build_btc_daily_features,
        build_token_daily_features,
        _compute_pair_volatility,
        TOKEN_MAP,
    )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Generate complete daily date range
    date_range = pd.date_range(start, end, freq="D")
    n_days = len(date_range)

    # BTC features
    btc_feat = build_btc_daily_features(list(trend_windows))

    # Token features
    mapped_a = TOKEN_MAP.get(token_a, token_a)
    mapped_b = TOKEN_MAP.get(token_b, token_b)
    feat_a = build_token_daily_features(mapped_a, list(trend_windows))
    feat_b = build_token_daily_features(mapped_b, list(trend_windows))

    # Pair volatility
    pair_vol = _compute_pair_volatility(token_a, token_b)

    # Identify which features are x_obs vs market
    # x_obs features: xobs_0 (intercept), xobs_1 (tvl), xobs_2 (dow_sin), xobs_3 (dow_cos)
    # Remaining xobs_4,5,6 are cross-pool — skip if not in feat_names
    n_xobs = sum(1 for f in feat_names if f.startswith("xobs_"))

    # Build market feature column list (everything after x_obs, before interactions)
    market_names = [f for f in feat_names
                    if not f.startswith("xobs_") and "\u00d7" not in f]

    # Build per-day feature vectors
    x_base_cols = n_xobs + len(market_names)
    x_base = np.zeros((n_days, x_base_cols), dtype=np.float32)

    for k, day in enumerate(date_range):
        day_norm = day.normalize()

        # x_obs
        x_base[k, 0] = 1.0  # intercept
        # x_base[k, 1] = 0.0  # log_tvl — placeholder, handled at runtime
        weekday = day.weekday()
        if n_xobs > 2:
            x_base[k, 2] = np.sin(2 * np.pi * weekday / 7)
        if n_xobs > 3:
            x_base[k, 3] = np.cos(2 * np.pi * weekday / 7)
        # xobs_4,5,6 (cross-pool) left as 0 if present

        # Market features
        col = n_xobs
        for mname in market_names:
            val = 0.0
            if mname.startswith("btc_") and btc_feat is not None:
                bcol = mname[4:]  # strip "btc_"
                if day_norm in btc_feat.index and bcol in btc_feat.columns:
                    v = btc_feat.loc[day_norm, bcol]
                    if np.isfinite(v):
                        val = v
            elif mname.startswith("tok_a_") and feat_a is not None:
                acol = mname[6:]
                if day_norm in feat_a.index and acol in feat_a.columns:
                    v = feat_a.loc[day_norm, acol]
                    if np.isfinite(v):
                        val = v
            elif mname.startswith("tok_b_") and feat_b is not None:
                bcol = mname[6:]
                if day_norm in feat_b.index and bcol in feat_b.columns:
                    v = feat_b.loc[day_norm, bcol]
                    if np.isfinite(v):
                        val = v
            elif mname == "pair_realized_vol_7d" and pair_vol is not None:
                if day_norm in pair_vol.index:
                    v = pair_vol.loc[day_norm, "pair_realized_vol_7d"]
                    if np.isfinite(v):
                        val = v
            x_base[k, col] = val
            col += 1

    # Standardize base features
    x_base = ((x_base - x_mean[:x_base_cols]) / x_std[:x_base_cols]).astype(np.float32)

    # Interaction terms
    base_feat_names = feat_names[:x_base_cols]
    col_idx = {name: i for i, name in enumerate(base_feat_names)}

    interactions = []
    for fname in feat_names[x_base_cols:]:
        if "\u00d7" in fname:
            parts = fname.split("\u00d7")
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

    return x_all, date_range.tolist()


def build_simulator_arrays(
    token_a: str,
    token_b: str,
    start_date: str,
    end_date: str,
    artifact_dir: str = "results/linear_market_noise",
    pool_id: Optional[str] = None,
) -> Dict:
    """Build noise_base and noise_tvl_coeff arrays for the simulator.

    No panel dependency — uses Binance data only.

    Parameters
    ----------
    token_a, token_b : str
        Token symbols (e.g. "AAVE", "ETH"). Mapped to Binance symbols
        internally (WETH→ETH, wstETH→ETH, etc.)
    start_date, end_date : str
        Date range (inclusive).
    artifact_dir : str
        Directory containing model.npz and meta.json.
    pool_id : str, optional
        Pool ID for per-pool coefficients. If None or not found,
        uses median coefficients.

    Returns
    -------
    dict with noise_base, noise_tvl_coeff, tvl_mean, tvl_std, dates, etc.
    """
    art, meta = load_artifact(artifact_dir)
    noise_coeffs = art["noise_coeffs"]
    feat_names = meta["feat_names"]
    pool_ids = meta["pool_ids"]
    x_mean = art["x_mean"]
    x_std = art["x_std"]
    per_pool = noise_coeffs.ndim == 2
    trend_windows = tuple(meta["hparams"]["trend_windows"])

    # Find pool coefficients
    pool_idx = -1
    if pool_id is not None:
        pool_idx = _find_pool_index(pool_id, pool_ids)

    if pool_idx >= 0 and per_pool:
        coeffs = noise_coeffs[pool_idx]
        print(f"  Using per-pool coefficients (pool idx {pool_idx})")
    elif per_pool:
        coeffs = np.median(noise_coeffs, axis=0)
        print(f"  Pool not found, using median coefficients")
    else:
        coeffs = noise_coeffs

    # Build daily features from Binance
    print(f"  Building features from Binance data: {token_a}/{token_b},"
          f" {start_date} → {end_date}")
    x_daily, dates = build_daily_features_from_binance(
        token_a, token_b, start_date, end_date,
        feat_names, x_mean, x_std, trend_windows,
    )
    n_days = len(dates)
    print(f"  {n_days} days, {len(feat_names)} features")

    # Decompose into base (non-TVL) and tvl_coeff
    tvl_col, tvl_interactions = _identify_tvl_columns(feat_names)

    tvl_coeff_daily = np.full(n_days, coeffs[tvl_col], dtype=np.float64)
    for inter_col, paired_col in tvl_interactions:
        tvl_coeff_daily += coeffs[inter_col] * x_daily[:, paired_col]

    tvl_related = {tvl_col} | {ic for ic, _ in tvl_interactions}
    base_daily = np.zeros(n_days, dtype=np.float64)
    for j in range(len(feat_names)):
        if j not in tvl_related:
            base_daily += coeffs[j] * x_daily[:, j]

    # Expand to minute resolution
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
