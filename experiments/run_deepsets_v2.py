"""DeepSets v2: full feature menu with Optuna feature selection.

Trains on total log_volume, evaluates on both total volume and noise
residual (log_vol - log_V_arb). V_arb precomputed from Option C fits.

Feature menu:
  Peer (encoder) — always: peer_attr, target_attr, vol_lag1, overlap
                   optional: vol_lag2, vol_change, tvl, volatility
                   relational: same_chain, log_tvl_ratio, log_fee_ratio
  Local (decoder) — always: target_attr, own_vol_lag1, dow_sin, dow_cos
                    optional: own_vol_lag2, own_vol_change, own_tvl, own_volatility

Model variants:
  encoder_type: "mlp" (2-layer ReLU) or "linear" (single affine)
  no_peers: decoder-only ablation (zero peer summary)
  huber_delta: Huber loss transition point (default 1.0)
  Per-pool loss weighting (equal weight per pool regardless of sample count)

Usage:
  python experiments/run_deepsets_v2.py                    # defaults
  python experiments/run_deepsets_v2.py --tune 50          # Optuna
  python experiments/run_deepsets_v2.py --no-peers         # decoder-only
  python experiments/run_deepsets_v2.py --encoder-type linear
  python experiments/run_deepsets_v2.py --loo              # LOO eval
"""

import argparse
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)


def load_stage1():
    path = os.path.join(CACHE_DIR, "stage1.pkl")
    if not os.path.exists(path):
        print("ERROR: no stage1 cache.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["matched_clean"], data["option_c_clean"]


# ---- Data construction ----


def build_all_features(matched_clean, option_c_clean):
    """Build all possible feature matrices. Called once."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import (
        build_pool_attributes, _parse_tokens, _canonicalize_token,
        build_x_obs, build_cross_pool_x_obs,
    )

    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)

    # Collect dates
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)
    n_dates = len(date_list)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    # Daily matrices: (n_dates, n_pools)
    vol_matrix = np.full((n_dates, n_pools), np.nan)
    tvl_matrix = np.full((n_dates, n_pools), np.nan)
    volatility_matrix = np.full((n_dates, n_pools), np.nan)
    v_arb_matrix = np.full((n_dates, n_pools), np.nan)
    weekday_arr = np.zeros(n_dates)

    for j, pid in enumerate(pool_ids):
        entry = matched_clean[pid]
        oc = option_c_clean[pid]
        panel = entry["panel"]

        v_arb_all = np.array(interpolate_pool_daily(
            entry["coeffs"],
            jnp.float64(oc["log_cadence"]),
            jnp.float64(np.exp(oc["log_gas"])),
        ))
        v_arb = v_arb_all[entry["day_indices"]]

        dates = panel["date"].values
        for k, date in enumerate(dates):
            t = date_to_idx[date]
            vol_matrix[t, j] = panel["log_volume"].values[k]
            tvl_matrix[t, j] = panel["log_tvl_lag1"].values[k]
            volatility_matrix[t, j] = panel["volatility"].values[k]
            v_arb_matrix[t, j] = v_arb[k]

    # Per-pool coeffs, gas, and day mapping for learnable cadence
    pool_coeffs = []
    pool_gas = []
    init_log_cadences = np.zeros(n_pools, dtype=np.float32)
    common_to_grid = np.full((n_pools, n_dates), 0, dtype=np.int32)

    for j, pid in enumerate(pool_ids):
        entry = matched_clean[pid]
        oc = option_c_clean[pid]
        pool_coeffs.append(entry["coeffs"])
        pool_gas.append(jnp.float64(np.exp(oc["log_gas"])))
        init_log_cadences[j] = oc["log_cadence"]
        dates_j = entry["panel"]["date"].values
        for k, date in enumerate(dates_j):
            common_to_grid[j, date_to_idx[date]] = entry["day_indices"][k]

    for t, date in enumerate(date_list):
        weekday_arr[t] = pd.Timestamp(date).weekday()

    # Pool attributes (static)
    X_attr, attr_names, _ = build_pool_attributes(matched_clean)
    attr_mean = np.mean(X_attr, axis=0)
    attr_std = np.std(X_attr, axis=0)
    attr_std[attr_std < 1e-6] = 1.0
    X_attr_norm = ((X_attr - attr_mean) / attr_std).astype(np.float32)
    k_attr = X_attr_norm.shape[1]

    # Raw per-pool values for relational features
    fee_idx = attr_names.index("log_fee")
    tvl_idx = attr_names.index("mean_log_tvl")
    raw_log_fee = X_attr[:, fee_idx]
    raw_mean_log_tvl = X_attr[:, tvl_idx]
    pool_chains = [matched_clean[pid]["chain"] for pid in pool_ids]

    # Token overlap
    pool_tokens = {}
    for i, pid in enumerate(pool_ids):
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        pool_tokens[i] = {_canonicalize_token(t) for t in toks[:2]}

    n_peers = n_pools - 1
    peer_attrs = np.zeros((n_pools, n_peers, k_attr), dtype=np.float32)
    peer_overlap = np.zeros((n_pools, n_peers), dtype=np.float32)
    peer_col_idx = np.zeros((n_pools, n_peers), dtype=np.int32)
    rel_same_chain = np.zeros((n_pools, n_peers), dtype=np.float32)
    rel_log_tvl_ratio = np.zeros((n_pools, n_peers), dtype=np.float32)
    rel_log_fee_ratio = np.zeros((n_pools, n_peers), dtype=np.float32)

    for i in range(n_pools):
        peers = [j for j in range(n_pools) if j != i]
        for p, j in enumerate(peers):
            peer_attrs[i, p] = X_attr_norm[j]
            peer_overlap[i, p] = len(pool_tokens[i] & pool_tokens[j])
            peer_col_idx[i, p] = j
            rel_same_chain[i, p] = float(pool_chains[i] == pool_chains[j])
            rel_log_tvl_ratio[i, p] = abs(raw_mean_log_tvl[i] - raw_mean_log_tvl[j])
            rel_log_fee_ratio[i, p] = abs(raw_log_fee[i] - raw_log_fee[j])

    # Standardize ratio features (new arrays, no in-place mutation)
    def _standardize(arr):
        mu = np.mean(arr)
        sigma = max(np.std(arr), 1e-6)
        return ((arr - mu) / sigma).astype(np.float32)

    rel_log_tvl_ratio = _standardize(rel_log_tvl_ratio)
    rel_log_fee_ratio = _standardize(rel_log_fee_ratio)

    # Cross-pool peer maps: which pools share tokens / chain
    from collections import defaultdict
    pool_tokens_ordered = {}
    token_to_pools = defaultdict(set)
    for i, pid in enumerate(pool_ids):
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        ordered = [_canonicalize_token(t) for t in toks[:2]]
        pool_tokens_ordered[i] = ordered
        for tok in ordered:
            token_to_pools[tok].add(i)

    token_a_peers = {}
    token_b_peers = {}
    chain_peer_map = {}
    for i in range(n_pools):
        toks = pool_tokens_ordered[i]
        token_a_peers[i] = sorted(token_to_pools[toks[0]] - {i})
        token_b_peers[i] = sorted(token_to_pools[toks[1]] - {i}) if len(toks) > 1 else []
        chain_peer_map[i] = [j for j in range(n_pools) if j != i and pool_chains[j] == pool_chains[i]]

    # Per-pool log_fee for interaction features
    pool_log_fee = raw_log_fee.copy()
    fee_mean = float(np.mean(pool_log_fee))
    fee_std = max(float(np.std(pool_log_fee)), 1e-6)

    # Standardization stats for volumes
    vol_mean = float(np.nanmean(vol_matrix))
    vol_std = float(np.nanstd(vol_matrix))
    tvl_mean = float(np.nanmean(tvl_matrix))
    tvl_std = float(np.nanstd(tvl_matrix))
    vola_mean = float(np.nanmean(volatility_matrix))
    vola_std = float(np.nanstd(volatility_matrix))

    # Build x_obs per pool, mapped to common date grid
    # x_obs_reduced: (n_dates, n_pools, 4), x_obs_cross: (n_dates, n_pools, 7)
    from quantammsim.calibration.pool_data import K_OBS_REDUCED, K_OBS_CROSS
    x_obs_reduced_grid = np.full((n_dates, n_pools, K_OBS_REDUCED), np.nan)
    x_obs_cross_grid = np.full((n_dates, n_pools, K_OBS_CROSS), np.nan)

    for j, pid in enumerate(pool_ids):
        entry = matched_clean[pid]
        panel = entry["panel"]
        dates_j = panel["date"].values

        # Reduced x_obs (4 features)
        xr = build_x_obs(panel, reduced=True)  # (n_obs, 4)
        for k, date in enumerate(dates_j):
            x_obs_reduced_grid[date_to_idx[date], j] = xr[k]

        # Cross-pool x_obs (7 features) — drops first day
        xc = build_cross_pool_x_obs(panel, matched_clean, pid)  # (n_obs-1, 7)
        for k, date in enumerate(dates_j[1:]):
            x_obs_cross_grid[date_to_idx[date], j] = xc[k]

    # Build samples: require t >= 2 (for lag-2), valid vol at t, t-1, t-2
    sample_pools, sample_days = [], []
    for i in range(n_pools):
        for t in range(2, n_dates):
            if (np.isnan(vol_matrix[t, i]) or np.isnan(vol_matrix[t - 1, i])
                    or np.isnan(vol_matrix[t - 2, i])):
                continue
            sample_pools.append(i)
            sample_days.append(t)

    sample_pools = np.array(sample_pools, dtype=np.int32)
    sample_days = np.array(sample_days, dtype=np.int32)
    n_samples = len(sample_pools)

    def _norm_vol(x):
        return (x - vol_mean) / vol_std

    def _norm_tvl(x):
        return (x - tvl_mean) / tvl_std

    def _norm_vola(x):
        return (x - vola_mean) / vola_std

    # Per-sample arrays
    # Peer features (per peer)
    pf_vol_lag1 = np.zeros((n_samples, n_peers), dtype=np.float32)
    pf_vol_lag2 = np.zeros((n_samples, n_peers), dtype=np.float32)
    pf_vol_change = np.zeros((n_samples, n_peers), dtype=np.float32)
    pf_tvl = np.zeros((n_samples, n_peers), dtype=np.float32)
    pf_volatility = np.zeros((n_samples, n_peers), dtype=np.float32)
    peer_mask = np.zeros((n_samples, n_peers), dtype=np.float32)

    # Local features
    lf_own_vol_lag1 = np.zeros(n_samples, dtype=np.float32)
    lf_own_vol_lag2 = np.zeros(n_samples, dtype=np.float32)
    lf_own_vol_change = np.zeros(n_samples, dtype=np.float32)
    lf_own_tvl = np.zeros(n_samples, dtype=np.float32)
    lf_own_volatility = np.zeros(n_samples, dtype=np.float32)
    lf_dow_sin = np.zeros(n_samples, dtype=np.float32)
    lf_dow_cos = np.zeros(n_samples, dtype=np.float32)
    # Interaction features (from calibration pipeline's x_obs)
    lf_tvl_x_vola = np.zeros(n_samples, dtype=np.float32)
    lf_tvl_x_fee = np.zeros(n_samples, dtype=np.float32)
    lf_vola_x_fee = np.zeros(n_samples, dtype=np.float32)
    # Cross-pool volume aggregates
    lf_cross_vol_tok_a = np.zeros(n_samples, dtype=np.float32)
    lf_cross_vol_tok_b = np.zeros(n_samples, dtype=np.float32)
    lf_cross_vol_chain = np.zeros(n_samples, dtype=np.float32)
    lf_market_vol = np.zeros(n_samples, dtype=np.float32)
    # Cross-pool momentum (peer volume changes)
    lf_cross_mom_tok_a = np.zeros(n_samples, dtype=np.float32)
    lf_cross_mom_tok_b = np.zeros(n_samples, dtype=np.float32)
    lf_cross_mom_chain = np.zeros(n_samples, dtype=np.float32)

    # Targets
    y_total = np.zeros(n_samples, dtype=np.float32)
    v_arb_samples = np.zeros(n_samples, dtype=np.float32)

    for s in range(n_samples):
        i = sample_pools[s]
        t = sample_days[s]
        cols = peer_col_idx[i]

        # Peer features at t-1
        pvols1 = vol_matrix[t - 1, cols]
        pvols2 = vol_matrix[t - 2, cols]
        valid = ~np.isnan(pvols1)
        peer_mask[s] = valid.astype(np.float32)

        pf_vol_lag1[s] = np.where(valid, _norm_vol(pvols1), 0.0)
        pf_vol_lag2[s] = np.where(valid & ~np.isnan(pvols2), _norm_vol(pvols2), 0.0)
        pf_vol_change[s] = np.where(
            valid & ~np.isnan(pvols2),
            _norm_vol(pvols1) - _norm_vol(pvols2), 0.0)

        ptvl = tvl_matrix[t - 1, cols]
        pf_tvl[s] = np.where(valid & ~np.isnan(ptvl), _norm_tvl(ptvl), 0.0)

        pvola = volatility_matrix[t - 1, cols]
        pf_volatility[s] = np.where(valid & ~np.isnan(pvola), _norm_vola(pvola), 0.0)

        # Local features
        lf_own_vol_lag1[s] = _norm_vol(vol_matrix[t - 1, i])
        lf_own_vol_lag2[s] = _norm_vol(vol_matrix[t - 2, i])
        lf_own_vol_change[s] = lf_own_vol_lag1[s] - lf_own_vol_lag2[s]

        tvl_val = tvl_matrix[t, i]
        lf_own_tvl[s] = _norm_tvl(tvl_val) if np.isfinite(tvl_val) else 0.0

        vola_val = volatility_matrix[t, i]
        lf_own_volatility[s] = _norm_vola(vola_val) if np.isfinite(vola_val) else 0.0

        wd = weekday_arr[t]
        lf_dow_sin[s] = np.sin(2 * np.pi * wd / 7)
        lf_dow_cos[s] = np.cos(2 * np.pi * wd / 7)

        # Interaction features (raw products, standardized after loop)
        norm_fee_i = (raw_log_fee[i] - fee_mean) / fee_std
        lf_tvl_x_vola[s] = lf_own_tvl[s] * lf_own_volatility[s]
        lf_tvl_x_fee[s] = lf_own_tvl[s] * norm_fee_i
        lf_vola_x_fee[s] = lf_own_volatility[s] * norm_fee_i

        # Cross-pool volume aggregates at t-1
        def _peer_vol_mean(peer_list, t_lag):
            if not peer_list:
                return vol_mean  # global fallback
            vals = vol_matrix[t_lag, peer_list]
            valid = vals[~np.isnan(vals)]
            return float(np.mean(valid)) if len(valid) > 0 else vol_mean

        def _peer_vol_change_mean(peer_list, t_lag):
            if not peer_list:
                return 0.0
            v1 = vol_matrix[t_lag, peer_list]
            v2 = vol_matrix[t_lag - 1, peer_list]
            valid = ~np.isnan(v1) & ~np.isnan(v2)
            if valid.sum() == 0:
                return 0.0
            return float(np.mean(v1[valid] - v2[valid]))

        lf_cross_vol_tok_a[s] = _norm_vol(_peer_vol_mean(token_a_peers[i], t - 1))
        lf_cross_vol_tok_b[s] = _norm_vol(_peer_vol_mean(token_b_peers[i], t - 1))
        lf_cross_vol_chain[s] = _norm_vol(_peer_vol_mean(chain_peer_map[i], t - 1))
        lf_market_vol[s] = _norm_vol(float(np.nanmean(vol_matrix[t - 1, :])))

        # Cross-pool momentum: mean volume change of peers (t-1 vs t-2)
        lf_cross_mom_tok_a[s] = _peer_vol_change_mean(token_a_peers[i], t - 1)
        lf_cross_mom_tok_b[s] = _peer_vol_change_mean(token_b_peers[i], t - 1)
        lf_cross_mom_chain[s] = _peer_vol_change_mean(chain_peer_map[i], t - 1)

        y_total[s] = vol_matrix[t, i]
        v_arb_val = v_arb_matrix[t, i]
        v_arb_samples[s] = v_arb_val if np.isfinite(v_arb_val) else 1e-6

    # Per-sample grid day indices for learnable cadence
    sample_grid_days = common_to_grid[sample_pools, sample_days]

    # Per-sample x_obs arrays
    x_obs_reduced = np.zeros((n_samples, K_OBS_REDUCED), dtype=np.float32)
    x_obs_cross = np.zeros((n_samples, K_OBS_CROSS), dtype=np.float32)
    for s in range(n_samples):
        xr = x_obs_reduced_grid[sample_days[s], sample_pools[s]]
        if np.all(np.isfinite(xr)):
            x_obs_reduced[s] = xr
        xc = x_obs_cross_grid[sample_days[s], sample_pools[s]]
        if np.all(np.isfinite(xc)):
            x_obs_cross[s] = xc

    # Standardize momentum features (raw volume differences)
    for arr in [lf_cross_mom_tok_a, lf_cross_mom_tok_b, lf_cross_mom_chain]:
        mu = np.mean(arr)
        sigma = max(np.std(arr), 1e-6)
        arr[:] = ((arr - mu) / sigma).astype(np.float32)

    return {
        # Static per-pool
        "peer_attrs": peer_attrs,       # (n_pools, n_peers, k_attr)
        "target_attrs": X_attr_norm,    # (n_pools, k_attr)
        "peer_overlap": peer_overlap,   # (n_pools, n_peers)
        "rel_same_chain": rel_same_chain,       # (n_pools, n_peers)
        "rel_log_tvl_ratio": rel_log_tvl_ratio, # (n_pools, n_peers)
        "rel_log_fee_ratio": rel_log_fee_ratio, # (n_pools, n_peers)
        # Per-sample peer features
        "pf_vol_lag1": pf_vol_lag1,
        "pf_vol_lag2": pf_vol_lag2,
        "pf_vol_change": pf_vol_change,
        "pf_tvl": pf_tvl,
        "pf_volatility": pf_volatility,
        "peer_mask": peer_mask,
        # Per-sample local features
        "lf_own_vol_lag1": lf_own_vol_lag1,
        "lf_own_vol_lag2": lf_own_vol_lag2,
        "lf_own_vol_change": lf_own_vol_change,
        "lf_own_tvl": lf_own_tvl,
        "lf_own_volatility": lf_own_volatility,
        "lf_dow_sin": lf_dow_sin,
        "lf_dow_cos": lf_dow_cos,
        # Interaction features
        "lf_tvl_x_vola": lf_tvl_x_vola,
        "lf_tvl_x_fee": lf_tvl_x_fee,
        "lf_vola_x_fee": lf_vola_x_fee,
        # Cross-pool volume aggregates
        "lf_cross_vol_tok_a": lf_cross_vol_tok_a,
        "lf_cross_vol_tok_b": lf_cross_vol_tok_b,
        "lf_cross_vol_chain": lf_cross_vol_chain,
        "lf_market_vol": lf_market_vol,
        # Cross-pool momentum
        "lf_cross_mom_tok_a": lf_cross_mom_tok_a,
        "lf_cross_mom_tok_b": lf_cross_mom_tok_b,
        "lf_cross_mom_chain": lf_cross_mom_chain,
        # Targets
        "y_total": y_total,
        "y_residual": (y_total - np.log(np.maximum(v_arb_samples, 1e-6))).astype(np.float32),
        "v_arb": v_arb_samples,
        # Cadence learning (per-pool, not subject to _subset)
        "pool_coeffs": pool_coeffs,              # list of PoolCoeffsDaily
        "pool_gas": pool_gas,                    # list of jnp scalars
        "init_log_cadences": init_log_cadences,  # (n_pools,)
        "sample_grid_days": sample_grid_days,    # (n_samples,)
        "x_obs_reduced": x_obs_reduced,          # (n_samples, 4)
        "x_obs_cross": x_obs_cross,              # (n_samples, 7)
        # Indices
        "pool_idx": sample_pools,
        "day_idx": sample_days,
        # Meta
        "n_pools": n_pools,
        "n_peers": n_peers,
        "k_attr": k_attr,
        "pool_ids": pool_ids,
        "vol_mean": vol_mean,
        "vol_std": vol_std,
        "fee_attr_idx": fee_idx,
        "tvl_attr_idx": tvl_idx,
    }


def assemble_inputs(data, feat_cfg):
    """Assemble encoder/decoder inputs based on feature config.

    Returns dict with JAX arrays ready for training.
    """
    # ---- Peer encoder input: (n_samples, n_peers, n_feat) ----
    pool_idx = data["pool_idx"]
    pa = data["peer_attrs"][pool_idx]  # (n_samples, n_peers, k_attr)
    ta = data["target_attrs"][pool_idx]  # (n_samples, k_attr)

    if feat_cfg.get("minimal_encoder"):
        # 7-feature encoder: peer_fee, peer_tvl, target_fee, target_tvl,
        # vol_lag1, overlap, same_chain — prevents pool identification
        fi = data["fee_attr_idx"]
        ti = data["tvl_attr_idx"]
        pa_min = np.stack([pa[:, :, fi], pa[:, :, ti]], axis=-1)
        ta_min = np.stack([ta[:, fi], ta[:, ti]], axis=-1)
        ta_min_broad = np.broadcast_to(
            ta_min[:, None, :], (pa_min.shape[0], pa_min.shape[1], 2))
        peer_parts = [
            pa_min, ta_min_broad,
            data["pf_vol_lag1"][:, :, None],
            data["peer_overlap"][pool_idx][:, :, None],
            data["rel_same_chain"][pool_idx][:, :, None],
        ]
    else:
        ta_broad = np.broadcast_to(ta[:, None, :], pa.shape)
        peer_parts = [
            pa, ta_broad,
            data["pf_vol_lag1"][:, :, None],
            data["peer_overlap"][pool_idx][:, :, None],
        ]
        # Relational features (optional via feat_cfg, default on)
        if feat_cfg.get("rel_same_chain", True):
            peer_parts.append(data["rel_same_chain"][pool_idx][:, :, None])

    # Optional temporal peer features (both modes)
    if feat_cfg.get("peer_vol_lag2"):
        peer_parts.append(data["pf_vol_lag2"][:, :, None])
    if feat_cfg.get("peer_vol_change"):
        peer_parts.append(data["pf_vol_change"][:, :, None])
    if feat_cfg.get("peer_tvl"):
        peer_parts.append(data["pf_tvl"][:, :, None])
    if feat_cfg.get("peer_volatility"):
        peer_parts.append(data["pf_volatility"][:, :, None])

    # Relational ratio features (both modes)
    if feat_cfg.get("rel_tvl_ratio", True):
        peer_parts.append(data["rel_log_tvl_ratio"][pool_idx][:, :, None])
    if feat_cfg.get("rel_fee_ratio", True):
        peer_parts.append(data["rel_log_fee_ratio"][pool_idx][:, :, None])

    peer_input = np.concatenate(peer_parts, axis=-1).astype(np.float32)

    # ---- Local decoder input: (n_samples, n_feat) ----
    # Always: target_attr, own_vol_lag1, dow_sin, dow_cos
    local_parts = [
        ta,
        data["lf_own_vol_lag1"][:, None],
        data["lf_dow_sin"][:, None],
        data["lf_dow_cos"][:, None],
    ]

    if feat_cfg.get("own_vol_lag2"):
        local_parts.append(data["lf_own_vol_lag2"][:, None])
    if feat_cfg.get("own_vol_change"):
        local_parts.append(data["lf_own_vol_change"][:, None])
    if feat_cfg.get("own_tvl"):
        local_parts.append(data["lf_own_tvl"][:, None])
    if feat_cfg.get("own_volatility"):
        local_parts.append(data["lf_own_volatility"][:, None])

    # Interaction features (tvl×vola, tvl×fee, vola×fee)
    if feat_cfg.get("interactions"):
        local_parts.append(data["lf_tvl_x_vola"][:, None])
        local_parts.append(data["lf_tvl_x_fee"][:, None])
        local_parts.append(data["lf_vola_x_fee"][:, None])

    # Cross-pool volume aggregates (token-peer, chain-peer, market)
    if feat_cfg.get("cross_pool_vol"):
        local_parts.append(data["lf_cross_vol_tok_a"][:, None])
        local_parts.append(data["lf_cross_vol_tok_b"][:, None])
        local_parts.append(data["lf_cross_vol_chain"][:, None])
        local_parts.append(data["lf_market_vol"][:, None])

    # Cross-pool momentum (peer volume changes)
    if feat_cfg.get("cross_pool_momentum"):
        local_parts.append(data["lf_cross_mom_tok_a"][:, None])
        local_parts.append(data["lf_cross_mom_tok_b"][:, None])
        local_parts.append(data["lf_cross_mom_chain"][:, None])

    # Option C x_obs covariates (none / reduced=4 / cross=7)
    x_obs_mode = feat_cfg.get("x_obs_mode", "none")
    if x_obs_mode == "reduced" and "x_obs_reduced" in data:
        local_parts.append(data["x_obs_reduced"])
    elif x_obs_mode == "cross" and "x_obs_cross" in data:
        local_parts.append(data["x_obs_cross"])

    local_input = np.concatenate(local_parts, axis=-1).astype(np.float32)

    result = {
        "peer_input": jnp.array(peer_input),
        "local_input": jnp.array(local_input),
        "peer_mask": jnp.array(data["peer_mask"]),
        "y": jnp.array(data["y_residual"] if feat_cfg.get("target_residual") else data["y_total"]),
        "y_total": jnp.array(data["y_total"]),
        "v_arb": jnp.array(data["v_arb"]),
        "pool_idx": jnp.array(pool_idx),
        "n_pools": data["n_pools"],
        "n_peer_feat": peer_input.shape[-1],
        "n_local_feat": local_input.shape[-1],
    }
    # Cadence learning arrays
    if "sample_grid_days" in data:
        result["sample_grid_days"] = jnp.array(data["sample_grid_days"])
    return result


# ---- Model ----


def init_params(key, n_peer_feat, n_local_feat, hidden, d_embed,
                encoder_type="mlp"):
    """Initialize model parameters.

    encoder_type: "mlp" (2-layer ReLU) or "linear" (single affine).
    Presence of "enc_W2" in params dict distinguishes the two at forward time.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    dec_in = d_embed + n_local_feat
    params = {}

    if encoder_type == "mlp":
        params["enc_W1"] = jax.random.normal(k1, (n_peer_feat, hidden)) * np.sqrt(2.0 / n_peer_feat)
        params["enc_b1"] = jnp.zeros(hidden)
        params["enc_W2"] = jax.random.normal(k2, (hidden, d_embed)) * np.sqrt(2.0 / hidden)
        params["enc_b2"] = jnp.zeros(d_embed)
    else:  # linear
        params["enc_W1"] = jax.random.normal(k1, (n_peer_feat, d_embed)) * np.sqrt(2.0 / n_peer_feat)
        params["enc_b1"] = jnp.zeros(d_embed)

    params["dec_W1"] = jax.random.normal(k3, (dec_in, hidden)) * np.sqrt(2.0 / dec_in)
    params["dec_b1"] = jnp.zeros(hidden)
    params["dec_W2"] = jax.random.normal(k4, (hidden, 1)) * 0.01
    params["dec_b2"] = jnp.zeros(1)
    return params


def warm_start_decoder(params, inputs, d_embed):
    """Set decoder output layer via OLS through hidden activations.

    Fits y ~ h(local_input) with zero peer summary, so the decoder
    starts predicting in the right volume range (~10-17 log scale).
    """
    local = np.array(inputs["local_input"])
    y = np.array(inputs["y"])
    n = local.shape[0]

    # Simulate decoder input with zero peer summary
    dec_in = np.concatenate(
        [np.zeros((n, d_embed), dtype=np.float32), local], axis=1)

    # Forward through first decoder layer with current (random) weights
    h = np.maximum(
        dec_in @ np.array(params["dec_W1"]) + np.array(params["dec_b1"]), 0.0)

    # OLS: y ≈ h @ W2 + b2
    h_bias = np.concatenate([h, np.ones((n, 1), dtype=np.float32)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(h_bias, y[:, None], rcond=None)

    params["dec_W2"] = jnp.array(sol[:-1].astype(np.float32))
    params["dec_b2"] = jnp.array(sol[-1:].astype(np.float32))
    return params


def forward(params, peer_input, peer_mask, local_input, no_peers=False):
    """Returns predicted log_volume (total) per sample."""
    batch, n_peers, _ = peer_input.shape

    if no_peers:
        d_embed = params["dec_W1"].shape[0] - local_input.shape[-1]
        summary = jnp.zeros((batch, d_embed))
    else:
        flat = peer_input.reshape(-1, peer_input.shape[-1])
        if "enc_W2" in params:
            # MLP encoder: 2-layer with ReLU
            h = jnp.maximum(flat @ params["enc_W1"] + params["enc_b1"], 0.0)
            h = h @ params["enc_W2"] + params["enc_b2"]
        else:
            # Linear encoder: single affine
            h = flat @ params["enc_W1"] + params["enc_b1"]
        h = h.reshape(batch, n_peers, -1)

        h_masked = h * peer_mask[:, :, None]
        n_valid = jnp.maximum(jnp.sum(peer_mask, axis=1, keepdims=True), 1.0)
        summary = jnp.sum(h_masked, axis=1) / n_valid

    dec_in = jnp.concatenate([summary, local_input], axis=-1)
    h_dec = jnp.maximum(dec_in @ params["dec_W1"] + params["dec_b1"], 0.0)
    return (h_dec @ params["dec_W2"] + params["dec_b2"])[:, 0]


def loss_fn(params, peer_input, peer_mask, local_input, y, l2_alpha,
            pool_idx, n_pools, huber_delta, no_peers):
    """Huber loss with per-pool weighting + L2 reg."""
    pred = forward(params, peer_input, peer_mask, local_input, no_peers)
    residuals = pred - y
    abs_r = jnp.abs(residuals)
    huber_vals = jnp.where(abs_r <= huber_delta, 0.5 * residuals ** 2,
                           huber_delta * (abs_r - 0.5 * huber_delta))

    # Per-pool mean loss, then average across active pools (handles LOO gaps)
    pool_counts = jnp.zeros(n_pools).at[pool_idx].add(jnp.ones_like(pool_idx, dtype=jnp.float32))
    active = (pool_counts > 0).astype(jnp.float32)
    n_active = jnp.maximum(jnp.sum(active), 1.0)
    pool_counts = jnp.maximum(pool_counts, 1.0)
    pool_sums = jnp.zeros(n_pools).at[pool_idx].add(huber_vals)
    data_loss = jnp.sum((pool_sums / pool_counts) * active) / n_active

    reg = sum(jnp.sum(v ** 2) for k, v in params.items() if "W" in k)
    return data_loss + l2_alpha * reg


# n_pools (arg 7): static for jnp.zeros shape; no_peers (arg 9): static for if/else in forward
_grad_fn = jax.jit(jax.value_and_grad(loss_fn), static_argnums=(7, 9))


# ---- Learnable cadence ----


def make_cadence_loss_fn(pool_coeffs, pool_gas, n_pools, no_peers):
    """Build a loss function with per-pool PCHIP coefficients closed over.

    The returned function is JIT-compiled. The Python loop over pools is
    unrolled at trace time, so each pool's coefficients are constants.

    The neural net predicts log(V_noise). V_arb comes from PCHIP at the
    current learnable log_cadence. Loss is Huber on log(V_arb + V_noise)
    vs log(V_obs).
    """
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    def loss_fn_cadence(params, peer_input, peer_mask, local_input, y_total,
                        sample_grid_days, pool_idx, l2_alpha, huber_delta):
        # Neural net predicts log(V_noise)
        log_v_noise = forward(params, peer_input, peer_mask, local_input, no_peers)

        # Compute V_arb per sample via PCHIP (loop unrolled at trace time)
        log_cadence = params["log_cadence"]
        n_samples = y_total.shape[0]
        v_arb = jnp.zeros(n_samples)

        for i in range(n_pools):
            v_arb_all = interpolate_pool_daily(
                pool_coeffs[i], log_cadence[i], pool_gas[i])
            # Index into this pool's daily V_arb; clip for safety on other pools' samples
            safe_days = jnp.clip(sample_grid_days, 0, v_arb_all.shape[0] - 1)
            v_arb = jnp.where(pool_idx == i, v_arb_all[safe_days], v_arb)

        # Combine: log(V_arb + V_noise) via numerically stable logaddexp
        log_v_arb = jnp.log(jnp.maximum(v_arb, 1e-10))
        log_v_total = jnp.logaddexp(log_v_arb, log_v_noise)

        # Huber loss with per-pool weighting
        residuals = log_v_total - y_total
        abs_r = jnp.abs(residuals)
        huber_vals = jnp.where(abs_r <= huber_delta, 0.5 * residuals ** 2,
                               huber_delta * (abs_r - 0.5 * huber_delta))

        pool_counts = jnp.zeros(n_pools).at[pool_idx].add(
            jnp.ones_like(pool_idx, dtype=jnp.float32))
        active = (pool_counts > 0).astype(jnp.float32)
        n_active = jnp.maximum(jnp.sum(active), 1.0)
        pool_counts = jnp.maximum(pool_counts, 1.0)
        pool_sums = jnp.zeros(n_pools).at[pool_idx].add(huber_vals)
        data_loss = jnp.sum((pool_sums / pool_counts) * active) / n_active

        reg = sum(jnp.sum(v ** 2) for k, v in params.items() if "W" in k)
        return data_loss + l2_alpha * reg

    grad_fn = jax.jit(jax.value_and_grad(loss_fn_cadence))
    return grad_fn


# ---- Training ----


def train(params, inputs, n_epochs, lr, l2_alpha, huber_delta=1.0,
          no_peers=False, verbose=True, grad_fn_override=None):
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    final_loss = float("inf")

    n_pools = int(inputs["n_pools"])
    pool_idx = inputs["pool_idx"]
    use_cadence = grad_fn_override is not None

    for epoch in range(n_epochs):
        if use_cadence:
            loss_val, grads = grad_fn_override(
                params, inputs["peer_input"], inputs["peer_mask"],
                inputs["local_input"], inputs["y_total"],
                inputs["sample_grid_days"], pool_idx, l2_alpha, huber_delta,
            )
        else:
            loss_val, grads = _grad_fn(
                params, inputs["peer_input"], inputs["peer_mask"],
                inputs["local_input"], inputs["y"], l2_alpha,
                pool_idx, n_pools, huber_delta, no_peers,
            )
        final_loss = float(loss_val)

        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            m_hat = m[k] / (1.0 - 0.9 ** (epoch + 1))
            v_hat = v[k] / (1.0 - 0.999 ** (epoch + 1))
            params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            if use_cadence:
                cads = np.exp(np.array(params["log_cadence"]))
                # Quick decomposition check: forward pass + V_arb at current cadences
                _lvn = np.array(forward(
                    params, inputs["peer_input"], inputs["peer_mask"],
                    inputs["local_input"], no_peers))
                _vn = np.exp(_lvn)
                _vo = np.exp(np.array(inputs["y_total"]))
                # Approximate arb fraction (use V_obs - V_noise as proxy to avoid PCHIP call)
                _arb_proxy = np.clip(1.0 - _vn / _vo, 0, None)
                _n_pathological = np.sum(_arb_proxy < -0.5)  # noise > 1.5x observed
                _n_bound = np.sum((cads <= 1.01) | (cads >= 59.9))
                print(f"    epoch {epoch:4d}  loss={final_loss:.6f}"
                      f"  cad=[{cads.min():.1f}-{np.median(cads):.1f}-{cads.max():.1f}]"
                      f"  |logVn|={np.mean(np.abs(_lvn)):.1f}"
                      f"  bound={_n_bound}")
            else:
                print(f"    epoch {epoch:4d}  loss={final_loss:.6f}")

    return params, final_loss


# ---- Evaluation ----


def evaluate(params, inputs, data, label="", no_peers=False,
             target_residual=False):
    """Per-pool R² on total volume and noise residual."""
    pred = np.array(forward(
        params, inputs["peer_input"], inputs["peer_mask"],
        inputs["local_input"], no_peers=no_peers,
    ))
    y = np.array(inputs["y"])
    v_arb = np.array(inputs["v_arb"])
    pool_idx = np.array(inputs["pool_idx"])

    log_v_arb = np.log(np.maximum(v_arb, 1e-6))

    if target_residual:
        # Model predicts noise residual directly
        resid_true = y
        resid_pred = pred
        y_total = y + log_v_arb       # reconstruct total for total R²
        pred_total = pred + log_v_arb
    else:
        # Model predicts total log_volume
        y_total = y
        pred_total = pred
        resid_true = y - log_v_arb
        resid_pred = pred - log_v_arb

    r2_total = {}
    r2_resid = {}
    pool_ids = data.get("pool_ids", [])

    for i in range(data["n_pools"]):
        mask = pool_idx == i
        if mask.sum() < 2:
            continue
        yt = y_total[mask]
        pt = pred_total[mask]
        ss_res_t = np.sum((yt - pt) ** 2)
        ss_tot_t = np.sum((yt - yt.mean()) ** 2)
        r2_total[i] = 1 - ss_res_t / max(ss_tot_t, 1e-10)

        rt = resid_true[mask]
        rp = resid_pred[mask]
        ss_res_r = np.sum((rt - rp) ** 2)
        ss_tot_r = np.sum((rt - rt.mean()) ** 2)
        r2_resid[i] = 1 - ss_res_r / max(ss_tot_r, 1e-10)

    def _med(d):
        v = [x for x in d.values() if np.isfinite(x)]
        return np.median(v) if v else float("nan")

    if label:
        print(f"\n  {label}:")
    for i in range(data["n_pools"]):
        if i in r2_total and i < len(pool_ids):
            pid = pool_ids[i]
            print(f"    {pid[:16]}  total={r2_total[i]:.3f}  resid={r2_resid[i]:.3f}")

    med_total = _med(r2_total)
    med_resid = _med(r2_resid)
    print(f"  Median R² total={med_total:.4f}  resid={med_resid:.4f}")
    return med_total, med_resid, r2_total, r2_resid


def _compute_cadence_decomposition(params, inputs, data, no_peers=False):
    """Compute V_arb, V_noise, and predictions for cadence mode. Returns numpy arrays."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    log_v_noise = np.array(forward(
        params, inputs["peer_input"], inputs["peer_mask"],
        inputs["local_input"], no_peers=no_peers,
    ))
    y_total = np.array(inputs["y_total"])
    pool_idx = np.array(inputs["pool_idx"])
    sample_grid_days = np.array(inputs["sample_grid_days"])

    pool_coeffs = data["pool_coeffs"]
    pool_gas = data["pool_gas"]
    log_cadence = np.array(params["log_cadence"])
    n_pools = data["n_pools"]

    v_arb = np.zeros(len(y_total))
    for i in range(n_pools):
        mask = pool_idx == i
        if not mask.any():
            continue
        v_arb_all = np.array(interpolate_pool_daily(
            pool_coeffs[i], jnp.float64(log_cadence[i]), pool_gas[i]))
        v_arb[mask] = v_arb_all[sample_grid_days[mask]]

    v_obs = np.exp(y_total)
    v_noise = np.exp(log_v_noise)
    log_v_arb = np.log(np.maximum(v_arb, 1e-10))
    pred_total = np.logaddexp(log_v_arb, log_v_noise)

    return {
        "v_arb": v_arb, "v_noise": v_noise, "v_obs": v_obs,
        "log_v_noise": log_v_noise, "log_v_arb": log_v_arb,
        "pred_total": pred_total, "y_total": y_total,
        "pool_idx": pool_idx, "log_cadence": log_cadence,
    }


def evaluate_cadence(params, inputs, data, label="", no_peers=False):
    """Evaluate with learned cadence: per-pool R², decomposition diagnostics."""
    dec = _compute_cadence_decomposition(params, inputs, data, no_peers)
    pool_ids = data.get("pool_ids", [])
    init_cads = data["init_log_cadences"]
    n_pools = data["n_pools"]

    if label:
        print(f"\n  {label}:")
    print(f"    {'Pool'[:16]:16s} {'R²':>6s} {'Cad init':>8s} {'→learn':>7s}"
          f" {'Arb%':>6s} {'Noise%':>7s} {'logVn μ':>7s} {'logVn σ':>7s} {'Flag':>5s}")
    print(f"    {'-'*80}")

    r2_total = {}
    pool_diag = []
    for i in range(n_pools):
        mask = dec["pool_idx"] == i
        if mask.sum() < 2:
            continue
        yt = dec["y_total"][mask]
        pt = dec["pred_total"][mask]
        ss_res = np.sum((yt - pt) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_total[i] = 1 - ss_res / max(ss_tot, 1e-10)

        pid = pool_ids[i] if i < len(pool_ids) else f"pool_{i}"
        cad_init = np.exp(init_cads[i])
        cad_learned = np.exp(dec["log_cadence"][i])

        va = dec["v_arb"][mask]
        vo = dec["v_obs"][mask]
        vn = dec["v_noise"][mask]
        lvn = dec["log_v_noise"][mask]

        arb_pct = np.median(va / vo) * 100
        noise_pct = np.median(vn / vo) * 100
        lvn_mu = np.mean(lvn)
        lvn_std = np.std(lvn)

        # Flags
        flags = []
        if arb_pct > 150:
            flags.append("A")   # arb dominates
        if cad_learned <= 1.01 or cad_learned >= 59.9:
            flags.append("B")   # cadence at bound
        if r2_total[i] < 0:
            flags.append("X")   # negative R²

        flag_str = "".join(flags) if flags else ""
        pool_diag.append({
            "idx": i, "pid": pid, "r2": r2_total[i],
            "cad_init": cad_init, "cad_learned": cad_learned,
            "arb_pct": arb_pct, "noise_pct": noise_pct,
            "lvn_mu": lvn_mu, "lvn_std": lvn_std, "flags": flag_str,
        })

        print(f"    {pid[:16]:16s} {r2_total[i]:6.3f} {cad_init:7.1f}m {cad_learned:6.1f}m"
              f" {arb_pct:6.0f}% {noise_pct:6.0f}% {lvn_mu:7.1f} {lvn_std:7.2f}"
              f" {flag_str:>5s}")

    # ── Summary statistics ──
    vals = [x for x in r2_total.values() if np.isfinite(x)]
    med_r2 = np.median(vals) if vals else float("nan")
    cads = np.exp(dec["log_cadence"])

    n_pathological = sum(1 for d in pool_diag if d["arb_pct"] > 150)
    n_at_bound = sum(1 for d in pool_diag
                     if d["cad_learned"] <= 1.01 or d["cad_learned"] >= 59.9)
    n_negative_r2 = sum(1 for d in pool_diag if d["r2"] < 0)
    healthy = [d for d in pool_diag if d["arb_pct"] <= 150 and d["r2"] > 0]
    med_r2_healthy = (np.median([d["r2"] for d in healthy])
                      if healthy else float("nan"))

    print(f"\n    ── Summary ──")
    print(f"    Median R² total:   {med_r2:.4f}  (healthy only: {med_r2_healthy:.4f})")
    print(f"    Cadence range:     {cads.min():.1f} - {np.median(cads):.1f}"
          f" - {cads.max():.1f} min")
    print(f"    Decomposition:     {len(pool_diag) - n_pathological}/{len(pool_diag)}"
          f" healthy (arb≤150%),  {n_pathological} pathological")
    print(f"    Cadence at bounds: {n_at_bound}/{len(pool_diag)}"
          f"  (≤1min or ≥60min)")
    print(f"    Negative R²:       {n_negative_r2}/{len(pool_diag)}")
    print(f"    Flags: A=arb>150%, B=cadence at bound, X=negative R²")

    return med_r2, r2_total, pool_diag


def print_cadence_comparison(train_diag, eval_diag):
    """Print train vs eval diagnostic comparison."""
    train_map = {d["pid"]: d for d in train_diag}
    eval_map = {d["pid"]: d for d in eval_diag}
    all_pids = sorted(set(train_map) | set(eval_map))

    print(f"\n    ── Train vs Eval Gap ──")
    print(f"    {'Pool'[:16]:16s} {'R² trn':>7s} {'R² eval':>7s} {'Gap':>6s}"
          f" {'ArbTrn%':>7s} {'ArbEval%':>8s}")
    print(f"    {'-'*55}")

    gaps = []
    for pid in all_pids:
        td = train_map.get(pid)
        ed = eval_map.get(pid)
        if td is None or ed is None:
            continue
        gap = td["r2"] - ed["r2"]
        gaps.append(gap)
        flag = " ***" if abs(gap) > 0.5 else ""
        print(f"    {pid[:16]:16s} {td['r2']:7.3f} {ed['r2']:7.3f} {gap:+6.3f}"
              f" {td['arb_pct']:6.0f}% {ed['arb_pct']:7.0f}%{flag}")

    if gaps:
        print(f"    Median gap: {np.median(gaps):+.3f}  "
              f"Mean gap: {np.mean(gaps):+.3f}  "
              f"Max gap: {max(gaps):+.3f}")


# Keys indexed by sample (shape[0] == n_samples)
_SAMPLE_KEYS = {
    "pf_vol_lag1", "pf_vol_lag2", "pf_vol_change", "pf_tvl", "pf_volatility",
    "peer_mask", "lf_own_vol_lag1", "lf_own_vol_lag2", "lf_own_vol_change",
    "lf_own_tvl", "lf_own_volatility", "lf_dow_sin", "lf_dow_cos",
    "lf_tvl_x_vola", "lf_tvl_x_fee", "lf_vola_x_fee",
    "lf_cross_vol_tok_a", "lf_cross_vol_tok_b", "lf_cross_vol_chain",
    "lf_market_vol", "lf_cross_mom_tok_a", "lf_cross_mom_tok_b",
    "lf_cross_mom_chain",
    "y_total", "y_residual", "v_arb", "sample_grid_days",
    "x_obs_reduced", "x_obs_cross",
    "pool_idx", "day_idx",
}


def _subset(d, mask):
    """Subset sample-indexed arrays by boolean mask."""
    out = {}
    for k, v in d.items():
        if k in _SAMPLE_KEYS and isinstance(v, np.ndarray):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


# ---- Temporal split ----


def run_temporal(data, feat_cfg, hparams, split_frac=0.7):
    """Train on first split_frac of days, eval on rest."""
    day_idx = data["day_idx"]
    split_day = int(day_idx.max() * split_frac)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day

    train_data = _subset(data, train_mask)
    eval_data = _subset(data, eval_mask)

    train_inputs = assemble_inputs(train_data, feat_cfg)
    eval_inputs = assemble_inputs(eval_data, feat_cfg)

    encoder_type = hparams.get("encoder_type", "mlp")
    no_peers = hparams.get("no_peers", False)
    huber_delta = hparams.get("huber_delta", 1.0)
    target_residual = feat_cfg.get("target_residual", False)
    learn_cadence = hparams.get("learn_cadence", False)

    n_pf = train_inputs["n_peer_feat"]
    n_lf = train_inputs["n_local_feat"]
    n_params = sum(v.size for v in init_params(
        jax.random.PRNGKey(0), n_pf, n_lf, hparams["hidden"], hparams["d_embed"],
        encoder_type=encoder_type,
    ).values())

    print(f"  Train: {int(train_mask.sum())}, Eval: {int(eval_mask.sum())}, "
          f"peer_feat={n_pf}, local_feat={n_lf}, params={n_params}")
    if learn_cadence:
        print(f"  learn_cadence=True (joint cadence+noise optimization)")
    if target_residual:
        print(f"  target=residual (log_vol - log_V_arb)")
    if encoder_type != "mlp":
        print(f"  encoder_type={encoder_type}")
    if no_peers:
        print(f"  no_peers=True (decoder-only ablation)")
    if huber_delta != 1.0:
        print(f"  huber_delta={huber_delta}")

    params = init_params(
        jax.random.PRNGKey(42), n_pf, n_lf, hparams["hidden"], hparams["d_embed"],
        encoder_type=encoder_type,
    )

    if learn_cadence:
        # Add learnable cadence, initialized from Option C
        params["log_cadence"] = jnp.array(data["init_log_cadences"])
        init_cads = np.exp(data["init_log_cadences"])
        print(f"  Init cadence: {init_cads.min():.1f}-{np.median(init_cads):.1f}"
              f"-{init_cads.max():.1f} min")

        # Warm-start decoder to predict noise residual (log_vol - log_V_arb)
        # using the Option C V_arb as the initial target
        ws_inputs = dict(train_inputs)
        ws_inputs["y"] = train_inputs["y_total"] - jnp.log(
            jnp.maximum(train_inputs["v_arb"], 1e-6))
        params = warm_start_decoder(params, ws_inputs, hparams["d_embed"])

        grad_fn = make_cadence_loss_fn(
            data["pool_coeffs"], data["pool_gas"],
            data["n_pools"], no_peers)

        print("  Compiling cadence loss (may take a moment)...")
        t0 = time.time()
        params, _ = train(
            params, train_inputs, hparams["n_epochs"], hparams["lr"],
            hparams["l2_alpha"], huber_delta=huber_delta, no_peers=no_peers,
            grad_fn_override=grad_fn,
        )
        print(f"  Training: {time.time() - t0:.1f}s")

        print("\n  --- Train ---")
        _, _, train_diag = evaluate_cadence(
            params, train_inputs, data, no_peers=no_peers)
        print("\n  --- Eval ---")
        _, _, eval_diag = evaluate_cadence(
            params, eval_inputs, data, no_peers=no_peers)
        print_cadence_comparison(train_diag, eval_diag)
    else:
        params = warm_start_decoder(params, train_inputs, hparams["d_embed"])
        t0 = time.time()
        params, _ = train(
            params, train_inputs, hparams["n_epochs"], hparams["lr"],
            hparams["l2_alpha"], huber_delta=huber_delta, no_peers=no_peers,
        )
        print(f"  Training: {time.time() - t0:.1f}s")

        eval_kw = dict(no_peers=no_peers, target_residual=target_residual)
        print("\n  --- Train ---")
        evaluate(params, train_inputs, data, **eval_kw)
        print("\n  --- Eval ---")
        _, med_resid_eval, _, _ = evaluate(params, eval_inputs, data, **eval_kw)

    return params


# ---- LOO cross-validation ----


def run_loo(data, feat_cfg, hparams):
    """Leave-one-pool-out: train on N-1 pools, evaluate on held-out pool.

    Tests cross-pool generalization — can the shared encoder+decoder predict
    volume for a pool it has never optimized on? The held-out pool's volume
    is still observable as peer features for training pools.

    Note: normalization stats are computed on the full dataset. With N=36
    the leakage from including one held-out pool is ~3% on mean/std.
    """
    n_pools = data["n_pools"]
    pool_idx = data["pool_idx"]
    pool_ids = data.get("pool_ids", [])

    encoder_type = hparams.get("encoder_type", "mlp")
    no_peers = hparams.get("no_peers", False)
    huber_delta = hparams.get("huber_delta", 1.0)
    target_residual = feat_cfg.get("target_residual", False)
    d_embed = hparams["d_embed"]

    r2_total_all = {}
    r2_resid_all = {}

    for held_out in range(n_pools):
        pid = pool_ids[held_out] if held_out < len(pool_ids) else f"pool_{held_out}"

        train_mask = pool_idx != held_out
        eval_mask = pool_idx == held_out
        n_eval = int(eval_mask.sum())

        if n_eval < 2:
            print(f"  [{held_out:2d}] {pid[:16]}: skipped ({n_eval} samples)")
            continue

        train_data = _subset(data, train_mask)
        eval_data = _subset(data, eval_mask)

        train_inputs = assemble_inputs(train_data, feat_cfg)
        eval_inputs = assemble_inputs(eval_data, feat_cfg)

        params = init_params(
            jax.random.PRNGKey(42),
            train_inputs["n_peer_feat"], train_inputs["n_local_feat"],
            hparams["hidden"], d_embed,
            encoder_type=encoder_type,
        )
        params = warm_start_decoder(params, train_inputs, d_embed)

        params, _ = train(
            params, train_inputs, hparams["n_epochs"],
            hparams["lr"], hparams["l2_alpha"],
            huber_delta=huber_delta, no_peers=no_peers,
            verbose=False,
        )

        # Evaluate on held-out pool
        pred = np.array(forward(
            params, eval_inputs["peer_input"], eval_inputs["peer_mask"],
            eval_inputs["local_input"], no_peers=no_peers,
        ))
        y = np.array(eval_inputs["y"])
        v_arb = np.array(eval_inputs["v_arb"])
        log_v_arb = np.log(np.maximum(v_arb, 1e-6))

        if target_residual:
            resid_true, resid_pred = y, pred
            y_total = y + log_v_arb
            pred_total = pred + log_v_arb
        else:
            y_total, pred_total = y, pred
            resid_true = y - log_v_arb
            resid_pred = pred - log_v_arb

        ss_res = np.sum((y_total - pred_total) ** 2)
        ss_tot = np.sum((y_total - y_total.mean()) ** 2)
        r2_t = 1 - ss_res / max(ss_tot, 1e-10)

        ss_res_r = np.sum((resid_true - resid_pred) ** 2)
        ss_tot_r = np.sum((resid_true - resid_true.mean()) ** 2)
        r2_r = 1 - ss_res_r / max(ss_tot_r, 1e-10)

        r2_total_all[held_out] = r2_t
        r2_resid_all[held_out] = r2_r

        print(f"  [{held_out:2d}] {pid[:16]}: total={r2_t:.3f}  resid={r2_r:.3f}  (n={n_eval})")

    def _med(d):
        v = [x for x in d.values() if np.isfinite(x)]
        return np.median(v) if v else float("nan")

    med_total = _med(r2_total_all)
    med_resid = _med(r2_resid_all)
    print(f"\n  LOO Median R² total={med_total:.4f}  resid={med_resid:.4f}")
    print(f"  ({len(r2_total_all)} pools evaluated)")

    return med_total, med_resid


# ---- Optuna ----


_FEAT_KEYS = [
    "peer_vol_lag2", "peer_vol_change", "peer_tvl", "peer_volatility",
    "own_vol_lag2", "own_vol_change", "own_tvl", "own_volatility",
    "rel_same_chain", "rel_tvl_ratio", "rel_fee_ratio", "minimal_encoder",
    "interactions", "cross_pool_vol", "cross_pool_momentum",
]


def run_optuna(data, n_trials, target_residual=False):
    import optuna

    day_idx = data["day_idx"]
    split_day = int(day_idx.max() * 0.7)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day

    train_data = _subset(data, train_mask)
    eval_data = _subset(data, eval_mask)

    def objective(trial):
        feat_cfg = {
            "peer_vol_lag2": trial.suggest_categorical("peer_vol_lag2", [True, False]),
            "peer_vol_change": trial.suggest_categorical("peer_vol_change", [True, False]),
            "peer_tvl": trial.suggest_categorical("peer_tvl", [True, False]),
            "peer_volatility": trial.suggest_categorical("peer_volatility", [True, False]),
            "own_vol_lag2": trial.suggest_categorical("own_vol_lag2", [True, False]),
            "own_vol_change": trial.suggest_categorical("own_vol_change", [True, False]),
            "own_tvl": trial.suggest_categorical("own_tvl", [True, False]),
            "own_volatility": trial.suggest_categorical("own_volatility", [True, False]),
            "rel_same_chain": trial.suggest_categorical("rel_same_chain", [True, False]),
            "rel_tvl_ratio": trial.suggest_categorical("rel_tvl_ratio", [True, False]),
            "rel_fee_ratio": trial.suggest_categorical("rel_fee_ratio", [True, False]),
            "minimal_encoder": trial.suggest_categorical("minimal_encoder", [True, False]),
            "interactions": trial.suggest_categorical("interactions", [True, False]),
            "cross_pool_vol": trial.suggest_categorical("cross_pool_vol", [True, False]),
            "cross_pool_momentum": trial.suggest_categorical("cross_pool_momentum", [True, False]),
            "target_residual": target_residual,
        }
        hparams = {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64, 128]),
            "d_embed": trial.suggest_categorical("d_embed", [4, 8, 16, 32]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "l2_alpha": trial.suggest_float("l2_alpha", 1e-5, 1e-1, log=True),
            "n_epochs": trial.suggest_categorical("n_epochs", [500, 1000, 2000]),
            "encoder_type": trial.suggest_categorical("encoder_type", ["mlp", "linear"]),
            "huber_delta": trial.suggest_categorical("huber_delta", [0.5, 1.0, 1.5, 2.0]),
            "no_peers": trial.suggest_categorical("no_peers", [True, False]),
        }

        train_inputs = assemble_inputs(train_data, feat_cfg)
        eval_inputs = assemble_inputs(eval_data, feat_cfg)

        params = init_params(
            jax.random.PRNGKey(42),
            train_inputs["n_peer_feat"], train_inputs["n_local_feat"],
            hparams["hidden"], hparams["d_embed"],
            encoder_type=hparams["encoder_type"],
        )
        params = warm_start_decoder(params, train_inputs, hparams["d_embed"])
        params, _ = train(
            params, train_inputs, hparams["n_epochs"],
            hparams["lr"], hparams["l2_alpha"],
            huber_delta=hparams["huber_delta"],
            no_peers=hparams["no_peers"],
            verbose=False,
        )

        # Eval R² on noise residual
        _tgt_resid = feat_cfg.get("target_residual", False)
        pred = np.array(forward(
            params, eval_inputs["peer_input"], eval_inputs["peer_mask"],
            eval_inputs["local_input"], no_peers=hparams["no_peers"],
        ))
        y = np.array(eval_inputs["y"])
        v_arb = np.array(eval_inputs["v_arb"])
        log_v_arb = np.log(np.maximum(v_arb, 1e-6))
        pool_idx = np.array(eval_data["pool_idx"])

        r2_resids = []
        r2_totals = []
        for i in range(data["n_pools"]):
            mask = pool_idx == i
            if mask.sum() < 2:
                continue
            yi = y[mask]
            pi = pred[mask]
            lva = log_v_arb[mask]

            if _tgt_resid:
                resid_true, resid_pred = yi, pi
                yt, pt = yi + lva, pi + lva
            else:
                yt, pt = yi, pi
                resid_true, resid_pred = yi - lva, pi - lva

            ss_res = np.sum((yt - pt) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2_totals.append(1 - ss_res / max(ss_tot, 1e-10))

            ss_res_r = np.sum((resid_true - resid_pred) ** 2)
            ss_tot_r = np.sum((resid_true - resid_true.mean()) ** 2)
            r2_resids.append(1 - ss_res_r / max(ss_tot_r, 1e-10))

        med_resid = float(np.median(r2_resids)) if r2_resids else -10.0
        med_total = float(np.median(r2_totals)) if r2_totals else -10.0

        trial.set_user_attr("med_total_r2", med_total)
        n_feat = sum(1 for k in _FEAT_KEYS if feat_cfg.get(k))
        print(f"  Trial {trial.number}: resid={med_resid:.4f} total={med_total:.4f} "
              f"enc={hparams['encoder_type']} h={hparams['hidden']} d={hparams['d_embed']} "
              f"hub={hparams['huber_delta']} "
              f"{'no_peers ' if hparams['no_peers'] else ''}"
              f"lr={hparams['lr']:.1e} a={hparams['l2_alpha']:.1e} "
              f"ep={hparams['n_epochs']} feat={n_feat}/15"
              f"{' minimal' if feat_cfg.get('minimal_encoder') else ''}")

        return med_resid

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{'='*70}")
    print("Optuna Results")
    print(f"{'='*70}")
    print(f"  Best eval noise resid R²: {study.best_value:.4f}")
    print(f"  Best total R²: {study.best_trial.user_attrs['med_total_r2']:.4f}")
    print(f"  Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}: {v}")

    print(f"\n  Top 10:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999,
                    reverse=True)
    for t in trials[:10]:
        if t.value is not None:
            feats = sum(1 for k in _FEAT_KEYS if t.params.get(k))
            print(f"    #{t.number}: resid={t.value:.4f} "
                  f"total={t.user_attrs.get('med_total_r2', '?'):.4f} "
                  f"enc={t.params['encoder_type']} "
                  f"h={t.params['hidden']} d={t.params['d_embed']} "
                  f"feat={feats}/15")

    return study


def run_optuna_cadence(data, n_trials):
    """Optuna sweep with learnable cadence. Optimizes median eval total R²."""
    import optuna
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    day_idx = data["day_idx"]
    split_day = int(day_idx.max() * 0.7)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day

    train_data = _subset(data, train_mask)
    eval_data = _subset(data, eval_mask)

    pool_coeffs = data["pool_coeffs"]
    pool_gas = data["pool_gas"]
    n_pools = data["n_pools"]

    # Pre-build grad_fn closures for (no_peers=True, no_peers=False)
    # to avoid recompiling on every trial with the same no_peers setting
    _grad_fn_cache = {}

    def _get_grad_fn(no_peers):
        if no_peers not in _grad_fn_cache:
            _grad_fn_cache[no_peers] = make_cadence_loss_fn(
                pool_coeffs, pool_gas, n_pools, no_peers)
        return _grad_fn_cache[no_peers]

    def objective(trial):
        feat_cfg = {
            "peer_vol_lag2": trial.suggest_categorical("peer_vol_lag2", [True, False]),
            "peer_vol_change": trial.suggest_categorical("peer_vol_change", [True, False]),
            "peer_tvl": trial.suggest_categorical("peer_tvl", [True, False]),
            "peer_volatility": trial.suggest_categorical("peer_volatility", [True, False]),
            "own_vol_lag2": trial.suggest_categorical("own_vol_lag2", [True, False]),
            "own_vol_change": trial.suggest_categorical("own_vol_change", [True, False]),
            "own_tvl": trial.suggest_categorical("own_tvl", [True, False]),
            "own_volatility": trial.suggest_categorical("own_volatility", [True, False]),
            "rel_same_chain": trial.suggest_categorical("rel_same_chain", [True, False]),
            "rel_tvl_ratio": trial.suggest_categorical("rel_tvl_ratio", [True, False]),
            "rel_fee_ratio": trial.suggest_categorical("rel_fee_ratio", [True, False]),
            "minimal_encoder": trial.suggest_categorical("minimal_encoder", [True, False]),
            "interactions": trial.suggest_categorical("interactions", [True, False]),
            "cross_pool_vol": trial.suggest_categorical("cross_pool_vol", [True, False]),
            "cross_pool_momentum": trial.suggest_categorical("cross_pool_momentum", [True, False]),
            "x_obs_mode": trial.suggest_categorical("x_obs_mode", ["none", "reduced", "cross"]),
        }
        hparams = {
            "hidden": trial.suggest_categorical("hidden", [16, 32, 64]),
            "d_embed": trial.suggest_categorical("d_embed", [4, 8, 16]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "l2_alpha": trial.suggest_float("l2_alpha", 1e-5, 1e-2, log=True),
            "n_epochs": trial.suggest_categorical("n_epochs", [500, 1000, 2000]),
            "encoder_type": trial.suggest_categorical("encoder_type", ["mlp", "linear"]),
            "huber_delta": trial.suggest_categorical("huber_delta", [0.5, 1.0, 1.5]),
            "no_peers": trial.suggest_categorical("no_peers", [True, False]),
        }

        no_peers = hparams["no_peers"]
        train_inputs = assemble_inputs(train_data, feat_cfg)
        eval_inputs = assemble_inputs(eval_data, feat_cfg)

        params = init_params(
            jax.random.PRNGKey(42),
            train_inputs["n_peer_feat"], train_inputs["n_local_feat"],
            hparams["hidden"], hparams["d_embed"],
            encoder_type=hparams["encoder_type"],
        )
        # Learnable cadence from Option C init
        params["log_cadence"] = jnp.array(data["init_log_cadences"])

        # Warm-start decoder on noise residual
        ws_inputs = dict(train_inputs)
        ws_inputs["y"] = train_inputs["y_total"] - jnp.log(
            jnp.maximum(train_inputs["v_arb"], 1e-6))
        params = warm_start_decoder(params, ws_inputs, hparams["d_embed"])

        grad_fn = _get_grad_fn(no_peers)
        params, _ = train(
            params, train_inputs, hparams["n_epochs"],
            hparams["lr"], hparams["l2_alpha"],
            huber_delta=hparams["huber_delta"], no_peers=no_peers,
            verbose=False, grad_fn_override=grad_fn,
        )

        # Eval: compute V_arb at learned cadences, combine with net
        log_v_noise = np.array(forward(
            params, eval_inputs["peer_input"], eval_inputs["peer_mask"],
            eval_inputs["local_input"], no_peers=no_peers,
        ))
        y_total = np.array(eval_inputs["y_total"])
        pool_idx = np.array(eval_data["pool_idx"])
        sample_grid_days = np.array(eval_inputs["sample_grid_days"])
        log_cadence = np.array(params["log_cadence"])

        v_arb = np.zeros(len(y_total))
        for i in range(n_pools):
            mask = pool_idx == i
            if not mask.any():
                continue
            v_arb_all = np.array(interpolate_pool_daily(
                pool_coeffs[i], jnp.float64(log_cadence[i]), pool_gas[i]))
            v_arb[mask] = v_arb_all[sample_grid_days[mask]]

        log_v_arb = np.log(np.maximum(v_arb, 1e-10))
        pred_total = np.logaddexp(log_v_arb, log_v_noise)

        r2_totals = []
        for i in range(n_pools):
            mask = pool_idx == i
            if mask.sum() < 2:
                continue
            yt = y_total[mask]
            pt = pred_total[mask]
            ss_res = np.sum((yt - pt) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2_totals.append(1 - ss_res / max(ss_tot, 1e-10))

        med_total = float(np.median(r2_totals)) if r2_totals else -10.0

        cads = np.exp(log_cadence)
        trial.set_user_attr("med_total_r2", med_total)
        trial.set_user_attr("cad_median", float(np.median(cads)))
        n_feat = sum(1 for k in _FEAT_KEYS if feat_cfg.get(k))
        print(f"  Trial {trial.number}: total={med_total:.4f} "
              f"enc={hparams['encoder_type']} h={hparams['hidden']} d={hparams['d_embed']} "
              f"hub={hparams['huber_delta']} "
              f"{'no_peers ' if no_peers else ''}"
              f"lr={hparams['lr']:.1e} a={hparams['l2_alpha']:.1e} "
              f"ep={hparams['n_epochs']} feat={n_feat}/15"
              f" cad=[{cads.min():.0f}-{np.median(cads):.0f}-{cads.max():.0f}]")

        return med_total

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{'='*70}")
    print("Optuna Results (learn_cadence)")
    print(f"{'='*70}")
    print(f"  Best eval total R²: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}: {v}")

    print(f"\n  Top 10:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999,
                    reverse=True)
    for t in trials[:10]:
        if t.value is not None:
            feats = sum(1 for k in _FEAT_KEYS if t.params.get(k))
            cad_med = t.user_attrs.get("cad_median", "?")
            print(f"    #{t.number}: total={t.value:.4f} "
                  f"enc={t.params['encoder_type']} "
                  f"h={t.params['hidden']} d={t.params['d_embed']} "
                  f"feat={feats}/15 cad_med={cad_med:.0f}")

    return study


# ---- Main ----


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", type=int, default=0)
    parser.add_argument("--loo", action="store_true")
    # Architecture
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--d-embed", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2-alpha", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--encoder-type", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--no-peers", action="store_true",
                        help="Decoder-only ablation (zero peer summary)")
    parser.add_argument("--learn-cadence", action="store_true",
                        help="Jointly optimize per-pool arb cadence via PCHIP")
    parser.add_argument("--x-obs", choices=["none", "reduced", "cross"],
                        default="none",
                        help="Append Option C x_obs covariates to decoder: "
                             "none, reduced (4: intercept,tvl,dow), "
                             "cross (7: +peer volumes)")
    parser.add_argument("--minimal-encoder", action="store_true",
                        help="7-feature encoder (fee, tvl, overlap, same_chain) "
                             "instead of full attributes")
    parser.add_argument("--target-residual", action="store_true",
                        help="Train on noise residual (log_vol - log_V_arb) "
                             "instead of total log_volume")
    # Feature flags
    parser.add_argument("--peer-vol-lag2", action="store_true")
    parser.add_argument("--peer-vol-change", action="store_true")
    parser.add_argument("--peer-tvl", action="store_true")
    parser.add_argument("--peer-volatility", action="store_true")
    parser.add_argument("--own-vol-lag2", action="store_true")
    parser.add_argument("--own-vol-change", action="store_true")
    parser.add_argument("--own-tvl", action="store_true")
    parser.add_argument("--own-volatility", action="store_true")
    parser.add_argument("--interactions", action="store_true",
                        help="tvl×vola, tvl×fee, vola×fee interaction terms")
    parser.add_argument("--cross-pool-vol", action="store_true",
                        help="Token-peer, chain-peer, market volume aggregates")
    parser.add_argument("--cross-pool-momentum", action="store_true",
                        help="Peer volume change momentum features")
    parser.add_argument("--all-features", action="store_true",
                        help="Enable all optional features")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    feat_cfg = {
        "peer_vol_lag2": args.peer_vol_lag2 or args.all_features,
        "peer_vol_change": args.peer_vol_change or args.all_features,
        "peer_tvl": args.peer_tvl or args.all_features,
        "peer_volatility": args.peer_volatility or args.all_features,
        "own_vol_lag2": args.own_vol_lag2 or args.all_features,
        "own_vol_change": args.own_vol_change or args.all_features,
        "own_tvl": args.own_tvl or args.all_features,
        "own_volatility": args.own_volatility or args.all_features,
        # Relational features: always on for CLI, searchable in Optuna
        "rel_same_chain": True,
        "rel_tvl_ratio": True,
        "rel_fee_ratio": True,
        "interactions": args.interactions or args.all_features,
        "cross_pool_vol": args.cross_pool_vol or args.all_features,
        "cross_pool_momentum": args.cross_pool_momentum or args.all_features,
        "minimal_encoder": args.minimal_encoder,
        "target_residual": args.target_residual,
        "x_obs_mode": args.x_obs,
    }
    hparams = {
        "hidden": args.hidden,
        "d_embed": args.d_embed,
        "lr": args.lr,
        "l2_alpha": args.l2_alpha,
        "n_epochs": args.epochs,
        "encoder_type": args.encoder_type,
        "huber_delta": args.huber_delta,
        "no_peers": args.no_peers,
        "learn_cadence": args.learn_cadence,
    }

    print("=" * 70)
    print("DeepSets v2: Total Volume Target + Noise Residual Eval")
    feat_on = [k for k, v in feat_cfg.items() if v]
    print(f"  Optional features: {feat_on or 'none'}")
    print(f"  Architecture: {hparams}")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()

    print("\nBuilding features...")
    t0 = time.time()
    data = build_all_features(matched_clean, option_c_clean)
    print(f"  {len(data['pool_idx'])} samples, {data['n_pools']} pools, "
          f"{time.time() - t0:.1f}s")

    if args.loo:
        print(f"\n{'='*70}")
        print("Leave-One-Pool-Out Cross-Validation")
        print(f"{'='*70}")
        run_loo(data, feat_cfg, hparams)
    elif args.tune > 0 and args.learn_cadence:
        run_optuna_cadence(data, args.tune)
    elif args.tune > 0:
        run_optuna(data, args.tune, target_residual=args.target_residual)
    else:
        print(f"\n{'='*70}")
        print("Temporal split (70/30)")
        print(f"{'='*70}")
        run_temporal(data, feat_cfg, hparams)

    print(f"\n  Baselines for comparison:")
    print(f"    Option C on residual:  median R² = 0.060")
    print(f"    Ridge+own on residual: median R² = 0.098")
    print(f"    Constant zero:         median R² = -0.083")


if __name__ == "__main__":
    main()
