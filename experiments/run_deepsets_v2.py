"""DeepSets v2: full feature menu with Optuna feature selection.

Trains on total log_volume, evaluates on both total volume and noise
residual (log_vol - log_V_arb). V_arb precomputed from Option C fits.

Feature menu:
  Peer (encoder) — always: peer_attr, target_attr, vol_lag1, overlap
                   optional: vol_lag2, vol_change, tvl, volatility
  Local (decoder) — always: target_attr, own_vol_lag1, dow_sin, dow_cos
                    optional: own_vol_lag2, own_vol_change, own_tvl, own_volatility

Usage:
  python experiments/run_deepsets_v2.py                    # defaults
  python experiments/run_deepsets_v2.py --tune 50          # Optuna
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

    for t, date in enumerate(date_list):
        weekday_arr[t] = pd.Timestamp(date).weekday()

    # Pool attributes (static)
    X_attr, attr_names, _ = build_pool_attributes(matched_clean)
    attr_mean = np.mean(X_attr, axis=0)
    attr_std = np.std(X_attr, axis=0)
    attr_std[attr_std < 1e-6] = 1.0
    X_attr_norm = ((X_attr - attr_mean) / attr_std).astype(np.float32)
    k_attr = X_attr_norm.shape[1]

    # Token overlap
    pool_tokens = {}
    for i, pid in enumerate(pool_ids):
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        pool_tokens[i] = {_canonicalize_token(t) for t in toks[:2]}

    n_peers = n_pools - 1
    peer_attrs = np.zeros((n_pools, n_peers, k_attr), dtype=np.float32)
    peer_overlap = np.zeros((n_pools, n_peers), dtype=np.float32)
    peer_col_idx = np.zeros((n_pools, n_peers), dtype=np.int32)

    for i in range(n_pools):
        peers = [j for j in range(n_pools) if j != i]
        for p, j in enumerate(peers):
            peer_attrs[i, p] = X_attr_norm[j]
            peer_overlap[i, p] = len(pool_tokens[i] & pool_tokens[j])
            peer_col_idx[i, p] = j

    # Standardization stats for volumes
    vol_mean = float(np.nanmean(vol_matrix))
    vol_std = float(np.nanstd(vol_matrix))
    tvl_mean = float(np.nanmean(tvl_matrix))
    tvl_std = float(np.nanstd(tvl_matrix))
    vola_mean = float(np.nanmean(volatility_matrix))
    vola_std = float(np.nanstd(volatility_matrix))

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

        y_total[s] = vol_matrix[t, i]
        v_arb_val = v_arb_matrix[t, i]
        v_arb_samples[s] = v_arb_val if np.isfinite(v_arb_val) else 1e-6

    return {
        # Static per-pool
        "peer_attrs": peer_attrs,       # (n_pools, n_peers, k_attr)
        "target_attrs": X_attr_norm,    # (n_pools, k_attr)
        "peer_overlap": peer_overlap,   # (n_pools, n_peers)
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
        # Targets
        "y_total": y_total,
        "v_arb": v_arb_samples,
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
    }


def assemble_inputs(data, feat_cfg):
    """Assemble encoder/decoder inputs based on feature config.

    Returns (peer_input, local_input, peer_mask, pool_idx, y, v_arb)
    all as JAX arrays ready for training.
    """
    n_samples = len(data["pool_idx"])
    n_peers = data["n_peers"]

    # ---- Peer encoder input: (n_samples, n_peers, n_feat) ----
    # Always: peer_attr, target_attr, vol_lag1, overlap
    pool_idx = data["pool_idx"]
    pa = data["peer_attrs"][pool_idx]  # (n_samples, n_peers, k_attr)
    ta = data["target_attrs"][pool_idx]  # (n_samples, k_attr)
    ta_broad = np.broadcast_to(ta[:, None, :], pa.shape)

    peer_parts = [
        pa, ta_broad,
        data["pf_vol_lag1"][:, :, None],
        data["peer_overlap"][pool_idx][:, :, None],
    ]

    if feat_cfg.get("peer_vol_lag2"):
        peer_parts.append(data["pf_vol_lag2"][:, :, None])
    if feat_cfg.get("peer_vol_change"):
        peer_parts.append(data["pf_vol_change"][:, :, None])
    if feat_cfg.get("peer_tvl"):
        peer_parts.append(data["pf_tvl"][:, :, None])
    if feat_cfg.get("peer_volatility"):
        peer_parts.append(data["pf_volatility"][:, :, None])

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

    local_input = np.concatenate(local_parts, axis=-1).astype(np.float32)

    return {
        "peer_input": jnp.array(peer_input),
        "local_input": jnp.array(local_input),
        "peer_mask": jnp.array(data["peer_mask"]),
        "y": jnp.array(data["y_total"]),
        "v_arb": jnp.array(data["v_arb"]),
        "pool_idx": jnp.array(pool_idx),
        "n_peer_feat": peer_input.shape[-1],
        "n_local_feat": local_input.shape[-1],
    }


# ---- Model ----


def init_params(key, n_peer_feat, n_local_feat, hidden, d_embed):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    dec_in = d_embed + n_local_feat
    return {
        "enc_W1": jax.random.normal(k1, (n_peer_feat, hidden)) * np.sqrt(2.0 / n_peer_feat),
        "enc_b1": jnp.zeros(hidden),
        "enc_W2": jax.random.normal(k2, (hidden, d_embed)) * np.sqrt(2.0 / hidden),
        "enc_b2": jnp.zeros(d_embed),
        "dec_W1": jax.random.normal(k3, (dec_in, hidden)) * np.sqrt(2.0 / dec_in),
        "dec_b1": jnp.zeros(hidden),
        "dec_W2": jax.random.normal(k4, (hidden, 1)) * 0.01,
        "dec_b2": jnp.zeros(1),
    }


def forward(params, peer_input, peer_mask, local_input):
    """Returns predicted log_volume (total) per sample."""
    batch, n_peers, _ = peer_input.shape

    flat = peer_input.reshape(-1, peer_input.shape[-1])
    h = jnp.maximum(flat @ params["enc_W1"] + params["enc_b1"], 0.0)
    h = h @ params["enc_W2"] + params["enc_b2"]
    h = h.reshape(batch, n_peers, -1)

    h_masked = h * peer_mask[:, :, None]
    n_valid = jnp.maximum(jnp.sum(peer_mask, axis=1, keepdims=True), 1.0)
    summary = jnp.sum(h_masked, axis=1) / n_valid

    dec_in = jnp.concatenate([summary, local_input], axis=-1)
    h_dec = jnp.maximum(dec_in @ params["dec_W1"] + params["dec_b1"], 0.0)
    return (h_dec @ params["dec_W2"] + params["dec_b2"])[:, 0]


def loss_fn(params, peer_input, peer_mask, local_input, y, l2_alpha):
    """MSE on total log_volume + L2 reg."""
    pred = forward(params, peer_input, peer_mask, local_input)
    mse = jnp.mean((pred - y) ** 2)
    reg = sum(jnp.sum(v ** 2) for k, v in params.items() if "W" in k)
    return mse + l2_alpha * reg


_grad_fn = jax.jit(jax.value_and_grad(loss_fn))


# ---- Training ----


def train(params, inputs, n_epochs, lr, l2_alpha, verbose=True):
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    final_loss = float("inf")

    for epoch in range(n_epochs):
        loss_val, grads = _grad_fn(
            params, inputs["peer_input"], inputs["peer_mask"],
            inputs["local_input"], inputs["y"], l2_alpha,
        )
        final_loss = float(loss_val)

        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            m_hat = m[k] / (1.0 - 0.9 ** (epoch + 1))
            v_hat = v[k] / (1.0 - 0.999 ** (epoch + 1))
            params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:4d}  loss={final_loss:.6f}")

    return params, final_loss


# ---- Evaluation ----


def evaluate(params, inputs, data, label=""):
    """Per-pool R² on total volume and noise residual."""
    pred_total = np.array(forward(
        params, inputs["peer_input"], inputs["peer_mask"], inputs["local_input"],
    ))
    y_total = np.array(inputs["y"])
    v_arb = np.array(inputs["v_arb"])
    pool_idx = np.array(data["pool_idx"]) if "pool_idx" in data else np.array(inputs["pool_idx"])

    # Noise residual: compare (pred - log(v_arb)) vs (y - log(v_arb))
    log_v_arb = np.log(np.maximum(v_arb, 1e-6))
    resid_true = y_total - log_v_arb
    resid_pred = pred_total - log_v_arb

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


# ---- Temporal split ----


def run_temporal(data, feat_cfg, hparams, split_frac=0.7):
    """Train on first split_frac of days, eval on rest."""
    day_idx = data["day_idx"]
    split_day = int(day_idx.max() * split_frac)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day

    def _subset(d, mask):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and len(v) == len(mask):
                out[k] = v[mask]
            else:
                out[k] = v
        return out

    train_data = _subset(data, train_mask)
    eval_data = _subset(data, eval_mask)

    train_inputs = assemble_inputs(train_data, feat_cfg)
    eval_inputs = assemble_inputs(eval_data, feat_cfg)

    n_pf = train_inputs["n_peer_feat"]
    n_lf = train_inputs["n_local_feat"]
    n_params = sum(v.size for v in init_params(
        jax.random.PRNGKey(0), n_pf, n_lf, hparams["hidden"], hparams["d_embed"],
    ).values())

    print(f"  Train: {int(train_mask.sum())}, Eval: {int(eval_mask.sum())}, "
          f"peer_feat={n_pf}, local_feat={n_lf}, params={n_params}")

    params = init_params(
        jax.random.PRNGKey(42), n_pf, n_lf, hparams["hidden"], hparams["d_embed"],
    )
    t0 = time.time()
    params, final_loss = train(
        params, train_inputs, hparams["n_epochs"], hparams["lr"], hparams["l2_alpha"],
    )
    print(f"  Training: {time.time() - t0:.1f}s")

    print("\n  --- Train ---")
    evaluate(params, train_inputs, data)
    print("\n  --- Eval ---")
    _, med_resid_eval, _, _ = evaluate(params, eval_inputs, data)

    return med_resid_eval


# ---- Optuna ----


def run_optuna(data, n_trials):
    import optuna

    day_idx = data["day_idx"]
    split_day = int(day_idx.max() * 0.7)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day

    def _subset(d, mask):
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and len(v) == len(mask):
                out[k] = v[mask]
            else:
                out[k] = v
        return out

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
        }
        hparams = {
            "hidden": trial.suggest_categorical("hidden", [8, 16, 32]),
            "d_embed": trial.suggest_categorical("d_embed", [4, 8, 16]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "l2_alpha": trial.suggest_float("l2_alpha", 1e-5, 1e-1, log=True),
            "n_epochs": trial.suggest_categorical("n_epochs", [500, 1000, 2000]),
        }

        train_inputs = assemble_inputs(train_data, feat_cfg)
        eval_inputs = assemble_inputs(eval_data, feat_cfg)

        params = init_params(
            jax.random.PRNGKey(42),
            train_inputs["n_peer_feat"], train_inputs["n_local_feat"],
            hparams["hidden"], hparams["d_embed"],
        )
        params, _ = train(params, train_inputs, hparams["n_epochs"],
                          hparams["lr"], hparams["l2_alpha"], verbose=False)

        # Eval R² on noise residual
        pred = np.array(forward(
            params, eval_inputs["peer_input"], eval_inputs["peer_mask"],
            eval_inputs["local_input"],
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
            yt = y[mask]
            pt = pred[mask]
            ss_res = np.sum((yt - pt) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2_totals.append(1 - ss_res / max(ss_tot, 1e-10))

            rt = yt - log_v_arb[mask]
            rp = pt - log_v_arb[mask]
            ss_res_r = np.sum((rt - rp) ** 2)
            ss_tot_r = np.sum((rt - rt.mean()) ** 2)
            r2_resids.append(1 - ss_res_r / max(ss_tot_r, 1e-10))

        med_resid = float(np.median(r2_resids)) if r2_resids else -10.0
        med_total = float(np.median(r2_totals)) if r2_totals else -10.0

        trial.set_user_attr("med_total_r2", med_total)
        n_feat = sum(feat_cfg.values())
        print(f"  Trial {trial.number}: resid={med_resid:.4f} total={med_total:.4f} "
              f"h={hparams['hidden']} d={hparams['d_embed']} "
              f"lr={hparams['lr']:.1e} a={hparams['l2_alpha']:.1e} "
              f"ep={hparams['n_epochs']} feat={n_feat}/8")

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
            feats = sum(1 for k in ["peer_vol_lag2", "peer_vol_change",
                                     "peer_tvl", "peer_volatility",
                                     "own_vol_lag2", "own_vol_change",
                                     "own_tvl", "own_volatility"]
                        if t.params.get(k))
            print(f"    #{t.number}: resid={t.value:.4f} "
                  f"total={t.user_attrs.get('med_total_r2', '?'):.4f} "
                  f"h={t.params['hidden']} d={t.params['d_embed']} "
                  f"feat={feats}/8")

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
    # Feature flags
    parser.add_argument("--peer-vol-lag2", action="store_true")
    parser.add_argument("--peer-vol-change", action="store_true")
    parser.add_argument("--peer-tvl", action="store_true")
    parser.add_argument("--peer-volatility", action="store_true")
    parser.add_argument("--own-vol-lag2", action="store_true")
    parser.add_argument("--own-vol-change", action="store_true")
    parser.add_argument("--own-tvl", action="store_true")
    parser.add_argument("--own-volatility", action="store_true")
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
    }
    hparams = {
        "hidden": args.hidden,
        "d_embed": args.d_embed,
        "lr": args.lr,
        "l2_alpha": args.l2_alpha,
        "n_epochs": args.epochs,
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

    if args.tune > 0:
        run_optuna(data, args.tune)
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
