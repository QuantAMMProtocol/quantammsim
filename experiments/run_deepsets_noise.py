"""DeepSets noise volume prediction with V_arb decomposition.

Predicts V_noise via a shared encoder-decoder over peer pools.
V_arb is precomputed from grids at Option C cadence/gas.
Loss: mean((log(V_arb + V_noise_predicted) - log_volume)^2)

Usage:
  python experiments/run_deepsets_noise.py              # default hparams
  python experiments/run_deepsets_noise.py --tune 50    # Optuna, 50 trials
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

# Default hyperparameters
DEFAULTS = dict(
    hidden=16,
    d_embed=8,
    lr=3e-4,
    l2_alpha=1e-3,
    n_epochs=1000,
    include_own_lag=True,
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


def build_data(matched_clean, option_c_clean, exclude_pool_idx=None):
    """Build training arrays with V_arb decomposition.

    For each (pool i, day t) sample:
      - peer_vols: other pools' log_volume at t-1
      - v_arb: precomputed arb volume for pool i at day t
      - local_features: [log_tvl_lag1, dow_sin, dow_cos]
      - own_lag: pool i's log_volume at t-1
      - y: log_volume at day t
    """
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import (
        build_pool_attributes, _parse_tokens, _canonicalize_token,
    )

    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)

    # ---- Collect all dates, build volume + V_arb matrices ----
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)
    n_dates = len(date_list)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    vol_matrix = np.full((n_dates, n_pools), np.nan)
    v_arb_matrix = np.full((n_dates, n_pools), np.nan)
    tvl_matrix = np.full((n_dates, n_pools), np.nan)
    weekday_matrix = np.full(n_dates, np.nan)

    for j, pid in enumerate(pool_ids):
        entry = matched_clean[pid]
        oc = option_c_clean[pid]
        panel = entry["panel"]

        # V_arb from grid
        v_arb_all = np.array(interpolate_pool_daily(
            entry["coeffs"],
            jnp.float64(oc["log_cadence"]),
            jnp.float64(np.exp(oc["log_gas"])),
        ))
        v_arb_day = v_arb_all[entry["day_indices"]]

        dates = panel["date"].values
        log_vols = panel["log_volume"].values.astype(float)
        tvl_vals = panel["log_tvl_lag1"].values.astype(float)

        for k, date in enumerate(dates):
            t = date_to_idx[date]
            vol_matrix[t, j] = log_vols[k]
            v_arb_matrix[t, j] = v_arb_day[k]
            tvl_matrix[t, j] = tvl_vals[k]

    # Weekdays
    for t, date in enumerate(date_list):
        dt = pd.Timestamp(date)
        weekday_matrix[t] = dt.weekday()

    # ---- Pool attributes ----
    X_attr, attr_names, _ = build_pool_attributes(matched_clean)
    attr_mean = np.mean(X_attr, axis=0)
    attr_std = np.std(X_attr, axis=0)
    attr_std[attr_std < 1e-6] = 1.0
    X_attr_norm = ((X_attr - attr_mean) / attr_std).astype(np.float32)
    k_attr = X_attr_norm.shape[1]

    # ---- Token overlap ----
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

    target_attrs = X_attr_norm

    # ---- Standardize volumes for encoder input ----
    vol_mean = float(np.nanmean(vol_matrix))
    vol_std = float(np.nanstd(vol_matrix))

    # ---- Build samples ----
    sample_pools, sample_days = [], []
    for i in range(n_pools):
        if i == exclude_pool_idx:
            continue
        for t in range(1, n_dates):
            if (np.isnan(vol_matrix[t, i]) or np.isnan(vol_matrix[t - 1, i])
                    or np.isnan(v_arb_matrix[t, i]) or np.isnan(tvl_matrix[t, i])):
                continue
            sample_pools.append(i)
            sample_days.append(t)

    sample_pools = np.array(sample_pools, dtype=np.int32)
    sample_days = np.array(sample_days, dtype=np.int32)
    n_samples = len(sample_pools)

    peer_vols_arr = np.zeros((n_samples, n_peers), dtype=np.float32)
    peer_mask_arr = np.zeros((n_samples, n_peers), dtype=np.float32)
    own_lag_arr = np.zeros(n_samples, dtype=np.float32)
    v_arb_arr = np.zeros(n_samples, dtype=np.float32)
    local_arr = np.zeros((n_samples, 3), dtype=np.float32)  # tvl, dow_sin, dow_cos
    y_arr = np.zeros(n_samples, dtype=np.float32)

    for s in range(n_samples):
        i = sample_pools[s]
        t = sample_days[s]
        cols = peer_col_idx[i]

        pvols_raw = vol_matrix[t - 1, cols]
        valid = ~np.isnan(pvols_raw)
        pvols_norm = (pvols_raw - vol_mean) / vol_std
        peer_vols_arr[s] = np.where(valid, pvols_norm, 0.0)
        peer_mask_arr[s] = valid.astype(np.float32)

        own_lag_arr[s] = (vol_matrix[t - 1, i] - vol_mean) / vol_std
        v_arb_arr[s] = v_arb_matrix[t, i]
        y_arr[s] = vol_matrix[t, i]  # raw log_volume (not standardized)

        wd = weekday_matrix[t]
        local_arr[s, 0] = tvl_matrix[t, i]
        local_arr[s, 1] = np.sin(2 * np.pi * wd / 7)
        local_arr[s, 2] = np.cos(2 * np.pi * wd / 7)

    # Standardize local features
    local_mean = np.mean(local_arr, axis=0)
    local_std = np.std(local_arr, axis=0)
    local_std[local_std < 1e-6] = 1.0
    local_arr = ((local_arr - local_mean) / local_std).astype(np.float32)

    return {
        "peer_attrs": jnp.array(peer_attrs),
        "target_attrs": jnp.array(target_attrs),
        "peer_overlap": jnp.array(peer_overlap),
        "peer_vols": jnp.array(peer_vols_arr),
        "peer_mask": jnp.array(peer_mask_arr),
        "own_lag": jnp.array(own_lag_arr),
        "v_arb": jnp.array(v_arb_arr),
        "local": jnp.array(local_arr),
        "y": jnp.array(y_arr),
        "pool_idx": jnp.array(sample_pools),
        "day_idx": sample_days,
        "n_pools": n_pools,
        "n_peers": n_peers,
        "k_attr": k_attr,
        "k_local": 3,
        "pool_ids": pool_ids,
    }


# ---- Model ----


def init_params(key, k_attr, k_local, hidden, d_embed, include_own_lag):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    enc_in = 2 * k_attr + 2  # peer_attr + target_attr + peer_vol + overlap
    dec_in = d_embed + k_attr + k_local + (1 if include_own_lag else 0)

    return {
        "enc_W1": jax.random.normal(k1, (enc_in, hidden)) * np.sqrt(2.0 / enc_in),
        "enc_b1": jnp.zeros(hidden),
        "enc_W2": jax.random.normal(k2, (hidden, d_embed)) * np.sqrt(2.0 / hidden),
        "enc_b2": jnp.zeros(d_embed),
        "dec_W1": jax.random.normal(k3, (dec_in, hidden)) * np.sqrt(2.0 / dec_in),
        "dec_b1": jnp.zeros(hidden),
        "dec_W2": jax.random.normal(k4, (hidden, 1)) * 0.01,
        "dec_b2": jnp.zeros(1),
    }


def forward(params, peer_attrs_all, target_attrs_all, peer_overlap_all,
            peer_vols, peer_mask, own_lag, local_feat, pool_idx,
            include_own_lag=True):
    """Returns log_v_noise per sample."""
    batch = peer_vols.shape[0]

    pa = peer_attrs_all[pool_idx]
    ta = target_attrs_all[pool_idx]
    ov = peer_overlap_all[pool_idx]
    ta_broad = jnp.broadcast_to(ta[:, None, :], pa.shape)

    enc_in = jnp.concatenate([
        pa, ta_broad,
        peer_vols[:, :, None],
        ov[:, :, None],
    ], axis=-1)

    flat = enc_in.reshape(-1, enc_in.shape[-1])
    h = jnp.maximum(flat @ params["enc_W1"] + params["enc_b1"], 0.0)
    h = h @ params["enc_W2"] + params["enc_b2"]
    h = h.reshape(batch, peer_vols.shape[1], -1)

    h_masked = h * peer_mask[:, :, None]
    n_valid = jnp.maximum(jnp.sum(peer_mask, axis=1, keepdims=True), 1.0)
    summary = jnp.sum(h_masked, axis=1) / n_valid

    dec_parts = [summary, ta, local_feat]
    if include_own_lag:
        dec_parts.append(own_lag[:, None])
    dec_in = jnp.concatenate(dec_parts, axis=-1)

    h_dec = jnp.maximum(dec_in @ params["dec_W1"] + params["dec_b1"], 0.0)
    log_v_noise = (h_dec @ params["dec_W2"] + params["dec_b2"])[:, 0]
    return log_v_noise


def loss_fn(params, static, peer_vols, peer_mask, own_lag, local_feat,
            pool_idx, v_arb, y, l2_alpha, include_own_lag):
    """Log-space V_arb + V_noise loss matching the calibration pipeline."""
    log_v_noise = forward(
        params, static["peer_attrs"], static["target_attrs"],
        static["peer_overlap"], peer_vols, peer_mask, own_lag, local_feat,
        pool_idx, include_own_lag,
    )
    v_noise = jnp.exp(log_v_noise)
    log_v_pred = jnp.log(jnp.maximum(v_arb + v_noise, 1e-6))
    mse = jnp.mean((log_v_pred - y) ** 2)
    reg = sum(jnp.sum(v ** 2) for k, v in params.items() if "W" in k)
    return mse + alpha * reg if (alpha := l2_alpha) else mse + l2_alpha * reg


@jax.jit
def _loss_and_grad(params, static, peer_vols, peer_mask, own_lag, local_feat,
                   pool_idx, v_arb, y, l2_alpha, include_own_lag):
    return jax.value_and_grad(loss_fn)(
        params, static, peer_vols, peer_mask, own_lag, local_feat,
        pool_idx, v_arb, y, l2_alpha, include_own_lag,
    )


# ---- Training ----


def train(params, data, hparams, verbose=True):
    """Full-batch Adam."""
    static = {k: data[k] for k in ["peer_attrs", "target_attrs", "peer_overlap"]}
    include_own_lag = hparams["include_own_lag"]
    lr = hparams["lr"]
    l2_alpha = hparams["l2_alpha"]
    n_epochs = hparams["n_epochs"]

    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}

    for epoch in range(n_epochs):
        loss_val, grads = _loss_and_grad(
            params, static, data["peer_vols"], data["peer_mask"],
            data["own_lag"], data["local"], data["pool_idx"],
            data["v_arb"], data["y"], l2_alpha, include_own_lag,
        )

        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            m_hat = m[k] / (1.0 - 0.9 ** (epoch + 1))
            v_hat = v[k] / (1.0 - 0.999 ** (epoch + 1))
            params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:4d}  loss={float(loss_val):.6f}")

    return params, float(loss_val)


# ---- Evaluation ----


def per_pool_r2(params, data, hparams):
    """Per-pool R² on the log(V_arb + V_noise) prediction."""
    static = {k: data[k] for k in ["peer_attrs", "target_attrs", "peer_overlap"]}
    log_v_noise = np.array(forward(
        params, static["peer_attrs"], static["target_attrs"],
        static["peer_overlap"], data["peer_vols"], data["peer_mask"],
        data["own_lag"], data["local"], data["pool_idx"],
        hparams["include_own_lag"],
    ))
    v_noise = np.exp(log_v_noise)
    v_arb = np.array(data["v_arb"])
    log_v_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
    y = np.array(data["y"])
    pool_idx = np.array(data["pool_idx"])

    r2s = {}
    for i in range(data["n_pools"]):
        mask = pool_idx == i
        if mask.sum() < 2:
            continue
        yi = y[mask]
        pi = log_v_pred[mask]
        ss_res = np.sum((yi - pi) ** 2)
        ss_tot = np.sum((yi - yi.mean()) ** 2)
        r2s[i] = 1 - ss_res / max(ss_tot, 1e-10)
    return r2s


def subset_data(data, mask):
    """Subset data arrays by boolean mask."""
    jmask = jnp.array(mask)
    static_keys = ["peer_attrs", "target_attrs", "peer_overlap",
                    "n_pools", "n_peers", "k_attr", "k_local", "pool_ids"]
    out = {k: data[k] for k in static_keys}
    for k in ["peer_vols", "peer_mask", "own_lag", "v_arb", "local", "y", "pool_idx"]:
        out[k] = data[k][jmask]
    out["day_idx"] = np.array(data["day_idx"])[mask]
    return out


# ---- Experiments ----


def run_single(matched_clean, option_c_clean, hparams, split_frac=0.7):
    """Train with temporal split, report in-sample and eval R²."""
    data = build_data(matched_clean, option_c_clean)
    n_params = sum(v.size for v in init_params(
        jax.random.PRNGKey(0), data["k_attr"], data["k_local"],
        hparams["hidden"], hparams["d_embed"], hparams["include_own_lag"],
    ).values())

    print(f"  {data['peer_vols'].shape[0]} samples, {data['n_pools']} pools, "
          f"{n_params} params")

    day_idx = np.array(data["day_idx"])
    split_day = int(day_idx.max() * split_frac)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day
    train_data = subset_data(data, train_mask)
    eval_data = subset_data(data, eval_mask)

    print(f"  Train: {int(train_mask.sum())} samples, "
          f"Eval: {int(eval_mask.sum())} samples")

    params = init_params(
        jax.random.PRNGKey(42), data["k_attr"], data["k_local"],
        hparams["hidden"], hparams["d_embed"], hparams["include_own_lag"],
    )
    t0 = time.time()
    params, final_loss = train(params, train_data, hparams)
    print(f"  Training: {time.time() - t0:.1f}s, final loss={final_loss:.6f}")

    r2_train = per_pool_r2(params, train_data, hparams)
    r2_eval = per_pool_r2(params, eval_data, hparams)

    pool_ids = data["pool_ids"]
    for i, pid in enumerate(pool_ids):
        r_tr = r2_train.get(i, float("nan"))
        r_ev = r2_eval.get(i, float("nan"))
        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"train={r_tr:.3f}  eval={r_ev:.3f}")

    vals_train = [v for v in r2_train.values() if np.isfinite(v)]
    vals_eval = [v for v in r2_eval.values() if np.isfinite(v)]
    med_train = np.median(vals_train) if vals_train else float("nan")
    med_eval = np.median(vals_eval) if vals_eval else float("nan")

    print(f"\n  Train: median R²={med_train:.4f}")
    print(f"  Eval:  median R²={med_eval:.4f}")
    print(f"  (Option C in-sample: 0.589)")

    return med_eval, params, data


def run_loo(matched_clean, option_c_clean, hparams):
    """Full LOO."""
    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)
    loo_r2s = []

    print(f"\n{'='*70}")
    print("LOO DeepSets Noise")
    print(f"{'='*70}")

    # Use fewer epochs for LOO
    loo_hparams = dict(hparams, n_epochs=min(hparams["n_epochs"], 500))

    for hold_out_idx in range(n_pools):
        hold_out_pid = pool_ids[hold_out_idx]
        train_data = build_data(matched_clean, option_c_clean,
                                exclude_pool_idx=hold_out_idx)

        params = init_params(
            jax.random.PRNGKey(42), train_data["k_attr"], train_data["k_local"],
            loo_hparams["hidden"], loo_hparams["d_embed"],
            loo_hparams["include_own_lag"],
        )
        params, _ = train(params, train_data, loo_hparams, verbose=False)

        # Eval on held-out pool
        full_data = build_data(matched_clean, option_c_clean)
        ho_mask = np.array(full_data["pool_idx"]) == hold_out_idx
        if ho_mask.sum() < 2:
            loo_r2s.append(float("nan"))
            continue

        eval_data = subset_data(full_data, ho_mask)
        r2s = per_pool_r2(params, eval_data, loo_hparams)
        r2 = r2s.get(hold_out_idx, float("nan"))
        loo_r2s.append(r2)

        tag = "OK" if r2 > 0 else "NEG"
        print(f"  {hold_out_pid[:16]} ({matched_clean[hold_out_pid]['tokens']:<14}) "
              f"R²={r2:.3f} [{tag}]")

    valid = [r for r in loo_r2s if np.isfinite(r)]
    med = np.median(valid) if valid else float("nan")
    print(f"\n  LOO: median R²={med:.4f}, "
          f"mean={np.mean(valid):.4f}, "
          f"n_neg={sum(1 for r in valid if r < 0)}")
    return loo_r2s


# ---- Optuna ----


def run_optuna(matched_clean, option_c_clean, n_trials):
    """Hyperparameter optimization with Optuna."""
    import optuna

    # Precompute data once (shared across trials)
    data = build_data(matched_clean, option_c_clean)
    day_idx = np.array(data["day_idx"])
    split_day = int(day_idx.max() * 0.7)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day
    train_data = subset_data(data, train_mask)
    eval_data = subset_data(data, eval_mask)

    def objective(trial):
        hp = {
            "hidden": trial.suggest_categorical("hidden", [8, 16, 32]),
            "d_embed": trial.suggest_categorical("d_embed", [4, 8, 16]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "l2_alpha": trial.suggest_float("l2_alpha", 1e-5, 1e-1, log=True),
            "n_epochs": trial.suggest_categorical("n_epochs", [500, 1000, 2000]),
            "include_own_lag": trial.suggest_categorical("include_own_lag", [True, False]),
        }

        params = init_params(
            jax.random.PRNGKey(42), data["k_attr"], data["k_local"],
            hp["hidden"], hp["d_embed"], hp["include_own_lag"],
        )
        params, final_loss = train(params, train_data, hp, verbose=False)

        r2s = per_pool_r2(params, eval_data, hp)
        vals = [v for v in r2s.values() if np.isfinite(v)]
        med_r2 = float(np.median(vals)) if vals else -10.0

        # Report train R² too for diagnostics
        r2s_tr = per_pool_r2(params, train_data, hp)
        vals_tr = [v for v in r2s_tr.values() if np.isfinite(v)]
        med_tr = float(np.median(vals_tr)) if vals_tr else -10.0

        trial.set_user_attr("train_median_r2", med_tr)
        trial.set_user_attr("final_loss", final_loss)

        print(f"  Trial {trial.number}: eval={med_r2:.4f} train={med_tr:.4f} "
              f"h={hp['hidden']} d={hp['d_embed']} lr={hp['lr']:.1e} "
              f"alpha={hp['l2_alpha']:.1e} epochs={hp['n_epochs']} "
              f"own_lag={hp['include_own_lag']}")

        return med_r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{'='*70}")
    print("Optuna Results")
    print(f"{'='*70}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best eval median R²: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    print(f"  Train median R²: {study.best_trial.user_attrs['train_median_r2']:.4f}")

    # Show top 5
    print(f"\n  Top 5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999,
                    reverse=True)
    for t in trials[:5]:
        if t.value is not None:
            print(f"    #{t.number}: eval={t.value:.4f} "
                  f"train={t.user_attrs.get('train_median_r2', '?'):.4f} "
                  f"{t.params}")

    return study


# ---- Main ----


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", type=int, default=0,
                        help="Run Optuna with N trials")
    parser.add_argument("--loo", action="store_true",
                        help="Run LOO evaluation")
    parser.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    parser.add_argument("--d-embed", type=int, default=DEFAULTS["d_embed"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--l2-alpha", type=float, default=DEFAULTS["l2_alpha"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["n_epochs"])
    parser.add_argument("--no-own-lag", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    hparams = {
        "hidden": args.hidden,
        "d_embed": args.d_embed,
        "lr": args.lr,
        "l2_alpha": args.l2_alpha,
        "n_epochs": args.epochs,
        "include_own_lag": not args.no_own_lag,
    }

    print("=" * 70)
    print("DeepSets Noise Volume Prediction (V_arb decomposition)")
    print(f"  {hparams}")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()

    if args.tune > 0:
        run_optuna(matched_clean, option_c_clean, args.tune)
    else:
        print(f"\n{'='*70}")
        print("Temporal split (70/30)")
        print(f"{'='*70}")
        med_eval, params, data = run_single(matched_clean, option_c_clean, hparams)

        if args.loo:
            run_loo(matched_clean, option_c_clean, hparams)


if __name__ == "__main__":
    main()
