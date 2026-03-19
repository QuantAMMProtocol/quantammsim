"""Linear noise model with market features and learnable cadence.

V_total = V_arb(cadence) + exp(x @ coeffs)

where x includes:
  - Option C x_obs (intercept, log_tvl_lag1, dow_sin, dow_cos)
  - Cross-pool lagged volumes (token-A, token-B, chain peers)
  - Market features (BTC price/vol/trend, token prices/vol/trend)

Cadence is per-pool, optimized jointly with noise coefficients via Adam
through the differentiable PCHIP grid.

Usage:
  python experiments/run_linear_market_noise.py
  python experiments/run_linear_market_noise.py --trend-windows 7 14 30
  python experiments/run_linear_market_noise.py --no-market  # x_obs only
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


def build_data(matched_clean, option_c_clean, trend_windows=(7, 14, 30),
               include_market=True, include_cross_pool=True):
    """Build feature matrix and targets."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import (
        build_x_obs, build_cross_pool_x_obs, K_OBS_REDUCED, K_OBS_CROSS,
    )
    from quantammsim.calibration.market_features import (
        build_pool_market_features, pool_market_features_to_matrix,
    )

    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)

    # Common date grid
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)
    n_dates = len(date_list)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    # Per-pool: V_arb, volumes, coeffs, gas, grid day mapping
    vol_matrix = np.full((n_dates, n_pools), np.nan)
    pool_coeffs = []
    pool_gas = []
    init_log_cadences = np.zeros(n_pools, dtype=np.float32)
    common_to_grid = np.full((n_pools, n_dates), 0, dtype=np.int32)

    for j, pid in enumerate(pool_ids):
        entry = matched_clean[pid]
        oc = option_c_clean[pid]
        panel = entry["panel"]

        pool_coeffs.append(entry["coeffs"])
        pool_gas.append(jnp.float64(np.exp(oc["log_gas"])))
        init_log_cadences[j] = oc["log_cadence"]

        dates = panel["date"].values
        log_vols = panel["log_volume"].values.astype(float)
        for k, date in enumerate(dates):
            t = date_to_idx[date]
            vol_matrix[t, j] = log_vols[k]
            common_to_grid[j, t] = entry["day_indices"][k]

    # Build samples: require t >= 1 (for lag)
    sample_pools, sample_days = [], []
    for i in range(n_pools):
        for t in range(1, n_dates):
            if np.isnan(vol_matrix[t, i]) or np.isnan(vol_matrix[t - 1, i]):
                continue
            sample_pools.append(i)
            sample_days.append(t)
    sample_pools = np.array(sample_pools, dtype=np.int32)
    sample_days = np.array(sample_days, dtype=np.int32)
    n_samples = len(sample_pools)

    # x_obs: reduced (4) or cross-pool (7)
    if include_cross_pool:
        k_obs = K_OBS_CROSS
        x_obs_grid = np.full((n_dates, n_pools, k_obs), np.nan)
        for j, pid in enumerate(pool_ids):
            panel = matched_clean[pid]["panel"]
            xc = build_cross_pool_x_obs(panel, matched_clean, pid)  # (n_obs-1, 7)
            dates = panel["date"].values
            for k, date in enumerate(dates[1:]):
                x_obs_grid[date_to_idx[date], j] = xc[k]
    else:
        k_obs = K_OBS_REDUCED
        x_obs_grid = np.full((n_dates, n_pools, k_obs), np.nan)
        for j, pid in enumerate(pool_ids):
            panel = matched_clean[pid]["panel"]
            xr = build_x_obs(panel, reduced=True)
            dates = panel["date"].values
            for k, date in enumerate(dates):
                x_obs_grid[date_to_idx[date], j] = xr[k]

    # Per-sample x_obs
    x_obs = np.zeros((n_samples, k_obs), dtype=np.float32)
    for s in range(n_samples):
        xval = x_obs_grid[sample_days[s], sample_pools[s]]
        if np.all(np.isfinite(xval)):
            x_obs[s] = xval

    # Market features
    if include_market:
        print("  Building market features...")
        pool_feat = build_pool_market_features(
            matched_clean, trend_windows=list(trend_windows))
        x_market, market_names = pool_market_features_to_matrix(
            pool_feat, matched_clean, date_to_idx, pool_ids,
            sample_pools, sample_days)
        print(f"  Market features: {len(market_names)} columns")
    else:
        x_market = np.zeros((n_samples, 0), dtype=np.float32)
        market_names = []

    # Combine features
    x_all = np.concatenate([x_obs, x_market], axis=1).astype(np.float32)
    feat_names = [f"xobs_{i}" for i in range(k_obs)] + market_names

    # Standardize (except intercept column 0)
    x_mean = np.mean(x_all, axis=0)
    x_std = np.std(x_all, axis=0)
    x_std[x_std < 1e-6] = 1.0
    x_mean[0] = 0.0  # don't center intercept
    x_std[0] = 1.0
    x_all = ((x_all - x_mean) / x_std).astype(np.float32)

    # Targets
    y_total = np.array([vol_matrix[sample_days[s], sample_pools[s]]
                        for s in range(n_samples)], dtype=np.float32)
    sample_grid_days = common_to_grid[sample_pools, sample_days]

    return {
        "x": x_all,                           # (n_samples, n_feat)
        "y_total": y_total,                    # (n_samples,)
        "pool_idx": sample_pools,              # (n_samples,)
        "day_idx": sample_days,                # (n_samples,)
        "sample_grid_days": sample_grid_days,  # (n_samples,)
        "pool_coeffs": pool_coeffs,
        "pool_gas": pool_gas,
        "init_log_cadences": init_log_cadences,
        "n_pools": n_pools,
        "n_feat": x_all.shape[1],
        "pool_ids": pool_ids,
        "feat_names": feat_names,
        "x_mean": x_mean,
        "x_std": x_std,
    }


def make_loss_fn(pool_coeffs, pool_gas, n_pools):
    """Loss function with learnable cadence + linear noise model."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    def loss_fn(params, x, y_total, sample_grid_days, pool_idx,
                l2_alpha, huber_delta):
        log_cadence = params["log_cadence"]
        noise_coeffs = params["noise_coeffs"]

        # V_noise = exp(x @ noise_coeffs + pool_intercept)
        log_v_noise = x @ noise_coeffs
        if "pool_intercepts" in params:
            log_v_noise = log_v_noise + params["pool_intercepts"][pool_idx]

        # V_arb from PCHIP at learned cadence
        n_samples = y_total.shape[0]
        v_arb = jnp.zeros(n_samples)
        for i in range(n_pools):
            v_arb_all = interpolate_pool_daily(
                pool_coeffs[i], log_cadence[i], pool_gas[i])
            safe_days = jnp.clip(sample_grid_days, 0, v_arb_all.shape[0] - 1)
            v_arb = jnp.where(pool_idx == i, v_arb_all[safe_days], v_arb)

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

        reg = l2_alpha * jnp.sum(noise_coeffs ** 2)
        return data_loss + reg

    return jax.jit(jax.value_and_grad(loss_fn))


def train(params, data, grad_fn, n_epochs, lr, l2_alpha, huber_delta,
          verbose=True):
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}

    x = jnp.array(data["x"])
    y = jnp.array(data["y_total"])
    sgd = jnp.array(data["sample_grid_days"])
    pidx = jnp.array(data["pool_idx"])

    for epoch in range(n_epochs):
        loss_val, grads = grad_fn(
            params, x, y, sgd, pidx, l2_alpha, huber_delta)
        loss_f = float(loss_val)

        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            m_hat = m[k] / (1.0 - 0.9 ** (epoch + 1))
            v_hat = v[k] / (1.0 - 0.999 ** (epoch + 1))
            params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            cads = np.exp(np.array(params["log_cadence"]))
            nc = np.array(params["noise_coeffs"])
            print(f"  epoch {epoch:4d}  loss={loss_f:.6f}"
                  f"  cad=[{cads.min():.1f}-{np.median(cads):.1f}-{cads.max():.1f}]"
                  f"  |coeffs|={np.mean(np.abs(nc)):.3f}")

    return params


def evaluate(params, data, label=""):
    """Evaluate decomposition quality."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    x = np.array(data["x"])
    y_total = np.array(data["y_total"])
    pool_idx = np.array(data["pool_idx"])
    sgd = np.array(data["sample_grid_days"])
    log_cadence = np.array(params["log_cadence"])
    noise_coeffs = np.array(params["noise_coeffs"])
    init_cads = data["init_log_cadences"]
    pool_ids = data["pool_ids"]
    n_pools = data["n_pools"]

    log_v_noise = x @ noise_coeffs
    if "pool_intercepts" in params:
        pool_intercepts = np.array(params["pool_intercepts"])
        log_v_noise = log_v_noise + pool_intercepts[pool_idx]
    v_noise = np.exp(log_v_noise)

    v_arb = np.zeros(len(y_total))
    for i in range(n_pools):
        mask = pool_idx == i
        if not mask.any():
            continue
        v_arb_all = np.array(interpolate_pool_daily(
            data["pool_coeffs"][i], jnp.float64(log_cadence[i]),
            data["pool_gas"][i]))
        v_arb[mask] = v_arb_all[sgd[mask]]

    v_obs = np.exp(y_total)
    log_v_arb = np.log(np.maximum(v_arb, 1e-10))
    pred_total = np.logaddexp(log_v_arb, log_v_noise)

    if label:
        print(f"\n  {label}:")
    print(f"    {'Pool'[:16]:16s} {'R²':>6s} {'Cad':>5s} {'→':>2s} {'learn':>5s}"
          f" {'Arb%':>6s} {'Noise%':>7s} {'Flag':>5s}")
    print(f"    {'-'*60}")

    r2s = {}
    pool_diag = []
    for i in range(n_pools):
        mask = pool_idx == i
        if mask.sum() < 2:
            continue
        yt = y_total[mask]
        pt = pred_total[mask]
        ss_res = np.sum((yt - pt) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2s[i] = 1 - ss_res / max(ss_tot, 1e-10)

        pid = pool_ids[i]
        ci = np.exp(init_cads[i])
        cl = np.exp(log_cadence[i])
        arb_pct = np.median(v_arb[mask] / v_obs[mask]) * 100
        noise_pct = np.median(v_noise[mask] / v_obs[mask]) * 100

        flags = []
        if arb_pct > 150:
            flags.append("A")
        if cl <= 1.01 or cl >= 59.9:
            flags.append("B")
        if r2s[i] < 0:
            flags.append("X")
        flag_str = "".join(flags)

        pool_diag.append({
            "pid": pid, "r2": r2s[i], "cad_init": ci, "cad_learned": cl,
            "arb_pct": arb_pct, "noise_pct": noise_pct, "flags": flag_str,
        })

        print(f"    {pid[:16]:16s} {r2s[i]:6.3f} {ci:5.1f} → {cl:5.1f}"
              f" {arb_pct:6.0f}% {noise_pct:6.0f}% {flag_str:>5s}")

    vals = [x for x in r2s.values() if np.isfinite(x)]
    med = np.median(vals) if vals else float("nan")
    healthy = [d for d in pool_diag if d["arb_pct"] <= 150 and d["r2"] > 0]
    med_h = np.median([d["r2"] for d in healthy]) if healthy else float("nan")
    n_path = sum(1 for d in pool_diag if d["arb_pct"] > 150)
    n_bound = sum(1 for d in pool_diag
                  if d["cad_learned"] <= 1.01 or d["cad_learned"] >= 59.9)

    print(f"\n    Median R²: {med:.4f} (healthy: {med_h:.4f})")
    print(f"    Healthy: {len(pool_diag) - n_path}/{len(pool_diag)},"
          f"  at bounds: {n_bound}")

    return med, r2s, pool_diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-alpha", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--trend-windows", type=int, nargs="+", default=[7, 14, 30])
    parser.add_argument("--no-market", action="store_true",
                        help="x_obs only, no market features")
    parser.add_argument("--no-cross-pool", action="store_true",
                        help="Reduced x_obs (4) instead of cross-pool (7)")
    parser.add_argument("--pool-intercepts", action="store_true",
                        help="Per-pool intercept (shared slopes + per-pool bias)")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Linear Noise Model + Learnable Cadence")
    print(f"  market={not args.no_market}, cross_pool={not args.no_cross_pool}"
          f", pool_intercepts={args.pool_intercepts}")
    print(f"  trend_windows={args.trend_windows}")
    print(f"  epochs={args.epochs}, lr={args.lr}, l2={args.l2_alpha}")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()

    print("\nBuilding data...")
    t0 = time.time()
    data = build_data(
        matched_clean, option_c_clean,
        trend_windows=tuple(args.trend_windows),
        include_market=not args.no_market,
        include_cross_pool=not args.no_cross_pool,
    )
    print(f"  {len(data['pool_idx'])} samples, {data['n_pools']} pools,"
          f" {data['n_feat']} features, {time.time() - t0:.1f}s")
    print(f"  Features: {data['feat_names']}")

    # Temporal split
    day_idx = data["day_idx"]
    split_day = int(day_idx.max() * 0.7)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day

    train_data = {k: v[train_mask] if isinstance(v, np.ndarray)
                  and v.shape[0] == len(day_idx) else v
                  for k, v in data.items()}
    eval_data = {k: v[eval_mask] if isinstance(v, np.ndarray)
                 and v.shape[0] == len(day_idx) else v
                 for k, v in data.items()}

    # Init params
    n_feat = data["n_feat"]
    n_pools = data["n_pools"]
    params = {
        "log_cadence": jnp.array(data["init_log_cadences"]),
        "noise_coeffs": jnp.zeros(n_feat),
    }

    # Warm-start noise_coeffs via OLS on train: y_total ≈ x @ coeffs
    x_trn = data["x"][train_mask]
    y_trn = data["y_total"][train_mask]
    sol, _, _, _ = np.linalg.lstsq(x_trn, y_trn, rcond=None)
    params["noise_coeffs"] = jnp.array(sol.astype(np.float32))

    if args.pool_intercepts:
        # Init per-pool intercepts from OLS residuals
        ols_pred = x_trn @ sol
        ols_resid = y_trn - ols_pred
        pool_idx_trn = data["pool_idx"][train_mask]
        intercepts = np.zeros(n_pools, dtype=np.float32)
        for i in range(n_pools):
            mask_i = pool_idx_trn == i
            if mask_i.sum() > 0:
                intercepts[i] = np.mean(ols_resid[mask_i])
        params["pool_intercepts"] = jnp.array(intercepts)
        print(f"  Per-pool intercepts: {n_pools} pools"
              f" (range {intercepts.min():.2f} to {intercepts.max():.2f})")

    print(f"\n  Init cadence: {np.exp(data['init_log_cadences']).min():.1f}"
          f"-{np.median(np.exp(data['init_log_cadences'])):.1f}"
          f"-{np.exp(data['init_log_cadences']).max():.1f} min")
    print(f"  OLS warm-start |coeffs|={np.mean(np.abs(sol)):.3f}")
    print(f"  Total params: {sum(v.size for v in params.values())}"
          f" ({n_feat} coeffs + {n_pools} cadences"
          f"{'+ ' + str(n_pools) + ' intercepts' if args.pool_intercepts else ''})")

    # Build loss and train
    grad_fn = make_loss_fn(data["pool_coeffs"], data["pool_gas"], data["n_pools"])

    print("\n  Compiling...")
    t0 = time.time()
    params = train(params, train_data, grad_fn, args.epochs, args.lr,
                   args.l2_alpha, args.huber_delta)
    print(f"  Training: {time.time() - t0:.1f}s")

    # Print learned coefficients
    nc = np.array(params["noise_coeffs"])
    print(f"\n  Noise coefficients ({len(nc)}):")
    for i, name in enumerate(data["feat_names"]):
        print(f"    {name:30s}  {nc[i]:+8.4f}")

    # Evaluate
    print("\n  --- Train ---")
    evaluate(params, train_data)
    print("\n  --- Eval ---")
    evaluate(params, eval_data)

    # Baselines
    print(f"\n  Baselines (eval, total volume R²):")
    print(f"    V_arb only:     median R² = -0.33")
    print(f"    Naive lag:      median R² =  0.01")
    print(f"    DeepSets best:  median R² =  0.43")


if __name__ == "__main__":
    main()
