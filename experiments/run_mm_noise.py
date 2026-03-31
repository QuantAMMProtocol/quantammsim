"""Michaelis-Menten noise model with market features.

Replaces the linear TVL term with a Michaelis-Menten saturation curve
while keeping all market features for temporal fit:

    log(V_noise) = log_alpha_i + x_market @ gamma
                   + log(TVL) - log(K_i + TVL)

    V_total = V_arb(cadence_i) + exp(log_V_noise)
    Loss = Huber(log(V_total) - log(V_obs))

The TVL feature (xobs_1) is removed from x_market and handled
structurally via the MM saturation term. All other features (dow,
BTC, token, pair vol, interactions) remain as shared linear covariates.

Parameters:
    log_alpha_i  : per-pool intercept
    log_K_i      : per-pool half-saturation TVL
    gamma        : shared coefficients on non-TVL features
    log_cadence_i: per-pool arb frequency (via PCHIP)

Usage:
    python experiments/run_mm_noise.py
    python experiments/run_mm_noise.py --epochs 5000 --lr 3e-4
    python experiments/run_mm_noise.py --per-pool-gamma  # per-pool market coeffs
"""

import argparse
import json
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
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["matched_clean"], data["option_c_clean"]


def build_mm_data(matched_clean, option_c_clean, trend_windows=(7,),
                  include_cross_pool=False):
    """Build data with MM structure: separate TVL from market features."""
    from experiments.run_linear_market_noise import build_data

    # Get full feature matrix from linear model's pipeline
    data = build_data(
        matched_clean, option_c_clean,
        trend_windows=trend_windows,
        include_market=True,
        include_cross_pool=include_cross_pool,
    )

    # Separate TVL from other features
    feat_names = data["feat_names"]
    x_full = data["x"]

    # Find TVL column (xobs_1) and TVL interaction columns
    tvl_col = feat_names.index("xobs_1")
    tvl_interaction_cols = [i for i, name in enumerate(feat_names)
                           if name.startswith("xobs_1\u00d7")]

    # Remove TVL and its interactions from market features
    remove_cols = {tvl_col} | set(tvl_interaction_cols)
    keep_cols = [i for i in range(len(feat_names)) if i not in remove_cols]
    x_market = x_full[:, keep_cols].astype(np.float32)
    market_names = [feat_names[i] for i in keep_cols]

    # TVL comes from the raw panel data (unstandardized log_tvl)
    # x_full[:, tvl_col] might be standardized, so get raw from panel
    pool_ids = data["pool_ids"]
    n_pools = data["n_pools"]

    # Rebuild raw log_tvl from panel
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(date_list)}
    n_dates = len(date_list)

    tvl_grid = np.full((n_dates, n_pools), np.nan)
    for j, pid in enumerate(pool_ids):
        panel = matched_clean[pid]["panel"]
        dates = panel["date"].values
        log_tvls = panel["log_tvl_lag1"].values.astype(float)
        for k, date in enumerate(dates):
            tvl_grid[date_to_idx[date], j] = log_tvls[k]

    pool_idx = data["pool_idx"]
    day_idx = data["day_idx"]
    log_tvl = np.array([tvl_grid[day_idx[s], pool_idx[s]]
                        for s in range(len(pool_idx))], dtype=np.float32)

    # Token info for display
    from quantammsim.calibration.pool_data import _parse_tokens
    pool_tokens = []
    for pid in pool_ids:
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        tok_a = toks[0]
        tok_b = toks[1] if len(toks) > 1 else toks[0]
        pool_tokens.append((tok_a, tok_b))

    removed_names = [feat_names[i] for i in sorted(remove_cols)]
    print(f"  Removed TVL features: {removed_names}")
    print(f"  Market features ({len(market_names)}): {market_names}")

    # Build per-pool temporal ordering for EWMA
    # For each pool, store the sample indices sorted by day_idx
    # Pad to max length so we can use lax.scan uniformly
    pool_time_indices = []  # (n_pools, max_T) — sample indices in time order
    pool_time_lengths = []  # (n_pools,) — actual length per pool
    for i in range(n_pools):
        mask = pool_idx == i
        idxs = np.where(mask)[0]
        # Sort by day_idx
        order = np.argsort(day_idx[idxs])
        pool_time_indices.append(idxs[order])
        pool_time_lengths.append(len(idxs))

    max_T = max(pool_time_lengths) if pool_time_lengths else 0
    # Pad to uniform length (pad with 0, masked later)
    pool_time_padded = np.zeros((n_pools, max_T), dtype=np.int32)
    pool_time_mask = np.zeros((n_pools, max_T), dtype=np.float32)
    for i in range(n_pools):
        L = pool_time_lengths[i]
        pool_time_padded[i, :L] = pool_time_indices[i]
        pool_time_mask[i, :L] = 1.0

    print(f"  EWMA: max_T={max_T}, pools with data: "
          f"{sum(1 for l in pool_time_lengths if l > 0)}")

    return {
        "x_market": x_market,
        "log_tvl": log_tvl,
        "y_total": data["y_total"],
        "pool_idx": pool_idx,
        "day_idx": day_idx,
        "sample_grid_days": data["sample_grid_days"],
        "pool_coeffs": data["pool_coeffs"],
        "pool_gas": data["pool_gas"],
        "init_log_cadences": data["init_log_cadences"],
        "n_pools": n_pools,
        "n_market_feat": x_market.shape[1],
        "pool_ids": pool_ids,
        "pool_tokens": pool_tokens,
        "market_names": market_names,
        "x_mean": data["x_mean"],
        "x_std": data["x_std"],
        "pool_time_padded": pool_time_padded,
        "pool_time_mask": pool_time_mask,
    }


# ---- Model ----

def ewma_smooth(log_tvl, raw_lambda, pool_time_padded, pool_time_mask):
    """Apply learned EWMA smoothing to log_tvl, per pool.

    smooth_t = λ * log_tvl_t + (1-λ) * smooth_{t-1}

    Returns smoothed log_tvl in the same sample order as input.
    """
    lam = jax.nn.sigmoid(raw_lambda)  # constrain to (0, 1)
    n_pools = pool_time_padded.shape[0]
    smoothed = jnp.array(log_tvl)  # copy

    for i in range(n_pools):
        idxs = pool_time_padded[i]    # (max_T,) sample indices
        mask = pool_time_mask[i]       # (max_T,) 1.0 or 0.0
        raw_vals = log_tvl[idxs]       # (max_T,) raw log_tvl in time order

        # lax.scan for EWMA
        def step(carry, x):
            prev_smooth, = carry
            raw_val, m = x
            new_smooth = jnp.where(
                m > 0,
                lam * raw_val + (1.0 - lam) * prev_smooth,
                prev_smooth)
            return (new_smooth,), new_smooth

        init = (raw_vals[0],)
        _, smooth_vals = jax.lax.scan(step, init, (raw_vals, mask))

        # Scatter smoothed values back to sample positions
        smoothed = smoothed.at[idxs].set(
            jnp.where(mask > 0, smooth_vals, smoothed[idxs]))

    return smoothed


def forward_mm(params, x_market, log_tvl_smooth, pool_idx):
    """MM forward pass → log(V_noise) per sample.

    log(V_noise) = log_alpha_i + x_market @ gamma[_i]
                   + log(TVL_smooth) - log(K_i + TVL_smooth)
    """
    log_alpha = params["log_alpha"]
    log_K = params["log_K"]
    gamma = params["gamma"]

    # Per-sample pool params
    alpha_i = log_alpha[pool_idx]
    K_i = jnp.exp(log_K[pool_idx])
    tvl = jnp.exp(log_tvl_smooth)

    # Market features: shared or per-pool gamma
    if gamma.ndim == 2:
        per_sample_gamma = gamma[pool_idx]
        market_term = jnp.sum(x_market * per_sample_gamma, axis=1)
    else:
        market_term = x_market @ gamma

    # MM saturation on smoothed TVL
    log_saturation = log_tvl_smooth - jnp.log(K_i + tvl)

    return alpha_i + market_term + log_saturation


def make_loss_fn(pool_coeffs, pool_gas, n_pools):
    """Loss with PCHIP arb + MM noise."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    def loss_fn(params, x_market, log_tvl, y_total,
                sample_grid_days, pool_idx, pool_time_padded,
                pool_time_mask, l2_alpha, huber_delta):
        log_cadence = params["log_cadence"]

        # EWMA smooth TVL
        log_tvl_smooth = ewma_smooth(
            log_tvl, params["raw_lambda"],
            pool_time_padded, pool_time_mask)

        # V_arb from PCHIP
        n_samples = x_market.shape[0]
        log_v_arb = jnp.zeros(n_samples)
        for i in range(n_pools):
            v_arb_all = interpolate_pool_daily(
                pool_coeffs[i], jnp.float64(log_cadence[i]), pool_gas[i])
            safe_days = jnp.clip(sample_grid_days, 0, v_arb_all.shape[0] - 1)
            log_v_arb = jnp.where(
                pool_idx == i,
                jnp.log(jnp.maximum(v_arb_all[safe_days], 1e-10)),
                log_v_arb)

        # V_noise from MM with smoothed TVL
        log_v_noise = forward_mm(params, x_market, log_tvl_smooth, pool_idx)

        # V_total
        log_v_total = jnp.logaddexp(log_v_arb, log_v_noise)

        # Huber
        residual = log_v_total - y_total
        abs_r = jnp.abs(residual)
        huber = jnp.where(
            abs_r <= huber_delta,
            0.5 * residual ** 2,
            huber_delta * (abs_r - 0.5 * huber_delta))

        # Per-pool equal weighting
        pool_counts = jnp.zeros(n_pools).at[pool_idx].add(
            jnp.ones_like(pool_idx, dtype=jnp.float32))
        active = (pool_counts > 0).astype(jnp.float32)
        n_active = jnp.maximum(jnp.sum(active), 1.0)
        pool_counts = jnp.maximum(pool_counts, 1.0)
        pool_sums = jnp.zeros(n_pools).at[pool_idx].add(huber)
        mean_loss = jnp.sum((pool_sums / pool_counts) * active) / n_active

        # L2 on gamma and log_alpha
        reg = l2_alpha * (
            jnp.mean(params["gamma"] ** 2)
            + jnp.mean(params["log_alpha"] ** 2)
        )

        return mean_loss + reg

    return jax.jit(jax.value_and_grad(loss_fn))


# ---- Training ----

def train(params, data, grad_fn, n_epochs, lr, l2_alpha, huber_delta,
          verbose=True):
    """Adam training loop."""
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    x_market = jnp.array(data["x_market"])
    log_tvl = jnp.array(data["log_tvl"])
    y_total = jnp.array(data["y_total"])
    sgd = jnp.array(data["sample_grid_days"])
    pidx = jnp.array(data["pool_idx"])
    pt_padded = jnp.array(data["pool_time_padded"])
    pt_mask = jnp.array(data["pool_time_mask"])

    for epoch in range(n_epochs):
        loss, grads = grad_fn(
            params, x_market, log_tvl, y_total, sgd, pidx,
            pt_padded, pt_mask, l2_alpha, huber_delta)

        for k in params:
            g = grads[k]
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g ** 2
            m_hat = m[k] / (1 - b1 ** (epoch + 1))
            v_hat = v[k] / (1 - b2 ** (epoch + 1))
            params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            log_K_med = float(jnp.median(params["log_K"]))
            K_med = float(jnp.exp(log_K_med))
            cad = np.exp(np.array(params["log_cadence"]))
            gamma_norm = float(jnp.sqrt(jnp.mean(params["gamma"] ** 2)))
            lam = float(jax.nn.sigmoid(params["raw_lambda"]))
            print(f"  epoch {epoch:5d}  loss={float(loss):.4f}"
                  f"  K_med=${K_med:,.0f}"
                  f"  λ={lam:.3f}"
                  f"  |γ|={gamma_norm:.3f}"
                  f"  cad=[{cad.min():.0f},{np.median(cad):.0f},{cad.max():.0f}]")

    return params


# ---- Evaluation ----

def evaluate(params, data):
    """Per-pool R² and diagnostics."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    n_pools = data["n_pools"]
    pool_idx = np.array(data["pool_idx"])
    sgd = np.array(data["sample_grid_days"])
    y = np.array(data["y_total"])
    log_cadence = np.array(params["log_cadence"])

    # V_arb
    v_arb = np.zeros(len(y))
    for i in range(n_pools):
        mask = pool_idx == i
        if not mask.any():
            continue
        v_arb_all = np.array(interpolate_pool_daily(
            data["pool_coeffs"][i], jnp.float64(log_cadence[i]),
            data["pool_gas"][i]))
        safe = np.clip(sgd[mask], 0, len(v_arb_all) - 1)
        v_arb[mask] = v_arb_all[safe]
    log_v_arb = np.log(np.maximum(v_arb, 1e-10))

    # Smooth TVL with learned lambda
    log_tvl_smooth = np.array(ewma_smooth(
        jnp.array(data["log_tvl"]), params["raw_lambda"],
        jnp.array(data["pool_time_padded"]),
        jnp.array(data["pool_time_mask"])))

    log_v_noise = np.array(forward_mm(
        params, jnp.array(data["x_market"]),
        jnp.array(log_tvl_smooth),
        jnp.array(data["pool_idx"])))

    log_v_total = np.logaddexp(log_v_arb, log_v_noise)
    v_noise = np.exp(log_v_noise)
    v_total = np.exp(log_v_total)

    r2s = {}
    noise_shares = {}
    for i in range(n_pools):
        mask = pool_idx == i
        if mask.sum() < 2:
            continue
        yt = y[mask]
        pt = log_v_total[mask]
        ss_res = np.sum((yt - pt) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2s[data["pool_ids"][i]] = 1 - ss_res / max(ss_tot, 1e-10)
        noise_shares[data["pool_ids"][i]] = float(np.median(
            v_noise[mask] / v_total[mask]))

    K_values = {data["pool_ids"][i]: float(np.exp(params["log_K"][i]))
                for i in range(n_pools)}

    return {
        "r2s": r2s,
        "noise_shares": noise_shares,
        "K_values": K_values,
        "median_r2": float(np.median(list(r2s.values()))),
    }


def tvl_response_check(params, data):
    """Print predicted noise at various TVL levels."""
    n_pools = data["n_pools"]
    pool_idx = np.array(data["pool_idx"])

    # Median market features per pool
    print(f"\n  TVL Response Check (per-pool median market features):")
    print(f"  {'Pool':>20s}  {'K ($M)':>10s}  {'TVL=100K':>10s}"
          f"  {'TVL=1M':>10s}  {'TVL=10M':>10s}  {'TVL=100M':>10s}"
          f"  {'TVL=1B':>10s}  {'ε@1M':>6s}  {'ε@100M':>6s}")

    tvl_test = [1e5, 1e6, 1e7, 1e8, 1e9]

    for i in range(min(n_pools, 15)):
        pid = data["pool_ids"][i]
        toks = data["pool_tokens"][i]
        label = f"{toks[0]}/{toks[1]}"
        mask = pool_idx == i
        if mask.sum() == 0:
            continue

        K_i = float(np.exp(params["log_K"][i]))
        x_med = np.median(data["x_market"][mask], axis=0)

        gamma = np.array(params["gamma"])
        if gamma.ndim == 2:
            market_term = float(x_med @ gamma[i])
        else:
            market_term = float(x_med @ gamma)
        log_alpha_i = float(params["log_alpha"][i])

        vols = []
        for tvl in tvl_test:
            log_sat = np.log(tvl) - np.log(K_i + tvl)
            log_v = log_alpha_i + market_term + log_sat
            vols.append(np.exp(log_v))

        # Elasticity at 1M and 100M
        eps_1m = K_i / (K_i + 1e6)
        eps_100m = K_i / (K_i + 1e8)

        print(f"  {label:>20s}  ${K_i/1e6:>9.1f}"
              + "".join(f"  ${v:>9,.0f}" for v in vols)
              + f"  {eps_1m:>6.3f}  {eps_100m:>6.3f}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2-alpha", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--init-log-K", type=float, default=17.0,
                        help="Initial log(K) ~ log($24M)")
    parser.add_argument("--per-pool-gamma", action="store_true",
                        help="Per-pool market feature coefficients")
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--trend-windows", type=int, nargs="+", default=[7])
    parser.add_argument("--include-cross-pool", action="store_true")
    parser.add_argument("--save-artifact", default="results/mm_noise")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Michaelis-Menten Noise Model + Market Features")
    print(f"  epochs={args.epochs}, lr={args.lr}, l2={args.l2_alpha}")
    print(f"  init log(K)={args.init_log_K} (K=${np.exp(args.init_log_K):,.0f})")
    print(f"  per_pool_gamma={args.per_pool_gamma}")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()

    print("\nBuilding data...")
    t0 = time.time()
    data = build_mm_data(matched_clean, option_c_clean,
                         trend_windows=tuple(args.trend_windows),
                         include_cross_pool=args.include_cross_pool)
    n_pools = data["n_pools"]
    n_market = data["n_market_feat"]
    n_samples = len(data["pool_idx"])
    print(f"  {n_samples} samples, {n_pools} pools,"
          f" {n_market} market features, {time.time() - t0:.1f}s")

    # Pool summary
    pool_idx = data["pool_idx"]
    for i, (pid, toks) in enumerate(
            zip(data["pool_ids"], data["pool_tokens"])):
        mask = pool_idx == i
        n = mask.sum()
        if n > 0:
            med_tvl = np.exp(np.median(data["log_tvl"][mask]))
            print(f"  {pid[:16]}  {toks[0]:>8s}/{toks[1]:<8s}"
                  f"  {n:>4d} days  TVL=${med_tvl:>12,.0f}")

    # Split
    if args.no_split:
        train_data = data
        eval_data = None
    else:
        day_idx = data["day_idx"]
        split_day = int(day_idx.max() * 0.7)
        train_mask = day_idx <= split_day
        eval_mask = day_idx > split_day
        train_data = {k: v[train_mask] if isinstance(v, np.ndarray)
                      and v.shape[0] == n_samples else v
                      for k, v in data.items()}
        eval_data = {k: v[eval_mask] if isinstance(v, np.ndarray)
                     and v.shape[0] == n_samples else v
                     for k, v in data.items()}
        print(f"\n  Split: {train_mask.sum()} train, {eval_mask.sum()} eval")

    # Init
    if args.per_pool_gamma:
        gamma_init = jnp.zeros((n_pools, n_market))
    else:
        gamma_init = jnp.zeros(n_market)

    params = {
        "log_alpha": jnp.zeros(n_pools),
        "log_K": jnp.full(n_pools, args.init_log_K),
        "gamma": gamma_init,
        "log_cadence": jnp.array(data["init_log_cadences"]),
        "raw_lambda": jnp.array(2.0),  # sigmoid(2) ≈ 0.88 — mostly raw
    }
    n_params = sum(v.size for v in params.values())
    print(f"\n  Parameters: {n_params}"
          f" (α: {n_pools}, K: {n_pools},"
          f" γ: {gamma_init.size}, cadence: {n_pools})")

    # Warm-start gamma via Ridge (numpy, no sklearn)
    print("  Warm-starting γ via Ridge on residuals...")

    def _ridge(X, y, alpha=1.0):
        """Ridge regression: (X'X + αI)^-1 X'y."""
        XtX = X.T @ X + alpha * np.eye(X.shape[1])
        Xty = X.T @ y
        return np.linalg.solve(XtX, Xty)

    x_trn = data["x_market"] if args.no_split else train_data["x_market"]
    y_trn = data["y_total"] if args.no_split else train_data["y_total"]
    if args.per_pool_gamma:
        pidx = data["pool_idx"] if args.no_split else train_data["pool_idx"]
        for i in range(n_pools):
            mask = pidx == i
            if mask.sum() < 5:
                continue
            # Add intercept column for warm-start
            X_i = np.concatenate([x_trn[mask], np.ones((mask.sum(), 1))], 1)
            w = _ridge(X_i, y_trn[mask])
            params["gamma"] = params["gamma"].at[i].set(
                jnp.array(w[:-1].astype(np.float32)))
            params["log_alpha"] = params["log_alpha"].at[i].set(float(w[-1]))
    else:
        X_all = np.concatenate([x_trn, np.ones((len(y_trn), 1))], 1)
        w = _ridge(X_all, y_trn)
        params["gamma"] = jnp.array(w[:-1].astype(np.float32))

    # Loss
    grad_fn = make_loss_fn(data["pool_coeffs"], data["pool_gas"], n_pools)

    print(f"\nTraining ({args.epochs} epochs)...")
    t0 = time.time()
    params = train(params, train_data, grad_fn, args.epochs, args.lr,
                   args.l2_alpha, args.huber_delta)
    print(f"  Training time: {time.time() - t0:.1f}s")

    # Evaluate
    print("\n" + "=" * 70)
    print("Results (train)")
    print("=" * 70)
    train_eval = evaluate(params, train_data)
    print(f"  Median R²: {train_eval['median_r2']:.4f}")

    print(f"\n  {'Pool':>16s}  {'Tokens':>16s}  {'R²':>6s}"
          f"  {'Noise%':>7s}  {'K ($M)':>10s}")
    for pid in data["pool_ids"]:
        i = data["pool_ids"].index(pid)
        toks = data["pool_tokens"][i]
        r2 = train_eval["r2s"].get(pid, float("nan"))
        ns = train_eval["noise_shares"].get(pid, float("nan"))
        K = train_eval["K_values"][pid]
        print(f"  {pid[:16]}  {toks[0]:>8s}/{toks[1]:<6s}"
              f"  {r2:>6.3f}  {ns*100:>6.1f}%  ${K/1e6:>9.1f}")

    if eval_data is not None:
        print("\n" + "=" * 70)
        print("Results (eval)")
        print("=" * 70)
        eval_result = evaluate(params, eval_data)
        print(f"  Median R²: {eval_result['median_r2']:.4f}")

    # TVL response
    tvl_response_check(params, data)

    # Gamma coefficients
    gamma = np.array(params["gamma"])
    if gamma.ndim == 1:
        print(f"\n  Shared γ coefficients:")
        for j, name in enumerate(data["market_names"]):
            print(f"    {name:>30s}: {gamma[j]:>8.4f}")

    # Save
    if args.save_artifact:
        os.makedirs(args.save_artifact, exist_ok=True)
        save_dict = {k: np.array(v) for k, v in params.items()}
        np.savez(os.path.join(args.save_artifact, "model.npz"), **save_dict)
        meta = {
            "model": "michaelis_menten",
            "pool_ids": data["pool_ids"],
            "pool_tokens": data["pool_tokens"],
            "market_names": data["market_names"],
            "n_pools": n_pools,
            "n_market_feat": n_market,
            "per_pool_gamma": args.per_pool_gamma,
            "hparams": {
                "epochs": args.epochs, "lr": args.lr,
                "l2_alpha": args.l2_alpha, "huber_delta": args.huber_delta,
                "init_log_K": args.init_log_K,
            },
        }
        with open(os.path.join(args.save_artifact, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Saved: {args.save_artifact}/")


if __name__ == "__main__":
    main()
