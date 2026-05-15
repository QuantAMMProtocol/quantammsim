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
    python scripts/run_mm_noise.py
    python scripts/run_mm_noise.py --epochs 5000 --lr 3e-4
    python scripts/run_mm_noise.py --per-pool-gamma  # per-pool market coeffs
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


COMPETITOR_TVL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "competitor_tvl", "competitor_tvl.npz",
)


def build_mm_data(matched_clean, option_c_clean, trend_windows=(7,),
                  include_cross_pool=False, competitor_tvl_path=None):
    """Build data with MM structure: separate TVL from market features.

    Loads observed competitor TVL from DeFi Llama for K.
    """
    from experiments.run_linear_market_noise import build_data
    from quantammsim.calibration.pool_data import _parse_tokens

    # Get full feature matrix from linear model's pipeline
    data = build_data(
        matched_clean, option_c_clean,
        trend_windows=trend_windows,
        include_market=True,
        include_cross_pool=include_cross_pool,
    )

    # Remove TVL column and TVL interaction terms — TVL handled by MM
    feat_names = data["feat_names"]
    x_full = data["x"]
    tvl_col = feat_names.index("xobs_1")
    tvl_interaction_cols = [i for i, name in enumerate(feat_names)
                           if name.startswith("xobs_1\u00d7")]
    remove_cols = {tvl_col} | set(tvl_interaction_cols)
    keep_cols = [i for i in range(len(feat_names)) if i not in remove_cols]
    x_market = x_full[:, keep_cols].astype(np.float32)
    market_names = [feat_names[i] for i in keep_cols]

    pool_ids = data["pool_ids"]
    n_pools = data["n_pools"]

    # Common date grid
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(date_list)}
    n_dates = len(date_list)

    # Rebuild raw log_tvl from panel
    tvl_grid = np.full((n_dates, n_pools), np.nan)
    for j, pid in enumerate(pool_ids):
        panel = matched_clean[pid]["panel"]
        dates = panel["date"].values
        log_tvls = panel["log_tvl_lag1"].values.astype(float)
        for k, date in enumerate(dates):
            tvl_grid[date_to_idx[date], j] = log_tvls[k]

    pool_idx = data["pool_idx"]
    day_idx = data["day_idx"]
    n_samples = len(pool_idx)
    log_tvl = np.array([tvl_grid[day_idx[s], pool_idx[s]]
                        for s in range(n_samples)], dtype=np.float32)

    # Load observed competitor TVL (K)
    comp_path = competitor_tvl_path or COMPETITOR_TVL_PATH
    if os.path.exists(comp_path):
        comp_data = np.load(comp_path, allow_pickle=True)
        comp_pool_ids = list(comp_data["pool_ids"])
        comp_dates = list(comp_data["date_list"])
        # Use K_eff (network conductance) if available, else direct competitor TVL
        if "k_eff" in comp_data:
            comp_tvl_matrix = comp_data["k_eff"]
            print(f"  Using network K_eff (direct + multi-hop)")
        else:
            comp_tvl_matrix = comp_data["competitor_tvl"]
            print(f"  Using direct competitor TVL only")

        # Build date index for competitor data (normalize to YYYY-MM-DD)
        comp_date_to_idx = {}
        for ci, d in enumerate(comp_dates):
            comp_date_to_idx[str(d)[:10]] = ci

        # Map competitor TVL to our (n_dates, n_pools) grid
        comp_tvl_grid = np.full((n_dates, n_pools), np.nan)
        for j, pid in enumerate(pool_ids):
            if pid not in comp_pool_ids:
                continue
            cj = comp_pool_ids.index(pid)
            for t, date in enumerate(date_list):
                date_str = str(pd.Timestamp(date))[:10]
                if date_str in comp_date_to_idx:
                    ci = comp_date_to_idx[date_str]
                    val = comp_tvl_matrix[ci, cj]
                    if np.isfinite(val) and val > 0:
                        comp_tvl_grid[t, j] = val

        # Forward-fill / back-fill gaps per pool
        for j in range(n_pools):
            col = comp_tvl_grid[:, j]
            mask = np.isfinite(col)
            if mask.any() and not mask.all():
                s = pd.Series(col, index=date_list).ffill().bfill()
                comp_tvl_grid[:, j] = s.values

        # Flag pools with no competitor data
        has_comp = np.zeros(n_pools, dtype=bool)
        for j in range(n_pools):
            has_comp[j] = np.isfinite(comp_tvl_grid[:, j]).any()

        n_with = has_comp.sum()
        print(f"  Competitor TVL: {n_with}/{n_pools} pools with data")

        # Per-sample log(competitor_tvl), floor at $1
        raw_comp = np.array([
            comp_tvl_grid[day_idx[s], pool_idx[s]]
            for s in range(n_samples)], dtype=np.float64)

        # For pools without data, impute with median of pools that have data
        valid_comp = raw_comp[np.isfinite(raw_comp) & (raw_comp > 0)]
        fallback_val = float(np.median(valid_comp)) if len(valid_comp) > 0 else 1e6
        raw_comp = np.where(np.isfinite(raw_comp) & (raw_comp > 0),
                            raw_comp, fallback_val)
        log_comp_tvl = np.log(np.maximum(raw_comp, 1.0)).astype(np.float32)
        print(f"  Fallback comp TVL for missing pools: ${fallback_val:,.0f}")
        for j in range(n_pools):
            if not has_comp[j]:
                print(f"    No competitor data: {pool_ids[j][:16]}"
                      f" ({matched_clean[pool_ids[j]].get('tokens', '?')})")
    else:
        print(f"  WARNING: no competitor TVL file at {comp_path}")
        log_comp_tvl = np.full(n_samples, np.log(1e6), dtype=np.float32)
        has_comp = np.zeros(n_pools, dtype=bool)

    # Token info
    pool_tokens = []
    for pid in pool_ids:
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        tok_a = toks[0]
        tok_b = toks[1] if len(toks) > 1 else toks[0]
        pool_tokens.append((tok_a, tok_b))

    removed_names = [feat_names[i] for i in sorted(remove_cols)]
    print(f"  Removed: {removed_names}")
    print(f"  Market features ({len(market_names)}): {market_names}")

    return {
        "x_market": x_market,
        "log_tvl": log_tvl,
        "log_comp_tvl": log_comp_tvl,
        "has_comp": has_comp,
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
    }


# ---- Model ----

def forward_mm(params, x_market, log_tvl, pool_idx, log_comp_tvl=None):
    """MM forward pass → log(V_noise) per sample.

    K modes (checked in order):
      - Observed: log_comp_tvl provided + params has "k_scale" (2,)
          K = exp(k_scale[0] + k_scale[1] * log_comp_tvl)
      - Per-pool: params contains "log_K" (n_pools,)
      - Shared k_params: params contains "k_params" (3,)  [legacy]
    """
    log_alpha = params["log_alpha"]
    gamma = params["gamma"]

    alpha_i = log_alpha[pool_idx]
    tvl = jnp.exp(log_tvl)

    # K
    if log_comp_tvl is not None and "k_scale" in params:
        # Observed competitor TVL with learned scale/offset
        k_s = params["k_scale"]
        log_K = k_s[0] + k_s[1] * log_comp_tvl
        K = jnp.exp(log_K)
    elif log_comp_tvl is not None and "k_scale" not in params and "log_K" not in params:
        # Observed competitor TVL, used directly as K
        K = jnp.exp(log_comp_tvl)
    elif "log_K" in params:
        K = jnp.exp(params["log_K"][pool_idx])
    elif "k_params" in params:
        # Legacy Binance-volume mode (kept for loading old models)
        K = jnp.exp(params["k_params"][0])
    else:
        K = jnp.exp(jnp.array(14.5))  # fallback

    # Market features: shared or per-pool gamma
    if gamma.ndim == 2:
        per_sample_gamma = gamma[pool_idx]
        market_term = jnp.sum(x_market * per_sample_gamma, axis=1)
    else:
        market_term = x_market @ gamma

    log_saturation = log_tvl - jnp.log(K + tvl)
    return alpha_i + market_term + log_saturation


def make_loss_fn(pool_coeffs, pool_gas, n_pools):
    """Loss with PCHIP arb + MM noise."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    def loss_fn(params, x_market, log_tvl, log_comp_tvl, y_total,
                sample_grid_days, pool_idx, l2_alpha, huber_delta):
        log_cadence = params["log_cadence"]

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

        # V_noise from MM
        log_v_noise = forward_mm(
            params, x_market, log_tvl, pool_idx,
            log_comp_tvl=log_comp_tvl)

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
    log_comp_tvl = jnp.array(data["log_comp_tvl"])
    y_total = jnp.array(data["y_total"])
    sgd = jnp.array(data["sample_grid_days"])
    pidx = jnp.array(data["pool_idx"])

    for epoch in range(n_epochs):
        loss, grads = grad_fn(
            params, x_market, log_tvl, log_comp_tvl,
            y_total, sgd, pidx, l2_alpha, huber_delta)

        for k in params:
            g = grads[k]
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g ** 2
            m_hat = m[k] / (1 - b1 ** (epoch + 1))
            v_hat = v[k] / (1 - b2 ** (epoch + 1))
            params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            ev = evaluate(params, data)
            if "k_scale" in params:
                ks = np.array(params["k_scale"])
                k_str = f"  k_s=[{ks[0]:.2f},{ks[1]:.3f}]"
            elif "k_params" in params:
                k_p = np.array(params["k_params"])
                k_str = f"  k=[{k_p[0]:.2f},{k_p[1]:.3f},{k_p[2]:.3f}]"
            else:
                k_str = ""
            K_med = float(np.median(list(ev["K_values"].values())))
            cad = np.exp(np.array(params["log_cadence"]))
            print(f"  epoch {epoch:5d}  loss={float(loss):.4f}"
                  f"  R²={ev['median_r2']:.3f}"
                  f"  K_med=${K_med/1e6:.1f}M{k_str}"
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

    log_v_noise = np.array(forward_mm(
        params, jnp.array(data["x_market"]),
        jnp.array(data["log_tvl"]),
        jnp.array(data["pool_idx"]),
        log_comp_tvl=jnp.array(data["log_comp_tvl"])))

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

    # Per-pool median K
    K_values = {}
    if "k_scale" in params:
        ks = np.array(params["k_scale"])
        for i in range(n_pools):
            mask = pool_idx == i
            if not mask.any():
                K_values[data["pool_ids"][i]] = 0
                continue
            lc = data["log_comp_tvl"][mask]
            log_K_i = ks[0] + ks[1] * lc
            K_values[data["pool_ids"][i]] = float(np.exp(np.median(log_K_i)))
    elif "log_K" in params:
        for i in range(n_pools):
            K_values[data["pool_ids"][i]] = float(np.exp(params["log_K"][i]))
    elif "k_params" in params:
        k_p = np.array(params["k_params"])
        for i in range(n_pools):
            K_values[data["pool_ids"][i]] = float(np.exp(k_p[0]))
    else:
        # Observed K: compute from log_comp_tvl directly
        for i in range(n_pools):
            mask = pool_idx == i
            if mask.any():
                K_values[data["pool_ids"][i]] = float(
                    np.exp(np.median(data["log_comp_tvl"][mask])))
            else:
                K_values[data["pool_ids"][i]] = 1e6

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

    for i in range(n_pools):
        pid = data["pool_ids"][i]
        toks = data["pool_tokens"][i]
        label = f"{toks[0]}/{toks[1]}"
        mask = pool_idx == i
        if mask.sum() == 0:
            continue

        # Per-pool K (median)
        if "k_scale" in params:
            ks = np.array(params["k_scale"])
            lc = data["log_comp_tvl"][mask]
            K_i = float(np.exp(np.median(ks[0] + ks[1] * lc)))
        elif "log_K" in params:
            K_i = float(np.exp(params["log_K"][i]))
        elif "k_params" in params:
            K_i = float(np.exp(np.array(params["k_params"])[0]))
        else:
            # Observed K directly from competitor TVL
            K_i = float(np.exp(np.median(data["log_comp_tvl"][mask])))
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
    parser.add_argument("--shared-K", action="store_true",
                        help="Predict K from Binance volumes (3 shared params)")
    parser.add_argument("--observed-K", action="store_true",
                        help="Use observed competitor TVL from DeFi Llama as K")
    parser.add_argument("--per-pool-gamma", action="store_true",
                        help="Per-pool market feature coefficients")
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--trend-windows", type=int, nargs="+", default=[7])
    parser.add_argument("--include-cross-pool", action="store_true")
    parser.add_argument("--tune", type=int, default=0,
                        help="Optuna sweep (0 = single run)")
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

    if args.tune > 0:
        run_optuna(data, args.tune)
        return

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
        "gamma": gamma_init,
        "log_cadence": jnp.array(data["init_log_cadences"]),
    }
    if args.observed_K:
        # K = competitor_tvl directly. No learned params for K.
        # log_comp_tvl is passed as data, not as a parameter.
        pass
    elif args.shared_K:
        params["k_params"] = jnp.array([args.init_log_K, 0.0, 0.0])
    else:
        params["log_K"] = jnp.full(n_pools, args.init_log_K)
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

    if "k_scale" in params:
        ks = np.array(params["k_scale"])
        print(f"  Observed K: offset={ks[0]:.3f}, slope={ks[1]:.3f}")
    elif "k_params" in params:
        k_p = np.array(params["k_params"])
        print(f"  k_params: k_0={k_p[0]:.2f}, k_min={k_p[1]:.4f}, k_max={k_p[2]:.4f}")
    elif "log_K" in params:
        K_med = float(np.exp(np.median(np.array(params["log_K"]))))
        print(f"  Per-pool K: median=${K_med/1e6:.1f}M")
    else:
        K_med = float(np.median(list(train_eval["K_values"].values())))
        print(f"  Observed K (fixed): median=${K_med/1e6:.1f}M")

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


def run_optuna(data, n_trials):
    """Optuna hyperparameter sweep for MM noise model."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_pools = data["n_pools"]
    n_market = data["n_market_feat"]
    n_samples = len(data["pool_idx"])

    # 70/30 temporal split
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
    print(f"  Optuna split: {train_mask.sum()} train, {eval_mask.sum()} eval")

    def _ridge(X, y, alpha=1.0):
        XtX = X.T @ X + alpha * np.eye(X.shape[1])
        return np.linalg.solve(XtX, X.T @ y)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 3e-2, log=True)
        l2_alpha = trial.suggest_float("l2_alpha", 1e-5, 1e-1, log=True)
        huber_delta = trial.suggest_categorical("huber_delta", [0.5, 1.0, 1.5])
        init_log_K = trial.suggest_float("init_log_K", 14.0, 20.0)
        n_epochs = trial.suggest_categorical("n_epochs", [2000, 3000, 5000])
        per_pool_gamma = trial.suggest_categorical("per_pool_gamma", [True, False])
        if per_pool_gamma:
            gamma_init = jnp.zeros((n_pools, n_market))
        else:
            gamma_init = jnp.zeros(n_market)

        params = {
            "log_alpha": jnp.zeros(n_pools),
            "k_params": jnp.array([init_log_K, 0.0, 0.0]),
            "gamma": gamma_init,
            "log_cadence": jnp.array(data["init_log_cadences"]),
        }

        # Warm-start gamma
        x_trn = train_data["x_market"]
        y_trn = train_data["y_total"]
        if per_pool_gamma:
            pidx = train_data["pool_idx"]
            for i in range(n_pools):
                mask_i = pidx == i
                if mask_i.sum() < 5:
                    continue
                X_i = np.concatenate([x_trn[mask_i],
                                      np.ones((mask_i.sum(), 1))], 1)
                w = _ridge(X_i, y_trn[mask_i])
                params["gamma"] = params["gamma"].at[i].set(
                    jnp.array(w[:-1].astype(np.float32)))
                params["log_alpha"] = params["log_alpha"].at[i].set(
                    float(w[-1]))
        else:
            X_all = np.concatenate([x_trn, np.ones((len(y_trn), 1))], 1)
            w = _ridge(X_all, y_trn)
            params["gamma"] = jnp.array(w[:-1].astype(np.float32))

        grad_fn = make_loss_fn(data["pool_coeffs"], data["pool_gas"], n_pools)
        params = train(params, train_data, grad_fn, n_epochs, lr,
                       l2_alpha, huber_delta, verbose=False)

        # Eval
        eval_result = evaluate(params, eval_data)
        med_r2 = eval_result["median_r2"]

        K_med = float(np.median([v for v in eval_result["K_values"].values()]))
        k_p = np.array(params["k_params"])
        pp_str = "pp" if per_pool_gamma else "sh"
        print(f"  Trial {trial.number}: eval={med_r2:.4f}"
              f"  K_med=${K_med/1e6:.1f}M"
              f"  k=[{k_p[0]:.1f},{k_p[1]:.3f},{k_p[2]:.3f}]"
              f"  {pp_str} ep={n_epochs} lr={lr:.1e} l2={l2_alpha:.1e}"
              f"  hub={huber_delta}")

        # Save every trial
        trial_dir = os.path.join("results", "mm_noise", "trials",
                                 f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        save_dict = {k: np.array(v) for k, v in params.items()}
        np.savez(os.path.join(trial_dir, "model.npz"), **save_dict)
        meta = {
            "pool_ids": data["pool_ids"],
            "pool_tokens": data["pool_tokens"],
            "market_names": data["market_names"],
            "n_pools": n_pools,
            "n_market_feat": n_market,
            "per_pool_gamma": per_pool_gamma,
            "eval_r2": med_r2,
            "hparams": {
                "lr": lr, "l2_alpha": l2_alpha, "huber_delta": huber_delta,
                "init_log_K": init_log_K, "n_epochs": n_epochs,
                "per_pool_gamma": per_pool_gamma,
            },
        }
        with open(os.path.join(trial_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return med_r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{'='*70}")
    print(f"Optuna Results (MM noise)")
    print(f"{'='*70}")
    print(f"  Best eval R²: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}: {v}")

    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999,
                    reverse=True)
    print(f"\n  Top 10:")
    for t in trials[:10]:
        if t.value is not None:
            print(f"    #{t.number}: eval={t.value:.4f}  {t.params}")

    # Copy best to top-level
    best_dir = os.path.join("results", "mm_noise", "trials",
                            f"trial_{study.best_trial.number:04d}")
    if os.path.exists(os.path.join(best_dir, "model.npz")):
        import shutil
        for fn in ("model.npz", "meta.json"):
            shutil.copy2(os.path.join(best_dir, fn),
                         os.path.join("results", "mm_noise", fn))
        print(f"\n  Copied best trial ({study.best_trial.number})"
              f" to results/mm_noise/")

    return study


if __name__ == "__main__":
    main()
