"""Causal noise volume estimation: TVL decomposition + deconfounder sensitivity.

Two identification strategies for the causal effect of TVL on noise volume:

**Primary: TVL decomposition (IV-style)**
  Decomposes Δlog(TVL) into:
    - Price-driven: Δlog(TVL) - Δlog(shares) — market price moves, more
      exogenous to pool-specific trading activity (conditional on BTC/token
      market features already in the model)
    - Flow-driven: Δlog(shares) — LP deposits/withdrawals, endogenous
      (LPs deposit when they expect fees → correlated with noise)
  If b_tvl estimated from price-driven variation ≈ observational b_tvl,
  the coefficient is likely causal.

**Secondary: Deconfounder sensitivity analysis (Wang & Blei 2019)**
  Fit a factor model (PPCA) on covariates only (no outcome), extract
  latent factors Z_hat, include them in the outcome model.
  This is a sensitivity analysis: if b_tvl shifts substantially when
  conditioning on Z_hat, there's evidence of unobserved confounding.
  If it's stable, confounding through the covariate structure is small.

  NB: The deconfounder has known theoretical limitations (D'Amour 2019).
  Wang & Blei (2020, arXiv:2003.04948) respond that D'Amour's
  counterexamples violate the required assumptions (pinpointability).
  The theory holds under its assumptions, but the key assumption (no
  unobserved single-cause confounders) is domain-specific and
  uncheckable. Results should be interpreted as sensitivity bounds.

**Diagnostics:**
  - Variance decomposition of log_tvl: between-pool vs within-pool
  - Within-pool simple regression: Δlog(V_obs) on Δlog(TVL) per pool
  - These test whether the observational b_tvl reflects cross-sectional
    or temporal variation

Usage:
  python experiments/run_deconfounder_noise.py
  python experiments/run_deconfounder_noise.py --n-factors 1 2 3 5
"""

import argparse
import os
import time

import jax.numpy as jnp
import numpy as np


# ---- Factor model ----


def fit_ppca(X, n_components):
    """Probabilistic PCA. Returns Z_hat and the model."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    Z_hat = pca.fit_transform(X)
    print(f"    PPCA({n_components}): explained var = "
          f"{pca.explained_variance_ratio_.sum():.3f}  "
          f"per-component: {np.round(pca.explained_variance_ratio_, 3)}")
    return Z_hat, pca


def build_augmented_data(data, Z_hat):
    """Augment covariate matrix with standardized substitute confounders."""
    x_orig = data["x"]
    n_z = Z_hat.shape[1]

    z_mean = Z_hat.mean(axis=0)
    z_std = Z_hat.std(axis=0)
    z_std[z_std < 1e-6] = 1.0
    Z_std = ((Z_hat - z_mean) / z_std).astype(np.float32)

    x_aug = np.concatenate([x_orig, Z_std], axis=1)
    data_aug = dict(data)
    data_aug["x"] = x_aug
    data_aug["n_feat"] = x_aug.shape[1]
    data_aug["feat_names"] = data["feat_names"] + [f"Z_{k}" for k in range(n_z)]
    data_aug["x_mean"] = np.concatenate([
        data["x_mean"], z_mean.astype(np.float32)])
    data_aug["x_std"] = np.concatenate([
        data["x_std"], z_std.astype(np.float32)])
    return data_aug


def _tvl_col_index(feat_names):
    """Find TVL column index from feature names (robust to reordering)."""
    return feat_names.index("xobs_1")


def _intercept_col_index(feat_names):
    """Find intercept column index."""
    return feat_names.index("xobs_0")


# ---- TVL decomposition ----


def decompose_tvl(matched_clean, pool_ids, sample_pools, sample_days,
                  date_to_idx, n_dates, n_pools):
    """Decompose Δlog(TVL) into price-driven and flow-driven components.

    Uses log_tvl (not log_tvl_lag1) for the decomposition to avoid
    mixing lags. total_shares is assumed to be contemporaneous with TVL.

    flow = Δlog(shares)  — LP deposits/withdrawals
    price = Δlog(tvl) - Δlog(shares)  — price changes

    Returns per-sample arrays and a validity mask.
    """
    log_shares = np.full((n_dates, n_pools), np.nan)
    log_tvl = np.full((n_dates, n_pools), np.nan)

    for j, pid in enumerate(pool_ids):
        panel = matched_clean[pid]["panel"]
        dates = panel["date"].values

        has_shares = ("total_shares" in panel.columns and
                      panel["total_shares"].notna().any())
        if not has_shares:
            continue

        shares = panel["total_shares"].values.astype(float)
        shares = np.maximum(shares, 1e-10)

        # Use log_tvl (not lag) for contemporaneous decomposition
        if "log_tvl" in panel.columns:
            tvl_vals = panel["log_tvl"].values.astype(float)
        else:
            tvl_vals = panel["log_tvl_lag1"].values.astype(float)

        for k, date in enumerate(dates):
            t = date_to_idx.get(date)
            if t is not None:
                log_shares[t, j] = np.log(shares[k])
                log_tvl[t, j] = tvl_vals[k]

    n_samples = len(sample_pools)
    tvl_flow = np.full(n_samples, np.nan, dtype=np.float32)
    tvl_price = np.full(n_samples, np.nan, dtype=np.float32)

    for s in range(n_samples):
        i = sample_pools[s]
        t = sample_days[s]
        if t >= 1:
            d_log_shares = log_shares[t, i] - log_shares[t - 1, i]
            d_log_tvl = log_tvl[t, i] - log_tvl[t - 1, i]
            if np.isfinite(d_log_shares) and np.isfinite(d_log_tvl):
                tvl_flow[s] = d_log_shares
                tvl_price[s] = d_log_tvl - d_log_shares

    valid = np.isfinite(tvl_flow) & np.isfinite(tvl_price)
    return tvl_flow, tvl_price, valid


def run_tvl_decomposition_analysis(data, matched_clean, tvl_flow, tvl_price, valid):
    """Primary identification: compare b_tvl from price-driven vs all TVL."""
    from sklearn.linear_model import RidgeCV

    y = data["y_total"]
    x = data["x"]
    tvl_idx = _tvl_col_index(data["feat_names"])

    print(f"\n  Valid samples (have LP shares data): {valid.sum()}/{len(valid)}")
    if valid.sum() < 100:
        print("  Insufficient LP shares data for TVL decomposition.")
        return None

    x_valid = x[valid]
    y_valid = y[valid]
    tvl_price_valid = tvl_price[valid]
    tvl_flow_valid = tvl_flow[valid]

    print(f"  Price component: mean={tvl_price_valid.mean():.4f},"
          f" std={tvl_price_valid.std():.4f}")
    print(f"  Flow component:  mean={tvl_flow_valid.mean():.4f},"
          f" std={tvl_flow_valid.std():.4f}")

    # Observational b_tvl (Ridge on all features)
    ridge_obs = RidgeCV(alphas=np.logspace(-2, 4, 50))
    ridge_obs.fit(x_valid, y_valid)
    b_tvl_obs = ridge_obs.coef_[tvl_idx]

    # Replace TVL column with price-driven component only
    x_price = x_valid.copy()
    ps = tvl_price_valid.std()
    x_price[:, tvl_idx] = (tvl_price_valid - tvl_price_valid.mean()) / max(ps, 1e-6)
    ridge_price = RidgeCV(alphas=np.logspace(-2, 4, 50))
    ridge_price.fit(x_price, y_valid)
    b_tvl_price = ridge_price.coef_[tvl_idx]

    # Replace TVL column with flow-driven component only
    x_flow = x_valid.copy()
    fs = tvl_flow_valid.std()
    x_flow[:, tvl_idx] = (tvl_flow_valid - tvl_flow_valid.mean()) / max(fs, 1e-6)
    ridge_flow = RidgeCV(alphas=np.logspace(-2, 4, 50))
    ridge_flow.fit(x_flow, y_valid)
    b_tvl_flow = ridge_flow.coef_[tvl_idx]

    print(f"\n  b_tvl estimates (Ridge, all 22 features):")
    print(f"    All TVL variation:         {b_tvl_obs:+.4f}")
    print(f"    Price-driven only:         {b_tvl_price:+.4f}"
          f"  (more exogenous)")
    print(f"    Flow-driven only:          {b_tvl_flow:+.4f}"
          f"  (endogenous)")

    if abs(b_tvl_price - b_tvl_obs) < 0.3 * abs(b_tvl_obs):
        print(f"\n  → Price-driven ≈ observational: confounding small.")
    else:
        print(f"\n  → Price-driven ≠ observational: potential confounding.")

    return {"obs": b_tvl_obs, "price": b_tvl_price, "flow": b_tvl_flow}


# ---- Variance decomposition ----


def run_variance_decomposition(matched_clean, pool_ids):
    """Decompose log_tvl variance into between-pool and within-pool."""
    all_tvls = []
    pool_labels = []

    for j, pid in enumerate(pool_ids):
        panel = matched_clean[pid]["panel"]
        tvls = panel["log_tvl_lag1"].values.astype(float)
        valid = np.isfinite(tvls)
        all_tvls.extend(tvls[valid])
        pool_labels.extend([j] * valid.sum())

    all_tvls = np.array(all_tvls)
    pool_labels = np.array(pool_labels)

    total_var = np.var(all_tvls)
    pool_means = np.array([all_tvls[pool_labels == j].mean()
                           for j in range(len(pool_ids))])
    between_var = np.var(pool_means)
    within_vars = [np.var(all_tvls[pool_labels == j])
                   for j in range(len(pool_ids))]
    within_var = np.mean(within_vars)

    print(f"  Total variance:    {total_var:.4f}")
    print(f"  Between-pool:      {between_var:.4f}  ({between_var/total_var*100:.1f}%)")
    print(f"  Within-pool (avg): {within_var:.4f}  ({within_var/total_var*100:.1f}%)")
    print(f"  Pool mean range:   {pool_means.min():.1f} to {pool_means.max():.1f}")

    return {"total": total_var, "between": between_var, "within": within_var}


# ---- Within-pool simple regression ----


def run_within_pool_regressions(matched_clean, pool_ids):
    """Per-pool: Δlog(V_obs) on Δlog(TVL), no other covariates."""
    print(f"\n  {'Pool':16s} {'Tokens':16s} {'b_tvl':>8s} {'R²':>6s}"
          f" {'n':>5s} {'ΔTVL_std':>8s}")
    print(f"  {'-'*65}")

    b_tvls = []
    for pid in pool_ids:
        panel = matched_clean[pid]["panel"]
        log_vol = panel["log_volume"].values.astype(float)
        log_tvl = panel["log_tvl_lag1"].values.astype(float)

        d_vol = np.diff(log_vol)
        d_tvl = np.diff(log_tvl)

        valid = np.isfinite(d_vol) & np.isfinite(d_tvl)
        if valid.sum() < 10:
            continue

        dv = d_vol[valid]
        dt = d_tvl[valid]

        # OLS: Δlog_vol = a + b * Δlog_tvl
        X = np.column_stack([np.ones(len(dt)), dt])
        sol, _, _, _ = np.linalg.lstsq(X, dv, rcond=None)
        b = sol[1]
        pred = X @ sol
        ss_res = np.sum((dv - pred) ** 2)
        ss_tot = np.sum((dv - dv.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        b_tvls.append(b)
        tokens = matched_clean[pid]["tokens"]
        print(f"  {pid[:16]:16s} {tokens[:16]:16s} {b:+8.3f} {r2:6.3f}"
              f" {valid.sum():5d} {dt.std():8.4f}")

    if b_tvls:
        print(f"\n  Median within-pool b_tvl: {np.median(b_tvls):+.4f}")
        print(f"  Mean:                     {np.mean(b_tvls):+.4f}")
        print(f"  Std across pools:         {np.std(b_tvls):.4f}")
    return b_tvls


# ---- Lagged-average TVL analysis ----


def run_lagged_average_analysis(matched_clean, pool_ids):
    """Test TVL→noise at different timescales.

    If the daily Δ elasticity is ~0 but the level elasticity is ~2.5,
    the effect may operate on longer timescales. Test by regressing
    noise on rolling-average TVL at windows of 7, 14, 30, 60, 90 days.
    If b_tvl grows with window size, the relationship is real but slow.
    """
    windows = [1, 7, 14, 30, 60, 90]

    print(f"\n  Window  Median b_tvl  Mean b_tvl  Pools w/ data")
    print(f"  {'-'*55}")

    for w in windows:
        b_tvls = []
        n_pools_used = 0
        for pid in pool_ids:
            panel = matched_clean[pid]["panel"]
            log_vol = panel["log_volume"].values.astype(float)
            log_tvl = panel["log_tvl_lag1"].values.astype(float)

            if len(log_vol) < w + 10:
                continue

            # Rolling mean TVL over window w
            if w == 1:
                tvl_avg = log_tvl
            else:
                # Simple trailing average
                tvl_avg = np.full_like(log_tvl, np.nan)
                for t in range(w, len(log_tvl)):
                    vals = log_tvl[t - w:t]
                    if np.all(np.isfinite(vals)):
                        tvl_avg[t] = np.mean(vals)

            # Within-pool: demean both series
            valid = np.isfinite(log_vol) & np.isfinite(tvl_avg)
            if valid.sum() < 15:
                continue

            vol = log_vol[valid]
            tvl = tvl_avg[valid]
            vol_dm = vol - vol.mean()
            tvl_dm = tvl - tvl.mean()

            # OLS: demeaned_vol = b * demeaned_tvl
            if np.var(tvl_dm) < 1e-10:
                continue
            b = np.sum(vol_dm * tvl_dm) / np.sum(tvl_dm ** 2)
            b_tvls.append(b)
            n_pools_used += 1

        if b_tvls:
            print(f"  {w:5d}d  {np.median(b_tvls):+11.4f}  {np.mean(b_tvls):+10.4f}"
                  f"  {n_pools_used:>13d}")

    return windows


# ---- Deconfounder sensitivity ----


def run_deconfounder(data, n_factors_list, args):
    """Secondary: deconfounder sensitivity analysis across n_factors."""
    from experiments.run_linear_market_noise import make_loss_fn, train
    from sklearn.linear_model import RidgeCV

    X = data["x"]
    tvl_idx = _tvl_col_index(data["feat_names"])
    intercept_idx = _intercept_col_index(data["feat_names"])
    results = {}

    for n_f in n_factors_list:
        print(f"\n  --- n_factors={n_f} ---")
        Z_hat, _ = fit_ppca(X, n_f)
        data_aug = build_augmented_data(data, Z_hat)

        n_feat = data_aug["n_feat"]
        n_pools = data_aug["n_pools"]

        # Ridge warm-start
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 50))
        ridge.fit(data_aug["x"], data_aug["y_total"])
        sol = ridge.coef_.copy()
        sol[intercept_idx] += ridge.intercept_

        params = {
            "log_cadence": jnp.array(data_aug["init_log_cadences"]),
            "noise_coeffs": jnp.array(sol.astype(np.float32)),
        }

        grad_fn = make_loss_fn(data_aug["pool_coeffs"], data_aug["pool_gas"], n_pools)
        params = train(params, data_aug, grad_fn, args.epochs, args.lr,
                       args.l2_alpha, args.huber_delta, verbose=False)

        nc = np.array(params["noise_coeffs"])
        b_tvl = nc[tvl_idx]
        n_orig = data["n_feat"]
        z_coeffs = nc[n_orig:n_orig + n_f]

        print(f"    b_tvl = {b_tvl:+.4f}  "
              f"  Z coeffs: {np.round(z_coeffs, 3)}")

        results[n_f] = {
            "b_tvl": float(b_tvl),
            "z_coeffs": z_coeffs.tolist(),
            "explained_var": float(Z_hat.var(axis=0).sum() / X.var(axis=0).sum()),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-factors", type=int, nargs="+", default=[1, 2, 3, 5],
                        help="Number of latent factors to sweep")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l2-alpha", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--trend-windows", type=int, nargs="+", default=[7])
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Causal Noise Volume Estimation")
    print("  1. Variance decomposition (between vs within pool)")
    print("  2. Within-pool simple regressions")
    print("  3. TVL decomposition (price vs flow)")
    print("  4. Deconfounder sensitivity (PPCA factors)")
    print(f"  n_factors sweep: {args.n_factors}")
    print("=" * 70)

    from experiments.run_linear_market_noise import load_stage1, build_data

    matched_clean, option_c_clean = load_stage1()

    print("\nBuilding data...")
    t0 = time.time()
    data = build_data(
        matched_clean, option_c_clean,
        trend_windows=tuple(args.trend_windows),
        include_market=True, include_cross_pool=True,
    )
    pool_ids = data["pool_ids"]
    n_pools = data["n_pools"]
    print(f"  {len(data['pool_idx'])} samples, {n_pools} pools,"
          f" {data['n_feat']} features, {time.time() - t0:.1f}s")

    # ---- 1. Variance decomposition ----
    print(f"\n{'='*70}")
    print("1. Variance Decomposition of log_tvl_lag1")
    print(f"{'='*70}")
    var_results = run_variance_decomposition(matched_clean, pool_ids)

    # ---- 2. Within-pool simple regressions ----
    print(f"\n{'='*70}")
    print("2. Within-pool: Δlog(V_obs) ~ Δlog(TVL) (no other covariates)")
    print(f"{'='*70}")
    within_b_tvls = run_within_pool_regressions(matched_clean, pool_ids)

    # ---- 2b. Lagged-average TVL (timescale test) ----
    print(f"\n{'='*70}")
    print("2b. Lagged-Average TVL: Does b_tvl grow with averaging window?")
    print(f"    (Tests whether the TVL→noise effect is slow-moving)")
    print(f"{'='*70}")
    run_lagged_average_analysis(matched_clean, pool_ids)

    # ---- 3. TVL decomposition ----
    print(f"\n{'='*70}")
    print("3. TVL Decomposition: Price-driven vs Flow-driven")
    print(f"{'='*70}")

    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    tvl_flow, tvl_price, valid = decompose_tvl(
        matched_clean, pool_ids, data["pool_idx"], data["day_idx"],
        date_to_idx, len(date_list), n_pools,
    )
    tvl_results = run_tvl_decomposition_analysis(
        data, matched_clean, tvl_flow, tvl_price, valid)

    # ---- 4. Deconfounder sensitivity ----
    print(f"\n{'='*70}")
    print("4. Deconfounder Sensitivity Analysis")
    print(f"   (Wang & Blei 2019; D'Amour 2019; Wang & Blei 2020)")
    print(f"{'='*70}")
    deconf_results = run_deconfounder(data, args.n_factors, args)

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY: b_tvl across identification strategies")
    print(f"{'='*70}")

    # Observational baseline from artifact
    obs_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "linear_market_noise", "model.npz",
    )
    if os.path.exists(obs_path):
        obs_nc = np.load(obs_path)["noise_coeffs"]
        tvl_idx = _tvl_col_index(data["feat_names"])
        if obs_nc.ndim == 2:
            b_obs = float(np.median(obs_nc[:, tvl_idx]))
            print(f"  Observational (per-pool median): {b_obs:+.4f}")
        else:
            b_obs = float(obs_nc[tvl_idx])
            print(f"  Observational (shared):          {b_obs:+.4f}")

    between_pct = var_results["between"] / var_results["total"] * 100
    print(f"\n  Variance decomposition: {between_pct:.0f}% between-pool,"
          f" {100-between_pct:.0f}% within-pool")

    if within_b_tvls:
        print(f"  Within-pool Δ regressions:       "
              f"median={np.median(within_b_tvls):+.4f}"
              f"  (mean={np.mean(within_b_tvls):+.4f})")

    if tvl_results:
        print(f"\n  TVL decomposition (Ridge, 22 features):")
        print(f"    All variation:                 {tvl_results['obs']:+.4f}")
        print(f"    Price-driven (exogenous):      {tvl_results['price']:+.4f}")
        print(f"    Flow-driven (endogenous):      {tvl_results['flow']:+.4f}")

    print(f"\n  Deconfounder sensitivity (shared, learnable cadence):")
    print(f"  {'n_factors':>10s}  {'b_tvl':>8s}")
    for n_f, r in sorted(deconf_results.items()):
        print(f"  {n_f:>10d}  {r['b_tvl']:+8.4f}")

    # Stability
    b_tvls_d = [r["b_tvl"] for r in deconf_results.values()]
    rng = max(b_tvls_d) - min(b_tvls_d)
    mn = np.mean(b_tvls_d)
    stable = rng < 0.3 * abs(mn)
    print(f"\n  Deconfounder: {'STABLE' if stable else 'VARIES'}"
          f" (range {rng:.3f}, mean {mn:+.3f})")

    print(f"\n  Interpretation:")
    if tvl_results and abs(tvl_results['price']) < 0.5:
        print(f"    Daily b_tvl (Δ regression, price-driven) is near zero.")
        print(f"    This does NOT mean the long-run effect is zero:")
        print(f"    - Noise may respond slowly to TVL (routing updates,")
        print(f"      aggregator discovery, ecosystem integration)")
        print(f"    - The lagged-average analysis above tests this")
        print(f"    - The per-pool b_tvl of ~1.0 captures medium-frequency")
        print(f"      within-pool variation and is the best working estimate")
        print(f"    - Changing reClAMM concentration is a structural change")
        print(f"      (like being a different pool), not a daily TVL shock")
        print(f"    → Use per-pool b_tvl (~1.0) for counterfactuals, with")
        print(f"      sensitivity analysis across [0.5, 1.0, 2.0]")


if __name__ == "__main__":
    main()
