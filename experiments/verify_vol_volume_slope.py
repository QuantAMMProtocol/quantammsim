"""Verify: is the volatility-volume slope identical across fee tiers?

The claim (from noise_calibration_review.md): "the relationship between
price volatility and swap volume is identical across fee tiers (slope 0.91
for both low-fee and high-fee pools)."

This script tests the claim by:
1. Loading all pool panel data
2. Splitting pools by fee tier (low vs high)
3. Regressing log(volume) on log(volatility) within each group
4. Comparing slopes

If the slopes are similar, it means volatility drives organic volume
identically regardless of fee — supporting the arb/noise decomposition
(since arb intensity differs across fee tiers but noise doesn't).

Usage:
    python experiments/verify_vol_volume_slope.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)


def main():
    path = os.path.join(CACHE_DIR, "stage1.pkl")
    if not os.path.exists(path):
        print("ERROR: no stage1 cache.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    matched_clean = data["matched_clean"]

    # Collect per-observation data
    rows = []
    for pid, entry in matched_clean.items():
        panel = entry["panel"]
        fee = entry.get("fee", np.exp(panel["log_fee"].values[0]))
        chain = entry.get("chain", "unknown")
        tokens = entry.get("tokens", "?")

        log_vol = panel["log_volume"].values.astype(float)
        vol_raw = panel["volatility"].values.astype(float)
        log_tvl = panel["log_tvl_lag1"].values.astype(float)

        for i in range(len(log_vol)):
            if vol_raw[i] > 1e-10 and np.isfinite(log_vol[i]):
                rows.append({
                    "pool_id": pid,
                    "tokens": tokens,
                    "chain": chain,
                    "fee": fee,
                    "log_fee": np.log(fee),
                    "log_volume": log_vol[i],
                    "log_sigma": np.log(max(vol_raw[i], 1e-10)),
                    "log_tvl": log_tvl[i] if np.isfinite(log_tvl[i]) else np.nan,
                })

    df = pd.DataFrame(rows).dropna()
    print(f"Loaded {len(df)} observations from {df['pool_id'].nunique()} pools")

    # Fee tier split
    fees = df.groupby("pool_id")["fee"].first()
    median_fee = fees.median()
    print(f"\nFee distribution:")
    print(f"  min={fees.min():.5f}  median={median_fee:.5f}  max={fees.max():.5f}")
    print(f"  Unique fees: {sorted(fees.unique())}")

    low_fee_pools = set(fees[fees <= median_fee].index)
    high_fee_pools = set(fees[fees > median_fee].index)
    print(f"  Low-fee pools (≤{median_fee:.4f}): {len(low_fee_pools)}")
    print(f"  High-fee pools (>{median_fee:.4f}): {len(high_fee_pools)}")

    df_low = df[df["pool_id"].isin(low_fee_pools)]
    df_high = df[df["pool_id"].isin(high_fee_pools)]

    # OLS: log_volume ~ intercept + log_sigma
    def ols_slope(x, y):
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        # Standard error of slope
        n = len(x)
        se = np.sqrt(ss_res / (n - 2) / np.sum((x - x.mean()) ** 2))
        return beta[1], beta[0], r2, se

    print(f"\n{'='*60}")
    print("OLS: log(volume) ~ intercept + log(sigma)")
    print(f"{'='*60}")

    # All pools
    slope, intercept, r2, se = ols_slope(df["log_sigma"].values, df["log_volume"].values)
    print(f"\n  All pools ({len(df)} obs):")
    print(f"    slope = {slope:.4f} ± {1.96*se:.4f}  (95% CI)")
    print(f"    R² = {r2:.4f}")

    # Low fee
    slope_l, int_l, r2_l, se_l = ols_slope(df_low["log_sigma"].values, df_low["log_volume"].values)
    print(f"\n  Low-fee pools ({len(df_low)} obs, {len(low_fee_pools)} pools):")
    print(f"    slope = {slope_l:.4f} ± {1.96*se_l:.4f}")
    print(f"    R² = {r2_l:.4f}")

    # High fee
    slope_h, int_h, r2_h, se_h = ols_slope(df_high["log_sigma"].values, df_high["log_volume"].values)
    print(f"\n  High-fee pools ({len(df_high)} obs, {len(high_fee_pools)} pools):")
    print(f"    slope = {slope_h:.4f} ± {1.96*se_h:.4f}")
    print(f"    R² = {r2_h:.4f}")

    print(f"\n  Difference: {abs(slope_l - slope_h):.4f}")
    print(f"  Ratio: {slope_l/slope_h:.3f}")

    # Per-pool slopes
    print(f"\n{'='*60}")
    print("Per-pool OLS slopes")
    print(f"{'='*60}")
    pool_slopes = []
    print(f"\n  {'Pool':>16s}  {'Tokens':>20s}  {'Fee':>8s}  {'Slope':>8s}"
          f"  {'R²':>6s}  {'N':>5s}")
    for pid in sorted(matched_clean.keys()):
        pool_df = df[df["pool_id"] == pid]
        if len(pool_df) < 20:
            continue
        s, _, r, se = ols_slope(pool_df["log_sigma"].values, pool_df["log_volume"].values)
        fee = pool_df["fee"].iloc[0]
        tokens = pool_df["tokens"].iloc[0]
        pool_slopes.append({"pool_id": pid, "tokens": tokens, "fee": fee,
                            "slope": s, "r2": r, "n": len(pool_df)})
        print(f"  {pid[:16]}  {tokens:>20s}  {fee:>8.5f}  {s:>8.4f}"
              f"  {r:>6.3f}  {len(pool_df):>5d}")

    ps = pd.DataFrame(pool_slopes)
    if len(ps) > 0:
        low_slopes = ps[ps["fee"] <= median_fee]["slope"]
        high_slopes = ps[ps["fee"] > median_fee]["slope"]
        print(f"\n  Per-pool slope summary:")
        print(f"    Low-fee:  median={low_slopes.median():.4f},"
              f"  mean={low_slopes.mean():.4f}  (n={len(low_slopes)})")
        print(f"    High-fee: median={high_slopes.median():.4f},"
              f"  mean={high_slopes.mean():.4f}  (n={len(high_slopes)})")

    # Also try with TVL control: log_volume ~ log_sigma + log_tvl
    print(f"\n{'='*60}")
    print("OLS: log(volume) ~ intercept + log(sigma) + log(tvl)")
    print(f"{'='*60}")

    def ols_multi(df_sub):
        x = np.column_stack([
            np.ones(len(df_sub)),
            df_sub["log_sigma"].values,
            df_sub["log_tvl"].values,
        ])
        y = df_sub["log_volume"].values
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        y_hat = x @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return beta, 1 - ss_res / ss_tot

    beta_all, r2_all = ols_multi(df)
    print(f"\n  All: σ_slope={beta_all[1]:.4f}, tvl_slope={beta_all[2]:.4f}, R²={r2_all:.4f}")

    beta_l, r2_l = ols_multi(df_low)
    print(f"  Low: σ_slope={beta_l[1]:.4f}, tvl_slope={beta_l[2]:.4f}, R²={r2_l:.4f}")

    beta_h, r2_h = ols_multi(df_high)
    print(f"  High: σ_slope={beta_h[1]:.4f}, tvl_slope={beta_h[2]:.4f}, R²={r2_h:.4f}")

    print(f"\n  σ slope difference (low-high): {beta_l[1] - beta_h[1]:.4f}")


    # ---- Volume/TVL vs TVL (cross-pool) ----
    print(f"\n{'='*60}")
    print("Volume/TVL vs TVL (cross-pool)")
    print(f"{'='*60}")

    # Per-pool median volume and TVL
    pool_stats = []
    for pid in sorted(matched_clean.keys()):
        pool_df = df[df["pool_id"] == pid]
        if len(pool_df) < 20:
            continue
        med_vol = np.exp(np.median(pool_df["log_volume"].values))
        med_tvl = np.exp(np.median(pool_df["log_tvl"].values))
        tokens = pool_df["tokens"].iloc[0]
        fee = pool_df["fee"].iloc[0]
        vol_tvl = med_vol / med_tvl
        pool_stats.append({
            "pool_id": pid, "tokens": tokens, "fee": fee,
            "med_vol": med_vol, "med_tvl": med_tvl,
            "vol_tvl_pct": vol_tvl * 100,
            "log_med_tvl": np.log(med_tvl),
        })

    ps = pd.DataFrame(pool_stats)
    print(f"\n  {'Pool':>16s}  {'Tokens':>20s}  {'TVL':>14s}  {'Vol/day':>14s}  {'Vol/TVL':>8s}")
    for _, row in ps.sort_values("med_tvl").iterrows():
        print(f"  {row['pool_id'][:16]}  {row['tokens']:>20s}"
              f"  ${row['med_tvl']:>13,.0f}  ${row['med_vol']:>13,.0f}"
              f"  {row['vol_tvl_pct']:>7.1f}%")

    # OLS: log(vol/tvl) ~ log(tvl)
    log_vol_tvl = np.log(ps["med_vol"].values / ps["med_tvl"].values)
    log_tvl_vals = ps["log_med_tvl"].values
    slope_vt, int_vt, r2_vt, se_vt = ols_slope(log_tvl_vals, log_vol_tvl)
    print(f"\n  OLS: log(Vol/TVL) ~ log(TVL)")
    print(f"    slope = {slope_vt:.4f} ± {1.96*se_vt:.4f}")
    print(f"    R² = {r2_vt:.4f}")
    print(f"    (slope < 0 means Vol/TVL declines with TVL)")

    # Equivalent: log(Vol) ~ α + β*log(TVL), β < 1 means sublinear
    slope_v, int_v, r2_v, se_v = ols_slope(log_tvl_vals, np.log(ps["med_vol"].values))
    print(f"\n  OLS: log(Vol) ~ log(TVL)")
    print(f"    slope = {slope_v:.4f} ± {1.96*se_v:.4f}")
    print(f"    R² = {r2_v:.4f}")
    print(f"    (slope < 1 means sublinear = Vol/TVL declines)")

    # ---- TVL elasticity by TVL quartile (MM signature) ----
    print(f"\n{'='*60}")
    print("TVL Elasticity by TVL Quartile")
    print(f"{'='*60}")

    # Use observation-level data, not pool medians — more power
    # Within-quartile regression: log(vol) ~ log(tvl) for pools in each bin
    ps_sorted = ps.sort_values("med_tvl")
    n_q = len(ps_sorted) // 4
    quartiles = []
    for q in range(4):
        start = q * n_q
        end = (q + 1) * n_q if q < 3 else len(ps_sorted)
        q_pools = set(ps_sorted.iloc[start:end]["pool_id"])
        q_df = df[df["pool_id"].isin(q_pools)]
        if len(q_df) < 20:
            continue
        s, intercept, r2, se = ols_slope(q_df["log_tvl"].values,
                                          q_df["log_volume"].values)
        tvl_lo = np.exp(q_df["log_tvl"].min())
        tvl_hi = np.exp(q_df["log_tvl"].max())
        tvl_med = np.exp(q_df["log_tvl"].median())
        quartiles.append({
            "q": q + 1, "n_pools": len(q_pools), "n_obs": len(q_df),
            "tvl_lo": tvl_lo, "tvl_hi": tvl_hi, "tvl_med": tvl_med,
            "slope": s, "se": se, "r2": r2,
        })
        print(f"\n  Q{q+1}: TVL ${tvl_lo:,.0f} – ${tvl_hi:,.0f}"
              f" (median ${tvl_med:,.0f})")
        print(f"    {len(q_pools)} pools, {len(q_df)} obs")
        print(f"    slope = {s:.4f} ± {1.96*se:.4f}  R² = {r2:.4f}")

    if len(quartiles) >= 2:
        print(f"\n  Summary:")
        print(f"  {'Quartile':>10s}  {'Med TVL':>14s}  {'Slope':>8s}  {'95% CI':>16s}")
        for q in quartiles:
            ci = f"[{q['slope']-1.96*q['se']:.3f}, {q['slope']+1.96*q['se']:.3f}]"
            print(f"  Q{q['q']:>9d}  ${q['tvl_med']:>13,.0f}  {q['slope']:>8.4f}  {ci:>16s}")

        slope_q1 = quartiles[0]["slope"]
        slope_q4 = quartiles[-1]["slope"]
        print(f"\n  Q1→Q4 slope change: {slope_q4 - slope_q1:+.4f}")
        print(f"  (Negative = elasticity declines with TVL = MM signature)")

    # Also try: rolling window across pools sorted by TVL
    print(f"\n  Rolling 10-pool window:")
    print(f"  {'Window':>8s}  {'Med TVL':>14s}  {'Slope':>8s}  {'R²':>6s}")
    window = 10
    for start_i in range(0, len(ps_sorted) - window + 1, 3):
        w_pools = set(ps_sorted.iloc[start_i:start_i + window]["pool_id"])
        w_df = df[df["pool_id"].isin(w_pools)]
        if len(w_df) < 30:
            continue
        s, _, r2, se = ols_slope(w_df["log_tvl"].values,
                                  w_df["log_volume"].values)
        tvl_med = np.exp(w_df["log_tvl"].median())
        print(f"  {start_i:>3d}-{start_i+window:>3d}  ${tvl_med:>13,.0f}  {s:>8.4f}  {r2:>6.3f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: Vol/TVL vs TVL
        ax = axes[0]
        ax.scatter(ps["med_tvl"] / 1e6, ps["vol_tvl_pct"],
                   s=30, alpha=0.7, c="steelblue")
        for _, row in ps.iterrows():
            ax.annotate(row["tokens"].split(",")[0],
                        (row["med_tvl"] / 1e6, row["vol_tvl_pct"]),
                        fontsize=5, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Median TVL ($M)")
        ax.set_ylabel("Median Vol/TVL (%)")
        ax.set_title(f"Volume/TVL Declines with TVL\n"
                     f"log(Vol/TVL) ~ {slope_vt:.2f}·log(TVL), R²={r2_vt:.2f}")
        ax.grid(True, alpha=0.3)

        # Fit line
        tvl_fit = np.logspace(np.log10(ps["med_tvl"].min()),
                              np.log10(ps["med_tvl"].max()), 100)
        vol_tvl_fit = np.exp(int_vt + slope_vt * np.log(tvl_fit)) * 100
        ax.plot(tvl_fit / 1e6, vol_tvl_fit, "r--", linewidth=1, alpha=0.7)

        # Panel 2: Vol vs TVL (log-log)
        ax = axes[1]
        ax.scatter(ps["med_tvl"] / 1e6, ps["med_vol"] / 1e6,
                   s=30, alpha=0.7, c="coral")
        for _, row in ps.iterrows():
            ax.annotate(row["tokens"].split(",")[0],
                        (row["med_tvl"] / 1e6, row["med_vol"] / 1e6),
                        fontsize=5, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Median TVL ($M)")
        ax.set_ylabel("Median Daily Volume ($M)")
        ax.set_title(f"Volume vs TVL (cross-pool)\n"
                     f"log(Vol) ~ {slope_v:.2f}·log(TVL), R²={r2_v:.2f}")
        ax.grid(True, alpha=0.3)

        # Fit line + linear reference
        vol_fit = np.exp(int_v + slope_v * np.log(tvl_fit))
        ax.plot(tvl_fit / 1e6, vol_fit / 1e6, "r--", linewidth=1,
                alpha=0.7, label=f"slope={slope_v:.2f}")
        # Linear reference (slope=1)
        vol_linear = np.exp(int_v + 1.0 * np.log(tvl_fit))
        ax.plot(tvl_fit / 1e6, vol_linear / 1e6, "k:", linewidth=0.5,
                alpha=0.3, label="slope=1 (linear)")
        ax.legend(fontsize=8)

        # Panel 3: by fee tier
        ax = axes[2]
        for _, row in ps.iterrows():
            color = "steelblue" if row["fee"] <= median_fee else "coral"
            ax.scatter(row["med_tvl"] / 1e6, row["vol_tvl_pct"],
                       s=30, alpha=0.7, c=color)
            ax.annotate(row["tokens"].split(",")[0],
                        (row["med_tvl"] / 1e6, row["vol_tvl_pct"]),
                        fontsize=5, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Median TVL ($M)")
        ax.set_ylabel("Median Vol/TVL (%)")
        ax.set_title("Vol/TVL by Fee Tier\n"
                     f"blue=low fee (≤{median_fee:.4f}), red=high fee")
        ax.grid(True, alpha=0.3)

        fig.suptitle("Cross-Pool Evidence for Volume Saturation", fontsize=13)
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "mm_noise", "plots",
                           "cross_pool_vol_tvl.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved: {out}")
    except Exception as e:
        print(f"  Plot failed: {e}")


if __name__ == "__main__":
    main()
