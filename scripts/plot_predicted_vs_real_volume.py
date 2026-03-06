"""Plot predicted vs real daily volume for pool registry pools.

Uses the fitted noise model (from calibrate_noise_unified.py) to compute
predicted daily log-volume for each pool in the registry, and overlays
the actual observed volume from the Balancer API panel data.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from pool_registry import POOL_REGISTRY, BALANCER_API_CHAIN


def main():
    fitted_path = "results/unified_full_90d.json"
    panel_path = "local_data/noise_calibration/panel.parquet"
    output_dir = "results/unified_full_90d"
    os.makedirs(output_dir, exist_ok=True)

    with open(fitted_path) as f:
        fitted = json.load(f)

    panel = pd.read_parquet(panel_path)

    # Deduplicate registry: multiple entries can share the same pool address
    # (e.g. cbBTC_WETH and cbBTC_WETH_post_oct). Group by address.
    unique_pools = {}
    for label, pool in POOL_REGISTRY.items():
        addr = pool.pool_address.lower()
        if addr not in unique_pools:
            unique_pools[addr] = (label, pool)

    # Match to panel
    matched = []
    for addr, (label, pool) in unique_pools.items():
        pid_matches = [
            pid for pid in fitted["pools"]
            if addr in pid.lower()
        ]
        if pid_matches:
            pid = pid_matches[0]
            matched.append((label, pool, pid))
        else:
            print(f"  {label}: not in fitted model (skipping)")

    if not matched:
        print("No registry pools found in the fitted model.")
        return

    print(f"Plotting {len(matched)} pools: {[m[0] for m in matched]}")

    # Determine grid layout
    n = len(matched)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, (label, pool, pid) in enumerate(matched):
        ax = axes[idx // ncols][idx % ncols]
        pool_data = fitted["pools"][pid]
        theta = np.array(pool_data["theta_median"])
        # theta = [intercept, b_tvl, b_sigma, b_weekend]

        # Get panel data for this pool
        pool_panel = panel[panel["pool_id"] == pid].copy()
        pool_panel = pool_panel.sort_values("date")

        if len(pool_panel) == 0:
            ax.set_title(f"{label}: no panel data")
            continue

        # Filter to last 90 days (matching training window)
        max_date = panel["date"].max()
        if hasattr(max_date, "date"):
            max_date = max_date
        from datetime import date, timedelta
        if isinstance(max_date, date):
            cutoff = max_date - timedelta(days=90)
        else:
            cutoff = pd.Timestamp(max_date) - pd.Timedelta(days=90)
        pool_panel = pool_panel[
            pool_panel["date"].apply(
                lambda d: d >= cutoff if isinstance(d, date)
                else pd.Timestamp(d).date() >= cutoff
            )
        ].copy()

        if len(pool_panel) < 5:
            ax.set_title(f"{label}: <5 obs in 90d window")
            continue

        # Build x_obs: [1, log_tvl_lag1, volatility, weekend]
        x_obs = np.column_stack([
            np.ones(len(pool_panel)),
            pool_panel["log_tvl_lag1"].values,
            pool_panel["volatility"].values,
            pool_panel["weekend"].values,
        ])

        predicted_log_vol = x_obs @ theta
        actual_log_vol = pool_panel["log_volume"].values

        # Convert to USD volume for interpretability
        predicted_vol = np.exp(predicted_log_vol)
        actual_vol = np.exp(actual_log_vol)

        dates = pd.to_datetime(pool_panel["date"].values)

        # Plot
        ax.plot(dates, actual_vol, "o-", color="steelblue", markersize=3,
                linewidth=1, alpha=0.7, label="Actual")
        ax.plot(dates, predicted_vol, "s--", color="orangered", markersize=3,
                linewidth=1, alpha=0.7, label="Predicted")
        ax.set_yscale("log")
        ax.set_ylabel("Daily volume (USD)")
        ax.set_title(f"{label}  ({pool_data['chain']})\n"
                      f"b_c={theta[1]:.2f}  b_σ={theta[2]:.2f}  "
                      f"b_wknd={theta[3]:.2f}")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        # Annotate R² for this pool
        ss_res = np.sum((actual_log_vol - predicted_log_vol) ** 2)
        ss_tot = np.sum((actual_log_vol - actual_log_vol.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax.text(0.02, 0.95, f"R²={r2:.3f}\nn={len(pool_panel)}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Noise model: predicted vs actual daily volume\n"
                 "(registry pools, 90-day training window)", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "registry_predicted_vs_real.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
