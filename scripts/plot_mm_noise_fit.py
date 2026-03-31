"""Plot MM noise model fit: per-pool time series + TVL response curves.

Produces two types of plots:
1. Per-pool 6-panel time series (like plot_model_vs_real_reclamm.py):
   TVL, volume decomposition, V_noise, fee revenue, vol/TVL, pred/obs
2. Cross-pool TVL response curves showing MM saturation

Usage:
    python scripts/plot_mm_noise_fit.py
    python scripts/plot_mm_noise_fit.py --pool 0x9d1fcf346ea1b0
    python scripts/plot_mm_noise_fit.py --all-pools
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax.numpy as jnp


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)


def load_model(artifact_dir):
    """Load MM model artifact."""
    art = dict(np.load(os.path.join(artifact_dir, "model.npz"),
                        allow_pickle=True))
    with open(os.path.join(artifact_dir, "meta.json")) as f:
        meta = json.load(f)
    params = {}
    for k in art:
        params[k] = jnp.array(art[k])
    return params, meta


def compute_decomposition(params, meta, matched_clean, option_c_clean):
    """Compute V_arb, V_noise, V_total for all pools."""
    from experiments.run_mm_noise import build_mm_data, forward_mm
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    data = build_mm_data(matched_clean, option_c_clean,
                         trend_windows=(7,),
                         include_cross_pool=False)

    pool_ids = data["pool_ids"]
    n_pools = data["n_pools"]
    pool_idx = np.array(data["pool_idx"])
    sgd = np.array(data["sample_grid_days"])
    day_idx = np.array(data["day_idx"])
    y = np.array(data["y_total"])
    log_tvl = np.array(data["log_tvl"])

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

    # V_noise
    log_v_noise = np.array(forward_mm(
        params, jnp.array(data["x_market"]),
        jnp.array(data["log_tvl"]),
        jnp.array(data["pool_idx"])))
    v_noise = np.exp(log_v_noise)
    v_total = v_arb + v_noise
    v_obs = np.exp(y)

    # Reconstruct dates
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)

    dates = np.array([pd.Timestamp(date_list[d]) for d in day_idx])

    return {
        "pool_ids": pool_ids,
        "pool_tokens": data["pool_tokens"],
        "pool_idx": pool_idx,
        "dates": dates,
        "v_arb": v_arb,
        "v_noise": v_noise,
        "v_total": v_total,
        "v_obs": v_obs,
        "log_tvl": log_tvl,
        "tvl": np.exp(log_tvl),
    }


def plot_pool_timeseries(decomp, params, pool_i, output_dir):
    """6-panel time series for a single pool."""
    pid = decomp["pool_ids"][pool_i]
    toks = decomp["pool_tokens"][pool_i]
    label = f"{toks[0]}/{toks[1]}"

    mask = decomp["pool_idx"] == pool_i
    if mask.sum() < 10:
        print(f"  Skipping {pid[:16]} ({label}): {mask.sum()} samples")
        return

    dates = decomp["dates"][mask]
    v_arb = decomp["v_arb"][mask]
    v_noise = decomp["v_noise"][mask]
    v_total = decomp["v_total"][mask]
    v_obs = decomp["v_obs"][mask]
    tvl = decomp["tvl"][mask]

    K_i = float(np.exp(params["log_K"][pool_i]))

    # R²
    log_pred = np.log(np.maximum(v_total, 1e-10))
    log_obs = np.log(np.maximum(v_obs, 1e-10))
    ss_res = np.sum((log_pred - log_obs) ** 2)
    ss_tot = np.sum((log_obs - log_obs.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    # 1. TVL
    ax = axes[0]
    ax.plot(dates, tvl / 1e6, "k-", linewidth=0.7)
    ax.axhline(K_i / 1e6, color="red", linestyle="--", alpha=0.5,
               label=f"K = ${K_i/1e6:.1f}M")
    ax.set_ylabel("TVL ($M)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title(f"{label} — TVL (K={K_i/1e6:.1f}M)")
    ax.grid(True, alpha=0.3)

    # 2. Volume decomposition
    ax = axes[1]
    ax.fill_between(dates, 0, v_arb / 1e6, alpha=0.4, color="steelblue",
                    label="V_arb")
    ax.fill_between(dates, v_arb / 1e6, v_total / 1e6, alpha=0.4,
                    color="coral", label="V_noise (MM)")
    ax.plot(dates, v_obs / 1e6, "k-", linewidth=0.5, alpha=0.7,
            label="V_obs")
    ax.plot(dates, v_total / 1e6, "r--", linewidth=0.5, alpha=0.7,
            label="V_pred")
    ax.set_ylabel("Volume ($M/day)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title(f"Volume Decomposition (R²={r2:.3f})")
    ax.grid(True, alpha=0.3)

    # 3. V_noise only
    ax = axes[2]
    ax.fill_between(dates, 0, v_noise / 1e6, alpha=0.4, color="coral")
    ax.plot(dates, v_noise / 1e6, "r-", linewidth=0.5)
    noise_med = np.median(v_noise)
    ax.axhline(noise_med / 1e6, color="red", linestyle=":", alpha=0.5,
               label=f"median=${noise_med:,.0f}")
    ax.set_ylabel("V_noise ($M/day)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Noise Volume (MM model)")
    ax.grid(True, alpha=0.3)

    # 4. Fee revenue (assuming 0.25% fee, 25% protocol take)
    fee_rate = 0.0025
    protocol_take = 0.25
    fee_arb = v_arb * fee_rate * (1 - protocol_take)
    fee_noise = v_noise * fee_rate * (1 - protocol_take)
    fee_obs = v_obs * fee_rate * (1 - protocol_take)
    ax = axes[3]
    ax.fill_between(dates, 0, fee_arb, alpha=0.4, color="steelblue",
                    label="Arb fees")
    ax.fill_between(dates, fee_arb, fee_arb + fee_noise, alpha=0.4,
                    color="coral", label="Noise fees")
    ax.plot(dates, fee_obs, "k-", linewidth=0.5, alpha=0.7,
            label="Obs fees (approx)")
    ax.set_ylabel("Fee revenue ($/day)")
    ax.legend(fontsize=7)
    ax.set_title("Fee Revenue (0.25% fee, 75% LP)")
    ax.grid(True, alpha=0.3)

    # 5. Vol/TVL
    ax = axes[4]
    vol_tvl_obs = v_obs / tvl
    vol_tvl_pred = v_total / tvl
    ax.plot(dates, vol_tvl_obs * 100, "k-", linewidth=0.5, alpha=0.5,
            label="Observed")
    ax.plot(dates, vol_tvl_pred * 100, "r-", linewidth=0.5, alpha=0.5,
            label="Predicted")
    ax.axhline(np.median(vol_tvl_obs) * 100, color="black", linestyle=":",
               alpha=0.3)
    ax.axhline(np.median(vol_tvl_pred) * 100, color="red", linestyle=":",
               alpha=0.3)
    ax.set_ylabel("Vol/TVL (%)")
    ax.legend(fontsize=8)
    ax.set_title("Volume as % of TVL")
    ax.grid(True, alpha=0.3)

    # 6. Pred/Obs ratio
    ax = axes[5]
    ratio = v_total / np.maximum(v_obs, 1)
    ax.plot(dates, ratio, "b-", linewidth=0.5, alpha=0.5)
    ax.axhline(1.0, color="black", linestyle="-", alpha=0.3)
    med_ratio = np.median(ratio)
    ax.axhline(med_ratio, color="blue", linestyle=":", alpha=0.5,
               label=f"median={med_ratio:.2f}")
    ax.set_ylabel("Pred / Obs")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 20)
    ax.legend(fontsize=8)
    ax.set_title("Prediction Ratio")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Date")

    fig.suptitle(f"MM Noise Model — {label} ({pid[:16]})", fontsize=13)
    fig.tight_layout()
    out = os.path.join(output_dir, f"{pid[:16]}_mm_fit.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_tvl_response(params, meta, decomp, output_dir):
    """Cross-pool TVL response curves showing MM saturation."""
    pool_ids = meta["pool_ids"]
    pool_tokens = meta["pool_tokens"]
    n_pools = len(pool_ids)

    tvl_range = np.logspace(4, 10, 200)  # $10K to $10B

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Select pools with enough data for interesting plots
    pool_idx_arr = np.array(decomp["pool_idx"])
    interesting = []
    for i in range(n_pools):
        n = (pool_idx_arr == i).sum()
        if n >= 50:
            interesting.append(i)

    colors = plt.cm.tab20(np.linspace(0, 1, len(interesting)))

    # Panel 1: Noise volume vs TVL (absolute)
    ax = axes[0]
    for ci, i in enumerate(interesting):
        K_i = float(np.exp(params["log_K"][i]))

        mask = pool_idx_arr == i
        actual_noise = np.median(decomp["v_noise"][mask])
        actual_tvl = np.median(decomp["tvl"][mask])
        # Scale: noise(TVL) = actual_noise * [TVL/(K+TVL)] / [actual_TVL/(K+actual_TVL)]
        mm_actual = actual_tvl / (K_i + actual_tvl)
        mm_curve = tvl_range / (K_i + tvl_range)
        noise_curve = actual_noise * mm_curve / mm_actual

        label = f"{pool_tokens[i][0]}/{pool_tokens[i][1]}"
        ax.plot(tvl_range / 1e6, noise_curve / 1e6, color=colors[ci],
                linewidth=1.0, alpha=0.7, label=label if ci < 10 else None)
        # Mark actual TVL
        ax.scatter([actual_tvl / 1e6], [actual_noise / 1e6],
                   color=colors[ci], s=20, zorder=5)

    ax.set_xlabel("TVL ($M)")
    ax.set_ylabel("Daily Noise Volume ($M)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Noise Volume vs TVL (MM saturation)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Noise/TVL ratio vs TVL
    ax = axes[1]
    for ci, i in enumerate(interesting):
        K_i = float(np.exp(params["log_K"][i]))
        mask = pool_idx_arr == i
        actual_noise = np.median(decomp["v_noise"][mask])
        actual_tvl = np.median(decomp["tvl"][mask])
        mm_actual = actual_tvl / (K_i + actual_tvl)
        mm_curve = tvl_range / (K_i + tvl_range)
        noise_curve = actual_noise * mm_curve / mm_actual
        ratio_curve = noise_curve / tvl_range * 100

        label = f"{pool_tokens[i][0]}/{pool_tokens[i][1]}"
        ax.plot(tvl_range / 1e6, ratio_curve, color=colors[ci],
                linewidth=1.0, alpha=0.7, label=label if ci < 10 else None)

    ax.set_xlabel("TVL ($M)")
    ax.set_ylabel("Noise / TVL (%)")
    ax.set_xscale("log")
    ax.set_title("Noise as Fraction of TVL")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Elasticity vs TVL
    ax = axes[2]
    for ci, i in enumerate(interesting):
        K_i = float(np.exp(params["log_K"][i]))
        eps_curve = K_i / (K_i + tvl_range)

        label = f"{pool_tokens[i][0]}/{pool_tokens[i][1]}"
        ax.plot(tvl_range / 1e6, eps_curve, color=colors[ci],
                linewidth=1.0, alpha=0.7, label=label if ci < 10 else None)
        # Mark actual TVL
        actual_tvl = np.median(decomp["tvl"][pool_idx_arr == i])
        eps_actual = K_i / (K_i + actual_tvl)
        ax.scatter([actual_tvl / 1e6], [eps_actual],
                   color=colors[ci], s=20, zorder=5)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="ε=0.5")
    ax.set_xlabel("TVL ($M)")
    ax.set_ylabel("Elasticity ε(TVL)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_title("TVL Elasticity (K/(K+TVL))")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Michaelis-Menten Noise Model — TVL Response", fontsize=13)
    fig.tight_layout()
    out = os.path.join(output_dir, "mm_tvl_response.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_K_distribution(params, meta, decomp, output_dir):
    """Plot K values across pools with data quality indicator."""
    pool_ids = meta["pool_ids"]
    pool_tokens = meta["pool_tokens"]
    n_pools = len(pool_ids)
    pool_idx_arr = np.array(decomp["pool_idx"])

    K_vals = []
    labels = []
    n_days = []
    tvl_ranges = []
    for i in range(n_pools):
        mask = pool_idx_arr == i
        n = mask.sum()
        K_i = float(np.exp(params["log_K"][i]))
        K_vals.append(K_i)
        tok = pool_tokens[i]
        labels.append(f"{tok[0]}/{tok[1]}")
        n_days.append(n)
        if n > 0:
            tvl = decomp["tvl"][mask]
            tvl_ranges.append(np.log10(tvl.max() / max(tvl.min(), 1)))
        else:
            tvl_ranges.append(0)

    K_vals = np.array(K_vals)
    n_days = np.array(n_days)
    tvl_ranges = np.array(tvl_ranges)

    # Sort by K
    order = np.argsort(K_vals)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: K values as horizontal bar
    ax = axes[0]
    y_pos = np.arange(n_pools)
    colors = plt.cm.viridis(tvl_ranges[order] / max(tvl_ranges.max(), 1))
    ax.barh(y_pos, K_vals[order] / 1e6, color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in order], fontsize=7)
    ax.set_xlabel("K ($M)")
    ax.set_title("Half-Saturation TVL by Pool\n(color = log10 TVL range)")
    ax.axvline(np.median(K_vals) / 1e6, color="red", linestyle="--",
               alpha=0.5, label=f"median=${np.median(K_vals)/1e6:.1f}M")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: K vs data quality (n_days and TVL range)
    ax = axes[1]
    valid = n_days > 0
    sc = ax.scatter(n_days[valid], K_vals[valid] / 1e6,
                    c=tvl_ranges[valid], cmap="viridis",
                    s=50, alpha=0.7)
    for i in range(n_pools):
        if n_days[i] > 0:
            ax.annotate(labels[i], (n_days[i], K_vals[i] / 1e6),
                        fontsize=5, alpha=0.7)
    ax.set_xlabel("Number of training days")
    ax.set_ylabel("K ($M)")
    ax.set_title("K vs Data Quantity\n(color = log10 TVL range)")
    plt.colorbar(sc, ax=ax, label="log10(TVL_max/TVL_min)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Michaelis-Menten K Distribution", fontsize=13)
    fig.tight_layout()
    out = os.path.join(output_dir, "mm_K_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default=None,
                        help="Plot single pool (prefix match)")
    parser.add_argument("--all-pools", action="store_true")
    parser.add_argument("--artifact-dir", default="results/mm_noise")
    parser.add_argument("--output-dir", default="results/mm_noise/plots")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Plot top N pools by sample count")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    print("Loading model...")
    params, meta = load_model(args.artifact_dir)

    print("Loading data...")
    with open(os.path.join(CACHE_DIR, "stage1.pkl"), "rb") as f:
        stage1 = pickle.load(f)
    matched_clean = stage1["matched_clean"]
    option_c_clean = stage1["option_c_clean"]

    print("Computing decomposition...")
    decomp = compute_decomposition(params, meta, matched_clean, option_c_clean)

    pool_ids = decomp["pool_ids"]
    pool_idx = decomp["pool_idx"]

    # Which pools to plot
    if args.pool:
        targets = [i for i, pid in enumerate(pool_ids)
                   if pid.startswith(args.pool)]
    elif args.all_pools:
        targets = list(range(len(pool_ids)))
    else:
        # Top N by sample count
        counts = [(pool_idx == i).sum() for i in range(len(pool_ids))]
        targets = sorted(range(len(pool_ids)), key=lambda i: -counts[i])
        targets = targets[:args.top_n]

    # Per-pool time series
    print(f"\nPlotting {len(targets)} pools...")
    for i in targets:
        plot_pool_timeseries(decomp, params, i, args.output_dir)

    # TVL response curves
    print("\nPlotting TVL response...")
    plot_tvl_response(params, meta, decomp, args.output_dir)

    # K distribution
    print("\nPlotting K distribution...")
    plot_K_distribution(params, meta, decomp, args.output_dir)

    print(f"\nDone. Plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
