"""Compare linear vs MLP noise models across TVL levels.

Evaluates both noise models for a given pool over the same date range,
sweeping initial TVL. Uses real price data, the PCHIP arb grid, and
both noise models to predict daily volume decomposition.

Produces a plot: predicted daily noise volume vs TVL for each model,
with the real observed volume overlaid where available.

Usage:
  python experiments/run_model_comparison.py
  python experiments/run_model_comparison.py --tvl-range 1e5 1e6 5e6 7e6 20e6 50e6
"""

import argparse
import json
import os
import pickle
import time

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
LINEAR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "linear_market_noise",
)
MLP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "mlp_noise",
)


def load_linear_model(artifact_dir, pool_id):
    """Load linear noise model for a pool."""
    art = np.load(os.path.join(artifact_dir, "model.npz"))
    with open(os.path.join(artifact_dir, "meta.json")) as f:
        meta = json.load(f)
    pool_ids = meta["pool_ids"]
    idx = next((i for i, p in enumerate(pool_ids)
                if p.startswith(pool_id) or pool_id.startswith(p)), -1)
    nc = art["noise_coeffs"]
    coeffs = nc[idx] if nc.ndim == 2 and idx >= 0 else (nc if nc.ndim == 1 else np.median(nc, axis=0))
    return {
        "coeffs": coeffs,
        "log_cadence": art["log_cadence"][idx] if idx >= 0 else np.log(10.0),
        "x_mean": art["x_mean"],
        "x_std": art["x_std"],
        "feat_names": meta["feat_names"],
        "type": "linear",
    }


def load_mlp_model(artifact_dir, pool_id):
    """Load MLP noise model."""
    art = dict(np.load(os.path.join(artifact_dir, "model.npz"), allow_pickle=True))
    with open(os.path.join(artifact_dir, "meta.json")) as f:
        meta = json.load(f)
    pool_ids = meta["pool_ids"]
    idx = next((i for i, p in enumerate(pool_ids)
                if p.startswith(pool_id) or pool_id.startswith(p)), -1)

    # Extract MLP params
    params = {}
    for k in art:
        if k.startswith("W") or k.startswith("b") or k == "log_cadence" or k == "pool_bias":
            params[k] = art[k]

    return {
        "params": params,
        "log_cadence": art["log_cadence"][idx] if idx >= 0 else np.log(10.0),
        "pool_idx": idx,
        "x_mean": art["x_mean"],
        "x_std": art["x_std"],
        "feat_names": meta["feat_names"],
        "hidden": meta["hidden"],
        "per_pool": meta.get("per_pool", False),
        "type": "mlp",
    }


def predict_noise_linear(model, x_daily, tvl_values):
    """Predict noise volume at multiple TVL levels using linear model."""
    tvl_col = 1  # xobs_1
    results = {}
    for tvl in tvl_values:
        x = x_daily.copy()
        x[:, tvl_col] = (np.log(tvl) - model["x_mean"][tvl_col]) / model["x_std"][tvl_col]
        # Update TVL interaction terms
        for i, name in enumerate(model["feat_names"]):
            if name.startswith("xobs_1\u00d7"):
                paired = name.split("\u00d7")[1]
                if paired in model["feat_names"]:
                    j = model["feat_names"].index(paired)
                    x[:, i] = x[:, tvl_col] * x_daily[:, j]
        log_noise = x @ model["coeffs"]
        results[tvl] = np.exp(log_noise)
    return results


def predict_noise_mlp(model, x_daily, tvl_values):
    """Predict noise volume at multiple TVL levels using MLP model."""
    from experiments.run_mlp_noise import forward_mlp
    tvl_col = 1
    params = model["params"]
    pool_idx_arr = (jnp.full(x_daily.shape[0], model["pool_idx"])
                    if model["per_pool"] and model["pool_idx"] >= 0 else None)
    results = {}
    for tvl in tvl_values:
        x = x_daily.copy()
        x[:, tvl_col] = (np.log(tvl) - model["x_mean"][tvl_col]) / model["x_std"][tvl_col]
        for i, name in enumerate(model["feat_names"]):
            if name.startswith("xobs_1\u00d7"):
                paired = name.split("\u00d7")[1]
                if paired in model["feat_names"]:
                    j = model["feat_names"].index(paired)
                    x[:, i] = x[:, tvl_col] * x_daily[:, j]
        log_noise = np.array(forward_mlp(params, jnp.array(x), pool_idx_arr))
        results[tvl] = np.exp(log_noise)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="0x9d1fcf346ea1b0")
    parser.add_argument("--tvl-range", type=float, nargs="+",
                        default=[100_000, 500_000, 1_000_000, 5_000_000,
                                 7_000_000, 20_000_000, 50_000_000])
    parser.add_argument("--linear-dir", default=LINEAR_DIR)
    parser.add_argument("--mlp-dir", default=MLP_DIR)
    parser.add_argument("--output-dir", default="results/model_comparison")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(os.path.join(CACHE_DIR, "stage1.pkl"), "rb") as f:
        data = pickle.load(f)
    mc = data["matched_clean"]
    oc = data["option_c_clean"]

    pid = args.pool
    entry = mc[pid]
    panel = entry["panel"]
    dates = pd.to_datetime(panel["date"])
    vol_obs = np.exp(panel["log_volume"].values.astype(float))
    tvl_obs = np.exp(panel["log_tvl_lag1"].values.astype(float))

    print(f"Pool: {pid} ({entry['tokens']}, {entry['chain']})")
    print(f"{len(dates)} days: {dates.min().date()} → {dates.max().date()}")

    # Build feature matrix (same for both models)
    from experiments.run_linear_market_noise import build_data
    data_full = build_data(mc, oc, trend_windows=(7,),
                           include_market=True, include_cross_pool=False)
    pool_ids = data_full["pool_ids"]
    pool_i = pool_ids.index(pid)
    pool_mask = data_full["pool_idx"] == pool_i
    x_pool = data_full["x"][pool_mask]
    day_idx = data_full["day_idx"][pool_mask]
    sgd = data_full["sample_grid_days"][pool_mask]

    all_dates = set()
    for p in pool_ids:
        all_dates.update(mc[p]["panel"]["date"].values)
    date_list = sorted(all_dates)
    sample_dates = np.array([pd.Timestamp(date_list[d]) for d in day_idx])

    n_days = len(sample_dates)
    print(f"Feature samples: {n_days}")

    # Load models
    print("\nLoading models...")
    linear_model = load_linear_model(args.linear_dir, pid)
    print(f"  Linear: {len(linear_model['coeffs'])} coefficients,"
          f" cadence={np.exp(linear_model['log_cadence']):.1f}min")

    has_mlp = os.path.exists(os.path.join(args.mlp_dir, "model.npz"))
    if has_mlp:
        mlp_model = load_mlp_model(args.mlp_dir, pid)
        print(f"  MLP: hidden={mlp_model['hidden']},"
              f" cadence={np.exp(mlp_model['log_cadence']):.1f}min")
    else:
        print(f"  MLP: no artifact at {args.mlp_dir}")
        mlp_model = None

    # V_arb from PCHIP (same for both — uses linear model's cadence)
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    cadence = float(np.exp(linear_model["log_cadence"]))
    gas = float(np.exp(oc[pid]["log_gas"]))
    v_arb_all = np.array(interpolate_pool_daily(
        entry["coeffs"], jnp.float64(np.log(cadence)), jnp.float64(gas)))
    v_arb = v_arb_all[sgd]

    # Predict at each TVL
    print(f"\nPredicting noise at {len(args.tvl_range)} TVL levels...")
    linear_noise = predict_noise_linear(linear_model, x_pool, args.tvl_range)
    mlp_noise = predict_noise_mlp(mlp_model, x_pool, args.tvl_range) if mlp_model else {}

    # Real observed volume for comparison
    tvl_for_samples = np.zeros(n_days)
    vol_for_samples = np.zeros(n_days)
    for i, sd in enumerate(sample_dates):
        matches = np.where(dates == sd)[0]
        if len(matches) > 0:
            tvl_for_samples[i] = tvl_obs[matches[0]]
            vol_for_samples[i] = vol_obs[matches[0]]

    # ---- Plot 1: Median noise volume vs TVL ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tvls = np.array(args.tvl_range)
    lin_medians = np.array([np.median(linear_noise[t]) for t in tvls])
    lin_q25 = np.array([np.percentile(linear_noise[t], 25) for t in tvls])
    lin_q75 = np.array([np.percentile(linear_noise[t], 75) for t in tvls])

    ax = axes[0]
    ax.fill_between(tvls / 1e6, lin_q25 / 1e6, lin_q75 / 1e6,
                    alpha=0.2, color="steelblue")
    ax.plot(tvls / 1e6, lin_medians / 1e6, "o-", color="steelblue",
            linewidth=2, label="Linear noise (median)")

    if mlp_noise:
        mlp_medians = np.array([np.median(mlp_noise[t]) for t in tvls])
        mlp_q25 = np.array([np.percentile(mlp_noise[t], 25) for t in tvls])
        mlp_q75 = np.array([np.percentile(mlp_noise[t], 75) for t in tvls])
        ax.fill_between(tvls / 1e6, mlp_q25 / 1e6, mlp_q75 / 1e6,
                        alpha=0.2, color="coral")
        ax.plot(tvls / 1e6, mlp_medians / 1e6, "s-", color="coral",
                linewidth=2, label="MLP noise (median)")

    # Add real observed volume at real TVL
    valid = tvl_for_samples > 100
    ax.scatter(tvl_for_samples[valid] / 1e6, vol_for_samples[valid] / 1e6,
               c="black", s=3, alpha=0.2, label="Observed total vol", zorder=1)

    ax.set_xlabel("Effective TVL ($M)")
    ax.set_ylabel("Daily volume ($M)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{entry['tokens']} — Noise Volume vs TVL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Noise/TVL ratio vs TVL ----
    ax = axes[1]
    ax.plot(tvls / 1e6, lin_medians / tvls * 100, "o-", color="steelblue",
            linewidth=2, label="Linear noise/TVL")
    if mlp_noise:
        ax.plot(tvls / 1e6, mlp_medians / tvls * 100, "s-", color="coral",
                linewidth=2, label="MLP noise/TVL")

    # Real vol/TVL
    ax.scatter(tvl_for_samples[valid] / 1e6,
               vol_for_samples[valid] / tvl_for_samples[valid] * 100,
               c="black", s=3, alpha=0.2, label="Observed vol/TVL")

    ax.set_xlabel("Effective TVL ($M)")
    ax.set_ylabel("Noise / TVL (%)")
    ax.set_xscale("log")
    ax.set_title("Noise as Fraction of TVL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Linear vs MLP Noise Model — {entry['tokens']}", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.output_dir, f"{pid[:16]}_model_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ---- Plot 3: Time series at selected TVLs ----
    fig, axes = plt.subplots(len(args.tvl_range), 1,
                             figsize=(14, 3 * len(args.tvl_range)),
                             sharex=True)
    if len(args.tvl_range) == 1:
        axes = [axes]

    for k, tvl in enumerate(args.tvl_range):
        ax = axes[k]
        v_total_lin = v_arb + linear_noise[tvl]
        ax.plot(sample_dates, v_total_lin / 1e6, "b-", linewidth=0.6,
                alpha=0.7, label="Linear (arb+noise)")
        if mlp_noise:
            v_total_mlp = v_arb + mlp_noise[tvl]
            ax.plot(sample_dates, v_total_mlp / 1e6, "r-", linewidth=0.6,
                    alpha=0.7, label="MLP (arb+noise)")
        ax.plot(sample_dates, vol_for_samples / 1e6, "k-", linewidth=0.5,
                alpha=0.3, label="Observed (at real TVL)")
        ax.set_ylabel(f"$M/day\nTVL=${tvl/1e6:.1f}M")
        ax.set_yscale("log")
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Volume Time Series at Different TVLs — {entry['tokens']}", fontsize=11)
    fig.tight_layout()
    out2 = os.path.join(args.output_dir, f"{pid[:16]}_tvl_sweep_timeseries.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"Summary: Median daily noise volume by TVL")
    print(f"{'='*70}")
    print(f"{'TVL':>14s}  {'Linear':>12s}  {'Lin/TVL':>8s}", end="")
    if mlp_noise:
        print(f"  {'MLP':>12s}  {'MLP/TVL':>8s}  {'MLP/Lin':>8s}")
    else:
        print()

    for tvl in args.tvl_range:
        lin = np.median(linear_noise[tvl])
        print(f"${tvl:>13,.0f}  ${lin:>11,.0f}  {lin/tvl*100:>7.1f}%", end="")
        if mlp_noise:
            mlp = np.median(mlp_noise[tvl])
            ratio = mlp / lin if lin > 0 else 0
            print(f"  ${mlp:>11,.0f}  {mlp/tvl*100:>7.1f}%  {ratio:>7.2f}x")
        else:
            print()


if __name__ == "__main__":
    main()
