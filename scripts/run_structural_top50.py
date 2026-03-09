"""Fit the structural mixture model and plot predicted vs actual for top 50 pools.

Uses the cached panel (last 90 days), fits with vanilla SVI, then generates
paginated plots showing V_arb + V_noise decomposition and predicted vs actual.
"""

import json
import os
import sys
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Config ----
PANEL_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "local_data", "noise_calibration", "panel.parquet",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "structural_hierarchical",
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "structural_fit.json")
TRAIN_DAYS = 90
SVI_STEPS = 20_000
SVI_LR = 1e-3
NUM_SAMPLES = 1000
SEED = 42
TOP_N = 50


def load_and_filter_panel():
    """Load cached panel, filter to 90 days, keep pools with >= 10 obs."""
    panel = pd.read_parquet(PANEL_CACHE)
    max_date = panel["date"].max()
    if not isinstance(max_date, date):
        max_date = pd.Timestamp(max_date).date()
    cutoff = max_date - timedelta(days=TRAIN_DAYS)
    panel = panel[
        panel["date"].apply(
            lambda d: d >= cutoff if isinstance(d, date)
            else pd.Timestamp(d).date() >= cutoff
        )
    ].copy()

    if "log_tvl_lag1" not in panel.columns:
        panel = panel.sort_values(["pool_id", "date"]).reset_index(drop=True)
        panel["log_tvl_lag1"] = panel.groupby("pool_id")["log_tvl"].shift(1)
        panel = panel.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)

    pool_counts = panel.groupby("pool_id").size()
    valid = pool_counts[pool_counts >= 10].index
    panel = panel[panel["pool_id"].isin(valid)].copy()

    print(f"Panel: {len(panel)} obs, {panel['pool_id'].nunique()} pools, "
          f"{cutoff} to {max_date}")
    return panel


def fit_structural(panel):
    """Run SVI on the structural mixture model."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from quantammsim.noise_calibration.covariate_encoding import (
        encode_covariates_structural,
    )
    from quantammsim.noise_calibration.model import structural_noise_model
    from quantammsim.noise_calibration.inference import run_svi
    from quantammsim.noise_calibration.postprocessing import (
        check_convergence, extract_structural_params,
    )
    from quantammsim.noise_calibration.output import generate_output_json

    import numpyro
    numpyro.enable_x64()

    # Load gas costs for mainnet from CSV if available
    gas_csv = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "formula_vs_real", "mainnet_gas_cost_daily.csv",
    )
    gas_arr = None
    if os.path.exists(gas_csv):
        gas_df = pd.read_csv(gas_csv)
        # CSV has columns: unix (ms timestamp), USD (gas cost)
        gas_df["date"] = pd.to_datetime(gas_df["unix"], unit="ms").dt.date
        gas_lookup = dict(zip(gas_df["date"], gas_df["USD"]))

        # Build per-observation gas array
        gas_vals = []
        for _, row in panel.iterrows():
            d = row["date"]
            if not isinstance(d, date):
                d = pd.Timestamp(d).date()
            chain = row["chain"]
            if chain == "MAINNET" and d in gas_lookup:
                gas_vals.append(gas_lookup[d])
            elif chain == "MAINNET":
                gas_vals.append(1.0)  # median fallback
            else:
                # L2 chains: ~$0.005
                from quantammsim.noise_calibration.constants import GAS_COSTS
                gas_vals.append(GAS_COSTS.get(chain, 0.005))
        gas_arr = np.array(gas_vals, dtype=np.float64)
        print(f"Gas costs: loaded ({len(gas_lookup)} mainnet days from CSV)")
    else:
        print("Gas costs: using defaults (no mainnet CSV)")

    data = encode_covariates_structural(panel, gas=gas_arr)

    print(f"\nFitting structural model: {SVI_STEPS} SVI steps, lr={SVI_LR}")
    samples, elbo_losses = run_svi(
        data,
        num_steps=SVI_STEPS,
        lr=SVI_LR,
        seed=SEED,
        num_samples=NUM_SAMPLES,
        model_fn=structural_noise_model,
    )
    convergence = check_convergence(elbo_losses, method="svi")

    pool_params = extract_structural_params(samples, data)

    # Save output JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inference_config = {
        "method": "svi", "svi_steps": SVI_STEPS,
        "svi_lr": SVI_LR, "num_samples": NUM_SAMPLES,
    }
    generate_output_json(
        pool_params, samples, data, convergence,
        OUTPUT_JSON, inference_config,
    )

    return samples, data, pool_params, elbo_losses


def compute_predictions(samples, data, panel):
    """Compute per-observation predicted V_arb and V_noise."""
    from quantammsim.noise_calibration.formula_arb import (
        formula_arb_volume_daily_jax,
    )
    import jax.numpy as jnp

    sample_dict = samples
    agg_fn = np.median

    # Cadence parameters
    alpha_0 = agg_fn(np.array(sample_dict["alpha_0"]))
    alpha_chain = agg_fn(np.array(sample_dict["alpha_chain"]), axis=0)
    alpha_tier = agg_fn(np.array(sample_dict["alpha_tier"]), axis=0)
    alpha_tvl = agg_fn(np.array(sample_dict["alpha_tvl"]))

    # Hierarchical noise: reconstruct theta
    B = agg_fn(np.array(sample_dict["B"]), axis=0)
    eta = agg_fn(np.array(sample_dict["eta"]), axis=0)
    sigma_theta = agg_fn(np.array(sample_dict["sigma_theta"]), axis=0)
    L_Omega = agg_fn(np.array(sample_dict["L_Omega"]), axis=0)

    pool_idx = np.array(data["pool_idx"])
    X_pool = np.array(data["X_pool"])
    x_obs = np.array(data["x_obs"])
    chain_idx = np.array(data["chain_idx"])
    tier_idx = np.array(data["tier_idx"])
    sigma_daily = np.array(data["sigma_daily"])
    lag_log_tvl = np.array(data["lag_log_tvl"])
    fee = np.array(data["fee"])
    gas = np.array(data["gas"])

    # Per-pool cadence
    padded_chain = np.concatenate([[0.0], alpha_chain])
    padded_tier = np.concatenate([[0.0], alpha_tier])

    N_pools = data["N_pools"]
    pool_log_cadence = np.zeros(N_pools)
    for p in range(N_pools):
        pool_log_cadence[p] = (
            alpha_0
            + padded_chain[chain_idx[p]]
            + padded_tier[tier_idx[p]]
            + alpha_tvl * np.median(lag_log_tvl[pool_idx == p])
        )

    # Per-obs V_arb
    log_cad_obs = pool_log_cadence[pool_idx]
    cadence_obs = np.exp(np.clip(log_cad_obs, -2.0, 6.0))
    tvl_obs = np.exp(lag_log_tvl)

    V_arb = np.array(formula_arb_volume_daily_jax(
        jnp.array(sigma_daily), jnp.array(tvl_obs),
        jnp.array(fee), jnp.array(gas), jnp.array(cadence_obs),
    ))

    # Per-pool theta from hierarchical model
    L_Sigma = np.diag(sigma_theta) @ L_Omega
    theta = X_pool @ B.T + eta @ L_Sigma.T  # (N_pools, K_obs_coeff)

    # Per-obs V_noise
    log_V_noise = np.sum(theta[pool_idx] * x_obs, axis=1)
    V_noise = np.exp(log_V_noise)

    # Predicted total
    V_total_pred = V_arb + V_noise
    log_V_pred = np.log(np.maximum(V_total_pred, 1e-6))

    return V_arb, V_noise, V_total_pred, log_V_pred, cadence_obs


def plot_top50(panel, data, pool_params, V_arb, V_noise, log_V_pred):
    """Plot top 50 pools by median TVL."""
    pool_meta = data["pool_meta"]
    pool_ids = data["pool_ids"]
    pool_idx = np.array(data["pool_idx"])
    y_obs = np.array(data["y_obs"])

    # Rank pools by median TVL
    pool_tvl = {}
    for i, pid in enumerate(pool_ids):
        mask = pool_idx == i
        pool_tvl[pid] = np.median(np.exp(np.array(data["lag_log_tvl"])[mask]))

    ranked = sorted(pool_tvl.items(), key=lambda x: -x[1])[:TOP_N]

    # Build param lookup
    param_lookup = {p["pool_id"]: p for p in pool_params}

    per_page = 10
    n_pages = (len(ranked) + per_page - 1) // per_page

    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, len(ranked))
        page_pools = ranked[start:end]
        n_this = len(page_pools)

        ncols = 2
        nrows = (n_this + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)

        for idx, (pid, median_tvl) in enumerate(page_pools):
            ax = axes[idx // ncols][idx % ncols]
            p_idx = pool_ids.index(pid)
            mask = pool_idx == p_idx

            pp = panel[panel["pool_id"] == pid].sort_values("date")
            dates = pd.to_datetime(pp["date"].values)
            actual_vol = np.exp(y_obs[mask])
            pred_arb = V_arb[mask]
            pred_noise = V_noise[mask]
            pred_total = pred_arb + pred_noise

            # R2
            actual_log = y_obs[mask]
            pred_log = log_V_pred[mask]
            ss_res = np.sum((actual_log - pred_log) ** 2)
            ss_tot = np.sum((actual_log - actual_log.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            # Arb fraction
            arb_frac = np.median(pred_arb / np.maximum(pred_total, 1.0))

            # Plot
            ax.fill_between(dates, 0, pred_arb, alpha=0.3, color="orangered",
                            label="V_arb (LVR)")
            ax.fill_between(dates, pred_arb, pred_total, alpha=0.3,
                            color="steelblue", label="V_noise (hier.)")
            ax.plot(dates, actual_vol, "k-", linewidth=0.8, alpha=0.7,
                    label="Actual")
            ax.plot(dates, pred_total, "--", color="purple", linewidth=0.8,
                    alpha=0.7, label="Predicted total")

            ax.set_yscale("log")
            ax.set_ylabel("Daily volume (USD)", fontsize=8)

            meta = pool_meta[pool_meta["pool_id"] == pid]
            if len(meta) > 0:
                m = meta.iloc[0]
                tokens = m["tokens"]
                if isinstance(tokens, str):
                    tokens = tokens.split(",")
                tok_str = "/".join(str(t)[:8] for t in tokens[:2])
                chain = str(m["chain"])
            else:
                tok_str = pid[:16]
                chain = "?"

            params = param_lookup.get(pid, {})
            arb_freq = params.get("arb_frequency", "?")

            ax.set_title(
                f"{tok_str} ({chain})\n"
                f"TVL ${median_tvl:,.0f}  |  R\u00b2={r2:.3f}  "
                f"arb_freq={arb_freq}min  arb_frac={arb_frac:.1%}  "
                f"n={mask.sum()}",
                fontsize=8,
            )
            ax.legend(fontsize=6, loc="upper right")
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="x", rotation=30)

        for idx in range(n_this, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            f"Structural mixture model: V_arb + V_noise decomposition "
            f"— page {page + 1}/{n_pages} "
            f"(top {TOP_N} by median TVL, 90d window)",
            fontsize=11,
        )
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"structural_top50_page{page + 1}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_elbo(elbo_losses):
    """Plot ELBO convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(elbo_losses, alpha=0.3, color="steelblue", linewidth=0.5)
    window = min(100, len(elbo_losses) // 10)
    if window > 1:
        smoothed = pd.Series(elbo_losses).rolling(window).mean().values
        ax.plot(smoothed, color="red", linewidth=1.5, label=f"Rolling {window}")
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("ELBO loss")
    ax.set_title("ELBO convergence")

    ax = axes[1]
    start = len(elbo_losses) * 4 // 5
    ax.plot(range(start, len(elbo_losses)), elbo_losses[start:],
            color="steelblue", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("ELBO loss")
    ax.set_title("ELBO convergence (last 20%)")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "elbo_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_summary(data, pool_params, V_arb, V_noise, log_V_pred):
    """Summary plots: arb frequency distribution, arb fraction, R2."""
    pool_idx = np.array(data["pool_idx"])
    y_obs = np.array(data["y_obs"])
    pool_ids = data["pool_ids"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Arb frequency histogram
    ax = axes[0]
    freqs = [p["arb_frequency"] for p in pool_params]
    ax.hist(freqs, bins=range(0, 62, 2), color="orangered", alpha=0.7,
            edgecolor="white")
    ax.set_xlabel("Arb frequency (minutes)")
    ax.set_ylabel("Count")
    ax.set_title(f"Arb frequency distribution (n={len(freqs)})")
    ax.axvline(np.median(freqs), color="black", linestyle="--",
               label=f"Median={np.median(freqs):.0f}min")
    ax.legend()

    # 2. Arb fraction per pool
    ax = axes[1]
    arb_fracs = []
    for i, pid in enumerate(pool_ids):
        mask = pool_idx == i
        total = V_arb[mask] + V_noise[mask]
        arb_fracs.append(np.median(V_arb[mask] / np.maximum(total, 1.0)))
    ax.hist(arb_fracs, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Median arb fraction")
    ax.set_ylabel("Count")
    ax.set_title("Arb fraction distribution")
    ax.axvline(np.median(arb_fracs), color="black", linestyle="--",
               label=f"Median={np.median(arb_fracs):.2f}")
    ax.legend()

    # 3. Per-pool R2
    ax = axes[2]
    r2_vals = []
    for i, pid in enumerate(pool_ids):
        mask = pool_idx == i
        actual = y_obs[mask]
        pred = log_V_pred[mask]
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2_vals.append(1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    r2_vals = np.array(r2_vals)
    ax.hist(r2_vals[np.isfinite(r2_vals)], bins=30, color="green", alpha=0.7,
            edgecolor="white")
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("Per-pool R² distribution")
    ax.axvline(np.nanmedian(r2_vals), color="black", linestyle="--",
               label=f"Median={np.nanmedian(r2_vals):.3f}")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "structural_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("=" * 70)
    print("Structural Mixture Model: Fit + Top 50 Plots")
    print("=" * 70)

    panel = load_and_filter_panel()
    samples, data, pool_params, elbo_losses = fit_structural(panel)

    print("\nComputing predictions...")
    V_arb, V_noise, V_total, log_V_pred, cadence = compute_predictions(
        samples, data, panel,
    )
    print(f"  V_arb median: ${np.median(V_arb):,.0f}")
    print(f"  V_noise median: ${np.median(V_noise):,.0f}")
    print(f"  Arb fraction (median pool): {np.median(V_arb / np.maximum(V_total, 1)):.2%}")

    print("\nGenerating plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_elbo(elbo_losses)
    plot_summary(data, pool_params, V_arb, V_noise, log_V_pred)
    plot_top50(panel, data, pool_params, V_arb, V_noise, log_V_pred)

    # Summary stats
    print(f"\n{'=' * 70}")
    print(f"Done. Output in: {OUTPUT_DIR}")
    arb_freqs = [p["arb_frequency"] for p in pool_params]
    print(f"  Arb frequency: median={np.median(arb_freqs):.0f}min, "
          f"range=[{np.min(arb_freqs)}, {np.max(arb_freqs)}]")
    print(f"  JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
