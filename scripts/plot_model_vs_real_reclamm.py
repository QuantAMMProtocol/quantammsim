"""Plot full model (V_arb + V_noise) vs real observed volume for a pool.

Uses the pool's actual historical TVL path, evaluates V_arb from the PCHIP
grid at the learned cadence, and V_noise from the per-pool linear model.
Compares against observed total volume.

Usage:
  python scripts/plot_model_vs_real_reclamm.py
  python scripts/plot_model_vs_real_reclamm.py --pool 0x3de27efa2f1aa6
  python scripts/plot_model_vs_real_reclamm.py --pool 0x9d1fcf346ea1b0
"""

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import jax.numpy as jnp
from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
from quantammsim.calibration.noise_model_arrays import load_artifact


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)
ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "linear_market_noise",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "model_vs_real",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="0x9d1fcf346ea1b0",
                        help="Pool ID prefix")
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
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
    tvl = np.exp(panel["log_tvl_lag1"].values.astype(float))

    # Load noise model
    art, meta = load_artifact(args.artifact_dir)
    pool_ids = meta["pool_ids"]
    idx = pool_ids.index(pid)
    coeffs = art["noise_coeffs"][idx]
    cadence = float(np.exp(art["log_cadence"][idx]))
    gas = float(np.exp(oc[pid]["log_gas"]))

    print(f"Pool: {pid} ({entry['tokens']}, {entry['chain']})")
    print(f"Cadence: {cadence:.1f} min, Gas: ${gas}")
    print(f"{len(dates)} days: {dates.min().date()} → {dates.max().date()}")
    print(f"TVL: ${tvl.min():,.0f} – ${tvl.max():,.0f}")

    # V_arb from PCHIP
    v_arb_all = np.array(interpolate_pool_daily(
        entry["coeffs"], jnp.float64(np.log(cadence)), jnp.float64(gas)))

    # V_noise from model at actual TVL
    from experiments.run_linear_market_noise import build_data
    data_full = build_data(mc, oc, trend_windows=(7,),
                           include_market=True, include_cross_pool=False)
    x_full = data_full["x"]
    pool_idx_full = data_full["pool_idx"]
    pool_mask = pool_idx_full == idx
    sample_x = x_full[pool_mask]
    sgd = data_full["sample_grid_days"][pool_mask]
    day_idx = data_full["day_idx"][pool_mask]

    log_v_noise = sample_x @ coeffs
    v_noise = np.exp(log_v_noise)
    v_arb_samples = v_arb_all[sgd]
    v_total_pred = v_arb_samples + v_noise

    # Align dates
    all_dates = set()
    for p in pool_ids:
        all_dates.update(mc[p]["panel"]["date"].values)
    date_list = sorted(all_dates)
    sample_dates = np.array([pd.Timestamp(date_list[d]) for d in day_idx])

    # Match TVL and obs volume
    tvl_samples = np.zeros(len(sample_dates))
    vol_obs_samples = np.zeros(len(sample_dates))
    for i, sd in enumerate(sample_dates):
        matches = np.where(dates == sd)[0]
        if len(matches) > 0:
            tvl_samples[i] = tvl[matches[0]]
            vol_obs_samples[i] = vol_obs[matches[0]]

    valid = tvl_samples > 100
    sd = sample_dates[valid]
    vo = vol_obs_samples[valid]
    va = v_arb_samples[valid]
    vn = v_noise[valid]
    vt = v_total_pred[valid]
    tv = tvl_samples[valid]

    # R²
    log_obs = np.log(np.maximum(vo, 1))
    log_pred = np.log(np.maximum(vt, 1))
    ss_res = np.sum((log_obs - log_pred) ** 2)
    ss_tot = np.sum((log_obs - log_obs.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    print(f"\nR² (log): {r2:.3f}")
    print(f"Median obs: ${np.median(vo):,.0f}, pred: ${np.median(vt):,.0f}")
    print(f"Median V_arb: ${np.median(va):,.0f}, V_noise: ${np.median(vn):,.0f}")

    # Fee rate
    fee_rate = float(panel["swap_fee"].iloc[0]) if "swap_fee" in panel.columns else 0.003

    # Plot
    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)

    # 1. TVL
    ax = axes[0]
    ax.plot(sd, tv, "b-", linewidth=1)
    ax.set_ylabel("TVL (USD)")
    ax.set_yscale("log")
    ax.set_title(f"{entry['tokens']} ({entry['chain']}) — "
                 f"Model (V_arb + V_noise) vs Observed  "
                 f"[R\u00b2={r2:.3f}, cadence={cadence:.0f}min, fee={fee_rate:.4f}]")
    ax.grid(True, alpha=0.3)

    # 2. Volume: stacked arb + noise vs observed
    ax = axes[1]
    ax.fill_between(sd, 0, va, alpha=0.3, color="steelblue", label="V_arb (PCHIP)")
    ax.fill_between(sd, va, va + vn, alpha=0.3, color="coral", label="V_noise (model)")
    ax.plot(sd, vo, "k-", linewidth=0.8, alpha=0.7, label="V_obs (actual)")
    ax.plot(sd, vt, "r--", linewidth=0.8, alpha=0.5, label="V_pred = V_arb + V_noise")
    ax.set_ylabel("Volume (USD/day)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. V_noise only
    ax = axes[2]
    ax.fill_between(sd, 0, vn, alpha=0.4, color="coral")
    ax.plot(sd, vn, "r-", linewidth=0.8, alpha=0.7, label="V_noise (model)")
    ax.axhline(np.median(vn), color="red", linestyle="--", alpha=0.5,
               label=f"median: ${np.median(vn):,.0f}")
    ax.set_ylabel("Noise volume (USD/day)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Fee revenue: observed vs predicted
    ax = axes[3]
    fee_obs = vo * fee_rate
    fee_pred = vt * fee_rate
    fee_noise_only = vn * fee_rate
    fee_arb = va * fee_rate
    ax.fill_between(sd, 0, fee_arb, alpha=0.3, color="steelblue", label="Arb fees")
    ax.fill_between(sd, fee_arb, fee_pred, alpha=0.3, color="coral", label="Noise fees")
    ax.plot(sd, fee_obs, "k-", linewidth=0.8, alpha=0.7, label="Observed fees")
    ax.plot(sd, fee_pred, "r--", linewidth=0.8, alpha=0.5, label="Predicted total fees")
    ax.plot(sd, fee_noise_only, "m-", linewidth=0.8, alpha=0.6,
            label=f"Noise fees only (med=${np.median(fee_noise_only):,.0f})")
    ax.set_ylabel("Fee revenue (USD/day)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Vol/TVL
    ax = axes[4]
    vol_tvl_obs = vo / tv * 100
    vol_tvl_pred = vt / tv * 100
    ax.plot(sd, vol_tvl_obs, "k-", linewidth=0.8, alpha=0.7, label="Observed")
    ax.plot(sd, vol_tvl_pred, "r--", linewidth=0.8, alpha=0.5, label="Predicted")
    ax.axhline(np.median(vol_tvl_obs), color="black", linestyle=":",
               alpha=0.3, label=f"obs median: {np.median(vol_tvl_obs):.1f}%")
    ax.axhline(np.median(vol_tvl_pred), color="red", linestyle=":",
               alpha=0.3, label=f"pred median: {np.median(vol_tvl_pred):.1f}%")
    ax.set_ylabel("Vol / TVL (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, min(np.percentile(vol_tvl_obs, 95) * 2, 200))

    # 6. Pred/Obs ratio
    ax = axes[5]
    ratio = vt / np.maximum(vo, 1)
    ax.plot(sd, ratio, "g-", linewidth=0.8, alpha=0.7)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="perfect")
    ax.axhline(np.median(ratio), color="red", linestyle="--", alpha=0.5,
               label=f"median: {np.median(ratio):.2f}")
    ax.set_ylabel("Pred / Obs")
    ax.set_xlabel("Date")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.01, 100)

    fig.tight_layout()
    out = os.path.join(args.output_dir, f"{pid[:16]}_model_vs_real.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Fee summary
    fee_obs_total = np.sum(vo * fee_rate)
    fee_pred_total = np.sum(vt * fee_rate)
    fee_noise_total = np.sum(vn * fee_rate)
    fee_arb_total = np.sum(va * fee_rate)
    print(f"\nFee revenue (cumulative, fee={fee_rate:.4f}):")
    print(f"  Observed:  ${fee_obs_total:,.0f}")
    print(f"  Predicted: ${fee_pred_total:,.0f}"
          f" (arb: ${fee_arb_total:,.0f}, noise: ${fee_noise_total:,.0f})")

    # Pre/post deposit stats (for reClAMM AAVE/ETH)
    pre = sd < pd.Timestamp("2026-01-10")
    post = sd >= pd.Timestamp("2026-01-20")
    if pre.sum() > 5 and post.sum() > 5:
        print(f"\nPre-deposit (before Jan 10):")
        print(f"  TVL: ${np.median(tv[pre]):,.0f}")
        print(f"  V_obs: ${np.median(vo[pre]):,.0f},"
              f" V_pred: ${np.median(vt[pre]):,.0f}")
        print(f"  V_arb: ${np.median(va[pre]):,.0f},"
              f" V_noise: ${np.median(vn[pre]):,.0f}")
        print(f"  Fees obs: ${np.median(vo[pre])*fee_rate:,.0f}/day,"
              f" pred: ${np.median(vt[pre])*fee_rate:,.0f}/day")
        print(f"  Pred/Obs: {np.median(vt[pre] / vo[pre]):.2f}")
        print(f"Post-deposit (after Jan 20):")
        print(f"  TVL: ${np.median(tv[post]):,.0f}")
        print(f"  V_obs: ${np.median(vo[post]):,.0f},"
              f" V_pred: ${np.median(vt[post]):,.0f}")
        print(f"  V_arb: ${np.median(va[post]):,.0f},"
              f" V_noise: ${np.median(vn[post]):,.0f}")
        print(f"  Fees obs: ${np.median(vo[post])*fee_rate:,.0f}/day,"
              f" pred: ${np.median(vt[post])*fee_rate:,.0f}/day")
        print(f"  Pred/Obs: {np.median(vt[post] / vo[post]):.2f}")


if __name__ == "__main__":
    main()
