"""Validate the noise model's TVL counterfactual predictions.

Uses the AAVE/WETH reClAMM pool's natural experiment (70x TVL increase
from LP deposit in Jan 2026) to check whether the model's combined
V_arb + V_noise prediction matches observed volume changes.

Tests whether the full model (PCHIP arb grid + per-pool linear noise
with b_tvl on standardized features) produces the right total volume
response, even though the noise-specific elasticity (~0.42 raw) is
lower than the event study's total elasticity (~0.9).

Also evaluates counterfactual noise volumes at specified TVL levels.

Usage:
  python experiments/validate_tvl_counterfactual.py
  python experiments/validate_tvl_counterfactual.py --pool 0x9d1fcf346ea1b0
  python experiments/validate_tvl_counterfactual.py --counterfactual-tvl 1e6 5e6 20e6 50e6
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="0x9d1fcf346ea1b0",
                        help="Pool ID prefix")
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR)
    parser.add_argument("--pre-cutoff", default="2026-01-10",
                        help="Date before which = pre-deposit")
    parser.add_argument("--post-cutoff", default="2026-01-20",
                        help="Date after which = post-deposit")
    parser.add_argument("--counterfactual-tvl", type=float, nargs="+",
                        default=[70_000, 500_000, 5_000_000, 20_000_000],
                        help="TVL values for counterfactual evaluation")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    # ---- Load model artifact ----
    art, meta = load_artifact(args.artifact_dir)
    nc = art["noise_coeffs"]
    log_cad = art["log_cadence"]
    x_mean = art["x_mean"]
    x_std = art["x_std"]
    pool_ids = meta["pool_ids"]
    feat_names = meta["feat_names"]
    per_pool = nc.ndim == 2

    # ---- Load pool data ----
    with open(os.path.join(CACHE_DIR, "stage1.pkl"), "rb") as f:
        data = pickle.load(f)
    mc = data["matched_clean"]
    oc = data["option_c_clean"]

    pid = args.pool
    if pid not in pool_ids:
        # Try prefix match
        matches = [p for p in pool_ids if p.startswith(pid) or pid.startswith(p)]
        if matches:
            pid = matches[0]
        else:
            print(f"Pool {args.pool} not found in calibration set")
            return

    idx = pool_ids.index(pid)
    coeffs = nc[idx] if per_pool else nc
    cadence = float(np.exp(log_cad[idx]))
    gas = float(np.exp(oc[pid]["log_gas"]))
    tvl_col = feat_names.index("xobs_1")

    print("=" * 70)
    print("TVL Counterfactual Validation")
    print(f"  Pool: {pid} ({mc[pid]['tokens']}, {mc[pid]['chain']})")
    print(f"  Learned cadence: {cadence:.1f} min")
    print(f"  Gas: ${gas:.2f}")
    print(f"  b_tvl (standardized): {coeffs[tvl_col]:.4f}")
    print(f"  TVL standardization: mean={x_mean[tvl_col]:.2f},"
          f" std={x_std[tvl_col]:.2f}")
    print(f"  Raw noise elasticity: {coeffs[tvl_col]/x_std[tvl_col]:.4f}")
    print("=" * 70)

    # ---- V_arb from PCHIP ----
    entry = mc[pid]
    v_arb_all = np.array(interpolate_pool_daily(
        entry["coeffs"], jnp.float64(np.log(cadence)), jnp.float64(gas)))
    day_indices = entry["day_indices"]
    v_arb = v_arb_all[day_indices]

    panel = entry["panel"]
    log_vol = panel["log_volume"].values.astype(float)
    log_tvl = panel["log_tvl_lag1"].values.astype(float)
    vol_obs = np.exp(log_vol)
    tvl = np.exp(log_tvl)
    dates = pd.to_datetime(panel["date"])

    pre_mask = dates < args.pre_cutoff
    post_mask = dates >= args.post_cutoff

    # ---- Build full feature vectors ----
    from experiments.run_linear_market_noise import build_data
    data_full = build_data(mc, oc, trend_windows=(7,),
                           include_market=True, include_cross_pool=True)
    x_full = data_full["x"]
    pool_idx_full = data_full["pool_idx"]
    day_idx_full = data_full["day_idx"]

    pool_i = pool_ids.index(pid)
    pool_mask = pool_idx_full == pool_i

    all_dates = set()
    for p in pool_ids:
        all_dates.update(mc[p]["panel"]["date"].values)
    date_list = sorted(all_dates)

    sample_dates = np.array([pd.Timestamp(date_list[d])
                             for d in day_idx_full[pool_mask]])
    sample_x = x_full[pool_mask]
    sgd = data_full["sample_grid_days"][pool_mask]
    v_arb_samples = v_arb_all[sgd]

    # Per-sample noise prediction
    if per_pool:
        log_v_noise = sample_x @ coeffs
    else:
        log_v_noise = sample_x @ coeffs
    v_noise = np.exp(log_v_noise)

    sample_pre = sample_dates < pd.Timestamp(args.pre_cutoff)
    sample_post = sample_dates >= pd.Timestamp(args.post_cutoff)

    # ---- Pre/post comparison ----
    print(f"\n=== Pre-deposit (before {args.pre_cutoff}) ===")
    print(f"  Median TVL:         ${np.median(tvl[pre_mask]):>14,.0f}")
    print(f"  Median V_obs:       ${np.median(vol_obs[pre_mask]):>14,.0f}")
    print(f"  Median V_arb:       ${np.median(v_arb[pre_mask]):>14,.0f}  (PCHIP)")
    print(f"  Median V_noise:     ${np.median(v_noise[sample_pre]):>14,.0f}  (model)")
    v_total_pre = v_arb_samples[sample_pre] + v_noise[sample_pre]
    print(f"  Median V_total:     ${np.median(v_total_pre):>14,.0f}  (V_arb + V_noise)")

    print(f"\n=== Post-deposit (after {args.post_cutoff}) ===")
    print(f"  Median TVL:         ${np.median(tvl[post_mask]):>14,.0f}")
    print(f"  Median V_obs:       ${np.median(vol_obs[post_mask]):>14,.0f}")
    print(f"  Median V_arb:       ${np.median(v_arb[post_mask]):>14,.0f}  (PCHIP)")
    print(f"  Median V_noise:     ${np.median(v_noise[sample_post]):>14,.0f}  (model)")
    v_total_post = v_arb_samples[sample_post] + v_noise[sample_post]
    print(f"  Median V_total:     ${np.median(v_total_post):>14,.0f}  (V_arb + V_noise)")

    # ---- Ratios ----
    tvl_ratio = np.median(tvl[post_mask]) / np.median(tvl[pre_mask])
    vol_ratio = np.median(vol_obs[post_mask]) / np.median(vol_obs[pre_mask])
    varb_ratio = np.median(v_arb[post_mask]) / np.median(v_arb[pre_mask])
    vnoise_ratio = np.median(v_noise[sample_post]) / np.median(v_noise[sample_pre])
    vtotal_ratio = np.median(v_total_post) / np.median(v_total_pre)

    print(f"\n=== Ratios (post / pre) ===")
    print(f"  TVL:       {tvl_ratio:>8.1f}x")
    print(f"  V_obs:     {vol_ratio:>8.1f}x  (ground truth)")
    print(f"  V_arb:     {varb_ratio:>8.1f}x  (PCHIP grid)")
    print(f"  V_noise:   {vnoise_ratio:>8.1f}x  (noise model)")
    print(f"  V_total:   {vtotal_ratio:>8.1f}x  (V_arb + V_noise)")
    print(f"  Gap:       {vtotal_ratio/vol_ratio:>8.2f}x  (pred/obs)")

    # ---- Decomposition shares ----
    print(f"\n=== Decomposition shares ===")
    arb_share_pre = np.median(v_arb[pre_mask]) / np.median(vol_obs[pre_mask]) * 100
    noise_share_pre = np.median(v_noise[sample_pre]) / np.median(vol_obs[pre_mask]) * 100
    arb_share_post = np.median(v_arb[post_mask]) / np.median(vol_obs[post_mask]) * 100
    noise_share_post = np.median(v_noise[sample_post]) / np.median(vol_obs[post_mask]) * 100

    print(f"  Pre:  arb={arb_share_pre:.0f}%  noise={noise_share_pre:.0f}%")
    print(f"  Post: arb={arb_share_post:.0f}%  noise={noise_share_post:.0f}%")

    # ---- Counterfactual evaluation ----
    print(f"\n=== Counterfactual noise volumes ===")
    print(f"  (Using median pre-deposit market features, varying TVL only)")
    print(f"  {'TVL':>14s}  {'V_noise/day':>12s}  {'V_noise/min':>12s}"
          f"  {'Ratio vs 70K':>12s}")
    print(f"  {'-'*55}")

    x_base = np.median(sample_x[sample_pre], axis=0).copy()
    baseline_tvl = 70_000
    x_baseline = x_base.copy()
    x_baseline[tvl_col] = (np.log(baseline_tvl) - x_mean[tvl_col]) / x_std[tvl_col]
    for i, name in enumerate(feat_names):
        if name.startswith("xobs_1" + "\u00d7"):
            paired_name = name.split("\u00d7")[1]
            if paired_name in feat_names:
                paired_idx = feat_names.index(paired_name)
                x_baseline[i] = x_baseline[tvl_col] * x_base[paired_idx]
    vn_baseline = np.exp(x_baseline @ coeffs)

    for cf_tvl in args.counterfactual_tvl:
        x_cf = x_base.copy()
        std_log_tvl = (np.log(cf_tvl) - x_mean[tvl_col]) / x_std[tvl_col]
        x_cf[tvl_col] = std_log_tvl
        for i, name in enumerate(feat_names):
            if name.startswith("xobs_1" + "\u00d7"):
                paired_name = name.split("\u00d7")[1]
                if paired_name in feat_names:
                    paired_idx = feat_names.index(paired_name)
                    x_cf[i] = std_log_tvl * x_base[paired_idx]

        vn = np.exp(x_cf @ coeffs)
        ratio = vn / vn_baseline
        print(f"  ${cf_tvl:>13,.0f}  ${vn:>11,.0f}  ${vn/1440:>11,.0f}"
              f"  {ratio:>11.1f}x")

    print(f"\n  Key finding: model predicts {vtotal_ratio:.1f}x total volume"
          f" increase vs {vol_ratio:.1f}x observed ({vtotal_ratio/vol_ratio:.0%} accuracy).")
    print(f"  V_arb ({varb_ratio:.0f}x) carries most of the response;"
          f" V_noise ({vnoise_ratio:.1f}x) is secondary but adds up.")


if __name__ == "__main__":
    main()
