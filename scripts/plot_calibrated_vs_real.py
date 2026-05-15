"""Plot predicted vs real volume using saved per-pool calibrated noise model.

Loads the artifact from experiments/run_linear_market_noise.py (--per-pool),
rebuilds the features, evaluates V_arb(learned cadence) + V_noise(x @ coeffs_i),
and generates stacked area plots showing the arb/noise decomposition per pool.

Usage:
  # First train and save:
  python experiments/run_linear_market_noise.py --per-pool --no-split --epochs 2000

  # Then plot:
  python scripts/plot_calibrated_vs_real.py
  python scripts/plot_calibrated_vs_real.py --artifact results/linear_market_noise/model.npz
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ARTIFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "linear_market_noise",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "calibrated_vs_real",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", default=os.path.join(ARTIFACT_DIR, "model.npz"))
    parser.add_argument("--meta", default=os.path.join(ARTIFACT_DIR, "meta.json"))
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load artifact ----
    print(f"Loading artifact: {args.artifact}")
    art = np.load(args.artifact, allow_pickle=True)
    noise_coeffs = art["noise_coeffs"]
    log_cadence = art["log_cadence"]
    init_log_cadences = art["init_log_cadences"]

    with open(args.meta) as f:
        meta = json.load(f)
    feat_names = meta["feat_names"]
    pool_ids = meta["pool_ids"]
    n_pools = meta["n_pools"]
    hparams = meta["hparams"]
    per_pool = noise_coeffs.ndim == 2

    print(f"  {n_pools} pools, {len(feat_names)} features, per_pool={per_pool}")
    print(f"  hparams: {hparams}")

    # ---- Rebuild data (features only, no training) ----
    import jax.numpy as jnp
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from experiments.run_linear_market_noise import load_stage1, build_data

    matched_clean, option_c_clean = load_stage1()

    print("\nRebuilding features...")
    t0 = time.time()
    data = build_data(
        matched_clean, option_c_clean,
        trend_windows=tuple(hparams["trend_windows"]),
        include_market=True, include_cross_pool=True,
    )
    print(f"  {len(data['pool_idx'])} samples, {time.time() - t0:.1f}s")

    x = data["x"]
    y_total = data["y_total"]
    pool_idx = data["pool_idx"]
    day_idx = data["day_idx"]
    sgd = data["sample_grid_days"]

    # ---- Compute predictions ----
    if per_pool:
        per_sample_coeffs = noise_coeffs[pool_idx]
        log_v_noise = np.sum(x * per_sample_coeffs, axis=1)
    else:
        log_v_noise = x @ noise_coeffs

    if "pool_intercepts" in art:
        log_v_noise = log_v_noise + art["pool_intercepts"][pool_idx]

    v_noise = np.exp(log_v_noise)

    v_arb = np.zeros(len(y_total))
    for i in range(n_pools):
        mask = pool_idx == i
        if not mask.any():
            continue
        v_arb_all = np.array(interpolate_pool_daily(
            data["pool_coeffs"][i], jnp.float64(log_cadence[i]),
            data["pool_gas"][i]))
        v_arb[mask] = v_arb_all[sgd[mask]]

    v_obs = np.exp(y_total)
    log_v_arb = np.log(np.maximum(v_arb, 1e-10))
    pred_total_log = np.logaddexp(log_v_arb, log_v_noise)

    # ---- Reconstruct dates ----
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)

    # ---- Plot: stacked area per pool ----
    per_page = 9
    n_pages = (n_pools + per_page - 1) // per_page

    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, n_pools)
        n_this = end - start

        ncols = 3
        nrows = (n_this + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)

        for idx, i in enumerate(range(start, end)):
            ax = axes[idx // ncols][idx % ncols]
            mask = pool_idx == i
            if mask.sum() < 5:
                ax.set_visible(False)
                continue

            days = day_idx[mask]
            dates = [pd.Timestamp(date_list[d]) for d in days]
            vo = v_obs[mask]
            va = v_arb[mask]
            vn = v_noise[mask]

            ax.fill_between(dates, 0, va, alpha=0.3, color="steelblue",
                            label="V_arb")
            ax.fill_between(dates, va, va + vn, alpha=0.3, color="coral",
                            label="V_noise")
            ax.plot(dates, vo, "k-", linewidth=0.8, alpha=0.7, label="V_obs")
            ax.plot(dates, va + vn, "--", color="darkred", linewidth=0.8,
                    alpha=0.7, label="V_pred")

            ax.set_yscale("log")
            ax.set_ylabel("USD/day", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.tick_params(axis="x", rotation=30)

            yt = y_total[mask]
            pt = pred_total_log[mask]
            ss_res = np.sum((yt - pt) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)

            pid = pool_ids[i]
            tokens = matched_clean[pid]["tokens"]
            chain = matched_clean[pid]["chain"]
            ci = np.exp(init_log_cadences[i])
            cl = np.exp(log_cadence[i])
            arb_share = np.median(va / vo) * 100
            noise_share = np.median(vn / vo) * 100
            b_tvl = noise_coeffs[i, 1] if per_pool else noise_coeffs[1]

            ax.set_title(
                f"{tokens} ({chain})\n"
                f"R\u00b2={r2:.3f}  cad={ci:.0f}\u2192{cl:.0f}min  "
                f"arb={arb_share:.0f}%  noise={noise_share:.0f}%  "
                f"b_tvl={b_tvl:.2f}",
                fontsize=7)
            ax.legend(fontsize=6, loc="upper right")

        for idx in range(n_this, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            f"Per-pool calibrated noise model \u2014 V_arb + V_noise "
            f"(page {page+1}/{n_pages})", fontsize=10)
        fig.tight_layout()
        out = os.path.join(args.output_dir, f"calibrated_page{page+1}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ---- Summary ----
    summary = []
    for i in range(n_pools):
        mask = pool_idx == i
        if mask.sum() < 2:
            continue
        yt = y_total[mask]
        pt = pred_total_log[mask]
        ss_res = np.sum((yt - pt) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        pid = pool_ids[i]
        va = v_arb[mask]
        vn = v_noise[mask]
        vo = v_obs[mask]

        summary.append({
            "pool_id": pid,
            "tokens": matched_clean[pid]["tokens"],
            "chain": matched_clean[pid]["chain"],
            "n_obs": int(mask.sum()),
            "R2": r2,
            "cadence_init": float(np.exp(init_log_cadences[i])),
            "cadence_learned": float(np.exp(log_cadence[i])),
            "median_arb_pct": float(np.median(va / vo) * 100),
            "median_noise_pct": float(np.median(vn / vo) * 100),
            "b_tvl": float(noise_coeffs[i, 1] if per_pool else noise_coeffs[1]),
        })

    summary_df = pd.DataFrame(summary)
    csv_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Summary: {csv_path}")
    print(f"  Median R\u00b2: {summary_df['R2'].median():.4f}")
    print(f"  Median arb: {summary_df['median_arb_pct'].median():.0f}%")
    print(f"  b_tvl: [{summary_df['b_tvl'].min():.2f},"
          f" {summary_df['b_tvl'].max():.2f}],"
          f" median={summary_df['b_tvl'].median():.2f}")


if __name__ == "__main__":
    main()
