#!/usr/bin/env python3
"""Price Ratio sweep for reClAMM pools.

Sweeps PR with fixed margin/shift over a single run period.
Plots RoH, fee revenue, and final value.
Optionally runs across multiple periods and produces a summary heatmap.

Usage:
    # Single period
    python scripts/run_pr_sweep.py --tokens COW ETH --start 2025-01-01 --end 2026-01-01

    # Multi-period with heatmap
    python scripts/run_pr_sweep.py --tokens COW ETH --multi-period
"""

import argparse
import json
import os

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from quantammsim.runners.jax_runners import do_run_on_historic_data
from quantammsim.pools.reCLAMM.reclamm_reserves import set_blessed_arb


BG = "#162536"
TC = "#E6CE97"

DEFAULT_PRS = [1.01, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.5, 10.0]


def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TC)
    for s in ax.spines.values():
        s.set_color(TC)
        s.set_alpha(0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color=TC)


def build_noise_arrays(pool_id, tokens, start, end):
    """Build or load cached MM noise arrays."""
    from quantammsim.calibration.noise_model_arrays import build_mm_simulator_arrays

    cache_dir = os.path.join("results", "mm_noise", "_sim_arrays")
    os.makedirs(cache_dir, exist_ok=True)
    arrays_path = os.path.join(cache_dir, f"{pool_id}_{start}_{end}_mm.npz")

    if not os.path.exists(arrays_path):
        print(f"  Building noise arrays for {pool_id} {start} -> {end}...")
        arrays = build_mm_simulator_arrays(
            token_a=tokens[0], token_b=tokens[1],
            start_date=start, end_date=end,
            mm_artifact_dir="results/mm_noise",
            competitor_tvl_path="results/competitor_tvl/competitor_tvl.npz",
            pool_id=pool_id,
        )
        np.savez(arrays_path,
                 noise_base=arrays["noise_base"],
                 competitor_tvl=arrays["competitor_tvl"])
    return arrays_path


def run_single_period(args, start, end):
    """Run PR sweep for a single period. Returns dict of metrics per PR."""
    tokens = args.tokens
    margin = args.margin
    shift = args.shift
    price_ratios = np.array(args.prs or DEFAULT_PRS)

    # Noise arrays
    noise_path = None
    if args.noise_model == "mm_observed" and args.pool_id:
        noise_path = build_noise_arrays(args.pool_id, tokens, start, end)

    fp = {
        "rule": "reclamm",
        "tokens": tokens,
        "startDateString": f"{start} 00:00:00",
        "endDateString": f"{end} 00:00:00",
        "initial_pool_value": args.initial_pool_value,
        "do_arb": True,
        "arb_frequency": args.arb_frequency,
        "fees": args.fees,
        "gas_cost": args.gas_cost,
        "arb_fees": 0.0,
        "protocol_fee_split": 0.25,
        "noise_trader_ratio": 0.0,
        "reclamm_interpolation_method": "geometric",
        "reclamm_centeredness_scaling": False,
        "reclamm_use_shift_exponent": True,
    }
    if noise_path:
        fp["noise_model"] = "mm_observed"
        fp["noise_arrays_path"] = noise_path
    else:
        fp["noise_trader_ratio"] = 0.0

    params_list = [
        {
            "price_ratio": jnp.array(float(pr)),
            "centeredness_margin": jnp.array(margin),
            "shift_exponent": jnp.array(shift),
        }
        for pr in price_ratios
    ]

    tok_str = "/".join(tokens)
    print(f"  PR sweep: {tok_str}, {start} -> {end}, {len(price_ratios)} PRs")

    results = do_run_on_historic_data(
        run_fingerprint=fp, params=params_list, verbose=False,
    )

    # Compute metrics
    rohs, fee_revs, final_vals = [], [], []
    hodl_final = None

    for i, pr in enumerate(price_ratios):
        out = results[i]
        val = np.array(out["value"])
        prices = np.array(out["prices"])
        reserves_0 = out["reserves"][0]
        hodl = np.sum(np.array(reserves_0) * prices, axis=1)
        if hodl_final is None:
            hodl_final = float(hodl[-1])

        roh = val[-1] / hodl[-1] - 1
        fr = np.array(out.get("fee_revenue", np.zeros(len(val))))

        rohs.append(roh)
        fee_revs.append(float(fr.sum()))
        final_vals.append(float(val[-1]))

    # Run a base Balancer 50/50 pool as comparator (arb-only, same fees/gas)
    bal_fp = {
        "rule": "balancer",
        "tokens": tokens,
        "startDateString": f"{start} 00:00:00",
        "endDateString": f"{end} 00:00:00",
        "initial_pool_value": args.initial_pool_value,
        "do_arb": True,
        "arb_frequency": args.arb_frequency,
        "fees": args.fees,
        "gas_cost": args.gas_cost,
        "arb_fees": 0.0,
        "protocol_fee_split": 0.25,
        "noise_trader_ratio": 0.0,
    }
    bal_params = {"initial_weights_logits": jnp.array([0.0, 0.0])}
    print(f"  Running Balancer 50/50 comparator...")
    bal_result = do_run_on_historic_data(
        run_fingerprint=bal_fp, params=bal_params, verbose=False,
    )
    bal_val = np.array(bal_result["value"])
    bal_hodl = np.sum(np.array(bal_result["reserves"][0]) * np.array(bal_result["prices"]), axis=1)
    bal_roh = bal_val[-1] / bal_hodl[-1] - 1
    bal_fee = float(np.array(bal_result.get("fee_revenue", np.zeros(1))).sum())
    print(f"  Balancer 50/50: RoH={bal_roh:+.2%}, fee=${bal_fee:,.0f}")

    return {
        "price_ratios": price_ratios,
        "rohs": rohs,
        "fee_revs": fee_revs,
        "final_vals": final_vals,
        "hodl_final": hodl_final,
        "balancer_roh": bal_roh,
        "balancer_fee": bal_fee,
        "balancer_final": float(bal_val[-1]),
        "start": start,
        "end": end,
    }


def plot_single_period(data, args, outpath):
    """3-panel plot for a single period."""
    price_ratios = data["price_ratios"]
    rohs = data["rohs"]
    fee_revs = data["fee_revs"]
    final_vals = data["final_vals"]
    hodl_final = data["hodl_final"]
    start, end = data["start"], data["end"]
    onchain_pr = args.onchain_pr
    tok_str = "/".join(args.tokens)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax in axes:
        _style_ax(ax)

    # Panel 1: RoH vs PR
    axes[0].plot(price_ratios, [r * 100 for r in rohs],
                     "o-", color="#e74c3c", markersize=5, label="reClAMM")
    axes[0].axhline(0, color="white", ls=":", alpha=0.3)
    bal_roh = data.get("balancer_roh")
    if bal_roh is not None:
        axes[0].axhline(bal_roh * 100, color="#3498db", ls="-", alpha=0.7,
                        label=f"Balancer 50/50 ({bal_roh*100:+.1f}%)")
    if onchain_pr:
        axes[0].axvline(onchain_pr, color="#f39c12", ls="--", alpha=0.7,
                        label=f"On-chain PR={onchain_pr}")
    selected_pr = getattr(args, "selected_pr", None)
    if selected_pr:
        axes[0].axvline(selected_pr, color="#9b59b6", ls="-.", alpha=0.8, linewidth=2,
                        label=f"Selected PR={selected_pr:.2f}")
    best_pr_idx = int(np.argmax(rohs))
    best_pr = price_ratios[best_pr_idx]
    axes[0].axvline(best_pr, color="#2ecc71", ls="-", alpha=0.8, linewidth=2,
                    label=f"Best PR={best_pr:.1f} ({rohs[best_pr_idx]*100:+.1f}%)")
    axes[0].set_xlabel("Price Ratio", color=TC)
    axes[0].set_ylabel("Returns over HODL (%)", color=TC)
    axes[0].set_title("RoH vs Price Ratio", color=TC)
    axes[0].legend(fontsize=8, facecolor=BG, edgecolor=TC, labelcolor=TC)

    # Panel 2: Fee revenue
    axes[1].plot(price_ratios, [f / 1000 for f in fee_revs],
                     "o-", color="#2ecc71", markersize=5)
    if onchain_pr:
        axes[1].axvline(onchain_pr, color="#f39c12", ls="--", alpha=0.7)
    if selected_pr:
        axes[1].axvline(selected_pr, color="#9b59b6", ls="-.", alpha=0.8, linewidth=2)
    axes[1].set_xlabel("Price Ratio", color=TC)
    axes[1].set_ylabel("Fee Revenue ($K)", color=TC)
    axes[1].set_title("Cumulative Fee Revenue", color=TC)

    fig.suptitle(
        f"{tok_str} reCLAMM PR Sweep (margin={args.margin}, shift={args.shift}, "
        f"gas=${args.gas_cost})\n{start} -> {end}",
        color=TC, fontsize=12, fontweight="bold",
    )
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_heatmap(all_data, args, outpath):
    """Heatmap: RoH as function of PR (x) and start date (y)."""
    tok_str = "/".join(args.tokens)
    price_ratios = all_data[0]["price_ratios"]
    n_pr = len(price_ratios)
    n_periods = len(all_data)

    roh_matrix = np.zeros((n_periods, n_pr))
    fee_matrix = np.zeros((n_periods, n_pr))
    period_labels = []

    for j, data in enumerate(all_data):
        roh_matrix[j, :] = data["rohs"]
        fee_matrix[j, :] = data["fee_revs"]
        period_labels.append(f"{data['start']} -> {data['end']}")

    fig, axes = plt.subplots(1, 2, figsize=(18, max(4, n_periods * 0.6 + 2)))
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TC)

    # RoH heatmap
    vmax = max(abs(roh_matrix.min()), abs(roh_matrix.max()))
    im0 = axes[0].imshow(
        roh_matrix * 100, aspect="auto", cmap="RdYlGn", vmin=-vmax * 100, vmax=vmax * 100,
    )
    axes[0].set_xticks(range(n_pr))
    axes[0].set_xticklabels([f"{p:.1f}" if p < 10 else f"{p:.0f}" for p in price_ratios],
                            rotation=45, fontsize=8, color=TC)
    axes[0].set_yticks(range(n_periods))
    axes[0].set_yticklabels(period_labels, fontsize=8, color=TC)
    axes[0].set_xlabel("Price Ratio", color=TC)
    axes[0].set_title("Returns over HODL (%)", color=TC, fontsize=12)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.8)
    cb0.ax.tick_params(colors=TC)

    # Annotate cells
    for j in range(n_periods):
        for i in range(n_pr):
            val = roh_matrix[j, i] * 100
            color = "black" if abs(val) < vmax * 50 else "white"
            axes[0].text(i, j, f"{val:+.1f}", ha="center", va="center",
                        fontsize=6, color=color)

    # Mark on-chain PR with white cross-hatching
    if args.onchain_pr:
        pr_arr = np.array(price_ratios)
        nearest = np.argmin(np.abs(np.log(pr_arr) - np.log(args.onchain_pr)))
        for j in range(n_periods):
            axes[0].add_patch(plt.Rectangle(
                (nearest - 0.5, j - 0.5), 1, 1,
                fill=False, edgecolor="white", linewidth=2,
                hatch="//", label="On-chain PR" if j == 0 else None,
            ))

    # Mark best PR per row with green border
    for j in range(n_periods):
        best_i = int(np.argmax(roh_matrix[j, :]))
        axes[0].add_patch(plt.Rectangle(
            (best_i - 0.5, j - 0.5), 1, 1,
            fill=False, edgecolor="#2ecc71", linewidth=3,
            label="Best RoH" if j == 0 else None,
        ))

    # Best overall PR: two metrics
    # 1) Geometric mean = product of (1+RoH) — penalises variance, correct for compounding
    compounded = np.prod(1 + roh_matrix, axis=0)
    geo_mean_roh = np.sign(compounded) * np.abs(compounded) ** (1.0 / n_periods) - 1
    best_geo_i = int(np.argmax(geo_mean_roh))
    # 2) Median — robust to outlier months
    median_roh = np.median(roh_matrix, axis=0)
    best_median_i = int(np.argmax(median_roh))

    # Print both for diagnostics
    print(f"  Best geo-mean PR={price_ratios[best_geo_i]:.2f} "
          f"(geo mean {geo_mean_roh[best_geo_i]*100:+.3f}%)")
    print(f"  Best median  PR={price_ratios[best_median_i]:.2f} "
          f"(median {median_roh[best_median_i]*100:+.3f}%)")
    for i, pr in enumerate(price_ratios):
        print(f"    PR={pr:5.2f}: geo={geo_mean_roh[i]*100:+.3f}%  "
              f"median={median_roh[i]*100:+.3f}%")

    # Use median for the column shading (robust to outlier months)
    best_overall_i = best_median_i
    best_overall_pr = price_ratios[best_overall_i]
    best_median = median_roh[best_overall_i]
    for j in range(n_periods):
        axes[0].add_patch(plt.Rectangle(
            (best_overall_i - 0.5, j - 0.5), 1, 1,
            fill=True, facecolor="#3498db", alpha=0.25, edgecolor="#3498db",
            linewidth=2, linestyle="--",
            label=(f"Best overall PR={best_overall_pr:.2f} "
                   f"(median {best_median*100:+.2f}%)") if j == 0 else None,
        ))

    # Legend for markings
    from matplotlib.patches import Patch
    legend_elements = []
    if args.onchain_pr:
        legend_elements.append(Patch(facecolor="none", edgecolor="white",
                                     hatch="//", label="On-chain PR"))
    legend_elements.append(Patch(facecolor="none", edgecolor="#2ecc71",
                                 linewidth=3, label="Best RoH (per period)"))
    legend_elements.append(Patch(facecolor="#3498db", alpha=0.25, edgecolor="#3498db",
                                 linestyle="--", linewidth=2,
                                 label=f"Best overall PR={best_overall_pr:.2f} "
                                       f"(median {best_median*100:+.2f}%)"))
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=7,
                   facecolor=BG, edgecolor=TC, labelcolor=TC)

    # Fee revenue heatmap
    im1 = axes[1].imshow(
        fee_matrix / 1000, aspect="auto", cmap="YlGn",
    )
    axes[1].set_xticks(range(n_pr))
    axes[1].set_xticklabels([f"{p:.1f}" if p < 10 else f"{p:.0f}" for p in price_ratios],
                            rotation=45, fontsize=8, color=TC)
    axes[1].set_yticks(range(n_periods))
    axes[1].set_yticklabels(period_labels, fontsize=8, color=TC)
    axes[1].set_xlabel("Price Ratio", color=TC)
    axes[1].set_title("Fee Revenue ($K)", color=TC, fontsize=12)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.8)
    cb1.ax.tick_params(colors=TC)

    for j in range(n_periods):
        for i in range(n_pr):
            val = fee_matrix[j, i] / 1000
            axes[1].text(i, j, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color="black")

    if args.onchain_pr:
        for j in range(n_periods):
            axes[1].add_patch(plt.Rectangle(
                (nearest - 0.5, j - 0.5), 1, 1,
                fill=False, edgecolor="white", linewidth=2,
                hatch="//",
            ))
    # Best fee revenue per row
    for j in range(n_periods):
        best_i = int(np.argmax(fee_matrix[j, :]))
        axes[1].add_patch(plt.Rectangle(
            (best_i - 0.5, j - 0.5), 1, 1,
            fill=False, edgecolor="#2ecc71", linewidth=3,
        ))
    # Best overall PR for fee revenue: maximise total fee across all periods
    total_fees = np.sum(fee_matrix, axis=0)
    best_fee_overall_i = int(np.argmax(total_fees))
    best_fee_overall_pr = price_ratios[best_fee_overall_i]
    for j in range(n_periods):
        axes[1].add_patch(plt.Rectangle(
            (best_fee_overall_i - 0.5, j - 0.5), 1, 1,
            fill=True, facecolor="#3498db", alpha=0.25, edgecolor="#3498db",
            linewidth=2, linestyle="--",
        ))

    fig.suptitle(
        f"{tok_str} reClAMM: PR x Period Summary (margin={args.margin}, "
        f"shift={args.shift}, gas=${args.gas_cost})",
        color=TC, fontsize=13, fontweight="bold",
    )
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved heatmap: {outpath}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", nargs=2, default=["COW", "ETH"])
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--multi-period", action="store_true",
                   help="Run multiple periods starting at regular intervals")
    p.add_argument("--period-months", type=int, default=6,
                   help="Length of each period in months (default: 6)")
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--shift", type=float, default=0.1)
    p.add_argument("--fees", type=float, default=0.003)
    p.add_argument("--gas-cost", type=float, default=3.0)
    p.add_argument("--arb-frequency", type=int, default=3)
    p.add_argument("--initial-pool-value", type=float, default=600_000.0)
    p.add_argument("--pool-id", default="0xd321300ef77067")
    p.add_argument("--noise-model", default="mm_observed",
                   choices=["mm_observed", "none"])
    p.add_argument("--onchain-pr", type=float, default=None,
                   help="On-chain PR to mark on plot")
    p.add_argument("--selected-pr", type=float, default=None,
                   help="Optimiser-selected PR to mark on plot")
    p.add_argument("--prs", type=float, nargs="+", default=None)
    p.add_argument("--output-dir", default="results/cow_sweep/pr_sweeps")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tok_tag = "_".join(args.tokens)

    if args.multi_period:
        from dateutil.relativedelta import relativedelta
        from datetime import date
        period_months = args.period_months
        step_months = 1
        starts = []
        d = date(2024, 1, 1)
        data_end = date(2026, 4, 1)
        while d + relativedelta(months=period_months) <= data_end:
            starts.append(d.isoformat())
            d += relativedelta(months=step_months)
        periods = [(s, (datetime.strptime(s, "%Y-%m-%d") +
                        relativedelta(months=period_months)).strftime("%Y-%m-%d"))
                   for s in starts]

        all_data = []
        for start, end in periods:
            print(f"\n{'='*60}")
            try:
                data = run_single_period(args, start, end)
                all_data.append(data)

                outpath = os.path.join(
                    args.output_dir,
                    f"pr_sweep_{tok_tag}_{start}_{end}_{args.period_months}mo_m{args.margin}_s{args.shift}.png",
                )
                plot_single_period(data, args, outpath)

                # Print table
                prs = data["price_ratios"]
                print(f"  {'PR':>8s} {'RoH':>8s} {'Fee $K':>8s} {'Final $K':>10s}")
                for i, pr in enumerate(prs):
                    print(f"  {pr:>8.2f} {data['rohs'][i]:>+7.2%} "
                          f"{data['fee_revs'][i]/1000:>8.0f} "
                          f"{data['final_vals'][i]/1000:>10.0f}")
            except Exception as e:
                print(f"  FAILED: {e}")

        if all_data:
            heatmap_path = os.path.join(
                args.output_dir,
                f"pr_heatmap_{tok_tag}_{args.period_months}mo_m{args.margin}_s{args.shift}.png",
            )
            plot_heatmap(all_data, args, heatmap_path)

    elif args.start and args.end:
        data = run_single_period(args, args.start, args.end)
        outpath = os.path.join(
            args.output_dir,
            f"pr_sweep_{tok_tag}_{args.start}_{args.end}_m{args.margin}_s{args.shift}.png",
        )
        plot_single_period(data, args, outpath)

        print(f"\n{'PR':>8s} {'RoH':>8s} {'Fee $K':>8s} {'Final $K':>10s}")
        print("-" * 40)
        for i, pr in enumerate(data["price_ratios"]):
            print(f"{pr:>8.2f} {data['rohs'][i]:>+7.2%} "
                  f"{data['fee_revs'][i]/1000:>8.0f} "
                  f"{data['final_vals'][i]/1000:>10.0f}")
    else:
        print("ERROR: provide --start/--end or --multi-period")


if __name__ == "__main__":
    main()
