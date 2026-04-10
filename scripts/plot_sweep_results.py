"""Plot all sweep results: rerun each (objective, period) combo and compare.

Reads results/sweep/*.json, groups by period, reruns forward passes over
the full date range (train + test to end_test_date), and produces per-period
comparison plots.

Usage:
    python scripts/plot_sweep_results.py
    python scripts/plot_sweep_results.py --end-test-date "2026-03-01 00:00:00"
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

import jax.numpy as jnp
from quantammsim.runners.jax_runners import do_run_on_historic_data


SWEEP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "sweep",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "sweep", "plots",
)

# Period definitions: (start, train_end, default_test_end)
# --end-test-date overrides default_test_end for all periods
PERIODS = {
    "bull_2023":    ("2023-06-01", "2024-06-01", "2026-03-01"),
    "default_2024": ("2024-06-01", "2025-06-01", "2026-03-01"),
    "recent_2025":  ("2025-01-01", "2025-09-01", "2026-03-01"),
}

# Base fingerprint for reClAMM AAVE/ETH with MM noise
BASE_FP = {
    "rule": "reclamm",
    "tokens": ["AAVE", "ETH"],
    "do_arb": True,
    "arb_frequency": 4,
    "fees": 0.0025,
    "gas_cost": 1.0,
    "arb_fees": 0.0,
    "protocol_fee_split": 0.25,
    "noise_trader_ratio": 0.0,
    "noise_model": "mm_observed",
    "noise_arrays_path": "results/mm_noise/_sim_arrays/"
                         "0x9d1fcf346ea1b0_2024-06-01_2026-03-01_mm.npz",
    "reclamm_interpolation_method": "geometric",
    "reclamm_centeredness_scaling": False,
    "reclamm_learn_arc_length_speed": False,
    "reclamm_use_shift_exponent": True,
    "initial_pool_value": 20_000_000.0,
}

OBJ_SHORT = {
    "daily_log_sharpe": "sharpe",
    "daily_log_sharpe_excess": "excess_sharpe",
    "fee_revenue_over_value": "fee_rev",
    "returns_over_hodl": "ret_hodl",
    "calmar": "calmar",
    "sterling": "sterling",
    "weekly_rovar": "rovar",
}

BG = "#162536"
TEXT_COLOR = "#E6CE97"
COLORS = [
    "#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#8e44ad",
]


def load_sweep_results():
    """Load all sweep result JSONs, grouped by period."""
    results = {}  # period -> [(obj_name, params)]
    for path in sorted(glob.glob(os.path.join(SWEEP_DIR, "*.json"))):
        fname = os.path.basename(path).replace(".json", "")
        # Parse: {objective}_{period_name}
        for period_name in PERIODS:
            if fname.endswith(f"_{period_name}"):
                obj_name = fname[: -(len(period_name) + 1)]
                with open(path) as f:
                    data = json.load(f)
                # Extract params from the first (only) objective key
                if isinstance(data, dict):
                    params_raw = list(data.values())[0]
                    if isinstance(params_raw, dict):
                        params = {
                            k: v for k, v in params_raw.items()
                            if k in ("price_ratio", "centeredness_margin",
                                     "shift_exponent", "arc_length_speed")
                        }
                        if period_name not in results:
                            results[period_name] = []
                        results[period_name].append((obj_name, params))
                break
    return results


_arrays_cache = {}


def _get_noise_arrays_path(start_date, end_date):
    """Build or retrieve MM noise arrays for this date range."""
    key = (start_date, end_date)
    if key in _arrays_cache:
        return _arrays_cache[key]

    from quantammsim.calibration.noise_model_arrays import build_mm_simulator_arrays

    cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "mm_noise", "_sim_arrays")
    os.makedirs(cache_dir, exist_ok=True)
    arrays_path = os.path.join(
        cache_dir, f"0x9d1fcf346ea1b0_{start_date}_{end_date}_mm.npz")

    if not os.path.exists(arrays_path):
        print(f"\n    Building noise arrays for {start_date}→{end_date}...",
              end=" ", flush=True)
        arrays = build_mm_simulator_arrays(
            token_a="AAVE", token_b="ETH",
            start_date=start_date, end_date=end_date,
            mm_artifact_dir="results/mm_noise",
            competitor_tvl_path="results/competitor_tvl/competitor_tvl.npz",
            pool_id="0x9d1fcf346ea1b0",
        )
        np.savez(arrays_path,
                 noise_base=arrays["noise_base"],
                 competitor_tvl=arrays["competitor_tvl"])
        print("done")

    _arrays_cache[key] = arrays_path
    return arrays_path


def run_config(params, start_date, end_date, end_test_date=None):
    """Run a forward pass over start→end_test (or end if no test)."""
    actual_end = end_test_date or end_date
    arrays_path = _get_noise_arrays_path(start_date, actual_end)

    fp = dict(BASE_FP)
    fp["startDateString"] = f"{start_date} 00:00:00"
    fp["endDateString"] = f"{actual_end} 00:00:00"
    fp["noise_arrays_path"] = arrays_path
    jax_params = {k: jnp.array(v) for k, v in params.items()}
    return do_run_on_historic_data(run_fingerprint=fp, params=jax_params)


def _style_axis(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_alpha(0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color=TEXT_COLOR)


def plot_period(period_name, obj_results, end_test_date, output_dir):
    """Plot all objectives for one period."""
    start, train_end, default_test_end = PERIODS[period_name]
    test_end = end_test_date or default_test_end

    print(f"\n{'='*80}")
    print(f"Period: {period_name}  ({start} → {train_end} → {test_end})")
    print(f"{'='*80}")

    # On-chain baselines
    ONCHAIN_CONFIGS = {
        "OnChain-launch": {
            "price_ratio": 1.5, "centeredness_margin": 0.5,
            "shift_exponent": 0.1,
        },
        "OnChain-current": {
            "price_ratio": 4.0, "centeredness_margin": 0.1,
            "shift_exponent": 0.001,
        },
    }

    # Run all configs + baselines
    runs = {}       # label -> forward pass output
    run_params = {} # label -> pool params dict
    all_configs = (
        [(name, params) for name, params in ONCHAIN_CONFIGS.items()]
        + [(f"{OBJ_SHORT.get(obj, obj)} (pr={p.get('price_ratio', 0):.2f})", p)
           for obj, p in obj_results]
    )
    for label, params in all_configs:
        print(f"  Running {label}...", end=" ", flush=True)
        try:
            out = run_config(params, start, train_end, test_end)
            runs[label] = out
            run_params[label] = params
            fv = float(out["final_value"])
            hodl = float((out["reserves"][0] * out["prices"][-1]).sum())
            print(f"final=${fv:,.0f}  RoH={fv/hodl - 1:+.2%}")
        except Exception as e:
            print(f"FAILED: {e}")

    if not runs:
        print("  No successful runs!")
        return

    # HODL baseline
    first_out = next(iter(runs.values()))
    hodl_reserves = first_out["reserves"][0]
    hodl_values = np.sum(
        np.array(hodl_reserves) * np.array(first_out["prices"]), axis=1)

    n_minutes = len(first_out["value"])
    start_dt = datetime.strptime(f"{start} 00:00:00", "%Y-%m-%d %H:%M:%S")
    train_end_dt = datetime.strptime(f"{train_end} 00:00:00", "%Y-%m-%d %H:%M:%S")
    dates = pd.date_range(start=start_dt, periods=n_minutes, freq="1min")
    step = 1440
    dates_daily = dates[::step]

    # ── Plot 1: Full period value ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    for ci, (label, out) in enumerate(runs.items()):
        vals = np.array(out["value"][::step]) / 1e6
        is_baseline = label.startswith("OnChain")
        ax.plot(dates_daily[:len(vals)], vals,
                linewidth=1.5 if is_baseline else 1.8,
                linestyle="--" if is_baseline else "-",
                alpha=0.7 if is_baseline else 1.0,
                color=COLORS[ci % len(COLORS)], label=label)

    hodl_daily = hodl_values[::step] / 1e6
    ax.plot(dates_daily[:len(hodl_daily)], hodl_daily, linewidth=2,
            color="white", alpha=0.7, linestyle="--", label="HODL")

    ax.axvline(x=train_end_dt, color="white", linestyle=":", alpha=0.5)
    _style_axis(ax)
    ax.set_ylabel("Pool Value ($M)", color=TEXT_COLOR)
    ax.set_title(f"reClAMM AAVE/ETH — {period_name}",
                 color=TEXT_COLOR, fontsize=14, pad=10)
    ax.legend(loc="upper left", fontsize=7, facecolor=BG,
              edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR, ncol=2)

    # Fee revenue
    ax = axes[1]
    for ci, (label, out) in enumerate(runs.items()):
        fr = out.get("fee_revenue")
        if fr is None:
            continue
        cumfee = np.cumsum(np.array(fr))[::step] / 1e3
        is_baseline = label.startswith("OnChain")
        ax.plot(dates_daily[:len(cumfee)], cumfee,
                linewidth=1.2 if is_baseline else 1.5,
                linestyle="--" if is_baseline else "-",
                alpha=0.7 if is_baseline else 1.0,
                color=COLORS[ci % len(COLORS)], label=label)

    ax.axvline(x=train_end_dt, color="white", linestyle=":", alpha=0.5)
    _style_axis(ax)
    ax.set_ylabel("Cum. Fee Revenue ($K)", color=TEXT_COLOR)
    ax.set_xlabel("Date", color=TEXT_COLOR)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"sweep_{period_name}_value.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── Plot 2: Test-only normalised value + cumulative fee revenue ──
    train_minutes = int((train_end_dt - start_dt).total_seconds() / 60)
    test_start = min(train_minutes, n_minutes - 1)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    test_dates = dates[test_start::step]

    for ci, (label, out) in enumerate(runs.items()):
        vals = np.array(out["value"])
        test_vals = vals[test_start::step]
        if len(test_vals) < 2:
            continue
        normed = test_vals / test_vals[0]
        is_baseline = label.startswith("OnChain")
        ax.plot(test_dates[:len(normed)], normed,
                linewidth=1.5 if is_baseline else 1.8,
                linestyle="--" if is_baseline else "-",
                alpha=0.7 if is_baseline else 1.0,
                color=COLORS[ci % len(COLORS)], label=label)

    hodl_test = hodl_values[test_start::step]
    if len(hodl_test) > 1:
        ax.plot(test_dates[:len(hodl_test)], hodl_test / hodl_test[0],
                linewidth=2, color="white", alpha=0.7, linestyle="--",
                label="HODL")

    ax.axhline(1.0, color="white", linestyle=":", alpha=0.3)
    _style_axis(ax)
    ax.set_title(f"Test Period (normalised) — {period_name}",
                 color=TEXT_COLOR, fontsize=14, pad=10)
    ax.set_ylabel("Normalised Value", color=TEXT_COLOR)
    ax.legend(loc="best", fontsize=7, facecolor=BG,
              edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR, ncol=2)

    # Cumulative fee revenue (test period only)
    ax = axes[1]
    for ci, (label, out) in enumerate(runs.items()):
        fr = out.get("fee_revenue")
        if fr is None:
            continue
        fr = np.array(fr)
        test_fr = fr[test_start:]
        cumfee = np.cumsum(test_fr)[::step] / 1e3
        is_baseline = label.startswith("OnChain")
        ax.plot(test_dates[:len(cumfee)], cumfee,
                linewidth=1.2 if is_baseline else 1.5,
                linestyle="--" if is_baseline else "-",
                alpha=0.7 if is_baseline else 1.0,
                color=COLORS[ci % len(COLORS)], label=label)

    _style_axis(ax)
    ax.set_ylabel("Cum. Fee Revenue ($K)", color=TEXT_COLOR)
    ax.set_xlabel("Date", color=TEXT_COLOR)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"sweep_{period_name}_test.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── Summary table ──
    print(f"\n  {'Objective':<25s} {'PR':>8s} {'Margin':>7s} {'ShiftExp':>9s}"
          f" {'Final $M':>10s} {'HODL $M':>10s} {'RoH':>8s} {'Fee $K':>8s}")
    for label, out in runs.items():
        fv = float(out["final_value"]) / 1e6
        hodl_end = float(hodl_values[-1]) / 1e6
        roh = float(out["final_value"]) / float(hodl_values[-1]) - 1
        fr = float(np.array(out.get("fee_revenue", [0])).sum()) / 1e3
        p = run_params.get(label, {})
        pr = p.get("price_ratio", float("nan"))
        margin = p.get("centeredness_margin", float("nan"))
        se = p.get("shift_exponent", float("nan"))
        print(f"  {label:<25s} {pr:>8.2f} {margin:>7.3f} {se:>9.4g}"
              f" ${fv:>9.2f} ${hodl_end:>9.2f} {roh:>+7.1%} ${fr:>7.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--end-test-date", default=None,
                        help="Override test end date for all periods")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--periods", nargs="+", default=None,
                        help="Only plot these periods (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_sweep_results()
    print(f"Loaded {sum(len(v) for v in results.values())} results"
          f" across {len(results)} periods")

    for period_name in sorted(results.keys()):
        if args.periods and period_name not in args.periods:
            continue
        plot_period(period_name, results[period_name],
                    args.end_test_date, args.output_dir)


if __name__ == "__main__":
    main()
