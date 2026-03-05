#!/usr/bin/env python3
"""Plot reClAMM pool performance from Optuna tuning results.

Reads the SGD-compatible JSON output of tune_reclamm_params.py (or any Optuna
run), extracts the best trial's pool params, re-runs a forward pass over the
full train+test window, and produces a value-over-time plot with on-chain
baselines and cumulative fee revenue.

Usage:
    python scripts/plot_reclamm_optuna_result.py results/run_<hash>.json
    python scripts/plot_reclamm_optuna_result.py results/run_<hash>.json --output my_plot.png
    python scripts/plot_reclamm_optuna_result.py results/run_<hash>.json --top-k 3
"""

import argparse
import json
import sys

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from quantammsim.runners.jax_runners import do_run_on_historic_data

# ── On-chain baselines ────────────────────────────────────────────────────
ONCHAIN_LAUNCH_PARAMS = {
    "price_ratio": 1.5, "centeredness_margin": 0.5, "shift_exponent": 0.1,
}
ONCHAIN_CURRENT_PARAMS = {
    "price_ratio": 4.0, "centeredness_margin": 0.1, "shift_exponent": 0.001,
}

BG = "#162536"
TEXT_COLOR = "#E6CE97"
COLORS = [
    "#3498db", "#2ecc71", "#e74c3c",  # top-k
    "#f39c12",  # on-chain launch
    "#9b59b6",  # on-chain current
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("results_json", help="Path to run_<hash>.json from Optuna")
    p.add_argument("--top-k", type=int, default=1,
                   help="Plot top K trials by objective (default 1)")
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: auto-generated)")
    p.add_argument("--no-onchain", action="store_true",
                   help="Skip on-chain baseline runs")
    return p.parse_args()


def load_results(path):
    """Load the double-encoded JSONL from Optuna results."""
    with open(path) as f:
        raw = f.read()
    data = json.loads(raw)
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, list) or len(data) < 2:
        print(f"ERROR: Expected [config, trial1, trial2, ...], got {type(data)}")
        sys.exit(1)
    config = data[0]
    trials = data[1:]
    return config, trials


def extract_pool_params(trial, config):
    """Extract reClAMM pool params from a trial entry."""
    param_keys = ["price_ratio", "centeredness_margin", "shift_exponent",
                  "arc_length_speed", "fees"]
    params = {}
    for k in param_keys:
        if k in trial:
            params[k] = trial[k]
    return params


def run_full_period(params, config, fees_override=None):
    """Run forward pass over the full train+test window."""
    fees = fees_override if fees_override is not None else config["fees"]
    fp = {
        "rule": "reclamm",
        "tokens": config["tokens"],
        "startDateString": config["startDateString"],
        "endDateString": config["endTestDateString"],  # full period
        "initial_pool_value": config["initial_pool_value"],
        "do_arb": config["do_arb"],
        "fees": fees,
        "gas_cost": config.get("gas_cost", 1.0),
        "arb_fees": config.get("arb_fees", 0.0),
        "protocol_fee_split": config.get("protocol_fee_split", 0.0),
        "reclamm_use_shift_exponent": config.get("reclamm_use_shift_exponent", True),
        "reclamm_interpolation_method": config.get("reclamm_interpolation_method", "geometric"),
        "reclamm_centeredness_scaling": config.get("reclamm_centeredness_scaling", False),
        "reclamm_learn_arc_length_speed": config.get("reclamm_learn_arc_length_speed", False),
    }
    jax_params = {k: jnp.array(v) for k, v in params.items()}
    return do_run_on_historic_data(run_fingerprint=fp, params=jax_params)


def plot_results(configs, time_series, hodl_values, config, args):
    """Two-panel plot: value-over-time + cumulative fee revenue."""
    train_end_str = config["endDateString"]
    train_end_dt = datetime.strptime(train_end_str, "%Y-%m-%d %H:%M:%S")

    first_out = next(iter(time_series.values()))
    n_minutes = len(first_out["value"])
    dates = pd.date_range(
        start=datetime.strptime(config["startDateString"], "%Y-%m-%d %H:%M:%S"),
        periods=n_minutes, freq="1min",
    )
    step = 1440
    dates_daily = dates[::step]

    has_fee_revenue = any(
        "fee_revenue" in time_series[n] and time_series[n]["fee_revenue"] is not None
        for n in time_series
    )
    n_panels = 2 if has_fee_revenue else 1
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 5 * n_panels),
        sharex=True, gridspec_kw={"height_ratios": [3, 1] if n_panels == 2 else [1]},
    )
    if n_panels == 1:
        axes = [axes]
    ax_val = axes[0]

    # ── Panel 1: Value over time ──────────────────────────────────────
    for i, (name, meta) in enumerate(configs.items()):
        out = time_series[name]
        vals = np.array(out["value"][::step]) / 1e6
        label = f"{name}"
        if "test_objective" in meta:
            obj_name = config.get("return_val", "objective")
            label += f" (OOS {obj_name}={meta['test_objective']:.4f})"
        ax_val.plot(dates_daily[:len(vals)], vals, linewidth=2,
                    color=COLORS[i % len(COLORS)], label=label)

    hodl_daily = hodl_values[::step] / 1e6
    ax_val.plot(dates_daily[:len(hodl_daily)], hodl_daily, linewidth=2,
                color="white", alpha=0.7, linestyle="--", label="HODL")

    ax_val.axvline(x=train_end_dt, color="white", linestyle=":", alpha=0.5, linewidth=1.5)
    ylims = ax_val.get_ylim()
    ax_val.text(train_end_dt - pd.Timedelta(days=10), ylims[1] * 0.97, "Train",
                color="white", alpha=0.6, fontsize=11, ha="right", va="top")
    ax_val.text(train_end_dt + pd.Timedelta(days=10), ylims[1] * 0.97, "Test",
                color="white", alpha=0.6, fontsize=11, ha="left", va="top")

    _style_axis(ax_val)
    ax_val.set_ylabel("Pool Value ($M USD)", color=TEXT_COLOR, fontsize=12)
    tokens_str = "/".join(config["tokens"])
    obj_name = config.get("return_val", "objective")
    ax_val.set_title(
        f"reClAMM Optuna-Optimized ({obj_name}) — {tokens_str}",
        color=TEXT_COLOR, fontsize=13, pad=15,
    )
    ax_val.legend(loc="upper left", fontsize=9, facecolor=BG,
                  edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)

    # ── Panel 2: Cumulative fee revenue ───────────────────────────────
    if has_fee_revenue:
        ax_fee = axes[1]
        for i, (name, meta) in enumerate(configs.items()):
            out = time_series[name]
            fr = out.get("fee_revenue")
            if fr is None:
                continue
            fr = np.array(fr)
            cumfee = np.cumsum(fr)[::step] / 1e3
            ax_fee.plot(dates_daily[:len(cumfee)], cumfee, linewidth=2,
                        color=COLORS[i % len(COLORS)], label=name)

        ax_fee.axvline(x=train_end_dt, color="white", linestyle=":", alpha=0.5, linewidth=1.5)
        _style_axis(ax_fee)
        ax_fee.set_ylabel("Cumulative Fee Revenue ($K)", color=TEXT_COLOR, fontsize=12)
        ax_fee.set_xlabel("Date", color=TEXT_COLOR, fontsize=12)
        ax_fee.legend(loc="upper left", fontsize=9, facecolor=BG,
                      edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)
    else:
        ax_val.set_xlabel("Date", color=TEXT_COLOR, fontsize=12)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()

    output = args.output or f"reclamm_optuna_{tokens_str.replace('/', '_')}.png"
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved plot to {output}")
    plt.close()


def _style_axis(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        spine.set_alpha(0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, color=TEXT_COLOR)


def main():
    args = parse_args()
    config, trials = load_results(args.results_json)
    tokens = config["tokens"]
    obj_name = config.get("return_val", "objective")

    # Sort trials by penalised objective
    trials_sorted = sorted(trials, key=lambda t: t.get("objective", 0), reverse=True)
    top_trials = trials_sorted[:args.top_k]

    print("=" * 80)
    print(f"reClAMM Optuna Result Plotter  —  objective: {obj_name}")
    print("=" * 80)
    print(f"  Results:  {args.results_json}")
    print(f"  Tokens:   {'/'.join(tokens)}")
    print(f"  Train:    {config['startDateString']} → {config['endDateString']}")
    print(f"  Test:     {config['endDateString']} → {config['endTestDateString']}")
    print(f"  Fees:     {config['fees']},  Gas: {config.get('gas_cost', 1.0)}")
    print(f"  Trials:   {len(trials)} total, plotting top {len(top_trials)}")

    configs = {}
    for i, trial in enumerate(top_trials):
        params = extract_pool_params(trial, config)
        name = f"#{trial.get('optuna_trial_number', i)} (rank {i+1})"
        configs[name] = {
            "params": params,
            "objective": trial.get("objective", 0),
            "train_objective": trial.get("train_objective", 0),
            "test_objective": trial.get("test_objective", 0),
            "train_sharpe": trial.get("train_sharpe", 0),
            "validation_sharpe": trial.get("validation_sharpe", 0),
        }
        print(f"\n  {name}:")
        print(f"    {obj_name}: train={trial.get('train_objective', 0):.4f}  "
              f"test={trial.get('test_objective', 0):.4f}  "
              f"penalised={trial.get('objective', 0):.4f}")
        print(f"    sharpe: train={trial.get('train_sharpe', 0):+.4f}  "
              f"val={trial.get('validation_sharpe', 0):+.4f}")
        for k, v in params.items():
            print(f"    {k}: {v:.6g}")

    if not args.no_onchain:
        configs["On-Chain (launch)"] = {"params": dict(ONCHAIN_LAUNCH_PARAMS)}
        configs["On-Chain (current)"] = {"params": dict(ONCHAIN_CURRENT_PARAMS)}

    # ── Full-period runs ──────────────────────────────────────────────
    print(f"\n--- Running full-period simulations ({config['startDateString']} → "
          f"{config['endTestDateString']}) ---")
    time_series = {}
    for name, cfg in configs.items():
        print(f"  {name}...", end=" ", flush=True)
        out = run_full_period(cfg["params"], config)
        time_series[name] = out
        fv = float(out["final_value"])
        fr = out.get("fee_revenue")
        fr_total = float(np.array(fr).sum()) if fr is not None else 0
        hodl = float((out["reserves"][0] * out["prices"][-1]).sum())
        print(f"final=${fv:,.0f}  hodl=${hodl:,.0f}  RoH={fv/hodl - 1:+.2%}  "
              f"fee_rev=${fr_total:,.0f}")

    first_out = next(iter(time_series.values()))
    hodl_reserves = first_out["reserves"][0]
    hodl_values = np.sum(
        np.array(hodl_reserves) * np.array(first_out["prices"]), axis=1,
    )

    # ── Plot ──────────────────────────────────────────────────────────
    plot_results(configs, time_series, hodl_values, config, args)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"SUMMARY  —  {'/'.join(tokens)}  —  {obj_name}")
    print(f"{'=' * 120}")
    hdr = (f"{'Config':<28s} {'Train '+obj_name:>20s} {'Test '+obj_name:>20s} "
           f"{'Train SR':>10s} {'Val SR':>10s} "
           f"{'PR':>7s} {'Margin':>7s} {'ShiftExp':>10s} {'Full RoH':>10s}")
    print(hdr)
    print("-" * 120)

    for name, cfg in configs.items():
        cp = cfg["params"]
        fv = float(time_series[name]["final_value"])
        full_roh = fv / float(hodl_values[-1]) - 1
        print(
            f"{name:<28s} "
            f"{cfg.get('train_objective', float('nan')):>20.4f} "
            f"{cfg.get('test_objective', float('nan')):>20.4f} "
            f"{cfg.get('train_sharpe', float('nan')):>+10.4f} "
            f"{cfg.get('validation_sharpe', float('nan')):>+10.4f} "
            f"{cp.get('price_ratio', float('nan')):>7.3f} "
            f"{cp.get('centeredness_margin', float('nan')):>7.4f} "
            f"{cp.get('shift_exponent', float('nan')):>10.4g} "
            f"{full_roh * 100:>+9.2f}%"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()
