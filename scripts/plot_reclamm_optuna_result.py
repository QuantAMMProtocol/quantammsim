#!/usr/bin/env python3
"""Plot reClAMM pool performance from Optuna tuning results.

Reads SGD-compatible JSON output(s) of tune_reclamm_params.py, extracts the
best trial's pool params, re-runs a forward pass over the full train+test
window, and produces a value-over-time plot with on-chain baselines and
cumulative fee revenue.

Usage:
    # Single result
    python scripts/plot_reclamm_optuna_result.py results/run_<hash>.json

    # Multiple results (comparison across objectives / noise models)
    python scripts/plot_reclamm_optuna_result.py results/run_*.json

    # Top-3 trials from each result
    python scripts/plot_reclamm_optuna_result.py results/run_*.json --top-k 3
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
# Extended palette for multi-file comparison
COLORS = [
    "#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#2980b9", "#c0392b", "#8e44ad",
    "#27ae60", "#d35400", "#16a085", "#f1c40f", "#7f8c8d",
]

# Short labels for objectives
_OBJ_SHORT = {
    "daily_log_sharpe": "sharpe",
    "returns_over_hodl": "ret/hodl",
    "fee_revenue_over_value": "fee_rev",
}


def _plot_order(configs):
    """Yield (name, meta, color_idx) with baselines first, optimized trials last."""
    optimized = []
    baselines = []
    for i, (name, meta) in enumerate(configs.items()):
        if "On-Chain" in name:
            baselines.append((name, meta, i))
        else:
            optimized.append((name, meta, i))
    return baselines + optimized


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("results_json", nargs="+",
                    help="Path(s) to run_<hash>.json from Optuna")
    p.add_argument("--top-k", type=int, default=1,
                   help="Plot top K trials per result file (default 1)")
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: auto-generated)")
    p.add_argument("--no-onchain", action="store_true",
                   help="Skip on-chain baseline runs")
    p.add_argument("--end-test-date", default=None,
                   help="Override endTestDateString (e.g. '2026-02-15 00:00:00')")
    p.add_argument("--noise-trader-ratio", type=float, default=None,
                   help="Override noise_trader_ratio from results config")
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


def _noise_model_label(config):
    """Short label describing the noise model in the config."""
    nm = config.get("noise_model", "ratio")
    if nm != "calibrated":
        ntr = config.get("noise_trader_ratio", 0.0)
        return f"{nm}(ntr={ntr})"
    nc = config.get("reclamm_noise_params", {})
    n_coeffs = len(nc)
    arb_freq = config.get("arb_frequency", 1)
    return f"cal-{n_coeffs}cov(af={arb_freq})"


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
        "noise_trader_ratio": config.get("noise_trader_ratio", 0.0),
        "reclamm_use_shift_exponent": config.get("reclamm_use_shift_exponent", True),
        "reclamm_interpolation_method": config.get("reclamm_interpolation_method", "geometric"),
        "reclamm_centeredness_scaling": config.get("reclamm_centeredness_scaling", False),
        "reclamm_learn_arc_length_speed": config.get("reclamm_learn_arc_length_speed", False),
    }
    # Forward noise model settings
    if "noise_model" in config:
        fp["noise_model"] = config["noise_model"]
    if "reclamm_noise_params" in config:
        fp["reclamm_noise_params"] = config["reclamm_noise_params"]
    if "noise_arrays_path" in config:
        fp["noise_arrays_path"] = config["noise_arrays_path"]
    if "arb_frequency" in config:
        fp["arb_frequency"] = config["arb_frequency"]
    jax_params = {k: jnp.array(v) for k, v in params.items()}
    return do_run_on_historic_data(run_fingerprint=fp, params=jax_params)


def plot_results(configs, time_series, hodl_values, ref_config, args):
    """Two-panel plot: value-over-time + cumulative fee revenue."""
    train_end_str = ref_config["endDateString"]
    train_end_dt = datetime.strptime(train_end_str, "%Y-%m-%d %H:%M:%S")

    first_out = next(iter(time_series.values()))
    n_minutes = len(first_out["value"])
    dates = pd.date_range(
        start=datetime.strptime(ref_config["startDateString"], "%Y-%m-%d %H:%M:%S"),
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
    for name, meta, ci in _plot_order(configs):
        out = time_series[name]
        vals = np.array(out["value"][::step]) / 1e6
        label = f"{name}"
        if "test_objective" in meta:
            obj_name = meta.get("obj_name", "objective")
            label += f" (OOS {obj_name}={meta['test_objective']:.4f})"
        is_optimized = "On-Chain" not in name
        ax_val.plot(dates_daily[:len(vals)], vals,
                    linewidth=2.5 if is_optimized else 1.8,
                    color=COLORS[ci % len(COLORS)], label=label,
                    zorder=3 if is_optimized else 2)

    hodl_daily = hodl_values[::step] / 1e6
    ax_val.plot(dates_daily[:len(hodl_daily)], hodl_daily, linewidth=2,
                color="white", alpha=0.7, linestyle="--", label="HODL")

    ax_val.axvline(x=train_end_dt, color="white", linestyle=":", alpha=0.5, linewidth=1.5)
    ylims = ax_val.get_ylim()
    ax_val.text(train_end_dt - pd.Timedelta(days=5), ylims[1] * 0.97, "Train",
                color="white", alpha=0.6, fontsize=11, ha="right", va="top")
    ax_val.text(train_end_dt + pd.Timedelta(days=5), ylims[1] * 0.97, "Test",
                color="white", alpha=0.6, fontsize=11, ha="left", va="top")

    _style_axis(ax_val)
    ax_val.set_ylabel("Pool Value ($M USD)", color=TEXT_COLOR, fontsize=12)
    tokens_str = "/".join(ref_config["tokens"])
    ax_val.set_title(
        f"reClAMM Optuna Comparison — {tokens_str}",
        color=TEXT_COLOR, fontsize=13, pad=15,
    )
    ax_val.legend(loc="upper left", fontsize=8, facecolor=BG,
                  edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)

    # ── Panel 2: Cumulative fee revenue ───────────────────────────────
    if has_fee_revenue:
        ax_fee = axes[1]
        for name, _meta, ci in _plot_order(configs):
            out = time_series[name]
            fr = out.get("fee_revenue")
            if fr is None:
                continue
            fr = np.array(fr)
            cumfee = np.cumsum(fr)[::step] / 1e3
            is_optimized = "On-Chain" not in name
            ax_fee.plot(dates_daily[:len(cumfee)], cumfee,
                        linewidth=2.5 if is_optimized else 1.8,
                        color=COLORS[ci % len(COLORS)], label=name,
                        zorder=3 if is_optimized else 2)

        ax_fee.axvline(x=train_end_dt, color="white", linestyle=":", alpha=0.5, linewidth=1.5)
        _style_axis(ax_fee)
        ax_fee.set_ylabel("Cumulative Fee Revenue ($K)", color=TEXT_COLOR, fontsize=12)
        ax_fee.set_xlabel("Date", color=TEXT_COLOR, fontsize=12)
        ax_fee.legend(loc="upper left", fontsize=8, facecolor=BG,
                      edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)
    else:
        ax_val.set_xlabel("Date", color=TEXT_COLOR, fontsize=12)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()

    output = args.output or f"reclamm_optuna_{tokens_str.replace('/', '_')}.png"
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved plot to {output}")
    plt.close()


def plot_test_only(configs, time_series, hodl_values, ref_config, args):
    """Test-period plot with all curves normalised to start at 1.0."""
    train_end_str = ref_config["endDateString"]
    train_end_dt = datetime.strptime(train_end_str, "%Y-%m-%d %H:%M:%S")
    start_dt = datetime.strptime(ref_config["startDateString"], "%Y-%m-%d %H:%M:%S")

    first_out = next(iter(time_series.values()))
    n_minutes = len(first_out["value"])
    dates = pd.date_range(start=start_dt, periods=n_minutes, freq="1min")

    # Find the index of the train/test boundary
    train_minutes = int((train_end_dt - start_dt).total_seconds() / 60)
    test_start_idx = min(train_minutes, n_minutes - 1)

    step = 1440
    test_dates = dates[test_start_idx::step]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    for name, _meta, ci in _plot_order(configs):
        out = time_series[name]
        vals = np.array(out["value"])
        test_vals = vals[test_start_idx::step]
        if len(test_vals) == 0:
            continue
        normalised = test_vals / test_vals[0]
        is_optimized = "On-Chain" not in name
        ax.plot(test_dates[:len(normalised)], normalised,
                linewidth=2.5 if is_optimized else 1.8,
                color=COLORS[ci % len(COLORS)], label=name,
                zorder=3 if is_optimized else 2)

    hodl_test = hodl_values[test_start_idx::step]
    if len(hodl_test) > 0:
        hodl_norm = hodl_test / hodl_test[0]
        ax.plot(test_dates[:len(hodl_norm)], hodl_norm, linewidth=2,
                color="white", alpha=0.7, linestyle="--", label="HODL")

    ax.axhline(1.0, color="white", linestyle=":", alpha=0.3, linewidth=1)
    _style_axis(ax)
    tokens_str = "/".join(ref_config["tokens"])
    ax.set_title(f"Test Period Only (normalised) — {tokens_str}",
                 color=TEXT_COLOR, fontsize=13, pad=15)
    ax.set_ylabel("Normalised Value", color=TEXT_COLOR, fontsize=12)
    ax.set_xlabel("Date", color=TEXT_COLOR, fontsize=12)
    ax.legend(loc="best", fontsize=8, facecolor=BG,
              edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    base = (args.output or f"reclamm_optuna_{tokens_str.replace('/', '_')}.png")
    output = base.replace(".png", "_test_only.png")
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved plot to {output}")
    plt.close()


def plot_weights(configs, time_series, ref_config, args):
    """Effective weight (value fraction) of token 0 over time."""
    start_dt = datetime.strptime(ref_config["startDateString"], "%Y-%m-%d %H:%M:%S")
    train_end_dt = datetime.strptime(ref_config["endDateString"], "%Y-%m-%d %H:%M:%S")

    first_out = next(iter(time_series.values()))
    n_minutes = len(first_out["value"])
    dates = pd.date_range(start=start_dt, periods=n_minutes, freq="1min")
    step = 1440
    dates_daily = dates[::step]

    token_name = ref_config["tokens"][0]

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    for name, _meta, ci in _plot_order(configs):
        out = time_series[name]
        weights = np.array(out["weights"])  # (T, 2)
        w0 = weights[::step, 0]
        is_optimized = "On-Chain" not in name
        ax.plot(dates_daily[:len(w0)], w0,
                linewidth=2.0 if is_optimized else 1.5,
                color=COLORS[ci % len(COLORS)], label=name,
                alpha=0.9 if is_optimized else 0.7,
                zorder=3 if is_optimized else 2)

    ax.axhline(0.5, color="white", linestyle="--", alpha=0.3, linewidth=1)
    ax.axvline(x=train_end_dt, color="white", linestyle=":", alpha=0.5, linewidth=1.5)
    ylims = ax.get_ylim()
    ax.text(train_end_dt - pd.Timedelta(days=5), ylims[1] * 0.97, "Train",
            color="white", alpha=0.6, fontsize=11, ha="right", va="top")
    ax.text(train_end_dt + pd.Timedelta(days=5), ylims[1] * 0.97, "Test",
            color="white", alpha=0.6, fontsize=11, ha="left", va="top")

    _style_axis(ax)
    tokens_str = "/".join(ref_config["tokens"])
    ax.set_title(f"Effective {token_name} Weight — {tokens_str}",
                 color=TEXT_COLOR, fontsize=13, pad=15)
    ax.set_ylabel(f"{token_name} weight (value fraction)", color=TEXT_COLOR, fontsize=12)
    ax.set_xlabel("Date", color=TEXT_COLOR, fontsize=12)
    ax.legend(loc="best", fontsize=8, facecolor=BG,
              edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR)

    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    base = (args.output or f"reclamm_optuna_{tokens_str.replace('/', '_')}.png")
    output = base.replace(".png", "_weights.png")
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved plot to {output}")
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

    # ── Load all result files ─────────────────────────────────────────
    all_loaded = []
    for path in args.results_json:
        config, trials = load_results(path)
        if args.end_test_date:
            config["endTestDateString"] = args.end_test_date
        if args.noise_trader_ratio is not None:
            config["noise_trader_ratio"] = args.noise_trader_ratio
        all_loaded.append((path, config, trials))

    # Use first file's config as reference for dates/tokens
    ref_config = all_loaded[0][1]
    tokens = ref_config["tokens"]

    print("=" * 100)
    print(f"reClAMM Optuna Result Plotter  —  {len(all_loaded)} result file(s)")
    print("=" * 100)
    print(f"  Tokens:   {'/'.join(tokens)}")
    print(f"  Train:    {ref_config['startDateString']} → {ref_config['endDateString']}")
    print(f"  Test:     {ref_config['endDateString']} → {ref_config['endTestDateString']}")

    # ── Build configs dict from all files ─────────────────────────────
    configs = {}
    for path, config, trials in all_loaded:
        obj_name = config.get("return_val", "objective")
        obj_short = _OBJ_SHORT.get(obj_name, obj_name)
        noise_label = _noise_model_label(config)

        trials_sorted = sorted(trials, key=lambda t: t.get("objective", 0), reverse=True)
        top_trials = trials_sorted[:args.top_k]

        for i, trial in enumerate(top_trials):
            params = extract_pool_params(trial, config)
            rank_suffix = f" r{i+1}" if args.top_k > 1 else ""
            name = f"{obj_short} {noise_label}{rank_suffix}"
            configs[name] = {
                "params": params,
                "config": config,  # per-file config for noise model
                "objective": trial.get("objective", 0),
                "train_objective": trial.get("train_objective", 0),
                "test_objective": trial.get("test_objective", 0),
                "train_sharpe": trial.get("train_sharpe", 0),
                "validation_sharpe": trial.get("validation_sharpe", 0),
                "obj_name": obj_name,
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
        configs["On-Chain (launch)"] = {
            "params": dict(ONCHAIN_LAUNCH_PARAMS),
            "config": ref_config,
        }
        configs["On-Chain (current)"] = {
            "params": dict(ONCHAIN_CURRENT_PARAMS),
            "config": ref_config,
        }

    # ── Full-period runs ──────────────────────────────────────────────
    print(f"\n--- Running full-period simulations ({ref_config['startDateString']} → "
          f"{ref_config['endTestDateString']}) ---")
    time_series = {}
    for name, cfg in configs.items():
        print(f"  {name}...", end=" ", flush=True)
        run_config = cfg.get("config", ref_config)
        out = run_full_period(cfg["params"], run_config)
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

    # ── Plots ─────────────────────────────────────────────────────────
    plot_results(configs, time_series, hodl_values, ref_config, args)
    plot_test_only(configs, time_series, hodl_values, ref_config, args)
    plot_weights(configs, time_series, ref_config, args)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"SUMMARY  —  {'/'.join(tokens)}")
    print(f"{'=' * 130}")
    hdr = (f"{'Config':<35s} {'Objective':>12s} {'Train':>10s} {'Test':>10s} "
           f"{'Train SR':>10s} {'Val SR':>10s} "
           f"{'PR':>7s} {'Margin':>7s} {'ShiftExp':>10s} {'Full RoH':>10s}")
    print(hdr)
    print("-" * 130)

    for name, cfg in configs.items():
        cp = cfg["params"]
        fv = float(time_series[name]["final_value"])
        full_roh = fv / float(hodl_values[-1]) - 1
        print(
            f"{name:<35s} "
            f"{cfg.get('obj_name', ''):>12s} "
            f"{cfg.get('train_objective', float('nan')):>10.4f} "
            f"{cfg.get('test_objective', float('nan')):>10.4f} "
            f"{cfg.get('train_sharpe', float('nan')):>+10.4f} "
            f"{cfg.get('validation_sharpe', float('nan')):>+10.4f} "
            f"{cp.get('price_ratio', float('nan')):>7.3f} "
            f"{cp.get('centeredness_margin', float('nan')):>7.4f} "
            f"{cp.get('shift_exponent', float('nan')):>10.4g} "
            f"{full_roh * 100:>+9.2f}%"
        )
    print("=" * 130)


if __name__ == "__main__":
    main()
