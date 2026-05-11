#!/usr/bin/env python3
"""Select best reClAMM params from sweep results.

Reads all result JSONs and corresponding log files for a given TVL config,
extracts train/val metrics, and picks the best param set.

Usage:
    python scripts/select_best_params.py --config aave_1m
    python scripts/select_best_params.py --config cow_500k --method cma_es
    python scripts/select_best_params.py --all  # summarise all configs
"""

import argparse
import json
import glob
import os
import re


SWEEP_DIR = "results/full_sweep"
LOG_DIR = "/tmp"

# TVL configs matching run_full_sweep.sh
CONFIGS = {
    "aave_1m":  {"tokens": ["AAVE", "ETH"], "pool_id": "0x9d1fcf346ea1b0", "gas_cost": 1.0, "fees": 0.0025, "initial_pool_value": 1_000_000},
    "aave_5m":  {"tokens": ["AAVE", "ETH"], "pool_id": "0x9d1fcf346ea1b0", "gas_cost": 1.0, "fees": 0.0025, "initial_pool_value": 5_000_000},
    "aave_20m": {"tokens": ["AAVE", "ETH"], "pool_id": "0x9d1fcf346ea1b0", "gas_cost": 1.0, "fees": 0.0025, "initial_pool_value": 20_000_000},
    "cow_500k": {"tokens": ["COW", "ETH"], "pool_id": "0xd321300ef77067", "gas_cost": 3.0, "fees": 0.003, "initial_pool_value": 500_000},
    "cow_2m":   {"tokens": ["COW", "ETH"], "pool_id": "0xd321300ef77067", "gas_cost": 3.0, "fees": 0.003, "initial_pool_value": 2_000_000},
    "cow_20m":  {"tokens": ["COW", "ETH"], "pool_id": "0xd321300ef77067", "gas_cost": 3.0, "fees": 0.003, "initial_pool_value": 20_000_000},
}


def parse_params(param_dict):
    """Normalise params from JSON — handles both Optuna (float) and CMA-ES (string array) formats."""
    out = {}
    for key in ("price_ratio", "centeredness_margin", "shift_exponent"):
        val = param_dict.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            # CMA-ES stores as "[1.234]"
            val = float(val.strip("[]"))
        out[key] = float(val)
    return out


def parse_log_metrics(log_path):
    """Extract train and val metrics from a tuning log file."""
    metrics = {}
    if not os.path.exists(log_path):
        return metrics
    with open(log_path) as f:
        text = f.read()

    # Find the final "Best trial" block
    m = re.search(
        r"Train \(IS\):\s*(.*?)\n.*?Val \(OOS\):\s*(.*?)(?:\n|$)",
        text, re.DOTALL,
    )
    if not m:
        return metrics

    for prefix, line in [("train_", m.group(1)), ("val_", m.group(2))]:
        for pair in re.findall(r"(\w+)=([+-]?\d+\.?\d*|[+-]?inf)", line):
            try:
                metrics[prefix + pair[0]] = float(pair[1])
            except ValueError:
                pass

    # Extract completed/failed counts
    cm = re.search(r"(\d+) completed.*?(\d+) failed", text)
    if cm:
        metrics["n_completed"] = int(cm.group(1))
        metrics["n_failed"] = int(cm.group(2))

    return metrics


def find_results(config_name, method="optuna"):
    """Find all result files for a given config and method."""
    method_tag = "_cmaes" if method == "cma_es" else ""
    pattern = os.path.join(SWEEP_DIR, f"*{method_tag}_{config_name}.json")
    results = []
    for json_path in sorted(glob.glob(pattern)):
        fname = os.path.basename(json_path).replace(".json", "")
        # Derive the log file name
        log_name = f"tune_full_{fname}.log"
        log_path = os.path.join(LOG_DIR, log_name)

        with open(json_path) as f:
            data = json.load(f)

        # The JSON has a single key = objective name
        obj_name = list(data.keys())[0]
        params = parse_params(data[obj_name])
        metrics = parse_log_metrics(log_path)

        results.append({
            "file": fname,
            "objective": obj_name,
            "params": params,
            "metrics": metrics,
        })
    return results


def rank_results(results, rank_by="val_returns_over_hodl"):
    """Rank results by a metric, handling missing/inf values."""
    def sort_key(r):
        v = r["metrics"].get(rank_by, float("-inf"))
        if v != v or v == float("-inf"):  # nan or -inf
            return float("-inf")
        return v
    return sorted(results, key=sort_key, reverse=True)


def print_summary(config_name, results, top_n=5):
    """Print a ranked summary table."""
    ranked = rank_results(results)
    print(f"\n{'='*80}")
    print(f"  {config_name} — {len(results)} results (ranked by val RoH)")
    print(f"{'='*80}")
    print(f"  {'Tag':<55s} {'PR':>6s} {'margin':>7s} {'shift':>8s}  {'train_roh':>10s} {'val_roh':>10s}")
    print(f"  {'-'*55} {'-'*6} {'-'*7} {'-'*8}  {'-'*10} {'-'*10}")
    for r in ranked[:top_n]:
        p = r["params"]
        m = r["metrics"]
        pr = f"{p.get('price_ratio', 0):.2f}"
        margin = f"{p.get('centeredness_margin', 0):.3f}"
        shift = f"{p.get('shift_exponent', 0):.4f}"
        train = m.get("train_ret_over_hodl", m.get("train_returns_over_hodl", float("nan")))
        val = m.get("val_returns_over_hodl", float("nan"))
        train_s = f"{train:+.4f}" if train == train else "N/A"
        val_s = f"{val:+.4f}" if val == val else "N/A"
        print(f"  {r['file']:<55s} {pr:>6s} {margin:>7s} {shift:>8s}  {train_s:>10s} {val_s:>10s}")

    if ranked:
        best = ranked[0]
        print(f"\n  BEST: {best['file']}")
        print(f"    PR={best['params'].get('price_ratio', '?'):.4f}  "
              f"margin={best['params'].get('centeredness_margin', '?'):.4f}  "
              f"shift={best['params'].get('shift_exponent', '?'):.6f}")
    return ranked


def export_best(config_name, ranked, method="optuna"):
    """Write the best params to a consolidated JSON for downstream scripts."""
    if not ranked:
        return
    best = ranked[0]
    method_tag = f"_{method}" if method != "optuna" else ""
    out = {
        "config": config_name,
        "method": method,
        "source": best["file"],
        "params": best["params"],
        "metrics": best["metrics"],
        **CONFIGS[config_name],
    }
    out_path = os.path.join(SWEEP_DIR, f"best{method_tag}_{config_name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="Config name (e.g. aave_1m)")
    p.add_argument("--all", action="store_true", help="Summarise all configs")
    p.add_argument("--method", default="optuna", choices=["optuna", "cma_es"])
    p.add_argument("--top", type=int, default=8, help="Show top N results")
    p.add_argument("--export", action="store_true", help="Export best params to JSON")
    args = p.parse_args()

    configs = list(CONFIGS.keys()) if args.all else [args.config]
    if not args.all and not args.config:
        p.error("Specify --config or --all")

    for config_name in configs:
        results = find_results(config_name, args.method)
        if not results:
            print(f"\n  {config_name}: no results found for method={args.method}")
            continue
        ranked = print_summary(config_name, results, top_n=args.top)
        if args.export:
            export_best(config_name, ranked, args.method)


if __name__ == "__main__":
    main()
