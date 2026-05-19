#!/usr/bin/env python3
"""Run final forward simulations for best sweep params at multiple TVLs.

For each token pair, loads the best params from the sweep, runs train and
test period forward passes at each TVL level, and generates comparison plots.

Usage:
    python scripts/run_final_sims.py --pair aave
    python scripts/run_final_sims.py --pair cow
    python scripts/run_final_sims.py --all
    python scripts/run_final_sims.py --all --method cma_es
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from quantammsim.runners.jax_runners import do_run_on_historic_data
from scripts.evaluate_trials import load_reclamm_results
from scripts.plot_reclamm_optuna_result import (
    plot_results as plot_optuna_results,
    plot_test_only,
    plot_weights as plot_optuna_weights,
)

BG = "#162536"
TC = "#E6CE97"

# Train/test split around the Oct 10 flash crash
TRAIN_START = "2025-01-01 00:00:00"
TRAIN_END   = "2025-10-05 00:00:00"
TEST_START  = "2025-10-25 00:00:00"
TEST_END    = "2026-05-01 00:00:00"

PAIR_CONFIGS = {
    "aave": {
        "tokens": ["AAVE", "ETH"],
        "pool_id": "0x9d1fcf346ea1b0",
        "gas_cost": 1.0,
        "fees": 0.0025,
        "tvls": {
            "1m": 1_000_000,
            "5m": 5_000_000,
            "20m": 20_000_000,
        },
    },
    "cow": {
        "tokens": ["COW", "ETH"],
        "pool_id": "0xd321300ef77067",
        "gas_cost": 3.0,
        "fees": 0.003,
        "tvls": {
            "500k": 500_000,
            "2m": 2_000_000,
            "20m": 20_000_000,
        },
    },
}


def select_best_params(trials, tokens_set, tvl, metric_key="returns_over_hodl"):
    """From loaded trials, pick the best for a given token pair and TVL.

    Ranks by val (OOS) returns_over_hodl regardless of what objective the
    trial was trained on — this is the consistent comparison metric.
    """
    matching = [
        t for t in trials
        if tuple(sorted(t["tokens"])) == tokens_set
        and abs(t["initial_pool_value"] - tvl) < 1.0
    ]
    if not matching:
        return None

    def _get_val_roh(t):
        """Extract OOS returns_over_hodl from test_objective or continuous_test_metrics."""
        # Try continuous_test_metrics first (more reliable)
        ct = t.get("continuous_test_metrics", {})
        if isinstance(ct, dict) and "returns_over_hodl" in ct:
            return float(ct["returns_over_hodl"])
        # Fall back to test_objective
        to = t.get("test_objective", {})
        if isinstance(to, dict) and "returns_over_hodl" in to:
            return float(to["returns_over_hodl"])
        if isinstance(to, list):
            for entry in to:
                if isinstance(entry, dict) and "returns_over_hodl" in entry:
                    return float(entry["returns_over_hodl"])
        # If the trial's own objective IS returns_over_hodl, use test_value
        if t.get("return_val") == "returns_over_hodl":
            return float(t.get("test_value", float("-inf")))
        return float("-inf")

    matching.sort(key=_get_val_roh, reverse=True)
    best = matching[0]
    best_roh = _get_val_roh(best)
    print(f"  Selection: {len(matching)} candidates, best val_roh={best_roh:+.4f} "
          f"(trained on {best['return_val']})")
    return best


def build_fingerprint(pair_cfg, tvl, start, end, noise_path=None):
    """Build a run_fingerprint for a forward pass."""
    fp = {
        "rule": "reclamm",
        "tokens": pair_cfg["tokens"],
        "startDateString": start,
        "endDateString": end,
        "initial_pool_value": float(tvl),
        "do_arb": True,
        "arb_frequency": 4,
        "fees": pair_cfg["fees"],
        "gas_cost": pair_cfg["gas_cost"],
        "arb_fees": 0.0,
        "protocol_fee_split": 0.25,
        "noise_trader_ratio": 0.0,
        "noise_model": "mm_observed",
        "noise_arrays_path": noise_path,
    }
    return fp


def build_noise_arrays(pair_cfg, start, end):
    """Build or load cached noise arrays for the period."""
    from quantammsim.calibration.noise_model_arrays import build_mm_simulator_arrays
    tok_a, tok_b = pair_cfg["tokens"]
    pool_id = pair_cfg["pool_id"]
    tag = f"{pool_id}_{start.split()[0]}_{end.split()[0]}_mm"
    cache_path = f"results/mm_noise/_sim_arrays/{tag}.npz"

    if not os.path.exists(cache_path):
        print(f"  Building noise arrays: {tok_a}/{tok_b} {start} → {end}")
        arrays = build_mm_simulator_arrays(
            token_a=tok_a, token_b=tok_b,
            start_date=start.split()[0], end_date=end.split()[0],
            mm_artifact_dir="results/mm_noise",
            competitor_tvl_path="results/competitor_tvl/competitor_tvl.npz",
            pool_id=pool_id,
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path,
                 noise_base=arrays["noise_base"],
                 competitor_tvl=arrays["competitor_tvl"])
    return cache_path


def run_forward(pair_cfg, tvl, params, start, end, noise_path):
    """Run a single forward pass and return results dict."""
    fp = build_fingerprint(pair_cfg, tvl, start, end, noise_path)
    result = do_run_on_historic_data(
        run_fingerprint=fp, params=params, verbose=False,
    )
    return result


def _trial_hash(best):
    """Extract the originating run hash from a selected trial."""
    study_id = str(best.get("study_id", "unknown"))
    return study_id.removeprefix("run_")


def _unix_values_for_result(result):
    """Return the unix timestamp vector aligned to result['value']."""
    value_len = len(np.asarray(result["value"]))
    if "unix_values" in result:
        unix_values = np.asarray(result["unix_values"])
    else:
        data_dict = result.get("data_dict")
        if data_dict is None or "unix_values" not in data_dict:
            raise KeyError("Forward result has no unix_values or data_dict['unix_values']")
        start_idx = int(data_dict.get("start_idx", 0))
        unix_values = np.asarray(data_dict["unix_values"])[start_idx:start_idx + value_len]

    if len(unix_values) != value_len:
        raise ValueError(
            f"Timestamp/value length mismatch: unix={len(unix_values)} value={value_len}"
        )
    return unix_values.astype(np.int64)


def export_forward_csvs(result, run_fingerprint, output_dir, identifier, source_hash):
    """Write value, reserves, and per-token value CSVs for one forward pass."""
    value = np.asarray(result["value"], dtype=np.float64)
    reserves = np.asarray(result["reserves"], dtype=np.float64)
    prices = np.asarray(result["prices"], dtype=np.float64)
    unix_values = _unix_values_for_result(result)

    tokens = list(run_fingerprint["tokens"])
    if tokens != sorted(tokens):
        raise ValueError(
            "Final-sim CSV export assumes run_fingerprint['tokens'] is alphabetically "
            f"ordered to match runner price/reserve arrays; got {tokens}"
        )

    if reserves.shape != prices.shape:
        raise ValueError(f"Reserves/prices shape mismatch: {reserves.shape} vs {prices.shape}")
    if reserves.shape[0] != value.shape[0]:
        raise ValueError(f"Reserves/value length mismatch: {reserves.shape[0]} vs {value.shape[0]}")

    token_values = reserves * prices
    value_from_tokens = token_values.sum(axis=1)
    if not np.allclose(value_from_tokens, value, rtol=1e-8, atol=1e-6):
        max_diff = float(np.max(np.abs(value_from_tokens - value)))
        raise ValueError(
            f"Token value sanity check failed for {identifier}: max_diff={max_diff:.6g}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_stem = f"{identifier}_{source_hash}"

    pd.DataFrame({"unix": unix_values, "value": value}).to_csv(
        output_path / f"run_Value_{file_stem}.csv", index=False,
    )

    reserve_df = pd.DataFrame({"unix": unix_values})
    for idx, token in enumerate(tokens):
        reserve_df[f"reserve_{token}"] = reserves[:, idx]
    reserve_df.to_csv(output_path / f"run_Reserves_{file_stem}.csv", index=False)

    token_value_df = pd.DataFrame({"unix": unix_values})
    for idx, token in enumerate(tokens):
        token_value_df[f"{token}_value"] = token_values[:, idx]
    token_value_df.to_csv(output_path / f"run_TokenValues_{file_stem}.csv", index=False)
    print(f"  Saved CSVs: run_{{Value,Reserves,TokenValues}}_{file_stem}.csv")


def make_plot_data(all_results, pair_cfg, start, end):
    """Convert our results into the format expected by plot_reclamm_optuna_result functions.

    All values are normalised to start at 1.0 for cross-TVL comparability.
    Fee revenue is expressed as fraction of initial pool value.
    Weights are computed from reserves × prices (effective weight of token 0).

    Returns (configs, time_series, hodl_values, ref_config).
    """
    from collections import OrderedDict
    configs = OrderedDict()
    time_series = {}

    start_str = start.split()[0]
    end_str = end.split()[0]
    ref_config = {
        "tokens": pair_cfg["tokens"],
        "startDateString": start,
        "endDateString": end,
    }

    # Normalised HODL: use the first result's initial reserves
    first_result = next(iter(all_results.values()))["result"]
    prices = np.array(first_result["prices"])
    reserves_0 = np.array(first_result["reserves"][0])
    hodl_raw = np.sum(reserves_0 * prices, axis=1)
    hodl_values = hodl_raw / hodl_raw[0]  # normalise to 1.0

    for tvl_label, data in all_results.items():
        result = data["result"]
        val = np.array(result["value"])
        initial_val = val[0]
        reserves = np.array(result["reserves"])
        res_prices = np.array(result["prices"])

        # Compute effective weights from reserves × prices
        token_values = reserves * res_prices  # (T, 2)
        total_value = token_values.sum(axis=1, keepdims=True)
        weights = token_values / np.maximum(total_value, 1e-30)

        # Normalise fee revenue as fraction of initial TVL
        fee_rev = np.array(result.get("fee_revenue", np.zeros(len(val))))
        fee_rev_normalised = fee_rev / initial_val

        p = data["params"]
        pr = float(jnp.asarray(p.get("price_ratio", 0)).flatten()[0])
        margin = float(jnp.asarray(p.get("centeredness_margin", 0)).flatten()[0])
        shift = float(jnp.asarray(p.get("shift_exponent", 0)).flatten()[0])
        name = f"Auto-range ${tvl_label} (PR={pr:.2f}, m={margin:.2f}, s={shift:.3f})"
        configs[name] = {
            "tvl_label": tvl_label,
            "params": p,
        }
        time_series[name] = {
            "value": val / initial_val,  # normalise to 1.0
            "fee_revenue": fee_rev_normalised,
            "reserves": reserves,
            "prices": res_prices,
            "initial_tvl": initial_val,  # for volume normalisation
            "weights": weights,
        }

    return configs, time_series, hodl_values, ref_config


def run_pair(pair_name, pair_cfg, trials, output_dir):
    """Run all TVL variants for a pair, for train and test periods."""
    tokens = pair_cfg["tokens"]
    tokens_set = tuple(sorted(tokens))
    print(f"\n{'='*60}")
    print(f"  {'/'.join(tokens)} — {pair_name}")
    print(f"{'='*60}")

    # Build noise arrays for both periods
    train_noise = build_noise_arrays(pair_cfg, TRAIN_START, TRAIN_END)
    test_noise = build_noise_arrays(pair_cfg, TEST_START, TEST_END)

    train_results = {}
    test_results = {}

    for tvl_label, tvl in pair_cfg["tvls"].items():
        full_label = f"{pair_name}_{tvl_label}"
        print(f"\n  --- {full_label} (TVL=${tvl:,.0f}) ---")

        # Select best params for this TVL
        best = select_best_params(trials, tokens_set, tvl)
        if best is None:
            print(f"  No results found for {tokens_set} TVL={tvl}")
            continue

        params = dict(best["params"])
        pr = float(jnp.asarray(params.get("price_ratio", 0)).flatten()[0])
        margin = float(jnp.asarray(params.get("centeredness_margin", 0)).flatten()[0])
        shift = float(jnp.asarray(params.get("shift_exponent", 0)).flatten()[0])
        print(f"  Best: obj={best['return_val']}  train={best['train_value']:+.4f}  test={best['test_value']:+.4f}")
        print(f"  Params: PR={pr:.3f}  margin={margin:.4f}  shift={shift:.6f}")
        print(f"  Source: {best['study_id']}")

        # Ensure initial_weights_logits exists
        if "initial_weights_logits" not in params:
            params["initial_weights_logits"] = jnp.zeros(len(tokens))

        # Train period forward pass
        print(f"  Running train period...")
        train_result = run_forward(pair_cfg, tvl, params, TRAIN_START, TRAIN_END, train_noise)
        source_hash = _trial_hash(best)
        train_fp = build_fingerprint(pair_cfg, tvl, TRAIN_START, TRAIN_END, train_noise)
        export_forward_csvs(
            train_result, train_fp, output_dir,
            identifier=f"{pair_name}_{tvl_label}_train",
            source_hash=source_hash,
        )
        train_results[tvl_label] = {"result": train_result, "params": params, "best": best}

        # Clear JIT caches between runs to manage memory
        jax.clear_caches()

        # Test period forward pass
        print(f"  Running test period...")
        test_result = run_forward(pair_cfg, tvl, params, TEST_START, TEST_END, test_noise)
        test_fp = build_fingerprint(pair_cfg, tvl, TEST_START, TEST_END, test_noise)
        export_forward_csvs(
            test_result, test_fp, output_dir,
            identifier=f"{pair_name}_{tvl_label}_test",
            source_hash=source_hash,
        )
        test_results[tvl_label] = {"result": test_result, "params": params, "best": best}

        jax.clear_caches()

        # Print summary
        train_val = np.array(train_result["value"])
        test_val = np.array(test_result["value"])
        train_hodl = np.sum(np.array(train_result["reserves"][0]) * np.array(train_result["prices"]), axis=1)
        test_hodl = np.sum(np.array(test_result["reserves"][0]) * np.array(test_result["prices"]), axis=1)
        train_roh = train_val[-1] / train_hodl[-1] - 1
        test_roh = test_val[-1] / test_hodl[-1] - 1
        train_fee = float(np.array(train_result.get("fee_revenue", np.zeros(1))).sum())
        test_fee = float(np.array(test_result.get("fee_revenue", np.zeros(1))).sum())
        print(f"  Train RoH: {train_roh:+.2%}  Fee: ${train_fee:,.0f}")
        print(f"  Test  RoH: {test_roh:+.2%}  Fee: ${test_fee:,.0f}")

    # Generate plots using existing plotting functions
    for period_name, results_dict, start, end in [
        ("train", train_results, TRAIN_START, TRAIN_END),
        ("test", test_results, TEST_START, TEST_END),
    ]:
        if not results_dict:
            continue
        configs, ts, hodl, ref_cfg = make_plot_data(results_dict, pair_cfg, start, end)
        # Create a simple args object for the plot functions
        class PlotArgs:
            output = os.path.join(output_dir, f"{pair_name}_{period_name}.png")
        plot_args = PlotArgs()
        plot_optuna_results(configs, ts, hodl, ref_cfg, plot_args)
        plot_optuna_weights(configs, ts, ref_cfg, plot_args)

    # Save results for later use
    cache_path = os.path.join(output_dir, f"{pair_name}_sim_results.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({"train": train_results, "test": test_results}, f)
    print(f"  Saved cache: {cache_path}")

    return train_results, test_results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", choices=["aave", "cow"], default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--method", default="optuna", choices=["optuna", "cma_es"],
                   help="Which sweep method results to use")
    p.add_argument("--output-dir", default="results/final_sims")
    p.add_argument("--metric", default="returns_over_hodl",
                   help="Metric to rank trials by for param selection")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = list(PAIR_CONFIGS.keys()) if args.all else [args.pair]
    if not args.all and not args.pair:
        p.error("Specify --pair or --all")

    # Load all trials
    print("Loading sweep results...")
    trials = load_reclamm_results("./results/", metric_key=args.metric)

    # Filter to our training window
    trials = [
        t for t in trials
        if t.get("start_date", "").startswith("2025-01-01")
        and "2025-10-05" in t.get("end_date", "")
    ]
    print(f"  {len(trials)} trials from Jan-Oct 2025 sweep")

    for pair_name in pairs:
        pair_cfg = PAIR_CONFIGS[pair_name]
        run_pair(pair_name, pair_cfg, trials, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
