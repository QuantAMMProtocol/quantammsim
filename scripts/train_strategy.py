#!/usr/bin/env python3
"""Sweep hyperparameters while training a pre-canned quantammsim strategy.

Replaces the legacy JSONL-driven ``do_jax_historic_training.py`` pattern
with an all-Python sweep: the hyperparameter grid lives at the top of
this file, the ``--optimiser`` flag picks which grid to use, and the
script loops over every combination.

Why a sweep and not a single run: for each strategy/objective pair,
the right optimiser hyperparameters (learning rate, schedule, etc.) are
not obvious a priori. Sweeping is currently the cheapest way to get a
defensible picked-winner under the honest train/val/test protocol the
rest of the pipeline expects.

The ``--optimiser`` flag selects a method and the matching sweep grid:

    adam   → method="gradient_descent", sweeps lr / schedule / return_val
    optuna → method="optuna"          , sweeps n_trials / return_val
    cma-es → method="cma_es"          , sweeps sigma0 / generations / return_val
    l-bfgs → method="bfgs"            , sweeps maxiter / return_val

CLI args control what is *not* swept: dataset, strategy, dates, fees.
To change a sweep's axes, edit ``SWEEPS`` below — this is deliberately
done in Python (not JSON) so comments can explain the choices.

Examples
--------
Default sweep over Adam hyperparams for momentum on ETH/USDC:
    python scripts/train_strategy.py

CMA-ES sweep on a different pair, limited to first 3 combos:
    python scripts/train_strategy.py --optimiser cma-es --tokens BTC ETH --max-runs 3

Print the sweep combos without running anything:
    python scripts/train_strategy.py --dry-run
"""

import argparse
import itertools
import json
import math
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantammsim.runners.jax_runners import train_on_historic_data
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults


OPTIMISER_TO_METHOD = {
    "adam": "gradient_descent",
    "optuna": "optuna",
    "cma-es": "cma_es",
    "l-bfgs": "bfgs",
}


# ---------------------------------------------------------------------------
# Sweep grids — what we loop over.
#
# Structure:
#   CROSS_CUTTING        — fp-level axes we sweep in EVERY optimiser grid.
#                          These are the knobs that most directly attack the
#                          open generalisation problem (STE flags, per-step
#                          change cap, turnover penalty, price noise aug).
#   _<OPTIMISER>_SPECIFIC — axes meaningful only for that optimiser family.
#   SWEEPS               — merge of specific + cross-cutting per optimiser.
#
# Keys use dotted paths into the run_fingerprint dict. A list of scalars is
# a simple axis; a list of dicts is a coupled "recipe" axis (all keys in the
# dict are applied atomically for each variant).
#
# Combo count = cartesian product of all axes. Edit in place to tighten or
# widen. See --max-runs / --dry-run for quick smoke tests.
# ---------------------------------------------------------------------------
CROSS_CUTTING = {
    # Straight-through estimator flags — kept coupled (both on or both off)
    # because that matches how use_ste_gradients behaved in the legacy sweep.
    "ste": [
        {"ste_max_change": False, "ste_min_max_weight": False},
        {"ste_max_change": True,  "ste_min_max_weight": True},
    ],
    # Per-step weight change cap. 3e-4 is the default; 1000.0 effectively
    # disables the cap (matches legacy _x_layer.py's experimental value).
    "maximum_change":     [3e-4, 1000.0],
    # Regularisation / data-augmentation axes targeting generalisation.
    "turnover_penalty":   [0.0, 1e-3],
    "price_noise_sigma":  [0.0, 0.001],
}

_ADAM_SPECIFIC = {
    # Coupled: weight_decay=0 ↔ plain adam; non-zero ↔ adamw.
    "optimiser_recipe": [
        {"optimisation_settings.optimiser": "adam",  "optimisation_settings.weight_decay": 0.0},
        {"optimisation_settings.optimiser": "adamw", "optimisation_settings.weight_decay": 0.01},
    ],
    "optimisation_settings.base_lr":          [0.01, 0.1, 1.0],
    "optimisation_settings.lr_schedule_type": ["constant", "cosine"],
    "return_val":                             ["daily_log_sharpe", "calmar"],
}

_OPTUNA_SPECIFIC = {
    "optimisation_settings.optuna_settings.n_trials":            [50, 100],
    # Direct generalisation lever: penalise solutions where train ≫ val.
    "optimisation_settings.optuna_settings.overfitting_penalty": [0.0, 0.2, 1.0],
    # NOTE: expand_around=True is currently broken (non-scalar params produce
    # low > high bounds in create_trial_params). Pinned to False in
    # base_fingerprint until upstream fix; sweep axis dropped.
    "return_val":                                                ["daily_log_sharpe", "calmar"],
}

_CMA_ES_SPECIFIC = {
    # sigma0 is the single most sensitive CMA-ES knob.
    "optimisation_settings.cma_es_settings.sigma0":        [0.3, 0.5, 1.0],
    "optimisation_settings.cma_es_settings.n_generations": [200, 400],
    "return_val":                                          ["daily_log_sharpe", "calmar"],
}

_L_BFGS_SPECIFIC = {
    "optimisation_settings.bfgs_settings.maxiter": [50, 100, 200],
    # Multi-start perturbation — critical for BFGS since each start is
    # deterministic, so without variety you risk the same local minimum.
    "optimisation_settings.noise_scale":           [0.1, 0.3],
    "return_val":                                  ["daily_log_sharpe", "calmar"],
}

SWEEPS = {
    "adam":   {**_ADAM_SPECIFIC,   **CROSS_CUTTING},
    "optuna": {**_OPTUNA_SPECIFIC, **CROSS_CUTTING},
    "cma-es": {**_CMA_ES_SPECIFIC, **CROSS_CUTTING},
    "l-bfgs": {**_L_BFGS_SPECIFIC, **CROSS_CUTTING},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Which family of optimiser (selects sweep grid) ---
    p.add_argument(
        "--optimiser",
        choices=list(OPTIMISER_TO_METHOD.keys()),
        default="adam",
    )

    # --- Things that are NOT swept (controlled from CLI) ---
    p.add_argument("--rule", default="momentum")
    p.add_argument("--tokens", nargs="+", default=["ETH", "USDC"])
    # Dates match the demo-case window used in the legacy _x_layer.py sweep.
    p.add_argument("--start",    default="2023-06-01 00:00:00")
    p.add_argument("--end",      default="2025-06-01 00:00:00")
    p.add_argument("--test-end", default="2026-01-01 00:00:00")
    p.add_argument("--fees", type=float, default=0.0)
    p.add_argument("--initial-pool-value", type=float, default=1_000_000.0)
    p.add_argument("--minimum-weight", type=float, default=0.01,
                   help="Min per-asset weight. None falls back to 0.1/n_assets.")

    # --- Shared knobs (not worth sweeping but exposed for convenience) ---
    p.add_argument("--n-parameter-sets", type=int, default=4)
    p.add_argument("--batch-size",       type=int, default=8)
    p.add_argument("--iterations",       type=int, default=2000,
                   help="Adam only: training epochs.")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Held-out validation fraction of training window.")

    # --- Seeds: repeat each combo across seeds for robustness ---
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="Run each sweep combo once per seed (seed → initial_random_key).")

    # --- Sweep orchestration ---
    p.add_argument("--max-runs", type=int, default=None,
                   help="Cap total runs (after seed expansion). Useful for smoke tests.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned combos and exit without training.")
    p.add_argument("--process-id", type=int, default=0,
                   help="Parallel worker index for chunking this sweep across processes.")
    p.add_argument("--process-total", type=int, default=1,
                   help="Total number of parallel workers.")
    p.add_argument("--summary-out", default=None,
                   help="Path to write per-run JSONL summary. "
                        "Default: results/sweep_<optimiser>_<timestamp>.jsonl")
    p.add_argument("--force-init", action="store_true",
                   help="Ignore any cached results/<hash>.json and retrain.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Override all inner budgets to 1 (iterations/trials/generations/"
                        "maxiter) plus n_parameter_sets=1, batch_size=1, for end-to-end "
                        "pipeline smoke-testing without burning compute.")

    return p


def set_nested(d: dict, dotted_key: str, value) -> None:
    """Set d[a][b][c] = value for dotted_key 'a.b.c', creating dicts as needed."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def base_fingerprint(args: argparse.Namespace) -> dict:
    """Fingerprint carrying everything that does NOT vary across the sweep."""
    fp = deepcopy(run_fingerprint_defaults)

    fp["tokens"] = args.tokens
    fp["rule"] = args.rule
    fp["startDateString"] = args.start
    fp["endDateString"] = args.end
    fp["endTestDateString"] = args.test_end
    fp["fees"] = args.fees
    fp["initial_pool_value"] = args.initial_pool_value
    if args.minimum_weight is not None:
        fp["minimum_weight"] = args.minimum_weight

    opt = fp["optimisation_settings"]
    opt["method"] = OPTIMISER_TO_METHOD[args.optimiser]
    opt["n_parameter_sets"] = args.n_parameter_sets
    opt["batch_size"] = args.batch_size
    opt["val_fraction"] = args.val_fraction
    # Adam's iteration budget; ignored by other methods (which have own budgets).
    opt["n_iterations"] = args.iterations
    # Adam-family sets optimiser name explicitly; other methods ignore it.
    if args.optimiser == "adam":
        opt["optimiser"] = "adam"

    # Optuna: expand_around=True is broken in create_trial_params
    # (jax_runner_utils.py:152-167) — for non-scalar params it computes
    # low = val - config.low, high = val + config.high, which produces
    # low > high for some parameter shapes and causes every trial to FAIL
    # with "`low <= high` must hold". Pin to False until the upstream bug
    # is fixed. (make_scalar=True also sidesteps it by forcing the scalar
    # branch, but that's masking the real bug.)
    if args.optimiser == "optuna":
        opt["optuna_settings"]["expand_around"] = False

    return fp


def apply_smoke(fp: dict) -> None:
    """Force all inner budgets to 1 so each run is ~1 forward pass.

    Applied AFTER the sweep combo so it overrides any sweep axis that sets
    n_iterations / n_trials / n_generations / maxiter to larger values.
    """
    opt = fp["optimisation_settings"]
    opt["n_iterations"] = 1
    opt["n_parameter_sets"] = 1
    opt["batch_size"] = 1
    # Optuna wants > 1 trials to produce stable completed state; 3 is still
    # tiny (~3 forward passes) and reliably populates metrics.
    optuna_s = opt.setdefault("optuna_settings", {})
    optuna_s["n_trials"] = 3
    optuna_s["n_startup_trials"] = 3
    opt.setdefault("cma_es_settings", {})["n_generations"] = 1
    opt.setdefault("bfgs_settings", {})["maxiter"] = 1


def enumerate_combos(grid: dict) -> list:
    """Cartesian product over a dict-of-lists grid. Returns list of dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def apply_combo(fp: dict, combo: dict, seed: int) -> dict:
    """Return a new fingerprint = base fp + sweep combo overrides + seed.

    A combo value may be either a scalar (applied to the combo's axis key as
    a dotted path) or a dict (a coupled recipe applied atomically, where the
    axis key is just a label and the dict's keys are the real dotted paths).
    """
    fp_copy = deepcopy(fp)
    for axis_key, value in combo.items():
        if isinstance(value, dict):
            for dotted_key, v in value.items():
                set_nested(fp_copy, dotted_key, v)
        else:
            set_nested(fp_copy, axis_key, value)
    set_nested(fp_copy, "optimisation_settings.initial_random_key", seed)
    return fp_copy


def _fmt_value(axis_key, value) -> str:
    if isinstance(value, dict):
        inner = ", ".join(f"{k.rsplit('.', 1)[-1]}={v}" for k, v in value.items())
        return f"{axis_key}={{{inner}}}"
    short = axis_key.rsplit(".", 1)[-1]
    return f"{short}={value}"


def format_combo(combo: dict, seed: int) -> str:
    """Human-readable one-liner for logs."""
    bits = [f"seed={seed}"] + [_fmt_value(k, v) for k, v in combo.items()]
    return ", ".join(bits)


def chunk_runs(runs: list, process_id: int, process_total: int) -> list:
    """Deterministic balanced slice of runs for this worker process."""
    if process_total <= 1:
        return runs
    n = len(runs)
    base = n // process_total
    remainder = n % process_total
    if process_id < remainder:
        start = process_id * (base + 1)
        end = start + base + 1
    else:
        start = process_id * base + remainder
        end = start + base
    return runs[start:end]


def extract_summary(combo: dict, seed: int, result, duration_s: float) -> dict:
    """Flatten train_on_historic_data's return value into a summary row."""
    row = {"seed": seed, "duration_s": round(duration_s, 2), **combo}

    if result is None:
        row["status"] = "no_result"
        return row

    if isinstance(result, list):
        # Legacy shape (some methods w/o return_training_metadata): list of
        # trial dicts. Not expected on the current code path but kept for
        # safety.
        row["status"] = "ok_list"
        row["n_best_trials"] = len(result)
        return row

    # (params, metadata) tuple. For optuna with 0 completed trials, params
    # is None and metadata's metric fields are None.
    params, metadata = result
    row["status"] = "ok"
    row["best_param_idx"] = int(metadata.get("best_param_idx", 0))
    best_idx = row["best_param_idx"]

    if params is None:
        row["status"] = "optuna_empty"
        if metadata.get("error"):
            row["error"] = str(metadata["error"])

    for label, key in [("train", "best_train_metrics"),
                       ("val",   "best_val_metrics"),
                       ("test",  "best_continuous_test_metrics")]:
        metrics_list = metadata.get(key)
        if not metrics_list:
            continue
        m = metrics_list[best_idx] if best_idx < len(metrics_list) else metrics_list[0]
        for metric in ("sharpe", "daily_log_sharpe", "returns_over_uniform_hodl", "calmar"):
            if metric in m:
                v = m[metric]
                try:
                    row[f"{label}_{metric}"] = float(v)
                except (TypeError, ValueError):
                    pass

    # Flag NaN metrics: a "completed" run that produced NaN is not success.
    metric_keys = [k for k in row if k.startswith(("train_", "val_", "test_"))]
    if any(isinstance(row.get(k), float) and math.isnan(row[k]) for k in metric_keys):
        row["status"] = "nan_metrics"

    return row


def print_result_line(row: dict) -> None:
    """One-line per-run progress print."""
    status = row.get("status", "?")
    dur = row.get("duration_s", 0)
    val_sharpe = row.get("val_daily_log_sharpe") or row.get("val_sharpe")
    test_sharpe = row.get("test_daily_log_sharpe") or row.get("test_sharpe")
    bits = [f"{status:9s}", f"{dur:>6.1f}s"]
    if val_sharpe is not None:
        bits.append(f"val={val_sharpe:+.3f}")
    if test_sharpe is not None:
        bits.append(f"test={test_sharpe:+.3f}")
    print("  →", " ".join(bits))


def default_summary_path(optimiser: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"sweep_{optimiser}_{ts}.jsonl"


def main() -> None:
    args = build_parser().parse_args()

    grid = SWEEPS[args.optimiser]
    combos = enumerate_combos(grid)
    runs = [(combo, seed) for combo in combos for seed in args.seeds]
    runs = chunk_runs(runs, args.process_id, args.process_total)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    print("=" * 70)
    print(f"quantammsim sweep: {args.rule} on {'/'.join(args.tokens)}")
    print("=" * 70)
    print(f"Optimiser: {args.optimiser} (method={OPTIMISER_TO_METHOD[args.optimiser]})")
    print(f"Train:     {args.start} → {args.end}")
    print(f"Test:      {args.end} → {args.test_end}")
    print(f"Val frac:  {args.val_fraction}   n_parameter_sets: {args.n_parameter_sets}")
    print(f"Seeds:     {args.seeds}")
    print(f"Sweep axes: {list(grid.keys())}")
    total_planned = len(combos) * len(args.seeds)
    chunk_note = ""
    if args.process_total > 1:
        chunk_note = f", chunk {args.process_id}/{args.process_total}"
    if args.max_runs is not None:
        chunk_note += ", capped by --max-runs"
    print(f"Planned runs: {len(runs)}  (combos={len(combos)} × seeds={len(args.seeds)}"
          f" = {total_planned}{chunk_note})")
    print("=" * 70)

    if args.dry_run:
        for i, (combo, seed) in enumerate(runs):
            print(f"[{i+1:3d}] {format_combo(combo, seed)}")
        return

    # Heads-up if the grid is huge — real training takes minutes per run.
    if len(runs) > 50:
        print(f"[warn] {len(runs)} runs planned — this will take a while. "
              f"Use --max-runs, --dry-run, or edit SWEEPS to trim.")
        print()

    summary_path = Path(args.summary_out) if args.summary_out else default_summary_path(args.optimiser)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing per-run summary to: {summary_path}")
    if args.smoke:
        print("[smoke] Overriding inner budgets to 1 (n_iterations/n_trials/"
              "n_startup_trials/n_generations/maxiter); n_parameter_sets=1, batch_size=1. "
              "Sweep axes for those budgets are shown for reference only.")
    print()

    fp_base = base_fingerprint(args)
    verbose = not args.quiet
    wins = []

    tag = "[smoke] " if args.smoke else ""
    with summary_path.open("w") as summary_f:
        for i, (combo, seed) in enumerate(runs):
            print(f"{tag}[{i+1}/{len(runs)}] {format_combo(combo, seed)}")
            fp = apply_combo(fp_base, combo, seed)
            if args.smoke:
                apply_smoke(fp)
            t0 = time.monotonic()
            try:
                result = train_on_historic_data(
                    fp,
                    verbose=verbose,
                    force_init=args.force_init,
                    return_training_metadata=True,
                )
                row = extract_summary(combo, seed, result, time.monotonic() - t0)
            except Exception as e:
                row = {
                    "seed": seed,
                    "duration_s": round(time.monotonic() - t0, 2),
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=3),
                    **combo,
                }
            summary_f.write(json.dumps(row, default=str) + "\n")
            summary_f.flush()
            print_result_line(row)
            wins.append(row)

    # --- Post-sweep picker: best by validation sharpe (then test as tiebreaker) ---
    # Only consider real wins — exclude errors, NaN metrics, and empty optuna runs.
    ok = [r for r in wins if r.get("status") == "ok"]
    if ok:
        def score(r):
            return (
                r.get("val_daily_log_sharpe")
                or r.get("val_sharpe")
                or r.get("test_daily_log_sharpe")
                or r.get("test_sharpe")
                or -np.inf
            )
        best = max(ok, key=score)
        print("\n" + "=" * 70)
        print("BEST (by validation sharpe, test as fallback)")
        print("=" * 70)
        for k, v in best.items():
            print(f"  {k}: {v}")
        print(f"\nFull summary JSONL: {summary_path}")
    else:
        print("\n[No successful runs to rank]")


if __name__ == "__main__":
    main()
