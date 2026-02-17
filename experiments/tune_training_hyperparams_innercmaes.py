#!/usr/bin/env python3
"""
Hyperparameter Tuning with Inner CMA-ES Optimization
======================================================

This script uses CMA-ES as the inner optimizer, with outer Optuna searching
over settings that shape the fitness landscape and restart strategy.

Uses power_channel rule: a simpler strategy than mean_reversion_channel with
only 6 learnable params (k, lambda, delta_lambda, exponents, pre_exp_scaling,
weights_logits). CMA-ES handles ~10 params comfortably — its sweet spot is
5-50 parameters with expensive evaluations.

Why CMA-ES?
-----------
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a derivative-free
optimizer designed for:
- **Expensive black-box evaluations**: Each forward pass costs ~23ms, and
  CMA-ES needs only forward passes (no backward pass), so each evaluation
  is ~2x cheaper than BFGS.
- **Non-convex landscapes**: The population naturally explores multiple
  basins. The covariance matrix adapts to the local curvature, giving
  quasi-Newton-like efficiency without computing gradients.
- **Essentially zero hyperparameters**: Population size and sigma0 have
  robust defaults from theory. The algorithm self-tunes learning rates,
  step sizes, and covariance adaptation.

What to tune (outer Optuna search):
------------------------------------
CMA-ES has fewer knobs than BFGS/SGD, so the search space is smaller:

CMA-ES-specific (~4D):
  - cma_es_n_evaluation_points: Fitness averaging (5-50)
  - cma_es_n_generations: Budget per restart (50-500)
  - cma_es_sigma0: Initial step size (0.1-2.0) — the ONE CMA-ES hyperparameter
  - n_parameter_sets: Number of independent restarts (1-4)

Training window / constraints (~4D):
  - bout_offset_days: Window timing
  - val_fraction: Validation holdout
  - maximum_change: Weight rate limiter
  - minimum_weight: Portfolio weight floor

Initial param center (~4D):
  - initial_k_per_day: Momentum sensitivity
  - initial_memory_length: EWMA lookback
  - initial_raw_exponents: Power-law shape
  - initial_pre_exp_scaling: Gradient normalisation

Note: noise_scale and parameter_init_method still matter (they control
the diversity of starting points for each restart), but sigma0 partially
subsumes their role — CMA-ES will explore away from the init regardless.

Usage:
------
python experiments/tune_training_hyperparams_innercmaes.py
python experiments/tune_training_hyperparams_innercmaes.py --quick
python experiments/tune_training_hyperparams_innercmaes.py -n 100 -c 6 --objective mean_oos_sharpe
"""

import sys
import os
import json
import argparse
import numpy as np
import jax
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantammsim.runners.hyperparam_tuner import (
    HyperparamTuner,
    HyperparamSpace,
    TuningResult,
    OUTER_TO_INNER_METRIC,
)
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults


# =============================================================================
# Configuration
# =============================================================================

TOKENS = ["ETH", "USDC"]

START_DATE = "2021-01-01 00:00:00"
WFA_END_DATE = "2025-01-01 00:00:00"
HOLDOUT_END_DATE = "2026-01-01 00:00:00"

RULE = "power_channel"
INITIAL_POOL_VALUE = 1_000_000.0
FEES = 0.0
ARB_FEES = 0.0

STUDY_DIR = Path(__file__).parent / "hyperparam_studies"
STUDY_NAME = "eth_usdc_innercmaes_v1"


# =============================================================================
# Search Space
# =============================================================================

def create_search_space(cycle_days: int = 180) -> HyperparamSpace:
    """
    Create search space for CMA-ES inner optimization of power_channel.

    Three groups of parameters:
    1. CMA-ES-specific: fitness definition (n_evaluation_points), budget
       (n_generations), and the one real CMA-ES hyperparameter (sigma0).
    2. Multi-start / restart strategy: n_parameter_sets controls independent
       restarts with different initializations.
    3. Training window and strategy constraints: shared across all inner
       methods, affect the landscape itself.

    Parameters
    ----------
    cycle_days : int
        WFA cycle length in days (for bout_offset range).
    """
    space = HyperparamSpace()

    # ======================================================================
    # CMA-ES-specific settings
    # ======================================================================
    # n_evaluation_points: how many fixed windows form the deterministic
    # fitness. Same role as in BFGS — controls bias-variance of the
    # objective. CMA-ES evaluates pop_size × n_eval_points forward passes
    # per generation, so this directly affects wall-clock time.
    space.params["cma_es_n_evaluation_points"] = {
        "low": 5, "high": 50, "log": False, "type": "int",
    }

    # n_generations: maximum generations per restart. CMA-ES typically
    # converges in 100-300 generations for n=10 (empirical from Hansen).
    # should_stop will terminate early if the distribution collapses.
    space.params["cma_es_n_generations"] = {
        "low": 50, "high": 500, "log": False, "type": "int",
    }

    # sigma0: initial step size. THE one CMA-ES hyperparameter.
    # Too small → stuck near init (slow adaptation).
    # Too large → wastes generations exploring irrelevant regions.
    # Rule of thumb: ~1/4 of the expected distance to the optimum.
    # For our squareplus-parameterised strategies, params live on O(1) scale.
    space.params["cma_es_sigma0"] = {
        "low": 0.1, "high": 2.0, "log": True, "type": "float",
    }

    # ======================================================================
    # Multi-start / initialization
    # ======================================================================
    # n_parameter_sets = number of independent CMA-ES restarts.
    # Each gets a different init (set 0 = canonical, rest = noisy).
    # CMA-ES explores within each restart via population, so fewer restarts
    # needed than BFGS — but restarts still help with widely separated basins.
    space.params["n_parameter_sets"] = {
        "low": 1, "high": 4, "log": False, "type": "int",
    }

    # noise_scale: std of Gaussian perturbation to initial params for
    # restarts 1+ (restart 0 is always canonical). Less critical for CMA-ES
    # than BFGS since sigma0 controls exploration, but still affects which
    # basin each restart starts in.
    space.params["noise_scale"] = {
        "low": 0.05, "high": 1.0, "log": True, "type": "float",
    }

    # ======================================================================
    # Training window / constraints
    # ======================================================================
    max_val_fraction = 0.3
    max_offset = max(1, int(cycle_days * (1 - max_val_fraction) * 4 / 5))
    space.params["bout_offset_days"] = {
        "low": 0, "high": max_offset, "log": False, "type": "int",
    }

    space.params["val_fraction"] = {
        "low": 0.1, "high": max_val_fraction, "log": False, "type": "float",
    }

    space.params["maximum_change"] = {
        "low": 3e-5, "high": 2.0, "log": True, "type": "float",
    }

    space.params["minimum_weight"] = {
        "low": 0.01, "high": 0.1, "log": True, "type": "float",
    }

    # ======================================================================
    # Initial param center (all 4 power_channel-relevant initial values)
    # ======================================================================
    # These set the mean of the CMA-ES distribution at generation 0.
    # sigma0 controls how quickly it moves away from this center.

    space.params["initial_k_per_day"] = {
        "low": 0.1, "high": 50.0, "log": True, "type": "float",
    }

    space.params["initial_memory_length"] = {
        "low": 3.0, "high": 200.0, "log": True, "type": "float",
    }

    space.params["initial_raw_exponents"] = {
        "low": 0.0, "high": 4.0, "log": False, "type": "float",
    }

    space.params["initial_pre_exp_scaling"] = {
        "low": 0.005, "high": 2.0, "log": True, "type": "float",
    }

    return space


def create_base_fingerprint() -> dict:
    """Create the base run fingerprint for inner CMA-ES optimization."""
    fp = deepcopy(run_fingerprint_defaults)

    fp["tokens"] = TOKENS
    fp["rule"] = RULE
    fp["startDateString"] = START_DATE
    fp["endDateString"] = WFA_END_DATE
    fp["endTestDateString"] = WFA_END_DATE
    fp["holdoutEndDateString"] = HOLDOUT_END_DATE

    fp["freq"] = "minute"
    fp["chunk_period"] = 1440
    fp["weight_interpolation_period"] = 1440

    fp["initial_pool_value"] = INITIAL_POOL_VALUE
    fp["fees"] = FEES
    fp["arb_fees"] = ARB_FEES
    fp["gas_cost"] = 0.0

    fp["do_arb"] = True
    fp["arb_frequency"] = 1
    fp["arb_quality"] = 1.0

    fp["minimum_weight"] = 0.01
    fp["max_memory_days"] = 365

    # --- Inner optimizer: CMA-ES ---
    fp["optimisation_settings"]["method"] = "cma_es"

    # Defaults that outer Optuna will override per trial
    fp["optimisation_settings"]["n_parameter_sets"] = 2
    fp["optimisation_settings"]["noise_scale"] = 0.3
    fp["optimisation_settings"]["parameter_init_method"] = "gaussian"
    fp["optimisation_settings"]["val_fraction"] = 0.2
    fp["optimisation_settings"]["early_stopping_metric"] = "daily_log_sharpe"

    fp["optimisation_settings"]["cma_es_settings"] = {
        "n_generations": 300,
        "sigma0": 0.5,
        "tol": 1e-8,
        "n_evaluation_points": 20,
        "population_size": None,  # Auto from dimension
        "memory_budget": None,    # Auto-size λ from probe (None = use Hansen default)
        "compute_dtype": "float32",
    }

    # --- Conservative initial strategy params ---
    fp["initial_k_per_day"] = 0.5
    fp["initial_memory_length"] = 30.0
    fp["initial_log_amplitude"] = -1.0
    fp["initial_raw_width"] = 1.0
    fp["initial_raw_exponents"] = 1.0
    fp["initial_pre_exp_scaling"] = 0.01

    # Training objective
    fp["return_val"] = "daily_log_sharpe"

    return fp


# =============================================================================
# GPU memory probe
# =============================================================================

def probe_cmaes_memory_budget(
    base_fp: dict,
    n_wfa_cycles: int = 4,
    max_lam: int = 1024,
    verbose: bool = True,
) -> Optional[int]:
    """Probe GPU memory by running actual CMA-ES (2 generations).

    Binary-searches for the largest population size (λ) that fits in GPU
    memory, using the real CMA-ES codepath (``train_on_historic_data`` with
    ``n_generations=2``).  This captures the true memory footprint of the
    fused ``lax.while_loop`` — nested vmap, carry state, XLA constant-folding
    — without safety-margin guesswork.

    Price data is loaded once and reused across all binary-search steps.

    Returns ``memory_budget = max_λ × n_eval_points``, which the runner's
    ``compute_cmaes_population_size()`` divides by each trial's
    ``n_eval_points`` to adapt λ per trial.

    On CPU, returns None (no OOM risk, no parallelism benefit from large λ).
    """
    if jax.default_backend() != "gpu":
        if verbose:
            print("[CMA-ES] CPU backend — skipping memory probe (using Hansen default λ)")
        return None

    import gc
    from jax import clear_caches
    from quantammsim.runners.robust_walk_forward import generate_walk_forward_cycles
    from quantammsim.runners.jax_runners import train_on_historic_data, get_unique_tokens
    from quantammsim.utils.data_processing.historic_data_utils import get_historic_parquet_data

    # Use first WFA cycle as representative window (all cycles are equal length).
    cycles = generate_walk_forward_cycles(
        base_fp["startDateString"],
        base_fp["endDateString"],
        n_wfa_cycles,
    )
    cycle = cycles[0]

    # Build probe fingerprint: minimal CMA-ES run (1 generation, 1 restart,
    # no validation) — just enough to exercise the fused while_loop.
    # lax.while_loop allocates body memory statically at compile time,
    # so 1 generation has the same footprint as 300.
    probe_fp = deepcopy(base_fp)
    probe_fp["startDateString"] = cycle.train_start_date
    probe_fp["endDateString"] = cycle.train_end_date
    probe_fp["endTestDateString"] = cycle.test_end_date
    probe_fp["optimisation_settings"]["n_parameter_sets"] = 1
    probe_fp["optimisation_settings"]["val_fraction"] = 0.0
    probe_fp["optimisation_settings"]["cma_es_settings"]["n_generations"] = 1

    n_eval = probe_fp["optimisation_settings"]["cma_es_settings"]["n_evaluation_points"]

    if verbose:
        print(f"[CMA-ES] Probing GPU memory for population auto-sizing...")
        print(f"[CMA-ES] Probe window: {cycle.train_start_date} → {cycle.train_end_date} "
              f"(1 of {n_wfa_cycles} WFA cycles)")
        print(f"[CMA-ES] Probe n_eval_points: {n_eval}, max_lam: {max_lam}")

    # Load price data once — get_data_dict slices per fingerprint dates.
    tokens = get_unique_tokens(probe_fp)
    price_df = get_historic_parquet_data(tokens, ["close"])

    if verbose:
        print(f"[CMA-ES] Price data loaded ({len(price_df)} rows)")

    # Binary search for max λ that fits in GPU memory.
    low, high = 4, max_lam
    best_lam = None

    while low <= high:
        mid = (low + high) // 2
        probe_fp["optimisation_settings"]["cma_es_settings"]["population_size"] = mid

        if verbose:
            print(f"[CMA-ES] Probing λ={mid}...", end=" ", flush=True)

        clear_caches()
        gc.collect()

        try:
            train_on_historic_data(probe_fp, price_data=price_df, verbose=False)
            if verbose:
                print("OK")
            best_lam = mid
            low = mid + 1
        except Exception as e:
            error_str = str(e).lower()
            if "resource" in error_str or "memory" in error_str or "oom" in error_str:
                if verbose:
                    print("OOM")
                high = mid - 1
            else:
                raise

        clear_caches()
        gc.collect()

    if best_lam is None:
        if verbose:
            print("[CMA-ES] WARNING: Even λ=4 OOMs — falling back to Hansen default")
        return None

    memory_budget = best_lam * n_eval

    if verbose:
        print(f"\n[CMA-ES] Memory probe results:")
        print(f"  Max λ at n_eval={n_eval}: {best_lam}")
        print(f"  Memory budget: {memory_budget} concurrent forward passes")

    return memory_budget


# =============================================================================
# Main
# =============================================================================

def run_tuning(
    n_trials: int = 60,
    n_wfa_cycles: int = 4,
    quick: bool = False,
    pruner: str = "percentile",
    objective: str = "mean_oos_daily_log_sharpe",
    total_timeout: float = None,
) -> Dict[str, Any]:
    """Run hyperparameter tuning with inner CMA-ES optimization."""
    if quick:
        n_trials = 5
        n_wfa_cycles = 2
        print("\n*** QUICK MODE ***\n")

    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    training_days = 365 * 4  # START_DATE to WFA_END_DATE = 4 years
    cycle_days = int(training_days / n_wfa_cycles)

    base_fp = create_base_fingerprint()

    # Probe GPU memory once at startup — every trial auto-sizes λ from this.
    memory_budget = probe_cmaes_memory_budget(base_fp, n_wfa_cycles=n_wfa_cycles, verbose=True)
    if memory_budget is not None:
        base_fp["optimisation_settings"]["cma_es_settings"]["memory_budget"] = memory_budget

    search_space = create_search_space(cycle_days=cycle_days)

    storage_path = STUDY_DIR / f"{STUDY_NAME}.db"
    storage = f"sqlite:///{storage_path}"

    print("=" * 70)
    print("INNER CMA-ES HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Basket:     {TOKENS}")
    print(f"Strategy:   {RULE}")
    print(f"Inner opt:  CMA-ES (derivative-free, population-based)")
    if memory_budget is not None:
        print(f"GPU budget: {memory_budget} concurrent fwd passes (λ auto-sized per trial)")
    else:
        print(f"GPU budget: N/A (CPU — using Hansen default λ)")
    print(f"WFA period: {START_DATE} to {WFA_END_DATE}")
    print(f"Holdout:    {WFA_END_DATE} to {HOLDOUT_END_DATE}")
    print(f"Objective:  {objective}")
    print(f"Pruner:     {pruner}")
    print(f"Search space ({len(search_space.params)}D):")
    for name, spec in sorted(search_space.params.items()):
        if "choices" in spec:
            print(f"  {name}: {spec['choices']}")
        elif spec.get("type") == "int":
            print(f"  {name}: [{spec['low']}, {spec['high']}] "
                  f"(int, log={spec.get('log', False)})")
        else:
            print(f"  {name}: [{spec['low']}, {spec['high']}] "
                  f"(log={spec.get('log', False)})")
    print(f"Trials:     {n_trials}")
    print(f"WFA cycles: {n_wfa_cycles} (~{cycle_days} days each)")
    print("=" * 70)

    tuner = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=n_trials,
        n_wfa_cycles=n_wfa_cycles,
        objective=objective,
        hyperparam_space=search_space,
        pruner=pruner,
        enable_pruning=(pruner != "none"),
        total_timeout=total_timeout,
        verbose=True,
        study_name=f"{STUDY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        storage=storage,
    )

    result = tuner.tune(base_fp)

    # --- Save results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = STUDY_DIR / f"best_innercmaes_params_{timestamp}.json"

    output = {
        "version": "1.0",
        "timestamp": timestamp,
        "method": "inner_cma_es",
        "basket": TOKENS,
        "rule": RULE,
        "training_period": {"start": START_DATE, "end": WFA_END_DATE},
        "holdout_end": HOLDOUT_END_DATE,
        "objective": objective,
        "best_params": result.best_params,
        "best_value": result.best_value,
        "n_completed": result.n_completed,
        "n_pruned": result.n_pruned,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # --- Print best params ---
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    print(f"Best value ({objective}): {result.best_value}")
    print()

    # Group params by category for readability
    cma_keys = [k for k in result.best_params if k.startswith("cma_es_")]
    init_keys = [k for k in result.best_params
                 if k.startswith("initial_") or k in ("noise_scale", "n_parameter_sets")]
    other_keys = [k for k in result.best_params
                  if k not in cma_keys and k not in init_keys]

    if cma_keys:
        print("CMA-ES settings:")
        for k in sorted(cma_keys):
            v = result.best_params[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")

    if init_keys:
        print("Initialization:")
        for k in sorted(init_keys):
            v = result.best_params[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")

    if other_keys:
        print("Training window / constraints:")
        for k in sorted(other_keys):
            v = result.best_params[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.6g}")
            else:
                print(f"  {k}: {v}")

    print("=" * 70)

    return {"result": result}


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for CMA-ES inner optimization",
    )
    parser.add_argument("--n-trials", "-n", type=int, default=60)
    parser.add_argument("--n-wfa-cycles", "-c", type=int, default=4)
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--pruner", "-p", default="percentile",
                        choices=["percentile", "median", "none"])
    parser.add_argument("--objective", "-o", default="mean_oos_daily_log_sharpe",
                        choices=[
                            "mean_oos_daily_log_sharpe", "worst_oos_daily_log_sharpe",
                            "mean_oos_sharpe", "worst_oos_sharpe",
                            "mean_oos_calmar", "worst_oos_calmar",
                            "mean_oos_sterling", "worst_oos_sterling",
                            "mean_oos_ulcer", "worst_oos_ulcer",
                            "mean_oos_returns_over_hodl", "worst_oos_returns_over_hodl",
                            "mean_wfe", "worst_wfe",
                        ])
    parser.add_argument("--timeout", type=float, default=None, help="Max hours")

    args = parser.parse_args()

    run_tuning(
        n_trials=args.n_trials,
        n_wfa_cycles=args.n_wfa_cycles,
        quick=args.quick,
        pruner=args.pruner,
        objective=args.objective,
        total_timeout=args.timeout * 3600 if args.timeout else None,
    )


if __name__ == "__main__":
    main()
