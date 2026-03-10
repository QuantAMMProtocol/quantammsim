#!/usr/bin/env python3
"""
Hyperparameter Tuning with Inner BFGS Optimization
====================================================

This script uses BFGS (via jax.scipy.optimize.minimize) as the inner optimizer,
with outer Optuna searching over settings that shape the BFGS landscape and
multi-start initialization.

Uses power_channel rule: a simpler strategy than mean_reversion_channel with
only 6 learnable params (k, lambda, delta_lambda, exponents, pre_exp_scaling,
weights_logits). Fewer params = fewer basins, better suited to BFGS tuning.

Why tune these?
---------------
BFGS is a local optimizer — it converges to the nearest stationary point.
This makes three things critical that don't matter as much for SGD:

1. **Objective surface**: n_evaluation_points controls how many fixed windows
   the deterministic objective averages over. Too few → the optimizer overfits
   to specific entry/exit timing. Too many → expensive and over-smoothed.

2. **Initialization strategy**: Since BFGS can't escape local optima via noise,
   the starting distribution (noise_scale, parameter_init_method, initial param
   values) determines which basins we explore. Multi-start (n_parameter_sets)
   compensates, but the center and spread of the starts matter.

3. **Convergence budget**: maxiter and tol control when BFGS stops. Usually
   not the binding constraint, but for non-smooth objectives it can matter.

Search Space (~13D):
--------------------
BFGS-specific:
  - bfgs_n_evaluation_points: Objective averaging (5-50)
  - bfgs_maxiter: Convergence budget (50-300)

Multi-start / initialization:
  - n_parameter_sets: Number of restarts (1-4, memory-constrained)
  - noise_scale: Diversity of starting points (0.05-1.0)
  - parameter_init_method: gaussian / sobol / lhs / centered_lhs

Training window / constraints:
  - bout_offset_days: Window timing
  - val_fraction: Validation holdout
  - maximum_change: Weight rate limiter
  - minimum_weight: Portfolio weight floor

Initial param center (determines basin):
  - initial_k_per_day: Momentum sensitivity
  - initial_memory_length: EWMA lookback
  - initial_raw_exponents: Power-law shape (signature param of power_channel)
  - initial_pre_exp_scaling: Gradient normalisation

Usage:
------
python experiments/tune_training_hyperparams_innerbfgs.py
python experiments/tune_training_hyperparams_innerbfgs.py --quick
python experiments/tune_training_hyperparams_innerbfgs.py -n 100 -c 6 --objective mean_oos_sharpe
"""

import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
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

START_DATE = "2019-01-01 00:00:00"
WFA_END_DATE = "2025-01-01 00:00:00"
HOLDOUT_END_DATE = "2026-01-01 00:00:00"

RULE = "power_channel"
INITIAL_POOL_VALUE = 1_000_000.0
FEES = 0.0
ARB_FEES = 0.0

STUDY_DIR = Path(__file__).parent / "hyperparam_studies"
STUDY_NAME = "eth_usdc_innerbfgs_v2"


# =============================================================================
# Search Space
# =============================================================================

def create_search_space(cycle_days: int = 180, bfgs_budget: int = None) -> HyperparamSpace:
    """
    Create search space for BFGS inner optimization of power_channel.

    Three groups of parameters:
    1. BFGS-specific: objective definition (n_evaluation_points) and
       convergence (maxiter). tol is fixed — BFGS rarely reaches
       gradient-norm tolerance on these objectives anyway.
    2. Multi-start initialization: the most important group for a local
       optimizer. Controls which basins of attraction we sample.
    3. Training window and strategy constraints: shared across all
       inner methods, affect the landscape itself.

    power_channel has 6 learnable params:
      sp_k, logit_lamb, logit_delta_lamb, sp_exponents,
      sp_pre_exp_scaling, initial_weights_logits

    We search over initial values for the 4 most impactful ones
    (k, memory, exponents, pre_exp_scaling). delta_lamb and
    weights_logits are left at defaults (0 and equal weight).

    Parameters
    ----------
    cycle_days : int
        WFA cycle length in days (for bout_offset range).
    bfgs_budget : int or None
        Max concurrent forward passes available (from memory probe).
        Constrains n_parameter_sets × n_eval_points. If None, no constraint.
    """
    space = HyperparamSpace()

    # ======================================================================
    # BFGS-specific settings
    # ======================================================================
    # n_evaluation_points: how many fixed windows form the deterministic
    # objective. This is the most BFGS-specific knob — it directly controls
    # the bias-variance trade-off of the objective surface.
    # Low (5-10) = cheap, noisy, risk of overfitting to specific timing
    # High (30-50) = smooth but expensive, may wash out useful structure
    max_eval_points = 50
    if bfgs_budget is not None:
        max_eval_points = min(max_eval_points, bfgs_budget)

    space.params["bfgs_n_evaluation_points"] = {
        "low": 5, "high": max_eval_points, "log": False, "type": "int",
    }

    # maxiter: convergence budget. BFGS usually converges in 30-80 iters
    # for our ~12-param problems (power_channel, 2 assets), but non-smooth
    # clipping/min-weight constraints can slow it down.
    space.params["bfgs_maxiter"] = {
        "low": 50, "high": 300, "log": False, "type": "int",
    }

    # ======================================================================
    # Multi-start / initialization
    # ======================================================================
    # n_parameter_sets: multi-start restarts. Each starts from a different
    # noisy initialization and converges independently. Best is selected
    # by BestParamsTracker. Memory-constrained: total concurrent forward
    # passes = n_parameter_sets × n_eval_points, capped by bfgs_budget.
    # Upper bound is dynamic: depends on the sampled n_eval_points.
    n_param_sets_spec = {
        "low": 1, "high": 4, "log": False, "type": "int",
    }
    if bfgs_budget is not None:
        n_param_sets_spec["dynamic_high"] = (
            lambda s, b=bfgs_budget: min(4, max(1, b // s["bfgs_n_evaluation_points"]))
        )
    space.params["n_parameter_sets"] = n_param_sets_spec

    # noise_scale: std of Gaussian perturbation to initial params for
    # sets 1+ (set 0 is always canonical). Larger = more diverse starts
    # but higher chance of starting in bad basins.
    space.params["noise_scale"] = {
        "low": 0.05, "high": 1.0, "log": True, "type": "float",
    }

    # parameter_init_method: how multi-start perturbations are sampled.
    # Quasi-random methods (sobol, lhs) give more uniform coverage of
    # the init space than iid Gaussian, which can cluster.
    space.params["parameter_init_method"] = {
        "choices": ["gaussian", "sobol", "lhs", "centered_lhs"],
    }

    # ======================================================================
    # Training window / constraints
    # ======================================================================
    max_val_fraction = 0.3
    # bout_offset must fit within the training period after val holdout.
    # Worst case: val_fraction = max_val_fraction, so effective train
    # days = cycle_days * (1 - max_val_fraction). Keep 4/5 of that.
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
    # For BFGS these matter more than for SGD: they set the center of the
    # multi-start distribution, which determines which basins we explore.

    # k_per_day: momentum sensitivity. Higher = more aggressive rebalancing.
    # Effective k = squareplus(sp_k) * memory_days, so this interacts with
    # memory_length.
    space.params["initial_k_per_day"] = {
        "low": 0.1, "high": 50.0, "log": True, "type": "float",
    }

    # memory_length: EWMA lookback in days. Controls gradient smoothing.
    # Short = reactive (noisy), long = sluggish (smooth).
    space.params["initial_memory_length"] = {
        "low": 3.0, "high": 200.0, "log": True, "type": "float",
    }

    # raw_exponents: power-law shape (squareplus-transformed, clipped ≥1).
    # This is the signature param of power_channel — controls how weight
    # updates scale with price gradient magnitude.
    # 1.0 = linear, >1 = superlinear (amplifies large moves).
    space.params["initial_raw_exponents"] = {
        "low": 0.0, "high": 4.0, "log": False, "type": "float",
    }

    # pre_exp_scaling: normalises gradients before the power-law.
    # Small = large effective gradients → more aggressive.
    # Large = attenuated gradients → more conservative.
    space.params["initial_pre_exp_scaling"] = {
        "low": 0.005, "high": 2.0, "log": True, "type": "float",
    }

    return space


def create_base_fingerprint() -> dict:
    """Create the base run fingerprint for inner BFGS optimization."""
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

    # --- Inner optimizer: BFGS ---
    fp["optimisation_settings"]["method"] = "bfgs"

    # Defaults that outer Optuna will override per trial
    fp["optimisation_settings"]["n_parameter_sets"] = 2
    fp["optimisation_settings"]["noise_scale"] = 0.3
    fp["optimisation_settings"]["parameter_init_method"] = "gaussian"
    fp["optimisation_settings"]["val_fraction"] = 0.2
    fp["optimisation_settings"]["early_stopping_metric"] = "daily_log_sharpe"

    fp["optimisation_settings"]["bfgs_settings"] = {
        "maxiter": 100,
        "tol": 1e-6,
        "n_evaluation_points": 20,
        "compute_dtype": "float64",
    }

    # --- Conservative initial strategy params ---
    # These are defaults; outer Optuna overrides k, memory, exponents,
    # pre_exp_scaling per trial. Others stay fixed.
    fp["initial_k_per_day"] = 0.5
    fp["initial_memory_length"] = 30.0
    fp["initial_log_amplitude"] = -1.0   # not used by power_channel, but harmless
    fp["initial_raw_width"] = 1.0        # not used by power_channel, but harmless
    fp["initial_raw_exponents"] = 1.0
    fp["initial_pre_exp_scaling"] = 0.01

    # Training objective: daily_log_sharpe by default
    fp["return_val"] = "daily_log_sharpe"

    # Fused chunked reserves: ~89% memory reduction, ~2.3x speedup
    fp["use_fused_reserves"] = True

    return fp


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
    """Run hyperparameter tuning with inner BFGS optimization."""
    if quick:
        n_trials = 5
        n_wfa_cycles = 2
        print("\n*** QUICK MODE ***\n")

    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    training_days = 365 * 6  # START_DATE to WFA_END_DATE = 6 years
    cycle_days = int(training_days / n_wfa_cycles)

    base_fp = create_base_fingerprint()

    # --- Probe GPU memory budget once, constrain search space ---
    from quantammsim.runners.jax_runner_utils import probe_max_n_parameter_sets
    probe_result = probe_max_n_parameter_sets(base_fp, verbose=True)
    max_forward_sets = probe_result["recommended_n_parameter_sets"]
    # BFGS memory ≈ n_parameter_sets × n_eval_points × 2 (grad overhead)
    # Budget: n_parameter_sets × n_eval_points ≤ max_forward_sets / 2
    bfgs_budget = max(1, max_forward_sets // 2)
    print(f"\n[Memory] Forward-pass budget: {max_forward_sets}")
    print(f"[Memory] BFGS budget (with grad overhead): {bfgs_budget}")
    print(f"[Memory] Constraint: n_parameter_sets × n_eval_points ≤ {bfgs_budget}")

    # Pass budget through to the BFGS branch for per-trial product capping
    base_fp["optimisation_settings"]["bfgs_settings"]["memory_budget"] = bfgs_budget

    search_space = create_search_space(cycle_days=cycle_days, bfgs_budget=bfgs_budget)

    storage_path = STUDY_DIR / f"{STUDY_NAME}.db"
    storage = f"sqlite:///{storage_path}"

    print("=" * 70)
    print("INNER BFGS HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Basket:     {TOKENS}")
    print(f"Strategy:   {RULE}")
    print(f"Inner opt:  BFGS (jax.scipy.optimize.minimize)")
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
    output_path = STUDY_DIR / f"best_innerbfgs_params_{timestamp}.json"

    output = {
        "version": "1.0",
        "timestamp": timestamp,
        "method": "inner_bfgs",
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
    bfgs_keys = [k for k in result.best_params if k.startswith("bfgs_")]
    init_keys = [k for k in result.best_params
                 if k.startswith("initial_") or k in ("noise_scale", "parameter_init_method", "n_parameter_sets")]
    other_keys = [k for k in result.best_params
                  if k not in bfgs_keys and k not in init_keys]

    if bfgs_keys:
        print("BFGS settings:")
        for k in sorted(bfgs_keys):
            v = result.best_params[k]
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
        description="Hyperparameter tuning for BFGS inner optimization",
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
