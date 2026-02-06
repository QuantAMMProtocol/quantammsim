#!/usr/bin/env python3
"""
Hyperparameter Tuning for Robust Training
==========================================

This script finds optimal training hyperparameters using Optuna with
percentile-based pruning and walk-forward analysis.

Key Features:
-------------
1. Percentile pruning: Filter obvious disasters without over-predicting
2. Risk-aware objectives: Optimize for daily_log_sharpe, Calmar, Sterling, Ulcer, or Sharpe
3. Stability analysis: Identifies unreliable hyperparameters
4. Final holdout: never touched during tuning
5. Includes all robustness features: turnover penalty, price noise,
   stratified sampling, Sobol/LHS init, log-space reserves

Why Percentile Pruning (not Hyperband)?
---------------------------------------
WFA cycles are NOT true multi-fidelity: cycle 1 doesn't predict cycles 2-4
because they're different market regimes, not cheap proxies. Hyperband/ASHA
assume correlation between fidelity levels we don't have.

PercentilePruner (25%) is better for our case:
- Just filters obvious disasters (bottom 25%)
- Doesn't make false predictions about future cycles
- Simpler, more appropriate for independent regime evaluation

Search Space:
-------------
The full search space (~20 dimensions) includes:
- Core: base_lr, batch_size, n_iterations, clip_norm, bout_offset
- Schedule: lr_schedule_type, lr_decay_ratio, warmup_fraction
- Regularization: weight_decay, noise_scale, turnover_penalty, price_noise_sigma
- Sampling: sample_method (uniform/stratified), parameter_init_method (gaussian/sobol/lhs)
- Strategy: maximum_change, training_objective
- Early stopping: patience, val_fraction
- Initial params (optional): memory_length, k_per_day, amplitude, width, exponents

Usage:
------
# Standard run (150 trials, daily_log_sharpe objective)
python experiments/tune_training_hyperparams.py

# Fewer trials for faster results
python experiments/tune_training_hyperparams.py -n 50

# Optimize for Calmar ratio instead
python experiments/tune_training_hyperparams.py --objective mean_oos_calmar

# Conservative search ranges
python experiments/tune_training_hyperparams.py --conservative

# Quick test (5 trials, 2 cycles)
python experiments/tune_training_hyperparams.py --quick

# No pruning (run all cycles for all trials)
python experiments/tune_training_hyperparams.py --pruner none
"""

import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantammsim.runners.hyperparam_tuner import (
    HyperparamTuner,
    HyperparamSpace,
    TuningResult,
)
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults


# =============================================================================
# Configuration
# =============================================================================

TOKENS = ["ETH", "USDC"]

# Date ranges with explicit holdout
START_DATE = "2021-01-01 00:00:00"
END_DATE = "2024-06-01 00:00:00"
END_TEST_DATE = "2025-01-01 00:00:00"  # WFA ends here (no peeking!)
FINAL_HOLDOUT_END = "2026-01-01 00:00:00"  # True OOS - only for final validation

RULE = "mean_reversion_channel"
INITIAL_POOL_VALUE = 1_000_000.0
FEES = 0.0
ARB_FEES = 0.0

STUDY_DIR = Path(__file__).parent / "hyperparam_studies"
STUDY_NAME = "eth_usdc_tuning_v3"


# =============================================================================
# Search Space
# =============================================================================

def create_search_space(
    cycle_days: int = 180,
    conservative: bool = False,
    include_initial_params: bool = True,
    objective_metric: str = "mean_oos_daily_log_sharpe",
) -> HyperparamSpace:
    """
    Create hyperparameter search space.

    Builds on HyperparamSpace.create() from the library, adding initial strategy
    parameter values which are crucial for gradient descent convergence.

    Parameters
    ----------
    cycle_days : int
        Approximate WFA cycle length in days
    conservative : bool
        If True, use tighter ranges for stability
    include_initial_params : bool
        If True, include initial strategy parameter values. These are crucial
        because gradient descent with poor initialization often converges to
        bad local minima.
    objective_metric : str
        Outer Optuna objective (e.g., "mean_oos_daily_log_sharpe", "mean_oos_calmar").
        Passed to library to conditionally include training_objective choice.
    """
    # Start from library defaults - includes:
    # - Core training params (base_lr, batch_size, n_iterations, bout_offset_days, clip_norm)
    # - Weight decay (use_weight_decay, weight_decay)
    # - LR schedule (lr_schedule_type, lr_decay_ratio, warmup_fraction)
    # - Early stopping (use_early_stopping, early_stopping_patience, val_fraction)
    # - Regularization (noise_scale, maximum_change, turnover_penalty, price_noise_sigma)
    # - Sampling (sample_method, parameter_init_method)
    # - Training objective (conditionally included based on objective_metric)
    space = HyperparamSpace.create(cycle_days=cycle_days, objective_metric=objective_metric)

    # Conservative mode: tighten ranges for stability
    if conservative:
        space.params["base_lr"] = {"low": 1e-5, "high": 1e-2, "log": True}
        space.params["batch_size"] = {"low": 8, "high": 32, "log": True, "type": "int"}
        space.params["n_iterations"] = {"low": 100, "high": 2000, "log": True, "type": "int"}

    if include_initial_params:
        # Initial strategy parameter values - crucial for gradient descent convergence!
        # Bad initializations lead to bad local minima regardless of learning rate.
        #
        # IMPORTANT: Ranges derived from PARAM_SCHEMA optuna bounds + transformations:
        # - sp_k uses squareplus, optuna range [-1, 100] -> k in [0.6, 100]
        # - logit_lamb optuna range [-4, 8] -> memory_days ~[1, 200] with chunk_period=1440
        # - sp_amplitude uses 2^log_amplitude then inverse_squareplus, optuna [-3, 4]
        # - sp_width uses 2^raw_width then inverse_squareplus, optuna [-3, 3]
        # - sp_exponents is DIRECT (not 2^x!), optuna range [-2, 4]
        space.params.update({
            # Memory length: how many days of history the strategy remembers
            # logit_lamb optuna [-4, 8] corresponds to ~[1, 200] days at chunk_period=1440
            "initial_memory_length": {"low": 2.0, "high": 100.0, "log": True},

            # k_per_day: aggressiveness of weight changes (updates per day)
            # sp_k optuna [-1, 100] -> squareplus gives [0.6, 100]
            "initial_k_per_day": {"low": 0.5, "high": 100.0, "log": True},

            # log_amplitude: amplitude of mean reversion effect (2^x scaling)
            # sp_amplitude optuna [-3, 4] -> squareplus gives [0.3, 4.2]
            # So 2^log_amplitude should be in [0.3, 4.2] -> log_amplitude in [-2, 2]
            "initial_log_amplitude": {"low": -2.0, "high": 2.0, "log": False},

            # raw_width: width of mean reversion channel (2^x scaling)
            # sp_width optuna [-3, 3] -> squareplus gives [0.3, 3.3]
            # So 2^raw_width should be in [0.3, 3.3] -> raw_width in [-2, 2]
            "initial_raw_width": {"low": -2.0, "high": 2.0, "log": False},

            # raw_exponents: DIRECT pass-through to sp_exponents (NOT 2^x!)
            # sp_exponents optuna range [-2, 4] -> squareplus gives [0.4, 4.2]
            "initial_raw_exponents": {"low": -2.0, "high": 4.0, "log": False},

            # pre_exp_scaling: scales price gradient before exponentiation
            # Goes through inverse_squareplus to get sp_pre_exp_scaling
            # Broad range from very small to very large scaling
            "initial_pre_exp_scaling": {"low": 0.01, "high": 100.0, "log": True},
        })

    return space


def create_base_fingerprint() -> dict:
    """Create the base run fingerprint."""
    fp = deepcopy(run_fingerprint_defaults)

    fp["tokens"] = TOKENS
    fp["rule"] = RULE
    fp["startDateString"] = START_DATE
    fp["endDateString"] = END_DATE
    fp["endTestDateString"] = END_TEST_DATE

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

    fp["optimisation_settings"]["optimiser"] = "adam"
    fp["optimisation_settings"]["method"] = "gradient_descent"
    fp["optimisation_settings"]["use_gradient_clipping"] = True
    fp["optimisation_settings"]["n_parameter_sets"] = 8

    return fp


# =============================================================================
# Stability Analysis
# =============================================================================

def analyze_stability(result: TuningResult) -> Dict[str, Any]:
    """Analyze stability of best hyperparameters across top trials."""
    if not result.all_trials:
        return {"error": "No trials to analyze"}

    completed = [t for t in result.all_trials if t["state"] == "TrialState.COMPLETE"]
    if len(completed) < 3:
        return {"error": "Need at least 3 completed trials"}

    completed.sort(key=lambda t: t["value"] if t["value"] else float("-inf"), reverse=True)
    top_trials = completed[:min(10, len(completed))]

    param_distributions = {}
    for param in result.best_params.keys():
        values = [t["params"].get(param) for t in top_trials if param in t["params"]]
        if values and all(isinstance(v, (int, float)) for v in values):
            mean_val = np.mean(values)
            param_distributions[param] = {
                "mean": mean_val,
                "std": np.std(values),
                "cv": np.std(values) / mean_val if mean_val != 0 else float("inf"),
            }

    unstable = [p for p, s in param_distributions.items() if s.get("cv", 0) > 0.5]

    return {
        "param_distributions": param_distributions,
        "n_analyzed": len(top_trials),
        "unstable_params": unstable,
        "recommendation": (
            "All parameters stable." if not unstable
            else f"Consider fixing: {', '.join(unstable)}"
        ),
    }


# =============================================================================
# Main Tuning
# =============================================================================

def run_tuning(
    n_trials: int = 150,
    n_wfa_cycles: int = 4,
    resume: bool = False,
    quick: bool = False,
    conservative: bool = False,
    pruner: str = "percentile",
    objective: str = "mean_oos_daily_log_sharpe",
    total_timeout: float = None,
    include_initial_params: bool = True,
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning.

    Parameters
    ----------
    pruner : str
        Pruning strategy: "percentile" (recommended), "median", "hyperband", "successive_halving", "none"
        Percentile (25%) filters obvious disasters without over-predicting future cycles.
        Hyperband/ASHA assume multi-fidelity correlation we don't have with WFA.
    include_initial_params : bool
        If True (default), tune initial strategy parameters (memory_length, k_per_day, etc.).
        This is crucial because gradient descent with poor initialization leads to bad local minima.
    """
    if quick:
        n_trials = 5
        n_wfa_cycles = 2
        print("\n*** QUICK MODE ***\n")

    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    training_days = 365 * 3.5
    cycle_days = int(training_days / n_wfa_cycles)

    base_fp = create_base_fingerprint()
    search_space = create_search_space(
        cycle_days=cycle_days,
        conservative=conservative,
        include_initial_params=include_initial_params,
        objective_metric=objective,
    )

    # Use in-memory storage to avoid SQLAlchemy version conflicts
    # For persistent storage, fix: pip install 'sqlalchemy<2.0' or upgrade typing_extensions
    storage_path = STUDY_DIR / f"{STUDY_NAME}.db"
    storage = f"sqlite:///{storage_path}"

    print("=" * 70)
    print("HYPERPARAMETER TUNING v3")
    print("=" * 70)
    print(f"Basket: {TOKENS}")
    print(f"Strategy: {RULE}")
    print(f"Tuning period: {START_DATE} to {END_DATE}")
    print(f"FINAL HOLDOUT (untouched): {END_DATE} to {FINAL_HOLDOUT_END}")
    print(f"Objective: {objective}")
    print(f"Pruner: {pruner}")
    print(f"Conservative: {conservative}")
    print(f"Tune initial params: {include_initial_params}")
    print(f"Search space: {len(search_space.params)} parameters")
    print(f"Trials: {n_trials}")
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
    stability = analyze_stability(result)

    # Print stability analysis
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS")
    print("=" * 70)
    if "error" not in stability:
        for param, stats in stability.get("param_distributions", {}).items():
            cv = stats.get("cv", 0)
            label = "STABLE" if cv < 0.3 else "MODERATE" if cv < 0.5 else "UNSTABLE"
            print(f"  {param}: CV={cv:.2f} ({label})")
        print(f"\n{stability['recommendation']}")
    print("=" * 70)

    # Save results
    save_results(result, stability, pruner)

    return {"result": result, "stability": stability}


def save_results(result, stability: Dict[str, Any], pruner: str):
    """Save tuning results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = STUDY_DIR / f"best_params_{timestamp}.json"

    output = {
        "version": "3.0",
        "timestamp": timestamp,
        "pruner": pruner,
        "basket": TOKENS,
        "rule": RULE,
        "training_period": {"start": START_DATE, "end": END_DATE},
        "final_holdout_end": FINAL_HOLDOUT_END,
        "best_params": result.best_params,
        "best_value": result.best_value,
        "tuning_summary": {
            "n_completed": result.n_completed,
            "n_pruned": getattr(result, 'n_pruned', 0),
        },
        "stability_analysis": stability,
        "next_steps": [
            "1. Validate on final holdout (2024 H2 - 2025)",
            "2. Review stability - fix unstable params if any",
            "3. Run on additional asset pairs to check transferability",
            "4. Paper trade before production",
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Print copy-paste config
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    print("optimisation_settings = {")
    for key, value in sorted(result.best_params.items()):
        if isinstance(value, str):
            print(f'    "{key}": "{value}",')
        elif isinstance(value, float):
            print(f'    "{key}": {value:.6g},')
        else:
            print(f'    "{key}": {value},')
    print("}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for crypto baskets",
    )
    parser.add_argument("--n-trials", "-n", type=int, default=150)
    parser.add_argument("--n-wfa-cycles", "-c", type=int, default=4)
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--conservative", action="store_true")
    parser.add_argument("--no-initial-params", action="store_true",
                        help="Skip tuning initial strategy params (memory_length, k_per_day, etc.)")
    parser.add_argument("--pruner", "-p", default="percentile",
                        choices=["percentile", "median", "hyperband", "successive_halving", "none"],
                        help="Pruning strategy: percentile (recommended), median, hyperband, successive_halving, none")
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
        resume=args.resume,
        quick=args.quick,
        conservative=args.conservative,
        pruner=args.pruner,
        objective=args.objective,
        total_timeout=args.timeout * 3600 if args.timeout else None,
        include_initial_params=not args.no_initial_params,
    )


if __name__ == "__main__":
    main()
