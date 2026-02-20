#!/usr/bin/env python3
"""
Hyperparameter Tuning with Inner Optuna Optimization
=====================================================

This script uses Optuna for the inner parameter optimization (instead of SGD).
The outer tuning varies settings that affect the inner Optuna search quality.

Key Differences from SGD version:
- fp["optimisation_settings"]["method"] = "optuna" instead of "gradient_descent"
- SGD-specific params (learning rate, batch size, etc.) are irrelevant
- Instead we tune: val_fraction, overfitting_penalty, n_startup_trials, bout_offset

Search Space for Inner Optuna:
- bout_offset_days: Training window timing (affects which market regimes are seen)
- val_fraction: Fraction of training data for validation (exploration vs exploitation)
- overfitting_penalty: Penalty for train/val gap (regularization strength)
- n_startup_trials: Random exploration phase length before TPE kicks in

Usage:
------
python experiments/tune_training_hyperparams_inneroptuna.py
python experiments/tune_training_hyperparams_inneroptuna.py --quick
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
)
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults


# =============================================================================
# Configuration
# =============================================================================

TOKENS = ["ETH", "USDC"]

START_DATE = "2021-01-01 00:00:00"
WFA_END_DATE = "2025-01-01 00:00:00"        # End of walk-forward analysis
HOLDOUT_END_DATE = "2026-01-01 00:00:00"    # End of true holdout

RULE = "mean_reversion_channel"
INITIAL_POOL_VALUE = 1_000_000.0
FEES = 0.0
ARB_FEES = 0.0

STUDY_DIR = Path(__file__).parent / "hyperparam_studies"
STUDY_NAME = "eth_usdc_inneroptuna_v1"


# =============================================================================
# Search Space
# =============================================================================

def create_search_space(cycle_days: int = 180) -> HyperparamSpace:
    """
    Create hyperparameter search space for inner Optuna optimization.

    Unlike SGD tuning (which varies learning rates, initial values, etc.),
    inner Optuna tuning focuses on:
    - Training window timing (bout_offset)
    - Validation/regularization settings (val_fraction, overfitting_penalty)
    - Search behavior (n_startup_trials, n_trials)
    - Strategy constraints (minimum_weight, maximum_change)
    """
    space = HyperparamSpace()

    # ==========================================================================
    # Training window timing
    # ==========================================================================
    # Bout offset (days from cycle start to begin training)
    # Affects which market regimes the model sees during training
    # Must fit within training period after val holdout (worst case: val_fraction=0.3)
    max_val_fraction = 0.3
    max_offset = max(1, int(cycle_days * (1 - max_val_fraction) * 4 / 5))
    space.params["bout_offset_days"] = {"low": 0, "high": max_offset, "log": False, "type": "int"}

    # ==========================================================================
    # Inner Optuna search settings
    # ==========================================================================
    # Validation fraction: how much training data to hold out for validation
    # Lower = more training data but less reliable validation signal
    # Higher = better validation estimate but less training data
    space.params["val_fraction"] = {"low": 0.1, "high": max_val_fraction, "log": False, "type": "float"}

    # Overfitting penalty: penalize train/val gap in inner Optuna objective
    # 0.0 = pure training performance, higher = more regularization
    space.params["optuna_overfitting_penalty"] = {"low": 0.0, "high": 0.5, "log": False, "type": "float"}

    # N startup trials: random sampling before TPE sampler kicks in
    # More = better exploration, fewer = faster convergence
    space.params["optuna_n_startup_trials"] = {"low": 10, "high": 40, "log": False, "type": "int"}

    # Inner n_trials: more = better search but diminishing returns + compute cost
    space.params["optuna_n_trials"] = {"low": 50, "high": 1000, "log": False, "type": "int"}

    # ==========================================================================
    # Strategy constraints
    # ==========================================================================
    # Minimum weight: floor on portfolio weights (prevents extreme allocations)
    # Lower = more aggressive strategies possible, higher = more conservative
    space.params["minimum_weight"] = {"low": 0.01, "high": 0.1, "log": True, "type": "float"}

    # Maximum change: max weight change per chunk (rate limiter)
    # Lower = smoother weight changes, higher = more responsive
    space.params["maximum_change"] = {"low": 3e-5, "high": 2.0, "log": True, "type": "float"}

    return space


def create_base_fingerprint() -> dict:
    """Create the base run fingerprint for inner Optuna optimization."""
    fp = deepcopy(run_fingerprint_defaults)

    fp["tokens"] = TOKENS
    fp["rule"] = RULE
    fp["startDateString"] = START_DATE
    fp["endDateString"] = WFA_END_DATE  # Per-cycle adapter overwrites this; set to WFA end as safe default
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

    # KEY DIFFERENCE: Use Optuna instead of gradient descent
    fp["optimisation_settings"]["method"] = "optuna"
    fp["optimisation_settings"]["n_parameter_sets"] = 1

    # Inner Optuna settings
    fp["optimisation_settings"]["optuna_settings"] = {
        "study_name": None,
        "storage": {"type": "sqlite", "url": None},
        "n_trials": 100,  # Inner Optuna trials per WFA cycle
        "n_jobs": 1,
        "timeout": None,
        "n_startup_trials": 20,
        "early_stopping": {"enabled": False, "patience": 50, "min_improvement": 0.001},
        "multi_objective": False,
        "make_scalar": False,
        "expand_around": False,
        "overfitting_penalty": 0.2,
        "parameter_config": {
            "memory_length": {"low": 1, "high": 200, "log_scale": True, "scalar": False},
            "k_per_day": {"low": 0.1, "high": 100, "log_scale": True, "scalar": False},
            "log_amplitude": {"low": -3, "high": 3, "log_scale": False, "scalar": False},
            "raw_width": {"low": -3, "high": 3, "log_scale": False, "scalar": False},
            "raw_exponents": {"low": -2, "high": 4, "log_scale": False, "scalar": False},
            "raw_pre_exp_scaling": {"low": -3, "high": 3, "log_scale": False, "scalar": False},
            "logit_lamb": {"low": -10, "high": 10, "log_scale": False, "scalar": False},
        },
    }

    # Fused chunked reserves: ~89% memory reduction, ~2.3x speedup
    fp["use_fused_reserves"] = True

    return fp


# =============================================================================
# Main
# =============================================================================

def run_tuning(
    n_trials: int = 30,
    n_wfa_cycles: int = 4,
    quick: bool = False,
    pruner: str = "percentile",
    objective: str = "mean_oos_sharpe",
    total_timeout: float = None,
) -> Dict[str, Any]:
    """Run hyperparameter tuning with inner Optuna optimization."""
    if quick:
        n_trials = 5
        n_wfa_cycles = 2
        print("\n*** QUICK MODE ***\n")

    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    training_days = 365 * 4  # START_DATE to WFA_END_DATE = 4 years
    cycle_days = int(training_days / n_wfa_cycles)

    base_fp = create_base_fingerprint()
    search_space = create_search_space(cycle_days=cycle_days)

    storage_path = STUDY_DIR / f"{STUDY_NAME}.db"
    storage = f"sqlite:///{storage_path}"

    print("=" * 70)
    print("INNER OPTUNA HYPERPARAMETER TUNING")
    print("=" * 70)
    print(f"Basket: {TOKENS}")
    print(f"Strategy: {RULE}")
    print(f"Inner optimization: OPTUNA (not SGD)")
    print(f"Inner n_trials: {base_fp['optimisation_settings']['optuna_settings']['n_trials']}")
    print(f"WFA period: {START_DATE} to {WFA_END_DATE}")
    print(f"Outer objective: {objective}")
    print(f"Pruner: {pruner}")
    print(f"Search space: {list(search_space.params.keys())}")
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

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = STUDY_DIR / f"best_inneroptuna_params_{timestamp}.json"

    output = {
        "version": "1.0",
        "timestamp": timestamp,
        "method": "inner_optuna",
        "basket": TOKENS,
        "rule": RULE,
        "best_params": result.best_params,
        "best_value": result.best_value,
        "n_completed": result.n_completed,
        "n_pruned": result.n_pruned,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print(f"Best value: {result.best_value}")
    print(f"Best params: {result.best_params}")

    return {"result": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", "-n", type=int, default=30)
    parser.add_argument("--n-wfa-cycles", "-c", type=int, default=4)
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--pruner", "-p", default="percentile",
                        choices=["percentile", "median", "none"])
    parser.add_argument("--objective", "-o", default="mean_oos_sharpe",
                        choices=[
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
