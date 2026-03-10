#!/usr/bin/env python3
"""
BFGS Optimizer Example
======================

Trains a mean_reversion_channel strategy on ETH/USDC using full-batch BFGS
via jax.scipy.optimize.minimize.

BFGS is a quasi-Newton method that approximates the Hessian from gradient
history. It converges much faster than Adam/SGD for small parameter counts
(our strategies have ~10-20 scalar params) because:
  - Full curvature information → superlinear convergence near optima
  - No learning rate to tune
  - Deterministic objective (fixed evaluation points) → no gradient noise

The trade-off: each BFGS iteration is more expensive (implicit Hessian
approximation), and it can't escape sharp local optima the way SGD's
noise can. Multi-start (n_parameter_sets > 1) mitigates the latter.

This example uses probe_max_n_parameter_sets to auto-size the number of
multi-start runs based on available device memory.

Usage:
------
python experiments/train_bfgs_example.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from copy import deepcopy
from quantammsim.runners.jax_runners import train_on_historic_data
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.runners.jax_runner_utils import probe_max_n_parameter_sets
from quantammsim.core_simulator.param_utils import recursive_default_set


def create_bfgs_fingerprint():
    """Create a run fingerprint for BFGS optimization."""
    fp = deepcopy(run_fingerprint_defaults)

    # --- Asset pair and dates ---
    fp["tokens"] = ["ETH", "USDC"]
    fp["rule"] = "mean_reversion_channel"
    fp["startDateString"] = "2023-01-01 00:00:00"
    fp["endDateString"] = "2023-06-01 00:00:00"
    fp["endTestDateString"] = "2023-09-01 00:00:00"

    # --- Pool settings ---
    fp["initial_pool_value"] = 1_000_000.0
    fp["fees"] = 0.003
    fp["arb_fees"] = 0.0
    fp["gas_cost"] = 0.0
    fp["minimum_weight"] = 0.05
    fp["max_memory_days"] = 365

    # --- Objective ---
    fp["return_val"] = "daily_log_sharpe"

    # --- BFGS optimization ---
    fp["optimisation_settings"]["method"] = "bfgs"
    fp["optimisation_settings"]["noise_scale"] = 0.3

    # Validation holdout for param selection
    fp["optimisation_settings"]["val_fraction"] = 0.2
    fp["optimisation_settings"]["early_stopping_metric"] = "daily_log_sharpe"

    # BFGS-specific settings
    fp["optimisation_settings"]["bfgs_settings"] = {
        "maxiter": 100,
        "tol": 1e-6,
        "n_evaluation_points": 20,
    }

    # --- Conservative initial strategy params ---
    fp["initial_k_per_day"] = 0.5
    fp["initial_memory_length"] = 30.0
    fp["initial_log_amplitude"] = -1.0
    fp["initial_raw_width"] = 1.0
    fp["initial_raw_exponents"] = 1.0
    fp["initial_pre_exp_scaling"] = 0.01

    return fp


def auto_size_bfgs(fp):
    """Probe device memory and set n_parameter_sets for BFGS.

    probe_max_n_parameter_sets tests a single nograd forward pass per param
    set. BFGS is heavier: each iteration evaluates n_evaluation_points
    forward+backward passes per param set. We scale down the probe result
    by n_evaluation_points (eval point fan-out) and a 2x factor for gradient
    tape overhead.
    """
    n_eval_points = fp["optimisation_settings"]["bfgs_settings"]["n_evaluation_points"]

    print("[Auto-size] Probing device memory...")
    probe_result = probe_max_n_parameter_sets(fp, verbose=True)
    probe_max = probe_result["recommended_n_parameter_sets"]

    # BFGS memory per param set ≈ n_eval_points * 2 (gradients) * single_fwd
    bfgs_factor = n_eval_points * 2
    bfgs_safe = max(1, probe_max // bfgs_factor)

    print(f"[Auto-size] Probe recommended: {probe_max} (single forward pass)")
    print(f"[Auto-size] BFGS adjustment: ÷{bfgs_factor} "
          f"({n_eval_points} eval pts × 2 for gradients)")
    print(f"[Auto-size] BFGS n_parameter_sets: {bfgs_safe}")

    fp["optimisation_settings"]["n_parameter_sets"] = bfgs_safe
    return bfgs_safe


def main():
    fp = create_bfgs_fingerprint()

    # Auto-size n_parameter_sets based on available device memory
    n_sets = auto_size_bfgs(fp)

    print("\n" + "=" * 70)
    print("BFGS TRAINING EXAMPLE")
    print("=" * 70)
    print(f"Tokens:    {fp['tokens']}")
    print(f"Rule:      {fp['rule']}")
    print(f"Train:     {fp['startDateString']} → {fp['endDateString']}")
    print(f"Test:      {fp['endDateString']} → {fp['endTestDateString']}")
    print(f"Objective: {fp['return_val']}")
    print(f"N starts:  {n_sets}")
    print(f"Val frac:  {fp['optimisation_settings']['val_fraction']}")
    bfgs = fp["optimisation_settings"]["bfgs_settings"]
    print(f"BFGS:      maxiter={bfgs['maxiter']}, tol={bfgs['tol']}, "
          f"n_eval_pts={bfgs['n_evaluation_points']}")
    print("=" * 70)

    params, metadata = train_on_historic_data(
        fp,
        verbose=True,
        force_init=True,
        return_training_metadata=True,
    )

    # --- Report ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    best_idx = metadata["best_param_idx"]
    print(f"Selection: {metadata['selection_method']} on {metadata['selection_metric']}")
    print(f"Best param set: {best_idx}")

    if metadata["best_train_metrics"]:
        tm = metadata["best_train_metrics"][best_idx]
        print(f"\nTrain (IS):")
        print(f"  Sharpe:            {tm.get('sharpe', np.nan):+.4f}")
        print(f"  Daily log Sharpe:  {tm.get('daily_log_sharpe', np.nan):+.4f}")
        print(f"  Return over HODL:  {tm.get('returns_over_uniform_hodl', np.nan):+.4f}")

    if metadata.get("best_val_metrics"):
        vm = metadata["best_val_metrics"][best_idx]
        print(f"\nValidation:")
        print(f"  Sharpe:            {vm.get('sharpe', np.nan):+.4f}")
        print(f"  Daily log Sharpe:  {vm.get('daily_log_sharpe', np.nan):+.4f}")
        print(f"  Return over HODL:  {vm.get('returns_over_uniform_hodl', np.nan):+.4f}")

    if metadata["best_continuous_test_metrics"]:
        ctm = metadata["best_continuous_test_metrics"][best_idx]
        print(f"\nTest (OOS):")
        print(f"  Sharpe:            {ctm.get('sharpe', np.nan):+.4f}")
        print(f"  Daily log Sharpe:  {ctm.get('daily_log_sharpe', np.nan):+.4f}")
        print(f"  Return over HODL:  {ctm.get('returns_over_uniform_hodl', np.nan):+.4f}")

    # Per-set convergence
    if "objective_per_set" in metadata:
        print(f"\nPer-set objectives:")
        for i, (obj, status) in enumerate(
            zip(metadata["objective_per_set"], metadata["status_per_set"])
        ):
            marker = " ← best" if i == best_idx else ""
            status_str = "converged" if status == 0 else f"status={status}"
            print(f"  Set {i}: {obj:+.6f} ({status_str}){marker}")

    print(f"\nOptimized params:")
    for k, v in sorted(params.items()):
        if k == "subsidary_params":
            continue
        if hasattr(v, "shape"):
            print(f"  {k}: {np.array(v)}")
        else:
            print(f"  {k}: {v}")

    print("=" * 70)


if __name__ == "__main__":
    main()
