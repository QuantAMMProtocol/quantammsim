"""Optuna tuning of reClAMM pool parameters with calibrated noise models.

Supports noise model modes:
  --noise-model none           (pure arb, no noise traders)
  --noise-model calibrated     (legacy 8-covariate model, AAVE/ETH only)
  --noise-model market_linear  (per-pool model with market features)
  --noise-model mm_observed    (MM model + DeFi Llama competitor TVL)

The market_linear model uses precomputed daily arrays from the per-pool
calibrated noise model artifact (results/linear_market_noise/). It evaluates:

    log(V_noise) = base_t + tvl_coeff_t * log(effective_TVL)

where base_t absorbs all non-TVL terms (market regime, token volatility,
pair volatility, day-of-week, cross-pool volumes) and tvl_coeff_t is the
effective TVL coefficient including interaction terms.

Default pool: AAVE/ETH (0x9d1fcf346ea1b0). Use --tokens to override.

Usage:
    cd <repo-root>
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim_reclamm_public

    # AAVE/ETH with market_linear noise (default)
    python experiments/tune_reclamm_calibrated_noise.py

    # COW/ETH with no noise model
    python experiments/tune_reclamm_calibrated_noise.py --tokens COW ETH --noise-model none

    # All objectives
    python experiments/tune_reclamm_calibrated_noise.py --all-objectives

    # More trials
    python experiments/tune_reclamm_calibrated_noise.py --n-trials 200
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from quantammsim.runners.jax_runners import train_on_historic_data

DEFAULT_POOL_ID = "0x9d1fcf346ea1b0"  # AAVE/WETH Mainnet
DEFAULT_TOKENS = ["AAVE", "ETH"]

# --- Legacy 8-covariate noise coefficients ---
NOISE_COEFFS_LEGACY = [
    -0.453,   # c_0: intercept
     0.025,   # c_1: log(TVL)
    -0.060,   # c_2: log(sigma)
     0.310,   # c_3: log(TVL) * log(sigma)
    -0.149,   # c_4: log(TVL) * fee
     0.359,   # c_5: log(sigma) * fee
     0.061,   # c_6: dow_sin
     0.060,   # c_7: dow_cos
]
LEGACY_LOG_CADENCE = 2.68
LEGACY_ARB_FREQUENCY = max(1, round(math.exp(LEGACY_LOG_CADENCE)))  # ~15 min

PARAMETER_CONFIG = {
    "price_ratio": {"low": 1.01, "high": 200.0, "log_scale": True, "scalar": True},
    "centeredness_margin": {"low": 0.01, "high": 0.99, "scalar": True},
    "shift_exponent": {"low": 1e-5, "high": 125.0, "log_scale": True, "scalar": True},
}

OBJECTIVES = [
    "daily_log_sharpe", "daily_log_sharpe_excess",
    "returns_over_hodl", "fee_revenue_over_value",
    "calmar", "sterling", "weekly_rovar",
]


def _build_market_linear_arrays(args, pool_id, tokens):
    """Precompute noise arrays from the per-pool market noise model artifact."""
    from quantammsim.calibration.noise_model_arrays import build_simulator_arrays

    # Parse dates — strip time component for the array builder
    start = args.start_date.split(" ")[0]
    end = args.end_test_date.split(" ")[0]

    print(f"  Building market_linear noise arrays for {pool_id}...")
    print(f"  Date range: {start} → {end}")
    arrays = build_simulator_arrays(
        token_a=tokens[0],
        token_b=tokens[1],
        start_date=start,
        end_date=end,
        artifact_dir=args.artifact_dir,
        pool_id=pool_id,
    )
    print(f"  {arrays['n_days']} days, {arrays['n_minutes']} minutes")
    print(f"  noise_base range: [{arrays['noise_base'].min():.2f},"
          f" {arrays['noise_base'].max():.2f}]")
    print(f"  noise_tvl_coeff range: [{arrays['noise_tvl_coeff'].min():.4f},"
          f" {arrays['noise_tvl_coeff'].max():.4f}]")

    # Save arrays to disk (fingerprint can't hold numpy arrays — it gets JSON-serialized)
    import os
    cache_dir = os.path.join(args.artifact_dir, "_sim_arrays")
    os.makedirs(cache_dir, exist_ok=True)
    arrays_path = os.path.join(cache_dir, f"{pool_id}_{start}_{end}.npz")
    np.savez(arrays_path,
             noise_base=arrays["noise_base"],
             noise_tvl_coeff=arrays["noise_tvl_coeff"],
             tvl_mean=arrays["tvl_mean"],
             tvl_std=arrays["tvl_std"])
    print(f"  Saved arrays: {arrays_path}")

    # Get learned cadence from artifact
    from quantammsim.calibration.noise_model_arrays import load_artifact, _find_pool_index
    art, meta = load_artifact(args.artifact_dir)
    pool_idx = _find_pool_index(pool_id, meta["pool_ids"])
    if pool_idx >= 0:
        learned_cadence = float(np.exp(art["log_cadence"][pool_idx]))
        print(f"  Learned cadence: {learned_cadence:.1f} min")
    else:
        learned_cadence = 5.0
        print(f"  Pool not in calibration set, using default cadence: {learned_cadence}")

    return arrays_path, max(1, round(learned_cadence))


def _build_mm_observed_arrays(args, pool_id, tokens):
    """Precompute noise arrays from the MM model + DeFi Llama competitor TVL."""
    from quantammsim.calibration.noise_model_arrays import (
        build_mm_simulator_arrays, load_artifact, _find_pool_index,
    )

    start = args.start_date.split(" ")[0]
    end = args.end_test_date.split(" ")[0]

    print(f"  Building mm_observed noise arrays for {pool_id}...")
    print(f"  Date range: {start} → {end}")
    arrays = build_mm_simulator_arrays(
        token_a=tokens[0],
        token_b=tokens[1],
        start_date=start,
        end_date=end,
        mm_artifact_dir=args.artifact_dir,
        competitor_tvl_path=args.competitor_tvl_path,
        pool_id=pool_id,
    )
    print(f"  {arrays['n_days']} days, {arrays['n_minutes']} minutes")
    print(f"  noise_base range: [{arrays['noise_base'].min():.2f},"
          f" {arrays['noise_base'].max():.2f}]")
    print(f"  competitor_tvl range: [${np.exp(np.log(arrays['competitor_tvl'].max())):.0f}]")

    # Save arrays to disk
    import os
    cache_dir = os.path.join(args.artifact_dir, "_sim_arrays")
    os.makedirs(cache_dir, exist_ok=True)
    arrays_path = os.path.join(cache_dir, f"{pool_id}_{start}_{end}_mm.npz")
    np.savez(arrays_path,
             noise_base=arrays["noise_base"],
             competitor_tvl=arrays["competitor_tvl"])
    print(f"  Saved arrays: {arrays_path}")

    # Get cadence from MM model artifact
    art, meta = load_artifact(args.artifact_dir)
    pool_idx = _find_pool_index(pool_id, meta["pool_ids"])
    if pool_idx >= 0 and "log_cadence" in art:
        learned_cadence = float(np.exp(art["log_cadence"][pool_idx]))
        print(f"  Learned cadence: {learned_cadence:.1f} min")
    else:
        learned_cadence = 5.0
        print(f"  Using default cadence: {learned_cadence}")

    return arrays_path, max(1, round(learned_cadence))


def _build_opt_settings(args):
    """Build optimisation_settings for optuna, bfgs, or cma_es."""
    robust = ({"robust_temperature": args.robust_temperature}
              if args.robust_temperature is not None else {})

    if args.method == "bfgs":
        return {
            "method": "bfgs",
            "n_parameter_sets": args.n_parameter_sets,
            **({"val_fraction": args.val_fraction} if args.val_fraction is not None else {}),
            **robust,
            "bfgs_settings": {
                "maxiter": args.bfgs_maxiter,
                "tol": args.bfgs_tol,
                "n_evaluation_points": args.bfgs_eval_points,
                "compute_dtype": "float64",
            },
        }
    elif args.method == "cma_es":
        return {
            "method": "cma_es",
            "n_parameter_sets": args.n_parameter_sets,
            **({"val_fraction": args.val_fraction} if args.val_fraction is not None else {}),
            **robust,
            "optuna_settings": {
                "parameter_config": PARAMETER_CONFIG,
            },
            "cma_es_settings": {
                "population_size": args.cma_pop_size,
                "n_generations": args.cma_generations,
                "sigma0": args.cma_sigma0,
                "tol": 1e-8,
                "n_evaluation_points": args.cma_eval_points,
                "compute_dtype": "float32",
                **({"overfitting_penalty": args.overfitting_penalty}
                   if args.overfitting_penalty is not None else {}),
            },
        }
    else:
        return {
            "method": "optuna",
            "n_parameter_sets": 1,
            **({"val_fraction": args.val_fraction} if args.val_fraction is not None else {}),
            **robust,
            "optuna_settings": {
                "make_scalar": True,
                "expand_around": False,
                "n_trials": args.n_trials,
                "multi_objective": False,
                "parameter_config": PARAMETER_CONFIG,
                **({"overfitting_penalty": args.overfitting_penalty}
                   if args.overfitting_penalty is not None else {}),
                **({"min_train_returns_over_hodl": args.min_train_ret}
                   if args.min_train_ret is not None else {}),
            },
        }


def build_fingerprint(objective, args, tokens, noise_arrays_path=None, arb_freq=None):
    """Build run fingerprint with calibrated noise model."""
    if args.noise_model == "mm_observed" and noise_arrays_path is not None:
        noise_block = {
            "noise_trader_ratio": 0.0,
            "noise_model": "mm_observed",
            "noise_arrays_path": noise_arrays_path,
        }
        freq = arb_freq or 5
    elif args.noise_model == "market_linear" and noise_arrays_path is not None:
        _arr = np.load(noise_arrays_path)
        noise_block = {
            "noise_trader_ratio": 0.0,
            "noise_model": "market_linear",
            "noise_arrays_path": noise_arrays_path,
            "reclamm_noise_params": {
                "tvl_mean": float(_arr["tvl_mean"]),
                "tvl_std": float(_arr["tvl_std"]),
            },
        }
        freq = arb_freq or 5
    else:
        noise_block = {
            "noise_trader_ratio": 0.0,
            "noise_model": "calibrated",
            "reclamm_noise_params": {
                f"c_{i}": NOISE_COEFFS_LEGACY[i] for i in range(8)
            },
        }
        freq = LEGACY_ARB_FREQUENCY

    return {
        "rule": "reclamm",
        "tokens": tokens,
        "startDateString": args.start_date,
        "endDateString": args.end_date,
        "endTestDateString": args.end_test_date,
        "initial_pool_value": args.initial_pool_value,
        "do_arb": True,
        "arb_frequency": freq,
        "fees": args.fees,
        "gas_cost": args.gas_cost,
        "arb_fees": 0.0,
        "protocol_fee_split": 0.25,
        **noise_block,
        "return_val": objective,
        "reclamm_interpolation_method": args.interpolation,
        "reclamm_centeredness_scaling": args.centeredness_scaling,
        "reclamm_learn_arc_length_speed": False,
        "reclamm_use_shift_exponent": True,
        **({"bout_offset": args.bout_offset} if args.bout_offset is not None else {}),
        "optimisation_settings": _build_opt_settings(args),
    }


def run_single(objective, args, tokens, pool_id, noise_arrays_path=None, arb_freq=None):
    """Run Optuna tuning for a single objective."""
    print(f"\n{'='*60}")
    print(f"  Objective: {objective}")
    print(f"  Noise model: {args.noise_model}")
    print(f"  Method: {args.method}")
    print(f"  Tokens: {'/'.join(tokens)} ({pool_id})")
    print(f"  Train: {args.start_date} → {args.end_date}")
    print(f"  Test:  {args.end_date} → {args.end_test_date}")
    if arb_freq:
        print(f"  Arb frequency: {arb_freq} min (learned)")
    print(f"{'='*60}\n")

    fp = build_fingerprint(objective, args, tokens, noise_arrays_path, arb_freq)
    result = train_on_historic_data(fp, verbose=True)

    if result is not None:
        print(f"\n=== Result ({objective}) ===")
        for k, v in result.items():
            print(f"  {k}: {v}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Tune reClAMM params with calibrated 8-covariate noise model"
    )
    parser.add_argument("--method", default="optuna",
                        choices=["optuna", "bfgs", "cma_es"],
                        help="Optimisation method")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials (ignored for bfgs)")
    parser.add_argument("--n-parameter-sets", type=int, default=1,
                        help="Number of parameter sets for bfgs")
    parser.add_argument("--bfgs-maxiter", type=int, default=100)
    parser.add_argument("--bfgs-tol", type=float, default=1e-6)
    parser.add_argument("--bfgs-eval-points", type=int, default=20,
                        help="Number of evaluation points for bfgs")
    # CMA-ES
    parser.add_argument("--cma-generations", type=int, default=300)
    parser.add_argument("--cma-sigma0", type=float, default=0.5,
                        help="Initial step size for CMA-ES")
    parser.add_argument("--cma-pop-size", type=int, default=None,
                        help="Population size (None = auto)")
    parser.add_argument("--cma-eval-points", type=int, default=20)
    parser.add_argument("--min-train-ret", type=float, default=-0.5,
                        help="Reject trials with IS returns_over_hodl below this")
    parser.add_argument("--tokens", nargs=2, default=DEFAULT_TOKENS,
                        help="Token pair (default: AAVE ETH)")
    parser.add_argument("--pool-id", default=DEFAULT_POOL_ID,
                        help="Pool ID prefix for noise model lookup")
    parser.add_argument("--noise-model", default="market_linear",
                        choices=["calibrated", "market_linear", "mm_observed"],
                        help="Noise model variant")
    parser.add_argument("--artifact-dir",
                        default="results/linear_market_noise",
                        help="Artifact dir for market_linear or mm_observed model")
    parser.add_argument("--competitor-tvl-path",
                        default="results/competitor_tvl/competitor_tvl.npz",
                        help="Path to competitor TVL data (mm_observed only)")
    parser.add_argument("--initial-pool-value", type=float, default=20_000_000.0,
                        help="Initial pool TVL in USD (default: 20M)")
    parser.add_argument("--fees", type=float, default=0.0025,
                        help="Pool fee rate (default: 0.0025 matching calibration)")
    parser.add_argument("--gas-cost", type=float, default=1.0)
    parser.add_argument("--objective", default="fee_revenue_over_value",
                        choices=OBJECTIVES)
    parser.add_argument("--all-objectives", action="store_true",
                        help="Run all three objectives sequentially")
    parser.add_argument("--interpolation", default="geometric",
                        choices=["geometric", "constant_arc_length"])
    parser.add_argument("--centeredness-scaling", action="store_true")
    parser.add_argument("--start-date", default="2024-06-01 00:00:00")
    parser.add_argument("--end-date", default="2025-06-01 00:00:00",
                        help="End of training / start of test")
    parser.add_argument("--end-test-date", default="2026-03-01 00:00:00",
                        help="End of test (latest available data)")
    parser.add_argument("--bout-offset", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--overfitting-penalty", type=float, default=None)
    parser.add_argument("--robust-temperature", type=float, default=None,
                        help="Robust aggregation temperature (lower=more robust)."
                             " None=standard mean. Try 0.5-2.0.")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--pr-max", type=float, default=None,
                        help="Override max price_ratio (default: 200)")
    args = parser.parse_args()

    if args.pr_max is not None:
        PARAMETER_CONFIG["price_ratio"]["high"] = args.pr_max

    if args.all_objectives:
        objectives = OBJECTIVES
    else:
        objectives = [args.objective]

    tokens = args.tokens
    pool_id = args.pool_id

    # Precompute noise arrays once
    noise_arrays_path = None
    arb_freq = None
    if args.noise_model == "market_linear":
        noise_arrays_path, arb_freq = _build_market_linear_arrays(args, pool_id, tokens)
    elif args.noise_model == "mm_observed":
        noise_arrays_path, arb_freq = _build_mm_observed_arrays(args, pool_id, tokens)

    all_results = {}
    for obj in objectives:
        result = run_single(obj, args, tokens, pool_id, noise_arrays_path, arb_freq)
        all_results[obj] = result

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
