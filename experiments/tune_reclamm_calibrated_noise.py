"""Optuna tuning of reClAMM pool parameters with calibrated noise models.

Supports two noise model modes:
  --noise-model calibrated   (legacy 8-covariate model)
  --noise-model market_linear (new per-pool model with market features)

The market_linear model uses precomputed daily arrays from the per-pool
calibrated noise model artifact (results/linear_market_noise/). It evaluates:

    log(V_noise) = base_t + tvl_coeff_t * log(effective_TVL)

where base_t absorbs all non-TVL terms (market regime, token volatility,
pair volatility, day-of-week, cross-pool volumes) and tvl_coeff_t is the
effective TVL coefficient including interaction terms.

Pool: 0x9d1fcf346ea1b0 = AAVE/WETH Mainnet

Usage:
    cd <repo-root>
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim_reclamm_public

    # New market_linear model (default)
    python experiments/tune_reclamm_calibrated_noise.py

    # Legacy 8-covariate model
    python experiments/tune_reclamm_calibrated_noise.py --noise-model calibrated

    # All three objectives
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

POOL_ID = "0x9d1fcf346ea1b0"  # AAVE/WETH Mainnet

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

OBJECTIVES = ["daily_log_sharpe", "returns_over_hodl", "fee_revenue_over_value"]


def _build_market_linear_arrays(args):
    """Precompute noise arrays from the per-pool market noise model artifact."""
    from quantammsim.calibration.noise_model_arrays import build_simulator_arrays

    # Parse dates — strip time component for the array builder
    start = args.start_date.split(" ")[0]
    end = args.end_test_date.split(" ")[0]

    print(f"  Building market_linear noise arrays for {POOL_ID}...")
    print(f"  Date range: {start} → {end}")
    arrays = build_simulator_arrays(
        token_a="AAVE",
        token_b="ETH",
        start_date=start,
        end_date=end,
        artifact_dir=args.artifact_dir,
        pool_id=POOL_ID,
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
    arrays_path = os.path.join(cache_dir, f"{POOL_ID}_{start}_{end}.npz")
    np.savez(arrays_path,
             noise_base=arrays["noise_base"],
             noise_tvl_coeff=arrays["noise_tvl_coeff"],
             tvl_mean=arrays["tvl_mean"],
             tvl_std=arrays["tvl_std"])
    print(f"  Saved arrays: {arrays_path}")

    # Get learned cadence from artifact
    from quantammsim.calibration.noise_model_arrays import load_artifact, _find_pool_index
    art, meta = load_artifact(args.artifact_dir)
    pool_idx = _find_pool_index(POOL_ID, meta["pool_ids"])
    if pool_idx >= 0:
        learned_cadence = float(np.exp(art["log_cadence"][pool_idx]))
        print(f"  Learned cadence: {learned_cadence:.1f} min")
    else:
        learned_cadence = 5.0
        print(f"  Pool not in calibration set, using default cadence: {learned_cadence}")

    return arrays_path, max(1, round(learned_cadence))


def build_fingerprint(objective, args, noise_arrays_path=None, arb_freq=None):
    """Build run fingerprint with calibrated noise model."""
    if args.noise_model == "market_linear" and noise_arrays_path is not None:
        # Load tvl standardization stats from the saved arrays
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
        "tokens": ["AAVE", "ETH"],
        "startDateString": args.start_date,
        "endDateString": args.end_date,
        "endTestDateString": args.end_test_date,
        "initial_pool_value": args.initial_pool_value,
        "do_arb": True,
        "arb_frequency": freq,
        "fees": args.fees,
        "gas_cost": args.gas_cost,
        "arb_fees": 0.0,
        "protocol_fee_split": 0.5,
        **noise_block,
        "return_val": objective,
        "reclamm_interpolation_method": args.interpolation,
        "reclamm_centeredness_scaling": args.centeredness_scaling,
        "reclamm_learn_arc_length_speed": False,
        "reclamm_use_shift_exponent": True,
        **({"bout_offset": args.bout_offset} if args.bout_offset is not None else {}),
        "optimisation_settings": {
            "method": "optuna",
            "n_parameter_sets": 1,
            **({"val_fraction": args.val_fraction} if args.val_fraction is not None else {}),
            "optuna_settings": {
                "make_scalar": True,
                "expand_around": False,
                "n_trials": args.n_trials,
                "multi_objective": False,
                "parameter_config": PARAMETER_CONFIG,
                **({"overfitting_penalty": args.overfitting_penalty} if args.overfitting_penalty is not None else {}),
            },
        },
    }


def run_single(objective, args, noise_arrays_path=None, arb_freq=None):
    """Run Optuna tuning for a single objective."""
    print(f"\n{'='*60}")
    print(f"  Objective: {objective}")
    print(f"  Noise model: {args.noise_model}")
    print(f"  Pool: AAVE/WETH Mainnet ({POOL_ID})")
    print(f"  Train: {args.start_date} → {args.end_date}")
    print(f"  Test:  {args.end_date} → {args.end_test_date}")
    if arb_freq:
        print(f"  Arb frequency: {arb_freq} min (learned)")
    print(f"{'='*60}\n")

    fp = build_fingerprint(objective, args, noise_arrays_path, arb_freq)
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
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--noise-model", default="market_linear",
                        choices=["calibrated", "market_linear"],
                        help="Noise model variant")
    parser.add_argument("--artifact-dir",
                        default="results/linear_market_noise",
                        help="Artifact dir for market_linear model")
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
    parser.add_argument("--start-date", default="2025-08-03 00:00:00")
    parser.add_argument("--end-date", default="2025-12-01 00:00:00",
                        help="End of training / start of test")
    parser.add_argument("--end-test-date", default="2026-02-18 00:00:00",
                        help="End of test (latest available data)")
    parser.add_argument("--bout-offset", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--overfitting-penalty", type=float, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    if args.all_objectives:
        objectives = OBJECTIVES
    else:
        objectives = [args.objective]

    # Precompute noise arrays once (if using market_linear)
    noise_arrays_path = None
    arb_freq = None
    if args.noise_model == "market_linear":
        noise_arrays_path, arb_freq = _build_market_linear_arrays(args)

    all_results = {}
    for obj in objectives:
        result = run_single(obj, args, noise_arrays_path, arb_freq)
        all_results[obj] = result

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
