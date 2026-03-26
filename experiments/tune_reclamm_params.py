"""Optuna tuning of reClAMM pool parameters via train_on_historic_data.

Usage:
    cd <repo-root>
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim-reclamm

    # Fee revenue objective (default)
    python experiments/tune_reclamm_params.py

    # Sharpe objective with constant arc-length
    python experiments/tune_reclamm_params.py --objective daily_log_sharpe \
        --interpolation constant_arc_length

    # More trials, custom fees
    python experiments/tune_reclamm_params.py --n-trials 200 --fees 0.005

    # With calibrated 8-covariate noise model and arb frequency from calibration
    python experiments/tune_reclamm_params.py --noise-model calibrated \
        --noise-params-json results/mlp_calibration/option_c_reduced.json \
        --noise-pool-id 0x9d1fcf346ea1b0
"""

import argparse
import json
import math
from quantammsim.runners.jax_runners import train_on_historic_data

PARAMETER_CONFIG = {
    "price_ratio": {"low": 1.01, "high": 200.0, "log_scale": True, "scalar": True},
    "centeredness_margin": {"low": 0.01, "high": 0.99, "scalar": True},
    "shift_exponent": {"low": 1e-5, "high": 125.0, "log_scale": True, "scalar": True},
}

ARC_LENGTH_SPEED_CONFIG = {
    "arc_length_speed": {"low": 1e-7, "high": 1e-2, "log_scale": True, "scalar": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--fees", type=float, default=0.003)
    parser.add_argument("--gas-cost", type=float, default=1.0)
    parser.add_argument("--objective", default="fee_revenue_over_value")
    parser.add_argument("--interpolation", default="geometric",
                        choices=["geometric", "constant_arc_length"])
    parser.add_argument("--centeredness-scaling", action="store_true")
    parser.add_argument("--noise-trader-ratio", type=float, default=0.0)
    parser.add_argument("--noise-model", default=None,
                        choices=["ratio", "loglinear", "calibrated", "arb_only"],
                        help="Noise volume model (default: ratio via noise-trader-ratio)")
    parser.add_argument("--noise-params-json", default=None,
                        help="JSON file with per-pool calibration results")
    parser.add_argument("--noise-pool-id", default=None,
                        help="Pool ID to load noise params for (from --noise-params-json)")
    parser.add_argument("--arb-frequency", type=int, default=None,
                        help="Arb frequency in minutes (default: from calibrated cadence or 1)")
    parser.add_argument("--start-date", default="2024-06-01 00:00:00")
    parser.add_argument("--end-date", default="2025-01-01 00:00:00",
                        help="End of training / start of test")
    parser.add_argument("--end-test-date", default="2025-06-01 00:00:00")
    parser.add_argument("--bout-offset", type=int, default=None,
                        help="bout_offset in minutes (default: 10080 = 7 days)")
    parser.add_argument("--val-fraction", type=float, default=None,
                        help="Validation holdout fraction (default: 0.2, use 0 to disable)")
    parser.add_argument("--overfitting-penalty", type=float, default=None,
                        help="Overfitting penalty weight (default: 0.2)")
    parser.add_argument("--n-eval-points", type=int, default=None,
                        help="Number of evaluation sub-windows (default: 20, use 1 for full-window)")
    args = parser.parse_args()

    learn_speed = args.interpolation == "constant_arc_length"
    param_config = {**PARAMETER_CONFIG}
    if learn_speed:
        param_config.update(ARC_LENGTH_SPEED_CONFIG)

    # --- Noise model setup ---
    pool_tokens = ["AAVE", "ETH"]  # default
    noise_fp = {"noise_trader_ratio": args.noise_trader_ratio}
    if args.noise_model:
        noise_fp["noise_model"] = args.noise_model
    if args.noise_params_json and args.noise_pool_id:
        with open(args.noise_params_json) as f:
            all_results = json.load(f)
        # Support both {"option_c_reduced": {pid: ...}} and {pid: ...} formats
        pool_results = all_results
        for key in all_results:
            if isinstance(all_results[key], dict) and args.noise_pool_id in all_results[key]:
                pool_results = all_results[key]
                break
        pool_data = pool_results[args.noise_pool_id]
        coeffs = pool_data["noise_coeffs"]
        if len(coeffs) == 8:
            # Full 8-covariate model: [intercept, log_tvl, log_sigma,
            #   tvl*sigma, tvl*fee, sigma*fee, dow_sin, dow_cos]
            noise_fp["reclamm_noise_params"] = {
                f"c_{i}": c for i, c in enumerate(coeffs)
            }
        elif len(coeffs) == 4:
            # Reduced 4-covariate model: [intercept, log_tvl, dow_sin, dow_cos]
            # Map to c_0, c_1, c_6, c_7 (sigma/fee terms stay at 0)
            noise_fp["reclamm_noise_params"] = {
                "c_0": coeffs[0], "c_1": coeffs[1],
                "c_6": coeffs[2], "c_7": coeffs[3],
            }
        else:
            raise ValueError(f"Expected 4 or 8 noise_coeffs, got {len(coeffs)}")
        # Derive arb_frequency from calibrated cadence if not explicitly set
        if args.arb_frequency is None:
            log_cad = pool_data["log_cadence"]
            args.arb_frequency = max(1, round(math.exp(log_cad)))
            print(f"  arb_frequency={args.arb_frequency} "
                  f"(from log_cadence={log_cad:.2f}, "
                  f"cadence={math.exp(log_cad):.1f} min)")
        # Use pool's fee and gas from calibration as defaults
        if "fee" in pool_data:
            args.fees = pool_data["fee"]
        if "gas_usd" in pool_data:
            args.gas_cost = pool_data["gas_usd"]
        # Pick up token pair from calibration
        # Map on-chain names (WETH, WBTC) to data-file names (ETH, BTC)
        _TOKEN_MAP = {"WETH": "ETH", "WBTC": "BTC"}
        if "tokens" in pool_data:
            pool_tokens = [
                _TOKEN_MAP.get(t, t) for t in pool_data["tokens"].split(",")
            ]
        print(f"  tokens={pool_tokens}, fee={args.fees}, gas={args.gas_cost}")

    fp = {
        "rule": "reclamm",
        "tokens": pool_tokens,
        "startDateString": args.start_date,
        "endDateString": args.end_date,
        "endTestDateString": args.end_test_date,
        "initial_pool_value": 1_000_000.0,
        "do_arb": True,
        **({"arb_frequency": args.arb_frequency} if args.arb_frequency is not None else {}),
        "fees": args.fees,
        "gas_cost": args.gas_cost,
        "arb_fees": 0.0,
        "protocol_fee_split": 0.5,
        **noise_fp,
        "return_val": args.objective,
        "reclamm_interpolation_method": args.interpolation,
        "reclamm_centeredness_scaling": args.centeredness_scaling,
        "reclamm_learn_arc_length_speed": learn_speed,
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
                "parameter_config": param_config,
                **({"overfitting_penalty": args.overfitting_penalty} if args.overfitting_penalty is not None else {}),
                **({"n_evaluation_points": args.n_eval_points} if args.n_eval_points is not None else {}),
            },
        },
    }

    result = train_on_historic_data(fp, verbose=True)
    if result is not None:
        print(f"\n=== Result ===")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
