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
"""

import argparse
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
    args = parser.parse_args()

    learn_speed = args.interpolation == "constant_arc_length"
    param_config = {**PARAMETER_CONFIG}
    if learn_speed:
        param_config.update(ARC_LENGTH_SPEED_CONFIG)

    fp = {
        "rule": "reclamm",
        "tokens": ["AAVE", "ETH"],
        "startDateString": args.start_date,
        "endDateString": args.end_date,
        "endTestDateString": args.end_test_date,
        "initial_pool_value": 1_000_000.0,
        "do_arb": True,
        "fees": args.fees,
        "gas_cost": args.gas_cost,
        "arb_fees": 0.0,
        "protocol_fee_split": 0.5,
        "noise_trader_ratio": args.noise_trader_ratio,
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
