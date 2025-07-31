import jax.numpy as jnp
from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
)
from quantammsim.runners.jax_runners import train_on_historic_data

# Default fingerprint used as base for all pools
DEFAULT_FINGERPRINT = {
    "startDateString": "2021-01-01 00:00:00",
    "endDateString": "2024-06-01 00:00:00",
    "endTestDateString": "2024-11-01 00:00:00",
    "chunk_period": 60,
    "weight_interpolation_period": 60,
    "fees": 0.0,
    "gas_cost": 0.0,
    "use_alt_lamb": False,
    "bout_offset": 24 * 60 * 90,
    "optimisation_settings": {
        "method": "gradient_descent",
        "optimiser": "adam",
        "use_gradient_clipping": True,
        "warmup_steps": 1000,
        "batch_size": 4,
        "n_parameter_sets": 3,
        "base_lr": 0.05,
        "use_plateau_decay": True,
        "decay_lr_plateau": 100,
    },
}

EXAMPLE_CONFIGS = {
    "momentum_btc_usdc": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["BTC", "PAXG", "USDC"],
            "rule": "flexible_channel",
            "n_iterations": 5000,
            "initial_raw_alpha":            [0.25, 0.20, 0.15],
            "initial_raw_exponents_up":     [0.30, 0.25, 0.35],
            "initial_raw_exponents_down":   [0.60, 0.70, 0.80],
            "initial_raw_exponents":        [0.30, 0.30, 0.30],

            "initial_risk_on":              [0.8, 0.2, 0.05],
            "initial_risk_off":             [0.05, 0.3, 0.9],

            "initial_raw_kelly_kappa":      [0.40, 0.35, 0.45],

            "initial_logit_lamb_vol":       [1.00, 0.80, 1.20],   

            "initial_raw_entropy_floor":    -2.0,
            "initial_memory_length_drawdown":[6.0, 10.0, 16.0],
        },
    },
}

if __name__ == "__main__":
    for name, config in EXAMPLE_CONFIGS.items():
        print(f"\nTraining {name}...")
        train_on_historic_data(
            run_fingerprint=config["fingerprint"],
            verbose=True,
        )
