import jax.numpy as jnp
from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
)
from quantammsim.runners.jax_runners import train_on_historic_data

# Default fingerprint used as base for all pools
DEFAULT_FINGERPRINT = {
    "startDateString": "2021-01-01 00:00:00",
    "endDateString": "2024-06-01 00:00:00",
    "endTestDateString": "2024-11-30 00:00:00",
    "chunk_period": 1440,
    "weight_interpolation_period": 1440,
    "fees": 0.0,
    "gas_cost": 0.0,
    "use_alt_lamb": False,
    "optimisation_settings": {
        "method": "gradient_descent",
    },
}

EXAMPLE_CONFIGS = {
    "momentum_btc_usdc": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["BTC", "USDC"],
            "rule": "momentum",
        },
    },
    "anti_momentum_btc_usdc": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["BTC", "USDC"],
            "rule": "anti_momentum",
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
