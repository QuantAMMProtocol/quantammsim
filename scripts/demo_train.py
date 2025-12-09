import jax.numpy as jnp
from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
)
from quantammsim.runners.jax_runners import train_on_historic_data

# Default fingerprint used as base for all pools
DEFAULT_FINGERPRINT = {
    "startDateString": "2024-01-01 00:00:00",
    "endDateString": "2025-01-01 00:00:00",
    "endTestDateString": "2025-12-06 00:00:00",
    "chunk_period": 1440, # 1 day
    "weight_interpolation_period": 1440, # 1 day
    "bout_offset": 24 * 60 * 7 * 4, # 4 weeks
    "optimisation_settings": {
        "method": "gradient_descent", # Training method
        "base_lr": 0.05, # (Initial) learning rate
        "optimiser": "adam", # Optimiser
        "batch_size": 8, # Batch size
        "n_iterations": 1000, # Number of iterations
        "n_parameter_sets": 4, # Number of parameter sets to train in parallel
    },
}

EXAMPLE_CONFIGS = {
    "momentum_aave_pendle": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["AAVE", "PENDLE"],
            "rule": "momentum",
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
