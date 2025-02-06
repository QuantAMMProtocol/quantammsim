Run Fingerprints
================

This guide explains how to configure runs using the `run_fingerprints` dictionary of settings that is used by high-level simulation runners.

Basic Settings
--------------

Core simulation configuration:

.. code-block:: python

    run_fingerprint = {
        "startDateString": "2024-02-03",     # Training start date
        "endDateString": "2024-06-03",       # Training end date
        "endTestDateString": "2024-07-03",   # Test period end date
        "tokens": ["BTC", "ETH", "USDC"],    # Assets to simulate
        "rule": "momentum",                  # Strategy/pool type
        "initial_pool_value": 1000000.0,     # Starting pool value
    }

Optimization Settings
---------------------

Control the training process:

.. code-block:: python

    run_fingerprint["optimisation_settings"] = {
        "base_lr": 0.01,                     # Initial learning rate
        "optimiser": "sgd",                  # Optimizer type
        "batch_size": 8,                     # Training batch size
        "n_iterations": 1000,                # Training iterations
        "n_parameter_sets": 3,               # Number of parallel parameters
        "training_data_kind": "historic",    # Data source type
    }

Initial Parameters
------------------

Starting values for strategy parameters:

.. code-block:: python

    run_fingerprint.update({
        "initial_memory_length": 10.0,       # Memory parameter
        "initial_k_per_day": 20,            # Trading intensity
        "initial_weights_logits": 1.0,      # Starting weights
        "initial_log_amplitude": -10.0,     # Signal amplitude
    })

Runtime Behavior
----------------

Configure execution details:

.. code-block:: python

    run_fingerprint.update({
        "maximum_change": 3e-4,             # Max weight change per minute
        "chunk_period": 60,                 # Update frequency (minutes)
        "fees": 0.0,                        # Trading fees
        "arb_fees": 0.0,                    # Arbitrage fees
        "gas_cost": 0.0,                    # Transaction costs
        "do_arb": True,                     # Enable arbitrage
        "arb_frequency": 1,                 # Arb check frequency
    })

Advanced Optimization
---------------------

For hyperparameter optimization using Optuna:

.. code-block:: python

    run_fingerprint["optimisation_settings"]["optuna_settings"] = {
        "n_trials": 20,                    # Number of trials
        "n_jobs": 4,                       # Parallel workers
        "timeout": 7200,                   # Max runtime (seconds)
        "parameter_config": {
            "memory_length": {
                "low": 1,                  # Min value
                "high": 200,               # Max value
                "log_scale": True,         # Use log scale
            },
            # ... other parameters ...
        }
    }

Implementation Notes
--------------------

- All settings have defaults in ``run_fingerprint_defaults``
- Settings are validated before use
- Some combinations may be invalid for certain strategies

For how the run_fingerprint is used in simulations, see :func:`quantammsim.runners.train_on_historic_data` and :func:`quantammsim.runners.do_run_on_historic_data`.
