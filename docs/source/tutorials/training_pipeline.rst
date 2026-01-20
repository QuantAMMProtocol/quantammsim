Training Pipeline
=================

This guide explains how to train QuantAMM pool strategies using quantammsim's gradient-based optimization.

Overview
--------

The training pipeline uses JAX for automatic differentiation to optimize strategy parameters. The main entry point is ``train_on_historic_data``:

.. code-block:: python

    from quantammsim.runners.jax_runners import train_on_historic_data

    train_on_historic_data(
        run_fingerprint=run_fingerprint,
        iterations_per_print=10,
        verbose=True
    )

Basic Training Setup
--------------------

1. Configure the Run Fingerprint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint = {
        # Data settings
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-12-01 00:00:00",
        "endTestDateString": "2024-01-01 00:00:00",  # Optional test period
        "tokens": ["BTC", "ETH"],

        # Strategy
        "rule": "momentum",

        # Pool settings
        "initial_pool_value": 1000000.0,
        "fees": 0.003,  # 30 bps
        "do_arb": True,

        # Optimization settings
        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "adam",
            "base_lr": 0.1,
            "n_iterations": 1000,
            "batch_size": 8,
            "n_parameter_sets": 4,
        },

        # Objective
        "return_val": "sharpe",
    }

2. Run Training
~~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.runners.jax_runners import train_on_historic_data

    train_on_historic_data(
        run_fingerprint=run_fingerprint,
        verbose=True
    )

Optimizer Configuration
-----------------------

Available Optimizers
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Adam (recommended)
    run_fingerprint["optimisation_settings"]["optimiser"] = "adam"

    # AdamW (with weight decay)
    run_fingerprint["optimisation_settings"]["optimiser"] = "adamw"

    # SGD
    run_fingerprint["optimisation_settings"]["optimiser"] = "sgd"

Learning Rate Schedules
~~~~~~~~~~~~~~~~~~~~~~~

**Constant Learning Rate:**

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "lr_schedule_type": "constant",
        "base_lr": 0.1,
    })

**Warmup with Cosine Decay:**

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "lr_schedule_type": "warmup_cosine",
        "base_lr": 0.1,
        "warmup_steps": 100,
        "min_lr": 1e-6,
    })

**Plateau-Based Decay:**

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "use_plateau_decay": True,
        "decay_lr_plateau": 100,   # Iterations without improvement
        "decay_lr_ratio": 0.8,     # Multiply LR by this factor
    })

Gradient Clipping
~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "use_gradient_clipping": True,
        "clip_norm": 10.0,
    })

Batch Training
--------------

The training uses batched gradient computation for efficiency:

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "batch_size": 8,              # Number of time periods per batch
        "n_parameter_sets": 4,        # Parallel parameter sets to optimize
        "sample_method": "uniform",   # How to sample training periods
    })

The ``bout_offset`` parameter controls training data variety:

.. code-block:: python

    # Train on different starting points within this offset window
    run_fingerprint["bout_offset"] = 24 * 60 * 7  # 1 week in minutes

Initial Parameters
------------------

Set starting values for optimization:

.. code-block:: python

    run_fingerprint.update({
        "initial_memory_length": 10.0,      # EWMA memory in days
        "initial_k_per_day": 20,            # Trading intensity
        "initial_weights_logits": 1.0,      # Starting weight distribution
        "initial_log_amplitude": 0.0,       # Signal amplitude
        "initial_raw_width": 0.0,           # Channel width
        "initial_raw_exponents": 0.0,       # Power exponents
        "initial_pre_exp_scaling": 0.5,     # Pre-exponential scaling
    })

Training Objectives
-------------------

Set the objective function:

.. code-block:: python

    # Maximize Sharpe ratio (default)
    run_fingerprint["return_val"] = "sharpe"

    # Maximize daily Sharpe
    run_fingerprint["return_val"] = "daily_sharpe"

    # Maximize total return
    run_fingerprint["return_val"] = "returns"

    # Maximize return over holding initial portfolio
    run_fingerprint["return_val"] = "returns_over_hodl"

    # Maximize Sortino ratio
    run_fingerprint["return_val"] = "sortino"

Advanced: Hessian-Based Training
--------------------------------

For second-order optimization (experimental):

.. code-block:: python

    run_fingerprint["optimisation_settings"]["train_on_hessian_trace"] = True

This uses Hessian trace information but is more computationally expensive.

Backpropagation Module
----------------------

The training pipeline uses the ``quantammsim.training.backpropagation`` module internally. Key functions:

**Objective Factories:**

- ``batched_objective_factory`` - Creates batched loss function
- ``batched_objective_with_hessian_factory`` - Includes Hessian computation

**Update Factories:**

- ``update_factory`` - Basic gradient update
- ``update_factory_with_optax`` - Uses Optax optimizers (Adam, SGD, etc.)
- ``update_with_hessian_factory_with_optax`` - Hessian-aware updates

**Optimizer Creation:**

.. code-block:: python

    from quantammsim.training.backpropagation import create_optimizer_chain

    # Creates an Optax optimizer chain based on run_fingerprint settings
    optimizer = create_optimizer_chain(run_fingerprint)

Straight-Through Estimators
---------------------------

For improved gradient flow through clipping operations:

.. code-block:: python

    run_fingerprint.update({
        "ste_max_change": True,      # STE for weight change clipping
        "ste_min_max_weight": True,  # STE for min/max weight bounds
    })

These allow gradients to flow through otherwise non-differentiable clipping operations.

Monitoring Training
-------------------

Training progress is printed at intervals:

.. code-block:: python

    train_on_historic_data(
        run_fingerprint=run_fingerprint,
        iterations_per_print=10,  # Print every 10 iterations
        verbose=True
    )

Output includes:

- Current iteration
- Training objective value
- Learning rate
- Best parameters found

Saving and Loading
------------------

Training state is automatically saved. To resume:

.. code-block:: python

    train_on_historic_data(
        run_fingerprint=run_fingerprint,
        run_location="path/to/saved/run",
        force_init=False  # Don't reinitialize, load existing state
    )

Example: Complete Training Script
---------------------------------

.. code-block:: python

    from quantammsim.runners.jax_runners import train_on_historic_data

    run_fingerprint = {
        # Data
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-10-01 00:00:00",
        "endTestDateString": "2024-01-01 00:00:00",
        "tokens": ["BTC", "ETH", "SOL"],

        # Strategy
        "rule": "momentum",
        "initial_pool_value": 1000000.0,
        "fees": 0.003,
        "do_arb": True,
        "arb_quality": 1.0,

        # Weight calculation
        "chunk_period": 1440,  # Daily
        "maximum_change": 0.001,

        # Initial params
        "initial_memory_length": 10.0,
        "initial_k_per_day": 20,

        # Optimization
        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "adam",
            "base_lr": 0.1,
            "n_iterations": 500,
            "batch_size": 8,
            "n_parameter_sets": 4,
            "lr_schedule_type": "warmup_cosine",
            "warmup_steps": 50,
            "min_lr": 1e-5,
            "use_gradient_clipping": True,
            "clip_norm": 10.0,
        },

        # Objective
        "return_val": "sharpe",
    }

    train_on_historic_data(
        run_fingerprint=run_fingerprint,
        iterations_per_print=10,
        verbose=True
    )

See Also
--------

- :doc:`../user_guide/run_fingerprints` - Complete run fingerprint reference
- :doc:`tuning` - Optuna hyperparameter optimization
- :func:`~quantammsim.runners.jax_runners.train_on_historic_data` - API reference
