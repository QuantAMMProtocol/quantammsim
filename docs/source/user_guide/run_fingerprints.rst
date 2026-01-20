Run Fingerprints
================

This guide explains how to configure runs using the ``run_fingerprint`` dictionary of settings that is used by high-level simulation runners.

Basic Settings
--------------

Core simulation configuration:

.. code-block:: python

    run_fingerprint = {
        "startDateString": "2024-02-03 00:00:00",  # Simulation start date
        "endDateString": "2024-06-03 00:00:00",    # Training end / simulation end date
        "endTestDateString": "2024-07-03 00:00:00", # Test period end (optional)
        "tokens": ["BTC", "ETH", "USDC"],           # Assets to simulate
        "rule": "momentum",                         # Strategy/pool type
        "initial_pool_value": 1000000.0,           # Starting pool value in USD
    }

Available Pool Rules
~~~~~~~~~~~~~~~~~~~~

The ``rule`` parameter accepts the following values:

**Static Weight Pools:**

* ``balancer`` - Standard Balancer-style constant-weight pool
* ``hodl`` - Simple hold strategy (no rebalancing)
* ``cow`` - CoW AMM pool
* ``gyroscope`` - Gyroscope ECLP pool

**Dynamic Weight Pools (QuantAMM):**

* ``momentum`` - Trend-following strategy
* ``anti_momentum`` - Counter-trend strategy
* ``mean_reversion_channel`` - Mean reversion with channel breakout
* ``power_channel`` - Power-law weighted momentum
* ``difference_momentum`` - Differential momentum strategy
* ``min_variance`` - Minimum variance portfolio
* ``index_market_cap`` - Market-cap weighted index

**Hooked Pools:**

Add hooks using the ``hookname__poolrule`` prefix format:

* ``lvr__momentum`` - Momentum with Loss-Versus-Rebalancing tracking
* ``rvr__balancer`` - Balancer with Rebalancing-Versus-Rebalancing tracking
* ``bounded__momentum`` - Momentum with per-asset weight bounds

See :doc:`per_asset_bounds` for details on bounded weight pools.

Optimization Settings for Gradient Descent
------------------------------------------

Control the training process:

.. code-block:: python

    run_fingerprint["optimisation_settings"] = {
        "method": "gradient_descent",
        "base_lr": 0.1,                      # Initial learning rate
        "optimiser": "adam",                 # Optimizer: "adam", "adamw", or "sgd"
        "batch_size": 8,                     # Training batch size
        "n_iterations": 1000,                # Training iterations
        "n_parameter_sets": 4,               # Parallel parameter sets
        "training_data_kind": "historic",    # Data source type
    }

Available Optimizers
~~~~~~~~~~~~~~~~~~~~

* ``adam`` - Adam optimizer (recommended for most cases)
* ``adamw`` - Adam with weight decay
* ``sgd`` - Stochastic gradient descent

Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "lr_schedule_type": "constant",      # "constant" or "warmup_cosine"
        "warmup_steps": 100,                 # Warmup steps for cosine schedule
        "min_lr": 1e-6,                      # Minimum learning rate
        "use_plateau_decay": False,          # Decay LR on plateau
        "decay_lr_plateau": 100,             # Iterations before decay
        "decay_lr_ratio": 0.8,               # Decay multiplier
    })

Gradient Clipping
~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "use_gradient_clipping": True,      # Enable gradient clipping
        "clip_norm": 10.0,                   # Maximum gradient norm
    })

Initial Parameters
~~~~~~~~~~~~~~~~~~

Starting values for strategy parameters, used only during training.

.. code-block:: python

    run_fingerprint.update({
        "initial_memory_length": 10.0,       # Memory parameter
        "initial_k_per_day": 20,            # Trading intensity
        "initial_weights_logits": 1.0,      # Starting weights. Provide a jnp.array of length = num of tokens for per-token allocation, otherwise defaults to uniform weights.
        "initial_log_amplitude": -10.0,     # Signal amplitude
    })

Optimization Settings for Gradient-Free Descent
-----------------------------------------------

For hyperparameter optimization using Optuna:

.. code-block:: python

    run_fingerprint["optimisation_settings"]["method"] = "optuna"
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

Complete Optuna Settings Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    optuna_settings = {
        # Study configuration
        "study_name": None,              # Auto-generated if None
        "storage": {
            "type": "sqlite",            # "sqlite", "mysql", or "postgresql"
            "url": None,                 # e.g., "sqlite:///studies.db"
        },

        # Trial settings
        "n_trials": 20,                  # Number of optimization trials
        "n_jobs": 4,                     # Parallel workers
        "timeout": 7200,                 # Max optimization time (seconds)
        "n_startup_trials": 10,          # Random trials before TPE sampler

        # Early stopping
        "early_stopping": {
            "enabled": False,
            "patience": 100,             # Trials without improvement
            "min_improvement": 0.001,    # Minimum relative improvement
        },

        # Search behavior
        "expand_around": True,           # Search around initial values (see below)
        "multi_objective": False,        # Multi-objective optimization
        "make_scalar": False,            # Force scalar objective

        # Overfitting control
        "overfitting_penalty": 0.0,      # Penalize train >> validation (see below)

        # Parameter search ranges
        "parameter_config": { ... }
    }

Search Behavior: expand_around
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``expand_around`` setting controls how parameter ranges are interpreted:

- ``expand_around: True`` - Search within a window around initial parameter values.
  Good for fine-tuning when you have reasonable starting points.

- ``expand_around: False`` - Search the full range specified in ``parameter_config``.
  Better for exploration when optimal values are unknown.

For financial strategies, ``False`` often gives better exploration of the parameter space.

Overfitting Penalty
~~~~~~~~~~~~~~~~~~~

The ``overfitting_penalty`` discourages solutions where training performance greatly exceeds validation:

.. code-block:: python

    # Penalty calculation:
    # penalty = overfitting_penalty * max(0, train_score - validation_score)
    #
    # Example: train=1.0, val=0.5, penalty_weight=0.5
    # penalty = 0.5 * (1.0 - 0.5) = 0.25
    # adjusted_score = validation_score - penalty = 0.5 - 0.25 = 0.25

    run_fingerprint["optimisation_settings"]["optuna_settings"]["overfitting_penalty"] = 0.3

Set to ``0.0`` to disable. Range ``[0.0, 1.0]`` recommended.

Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Each parameter in ``parameter_config`` accepts:

.. code-block:: python

    "parameter_name": {
        "low": 1,              # Minimum value
        "high": 200,           # Maximum value
        "log_scale": True,     # Use logarithmic scale
        "scalar": False,       # Same value for all assets (True) or per-asset (False)
    }

Available parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Default Range
     - Log Scale
     - Description
   * - ``memory_length``
     - 1-200
     - Yes
     - EWMA memory in days
   * - ``memory_length_delta``
     - 0.1-100
     - Yes
     - Memory length variation
   * - ``log_k``
     - -10 to 10
     - No
     - Log trading intensity
   * - ``k_per_day``
     - 0.1-1000
     - Yes
     - Trading intensity per day
   * - ``weights_logits``
     - -10 to 10
     - No
     - Initial weight logits
   * - ``log_amplitude``
     - -10 to 10
     - No
     - Signal amplitude (log scale)
   * - ``raw_width``
     - -10 to 10
     - No
     - Channel width
   * - ``raw_exponents``
     - 0-10
     - No
     - Power exponents
   * - ``raw_pre_exp_scaling``
     - -10 to 10
     - No
     - Pre-exponential scaling
   * - ``logit_lamb``
     - -10 to 10
     - No
     - Logit-transformed lambda

For more details on Optuna optimization see :doc:`../tutorials/tuning`.


Return Metrics
--------------

The ``return_val`` parameter determines the objective function for training:

.. code-block:: python

    run_fingerprint["return_val"] = "sharpe"  # Default

Available metrics:

* ``sharpe`` - Sharpe ratio (annualized, default)
* ``daily_sharpe`` - Daily Sharpe ratio
* ``returns`` - Total return over simulation period
* ``returns_over_hodl`` - Return relative to holding the initial portfolio
* ``sortino`` - Sortino ratio (downside risk-adjusted)

Runtime Behavior
----------------

Configure execution details:

.. code-block:: python

    run_fingerprint.update({
        "maximum_change": 3e-4,             # Max weight change per update
        "chunk_period": 1440,               # Strategy update frequency (minutes)
        "weight_interpolation_period": 1440, # Weight change frequency (minutes)
        "weight_interpolation_method": "linear",  # "linear" or "optimal"
        "minimum_weight": None,             # Min weight per asset (default: 0.1/n_assets)
        "max_memory_days": 365,             # Maximum lookback for estimators
        "weight_calculation_method": "auto", # Weight calculation path (see below)
    })

Weight Calculation Method
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``weight_calculation_method`` parameter controls how pool weights are computed:

* ``auto`` (default) - Automatically selects the best available path for the pool
* ``vectorized`` - Uses vectorized convolution-based computation (faster for most pools)
* ``scan`` - Uses sequential scan-based computation (mirrors production execution)

.. code-block:: python

    # Force scan-based computation (matches on-chain execution)
    run_fingerprint["weight_calculation_method"] = "scan"

    # Force vectorized computation (typically faster)
    run_fingerprint["weight_calculation_method"] = "vectorized"

**When to use each:**

- Use ``auto`` for most cases - it selects vectorized when available
- Use ``scan`` when you need to verify results match production/on-chain behavior
- Use ``vectorized`` explicitly if you want to ensure the faster path is used

**Pool support:**

Most QuantAMM pools (momentum, power_channel, mean_reversion_channel) support both paths
and produce numerically equivalent results. Some pools (e.g., min_variance) only support
the vectorized path.

See :doc:`weight_calculation_paths` for detailed information about the two computation paths.

Fee and Arbitrage Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint.update({
        "fees": 0.003,                      # Trading fees (e.g., 30bps)
        "arb_fees": 0.0,                    # Fees paid by arbitrageurs
        "gas_cost": 0.0,                    # Gas cost per arbitrage trade
        "do_arb": True,                     # Enable arbitrage simulation
        "arb_frequency": 1,                 # Arb check frequency (minutes)
        "arb_quality": 1.0,                 # Arbitrage efficiency (0-1)
    })

Advanced Settings
-----------------

Straight-Through Estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For improved gradient flow during training through non-differentiable clipping operations:

.. code-block:: python

    run_fingerprint.update({
        "ste_max_change": False,            # STE for max weight change clipping
        "ste_min_max_weight": False,        # STE for min/max weight bounds
    })

When ``True``, these allow gradients to flow through clipping operations during backpropagation,
which can improve training stability and convergence.

Alternative Lambda Parameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint.update({
        "use_alt_lamb": False,              # Per-estimator memory lengths
        "use_pre_exp_scaling": True,        # Pre-exponential scaling
    })

- ``use_alt_lamb`` - When ``True``, allows different memory lengths for different estimators
- ``use_pre_exp_scaling`` - When ``True``, applies pre-exponential scaling to weight changes

Noise Traders
~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint.update({
        "noise_trader_ratio": 0.0,          # Ratio of noise trader volume (0-1)
    })

Simulates uninformed trading activity. Value of ``0.1`` means 10% of volume comes from noise traders.

Numeraire Token
~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint.update({
        "numeraire": None,                  # Token used as price base (default: last token)
    })

When ``None``, the last token in the ``tokens`` list is used as the numeraire.
Set explicitly if you want prices quoted in a specific token.

Timing Parameters
~~~~~~~~~~~~~~~~~

Understanding the relationship between timing parameters:

.. code-block:: python

    run_fingerprint.update({
        "chunk_period": 1440,               # Strategy evaluation frequency (minutes)
        "weight_interpolation_period": 1440, # Weight update frequency (minutes)
        "bout_offset": 24 * 60 * 7,         # Training window offset (minutes)
    })

- ``chunk_period`` - How often the strategy calculates new target weights.
  1440 = daily, 60 = hourly, 1 = per minute.

- ``weight_interpolation_period`` - How often weights actually change.
  Must be <= chunk_period. When < chunk_period, weights are interpolated
  between chunk evaluations.

- ``bout_offset`` - Controls training data sampling. Training uses periods of
  length (total_duration - bout_offset), sampled from different starting points
  within the bout_offset window.

Weight Interpolation Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint.update({
        "weight_interpolation_method": "linear",  # "linear" or "optimal"
    })

- ``linear`` - Linear interpolation between weight updates
- ``optimal`` - Uses optimal interpolation that minimizes tracking error

Trade Simulation
~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint.update({
        "do_trades": False,                 # Enable explicit trade simulation
    })

When ``True``, allows simulating specific trade sequences through the ``trade_array``
input to ``calculate_reserves_with_dynamic_inputs``.

Implementation Notes
--------------------

- All settings have defaults in ``quantammsim/runners/default_run_fingerprint.py``
- Settings are validated before use
- Some combinations may be invalid for certain strategies

For how the run_fingerprint is used in simulations, see :func:`~quantammsim.runners.jax_runners.train_on_historic_data` and :func:`~quantammsim.runners.jax_runners.do_run_on_historic_data`.
