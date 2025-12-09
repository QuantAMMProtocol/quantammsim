Training Pools
==============

Why Train Pools?
----------------

QuantAMM pools implement trading strategies with configurable parameters that determine how they respond to market conditions.
While you can set these parameters manually, finding good values often requires systematic optimization.

Training helps you:

* Maximize desired metrics (Sharpe ratio, returns, etc.)
* Adapt strategies to specific market conditions
* Balance between responsiveness and stability
* Validate strategy effectiveness

Basic Training Setup
--------------------

To train a pool, you need to define:

1. A run fingerprint (configuration) including initial parameters
2. Training data specifications

Choosing an Optimization Approach
---------------------------------

We provide two main approaches to parameter optimization:

**Gradient Descent** leverages directly JAX's ability to compute gradients. This is ideal when:

* You have a good starting point and want to fine-tune
* The relationship between parameters and performance is smooth
* You have GPU acceleration
* You are training in the zero-fees regime (as then there are extremely fast GPU-accelerated algorithms)

**Gradient-Free Search** works better when:

* You're exploring a wide parameter space
* The performance landscape might be non-smooth (the pool strategy does not have to be differentiable with respect to the strategy parameters)
* You are training in the presence of fees (as the GPU-accelerated algorithms are not as fast, as fees make modelling AMMs intrinsically sequential)

Parameter Spaces
~~~~~~~~~~~~~~~~

QuantAMM strategies often use transformed parameters for better optimization. For example:

* Memory length (:math:`\text{days}`) → logit(λ)

  - Ensures λ stays between 0 and 1 by construction
  - Which thus allows unconstrained optimization
  - :math:`\lambda = \text{sigmoid}(\text{logit_lambda})`

* Strategy aggressiveness (k) → log2(k)

  - Handles wide range of scales
  - Maintains positivity
  - :math:`k = 2^{\text{\log2(k)}}`

See :ref:`constrained-vs-unconstrained` for more examples.


Getting Started
---------------

Let's optimize a QuantAMMmomentum strategy for a BTC/ETH pool:

.. code-block:: python

    from quantammsim.runners.jax_runners import train_on_historic_data
    
    # Basic configuration
    run_fingerprint = {
        "tokens": ["BTC", "ETH"],
        "rule": "momentum",
        "startDateString": "2024-01-01 00:00:00",
        "endDateString": "2024-03-01 00:00:00",
        "endTestDateString": "2024-04-01 00:00:00",
        "chunk_period": 60,
        "return_val": "sharpe",
        "initial_pool_value": 1000000.0
    }

Data Handling
-------------

The simulator supports flexible data configuration:

1. Training/Test Split:

   * Specify date ranges for training and testing
   * Automatic data windowing

2. Data Processing:

   * Minute-level granularity
   * Automatic resampling
   * Missing data handling
   * Price normalization

3. Training Windows:

   * Fixed or random windows
   * Overlapping periods
   * Custom bout lengths

Gradient Descent Training
-------------------------

For gradient-based optimization:

1. Set optimization parameters:

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "method": "gradient_descent",
        "optimiser": "adam",       # Optimizer type 
        "base_lr": 0.01,           # Learning rate
        "batch_size": 16,          # Training batch size
        "n_parameter_sets": 4,     # Number of parameter sets to train in parallel
        "n_iterations": 10000,     # Total iterations
        "decay_lr_plateau": 200,   # Iterations of no improvement before decay
        "decay_lr_ratio": 0.8,     # Learning rate decay on plateau
    })

2. Set off and monitor training:

.. code-block:: python

    result = train_on_historic_data(
        run_fingerprint,
        iterations_per_print=100,  # Progress update frequency
        verbose=True               # Detailed logging
    )

Gradient-Free Optimization
--------------------------


1. Configure Optuna settings:

Optuna provides sophisticated optimization without the need for gradients:

.. code-block:: python

    run_fingerprint["optimisation_settings"]["optuna_settings"].update({
        "method": "optuna",
        "n_trials": 100,              # Total optimization trials
        "n_jobs": 4,                  # Parallel workers
        "timeout": 7200,              # Max runtime in seconds
        "n_startup_trials": 10,       # Random trials before optimization
        "early_stopping": {
            "enabled": True,
            "patience": 100,          # Trials without improvement
            "min_improvement": 0.001  # Minimum relative improvement
        },
        "parameter_config": {
            "memory_length": {
                "low": 1,
                "high": 200,
                "log_scale": True,    # Search on log scale
                "scalar": False       # Different values per asset
            }
        }
    })

The parameter_config supports extensive customization per parameter, including:

* Search range bounds
* Linear vs logarithmic scaling
* Per-asset vs global parameters (see below)

If no parameter_config is provided for a parameter, the simulator will use the default parameter_config.

If a strategy is not differentiable with respect to a parameter, you will have to use this gradient-free optimization approach.

2. Set off and monitor training:

.. code-block:: python

    result = train_on_historic_data(
        run_fingerprint,
        iterations_per_print=100,   # Progress update frequency
        verbose=True               # Detailed logging
    )

Parameter & Run Configuration
-----------------------------

Pools will automatically initialise as JAX arrays all the parameters that they are set to train.
If you wish to set particular initial values for them (for gradient descent training), or the range/sampling method for them (for gradient-free training), you can do so via the ``run_fingerprint`` dictionary.

The simulator provides extensive parameter configuration options through the run_fingerprint dictionary. Default values are provided but can be overridden:

.. code-block:: python

        "bout_offset": 24 * 60 * 7,         # Training window offset (in minutes)
        "maximum_change": 3e-4,             # Max weight change per update
        "chunk_period": 1440,               # Strategy update frequency in minutes (1 day)
        "weight_interpolation_period": 1440 # Weight update frequency (1 day)
    })

Advanced parameters include:

* ``use_alt_lamb``: Alternative lambda parameterization so different parts of estimators can have different memory lengths (not supported in QuantAMM V1)
* ``use_pre_exp_scaling``: Pre-exponential scaling
* ``weight_interpolation_method``: "linear" or "optimal" (only linear is supported in QuantAMM V1)
* ``arb_frequency``: Arbitrage check frequency (in minutes)
* ``arb_quality``: Arbitrage execution efficiency (0-1) (only used for CoW AMM pools)


Bout offset is the number of minutes to offset the training window by. Given how much it can affect the results, it is worth understanding how it works.

Understanding Bout Offset
-------------------------

Bout offset is a crucial parameter that controls how training windows are constructed relative to price data.
It specifies an offset in minutes from the duration of the specified training period, in effect shortening the training period by the specified amount.

So if a training run is 4 months long, and the bout offset is 1 month, each window of price data actually used during training will be 3 months long.
Those windows can start as early as the first moment of the training period and end as the last moment of the training period.

1. Configuration

   .. code-block:: python

       run_fingerprint.update({
           "bout_offset": 24 * 60 * 7,  # One week in minutes
       })

3. Common Configurations

   * Daily offset: 24 * 60 minutes
   * Weekly offset: 24 * 60 * 7 minutes (default)
   * Custom periods: Any duration

4. Training Considerations

   * Larger offsets reduce available training data
   * They make the strategy care more about achieving the return metric over the effective (bout-offset reduced) training period

Best Practices:

* If you want your pool to be more sensitive to partial market conditions, use a longer offset, reducing the length of the windows of data used for training.
* If you want your pool to optimise more strongly for performance over the exact start date to end date of the training period, use a shorter offset, increasing the length of the windows of data used for training.

Advanced Features
-----------------

1. Custom Return Metrics:

.. code-block:: python

    run_fingerprint["return_val"] = "sharpe"  # or "returns", "sortino" among others

2. Multi-period Training (Gradient-free only):

.. code-block:: python

    run_fingerprint.update({
        "optimisation_settings": {
            "method": "optuna",
            "optuna_settings": {
                "multi_objective": True,
            }
        }
    })

This sets the optimisation to be multi-objective, and will optimise the chosen return metric for a range of different periods within the training data, a mixture of sequenital, periodically placed windows and randomly placed windows (to avoid periodic biases).
This is useful for finding a good set of parameters that perform well across a range of market conditions.
The mean and standard deviation of the return metric across the different periods is returned in the result, along with the worst value.

.. note::

    This is only supported in the gradient-free optimisation approac, but gradient-based optimisation naturally has this property as each window of data used for training is a) reduced in length by the bout offset and b) randomly placed.

3. Constrain a parameter to be "universal" (i.e. the same value for all assets) (Gradient-free only):

.. code-block:: python

    run_fingerprint["optimisation_settings"]["optuna_settings"]["parameter_config"]["k_per_day"]["scalar"] = True

This is set on a per-parameter basis, and will force the parameter to be the same for all assets.
Constraining parameters in this way can help with generalisation.

4. Set initial parameter values (Gradient-descent only):

.. code-block:: python

    run_fingerprint.update({
        "initial_memory_length": 10.0,    # Starting memory length in days
        "initial_k_per_day": 20,          # Starting strategy aggressiveness
    })

Performance Considerations
--------------------------

* Use GPU acceleration when available
* Batch size, n_parameter_sets (for gradient descent) and n_jobs (for gradient-free) affect memory usage
* Monitor for overfitting using test period and other post-run analysis

See Also
--------
* :doc:`../tutorials/quantamm_pools` for strategy details
* `TFMM litepaper <https://quantamm.fi/research>`_ for theoretical background