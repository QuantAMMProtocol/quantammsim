Robustness Features
===================

quantammsim provides several mechanisms to improve out-of-sample performance
and detect overfitting.  This guide covers the full suite of robustness tools.


Price Noise Augmentation
------------------------

Adds multiplicative log-normal noise to training prices:

.. math::

   p'_t = p_t \cdot \exp(\epsilon_t), \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)

This forces the strategy to be robust to small price perturbations rather than
memorising exact historical paths.

.. code-block:: python

    run_fingerprint["price_noise_sigma"] = 0.001  # Typical range: 0.0005-0.005

Higher values increase regularisation but may wash out genuine signals.


Turnover Penalty
----------------

Penalises strategies that make large weight changes, encouraging smoother
trajectories with lower implementation costs:

.. code-block:: python

    run_fingerprint["turnover_penalty"] = 0.01  # Typical range: 0.001-0.1

The penalty is proportional to the sum of absolute weight changes across all
assets and timesteps.


Data Augmentation
-----------------

Time-reversed price series can be included in the training set:

.. code-block:: python

    run_fingerprint["include_flipped_training_data"] = True

This doubles the effective training set size and reduces directional bias
(strategies that only work in one market direction).


Early Stopping
--------------

Monitors a held-out validation set and stops training when performance plateaus:

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "early_stopping": True,
        "early_stopping_patience": 100,       # Epochs without improvement
        "early_stopping_metric": "daily_log_sharpe",
        "val_fraction": 0.2,                  # 20% of data for validation
    })

Early stopping is one of the most effective regularisation techniques.  The
``val_fraction`` controls the IS/validation split within the training window
(distinct from the OOS test window used in walk-forward analysis).


Stochastic Weight Averaging (SWA)
---------------------------------

Averages parameter snapshots from the later stages of training, producing
flatter optima that generalise better:

.. code-block:: python

    run_fingerprint["optimisation_settings"].update({
        "use_swa": True,
        "swa_start_frac": 0.75,   # Start at 75% of training
        "swa_freq": 5,            # Average every 5 epochs
    })


Weight Decay
------------

L2 regularisation on strategy parameters, preventing large parameter values:

.. code-block:: python

    run_fingerprint["optimisation_settings"]["weight_decay"] = 0.01


Ensemble Training
-----------------

Training multiple parameter sets simultaneously and averaging their outputs
provides implicit regularisation through diversity.  See :doc:`hooks` for
configuration details.


Walk-Forward Validation
-----------------------

The gold standard for assessing strategy robustness.  Train on rolling
windows and evaluate on subsequent out-of-sample periods:

.. code-block:: python

    from quantammsim.runners.training_evaluator import TrainingEvaluator

    evaluator = TrainingEvaluator.from_runner(
        runner_name="train_on_historic_data",
        n_cycles=4,
        compute_rademacher=True,
    )
    result = evaluator.evaluate(run_fingerprint)

Key metrics from walk-forward analysis:

* **Walk-Forward Efficiency (WFE)**: ``OOS Sharpe / IS Sharpe``.
  Values > 0.5 suggest robustness; values near 1.0 are ideal.
* **IS-OOS Gap**: ``IS Sharpe - OOS Sharpe``. Large positive gaps indicate
  overfitting.
* **Rademacher Complexity**: Measures the strategy class's capacity to fit
  random noise.  Higher complexity = more overfitting risk.

See :doc:`../api/core/walk_forward` for the full API reference.


Rademacher Complexity
---------------------

Empirical Rademacher complexity (Paleologo, 2024) provides a data-dependent
upper bound on overfitting:

.. math::

   \hat{R} = \mathbb{E}_{\sigma}\left[\sup_s \frac{1}{T} \sum_t \sigma_t r_s(t)\right]

where :math:`\sigma_t` are random Rademacher variables and :math:`r_s(t)` are
returns from strategy checkpoint :math:`s` at time :math:`t`.

The **Rademacher haircut** adjusts observed OOS performance:

.. math::

   \theta_n \geq \hat{\theta}_n - 2\hat{R} - 3\sqrt{\frac{2\log(2/\delta)}{T}}

Enable checkpoint tracking and Rademacher computation:

.. code-block:: python

    run_fingerprint["optimisation_settings"]["track_checkpoints"] = True

    evaluator = TrainingEvaluator.from_runner(
        runner_name="train_on_historic_data",
        compute_rademacher=True,
    )


Recommended Workflow
--------------------

1. **Start simple**: Train with ``daily_log_sharpe``, no augmentation.
2. **Add early stopping**: Usually the single biggest improvement.
3. **Add price noise**: ``sigma = 0.001`` is a safe starting point.
4. **Run walk-forward analysis**: Check WFE and IS-OOS gap.
5. **If overfitting persists**: Add ensemble training, SWA, or weight decay.
6. **Use hyperparameter tuning**: Optimise robustness metrics (WFE, adjusted
   Sharpe) rather than just IS performance.
