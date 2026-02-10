Walk-Forward Analysis
=====================

Walk-forward analysis (WFA) is the gold standard for assessing whether a
trained strategy generalises beyond its training data.  Rather than training
once and hoping for the best, WFA trains on rolling windows and evaluates on
subsequent out-of-sample (OOS) periods — the closest thing to a live test you
can run on historical data.

This tutorial covers:

1. Why walk-forward analysis matters
2. Running a basic walk-forward evaluation
3. Interpreting WFE and IS-OOS gap
4. Adding Rademacher complexity estimation
5. Comparing training approaches
6. Walk-forward with early stopping and regularisation


Why Walk-Forward Analysis?
--------------------------

Standard backtests train on *all* the data and report the result.  This tells
you how well the strategy fits history, not how it would have performed in
real time.  WFA addresses this by mimicking the deployment cycle:

1. Train on data up to time *t*
2. Deploy (evaluate) on the next unseen period *t* to *t + Δ*
3. Roll forward and repeat

The key insight from Pardo (2008): if a strategy's OOS performance is
consistently close to its in-sample performance across multiple cycles, you
have evidence of genuine generalisation rather than overfitting.


Basic Walk-Forward Evaluation
-----------------------------

The :class:`~quantammsim.runners.training_evaluator.TrainingEvaluator`
orchestrates the full workflow: generating cycles, training, evaluating, and
aggregating results.

.. code-block:: python

    from quantammsim.runners.training_evaluator import TrainingEvaluator

    # Configure the base run fingerprint
    run_fingerprint = {
        "tokens": ["BTC", "ETH"],
        "rule": "mean_reversion_channel",
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2024-07-01 00:00:00",
        "initial_pool_value": 1_000_000.0,
        "fees": 0.003,
        "do_arb": True,
        "return_val": "daily_log_sharpe",
        "chunk_period": 1440,
        "bout_offset": 1440 * 14,  # 2-week offset
        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "adam",
            "base_lr": 0.05,
            "n_iterations": 300,
            "batch_size": 16,
            "n_parameter_sets": 4,
            "use_gradient_clipping": True,
            "clip_norm": 10.0,
        },
    }

    # Create evaluator wrapping the standard trainer
    evaluator = TrainingEvaluator.from_runner(
        "train_on_historic_data",
        n_cycles=4,          # 4 train/test cycles
        verbose=True,
    )

    # Run the full walk-forward evaluation
    result = evaluator.evaluate(run_fingerprint)

    # Print summary report
    evaluator.print_report(result)

The evaluator divides the date range into ``n_cycles + 1`` equal segments.
Each cycle trains on one segment and tests on the next:

.. code-block:: text

    |--- Seg 0 ---|--- Seg 1 ---|--- Seg 2 ---|--- Seg 3 ---|--- Seg 4 ---|
    |  Train C0   |  Test C0    |             |             |             |
    |             |  Train C1   |  Test C1    |             |             |
    |             |             |  Train C2   |  Test C2    |             |
    |             |             |             |  Train C3   |  Test C3    |


Rolling vs Expanding Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, training windows *roll forward* — each cycle trains only on its
own segment.  Set ``keep_fixed_start=True`` for *expanding windows*, where
training always starts from the beginning:

.. code-block:: python

    # Expanding window: later cycles see more data
    evaluator = TrainingEvaluator.from_runner(
        "train_on_historic_data",
        n_cycles=4,
        keep_fixed_start=True,
    )

Rolling windows test how the strategy adapts to regime changes.  Expanding
windows test whether more data always helps (and can reveal when old data
hurts).


Interpreting Results
--------------------

The :class:`~quantammsim.runners.training_evaluator.EvaluationResult`
contains per-cycle evaluations and aggregate statistics.

Walk-Forward Efficiency (WFE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WFE is the ratio of OOS to IS performance:

.. math::

   \text{WFE} = \frac{\text{OOS Sharpe}}{\text{IS Sharpe}}

Rules of thumb (Pardo, 2008):

* **WFE > 0.5** — Suggests robustness; the strategy retains at least half
  its in-sample edge out of sample.
* **WFE ≈ 1.0** — Ideal. OOS performance matches IS (no overfitting).
* **WFE > 1.0** — OOS outperformed IS. Possible with mean-reverting strategies
  in favorable market conditions.
* **WFE < 0.3** — Red flag. Strategy likely overfits to training data.

.. code-block:: python

    print(f"Mean WFE: {result.mean_wfe:.3f}")
    print(f"Mean OOS Sharpe: {result.mean_oos_sharpe:.3f}")
    print(f"Worst OOS Sharpe: {result.worst_oos_sharpe:.3f}")

    # Per-cycle breakdown
    for c in result.cycles:
        print(
            f"Cycle {c.cycle_number}: "
            f"IS={c.is_sharpe:.2f}, OOS={c.oos_sharpe:.2f}, "
            f"WFE={c.walk_forward_efficiency:.2f}"
        )


IS-OOS Gap
~~~~~~~~~~

The gap ``IS Sharpe - OOS Sharpe`` directly measures overfitting.  A large
positive gap means the strategy performed much better in-sample than out.

.. code-block:: python

    print(f"Mean IS-OOS gap: {result.mean_is_oos_gap:.3f}")

    # Investigate cycle-level gaps
    for c in result.cycles:
        flag = " ⚠ OVERFIT" if c.is_oos_gap > 0.5 else ""
        print(f"Cycle {c.cycle_number}: gap = {c.is_oos_gap:.3f}{flag}")


Rademacher Complexity
---------------------

Rademacher complexity (Paleologo, 2024) provides a data-dependent upper
bound on overfitting.  It measures how well the *set of strategies explored
during training* can fit random noise — a strategy class with high Rademacher
complexity can fit anything, which means observed performance may be spurious.

Enable checkpoint tracking and Rademacher computation:

.. code-block:: python

    evaluator = TrainingEvaluator.from_runner(
        "train_on_historic_data",
        n_cycles=4,
        compute_rademacher=True,  # Track parameter checkpoints
    )

    result = evaluator.evaluate(run_fingerprint)

    print(f"Rademacher complexity: {result.aggregate_rademacher:.4f}")
    print(f"Adjusted OOS Sharpe: {result.adjusted_mean_oos_sharpe:.3f}")

The **Rademacher haircut** adjusts observed OOS performance downward:

.. math::

   \theta_n \geq \hat{\theta}_n - 2\hat{R}
   - 3\sqrt{\frac{2\log(2/\delta)}{T}}

where :math:`\hat{R}` is the empirical Rademacher complexity and :math:`T` is
the number of test periods.  If the adjusted Sharpe is still positive, you have
stronger evidence that performance is genuine.

.. code-block:: python

    for c in result.cycles:
        if c.rademacher_complexity is not None:
            print(
                f"Cycle {c.cycle_number}: "
                f"OOS Sharpe={c.oos_sharpe:.2f}, "
                f"R̂={c.rademacher_complexity:.4f}, "
                f"Adjusted={c.adjusted_oos_sharpe:.2f}"
            )


Comparing Training Approaches
------------------------------

Use :func:`~quantammsim.runners.training_evaluator.compare_trainers` to
benchmark different training configurations side-by-side:

.. code-block:: python

    from quantammsim.runners.training_evaluator import (
        TrainingEvaluator,
        compare_trainers,
    )

    comparison = compare_trainers(
        run_fingerprint,
        trainers={
            "sgd_conservative": TrainingEvaluator.from_runner(
                "train_on_historic_data",
                n_cycles=4,
                max_iterations=200,
            ),
            "sgd_aggressive": TrainingEvaluator.from_runner(
                "train_on_historic_data",
                n_cycles=4,
                max_iterations=2000,
            ),
            "random_baseline": TrainingEvaluator.random_baseline(n_cycles=4),
        },
    )

    # comparison is a dict of {name: EvaluationResult}
    for name, res in comparison.items():
        print(f"{name}: WFE={res.mean_wfe:.2f}, OOS Sharpe={res.mean_oos_sharpe:.3f}")

The random baseline is essential — it trains with random parameters and tells
you how much of the OOS performance comes from the strategy class structure
versus the optimisation itself.  If your tuned strategy barely beats random,
the optimisation is adding noise, not signal.


Walk-Forward with Regularisation
---------------------------------

WFA benefits enormously from regularisation features.  Here's a complete
example combining early stopping, price noise, and turnover penalty:

.. code-block:: python

    run_fingerprint = {
        "tokens": ["BTC", "ETH", "SOL"],
        "rule": "mean_reversion_channel",
        "startDateString": "2022-06-01 00:00:00",
        "endDateString": "2024-06-01 00:00:00",
        "initial_pool_value": 1_000_000.0,
        "fees": 0.003,
        "do_arb": True,
        "return_val": "daily_log_sharpe",
        "chunk_period": 1440,
        "bout_offset": 1440 * 14,

        # Regularisation
        "price_noise_sigma": 0.001,          # Multiplicative log-normal noise
        "turnover_penalty": 0.01,            # Penalise excessive weight changes
        "include_flipped_training_data": True,  # Time-reversed augmentation

        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "adam",
            "base_lr": 0.05,
            "n_iterations": 1000,
            "batch_size": 16,
            "n_parameter_sets": 4,
            "use_gradient_clipping": True,
            "clip_norm": 10.0,

            # Early stopping on validation set
            "early_stopping": True,
            "early_stopping_patience": 100,
            "early_stopping_metric": "daily_log_sharpe",
            "val_fraction": 0.2,

            # SWA for flatter optima
            "use_swa": True,
            "swa_start_frac": 0.75,
            "swa_freq": 5,
        },
    }

    evaluator = TrainingEvaluator.from_runner(
        "train_on_historic_data",
        n_cycles=5,
        compute_rademacher=True,
    )

    result = evaluator.evaluate(run_fingerprint)
    evaluator.print_report(result)


Choosing the WFE Metric
~~~~~~~~~~~~~~~~~~~~~~~~

By default, WFE is computed from annualised Sharpe ratios (Pardo's original
definition).  You can change this to any metric from
:doc:`../user_guide/metrics_reference`:

.. code-block:: python

    # Use Calmar ratio for drawdown-sensitive WFE
    evaluator = TrainingEvaluator.from_runner(
        "train_on_historic_data",
        n_cycles=4,
        wfe_metric="calmar",
    )


Warm Starting
~~~~~~~~~~~~~

Parameters from each cycle can seed the next cycle's training.  This is
enabled by default in the evaluator — the ``warm_start_params`` from cycle
*n* become the initial parameters for cycle *n + 1*.  This mirrors deployment
where you'd warm-start retraining from the last known good parameters.


Generating Cycles Manually
--------------------------

For advanced use, generate cycle specifications directly:

.. code-block:: python

    from quantammsim.runners.robust_walk_forward import (
        generate_walk_forward_cycles,
    )

    cycles = generate_walk_forward_cycles(
        start_date="2022-01-01 00:00:00",
        end_date="2024-01-01 00:00:00",
        n_cycles=6,
        keep_fixed_start=False,  # Rolling windows
    )

    for c in cycles:
        print(
            f"Cycle {c.cycle_number}: "
            f"Train {c.train_start_date} → {c.train_end_date}, "
            f"Test {c.test_start_date} → {c.test_end_date}"
        )


Decision Framework
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Diagnostic
     - Healthy
     - Action if Unhealthy
   * - Mean WFE
     - > 0.5
     - Add regularisation, reduce model complexity
   * - IS-OOS gap
     - < 0.3
     - Enable early stopping, add price noise
   * - Rademacher complexity
     - < 0.05
     - Reduce training iterations, use SWA
   * - Worst OOS Sharpe
     - > 0.0
     - Check for regime sensitivity, use expanding windows
   * - ``is_effective``
     - ``True``
     - Review ``effectiveness_reasons`` for specifics


See Also
--------

- :doc:`../user_guide/robustness_features` — Full robustness feature guide
- :doc:`../user_guide/metrics_reference` — Available training and evaluation metrics
- :doc:`hyperparameter_tuning` — Optimise training hyperparameters using WFA as the objective
- :doc:`ensemble_training` — Ensemble averaging for implicit regularisation
- :mod:`quantammsim.runners.training_evaluator` — API reference
- :mod:`quantammsim.runners.robust_walk_forward` — Rademacher and WFE utilities
