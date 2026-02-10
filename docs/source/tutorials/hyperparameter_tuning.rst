Hyperparameter Tuning
=====================

Training a strategy involves choosing hyperparameters — learning rate, batch
size, bout offset, regularisation strength — that sit above the strategy
parameters themselves.  Poor hyperparameters can cause good strategies to
underperform or overfit.  This tutorial covers how to tune training
hyperparameters using walk-forward analysis as the objective.

This tutorial covers:

1. The three-level optimisation hierarchy
2. Basic hyperparameter tuning
3. Multi-period SGD as an alternative trainer
4. Multi-objective tuning (Pareto fronts)
5. Custom search spaces
6. Production workflow


The Three-Level Hierarchy
-------------------------

quantammsim's optimisation stack has three nested levels:

.. code-block:: text

    Level 3: HyperparamTuner
        │  Optuna/TPE — varies (lr, batch_size, bout_offset, ...)
        │  Objective: OOS Sharpe, WFE, or Rademacher-adjusted Sharpe
        ▼
    Level 2: TrainingEvaluator
        │  Walk-forward cycles — trains & evaluates on rolling windows
        │  Computes: WFE, IS-OOS gap, Rademacher complexity
        ▼
    Level 1: Trainer (train_on_historic_data or multi_period_sgd)
        │  Gradient descent — optimises strategy params (λ, k, weights)
        │  Objective: daily_log_sharpe (or other return_val)
        ▼
    Level 0: Forward pass
        Simulate pool → arbitrage → compute financial metric

The key insight: **each level optimises something the level below cannot see**.

* Level 1 optimises strategy parameters for a given training window.
* Level 2 evaluates whether Level 1's output generalises across windows.
* Level 3 finds the hyperparameters that make Level 2's evaluation best.

Optimising for OOS metrics at the outer level avoids the fundamental trap of
tuning hyperparameters on in-sample performance.


Basic Hyperparameter Tuning
---------------------------

The :class:`~quantammsim.runners.hyperparam_tuner.HyperparamTuner` wraps a
:class:`~quantammsim.runners.training_evaluator.TrainingEvaluator` inside an
Optuna study:

.. code-block:: python

    from quantammsim.runners.hyperparam_tuner import HyperparamTuner

    run_fingerprint = {
        "tokens": ["BTC", "ETH"],
        "rule": "mean_reversion_channel",
        "startDateString": "2022-06-01 00:00:00",
        "endDateString": "2024-06-01 00:00:00",
        "initial_pool_value": 1_000_000.0,
        "fees": 0.003,
        "do_arb": True,
        "return_val": "daily_log_sharpe",
        "chunk_period": 1440,
        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "adam",
            "n_parameter_sets": 4,
            "use_gradient_clipping": True,
            "clip_norm": 10.0,
        },
    }

    tuner = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=30,               # Optuna trials
        n_wfa_cycles=3,            # WFA cycles per trial
        objective="mean_oos_sharpe",
    )

    result = tuner.tune(run_fingerprint)

    print(f"Best OOS Sharpe: {result.best_value:.3f}")
    print(f"Best params: {result.best_params}")

    # Apply best hyperparameters for final training
    run_fingerprint["optimisation_settings"].update(result.best_params)

The tuner varies training hyperparameters (learning rate, batch size, bout
offset, LR schedule, early stopping, weight decay) while keeping strategy
structure and data fixed.

Default Search Space
~~~~~~~~~~~~~~~~~~~~

The default search space covers:

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - Parameter
     - Range
     - Scale
     - Notes
   * - ``base_lr``
     - [1e-5, 0.1]
     - Log
     - Adam/AdamW range; SGD uses [1e-3, 1.0]
   * - ``batch_size``
     - [8, 64]
     - Log
     - Powers of 2 preferred
   * - ``n_iterations``
     - [50, 5000]
     - Log
     - Training epochs
   * - ``bout_offset_days``
     - [7, ~90% of cycle]
     - Log
     - Converted to minutes internally
   * - ``clip_norm``
     - [0.5, 50]
     - Log
     - Gradient clipping threshold
   * - ``lr_schedule_type``
     - constant / cosine / warmup_cosine / exponential
     - Categorical
     - Conditional parameters follow
   * - ``use_early_stopping``
     - True / False
     - Categorical
     - Patience and val_fraction are conditional
   * - ``use_weight_decay``
     - True / False
     - Categorical
     - Decay value is conditional
   * - ``noise_scale``
     - [0.01, 0.5]
     - Log
     - Initialisation diversity


Objective Functions
~~~~~~~~~~~~~~~~~~~~

Available objectives:

* **``"mean_oos_sharpe"``** (default) — Average OOS Sharpe across cycles.
  Best for maximising expected performance.
* **``"worst_oos_sharpe"``** — Worst-case OOS Sharpe.  Best for robustness
  across market regimes.
* **``"mean_wfe"``** — Average Walk-Forward Efficiency.  Optimises for
  consistency rather than magnitude.
* **``"adjusted_mean_oos_sharpe"``** — Rademacher-adjusted Sharpe.
  Requires ``compute_rademacher=True`` on the inner evaluator.
* **``"multi"``** — Multi-objective (see below).

Also available: ``"mean_oos_calmar"``, ``"mean_oos_sterling"``,
``"worst_oos_calmar"``, ``"mean_oos_daily_log_sharpe"``, and others.


Trial Pruning
~~~~~~~~~~~~~

Unpromising trials are pruned early to save compute.  After each walk-forward
cycle completes, the tuner reports the intermediate OOS metric to Optuna.
If the trial's trajectory is worse than the bottom 25th percentile of
completed trials, it is terminated:

.. code-block:: python

    tuner = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=50,
        n_wfa_cycles=4,
        objective="mean_oos_sharpe",
        enable_pruning=True,  # Default
        pruner="percentile",  # Default: prune bottom 25%
    )

Available pruners:

* ``"percentile"`` (default) — Prune bottom 25%.  Good for WFA where cycles
  are independent market regimes.
* ``"median"`` — Prune below median.  More aggressive.
* ``"hyperband"`` / ``"successive_halving"`` — Multi-fidelity pruners.
  Use cautiously with WFA since cycles are not true fidelity levels.
* ``None`` — Disable pruning.


Multi-Period SGD
----------------

:mod:`~quantammsim.runners.multi_period_sgd` is an alternative Level-1
trainer that divides training data into multiple periods and optimises across
all of them simultaneously.  It's particularly effective for strategies that
need to work across different market regimes:

.. code-block:: python

    tuner = HyperparamTuner(
        runner_name="multi_period_sgd",
        n_trials=30,
        n_wfa_cycles=3,
        objective="mean_oos_sharpe",
    )

    result = tuner.tune(run_fingerprint)

The multi-period search space automatically includes:

* ``n_periods``: Number of sub-periods (2-8)
* ``max_epochs``: Training epochs (50-300)
* ``aggregation``: How to combine period losses (``mean``, ``worst``, ``softmin``)
* ``softmin_temperature``: Temperature for softmin (conditional on aggregation)

The ``worst`` aggregation trains to maximise the minimum performance across
all periods — a minimax objective that produces conservative but robust
strategies.


Multi-Objective Tuning
----------------------

Sometimes you want to optimise multiple objectives simultaneously rather
than collapsing them into a single scalar.  Multi-objective tuning returns
a Pareto front:

.. code-block:: python

    tuner = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=50,
        n_wfa_cycles=3,
        objective="multi",
        multi_objectives=["mean_oos_sharpe", "mean_wfe"],
    )

    result = tuner.tune(run_fingerprint)

    # Inspect the Pareto front
    for trial in result.pareto_front:
        print(
            f"OOS Sharpe={trial['values'][0]:.3f}, "
            f"WFE={trial['values'][1]:.3f}, "
            f"params={trial['params']}"
        )

The Pareto front contains all non-dominated solutions — configurations where
no other trial is strictly better on *all* objectives.  You then choose
from the front based on your priorities:

* If deployment risk tolerance is low, pick the trial with highest WFE
  (most consistent generalisation).
* If absolute performance matters more, pick the trial with highest OOS
  Sharpe (even if WFE is lower).


Custom Search Spaces
--------------------

Create a custom search space using
:class:`~quantammsim.runners.hyperparam_tuner.HyperparamSpace`:

.. code-block:: python

    from quantammsim.runners.hyperparam_tuner import HyperparamSpace

    # Minimal space for quick exploration
    space = HyperparamSpace.create(minimal=True)
    # Only tunes: base_lr ∈ [0.01, 0.5], n_iterations ∈ [50, 200]

    # Full space with custom cycle duration
    space = HyperparamSpace.create(
        runner="train_on_historic_data",
        cycle_days=90,                  # Shorter cycles → smaller bout_offset range
        optimizer="adam",
        include_lr_schedule=True,       # Include LR schedule choices
        include_early_stopping=True,    # Include early stopping
        include_weight_decay=True,      # Include weight decay
        objective_metric="mean_oos_sharpe",
    )

    # Pass to tuner
    tuner = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=30,
        n_wfa_cycles=3,
        hyperparam_space=space,
    )

You can also define the space manually for full control:

.. code-block:: python

    space = HyperparamSpace(params={
        "base_lr": {"low": 0.001, "high": 0.1, "log": True},
        "batch_size": {"low": 8, "high": 32, "log": True, "type": "int"},
        "n_iterations": {"low": 100, "high": 1000, "log": True, "type": "int"},
        "bout_offset_days": {"low": 7, "high": 60, "log": True, "type": "int"},
    })

Conditional parameters are supported:

.. code-block:: python

    space = HyperparamSpace(params={
        "base_lr": {"low": 0.001, "high": 0.1, "log": True},
        "use_weight_decay": {"choices": [True, False]},
        "weight_decay": {
            "low": 0.0001, "high": 0.1, "log": True,
            "conditional_on": "use_weight_decay",
            "conditional_value": True,
        },
    })


Persistent Studies
~~~~~~~~~~~~~~~~~~

For long-running tuning, persist the Optuna study to a database so it
survives interruptions:

.. code-block:: python

    tuner = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=100,
        n_wfa_cycles=4,
        study_name="btc_eth_momentum_tune",
        storage="sqlite:///tuning_results.db",
        total_timeout=3600 * 4,  # Stop after 4 hours
    )

    result = tuner.tune(run_fingerprint)

Rerunning with the same ``study_name`` and ``storage`` resumes from where
it left off.


Production Workflow
-------------------

A recommended end-to-end workflow:

.. code-block:: python

    from quantammsim.runners.hyperparam_tuner import HyperparamTuner
    from quantammsim.runners.training_evaluator import TrainingEvaluator
    from quantammsim.runners.jax_runners import train_on_historic_data

    # ── Step 1: Quick exploration with minimal space ──
    tuner_quick = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=15,
        n_wfa_cycles=3,
        objective="mean_oos_sharpe",
        hyperparam_space=HyperparamSpace.create(minimal=True),
    )
    quick_result = tuner_quick.tune(run_fingerprint)
    print(f"Quick pass best: {quick_result.best_value:.3f}")

    # ── Step 2: Full tuning around the promising region ──
    tuner_full = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=50,
        n_wfa_cycles=4,
        objective="mean_oos_sharpe",
        enable_pruning=True,
    )
    full_result = tuner_full.tune(run_fingerprint)
    print(f"Full tuning best: {full_result.best_value:.3f}")

    # ── Step 3: Validate best hyperparameters with more cycles ──
    run_fingerprint["optimisation_settings"].update(full_result.best_params)

    evaluator = TrainingEvaluator.from_runner(
        "train_on_historic_data",
        n_cycles=6,           # More cycles for final validation
        compute_rademacher=True,
    )
    validation = evaluator.evaluate(run_fingerprint)
    evaluator.print_report(validation)

    # ── Step 4: Final training on full data ──
    if validation.is_effective:
        print("Strategy validated — running final training")
        train_on_historic_data(run_fingerprint, verbose=True)
    else:
        print("Strategy did not pass validation:")
        for reason in validation.effectiveness_reasons:
            print(f"  - {reason}")


Analysing Results
-----------------

The :class:`~quantammsim.runners.hyperparam_tuner.TuningResult` provides
full trial-level data for post-hoc analysis:

.. code-block:: python

    result = tuner.tune(run_fingerprint)

    # Summary statistics
    print(f"Trials: {result.n_completed} completed, "
          f"{result.n_pruned} pruned, {result.n_failed} failed")
    print(f"Total time: {result.total_time_seconds:.0f}s")

    # Per-trial data
    for trial in result.all_trials[:5]:
        print(f"  Trial {trial['number']}: "
              f"value={trial['value']:.3f}, params={trial['params']}")

    # Best trial's full WFA evaluation
    if result.best_evaluation is not None:
        print(f"\nBest trial WFE: {result.best_evaluation.mean_wfe:.3f}")
        for c in result.best_evaluation.cycles:
            print(f"  Cycle {c.cycle_number}: "
                  f"IS={c.is_sharpe:.2f}, OOS={c.oos_sharpe:.2f}")


See Also
--------

- :doc:`walk_forward_analysis` — Walk-forward validation tutorial
- :doc:`ensemble_training` — Ensemble training tutorial
- :doc:`../user_guide/robustness_features` — Regularisation techniques
- :doc:`../user_guide/metrics_reference` — Available metrics for objectives
- :mod:`quantammsim.runners.hyperparam_tuner` — HyperparamTuner API reference
- :mod:`quantammsim.runners.training_evaluator` — TrainingEvaluator API reference
- :mod:`quantammsim.runners.multi_period_sgd` — Multi-period SGD API reference
