Ensemble Training
=================

Ensemble training trains multiple parameter sets ("members") simultaneously
and averages their weight outputs.  This provides implicit regularisation
through diversity: individual members may overfit in different ways, but their
average tends toward the robust core signal.

This tutorial covers:

1. Why ensemble averaging works
2. Basic ensemble setup
3. Initialisation methods and their trade-offs
4. Multi-hook chaining (ensemble + bounded weights)
5. Ensemble with walk-forward validation
6. Best practices


Why Ensemble Averaging?
-----------------------

Single-strategy training optimises one set of parameters.  If the optimisation
landscape has multiple local optima (common for financial strategies), the
training outcome depends strongly on initialisation.  Worse, a single solution
may overfit to idiosyncratic features of the training data.

Ensemble averaging mitigates both problems:

* **Exploration**: Members start from different positions in parameter space,
  increasing the chance that at least one finds a good basin.
* **Regularisation**: Averaging rule outputs smooths out member-specific
  overfitting.  The ensemble's effective hypothesis class is more constrained
  than any individual member's.
* **Gradient flow**: Because the averaging uses ``jnp.mean`` (not
  ``stop_gradient``), gradients flow back to all members proportionally:

  .. math::

     \frac{\partial \mathcal{L}}{\partial \theta_i}
     = \frac{1}{N} \cdot \frac{\partial \mathcal{L}}{\partial \bar{w}}
     \cdot \frac{\partial w_i}{\partial \theta_i}

  Each member receives gradients weighted by how its output affected the mean.


Basic Ensemble Setup
--------------------

Ensembles are enabled via the ``ensemble`` hook and the
``n_ensemble_members`` fingerprint key:

.. code-block:: python

    from quantammsim.runners.jax_runners import train_on_historic_data

    run_fingerprint = {
        "tokens": ["BTC", "ETH"],
        "rule": "ensemble__momentum",  # Hook prefix
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2024-01-01 00:00:00",
        "initial_pool_value": 1_000_000.0,
        "fees": 0.003,
        "do_arb": True,
        "return_val": "daily_log_sharpe",
        "chunk_period": 1440,

        # Ensemble configuration
        "n_ensemble_members": 4,
        "ensemble_init_method": "lhs",    # Latin Hypercube Sampling
        "ensemble_init_scale": 0.5,       # Spread around initial values
        "ensemble_init_seed": 42,         # Reproducibility

        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "adam",
            "base_lr": 0.05,
            "n_iterations": 500,
            "batch_size": 16,
            "n_parameter_sets": 2,
        },
    }

    train_on_historic_data(run_fingerprint, verbose=True)

With ``n_parameter_sets=2`` and ``n_ensemble_members=4``, the parameter
tensors have shape ``(2, 4, ...)``:

* **Outer dimension** (2): independent training runs (vmapped in the runner)
* **Inner dimension** (4): ensemble members that share gradients through
  averaging

The ensemble hook averages the *rule outputs* (weight changes), not the raw
parameters.  Each member maintains its own EWMA estimator state and produces
its own weight trajectory; the final weights are the arithmetic mean across
members.


Initialisation Methods
----------------------

How ensemble members are spread across parameter space at initialisation
significantly affects diversity and convergence.  Set the method via
``run_fingerprint["ensemble_init_method"]``.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Description
     - Best for
   * - ``"lhs"``
     - Latin Hypercube Sampling.  Each parameter dimension is divided into
       *N* equal strata, and exactly one sample is placed in each stratum.
     - General use. Good space coverage with low sample counts.  **Recommended
       default.**
   * - ``"centered_lhs"``
     - LHS with samples at stratum centres rather than random positions
       within each stratum.
     - When you want deterministic, evenly-spaced initialisation.
   * - ``"sobol"``
     - Sobol quasi-random sequence (low-discrepancy).  Provides more
       uniform coverage than pseudo-random sampling, especially at higher
       dimensions.
     - Larger ensembles (8+) or high-dimensional parameter spaces.
   * - ``"grid"``
     - Regular grid over the parameter space.  Deterministic and maximally
       uniform, but scales poorly with dimension.
     - Small ensembles (2-4 members) with few parameters.
   * - ``"gaussian"``
     - Independent Gaussian noise around initial values (the original,
       backwards-compatible approach).
     - Quick experiments.  Provides no space-coverage guarantees.


The ``ensemble_init_scale`` parameter controls the spread.  For structured
methods (LHS, Sobol, grid), samples are drawn in [0, 1] and mapped to:

.. code-block:: text

    value = base_value × ((1 - scale) + sample × 2 × scale)

So ``scale=0.5`` maps samples to [0.5×base, 1.5×base].  If the pool has a
:class:`~quantammsim.core_simulator.param_schema.ParamSpec` with Optuna
ranges, those ranges are used instead for tighter, schema-aware initialisation.

Example — comparing LHS and Gaussian:

.. code-block:: python

    import matplotlib.pyplot as plt

    # Train with LHS initialisation
    run_fp_lhs = {**base_fingerprint, "ensemble_init_method": "lhs"}
    result_lhs = train_on_historic_data(run_fp_lhs, verbose=True)

    # Train with Gaussian initialisation
    run_fp_gauss = {**base_fingerprint, "ensemble_init_method": "gaussian"}
    result_gauss = train_on_historic_data(run_fp_gauss, verbose=True)


Multi-Hook Chaining
-------------------

The ensemble hook composes with other hooks via the double-underscore syntax.
Hooks are applied left-to-right (leftmost = highest MRO priority):

.. code-block:: python

    # Ensemble + bounded weights + mean reversion channel
    run_fingerprint["rule"] = "ensemble__bounded__mean_reversion_channel"

    # Ensemble + LVR tracking + momentum
    run_fingerprint["rule"] = "ensemble__lvr__momentum"

For example, combining ensemble training with per-asset weight bounds:

.. code-block:: python

    import jax.numpy as jnp

    run_fingerprint = {
        "tokens": ["BTC", "ETH", "SOL"],
        "rule": "ensemble__bounded__mean_reversion_channel",
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2024-01-01 00:00:00",
        "initial_pool_value": 1_000_000.0,
        "fees": 0.003,
        "do_arb": True,
        "return_val": "daily_log_sharpe",
        "chunk_period": 1440,

        # Ensemble config
        "n_ensemble_members": 4,
        "ensemble_init_method": "lhs",
        "ensemble_init_scale": 0.5,

        # Per-asset bounds (applied after ensemble averaging)
        "min_weights_per_asset": jnp.array([0.2, 0.2, 0.1]),
        "max_weights_per_asset": jnp.array([0.5, 0.5, 0.3]),

        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "adam",
            "base_lr": 0.05,
            "n_iterations": 500,
            "batch_size": 16,
            "n_parameter_sets": 4,
        },
    }

    train_on_historic_data(run_fingerprint, verbose=True)

The order matters: ``ensemble__bounded__rule`` means the ensemble hook has
higher priority than the bounded hook.  The ensemble averages raw rule outputs
*before* bounds are enforced — this is usually what you want, since bounds
should constrain the final output, not the individual member contributions.

You can also construct the hooked pool class manually:

.. code-block:: python

    from quantammsim.pools.creator import create_hooked_pool_instance
    from quantammsim.hooks.ensemble_averaging_hook import EnsembleAveragingHook
    from quantammsim.hooks.bounded_weights_hook import BoundedWeightsHook
    from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import (
        MeanReversionChannelPool,
    )

    pool = create_hooked_pool_instance(
        MeanReversionChannelPool,
        BoundedWeightsHook,
        EnsembleAveragingHook,
    )


Ensemble + Walk-Forward Validation
-----------------------------------

Ensemble training is most powerful when combined with walk-forward analysis
to verify that the regularisation effect translates to OOS performance:

.. code-block:: python

    from quantammsim.runners.training_evaluator import TrainingEvaluator

    run_fingerprint = {
        "tokens": ["BTC", "ETH"],
        "rule": "ensemble__mean_reversion_channel",
        "startDateString": "2022-06-01 00:00:00",
        "endDateString": "2024-06-01 00:00:00",
        "initial_pool_value": 1_000_000.0,
        "fees": 0.003,
        "do_arb": True,
        "return_val": "daily_log_sharpe",
        "chunk_period": 1440,
        "bout_offset": 1440 * 14,

        # Ensemble
        "n_ensemble_members": 4,
        "ensemble_init_method": "lhs",
        "ensemble_init_scale": 0.5,

        # Early stopping
        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "adam",
            "base_lr": 0.05,
            "n_iterations": 1000,
            "batch_size": 16,
            "n_parameter_sets": 4,
            "early_stopping": True,
            "early_stopping_patience": 100,
            "early_stopping_metric": "daily_log_sharpe",
            "val_fraction": 0.2,
        },
    }

    evaluator = TrainingEvaluator.from_runner(
        "train_on_historic_data",
        n_cycles=4,
        compute_rademacher=True,
    )

    result = evaluator.evaluate(run_fingerprint)
    evaluator.print_report(result)

Compare against a non-ensemble baseline to quantify the regularisation
benefit:

.. code-block:: python

    from quantammsim.runners.training_evaluator import compare_trainers

    # Same config but without ensemble
    run_fp_no_ensemble = {**run_fingerprint, "rule": "mean_reversion_channel"}
    run_fp_no_ensemble.pop("n_ensemble_members", None)

    comparison = compare_trainers(
        run_fingerprint,
        trainers={
            "ensemble_4": TrainingEvaluator.from_runner(
                "train_on_historic_data", n_cycles=4,
            ),
            "no_ensemble": TrainingEvaluator.from_runner(
                "train_on_historic_data", n_cycles=4,
            ),
        },
    )


Parameter Shapes
----------------

Understanding the parameter tensor layout is important for debugging:

.. code-block:: text

    Without ensemble:
      params["log_k"]           shape: (n_parameter_sets, n_assets)
      params["logit_lamb"]      shape: (n_parameter_sets,)

    With 4 ensemble members:
      params["log_k"]           shape: (n_parameter_sets, 4, n_assets)
      params["logit_lamb"]      shape: (n_parameter_sets, 4)
      params["initial_weights_logits"]  shape: (n_parameter_sets, n_assets)
                                        ← SHARED, no ensemble dim

Note that ``initial_weights_logits`` is shared across ensemble members
because the ensemble is about the *strategy* (rule outputs), not the starting
allocation.  All members begin with the same initial weights and diverge
through their different rule parameters.


Best Practices
--------------

**Member count**: 4 members is a good starting point.  Below 3, the
diversity benefit is marginal.  Above 8, returns diminish while memory usage
grows linearly.  The compute cost is proportional to ``n_parameter_sets ×
n_ensemble_members``.

**Initialisation method**: Use ``"lhs"`` unless you have reason not to.  It
provides good space coverage without the pathologies of pure random sampling
(clumping, poor tail coverage).

**Init scale**: Start with 0.5.  Too small (< 0.1) and members collapse to
the same solution.  Too large (> 2.0) and some members start in poor regions
and drag down the average.

**Combine with other regularisation**: Ensemble training is complementary to
early stopping, price noise, and SWA.  The strongest configs typically use
ensemble + early stopping + price noise:

.. code-block:: python

    run_fingerprint.update({
        "n_ensemble_members": 4,
        "ensemble_init_method": "lhs",
        "ensemble_init_scale": 0.5,
        "price_noise_sigma": 0.001,
        "optimisation_settings": {
            **run_fingerprint["optimisation_settings"],
            "early_stopping": True,
            "early_stopping_patience": 100,
            "val_fraction": 0.2,
        },
    })

**Seed control**: Set ``ensemble_init_seed`` for reproducibility.  Different
seeds with the same method will produce different member placements, which
can cause variance in results.  Pin the seed for production configs.


See Also
--------

- :doc:`../user_guide/hooks` — Hook system overview and custom hooks
- :doc:`../user_guide/robustness_features` — All regularisation techniques
- :doc:`walk_forward_analysis` — Walk-forward validation tutorial
- :doc:`../user_guide/per_asset_bounds` — Per-asset weight bounds (composable with ensemble)
- :mod:`quantammsim.hooks.ensemble_averaging_hook` — API reference
