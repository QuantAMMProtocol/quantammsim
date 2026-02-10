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


Deflated Sharpe Ratio
---------------------

When evaluating many strategies (e.g. via Optuna), the best observed Sharpe
ratio is inflated by selection bias.  The **Deflated Sharpe Ratio** (Bailey &
Lopez de Prado, 2014) corrects for this multiple-testing effect by comparing
the observed SR against the expected maximum SR under the null hypothesis that
all strategies are noise.

.. code-block:: python

    from quantammsim.utils.post_train_analysis import deflated_sharpe_ratio

    dsr = deflated_sharpe_ratio(
        observed_sr=1.2,   # best OOS Sharpe
        n_trials=50,       # number of Optuna trials tested
        T=365,             # number of OOS daily observations
    )

    if dsr["significant"]:
        print("Strategy is significant at 95% confidence")
    else:
        print(f"DSR = {dsr['dsr']:.3f} — likely selection bias")

DSR is intended for use after hyperparameter tuning — pass
``n_trials`` from the Optuna study and the best trial's OOS Sharpe.


Block Bootstrap Confidence Intervals
-------------------------------------

Standard confidence intervals for Sharpe ratios assume i.i.d. returns, which
is violated in practice (autocorrelation from market microstructure, regime
persistence, etc.).  **Block bootstrap** preserves the autocorrelation
structure by resampling contiguous blocks of returns.

.. code-block:: python

    from quantammsim.utils.post_train_analysis import block_bootstrap_sharpe_ci

    ci = block_bootstrap_sharpe_ci(
        daily_returns=oos_daily_returns,
        block_length=10,    # 10 days captures weekly autocorrelation
        n_bootstrap=10000,
        confidence=0.95,
    )
    print(f"Sharpe 95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")

The evaluator automatically concatenates OOS daily returns across walk-forward
cycles and computes bootstrap CIs on the aggregate.


Return Decomposition
--------------------

Pool returns can be decomposed into four components:

.. math::

   r_{\text{pool}} = r_{\text{hodl}} + \Delta_{\text{divergence}} + f_{\text{fees}} + \alpha_{\text{strategy}}

where:

* **HODL return** — what the initial reserves would be worth at final prices
* **Divergence loss** — the cost of continuous rebalancing in a constant-weight
  AMM (always ≤ 0 for G3M pools)
* **Fee income** — revenue from swap fees (external input)
* **Strategy alpha** — residual value from dynamic weight changes

.. code-block:: python

    from quantammsim.utils.post_train_analysis import decompose_pool_returns

    decomp = decompose_pool_returns(
        values=result["value"],
        reserves=result["reserves"],
        prices=result["prices"],
    )

This decomposition answers: *"Is the strategy actually generating alpha, or
is performance just from HODL returns in a bull market?"*


Regime-Tagged Evaluation
------------------------

Each walk-forward cycle is automatically tagged with the OOS period's
**volatility regime** (low / medium / high) and **trend direction**
(bull / bear / sideways).  This allows post-hoc analysis of strategy
robustness across market conditions:

.. code-block:: python

    result = evaluator.evaluate(run_fingerprint)

    for cycle in result.cycles:
        print(f"Cycle {cycle.cycle_number}: "
              f"{cycle.volatility_regime} / {cycle.trend_regime} "
              f"→ OOS Sharpe = {cycle.oos_sharpe:.3f}")

Regime classification uses the mean of daily log returns across all assets:

* **Volatility**: annualised vol < 0.4 = low, < 0.8 = medium, ≥ 0.8 = high
* **Trend**: cumulative log return > 0.1 = bull, < −0.1 = bear, else sideways


Recommended Workflow
--------------------

1. **Start simple**: Train with ``daily_log_sharpe``, no augmentation.
2. **Add early stopping**: Usually the single biggest improvement.
3. **Add price noise**: ``sigma = 0.001`` is a safe starting point.
4. **Run walk-forward analysis**: Check WFE and IS-OOS gap.
5. **If overfitting persists**: Add ensemble training, SWA, or weight decay.
6. **Use hyperparameter tuning**: Optimise robustness metrics (WFE, adjusted
   Sharpe) rather than just IS performance.
7. **Validate statistically**: Use the Deflated Sharpe Ratio to check
   whether performance survives multiple-testing correction, and bootstrap
   CIs to quantify uncertainty.
8. **Decompose returns**: Use return decomposition to verify that alpha
   comes from dynamic weight management, not just holding in a bull market.
