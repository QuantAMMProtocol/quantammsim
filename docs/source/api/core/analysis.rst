Financial Analysis Tools
========================

The ``quantammsim.simulator_analysis_tools.finance`` module provides comprehensive financial analysis capabilities for evaluating simulation results.

Risk-Adjusted Return Metrics
----------------------------

Jensen's Alpha
~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_jensens_alpha
    )

    alpha = calculate_jensens_alpha(
        portfolio_returns=daily_returns,
        rf_values=risk_free_rates,
        benchmark_returns=benchmark_daily_returns
    )

Calculates annualized Jensen's Alpha, measuring risk-adjusted excess return over the benchmark.

Sharpe Ratio
~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_sharpe_ratio
    )

    result = calculate_sharpe_ratio(
        portfolio_returns=daily_returns,
        rf_values=risk_free_rates
    )
    print(f"Daily Sharpe: {result['sharpe_ratio']}")
    print(f"Annualized Sharpe: {result['annualized_sharpe_ratio']}")

Returns both daily and annualized Sharpe ratios.

Sortino Ratio
~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_sortino_ratio
    )

    sortino = calculate_sortino_ratio(
        portfolio_returns=daily_returns,
        rf_values=risk_free_rates
    )

Measures risk-adjusted return using only downside deviation.

Tracking and Information Ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_tracking_error_and_information_ratio
    )

    result = calculate_tracking_error_and_information_ratio(
        portfolio_returns=daily_returns,
        benchmark_returns=benchmark_returns
    )
    print(f"Tracking Error: {result['tracking_error']}")
    print(f"Information Ratio: {result['information_ratio']}")

Drawdown Analysis
-----------------

Drawdown Statistics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_drawdown_statistics
    )

    stats = calculate_drawdown_statistics(
        daily_returns=daily_returns,
        rf_values=risk_free_rates
    )

Returns a dictionary containing:

- ``max_drawdown`` - Maximum peak-to-trough decline
- ``max_drawdown_duration`` - Longest drawdown period
- ``average_drawdown`` - Mean drawdown across all periods
- ``drawdown_timeseries`` - Full drawdown series for plotting

Maximum Daily Drawdown
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_max_daily_drawdown
    )

    # Calculate weekly maximum drawdowns
    weekly_max_dd = calculate_max_daily_drawdown(
        daily_returns=daily_returns,
        period="weekly"
    )

Periods can be: ``"daily"``, ``"weekly"``, ``"monthly"``, ``"yearly"``

Risk Metrics
------------

Ulcer Index
~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_ulcer_index,
        calcuate_period_ulcer_index
    )

    # Overall ulcer index
    ulcer = calculate_ulcer_index(daily_returns)

    # Period-based ulcer index
    monthly_ulcer = calcuate_period_ulcer_index(
        daily_returns=daily_returns,
        period="monthly"
    )

Measures downside risk by penalizing depth and duration of drawdowns.

Sterling Ratio
~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_sterling_ratio,
        calcuate_period_sterling_index
    )

    sterling = calculate_sterling_ratio(
        returns=daily_returns,
        rf=risk_free_rates
    )

Risk-adjusted return relative to average drawdown.

Value at Risk (VaR) Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_return_on_VaR,
        calculate_cdar,
        calculate_monthly_cdar
    )

    # Return on VaR
    rovar = calculate_return_on_VaR(
        portfolio_returns=daily_returns,
        rf_values=risk_free_rates,
        confidence_level=0.95
    )

    # Conditional Drawdown at Risk
    cdar = calculate_cdar(
        portfolio_returns=daily_returns,
        confidence_level=0.95
    )

Omega Ratio
~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_omega_ratio
    )

    omega = calculate_omega_ratio(
        portfolio_returns=daily_returns,
        rf_values=risk_free_rates,
        threshold=0  # Threshold return
    )

Ratio of probability-weighted gains to losses.

Benchmark Comparison
--------------------

Capture Ratios
~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_capture_ratios
    )

    capture = calculate_capture_ratios(
        portfolio_returns=daily_returns,
        benchmark_returns=benchmark_returns
    )
    print(f"Upside Capture: {capture['upside_capture']}")
    print(f"Downside Capture: {capture['downside_capture']}")
    print(f"Capture Ratio: {capture['capture_ratio']}")

Measures how much of the benchmark's up/down moves the portfolio captures.

Distribution Statistics
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
        calculate_distribution_statistics
    )

    stats = calculate_distribution_statistics(daily_returns)
    print(f"Mean: {stats['mean']}")
    print(f"Std: {stats['std']}")
    print(f"Skewness: {stats['skewness']}")
    print(f"Kurtosis: {stats['kurtosis']}")

Complete Analysis Pipeline
--------------------------

For a complete financial analysis, use the high-level calculator:

.. code-block:: python

    from quantammsim.simulator_analysis_tools.finance.param_financial_calculator import (
        run_financial_analysis
    )

    results = run_financial_analysis(
        portfolio_daily_returns=daily_returns,
        startDateString="2023-01-01 00:00:00",
        endDateString="2024-01-01 00:00:00",
        bechmark_names=["BTC", "ETH"],
        benchmarks_returns=benchmark_returns_dict
    )

This returns a comprehensive analysis including all metrics and time series data.

Using with Simulation Results
-----------------------------

After running a simulation with ``do_run_on_historic_data``:

.. code-block:: python

    from quantammsim.runners.jax_runners import do_run_on_historic_data
    from quantammsim.utils.post_train_analysis import calculate_period_metrics

    # Run simulation
    result = do_run_on_historic_data(
        run_fingerprint=fingerprint,
        params=params
    )

    # Calculate metrics
    metrics = calculate_period_metrics(result)
    print(f"Sharpe: {metrics['sharpe']}")
    print(f"Return: {metrics['return']}")
    print(f"Ulcer Index: {metrics['ulcer']}")
    print(f"Calmar Ratio: {metrics['calmar']}")
    print(f"Sterling Ratio: {metrics['sterling']}")

Available Metrics
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - Jensen's Alpha
     - Risk-adjusted excess return vs benchmark (annualized)
   * - Sharpe Ratio
     - Risk-adjusted return (daily and annualized)
   * - Sortino Ratio
     - Downside risk-adjusted return
   * - Tracking Error
     - Standard deviation of active returns
   * - Information Ratio
     - Active return per unit tracking error
   * - Max Drawdown
     - Maximum peak-to-trough decline
   * - Ulcer Index
     - Depth and duration weighted drawdown measure
   * - Sterling Ratio
     - Return relative to average drawdown
   * - Return on VaR
     - Return relative to Value at Risk
   * - CDaR
     - Conditional Drawdown at Risk
   * - Omega Ratio
     - Probability-weighted gains vs losses
   * - Capture Ratios
     - Upside/downside benchmark capture

Post-Training Analysis
----------------------

The ``quantammsim.utils.post_train_analysis`` module provides utilities for
analysing results after training: period metrics, statistical validation of
Sharpe ratios, and return decomposition.

.. automodule:: quantammsim.utils.post_train_analysis
   :members:
   :show-inheritance:

Usage Examples
~~~~~~~~~~~~~~

**Period metrics** — after running a simulation:

.. code-block:: python

    from quantammsim.utils.post_train_analysis import calculate_period_metrics

    result = do_run_on_historic_data(fingerprint, params)
    metrics = calculate_period_metrics(result)
    print(f"Sharpe: {metrics['sharpe']}")
    print(f"Calmar: {metrics['calmar']}")

**Deflated Sharpe Ratio** — correct for multiple testing:

.. code-block:: python

    from quantammsim.utils.post_train_analysis import deflated_sharpe_ratio

    dsr = deflated_sharpe_ratio(
        observed_sr=1.2,   # best OOS Sharpe
        n_trials=50,       # number of Optuna trials
        T=365,             # number of OOS daily observations
    )
    print(f"DSR p-value: {dsr['dsr']:.3f}")
    print(f"Significant: {dsr['significant']}")

**Block bootstrap CIs** — confidence interval preserving autocorrelation:

.. code-block:: python

    from quantammsim.utils.post_train_analysis import block_bootstrap_sharpe_ci

    ci = block_bootstrap_sharpe_ci(
        daily_returns=metrics["daily_returns"],
        block_length=10,
    )
    print(f"Sharpe 95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")

**Return decomposition** — isolate strategy alpha from divergence loss:

.. code-block:: python

    from quantammsim.utils.post_train_analysis import decompose_pool_returns

    decomp = decompose_pool_returns(
        values=result["value"],
        reserves=result["reserves"],
        prices=result["prices"],
    )
    print(f"HODL return:      {decomp['hodl_return']:.4f}")
    print(f"Divergence loss:  {decomp['divergence_loss']:.4f}")
    print(f"Strategy alpha:   {decomp['strategy_alpha']:.4f}")
