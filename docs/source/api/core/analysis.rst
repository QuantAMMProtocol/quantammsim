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

The ``quantammsim.utils.post_train_analysis`` module provides utilities for analyzing results after training.

Period Metrics
~~~~~~~~~~~~~~

Calculate comprehensive metrics for a simulation period:

.. code-block:: python

    from quantammsim.utils.post_train_analysis import calculate_period_metrics

    # After running a simulation
    result = do_run_on_historic_data(fingerprint, params)

    # Calculate all metrics
    metrics = calculate_period_metrics(result)

Returns a dictionary with:

- ``sharpe`` - Daily Sharpe ratio (annualized)
- ``jax_sharpe`` - JAX-computed Sharpe ratio
- ``return`` - Total return
- ``returns_over_hodl`` - Return relative to holding initial portfolio
- ``returns_over_uniform_hodl`` - Return relative to uniform hold
- ``annualised_returns`` - Annualized total return
- ``annualised_returns_over_hodl`` - Annualized return vs HODL
- ``annualised_returns_over_uniform_hodl`` - Annualized return vs uniform HODL
- ``ulcer`` - Ulcer index
- ``calmar`` - Calmar ratio
- ``sterling`` - Sterling ratio

Continuous Test Metrics
~~~~~~~~~~~~~~~~~~~~~~~

For walk-forward analysis with separate train and test periods:

.. code-block:: python

    from quantammsim.utils.post_train_analysis import calculate_continuous_test_metrics

    # Assuming continuous_results spans train + test
    test_metrics = calculate_continuous_test_metrics(
        continuous_results=full_results,
        train_len=train_period_length,
        test_len=test_period_length,
        prices=price_data
    )

    # Returns metrics prefixed with 'continuous_test_'
    print(test_metrics['continuous_test_sharpe'])
    print(test_metrics['continuous_test_return'])

This extracts only the test period from a continuous simulation and calculates metrics on that portion.
