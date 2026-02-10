Metrics Reference
=================

quantammsim provides ~30 financial metrics for training objectives and
evaluation.  All metrics are computed by
:func:`~quantammsim.core_simulator.forward_pass._calculate_return_value`,
which accepts the ``return_val`` string from the run fingerprint.

.. note::

   The **default training metric** is ``daily_log_sharpe``, not ``sharpe``.
   This uses log returns and daily periodicity, which is more numerically
   stable for gradient-based optimisation.


Return Metrics
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - ``return_val``
     - Description
     - Differentiable
   * - ``returns``
     - Total cumulative return: ``final_value / initial_value - 1``
     - Yes
   * - ``annualised_returns``
     - Returns annualised by simulation length: ``(final / initial)^(365*1440 / T) - 1``
     - Yes
   * - ``returns_over_hodl``
     - Return relative to holding the initial reserves
     - Yes
   * - ``annualised_returns_over_hodl``
     - Annualised return relative to holding the initial reserves
     - Yes
   * - ``returns_over_uniform_hodl``
     - Return relative to a uniform (equal-value) hold of all assets
     - Yes
   * - ``annualised_returns_over_uniform_hodl``
     - Annualised return relative to uniform hold
     - Yes

Risk-Adjusted Metrics
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - ``return_val``
     - Description
     - Differentiable
   * - ``daily_log_sharpe``
     - Daily log-return Sharpe ratio, annualised via sqrt(365) (**default**)
     - Yes
   * - ``sharpe``
     - Annualised Sharpe ratio from minute-resolution arithmetic returns
     - Yes
   * - ``daily_sharpe``
     - Daily arithmetic-return Sharpe ratio, annualised via sqrt(365)
     - Yes

Drawdown Metrics
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - ``return_val``
     - Description
     - Differentiable
   * - ``greatest_draw_down``
     - Maximum peak-to-trough drawdown from initial value (negative)
     - Approx
   * - ``weekly_max_drawdown``
     - Worst maximum drawdown across non-overlapping weekly chunks
     - No (argmax)
   * - ``calmar``
     - Calmar ratio: annualised return / max drawdown
     - No
   * - ``sterling``
     - Sterling ratio: annualised return / average of chunk drawdowns (monthly chunks)
     - No
   * - ``ulcer``
     - Negated Ulcer Index: RMS of percentage drawdowns from running peak (monthly chunks)
     - Approx

Value at Risk Metrics
---------------------

VaR metrics are available at daily (``daily_``) and weekly (``weekly_``)
frequencies, at 95% and 99% confidence levels.  The ``_trad`` suffix uses
end-of-period (close-to-close) returns; the non-``_trad`` variant uses all
intraday returns within each chunk.

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - ``return_val``
     - Description
     - Differentiable
   * - ``daily_var_95%``
     - 5th percentile of intraday returns (daily chunks)
     - No (sort)
   * - ``daily_var_95%_trad``
     - 5th percentile of daily close-to-close returns
     - No (sort)
   * - ``daily_var_99%``
     - 1st percentile of intraday returns (daily chunks)
     - No (sort)
   * - ``daily_var_99%_trad``
     - 1st percentile of daily close-to-close returns
     - No (sort)
   * - ``weekly_var_95%``
     - 5th percentile of intraday returns (weekly chunks)
     - No (sort)
   * - ``weekly_var_95%_trad``
     - 5th percentile of weekly close-to-close returns
     - No (sort)
   * - ``weekly_var_99%``
     - 1st percentile of intraday returns (weekly chunks)
     - No (sort)
   * - ``weekly_var_99%_trad``
     - 1st percentile of weekly close-to-close returns
     - No (sort)

RAROC and ROVAR Metrics
-----------------------

RAROC (Risk-Adjusted Return on Capital) divides annualised total return by
annualised VaR.  ROVAR (Return Over VaR) annualises per-chunk returns
independently, averages, then divides by annualised VaR.

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - ``return_val``
     - Description
     - Differentiable
   * - ``daily_raroc``
     - RAROC using daily intraday VaR (95%)
     - No
   * - ``weekly_raroc``
     - RAROC using weekly intraday VaR (95%)
     - No
   * - ``daily_rovar``
     - ROVAR using daily intraday VaR (95%)
     - No
   * - ``weekly_rovar``
     - ROVAR using weekly intraday VaR (95%)
     - No
   * - ``monthly_rovar``
     - ROVAR using monthly intraday VaR (95%)
     - No
   * - ``daily_rovar_trad``
     - ROVAR using daily close-to-close VaR (95%)
     - Approx
   * - ``weekly_rovar_trad``
     - ROVAR using weekly close-to-close VaR (95%)
     - Approx
   * - ``monthly_rovar_trad``
     - ROVAR using monthly close-to-close VaR (95%)
     - Approx


Special / Diagnostic Metrics
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - ``return_val``
     - Description
     - Differentiable
   * - ``value``
     - Full value-over-time array (not a scalar objective)
     - N/A
   * - ``reserves_and_values``
     - Dict with ``final_reserves``, ``final_value``, ``value``, ``prices``, ``reserves``
     - N/A

Choosing a Training Metric
--------------------------

For gradient-based training:

* **``daily_log_sharpe``** (default) -- Recommended. Log returns are more
  numerically stable, and daily periodicity avoids annualisation artifacts.
* **``sharpe``** -- Classic choice. Works well but can have gradient issues
  with very short or very long simulation windows.
* **``calmar``** and **``sterling``** -- Drawdown-aware, but contain
  non-differentiable operations (argmax). Gradients are approximate.

For Optuna-based optimisation (gradient-free):

* Any metric works, since Optuna doesn't need gradients.
* **``calmar``** and **``weekly_max_drawdown``** are popular for robust
  strategy selection.
