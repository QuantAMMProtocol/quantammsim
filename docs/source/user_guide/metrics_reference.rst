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
   :widths: 25 50 25

   * - ``return_val``
     - Description
     - Differentiable
   * - ``returns``
     - Total cumulative return: ``final_value / initial_value - 1``
     - Yes
   * - ``returns_over_hodl``
     - Return relative to uniform HODL baseline
     - Yes
   * - ``log_returns``
     - Log return: ``log(final_value / initial_value)``
     - Yes
   * - ``neg_returns``
     - Negated returns (for minimisation)
     - Yes

Risk-Adjusted Metrics
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - ``return_val``
     - Description
     - Differentiable
   * - ``daily_log_sharpe``
     - Daily log-return Sharpe ratio (**default**)
     - Yes
   * - ``sharpe``
     - Annualised Sharpe ratio (arithmetic returns)
     - Yes
   * - ``daily_sharpe``
     - Daily Sharpe ratio
     - Yes
   * - ``neg_sharpe``
     - Negated Sharpe (for minimisation)
     - Yes
   * - ``sortino``
     - Sortino ratio (downside deviation only)
     - Approximately

Drawdown Metrics
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - ``return_val``
     - Description
     - Differentiable
   * - ``max_drawdown``
     - Maximum peak-to-trough drawdown (negative value)
     - No (argmax)
   * - ``calmar``
     - Calmar ratio: annualised return / max drawdown
     - No
   * - ``sterling``
     - Sterling ratio: annualised return / average of top-N drawdowns
     - No
   * - ``ulcer_index``
     - Ulcer Index: RMS of percentage drawdowns from running peak
     - Approximately

Value at Risk Metrics
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - ``return_val``
     - Description
     - Differentiable
   * - ``var_5``
     - 5th percentile of daily returns (traditional VaR)
     - No (sort)
   * - ``var_trad``
     - Parametric VaR using mean and std of returns
     - Yes
   * - ``raroc``
     - Risk-Adjusted Return on Capital: return / VaR
     - No
   * - ``rovar``
     - Return on VaR using empirical quantile
     - No
   * - ``rovar_trad``
     - Return on VaR using parametric estimate
     - Yes


Special / Combined Metrics
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - ``return_val``
     - Description
     - Differentiable
   * - ``reserves_and_values``
     - Returns raw reserves + values (not a scalar objective)
     - N/A
   * - ``neg_returns``
     - Negated returns
     - Yes
   * - ``neg_sharpe``
     - Negated Sharpe
     - Yes

Choosing a Training Metric
--------------------------

For gradient-based training:

* **``daily_log_sharpe``** (default) — Recommended. Log returns are more
  numerically stable, and daily periodicity avoids annualisation artifacts.
* **``sharpe``** — Classic choice. Works well but can have gradient issues
  with very short or very long simulation windows.
* **``calmar``** and **``sterling``** — Drawdown-aware, but contain
  non-differentiable operations (argmax). Gradients are approximate.

For Optuna-based optimisation (gradient-free):

* Any metric works, since Optuna doesn't need gradients.
* **``calmar``** and **``worst-case``** metrics are popular for robust
  strategy selection.
