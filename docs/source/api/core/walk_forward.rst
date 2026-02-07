Walk-Forward Analysis
=====================

Robust Walk-Forward Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Core utilities for walk-forward validation: Rademacher complexity, Walk-Forward
Efficiency (WFE), and cycle generation.

.. automodule:: quantammsim.runners.robust_walk_forward
   :members:
   :show-inheritance:
   :exclude-members: cycle_number, train_start_date, train_end_date, test_start_date, test_end_date, train_start_idx, train_end_idx, test_start_idx, test_end_idx

Training Evaluator
~~~~~~~~~~~~~~~~~~

Walk-forward evaluation framework with pluggable trainer wrappers, per-cycle
IS/OOS metric extraction, and aggregate robustness diagnostics.

.. automodule:: quantammsim.runners.training_evaluator
   :members:
   :show-inheritance:
   :exclude-members: cycle_number, is_sharpe, is_returns_over_hodl, oos_sharpe, oos_returns_over_hodl, walk_forward_efficiency, is_oos_gap, epochs_trained, rademacher_complexity, adjusted_oos_sharpe, is_calmar, oos_calmar, is_sterling, oos_sterling, is_ulcer, oos_ulcer, is_returns, oos_returns, is_daily_log_sharpe, oos_daily_log_sharpe, trained_params, train_start_date, train_end_date, test_start_date, test_end_date, run_location, run_fingerprint, trainer_name, trainer_config, cycles, mean_wfe, mean_oos_sharpe, std_oos_sharpe, worst_oos_sharpe, mean_is_oos_gap, aggregate_rademacher, adjusted_mean_oos_sharpe, is_effective, effectiveness_reasons
