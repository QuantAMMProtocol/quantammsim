Hyperparameter Tuning
=====================

HyperparamTuner
~~~~~~~~~~~~~~~

Optuna-based hyperparameter optimisation with walk-forward evaluation as the
objective.  Supports single-objective (OOS Sharpe, WFE) and multi-objective
(Pareto front) tuning with trial pruning.

.. automodule:: quantammsim.runners.hyperparam_tuner
   :members:
   :exclude-members: best_params, best_value, best_evaluation, n_trials, n_completed, n_pruned, n_failed, all_trials, pareto_front, total_time_seconds, params
   :show-inheritance:
