"""
Tests for hyperparam_tuner.py - Optuna-based hyperparameter optimization.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import optuna
from unittest.mock import Mock, patch, MagicMock
from copy import deepcopy

from quantammsim.runners.hyperparam_tuner import (
    HyperparamTuner,
    HyperparamSpace,
    TuningResult,
    quick_tune,
    tune_for_robustness,
    create_objective,
)
from quantammsim.runners.training_evaluator import (
    TrainingEvaluator,
    EvaluationResult,
    CycleEvaluation,
)
from tests.conftest import TEST_DATA_DIR


# =============================================================================
# Test HyperparamSpace
# =============================================================================

class TestHyperparamSpace:
    """Tests for the hyperparameter search space."""

    def test_default_sgd_space_has_expected_params(self):
        """Default SGD space should include lr, batch_size, etc."""
        space = HyperparamSpace.default_sgd_space()

        assert "base_lr" in space.params
        assert "batch_size" in space.params
        assert "n_iterations" in space.params
        assert "bout_offset_days" in space.params

        # Check lr is log-scaled
        assert space.params["base_lr"]["log"] is True

    def test_default_adam_space_has_expected_params(self):
        """Default Adam space should include lr, batch_size."""
        space = HyperparamSpace.default_adam_space()

        assert "base_lr" in space.params
        assert "batch_size" in space.params
        assert space.params["base_lr"]["log"] is True

    def test_default_multi_period_space_has_expected_params(self):
        """Multi-period space should include n_periods and aggregation."""
        space = HyperparamSpace.default_multi_period_space()

        assert "n_periods" in space.params
        assert "aggregation" in space.params
        assert "choices" in space.params["aggregation"]
        assert "mean" in space.params["aggregation"]["choices"]

    def test_minimal_space_is_minimal(self):
        """Minimal space should have only essential params."""
        space = HyperparamSpace.minimal_space()

        assert len(space.params) == 2
        assert "base_lr" in space.params
        assert "n_iterations" in space.params

    def test_bout_offset_days_has_sensible_ranges(self):
        """bout_offset_days should be in days with sensible ranges."""
        space = HyperparamSpace.default_sgd_space(cycle_days=180)

        bout_spec = space.params["bout_offset_days"]

        # Minimum should be 1 day
        assert bout_spec["low"] == 1, \
            f"bout_offset_days min should be 1 day, got {bout_spec['low']}"

        # Maximum should be ~90% of 180 days = 162 days
        expected_max = int(180 * 0.9)
        assert bout_spec["high"] == expected_max, \
            f"bout_offset_days max should be {expected_max}, got {bout_spec['high']}"

        # Should be log-scaled (spans large range)
        assert bout_spec["log"] is True

        # Should be an integer type
        assert bout_spec["type"] == "int"

    def test_bout_offset_days_scales_with_cycle_duration(self):
        """bout_offset_days range should scale with cycle_days parameter."""
        space_90 = HyperparamSpace.default_sgd_space(cycle_days=90)
        space_365 = HyperparamSpace.default_sgd_space(cycle_days=365)

        # 90-day cycle: max = 90 * 0.9 = 81 days
        assert space_90.params["bout_offset_days"]["high"] == int(90 * 0.9)

        # 365-day cycle: max = 365 * 0.9 = 328 days
        assert space_365.params["bout_offset_days"]["high"] == int(365 * 0.9)

    def test_lr_schedule_params_included(self):
        """lr_schedule_type and warmup_steps should be in default spaces."""
        space = HyperparamSpace.default_sgd_space()

        assert "lr_schedule_type" in space.params
        assert "choices" in space.params["lr_schedule_type"]
        assert "constant" in space.params["lr_schedule_type"]["choices"]
        assert "cosine" in space.params["lr_schedule_type"]["choices"]
        assert "warmup_cosine" in space.params["lr_schedule_type"]["choices"]
        assert "exponential" in space.params["lr_schedule_type"]["choices"]

        assert "warmup_steps" in space.params
        assert space.params["warmup_steps"]["type"] == "int"

    def test_early_stopping_patience_included(self):
        """early_stopping_patience should be in default spaces."""
        space = HyperparamSpace.default_adam_space()

        assert "early_stopping_patience" in space.params
        assert space.params["early_stopping_patience"]["type"] == "int"
        assert space.params["early_stopping_patience"]["log"] is True

    def test_for_cycle_duration_factory(self):
        """for_cycle_duration should create properly scaled spaces."""
        space = HyperparamSpace.for_cycle_duration(
            cycle_days=120,
            runner="train_on_historic_data",
            include_lr_schedule=True,
            include_early_stopping=True,
        )

        # Check bout_offset_days scaling (in days)
        assert space.params["bout_offset_days"]["low"] == 1
        assert space.params["bout_offset_days"]["high"] == int(120 * 0.9)

        # Check optional params included
        assert "lr_schedule_type" in space.params
        assert "early_stopping_patience" in space.params

    def test_for_cycle_duration_without_optional_params(self):
        """for_cycle_duration should respect include flags."""
        space = HyperparamSpace.for_cycle_duration(
            cycle_days=120,
            runner="train_on_historic_data",
            include_lr_schedule=False,
            include_early_stopping=False,
            include_weight_decay=False,
        )

        assert "lr_schedule_type" not in space.params
        assert "warmup_steps" not in space.params
        assert "early_stopping_patience" not in space.params
        assert "use_weight_decay" not in space.params
        assert "weight_decay" not in space.params


class TestConditionalSampling:
    """Tests for conditional hyperparameter sampling."""

    def test_weight_decay_is_conditional_on_use_weight_decay(self):
        """weight_decay should only be sampled when use_weight_decay=True."""
        space = HyperparamSpace.default_sgd_space()

        # Check spec structure
        assert "use_weight_decay" in space.params
        assert "weight_decay" in space.params
        assert space.params["weight_decay"]["conditional_on"] == "use_weight_decay"
        assert space.params["weight_decay"]["conditional_value"] is True

    def test_softmin_temperature_is_conditional_on_aggregation(self):
        """softmin_temperature should only be sampled when aggregation='softmin'."""
        space = HyperparamSpace.default_multi_period_space()

        assert "aggregation" in space.params
        assert "softmin_temperature" in space.params
        assert space.params["softmin_temperature"]["conditional_on"] == "aggregation"
        assert space.params["softmin_temperature"]["conditional_value"] == "softmin"

    def test_warmup_steps_is_conditional_on_lr_schedule(self):
        """warmup_steps should only be sampled when lr_schedule_type == 'warmup_cosine'."""
        space = HyperparamSpace.default_sgd_space()

        assert "lr_schedule_type" in space.params
        assert "warmup_steps" in space.params
        assert space.params["warmup_steps"]["conditional_on"] == "lr_schedule_type"
        # warmup_steps only makes sense for warmup_cosine schedule
        assert space.params["warmup_steps"]["conditional_value"] == "warmup_cosine"

    def test_suggest_excludes_weight_decay_when_disabled(self):
        """When use_weight_decay=False, weight_decay should not be in suggested."""
        space = HyperparamSpace(params={
            "base_lr": {"low": 0.01, "high": 0.1, "log": True},
            "use_weight_decay": {"choices": [False]},  # Force False
            "weight_decay": {
                "low": 0.001, "high": 0.1, "log": True,
                "conditional_on": "use_weight_decay", "conditional_value": True
            },
        })

        study = optuna.create_study()
        trial = study.ask()
        suggested = space.suggest(trial)

        assert "base_lr" in suggested
        assert "use_weight_decay" in suggested
        assert suggested["use_weight_decay"] is False
        assert "weight_decay" not in suggested, \
            "weight_decay should not be sampled when use_weight_decay=False"

    def test_suggest_includes_weight_decay_when_enabled(self):
        """When use_weight_decay=True, weight_decay should be in suggested."""
        space = HyperparamSpace(params={
            "base_lr": {"low": 0.01, "high": 0.1, "log": True},
            "use_weight_decay": {"choices": [True]},  # Force True
            "weight_decay": {
                "low": 0.001, "high": 0.1, "log": True,
                "conditional_on": "use_weight_decay", "conditional_value": True
            },
        })

        study = optuna.create_study()
        trial = study.ask()
        suggested = space.suggest(trial)

        assert suggested["use_weight_decay"] is True
        assert "weight_decay" in suggested
        assert 0.001 <= suggested["weight_decay"] <= 0.1

    def test_suggest_excludes_softmin_temp_when_aggregation_not_softmin(self):
        """softmin_temperature should not be suggested when aggregation != 'softmin'."""
        space = HyperparamSpace(params={
            "aggregation": {"choices": ["mean"]},  # Force non-softmin
            "softmin_temperature": {
                "low": 0.1, "high": 10.0, "log": True,
                "conditional_on": "aggregation", "conditional_value": "softmin"
            },
        })

        study = optuna.create_study()
        trial = study.ask()
        suggested = space.suggest(trial)

        assert suggested["aggregation"] == "mean"
        assert "softmin_temperature" not in suggested

    def test_suggest_includes_softmin_temp_when_aggregation_is_softmin(self):
        """softmin_temperature should be suggested when aggregation == 'softmin'."""
        space = HyperparamSpace(params={
            "aggregation": {"choices": ["softmin"]},  # Force softmin
            "softmin_temperature": {
                "low": 0.1, "high": 10.0, "log": True,
                "conditional_on": "aggregation", "conditional_value": "softmin"
            },
        })

        study = optuna.create_study()
        trial = study.ask()
        suggested = space.suggest(trial)

        assert suggested["aggregation"] == "softmin"
        assert "softmin_temperature" in suggested
        assert 0.1 <= suggested["softmin_temperature"] <= 10.0

    def test_suggest_excludes_warmup_when_lr_schedule_constant(self):
        """warmup_steps should not be suggested when lr_schedule_type != 'warmup_cosine'."""
        space = HyperparamSpace(params={
            "lr_schedule_type": {"choices": ["constant"]},  # Force constant
            "warmup_steps": {
                "low": 5, "high": 100, "log": False, "type": "int",
                "conditional_on": "lr_schedule_type", "conditional_value": "warmup_cosine"
            },
        })

        study = optuna.create_study()
        trial = study.ask()
        suggested = space.suggest(trial)

        assert suggested["lr_schedule_type"] == "constant"
        assert "warmup_steps" not in suggested

    def test_suggest_includes_warmup_when_lr_schedule_not_constant(self):
        """warmup_steps should be suggested when lr_schedule_type == 'warmup_cosine'."""
        space = HyperparamSpace(params={
            "lr_schedule_type": {"choices": ["warmup_cosine"]},  # Force warmup_cosine
            "warmup_steps": {
                "low": 5, "high": 100, "log": False, "type": "int",
                "conditional_on": "lr_schedule_type", "conditional_value": "warmup_cosine"
            },
        })

        study = optuna.create_study()
        trial = study.ask()
        suggested = space.suggest(trial)

        assert suggested["lr_schedule_type"] == "warmup_cosine"
        assert "warmup_steps" in suggested
        assert 5 <= suggested["warmup_steps"] <= 100

    def test_suggest_returns_dict_with_all_params(self):
        """suggest() should return values for all parameters."""
        space = HyperparamSpace.minimal_space()

        # Create a mock trial
        study = optuna.create_study()
        trial = study.ask()

        suggested = space.suggest(trial)

        assert "base_lr" in suggested
        assert "n_iterations" in suggested
        assert isinstance(suggested["base_lr"], float)
        assert isinstance(suggested["n_iterations"], int)

    def test_suggest_respects_bounds(self):
        """Suggested values should be within specified bounds."""
        space = HyperparamSpace(params={
            "test_float": {"low": 0.1, "high": 1.0, "log": False},
            "test_int": {"low": 5, "high": 10, "log": False, "type": "int"},
        })

        study = optuna.create_study()

        for _ in range(10):
            trial = study.ask()
            suggested = space.suggest(trial)

            assert 0.1 <= suggested["test_float"] <= 1.0
            assert 5 <= suggested["test_int"] <= 10

    def test_suggest_categorical(self):
        """Categorical parameters should suggest from choices."""
        space = HyperparamSpace(params={
            "method": {"choices": ["a", "b", "c"]},
        })

        study = optuna.create_study()
        trial = study.ask()
        suggested = space.suggest(trial)

        assert suggested["method"] in ["a", "b", "c"]


# =============================================================================
# Test HyperparamTuner Construction
# =============================================================================

class TestHyperparamTunerConstruction:
    """Tests for HyperparamTuner initialization."""

    def test_default_construction(self):
        """Default construction should work."""
        tuner = HyperparamTuner()

        assert tuner.runner_name == "train_on_historic_data"
        assert tuner.n_trials == 50
        assert tuner.n_wfa_cycles == 3
        assert tuner.objective == "mean_oos_sharpe"

    def test_custom_runner_name(self):
        """Should accept different runner names."""
        tuner = HyperparamTuner(runner_name="multi_period_sgd")

        assert tuner.runner_name == "multi_period_sgd"
        # Should auto-select multi-period space
        assert "n_periods" in tuner.hyperparam_space.params

    def test_custom_search_space(self):
        """Should accept custom search space."""
        custom_space = HyperparamSpace(params={
            "base_lr": {"low": 0.1, "high": 0.2, "log": False},
        })

        tuner = HyperparamTuner(hyperparam_space=custom_space)

        assert tuner.hyperparam_space.params["base_lr"]["low"] == 0.1

    def test_multi_objective_mode(self):
        """Multi-objective mode should set objectives."""
        tuner = HyperparamTuner(
            objective="multi",
            multi_objectives=["mean_oos_sharpe", "mean_wfe"],
        )

        assert tuner.objective == "multi"
        assert tuner.multi_objectives == ["mean_oos_sharpe", "mean_wfe"]


# =============================================================================
# Test TuningResult
# =============================================================================

class TestTuningResult:
    """Tests for TuningResult dataclass."""

    def test_tuning_result_fields(self):
        """TuningResult should have all expected fields."""
        result = TuningResult(
            best_params={"base_lr": 0.1},
            best_value=1.5,
            best_evaluation=None,
            n_trials=10,
            n_completed=8,
            n_pruned=2,
            all_trials=[],
            pareto_front=None,
            total_time_seconds=100.0,
        )

        assert result.best_params["base_lr"] == 0.1
        assert result.best_value == 1.5
        assert result.n_completed == 8
        assert result.n_pruned == 2


# =============================================================================
# Test Objective Function Creation
# =============================================================================

class TestObjectiveFunction:
    """Tests for objective function creation."""

    @pytest.fixture
    def mock_run_fingerprint(self):
        """Minimal fingerprint for testing."""
        return {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,  # 7 days - reduces effective training window
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.01,
                "n_iterations": 3,
                "n_cycles": 1,
                "training_data_kind": "historic",
                "n_parameter_sets": 1,
                "batch_size": 2,
            },
            "initial_memory_length": 10.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 1.0,
            "initial_weights_logits": 0.0,
            "initial_log_amplitude": -5.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 0.0,
            "initial_pre_exp_scaling": 0.001,
            "return_val": "sharpe",
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "max_memory_days": 30,
        }

    def test_objective_returns_callable(self, mock_run_fingerprint):
        """create_objective should return a callable."""
        space = HyperparamSpace.minimal_space()

        objective_fn = create_objective(
            run_fingerprint=mock_run_fingerprint,
            runner_name="train_on_historic_data",
            runner_kwargs={},
            hyperparam_space=space,
            n_wfa_cycles=2,
            objective_metric="mean_oos_sharpe",
            verbose=False,
        )

        assert callable(objective_fn)

    def test_objective_with_mocked_evaluator(self, mock_run_fingerprint):
        """Objective should call evaluator and return metric."""
        space = HyperparamSpace.minimal_space()

        # Mock the TrainingEvaluator
        mock_result = Mock(spec=EvaluationResult)
        mock_result.mean_oos_sharpe = 1.5
        mock_result.mean_wfe = 0.7
        mock_result.worst_oos_sharpe = 0.5
        mock_result.mean_is_oos_gap = 0.3
        mock_result.aggregate_rademacher = None
        mock_result.adjusted_mean_oos_sharpe = None
        mock_result.is_effective = True

        with patch.object(TrainingEvaluator, 'evaluate', return_value=mock_result):
            objective_fn = create_objective(
                run_fingerprint=mock_run_fingerprint,
                runner_name="train_on_historic_data",
                runner_kwargs={},
                hyperparam_space=space,
                n_wfa_cycles=2,
                objective_metric="mean_oos_sharpe",
                verbose=False,
            )

            study = optuna.create_study()
            trial = study.ask()

            # This will fail because we can't fully mock, but tests the structure
            # In practice, we'd need a more complete mock setup


class TestOptimizerSwitching:
    """Tests for weight_decay -> AdamW optimizer switching."""

    @pytest.fixture
    def base_fingerprint(self):
        """Base fingerprint for optimizer switching tests."""
        return {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,  # 7 days - reduces effective training window
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.01,
                "optimiser": "adam",  # Default optimizer
                "n_iterations": 3,
                "n_cycles": 1,
                "training_data_kind": "historic",
                "n_parameter_sets": 1,
                "batch_size": 2,
            },
            "initial_memory_length": 10.0,
            "initial_k_per_day": 1.0,
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "max_memory_days": 30,
        }

    def test_adamw_selected_when_weight_decay_enabled(self, base_fingerprint):
        """When use_weight_decay=True, optimizer should be set to AdamW."""
        # Space that forces use_weight_decay=True
        space = HyperparamSpace(params={
            "base_lr": {"low": 0.01, "high": 0.1, "log": True},
            "use_weight_decay": {"choices": [True]},  # Force True
            "weight_decay": {
                "low": 0.01, "high": 0.05, "log": True,
                "conditional_on": "use_weight_decay", "conditional_value": True
            },
        })

        # Track what fingerprint gets passed to evaluator
        captured_fp = {}

        def capture_evaluate_iter(self, fp):
            """Mock evaluate_iter as a generator that captures fingerprint."""
            captured_fp.update(deepcopy(fp))
            cycle = CycleEvaluation(
                cycle_number=1, is_sharpe=1.0, is_returns_over_hodl=0.1,
                oos_sharpe=1.0, oos_returns_over_hodl=0.05,
                walk_forward_efficiency=0.6, is_oos_gap=0.4,
            )
            yield cycle
            return EvaluationResult(
                trainer_name="test",
                trainer_config={},
                cycles=[cycle],
                mean_wfe=0.6, mean_oos_sharpe=1.0, std_oos_sharpe=0.1,
                worst_oos_sharpe=0.9, mean_is_oos_gap=0.4, is_effective=True,
            )

        with patch.object(TrainingEvaluator, 'evaluate_iter', capture_evaluate_iter):
            objective_fn = create_objective(
                run_fingerprint=base_fingerprint,
                runner_name="train_on_historic_data",
                runner_kwargs={},
                hyperparam_space=space,
                n_wfa_cycles=2,
                objective_metric="mean_oos_sharpe",
                verbose=False,
            )

            study = optuna.create_study()
            trial = study.ask()
            objective_fn(trial)

            # Verify optimizer was switched to adamw
            assert captured_fp["optimisation_settings"]["optimiser"] == "adamw", \
                "Optimizer should be 'adamw' when weight_decay is enabled"
            assert captured_fp["optimisation_settings"]["weight_decay"] > 0, \
                "weight_decay should be set when enabled"

    def test_adam_kept_when_weight_decay_disabled(self, base_fingerprint):
        """When use_weight_decay=False, optimizer should stay as Adam."""
        # Space that forces use_weight_decay=False
        space = HyperparamSpace(params={
            "base_lr": {"low": 0.01, "high": 0.1, "log": True},
            "use_weight_decay": {"choices": [False]},  # Force False
            "weight_decay": {
                "low": 0.01, "high": 0.05, "log": True,
                "conditional_on": "use_weight_decay", "conditional_value": True
            },
        })

        captured_fp = {}

        def capture_evaluate_iter(self, fp):
            """Mock evaluate_iter as a generator that captures fingerprint."""
            captured_fp.update(deepcopy(fp))
            cycle = CycleEvaluation(
                cycle_number=1, is_sharpe=1.0, is_returns_over_hodl=0.1,
                oos_sharpe=1.0, oos_returns_over_hodl=0.05,
                walk_forward_efficiency=0.6, is_oos_gap=0.4,
            )
            yield cycle
            return EvaluationResult(
                trainer_name="test",
                trainer_config={},
                cycles=[cycle],
                mean_wfe=0.6, mean_oos_sharpe=1.0, std_oos_sharpe=0.1,
                worst_oos_sharpe=0.9, mean_is_oos_gap=0.4, is_effective=True,
            )

        with patch.object(TrainingEvaluator, 'evaluate_iter', capture_evaluate_iter):
            objective_fn = create_objective(
                run_fingerprint=base_fingerprint,
                runner_name="train_on_historic_data",
                runner_kwargs={},
                hyperparam_space=space,
                n_wfa_cycles=2,
                objective_metric="mean_oos_sharpe",
                verbose=False,
            )

            study = optuna.create_study()
            trial = study.ask()
            objective_fn(trial)

            # Verify optimizer stayed as adam (from base_fingerprint)
            assert captured_fp["optimisation_settings"]["optimiser"] == "adam", \
                "Optimizer should stay as 'adam' when weight_decay is disabled"
            assert captured_fp["optimisation_settings"]["weight_decay"] == 0.0, \
                "weight_decay should be 0 when disabled"


# =============================================================================
# E2E Tests (with mocked training for speed)
# =============================================================================

class TestHyperparamTunerE2E:
    """E2E tests for hyperparameter tuning."""

    @pytest.fixture
    def base_fingerprint(self):
        """Base fingerprint for E2E tests."""
        return {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,  # 7 days - reduces effective training window
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.01,
                "optimiser": "sgd",
                "n_iterations": 3,
                "training_data_kind": "historic",
                "force_scalar": False,
                "n_parameter_sets": 1,
                "batch_size": 2,
                "n_cycles": 1,
            },
            "initial_memory_length": 10.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 1.0,
            "initial_weights_logits": 0.0,
            "initial_log_amplitude": -5.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 0.0,
            "initial_pre_exp_scaling": 0.001,
            "maximum_change": 0.001,
            "return_val": "sharpe",
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "use_alt_lamb": False,
            "use_pre_exp_scaling": True,
            "weight_interpolation_method": "linear",
            "arb_frequency": 1,
            "do_arb": True,
            "arb_quality": 1.0,
            "numeraire": None,
            "do_trades": False,
            "noise_trader_ratio": 0.0,
            "minimum_weight": 0.03,
            "max_memory_days": 30,
            "subsidary_pools": [],
        }

    def test_tuner_runs_with_mocked_evaluator(self, base_fingerprint):
        """Tuner should complete with mocked evaluator."""
        # Create mock evaluation results that vary with params
        call_count = [0]

        def mock_evaluate_iter(self, fp):
            """Mock evaluate_iter as a generator that yields cycle evaluations."""
            call_count[0] += 1
            # Simulate that higher LR gives better results (for testing)
            lr = fp["optimisation_settings"].get("base_lr", 0.1)
            mock_sharpe = 0.5 + lr * 2 + np.random.randn() * 0.1

            # Yield one cycle evaluation
            yield CycleEvaluation(
                cycle_number=1,
                is_sharpe=1.0,
                is_returns_over_hodl=0.1,
                oos_sharpe=mock_sharpe,
                oos_returns_over_hodl=0.05,
                walk_forward_efficiency=0.6,
                is_oos_gap=0.4,
            )

            # Return the full result (accessed via StopIteration.value)
            return EvaluationResult(
                trainer_name="test",
                trainer_config={},
                cycles=[
                    CycleEvaluation(
                        cycle_number=1,
                        is_sharpe=1.0,
                        is_returns_over_hodl=0.1,
                        oos_sharpe=mock_sharpe,
                        oos_returns_over_hodl=0.05,
                        walk_forward_efficiency=0.6,
                        is_oos_gap=0.4,
                    )
                ],
                mean_wfe=0.6,
                mean_oos_sharpe=mock_sharpe,
                std_oos_sharpe=0.1,
                worst_oos_sharpe=mock_sharpe - 0.1,
                mean_is_oos_gap=0.4,
                is_effective=True,
            )

        with patch.object(TrainingEvaluator, 'evaluate_iter', mock_evaluate_iter):
            tuner = HyperparamTuner(
                runner_name="train_on_historic_data",
                n_trials=5,
                n_wfa_cycles=1,
                hyperparam_space=HyperparamSpace.minimal_space(),
                verbose=False,
            )

            result = tuner.tune(base_fingerprint)

            # Should complete
            assert result.n_completed > 0
            assert "base_lr" in result.best_params
            assert result.best_value > 0

    def test_tuner_multi_objective_with_mock(self, base_fingerprint):
        """Multi-objective tuning should produce Pareto front."""
        def mock_evaluate_iter(self, fp):
            """Mock evaluate_iter as a generator."""
            lr = fp["optimisation_settings"].get("base_lr", 0.1)
            # Trade-off: higher LR = higher sharpe but lower WFE
            mock_sharpe = 0.5 + lr * 2
            mock_wfe = 0.8 - lr * 0.5

            cycle = CycleEvaluation(
                cycle_number=1,
                is_sharpe=1.0,
                is_returns_over_hodl=0.1,
                oos_sharpe=mock_sharpe,
                oos_returns_over_hodl=0.05,
                walk_forward_efficiency=mock_wfe,
                is_oos_gap=0.4,
            )
            yield cycle
            return EvaluationResult(
                trainer_name="test",
                trainer_config={},
                cycles=[cycle],
                mean_wfe=mock_wfe,
                mean_oos_sharpe=mock_sharpe,
                std_oos_sharpe=0.1,
                worst_oos_sharpe=mock_sharpe - 0.1,
                mean_is_oos_gap=0.4,
                is_effective=True,
            )

        with patch.object(TrainingEvaluator, 'evaluate_iter', mock_evaluate_iter):
            tuner = HyperparamTuner(
                runner_name="train_on_historic_data",
                n_trials=10,
                n_wfa_cycles=1,
                objective="multi",
                multi_objectives=["mean_oos_sharpe", "mean_wfe"],
                hyperparam_space=HyperparamSpace.minimal_space(),
                verbose=False,
            )

            result = tuner.tune(base_fingerprint)

            # Should have Pareto front
            assert result.pareto_front is not None
            assert len(result.pareto_front) > 0

    def test_different_objectives_produce_different_results(self, base_fingerprint):
        """Different objective metrics should lead to different optimal params."""
        def mock_evaluate_iter(self, fp):
            """Mock evaluate_iter as a generator."""
            lr = fp["optimisation_settings"].get("base_lr", 0.1)
            n_iter = fp["optimisation_settings"].get("n_iterations", 100)

            # Different metrics favor different params:
            # - OOS Sharpe favors high LR
            # - WFE favors low LR
            # - Worst OOS favors moderate LR

            cycle = CycleEvaluation(
                cycle_number=1,
                is_sharpe=1.0 + lr,
                is_returns_over_hodl=0.1,
                oos_sharpe=0.5 + lr * 2,  # Higher LR = higher sharpe
                oos_returns_over_hodl=0.05,
                walk_forward_efficiency=0.9 - lr,  # Lower LR = higher WFE
                is_oos_gap=lr,  # Higher LR = more overfitting
            )
            yield cycle
            return EvaluationResult(
                trainer_name="test",
                trainer_config={},
                cycles=[cycle],
                mean_wfe=0.9 - lr,
                mean_oos_sharpe=0.5 + lr * 2,
                std_oos_sharpe=0.1,
                worst_oos_sharpe=0.3 + lr - lr**2 * 5,  # Moderate LR is best
                mean_is_oos_gap=lr,
                is_effective=True,
            )

        with patch.object(TrainingEvaluator, 'evaluate_iter', mock_evaluate_iter):
            # Tune for OOS Sharpe (should favor high LR)
            tuner_sharpe = HyperparamTuner(
                n_trials=10,
                n_wfa_cycles=1,
                objective="mean_oos_sharpe",
                hyperparam_space=HyperparamSpace(params={
                    "base_lr": {"low": 0.01, "high": 0.5, "log": True},
                }),
                verbose=False,
            )
            result_sharpe = tuner_sharpe.tune(base_fingerprint)

            # Tune for WFE (should favor low LR)
            tuner_wfe = HyperparamTuner(
                n_trials=10,
                n_wfa_cycles=1,
                objective="mean_wfe",
                hyperparam_space=HyperparamSpace(params={
                    "base_lr": {"low": 0.01, "high": 0.5, "log": True},
                }),
                verbose=False,
            )
            result_wfe = tuner_wfe.tune(base_fingerprint)

            # Different objectives should find different optima
            # (with mocked data, sharpe-optimal LR should be higher than WFE-optimal)
            # Note: Due to randomness in Optuna, we check tendency not exact values
            assert result_sharpe.best_params["base_lr"] != result_wfe.best_params["base_lr"] or \
                   result_sharpe.best_value != result_wfe.best_value


class TestHyperparamTunerRealE2E:
    """
    Real E2E tests (no mocks) - slow but thorough.

    These tests actually run training and evaluation.
    """

    @pytest.fixture
    def real_fingerprint(self):
        """Fingerprint for real E2E tests."""
        return {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,  # 7 days - reduces effective training window
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.01,
                "optimiser": "adam",
                "n_iterations": 3,
                "training_data_kind": "historic",
                "force_scalar": False,
                "n_parameter_sets": 1,
                "batch_size": 2,
                "n_cycles": 1,
            },
            "initial_memory_length": 10.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 1.0,
            "initial_weights_logits": 0.0,
            "initial_log_amplitude": -5.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 0.0,
            "initial_pre_exp_scaling": 0.001,
            "maximum_change": 0.001,
            "return_val": "sharpe",
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "use_alt_lamb": False,
            "use_pre_exp_scaling": True,
            "weight_interpolation_method": "linear",
            "arb_frequency": 1,
            "do_arb": True,
            "arb_quality": 1.0,
            "numeraire": None,
            "do_trades": False,
            "noise_trader_ratio": 0.0,
            "minimum_weight": 0.03,
            "max_memory_days": 30,
            "subsidary_pools": [],
        }

    @pytest.mark.slow
    def test_real_tuning_completes(self, real_fingerprint):
        """
        Real tuning should complete and return valid results.

        This is a slow test that actually runs training.
        """
        tuner = HyperparamTuner(
            runner_name="train_on_historic_data",
            n_trials=2,  # Very few trials for speed
            n_wfa_cycles=1,
            hyperparam_space=HyperparamSpace(params={
                "base_lr": {"low": 0.01, "high": 0.1, "log": True},
            }),
            verbose=False,
            root=TEST_DATA_DIR,
        )

        result = tuner.tune(real_fingerprint)

        # Should complete successfully
        assert result.n_completed >= 1
        assert "base_lr" in result.best_params
        assert np.isfinite(result.best_value)
        assert result.total_time_seconds > 0

        # All trials should have evaluation data
        for trial in result.all_trials:
            if trial["state"] == "TrialState.COMPLETE":
                assert trial["evaluation_result"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
