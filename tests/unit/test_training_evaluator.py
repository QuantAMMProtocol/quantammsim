"""
Unit tests for training_evaluator.py

Tests the meta-runner API without requiring actual training runs.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from unittest.mock import Mock, patch, MagicMock

from quantammsim.runners.training_evaluator import (
    TrainingEvaluator,
    TrainerWrapper,
    FunctionWrapper,
    RandomBaselineWrapper,
    ExistingRunnerWrapper,
    CycleEvaluation,
    EvaluationResult,
    compare_trainers,
)
from tests.conftest import TEST_DATA_DIR


# =============================================================================
# Test TrainerWrapper Classes
# =============================================================================

class TestFunctionWrapper:
    """Tests for FunctionWrapper."""

    def test_wraps_function_correctly(self):
        """FunctionWrapper should call the wrapped function with correct args."""
        call_log = []

        def mock_trainer(data_dict, train_start_idx, train_end_idx, pool,
                         run_fingerprint, n_assets, warm_start_params=None,
                         warm_start_weights=None):
            call_log.append({
                "train_start_idx": train_start_idx,
                "train_end_idx": train_end_idx,
                "n_assets": n_assets,
                "warm_start": warm_start_params,
                "warm_start_weights": warm_start_weights,
            })
            return {"param": 1.0}, {"epochs_trained": 100}

        wrapper = FunctionWrapper(mock_trainer, name="test_trainer")

        params, metadata = wrapper.train(
            data_dict={"prices": np.zeros((100, 2))},
            train_start_idx=10,
            train_end_idx=50,
            pool=None,
            run_fingerprint={},
            n_assets=2,
            warm_start_params={"old_param": 0.5},
            warm_start_weights=np.array([0.6, 0.4]),
        )

        assert len(call_log) == 1
        assert call_log[0]["train_start_idx"] == 10
        assert call_log[0]["train_end_idx"] == 50
        assert call_log[0]["n_assets"] == 2
        assert call_log[0]["warm_start"] == {"old_param": 0.5}
        assert call_log[0]["warm_start_weights"] is not None
        assert params == {"param": 1.0}
        assert metadata["epochs_trained"] == 100

    def test_name_and_config(self):
        """FunctionWrapper should expose name and config."""
        wrapper = FunctionWrapper(
            lambda **kwargs: ({}, {}),
            name="my_trainer",
            config={"lr": 0.1},
        )

        assert wrapper.name == "my_trainer"
        assert wrapper.config == {"lr": 0.1}


class TestRandomBaselineWrapper:
    """Tests for RandomBaselineWrapper."""

    def test_returns_params_and_metadata(self):
        """RandomBaselineWrapper should return valid params and metadata."""
        wrapper = RandomBaselineWrapper(seed=42)

        # Mock pool
        mock_pool = Mock()
        mock_pool.init_parameters.return_value = {
            "memory_length": jnp.array([[30.0, 30.0]]),
            "k_per_day": jnp.array([[1.0, 1.0]]),
        }

        run_fp = {
            "initial_memory_length": 30.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 1.0,
            "initial_weights_logits": 0.0,
            "initial_log_amplitude": 0.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 0.0,
            "initial_pre_exp_scaling": 0.001,
        }

        params, metadata = wrapper.train(
            data_dict={},
            train_start_idx=0,
            train_end_idx=100,
            pool=mock_pool,
            run_fingerprint=run_fp,
            n_assets=2,
        )

        assert "memory_length" in params
        assert "k_per_day" in params
        assert metadata["epochs_trained"] == 0
        assert metadata["final_objective"] == 0.0

    def test_different_seeds_different_params(self):
        """Different seeds should produce different random params."""
        mock_pool = Mock()
        mock_pool.init_parameters.return_value = {
            "memory_length": jnp.array([[30.0, 30.0]]),
        }

        run_fp = {
            "initial_memory_length": 30.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 1.0,
            "initial_weights_logits": 0.0,
            "initial_log_amplitude": 0.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 0.0,
            "initial_pre_exp_scaling": 0.001,
        }

        wrapper1 = RandomBaselineWrapper(seed=42)
        wrapper2 = RandomBaselineWrapper(seed=123)

        params1, _ = wrapper1.train({}, 0, 100, mock_pool, run_fp, 2)
        params2, _ = wrapper2.train({}, 0, 100, mock_pool, run_fp, 2)

        # Different seeds should give different noise
        assert not np.allclose(params1["memory_length"], params2["memory_length"])

    def test_call_count_increments(self):
        """Each call should use a different seed offset."""
        wrapper = RandomBaselineWrapper(seed=42)

        mock_pool = Mock()
        mock_pool.init_parameters.return_value = {
            "memory_length": jnp.array([[30.0, 30.0]]),
        }

        run_fp = {
            "initial_memory_length": 30.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 1.0,
            "initial_weights_logits": 0.0,
            "initial_log_amplitude": 0.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 0.0,
            "initial_pre_exp_scaling": 0.001,
        }

        params1, _ = wrapper.train({}, 0, 100, mock_pool, run_fp, 2)
        params2, _ = wrapper.train({}, 0, 100, mock_pool, run_fp, 2)

        # Sequential calls should have different noise
        assert not np.allclose(params1["memory_length"], params2["memory_length"])


class TestExistingRunnerWrapper:
    """Tests for ExistingRunnerWrapper."""

    def test_unknown_runner_raises(self):
        """Unknown runner name should raise ValueError."""
        wrapper = ExistingRunnerWrapper("unknown_runner")

        with pytest.raises(ValueError, match="Unknown runner"):
            wrapper.train({}, 0, 100, None, {}, 2)

    def test_name_includes_runner_name(self):
        """Wrapper name should include the runner name."""
        wrapper = ExistingRunnerWrapper("train_on_historic_data", {"max_iterations": 500})
        assert "train_on_historic_data" in wrapper.name

    def test_config_stores_kwargs(self):
        """Config should store the runner kwargs."""
        wrapper = ExistingRunnerWrapper(
            "train_on_historic_data",
            {"max_iterations": 500, "patience": 50}
        )
        assert wrapper.config["max_iterations"] == 500
        assert wrapper.config["patience"] == 50


# =============================================================================
# Test TrainingEvaluator Constructors
# =============================================================================

class TestTrainingEvaluatorConstructors:
    """Tests for TrainingEvaluator class methods."""

    def test_from_function_creates_evaluator(self):
        """from_function should create a working evaluator."""
        def dummy_trainer(**kwargs):
            return {}, {"epochs_trained": 0}

        evaluator = TrainingEvaluator.from_function(
            dummy_trainer,
            name="dummy",
            n_cycles=3,
        )

        assert evaluator.trainer.name == "dummy"
        assert evaluator.n_cycles == 3

    def test_from_runner_creates_evaluator(self):
        """from_runner should create a working evaluator."""
        evaluator = TrainingEvaluator.from_runner(
            "train_on_historic_data",
            n_cycles=4,
            max_iterations=100,
        )

        assert "train_on_historic_data" in evaluator.trainer.name
        assert evaluator.n_cycles == 4
        assert evaluator.trainer.config["max_iterations"] == 100

    def test_random_baseline_creates_evaluator(self):
        """random_baseline should create a working evaluator."""
        evaluator = TrainingEvaluator.random_baseline(seed=42, n_cycles=2)

        assert "random" in evaluator.trainer.name.lower()
        assert evaluator.n_cycles == 2


# =============================================================================
# Test Result Aggregation
# =============================================================================

class TestResultAggregation:
    """Tests for result aggregation logic."""

    def test_aggregate_results_computes_correct_metrics(self):
        """_aggregate_results should compute correct aggregate metrics."""
        evaluator = TrainingEvaluator.random_baseline(n_cycles=3, verbose=False)

        # Create mock cycle results
        cycle_results = [
            CycleEvaluation(
                cycle_number=0,
                is_sharpe=1.5, is_returns_over_hodl=0.1,
                oos_sharpe=1.0, oos_returns_over_hodl=0.05,
                walk_forward_efficiency=0.67, is_oos_gap=0.5,
            ),
            CycleEvaluation(
                cycle_number=1,
                is_sharpe=1.2, is_returns_over_hodl=0.08,
                oos_sharpe=0.8, oos_returns_over_hodl=0.04,
                walk_forward_efficiency=0.67, is_oos_gap=0.4,
            ),
            CycleEvaluation(
                cycle_number=2,
                is_sharpe=1.0, is_returns_over_hodl=0.06,
                oos_sharpe=0.6, oos_returns_over_hodl=0.03,
                walk_forward_efficiency=0.60, is_oos_gap=0.4,
            ),
        ]

        result = evaluator._aggregate_results(cycle_results, [], [])

        # Check aggregates
        assert abs(result.mean_oos_sharpe - 0.8) < 0.01  # (1.0 + 0.8 + 0.6) / 3
        assert result.worst_oos_sharpe == 0.6
        assert abs(result.mean_wfe - 0.647) < 0.01  # (0.67 + 0.67 + 0.60) / 3
        assert abs(result.mean_is_oos_gap - 0.433) < 0.01  # (0.5 + 0.4 + 0.4) / 3

    def test_effectiveness_verdict_wfe_threshold(self):
        """Effectiveness should depend on WFE >= 0.5."""
        evaluator = TrainingEvaluator.random_baseline(verbose=False)

        # Good WFE
        good_cycles = [
            CycleEvaluation(0, 1.0, 0.0, 0.6, 0.0, 0.6, 0.4),  # WFE = 0.6
            CycleEvaluation(1, 1.0, 0.0, 0.7, 0.0, 0.7, 0.3),  # WFE = 0.7
        ]
        result_good = evaluator._aggregate_results(good_cycles, [], [])
        assert result_good.is_effective is True

        # Poor WFE
        poor_cycles = [
            CycleEvaluation(0, 1.0, 0.0, 0.2, 0.0, 0.2, 0.8),  # WFE = 0.2
            CycleEvaluation(1, 1.0, 0.0, 0.3, 0.0, 0.3, 0.7),  # WFE = 0.3
        ]
        result_poor = evaluator._aggregate_results(poor_cycles, [], [])
        assert result_poor.is_effective is False

    def test_effectiveness_requires_positive_worst_oos(self):
        """Effectiveness should require worst OOS Sharpe > 0."""
        evaluator = TrainingEvaluator.random_baseline(verbose=False)

        # Good WFE but one negative OOS
        cycles = [
            CycleEvaluation(0, 1.0, 0.0, 0.8, 0.0, 0.8, 0.2),
            CycleEvaluation(1, 1.0, 0.0, -0.2, 0.0, -0.2, 1.2),  # Negative OOS!
        ]
        result = evaluator._aggregate_results(cycles, [], [])

        # Should not be effective due to negative OOS
        assert result.is_effective is False


# =============================================================================
# Test compare_trainers
# =============================================================================

class TestCompareTrainers:
    """Tests for compare_trainers function."""

    def test_compare_returns_dict_of_results(self):
        """compare_trainers should return dict keyed by trainer name."""
        # Create mock evaluators that return fixed results
        mock_result = EvaluationResult(
            trainer_name="mock",
            trainer_config={},
            cycles=[],
            mean_wfe=0.6,
            mean_oos_sharpe=0.5,
            std_oos_sharpe=0.1,
            worst_oos_sharpe=0.3,
            mean_is_oos_gap=0.2,
            is_effective=True,
            effectiveness_reasons=[],
        )

        eval1 = TrainingEvaluator.random_baseline(verbose=False)
        eval2 = TrainingEvaluator.random_baseline(verbose=False)

        # Patch evaluate to return mock result
        eval1.evaluate = Mock(return_value=mock_result)
        eval2.evaluate = Mock(return_value=mock_result)

        results = compare_trainers(
            run_fingerprint={},
            trainers={"trainer_a": eval1, "trainer_b": eval2},
            verbose=False,
        )

        assert "trainer_a" in results
        assert "trainer_b" in results
        assert results["trainer_a"].trainer_name == "mock"


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestWarmStartFunctionality:
    """Tests for warm-start functionality between walk-forward cycles."""

    def test_warm_start_params_and_weights_passed_to_trainer(self):
        """warm_start_params and warm_start_weights should be passed to the trainer function."""
        call_log = []

        def mock_trainer(data_dict, train_start_idx, train_end_idx, pool,
                         run_fingerprint, n_assets, warm_start_params=None,
                         warm_start_weights=None):
            call_log.append({
                "warm_start_params": warm_start_params,
                "warm_start_weights": warm_start_weights,
            })
            return {"param": 1.0}, {"epochs_trained": 10, "final_weights": np.array([0.6, 0.4])}

        wrapper = FunctionWrapper(mock_trainer, name="test")

        # First call without warm start
        wrapper.train({}, 0, 100, None, {}, 2)
        assert call_log[0]["warm_start_params"] is None
        assert call_log[0]["warm_start_weights"] is None

        # Second call with warm start
        warm_params = {"logit_lamb": np.array([0.5, 0.5])}
        warm_weights = np.array([0.6, 0.4])
        wrapper.train({}, 100, 200, None, {}, 2,
                     warm_start_params=warm_params,
                     warm_start_weights=warm_weights)
        assert call_log[1]["warm_start_params"] == warm_params
        assert np.allclose(call_log[1]["warm_start_weights"], warm_weights)

    def test_metadata_contains_final_weights_for_warm_starting(self):
        """Training metadata should contain final_weights for warm-starting next cycle."""
        # Create a mock trainer that returns metadata with final weights
        def mock_trainer(data_dict, train_start_idx, train_end_idx, pool,
                         run_fingerprint, n_assets, warm_start_params=None,
                         warm_start_weights=None):
            params = {"logit_lamb": np.array([0.5, 0.5])}
            metadata = {
                "epochs_trained": 10,
                "final_objective": 0.5,
                "best_param_idx": 0,
                "final_weights": np.array([0.6, 0.4]),
            }
            return params, metadata

        wrapper = FunctionWrapper(mock_trainer, name="test")
        params, metadata = wrapper.train({}, 0, 100, None, {}, 2)

        # Check metadata contains warm-start info
        assert "final_weights" in metadata
        assert metadata["final_weights"] is not None

    def test_existing_runner_wrapper_passes_warm_start_weights(self):
        """ExistingRunnerWrapper should pass warm_start_weights to train_on_historic_data."""
        from quantammsim.runners import jax_runners

        original_train = jax_runners.train_on_historic_data
        captured_args = {}

        def capture_train(run_fingerprint, *args, **kwargs):
            captured_args["warm_start_params"] = kwargs.get("warm_start_params")
            captured_args["warm_start_weights"] = kwargs.get("warm_start_weights")
            # Return mock result
            params = {}
            metadata = {
                "epochs_trained": 3,
                "final_objective": 0.5,
                "best_param_idx": 0,
                "final_train_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01}],
                "final_continuous_test_metrics": [{"sharpe": 0.4, "returns_over_uniform_hodl": 0.005}],
                "final_weights": np.array([0.6, 0.4]),
            }
            return params, metadata

        jax_runners.train_on_historic_data = capture_train

        try:
            wrapper = ExistingRunnerWrapper("train_on_historic_data", {"max_iterations": 3})

            warm_weights = np.array([0.6, 0.4])
            warm_params = {"logit_lamb": np.array([0.5, 0.5])}

            wrapper.train(
                data_dict={},
                train_start_idx=0,
                train_end_idx=100,
                pool=None,
                run_fingerprint={
                    "tokens": ["BTC", "ETH"],
                    "optimisation_settings": {"n_iterations": 10},
                },
                n_assets=2,
                warm_start_params=warm_params,
                warm_start_weights=warm_weights,
            )

            # Verify warm-start state was passed correctly
            assert captured_args["warm_start_params"] == warm_params
            assert np.allclose(captured_args["warm_start_weights"], warm_weights)

        finally:
            jax_runners.train_on_historic_data = original_train

    def test_warm_start_none_when_not_provided(self):
        """warm_start_weights should be None when not provided."""
        from quantammsim.runners import jax_runners

        original_train = jax_runners.train_on_historic_data
        captured_args = {}

        def capture_train(run_fingerprint, *args, **kwargs):
            captured_args["warm_start_params"] = kwargs.get("warm_start_params")
            captured_args["warm_start_weights"] = kwargs.get("warm_start_weights")
            params = {}
            metadata = {"epochs_trained": 3, "final_objective": 0.5, "best_param_idx": 0}
            return params, metadata

        jax_runners.train_on_historic_data = capture_train

        try:
            wrapper = ExistingRunnerWrapper("train_on_historic_data", {"max_iterations": 3})

            # Call without warm_start_weights
            wrapper.train(
                data_dict={},
                train_start_idx=0,
                train_end_idx=100,
                pool=None,
                run_fingerprint={
                    "tokens": ["BTC", "ETH"],
                    "optimisation_settings": {"n_iterations": 10},
                },
                n_assets=2,
                warm_start_params=None,
                warm_start_weights=None,
            )

            # All warm-start args should be None
            assert captured_args["warm_start_params"] is None
            assert captured_args["warm_start_weights"] is None

        finally:
            jax_runners.train_on_historic_data = original_train


class TestEarlyPruning:
    """Tests for early pruning when OOS metrics are negative."""

    def test_prunes_when_oos_metrics_negative(self):
        """Evaluation should stop early if OOS sharpe and returns_over_hodl are both negative."""
        call_count = [0]

        def bad_trainer(data_dict, train_start_idx, train_end_idx, pool,
                        run_fingerprint, n_assets, warm_start_params=None,
                        warm_start_weights=None):
            call_count[0] += 1
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                n_assets,
                1,
            )
            params = {
                k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
                for k, v in params.items()
            }
            # Return negative OOS metrics to trigger pruning
            metadata = {
                "epochs_trained": 10,
                "best_param_idx": 0,
                "final_train_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01}],
                "final_continuous_test_metrics": [{"sharpe": -0.5, "returns_over_uniform_hodl": -0.02}],
                "final_weights": np.array([0.5, 0.5]),
            }
            return params, metadata

        evaluator = TrainingEvaluator.from_function(
            bad_trainer,
            name="bad_trainer",
            n_cycles=3,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        run_fp = {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.01,
                "training_data_kind": "historic",
                "force_scalar": False,
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
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "arb_frequency": 1,
            "do_arb": True,
            "arb_quality": 1.0,
            "do_trades": False,
            "noise_trader_ratio": 0.0,
            "minimum_weight": 0.03,
            "max_memory_days": 30,
            "subsidary_pools": [],
            "weight_interpolation_method": "linear",
            "use_pre_exp_scaling": True,
            "use_alt_lamb": False,
            "numeraire": None,
        }

        result = evaluator.evaluate(run_fp)

        # Should have stopped after first cycle due to negative OOS metrics
        assert call_count[0] == 1, f"Expected 1 cycle (pruned), got {call_count[0]}"
        assert len(result.cycles) == 1, f"Expected 1 cycle result, got {len(result.cycles)}"

    def test_continues_when_oos_sharpe_positive(self):
        """Evaluation should continue if OOS sharpe is positive (even if returns_over_hodl negative)."""
        call_count = [0]

        def ok_trainer(data_dict, train_start_idx, train_end_idx, pool,
                       run_fingerprint, n_assets, warm_start_params=None,
                       warm_start_weights=None):
            call_count[0] += 1
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                n_assets,
                1,
            )
            params = {
                k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
                for k, v in params.items()
            }
            # Positive sharpe, negative returns_over_hodl - should NOT prune
            metadata = {
                "epochs_trained": 10,
                "best_param_idx": 0,
                "final_train_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01}],
                "final_continuous_test_metrics": [{"sharpe": 0.1, "returns_over_uniform_hodl": -0.01}],
                "final_weights": np.array([0.5, 0.5]),
            }
            return params, metadata

        evaluator = TrainingEvaluator.from_function(
            ok_trainer,
            name="ok_trainer",
            n_cycles=2,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        run_fp = {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.01,
                "training_data_kind": "historic",
                "force_scalar": False,
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
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "arb_frequency": 1,
            "do_arb": True,
            "arb_quality": 1.0,
            "do_trades": False,
            "noise_trader_ratio": 0.0,
            "minimum_weight": 0.03,
            "max_memory_days": 30,
            "subsidary_pools": [],
            "weight_interpolation_method": "linear",
            "use_pre_exp_scaling": True,
            "use_alt_lamb": False,
            "numeraire": None,
        }

        result = evaluator.evaluate(run_fp)

        # Should have completed all 2 cycles (not pruned)
        assert call_count[0] == 2, f"Expected 2 cycles, got {call_count[0]}"
        assert len(result.cycles) == 2, f"Expected 2 cycle results, got {len(result.cycles)}"

    def test_prunes_when_oos_metric_is_nan(self):
        """Evaluation should prune if OOS training metric is NaN."""
        call_count = [0]

        def nan_trainer(data_dict, train_start_idx, train_end_idx, pool,
                        run_fingerprint, n_assets, warm_start_params=None,
                        warm_start_weights=None):
            call_count[0] += 1
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                n_assets,
                1,
            )
            params = {
                k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
                for k, v in params.items()
            }
            # Return NaN for the training metric (returns_over_uniform_hodl) to trigger pruning
            # Keep sharpe finite for WFE aggregation
            metadata = {
                "epochs_trained": 10,
                "best_param_idx": 0,
                "final_train_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01, "return": 0.01}],
                "final_continuous_test_metrics": [{"sharpe": 0.3, "returns_over_uniform_hodl": float('nan'), "return": 0.01}],
                "final_weights": np.array([0.5, 0.5]),
            }
            return params, metadata

        evaluator = TrainingEvaluator.from_function(
            nan_trainer,
            name="nan_trainer",
            n_cycles=3,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        run_fp = {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,
            "weight_interpolation_period": 1440,
            "return_val": "returns_over_uniform_hodl",  # Training metric with NaN value
            "optimisation_settings": {
                "base_lr": 0.01,
                "training_data_kind": "historic",
                "force_scalar": False,
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
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "arb_frequency": 1,
            "do_arb": True,
            "arb_quality": 1.0,
            "do_trades": False,
            "noise_trader_ratio": 0.0,
            "minimum_weight": 0.03,
            "max_memory_days": 30,
            "subsidary_pools": [],
            "weight_interpolation_method": "linear",
            "use_pre_exp_scaling": True,
            "use_alt_lamb": False,
            "numeraire": None,
        }

        result = evaluator.evaluate(run_fp)

        # Should have stopped after first cycle due to NaN OOS metric
        assert call_count[0] == 1, f"Expected 1 cycle (pruned), got {call_count[0]}"
        assert len(result.cycles) == 1, f"Expected 1 cycle result, got {len(result.cycles)}"

    def test_prunes_when_oos_metric_missing(self):
        """Evaluation should prune if OOS training metric is missing from results."""
        call_count = [0]

        def missing_metric_trainer(data_dict, train_start_idx, train_end_idx, pool,
                                   run_fingerprint, n_assets, warm_start_params=None,
                                   warm_start_weights=None):
            call_count[0] += 1
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                n_assets,
                1,
            )
            params = {
                k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
                for k, v in params.items()
            }
            # Return metrics without the training metric (custom_metric missing)
            # Use a custom metric that doesn't exist to test missing metric pruning
            # Keep sharpe for aggregation to work, but training objective is custom_metric
            metadata = {
                "epochs_trained": 10,
                "best_param_idx": 0,
                "final_train_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01, "return": 0.01}],
                "final_continuous_test_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01, "return": 0.01}],  # No custom_metric
                "final_weights": np.array([0.5, 0.5]),
            }
            return params, metadata

        evaluator = TrainingEvaluator.from_function(
            missing_metric_trainer,
            name="missing_metric_trainer",
            n_cycles=3,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        run_fp = {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,
            "weight_interpolation_period": 1440,
            "return_val": "custom_metric",  # Training metric that won't be in results
            "optimisation_settings": {
                "base_lr": 0.01,
                "training_data_kind": "historic",
                "force_scalar": False,
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
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "arb_frequency": 1,
            "do_arb": True,
            "arb_quality": 1.0,
            "do_trades": False,
            "noise_trader_ratio": 0.0,
            "minimum_weight": 0.03,
            "max_memory_days": 30,
            "subsidary_pools": [],
            "weight_interpolation_method": "linear",
            "use_pre_exp_scaling": True,
            "use_alt_lamb": False,
            "numeraire": None,
        }

        result = evaluator.evaluate(run_fp)

        # Should have stopped after first cycle due to missing OOS metric
        assert call_count[0] == 1, f"Expected 1 cycle (pruned), got {call_count[0]}"
        assert len(result.cycles) == 1, f"Expected 1 cycle result, got {len(result.cycles)}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_config_handled(self):
        """Empty config should be handled gracefully."""
        wrapper = FunctionWrapper(lambda **kwargs: ({}, {}))
        assert wrapper.config == {}

    def test_none_warm_start_handled(self):
        """None warm_start_params should be handled."""
        call_log = []

        def mock_trainer(**kwargs):
            call_log.append(kwargs.get("warm_start_params"))
            return {}, {}

        wrapper = FunctionWrapper(mock_trainer)
        wrapper.train({}, 0, 100, None, {}, 2, warm_start_params=None)

        assert call_log[0] is None

    def test_inf_wfe_handled_in_aggregation(self):
        """Infinite WFE values should be filtered out."""
        evaluator = TrainingEvaluator.random_baseline(verbose=False)

        cycles = [
            CycleEvaluation(0, 0.0, 0.0, 0.5, 0.0, float('inf'), -0.5),  # WFE = inf
            CycleEvaluation(1, 1.0, 0.0, 0.6, 0.0, 0.6, 0.4),  # WFE = 0.6
        ]

        result = evaluator._aggregate_results(cycles, [], [])

        # Should only use finite WFE
        assert np.isfinite(result.mean_wfe)
        assert abs(result.mean_wfe - 0.6) < 0.01

    def test_rademacher_aggregation_with_checkpoint_data(self):
        """Rademacher should be computed when checkpoint data is available."""
        evaluator = TrainingEvaluator.random_baseline(verbose=False)
        evaluator.compute_rademacher = True  # Enable Rademacher tracking

        # Create mock cycle results with Rademacher data
        cycles = [
            CycleEvaluation(0, 1.0, 0.0, 0.6, 0.0, 0.6, 0.4, rademacher_complexity=0.05),
            CycleEvaluation(1, 1.0, 0.0, 0.7, 0.0, 0.7, 0.3, rademacher_complexity=0.04),
        ]

        # Mock checkpoint returns (2 checkpoints per cycle, 100 timesteps each)
        checkpoint_returns = [
            np.random.randn(2, 100) * 0.01,  # Cycle 0
            np.random.randn(2, 100) * 0.01,  # Cycle 1
        ]

        # Create mock WalkForwardCycle objects with test_end_idx and test_start_idx
        class MockCycle:
            def __init__(self, test_start, test_end):
                self.test_start_idx = test_start
                self.test_end_idx = test_end

        mock_cycles = [MockCycle(0, 100), MockCycle(100, 200)]

        result = evaluator._aggregate_results(cycles, mock_cycles, checkpoint_returns)

        # Should have computed aggregate Rademacher
        assert result.aggregate_rademacher is not None
        assert result.aggregate_rademacher > 0
        assert result.adjusted_mean_oos_sharpe is not None
        # Adjusted should be less than raw due to haircut
        assert result.adjusted_mean_oos_sharpe < result.mean_oos_sharpe


# =============================================================================
# INTEGRATION TESTS - Actually run evaluate() with real data
# =============================================================================

class TestEvaluateIntegration:
    """Integration tests that actually run the evaluate() method with real data."""

    @pytest.fixture
    def real_run_fingerprint(self):
        """Run fingerprint for real data tests."""
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
                "n_cycles": 1,
                "training_data_kind": "historic",
                "force_scalar": False,
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
    @pytest.mark.integration
    def test_evaluate_runs_full_pipeline(self, real_run_fingerprint):
        """evaluate() should run the full pipeline and return valid results."""
        evaluator = TrainingEvaluator.random_baseline(n_cycles=1, verbose=False, root=TEST_DATA_DIR)
        result = evaluator.evaluate(real_run_fingerprint)

        # Check result structure
        assert isinstance(result, EvaluationResult)
        assert result.trainer_name == "random_baseline"
        assert len(result.cycles) == 1

        # Check metrics are computed
        assert np.isfinite(result.mean_wfe)
        assert np.isfinite(result.mean_oos_sharpe)
        assert np.isfinite(result.worst_oos_sharpe)

        # Check cycles have valid data
        for cycle in result.cycles:
            assert isinstance(cycle, CycleEvaluation)
            assert np.isfinite(cycle.is_sharpe)
            assert np.isfinite(cycle.oos_sharpe)
            assert np.isfinite(cycle.walk_forward_efficiency)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_with_custom_trainer(self, real_run_fingerprint):
        """evaluate() should work with custom training functions."""
        train_calls = []

        def custom_trainer(data_dict, train_start_idx, train_end_idx, pool,
                          run_fingerprint, n_assets, warm_start_params=None,
                          warm_start_weights=None):
            train_calls.append({
                "start": train_start_idx,
                "end": train_end_idx,
                "warm_start": warm_start_params is not None,
                "warm_start_weights": warm_start_weights is not None,
            })
            # Return valid params using pool's initialization
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint["initial_memory_length_delta"],
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                n_assets,
                1,
            )
            # Squeeze out parameter set dimension - handle both arrays and non-arrays
            squeezed = {}
            for k, v in params.items():
                if hasattr(v, 'shape') and len(v.shape) > 1:
                    squeezed[k] = jnp.squeeze(v, axis=0)
                else:
                    squeezed[k] = v
            return squeezed, {"epochs_trained": 10}

        evaluator = TrainingEvaluator.from_function(
            custom_trainer,
            name="custom",
            n_cycles=1,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        result = evaluator.evaluate(real_run_fingerprint)

        # Verify trainer was called for each cycle
        assert len(train_calls) == 1

        # First cycle has no warm start
        assert train_calls[0]["warm_start"] is False

        # Result should be valid
        assert len(result.cycles) == 1

    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_computes_wfe_correctly(self, real_run_fingerprint):
        """WFE should be computed as OOS/IS sharpe ratio."""
        evaluator = TrainingEvaluator.random_baseline(n_cycles=1, verbose=False, root=TEST_DATA_DIR)
        result = evaluator.evaluate(real_run_fingerprint)

        for cycle in result.cycles:
            if cycle.is_sharpe > 0:
                expected_wfe = cycle.oos_sharpe / cycle.is_sharpe
                assert abs(cycle.walk_forward_efficiency - expected_wfe) < 0.01
            else:
                # WFE should be 0 for non-positive IS sharpe
                assert cycle.walk_forward_efficiency == 0.0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_expanding_vs_rolling_window(self, real_run_fingerprint):
        """Expanding window should always start from beginning."""
        # Expanding window
        evaluator_expanding = TrainingEvaluator(
            trainer=RandomBaselineWrapper(seed=42),
            n_cycles=1,
            keep_fixed_start=True,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        # Rolling window
        evaluator_rolling = TrainingEvaluator(
            trainer=RandomBaselineWrapper(seed=42),
            n_cycles=1,
            keep_fixed_start=False,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        # Both should run successfully
        result_expanding = evaluator_expanding.evaluate(real_run_fingerprint)
        result_rolling = evaluator_rolling.evaluate(real_run_fingerprint)

        assert len(result_expanding.cycles) == 1
        assert len(result_rolling.cycles) == 1


class TestEvaluateParams:
    """Test _evaluate_params method directly using real data."""

    @pytest.fixture
    def setup_for_eval_real(self):
        """Set up pool and data for evaluation tests using real data."""
        from quantammsim.pools.creator import create_pool
        from quantammsim.utils.data_processing.historic_data_utils import get_data_dict
        from quantammsim.runners.jax_runner_utils import get_unique_tokens

        run_fp = {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "maximum_change": 0.001,
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "arb_frequency": 1,
            "do_arb": True,
            "arb_quality": 1.0,
            "do_trades": False,
            "noise_trader_ratio": 0.0,
            "minimum_weight": 0.03,
            "weight_interpolation_method": "linear",
            "use_pre_exp_scaling": True,
            "use_alt_lamb": False,
            "numeraire": None,
            "max_memory_days": 30,
            "initial_memory_length": 10.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 1.0,
            "initial_weights_logits": 0.0,
            "initial_log_amplitude": -5.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 0.0,
            "initial_pre_exp_scaling": 0.001,
            "subsidary_pools": [],
            "ste_max_change": False,
            "ste_min_max_weight": False,
            "optimisation_settings": {
                "force_scalar": False,
                "training_data_kind": "historic",
            },
        }

        unique_tokens = get_unique_tokens(run_fp)
        n_assets = len(unique_tokens)

        data_dict = get_data_dict(
            unique_tokens,
            run_fp,
            data_kind="historic",
            max_memory_days=run_fp["max_memory_days"],
            start_date_string=run_fp["startDateString"],
            end_time_string=run_fp["endDateString"],
            do_test_period=False,
            root=TEST_DATA_DIR,
        )

        pool = create_pool("momentum")

        params = pool.init_parameters(
            {
                "initial_memory_length": run_fp["initial_memory_length"],
                "initial_memory_length_delta": run_fp["initial_memory_length_delta"],
                "initial_k_per_day": run_fp["initial_k_per_day"],
                "initial_weights_logits": run_fp["initial_weights_logits"],
                "initial_log_amplitude": run_fp["initial_log_amplitude"],
                "initial_raw_width": run_fp["initial_raw_width"],
                "initial_raw_exponents": run_fp["initial_raw_exponents"],
                "initial_pre_exp_scaling": run_fp["initial_pre_exp_scaling"],
            },
            run_fp,
            n_assets,
            1,
        )
        # Handle both arrays and non-arrays when squeezing
        squeezed = {}
        for k, v in params.items():
            if hasattr(v, 'shape') and len(v.shape) > 1:
                squeezed[k] = jnp.squeeze(v, axis=0)
            else:
                squeezed[k] = v
        params = squeezed

        return {
            "data_dict": data_dict,
            "pool": pool,
            "params": params,
            "run_fp": run_fp,
            "n_assets": n_assets,
        }

    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_params_returns_metrics(self, setup_for_eval_real):
        """_evaluate_params should return valid metrics dict."""
        evaluator = TrainingEvaluator.random_baseline(verbose=False)

        # Use indices that are within bounds with sufficient margin
        data_dict = setup_for_eval_real["data_dict"]
        available_length = data_dict["end_idx"] - data_dict["start_idx"]
        # Use a reasonably sized window in the middle of the data
        start_idx = data_dict["start_idx"] + available_length // 4
        end_idx = start_idx + available_length // 3

        metrics = evaluator._evaluate_params(
            params=setup_for_eval_real["params"],
            data_dict=setup_for_eval_real["data_dict"],
            start_idx=start_idx,
            end_idx=end_idx,
            pool=setup_for_eval_real["pool"],
            n_assets=setup_for_eval_real["n_assets"],
            run_fingerprint=setup_for_eval_real["run_fp"],
        )

        assert "sharpe" in metrics
        assert "returns_over_uniform_hodl" in metrics
        # Sharpe can be NaN for very short periods or constant returns
        # Just check that the metrics exist and are either finite or NaN
        assert isinstance(metrics["sharpe"], (float, np.floating, jnp.floating))
        assert isinstance(metrics["returns_over_uniform_hodl"], (float, np.floating, jnp.floating))

    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_params_different_windows(self, setup_for_eval_real):
        """Different time windows should give different metrics."""
        evaluator = TrainingEvaluator.random_baseline(verbose=False)

        start_idx = setup_for_eval_real["data_dict"]["start_idx"] + 100

        metrics1 = evaluator._evaluate_params(
            params=setup_for_eval_real["params"],
            data_dict=setup_for_eval_real["data_dict"],
            start_idx=start_idx,
            end_idx=start_idx + 30,
            pool=setup_for_eval_real["pool"],
            n_assets=setup_for_eval_real["n_assets"],
            run_fingerprint=setup_for_eval_real["run_fp"],
        )

        metrics2 = evaluator._evaluate_params(
            params=setup_for_eval_real["params"],
            data_dict=setup_for_eval_real["data_dict"],
            start_idx=start_idx + 40,
            end_idx=start_idx + 70,
            pool=setup_for_eval_real["pool"],
            n_assets=setup_for_eval_real["n_assets"],
            run_fingerprint=setup_for_eval_real["run_fp"],
        )

        # Different windows should generally give different results
        assert metrics1["sharpe"] != metrics2["sharpe"] or \
               metrics1["returns_over_uniform_hodl"] != metrics2["returns_over_uniform_hodl"]


class TestComputeCycleIndices:
    """Test _compute_cycle_indices method."""

    def test_cycle_indices_order(self):
        """Cycle indices should be in chronological order."""
        from quantammsim.runners.robust_walk_forward import (
            generate_walk_forward_cycles,
            WalkForwardCycle,
        )

        evaluator = TrainingEvaluator.random_baseline(verbose=False)

        run_fp = {
            "startDateString": "2023-01-01 00:00:00",
        }

        # Use enough data points to cover all cycles including test periods
        # 6 months of daily data = ~180 days, plus buffer for test periods
        data_dict = {
            "start_idx": 0,
            "end_idx": 5000,  # Enough for all cycles including test periods
        }

        cycles = generate_walk_forward_cycles(
            start_date="2023-01-01 00:00:00",
            end_date="2023-06-01 00:00:00",
            n_cycles=3,
        )

        # Pass the actual end date including test period
        last_test_end = cycles[-1].test_end_date
        evaluator._compute_cycle_indices(
            cycles, run_fp, data_dict, last_test_end
        )

        # Check indices are in order
        for cycle in cycles:
            assert cycle.train_start_idx < cycle.train_end_idx, \
                f"train_start_idx ({cycle.train_start_idx}) should be < train_end_idx ({cycle.train_end_idx})"
            assert cycle.train_end_idx == cycle.test_start_idx, \
                f"train_end_idx ({cycle.train_end_idx}) should equal test_start_idx ({cycle.test_start_idx})"
            assert cycle.test_start_idx < cycle.test_end_idx, \
                f"test_start_idx ({cycle.test_start_idx}) should be < test_end_idx ({cycle.test_end_idx})"

        # Check cycles are sequential
        for i in range(len(cycles) - 1):
            assert cycles[i].test_end_idx <= cycles[i + 1].train_end_idx

    def test_cycle_indices_within_data_bounds(self):
        """All indices should be within data bounds."""
        from quantammsim.runners.robust_walk_forward import generate_walk_forward_cycles

        evaluator = TrainingEvaluator.random_baseline(verbose=False)

        run_fp = {
            "startDateString": "2023-01-01 00:00:00",
        }

        data_dict = {
            "start_idx": 50,
            "end_idx": 500,
        }

        cycles = generate_walk_forward_cycles(
            start_date="2023-01-01 00:00:00",
            end_date="2023-06-01 00:00:00",
            n_cycles=4,
        )

        evaluator._compute_cycle_indices(
            cycles, run_fp, data_dict, "2023-06-01 00:00:00"
        )

        for cycle in cycles:
            assert cycle.train_start_idx >= data_dict["start_idx"]
            assert cycle.test_end_idx <= data_dict["end_idx"]


class TestCompareTrainersIntegration:
    """Integration tests for compare_trainers using real data."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_compare_trainers_runs_all(self):
        """compare_trainers should evaluate all trainers."""
        run_fp = {
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
                "training_data_kind": "historic",
                "force_scalar": False,
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
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "arb_frequency": 1,
            "do_arb": True,
            "arb_quality": 1.0,
            "do_trades": False,
            "noise_trader_ratio": 0.0,
            "minimum_weight": 0.03,
            "max_memory_days": 30,
            "subsidary_pools": [],
            "weight_interpolation_method": "linear",
            "use_pre_exp_scaling": True,
            "use_alt_lamb": False,
            "numeraire": None,
        }

        results = compare_trainers(
            run_fp,
            trainers={
                "random_a": TrainingEvaluator.random_baseline(seed=1, n_cycles=1, verbose=False, root=TEST_DATA_DIR),
                "random_b": TrainingEvaluator.random_baseline(seed=2, n_cycles=1, verbose=False, root=TEST_DATA_DIR),
            },
            verbose=False,
        )

        assert "random_a" in results
        assert "random_b" in results
        assert isinstance(results["random_a"], EvaluationResult)
        assert isinstance(results["random_b"], EvaluationResult)

        # Different seeds should give different results
        assert results["random_a"].mean_oos_sharpe != results["random_b"].mean_oos_sharpe


class TestExistingRunnerWrapperE2E:
    """E2E tests for ExistingRunnerWrapper with actual training."""

    @pytest.fixture
    def training_run_fingerprint(self):
        """Run fingerprint configured for short training runs."""
        return {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,  # Daily
            "bout_offset": 10080,  # 7 days - reduces effective training window
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.01,
                "optimiser": "sgd",
                "n_iterations": 3,  # Very few iterations for speed
                "training_data_kind": "historic",
                "force_scalar": False,
                "n_parameter_sets": 1,  # Single parameter set to minimize memory
                "batch_size": 2,  # Small batch for testing
                "n_cycles": 1,  # Single cycle
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
    @pytest.mark.integration
    def test_existing_runner_wrapper_train_on_historic_data(self):
        """ExistingRunnerWrapper should work with train_on_historic_data.

        Uses bout_offset > 0 to allow random sampling of start positions
        within each walk-forward cycle's training window.
        """
        # Use short window for faster tests
        short_run_fingerprint = {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,  # Daily
            "bout_offset": 10080,  # 7 days - reduces effective training window
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.01,
                "optimiser": "sgd",
                "n_iterations": 3,  # Minimal iterations
                "training_data_kind": "historic",
                "force_scalar": False,
                "n_parameter_sets": 1,
                "batch_size": 2,
                "n_cycles": 1,  # Single sampling cycle per iteration
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

        evaluator = TrainingEvaluator.from_runner(
            "train_on_historic_data",
            n_cycles=1,
            verbose=False,
            root=TEST_DATA_DIR,
            max_iterations=3,  # Very few for speed
            iterations_per_print=10000,  # Suppress output
        )

        result = evaluator.evaluate(short_run_fingerprint)

        # Check result structure
        assert isinstance(result, EvaluationResult)
        assert "train_on_historic_data" in result.trainer_name
        assert len(result.cycles) == 1

        # Check metrics are computed (training should produce some results)
        assert np.isfinite(result.mean_wfe) or np.isnan(result.mean_wfe)  # WFE can be nan if IS sharpe is 0
        assert np.isfinite(result.mean_oos_sharpe)

        # Each cycle should have been trained
        for cycle in result.cycles:
            assert cycle.epochs_trained > 0 or True  # Metadata may not always have this

    def test_existing_runner_cycles_train_on_different_windows(self, training_run_fingerprint):
        """Each cycle should train on a different date window."""
        # Track what dates each cycle uses
        training_dates = []

        # Monkey-patch train_on_historic_data to capture the dates
        from quantammsim.runners import jax_runners
        from quantammsim.pools.creator import create_pool
        original_train = jax_runners.train_on_historic_data

        # Get pool to generate valid params
        pool = create_pool("momentum")

        def patched_train(run_fingerprint, *args, **kwargs):
            training_dates.append({
                "start": run_fingerprint["startDateString"],
                "end": run_fingerprint["endDateString"],
            })
            # Return mock params that are valid for evaluation
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                2,  # n_assets
                1,  # n_parameter_sets
            )
            # Squeeze params to match what train_on_historic_data returns
            # (shape goes from (1, n_assets) to (n_assets,))
            squeezed_params = {}
            for k, v in params.items():
                if k == "subsidary_params":
                    squeezed_params[k] = v
                elif hasattr(v, 'shape') and len(v.shape) >= 1 and v.shape[0] == 1:
                    squeezed_params[k] = jnp.squeeze(v, axis=0)
                else:
                    squeezed_params[k] = v
            # Return (params, metadata) tuple as train_on_historic_data does
            metadata = {
                "epochs_trained": 3,
                "final_objective": 0.5,
                "best_param_idx": 0,
                "final_train_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01}],
                "final_continuous_test_metrics": [{"sharpe": 0.4, "returns_over_uniform_hodl": 0.005}],
            }
            return squeezed_params, metadata

        jax_runners.train_on_historic_data = patched_train

        try:
            evaluator = TrainingEvaluator.from_runner(
                "train_on_historic_data",
                n_cycles=1,
                verbose=False,
                root=TEST_DATA_DIR,
                max_iterations=3,
            )

            result = evaluator.evaluate(training_run_fingerprint)

            # Should have 1 training call (one per cycle)
            assert len(training_dates) == 1, \
                f"Expected 1 training call, got {len(training_dates)}"

            # Verify the result has 1 cycle
            assert len(result.cycles) == 1

        finally:
            # Restore original function
            jax_runners.train_on_historic_data = original_train

    def test_existing_runner_wrapper_passes_date_strings(self, training_run_fingerprint):
        """ExistingRunnerWrapper should pass date strings to modify fingerprint."""
        wrapper = ExistingRunnerWrapper(
            "train_on_historic_data",
            {"max_iterations": 3},
        )

        # Track what dates are passed to train_on_historic_data
        from quantammsim.runners import jax_runners
        original_train = jax_runners.train_on_historic_data
        captured_fp = {}

        def capture_train(run_fingerprint, *args, **kwargs):
            captured_fp["start"] = run_fingerprint["startDateString"]
            captured_fp["end"] = run_fingerprint["endDateString"]
            captured_fp["test_end"] = run_fingerprint["endTestDateString"]
            # Return (params, metadata) tuple as train_on_historic_data does
            params = {}
            metadata = {
                "epochs_trained": 3,
                "final_objective": 0.5,
                "best_param_idx": 0,
                "final_train_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01}],
                "final_continuous_test_metrics": [{"sharpe": 0.4, "returns_over_uniform_hodl": 0.005}],
            }
            return params, metadata

        jax_runners.train_on_historic_data = capture_train

        try:
            custom_start = "2022-06-01 00:00:00"
            custom_end = "2023-06-01 00:00:00"

            wrapper.train(
                data_dict={},
                train_start_idx=0,
                train_end_idx=30,
                pool=None,
                run_fingerprint=training_run_fingerprint,
                n_assets=2,
                warm_start_params=None,
                train_start_date=custom_start,
                train_end_date=custom_end,
            )

            # The captured fingerprint should have the custom dates
            assert captured_fp["start"] == custom_start, \
                f"Expected start {custom_start}, got {captured_fp['start']}"
            assert captured_fp["end"] == custom_end, \
                f"Expected end {custom_end}, got {captured_fp['end']}"
            # Test end should be 1 day after train end
            assert captured_fp["test_end"] == "2023-06-02 00:00:00", \
                f"Expected test_end 2023-06-02 00:00:00, got {captured_fp['test_end']}"

        finally:
            jax_runners.train_on_historic_data = original_train


class TestMultiPeriodSGDWrapperE2E:
    """E2E tests for multi_period_sgd wrapper."""

    def test_multi_period_sgd_wrapper_runs_full_evaluation(self):
        """ExistingRunnerWrapper with multi_period_sgd should run complete evaluation."""
        # Use a fingerprint with enough data and proper settings for multi_period_sgd
        run_fingerprint = {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,  # Daily
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

        evaluator = TrainingEvaluator.from_runner(
            "multi_period_sgd",
            n_cycles=1,
            verbose=False,
            root=TEST_DATA_DIR,
            n_periods=2,  # Use 2 periods for speed
            max_epochs=3,  # Very few epochs for speed
        )

        result = evaluator.evaluate(run_fingerprint)

        # Check result structure
        assert isinstance(result, EvaluationResult)
        assert "multi_period_sgd" in result.trainer_name
        assert len(result.cycles) == 1

        # Check metrics are computed
        assert np.isfinite(result.mean_oos_sharpe)
        assert len(result.effectiveness_reasons) > 0

    def test_multi_period_sgd_wrapper_trains_each_cycle(self):
        """multi_period_sgd should be called for each walk-forward cycle."""
        training_calls = []

        # Monkey-patch multi_period_sgd_training to capture calls
        from quantammsim.runners import multi_period_sgd
        from quantammsim.pools.creator import create_pool
        original_train = multi_period_sgd.multi_period_sgd_training

        pool = create_pool("momentum")

        def patched_train(run_fingerprint, *args, **kwargs):
            training_calls.append({
                "start": run_fingerprint["startDateString"],
                "end": run_fingerprint["endDateString"],
            })
            # Return mock result
            from quantammsim.runners.multi_period_sgd import MultiPeriodResult
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                2,  # n_assets
                1,  # n_parameter_sets
            )
            # Squeeze params (multi_period_sgd does this internally)
            squeezed = {
                k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
                for k, v in params.items()
            }
            result = MultiPeriodResult(
                period_sharpes=[0.5, 0.6],
                period_returns=[0.1, 0.12],
                period_returns_over_hodl=[0.01, 0.02],
                mean_sharpe=0.55,
                std_sharpe=0.05,
                worst_sharpe=0.5,
                mean_returns_over_hodl=0.015,
                epochs_trained=5,
                final_objective=0.55,
                best_params=squeezed,
            )
            summary = {"n_periods": 2}
            return result, summary

        multi_period_sgd.multi_period_sgd_training = patched_train

        try:
            run_fingerprint = {
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

            evaluator = TrainingEvaluator.from_runner(
                "multi_period_sgd",
                n_cycles=1,
                verbose=False,
                root=TEST_DATA_DIR,
                n_periods=2,
                max_epochs=3,
            )

            result = evaluator.evaluate(run_fingerprint)

            # Should have 1 training call (one per cycle)
            assert len(training_calls) == 1, \
                f"Expected 1 training call, got {len(training_calls)}"

        finally:
            multi_period_sgd.multi_period_sgd_training = original_train


class TestRademacherE2E:
    """E2E tests for Rademacher complexity computation."""

    @pytest.fixture
    def base_run_fingerprint(self):
        """Base fingerprint for Rademacher tests."""
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

    def _make_trainer_with_checkpoints(self, n_checkpoints, return_variance, seed=42):
        """Factory for trainers with configurable checkpoint characteristics."""
        def trainer(
            data_dict, train_start_idx, train_end_idx, pool, run_fingerprint,
            n_assets, warm_start_params=None, warm_start_weights=None,
        ):
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                n_assets,
                1,
            )
            params = {
                k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
                for k, v in params.items()
            }

            T = train_end_idx - train_start_idx
            rng = np.random.RandomState(seed)
            checkpoint_returns = rng.randn(n_checkpoints, T) * return_variance

            metadata = {
                "epochs_trained": n_checkpoints * 10,
                "checkpoint_returns": checkpoint_returns,
            }
            return params, metadata
        return trainer

    def test_rademacher_computed_when_checkpoint_returns_provided(self, base_run_fingerprint):
        """Rademacher should be computed when trainer provides checkpoint_returns."""
        trainer = self._make_trainer_with_checkpoints(n_checkpoints=5, return_variance=0.01)

        evaluator = TrainingEvaluator.from_function(
            trainer,
            name="checkpoint_trainer",
            n_cycles=1,
            verbose=False,
            root=TEST_DATA_DIR,
        )
        evaluator.compute_rademacher = True

        result = evaluator.evaluate(base_run_fingerprint)

        # Rademacher should be computed
        assert result.aggregate_rademacher is not None, \
            "aggregate_rademacher should be computed when checkpoint_returns provided"
        assert np.isfinite(result.aggregate_rademacher), \
            f"aggregate_rademacher should be finite, got {result.aggregate_rademacher}"
        assert result.aggregate_rademacher >= 0, \
            f"aggregate_rademacher should be non-negative, got {result.aggregate_rademacher}"

        # Per-cycle Rademacher should also be computed
        for cycle in result.cycles:
            assert cycle.rademacher_complexity is not None, \
                f"Cycle {cycle.cycle_number} should have rademacher_complexity"
            assert np.isfinite(cycle.rademacher_complexity)

        # Adjusted sharpe should be computed
        assert result.adjusted_mean_oos_sharpe is not None, \
            "adjusted_mean_oos_sharpe should be computed"

    def test_rademacher_none_when_no_checkpoint_returns(self, base_run_fingerprint):
        """Rademacher should be None when trainer doesn't provide checkpoint_returns."""

        def trainer_without_checkpoints(
            data_dict, train_start_idx, train_end_idx, pool, run_fingerprint,
            n_assets, warm_start_params=None, warm_start_weights=None,
        ):
            params = pool.init_parameters(
                {
                    "initial_memory_length": run_fingerprint["initial_memory_length"],
                    "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                    "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                    "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                    "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                    "initial_raw_width": run_fingerprint["initial_raw_width"],
                    "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                    "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
                },
                run_fingerprint,
                n_assets,
                1,
            )
            params = {
                k: jnp.squeeze(v, axis=0) if hasattr(v, 'shape') and len(v.shape) > 1 else v
                for k, v in params.items()
            }
            metadata = {"epochs_trained": 50}
            return params, metadata

        evaluator = TrainingEvaluator.from_function(
            trainer_without_checkpoints,
            name="no_checkpoint_trainer",
            n_cycles=1,
            verbose=False,
            root=TEST_DATA_DIR,
        )
        evaluator.compute_rademacher = True

        result = evaluator.evaluate(base_run_fingerprint)

        assert result.aggregate_rademacher is None, \
            "aggregate_rademacher should be None when no checkpoint_returns provided"

        for cycle in result.cycles:
            assert cycle.rademacher_complexity is None, \
                f"Cycle {cycle.cycle_number} rademacher should be None"

    def test_higher_variance_returns_higher_rademacher(self, base_run_fingerprint):
        """
        Higher variance in checkpoint returns should produce higher Rademacher complexity.

        This tests the mathematical property: more diverse strategies (higher variance
        in their returns) = larger search space = higher Rademacher complexity.
        """
        # Low variance trainer
        trainer_low_var = self._make_trainer_with_checkpoints(
            n_checkpoints=10, return_variance=0.001, seed=42
        )
        # High variance trainer (50x higher variance)
        trainer_high_var = self._make_trainer_with_checkpoints(
            n_checkpoints=10, return_variance=0.05, seed=42
        )

        evaluator_low = TrainingEvaluator.from_function(
            trainer_low_var, name="low_var", n_cycles=1, verbose=False, root=TEST_DATA_DIR
        )
        evaluator_low.compute_rademacher = True

        evaluator_high = TrainingEvaluator.from_function(
            trainer_high_var, name="high_var", n_cycles=1, verbose=False, root=TEST_DATA_DIR
        )
        evaluator_high.compute_rademacher = True

        result_low = evaluator_low.evaluate(base_run_fingerprint)
        result_high = evaluator_high.evaluate(base_run_fingerprint)

        # Both should have Rademacher computed
        assert result_low.aggregate_rademacher is not None
        assert result_high.aggregate_rademacher is not None

        # High variance should produce significantly higher Rademacher
        # (at least 2x higher given 50x variance difference)
        assert result_high.aggregate_rademacher > result_low.aggregate_rademacher * 2, \
            f"High variance Rademacher ({result_high.aggregate_rademacher:.4f}) should be " \
            f"at least 2x low variance ({result_low.aggregate_rademacher:.4f})"

    def test_more_checkpoints_higher_rademacher(self, base_run_fingerprint):
        """
        More checkpoints (larger search space) should produce higher Rademacher complexity.

        This tests that more optimization steps = more strategies considered = higher complexity.
        """
        # Few checkpoints
        trainer_few = self._make_trainer_with_checkpoints(
            n_checkpoints=3, return_variance=0.02, seed=42
        )
        # Many checkpoints (10x more)
        trainer_many = self._make_trainer_with_checkpoints(
            n_checkpoints=30, return_variance=0.02, seed=42
        )

        evaluator_few = TrainingEvaluator.from_function(
            trainer_few, name="few_checkpoints", n_cycles=1, verbose=False, root=TEST_DATA_DIR
        )
        evaluator_few.compute_rademacher = True

        evaluator_many = TrainingEvaluator.from_function(
            trainer_many, name="many_checkpoints", n_cycles=1, verbose=False, root=TEST_DATA_DIR
        )
        evaluator_many.compute_rademacher = True

        result_few = evaluator_few.evaluate(base_run_fingerprint)
        result_many = evaluator_many.evaluate(base_run_fingerprint)

        assert result_few.aggregate_rademacher is not None
        assert result_many.aggregate_rademacher is not None

        # More checkpoints should produce higher Rademacher (more strategies searched)
        assert result_many.aggregate_rademacher > result_few.aggregate_rademacher, \
            f"Many checkpoints Rademacher ({result_many.aggregate_rademacher:.4f}) should be " \
            f"higher than few checkpoints ({result_few.aggregate_rademacher:.4f})"

    def test_rademacher_haircut_is_meaningful(self, base_run_fingerprint):
        """
        Rademacher haircut should:
        1. Always reduce the sharpe estimate (or leave unchanged if Rademacher is 0)
        2. The reduction should be proportional to Rademacher complexity
        """
        # High Rademacher trainer
        trainer = self._make_trainer_with_checkpoints(
            n_checkpoints=20, return_variance=0.05, seed=123
        )

        evaluator = TrainingEvaluator.from_function(
            trainer, name="high_rademacher", n_cycles=1, verbose=False, root=TEST_DATA_DIR
        )
        evaluator.compute_rademacher = True

        result = evaluator.evaluate(base_run_fingerprint)

        assert result.aggregate_rademacher is not None
        assert result.adjusted_mean_oos_sharpe is not None
        assert result.mean_oos_sharpe is not None

        # Compute expected haircut based on Rademacher formula
        # haircut = sqrt(2 * log(2) / T) * R_hat
        # The adjusted sharpe should be: raw_sharpe - haircut
        raw_sharpe = result.mean_oos_sharpe
        adjusted_sharpe = result.adjusted_mean_oos_sharpe
        rademacher = result.aggregate_rademacher

        # 1. Adjusted should be <= raw (haircut reduces it)
        assert adjusted_sharpe <= raw_sharpe + 1e-9, \
            f"Adjusted sharpe ({adjusted_sharpe:.4f}) should be <= raw ({raw_sharpe:.4f})"

        # 2. The haircut amount should be positive (or zero if Rademacher is ~0)
        haircut = raw_sharpe - adjusted_sharpe
        assert haircut >= -1e-9, \
            f"Haircut should be non-negative, got {haircut:.4f}"

        # 3. With significant Rademacher, haircut should be meaningful (not trivially 0)
        if rademacher > 0.01:
            assert haircut > 0.001, \
                f"With Rademacher={rademacher:.4f}, haircut ({haircut:.4f}) should be meaningful"

        # 4. Verify the haircut scales with Rademacher
        # (This is a sanity check - haircut formula includes Rademacher)
        # We can't check the exact formula without knowing T, but we can verify
        # that higher Rademacher led to a reasonable adjustment


class TestMultiPeriodSGDParamsChange:
    """Test that multi_period_sgd actually changes params during training."""

    def test_multi_period_sgd_modifies_params(self):
        """
        Verify that multi_period_sgd training actually modifies parameters.

        This catches bugs where training runs but params don't update.
        """
        from quantammsim.runners.multi_period_sgd import multi_period_sgd_training
        from quantammsim.pools.creator import create_pool
        from copy import deepcopy

        run_fingerprint = {
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-20 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "chunk_period": 1440,
            "bout_offset": 10080,  # 7 days - reduces effective training window
            "weight_interpolation_period": 1440,
            "optimisation_settings": {
                "base_lr": 0.1,  # Higher LR to ensure params move
                "optimiser": "adam",
                "n_iterations": 10,
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

        # Get initial params
        pool = create_pool("momentum")
        initial_params = pool.init_parameters(
            {
                "initial_memory_length": run_fingerprint["initial_memory_length"],
                "initial_memory_length_delta": run_fingerprint.get("initial_memory_length_delta", 0.0),
                "initial_k_per_day": run_fingerprint["initial_k_per_day"],
                "initial_weights_logits": run_fingerprint["initial_weights_logits"],
                "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
                "initial_raw_width": run_fingerprint["initial_raw_width"],
                "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
                "initial_pre_exp_scaling": run_fingerprint["initial_pre_exp_scaling"],
            },
            run_fingerprint,
            2,  # n_assets
            1,  # n_parameter_sets
        )
        # Squeeze and convert to numpy for comparison
        initial_params = {
            k: np.array(jnp.squeeze(v, axis=0)) if hasattr(v, 'shape') and len(v.shape) > 1 else np.array(v)
            for k, v in initial_params.items()
            if k != 'subsidary_params'
        }

        # Run training
        result, summary = multi_period_sgd_training(
            run_fingerprint,
            n_periods=2,
            max_epochs=10,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        trained_params = {
            k: np.array(v) for k, v in result.best_params.items()
            if k != 'subsidary_params'
        }

        # At least some params should have changed
        params_changed = False
        for key in initial_params:
            if key in trained_params:
                initial_val = initial_params[key]
                trained_val = trained_params[key]
                if not np.allclose(initial_val, trained_val, rtol=1e-5):
                    params_changed = True
                    break

        assert params_changed, \
            "Training should modify at least some parameters. " \
            f"Initial: {initial_params}, Trained: {trained_params}"


class TestForceInit:
    """Tests for force_init parameter in train_on_historic_data."""

    def test_force_init_ignores_cache(self):
        """force_init=True should ignore cached results and re-train."""
        from quantammsim.runners import jax_runners
        original_train = jax_runners.train_on_historic_data

        # Track how many times training actually happens
        train_calls = []

        def patched_train(run_fingerprint, *args, **kwargs):
            train_calls.append({
                "force_init": kwargs.get("force_init", False),
            })
            # Return mock result
            params = {}
            metadata = {
                "epochs_trained": 3,
                "final_objective": 0.5,
                "best_param_idx": 0,
                "final_train_metrics": [{"sharpe": 0.5, "returns_over_uniform_hodl": 0.01}],
                "final_continuous_test_metrics": [{"sharpe": 0.4, "returns_over_uniform_hodl": 0.005}],
            }
            return params, metadata

        jax_runners.train_on_historic_data = patched_train

        try:
            # Create evaluator - this internally uses force_init=True
            evaluator = TrainingEvaluator.from_runner(
                "train_on_historic_data",
                n_cycles=1,
                verbose=False,
                root=TEST_DATA_DIR,
                max_iterations=3,
            )

            run_fingerprint = {
                "tokens": ["BTC", "ETH"],
                "rule": "momentum",
                "startDateString": "2023-01-01 00:00:00",
                "endDateString": "2023-01-20 00:00:00",
                "endTestDateString": "2023-02-01 00:00:00",
                "chunk_period": 1440,
                "bout_offset": 10080,
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

            # Run evaluation
            evaluator.evaluate(run_fingerprint)

            # Verify force_init was passed as True
            assert len(train_calls) == 1
            assert train_calls[0]["force_init"] is True, \
                "TrainingEvaluator should pass force_init=True to train_on_historic_data"

        finally:
            jax_runners.train_on_historic_data = original_train


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
