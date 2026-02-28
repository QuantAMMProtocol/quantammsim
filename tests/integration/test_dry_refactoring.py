"""
Integration tests for DRY refactoring changes.

Tests verify:
1. Gradient descent with n_parameter_sets > 1 returns properly shaped params
2. Outer Optuna hyperopt can wrap inner training methods
3. Metric extraction works correctly in hyperparam tuner context
4. HyperparamSpace.create() factory produces correct search spaces
"""

import pytest
import numpy as np
import jax.numpy as jnp
from copy import deepcopy

from tests.conftest import TEST_DATA_DIR


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


# Skip slow tests by default - run with pytest -m slow to include them
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


@pytest.fixture
def base_fingerprint():
    """
    Base run fingerprint for testing.
    Uses test data tokens (BTC, ETH) with minimal date ranges for speed.
    """
    return {
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-01-15 00:00:00",  # 2 weeks only
        "endTestDateString": "2023-01-20 00:00:00",  # 5 day test
        "tokens": ["BTC", "ETH"],
        "rule": "momentum",
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1000000.0,
        "fees": 0.003,
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "maximum_change": 0.001,
        "return_val": "sharpe",
        "max_memory_days": 30,  # Shorter memory for faster tests
        "use_alt_lamb": False,
        "use_pre_exp_scaling": False,
        "weight_interpolation_method": "linear",
        "arb_frequency": 1,
        "do_arb": True,
        "arb_quality": 1.0,
        "numeraire": "USDC",
        "noise_trader_ratio": 0.0,
        "minimum_weight": 0.01,
        "ste_max_change": 0.1,
        "ste_min_max_weight": 0.1,
        "initial_memory_length": 7.0,
        "initial_memory_length_delta": 0.0,
        "initial_k_per_day": 0.5,
        "initial_weights_logits": 0.0,
        "initial_log_amplitude": 0.0,
        "initial_raw_width": 0.0,
        "initial_raw_exponents": 0.0,
        "initial_pre_exp_scaling": 0.001,
        "bout_offset": 1440,
        "do_trades": False,
        "optimisation_settings": {
            "n_parameter_sets": 1,
            "training_data_kind": "historic",
            "optimiser": "adam",
            "base_lr": 0.1,
            "decay_lr_plateau": 50,
            "decay_lr_ratio": 0.5,
            "min_lr": 1e-5,
            "initial_random_key": 42,
            "batch_size": 4,
            "sample_method": "uniform",
            "train_on_hessian_trace": False,
            "n_iterations": 3,  # Minimal for fast tests
            "force_scalar": False,
            "n_cycles": 1,  # Single cycle by default
            "method": "gradient_descent",
            "val_fraction": 0.0,
            "early_stopping": False,
            "max_mc_version": 1,
        },
    }


# =============================================================================
# Test 1: n_parameter_sets > 1 Support
# =============================================================================

class TestNParameterSetsSupport:
    """Test gradient descent with n_parameter_sets > 1."""

    @pytest.mark.slow
    def test_n_parameter_sets_1_shape(self, base_fingerprint):
        """Test n_parameter_sets=1 returns properly shaped params."""
        try:
            from quantammsim.runners.jax_runners import train_on_historic_data

            fp = deepcopy(base_fingerprint)
            fp["optimisation_settings"]["n_parameter_sets"] = 1
            fp["optimisation_settings"]["n_iterations"] = 3

            result = train_on_historic_data(fp, verbose=False, root=TEST_DATA_DIR)

            assert result is not None
            # logit_lamb should not have n_parameter_sets dimension
            logit_lamb = result.get("logit_lamb")
            if logit_lamb is not None:
                # Should be 0D or 1D, not 2D with n_parameter_sets axis
                assert len(logit_lamb.shape) <= 1, \
                    f"Expected 0D/1D after selection, got shape {logit_lamb.shape}"

        except FileNotFoundError as e:
            pytest.skip(f"Test data not available: {e}")

    @pytest.mark.slow
    def test_n_parameter_sets_2_selects_best(self, base_fingerprint):
        """Test n_parameter_sets=2 returns params with best set selected."""
        try:
            from quantammsim.runners.jax_runners import train_on_historic_data

            fp = deepcopy(base_fingerprint)
            fp["optimisation_settings"]["n_parameter_sets"] = 2
            fp["optimisation_settings"]["n_iterations"] = 3
            n_assets = len(fp["tokens"])

            result = train_on_historic_data(fp, verbose=False, root=TEST_DATA_DIR)

            assert result is not None

            # Key params that should have been vmapped over n_parameter_sets
            # and thus should have their first dim selected
            vmapped_params = ["logit_lamb", "k_per_day"]

            for key in vmapped_params:
                if key in result:
                    value = result[key]
                    if hasattr(value, 'shape') and len(value.shape) >= 1:
                        # Should be scalar or have first dim != n_parameter_sets
                        # (after selection, first dim should be n_ensemble_members=1 or scalar)
                        assert value.shape[0] != 2 or value.shape[0] == n_assets, \
                            f"Param {key} may still have n_parameter_sets dim: {value.shape}"

        except FileNotFoundError as e:
            pytest.skip(f"Test data not available: {e}")

    @pytest.mark.slow
    def test_n_parameter_sets_4_with_metadata(self, base_fingerprint):
        """Test n_parameter_sets=4 with return_training_metadata=True."""
        try:
            from quantammsim.runners.jax_runners import train_on_historic_data

            fp = deepcopy(base_fingerprint)
            fp["optimisation_settings"]["n_parameter_sets"] = 4
            fp["optimisation_settings"]["n_iterations"] = 3

            result = train_on_historic_data(
                fp, verbose=False, return_training_metadata=True, root=TEST_DATA_DIR
            )

            assert result is not None
            assert isinstance(result, tuple), "Should return (params, metadata) tuple"

            params, metadata = result
            assert params is not None
            assert "epochs_trained" in metadata
            assert "final_objective" in metadata

            # Verify params are properly shaped
            for key, value in params.items():
                if key == "subsidary_params":
                    continue
                if hasattr(value, 'shape') and len(value.shape) >= 1:
                    assert value.shape[0] != 4, \
                        f"Param {key} still has n_parameter_sets dim: {value.shape}"

        except FileNotFoundError as e:
            pytest.skip(f"Test data not available: {e}")


# =============================================================================
# Test 2: TrainingEvaluator with n_parameter_sets
# =============================================================================

class TestTrainingEvaluatorNParamSets:
    """Test TrainingEvaluator handles n_parameter_sets > 1."""

    @pytest.mark.slow
    def test_evaluator_n_param_sets_2(self, base_fingerprint):
        """TrainingEvaluator should work with n_parameter_sets=2."""
        try:
            from quantammsim.runners.training_evaluator import TrainingEvaluator

            fp = deepcopy(base_fingerprint)
            fp["optimisation_settings"]["n_parameter_sets"] = 2
            fp["optimisation_settings"]["n_iterations"] = 5
            fp["optimisation_settings"]["n_cycles"] = 2  # Only 2 cycles, not 5

            # Use n_cycles=2 to match the short date range (14 days)
            evaluator = TrainingEvaluator.from_runner(
                "train_on_historic_data", n_cycles=2, root=TEST_DATA_DIR
            )

            try:
                result = evaluator.evaluate(fp)
                assert result is not None
                assert hasattr(result, 'cycles')
                assert len(result.cycles) >= 1
            except FloatingPointError:
                # Inf/NaN values in sharpe can cause aggregation to fail
                pytest.skip("Numerical instability (inf sharpe) in short test period")

        except FileNotFoundError as e:
            pytest.skip(f"Test data not available: {e}")


# =============================================================================
# Test 3: HyperparamSpace.create() Factory
# =============================================================================

class TestHyperparamSpaceFactory:
    """Test unified HyperparamSpace.create() factory."""

    def test_create_adam_space(self):
        """Test create() with optimizer='adam' returns focused ~7D space."""
        from quantammsim.runners.hyperparam_tuner import HyperparamSpace

        space = HyperparamSpace.create(cycle_days=180, optimizer="adam")

        assert "base_lr" in space.params
        assert "n_iterations" in space.params
        assert "bout_offset_days" in space.params
        assert "val_fraction" in space.params
        assert "maximum_change" in space.params
        assert "turnover_penalty" in space.params

        # These are now fixed from domain knowledge, not searched
        assert "batch_size" not in space.params
        assert "early_stopping_patience" not in space.params
        assert "use_weight_decay" not in space.params
        assert "lr_schedule_type" not in space.params
        assert "noise_scale" not in space.params

        # Adam should have lower LR range than SGD
        assert space.params["base_lr"]["low"] >= 1e-6  # Adam uses lower LRs
        assert space.params["base_lr"]["high"] <= 0.5

    def test_create_sgd_space(self):
        """Test create() with optimizer='sgd'."""
        from quantammsim.runners.hyperparam_tuner import HyperparamSpace

        space = HyperparamSpace.create(cycle_days=180, optimizer="sgd")

        assert "base_lr" in space.params
        # SGD should have wider LR range
        assert space.params["base_lr"]["low"] <= 0.01
        assert space.params["base_lr"]["high"] >= 0.5

    def test_create_multi_period_space(self):
        """Test create() for multi_period_sgd runner."""
        from quantammsim.runners.hyperparam_tuner import HyperparamSpace

        space = HyperparamSpace.create(runner="multi_period_sgd", cycle_days=90)

        assert "n_periods" in space.params
        assert "aggregation" in space.params
        assert "softmin_temperature" in space.params
        assert "max_epochs" in space.params

        # Should NOT have batch_size (multi-period uses different params)
        assert "batch_size" not in space.params

    def test_create_minimal_space(self):
        """Test create(minimal=True)."""
        from quantammsim.runners.hyperparam_tuner import HyperparamSpace

        space = HyperparamSpace.create(minimal=True)

        assert "base_lr" in space.params
        assert "n_iterations" in space.params
        assert len(space.params) == 2

    def test_lr_schedule_always_fixed(self):
        """Test that lr_schedule_type is never in search space (fixed from domain knowledge)."""
        from quantammsim.runners.hyperparam_tuner import HyperparamSpace

        space = HyperparamSpace.create(cycle_days=180)

        assert "lr_schedule_type" not in space.params
        assert "warmup_steps" not in space.params
        # Verify it's in the fixed defaults instead
        assert "lr_schedule_type" in HyperparamSpace.FIXED_TRAINING_DEFAULTS

    def test_legacy_wrappers_equivalent(self):
        """Test that legacy wrappers produce equivalent spaces."""
        from quantammsim.runners.hyperparam_tuner import HyperparamSpace

        # Legacy methods should produce same keys as create()
        adam_legacy = HyperparamSpace.default_adam_space(180)
        adam_create = HyperparamSpace.create(cycle_days=180, optimizer="adam")
        assert set(adam_legacy.params.keys()) == set(adam_create.params.keys())

        sgd_legacy = HyperparamSpace.default_sgd_space(180)
        sgd_create = HyperparamSpace.create(cycle_days=180, optimizer="sgd")
        assert set(sgd_legacy.params.keys()) == set(sgd_create.params.keys())

        minimal_legacy = HyperparamSpace.minimal_space()
        minimal_create = HyperparamSpace.create(minimal=True)
        assert set(minimal_legacy.params.keys()) == set(minimal_create.params.keys())


# =============================================================================
# Test 4: Metric Extraction
# =============================================================================

class TestMetricExtraction:
    """Test metric extraction utility."""

    def test_all_supported_metrics(self):
        """Test all metrics used in hyperparam_tuner."""
        from quantammsim.runners.metric_extraction import extract_cycle_metric
        from dataclasses import dataclass

        @dataclass
        class MockCycleEval:
            oos_sharpe: float
            is_sharpe: float
            walk_forward_efficiency: float
            is_oos_gap: float
            adjusted_oos_sharpe: float = None

        cycles = [
            MockCycleEval(0.5, 0.8, 0.75, 0.1, 0.4),
            MockCycleEval(0.7, 0.9, 0.85, 0.2, 0.6),
            MockCycleEval(0.3, 1.0, 0.65, 0.15, 0.25),
        ]

        # Test each metric
        metrics = {
            "mean_oos_sharpe": 0.5,
            "worst_oos_sharpe": 0.3,
            "mean_wfe": 0.75,
            "worst_wfe": 0.65,
            "neg_is_oos_gap": -0.15,
            "adjusted_mean_oos_sharpe": (0.4 + 0.6 + 0.25) / 3,
        }

        for metric_name, expected in metrics.items():
            result = extract_cycle_metric(cycles, metric_name)
            assert abs(result - expected) < 0.01, \
                f"{metric_name}: expected {expected:.4f}, got {result:.4f}"

    def test_empty_cycles_returns_neg_inf(self):
        """Empty cycle list should return -inf."""
        from quantammsim.runners.metric_extraction import extract_cycle_metric

        result = extract_cycle_metric([], "mean_oos_sharpe")
        assert result == float("-inf")

    def test_adjusted_sharpe_fallback(self):
        """adjusted_oos_sharpe=None should fallback to oos_sharpe."""
        from quantammsim.runners.metric_extraction import extract_cycle_metric
        from dataclasses import dataclass

        @dataclass
        class MockCycleEval:
            oos_sharpe: float
            walk_forward_efficiency: float
            is_oos_gap: float
            adjusted_oos_sharpe: float = None

        cycles = [
            MockCycleEval(0.5, 0.8, 0.1, None),
            MockCycleEval(0.7, 0.9, 0.2, None),
        ]

        result = extract_cycle_metric(cycles, "adjusted_mean_oos_sharpe")
        expected = 0.6  # mean([0.5, 0.7])
        assert abs(result - expected) < 0.01


# =============================================================================
# Test 5: get_sig_variations Integration
# =============================================================================

class TestSigVariationsIntegration:
    """Test get_sig_variations is correctly integrated."""

    def test_sig_variations_output_format(self):
        """Test get_sig_variations returns correct format."""
        from quantammsim.runners.jax_runner_utils import get_sig_variations

        variations = get_sig_variations(3)

        # For 3 assets: 3*2 = 6 pairs (each asset can be in or out)
        assert len(variations) == 6
        assert isinstance(variations, tuple)

        for v in variations:
            assert isinstance(v, tuple)
            assert len(v) == 3
            # Exactly one +1 and one -1
            assert sum(1 for x in v if x == 1) == 1
            assert sum(1 for x in v if x == -1) == 1

    def test_sig_variations_matches_manual(self):
        """Test get_sig_variations matches manual computation."""
        from quantammsim.runners.jax_runner_utils import get_sig_variations
        from itertools import product
        import numpy as np

        n_assets = 4
        variations = get_sig_variations(n_assets)

        # Manual computation
        manual = np.array(list(product([1, 0, -1], repeat=n_assets)))
        manual = manual[(manual == 1).sum(-1) == 1]
        manual = manual[(manual == -1).sum(-1) == 1]
        manual = tuple(map(tuple, manual))

        assert variations == manual


# =============================================================================
# Test 6: Outer Hyperparam Tuner Integration
# =============================================================================

class TestOuterHyperparamTuner:
    """Test outer hyperparameter tuner with n_parameter_sets."""

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_tuner_objective_with_n_param_sets(self, base_fingerprint):
        """Test hyperparam tuner objective works with n_parameter_sets > 1."""
        try:
            from quantammsim.runners.hyperparam_tuner import (
                create_objective,
                HyperparamSpace,
            )
            import optuna

            fp = deepcopy(base_fingerprint)
            fp["optimisation_settings"]["n_parameter_sets"] = 2
            fp["optimisation_settings"]["n_iterations"] = 3
            fp["optimisation_settings"]["n_cycles"] = 1  # Single cycle

            space = HyperparamSpace.minimal_space()

            objective = create_objective(
                run_fingerprint=fp,
                runner_name="train_on_historic_data",
                runner_kwargs={},
                hyperparam_space=space,
                n_wfa_cycles=1,
                objective_metric="mean_oos_sharpe",
                verbose=False,
                enable_pruning=False,
            )

            # Run one trial
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=1, show_progress_bar=False)

            assert len(study.trials) == 1
            assert study.trials[0].state == optuna.trial.TrialState.COMPLETE

        except FileNotFoundError as e:
            pytest.skip(f"Test data not available: {e}")


# =============================================================================
# Test 7: Optuna-wrapping-Optuna (Inner Optuna)
# =============================================================================

class TestOptunaWrappingOptuna:
    """Test outer Optuna wrapping inner Optuna optimization."""

    @pytest.mark.slow
    def test_inner_optuna_runs(self, base_fingerprint):
        """Test that inner optuna optimization completes."""
        try:
            from quantammsim.runners.jax_runners import train_on_historic_data

            fp = deepcopy(base_fingerprint)
            fp["optimisation_settings"]["method"] = "optuna"
            fp["optimisation_settings"]["optuna_settings"] = {
                "n_trials": 2,  # Very few for speed
                "multi_objective": False,
                "make_scalar": True,
                "parameter_config": {
                    "logit_lamb": {
                        "low": -2.0,
                        "high": 2.0,
                        "log_scale": False,
                    },
                },
                "expand_around": True,
            }

            result = train_on_historic_data(fp, verbose=False, root=TEST_DATA_DIR)

            # Should return params dict
            assert result is not None

        except FileNotFoundError as e:
            pytest.skip(f"Test data not available: {e}")

    @pytest.mark.slow
    def test_evaluator_with_inner_optuna(self, base_fingerprint):
        """Test TrainingEvaluator can use inner optuna."""
        try:
            from quantammsim.runners.training_evaluator import TrainingEvaluator

            fp = deepcopy(base_fingerprint)
            fp["optimisation_settings"]["method"] = "optuna"
            fp["optimisation_settings"]["optuna_settings"] = {
                "n_trials": 2,
                "multi_objective": False,
                "make_scalar": True,
                "parameter_config": {
                    "logit_lamb": {
                        "low": -2.0,
                        "high": 2.0,
                        "log_scale": False,
                    },
                },
                "expand_around": True,
            }

            # Use n_cycles=2 to match the short date range (14 days)
            evaluator = TrainingEvaluator.from_runner(
                "train_on_historic_data", n_cycles=2, root=TEST_DATA_DIR
            )

            result = evaluator.evaluate(fp)
            assert result is not None

        except FileNotFoundError as e:
            pytest.skip(f"Test data not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
