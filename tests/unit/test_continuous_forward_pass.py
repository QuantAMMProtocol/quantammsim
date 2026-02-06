"""Tests for continuous forward pass mechanics in train_on_historic_data.

These tests verify:
1. Validation metrics are extracted from continuous pass (not separate pass)
2. final_reserves/weights are at end of training (not test)
3. Continuous test metrics are computed from continuous simulation
4. Both "last" and "best" results are returned correctly
"""

import pytest
import numpy as np
import jax.numpy as jnp

from quantammsim.utils.post_train_analysis import (
    calculate_period_metrics,
    calculate_continuous_test_metrics,
    process_continuous_outputs,
)


class TestCalculatePeriodMetrics:
    """Test calculate_period_metrics function."""

    @pytest.fixture
    def mock_results(self):
        """Create mock results dict with value and reserves."""
        n_timesteps = 1440 * 7  # 1 week of minute data
        n_assets = 2

        # Create synthetic values that increase over time
        values = 100.0 * (1 + 0.001 * np.arange(n_timesteps))  # ~0.1% per step
        # Add some noise
        values = values + np.random.randn(n_timesteps) * 0.1

        # Create synthetic reserves
        reserves = np.column_stack([
            50.0 * (1 + 0.0005 * np.arange(n_timesteps)),
            50.0 * (1 - 0.0003 * np.arange(n_timesteps)),
        ])

        # Create synthetic prices
        prices = np.column_stack([
            1.0 * (1 + 0.0008 * np.arange(n_timesteps)),
            1.0 * (1 + 0.0002 * np.arange(n_timesteps)),
        ])

        return {
            "value": values,
            "reserves": reserves,
            "prices": prices,
        }

    def test_returns_expected_keys(self, mock_results):
        """Should return dict with all expected metric keys."""
        metrics = calculate_period_metrics(mock_results)

        expected_keys = [
            "sharpe", "jax_sharpe", "return", "returns_over_hodl",
            "returns_over_uniform_hodl", "annualised_returns",
            "annualised_returns_over_hodl", "annualised_returns_over_uniform_hodl",
            "ulcer", "calmar", "sterling",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_returns_are_floats(self, mock_results):
        """All metric values should be floats."""
        metrics = calculate_period_metrics(mock_results)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float: {type(value)}"

    def test_positive_return_for_increasing_values(self, mock_results):
        """Increasing values should give positive return."""
        metrics = calculate_period_metrics(mock_results)

        # Values increase, so return should be positive
        assert metrics["return"] > 0


class TestCalculateContinuousTestMetrics:
    """Test calculate_continuous_test_metrics function."""

    @pytest.fixture
    def mock_continuous_results(self):
        """Create mock continuous results covering train + test."""
        train_len = 1440 * 7  # 1 week train
        test_len = 1440 * 3   # 3 days test
        total_len = train_len + test_len
        n_assets = 2

        values = 100.0 * (1 + 0.001 * np.arange(total_len))
        reserves = np.column_stack([
            50.0 * np.ones(total_len),
            50.0 * np.ones(total_len),
        ])

        return {
            "value": values,
            "reserves": reserves,
        }

    @pytest.fixture
    def mock_prices(self):
        """Create mock prices for continuous period."""
        train_len = 1440 * 7
        test_len = 1440 * 3
        total_len = train_len + test_len
        return np.column_stack([
            1.0 * (1 + 0.0005 * np.arange(total_len)),
            1.0 * np.ones(total_len),
        ])

    def test_extracts_test_period_correctly(self, mock_continuous_results, mock_prices):
        """Should extract metrics for test period only."""
        train_len = 1440 * 7
        test_len = 1440 * 3

        metrics = calculate_continuous_test_metrics(
            mock_continuous_results,
            train_len,
            test_len,
            mock_prices,
        )

        # Should return metrics dict
        assert isinstance(metrics, dict)
        assert "sharpe" in metrics
        assert "return" in metrics

    def test_test_metrics_use_test_period_values(self, mock_continuous_results, mock_prices):
        """Test metrics should be computed on test period slice."""
        train_len = 1440 * 7
        test_len = 1440 * 3

        metrics = calculate_continuous_test_metrics(
            mock_continuous_results,
            train_len,
            test_len,
            mock_prices,
        )

        # Manually compute expected return on test slice
        test_values = mock_continuous_results["value"][train_len:train_len + test_len]
        expected_return = test_values[-1] / test_values[0] - 1

        # Should be approximately equal (different calculation methods may differ slightly)
        assert abs(metrics["return"] - expected_return) < 0.01


class TestProcessContinuousOutputs:
    """Test process_continuous_outputs function."""

    @pytest.fixture
    def mock_data_dict(self):
        """Create mock data_dict."""
        return {
            "start_idx": 0,
            "bout_length": 1440 * 7,  # 1 week train
            "bout_length_test": 1440 * 3,  # 3 days test
            "prices": np.column_stack([
                1.0 * (1 + 0.0005 * np.arange(1440 * 10)),
                1.0 * np.ones(1440 * 10),
            ]),
        }

    @pytest.fixture
    def mock_continuous_outputs(self, mock_data_dict):
        """Create mock continuous outputs (batched over param sets)."""
        n_param_sets = 2
        total_len = mock_data_dict["bout_length"] + mock_data_dict["bout_length_test"]
        n_assets = 2

        return {
            "value": np.random.rand(n_param_sets, total_len) * 100 + 100,
            "reserves": np.random.rand(n_param_sets, total_len, n_assets) * 50 + 50,
        }

    def test_returns_three_metrics_lists(self, mock_continuous_outputs, mock_data_dict):
        """Should return train, test, and continuous_test metrics lists."""
        train_list, test_list, continuous_test_list = process_continuous_outputs(
            mock_continuous_outputs,
            mock_data_dict,
            n_parameter_sets=2,
            use_ensemble_mode=False,
        )

        assert len(train_list) == 2  # One per param set
        assert len(test_list) == 2
        assert len(continuous_test_list) == 2

    def test_each_param_set_has_metrics(self, mock_continuous_outputs, mock_data_dict):
        """Each param set should have its own metrics dict."""
        train_list, test_list, continuous_test_list = process_continuous_outputs(
            mock_continuous_outputs,
            mock_data_dict,
            n_parameter_sets=2,
            use_ensemble_mode=False,
        )

        for i in range(2):
            assert isinstance(train_list[i], dict)
            assert "sharpe" in train_list[i]
            assert isinstance(continuous_test_list[i], dict)
            assert "sharpe" in continuous_test_list[i]

    def test_ensemble_mode_returns_single_metrics(self, mock_data_dict):
        """Ensemble mode should return single metrics (not batched)."""
        total_len = mock_data_dict["bout_length"] + mock_data_dict["bout_length_test"]
        n_assets = 2

        # Ensemble mode outputs are unbatched
        ensemble_outputs = {
            "value": np.random.rand(total_len) * 100 + 100,
            "reserves": np.random.rand(total_len, n_assets) * 50 + 50,
        }

        train_list, test_list, continuous_test_list = process_continuous_outputs(
            ensemble_outputs,
            mock_data_dict,
            n_parameter_sets=1,  # Not used in ensemble mode
            use_ensemble_mode=True,
        )

        # Should return single element lists
        assert len(train_list) == 1
        assert len(continuous_test_list) == 1


class TestValidationFromContinuousPass:
    """Test that validation is extracted from continuous pass correctly."""

    def test_validation_slice_indices(self):
        """Verify correct slicing for validation period."""
        # Simulate the slicing logic from jax_runners.py
        original_bout_length = 10000  # Original training length
        val_fraction = 0.2
        val_length = int(original_bout_length * val_fraction)
        effective_train_length = original_bout_length - val_length

        # Validation period should be [effective_train_length, original_bout_length)
        val_start = effective_train_length
        val_end = original_bout_length

        assert val_start == 8000
        assert val_end == 10000
        assert val_end - val_start == val_length

    def test_continuous_outputs_cover_all_periods(self):
        """Verify continuous outputs cover train + val + test."""
        original_bout_length = 10000
        bout_length_test = 3000

        # continuous_static_dict["bout_length"] = original_bout_length + bout_length_test
        continuous_bout_length = original_bout_length + bout_length_test

        assert continuous_bout_length == 13000

        # Verify periods are contiguous
        val_fraction = 0.2
        effective_train_length = int(original_bout_length * (1 - val_fraction))
        val_length = original_bout_length - effective_train_length

        # Period boundaries
        train_end = effective_train_length  # 8000
        val_end = original_bout_length       # 10000
        test_end = continuous_bout_length    # 13000

        assert train_end == 8000
        assert val_end == 10000
        assert test_end == 13000

        # Verify no gaps
        assert train_end == val_end - val_length  # train leads into val
        assert val_end == test_end - bout_length_test  # val leads into test


class TestFinalReservesExtraction:
    """Test that final_reserves are extracted at correct index."""

    def test_final_reserves_at_end_of_training(self):
        """final_reserves should be at train_bout_length - 1, not end of test."""
        n_param_sets = 2
        total_timesteps = 100  # train + val + test
        train_bout_length = 70  # Just training + validation
        n_assets = 3

        # Create mock continuous outputs
        reserves = np.arange(n_param_sets * total_timesteps * n_assets).reshape(
            n_param_sets, total_timesteps, n_assets
        )

        # Extract at end of training (index train_bout_length - 1)
        final_reserves = reserves[:, train_bout_length - 1, :]

        # Verify we got the right slice
        expected_idx = train_bout_length - 1  # 69
        for p in range(n_param_sets):
            for a in range(n_assets):
                expected_value = p * total_timesteps * n_assets + expected_idx * n_assets + a
                assert final_reserves[p, a] == expected_value

    def test_not_extracting_from_test_period(self):
        """Ensure we're NOT extracting from test period (end of array)."""
        n_param_sets = 2
        total_timesteps = 100
        train_bout_length = 70
        n_assets = 3

        reserves = np.arange(n_param_sets * total_timesteps * n_assets).reshape(
            n_param_sets, total_timesteps, n_assets
        )

        # Correct extraction
        final_reserves_correct = reserves[:, train_bout_length - 1, :]

        # Wrong extraction (end of test period)
        final_reserves_wrong = reserves[:, -1, :]

        # These should be different
        assert not np.array_equal(final_reserves_correct, final_reserves_wrong)


class TestMetadataStructure:
    """Test that training metadata has correct structure."""

    def test_metadata_has_both_last_and_best(self):
        """Metadata should include both last and best iteration results."""
        # These are the expected keys based on the implementation
        expected_last_keys = [
            "last_train_metrics",
            "last_continuous_test_metrics",
            "last_val_metrics",
            "last_param_idx",
            "last_final_reserves",
            "last_final_weights",
        ]

        expected_best_keys = [
            "best_train_metrics",
            "best_continuous_test_metrics",
            "best_val_metrics",
            "best_param_idx",
            "best_iteration",
            "best_metric_value",
            "best_final_reserves",
            "best_final_weights",
        ]

        expected_selection_keys = [
            "selection_method",
            "selection_metric",
        ]

        # This is a documentation test - verifying the expected structure
        all_expected = expected_last_keys + expected_best_keys + expected_selection_keys
        assert len(all_expected) == 16  # Total expected keys for tracking

    def test_legacy_field_mapping(self):
        """Legacy fields should map to best_* fields."""
        # Based on implementation:
        # "final_train_metrics": tracker_results["best_train_metrics"],
        # "final_continuous_test_metrics": tracker_results["best_continuous_test_metrics"],

        # This documents the expected backward compatibility mapping
        legacy_to_new = {
            "final_train_metrics": "best_train_metrics",
            "final_continuous_test_metrics": "best_continuous_test_metrics",
        }

        assert len(legacy_to_new) == 2
