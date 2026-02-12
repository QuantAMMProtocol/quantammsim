"""Tests for BestParamsTracker and compute_selection_metric.

These tests verify the unified param selection logic used by both SGD and Optuna paths.
"""

import pytest
import numpy as np
from copy import deepcopy

from quantammsim.runners.jax_runner_utils import (
    BestParamsTracker,
    compute_selection_metric,
    SELECTION_METHODS,
)


class TestSelectionMethods:
    """Test that all selection methods are defined and valid."""

    def test_selection_methods_list_not_empty(self):
        assert len(SELECTION_METHODS) > 0

    def test_selection_methods_contains_expected(self):
        expected = ["last", "best_train", "best_val", "best_continuous_test", "best_train_min_test"]
        for method in expected:
            assert method in SELECTION_METHODS


class TestComputeSelectionMetric:
    """Test compute_selection_metric function."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        return {
            "train": [
                {"sharpe": 0.5, "returns_over_uniform_hodl": 0.1},
                {"sharpe": 0.8, "returns_over_uniform_hodl": 0.2},
                {"sharpe": 0.3, "returns_over_uniform_hodl": 0.05},
            ],
            "val": [
                {"sharpe": 0.4, "returns_over_uniform_hodl": 0.08},
                {"sharpe": 0.6, "returns_over_uniform_hodl": 0.15},
                {"sharpe": 0.9, "returns_over_uniform_hodl": 0.25},
            ],
            "test": [
                {"sharpe": 0.2, "returns_over_uniform_hodl": 0.03},
                {"sharpe": 0.7, "returns_over_uniform_hodl": 0.18},
                {"sharpe": 0.5, "returns_over_uniform_hodl": 0.12},
            ],
        }

    def test_unknown_method_raises(self, sample_metrics):
        with pytest.raises(ValueError, match="Unknown selection method"):
            compute_selection_metric(
                sample_metrics["train"],
                method="unknown_method",
            )

    def test_last_always_returns_inf_and_zero(self, sample_metrics):
        """'last' method should always return inf (so it always 'wins') and idx 0."""
        val, idx = compute_selection_metric(
            sample_metrics["train"],
            method="last",
        )
        assert val == float("inf")
        assert idx == 0

    def test_best_train_returns_best_train_idx(self, sample_metrics):
        """best_train should return idx of highest train metric."""
        val, idx = compute_selection_metric(
            sample_metrics["train"],
            method="best_train",
            metric="sharpe",
        )
        # idx 1 has sharpe=0.8 (highest)
        assert idx == 1
        # value should be mean of all sharpes
        expected_mean = np.mean([0.5, 0.8, 0.3])
        assert abs(val - expected_mean) < 1e-6

    def test_best_val_returns_best_val_idx(self, sample_metrics):
        """best_val should return idx of highest validation metric."""
        val, idx = compute_selection_metric(
            sample_metrics["train"],
            val_metrics=sample_metrics["val"],
            method="best_val",
            metric="sharpe",
        )
        # idx 2 has val sharpe=0.9 (highest)
        assert idx == 2

    def test_best_val_without_val_metrics_raises(self, sample_metrics):
        """best_val without val_metrics should raise."""
        with pytest.raises(ValueError, match="best_val method requires val_metrics"):
            compute_selection_metric(
                sample_metrics["train"],
                method="best_val",
            )

    def test_best_continuous_test_returns_best_test_idx(self, sample_metrics):
        """best_continuous_test should return idx of highest test metric."""
        val, idx = compute_selection_metric(
            sample_metrics["train"],
            continuous_test_metrics=sample_metrics["test"],
            method="best_continuous_test",
            metric="sharpe",
        )
        # idx 1 has test sharpe=0.7 (highest)
        assert idx == 1

    def test_best_train_min_test_respects_threshold(self, sample_metrics):
        """best_train_min_test should find best train among those meeting test threshold."""
        val, idx = compute_selection_metric(
            sample_metrics["train"],
            continuous_test_metrics=sample_metrics["test"],
            method="best_train_min_test",
            metric="sharpe",
            min_threshold=0.5,  # Only idx 1 (0.7) and idx 2 (0.5) meet this
        )
        # idx 1 has train sharpe=0.8 and test sharpe=0.7 >= 0.5
        # idx 2 has train sharpe=0.3 and test sharpe=0.5 >= 0.5
        # Best train among qualifying is idx 1
        assert idx == 1
        assert val == 0.8

    def test_best_train_min_test_falls_back_when_none_meet_threshold(self, sample_metrics):
        """If no param set meets threshold, fall back to best train."""
        val, idx = compute_selection_metric(
            sample_metrics["train"],
            continuous_test_metrics=sample_metrics["test"],
            method="best_train_min_test",
            metric="sharpe",
            min_threshold=1.0,  # None meet this threshold
        )
        # Should fall back to best train (idx 1 with sharpe=0.8)
        assert idx == 1

    def test_handles_nan_in_metrics(self):
        """Should handle NaN values gracefully."""
        train_metrics = [
            {"sharpe": float("nan")},
            {"sharpe": 0.5},
            {"sharpe": 0.3},
        ]
        val, idx = compute_selection_metric(
            train_metrics,
            method="best_train",
            metric="sharpe",
        )
        # idx 1 has sharpe=0.5 (highest non-NaN)
        assert idx == 1

    def test_empty_metrics_returns_negative_inf(self):
        """Empty metrics list should return -inf."""
        val, idx = compute_selection_metric(
            [],
            method="best_train",
        )
        assert val == -float("inf")
        assert idx == 0

    def test_different_metric_key(self, sample_metrics):
        """Should work with different metric keys."""
        val, idx = compute_selection_metric(
            sample_metrics["train"],
            method="best_train",
            metric="returns_over_uniform_hodl",
        )
        # idx 1 has returns_over_uniform_hodl=0.2 (highest)
        assert idx == 1


class TestBestParamsTracker:
    """Test BestParamsTracker class."""

    @pytest.fixture
    def mock_params(self):
        """Create mock batched parameters."""
        return {
            "log_k": np.random.rand(3, 2),  # 3 param sets, 2 assets
            "logit_lamb": np.random.rand(3, 2),
            "subsidary_params": [],
        }

    @pytest.fixture
    def mock_continuous_outputs(self):
        """Create mock continuous outputs."""
        return {
            "reserves": np.random.rand(3, 100, 2),  # 3 param sets, 100 timesteps, 2 assets
            "weights": np.random.rand(3, 100, 2),
            "value": np.random.rand(3, 100),
        }

    @pytest.fixture
    def mock_metrics(self):
        """Create mock metrics for 3 param sets."""
        return [
            {"sharpe": 0.3, "returns_over_uniform_hodl": 0.05},
            {"sharpe": 0.5, "returns_over_uniform_hodl": 0.10},
            {"sharpe": 0.4, "returns_over_uniform_hodl": 0.08},
        ]

    def test_init_with_valid_method(self):
        """Should initialize with valid selection method."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")
        assert tracker.selection_method == "best_train"
        assert tracker.metric == "sharpe"

    def test_init_with_invalid_method_raises(self):
        """Should raise with invalid selection method."""
        with pytest.raises(ValueError, match="Unknown selection method"):
            BestParamsTracker(selection_method="invalid_method")

    def test_first_update_always_improves(self, mock_params, mock_continuous_outputs, mock_metrics):
        """First update should always improve (sets baseline)."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")
        improved = tracker.update(
            iteration=0,
            params=mock_params,
            continuous_outputs=mock_continuous_outputs,
            train_metrics_list=mock_metrics,
        )
        assert bool(improved) is True
        assert tracker.best_iteration == 0
        assert tracker.last_iteration == 0

    def test_better_iteration_updates_best(self, mock_params, mock_continuous_outputs):
        """Better metrics should update best state."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        # First update with low metrics
        low_metrics = [{"sharpe": 0.1}, {"sharpe": 0.2}, {"sharpe": 0.15}]
        tracker.update(0, mock_params, mock_continuous_outputs, low_metrics)
        assert tracker.best_iteration == 0

        # Second update with higher metrics
        high_metrics = [{"sharpe": 0.5}, {"sharpe": 0.6}, {"sharpe": 0.55}]
        params2 = deepcopy(mock_params)
        improved = tracker.update(1, params2, mock_continuous_outputs, high_metrics)
        assert bool(improved) is True
        assert tracker.best_iteration == 1
        assert tracker.last_iteration == 1

    def test_worse_iteration_does_not_update_best(self, mock_params, mock_continuous_outputs):
        """Worse metrics should not update best state."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        # First update with high metrics
        high_metrics = [{"sharpe": 0.5}, {"sharpe": 0.6}, {"sharpe": 0.55}]
        tracker.update(0, mock_params, mock_continuous_outputs, high_metrics)

        # Second update with lower metrics
        low_metrics = [{"sharpe": 0.1}, {"sharpe": 0.2}, {"sharpe": 0.15}]
        params2 = deepcopy(mock_params)
        improved = tracker.update(1, params2, mock_continuous_outputs, low_metrics)
        assert bool(improved) is False
        assert tracker.best_iteration == 0  # Still iteration 0
        assert tracker.last_iteration == 1  # Last is updated

    def test_last_always_updated(self, mock_params, mock_continuous_outputs, mock_metrics):
        """Last state should always be updated regardless of improvement."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        tracker.update(0, mock_params, mock_continuous_outputs, mock_metrics)
        tracker.update(1, mock_params, mock_continuous_outputs, mock_metrics)
        tracker.update(2, mock_params, mock_continuous_outputs, mock_metrics)

        assert tracker.last_iteration == 2

    def test_get_results_returns_both_last_and_best(self, mock_params, mock_continuous_outputs):
        """get_results should return both last and best state."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        # Iteration 0: medium metrics
        medium_metrics = [{"sharpe": 0.4}, {"sharpe": 0.5}, {"sharpe": 0.45}]
        tracker.update(0, mock_params, mock_continuous_outputs, medium_metrics)

        # Iteration 1: high metrics (best)
        high_metrics = [{"sharpe": 0.7}, {"sharpe": 0.8}, {"sharpe": 0.75}]
        tracker.update(1, deepcopy(mock_params), mock_continuous_outputs, high_metrics)

        # Iteration 2: low metrics (last)
        low_metrics = [{"sharpe": 0.2}, {"sharpe": 0.3}, {"sharpe": 0.25}]
        tracker.update(2, deepcopy(mock_params), mock_continuous_outputs, low_metrics)

        results = tracker.get_results(n_param_sets=3, train_bout_length=50)

        assert results["best_iteration"] == 1
        assert results["last_iteration"] == 2
        assert results["best_train_metrics"] == high_metrics
        assert results["last_train_metrics"] == low_metrics

    def test_get_results_extracts_correct_time_index(self, mock_params, mock_continuous_outputs):
        """final_reserves/weights should be extracted at train_bout_length - 1."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        metrics = [{"sharpe": 0.5}, {"sharpe": 0.6}, {"sharpe": 0.55}]
        tracker.update(0, mock_params, mock_continuous_outputs, metrics)

        train_bout_length = 50
        results = tracker.get_results(n_param_sets=3, train_bout_length=train_bout_length)

        # Should extract at index train_bout_length - 1 = 49
        expected_reserves = mock_continuous_outputs["reserves"][:, train_bout_length - 1, :]
        expected_weights = mock_continuous_outputs["weights"][:, train_bout_length - 1, :]

        np.testing.assert_array_equal(results["last_final_reserves"], expected_reserves)
        np.testing.assert_array_equal(results["last_final_weights"], expected_weights)

    def test_select_param_set_extracts_single_set(self, mock_params):
        """select_param_set should extract single param set from batched."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        # Select param set at index 1
        selected = tracker.select_param_set(mock_params, idx=1, n_param_sets=3)

        assert selected["log_k"].shape == (2,)  # Single param set, 2 assets
        np.testing.assert_array_equal(selected["log_k"], mock_params["log_k"][1])

    def test_select_param_set_handles_single_param_set(self):
        """select_param_set should handle single param set (squeeze)."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        single_params = {
            "log_k": np.random.rand(1, 2),  # 1 param set
            "subsidary_params": [],
        }

        selected = tracker.select_param_set(single_params, idx=0, n_param_sets=1)
        assert selected["log_k"].shape == (2,)

    def test_best_val_selection_method(self, mock_params, mock_continuous_outputs):
        """Test best_val selection method."""
        tracker = BestParamsTracker(selection_method="best_val", metric="sharpe")

        train_metrics = [{"sharpe": 0.8}, {"sharpe": 0.5}, {"sharpe": 0.3}]
        val_metrics = [{"sharpe": 0.3}, {"sharpe": 0.9}, {"sharpe": 0.4}]  # idx 1 has best val

        tracker.update(
            0, mock_params, mock_continuous_outputs,
            train_metrics, val_metrics_list=val_metrics
        )

        results = tracker.get_results(n_param_sets=3, train_bout_length=50)

        # Best param idx should be 1 (best validation sharpe)
        assert results["best_param_idx"] == 1


class TestBestParamsTrackerIntegration:
    """Integration tests for BestParamsTracker with realistic scenarios."""

    def test_sgd_with_validation_scenario(self):
        """Simulate SGD training with validation holdout."""
        tracker = BestParamsTracker(selection_method="best_val", metric="sharpe")

        n_param_sets = 2
        n_timesteps = 100
        n_assets = 3

        best_val_iteration = None
        best_val_sharpe = -float("inf")

        for iteration in range(10):
            # Simulate params
            params = {
                "log_k": np.random.rand(n_param_sets, n_assets),
                "logit_lamb": np.random.rand(n_param_sets, n_assets),
                "subsidary_params": [],
            }

            # Simulate continuous outputs
            continuous_outputs = {
                "reserves": np.random.rand(n_param_sets, n_timesteps, n_assets),
                "weights": np.random.rand(n_param_sets, n_timesteps, n_assets),
                "value": np.random.rand(n_param_sets, n_timesteps),
            }

            # Simulate metrics with some variation
            train_metrics = [
                {"sharpe": np.random.rand() * 0.5 + iteration * 0.05}
                for _ in range(n_param_sets)
            ]
            val_metrics = [
                {"sharpe": np.random.rand() * 0.3 + (iteration % 5) * 0.1}
                for _ in range(n_param_sets)
            ]

            improved = tracker.update(
                iteration, params, continuous_outputs,
                train_metrics, val_metrics_list=val_metrics
            )

            # Track expected best for verification
            current_val_sharpe = np.mean([m["sharpe"] for m in val_metrics])
            if current_val_sharpe > best_val_sharpe:
                best_val_sharpe = current_val_sharpe
                best_val_iteration = iteration

        results = tracker.get_results(n_param_sets, train_bout_length=80)

        # Verify last is always the final iteration
        assert results["last_iteration"] == 9

        # Verify best tracks the iteration with best validation
        assert results["best_iteration"] == best_val_iteration

        # Verify selection info
        assert results["selection_method"] == "best_val"
        assert results["selection_metric"] == "sharpe"

    def test_optuna_scenario_best_train(self):
        """Simulate Optuna optimization tracking best train."""
        tracker = BestParamsTracker(selection_method="best_train", metric="returns_over_uniform_hodl")

        n_timesteps = 100
        n_assets = 2

        for trial in range(20):
            # Optuna uses single param set
            params = {
                "log_k": np.random.rand(1, n_assets),
                "logit_lamb": np.random.rand(1, n_assets),
                "subsidary_params": [],
            }

            continuous_outputs = {
                "reserves": np.random.rand(1, n_timesteps, n_assets),
                "weights": np.random.rand(1, n_timesteps, n_assets),
                "value": np.random.rand(1, n_timesteps),
            }

            # Simulate training metric (higher for some trials)
            train_return = np.random.rand() * 0.2 - 0.05  # -0.05 to 0.15
            train_metrics = [{"returns_over_uniform_hodl": train_return, "sharpe": np.random.rand()}]

            tracker.update(trial, params, continuous_outputs, train_metrics)

        results = tracker.get_results(n_param_sets=1, train_bout_length=80)

        # Verify we tracked something
        assert results["best_iteration"] >= 0
        assert results["best_iteration"] <= 19
        assert results["last_iteration"] == 19

        # Verify final reserves shape (single param set)
        assert results["last_final_reserves"].shape == (1, n_assets)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_iteration(self):
        """Test with only one iteration."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        params = {"log_k": np.random.rand(2, 3), "subsidary_params": []}
        outputs = {
            "reserves": np.random.rand(2, 50, 3),
            "weights": np.random.rand(2, 50, 3),
        }
        metrics = [{"sharpe": 0.5}, {"sharpe": 0.6}]

        tracker.update(0, params, outputs, metrics)
        results = tracker.get_results(n_param_sets=2, train_bout_length=40)

        assert results["best_iteration"] == 0
        assert results["last_iteration"] == 0

    def test_all_nan_metrics(self):
        """Test handling of all-NaN metrics."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        params = {"log_k": np.random.rand(2, 3), "subsidary_params": []}
        outputs = {
            "reserves": np.random.rand(2, 50, 3),
            "weights": np.random.rand(2, 50, 3),
        }
        nan_metrics = [{"sharpe": float("nan")}, {"sharpe": float("nan")}]

        # Should not crash
        tracker.update(0, params, outputs, nan_metrics)
        results = tracker.get_results(n_param_sets=2, train_bout_length=40)

        # Should still have results (even if metric value is -inf)
        assert "best_iteration" in results

    def test_train_bout_length_at_boundary(self):
        """Test when train_bout_length equals total timesteps."""
        tracker = BestParamsTracker(selection_method="best_train", metric="sharpe")

        params = {"log_k": np.random.rand(2, 3), "subsidary_params": []}
        n_timesteps = 50
        outputs = {
            "reserves": np.random.rand(2, n_timesteps, 3),
            "weights": np.random.rand(2, n_timesteps, 3),
        }
        metrics = [{"sharpe": 0.5}, {"sharpe": 0.6}]

        tracker.update(0, params, outputs, metrics)

        # Extract at the last valid index
        results = tracker.get_results(n_param_sets=2, train_bout_length=n_timesteps)

        # Should extract at index 49 (n_timesteps - 1)
        expected_reserves = outputs["reserves"][:, n_timesteps - 1, :]
        np.testing.assert_array_equal(results["last_final_reserves"], expected_reserves)

    def test_zero_min_threshold(self):
        """Test best_train_min_test with zero threshold (all pass)."""
        tracker = BestParamsTracker(
            selection_method="best_train_min_test",
            metric="sharpe",
            min_threshold=0.0,
        )

        params = {"log_k": np.random.rand(3, 2), "subsidary_params": []}
        outputs = {
            "reserves": np.random.rand(3, 50, 2),
            "weights": np.random.rand(3, 50, 2),
        }
        train_metrics = [{"sharpe": 0.3}, {"sharpe": 0.8}, {"sharpe": 0.5}]
        test_metrics = [{"sharpe": 0.1}, {"sharpe": 0.2}, {"sharpe": 0.15}]

        tracker.update(0, params, outputs, train_metrics, continuous_test_metrics_list=test_metrics)
        results = tracker.get_results(n_param_sets=3, train_bout_length=40)

        # All pass threshold, so best train (idx 1, sharpe=0.8) should be selected
        assert results["best_param_idx"] == 1
