"""Tests for quantammsim.calibration.learned_mapping — attribute -> params mapping."""

import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, POOL_PREFIXES


class TestBuildTargets:
    """Test build_targets: stack per-pool fitted params into target matrix."""

    def test_target_matrix_shape(self, synthetic_pool_fit_result):
        from quantammsim.calibration.learned_mapping import build_targets

        pool_order = sorted(synthetic_pool_fit_result.keys())
        Y = build_targets(synthetic_pool_fit_result, pool_order)
        assert Y.shape == (len(pool_order), 2 + K_OBS)

    def test_target_ordering_matches_attributes(self, synthetic_pool_fit_result):
        from quantammsim.calibration.learned_mapping import build_targets

        pool_order = sorted(synthetic_pool_fit_result.keys())
        Y = build_targets(synthetic_pool_fit_result, pool_order)
        # First pool's log_cadence should be in first row
        expected_lc = synthetic_pool_fit_result[pool_order[0]]["log_cadence"]
        np.testing.assert_allclose(Y[0, 0], expected_lc)


class TestFitMapping:
    """Test fit_mapping: Ridge regression from attributes to params."""

    def _make_data(self, n_pools=10):
        np.random.seed(42)
        k_attr = 4
        X = np.random.randn(n_pools, k_attr)
        X[:, 0] = 1.0  # intercept
        W_true = np.random.randn(k_attr, 2 + K_OBS) * 0.5
        Y = X @ W_true + np.random.randn(n_pools, 2 + K_OBS) * 0.01
        return X, Y

    def test_returns_model(self):
        from quantammsim.calibration.learned_mapping import fit_mapping

        X, Y = self._make_data()
        model = fit_mapping(X, Y)
        assert isinstance(model, dict)
        assert "weights" in model
        assert "intercept" in model

    def test_predict_shape(self):
        from quantammsim.calibration.learned_mapping import fit_mapping

        X, Y = self._make_data()
        model = fit_mapping(X, Y)
        Y_pred = X @ model["weights"] + model["intercept"]
        assert Y_pred.shape == Y.shape

    def test_predict_reasonable_range(self):
        from quantammsim.calibration.learned_mapping import fit_mapping, predict_pool

        X, Y = self._make_data()
        # Constrain Y targets to reasonable range
        Y[:, 0] = np.log(np.random.uniform(1, 60, len(Y)))  # log_cadence
        Y[:, 1] = np.log(np.random.uniform(0.01, 10, len(Y)))  # log_gas
        model = fit_mapping(X, Y)

        result = predict_pool(model, X[0])
        assert 0.5 <= result["cadence_minutes"] <= 120.0
        assert result["gas_usd"] > 0

    def test_overfit_on_training_data(self):
        from quantammsim.calibration.learned_mapping import fit_mapping

        X, Y = self._make_data(n_pools=10)
        model = fit_mapping(X, Y, alpha=0.001)
        Y_pred = X @ model["weights"] + model["intercept"]
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.8

    def test_leave_one_out_runs(self):
        from quantammsim.calibration.learned_mapping import cross_validate_loo

        X, Y = self._make_data(n_pools=10)
        cv_result = cross_validate_loo(X, Y)
        assert "per_pool_errors" in cv_result
        assert len(cv_result["per_pool_errors"]) == 10


class TestPredictNewPool:
    """Test predict_pool: predict params for a single pool."""

    def test_predict_single_pool(self):
        from quantammsim.calibration.learned_mapping import fit_mapping, predict_pool

        np.random.seed(42)
        X = np.random.randn(5, 3)
        X[:, 0] = 1.0
        Y = np.random.randn(5, 2 + K_OBS)
        model = fit_mapping(X, Y)
        result = predict_pool(model, X[0])
        assert isinstance(result, dict)

    def test_predict_cadence_and_gas(self):
        from quantammsim.calibration.learned_mapping import fit_mapping, predict_pool

        np.random.seed(42)
        X = np.random.randn(5, 3)
        X[:, 0] = 1.0
        Y = np.random.randn(5, 2 + K_OBS)
        model = fit_mapping(X, Y)
        result = predict_pool(model, X[0])
        assert "cadence_minutes" in result
        assert "gas_usd" in result
        assert "log_cadence" in result
        assert "log_gas" in result

    def test_predict_noise_coeffs(self):
        from quantammsim.calibration.learned_mapping import fit_mapping, predict_pool

        np.random.seed(42)
        X = np.random.randn(5, 3)
        X[:, 0] = 1.0
        Y = np.random.randn(5, 2 + K_OBS)
        model = fit_mapping(X, Y)
        result = predict_pool(model, X[0])
        assert "noise_coeffs" in result
        assert len(result["noise_coeffs"]) == K_OBS

    def test_different_chains_different_predictions(self):
        from quantammsim.calibration.learned_mapping import fit_mapping, predict_pool

        np.random.seed(42)
        # X with chain dummy in col 1
        X = np.random.randn(10, 4)
        X[:, 0] = 1.0
        X[:5, 1] = 1.0   # chain A
        X[5:, 1] = 0.0   # chain B
        Y = np.random.randn(10, 2 + K_OBS)
        Y[:5, :] += 1.0   # chain A has different targets

        model = fit_mapping(X, Y, alpha=0.01)

        x_a = X[0].copy()
        x_b = X[0].copy()
        x_b[1] = 0.0  # flip chain

        pred_a = predict_pool(model, x_a)
        pred_b = predict_pool(model, x_b)
        # Predictions should differ
        assert pred_a["log_cadence"] != pred_b["log_cadence"]
