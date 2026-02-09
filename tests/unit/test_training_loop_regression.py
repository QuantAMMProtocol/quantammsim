"""Regression tests for training loop internals.

Captures exact outputs of:
- Gradient computation (value_and_grad through the forward pass)
- NaN detection (has_nan_params)
- NaN reinitialization (nan_param_reinit)
- Period metrics (calculate_period_metrics)
- Continuous test metrics slicing

These serve as a safety net when refactoring these functions for
GPU utilization improvements. Tests pin actual numeric values so
that semantic bugs (wrong metric formula, wrong slicing, wrong
replacement logic) are caught even if the code runs without errors.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from copy import deepcopy

from quantammsim.runners.jax_runner_utils import (
    has_nan_params,
    nan_param_reinit,
)
from quantammsim.utils.post_train_analysis import (
    calculate_period_metrics,
    calculate_continuous_test_metrics,
)
from quantammsim.core_simulator.forward_pass import _calculate_return_value
from quantammsim.runners.jax_runners import do_run_on_historic_data

TEST_DATA_DIR = Path(__file__).parent.parent / "data"

# Tolerance for pinned value comparisons.
# Tight enough to catch formula bugs, loose enough for platform float differences.
RTOL = 1e-6


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def momentum_result():
    """BTC/ETH momentum forward pass with known params."""
    from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb

    fingerprint = {
        "tokens": ["BTC", "ETH"],
        "rule": "momentum",
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-06-01 00:00:00",
        "initial_pool_value": 1000000.0,
        "do_arb": True,
        "arb_quality": 1.0,
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
    }
    params = {
        "log_k": jnp.array([3.0, 3.0]),
        "logit_lamb": jnp.array([
            memory_days_to_logit_lamb(10.0, chunk_period=1440),
            memory_days_to_logit_lamb(10.0, chunk_period=1440),
        ]),
        "initial_weights_logits": jnp.array([0.0, 0.0]),
    }
    return do_run_on_historic_data(
        run_fingerprint=fingerprint, params=params, root=str(TEST_DATA_DIR),
    )


@pytest.fixture(scope="module")
def mean_reversion_result():
    """BTC/ETH mean_reversion_channel forward pass with known params.

    Tests the actual rule used in the tuning config.
    """
    from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb

    fingerprint = {
        "tokens": ["BTC", "ETH"],
        "rule": "mean_reversion_channel",
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-06-01 00:00:00",
        "initial_pool_value": 1000000.0,
        "do_arb": True,
        "arb_quality": 1.0,
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
    }
    params = {
        "log_k": jnp.array([3.0, 3.0]),
        "logit_lamb": jnp.array([
            memory_days_to_logit_lamb(10.0, chunk_period=1440),
            memory_days_to_logit_lamb(10.0, chunk_period=1440),
        ]),
        "initial_weights_logits": jnp.array([0.0, 0.0]),
        "log_amplitude": jnp.array([0.0, 0.0]),
        "raw_width": jnp.array([0.0, 0.0]),
        "raw_exponents": jnp.array([1.0, 1.0]),
        "raw_pre_exp_scaling": jnp.array([0.5, 0.5]),
    }
    return do_run_on_historic_data(
        run_fingerprint=fingerprint, params=params, root=str(TEST_DATA_DIR),
    )


@pytest.fixture(scope="module")
def mock_params_clean():
    """Clean params dict (no NaNs), shaped like batched mean_reversion_channel params.

    Includes all keys that mean_reversion_channel uses, not just momentum keys.
    Shape: (n_parameter_sets=4, n_tokens=2).
    """
    return {
        "log_k": jnp.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [2.0, 2.0]]),
        "logit_lamb": jnp.array([[-0.22, -0.22], [-0.5, -0.5], [0.1, 0.1], [-1.0, -1.0]]),
        "initial_weights_logits": jnp.array([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2], [0.0, 0.0]]),
        "log_amplitude": jnp.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1], [0.5, 0.5]]),
        "raw_width": jnp.array([[0.0, 0.0], [-0.5, -0.5], [0.5, 0.5], [0.0, 0.0]]),
        "raw_exponents": jnp.array([[1.0, 1.0], [1.5, 1.5], [0.5, 0.5], [2.0, 2.0]]),
        "raw_pre_exp_scaling": jnp.array([[0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [0.5, 0.5]]),
    }


@pytest.fixture(scope="module")
def mock_params_with_nans():
    """Params with NaNs in param sets 1 and 3 (0-indexed).

    NaN in log_k[1] and raw_exponents[3] — different keys to test cross-key detection.
    """
    return {
        "log_k": jnp.array([[3.0, 3.0], [float("nan"), 4.0], [5.0, 5.0], [2.0, 2.0]]),
        "logit_lamb": jnp.array([[-0.22, -0.22], [-0.5, -0.5], [0.1, 0.1], [-1.0, -1.0]]),
        "initial_weights_logits": jnp.array([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2], [0.0, 0.0]]),
        "log_amplitude": jnp.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1], [0.5, 0.5]]),
        "raw_width": jnp.array([[0.0, 0.0], [-0.5, -0.5], [0.5, 0.5], [0.0, 0.0]]),
        "raw_exponents": jnp.array([[1.0, 1.0], [1.5, 1.5], [0.5, 0.5], [float("nan"), float("nan")]]),
        "raw_pre_exp_scaling": jnp.array([[0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [0.5, 0.5]]),
    }


def _make_mock_pool(n_tokens, keys):
    """Create a MockPool that generates replacement params for given keys."""
    class MockPool:
        def init_parameters(self, initial_params, run_fingerprint, n_tokens_, n_parameter_sets):
            result = {}
            for key in keys:
                if key in ["initial_weights_logits"]:
                    result[key] = jnp.zeros((n_parameter_sets, n_tokens))
                else:
                    result[key] = jnp.ones((n_parameter_sets, n_tokens)) * 99.0
            return result
    return MockPool()


# ── has_nan_params tests ─────────────────────────────────────────────────────


class TestHasNanParams:
    """Test NaN detection in parameter dicts.

    Current contract: checks all keys EXCEPT initial_weights, initial_weights_logits,
    subsidary_params. Skips non-array values (no .shape). Detects NaN, NOT inf.
    """

    def test_clean_params_no_nans(self, mock_params_clean):
        assert has_nan_params(mock_params_clean) is False

    def test_nan_in_log_k_detected(self, mock_params_with_nans):
        assert has_nan_params(mock_params_with_nans) is True

    def test_nan_only_in_initial_weights_ignored(self):
        """NaNs in excluded keys should be skipped."""
        params = {
            "log_k": jnp.array([[3.0, 3.0]]),
            "logit_lamb": jnp.array([[-0.22, -0.22]]),
            "initial_weights_logits": jnp.array([[float("nan"), 0.0]]),
        }
        assert has_nan_params(params) is False

    def test_nan_in_non_standard_key_detected(self):
        """NaN in raw_exponents (mean_reversion_channel key) should be caught."""
        params = {
            "log_k": jnp.array([[3.0, 3.0]]),
            "logit_lamb": jnp.array([[-0.22, -0.22]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0]]),
            "raw_exponents": jnp.array([[float("nan"), 1.0]]),
        }
        assert has_nan_params(params) is True

    def test_single_nan_in_one_param_set(self):
        params = {
            "log_k": jnp.array([[3.0, 3.0], [3.0, float("nan")]]),
            "logit_lamb": jnp.array([[-0.22, -0.22], [-0.22, -0.22]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        }
        assert has_nan_params(params) is True

    def test_all_nans(self):
        params = {
            "log_k": jnp.array([[float("nan"), float("nan")]]),
            "logit_lamb": jnp.array([[float("nan"), float("nan")]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0]]),
        }
        assert has_nan_params(params) is True

    def test_inf_not_detected_as_nan(self):
        """Current contract: inf is NOT treated as NaN. Document this behavior
        so refactoring that changes it (e.g., switching to ~isfinite) is caught."""
        params = {
            "log_k": jnp.array([[float("inf"), 3.0]]),
            "logit_lamb": jnp.array([[-0.22, -0.22]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0]]),
        }
        assert has_nan_params(params) is False

    def test_non_array_value_skipped(self):
        """Params with non-array values (e.g., dict) should be silently skipped.
        This documents the hasattr(x, 'shape') guard in the current implementation."""
        params = {
            "log_k": jnp.array([[3.0, 3.0]]),
            "logit_lamb": jnp.array([[-0.22, -0.22]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0]]),
            "some_dict_param": {"nested": "data"},  # no .shape
        }
        # Should not raise, and should return False (no NaN in array params)
        assert has_nan_params(params) is False


# ── nan_param_reinit tests ───────────────────────────────────────────────────


class TestNanParamReinit:
    """Test selective reinitialization of NaN parameter sets."""

    def test_clean_params_unchanged(self, mock_params_clean):
        """When no NaNs, ALL params returned bitwise-identical."""
        original = deepcopy(mock_params_clean)
        grads = deepcopy(mock_params_clean)
        pool = _make_mock_pool(2, list(mock_params_clean.keys()))

        result = nan_param_reinit(
            deepcopy(mock_params_clean), grads, pool, {}, {}, 2, 4
        )
        for key in original:
            np.testing.assert_array_equal(
                np.array(result[key]), np.array(original[key]),
                err_msg=f"Key '{key}' changed despite no NaNs",
            )

    def test_nan_sets_replaced_others_preserved(self, mock_params_with_nans):
        """Param sets 1 and 3 have NaNs — they get replaced.
        Param sets 0 and 2 must be bitwise-identical to input."""
        keys = list(mock_params_with_nans.keys())
        pool = _make_mock_pool(2, keys)
        params = deepcopy(mock_params_with_nans)
        original = deepcopy(mock_params_with_nans)
        grads = deepcopy(mock_params_with_nans)

        result = nan_param_reinit(params, grads, pool, {}, {}, 2, 4)

        # Set 0: preserved
        for key in keys:
            if key != "initial_weights_logits":
                np.testing.assert_array_equal(
                    np.array(result[key][0]), np.array(original[key][0]),
                    err_msg=f"Set 0, key '{key}' should be unchanged",
                )

        # Set 1: ALL non-excluded keys replaced with 99.0
        for key in keys:
            if key not in ["initial_weights_logits", "subsidary_params", "initial_weights"]:
                np.testing.assert_allclose(
                    np.array(result[key][1]), [99.0, 99.0],
                    err_msg=f"Set 1, key '{key}' should be replaced with 99.0",
                )

        # Set 2: preserved
        for key in keys:
            if key != "initial_weights_logits":
                np.testing.assert_array_equal(
                    np.array(result[key][2]), np.array(original[key][2]),
                    err_msg=f"Set 2, key '{key}' should be unchanged",
                )

        # Set 3: replaced
        for key in keys:
            if key not in ["initial_weights_logits", "subsidary_params", "initial_weights"]:
                np.testing.assert_allclose(
                    np.array(result[key][3]), [99.0, 99.0],
                    err_msg=f"Set 3, key '{key}' should be replaced with 99.0",
                )

    def test_no_nans_in_output(self, mock_params_with_nans):
        """After reinit, no checked parameter should contain NaN."""
        keys = list(mock_params_with_nans.keys())
        pool = _make_mock_pool(2, keys)
        result = nan_param_reinit(
            deepcopy(mock_params_with_nans), deepcopy(mock_params_with_nans),
            pool, {}, {}, 2, 4,
        )
        assert not has_nan_params(result)

    def test_replacement_is_per_set_not_global(self):
        """Only the NaN set gets replaced — adjacent sets are not touched."""
        params = {
            "log_k": jnp.array([[1.0, 1.0], [float("nan"), 2.0], [3.0, 3.0]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        }
        pool = _make_mock_pool(2, ["log_k", "initial_weights_logits"])
        result = nan_param_reinit(
            deepcopy(params), deepcopy(params), pool, {}, {}, 2, 3,
        )
        np.testing.assert_array_equal(np.array(result["log_k"][0]), [1.0, 1.0])
        np.testing.assert_array_equal(np.array(result["log_k"][2]), [3.0, 3.0])
        assert not np.any(np.isnan(np.array(result["log_k"][1])))

    def test_nan_grads_clean_params_unchanged(self):
        """CRITICAL: nan_param_reinit checks PARAMS not GRADS.

        If grads have NaN but params don't, params must be returned unchanged.
        This documents the current contract: the grads argument is accepted
        but not used for NaN detection. A refactoring that starts checking
        grads would silently change behavior without this test.
        """
        clean_params = {
            "log_k": jnp.array([[3.0, 3.0], [4.0, 4.0]]),
            "logit_lamb": jnp.array([[-0.22, -0.22], [-0.5, -0.5]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        }
        nan_grads = {
            "log_k": jnp.array([[float("nan"), float("nan")], [float("nan"), float("nan")]]),
            "logit_lamb": jnp.array([[float("nan"), float("nan")], [float("nan"), float("nan")]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        }
        original = deepcopy(clean_params)
        pool = _make_mock_pool(2, list(clean_params.keys()))

        result = nan_param_reinit(
            deepcopy(clean_params), nan_grads, pool, {}, {}, 2, 2,
        )
        # ALL params must be bitwise-identical — grads should not trigger reinit
        for key in original:
            np.testing.assert_array_equal(
                np.array(result[key]), np.array(original[key]),
                err_msg=f"Key '{key}' changed despite clean params (NaN grads should be ignored)",
            )

    def test_single_param_set(self):
        """n_parameter_sets=1 must work — broadcasting risk with jnp.where refactoring."""
        params = {
            "log_k": jnp.array([[float("nan"), 3.0]]),
            "logit_lamb": jnp.array([[-0.22, -0.22]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0]]),
        }
        pool = _make_mock_pool(2, list(params.keys()))
        result = nan_param_reinit(
            deepcopy(params), deepcopy(params), pool, {}, {}, 2, 1,
        )
        assert not has_nan_params(result)
        # Verify replacement happened
        np.testing.assert_allclose(
            np.array(result["log_k"][0]), [99.0, 99.0],
            err_msg="Single param set should be replaced",
        )

    def test_initial_weights_logits_preserved_in_nan_sets(self, mock_params_with_nans):
        """When a param set has NaN, initial_weights_logits must NOT be replaced.

        The exclusion of initial_weights_logits from replacement is critical —
        if vectorized jnp.where accidentally replaces it, weight initialization
        would be corrupted.
        """
        keys = list(mock_params_with_nans.keys())
        pool = _make_mock_pool(2, keys)
        original = deepcopy(mock_params_with_nans)

        result = nan_param_reinit(
            deepcopy(mock_params_with_nans), deepcopy(mock_params_with_nans),
            pool, {}, {}, 2, 4,
        )

        # Sets 1 and 3 have NaN — their initial_weights_logits must be UNCHANGED
        np.testing.assert_array_equal(
            np.array(result["initial_weights_logits"][1]),
            np.array(original["initial_weights_logits"][1]),
            err_msg="initial_weights_logits[1] should NOT be replaced even though set 1 has NaN",
        )
        np.testing.assert_array_equal(
            np.array(result["initial_weights_logits"][3]),
            np.array(original["initial_weights_logits"][3]),
            err_msg="initial_weights_logits[3] should NOT be replaced even though set 3 has NaN",
        )


# ── calculate_period_metrics tests ───────────────────────────────────────────

# Pinned values captured from pre-refactor code (float64, test data).
MOMENTUM_PINNED = {
    "sharpe": 2.2391839871363963,
    "jax_sharpe": 2.1344858727490044,
    "daily_log_sharpe": 2.009381850687709,
    "return": 0.46374115737291843,
    "returns_over_hodl": -0.08834374131141964,
    "returns_over_uniform_hodl": -0.08834374131141964,
    "annualised_returns": 1.5116850237032837,
    "annualised_returns_over_hodl": -0.2003451297829858,
    "annualised_returns_over_uniform_hodl": -0.2003451297829858,
    "ulcer": -0.05910269573181366,
    "calmar": 6.774285392784477,
    "sterling": 11.41020135542993,
}

MEAN_REVERSION_PINNED = {
    "sharpe": 2.6990907503295687,
    "jax_sharpe": 2.686547390723503,
    "daily_log_sharpe": 2.466452678664176,
    "return": 0.6421280241088752,
    "returns_over_hodl": 0.02276026277326415,
    "returns_over_uniform_hodl": 0.02276026277326415,
    "annualised_returns": 2.3165627607181425,
    "annualised_returns_over_hodl": 0.05590690670214893,
    "annualised_returns_over_uniform_hodl": 0.05590690670214893,
    "ulcer": -0.06259363887300436,
    "calmar": 10.665322406355822,
    "sterling": 17.002481715411022,
}


class TestCalculatePeriodMetrics:
    """Test that period metrics produce stable, correct values."""

    def test_metrics_keys(self, momentum_result):
        results_dict = {
            "value": momentum_result["value"],
            "reserves": momentum_result["reserves"],
            "prices": momentum_result["prices"],
        }
        metrics = calculate_period_metrics(results_dict)
        expected_keys = set(MOMENTUM_PINNED.keys()) | {"daily_returns"}
        assert set(metrics.keys()) == expected_keys

    @pytest.mark.parametrize("metric_key", list(MOMENTUM_PINNED.keys()))
    def test_momentum_pinned_values(self, momentum_result, metric_key):
        """Each momentum metric must match its pinned value."""
        results_dict = {
            "value": momentum_result["value"],
            "reserves": momentum_result["reserves"],
            "prices": momentum_result["prices"],
        }
        metrics = calculate_period_metrics(results_dict)
        np.testing.assert_allclose(
            metrics[metric_key], MOMENTUM_PINNED[metric_key], rtol=RTOL,
            err_msg=f"Momentum metric '{metric_key}' drifted from pinned value",
        )

    @pytest.mark.parametrize("metric_key", list(MEAN_REVERSION_PINNED.keys()))
    def test_mean_reversion_pinned_values(self, mean_reversion_result, metric_key):
        """Each mean_reversion_channel metric must match its pinned value."""
        results_dict = {
            "value": mean_reversion_result["value"],
            "reserves": mean_reversion_result["reserves"],
            "prices": mean_reversion_result["prices"],
        }
        metrics = calculate_period_metrics(results_dict)
        np.testing.assert_allclose(
            metrics[metric_key], MEAN_REVERSION_PINNED[metric_key], rtol=RTOL,
            err_msg=f"Mean reversion metric '{metric_key}' drifted from pinned value",
        )

    def test_metrics_deterministic(self, momentum_result):
        results_dict = {
            "value": momentum_result["value"],
            "reserves": momentum_result["reserves"],
            "prices": momentum_result["prices"],
        }
        m1 = calculate_period_metrics(results_dict)
        m2 = calculate_period_metrics(results_dict)
        for key in m1:
            if key == "daily_returns":
                np.testing.assert_array_equal(m1[key], m2[key])
            else:
                assert m1[key] == m2[key], f"Metric '{key}' not deterministic"

    def test_metrics_with_separate_prices(self, momentum_result):
        """Passing prices via kwarg gives identical results to embedding in dict."""
        rd_embedded = {
            "value": momentum_result["value"],
            "reserves": momentum_result["reserves"],
            "prices": momentum_result["prices"],
        }
        rd_separate = {
            "value": momentum_result["value"],
            "reserves": momentum_result["reserves"],
        }
        m1 = calculate_period_metrics(rd_embedded)
        m2 = calculate_period_metrics(rd_separate, prices=momentum_result["prices"])
        for key in MOMENTUM_PINNED:
            assert m1[key] == m2[key], f"Mismatch for '{key}'"


# ── calculate_continuous_test_metrics tests ──────────────────────────────────


# Pinned continuous test metrics for 60/40 train/test split on momentum result.
# This catches slicing bugs even if both functions are wrong in the same way.
CONTINUOUS_TEST_PINNED = {
    "annualised_returns": -0.33299817740801574,
    "annualised_returns_over_hodl": -0.16431950930843398,
    "annualised_returns_over_uniform_hodl": -0.31052566691913,
    "calmar": -1.7058500423965588,
    "daily_log_sharpe": -1.2250993771346774,
    "jax_sharpe": -0.7540335005797353,
    "return": -0.06481621004830218,
    "returns_over_hodl": -0.029267849012868163,
    "returns_over_uniform_hodl": -0.059674158144114586,
    "sharpe": -1.0286642910253756,
    "sterling": -2.3937949129490375,
    "ulcer": -0.07350442116690786,
}


class TestCalculateContinuousTestMetrics:
    """Test that continuous test metrics slice correctly and produce correct values."""

    def test_continuous_test_metrics_match_manual_slice(self, momentum_result):
        """Continuous test metrics on a slice should match calculate_period_metrics
        on the same manually-sliced data."""
        value = momentum_result["value"]
        reserves = momentum_result["reserves"]
        prices = momentum_result["prices"]
        T = len(value)

        # Split at 60% for "train", rest for "test"
        train_len = int(T * 0.6)
        test_len = T - train_len

        continuous_dict = {"value": value, "reserves": reserves}
        continuous_metrics = calculate_continuous_test_metrics(
            continuous_dict, train_len, test_len, prices,
        )

        manual_dict = {
            "value": value[train_len:train_len + test_len],
            "reserves": reserves[train_len:train_len + test_len],
            "prices": prices[train_len:train_len + test_len],
        }
        manual_metrics = calculate_period_metrics(manual_dict)

        for key in MOMENTUM_PINNED:
            np.testing.assert_allclose(
                continuous_metrics[key], manual_metrics[key], rtol=RTOL,
                err_msg=f"Continuous vs manual slice mismatch for '{key}'",
            )

    @pytest.mark.parametrize("metric_key", list(CONTINUOUS_TEST_PINNED.keys()))
    def test_continuous_test_pinned_values(self, momentum_result, metric_key):
        """Each continuous test metric must match its pinned value.

        This catches bugs where both calculate_continuous_test_metrics and
        calculate_period_metrics are wrong in the same way (e.g., off-by-one
        in slicing that the consistency test above wouldn't catch).
        """
        value = momentum_result["value"]
        reserves = momentum_result["reserves"]
        prices = momentum_result["prices"]
        T = len(value)
        train_len = int(T * 0.6)
        test_len = T - train_len

        continuous_dict = {"value": value, "reserves": reserves}
        metrics = calculate_continuous_test_metrics(
            continuous_dict, train_len, test_len, prices,
        )
        np.testing.assert_allclose(
            metrics[metric_key], CONTINUOUS_TEST_PINNED[metric_key], rtol=RTOL,
            err_msg=f"Continuous test metric '{metric_key}' drifted from pinned value",
        )


# ── _calculate_return_value tests ─────────────────────────────────────────────


class TestCalculateReturnValue:
    """Test individual metric computations for finiteness and consistency."""

    @pytest.mark.parametrize("metric_name", [
        "sharpe", "returns", "returns_over_hodl", "daily_log_sharpe",
        "annualised_returns", "ulcer", "calmar", "sterling",
        "returns_over_uniform_hodl", "annualised_returns_over_hodl",
        "annualised_returns_over_uniform_hodl",
    ])
    def test_individual_metrics_finite(self, momentum_result, metric_name):
        reserves = momentum_result["reserves"]
        prices = momentum_result["prices"]
        value = momentum_result["value"]

        needs_initial = metric_name in [
            "returns_over_hodl", "returns_over_uniform_hodl",
            "annualised_returns_over_hodl", "annualised_returns_over_uniform_hodl",
            "ulcer", "calmar", "sterling",
        ]
        kwargs = {"initial_reserves": reserves[0]} if needs_initial else {}

        result = _calculate_return_value(
            metric_name, reserves, prices, value, **kwargs
        )
        assert jnp.isfinite(result), f"Metric '{metric_name}' not finite: {result}"


# ── Training gradient regression ──────────────────────────────────────────────

# ── Gradient pinning ──────────────────────────────────────────────────────────

# Pinned gradients from a single forward pass + value_and_grad.
# These directly verify the gradient computation hasn't changed.
MOMENTUM_GRAD_PINNED = {
    "objective": 1.9889377911120243,
    "grads": {
        "initial_weights_logits": [0.0, 0.0],
        "log_k": [-0.031844837485746444, -0.03184483748574646],
        "logit_lamb": [-0.3434598753488577, 0.34015511094382833],
    },
}

MEAN_REVERSION_GRAD_PINNED = {
    "objective": 2.4712922933760324,
    "grads": {
        "initial_weights_logits": [0.0, 0.0],
        "log_amplitude": [-0.3426686539635894, 0.29756526427259855],
        "log_k": [-0.02255169250900112, -0.02255169250900117],
        "logit_lamb": [1.4715955399584866, -1.054051430795321],
        "raw_exponents": [-4.22632651436977e-08, 1.5820578824625324e-08],
        "raw_pre_exp_scaling": [-1.169801604450293e-08, 4.136961722473516e-09],
        "raw_width": [0.3426341167623295, -0.29755034271355185],
    },
}


def _setup_gradient_test(rule, params):
    """Build the same differentiable objective the training loop uses.

    Replicates: Partial(forward_pass, ...) -> batched_objective_factory -> value_and_grad.
    """
    from jax.tree_util import Partial
    from quantammsim.pools.creator import create_pool
    from quantammsim.core_simulator.forward_pass import forward_pass
    from quantammsim.runners.jax_runner_utils import Hashabledict
    from quantammsim.runners.jax_runners import (
        get_data_dict, get_unique_tokens, get_sig_variations,
        create_static_dict, recursive_default_set, run_fingerprint_defaults,
    )
    from quantammsim.training.backpropagation import (
        batched_partial_training_step_factory, batched_objective_factory,
    )

    fp = {
        "tokens": ["BTC", "ETH"],
        "rule": rule,
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-06-01 00:00:00",
        "endTestDateString": "2023-06-15 00:00:00",
        "initial_pool_value": 1000000.0,
        "do_arb": True,
        "arb_quality": 1.0,
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "return_val": "daily_log_sharpe",
        "minimum_weight": 0.03,
        "max_memory_days": 30,
        "bout_offset": 10080,
        "subsidary_pools": [],
    }
    recursive_default_set(fp, run_fingerprint_defaults)

    unique_tokens = get_unique_tokens(fp)
    n_assets = len(fp["tokens"])
    all_sig_variations = get_sig_variations(n_assets)

    data_dict = get_data_dict(
        unique_tokens, fp,
        data_kind="historic",
        root=str(TEST_DATA_DIR),
        max_memory_days=fp["max_memory_days"],
        start_date_string=fp["startDateString"],
        end_time_string=fp["endDateString"],
        start_time_test_string=fp["endDateString"],
        end_time_test_string=fp["endTestDateString"],
        max_mc_version=9,
    )

    pool = create_pool(rule)
    static_dict = create_static_dict(
        fp,
        bout_length=data_dict["bout_length"],
        all_sig_variations=all_sig_variations,
        overrides={"n_assets": n_assets, "training_data_kind": "historic", "do_trades": False},
    )

    partial_training_step = Partial(
        forward_pass,
        prices=data_dict["prices"],
        static_dict=Hashabledict(static_dict),
        pool=pool,
    )
    batched_step = batched_partial_training_step_factory(partial_training_step)
    batched_obj = batched_objective_factory(batched_step)

    start_indexes = jnp.array([[data_dict["start_idx"], 0]])
    return batched_obj, start_indexes


class TestGradientPinnedValues:
    """Pin exact gradient values from value_and_grad through the forward pass.

    This is the most direct test of gradient correctness. If the gradient
    computation changes (e.g., from modifying update_factory_with_optax,
    changing the forward pass, or altering how value_and_grad is called),
    these tests will catch it.
    """

    @pytest.fixture(scope="class")
    def momentum_grad_result(self):
        from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
            ]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }
        batched_obj, start_indexes = _setup_gradient_test("momentum", params)
        obj_val, grads = jax.value_and_grad(batched_obj)(params, start_indexes)
        return float(obj_val), {k: np.array(grads[k]).tolist() for k in grads}

    @pytest.fixture(scope="class")
    def mr_grad_result(self):
        from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
            ]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "log_amplitude": jnp.array([0.0, 0.0]),
            "raw_width": jnp.array([0.0, 0.0]),
            "raw_exponents": jnp.array([1.0, 1.0]),
            "raw_pre_exp_scaling": jnp.array([0.5, 0.5]),
        }
        batched_obj, start_indexes = _setup_gradient_test("mean_reversion_channel", params)
        obj_val, grads = jax.value_and_grad(batched_obj)(params, start_indexes)
        return float(obj_val), {k: np.array(grads[k]).tolist() for k in grads}

    def test_momentum_objective_pinned(self, momentum_grad_result):
        obj_val, _ = momentum_grad_result
        np.testing.assert_allclose(
            obj_val, MOMENTUM_GRAD_PINNED["objective"], rtol=RTOL,
            err_msg="Momentum forward pass objective drifted",
        )

    @pytest.mark.parametrize("key", list(MOMENTUM_GRAD_PINNED["grads"].keys()))
    def test_momentum_gradient_pinned(self, momentum_grad_result, key):
        _, grads = momentum_grad_result
        np.testing.assert_allclose(
            grads[key], MOMENTUM_GRAD_PINNED["grads"][key], rtol=RTOL,
            err_msg=f"Momentum gradient for '{key}' drifted",
        )

    def test_mr_objective_pinned(self, mr_grad_result):
        obj_val, _ = mr_grad_result
        np.testing.assert_allclose(
            obj_val, MEAN_REVERSION_GRAD_PINNED["objective"], rtol=RTOL,
            err_msg="Mean reversion forward pass objective drifted",
        )

    @pytest.mark.parametrize("key", list(MEAN_REVERSION_GRAD_PINNED["grads"].keys()))
    def test_mr_gradient_pinned(self, mr_grad_result, key):
        _, grads = mr_grad_result
        np.testing.assert_allclose(
            grads[key], MEAN_REVERSION_GRAD_PINNED["grads"][key], rtol=RTOL,
            err_msg=f"Mean reversion gradient for '{key}' drifted",
        )

    def test_gradient_keys_match_params(self, momentum_grad_result, mr_grad_result):
        """Gradient tree must have exactly the same keys as params."""
        _, mom_grads = momentum_grad_result
        _, mr_grads = mr_grad_result
        assert set(mom_grads.keys()) == set(MOMENTUM_GRAD_PINNED["grads"].keys())
        assert set(mr_grads.keys()) == set(MEAN_REVERSION_GRAD_PINNED["grads"].keys())


# ── Training loop regression ──────────────────────────────────────────────────

# Pinned from pre-refactor code.
PINNED_TRAINING_OBJECTIVE = 11.12668681990391
PINNED_MR_TRAINING_OBJECTIVE = 9.962990368217547


def _make_training_fingerprint(rule="momentum"):
    """Minimal training config for regression tests.

    Supports both momentum (3 param keys) and mean_reversion_channel (7 param keys)
    to catch bugs that only manifest with larger param dicts.
    """
    fp = {
        "tokens": ["BTC", "ETH"],
        "rule": rule,
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-02-01 00:00:00",
        "endTestDateString": "2023-02-15 00:00:00",
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1000000.0,
        "fees": 0,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "do_arb": True,
        "arb_quality": 1.0,
        "arb_frequency": 1,
        "bout_offset": 10080,
        "return_val": "daily_log_sharpe",
        "minimum_weight": 0.03,
        "max_memory_days": 30,
        "subsidary_pools": [],
        "optimisation_settings": {
            "method": "gradient_descent",
            "base_lr": 0.05,
            "optimiser": "adam",
            "batch_size": 2,
            "n_iterations": 3,
            "n_parameter_sets": 2,
            "training_data_kind": "historic",
            "train_on_hessian_trace": False,
            "use_gradient_clipping": True,
            "sample_method": "uniform",
            "initial_random_key": 42,
            "n_cycles": 1,
            "decay_lr_ratio": 0.8,
            "decay_lr_plateau": 200,
            "min_lr": 1e-6,
            "include_flipped_training_data": False,
            "max_mc_version": 9,
            "force_scalar": False,
        },
    }
    return fp


class TestTrainingGradientRegression:
    """Test that training produces stable, pinned results."""

    @pytest.fixture(scope="class")
    def training_result(self):
        from quantammsim.runners.jax_runners import train_on_historic_data
        _, metadata = train_on_historic_data(
            _make_training_fingerprint(),
            root=str(TEST_DATA_DIR),
            verbose=False,
            force_init=True,
            return_training_metadata=True,
            iterations_per_print=999999,
        )
        return metadata

    def test_training_completes(self, training_result):
        assert training_result["epochs_trained"] >= 3

    def test_objective_is_finite(self, training_result):
        assert np.isfinite(training_result["final_objective"])

    def test_objective_matches_pinned(self, training_result):
        """Final objective must match pre-refactor pinned value."""
        np.testing.assert_allclose(
            training_result["final_objective"],
            PINNED_TRAINING_OBJECTIVE,
            rtol=RTOL,
            err_msg="Training objective drifted from pinned value",
        )

    def test_training_deterministic(self):
        """Two runs with same config and seed produce identical objective."""
        from quantammsim.runners.jax_runners import train_on_historic_data

        _, meta1 = train_on_historic_data(
            _make_training_fingerprint(),
            root=str(TEST_DATA_DIR),
            verbose=False,
            force_init=True,
            return_training_metadata=True,
            iterations_per_print=999999,
        )
        _, meta2 = train_on_historic_data(
            _make_training_fingerprint(),
            root=str(TEST_DATA_DIR),
            verbose=False,
            force_init=True,
            return_training_metadata=True,
            iterations_per_print=999999,
        )
        assert meta1["final_objective"] == meta2["final_objective"], (
            f"Non-deterministic: {meta1['final_objective']} vs {meta2['final_objective']}"
        )


class TestMeanReversionTrainingRegression:
    """Training regression for mean_reversion_channel — the actual tuning target.

    The momentum tests above use only 3 param keys. Mean_reversion_channel has 7
    (log_k, logit_lamb, initial_weights_logits, log_amplitude, raw_width,
    raw_exponents, raw_pre_exp_scaling). This catches bugs in:
    - Gradient tree structure (value_and_grad with 7-key dict)
    - nan_param_reinit vectorization over more keys
    - update_factory_with_optax return tuple with larger grad tree
    """

    @pytest.fixture(scope="class")
    def mr_training_result(self):
        from quantammsim.runners.jax_runners import train_on_historic_data
        _, metadata = train_on_historic_data(
            _make_training_fingerprint(rule="mean_reversion_channel"),
            root=str(TEST_DATA_DIR),
            verbose=False,
            force_init=True,
            return_training_metadata=True,
            iterations_per_print=999999,
        )
        return metadata

    def test_mr_training_completes(self, mr_training_result):
        assert mr_training_result["epochs_trained"] >= 3

    def test_mr_objective_is_finite(self, mr_training_result):
        assert np.isfinite(mr_training_result["final_objective"])

    def test_mr_objective_matches_pinned(self, mr_training_result):
        """Mean reversion training objective must match pre-refactor pinned value."""
        np.testing.assert_allclose(
            mr_training_result["final_objective"],
            PINNED_MR_TRAINING_OBJECTIVE,
            rtol=RTOL,
            err_msg="Mean reversion training objective drifted from pinned value",
        )
