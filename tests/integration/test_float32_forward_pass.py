"""Float32 forward pass integration tests.

Runs do_run_on_historic_data with x64 disabled so the entire forward pass
naturally runs in float32. Verifies results match the float64 baselines at
the same tight tolerances — proving float32 is sufficient for this workload.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from contextlib import contextmanager

from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb
from quantammsim.runners.jax_runners import do_run_on_historic_data
from tests.conftest import TEST_DATA_DIR


@contextmanager
def float32_mode():
    """Disable x64 so all JAX computation runs float32."""
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", True)


@contextmanager
def override_backend(backend):
    """Temporarily override the DEFAULT_BACKEND."""
    from quantammsim.pools.G3M.quantamm.update_rule_estimators import estimators
    original = estimators.DEFAULT_BACKEND
    estimators.DEFAULT_BACKEND = backend
    try:
        yield
    finally:
        estimators.DEFAULT_BACKEND = original


# Same baseline configs as test_baseline_values.py, with float64 reference values
BASELINE_CONFIGS = {
    "QuantAMM_momentum_pool_3_assets": {
        "fingerprint": {
            "tokens": ["BTC", "ETH", "SOL"],
            "rule": "momentum",
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "initial_pool_value": 1000000.0,
            "do_arb": True,
            "arb_quality": 1.0,
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "use_alt_lamb": False,
        },
        "params": {
            "log_k": jnp.array([5, 5, 5]),
            "logit_lamb": jnp.array([
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
            ]),
            "initial_weights_logits": jnp.array(
                [-0.41062212, -1.16763663, -3.66277593]
            ),
        },
        "expected_final_value": 1815422.5738306814,
        "expected_return_pct": 81.54225738306813,
        "expected_first_weights": [0.6632375, 0.31110132, 0.02566118],
        "expected_last_weights": [0.03333333, 0.45499836, 0.51166831],
    },
    "forward_pass_test_1": {
        "fingerprint": {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "maximum_change": 1.0,
            "do_arb": True,
        },
        "params": {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([-0.22066515, -0.22066515]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        },
        "expected_final_value": 1500094.138254407,
        "expected_return_pct": 50.00941382544071,
        "expected_first_weights": [0.5, 0.5],
        "expected_last_weights": [0.05000921, 0.94999079],
    },
    "forward_pass_test_2": {
        "fingerprint": {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "maximum_change": 1.0,
            "do_arb": True,
        },
        "params": {
            "log_k": jnp.array([7.0, 7.0]),
            "logit_lamb": jnp.array([2.02840786, 2.02840786]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        },
        "expected_final_value": 1368731.4974473487,
        "expected_return_pct": 36.87314974473486,
        "expected_first_weights": [0.5, 0.5],
        "expected_last_weights": [0.05, 0.95],
    },
}


# ============================================================================
# CPU path (scan) with float32 (x64 disabled)
# ============================================================================

class TestFloat32CPUPath:
    """Float32 forward pass on CPU (scan) path — same tolerances as float64 baselines."""

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_final_value_matches_baseline(self, config_name):
        """Float32 final value within 0.6% of float64 baseline."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode():
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        actual = float(result["final_value"])
        expected = config["expected_final_value"]
        rel_diff = abs(actual - expected) / expected
        assert rel_diff < 0.006, (
            f"{config_name} f32 CPU: final value {actual:.2f} vs "
            f"f64 baseline {expected:.2f} ({rel_diff*100:.4f}%)"
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_return_matches_baseline(self, config_name):
        """Float32 return pct within 1% absolute of float64 baseline."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode():
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        actual_return = (result["final_value"] / result["value"][0] - 1) * 100
        expected_return = config["expected_return_pct"]
        assert abs(actual_return - expected_return) < 1.0, (
            f"{config_name} f32 CPU: return {actual_return:.2f}% vs "
            f"f64 baseline {expected_return:.2f}%"
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_first_weights_match_baseline(self, config_name):
        """Float32 first weights match float64 baseline to 4 decimal places."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode():
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        expected = np.array(config["expected_first_weights"])
        actual = np.array(result["weights"][0])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=4,
            err_msg=f"{config_name} f32 CPU: first weights diverge from f64",
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_last_weights_match_baseline(self, config_name):
        """Float32 last weights match float64 baseline to 4 decimal places."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode():
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        expected = np.array(config["expected_last_weights"])
        actual = np.array(result["weights"][-1])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=4,
            err_msg=f"{config_name} f32 CPU: last weights diverge from f64",
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_weights_sum_to_one(self, config_name):
        """Float32 weights sum to 1."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode():
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        weight_sums = np.sum(result["weights"], axis=1)
        np.testing.assert_array_almost_equal(
            weight_sums, np.ones_like(weight_sums), decimal=6,
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_reserves_positive(self, config_name):
        """Float32 reserves always positive."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode():
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        assert np.all(result["reserves"] > 0), (
            f"{config_name} f32 CPU: non-positive reserves"
        )


# ============================================================================
# GPU path (conv/FFT) with float32 (x64 disabled)
# ============================================================================

class TestFloat32GPUPath:
    """Float32 forward pass on GPU (conv) path — same tolerances as float64 baselines."""

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_final_value_matches_baseline(self, config_name):
        """Float32 GPU final value within 0.6% of float64 baseline."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode(), override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        actual = float(result["final_value"])
        expected = config["expected_final_value"]
        rel_diff = abs(actual - expected) / expected
        assert rel_diff < 0.006, (
            f"{config_name} f32 GPU: final value {actual:.2f} vs "
            f"f64 baseline {expected:.2f} ({rel_diff*100:.4f}%)"
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_return_matches_baseline(self, config_name):
        """Float32 GPU return pct within 1% absolute of float64 baseline."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode(), override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        actual_return = (result["final_value"] / result["value"][0] - 1) * 100
        expected_return = config["expected_return_pct"]
        assert abs(actual_return - expected_return) < 1.0, (
            f"{config_name} f32 GPU: return {actual_return:.2f}% vs "
            f"f64 baseline {expected_return:.2f}%"
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_first_weights_match_baseline(self, config_name):
        """Float32 GPU first weights match float64 baseline to 4 decimal places."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode(), override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        expected = np.array(config["expected_first_weights"])
        actual = np.array(result["weights"][0])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=4,
            err_msg=f"{config_name} f32 GPU: first weights diverge from f64",
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_last_weights_match_baseline(self, config_name):
        """Float32 GPU last weights match float64 baseline to 4 decimal places."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode(), override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        expected = np.array(config["expected_last_weights"])
        actual = np.array(result["weights"][-1])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=4,
            err_msg=f"{config_name} f32 GPU: last weights diverge from f64",
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_weights_sum_to_one(self, config_name):
        """Float32 GPU weights sum to 1."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode(), override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        weight_sums = np.sum(result["weights"], axis=1)
        np.testing.assert_array_almost_equal(
            weight_sums, np.ones_like(weight_sums), decimal=6,
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_reserves_positive(self, config_name):
        """Float32 GPU reserves always positive."""
        config = BASELINE_CONFIGS[config_name]

        with float32_mode(), override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        assert np.all(result["reserves"] > 0), (
            f"{config_name} f32 GPU: non-positive reserves"
        )


# ============================================================================
# Different pool types with float32
# ============================================================================

class TestFloat32PoolTypes:
    """Float32 forward pass for different pool types."""

    def _run_and_validate(self, fingerprint, params, backend=None):
        """Run forward pass with x64 disabled and check basic validity."""
        ctx = override_backend(backend) if backend else contextmanager(lambda: (yield))()

        with float32_mode(), ctx:
            result = do_run_on_historic_data(
                run_fingerprint=fingerprint,
                params=params,
                root=TEST_DATA_DIR,
            )

        assert result["final_value"] > 0, "Negative final value"
        weights = np.array(result["weights"])
        assert np.all(np.isfinite(weights)), "Non-finite weights"
        assert np.all(weights >= 0), "Negative weights"
        assert np.all(weights <= 1), "Weights > 1"
        if weights.ndim == 2:
            weight_sums = np.sum(weights, axis=1)
            np.testing.assert_array_almost_equal(
                weight_sums, np.ones_like(weight_sums), decimal=6,
            )
        assert np.all(np.array(result["reserves"]) > 0), "Non-positive reserves"
        return result

    @pytest.mark.parametrize("backend", [None, "gpu"])
    def test_balancer_pool_f32(self, backend):
        """Balancer pool works with float32."""
        fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "balancer",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
        }
        params = {
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }
        result = self._run_and_validate(fingerprint, params, backend)

        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(
            result["weights"][0], expected, decimal=6,
            err_msg="Balancer f32: weights not constant 50/50",
        )

    @pytest.mark.parametrize("backend", [None, "gpu"])
    def test_power_channel_pool_f32(self, backend):
        """Power channel pool works with float32."""
        fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "power_channel",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
        }
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([-0.22066515, -0.22066515]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "raw_exponents": jnp.array([1.0, 1.0]),
            "raw_pre_exp_scaling": jnp.array([0.5, 0.5]),
        }
        self._run_and_validate(fingerprint, params, backend)

    @pytest.mark.parametrize("backend", [None, "gpu"])
    def test_mean_reversion_channel_pool_f32(self, backend):
        """Mean reversion channel pool works with float32."""
        fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "mean_reversion_channel",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
        }
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([-0.22066515, -0.22066515]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "log_amplitude": jnp.array([0.0, 0.0]),
            "raw_width": jnp.array([0.0, 0.0]),
            "raw_exponents": jnp.array([1.0, 1.0]),
            "raw_pre_exp_scaling": jnp.array([0.5, 0.5]),
        }
        self._run_and_validate(fingerprint, params, backend)
