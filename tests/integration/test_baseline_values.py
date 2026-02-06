"""
Baseline regression tests for demo_run_.py configurations.

These tests capture ground truth values from known-good runs to protect
against regressions during refactoring. If these tests fail after a change,
either:
1. The change introduced a bug (fix it)
2. The change intentionally altered behavior (update the baselines)
"""

import pytest
import jax.numpy as jnp
import numpy as np
from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb
from quantammsim.runners.jax_runners import do_run_on_historic_data
from tests.conftest import TEST_DATA_DIR


# Baseline values - to be recaptured after date range update
# These should only be updated intentionally when behavior changes are expected

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
        "expected": {
            "final_value": 1815422.5738306814,
            "return_pct": 81.54225738306813,
            # First and last weights for regression check
            "first_weights": [0.6632375, 0.31110132, 0.02566118],
            "last_weights": [0.03333333, 0.45499836, 0.51166831],
        },
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
        "expected": {
            "final_value": 1500094.138254407,
            "return_pct": 50.00941382544071,
            "first_weights": [0.5, 0.5],
            "last_weights": [0.05000921, 0.94999079],
        },
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
        "expected": {
            "final_value": 1368731.4974473487,
            "return_pct": 36.87314974473486,
            "first_weights": [0.5, 0.5],
            "last_weights": [0.05, 0.95],
        },
    },
}


class TestBaselineValues:
    """Test that simulation outputs match known baseline values."""

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_final_value_matches_baseline(self, config_name):
        """Test that final pool value matches baseline within tolerance."""
        config = BASELINE_CONFIGS[config_name]
        expected_final = config["expected"]["final_value"]

        if expected_final is None:
            pytest.skip("Baseline value not yet captured")

        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
            root=TEST_DATA_DIR,
        )

        actual_final = float(result["final_value"])

        # Allow 0.5% tolerance for cross-platform floating point differences
        # (different CPUs/BLAS implementations can cause small variations)
        relative_diff = abs(actual_final - expected_final) / expected_final
        assert relative_diff < 0.006, (
            f"{config_name}: Final value {actual_final:.2f} differs from "
            f"baseline {expected_final:.2f} by {relative_diff*100:.4f}%"
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_return_matches_baseline(self, config_name):
        """Test that return percentage matches baseline."""
        config = BASELINE_CONFIGS[config_name]
        expected_return = config["expected"]["return_pct"]

        if expected_return is None:
            pytest.skip("Baseline value not yet captured")

        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
            root=TEST_DATA_DIR,
        )

        actual_return = (result["final_value"] / result["value"][0] - 1) * 100

        # Allow 1% absolute tolerance for cross-platform floating point differences
        # (different CPUs/BLAS implementations can cause small variations that compound)
        assert abs(actual_return - expected_return) < 1.0, (
            f"{config_name}: Return {actual_return:.2f}% differs from "
            f"baseline {expected_return:.2f}%"
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_first_weights_match_baseline(self, config_name):
        """Test that initial weights match baseline."""
        config = BASELINE_CONFIGS[config_name]

        if config["expected"]["first_weights"] is None:
            pytest.skip("Baseline value not yet captured")

        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
            root=TEST_DATA_DIR,
        )

        expected_first = np.array(config["expected"]["first_weights"])
        actual_first = np.array(result["weights"][0])

        np.testing.assert_array_almost_equal(
            actual_first, expected_first, decimal=4,
            err_msg=f"{config_name}: First weights don't match baseline"
        )

    @pytest.mark.parametrize("config_name", [
        k for k, v in BASELINE_CONFIGS.items()
        if "last_weights" in v["expected"]
    ])
    def test_last_weights_match_baseline(self, config_name):
        """Test that final weights match baseline."""
        config = BASELINE_CONFIGS[config_name]

        if config["expected"]["last_weights"] is None:
            pytest.skip("Baseline value not yet captured")

        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
            root=TEST_DATA_DIR,
        )

        expected_last = np.array(config["expected"]["last_weights"])
        actual_last = np.array(result["weights"][-1])

        np.testing.assert_array_almost_equal(
            actual_last, expected_last, decimal=4,
            err_msg=f"{config_name}: Last weights don't match baseline"
        )

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1 for all configurations."""
        for config_name, config in BASELINE_CONFIGS.items():
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

            weight_sums = np.sum(result["weights"], axis=1)
            np.testing.assert_array_almost_equal(
                weight_sums, np.ones_like(weight_sums), decimal=6,
                err_msg=f"{config_name}: Weights don't sum to 1"
            )

    def test_reserves_positive(self):
        """Test that reserves are always positive."""
        for config_name, config in BASELINE_CONFIGS.items():
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

            assert np.all(result["reserves"] > 0), (
                f"{config_name}: Found non-positive reserves"
            )


class TestDifferentPoolTypes:
    """Test baseline values for different pool types."""

    def test_balancer_pool_static_weights(self):
        """Test that balancer pool maintains constant weights."""
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

        result = do_run_on_historic_data(
            run_fingerprint=fingerprint,
            params=params,
            root=TEST_DATA_DIR,
        )

        # All weights should be [0.5, 0.5] for balancer with equal logits
        expected_weights = np.array([0.5, 0.5])
        for i, weights in enumerate(result["weights"]):
            np.testing.assert_array_almost_equal(
                weights, expected_weights, decimal=6,
                err_msg=f"Balancer weights at step {i} are not constant"
            )


class TestPowerChannelBaseline:
    """Test baseline values for power channel pool."""

    def test_power_channel_pool_runs(self):
        """Test that power channel pool produces valid output."""
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

        result = do_run_on_historic_data(
            run_fingerprint=fingerprint,
            params=params,
            root=TEST_DATA_DIR,
        )

        # Basic sanity checks
        assert result["final_value"] > 0
        assert np.all(result["weights"] >= 0)
        assert np.all(result["weights"] <= 1)
        weight_sums = np.sum(result["weights"], axis=1)
        np.testing.assert_array_almost_equal(
            weight_sums, np.ones_like(weight_sums), decimal=6
        )


class TestMeanReversionBaseline:
    """Test baseline values for mean reversion channel pool."""

    def test_mean_reversion_pool_runs(self):
        """Test that mean reversion pool produces valid output."""
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

        result = do_run_on_historic_data(
            run_fingerprint=fingerprint,
            params=params,
            root=TEST_DATA_DIR,
        )

        # Basic sanity checks
        assert result["final_value"] > 0
        assert np.all(result["weights"] >= 0)
        assert np.all(result["weights"] <= 1)
        weight_sums = np.sum(result["weights"], axis=1)
        np.testing.assert_array_almost_equal(
            weight_sums, np.ones_like(weight_sums), decimal=6
        )
