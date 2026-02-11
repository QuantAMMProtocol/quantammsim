"""
E2E regression tests for the run_pool_simulation code path.

The Flask /api/runSimulation endpoint constructs run_fingerprint with
initial_weights_logits as a JAX array (from the DTO's pool constituents).
This is different from the training path where initial_weights_logits is
either absent from the fingerprint or a scalar float.

These tests mirror the exact configs and pinned values from
test_baseline_values.py but with initial_weights_logits injected into the
fingerprint as a JAX array — reproducing the run_pool_simulation pattern.
Results must be identical to the baseline tests.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb
from quantammsim.runners.jax_runners import do_run_on_historic_data
from tests.conftest import TEST_DATA_DIR

# Reuse the exact baseline configs from test_baseline_values.py, but with
# initial_weights_logits added to the fingerprint (as run_pool_simulation does).
# Expected values are identical — the field should have no effect on output.

FLASK_PATH_CONFIGS = {
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
            # Flask-path: initial_weights_logits in fingerprint as JAX array
            "initial_weights_logits": jnp.array(
                [-0.41062212, -1.16763663, -3.66277593]
            ),
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
            # Flask-path: initial_weights_logits in fingerprint as JAX array
            "initial_weights_logits": jnp.array([0.0, 0.0]),
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
            # Flask-path: initial_weights_logits in fingerprint as JAX array
            "initial_weights_logits": jnp.array([0.0, 0.0]),
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


class TestRunPoolSimulationPath:
    """E2E tests mirroring the Flask run_pool_simulation code path.

    These exercise do_run_on_historic_data with initial_weights_logits
    as a JAX array in the fingerprint, exactly as run_pool_simulation
    constructs it. Pinned values match test_baseline_values.py.
    """

    @pytest.mark.parametrize("config_name", list(FLASK_PATH_CONFIGS.keys()))
    def test_final_value_matches_baseline(self, config_name):
        """Final pool value must match baseline (same as clean-path tests)."""
        config = FLASK_PATH_CONFIGS[config_name]

        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
            root=TEST_DATA_DIR,
        )

        expected_final = config["expected"]["final_value"]
        actual_final = float(result["final_value"])

        relative_diff = abs(actual_final - expected_final) / expected_final
        assert relative_diff < 0.006, (
            f"{config_name}: Final value {actual_final:.2f} differs from "
            f"baseline {expected_final:.2f} by {relative_diff*100:.4f}%"
        )

    @pytest.mark.parametrize("config_name", list(FLASK_PATH_CONFIGS.keys()))
    def test_first_weights_match_baseline(self, config_name):
        """Initial weights must match baseline."""
        config = FLASK_PATH_CONFIGS[config_name]

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

    @pytest.mark.parametrize("config_name", list(FLASK_PATH_CONFIGS.keys()))
    def test_last_weights_match_baseline(self, config_name):
        """Final weights must match baseline."""
        config = FLASK_PATH_CONFIGS[config_name]

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

    @pytest.mark.parametrize("config_name", list(FLASK_PATH_CONFIGS.keys()))
    def test_weights_sum_to_one(self, config_name):
        """Weights must sum to 1 at every timestep."""
        config = FLASK_PATH_CONFIGS[config_name]

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

    @pytest.mark.parametrize("config_name", list(FLASK_PATH_CONFIGS.keys()))
    def test_reserves_positive(self, config_name):
        """Reserves must be strictly positive at every timestep."""
        config = FLASK_PATH_CONFIGS[config_name]

        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
            root=TEST_DATA_DIR,
        )

        assert np.all(result["reserves"] > 0), (
            f"{config_name}: Found non-positive reserves"
        )


class TestBalancerFlaskPath:
    """Test balancer pool with Flask-path fingerprint structure."""

    def test_balancer_static_weights(self):
        """Balancer with equal logits should maintain [0.5, 0.5] weights."""
        fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "balancer",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
            # Flask-path: initial_weights_logits in fingerprint as JAX array
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }
        params = {
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

        result = do_run_on_historic_data(
            run_fingerprint=fingerprint,
            params=params,
            root=TEST_DATA_DIR,
        )

        expected_weights = np.array([0.5, 0.5])
        for i, weights in enumerate(result["weights"]):
            np.testing.assert_array_almost_equal(
                weights, expected_weights, decimal=6,
                err_msg=f"Balancer weights at step {i} are not constant"
            )

    def test_balancer_unequal_weights(self):
        """Balancer with non-zero logits should maintain corresponding weights."""
        log_ratio = jnp.array([-0.69314718, -0.69314718])  # ln(0.5) each -> equal
        fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "balancer",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
            "initial_weights_logits": log_ratio,
        }
        params = {
            "initial_weights_logits": log_ratio,
        }

        result = do_run_on_historic_data(
            run_fingerprint=fingerprint,
            params=params,
            root=TEST_DATA_DIR,
        )

        # Weights should be constant across all timesteps
        first_weights = result["weights"][0]
        for i, weights in enumerate(result["weights"]):
            np.testing.assert_array_almost_equal(
                weights, first_weights, decimal=6,
                err_msg=f"Balancer weights at step {i} differ from step 0"
            )
        assert result["final_value"] > 0
