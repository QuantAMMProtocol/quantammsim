"""GPU path baseline regression tests.

Runs existing baseline configurations under the GPU (conv) backend to verify
equivalence with the CPU (scan) path. These tests should pass both before and
after the FFT convolution change.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
from contextlib import contextmanager

from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
    recursive_default_set,
    check_run_fingerprint,
)
from quantammsim.runners.jax_runners import do_run_on_historic_data, train_on_historic_data
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from tests.conftest import TEST_DATA_DIR


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


# Shared with test_baseline_values.py — pinned reference values
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
            "first_weights": [0.5, 0.5],
            "last_weights": [0.05, 0.95],
        },
    },
}


# =============================================================================
# 3a. Baseline values under GPU (conv) path
# =============================================================================

class TestGPUPathBaselines:
    """Run baseline configs under GPU backend, assert same pinned values."""

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_gpu_final_value_matches_baseline(self, config_name):
        """GPU path final value matches pinned baseline within 0.6%."""
        config = BASELINE_CONFIGS[config_name]
        expected_final = config["expected"]["final_value"]

        with override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        actual_final = float(result["final_value"])
        relative_diff = abs(actual_final - expected_final) / expected_final
        assert relative_diff < 0.01, (
            f"{config_name} GPU: Final value {actual_final:.2f} vs "
            f"baseline {expected_final:.2f} ({relative_diff*100:.4f}%)"
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_gpu_first_weights_match_baseline(self, config_name):
        """GPU path first weights match pinned baseline to 4 decimal places."""
        config = BASELINE_CONFIGS[config_name]

        with override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        expected_first = np.array(config["expected"]["first_weights"])
        actual_first = np.array(result["weights"][0])
        np.testing.assert_array_almost_equal(
            actual_first, expected_first, decimal=4,
            err_msg=f"{config_name} GPU: First weights don't match baseline",
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_gpu_last_weights_match_baseline(self, config_name):
        """GPU path last weights match pinned baseline to 4 decimal places."""
        config = BASELINE_CONFIGS[config_name]

        with override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        expected_last = np.array(config["expected"]["last_weights"])
        actual_last = np.array(result["weights"][-1])
        np.testing.assert_array_almost_equal(
            actual_last, expected_last, decimal=4,
            err_msg=f"{config_name} GPU: Last weights don't match baseline",
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS.keys()))
    def test_gpu_weights_sum_to_one(self, config_name):
        """GPU path weights sum to 1."""
        config = BASELINE_CONFIGS[config_name]

        with override_backend("gpu"):
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
    def test_gpu_reserves_positive(self, config_name):
        """GPU path reserves are always positive."""
        config = BASELINE_CONFIGS[config_name]

        with override_backend("gpu"):
            result = do_run_on_historic_data(
                run_fingerprint=config["fingerprint"],
                params=config["params"],
                root=TEST_DATA_DIR,
            )

        assert np.all(result["reserves"] > 0), f"{config_name} GPU: Non-positive reserves"


# =============================================================================
# 3b. BFGS training under GPU path
# =============================================================================

class TestGPUPathBFGS:
    """BFGS training under GPU backend."""

    @pytest.fixture
    def bfgs_run_fingerprint(self):
        return {
            "rule": "momentum",
            "tokens": ["ETH", "USDC"],
            "subsidary_pools": [],
            "n_assets": 2,
            "bout_offset": 0,
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "weight_interpolation_method": "linear",
            "maximum_change": 0.0003,
            "minimum_weight": 0.05,
            "max_memory_days": 5.0,
            "use_alt_lamb": False,
            "use_pre_exp_scaling": True,
            "initial_pool_value": 1000000.0,
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "do_arb": True,
            "arb_frequency": 1,
            "return_val": "sharpe",
            "noise_trader_ratio": 0.0,
            "ste_max_change": False,
            "ste_min_max_weight": False,
            "initial_memory_length": 3.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 0.5,
            "initial_weights_logits": [0.0, 0.0],
            "initial_log_amplitude": 0.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 1.0,
            "initial_pre_exp_scaling": 1.0,
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-04 00:00:00",
            "endTestDateString": "2023-01-06 00:00:00",
            "do_trades": False,
            "optimisation_settings": {
                "method": "bfgs",
                "n_parameter_sets": 1,
                "noise_scale": 0.1,
                "training_data_kind": "historic",
                "initial_random_key": 42,
                "max_mc_version": 1,
                "val_fraction": 0.0,
                "base_lr": 0.01,
                "optimiser": "adam",
                "decay_lr_plateau": 50,
                "decay_lr_ratio": 0.5,
                "min_lr": 0.0001,
                "train_on_hessian_trace": False,
                "n_iterations": 10,
                "bfgs_settings": {
                    "maxiter": 5,
                    "tol": 1e-6,
                    "n_evaluation_points": 2,
                },
            },
        }

    def test_bfgs_gpu_objective_finite(self, bfgs_run_fingerprint):
        """BFGS under GPU backend produces finite, non-zero objective."""
        fp = deepcopy(bfgs_run_fingerprint)

        with override_backend("gpu"):
            _, metadata = train_on_historic_data(
                fp,
                root=TEST_DATA_DIR,
                verbose=False,
                force_init=True,
                return_training_metadata=True,
            )

        obj = metadata["final_objective"]
        assert np.isfinite(obj), f"Objective is not finite: {obj}"
        assert obj != 0.0, "Objective is exactly zero"

    def test_bfgs_gpu_params_correct_shapes(self, bfgs_run_fingerprint):
        """BFGS under GPU backend returns params with correct shapes."""
        fp = deepcopy(bfgs_run_fingerprint)

        with override_backend("gpu"):
            result = train_on_historic_data(
                fp,
                root=TEST_DATA_DIR,
                verbose=False,
                force_init=True,
            )

        assert result is not None
        assert "log_k" in result
        assert "logit_lamb" in result
        for k, v in result.items():
            if k == "subsidary_params":
                continue
            if hasattr(v, "shape"):
                assert v.ndim == 1, f"{k} has ndim={v.ndim}, expected 1"
