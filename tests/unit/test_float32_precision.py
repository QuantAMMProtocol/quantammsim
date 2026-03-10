"""Tests for float32 computation: precision vs float64 and dtype propagation.

Validates that running the estimator primitives and forward pass in float32
produces results within acceptable tolerance of float64, and that hardcoded
float64 sites don't silently upcast float32 inputs.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from jax import random
from copy import deepcopy
from contextlib import contextmanager

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    make_ewma_kernel,
    make_a_kernel,
    _jax_ewma_at_infinity_via_conv_1D,
    _jax_gradients_at_infinity_via_conv_1D_padded,
    _jax_variance_at_infinity_via_conv_1D,
    _jax_gradients_at_infinity_via_scan,
    _jax_variance_at_infinity_via_scan,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_ewma_pair,
    calc_gradients,
    calc_return_variances,
)
from quantammsim.runners.jax_runners import do_run_on_historic_data, train_on_historic_data
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import (
    recursive_default_set,
    check_run_fingerprint,
    memory_days_to_logit_lamb,
)
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


def generate_test_prices(key, n_timesteps=100, n_assets=3):
    """Generate test price data."""
    key1, key2 = random.split(key)
    returns = random.normal(key1, (n_timesteps, n_assets)) * 0.01
    prices = jnp.exp(jnp.cumsum(returns, axis=0))
    prices = prices - jnp.min(prices) + 1.0
    return prices


# =============================================================================
# 2a. Estimator primitives: float32 vs float64 and dtype propagation
# =============================================================================

class TestFloat32EstimatorPrimitives:
    """Test that float32 inputs produce correct results and preserve dtype."""

    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(42)

    def test_make_ewma_kernel_float32(self):
        """make_ewma_kernel with float32 lamb matches float64 version."""
        lamb_f64 = jnp.array([0.99, 0.95], dtype=jnp.float64)
        lamb_f32 = lamb_f64.astype(jnp.float32)

        kernel_f64 = make_ewma_kernel(lamb_f64, 30, 1440)
        kernel_f32 = make_ewma_kernel(lamb_f32, 30, 1440)

        assert kernel_f64.shape == kernel_f32.shape
        np.testing.assert_allclose(
            np.array(kernel_f32), np.array(kernel_f64), rtol=1e-4,
            err_msg="EWMA kernel float32 vs float64",
        )

    def test_make_a_kernel_float32(self):
        """make_a_kernel with float32 lamb matches float64 version."""
        lamb_f64 = jnp.array([0.99, 0.95], dtype=jnp.float64)
        lamb_f32 = lamb_f64.astype(jnp.float32)

        kernel_f64 = make_a_kernel(lamb_f64, 30, 1440)
        kernel_f32 = make_a_kernel(lamb_f32, 30, 1440)

        assert kernel_f64.shape == kernel_f32.shape
        np.testing.assert_allclose(
            np.array(kernel_f32), np.array(kernel_f64), rtol=1e-4,
            err_msg="A kernel float32 vs float64",
        )

    def test_ewma_conv_float32_matches_float64(self, rng_key):
        """EWMA via conv with float32 inputs matches float64 within rtol=1e-4."""
        prices = generate_test_prices(rng_key, n_timesteps=200, n_assets=3)
        lamb_f64 = jnp.array([0.99, 0.95, 0.90], dtype=jnp.float64)

        kernel_f64 = make_ewma_kernel(lamb_f64, 30, 1440)
        kernel_f32 = make_ewma_kernel(lamb_f64.astype(jnp.float32), 30, 1440)

        ewma_f64 = _jax_ewma_at_infinity_via_conv_1D(prices[:, 0], kernel_f64[:, 0])
        ewma_f32 = _jax_ewma_at_infinity_via_conv_1D(
            prices[:, 0].astype(jnp.float32), kernel_f32[:, 0]
        )

        np.testing.assert_allclose(
            np.array(ewma_f32), np.array(ewma_f64), rtol=1e-4,
            err_msg="EWMA conv float32 vs float64",
        )

    def test_variance_scan_float32_matches_float64(self, rng_key):
        """Variance via scan with float32 inputs matches float64 within rtol=1e-3."""
        prices_f64 = generate_test_prices(rng_key, n_timesteps=200, n_assets=3)
        prices_f32 = prices_f64.astype(jnp.float32)
        lamb = jnp.array([0.99, 0.95, 0.90])

        var_f64 = _jax_variance_at_infinity_via_scan(prices_f64, lamb.astype(jnp.float64))
        var_f32 = _jax_variance_at_infinity_via_scan(prices_f32, lamb.astype(jnp.float32))

        # Skip first row (initialization)
        np.testing.assert_allclose(
            np.array(var_f32[1:]), np.array(var_f64[1:]), rtol=1e-3,
            err_msg="Variance scan float32 vs float64",
        )

    def test_gradient_scan_float32_matches_float64(self, rng_key):
        """Gradient scan with float32 inputs matches float64 within rtol=1e-3."""
        prices_f64 = generate_test_prices(rng_key, n_timesteps=200, n_assets=3)
        prices_f32 = prices_f64.astype(jnp.float32)
        lamb = jnp.array([0.99, 0.95, 0.90])

        grad_f64 = _jax_gradients_at_infinity_via_scan(prices_f64, lamb.astype(jnp.float64))
        grad_f32 = _jax_gradients_at_infinity_via_scan(prices_f32, lamb.astype(jnp.float32))

        np.testing.assert_allclose(
            np.array(grad_f32), np.array(grad_f64), rtol=1e-3, atol=1e-6,
            err_msg="Gradient scan float32 vs float64",
        )

    def test_output_dtype_matches_input(self, rng_key):
        """Output dtype of scan/conv functions matches input dtype (no silent upcasting)."""
        prices_f32 = generate_test_prices(rng_key, n_timesteps=100, n_assets=3).astype(jnp.float32)
        lamb_f32 = jnp.array([0.99, 0.95, 0.90], dtype=jnp.float32)

        grads = _jax_gradients_at_infinity_via_scan(prices_f32, lamb_f32)
        assert grads.dtype == jnp.float32, f"Gradient dtype {grads.dtype} != float32"

        variances = _jax_variance_at_infinity_via_scan(prices_f32, lamb_f32)
        assert variances.dtype == jnp.float32, f"Variance dtype {variances.dtype} != float32"


# =============================================================================
# 2b. Forward pass: float32 vs float64
# =============================================================================

BASELINE_CONFIGS_FOR_DTYPE = {
    "momentum_2asset": {
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
    },
    "momentum_3asset": {
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
    },
}


class TestFloat32ForwardPass:
    """Test that forward pass with float32-cast inputs matches float64 within tolerance."""

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS_FOR_DTYPE.keys()))
    def test_float32_forward_pass_matches_float64(self, config_name):
        """Forward pass with float32-cast params matches float64 within 1%."""
        config = BASELINE_CONFIGS_FOR_DTYPE[config_name]

        # Run float64 (baseline)
        result_f64 = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
            root=TEST_DATA_DIR,
        )

        # Cast params to float32
        params_f32 = {}
        for k, v in config["params"].items():
            if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating):
                params_f32[k] = v.astype(jnp.float32)
            else:
                params_f32[k] = v

        result_f32 = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=params_f32,
            root=TEST_DATA_DIR,
        )

        # Final value within 1%
        f64_val = float(result_f64["final_value"])
        f32_val = float(result_f32["final_value"])
        rel_diff = abs(f32_val - f64_val) / abs(f64_val)
        assert rel_diff < 0.01, (
            f"{config_name}: float32 final_value {f32_val:.2f} vs "
            f"float64 {f64_val:.2f} ({rel_diff*100:.2f}% diff)"
        )

        # Weights within atol=0.01
        np.testing.assert_allclose(
            np.array(result_f32["weights"]),
            np.array(result_f64["weights"]),
            atol=0.01,
            err_msg=f"{config_name}: float32 vs float64 weights",
        )

    @pytest.mark.parametrize("config_name", list(BASELINE_CONFIGS_FOR_DTYPE.keys()))
    def test_float32_weights_valid(self, config_name):
        """Float32 forward pass produces valid weights (sum=1, positive)."""
        config = BASELINE_CONFIGS_FOR_DTYPE[config_name]
        params_f32 = {}
        for k, v in config["params"].items():
            if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating):
                params_f32[k] = v.astype(jnp.float32)
            else:
                params_f32[k] = v

        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=params_f32,
            root=TEST_DATA_DIR,
        )

        weights = np.array(result["weights"])
        weight_sums = np.sum(weights, axis=1)
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-5, atol=1e-5)
        assert np.all(result["reserves"] > 0), "Float32 reserves should be positive"


# =============================================================================
# 2c. BFGS with float32
# =============================================================================

class TestBFGSFloat32:
    """Test BFGS optimization path with compute_dtype='float32'."""

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
                    "compute_dtype": "float32",
                },
            },
        }

    def test_bfgs_float32_runs_without_nan(self, bfgs_run_fingerprint):
        """BFGS with compute_dtype='float32' produces finite results."""
        fp = deepcopy(bfgs_run_fingerprint)

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

    def test_bfgs_float32_params_are_finite(self, bfgs_run_fingerprint):
        """Optimized params from float32 BFGS are all finite."""
        fp = deepcopy(bfgs_run_fingerprint)

        result = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
        )

        assert result is not None
        for k, v in result.items():
            if k == "subsidary_params":
                continue
            if hasattr(v, "shape"):
                assert jnp.all(jnp.isfinite(v)), f"Param {k} has non-finite values"

    def test_bfgs_float64_still_works(self, bfgs_run_fingerprint):
        """BFGS with compute_dtype='float64' still works (opt-out path)."""
        fp = deepcopy(bfgs_run_fingerprint)
        fp["optimisation_settings"]["bfgs_settings"]["compute_dtype"] = "float64"

        _, metadata = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
            return_training_metadata=True,
        )

        obj = metadata["final_objective"]
        assert np.isfinite(obj), f"Float64 BFGS objective is not finite: {obj}"
