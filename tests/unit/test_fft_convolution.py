"""Tests for FFT convolution and its equivalence with direct convolution.

Validates that _fft_convolve_1d produces identical results to jnp.convolve,
and that the GPU (conv) estimator path matches the CPU (scan) path both
before and after the FFT change.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from jax import random, jit, vmap
from contextlib import contextmanager

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _fft_convolve_1d,
    _fft_convolve_full,
    make_ewma_kernel,
    make_a_kernel,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_ewma_pair,
    calc_gradients,
    calc_return_variances,
)


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
    """Generate test price data with known properties."""
    key1, key2 = random.split(key)
    returns = random.normal(key1, (n_timesteps, n_assets)) * 0.01
    prices = jnp.exp(jnp.cumsum(returns, axis=0))
    prices = prices - jnp.min(prices) + 1.0
    return prices


# =============================================================================
# 1a. _fft_convolve_1d core accuracy
# =============================================================================

class TestFFTConvolve1D:
    """Core accuracy tests: FFT conv vs jnp.convolve."""

    @pytest.mark.parametrize("n_signal,n_kernel", [
        (10, 5),
        (100, 30),
        (200_000, 1825),
    ])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_full_mode_matches_direct(self, n_signal, n_kernel, dtype):
        """FFT full convolution matches jnp.convolve(mode='full')."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)
        x = random.normal(k1, (n_signal,)).astype(dtype)
        k = random.normal(k2, (n_kernel,)).astype(dtype)

        n_out = n_signal + n_kernel - 1
        fft_result = _fft_convolve_1d(x, k, n_out)
        direct_result = jnp.convolve(x, k, mode="full")

        # FFT and direct convolution have different rounding characteristics.
        # Large float32 convolutions accumulate more error; use atol to handle
        # near-zero values where rtol is meaningless.
        if dtype == jnp.float32:
            rtol = 1e-3 if n_signal > 10_000 else 5e-5
            atol = 1e-4 if n_signal > 10_000 else 0
        else:
            rtol = 1e-9 if n_signal > 10_000 else 1e-10
            atol = 0
        np.testing.assert_allclose(
            np.array(fft_result), np.array(direct_result), rtol=rtol, atol=atol,
            err_msg=f"Full-mode mismatch at ({n_signal}, {n_kernel}), {dtype}",
        )

    @pytest.mark.parametrize("n_signal,n_kernel", [
        (10, 5),
        (100, 30),
        (200_000, 1825),
    ])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_valid_mode_via_slicing(self, n_signal, n_kernel, dtype):
        """full[len(k)-1 : len(x)] matches jnp.convolve(mode='valid')."""
        key = random.PRNGKey(42)
        k1, k2 = random.split(key)
        x = random.normal(k1, (n_signal,)).astype(dtype)
        k = random.normal(k2, (n_kernel,)).astype(dtype)

        n_out = n_signal + n_kernel - 1
        full_conv = _fft_convolve_1d(x, k, n_out)
        fft_valid = full_conv[n_kernel - 1 : n_signal]
        direct_valid = jnp.convolve(x, k, mode="valid")

        if dtype == jnp.float32:
            rtol = 1e-3 if n_signal > 10_000 else 5e-5
            atol = 1e-4 if n_signal > 10_000 else 0
        else:
            rtol = 1e-9 if n_signal > 10_000 else 1e-10
            atol = 0
        np.testing.assert_allclose(
            np.array(fft_valid), np.array(direct_valid), rtol=rtol, atol=atol,
            err_msg=f"Valid-mode mismatch at ({n_signal}, {n_kernel}), {dtype}",
        )


# =============================================================================
# 1b. Estimator CPU/GPU equivalence (should pass before AND after FFT change)
# =============================================================================

class TestEstimatorCPUGPUEquivalence:
    """GPU (conv) path matches CPU (scan) path for each estimator."""

    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(0)

    @pytest.mark.parametrize("n_timesteps,max_mem", [
        (100, 30),
        (500, 60),
    ])
    def test_ewma_cpu_gpu_equivalence(self, rng_key, n_timesteps, max_mem):
        """EWMA via conv matches EWMA via scan."""
        prices = generate_test_prices(rng_key, n_timesteps, n_assets=3)
        mem_days_1 = jnp.full(3, 5.0)
        mem_days_2 = jnp.full(3, 10.0)

        with override_backend("cpu"):
            cpu_e1, cpu_e2 = calc_ewma_pair(
                mem_days_1, mem_days_2, prices, 1440, max_mem, cap_lamb=True
            )
        with override_backend("gpu"):
            gpu_e1, gpu_e2 = calc_ewma_pair(
                mem_days_1, mem_days_2, prices, 1440, max_mem, cap_lamb=True
            )

        assert jnp.allclose(cpu_e1, gpu_e1, rtol=1e-10, atol=1e-10), \
            f"EWMA1 max diff: {jnp.max(jnp.abs(cpu_e1 - gpu_e1))}"
        assert jnp.allclose(cpu_e2, gpu_e2, rtol=1e-10, atol=1e-10), \
            f"EWMA2 max diff: {jnp.max(jnp.abs(cpu_e2 - gpu_e2))}"

    @pytest.mark.parametrize("use_alt_lamb", [False, True])
    def test_gradients_cpu_gpu_equivalence(self, rng_key, use_alt_lamb):
        """Gradients via conv match gradients via scan."""
        prices = generate_test_prices(rng_key, n_timesteps=200, n_assets=3)
        params = {
            "logit_lamb": jnp.array([-2.0, -2.0, -2.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0, 0.0]),
        }
        if use_alt_lamb:
            params["logit_delta_lamb"] = jnp.array([1.0, 1.0, 1.0])

        with override_backend("cpu"):
            cpu_grads = calc_gradients(
                params, prices, 1440, 30,
                use_alt_lamb=use_alt_lamb, cap_lamb=True,
            )
        with override_backend("gpu"):
            gpu_grads = calc_gradients(
                params, prices, 1440, 30,
                use_alt_lamb=use_alt_lamb, cap_lamb=True,
            )

        assert jnp.allclose(cpu_grads, gpu_grads, rtol=1e-10, atol=1e-10), \
            f"Gradient max diff: {jnp.max(jnp.abs(cpu_grads - gpu_grads))}"

    def test_variance_cpu_gpu_equivalence(self, rng_key):
        """Variance via conv matches variance via scan."""
        prices = generate_test_prices(rng_key, n_timesteps=200, n_assets=3)
        params = {"logit_lamb": jnp.array([-2.0, -2.0, -2.0])}

        with override_backend("cpu"):
            cpu_var = calc_return_variances(params, prices, 1440, 30, cap_lamb=True)
        with override_backend("gpu"):
            gpu_var = calc_return_variances(params, prices, 1440, 30, cap_lamb=True)

        # Skip first row (initialization difference)
        assert jnp.allclose(cpu_var[1:], gpu_var[1:], rtol=1e-10, atol=1e-10), \
            f"Variance max diff: {jnp.max(jnp.abs(cpu_var[1:] - gpu_var[1:]))}"


# =============================================================================
# 1c. FFT slicing correctness and JIT/vmap compatibility
# =============================================================================

class TestFFTConvolveEdgeCases:
    """Edge cases, output sizes, JIT/vmap compatibility."""

    def test_output_size_various_n_out(self):
        """_fft_convolve_1d produces correctly-sized output."""
        x = jnp.ones(10)
        k = jnp.ones(5)
        for n_out in [14, 10, 5, 1]:
            result = _fft_convolve_1d(x, k, n_out)
            assert result.shape == (n_out,), f"Expected ({n_out},), got {result.shape}"

    def test_kernel_longer_than_signal(self):
        """Works when kernel is longer than signal."""
        x = jnp.array([1.0, 2.0, 3.0])
        k = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
        n_out = len(x) + len(k) - 1
        fft_result = _fft_convolve_1d(x, k, n_out)
        direct_result = jnp.convolve(x, k, mode="full")
        np.testing.assert_allclose(np.array(fft_result), np.array(direct_result), rtol=1e-10)

    def test_equal_length_inputs(self):
        """Works when signal and kernel have equal lengths."""
        x = jnp.array([1.0, 2.0, 3.0])
        k = jnp.array([1.0, 1.0, 1.0])
        n_out = len(x) + len(k) - 1
        fft_result = _fft_convolve_1d(x, k, n_out)
        direct_result = jnp.convolve(x, k, mode="full")
        np.testing.assert_allclose(np.array(fft_result), np.array(direct_result), rtol=1e-10)

    def test_power_of_two_lengths(self):
        """Works with power-of-2 lengths."""
        x = jnp.ones(64)
        k = jnp.ones(32)
        n_out = len(x) + len(k) - 1
        np.testing.assert_allclose(
            np.array(_fft_convolve_1d(x, k, n_out)),
            np.array(jnp.convolve(x, k, mode="full")),
            rtol=1e-10,
        )

    def test_non_power_of_two_lengths(self):
        """Works with non-power-of-2 lengths."""
        x = jnp.ones(100)
        k = jnp.ones(37)
        n_out = len(x) + len(k) - 1
        np.testing.assert_allclose(
            np.array(_fft_convolve_1d(x, k, n_out)),
            np.array(jnp.convolve(x, k, mode="full")),
            rtol=1e-10,
        )

    def test_works_under_jit(self):
        """_fft_convolve_1d works under jit compilation."""
        x = jnp.ones(20)
        k = jnp.ones(5)
        n_out = 24

        @jit
        def f(x, k):
            return _fft_convolve_1d(x, k, n_out)

        result = f(x, k)
        expected = jnp.convolve(x, k, mode="full")
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-10)

    def test_works_under_vmap(self):
        """_fft_convolve_1d works under vmap."""
        key = random.PRNGKey(0)
        x_batch = random.normal(key, (4, 20))
        k = jnp.ones(5)
        n_out = 24

        def convolve_one(x):
            return _fft_convolve_1d(x, k, n_out)

        results = vmap(convolve_one)(x_batch)

        for i in range(4):
            expected = jnp.convolve(x_batch[i], k, mode="full")
            np.testing.assert_allclose(
                np.array(results[i]), np.array(expected), rtol=1e-10,
            )

    def test_fft_convolve_full_wrapper(self):
        """_fft_convolve_full convenience wrapper matches full-mode conv."""
        key = random.PRNGKey(7)
        k1, k2 = random.split(key)
        x = random.normal(k1, (50,))
        k = random.normal(k2, (10,))

        result = _fft_convolve_full(x, k)
        expected = jnp.convolve(x, k, mode="full")
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-10)
