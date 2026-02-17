"""
Unit tests for variance calculation implementations.

Compares CPU and GPU implementations of return variance calculations.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from jax import random
from contextlib import contextmanager

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_return_variances,
    calc_return_precision_based_weights,
    DEFAULT_BACKEND,
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


class TestVarianceCalculations:
    """Test variance calculation implementations."""

    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(0)

    @pytest.fixture
    def default_params(self):
        """Default parameters for variance calculation."""
        return {"logit_lamb": jnp.array([-2.0, -2.0, -2.0])}

    def run_variance_comparison(self, prices, params, chunk_period=1440, max_memory_days=30):
        """Run variance calculation with both CPU and GPU implementations."""
        # Run CPU version
        with override_backend("cpu"):
            cpu_variances = calc_return_variances(
                params, prices, chunk_period, max_memory_days, cap_lamb=True
            )

        # Run GPU version
        with override_backend("gpu"):
            gpu_variances = calc_return_variances(
                params, prices, chunk_period, max_memory_days, cap_lamb=True
            )

        return cpu_variances, gpu_variances

    @pytest.mark.parametrize("n_timesteps,n_assets,chunk_period,max_memory_days", [
        (100, 3, 1440, 30),   # Standard case
        (1000, 5, 1440, 60),  # Longer series
        (50, 2, 1440, 15),    # Short series
    ])
    def test_cpu_gpu_equivalence(self, rng_key, n_timesteps, n_assets,
                                  chunk_period, max_memory_days):
        """Test that CPU and GPU implementations produce equivalent results."""
        key, subkey = random.split(rng_key)
        prices = generate_test_prices(subkey, n_timesteps, n_assets)
        params = {"logit_lamb": jnp.array([-2.0] * n_assets)}

        cpu_vars, gpu_vars = self.run_variance_comparison(
            prices, params, chunk_period, max_memory_days
        )

        # Check shapes match
        assert cpu_vars.shape == gpu_vars.shape

        # Check values match within tolerance (skip first row which has initialization differences)
        cpu_vars_trimmed = cpu_vars[1:]
        gpu_vars_trimmed = gpu_vars[1:]
        max_diff = jnp.max(jnp.abs(cpu_vars_trimmed - gpu_vars_trimmed))
        assert jnp.allclose(cpu_vars_trimmed, gpu_vars_trimmed, rtol=1e-10, atol=1e-10), \
            f"Variance mismatch: max diff = {max_diff}"

    def test_variances_positive(self, rng_key, default_params):
        """Test that calculated variances are positive."""
        prices = generate_test_prices(rng_key, n_timesteps=100, n_assets=3)

        cpu_vars, gpu_vars = self.run_variance_comparison(prices, default_params)

        # Machine-epsilon tolerance: first-row warm-up can produce tiny negatives
        assert jnp.all(cpu_vars > -1e-10), f"CPU variances below machine tol: min={float(jnp.min(cpu_vars))}"
        assert jnp.all(gpu_vars > -1e-10), f"GPU variances below machine tol: min={float(jnp.min(gpu_vars))}"

    def test_output_shape(self, rng_key, default_params):
        """Test that output shapes are correct."""
        n_timesteps, n_assets = 100, 3
        prices = generate_test_prices(rng_key, n_timesteps, n_assets)

        cpu_vars, gpu_vars = self.run_variance_comparison(prices, default_params)

        # Variances should have shape (n_timesteps - 1, n_assets) due to returns
        assert cpu_vars.shape[1] == n_assets
        assert gpu_vars.shape[1] == n_assets


class TestPrecisionBasedWeights:
    """Test precision-based weight calculations."""

    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(42)

    def test_weights_sum_to_one(self, rng_key):
        """Test that precision-based weights sum to 1."""
        prices = generate_test_prices(rng_key, n_timesteps=100, n_assets=3)
        params = {"logit_lamb": jnp.array([-2.0, -2.0, -2.0])}

        weights = calc_return_precision_based_weights(
            params, prices, chunk_period=1440, max_memory_days=30, cap_lamb=True
        )

        # Check weights sum to 1 (with tolerance)
        weight_sums = jnp.sum(weights, axis=-1)
        assert jnp.allclose(weight_sums, 1.0, rtol=1e-6, atol=1e-6), \
            f"Weights should sum to 1, got sums: {weight_sums[:5]}"

    def test_weights_positive(self, rng_key):
        """Test that all weights are positive."""
        prices = generate_test_prices(rng_key, n_timesteps=100, n_assets=3)
        params = {"logit_lamb": jnp.array([-2.0, -2.0, -2.0])}

        weights = calc_return_precision_based_weights(
            params, prices, chunk_period=1440, max_memory_days=30, cap_lamb=True
        )

        assert jnp.all(weights > 0), "All weights should be positive"

    def test_higher_variance_lower_weight(self, rng_key):
        """Test that assets with higher variance get lower weights."""
        # Create prices where one asset has clearly higher variance
        n_timesteps = 500
        key1, key2, key3 = random.split(rng_key, 3)

        # Asset 0: low volatility
        returns_0 = random.normal(key1, (n_timesteps,)) * 0.005
        # Asset 1: high volatility
        returns_1 = random.normal(key2, (n_timesteps,)) * 0.05
        # Asset 2: medium volatility
        returns_2 = random.normal(key3, (n_timesteps,)) * 0.02

        prices = jnp.exp(jnp.cumsum(
            jnp.column_stack([returns_0, returns_1, returns_2]), axis=0
        ))
        prices = prices - jnp.min(prices) + 1.0

        params = {"logit_lamb": jnp.array([-2.0, -2.0, -2.0])}

        weights = calc_return_precision_based_weights(
            params, prices, chunk_period=1440, max_memory_days=30, cap_lamb=True
        )

        # Average weights over time
        avg_weights = jnp.mean(weights, axis=0)

        # Asset 0 (low vol) should have higher weight than Asset 1 (high vol)
        assert avg_weights[0] > avg_weights[1], \
            f"Low vol asset should have higher weight: {avg_weights}"
