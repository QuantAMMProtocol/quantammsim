"""
Unit tests for EWMA pair calculations.

Compares CPU and GPU implementations to ensure numerical equivalence.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from jax import random
from contextlib import contextmanager

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_ewma_pair,
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


class TestEWMAPairImplementations:
    """Test EWMA pair calculations across CPU and GPU implementations."""

    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(0)

    def run_ewma_pair_comparison(self, prices, memory_days_1, memory_days_2,
                                  chunk_period=1440, max_memory_days=30):
        """Run EWMA pair calculation with both CPU and GPU implementations."""
        # Broadcast memory days to match number of assets
        mem_days_1 = jnp.full(prices.shape[1], memory_days_1)
        mem_days_2 = jnp.full(prices.shape[1], memory_days_2)

        # Run CPU version
        with override_backend("cpu"):
            cpu_ewma1, cpu_ewma2 = calc_ewma_pair(
                mem_days_1, mem_days_2, prices, chunk_period, max_memory_days, cap_lamb=True
            )

        # Run GPU version
        with override_backend("gpu"):
            gpu_ewma1, gpu_ewma2 = calc_ewma_pair(
                mem_days_1, mem_days_2, prices, chunk_period, max_memory_days, cap_lamb=True
            )

        return (cpu_ewma1, cpu_ewma2), (gpu_ewma1, gpu_ewma2)

    @pytest.mark.parametrize("n_timesteps,n_assets,mem1,mem2,chunk_period,max_mem", [
        (100, 3, 5, 10, 1440, 30),    # Standard case
        (1000, 5, 10, 20, 1440, 60),  # Longer series
        (50, 2, 3, 6, 1440, 15),      # Short series
        (200, 4, 15, 15, 1440, 45),   # Equal memory lengths
    ])
    def test_cpu_gpu_equivalence(self, rng_key, n_timesteps, n_assets, mem1, mem2,
                                  chunk_period, max_mem):
        """Test that CPU and GPU implementations produce equivalent results."""
        key, subkey = random.split(rng_key)
        prices = generate_test_prices(subkey, n_timesteps, n_assets)

        cpu_results, gpu_results = self.run_ewma_pair_comparison(
            prices, mem1, mem2, chunk_period, max_mem
        )

        # Check shapes match
        assert cpu_results[0].shape == gpu_results[0].shape
        assert cpu_results[1].shape == gpu_results[1].shape

        # Check values match within tolerance
        for i, (cpu_ewma, gpu_ewma) in enumerate(zip(cpu_results, gpu_results)):
            max_diff = jnp.max(jnp.abs(cpu_ewma - gpu_ewma))
            assert jnp.allclose(cpu_ewma, gpu_ewma, rtol=1e-10, atol=1e-10), \
                f"EWMA {i+1} mismatch: max diff = {max_diff}"

    def test_output_shapes(self, rng_key):
        """Test that output shapes are correct."""
        n_timesteps, n_assets = 100, 3
        prices = generate_test_prices(rng_key, n_timesteps, n_assets)

        cpu_results, gpu_results = self.run_ewma_pair_comparison(
            prices, memory_days_1=5, memory_days_2=10
        )

        # Both EWMAs should have same shape as input prices (minus 1 for returns)
        for ewma in cpu_results + gpu_results:
            assert ewma.shape[1] == n_assets

    def test_ewma_ordering(self, rng_key):
        """Test that shorter memory EWMA is more responsive than longer."""
        prices = generate_test_prices(rng_key, n_timesteps=200, n_assets=2)

        cpu_results, _ = self.run_ewma_pair_comparison(
            prices, memory_days_1=5, memory_days_2=20, max_memory_days=60
        )

        ewma_short, ewma_long = cpu_results

        # Shorter memory EWMA should have higher variance (more responsive)
        var_short = jnp.var(ewma_short, axis=0)
        var_long = jnp.var(ewma_long, axis=0)

        # This should generally hold for most assets
        assert jnp.mean(var_short > var_long) >= 0.5, \
            "Shorter memory EWMA should generally be more responsive"

    def test_equal_memory_lengths(self, rng_key):
        """Test behavior when both memory lengths are equal."""
        prices = generate_test_prices(rng_key, n_timesteps=100, n_assets=3)

        cpu_results, _ = self.run_ewma_pair_comparison(
            prices, memory_days_1=10, memory_days_2=10
        )

        ewma1, ewma2 = cpu_results

        # With equal memory lengths, both EWMAs should be identical
        assert jnp.allclose(ewma1, ewma2, rtol=1e-10, atol=1e-10), \
            "Equal memory lengths should produce identical EWMAs"
