"""
Unit tests for minimum variance pool parameterizations.

Tests that different parameter formats (logit_lamb vs memory_days)
produce equivalent results.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from jax.nn import softmax

from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool
from quantammsim.runners.jax_runner_utils import NestedHashabledict
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    jax_memory_days_to_lamb,
    memory_days_to_logit_lamb,
)


def create_test_prices(n_timesteps=1000, n_assets=2, seed=0):
    """Create test price data with some volatility."""
    np.random.seed(seed)
    base_prices = np.exp(np.random.randn(n_timesteps, n_assets) * 0.1)
    base_prices = base_prices / base_prices[0]
    return jnp.array(base_prices)


def create_params_logit_lamb(memory_days_1=10.0, memory_days_2=20.0, chunk_period=60, n_assets=2):
    """Create params using logit_lamb parameterization."""
    logit_lamb = memory_days_to_logit_lamb(jnp.array([memory_days_1]), chunk_period)

    lamb_1 = jax_memory_days_to_lamb(memory_days_1, chunk_period)
    lamb_2 = jax_memory_days_to_lamb(memory_days_2, chunk_period)
    logit_lamb_2 = jnp.log(lamb_2 / (1.0 - lamb_2))
    logit_delta_lamb = logit_lamb_2 - logit_lamb

    return {
        'logit_lamb': logit_lamb,
        'logit_delta_lamb': logit_delta_lamb,
        'initial_weights_logits': jnp.zeros(n_assets),
    }


def create_params_memory_days(memory_days_1=10.0, memory_days_2=20.0, n_assets=2):
    """Create params using direct memory_days parameterization."""
    return {
        'memory_days_1': jnp.array([memory_days_1]),
        'memory_days_2': jnp.array([memory_days_2]),
        'initial_weights_logits': jnp.zeros(n_assets),
    }


def create_run_fingerprint(n_timesteps, n_assets):
    """Create basic run fingerprint for testing."""
    rf = NestedHashabledict({
        'chunk_period': 60,
        'weight_interpolation_period': 60,
        'max_memory_days': 365,
        'minimum_weight': 0.1,
        'use_alt_lamb': True,
        'bout_length': n_timesteps - 1,  # bout_length should be less than prices length
        'n_assets': n_assets,
        'rule': 'min_variance',
        'return_val': 'final_reserves_value_and_weights',
        'maximum_change': 0.0003,
        'arb_fees': 0.0,
        'gas_cost': 0.0,
        'arb_quality': 0.0,
        'do_arb': True,
        'weight_interpolation_method': 'linear',
        'arb_frequency': 1,
        'do_trades': False,
    })
    return rf


class TestMinVarParameterizations:
    """Test minimum variance parameterization equivalence."""

    @pytest.fixture
    def pool(self):
        return MinVariancePool()

    @pytest.fixture
    def test_prices(self):
        return create_test_prices(n_timesteps=1000, n_assets=2)

    @pytest.mark.parametrize("memory_days_1,memory_days_2", [
        (10.0, 2.0),
        (5.0, 10.0),
        (20.0, 20.0),
        (15.0, 30.0),
    ])
    def test_parameterization_equivalence(self, pool, memory_days_1, memory_days_2):
        """Test that logit_lamb and memory_days parameterizations give same results."""
        prices = create_test_prices(n_timesteps=500, n_assets=2)
        rf = create_run_fingerprint(len(prices), 2)

        params_logit = create_params_logit_lamb(
            memory_days_1, memory_days_2, rf["chunk_period"], n_assets=2
        )
        params_memory = create_params_memory_days(
            memory_days_1, memory_days_2, n_assets=2
        )

        # Calculate raw weights with both parameterizations
        raw_weights_logit = pool.calculate_rule_outputs(
            params_logit, rf, prices, None
        )
        raw_weights_memory = pool.calculate_rule_outputs(
            params_memory, rf, prices, None
        )

        max_diff = jnp.max(jnp.abs(raw_weights_logit - raw_weights_memory))
        assert jnp.allclose(raw_weights_logit, raw_weights_memory, rtol=1e-6, atol=1e-6), \
            f"Raw weights differ by max {max_diff} for memory_days ({memory_days_1}, {memory_days_2})"

    def test_raw_weights_shape(self, pool):
        """Test that raw weights have correct shape."""
        prices = create_test_prices(n_timesteps=500, n_assets=2)
        rf = create_run_fingerprint(len(prices), 2)
        params = create_params_logit_lamb(10.0, 20.0, rf["chunk_period"], n_assets=2)

        raw_weights = pool.calculate_rule_outputs(params, rf, prices, None)

        assert raw_weights.shape[1] == 2, \
            f"Expected 2 weight columns, got {raw_weights.shape[1]}"

    def test_raw_weights_sum_approximately_one(self, pool):
        """Test that raw weights sum approximately to 1."""
        prices = create_test_prices(n_timesteps=500, n_assets=3)
        rf = create_run_fingerprint(len(prices), 3)
        params = create_params_logit_lamb(10.0, 20.0, rf["chunk_period"], n_assets=3)

        raw_weights = pool.calculate_rule_outputs(params, rf, prices, None)
        weight_sums = jnp.sum(raw_weights, axis=1)

        # Raw weights should sum to approximately 1
        assert jnp.allclose(weight_sums, 1.0, rtol=0.1, atol=0.1), \
            f"Raw weights should sum close to 1, got: {weight_sums[:5]}..."
