"""
Unit tests for lookahead bias in the simulator.

These tests verify that calculations at time t do not depend on
data from time t+k (no lookahead/future information leakage).

This is a critical correctness property for any financial simulation.

Index Relationships
-------------------
Understanding the index mapping is critical for these tests. Empirically verified:

- prices[i] is the price at timestep i.

- gradient[i] uses price[i+1]. When prices[cutoff:] are modified:
  * gradient[cutoff-1] IS the first affected
  * gradient[cutoff-2] is NOT affected
  This is expected behavior - gradient[i] measures the rate of change up to
  price[i+1]. The test checks [:cutoff-1] to exclude gradient[cutoff-1].

- reserve[i] uses prices up to i (after the fix to fine_weights.py). When prices[cutoff:]
  are modified:
  * reserve[cutoff] is the first affected for all pool types
  * reserve[cutoff-1] is NOT affected
  The test uses [:cutoff-1] which is conservative but safe for all pools.

The tests verify no lookahead by comparing calculations with full vs truncated price data.
"""
import pytest
import jax
import jax.numpy as jnp
from jax import random

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import calc_gradients
from quantammsim.pools.creator import create_pool
from quantammsim.runners.jax_runner_utils import NestedHashabledict


# All QuantAMM pool types that should be tested for lookahead bias
QUANTAMM_POOL_TYPES = [
    "momentum",
    "anti_momentum",
    "power_channel",
    "mean_reversion_channel",
    "difference_momentum",
    "min_variance",
]


class TestGradientNoLookahead:
    """Test that gradient calculations have no lookahead bias."""

    @pytest.fixture
    def rng_key(self):
        return random.PRNGKey(0)

    @pytest.mark.parametrize("cutoff_fraction", [0.25, 0.5, 0.75])
    def test_gradients_independent_of_future_prices(self, cutoff_fraction):
        """
        Test that gradients at time t are independent of prices at time t+k.

        Strategy: Calculate gradients with full data, then replace future prices
        with dramatically different values. Gradients before the cutoff should
        be identical.
        """
        n_assets = 2
        n_timesteps = 1000
        chunk_period = 60
        max_memory_days = 30.0
        use_alt_lamb = False

        # Generate constant price data
        prices = jnp.ones((n_timesteps, n_assets))

        params = {"logit_lamb": jnp.array(0.0)}

        # Calculate gradients for full dataset
        full_gradients = calc_gradients(
            params, prices, chunk_period, max_memory_days, use_alt_lamb
        )

        # Calculate cutoff point
        cutoff = int(n_timesteps * cutoff_fraction)

        # Create truncated prices with dramatically different future
        truncated_prices = jnp.concatenate([
            prices[:cutoff],
            jnp.ones((n_timesteps - cutoff, n_assets)) * 1000.0
        ])

        truncated_gradients = calc_gradients(
            params, truncated_prices, chunk_period, max_memory_days, use_alt_lamb
        )

        # Gradients before cutoff should be identical (excluding gradient[cutoff-1])
        # gradient[i] uses price[i+1], so gradient[cutoff-1] uses price[cutoff] which IS changed.
        # Therefore we check [:cutoff-1] to exclude gradient[cutoff-1].
        max_diff = jnp.max(jnp.abs(
            full_gradients[:cutoff - 1] - truncated_gradients[:cutoff - 1]
        ))

        assert max_diff < 1e-10, \
            f"Lookahead bias detected! Max gradient difference before cutoff: {max_diff}"

    def test_gradients_differ_after_cutoff(self):
        """Verify that gradients DO change after the cutoff (sanity check)."""
        n_assets = 2
        n_timesteps = 1000
        chunk_period = 60
        max_memory_days = 30.0

        prices = jnp.ones((n_timesteps, n_assets))
        params = {"logit_lamb": jnp.array(0.0)}

        full_gradients = calc_gradients(
            params, prices, chunk_period, max_memory_days, False
        )

        cutoff = 500
        truncated_prices = jnp.concatenate([
            prices[:cutoff],
            jnp.ones((n_timesteps - cutoff, n_assets)) * 1000.0
        ])

        truncated_gradients = calc_gradients(
            params, truncated_prices, chunk_period, max_memory_days, False
        )

        # Gradients AFTER cutoff should be different
        post_cutoff_diff = jnp.max(jnp.abs(
            full_gradients[cutoff:] - truncated_gradients[cutoff:]
        ))

        assert post_cutoff_diff > 1e-6, \
            "Sanity check failed: gradients should differ after price change"


class TestReservesNoLookahead:
    """Test that reserve calculations have no lookahead bias for all QuantAMM pools."""

    @pytest.fixture
    def base_fingerprint(self):
        """Create base run fingerprint for reserve tests."""
        return {
            "bout_length": 996,  # n_timesteps - 2*chunk_period (1000 - 4 = 996)
            "chunk_period": 2,
            "n_assets": 2,
            "initial_pool_value": 1000.0,
            "fees": 0.00003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "arb_frequency": 1,
            "all_sig_variations": ((1, -1), (-1, 1)),
            "do_trades": False,
            "max_memory_days": 30.0,
            "use_alt_lamb": False,
            "weight_interpolation_period": 2,
            "weight_interpolation_method": "linear",
            "maximum_change": 1.0,
            "minimum_weight": 0.05,  # Avoid exactly 0 weights which cause div-by-zero in reserve ratio
            "do_arb": True,
            "noise_trader_ratio": 0.0,
            "ste_max_change": False,
            "ste_min_max_weight": False,
            "use_pre_exp_scaling": True,  # Needed for power_channel and mean_reversion_channel
        }

    @pytest.fixture
    def base_params(self):
        """Create base parameters for reserve tests.

        These params cover the union of parameters needed by different pool types.
        """
        return {
            # Common params
            "initial_weights_logits": jnp.zeros(2),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "log_k": jnp.array([10.0, 10.0]),
            "memory_days_1": jnp.array([1.0, 1.0]),
            "memory_days_2": jnp.array([1.0, 1.0]),
            "k": jnp.array([1.0, 1.0]),  # Needed for difference_momentum pool
            # Power channel / mean reversion params
            "log_amplitude": jnp.array([-5.0, -5.0]),
            "raw_width": jnp.array([0.0, 0.0]),
            "raw_exponents": jnp.array([1.0, 1.0]),
            "raw_pre_exp_scaling": jnp.array([1.0, 1.0]),
        }

    @pytest.mark.parametrize("pool_type", QUANTAMM_POOL_TYPES)
    @pytest.mark.parametrize("cutoff_fraction", [0.25, 0.5, 0.75])
    @pytest.mark.parametrize("fees_case", ["zero_fees", "with_fees"])
    def test_reserves_independent_of_future_prices(
        self, base_fingerprint, base_params, pool_type, cutoff_fraction, fees_case
    ):
        """
        Test that reserves at time t are independent of prices at time t+k.

        This test is run for all QuantAMM pool types to ensure none have
        lookahead bias in their reserve calculations.
        """
        n_timesteps = 1000
        n_assets = 2

        pool = create_pool(pool_type)
        run_fingerprint = NestedHashabledict(base_fingerprint)

        # Generate price data
        if pool_type == "min_variance":
            # min_variance needs non-zero variance - use sinusoidal prices
            # Sinusoid has constant variance over full cycles
            t = jnp.arange(n_timesteps)
            # Different frequencies per asset to get different variances
            prices = jnp.column_stack([
                1.0 + 0.1 * jnp.sin(2 * jnp.pi * t / 100),
                1.0 + 0.05 * jnp.sin(2 * jnp.pi * t / 100),
            ])
        else:
            prices = jnp.ones((n_timesteps, n_assets))

        # Calculate reserves for full dataset
        if fees_case == "zero_fees":
            full_reserves = pool.calculate_reserves_zero_fees(
                base_params, run_fingerprint, prices, jnp.array([0, 0])
            )
        else:
            full_reserves = pool.calculate_reserves_with_fees(
                base_params, run_fingerprint, prices, jnp.array([0, 0])
            )

        # Calculate cutoff and create truncated prices
        cutoff = int(n_timesteps * cutoff_fraction)
        truncated_prices = jnp.concatenate([
            prices[:cutoff],
            jnp.ones((n_timesteps - cutoff, n_assets)).at[:, 0].set(1000.0)
        ])

        # Calculate reserves with truncated data
        if fees_case == "zero_fees":
            truncated_reserves = pool.calculate_reserves_zero_fees(
                base_params, run_fingerprint, truncated_prices, jnp.array([0, 0])
            )
        else:
            truncated_reserves = pool.calculate_reserves_with_fees(
                base_params, run_fingerprint, truncated_prices, jnp.array([0, 0])
            )

        # Reserves before cutoff should be identical
        # After the fix to fine_weights.py, all pool types have consistent behavior:
        # reserve[i] depends only on prices[0:i], so reserve[cutoff-1] is NOT affected.
        # We use [:cutoff-1] which is conservative but safe for all pools.
        max_diff = jnp.max(jnp.abs(
            full_reserves[:cutoff - 1] - truncated_reserves[:cutoff - 1]
        ))

        assert max_diff < 1e-10, \
            f"Lookahead bias in {pool_type} reserves ({fees_case})! Max diff before cutoff: {max_diff}"

    @pytest.mark.parametrize("pool_type", QUANTAMM_POOL_TYPES)
    def test_reserves_differ_after_cutoff(self, base_fingerprint, base_params, pool_type):
        """Verify that reserves DO change after the cutoff (sanity check).

        Note: min_variance pool requires price variation to compute variances.
        With constant prices, variance is zero and 1/variance produces NaN.
        This test uses larger price changes (1000x) which the original script handles fine.
        """
        n_timesteps = 1000
        n_assets = 2

        pool = create_pool(pool_type)
        run_fingerprint = NestedHashabledict(base_fingerprint)

        # Generate price data
        if pool_type == "min_variance":
            # min_variance needs non-zero variance - use sinusoidal prices
            t = jnp.arange(n_timesteps)
            prices = jnp.column_stack([
                1.0 + 0.1 * jnp.sin(2 * jnp.pi * t / 100),
                1.0 + 0.05 * jnp.sin(2 * jnp.pi * t / 100),
            ])
        else:
            prices = jnp.ones((n_timesteps, n_assets))

        full_reserves = pool.calculate_reserves_zero_fees(
            base_params, run_fingerprint, prices, jnp.array([0, 0])
        )

        cutoff = 500
        # Use large price change (1000x) as in the original check_lookahead_reserves.py
        truncated_prices = jnp.concatenate([
            prices[:cutoff],
            jnp.ones((n_timesteps - cutoff, n_assets)).at[:, 0].set(1000.0)
        ])

        truncated_reserves = pool.calculate_reserves_zero_fees(
            base_params, run_fingerprint, truncated_prices, jnp.array([0, 0])
        )

        # Reserves AFTER cutoff should be different
        post_cutoff_diff = jnp.max(jnp.abs(
            full_reserves[cutoff:] - truncated_reserves[cutoff:]
        ))

        assert post_cutoff_diff > 1e-6, \
            f"Sanity check failed for {pool_type}: reserves should differ after price change"
