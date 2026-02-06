"""
Tests for BoundedWeightsHook scan path support.

These tests verify that per-asset weight bounds work correctly with the
scan-based weight calculation path.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from functools import partial

from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.pools.G3M.quantamm.power_channel_pool import PowerChannelPool
from quantammsim.hooks.bounded_weights_hook import BoundedWeightsHook
from quantammsim.runners.jax_runner_utils import NestedHashabledict


# Create bounded pool classes for testing
class BoundedMomentumPool(BoundedWeightsHook, MomentumPool):
    """Momentum pool with per-asset weight bounds."""
    pass


class BoundedPowerChannelPool(BoundedWeightsHook, PowerChannelPool):
    """Power channel pool with per-asset weight bounds."""
    pass


class TestBoundedWeightsCoarseWeightStep:
    """Test calculate_coarse_weight_step with per-asset bounds."""

    @pytest.fixture
    def pool(self):
        return BoundedMomentumPool()

    @pytest.fixture
    def run_fingerprint(self):
        return {
            "n_assets": 2,
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "maximum_change": 0.01,
            "minimum_weight": 0.1,
            "ste_max_change": False,
            "ste_min_max_weight": False,
            "max_memory_days": 365,
        }

    @pytest.fixture
    def params_with_bounds(self):
        return {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "min_weights_per_asset": jnp.array([[0.2, 0.2]]),
            "max_weights_per_asset": jnp.array([[0.8, 0.8]]),
        }

    @pytest.fixture
    def params_without_bounds(self):
        return {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
        }

    def test_coarse_weight_step_with_bounds_respects_min(
        self, pool, run_fingerprint, params_with_bounds
    ):
        """Test that scan path respects minimum weight bounds."""
        # Initial state
        initial_price = jnp.array([100.0, 50.0])
        estimator_carry = pool.get_initial_rule_state(
            initial_price, params_with_bounds, run_fingerprint
        )
        weight_carry = {"prev_actual_weight": jnp.array([0.5, 0.5])}

        # Run several steps
        prices = [
            jnp.array([100.0, 50.0]),
            jnp.array([110.0, 45.0]),  # BTC up, ETH down
            jnp.array([120.0, 40.0]),  # More extreme
        ]

        for price in prices:
            estimator_carry, weight_carry, step_output = pool.calculate_coarse_weight_step(
                estimator_carry, weight_carry, price, params_with_bounds, run_fingerprint
            )

            # Check target weights respect bounds
            target_weight = step_output["target_weight"]
            assert jnp.all(target_weight >= 0.2 - 1e-6), (
                f"Target weight {target_weight} below min bound 0.2"
            )
            assert jnp.all(target_weight <= 0.8 + 1e-6), (
                f"Target weight {target_weight} above max bound 0.8"
            )

    def test_coarse_weight_step_with_bounds_respects_max(
        self, pool, run_fingerprint, params_with_bounds
    ):
        """Test that scan path respects maximum weight bounds."""
        # Start with weights near the max
        initial_price = jnp.array([100.0, 50.0])
        estimator_carry = pool.get_initial_rule_state(
            initial_price, params_with_bounds, run_fingerprint
        )
        weight_carry = {"prev_actual_weight": jnp.array([0.75, 0.25])}

        # Price movement that would push first asset weight higher
        price = jnp.array([150.0, 40.0])

        estimator_carry, weight_carry, step_output = pool.calculate_coarse_weight_step(
            estimator_carry, weight_carry, price, params_with_bounds, run_fingerprint
        )

        target_weight = step_output["target_weight"]
        # First asset should be capped at max
        assert target_weight[0] <= 0.8 + 1e-6, (
            f"First asset weight {target_weight[0]} exceeds max bound 0.8"
        )

    def test_coarse_weight_step_without_bounds(
        self, pool, run_fingerprint, params_without_bounds
    ):
        """Test that scan path works without per-asset bounds."""
        initial_price = jnp.array([100.0, 50.0])
        estimator_carry = pool.get_initial_rule_state(
            initial_price, params_without_bounds, run_fingerprint
        )
        weight_carry = {"prev_actual_weight": jnp.array([0.5, 0.5])}

        price = jnp.array([110.0, 45.0])

        # Should not raise an error
        estimator_carry, weight_carry, step_output = pool.calculate_coarse_weight_step(
            estimator_carry, weight_carry, price, params_without_bounds, run_fingerprint
        )

        # Basic sanity check
        target_weight = step_output["target_weight"]
        assert jnp.allclose(jnp.sum(target_weight), 1.0, atol=1e-6)

    def test_coarse_weight_step_weights_sum_to_one(
        self, pool, run_fingerprint, params_with_bounds
    ):
        """Test that target weights always sum to 1."""
        initial_price = jnp.array([100.0, 50.0])
        estimator_carry = pool.get_initial_rule_state(
            initial_price, params_with_bounds, run_fingerprint
        )
        weight_carry = {"prev_actual_weight": jnp.array([0.5, 0.5])}

        prices = [
            jnp.array([100.0, 50.0]),
            jnp.array([80.0, 60.0]),
            jnp.array([120.0, 40.0]),
            jnp.array([90.0, 55.0]),
        ]

        for price in prices:
            estimator_carry, weight_carry, step_output = pool.calculate_coarse_weight_step(
                estimator_carry, weight_carry, price, params_with_bounds, run_fingerprint
            )
            target_weight = step_output["target_weight"]
            assert jnp.allclose(jnp.sum(target_weight), 1.0, atol=1e-6), (
                f"Target weights {target_weight} don't sum to 1"
            )


class TestBoundedWeightsThreeAssets:
    """Test bounded weights scan path with 3 assets."""

    @pytest.fixture
    def pool(self):
        return BoundedMomentumPool()

    @pytest.fixture
    def run_fingerprint(self):
        return {
            "n_assets": 3,
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "maximum_change": 0.01,
            "minimum_weight": 0.05,
            "ste_max_change": False,
            "ste_min_max_weight": False,
            "max_memory_days": 365,
        }

    @pytest.fixture
    def params_with_bounds(self):
        return {
            "log_k": jnp.array([3.0, 3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0, 0.0]),
            "min_weights_per_asset": jnp.array([[0.1, 0.1, 0.1]]),
            "max_weights_per_asset": jnp.array([[0.6, 0.6, 0.6]]),
        }

    def test_three_asset_bounds_respected(self, pool, run_fingerprint, params_with_bounds):
        """Test that bounds are respected with 3 assets."""
        initial_price = jnp.array([100.0, 50.0, 25.0])
        estimator_carry = pool.get_initial_rule_state(
            initial_price, params_with_bounds, run_fingerprint
        )
        weight_carry = {"prev_actual_weight": jnp.array([0.33, 0.33, 0.34])}

        prices = [
            jnp.array([100.0, 50.0, 25.0]),
            jnp.array([120.0, 40.0, 30.0]),
            jnp.array([80.0, 60.0, 20.0]),
        ]

        for price in prices:
            estimator_carry, weight_carry, step_output = pool.calculate_coarse_weight_step(
                estimator_carry, weight_carry, price, params_with_bounds, run_fingerprint
            )

            target_weight = step_output["target_weight"]
            assert jnp.all(target_weight >= 0.1 - 1e-6), (
                f"Weight below min: {target_weight}"
            )
            assert jnp.all(target_weight <= 0.6 + 1e-6), (
                f"Weight above max: {target_weight}"
            )
            assert jnp.allclose(jnp.sum(target_weight), 1.0, atol=1e-6), (
                f"Weights don't sum to 1: {target_weight}"
            )


class TestBoundedWeightsPowerChannelPool:
    """Test bounded weights scan path with PowerChannelPool."""

    @pytest.fixture
    def pool(self):
        return BoundedPowerChannelPool()

    @pytest.fixture
    def run_fingerprint(self):
        return {
            "n_assets": 2,
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "maximum_change": 0.01,
            "minimum_weight": 0.1,
            "ste_max_change": False,
            "ste_min_max_weight": False,
            "max_memory_days": 365,
            "use_pre_exp_scaling": True,
        }

    @pytest.fixture
    def params_with_bounds(self):
        return {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "raw_exponents": jnp.array([1.0, 1.0]),
            "raw_pre_exp_scaling": jnp.array([0.5, 0.5]),
            "min_weights_per_asset": jnp.array([[0.25, 0.25]]),
            "max_weights_per_asset": jnp.array([[0.75, 0.75]]),
        }

    def test_power_channel_bounds_respected(
        self, pool, run_fingerprint, params_with_bounds
    ):
        """Test that PowerChannelPool respects bounds in scan path."""
        initial_price = jnp.array([100.0, 50.0])
        estimator_carry = pool.get_initial_rule_state(
            initial_price, params_with_bounds, run_fingerprint
        )
        weight_carry = {"prev_actual_weight": jnp.array([0.5, 0.5])}

        prices = [
            jnp.array([100.0, 50.0]),
            jnp.array([110.0, 45.0]),
            jnp.array([90.0, 55.0]),
        ]

        for price in prices:
            estimator_carry, weight_carry, step_output = pool.calculate_coarse_weight_step(
                estimator_carry, weight_carry, price, params_with_bounds, run_fingerprint
            )

            target_weight = step_output["target_weight"]
            assert jnp.all(target_weight >= 0.25 - 1e-6), (
                f"Weight below min: {target_weight}"
            )
            assert jnp.all(target_weight <= 0.75 + 1e-6), (
                f"Weight above max: {target_weight}"
            )
            assert jnp.allclose(jnp.sum(target_weight), 1.0, atol=1e-6)


class TestBoundedWeightsScanMultiStep:
    """Test bounded weights scan path over multiple timesteps."""

    @pytest.fixture
    def pool(self):
        return BoundedMomentumPool()

    @pytest.fixture
    def run_fingerprint(self):
        return {
            "n_assets": 2,
            "chunk_period": 60,
            "weight_interpolation_period": 60,
            "maximum_change": 0.001,
            "minimum_weight": 0.1,
            "ste_max_change": False,
            "ste_min_max_weight": False,
            "max_memory_days": 30,
        }

    @pytest.fixture
    def params(self):
        return {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "min_weights_per_asset": jnp.array([[0.2, 0.2]]),
            "max_weights_per_asset": jnp.array([[0.8, 0.8]]),
        }

    def test_multi_step_scan_respects_bounds(
        self, pool, run_fingerprint, params
    ):
        """Test that scan path respects bounds over many steps with trending prices."""
        # Generate synthetic price data with trends
        np.random.seed(42)
        n_steps = 50

        prices = np.zeros((n_steps, 2))
        prices[0] = [100.0, 50.0]
        for i in range(1, n_steps):
            # First asset trends up, second trends down (to push weights to extremes)
            prices[i, 0] = prices[i-1, 0] * (1 + 0.02 + np.random.randn() * 0.005)
            prices[i, 1] = prices[i-1, 1] * (1 - 0.01 + np.random.randn() * 0.005)
        prices = jnp.array(prices)

        # Initialize state
        initial_price = prices[0]
        estimator_carry = pool.get_initial_rule_state(
            initial_price, params, run_fingerprint
        )
        weight_carry = {"prev_actual_weight": jnp.array([0.5, 0.5])}

        all_target_weights = []

        # Run many steps
        for i in range(1, n_steps):
            price = prices[i]
            estimator_carry, weight_carry, step_output = pool.calculate_coarse_weight_step(
                estimator_carry, weight_carry, price, params, run_fingerprint
            )
            all_target_weights.append(step_output["target_weight"])

        all_target_weights = jnp.stack(all_target_weights)

        # All weights should respect the per-asset bounds
        min_bound = 0.2
        max_bound = 0.8
        assert jnp.all(all_target_weights >= min_bound - 1e-6), (
            f"Found weight below min {min_bound}: {all_target_weights.min()}"
        )
        assert jnp.all(all_target_weights <= max_bound + 1e-6), (
            f"Found weight above max {max_bound}: {all_target_weights.max()}"
        )

        # All weights should sum to 1
        weight_sums = jnp.sum(all_target_weights, axis=1)
        np.testing.assert_allclose(
            weight_sums, np.ones(len(weight_sums)), atol=1e-6,
            err_msg="Weights don't sum to 1"
        )

    def test_bounds_different_per_asset(self, pool, run_fingerprint):
        """Test with asymmetric per-asset bounds."""
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            # Different bounds per asset
            "min_weights_per_asset": jnp.array([[0.1, 0.3]]),
            "max_weights_per_asset": jnp.array([[0.7, 0.9]]),
        }

        # Generate prices
        np.random.seed(123)
        n_steps = 30
        prices = np.zeros((n_steps, 2))
        prices[0] = [100.0, 50.0]
        for i in range(1, n_steps):
            prices[i] = prices[i-1] * (1 + np.random.randn(2) * 0.02)
        prices = jnp.array(prices)

        # Initialize
        initial_price = prices[0]
        estimator_carry = pool.get_initial_rule_state(
            initial_price, params, run_fingerprint
        )
        weight_carry = {"prev_actual_weight": jnp.array([0.4, 0.6])}

        # Run steps
        for i in range(1, n_steps):
            price = prices[i]
            estimator_carry, weight_carry, step_output = pool.calculate_coarse_weight_step(
                estimator_carry, weight_carry, price, params, run_fingerprint
            )
            target_weight = step_output["target_weight"]

            # Check per-asset bounds
            assert target_weight[0] >= 0.1 - 1e-6, f"Asset 0 below min: {target_weight[0]}"
            assert target_weight[0] <= 0.7 + 1e-6, f"Asset 0 above max: {target_weight[0]}"
            assert target_weight[1] >= 0.3 - 1e-6, f"Asset 1 below min: {target_weight[1]}"
            assert target_weight[1] <= 0.9 + 1e-6, f"Asset 1 above max: {target_weight[1]}"
            assert jnp.allclose(jnp.sum(target_weight), 1.0, atol=1e-6)
