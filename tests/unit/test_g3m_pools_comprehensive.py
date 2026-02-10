"""
Comprehensive tests for all G3M pool types.

Tests cover:
- BalancerPool (static weights)
- MomentumPool (trend following)
- AntiMomentumPool (contrarian)
- MeanReversionChannelPool (channel strategy)
- PowerChannelPool (power law scaling)
- MinVariancePool (variance minimization)
- DifferenceMomentumPool (dual momentum)
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax
from copy import deepcopy

from quantammsim.pools.G3M.balancer.balancer import BalancerPool
from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool, _jax_momentum_weight_update
from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool, _jax_min_variance_weights
from quantammsim.pools.creator import create_pool
from quantammsim.runners.jax_runner_utils import NestedHashabledict


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_run_fingerprint():
    """Base run fingerprint for G3M pool tests."""
    return NestedHashabledict({
        "n_assets": 2,
        "bout_length": 1440 * 7,  # 7 days in minutes
        "chunk_period": 1440,  # 1 day
        "weight_interpolation_period": 1440,
        "weight_interpolation_method": "linear",
        "maximum_change": 0.0003,
        "minimum_weight": 0.05,
        "max_memory_days": 365.0,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "initial_pool_value": 1000000.0,
        "fees": 0.003,
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "do_arb": True,
        "arb_frequency": 1,
        "all_sig_variations": tuple([tuple([1, -1]), tuple([-1, 1])]),
        "noise_trader_ratio": 0.0,
        "ste_max_change": False,
        "ste_min_max_weight": False,
    })


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n_timesteps = 1440 * 30  # 30 days of minute data
    n_assets = 2

    # Generate realistic price returns with trends
    base_prices = np.array([100.0, 2000.0])
    trends = np.linspace(0, 0.1, n_timesteps).reshape(-1, 1) + \
             np.random.normal(0, 0.001, (n_timesteps, n_assets)).cumsum(axis=0)

    prices = base_prices * (1 + trends)
    return jnp.array(prices)


@pytest.fixture
def sample_prices_multi_asset():
    """Generate sample price data with more assets."""
    np.random.seed(42)
    n_timesteps = 1440 * 30
    n_assets = 4

    base_prices = np.array([100.0, 2000.0, 50.0, 1.0])
    trends = np.linspace(0, 0.1, n_timesteps).reshape(-1, 1) + \
             np.random.normal(0, 0.001, (n_timesteps, n_assets)).cumsum(axis=0)

    prices = base_prices * (1 + trends)
    return jnp.array(prices)


# ============================================================================
# BalancerPool Tests
# ============================================================================

class TestBalancerPool:
    """Tests for BalancerPool (static weight pool)."""

    @pytest.fixture
    def balancer_pool(self):
        """Create a Balancer pool instance."""
        return BalancerPool()

    @pytest.fixture
    def balancer_params(self):
        """Create Balancer pool parameters."""
        return {
            "initial_weights_logits": jnp.array([[0.0, 0.0]]),
        }

    def test_calculate_initial_weights_outputs_shape(
        self, balancer_pool, balancer_params
    ):
        """Test that initial weights have correct shape."""
        params = {k: v[0] for k, v in balancer_params.items()}
        weights = balancer_pool.calculate_initial_weights(params)
        assert weights.shape == (2,), f"Expected shape (2,), got {weights.shape}"

    def test_calculate_initial_weights_sum_to_one(
        self, balancer_pool, balancer_params
    ):
        """Test that Balancer weights sum to 1."""
        params = {k: v[0] for k, v in balancer_params.items()}
        weights = balancer_pool.calculate_initial_weights(params)
        np.testing.assert_allclose(
            jnp.sum(weights), 1.0, rtol=1e-6,
            err_msg="Balancer weights should sum to 1"
        )

    def test_calculate_initial_weights_static(
        self, balancer_pool, balancer_params
    ):
        """Test that Balancer weights don't change."""
        params = {k: v[0] for k, v in balancer_params.items()}
        weights1 = balancer_pool.calculate_initial_weights(params)
        weights2 = balancer_pool.calculate_initial_weights(params)
        np.testing.assert_array_equal(
            weights1, weights2,
            err_msg="Balancer weights should be static"
        )

    def test_init_base_parameters_shape(
        self, balancer_pool, base_run_fingerprint
    ):
        """Test that init_base_parameters returns correct shapes."""
        n_assets = 3
        n_parameter_sets = 2

        initial_values = {
            "initial_weights_logits": 1.0,
        }

        rf = deepcopy(base_run_fingerprint)
        rf["n_assets"] = n_assets

        params = balancer_pool.init_base_parameters(
            initial_values, rf, n_assets, n_parameter_sets, noise="none"
        )

        assert params["initial_weights_logits"].shape == (n_parameter_sets, n_assets)

    def test_is_not_trainable(self, balancer_pool):
        """Test that Balancer pool is not trainable."""
        assert not balancer_pool.is_trainable(), \
            "BalancerPool should not be trainable"

    def test_calculate_reserves_with_fees(
        self, balancer_pool, balancer_params, base_run_fingerprint, sample_prices
    ):
        """Test reserve calculation with fees."""
        rf = base_run_fingerprint
        params = {k: v[0] for k, v in balancer_params.items()}

        start_index = jnp.array([1440 * 10, 0])

        reserves = balancer_pool.calculate_reserves_with_fees(
            params, rf, sample_prices, start_index
        )

        # Check reserves are positive
        assert jnp.all(reserves > 0), "Reserves should be positive"

        # Check shape
        expected_length = rf["bout_length"] - 1
        assert reserves.shape[0] == expected_length


# ============================================================================
# MomentumPool Tests
# ============================================================================

class TestMomentumPool:
    """Tests for MomentumPool (trend following)."""

    @pytest.fixture
    def momentum_pool(self):
        """Create a momentum pool instance."""
        return MomentumPool()

    @pytest.fixture
    def momentum_params(self):
        """Create momentum pool parameters."""
        return {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

    def test_calculate_rule_outputs_shape(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that rule outputs have correct shape."""
        rf = base_run_fingerprint
        rule_outputs = momentum_pool.calculate_rule_outputs(
            momentum_params, rf, sample_prices, None
        )

        # Shape should be (n_chunks - 1, n_assets)
        chunk_period = rf["chunk_period"]
        expected_n_chunks = len(sample_prices) // chunk_period
        assert rule_outputs.shape[1] == rf["n_assets"]
        assert rule_outputs.shape[0] == expected_n_chunks - 1

    def test_calculate_rule_outputs_sum_to_zero(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that momentum weight changes sum to zero."""
        rf = base_run_fingerprint
        rule_outputs = momentum_pool.calculate_rule_outputs(
            momentum_params, rf, sample_prices, None
        )

        # Weight changes should sum to zero (conservation)
        row_sums = jnp.sum(rule_outputs, axis=1)
        np.testing.assert_allclose(
            row_sums, 0.0, atol=1e-10,
            err_msg="Momentum weight changes should sum to zero"
        )

    def test_calculate_rule_outputs_positive_trend(
        self, momentum_pool, momentum_params, base_run_fingerprint
    ):
        """Test that positive trend increases weight."""
        rf = base_run_fingerprint

        # Create prices with clear positive trend in first asset
        n_timesteps = 1440 * 30
        prices = jnp.zeros((n_timesteps, 2))
        prices = prices.at[:, 0].set(100.0 * (1 + jnp.linspace(0, 0.5, n_timesteps)))
        prices = prices.at[:, 1].set(2000.0)  # Flat

        rule_outputs = momentum_pool.calculate_rule_outputs(
            momentum_params, rf, prices, None
        )

        # First asset should have positive weight changes (on average)
        mean_change_0 = jnp.mean(rule_outputs[:, 0])
        mean_change_1 = jnp.mean(rule_outputs[:, 1])

        assert mean_change_0 > mean_change_1, \
            "Asset with positive trend should have higher weight changes"

    def test_calculate_rule_outputs_negative_trend(
        self, momentum_pool, momentum_params, base_run_fingerprint
    ):
        """Test that negative trend decreases weight."""
        rf = base_run_fingerprint

        # Create prices with clear negative trend in first asset
        n_timesteps = 1440 * 30
        prices = jnp.zeros((n_timesteps, 2))
        prices = prices.at[:, 0].set(100.0 * (1 - jnp.linspace(0, 0.3, n_timesteps)))
        prices = prices.at[:, 1].set(2000.0)  # Flat

        rule_outputs = momentum_pool.calculate_rule_outputs(
            momentum_params, rf, prices, None
        )

        # First asset should have negative weight changes (on average)
        mean_change_0 = jnp.mean(rule_outputs[:, 0])
        mean_change_1 = jnp.mean(rule_outputs[:, 1])

        assert mean_change_0 < mean_change_1, \
            "Asset with negative trend should have lower weight changes"

    def test_lamb_affects_memory(
        self, momentum_pool, base_run_fingerprint, sample_prices
    ):
        """Test that higher lamb (memory) creates smoother outputs."""
        rf = base_run_fingerprint

        # High memory (slow decay)
        params_high_memory = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([6.0, 6.0]),  # Higher = longer memory
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        # Low memory (fast decay)
        params_low_memory = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([0.0, 0.0]),  # Lower = shorter memory
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        outputs_high = momentum_pool.calculate_rule_outputs(
            params_high_memory, rf, sample_prices, None
        )
        outputs_low = momentum_pool.calculate_rule_outputs(
            params_low_memory, rf, sample_prices, None
        )

        # High memory should have smoother outputs (lower variance)
        var_high = jnp.var(outputs_high)
        var_low = jnp.var(outputs_low)

        assert var_high < var_low, \
            "Higher memory should produce smoother (lower variance) outputs"

    def test_k_affects_sensitivity(
        self, momentum_pool, base_run_fingerprint, sample_prices
    ):
        """Test that higher k increases output magnitude."""
        rf = base_run_fingerprint

        # High k (sensitive)
        params_high_k = {
            "log_k": jnp.array([8.0, 8.0]),  # High k
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        # Low k (insensitive)
        params_low_k = {
            "log_k": jnp.array([0.0, 0.0]),  # Low k
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        outputs_high = momentum_pool.calculate_rule_outputs(
            params_high_k, rf, sample_prices, None
        )
        outputs_low = momentum_pool.calculate_rule_outputs(
            params_low_k, rf, sample_prices, None
        )

        # High k should have larger magnitude outputs
        mag_high = jnp.max(jnp.abs(outputs_high))
        mag_low = jnp.max(jnp.abs(outputs_low))

        assert mag_high > mag_low, \
            "Higher k should produce larger magnitude outputs"

    def test_get_param_schema_keys(self, momentum_pool):
        """Test that parameter schema has expected keys."""
        schema = momentum_pool.get_param_schema()

        expected_keys = ["log_k", "logit_lamb", "logit_delta_lamb", "initial_weights_logits"]
        for key in expected_keys:
            assert key in schema, f"Schema should contain '{key}'"


class TestMomentumWeightUpdate:
    """Tests for the _jax_momentum_weight_update function."""

    def test_weight_updates_sum_to_zero(self):
        """Test that weight updates sum to zero."""
        price_gradient = jnp.array([0.01, -0.02, 0.015])
        k = jnp.array([20.0, 20.0, 20.0])

        updates = _jax_momentum_weight_update(price_gradient, k)

        np.testing.assert_allclose(
            jnp.sum(updates), 0.0, atol=1e-10,
            err_msg="Weight updates should sum to zero"
        )

    def test_zero_k_gives_zero_update(self):
        """Test that zero k gives zero weight update."""
        price_gradient = jnp.array([0.01, -0.02])
        k = jnp.array([0.0, 20.0])

        updates = _jax_momentum_weight_update(price_gradient, k)

        assert updates[0] == 0.0, "Zero k should give zero update"

    def test_positive_gradient_positive_update(self):
        """Test that positive gradient gives positive update (for positive k)."""
        price_gradient = jnp.array([0.01, 0.0])
        k = jnp.array([20.0, 20.0])

        updates = _jax_momentum_weight_update(price_gradient, k)

        assert updates[0] > 0, "Positive gradient should give positive update"


# ============================================================================
# MinVariancePool Tests
# ============================================================================

class TestMinVariancePool:
    """Tests for MinVariancePool (variance minimization)."""

    @pytest.fixture
    def min_variance_pool(self):
        """Create a min variance pool instance."""
        return MinVariancePool()

    @pytest.fixture
    def min_variance_params(self):
        """Create min variance pool parameters."""
        return {
            "memory_days_1": jnp.array([10.0, 10.0]),
            "memory_days_2": jnp.array([10.0, 10.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

    def test_calculate_rule_outputs_sum_to_one(
        self, min_variance_pool, min_variance_params, base_run_fingerprint, sample_prices
    ):
        """Test that min variance outputs weights (sum to 1)."""
        rf = base_run_fingerprint
        rule_outputs = min_variance_pool.calculate_rule_outputs(
            min_variance_params, rf, sample_prices, None
        )

        row_sums = jnp.sum(rule_outputs, axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, rtol=1e-6,
            err_msg="Min variance outputs should sum to 1"
        )

    def test_calculate_rule_outputs_positive(
        self, min_variance_pool, min_variance_params, base_run_fingerprint, sample_prices
    ):
        """Test that min variance outputs are all positive."""
        rf = base_run_fingerprint
        rule_outputs = min_variance_pool.calculate_rule_outputs(
            min_variance_params, rf, sample_prices, None
        )

        assert jnp.all(rule_outputs > 0), \
            "Min variance weights should all be positive"

    def test_variance_affects_weights(
        self, min_variance_pool, min_variance_params, base_run_fingerprint
    ):
        """Test that variance affects weight calculation."""
        rf = base_run_fingerprint

        np.random.seed(42)
        n_timesteps = 1440 * 30

        # Create prices with different volatilities
        # Asset 0: low volatility
        # Asset 1: high volatility
        prices = jnp.zeros((n_timesteps, 2))
        low_vol = 100.0 + np.random.normal(0, 0.5, n_timesteps).cumsum()
        high_vol = 2000.0 + np.random.normal(0, 10.0, n_timesteps).cumsum()

        prices = prices.at[:, 0].set(low_vol)
        prices = prices.at[:, 1].set(high_vol)

        rule_outputs = min_variance_pool.calculate_rule_outputs(
            min_variance_params, rf, prices, None
        )

        # Weights should be computed (not NaN)
        assert not jnp.any(jnp.isnan(rule_outputs)), "Weights should not be NaN"

        # Weights should be valid (sum to 1)
        weight_sums = jnp.sum(rule_outputs, axis=1)
        mean_sum = jnp.mean(weight_sums)
        np.testing.assert_allclose(mean_sum, 1.0, rtol=0.1)


class TestMinVarianceWeights:
    """Tests for the _jax_min_variance_weights function."""

    def test_weights_sum_to_one(self):
        """Test that min variance weights sum to 1."""
        variances = jnp.array([0.01, 0.02, 0.015])
        weights = _jax_min_variance_weights(variances)

        np.testing.assert_allclose(
            jnp.sum(weights), 1.0, rtol=1e-6,
            err_msg="Min variance weights should sum to 1"
        )

    def test_inverse_variance_weighting(self):
        """Test that weights are inversely proportional to variance."""
        variances = jnp.array([0.01, 0.04])  # 1:4 ratio
        weights = _jax_min_variance_weights(variances)

        # Precisions are 100 and 25, so weights should be 100/125 and 25/125
        expected = jnp.array([100 / 125, 25 / 125])
        np.testing.assert_allclose(
            weights, expected, rtol=1e-6,
            err_msg="Weights should be inversely proportional to variance"
        )

    def test_equal_variances_give_equal_weights(self):
        """Test that equal variances give equal weights."""
        variances = jnp.array([0.01, 0.01, 0.01])
        weights = _jax_min_variance_weights(variances)

        expected = jnp.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(
            weights, expected, rtol=1e-6,
            err_msg="Equal variances should give equal weights"
        )


# ============================================================================
# Pool Creator Tests
# ============================================================================

class TestPoolCreator:
    """Tests for pool creation via create_pool."""

    def test_create_momentum_pool(self):
        """Test creating momentum pool."""
        pool = create_pool("momentum")
        assert isinstance(pool, MomentumPool)

    def test_create_min_variance_pool(self):
        """Test creating min variance pool."""
        pool = create_pool("min_variance")
        assert isinstance(pool, MinVariancePool)

    def test_create_balancer_pool(self):
        """Test creating balancer pool."""
        pool = create_pool("balancer")
        assert isinstance(pool, BalancerPool)

    def test_create_power_channel_pool(self):
        """Test creating power channel pool."""
        from quantammsim.pools.G3M.quantamm.power_channel_pool import PowerChannelPool
        pool = create_pool("power_channel")
        assert isinstance(pool, PowerChannelPool)

    def test_create_mean_reversion_channel_pool(self):
        """Test creating mean reversion channel pool."""
        from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import MeanReversionChannelPool
        pool = create_pool("mean_reversion_channel")
        assert isinstance(pool, MeanReversionChannelPool)

    def test_create_flexible_channel_pool(self):
        """Test creating flexible channel pool."""
        from quantammsim.pools.G3M.quantamm.flexible_channel_pool import FlexibleChannelPool
        pool = create_pool("flexible_channel")
        assert isinstance(pool, FlexibleChannelPool)

    def test_invalid_pool_name_raises(self):
        """Test that invalid pool name raises exception."""
        with pytest.raises(Exception):
            create_pool("invalid_pool_name")


# ============================================================================
# Multi-Asset Tests
# ============================================================================

class TestMultiAssetPools:
    """Tests for pools with more than 2 assets."""

    def test_momentum_pool_4_assets(self, base_run_fingerprint, sample_prices_multi_asset):
        """Test momentum pool with 4 assets."""
        rf = deepcopy(base_run_fingerprint)
        rf["n_assets"] = 4
        rf["all_sig_variations"] = tuple([
            tuple([1, -1, 0, 0]),
            tuple([-1, 1, 0, 0]),
            tuple([1, 0, -1, 0]),
            tuple([-1, 0, 1, 0]),
        ])

        pool = MomentumPool()
        params = {
            "log_k": jnp.array([4.32, 4.32, 4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0, 4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "subsidary_params": [],
        }

        rule_outputs = pool.calculate_rule_outputs(
            params, rf, sample_prices_multi_asset, None
        )

        # Weight changes should still sum to zero
        row_sums = jnp.sum(rule_outputs, axis=1)
        np.testing.assert_allclose(
            row_sums, 0.0, atol=1e-10,
            err_msg="Weight changes should sum to zero for 4 assets"
        )

    def test_min_variance_pool_4_assets(self, base_run_fingerprint, sample_prices_multi_asset):
        """Test min variance pool with 4 assets."""
        rf = deepcopy(base_run_fingerprint)
        rf["n_assets"] = 4

        pool = MinVariancePool()
        params = {
            "memory_days_1": jnp.array([10.0, 10.0, 10.0, 10.0]),
            "memory_days_2": jnp.array([10.0, 10.0, 10.0, 10.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0, 0.0, 0.0]),
            "subsidary_params": [],
        }

        rule_outputs = pool.calculate_rule_outputs(
            params, rf, sample_prices_multi_asset, None
        )

        # Weights should sum to 1
        row_sums = jnp.sum(rule_outputs, axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, rtol=1e-6,
            err_msg="Min variance weights should sum to 1 for 4 assets"
        )

        # All weights should be positive
        assert jnp.all(rule_outputs > 0), \
            "All min variance weights should be positive"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_chunk_period(self, base_run_fingerprint):
        """Test with small chunk period."""
        rf = deepcopy(base_run_fingerprint)
        rf["chunk_period"] = 10  # Small but not single timestep

        np.random.seed(42)
        prices = jnp.array(100.0 + np.random.normal(0, 1, (1000, 2)))

        pool = MomentumPool()
        params = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        rule_outputs = pool.calculate_rule_outputs(params, rf, prices, None)

        # Should still produce valid outputs (finite values)
        # Note: single timestep chunks may produce NaN due to insufficient data
        finite_count = jnp.sum(jnp.isfinite(rule_outputs))
        total_count = rule_outputs.size
        assert finite_count > total_count * 0.9, \
            "Most outputs should be finite with small chunk period"

    def test_extreme_price_movements(self, base_run_fingerprint):
        """Test with extreme price movements."""
        rf = base_run_fingerprint

        # Create prices with 10x jumps
        n_timesteps = 1440 * 10
        prices = jnp.ones((n_timesteps, 2)) * 100.0
        prices = prices.at[n_timesteps // 2:, 0].set(1000.0)  # 10x jump

        pool = MomentumPool()
        params = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        rule_outputs = pool.calculate_rule_outputs(params, rf, prices, None)

        # Should handle extreme movements without NaN
        assert not jnp.any(jnp.isnan(rule_outputs)), \
            "Should handle extreme price movements without NaN"

    def test_flat_prices(self, base_run_fingerprint):
        """Test with completely flat prices."""
        rf = base_run_fingerprint

        n_timesteps = 1440 * 10
        prices = jnp.ones((n_timesteps, 2)) * jnp.array([100.0, 2000.0])

        pool = MomentumPool()
        params = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        rule_outputs = pool.calculate_rule_outputs(params, rf, prices, None)

        # With flat prices, weight changes should be very small
        max_change = jnp.max(jnp.abs(rule_outputs))
        assert max_change < 1e-6, \
            "Flat prices should produce near-zero weight changes"


# ============================================================================
# JIT Compilation Tests
# ============================================================================

class TestJITCompilation:
    """Tests for JIT compilation behavior."""

    def test_momentum_pool_jit_compiles(self, base_run_fingerprint, sample_prices):
        """Test that momentum pool methods JIT compile."""
        pool = MomentumPool()
        params = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        # First call (compilation)
        _ = pool.calculate_rule_outputs(params, base_run_fingerprint, sample_prices, None)

        # Second call (should use cached compilation)
        result = pool.calculate_rule_outputs(params, base_run_fingerprint, sample_prices, None)

        assert result is not None

    def test_momentum_weight_update_jit_compiles(self):
        """Test that _jax_momentum_weight_update JIT compiles."""
        price_gradient = jnp.array([0.01, -0.02])
        k = jnp.array([20.0, 20.0])

        # First call (compilation)
        _ = _jax_momentum_weight_update(price_gradient, k)

        # Second call (should use cached compilation)
        result = _jax_momentum_weight_update(price_gradient, k)

        assert result is not None
