"""
Comprehensive tests for TFMMBasePool class.

Tests cover:
- Fine weight output shape and normalization
- Weight bounds and guardrails
- Weight calculation methods (scan vs vectorized)
- Weight interpolation (linear and non-linear)
- Path support detection
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax
from copy import deepcopy

from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool
from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool
from quantammsim.pools.creator import create_pool
from quantammsim.runners.jax_runner_utils import NestedHashabledict


@pytest.fixture
def base_run_fingerprint():
    """Base run fingerprint for TFMM pool tests."""
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
def momentum_pool():
    """Create a momentum pool for testing."""
    return MomentumPool()


@pytest.fixture
def min_variance_pool():
    """Create a min variance pool for testing."""
    return MinVariancePool()


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n_timesteps = 1440 * 30  # 30 days of minute data
    n_assets = 2

    # Generate realistic price returns with trends
    base_prices = np.array([100.0, 2000.0])  # BTC-like and ETH-like
    trends = np.linspace(0, 0.1, n_timesteps).reshape(-1, 1) + \
             np.random.normal(0, 0.001, (n_timesteps, n_assets)).cumsum(axis=0)

    prices = base_prices * (1 + trends)
    return jnp.array(prices)


@pytest.fixture
def momentum_params(base_run_fingerprint):
    """Create momentum pool parameters."""
    n_assets = base_run_fingerprint["n_assets"]
    n_parameter_sets = 1

    return {
        "log_k": jnp.array([[4.32, 4.32]]),  # log2(20) ≈ 4.32
        "logit_lamb": jnp.array([[4.0, 4.0]]),  # ~10 day memory
        "logit_delta_lamb": jnp.array([[0.0, 0.0]]),
        "initial_weights_logits": jnp.array([[0.0, 0.0]]),  # Equal weights
        "subsidary_params": [],
    }


@pytest.fixture
def min_variance_params(base_run_fingerprint):
    """Create min variance pool parameters."""
    n_assets = base_run_fingerprint["n_assets"]
    n_parameter_sets = 1

    return {
        "memory_days_1": jnp.array([[10.0, 10.0]]),
        "memory_days_2": jnp.array([[10.0, 10.0]]),
        "initial_weights_logits": jnp.array([[0.0, 0.0]]),
        "subsidary_params": [],
    }


class TestTFMMBasePoolWeightCalculation:
    """Tests for weight calculation methods."""

    def test_calculate_weights_output_shape(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that calculated weights have correct shape."""
        rf = base_run_fingerprint
        bout_length = rf["bout_length"]
        n_assets = rf["n_assets"]

        # Single parameter set
        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])  # Start after 10 days burn-in

        weights = momentum_pool.calculate_weights(
            params, rf, sample_prices, start_index, None
        )

        assert weights.shape == (bout_length - 1, n_assets), \
            f"Expected shape ({bout_length - 1}, {n_assets}), got {weights.shape}"

    def test_calculate_weights_sum_to_one(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that weights sum to 1 at each timestep."""
        rf = base_run_fingerprint
        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        weights = momentum_pool.calculate_weights(
            params, rf, sample_prices, start_index, None
        )

        weight_sums = jnp.sum(weights, axis=1)
        np.testing.assert_allclose(
            weight_sums, 1.0, rtol=1e-6,
            err_msg="Weights should sum to 1 at each timestep"
        )

    def test_calculate_weights_respect_minimum(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that weights respect minimum weight constraint."""
        rf = base_run_fingerprint
        minimum_weight = rf["minimum_weight"]

        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        weights = momentum_pool.calculate_weights(
            params, rf, sample_prices, start_index, None
        )

        assert jnp.all(weights >= minimum_weight - 1e-6), \
            f"All weights should be >= minimum_weight ({minimum_weight})"

    def test_calculate_weights_vectorized_equals_scan_momentum(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that vectorized and scan paths produce same results for momentum pool."""
        rf_vectorized = deepcopy(base_run_fingerprint)
        rf_vectorized["weight_calculation_method"] = "vectorized"

        rf_scan = deepcopy(base_run_fingerprint)
        rf_scan["weight_calculation_method"] = "scan"

        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        weights_vectorized = momentum_pool.calculate_weights(
            params, rf_vectorized, sample_prices, start_index, None
        )

        weights_scan = momentum_pool.calculate_weights(
            params, rf_scan, sample_prices, start_index, None
        )

        np.testing.assert_allclose(
            weights_vectorized, weights_scan, rtol=1e-4, atol=1e-6,
            err_msg="Vectorized and scan paths should produce same results"
        )


class TestTFMMBasePoolWeightInterpolation:
    """Tests for weight interpolation methods."""

    def test_linear_interpolation_smooth(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that linear interpolation produces smooth weight transitions."""
        rf = deepcopy(base_run_fingerprint)
        rf["weight_interpolation_method"] = "linear"

        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        weights = momentum_pool.calculate_weights(
            params, rf, sample_prices, start_index, None
        )

        # Check that weight changes are smooth (no large jumps)
        weight_diffs = jnp.diff(weights, axis=0)
        max_diff = jnp.max(jnp.abs(weight_diffs))

        # Maximum change should be close to maximum_change / weight_interpolation_period
        expected_max = rf["maximum_change"]
        assert max_diff <= expected_max + 1e-6, \
            f"Weight changes should respect maximum_change constraint"

    def test_approx_optimal_interpolation(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that approx_optimal interpolation works."""
        rf = deepcopy(base_run_fingerprint)
        rf["weight_interpolation_method"] = "approx_optimal"

        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        weights = momentum_pool.calculate_weights(
            params, rf, sample_prices, start_index, None
        )

        # Should still sum to 1
        weight_sums = jnp.sum(weights, axis=1)
        np.testing.assert_allclose(
            weight_sums, 1.0, rtol=1e-6,
            err_msg="Weights should sum to 1 with approx_optimal interpolation"
        )


class TestTFMMBasePoolGuardrails:
    """Tests for weight guardrails."""

    def test_maximum_change_affects_smoothness(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that maximum_change constraint affects weight smoothness."""
        rf_restrictive = deepcopy(base_run_fingerprint)
        rf_restrictive["maximum_change"] = 0.0001  # Very restrictive

        rf_permissive = deepcopy(base_run_fingerprint)
        rf_permissive["maximum_change"] = 0.1  # Very permissive

        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        weights_restrictive = momentum_pool.calculate_weights(
            params, rf_restrictive, sample_prices, start_index, None
        )
        weights_permissive = momentum_pool.calculate_weights(
            params, rf_permissive, sample_prices, start_index, None
        )

        # With restrictive maximum_change, weight changes should be smaller
        diffs_restrictive = jnp.diff(weights_restrictive, axis=0)
        diffs_permissive = jnp.diff(weights_permissive, axis=0)

        max_restrictive = jnp.max(jnp.abs(diffs_restrictive))
        max_permissive = jnp.max(jnp.abs(diffs_permissive))

        # Restrictive should have smaller or equal max change
        assert max_restrictive <= max_permissive + 1e-6, \
            "Restrictive maximum_change should produce smoother weights"

    def test_minimum_weight_with_extreme_params(
        self, momentum_pool, base_run_fingerprint, sample_prices
    ):
        """Test minimum weight enforcement with extreme k values."""
        rf = base_run_fingerprint

        # Very high k should push weights to extremes, but min weight should hold
        params = {
            "log_k": jnp.array([12.0, 12.0]),  # Very high k
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        start_index = jnp.array([1440 * 10, 0])

        weights = momentum_pool.calculate_weights(
            params, rf, sample_prices, start_index, None
        )

        minimum_weight = rf["minimum_weight"]
        assert jnp.all(weights >= minimum_weight - 1e-6), \
            "Minimum weight should be enforced even with extreme params"


class TestTFMMBasePoolPathSupport:
    """Tests for path support detection."""

    def test_momentum_pool_supports_vectorized(self, momentum_pool):
        """Test that momentum pool supports vectorized path."""
        assert momentum_pool.supports_vectorized_path(), \
            "MomentumPool should support vectorized path"

    def test_momentum_pool_supports_scan(self, momentum_pool):
        """Test that momentum pool supports scan path."""
        assert momentum_pool.supports_scan_path(), \
            "MomentumPool should support scan path"

    def test_min_variance_pool_supports_vectorized(self, min_variance_pool):
        """Test that min variance pool supports vectorized path."""
        assert min_variance_pool.supports_vectorized_path(), \
            "MinVariancePool should support vectorized path"

    def test_is_trainable(self, momentum_pool, min_variance_pool):
        """Test that TFMM pools are trainable."""
        assert momentum_pool.is_trainable(), \
            "MomentumPool should be trainable"
        assert min_variance_pool.is_trainable(), \
            "MinVariancePool should be trainable"


class TestTFMMBasePoolReservesCalculation:
    """Tests for reserve calculation methods."""

    def test_calculate_reserves_with_fees_shape(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that calculate_reserves_with_fees returns correct shape."""
        rf = base_run_fingerprint

        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        reserves = momentum_pool.calculate_reserves_with_fees(
            params, rf, sample_prices, start_index, None
        )

        expected_length = rf["bout_length"] - 1
        if rf["arb_frequency"] != 1:
            expected_length = expected_length // rf["arb_frequency"]

        assert reserves.shape[0] == expected_length
        assert reserves.shape[1] == rf["n_assets"]

    def test_calculate_reserves_zero_fees_shape(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that calculate_reserves_zero_fees returns correct shape."""
        rf = base_run_fingerprint

        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        reserves = momentum_pool.calculate_reserves_zero_fees(
            params, rf, sample_prices, start_index, None
        )

        expected_length = rf["bout_length"] - 1
        if rf["arb_frequency"] != 1:
            expected_length = expected_length // rf["arb_frequency"]

        assert reserves.shape[0] == expected_length
        assert reserves.shape[1] == rf["n_assets"]

    def test_reserves_positive(
        self, momentum_pool, momentum_params, base_run_fingerprint, sample_prices
    ):
        """Test that all reserves remain positive."""
        rf = base_run_fingerprint

        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        start_index = jnp.array([1440 * 10, 0])

        reserves = momentum_pool.calculate_reserves_with_fees(
            params, rf, sample_prices, start_index, None
        )

        assert jnp.all(reserves > 0), "All reserves should be positive"


class TestTFMMBasePoolInitialWeights:
    """Tests for initial weight calculation."""

    def test_calculate_initial_weights_sum_to_one(
        self, momentum_pool, momentum_params
    ):
        """Test that initial weights sum to 1."""
        params = {k: v[0] for k, v in momentum_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        initial_weights = momentum_pool.calculate_initial_weights(params)

        np.testing.assert_allclose(
            jnp.sum(initial_weights), 1.0, rtol=1e-6,
            err_msg="Initial weights should sum to 1"
        )

    def test_calculate_initial_weights_equal_logits(
        self, momentum_pool
    ):
        """Test that equal logits give equal weights."""
        params = {
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        initial_weights = momentum_pool.calculate_initial_weights(params)

        np.testing.assert_allclose(
            initial_weights, [0.5, 0.5], rtol=1e-6,
            err_msg="Equal logits should give equal weights"
        )

    def test_calculate_initial_weights_unequal_logits(
        self, momentum_pool
    ):
        """Test that unequal logits give unequal weights."""
        params = {
            "initial_weights_logits": jnp.array([1.0, 0.0]),  # First asset higher
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
            "subsidary_params": [],
        }

        initial_weights = momentum_pool.calculate_initial_weights(params)

        assert initial_weights[0] > initial_weights[1], \
            "Higher logit should give higher weight"
        np.testing.assert_allclose(
            jnp.sum(initial_weights), 1.0, rtol=1e-6,
            err_msg="Weights should still sum to 1"
        )


class TestTFMMBasePoolParameterInitialization:
    """Tests for parameter initialization."""

    def test_init_base_parameters_shape(
        self, momentum_pool, base_run_fingerprint
    ):
        """Test that init_base_parameters returns correct shapes."""
        n_assets = 3
        n_parameter_sets = 2

        initial_values = {
            "initial_weights_logits": 1.0,
            "initial_k_per_day": 20.0,
            "initial_memory_length": 10.0,
            "initial_memory_length_delta": 0.0,
        }

        rf = deepcopy(base_run_fingerprint)
        rf["n_assets"] = n_assets
        rf["optimisation_settings"] = {"force_scalar": False}

        params = momentum_pool.init_base_parameters(
            initial_values, rf, n_assets, n_parameter_sets, noise="none"
        )

        assert params["log_k"].shape == (n_parameter_sets, n_assets)
        assert params["logit_lamb"].shape == (n_parameter_sets, n_assets)
        assert params["initial_weights_logits"].shape == (n_parameter_sets, n_assets)

    def test_init_base_parameters_with_gaussian_noise(
        self, momentum_pool, base_run_fingerprint
    ):
        """Test that gaussian noise is applied to parameters."""
        n_assets = 2
        n_parameter_sets = 5

        initial_values = {
            "initial_weights_logits": 1.0,
            "initial_k_per_day": 20.0,
            "initial_memory_length": 10.0,
            "initial_memory_length_delta": 0.0,
        }

        rf = deepcopy(base_run_fingerprint)
        rf["optimisation_settings"] = {"force_scalar": False}

        params = momentum_pool.init_base_parameters(
            initial_values, rf, n_assets, n_parameter_sets, noise="gaussian"
        )

        # With gaussian noise, parameter sets should differ
        log_k_std = jnp.std(params["log_k"], axis=0)
        assert jnp.any(log_k_std > 0), \
            "Gaussian noise should create variation across parameter sets"


class TestTFMMBasePoolRuleOutputStep:
    """Tests for single-step rule output calculation."""

    def test_get_initial_rule_state_keys(
        self, momentum_pool, base_run_fingerprint
    ):
        """Test that get_initial_rule_state returns expected keys."""
        initial_price = jnp.array([100.0, 2000.0])
        params = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
        }

        initial_state = momentum_pool.get_initial_rule_state(
            initial_price, params, base_run_fingerprint
        )

        assert "ewma" in initial_state
        assert "running_a" in initial_state

    def test_calculate_rule_output_step_returns_tuple(
        self, momentum_pool, base_run_fingerprint
    ):
        """Test that calculate_rule_output_step returns (carry, output) tuple."""
        initial_price = jnp.array([100.0, 2000.0])
        params = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
        }

        carry = momentum_pool.get_initial_rule_state(
            initial_price, params, base_run_fingerprint
        )

        new_price = jnp.array([101.0, 2020.0])
        new_carry, rule_output = momentum_pool.calculate_rule_output_step(
            carry, new_price, params, base_run_fingerprint
        )

        assert isinstance(new_carry, dict)
        assert "ewma" in new_carry
        assert "running_a" in new_carry
        assert rule_output.shape == (2,)

    def test_rule_output_step_sums_to_zero(
        self, momentum_pool, base_run_fingerprint
    ):
        """Test that momentum rule output sums to zero (weight changes)."""
        initial_price = jnp.array([100.0, 2000.0])
        params = {
            "log_k": jnp.array([4.32, 4.32]),
            "logit_lamb": jnp.array([4.0, 4.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0]),
        }

        carry = momentum_pool.get_initial_rule_state(
            initial_price, params, base_run_fingerprint
        )

        new_price = jnp.array([101.0, 2020.0])
        _, rule_output = momentum_pool.calculate_rule_output_step(
            carry, new_price, params, base_run_fingerprint
        )

        np.testing.assert_allclose(
            jnp.sum(rule_output), 0.0, atol=1e-10,
            err_msg="Momentum weight changes should sum to zero"
        )


class TestTFMMBasePoolMinVariance:
    """Tests specific to MinVariancePool behavior."""

    def test_min_variance_outputs_weights_not_changes(
        self, min_variance_pool, min_variance_params, base_run_fingerprint, sample_prices
    ):
        """Test that min variance pool outputs weights directly."""
        rf = base_run_fingerprint
        params = {k: v[0] for k, v in min_variance_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        rule_outputs = min_variance_pool.calculate_rule_outputs(
            params, rf, sample_prices, None
        )

        # Rule outputs should sum to 1 (they are weights, not changes)
        output_sums = jnp.sum(rule_outputs, axis=1)
        np.testing.assert_allclose(
            output_sums, 1.0, rtol=1e-6,
            err_msg="Min variance rule outputs should sum to 1"
        )

    def test_min_variance_weights_positive(
        self, min_variance_pool, min_variance_params, base_run_fingerprint, sample_prices
    ):
        """Test that min variance produces positive weights."""
        rf = base_run_fingerprint
        params = {k: v[0] for k, v in min_variance_params.items() if k != "subsidary_params"}
        params["subsidary_params"] = []

        rule_outputs = min_variance_pool.calculate_rule_outputs(
            params, rf, sample_prices, None
        )

        assert jnp.all(rule_outputs > 0), \
            "Min variance weights should all be positive"
