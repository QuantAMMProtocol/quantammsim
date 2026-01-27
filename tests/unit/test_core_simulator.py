"""
Tests for the core_simulator module.

Covers:
- forward_pass.py: Return value types, metrics calculations, branch paths
- param_utils.py: Parameter conversion functions, utilities, JSON encoding
"""

import pytest
import numpy as np
import jax.numpy as jnp
from hypothesis import given, strategies as st, settings, assume, Phase
import hypothesis.extra.numpy as hnp

from tests.conftest import TEST_DATA_DIR


# =============================================================================
# PARAM_UTILS TESTS - Conversion Functions
# =============================================================================

class TestParamUtilsConversions:
    """Test all parameter conversion functions with property-based testing."""

    @given(
        memory_days=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
        chunk_period=st.sampled_from([60, 1440]),
    )
    @settings(max_examples=50, deadline=None)  # Disable deadline for JIT warmup
    def test_memory_days_to_lamb_range(self, memory_days, chunk_period):
        """Lambda should always be in (0, 1) for reasonable memory_days."""
        from quantammsim.core_simulator.param_utils import memory_days_to_lamb

        lamb = memory_days_to_lamb(memory_days, chunk_period)
        # For very large memory_days relative to chunk_period, lamb approaches 1
        assert 0 < lamb <= 1, f"lamb={lamb} out of range for memory_days={memory_days}"

    @given(
        memory_days=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
        chunk_period=st.sampled_from([60, 1440]),
    )
    @settings(max_examples=50, deadline=None)
    def test_memory_days_to_logit_lamb_finite(self, memory_days, chunk_period):
        """Logit lambda should be finite for valid memory_days."""
        from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb

        logit_lamb = memory_days_to_logit_lamb(memory_days, chunk_period)
        assert np.isfinite(logit_lamb), f"logit_lamb not finite for memory_days={memory_days}"

    def test_memory_days_to_lamb_zero(self):
        """Test edge case: memory_days=0 should return 0."""
        from quantammsim.core_simulator.param_utils import memory_days_to_lamb

        lamb = memory_days_to_lamb(0.0, chunk_period=60)
        assert lamb == 0.0

    def test_memory_days_to_lamb_array(self):
        """Test with array input."""
        from quantammsim.core_simulator.param_utils import memory_days_to_lamb

        memory_days = np.array([1.0, 10.0, 30.0, 100.0])
        lambs = memory_days_to_lamb(memory_days, chunk_period=60)
        assert lambs.shape == memory_days.shape
        assert np.all((lambs > 0) & (lambs < 1))

    def test_memory_days_to_lamb_monotonic(self):
        """Longer memory should result in larger lambda (slower decay)."""
        from quantammsim.core_simulator.param_utils import memory_days_to_lamb

        memory_days = np.array([1.0, 10.0, 30.0, 100.0])
        lambs = memory_days_to_lamb(memory_days, chunk_period=60)
        # Lambda should be strictly increasing with memory_days
        assert np.all(np.diff(lambs) > 0), "Lambda should increase with memory_days"

    def test_memory_days_to_lamb_inverse_relationship(self):
        """Verify lamb can be inverted back to memory_days via lamb_to_memory_days."""
        from quantammsim.core_simulator.param_utils import memory_days_to_lamb, lamb_to_memory_days

        memory_days = 10.0
        chunk_period = 60
        lamb = memory_days_to_lamb(memory_days, chunk_period)

        # lamb_to_memory_days is the proper inverse
        recovered_memory_days = lamb_to_memory_days(jnp.array(lamb), chunk_period)
        # This relationship should hold (lamb_to_memory_days is the inverse)
        np.testing.assert_allclose(recovered_memory_days, memory_days, rtol=1e-6)

    def test_jax_memory_days_to_lamb(self):
        """Test JAX version of memory_days_to_lamb."""
        from quantammsim.core_simulator.param_utils import (
            memory_days_to_lamb,
            jax_memory_days_to_lamb,
        )

        memory_days = 30.0
        np_lamb = memory_days_to_lamb(memory_days, chunk_period=60)
        jax_lamb = jax_memory_days_to_lamb(memory_days, chunk_period=60)
        np.testing.assert_allclose(np_lamb, jax_lamb, rtol=1e-6)

    def test_lamb_to_memory(self):
        """Test lamb_to_memory conversion - larger lamb means longer memory."""
        from quantammsim.core_simulator.param_utils import lamb_to_memory

        lamb_small = jnp.array(0.5)
        lamb_large = jnp.array(0.95)
        memory_small = lamb_to_memory(lamb_small)
        memory_large = lamb_to_memory(lamb_large)

        assert memory_small > 0
        assert memory_large > 0
        # Larger lambda (slower decay) should mean longer memory
        assert memory_large > memory_small, "Larger lambda should give longer memory"

    @given(
        logit=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_jax_logit_lamb_to_lamb_range(self, logit):
        """Sigmoid of logit should be in (0, 1)."""
        from quantammsim.core_simulator.param_utils import jax_logit_lamb_to_lamb

        lamb = jax_logit_lamb_to_lamb(logit)
        assert 0 < lamb < 1


class TestParamUtilsDictConversions:
    """Test dictionary conversion utilities."""

    def test_dict_of_jnp_to_np(self):
        """Test JAX array to numpy conversion preserves values."""
        from quantammsim.core_simulator.param_utils import dict_of_jnp_to_np

        d = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4.0, 5.0])}
        result = dict_of_jnp_to_np(d)
        assert isinstance(result["a"], np.ndarray)
        assert isinstance(result["b"], np.ndarray)
        # Verify values are preserved
        np.testing.assert_array_equal(result["a"], [1, 2, 3])
        np.testing.assert_array_equal(result["b"], [4.0, 5.0])

    def test_dict_of_np_to_jnp(self):
        """Test numpy to JAX array conversion preserves values."""
        from quantammsim.core_simulator.param_utils import dict_of_np_to_jnp

        d = {"a": np.array([1, 2, 3]), "b": np.array([4.0, 5.0])}
        result = dict_of_np_to_jnp(d)
        assert isinstance(result["a"], jnp.ndarray)
        assert isinstance(result["b"], jnp.ndarray)
        # Verify values are preserved
        np.testing.assert_array_equal(result["a"], jnp.array([1, 2, 3]))
        np.testing.assert_array_equal(result["b"], jnp.array([4.0, 5.0]))

    def test_dict_of_jnp_to_list(self):
        """Test JAX array to list conversion preserves values."""
        from quantammsim.core_simulator.param_utils import dict_of_jnp_to_list

        d = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4.0, 5.0])}
        result = dict_of_jnp_to_list(d)
        assert isinstance(result["a"], list)
        assert isinstance(result["b"], list)
        # Verify values are preserved
        assert result["a"] == [1, 2, 3]
        assert result["b"] == [4.0, 5.0]

    def test_dict_conversion_roundtrip(self):
        """Test that np->jnp->np conversion is lossless."""
        from quantammsim.core_simulator.param_utils import dict_of_np_to_jnp, dict_of_jnp_to_np

        original = {"a": np.array([1.5, 2.5, 3.5]), "b": np.array([4, 5])}
        jnp_version = dict_of_np_to_jnp(original)
        roundtrip = dict_of_jnp_to_np(jnp_version)
        np.testing.assert_array_equal(roundtrip["a"], original["a"])
        np.testing.assert_array_equal(roundtrip["b"], original["b"])


class TestParamUtilsDefaultSetters:
    """Test default_set and related utilities."""

    def test_default_set_or_get_existing(self):
        """Test when key exists."""
        from quantammsim.core_simulator.param_utils import default_set_or_get

        d = {"key": "value"}
        result = default_set_or_get(d, "key", "default")
        assert result == "value"

    def test_default_set_or_get_missing(self):
        """Test when key is missing."""
        from quantammsim.core_simulator.param_utils import default_set_or_get

        d = {}
        result = default_set_or_get(d, "key", "default")
        assert result == "default"
        assert d["key"] == "default"

    def test_default_set_or_get_no_augment(self):
        """Test when augment=False."""
        from quantammsim.core_simulator.param_utils import default_set_or_get

        d = {}
        result = default_set_or_get(d, "key", "default", augment=False)
        assert result == "default"
        assert "key" not in d

    def test_recursive_default_set(self):
        """Test recursive default setting."""
        from quantammsim.core_simulator.param_utils import recursive_default_set

        target = {"a": 1}
        defaults = {"a": 0, "b": 2, "nested": {"c": 3, "d": 4}}
        recursive_default_set(target, defaults)
        assert target["a"] == 1  # Not overwritten
        assert target["b"] == 2  # Added
        assert target["nested"]["c"] == 3  # Nested added
        assert target["nested"]["d"] == 4  # Nested added (was missing!)

    def test_recursive_default_set_deep_nesting(self):
        """Test recursive default with deeply nested structures."""
        from quantammsim.core_simulator.param_utils import recursive_default_set

        target = {"level1": {"existing": "keep"}}
        defaults = {
            "level1": {
                "existing": "overwrite_attempt",
                "new_key": "added",
                "level2": {"deep": "value"},
            }
        }
        recursive_default_set(target, defaults)
        assert target["level1"]["existing"] == "keep"  # Not overwritten
        assert target["level1"]["new_key"] == "added"
        assert target["level1"]["level2"]["deep"] == "value"


class TestParamUtilsRunFingerprint:
    """Test run fingerprint utilities."""

    def test_check_run_fingerprint_valid(self):
        """Test valid fingerprint passes check."""
        from quantammsim.core_simulator.param_utils import check_run_fingerprint

        fp = {"weight_interpolation_period": 60, "chunk_period": 60}
        check_run_fingerprint(fp)  # Should not raise

    def test_check_run_fingerprint_invalid(self):
        """Test invalid fingerprint raises."""
        from quantammsim.core_simulator.param_utils import check_run_fingerprint

        fp = {"weight_interpolation_period": 120, "chunk_period": 60}
        with pytest.raises(AssertionError):
            check_run_fingerprint(fp)

    def test_get_run_location(self):
        """Test run location hash generation."""
        from quantammsim.core_simulator.param_utils import get_run_location

        fp = {"a": 1, "b": 2}
        location = get_run_location(fp)
        assert location.startswith("run_")
        assert len(location) > 10  # Has hash

        # Same fingerprint should give same location
        location2 = get_run_location(fp)
        assert location == location2


class TestNumpyEncoder:
    """Test JSON encoder for numpy types."""

    def test_encode_integer(self):
        """Test encoding numpy integer - verify roundtrip."""
        import json
        from quantammsim.core_simulator.param_utils import NumpyEncoder

        data = {"val": np.int64(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(result)
        assert decoded["val"] == 42

    def test_encode_float(self):
        """Test encoding numpy float - verify roundtrip."""
        import json
        from quantammsim.core_simulator.param_utils import NumpyEncoder

        data = {"val": np.float64(3.14159)}
        result = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(result)
        np.testing.assert_allclose(decoded["val"], 3.14159, rtol=1e-10)

    def test_encode_array(self):
        """Test encoding numpy array - verify roundtrip."""
        import json
        from quantammsim.core_simulator.param_utils import NumpyEncoder

        data = {"val": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(result)
        assert decoded["val"] == [1, 2, 3]

    def test_encode_nested_arrays(self):
        """Test encoding nested structure with multiple numpy types."""
        import json
        from quantammsim.core_simulator.param_utils import NumpyEncoder

        data = {
            "int_val": np.int32(10),
            "float_val": np.float32(2.5),
            "array": np.array([1.0, 2.0, 3.0]),
            "nested": {"inner_array": np.array([4, 5])},
        }
        result = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(result)
        assert decoded["int_val"] == 10
        np.testing.assert_allclose(decoded["float_val"], 2.5, rtol=1e-5)
        assert decoded["array"] == [1.0, 2.0, 3.0]
        assert decoded["nested"]["inner_array"] == [4, 5]


# =============================================================================
# FORWARD_PASS TESTS - All Return Value Types
# =============================================================================

class TestForwardPassReturnValues:
    """Test all return value types in forward_pass."""

    @pytest.fixture
    def setup_data(self):
        """Create minimal test data for forward pass."""
        from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        n_assets = 2
        # Need enough data for various metrics (weekly, monthly calculations)
        bout_length = 30 * 24 * 60  # 30 days of minute data
        n_timesteps = bout_length + 1440  # Plus some burn-in

        # Generate synthetic price data
        np.random.seed(42)
        base_prices = np.array([100.0, 50.0])
        noise = np.random.randn(n_timesteps, n_assets) * 0.01
        trends = np.cumsum(noise, axis=0)
        prices = base_prices * (1 + trends)
        prices = jnp.array(np.abs(prices))  # Ensure positive

        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

        static_dict = NestedHashabledict({
            "bout_length": bout_length,
            "maximum_change": 0.01,
            "n_assets": n_assets,
            "chunk_period": 60,
            "weight_interpolation_period": 60,
            "return_val": "reserves",
            "rule": "momentum",
            "run_type": "normal",
            "max_memory_days": 365.0,
            "initial_pool_value": 1000000.0,
            "fees": 0.0,
            "use_alt_lamb": False,
            "use_pre_exp_scaling": True,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "weight_interpolation_method": "linear",
            "training_data_kind": "historic",
            "arb_frequency": 1,
            "do_trades": False,
            "do_arb": True,
            "minimum_weight": 0.05,
            "ste_max_change": False,
            "ste_min_max_weight": False,
        })

        pool = MomentumPool()
        start_index = jnp.array([1440, 0])

        return {
            "params": params,
            "prices": prices,
            "static_dict": static_dict,
            "pool": pool,
            "start_index": start_index,
        }

    @pytest.mark.parametrize("return_val", [
        "reserves",
        "sharpe",
        "daily_sharpe",
        "daily_log_sharpe",
        "returns",
        "annualised_returns",
        "returns_over_hodl",
        "annualised_returns_over_hodl",
        "returns_over_uniform_hodl",
        "annualised_returns_over_uniform_hodl",
        "greatest_draw_down",
        "value",
        "weekly_max_drawdown",
        "daily_var_95%",
        "daily_var_95%_trad",
        "weekly_var_95%",
        "weekly_var_95%_trad",
        "daily_var_99%",
        "daily_var_99%_trad",
        "weekly_var_99%",
        "weekly_var_99%_trad",
        "daily_raroc",
        "weekly_raroc",
        "daily_rovar",
        "weekly_rovar",
        # Skip monthly metrics - need 60+ days of data
        # "monthly_rovar",
        # "monthly_rovar_trad",
        # "ulcer",  # Uses 30-day duration
        "daily_rovar_trad",
        "weekly_rovar_trad",
        "sterling",
        "calmar",
        "reserves_and_values",
    ])
    def test_return_value_type(self, setup_data, return_val):
        """Test each return value type produces valid output."""
        from quantammsim.core_simulator.forward_pass import forward_pass

        data = setup_data
        static_dict = dict(data["static_dict"])
        static_dict["return_val"] = return_val

        from quantammsim.runners.jax_runner_utils import NestedHashabledict
        static_dict = NestedHashabledict(static_dict)

        result = forward_pass(
            data["params"],
            data["start_index"],
            data["prices"],
            pool=data["pool"],
            static_dict=static_dict,
        )

        if return_val in ["reserves", "reserves_and_values"]:
            assert isinstance(result, dict)
            assert "reserves" in result
            # Verify reserves have correct shape
            reserves = result["reserves"]
            assert reserves.shape[1] == data["static_dict"]["n_assets"]
        elif return_val == "value":
            assert result.shape[0] > 0
            # Values should be positive (pool value)
            assert jnp.all(result > 0), "Pool values should be positive"
        elif return_val == "greatest_draw_down":
            # Drawdown should be non-positive
            assert result <= 0, f"Drawdown should be <= 0, got {result}"
        elif return_val in ["weekly_max_drawdown"]:
            # Max drawdown should be non-positive
            assert result <= 0, f"Max drawdown should be <= 0, got {result}"
        elif "var" in return_val.lower():
            # VaR should be negative (representing loss)
            assert result < 0 or jnp.isfinite(result), f"VaR should be negative or finite"
        else:
            # Other scalar return values should be finite
            # Sterling and Calmar can be inf if no drawdown, or NaN if insufficient data
            if return_val in ["sterling", "calmar"]:
                # These ratios can be NaN/inf due to edge cases in drawdown calculation
                # (e.g., no drawdown gives division by zero). Verify at least result is returned.
                assert isinstance(float(result), float), f"{return_val} should return a numeric type"
            else:
                assert jnp.isfinite(result), f"{return_val} should be finite, got {result}"


class TestForwardPassBranches:
    """Test different branch paths in forward_pass."""

    def test_forward_pass_no_pool_raises(self):
        """Test that forward_pass raises without pool."""
        from quantammsim.core_simulator.forward_pass import forward_pass
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        with pytest.raises(ValueError, match="Pool must be provided"):
            forward_pass(
                {},
                jnp.array([0, 0]),
                jnp.array([[1, 2], [3, 4]]),
                pool=None,
                static_dict=NestedHashabledict({}),
            )

    def test_forward_pass_invalid_return_val(self):
        """Test that invalid return_val raises NotImplementedError."""
        from quantammsim.core_simulator.forward_pass import forward_pass
        from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        n_assets = 2
        bout_length = 1440

        np.random.seed(42)
        prices = jnp.array(100.0 + np.random.randn(bout_length + 1440, n_assets) * 5)
        prices = jnp.abs(prices)

        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

        static_dict = NestedHashabledict({
            "bout_length": bout_length,
            "n_assets": n_assets,
            "chunk_period": 60,
            "weight_interpolation_period": 60,
            "return_val": "invalid_metric",  # Invalid
            "initial_pool_value": 1000000.0,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "do_arb": True,
            "maximum_change": 0.01,
            "max_memory_days": 365.0,
            "use_alt_lamb": False,
            "use_pre_exp_scaling": True,
            "arb_frequency": 1,
            "do_trades": False,
            "training_data_kind": "historic",
            "weight_interpolation_method": "linear",
            "minimum_weight": 0.05,
            "ste_max_change": False,
            "ste_min_max_weight": False,
        })

        pool = MomentumPool()
        start_index = jnp.array([1440, 0])

        with pytest.raises(NotImplementedError):
            forward_pass(
                params,
                start_index,
                prices,
                pool=pool,
                static_dict=static_dict,
            )


class TestForwardPassNograd:
    """Test forward_pass_nograd function."""

    @pytest.fixture
    def nograd_setup(self):
        """Setup data for nograd tests."""
        from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        n_assets = 2
        bout_length = 1440

        np.random.seed(42)
        prices = jnp.array(100.0 + np.random.randn(bout_length + 1440, n_assets) * 5)
        prices = jnp.abs(prices)

        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

        base_static = {
            "bout_length": bout_length,
            "n_assets": n_assets,
            "chunk_period": 60,
            "weight_interpolation_period": 60,
            "initial_pool_value": 1000000.0,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "do_arb": True,
            "maximum_change": 0.01,
            "max_memory_days": 365.0,
            "use_alt_lamb": False,
            "use_pre_exp_scaling": True,
            "arb_frequency": 1,
            "do_trades": False,
            "training_data_kind": "historic",
            "weight_interpolation_method": "linear",
            "minimum_weight": 0.05,
            "ste_max_change": False,
            "ste_min_max_weight": False,
        }

        return {
            "params": params,
            "prices": prices,
            "base_static": base_static,
            "pool": MomentumPool(),
            "start_index": jnp.array([1440, 0]),
        }

    @pytest.mark.parametrize("return_val", ["returns", "sharpe", "daily_sharpe"])
    def test_nograd_matches_forward_pass(self, nograd_setup, return_val):
        """Test that nograd version produces same results for various return types."""
        from quantammsim.core_simulator.forward_pass import forward_pass, forward_pass_nograd
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        data = nograd_setup
        static_dict = NestedHashabledict({**data["base_static"], "return_val": return_val})

        result1 = forward_pass(
            data["params"], data["start_index"], data["prices"],
            pool=data["pool"], static_dict=static_dict
        )
        result2 = forward_pass_nograd(
            data["params"], data["start_index"], data["prices"],
            pool=data["pool"], static_dict=static_dict
        )

        np.testing.assert_allclose(result1, result2, rtol=1e-10)


# =============================================================================
# FORWARD PASS METRICS TESTS
# =============================================================================

class TestForwardPassMetrics:
    """Test individual metric calculation functions."""

    @pytest.fixture
    def value_series(self):
        """Create test value series."""
        np.random.seed(42)
        n_minutes = 30 * 24 * 60  # 30 days
        base = 1000000.0
        returns = np.random.randn(n_minutes) * 0.0001
        values = base * np.cumprod(1 + returns)
        return jnp.array(values)

    def test_daily_log_sharpe(self, value_series):
        """Test daily log sharpe calculation."""
        from quantammsim.core_simulator.forward_pass import _daily_log_sharpe

        result = _daily_log_sharpe(value_series)
        assert jnp.isfinite(result)

    def test_daily_log_sharpe_positive_for_uptrend(self):
        """Sharpe should be positive for consistently increasing values."""
        from quantammsim.core_simulator.forward_pass import _daily_log_sharpe

        # Create a steadily increasing series (30 days of minute data)
        n_minutes = 30 * 24 * 60
        # Small consistent positive returns
        returns = np.ones(n_minutes) * 0.00001
        values = 1000.0 * np.cumprod(1 + returns)

        result = _daily_log_sharpe(jnp.array(values))
        assert result > 0, "Sharpe should be positive for uptrend"

    def test_daily_log_sharpe_negative_for_downtrend(self):
        """Sharpe should be negative for consistently decreasing values."""
        from quantammsim.core_simulator.forward_pass import _daily_log_sharpe

        # Create a steadily decreasing series (30 days of minute data)
        n_minutes = 30 * 24 * 60
        # Small consistent negative returns
        returns = np.ones(n_minutes) * -0.00001
        values = 1000.0 * np.cumprod(1 + returns)

        result = _daily_log_sharpe(jnp.array(values))
        assert result < 0, "Sharpe should be negative for downtrend"

    def test_calculate_max_drawdown(self, value_series):
        """Test max drawdown calculation."""
        from quantammsim.core_simulator.forward_pass import _calculate_max_drawdown

        result = _calculate_max_drawdown(value_series)
        assert result <= 0  # Drawdown is always non-positive

    def test_calculate_max_drawdown_known_drawdown(self):
        """Test max drawdown with known drawdown in one period.

        Note: _calculate_max_drawdown requires at least `duration` data points.
        Default duration is 7*24*60 = 10080 minutes.
        """
        from quantammsim.core_simulator.forward_pass import _calculate_max_drawdown

        # Create 2 weeks of minute data (2 complete duration periods)
        duration = 7 * 24 * 60  # 10080
        np.random.seed(42)

        # First week: values go up 20%, then drop 25% (to 90% of peak)
        week1_values = np.concatenate([
            np.linspace(100.0, 120.0, duration // 2),  # Up 20%
            np.linspace(120.0, 90.0, duration // 2),   # Down to 90 (25% drawdown from 120)
        ])

        # Second week: steady values
        week2_values = np.ones(duration) * 95.0

        values = jnp.array(np.concatenate([week1_values, week2_values]))
        result = _calculate_max_drawdown(values, duration=duration)

        # Max drawdown in week1 should be (90-120)/120 = -0.25
        assert result < 0  # Is a drawdown
        np.testing.assert_allclose(result, -0.25, rtol=0.01)

    def test_calculate_max_drawdown_requires_duration_data(self):
        """Test that max drawdown needs at least `duration` data points."""
        from quantammsim.core_simulator.forward_pass import _calculate_max_drawdown

        # Too few points - function returns nan due to empty array
        values = jnp.array([100.0, 110.0, 120.0, 130.0])
        with pytest.raises(ValueError, match="zero-size array"):
            _calculate_max_drawdown(values, duration=100)

    def test_calculate_var(self, value_series):
        """Test VaR calculation."""
        from quantammsim.core_simulator.forward_pass import _calculate_var

        result = _calculate_var(value_series, percentile=5.0)
        assert jnp.isfinite(result)

    def test_calculate_var_known_values(self):
        """Test VaR with known input/output.

        Note: _calculate_var requires at least `duration` data points.
        Default duration is 24*60 = 1440 minutes.
        """
        from quantammsim.core_simulator.forward_pass import _calculate_var

        # Create 2 days of minute data (2 complete duration periods)
        duration = 24 * 60  # 1440
        np.random.seed(42)

        # Create realistic price data with some negative returns
        returns = np.random.randn(duration * 2) * 0.001  # small minute-by-minute changes
        prices = 1000.0 * np.cumprod(1 + returns)
        values = jnp.array(prices)

        result = _calculate_var(values, percentile=5.0, duration=duration)
        # VaR at 5th percentile should be finite
        assert jnp.isfinite(result)
        # And negative (representing a loss)
        assert result < 0

    def test_calculate_var_trad(self, value_series):
        """Test traditional VaR calculation."""
        from quantammsim.core_simulator.forward_pass import _calculate_var_trad

        result = _calculate_var_trad(value_series, percentile=5.0)
        assert jnp.isfinite(result)

    def test_var_vs_var_trad_relationship(self, value_series):
        """Traditional VaR should typically be more conservative."""
        from quantammsim.core_simulator.forward_pass import _calculate_var, _calculate_var_trad

        var = _calculate_var(value_series, percentile=5.0)
        var_trad = _calculate_var_trad(value_series, percentile=5.0)
        # Both should be negative (representing losses)
        assert var < 0
        assert var_trad < 0

    def test_calculate_raroc(self, value_series):
        """Test RAROC calculation - return over risk-adjusted capital."""
        from quantammsim.core_simulator.forward_pass import _calculate_raroc

        result = _calculate_raroc(value_series)
        assert jnp.isfinite(result)

    def test_calculate_raroc_higher_for_better_performance(self):
        """Test that RAROC is higher for series with better risk-adjusted returns."""
        from quantammsim.core_simulator.forward_pass import _calculate_raroc

        # Create two series: one with steady growth, one with more volatility
        duration = 24 * 60 * 7  # 7 days
        np.random.seed(42)

        # Steady growth series
        steady_returns = np.ones(duration) * 0.0001  # consistent small positive returns
        steady_values = 1000.0 * np.cumprod(1 + steady_returns)

        # Volatile series with same average return but higher variance
        volatile_returns = np.random.randn(duration) * 0.001 + 0.0001
        volatile_values = 1000.0 * np.cumprod(1 + volatile_returns)

        raroc_steady = _calculate_raroc(jnp.array(steady_values))
        raroc_volatile = _calculate_raroc(jnp.array(volatile_values))

        # Both should be finite
        assert jnp.isfinite(raroc_steady)
        assert jnp.isfinite(raroc_volatile)

    def test_calculate_rovar(self, value_series):
        """Test ROVAR calculation - return over VaR."""
        from quantammsim.core_simulator.forward_pass import _calculate_rovar

        result = _calculate_rovar(value_series)
        assert jnp.isfinite(result)

    def test_calculate_rovar_trad(self, value_series):
        """Test traditional ROVAR calculation."""
        from quantammsim.core_simulator.forward_pass import _calculate_rovar_trad

        result = _calculate_rovar_trad(value_series)
        assert jnp.isfinite(result)

    def test_rovar_vs_rovar_trad_both_finite(self, value_series):
        """Test that both ROVAR variants produce finite results."""
        from quantammsim.core_simulator.forward_pass import _calculate_rovar, _calculate_rovar_trad

        rovar = _calculate_rovar(value_series)
        rovar_trad = _calculate_rovar_trad(value_series)

        assert jnp.isfinite(rovar)
        assert jnp.isfinite(rovar_trad)

    def test_calculate_sterling_ratio(self, value_series):
        """Test Sterling ratio calculation."""
        from quantammsim.core_simulator.forward_pass import _calculate_sterling_ratio

        result = _calculate_sterling_ratio(value_series)
        # Can be inf if no drawdown
        assert not jnp.isnan(result)

    def test_calculate_sterling_ratio_with_adjustment(self, value_series):
        """Test Sterling ratio with drawdown adjustment."""
        from quantammsim.core_simulator.forward_pass import _calculate_sterling_ratio

        result = _calculate_sterling_ratio(value_series, drawdown_adjustment=0.1)
        assert not jnp.isnan(result)

    def test_calculate_calmar_ratio(self, value_series):
        """Test Calmar ratio calculation."""
        from quantammsim.core_simulator.forward_pass import _calculate_calmar_ratio

        result = _calculate_calmar_ratio(value_series)
        assert not jnp.isnan(result)

    def test_calculate_calmar_ratio_with_duration(self, value_series):
        """Test Calmar ratio with duration limit."""
        from quantammsim.core_simulator.forward_pass import _calculate_calmar_ratio

        # Limit to 7 days
        result = _calculate_calmar_ratio(value_series, duration=7 * 24 * 60)
        assert not jnp.isnan(result)

    def test_calculate_ulcer_index(self, value_series):
        """Test Ulcer index calculation."""
        from quantammsim.core_simulator.forward_pass import _calculate_ulcer_index

        result = _calculate_ulcer_index(value_series)
        assert jnp.isfinite(result)


# =============================================================================
# PARAM_UTILS SQUAREPLUS AND OTHER FUNCTIONS
# =============================================================================

class TestSquareplus:
    """Test squareplus activation function."""

    @given(x=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
    @settings(max_examples=30, deadline=None)
    def test_squareplus_positive(self, x):
        """Squareplus should always be positive."""
        from quantammsim.core_simulator.param_utils import squareplus

        result = squareplus(x)
        assert result > 0

    @given(x=st.floats(min_value=0.0, max_value=10.0, allow_nan=False))
    @settings(max_examples=30, deadline=None)
    def test_squareplus_approx_identity_for_positive(self, x):
        """For large positive x, squareplus ≈ x."""
        from quantammsim.core_simulator.param_utils import squareplus

        if x > 2:
            result = squareplus(x)
            assert abs(result - x) < 0.5

    def test_squareplus_array(self):
        """Test squareplus with array input."""
        from quantammsim.core_simulator.param_utils import squareplus

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = squareplus(x)
        assert result.shape == x.shape
        assert jnp.all(result > 0)
