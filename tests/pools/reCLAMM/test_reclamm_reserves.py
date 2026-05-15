"""Integration tests for reClAMM scan-based reserve calculations and pool class.

Tests the full pipeline: initialization → scan → reserves, plus pool creation
and registration via creator.py.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from tests.conftest import TEST_DATA_DIR
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    compute_invariant,
    compute_price_ratio,
    compute_centeredness,
    initialise_reclamm_reserves,
    calibrate_arc_length_speed,
    _jax_calc_reclamm_reserves_zero_fees,
    _jax_calc_reclamm_reserves_zero_fees_full_state,
    _jax_calc_reclamm_reserves_with_fees,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_fees,
)

# For n=2: sig variations with exactly one +1 and one -1
ALL_SIG_VARIATIONS_2 = jnp.array([[1, -1], [-1, 1]])

# Default pool parameters
DEFAULT_CENTEREDNESS_MARGIN = 0.2
DEFAULT_DAILY_PRICE_SHIFT_BASE = 1.0 - 1.0 / 124000.0
DEFAULT_PRICE_RATIO = 4.0
DEFAULT_SECONDS_PER_STEP = 60.0  # 1-minute arb frequency


def _make_constant_prices(price_a, price_b, n_steps):
    """Create constant price array."""
    return jnp.tile(jnp.array([price_a, price_b]), (n_steps, 1))


def _make_trending_prices(start_a, end_a, price_b, n_steps):
    """Create linearly trending price array for token A."""
    prices_a = jnp.linspace(start_a, end_a, n_steps)
    prices_b = jnp.full(n_steps, price_b)
    return jnp.stack([prices_a, prices_b], axis=1)


def _init_pool(initial_pool_value=1_000_000.0, price_a=2500.0, price_b=1.0,
               price_ratio=DEFAULT_PRICE_RATIO):
    """Initialize pool reserves and virtual balances."""
    initial_prices = jnp.array([price_a, price_b])
    reserves, Va, Vb = initialise_reclamm_reserves(
        initial_pool_value, initial_prices, price_ratio
    )
    return reserves, Va, Vb


class TestConstantPricesNoArb:
    """When prices don't change, reserves should stay constant."""

    def test_zero_fees(self):
        reserves, Va, Vb = _init_pool()
        prices = _make_constant_prices(2500.0, 1.0, 10)

        result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
        )
        # All timesteps should have same reserves (no price change → no arb)
        for i in range(result.shape[0]):
            npt.assert_allclose(result[i], reserves, rtol=1e-6)

    def test_with_fees(self):
        reserves, Va, Vb = _init_pool()
        prices = _make_constant_prices(2500.0, 1.0, 10)

        result = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )
        for i in range(result.shape[0]):
            npt.assert_allclose(result[i], reserves, rtol=1e-6)


class TestSingleStepArb:
    """Single price step: verify reserves move toward equilibrium."""

    def test_zero_fees(self):
        reserves, Va, Vb = _init_pool()
        # Price jumps from 2500 to 3000 — arb should rebalance
        prices = jnp.array([[3000.0, 1.0]])

        result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
        )

        # Token A should decrease (arb buys cheap A from pool, sells on market)
        # Token B should increase
        assert float(result[0, 0]) < float(reserves[0])
        assert float(result[0, 1]) > float(reserves[1])

    def test_with_fees_less_movement(self):
        """With fees, arb should cause less reserve movement than zero-fee."""
        reserves, Va, Vb = _init_pool()
        prices = jnp.array([[3000.0, 1.0]])

        zero_fee_result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
        )

        fee_result = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        # Fee case: less total trade magnitude
        zero_fee_delta = jnp.abs(zero_fee_result[0] - reserves).sum()
        fee_delta = jnp.abs(fee_result[0] - reserves).sum()
        assert float(fee_delta) <= float(zero_fee_delta) + 1e-10


class TestReservesPositiveThroughout:
    """Reserves should never go negative during multi-step scan."""

    def test_trending_up(self):
        reserves, Va, Vb = _init_pool()
        prices = _make_trending_prices(2500.0, 4000.0, 1.0, 50)

        result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
        )
        assert jnp.all(result >= 0), "Negative reserves found during uptrend"

    def test_trending_down(self):
        reserves, Va, Vb = _init_pool()
        prices = _make_trending_prices(2500.0, 1200.0, 1.0, 50)

        result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
        )
        assert jnp.all(result >= 0), "Negative reserves found during downtrend"

    def test_volatile_prices(self):
        reserves, Va, Vb = _init_pool()
        # Random walk around 2500
        np.random.seed(42)
        n_steps = 100
        log_returns = np.random.normal(0, 0.02, n_steps)
        price_a = 2500.0 * np.exp(np.cumsum(log_returns))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
        )
        assert jnp.all(result >= 0), "Negative reserves found during volatile prices"


class TestFeePoolRetainsMoreValue:
    """Fee pool should retain more value than zero-fee pool.

    With zero fees, arbitrageurs extract more value from the pool (LVR).
    Fees protect the pool by reducing the arb's profit margin.
    """

    def test_value_comparison(self):
        reserves, Va, Vb = _init_pool()
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, 20)

        zero_fee_result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
        )

        fee_result = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        # Compare final values
        final_prices = prices[-1]
        zero_fee_value = (zero_fee_result[-1] * final_prices).sum()
        fee_value = (fee_result[-1] * final_prices).sum()

        # Fee pool retains more value — fees reduce arb extraction (LVR)
        assert float(fee_value) >= float(zero_fee_value) - 1e-6


class TestPoolCreation:
    """Test pool creation and registration."""

    def test_create_pool(self):
        from quantammsim.pools.creator import create_pool
        pool = create_pool("reclamm")
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool
        assert isinstance(pool, ReClammPool)

    def test_pool_is_trainable(self):
        from quantammsim.pools.creator import create_pool
        pool = create_pool("reclamm")
        assert pool.is_trainable() is True

    def test_pool_weights_needs_original_methods(self):
        from quantammsim.pools.creator import create_pool
        pool = create_pool("reclamm")
        assert pool.weights_needs_original_methods() is True


class TestPoolIntegration:
    """Test full pipeline through the pool class."""

    def test_calculate_reserves_with_fees(self):
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        # Scalar params — vmap peels the n_parameter_sets dim in real usage
        params = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
        }

        # 12 price steps + 1 for bout_length
        n_steps = 12
        np.random.seed(42)
        price_a = 2500.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_steps)))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        run_fingerprint = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": 1_000_000.0,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "ste_temperature": 10.0,
        })

        start_index = jnp.array([0, 0])

        reserves = pool.calculate_reserves_with_fees(
            params, run_fingerprint, prices, start_index
        )

        # Shape should be (n_steps, 2)
        assert reserves.shape == (n_steps, 2), f"Expected ({n_steps}, 2), got {reserves.shape}"
        # All positive
        assert jnp.all(reserves > 0), "Negative reserves in integration test"

    def test_calculate_reserves_zero_fees(self):
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
        }

        n_steps = 12
        np.random.seed(42)
        price_a = 2500.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_steps)))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        run_fingerprint = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": 1_000_000.0,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "ste_temperature": 10.0,
        })

        start_index = jnp.array([0, 0])

        reserves = pool.calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index
        )

        assert reserves.shape == (n_steps, 2)
        assert jnp.all(reserves > 0)

    def test_calculate_weights(self):
        """Empirical weights should sum to 1 and be positive."""
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
        }

        n_steps = 10
        prices = _make_constant_prices(2500.0, 1.0, n_steps)

        run_fingerprint = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": 1_000_000.0,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "ste_temperature": 10.0,
        })

        start_index = jnp.array([0, 0])
        weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index
        )

        assert weights.shape == (n_steps, 2)
        # Weights sum to 1
        npt.assert_allclose(jnp.sum(weights, axis=-1), jnp.ones(n_steps), rtol=1e-6)
        # All positive
        assert jnp.all(weights > 0)


class TestConstantArcLengthScan:
    """Integration tests for constant-arc-length thermostat through the scan."""

    def _calibrate_speed(self, reserves, Va, Vb, seconds_per_step=60.0):
        """Helper to calibrate arc-length speed at the onset state."""
        sqrt_Q = jnp.sqrt(compute_price_ratio(
            float(reserves[0]), float(reserves[1]), float(Va), float(Vb),
        ))
        market_price = (float(reserves[1]) + float(Vb)) / (float(reserves[0]) + float(Va))
        return calibrate_arc_length_speed(
            reserves[0], reserves[1], Va, Vb,
            DEFAULT_DAILY_PRICE_SHIFT_BASE, seconds_per_step, sqrt_Q, market_price,
            centeredness_margin=DEFAULT_CENTEREDNESS_MARGIN,
        )

    def test_scan_runs(self):
        """Constant-arc-length scan completes and differs from geometric."""
        reserves, Va, Vb = _init_pool()
        # Large price swing to push centeredness below margin
        n_steps = 100
        prices = _make_trending_prices(2500.0, 5000.0, 1.0, n_steps)
        speed = self._calibrate_speed(reserves, Va, Vb)

        # Speed should be non-trivial
        assert float(speed) > 1e-6, f"Speed should be non-trivial, got {float(speed):.2e}"

        result_cal = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=speed,
        )
        assert result_cal.shape == (n_steps, 2)

        # Verify it produces different reserves than geometric
        result_geo = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=0.0,
        )
        rel_diff = jnp.abs(result_cal[-1] - result_geo[-1]) / jnp.maximum(result_geo[-1], 1e-10)
        assert float(rel_diff.max()) > 1e-4, (
            f"Constant-arc-length should differ from geometric, got max rel diff = {float(rel_diff.max()):.2e}"
        )

    def test_reserves_positive(self):
        """All reserves should be >= 0 throughout the constant-arc-length scan."""
        reserves, Va, Vb = _init_pool()
        # Large swing to ensure thermostat fires
        prices = _make_trending_prices(2500.0, 6000.0, 1.0, 150)
        speed = self._calibrate_speed(reserves, Va, Vb)

        assert float(speed) > 1e-6, f"Speed should be non-trivial, got {float(speed):.2e}"

        result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=speed,
        )
        assert jnp.all(result >= 0), "Negative reserves in constant-arc-length scan"

    def test_geometric_default(self):
        """arc_length_speed=0 should reproduce existing geometric behavior exactly."""
        reserves, Va, Vb = _init_pool()
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, 30)

        result_default = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
        )
        result_explicit = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=0.0,
        )
        npt.assert_allclose(result_default, result_explicit, rtol=1e-12)

    def test_fingerprint_dispatch(self):
        """Pool class should accept "constant_arc_length" via fingerprint."""
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
        }

        n_steps = 12
        np.random.seed(42)
        price_a = 2500.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_steps)))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        run_fingerprint = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": 1_000_000.0,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "reclamm_interpolation_method": "constant_arc_length",
            "reclamm_arc_length_speed": None,  # auto-calibrate
            "ste_temperature": 10.0,
        })

        start_index = jnp.array([0, 0])
        reserves = pool.calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index
        )

        assert reserves.shape == (n_steps, 2)
        assert jnp.all(reserves > 0)


class TestCenterednessScaledScan:
    """Integration tests for centeredness-proportional speed scaling."""

    def _calibrate_speed(self, reserves, Va, Vb, seconds_per_step=60.0):
        """Helper to calibrate arc-length speed at the onset state."""
        sqrt_Q = jnp.sqrt(compute_price_ratio(
            float(reserves[0]), float(reserves[1]), float(Va), float(Vb),
        ))
        market_price = (float(reserves[1]) + float(Vb)) / (float(reserves[0]) + float(Va))
        return calibrate_arc_length_speed(
            reserves[0], reserves[1], Va, Vb,
            DEFAULT_DAILY_PRICE_SHIFT_BASE, seconds_per_step, sqrt_Q, market_price,
            centeredness_margin=DEFAULT_CENTEREDNESS_MARGIN,
        )

    def test_scan_runs_with_scaling(self):
        """Centeredness-scaled scan completes without errors on trending prices."""
        reserves, Va, Vb = _init_pool()
        n_steps = 100
        prices = _make_trending_prices(2500.0, 5000.0, 1.0, n_steps)
        speed = self._calibrate_speed(reserves, Va, Vb)

        result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=speed,
            centeredness_scaling=True,
        )
        assert result.shape == (n_steps, 2)

    def test_reserves_positive(self):
        """All reserves should be >= 0 with centeredness scaling enabled."""
        reserves, Va, Vb = _init_pool()
        prices = _make_trending_prices(2500.0, 6000.0, 1.0, 150)
        speed = self._calibrate_speed(reserves, Va, Vb)

        result = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=speed,
            centeredness_scaling=True,
        )
        assert jnp.all(result >= 0), "Negative reserves with centeredness scaling"

    def test_differs_from_constant_speed(self):
        """On trending prices, centeredness-scaled should differ from constant speed."""
        reserves, Va, Vb = _init_pool()
        n_steps = 100
        prices = _make_trending_prices(2500.0, 5000.0, 1.0, n_steps)
        speed = self._calibrate_speed(reserves, Va, Vb)

        result_const = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=speed,
            centeredness_scaling=False,
        )
        result_scaled = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=speed,
            centeredness_scaling=True,
        )

        rel_diff = jnp.abs(result_const[-1] - result_scaled[-1]) / jnp.maximum(result_const[-1], 1e-10)
        assert float(rel_diff.max()) > 1e-4, (
            f"Centeredness scaling should differ from constant speed, got max rel diff = {float(rel_diff.max()):.2e}"
        )

    def test_backward_compat_flag_off(self):
        """flag=False reproduces existing constant-arc-length behavior exactly."""
        reserves, Va, Vb = _init_pool()
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, 30)
        speed = self._calibrate_speed(reserves, Va, Vb)

        result_default = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=speed,
        )
        result_explicit = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            arc_length_speed=speed,
            centeredness_scaling=False,
        )
        npt.assert_allclose(result_default, result_explicit, rtol=1e-12)


class TestReClammTrainable:
    """Tests for reClAMM trainability via train_on_historic_data."""

    def test_is_trainable(self):
        """ReClammPool.is_trainable() should return True."""
        from quantammsim.pools.creator import create_pool

        pool = create_pool("reclamm")
        assert pool.is_trainable() is True

    def test_init_base_parameters_shapes(self):
        """All params from init_base_parameters should be (n_parameter_sets, 1)."""
        from quantammsim.pools.creator import create_pool

        pool = create_pool("reclamm")
        n_parameter_sets = 4
        initial_values = {
            "price_ratio": 4.0,
            "centeredness_margin": 0.2,
            "daily_price_shift_base": 1.0 - 1.0 / 124000.0,
        }
        params = pool.init_base_parameters(
            initial_values, {}, n_assets=2, n_parameter_sets=n_parameter_sets
        )
        for key in ("price_ratio", "centeredness_margin", "daily_price_shift_base"):
            assert params[key].shape == (n_parameter_sets, 1), (
                f"{key} shape should be ({n_parameter_sets}, 1), got {params[key].shape}"
            )

    def test_init_base_parameters_includes_arc_length_speed(self):
        """When reclamm_learn_arc_length_speed=True and interpolation is
        constant_arc_length, init_base_parameters should include arc_length_speed."""
        from quantammsim.pools.creator import create_pool

        pool = create_pool("reclamm")
        n_parameter_sets = 4
        initial_values = {
            "price_ratio": 4.0,
            "centeredness_margin": 0.2,
            "daily_price_shift_base": 1.0 - 1.0 / 124000.0,
            "arc_length_speed": 1e-4,
        }
        fp = {
            "reclamm_learn_arc_length_speed": True,
            "reclamm_interpolation_method": "constant_arc_length",
        }
        params = pool.init_base_parameters(
            initial_values, fp, n_assets=2, n_parameter_sets=n_parameter_sets
        )
        assert "arc_length_speed" in params, (
            "arc_length_speed should be in params when learn flag is True"
        )
        assert params["arc_length_speed"].shape == (n_parameter_sets, 1)

    def test_init_base_parameters_excludes_arc_length_speed_by_default(self):
        """Without the learn flag, arc_length_speed should NOT be in params."""
        from quantammsim.pools.creator import create_pool

        pool = create_pool("reclamm")
        initial_values = {
            "price_ratio": 4.0,
            "centeredness_margin": 0.2,
            "daily_price_shift_base": 1.0 - 1.0 / 124000.0,
        }
        params = pool.init_base_parameters(
            initial_values, {}, n_assets=2, n_parameter_sets=1
        )
        assert "arc_length_speed" not in params

    def test_learnable_arc_length_speed_forward_pass(self):
        """Forward pass should use arc_length_speed from params when present."""
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        n_steps = 50
        prices = _make_trending_prices(2500.0, 4000.0, 1.0, n_steps)

        run_fingerprint = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": 1_000_000.0,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "reclamm_interpolation_method": "constant_arc_length",
            "reclamm_learn_arc_length_speed": True,
            "ste_temperature": 10.0,
        })

        start_index = jnp.array([0, 0])

        # Two different arc_length_speed values should produce different reserves
        params_slow = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
            "arc_length_speed": jnp.float64(1e-6),
        }
        params_fast = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
            "arc_length_speed": jnp.float64(1e-3),
        }

        reserves_slow = pool.calculate_reserves_with_fees(
            params_slow, run_fingerprint, prices, start_index
        )
        reserves_fast = pool.calculate_reserves_with_fees(
            params_fast, run_fingerprint, prices, start_index
        )

        # Different speeds should produce different final reserves
        rel_diff = jnp.abs(reserves_slow[-1] - reserves_fast[-1]) / jnp.maximum(
            reserves_slow[-1], 1e-10
        )
        assert float(rel_diff.max()) > 1e-4, (
            f"Different arc_length_speed values should produce different reserves, "
            f"got max rel diff = {float(rel_diff.max()):.2e}"
        )

    def test_shift_exponent_equivalent_to_base(self):
        """shift_exponent param produces identical reserves to daily_price_shift_base."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool, SHIFT_EXPONENT_DIVISOR
        from quantammsim.runners.jax_runners import do_run_on_historic_data

        shift_exp = 1.0
        base = 1.0 - shift_exp / SHIFT_EXPONENT_DIVISOR

        fp_common = {
            "rule": "reclamm",
            "tokens": ["ETH", "USDC"],
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-15 00:00:00",
            "initial_pool_value": 1_000_000.0,
            "do_arb": True,
            "fees": 0.0,
        }

        result_base = do_run_on_historic_data(
            run_fingerprint=fp_common,
            params={
                "price_ratio": jnp.array(4.0),
                "centeredness_margin": jnp.array(0.2),
                "daily_price_shift_base": jnp.array(base),
            },
            root=TEST_DATA_DIR,
        )
        result_exp = do_run_on_historic_data(
            run_fingerprint={**fp_common, "reclamm_use_shift_exponent": True},
            params={
                "price_ratio": jnp.array(4.0),
                "centeredness_margin": jnp.array(0.2),
                "shift_exponent": jnp.array(shift_exp),
            },
            root=TEST_DATA_DIR,
        )

        np.testing.assert_allclose(
            float(result_base["final_value"]),
            float(result_exp["final_value"]),
            rtol=1e-10,
            err_msg="shift_exponent and daily_price_shift_base should produce identical results",
        )

    def test_train_on_historic_data_optuna(self):
        """End-to-end: Optuna finds params via train_on_historic_data."""
        from quantammsim.runners.jax_runners import train_on_historic_data

        fp = {
            "rule": "reclamm",
            "tokens": ["ETH", "USDC"],
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-15 00:00:00",
            "endTestDateString": "2023-02-01 00:00:00",
            "endTestDateString": "2023-03-01 00:00:00",
            "initial_pool_value": 1_000_000.0,
            "do_arb": True,
            "fees": 0.0025,
            "initial_price_ratio": 4.0,
            "initial_centeredness_margin": 0.2,
            "initial_daily_price_shift_base": 1.0 - 1.0 / 124000.0,
            "optimisation_settings": {
                "method": "optuna",
                "n_trials": 3,
                "n_parameter_sets": 1,
                "optuna_settings": {
                    "make_scalar": True,
                    "expand_around": False,
                    "parameter_config": {
                        "price_ratio": {
                            "low": 1.5,
                            "high": 10.0,
                            "log_scale": True,
                            "scalar": True,
                        },
                        "centeredness_margin": {
                            "low": 0.1,
                            "high": 0.9,
                            "scalar": True,
                        },
                        "daily_price_shift_base": {
                            "low": 0.99990,
                            "high": 0.99999,
                            "scalar": True,
                        },
                    },
                },
            },
        }
        result = train_on_historic_data(fp, verbose=False, root=TEST_DATA_DIR)
        assert result is not None


class TestNoiseTraderRatio:
    """Noise trader fee income wiring for reClAMM pools."""

    def _run_with_noise(self, noise_trader_ratio, n_steps=50, fees=0.003):
        """Run reClAMM with fees and return reserves + fee revenue."""
        reserves, Va, Vb = _init_pool()
        # Trending prices so arb trades happen and noise trade has non-zero effect
        prices = _make_trending_prices(2500.0, 3000.0, 1.0, n_steps)

        return _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=fees,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_trader_ratio=noise_trader_ratio,
        )

    def test_noise_trader_ratio_zero_is_default(self):
        """noise_trader_ratio=0.0 should produce identical results to omitting it."""
        reserves, Va, Vb = _init_pool()
        prices = _make_trending_prices(2500.0, 3000.0, 1.0, 50)

        # Explicit zero
        res_zero, rev_zero = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_trader_ratio=0.0,
        )

        # Default (omitted)
        res_default, rev_default = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        npt.assert_array_equal(res_zero, res_default)
        npt.assert_array_equal(rev_zero, rev_default)

    def test_noise_trader_ratio_increases_reserves(self):
        """Noise traders add fee income, so pool value should be higher."""
        res_no_noise, _ = self._run_with_noise(0.0)
        res_noise, _ = self._run_with_noise(0.1)

        # Final pool value in USD (sum of reserves * price at last step)
        final_prices = jnp.array([3000.0, 1.0])
        value_no_noise = (res_no_noise[-1] * final_prices).sum()
        value_noise = (res_noise[-1] * final_prices).sum()

        assert value_noise > value_no_noise, (
            f"Noise traders should increase pool value: {value_noise} <= {value_no_noise}"
        )

        # Reserves should differ
        assert not jnp.allclose(res_no_noise, res_noise), (
            "Reserves should differ with noise traders"
        )

    def test_noise_trader_ratio_through_pool_class(self):
        """noise_trader_ratio flows through the pool class methods."""
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
        }

        n_steps = 50
        np.random.seed(42)
        price_a = 2500.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_steps)))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)
        start_index = jnp.array([0, 0])

        base_fp = {
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": 1_000_000.0,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "ste_temperature": 10.0,
        }

        fp_no_noise = Hashabledict({**base_fp, "noise_trader_ratio": 0.0})
        fp_noise = Hashabledict({**base_fp, "noise_trader_ratio": 0.1})

        res_no = pool.calculate_reserves_with_fees(
            params, fp_no_noise, prices, start_index
        )
        res_yes = pool.calculate_reserves_with_fees(
            params, fp_noise, prices, start_index
        )

        # Should run and produce different results
        assert res_no.shape == (n_steps, 2)
        assert res_yes.shape == (n_steps, 2)
        assert not jnp.allclose(res_no, res_yes), (
            "Pool class should produce different reserves with noise traders"
        )

    def test_noise_trade_does_not_affect_virtual_balances(self):
        """Noise trade fee income only grows real reserves, not Va/Vb."""
        from quantammsim.pools.reCLAMM.reclamm_reserves import (
            _reclamm_scan_step_with_fees_and_revenue,
            precalc_shared_values_for_all_signatures,
            precalc_components_of_optimal_trade_across_prices,
        )

        reserves, Va, Vb = _init_pool()
        # Price must differ from init (2500) to trigger an arb trade
        prices_arr = jnp.array([[3000.0, 1.0]])
        prices = prices_arr[0]

        weights = jnp.array([0.5, 0.5])
        gamma = 1.0 - 0.003
        n_assets = 2

        _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
            precalc_shared_values_for_all_signatures(ALL_SIG_VARIATIONS_2, n_assets)
        )

        active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
            precalc_components_of_optimal_trade_across_prices(
                weights, prices_arr, gamma, tokens_to_drop,
                active_trade_directions, leave_one_out_idxs,
            )
        )

        carry = [
            reserves, Va, Vb,
            jnp.float64(1.0),   # prev_lp_supply
            jnp.float64(0.0),   # step_idx
            jnp.float64(0.0),   # active_start_ratio
            jnp.float64(0.0),   # active_target_ratio
            jnp.float64(0.0),   # active_start_step
            jnp.float64(0.0),   # active_end_step
            jnp.array(False),   # active_enabled
        ]
        inputs = [
            prices,
            active_initial_weights[0],
            per_asset_ratios[0],
            all_other_assets_ratios[0],
            gamma,
            0.0,  # arb_thresh
            0.0,  # arb_fees
            jnp.array([0.0, 0.0, 0.0]),  # price_ratio_update (no-op)
            jnp.float64(1.0),  # lp_supply
        ]

        # Without noise
        carry_no, _ = _reclamm_scan_step_with_fees_and_revenue(
            carry, inputs,
            weights=weights,
            tokens_to_drop=tokens_to_drop,
            active_trade_directions=active_trade_directions,
            n=n_assets,
            centeredness_margin=DEFAULT_CENTEREDNESS_MARGIN,
            daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
            seconds_per_step=DEFAULT_SECONDS_PER_STEP,
            noise_trader_ratio=0.0,
        )

        # With noise
        carry_yes, _ = _reclamm_scan_step_with_fees_and_revenue(
            carry, inputs,
            weights=weights,
            tokens_to_drop=tokens_to_drop,
            active_trade_directions=active_trade_directions,
            n=n_assets,
            centeredness_margin=DEFAULT_CENTEREDNESS_MARGIN,
            daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
            seconds_per_step=DEFAULT_SECONDS_PER_STEP,
            noise_trader_ratio=0.5,
        )

        # Virtual balances should be identical
        npt.assert_array_equal(carry_no[1], carry_yes[1], err_msg="Va changed")
        npt.assert_array_equal(carry_no[2], carry_yes[2], err_msg="Vb changed")

        # But real reserves should differ (noise adds fee income)
        assert not jnp.allclose(carry_no[0], carry_yes[0]), (
            "Real reserves should differ with noise traders"
        )


class TestLpSupply:
    """LP supply (BPT) scaling for reClAMM pools.

    Ported from Foundry fuzz test invariants in ReClammLiquidity.t.sol:
    both real AND virtual reserves scale proportionally with BPT supply,
    preserving price and centeredness.
    """

    def test_lp_supply_none_matches_default(self):
        """lp_supply_array=None vs omitting param → identical results."""
        reserves, Va, Vb = _init_pool()
        prices = _make_trending_prices(2500.0, 3000.0, 1.0, 20)

        result_none = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            lp_supply_array=None,
        )
        result_omit = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )
        npt.assert_array_equal(result_none, result_omit)

    def test_lp_supply_constant_one_matches_default(self):
        """lp_supply_array=jnp.ones(T) vs None → identical results."""
        reserves, Va, Vb = _init_pool()
        n_steps = 20
        prices = _make_trending_prices(2500.0, 3000.0, 1.0, n_steps)

        result_ones = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            lp_supply_array=jnp.ones(n_steps),
        )
        result_none = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )
        npt.assert_array_equal(result_ones, result_none)

    def test_lp_supply_doubling_scales_reserves(self):
        """BPT supply doubling halfway → reserves ~2x at that step.

        Ported from Foundry testAddLiquidity__Fuzz: reserves scale with BPT.
        """
        reserves, Va, Vb = _init_pool()
        n_steps = 40
        half = n_steps // 2
        prices = _make_constant_prices(2500.0, 1.0, n_steps)

        # No LP supply change (baseline)
        result_base = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        # LP supply doubles at step `half`
        lp_supply = jnp.concatenate([jnp.ones(half), 2.0 * jnp.ones(n_steps - half)])
        result_lp = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            lp_supply_array=lp_supply,
        )

        # Before doubling: identical
        npt.assert_allclose(result_lp[:half], result_base[:half], rtol=1e-10)
        # After doubling: reserves ~2x the baseline
        npt.assert_allclose(result_lp[half], result_base[half] * 2.0, rtol=1e-6)

    def test_lp_supply_halving_scales_reserves(self):
        """BPT supply halving halfway → reserves ~0.5x at that step."""
        reserves, Va, Vb = _init_pool()
        n_steps = 40
        half = n_steps // 2
        prices = _make_constant_prices(2500.0, 1.0, n_steps)

        result_base = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        lp_supply = jnp.concatenate([jnp.ones(half), 0.5 * jnp.ones(n_steps - half)])
        result_lp = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            lp_supply_array=lp_supply,
        )

        # Before halving: identical
        npt.assert_allclose(result_lp[:half], result_base[:half], rtol=1e-10)
        # After halving: reserves ~0.5x the baseline
        npt.assert_allclose(result_lp[half], result_base[half] * 0.5, rtol=1e-6)

    def test_lp_supply_preserves_price(self):
        """Price ratio (Ra+Va)/(Rb+Vb) unchanged through LP supply scaling.

        Ported from Foundry: assertEq(price_before, price_after) in
        testAddLiquidity__Fuzz.
        """
        reserves, Va, Vb = _init_pool()
        n_steps = 40
        half = n_steps // 2
        # Trending so virtual balances shift — makes price preservation non-trivial
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        lp_supply = jnp.concatenate([jnp.ones(half), 2.0 * jnp.ones(n_steps - half)])

        # Use full_state variant to get Va/Vb history
        result_reserves, Va_hist, Vb_hist = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            lp_supply_array=lp_supply,
        )

        # Price = (Rb + Vb) / (Ra + Va) — should be identical at step half-1 and half
        # (the supply change happens at start of step `half`, before arb)
        Ra_before = result_reserves[half - 1, 0]
        Rb_before = result_reserves[half - 1, 1]
        Va_before = Va_hist[half - 1]
        Vb_before = Vb_hist[half - 1]
        price_before = (Rb_before + Vb_before) / (Ra_before + Va_before)

        # At step `half`, the carry from step half-1 gets scaled, then arb runs.
        # We need to check price right after scaling, before arb.
        # The closest check: price at step `half` output should reflect
        # scaled reserves with arb on top. Instead, verify that a constant-price
        # run preserves exact ratio.
        prices_const = _make_constant_prices(2500.0, 1.0, n_steps)
        res_c, Va_c, Vb_c = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices_const,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            lp_supply_array=lp_supply,
        )
        # With constant prices and no arb, price is just initial ratio throughout
        price_at_half_minus_1 = (res_c[half - 1, 1] + Vb_c[half - 1]) / (
            res_c[half - 1, 0] + Va_c[half - 1]
        )
        price_at_half = (res_c[half, 1] + Vb_c[half]) / (
            res_c[half, 0] + Va_c[half]
        )
        npt.assert_allclose(
            float(price_at_half_minus_1), float(price_at_half), rtol=1e-10,
            err_msg="Price ratio should be preserved through LP supply change",
        )

    def test_lp_supply_preserves_centeredness(self):
        """Centeredness unchanged through LP supply scaling.

        Ported from Foundry: centeredness invariance in testAddLiquidity__Fuzz.
        """
        reserves, Va, Vb = _init_pool()
        n_steps = 40
        half = n_steps // 2
        prices = _make_constant_prices(2500.0, 1.0, n_steps)

        lp_supply = jnp.concatenate([jnp.ones(half), 2.0 * jnp.ones(n_steps - half)])

        res, Va_hist, Vb_hist = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            lp_supply_array=lp_supply,
        )

        c_before, _ = compute_centeredness(
            res[half - 1, 0], res[half - 1, 1], Va_hist[half - 1], Vb_hist[half - 1]
        )
        c_after, _ = compute_centeredness(
            res[half, 0], res[half, 1], Va_hist[half], Vb_hist[half]
        )
        npt.assert_allclose(
            float(c_before), float(c_after), rtol=1e-10,
            err_msg="Centeredness should be preserved through LP supply change",
        )

    def test_lp_supply_through_pool_class(self):
        """Pool class passes lp_supply_array to underlying computation."""
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
        }

        n_steps = 20
        prices = _make_trending_prices(2500.0, 3000.0, 1.0, n_steps)

        run_fingerprint = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": 1_000_000.0,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "ste_temperature": 10.0,
        })

        start_index = jnp.array([0, 0])

        from quantammsim.core_simulator.dynamic_inputs import DynamicInputArrays

        lp_supply = jnp.concatenate([jnp.ones(10), 2.0 * jnp.ones(10)])

        di_with_lp = DynamicInputArrays(
            trades=None,
            fees=jnp.full(n_steps, 0.003),
            gas_cost=jnp.zeros(n_steps),
            arb_fees=jnp.zeros(n_steps),
            lp_supply=lp_supply,
            reclamm_price_ratio_updates=jnp.array([[0.0, 0.0, 0.0, jnp.nan]]),
        )
        di_without_lp = DynamicInputArrays(
            trades=None,
            fees=jnp.full(n_steps, 0.003),
            gas_cost=jnp.zeros(n_steps),
            arb_fees=jnp.zeros(n_steps),
            lp_supply=jnp.ones(n_steps),
            reclamm_price_ratio_updates=jnp.array([[0.0, 0.0, 0.0, jnp.nan]]),
        )

        res_with_lp, _ = pool.calculate_reserves_and_fee_revenue_with_dynamic_inputs(
            params, run_fingerprint, prices, start_index,
            dynamic_inputs=di_with_lp,
        )
        res_without_lp, _ = pool.calculate_reserves_and_fee_revenue_with_dynamic_inputs(
            params, run_fingerprint, prices, start_index,
            dynamic_inputs=di_without_lp,
        )

        # First 10 steps identical, then diverge
        npt.assert_allclose(res_with_lp[:10], res_without_lp[:10], rtol=1e-10)
        assert not jnp.allclose(res_with_lp[10:], res_without_lp[10:]), (
            "LP supply change should produce different reserves"
        )

    def test_lp_supply_with_fee_revenue(self):
        """Doubling LP supply → fee revenue increases (bigger pool → bigger arb trades)."""
        reserves, Va, Vb = _init_pool()
        n_steps = 40
        half = n_steps // 2
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        # Baseline: no supply change
        _, rev_base = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        # Supply doubles halfway
        lp_supply = jnp.concatenate([jnp.ones(half), 2.0 * jnp.ones(n_steps - half)])
        _, rev_lp = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            lp_supply_array=lp_supply,
        )

        # After doubling, fee revenue per step should be larger
        # (pool is 2x bigger → arb trades are 2x bigger → fees are 2x)
        post_double_base = rev_base[half:].sum()
        post_double_lp = rev_lp[half:].sum()
        assert float(post_double_lp) > float(post_double_base), (
            f"Fee revenue should increase after doubling: {post_double_lp} <= {post_double_base}"
        )

    def test_lp_supply_e2e_do_run_on_historic_data(self):
        """End-to-end: lp_supply flows through do_run_on_historic_data via DynamicInputFrames."""
        import pandas as pd
        from quantammsim.runners.jax_runners import do_run_on_historic_data
        from quantammsim.core_simulator.dynamic_inputs import DynamicInputFrames

        fp = {
            "rule": "reclamm",
            "tokens": ["ETH", "USDC"],
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-01-15 00:00:00",
            "initial_pool_value": 1_000_000.0,
            "do_arb": True,
            "fees": 0.003,
        }
        params = {
            "price_ratio": jnp.array(4.0),
            "centeredness_margin": jnp.array(0.2),
            "daily_price_shift_base": jnp.array(DEFAULT_DAILY_PRICE_SHIFT_BASE),
        }

        # Baseline: no LP supply change
        result_base = do_run_on_historic_data(
            run_fingerprint={**fp},
            params={**params},
            root=TEST_DATA_DIR,
        )

        # LP supply doubles halfway through the period
        # unix column must be in milliseconds (matches windowing_utils convention)
        start_unix_ms = int(pd.Timestamp("2023-01-01").timestamp() * 1000)
        mid_unix_ms = int(pd.Timestamp("2023-01-08").timestamp() * 1000)
        lp_supply_df = pd.DataFrame({
            "unix": [start_unix_ms, mid_unix_ms],
            "lp_supply": [1.0, 2.0],
        })

        result_lp = do_run_on_historic_data(
            run_fingerprint={**fp},
            params={**params},
            dynamic_input_frames=DynamicInputFrames(lp_supply=lp_supply_df),
            root=TEST_DATA_DIR,
        )

        # Final values should differ — doubling LP supply changes pool dynamics
        base_val = float(result_base["final_value"])
        lp_val = float(result_lp["final_value"])
        assert base_val != lp_val, (
            f"LP supply change should affect final value: base={base_val}, lp={lp_val}"
        )
        # Doubled pool should have higher final value (more reserves)
        assert lp_val > base_val, (
            f"Doubled LP supply should increase final value: {lp_val} <= {base_val}"
        )
