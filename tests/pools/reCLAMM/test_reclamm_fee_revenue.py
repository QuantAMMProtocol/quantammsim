"""Tests for reClAMM fee revenue tracking.

Validates that fee revenue is correctly computed, returned, and propagated
through the pool class and forward pass.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from quantammsim.pools.reCLAMM.reclamm_reserves import (
    initialise_reclamm_reserves,
    _jax_calc_reclamm_reserves_with_fees,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_fees,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs,
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


class TestFeeRevenueShape:
    """_jax_calc_reclamm_reserves_and_fee_revenue_with_fees returns correct shapes."""

    def test_fee_revenue_shape_with_fees(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 20
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        result_reserves, fee_revenue = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )
        assert result_reserves.shape == (n_steps, 2), (
            f"Expected reserves shape ({n_steps}, 2), got {result_reserves.shape}"
        )
        assert fee_revenue.shape == (n_steps,), (
            f"Expected fee_revenue shape ({n_steps},), got {fee_revenue.shape}"
        )


class TestFeeRevenueZeroWhenNoTrade:
    """Constant prices means no arb, so fee_revenue should be all zeros."""

    def test_fee_revenue_zero_when_no_trade(self):
        reserves, Va, Vb = _init_pool()
        prices = _make_constant_prices(2500.0, 1.0, 10)

        _, fee_revenue = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )
        npt.assert_allclose(fee_revenue, jnp.zeros(10), atol=1e-10)


class TestFeeRevenuePositiveOnPriceJump:
    """Price jumps force arb trades, which should generate positive fee revenue."""

    def test_fee_revenue_positive_on_price_jump(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 20
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        _, fee_revenue = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )
        assert float(fee_revenue.sum()) > 0, (
            f"Expected positive total fee revenue on trending prices, got {float(fee_revenue.sum())}"
        )
        assert jnp.all(fee_revenue >= 0), "fee_revenue should never be negative"


class TestHigherFeesMoreRevenue:
    """Higher fee rate should collect more fee revenue on the same price path."""

    def test_higher_fees_more_revenue(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 30
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        _, fee_revenue_low = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        _, fee_revenue_high = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.01,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        assert float(fee_revenue_high.sum()) > float(fee_revenue_low.sum()), (
            f"1% fees ({float(fee_revenue_high.sum()):.2f}) should collect more "
            f"than 0.3% fees ({float(fee_revenue_low.sum()):.2f})"
        )


class TestProtocolSplitReducesLpRevenue:
    """protocol_fee_split=0.5 should give ~half the LP fee_revenue of split=0.0."""

    def test_protocol_split_reduces_lp_revenue(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 30
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        _, fee_revenue_no_split = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            protocol_fee_split=0.0,
        )

        _, fee_revenue_half_split = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            protocol_fee_split=0.5,
        )

        total_no_split = float(fee_revenue_no_split.sum())
        total_half_split = float(fee_revenue_half_split.sum())
        assert total_no_split > 0, "Need nonzero revenue for this test to be meaningful"
        # Half-split LP revenue should be roughly half (not exact due to path-dependence
        # — the protocol fee changes reserves which changes subsequent arbs)
        ratio = total_half_split / total_no_split
        assert 0.3 < ratio < 0.7, (
            f"Expected ~0.5 ratio, got {ratio:.3f} "
            f"(no_split={total_no_split:.2f}, half_split={total_half_split:.2f})"
        )


class TestReservesUnchangedByTracking:
    """Reserves from the fee-revenue function should be bitwise identical to the old function."""

    def test_reserves_unchanged_by_tracking(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 20
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        old_reserves = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        new_reserves, _ = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=0.003,
            arb_thresh=0.0,
            arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        npt.assert_array_equal(
            old_reserves, new_reserves,
            err_msg="Fee-revenue tracking should not alter reserve values"
        )


class TestDynamicInputsFeeRevenue:
    """_jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs returns correct shapes."""

    def test_dynamic_inputs_fee_revenue(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 20
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        fees = jnp.full(n_steps, 0.003)
        arb_thresh = jnp.full(n_steps, 0.0)
        arb_fees = jnp.full(n_steps, 0.0)

        result_reserves, fee_revenue = _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs(
            reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN,
            DEFAULT_DAILY_PRICE_SHIFT_BASE,
            DEFAULT_SECONDS_PER_STEP,
            fees=fees,
            arb_thresh=arb_thresh,
            arb_fees=arb_fees,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        assert result_reserves.shape == (n_steps, 2)
        assert fee_revenue.shape == (n_steps,)
        assert jnp.all(fee_revenue >= 0), "fee_revenue should never be negative"
        assert float(fee_revenue.sum()) > 0, "Expected positive total fee revenue"


class TestPoolMethodWithFees:
    """pool.calculate_reserves_and_fee_revenue_with_fees returns correct tuple."""

    def test_pool_method_with_fees(self):
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
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
        })

        start_index = jnp.array([0, 0])

        reserves, fee_revenue = pool.calculate_reserves_and_fee_revenue_with_fees(
            params, run_fingerprint, prices, start_index
        )

        assert reserves.shape == (n_steps, 2)
        assert fee_revenue.shape == (n_steps,)
        assert jnp.all(fee_revenue >= 0)


class TestPoolMethodWithDynamicInputs:
    """pool.calculate_reserves_and_fee_revenue_with_dynamic_inputs returns correct tuple."""

    def test_pool_method_with_dynamic_inputs(self):
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
            "fees": 0.003,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
        })

        start_index = jnp.array([0, 0])

        fees_array = jnp.array([0.003])
        arb_thresh_array = jnp.array([0.0])
        arb_fees_array = jnp.array([0.0])

        reserves, fee_revenue = pool.calculate_reserves_and_fee_revenue_with_dynamic_inputs(
            params, run_fingerprint, prices, start_index,
            fees_array=fees_array,
            arb_thresh_array=arb_thresh_array,
            arb_fees_array=arb_fees_array,
            trade_array=None,
        )

        assert reserves.shape == (n_steps, 2)
        assert fee_revenue.shape == (n_steps,)
        assert jnp.all(fee_revenue >= 0)


class TestForwardPassReturnsFeeRevenue:
    """forward_pass output dict has 'fee_revenue' key with correct shape."""

    def test_forward_pass_returns_fee_revenue(self):
        from quantammsim.pools.creator import create_pool
        from quantammsim.core_simulator.forward_pass import forward_pass
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": DEFAULT_PRICE_RATIO,
            "centeredness_margin": DEFAULT_CENTEREDNESS_MARGIN,
            "daily_price_shift_base": DEFAULT_DAILY_PRICE_SHIFT_BASE,
        }

        n_steps = 100
        np.random.seed(42)
        price_a = 2500.0 * np.exp(np.cumsum(np.random.normal(0, 0.005, n_steps)))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        static_dict = Hashabledict({
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
            "return_val": "reserves_and_values",
            "rule": "reclamm",
            "training_data_kind": "historic",
            "do_trades": False,
        })

        start_index = jnp.array([0, 0])

        result = forward_pass(
            params, start_index, prices, pool=pool, static_dict=static_dict,
        )

        assert "fee_revenue" in result, (
            f"Expected 'fee_revenue' in result dict, got keys: {list(result.keys())}"
        )
        assert result["fee_revenue"].shape == (n_steps,), (
            f"Expected fee_revenue shape ({n_steps},), got {result['fee_revenue'].shape}"
        )
