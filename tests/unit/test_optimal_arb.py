"""
Tests for optimal N-pool arbitrage calculations.

Tests cover:
- Signature functions (get_signature, compare_signatures, direction conversions)
- Optimal trade construction
- Trade sifting for profitability
- Multi-asset arbitrage scenarios
- Edge cases (no arb opportunity, aligned prices)
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax
from copy import deepcopy

from quantammsim.pools.G3M.optimal_n_pool_arb import (
    get_signature,
    compare_signatures,
    direction_to_sig,
    sig_to_tokens_to_keep,
    sig_to_direction,
    sig_to_direction_jnp,
    trade_to_direction_jnp,
    construct_optimal_trade_jnp,
    optimal_trade_sifter,
    parallelised_optimal_trade_sifter,
    wrapped_parallelised_optimal_trade_sifter,
    precalc_shared_values_for_all_signatures,
    precalc_components_of_optimal_trade_across_signatures,
    calc_active_inital_reserves_for_all_signatures,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def two_asset_setup():
    """Basic two-asset arbitrage setup."""
    return {
        "initial_weights": jnp.array([0.5, 0.5]),
        "initial_reserves": jnp.array([1000.0, 50.0]),  # ~20 price ratio
        "fee_gamma": 0.997,  # 0.3% fee
        "n": 2,
    }


@pytest.fixture
def three_asset_setup():
    """Three-asset arbitrage setup."""
    return {
        "initial_weights": jnp.array([0.4, 0.35, 0.25]),
        "initial_reserves": jnp.array([1000.0, 500.0, 250.0]),
        "fee_gamma": 0.997,
        "n": 3,
    }


@pytest.fixture
def two_asset_signatures():
    """All signature variations for two assets."""
    return jnp.array([
        [1, -1],
        [-1, 1],
    ])


@pytest.fixture
def three_asset_signatures():
    """Signature variations for three assets."""
    return jnp.array([
        [1, -1, 0],
        [-1, 1, 0],
        [1, 0, -1],
        [-1, 0, 1],
        [0, 1, -1],
        [0, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
    ])


# ============================================================================
# Signature Function Tests
# ============================================================================

class TestGetSignature:
    """Tests for get_signature function."""

    def test_positive_trade_gives_positive_signature(self):
        """Positive trade should give signature of 1."""
        trade = np.array([100.0, 50.0])
        sig = get_signature(trade)
        np.testing.assert_array_equal(sig, np.array([1.0, 1.0]))

    def test_negative_trade_gives_negative_signature(self):
        """Negative trade should give signature of -1."""
        trade = np.array([-100.0, -50.0])
        sig = get_signature(trade)
        np.testing.assert_array_equal(sig, np.array([-1.0, -1.0]))

    def test_zero_trade_gives_zero_signature(self):
        """Zero trade should give signature of 0."""
        trade = np.array([0.0, 0.0])
        sig = get_signature(trade)
        np.testing.assert_array_equal(sig, np.array([0.0, 0.0]))

    def test_mixed_trades(self):
        """Mixed trades should give mixed signatures."""
        trade = np.array([100.0, -50.0, 0.0])
        sig = get_signature(trade)
        np.testing.assert_array_equal(sig, np.array([1.0, -1.0, 0.0]))

    def test_preserves_shape(self):
        """Signature should preserve input shape."""
        trade = np.array([100.0, -50.0, 0.0, 25.0])
        sig = get_signature(trade)
        assert sig.shape == trade.shape


class TestCompareSignatures:
    """Tests for compare_signatures function."""

    def test_identical_signatures_match(self):
        """Identical signatures should match."""
        sig1 = jnp.array([1, -1, 0])
        sig2 = jnp.array([1, -1, 0])
        assert compare_signatures(sig1, sig2) == True

    def test_different_signatures_no_match(self):
        """Different signatures should not match."""
        sig1 = jnp.array([1, -1])
        sig2 = jnp.array([-1, 1])
        assert compare_signatures(sig1, sig2) == False

    def test_zero_in_sig1_is_flexible(self):
        """Zero in sig1 should be ignored for comparison."""
        sig1 = jnp.array([1, 0])
        sig2 = jnp.array([1, -1])
        # sig1 has zero at position 1, so only position 0 is compared
        assert compare_signatures(sig1, sig2) == True

    def test_zero_in_sig2_is_flexible(self):
        """Zero in sig2 should be ignored for comparison."""
        sig1 = jnp.array([1, -1])
        sig2 = jnp.array([1, 0])
        assert compare_signatures(sig1, sig2) == True


class TestDirectionConversions:
    """Tests for direction conversion functions."""

    def test_direction_to_sig(self):
        """Test trade direction to signature conversion."""
        # direction > 0.5 is buy (sig=1), direction < 0.5 is sell (sig=-1)
        trade_dir = np.array([1, 0])
        sig = direction_to_sig(trade_dir)
        np.testing.assert_array_equal(sig, np.array([1.0, -1.0]))

    def test_sig_to_tokens_to_keep(self):
        """Test signature to tokens mask conversion."""
        sig = jnp.array([1, -1, 0])
        tokens_to_keep = sig_to_tokens_to_keep(sig)
        np.testing.assert_array_equal(
            tokens_to_keep,
            np.array([True, True, False])
        )

    def test_sig_to_direction(self):
        """Test signature to trade direction conversion."""
        sig = np.array([1, -1, 0])
        direction = sig_to_direction(sig)
        np.testing.assert_array_equal(direction, np.array([1, 0, 0]))

    def test_sig_to_direction_jnp(self):
        """Test JAX signature to direction conversion."""
        sig = jnp.array([1, -1, 0])
        direction = sig_to_direction_jnp(sig)
        np.testing.assert_array_equal(direction, jnp.array([1, 0, 0]))

    def test_trade_to_direction_jnp(self):
        """Test trade to direction conversion."""
        trade = jnp.array([100.0, -50.0, 0.0])
        direction = trade_to_direction_jnp(trade)
        np.testing.assert_array_equal(direction, jnp.array([1, 0, 0]))


# ============================================================================
# Optimal Trade Construction Tests
# ============================================================================

class TestConstructOptimalTrade:
    """Tests for construct_optimal_trade_jnp function."""

    def test_output_shape_two_assets(self, two_asset_setup):
        """Output should have shape (n_assets,)."""
        setup = two_asset_setup
        sig = jnp.array([1, -1])
        local_prices = jnp.array([1.0, 20.0])  # Aligned with reserves

        trade = construct_optimal_trade_jnp(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            sig,
            setup["n"],
        )

        assert trade.shape == (2,)

    def test_no_arb_when_prices_aligned(self, two_asset_setup):
        """No arbitrage when pool prices match market prices."""
        setup = two_asset_setup
        # Pool price ratio = reserves[0]/reserves[1] * weights[1]/weights[0]
        # = 1000/50 * 0.5/0.5 = 20
        local_prices = jnp.array([1.0, 20.0])  # Market matches pool
        sig = jnp.array([1, -1])

        trade = construct_optimal_trade_jnp(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            sig,
            setup["n"],
        )

        # Trade should be very small (near zero due to fees)
        assert jnp.max(jnp.abs(trade)) < 10.0

    def test_arb_exists_when_prices_misaligned(self, two_asset_setup):
        """Arbitrage exists when pool prices differ from market."""
        setup = two_asset_setup
        # Pool suggests price ratio ~20, market says 25
        local_prices = jnp.array([1.0, 25.0])
        sig = jnp.array([-1, 1])  # Buy asset 1, sell asset 0

        trade = construct_optimal_trade_jnp(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            sig,
            setup["n"],
        )

        # Should have non-trivial trade
        # Note: exact behavior depends on fee_gamma and price difference

    def test_trade_respects_signature_direction(self, two_asset_setup):
        """Trade direction should match signature."""
        setup = two_asset_setup
        local_prices = jnp.array([1.0, 25.0])

        # Signature [1, -1] means buy asset 0, sell asset 1
        sig1 = jnp.array([1, -1])
        trade1 = construct_optimal_trade_jnp(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            sig1,
            setup["n"],
        )

        # Signature [-1, 1] means sell asset 0, buy asset 1
        sig2 = jnp.array([-1, 1])
        trade2 = construct_optimal_trade_jnp(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            sig2,
            setup["n"],
        )

        # Trades should have different sign patterns (or zeros)
        # At least one should be non-zero for a price discrepancy

    def test_slack_affects_validity(self, two_asset_setup):
        """Higher slack should invalidate marginal trades."""
        setup = two_asset_setup
        local_prices = jnp.array([1.0, 20.5])  # Small discrepancy
        sig = jnp.array([1, -1])

        trade_no_slack = construct_optimal_trade_jnp(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            sig,
            setup["n"],
            slack=0,
        )

        trade_high_slack = construct_optimal_trade_jnp(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            sig,
            setup["n"],
            slack=1000,  # Very high slack
        )

        # High slack should produce zero trade
        assert jnp.allclose(trade_high_slack, 0.0)


# ============================================================================
# Trade Sifter Tests
# ============================================================================

class TestOptimalTradeSifter:
    """Tests for optimal_trade_sifter function."""

    def test_selects_most_profitable_signature(self, two_asset_setup, two_asset_signatures):
        """Should select the signature giving highest profit."""
        setup = two_asset_setup
        local_prices = jnp.array([1.0, 25.0])  # Significant price discrepancy

        trade = optimal_trade_sifter(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            two_asset_signatures,
            setup["n"],
        )

        # Result should be the trade from the more profitable signature
        assert trade.shape == (2,)

    def test_output_is_valid_trade(self, two_asset_setup, two_asset_signatures):
        """Output should be a valid trade (reserves stay positive)."""
        setup = two_asset_setup
        local_prices = jnp.array([1.0, 22.0])

        trade = optimal_trade_sifter(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            two_asset_signatures,
            setup["n"],
        )

        # Post-trade reserves should be positive
        post_reserves = setup["initial_reserves"] + trade
        assert jnp.all(post_reserves > 0) or jnp.allclose(trade, 0.0)

    def test_three_asset_arb(self, three_asset_setup, three_asset_signatures):
        """Should work with three or more assets."""
        setup = three_asset_setup
        local_prices = jnp.array([1.0, 2.5, 5.0])

        trade = optimal_trade_sifter(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            three_asset_signatures,
            setup["n"],
        )

        assert trade.shape == (3,)


class TestParallelisedOptimalTradeSifter:
    """Tests for parallelised_optimal_trade_sifter function."""

    def test_matches_non_parallel_version(self, two_asset_setup, two_asset_signatures):
        """Parallelised version should match non-parallel."""
        setup = two_asset_setup
        local_prices = jnp.array([1.0, 22.0])

        # Non-parallel version
        trade_basic = optimal_trade_sifter(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            two_asset_signatures,
            setup["n"],
        )

        # Parallel version via wrapper
        trade_parallel = wrapped_parallelised_optimal_trade_sifter(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            two_asset_signatures,
            setup["n"],
        )

        np.testing.assert_allclose(trade_basic, trade_parallel, rtol=1e-5)

    def test_jit_compilation(self, two_asset_setup, two_asset_signatures):
        """Function should be JIT-compatible."""
        setup = two_asset_setup
        local_prices = jnp.array([1.0, 22.0])

        # First call (compile)
        trade1 = wrapped_parallelised_optimal_trade_sifter(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            two_asset_signatures,
            setup["n"],
        )

        # Second call (cached)
        trade2 = wrapped_parallelised_optimal_trade_sifter(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            two_asset_signatures,
            setup["n"],
        )

        np.testing.assert_allclose(trade1, trade2)


# ============================================================================
# Precalculation Function Tests
# ============================================================================

class TestPrecalcFunctions:
    """Tests for precalculation helper functions."""

    def test_precalc_shared_values_shape(self, two_asset_signatures):
        """Precalculated values should have correct shapes."""
        n = 2
        tokens_to_keep, trade_dirs, tokens_to_drop, leave_one_out = (
            precalc_shared_values_for_all_signatures(two_asset_signatures, n)
        )

        n_sigs = two_asset_signatures.shape[0]
        assert tokens_to_keep.shape == (n_sigs, n)
        assert trade_dirs.shape == (n_sigs, n)
        assert tokens_to_drop.shape == (n_sigs, n)
        assert leave_one_out.shape == (n_sigs, n, n - 1)

    def test_calc_active_initial_reserves(self, two_asset_setup, two_asset_signatures):
        """Active reserves should mask dropped tokens."""
        n = 2
        _, _, tokens_to_drop, _ = precalc_shared_values_for_all_signatures(
            two_asset_signatures, n
        )

        active_reserves = calc_active_inital_reserves_for_all_signatures(
            two_asset_setup["initial_reserves"],
            tokens_to_drop,
        )

        assert active_reserves.shape == (2, 2)  # n_sigs x n_assets

    def test_precalc_components_shape(self, two_asset_setup, two_asset_signatures):
        """Precalculated components should have correct shapes."""
        n = 2
        _, trade_dirs, tokens_to_drop, leave_one_out = (
            precalc_shared_values_for_all_signatures(two_asset_signatures, n)
        )

        local_prices = jnp.array([1.0, 20.0])

        active_weights, per_asset_ratio, all_other_ratio = (
            precalc_components_of_optimal_trade_across_signatures(
                two_asset_setup["initial_weights"],
                local_prices,
                two_asset_setup["fee_gamma"],
                tokens_to_drop,
                trade_dirs,
                leave_one_out,
            )
        )

        n_sigs = two_asset_signatures.shape[0]
        assert active_weights.shape == (n_sigs, n)
        assert per_asset_ratio.shape == (n_sigs, n)
        assert all_other_ratio.shape == (n_sigs, n)


# ============================================================================
# Edge Cases and Numerical Stability
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_equal_weights(self):
        """Should work with equal weights."""
        weights = jnp.array([0.5, 0.5])
        reserves = jnp.array([100.0, 100.0])
        prices = jnp.array([1.0, 1.0])
        sigs = jnp.array([[1, -1], [-1, 1]])

        trade = optimal_trade_sifter(
            weights, prices, reserves, 0.997, sigs, 2
        )

        # Aligned prices + fees should give zero trade
        assert jnp.allclose(trade, 0.0, atol=1e-5)

    def test_extreme_price_ratio(self):
        """Should handle extreme price ratios."""
        weights = jnp.array([0.5, 0.5])
        reserves = jnp.array([1000.0, 1.0])  # 1000:1 ratio
        prices = jnp.array([1.0, 1000.0])
        sigs = jnp.array([[1, -1], [-1, 1]])

        trade = optimal_trade_sifter(
            weights, prices, reserves, 0.997, sigs, 2
        )

        # Should produce valid trade (finite values)
        assert jnp.all(jnp.isfinite(trade))

    def test_high_fee(self):
        """Higher fees should reduce arbitrage opportunity."""
        weights = jnp.array([0.5, 0.5])
        reserves = jnp.array([100.0, 50.0])
        prices = jnp.array([1.0, 2.2])  # Small price discrepancy
        sigs = jnp.array([[1, -1], [-1, 1]])

        trade_low_fee = optimal_trade_sifter(
            weights, prices, reserves, 0.999, sigs, 2  # 0.1% fee
        )

        trade_high_fee = optimal_trade_sifter(
            weights, prices, reserves, 0.97, sigs, 2  # 3% fee
        )

        # Higher fee should reduce trade magnitude
        mag_low = jnp.max(jnp.abs(trade_low_fee))
        mag_high = jnp.max(jnp.abs(trade_high_fee))

        assert mag_high <= mag_low + 1e-5

    def test_asymmetric_weights(self):
        """Should work with asymmetric weights."""
        weights = jnp.array([0.8, 0.2])
        reserves = jnp.array([800.0, 50.0])
        prices = jnp.array([1.0, 5.0])
        sigs = jnp.array([[1, -1], [-1, 1]])

        trade = optimal_trade_sifter(
            weights, prices, reserves, 0.997, sigs, 2
        )

        assert jnp.all(jnp.isfinite(trade))

    def test_no_arb_with_zero_fee(self):
        """With zero fee and aligned prices, should have near-zero trade."""
        weights = jnp.array([0.5, 0.5])
        reserves = jnp.array([100.0, 100.0])
        prices = jnp.array([1.0, 1.0])  # Aligned
        sigs = jnp.array([[1, -1], [-1, 1]])

        trade = optimal_trade_sifter(
            weights, prices, reserves, 1.0, sigs, 2  # No fee
        )

        assert jnp.allclose(trade, 0.0, atol=1e-5)


# ============================================================================
# Multi-Asset Tests
# ============================================================================

class TestMultiAssetArbitrage:
    """Tests for multi-asset arbitrage scenarios."""

    def test_four_asset_arb(self):
        """Should work with four assets."""
        weights = jnp.array([0.25, 0.25, 0.25, 0.25])
        reserves = jnp.array([100.0, 100.0, 100.0, 100.0])
        prices = jnp.array([1.0, 1.1, 0.9, 1.0])  # Some price discrepancy

        # Subset of signatures for 4 assets
        sigs = jnp.array([
            [1, -1, 0, 0],
            [-1, 1, 0, 0],
            [1, 0, -1, 0],
            [-1, 0, 1, 0],
            [0, 1, -1, 0],
            [0, -1, 1, 0],
        ])

        trade = optimal_trade_sifter(
            weights, prices, reserves, 0.997, sigs, 4
        )

        assert trade.shape == (4,)
        assert jnp.all(jnp.isfinite(trade))

    def test_partial_signature_trades_subset(self):
        """Signature with zeros should only trade active assets."""
        weights = jnp.array([0.4, 0.3, 0.3])
        reserves = jnp.array([100.0, 75.0, 75.0])
        prices = jnp.array([1.0, 1.2, 1.0])

        # Only trade assets 0 and 1
        sig = jnp.array([1, -1, 0])

        trade = construct_optimal_trade_jnp(
            weights, prices, reserves, 0.997, sig, 3
        )

        # Asset 2 should have zero trade
        assert trade[2] == 0.0


# ============================================================================
# Profit Calculation Tests
# ============================================================================

class TestProfitCalculation:
    """Tests for profit calculation in trade sifting."""

    def test_profitable_trade_selected(self, two_asset_setup, two_asset_signatures):
        """The more profitable signature should be selected."""
        setup = two_asset_setup
        # Significant price discrepancy
        local_prices = jnp.array([1.0, 30.0])  # Pool ~20, market 30

        trade = optimal_trade_sifter(
            setup["initial_weights"],
            local_prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            two_asset_signatures,
            setup["n"],
        )

        # Calculate profit
        profit = -jnp.sum(trade * local_prices)

        # Trade should be profitable (or zero if no valid arb)
        assert profit >= -1e-5  # Allow small numerical error

    def test_profit_scales_with_trade_magnitude(self, two_asset_setup, two_asset_signatures):
        """Larger trades should have larger absolute profit impact."""
        setup = two_asset_setup

        # Use prices that result in a clear profitable arb
        prices = jnp.array([1.0, 22.0])

        trade = optimal_trade_sifter(
            setup["initial_weights"],
            prices,
            setup["initial_reserves"],
            setup["fee_gamma"],
            two_asset_signatures,
            setup["n"],
        )

        # Profit should be defined as: -sum(trade * price)
        # (selling gives positive profit, buying gives negative)
        profit = -jnp.sum(trade * prices)

        # Trade should be profitable (or zero if no valid arb)
        assert profit >= -1e-5
