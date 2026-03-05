"""End-to-end temporal tests for reClAMM, ported from reClammPool.test.ts.

Tests multi-step behaviour with pinned numeric values: virtual balance
evolution, invariant preservation, fee accumulation, price-range tracking.

Pool parameters match the Solidity integration test suite:
    MIN_PRICE = 0.5, MAX_PRICE = 8, TARGET_PRICE = 3
    PRICE_RATIO = 16, CENTEREDNESS_MARGIN = 0.5
    dailyPriceShiftBase = 1 - 1/124649

Trades are applied using compute_in_given_out / compute_out_given_in
(the reClAMM swap math) to push the pool into known out-of-range states,
mirroring the Solidity test's swapSingleTokenExactOut pattern.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from quantammsim.pools.reCLAMM.reclamm_reserves import (
    compute_invariant,
    compute_centeredness,
    compute_price_range,
    compute_price_ratio,
    compute_theoretical_balances,
    compute_in_given_out,
    compute_out_given_in,
    compute_virtual_balances_updating_price_range,
    initialise_reclamm_reserves,
    _jax_calc_reclamm_reserves_zero_fees,
    _jax_calc_reclamm_reserves_with_fees,
)
from tests.pools.reCLAMM.helpers import _jax_calc_reclamm_reserves_zero_fees_full_state

ALL_SIG_VARIATIONS_2 = jnp.array([[1, -1], [-1, 1]])

# ---------------------------------------------------------------------------
# Solidity pool test parameters (from reClammPool.test.ts)
# ---------------------------------------------------------------------------
SOL_MIN_PRICE = 0.5
SOL_MAX_PRICE = 8.0
SOL_TARGET_PRICE = 3.0
SOL_PRICE_RATIO = 16.0  # 8 / 0.5
SOL_DAILY_PRICE_SHIFT_BASE = 1.0 - 1.0 / 124649.0  # toDailyPriceShiftBase(fp(1))
SOL_CENTEREDNESS_MARGIN = 0.5
SOL_SECONDS_PER_STEP = 60.0
SOL_MIN_POOL_BALANCE = 0.0001

# ---------------------------------------------------------------------------
# Pinned initial state (from compute_theoretical_balances, scaled to Ra=100)
# These match the Solidity test's INITIAL_BALANCE_A = 100.
# ---------------------------------------------------------------------------
_ref_balances, _Va_ref, _Vb_ref = compute_theoretical_balances(
    SOL_MIN_PRICE, SOL_MAX_PRICE, SOL_TARGET_PRICE
)
_SCALE = 100.0 / float(_ref_balances[0])

PINNED_Ra = 100.0
PINNED_Rb = float(_ref_balances[1]) * _SCALE  # 457.9795897113272
PINNED_Va = float(_Va_ref) * _SCALE            # 157.97958971132715
PINNED_Vb = float(_Vb_ref) * _SCALE            # 315.9591794226543
PINNED_L = (PINNED_Ra + PINNED_Va) * (PINNED_Rb + PINNED_Vb)  # ~199660.4
PINNED_SPOT = 3.0
PINNED_INITIAL_CENTEREDNESS = 0.4367006838144547


def _sol_pool():
    """Return the Solidity test's initial pool state."""
    return (
        jnp.array([PINNED_Ra, PINNED_Rb]),
        jnp.array(PINNED_Va),
        jnp.array(PINNED_Vb),
    )


def _apply_swap_exact_out(Ra, Rb, Va, Vb, token_in, token_out, amount_out):
    """Apply a swap (like Solidity's swapSingleTokenExactOut) and return post-trade state.

    Returns (Ra_post, Rb_post) — virtual balances are unchanged by swaps.
    """
    amount_in = float(compute_in_given_out(
        jnp.array(Ra), jnp.array(Rb), jnp.array(Va), jnp.array(Vb),
        token_in, token_out, jnp.array(amount_out),
    ))
    balances = [Ra, Rb]
    balances[token_in] += amount_in
    balances[token_out] -= amount_out
    return balances[0], balances[1]


# ---------------------------------------------------------------------------
# Pinned initial state verification
# ---------------------------------------------------------------------------

class TestPinnedInitialState:
    """Verify the Solidity test's initial pool state is correctly reproduced."""

    def test_spot_price(self):
        spot = (PINNED_Rb + PINNED_Vb) / (PINNED_Ra + PINNED_Va)
        npt.assert_allclose(spot, SOL_TARGET_PRICE, rtol=1e-10)

    def test_price_ratio(self):
        ratio = float(compute_price_ratio(
            jnp.array(PINNED_Ra), jnp.array(PINNED_Rb),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        ))
        npt.assert_allclose(ratio, SOL_PRICE_RATIO, rtol=1e-10)

    def test_initial_centeredness(self):
        c, _ = compute_centeredness(
            jnp.array(PINNED_Ra), jnp.array(PINNED_Rb),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        )
        npt.assert_allclose(float(c), PINNED_INITIAL_CENTEREDNESS, rtol=1e-10)


# ---------------------------------------------------------------------------
# Cross-validation against TypeScript reference (test/pinnedValues.test.ts)
#
# These values were computed by running the TypeScript off-chain math library
# (reClammMath.ts) in the Solidity repo. They use 18-decimal fixed-point
# arithmetic, so expect ~1e-13 relative error vs Python float64.
# ---------------------------------------------------------------------------

class TestCrossValidationVsTypeScript:
    """Cross-validate Python values against TypeScript reference implementation.

    Pinned values from: reclamm/test/pinnedValues.test.ts (7 passing tests).
    Tolerance rtol=1e-10 accounts for fp18 floor-division vs float64.
    """

    def test_initial_state_matches_ts(self):
        """TS: Ra=99.99999999999991, Rb=457.97958971132673, etc."""
        # TS uses fpMulDown(realBalances[0], scale) which introduces fp18 rounding.
        # Python uses exact float. Difference is ~1e-14.
        npt.assert_allclose(PINNED_Ra, 100.0, rtol=1e-10)
        npt.assert_allclose(PINNED_Rb, 457.97958971132673, rtol=1e-10)
        npt.assert_allclose(PINNED_Va, 157.97958971132700, rtol=1e-10)
        npt.assert_allclose(PINNED_Vb, 315.95917942265400, rtol=1e-10)
        npt.assert_allclose(PINNED_INITIAL_CENTEREDNESS, 0.43670068381445478, rtol=1e-10)

    def test_vb_update_above_center_1hr_matches_ts(self):
        """TS pinned: Va=157.97959166481461, Vb=306.96440990737763."""
        amount_out_B = PINNED_Rb - SOL_MIN_POOL_BALANCE
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=0, token_out=1, amount_out=amount_out_B,
        )

        sqrt_Q = float(jnp.sqrt(compute_price_ratio(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        )))
        Va_exp, Vb_exp = compute_virtual_balances_updating_price_range(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            is_pool_above_center=jnp.array(True),
            daily_price_shift_base=SOL_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=3600.0,
            sqrt_price_ratio=jnp.array(sqrt_Q),
        )

        # TS reference values (fp18)
        npt.assert_allclose(float(Va_exp), 157.97959166481461, rtol=1e-10)
        npt.assert_allclose(float(Vb_exp), 306.96440990737763, rtol=1e-10)

    def test_vb_update_below_center_1hr_matches_ts(self):
        """TS pinned: Va=153.48220495368882, Vb=315.95918723660285."""
        amount_out_A = PINNED_Ra - SOL_MIN_POOL_BALANCE
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=1, token_out=0, amount_out=amount_out_A,
        )

        sqrt_Q = float(jnp.sqrt(compute_price_ratio(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        )))
        Va_exp, Vb_exp = compute_virtual_balances_updating_price_range(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            is_pool_above_center=jnp.array(False),
            daily_price_shift_base=SOL_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=3600.0,
            sqrt_price_ratio=jnp.array(sqrt_Q),
        )

        # TS reference values (fp18)
        npt.assert_allclose(float(Va_exp), 153.48220495368882, rtol=1e-10)
        npt.assert_allclose(float(Vb_exp), 315.95918723660285, rtol=1e-10)

    def test_vb_update_above_center_60s_matches_ts(self):
        """TS pinned: Va=157.97958974342494, Vb=315.80712794304925."""
        amount_out_B = PINNED_Rb - SOL_MIN_POOL_BALANCE
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=0, token_out=1, amount_out=amount_out_B,
        )

        sqrt_Q = float(jnp.sqrt(compute_price_ratio(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        )))
        Va_exp, Vb_exp = compute_virtual_balances_updating_price_range(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            is_pool_above_center=jnp.array(True),
            daily_price_shift_base=SOL_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=60.0,
            sqrt_price_ratio=jnp.array(sqrt_Q),
        )

        # TS reference values (fp18)
        npt.assert_allclose(float(Va_exp), 157.97958974342494, rtol=1e-10)
        npt.assert_allclose(float(Vb_exp), 315.80712794304925, rtol=1e-10)

    def test_initial_pool_vb_update_60s_matches_ts(self):
        """TS pinned: Va=157.90356397152462, Vb=316.05884168753558.

        Pool starts out of range (centeredness=0.44 < margin=0.5), so
        VB update fires even without a trade. isAboveCenter=False, so
        Va decays and Vb is recalculated.
        """
        sqrt_Q = float(jnp.sqrt(compute_price_ratio(
            jnp.array(PINNED_Ra), jnp.array(PINNED_Rb),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        )))
        Va_exp, Vb_exp = compute_virtual_balances_updating_price_range(
            jnp.array(PINNED_Ra), jnp.array(PINNED_Rb),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            is_pool_above_center=jnp.array(False),
            daily_price_shift_base=SOL_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=60.0,
            sqrt_price_ratio=jnp.array(sqrt_Q),
        )

        # TS reference values (fp18)
        npt.assert_allclose(float(Va_exp), 157.90356397152462, rtol=1e-10)
        npt.assert_allclose(float(Vb_exp), 316.05884168753558, rtol=1e-10)


# ---------------------------------------------------------------------------
# Pinned virtual balance update (ported from reClammPool.test.ts lines 215-325)
# ---------------------------------------------------------------------------

class TestPinnedVirtualBalanceUpdate:
    """Test compute_virtual_balances_updating_price_range with exact pinned
    values from the TypeScript reference implementation.

    Pattern (matching Solidity):
    1. Big swap pushes pool to edge → known post-trade state
    2. Compute expected virtual balances after time decay
    3. Compare at tight tolerance

    All pinned values sourced from pinnedValues.test.ts. Post-trade states
    from "Post-trade pinned values" section, VB values from "Virtual balance
    update" section. Tolerance rtol=1e-10 for fp18 vs float64.
    """

    def test_above_center_1hour(self):
        """Big A→B swap → pool above center → Vb decays, Va grows.

        TS reference: pinnedValues.test.ts "above center, 1 hour"
        """
        # Apply big A→B swap (remove nearly all B, like Solidity test)
        amount_out_B = PINNED_Rb - SOL_MIN_POOL_BALANCE
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=0, token_out=1, amount_out=amount_out_B,
        )

        # TS pinned post-trade state (pinnedValues.test.ts "Post-trade pinned values")
        npt.assert_allclose(Ra_post, 473.93856913404424863, rtol=1e-10)
        npt.assert_allclose(Rb_post, 0.0001, rtol=1e-6)

        # Post-trade: pool is above center
        center, above = compute_centeredness(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        )
        assert bool(above) is True
        assert float(center) < SOL_CENTEREDNESS_MARGIN

        # Expected virtual balances after 1 hour (3600s)
        sqrt_Q = float(jnp.sqrt(jnp.array(compute_price_ratio(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        ))))
        Va_exp, Vb_exp = compute_virtual_balances_updating_price_range(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            is_pool_above_center=jnp.array(True),
            daily_price_shift_base=SOL_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=3600.0,
            sqrt_price_ratio=jnp.array(sqrt_Q),
        )

        # TS pinned VB values (pinnedValues.test.ts "above center, 1 hour")
        npt.assert_allclose(float(Va_exp), 157.97959166481461, rtol=1e-10)
        npt.assert_allclose(float(Vb_exp), 306.96440990737763, rtol=1e-10)

        # Direction: Va grows (recalculated), Vb decays (overvalued)
        assert float(Va_exp) > PINNED_Va
        assert float(Vb_exp) < PINNED_Vb

    def test_below_center_1hour(self):
        """Big B→A swap → pool below center → Va decays, Vb grows.

        TS reference: pinnedValues.test.ts "below center, 1 hour"
        """
        amount_out_A = PINNED_Ra - SOL_MIN_POOL_BALANCE
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=1, token_out=0, amount_out=amount_out_A,
        )

        # TS pinned post-trade state (pinnedValues.test.ts "Post-trade pinned values")
        npt.assert_allclose(Ra_post, 0.0001, rtol=1e-6)
        npt.assert_allclose(Rb_post, 947.87673826846829288, rtol=1e-10)

        center, above = compute_centeredness(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        )
        assert bool(above) is False
        assert float(center) < SOL_CENTEREDNESS_MARGIN

        sqrt_Q = float(jnp.sqrt(jnp.array(compute_price_ratio(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        ))))
        Va_exp, Vb_exp = compute_virtual_balances_updating_price_range(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            is_pool_above_center=jnp.array(False),
            daily_price_shift_base=SOL_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=3600.0,
            sqrt_price_ratio=jnp.array(sqrt_Q),
        )

        # TS pinned VB values (pinnedValues.test.ts "below center, 1 hour")
        npt.assert_allclose(float(Va_exp), 153.48220495368882, rtol=1e-10)
        npt.assert_allclose(float(Vb_exp), 315.95918723660285, rtol=1e-10)

        # Direction: Va decays (overvalued), Vb grows (recalculated)
        assert float(Va_exp) < PINNED_Va
        assert float(Vb_exp) > PINNED_Vb

    def test_above_center_1step(self):
        """Same as above but for a single 60s step — matches scan step size.

        TS reference: pinnedValues.test.ts "above center, 60 seconds (1 scan step)"
        """
        amount_out_B = PINNED_Rb - SOL_MIN_POOL_BALANCE
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=0, token_out=1, amount_out=amount_out_B,
        )

        sqrt_Q = float(jnp.sqrt(jnp.array(compute_price_ratio(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
        ))))
        Va_exp, Vb_exp = compute_virtual_balances_updating_price_range(
            jnp.array(Ra_post), jnp.array(Rb_post),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            is_pool_above_center=jnp.array(True),
            daily_price_shift_base=SOL_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=60.0,
            sqrt_price_ratio=jnp.array(sqrt_Q),
        )

        # TS pinned VB values (pinnedValues.test.ts "above center, 60 seconds")
        npt.assert_allclose(float(Va_exp), 157.97958974342494, rtol=1e-10)
        npt.assert_allclose(float(Vb_exp), 315.80712794304925, rtol=1e-10)


# ---------------------------------------------------------------------------
# Pinned scan output (trade → scan → compare reserves + virtual balances)
#
# Values cross-validated against TypeScript reference (pinnedValues.test.ts,
# "Multi-step scan with arb" tests). TS uses fp18 fixed-point; expect
# ~1e-12 relative difference vs Python float64.
# ---------------------------------------------------------------------------

class TestPinnedScanFromTrade:
    """Apply a trade to push pool out of range, then run the scan and
    compare reserves and virtual balances to pinned expected values.

    This tests the full pipeline: virtual balance update + arb in one step.
    Pinned values sourced from TypeScript reference (simulateScanStep).
    """

    def test_above_center_scan_3_steps(self):
        """A→B swap → above center → scan 3 steps at target price.

        TS reference: pinnedValues.test.ts "above center: big A→B swap then 3 scan steps"
        """
        amount_out_B = PINNED_Rb - SOL_MIN_POOL_BALANCE - 1e-10
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=0, token_out=1, amount_out=amount_out_B,
        )

        prices = jnp.tile(jnp.array([SOL_TARGET_PRICE, 1.0]), (3, 1))
        R_out, Va_h, Vb_h = _jax_calc_reclamm_reserves_zero_fees_full_state(
            jnp.array([Ra_post, Rb_post]),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            prices, SOL_CENTEREDNESS_MARGIN,
            SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        # Step 0 (TS: Ra=99.9379177675, Rb=457.9453945898)
        npt.assert_allclose(float(R_out[0, 0]),  99.9379177675, rtol=1e-8)
        npt.assert_allclose(float(R_out[0, 1]), 457.9453945898, rtol=1e-8)
        # TS: Va=157.979589743424939, Vb=315.807127943049246
        npt.assert_allclose(float(Va_h[0]), 157.979589743424939, rtol=1e-10)
        npt.assert_allclose(float(Vb_h[0]), 315.807127943049246, rtol=1e-10)

        # Step 1 (TS: Ra=99.9925181704, Rb=457.7815586946)
        npt.assert_allclose(float(R_out[1, 0]),  99.9925181704, rtol=1e-8)
        npt.assert_allclose(float(R_out[1, 1]), 457.7815586946, rtol=1e-8)
        # TS: Va=157.903564003607115, Vb=315.906687827474542
        npt.assert_allclose(float(Va_h[1]), 157.903564003607115, rtol=1e-10)
        npt.assert_allclose(float(Vb_h[1]), 315.906687827474542, rtol=1e-10)

        # Step 2 (TS: Ra=100.0471205253, Rb=457.6177169380)
        npt.assert_allclose(float(R_out[2, 0]), 100.0471205253, rtol=1e-8)
        npt.assert_allclose(float(R_out[2, 1]), 457.6177169380, rtol=1e-8)
        # TS: Va=157.827574850244062, Vb=316.006369188706484
        npt.assert_allclose(float(Va_h[2]), 157.827574850244062, rtol=1e-10)
        npt.assert_allclose(float(Vb_h[2]), 316.006369188706484, rtol=1e-10)

    def test_below_center_scan_3_steps(self):
        """B→A swap → below center → scan 3 steps at target price.

        TS reference: pinnedValues.test.ts "below center: big B→A swap then 3 scan steps"
        """
        amount_out_A = PINNED_Ra - SOL_MIN_POOL_BALANCE - 1e-10
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=1, token_out=0, amount_out=amount_out_A,
        )

        prices = jnp.tile(jnp.array([SOL_TARGET_PRICE, 1.0]), (3, 1))
        R_out, Va_h, Vb_h = _jax_calc_reclamm_reserves_zero_fees_full_state(
            jnp.array([Ra_post, Rb_post]),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            prices, SOL_CENTEREDNESS_MARGIN,
            SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        # Step 0 (TS: Ra=100.0139435656, Rb=457.7933430604)
        npt.assert_allclose(float(R_out[0, 0]), 100.0139435656, rtol=1e-8)
        npt.assert_allclose(float(R_out[0, 1]), 457.7933430604, rtol=1e-8)
        # TS: Va=157.903563971524623, Vb=315.959179551045762
        npt.assert_allclose(float(Va_h[0]), 157.903563971524623, rtol=1e-10)
        npt.assert_allclose(float(Vb_h[0]), 315.959179551045762, rtol=1e-10)

        # Step 1 (TS: Ra=100.0685518134, Rb=457.6294836205)
        npt.assert_allclose(float(R_out[1, 0]), 100.0685518134, rtol=1e-8)
        npt.assert_allclose(float(R_out[1, 1]), 457.6294836205, rtol=1e-8)
        # TS: Va=157.827574818177010, Vb=316.058896274364598
        npt.assert_allclose(float(Va_h[1]), 157.827574818177010, rtol=1e-10)
        npt.assert_allclose(float(Vb_h[1]), 316.058896274364598, rtol=1e-10)

        # Step 2 (TS: Ra=100.1231620569, Rb=457.4656181882)
        npt.assert_allclose(float(R_out[2, 0]), 100.1231620569, rtol=1e-8)
        npt.assert_allclose(float(R_out[2, 1]), 457.4656181882, rtol=1e-8)
        # TS: Va=157.751622233677377, Vb=316.158734683673449
        npt.assert_allclose(float(Va_h[2]), 157.751622233677377, rtol=1e-10)
        npt.assert_allclose(float(Vb_h[2]), 316.158734683673449, rtol=1e-10)

    def test_above_center_with_fees(self):
        """A→B swap → above center → scan with 1% fee.

        Fees reduce arb magnitude: fee reserves should be closer to
        the pre-arb state than zero-fee reserves.

        Zero-fee step 0 reserves from TS (pinnedValues.test.ts "above center scan").
        Fee reserves are Python-only (no TS equivalent — TS doesn't model fees).
        """
        amount_out_B = PINNED_Rb - SOL_MIN_POOL_BALANCE
        Ra_post, Rb_post = _apply_swap_exact_out(
            PINNED_Ra, PINNED_Rb, PINNED_Va, PINNED_Vb,
            token_in=0, token_out=1, amount_out=amount_out_B,
        )

        prices = jnp.tile(jnp.array([SOL_TARGET_PRICE, 1.0]), (3, 1))

        fee_R = _jax_calc_reclamm_reserves_with_fees(
            jnp.array([Ra_post, Rb_post]),
            jnp.array(PINNED_Va), jnp.array(PINNED_Vb),
            prices, SOL_CENTEREDNESS_MARGIN,
            SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
            fees=0.01, arb_thresh=0.0, arb_fees=0.0,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )

        # Fee reserves should have less trade magnitude than zero-fee
        # Zero-fee step 0 from TS: Ra=99.9379177675, Rb=457.9453945898
        zf_Ra_0 = 99.9379177675   # TS pinned
        zf_Rb_0 = 457.9453945898  # TS pinned
        zf_delta = abs(zf_Ra_0 - Ra_post) + abs(zf_Rb_0 - Rb_post)
        fee_delta = abs(float(fee_R[0, 0]) - Ra_post) + abs(float(fee_R[0, 1]) - Rb_post)
        assert fee_delta < zf_delta


# ---------------------------------------------------------------------------
# De novo: Invariant behaviour under SOL params
#
# NOT ported from Solidity. These test invariant properties specific to our
# scan-based implementation with the SOL pool configuration.
#
# Key fact: with SOL params (centeredness_margin=0.5), the pool starts at
# centeredness=0.44, which is BELOW the margin. So VB updates fire from
# step 0 even at constant prices. Each VB update changes L.
# ---------------------------------------------------------------------------

class TestDeNovoInvariantBehaviour:
    """L = (Ra + Va) * (Rb + Vb) behaviour under SOL params.

    With centeredness_margin=0.5, the pool starts out of range
    (initial centeredness=0.44). VB updates fire every step, changing L.
    L decreases monotonically as VB updates shift the range toward market price.

    NOT ported from Solidity. L values cross-validated against TypeScript
    reference (pinnedValues.test.ts "from initial pool: 5 scan steps").
    """

    def test_invariant_step0_shift(self):
        """At step 0, VB update fires (pool out of range), L decreases slightly.

        TS reference step 0: L=199627.270109
        """
        reserves, Va, Vb = _sol_pool()
        prices = jnp.tile(jnp.array([SOL_TARGET_PRICE, 1.0]), (5, 1))

        R_out, Va_h, Vb_h = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        npt.assert_allclose(float(PINNED_L), 199660.40612287412, rtol=1e-10)

        L_0 = float(compute_invariant(R_out[0, 0], R_out[0, 1], Va_h[0], Vb_h[0]))
        # TS pinned: L=199627.270109 at step 0
        npt.assert_allclose(L_0, 199627.270109, rtol=1e-8)
        assert L_0 < float(PINNED_L)

    def test_invariant_decreases_monotonically(self):
        """L decreases slowly each step as VB updates shift the range.

        TS reference: step 1 L=199594.196522, step 4 L=199495.350462
        """
        reserves, Va, Vb = _sol_pool()
        prices = jnp.tile(jnp.array([SOL_TARGET_PRICE, 1.0]), (5, 1))

        R_out, Va_h, Vb_h = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        L_values = [
            float(compute_invariant(R_out[i, 0], R_out[i, 1], Va_h[i], Vb_h[i]))
            for i in range(R_out.shape[0])
        ]

        # TS pinned values
        npt.assert_allclose(L_values[1], 199594.196522, rtol=1e-8)
        npt.assert_allclose(L_values[4], 199495.350462, rtol=1e-8)

        for i in range(1, len(L_values)):
            assert L_values[i] < L_values[i - 1], \
                f"L should decrease: step {i-1}={L_values[i-1]:.4f}, step {i}={L_values[i]:.4f}"

    def test_invariant_positive_finite_under_stress(self):
        """Under large price moves with virtual balance updates, L should
        stay positive and finite (it may shift value due to VB updates).
        """
        reserves, Va, Vb = _sol_pool()
        n_steps = 30
        prices = jnp.tile(jnp.array([6.0, 1.0]), (n_steps, 1))

        R_out, Va_h, Vb_h = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        for i in range(R_out.shape[0]):
            L_i = compute_invariant(R_out[i, 0], R_out[i, 1], Va_h[i], Vb_h[i])
            assert jnp.isfinite(L_i), f"Non-finite invariant at step {i}"
            assert float(L_i) > 0, f"Non-positive invariant at step {i}"


# ---------------------------------------------------------------------------
# Fee accumulation with pinned values
# ---------------------------------------------------------------------------

class TestPinnedFeeAccumulation:
    """Fees protect pool value against LVR. Higher fees → more value retained."""

    def test_fee_monotonic_with_pinned_values(self):
        """Run the same volatile path with 0%, 1%, 5%, 10% fees.
        Pin the final pool values. Verify monotonic increase.
        """
        reserves, Va, Vb = _sol_pool()

        np.random.seed(42)
        n_steps = 50
        log_returns = np.random.normal(0, 0.03, n_steps)
        price_a = SOL_TARGET_PRICE * np.exp(np.cumsum(log_returns))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        # Zero-fee
        zf_R = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )
        zf_value = float((zf_R[-1] * prices[-1]).sum())

        # Fee runs
        fee_values = {}
        for fee in [0.01, 0.05, 0.10]:
            fee_R = _jax_calc_reclamm_reserves_with_fees(
                reserves, Va, Vb, prices,
                SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE,
                SOL_SECONDS_PER_STEP,
                fees=fee, arb_thresh=0.0, arb_fees=0.0,
                all_sig_variations=ALL_SIG_VARIATIONS_2,
            )
            fee_values[fee] = float((fee_R[-1] * prices[-1]).sum())

        # Monotonic: 0% <= 1% <= 5% <= 10%
        assert zf_value <= fee_values[0.01] + 1e-6
        assert fee_values[0.01] <= fee_values[0.05] + 1e-6
        assert fee_values[0.05] <= fee_values[0.10] + 1e-6

        # 10% fee should retain substantially more than zero-fee
        assert fee_values[0.10] > zf_value * 1.01, \
            f"10% fee should retain >1% more: zf={zf_value:.4f}, 10%={fee_values[0.10]:.4f}"


# ---------------------------------------------------------------------------
# Price range tracking under SOL params
#
# All midpoint values cross-validated against TypeScript reference
# (pinnedValues.test.ts "Price range midpoints for trending paths").
#
# With SOL params (centeredness_margin=0.5), the pool starts out of range
# (centeredness=0.44), so VB updates fire from step 0.
# ---------------------------------------------------------------------------

class TestPinnedPriceRangeTracking:
    """The pool's price range shifts toward market price over time.

    This is the defining property of reClAMM vs static concentrated liquidity.
    Uses full SOL params (centeredness_margin=0.5).

    All midpoint values sourced from TypeScript reference
    (pinnedValues.test.ts "Price range midpoints for trending paths").
    Tolerance rtol=1e-8 for fp18 vs float64 accumulated over 120 scan steps.
    """

    def test_initial_range_shift_at_step0(self):
        """With SOL params, the pool starts out of range (centeredness=0.44 < 0.5).
        At step 0, the VB update fires and the midpoint shifts slightly upward.

        TS reference: pinnedValues.test.ts "up path" and "down path" step 0
        both give mid=2.0015940979 (identical since both start at price=3.0).
        """
        reserves, Va, Vb = _sol_pool()

        # Pinned initial range
        min_p0, max_p0 = compute_price_range(reserves[0], reserves[1], Va, Vb)
        mid_0 = float(jnp.sqrt(min_p0 * max_p0))
        npt.assert_allclose(mid_0, 2.0, rtol=1e-6)  # sqrt(0.5 * 8) = 2.0

        prices = jnp.tile(jnp.array([SOL_TARGET_PRICE, 1.0]), (1, 1))
        R_out, Va_h, Vb_h = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        min_p1, max_p1 = compute_price_range(R_out[0, 0], R_out[0, 1], Va_h[0], Vb_h[0])
        mid_1 = float(jnp.sqrt(min_p1 * max_p1))

        # TS pinned: step 0 mid=2.0015940979
        npt.assert_allclose(mid_1, 2.0015940979, rtol=1e-8)
        assert mid_1 > mid_0  # slight increase

    def test_up_vs_down_divergence(self):
        """Sustained price increase vs decrease → midpoints diverge.

        The core property: range tracks market price direction.

        TS reference: pinnedValues.test.ts
          "up path: 3→6 over 120 steps" step 119 mid=2.1712290354
          "down path: 3→1 over 120 steps" step 119 mid=1.9796381889
        """
        reserves, Va, Vb = _sol_pool()
        n_steps = 120

        # Up path: 3 → 6
        price_up = jnp.linspace(SOL_TARGET_PRICE, 6.0, n_steps)
        prices_up = jnp.stack([price_up, jnp.ones(n_steps)], axis=1)
        R_up, Va_up, Vb_up = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices_up,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        # Down path: 3 → 1
        price_dn = jnp.linspace(SOL_TARGET_PRICE, 1.0, n_steps)
        prices_dn = jnp.stack([price_dn, jnp.ones(n_steps)], axis=1)
        R_dn, Va_dn, Vb_dn = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices_dn,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        # TS pinned step 0: both paths start at mid=2.0015940979
        min_up_0, max_up_0 = compute_price_range(R_up[0, 0], R_up[0, 1], Va_up[0], Vb_up[0])
        min_dn_0, max_dn_0 = compute_price_range(R_dn[0, 0], R_dn[0, 1], Va_dn[0], Vb_dn[0])
        mid_up_0 = float(jnp.sqrt(min_up_0 * max_up_0))
        mid_dn_0 = float(jnp.sqrt(min_dn_0 * max_dn_0))
        npt.assert_allclose(mid_up_0, 2.0015940979, rtol=1e-8)
        npt.assert_allclose(mid_dn_0, 2.0015940979, rtol=1e-8)

        # TS pinned final midpoints (step 119)
        min_up_f, max_up_f = compute_price_range(R_up[-1, 0], R_up[-1, 1], Va_up[-1], Vb_up[-1])
        min_dn_f, max_dn_f = compute_price_range(R_dn[-1, 0], R_dn[-1, 1], Va_dn[-1], Vb_dn[-1])
        mid_up_f = float(jnp.sqrt(min_up_f * max_up_f))
        mid_dn_f = float(jnp.sqrt(min_dn_f * max_dn_f))

        npt.assert_allclose(mid_up_f, 2.1712290354, rtol=1e-8)
        npt.assert_allclose(mid_dn_f, 1.9796381889, rtol=1e-8)

        # Core property: up path midpoint > down path midpoint
        assert mid_up_f > mid_dn_f, \
            f"Up midpoint should exceed down: up={mid_up_f:.6f}, down={mid_dn_f:.6f}"

    def test_range_midpoint_trajectory_pinned(self):
        """Pin the midpoint trajectory at specific steps for both paths.

        TS reference: pinnedValues.test.ts "Price range midpoints for trending paths"
          up step 0: 2.0015940979, step 59: 2.0899852595, step 119: 2.1712290354
          down step 0: 2.0015940979, step 59: 2.0178247023, step 119: 1.9796381889
        """
        reserves, Va, Vb = _sol_pool()
        n_steps = 120

        # Up path: 3 → 6
        price_up = jnp.linspace(SOL_TARGET_PRICE, 6.0, n_steps)
        prices_up = jnp.stack([price_up, jnp.ones(n_steps)], axis=1)
        R_up, Va_up, Vb_up = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices_up,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        # Down path: 3 → 1
        price_dn = jnp.linspace(SOL_TARGET_PRICE, 1.0, n_steps)
        prices_dn = jnp.stack([price_dn, jnp.ones(n_steps)], axis=1)
        R_dn, Va_dn, Vb_dn = _jax_calc_reclamm_reserves_zero_fees_full_state(
            reserves, Va, Vb, prices_dn,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        def _mid(R, Va_h, Vb_h, i):
            min_p, max_p = compute_price_range(R[i, 0], R[i, 1], Va_h[i], Vb_h[i])
            return float(jnp.sqrt(min_p * max_p))

        # TS pinned up path midpoints
        npt.assert_allclose(_mid(R_up, Va_up, Vb_up, 0), 2.0015940979, rtol=1e-8)
        npt.assert_allclose(_mid(R_up, Va_up, Vb_up, 59), 2.0899852595, rtol=1e-8)
        npt.assert_allclose(_mid(R_up, Va_up, Vb_up, 119), 2.1712290354, rtol=1e-8)

        # TS pinned down path midpoints
        npt.assert_allclose(_mid(R_dn, Va_dn, Vb_dn, 0), 2.0015940979, rtol=1e-8)
        npt.assert_allclose(_mid(R_dn, Va_dn, Vb_dn, 59), 2.0178247023, rtol=1e-8)
        npt.assert_allclose(_mid(R_dn, Va_dn, Vb_dn, 119), 1.9796381889, rtol=1e-8)

        # Up path: midpoint increases monotonically
        up_mids = [_mid(R_up, Va_up, Vb_up, i) for i in range(n_steps)]
        for i in range(1, len(up_mids)):
            assert up_mids[i] >= up_mids[i-1] - 1e-10, \
                f"Up midpoint should not decrease: step {i-1}={up_mids[i-1]:.6f}, step {i}={up_mids[i]:.6f}"


# ---------------------------------------------------------------------------
# Pool value trajectory (LVR)
# ---------------------------------------------------------------------------

class TestPinnedPoolValue:
    """Zero-fee pool loses value to LVR. Round-trip should not create value.

    Initial pool value = Ra*3 + Rb*1. Ra and Rb are cross-validated against TS
    (TestCrossValidationVsTypeScript::test_initial_state_matches_ts), so the
    initial value is transitively TS-sourced: 100*3 + 457.97958971132673 = 757.9796.
    """

    def test_round_trip_no_value_creation(self):
        """Price round trip (3 → 5 → 3): pool should lose value to LVR."""
        reserves, Va, Vb = _sol_pool()
        initial_value = float((reserves * jnp.array([SOL_TARGET_PRICE, 1.0])).sum())

        n_steps = 100
        half = n_steps // 2
        price_up = np.linspace(SOL_TARGET_PRICE, 5.0, half)
        price_down = np.linspace(5.0, SOL_TARGET_PRICE, n_steps - half)
        price_a = np.concatenate([price_up, price_down])
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        R_out = _jax_calc_reclamm_reserves_zero_fees(
            reserves, jnp.array(PINNED_Va), jnp.array(PINNED_Vb), prices,
            SOL_CENTEREDNESS_MARGIN, SOL_DAILY_PRICE_SHIFT_BASE, SOL_SECONDS_PER_STEP,
        )

        final_value = float((R_out[-1] * prices[-1]).sum())

        # Pinned initial value
        npt.assert_allclose(initial_value, 757.9795897113272, rtol=1e-10)

        # Pool loses value on round trip (LVR)
        assert final_value < initial_value, \
            f"Pool should lose value on round trip: initial={initial_value:.4f}, final={final_value:.4f}"
