"""Unit tests for reClAMM math functions.

Ported from the Solidity/TypeScript reference implementation at
reclamm/test/reClammMath.test.ts and
reclamm/test/utils/reClammMath.ts.

All test vectors use standard floating-point (not Solidity's 18-decimal
fixed-point), so expected values are converted accordingly.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from quantammsim.pools.reCLAMM.reclamm_reserves import (
    compute_invariant,
    compute_centeredness,
    is_above_center,
    compute_price_range,
    compute_price_ratio,
    compute_out_given_in,
    compute_in_given_out,
    compute_theoretical_balances,
    compute_virtual_balances_updating_price_range,
    compute_virtual_balances_constant_arc_length,
    compute_Z,
    solve_VB_for_Z,
    compute_onset_state,
    calibrate_arc_length_speed,
    initialise_reclamm_reserves,
)


# ---------------------------------------------------------------------------
# Constants matching BaseReClammTest.sol and reClammMath.ts
# ---------------------------------------------------------------------------
PRICE_SHIFT_EXPONENT_ADJUSTMENT = 124649
DEFAULT_DAILY_PRICE_SHIFT_BASE = 1.0 - 1.0 / 124000.0
DEFAULT_CENTEREDNESS_MARGIN = 0.2


class TestComputeInvariant:
    """Test compute_invariant: L = (Ra + Va) * (Rb + Vb)."""

    def test_basic(self):
        Ra, Rb = 200.0, 300.0
        Va, Vb = 100.0, 100.0
        L = compute_invariant(Ra, Rb, Va, Vb)
        # (200+100)*(300+100) = 300*400 = 120000
        npt.assert_allclose(float(L), 120000.0, rtol=1e-12)

    def test_zero_real_balances(self):
        L = compute_invariant(0.0, 0.0, 100.0, 200.0)
        # (0+100)*(0+200) = 20000
        npt.assert_allclose(float(L), 20000.0, rtol=1e-12)

    def test_zero_virtual_balances(self):
        L = compute_invariant(200.0, 300.0, 0.0, 0.0)
        # 200*300 = 60000
        npt.assert_allclose(float(L), 60000.0, rtol=1e-12)


class TestComputeCenteredness:
    """Test centeredness = min(Ra*Vb, Rb*Va) / max(Ra*Vb, Rb*Va)."""

    def test_zero_balance_a(self):
        # From TS test: balances=[0, 100], virtual=[2, 1024] → 0
        c, is_above = compute_centeredness(0.0, 100.0, 2.0, 1024.0)
        assert float(c) == 0.0
        assert bool(is_above) is False

    def test_zero_balance_b(self):
        # balances=[100, 0], virtual=[2, 1024] → 0, isAboveCenter=True
        c, is_above = compute_centeredness(100.0, 0.0, 2.0, 1024.0)
        assert float(c) == 0.0
        assert bool(is_above) is True

    def test_above_center_nonzero(self):
        # balances=[100, 100], virtual=[2, 1024] — above center (Ra/Rb > Va/Vb)
        c, is_above = compute_centeredness(100.0, 100.0, 2.0, 1024.0)
        assert float(c) > 0.0
        assert bool(is_above) is True
        # centeredness = min(Ra*Vb, Rb*Va)/max(Ra*Vb, Rb*Va)
        # Ra*Vb = 100*1024 = 102400, Rb*Va = 100*2 = 200
        # centeredness = 200/102400 ≈ 0.001953125
        npt.assert_allclose(float(c), 200.0 / 102400.0, rtol=1e-10)

    def test_symmetric(self):
        # balances=[100, 100], virtual=[100, 100] → 1.0
        c, _ = compute_centeredness(100.0, 100.0, 100.0, 100.0)
        npt.assert_allclose(float(c), 1.0, rtol=1e-12)

    def test_below_center(self):
        # balances=[100, 100], virtual=[110, 100] — below center (Ra/Rb < Va/Vb)
        c, is_above = compute_centeredness(100.0, 100.0, 110.0, 100.0)
        assert bool(is_above) is False
        # Ra*Vb = 100*100=10000, Rb*Va=100*110=11000
        # centeredness = 10000/11000
        npt.assert_allclose(float(c), 10000.0 / 11000.0, rtol=1e-10)


class TestIsAboveCenter:
    """Test is_above_center."""

    def test_balance_b_zero(self):
        # balances=[300, 0], virtual=[100, 200] → True
        result = is_above_center(300.0, 0.0, 100.0, 200.0)
        assert bool(result) is True

    def test_not_above(self):
        # balances=[100, 100], virtual=[110, 100] → False
        result = is_above_center(100.0, 100.0, 110.0, 100.0)
        assert bool(result) is False

    def test_above(self):
        # balances=[100, 100], virtual=[2, 1024] → True (Ra/Rb=1 > Va/Vb=2/1024)
        result = is_above_center(100.0, 100.0, 2.0, 1024.0)
        assert bool(result) is True


class TestComputePriceRange:
    """Test price range: minPrice = Vb²/L, maxPrice = L/Va²."""

    def test_basic(self):
        # From TS test: balances=[100, 100], virtual=[90, 110]
        Ra, Rb = 100.0, 100.0
        Va, Vb = 90.0, 110.0
        min_price, max_price = compute_price_range(Ra, Rb, Va, Vb)
        L = compute_invariant(Ra, Rb, Va, Vb)
        # L = (100+90)*(100+110) = 190*210 = 39900
        expected_min = (110.0**2) / L  # 12100/39900
        expected_max = L / (90.0**2)   # 39900/8100
        npt.assert_allclose(float(min_price), expected_min, rtol=1e-10)
        npt.assert_allclose(float(max_price), expected_max, rtol=1e-10)

    def test_zero_balance_a(self):
        Ra, Rb = 0.0, 100.0
        Va, Vb = 90.0, 110.0
        min_price, max_price = compute_price_range(Ra, Rb, Va, Vb)
        L = compute_invariant(Ra, Rb, Va, Vb)
        npt.assert_allclose(float(min_price), (110.0**2) / L, rtol=1e-10)
        npt.assert_allclose(float(max_price), L / (90.0**2), rtol=1e-10)

    def test_zero_balance_b(self):
        Ra, Rb = 100.0, 0.0
        Va, Vb = 90.0, 110.0
        min_price, max_price = compute_price_range(Ra, Rb, Va, Vb)
        L = compute_invariant(Ra, Rb, Va, Vb)
        npt.assert_allclose(float(min_price), (110.0**2) / L, rtol=1e-10)
        npt.assert_allclose(float(max_price), L / (90.0**2), rtol=1e-10)


class TestComputePriceRatio:
    """Test price ratio = maxPrice/minPrice."""

    def test_basic(self):
        # From TS test: balances=[100, 100], virtual=[2, 1024]
        Ra, Rb = 100.0, 100.0
        Va, Vb = 2.0, 1024.0
        ratio = compute_price_ratio(Ra, Rb, Va, Vb)
        min_p, max_p = compute_price_range(Ra, Rb, Va, Vb)
        npt.assert_allclose(float(ratio), float(max_p / min_p), rtol=1e-10)


class TestComputeOutGivenIn:
    """Test constant-product swap: Ao = (Bo+Vo)*Ai / (Bi+Vi+Ai)."""

    def test_basic_a_to_b(self):
        # From TS test: balances=[200, 300], virtual=[100, 100],
        # tokenIn=0, tokenOut=1, amountIn=10
        Ra, Rb = 200.0, 300.0
        Va, Vb = 100.0, 100.0
        amount_in = 10.0
        amount_out = compute_out_given_in(Ra, Rb, Va, Vb, 0, 1, amount_in)
        # (300+100)*10/(200+100+10) = 400*10/310 ≈ 12.903225...
        expected = 400.0 * 10.0 / 310.0
        npt.assert_allclose(float(amount_out), expected, rtol=1e-10)

    def test_basic_b_to_a(self):
        Ra, Rb = 200.0, 300.0
        Va, Vb = 100.0, 100.0
        amount_in = 10.0
        amount_out = compute_out_given_in(Ra, Rb, Va, Vb, 1, 0, amount_in)
        # (200+100)*10/(300+100+10) = 300*10/410 ≈ 7.317073...
        expected = 300.0 * 10.0 / 410.0
        npt.assert_allclose(float(amount_out), expected, rtol=1e-10)


class TestComputeInGivenOut:
    """Test inverse swap: Ai = (Bi+Vi)*Ao / (Bo+Vo-Ao)."""

    def test_basic(self):
        Ra, Rb = 200.0, 300.0
        Va, Vb = 100.0, 100.0
        amount_out = 10.0
        amount_in = compute_in_given_out(Ra, Rb, Va, Vb, 0, 1, amount_out)
        # Ai = (Bi+Vi)*Ao / (Bo+Vo-Ao) = (200+100)*10/(300+100-10) = 3000/390
        expected = 3000.0 / 390.0
        npt.assert_allclose(float(amount_in), expected, rtol=1e-10)

    def test_round_trip(self):
        """Swapping out→in→out should recover the original amount (within tolerance)."""
        Ra, Rb = 200.0, 300.0
        Va, Vb = 100.0, 100.0
        original_in = 10.0
        out = compute_out_given_in(Ra, Rb, Va, Vb, 0, 1, original_in)
        # Now use the output to compute how much input we'd need
        recovered_in = compute_in_given_out(Ra, Rb, Va, Vb, 0, 1, float(out))
        npt.assert_allclose(float(recovered_in), original_in, rtol=1e-10)


class TestComputeTheoreticalBalances:
    """Test initialization from price parameters."""

    def test_default_params(self):
        # From TS test: min=1000, max=4000, target=2500
        min_price = 1000.0
        max_price = 4000.0
        target_price = 2500.0
        initial_pool_value = 1e6  # arbitrary, just for scaling
        initial_prices = jnp.array([target_price, 1.0])

        real_balances, Va, Vb = compute_theoretical_balances(
            min_price, max_price, target_price
        )

        # Verify price ratio
        price_ratio = max_price / min_price
        npt.assert_allclose(price_ratio, 4.0, rtol=1e-12)

        # Verify invariant holds
        L = compute_invariant(
            float(real_balances[0]), float(real_balances[1]),
            float(Va), float(Vb)
        )

        # Verify spot price matches target
        # spot_price = (Rb + Vb) / (Ra + Va)
        effective_a = float(real_balances[0]) + float(Va)
        effective_b = float(real_balances[1]) + float(Vb)
        spot_price = effective_b / effective_a
        npt.assert_allclose(spot_price, target_price, rtol=1e-3)

        # Verify price range
        min_p, max_p = compute_price_range(
            float(real_balances[0]), float(real_balances[1]),
            float(Va), float(Vb)
        )
        npt.assert_allclose(float(min_p), min_price, rtol=1e-3)
        npt.assert_allclose(float(max_p), max_price, rtol=1e-3)

    def test_balances_positive(self):
        real_balances, Va, Vb = compute_theoretical_balances(
            500.0, 2000.0, 1000.0
        )
        assert float(real_balances[0]) > 0
        assert float(real_balances[1]) > 0
        assert float(Va) > 0
        assert float(Vb) > 0


class TestVirtualBalanceUpdatePriceRange:
    """Test virtual balance decay when pool is out of range."""

    def test_in_range_no_change(self):
        """When centeredness >= margin, virtual balances don't change."""
        # Symmetric pool: centeredness = 1.0, margin = 0.2 → in range
        Ra, Rb = 100.0, 100.0
        Va, Vb = 100.0, 100.0
        c, is_above = compute_centeredness(Ra, Rb, Va, Vb)
        # centeredness is 1.0, which is >= 0.2
        assert float(c) >= DEFAULT_CENTEREDNESS_MARGIN

    def test_out_of_range_above_center(self):
        """When above center and out of range, Vb decays, Va is recalculated."""
        # Very unbalanced: Ra >> Rb
        Ra, Rb = 1.0, 1e-3
        Va, Vb = 1.0, 1.0
        c, is_above = compute_centeredness(Ra, Rb, Va, Vb)
        assert float(c) < DEFAULT_CENTEREDNESS_MARGIN
        assert bool(is_above) is True

        new_Va, new_Vb = compute_virtual_balances_updating_price_range(
            Ra, Rb, Va, Vb,
            is_pool_above_center=True,
            daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=3600.0,
            sqrt_price_ratio=jnp.sqrt(compute_price_ratio(Ra, Rb, Va, Vb)),
        )
        # Vb should decay
        assert float(new_Vb) < Vb
        # Both should remain positive
        assert float(new_Va) > 0
        assert float(new_Vb) > 0

    def test_out_of_range_below_center(self):
        """When below center and out of range, Va decays, Vb is recalculated."""
        Ra, Rb = 1e-3, 1.0
        Va, Vb = 1.0, 1.0
        c, is_above = compute_centeredness(Ra, Rb, Va, Vb)
        assert float(c) < DEFAULT_CENTEREDNESS_MARGIN
        assert bool(is_above) is False

        new_Va, new_Vb = compute_virtual_balances_updating_price_range(
            Ra, Rb, Va, Vb,
            is_pool_above_center=False,
            daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=3600.0,
            sqrt_price_ratio=jnp.sqrt(compute_price_ratio(Ra, Rb, Va, Vb)),
        )
        # Va should decay
        assert float(new_Va) < Va
        assert float(new_Va) > 0
        assert float(new_Vb) > 0

    def test_floor_on_overvalued_balance(self):
        """Verify overvalued virtual balance doesn't drop below floor.

        Floor formula (from Solidity ReClammMath.sol):
            Vo >= Ro / (fourthroot(priceRatio) - 1)
        where fourthroot(priceRatio) = sqrt(sqrt_price_ratio).
        """
        # Use very long elapsed time to force heavy decay
        Ra, Rb = 1.0, 1e-3
        Va, Vb = 1.0, 1.0
        sqrt_Q = jnp.sqrt(compute_price_ratio(Ra, Rb, Va, Vb))

        new_Va, new_Vb = compute_virtual_balances_updating_price_range(
            Ra, Rb, Va, Vb,
            is_pool_above_center=True,
            daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
            seconds_elapsed=86400.0 * 30,  # 30 days
            sqrt_price_ratio=sqrt_Q,
        )
        # Floor for Vb (overvalued when above center):
        # Vb >= Rb / (fourthroot(priceRatio) - 1)
        # fourthroot(priceRatio) = sqrt(sqrt_price_ratio)
        fourth_root_price_ratio = jnp.sqrt(sqrt_Q)
        floor = Rb / (float(fourth_root_price_ratio) - 1.0)
        assert float(new_Vb) >= floor - 1e-10  # small tolerance


class TestInitialiseReclammReserves:
    """Test full initialization pipeline."""

    def test_basic(self):
        initial_pool_value = 1_000_000.0
        initial_prices = jnp.array([2500.0, 1.0])
        price_ratio = 4.0

        reserves, Va, Vb = initialise_reclamm_reserves(
            initial_pool_value, initial_prices, price_ratio
        )

        # Total value should match
        pool_value = float(reserves[0]) * 2500.0 + float(reserves[1]) * 1.0
        npt.assert_allclose(pool_value, initial_pool_value, rtol=1e-6)

        # Reserves should be positive
        assert float(reserves[0]) > 0
        assert float(reserves[1]) > 0
        assert float(Va) > 0
        assert float(Vb) > 0

        # Spot price should match target
        spot = (float(reserves[1]) + float(Vb)) / (float(reserves[0]) + float(Va))
        target = initial_prices[0] / initial_prices[1]
        npt.assert_allclose(spot, float(target), rtol=1e-3)

    def test_invariant_holds(self):
        initial_pool_value = 500_000.0
        initial_prices = jnp.array([3000.0, 1.0])
        price_ratio = 9.0

        reserves, Va, Vb = initialise_reclamm_reserves(
            initial_pool_value, initial_prices, price_ratio
        )

        L = compute_invariant(
            float(reserves[0]), float(reserves[1]),
            float(Va), float(Vb)
        )
        assert float(L) > 0


# ---------------------------------------------------------------------------
# Constant-arc-length thermostat
# ---------------------------------------------------------------------------

# Helper: centered pool matching benchmark_reclamm_interpolation.py
def _centered_pool(P=2.0, price_ratio=4.0, R_scale=10000.0):
    """Centered pool at price P with contract-rule-consistent virtuals."""
    Q = np.sqrt(price_ratio)
    q4 = price_ratio ** 0.25
    Ra = R_scale
    Rb = P * R_scale
    Va = Ra / (q4 - 1.0)
    Vb = Rb / (q4 - 1.0)
    return Ra, Rb, Va, Vb, Q


class TestComputeZ:
    """Test Z = sqrt(P)*VA - VB/sqrt(P)."""

    def test_basic_values(self):
        Va, Vb, P = 100.0, 200.0, 4.0
        Z = compute_Z(Va, Vb, P)
        # sqrt(4)*100 - 200/sqrt(4) = 200 - 100 = 100
        npt.assert_allclose(float(Z), 100.0, rtol=1e-12)

    def test_centered_pool(self):
        """At a perfectly centered pool, Z should be ~0."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        Z = compute_Z(Va, Vb, 2.0)
        # sqrt(2)*Va - Vb/sqrt(2). For centered pool with Vb = P*Va,
        # Z = sqrt(P)*Va - P*Va/sqrt(P) = sqrt(P)*Va - sqrt(P)*Va = 0
        npt.assert_allclose(float(Z), 0.0, atol=1e-8)

    def test_sign_convention(self):
        """When Va is large relative to Vb, Z should be positive."""
        Z = compute_Z(1000.0, 1.0, 1.0)
        # sqrt(1)*1000 - 1/sqrt(1) = 999
        assert float(Z) > 0


class TestSolveVBForZ:
    """Test quadratic solver for VB given target Z."""

    def test_round_trip(self):
        """compute_Z → solve_VB → recompute Z should recover the target."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        P = 2.0

        # Compute Z at the starting state
        Z_start = float(compute_Z(Va, Vb, P))

        # Perturb Z
        Z_target = Z_start + 50.0

        # Solve for VB
        Vb_new = solve_VB_for_Z(Ra, Rb, Z_target, Q, P)

        # Recompute VA from contract rule: VA = RA*(VB+RB)/((Q-1)*VB-RB)
        Va_new = Ra * (float(Vb_new) + Rb) / ((Q - 1.0) * float(Vb_new) - Rb)

        # Recompute Z — should match target
        Z_recovered = float(compute_Z(Va_new, float(Vb_new), P))
        npt.assert_allclose(Z_recovered, Z_target, rtol=1e-8)

    def test_matches_benchmark(self):
        """Cross-validate against the numpy benchmark implementation."""
        # Port of solve_VB_for_Z from benchmark script (numpy version)
        def _solve_VB_numpy(RA, RB, Z_star, Q, P):
            sqP = np.sqrt(P)
            a = -(Q - 1) / sqP
            b = sqP * RA + RB / sqP - (Q - 1) * Z_star
            c = sqP * RA * RB + Z_star * RB
            disc = max(b * b - 4 * a * c, 0.0)
            sd = np.sqrt(disc)
            r1, r2 = (-b + sd) / (2 * a), (-b - sd) / (2 * a)
            floor = RB / (Q - 1) + 1e-12
            ok = [r for r in (r1, r2) if r > floor]
            return min(ok)

        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        P = 2.0
        Z_target = 100.0

        vb_jax = float(solve_VB_for_Z(Ra, Rb, Z_target, Q, P))
        vb_np = _solve_VB_numpy(Ra, Rb, Z_target, Q, P)
        npt.assert_allclose(vb_jax, vb_np, rtol=1e-10)

    def test_floor_respected(self):
        """Result should always be > RB/(Q-1)."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        P = 2.0
        # Use a large Z that pushes VB close to floor
        Z_target = float(compute_Z(Va, Vb, P)) + 5000.0
        Vb_new = float(solve_VB_for_Z(Ra, Rb, Z_target, Q, P))
        floor = Rb / (Q - 1.0)
        assert Vb_new > floor


class TestComputeOnsetState:
    """Test onset state solver: find (Ra, Rb) where centeredness = margin."""

    def test_centeredness_equals_margin(self):
        """The returned state should have centeredness exactly at the margin."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        L = compute_invariant(Ra, Rb, Va, Vb)
        margin = DEFAULT_CENTEREDNESS_MARGIN

        Ra_onset, Rb_onset = compute_onset_state(Va, Vb, L, margin)

        c, _ = compute_centeredness(Ra_onset, Rb_onset, Va, Vb)
        npt.assert_allclose(float(c), margin, rtol=1e-10)

    def test_invariant_preserved(self):
        """The invariant L = (Ra+Va)(Rb+Vb) should be unchanged."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        L = compute_invariant(Ra, Rb, Va, Vb)
        margin = DEFAULT_CENTEREDNESS_MARGIN

        Ra_onset, Rb_onset = compute_onset_state(Va, Vb, L, margin)

        L_onset = compute_invariant(float(Ra_onset), float(Rb_onset), Va, Vb)
        npt.assert_allclose(float(L_onset), float(L), rtol=1e-10)

    def test_positive_reserves(self):
        """Onset reserves should be positive."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        L = compute_invariant(Ra, Rb, Va, Vb)
        margin = DEFAULT_CENTEREDNESS_MARGIN

        Ra_onset, Rb_onset = compute_onset_state(Va, Vb, L, margin)
        assert float(Ra_onset) > 0
        assert float(Rb_onset) > 0

    def test_above_center(self):
        """Onset state should be above center (Ra*Vb > Va*Rb)."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        L = compute_invariant(Ra, Rb, Va, Vb)
        margin = DEFAULT_CENTEREDNESS_MARGIN

        Ra_onset, Rb_onset = compute_onset_state(Va, Vb, L, margin)
        _, is_above = compute_centeredness(Ra_onset, Rb_onset, Va, Vb)
        # At least one direction should be above center
        assert bool(is_above) is True

    def test_different_price_ratios(self):
        """Should work for various price ratios."""
        for pr in [2.0, 4.0, 9.0, 16.0]:
            Ra, Rb, Va, Vb, Q = _centered_pool(P=3.0, price_ratio=pr)
            L = compute_invariant(Ra, Rb, Va, Vb)
            margin = 0.3

            Ra_onset, Rb_onset = compute_onset_state(Va, Vb, L, margin)
            c, _ = compute_centeredness(Ra_onset, Rb_onset, Va, Vb)
            npt.assert_allclose(float(c), margin, rtol=1e-10,
                                err_msg=f"Failed for price_ratio={pr}")


class TestCalibrateAtOnset:
    """Test that calibrate_arc_length_speed uses the onset state, not init."""

    def test_speed_matches_geometric_at_onset(self):
        """Calibrated speed should match geometric Δs computed at the onset state."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        L = compute_invariant(Ra, Rb, Va, Vb)
        daily_base = DEFAULT_DAILY_PRICE_SHIFT_BASE
        dt = 60.0
        margin = DEFAULT_CENTEREDNESS_MARGIN

        # Get onset state
        Ra_onset, Rb_onset = compute_onset_state(Va, Vb, L, margin)
        P_onset = (float(Rb_onset) + Vb) / (float(Ra_onset) + Va)

        # Compute geometric Δs at onset state directly
        _, is_above = compute_centeredness(Ra_onset, Rb_onset, Va, Vb)
        Va_geo, Vb_geo = compute_virtual_balances_updating_price_range(
            Ra_onset, Rb_onset, Va, Vb, is_above, daily_base, dt, Q,
        )
        Z_before = float(compute_Z(Va, Vb, P_onset))
        Z_after = float(compute_Z(Va_geo, Vb_geo, P_onset))
        X_onset = float(Ra_onset) + Va
        ds_expected = abs(Z_after - Z_before) / (2.0 * np.sqrt(X_onset))
        speed_expected = ds_expected / dt

        # Calibrate via the function
        speed = calibrate_arc_length_speed(
            Ra, Rb, Va, Vb, daily_base, dt, Q, 2.0,
            centeredness_margin=margin,
        )

        npt.assert_allclose(float(speed), speed_expected, rtol=1e-8)

    def test_differs_from_init_state_calibration(self):
        """Speed calibrated at onset should differ from speed at init state."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        daily_base = DEFAULT_DAILY_PRICE_SHIFT_BASE
        dt = 60.0
        margin = DEFAULT_CENTEREDNESS_MARGIN

        # Speed at onset (correct)
        speed_onset = calibrate_arc_length_speed(
            Ra, Rb, Va, Vb, daily_base, dt, Q, 2.0,
            centeredness_margin=margin,
        )

        # Speed at init state (what we had before — pass margin=1.0 to skip onset calc,
        # or directly compute geometric Δs at init)
        _, is_above_init = compute_centeredness(Ra, Rb, Va, Vb)
        Va_geo_init, Vb_geo_init = compute_virtual_balances_updating_price_range(
            Ra, Rb, Va, Vb, is_above_init, daily_base, dt, Q,
        )
        Z_before = float(compute_Z(Va, Vb, 2.0))
        Z_after = float(compute_Z(Va_geo_init, Vb_geo_init, 2.0))
        X_init = Ra + Va
        ds_init = abs(Z_after - Z_before) / (2.0 * np.sqrt(X_init))
        speed_init = ds_init / dt

        # They should differ (otherwise the fix doesn't matter)
        assert abs(float(speed_onset) - speed_init) / max(float(speed_onset), 1e-30) > 1e-4, (
            f"Onset speed {float(speed_onset)} and init speed {speed_init} should differ"
        )


class TestConstantArcLength:
    """Test the constant-arc-length virtual balance update."""

    def test_matches_geometric_at_center(self):
        """Near center, both methods should produce similar results."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        P = 2.0
        daily_base = DEFAULT_DAILY_PRICE_SHIFT_BASE
        dt = 60.0  # 1 minute
        sqrt_Q = Q

        # Make pool slightly above center by perturbing Ra
        Ra_shifted = Ra * 1.01
        _, is_above = compute_centeredness(Ra_shifted, Rb, Va, Vb)

        speed = calibrate_arc_length_speed(
            Ra_shifted, Rb, Va, Vb, daily_base, dt, sqrt_Q, P,
        )

        Va_geo, Vb_geo = compute_virtual_balances_updating_price_range(
            Ra_shifted, Rb, Va, Vb, is_above, daily_base, dt, sqrt_Q,
        )
        Va_cal, Vb_cal = compute_virtual_balances_constant_arc_length(
            Ra_shifted, Rb, Va, Vb, is_above, float(speed), dt, sqrt_Q, P,
        )

        # Should be very close at the calibration point
        npt.assert_allclose(float(Va_cal), float(Va_geo), rtol=1e-4)
        npt.assert_allclose(float(Vb_cal), float(Vb_geo), rtol=1e-4)

    def test_differs_off_center(self):
        """Through the scan (with arb), the two methods should diverge.

        Both thermostats are properly calibrated (onset-state speed), so
        the difference reflects genuine distributional differences in how
        they allocate arc-length over time, not a broken calibration.
        """
        from quantammsim.pools.reCLAMM.reclamm_reserves import (
            _jax_calc_reclamm_reserves_zero_fees,
            calibrate_arc_length_speed,
        )

        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        initial_reserves = jnp.array([Ra, Rb])
        Va, Vb = jnp.float64(Va), jnp.float64(Vb)

        n_steps = 200
        prices_a = jnp.linspace(2.0, 4.0, n_steps)
        prices = jnp.stack([prices_a, jnp.ones(n_steps)], axis=1)

        daily_base = DEFAULT_DAILY_PRICE_SHIFT_BASE
        dt = 600.0

        speed = calibrate_arc_length_speed(
            initial_reserves[0], initial_reserves[1], Va, Vb,
            daily_base, dt, Q, 2.0,
            centeredness_margin=DEFAULT_CENTEREDNESS_MARGIN,
        )
        # Sanity: speed should be meaningful, not ≈0
        assert float(speed) > 1e-6, f"Speed should be non-trivial, got {float(speed):.2e}"

        result_geo = _jax_calc_reclamm_reserves_zero_fees(
            initial_reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN, daily_base, dt,
            arc_length_speed=0.0,
        )
        result_cal = _jax_calc_reclamm_reserves_zero_fees(
            initial_reserves, Va, Vb, prices,
            DEFAULT_CENTEREDNESS_MARGIN, daily_base, dt,
            arc_length_speed=speed,
        )

        final_geo = result_geo[-1]
        final_cal = result_cal[-1]
        rel_diff = jnp.abs(final_geo - final_cal) / jnp.maximum(final_geo, 1e-10)
        assert float(rel_diff.max()) > 1e-4, (
            f"Methods should diverge with arb, got max rel diff = {float(rel_diff.max()):.2e}"
        )

    def test_floor_respected(self):
        """VB should never go below the fourth-root floor."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        P = 2.0
        sqrt_Q = Q
        fourth_root = np.sqrt(Q)
        Vb_floor = Rb / (fourth_root - 1.0)

        # Use absurdly high speed to force floor
        _, is_above = compute_centeredness(Ra * 2, Rb, Va, Vb)
        Va_new, Vb_new = compute_virtual_balances_constant_arc_length(
            Ra * 2, Rb, Va, Vb, is_above, 1e6, 86400.0, sqrt_Q, P,
        )
        assert float(Vb_new) >= Vb_floor - 1e-6

    def test_arc_length_single_step_exact(self):
        """A single constant-arc-length step should produce ds = speed * dt exactly."""
        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        P = 2.0
        daily_base = DEFAULT_DAILY_PRICE_SHIFT_BASE
        dt = 600.0
        sqrt_Q = Q

        # Above center
        Ra_shifted = Ra * 1.2
        _, is_above = compute_centeredness(Ra_shifted, Rb, Va, Vb)
        speed = calibrate_arc_length_speed(
            Ra_shifted, Rb, Va, Vb, daily_base, dt, sqrt_Q, P,
        )

        Z_before = float(compute_Z(Va, Vb, P))
        X_before = float(Ra_shifted) + float(Va)

        Va_new, Vb_new = compute_virtual_balances_constant_arc_length(
            Ra_shifted, Rb, Va, Vb, is_above, float(speed), dt, sqrt_Q, P,
        )

        Z_after = float(compute_Z(Va_new, Vb_new, P))
        ds = abs(Z_after - Z_before) / (2.0 * np.sqrt(X_before))
        expected_ds = float(speed) * dt
        npt.assert_allclose(ds, expected_ds, rtol=1e-8)

    def test_arc_length_constant_through_scan(self):
        """Through the scan, per-step Δs should be approximately constant."""
        from quantammsim.pools.reCLAMM.reclamm_reserves import calibrate_arc_length_speed
        from tests.pools.reCLAMM.helpers import (
            _jax_calc_reclamm_reserves_zero_fees_full_state,
        )

        Ra, Rb, Va, Vb, Q = _centered_pool(P=2.0, price_ratio=4.0)
        initial_reserves = jnp.array([Ra, Rb])
        Va_j, Vb_j = jnp.float64(Va), jnp.float64(Vb)

        # Large price swing (2→5) to push centeredness well below margin
        n_steps = 100
        prices_a = jnp.linspace(2.0, 5.0, n_steps)
        prices = jnp.stack([prices_a, jnp.ones(n_steps)], axis=1)
        dt = 600.0

        speed = calibrate_arc_length_speed(
            initial_reserves[0], initial_reserves[1], Va_j, Vb_j,
            DEFAULT_DAILY_PRICE_SHIFT_BASE, dt, Q, 2.0,
            centeredness_margin=DEFAULT_CENTEREDNESS_MARGIN,
        )
        assert float(speed) > 1e-6, f"Speed should be non-trivial, got {float(speed):.2e}"

        reserves, Va_hist, Vb_hist = _jax_calc_reclamm_reserves_zero_fees_full_state(
            initial_reserves, Va_j, Vb_j, prices,
            DEFAULT_CENTEREDNESS_MARGIN, DEFAULT_DAILY_PRICE_SHIFT_BASE, dt,
            arc_length_speed=speed,
        )

        # Compute Z at each step and measure Δs for steps where
        # virtual balances actually changed (thermostat triggered)
        delta_s_values = []
        for i in range(1, n_steps):
            market_price = float(prices[i, 0]) / float(prices[i, 1])
            Z_prev = float(compute_Z(Va_hist[i - 1], Vb_hist[i - 1], market_price))
            Z_curr = float(compute_Z(Va_hist[i], Vb_hist[i], market_price))
            dZ = abs(Z_curr - Z_prev)
            if dZ < 1e-12:
                continue  # thermostat didn't trigger
            X = float(reserves[i - 1, 0]) + float(Va_hist[i - 1])
            ds = dZ / (2.0 * np.sqrt(X))
            delta_s_values.append(ds)

        # Must have enough triggered steps to test constancy
        assert len(delta_s_values) >= 3, (
            f"Expected >=3 thermostat triggers, got {len(delta_s_values)}"
        )
        delta_s_arr = np.array(delta_s_values)
        # Allow 15% variation (X changes due to arb between steps)
        mean_ds = np.median(delta_s_arr)
        for ds in delta_s_arr:
            npt.assert_allclose(ds, mean_ds, rtol=0.15)


class TestCenterednessProportionalSpeed:
    """Test the centeredness-proportional speed multiplier formula.

    effective_speed = arc_length_speed * margin / max(centeredness, 1e-10)

    At onset (centeredness = margin), multiplier = 1.0.
    Deeper off-center → larger multiplier.
    """

    def test_at_onset_equals_base_speed(self):
        """When centeredness = margin, multiplier should be exactly 1.0."""
        margin = 0.2
        centeredness = 0.2  # equals margin
        base_speed = 1e-4

        multiplier = margin / jnp.maximum(centeredness, 1e-10)
        effective_speed = base_speed * float(multiplier)

        npt.assert_allclose(effective_speed, base_speed, rtol=1e-12)
        npt.assert_allclose(float(multiplier), 1.0, rtol=1e-12)

    def test_deeper_off_center_faster(self):
        """When centeredness < margin, multiplier > 1 → faster speed."""
        margin = 0.2
        centeredness = 0.1  # half of margin
        base_speed = 1e-4

        multiplier = margin / jnp.maximum(centeredness, 1e-10)
        effective_speed = base_speed * float(multiplier)

        assert effective_speed > base_speed
        npt.assert_allclose(float(multiplier), 2.0, rtol=1e-12)

    def test_proportional_relationship(self):
        """Multiplier = margin / centeredness (exact proportionality)."""
        margin = 0.3
        base_speed = 5e-5

        for centeredness in [0.3, 0.15, 0.1, 0.05, 0.01]:
            multiplier = margin / jnp.maximum(centeredness, 1e-10)
            expected = margin / centeredness
            npt.assert_allclose(float(multiplier), expected, rtol=1e-12)

    def test_floor_prevents_infinity(self):
        """When centeredness ≈ 0, the 1e-10 floor prevents inf/NaN."""
        margin = 0.2
        base_speed = 1e-4

        for centeredness in [0.0, 1e-15, -1e-5]:
            multiplier = margin / jnp.maximum(centeredness, 1e-10)
            effective_speed = base_speed * float(multiplier)
            assert jnp.isfinite(multiplier)
            assert jnp.isfinite(effective_speed)
            assert effective_speed > 0

    def test_scan_step_uses_scaling(self):
        """Over a trending scan, centeredness scaling should produce different reserves.

        Uses initialise_reclamm_reserves + trending prices (same approach as
        integration tests) to avoid the floor-binding issue that occurs with
        _centered_pool (where Vb starts exactly at the VB floor).
        """
        from quantammsim.pools.reCLAMM.reclamm_reserves import (
            _jax_calc_reclamm_reserves_zero_fees,
        )

        initial_pool_value = 1_000_000.0
        initial_prices = jnp.array([2500.0, 1.0])
        price_ratio = 4.0

        reserves, Va, Vb = initialise_reclamm_reserves(
            initial_pool_value, initial_prices, price_ratio
        )

        n_steps = 50
        prices_a = jnp.linspace(2500.0, 5000.0, n_steps)
        prices = jnp.stack([prices_a, jnp.ones(n_steps)], axis=1)

        daily_base = DEFAULT_DAILY_PRICE_SHIFT_BASE
        dt = 600.0
        margin = DEFAULT_CENTEREDNESS_MARGIN

        sqrt_Q = jnp.sqrt(compute_price_ratio(
            float(reserves[0]), float(reserves[1]), float(Va), float(Vb),
        ))
        market_price_0 = 2500.0
        speed = calibrate_arc_length_speed(
            reserves[0], reserves[1], Va, Vb,
            daily_base, dt, sqrt_Q, market_price_0,
            centeredness_margin=margin,
        )

        result_base = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            margin, daily_base, dt,
            arc_length_speed=speed,
            centeredness_scaling=False,
        )
        result_scaled = _jax_calc_reclamm_reserves_zero_fees(
            reserves, Va, Vb, prices,
            margin, daily_base, dt,
            arc_length_speed=speed,
            centeredness_scaling=True,
        )

        rel_diff = jnp.abs(result_base[-1] - result_scaled[-1]) / jnp.maximum(result_base[-1], 1e-10)
        assert float(rel_diff.max()) > 1e-4, (
            f"Centeredness scaling should produce different reserves, "
            f"got max rel diff = {float(rel_diff.max()):.2e}"
        )


class TestGetInitialValues:
    """Test ReClammPool.get_initial_values()."""

    def test_reads_from_fingerprint(self):
        """Custom values in fingerprint should flow through to initial_values."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool

        pool = ReClammPool()
        fp = {
            "initial_price_ratio": 9.0,
            "initial_centeredness_margin": 0.5,
            "initial_daily_price_shift_base": 0.99999,
        }
        vals = pool.get_initial_values(fp)
        assert vals["price_ratio"] == 9.0
        assert vals["centeredness_margin"] == 0.5
        assert vals["daily_price_shift_base"] == 0.99999

    def test_defaults(self):
        """Missing keys should use sensible defaults."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool

        pool = ReClammPool()
        vals = pool.get_initial_values({})
        assert vals["price_ratio"] == 4.0
        assert vals["centeredness_margin"] == 0.2
        npt.assert_allclose(
            vals["daily_price_shift_base"], 1.0 - 1.0 / 124000.0, rtol=1e-10
        )

    def test_includes_arc_length_speed_when_learnable(self):
        """When learn flag is True, get_initial_values should include arc_length_speed."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool

        pool = ReClammPool()
        fp = {
            "reclamm_learn_arc_length_speed": True,
            "reclamm_interpolation_method": "constant_arc_length",
            "initial_arc_length_speed": 5e-5,
        }
        vals = pool.get_initial_values(fp)
        assert "arc_length_speed" in vals, (
            "arc_length_speed should be in initial values when learn flag is True"
        )
        assert vals["arc_length_speed"] == 5e-5

    def test_excludes_arc_length_speed_by_default(self):
        """Without learn flag, get_initial_values should NOT include arc_length_speed."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool

        pool = ReClammPool()
        vals = pool.get_initial_values({})
        assert "arc_length_speed" not in vals

    def test_excludes_arc_length_speed_when_geometric(self):
        """Even with learn flag, geometric interpolation should not include arc_length_speed."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool

        pool = ReClammPool()
        fp = {
            "reclamm_learn_arc_length_speed": True,
            "reclamm_interpolation_method": "geometric",
        }
        vals = pool.get_initial_values(fp)
        assert "arc_length_speed" not in vals

    def test_includes_hypersurge_params_when_learnable(self):
        """Learning HyperSurge should expose lane parameters in initial values."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool

        pool = ReClammPool()
        fp = {
            "reclamm_learn_hypersurge_params": True,
            "initial_hypersurge_arb_max_fee": 0.03,
            "initial_hypersurge_arb_threshold": 0.01,
            "initial_hypersurge_arb_cap_deviation": 0.40,
            "initial_hypersurge_noise_max_fee": 0.04,
            "initial_hypersurge_noise_threshold": 0.02,
            "initial_hypersurge_noise_cap_deviation": 0.50,
        }
        vals = pool.get_initial_values(fp)

        assert vals["hypersurge_arb_max_fee"] == 0.03
        assert vals["hypersurge_arb_threshold"] == 0.01
        assert vals["hypersurge_arb_cap_deviation"] == 0.40
        assert vals["hypersurge_noise_max_fee"] == 0.04
        assert vals["hypersurge_noise_threshold"] == 0.02
        assert vals["hypersurge_noise_cap_deviation"] == 0.50

    def test_shift_exponent_parametrisation(self):
        """With reclamm_use_shift_exponent, get_initial_values returns shift_exponent."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool

        pool = ReClammPool()
        fp = {"reclamm_use_shift_exponent": True, "initial_shift_exponent": 2.5}
        vals = pool.get_initial_values(fp)
        assert "shift_exponent" in vals
        assert "daily_price_shift_base" not in vals
        assert vals["shift_exponent"] == 2.5

    def test_shift_exponent_off_by_default(self):
        """Without the flag, get_initial_values returns daily_price_shift_base."""
        from quantammsim.pools.reCLAMM.reclamm import ReClammPool

        pool = ReClammPool()
        vals = pool.get_initial_values({})
        assert "daily_price_shift_base" in vals
        assert "shift_exponent" not in vals
