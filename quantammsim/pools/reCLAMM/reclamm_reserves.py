"""Reserve calculations for reClAMM pools.

Implements the reClAMM (Rebalancing Concentrated Liquidity AMM) math and
scan-based reserve computation. The reClAMM is a 2-token constant-product
AMM with dynamic virtual reserves that track market price.

Invariant: L = (Ra + Va) * (Rb + Vb)

Ported from the Solidity implementation at
contracts/lib/ReClammMath.sol and the TypeScript reference at
test/utils/reClammMath.ts.
"""

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.lax import scan, cond
from jax.tree_util import Partial
from functools import partial

from quantammsim.pools.G3M.optimal_n_pool_arb import (
    precalc_shared_values_for_all_signatures,
    precalc_components_of_optimal_trade_across_prices,
    precalc_components_of_optimal_trade_across_prices_and_dynamic_fees,
    parallelised_optimal_trade_sifter,
)
from quantammsim.pools.G3M.G3M_trades import (
    _jax_calc_G3M_trade_from_exact_in_given_out,
)
from quantammsim.pools.noise_trades import (
    calculate_reserves_after_noise_trade,
    reclamm_tsoukalas_sqrt_noise_volume,
    reclamm_tsoukalas_log_noise_volume,
    reclamm_loglinear_noise_volume,
    reclamm_calibrated_noise_volume,
    reclamm_market_linear_noise_volume,
)

# Reference balance for initialisation (matches Solidity _INITIALIZATION_MAX_BALANCE_A)
_INITIALIZATION_MAX_BALANCE_A = 1e6

# Virtual balance decay is capped at 30 days to prevent overflow
_MAX_DECAY_DURATION_SECONDS = 30 * 86400

# Minimum real reserve kept after a clamp-to-edge arb (in USD).
# Prevents Ra or Rb reaching exactly 0, which causes NaN in the
# constant-arc-length thermostat (Va_floor → 0 → L → 0 → sqrt(0/p)).
_DUST_USD = 0.01



# ---------------------------------------------------------------------------
# Pure math functions
# ---------------------------------------------------------------------------

def compute_invariant(Ra, Rb, Va, Vb):
    """Compute constant-product invariant L = (Ra + Va) * (Rb + Vb)."""
    return (Ra + Va) * (Rb + Vb)


def compute_centeredness(Ra, Rb, Va, Vb):
    """Compute pool centeredness and whether pool is above center.

    Centeredness measures how balanced the pool is within its price range.
    Returns (centeredness, is_above_center) where centeredness ∈ [0, 1]
    and 1.0 means perfectly centered.

    Parameters
    ----------
    Ra, Rb : float
        Real balances of tokens A and B.
    Va, Vb : float
        Virtual balances of tokens A and B.

    Returns
    -------
    centeredness : float
        Value in [0, 1]. 1.0 = perfectly centered.
    is_above_center : bool
        True if Ra*Vb > Rb*Va (token A is undervalued / more abundant).
    """
    # Handle zero balances
    is_Ra_zero = Ra == 0.0
    is_Rb_zero = Rb == 0.0

    numerator = Ra * Vb
    denominator = Va * Rb

    is_above = numerator > denominator

    # centeredness = min(num, den) / max(num, den)
    centeredness = jnp.where(
        is_above,
        denominator / jnp.maximum(numerator, 1e-30),
        numerator / jnp.maximum(denominator, 1e-30),
    )

    # Zero balance edge cases
    centeredness = jnp.where(is_Ra_zero, 0.0, centeredness)
    centeredness = jnp.where(is_Rb_zero, 0.0, centeredness)

    is_above = jnp.where(is_Ra_zero, False, is_above)
    is_above = jnp.where(is_Rb_zero, True, is_above)

    # If both zero, consistent with Solidity: return (0, False)
    is_above = jnp.where(is_Ra_zero & is_Rb_zero, False, is_above)

    return centeredness, is_above


def is_above_center(Ra, Rb, Va, Vb):
    """Check if pool is above center (token A undervalued).

    Above center means Ra/Rb > Va/Vb, or equivalently Ra*Vb > Rb*Va.
    """
    _, result = compute_centeredness(Ra, Rb, Va, Vb)
    return result


def compute_price_range(Ra, Rb, Va, Vb):
    """Compute min and max prices from current state.

    minPrice = Vb² / L  (price when all real balance is in token A)
    maxPrice = L / Va²   (price when all real balance is in token B)

    Price is defined as token B per token A (how much B for 1 A).
    """
    L = compute_invariant(Ra, Rb, Va, Vb)
    min_price = (Vb * Vb) / L
    max_price = L / (Va * Va)
    return min_price, max_price


def compute_price_ratio(Ra, Rb, Va, Vb):
    """Compute price ratio = maxPrice / minPrice."""
    min_price, max_price = compute_price_range(Ra, Rb, Va, Vb)
    return max_price / min_price


def compute_out_given_in(Ra, Rb, Va, Vb, token_in, token_out, amount_in):
    """Compute output amount for a given input in constant-product swap.

    Ao = (Bo + Vo) * Ai / (Bi + Vi + Ai)

    where Bi, Vi are balance/virtual of the input token and
    Bo, Vo are balance/virtual of the output token.
    """
    balances = jnp.array([Ra, Rb])
    virtuals = jnp.array([Va, Vb])

    Bi = balances[token_in]
    Vi = virtuals[token_in]
    Bo = balances[token_out]
    Vo = virtuals[token_out]

    amount_out = (Bo + Vo) * amount_in / (Bi + Vi + amount_in)
    return amount_out


def compute_in_given_out(Ra, Rb, Va, Vb, token_in, token_out, amount_out):
    """Compute input amount required for a given output.

    Ai = (Bi + Vi) * Ao / (Bo + Vo - Ao)
    """
    balances = jnp.array([Ra, Rb])
    virtuals = jnp.array([Va, Vb])

    Bi = balances[token_in]
    Vi = virtuals[token_in]
    Bo = balances[token_out]
    Vo = virtuals[token_out]

    amount_in = (Bi + Vi) * amount_out / (Bo + Vo - amount_out)
    return amount_in


def compute_theoretical_balances(min_price, max_price, target_price):
    """Compute theoretical initial balances from price parameters.

    Ports computeTheoreticalPriceRatioAndBalances from Solidity.
    Uses a reference balance Ra_ref = _INITIALIZATION_MAX_BALANCE_A
    and derives all other balances from the price parameters.

    Parameters
    ----------
    min_price, max_price : float
        Price range bounds (B per A).
    target_price : float
        Desired initial spot price (B per A).

    Returns
    -------
    real_balances : jnp.ndarray, shape (2,)
        [Ra, Rb] reference real balances (unscaled).
    Va : float
        Virtual balance of token A.
    Vb : float
        Virtual balance of token B.
    """
    price_ratio = max_price / min_price
    sqrt_price_ratio = jnp.sqrt(price_ratio)

    Ra_ref = _INITIALIZATION_MAX_BALANCE_A

    # Va = Ra_ref / (sqrt(Q) - 1)
    Va = Ra_ref / (sqrt_price_ratio - 1.0)

    # Vb = minPrice * (Va + Ra_ref)
    Vb = min_price * (Va + Ra_ref)

    # Rb = sqrt(targetPrice * Vb * (Ra_ref + Va)) - Vb
    Rb = jnp.sqrt(target_price * Vb * (Ra_ref + Va)) - Vb

    # Ra = (Rb + Vb - Va * targetPrice) / targetPrice
    Ra = (Rb + Vb - Va * target_price) / target_price

    real_balances = jnp.array([Ra, Rb])
    return real_balances, Va, Vb


def compute_virtual_balances_updating_price_range(
    Ra, Rb, Va, Vb,
    is_pool_above_center,
    daily_price_shift_base,
    seconds_elapsed,
    sqrt_price_ratio,
):
    """Update virtual balances when pool is outside target range.

    Decays the overvalued token's virtual balance and recalculates the
    undervalued token's virtual balance to maintain the price ratio.

    Parameters
    ----------
    Ra, Rb : float
        Real balances.
    Va, Vb : float
        Current virtual balances.
    is_pool_above_center : bool
        True if pool is above center (A undervalued, B overvalued).
    daily_price_shift_base : float
        Decay base per second, typically 1 - 1/124000.
    seconds_elapsed : float
        Time since last update in seconds.
    sqrt_price_ratio : float
        Square root of the current price ratio.

    Returns
    -------
    new_Va, new_Vb : float
        Updated virtual balances.
    """
    # Cap duration at 30 days
    duration = jnp.minimum(seconds_elapsed, _MAX_DECAY_DURATION_SECONDS)

    # Decay factor: base^duration
    decay = daily_price_shift_base ** duration

    # Fourth root of price ratio = sqrt(sqrt_price_ratio).
    # Solidity: sqrtScaled18(sqrtPriceRatio) where sqrtPriceRatio = sqrt(priceRatio).
    fourth_root_price_ratio = jnp.sqrt(sqrt_price_ratio)

    # When above center: B is overvalued, decay Vb, recalculate Va
    # When below center: A is overvalued, decay Va, recalculate Vb
    def update_above_center():
        # Decay Vb (overvalued)
        Vb_decayed = Vb * decay
        # Floor: Vo >= Ro / (fourthroot(priceRatio) - 1)
        Vb_floor = Rb / jnp.maximum(fourth_root_price_ratio - 1.0, 1e-30)
        Vb_new = jnp.maximum(Vb_decayed, Vb_floor)
        # Recalculate Va: Vu = Ru * (Vo + Ro) / ((sqrt_Q - 1) * Vo - Ro)
        denominator = (sqrt_price_ratio - 1.0) * Vb_new - Rb
        Va_new = Ra * (Vb_new + Rb) / jnp.maximum(denominator, 1e-30)
        return Va_new, Vb_new

    def update_below_center():
        # Decay Va (overvalued)
        Va_decayed = Va * decay
        # Floor: Vo >= Ro / (fourthroot(priceRatio) - 1)
        Va_floor = Ra / jnp.maximum(fourth_root_price_ratio - 1.0, 1e-30)
        Va_new = jnp.maximum(Va_decayed, Va_floor)
        # Recalculate Vb: Vu = Ru * (Vo + Ro) / ((sqrt_Q - 1) * Vo - Ra)
        denominator = (sqrt_price_ratio - 1.0) * Va_new - Ra
        Vb_new = Rb * (Va_new + Ra) / jnp.maximum(denominator, 1e-30)
        return Va_new, Vb_new

    Va_above, Vb_above = update_above_center()
    Va_below, Vb_below = update_below_center()

    new_Va = jnp.where(is_pool_above_center, Va_above, Va_below)
    new_Vb = jnp.where(is_pool_above_center, Vb_above, Vb_below)

    return new_Va, new_Vb


def compute_Z(Va, Vb, market_price):
    """Compute Z = sqrt(P)*VA - VB/sqrt(P), the thermostat coordinate.

    Z measures displacement from center in a geometry-aware way. At center,
    Z ≈ 0; above center (B overvalued), Z increases as VB decays.
    """
    sqP = jnp.sqrt(market_price)
    return sqP * Va - Vb / sqP


def solve_VB_for_Z(Ra, Rb, Z_target, sqrt_price_ratio, market_price):
    """Solve for VB that achieves a target Z value.

    Substitutes the contract rule VA = RA*(VB+RB)/((Q-1)*VB - RB) into
    Z = sqrt(P)*VA - VB/sqrt(P) and solves the resulting quadratic.
    Returns the physically valid root (VB > RB/(Q-1)).

    Parameters
    ----------
    Ra, Rb : float
        Real balances.
    Z_target : float
        Desired Z value.
    sqrt_price_ratio : float
        sqrt(max_price/min_price), i.e. Q from the paper.
    market_price : float
        Current market price (token A in terms of token B).
    """
    sqP = jnp.sqrt(market_price)
    Q = sqrt_price_ratio
    a = -(Q - 1.0) / sqP
    b = sqP * Ra + Rb / sqP - (Q - 1.0) * Z_target
    c = sqP * Ra * Rb + Z_target * Rb
    disc = jnp.maximum(b * b - 4.0 * a * c, 1e-30)
    sd = jnp.sqrt(disc)
    r1 = (-b + sd) / (2.0 * a)
    r2 = (-b - sd) / (2.0 * a)
    floor = Rb / (Q - 1.0) + 1e-8
    return jnp.where(r2 > floor, r2, r1)


def compute_virtual_balances_constant_arc_length(
    Ra, Rb, Va, Vb,
    is_pool_above_center,
    arc_length_speed,
    seconds_elapsed,
    sqrt_price_ratio,
    market_price,
):
    """Update virtual balances using constant-arc-length thermostat.

    Instead of geometric VB decay (front-loaded arb loss), steps by constant
    arc-length increments in Z-space: ΔZ = 2 * speed * √X * dt. This
    equalises per-step loss Δs_k = |ΔZ_k|/(2√X_k) = const, minimising
    total loss by Cauchy-Schwarz.

    Parameters
    ----------
    Ra, Rb : float
        Real balances.
    Va, Vb : float
        Current virtual balances.
    is_pool_above_center : bool
        True if pool is above center.
    arc_length_speed : float
        Arc-length increment per second (Δs/dt).
    seconds_elapsed : float
        Time since last update.
    sqrt_price_ratio : float
        sqrt(max_price/min_price).
    market_price : float
        Current market price (A in terms of B).

    Returns
    -------
    new_Va, new_Vb : float
        Updated virtual balances.
    """
    duration = jnp.minimum(seconds_elapsed, _MAX_DECAY_DURATION_SECONDS)
    fourth_root_price_ratio = jnp.sqrt(sqrt_price_ratio)

    # Current state in Z-space
    Z = compute_Z(Va, Vb, market_price)
    X = Ra + Va

    # Constant arc-length step: ΔZ = 2 * speed * √X * dt
    delta_Z = 2.0 * arc_length_speed * jnp.sqrt(jnp.maximum(X, 1e-30)) * duration

    # --- Above center: VB decays → Z increases ---
    Z_above = Z + delta_Z
    Vb_above_raw = solve_VB_for_Z(Ra, Rb, Z_above, sqrt_price_ratio, market_price)
    Vb_floor = Rb / jnp.maximum(fourth_root_price_ratio - 1.0, 1e-30)
    Vb_above = jnp.maximum(Vb_above_raw, Vb_floor)
    Va_above = Ra * (Vb_above + Rb) / jnp.maximum(
        (sqrt_price_ratio - 1.0) * Vb_above - Rb, 1e-30
    )

    # --- Below center: VA decays → Z decreases ---
    Z_below = Z - delta_Z
    Vb_below_raw = solve_VB_for_Z(Ra, Rb, Z_below, sqrt_price_ratio, market_price)
    Va_below_raw = Ra * (Vb_below_raw + Rb) / jnp.maximum(
        (sqrt_price_ratio - 1.0) * Vb_below_raw - Rb, 1e-30
    )
    Va_floor = Ra / jnp.maximum(fourth_root_price_ratio - 1.0, 1e-30)
    need_va_floor = Va_below_raw < Va_floor
    Va_below = jnp.where(need_va_floor, Va_floor, Va_below_raw)
    Vb_below = jnp.where(
        need_va_floor,
        Rb * (Va_below + Ra) / jnp.maximum(
            (sqrt_price_ratio - 1.0) * Va_below - Ra, 1e-30
        ),
        Vb_below_raw,
    )

    new_Va = jnp.where(is_pool_above_center, Va_above, Va_below)
    new_Vb = jnp.where(is_pool_above_center, Vb_above, Vb_below)

    return new_Va, new_Vb


def compute_onset_state(Va, Vb, L, centeredness_margin):
    """Solve for the reserve state where centeredness first equals the margin.

    At onset the thermostat fires for the first time. Virtual balances are
    still at their initial values (unchanged since pool creation), but arb
    has shifted the real reserves (Ra, Rb) such that
        centeredness = min(Ra·Vb, Va·Rb) / max(Ra·Vb, Va·Rb) = margin.

    We solve the "above center" case (Ra·Vb > Va·Rb):
        Va·Rb / (Ra·Vb) = C_m  ⟹  Rb = C_m · Ra · Vb / Va

    Combined with the invariant L = (Ra+Va)(Rb+Vb) this gives a quadratic
    in Ra:
        C_m · u² + Va(1+C_m)·u + Va² − L·Va/Vb = 0

    Parameters
    ----------
    Va, Vb : float
        Virtual balances (unchanged since pool init).
    L : float
        Pool invariant (Ra+Va)(Rb+Vb), constant throughout pool life.
    centeredness_margin : float
        Centeredness threshold at which the thermostat fires.

    Returns
    -------
    Ra_onset, Rb_onset : jnp.ndarray
        Real reserves at the onset state (above-center direction).
    """
    C_m = centeredness_margin
    a = C_m
    b = Va * (1.0 + C_m)
    c = Va * Va - L * Va / jnp.maximum(Vb, 1e-30)

    disc = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    sd = jnp.sqrt(disc)

    # Positive root (Ra must be positive)
    Ra_onset = (-b + sd) / (2.0 * a)
    Rb_onset = C_m * Ra_onset * Vb / jnp.maximum(Va, 1e-30)

    return Ra_onset, Rb_onset


def calibrate_arc_length_speed(
    Ra, Rb, Va, Vb,
    daily_price_shift_base,
    seconds_per_step,
    sqrt_price_ratio,
    market_price,
    centeredness_margin=None,
):
    """Calibrate constant-arc-length speed to match geometric onset.

    Simulates one geometric decay step and measures the resulting arc-length
    increment Δs = |ΔZ| / (2√X). Returns Δs / dt as the speed.

    When centeredness_margin is provided, the geometric step is computed at
    the onset state (where centeredness first crosses the margin), which is
    the physically correct calibration point. When None, uses the passed-in
    state directly (for unit-testing the thermostat mechanics).

    Parameters
    ----------
    Ra, Rb, Va, Vb : float
        Pool state. When centeredness_margin is provided, these are used only
        to compute L; the onset state is solved analytically.
    daily_price_shift_base : float
        Geometric decay base per second.
    seconds_per_step : float
        Time between blocks.
    sqrt_price_ratio : float
        √(max_price/min_price).
    market_price : float
        Current market price (token A in terms of token B).
    centeredness_margin : float, optional
        If provided, compute the onset state and calibrate there.
    """
    if centeredness_margin is not None:
        L = (Ra + Va) * (Rb + Vb)
        Ra_cal, Rb_cal = compute_onset_state(Va, Vb, L, centeredness_margin)
        P_cal = (Rb_cal + Vb) / jnp.maximum(Ra_cal + Va, 1e-30)
    else:
        Ra_cal, Rb_cal = Ra, Rb
        P_cal = market_price

    _, is_above = compute_centeredness(Ra_cal, Rb_cal, Va, Vb)

    Va_geo, Vb_geo = compute_virtual_balances_updating_price_range(
        Ra_cal, Rb_cal, Va, Vb, is_above, daily_price_shift_base,
        seconds_per_step, sqrt_price_ratio,
    )

    Z_before = compute_Z(Va, Vb, P_cal)
    Z_after = compute_Z(Va_geo, Vb_geo, P_cal)

    X = Ra_cal + Va
    delta_s = jnp.abs(Z_after - Z_before) / (2.0 * jnp.sqrt(jnp.maximum(X, 1e-30)))
    speed = delta_s / seconds_per_step

    return speed


def initialise_reclamm_reserves(initial_pool_value, initial_prices, price_ratio):
    """Initialize reClAMM pool reserves for a given pool value and prices.

    Parameters
    ----------
    initial_pool_value : float
        Total pool value in numeraire terms.
    initial_prices : jnp.ndarray, shape (2,)
        Initial prices [price_a, price_b].
    price_ratio : float
        Desired max_price / min_price ratio.

    Returns
    -------
    reserves : jnp.ndarray, shape (2,)
        Initial real reserves [Ra, Rb].
    Va : float
        Initial virtual balance A.
    Vb : float
        Initial virtual balance B.
    """
    target_price = initial_prices[0] / initial_prices[1]
    sqrt_Q = jnp.sqrt(price_ratio)
    min_price = target_price / sqrt_Q
    max_price = target_price * sqrt_Q

    real_balances, Va, Vb = compute_theoretical_balances(
        min_price, max_price, target_price
    )

    # Scale to match desired pool value
    ref_value = real_balances[0] * initial_prices[0] + real_balances[1] * initial_prices[1]
    scale = initial_pool_value / ref_value

    reserves = real_balances * scale
    Va = Va * scale
    Vb = Vb * scale

    return reserves, Va, Vb


# ---------------------------------------------------------------------------
# Scan-based reserve calculations
# ---------------------------------------------------------------------------

def apply_target_price_ratio_to_virtual_balances(Ra, Rb, Va, Vb, target_price_ratio):
    """Retarget virtual balances to a desired price ratio while preserving centeredness.

    Uses the closed-form quadratic solution from ReClammMath.sol
    ``computeVirtualBalancesUpdatingPriceRatio``:

        Vu = Ru * (1 + C + sqrt(1 + C*(C + 4*Q0 - 2))) / (2*(Q0 - 1))
        Vo = Vu * lastVo / lastVu

    where Q0 = sqrt(price_ratio), C = centeredness, Ru is the real balance of
    the undervalued token.  The overvalued virtual balance is then scaled
    proportionally so that Va/Vb is preserved, which keeps centeredness constant.
    """
    safe_ratio = jnp.maximum(target_price_ratio, 1.0 + 1e-12)
    Q0 = jnp.sqrt(safe_ratio)  # sqrt(price_ratio)
    centeredness, is_above = compute_centeredness(Ra, Rb, Va, Vb)
    C = centeredness

    # Closed-form quadratic solution for the undervalued virtual balance.
    discriminant = jnp.maximum(1.0 + C * (C + 4.0 * Q0 - 2.0), 0.0)
    numerator_factor = 1.0 + C + jnp.sqrt(discriminant)
    denominator = 2.0 * jnp.maximum(Q0 - 1.0, 1e-30)

    # Above center: A is undervalued (Ra abundant), B is overvalued.
    Vu_above = Ra * numerator_factor / denominator  # new Va
    Vo_above = Vu_above * Vb / jnp.maximum(Va, 1e-30)  # new Vb, scaled

    # Below center: B is undervalued (Rb abundant), A is overvalued.
    Vu_below = Rb * numerator_factor / denominator  # new Vb
    Vo_below = Vu_below * Va / jnp.maximum(Vb, 1e-30)  # new Va, scaled

    Va_new = jnp.where(is_above, Vu_above, Vo_below)
    Vb_new = jnp.where(is_above, Vo_above, Vu_below)

    # When centeredness is degenerate (e.g. both sides zero), preserve current virtuals.
    invalid_centeredness = ~jnp.isfinite(centeredness)
    Va_new = jnp.where(invalid_centeredness, Va, Va_new)
    Vb_new = jnp.where(invalid_centeredness, Vb, Vb_new)
    return Va_new, Vb_new

def _reclamm_scan_step_zero_fees(
    carry_list,
    input_list,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
):
    """Single scan step for zero-fee reClAMM pool.

    Zero-fee means no trading fees, but the pool still needs to:
    1. Update virtual balances (path-dependent)
    2. Compute analytical constant-product arb (no fee friction)

    Carry: [real_reserves (2,), Va (0-d), Vb (0-d), prev_lp_supply (0-d)]
    Input: [prices (2,), lp_supply (0-d)]
    """
    prev_reserves = carry_list[0]
    Va = carry_list[1]
    Vb = carry_list[2]
    prev_lp_supply = carry_list[3]

    prices = input_list[0]
    lp_supply = input_list[1]

    # Scale both real and virtual reserves by LP supply ratio.
    scale = lp_supply / prev_lp_supply
    lp_supply_change = lp_supply != prev_lp_supply
    prev_reserves = jnp.where(lp_supply_change, prev_reserves * scale, prev_reserves)
    Va = jnp.where(lp_supply_change, Va * scale, Va)
    Vb = jnp.where(lp_supply_change, Vb * scale, Vb)

    Ra = prev_reserves[0]
    Rb = prev_reserves[1]

    # Step 1: Update virtual balances if out of range
    centeredness, is_above = compute_centeredness(Ra, Rb, Va, Vb)
    sqrt_Q = jnp.sqrt(compute_price_ratio(Ra, Rb, Va, Vb))
    out_of_range = centeredness < centeredness_margin
    market_price = prices[0] / prices[1]

    # Centeredness-proportional scaling: margin/centeredness multiplier
    # Applies to both geometric (via seconds_elapsed) and arc-length (via speed)
    speed_multiplier = jnp.where(
        centeredness_scaling,
        centeredness_margin / jnp.maximum(centeredness, 1e-10),
        1.0,
    )

    Va_geo, Vb_geo = compute_virtual_balances_updating_price_range(
        Ra, Rb, Va, Vb,
        is_pool_above_center=is_above,
        daily_price_shift_base=daily_price_shift_base,
        seconds_elapsed=seconds_per_step * speed_multiplier,
        sqrt_price_ratio=sqrt_Q,
    )

    Va_cal, Vb_cal = compute_virtual_balances_constant_arc_length(
        Ra, Rb, Va, Vb,
        is_pool_above_center=is_above,
        arc_length_speed=arc_length_speed * speed_multiplier,
        seconds_elapsed=seconds_per_step,
        sqrt_price_ratio=sqrt_Q,
        market_price=market_price,
    )
    use_cal = arc_length_speed > 0.0
    Va_updated = jnp.where(use_cal, Va_cal, Va_geo)
    Vb_updated = jnp.where(use_cal, Vb_cal, Vb_geo)

    Va = jnp.where(out_of_range, Va_updated, Va)
    Vb = jnp.where(out_of_range, Vb_updated, Vb)

    # Step 2: Analytical zero-fee arb on effective reserves
    L = compute_invariant(Ra, Rb, Va, Vb)

    Ea_new = jnp.sqrt(L / market_price)
    Eb_new = jnp.sqrt(L * market_price)

    Ra_new = Ea_new - Va
    Rb_new = Eb_new - Vb

    # Clamp-to-edge: if a real reserve would go negative, apply an
    # exact-in-given-out edge trade that drains that token to _DUST_USD
    # worth of reserves (preserving the AMM invariant).
    dust_a = _DUST_USD / prices[0]
    dust_b = _DUST_USD / prices[1]
    drain_a = jnp.maximum(Ra - dust_a, 0.0)
    drain_b = jnp.maximum(Rb - dust_b, 0.0)

    effective = jnp.array([Ra + Va, Rb + Vb])
    _weights = jnp.array([0.5, 0.5])

    edge_a = _jax_calc_G3M_trade_from_exact_in_given_out(
        effective, _weights, token_in=1, token_out=0, amount_out=drain_a, gamma=1.0,
    )
    edge_b = _jax_calc_G3M_trade_from_exact_in_given_out(
        effective, _weights, token_in=0, token_out=1, amount_out=drain_b, gamma=1.0,
    )

    clamp_a = Ra_new < 0
    clamp_b = Rb_new < 0
    Ra_new = jnp.where(clamp_a, Ra + edge_a[0], jnp.where(clamp_b, Ra + edge_b[0], Ra_new))
    Rb_new = jnp.where(clamp_a, Rb + edge_a[1], jnp.where(clamp_b, Rb + edge_b[1], Rb_new))

    new_reserves = jnp.array([Ra_new, Rb_new])
    return [new_reserves, Va, Vb, lp_supply], new_reserves


# ---------------------------------------------------------------------------
# Test-only diagnostic helpers (virtual-balance history)
# ---------------------------------------------------------------------------
# These helpers mirror production kernels but additionally return Va/Vb
# trajectories for assertions in tests. Production pool paths should use the
# reserve-only kernels above.

def _reclamm_scan_step_zero_fees_full_state(
    carry_list,
    input_list,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
):
    """TEST-ONLY: scan step that outputs (reserves, Va, Vb)."""
    new_carry, new_reserves = _reclamm_scan_step_zero_fees(
        carry_list, input_list, centeredness_margin, daily_price_shift_base, seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
    )
    return new_carry, (new_reserves, new_carry[1], new_carry[2])


def _reclamm_scan_step_with_fees_and_revenue(
    carry_list,
    input_list,
    weights,
    tokens_to_drop,
    active_trade_directions,
    n,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    noise_model="ratio",
    noise_params=None,
):
    """Single scan step for reClAMM pool with fees, returning LP fee revenue.

    Primary implementation — ``_reclamm_scan_step_with_fees`` wraps this.

    Carry: [real_reserves (2,), Va, Vb, prev_lp_supply, step_idx, active_start_ratio,
            active_target_ratio, active_start_step, active_end_step, active_enabled]
    Input: [prices, active_initial_weights, per_asset_ratios,
            all_other_assets_ratios, gamma, arb_thresh, arb_fees, price_ratio_update,
            lp_supply, (optional) volatility]

    Returns
    -------
    new_carry : list
    (new_reserves, lp_fee_revenue_usd) : tuple
        ``lp_fee_revenue_usd`` is a scalar: USD value of LP fee income this step.
    """
    prev_reserves = carry_list[0]
    Va = carry_list[1]
    Vb = carry_list[2]
    prev_lp_supply = carry_list[3]
    step_idx = carry_list[4]
    active_start_ratio = carry_list[5]
    active_target_ratio = carry_list[6]
    active_start_step = carry_list[7]
    active_end_step = carry_list[8]
    active_enabled = carry_list[9]

    prices = input_list[0]
    active_initial_weights = input_list[1]
    per_asset_ratios = input_list[2]
    all_other_assets_ratios = input_list[3]
    gamma = input_list[4]
    arb_thresh = input_list[5]
    arb_fees = input_list[6]
    price_ratio_update = input_list[7]
    lp_supply = input_list[8]

    # Scale both real and virtual reserves by LP supply ratio.
    # Matches ReClammPool.sol onBeforeAddLiquidity / onBeforeRemoveLiquidity:
    # all balances (real + virtual) scale proportionally with BPT supply.
    scale = lp_supply / prev_lp_supply
    lp_supply_change = lp_supply != prev_lp_supply
    prev_reserves = jnp.where(lp_supply_change, prev_reserves * scale, prev_reserves)
    Va = jnp.where(lp_supply_change, Va * scale, Va)
    Vb = jnp.where(lp_supply_change, Vb * scale, Vb)

    Ra = prev_reserves[0]
    Rb = prev_reserves[1]

    # Price-ratio schedule: apply target price ratio changes over time.
    event_has = price_ratio_update[0] > 0.5
    event_target_ratio = jnp.maximum(
        jnp.where(jnp.isfinite(price_ratio_update[1]), price_ratio_update[1], 1.0),
        1.0 + 1e-12,
    )
    event_end_step = jnp.where(
        jnp.isfinite(price_ratio_update[2]), price_ratio_update[2], step_idx
    )
    event_start_override = price_ratio_update[3]

    def _apply_schedule_state(_):
        current_price_ratio = compute_price_ratio(Ra, Rb, Va, Vb)
        start_ratio_from_event = jnp.where(
            jnp.isfinite(event_start_override),
            event_start_override,
            current_price_ratio,
        )
        next_active_start_ratio = jnp.where(
            event_has, start_ratio_from_event, active_start_ratio
        )
        next_active_target_ratio = jnp.where(
            event_has, event_target_ratio, active_target_ratio
        )
        next_active_start_step = jnp.where(event_has, step_idx, active_start_step)
        next_active_end_step = jnp.where(
            event_has, jnp.maximum(event_end_step, step_idx), active_end_step
        )
        next_active_enabled = jnp.where(event_has, True, active_enabled)
        next_active_enabled = jnp.logical_and(
            next_active_enabled, step_idx <= next_active_end_step
        )

        schedule_duration = next_active_end_step - next_active_start_step
        schedule_progress = jnp.where(
            schedule_duration <= 0.0,
            1.0,
            jnp.clip((step_idx - next_active_start_step) / schedule_duration, 0.0, 1.0),
        )
        safe_start_ratio = jnp.maximum(next_active_start_ratio, 1.0 + 1e-12)
        safe_target_ratio = jnp.maximum(next_active_target_ratio, 1.0 + 1e-12)
        scheduled_price_ratio = safe_start_ratio * (
            safe_target_ratio / safe_start_ratio
        ) ** schedule_progress
        scheduled_price_ratio = jnp.where(
            next_active_enabled, scheduled_price_ratio, current_price_ratio
        )
        Va_scheduled, Vb_scheduled = apply_target_price_ratio_to_virtual_balances(
            Ra, Rb, Va, Vb, scheduled_price_ratio
        )
        next_Va = jnp.where(next_active_enabled, Va_scheduled, Va)
        next_Vb = jnp.where(next_active_enabled, Vb_scheduled, Vb)
        return (
            next_Va,
            next_Vb,
            next_active_start_ratio,
            next_active_target_ratio,
            next_active_start_step,
            next_active_end_step,
            next_active_enabled,
        )

    def _skip_schedule_state(_):
        retained_active_enabled = jnp.logical_and(
            active_enabled, step_idx <= active_end_step
        )
        return (
            Va,
            Vb,
            active_start_ratio,
            active_target_ratio,
            active_start_step,
            active_end_step,
            retained_active_enabled,
        )

    active_not_expired = jnp.logical_and(active_enabled, step_idx <= active_end_step)
    schedule_active = jnp.logical_or(event_has, active_not_expired)
    (
        Va,
        Vb,
        active_start_ratio,
        active_target_ratio,
        active_start_step,
        active_end_step,
        active_enabled,
    ) = cond(
        schedule_active,
        _apply_schedule_state,
        _skip_schedule_state,
        operand=None,
    )

    # Step 1: Update virtual balances if out of range
    centeredness, is_above = compute_centeredness(Ra, Rb, Va, Vb)
    sqrt_Q = jnp.sqrt(compute_price_ratio(Ra, Rb, Va, Vb))
    out_of_range = centeredness < centeredness_margin
    market_price = prices[0] / prices[1]

    # Centeredness-proportional scaling: margin/centeredness multiplier
    speed_multiplier_fees = jnp.where(
        centeredness_scaling,
        centeredness_margin / jnp.maximum(centeredness, 1e-10),
        1.0,
    )

    Va_geo, Vb_geo = compute_virtual_balances_updating_price_range(
        Ra, Rb, Va, Vb,
        is_pool_above_center=is_above,
        daily_price_shift_base=daily_price_shift_base,
        seconds_elapsed=seconds_per_step * speed_multiplier_fees,
        sqrt_price_ratio=sqrt_Q,
    )

    Va_cal, Vb_cal = compute_virtual_balances_constant_arc_length(
        Ra, Rb, Va, Vb,
        is_pool_above_center=is_above,
        arc_length_speed=arc_length_speed * speed_multiplier_fees,
        seconds_elapsed=seconds_per_step,
        sqrt_price_ratio=sqrt_Q,
        market_price=market_price,
    )
    use_cal = arc_length_speed > 0.0
    Va_updated = jnp.where(use_cal, Va_cal, Va_geo)
    Vb_updated = jnp.where(use_cal, Vb_cal, Vb_geo)

    Va = jnp.where(out_of_range, Va_updated, Va)
    Vb = jnp.where(out_of_range, Vb_updated, Vb)

    # Step 2: Compute arb trade using G3M machinery on effective reserves
    effective_reserves = jnp.array([Ra + Va, Rb + Vb])

    fees_are_being_charged = gamma != 1.0

    # Zero-fee analytical arb
    L = compute_invariant(Ra, Rb, Va, Vb)
    market_price = prices[0] / prices[1]
    Ea_new = jnp.sqrt(L / market_price)
    Eb_new = jnp.sqrt(L * market_price)
    zero_fee_trade = jnp.array([Ea_new - (Ra + Va), Eb_new - (Rb + Vb)])

    # Fee-based arb using G3M optimal trade sifter on effective reserves
    fee_trade = parallelised_optimal_trade_sifter(
        effective_reserves,
        weights,
        prices,
        active_initial_weights,
        active_trade_directions,
        per_asset_ratios,
        all_other_assets_ratios,
        tokens_to_drop,
        gamma,
        n,
        0,
    )

    optimal_arb_trade = jnp.where(fees_are_being_charged, fee_trade, zero_fee_trade)

    # Check profitability for arb
    profit_to_arb = -(optimal_arb_trade * prices).sum() - arb_thresh
    arb_external_cost = 0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
    do_trade = profit_to_arb >= arb_external_cost

    # Apply trade to REAL reserves only
    applied_trade = jnp.where(do_trade, optimal_arb_trade, 0.0)
    Ra_new = Ra + applied_trade[0]
    Rb_new = Rb + applied_trade[1]

    # --- Noise model dispatch ---
    # noise_model is a concrete Python string (passed via Partial as static
    # aux_data), so if/elif branches resolve at trace time.
    if noise_model == "ratio":
        noisy_reserves = calculate_reserves_after_noise_trade(
            applied_trade, jnp.array([Ra_new, Rb_new]), prices,
            noise_trader_ratio, gamma,
        )
        Ra_new = jnp.where(noise_trader_ratio > 0, noisy_reserves[0], Ra_new)
        Rb_new = jnp.where(noise_trader_ratio > 0, noisy_reserves[1], Rb_new)
    elif noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        volatility = input_list[9]
        arb_volume = 0.5 * jnp.sum(jnp.abs(applied_trade) * prices)
        real_value = jnp.sum(jnp.array([Ra_new, Rb_new]) * prices)
        effective_value = (Ra_new + Va) * prices[0] + (Rb_new + Vb) * prices[1]

        _np = noise_params if noise_params is not None else {}
        if noise_model == "tsoukalas_sqrt":
            noise_vol = reclamm_tsoukalas_sqrt_noise_volume(
                effective_value, gamma, volatility,
                arb_volume, _np,
            )
        elif noise_model == "tsoukalas_log":
            noise_vol = reclamm_tsoukalas_log_noise_volume(
                effective_value, gamma, volatility,
                arb_volume, _np,
            )
        else:  # loglinear
            noise_vol = reclamm_loglinear_noise_volume(
                effective_value, gamma, volatility,
                arb_volume, _np,
            )

        # Scale effective reserves uniformly to preserve quoted price.
        # For a 2-CLP: price ∝ (Ra+Va)/(Rb+Vb), so we must scale
        # effective reserves (Ra+Va, Rb+Vb) by the same factor, then
        # subtract back the fixed virtual reserves.
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_income = (1.0 - gamma) * noise_vol * minutes_per_step
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        Ra_new = (Ra_new + Va) * scale - Va
        Rb_new = (Rb_new + Vb) * scale - Vb
    elif noise_model == "calibrated":
        volatility = input_list[9]
        dow_sin = input_list[10]
        dow_cos = input_list[11]
        arb_volume = 0.5 * jnp.sum(jnp.abs(applied_trade) * prices)
        effective_value = (Ra_new + Va) * prices[0] + (Rb_new + Vb) * prices[1]

        _np = noise_params if noise_params is not None else {}
        noise_vol = reclamm_calibrated_noise_volume(
            effective_value, gamma, volatility,
            arb_volume, dow_sin, dow_cos, _np,
        )

        minutes_per_step = seconds_per_step / 60.0
        noise_fee_income = (1.0 - gamma) * noise_vol * minutes_per_step
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        Ra_new = (Ra_new + Va) * scale - Va
        Rb_new = (Rb_new + Vb) * scale - Vb
    elif noise_model == "market_linear":
        noise_base = input_list[9]
        noise_tvl_coeff = input_list[10]
        effective_value = (Ra_new + Va) * prices[0] + (Rb_new + Vb) * prices[1]

        _np = noise_params if noise_params is not None else {}
        noise_vol = reclamm_market_linear_noise_volume(
            effective_value, noise_base, noise_tvl_coeff,
            tvl_mean=_np.get("tvl_mean", 0.0),
            tvl_std=_np.get("tvl_std", 1.0),
        )

        minutes_per_step = seconds_per_step / 60.0
        noise_fee_income = (1.0 - gamma) * noise_vol * minutes_per_step
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        Ra_new = (Ra_new + Va) * scale - Va
        Rb_new = (Rb_new + Vb) * scale - Vb
    # else: "arb_only" — no noise trades

    # Clamp-to-edge: if a real reserve would go negative, apply an
    # exact-in-given-out edge trade that drains that token to _DUST_USD
    # worth of reserves (preserving the AMM invariant).
    dust_a = _DUST_USD / prices[0]
    dust_b = _DUST_USD / prices[1]
    drain_a = jnp.maximum(Ra - dust_a, 0.0)
    drain_b = jnp.maximum(Rb - dust_b, 0.0)

    _weights = jnp.array([0.5, 0.5])

    edge_a = _jax_calc_G3M_trade_from_exact_in_given_out(
        effective_reserves, _weights, token_in=1, token_out=0,
        amount_out=drain_a, gamma=gamma,
    )
    edge_b = _jax_calc_G3M_trade_from_exact_in_given_out(
        effective_reserves, _weights, token_in=0, token_out=1,
        amount_out=drain_b, gamma=gamma,
    )

    clamp_a = Ra_new < 0
    clamp_b = Rb_new < 0
    Ra_new = jnp.where(clamp_a, Ra + edge_a[0], jnp.where(clamp_b, Ra + edge_b[0], Ra_new))
    Rb_new = jnp.where(clamp_a, Rb + edge_a[1], jnp.where(clamp_b, Rb + edge_b[1], Rb_new))

    # Protocol fee: divert protocol_fee_split of inbound swap fees from LP reserves.
    # Computed on the final trade (normal arb or edge trade).
    final_trade = jnp.array([Ra_new - Ra, Rb_new - Rb])
    fee_rate = 1.0 - gamma
    inbound = jnp.maximum(final_trade, 0.0)
    protocol_fee = inbound * fee_rate * protocol_fee_split
    Ra_new = Ra_new - protocol_fee[0]
    Rb_new = Rb_new - protocol_fee[1]

    # LP fee revenue: total fee income minus protocol's share, in USD.
    lp_fee_income = inbound * fee_rate * (1.0 - protocol_fee_split)
    lp_fee_revenue_usd = (lp_fee_income * prices).sum()

    new_reserves = jnp.array([Ra_new, Rb_new])
    return [
        new_reserves,
        Va,
        Vb,
        lp_supply,
        step_idx + 1.0,
        active_start_ratio,
        active_target_ratio,
        active_start_step,
        active_end_step,
        active_enabled,
    ], (new_reserves, lp_fee_revenue_usd)


def _reclamm_scan_step_with_fees(
    carry_list,
    input_list,
    weights,
    tokens_to_drop,
    active_trade_directions,
    n,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    noise_model="ratio",
    noise_params=None,
):
    """Single scan step for reClAMM pool with fees (reserves only).

    Thin wrapper around ``_reclamm_scan_step_with_fees_and_revenue`` that
    discards the fee revenue output. JIT dead-code-eliminates the unused value.
    """
    new_carry, (new_reserves, _fee_rev) = _reclamm_scan_step_with_fees_and_revenue(
        carry_list, input_list,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        noise_params=noise_params,
    )
    return new_carry, new_reserves


def _reclamm_scan_step_with_fees_full_state(
    carry_list,
    input_list,
    weights,
    tokens_to_drop,
    active_trade_directions,
    n,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
):
    """TEST-ONLY: fee scan step that also outputs virtual balances."""
    new_carry, (new_reserves, _fee_rev) = _reclamm_scan_step_with_fees_and_revenue(
        carry_list, input_list,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
    )
    return new_carry, (new_reserves, new_carry[1], new_carry[2])


@jit
def _jax_calc_reclamm_reserves_zero_fees(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    lp_supply_array=None,
):
    """Calculate reClAMM reserves over time with zero fees.

    Parameters
    ----------
    initial_reserves : jnp.ndarray, shape (2,)
        Initial real reserves [Ra, Rb].
    initial_Va, initial_Vb : float
        Initial virtual balances.
    prices : jnp.ndarray, shape (T, 2)
        Asset prices over time.
    centeredness_margin : float
        Threshold for triggering virtual balance updates.
    daily_price_shift_base : float
        Decay base for virtual balance updates.
    seconds_per_step : float
        Time between price observations in seconds.
    arc_length_speed : float
        If > 0, use constant-arc-length thermostat instead of geometric.
    centeredness_scaling : bool
        If True, scale speed by margin/centeredness (proportional controller).
    lp_supply_array : jnp.ndarray, optional
        LP token supply over time, shape (T,). Defaults to constant 1.0.

    Returns
    -------
    reserves : jnp.ndarray, shape (T, 2)
        Real reserves over time.
    """
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    scan_fn = Partial(
        _reclamm_scan_step_zero_fees,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
    )

    carry_init = [initial_reserves, initial_Va, initial_Vb, lp_supply_array[0]]
    _, reserves = scan(scan_fn, carry_init, [prices, lp_supply_array])
    return reserves


@jit
def _jax_calc_reclamm_reserves_zero_fees_full_state(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    lp_supply_array=None,
):
    """TEST-ONLY: Like _jax_calc_reclamm_reserves_zero_fees but returns Va/Vb.

    Returns
    -------
    reserves : jnp.ndarray, shape (T, 2)
    Va_history : jnp.ndarray, shape (T,)
    Vb_history : jnp.ndarray, shape (T,)
    """
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    scan_fn = Partial(
        _reclamm_scan_step_zero_fees_full_state,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
    )

    carry_init = [initial_reserves, initial_Va, initial_Vb, lp_supply_array[0]]
    _, (reserves, Va_history, Vb_history) = scan(scan_fn, carry_init, [prices, lp_supply_array])
    return reserves, Va_history, Vb_history


@partial(jit, static_argnames=('noise_model',))
def _jax_calc_reclamm_reserves_with_fees(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    lp_supply_array=None,
    noise_model="ratio",
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
):
    """Calculate reClAMM reserves over time with fees.

    Uses the G3M optimal arb machinery with constant weights [0.5, 0.5]
    applied to effective reserves (real + virtual).
    """
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    n_assets = 2
    weights = jnp.array([0.5, 0.5])
    gamma = 1.0 - fees

    # Precalculate shared values for arb
    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_prices(
            weights, prices, gamma, tokens_to_drop,
            active_trade_directions, leave_one_out_idxs,
        )
    )

    gamma_array = jnp.full(prices.shape[0], gamma)
    arb_thresh_array = jnp.full(prices.shape[0], arb_thresh)
    arb_fees_array = jnp.full(prices.shape[0], arb_fees)
    price_ratio_updates = jnp.zeros((prices.shape[0], 4), dtype=prices.dtype)
    price_ratio_updates = price_ratio_updates.at[:, 3].set(jnp.nan)

    scan_fn = Partial(
        _reclamm_scan_step_with_fees,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        noise_params=noise_params if noise_params is not None else {},
    )

    scan_inputs = [prices, active_initial_weights, per_asset_ratios,
                   all_other_assets_ratios, gamma_array, arb_thresh_array, arb_fees_array,
                   price_ratio_updates, lp_supply_array]
    if noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        scan_inputs.append(volatility_array)
    elif noise_model == "calibrated":
        scan_inputs.append(volatility_array)
        scan_inputs.append(dow_sin_array)
        scan_inputs.append(dow_cos_array)
    elif noise_model == "market_linear":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(noise_tvl_coeff_array)

    carry_init = [
        initial_reserves,
        initial_Va,
        initial_Vb,
        lp_supply_array[0],
        jnp.float64(0.0),  # step_idx
        jnp.float64(0.0),  # active_start_ratio
        jnp.float64(0.0),  # active_target_ratio
        jnp.float64(0.0),  # active_start_step
        jnp.float64(0.0),  # active_end_step
        jnp.array(False),  # active_enabled
    ]
    _, reserves = scan(scan_fn, carry_init, scan_inputs)
    return reserves


@partial(jit, static_argnums=(11,), static_argnames=('noise_model',))
def _jax_calc_reclamm_reserves_with_dynamic_inputs(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees,
    arb_thresh,
    arb_fees,
    price_ratio_updates=None,
    do_trades=False,
    trades=None,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    lp_supply_array=None,
    noise_model="ratio",
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
):
    """Calculate reClAMM reserves with time-varying fees/arb arrays."""
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    n_assets = 2
    weights = jnp.array([0.5, 0.5])

    # Handle scalar vs array fees
    gamma = jnp.where(fees.size == 1, jnp.full(prices.shape[0], 1.0 - fees), 1.0 - fees)
    arb_thresh = jnp.where(
        arb_thresh.size == 1, jnp.full(prices.shape[0], arb_thresh), arb_thresh
    )
    arb_fees = jnp.where(
        arb_fees.size == 1, jnp.full(prices.shape[0], arb_fees), arb_fees
    )
    if price_ratio_updates is None:
        price_ratio_updates = jnp.zeros((prices.shape[0], 4), dtype=prices.dtype)
        price_ratio_updates = price_ratio_updates.at[:, 3].set(jnp.nan)
    else:
        if price_ratio_updates.ndim == 1:
            price_ratio_updates = jnp.broadcast_to(
                price_ratio_updates, (prices.shape[0], price_ratio_updates.shape[0])
            )
        elif price_ratio_updates.shape[0] == 1 and prices.shape[0] != 1:
            price_ratio_updates = jnp.broadcast_to(
                price_ratio_updates, (prices.shape[0], price_ratio_updates.shape[1])
            )

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_prices_and_dynamic_fees(
            weights, prices, gamma, tokens_to_drop,
            active_trade_directions, leave_one_out_idxs,
        )
    )

    scan_fn = Partial(
        _reclamm_scan_step_with_fees,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        noise_params=noise_params if noise_params is not None else {},
    )

    scan_inputs = [prices, active_initial_weights, per_asset_ratios,
                   all_other_assets_ratios, gamma, arb_thresh, arb_fees,
                   price_ratio_updates, lp_supply_array]
    if noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        scan_inputs.append(volatility_array)
    elif noise_model == "calibrated":
        scan_inputs.append(volatility_array)
        scan_inputs.append(dow_sin_array)
        scan_inputs.append(dow_cos_array)
    elif noise_model == "market_linear":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(noise_tvl_coeff_array)

    carry_init = [
        initial_reserves,
        initial_Va,
        initial_Vb,
        lp_supply_array[0],
        jnp.float64(0.0),  # step_idx
        jnp.float64(0.0),  # active_start_ratio
        jnp.float64(0.0),  # active_target_ratio
        jnp.float64(0.0),  # active_start_step
        jnp.float64(0.0),  # active_end_step
        jnp.array(False),  # active_enabled
    ]
    _, reserves = scan(scan_fn, carry_init, scan_inputs)
    return reserves


@partial(jit, static_argnums=(11,))
def _jax_calc_reclamm_reserves_with_dynamic_inputs_full_state(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees,
    arb_thresh,
    arb_fees,
    price_ratio_updates=None,
    do_trades=False,
    trades=None,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
):
    """TEST-ONLY: dynamic-input reserve path returning virtual-balance history."""
    n_assets = 2
    weights = jnp.array([0.5, 0.5])

    gamma = jnp.where(fees.size == 1, jnp.full(prices.shape[0], 1.0 - fees), 1.0 - fees)
    arb_thresh = jnp.where(
        arb_thresh.size == 1, jnp.full(prices.shape[0], arb_thresh), arb_thresh
    )
    arb_fees = jnp.where(
        arb_fees.size == 1, jnp.full(prices.shape[0], arb_fees), arb_fees
    )
    if price_ratio_updates is None:
        price_ratio_updates = jnp.zeros((prices.shape[0], 4), dtype=prices.dtype)
        price_ratio_updates = price_ratio_updates.at[:, 3].set(jnp.nan)
    else:
        if price_ratio_updates.ndim == 1:
            price_ratio_updates = jnp.broadcast_to(
                price_ratio_updates, (prices.shape[0], price_ratio_updates.shape[0])
            )
        elif price_ratio_updates.shape[0] == 1 and prices.shape[0] != 1:
            price_ratio_updates = jnp.broadcast_to(
                price_ratio_updates, (prices.shape[0], price_ratio_updates.shape[1])
            )

    lp_supply_array = jnp.ones(prices.shape[0], dtype=prices.dtype)

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_prices_and_dynamic_fees(
            weights, prices, gamma, tokens_to_drop,
            active_trade_directions, leave_one_out_idxs,
        )
    )

    scan_fn = Partial(
        _reclamm_scan_step_with_fees_full_state,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
    )

    carry_init = [
        initial_reserves,
        initial_Va,
        initial_Vb,
        lp_supply_array[0],
        jnp.float64(0.0),  # step_idx
        jnp.float64(0.0),  # active_start_ratio
        jnp.float64(0.0),  # active_target_ratio
        jnp.float64(0.0),  # active_start_step
        jnp.float64(0.0),  # active_end_step
        jnp.array(False),  # active_enabled
    ]
    _, (reserves, Va_history, Vb_history) = scan(
        scan_fn,
        carry_init,
        [prices, active_initial_weights, per_asset_ratios,
         all_other_assets_ratios, gamma, arb_thresh, arb_fees,
         price_ratio_updates, lp_supply_array],
    )
    return reserves, Va_history, Vb_history


@partial(jit, static_argnames=('noise_model',))
def _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    lp_supply_array=None,
    noise_model="ratio",
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
):
    """Calculate reClAMM reserves and LP fee revenue over time with fees.

    Returns
    -------
    reserves : jnp.ndarray, shape (T, 2)
    fee_revenue : jnp.ndarray, shape (T,)
        LP fee revenue per timestep in USD.
    """
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    n_assets = 2
    weights = jnp.array([0.5, 0.5])
    gamma = 1.0 - fees

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_prices(
            weights, prices, gamma, tokens_to_drop,
            active_trade_directions, leave_one_out_idxs,
        )
    )

    gamma_array = jnp.full(prices.shape[0], gamma)
    arb_thresh_array = jnp.full(prices.shape[0], arb_thresh)
    arb_fees_array = jnp.full(prices.shape[0], arb_fees)
    price_ratio_updates = jnp.zeros((prices.shape[0], 4), dtype=prices.dtype)
    price_ratio_updates = price_ratio_updates.at[:, 3].set(jnp.nan)

    scan_fn = Partial(
        _reclamm_scan_step_with_fees_and_revenue,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        noise_params=noise_params if noise_params is not None else {},
    )

    scan_inputs = [prices, active_initial_weights, per_asset_ratios,
                   all_other_assets_ratios, gamma_array, arb_thresh_array, arb_fees_array,
                   price_ratio_updates, lp_supply_array]
    if noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        scan_inputs.append(volatility_array)
    elif noise_model == "calibrated":
        scan_inputs.append(volatility_array)
        scan_inputs.append(dow_sin_array)
        scan_inputs.append(dow_cos_array)
    elif noise_model == "market_linear":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(noise_tvl_coeff_array)

    carry_init = [
        initial_reserves,
        initial_Va,
        initial_Vb,
        lp_supply_array[0],
        jnp.float64(0.0),  # step_idx
        jnp.float64(0.0),  # active_start_ratio
        jnp.float64(0.0),  # active_target_ratio
        jnp.float64(0.0),  # active_start_step
        jnp.float64(0.0),  # active_end_step
        jnp.array(False),  # active_enabled
    ]
    _, (reserves, fee_revenue) = scan(scan_fn, carry_init, scan_inputs)
    return reserves, fee_revenue


@partial(jit, static_argnums=(11,), static_argnames=('noise_model',))
def _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees,
    arb_thresh,
    arb_fees,
    price_ratio_updates=None,
    do_trades=False,
    trades=None,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    lp_supply_array=None,
    noise_model="ratio",
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
):
    """Calculate reClAMM reserves and LP fee revenue with time-varying fees/arb arrays.

    Returns
    -------
    reserves : jnp.ndarray, shape (T, 2)
    fee_revenue : jnp.ndarray, shape (T,)
        LP fee revenue per timestep in USD.
    """
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    n_assets = 2
    weights = jnp.array([0.5, 0.5])

    gamma = jnp.where(fees.size == 1, jnp.full(prices.shape[0], 1.0 - fees), 1.0 - fees)
    arb_thresh = jnp.where(
        arb_thresh.size == 1, jnp.full(prices.shape[0], arb_thresh), arb_thresh
    )
    arb_fees = jnp.where(
        arb_fees.size == 1, jnp.full(prices.shape[0], arb_fees), arb_fees
    )
    if price_ratio_updates is None:
        price_ratio_updates = jnp.zeros((prices.shape[0], 4), dtype=prices.dtype)
        price_ratio_updates = price_ratio_updates.at[:, 3].set(jnp.nan)
    else:
        if price_ratio_updates.ndim == 1:
            price_ratio_updates = jnp.broadcast_to(
                price_ratio_updates, (prices.shape[0], price_ratio_updates.shape[0])
            )
        elif price_ratio_updates.shape[0] == 1 and prices.shape[0] != 1:
            price_ratio_updates = jnp.broadcast_to(
                price_ratio_updates, (prices.shape[0], price_ratio_updates.shape[1])
            )

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_prices_and_dynamic_fees(
            weights, prices, gamma, tokens_to_drop,
            active_trade_directions, leave_one_out_idxs,
        )
    )

    scan_fn = Partial(
        _reclamm_scan_step_with_fees_and_revenue,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        noise_params=noise_params if noise_params is not None else {},
    )

    scan_inputs = [prices, active_initial_weights, per_asset_ratios,
                   all_other_assets_ratios, gamma, arb_thresh, arb_fees,
                   price_ratio_updates, lp_supply_array]
    if noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        scan_inputs.append(volatility_array)
    elif noise_model == "calibrated":
        scan_inputs.append(volatility_array)
        scan_inputs.append(dow_sin_array)
        scan_inputs.append(dow_cos_array)
    elif noise_model == "market_linear":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(noise_tvl_coeff_array)

    carry_init = [
        initial_reserves,
        initial_Va,
        initial_Vb,
        lp_supply_array[0],
        jnp.float64(0.0),  # step_idx
        jnp.float64(0.0),  # active_start_ratio
        jnp.float64(0.0),  # active_target_ratio
        jnp.float64(0.0),  # active_start_step
        jnp.float64(0.0),  # active_end_step
        jnp.array(False),  # active_enabled
    ]
    _, (reserves, fee_revenue) = scan(scan_fn, carry_init, scan_inputs)
    return reserves, fee_revenue
