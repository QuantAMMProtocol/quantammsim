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
from jax.lax import scan
from jax.tree_util import Partial
from functools import partial

from quantammsim.pools.G3M.optimal_n_pool_arb import (
    precalc_shared_values_for_all_signatures,
    precalc_components_of_optimal_trade_across_prices,
    precalc_components_of_optimal_trade_across_prices_and_dynamic_fees,
    parallelised_optimal_trade_sifter,
)

# Reference balance for initialisation (matches Solidity _INITIALIZATION_MAX_BALANCE_A)
_INITIALIZATION_MAX_BALANCE_A = 1e6

# Virtual balance decay is capped at 30 days to prevent overflow
_MAX_DECAY_DURATION_SECONDS = 30 * 86400


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

def _reclamm_scan_step_zero_fees(
    carry_list,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
):
    """Single scan step for zero-fee reClAMM pool.

    Zero-fee means no trading fees, but the pool still needs to:
    1. Update virtual balances (path-dependent)
    2. Compute analytical constant-product arb (no fee friction)

    Carry: [real_reserves (2,), Va (0-d), Vb (0-d)]
    """
    prev_reserves = carry_list[0]
    Va = carry_list[1]
    Vb = carry_list[2]

    Ra = prev_reserves[0]
    Rb = prev_reserves[1]

    # Step 1: Update virtual balances if out of range
    centeredness, is_above = compute_centeredness(Ra, Rb, Va, Vb)
    sqrt_Q = jnp.sqrt(compute_price_ratio(Ra, Rb, Va, Vb))
    out_of_range = centeredness < centeredness_margin

    Va_updated, Vb_updated = compute_virtual_balances_updating_price_range(
        Ra, Rb, Va, Vb,
        is_pool_above_center=is_above,
        daily_price_shift_base=daily_price_shift_base,
        seconds_elapsed=seconds_per_step,
        sqrt_price_ratio=sqrt_Q,
    )
    Va = jnp.where(out_of_range, Va_updated, Va)
    Vb = jnp.where(out_of_range, Vb_updated, Vb)

    # Step 2: Analytical zero-fee arb on effective reserves
    # For constant product xy=k with effective reserves:
    # After arb, spot price = market price = prices[0]/prices[1]
    # New effective reserves: Ea_new = sqrt(L/p), Eb_new = sqrt(L*p)
    # where L = (Ra+Va)*(Rb+Vb) and p = prices[0]/prices[1]
    L = compute_invariant(Ra, Rb, Va, Vb)
    market_price = prices[0] / prices[1]

    # Effective reserves after arb at market price
    Ea_new = jnp.sqrt(L / market_price)
    Eb_new = jnp.sqrt(L * market_price)

    # Real reserves = effective - virtual
    Ra_new = Ea_new - Va
    Rb_new = Eb_new - Vb

    # Only apply if reserves remain non-negative (zero is valid at range boundary)
    valid = (Ra_new >= 0) & (Rb_new >= 0)
    Ra_new = jnp.where(valid, Ra_new, Ra)
    Rb_new = jnp.where(valid, Rb_new, Rb)

    new_reserves = jnp.array([Ra_new, Rb_new])
    return [new_reserves, Va, Vb], new_reserves


def _reclamm_scan_step_zero_fees_full_state(
    carry_list,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
):
    """Like _reclamm_scan_step_zero_fees but outputs (reserves, Va, Vb)."""
    new_carry, new_reserves = _reclamm_scan_step_zero_fees(
        carry_list, prices, centeredness_margin, daily_price_shift_base, seconds_per_step,
    )
    return new_carry, (new_reserves, new_carry[1], new_carry[2])


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
    arb_thresh=0.0,
    arb_fees=0.0,
):
    """Single scan step for reClAMM pool with fees.

    Uses the G3M optimal arb machinery with effective reserves (real + virtual)
    and weights = [0.5, 0.5].

    Carry: [real_reserves (2,), Va (0-d), Vb (0-d)]
    Input: [prices, active_initial_weights, per_asset_ratios, all_other_assets_ratios]
    """
    prev_reserves = carry_list[0]
    Va = carry_list[1]
    Vb = carry_list[2]

    Ra = prev_reserves[0]
    Rb = prev_reserves[1]

    prices = input_list[0]
    active_initial_weights = input_list[1]
    per_asset_ratios = input_list[2]
    all_other_assets_ratios = input_list[3]
    gamma = input_list[4]

    # Step 1: Update virtual balances if out of range
    centeredness, is_above = compute_centeredness(Ra, Rb, Va, Vb)
    sqrt_Q = jnp.sqrt(compute_price_ratio(Ra, Rb, Va, Vb))
    out_of_range = centeredness < centeredness_margin

    Va_updated, Vb_updated = compute_virtual_balances_updating_price_range(
        Ra, Rb, Va, Vb,
        is_pool_above_center=is_above,
        daily_price_shift_base=daily_price_shift_base,
        seconds_elapsed=seconds_per_step,
        sqrt_price_ratio=sqrt_Q,
    )
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

    # Apply trade to REAL reserves only (virtual are separate)
    # The arb trade is computed on effective reserves, so we apply it directly
    # to real reserves since effective = real + virtual and virtual doesn't change from arb
    Ra_new = Ra + jnp.where(do_trade, optimal_arb_trade[0], 0.0)
    Rb_new = Rb + jnp.where(do_trade, optimal_arb_trade[1], 0.0)

    # Revert if negative (zero is valid at range boundary)
    valid = (Ra_new >= 0) & (Rb_new >= 0)
    Ra_new = jnp.where(valid, Ra_new, Ra)
    Rb_new = jnp.where(valid, Rb_new, Rb)

    new_reserves = jnp.array([Ra_new, Rb_new])
    return [new_reserves, Va, Vb], new_reserves


@jit
def _jax_calc_reclamm_reserves_zero_fees(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
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

    Returns
    -------
    reserves : jnp.ndarray, shape (T, 2)
        Real reserves over time.
    """
    scan_fn = Partial(
        _reclamm_scan_step_zero_fees,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
    )

    carry_init = [initial_reserves, initial_Va, initial_Vb]
    _, reserves = scan(scan_fn, carry_init, prices)
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
):
    """Like _jax_calc_reclamm_reserves_zero_fees but also returns virtual balances.

    Returns
    -------
    reserves : jnp.ndarray, shape (T, 2)
    Va_history : jnp.ndarray, shape (T,)
    Vb_history : jnp.ndarray, shape (T,)
    """
    scan_fn = Partial(
        _reclamm_scan_step_zero_fees_full_state,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
    )

    carry_init = [initial_reserves, initial_Va, initial_Vb]
    _, (reserves, Va_history, Vb_history) = scan(scan_fn, carry_init, prices)
    return reserves, Va_history, Vb_history


@jit
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
):
    """Calculate reClAMM reserves over time with fees.

    Uses the G3M optimal arb machinery with constant weights [0.5, 0.5]
    applied to effective reserves (real + virtual).
    """
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

    scan_fn = Partial(
        _reclamm_scan_step_with_fees,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        n=n_assets,
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arb_thresh=arb_thresh,
        arb_fees=arb_fees,
    )

    carry_init = [initial_reserves, initial_Va, initial_Vb]
    _, reserves = scan(
        scan_fn,
        carry_init,
        [prices, active_initial_weights, per_asset_ratios,
         all_other_assets_ratios, gamma_array],
    )
    return reserves


@partial(jit, static_argnums=(10,))
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
    do_trades=False,
    trades=None,
    all_sig_variations=None,
):
    """Calculate reClAMM reserves with time-varying fees/arb arrays."""
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
    )

    carry_init = [initial_reserves, initial_Va, initial_Vb]
    _, reserves = scan(
        scan_fn,
        carry_init,
        [prices, active_initial_weights, per_asset_ratios,
         all_other_assets_ratios, gamma],
    )
    return reserves
