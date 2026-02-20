"""Optimal arbitrage computation for n-asset G3M pools.

Computes the profit-maximising arbitrage trade for a Geometric Mean
Market Maker with an arbitrary number of assets. Enumerates trade-
direction signatures (which tokens flow in vs out), constructs the
closed-form optimal trade for each signature, and selects the most
profitable. Supports pre-computation and vmapping for efficient use
inside the simulation scan loop.
"""

import numpy as np

from functools import partial

from jax import config, jit, vmap
import jax.numpy as jnp

config.update("jax_enable_x64", True)

np.seterr(all="raise")
np.seterr(under="print")


def get_signature(trade):
    """
    Converts trade direction to signature
    Returns: 1 for buy, -1 for sell, 0 for hold
    args: trade: np.array of trade directions

    """
    sig = np.zeros_like(trade)
    sig[trade != 0] = trade[trade != 0] / np.abs(trade[trade != 0])
    return sig


def compare_signatures(sig1, sig2):
    """
    Compares two signatures for equality, ignoring zeros
    Returns: True if signatures are equal, False otherwise
    args: sig1, sig2: np.array of signatures

    """
    if jnp.sum(sig1[sig1 != 0] == sig2[sig1 != 0]) == len(sig1[sig1 != 0]):
        return True
    elif jnp.sum(sig1[sig2 != 0] == sig2[sig2 != 0]) == len(sig2[sig2 != 0]):
        return True
    elif jnp.sum(sig1 == sig2) != len(sig1):
        # print('SIG NOT MATCH')
        return False
    else:
        return True


def direction_to_sig(trade_direction):
    """
    Converts trade direction to signature
    Returns: 1 for buy, -1 for sell, 0 for hold
    args: trade_direction: np.array of trade directions

    """
    return get_signature(trade_direction - 0.5)


def sig_to_tokens_to_keep(sig):
    """
    Converts signature to boolean array of tokens to keep
    Returns: boolean array of tokens to keep
    args: sig: np.array of signatures

    """
    return sig != 0


def sig_to_direction(sig):
    """
    Converts signature to trade direction
    Returns: trade direction
    args: sig: np.array of signatures

    """
    trade_direction = np.zeros_like(sig)
    trade_direction[sig == 1] = 1
    return trade_direction


def sig_to_direction_jnp(sig):
    """
    Converts signature to trade direction
    Returns: trade direction
    args: sig: jax np.array of signatures

    """
    return jnp.where(sig == 1, 1, 0)


def trade_to_direction_jnp(trade):
    """
    Converts trade to trade direction
    Returns: trade direction
    args: trade: jax np.array of trades

    """
    return jnp.where(trade > 0, 1, 0)


# @partial(jit, static_argnums=(5,))
def construct_optimal_trade_jnp(
    initial_weights, local_prices, initial_reserves, fee_gamma, sig, n, slack=0
):
    """

    Constructs optimal trade given initial conditions and signature
    Returns: optimal trade
    args: initial_weights: jax np.array of initial weights
          local_prices: jax np.array of local prices
          initial_reserves: jax np.array of initial reserves
          fee_gamma: float of fee gamma
          sig: jax np.array of signatures
          n: int of number of tokens
          slack: float of slack value

    """

    # central_reserves = current_value * dex_weights_local/market_prices
    tokens_to_keep = sig_to_tokens_to_keep(sig)
    tokens_to_drop = jnp.invert(tokens_to_keep)
    active_local_prices = local_prices
    active_initial_reserves = initial_reserves
    active_initial_reserves = jnp.where(tokens_to_drop, 1.0, active_initial_reserves)
    partial_initial_weigts = jnp.where(tokens_to_drop, 0.0, initial_weights)
    active_initial_weights = initial_weights / partial_initial_weigts.sum()
    # active_initial_weights = active_initial_weights / jnp.sum(active_initial_weights)
    # active_current_value = (active_initial_reserves * active_local_prices).sum()

    active_n = n
    active_trade_direction = sig_to_direction_jnp(sig)
    per_asset_ratio = (
        (active_initial_weights * (fee_gamma ** (active_trade_direction)))
        / (active_local_prices)
    ) ** (1.0 - active_initial_weights)
    # log_per_asset_ratio = (1.0-initial_weights) * (np.log(initial_weights)
    # + trade_direction*np.log(fee_gamma)-np.log(local_prices)-np.log(initial_reserves))
    all_other_assets_quantities = (
        (active_local_prices)
        / ((fee_gamma ** (active_trade_direction)) * active_initial_weights)
    ) ** (active_initial_weights)
    all_other_assets_quantities = jnp.where(
        tokens_to_drop, 1.0, all_other_assets_quantities
    )
    # log_all_other_assets_quantities = (initial_weights) * (np.log(local_prices)+ np.log(initial_reserves) - trade_direction*np.log(fee_gamma)- np.log(initial_weights))
    leave_one_out_idx = jnp.arange(1, active_n) - jnp.tri(
        active_n, active_n - 1, k=-1, dtype=bool
    )
    all_other_assets_ratio = jnp.prod(
        all_other_assets_quantities[leave_one_out_idx], axis=-1
    )
    active_initial_constant = jnp.prod(active_initial_reserves**active_initial_weights)
    active_overall_trade = (1.0 / ((fee_gamma ** (active_trade_direction)))) * (
        ((active_initial_constant) * per_asset_ratio * all_other_assets_ratio)
        - active_initial_reserves
    )
    # print(sig)
    # print(per_asset_ratio)
    active_overall_trade = jnp.where(tokens_to_drop, 0.0, active_overall_trade)
    initial_constant = jnp.prod((initial_reserves) ** initial_weights)
    valid_post_trade_reserves = (
        jnp.sum(initial_reserves + active_overall_trade > 0) == n
    )
    post_trade_constant = jnp.prod(
        (
            initial_reserves
            + active_overall_trade
            * (fee_gamma ** (trade_to_direction_jnp(active_overall_trade)))
        )
        ** initial_weights
    )
    # Use proportional tolerance: (post - initial) / initial >= slack
    # This makes the tolerance scale-invariant with pool size
    relative_diff = (post_trade_constant - initial_constant) / initial_constant
    valid_post_trade_constant = relative_diff >= slack
    valid_trade = jnp.logical_and(valid_post_trade_reserves, valid_post_trade_constant)
    return jnp.where(valid_trade, active_overall_trade, 0)
    # return active_overall_trade, valid_post_trade_reserves * valid_post_trade_constant


construct_optimal_trade_jnp_vmapped = vmap(
    construct_optimal_trade_jnp, in_axes=[None, None, None, None, 0, None, None]
)


def calc_active_inital_reserves_for_one_signature(initial_reserves, tokens_to_drop):
    active_initial_reserves = jnp.where(tokens_to_drop, 1.0, initial_reserves)
    return active_initial_reserves


calc_active_inital_reserves_for_all_signatures = vmap(
    calc_active_inital_reserves_for_one_signature, in_axes=[None, 0]
)


# @partial(jit, static_argnums=(5,))
def optimal_trade_sifter(
    initial_weights,
    local_prices,
    initial_reserves,
    fee_gamma,
    all_sig_variations,
    n,
    slack=0,
):
    """

    Filters optimal trades based on profitability and arbitrage conditions
    Returns: optimal trade
    args: initial_weights: jax np.array of initial weights
          local_prices: jax np.array of local prices
          initial_reserves: jax np.array of initial reserves
          fee_gamma: float of fee gamma
          all_sig_variations: jax np.array of all signature variations
          n: int of number of tokens
          slack: float of slack value

    """
    overall_trades = construct_optimal_trade_jnp_vmapped(
        initial_weights,
        local_prices,
        initial_reserves,
        fee_gamma,
        all_sig_variations,
        n,
        slack,
    )
    profits = -(overall_trades * local_prices).sum(-1)
    mask = jnp.zeros_like(profits)
    mask = jnp.where(profits == jnp.max(profits), 1.0, 0.0)
    return mask @ overall_trades


def precalc_shared_values_for_one_signature(sig, n):
    tokens_to_keep = sig_to_tokens_to_keep(sig)
    active_trade_direction = sig_to_direction_jnp(sig)
    tokens_to_drop = jnp.invert(tokens_to_keep)
    leave_one_out_idx = jnp.arange(1, n) - jnp.tri(n, n - 1, k=-1, dtype=bool)
    return tokens_to_keep, active_trade_direction, tokens_to_drop, leave_one_out_idx


precalc_shared_values_for_all_signatures = vmap(
    precalc_shared_values_for_one_signature, in_axes=[0, None]
)


def precalc_components_of_optimal_trade_for_one_signature(
    initial_weights,
    local_prices,
    fee_gamma,
    tokens_to_drop,
    active_trade_direction,
    leave_one_out_idx,
):
    # central_reserves = current_value * dex_weights_local/market_prices

    active_local_prices = local_prices
    partial_initial_weigts = jnp.where(tokens_to_drop, 0.0, initial_weights)
    active_initial_weights = initial_weights / partial_initial_weigts.sum()
    # active_initial_weights = active_initial_weights / jnp.sum(active_initial_weights)
    # active_current_value = (active_initial_reserves * active_local_prices).sum()

    per_asset_ratio = (
        (active_initial_weights * (fee_gamma ** (active_trade_direction)))
        / (active_local_prices)
    ) ** (1.0 - active_initial_weights)
    # log_per_asset_ratio = (1.0-initial_weights) * (np.log(initial_weights) 
    # + trade_direction*np.log(fee_gamma)-np.log(local_prices)-np.log(initial_reserves))
    all_other_assets_quantities = (
        (active_local_prices)
        / ((fee_gamma ** (active_trade_direction)) * active_initial_weights)
    ) ** (active_initial_weights)
    all_other_assets_quantities = jnp.where(
        tokens_to_drop, 1.0, all_other_assets_quantities
    )
    # log_all_other_assets_quantities = (initial_weights) * (np.log(local_prices)
    # + np.log(initial_reserves) - trade_direction*np.log(fee_gamma)- np.log(initial_weights))
    all_other_assets_ratio = jnp.prod(
        all_other_assets_quantities[leave_one_out_idx], axis=-1
    )
    return active_initial_weights, per_asset_ratio, all_other_assets_ratio


precalc_components_of_optimal_trade_across_signatures = vmap(
    precalc_components_of_optimal_trade_for_one_signature,
    in_axes=[None, None, None, 0, 0, 0],
)

precalc_components_of_optimal_trade_across_weights_and_prices = vmap(
    precalc_components_of_optimal_trade_across_signatures,
    in_axes=[0, 0, None, None, None, None],
)

precalc_components_of_optimal_trade_across_prices = vmap(
    precalc_components_of_optimal_trade_across_signatures,
    in_axes=[None, 0, None, None, None, None],
)

precalc_components_of_optimal_trade_across_weights_and_prices_and_dynamic_fees = vmap(
    precalc_components_of_optimal_trade_across_signatures,
    in_axes=[0, 0, 0, None, None, None],
)

precalc_components_of_optimal_trade_across_prices_and_dynamic_fees = vmap(
    precalc_components_of_optimal_trade_across_signatures,
    in_axes=[None, 0, 0, None, None, None],
)


# @jit
def calc_optimal_trade_for_one_signature(
    initial_reserves,
    initial_weights,
    active_initial_reserves,
    active_initial_weights,
    active_trade_direction,
    per_asset_ratio,
    all_other_assets_ratio,
    tokens_to_drop,
    fee_gamma,
    n,
    slack=0,
):

    active_initial_constant = jnp.prod(active_initial_reserves**active_initial_weights)
    active_overall_trade = (1.0 / ((fee_gamma ** (active_trade_direction)))) * (
        ((active_initial_constant) * per_asset_ratio * all_other_assets_ratio)
        - active_initial_reserves
    )

    active_overall_trade = jnp.where(tokens_to_drop, 0.0, active_overall_trade)
    initial_constant = jnp.prod((initial_reserves) ** initial_weights)
    valid_post_trade_reserves = (
        jnp.sum(initial_reserves + active_overall_trade > 0) == n
    )
    post_trade_constant = jnp.prod(
        (
            initial_reserves
            + active_overall_trade
            * (fee_gamma ** (trade_to_direction_jnp(active_overall_trade)))
        )
        ** initial_weights
    )
    # Use proportional tolerance: (post - initial) / initial >= slack
    # This makes the tolerance scale-invariant with pool size
    relative_diff = (post_trade_constant - initial_constant) / initial_constant
    valid_post_trade_constant = relative_diff >= slack
    valid_trade = jnp.logical_and(valid_post_trade_reserves, valid_post_trade_constant)
    return jnp.where(valid_trade, active_overall_trade, 0)
    # return {
    #     "initial_reserves":initial_reserves,
    #     "n":n,
    #     "active_initial_constant": active_initial_constant,
    #     "active_overall_trade": active_overall_trade,
    #     "initial_constant": initial_constant,
    #     "valid_post_trade_reserves": valid_post_trade_reserves,
    #     "valid_post_trade_constant": valid_post_trade_constant,
    #     "valid_trade": valid_trade,
    #     "trade": jnp.where(valid_trade, active_overall_trade, 0),
    # }


calc_optimal_trade_across_signatures = jit(
    vmap(
        calc_optimal_trade_for_one_signature,
        in_axes=[None, None, 0, 0, 0, 0, 0, 0, None, None, None],
    )
)


@partial(jit, static_argnums=(9,))
def parallelised_optimal_trade_sifter(
    initial_reserves,
    initial_weights,
    local_prices,
    active_initial_weights,
    active_trade_directions,
    per_asset_ratio,
    all_other_assets_ratio,
    tokens_to_drop,
    fee_gamma,
    n,
    slack=0,
):
    """

    Filters optimal trades based on profitability and arbitrage conditions
    Returns: optimal trade
    args: initial_weights: jax np.array of initial weights
          local_prices: jax np.array of local prices
          initial_reserves: jax np.array of initial reserves
          fee_gamma: float of fee gamma
          n: int of number of tokens
          slack: float of slack value

    """
    active_initial_reserves = calc_active_inital_reserves_for_all_signatures(
        initial_reserves, tokens_to_drop
    )
    overall_trades = calc_optimal_trade_across_signatures(
        initial_reserves,
        initial_weights,
        active_initial_reserves,
        active_initial_weights,
        active_trade_directions,
        per_asset_ratio,
        all_other_assets_ratio,
        tokens_to_drop,
        fee_gamma,
        n,
        slack,
    )

    profits = -(overall_trades * local_prices).sum(-1)
    # idx = (profits>0) * (constant_differences > constant_slack)
    # filtered_trades = jnp.where(idx, overall_trades, 0.0)
    # filtered_profits = jnp.where(idx, profits, 0)
    mask = jnp.zeros_like(profits)
    mask = jnp.where(profits == jnp.max(profits), 1.0, 0.0)
    return mask @ overall_trades


def wrapped_parallelised_optimal_trade_sifter(
    initial_weights,
    local_prices,
    initial_reserves,
    fee_gamma,
    all_sig_variations,
    n,
    slack=0,
):
    _, active_trade_directions, tokens_to_drop, leave_one_out_idx = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n)
    )

    active_initial_weights, per_asset_ratio, all_other_assets_ratio = (
        precalc_components_of_optimal_trade_across_signatures(
            initial_weights,
            local_prices,
            fee_gamma,
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idx,
        )
    )
    trade = parallelised_optimal_trade_sifter(
        initial_reserves,
        initial_weights,
        local_prices,
        active_initial_weights,
        active_trade_directions,
        per_asset_ratio,
        all_other_assets_ratio,
        tokens_to_drop,
        fee_gamma,
        n,
        slack=slack,
    )
    return trade


# ============================================================================
# Arbitrage Fee-Adjusted Optimal Trade Functions
# ============================================================================
#
# When arb_fees > 0, the arbitrageur faces external rebalancing costs that
# affect the optimal trade size. These wrapper functions incorporate arb_fees
# into the optimal trade calculation by adjusting effective market prices.
#
# Mathematical derivation:
# The modified objective is: max_Φ: -∑(mₚ,ᵢΦᵢ) - (arb_fees/2) × ∑|mₚ,ᵢΦᵢ|
# For a fixed trade signature, |Φᵢ| = sign(Φᵢ) × Φᵢ, so:
#   objective = -∑(mₚ,ᵢ × (1 + arb_fees/2 × sᵢ) × Φᵢ)
# where sᵢ = +1 for tokens in, -1 for tokens out.
#
# This is equivalent to using adjusted effective market prices:
#   m̃ₚ,ᵢ = mₚ,ᵢ × (1 + arb_fees/2 × sᵢ)
#
# See https://arxiv.org/abs/2402.06731 for the base derivation.
# ============================================================================


def adjust_prices_for_arb_fees(
    local_prices,
    active_trade_direction,
    tokens_to_drop,
    arb_fees,
):
    """
    Adjust market prices to account for external arbitrage costs.

    When arb_fees > 0, the arbitrageur faces additional costs for rebalancing
    their portfolio on external venues. This function computes effective prices
    that incorporate these costs into the optimization.

    Parameters
    ----------
    local_prices : jnp.ndarray
        Array of market prices for each token.
    active_trade_direction : jnp.ndarray
        Array where 1 = token going into pool, 0 = token coming out.
    tokens_to_drop : jnp.ndarray
        Boolean array indicating inactive tokens for this signature.
    arb_fees : float
        External arbitrage fees as a fraction (e.g., 0.001 = 0.1%).

    Returns
    -------
    jnp.ndarray
        Adjusted prices incorporating external cost effects.
    """
    # sig_direction: +1 for tokens IN (arb sells to pool, buys externally at higher price)
    #               -1 for tokens OUT (arb buys from pool, sells externally at lower price)
    sig_direction = 2 * active_trade_direction - 1  # converts 0->-1, 1->+1
    price_adjustment = 1.0 + 0.5 * arb_fees * sig_direction
    # Only adjust active tokens (not dropped ones)
    price_adjustment = jnp.where(tokens_to_drop, 1.0, price_adjustment)
    return local_prices * price_adjustment


adjust_prices_for_arb_fees_across_signatures = vmap(
    adjust_prices_for_arb_fees, in_axes=[None, 0, 0, None]
)


def precalc_components_with_arb_fees_for_one_signature(
    initial_weights,
    local_prices,
    fee_gamma,
    tokens_to_drop,
    active_trade_direction,
    leave_one_out_idx,
    arb_fees,
):
    """
    Wrapper that adjusts prices for arb_fees then calls the base precalc function.

    Parameters
    ----------
    initial_weights : jnp.ndarray
        Array of pool weights for each token.
    local_prices : jnp.ndarray
        Array of market prices for each token.
    fee_gamma : float
        Pool fee parameter (1 - swap_fee).
    tokens_to_drop : jnp.ndarray
        Boolean array indicating inactive tokens for this signature.
    active_trade_direction : jnp.ndarray
        Array where 1 = token going into pool, 0 = token coming out.
    leave_one_out_idx : jnp.ndarray
        Index array for computing products excluding each element.
    arb_fees : float
        External arbitrage fees as a fraction.

    Returns
    -------
    tuple
        (active_initial_weights, per_asset_ratio, all_other_assets_ratio)
    """
    adjusted_prices = adjust_prices_for_arb_fees(
        local_prices, active_trade_direction, tokens_to_drop, arb_fees
    )
    return precalc_components_of_optimal_trade_for_one_signature(
        initial_weights,
        adjusted_prices,
        fee_gamma,
        tokens_to_drop,
        active_trade_direction,
        leave_one_out_idx,
    )


precalc_components_with_arb_fees_across_signatures = vmap(
    precalc_components_with_arb_fees_for_one_signature,
    in_axes=[None, None, None, 0, 0, 0, None],
)


precalc_components_with_arb_fees_across_weights_and_prices = vmap(
    precalc_components_with_arb_fees_across_signatures,
    in_axes=[0, 0, None, None, None, None, None],
)


precalc_components_with_arb_fees_across_weights_and_prices_and_dynamic_fees = vmap(
    precalc_components_with_arb_fees_across_signatures,
    in_axes=[0, 0, 0, None, None, None, 0],
)


def wrapped_parallelised_optimal_trade_sifter_with_arb_fees(
    initial_weights,
    local_prices,
    initial_reserves,
    fee_gamma,
    all_sig_variations,
    n,
    arb_fees=0.0,
    slack=0,
):
    """
    Compute optimal arbitrage trade incorporating external arb fees.

    This function extends wrapped_parallelised_optimal_trade_sifter to account
    for external arbitrage costs (e.g., CEX fees, gas costs) by adjusting
    effective market prices in the optimization.

    When arb_fees > 0, the optimal trade will be smaller than the zero-arb-fee
    case because the arbitrageur stops trading earlier (marginal profit goes
    to zero sooner due to external costs).

    Parameters
    ----------
    initial_weights : jnp.ndarray
        Array of pool weights for each token.
    local_prices : jnp.ndarray
        Array of market prices for each token.
    initial_reserves : jnp.ndarray
        Array of current pool reserves for each token.
    fee_gamma : float
        Pool fee parameter (1 - swap_fee).
    all_sig_variations : jnp.ndarray
        Array of all signature variations to test.
    n : int
        Number of tokens.
    arb_fees : float, optional
        External arbitrage fees as a fraction (default 0.0).
    slack : float, optional
        Slack for invariant validation (default 0).

    Returns
    -------
    jnp.ndarray
        Optimal trade vector incorporating arb_fees.
    """
    _, active_trade_directions, tokens_to_drop, leave_one_out_idx = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n)
    )

    active_initial_weights, per_asset_ratio, all_other_assets_ratio = (
        precalc_components_with_arb_fees_across_signatures(
            initial_weights,
            local_prices,
            fee_gamma,
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idx,
            arb_fees,
        )
    )
    trade = parallelised_optimal_trade_sifter(
        initial_reserves,
        initial_weights,
        local_prices,
        active_initial_weights,
        active_trade_directions,
        per_asset_ratio,
        all_other_assets_ratio,
        tokens_to_drop,
        fee_gamma,
        n,
        slack=slack,
    )
    return trade
