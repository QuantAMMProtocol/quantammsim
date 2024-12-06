
import numpy as np
from itertools import product

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
    if sum(sig1[sig1 != 0] == sig2[sig1 != 0]) == len(sig1[sig1 != 0]):
        return True
    elif sum(sig1[sig2 != 0] == sig2[sig2 != 0]) == len(sig2[sig2 != 0]):
        return True
    elif sum(sig1 == sig2) != len(sig1):
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
    valid_post_trade_constant = (
        jnp.prod(
            (
                initial_reserves
                + active_overall_trade
                * (fee_gamma ** (trade_to_direction_jnp(active_overall_trade)))
            )
            ** initial_weights
        )
        - initial_constant
        >= slack
    )
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
    valid_post_trade_constant = (
        jnp.prod(
            (
                initial_reserves
                + active_overall_trade
                * (fee_gamma ** (trade_to_direction_jnp(active_overall_trade)))
            )
            ** initial_weights
        )
        - initial_constant
        >= slack
    )
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
        0,
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
        slack=0,
    )
    return trade


if __name__ == "__main__":

    np.random.seed(0)
    n = 2
    initial_weights = jnp.array([1.0 / n] * n)

    local_prices = jnp.arange(n) + 1.0

    intial_value = 100.0
    initial_reserves = intial_value * initial_weights / local_prices
    fee_gamma = 0.99

    local_prices = local_prices * (np.random.randn(n) * 0.5 + 1.0)
    all_sig_variations = np.array(list(product([1, 0, -1], repeat=n)))
    all_sig_variations = all_sig_variations[(all_sig_variations != 0).sum(-1) > 1]
    all_sig_variations = all_sig_variations[np.any(all_sig_variations == 1, -1)]
    all_sig_variations = all_sig_variations[np.any(all_sig_variations == -1, -1)]
    all_sig_variations = jnp.array(tuple(map(tuple, all_sig_variations)))

    pp = wrapped_parallelised_optimal_trade_sifter(
        initial_weights,
        local_prices,
        initial_reserves,
        fee_gamma,
        all_sig_variations,
        n,
        slack=0,
    )

    lin = optimal_trade_sifter(
        initial_weights,
        local_prices,
        initial_reserves,
        fee_gamma,
        all_sig_variations,
        n,
        slack=0,
    )

    tokens_to_keep, active_trade_directions, tokens_to_drop, leave_one_out_idx = (
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

    active_initial_reserves = calc_active_inital_reserves_for_all_signatures(
        initial_reserves, tokens_to_drop
    )
    overall_trades = calc_optimal_trade_for_one_signature(
        initial_reserves,
        initial_weights,
        active_initial_reserves[-1],
        active_initial_weights[-1],
        active_trade_directions[-1],
        per_asset_ratio[-1],
        all_other_assets_ratio[-1],
        tokens_to_drop[-1],
        fee_gamma,
        0,
        n,
    )
    print(
        initial_reserves,
        initial_weights,
        active_initial_reserves[-1],
        active_initial_weights[-1],
        per_asset_ratio[-1],
        all_other_assets_ratio[-1],
        tokens_to_drop[-1],
    )
    slack = 0
    sig = all_sig_variations[-1]
    current_value = (initial_reserves * local_prices).sum()
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
    #  + trade_direction*np.log(fee_gamma)-np.log(local_prices)-np.log(initial_reserves))
    all_other_assets_quantities = (
        (active_local_prices)
        / ((fee_gamma ** (active_trade_direction)) * active_initial_weights)
    ) ** (active_initial_weights)
    all_other_assets_quantities = jnp.where(
        tokens_to_drop, 1.0, all_other_assets_quantities
    )
    # log_all_other_assets_quantities = (initial_weights) * (np.log(local_prices)
    # + np.log(initial_reserves) - trade_direction*np.log(fee_gamma)- np.log(initial_weights))
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
    active_overall_trade = jnp.where(tokens_to_drop, 0.0, active_overall_trade)
    initial_constant = jnp.prod((initial_reserves) ** initial_weights)
    valid_post_trade_reserves = (
        jnp.sum(initial_reserves + active_overall_trade > 0) == n
    )
    valid_post_trade_constant = (
        jnp.prod(
            (
                initial_reserves
                + active_overall_trade
                * (fee_gamma ** (trade_to_direction_jnp(active_overall_trade)))
            )
            ** initial_weights
        )
        - initial_constant
        >= slack
    )
    valid_trade = jnp.logical_and(valid_post_trade_reserves, valid_post_trade_constant)
    proposed_trade = jnp.where(valid_trade, active_overall_trade, 0)
    print(
        initial_reserves,
        initial_weights,
        active_initial_reserves,
        active_initial_weights,
        per_asset_ratio,
        all_other_assets_ratio,
        tokens_to_drop,
    )

    sig = all_sig_variations[8]
    initial_reserves = prev_reserves
    local_prices = prices
    initial_weights = prev_weights
    fee_gamma = gamma
    active_n = n

    tokens_to_keep = sig_to_tokens_to_keep(sig)
    tokens_to_drop = jnp.invert(tokens_to_keep)
    active_local_prices = local_prices
    active_initial_reserves = initial_reserves
    active_initial_reserves = jnp.where(tokens_to_drop, 1.0, active_initial_reserves)
    partial_initial_weigts = jnp.where(tokens_to_drop, 0.0, initial_weights)
    active_initial_weights = initial_weights / partial_initial_weigts.sum()
    active_trade_direction = sig_to_direction_jnp(sig)
    per_asset_ratio = (
        (active_initial_weights * (fee_gamma ** (active_trade_direction)))
        / (active_local_prices)
    ) ** (1.0 - active_initial_weights)
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
