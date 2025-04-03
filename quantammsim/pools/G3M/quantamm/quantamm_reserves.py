# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jax import jit, vmap
from jax import devices
from jax.tree_util import Partial
from jax.lax import scan
from jax import default_backend
from jax import local_device_count, devices

from functools import partial

from quantammsim.pools.G3M.optimal_n_pool_arb import (
    precalc_shared_values_for_all_signatures,
    precalc_components_of_optimal_trade_across_weights_and_prices,
    precalc_components_of_optimal_trade_across_weights_and_prices_and_dynamic_fees,
    parallelised_optimal_trade_sifter,
)
from quantammsim.pools.G3M.G3M_trades import jitted_G3M_cond_trade
from quantammsim.pools.noise_trades import calculate_reserves_after_noise_trade

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
else:
    GPU_DEVICE = devices("cpu")[0]


@jit
def _jax_calc_quantAMM_reserve_ratio(prev_weights, prev_prices, weights, prices):
    """
    Calculate reserves ratio changes for a single timestep.

    This function computes the changes in reserves for an automated market maker (AMM) model
    based on a single change in asset weights and prices. It is optimized for GPU execution.

    Parameters
    ----------
    prev_weights : jnp.ndarray
        Array of previous asset weights.
    prev_prices : jnp.ndarray
        Array of previous asset prices.
    weights : jnp.ndarray
        Array of current asset weights.
    prices : jnp.ndarray
        Array of current asset prices.

    Returns
    -------
    jnp.ndarray
        Array of reserves ratio changes.
    """

    # first we do the change in reserve that corresponds to
    # Weights Constant, Prices Change, appendix B1 of QuantAMM whitepaper
    # from the change in prices over the course of the block
    price_change_ratio = prices / prev_prices
    price_product_change_ratio = jnp.prod(price_change_ratio**prev_weights)
    reserves_ratios_from_price_change = price_product_change_ratio / price_change_ratio
    # second do part of change in reserves that corresponds to
    # Weights Change, Prices Constant, appendix B2of QuantAMM whitepaper
    # this is, in effect, at start of a block where we are quoting
    # new weights, creating an arb opportunity
    weight_change_ratio = weights / prev_weights
    weight_product_change_ratio = jnp.prod(weight_change_ratio**weights)
    reserves_ratio_from_weight_change = (
        weight_change_ratio / weight_product_change_ratio
    )
    reserves_ratios = (
        reserves_ratios_from_price_change * reserves_ratio_from_weight_change
    )
    return reserves_ratios


_jax_calc_quantAMM_reserve_ratios = jit(
    vmap(_jax_calc_quantAMM_reserve_ratio, in_axes=(0, 0, 0, 0))
)


@jit
def _jax_calc_quantAMM_reserves_with_fees_using_precalcs(
    initial_reserves,
    weights,
    prices,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    all_sig_variations=None,
    noise_trader_ratio=0.0,
):
    """
    Calculate AMM reserves considering fees and arbitrage opportunities using signature variations,
    using the approach given in https://arxiv.org/abs/2402.06731, for QuantAMM pools, pools with
    time-varying weights.

    This function computes the changes in reserves for an automated market maker (AMM) model
    considering transaction fees and potential arbitrage opportunities.
    It uses a scan operation to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    weights : jnp.ndarray
        Two-dimensional array of asset weights over time.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    fees : float, optional
        Swap fee charged on transactions, by default 0.003.
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.
    all_sig_variations : jnp.ndarray, optional
        Array of all signature variations used for arbitrage calculations.
    noise_trader_ratio : float, optional
        Ratio of noise traders to arbitrageurs, by default 0.0.

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """
    # initial_weights = coarse_weights[0]
    # initial_i = 0
    n_assets = weights.shape[1]

    # We do this like a block, so first there is the new
    # weight value and THEN we get new prices by the end of
    # the block.

    # So, for first weight, we have initial reserves, weights and
    # prices, so the change is 1

    n = prices.shape[0]

    initial_prices = prices[0]

    initial_weights = weights[0]

    gamma = 1.0 - fees

    # pre-calculate some values that are repeatedly used in optimal arb calculations

    tokens_to_keep, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    # calculate values that can be done in parallel

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_weights_and_prices(
            weights,
            prices,
            gamma,
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idxs,
        )
    )

    (
        lagged_active_initial_weights,
        lagged_per_asset_ratios,
        lagged_all_other_assets_ratios,
    ) = precalc_components_of_optimal_trade_across_weights_and_prices(
        jnp.vstack(
            [
                weights[0],
                weights[:-1],
            ]
        ),
        prices,
        gamma,
        tokens_to_drop,
        active_trade_directions,
        leave_one_out_idxs,
    )

    scan_fn = Partial(
        _jax_calc_quantAMM_reserves_with_fees_scan_function_using_precalcs,
        gamma=gamma,
        arb_thresh=arb_thresh,
        arb_fees=arb_fees,
        n=n_assets,
        all_sig_variations=all_sig_variations,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        noise_trader_ratio=noise_trader_ratio,
    )

    carry_list_init = [
        initial_weights,
        initial_prices,
        initial_reserves,
        # active_initial_weights[0],
        # per_asset_ratios[0],
        # all_other_assets_ratios[0],
        0,
    ]
    # carry_list_init = [initial_weights, initial_i]
    # nojit_scan = jax.disable_jit()(jax.lax.scan)
    carry_list_end, reserves = scan(
        scan_fn,
        carry_list_init,
        [
            weights,
            prices,
            active_initial_weights,
            per_asset_ratios,
            all_other_assets_ratios,
            lagged_active_initial_weights,
            lagged_per_asset_ratios,
            lagged_all_other_assets_ratios,
        ],
    )

    return reserves


@partial(jit, static_argnums=(5,))
def _jax_calc_quantAMM_reserves_with_fees_scan_function_using_precalcs(
    carry_list,
    weights_and_prices_and_precalcs,
    all_sig_variations,
    tokens_to_drop,
    active_trade_directions,
    n,
    gamma=0.997,
    arb_thresh=0.0,
    arb_fees=0.0,
    noise_trader_ratio=0.0,
):
    """
    Calculate changes in AMM reserves considering fees and arbitrage opportunities using signature variations.

    This function extends the basic reserve calculation by incorporating transaction fees
    and potential arbitrage opportunities, following the methodology described in
    https://arxiv.org/abs/2402.06731. It tests different trade signature variations to determine
    optimal arbitrage trades.

    Note that this function is written to have maximum pre-calculation of values
    that are repeatedly re-used in the arbitrage calculations, which we denote 'precalcs'.

    Parameters
    ----------
    carry_list : list
        List containing the previous weights, prices, and reserves.
    weights_and_prices_and_precalcs : jnp.ndarray
        Array containing the current weights and prices.
    all_sig_variations : jnp.ndarray
        Array of all signature variations used for arbitrage calculations.
    n : int
        Number of tokens or assets.
    gamma : float, optional
        Discount factor for no-arbitrage bounds, by default 0.997.
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves changes.
    """

    # carry_list[0] is previous weights
    prev_weights = carry_list[0]

    # carry_list[1] is previous prices
    prev_prices = carry_list[1]

    # carry_list[2] is previous reserves
    prev_reserves = carry_list[2]

    counter = carry_list[3]

    # weights_and_prices_and_precalcs are the weights and prices, in that order
    # we have a lot of precalcs
    weights = weights_and_prices_and_precalcs[0]
    prices = weights_and_prices_and_precalcs[1]
    active_initial_weights = weights_and_prices_and_precalcs[2]
    per_asset_ratios = weights_and_prices_and_precalcs[3]
    all_other_assets_ratios = weights_and_prices_and_precalcs[4]
    lagged_active_initial_weights = weights_and_prices_and_precalcs[5]
    lagged_per_asset_ratios = weights_and_prices_and_precalcs[6]
    lagged_all_other_assets_ratios = weights_and_prices_and_precalcs[7]

    fees_are_being_charged = gamma != 1.0

    current_value = (prev_reserves * prices).sum()
    quoted_prices = current_value * prev_weights / prev_reserves

    price_change_ratio = prices / quoted_prices
    price_product_change_ratio = jnp.prod(price_change_ratio**prev_weights)
    reserves_ratios_from_price_change = price_product_change_ratio / price_change_ratio

    post_price_reserves_zero_fees = prev_reserves * reserves_ratios_from_price_change

    optimal_arb_trade_zero_fees = post_price_reserves_zero_fees - prev_reserves

    optimal_arb_trade_fees = parallelised_optimal_trade_sifter(
        prev_reserves,
        prev_weights,
        prices,
        lagged_active_initial_weights,
        active_trade_directions,
        lagged_per_asset_ratios,
        lagged_all_other_assets_ratios,
        tokens_to_drop,
        gamma,
        n,
        0,
    )

    optimal_arb_trade = jnp.where(
        fees_are_being_charged,
        optimal_arb_trade_fees,
        optimal_arb_trade_zero_fees,
    )
    post_price_reserves = prev_reserves + optimal_arb_trade

    # apply noise trade if noise_trader_ratio is greater than 0
    post_price_reserves = jnp.where(
        noise_trader_ratio > 0,
        calculate_reserves_after_noise_trade(
            optimal_arb_trade, post_price_reserves, prices, noise_trader_ratio, gamma
        ),
        post_price_reserves,
    )
    # check if this is worth the cost to arbs
    # delta = post_price_reserves - prev_reserves
    # is this delta a good deal for the arb?
    profit_to_arb = -(optimal_arb_trade * prices).sum() - arb_thresh
    profit_prices = profit_to_arb

    arb_external_rebalance_cost = (
        0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
    )

    arb_profitable = profit_to_arb >= arb_external_rebalance_cost

    # if arb trade IS profitable
    # then reserves is equal to post_price_reserves, otherwise equal to prev_reserves
    do_price_arb_trade = arb_profitable

    reserves = jnp.where(do_price_arb_trade, post_price_reserves, prev_reserves)

    current_value = (reserves * prices).sum()
    quoted_prices = current_value * weights / reserves

    price_change_ratio = prices / quoted_prices
    price_product_change_ratio = jnp.prod(price_change_ratio**weights)
    reserves_ratios_from_weight_change = price_product_change_ratio / price_change_ratio

    post_weight_reserves_zero_fees = reserves_ratios_from_weight_change * reserves

    optimal_arb_trade_zero_fees = post_weight_reserves_zero_fees - reserves

    optimal_arb_trade_fees = parallelised_optimal_trade_sifter(
        reserves,
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

    optimal_arb_trade = jnp.where(
        fees_are_being_charged,
        optimal_arb_trade_fees,
        optimal_arb_trade_zero_fees,
    )
    post_weight_reserves = reserves + optimal_arb_trade

    # apply noise trade if noise_trader_ratio is greater than 0
    post_weight_reserves = jnp.where(
        noise_trader_ratio > 0,
        calculate_reserves_after_noise_trade(
            optimal_arb_trade, post_weight_reserves, prices, noise_trader_ratio, gamma
        ),
        post_weight_reserves,
    )
    # check if this is worth the cost to arbs
    # delta = post_weight_reserves - reserves
    # is this delta a good deal for the arb?
    profit_to_arb = -(optimal_arb_trade * prices).sum() - arb_thresh
    profit_weights = profit_to_arb

    arb_external_rebalance_cost = (
        0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
    )

    arb_profitable = profit_to_arb >= arb_external_rebalance_cost

    # if arb trade IS profitable
    # then reserves is equal to post_weight_reserves, otherwise equal to the prior reserves
    do_weight_arb_trade = arb_profitable

    reserves = jnp.where(do_weight_arb_trade, post_weight_reserves, reserves)
    counter += 1
    return [
        weights,
        prices,
        reserves,
        counter,
    ], reserves


@partial(jit, static_argnums=(5, 6,))
def _jax_calc_quantAMM_reserves_with_dynamic_fees_and_trades_scan_function_using_precalcs(
    carry_list,
    input_list,
    all_sig_variations,
    tokens_to_drop,
    active_trade_directions,
    n,
    do_trades,
):
    """
    Calculate changes in AMM reserves considering fees and arbitrage opportunities using signature variations.

    This function extends the basic reserve calculation by incorporating transaction fees
    and potential arbitrage opportunities, following the methodology described in
    https://arxiv.org/abs/2402.06731. It tests different trade signature variations to determine
    optimal arbitrage trades.

    Parameters
    ----------
    carry_list : list
        List containing the previous weights, prices, and reserves.
    input_list : list
        List containing:
        weights : jnp.ndarray
            Array containing the current weights.
        prices : jnp.ndarray
            Array containing the current prices.
        active_initial_weights
        per_asset_ratios : jnp.array
            Array containing precalculated value
        all_other_assets_ratios : jnp.array
            Array containing precalculated value
        lagged_active_initial_weights : jnp.array
            Array containing precalculated value
        lagged_per_asset_ratios : jnp.array
            Array containing precalculated value
        lagged_all_other_assets_ratios : jnp.array
            Array containing precalculated value
        gamma: jnp.ndarray
            Array containing the AMM pool's 1-fees over time.
        arb_thresh: jnp.ndarray
            Array containing the arb's threshold for profitable arbitrage over time.
        arb_fees: jnp.ndarray
            Array containing the fees associated with arbitrage.
        trades: jnp.ndarray
            Array containing the indexs of the in and out tokens and the in amount for trades at each time.
        do_arb: jnp.ndarray
            Array containing whether or not to apply arbitrage at each time.
    all_sig_variations : jnp.ndarray
        Array of all signature variations used for arbitrage calculations.
    n : int
        Number of tokens or assets.
    do_trades : bool
        Whether or not to apply the trades

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves changes.
    """
    # NOTE: MAYBE THIS SHOULD BE DONE IN LOG SPACE?

    # arb_fees = 0.0002
    # carry_list[0] is previous weights
    prev_weights = carry_list[0]

    # carry_list[1] is previous prices
    prev_prices = carry_list[1]

    # carry_list[2] is previous reserves
    prev_reserves = carry_list[2]

    # prev_active_initial_weights = carry_list[3]

    # prev_per_asset_ratios = carry_list[4]

    # prev_all_other_assets_ratios = carry_list[5]

    counter = carry_list[3]

    # input_list contains weights, prices, precalcs and fee/arb amounts
    weights = input_list[0]
    prices = input_list[1]
    active_initial_weights = input_list[2]
    per_asset_ratios = input_list[3]
    all_other_assets_ratios = input_list[4]
    lagged_active_initial_weights = input_list[5]
    lagged_per_asset_ratios = input_list[6]
    lagged_all_other_assets_ratios = input_list[7]
    gamma = input_list[8]
    arb_thresh = input_list[9]
    arb_fees = input_list[10]
    trade = input_list[11]
    do_arb = input_list[12]
    fees_are_being_charged = gamma != 1.0

    current_value = (prev_reserves * prices).sum()
    quoted_prices = current_value * prev_weights / prev_reserves

    price_change_ratio = prices / quoted_prices
    price_product_change_ratio = jnp.prod(price_change_ratio**prev_weights)
    reserves_ratios_from_price_change = price_product_change_ratio / price_change_ratio

    post_price_reserves_zero_fees = prev_reserves * reserves_ratios_from_price_change

    # optimal_arb_trade_zero_fees = prev_reserves * (
    #     reserves_ratios_from_price_change - 1.0
    # )
    optimal_arb_trade_zero_fees = post_price_reserves_zero_fees - prev_reserves

    # optimal_arb_trade_fees = optimal_trade_sifter(
    #     prev_weights, prices, prev_reserves, gamma, all_sig_variations, n, 0
    # )
    optimal_arb_trade_fees = parallelised_optimal_trade_sifter(
        prev_reserves,
        prev_weights,
        prices,
        lagged_active_initial_weights,
        active_trade_directions,
        lagged_per_asset_ratios,
        lagged_all_other_assets_ratios,
        tokens_to_drop,
        gamma,
        n,
        0,
    )
    # og_optimal_arb_trade = optimal_trade_sifter(
    #     prev_weights, prices, prev_reserves, gamma, all_sig_variations, n, 0
    # )
    # if jnp.sum((optimal_arb_trade_fees - og_optimal_arb_trade) ** 2) > 0:
    #     raise Exception

    optimal_arb_trade = jnp.where(
        fees_are_being_charged,
        optimal_arb_trade_fees,
        optimal_arb_trade_zero_fees,
    )
    post_price_reserves = prev_reserves + optimal_arb_trade

    # check if this is worth the cost to arbs
    # delta = post_price_reserves - prev_reserves
    # is this delta a good deal for the arb?
    profit_to_arb = -(optimal_arb_trade * prices).sum() - arb_thresh
    profit_prices = profit_to_arb

    arb_external_rebalance_cost = (
        0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
    )

    arb_profitable = profit_to_arb >= arb_external_rebalance_cost

    # if arb trade IS profitable AND outside_no_arb_region IS true
    # then reserves is equal to post_price_reserves, otherwise equal to prev_reserves
    do_price_arb_trade = arb_profitable * do_arb

    reserves = jnp.where(do_price_arb_trade, post_price_reserves, prev_reserves)

    # apply trade if trade is present
    if do_trades:
        reserves += jitted_G3M_cond_trade(do_trades, reserves, weights, trade, gamma)
    current_value = (reserves * prices).sum()
    quoted_prices = current_value * weights / reserves

    # weight_change_ratio = weights / prev_weights
    # weight_product_change_ratio = jnp.prod(weight_change_ratio**weights)
    # reserves_ratio_from_weight_change = (
    #     weight_change_ratio / weight_product_change_ratio
    # )
    price_change_ratio = prices / quoted_prices
    price_product_change_ratio = jnp.prod(price_change_ratio**weights)
    reserves_ratios_from_weight_change = price_product_change_ratio / price_change_ratio

    post_weight_reserves_zero_fees = reserves_ratios_from_weight_change * reserves
    # optimal_arb_trade_zero_fees = reserves * (
    #     reserves_ratios_from_weight_change - 1.0
    # )
    optimal_arb_trade_zero_fees = post_weight_reserves_zero_fees - reserves

    # og_optimal_arb_trade_fees = optimal_trade_sifter(
    #     weights, prices, reserves, gamma, all_sig_variations, n, 0
    # )
    optimal_arb_trade_fees = parallelised_optimal_trade_sifter(
        reserves,
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

    # og_optimal_arb_trade = optimal_trade_sifter(
    #     prev_weights, prices, prev_reserves, gamma, all_sig_variations, n, 0
    # )
    # if jnp.sum((optimal_arb_trade_fees - og_optimal_arb_trade_fees) ** 2) > 0:
    #     raise Exception

    optimal_arb_trade = jnp.where(
        fees_are_being_charged,
        optimal_arb_trade_fees,
        optimal_arb_trade_zero_fees,
    )
    post_weight_reserves = reserves + optimal_arb_trade

    # check if this is worth the cost to arbs
    # delta = post_weight_reserves - reserves
    # is this delta a good deal for the arb?
    profit_to_arb = -(optimal_arb_trade * prices).sum() - arb_thresh
    profit_weights = profit_to_arb

    arb_external_rebalance_cost = (
        0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
    )

    arb_profitable = profit_to_arb >= arb_external_rebalance_cost

    # if arb trade IS profitable AND outside_no_arb_region IS true
    # then reserves is equal to post_weight_reserves, otherwise equal to prev_reserves
    do_weight_arb_trade = arb_profitable * do_arb

    reserves = jnp.where(do_weight_arb_trade, post_weight_reserves, reserves)
    counter += 1
    return [
        weights,
        prices,
        reserves,
        counter,
    ], reserves


@partial(jit, static_argnums=(8,))
def _jax_calc_quantAMM_reserves_with_dynamic_inputs(
    initial_reserves,
    weights,
    prices,
    fees,
    arb_thresh,
    arb_fees,
    all_sig_variations=None,
    trades=None,
    do_trades=False,
    do_arb=True,
):
    """
    Calculate AMM reserves considering fees and arbitrage opportunities using signature variations,
    using the approach given in https://arxiv.org/abs/2402.06731.

    This function computes the changes in reserves for an automated market maker (AMM) model
    considering dynamic transaction fees, dynamic arbitrage costs, external trades and
    potential arbitrage opportunities.
    It uses a scan operation to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    weights : jnp.ndarray
        Two-dimensional array of asset weights over time.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    fees : jnp.ndarray
        Swap fee charged on transactions, by default 0.003.
    arb_thresh : jnp.ndarray
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : jnp.ndarray
        Fees associated with arbitrage, by default 0.0.
    trades :  jnp.ndarray, optional
        array of trades for each timestep.
        format, for each row:
            trades[0] = index of token to trade in to pool
            trades[1] = index of token to trade out to pool
            trades[2] = amount of 'token in' to trade
    all_sig_variations : jnp.ndarray, optional
        Array of all signature variations used for arbitrage calculations.
    do_trades : bool, optional
        Whether or not to apply the trades, by default False
    do_arb : bool, optional
        Whether or not to apply arbitrage, by default True

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """
    # initial_weights = coarse_weights[0]
    # initial_i = 0
    n_assets = weights.shape[1]

    # We do this like a block, so first there is the new
    # weight value and THEN we get new prices by the end of
    # the block.

    # So, for first weight, we have initial reserves, weights and
    # prices, so the change is 1

    n = prices.shape[0]

    initial_prices = prices[0]

    initial_weights = weights[0]

    gamma = jnp.where(
        fees.size == 1, jnp.full(weights.shape[0], 1.0 - fees), 1.0 - fees
    )

    arb_thresh = jnp.where(
        arb_thresh.size == 1, jnp.full(weights.shape[0], arb_thresh), arb_thresh
    )

    arb_fees = jnp.where(
        arb_fees.size == 1, jnp.full(weights.shape[0], arb_fees), arb_fees
    )

    do_arb = jnp.where(
        isinstance(do_arb, bool) or do_arb.size == 1, jnp.full(weights.shape[0], do_arb), do_arb
    )

    # pre-calculate some values that are repeatedly used in optimal arb calculations

    array_of_trues = jnp.ones((n_assets,), dtype=bool)

    tokens_to_keep, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    # calculate values that can be done in parallel

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_weights_and_prices_and_dynamic_fees(
            weights,
            prices,
            gamma,
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idxs,
        )
    )

    (
        lagged_active_initial_weights,
        lagged_per_asset_ratios,
        lagged_all_other_assets_ratios,
    ) = precalc_components_of_optimal_trade_across_weights_and_prices_and_dynamic_fees(
        jnp.vstack(
            [
                weights[0],
                weights[:-1],
            ]
        ),
        prices,
        gamma,
        tokens_to_drop,
        active_trade_directions,
        leave_one_out_idxs,
    )

    scan_fn = Partial(
        _jax_calc_quantAMM_reserves_with_dynamic_fees_and_trades_scan_function_using_precalcs,
        n=n_assets,
        all_sig_variations=all_sig_variations,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        do_trades=do_trades,
        do_arb=do_arb,
    )

    carry_list_init = [
        initial_weights,
        initial_prices,
        initial_reserves,
        # active_initial_weights[0],
        # per_asset_ratios[0],
        # all_other_assets_ratios[0],
        0,
    ]
    # carry_list_init = [initial_weights, initial_i]
    # nojit_scan = jax.disable_jit()(jax.lax.scan)
    carry_list_end, reserves = scan(
        scan_fn,
        carry_list_init,
        [
            weights,
            prices,
            active_initial_weights,
            per_asset_ratios,
            all_other_assets_ratios,
            lagged_active_initial_weights,
            lagged_per_asset_ratios,
            lagged_all_other_assets_ratios,
            gamma,
            arb_thresh,
            arb_fees,
            trades,
            do_arb,
        ],
    )

    return reserves
