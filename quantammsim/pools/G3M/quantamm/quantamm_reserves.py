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
    price_product_change_ratio = jnp.exp(jnp.sum(prev_weights * jnp.log(price_change_ratio)))
    reserves_ratios_from_price_change = price_product_change_ratio / price_change_ratio
    # second do part of change in reserves that corresponds to
    # Weights Change, Prices Constant, appendix B2of QuantAMM whitepaper
    # this is, in effect, at start of a block where we are quoting
    # new weights, creating an arb opportunity
    weight_change_ratio = weights / prev_weights
    weight_product_change_ratio = jnp.exp(jnp.sum(weights * jnp.log(weight_change_ratio)))
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
    price_product_change_ratio = jnp.exp(jnp.sum(prev_weights * jnp.log(price_change_ratio)))
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
    price_product_change_ratio = jnp.exp(jnp.sum(weights * jnp.log(price_change_ratio)))
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
    noise_trader_ratio=0.0,
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
    noise_trader_ratio : float, optional
        Ratio of noise traders to signal traders, by default 0.0

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves changes.
    """
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
    prev_lp_supply = carry_list[4]

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
    lp_supply = input_list[13]
    fees_are_being_charged = gamma != 1.0

    # if there has been a change in lp supply, we need to update the reserves
    # by the ratio of the new lp supply to the old lp supply
    # this assumes that all deposits and withdrawals are done 'proportionally'
    # meaning that the ratio of the new lp supply to the old lp supply is the
    # same as the ratio of the new reserves to the old reserves. This is a
    # conservative assumption, as the pool actually benefits from unbalanced
    # deposits and withdrawals.

    lp_supply_change = lp_supply != prev_lp_supply
    prev_reserves = jnp.where(lp_supply_change, prev_reserves * lp_supply / prev_lp_supply, prev_reserves)
    prev_lp_supply = lp_supply

    current_value = (prev_reserves * prices).sum()
    quoted_prices = current_value * prev_weights / prev_reserves

    price_change_ratio = prices / quoted_prices
    price_product_change_ratio = jnp.exp(jnp.sum(prev_weights * jnp.log(price_change_ratio)))
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
    price_product_change_ratio = jnp.exp(jnp.sum(weights * jnp.log(price_change_ratio)))
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
        lp_supply,
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
    noise_trader_ratio=0.0,
    lp_supply_array=None,
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
    noise_trader_ratio : float, optional
        Ratio of noise traders to signal traders, by default 0.0
    lp_supply_array : jnp.ndarray, optional
        Array of LP token supply over time, by default None

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

    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1, jnp.full(weights.shape[0], lp_supply_array), lp_supply_array
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
        # do_arb=do_arb,
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
        lp_supply_array[0]
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
            lp_supply_array,
        ],
    )

    return reserves


# ============================================================================
# Fused chunked reserve computation
# ============================================================================


def _intra_chunk_ratio_product(actual_start, scaled_diff, chunk_prices,
                               interpol_num, chunk_period, interpolation_fn):
    """Per-chunk: interpolate weights, compute ratios, return product.

    This is the inner kernel of the fused path.  It materialises a
    ``(chunk_period, n_assets)`` weight block, computes ``chunk_period - 1``
    reserve ratios, and returns their product — a single ``(n_assets,)``
    vector.  The intermediates are local to this call and never coexist
    across chunks, achieving the memory reduction.

    Parameters
    ----------
    actual_start : (n_assets,)
    scaled_diff : (n_assets,)
    chunk_prices : (chunk_period, n_assets)
    interpol_num : int
    chunk_period : int
    interpolation_fn : callable
        Maps (actual_start, scaled_diff, interpol_arange, fine_ones, interpol_num)
        → (chunk_period, n_assets) fine weights.

    Returns
    -------
    intra_product : (n_assets,) — product of intra-chunk reserve ratios
    first_weight : (n_assets,) — first fine weight in this chunk
    last_weight : (n_assets,) — last fine weight in this chunk
    """
    n_assets = actual_start.shape[0]
    interpol_arange = jnp.expand_dims(jnp.arange(interpol_num), 1)
    fine_ones = jnp.ones((chunk_period, n_assets))

    fine_weights = interpolation_fn(
        actual_start, scaled_diff, interpol_arange, fine_ones, interpol_num,
    )
    # fine_weights: (chunk_period, n_assets)

    # Intra-chunk ratios (chunk_period - 1 transitions)
    ratios = _jax_calc_quantAMM_reserve_ratios(
        fine_weights[:-1], chunk_prices[:-1],
        fine_weights[1:], chunk_prices[1:],
    )
    # (chunk_period - 1, n_assets)
    intra_product = jnp.prod(ratios, axis=0)
    return intra_product, fine_weights[0], fine_weights[-1]


@partial(jit, static_argnums=(5, 6, 7, 8, 9, 10, 11))
def _fused_chunked_reserves(
    actual_starts, scaled_diffs, local_prices, initial_reserves,
    initial_weights,
    chunk_period, interpol_num, metric_period,
    interpolation_fn, rule_outputs_are_weights,
    n_chunks_total, n_metric_periods,
):
    """Fused chunked reserve computation — fully vectorised (no scans).

    Computes metric-cadence boundary values matching ``values[::metric_period]``
    from the full-resolution path, without materialising the full
    ``(T_fine, n_assets)`` weight or reserve arrays.

    The fine-weight pipeline produces exactly ``chunk_period`` fine weights
    per coarse interval (the ``interpol_num``-th ramp endpoint is computed
    but dropped by the interpolation function).  Consecutive blocks are
    separated by exactly one ``scaled_diff`` step, so blocks align perfectly
    with the daily grid.

    Each metric period of ``metric_period`` fine steps decomposes into
    ``chunks_per_metric`` chunks, each contributing ``chunk_period - 1``
    intra-transitions + 1 boundary transition = ``chunk_period`` transitions.

    Algorithm (no ``lax.scan``):
      1. Compute per-chunk intra products via ``vmap``  (embarrassingly parallel).
      2. Compute per-chunk boundary ratios via ``vmap`` (embarrassingly parallel).
      3. Combine:  ``chunk_ratio[k] = intra[k] * boundary[k]``.
      4. Group into metric periods, product over ``chunks_per_metric``.
      5. ``cumprod`` over metric periods → cumulative reserve ratios.
      6. Evaluate boundary values at ``prices[k * metric_period]``.

    Parameters
    ----------
    actual_starts : (n_coarse_for_rules, n_assets)
        Coarse weight start positions.  Includes one extra entry beyond
        what is needed for intra products, providing the start weight
        for the final boundary transition.
    scaled_diffs : (n_coarse_for_rules, n_assets)
        Per-step weight increments (only the first ``n_coarse_for_intra``
        entries are used for intra products).
    local_prices : (T_fine, n_assets)
        Bout prices at minute resolution.
    initial_reserves : (n_assets,)
    initial_weights : (n_assets,)
    chunk_period : int
    interpol_num : int
    metric_period : int
    interpolation_fn : callable
    rule_outputs_are_weights : bool
    n_chunks_total : int
        Number of chunks (including virtual for delta pools).
    n_metric_periods : int

    Returns
    -------
    boundary_values : (n_metric_periods + 1,)
    final_reserves : (n_assets,)
    """
    n_assets = initial_weights.shape[0]
    chunks_per_metric = metric_period // chunk_period

    # --- Step 1: Build per-chunk data arrays ---
    # All chunks are laid out as: local_prices[k*cp : (k+1)*cp] for chunk k.
    # For delta pools, chunk 0 = virtual (initial weights), chunk 1..N = coarse 0..N-1.
    # For target pools, chunk 0..N-1 = coarse 0..N-1.
    all_chunk_prices = local_prices[:n_chunks_total * chunk_period].reshape(
        n_chunks_total, chunk_period, n_assets
    )

    if not rule_outputs_are_weights:
        # Delta pool: prepend virtual chunk (constant initial_weights)
        n_coarse_for_intra = n_chunks_total - 1
        intra_starts = jnp.concatenate(
            [initial_weights[None, :], actual_starts[:n_coarse_for_intra]], axis=0
        )
        intra_diffs = jnp.concatenate(
            [jnp.zeros((1, n_assets)), scaled_diffs[:n_coarse_for_intra]], axis=0
        )
        # Boundary "next" weights: chunk k+1 = coarse k → actual_starts[k]
        next_start_weights = actual_starts[:n_chunks_total]
    else:
        # Target pool: all chunks are coarse
        n_coarse_for_intra = n_chunks_total
        intra_starts = actual_starts[:n_coarse_for_intra]
        intra_diffs = scaled_diffs[:n_coarse_for_intra]
        # Boundary "next" weights: chunk k+1 = coarse k+1 → actual_starts[k+1]
        next_start_weights = actual_starts[1:n_chunks_total + 1]

    # --- Step 2: Per-chunk intra products (embarrassingly parallel) ---
    _intra_fn = partial(
        _intra_chunk_ratio_product,
        interpol_num=interpol_num,
        chunk_period=chunk_period,
        interpolation_fn=interpolation_fn,
    )
    all_intra_products, _, all_end_weights = vmap(_intra_fn)(
        intra_starts, intra_diffs, all_chunk_prices,
    )
    # all_intra_products: (n_chunks_total, n_assets) — product of chunk_period-1 ratios
    # all_end_weights: (n_chunks_total, n_assets) — last fine weight of each chunk

    # --- Step 3: Per-chunk boundary ratios (embarrassingly parallel) ---
    # Boundary k: from end of chunk k to start of chunk k+1
    #   prev_w = all_end_weights[k], prev_p = all_chunk_prices[k, -1]
    #   next_w = next_start_weights[k], next_p = local_prices[(k+1)*chunk_period]
    boundary_end_prices = all_chunk_prices[:, -1, :]  # (n_chunks_total, n_assets)
    next_start_price_indices = jnp.arange(1, n_chunks_total + 1) * chunk_period
    next_start_prices = local_prices[next_start_price_indices]  # (n_chunks_total, n_assets)

    boundary_ratios = _jax_calc_quantAMM_reserve_ratios(
        all_end_weights, boundary_end_prices,
        next_start_weights, next_start_prices,
    )
    # (n_chunks_total, n_assets)

    # --- Step 4: Combine intra + boundary per chunk ---
    # chunk_ratio[k] = intra[k] * boundary[k]
    # This covers chunk_period transitions: (chunk_period-1) intra + 1 boundary
    chunk_ratios = all_intra_products * boundary_ratios
    # (n_chunks_total, n_assets)

    # --- Step 5: Group into metric periods and take product ---
    metric_ratios = chunk_ratios.reshape(n_metric_periods, chunks_per_metric, n_assets)
    period_ratios = jnp.prod(metric_ratios, axis=1)
    # (n_metric_periods, n_assets)

    # --- Step 6: Cumprod over metric periods ---
    cum_ratios = jnp.cumprod(period_ratios, axis=0)
    # (n_metric_periods, n_assets)

    boundary_reserves = initial_reserves * cum_ratios
    # (n_metric_periods, n_assets)

    # --- Step 7: Evaluate boundary values ---
    # Value at metric boundary k (for k=1..n_metric_periods) is at
    # local_prices[k * metric_period], which is the start of the next period.
    metric_price_indices = jnp.arange(1, n_metric_periods + 1) * metric_period
    metric_boundary_prices = local_prices[metric_price_indices]
    # (n_metric_periods, n_assets)

    boundary_values_after = jnp.sum(boundary_reserves * metric_boundary_prices, axis=1)
    # (n_metric_periods,)

    initial_value = jnp.sum(initial_reserves * local_prices[0])
    boundary_values = jnp.concatenate([initial_value[None], boundary_values_after])
    # (n_metric_periods + 1,)

    final_reserves = boundary_reserves[-1]

    return boundary_values, final_reserves
