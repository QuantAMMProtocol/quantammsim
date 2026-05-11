from functools import partial

import jax.numpy as jnp

from jax import jit, vmap, devices
from jax.tree_util import Partial
from jax.lax import scan

from jax import default_backend

from quantammsim.pools.G3M.optimal_n_pool_arb import (
    precalc_shared_values_for_all_signatures,
    precalc_components_of_optimal_trade_across_prices,
    precalc_components_of_optimal_trade_across_prices_and_dynamic_fees,
    parallelised_optimal_trade_sifter,
)
from quantammsim.pools.G3M.G3M_trades import jitted_G3M_cond_trade
from quantammsim.pools.noise_trades import calculate_reserves_after_noise_trade
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    reclamm_mm_observed_noise_volume,
    reclamm_market_linear_noise_volume,
    reclamm_calibrated_noise_volume,
)


DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
else:
    GPU_DEVICE = devices("cpu")[0]


@jit
def _jax_calc_balancer_reserve_ratio(prev_prices, weights, prices):
    """
    Calculate reserves ratio changes for a single timestep.

    This function computes the changes in reserves for an vanilla G3M automated market maker (AMM)
    pool, so constant weights. Prices change. It is optimized for GPU execution.

    Parameters
    ----------
    prev_prices : jnp.ndarray
        Array of previous asset prices.
    weights : jnp.ndarray
        Array of weights.
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
    price_product_change_ratio = jnp.prod(price_change_ratio**weights)
    reserves_ratios_from_price_change = price_product_change_ratio / price_change_ratio
    return reserves_ratios_from_price_change


_jax_calc_balancer_reserve_ratios = jit(
    vmap(_jax_calc_balancer_reserve_ratio, in_axes=(0, None, 0))
)


@partial(jit, static_argnums=(5, 6))
def _jax_calc_balancer_reserves_with_fees_scan_function_using_precalcs(
    carry_list,
    prices_and_precalcs,
    weights,
    tokens_to_drop,
    active_trade_directions,
    n,
    noise_model="ratio",
    gamma=0.997,
    arb_thresh=0.0,
    arb_fees=0.0,
    noise_trader_ratio=0.0,
    protocol_fee_split=0.0,
    seconds_per_step=60.0,
    noise_params=None,
):
    """
    Calculate changes in AMM reserves considering fees 
    and arbitrage opportunities using signature variations.

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

    # carry_list[1] is previous reserves
    prev_reserves = carry_list[1]

    counter = carry_list[2]

    # prices_and_precalcs are the weights and prices, in that order
    # we have a lot of precalcs
    prices = prices_and_precalcs[0]
    active_initial_weights = prices_and_precalcs[1]
    per_asset_ratios = prices_and_precalcs[2]
    all_other_assets_ratios = prices_and_precalcs[3]

    fees_are_being_charged = gamma != 1.0

    current_value = (prev_reserves * prices).sum()
    quoted_prices = current_value * weights / prev_reserves

    price_change_ratio = prices / quoted_prices
    price_product_change_ratio = jnp.prod(price_change_ratio**weights)
    reserves_ratios_from_price_change = price_product_change_ratio / price_change_ratio

    post_price_reserves_zero_fees = prev_reserves * reserves_ratios_from_price_change

    optimal_arb_trade_zero_fees = post_price_reserves_zero_fees - prev_reserves

    optimal_arb_trade_fees = parallelised_optimal_trade_sifter(
        prev_reserves,
        weights,
        prices,
        active_initial_weights,
        active_trade_directions,
        per_asset_ratios,
        all_other_assets_ratios,
        tokens_to_drop,
        gamma,
        n,
        -1e-15,
    )

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

    arb_external_rebalance_cost = (
        0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
    )

    arb_profitable = profit_to_arb >= arb_external_rebalance_cost

    # if arb trade IS profitable
    # then reserves is equal to post_price_reserves, otherwise equal to prev_reserves
    do_price_arb_trade = arb_profitable

    reserves = jnp.where(do_price_arb_trade, post_price_reserves, prev_reserves)

    # --- Noise model dispatch (non-dynamic path) ---
    if noise_model == "ratio":
        if_noise = noise_trader_ratio > 0
        applied_trade = reserves - prev_reserves
        noisy = calculate_reserves_after_noise_trade(
            applied_trade, reserves, prices, noise_trader_ratio, gamma,
        )
        reserves = jnp.where(if_noise, noisy, reserves)
    elif noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        from quantammsim.pools.reCLAMM.reclamm_reserves import (
            reclamm_tsoukalas_sqrt_noise_volume,
            reclamm_tsoukalas_log_noise_volume,
            reclamm_loglinear_noise_volume,
        )
        volatility = prices_and_precalcs[4]
        arb_volume = 0.5 * jnp.sum(jnp.abs(reserves - prev_reserves) * prices)
        effective_value = (reserves * prices).sum()
        _np = noise_params if noise_params is not None else {}
        if noise_model == "tsoukalas_sqrt":
            noise_vol = reclamm_tsoukalas_sqrt_noise_volume(
                effective_value, gamma, volatility, arb_volume, _np)
        elif noise_model == "tsoukalas_log":
            noise_vol = reclamm_tsoukalas_log_noise_volume(
                effective_value, gamma, volatility, arb_volume, _np)
        else:
            noise_vol = reclamm_loglinear_noise_volume(
                effective_value, gamma, volatility, arb_volume, _np)
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_total = (1.0 - gamma) * noise_vol * minutes_per_step
        noise_fee_income = noise_fee_total * (1.0 - protocol_fee_split)
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        reserves = reserves * scale
    elif noise_model == "calibrated":
        volatility = prices_and_precalcs[4]
        dow_sin = prices_and_precalcs[5]
        dow_cos = prices_and_precalcs[6]
        arb_volume = 0.5 * jnp.sum(jnp.abs(reserves - prev_reserves) * prices)
        effective_value = (reserves * prices).sum()
        _np = noise_params if noise_params is not None else {}
        noise_vol = reclamm_calibrated_noise_volume(
            effective_value, gamma, volatility, arb_volume, dow_sin, dow_cos, _np)
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_total = (1.0 - gamma) * noise_vol * minutes_per_step
        noise_fee_income = noise_fee_total * (1.0 - protocol_fee_split)
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        reserves = reserves * scale
    elif noise_model == "market_linear":
        noise_base = prices_and_precalcs[4]
        noise_tvl_coeff = prices_and_precalcs[5]
        effective_value = (reserves * prices).sum()
        _np = noise_params if noise_params is not None else {}
        noise_vol = reclamm_market_linear_noise_volume(
            effective_value, noise_base, noise_tvl_coeff,
            tvl_mean=_np.get("tvl_mean", 0.0), tvl_std=_np.get("tvl_std", 1.0))
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_total = (1.0 - gamma) * noise_vol * minutes_per_step
        noise_fee_income = noise_fee_total * (1.0 - protocol_fee_split)
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        reserves = reserves * scale
    elif noise_model == "mm_observed":
        noise_base = prices_and_precalcs[4]
        competitor_tvl = prices_and_precalcs[5]
        effective_value = (reserves * prices).sum()
        noise_vol = reclamm_mm_observed_noise_volume(
            effective_value, noise_base, competitor_tvl)
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_total = (1.0 - gamma) * noise_vol * minutes_per_step
        noise_fee_income = noise_fee_total * (1.0 - protocol_fee_split)
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        reserves = reserves * scale

    counter += 1
    return [
        prices,
        reserves,
        counter,
    ], reserves


@partial(jit, static_argnums=(), static_argnames=("noise_model",))
def _jax_calc_balancer_reserves_with_fees_using_precalcs(
    initial_reserves,
    weights,
    prices,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    all_sig_variations=None,
    noise_model="ratio",
    noise_trader_ratio=0.0,
    protocol_fee_split=0.0,
    seconds_per_step=60.0,
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
    competitor_tvl_array=None,
):
    """
    Calculate AMM reserves considering fees and arbitrage opportunities using signature variations,
    using the approach given in https://arxiv.org/abs/2402.06731, for Balancer pools, pools with
    constant weights.

    This function computes the changes in reserves for an automated market maker (AMM) model
    considering transaction fees and potential arbitrage opportunities.
    It uses a scan operation to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    weights : jnp.ndarray
        One-dimensional array of asset weights.
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

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """
    n_assets = weights.shape[0]

    # or first weight, we have initial reserves, weights and
    # prices, so the change is 1

    initial_prices = prices[0]

    gamma = 1.0 - fees

    # pre-calculate some values that are repeatedly used in optimal arb calculations

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    # calculate values that can be done in parallel

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_prices(
            weights,
            prices,
            gamma,
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idxs,
        )
    )

    scan_fn = Partial(
        _jax_calc_balancer_reserves_with_fees_scan_function_using_precalcs,
        weights=weights,
        gamma=gamma,
        arb_thresh=arb_thresh,
        arb_fees=arb_fees,
        n=n_assets,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        noise_model=noise_model,
        noise_trader_ratio=noise_trader_ratio,
        protocol_fee_split=protocol_fee_split,
        seconds_per_step=seconds_per_step,
        noise_params=noise_params,
    )

    carry_list_init = [
        initial_prices,
        initial_reserves,
        0,
    ]

    scan_inputs = [
        prices,
        active_initial_weights,
        per_asset_ratios,
        all_other_assets_ratios,
    ]

    # Append noise arrays at position 4+ in prices_and_precalcs
    if noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        scan_inputs.append(volatility_array)
    elif noise_model == "calibrated":
        scan_inputs.append(volatility_array)
        scan_inputs.append(dow_sin_array)
        scan_inputs.append(dow_cos_array)
    elif noise_model == "market_linear":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(noise_tvl_coeff_array)
    elif noise_model == "mm_observed":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(competitor_tvl_array)

    _, reserves = scan(
        scan_fn,
        carry_list_init,
        scan_inputs,
    )

    return reserves


@partial(jit, static_argnums=(6, 7, 8, 9))
def _jax_calc_balancer_reserves_with_dynamic_fees_and_trades_scan_function_using_precalcs(
    carry_list,
    input_list,
    weights,
    all_sig_variations,
    tokens_to_drop,
    active_trade_directions,
    n,
    do_trades,
    do_arb,
    noise_model="ratio",
    protocol_fee_split=0.0,
    seconds_per_step=60.0,
    noise_params=None,
):
    """
    Calculate changes in AMM reserves considering fees 
    and arbitrage opportunities using signature variations.

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
        prices : jnp.ndarray
            Array containing the current prices.
        active_initial_weights
        per_asset_ratios : jnp.array
            Array containing precalculated value
        all_other_assets_ratios : jnp.array
            Array containing precalculated value
        gamma: jnp.ndarray
            Array containing the AMM pool's 1-fees over time.
        arb_thresh: jnp.ndarray
            Array containing the arb's threshold for profitable arbitrage over time.
        arb_fees: jnp.ndarray
            Array containing the fees associated with arbitrage.
        trades: jnp.ndarray
            Array containing the indexs of the in and out tokens and the in amount for trades at each time.
    weights : jnp.ndarray
        Array of asset weights.
    all_sig_variations : jnp.ndarray
        Array of all signature variations used for arbitrage calculations.
    tokens_to_drop : jnp.ndarray
        Array indicating which tokens should be excluded from arbitrage calculations.
    active_trade_directions : jnp.ndarray
        Array indicating which trade directions are allowed.
    n : int
        Number of tokens or assets.
    do_trades : bool
        Flag indicating whether to process trades.
    do_arb : bool 
        Flag indicating whether to process arbitrage.

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves changes.
    """

    # carry_list[1] is previous reserves
    prev_reserves = carry_list[1]

    counter = carry_list[2]
    prev_lp_supply = carry_list[3]

    # input_list contains weights, prices, precalcs and fee/arb amounts
    prices = input_list[0]
    active_initial_weights = input_list[1]
    per_asset_ratios = input_list[2]
    all_other_assets_ratios = input_list[3]
    gamma = input_list[4]
    arb_thresh = input_list[5]
    arb_fees = input_list[6]
    trade = input_list[7]
    lp_supply = input_list[8]

    # Scale reserves for LP supply changes (proportional deposits/withdrawals)
    lp_supply_change = lp_supply != prev_lp_supply
    prev_reserves = jnp.where(
        lp_supply_change,
        prev_reserves * lp_supply / prev_lp_supply,
        prev_reserves,
    )

    fees_are_being_charged = gamma != 1.0

    current_value = (prev_reserves * prices).sum()
    quoted_prices = current_value * weights / prev_reserves

    price_change_ratio = prices / quoted_prices
    price_product_change_ratio = jnp.prod(price_change_ratio**weights)
    reserves_ratios_from_price_change = price_product_change_ratio / price_change_ratio

    post_price_reserves_zero_fees = prev_reserves * reserves_ratios_from_price_change

    optimal_arb_trade_zero_fees = post_price_reserves_zero_fees - prev_reserves

    optimal_arb_trade_fees = parallelised_optimal_trade_sifter(
        prev_reserves,
        weights,
        prices,
        active_initial_weights,
        active_trade_directions,
        per_asset_ratios,
        all_other_assets_ratios,
        tokens_to_drop,
        gamma,
        n,
        -1e-15,
    )

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

    arb_external_rebalance_cost = (
        0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
    )

    arb_profitable = profit_to_arb >= arb_external_rebalance_cost

    # if arb trade IS profitable AND outside_no_arb_region IS true
    # then reserves is equal to post_price_reserves, otherwise equal to prev_reserves
    do_price_arb_trade = arb_profitable * do_arb

    reserves = jnp.where(do_price_arb_trade, post_price_reserves, prev_reserves)

    # --- Noise model dispatch ---
    # Same pattern as reclamm_reserves.py: input_list[9+] are noise arrays,
    # protocol_fee_split / seconds_per_step / noise_params are passed via Partial.
    noise_fee_income = jnp.asarray(0.0, dtype=prices.dtype)
    if noise_model == "ratio":
        noise_trader_ratio = input_list[9]
        if_noise = noise_trader_ratio > 0
        applied_trade = reserves - prev_reserves
        noisy = calculate_reserves_after_noise_trade(
            applied_trade, reserves, prices, noise_trader_ratio, gamma,
        )
        reserves = jnp.where(if_noise, noisy, reserves)
    elif noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        from quantammsim.pools.reCLAMM.reclamm_reserves import (
            reclamm_tsoukalas_sqrt_noise_volume,
            reclamm_tsoukalas_log_noise_volume,
            reclamm_loglinear_noise_volume,
        )
        volatility = input_list[9]
        arb_volume = 0.5 * jnp.sum(jnp.abs(reserves - prev_reserves) * prices)
        effective_value = (reserves * prices).sum()
        _np = noise_params if noise_params is not None else {}
        if noise_model == "tsoukalas_sqrt":
            noise_vol = reclamm_tsoukalas_sqrt_noise_volume(
                effective_value, gamma, volatility, arb_volume, _np)
        elif noise_model == "tsoukalas_log":
            noise_vol = reclamm_tsoukalas_log_noise_volume(
                effective_value, gamma, volatility, arb_volume, _np)
        else:
            noise_vol = reclamm_loglinear_noise_volume(
                effective_value, gamma, volatility, arb_volume, _np)
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_total = (1.0 - gamma) * noise_vol * minutes_per_step
        noise_fee_income = noise_fee_total * (1.0 - protocol_fee_split)
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        reserves = reserves * scale
    elif noise_model == "calibrated":
        volatility = input_list[9]
        dow_sin = input_list[10]
        dow_cos = input_list[11]
        arb_volume = 0.5 * jnp.sum(jnp.abs(reserves - prev_reserves) * prices)
        effective_value = (reserves * prices).sum()
        _np = noise_params if noise_params is not None else {}
        noise_vol = reclamm_calibrated_noise_volume(
            effective_value, gamma, volatility, arb_volume, dow_sin, dow_cos, _np)
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_total = (1.0 - gamma) * noise_vol * minutes_per_step
        noise_fee_income = noise_fee_total * (1.0 - protocol_fee_split)
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        reserves = reserves * scale
    elif noise_model == "market_linear":
        noise_base = input_list[9]
        noise_tvl_coeff = input_list[10]
        effective_value = (reserves * prices).sum()
        _np = noise_params if noise_params is not None else {}
        noise_vol = reclamm_market_linear_noise_volume(
            effective_value, noise_base, noise_tvl_coeff,
            tvl_mean=_np.get("tvl_mean", 0.0), tvl_std=_np.get("tvl_std", 1.0))
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_total = (1.0 - gamma) * noise_vol * minutes_per_step
        noise_fee_income = noise_fee_total * (1.0 - protocol_fee_split)
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        reserves = reserves * scale
    elif noise_model == "mm_observed":
        noise_base = input_list[9]
        competitor_tvl = input_list[10]
        effective_value = (reserves * prices).sum()
        noise_vol = reclamm_mm_observed_noise_volume(
            effective_value, noise_base, competitor_tvl)
        minutes_per_step = seconds_per_step / 60.0
        noise_fee_total = (1.0 - gamma) * noise_vol * minutes_per_step
        noise_fee_income = noise_fee_total * (1.0 - protocol_fee_split)
        scale = 1.0 + noise_fee_income / jnp.maximum(effective_value, 1e-8)
        reserves = reserves * scale

    # apply trade if trade is present
    if do_trades:
        reserves += jitted_G3M_cond_trade(do_trades, reserves, weights, trade, gamma)

    counter += 1
    return [
        prices,
        reserves,
        counter,
        lp_supply,
    ], reserves


@partial(jit, static_argnums=(8, 9, 10))
def _jax_calc_balancer_reserves_with_dynamic_inputs(
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
    noise_model="ratio",
    lp_supply_array=None,
    protocol_fee_split=0.0,
    noise_trader_ratio=0.0,
    seconds_per_step=60.0,
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
    competitor_tvl_array=None,
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
    fees : float / jnp.ndarray
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
        Whether or not to apply the arbitrage, by default True

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """
    n_assets = weights.shape[0]

    # or first weight, we have initial reserves, weights and
    # prices, so the change is 1

    initial_prices = prices[0]

    gamma = jnp.where(fees.size == 1, jnp.full(prices.shape[0], 1.0 - fees), 1.0 - fees)

    arb_thresh = jnp.where(
        arb_thresh.size == 1, jnp.full(prices.shape[0], arb_thresh), arb_thresh
    )

    arb_fees = jnp.where(
        arb_fees.size == 1, jnp.full(prices.shape[0], arb_fees), arb_fees
    )
    if do_trades and trades is None:
        raise ValueError("Trades must be provided when do_trades=True.")

    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(prices.shape[0], lp_supply_array),
        lp_supply_array,
    )

    # pre-calculate some values that are repeatedly used in optimal arb calculations
    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    # calculate values that can be done in parallel

    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_prices_and_dynamic_fees(
            weights,
            prices,
            gamma,
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idxs,
        )
    )
    scan_fn = Partial(
        _jax_calc_balancer_reserves_with_dynamic_fees_and_trades_scan_function_using_precalcs,
        weights=weights,
        n=n_assets,
        all_sig_variations=all_sig_variations,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        do_trades=do_trades,
        do_arb=do_arb,
        noise_model=noise_model,
        protocol_fee_split=protocol_fee_split,
        seconds_per_step=seconds_per_step,
        noise_params=noise_params,
    )

    carry_list_init = [
        initial_prices,
        initial_reserves,
        0,
        lp_supply_array[0],
    ]

    scan_inputs = [
        prices,
        active_initial_weights,
        per_asset_ratios,
        all_other_assets_ratios,
        gamma,
        arb_thresh,
        arb_fees,
        trades,
        lp_supply_array,
    ]

    # Append noise-model-specific arrays at position 9+
    # (same ordering as reclamm_reserves.py scan inputs)
    if noise_model == "ratio":
        ntr = jnp.broadcast_to(jnp.asarray(noise_trader_ratio), (prices.shape[0],))
        scan_inputs.append(ntr)
    elif noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        scan_inputs.append(volatility_array)
    elif noise_model == "calibrated":
        scan_inputs.append(volatility_array)
        scan_inputs.append(dow_sin_array)
        scan_inputs.append(dow_cos_array)
    elif noise_model == "market_linear":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(noise_tvl_coeff_array)
    elif noise_model == "mm_observed":
        scan_inputs.append(noise_base_array)
        scan_inputs.append(competitor_tvl_array)

    _, reserves = scan(
        scan_fn,
        carry_list_init,
        scan_inputs,
    )

    return reserves
