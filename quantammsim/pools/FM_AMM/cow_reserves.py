"""Reserve calculations for CoW AMM (FM-AMM) pools.

Provides JAX-based functions for computing CoW AMM reserves over time,
following the FM-AMM paper. Includes perfect-arbitrage and single-
arbitrageur rebalancing models, with support for fees, arbitrage
thresholds, noise traders, and dynamic inputs via ``jax.lax.scan``.
"""
from functools import partial

import math
from typing import List, Callable
import numpy as np


# again, this only works on startup!
from jax import jit, vmap
import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import Partial


from quantammsim.pools.FM_AMM.FMAMM_trades import jitted_FMAMM_cond_trade
from quantammsim.pools.noise_trades import calculate_reserves_after_noise_trade

np.seterr(all="raise")
np.seterr(under="print")


def align_CPAMM(reserves: List[float], price: float) -> List[float]:
    """
    Calculate the new CPAMM reserves at the align price.

    Args:
        reserves (List[float]): A list of two values representing the current reserves of the CPAMM.
        price (float): The new price to align the reserves with.

    Returns:
        List[float]: A list of two values representing
        the new reserves of the CPAMM after the alignment.
    """

    # Calculate new reserves according to the constant product formula
    align_reserves = [
        math.sqrt(reserves[0] * reserves[1]) / math.sqrt(price),
        math.sqrt(reserves[0] * reserves[1]) * math.sqrt(price),
    ]

    return align_reserves


def align_FMAMM(reserves: List[float], price: float) -> List[float]:
    """
    Calculate the new reserves of a FMAMM based on the align price.

    Args:
        reserves (List[float]): A list of two values representing the current reserves of FMAMM.
        price (float): The new price to align the reserves with.

    Returns:
        List[float]: A list of two values representing
        the new reserves of FMAMM after the alignment.
    """

    # Calculate new reserves s.t. the rebalancing price
    # and the final reserve ratio equal the new price
    align_reserves = [
        0.5 * reserves[0] + 0.5 * reserves[1] / price,
        0.5 * reserves[0] * price + 0.5 * reserves[1],
    ]

    return align_reserves


def align_position(
    reserves: List[float],
    price: float,
    align_function: Callable,
    amm_fee: float = 0.0,
    ext_fee: float = 0.0,
    ext_thresh: float = 0.0,
) -> List[float]:
    """
    Return new AMM position reserves the it has been aligned
    to a new price through an arbitrage trade.

    Args:
        reserves (List[float]):
            A list of two values representing the current reserves of the AMM.
        price (float): The new price to align the reserves with:
            Either a single value or a list of two values representing the bid and ask price.
        align_function (callable):
            A function used to calculate the new reserves based on the align price.
        amm_fee (float, optional):
            The trading fee that the AMM charges. Default is 0.0.
        ext_fee (float, optional):
            An additional external fee the arbitrageur needs to pay
            (could e.g. a fee on the external exchange). Default is 0.0.

    Returns:
        List[float]: A list of two values representing the new reserves of AMM after the alignment.
    """

    # if price is a tuple, calculate the average price
    if isinstance(price, tuple):
        bid_price = price[0]
        ask_price = price[1]
    else:
        bid_price = ask_price = price

    current_price = reserves[1] / reserves[0]

    # Check if the current reserves are already within the no-arbitrage price range
    if bid_price * (1 - amm_fee) * (
        1 - ext_fee
    ) <= current_price and current_price <= ask_price / (1 - amm_fee) / (1 - ext_fee):
        return reserves

    # If the current reserves are below the no-arbitrage price range, calculate the align price
    if current_price < bid_price * (1 - amm_fee) * (1 - ext_fee):
        align_price = bid_price * (1 - amm_fee) * (1 - ext_fee)

    # If the current reserves are above the no-arbitrage price range, calculate the align price
    if ask_price / (1 - amm_fee) / (1 - ext_fee) < current_price:
        align_price = ask_price / (1 - amm_fee) / (1 - ext_fee)

    # Calculate the new reserves based on the align price using the provided align_function

    # check if align_price is defined
    if "align_price" not in locals():
        print(current_price, price, amm_fee, ext_fee)

    if align_price == 0.0:
        return reserves

    align_reserves = align_function(reserves, align_price)

    # Add trading fees paid to reserves
    for i in [0, 1]:
        align_reserves[i] += (
            max(0.0, align_reserves[i] - reserves[i]) * amm_fee / (1 - amm_fee)
        )

    return align_reserves


@jit
def align_FMAMM_jax(reserves, price):
    """
    Calculate the new reserves of a FMAMM based on the align price using JAX. This function
    assumes weights of 0.5 for both tokens.

    Args:
        reserves (jnp.ndarray):
            A JAX array of two values representing the current reserves of FMAMM.
        price (jnp.ndarray): The new price to align the reserves with.

    Returns:
        jnp.ndarray: A JAX array of two values representing
            the new reserves of FMAMM after the alignment.
    """
    align_reserves = jnp.array(
        [
            0.5 * reserves[0] + 0.5 * reserves[1] / price,
            0.5 * reserves[0] * price + 0.5 * reserves[1],
        ]
    )
    return align_reserves


@jit
def align_FMAMM_onearb_jax(reserves, price):
    """
    Calculate the new reserves of a FMAMM based on results in sec. 6 of the FM-AMM paper, using JAX.
    This function assumes weights of 0.5 for both tokens.

    Args:
        reserves (jnp.ndarray):
            A JAX array of two values representing the current reserves of FMAMM.
            price (jnp.ndarray): The new price to align the reserves with.

    Returns:
        jnp.ndarray: A JAX array of two values representing
            the new reserves of FMAMM after the alignment.
    """
    constant = reserves[0] * reserves[1]
    align_reserves = jnp.array(
        [
            0.5 * reserves[0] + 0.5 * jnp.sqrt(constant / price),
            0.5 * jnp.sqrt(constant * price) + 0.5 * reserves[1],
        ]
    )
    return align_reserves


@jit
def _jax_calc_cowamm_reserves_with_fees_scan_function(
    carry_list, prices, weights, gamma=0.997, arb_thresh=0.0, arb_fees=0.0, noise_trader_ratio=0.0
):
    """
    Calculate changes in COW AMM (FM-AMM) reserves considering fees and arbitrage opportunities.

    This function extends the basic reserve calculation by incorporating transaction fees
    and potential arbitrage opportunities, following the methodology described in the
    FM-AMM paper.

    Parameters
    ----------
    carry_list : list
        List containing the previous prices and reserves.
    prices : jnp.ndarray
        Array containing the current prices.
    weights : jnp.ndarray
        Array containing the weights of the pool, assumed to be of shape (2,) and sum to 1.
    gamma : float, optional
        Fee factor for no-arbitrage bounds, by default 0.997.
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.
    noise_trader_ratio : float, optional
        Ratio of noise traders to signal traders, by default 0.0

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves.
    """
    # carry_list[1] is previous reserves
    prev_reserves = carry_list[1]

    # first find quoted price
    current_price = (prev_reserves[1] / prev_reserves[0]) * (weights[0] / weights[1])

    # envelope of no arb region
    ### see if prices are out of envelope
    bid_price = ask_price = prices[0] / prices[1]
    align_price_if_current_price_below_bid = bid_price * gamma * (1 - arb_fees)
    align_price_if_current_price_above_ask = ask_price / gamma / (1 - arb_fees)
    current_price_below_bid = current_price <= align_price_if_current_price_below_bid
    current_price_above_ask = align_price_if_current_price_above_ask <= current_price
    outside_no_arb_region = jnp.any(current_price_below_bid | current_price_above_ask)

    # now construct new values of reserves, profit, for IF
    # we are out of region -- and then use jnp where function to 'paste in'
    # values if outside_no_arb_region is True and trade is profitable to arb
    # We have to handle two cases here, for if prices are above or below the
    # no-arb region

    align_price = jnp.where(
        current_price_below_bid,
        align_price_if_current_price_below_bid,
        current_price,
    )
    align_price = jnp.where(
        current_price_above_ask,
        align_price_if_current_price_above_ask,
        align_price,
    )
    # calc align reserves including added trading fees paid to reserves
    align_reserves = (
        align_FMAMM_jax(prev_reserves * weights[::-1] * 2.0, align_price)
        - prev_reserves * weights[::-1] * 2.0
        + prev_reserves
    )
    align_reserves += (
        jnp.maximum(0, align_reserves - prev_reserves) * (1 - gamma) / gamma
    )

    align_reserves = jnp.where(
        noise_trader_ratio > 0,
        calculate_reserves_after_noise_trade(
            align_reserves - prev_reserves,
            align_reserves,
            prices,
            noise_trader_ratio,
            gamma,
        ),
        align_reserves,
    )

    reserves = jnp.where(outside_no_arb_region, align_reserves, prev_reserves)
    return [prices, reserves], reserves


@jit
def _jax_calc_cowamm_reserves_with_fees(
    initial_reserves, prices, weight=0.5, fees=0.003, arb_thresh=0.0, arb_fees=0.0, noise_trader_ratio=0.0
):
    """
    Calculate AMM reserves considering fees for CowAMM where weights
    (which can be other than 50-50) are used.

    This function computes the changes in reserves for a CowAMM that takes into account
    transaction fees and potential arbitrage opportunities. It uses a scan operation
    to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    weight : float
        Weight of the first token (must be between 0 and 1, assumes second token's weight is one minus this). Defaults to 0.5 (equal weights).
    fees : float, optional
        Swap fee charged on transactions, by default 0.003.
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.
    noise_trader_ratio : float, optional
        Ratio of noise traders to signal traders, by default 0.0

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """

    n_assets = prices.shape[1]
    assert n_assets == 2


    initial_prices = prices[0]

    gamma = 1.0 - fees

    weights = jnp.array([weight, 1 - weight])

    scan_fn = Partial(
        _jax_calc_cowamm_reserves_with_fees_scan_function,
        weights=weights,
        gamma=gamma,
        arb_thresh=arb_thresh,
        arb_fees=arb_fees,
        noise_trader_ratio=noise_trader_ratio,
    )

    carry_list_init = [initial_prices, initial_reserves]
    carry_list_end, reserves = scan(scan_fn, carry_list_init, prices)

    return reserves


# @jit
def _jax_calc_cowamm_reserves_one_arb_with_fees_scan_function(
    carry_list,
    prices,
    weight=0.5,
    gamma=0.997,
    arb_thresh=0.0,
    arb_fees=0.0,
    noise_trader_ratio=0.0
):
    """
    Calculate changes in COW AMM (FM-AMM) reserves considering fees and arbitrage opportunities.

    This function extends the basic reserve calculation by incorporating transaction fees
    and potential arbitrage opportunities, following the methodology described in the
    FM-AMM paper.

    Parameters
    ----------
    carry_list : list
        List containing the previous prices and reserves.
    prices : jnp.ndarray
        Array containing the current prices.
    weight : float, optional
        Weight of the first token (must be between 0 and 1, assumes second token's weight is one minus this). Defaults to 0.5 (equal weights).
    gamma : float, optional
        Fee factor for no-arbitrage bounds, by default 0.997.
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.
    noise_trader_ratio : float, optional
        Ratio of noise traders to signal traders, by default 0.0

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves.
    """
    # NOTE: MAYBE THIS SHOULD BE DONE IN LOG SPACE?

    # carry_list[1] is previous reserves
    prev_reserves = carry_list[1]

    prev_product = prev_reserves[0] * prev_reserves[1]

    scalar_price = prices[0] / prices[1]

    # calculate the two possible trades:
    # case 1:
    # change in reserves[0] > 0
    # change in reserves[1] < 0

    # case 2:
    # change in reserves[0] < 0
    # change in reserves[1] > 0
    product_w = (1.0 - weight) * weight
    delta_reserves_0_case1 = (1.0 - weight) * prev_reserves[0] - (
        jnp.sqrt(product_w * prev_product / (gamma * scalar_price))
    )
    delta_reserves_1_case1 = weight * prev_reserves[1] / gamma - (
        jnp.sqrt(product_w * prev_product * scalar_price / (gamma))
    )

    delta_reserves_0_case2 = (1.0 - weight) * prev_reserves[0] / gamma - (
        jnp.sqrt(product_w * prev_product / (gamma * scalar_price))
    )
    delta_reserves_1_case2 = weight * prev_reserves[1] - (
        jnp.sqrt(product_w * prev_product * scalar_price / (gamma))
    )

    overall_trades = jnp.array(
        [
            [-delta_reserves_0_case1, -delta_reserves_1_case1],
            [-delta_reserves_0_case2, -delta_reserves_1_case2],
        ]
    )
    profits = -(overall_trades * prices).sum(-1)

    mask = jnp.zeros_like(profits)
    # of the two profits, take the minimum
    # this is counterintuitive, but it is correct
    # as the other trade/profit corresponds to a trade that is not
    # actually possible/acceptable.
    correct_profit = jnp.min(profits)

    mask = jnp.where(profits == correct_profit, 1.0, 0.0)
    delta = mask @ overall_trades
    align_reserves = prev_reserves + delta

    align_reserves = jnp.where(
        noise_trader_ratio > 0,
        calculate_reserves_after_noise_trade(
            delta, align_reserves, prices, noise_trader_ratio, gamma
        ),
        align_reserves,
    )

    # check if this is worth the cost to arbs
    # is this delta a good deal for the arb?
    profit_to_arb = correct_profit - arb_thresh

    arb_external_rebalance_cost = 0.5 * arb_fees * (jnp.abs(delta) * prices).sum()
    arb_profitable = profit_to_arb >= arb_external_rebalance_cost
    reserves = jnp.where(arb_profitable, align_reserves, prev_reserves)

    return [prices, reserves], reserves


@jit
def _jax_calc_cowamm_reserves_one_arb_with_fees(
    initial_reserves, prices, weight=0.5, fees=0.003, arb_thresh=0.0, arb_fees=0.0, noise_trader_ratio=0.0
):
    """
    Calculate AMM reserves considering fees for CowAMM

    This function computes the changes in reserves for a CowAMM that takes into account
    transaction fees and potential arbitrage opportunities. It uses a scan operation
    to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    weight : float, optional
        Weight of the first token (must be between 0 and 1, assumes second token's weight is
        one minus this). Defaults to 0.5 (equal weights).
    fees : float, optional
        Swap fee charged on transactions, by default 0.003.
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.
    noise_trader_ratio : float, optional
        Ratio of noise traders to signal traders, by default 0.0

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """

    n_assets = prices.shape[1]
    assert n_assets == 2


    initial_prices = prices[0]

    gamma = 1.0 - fees

    scan_fn = Partial(
        _jax_calc_cowamm_reserves_one_arb_with_fees_scan_function,
        weight=weight,
        gamma=gamma,
        arb_thresh=arb_thresh,
        arb_fees=arb_fees,
        noise_trader_ratio=noise_trader_ratio,
    )

    carry_list_init = [initial_prices, initial_reserves]
    carry_list_end, reserves = scan(scan_fn, carry_list_init, prices)

    return reserves


@jit
def _jax_calc_cowamm_reserves_one_arb_zero_fees_scan_function(
    carry_list,
    prices,
    weight=0.5,
    arb_thresh=0.0,
    arb_fees=0.0,
):
    """
    Calculate changes in COW AMM (FM-AMM) reserves considering fees and arbitrage opportunities.

    This function extends the basic reserve calculation by incorporating transaction fees
    and potential arbitrage opportunities, following the methodology described in the
    FM-AMM paper.

    Parameters
    ----------
    carry_list : list
        List containing the previous reserves.
    prices : jnp.ndarray
        Array containing the current prices.
    weight : float, optional
        Weight of the first token (must be between 0 and 1, assumes second token's weight is
        one minus this). Defaults to 0.5 (equal weights).
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves.
    """
    # NOTE: MAYBE THIS SHOULD BE DONE IN LOG SPACE?

    # carry_list[0] is previous reserves
    prev_reserves = carry_list[0]

    # first find quoted price

    prev_product = prev_reserves[0] * prev_reserves[1]

    scalar_price = prices[0] / prices[1]
    product_w = weight * (1.0 - weight)
    delta_reserves_0 = (1.0 - weight) * prev_reserves[0] - (
        jnp.sqrt(product_w * prev_product / (scalar_price))
    )
    delta_reserves_1 = weight * prev_reserves[1] - (
        jnp.sqrt(product_w * prev_product * scalar_price)
    )

    # Rather than explicitly checking no-arb bounds, we calculate the optimal trade and check
    # if it's profitable. If so, execute it. Otherwise reserves stay unchanged.
    overall_trade = jnp.array([-delta_reserves_0, -delta_reserves_1])
    profit = -(overall_trade * prices).sum(-1)

    align_reserves = prev_reserves + overall_trade
    # check if this is worth the cost to arbs
    # is this delta a good deal for the arb?
    profit_to_arb = profit - arb_thresh

    arb_external_rebalance_cost = (
        0.5 * arb_fees * (jnp.abs(overall_trade) * prices).sum()
    )
    arb_profitable = profit_to_arb >= arb_external_rebalance_cost
    reserves = jnp.where(arb_profitable, align_reserves, prev_reserves)
    return [reserves], reserves


@jit
def _jax_calc_cowamm_reserves_one_arb_zero_fees(
    initial_reserves, prices, weight=0.5, arb_thresh=0.0, arb_fees=0.0
):
    """
    Calculate AMM reserves considering fees for CowAMM

    This function computes the changes in reserves for a CowAMM that takes into account
    transaction fees and potential arbitrage opportunities. It uses a scan operation
    to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    weight : float, optional
        Weight of the first token (must be between 0 and 1, assumes second token's weight is
        one minus this). Defaults to 0.5 (equal weights).
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """

    n_assets = prices.shape[1]
    assert n_assets == 2


    scan_fn = Partial(
        _jax_calc_cowamm_reserves_one_arb_zero_fees_scan_function,
        weight=weight,
        arb_thresh=arb_thresh,
        arb_fees=arb_fees,
    )

    carry_list_init = [initial_reserves]
    carry_list_end, reserves = scan(scan_fn, carry_list_init, prices)

    return reserves


@jit
def _jax_calc_cowamm_reserve_ratio(prev_prices, prices):
    """
    Calculate reserves ratio changes for a single timestep.

    This function computes the changes in reserves for an COW automated market maker
    (aka FM-AMM) based on a single change in asset prices. It is optimized for GPU execution.

    This function assumes equal weights (0.5) for both assets in the pool. For unequal weights,
    use _jax_calc_cowamm_reserves_with_weights_with_fees instead with fees=0.0.

    Parameters
    ----------
    prev_prices : jnp.ndarray
        Array of previous asset prices.
    prices : jnp.ndarray
        Array of current asset prices.

    Returns
    -------
    jnp.ndarray
        Array of reserves ratio changes.
    """
    price = prices[0] / prices[1]
    old_price = prev_prices[0] / prev_prices[1]
    price_ratio = jnp.array([old_price / price, price / old_price], dtype=jnp.float64)
    return 0.5 * (1 + price_ratio)


_jax_calc_cowamm_reserve_ratio_vmapped = vmap(_jax_calc_cowamm_reserve_ratio)


@partial(
    jit,
    static_argnums=(
        4,
        5,
    ),
)
def _jax_calc_cowamm_reserves_with_dynamic_fees_and_trades_scan_function(
    carry_list, input_list, arb_quality, weights, do_trades=True, do_arb=True, noise_trader_ratio=0.0
):
    """
    Calculate changes in COW AMM (FM-AMM) reserves considering fees and arbitrage opportunities.

    This function extends the basic reserve calculation by incorporating transaction fees
    and potential arbitrage opportunities, following the methodology described in the
    FM-AMM paper.

    Parameters
    ----------
    carry_list : list
        List containing the previous prices and reserves.
    input_list: list
        List containing:
        prices : jnp.ndarray
            Array containing the current prices.
        gamma : float, optional
            Fee factor for no-arbitrage bounds, by default 0.997.
        arb_thresh : float, optional
            Threshold for profitable arbitrage, by default 0.0.
        arb_fees : float, optional
            Fees associated with arbitrage, by default 0.0.
        trade : jnp.ndarray
            Array of trades for each timestep. Format for each row:
            trades[0] = index of token to trade in to pool
            trades[1] = index of token to trade out of pool
            trades[2] = amount of 'token in' to trade
    arb_quality : float
        Quality of arbitrage execution, between 0 and 1. Higher values mean more efficient arbitrage.
    weights : jnp.ndarray
        Array of asset weights.
    do_trades : bool
        Whether to execute trades.
    do_arb : bool
        Whether to execute arbitrage trades.
    noise_trader_ratio : float, optional
        Ratio of noise traders to signal traders, by default 0.0

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves.
    """


    # carry_list[1] is previous reserves
    prev_reserves = carry_list[1]

    # input_list[0] is prices
    prices = input_list[0]
    gamma = input_list[1]
    arb_thresh = input_list[2]
    arb_fees = input_list[3]
    trade = input_list[4]

    if do_arb:
        reserves_with_perfect_arb = _jax_calc_cowamm_reserves_with_fees_scan_function(
            carry_list=carry_list,
            prices=prices,
            weights=weights,
            gamma=gamma,
            arb_thresh=arb_thresh,
            arb_fees=arb_fees,
            noise_trader_ratio=noise_trader_ratio,
        )[-1]

        reserves_with_one_arb = (
            _jax_calc_cowamm_reserves_one_arb_with_fees_scan_function(
                carry_list=carry_list,
                prices=prices,
                weight=weights[0],
                gamma=gamma,
                arb_thresh=arb_thresh,
                arb_fees=arb_fees,
                noise_trader_ratio=noise_trader_ratio,
            )[-1]
        )

        reserves = (
            arb_quality * reserves_with_perfect_arb
            + (1 - arb_quality) * reserves_with_one_arb
        )
    else:
        reserves = prev_reserves

    # apply trade if trade is present
    if do_trades:
        reserves += jitted_FMAMM_cond_trade(do_trades, reserves, weights, trade, gamma)

    return [prices, reserves], reserves


@partial(jit, static_argnums=(8, 9))
def _jax_calc_cowamm_reserves_with_dynamic_inputs(
    initial_reserves,
    prices,
    fees,
    arb_thresh,
    arb_fees,
    weights,
    arb_quality,
    trades=None,
    do_trades=True,
    do_arb=True,
    noise_trader_ratio=0.0,
):
    """
    Calculate AMM reserves considering fees for CowAMM

    This function computes the changes in reserves for a CowAMM that takes into account
    transaction fees and potential arbitrage opportunities. It uses a scan operation
    to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    fees : jnp.ndarray
        Swap fee charged over time
    arb_thresh : jnp.ndarray
        Threshold for profitable arbitrage over time
    arb_fees : jnp.ndarray
        Fees associated with arbitrage over time
    weight : float, optional
        Weight parameter for the pool, by default 0.5
    trades : jnp.ndarray, optional
        Array of trades for each timestep. Format for each row:
        trades[0] = index of token to trade in to pool
        trades[1] = index of token to trade out of pool
        trades[2] = amount of 'token in' to trade
    do_trades : bool, optional
        Whether to process trades, by default False
    do_arb : bool, optional
        Whether to process arbitrage, by default True
    noise_trader_ratio : float, optional
        Ratio of noise traders to signal traders, by default 0.0

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """

    n_assets = prices.shape[1]
    assert n_assets == 2

    initial_prices = prices[0]

    gamma = 1.0 - fees

    scan_fn = Partial(
        _jax_calc_cowamm_reserves_with_dynamic_fees_and_trades_scan_function,
        weights=weights,
        do_trades=do_trades,
        do_arb=do_arb,
        arb_quality=arb_quality,
        noise_trader_ratio=noise_trader_ratio,
    )

    carry_list_init = [initial_prices, initial_reserves]
    _, reserves = scan(
        scan_fn, carry_list_init, [prices, gamma, arb_thresh, arb_fees, trades]
    )

    return reserves
