"""On-chain HODLing index pool for QuantAMM.

Extends :class:`IndexMarketCapPool` with a HODLing regime: reserves are only
rebalanced via on-chain G3M arbitrage during a short interpolation window at
the start of each chunk period, and frozen (HODLed) otherwise. This models
the behaviour of an AMM-based index product that limits rebalancing frequency
to reduce impermanent loss.
"""
# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
from jax import default_backend
from jax import local_device_count, devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import jit, vmap
from jax import devices, device_put
from jax import tree_util
from jax.lax import stop_gradient, dynamic_slice, while_loop, scan, cond
from jax.nn import softmax
from jax.tree_util import Partial

from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
)
from quantammsim.pools.G3M.quantamm.index_market_cap_pool import IndexMarketCapPool
from quantammsim.pools.G3M.optimal_n_pool_arb import (
    precalc_shared_values_for_all_signatures,
    precalc_components_of_optimal_trade_across_weights_and_prices,
    precalc_components_of_optimal_trade_across_weights_and_prices_and_dynamic_fees,
    parallelised_optimal_trade_sifter,
)
from quantammsim.pools.G3M.G3M_trades import jitted_G3M_cond_trade
from quantammsim.pools.noise_trades import calculate_reserves_after_noise_trade

from typing import Dict, Any, Optional, List
from functools import partial
from abc import abstractmethod
import numpy as np
import pandas as pd
from importlib import resources as impresources

from quantammsim import data

from pathlib import Path

# import the fine weight output function which has pre-set argument rule_outputs_are_weights
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output_from_weights,
)

from quantammsim.pools.G3M.quantamm.quantamm_reserves import _jax_calc_quantAMM_reserve_ratios


@jit
def _jax_calc_quantAMM_reserves_with_fees_using_precalcs(
    initial_reserves,
    weights,
    prices,
    do_trades,
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
    do_trades : jnp.ndarray
        One-dimensional array of booleans indicating whether a trade was made at each timestep.
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
            do_trades,
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
    do_trades = weights_and_prices_and_precalcs[8]

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

    reserves = jnp.where(do_trades, reserves, prev_reserves)
    return [
        weights,
        prices,
        reserves,
        counter,
    ], reserves


class HodlingIndexPool(IndexMarketCapPool):
    """
    Market-cap index pool that HODLs reserves between weight updates.

    Unlike the base ``IndexMarketCapPool``, which allows continuous
    arbitrage-driven rebalancing at every timestep, this variant only
    permits trades during a ``weight_interpolation_period`` window at the
    start of each ``chunk_period``. Outside that window the pool's
    reserves are frozen (HODLed), so the portfolio drifts with market
    prices rather than being continuously rebalanced.

    Reserve calculations use the on-chain QuantAMM arbitrage mechanics
    (optimal-trade sifter with swap fees), matching what would happen
    on an actual G3M AMM deployment. The ``do_trades`` mask passed to
    the reserve calculator encodes which timesteps fall inside the
    active rebalancing window.

    Inherits weight calculation logic (market-cap weighting) from
    ``IndexMarketCapPool`` and overrides ``calculate_reserves_with_fees``
    and ``calculate_reserves_zero_fees`` to implement the HODLing
    behaviour.

    See Also
    --------
    IndexMarketCapPool : Continuously-rebalanced base class.
    TradHodlingIndexPool : Off-chain (CEX) variant with realistic
        trading costs instead of on-chain AMM mechanics.
    """

    @partial(jit, static_argnums=(2))
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        chunk_period = run_fingerprint["chunk_period"]
        weight_interpolation_period = run_fingerprint.get(
            "weight_interpolation_period", chunk_period
        )

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        weights = self.calculate_weights(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )

        # First create the full timeline mask
        full_timeline = jnp.arange(bout_length - 1)  # -1 because reserve ratios are between points
        chunk_positions = full_timeline % chunk_period
        full_mask = chunk_positions < weight_interpolation_period

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_weights = weights[:: run_fingerprint["arb_frequency"]]
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
            interpolation_mask = full_mask[:: run_fingerprint["arb_frequency"]]
        else:
            arb_acted_upon_weights = weights
            arb_acted_upon_local_prices = local_prices
            interpolation_mask = full_mask

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = arb_acted_upon_weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]
        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_quantAMM_reserves_with_fees_using_precalcs(
                initial_reserves,
                arb_acted_upon_weights,
                arb_acted_upon_local_prices,
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
                noise_trader_ratio=run_fingerprint["noise_trader_ratio"],
                do_trades=interpolation_mask,
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves, arb_acted_upon_local_prices.shape
            )

        return reserves

    @partial(jit, static_argnums=(2))
    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        chunk_period = run_fingerprint["chunk_period"]
        weight_interpolation_period = run_fingerprint.get("weight_interpolation_period", chunk_period)

        # Get local prices and calculate weights (inherited from index_market_cap_pool)
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        weights = self.calculate_weights(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )

        # Calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        if run_fingerprint["do_arb"]:
            # First create the full timeline mask
            full_timeline = jnp.arange(bout_length - 1)  # -1 because reserve ratios are between points
            chunk_positions = full_timeline % chunk_period
            full_mask = chunk_positions < weight_interpolation_period

            # Apply arb_frequency to weights, prices, and mask
            if run_fingerprint["arb_frequency"] != 1:
                arb_acted_upon_weights = weights[::run_fingerprint["arb_frequency"]]
                arb_acted_upon_local_prices = local_prices[::run_fingerprint["arb_frequency"]]
                interpolation_mask = full_mask[::run_fingerprint["arb_frequency"]]
            else:
                arb_acted_upon_weights = weights
                arb_acted_upon_local_prices = local_prices
                interpolation_mask = full_mask

            # Calculate reserve ratios
            reserve_ratios = _jax_calc_quantAMM_reserve_ratios(
                arb_acted_upon_weights[:-1],
                arb_acted_upon_local_prices[:-1],
                arb_acted_upon_weights[1:],
                arb_acted_upon_local_prices[1:],
            )

            # Apply mask to reserve ratios
            masked_ratios = jnp.where(
                interpolation_mask[:-1, jnp.newaxis],  # Add dimension for n_assets
                reserve_ratios,
                jnp.ones((1, n_assets), dtype=jnp.float64)
            )

            # Calculate final reserves
            reserves = jnp.vstack(
                [
                    initial_reserves,
                    initial_reserves * jnp.cumprod(masked_ratios, axis=0),
                ]
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves,
                (bout_length - 1, n_assets)
            )

        return reserves

tree_util.register_pytree_node(
    HodlingIndexPool,
    HodlingIndexPool._tree_flatten,
    HodlingIndexPool._tree_unflatten,
)
