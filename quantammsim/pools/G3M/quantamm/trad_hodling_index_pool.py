"""Traditional (off-chain) HODLing index pool for QuantAMM.

Extends :class:`IndexMarketCapPool` with periodic rebalancing and realistic
centralised-exchange (CEX) execution costs: proportional trading fees
(``cex_tau``), bid-ask spread, and an annual streaming/management fee. Reserves
are HODLed between rebalancing windows, modelling a traditional index fund that
incurs real-world trading frictions on each rebalance.
"""
# again, this only works on startup!
from jax import config

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
from quantammsim.utils.data_processing.historic_data_utils import get_data_dict

from typing import Dict, Any, Optional, List
from functools import partial
from abc import abstractmethod
import numpy as np
import pandas as pd
from importlib import resources as impresources

from quantammsim import data
from pathlib import Path
from copy import deepcopy
# import the fine weight output function which has pre-set argument rule_outputs_are_weights
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output_from_weights,
)

from quantammsim.pools.G3M.quantamm.quantamm_reserves import _jax_calc_quantAMM_reserve_ratios


@jit
def calc_rvr_trade_cost(
    trade,
    prices,
    volatility,
    cex_volume,
    cex_slippage_from_spread,
    cex_tau,
    grinold_alpha,
):

    # market_impact = model_market_impact(cex_volume, volatility, trade, grinold_alpha)
    # market_impact = 0
    # estimated_trade_cost = (
    #     0.5*cex_tau
    #     #  + market_impact + 0.5*cex_slippage_from_spread
    # ) * jnp.sum(jnp.abs(trade) * prices)

    estimated_trade = jnp.where(
        trade < 0,
        -trade,
        0.0,
    )
    abs_trade = jnp.abs(trade)
    estimated_trade_cost_from_cex_fees = cex_tau * jnp.sum(estimated_trade * prices)
    estimated_trade_cost_from_cex_spread = 0.5 * jnp.sum(
        cex_slippage_from_spread * abs_trade * prices
    )
    # estimated_trade_cost_from_cex_market_impact = model_market_impact(
    #     cex_volume, volatility, trade, grinold_alpha
    # ).sum()

    return (
        estimated_trade_cost_from_cex_fees
        + estimated_trade_cost_from_cex_spread
        # + estimated_trade_cost_from_cex_market_impact
    )


@jit
def _jax_calc_rvr_scan_function(
    carry_list,
    input_list,
    cex_tau,
    grinold_alpha,
    per_step_fee=0.0,
):
    """
    Calculate traditional reserve changes considering transaction fees.

    This function computes the changes in reserves for a traditional market model based on
    changes in asset weights and prices, incorporating transaction fees.

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
        volatilities: jnp.ndarray
            Array containing each assets volatility (std of log returns) over time.
        cex_volumes: jnp.ndarray
            Array containing each assets volume over time on an external CEX.
        cex_spread: jnp.ndarray
            Array containing each assets volume over time on an external CEX.
        do_trade : jnp.ndarray
            one-dimensional array of booleans, indicating whether a trade was made at each timestep.
    cex_tau : float
        Transaction fee rate on an external CEX.
    grinold_alpha : float

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of new reserves.
    """

    # carry_list[0] is previous weights
    prev_weights = carry_list[0]

    # carry_list[2] is previous reserves
    prev_reserves = carry_list[2]

    # weights_and_prices are the weigthts and prices, in that order
    weights = input_list[0]
    prices = input_list[1]
    volatilities = input_list[2]
    cex_volumes = input_list[3]
    cex_spread = input_list[4]
    do_trade = input_list[5]

    # First calculate change in reserves from new prices
    temp_price_value = jnp.sum(prev_reserves * prices)
    temp_price_reserves = prev_weights * temp_price_value / prices

    # then look at the 'outgoing' reserve change and charge fees on
    # that value
    delta_reserves_from_change_in_prices = temp_price_reserves - prev_reserves

    # feeable_reserves_change_from_change_in_prices = jnp.where(
    #     delta_reserves_from_change_in_prices < 0,
    #     -delta_reserves_from_change_in_prices,
    #     0.0,
    # )
    # # calculate effective fee rate tau

    # fee_charged_from_change_in_prices = cex_tau * jnp.sum(
    #     feeable_reserves_change_from_change_in_prices * prices
    # )
    rvr_trade_cost_from_change_in_prices = calc_rvr_trade_cost(
        delta_reserves_from_change_in_prices,
        prices,
        volatilities,
        cex_volumes,
        cex_spread,
        cex_tau,
        grinold_alpha,
    )
    # reduce total value by that amount, and recalc portfolio
    post_fees_value_from_change_in_prices = (
        temp_price_value - rvr_trade_cost_from_change_in_prices
    )

    reserves_from_change_in_prices = (
        prev_weights * post_fees_value_from_change_in_prices / prices
    )

    # Second calculate change in reserves from new weights
    # (note that as prices are constant, there is no change in
    # value at this point)
    temp_weights_reserves = weights * post_fees_value_from_change_in_prices / prices

    # then look at the 'outgoing' reserve change and charge fees on
    # that value

    delta_reserves_from_change_in_weights = (
        temp_weights_reserves - reserves_from_change_in_prices
    )
    # feeable_reserves_change_from_change_in_weights = jnp.where(
    #     delta_reserves_from_change_in_weights < 0,
    #     -delta_reserves_from_change_in_weights,
    #     0.0,
    # )
    # fee_charged_from_change_in_weights = cex_tau * jnp.sum(
    #     feeable_reserves_change_from_change_in_weights * prices
    # )

    rvr_trade_cost_from_change_in_weights = calc_rvr_trade_cost(
        delta_reserves_from_change_in_weights,
        prices,
        volatilities,
        cex_volumes,
        cex_spread,
        cex_tau,
        grinold_alpha,
    )
    # reduce total value by that amount, and recalc portfolio
    post_fees_value_from_change_in_weights = (
        post_fees_value_from_change_in_prices - rvr_trade_cost_from_change_in_weights
    )
    new_reserves = weights * post_fees_value_from_change_in_weights / prices
    new_reserves = jnp.where(do_trade, new_reserves * (1.0 - per_step_fee), prev_reserves)
    return [
        weights,
        prices,
        new_reserves,
    ], new_reserves


@jit
def _jax_calc_rvr_reserve_change(
    initial_reserves,
    weights,
    prices,
    volatilities,
    cex_volumes,
    cex_spread,
    do_trade,
    gamma=0.998,
    per_step_fee=0.0,
):
    """
    Calculate traditional reserve changes considering transaction fees.

    This function computes the changes in reserves for a traditional market model based on
    changes in asset weights and prices, incorporating transaction fees. It uses a scan operation
    to apply these calculations over multiple timesteps, simulating the effect of sequential
    trading sessions.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    weights : jnp.ndarray
        Two-dimensional array of asset weights over time.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    volatilities: jnp.ndarray
        Two-dimensional array of asset volatilities over time.
    cex_volumes: jnp.ndarray
        Two-dimensional array of asset volumes over time on an external CEX.
    cex_spread: jnp.ndarray
        Two-dimensional array of asset spreads over time on an external CEX.
    do_trade : jnp.ndarray
        one-dimensional array of booleans, indicating whether a trade was made at each timestep.
    gamma : float, optional
        1 minus the transaction fee rate, by default 0.998.

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """
    # NOTE: MAYBE THIS SHOULD BE DONE IN LOG SPACE?

    # We do this like a block, so first there is the new
    # weight value and THEN we get new prices by the end of
    # the block.

    # So, for first weight, we have initial reserves, weights and
    # prices, so the change is 1

    scan_fn = Partial(
        _jax_calc_rvr_scan_function, cex_tau=1.0 - gamma, grinold_alpha=0.5, per_step_fee=per_step_fee
    )

    carry_list_init = [weights[0], prices[0], initial_reserves]
    carry_list_end, reserves = scan(
        scan_fn,
        carry_list_init,
        [
            weights,
            prices,
            volatilities,
            cex_volumes,
            cex_spread,
            do_trade,
        ],
    )
    return reserves, carry_list_init, carry_list_end


@jit
def _jax_calc_lvr_reserve_change_scan_function(carry_list, weights_and_prices, tau, per_step_fee=0.0):
    """
    Calculate traditional reserve changes considering transaction fees.

    This function computes the changes in reserves for a traditional market model based on
    changes in asset weights and prices, incorporating transaction fees.

    Parameters
    ----------
    carry_list : list
        List containing the previous weights, prices, and reserves.
    weights_and_prices : jnp.ndarray
        Array containing the current weights, prices, and do_trade.
    tau : float
        Transaction fee rate.

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of new reserves.
    """

    # carry_list[0] is previous weights
    prev_weights = carry_list[0]

    # carry_list[2] is previous reserves
    prev_reserves = carry_list[2]

    # weights_and_prices are the weigthts and prices, in that order
    weights = weights_and_prices[0]
    prices = weights_and_prices[1]
    do_trade = weights_and_prices[2]
    # First calculate change in reserves from new prices
    temp_price_value = jnp.sum(prev_reserves * prices)
    temp_price_reserves = prev_weights * temp_price_value / prices

    # then look at the 'outgoing' reserve change and charge fees on
    # that value
    delta_reserves_from_change_in_prices = temp_price_reserves - prev_reserves
    feeable_reserves_change_from_change_in_prices = jnp.where(
        delta_reserves_from_change_in_prices < 0,
        -delta_reserves_from_change_in_prices,
        0.0,
    )
    fee_charged_from_change_in_prices = tau * jnp.sum(
        feeable_reserves_change_from_change_in_prices * prices
    )

    # reduce total value by that amount, and recalc portfolio
    post_fees_value_from_change_in_prices = (
        temp_price_value - fee_charged_from_change_in_prices
    )
    reserves_from_change_in_prices = (
        prev_weights * post_fees_value_from_change_in_prices / prices
    )

    # Second calculate change in reserves from new weights
    # (note that as prices are constant, there is no change in
    # value at this point)
    temp_weights_reserves = weights * post_fees_value_from_change_in_prices / prices

    # then look at the 'outgoing' reserve change and charge fees on
    # that value

    delta_reserves_from_change_in_weights = (
        temp_weights_reserves - reserves_from_change_in_prices
    )
    feeable_reserves_change_from_change_in_weights = jnp.where(
        delta_reserves_from_change_in_weights < 0,
        -delta_reserves_from_change_in_weights,
        0.0,
    )
    fee_charged_from_change_in_weights = tau * jnp.sum(
        feeable_reserves_change_from_change_in_weights * prices
    )

    # reduce total value by that amount, and recalc portfolio
    post_fees_value_from_change_in_weights = (
        post_fees_value_from_change_in_prices - fee_charged_from_change_in_weights
    )
    new_reserves = weights * post_fees_value_from_change_in_weights / prices

    # if do_trade is true, then we need to add the trade to the reserves
    new_reserves = jnp.where(do_trade, new_reserves * (1.0 - per_step_fee), prev_reserves)
    return [
        weights,
        prices,
        new_reserves,
    ], new_reserves


@jit
def _jax_calc_lvr_reserve_change(initial_reserves, weights, prices, do_trade, gamma=0.998, per_step_fee=0.0):
    """
    Calculate traditional reserve changes considering transaction fees.

    This function computes the changes in reserves for a traditional market model based on
    changes in asset weights and prices, incorporating transaction fees. It uses a scan operation
    to apply these calculations over multiple timesteps, simulating the effect of sequential
    trading sessions.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    weights : jnp.ndarray
        Two-dimensional array of asset weights over time.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    do_trade : jnp.ndarray
        Two-dimensional array of trade amounts over time.
    gamma : float, optional
        1 minus the transaction fee rate, by default 0.998.

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """
    # NOTE: MAYBE THIS SHOULD BE DONE IN LOG SPACE?

    # We do this like a block, so first there is the new
    # weight value and THEN we get new prices by the end of
    # the block.

    # So, for first weight, we have initial reserves, weights and
    # prices, so the change is 1

    scan_fn = Partial(_jax_calc_lvr_reserve_change_scan_function, tau=1.0 - gamma, per_step_fee=per_step_fee)

    carry_list_init = [weights[0], prices[0], initial_reserves]
    _, reserves = scan(scan_fn, carry_list_init, [weights, prices, do_trade])
    return reserves


class TradHodlingIndexPool(IndexMarketCapPool):
    """
    Market-cap index pool simulating traditional (off-chain) rebalancing.

    Like ``HodlingIndexPool``, this variant only rebalances during the
    ``weight_interpolation_period`` window at the start of each
    ``chunk_period`` and HODLs reserves otherwise. The key difference is
    that reserve updates model **centralised-exchange (CEX) execution
    costs** rather than on-chain AMM swap mechanics:

    - **CEX fees** (``cex_tau``): flat proportional fee on sold tokens.
    - **Bid-ask spread** (``cex_spread``): per-asset half-spread cost.
    - **Market impact** (Grinold-alpha model, currently commented out):
      square-root impact scaled by volatility and volume.

    An ``annual_streaming_fee`` (default 4 %) is also compounded into
    a per-step multiplicative fee applied to reserves at each active
    trading step, modelling the management fee charged by traditional
    index products.

    This pool loads auxiliary market-microstructure data (volatility,
    volume, spread) via ``get_data_dict`` at reserve-calculation time,
    so it requires the full historic data pipeline to be available.

    Inherits weight calculation logic (market-cap weighting) from
    ``IndexMarketCapPool`` and overrides ``calculate_reserves_with_fees``
    and ``calculate_reserves_zero_fees``.

    See Also
    --------
    HodlingIndexPool : On-chain AMM variant (uses G3M swap-fee mechanics).
    IndexMarketCapPool : Continuously-rebalanced base class.
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

        all_tokens = [run_fingerprint["tokens"]]
        all_tokens = [item for sublist in all_tokens for item in sublist]
        unique_tokens = list(set(all_tokens))
        unique_tokens.sort()

        data_dict = get_data_dict(
            unique_tokens,
            run_fingerprint,
            data_kind="historic",
            root=None,
            max_memory_days=365.0,
            start_date_string=run_fingerprint["startDateString"],
            end_time_string=run_fingerprint["endDateString"],
            start_time_test_string=run_fingerprint["endDateString"],
            end_time_test_string=run_fingerprint["endTestDateString"],
            max_mc_version=None,
            return_slippage=True,
        )

        volatilities = data_dict["annualised_daily_volatility"][
            data_dict["start_idx"] : data_dict["start_idx"]
            + data_dict["bout_length"]
            - 1
        ]
        cex_volumes = data_dict["daily_volume"][
            data_dict["start_idx"] : data_dict["start_idx"]
            + data_dict["bout_length"]
            - 1
        ]
        cex_spread = data_dict["spread"][
            data_dict["start_idx"] : data_dict["start_idx"]
            + data_dict["bout_length"]
            - 1
        ]

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

            # calculate what proportion of the time the reserve are being updated
            reserve_update_frequency = weight_interpolation_period / chunk_period

            # calculate the number of reserve updates per year
            minutes_per_year = 525960
            chunks_per_year = minutes_per_year / run_fingerprint["chunk_period"]
            trading_steps_per_year = weight_interpolation_period * chunks_per_year / run_fingerprint["arb_frequency"]

            # calculate the fees per reserve update
            annual_streaming_fee = run_fingerprint.get("annual_streaming_fee", 0.04)
            per_step_fee = 1 - (1 - annual_streaming_fee)**(1/trading_steps_per_year)

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
            reserves = _jax_calc_rvr_reserve_change(
                initial_reserves,
                weights,
                local_prices,
                volatilities,
                cex_volumes,
                cex_spread,
                interpolation_mask,
                gamma=1 - run_fingerprint["fees"],
                per_step_fee=per_step_fee,
            )[0]
        else:
            reserves = jnp.broadcast_to(
                initial_reserves,
                (bout_length - 1, n_assets)
            )

        return reserves

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        local_run_fingerprint = deepcopy(run_fingerprint)
        local_run_fingerprint["fees"] = 0.0
        return self.calculate_reserves_with_fees(params, local_run_fingerprint, prices, start_index, additional_oracle_input)

tree_util.register_pytree_node(
    TradHodlingIndexPool,
    TradHodlingIndexPool._tree_flatten,
    TradHodlingIndexPool._tree_unflatten,
)
