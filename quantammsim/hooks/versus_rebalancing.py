from abc import ABC
from typing import Dict, Any, Optional
from copy import deepcopy

# TODO above is all from jax utils, tidy up required

import jax.numpy as jnp
from jax import jit
from jax.tree_util import Partial
from jax.lax import scan
from jax.lax import dynamic_slice


from quantammsim.utils.data_processing.historic_data_utils import get_data_dict
from quantammsim.pools.base_pool import AbstractPool


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
):
    """
    Calculate traditional reserve changes considering transaction fees.

    This function computes the changes in reserves for a traditional market model based on
    changes in asset weights and prices, incorporating transaction fees.

    Parameters
    ----------
    carry_list : list
        List containing the previous weights, prices, reserves, and LP supply.
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
            Array containing each assets spread over time on an external CEX.
        lp_supply: jnp.ndarray
            Array containing LP token supply over time.
    cex_tau : float
        Transaction fee rate on an external CEX.
    grinold_alpha : float

    Returns
    -------
    list
        Updated list containing the new weights, prices, reserves, and LP supply.
    jnp.ndarray
        Array of new reserves.
    """

    # carry_list[0] is previous weights
    prev_weights = carry_list[0]

    # carry_list[1] is previous prices
    prev_prices = carry_list[1]

    # carry_list[2] is previous reserves
    prev_reserves = carry_list[2]

    # carry_list[3] is previous lp_supply
    prev_lp_supply = carry_list[3]

    # input_list[0] is weights
    weights = input_list[0]
    # input_list[1] is prices
    prices = input_list[1]
    # input_list[2] is volatilities
    volatilities = input_list[2]
    # input_list[3] is cex_volumes
    cex_volumes = input_list[3]
    # input_list[4] is cex_spread
    cex_spread = input_list[4]
    # input_list[5] is lp_supply
    lp_supply = input_list[5]

    # Handle LP supply changes
    lp_supply_change = lp_supply != prev_lp_supply
    prev_reserves = jnp.where(
        lp_supply_change, prev_reserves * lp_supply / prev_lp_supply, prev_reserves
    )

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

    return [
        weights,
        prices,
        new_reserves,
        lp_supply,
    ], new_reserves


@jit
def _jax_calc_rvr_reserve_change(
    initial_reserves,
    weights,
    prices,
    volatilities,
    cex_volumes,
    cex_spread,
    gamma=0.998,
    lp_supply_array=None,
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
    volatilities : jnp.ndarray
        Array containing each assets volatility (std of log returns) over time.
    cex_volumes : jnp.ndarray
        Array containing each assets volume over time on an external CEX.
    cex_spread : jnp.ndarray
        Array containing each assets spread over time on an external CEX.
    gamma : float, optional
        1 minus the transaction fee rate, by default 0.998.
    lp_supply_array : jnp.ndarray, optional
        Array of LP token supply over time, by default None.

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

    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(weights.shape[0], lp_supply_array),
        lp_supply_array,
    )

    scan_fn = Partial(
        _jax_calc_rvr_scan_function, cex_tau=1.0 - gamma, grinold_alpha=0.5
    )

    carry_list_init = [weights[0], prices[0], initial_reserves, lp_supply_array[0]]
    carry_list_end, reserves = scan(
        scan_fn,
        carry_list_init,
        [
            weights,
            prices,
            volatilities,
            cex_volumes,
            cex_spread,
            lp_supply_array,
        ],
    )
    return reserves, carry_list_init, carry_list_end


@jit
def _jax_calc_lvr_reserve_change_scan_function(carry_list, weights_and_prices, tau):
    """
    Calculate traditional reserve changes considering transaction fees.

    This function computes the changes in reserves for a traditional market model based on
    changes in asset weights and prices, incorporating transaction fees.

    Parameters
    ----------
    carry_list : list
        List containing the previous weights, prices, reserves, and LP supply.
    weights_and_prices : jnp.ndarray
        Array containing the current weights, prices, and LP supply.
    tau : float
        Transaction fee rate.

    Returns
    -------
    list
        Updated list containing the new weights, prices, reserves, and LP supply.
    jnp.ndarray
        Array of new reserves.
    """

    # carry_list[0] is previous weights
    prev_weights = carry_list[0]

    # carry_list[1] is previous prices
    prev_prices = carry_list[1]

    # carry_list[2] is previous reserves
    prev_reserves = carry_list[2]

    # carry_list[3] is previous lp_supply
    prev_lp_supply = carry_list[3]

    # weights_and_prices[0] is weights
    weights = weights_and_prices[0]
    # weights_and_prices[1] is prices
    prices = weights_and_prices[1]
    # weights_and_prices[2] is lp_supply
    lp_supply = weights_and_prices[2]

    # Handle LP supply changes
    lp_supply_change = lp_supply != prev_lp_supply
    prev_reserves = jnp.where(
        lp_supply_change, prev_reserves * lp_supply / prev_lp_supply, prev_reserves
    )

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

    return [
        weights,
        prices,
        new_reserves,
        lp_supply,
    ], new_reserves


@jit
def _jax_calc_lvr_reserve_change(
    initial_reserves, weights, prices, gamma=0.998, lp_supply_array=None
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
    gamma : float, optional
        1 minus the transaction fee rate, by default 0.998.
    lp_supply_array : jnp.ndarray, optional
        Array of LP token supply over time, by default None.

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

    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)
    lp_supply_array = jnp.where(
        lp_supply_array.size == 1,
        jnp.full(weights.shape[0], lp_supply_array),
        lp_supply_array,
    )

    scan_fn = Partial(_jax_calc_lvr_reserve_change_scan_function, tau=1.0 - gamma)

    carry_list_init = [weights[0], prices[0], initial_reserves, lp_supply_array[0]]
    _, reserves = scan(scan_fn, carry_list_init, [weights, prices, lp_supply_array])
    return reserves


class CalculateLossVersusRebalancing(ABC):
    """Mixin class to add Loss-Versus-Rebalancing (LVR) style reserve changes.
    Note: this only works if the pool has a calculate_weights method, which is the case
    for all pools in quantammsim so far but is not a requirement of the base pool class.
    """

    def __init__(self):
        pass

    def get_original_pool_class(self):
        """Get the original pool class from MRO."""
        return next(
            cls
            for cls in self.__class__.__mro__
            if issubclass(cls, AbstractPool) and cls != AbstractPool
        )

    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if self.weights_needs_original_methods():
            original_pool_class = self.get_original_pool_class()
            # Create bound method that will use original pool's methods
            bound_calculate_weights = original_pool_class.calculate_weights.__get__(
                self, original_pool_class
            )
            weights = bound_calculate_weights(
                params, run_fingerprint, prices, start_index, additional_oracle_input
            )
        else:
            weights = self.calculate_weights(
                params, run_fingerprint, prices, start_index, additional_oracle_input
            )
        n_assets = run_fingerprint["n_assets"]

        # Calculate loss versus rebalancing reserve changes
        # some pools might return a single weight vector, not a time series
        weights = jnp.broadcast_to(
            weights, (run_fingerprint["bout_length"] - 1, n_assets)
        )
        local_prices = dynamic_slice(
            prices, start_index, (run_fingerprint["bout_length"] - 1, n_assets)
        )

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        # Use existing dynamic inputs infrastructure
        return _jax_calc_lvr_reserve_change(
            initial_reserves, weights, local_prices, gamma=1 - run_fingerprint["fees"]
        )

    def calculate_reserves_with_dynamic_inputs(self, *args, **kwargs):
        """
        Calculate reserves with dynamic inputs.

        Not implemented for LVR pools.

        Raises
        ------
        NotImplementedError
            Always raised.
        """

        raise NotImplementedError("This method is not implemented for LVR pools.")

    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        local_run_fingerprint = deepcopy(run_fingerprint)
        local_run_fingerprint["fees"] = 0
        return self.calculate_reserves_with_fees(
            params, local_run_fingerprint, prices, start_index, additional_oracle_input
        )


class CalculateRebalancingVersusRebalancing(ABC):
    """Mixin class to add Rebalancing-Versus-Rebalancing (RVR) style reserve changes.
    Note: this only works if the pool has a calculate_weights method, which is the case
    for all pools in quantammsim so far but is not a requirement of the base pool class.
    """

    def __init__(self):
        pass

    def get_original_pool_class(self):
        """Get the original pool class from MRO."""
        return next(
            cls
            for cls in self.__class__.__mro__
            if issubclass(cls, AbstractPool) and cls != AbstractPool
        )

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
            do_test_period=True,
        )

        n_assets = run_fingerprint["n_assets"]

        volatilities = dynamic_slice(
            data_dict["annualised_daily_volatility"],
            start_index,
            (run_fingerprint["bout_length"] - 1, n_assets),
        )

        cex_volumes = dynamic_slice(
            data_dict["daily_volume"],
            start_index,
            (run_fingerprint["bout_length"] - 1, n_assets),
        )

        cex_spread = dynamic_slice(
            data_dict["spread"],
            start_index,
            (run_fingerprint["bout_length"] - 1, n_assets),
        )

        if self.weights_needs_original_methods():
            original_pool_class = self.get_original_pool_class()
            # Create bound method that will use original pool's methods
            bound_calculate_weights = original_pool_class.calculate_weights.__get__(
                self, original_pool_class
            )
            weights = bound_calculate_weights(
                params, run_fingerprint, prices, start_index, additional_oracle_input
            )
        else:
            weights = self.calculate_weights(
                params, run_fingerprint, prices, start_index, additional_oracle_input
            )
        # Calculate loss versus rebalancing reserve changes

        # some pools might return a single weight vector, not a time series
        weights = jnp.broadcast_to(
            weights, (run_fingerprint["bout_length"] - 1, n_assets)
        )
        local_prices = dynamic_slice(
            prices, start_index, (run_fingerprint["bout_length"] - 1, n_assets)
        )

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights[0] * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        # Use existing dynamic inputs infrastructure
        return _jax_calc_rvr_reserve_change(
            initial_reserves,
            weights,
            local_prices,
            volatilities,
            cex_volumes,
            cex_spread,
            gamma=1 - run_fingerprint["fees"],
        )[0]

    def calculate_reserves_with_dynamic_inputs(self, *args, **kwargs):
        """
        Calculate reserves with dynamic inputs.

        Not implemented for RVR pools.

        Raises
        ------
        NotImplementedError
            Always raised.
        """
        raise NotImplementedError("This method is not implemented for RVR pools.")

    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        local_run_fingerprint = deepcopy(run_fingerprint)
        local_run_fingerprint["fees"] = 0
        return self.calculate_reserves_with_fees(
            params, local_run_fingerprint, prices, start_index, additional_oracle_input
        )
