from jax import config

config.update("jax_enable_x64", True)
from jax import default_backend
from jax import devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")


import jax.numpy as jnp
from jax import jit, vmap, devices
from jax.lax import stop_gradient, dynamic_slice, associative_scan


import numpy as np

from functools import partial

np.seterr(all="raise")
np.seterr(under="print")


def _daily_log_sharpe(values: jnp.ndarray) -> jnp.ndarray:
    """
    Daily Sharpe on log close-to-close returns.
    Uses 1 day = 1440 minutes; annualizes with sqrt(365).
    """
    # Sample daily values using stride slice
    daily_values = values[::1440]
    
    # Calculate daily log returns
    log_rets = jnp.diff(jnp.log(daily_values + 1e-12))

    mean = log_rets.mean()
    std  = log_rets.std()

    # Annualize daily stats (calendar days)
    return jnp.sqrt(365.0) * (mean / (std + 1e-8))

def _calculate_max_drawdown(value_over_time, duration=7 * 24 * 60):
    """Calculate maximum drawdown on a chosen basis."""
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    values = value_over_time_truncated.reshape(-1, duration)
    running_max = vmap(lambda x: associative_scan(jnp.maximum, x))(values)
    drawdowns = (values - running_max) / running_max
    max_drawdowns = jnp.min(drawdowns, axis=1)
    return jnp.min(max_drawdowns)


def _calculate_var(value_over_time, percentile=5.0, duration=24 * 60):
    """Calculate VaR using intraday returns."""
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    values = value_over_time_truncated.reshape(-1, duration)
    returns = jnp.diff(values, axis=-1) / values[:, :-1]
    var = vmap(lambda x: jnp.percentile(x, percentile))(returns)
    return jnp.mean(var)


def _calculate_var_trad(value_over_time, percentile=5.0, duration=24 * 60):
    """Calculate traditional VaR using daily returns."""
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    value_over_time = value_over_time_truncated.reshape(-1, duration)[:, -1]
    returns = jnp.diff(value_over_time) / value_over_time[:-1]
    return jnp.percentile(returns, percentile)


def _calculate_raroc(value_over_time, percentile=5.0, duration=24 * 60):
    # Calculate returns
    total_return = value_over_time[-1] / value_over_time[0] - 1.0

    # Drop any incomplete chunks at the end by truncating to multiple of duration
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    value_over_time_chunks = value_over_time_truncated.reshape(-1, duration)
    # Calculate VaR (using our intraday method)
    returns = jnp.diff(value_over_time_chunks) / value_over_time_chunks[:, :-1]
    var = vmap(lambda x: jnp.percentile(x, percentile))(returns)
    var = jnp.mean(var)  # This is already negative

    # Calculate annualized RAROC
    days_in_sample = len(value_over_time) / (24 * 60)
    annualization_factor = 365 / days_in_sample

    annualized_return = (1 + total_return) ** annualization_factor - 1
    annualized_var = var * jnp.sqrt(annualization_factor * 24 * 60 / duration)

    # RAROC = Annualized Return / VaR (VaR is already negative)
    return -annualized_return / annualized_var


def _calculate_rovar(value_over_time, percentile=5.0, duration=24 * 60):
    # Drop any incomplete chunks at the end by truncating to multiple of duration
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    value_over_time_chunks = value_over_time_truncated.reshape(-1, duration)

    # Calculate returns per 'duration'
    period_returns = value_over_time_chunks[:, -1] / value_over_time_chunks[:, 0] - 1.0
    # Calculate VaR (using our intraday method)
    returns = jnp.diff(value_over_time_chunks) / value_over_time_chunks[:, :-1]
    var = vmap(lambda x: jnp.percentile(x, percentile))(returns)
    mean_var = jnp.mean(var)
    # Calculate annualized rovar
    days_in_sample = len(value_over_time) / (24 * 60)
    annualization_factor = 365 / days_in_sample

    annualized_return = (1 + period_returns) ** ((365 * 24 * 60) / duration) - 1
    mean_annualized_return = jnp.mean(annualized_return)
    annualized_var = mean_var * jnp.sqrt(annualization_factor * 24 * 60 / duration)

    # ROVAR = mean of: annualized Return per chunk / VaR (VaR is already negative) per chunk
    return -mean_annualized_return / annualized_var


def _calculate_rovar_trad(value_over_time, percentile=5.0, duration=24 * 60):
    # Drop any incomplete chunks at the end by truncating to multiple of duration
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    value_over_time_chunks = value_over_time_truncated.reshape(-1, duration)

    # Calculate returns per 'duration' using end-of-period values
    period_returns = value_over_time_chunks[:, -1] / value_over_time_chunks[:, 0] - 1.0

    # Calculate VaR using traditional method (end-of-period returns)
    end_of_period_values = value_over_time_chunks[:, -1]
    var_returns = jnp.diff(end_of_period_values) / end_of_period_values[:-1]
    var = jnp.percentile(var_returns, percentile)

    # Calculate annualized rovar
    days_in_sample = len(value_over_time) / (24 * 60)
    annualization_factor = 365 / days_in_sample

    annualized_return = (1 + period_returns) ** ((365 * 24 * 60) / duration) - 1
    mean_annualized_return = jnp.mean(annualized_return)
    annualized_var = var * jnp.sqrt(annualization_factor * 24 * 60 / duration)

    # ROVAR = mean of annualized returns / VaR (VaR is already negative)
    return -mean_annualized_return / annualized_var


def _calculate_sterling_ratio(
    value_over_time, duration=24 * 60, drawdown_adjustment=None
):
    """
    Calculate the Sterling ratio using JAX for a given value over time series.

    Parameters
    ----------
    value_over_time : jnp.ndarray
        Array of portfolio values over time
    duration : int
        Duration in minutes to calculate returns over
    drawdown_adjustment : float, optional
        Adjustment to add to average drawdown (e.g., 0.1 for traditional 10% adjustment).
        If None, no adjustment is applied.

    Returns
    -------
    float
        Sterling ratio (annualized)
    """
    # Handle incomplete chunks
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    values = value_over_time_truncated.reshape(-1, duration)

    # Calculate running max using associative_scan for efficiency
    running_max = vmap(lambda x: associative_scan(jnp.maximum, x))(values)

    # Calculate drawdowns per chunk
    drawdowns = (values - running_max) / running_max
    chunk_max_drawdowns = jnp.min(drawdowns, axis=1)

    # Calculate average of annual maximum drawdowns
    avg_drawdown = jnp.mean(chunk_max_drawdowns)

    # Calculate annualized return
    days_in_sample = len(value_over_time) / (24 * 60)
    annualization_factor = 365 / days_in_sample

    total_return = value_over_time[-1] / value_over_time[0] - 1.0
    annualized_return = (1 + total_return) ** annualization_factor - 1

    # Apply drawdown adjustment if specified
    if drawdown_adjustment is not None:
        denominator = -(avg_drawdown + drawdown_adjustment)
    else:
        denominator = -avg_drawdown

    # Handle zero/positive drawdown case
    sterling = jnp.where(denominator <= 0, jnp.inf, annualized_return / denominator)

    return sterling


def _calculate_calmar_ratio(value_over_time, duration=None):
    """
    Calculate the Calmar ratio using JAX for a given value over time series.

    Parameters
    ----------
    value_over_time : jnp.ndarray
        Array of portfolio values over time
    duration : int
        Maximum lookback period in minutes (default is 36 months)
        Only used to truncate the data if needed

    Returns
    -------
    float
        Calmar ratio (annualized)
    """
    # Truncate to maximum lookback period if needed
    if duration is not None and len(value_over_time) > duration:
        value_over_time = value_over_time[-duration:]

    # Calculate running max for entire series
    running_max = associative_scan(jnp.maximum, value_over_time)

    # Calculate drawdowns and find maximum drawdown
    drawdowns = (value_over_time - running_max) / running_max
    max_drawdown = jnp.min(drawdowns)

    # Calculate annualized return
    days_in_sample = len(value_over_time) / (24 * 60)
    annualization_factor = 365 / days_in_sample

    total_return = value_over_time[-1] / value_over_time[0] - 1.0
    annualized_return = (1 + total_return) ** annualization_factor - 1

    # Handle zero/positive drawdown case
    calmar = jnp.where(max_drawdown >= 0, jnp.inf, annualized_return / -max_drawdown)

    return calmar


def _calculate_ulcer_index(value_over_time, duration=7 * 24 * 60):
    """Calculate Ulcer Index on a chosen basis.

    The Ulcer Index measures downside risk considering both depth and duration of drawdowns.
    """
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    values = value_over_time_truncated.reshape(-1, duration)
    running_max = vmap(lambda x: associative_scan(jnp.maximum, x))(values)
    drawdowns = (values - running_max) / running_max
    squared_drawdowns = jnp.square(drawdowns)
    ulcer_indices = jnp.sqrt(jnp.mean(squared_drawdowns, axis=1))
    return -jnp.mean(ulcer_indices)


@partial(jit, static_argnums=(0,))
def _calculate_return_value(
    return_val, reserves, local_prices, value_over_time, initial_reserves=None
):
    """Helper function to calculate different return metrics based on the specified return_val."""

    if return_val == "reserves":
        return {"reserves": reserves}

    pool_returns = None
    if return_val in ["sharpe", "returns", "returns_over_hodl"]:
        pool_returns = jnp.diff(value_over_time) / value_over_time[:-1]
    if return_val == "daily_sharpe":
        daily_returns = (
            jnp.diff(value_over_time[::24 * 60])
            / value_over_time[::24 * 60][:-1]
        )
    return_metrics = {
        # "sharpe": lambda: jnp.sqrt(365 * 24 * 60)
        # * (
        #     (pool_returns - ((1.05 ** (1.0 / (60 * 24 * 365)) - 1) + 1) - 1.0).mean()
        #     / pool_returns.std()
        # ),
        "sharpe": lambda: jnp.sqrt(365 * 24 * 60)
        * ((pool_returns).mean() / pool_returns.std()),
        "daily_sharpe": lambda: jnp.sqrt(365)
        * (daily_returns.mean() / daily_returns.std()),
        "daily_log_sharpe": lambda: _daily_log_sharpe(value_over_time),
        "returns": lambda: value_over_time[-1] / value_over_time[0] - 1.0,
        "returns_over_hodl": lambda: (
            value_over_time[-1]
            / (stop_gradient(initial_reserves) * local_prices[-1]).sum()
            - 1.0
        ),
        "returns_over_uniform_hodl": lambda: (
            value_over_time[-1]
            / (stop_gradient((initial_reserves * local_prices[0]).sum()/(reserves.shape[1]*local_prices[0])) * local_prices[-1]).sum()
            - 1.0
        ),
        "greatest_draw_down": lambda: jnp.min(value_over_time - value_over_time[0])
        / value_over_time[0],
        "value": lambda: value_over_time,
        "weekly_max_drawdown": lambda: _calculate_max_drawdown(
            value_over_time, duration=7 * 24 * 60
        ),
        "daily_var_95%": lambda: _calculate_var(
            value_over_time, percentile=5.0, duration=24 * 60
        ),
        "daily_var_95%_trad": lambda: _calculate_var_trad(
            value_over_time, percentile=5.0, duration=24 * 60
        ),
        "weekly_var_95%": lambda: _calculate_var(
            value_over_time, percentile=5.0, duration=7 * 24 * 60
        ),
        "weekly_var_95%_trad": lambda: _calculate_var_trad(
            value_over_time, percentile=5.0, duration=7 * 24 * 60
        ),
        "daily_var_99%": lambda: _calculate_var(
            value_over_time, percentile=1.0, duration=24 * 60
        ),
        "daily_var_99%_trad": lambda: _calculate_var_trad(
            value_over_time, percentile=1.0, duration=24 * 60
        ),
        "weekly_var_99%": lambda: _calculate_var(
            value_over_time, percentile=1.0, duration=7 * 24 * 60
        ),
        "weekly_var_99%_trad": lambda: _calculate_var_trad(
            value_over_time, percentile=1.0, duration=7 * 24 * 60
        ),
        "daily_raroc": lambda: _calculate_raroc(
            value_over_time, percentile=5.0, duration=24 * 60
        ),
        "weekly_raroc": lambda: _calculate_raroc(
            value_over_time, percentile=5.0, duration=7 * 24 * 60
        ),
        "daily_rovar": lambda: _calculate_rovar(
            value_over_time, percentile=5.0, duration=24 * 60
        ),
        "weekly_rovar": lambda: _calculate_rovar(
            value_over_time, percentile=5.0, duration=7 * 24 * 60
        ),
        "monthly_rovar": lambda: _calculate_rovar(
            value_over_time, percentile=5.0, duration=30 * 24 * 60
        ),
        "daily_rovar_trad": lambda: _calculate_rovar_trad(
            value_over_time, percentile=5.0, duration=24 * 60
        ),
        "weekly_rovar_trad": lambda: _calculate_rovar_trad(
            value_over_time, percentile=5.0, duration=7 * 24 * 60
        ),
        "monthly_rovar_trad": lambda: _calculate_rovar_trad(
            value_over_time, percentile=5.0, duration=30 * 24 * 60
        ),
        "ulcer": lambda: _calculate_ulcer_index(value_over_time, duration=30 * 24 * 60),
        "sterling": lambda: _calculate_sterling_ratio(
            value_over_time, duration=30 * 24 * 60
        ),
        "calmar": lambda: _calculate_calmar_ratio(value_over_time),
        "reserves_and_values": lambda: {
            "final_reserves": reserves[-1],
            "final_value": (reserves[-1] * local_prices[-1]).sum(),
            "value": value_over_time,
            "prices": local_prices,
            "reserves": reserves,
        },
    }

    if return_val not in return_metrics:
        raise NotImplementedError(f"Return value type '{return_val}' not implemented")
    return return_metrics[return_val]()


@partial(jit, static_argnums=(7, 8))
def forward_pass(
    params,
    start_index,
    prices,
    trades_array=None,
    fees_array=None,
    gas_cost_array=None,
    arb_fees_array=None,
    pool=None,
    static_dict={
        "bout_length": 1000,
        "maximum_change": 1.0,
        "n_assets": 3,
        "chunk_period": 60,
        "weight_interpolation_period": 60,
        "return_val": "reserves",
        "rule": "momentum",
        "run_type": "normal",
        "max_memory_days": 365.0,
        "initial_pool_value": 1000000.0,
        "fees": 0.0,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "all_sig_variations": None,
        "weight_interpolation_method": "linear",
        "training_data_kind": "historic",
        "arb_frequency": 1,
        "do_trades": False,
    },
):
    """
    Simulates a forward pass of a liquidity pool using specified parameters and market data.

    This function models the behavior of a liquidity pool over a given period, 
    considering various factors such as fees, gas costs, and arbitrage fees.
    It calculates reserves and other metrics based on the provided parameters and market prices.

    Parameters
    ----------
    params : dict
        A dictionary containing the parameters for the simulation, 
        such as initial weights and other configuration settings.

    start_index : array-like
        The starting index for the simulation, used to slice the price data.

    prices : array-like
        A 2D array of market prices for the assets involved in the simulation.

    trades_array : array-like, optional
        An array of trades to be considered in the simulation. Defaults to None.

    fees_array : array-like, optional
        An array of fees to be applied during the simulation. Defaults to None.

    gas_cost_array : array-like, optional
        An array of gas costs to be considered in the simulation. Defaults to None.

    arb_fees_array : array-like, optional
        An array of arbitrage fees to be applied during the simulation. Defaults to None.

    pool : object
        An instance of a pool object that provides methods 
        to calculate reserves based on the inputs. 
        Must be provided.

    static_dict : dict, optional
        A dictionary of static configuration values for the simulation, such as bout length, 
        number of assets, and return value type. Defaults to a predefined set of values.

    Returns
    -------
    dict or float
        Depending on the `return_val` specified in `static_dict`, the function returns 
        different types of results:

        - "reserves": A dictionary containing the reserves over time.

        - "sharpe": The Sharpe ratio of the pool returns.

        - "returns": The total return over the simulation period.

        - "returns_over_hodl": The return over a hold strategy.

        - "greatest_draw_down": The greatest drawdown during the simulation.

        - "alpha": Not implemented.

        - "value": The value of the pool over time.

        - "reserves_and_values": A dictionary containing final reserves, final value, 
          value over time, prices, and reserves.

    Raises
    ------
    ValueError
        If the `pool` is not provided.

    NotImplementedError
        If the `return_val` is set to "alpha" or any other unsupported value.

    Notes
    -----
    - The function is decorated with `jax.jit` for performance optimization, 
      with static arguments specified for JIT compilation.

    - The function handles different cases for fees and trades, 
      adjusting the calculation method accordingly:

      1. If any of `fees_array`, `gas_cost_array`, `arb_fees_array`, 
         or `trades_array` is provided, it uses `pool.calculate_reserves_with_dynamic_inputs`.

      2. If any of `fees`, `gas_cost`, or `arb_fees` in `static_dict` is a nonzero scalar value, 
         it uses `pool.calculate_reserves_with_fees`.

      3. If all fees and costs are zero and no trades are provided, 
         it uses `pool.calculate_reserves_zero_fees`.

    - The function supports different types of return values, 
      allowing for flexible output based on the simulation needs.

    - The `arb_frequency` in `static_dict` can alter the frequency of arbitrage operations, 
      affecting the reserves calculation and this size of returned arrays.

    Examples
    --------
    >>> forward_pass(params, start_index, prices, pool=my_pool)
    {'reserves': array([...])}
    """

    # 'pool' has default of None only to handle how partial function
    # evaluation works with jitted functions in jax. If no pool is provided
    # the forward pass cannot be performed.
    if pool is None:
        raise ValueError("Pool must be provided to forward_pass")
    training_data_kind = static_dict["training_data_kind"]
    minimum_weight = static_dict.get("minimum_weight")
    n_assets = static_dict["n_assets"]
    return_val = static_dict["return_val"]
    bout_length = static_dict["bout_length"]

    if minimum_weight is None:
        minimum_weight = 0.1 / n_assets

    if training_data_kind == "mc":
        # do 'mc'-level indexing now
        prices = dynamic_slice(
            prices, (0, 0, start_index[-1]), (prices.shape[0], prices.shape[1], 1)
        )[:, :, 0]
        start_index = start_index[0:2]

    # Now we can calculate the reserves over time useing the pool.
    # We have to handle three cases:
    # 1. Any of Fees, gas costs, and arb fees are provided as arrays, or trades are provided
    # 2. Any of Fees, gas costs, and arb fees are nonzero scalar values, with no trades provided
    # 3. Fees, gas costs, and arb fees are all zero, with no trades provided
    if any(
        ele is not None
        for ele in [fees_array, gas_cost_array, arb_fees_array, trades_array]
    ):
        # Case 1, at least one of fees, gas costs, or arb fees is not None
        if fees_array is None:
            fees_array = jnp.array([static_dict["fees"]])
        if gas_cost_array is None:
            gas_cost_array = jnp.array([static_dict["gas_cost"]])
        if arb_fees_array is None:
            arb_fees_array = jnp.array([static_dict["arb_fees"]])
        reserves = pool.calculate_reserves_with_dynamic_inputs(
            params,
            static_dict,
            prices,
            start_index,
            fees_array=fees_array,
            arb_thresh_array=gas_cost_array,
            arb_fees_array=arb_fees_array,
            trade_array=trades_array,
        )
    elif True in (
        ele > 0.0
        for ele in [
            static_dict["fees"],
            static_dict["gas_cost"],
            static_dict["arb_fees"],
        ]
    ):
        # Case 2, at least one of fees, gas costs, or arb fees is a nonzero scalar value
        reserves = pool.calculate_reserves_with_fees(
            params, static_dict, prices, start_index
        )
    else:
        reserves = pool.calculate_reserves_zero_fees(
            params, static_dict, prices, start_index
        )

    if static_dict["arb_frequency"] != 1:
        reserves = jnp.repeat(
            reserves,
            static_dict["arb_frequency"],
            axis=0,
            total_repeat_length=local_prices.shape[0],
        )

    if return_val == "reserves":
        return {
            "reserves": reserves,
        }
    local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
    value_over_time = jnp.sum(jnp.multiply(reserves, local_prices), axis=-1)
    if return_val == "reserves_and_values":
        return_dict = {
            "final_reserves": reserves[-1],
            "final_value": (reserves[-1] * local_prices[-1]).sum(),
            "value": value_over_time,
            "prices": local_prices,
            "reserves": reserves,
            "weights": pool.calculate_weights(
                params, static_dict, prices, start_index, additional_oracle_input=None
            ) if hasattr(pool, "calculate_weights") else None,
            "raw_weight_outputs": pool.calculate_raw_weights_outputs(
                params, static_dict, prices, additional_oracle_input=None
            ) if hasattr(pool, "calculate_raw_weights_outputs") else None,
        }
        if hasattr(pool, "calculate_readouts"):
            return_dict.update({
                "readouts": pool.calculate_readouts(
                    params, static_dict, prices, start_index, additional_oracle_input=None
                )
            })
        return return_dict
    return _calculate_return_value(
        return_val,
        reserves,
        local_prices,
        value_over_time,
        initial_reserves=reserves[0],
    )


@partial(jit, static_argnums=(7, 8))
def forward_pass_nograd(
    params,
    start_index,
    prices,
    trades_array=None,
    fees_array=None,
    gas_cost_array=None,
    arb_fees_array=None,
    pool=None,
    static_dict={
        "bout_length": 1000,
        "maximum_change": 1.0,
        "n_assets": 3,
        "chunk_period": 60,
        "weight_interpolation_period": 60,
        "return_val": "reserves",
        "rule": "momentum",
        "run_type": "normal",
        "max_memory_days": 365.0,
        "initial_pool_value": 1000000.0,
        "fees": 0.0,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "all_sig_variations": None,
        "weight_interpolation_method": "linear",
        "training_data_kind": "historic",
        "arb_frequency": 1,
        "do_trades": False,
    },
):
    """
    Simulates a forward pass of a liquidity pool without gradient tracking
    using specified parameters and market data.

    This function models the behavior of a liquidity pool over a given period,
    similar to `forward_pass`, but ensures that no gradients are tracked
    for the input parameters and data. It is useful
    for scenarios where gradient computation is not required, such as evaluation or inference.

    Parameters
    ----------
    params : dict
        A dictionary containing the parameters for the simulation,
        such as initial weights and other configuration settings.

    start_index : array-like
        The starting index for the simulation, used to slice the price data.

    prices : array-like
        A 2D array of market prices for the assets involved in the simulation.

    trades_array : array-like, optional
        An array of trades to be considered in the simulation. Defaults to None.

    fees_array : array-like, optional
        An array of fees to be applied during the simulation. Defaults to None.

    gas_cost_array : array-like, optional
        An array of gas costs to be considered in the simulation. Defaults to None.

    arb_fees_array : array-like, optional
        An array of arbitrage fees to be applied during the simulation. Defaults to None.

    pool : object
        An instance of a pool object that provides methods
        to calculate reserves based on the inputs.
        Must be provided.

    static_dict : dict, optional
        A dictionary of static configuration values for the simulation, such as bout length,
        number of assets, and return value type. Defaults to a predefined set of values.

    Returns
    -------
    dict or float
        Depending on the `return_val` specified in `static_dict`, the function returns
        different types of results:

        - "reserves": A dictionary containing the reserves over time.

        - "sharpe": The Sharpe ratio of the pool returns.

        - "returns": The total return over the simulation period.

        - "returns_over_hodl": The return over a hold strategy.

        - "greatest_draw_down": The greatest drawdown during the simulation.

        - "alpha": Not implemented.

        - "value": The value of the pool over time.

        - "reserves_and_values": A dictionary containing final reserves, final value,
          value over time, prices, and reserves.

    Raises
    ------
    ValueError
        If the `pool` is not provided.

    NotImplementedError
        If the `return_val` is set to "alpha" or any other unsupported value.

    Notes
    -----
    - The function is decorated with `jax.jit` for performance optimization,
      with static arguments specified for JIT compilation.

    - The function handles different cases for fees and trades,
      adjusting the calculation method accordingly:

      1. If any of `fees_array`, `gas_cost_array`, `arb_fees_array`,
         or `trades_array` is provided, it uses `pool.calculate_reserves_with_dynamic_inputs`.

      2. If any of `fees`, `gas_cost`, or `arb_fees` in `static_dict` is a nonzero scalar value,
         it uses `pool.calculate_reserves_with_fees`.

      3. If all fees and costs are zero and no trades are provided,
         it uses `pool.calculate_reserves_zero_fees`.

    - The function supports different types of return values,
      allowing for flexible output based on the simulation needs.

    - The `arb_frequency` in `static_dict` can alter the frequency of arbitrage operations,
      affecting the reserves calculation and this size of returned arrays.

    - The function uses `jax.lax.stop_gradient` to ensure that no gradients are tracked
        for the input parameters and data.

    Examples
    --------
    >>> forward_pass_nograd(params, start_index, prices, pool=my_pool)
    {'reserves': array([...])}
    """
    params = {k: stop_gradient(v) for k, v in params.items()}
    start_index = stop_gradient(start_index)
    prices = stop_gradient(prices)
    return forward_pass(
        params,
        start_index,
        prices,
        trades_array,
        fees_array,
        gas_cost_array,
        arb_fees_array,
        pool,
        static_dict,
    )
