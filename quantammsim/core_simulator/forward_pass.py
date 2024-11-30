# BATCH_SIZE=32
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count='+str(BATCH_SIZE)

# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
from jax.lib.xla_bridge import default_backend
from jax import devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

# jax.set_cpu_device_count(n)
# print(devices("cpu"))

import jax.numpy as jnp
from jax import jit, vmap, devices
from jax.lax import stop_gradient, dynamic_slice


import numpy as np

from functools import partial

np.seterr(all="raise")
np.seterr(under="print")

# TODO above is all from jax utils, tidy up required


def _calculate_max_drawdown(value_over_time, duration=7 * 24 * 60):
    """Calculate maximum drawdown on a chosen basis."""
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    values = value_over_time_truncated.reshape(-1, duration)
    running_max = jnp.maximum.accumulate(values, axis=1)
    drawdowns = (values - running_max) / running_max
    max_drawdowns = jnp.min(drawdowns, axis=1)
    return jnp.min(max_drawdowns)


def _calculate_var(value_over_time, percentile=5.0, duration=24 * 60):
    """Calculate 95% VaR using intraday returns."""
    n_complete_chunks = (len(value_over_time) // duration) * duration
    value_over_time_truncated = value_over_time[:n_complete_chunks]
    values = value_over_time_truncated.reshape(-1, duration)
    returns = jnp.diff(values, axis=-1) / values[:, :-1]
    var = vmap(lambda x: jnp.percentile(x, percentile))(returns)
    return jnp.mean(var)


def _calculate_var_trad(value_over_time, percentile=5.0, duration=24 * 60):
    """Calculate traditional 95% VaR using daily returns."""
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
    return annualized_return / annualized_var


def _calculate_return_value(
    return_val, reserves, local_prices, value_over_time, initial_reserves=None
):
    """Helper function to calculate different return metrics based on the specified return_val."""

    if return_val == "reserves":
        return {"reserves": reserves}

    pool_returns = None
    if return_val in ["sharpe", "returns", "returns_over_hodl"]:
        pool_returns = jnp.diff(value_over_time) / value_over_time[:-1]

    return_metrics = {
        # "sharpe": lambda: jnp.sqrt(365 * 24 * 60)
        # * (
        #     (pool_returns - ((1.05 ** (1.0 / (60 * 24 * 365)) - 1) + 1) - 1.0).mean()
        #     / pool_returns.std()
        # ),
        "sharpe": lambda: jnp.sqrt(365 * 24 * 60)
        * ((pool_returns).mean() / pool_returns.std()),
        "returns": lambda: value_over_time[-1] / value_over_time[0] - 1.0,
        "returns_over_hodl": lambda: (
            value_over_time[-1]
            / (stop_gradient(initial_reserves) * local_prices[-1]).sum()
            - 1.0
        ),
        "greatest_draw_down": lambda: jnp.min(value_over_time - value_over_time[0])
        / value_over_time[0],
        "value": lambda: value_over_time,
        "weekly_max_drawdown": lambda: _calculate_weekly_max_drawdown(value_over_time),
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
        Gradients will not be tracked for these parameters.
    start_index : array-like
        The starting index for the simulation, used to slice the price data. 
        Gradients will not be tracked for this index.
    prices : array-like
        A 2D array of market prices for the assets involved in the simulation. 
        Gradients will not be tracked for these prices.
    trades_array : array-like, optional
        An array of trades to be considered in the simulation. Defaults to None.
    fees_array : array-like, optional
        An array of fees to be applied during the simulation. Defaults to None.
    gas_cost_array : array-like, optional
        An array of gas costs to be considered in the simulation. Defaults to None.
    arb_fees_array : array-like, optional
        An array of arbitrage fees to be applied during the simulation. Defaults to None.
    pool : object
        An instance of a pool object that provides methods to calculate 
        reserves based on the inputs. 
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
        - "constant": A constant value based on reserves and weights.
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
    - The function uses `jax.lax.stop_gradient` to ensure that no gradients are tracked 
        for the input parameters and data.
    - The function handles different cases for fees and trades, adjusting the calculation method 
        accordingly:
      1. If any of `fees_array`, `gas_cost_array`, `arb_fees_array`, 
        or `trades_array` is provided, it uses `pool.calculate_reserves_with_dynamic_inputs`.
      2. If any of `fees`, `gas_cost`, or `arb_fees` in `static_dict` is a nonzero scalar value, 
        it uses `pool.calculate_reserves_with_fees`.
      3. If all fees and costs are zero and no trades are provided, 
        it uses `pool.calculate_reserves_zero_fees`.
    - The function supports different types of return values, 
        allowing for flexible output based on the simulation needs.
    - The `arb_frequency` in `static_dict` can alter the frequency of arbitrage operations, 
        affecting the reserves calculation.

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
