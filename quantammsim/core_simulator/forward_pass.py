import os
import glob

# BATCH_SIZE=32
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count='+str(BATCH_SIZE)

# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
from jax.lib.xla_bridge import default_backend
from jax import local_device_count, devices

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
from jax import jit, vmap
from jax import devices, device_put
from jax.lax import stop_gradient, dynamic_slice
from jax import hessian, lax

from jax.nn import softmax


import numpy as np
import math

from quantammsim.training.hessian_trace import hessian_trace
from functools import partial
import gc

from copy import deepcopy
from itertools import product

from quantammsim.core_simulator.param_utils import lamb_to_memory


np.seterr(all="raise")
np.seterr(under="print")

# TODO above is all from jax utils, tidy up required


@partial(jit, static_argnums=(7,8))
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

    This function models the behavior of a liquidity pool over a given period, considering various factors such as fees, gas costs, and arbitrage fees. It calculates reserves and other metrics based on the provided parameters and market prices.

    Parameters
    ----------
    params : dict
        A dictionary containing the parameters for the simulation, such as initial weights and other configuration settings.
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
        An instance of a pool object that provides methods to calculate reserves based on the inputs. Must be provided.
    static_dict : dict, optional
        A dictionary of static configuration values for the simulation, such as bout length, number of assets, and return value type. Defaults to a predefined set of values.

    Returns
    -------
    dict or float
        Depending on the `return_val` specified in `static_dict`, the function returns different types of results:
        - "reserves": A dictionary containing the reserves over time.
        - "sharpe": The Sharpe ratio of the pool returns.
        - "returns": The total return over the simulation period.
        - "returns_over_hodl": The return over a hold strategy.
        - "greatest_draw_down": The greatest drawdown during the simulation.
        - "alpha": Not implemented.
        - "value": The value of the pool over time.
        - "reserves_and_values": A dictionary containing final reserves, final value, value over time, prices, and reserves.

    Raises
    ------
    ValueError
        If the `pool` is not provided.
    NotImplementedError
        If the `return_val` is set to "alpha" or any other unsupported value.

    Notes
    -----
    - The function is decorated with `jax.jit` for performance optimization, with static arguments specified for JIT compilation.
    - The function handles different cases for fees and trades, adjusting the calculation method accordingly:
      1. If any of `fees_array`, `gas_cost_array`, `arb_fees_array`, or `trades_array` is provided, it uses `pool.calculate_reserves_with_dynamic_inputs`.
      2. If any of `fees`, `gas_cost`, or `arb_fees` in `static_dict` is a nonzero scalar value, it uses `pool.calculate_reserves_with_fees`.
      3. If all fees and costs are zero and no trades are provided, it uses `pool.calculate_reserves_zero_fees`.
    - The function supports different types of return values, allowing for flexible output based on the simulation needs.
    - The `arb_frequency` in `static_dict` can alter the frequency of arbitrage operations, affecting the reserves calculation and this size of returned arrays.

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
    all_sig_variations = jnp.array(static_dict["all_sig_variations"])
    training_data_kind = static_dict["training_data_kind"]
    minimum_weight = static_dict.get("minimum_weight")
    n_assets = static_dict["n_assets"]
    return_val = static_dict["return_val"]
    bout_length = static_dict["bout_length"]

    if minimum_weight == None:
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
    if any(ele is not None for ele in [fees_array, gas_cost_array, arb_fees_array, trades_array]):
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
        reserves = pool.calculate_reserves_zero_fees(params, static_dict, prices, start_index)

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
    pool_returns = jnp.diff(value_over_time) / value_over_time[:-1]
    if return_val == "sharpe":
        # calculate sharpe ratio over time
        minute_level_rf = ((1.05 ** (1.0 / (60 * 24 * 365)) - 1) + 1) - 1.0
        sharpe_ratio_minute = (
            pool_returns - minute_level_rf
        ).mean() / pool_returns.std()
        sharpe_ratio_annual = jnp.sqrt(365 * 24 * 60) * sharpe_ratio_minute
        return sharpe_ratio_annual
    elif return_val == "returns":
        return value_over_time[-1] / value_over_time[0] - 1.0
    elif return_val == "returns_over_hodl":
        return (
            value_over_time[-1] / (stop_gradient(initial_reserves) * local_prices[-1]).sum()
            - 1.0
        )
    elif return_val == "greatest_draw_down":
        return jnp.min(value_over_time - value_over_time[0]) / value_over_time[0]
    elif return_val == "alpha":
        raise NotImplementedError
    elif return_val == "value":
        return value_over_time
    elif return_val == "reserves_and_values":
        return {
            "final_reserves": reserves[-1],
            "final_value": (reserves[-1] * local_prices[-1]).sum(),
            "value": value_over_time,
            "prices": local_prices,
            "reserves": reserves,
        }
    else:
        raise NotImplementedError


@partial(jit, static_argnums=(7,8))
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
    Simulates a forward pass of a liquidity pool without gradient tracking using specified parameters and market data.

    This function models the behavior of a liquidity pool over a given period, similar to `forward_pass`, but ensures that no gradients are tracked for the input parameters and data. It is useful for scenarios where gradient computation is not required, such as evaluation or inference.

    Parameters
    ----------
    params : dict
        A dictionary containing the parameters for the simulation, such as initial weights and other configuration settings. Gradients will not be tracked for these parameters.
    start_index : array-like
        The starting index for the simulation, used to slice the price data. Gradients will not be tracked for this index.
    prices : array-like
        A 2D array of market prices for the assets involved in the simulation. Gradients will not be tracked for these prices.
    trades_array : array-like, optional
        An array of trades to be considered in the simulation. Defaults to None.
    fees_array : array-like, optional
        An array of fees to be applied during the simulation. Defaults to None.
    gas_cost_array : array-like, optional
        An array of gas costs to be considered in the simulation. Defaults to None.
    arb_fees_array : array-like, optional
        An array of arbitrage fees to be applied during the simulation. Defaults to None.
    pool : object
        An instance of a pool object that provides methods to calculate reserves based on the inputs. Must be provided.
    static_dict : dict, optional
        A dictionary of static configuration values for the simulation, such as bout length, number of assets, and return value type. Defaults to a predefined set of values.

    Returns
    -------
    dict or float
        Depending on the `return_val` specified in `static_dict`, the function returns different types of results:
        - "reserves": A dictionary containing the reserves over time.
        - "sharpe": The Sharpe ratio of the pool returns.
        - "returns": The total return over the simulation period.
        - "returns_over_hodl": The return over a hold strategy.
        - "greatest_draw_down": The greatest drawdown during the simulation.
        - "alpha": Not implemented.
        - "value": The value of the pool over time.
        - "constant": A constant value based on reserves and weights.
        - "reserves_and_values": A dictionary containing final reserves, final value, value over time, prices, and reserves.

    Raises
    ------
    ValueError
        If the `pool` is not provided.
    NotImplementedError
        If the `return_val` is set to "alpha" or any other unsupported value.

    Notes
    -----
    - The function is decorated with `jax.jit` for performance optimization, with static arguments specified for JIT compilation.
    - The function uses `jax.lax.stop_gradient` to ensure that no gradients are tracked for the input parameters and data.
    - The function handles different cases for fees and trades, adjusting the calculation method accordingly:
      1. If any of `fees_array`, `gas_cost_array`, `arb_fees_array`, or `trades_array` is provided, it uses `pool.calculate_reserves_with_dynamic_inputs`.
      2. If any of `fees`, `gas_cost`, or `arb_fees` in `static_dict` is a nonzero scalar value, it uses `pool.calculate_reserves_with_fees`.
      3. If all fees and costs are zero and no trades are provided, it uses `pool.calculate_reserves_zero_fees`.
    - The function supports different types of return values, allowing for flexible output based on the simulation needs.
    - The `arb_frequency` in `static_dict` can alter the frequency of arbitrage operations, affecting the reserves calculation.

    Examples
    --------
    >>> forward_pass_nograd(params, start_index, prices, pool=my_pool)
    {'reserves': array([...])}
    """
    params = {
        k: stop_gradient(v) for k, v in params.items()
    }
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
