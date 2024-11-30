# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

from jax.tree_util import tree_map

# jax.set_cpu_device_count(n)
# print(devices("cpu"))

import jax.numpy as jnp

import numpy as np
import pandas as pd
import math
from itertools import product

import json
import hashlib
from quantammsim.utils.data_processing.historic_data_utils import get_data_dict
from quantammsim.core_simulator.windowing_utils import (
    raw_fee_like_amounts_to_fee_like_array,
    raw_trades_to_trade_array,
)
from quantammsim.apis.rest_apis.simulator_dtos.simulation_run_dto import (
    LiquidityPoolCoinDto,
    SimulationResultTimestepDto,
)
import time


class Hashabledict(dict):
    """A hashable dictionary class that enables using dictionaries as dictionary keys.

    This class extends the built-in dict class to make dictionaries hashable by
    implementing the __hash__ and __eq__ methods. The hash is computed based on a
    sorted tuple of key-value pairs.

    Methods
    -------
    __key()
        Returns a tuple of sorted key-value pairs representing the dictionary.
    __hash__()
        Returns an integer hash value for the dictionary.
    __eq__(other)
        Checks equality between this dictionary and another by comparing their sorted
        key-value pairs.

    Examples
    --------
    >>> d1 = Hashabledict({'a': 1, 'b': 2})
    >>> d2 = Hashabledict({'b': 2, 'a': 1})
    >>> hash(d1) == hash(d2)
    True
    >>> d1 == d2
    True
    >>> d3 = {d1: 'value'}  # Can use as dictionary key
    """

    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


def get_run_location(run_fingerprint):
    """Generate a unique run location identifier based on the run fingerprint.

    This function creates a unique identifier for a simulation run by hashing the
    run_fingerprint dictionary. The run_fingerprint contains configuration parameters
    that define the simulation run.

    Parameters
    ----------
    run_fingerprint : dict
        A dictionary containing the configuration parameters for the simulation run.
        This typically includes parameters like start/end dates, tokens, rules, etc.

    Returns
    -------
    str
        A string identifier in the format "run_<sha256_hash>" where the hash is
        generated from the sorted JSON representation of the run_fingerprint.

    Examples
    --------
    >>> fingerprint = {"startDate": "2023-01-01", "tokens": ["BTC", "ETH"]}
    >>> get_run_location(fingerprint)
    'run_8d147a1f8b8...'
    """
    run_location = "run_" + str(
        hashlib.sha256(
            json.dumps(run_fingerprint, sort_keys=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
    )
    return run_location


def nan_rollback(grads, params, old_params):
    """Handles NaN values in gradients by rolling back to previous parameter values.

    This function checks for NaN values in gradients and reverts the corresponding
    parameters back to their previous values when NaNs are detected. This helps
    maintain numerical stability during optimization.

    Parameters
    ----------
    grads : dict
        Dictionary containing the current gradients
    params : dict
        Dictionary containing the current parameter values
    old_params : dict
        Dictionary containing the previous parameter values

    Returns
    -------
    dict
        Updated parameters with NaN values rolled back to previous values

    Examples
    --------
    >>> grads = {"log_k": jnp.array([[1.0, jnp.nan], [3.0, 4.0]])}
    >>> params = {"log_k": jnp.array([[0.1, 0.2], [0.3, 0.4]])}
    >>> old_params = {"log_k": jnp.array([[0.05, 0.15], [0.25, 0.35]])}
    >>> rolled_back = nan_rollback(grads, params, old_params)
    """
    if "log_k" in grads:
        bool_idx = jnp.sum(jnp.isnan(grads["log_k"]), axis=-1, keepdims=True) > 0
        return tree_map(
            lambda p, old_p: jnp.where(bool_idx, old_p, p), params, old_params
        )
    else:
        return params


def get_unique_tokens(run_fingerprint):
    """Gets unique tokens from run fingerprint including subsidiary pools.

    Extracts all tokens from the main pool and subsidiary pools in the run fingerprint,
    removes duplicates, and returns a sorted list of unique tokens.

    Parameters
    ----------
    run_fingerprint : dict
        Dictionary containing run configuration including tokens and subsidiary pools

    Returns
    -------
    list
        Sorted list of unique token symbols

    Examples
    --------
    >>> fingerprint = {
    ...     "tokens": ["BTC", "ETH"],
    ...     "subsidary_pools": [{"tokens": ["ETH", "DAI"]}]
    ... }
    >>> get_unique_tokens(fingerprint)
    ['BTC', 'DAI', 'ETH']
    """
    all_tokens = [run_fingerprint["tokens"]] + [
        cprd["tokens"] for cprd in run_fingerprint["subsidary_pools"]
    ]
    all_tokens = [item for sublist in all_tokens for item in sublist]
    unique_tokens = list(set(all_tokens))
    unique_tokens.sort()
    return unique_tokens


def split_list(lst, num_splits):
    """Splits a list into a specified number of roughly equal sublists.

    Divides a list into num_splits sublists, distributing any remainder elements
    evenly among the first sublists.

    Parameters
    ----------
    lst : list
        The input list to split
    num_splits : int
        Number of sublists to create

    Returns
    -------
    list
        List of sublists

    Examples
    --------
    >>> split_list([1,2,3,4,5], 2)
    [[1,2,3], [4,5]]
    >>> split_list([1,2,3,4,5,6], 3)
    [[1,2], [3,4], [5,6]]
    """
    # Calculate the length of each sublist
    sub_len = len(lst) // num_splits

    # Determine the number of sublists that should be one element longer
    num_longer = len(lst) % num_splits

    # Initialize variables
    result = []
    start_idx = 0

    # Iterate over the number of sublists
    for i in range(num_splits):
        # Calculate the end index of the sublist
        end_idx = start_idx + sub_len

        # If there are remaining elements to distribute, add one to the sublist length
        if num_longer > 0:
            end_idx += 1
            num_longer -= 1

        # Add the sublist to the result
        result.append(lst[start_idx:end_idx])

        # Update the start index for the next sublist
        start_idx = end_idx

    return result


def invert_permutation(perm):
    """
    Compute the inverse of a permutation.

    Given a permutation array that maps indices to their new positions,
    returns the inverse permutation that maps the new positions back to
    their original indices.

    Parameters
    ----------
    perm : numpy.ndarray
        Array representing a permutation of indices

    Returns
    -------
    numpy.ndarray
        The inverse permutation array

    Examples
    --------
    >>> perm = np.array([2,0,1])
    >>> invert_permutation(perm)
    array([1, 2, 0])
    """
    s = np.zeros(perm.size, perm.dtype)
    s[perm] = range(perm.size)
    return s


def permute_list_of_params(list_of_params, seed=0):
    """
    Randomly permute a list of parameters using a fixed random seed.

    This function takes a list of parameters and returns a new list with the same elements
    in a randomly permuted order. The permutation is deterministic based on the provided
    random seed.

    Parameters
    ----------
    list_of_params : list
        The list of parameters to permute
    seed : int, optional
        Random seed to use for reproducible permutations (default: 0)

    Returns
    -------
    list
        A new list containing the same elements as the input list but in a randomly
        permuted order

    Examples
    --------
    >>> params = [1, 2, 3, 4]
    >>> permute_list_of_params(params, seed=42)
    [3, 1, 4, 2]
    >>> permute_list_of_params(params, seed=42)  # Same seed gives same permutation
    [3, 1, 4, 2]
    """
    np.random.seed(seed)
    # permute
    idx = np.random.permutation(len(list_of_params))
    list_of_params_to_return = [list_of_params[i] for i in idx]
    return list_of_params_to_return


def unpermute_list_of_params(list_of_params):
    """
    Restore the original order of a previously permuted list of parameters.

    This function takes a list that was permuted using permute_list_of_params() and
    restores it to its original order by applying the inverse permutation with the
    same random seed.

    Parameters
    ----------
    list_of_params : list
        The permuted list of parameters to restore to original order

    Returns
    -------
    list
        A new list containing the same elements as the input list but restored to
        their original order before permutation

    Examples
    --------
    >>> params = [1, 2, 3, 4]
    >>> permuted = permute_list_of_params(params)  # [3, 1, 4, 2]
    >>> unpermute_list_of_params(permuted)  # Restores original order
    [1, 2, 3, 4]
    """
    # unpermute
    idx = np.random.permutation(len(list_of_params))
    idx_unpermute = invert_permutation(idx)
    list_of_params_to_return = [list_of_params[i] for i in idx_unpermute]
    return list_of_params_to_return


def get_trades_and_fees(
    run_fingerprint, raw_trades, fees_df, gas_cost_df, arb_fees_df, do_test_period=False
):
    """
    Process trade and fee data for a simulation run.

    Takes raw trades, fees, gas costs and arbitrage fees and converts them into arrays
    suitable for simulation. Handles both training and test periods if specified.

    Parameters
    ----------
    run_fingerprint : dict
        Dictionary containing run configuration including start/end dates and tokens
    raw_trades : pd.DataFrame, optional
        DataFrame containing raw trade data
    fees_df : pd.DataFrame, optional
        DataFrame containing fee data
    gas_cost_df : pd.DataFrame, optional
        DataFrame containing gas cost data
    arb_fees_df : pd.DataFrame, optional
        DataFrame containing arbitrage fee data
    do_test_period : bool, optional
        Whether to process data for a test period after training period (default False)

    Returns
    -------
    dict
        Contains processed arrays for trades, fees, gas costs and arb fees for both
        training and test periods as applicable
    """
    # Process raw trades if provided
    if raw_trades is not None:
        train_period_trades = raw_trades_to_trade_array(
            raw_trades,
            start_date_string=run_fingerprint["startDateString"],
            end_date_string=run_fingerprint["endDateString"],
            tokens=get_unique_tokens(run_fingerprint),
        )
        if do_test_period:
            test_period_trades = raw_trades_to_trade_array(
                raw_trades,
                start_date_string=run_fingerprint["endDateString"],
                end_date_string=run_fingerprint["endTestDateString"],
                tokens=get_unique_tokens(run_fingerprint),
            )
        do_trades = True
    else:
        train_period_trades = None
        test_period_trades = None
        do_trades = False
    # Process fees, gas costs, and arb fees if provided
    fees_array = (
        raw_fee_like_amounts_to_fee_like_array(
            fees_df,
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
            names=["fees"],
            fill_method="ffill",
        )
        if fees_df is not None
        else None
    )
    if do_test_period:
        test_fees_array = (
            raw_fee_like_amounts_to_fee_like_array(
                fees_df,
                run_fingerprint["startDateString"],
                run_fingerprint["endDateString"],
                names=["fees"],
                fill_method="ffill",
            )
            if fees_df is not None
            else None
        )

    gas_cost_array = (
        raw_fee_like_amounts_to_fee_like_array(
            gas_cost_df,
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
            fill_method="ffill",
        )
        if gas_cost_df is not None
        else None
    )
    if do_test_period:
        test_gas_cost_array = (
            raw_fee_like_amounts_to_fee_like_array(
                gas_cost_df,
                run_fingerprint["endDateString"],
                run_fingerprint["endTestDateString"],
                fill_method="ffill",
            )
            if gas_cost_df is not None
            else None
        )

    arb_fees_array = (
        raw_fee_like_amounts_to_fee_like_array(
            arb_fees_df,
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
            fill_method="ffill",
        )
        if arb_fees_df is not None
        else None
    )
    if do_test_period:
        test_arb_fees_array = (
            raw_fee_like_amounts_to_fee_like_array(
                arb_fees_df,
                run_fingerprint["endDateString"],
                run_fingerprint["endTestDateString"],
                fill_method="ffill",
            )
            if arb_fees_df is not None
            else None
        )
    if do_test_period:
        return {
            "train_period_trades": train_period_trades,
            "test_period_trades": test_period_trades,
            "fees_array": fees_array,
            "gas_cost_array": gas_cost_array,
            "arb_fees_array": arb_fees_array,
            "test_fees_array": test_fees_array,
            "test_gas_cost_array": test_gas_cost_array,
            "test_arb_fees_array": test_arb_fees_array,
        }
    else:
        return {
            "train_period_trades": train_period_trades,
            "fees_array": fees_array,
            "gas_cost_array": gas_cost_array,
            "arb_fees_array": arb_fees_array,
        }


def create_daily_unix_array(start_date_str, end_date_str):
    """
    Creates an array of daily Unix timestamps in milliseconds between two dates.

    Args:
        start_date_str (str): Start date string in format 'YYYY-MM-DD HH:MM:SS'
        end_date_str (str): End date string in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        list: Array of Unix timestamps in milliseconds for each day between start and end dates
    """
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    # Create a date range ending the day before the end_date
    date_range = pd.date_range(start=start_date_str, end=end_date, freq="D")
    daily_unix_values = date_range.view("int64") // 10**6
    return daily_unix_values.tolist()


def create_time_step(row, unix_values, tokens, index):
    """
    Creates a SimulationResultTimestepDto object for a single time step.

    Args:
        row (pd.Series): Row containing prices, reserves and weights data for this timestep
        unix_values (list): List of Unix timestamps in milliseconds
        tokens (list): List of token symbols
        index (int): Index of current timestep

    Returns:
        SimulationResultTimestepDto: Object containing timestamp and coin data for this timestep
    """
    timeStep = SimulationResultTimestepDto(unix_values[index], [], 0)

    for coinIndex, token in enumerate(tokens):
        coin = LiquidityPoolCoinDto()
        coin.coinCode = token
        coin.currentPrice = row["prices"][coinIndex].item()
        coin.amount = row["reserves"][coinIndex].item()
        coin.weight = row["weights"][coinIndex].item()
        coin.marketValue = coin.currentPrice * coin.amount
        timeStep.coinsHeld.append(coin)

    return timeStep


def optimized_output_conversion(simulationRunDto, outputDict, tokens):
    """
    Converts simulation output dictionary to a list of SimulationResultTimestepDto objects.

    Args:
        simulationRunDto (SimulationRunDto): Object containing simulation run parameters
        outputDict (dict): Dictionary containing simulation output data including prices, reserves, and values
        tokens (list): List of token symbols used in simulation

    Returns:
        list: List of SimulationResultTimestepDto objects containing timestep data

    The function:
    1. Creates Unix timestamps for each day between start and end dates
    2. Downsamples simulation data from minutes to daily frequency
    3. Calculates token weights from reserves, prices and total value
    4. Combines data into timestep DTOs with coin holdings and values
    """
    print(simulationRunDto.startDateString)
    print(simulationRunDto.endDateString)
    print(tokens)
    # Create a date range with daily frequency and convert to Unix timestamps in milliseconds
    unix_values = create_daily_unix_array(
        simulationRunDto.startDateString, simulationRunDto.endDateString
    )

    # Convert outputDict data to pandas DataFrame for efficient slicing
    prices_df = pd.DataFrame(outputDict["prices"])[::1440]
    reserves_df = pd.DataFrame(outputDict["reserves"])[::1440]
    values_df = pd.DataFrame(outputDict["value"])[::1440]
    print(len(outputDict["prices"]))
    print(len(outputDict["reserves"]))
    print(len(outputDict["value"]))
    print("--------------")
    print(len(outputDict["prices"][::1440]))
    print(len(outputDict["reserves"][::1440]))
    print(len(outputDict["value"][::1440]))
    # note that the returned weights are empirical weights, not calculated weights
    # this is because the calculated weights are not returned in the outputDict as
    # they are not guaranteed to exist for all possible pool types
    weights_df = pd.DataFrame(
        outputDict["reserves"]
        * outputDict["prices"]
        / outputDict["value"][:, np.newaxis]
    )[::1440]

    print("prices_df: ", len(prices_df))
    print("reserves_df: ", len(reserves_df))
    print("weights_df: ", len(weights_df))
    print("unix_values: ", len(unix_values))

    # Combine DataFrames
    combined_df = pd.concat(
        [prices_df, reserves_df, weights_df],
        axis=1,
        keys=["prices", "reserves", "weights"],
    )

    print(len(unix_values))
    print(len(combined_df))

    # Check if the length of unix_values matches the number of rows in combined_df
    if len(unix_values) != len(combined_df):
        print(len(unix_values))
        print(len(combined_df))
        raise ValueError(
            "The length of unix_values does not match the number of rows in combined_df"
        )

    # Ensure index alignment by resetting index
    combined_df = combined_df.reset_index(drop=True)

    # Convert DataFrame to list of DTO objects using apply
    resultTimeSteps = combined_df.apply(
        lambda row: create_time_step(row, unix_values, tokens, row.name), axis=1
    ).tolist()

    return resultTimeSteps
