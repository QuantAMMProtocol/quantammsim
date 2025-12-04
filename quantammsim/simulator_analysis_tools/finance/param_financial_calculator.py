import sys, os
import numpy as np
import jax.numpy as jnp
import pandas as pd
import copy
from itertools import product

import random
from datetime import datetime, timedelta

# from quantamm.runners.jax_runners import do_run_on_historic_data, optimized_output_conversion
# import quantamm.simulator_analysis_tools.finance.financial_analysis_calculator as fac
# import quantamm.simulator_analysis_tools.finance.financial_analysis_functions as faf
# import quantamm.simulator_analysis_tools.finance.financial_analysis_utils as fau
# import quantamm.simulator_analysis_tools.finance.financial_analysis_charting as fach
# from quantamm.core_amm_simulator.param_utils import dict_of_np_to_jnp, memory_days_to_lamb
# from quantamm.utils.data_processing.datetime_utils import unixtimestamp_to_precise_datetime
# from quantamm.utils.data_processing.dtb3_data_utils import filter_dtb3_values
# from quantamm.utils.data_processing.historic_data_utils import  get_data_dict,get_historic_csv_data

from quantammsim.runners.jax_runners import do_run_on_historic_data
from quantammsim.runners.jax_runner_utils import optimized_output_conversion
import quantammsim.simulator_analysis_tools.finance.financial_analysis_calculator as fac
import quantammsim.simulator_analysis_tools.finance.financial_analysis_functions as faf
import quantammsim.simulator_analysis_tools.finance.financial_analysis_utils as fau
import quantammsim.simulator_analysis_tools.finance.financial_analysis_charting as fach
from quantammsim.core_simulator.param_utils import (
    dict_of_np_to_jnp,
    memory_days_to_lamb,
    convert_parameter_values,
    _to_bd18_string_list,
    _to_float64_list,
)
from quantammsim.utils.data_processing.datetime_utils import (
    unixtimestamp_to_precise_datetime,
    unixtimestamp_to_midnight_datetime,
    datetime_to_unixtimestamp,
)
from quantammsim.utils.data_processing.dtb3_data_utils import filter_dtb3_values
from quantammsim.utils.data_processing.historic_data_utils import (
    get_data_dict,
    get_historic_csv_data,
    get_historic_parquet_data,
)
from quantammsim.pools.creator import create_pool


def get_static_value(df, value_column):
    """Check if DataFrame has a single static value.

    Args:
        df: DataFrame to check
        value_column: Column name to check for static value

    Returns:
        float or None: The static value if found, None otherwise
    """
    if df is not None and not df.empty:
        unique_values = df[value_column].unique()
        if len(unique_values) == 1:
            return float(unique_values[0])
    return None


def slice_minutes_array(
    minutes_array, array_start_date_str, start_date_str, end_date_str
):
    """
    Slices an array of minutes for the given start and end dates.

    :param minutes_array: List of minute values starting from array_start_date
    :param array_start_date_str: The start datetime of the array as a string (YYYY-MM-DD HH:MM:SS)
    :param start_date_str: The start datetime for the slice as a string (YYYY-MM-DD HH:MM:SS)
    :param end_date_str: The end datetime for the slice as a string (YYYY-MM-DD HH:MM:SS)
    :return: Sliced array of minutes
    """
    # Convert date strings to datetime objects
    array_start_date = datetime.strptime(array_start_date_str, "%Y-%m-%d %H:%M:%S")
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")

    # Calculate the difference in minutes
    start_diff_minutes = int((start_date - array_start_date).total_seconds() / 60)
    end_diff_minutes = int((end_date - array_start_date).total_seconds() / 60)

    # Check if the indices are within the bounds of the array
    if start_diff_minutes < 0 or end_diff_minutes > len(minutes_array):
        raise IndexError("The calculated indices are out of the bounds of the array.")

    # Slice the array
    return minutes_array[start_diff_minutes:end_diff_minutes]


def process_basic_parameters(update_rule_parameters, run_fingerprint):
    """Convert update rule parameters into the required format.
    
    Args:
        update_rule_parameters: List of update rule parameter objects
        run_fingerprint: Run fingerprint dictionary
        
    Returns:
        Dict containing the converted parameters
    """
    update_rule_parameter_dict_converted = {}
    for urp in update_rule_parameters:
        if urp.name == "memory_days":
            logit_lamb_vals = []
            for tokenValue in urp.value:
                initial_lamb = memory_days_to_lamb(tokenValue, run_fingerprint["chunk_period"])
                logit_lamb = np.log(initial_lamb / (1.0 - initial_lamb))
                logit_lamb_vals.append(logit_lamb)
            update_rule_parameter_dict_converted["logit_lamb"] = np.array(
                logit_lamb_vals
            )
        elif urp.name == "k_per_day":
            update_rule_parameter_dict_converted["k"] = np.array(urp.value)
        else:
            update_rule_parameter_dict_converted[urp.name] = np.array(urp.value)
    return update_rule_parameter_dict_converted

def run_pool_simulation(simulationRunDto):
    # take in data transfer object and do run using 'update_pool' function

    tokens = [
        constituent.coinCode for constituent in simulationRunDto.pool.poolConstituents
    ]
    print(tokens)

    # initial pool constituents
    initial_value_per_token = [
        constituent.marketValue
        for constituent in simulationRunDto.pool.poolConstituents
    ]

    total_initial_value = sum(initial_value_per_token)

    initial_value_ratio = [val / total_initial_value for val in initial_value_per_token]

    initial_value_log_ratio = jnp.array([np.log(val) for val in initial_value_ratio])

    update_rule = simulationRunDto.pool.updateRule.name
    update_rule_parameters = simulationRunDto.pool.updateRule.updateRuleFactors

    # Extract chunk_period from update rule parameters if it exists
    chunk_period = None
    weight_interpolation_period = None
    for urp in update_rule_parameters:
        if urp.name == "chunk_period":
            chunk_period = urp.value
        if urp.name == "weight_interpolation_period":
            weight_interpolation_period = urp.value

    run_fingerprint = {
        "startDateString": simulationRunDto.startDateString,
        "endDateString": simulationRunDto.endDateString,
        "tokens": tokens,
        "rule": update_rule,
        "bout_offset": 14400,
        "initial_weights_logits": initial_value_log_ratio,
        "initial_pool_value": total_initial_value,
        "use_alt_lamb": False,
        "return_val": "final_reserves_value_and_weights",
        "do_arb": simulationRunDto.pool.enableAutomaticArbBots,
        "numeraire": getattr(simulationRunDto.pool, "poolNumeraireCoinCode", None),
        "chunk_period": int(chunk_period[0]),
        "weight_interpolation_period": int(weight_interpolation_period[0]),
        "arb_quality": float(next((urp.value[0] for urp in update_rule_parameters if urp.name == "arb_quality"), 0.0)),
        "noise_trader_ratio": float(next((urp.value[0] for urp in update_rule_parameters if urp.name == "noise_trader_ratio"), 0.0)),
    }

    pool = create_pool(update_rule)
    pool_class = pool.__class__

    # make dict of params
    # Use pool's parameter processor if available
    if hasattr(pool_class, "process_parameters"):
        update_rule_parameter_dict_converted = pool_class.process_parameters(
            update_rule_parameters, run_fingerprint
        )
    elif len(update_rule_parameters) > 0:
        # Fallback to basic parameter processing
        update_rule_parameter_dict_converted = process_basic_parameters(
            update_rule_parameters, run_fingerprint
        )
    else:
        update_rule_parameter_dict_converted = {}

    update_rule_parameter_dict_converted["initial_weights_logits"] = (
        initial_value_log_ratio
    )
    update_rule_parameter_dict_converted["initial_pool_value"] = total_initial_value

    time_series_fee_hook_variable = None
    for feeHook in simulationRunDto.feeHooks:
        if feeHook.hookName == "timeSeriesFeeImport":
            time_series_fee_hook_variable = feeHook
            break
    fee_steps_df = None
    if time_series_fee_hook_variable is not None:
        if len(time_series_fee_hook_variable.hookTimeSteps) > 0:
            divisor = 1
            if time_series_fee_hook_variable.unit == "bps":
                divisor = 10000
            if time_series_fee_hook_variable.unit == "%":
                divisor = 100
            fee_steps_df = pd.DataFrame({
                "unix": [step.unix for step in time_series_fee_hook_variable.hookTimeSteps],
                "fees": [float(step.value)/float(divisor) for step in time_series_fee_hook_variable.hookTimeSteps],
            })
    raw_trades = None

    if len(simulationRunDto.swapImports) > 0:
        raw_trades = pd.DataFrame({
            "unix": [swap.unix for swap in simulationRunDto.swapImports],
            "token_in": [swap.tokenIn for swap in simulationRunDto.swapImports],
            "token_out": [swap.tokenOut for swap in simulationRunDto.swapImports],
            "amount_in": [float(swap.amountIn) for swap in simulationRunDto.swapImports],
        })

    run_fingerprint["do_trades"] = raw_trades is not None

    gas_cost_df = None

    if len(simulationRunDto.gasSteps) > 0:
        gas_cost_df = pd.DataFrame({
            "unix": [gasStep.unix for gasStep in simulationRunDto.gasSteps],
            "trade_gas_cost_usd": [float(gasStep.value) for gasStep in simulationRunDto.gasSteps],
        })

    price_data_local = get_historic_parquet_data(tokens)
    print("price data local: ", price_data_local)

    # Check for static values, if found, set in run_fingerprint and remove df
    # this is to enable faster subroutines
    if (static_gas := get_static_value(gas_cost_df, "trade_gas_cost_usd")) is not None:
        run_fingerprint["gas_cost"] = static_gas
        gas_cost_df = None

    if (static_fee := get_static_value(fee_steps_df, "fees")) is not None:
        run_fingerprint["fees"] = static_fee
        fee_steps_df = None

    print("run fingerprint-------------------", run_fingerprint)
    print("update rule parameter dict converted-------------------", update_rule_parameter_dict_converted)
    outputDict = do_run_on_historic_data(
        run_fingerprint,
        update_rule_parameter_dict_converted,
        root=None,
        price_data=price_data_local,
        verbose=True,
        do_test_period=False,
        raw_trades=raw_trades,
        gas_cost_df=gas_cost_df,
        fees_df=fee_steps_df
    )
    print("outputDict: ", outputDict.keys())
    resultTimeSteps = optimized_output_conversion(simulationRunDto, outputDict, tokens)

    analysis = retrieve_simulation_run_analysis_results(
        run_fingerprint,
        update_rule_parameter_dict_converted,
        outputDict,
        price_data_local,
    )


    # add readouts to analysis
    if "readouts" in outputDict:
        readouts = outputDict["readouts"]
        analysis["readouts"] = {"values": {}, "strings": {}}
        for readout in readouts:
            analysis["readouts"]["values"][readout] = _to_float64_list(readouts[readout][-1])
            analysis["readouts"]["strings"][readout] = _to_bd18_string_list(readouts[readout][-1])
    else:
        analysis["readouts"] = None

    # Get final unix timestamp from price_data_local
    final_unix_timestamp = price_data_local.index[-1]
    
    analysis["final_unix_timestamp"] = final_unix_timestamp
    print("final unix timestamp: ", final_unix_timestamp)
    # Convert final unix timestamp to most recent midnight and get final weights
    # Get the most recent midnight before the final_unix_timestamp
    midnight_end_date_str = unixtimestamp_to_midnight_datetime(final_unix_timestamp)
    
    # Create a new run_fingerprint for the final weights run
    final_weights_fingerprint = copy.deepcopy(run_fingerprint)
    final_weights_fingerprint["endDateString"] = midnight_end_date_str
    
    # Run simulation to get final weights at midnight
    final_weights_output = do_run_on_historic_data(
        final_weights_fingerprint,
        update_rule_parameter_dict_converted,
        root=None,
        price_data=price_data_local,
        verbose=False,
        do_test_period=False,
        raw_trades=raw_trades,
        gas_cost_df=gas_cost_df,
        fees_df=fee_steps_df,
    )
    # Extract final weights from the result
    if "weights" in final_weights_output and len(final_weights_output["weights"]) > 0:
        final_weights = final_weights_output["weights"][-1]
    else:
        # Fallback to equal weights if weights not available
        n_tokens = len(tokens)
        final_weights = jnp.ones(n_tokens) / n_tokens
    
    # Store final weights in analysis
    analysis["final_weights"] = _to_float64_list(final_weights)
    analysis["final_weights_strings"] = _to_bd18_string_list(final_weights)

    update_rule_parameter_dict_converted.pop("initial_weights_logits", None)
    update_rule_parameter_dict_converted.pop("initial_pool_value", None)
    update_rule_parameter_dict_converted.pop("chunk_period", None)
    update_rule_parameter_dict_converted.pop("weight_interpolation_period", None)

    # add parameters to analysis
    analysis["jax_parameters"] = dict_of_np_to_jnp(update_rule_parameter_dict_converted)
    analysis["smart_contract_parameters"] = convert_parameter_values(
        update_rule_parameter_dict_converted, run_fingerprint
    )
    
    return {"resultTimeSteps": resultTimeSteps, "analysis": analysis}


def retrieve_simulation_run_analysis_results(
    run_fingerprint, params, portfolio_result, price_data=None, btc_price_data=None
):

    minute_index = pd.date_range(
        start=run_fingerprint["startDateString"],
        periods=len(portfolio_result["value"]),
        freq="T",
    )

    hodl_params = copy.deepcopy(params)
    hodl_fingerprint = copy.deepcopy(run_fingerprint)
    hodl_fingerprint["rule"] = "hodl"

    hodl_result = do_run_on_historic_data(
        hodl_fingerprint,
        dict_of_np_to_jnp(hodl_params),
        do_test_period=False,
        price_data=price_data,
    )

    # btc_params = copy.deepcopy(params)
    # btc_fingerprint = copy.deepcopy(run_fingerprint)
    # btc_fingerprint["tokens"] = ["BTC"]
    # btc_fingerprint["rule"] = "hodl"
    #
    # btc_result, btc_test_result = do_run_on_historic_data(
    #     btc_fingerprint, dict_of_np_to_jnp(btc_params)
    # )

    yearly_daily_rf_values = fau.convert_annual_to_daily_returns(
        filter_dtb3_values(
            "DTB3.csv",
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
        )
    )

    portfolio_daily_returns = calculate_daily_returns(
        portfolio_result["value"], run_fingerprint["startDateString"], "portfolio"
    )

    # Calculate returns for hodl
    hodl_daily_returns = calculate_daily_returns(
        hodl_result["value"], run_fingerprint["startDateString"], "hodl"
    )

    yearly_daily_rf_values = fau.convert_annual_to_daily_returns(
        filter_dtb3_values(
            "DTB3.csv",
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
        )
    )

    results = fac.perform_porfolio_financial_analysis(
        portfolio_daily_returns,
        yearly_daily_rf_values,
        run_fingerprint["startDateString"],
        [hodl_daily_returns],
        ["hodl"],
        "3M TBill (DTB3)",
    )

    return results


def process_return_array(return_array, benchmark_names):
    # Create column names
    columns = ["unix", "portfolio_returns"] + benchmark_names

    # Convert input array into DataFrame
    df = pd.DataFrame(return_array, columns=columns)

    # Convert UNIX to datetime and extract only the date (ignore time)
    df["datetime"] = pd.to_datetime(df["unix"], unit="ms")
    df["date"] = df["datetime"].dt.date

    # Sort by date and then by the original unix timestamps to ensure correct order
    df.sort_values(by=["date", "unix"], inplace=True)

    # Drop duplicates within the same day (keep the first entry of the day based on earliest unix)
    df = df.drop_duplicates(subset="date", keep="first")

    # Count number of unique days in the original data (before filling)
    original_days_count = df["date"].nunique()

    # Create a date range from the min to max date
    full_date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())

    # Set 'date' as the new index
    df.set_index("date", inplace=True)

    # Reindex the DataFrame to fill in missing days with 0 for returns
    df = df.reindex(full_date_range, fill_value=0)

    # Reset index to turn date index back into a column
    df.reset_index(inplace=True)

    # Rename the 'index' column back to 'date'
    df.rename(columns={"index": "date"}, inplace=True)

    # Convert the 'date' column back to UNIX timestamp (set time to midnight for the filled dates)
    df["unix"] = pd.to_datetime(df["date"]).astype(int) // 10**9
    df.to_csv("portfolio_debug.csv")
    # Extract portfolio returns as a 1D array
    portfolio_returns = df["portfolio_returns"].to_numpy()

    # Extract benchmark returns as a 2D array (each sub-array is a benchmark's returns)
    benchmark_returns = [df[benchmark].to_numpy() for benchmark in benchmark_names]
    benchmark_returns = np.array(benchmark_returns)

    # Calculate the total number of days after filling
    total_days_count = len(df)

    # Calculate how many days had to be filled
    filled_days_count = total_days_count - original_days_count

    return portfolio_returns, benchmark_returns, filled_days_count, total_days_count


def run_financial_analysis(
    portfolio_daily_returns,
    startDateString,
    endDateString,
    bechmark_names,
    benchmarks_returns,
):
    if isinstance(portfolio_daily_returns, list):
        portfolio_daily_returns = np.array(portfolio_daily_returns)
    benchmark_results = []
    for benchmark_returns in benchmarks_returns:
        if isinstance(benchmark_returns, list):
            benchmark_returns = np.array(benchmark_returns)
        benchmark_results.append(benchmark_returns)

    yearly_daily_rf_values = fau.convert_annual_to_daily_returns(
        filter_dtb3_values("DTB3.csv", startDateString, endDateString)
    )

    results = fac.perform_porfolio_financial_analysis(
        portfolio_daily_returns,
        yearly_daily_rf_values,
        startDateString,
        benchmark_results,
        bechmark_names,
        "3M T-Bill Returns",
    )

    return results


def run_bencharks_and_financial_analysis(
    tokens,
    portfolio_daily_returns,
    startDateString,
    endDateString,
    benchmarks,
    initial_token_weights,
):
    if isinstance(portfolio_daily_returns, list):
        portfolio_daily_returns = np.array(portfolio_daily_returns)
    benchmark_results = []

    yearly_daily_rf_values = fau.convert_annual_to_daily_returns(
        filter_dtb3_values("DTB3.csv", startDateString, endDateString)
    )

    initial_value_log_ratio = jnp.array([np.log(val) for val in initial_token_weights])

    benchmark_fingerprint = {
        "startDateString": startDateString,
        "endDateString": endDateString,
        "tokens": tokens,
        "initial_weights_logits": initial_value_log_ratio,
        "bout_offset": 0,
    }

    if "hodl" in benchmarks:
        hodl_params = {"initial_weights_logits": initial_value_log_ratio}
        hodl_fingerprint = copy.deepcopy(benchmark_fingerprint)
        hodl_fingerprint["rule"] = "hodl"

        hodl_result = do_run_on_historic_data(
            hodl_fingerprint, dict_of_np_to_jnp(hodl_params), do_test_period=False
        )

        hodl_daily_returns = calculate_daily_returns(
            hodl_result["value"], startDateString, "hodl"
        )

        benchmark_results.append(hodl_daily_returns)

    results = fac.perform_porfolio_financial_analysis(
        portfolio_daily_returns,
        yearly_daily_rf_values,
        startDateString,
        benchmark_results,
        ["hodl"],
        "3M T-Bill Returns",
    )

    return results


def retrieve_param_and_mc_financial_analysis_results(
    run_fingerprint,
    params,
    testEndDateString,
    plot_drawdowns,
    plot_returns,
    price_data=None,
    btc_price_data=None,
):

    train_end_date_str = run_fingerprint["endDateString"]
    local_run_fingerprint = copy.deepcopy(run_fingerprint)
    local_run_fingerprint["endDateString"] = testEndDateString

    portfolio_result = do_run_on_historic_data(
        local_run_fingerprint,
        dict_of_np_to_jnp(params),
        price_data=price_data,
        do_test_period=False,
    )

    hodl_params = copy.deepcopy(params)
    hodl_fingerprint = copy.deepcopy(local_run_fingerprint)
    hodl_fingerprint["rule"] = "hodl"
    hodl_result = do_run_on_historic_data(
        hodl_fingerprint,
        dict_of_np_to_jnp(hodl_params),
        price_data=price_data,
        do_test_period=False,
    )

    fach.plot_line_chart_from_results(
        [portfolio_result["value"], hodl_result["value"]],
        ["QuantAMM Momentum", "basket HODL"],
        run_fingerprint["startDateString"],
        "./results",
        "portfolio_result_abs_ " + "_".join(run_fingerprint["tokens"]) + ".png",
        "Portfolio Value",
        "Date",
        "Pool Value",
        train_end_date_str,
    )

    train_results = single_run_results(
        "Training_Run",
        plot_drawdowns,
        plot_returns,
        run_fingerprint,
        train_end_date_str,
        local_run_fingerprint["startDateString"],
        local_run_fingerprint["startDateString"],
        portfolio_result,
        hodl_result,
    )

    full_test_results = single_run_results(
        "Full_Test_Run",
        plot_drawdowns,
        plot_returns,
        run_fingerprint,
        testEndDateString,
        local_run_fingerprint["startDateString"],
        train_end_date_str,
        portfolio_result,
        hodl_result,
    )

    mc_results = retrieve_mc_param_financial_results(
        run_fingerprint, params, testEndDateString
    )

    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(run_fingerprint["startDateString"], date_format)
    end_date = datetime.strptime(train_end_date_str, date_format)
    training_window_length = (end_date - start_date).total_seconds() // 60
    bout_length = training_window_length - run_fingerprint["bout_offset"]

    train_batch_results = retrieve_batch_window_analysis_results(
        "Train_batch",
        False,
        run_fingerprint["startDateString"],
        run_fingerprint["startDateString"],
        train_end_date_str,
        portfolio_result["value"],
        hodl_result["value"],
        4,
        bout_length,
    )

    test_fingerprint = copy.deepcopy(run_fingerprint)

    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(run_fingerprint["startDateString"], date_format)
    end_date = datetime.strptime(run_fingerprint["endDateString"], date_format)
    training_window_length = (end_date - start_date).total_seconds() // 60
    bout_length = training_window_length - run_fingerprint["bout_offset"]
    test_fingerprint["startDateString"] = train_end_date_str
    test_fingerprint["endDateString"] = testEndDateString

    test_batch_results = retrieve_batch_window_analysis_results(
        "Test_batch",
        False,
        run_fingerprint["startDateString"],
        test_fingerprint["startDateString"],
        test_fingerprint["endDateString"],
        portfolio_result["value"],
        hodl_result["value"],
        4,
        bout_length,
    )
    return {
        "train_period": train_results,
        "full_test_period": full_test_results,
        # "mc_train_period": mc_results,
        "batch_train": train_batch_results,
        "batch_test": test_batch_results,
    }


def single_run_results(
    run_name,
    plot_drawdowns,
    plot_train_returns,
    run_fingerprint,
    end_date,
    dataStartDateString,
    single_run_start_date,
    portfolio_result,
    hodl_result,
):

    portfolio_train_results = slice_minutes_array(
        portfolio_result["value"], dataStartDateString, single_run_start_date, end_date
    )

    hodl_train_results = slice_minutes_array(
        hodl_result["value"], dataStartDateString, single_run_start_date, end_date
    )

    if plot_train_returns:
        fach.plot_line_chart_from_results(
            [portfolio_train_results, hodl_train_results],
            ["QuantAMM Momentum", "basket HODL"],
            single_run_start_date,
            "./results",
            run_name + ".png",
            "Portfolio Value",
            "Date",
            "Pool Value",
        )

    normal_results = retrieve_param_financial_analysis_results(
        run_name,
        plot_drawdowns,
        single_run_start_date,
        end_date,
        portfolio_train_results,
        hodl_train_results,
    )

    return normal_results


def fill_missing_values(target_directory, filename, output_filename):
    # Load the CSV file
    file_path = os.path.join(target_directory, filename)
    df = pd.read_csv(file_path)

    # Convert DATE to datetime format
    df["DATE"] = pd.to_datetime(df["DATE"])

    # Set DATE as index
    df.set_index("DATE", inplace=True)

    # Reindex to fill in missing dates
    all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(all_dates)

    # Forward fill to replace NaNs
    df["DTB3"].replace(".", method="ffill", inplace=True)

    # Forward fill again to handle any NaNs created by reindexing
    df["DTB3"].fillna(method="ffill", inplace=True)

    # Reset index to bring DATE back as a column
    df.reset_index(inplace=True)

    # Rename columns back to original names
    df.rename(columns={"index": "DATE"}, inplace=True)

    # Save the cleaned dataframe back to a CSV
    output_file_path = os.path.join(target_directory, f"{output_filename}")
    df.to_csv(output_file_path, index=False)

    return output_file_path


def calculate_daily_returns(minute_values, startDateString, name):
    # Create a pandas Series with minute-level values and a datetime index
    num_minutes = len(minute_values)
    minute_index = pd.date_range(start=startDateString, periods=num_minutes, freq="T")
    minute_series = pd.Series(minute_values, index=minute_index)
    # Resample to daily frequency by taking the last value of each day

    daily_series = minute_series.resample("D").first()
    # Calculate daily returns
    daily_returns = daily_series.pct_change().dropna()

    daily_return_array = daily_returns.to_numpy()

    # Convert daily returns to numpy array and return
    daily_return_array = np.insert(daily_return_array, 0, 0)

    return daily_return_array


def calculate_daily_old_returns(minute_values):

    daily_values = minute_values[::1440]

    return np.array(jnp.diff(daily_values) / daily_values[:-1])


def calculate_daily_old_returns(minute_values):
    daily_values = minute_values[::1440]
    return np.array(jnp.diff(daily_values) / daily_values[:-1])


def retrieve_param_financial_analysis_results(
    run_name,
    plot_drawdowns,
    run_fingerprint_start_date,
    run_fingerprint_end_date,
    portfolio_result,
    hodl_result,
):
    # Calculate returns for portfolio
    portfolio_daily_returns = calculate_daily_returns(
        portfolio_result, run_fingerprint_start_date, "portfolio"
    )

    # Calculate returns for hodl
    hodl_daily_returns = calculate_daily_returns(
        hodl_result, run_fingerprint_start_date, "hodl"
    )

    yearly_daily_rf_values = fau.convert_annual_to_daily_returns(
        filter_dtb3_values(
            "DTB3.csv", run_fingerprint_start_date, run_fingerprint_end_date
        )
    )

    # Perform financial analysis
    analysis_result = fac.perform_financial_analysis(
        portfolio_daily_returns,
        hodl_daily_returns,
        yearly_daily_rf_values,
    )

    if plot_drawdowns:
        plot_drawdown_charts(analysis_result, run_name)

    # Return the analysis result
    return analysis_result


def plot_drawdown_charts(analysis_result, run_name):
    targetDrawdown = "Avg Daily Drawdown per week"
    target_drawdown_value = "avg_drawdown"
    plot_drawdown_chart(
        "Avg Daily Drawdown per week",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Daily Maximum Drawdown per week"
    target_drawdown_value = "max_drawdown"
    plot_drawdown_chart(
        "Daily Maximum Drawdown per week",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Avg Daily Drawdown per month"
    target_drawdown_value = "avg_drawdown"
    plot_drawdown_chart(
        "Avg Daily Drawdown per month",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Daily Maximum Drawdown per month"
    target_drawdown_value = "max_drawdown"
    plot_drawdown_chart(
        "Daily Maximum Drawdown per month",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Weekly CDaR"
    target_drawdown_value = "avg_cDaR"
    plot_drawdown_chart(
        "Weekly Conditional Value at Risk",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Monthly CDaR"
    plot_drawdown_chart(
        "Monthly Conditional Value at Risk",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Weekly Ulcer Index"
    target_drawdown_value = "ulcer_index"
    plot_drawdown_chart(
        "Weekly Ulcer Index",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Monthly Ulcer Index"
    plot_drawdown_chart(
        "Monthly Ulcer Index",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Weekly Sterling Ratio"
    target_drawdown_value = "sterling_ratio"
    plot_drawdown_chart(
        "Weekly Sterling Ratio",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )

    targetDrawdown = "Monthly Sterling Ratio"
    plot_drawdown_chart(
        "Monthly Sterling Ratio",
        target_drawdown_value,
        "2019-01-01",
        analysis_result["portfolio"]["drawdown"][targetDrawdown],
        analysis_result["hodl"]["drawdown"][targetDrawdown],
        "./results",
        run_name + "_" + targetDrawdown + ".png",
    )


def retrieve_mc_param_financial_results(run_fingerprint, params, testEndDateString):
    results = []
    price_data_cache = {}
    for token in run_fingerprint["tokens"]:
        newTokens = []
        if token == "DAI":
            newTokens.append(token)
            price_data_cache.append(get_data_dict(token, "DAI", "1h"))
        else:
            for i in range(0, 20):
                newTokens.append(token + str(i))
                price_data_cache.append(get_data_dict(token + str(i), "DAI", "1h"))
        results.append(newTokens)

    mc_variations = generate_interarray_permutations(*results)

    mc_results = []

    mc_data = get_historic_csv_data(results,["close"],None,run_fingerprint["startDateString"],testEndDateString)

    for variation in mc_variations:
        run_fingerprint["tokens"] = variation
        variation_plus_unix = ["unix"].append(variation)
        variation_dict = mc_data[variation_plus_unix]
        train_end_date_str = run_fingerprint["endDateString"]
        local_run_fingerprint = copy.deepcopy(run_fingerprint)
        local_run_fingerprint["endDateString"] = testEndDateString

        portfolio_result = do_run_on_historic_data(
            local_run_fingerprint,
            dict_of_np_to_jnp(params),
            price_data=variation_dict,
            do_test_period=False,
        )

        minute_index = pd.date_range(
            start=run_fingerprint["startDateString"],
            periods=len(portfolio_result["value"]),
            freq="T",
        )
        minute_series = pd.Series(portfolio_result["value"], index=minute_index)
        minute_series.to_csv("./results/portfolio_result_abs.csv")

        hodl_params = copy.deepcopy(params)
        hodl_fingerprint = copy.deepcopy(local_run_fingerprint)
        hodl_fingerprint["rule"] = "hodl"
        hodl_result = do_run_on_historic_data(
            hodl_fingerprint,
            dict_of_np_to_jnp(hodl_params),
            price_data=variation_dict,
            do_test_period=False,
        )

        mc_results.append(
            single_run_results(
                False,
                False,
                "MC_Run_" + str(variation),
                run_fingerprint,
                train_end_date_str,
                local_run_fingerprint["startDateString"],
                local_run_fingerprint["startDateString"],
                portfolio_result,
                hodl_result,
            )
        )

    flattened_results = recurse_structure(mc_results[0], [], mc_results)

    return flattened_results


def retrieve_batch_window_analysis_results(
    run_name,
    plot_drawdowns,
    original_start_date,
    run_fingerprint_start_date,
    run_fingerprint_end_date,
    portfolio_result,
    hodl_result,
    batch_count,
    bout_length,
):
    date_pairs = generate_date_pairs(
        run_fingerprint_start_date, run_fingerprint_end_date, batch_count, bout_length
    )

    batch_results = []

    for batch_param in date_pairs:
        sliced_portfolio_result = slice_minutes_array(
            portfolio_result,
            original_start_date,
            batch_param["startDateString"],
            batch_param["endDateString"],
        )
        sliced_hodl_result = slice_minutes_array(
            hodl_result,
            original_start_date,
            batch_param["startDateString"],
            batch_param["endDateString"],
        )

        batch_results.append(
            retrieve_param_financial_analysis_results(
                run_name,
                plot_drawdowns,
                batch_param["startDateString"],
                batch_param["endDateString"],
                sliced_portfolio_result,
                sliced_hodl_result,
            )
        )

    flattened_results = recurse_structure(batch_results[0], [], batch_results)
    flattened_results["startDateString"] = run_fingerprint_start_date
    flattened_results["endDateString"] = run_fingerprint_end_date

    return flattened_results


def extract_values(data, keys):
    values = []
    for entry in data:
        value = entry
        for key in keys:
            value = value[key]
        values.append(value)
    return np.array(values)


def recurse_structure(structure, keys, data):
    if isinstance(structure, dict):
        result = {}
        for key, value in structure.items():
            result[key] = recurse_structure(value, keys + [key], data)
        return result
    elif isinstance(structure, (float, int)):
        values = extract_values(data, keys)
        return faf.calculate_distribution_statistics(values)
    return structure  # for lists or other types, return as is


def generate_interarray_combinations(*arrays):
    """
    Generate all inter-array permutations of elements from multiple input arrays of strings.

    Parameters:
    *arrays (list): Variable number of arrays of strings.

    Returns:
    list: List of tuples containing inter-array permutations.
    """
    interarray_combinations = list(product(*arrays))
    return interarray_combinations


def generate_date_pairs(start_date_str, end_date_str, batch_count, bout_length):
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    data_length = (end_date - start_date).total_seconds() // 60
    date_pairs = []
    max_offset = data_length - bout_length

    if max_offset <= 0:
        date_pairs.append(
            {"startDateString": start_date_str, "endDateString": end_date_str}
        )
        return date_pairs

    while len(date_pairs) < batch_count:
        random_offset = random.randint(0, max_offset)

        random_start_date = start_date + timedelta(minutes=random_offset)
        random_end_date = random_start_date + timedelta(minutes=bout_length)

        if random_end_date <= end_date:
            date_pairs.append(
                {
                    "startDateString": random_start_date.strftime(date_format),
                    "endDateString": random_end_date.strftime(date_format),
                }
            )

    return date_pairs


def plot_drawdown_chart(
    title,
    value_column,
    start_date,
    portfolio_drawdown,
    hodl_drawdown,
    target_directory,
    filename,
):
    portfolio_drawdown_series = fau.convert_to_series(portfolio_drawdown, value_column)
    hodl_drawdown_series = fau.convert_to_series(hodl_drawdown, value_column)

    fach.plot_line_chart_from_series(
        [portfolio_drawdown_series, hodl_drawdown_series],
        ["Portfolio", "basket HODL"],
        start_date,
        target_directory,
        filename,
        title,
        "Date",
        "Drawdown",
    )
