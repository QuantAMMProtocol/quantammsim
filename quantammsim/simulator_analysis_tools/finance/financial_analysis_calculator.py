from datetime import datetime

# from quantammsim.apis.rest_apis.simulator_dtos.simulation_run_dto import (
#     SimulationResultTimestepDto,
#     SimulationResultTimeseries,
#     SimulationRunMetric,
# )
from quantammsim._simulationRunDto import SimulationResultTimestepDto
from quantammsim.simulatorMocks.simulationRunDto import (
    SimulationResultTimeseries,
    SimulationRunMetric,
)
import quantammsim.simulator_analysis_tools.finance.financial_analysis_functions as faf
import quantammsim.simulator_analysis_tools.finance.financial_analysis_utils as fau
import numpy as np


def convert_simulation_timeseries_to_run_metric(
    timeseries, startDateString, timeSeriesName
):
    """
    Convert a simulation timeseries to a run metric format.

    Parameters
    ----------
    timeseries : array-like or numpy.ndarray
        The timeseries data to convert
    startDateString : str
        The start date string in format 'YYYY-MM-DD HH:MM:SS'
    timeSeriesName : str
        Name of the timeseries, used to determine time intervals

    Returns
    -------
    list
        List of SimulationResultTimestepDto objects containing the converted timeseries data

    Notes
    -----
    This function converts a timeseries array into a list of timestep DTOs with proper Unix
    timestamps based on the timeSeriesName. It handles different time intervals:
    - Daily (86400 seconds)
    - Weekly (604800 seconds)
    - Monthly (2628000 seconds)
    - Yearly (31536000 seconds)

    The function skips any NaN or infinite values in the timeseries.
    """
    drawdownTimeSeries = []
    if isinstance(timeseries, np.ndarray):
        dt_object = datetime.strptime(startDateString, "%Y-%m-%d %H:%M:%S")
        currentUnix = int(dt_object.timestamp()) + 1000
        for i in range(len(timeseries)):
            if (
                "daily" in timeSeriesName.lower()
                and not "per week" in timeSeriesName.lower()
                and not "per month" in timeSeriesName.lower()
                and not "per year" in timeSeriesName.lower()
            ):
                timeseries[i]["date"] = currentUnix + (i * 86400)
            elif "week" in timeSeriesName.lower():
                timeseries[i]["date"] = currentUnix + (i * 604800)
            elif "month" in timeSeriesName.lower():
                timeseries[i]["date"] = currentUnix + (i * 2628000)
            elif "year" in timeSeriesName.lower():
                timeseries[i]["date"] = currentUnix + (i * 31536000)
            else:
                dt_object = datetime.strptime(
                    timeseries[i]["date"], "%Y-%m-%d %H:%M:%S"
                )
                timeseries[i]["date"] = int(dt_object.timestamp())

            timeseries_keys = list(timeseries[i].keys())
            isDateKey = timeseries_keys[0] == "date"
            if not isDateKey:
                timestepVal = timeseries[i][timeseries_keys[0]]
            else:
                timestepVal = timeseries[i][timeseries_keys[1]]
            # Convert the datetime object to a Unix timestamp
            timestepVal_is_nan_or_inf = np.isnan(timestepVal) or np.isinf(timestepVal)
            if not timestepVal_is_nan_or_inf:
                timestep = SimulationResultTimestepDto(
                    timeseries[i]["date"], None, float(timestepVal)
                )
                drawdownTimeSeries.append(timestep)
    return drawdownTimeSeries


def convert_return_analysis_to_run_metric(return_analysis, rf_name, startDateString):
    """
    Converts return analysis results into run metrics format.

    Args:
        return_analysis (dict): Dictionary containing return analysis results including metrics like Sharpe ratio,
            drawdown statistics, etc.
        rf_name (str): Name of the risk-free rate used in calculations
        startDateString (str): Start date string in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        tuple: Contains two lists:
            - List of SimulationRunMetric objects for scalar metrics
            - List of SimulationResultTimeseries objects for time series metrics

    The function:
    1. Extracts metrics from the return analysis dictionary
    2. Converts scalar metrics into SimulationRunMetric objects
    3. Converts time series metrics into SimulationResultTimeseries objects
    4. Handles nested dictionaries for risk metrics and drawdown statistics
    """
    run_metrics = []
    run_timestep_metrics = []
    for key in return_analysis:
        if key == "Benchmark Name":
            continue
        if key == "risk_metrics":
            for risk_metric in return_analysis[key]:
                if isinstance(return_analysis[key][risk_metric], np.ndarray):
                    timeseries = convert_simulation_timeseries_to_run_metric(
                        return_analysis[key][risk_metric], startDateString, risk_metric
                    )
                    run_timestep_metrics.append(
                        SimulationResultTimeseries(
                            timeseries, rf_name, risk_metric, "Daily"
                        )
                    )
                else:
                    run_metrics.append(
                        SimulationRunMetric(
                            rf_name,
                            risk_metric,
                            return_analysis["risk_metrics"][risk_metric],
                            "",
                            "Daily",
                        )
                    )
        elif key == "Drawdown Statistics" or key == "Statistical Properties":
            for drawdown_metric in return_analysis[key]:
                if isinstance(return_analysis[key][drawdown_metric], np.ndarray):
                    timeseries = convert_simulation_timeseries_to_run_metric(
                        return_analysis[key][drawdown_metric],
                        startDateString,
                        drawdown_metric,
                    )
                    run_timestep_metrics.append(
                        SimulationResultTimeseries(
                            timeseries, rf_name, drawdown_metric, "Daily"
                        )
                    )
                else:
                    run_metrics.append(
                        SimulationRunMetric(
                            rf_name,
                            drawdown_metric,
                            return_analysis[key][drawdown_metric],
                            "",
                            "Daily",
                        )
                    )
        else:
            run_metrics.append(
                SimulationRunMetric(rf_name, key, return_analysis[key], "", "Daily")
            )
    return run_metrics, run_timestep_metrics


def convert_benchmark_analysis_to_run_metric(
    benchmark_analysis, rf_name, benchmark_name, startDateString
):
    """
    Converts benchmark analysis results into SimulationRunMetric and SimulationResultTimeseries objects.

    Args:
        benchmark_analysis (list): List of dictionaries containing benchmark analysis results
        rf_name (str): Name of the risk-free rate used
        benchmark_name (str): Name of the benchmark being analyzed
        startDateString (str): Start date string in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        tuple: A tuple containing:
            - list[SimulationRunMetric]: List of scalar metric results
            - list[SimulationResultTimeseries]: List of timeseries metric results

    The function processes benchmark analysis results and converts them into standardized metric objects:
    - Scalar metrics are converted to SimulationRunMetric objects
    - Timeseries metrics are converted to SimulationResultTimeseries objects
    - Handles risk metrics, capture ratios and other benchmark comparison metrics
    """
    run_metrics = []
    run_timeline_metrics = []
    for element in benchmark_analysis:
        benchmark_name = element["Benchmark Name"]
        for key in element:
            if key == "Benchmark Name":
                continue
            if key == "risk_metrics":
                for risk_metric in element[key]:
                    if isinstance(element[key][risk_metric], np.ndarray):
                        timeseries = convert_simulation_timeseries_to_run_metric(
                            element[key][risk_metric],
                            startDateString,
                            risk_metric,
                            risk_metric,
                        )
                        run_timeline_metrics.append(
                            SimulationResultTimeseries(
                                timeseries, rf_name, risk_metric, "Daily"
                            )
                        )
                    else:
                        run_metrics.append(
                            SimulationRunMetric(
                                rf_name,
                                risk_metric,
                                element["risk_metrics"][risk_metric],
                                benchmark_name,
                                "Daily",
                            )
                        )
            elif key == "capture_ratios":
                for capture_ratio in element[key]:
                    if isinstance(element[key][capture_ratio], np.ndarray):
                        timeseries = convert_simulation_timeseries_to_run_metric(
                            element[key][capture_ratio], startDateString, capture_ratio
                        )
                        run_timeline_metrics.append(
                            SimulationResultTimeseries(
                                timeseries, rf_name, capture_ratio, "Daily"
                            )
                        )
                    else:
                        run_metrics.append(
                            SimulationRunMetric(
                                rf_name,
                                capture_ratio,
                                element["capture_ratios"][capture_ratio],
                                benchmark_name,
                                "Daily",
                            )
                        )
            elif key != "benchmark_name":
                if isinstance(element[key], np.ndarray):
                    timeseries = convert_simulation_timeseries_to_run_metric(
                        element[key], rf_name
                    )
                    run_timeline_metrics.append(
                        SimulationResultTimeseries(timeseries, rf_name, key, "Daily")
                    )
                else:
                    run_metrics.append(
                        SimulationRunMetric(
                            rf_name, key, element[key], benchmark_name, "Daily"
                        )
                    )

    return run_metrics, run_timeline_metrics


def perform_return_analysis(returns, dailyRfValues):
    """
    Performs comprehensive return analysis on a set of returns data.

    Args:
        returns (numpy.ndarray): Array of return values to analyze
        dailyRfValues (numpy.ndarray): Array of daily risk-free rates

    Returns:
        dict: Dictionary containing various return metrics including:
            - Absolute Return (%)
            - Sharpe Ratio
            - Annualized Sharpe Ratio
            - Annualized Sortino Ratio
            - Annualized Calmar Ratio
            - Annualized Omega Ratio
            - Return on VaR
            - Annualized Return on VaR
            - Annualized Information Ratio
            - Tracking Error
            - Drawdown Statistics
            - Statistical Properties
    """
    # Calculate the Sharpe ratio
    sharpe = faf.calculate_sharpe_ratio(returns, dailyRfValues)

    # Calculate the Sortino ratio
    sortino = faf.calculate_sortino_ratio(returns, dailyRfValues)

    # Calculate the Calmar ratio
    calmar = faf.calculate_calmar_ratio(returns, dailyRfValues)

    # Calculate the Omega ratio
    omega = faf.calculate_omega_ratio(returns, dailyRfValues)

    # Calculate information ratio
    information_ratio = faf.calculate_tracking_error_and_information_ratio(
        returns, dailyRfValues
    )

    # Calculate Return on VaR
    rov = faf.calculate_return_on_VaR(returns, dailyRfValues)

    # Calculate the Drawdown and Max Drawdown
    drawdown = faf.calculate_drawdown_statistics(returns, dailyRfValues)

    # Calculate Distributions
    distribution = faf.calculate_distribution_statistics(returns)

    # Return the results
    cumulative_returns = (1 + returns).cumprod()
    return {
        "Absolute Return (%)": float(cumulative_returns[-1] - 1) * 100,
        "Sharpe Ratio": sharpe["sharpe_ratio"],
        "Annualized Sharpe Ratio": sharpe["annualized_sharpe_ratio"],
        "Annualized Sortino Ratio": sortino.item(),
        "Annualized Calmer Ratio": calmar.item(),
        "Annualized Omega Ratio": omega["Annualized Omega Ratio"].item(),
        "Return on VaR": rov["Return on VaR"].item(),
        "Annualized Return on VaR": rov["Annualized Return on VaR"].item(),
        "Annualized Information Ratio": information_ratio["information_ratio"],
        "Tracking Error": information_ratio["tracking_error"],
        "Drawdown Statistics": drawdown,
        "Statistical Properties": distribution,
    }


def portfolio_benchmark_analysis(
    portfolio_returns, benchmark_returns, dailyRfValues, benchmark
):
    """
    Performs analysis comparing portfolio returns against benchmark returns.

    Args:
        portfolio_returns (numpy.ndarray): Array of portfolio returns
        benchmark_returns (numpy.ndarray): Array of benchmark returns
        dailyRfValues (numpy.ndarray): Array of daily risk-free rates
        benchmark (str): Name of the benchmark being compared against

    Returns:
        dict: Dictionary containing benchmark comparison metrics including:
            - Benchmark Name: Name of benchmark used
            - capture_ratios: Up/down market capture ratios
            - Annualized Jensen's Alpha: Risk-adjusted excess return
            - Annualized Information Ratio: Risk-adjusted relative return
            - Tracking Error: Standard deviation of return differences
            - risk_metrics: Various risk metrics vs benchmark
    """
    # Calculate the Capture Ratios
    capture_ratios = faf.calculate_capture_ratios(portfolio_returns, benchmark_returns)

    # Calculate the Alpha
    alpha = faf.calculate_jensens_alpha(
        portfolio_returns, dailyRfValues, benchmark_returns
    )

    information_ratio = faf.calculate_tracking_error_and_information_ratio(
        portfolio_returns, benchmark_returns
    )

    # Calculate the VaR
    risk_metrics = faf.calculate_portfolio_risk_metrics(
        portfolio_returns, dailyRfValues, benchmark_returns
    )

    # Return the results
    return {
        "Benchmark Name": benchmark,
        "capture_ratios": capture_ratios,
        "Annualized Jensen's Alpha": alpha.item(),
        "Annualized Information Ratio": information_ratio["information_ratio"],
        "Tracking Error": information_ratio["tracking_error"],
        "risk_metrics": risk_metrics,
    }


def perform_porfolio_financial_analysis(
    portfolio_returns,
    dailyRfValues,
    startDateString,
    benchmark_return_array=None,
    benchmark_name=None,
    rf_name=None,
):
    """Performs financial analysis on portfolio returns.

    Calculates various financial metrics and analyses for a portfolio's returns, including
    return analysis, benchmark comparisons, and time series analysis.

    Args:
        portfolio_returns (numpy.ndarray): Array of portfolio returns
        dailyRfValues (numpy.ndarray): Array of daily risk-free rates
        startDateString (str): Start date string in format 'YYYY-MM-DD HH:MM:SS'
        benchmark_return_array (list[numpy.ndarray], optional): List of benchmark return arrays. Defaults to None.
        benchmark_name (list[str], optional): List of benchmark names. Defaults to None.
        rf_name (str, optional): Name of risk-free rate used. Defaults to None.

    Returns:
        dict: Dictionary containing:
            - return_analysis (list): Portfolio return metrics
            - benchmark_analysis (list): Benchmark comparison metrics
            - return_timeseries_analysis (list): Time series analysis metrics
    """
    return_analysis = perform_return_analysis(portfolio_returns, dailyRfValues)
    result_analysis_array = []
    return_analysis_timeseries_array = []
    if benchmark_return_array is not None:
        i = 0
        for benchmark_returns in benchmark_return_array:
            benchmark_analysis = portfolio_benchmark_analysis(
                portfolio_returns, benchmark_returns, dailyRfValues, benchmark_name[i]
            )
            result_analysis_array.append(benchmark_analysis)
            i += 1

    return_analysis_array, return_analysis_timeseries_array = (
        convert_return_analysis_to_run_metric(return_analysis, rf_name, startDateString)
    )

    return_benchmark_analysis_array, return_benchmark_analysis_timeseries_array = (
        convert_benchmark_analysis_to_run_metric(
            result_analysis_array, rf_name, benchmark_name, startDateString
        )
    )

    return_analysis_timeseries_array.extend(return_benchmark_analysis_timeseries_array)

    return {
        "return_analysis": return_analysis_array,
        "benchmark_analysis": return_benchmark_analysis_array,
        "return_timeseries_analysis": return_analysis_timeseries_array,
    }


def perform_financial_analysis(
    portfolio_returns, hodl_returns, dailyRfValues
):
    """
    Performs comprehensive financial analysis comparing portfolio performance against HODL benchmarks.

    Calculates various risk-adjusted return metrics, ratios, and statistics including:
    - Sharpe, Sortino, Calmar, and Omega ratios
    - Information ratio and tracking error vs benchmarks
    - Return on VaR and capture ratios
    - Portfolio risk metrics and VaR
    - Drawdown statistics and distribution analysis
    - Alpha and other benchmark-relative metrics

    Args:
        portfolio_returns (numpy.ndarray): Array of portfolio returns
        hodl_returns (numpy.ndarray): Array of HODL strategy returns
        dailyRfValues (numpy.ndarray): Array of daily risk-free rates

    Returns:
        dict: Dictionary containing analysis results with keys:
            - portfolio: Portfolio metrics and statistics
            - hodl: HODL benchmark metrics
            Each contains nested dicts of ratios, risk metrics, drawdowns etc.
    """

    # Calculate the Sharpe ratio
    porfolio_sharpe = faf.calculate_sharpe_ratio(portfolio_returns, dailyRfValues)
    hodl_sharpe = faf.calculate_sharpe_ratio(hodl_returns, dailyRfValues)

    # Calculate the Sortino ratio
    porfolio_sortino = faf.calculate_sortino_ratio(portfolio_returns, dailyRfValues)
    hodl_sortino = faf.calculate_sortino_ratio(hodl_returns, dailyRfValues)

    # Calculate the Calmar ratio
    porfolio_calmar = faf.calculate_calmar_ratio(portfolio_returns, dailyRfValues)
    hodl_calmar = faf.calculate_calmar_ratio(hodl_returns, dailyRfValues)

    # Calculate the Omega ratio
    porfolio_omega = faf.calculate_omega_ratio(portfolio_returns, dailyRfValues)
    hodl_omega = faf.calculate_omega_ratio(hodl_returns, dailyRfValues)

    # Calculate information ratio
    hodl_information_ratio = faf.calculate_tracking_error_and_information_ratio(
        portfolio_returns, hodl_returns
    )

    # Calculate Return on VaR
    porfolio_rov = faf.calculate_return_on_VaR(portfolio_returns, dailyRfValues)
    hodl_rov = faf.calculate_return_on_VaR(hodl_returns, dailyRfValues)

    # Calculate Capture Ratios
    porfolio_capture_hodl = faf.calculate_capture_ratios(
        portfolio_returns, hodl_returns
    )

    # Calculate the VaR
    risk_metrics_hodl_rb = faf.calculate_portfolio_risk_metrics(
        portfolio_returns, dailyRfValues, hodl_returns
    )

    # calculate drawdown and max drawdown
    # Calculate the Drawdown and Max Drawdown
    porfolio_drawdown = faf.calculate_drawdown_statistics(
        portfolio_returns, dailyRfValues
    )
    hodl_drawdown = faf.calculate_drawdown_statistics(hodl_returns, dailyRfValues)

    # Calculate Distributions
    hodl_distribution = faf.calculate_distribution_statistics(hodl_returns)

    porfolio_distribution = faf.calculate_distribution_statistics(portfolio_returns)

    hodl_alpha = faf.calculate_jensens_alpha(
        portfolio_returns, dailyRfValues, hodl_returns
    )

    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    hodl_cumulative_returns = (1 + hodl_returns).cumprod()

    # Return the results
    return {
        "portfolio": {
            "return": portfolio_cumulative_returns[-1] - 1,
            "sharpe": porfolio_sharpe,
            "sortino": porfolio_sortino,
            "calmar": porfolio_calmar,
            "omega": porfolio_omega,
            "rov": porfolio_rov,
            "capture_hodl_rb": porfolio_capture_hodl,
            "information_ratio_hodl_rb": hodl_information_ratio,
            "risk_metrics_hodl_rb": risk_metrics_hodl_rb,
            "alpha_hodl_rb": hodl_alpha,
            "drawdown": porfolio_drawdown,
            "distribution": porfolio_distribution,
        },
        "hodl": {
            "return": hodl_cumulative_returns[-1] - 1,
            "sharpe": hodl_sharpe,
            "sortino": hodl_sortino,
            "calmar": hodl_calmar,
            "omega": hodl_omega,
            "rov": hodl_rov,
            "distribution": hodl_distribution,
            "drawdown": hodl_drawdown,
        }
    }
