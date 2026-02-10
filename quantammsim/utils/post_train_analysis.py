"""Performance metric calculations for SGD analysis."""

import numpy as np
import jax.numpy as jnp
from quantammsim.core_simulator.forward_pass import _calculate_return_value


def calculate_period_metrics(results_dict, prices=None):
    """Calculate performance metrics for a given period.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing reserves and value data
    prices : array-like, optional
        Price data. If not provided, will look for prices in results_dict
    """
    # Use provided prices if available, otherwise get from results_dict
    price_data = prices if prices is not None else results_dict["prices"]

    sharpe = _calculate_return_value(
        "sharpe",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
    )
    returns = _calculate_return_value(
        "returns",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
    )
    returns_over_hodl = _calculate_return_value(
        "returns_over_hodl",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    returns_over_uniform_hodl = _calculate_return_value(
        "returns_over_uniform_hodl",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    annualised_returns = _calculate_return_value(
        "annualised_returns",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
    )
    annualised_returns_over_hodl = _calculate_return_value(
        "annualised_returns_over_hodl",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    annualised_returns_over_uniform_hodl = _calculate_return_value(
        "annualised_returns_over_uniform_hodl",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    daily_returns = (
        jnp.diff(results_dict["value"][::24 * 60])
        / results_dict["value"][::24 * 60][:-1]
    )
    daily_sharpe = jnp.sqrt(365) * (daily_returns.mean() / daily_returns.std())

    daily_log_sharpe = _calculate_return_value(
        "daily_log_sharpe",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
    )

    ulcer_index = _calculate_return_value(
        "ulcer",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    calmar_ratio = _calculate_return_value(
        "calmar",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    sterling_ratio = _calculate_return_value(
        "sterling",
        results_dict["reserves"],
        price_data,
        results_dict["value"],
        initial_reserves=results_dict["reserves"][0],
    )
    return {
        "sharpe": float(daily_sharpe),
        "jax_sharpe": float(sharpe),
        "daily_log_sharpe": float(daily_log_sharpe),
        "return": float(returns),
        "returns_over_hodl": float(returns_over_hodl),
        "returns_over_uniform_hodl": float(returns_over_uniform_hodl),
        "annualised_returns": float(annualised_returns),
        "annualised_returns_over_hodl": float(annualised_returns_over_hodl),
        "annualised_returns_over_uniform_hodl": float(annualised_returns_over_uniform_hodl),
        "ulcer": float(ulcer_index),
        "calmar": float(calmar_ratio),
        "sterling": float(sterling_ratio),
    }

def calculate_continuous_test_metrics(continuous_results, train_len, test_len, prices):
    """Calculate metrics for continuous test period.
    
    Parameters
    ----------  
    continuous_results : dict
        Results from continuous simulation
    train_len : int
        Length of training period
    test_len : int
        Length of test period
    prices : array-like
        Price data for continuous period
    """
    # Extract test period portion

    price_data = prices if prices is not None else continuous_results["prices"]
    continuous_test_results = {
        "value": continuous_results["value"][train_len : train_len + test_len],
        "reserves": continuous_results["reserves"][train_len : train_len + test_len],
        "prices": price_data[train_len : train_len + test_len],
    }

    metrics = calculate_period_metrics(continuous_test_results)
    return metrics


def process_continuous_outputs(
    continuous_outputs,
    data_dict,
    n_parameter_sets,
    use_ensemble_mode=False,
):
    """
    Process continuous forward pass outputs into per-parameter-set metrics.

    This function handles both standard mode (outputs batched over param sets)
    and ensemble mode (single output from averaged rule outputs).

    Parameters
    ----------
    continuous_outputs : dict
        Output from continuous forward pass containing:
        - "value": Pool value over time
        - "reserves": Pool reserves over time
        Shape depends on mode:
        - Standard: (n_parameter_sets, time_steps) / (n_parameter_sets, time_steps, n_assets)
        - Ensemble: (time_steps,) / (time_steps, n_assets)
    data_dict : dict
        Data dictionary containing:
        - "start_idx": Start index for the simulation
        - "bout_length": Length of training period
        - "bout_length_test": Length of test period
        - "prices": Price data array
    n_parameter_sets : int
        Number of parameter sets (used for standard mode iteration)
    use_ensemble_mode : bool, optional
        If True, outputs are unbatched (ensemble averaging was used).
        Default is False.

    Returns
    -------
    tuple of (list, list, list)
        (train_metrics_list, test_metrics_list, continuous_test_metrics_list)
        Each list contains one dict per parameter set (or one dict total for ensemble mode).
    """
    train_metrics_list = []
    test_metrics_list = []
    continuous_test_metrics_list = []

    start_idx = data_dict["start_idx"]
    bout_length = data_dict["bout_length"]
    bout_length_test = data_dict["bout_length_test"]
    prices = data_dict["prices"]

    if use_ensemble_mode:
        # Ensemble mode: single output (not batched over param sets)
        param_value = continuous_outputs["value"]
        param_reserves = continuous_outputs["reserves"]

        # Slice train period
        train_dict = {
            "value": param_value[:bout_length],
            "reserves": param_reserves[:bout_length],
        }
        train_prices = prices[start_idx : start_idx + bout_length]

        # Slice test period
        test_dict = {
            "value": param_value[bout_length:],
            "reserves": param_reserves[bout_length:],
        }
        test_prices = prices[
            start_idx + bout_length : start_idx + bout_length + bout_length_test
        ]

        # Continuous dict for test metrics
        param_continuous_dict = {
            "value": param_value,
            "reserves": param_reserves,
        }
        continuous_prices = prices[
            start_idx : start_idx + bout_length + bout_length_test
        ]

        # Calculate metrics
        train_metrics = calculate_period_metrics(train_dict, train_prices)
        test_metrics = calculate_period_metrics(test_dict, test_prices)
        continuous_test_metrics = calculate_continuous_test_metrics(
            param_continuous_dict, bout_length, bout_length_test, continuous_prices
        )

        train_metrics_list.append(train_metrics)
        test_metrics_list.append(test_metrics)
        continuous_test_metrics_list.append(continuous_test_metrics)

    else:
        # Standard mode: outputs batched over param sets
        for param_idx in range(n_parameter_sets):
            # Extract outputs for this parameter set
            param_value = continuous_outputs["value"][param_idx]
            param_reserves = continuous_outputs["reserves"][param_idx]

            # Slice train period
            train_dict = {
                "value": param_value[:bout_length],
                "reserves": param_reserves[:bout_length],
            }
            train_prices = prices[start_idx : start_idx + bout_length]

            # Slice test period
            test_dict = {
                "value": param_value[bout_length:],
                "reserves": param_reserves[bout_length:],
            }
            test_prices = prices[
                start_idx + bout_length : start_idx + bout_length + bout_length_test
            ]

            # Continuous dict for test metrics
            param_continuous_dict = {
                "value": param_value,
                "reserves": param_reserves,
            }
            continuous_prices = prices[
                start_idx : start_idx + bout_length + bout_length_test
            ]

            # Calculate metrics
            train_metrics = calculate_period_metrics(train_dict, train_prices)
            test_metrics = calculate_period_metrics(test_dict, test_prices)
            continuous_test_metrics = calculate_continuous_test_metrics(
                param_continuous_dict, bout_length, bout_length_test, continuous_prices
            )

            train_metrics_list.append(train_metrics)
            test_metrics_list.append(test_metrics)
            continuous_test_metrics_list.append(continuous_test_metrics)

    return train_metrics_list, test_metrics_list, continuous_test_metrics_list
