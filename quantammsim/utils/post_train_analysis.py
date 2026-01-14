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
    daily_returns = (
        jnp.diff(results_dict["value"][::24 * 60])
        / results_dict["value"][::24 * 60][:-1]
    )
    daily_sharpe = jnp.sqrt(365) * (daily_returns.mean() / daily_returns.std())

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
        "return": float(returns),
        "returns_over_hodl": float(returns_over_hodl),
        "returns_over_uniform_hodl": float(returns_over_uniform_hodl),
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
    return {f"continuous_test_{k}": v for k, v in metrics.items()}
