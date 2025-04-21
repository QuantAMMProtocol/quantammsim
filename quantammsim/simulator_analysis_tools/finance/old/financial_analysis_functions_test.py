import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantammsim.simulator_analysis_tools.finance.financial_analysis_functions import (
    calculate_jensens_alpha,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_tracking_error_and_information_ratio,
    calculate_portfolio_risk_metrics,
    calculate_return_on_VaR,
    calculate_omega_ratio,
    calculate_capture_ratios,
    calculate_distribution_statistics,
)


@pytest.fixture
def sample_data():
    # Generate sample data for testing
    np.random.seed(0)
    portfolio_returns = np.random.normal(0.05, 0.1, 100)
    rf_values = np.random.normal(0.02, 0.01, 100)
    benchmark_returns = np.random.normal(0.06, 0.08, 100)
    dtb3_rates = np.random.normal(0.03, 0.01, 100)

    return portfolio_returns, rf_values, benchmark_returns, dtb3_rates


def test_calculate_jensens_alpha(sample_data):
    portfolio_returns, rf_values, benchmark_returns, _ = sample_data

    alpha = calculate_jensens_alpha(portfolio_returns, rf_values, benchmark_returns)

    assert isinstance(alpha, float)


def test_calculate_sharpe_ratio(sample_data):
    portfolio_returns, rf_values, _, _ = sample_data

    sharpe_ratio, annualized_sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, rf_values)

    assert isinstance(sharpe_ratio, float)
    assert isinstance(annualized_sharpe_ratio, float)


def test_calculate_sortino_ratio(sample_data):
    portfolio_returns, rf_values, _, _ = sample_data

    sortino_ratio = calculate_sortino_ratio(portfolio_returns, rf_values)

    assert isinstance(sortino_ratio, float)


def test_calculate_tracking_error_and_information_ratio(sample_data):
    portfolio_returns, _, benchmark_returns, _ = sample_data

    tracking_error, information_ratio = calculate_tracking_error_and_information_ratio(
        portfolio_returns, benchmark_returns
    )

    assert isinstance(tracking_error, float)
    assert isinstance(information_ratio, float)


def test_calculate_portfolio_risk_metrics(sample_data):
    portfolio_returns, rf_values, benchmark_returns, _ = sample_data

    risk_metrics = calculate_portfolio_risk_metrics(portfolio_returns, rf_values, benchmark_returns)

    assert isinstance(risk_metrics, dict)
    assert "Daily VaR (95%)" in risk_metrics
    assert "Daily Maximum Drawdown" in risk_metrics
    assert "Weekly Maximum Drawdown" in risk_metrics
    assert "Monthly Maximum Drawdown" in risk_metrics
    assert "Volatility" in risk_metrics
    assert "Beta" in risk_metrics


def test_calculate_return_on_VaR(sample_data):
    portfolio_returns, rf_values, _, _ = sample_data

    return_on_VaR = calculate_return_on_VaR(portfolio_returns, rf_values)

    assert isinstance(return_on_VaR, dict)
    assert "VaR" in return_on_VaR
    assert "Return on VaR" in return_on_VaR
    assert "Annualized Return on VaR" in return_on_VaR


def test_calculate_omega_ratio(sample_data):
    portfolio_returns, rf_values, _, _ = sample_data

    omega_ratio, probability_positive, probability_negative = calculate_omega_ratio(portfolio_returns, rf_values)

    assert isinstance(omega_ratio, float)
    assert isinstance(probability_positive, float)
    assert isinstance(probability_negative, float)


def test_calculate_capture_ratios(sample_data):
    portfolio_returns, benchmark_returns, _, _ = sample_data

    capture_ratios = calculate_capture_ratios(portfolio_returns, benchmark_returns)

    assert isinstance(capture_ratios, dict)
    assert "upside_capture_ratio" in capture_ratios
    assert "downside_capture_ratio" in capture_ratios
    assert "total_capture_ratio" in capture_ratios


def test_calculate_distribution_statistics(sample_data):
    array = np.array([1, 2, 3, 4, 5])

    statistics = calculate_distribution_statistics(array)

    assert isinstance(statistics, dict)
    assert "mean" in statistics
    assert "std" in statistics
    assert "kurtosis" in statistics
    assert "skewness" in statistics