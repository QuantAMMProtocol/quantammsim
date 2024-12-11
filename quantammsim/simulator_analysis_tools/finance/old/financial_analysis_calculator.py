import quantammsim.simulator_analysis_tools.finance.financial_analysis_functions as faf
import quantammsim.simulator_analysis_tools.finance.financial_analysis_utils as fau

def perform_financial_analysis(portfolio_returns, hodl_returns, dailyRfValues):

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
    hodl_information_ratio = faf.calculate_tracking_error_and_information_ratio(portfolio_returns, hodl_returns)

    # Calculate Return on VaR
    porfolio_rov = faf.calculate_return_on_VaR(portfolio_returns, dailyRfValues)
    hodl_rov = faf.calculate_return_on_VaR(hodl_returns, dailyRfValues)

    # Calculate Capture Ratios
    porfolio_capture_hodl = faf.calculate_capture_ratios(portfolio_returns, hodl_returns)

    # Calculate the VaR
    risk_metrics_hodl_rb = faf.calculate_portfolio_risk_metrics(portfolio_returns, dailyRfValues, hodl_returns)

    #calculate drawdown and max drawdown
    # Calculate the Drawdown and Max Drawdown
    porfolio_drawdown = faf.calculate_drawdown_statistics(portfolio_returns, dailyRfValues)
    hodl_drawdown = faf.calculate_drawdown_statistics(hodl_returns, dailyRfValues)

    # Calculate Distributions
    hodl_distribution = faf.calculate_distribution_statistics(hodl_returns)
    porfolio_distribution = faf.calculate_distribution_statistics(portfolio_returns)

    hodl_alpha = faf.calculate_jensens_alpha(portfolio_returns, dailyRfValues, hodl_returns)

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
