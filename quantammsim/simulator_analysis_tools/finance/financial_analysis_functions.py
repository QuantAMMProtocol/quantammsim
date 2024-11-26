import numpy as np
from scipy.stats import kurtosis, skew, linregress
import pandas as pd
import csv

def calculate_jensens_alpha(portfolio_returns, rf_values, benchmark_returns):
    """
    Calculate Jensen's Alpha for a given set of portfolio returns, risk-free rates, and benchmark returns.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    rf_values (np.array): Daily risk-free rates.
    benchmark_returns (np.array): Daily returns of the benchmark (market index).

    Returns:
    float: Jensen's Alpha
    """
    # Calculate excess returns
    excess_portfolio_returns = portfolio_returns - rf_values
    excess_benchmark_returns = benchmark_returns - rf_values

    # Perform linear regression to get the beta (slope)
    # Convert excess returns to numpy arrays if they are not already
    
    # Filter out extreme values if present (optional)
    valid_indices = np.isfinite(excess_benchmark_returns) & np.isfinite(excess_portfolio_returns)
    excess_benchmark_returns = np.where(np.abs(excess_benchmark_returns) == np.inf, np.nan, excess_benchmark_returns)
    excess_portfolio_returns = np.where(np.abs(excess_portfolio_returns) == np.inf, np.nan, excess_portfolio_returns)
    excess_benchmark_returns = excess_benchmark_returns[valid_indices]
    excess_portfolio_returns = excess_portfolio_returns[valid_indices]



    beta, _, _, _, _ = linregress(excess_benchmark_returns.astype(np.float64), excess_portfolio_returns.astype(np.float64))

    # Calculate the average returns
    average_portfolio_return = np.mean(portfolio_returns)
    average_rf_rate = np.mean(rf_values)
    average_benchmark_return = np.mean(benchmark_returns)

    # Calculate Jensen's Alpha
    jensens_alpha = average_portfolio_return - (average_rf_rate + beta * (average_benchmark_return - average_rf_rate))

    annualized_jensens_alpha = jensens_alpha * np.sqrt(365)
    return annualized_jensens_alpha

def calculate_sharpe_ratio(portfolio_returns, rf_values):
    """
    Calculate the Sharpe Ratio and annualized Sharpe Ratio for a given set of portfolio returns and risk-free rates.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    rf_values (np.array): Daily risk-free rates.

    Returns:
    tuple: (Sharpe Ratio, Annualized Sharpe Ratio)
    """

    # Calculate excess returns
    excess_returns = portfolio_returns - rf_values
    # Calculate the mean and standard deviation of excess returns
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)  # ddof=1 provides sample standard deviation

    # Calculate the Sharpe Ratio
    sharpe_ratio = mean_excess_return / std_excess_return

    # Annualize the Sharpe Ratio
    trading_days_per_year = 365 
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(trading_days_per_year)

    return { 'sharpe_ratio': sharpe_ratio.item(), 'annualized_sharpe_ratio': annualized_sharpe_ratio.item() }

def calculate_sortino_ratio(portfolio_returns, rf_values):
    """
    Calculate the Sortino Ratio for a given set of portfolio returns and risk-free rates.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    rf_values (np.array): Daily risk-free rates.

    Returns:
    float: Sortino Ratio
    """
    # Calculate excess returns
    excess_returns = portfolio_returns - rf_values

    # Calculate the mean excess return
    mean_excess_return = np.mean(excess_returns)

    # Calculate the downside deviation (standard deviation of negative excess returns)
    negative_excess_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.std(negative_excess_returns, ddof=1)  # ddof=1 provides sample standard deviation

    # Calculate the Sortino Ratio
    sortino_ratio = mean_excess_return / downside_deviation

    return sortino_ratio * np.sqrt(365)


def calculate_tracking_error(portfolio_returns, benchmark_returns):
    """
    Calculate the Tracking Error for a portfolio compared to a benchmark.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    benchmark_returns (np.array): Daily returns of the benchmark.

    Returns:
    float: Tracking Error
    """
    # Calculate the differences between portfolio and benchmark returns
    tracking_error = np.sqrt(np.mean((portfolio_returns - benchmark_returns) ** 2))
    return tracking_error

def calculate_tracking_error_and_information_ratio(portfolio_returns, benchmark_returns):
    """
    Calculate the Tracking Error and Information Ratio for a given set of portfolio returns and benchmark returns.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    benchmark_returns (np.array): Daily returns of the benchmark (market index).

    Returns:
    tuple: (Tracking Error, Information Ratio)
    """
    # Calculate the excess returns
    excess_returns = portfolio_returns - benchmark_returns

    # Calculate the mean of the excess returns
    mean_excess_return = np.mean(excess_returns)

    # Calculate the Tracking Error (standard deviation of the excess returns)
    tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns)
    
    # Calculate the Information Ratio
    information_ratio = mean_excess_return / tracking_error

    annualized_information_ratio = information_ratio * np.sqrt(365)

    return { 'tracking_error': tracking_error.item(), 'information_ratio': annualized_information_ratio.item() }


def calculate_portfolio_risk_metrics(portfolio_returns, rf_values, benchmark_returns):
    """
    Calculate various portfolio risk metrics including daily VaR, daily/weekly/monthly maximum drawdowns,
    volatility, and Beta.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    rf_values (np.array): Daily risk-free rates.
    benchmark_returns (np.array): Daily returns of the benchmark (market index).

    Returns:
    dict: A dictionary containing the risk metrics.
    """
    # Convert to pandas Series for easier handling of date ranges
    dates = pd.date_range(start='2021-02-03', periods=len(portfolio_returns), freq='D')
    portfolio_returns = pd.Series(portfolio_returns, index=dates)
    rf_values = pd.Series(rf_values, index=dates)
    benchmark_returns = pd.Series(benchmark_returns, index=dates)

    # Daily Historic Value at Risk (VaR) at 95% confidence level
    confidence_level = 0.95
    VaR_95_array = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    if isinstance(VaR_95_array, np.ndarray):
        VaR_95 = VaR_95_array[0]
    else:
        VaR_95 = VaR_95_array
    # Volatility (standard deviation of daily returns)
    volatility = portfolio_returns.std()

    # Calculate Beta (sensitivity of portfolio returns to market returns)
    beta, alpha, r_value, p_value, std_err = linregress(benchmark_returns, portfolio_returns)
    
    return {
        'Daily VaR (95)': VaR_95.item(),
        'Volatility': volatility,
        'Beta': beta.item()
    }

def calculate_drawdown_statistics(daily_returns, rf_values):
    """
    Calculate daily, weekly, and monthly maximum drawdowns for a given set of daily portfolio returns.

    Parameters:
    daily_returns (np.array or pd.Series): Daily returns of the portfolio.

    Returns:
    dict: A dictionary containing the daily, weekly, and monthly maximum drawdowns.
    """
    if isinstance(daily_returns, np.ndarray) or isinstance(daily_returns, list):
        # Convert to pandas Series for easier handling of date ranges
        dates = pd.date_range(start='2021-02-03', periods=len(daily_returns), freq='D')
        daily_returns = pd.Series(daily_returns, index=dates)

    # Calculate daily maximum drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    daily_max_drawdown = drawdown.min()

    # Weekly maximum drawdown
    weekly_returns = daily_returns.resample('W').apply(lambda x: (1 + x).prod() - 1)
    cumulative_weekly_returns = (1 + weekly_returns).cumprod()
    peak_weekly = cumulative_weekly_returns.cummax()
    drawdown_weekly = (cumulative_weekly_returns - peak_weekly) / peak_weekly
    weekly_max_drawdown = drawdown_weekly.min()

    # Monthly maximum drawdown
    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    cumulative_monthly_returns = (1 + monthly_returns).cumprod()
    peak_monthly = cumulative_monthly_returns.cummax()
    drawdown_monthly = (cumulative_monthly_returns - peak_monthly) / peak_monthly
    monthly_max_drawdown = drawdown_monthly.min()

    daily_weekly_avg = calculate_average_daily_drawdown(daily_returns, "W")
    daily_monthly_avg = calculate_average_daily_drawdown(daily_returns, "M")

    daily_weekly_max = calculate_max_daily_drawdown(daily_returns, "W")
    daily_monthly_max = calculate_max_daily_drawdown(daily_returns, "M")

    ulcer_index = calculate_ulcer_index(daily_returns)

    daily_weekly_ulcer = calcuate_period_ulcer_index(daily_returns, "W")
    daily_monthly_ulcer = calcuate_period_ulcer_index(daily_returns, "M")

    sterling = calculate_sterling_ratio(daily_returns, rf_values)

    daily_weekly_sterling = calcuate_period_sterling_index(daily_returns, rf_values, "W")
    daily_monthly_sterling = calcuate_period_sterling_index(daily_returns, rf_values, "M")

    annualized_cDaR = calculate_cdar(daily_returns) * np.sqrt(365)

    weekly_cDaR = calculate_monthly_cdar(daily_returns, "W")
    monthly_cDaR = calculate_monthly_cdar(daily_returns, "M")

    return {
        'Daily Returns Maximum Drawdown': abs(daily_max_drawdown),
        'Weekly Returns Maximum Drawdown': abs(weekly_max_drawdown),
        'Monthly Returns Maximum Drawdown': abs(monthly_max_drawdown),
        'Avg Daily Drawdown per week': daily_weekly_avg,
        'Avg Daily Drawdown per month': daily_monthly_avg,
        'Daily Maximum Drawdown per week': daily_weekly_max,
        'Daily Maximum Drawdown per month': daily_monthly_max,
        'Conditional Drawdown at Risk': annualized_cDaR.item(),
        'Weekly CDaR': weekly_cDaR,
        'Monthly CDaR': monthly_cDaR,
        'Weekly Ulcer Index': daily_weekly_ulcer,
        'Monthly Ulcer Index': daily_monthly_ulcer,
        'Ulcer Index': ulcer_index.item(),
        'Weekly Sterling Ratio': daily_weekly_sterling,
        'Monthly Sterling Ratio': daily_monthly_sterling,
        'Sterling Ratio': sterling.item()
    }


def calculate_max_daily_drawdown(daily_returns, period):
    """
    Calculate the Conditional Drawdown at Risk (CDaR) per month for daily portfolio returns.

    Parameters:
    portfolio_returns (np.array or pd.Series): Daily returns of the portfolio.
    confidence_level (float): The confidence level for CDaR, default is 0.95.

    Returns:
    pd.Series: Monthly CDaR values.
    """

    portfolio_returns = daily_returns

    if not isinstance(daily_returns, pd.Series):
        portfolio_returns = pd.Series(daily_returns, index=pd.date_range(start='2021-01-01', periods=len(portfolio_returns), freq='D'))

    # Resample to get the start and end of each month
    periods = portfolio_returns.resample(period).apply(lambda x: x.index)

    period_drawdowns = []
    for period_item, dates in periods.items():
        
        if isinstance(dates, pd.Timestamp) or len(dates) < 2:
            break
        # Get the daily returns for the period
        period_portfolio_returns = portfolio_returns[dates[0]:dates[-1]]

        # Calculate cumulative returns
        cumulative_returns = (1 + period_portfolio_returns).cumprod()

        # Calculate the rolling maximum
        rolling_max = cumulative_returns.cummax()

        # Calculate the drawdowns
        drawdowns = (cumulative_returns - rolling_max) / rolling_max

        max_daily_drawdown = abs(drawdowns.min())
        period_drawdowns.append({"date": period_item.isoformat(), "max_drawdown": max_daily_drawdown})

    return np.array(period_drawdowns)

def calcuate_period_ulcer_index(daily_returns, period):

    portfolio_returns = daily_returns

    if not isinstance(daily_returns, pd.Series):
        portfolio_returns = pd.Series(daily_returns, index=pd.date_range(start='2021-01-01', periods=len(portfolio_returns), freq='D'))

    # Resample to get the start and end of each month
    periods = portfolio_returns.resample(period).apply(lambda x: x.index)

    period_drawdowns = []
    for period_item, dates in periods.items():
        
        if isinstance(dates, pd.Timestamp) or len(dates) < 2:
            break
        # Get the daily returns for the period
        period_portfolio_returns = portfolio_returns[dates[0]:dates[-1]]

        index = calculate_ulcer_index(period_portfolio_returns)
        period_drawdowns.append({"date": period_item.isoformat(), "ulcer_index": index})

    return np.array(period_drawdowns)

def calculate_average_daily_drawdown(daily_returns, period):
    """
    Calculate the Conditional Drawdown at Risk (CDaR) per month for daily portfolio returns.

    Parameters:
    portfolio_returns (np.array or pd.Series): Daily returns of the portfolio.
    confidence_level (float): The confidence level for CDaR, default is 0.95.

    Returns:
    pd.Series: Monthly CDaR values.
    """
    portfolio_returns = daily_returns

    if not isinstance(daily_returns, pd.Series):
        portfolio_returns = pd.Series(daily_returns, index=pd.date_range(start='2021-01-01', periods=len(portfolio_returns), freq='D'))
    # Resample to get the start and end of each month
    periods = portfolio_returns.resample(period).apply(lambda x: x.index)

    period_drawdowns = []
    for period_item, dates in periods.items():

        # Get the daily returns for the period
        if isinstance(dates, pd.Timestamp) or len(dates) < 2:
            break

        period_portfolio_returns = portfolio_returns[dates[0]:dates[-1]]

        # Calculate cumulative returns
        cumulative_returns = (1 + period_portfolio_returns).cumprod()

        # Calculate the rolling maximum
        rolling_max = cumulative_returns.cummax()

        # Calculate the drawdowns
        drawdowns = (cumulative_returns - rolling_max) / rolling_max

        # Calculate the average daily drawdown for the period
        avg_daily_drawdown = drawdowns.mean()
        
        period_drawdowns.append({"date": period_item.isoformat(), "avg_drawdown": avg_daily_drawdown})

    return np.array(period_drawdowns)

def calculate_ulcer_index(daily_returns):
    """
    Calculate the Ulcer Index for a given set of daily returns.

    Parameters:
    daily_returns (np.array or pd.Series): Daily returns of the portfolio.

    Returns:
    float: Ulcer Index
    """
    # Ensure daily_returns is a pandas Series for easier handling
    if isinstance(daily_returns, np.ndarray):
        daily_returns = pd.Series(daily_returns)

    # Calculate the cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()

    # Calculate the rolling maximum
    rolling_max = cumulative_returns.cummax()
    
    # Calculate the drawdowns
    drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
    
    # Square the drawdowns
    drawdown_squared = drawdowns ** 2
    
    # Calculate the mean of the squared drawdowns
    mean_drawdown_squared = drawdown_squared.mean()
    
    # Take the square root to get the Ulcer Index
    ulcer_index = np.sqrt(mean_drawdown_squared)
    
    return ulcer_index


def calculate_sterling_ratio(returns, rf):
    """
    Calculate the Sterling ratio given daily returns and daily risk-free rate.

    Parameters:
    returns : numpy array or list
        Daily returns of the investment or portfolio.
    rf : numpy array or list
        Daily risk-free rate.

    Returns:
    float
        Sterling ratio of the investment or portfolio.
    """
    excess_returns = returns - rf
    downside_deviation = np.std(np.minimum(excess_returns, 0))
    
    if downside_deviation == 0:
        return np.inf  # Handle case where downside deviation is zero
    
    mean_excess_return = np.mean(excess_returns)
    
    sterling_ratio = mean_excess_return / downside_deviation
    
    return sterling_ratio


def calcuate_period_sterling_index(daily_returns, rf_values, period):

    portfolio_returns = daily_returns

    if not isinstance(daily_returns, pd.Series):
        portfolio_returns = pd.Series(daily_returns, index=pd.date_range(start='2021-01-01', periods=len(portfolio_returns), freq='D'))

    if not isinstance(rf_values, pd.Series):
        rf_values = pd.Series(rf_values, index=pd.date_range(start='2021-01-01', periods=len(rf_values), freq='D'))

    # Resample to get the start and end of each month
    periods = portfolio_returns.resample(period).apply(lambda x: x.index)

    period_sterlings = []
    for period_item, dates in periods.items():
        
        if isinstance(dates, pd.Timestamp) or len(dates) < 2:
            break
        # Get the daily returns for the period
        period_portfolio_returns = portfolio_returns[dates[0]:dates[-1]]
        period_rf = rf_values[dates[0]:dates[-1]]

        sterling = calculate_sterling_ratio(period_portfolio_returns, period_rf)
        period_sterlings.append({"date": period_item.isoformat(), "sterling_ratio": sterling})

    return np.array(period_sterlings)


def calculate_return_on_VaR(portfolio_returns, rf_values, confidence_level=0.95):
    """
    Calculate the return on VaR, annualized return on VaR, and VaR for a given set of portfolio returns and risk-free rates.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    rf_values (np.array): Daily risk-free rates.
    confidence_level (float): Confidence level for VaR calculation (default is 0.95).

    Returns:
    dict: A dictionary containing the return on VaR, annualized return on VaR, and VaR.
    """
    # Calculate daily excess returns
    excess_returns = portfolio_returns - rf_values
    
    # Daily VaR at the specified confidence level
    VaR = -np.percentile(excess_returns, (1 - confidence_level) * 100)

    # Return on VaR
    mean_excess_return = np.mean(excess_returns)
    return_on_VaR = mean_excess_return / abs(VaR)

    # Annualize the return on VaR
    trading_days_per_year = 365
    annualized_return_on_VaR = return_on_VaR * np.sqrt(trading_days_per_year)

    return {
        'VaR': VaR,
        'Return on VaR': return_on_VaR,
        'Annualized Return on VaR': annualized_return_on_VaR
    }


def calculate_cdar(portfolio_returns, confidence_level=0.95):
    """
    Calculate the Conditional Drawdown at Risk (CDaR) for daily portfolio returns.

    Parameters:
    portfolio_returns (pd.Series): Daily returns of the portfolio.
    confidence_level (float): The confidence level for CDaR, default is 0.95.

    Returns:
    float: Conditional Drawdown at Risk (CDaR)
    """

    # Ensure portfolio_returns is a pandas Series
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns)

    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Calculate drawdowns
    peak = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - peak) / peak

    # Calculate the drawdown values
    drawdown_values = drawdowns[drawdowns < 0]


    # Check if there are any drawdown values
    if drawdown_values.empty:
        return 0.0
    
    # Calculate the threshold for the worst drawdowns
    threshold = np.percentile(drawdown_values, (1 - confidence_level) * 100)

    # Calculate CDaR
    cdar = drawdown_values[drawdown_values <= threshold].mean()

    return abs(cdar)

def calculate_monthly_cdar(portfolio_returns, period, confidence_level=0.95):
    """
    Calculate the Conditional Drawdown at Risk (CDaR) per month for daily portfolio returns.

    Parameters:
    portfolio_returns (np.array or pd.Series): Daily returns of the portfolio.
    confidence_level (float): The confidence level for CDaR, default is 0.95.

    Returns:
    pd.Series: Monthly CDaR values.
    """
    if not isinstance(portfolio_returns, pd.Series):
        portfolio_returns = pd.Series(portfolio_returns, index=pd.date_range(start='2021-01-01', periods=len(portfolio_returns), freq='D'))

    # Resample to get the start and end of each month
    periods = portfolio_returns.resample(period).apply(lambda x: x.index)

    monthly_cdar = []
    for period_item, dates in periods.items():

        if isinstance(dates, pd.Timestamp) or len(dates) < 2:
            break
        
        # Get the daily returns for the period
        period_portfolio_returns = portfolio_returns[dates[0]:dates[-1]]

        # Calculate CDaR for the period
        cdar = calculate_cdar(period_portfolio_returns, confidence_level)
        monthly_cdar.append({"date": period_item.isoformat(), "avg_cDaR": cdar})


    return np.array(monthly_cdar)

def calculate_omega_ratio(portfolio_returns, rf_values, threshold=0):
    """
    Calculate the Omega Ratio for a given set of portfolio returns relative to a specified threshold.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    rf_values (np.array): Daily risk-free rates.
    threshold (float): Threshold return value (default is 0).

    Returns:
    float: Omega Ratio
    """

    # Calculate the excess returns relative to the threshold
    excess_returns = portfolio_returns - rf_values - threshold

    # Calculate the probability-weighted returns
    positive_returns = excess_returns[excess_returns > 0]
    negative_returns = -excess_returns[excess_returns <= 0]
    
    probability_positive = len(positive_returns) / len(excess_returns)
    probability_negative = len(negative_returns) / len(excess_returns)

    # Calculate Omega Ratio
    omega_ratio = (1 + np.mean(positive_returns)) / (1 - np.mean(negative_returns))

    annualized_omega_ratio = omega_ratio * np.sqrt(365)

    return { 'Annualized Omega Ratio': annualized_omega_ratio, 'Probability Positive': probability_positive, 'Probability Negative': probability_negative } 


def calculate_calmar_ratio(portfolio_returns, rf_values):
    """
    Calculate the Calmar Ratio for a given set of portfolio returns.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    rf_values (np.array): Daily risk-free rates.

    Returns:
    float: Calmar Ratio
    """
    trading_days_per_year = 365
# Step 1: Calculate the excess returns by subtracting risk-free rate
    excess_returns = portfolio_returns - rf_values

    # Step 2: Calculate cumulative returns (based on portfolio returns, not excess returns)
    cumulative_returns = np.cumprod(1 + portfolio_returns)

    # Step 3: Calculate the annualized return (based on excess returns)
    total_return = cumulative_returns[-1]  # Total return of the portfolio
    num_days = len(portfolio_returns)
    annualized_return = (total_return ** (trading_days_per_year / num_days)) - 1  # Annualized total return

    # Step 4: Calculate the maximum drawdown (based on cumulative returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)  # Most negative drawdown

    # Step 5: Calculate the Calmar Ratio (if max_drawdown is 0, return infinity)
    if max_drawdown == 0:
        return np.inf
    else:
        calmar_ratio = annualized_return / abs(max_drawdown)

    return calmar_ratio


def calculate_capture_ratios(portfolio_returns, benchmark_returns):
    """
    Calculate the Upside Capture Ratio, Downside Capture Ratio, and Total Capture Ratio for a portfolio relative to a benchmark.

    Parameters:
    portfolio_returns (np.array): Daily returns of the portfolio.
    benchmark_returns (np.array): Daily returns of the benchmark (market index).

    Returns:
    dict: A dictionary containing the capture ratios.
    """
    # Identify positive and negative benchmark returns
    positive_benchmark = benchmark_returns > 0
    negative_benchmark = benchmark_returns <= 0

    positive_portfolio_returns = portfolio_returns[positive_benchmark]
    negative_portfolio_returns = portfolio_returns[negative_benchmark]
    
    # Calculate benchmark returns during positive and negative benchmark return periods
    positive_benchmark_returns = benchmark_returns[positive_benchmark]
    negative_benchmark_returns = benchmark_returns[negative_benchmark]
    
    # Calculate Upside Capture Ratio
    upside_capture_ratio = np.mean(positive_portfolio_returns) / np.mean(positive_benchmark_returns) * 100 if np.mean(positive_benchmark_returns) != 0 else np.nan
    
    # Calculate Downside Capture Ratio
    downside_capture_ratio = np.mean(negative_portfolio_returns) / np.mean(negative_benchmark_returns) * 100 if np.mean(negative_benchmark_returns) != 0 else np.nan
    
    # Calculate Total Capture Ratio
    total_capture_ratio = upside_capture_ratio - downside_capture_ratio

    if(type(upside_capture_ratio) == np.float64):
        upside_capture_ratio = upside_capture_ratio.item()
    if(type(downside_capture_ratio) == np.float64):
        downside_capture_ratio = downside_capture_ratio.item()
    if(type(total_capture_ratio) == np.float64):
        total_capture_ratio = total_capture_ratio.item()

    return {
        'Upside Capture Ratio': upside_capture_ratio,
        'Downside Capture Ratio': downside_capture_ratio,
        'Total Capture Ratio': total_capture_ratio
    }

def calculate_distribution_statistics(array):
    """
    Calculate the mean, standard deviation, kurtosis, and skewness of an array.

    Parameters:
    array (np.array): Target Array.

    Returns:
    dict: A dictionary containing the calculated statistics.
    """

    mean = np.mean(array)
    std = np.std(array)
    kurtosis_ = kurtosis(array)
    skewness = skew(array)

    return {
        'mean': mean.item(),
        'std': std.item(),
        'kurtosis': kurtosis_,
        'skewness': skewness
    }
