import numpy as np
import pandas as pd

def expand_daily_to_minute_data(daily_data, scale='ms'):
    """
    Expand daily data to minute-level data by repeating daily values for each minute.

    Parameters:
    - daily_data (DataFrame): A DataFrame containing daily values with a 'unix' timestamp.

    Returns:
    - DataFrame: A DataFrame with minute-level data where daily values are repeated for each minute.
    """
    # Convert unix timestamp to datetime
    daily_data["datetime"] = pd.to_datetime(daily_data["unix"], unit=scale)

    # Set datetime as index
    daily_data.set_index("datetime", inplace=True)

    # Create a date range with minute frequency
    minute_range = pd.date_range(start=daily_data.index.min(), end=daily_data.index.max(), freq='T')

    # Reindex the daily data to the minute range, forward filling the values
    minute_data = daily_data.reindex(minute_range, method='ffill')

    # Reset index to get 'datetime' back as a column
    minute_data.reset_index(inplace=True)

    # Rename the 'index' column to 'datetime'
    minute_data.rename(columns={'index': 'datetime'}, inplace=True)

    # Convert 'datetime' back to unix timestamp
    minute_data["unix"] = minute_data["datetime"].astype(np.int64) // 10**6

    return minute_data


def resample_minute_level_OHLC_data_to_daily(OHLC_df, ticker):
    """
    Resample minute-level OHLC data to daily OHLC data.

    Parameters:
    - OHLC_df (DataFrame): A DataFrame containing minute-level OHLC data with columns
                           'open', 'high', 'low', 'close', and 'volume'.

    Returns:
    - DataFrame: A DataFrame with daily OHLC data.
    """
    # Convert unix timestamp to datetime and set as index
    OHLC_df["datetime"] = pd.to_datetime(OHLC_df.index, unit="ms")
    OHLC_df.set_index("datetime", inplace=True)

    # Resample to daily data
    daily_OHLC = OHLC_df.resample("D").agg({
        'open_'+ticker: 'first',
        'high_'+ticker: 'max',
        'low_'+ticker: 'min',
        'close_'+ticker: 'last',
        'Volume USD_'+ticker: 'sum'
    })

    return daily_OHLC

def gkyz_var(open, high, low, close, close_tm1): # Garman Klass Yang Zhang extension OHLC volatility estimate
    return np.log(open/close_tm1)**2 + 0.5*(np.log(high/low)**2) \
        - (2*np.log(2)-1)*(np.log(close/open)**2)


def calculate_annualised_daily_volatility_from_minute_data(price_data, ticker):
    """
    Calculate the annualised daily volatility of an asset from minute-level price data.

    Parameters:
    - price_data (DataFrame): A DataFrame containing minute-level price data with columns
                              for 'unix' and 'close_' + ticker.
    - ticker (str): The ticker symbol of the asset to calculate volatility for.

    Returns:
    - DataFrame: A DataFrame with daily volatility calculated from the minute data.
    """
    # Convert unix timestamp to datetime and set as index
    price_data["datetime"] = pd.to_datetime(price_data.index, unit="ms")
    price_data.set_index("datetime", inplace=True)

    # Calculate log returns for the specified ticker
    price_data["log_return"] = np.log(
        price_data["close_" + ticker] / price_data["close_" + ticker].shift(1)
    )

    # Resample to daily data and calculate volatility (standard deviation of log returns)
    daily_volatility = price_data["log_return"].resample("D").std()

    return daily_volatility * np.sqrt(365.25)

