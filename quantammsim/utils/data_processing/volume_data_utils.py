import pandas as pd

def calculate_daily_volume_from_minute_data(price_data, ticker):
    """
    Calculate the daily dollar volume from minute-level trading data

    Parameters:
    - price_data (DataFrame): A DataFrame containing minute-level trading data with columns
                              for 'unix', 'Volume USD', and potentially other volume-related columns.

    Returns:
    - DataFrame: A DataFrame with daily volumes summed from the minute data.
    """
    # Convert unix timestamp to datetime and set as index
    price_data["datetime"] = pd.to_datetime(price_data.index, unit="ms")
    price_data.set_index("datetime", inplace=True)

    # Resample to daily data and sum volumes
    daily_dollar_volume = price_data["Volume USD_" + ticker].resample("D").sum()
    daily_mean_value = price_data["close_" + ticker].resample("D").mean()
    daily_volume = daily_dollar_volume / daily_mean_value
    # daily_volume.set_index("datetime")
    # need to set zero values to the minimum
    # daily_volume[daily_volume == 0] = min(daily_volume[daily_volume != 0])
    return daily_volume
