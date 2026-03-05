import pandas as pd
import numpy as np
from pathlib import Path
from quantammsim.utils.data_processing.binance_data_utils import plot_exchange_data

def import_crypto_historical_data(token, root_path):
    """Import 1-minute data from Crypto Historical Dataset.

    Args:
        token (str): Token symbol (e.g., 'BTC', 'ETH')
        root_path (str): Path to the data directory
    """
    # Construct file path
    file_path = Path(root_path) / f"{token}_full_1min.txt"

    # Read CSV with specified format
    df = pd.read_csv(
        file_path,
        names=["datetime", "open", "high", "low", "close", "volume"],
        parse_dates=["datetime"],
    )

    # Convert UTC datetime to unix timestamp (ms)
    df["unix"] = df["datetime"].astype(np.int64) // 10**6

    # Add required columns to match existing format
    df["symbol"] = f"{token}/USD"
    df["Volume USD"] = df["volume"] * df["close"]  # Approximate USD volume
    df[f"Volume {token}"] = df["volume"]

    df = forward_fill_ohlcv_data(df, token)

    return df


def forward_fill_ohlcv_data(df, token):
    # Set unix as index
    df.set_index("unix", inplace=True)

    # Create complete minute-level index
    full_index = pd.date_range(
        start=pd.to_datetime(df.index.min(), unit="ms"),
        end=pd.to_datetime(df.index.max(), unit="ms"),
        freq="1min",
    )
    full_index = full_index.astype(np.int64) // 10**6
    # Reindex with the complete minute-level index
    df = df.reindex(full_index)

    # Handle missing values appropriately:
    # 1. Forward fill close price
    df["close"] = df["close"].ffill()

    # 2. For missing rows, set open/high/low to the previous close
    df["open"] = df["open"].fillna(df["close"].shift())
    df["high"] = df["high"].fillna(df["close"].shift())
    df["low"] = df["low"].fillna(df["close"].shift())

    # 3. Fill volume with 0 for missing periods
    if "volume" in df.columns:
        df = df.drop('volume', axis=1)
    if "datetime" in df.columns:
        df = df.drop('datetime', axis=1)
    df["Volume USD"] = df["Volume USD"].fillna(0)
    df[f"Volume {token}"] = df[f"Volume {token}"].fillna(0)

    # 4. Forward fill symbol
    df["symbol"] = df["symbol"].ffill()

    # Name the index 'unix'
    df.index.name = "unix"

    # Add datetime column using the unix index
    df["date"] = pd.to_datetime(df.index, unit="ms")
    return df

def fill_missing_rows_with_historical_data(concatenated_df, token, root_path):
    """Fill missing rows using Crypto Historical Dataset.

    Args:
        concatenated_df (pd.DataFrame): Original dataframe with gaps
        token (str): Token symbol
        root_path (str): Path to historical data directory

    Returns:
        tuple: (filled DataFrame, list of filled timestamps or None)
    """
    # Check if historical data file exists
    file_path = Path(root_path) / f"{token}_full_1min.txt"
    if not file_path.exists():
        print(f"No historical data file found for {token} at {file_path}")
        return concatenated_df, None

    # Standardize index format for concatenated_df
    if len(concatenated_df.index) > 0:
        if len(str(int(concatenated_df.index.max()))) <= 10:
            concatenated_df.index = (concatenated_df.index * 1000).astype(int)

    # Import historical data
    historical_data = import_crypto_historical_data(token, root_path)
    plot_exchange_data(historical_data, token, root_path + token + "_data.png")

    if historical_data.empty:
        print(f"No valid historical data found for {token}")
        return concatenated_df, None

    # Ensure index formats match
    if len(str(int(historical_data.index.max()))) <= 10:
        historical_data.index = (historical_data.index * 1000).astype(int)

    # Identify missing timestamps
    missing_timestamps = historical_data.index.difference(concatenated_df.index)
    if missing_timestamps.empty:
        print(f"No missing timestamps to fill from historical data for {token}")
        return concatenated_df, None

    # Get missing data rows
    missing_data = historical_data.loc[missing_timestamps]

    # Concatenate original data with missing rows
    filled_df = pd.concat([concatenated_df, missing_data])

    # Sort by index and remove duplicates
    filled_df.sort_index(inplace=True)
    filled_df = filled_df[~filled_df.index.duplicated(keep="first")]

    return filled_df, missing_timestamps.tolist()
