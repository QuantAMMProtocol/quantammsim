import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def import_st0x_data(file_path):
    """Import CMC 3-hour interval data from CSV.
    
    Args:
        file_path (str): Path to the CMC CSV file
        
    Returns:
        pd.DataFrame: Raw CMC data with parsed timestamps
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    # Read CSV with specified format
    df = pd.read_csv(
        file_path,
        sep=',',
        parse_dates=['timestamp'],
    )

    # Validate required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


    return df

def process_st0x_timestamps(df):
    """Process CMC data by adding 3 hours to all timestamps.

    Args:
        df (pd.DataFrame): DataFrame with CMC data

    Returns:
        pd.DataFrame: Processed DataFrame with adjusted timestamps
    """
    processed_df = df.copy()

    # Parse string timestamps and add 3 hours
    processed_df["timestamp"] = pd.to_datetime(
        processed_df["timestamp"], format="ISO8601"
    ) + timedelta(hours=3)

    # Convert timestamp to unix milliseconds
    processed_df["unix"] = processed_df["timestamp"].astype(np.int64) // 10**6

    # Set unix as index and sort
    processed_df.set_index("unix", inplace=True)
    processed_df.sort_index(inplace=True)

    # Remove any duplicate indices
    processed_df = processed_df[~processed_df.index.duplicated(keep="last")]

    return processed_df


def forward_fill_st0x_data(df, token):
    """Forward fill CMC data to create continuous minute-level time series.

    Args:
        df (pd.DataFrame): Processed CMC DataFrame

    Returns:
        pd.DataFrame: Forward-filled DataFrame with continuous 1-minute intervals
    """
    # Create complete minute-level index
    full_index = pd.date_range(
        start=pd.to_datetime(df.index.min(), unit="ms"),
        end=pd.to_datetime(df.index.max(), unit="ms"),
        freq="1min",
    )
    full_index = full_index.astype(np.int64) // 10**6

    # Reindex with the complete minute-level index
    df = df.reindex(full_index)

    # Forward fill all columns except volume-related ones
    # 1. Forward fill close price
    df["close"] = df["close"].ffill()

    # 2. For missing rows, set open/high/low to the previous close
    df["open"] = df["open"].fillna(df["close"].shift())
    df["high"] = df["high"].fillna(df["close"].shift())
    df["low"] = df["low"].fillna(df["close"].shift())

    # 4. Forward fill symbol
    df["symbol"] = df["symbol"].ffill()

    # Fill volume columns with 0
    for col in ['Volume USD', 'Volume ' + token]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

def load_and_process_st0x_data(file_path, token="TSLA", root_path=None):
    """Load and process CMC data with proper timestamp handling.
    
    Args:
        file_path (str): Path to the CMC CSV file
        token (str): Token symbol (default: "PAXG")
        root_path (str): Path to historical data directory
        
    Returns:
        pd.DataFrame: Processed data with continuous minute-level intervals
    """

    if root_path:
        base_path = Path(root_path) / f"{token}_1min_intraday_2020-01_to_2025-07.csv"
        new_path = (
            Path(root_path) / "new" / f"{token}_1min_intraday_2020-01_to_2025-07.csv"
        )

        # If we have both a base file and a new file, merge them
        if base_path.exists() and new_path.exists():
            # Load and process both datasets
            base_df = import_st0x_data(str(base_path))
            new_df = import_st0x_data(str(new_path))

            # For rows with same timestamps, keep new_df values, otherwise keep both
            merged_df = pd.concat([base_df.iloc[:-1], new_df]).drop_duplicates(subset=['timestamp'], keep='last')
            # Save merged result back to base location
            merged_df.to_csv(base_path, sep=";")

            # Use merged data for further processing
            raw_df = merged_df
        else:
            # If no merge needed, import raw data as normal
            raw_df = import_st0x_data(file_path)
    else:
        # No root_path directory specified, proceed with normal import
        raw_df = import_st0x_data(file_path)
    # Process timestamps
    processed_df = process_st0x_timestamps(raw_df)

    # Add required columns
    processed_df["symbol"] = f"{token}/USD"
    processed_df["Volume USD"] = processed_df["volume"]
    processed_df["Volume " + token] = processed_df["volume"] / processed_df["close"]

    # Drop unnecessary columns
    columns_to_keep = ['open', 'high', 'low', 'close', 'symbol', 
                        'Volume USD', f'Volume {token}']
    processed_df = processed_df[columns_to_keep]

    # Forward fill data
    filled_df = forward_fill_st0x_data(processed_df, token)

    # Add date column
    filled_df["date"] = pd.to_datetime(filled_df.index, unit="ms")

    return filled_df

def fill_missing_rows_with_st0x_historical_data(concatenated_df, root_path, token="PAXG"):
    """Fill missing rows using Historical Dataset.

    Args:
        concatenated_df (pd.DataFrame): Original dataframe with gaps
        root_path (str): Path to historical data directory
        token (str): Token symbol (default: "PAXG")

    Returns:
        tuple: (filled DataFrame, list of filled timestamps or None)
    """
    # Check if historical data file exists
    file_path = Path(root_path) / f"{token}_1min_intraday_2020-01_to_2025-07.csv"
    if not file_path.exists():
        print(f"No historical data file found for {token} at {file_path}")
        return concatenated_df, None

    # Import and process historical data
    historical_data = load_and_process_st0x_data(file_path, token, root_path)

    if historical_data.empty:
        print(f"No valid historical data found for {token}")
        return concatenated_df, None

    # Standardize index format for concatenated_df
    if len(concatenated_df.index) > 0:
        if len(str(int(concatenated_df.index.max()))) <= 10:
            concatenated_df.index = (concatenated_df.index * 1000).astype(int)

    # Ensure index formats match
    if len(str(int(historical_data.index.max()))) <= 10:
        historical_data.index = (historical_data.index * 1000).astype(int)

    # Drop unnecessary columns and rename to match expected format
    columns_to_keep = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "symbol": "symbol",
        "Volume USD": "Volume USD",
        "Volume " + token: "Volume " + token,
        "date": "date",
    }
    historical_data = historical_data.rename(columns=columns_to_keep)
    historical_data = historical_data[list(columns_to_keep.values())]
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
    filled_df.index.name = "unix"
    return filled_df, missing_timestamps.tolist()
