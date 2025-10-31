import pandas as pd
import os
import numpy as np
from quantammsim.utils.data_processing.binance_data_utils import plot_exchange_data
from quantammsim.utils.data_processing.datetime_utils import pddatetime_to_unixtimestamp


def fill_missing_rows_with_aerodrome_data(concatenated_df, token, root):
    """Fill missing rows with data from Aerodrome.

    Args:
        concatenated_df (pd.DataFrame): Original dataframe with potential gaps
        token (str): Token symbol
        root (str): Root directory path

    Returns:
        tuple: (filled DataFrame, list of filled timestamps or None)
    """
    file_path = os.path.join(
        root, f"aerodrome_prices_{token}_USD.csv"
    )

    # Check if Aerodrome data exists
    if not os.path.exists(file_path):
        print(f"No Aerodrome data available for {token}")
        return concatenated_df, None

    try:
        # Load Aerodrome data
        aerodrome_data = pd.read_csv(
            file_path, parse_dates=["date"]
        )

        # Convert date to unix timestamp (in milliseconds)
        aerodrome_data["unix"] = pddatetime_to_unixtimestamp(aerodrome_data["date"])

        # Create full OHLCV structure
        exchange_data = pd.DataFrame(
            {
                "unix": aerodrome_data["unix"],
                "date": aerodrome_data["date"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": f"{token}/USD",
                "open": aerodrome_data["price"],
                "high": aerodrome_data["price"],
                "low": aerodrome_data["price"],
                "close": aerodrome_data["price"],
                "Volume USD": 0,  # No volume data available
                f"Volume {token}": 0,  # No volume data available
            }
        )

        # Set index for proper joining
        exchange_data.set_index("unix", inplace=True)

        # Ensure timestamps are in milliseconds
        if len(str(int(exchange_data.index.max()))) <= 10:
            exchange_data.index = (exchange_data.index * 1000).astype(np.int64)

        # Ensure proper index setup for concatenated_df
        if concatenated_df.index.name != "unix":
            if "unix" in concatenated_df.columns:
                concatenated_df.set_index("unix", inplace=True)
            else:
                print("No unix column found in concatenated_df")
                return concatenated_df, None

        # Convert concatenated_df index to milliseconds if needed
        if len(str(int(concatenated_df.index.max()))) <= 10:
            concatenated_df.index = (concatenated_df.index * 1000).astype(np.int64)

        # Create visualization of exchange data
        os.makedirs(os.path.join(root, "aerodrome_data"), exist_ok=True)
        plot_success = plot_exchange_data(
            exchange_data,
            token,
            os.path.join(root, "aerodrome_data", f"exchange_data_{token}.png"),
        )
        if not plot_success:
            print(f"Warning: Could not create visualization for {token} Aerodrome data")

        # Find missing timestamps
        missing_timestamps = exchange_data.index.difference(concatenated_df.index)
        if missing_timestamps.empty:
            print("No missing timestamps to fill from Aerodrome")
            return concatenated_df, None

        # Fill missing data
        filled_in_df = pd.concat(
            [concatenated_df, exchange_data.loc[missing_timestamps]]
        )

        # Sort by unix timestamp
        filled_in_df.sort_index(inplace=True)

        # Remove duplicates, keeping original data where available
        filled_in_df = filled_in_df[~filled_in_df.index.duplicated(keep="first")]

        return filled_in_df, missing_timestamps.tolist()

    except Exception as e:
        print(f"Error processing Aerodrome data for {token}: {str(e)}")
        return concatenated_df, None
