import numpy as np
import pandas as pd
import os.path
import os
import pyarrow as pa
import matplotlib.pyplot as plt
import dask.dataframe as dd

# from numba import jit
# from numba import float64
# from numba import int64
from Historic_Crypto import Cryptocurrencies, HistoricalData
from datetime import datetime, timezone
from importlib import resources as impresources
from quantammsim import data
from pathlib import Path
from bidask import edge
from quantammsim.utils.data_processing.binance_data_utils import (
    concat_csv_files,
    plot_exchange_data,
)
from quantammsim.utils.data_processing.coinbase_data_utils import (
    _cleaned_up_coinbase_data,
    fill_missing_rows_with_coinbase_data,
)
from quantammsim.utils.data_processing.amalgamated_data_utils import (
    fill_missing_rows_with_historical_data,
    forward_fill_ohlcv_data,
)
from quantammsim.utils.data_processing.minute_daily_conversion_utils import (
    calculate_annualised_daily_volatility_from_minute_data,
    expand_daily_to_minute_data,
    resample_minute_level_OHLC_data_to_daily,
)
from quantammsim.utils.data_processing.datetime_utils import (
    datetime_to_unixtimestamp,
    unixtimestamp_to_datetime,
    unixtimestamp_to_minute_datetime,
    unixtimestamp_to_midnight_datetime,
    unixtimestamp_to_precise_datetime,
    pddatetime_to_unixtimestamp,
)
import gc

mc_data_available_for = ["ETH", "BTC"]


def default_set_or_get(dictionary, key, default, augment=True):
    value = dictionary.get(key)
    if value is None:
        if augment:
            dictionary[key] = default
        return default
    else:
        return value


def default_set(dictionary, key, default):
    value = dictionary.get(key)
    if value is None:
        dictionary[key] = default


def start_and_end_calcs(
    unix_values,
    prices=None,
    oracle_values=None,
    start_date=None,
    end_date=None,
):
    if start_date is not None and end_date is not None:
        start_idx = np.where(unix_values == start_date)[0][0]

        # prep prices/oracle_values/unix_values so that prices naturally get chunked into bit starting at midnight
        remainder_idx = start_idx % 1440
        unix_values = unix_values[remainder_idx:]

        if prices is not None:
            prices = prices[remainder_idx:]
        if oracle_values is not None:
            oracle_values = oracle_values[remainder_idx:]

        start_idx = np.where(unix_values == start_date)[0][0]
        end_idx = np.where(unix_values == end_date)[0][0] + 1
    else:
        start_idx = 0
        end_idx = len(prices)

    bout_length = end_idx - start_idx
    return (
        start_idx,
        end_idx,
        bout_length,
        unix_values,
        prices,
        oracle_values,
        remainder_idx,
    )

def load_data_from_candles_parquet(parquet_path, asset):
    """Load and process data from a candles parquet file for a specific asset.
    
    Args:
        parquet_path (str): Path to the parquet file
        asset (dict): Dictionary containing pair_id and token name
        
    Returns:
        pd.DataFrame: Processed dataframe with standardized columns
    """
    # Read the parquet file
    df = dd.read_parquet(parquet_path)
    
    # Filter for specific asset
    filtered_df = df[df["pair_id"] == asset["pair_id"]].compute()

    # Create unix timestamp column
    filtered_df["unix"] = filtered_df["timestamp"].astype(np.int64)

    # Make volume columns
    filtered_df["Volume USD"] = filtered_df["volume"].fillna(0)
    filtered_df[f"Volume {asset['token']}"] = filtered_df["volume"] / filtered_df["close"]

    # Make symbol column
    filtered_df["symbol"] = asset["token"]

    # Drop unused columns
    filtered_df = filtered_df.drop(
        columns=[
            "timestamp",
            "volume",
            "pair_id", 
            "avg",
            "start_block",
            "end_block",
            "buy_volume",
            "sell_volume",
            "buys",
            "sells",
            "exchange_rate",
        ]
    )
    
    return filtered_df

def get_available_years(root, exchange_directory, token, numeraire, prefix):
    """Check which years of data are available for a given token.

    Args:
        root (str): Root directory path
        exchange_directory (str): Exchange directory name in root
        token (str): Token symbol
        numeraire (str): Numeraire symbol
        prefix (str): File prefix (e.g., 'Binance_')

    Returns:
        list: List of years with available data
    """
    available_years = []
    for year in range(2018, datetime.now().year + 1):
        filename = f"{prefix}{token}{numeraire}_{year}_minute.csv"
        if os.path.exists(os.path.join(root, exchange_directory, filename)):
            available_years.append(str(year))
    return available_years


def fill_in_missing_rows_with_exchange_data(
    concatenated_df, token1, numeraire, root, raw_data_folder, prefix
):
    """Fill missing rows with data from another exchange.

    Args:
        concatenated_df (pd.DataFrame): Original dataframe
        token1 (str): Token symbol
        numeraire (str): Numeraire symbol
        root (str): Root directory path
        raw_data_folder (str): Folder containing raw exchange data
        prefix (str): Exchange prefix for filenames

    Returns:
        tuple: (filled DataFrame, list of filled timestamps or None)
    """
    # Check if exchange data exists
    available_years = get_available_years(
        root, raw_data_folder, token1, numeraire, prefix
    )
    if not available_years:
        print(f"No {prefix.strip('_')} data available for {token1}")
        return concatenated_df, None

    # Load exchange data
    exchange_data = concat_csv_files(
        root=root + raw_data_folder,
        save_root=root + "concat_" + raw_data_folder,
        token1=token1,
        token2=numeraire,
        prefix=prefix,
        postfix="_minute",
        years_array_str=available_years,
    )

    if exchange_data.empty:
        print(f"No valid {prefix.strip('_')} data found for {token1}")
        return concatenated_df, None

    # Ensure proper index setup
    if exchange_data.index.name != "unix":
        exchange_data.set_index("unix", inplace=True)

    # Standardize timestamp format
    if len(str(int(exchange_data.index.max()))) <= 10:
        exchange_data.index = (exchange_data.index * 1000).astype(int)

    if len(concatenated_df.index) > 0:
        if len(str(int(concatenated_df.index.max()))) <= 10:
            concatenated_df.index = (concatenated_df.index * 1000).astype(int)

    os.makedirs(os.path.join(root, "concat_" + raw_data_folder), exist_ok=True)
    # Create visualization of exchange data
    plot_success = plot_exchange_data(
        exchange_data,
        token1,
        os.path.join(root, "concat_" + raw_data_folder, f"exchange_data_{token1}.png"),
    )
    if not plot_success:
        print(
            f"Warning: Could not create visualization for {token1} {prefix.strip('_')} data"
        )

    # Find and fill missing data
    missing_timestamps = exchange_data.index.difference(concatenated_df.index)
    if missing_timestamps.empty:
        print(f"No missing timestamps to fill from {prefix.strip('_')}")
        return concatenated_df, None

    # Get missing rows and combine data
    missing_data = exchange_data.loc[missing_timestamps]
    filled_in_df = pd.concat([concatenated_df, missing_data])

    # Clean up the combined data
    filled_in_df.sort_index(inplace=True)
    filled_in_df = filled_in_df[~filled_in_df.index.duplicated(keep="first")]

    return filled_in_df, missing_timestamps.tolist()


def merge_exchange_data_frames(base_df, exchange_df, token1, root, raw_data_folder, prefix):
    """Merge two DataFrames with exchange data, handling missing timestamps and duplicates.
    
    Args:
        base_df (pd.DataFrame): Base DataFrame to fill gaps in
        exchange_df (pd.DataFrame): Exchange data to fill gaps with
        token1 (str): Token symbol for visualization
        root (str): Root directory path for saving visualizations
        raw_data_folder (str): Folder name for saving visualizations
        prefix (str): Exchange prefix for logging
        
    Returns:
        tuple: (merged DataFrame, list of filled timestamps or None)
    """
    # Ensure proper index setup
    for df in [base_df, exchange_df]:
        if df.index.name != "unix" and "unix" in df.columns:
            df.set_index("unix", inplace=True)
            
    # Standardize timestamp format to milliseconds
    for df in [base_df, exchange_df]:
        if len(df.index) > 0 and len(str(int(df.index.max()))) <= 10:
            df.index = (df.index * 1000).astype(np.int64)
    # Create visualization of exchange data
    plot_success = plot_exchange_data(
        exchange_df,
        token1,
        os.path.join(root, "concat_" + raw_data_folder, f"exchange_data_{token1}.png"),
    )
    if not plot_success:
        print(f"Warning: Could not create visualization for {token1} {prefix.strip('_')} data")

    # Find and fill missing data
    missing_timestamps = exchange_df.index.difference(base_df.index)
    if missing_timestamps.empty:
        print(f"No missing timestamps to fill from {prefix.strip('_')}")
        return base_df, None

    # Get missing rows and combine data
    missing_data = exchange_df.loc[missing_timestamps]
    filled_df = pd.concat([base_df, missing_data])

    # Clean up the combined data
    filled_df.sort_index(inplace=True)
    filled_df = filled_df[~filled_df.index.duplicated(keep="first")]

    return filled_df, missing_timestamps.tolist()


def update_historic_data_old(token, root):
    outputPath = root + "combined_data/"
    outputMinutePath = outputPath + token + "_USD.csv"
    parquetPath = outputPath + token + "_USD.parquet"
    path = root + "concat_binance_data/" + token + "_USD.csv"
    dailyPath = outputPath + token + "_USD_daily.csv"
    hourlyPath = outputPath + token + "_USD_hourly.csv"

    long_years_array_str = ["2020", "2021", "2022", "2023", "2024"]
    short_years_array_str = long_years_array_str[1:]
    if os.path.isfile(root + "concat_binance_data/" + token + "_USD.csv") is False:
        print(token)
        try:
            concated_df = concat_csv_files(
                root=root + "raw_binance_data/",
                save_root=root + "concat_binance_data/",
                token1=token,
                token2="USDT",
                prefix="Binance_",
                postfix="_minute",
                years_array_str=long_years_array_str,
            )
        except Exception as e:
            print(e)
            concated_df = concat_csv_files(
                root=root + "raw_binance_data/",
                save_root=root + "concat_binance_data/",
                token1=token,
                token2="USDT",
                prefix="Binance_",
                postfix="_minute",
                years_array_str=short_years_array_str,
            )

    else:
        print("reading from csv")
        concated_df = pd.read_csv(root + "concat_binance_data/" + token + "_USD.csv")

    concated_df, filled_gemini__unix_values = fill_in_missing_rows_with_exchange_data(
        concated_df, token, root, "raw_gemini_data/", "Gemini_"
    )

    concated_df, filled_bitstamp_unix_values = fill_in_missing_rows_with_exchange_data(
        concated_df, token, root, "raw_bitstamp_data/", "Bitstamp_"
    )

    # Add historical data filling
    concated_df, filled_historical_unix_values = fill_missing_rows_with_historical_data(
        concated_df, token, root + "historical_data/"
    )
    # filled_bitstamp_unix_values = None
    # if max(np.diff(np.array(out.index))) > 60000:
    #    raise Exception
    concated_df.to_csv(root + "concat_binance_data/" + token + "_USD.csv")

    concat_csv = pd.read_csv(
        path,
        dtype={
            "unix": float,
            "date": "string",
            "symbol": "string",
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "Volume USD": float,
            "Volume " + token: float,
            "tradecount": float,
        },
    )
    original_unix_values = concat_csv["unix"].to_numpy()
    if "tradecount" in concat_csv.columns:
        concat_csv = concat_csv.drop(columns=["tradecount"])

    totalMissingUnixPoints = list()
    totalMissingClosePoints = list()

    # Print rows with an index or unix value of 1606716420000
    # Reindex on minute unix
    concat_csv.set_index("unix", inplace=True)

    print("fill with coinbase")
    if os.path.exists(root + "coinbase_data/" + token + "_cb_sorted_.csv"):
        csvData, coinbaseFilledUnixVals = fill_missing_rows_with_coinbase_data(
            concat_csv, token, root
        )
    else:
        csvData = concat_csv
        csvData["unix"] = concat_csv.index
        coinbaseFilledUnixVals = []
    concat_csv["unix"] = concat_csv.index
    # Reindex on minute unix
    # Create a new DataFrame with unix index and minute rows between csvData min and max
    new_index = (
        pd.date_range(
            start=pd.to_datetime(csvData.index.min(), unit="ms"),
            end=pd.to_datetime(csvData.index.max(), unit="ms"),
            freq="T",
        ).astype(int)
        // 10**6
    )

    new_csvData = pd.DataFrame(index=new_index)
    new_csvData.index.name = "unix"
    print("joining")
    # TODO CHECKOUT
    # Populate the new DataFrame with the data from the original csvData
    csvData = new_csvData.join(csvData, how="left", lsuffix="_left", rsuffix="_right")
    csvData["unix"] = csvData.index
    del new_csvData
    gc.collect()
    # Save the total unix with empty rows
    totalMissingUnixPoints = csvData[csvData.isnull().any(axis=1)].index.tolist()
    print("forward filling")
    # Forward fill the empty rows
    # Forward fill all rows apart from columns 'date' and 'unix'
    columns_to_ffill = csvData.columns.difference(["date", "unix"])
    csvData[columns_to_ffill] = csvData[columns_to_ffill].ffill()
    # Retrieve the unix values where the date column is null
    missing_date_unix_values = csvData[csvData["date"].isnull()].index
    totalMissingClosePoints = csvData[csvData.index.isin(totalMissingUnixPoints)][
        "close"
    ].tolist()

    # Generate dates in the new dataframe given the unix values
    missing_dates_df = pd.DataFrame(
        {
            "unix": missing_date_unix_values,
            "date": pd.to_datetime(missing_date_unix_values, unit="ms").strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
    )

    # Merge the date column from the new dataframe into csvData
    csvData.update(missing_dates_df.set_index("unix"))

    csvData["unix"] = csvData.index
    print("plotting")
    # Plotting the data
    plt.figure(figsize=(14, 7))
    # Original csvData
    plt.plot(
        pd.to_datetime(
            csvData[csvData["unix"].isin(original_unix_values)]["unix"], unit="ms"
        ),
        csvData[csvData["unix"].isin(original_unix_values)]["close"],
        label="Binance Minute Data",
        linestyle="None",
        marker="o",
        markersize=0.5,
    )

    if "coinbaseFilledUnixVals" in locals():
        coinbase_filled_data = csvData[
            csvData["unix"].isin(coinbaseFilledUnixVals)
            & ~csvData["unix"].isin(concat_csv["unix"])
        ]

        plt.plot(
            pd.to_datetime(coinbase_filled_data["unix"], unit="ms"),
            coinbase_filled_data["close"],
            label="Coinbase Minute Data",
            linestyle="None",
            marker="o",
            markersize=0.5,
        )

    # Gemini filled data
    if filled_gemini__unix_values:
        gemini_filled_data = csvData[csvData["unix"].isin(filled_gemini__unix_values)]
        plt.plot(
            pd.to_datetime(gemini_filled_data["unix"], unit="ms"),
            gemini_filled_data["close"],
            label="Gemini Minute Data",
            linestyle="None",
            marker="o",
            markersize=0.5,
        )

    # Bitstamp filled data
    if filled_bitstamp_unix_values:
        bitstamp_filled_data = csvData[
            csvData["unix"].isin(filled_bitstamp_unix_values)
        ]
        plt.plot(
            pd.to_datetime(bitstamp_filled_data["unix"], unit="ms"),
            bitstamp_filled_data["close"],
            label="Bitstamp Minute Data",
            linestyle="None",
            marker="o",
            markersize=0.5,
        )

    # Total missing unix points data
    if len(totalMissingUnixPoints) > 0:
        plt.plot(
            pd.to_datetime(totalMissingUnixPoints, unit="ms"),
            totalMissingClosePoints,
            label="Forward Filled Data",
            linestyle="None",
            marker="o",
            markersize=0.5,
        )

    plt.xlabel("Date (YY-MM-DD)")
    plt.ylabel("Close Price")
    plt.title(f"{token} Close Price Over Time")
    plt.legend()
    # Save the plot to a file
    plot_filename = outputPath + f"{token}_close_price_over_time.png"
    plt.savefig(plot_filename)

    csvData = csvData.reset_index(drop=True).sort_values(by="unix", ascending=True)

    # Plotting the final data
    plt.figure(figsize=(14, 7))
    plt.plot(
        pd.to_datetime(csvData["unix"], unit="ms"),
        csvData["close"],
        label="Close Price",
        marker="o",
        markersize=0.5,
    )
    plt.xlabel("Date (YY-MM-DD)")
    plt.ylabel("Close Price")
    plt.title(f"{token} Close Price Over Unix Time")
    plt.legend()
    # Save the plot to a file
    final_plot_filename = outputPath + f"{token}_final_close_price_over_unix_time.png"
    plt.savefig(final_plot_filename)
    csvData = csvData.sort_values(by="unix", ascending=True)

    # usdtData = pd.read_csv(
    #     root + "USDT_USD.csv",
    #     dtype={
    #         "unix": float,
    #         "date": "string",
    #         "symbol": "string",
    #         "open": float,
    #         "high": float,
    #         "low": float,
    #         "close": float,
    #         "Volume USD": float,
    #         "Volume " + token: float,
    #         "tradecount": float,
    #     },
    # )

    # usdtData = usdtData.set_index("unix")
    # usdtToken = token + "/USDT"
    # for index, row in csvData.iterrows():
    #    try:
    #        if(usdtToken == row["symbol"]):
    #            usdtRow = usdtData.iloc[usdtData.index.get_loc(row["unix"])]
    #            csvData.at[index, "close"] = row["close"] * usdtRow["close"]
    #            csvData.at[index, "open"] = row["open"] * usdtRow["open"]
    #            csvData.at[index, "low"] = row["low"] * usdtRow["low"]
    #            csvData.at[index, "high"] = row["high"] * usdtRow["high"]
    #            csvData.at[index, "symbol"] = token + "/USD"
    #    except  Exception as e:
    #        print(e)

    csvData["unix"] = csvData["unix"].astype(int)
    cols = csvData.columns.tolist()
    cols.remove("unix")
    cols = ["unix"] + cols
    csvData = csvData[cols]

    # csvData.to_csv(outputMinutePath, mode="w", index=False)

    plot_exchange_data(csvData.set_index("unix"), token, outputMinutePath[:-4] + ".png")

    csvData.to_parquet(parquetPath, engine="pyarrow")
    csvData[csvData["date"].str.contains(":00:00")].to_csv(
        hourlyPath, mode="w", index=False
    )
    # Create a minute level csv from the hourly data
    hourly_data = csvData[csvData["date"].str.contains(":00:00")]
    hourly_data.set_index("unix", inplace=True)
    hourly_data = hourly_data[~hourly_data.index.duplicated(keep="first")]
    hourly_data = hourly_data.reindex(
        pd.date_range(
            start=pd.to_datetime(hourly_data.index.min(), unit="ms"),
            end=pd.to_datetime(hourly_data.index.max(), unit="ms"),
            freq="H",
        ).astype(int)
        // 10**6
    )
    hourly_data["unix"] = hourly_data.index
    hourly_data["close"] = hourly_data["close"].interpolate(method="linear")

    # Create a new DataFrame with minute level data
    minute_index = (
        pd.date_range(
            start=pd.to_datetime(hourly_data.index.min(), unit="ms"),
            end=pd.to_datetime(hourly_data.index.max(), unit="ms"),
            freq="T",
        ).astype(int)
        // 10**6
    )
    minute_data = pd.DataFrame(index=minute_index)
    minute_data.index.name = "unix"
    minute_data["unix"] = minute_data.index

    # Populate the new DataFrame with the data from the hourly_data
    minute_data = minute_data.join(hourly_data["close"], how="left")
    minute_data["close"] = minute_data["close"].interpolate(method="linear")

    # Calculate the average price % difference between the actual original minute level prices and the new linear interpolated minute prices
    csvData_reset = csvData.reset_index(drop=True)
    minute_data_reset = minute_data.reset_index(drop=True)
    merged_data = csvData_reset[["unix", "close"]].merge(
        minute_data_reset[["unix", "close"]],
        on="unix",
        suffixes=("_original", "_interpolated"),
    )
    merged_data["price_pct_diff"] = (
        (merged_data["close_original"] / merged_data["close_interpolated"]) * 100
    ) - 100

    # Plot the average price % difference
    plt.figure(figsize=(14, 7))
    plt.plot(
        pd.to_datetime(merged_data["unix"], unit="ms"),
        merged_data["price_pct_diff"],
        label="avg % deviation from interpolated price",
    )
    plt.xlabel("Date (YY-MM-DD)")
    plt.ylabel("% deviation from interpolated price")
    plt.title(
        f"{token} Minute Price average % deviation from Interpolated Price Over Time"
    )
    plt.legend()
    # Save the plot to a file
    price_pct_plot_filename = outputPath + f"{token}_price_pct_diff_over_time.png"
    plt.savefig(price_pct_plot_filename)

    plt.close()

    # Aggregate volume data for daily rows
    daily_data = csvData[csvData["date"].str.contains(":01:00:00")].copy()
    daily_data.set_index("unix", inplace=True)

    ## Aggregate volume columns over the day
    # volume_columns = [col for col in csvData.columns if col.startswith('Volume')]
    # for col in volume_columns:
    #    daily_data[col] = csvData.resample('D', on='unix')[col].sum().reindex(daily_data.index)

    # Save the daily data to CSV
    daily_data.to_csv(dailyPath, mode="w", index=False)


def get_binance_vision_data(token, numeraire, root):
    """Get data directly from binance.vision using binance_historical_data package.
    
    Args:
        token (str): Token symbol
        numeraire (str): Quote currency (e.g., 'USDT', 'USD')
        root (str): Root directory path
        
    Returns:
        pd.DataFrame: DataFrame with standardized format or None if data not available
    """
    from binance_historical_data import BinanceDataDumper
    # Initialize downloader in a subdirectory to keep monthly files separate
    vision_dir = os.path.join(root, "binance_vision_data")
    os.makedirs(vision_dir, exist_ok=True)
    
    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=vision_dir,
        asset_class="spot",
        data_type="klines",
        data_frequency="1m"
    )
    
    # Download all available data
    data_dumper.dump_data(
        tickers=[f"{token}{numeraire}"],
        is_to_update_existing=True
    )
    
    # Find and combine all monthly files
    monthly_files = []
    base_path = os.path.join(vision_dir, "spot", "monthly", "klines", f"{token}{numeraire}", "1m")
    
    if os.path.exists(base_path):
        for filename in os.listdir(base_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(base_path, filename)
                df = pd.read_csv(file_path, header=None, names=[
                    "unix",
                    "open",
                    "high",
                    "low", 
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_base_volume",
                    "taker_quote_volume",
                    "ignore"
                ])
                monthly_files.append(df)
    
    if not monthly_files:
        print(f"No Binance vision data found for {token}")
        return None

    daily_files = []
    base_path = os.path.join(vision_dir, "spot", "daily", "klines", f"{token}{numeraire}", "1m")
    
    if os.path.exists(base_path):
        for filename in os.listdir(base_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(base_path, filename)
                df = pd.read_csv(file_path, header=None, names=[
                    "unix",
                    "open",
                    "high",
                    "low", 
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_base_volume",
                    "taker_quote_volume",
                    "ignore"
                ])
                daily_files.append(df)

    # Combine and format data
    combined_df = pd.concat(monthly_files + daily_files)
    # Convert unix timestamps to milliseconds if they're in nanoseconds or seconds
    # Typical millisecond timestamps are ~13 digits
    # Nanosecond timestamps are ~19 digits
    # Second timestamps are ~10 digits
    # First convert nanoseconds to milliseconds
    combined_df["unix"] = combined_df["unix"].apply(
        lambda x: x // 1_000_000 if len(str(int(x))) > 13 else x
    )
    # Then convert seconds to milliseconds
    combined_df["unix"] = combined_df["unix"].apply(
        lambda x: x * 1000 if len(str(int(x))) <= 10 else x
    )
    combined_df["date"] = pd.to_datetime(combined_df["unix"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
    combined_df["symbol"] = f"{token}/{numeraire}"
    combined_df[f"Volume {token}"] = combined_df["volume"]
    combined_df["Volume USD"] = combined_df["quote_volume"]
    
    # Select and order columns to match expected format
    result_df = combined_df[["unix", "date", "symbol", "open", "high", "low", "close",
                            "Volume USD", f"Volume {token}", "trades"]]
    result_df = result_df.sort_values("unix").reset_index(drop=True)
    
    return result_df
    
    # except Exception as e:
    #     print(f"Failed to get Binance vision data for {token}: {str(e)}")
    #     return None

def update_historic_data(token, root):
    """Update historic data for a given token, handling reruns gracefully.

    Args:
        token (str): Token symbol (e.g., 'BTC', 'ETH')
        root (str): Root directory path
    """
    outputPath = root + "combined_data/"
    outputMinutePath = outputPath + token + "_USD.csv"
    parquetPath = outputPath + token + "_USD.parquet"
    path = root + "concat_binance_data/" + token + "_USD.csv"
    minutePath = outputPath + token + "_USD.csv"
    dailyPath = outputPath + token + "_USD_daily.csv"
    hourlyPath = outputPath + token + "_USD_hourly.csv"

    # Create output directories if they don't exist
    os.makedirs(outputPath, exist_ok=True)
    os.makedirs(root + "concat_binance_data/", exist_ok=True)

    # Try binance.vision data first
    print(f"Attempting to get Binance vision data for {token}")
    filled_timestamps = {}
    concated_df = get_binance_vision_data(token, "USDT", root)
    if concated_df is not None:
        print(f"No Binance vision data available for {token}")
        if concated_df.index.name != "unix":
            concated_df.set_index("unix", inplace=True)
        filled_binance_vision_unix_values = concated_df.index.tolist()
        filled_timestamps["Binance Vision"] = filled_binance_vision_unix_values
    else:
        print(f"No Binance vision data available for {token}")
        concated_df = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "Volume USD",
                f"Volume {token}",
            ]
        )
        filled_timestamps["Binance Vision"] = []
    # Fill gaps with cryptodatadownload Binance data
    print("Filling gaps with cryptodatadownload Binance data")
    concated_df, filled_binance_unix_values = fill_in_missing_rows_with_exchange_data(
        concated_df, token, "USDT", root, "raw_binance_data/", "Binance_"
    )
    filled_timestamps["Binance CDD"] = filled_binance_unix_values
    if concated_df is None:
        print(f"No cryptodatadownload Binance data available for {token}")
        # Try getting historical data years as fallback
        historical_years = get_available_years(
            root, "historical_data", token, "USD", ""
        )
        if historical_years:
            print(f"Found historical data years for {token}: {historical_years}")
            available_years = historical_years
            concated_df = import_crypto_historical_data(
                token, root + "historical_data/"
            )
        else:
            print(f"No historical data years found for {token}")
            # Create empty DataFrame with correct columns and format
            concated_df = pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "Volume USD",
                    f"Volume {token}",
                ]
            )
            concated_df.index.name = "unix"
            concated_df["symbol"] = f"{token}/USD"    
    # Ensure proper index setup
    if "unix" in concated_df.columns:
        concated_df.set_index("unix", inplace=True)
    # Fill missing data from different sources

    print("Filling gaps with Coinbase data")
    if os.path.exists(root + "coinbase_data/" + token + "_cb_sorted_.csv"):
        concated_df, coinbaseFilledUnixVals = fill_missing_rows_with_coinbase_data(
            concated_df, token, root
        )
        # concated_df = concated_df.reset_index()
        filled_timestamps["Coinbase"] = coinbaseFilledUnixVals
    else:
        print(f"No Coinbase data available for {token}")
        coinbaseFilledUnixVals = []

    sources = [
        ("Gemini", "raw_gemini_data/", "Gemini_", "USD"),
        ("Bitstamp", "raw_bitstamp_data/", "Bitstamp_", "USD"),
    ]

    for source_name, data_folder, prefix, numeraire in sources:
        print(f"Filling gaps with {source_name} data")
        concated_df, filled_unix_values = fill_in_missing_rows_with_exchange_data(
            concated_df.copy(), token, numeraire, root, data_folder, prefix
        )
        if filled_unix_values:
            filled_timestamps[source_name] = filled_unix_values

    # Fill with historical data
    print("Filling remaining gaps with historical data")
    concated_df, filled_historical_unix_values = fill_missing_rows_with_historical_data(
        concated_df.copy(), token, root + "historical_data/"
    )
    if filled_historical_unix_values:
        filled_timestamps["Historical"] = filled_historical_unix_values

    # if ticker is in a harcoded dict, load from parquet

    assets = [
    {"pair_id": 3010484, "token": "PEPE"},
    {"pair_id": 1497, "token": "BAL"},
    {"pair_id": 5241020, "token": "VVV"},
    {"pair_id": 5388941, "token": "KAITO"},
    {"pair_id": 4569519, "token": "DEGEN"},
    {"pair_id": 4567392, "token": "VIRTUAL"},
    ]
    if token in [asset["token"] for asset in assets]:
        print("Filling remaining gaps with candles data")
        asset = next(asset for asset in assets if asset["token"] == token)
        exchange_df = load_data_from_candles_parquet(root + "candles-1m.parquet", asset)
        exchange_df = forward_fill_ohlcv_data(exchange_df.copy(), asset["token"])
        concated_df, filled_candles_unix_values = merge_exchange_data_frames(concated_df, exchange_df, token, root, "raw_candles_data/", "Candles_")
        filled_timestamps["Candles"] = filled_candles_unix_values

    # Ensure data is properly sorted and has no duplicates
    concated_df = concated_df.sort_index()
    concated_df = concated_df[~concated_df.index.duplicated(keep="first")]

    # Reset index for CSV export
    concated_df = concated_df.reset_index()

    # Save the processed data
    print(f"Saving processed data for {token}")
    concated_df = forward_fill_ohlcv_data(concated_df.copy(), token).reset_index()
    # Check that all unix timestamp differences are 60000 ms (1 minute)
    unix_diffs = np.diff(concated_df["unix"].values)
    if not np.all(unix_diffs == 60000):
        invalid_diffs = np.where(unix_diffs != 60000)[0]
        first_invalid = invalid_diffs[0]
        invalid_time = pd.to_datetime(
            concated_df["unix"].iloc[first_invalid], unit="ms"
        )
        raise Exception(
            f"Invalid unix timestamp difference found at index {first_invalid} ({invalid_time}). All differences should be 60000ms (1 minute)."
        )
    # concated_df.to_csv(minutePath, index=False)
    # Create visualization of the data sources
    print(f"Creating visualizations for {token}")
    plot_exchange_data(concated_df.set_index("unix"), token, minutePath[:-4] + ".png")

    plt.figure(figsize=(14, 7))

    # Plot original Binance data
    # plt.plot(
    #     pd.to_datetime(concated_df["unix"], unit="ms"),
    #     concated_df["close"],
    #     label="Binance Minute Data",
    #     linestyle="None",
    #     marker="o",
    #     markersize=0.5,
    #     color="yellow",
    # )

    # Plot filled data from each source
    colors = {
        "Gemini": "green",
        "Bitstamp": "red",
        "Historical": "purple",
        "Coinbase": "blue",
        "Binance CDD": "orange",
        "Binance Vision": "cyan",
        "Candles": "magenta",
    }
    for source, timestamps in filled_timestamps.items():
        if timestamps:
            source_data = concated_df[concated_df["unix"].isin(timestamps)]
            plt.plot(
                pd.to_datetime(source_data["unix"], unit="ms"),
                source_data["close"],
                label=f"{source} Minute Data",
                linestyle="None",
                marker="o",
                markersize=0.5,
                color=colors.get(source, "blue"),
            )

    plt.xlabel("Date (YY-MM-DD)")
    plt.ylabel("Close Price")
    plt.title(f"{token} Close Price Over Time")
    plt.legend()
    plt.savefig(outputPath + f"{token}_data_sources.png")
    plt.close()

    # Process for different time frequencies
    print(f"Processing different time frequencies for {token}")

    # Hourly data
    concated_df_hourly = concated_df.copy()

    # Debug: Print columns before processing

    # Convert date to datetime
    concated_df_hourly["date"] = pd.to_datetime(concated_df_hourly["date"])
    concated_df_hourly = concated_df_hourly.set_index("date")

    # Debug: Print columns after setting index

    # Perform hourly resampling with only the columns that exist
    agg_dict = {
        "unix": "last",
        "symbol": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "Volume USD": "sum",
        f"Volume {token}": "sum",
    }

    # Filter aggregation dictionary to only include existing columns
    agg_dict = {k: v for k, v in agg_dict.items() if k in concated_df_hourly.columns}

    # Perform resampling
    hourly_data = concated_df_hourly.resample("1H").agg(agg_dict).reset_index()

    # Save hourly data
    hourly_data.to_csv(hourlyPath, index=False)

    # Daily data
    concated_df_daily = concated_df.copy()
    concated_df_daily["date"] = pd.to_datetime(concated_df_daily["date"])
    concated_df_daily = concated_df_daily.set_index("date")
    # Filter to only include existing columns
    agg_dict = {k: v for k, v in agg_dict.items() if k in concated_df_daily.columns}

    # Perform daily resampling
    daily_data = concated_df_daily.resample("D").agg(agg_dict).reset_index()

    # Save daily data
    daily_data.to_csv(dailyPath, index=False)
    # Save parquet format
    concated_df["date"] = concated_df["date"].astype(str)
    concated_df["symbol"] = concated_df["symbol"].astype(str)
    # Try direct parquet save without conversion
    concated_df.to_parquet(parquetPath, engine="pyarrow")

    print(f"Completed processing for {token}")
    return concated_df


def createMissingDataFrameFromClosePrices(startUnix, closePrices, token):
    totalMissingUnixPoints = list()
    totalMissingDatePoints = list()
    totalMissingCoinVolumePoints = list()
    totalMissingUsdVolumePoints = list()
    totalMissingToken = list()
    counter = 0
    oneMinute = 60000

    for close in closePrices:
        counter += 1
        totalMissingUnixPoints.append(startUnix + (oneMinute * counter))
        totalMissingDatePoints.append(
            unixtimestamp_to_precise_datetime(startUnix + (oneMinute * counter))
        )
        totalMissingCoinVolumePoints.append(0)
        totalMissingUsdVolumePoints.append(0)  # todo improve
        totalMissingToken.append(token + "/USD")

    return pd.DataFrame(
        {
            "unix": totalMissingUnixPoints,
            "date": totalMissingDatePoints,
            "symbol": totalMissingToken,
            "close": closePrices,
            "high": closePrices,
            "open": closePrices,
            "low": closePrices,
            "Volume USD": totalMissingUsdVolumePoints,
            "Volume " + token: totalMissingCoinVolumePoints,
        }
    )


def get_historic_parquet_data(
    list_of_tickers, cols=["close"], root=None, start_time_unix=None, end_time_unix=None
):
    firstTicker = list_of_tickers[0]
    # print('cwd: ', os.getcwd())
    filename = firstTicker + "_USD.parquet"
    renamedCols = [col + "_" + firstTicker for col in cols]
    baseCols = [col for col in cols]
    if root is not None:
        inp_file = Path(root) / filename
    else:
        inp_file = impresources.files(data) / filename
    with inp_file.open("rb") as f:
        # path = root + firstTicker + "_USD.csv"
        csvData = pd.read_parquet(f, engine="pyarrow")
    csvData = csvData.filter(items=["unix"] + baseCols)
    csvData = csvData.rename(columns=dict(zip(baseCols, renamedCols)))
    if csvData.index.name != "unix":
        csvData = csvData.set_index("unix")
    if len(list_of_tickers) > 1:
        for ticker in list_of_tickers[1:]:
            renamedCols = [col + "_" + ticker for col in cols]
            baseCols = [col for col in cols]
            # path = root + ticker + "_USD.csv"
            filename = ticker + "_USD.parquet"
            if root is not None:
                inp_file = Path(root) / filename
            else:
                inp_file = impresources.files(data) / filename
            with inp_file.open("rb") as f:
                newCsvData = pd.read_parquet(f, engine="pyarrow").filter(
                    items=["unix"] + baseCols
                )
                newCsvData = newCsvData.rename(columns=dict(zip(baseCols, renamedCols)))
                if newCsvData.index.name != "unix":
                    newCsvData = newCsvData.set_index("unix")
            csvData = csvData.join(newCsvData)
    csvData = csvData.dropna()
    if start_time_unix is not None and end_time_unix is not None:
        csvData = csvData[start_time_unix - 1 : end_time_unix + 1]
        return csvData
    else:
        return csvData


def get_historic_csv_data(
    list_of_tickers, cols=["close"], root=None, start_time_unix=None, end_time_unix=None
):
    firstTicker = list_of_tickers[0]
    # print('cwd: ', os.getcwd())
    filename = firstTicker + "_USD.csv"
    if root is not None:
        inp_file = Path(root) / filename
    else:
        inp_file = impresources.files(data) / filename
    with inp_file.open("rt") as f:
        # path = root + firstTicker + "_USD.csv"
        csvData = pd.read_csv(
            f,
            dtype={
                "unix": float,
                "date": "string",
                "symbol": "string",
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "Volume USD": float,
                "Volume " + firstTicker: float,
                "tradecount": float,
            },
        ).filter(items=["unix"] + cols)
    rename_cols = {element: element + "_" + firstTicker for element in cols}
    csvData = csvData.rename(columns=rename_cols)
    csvData = csvData.set_index("unix")
    if len(list_of_tickers) > 1:
        for ticker in list_of_tickers[1:]:
            # path = root + ticker + "_USD.csv"
            filename = ticker + "_USD.csv"
            if root is not None:
                inp_file = Path(root) / filename
            else:
                inp_file = impresources.files(data) / filename
            with inp_file.open("rt") as f:
                newCsvData = pd.read_csv(
                    f,
                    dtype={
                        "unix": float,
                        "date": "string",
                        "symbol": "string",
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float,
                        "Volume USD": float,
                        "Volume " + ticker: float,
                        "tradecount": float,
                    },
                ).filter(items=["unix"] + cols)
            rename_cols = {element: element + "_" + ticker for element in cols}
            newCsvData = newCsvData.rename(columns=rename_cols)
            csvData = csvData.join(newCsvData.set_index("unix"))
    csvData = csvData.dropna()
    if start_time_unix is not None and end_time_unix is not None:
        return csvData[start_time_unix - 1 : end_time_unix + 1]
    else:
        return csvData


def get_stub_historic_close_csv_data(
    list_of_tickers, root, start_time_unix=None, end_time_unix=None
):
    # For testing
    n_assets = len(list_of_tickers)
    unix_timestamps = (np.arange(26340960.0, 28164960.0) * 60 * 1000).astype("int")
    prices = (
        np.arange(1.0, n_assets + 1.0)
        * 5
        * np.random.rand(len(unix_timestamps), n_assets)
    )
    df = pd.DataFrame(data=unix_timestamps, cols=["unix"])
    for i in range(n_assets):
        df[list_of_tickers[i]] = prices[:, i]
    return df


def get_historic_csv_data_w_versions(
    list_of_tickers, cols=["close"], root=None, max_verion=9
):
    tickers_data = []

    version = 0
    while version <= max_verion:
        local_list_of_tickers = [ticker + str(version) for ticker in list_of_tickers]
        tickers_data.append(get_historic_csv_data(local_list_of_tickers, cols, root))
        version += 1
    return {
        "prices": np.array(tickers_data),
        "unix_values": np.array([t_d.index.to_numpy() for t_d in tickers_data]),
    }


def load_market_cap_data(token: str, root: str = None) -> pd.DataFrame:
    """
    Load market cap data for a given token.

    Parameters
    ----------
    token : str
        Token symbol
    root : str, optional
        Root directory containing market cap data. If None, uses package data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: unix, price, market_cap, circulating_supply
    """
    if root is not None:
        inp_file = Path(root) / "market_cap_data" / f"{token.lower()}-usd-max.csv"
    else:
        inp_file = (
            impresources.files(data) / "market_cap_data" / f"{token.lower()}-usd-max.csv"
        )

    if not inp_file.exists():
        raise FileNotFoundError(f"No market cap data found for {token}")

    # Load market cap data
    df = pd.read_csv(inp_file, parse_dates=["snapped_at"])

    # Validate data
    required_columns = ["snapped_at", "price", "market_cap"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert UTC timestamp to unix milliseconds using existing utility
    df["unix"] = pddatetime_to_unixtimestamp(df["snapped_at"])

    # Calculate circulating supply with validation
    df["circulating_supply"] = np.where(
        (df["price"] > 1e-10) & (df["market_cap"] > 0),
        df["market_cap"] / df["price"],
        np.nan,
    )

    # Clip extreme values (3 sigma rule)
    valid_supplies = df["circulating_supply"].dropna()
    if len(valid_supplies) > 0:
        mean_supply = valid_supplies.mean()
        std_supply = valid_supplies.std()
        df["circulating_supply"] = df["circulating_supply"].clip(
            lower=mean_supply - 3 * std_supply, upper=mean_supply + 3 * std_supply
        )

    # Fill missing values with warnings
    max_fill_window = pd.Timedelta(days=7)

    # Forward fill with limit
    df["circulating_supply"] = df["circulating_supply"].fillna(
        method="ffill", limit=int(max_fill_window.total_seconds() / 86400)
    )

    # Backward fill with limit
    df["circulating_supply"] = df["circulating_supply"].fillna(
        method="bfill", limit=int(max_fill_window.total_seconds() / 86400)
    )

    return df[["unix", "circulating_supply"]]


def get_data_dict(
    list_of_tickers,
    run_fingerprint,
    data_kind="historic",
    root=None,
    max_memory_days=365.0,
    start_date_string=None,
    end_time_string=None,
    start_time_test_string=None,
    end_time_test_string=None,
    max_mc_version=9,
    return_slippage=False,
    return_gas_prices=False,
    return_supply=False,
    price_data=None,
    do_test_period=False,
):

    if return_slippage:
        price_data = None
        cols = ["high", "low", "open", "close", "Volume USD"]
    else:
        cols = ["close"]
    if return_gas_prices:
        # make sure we have ETH present
        if "ETH" not in list_of_tickers:
            list_of_tickers.append("ETH")
    list_of_tickers.sort()

    chunk_period = run_fingerprint["chunk_period"]

    startDate = (
        datetime_to_unixtimestamp(start_date_string, str_format="%Y-%m-%d %H:%M:%S")
        * 1000
    )
    endDate = (
        datetime_to_unixtimestamp(end_time_string, str_format="%Y-%m-%d %H:%M:%S")
        * 1000
    )

    if data_kind == "historic":
        if price_data is None:
            price_data = get_historic_parquet_data(list_of_tickers, cols, root)
        unix_values = price_data.index.to_numpy()
        prices = price_data.filter(
            items=["close_" + ticker for ticker in list_of_tickers]
        ).to_numpy()
        # if return_slippage:
        #     spread = np.array(
        #         [
        #             edge(
        #                 open=price_data["open_" + ticker],
        #                 high=price_data["high_" + ticker],
        #                 low=price_data["low_" + ticker],
        #                 close=price_data["close_" + ticker],
        #                 sign=False,
        #             )
        #             for ticker in list_of_tickers
        #         ]
        #     ).clip(min=0.0)
        #     # set spread of USD asset to 0
        #     idx = list_of_tickers.index("DAI")
        #     spread[idx] = 0.0
    elif data_kind == "mc":
        if price_data is None:
            mc_tokens = [
                value for value in list_of_tickers if value in mc_data_available_for
            ]
            non_mc_tokens = [
                value for value in list_of_tickers if value not in mc_data_available_for
            ]
            price_data_mc = get_historic_csv_data_w_versions(
                mc_tokens, ["close"], root, max_verion=max_mc_version
            )
            price_data_non_mc = get_historic_parquet_data(non_mc_tokens, cols, root)
        else:
            price_data_mc, price_data_non_mc = price_data
        if len(non_mc_tokens) > 0:
            non_mc_tokens_idx = [
                True if value not in mc_data_available_for else False
                for value in list_of_tickers
            ]
            mc_tokens_idx = [
                True if value in mc_data_available_for else False
                for value in list_of_tickers
            ]
            # trim by unix timestamp to the available range
            latest_start_timestamp = max(
                price_data_non_mc.index[0], *price_data_mc["unix_values"][:, 0]
            )
            earliest_end_timestamp = min(
                price_data_non_mc.index[-1], *price_data_mc["unix_values"][:, -1]
            )
            price_data_non_mc = price_data_non_mc.iloc[
                price_data_non_mc.index >= latest_start_timestamp
            ]
            price_data_non_mc = price_data_non_mc.iloc[
                price_data_non_mc.index <= earliest_end_timestamp
            ]
            MC_idx = (price_data_mc["unix_values"][0] >= latest_start_timestamp) * (
                price_data_mc["unix_values"][0] <= earliest_end_timestamp
            )
            price_data_mc["prices"] = price_data_mc["prices"][:, MC_idx]
            price_data_mc["unix_values"] = price_data_mc["unix_values"][0, MC_idx]
            # duplicate up non-MC data to the number of versions of MC data present
            # this is NOT efficient but it works
            prices = np.empty(
                (
                    price_data["prices"].shape[1],
                    len(list_of_tickers),
                    max_mc_version + 1,
                )
            )
            prices[:, non_mc_tokens_idx, :] = price_data_non_mc.to_numpy()[
                ..., np.newaxis
            ]
            prices[:, mc_tokens_idx, :] = np.moveaxis(price_data_mc["prices"], 0, 2)
            unix_values = price_data_mc["unix_values"]
        else:
            prices = price_data_mc["prices"]
            unix_values = price_data_mc["unix_values"][0]
    else:
        raise NotImplementedError
    (
        start_idx,
        end_idx,
        bout_length,
        unix_values_rebased,
        prices_rebased,
        oracle_values_rebased,
        remainder_idx,
    ) = start_and_end_calcs(
        unix_values,
        prices=prices,
        start_date=startDate,
        end_date=endDate,
    )

    if start_idx / 1440 < max_memory_days:
        max_memory_days = start_idx / 1440 - 1.0

    # n_chunks = (len(prices) - remainder_idx) / chunk_period
    n_chunks = int((len(prices) - remainder_idx) / 1440) * 1440 / chunk_period
    # check that we can cleanly divide data into 'chunk_period' units
    # if not, we will remove off the last little bit of the dataset.
    # (note that this doesn't interefere with the above burnin manipulations
    # as we made sure to 'add' a chunk-divisible portion to the start)
    if return_slippage:
        n_chunks = int((len(prices) - remainder_idx) / 1440) * 1440 / chunk_period
        if chunk_period < 1440 or chunk_period % 1440 != 0:
            price_data_filtered = price_data.iloc[
                remainder_idx : int(n_chunks) * chunk_period + remainder_idx
            ]
        else:
            # Calculate daily n_chunks by dividing total minutes by minutes per day
            n_chunks = int((len(prices) - remainder_idx) / 1440)
            price_data_filtered = price_data.iloc[
                remainder_idx : int(n_chunks) * chunk_period + remainder_idx
            ]
        spread = list()
        annualised_daily_volatility = list()
        daily_volume = list()
        for ticker in list_of_tickers:
            per_ticker_spread = list()
            open_ = np.array(price_data_filtered["open_" + ticker]).reshape(-1, 1440)
            high_ = np.array(price_data_filtered["high_" + ticker]).reshape(-1, 1440)
            low_ = np.array(price_data_filtered["low_" + ticker]).reshape(-1, 1440)
            close_ = np.array(price_data_filtered["close_" + ticker]).reshape(-1, 1440)
            for i in range(len(open_)):
                per_ticker_spread.append(
                    edge(
                        open=open_[i],
                        high=high_[i],
                        low=low_[i],
                        close=close_[i],
                        sign=False,
                    )
                )
                if np.isnan(per_ticker_spread[-1]):
                    per_ticker_spread[-1] = 0.0
            spread.append(np.array(per_ticker_spread))
            per_ticker_annualised_daily_volatility = np.array(
                calculate_annualised_daily_volatility_from_minute_data(
                    price_data_filtered.copy(), ticker
                )
            )
            # per_ticker_daily_volume = calculate_daily_volume_from_minute_data(
            #     price_data_filtered, ticker
            # )
            annualised_daily_volatility.append(per_ticker_annualised_daily_volatility)
            # daily_volume.append(per_ticker_daily_volume)
            daily_OHLC_data = resample_minute_level_OHLC_data_to_daily(
                price_data_filtered.copy(), ticker
            )
            daily_volume.append(np.array(daily_OHLC_data["Volume USD_" + ticker]))
        spread = np.repeat(np.array(spread).T, 1440, axis=0)
        daily_volume = np.repeat(np.array(daily_volume).T, 1440, axis=0)
        annualised_daily_volatility = np.repeat(
            np.array(annualised_daily_volatility).T, 1440, axis=0
        )
        # set spread of USD asset to 0
        # idx = list_of_tickers.index("DAI")
        # spread[idx, :] = 0.0
    # if return_slippage:
    # spread_rebased = spread[remainder_idx:]
    if return_supply:
        print("Loading market cap data for supply calculation")
        supply_data = []
        for token in list_of_tickers:
            token_supply = load_market_cap_data(token, root)
            token_supply.set_index("unix", inplace=True)

            # Align with price timeline using efficient reindexing
            price_timeline = pd.Index(unix_values)
            aligned_supply = token_supply.reindex(
                price_timeline,
                method="ffill",
                limit=int(
                    pd.Timedelta(days=7).total_seconds() / 60
                ),  # 7 day limit in minutes
            )

            if aligned_supply["circulating_supply"].isna().any():
                print(
                    f"Warning: Missing supply data for {token} after forward-filling"
                )
                # Use last valid value or 1.0 if no valid data
                last_valid = (
                    aligned_supply["circulating_supply"].dropna().iloc[-1]
                    if not aligned_supply["circulating_supply"].isna().all()
                    else 1.0
                )
                aligned_supply["circulating_supply"].fillna(
                    last_valid, inplace=True
                )
                print(f"Using placeholder supply value of {last_valid} for {token}")

            supply_data.append(aligned_supply["circulating_supply"].values)

    prices_rebased = prices_rebased[: round(n_chunks * chunk_period)]
    unix_values_rebased = unix_values_rebased[: round(n_chunks * chunk_period)]

    # if return_slippage:
    #     spread_rebased = spread[: int(n_chunks) * chunk_period]
    if return_gas_prices:
        if root is not None:
            inp_file = Path(root) / "export-AvgGasPrice.csv"
        else:
            inp_file = impresources.files(data) / "export-AvgGasPrice.csv"
        with inp_file.open("rt") as f:
            # path = root + firstTicker + "_USD.csv"
            gas_prices = (
                pd.read_csv(f)
                .filter(items=["UnixTimeStamp", "Value (Wei)"])
                .rename(columns={"UnixTimeStamp": "unix", "Value (Wei)": "gas_cost"})
            )
        eth_price = prices_rebased[:, list_of_tickers.index("ETH")]
        # gas_price is daily, expland to minute
        gas_prices_minute = expand_daily_to_minute_data(gas_prices, scale="s")
        eth_price = pd.dataframe([unix_values_rebased, eth_price]).join(
            gas_prices_minute.set_index("unix")
        )
        gas_prices_in_dollars = eth_price["close_ETH"] * eth_price["gas_cost"]
    data_dict = {
        "prices": prices_rebased,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "bout_length": bout_length,
        "unix_values": unix_values_rebased,
        "n_chunks": n_chunks,
    }
    data_dict["max_memory_days"] = max_memory_days
    if return_slippage:
        data_dict["spread"] = spread
        data_dict["daily_volume"] = daily_volume
        data_dict["annualised_daily_volatility"] = annualised_daily_volatility
    if return_gas_prices:
        data_dict["gas_prices_in_dollars"] = gas_prices_in_dollars
    if return_supply:
        data_dict["supply"] = np.array(supply_data).T
    if start_time_test_string is not None and end_time_test_string is not None:
        startDateTest = (
            datetime_to_unixtimestamp(
                start_time_test_string, str_format="%Y-%m-%d %H:%M:%S"
            )
            * 1000
        )
        endDateTest = (
            datetime_to_unixtimestamp(
                end_time_test_string, str_format="%Y-%m-%d %H:%M:%S"
            )
            * 1000
        )

        (
            start_idx_test,
            end_idx_test,
            bout_length_test,
            unix_values_test,
            price_values_test,
            oracle_values_test,
            remainder_idx_test,
        ) = start_and_end_calcs(
            unix_values,
            prices=prices,
            start_date=startDateTest,
            end_date=endDateTest,
        )

        data_dict["prices_test"] = price_values_test
        data_dict["start_idx_test"] = start_idx_test
        data_dict["end_idx_test"] = end_idx_test
        data_dict["bout_length_test"] = bout_length_test
        data_dict["unix_values_test"] = unix_values_test
        if return_slippage:
            spread_test = spread[remainder_idx_test:]
            spread_test = spread_test[: int(n_chunks) * chunk_period]
            data_dict["spread_test"] = spread_test
    return data_dict


def get_historic_daily_csv_data(list_of_tickers, root):
    firstTicker = list_of_tickers[0]
    print("cwd: ", os.getcwd())
    path = root + firstTicker + "_USD_daily.csv"
    csvData = pd.read_csv(
        path,
        dtype={
            "unix": float,
            "date": "string",
            "symbol": "string",
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "Volume USD": float,
            "Volume " + firstTicker: float,
            "tradecount": float,
        },
    )

    if len(list_of_tickers) > 1:
        csvData = csvData.rename(columns={"close": firstTicker})
        for ticker in list_of_tickers[1:]:
            path = root + ticker + "_USD.csv"
            newCsvData = pd.read_csv(
                path,
                dtype={
                    "unix": float,
                    "date": "string",
                    "symbol": "string",
                    "open": float,
                    "high": float,
                    "low": float,
                    "close": float,
                    "Volume USD": float,
                    "Volume " + ticker: float,
                    "tradecount": float,
                },
            ).filter(items=["unix", "close"])
            newCsvData = newCsvData.rename(columns={"close": ticker})
    return csvData


def get_coin_comparison_data(root):
    path = root + "coinComparisonData.csv"
    csvData = pd.read_csv(
        path, dtype={"coinPair": "string", "covariance": float, "trackingError": float}
    )
    return csvData


def get_historic_data(
    list_of_tokens,
    start_time,
    end_time=None,
    period=60,
    numeraire="USD",
    transform=None,
):
    """Historic Crypto dataset, using 'Historic_Crypto' package to download."""

    """
    Args:
        list_of_tokens (list): List of tokens to include in data
        start_time (str, YYYY-MM-DD-HH-MM format): The time to start at
        end time (str, YYYY-MM-DD-HH-MM format, optional): The time to end at
                    if None (default value), defaults to current time
        period (int, optional): the time period to sample at, in seconds
                    must be one of: 60, 300, 900, 3600, 21600, 86400
        numeraire: (str, optional): the currency or token to use as the numeraire
        transform (callable, optional): Optional transform to be applied
            on a sample.

    Returns
    -------
    list of numpy arrays containing historic data

    """

    list_of_tickers = [t + "-" + numeraire for t in list_of_tokens]

    # check that all tickers are in available data
    available_tickers = Cryptocurrencies().find_crypto_pairs()["id"]
    if set(list_of_tickers).issubset(available_tickers) is False:
        raise Exception

    # Download ticker data, keeping close at end time step
    data = [
        HistoricalData(ticker, period, start_time, end_time).retrieve_data()["close"]
        for ticker in list_of_tickers
    ]

    start_time_as_unix_timestamp = datetime_to_unixtimestamp(
        start_time, str_format="%Y-%m-%d-%H-%M"
    )
    end_time_as_unix_timestamp = datetime_to_unixtimestamp(
        end_time, str_format="%Y-%m-%d-%H-%M"
    )

    cleaned_data = []
    raise_exception = 0
    for d in data:
        # make array of times as timestamps
        unix_timestamps = pddatetime_to_unixtimestamp(d.index)
        price_data = np.array(d)
        cleaned_data.append(
            _cleaned_up_coinbase_data(
                unix_timestamps,
                price_data,
                start_time_as_unix_timestamp,
                end_time_as_unix_timestamp,
                period=period,
            )
        )
        raise_exception = 1

    return np.array(cleaned_data).T
