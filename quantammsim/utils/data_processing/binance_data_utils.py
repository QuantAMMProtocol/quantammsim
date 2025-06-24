import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def concat_csv_files(root, save_root, token1, token2, prefix, postfix, years_array_str):
    print("concatenating files")
    filenames = []
    for year in years_array_str:
        filenames = filenames + glob.glob(
            root + prefix + token1 + token2 + "_" + year + postfix + "*.csv"
        )
    dataframes = []
    # Create save directory if it doesn't exist
    os.makedirs(save_root, exist_ok=True)

    for filename in filenames:
        file_path = filename
        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue

        # Read the first line to check if it contains the unwanted string
        with open(file_path, "r") as file:
            first_line = file.readline().strip()

        # Load the file into a DataFrame, skipping the first line if necessary
        if first_line == "https://www.CryptoDataDownload.com":
            df = pd.read_csv(file_path, skiprows=1)
        else:
            df = pd.read_csv(file_path)

        # Ensure the columns are as expected
        expected_columns_variations = [
            {
                "base_cols": [
                    "unix",
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "volume_from",
                    "marketorder_volume",
                    "marketorder_volume_from",
                    "tradecount",
                    "date_close",
                    "close_unix",
                ],
                "rename_cols": {
                    "volume": "Volume " + token1,
                    "volume_from": "Volume USD",
                },
            },
            {
                "base_cols": [
                    "unix",
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "volume_from",
                    "tradecount",
                ],
                "rename_cols": {
                    "volume": "Volume " + token1,
                    "volume_from": "Volume USD",
                },
            },
            {
                "base_cols": [
                    "unix",
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    f"Volume {token1}",
                    f"Volume {token2}",
                    "tradecount",
                ],
                "rename_cols": {
                    f"Volume {token2}": "Volume USD",
                },
            },
            {
                "base_cols": [
                    "Unix",
                    "Date",
                    "Symbol",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    f"Volume {token1}",
                    f"Volume {token2}",
                    "tradecount",
                ],
                "rename_cols": {
                    "Unix": "unix",
                    "Date": "date",
                    "Symbol": "symbol",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    f"Volume {token2}": "Volume USD",
                },
            },
            {
                "base_cols": [
                    "unix",
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    f"Volume {token1}",
                    f"Volume {token2}",
                ],
                "rename_cols": {
                    "Unix": "unix",
                    "Date": "date",
                    "Symbol": "symbol",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    f"Volume {token2}": "Volume USD",
                },
            },
        ]

        reduced_columns = [
            "unix",
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "Volume USD",
            "Volume " + token1,
            # "tradecount",
        ]
        df_cols_list = df.columns.tolist()
        base_columns = [ecv["base_cols"] for ecv in expected_columns_variations]
        rename_columns = [ecv["rename_cols"] for ecv in expected_columns_variations]
        if df_cols_list in base_columns:
            cols_index = base_columns.index(df_cols_list)
            df = df.rename(columns=rename_columns[cols_index])
            df = df.filter(reduced_columns)
        else:
            raise ValueError(f"Unexpected columns in file {filename}")


        # Ensure unix timestamps are in milliseconds
        if len(str(int(df["unix"].max()))) <= 10:  # If timestamps are in seconds
            df["unix"] = df["unix"] * 1000
        # Set the 'Unix' column as the index
        df.set_index("unix", inplace=True)
        df = df[::-1]
        df.sort_index(inplace=True)

        # Append the DataFrame to the list
        dataframes.append(df)
    # Concatenate all DataFrames
    concatenated_df = pd.concat(dataframes)

    # Sort by 'Unix' index
    concatenated_df.sort_index(inplace=True)
    # Convert index to int if it's not already
    if not np.issubdtype(concatenated_df.index.dtype, np.integer):
        concatenated_df.index = concatenated_df.index.astype(int)

    report_gaps(concatenated_df, save_root + token1 + "_" + token2 + "_gaps.csv")
    # Write the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(save_root + token1 + "_" + token2 + ".csv")
    # Skip the unix column if it exists
    plot_exchange_data(concatenated_df, token1, save_root + token1 + "_" + token2 + "_data.png")
    
    return concatenated_df


def report_gaps(concatenated_df, gaps_output_file=None):

    # Create a DataFrame with a continuous range of Unix timestamps
    start_unix = concatenated_df.index.min()
    end_unix = concatenated_df.index.max()
    full_index = (
        pd.date_range(
            start=pd.to_datetime(start_unix, unit="ms"),
            end=pd.to_datetime(end_unix, unit="ms"),
            freq="T",
        ).astype(int)
        // 10**9
    )
    full_index_df = pd.DataFrame(index=full_index)

    # Identify missing timestamps
    missing_timestamps = full_index_df.index.difference(concatenated_df.index)
    # raise Exception
    # Find gaps
    gaps = []
    if not missing_timestamps.empty:
        # Find gaps in the data
        time_diffs = np.diff(concatenated_df.index)
        expected_diff = 60000  # 60 seconds in milliseconds

        # Find positions where there are gaps (timestamps more than 60s apart)
        gap_positions = np.where(time_diffs > expected_diff)[0]

        # For each gap position, record:
        # 1. The timestamp where the gap starts (the index after the gap_position)
        # 2. How many minutes are missing
        gaps = []
        for pos in gap_positions:
            gap_start = concatenated_df.index[pos + 1]  # First missing timestamp
            gap_duration = (
                time_diffs[pos] // expected_diff
            ) - 1  # Number of missing minutes
            gaps.append((gap_start, gap_duration))
    # Add datetime for readability
    gaps_with_datetime = []
    for gap_start, gap_duration in gaps:
        datetime_str = pd.to_datetime(gap_start, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
        gaps_with_datetime.append(
            (
                gap_start,
                datetime_str,
                gap_duration,
            )
        )

    # Write gaps to CSV
    gaps_df = pd.DataFrame(
        gaps_with_datetime, columns=["Gap Start Unix", "Gap Start Datetime", "Gap Duration (minutes)"]
    )
    if gaps_output_file is not None:
        gaps_df.to_csv(gaps_output_file, index=False)
    return gaps_df


def get_gaps(df, gaps_df):
    gap_data = []
    for i in range(len(gaps_df)):
        start_idx = gaps_df["Gap Start Unix"].iloc[i]
        end_idx = (
            gaps_df["Gap Start Unix"].iloc[i]
            + gaps_df["Gap Duration (minutes)"].iloc[i] * 60
        )
        gap = df.loc[start_idx:end_idx]
        gap_data.append(gap)
    return gap_data


def get_gap_ratio(gaps1, gaps2):
    assert len(gaps1) == len(gaps2)
    out = []
    for i in range(len(gaps1)):
        if len(gaps1[i]) > 0 and len(gaps2[i]) > 0:
            assert len(gaps1[i]) == len(gaps2[i])
            out.append(np.array(gaps2[i]) / np.array(gaps1[i]) - 1.0)
    return out

def plot_exchange_data(csvData, token, output_path):
    """
    Creates a multi-subplot figure showing exchange data for a token, with gaps highlighted.

    Args:
        csvData (pd.DataFrame): DataFrame containing exchange data
        token (str): Token symbol being plotted
        output_path (str): Path where the output plot should be saved

    Returns:
        bool: True if plot was successfully created and saved
    """
    if csvData.empty:
        print(f"No data to plot for {token}")
        return False

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    columns_to_plot = [col for col in csvData.columns if col not in ["date", "symbol", "datetime"]]
    if not columns_to_plot:
        print(f"No numeric columns found to plot for {token}")
        return False

    num_cols = len(columns_to_plot)
    fig, axes = plt.subplots(num_cols, 1, figsize=(15, 4 * num_cols))
    fig.suptitle(f"Exchange Data for {token}")

    # Sample unix timestamps for x-axis points
    sample_rate = 1  # Sample every point
    x_points = csvData.index[::sample_rate].astype(np.float64)

    # Print raw data folder and date range
    start_date = pd.to_datetime(x_points[0], unit="ms").strftime("%Y-%m-%d %H:%M:%S")
    end_date = pd.to_datetime(x_points[-1], unit="ms").strftime("%Y-%m-%d %H:%M:%S")
    print(f"Date range: {start_date} to {end_date}")

    # Find gaps in the data
    time_diffs = np.diff(x_points)
    expected_diff = 60000  # 60 seconds in milliseconds
    gap_positions = np.where(time_diffs > expected_diff)[0]

    # Handle case where there's only one subplot
    if num_cols == 1:
        axes = [axes]

    # Plot each column in a separate subplot
    for i, column in enumerate(columns_to_plot):
        data = csvData[column].iloc[::sample_rate].astype(np.float64)

        # Shade the gaps first (so points appear on top)
        for gap_pos in gap_positions:
            gap_start = x_points[gap_pos]
            gap_end = x_points[gap_pos + 1]
            axes[i].axvspan(gap_start, gap_end, color="red", alpha=0.1)

        axes[i].scatter(x_points, data, s=0.1, alpha=0.5, color="blue")
        axes[i].set_title(column)
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    plt.clf()
    return True