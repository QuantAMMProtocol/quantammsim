import numpy as np
import pandas as pd
import glob
import os

def concat_csv_files(root, save_root, token1, token2, prefix, postfix, years_array_str):
    print("concatenating files")
    filenames = []
    for year in years_array_str:
        filenames = filenames + glob.glob(root + prefix + token1 + token2 + "_" + year + postfix + "*.csv")
    dataframes = []

    for filename in filenames:
        file_path = filename
        if not os.path.isfile(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue

        # Read the first line to check if it contains the unwanted string
        with open(file_path, 'r') as file:
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
            df=df.rename(columns=rename_columns[cols_index])
            df=df.filter(reduced_columns)
        else:
            raise ValueError(f"Unexpected columns in file {filename}")

        # Set the 'Unix' column as the index
        df.set_index('unix', inplace=True)
        df = df[::-1]
        df.sort_index(inplace=True)

        # Append the DataFrame to the list
        dataframes.append(df)

    # Concatenate all DataFrames
    concatenated_df = pd.concat(dataframes)

    # Sort by 'Unix' index
    concatenated_df.sort_index(inplace=True)

    return concatenated_df


def report_gaps(concatenated_df, gaps_output_file=None):
        
    # Create a DataFrame with a continuous range of Unix timestamps
    start_unix = concatenated_df.index.min()
    end_unix = concatenated_df.index.max()
    full_index = pd.date_range(start=pd.to_datetime(start_unix, unit='ms'), end=pd.to_datetime(end_unix, unit='ms'), freq='T').astype(int) // 10**9
    full_index_df = pd.DataFrame(index=full_index)

    # Identify missing timestamps
    missing_timestamps = full_index_df.index.difference(concatenated_df.index)
    # raise Exception
    # Find gaps
    gaps = []
    if not missing_timestamps.empty:
        gap_start = missing_timestamps[0]
        gap_length = 1
        for current, following in zip(missing_timestamps, missing_timestamps[1:]):
            if following == current + 60:  # Next minute timestamp
                gap_length += 1
            else:
                gaps.append((gap_start, gap_length))
                gap_start = following
                gap_length = 1
        gaps.append((gap_start, gap_length))  # Add the last gap
    # Write gaps to CSV
    gaps_df = pd.DataFrame(gaps, columns=['Gap Start Unix', 'Gap Duration (minutes)'])
    if gaps_output_file is not None:
        gaps_df.to_csv(gaps_output_file, index=False)
    return gaps_df


def get_gaps(df, gaps_df):
    gap_data = []
    for i in range(len(gaps_df)):
        start_idx = gaps_df["Gap Start Unix"].iloc[i]
        end_idx = gaps_df["Gap Start Unix"].iloc[i] + gaps_df["Gap Duration (minutes)"].iloc[i] * 60
        gap = df.loc[start_idx:end_idx]
        gap_data.append(gap)
    return gap_data

def get_gap_ratio(gaps1, gaps2):
    assert len(gaps1) == len(gaps2)
    out = []
    for i in range(len(gaps1)):
        if len(gaps1[i]) >0 and len(gaps2[i]) >0:
            assert len(gaps1[i]) == len(gaps2[i])
            out.append(np.array(gaps2[i]) / np.array(gaps1[i]) - 1.0)
    return out

