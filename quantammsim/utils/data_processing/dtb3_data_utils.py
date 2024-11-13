import pandas as pd
from importlib import resources as impresources
from quantammsim import data
from pathlib import Path


def filter_dtb3_values(filename, start_date, end_date, target_directory=None):
    # Load the CSV file
    if target_directory is not None:
        inp_file = Path(target_directory) / filename
    else:
        inp_file = impresources.files(data) / filename
    with inp_file.open("rt") as f:
        # path = root + firstTicker + "_USD.csv"
        df = pd.read_csv(f)

    # Convert DATE to datetime format
    df["DATE"] = pd.to_datetime(df["DATE"])
    # Strip time component from start and end date

    start_date_datetime = pd.to_datetime(start_date).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_date_datetime = pd.to_datetime(end_date).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Add rows for any missing days
    date_range = pd.date_range(start=start_date_datetime, end=end_date_datetime)

    missing_dates = date_range[~date_range.isin(df["DATE"])]
    missing_rows = pd.DataFrame({"DATE": missing_dates, "DTB3": float("nan")})

    df = pd.concat([df, missing_rows]).sort_values("DATE").reset_index(drop=True)

    # Add a Unix timestamp column
    df["UNIX_TIMESTAMP"] = df["DATE"].astype(int) // 10**9

    # Filter the DataFrame based on the start and end date Unix timestamps
    mask = (df["UNIX_TIMESTAMP"] >= start_date_datetime.timestamp()) & (
        df["UNIX_TIMESTAMP"] <= end_date_datetime.timestamp()
    )
    filtered_df = df.loc[mask]

    # Save the DataFrame to a CSV file
    filtered_df.to_csv("filtered_debug_rf.csv", index=False)

    # Fill blank date rows with previous "DTB3" value
    filtered_df["DTB3"].replace(".", float("nan"), inplace=True)
    filtered_df["DTB3"].fillna(method="ffill", inplace=True)

    # Convert DTB3 column to numpy array
    dtb3_values = filtered_df["DTB3"].astype(float).to_numpy()

    # Convert percentage values in DTB3 column to decimals
    dtb3_values = dtb3_values / 100.0

    return dtb3_values
