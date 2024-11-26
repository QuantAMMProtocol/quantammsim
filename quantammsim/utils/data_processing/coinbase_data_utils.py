import numpy as np
import pandas as pd
from Historic_Crypto import HistoricalData
from quantammsim.utils.data_processing.datetime_utils import datetime_to_unixtimestamp, pddatetime_to_unixtimestamp, unixtimestamp_to_precise_datetime

def import_historic_coinbase_data(
    token,
    start_time,
    end_time=None,
    interpolate=False,
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
    print("importing data " + start_time + "   " + end_time)
    # Download ticker data, keeping close at end time step
    retrievedData = pd.DataFrame()
    try:
        retrievedData = HistoricalData(
            token + "-USD", period, start_time, end_time
        ).retrieve_data()
    except Exception as e:
        print(e)
        retrievedData = HistoricalData(
            token + "-USDT", period, start_time, end_time
        ).retrieve_data()
        print("retrieved USDT gap from coinbase")
        # TODO CH convert to USD
    start_time_as_unix_timestamp = datetime_to_unixtimestamp(
        start_time, str_format="%Y-%m-%d-%H-%M"
    )
    end_time_as_unix_timestamp = datetime_to_unixtimestamp(
        end_time, str_format="%Y-%m-%d-%H-%M"
    )

    prevRow = retrievedData.iloc[0]
    prevIndex = start_time_as_unix_timestamp * 1000
    notFirstRow = False
    totalMissingUnixPoints = list()
    totalMissingDatePoints = list()
    totalMissingClosePoints = list()
    totalMissingCoinVolumePoints = list()
    totalMissingUsdVolumePoints = list()
    totalMissingToken = list()
    consequtiveMissing = 1
    oneMinute = 60000

    for index, row in retrievedData.iterrows():
        rowMilliUnixIndex = index.value / 1000000
        nextUnix = prevIndex + (consequtiveMissing * oneMinute)
        if notFirstRow and nextUnix != rowMilliUnixIndex:
            tightLoopBreak = 10000000
            missingUnixPoints = list()
            missingDatePoints = list()
            missingClosePoints = list()
            missingCoinVolumePoints = list()
            missingUsdVolumePoints = list()
            missingToken = list()

            while nextUnix < rowMilliUnixIndex and consequtiveMissing < tightLoopBreak:
                missingUnixPoints.append(int(nextUnix))
                missingClosePoints.append(prevRow.close)
                missingDatePoints.append(unixtimestamp_to_precise_datetime(nextUnix))
                missingCoinVolumePoints.append(int(0))
                missingUsdVolumePoints.append(int(0))
                missingToken.append(token + "/USD")
                consequtiveMissing += 1
                nextUnix = prevIndex + (consequtiveMissing * oneMinute)

            if tightLoopBreak > 10000000:
                print("ERROR TIGHT LOOP")

            totalMissingUnixPoints = totalMissingUnixPoints + missingUnixPoints
            totalMissingDatePoints = totalMissingDatePoints + missingDatePoints
            totalMissingClosePoints = totalMissingClosePoints + missingClosePoints
            totalMissingCoinVolumePoints = (
                totalMissingCoinVolumePoints + missingCoinVolumePoints
            )
            totalMissingUsdVolumePoints = (
                totalMissingUsdVolumePoints + missingUsdVolumePoints
            )
            totalMissingToken = totalMissingToken + missingToken
            prevRow = row
            prevIndex = rowMilliUnixIndex
            consequtiveMissing = 1
        else:
            consequtiveMissing = 1
            prevRow = row
            prevIndex = rowMilliUnixIndex
        notFirstRow = True

    if len(totalMissingUnixPoints) != len(totalMissingClosePoints):
        print("different lengths")
    if len(totalMissingUnixPoints) > 0:
        missingDf = pd.DataFrame(
            {
                "unix": totalMissingUnixPoints,
                "date": totalMissingDatePoints,
                "symbol": totalMissingToken,
                "close": totalMissingClosePoints,
                "high": totalMissingClosePoints,
                "open": totalMissingClosePoints,
                "low": totalMissingClosePoints,
                "Volume USD": totalMissingUsdVolumePoints,
                "Volume " + token: totalMissingCoinVolumePoints,
            }
        )

        retrievedData = retrievedData.append(missingDf)
        retrievedData = clean_up_coinbase_df(retrievedData)
    return retrievedData


def clean_up_coinbase_df(df):
    clean_data = df.loc[:0].iloc[:-1]
    clean_data = clean_data[["low", "high", "open", "close", "volume"]]
    clean_datetime = pd.Series([pd.Timestamp(d) for d in clean_data.index], name='date')
    clean_data.index = clean_datetime
    unsorted_data = df.loc[0:]
    unsorted_data_timestamps = pd.Series([pd.Timestamp(d) for d in unsorted_data["date"]])
    unsorted_data["date"] = unsorted_data_timestamps
    # unsorted_data.assign(date=unsorted_data_timestamps)
    unsorted_data=unsorted_data.set_index("date")
    unsorted_data_filtered = unsorted_data[["low", "high", "open", "close", "Volume USD"]].rename(columns={"Volume USD": "volume"})
    merged_data=pd.concat([clean_data,unsorted_data_filtered])
    merged_data=merged_data.sort_index()
    merged_data["unix"] = pddatetime_to_unixtimestamp(merged_data.index) * 1000
    return merged_data


def import_clean_historic_coinbase_dataframe(
    token,
    start_time,
    end_time=None,
    interpolate=False,
    period=60,
    numeraire="USD",
    transform=None,
):
    # Download ticker data, keeping close at end time step
    data = [
        HistoricalData(token, period, start_time, end_time).retrieve_data()["close"]
    ]

    start_time_as_unix_timestamp = datetime_to_unixtimestamp(
        start_time, str_format="%Y-%m-%d-%H-%M"
    )
    end_time_as_unix_timestamp = datetime_to_unixtimestamp(
        end_time, str_format="%Y-%m-%d-%H-%M"
    )

    if not interpolate:
        return np.array(data[0])

    cleaned_data = []
    unix_timestamps = []
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

    return unix_timestamps, np.array(cleaned_data).T


## data loader
# @jit((int64[:], float64[:], int64, int64, int64), nopython=True, nogil=True)
def _cleaned_up_coinbase_data(
    coinbase_times_as_unix_timestamps, coinbase_data, start_time, end_time, period=60000
):
    # function to clean up coinbase data
    # coinbase data has gaps in it. If a token
    # isnt traded in the given window (the given minute, say)
    # then no row of data is returned
    # we can assume that if no trade took place then the previous
    # close remains the current price

    # make array

    n_time_steps = (end_time - start_time) / period

    if n_time_steps != float(int(n_time_steps)):
        print(n_time_steps, float(int(n_time_steps)))
        raise Exception
    print("1")
    cleaned_prices = np.empty((int(n_time_steps),), dtype=np.float64)

    # first check that the first timestamp matches the start time,
    # and if not project backwards from the first time
    print(coinbase_times_as_unix_timestamps[0])
    if coinbase_times_as_unix_timestamps[0] != start_time:
        # number of steps to 'backfill' is difference in time
        # divided by the period
        initial_gap_in_seconds = coinbase_times_as_unix_timestamps[0] - start_time
        initial_gap_in_rows = initial_gap_in_seconds / period
        cleaned_prices[:initial_gap_in_rows] = coinbase_data[0]
        have_backfilled = True
    else:
        have_backfilled = False
    running_time = start_time

    raw_data_index = 0

    for i in range(n_time_steps):
        # try to match by time index
        if running_time == coinbase_times_as_unix_timestamps[raw_data_index]:
            cleaned_prices[i] = coinbase_data[raw_data_index]
            raw_data_index += 1
            print("correct")
        else:
            if i < initial_gap_in_rows and have_backfilled == True:
                pass
            else:
                cleaned_prices[i] = cleaned_prices[i - 1]
                # print("backfilled")
                # print(running_time)
        running_time += period
    print("3")
    return cleaned_prices

import os

def fill_missing_rows_with_coinbase_data(concatenated_df, token1, root):

    file_path = root + 'coinbase_data/' + token1 + '_cb_sorted_.csv'
    if not os.path.exists(file_path):
        return concatenated_df, []
    # Load Coinbase data
    coinbase_data = pd.read_csv(file_path)
    # Set the 'Unix' column as the index for Coinbase data
    coinbase_data.set_index('unix', inplace=True)

    # Identify missing timestamps in the concatenated DataFrame

    coinbase_data["symbol"] = concatenated_df.iloc[0]["symbol"]

    missing_timestamps = coinbase_data.index.difference(concatenated_df.index)
    filled_in_df = pd.concat([concatenated_df, coinbase_data.loc[missing_timestamps]])
    # Sort by 'Unix' index after filling missing rows
    filled_in_df.sort_index(inplace=True)
    filled_unix_values = missing_timestamps.tolist()
    # Drop duplicate indexes, the first are from the concatenated data, the second are from the Coinbase data
    filled_in_df = filled_in_df[~filled_in_df.index.duplicated(keep='first')]
    filled_unix_values = filled_in_df.index.tolist()
    return filled_in_df, filled_unix_values



