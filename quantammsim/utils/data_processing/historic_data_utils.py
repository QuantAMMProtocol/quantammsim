import numpy as np
import pandas as pd
import os.path
import os

# from numba import jit
# from numba import float64
# from numba import int64
from Historic_Crypto import Cryptocurrencies, HistoricalData
from datetime import datetime, timezone
from importlib import resources as impresources
from quantammsim import data
from pathlib import Path
from bidask import edge
from quantammsim.utils.data_processing.coinbase_data_utils import _cleaned_up_coinbase_data, import_historic_coinbase_data
from quantammsim.utils.data_processing.minute_daily_conversion_utils import calculate_annualised_daily_volatility_from_minute_data, expand_daily_to_minute_data, resample_minute_level_OHLC_data_to_daily
from quantammsim.utils.data_processing.datetime_utils import (
    datetime_to_unixtimestamp,
    unixtimestamp_to_datetime,
    unixtimestamp_to_minute_datetime,
    unixtimestamp_to_midnight_datetime,
    unixtimestamp_to_precise_datetime,
    pddatetime_to_unixtimestamp,
)

mc_data_available_for = ["ETH", "BTC"]

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


def update_historic_data(token, root):
    print("get starting")
    path = root + token + "_USD.csv"
    dailyPath = root + token + "_USD_daily.csv"
    hourlyPath = root + token + "_USD_hourly.csv"

    # TODO change variables if symbol is usdt
    usdtPath = root + token + "_USDT.csv"
    usdt = False
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
            "Volume " + token: float,
            "tradecount": float,
        },
    )

    if "tradecount" in csvData.columns:
        csvData = csvData.drop(
            columns=["tradecount"]
        )  # TODO do we need to drop or just not read?

    print("sorting")
    csvData = csvData.sort_values(by="unix", ascending=True)
    print(csvData.shape[0])
    print("sorted")
    prevRow = csvData.iloc[0]
    notFirstRow = False
    totalMissingUnixPoints = list()
    totalMissingDatePoints = list()
    totalMissingOpenPoints = list()
    totalMissingHighPoints = list()
    totalMissingLowPoints = list()
    totalMissingClosePoints = list()
    totalMissingCoinVolumePoints = list()
    totalMissingUsdVolumePoints = list()
    totalMissingToken = list()
    consequtiveMissing = 1
    oneMinute = 60000

    for index, row in csvData.iterrows():
        # try:
        nextUnix = prevRow["unix"] + (consequtiveMissing * oneMinute)
        if notFirstRow and nextUnix != row["unix"]:
            tightLoopBreak = 10000000000
            missingUnixPoints = list()
            missingDatePoints = list()
            missingOpenPoints = list()
            missingHighPoints = list()
            missingLowPoints = list()
            missingClosePoints = list()
            missingCoinVolumePoints = list()
            missingUsdVolumePoints = list()
            missingToken = list()

            while nextUnix < row["unix"] and consequtiveMissing < tightLoopBreak:
                missingUnixPoints.append(int(nextUnix))
                missingOpenPoints.append(prevRow["close"])
                missingHighPoints.append(prevRow["close"])
                missingLowPoints.append(prevRow["close"])
                missingClosePoints.append(prevRow["close"])
                missingDatePoints.append(
                    unixtimestamp_to_precise_datetime(nextUnix)
                )
                missingCoinVolumePoints.append(int(0))
                missingUsdVolumePoints.append(int(0))
                missingToken.append(token + "/USD")
                consequtiveMissing += 1
                nextUnix = prevRow["unix"] + (consequtiveMissing * oneMinute)

            if tightLoopBreak > 10000000000:
                print("ERROR TIGHT LOOP")

            if len(missingUnixPoints) > 60:
                # try:
                start_time = unixtimestamp_to_datetime(
                    missingUnixPoints[0] - oneMinute
                )
                end_time = unixtimestamp_to_datetime(
                    missingUnixPoints[len(missingUnixPoints) - 1]
                )
                coinBaseResults = import_historic_coinbase_data(
                    token, start_time, end_time
                )
                for retrievedIndex, retrievedRow in coinBaseResults.iterrows():
                    unixIndex = retrievedIndex.value / 1000000
                    try:
                        matchedMissingUnixIndex = missingUnixPoints.index(
                            unixIndex
                        )  # throws exception if not found
                        missingClosePoints[matchedMissingUnixIndex] = (
                            retrievedRow["close"]
                        )
                        
                        # add Open, Low, High, and Volume of Index
                        missingOpenPoints[matchedMissingUnixIndex] = retrievedRow["open"]
                        missingHighPoints[matchedMissingUnixIndex] = retrievedRow["high"]
                        missingLowPoints[matchedMissingUnixIndex] = retrievedRow["low"]
                        missingCoinVolumePoints[matchedMissingUnixIndex] = retrievedRow["volume"]
                        missingUsdVolumePoints[matchedMissingUnixIndex] = retrievedRow["volume"] * retrievedRow["close"]
                    except Exception as e:
                        print(e)
#                 except Exception as e:
                    # print(e)

            totalMissingUnixPoints = totalMissingUnixPoints + missingUnixPoints
            totalMissingDatePoints = totalMissingDatePoints + missingDatePoints
            totalMissingOpenPoints = totalMissingOpenPoints + missingOpenPoints
            totalMissingHighPoints = totalMissingHighPoints + missingHighPoints
            totalMissingLowPoints = totalMissingLowPoints + missingLowPoints
            totalMissingClosePoints = totalMissingClosePoints + missingClosePoints
            totalMissingCoinVolumePoints = (
                totalMissingCoinVolumePoints + missingCoinVolumePoints
            )
            totalMissingUsdVolumePoints = (
                totalMissingUsdVolumePoints + missingUsdVolumePoints
            )
            totalMissingToken = totalMissingToken + missingToken
            prevRow = row
            consequtiveMissing = 1
        else:
            consequtiveMissing = 1
            prevRow = row
        notFirstRow = True
        # except Exception as e:
        #     print(e)

    if usdt:
        # TODO convert prices to USD
        print("usdt file converstion started")

    if len(totalMissingUnixPoints) != len(totalMissingClosePoints):
        print("different lengths")

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

    csvData = csvData.append(missingDf)
    csvData = csvData.sort_values(by="unix", ascending=True)
    todayUnix = 1662246000000
    lastUnix = csvData.iloc[csvData.shape[0] - 1]["unix"]
    lastTime = unixtimestamp_to_datetime(lastUnix)
    todayTime = unixtimestamp_to_datetime(todayUnix)

    if todayUnix > lastUnix + 1:
        print("fill to current")
        coinBaseClosePrices = import_historic_coinbase_data(token, lastTime, todayTime)
        coinBaseList = coinBaseClosePrices["close"].tolist()

        if len(coinBaseList) != 0:
            csvData = csvData.append(
                createMissingDataFrameFromClosePrices(lastUnix, coinBaseList, token)
            )

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

    csvData.to_csv(path, mode="w", index=False)
    csvData[csvData["date"].str.contains(":00:00")].to_csv(
        hourlyPath, mode="w", index=False
    )
    csvData[csvData["date"].str.contains("05:00:00")].to_csv(
        dailyPath, mode="w", index=False
    )


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
            price_data = get_historic_csv_data(list_of_tickers, cols, root)
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
            price_data_non_mc = get_historic_csv_data(non_mc_tokens, cols, root)
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

    n_chunks = (len(prices) - remainder_idx) / chunk_period
    # check that we can cleanly divide data into 'chunk_period' units
    # if not, we will remove off the last little bit of the dataset.
    # (note that this doesn't interefere with the above burnin manipulations
    # as we made sure to 'add' a chunk-divisible portion to the start)
    if return_slippage:
        n_chunks = int((len(prices) - remainder_idx) / 1440) * 1440 / chunk_period
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
            spread.append(np.array(per_ticker_spread))
            per_ticker_annualised_daily_volatility = (
                np.array(calculate_annualised_daily_volatility_from_minute_data(
                    price_data_filtered.copy(), ticker
                )
            ))
            # per_ticker_daily_volume = calculate_daily_volume_from_minute_data(
            #     price_data_filtered, ticker
            # )
            annualised_daily_volatility.append(per_ticker_annualised_daily_volatility)
            # daily_volume.append(per_ticker_daily_volume)
            daily_OHLC_data=resample_minute_level_OHLC_data_to_daily(price_data_filtered.copy(),ticker)
            daily_volume.append(np.array(daily_OHLC_data["Volume USD_"+ticker]))
        spread = np.repeat(np.array(spread).T,1440,axis=0)
        daily_volume =  np.repeat(np.array(daily_volume).T,1440,axis=0)
        annualised_daily_volatility = np.repeat(
            np.array(annualised_daily_volatility).T, 1440, axis=0
        )
        # set spread of USD asset to 0
        # idx = list_of_tickers.index("DAI")
        # spread[idx, :] = 0.0
    # if return_slippage:
    # spread_rebased = spread[remainder_idx:]
    # if n_chunks.is_integer() is False:
    prices_rebased = prices_rebased[: int(n_chunks) * chunk_period]
    unix_values_rebased = unix_values_rebased[: int(n_chunks) * chunk_period]

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
        eth_price = prices_rebased[:,list_of_tickers.index("ETH")]
        # gas_price is daily, expland to minute
        gas_prices_minute = expand_daily_to_minute_data(gas_prices, scale='s')
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
    if return_slippage:
        data_dict["spread"] = spread
        data_dict["daily_volume"] = daily_volume
        data_dict["annualised_daily_volatility"] = annualised_daily_volatility
    if return_gas_prices:
        data_dict["gas_prices_in_dollars"] = gas_prices_in_dollars
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
        data_dict["max_memory_days"] = max_memory_days
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


if __name__ == "__main__":

    run_fingerprint_defaults = {
        "freq": "minute",
        "startDateString": "2021-02-03 00:00:00",
        # "endDateString": "2021-02-22 00:00:00",
        # "endTestDateString": "2021-03-03 00:00:00",
        "endDateString": "2022-06-03 00:00:00",
        "endTestDateString": "2022-07-03 00:00:00",
        "tokens": ["BTC", "DAI", "ETH"],
        "rule": "mean_reversion_channel",
        "optimisation_settings": {
            "base_lr": 0.01,
            "optimiser": "sgd",
            "decay_lr_ratio": 0.8,
            "decay_lr_plateau": 100,
            "batch_size": 1,
            "train_on_hessian_trace": False,
            "min_lr": 1e-6,
            "n_iterations": 1000,
            "n_cycles": 5,
            "sample_method": "uniform",
            "n_parameter_sets": 3,
            "training_data_kind": "historic",
            "include_flipped_training_data": False,
        },
        "initial_memory_length": 10.0,
        "initial_memory_length_delta": 0.0,
        "initial_k_per_day": 20,
        "bout_offset": 30 * 24 * 60 * 6,
        # "bout_offset": 24 * 60,
        "initial_weights_logits": 1.0,
        "initial_log_amplitude": -10.0,
        "initial_raw_width": -8.0,
        "initial_raw_exponents": 0.0,
        "subsidary_pools": [
            # {
            #     "tokens": ["BTC", "DAI"],
            #     "update_rule": "anti_momentum",
            #     "initial_memory_length": 5,
            #     "initial_k_per_day": 50,
            # },
            # {
            #     "tokens": ["BTC", "DAI"],
            #     "update_rule": "momentum",
            #     "initial_memory_length": 5,
            #     "initial_k_per_day": 50,
            # },
        ],
        "maximum_change": 3e-4,
        "chunk_period": 60,
        "weight_interpolation_period": 60,
        "return_val": "sharpe",
        "initial_pool_value": 1000000.0,
        "fees": 0.0,
        "use_alt_lamb": True,
        "use_pre_exp_scaling": True,
        "weight_interpolation_method": "linear",
    }

    # out_mc = get_data_dict(
    #     ["ETH", "BTC", "DAI"],
    #     run_fingerprint_defaults,
    #     data_kind="mc",
    #     root="../data/",
    #     max_memory_days=365.0,
    #     start_date_string=run_fingerprint_defaults["startDateString"],
    #     end_time_string=run_fingerprint_defaults["endDateString"],
    #     start_time_test_string=run_fingerprint_defaults["endDateString"],
    #     end_time_test_string=run_fingerprint_defaults["endTestDateString"],
    # )
    # out = get_data_dict(
    #     ["ETH", "BTC", "DAI"],
    #     run_fingerprint_defaults,
    #     data_kind="historic",
    #     root="../data/",
    #     max_memory_days=365.0,
    #     start_date_string=run_fingerprint_defaults["startDateString"],
    #     end_time_string=run_fingerprint_defaults["endDateString"],
    #     start_time_test_string=run_fingerprint_defaults["endDateString"],
    #     end_time_test_string=run_fingerprint_defaults["endTestDateString"],
    # )
