import numpy as np
import pandas as pd
import json
import pyarrow as pa

from quantammsim.utils.data_processing.binance_data_utils import concat_csv_files
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_csv_data,
    unixtimestamp_to_datetime,
    update_historic_data,
)
from multiprocessing import Pool
import time

# load data
startUnix = 1577836860000  # 2021-01-01
endUnix = 1726185540000

tokens = [
    #"AAVE",
    #"ADA",
    #"ATOM",
    #"AVAX",
    #"BAT",
    #"BNB",
    #"BTC",
    #"DOGE",
    #"EOS",
    #"ETH",
    #"LINK",
    #"LTC",
    #"MATIC",
    #"QTUM",
    #"SOL",
    #"TRX",
    #"UNI",
    #"XLM",
    #"XMR",
    #"XRP",
    "USDC",
]


def process_token(token):
    print(token + " starting")
    root = "/media/cadeh/3137-3364/local_data/"
    return update_historic_data(token, root)

def convert_to_parquet(token):
    print(token + " starting")
    root = "/media/cadeh/3137-3364/local_data/combined_data"


    csvDf = get_historic_csv_data([token], ["date", "symbol", "open", "high", 
                                            "low", "close", "Volume USD", "Volume " + token, "tradecount"], root)
    

    csvDf.to_parquet(root + '/' + token + '_USD' + '.parquet', engine='pyarrow')
    

for token in tokens:
    process_token(token)