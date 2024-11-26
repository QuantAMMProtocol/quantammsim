import numpy as np
import pandas as pd
import json

from quantamm.utils.data_processing.binance_data_utils import concat_csv_files
from quantamm.utils.data_processing.historic_data_utils import (
    unixtimestamp_to_datetime,
    update_historic_data,
)
from multiprocessing import Pool

# load data
startUnix = 1577836860000  # 2021-01-01
endUnix = 1726185540000

tokens = [
    # "AAVE",
    # "ADA",
    # "ATOM",
    # "AVAX",
    # "BAT",
    # "BNB",
    # "BTC",
    # "DOGE",
    # "EOS",
    #"ETH",
    #"LINK",
    #"LTC",
    #"MATIC",
    #"QTUM",
    #"SOL",
    #"TRX",
    "UNI",
    "XLM",
    "XMR",
    "XRP",
    "DAI",
]


def process_token(token):
    print(token + " starting")
    root = "/media/cadeh/3137-3364/local_data/"
    return update_historic_data(token, root)

for token in tokens:
    process_token(token)
#with Pool() as pool:
#    pool.map(process_token, tokens)
