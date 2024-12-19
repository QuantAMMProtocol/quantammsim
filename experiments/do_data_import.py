
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_csv_data,
    update_historic_data,
)

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


def process_token(token_name):
    """
    Processes a given token by updating its historic data.

    Args:
        token_name (str): The token to be processed.

    Returns:
        The result of the update_historic_data function.

    Prints:
        A message indicating the start of the process for the given token.
    """
    print(token_name + " starting")
    root = "/media/cadeh/3137-3364/local_data/"
    return update_historic_data(token_name, root)


def convert_to_parquet(token_name):
    """
    Converts historical CSV data for a given token to a Parquet file.

    Args:
        token_name (str): The token symbol for which the data is to be converted.

    Returns:
        None

    Side Effects:
        Saves a Parquet file in the specified root directory with the token's data.

    Example:
        convert_to_parquet("BTC")
    """
    print(token + " starting")
    root = "/media/cadeh/3137-3364/local_data/combined_data"

    csvDf = get_historic_csv_data(
        [token_name],
        [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "Volume USD",
            "Volume " + token_name,
            "tradecount",
        ],
        root,
    )

    csvDf.to_parquet(root + "/" + token_name + "_USD" + ".parquet", engine="pyarrow")


for token in tokens:
    process_token(token)

    process_token(token)