
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_csv_data,
    update_historic_data,
)
import debug
tokens = [
    "ARB"
    "AAVE",
    "ADA",
    "ATOM",
    "AVAX",
    "BAT",
    "BNB",
    "BTC",
    "DOGE",
    "EOS",
    "ETH",
    "LINK",
    "LTC",
    "MATIC",
    "QTUM",
    "SOL",
    "TRX",
    "UNI",
    "XLM",
    "XMR",
    "XRP",
    "USDC",
    "DYDX",
    "ALGO",
    "CRV",
    "MKR",
    "SUSHI",
    "1INCH",
    "BAL",
    "COMP",
    "FIL",
    "FRAX",
    "INJ",
    "LDO",
    "PAXG"
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
    root = "../local_data/"
    return update_historic_data(token_name, root)


for token in tokens:
    process_token(token)

    # process_token(token)