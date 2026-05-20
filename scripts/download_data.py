import os
import hashlib
import zipfile
import argparse
from pathlib import Path

# binance_historical_data uses mpire with the default fork start method. quantammsim
# transitively imports JAX (multithreaded), and fork-after-threads deadlocks on macOS.
# Force spawn workers before any mpire pool is constructed.
import mpire

_orig_workerpool_init = mpire.WorkerPool.__init__


def _workerpool_init_spawn(self, *args, **kwargs):
    kwargs.setdefault("start_method", "spawn")
    return _orig_workerpool_init(self, *args, **kwargs)


mpire.WorkerPool.__init__ = _workerpool_init_spawn

from tqdm import tqdm
from quantammsim.utils.data_processing.historic_data_utils import (
    update_historic_data,
)

# Get absolute paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "quantammsim" / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_DIR_STR = str(DATA_DIR) + '/'

TICKER_FILE = SCRIPT_DIR / "ticker_list.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historic data for tickers.")
    parser.add_argument(
        "tickers",
        nargs="*",
        help="List of tickers to process. If provided, overrides ticker_list.txt",
    )
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        with open(TICKER_FILE, "r") as f:
            tickers = f.readlines()

    for token in tickers:
        token = token.strip().upper()
        print(f"Processing {token}")
        update_historic_data(token, DATA_DIR_STR)
        print(f"Finished processing {token}")
        os.rename(
            DATA_DIR_STR + "combined_data/" + token + "_USD.parquet",
            DATA_DIR_STR + token + "_USD.parquet",
        )
        os.rename(
            DATA_DIR_STR + "combined_data/" + token.strip().upper() + "_USD_daily.csv",
            DATA_DIR_STR + token.strip().upper() + "_USD_daily.csv",
        )
