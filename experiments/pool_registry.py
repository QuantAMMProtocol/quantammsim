"""Registry of on-chain reClAMM pools for sim-vs-world gas calibration.

Extracts pool state from reclamm-simulations DB and computes TVL in USD
at each pool's plausible_start date. Maps chain → realistic gas costs.
Also provides initial on-chain state (Ra, Rb, Va, Vb) and world balance
history for comparison.

Pools excluded:
  - EUR_USDC_b, sUSDai_USDT0, WXPL_USDT0: stable/stable pairs
  - wstETH_GNO: boosted (wstETH yield-bearing)
"""

import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Database path (reclamm-simulations repo)
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = os.path.expanduser(
    "~/Projects/reclamm-simulations/data/pools_history.db"
)

# ---------------------------------------------------------------------------
# Chain → gas cost batteries (USD)
# ---------------------------------------------------------------------------
# Non-mainnet chains use flat gas costs.
# Ethereum uses time-varying gas from on-chain percentile CSVs.
CHAIN_GAS_COSTS = {
    "base": [0.0, 0.01, 0.1, 0.5],
    "gnosis": [0.0, 0.01, 0.1, 0.5],
    "avalanche": [0.0, 0.01, 0.1, 0.5],
}

# Ethereum mainnet: time-varying gas percentiles + flat zero baseline.
# CSVs live in gas_csvs/ with columns [unix, USD].
GAS_CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "gas_csvs")
ETHEREUM_GAS_PERCENTILES = ["50p", "75p", "90p", "95p"]


@dataclass
class PoolConfig:
    """Static metadata for a simulatable on-chain reClAMM pool."""

    label: str
    tokens: list  # quantammsim ticker names, e.g. ['BTC', 'ETH']
    chain: str
    swap_fee: float
    db_label: str  # table name in pools_history.db
    plausible_start: str  # YYYY-MM-DD
    reverse: bool  # True if DB token order is reversed vs quantammsim
    pool_address: str = ""  # on-chain contract address (hex, no 0x prefix)
    # Filled by extract_on_chain_state():
    on_chain_params: Optional[dict] = None  # price_ratio, margin, shift_rate
    initial_pool_value_usd: Optional[float] = None


# ---------------------------------------------------------------------------
# Pool definitions (non-stable, non-boosted pools with quantammsim tickers)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Chain → Balancer V3 API chain identifier
# ---------------------------------------------------------------------------
BALANCER_API_CHAIN = {
    "base": "BASE",
    "ethereum": "MAINNET",
    "gnosis": "GNOSIS",
    "avalanche": "AVALANCHE",
    "arbitrum": "ARBITRUM",
    "polygon": "POLYGON",
    "optimism": "OPTIMISM",
    "sonic": "SONIC",
}


POOL_REGISTRY = {
    "cbBTC_WETH": PoolConfig(
        label="cbBTC_WETH",
        tokens=["BTC", "ETH"],
        chain="base",
        swap_fee=0.0005,
        db_label="cbBTC_WETH",
        plausible_start="2025-08-01",
        reverse=True,
        pool_address="19aeb8168d921bb069c6771bbaff7c09116720d0",
    ),
    "cbBTC_WETH_post_oct": PoolConfig(
        label="cbBTC_WETH_post_oct",
        tokens=["BTC", "ETH"],
        chain="base",
        swap_fee=0.0005,
        db_label="cbBTC_WETH",
        plausible_start="2025-12-01",
        reverse=True,
        pool_address="19aeb8168d921bb069c6771bbaff7c09116720d0",
    ),
    "AAVE_WETH": PoolConfig(
        label="AAVE_WETH",
        tokens=["AAVE", "ETH"],
        chain="ethereum",
        swap_fee=0.0025,
        db_label="AAVE_WETH",
        plausible_start="2025-08-15",
        reverse=False,
        pool_address="9d1fcf346ea1b073de4d5834e25572cc6ad71f4d",
    ),
    "AAVE_WETH_post_gov": PoolConfig(
        label="AAVE_WETH_post_gov",
        tokens=["AAVE", "ETH"],
        chain="ethereum",
        swap_fee=0.0025,
        db_label="AAVE_WETH",
        plausible_start="2025-12-21",
        reverse=False,
        pool_address="9d1fcf346ea1b073de4d5834e25572cc6ad71f4d",
    ),
    "COW_WETH_b": PoolConfig(
        label="COW_WETH_b",
        tokens=["COW", "ETH"],
        chain="base",
        swap_fee=0.003,
        db_label="COW_WETH_b",
        plausible_start="2025-07-18",
        reverse=True,
        pool_address="ff028c1ec4559d3aa2b0859aa582925b5cc28069",
    ),
    "COW_WETH_e": PoolConfig(
        label="COW_WETH_e",
        tokens=["COW", "ETH"],
        chain="ethereum",
        swap_fee=0.003,
        db_label="COW_WETH_e",
        plausible_start="2025-09-21",
        reverse=True,
        pool_address="d321300ef77067d4a868f117d37706eb81368e98",
    ),
    "WAVAX_USDC": PoolConfig(
        label="WAVAX_USDC",
        tokens=["AVAX", "USDC"],
        chain="avalanche",
        swap_fee=0.001,
        db_label="WAVAX_USDC",
        plausible_start="2025-08-17",
        reverse=False,
        pool_address="8750ccffcddbff81b63790dbcb1ffd8c7dc4c16d",
    ),
    "GNO_USDC": PoolConfig(
        label="GNO_USDC",
        tokens=["GNO", "USDC"],
        chain="gnosis",
        swap_fee=0.003,
        db_label="GNO_USDC",
        plausible_start="2025-09-18",
        reverse=True,
        pool_address="70b3b56773ace43fe86ee1d80cbe03176cbe4c09",
    ),
}


def _date_to_unix(date_str: str) -> int:
    """Convert YYYY-MM-DD or YYYY-MM-DD HH:MM:SS to unix timestamp (seconds)."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


def _get_usd_price_at(ticker: str, unix_ms: int, data_root: str) -> float:
    """Get the USD price of a ticker at a given unix timestamp (ms)."""
    path = os.path.join(data_root, f"{ticker}_USD.parquet")
    df = pd.read_parquet(path)
    idx = (df["unix"] - unix_ms).abs().idxmin()
    return float(df.iloc[idx]["close"])


def extract_on_chain_state(
    pool: PoolConfig,
    db_path: str = DEFAULT_DB_PATH,
    data_root: str = None,
) -> PoolConfig:
    """Query the DB for on-chain state at plausible_start and compute USD TVL.

    Mutates and returns the pool config with on_chain_params and
    initial_pool_value_usd filled in.
    """
    if data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "quantammsim", "data"
        )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = _date_to_unix(pool.plausible_start)

    cur.execute(
        f"""SELECT * FROM {pool.db_label}
            WHERE timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1""",
        (ts + 3600,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise ValueError(
            f"No DB data for {pool.db_label} at {pool.plausible_start}"
        )

    # DB columns: timestamp, block_number, bpt_supply, balance_0, balance_1,
    #   spot_price, virtual_0, virtual_1, time_last_interaction,
    #   price_ratio, margin, shift_rate, swap_fee
    balance_0, balance_1 = row[3], row[4]
    price_ratio = row[9]
    margin = row[10]
    shift_rate = row[11]

    pool.on_chain_params = {
        "price_ratio": price_ratio,
        "margin": margin,
        "shift_rate": shift_rate,
        "swap_fee": row[12],
    }

    # Compute TVL in USD from per-token USD prices.
    # DB stores balances in contract token order (bring_pool_data.py never
    # applies reverse). The reverse flag tells us the mapping:
    #   reverse=False → balance_0=tokens[0], balance_1=tokens[1]
    #   reverse=True  → balance_0=tokens[1], balance_1=tokens[0]
    unix_ms = ts * 1000
    if pool.reverse:
        tickers_in_db_order = [pool.tokens[1], pool.tokens[0]]
    else:
        tickers_in_db_order = [pool.tokens[0], pool.tokens[1]]

    usd_prices = []
    for ticker in tickers_in_db_order:
        if ticker == "USDC":
            usd_prices.append(1.0)
        else:
            usd_prices.append(
                _get_usd_price_at(ticker, unix_ms, data_root)
            )

    pool.initial_pool_value_usd = (
        balance_0 * usd_prices[0] + balance_1 * usd_prices[1]
    )
    return pool


def extract_initial_state(
    pool: PoolConfig,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Extract on-chain Ra, Rb, Va, Vb at plausible_start in quantammsim order.

    quantammsim sorts tokens alphabetically, so token[0] is the
    alphabetically-first ticker. The reverse flag maps DB contract
    order to this sorted order.

    Returns dict with keys Ra, Rb, Va, Vb (floats).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ts = _date_to_unix(pool.plausible_start)

    cur.execute(
        f"""SELECT balance_0, balance_1, virtual_0, virtual_1
            FROM {pool.db_label}
            WHERE timestamp <= ?
            ORDER BY timestamp DESC LIMIT 1""",
        (ts + 3600,),
    )
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise ValueError(
            f"No DB data for {pool.db_label} at {pool.plausible_start}"
        )

    b0, b1, v0, v1 = row
    if pool.reverse:
        # DB contract order is opposite to quantammsim sorted order
        return {"Ra": b1, "Rb": b0, "Va": v1, "Vb": v0}
    else:
        return {"Ra": b0, "Rb": b1, "Va": v0, "Vb": v1}


def load_world_history(
    pool: PoolConfig,
    end_date: str = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """Load on-chain balance history from the DB.

    Returns dict with:
      timestamps: array of unix timestamps (seconds)
      bal_0: BPT-normalized balance of quantammsim token[0]
      bal_1: BPT-normalized balance of quantammsim token[1]
      raw_bal_0: raw (un-normalized) balance of quantammsim token[0]
      raw_bal_1: raw (un-normalized) balance of quantammsim token[1]
      governance_events: list of (timestamp, field, old_val, new_val)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    ts_start = _date_to_unix(pool.plausible_start) - 1000
    if end_date:
        ts_end = _date_to_unix(end_date)
    else:
        ts_end = 2_000_000_000  # far future

    cur.execute(
        f"""SELECT timestamp, bpt_supply, balance_0, balance_1,
                   price_ratio, margin, shift_rate, swap_fee
            FROM {pool.db_label}
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp""",
        (ts_start, ts_end),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No world history for {pool.db_label}")

    initial_bpt = rows[0][1]
    timestamps = []
    bal_db_0_norm = []
    bal_db_1_norm = []
    bal_db_0_raw = []
    bal_db_1_raw = []
    governance_events = []

    for i, row in enumerate(rows):
        ts, bpt, b0, b1, pr, margin, shift_rate, swap_fee = row
        timestamps.append(ts)
        norm = initial_bpt / bpt
        bal_db_0_norm.append(b0 * norm)
        bal_db_1_norm.append(b1 * norm)
        bal_db_0_raw.append(b0)
        bal_db_1_raw.append(b1)

        # Detect governance changes.
        # price_ratio drifts continuously via the shift mechanism, so
        # only flag large discrete jumps (>1% relative change) as governance.
        # margin, shift_rate, and swap_fee are set by governance and don't drift.
        if i > 0:
            prev = rows[i - 1]
            if not math.isclose(prev[4], pr, rel_tol=0.01):
                governance_events.append((ts, "price_ratio", prev[4], pr))
            if not math.isclose(prev[5], margin, rel_tol=1e-6):
                governance_events.append((ts, "margin", prev[5], margin))
            if not math.isclose(prev[6], shift_rate, rel_tol=1e-6):
                governance_events.append((ts, "shift_rate", prev[6], shift_rate))

    bal_db_0_norm = np.array(bal_db_0_norm)
    bal_db_1_norm = np.array(bal_db_1_norm)
    bal_db_0_raw = np.array(bal_db_0_raw)
    bal_db_1_raw = np.array(bal_db_1_raw)

    # Apply reverse: swap to quantammsim sorted token order
    if pool.reverse:
        bal_sorted_0, bal_sorted_1 = bal_db_1_norm, bal_db_0_norm
        raw_sorted_0, raw_sorted_1 = bal_db_1_raw, bal_db_0_raw
    else:
        bal_sorted_0, bal_sorted_1 = bal_db_0_norm, bal_db_1_norm
        raw_sorted_0, raw_sorted_1 = bal_db_0_raw, bal_db_1_raw

    return {
        "timestamps": np.array(timestamps),
        "bal_0": bal_sorted_0,
        "bal_1": bal_sorted_1,
        "raw_bal_0": raw_sorted_0,
        "raw_bal_1": raw_sorted_1,
        "governance_events": governance_events,
    }


def load_bpt_supply_df(
    pool: PoolConfig,
    end_date: str = None,
    db_path: str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load BPT supply as a DataFrame suitable for do_run_on_historic_data.

    Returns DataFrame with columns:
      unix: timestamps in milliseconds
      lp_supply: BPT normalized to 1.0 at plausible_start

    The normalization matches the simulator convention: lp_supply=1.0 at the
    start of the sim, scaling proportionally as the on-chain pool grows/shrinks.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    ts_start = _date_to_unix(pool.plausible_start) - 1000
    if end_date:
        ts_end = _date_to_unix(end_date)
    else:
        ts_end = 2_000_000_000

    cur.execute(
        f"""SELECT timestamp, bpt_supply
            FROM {pool.db_label}
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp""",
        (ts_start, ts_end),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No BPT data for {pool.db_label}")

    initial_bpt = rows[0][1]
    return pd.DataFrame({
        # Round to nearest minute boundary so timestamps land on the minute grid
        # used by raw_fee_like_amounts_to_fee_like_array.
        "unix": [round(r[0] / 60) * 60 * 1000 for r in rows],
        "lp_supply": [r[1] / initial_bpt for r in rows],
    })


def get_data_end_date(tokens: list, data_root: str = None) -> str:
    """Find the latest common date across all token parquets.

    Returns a date string like '2026-02-18 00:00:00'.
    """
    if data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "quantammsim", "data"
        )

    min_end = float("inf")
    for ticker in tokens:
        path = os.path.join(data_root, f"{ticker}_USD.parquet")
        df = pd.read_parquet(path, columns=["unix"])
        last = float(df["unix"].iloc[-1])
        if last < min_end:
            min_end = last

    # Convert ms to datetime
    dt = datetime.utcfromtimestamp(min_end / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def load_gas_csv(percentile: str) -> pd.DataFrame:
    """Load a gas percentile CSV as a DataFrame for do_run_on_historic_data.

    Returns DataFrame with columns [unix, trade_gas_cost_usd], timestamps
    floored to minute boundaries.
    """
    path = os.path.join(GAS_CSV_DIR, f"Gas_{percentile}.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"USD": "trade_gas_cost_usd"})
    df["unix"] = (df["unix"] // 60000) * 60000  # floor to minute boundary
    return df


def get_gas_costs(pool: PoolConfig, custom: list = None) -> list:
    """Return the gas cost battery for a pool's chain.

    For Ethereum, returns a list mixing flat 0.0 with gas percentile labels
    (e.g. ["0.0", "50p", "75p", "90p", "95p"]).
    For other chains, returns flat USD values.
    """
    if custom is not None:
        return custom
    if pool.chain == "ethereum":
        flat = [0.0, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0]
        return flat + ETHEREUM_GAS_PERCENTILES
    return CHAIN_GAS_COSTS.get(pool.chain, [0.0, 0.1, 1.0])


def print_pool_summary(pool: PoolConfig):
    """Print a summary of the pool's on-chain state."""
    print(f"\n{'='*60}")
    print(f"Pool: {pool.label}")
    print(f"  Chain:    {pool.chain}")
    print(f"  Tokens:   {pool.tokens[0]}/{pool.tokens[1]}")
    print(f"  Swap fee: {pool.swap_fee}")
    print(f"  Start:    {pool.plausible_start}")
    if pool.on_chain_params:
        p = pool.on_chain_params
        print(f"  On-chain: PR={p['price_ratio']:.4f}  "
              f"margin={p['margin']}  shift_rate={p['shift_rate']}  "
              f"fee={p['swap_fee']}")
    if pool.initial_pool_value_usd:
        print(f"  TVL:      ${pool.initial_pool_value_usd:,.0f} USD")
    print(f"  Gas battery: {get_gas_costs(pool)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Print summary of all pools
    for label, pool in POOL_REGISTRY.items():
        try:
            extract_on_chain_state(pool)
            print_pool_summary(pool)
        except Exception as e:
            print(f"\n{label}: FAILED — {e}")
