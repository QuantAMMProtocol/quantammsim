"""Build 2D arb-volume grids (cadence x gas) for all Binance-matchable pools.

v2: Per-day daily arb volumes at each grid point, correct pool weights,
    pool type dispatch (WEIGHTED vs RECLAMM).

For each real Balancer pool where both tokens have Binance minute data:
  - Uses actual LP supply trajectory (BPT totalShares from panel)
  - Uses actual initial TVL from panel
  - Uses correct pool weights from pools.parquet
  - Dispatches reCLAMM pools with on-chain params from pools_history.db
  - Sweeps cadence x gas_cost as scalar grid
  - Stores per-day V_arb at each grid point (not aggregated)

Output: results/pool_grids_v2/{pool_id_prefix}_daily.parquet + summary CSV.

Usage:
    python scripts/build_pool_grids.py
    python scripts/build_pool_grids.py --workers 6 --train-days 90
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import ast
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantammsim.runners.jax_runners import do_run_on_historic_data
from quantammsim.utils.data_processing.historic_data_utils import get_historic_parquet_data

# ── Output ────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "pool_grids_v2",
)

# ── Grid ──────────────────────────────────────────────────────────────────
CADENCES = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60]

# Finer gas grids — concentrated in $0.1-$3.0 where most fitted values land.
# Mainnet: 19 points (was 10). Fills the $0.25→$1.50 gap that caused
# zero-arb artifacts on low-volatility days.
GAS_COSTS_MAINNET = [
    0.0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8,
    1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0,
]
# L2: 14 points (was 10). Fills $0.05→$0.50 range.
GAS_COSTS_L2 = [
    0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.05,
    0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 2.0,
]

# ── Token mapping ─────────────────────────────────────────────────────────
# Maps Balancer pool token symbols to Binance trading symbols.
# Validated via Balancer hourly vs Binance minute price comparison:
# all wrappers below have daily return correlation > 0.75 with their
# underlying, or are stablecoins (basis < 0.5%).
TOKEN_MAP = {
    # Wrapped natives (corr > 0.96)
    "WBTC": "BTC", "WETH": "ETH", "cbBTC": "BTC",
    # ETH LSTs / Aave wrappers (corr 0.87-0.97)
    "wstETH": "ETH", "stETH": "ETH", "rETH": "ETH", "cbETH": "ETH",
    "waEthLidoWETH": "ETH", "waEthLidowstETH": "ETH",
    "waBasWETH": "ETH",       # Aave wrapped WETH on Base (corr 0.975)
    "waGnowstETH": "ETH",    # Aave wrapped wstETH on Gnosis (corr 0.976)
    # GNO wrappers (corr 0.66-0.98)
    "waGnoGNO": "GNO",       # Aave wrapped GNO on Gnosis (corr 0.979)
    "osGNO": "GNO",          # StakeWise staked GNO (corr 0.755)
    # S (Sonic) wrappers (corr 0.945)
    "wS": "S",
    "stS": "S",              # Staked Sonic
    # SOL LSTs (corr 0.922)
    "JitoSOL": "SOL",        # Jito staked SOL
    # POL/MATIC variants (corr 0.96)
    "wPOL": "POL", "WMATIC": "POL", "MATIC": "POL",
    # Stablecoin equivalents (all ~$1.00, basis < 0.5%)
    "USDC.e": "USDC", "USDbC": "USDC", "waBasUSDC": "USDC",
    "DAI": "USDC",                     # Stale Binance data; basis < 10bps vs USDC
    "WXDAI": "USDC", "sDAI": "USDC",   # Gnosis DAI variants → USDC
    "USDT": "USDC",
    "DOLA": "USDC",
    "scUSD": "USDC",
}

# ── reCLAMM pool → DB table mapping ──────────────────────────────────────
RECLAMM_DB_PATH = "/Users/matthew/Projects/reclamm-simulations/data/pools_history.db"
RECLAMM_POOL_TABLE_MAP = {
    "0x9d1fcf346ea1b0": "AAVE_WETH",
}


def _get_binance_tokens():
    """Get set of tokens with Binance minute parquets."""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "quantammsim", "data",
    )
    tokens = set()
    for f in os.listdir(data_dir):
        if f.endswith("_USD.parquet"):
            tokens.add(f.replace("_USD.parquet", ""))
    return tokens


def _map_token(tok, binance_tokens):
    """Map a Balancer token symbol to Binance symbol, or None."""
    mapped = TOKEN_MAP.get(tok, tok)
    return mapped if mapped in binance_tokens else None


def gas_costs_for_chain(chain):
    return GAS_COSTS_L2 if chain != "MAINNET" else GAS_COSTS_MAINNET


def _parse_weights(weights_raw):
    """Parse weights from pools.parquet (numpy array of strings or list)."""
    if weights_raw is None:
        return None
    if isinstance(weights_raw, str):
        weights_raw = ast.literal_eval(weights_raw)
    if hasattr(weights_raw, 'tolist'):
        weights_raw = weights_raw.tolist()
    parsed = []
    for w in weights_raw:
        if w is None or str(w).lower() == 'none':
            return None
        parsed.append(float(w))
    return parsed


def _parse_tokens(tokens_raw):
    """Parse tokens from pools.parquet."""
    if isinstance(tokens_raw, str):
        try:
            return ast.literal_eval(tokens_raw)
        except (ValueError, SyntaxError):
            return [t.strip() for t in tokens_raw.split(",")]
    if hasattr(tokens_raw, 'tolist'):
        return tokens_raw.tolist()
    return list(tokens_raw)


# ── Pool metadata loading ─────────────────────────────────────────────────

def load_pools_metadata():
    """Load pools.parquet to get weights, pool_type, and true token count."""
    pools_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "local_data", "noise_calibration", "pools.parquet",
    )
    pools = pd.read_parquet(pools_path)
    meta = {}
    for _, row in pools.iterrows():
        pool_id = row["pool_id"]
        prefix = pool_id[:16]
        tokens = _parse_tokens(row["tokens"])
        weights = _parse_weights(row["weights"])
        meta[prefix] = {
            "pool_id": pool_id,
            "pool_type": row["pool_type"],
            "tokens_full": tokens,
            "n_tokens": len(tokens),
            "weights": weights,
        }
    return meta


def load_reclamm_params(pool_id_prefix):
    """Load reCLAMM on-chain params from pools_history.db."""
    table_name = RECLAMM_POOL_TABLE_MAP.get(pool_id_prefix)
    if table_name is None:
        return None
    if not os.path.exists(RECLAMM_DB_PATH):
        return None
    conn = sqlite3.connect(RECLAMM_DB_PATH)
    try:
        df = pd.read_sql(f"SELECT * FROM [{table_name}] ORDER BY timestamp DESC LIMIT 1", conn)
        if len(df) == 0:
            return None
        row = df.iloc[0]
        return {
            "price_ratio": float(row["price_ratio"]),
            "centeredness_margin": float(row["margin"]),
            "shift_exponent": float(row["shift_rate"]),
            "swap_fee": float(row["swap_fee"]),
        }
    except Exception:
        return None
    finally:
        conn.close()


# ── Panel loading and pool matching ───────────────────────────────────────

def load_panel_and_match(train_days):
    """Load panel, filter to last N days, find matchable 2-token pools."""
    panel_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "local_data", "noise_calibration", "panel.parquet",
    )
    panel = pd.read_parquet(panel_path)

    if "obs_date" not in panel.columns and "date" in panel.columns:
        panel = panel.rename(columns={"date": "obs_date"})
    panel["obs_date"] = pd.to_datetime(panel["obs_date"])

    if "tvl" not in panel.columns and "log_tvl" in panel.columns:
        panel["tvl"] = np.exp(panel["log_tvl"])

    if train_days > 0:
        cutoff = panel["obs_date"].max() - pd.Timedelta(days=train_days)
        panel = panel[panel["obs_date"] >= cutoff].copy()
    else:
        panel = panel.copy()  # no date filter — use all data

    binance_tokens = _get_binance_tokens()
    pools_meta = load_pools_metadata()

    pools = []
    for pool_id, grp in panel.groupby("pool_id"):
        prefix = pool_id[:16]
        meta = pools_meta.get(prefix)
        if meta is None:
            continue

        # Skip multi-token pools
        if meta["n_tokens"] > 2:
            continue

        pool_type = meta["pool_type"]

        # Get tokens for Binance matching from panel
        row = grp.iloc[0]
        tokens_str = row["tokens"]
        toks = [t.strip() for t in tokens_str.split(",")]
        if len(toks) != 2:
            continue

        mapped = []
        for t in toks:
            m = _map_token(t, binance_tokens)
            if m is None:
                break
            mapped.append(m)
        else:
            if mapped[0] == mapped[1]:
                continue

            chain = row["chain"]
            fee = row.get("swap_fee", np.exp(row["log_fee"]))

            # Get weights
            weights = meta["weights"]
            if pool_type == "WEIGHTED" and weights is None:
                weights = [0.5, 0.5]

            # reCLAMM params
            reclamm_params = None
            if pool_type == "RECLAMM":
                reclamm_params = load_reclamm_params(prefix)
                if reclamm_params is None:
                    print(f"  SKIP reCLAMM {'/'.join(mapped)} ({chain}): "
                          f"no DB params for {prefix}")
                    continue

            pools.append({
                "pool_id": pool_id,
                "pool_id_prefix": prefix,
                "tokens": mapped,
                "chain": chain,
                "fee": float(fee),
                "panel_data": grp,
                "pool_type": pool_type,
                "weights": weights,
                "reclamm_params": reclamm_params,
            })

    return pools


def build_lp_supply_df(panel_pool):
    """Build lp_supply_df from daily BPT supply. Returns (df, initial_tvl)."""
    panel_pool = panel_pool.sort_values("obs_date")

    has_bpt = (
        "total_shares" in panel_pool.columns
        and not panel_pool["total_shares"].isna().all()
        and (panel_pool["total_shares"] > 0).any()
    )

    initial_tvl = float(panel_pool["tvl"].iloc[0]) if len(panel_pool) > 0 else 0.0

    if not has_bpt:
        return None, initial_tvl

    bpt = panel_pool[["obs_date", "total_shares", "tvl"]].drop_duplicates("obs_date")
    initial_bpt = bpt["total_shares"].iloc[0]

    if initial_bpt <= 0 or initial_tvl <= 0:
        return None, initial_tvl

    unix_ms = bpt["obs_date"].apply(
        lambda d: int(pd.Timestamp(d).timestamp() * 1000)
    ).values
    lp_supply = (bpt["total_shares"].values / initial_bpt).astype(float)

    return pd.DataFrame({"unix": unix_ms, "lp_supply": lp_supply}), initial_tvl


def get_date_range(panel_data):
    """Get start/end date strings from panel data."""
    dates = panel_data["obs_date"].sort_values()
    start = dates.iloc[0].strftime("%Y-%m-%d %H:%M:%S")
    end = dates.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
    return start, end


# ── Simulation ────────────────────────────────────────────────────────────

def run_arb_sim(tokens, fee, initial_tvl, start, end, cadence, gas_cost,
                lp_supply_df=None, weights=None, pool_type="WEIGHTED",
                reclamm_params=None, price_data=None):
    """Run arb-only sim at one (cadence, gas_cost) point.

    Returns: pd.Series with date index → daily arb volume.
    """
    fp = {
        "tokens": tokens,
        "startDateString": start,
        "endDateString": end,
        "initial_pool_value": initial_tvl,
        "fees": fee,
        "gas_cost": float(gas_cost),
        "arb_fees": 0.0,
        "do_arb": True,
        "noise_trader_ratio": 0.0,
        "arb_frequency": int(cadence),
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "max_memory_days": 0,
    }

    if pool_type == "RECLAMM" and reclamm_params is not None:
        fp["rule"] = "reclamm"
        fp["reclamm_use_shift_exponent"] = True
        fp["fees"] = reclamm_params["swap_fee"]
        params = {
            "price_ratio": jnp.array(reclamm_params["price_ratio"]),
            "centeredness_margin": jnp.array(reclamm_params["centeredness_margin"]),
            "shift_exponent": jnp.array(reclamm_params["shift_exponent"]),
        }
    else:
        fp["rule"] = "balancer"
        if weights is not None and len(weights) == 2:
            logits = np.log(np.array(weights, dtype=float))
            params = {"initial_weights_logits": jnp.array(logits)}
        else:
            params = {"initial_weights_logits": jnp.array([0.0, 0.0])}

    result = do_run_on_historic_data(
        fp, params, lp_supply_df=lp_supply_df, verbose=False,
        price_data=price_data, preslice_burnin=False,
    )

    reserves = np.array(result["reserves"])
    prices = np.array(result["data_dict"]["prices"])
    unix_ms = np.array(result["data_dict"]["unix_values"])
    start_idx = int(result["data_dict"]["start_idx"])

    T = reserves.shape[0] - 1
    prices_window = prices[start_idx:start_idx + T + 1]
    delta_r = np.diff(reserves, axis=0)
    step_vol = np.sum(np.abs(delta_r * prices_window[1:]), axis=1) / 2.0

    dates = pd.to_datetime(
        unix_ms[start_idx + 1:start_idx + T + 1], unit="ms",
    ).normalize()
    daily = pd.DataFrame(
        {"date": dates, "volume": step_vol},
    ).groupby("date")["volume"].sum()

    return daily


def _run_cadence_sweep(pool_info, cadence, gas_costs):
    """Worker: sweep all gas costs for one (pool, cadence).

    Returns list of dicts with per-day data: one dict per (gas, date) pair,
    plus a summary list.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    tokens = pool_info["tokens"]
    fee = pool_info["fee"]
    initial_tvl = pool_info["initial_tvl"]
    start = pool_info["start"]
    end = pool_info["end"]
    lp_supply_df = pool_info["lp_supply_df"]
    weights = pool_info.get("weights")
    pool_type = pool_info.get("pool_type", "WEIGHTED")
    reclamm_params = pool_info.get("reclamm_params")

    price_data = pool_info.get("price_data")
    if price_data is None:
        sorted_tokens = sorted(tokens)
        price_data = get_historic_parquet_data(sorted_tokens, ["close"])

    daily_rows = []
    summary_rows = []

    for gas in gas_costs:
        try:
            daily = run_arb_sim(
                tokens, fee, initial_tvl, start, end, cadence, gas,
                lp_supply_df=lp_supply_df,
                weights=weights,
                pool_type=pool_type,
                reclamm_params=reclamm_params,
                price_data=price_data,
            )
            # Store per-day data
            for date_val, vol in daily.items():
                daily_rows.append({
                    "cadence": cadence,
                    "gas_cost": gas,
                    "date": date_val,
                    "daily_arb_volume": vol,
                })
            # Summary for diagnostics
            summary_rows.append({
                "cadence": cadence,
                "gas_cost": gas,
                "total_arb_volume": daily.sum(),
                "median_daily_arb_volume": daily.median(),
                "mean_daily_arb_volume": daily.mean(),
                "n_days": len(daily),
            })
        except Exception as e:
            summary_rows.append({
                "cadence": cadence,
                "gas_cost": gas,
                "total_arb_volume": np.nan,
                "median_daily_arb_volume": np.nan,
                "mean_daily_arb_volume": np.nan,
                "n_days": 0,
                "error": str(e),
            })

    return daily_rows, summary_rows


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_pool_grid(summary_df, pool_id, tokens, chain, fee, tvl, pool_type,
                   gas_costs, output_dir):
    """Simple 2-panel diagnostic: V_arb vs cadence, gas attenuation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    df = summary_df

    ax = axes[0]
    for gas in gas_costs:
        sub = df[df["gas_cost"] == gas].sort_values("cadence")
        if len(sub) > 0:
            ax.plot(sub["cadence"], sub["median_daily_arb_volume"],
                    "o-", label=f"${gas}", markersize=3)
    ax.set_xlabel("Cadence (min)")
    ax.set_ylabel("Median daily V_arb ($)")
    ax.set_title("V_arb vs cadence")
    ax.legend(fontsize=5, ncol=2, title="gas")

    ax = axes[1]
    for cadence in CADENCES:
        sub = df[df["cadence"] == cadence].sort_values("gas_cost")
        v0 = sub[sub["gas_cost"] == 0.0]["median_daily_arb_volume"].values
        if len(v0) > 0 and v0[0] > 0:
            ratio = sub["median_daily_arb_volume"].values / v0[0]
            ax.plot(sub["gas_cost"].values, ratio, "o-",
                    label=f"{cadence}min", markersize=3)
    ax.set_xlabel("Gas cost ($)")
    ax.set_ylabel("V_arb / V_arb(gas=0)")
    ax.set_title("Gas attenuation")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=5, ncol=2)

    ax = axes[2]
    for gas in gas_costs:
        sub = df[df["gas_cost"] == gas].sort_values("cadence")
        vals = sub["median_daily_arb_volume"].values
        if len(vals) > 0 and np.all(vals > 0):
            ax.plot(sub["cadence"], vals, "o-", label=f"${gas}", markersize=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cadence (min)")
    ax.set_ylabel("Median daily V_arb ($)")
    ax.set_title("Log-log")
    ax.legend(fontsize=5, ncol=2, title="gas")

    tok_str = "/".join(tokens)
    type_str = f" [{pool_type}]" if pool_type != "WEIGHTED" else ""
    fig.suptitle(
        f"{tok_str} ({chain}, fee={fee:.2%}, TVL=${tvl:,.0f}){type_str}\n"
        f"{pool_id[:16]}",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, f"{pool_id[:16]}_grid.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--pools", type=str, default=None,
                        help="Comma-separated pool_id prefixes to run (default: all)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading panel and matching pools...")
    pools = load_panel_and_match(args.train_days)
    print(f"Found {len(pools)} matchable 2-token pools\n")

    for p in pools:
        # Preload price data to determine actual date coverage
        sorted_tokens = sorted(p["tokens"])
        price_data = get_historic_parquet_data(sorted_tokens, ["close"])
        p["price_data"] = price_data

        if len(price_data) == 0:
            p["panel_data"] = p["panel_data"].iloc[:0]  # empty
            p["lp_supply_df"] = None
            p["initial_tvl"] = 0.0
            p["start"], p["end"] = "2000-01-01", "2000-01-01"
            continue

        # Clip panel to price data's actual date range
        price_dates = pd.to_datetime(price_data.index, unit="ms")
        price_start = price_dates.min().normalize()
        price_end = price_dates.max().normalize()
        panel_data = p["panel_data"]
        panel_data = panel_data[
            (panel_data["obs_date"] >= price_start)
            & (panel_data["obs_date"] <= price_end)
        ].copy()
        p["panel_data"] = panel_data

        lp_df, tvl = build_lp_supply_df(panel_data)
        p["lp_supply_df"] = lp_df
        p["initial_tvl"] = tvl

        # Start/end must be midnight timestamps that exist in the price data.
        # start_and_end_calcs does an exact unix match and assumes alignment.
        # If the price data starts mid-day (e.g. COW at noon), advance to
        # the next midnight so the sim has a clean day boundary.
        if len(panel_data) > 0:
            first_midnight = (price_dates.min() + pd.Timedelta(days=1)).normalize()
            last_midnight = price_dates.max().normalize()
            first_midnight_ms = int(first_midnight.timestamp() * 1000)
            last_midnight_ms = int(last_midnight.timestamp() * 1000)
            # Verify these timestamps exist in the price data
            if first_midnight_ms in price_data.index and last_midnight_ms in price_data.index:
                p["start"] = first_midnight.strftime("%Y-%m-%d %H:%M:%S")
                p["end"] = last_midnight.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Fallback: use panel dates (works when price data covers full range)
                p["start"], p["end"] = get_date_range(panel_data)
        else:
            p["start"], p["end"] = "2000-01-01", "2000-01-01"

    pools = [p for p in pools if p["panel_data"]["obs_date"].nunique() >= 14]
    pools.sort(key=lambda p: p["initial_tvl"], reverse=True)

    # Filter to specific pools if requested
    if args.pools:
        requested = set(args.pools.split(","))
        pools = [p for p in pools if p["pool_id_prefix"] in requested]
        print(f"Filtered to {len(pools)} requested pools\n")

    # Print pool summary
    print(f"{'#':>3} {'Tokens':<12} {'Chain':<10} {'Type':<9} {'Weights':<10} "
          f"{'Fee':>6} {'TVL':>12} {'Days':>5} {'Pool ID':<18}")
    print("-" * 95)
    for i, p in enumerate(pools):
        n_days = p["panel_data"]["obs_date"].nunique()
        w_str = "/".join(f"{w:.0%}" for w in p["weights"]) if p["weights"] else "N/A"
        print(f"{i+1:3d} {'/'.join(p['tokens']):<12} {p['chain']:<10} "
              f"{p['pool_type']:<9} {w_str:<10} "
              f"{p['fee']:5.2%} ${p['initial_tvl']:>10,.0f} "
              f"{n_days:5d} {p['pool_id'][:16]}")

    all_summaries = []
    t_total = time.time()

    for pool_idx, pool in enumerate(pools):
        pool_id = pool["pool_id"]
        prefix = pool["pool_id_prefix"]
        tokens = pool["tokens"]
        chain = pool["chain"]
        fee = pool["fee"]
        tvl = pool["initial_tvl"]
        pool_type = pool["pool_type"]
        gas_costs = gas_costs_for_chain(chain)
        n_runs = len(CADENCES) * len(gas_costs)

        if tvl <= 0:
            print(f"\n  SKIP {'/'.join(tokens)} ({chain}): TVL=0")
            continue

        w_str = "/".join(f"{w:.0%}" for w in pool["weights"]) if pool["weights"] else "N/A"
        print(f"\n{'='*60}")
        print(f"  [{pool_idx+1}/{len(pools)}] {'/'.join(tokens)} "
              f"({chain}, {pool_type}, {w_str}, fee={fee:.2%}, TVL=${tvl:,.0f})")
        print(f"  Grid: {len(CADENCES)} cadences x {len(gas_costs)} gas = {n_runs} runs")
        print(f"{'='*60}")

        t0 = time.time()

        # Price data was preloaded during panel clipping
        price_data = pool["price_data"]

        pool_info = {
            "tokens": tokens,
            "fee": fee,
            "initial_tvl": tvl,
            "start": pool["start"],
            "end": pool["end"],
            "lp_supply_df": pool["lp_supply_df"],
            "weights": pool["weights"],
            "pool_type": pool_type,
            "reclamm_params": pool.get("reclamm_params"),
            # Only pass preloaded price_data in single-worker mode.
            # For multi-worker, each subprocess loads its own to avoid
            # pickling multi-million-row DataFrames across processes.
            "price_data": price_data if args.workers <= 1 else None,
        }

        all_daily_rows = []
        all_summary_rows = []

        if args.workers <= 1:
            for cadence in CADENCES:
                daily_rows, summary_rows = _run_cadence_sweep(
                    pool_info, cadence, gas_costs,
                )
                all_daily_rows.extend(daily_rows)
                all_summary_rows.extend(summary_rows)
                for r in summary_rows:
                    print(f"    cad={r['cadence']:3d} gas=${r['gas_cost']:6.3f} -> "
                          f"median=${r['median_daily_arb_volume']:,.0f}/day")
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for cadence in CADENCES:
                    fut = executor.submit(
                        _run_cadence_sweep, pool_info, cadence, gas_costs,
                    )
                    futures[fut] = cadence

                done = 0
                for fut in as_completed(futures):
                    cadence = futures[fut]
                    done += 1
                    try:
                        daily_rows, summary_rows = fut.result()
                        all_daily_rows.extend(daily_rows)
                        all_summary_rows.extend(summary_rows)
                        medians = [r["median_daily_arb_volume"] for r in summary_rows
                                   if not np.isnan(r.get("median_daily_arb_volume", np.nan))]
                        if medians:
                            print(f"    [{done:2d}/{len(CADENCES)}] cad={cadence:3d} — "
                                  f"V_arb: ${min(medians):,.0f} – ${max(medians):,.0f}/day")
                        else:
                            print(f"    [{done:2d}/{len(CADENCES)}] cad={cadence:3d} — "
                                  f"all failed")
                    except Exception as e:
                        print(f"    [{done:2d}/{len(CADENCES)}] cad={cadence:3d} FAILED: {e}")

        elapsed = time.time() - t0
        print(f"  {n_runs} runs in {elapsed:.1f}s ({elapsed/max(n_runs,1):.2f}s/run)")

        # Save per-day parquet
        if all_daily_rows:
            daily_df = pd.DataFrame(all_daily_rows)
            daily_df["date"] = pd.to_datetime(daily_df["date"])
            parquet_path = os.path.join(OUTPUT_DIR, f"{prefix}_daily.parquet")
            daily_df.to_parquet(parquet_path, index=False)
            print(f"  Saved {len(daily_df)} daily rows -> {parquet_path}")

        # Save summary CSV for diagnostics
        summary_df = pd.DataFrame(all_summary_rows)
        csv_path = os.path.join(OUTPUT_DIR, f"{prefix}_summary.csv")
        summary_df.to_csv(csv_path, index=False)

        # Plot
        if len(summary_df) > 0:
            plot_pool_grid(summary_df, pool_id, tokens, chain, fee, tvl,
                           pool_type, gas_costs, OUTPUT_DIR)

        # Global summary
        g0 = summary_df[summary_df["gas_cost"] == 0.0] if len(summary_df) > 0 else pd.DataFrame()
        all_summaries.append({
            "pool_id": prefix,
            "tokens": "/".join(tokens),
            "chain": chain,
            "pool_type": pool_type,
            "weights": str(pool["weights"]),
            "fee": fee,
            "tvl": tvl,
            "n_days": summary_df["n_days"].max() if len(summary_df) > 0 else 0,
            "n_daily_rows": len(all_daily_rows),
            "v_arb_cad1_gas0": g0[g0["cadence"] == 1]["median_daily_arb_volume"].values[0]
                if len(g0[g0["cadence"] == 1]) > 0 else np.nan,
            "v_arb_cad60_gas0": g0[g0["cadence"] == 60]["median_daily_arb_volume"].values[0]
                if len(g0[g0["cadence"] == 60]) > 0 else np.nan,
            "elapsed_s": elapsed,
        })

    total_elapsed = time.time() - t_total

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(OUTPUT_DIR, "grid_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"  Summary saved: {summary_path}")
        print(f"  Total: {len(all_summaries)} pools, "
              f"{total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
