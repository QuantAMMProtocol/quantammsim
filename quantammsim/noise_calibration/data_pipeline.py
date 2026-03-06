"""Data pipeline: fetch pools, snapshots, prices, and assemble panel."""

import json
import os
import time
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from .constants import BALANCER_API_URL, BALANCER_API_CHAINS
from .token_classification import classify_token_tier


def _graphql_request(query: dict, base_url: str = BALANCER_API_URL,
                     timeout: int = 30) -> dict:
    """Send a GraphQL request to the Balancer V3 API."""
    data = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(
        base_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "quantammsim/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def enumerate_balancer_pools(
    chains: list = None,
    pool_types: list = None,
    min_tvl: float = 10000.0,
) -> pd.DataFrame:
    """Enumerate all WEIGHTED + RECLAMM pools across chains from Balancer API."""
    if chains is None:
        chains = BALANCER_API_CHAINS
    if pool_types is None:
        pool_types = ["WEIGHTED", "RECLAMM"]

    all_pools = []
    for chain in chains:
        print(f"  Querying {chain}...", end=" ", flush=True)
        query = {
            "query": """
            query GetPools($chain: GqlChain!, $types: [GqlPoolType!],
                           $minTvl: Float) {
              poolGetPools(
                where: {
                  chainIn: [$chain]
                  poolTypeIn: $types
                  minTvl: $minTvl
                }
              ) {
                id
                chain
                type
                createTime
                protocolVersion
                poolTokens {
                  symbol
                  weight
                  address
                }
                dynamicData {
                  totalLiquidity
                  swapFee
                }
              }
            }
            """,
            "variables": {
                "chain": chain,
                "types": pool_types,
                "minTvl": min_tvl,
            },
        }

        try:
            body = _graphql_request(query)
            pools = body.get("data", {}).get("poolGetPools", [])
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        for p in pools:
            tokens = [t["symbol"] for t in p.get("poolTokens", [])]
            weights = [t.get("weight") for t in p.get("poolTokens", [])]
            token_addresses = [t.get("address", "") for t in p.get("poolTokens", [])]
            tvl = float(p.get("dynamicData", {}).get("totalLiquidity", 0))
            fee = float(p.get("dynamicData", {}).get("swapFee", 0))

            all_pools.append({
                "pool_id": p["id"],
                "chain": p["chain"],
                "pool_type": p["type"],
                "protocol_version": p.get("protocolVersion", 0),
                "tokens": tokens,
                "token_addresses": token_addresses,
                "weights": weights,
                "swap_fee": fee,
                "create_time": p.get("createTime", 0),
                "current_tvl": tvl,
            })

        print(f"{len(pools)} pools")
        time.sleep(0.3)

    df = pd.DataFrame(all_pools)
    print(f"\n  Total: {len(df)} pools across {len(chains)} chains")
    return df


def fetch_pool_snapshots(pool_id: str, chain: str,
                         base_url: str = BALANCER_API_URL) -> pd.DataFrame:
    """Fetch ALL_TIME daily snapshots for a single pool."""
    query = {
        "query": """
        query GetSnapshots($poolId: String!, $chain: GqlChain!,
                           $range: GqlPoolSnapshotDataRange!) {
          poolGetSnapshots(id: $poolId, chain: $chain, range: $range) {
            timestamp
            volume24h
            totalLiquidity
            totalShares
          }
        }
        """,
        "variables": {
            "poolId": pool_id,
            "chain": chain,
            "range": "ALL_TIME",
        },
    }

    body = _graphql_request(query)
    snapshots = body.get("data", {}).get("poolGetSnapshots", [])

    if not snapshots:
        return pd.DataFrame(columns=["timestamp", "volume_usd",
                                      "total_liquidity_usd", "total_shares"])

    records = []
    for snap in snapshots:
        records.append({
            "timestamp": int(snap["timestamp"]),
            "volume_usd": float(snap["volume24h"]),
            "total_liquidity_usd": float(snap["totalLiquidity"]),
            "total_shares": float(snap.get("totalShares", 0)),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    df = df.sort_values("timestamp").drop_duplicates("date", keep="last")
    return df


def fetch_all_snapshots(pools_df: pd.DataFrame,
                        cache_path: str = None) -> pd.DataFrame:
    """Fetch daily snapshots for all pools, with caching."""
    cached = pd.DataFrame()
    cached_pool_ids = set()
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        cached_pool_ids = set(cached["pool_id"].unique())
        print(f"  Cache has {len(cached_pool_ids)} pools, "
              f"{len(cached)} pool-days")

    if len(pools_df) == 0:
        print("  No pools to fetch.")
        return cached if len(cached) > 0 else pd.DataFrame(
            columns=["pool_id", "chain", "date", "volume_usd",
                     "total_liquidity_usd", "total_shares"]
        )
    to_fetch = pools_df[~pools_df["pool_id"].isin(cached_pool_ids)]
    print(f"  Need to fetch {len(to_fetch)} new pools")

    new_records = []
    for i, (_, pool) in enumerate(to_fetch.iterrows()):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Fetching {i+1}/{len(to_fetch)}: {pool['pool_id'][:10]}... "
                  f"({pool['chain']})", flush=True)
        try:
            snap_df = fetch_pool_snapshots(pool["pool_id"], pool["chain"])
            if len(snap_df) > 0:
                snap_df["pool_id"] = pool["pool_id"]
                snap_df["chain"] = pool["chain"]
                cols = ["pool_id", "chain", "date", "volume_usd",
                        "total_liquidity_usd"]
                if "total_shares" in snap_df.columns:
                    cols.append("total_shares")
                new_records.append(snap_df[cols])
        except Exception as e:
            print(f"    FAILED {pool['pool_id'][:10]}: {e}")
        time.sleep(0.5)

    if new_records:
        new_df = pd.concat(new_records, ignore_index=True)
        combined = pd.concat([cached, new_df], ignore_index=True)
    else:
        combined = cached

    if cache_path and len(combined) > 0:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        combined.to_parquet(cache_path, index=False)
        print(f"  Saved cache: {len(combined)} pool-days -> {cache_path}")

    return combined


def fetch_token_prices(token_addresses_by_chain: dict,
                       cache_dir: str = None) -> dict:
    """Fetch hourly token prices from Balancer API."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    prices = {}

    for chain, tokens in token_addresses_by_chain.items():
        uncached = {}
        for symbol, address in tokens.items():
            cache_key = f"{chain}_{symbol}".replace("/", "_")
            cp = os.path.join(cache_dir, f"{cache_key}.parquet") if cache_dir else None

            if cp and os.path.exists(cp):
                prices[(chain, symbol)] = pd.read_parquet(cp)
            else:
                uncached[symbol] = address

        if not uncached:
            continue

        addr_to_symbol = {addr: sym for sym, addr in uncached.items()}
        addresses = list(uncached.values())

        print(f"    Fetching {len(addresses)} prices on {chain}...", flush=True)

        batch_size = 20
        for batch_start in range(0, len(addresses), batch_size):
            batch_addrs = addresses[batch_start:batch_start + batch_size]
            query = {
                "query": """
                query GetPrices($chain: GqlChain!, $addresses: [String!]!,
                                $range: GqlTokenChartDataRange!) {
                  tokenGetHistoricalPrices(
                    addresses: $addresses, chain: $chain, range: $range
                  ) {
                    address
                    prices {
                      timestamp
                      price
                    }
                  }
                }
                """,
                "variables": {
                    "chain": chain,
                    "addresses": batch_addrs,
                    "range": "ONE_YEAR",
                },
            }

            try:
                body = _graphql_request(query, timeout=60)
                results = body.get("data", {}).get(
                    "tokenGetHistoricalPrices", [])
                for result in results:
                    addr = result.get("address", "")
                    price_list = result.get("prices", [])
                    symbol = addr_to_symbol.get(addr)
                    if symbol and price_list:
                        pdf = pd.DataFrame(price_list)
                        pdf["timestamp"] = pdf["timestamp"].astype(int)
                        pdf["price"] = pdf["price"].astype(float)
                        prices[(chain, symbol)] = pdf
                        if cache_dir:
                            cache_key = f"{chain}_{symbol}".replace("/", "_")
                            cp = os.path.join(cache_dir, f"{cache_key}.parquet")
                            pdf.to_parquet(cp, index=False)
            except Exception as e:
                print(f"    FAILED batch on {chain}: {e}")

            time.sleep(0.5)

    print(f"  Got prices for {len(prices)} token-chain pairs")
    return prices


def compute_pair_volatility(
    snapshots_df: pd.DataFrame,
    pool_row: pd.Series,
    token_prices: dict,
) -> pd.Series:
    """Compute daily annualised volatility for a pool's pair ratio."""
    tokens = pool_row["tokens"]
    chain = pool_row["chain"]

    if len(tokens) < 2:
        return pd.Series(dtype=float)

    def _get_price_df(symbol):
        key = (chain, symbol)
        if key in token_prices:
            return token_prices[key]
        for k, v in token_prices.items():
            if k[1] == symbol:
                return v
        return None

    p0_df = _get_price_df(tokens[0])
    p1_df = _get_price_df(tokens[1])

    stables = {"USDC", "USDT", "DAI", "LUSD", "GHO", "crvUSD", "sDAI",
               "WXDAI", "xDAI", "USDC.e", "USDbC"}

    if tokens[0] in stables and tokens[1] in stables:
        dates = snapshots_df["date"].unique()
        return pd.Series(0.01, index=dates)

    if p0_df is None and tokens[0] not in stables:
        dates = snapshots_df["date"].unique()
        return pd.Series(0.5, index=dates)
    if p1_df is None and tokens[1] not in stables:
        dates = snapshots_df["date"].unique()
        return pd.Series(0.5, index=dates)

    if tokens[0] in stables:
        if p1_df is None or len(p1_df) == 0:
            dates = snapshots_df["date"].unique()
            return pd.Series(0.5, index=dates)
        ratio_df = p1_df.copy()
        ratio_df["ratio"] = 1.0 / ratio_df["price"]
    elif tokens[1] in stables:
        if p0_df is None or len(p0_df) == 0:
            dates = snapshots_df["date"].unique()
            return pd.Series(0.5, index=dates)
        ratio_df = p0_df.copy()
        ratio_df["ratio"] = ratio_df["price"]
    else:
        if p0_df is None or p1_df is None or len(p0_df) == 0 or len(p1_df) == 0:
            dates = snapshots_df["date"].unique()
            return pd.Series(0.5, index=dates)
        merged = pd.merge_asof(
            p0_df.sort_values("timestamp"),
            p1_df.sort_values("timestamp"),
            on="timestamp",
            suffixes=("_0", "_1"),
            tolerance=7200,
        ).dropna()
        if len(merged) == 0:
            dates = snapshots_df["date"].unique()
            return pd.Series(0.5, index=dates)
        ratio_df = merged.copy()
        ratio_df["ratio"] = merged["price_0"] / merged["price_1"]

    ratio_df["datetime"] = pd.to_datetime(ratio_df["timestamp"], unit="s")
    ratio_df["date"] = ratio_df["datetime"].dt.date
    ratio_df = ratio_df.sort_values("timestamp")

    ratio_df["log_return"] = np.log(
        ratio_df["ratio"] / ratio_df["ratio"].shift(1)
    )
    ratio_df = ratio_df.dropna(subset=["log_return"])

    daily_vol = ratio_df.groupby("date")["log_return"].std()
    daily_vol_ann = daily_vol * np.sqrt(24 * 365)

    return daily_vol_ann


def assemble_panel(
    pools_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    token_prices: dict,
) -> pd.DataFrame:
    """Assemble the full panel DataFrame with lagged TVL.

    Adds log_tvl_lag1 = per-pool shift(1) of log_tvl to break
    the TVL-volume simultaneity bias. Drops the first observation
    per pool (~1 obs per pool).
    """
    records = []
    pool_ids = snapshots_df["pool_id"].unique()
    n_pools = len(pool_ids)

    # Track volatility fallback rate
    n_obs_total = 0
    n_obs_fallback = 0
    pools_all_fallback = []  # pools where every obs hit fallback

    for i, pool_id in enumerate(pool_ids):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Assembling {i+1}/{n_pools}...", flush=True)

        pool_snaps = snapshots_df[snapshots_df["pool_id"] == pool_id]
        pool_meta = pools_df[pools_df["pool_id"] == pool_id]
        if len(pool_meta) == 0:
            continue
        pool_row = pool_meta.iloc[0]

        tokens = pool_row["tokens"]
        if len(tokens) < 2:
            continue

        chain = pool_row["chain"]
        swap_fee = pool_row["swap_fee"]

        tiers = sorted([classify_token_tier(t) for t in tokens])
        tier_a = tiers[0]
        tier_b = tiers[1] if len(tiers) > 1 else tiers[0]

        vol_series = compute_pair_volatility(pool_snaps, pool_row, token_prices)

        pool_obs = 0
        pool_fallback = 0
        has_shares = "total_shares" in pool_snaps.columns
        for _, snap in pool_snaps.iterrows():
            date = snap["date"]
            volume = snap["volume_usd"]
            tvl = snap["total_liquidity_usd"]
            shares = float(snap["total_shares"]) if has_shares else 0.0

            if tvl <= 0 or volume <= 0:
                continue

            used_fallback = False
            if isinstance(vol_series, pd.Series) and date in vol_series.index:
                vol = vol_series[date]
            else:
                vol = 0.5
                used_fallback = True

            if not np.isfinite(vol) or vol <= 0:
                vol = 0.5
                used_fallback = True

            n_obs_total += 1
            if used_fallback:
                n_obs_fallback += 1
                pool_fallback += 1
            pool_obs += 1

            if isinstance(date, datetime):
                is_weekend = date.weekday() >= 5
            else:
                is_weekend = pd.Timestamp(date).weekday() >= 5

            # DOW harmonics (deterministic from date)
            if isinstance(date, datetime):
                dow = date.weekday()
            else:
                dow = pd.Timestamp(date).weekday()
            dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
            dow_cos = np.cos(2.0 * np.pi * dow / 7.0)

            record = {
                "pool_id": pool_id,
                "chain": chain,
                "date": date,
                "log_volume": np.log(volume),
                "log_tvl": np.log(tvl),
                "volatility": vol,
                "log_sigma": np.log(max(vol, 1e-6)),
                "weekend": 1.0 if is_weekend else 0.0,
                "log_fee": np.log(max(swap_fee, 1e-6)),
                "dow_sin": dow_sin,
                "dow_cos": dow_cos,
                "tier_A": tier_a,
                "tier_B": tier_b,
                "tokens": ",".join(tokens[:2]),
                "swap_fee": swap_fee,
            }
            if shares > 0:
                record["total_shares"] = shares
            records.append(record)

        if pool_obs > 0 and pool_fallback == pool_obs:
            pools_all_fallback.append(
                (pool_id[:16], chain, ",".join(tokens[:2]))
            )

    panel = pd.DataFrame(records)

    # Add lagged TVL to break simultaneity bias
    panel = panel.sort_values(["pool_id", "date"]).reset_index(drop=True)
    panel["log_tvl_lag1"] = panel.groupby("pool_id")["log_tvl"].shift(1)
    n_before = len(panel)
    panel = panel.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)
    n_dropped = n_before - len(panel)

    # Interaction terms (use lagged TVL to break simultaneity)
    panel["tvl_x_sigma"] = panel["log_tvl_lag1"] * panel["log_sigma"]
    panel["tvl_x_fee"] = panel["log_tvl_lag1"] * panel["log_fee"]
    panel["sigma_x_fee"] = panel["log_sigma"] * panel["log_fee"]

    print(f"\n  Panel: {len(panel)} observations, "
          f"{panel['pool_id'].nunique()} pools, "
          f"{panel['chain'].nunique()} chains")
    print(f"  Dropped {n_dropped} first-day obs for lagged TVL")

    # Volatility coverage report
    if n_obs_total > 0:
        pct = 100 * n_obs_fallback / n_obs_total
        print(f"\n  Volatility coverage:")
        print(f"    {n_obs_fallback}/{n_obs_total} obs used fallback "
              f"vol=0.5 ({pct:.1f}%)")
        print(f"    {len(pools_all_fallback)} pools had 100% fallback")
        if pools_all_fallback:
            for pid, ch, toks in pools_all_fallback[:10]:
                print(f"      {pid}... ({ch}) {toks}")
            if len(pools_all_fallback) > 10:
                print(f"      ... and {len(pools_all_fallback) - 10} more")

    return panel
