"""Bayesian hierarchical noise volume model across Balancer WEIGHTED + RECLAMM pools.

Pools data cross-sectionally across all Balancer weighted/reCLAMM pools,
fits a Bayesian hierarchical model where pool covariates (chain, token tier,
fee) modulate all coefficients via group-level regression, with full
posterior inference via NumPyro.

Model:
    Hyperpriors:
        Φ ~ Normal(0, 2)                   (K × 3) group-level regression
        σ_θ ~ HalfNormal(2)                (3,) per-coefficient scales
        L_ω ~ LKJCholesky(3, η=2)          correlation structure
        β_weekend ~ Normal(0, 2)           shared nuisance
        σ_ε ~ HalfNormal(3)               observation noise

    For each pool i:
        x_i = [1, chain_dummies, tier_dummies, log_fee]  (K,) covariates
        z_i ~ N(0, I₃)                                    non-centered
        θ_i = Φᵀx_i + diag(σ_θ)·L_ω·z_i                 (α_i, β_tvl_i, β_vol_i)

    For each observation (i, t):
        log(V) ~ N(α_i + β_tvl_i·log_tvl + β_vol_i·vol + β_weekend·weekend, σ²_ε)

Usage:
    # Full pipeline: fetch data + fit model + output
    python scripts/calibrate_noise_hierarchical.py \\
        --fetch --fit --output results/hierarchical_noise_params.json --plot

    # Use cached data, re-fit only
    python scripts/calibrate_noise_hierarchical.py \\
        --fit --output results/hierarchical_noise_params.json

    # Predict for a new pool
    python scripts/calibrate_noise_hierarchical.py \\
        --predict --chain BASE --tokens ETH BTC --fee 0.003

    # Use NUTS instead of SVI
    python scripts/calibrate_noise_hierarchical.py \\
        --fit --nuts --output results/hierarchical_noise_params.json
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, MCMC, NUTS, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal

numpyro.enable_x64()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BALANCER_API_URL = "https://api-v3.balancer.fi/"

BALANCER_API_CHAINS = [
    "MAINNET", "POLYGON", "ARBITRUM", "GNOSIS", "BASE", "SONIC", "OPTIMISM",
    "AVALANCHE",
]

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "local_data", "noise_calibration"
)

# ---------------------------------------------------------------------------
# Token tier classification
# ---------------------------------------------------------------------------

# Tier 0: blue-chip — top by volume, wrapped native, major stables
_TIER_0 = {
    "ETH", "WETH", "BTC", "WBTC", "cbBTC", "USDC", "USDT", "DAI",
    "wstETH", "stETH", "rETH", "cbETH", "WMATIC", "MATIC", "POL",
    "WAVAX", "AVAX", "GNO", "WXDAI", "xDAI",
    "S", "wS",  # Sonic native
}

# Tier 1: mid-cap DeFi blue-chips (approx CoinGecko rank < 200)
_TIER_1 = {
    "AAVE", "LINK", "UNI", "BAL", "MKR", "CRV", "COMP", "SNX",
    "LDO", "RPL", "SUSHI", "YFI", "1INCH", "ENS", "DYDX",
    "FXS", "FRAX", "LUSD", "sDAI", "GHO", "crvUSD",
    "ARB", "OP", "PENDLE", "ENA", "EIGEN",
    "SAFE", "COW",
}


def _normalise_symbol(symbol: str) -> str:
    """Normalise wrapped/bridged variants to canonical form."""
    s = symbol.strip()
    # Common wrapped → unwrapped
    mapping = {
        "WETH": "WETH",  # keep WETH as-is (it's in tier 0)
        "WBTC": "WBTC",
        "cbBTC": "cbBTC",
        "WMATIC": "WMATIC",
        "WAVAX": "WAVAX",
        "WXDAI": "WXDAI",
        "wS": "wS",
    }
    return mapping.get(s, s)


def classify_token_tier(symbol: str) -> int:
    """Classify a token symbol into tier 0/1/2.

    Returns
    -------
    int
        0 = blue-chip, 1 = mid-cap, 2 = long-tail
    """
    s = _normalise_symbol(symbol)
    if s in _TIER_0:
        return 0
    if s in _TIER_1:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Phase 1: API data ingestion
# ---------------------------------------------------------------------------

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
    """Enumerate all WEIGHTED + RECLAMM pools across chains from Balancer API.

    Parameters
    ----------
    chains : list of str
        API chain identifiers (e.g. ["MAINNET", "BASE"]).
    pool_types : list of str
        Pool type filters (e.g. ["WEIGHTED", "STABLE"]).
    min_tvl : float
        Minimum TVL in USD to include.

    Returns
    -------
    pd.DataFrame
        Columns: pool_id, chain, pool_type, tokens (list of symbols),
        swap_fee, create_time, dynamic_data_tvl.
    """
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
    """Fetch ALL_TIME daily snapshots for a single pool.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, volume_usd, total_liquidity_usd.
    """
    query = {
        "query": """
        query GetSnapshots($poolId: String!, $chain: GqlChain!,
                           $range: GqlPoolSnapshotDataRange!) {
          poolGetSnapshots(id: $poolId, chain: $chain, range: $range) {
            timestamp
            volume24h
            totalLiquidity
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
        return pd.DataFrame(columns=["timestamp", "volume_usd", "total_liquidity_usd"])

    records = []
    for snap in snapshots:
        records.append({
            "timestamp": int(snap["timestamp"]),
            "volume_usd": float(snap["volume24h"]),
            "total_liquidity_usd": float(snap["totalLiquidity"]),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    # Deduplicate by date (keep last snapshot per day)
    df = df.sort_values("timestamp").drop_duplicates("date", keep="last")
    return df


def fetch_all_snapshots(pools_df: pd.DataFrame,
                        cache_path: str = None) -> pd.DataFrame:
    """Fetch daily snapshots for all pools, with caching.

    Parameters
    ----------
    pools_df : pd.DataFrame
        Pool enumeration from enumerate_balancer_pools.
    cache_path : str, optional
        Path to parquet cache. If it exists, only fetch missing pools.

    Returns
    -------
    pd.DataFrame
        Panel with columns: pool_id, chain, date, volume_usd,
        total_liquidity_usd.
    """
    # Load cache if exists
    cached = pd.DataFrame()
    cached_pool_ids = set()
    if cache_path and os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        cached_pool_ids = set(cached["pool_id"].unique())
        print(f"  Cache has {len(cached_pool_ids)} pools, "
              f"{len(cached)} pool-days")

    # Determine which pools need fetching
    if len(pools_df) == 0:
        print("  No pools to fetch.")
        return cached if len(cached) > 0 else pd.DataFrame(
            columns=["pool_id", "chain", "date", "volume_usd",
                     "total_liquidity_usd"]
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
                new_records.append(snap_df[
                    ["pool_id", "chain", "date", "volume_usd",
                     "total_liquidity_usd"]
                ])
        except Exception as e:
            print(f"    FAILED {pool['pool_id'][:10]}: {e}")
        time.sleep(0.5)  # Rate limit

    if new_records:
        new_df = pd.concat(new_records, ignore_index=True)
        combined = pd.concat([cached, new_df], ignore_index=True)
    else:
        combined = cached

    # Save cache
    if cache_path and len(combined) > 0:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        combined.to_parquet(cache_path, index=False)
        print(f"  Saved cache: {len(combined)} pool-days → {cache_path}")

    return combined


def fetch_token_prices(token_addresses_by_chain: dict,
                       cache_dir: str = None) -> dict:
    """Fetch hourly token prices from Balancer API.

    Parameters
    ----------
    token_addresses_by_chain : dict
        {chain: {symbol: address, ...}, ...}
    cache_dir : str, optional
        Directory for per-token price caches.

    Returns
    -------
    dict
        {(chain, symbol): pd.DataFrame with columns [timestamp, price], ...}
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    prices = {}

    for chain, tokens in token_addresses_by_chain.items():
        # Check cache first, collect uncached addresses
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

        # Batch fetch: API supports multiple addresses per request
        addr_to_symbol = {addr: sym for sym, addr in uncached.items()}
        addresses = list(uncached.values())

        print(f"    Fetching {len(addresses)} prices on {chain}...",
              flush=True)

        # Batch in groups of 20 to avoid oversized requests
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
    """Compute daily annualised volatility for a pool's pair ratio.

    Uses hourly prices from the API to compute daily realised volatility.
    Falls back to a default of 0.5 if price data is insufficient.

    Parameters
    ----------
    snapshots_df : pd.DataFrame
        Pool's daily snapshots (need dates).
    pool_row : pd.Series
        Pool metadata row (need tokens, chain).
    token_prices : dict
        {(chain, symbol): DataFrame, ...}

    Returns
    -------
    pd.Series
        Indexed by date, values are annualised daily volatility.
    """
    tokens = pool_row["tokens"]
    chain = pool_row["chain"]

    if len(tokens) < 2:
        return pd.Series(dtype=float)

    # Get price series for token[0] and token[1]
    # Try chain-specific first, then any chain
    def _get_price_df(symbol):
        # Exact match
        key = (chain, symbol)
        if key in token_prices:
            return token_prices[key]
        # Any chain
        for k, v in token_prices.items():
            if k[1] == symbol:
                return v
        return None

    p0_df = _get_price_df(tokens[0])
    p1_df = _get_price_df(tokens[1])

    # If either is a stablecoin, use $1
    stables = {"USDC", "USDT", "DAI", "LUSD", "GHO", "crvUSD", "sDAI",
               "WXDAI", "xDAI", "USDC.e", "USDbC"}

    if tokens[0] in stables and tokens[1] in stables:
        # Stable-stable pair: near-zero vol
        dates = snapshots_df["date"].unique()
        return pd.Series(0.01, index=dates)

    if p0_df is None and tokens[0] not in stables:
        dates = snapshots_df["date"].unique()
        return pd.Series(0.5, index=dates)  # fallback
    if p1_df is None and tokens[1] not in stables:
        dates = snapshots_df["date"].unique()
        return pd.Series(0.5, index=dates)  # fallback

    # Build hourly price ratio
    if tokens[0] in stables:
        # ratio = 1 / p1
        if p1_df is None or len(p1_df) == 0:
            dates = snapshots_df["date"].unique()
            return pd.Series(0.5, index=dates)
        ratio_df = p1_df.copy()
        ratio_df["ratio"] = 1.0 / ratio_df["price"]
    elif tokens[1] in stables:
        # ratio = p0
        if p0_df is None or len(p0_df) == 0:
            dates = snapshots_df["date"].unique()
            return pd.Series(0.5, index=dates)
        ratio_df = p0_df.copy()
        ratio_df["ratio"] = ratio_df["price"]
    else:
        # Both non-stable: ratio = p0/p1
        if p0_df is None or p1_df is None or len(p0_df) == 0 or len(p1_df) == 0:
            dates = snapshots_df["date"].unique()
            return pd.Series(0.5, index=dates)
        # Merge on nearest timestamp
        merged = pd.merge_asof(
            p0_df.sort_values("timestamp"),
            p1_df.sort_values("timestamp"),
            on="timestamp",
            suffixes=("_0", "_1"),
            tolerance=7200,  # 2 hour tolerance
        ).dropna()
        if len(merged) == 0:
            dates = snapshots_df["date"].unique()
            return pd.Series(0.5, index=dates)
        ratio_df = merged.copy()
        ratio_df["ratio"] = merged["price_0"] / merged["price_1"]

    ratio_df["datetime"] = pd.to_datetime(ratio_df["timestamp"], unit="s")
    ratio_df["date"] = ratio_df["datetime"].dt.date
    ratio_df = ratio_df.sort_values("timestamp")

    # Log returns
    ratio_df["log_return"] = np.log(
        ratio_df["ratio"] / ratio_df["ratio"].shift(1)
    )
    ratio_df = ratio_df.dropna(subset=["log_return"])

    # Daily vol from hourly returns, annualised
    daily_vol = ratio_df.groupby("date")["log_return"].std()
    # Hourly data → ~24 returns/day. Annualise: σ_daily * sqrt(365)
    # But std() already gives daily std from hourly returns, so:
    # σ_annual = σ_hourly * sqrt(24 * 365)
    daily_vol_ann = daily_vol * np.sqrt(24 * 365)

    return daily_vol_ann


def assemble_panel(
    pools_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    token_prices: dict,
) -> pd.DataFrame:
    """Assemble the full panel DataFrame for hierarchical estimation.

    Parameters
    ----------
    pools_df : pd.DataFrame
        Pool enumeration from enumerate_balancer_pools.
    snapshots_df : pd.DataFrame
        Daily snapshots from fetch_all_snapshots.
    token_prices : dict
        Token prices from fetch_token_prices.

    Returns
    -------
    pd.DataFrame
        Panel with columns: pool_id, chain, date, log_volume, log_tvl,
        volatility, weekend, log_fee, tier_A, tier_B, tokens.
    """
    records = []
    pool_ids = snapshots_df["pool_id"].unique()
    n_pools = len(pool_ids)

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

        # Token tiers: sort by tier (best tier first)
        tiers = sorted([classify_token_tier(t) for t in tokens])
        tier_a = tiers[0]  # best (lowest) tier
        tier_b = tiers[1] if len(tiers) > 1 else tiers[0]

        # Compute volatility for this pool's pair
        vol_series = compute_pair_volatility(pool_snaps, pool_row, token_prices)

        for _, snap in pool_snaps.iterrows():
            date = snap["date"]
            volume = snap["volume_usd"]
            tvl = snap["total_liquidity_usd"]

            # Skip zero/negative TVL or volume
            if tvl <= 0 or volume <= 0:
                continue

            # Volatility lookup
            if isinstance(vol_series, pd.Series) and date in vol_series.index:
                vol = vol_series[date]
            else:
                vol = 0.5  # fallback

            if not np.isfinite(vol) or vol <= 0:
                vol = 0.5

            # Weekend indicator
            if isinstance(date, datetime):
                is_weekend = date.weekday() >= 5
            else:
                is_weekend = pd.Timestamp(date).weekday() >= 5

            records.append({
                "pool_id": pool_id,
                "chain": chain,
                "date": date,
                "log_volume": np.log(volume),
                "log_tvl": np.log(tvl),
                "volatility": vol,
                "weekend": 1.0 if is_weekend else 0.0,
                "log_fee": np.log(max(swap_fee, 1e-6)),
                "tier_A": tier_a,
                "tier_B": tier_b,
                "tokens": ",".join(tokens[:2]),
                "swap_fee": swap_fee,
            })

    panel = pd.DataFrame(records)
    print(f"\n  Panel: {len(panel)} observations, "
          f"{panel['pool_id'].nunique()} pools, "
          f"{panel['chain'].nunique()} chains")

    return panel


# ---------------------------------------------------------------------------
# Phase 2: Bayesian hierarchical model
# ---------------------------------------------------------------------------

def _encode_covariates(panel: pd.DataFrame) -> dict:
    """Build NumPyro-ready arrays from the panel DataFrame.

    Returns
    -------
    dict with keys:
        pool_idx : (N_obs,) int array mapping each observation to its pool
        X_pool : (N_pools, K) covariate matrix (intercept + dummies + log_fee)
        log_tvl, volatility, weekend, log_volume : (N_obs,) float arrays
        pool_ids : (N_pools,) pool ID strings
        covariate_names : list of str, column names for X_pool
        ref_chain, ref_tier_a, ref_tier_b : reference categories
        chains : sorted list of all chains
        pool_meta : DataFrame of per-pool metadata
    """
    pool_meta = panel.drop_duplicates("pool_id").reset_index(drop=True)
    pool_ids = pool_meta["pool_id"].values
    pool_id_to_idx = {pid: i for i, pid in enumerate(pool_ids)}

    pool_idx = panel["pool_id"].map(pool_id_to_idx).values

    # Build X_pool columns
    chains = sorted(panel["chain"].unique())
    ref_chain = chains[0]
    chain_cols = []
    chain_names = []
    for c in chains[1:]:
        chain_cols.append((pool_meta["chain"] == c).astype(float).values)
        chain_names.append(f"chain_{c}")

    tier_a_vals = sorted(pool_meta["tier_A"].astype(str).unique())
    ref_tier_a = tier_a_vals[0]
    tier_a_cols = []
    tier_a_names = []
    for t in tier_a_vals[1:]:
        tier_a_cols.append(
            (pool_meta["tier_A"].astype(str) == t).astype(float).values
        )
        tier_a_names.append(f"tier_A_{t}")

    tier_b_vals = sorted(pool_meta["tier_B"].astype(str).unique())
    ref_tier_b = tier_b_vals[0]
    tier_b_cols = []
    tier_b_names = []
    for t in tier_b_vals[1:]:
        tier_b_cols.append(
            (pool_meta["tier_B"].astype(str) == t).astype(float).values
        )
        tier_b_names.append(f"tier_B_{t}")

    N_pools = len(pool_ids)
    columns = [np.ones((N_pools, 1))]
    col_names = ["intercept"]

    for arr, name in zip(chain_cols, chain_names):
        columns.append(arr.reshape(-1, 1))
        col_names.append(name)
    for arr, name in zip(tier_a_cols, tier_a_names):
        columns.append(arr.reshape(-1, 1))
        col_names.append(name)
    for arr, name in zip(tier_b_cols, tier_b_names):
        columns.append(arr.reshape(-1, 1))
        col_names.append(name)
    columns.append(pool_meta["log_fee"].values.reshape(-1, 1))
    col_names.append("log_fee")

    X_pool = np.hstack(columns)

    return {
        "pool_idx": pool_idx.astype(np.int32),
        "X_pool": X_pool.astype(np.float64),
        "log_tvl": panel["log_tvl"].values.astype(np.float64),
        "volatility": panel["volatility"].values.astype(np.float64),
        "weekend": panel["weekend"].values.astype(np.float64),
        "log_volume": panel["log_volume"].values.astype(np.float64),
        "pool_ids": pool_ids,
        "covariate_names": col_names,
        "ref_chain": ref_chain,
        "ref_tier_a": ref_tier_a,
        "ref_tier_b": ref_tier_b,
        "chains": chains,
        "pool_meta": pool_meta,
    }


def _hierarchical_noise_model(
    pool_idx, X_pool, log_tvl, volatility, weekend, log_volume=None,
):
    """NumPyro model: Bayesian hierarchical loglinear noise volume.

    All pool covariates modulate all three coefficients (α, β_tvl, β_vol)
    through the group-level regression matrix Φ, with correlated random
    effects via LKJ-Cholesky.
    """
    N_pools = X_pool.shape[0]
    K = X_pool.shape[1]

    # Hyperpriors
    Phi = numpyro.sample("Phi", dist.Normal(0, 2).expand([K, 3]).to_event(2))
    sigma_theta = numpyro.sample(
        "sigma_theta", dist.HalfNormal(2).expand([3]).to_event(1)
    )
    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(3, concentration=2))
    beta_weekend = numpyro.sample("beta_weekend", dist.Normal(0, 2))
    sigma_eps = numpyro.sample("sigma_eps", dist.HalfNormal(3))

    # Non-centered pool random effects
    with numpyro.plate("pools", N_pools):
        z = numpyro.sample("z", dist.Normal(jnp.zeros(3), 1).to_event(1))

    # θ_i = Φᵀx_i + diag(σ_θ)·L_ω·z_i
    mu = X_pool @ Phi                                    # (N_pools, 3)
    L_Sigma = sigma_theta[:, None] * L_omega             # (3, 3)
    theta = mu + z @ L_Sigma.T                           # (N_pools, 3)

    alpha = theta[:, 0]
    beta_tvl = theta[:, 1]
    beta_vol = theta[:, 2]

    # Observation model
    loc = (alpha[pool_idx]
           + beta_tvl[pool_idx] * log_tvl
           + beta_vol[pool_idx] * volatility
           + beta_weekend * weekend)

    with numpyro.plate("obs", pool_idx.shape[0]):
        numpyro.sample("log_volume", dist.Normal(loc, sigma_eps), obs=log_volume)


def fit_bayesian_model(
    panel: pd.DataFrame, use_nuts: bool = False,
) -> tuple:
    """Fit the Bayesian hierarchical model via SVI or NUTS.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel from assemble_panel.
    use_nuts : bool
        If True, use NUTS MCMC (slower, exact). Otherwise SVI with
        AutoMultivariateNormal guide.

    Returns
    -------
    samples : dict
        Posterior samples keyed by parameter name.
    encoding : dict
        From _encode_covariates (needed downstream).
    """
    encoding = _encode_covariates(panel)

    model_kwargs = dict(
        pool_idx=jnp.array(encoding["pool_idx"]),
        X_pool=jnp.array(encoding["X_pool"]),
        log_tvl=jnp.array(encoding["log_tvl"]),
        volatility=jnp.array(encoding["volatility"]),
        weekend=jnp.array(encoding["weekend"]),
        log_volume=jnp.array(encoding["log_volume"]),
    )

    N_pools = encoding["X_pool"].shape[0]
    K = encoding["X_pool"].shape[1]
    print(f"    N obs = {len(encoding['pool_idx'])}, "
          f"N pools = {N_pools}, K covariates = {K}")
    print(f"    Covariates: {encoding['covariate_names']}")

    rng_key = jax.random.PRNGKey(0)

    if use_nuts:
        print("  Running NUTS (500 warmup + 1000 samples)...")
        kernel = NUTS(_hierarchical_noise_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
        mcmc.run(rng_key, **model_kwargs)
        samples = mcmc.get_samples()
        print("  NUTS complete.")
    else:
        print("  Running SVI with AutoMultivariateNormal (20k steps)...")
        guide = AutoMultivariateNormal(_hierarchical_noise_model)
        optimizer = numpyro.optim.Adam(1e-3)
        svi = SVI(_hierarchical_noise_model, guide, optimizer,
                   loss=Trace_ELBO())
        svi_result = svi.run(rng_key, 20_000, **model_kwargs)
        print(f"  SVI complete. Final ELBO loss: {svi_result.losses[-1]:.2f}")

        predictive = Predictive(
            guide, params=svi_result.params, num_samples=1000,
        )
        samples = predictive(jax.random.PRNGKey(1), **model_kwargs)
        print("  Drew 1000 posterior samples.")

    return samples, encoding


def extract_posteriors(samples: dict, encoding: dict) -> dict:
    """Reconstruct pool-specific coefficients from posterior samples.

    Computes θ_i = Φᵀx_i + diag(σ_θ)·L_ω·z_i for each posterior draw,
    then returns posterior means and variance components.

    Returns
    -------
    dict with keys:
        pool_effects : {pool_id: {alpha, beta_tvl, beta_vol}}
        Phi_mean : (K, 3) array
        sigma_theta_mean : (3,) array
        correlation_matrix : (3, 3) array
        beta_weekend_mean : float
        sigma_eps_mean : float
        theta_samples : (S, N_pools, 3) array (for diagnostics)
    """
    Phi = np.array(samples["Phi"])                # (S, K, 3)
    sigma_theta = np.array(samples["sigma_theta"])  # (S, 3)
    L_omega = np.array(samples["L_omega"])        # (S, 3, 3)
    z = np.array(samples["z"])                    # (S, N_pools, 3)
    beta_weekend = np.array(samples["beta_weekend"])  # (S,)
    sigma_eps = np.array(samples["sigma_eps"])    # (S,)

    X_pool = encoding["X_pool"]   # (N_pools, K)
    pool_ids = encoding["pool_ids"]

    # mu = X_pool @ Phi for each sample: (S, N_pools, 3)
    mu = np.einsum("pk,skj->spj", X_pool, Phi)

    # L_Sigma = diag(sigma_theta) @ L_omega: (S, 3, 3)
    L_Sigma = sigma_theta[:, :, None] * L_omega

    # offset = z @ L_Sigma^T: (S, N_pools, 3)
    offset = np.einsum("spi,sji->spj", z, L_Sigma)

    theta = mu + offset  # (S, N_pools, 3)
    theta_mean = theta.mean(axis=0)  # (N_pools, 3)

    pool_effects = {}
    for i, pid in enumerate(pool_ids):
        pool_effects[pid] = {
            "alpha": float(theta_mean[i, 0]),
            "beta_tvl": float(theta_mean[i, 1]),
            "beta_vol": float(theta_mean[i, 2]),
        }

    Phi_mean = Phi.mean(axis=0)  # (K, 3)

    # Correlation matrix: R = L_omega @ L_omega^T, averaged over samples
    R_samples = np.einsum("sij,skj->sik", L_omega, L_omega)
    R_mean = R_samples.mean(axis=0)

    return {
        "pool_effects": pool_effects,
        "Phi_mean": Phi_mean,
        "sigma_theta_mean": sigma_theta.mean(axis=0),
        "correlation_matrix": R_mean,
        "beta_weekend_mean": float(beta_weekend.mean()),
        "sigma_eps_mean": float(sigma_eps.mean()),
        "theta_samples": theta,
    }


def compute_noise_params(posteriors: dict, panel: pd.DataFrame) -> list:
    """Convert posterior pool effects to per-pool noise_params dicts.

    Each pool now has its own β_tvl and β_vol (from the hierarchical
    posterior), rather than sharing global slopes.

    Parameters
    ----------
    posteriors : dict
        From extract_posteriors.
    panel : pd.DataFrame
        Panel data.

    Returns
    -------
    list of dict
        Each dict has: pool_id, chain, tokens, noise_params.
    """
    pool_effects = posteriors["pool_effects"]
    pool_meta = panel.drop_duplicates("pool_id").set_index("pool_id")

    results = []
    for pool_id, effects in pool_effects.items():
        if pool_id not in pool_meta.index:
            continue
        meta = pool_meta.loc[pool_id]
        swap_fee = float(meta.get("swap_fee", 0.003))

        results.append({
            "pool_id": pool_id,
            "chain": meta["chain"],
            "tokens": (meta["tokens"].split(",")
                       if isinstance(meta["tokens"], str)
                       else meta["tokens"]),
            "noise_params": {
                "b_0": effects["alpha"],
                "b_sigma": effects["beta_vol"],
                "b_c": effects["beta_tvl"],
                "base_fee": swap_fee,
            },
        })

    return results


def _build_covariate_vector(
    encoding: dict, chain: str, tokens: list, fee: float,
) -> np.ndarray:
    """Construct a covariate vector x for a new pool.

    Matches the column order of X_pool from _encode_covariates so that
    x @ Phi_mean gives population-level predictions for all 3 coefficients.
    """
    col_names = encoding["covariate_names"]
    x = np.zeros(len(col_names))

    tiers = sorted([classify_token_tier(t) for t in tokens])
    tier_a = str(tiers[0])
    tier_b = str(tiers[1]) if len(tiers) > 1 else tier_a

    for i, name in enumerate(col_names):
        if name == "intercept":
            x[i] = 1.0
        elif name == "log_fee":
            x[i] = np.log(max(fee, 1e-6))
        elif name == f"chain_{chain}":
            x[i] = 1.0
        elif name == f"tier_A_{tier_a}":
            x[i] = 1.0
        elif name == f"tier_B_{tier_b}":
            x[i] = 1.0

    return x


def predict_new_pool(
    posteriors: dict,
    encoding: dict,
    chain: str,
    tokens: list,
    fee: float,
) -> dict:
    """Predict noise params for an unseen pool.

    Uses population-level estimates only (x @ Φ, no pool random effect).

    Parameters
    ----------
    posteriors : dict
        From extract_posteriors.
    encoding : dict
        From _encode_covariates (or loaded from cache).
    chain : str
        Chain API identifier (e.g. "BASE").
    tokens : list
        Token symbols (e.g. ["ETH", "BTC"]).
    fee : float
        Swap fee rate.

    Returns
    -------
    dict
        noise_params dict with pool-predicted coefficients.
    """
    x = _build_covariate_vector(encoding, chain, tokens, fee)
    Phi_mean = posteriors["Phi_mean"]

    theta_pred = x @ Phi_mean  # (3,) — population-level prediction

    tiers = sorted([classify_token_tier(t) for t in tokens])
    tier_a = tiers[0]
    tier_b = tiers[1] if len(tiers) > 1 else tiers[0]

    return {
        "b_0": float(theta_pred[0]),
        "b_sigma": float(theta_pred[2]),
        "b_c": float(theta_pred[1]),
        "base_fee": float(fee),
        "_prediction_source": "population_level",
        "_alpha": float(theta_pred[0]),
        "_beta_tvl": float(theta_pred[1]),
        "_beta_vol": float(theta_pred[2]),
        "_tier_a": tier_a,
        "_tier_b": tier_b,
    }


# ---------------------------------------------------------------------------
# Phase 3: Diagnostics and output
# ---------------------------------------------------------------------------

def plot_hierarchical_diagnostics(
    panel: pd.DataFrame,
    posteriors: dict,
    encoding: dict,
    output_dir: str = "results",
):
    """Generate diagnostic plots for the Bayesian hierarchical model.

    Figure 1 (2x2):
      (0,0) Pool-specific coefficient distributions (α, β_tvl, β_vol)
      (0,1) Chain effects on all 3 coefficients
      (1,0) Tier effects on all 3 coefficients
      (1,1) Model summary (Φ, σ_θ, correlations, σ_ε)

    Figure 2:
      β_tvl vs β_vol scatter colored by chain
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    pool_effects = posteriors["pool_effects"]
    Phi_mean = posteriors["Phi_mean"]
    col_names = encoding["covariate_names"]
    pool_meta = panel.drop_duplicates("pool_id")

    alphas = [e["alpha"] for e in pool_effects.values()]
    beta_tvls = [e["beta_tvl"] for e in pool_effects.values()]
    beta_vols = [e["beta_vol"] for e in pool_effects.values()]

    # --- Figure 1: Diagnostics 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Pool-specific coefficient distributions
    ax = axes[0, 0]
    bins = 25
    ax.hist(alphas, bins=bins, alpha=0.6, label="α (intercept)",
            color="steelblue", edgecolor="white")
    ax.hist(beta_tvls, bins=bins, alpha=0.6, label="β_tvl",
            color="coral", edgecolor="white")
    ax.hist(beta_vols, bins=bins, alpha=0.6, label="β_vol",
            color="seagreen", edgecolor="white")
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Count")
    ax.set_title(f"Pool-specific coefficients (n={len(alphas)})")
    ax.legend()

    # (0,1) Chain effects on all 3 coefficients
    ax = axes[0, 1]
    chain_rows = {name: i for i, name in enumerate(col_names)
                  if name.startswith("chain_")}
    chain_labels = []
    chain_alpha_effects = []
    chain_tvl_effects = []
    chain_vol_effects = []
    # Reference chain (effect = 0)
    ref_chain = encoding["ref_chain"]
    chain_counts = pool_meta["chain"].value_counts()
    chain_labels.append(f"{ref_chain}\n(n={chain_counts.get(ref_chain, 0)})")
    chain_alpha_effects.append(0.0)
    chain_tvl_effects.append(0.0)
    chain_vol_effects.append(0.0)
    for name, row_idx in sorted(chain_rows.items()):
        chain_name = name.replace("chain_", "")
        chain_labels.append(
            f"{chain_name}\n(n={chain_counts.get(chain_name, 0)})"
        )
        chain_alpha_effects.append(Phi_mean[row_idx, 0])
        chain_tvl_effects.append(Phi_mean[row_idx, 1])
        chain_vol_effects.append(Phi_mean[row_idx, 2])

    y_pos = np.arange(len(chain_labels))
    bar_h = 0.25
    ax.barh(y_pos - bar_h, chain_alpha_effects, bar_h, label="α",
            color="steelblue", alpha=0.8)
    ax.barh(y_pos, chain_tvl_effects, bar_h, label="β_tvl",
            color="coral", alpha=0.8)
    ax.barh(y_pos + bar_h, chain_vol_effects, bar_h, label="β_vol",
            color="seagreen", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chain_labels)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Effect (relative to reference)")
    ax.set_title("Chain effects on all coefficients")
    ax.legend(fontsize=8)

    # (1,0) Tier effects on all 3 coefficients
    ax = axes[1, 0]
    tier_names = ["0 (blue-chip)", "1 (mid-cap)", "2 (long-tail)"]
    coeff_labels = ["α", "β_tvl", "β_vol"]
    tier_a_rows = {name: i for i, name in enumerate(col_names)
                   if name.startswith("tier_A_")}
    tier_b_rows = {name: i for i, name in enumerate(col_names)
                   if name.startswith("tier_B_")}

    # Build effects matrix: (3 tiers) x (3 coefficients) x (A/B)
    x_pos = np.arange(len(tier_names))
    width = 0.13
    for coeff_idx, (coeff_name, color) in enumerate(
        zip(coeff_labels, ["steelblue", "coral", "seagreen"])
    ):
        tier_a_vals = [0.0, 0.0, 0.0]  # reference tier gets 0
        tier_b_vals = [0.0, 0.0, 0.0]
        for name, row_idx in tier_a_rows.items():
            tier_val = name.replace("tier_A_", "")
            if tier_val in ("0", "1", "2"):
                tier_a_vals[int(tier_val)] = Phi_mean[row_idx, coeff_idx]
        for name, row_idx in tier_b_rows.items():
            tier_val = name.replace("tier_B_", "")
            if tier_val in ("0", "1", "2"):
                tier_b_vals[int(tier_val)] = Phi_mean[row_idx, coeff_idx]
        offset = (coeff_idx - 1) * width * 2
        ax.bar(x_pos + offset - width / 2, tier_a_vals, width,
               label=f"{coeff_name} (A)" if coeff_idx == 0 else "",
               color=color, alpha=0.7, edgecolor="white")
        ax.bar(x_pos + offset + width / 2, tier_b_vals, width,
               label=f"{coeff_name} (B)" if coeff_idx == 0 else "",
               color=color, alpha=0.4, edgecolor="white", hatch="//")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_names)
    ax.set_ylabel("Effect on coefficient")
    ax.set_title("Token tier effects (solid=A, hatched=B)")
    ax.axhline(0, color="black", linewidth=0.5)
    # Manual legend for coefficient colors
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in
                       zip(["steelblue", "coral", "seagreen"], coeff_labels)],
              fontsize=8)

    # (1,1) Model summary text
    ax = axes[1, 1]
    ax.axis("off")
    sigma_theta = posteriors["sigma_theta_mean"]
    R = posteriors["correlation_matrix"]
    summary = "Group-level regression Φ (posterior mean):\n"
    summary += f"  {'covariate':<20s} {'α':>8s} {'β_tvl':>8s} {'β_vol':>8s}\n"
    summary += "  " + "-" * 46 + "\n"
    for j, name in enumerate(col_names):
        summary += (f"  {name:<20s} {Phi_mean[j,0]:>8.3f} "
                    f"{Phi_mean[j,1]:>8.3f} {Phi_mean[j,2]:>8.3f}\n")
    summary += f"\nσ_θ: [{sigma_theta[0]:.3f}, {sigma_theta[1]:.3f}, "
    summary += f"{sigma_theta[2]:.3f}]\n"
    summary += f"Correlation:\n"
    for i in range(3):
        summary += f"  [{R[i,0]:>6.3f} {R[i,1]:>6.3f} {R[i,2]:>6.3f}]\n"
    summary += f"β_weekend: {posteriors['beta_weekend_mean']:.4f}\n"
    summary += f"σ_ε: {posteriors['sigma_eps_mean']:.4f}\n"
    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace")
    ax.set_title("Model Summary")

    fig.suptitle(
        "Bayesian Hierarchical Noise Model — Diagnostics", fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "hierarchical_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 2: β_tvl vs β_vol scatter colored by chain ---
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    pool_id_to_chain = dict(
        zip(pool_meta["pool_id"], pool_meta["chain"])
    )
    chain_colors = {}
    cmap = plt.cm.tab10
    unique_chains = sorted(pool_meta["chain"].unique())
    for i, c in enumerate(unique_chains):
        chain_colors[c] = cmap(i % 10)

    for pid, effects in pool_effects.items():
        c = pool_id_to_chain.get(pid, "?")
        ax2.scatter(effects["beta_tvl"], effects["beta_vol"],
                    color=chain_colors.get(c, "gray"), alpha=0.6, s=20,
                    edgecolors="white", linewidths=0.3)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=chain_colors[c], markersize=8,
                       label=c)
               for c in unique_chains if c in chain_colors]
    ax2.legend(handles=handles, fontsize=8, loc="best")
    ax2.set_xlabel("β_tvl (TVL elasticity)")
    ax2.set_ylabel("β_vol (volatility sensitivity)")
    ax2.set_title("Pool-specific coefficients by chain")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    path2 = os.path.join(output_dir, "beta_tvl_vs_beta_vol.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    return path, path2


def generate_noise_params_json(
    pool_params: list,
    posteriors: dict,
    encoding: dict,
    output_path: str,
    inference_method: str = "svi",
):
    """Write per-pool noise params to JSON.

    Parameters
    ----------
    pool_params : list of dict
        From compute_noise_params.
    posteriors : dict
        From extract_posteriors.
    encoding : dict
        From _encode_covariates.
    output_path : str
        Output JSON path.
    inference_method : str
        "svi" or "nuts".
    """
    output = {
        "model": "bayesian_hierarchical_loglinear",
        "inference_method": inference_method,
        "Phi": posteriors["Phi_mean"].tolist(),
        "covariate_names": encoding["covariate_names"],
        "sigma_theta": posteriors["sigma_theta_mean"].tolist(),
        "correlation_matrix": posteriors["correlation_matrix"].tolist(),
        "beta_weekend": posteriors["beta_weekend_mean"],
        "sigma_eps": posteriors["sigma_eps_mean"],
        "pools": pool_params,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Wrote {len(pool_params)} pool params → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hierarchical noise volume model for Balancer pools"
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch pool data from Balancer API (cached to local_data/)",
    )
    parser.add_argument(
        "--fit", action="store_true",
        help="Fit Bayesian hierarchical model (requires fetched data)",
    )
    parser.add_argument(
        "--nuts", action="store_true",
        help="Use NUTS MCMC instead of SVI (slower, exact posteriors)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate diagnostic plots",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path for per-pool noise params",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for diagnostic plots (default: results)",
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Predict noise params for a new pool",
    )
    parser.add_argument(
        "--chain", default=None,
        help="Chain for --predict (e.g. BASE, MAINNET)",
    )
    parser.add_argument(
        "--tokens", nargs="+", default=None,
        help="Token symbols for --predict (e.g. ETH BTC)",
    )
    parser.add_argument(
        "--fee", type=float, default=0.003,
        help="Swap fee for --predict",
    )
    parser.add_argument(
        "--min-tvl", type=float, default=10000.0,
        help="Minimum TVL filter for pool enumeration",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Cache directory (default: local_data/noise_calibration/)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or CACHE_DIR

    if not any([args.fetch, args.fit, args.predict]):
        parser.error("At least one of --fetch, --fit, --predict is required")

    # --- Fetch ---
    pools_cache = os.path.join(cache_dir, "pools.parquet")
    snaps_cache = os.path.join(cache_dir, "pool_snapshots.parquet")
    prices_cache = os.path.join(cache_dir, "token_prices")
    panel_cache = os.path.join(cache_dir, "panel.parquet")

    if args.fetch:
        print("Phase 1: Fetching data from Balancer API")
        print("=" * 60)

        # Step 1: Enumerate pools
        print("\n1. Enumerating pools...")
        pools_df = enumerate_balancer_pools(min_tvl=args.min_tvl)
        os.makedirs(cache_dir, exist_ok=True)
        pools_df.to_parquet(pools_cache, index=False)
        print(f"   Saved {len(pools_df)} pools → {pools_cache}")

        # Step 2: Fetch snapshots
        print("\n2. Fetching daily snapshots...")
        snapshots_df = fetch_all_snapshots(pools_df, cache_path=snaps_cache)

        # Step 3: Fetch token prices
        print("\n3. Fetching token prices...")
        token_addr_by_chain = {}
        for _, pool in pools_df.iterrows():
            chain = pool["chain"]
            tokens = pool["tokens"]
            addresses = pool["token_addresses"]
            if chain not in token_addr_by_chain:
                token_addr_by_chain[chain] = {}
            for sym, addr in zip(tokens, addresses):
                if sym and addr:
                    token_addr_by_chain[chain][sym] = addr

        token_prices = fetch_token_prices(
            token_addr_by_chain, cache_dir=prices_cache
        )

        # Step 4: Assemble panel
        print("\n4. Assembling panel...")
        panel = assemble_panel(pools_df, snapshots_df, token_prices)
        panel.to_parquet(panel_cache, index=False)
        print(f"   Saved panel → {panel_cache}")

        print(f"\nFetch complete. Panel: {len(panel)} obs, "
              f"{panel['pool_id'].nunique()} pools")

    # --- Fit ---
    if args.fit:
        inference_method = "nuts" if args.nuts else "svi"
        print(f"\nPhase 2: Fitting Bayesian hierarchical model ({inference_method})")
        print("=" * 60)

        # Load panel
        if not os.path.exists(panel_cache):
            print(f"ERROR: Panel cache not found at {panel_cache}",
                  file=sys.stderr)
            print("Run with --fetch first.", file=sys.stderr)
            sys.exit(1)

        panel = pd.read_parquet(panel_cache)
        print(f"  Loaded panel: {len(panel)} obs, "
              f"{panel['pool_id'].nunique()} pools, "
              f"{panel['chain'].nunique()} chains")

        # Filter: need at least 10 days per pool for stable estimates
        pool_counts = panel.groupby("pool_id").size()
        valid_pools = pool_counts[pool_counts >= 10].index
        panel_filtered = panel[panel["pool_id"].isin(valid_pools)]
        print(f"  After filtering (≥10 days): {len(panel_filtered)} obs, "
              f"{panel_filtered['pool_id'].nunique()} pools")

        samples, encoding = fit_bayesian_model(
            panel_filtered, use_nuts=args.nuts,
        )
        posteriors = extract_posteriors(samples, encoding)
        pool_params = compute_noise_params(posteriors, panel_filtered)

        # Print key diagnostics
        Phi_mean = posteriors["Phi_mean"]
        col_names = encoding["covariate_names"]
        intercept_idx = col_names.index("intercept")
        log_fee_idx = col_names.index("log_fee")

        print(f"\n  Key results:")
        print(f"    Population intercept (Φ[intercept]):")
        print(f"      α:     {Phi_mean[intercept_idx, 0]:.4f}")
        print(f"      β_tvl: {Phi_mean[intercept_idx, 1]:.4f}")
        print(f"      β_vol: {Phi_mean[intercept_idx, 2]:.4f}")
        print(f"    Fee effect (Φ[log_fee]):")
        print(f"      α:     {Phi_mean[log_fee_idx, 0]:.4f}")
        print(f"      β_tvl: {Phi_mean[log_fee_idx, 1]:.4f}")
        print(f"      β_vol: {Phi_mean[log_fee_idx, 2]:.4f}")
        print(f"    β_weekend: {posteriors['beta_weekend_mean']:.4f}")
        print(f"    σ_θ: {posteriors['sigma_theta_mean']}")
        print(f"    σ_ε: {posteriors['sigma_eps_mean']:.4f}")

        # Verify pool-specific variation
        b_sigmas = [p["noise_params"]["b_sigma"] for p in pool_params]
        b_cs = [p["noise_params"]["b_c"] for p in pool_params]
        print(f"    b_sigma range: [{min(b_sigmas):.4f}, {max(b_sigmas):.4f}]")
        print(f"    b_c range:     [{min(b_cs):.4f}, {max(b_cs):.4f}]")

        # Cache posteriors + encoding for --predict and --plot
        posteriors_cache = os.path.join(cache_dir, "posteriors.json")
        cache_data = {
            "Phi_mean": posteriors["Phi_mean"].tolist(),
            "sigma_theta_mean": posteriors["sigma_theta_mean"].tolist(),
            "correlation_matrix": posteriors["correlation_matrix"].tolist(),
            "beta_weekend_mean": posteriors["beta_weekend_mean"],
            "sigma_eps_mean": posteriors["sigma_eps_mean"],
            "pool_effects": posteriors["pool_effects"],
            "covariate_names": encoding["covariate_names"],
            "ref_chain": encoding["ref_chain"],
            "ref_tier_a": encoding["ref_tier_a"],
            "ref_tier_b": encoding["ref_tier_b"],
            "chains": encoding["chains"],
            "inference_method": inference_method,
        }
        with open(posteriors_cache, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)
        print(f"  Cached posteriors → {posteriors_cache}")

        if args.output:
            generate_noise_params_json(
                pool_params, posteriors, encoding,
                args.output, inference_method=inference_method,
            )

        if args.plot:
            print("\nPhase 3: Generating diagnostics")
            print("=" * 60)
            plot_hierarchical_diagnostics(
                panel_filtered, posteriors, encoding,
                output_dir=args.output_dir,
            )

    # --- Predict ---
    if args.predict:
        if args.chain is None or args.tokens is None:
            parser.error("--predict requires --chain and --tokens")

        print(f"\nPredicting noise params for new pool:")
        print(f"  Chain:  {args.chain}")
        print(f"  Tokens: {args.tokens}")
        print(f"  Fee:    {args.fee}")

        # Load cached posteriors + encoding metadata
        posteriors_cache = os.path.join(cache_dir, "posteriors.json")
        if not os.path.exists(posteriors_cache):
            print(f"ERROR: Posteriors cache not found at {posteriors_cache}",
                  file=sys.stderr)
            print("Run with --fit first.", file=sys.stderr)
            sys.exit(1)

        with open(posteriors_cache) as f:
            cache_data = json.load(f)

        posteriors = {
            "Phi_mean": np.array(cache_data["Phi_mean"]),
            "sigma_theta_mean": np.array(cache_data["sigma_theta_mean"]),
            "correlation_matrix": np.array(cache_data["correlation_matrix"]),
            "beta_weekend_mean": cache_data["beta_weekend_mean"],
            "sigma_eps_mean": cache_data["sigma_eps_mean"],
            "pool_effects": cache_data["pool_effects"],
        }
        encoding = {
            "covariate_names": cache_data["covariate_names"],
            "ref_chain": cache_data["ref_chain"],
            "ref_tier_a": cache_data["ref_tier_a"],
            "ref_tier_b": cache_data["ref_tier_b"],
            "chains": cache_data["chains"],
        }

        params = predict_new_pool(
            posteriors, encoding, args.chain, args.tokens, args.fee,
        )
        print(f"\n  Predicted noise_params:")
        print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
