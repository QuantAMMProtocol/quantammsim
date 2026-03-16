"""Data assembly for the direct calibration pipeline.

Matches precomputed per-day arb grids to panel observations and builds
model-ready arrays for the loss function.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantammsim.calibration.grid_interpolation import (
    PoolCoeffsDaily,
    load_daily_grid,
    precompute_pool_coeffs_daily,
)

K_OBS = 8  # observation-level covariates
K_OBS_REDUCED = 4  # [intercept, log_tvl_lag1, dow_sin, dow_cos]
K_OBS_CROSS = 7    # [intercept, log_tvl_lag1, dow_sin, dow_cos,
                    #  cross_vol_token_a_{t-1}, cross_vol_token_b_{t-1},
                    #  cross_vol_chain_{t-1}]

# Default path for cached token market caps
_MCAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "local_data", "noise_calibration", "token_mcaps.json",
)

# Default path for Binance minute parquets
_BINANCE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data",
)

# Asset type classification (fallback if not in mcap JSON)
_STABLECOINS = {
    "USDC", "USDT", "DAI", "WXDAI", "xDAI", "GHO", "LUSD", "crvUSD",
    "FRAX", "sDAI", "scUSD", "DOLA",
    "waBasUSDC", "waEthUSDC",
}
_NATIVE_LST = {
    "WETH", "ETH", "wstETH", "stETH", "rETH", "cbETH",
    "WBTC", "BTC", "cbBTC",
    "WMATIC", "MATIC", "POL", "wPOL",
    "WAVAX", "AVAX",
    "GNO", "S", "wS", "stS",
    "JitoSOL",
    "waEthLidoWETH", "waEthLidowstETH",
    "waBasWETH", "waGnoGNO", "waGnowstETH",
}

# Balancer token → Binance parquet symbol mapping.
# Matches build_pool_grids.py TOKEN_MAP.
TOKEN_MAP = {
    "WBTC": "BTC", "WETH": "ETH", "cbBTC": "BTC",
    "wstETH": "ETH", "stETH": "ETH", "rETH": "ETH", "cbETH": "ETH",
    "waEthLidoWETH": "ETH", "waEthLidowstETH": "ETH",
    "waBasWETH": "ETH", "waGnowstETH": "ETH",
    "waGnoGNO": "GNO", "osGNO": "GNO",
    "wS": "S", "stS": "S",
    "JitoSOL": "SOL",
    "wPOL": "POL", "WMATIC": "POL", "MATIC": "POL",
    "USDC.e": "USDC", "USDbC": "USDC", "waBasUSDC": "USDC",
    "DAI": "USDC", "WXDAI": "USDC", "sDAI": "USDC",
    "USDT": "USDC", "DOLA": "USDC", "scUSD": "USDC",
}


def _load_token_mcaps(path: str = None) -> dict:
    """Load cached token market caps. Returns {} if file missing."""
    path = path or _MCAP_PATH
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _get_asset_type(symbol: str, mcaps: dict) -> int:
    """Return asset type: 0=stable, 1=native/LST, 2=volatile."""
    if symbol in mcaps and "asset_type" in mcaps[symbol]:
        t = mcaps[symbol]["asset_type"]
        return {"stable": 0, "native_lst": 1, "volatile": 2}.get(t, 2)
    if symbol in _STABLECOINS:
        return 0
    if symbol in _NATIVE_LST:
        return 1
    return 2


def _parse_tokens(tokens_str: str) -> List[str]:
    """Parse comma-separated token string into list."""
    if isinstance(tokens_str, (list, tuple)):
        return list(tokens_str)
    return [t.strip() for t in tokens_str.split(",")]


def _resolve_binance_symbol(token: str) -> str:
    """Map Balancer token name to Binance parquet symbol."""
    return TOKEN_MAP.get(token, token)


def _load_binance_minute(symbol: str, data_dir: str = None) -> Optional[pd.DataFrame]:
    """Load Binance minute close prices. Returns DataFrame with unix index."""
    if data_dir is None:
        data_dir = _BINANCE_DATA_DIR
    path = os.path.join(data_dir, f"{symbol}_USD.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=["unix", "close"])
    if df.index.name != "unix":
        df = df.set_index("unix")
    return df


def compute_binance_pair_volatility(
    token_a: str, token_b: str, data_dir: str = None,
) -> Optional[pd.Series]:
    """Compute daily annualized realized volatility from Binance minute data.

    Resamples minute data to hourly, computes hourly log returns of the pair
    ratio, then daily std × sqrt(24 × 365). Matches the Balancer hourly
    pipeline's annualization convention.

    Args:
        token_a, token_b: Balancer token symbols (e.g. "WETH", "USDC")
        data_dir: directory containing {SYMBOL}_USD.parquet files

    Returns:
        pd.Series with datetime.date index → annualized volatility,
        or None if both tokens are stablecoins / same underlying / missing data.
    """
    sym_a = _resolve_binance_symbol(token_a)
    sym_b = _resolve_binance_symbol(token_b)

    is_stable_a = token_a in _STABLECOINS or sym_a == "USDC"
    is_stable_b = token_b in _STABLECOINS or sym_b == "USDC"

    if is_stable_a and is_stable_b:
        return None  # caller should use constant 0.01

    if sym_a == sym_b:
        return None  # same underlying (e.g. wstETH/WETH)

    # Load minute data and compute pair ratio
    if is_stable_b:
        df = _load_binance_minute(sym_a, data_dir)
        if df is None:
            return None
        ratio = df["close"]
    elif is_stable_a:
        df = _load_binance_minute(sym_b, data_dir)
        if df is None:
            return None
        ratio = 1.0 / df["close"]
    else:
        df_a = _load_binance_minute(sym_a, data_dir)
        df_b = _load_binance_minute(sym_b, data_dir)
        if df_a is None or df_b is None:
            return None
        merged = df_a.join(df_b, lsuffix="_a", rsuffix="_b", how="inner")
        ratio = merged["close_a"] / merged["close_b"]

    # Resample to hourly (last close per hour)
    ratio_df = pd.DataFrame({"ratio": ratio})
    ratio_df.index = pd.to_datetime(ratio_df.index, unit="ms", utc=True)
    hourly = ratio_df.resample("1h").last().dropna()

    # Hourly log returns
    hourly["log_return"] = np.log(hourly["ratio"] / hourly["ratio"].shift(1))
    hourly = hourly.dropna()

    # Daily std → annualized
    hourly["date"] = hourly.index.date
    daily_vol = hourly.groupby("date")["log_return"].std()
    annualized = daily_vol * np.sqrt(24 * 365)

    # Clean
    annualized = annualized.replace([np.inf, -np.inf], np.nan).dropna()
    annualized = annualized[annualized > 0]

    return annualized


def replace_panel_volatility_with_binance(
    panel: pd.DataFrame, data_dir: str = None,
) -> pd.DataFrame:
    """Replace panel 'volatility' column with Binance-derived daily values.

    For each pool, computes daily realized volatility from Binance minute data.
    Pools without Binance data keep their existing (possibly fallback) values.
    Stablecoin-stablecoin and same-underlying pairs get vol=0.01.

    Returns a copy of the panel with updated volatility.
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # Cache: (sym_a, sym_b) → vol_series to avoid reloading
    _vol_cache: Dict[tuple, Optional[pd.Series]] = {}

    n_replaced = 0
    n_pools = 0

    for pool_id, grp in panel.groupby("pool_id"):
        tokens_str = grp.iloc[0]["tokens"]
        toks = _parse_tokens(tokens_str)
        if len(toks) < 2:
            continue

        sym_a = _resolve_binance_symbol(toks[0])
        sym_b = _resolve_binance_symbol(toks[1])
        cache_key = (min(sym_a, sym_b), max(sym_a, sym_b))

        if cache_key not in _vol_cache:
            _vol_cache[cache_key] = compute_binance_pair_volatility(
                toks[0], toks[1], data_dir)

        vol_series = _vol_cache[cache_key]

        if vol_series is None:
            # Stablecoins or same underlying → low constant vol
            is_stable_a = toks[0] in _STABLECOINS or sym_a == "USDC"
            is_stable_b = toks[1] in _STABLECOINS or sym_b == "USDC"
            if (is_stable_a and is_stable_b) or sym_a == sym_b:
                panel.loc[grp.index, "volatility"] = 0.01
                n_pools += 1
            continue

        # Vectorized date matching
        panel_dates = pd.to_datetime(grp["date"]).dt.date
        vol_dict = vol_series.to_dict()
        new_vol = panel_dates.map(vol_dict)
        has_vol = new_vol.notna()
        if has_vol.any():
            panel.loc[grp.index[has_vol.values], "volatility"] = (
                new_vol[has_vol].values.astype(float))
            n_replaced += has_vol.sum()
        n_pools += 1

    print(f"  Binance volatility: {n_pools} pools, {n_replaced} obs replaced")
    return panel


def match_grids_to_panel(
    grid_dir: str, panel: pd.DataFrame, pools_path: str = None,
) -> Dict[str, dict]:
    """Match grid parquets to panel rows by pool_id prefix.

    For each _daily.parquet in grid_dir, find the panel pool whose
    pool_id starts with the same 16-char prefix. Build PoolCoeffsDaily
    and compute day_indices mapping panel dates to grid date indices.

    Args:
        grid_dir: directory containing {prefix}_daily.parquet files
        panel: panel DataFrame with pool observations
        pools_path: path to pools.parquet for weight metadata.
            Defaults to local_data/noise_calibration/pools.parquet.

    Returns dict: prefix -> {
        'panel': DataFrame (obs for this pool),
        'coeffs': PoolCoeffsDaily (per-day),
        'day_indices': np.ndarray (panel date -> grid day index),
        'pool_id': full pool_id from panel,
        'chain': str, 'fee': float, 'tokens': str,
        'weights': list of float (pool weights, e.g. [0.5, 0.5]),
    }
    """
    # Discover grid files
    grid_prefixes = []
    for f in sorted(os.listdir(grid_dir)):
        if f.endswith("_daily.parquet"):
            prefix = f.replace("_daily.parquet", "")
            grid_prefixes.append(prefix)

    if not grid_prefixes:
        return {}

    # Load pool metadata for weights
    if pools_path is None:
        pools_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "local_data", "noise_calibration", "pools.parquet",
        )
    pools_meta = {}
    if os.path.exists(pools_path):
        pools_df = pd.read_parquet(pools_path)
        pools_df["_prefix"] = pools_df["pool_id"].str[:16]
        for _, row in pools_df.iterrows():
            w = row.get("weights")
            if w is not None:
                try:
                    weights = [float(x) for x in w]
                except (TypeError, ValueError):
                    weights = [0.5, 0.5]
            else:
                weights = [0.5, 0.5]
            pools_meta[row["_prefix"]] = {"weights": weights}

    # Ensure date column
    panel = panel.copy()
    if "date" in panel.columns:
        panel["date"] = pd.to_datetime(panel["date"])

    # Build prefix -> panel rows mapping
    panel["_prefix"] = panel["pool_id"].str[:16]

    matched = {}
    for prefix in grid_prefixes:
        pool_rows = panel[panel["_prefix"] == prefix]
        if len(pool_rows) == 0:
            continue

        # Load and precompute grid
        grid_df = load_daily_grid(prefix, grid_dir)
        coeffs = precompute_pool_coeffs_daily(grid_df)

        # Build date alignment: panel date ordinals -> grid day indices
        grid_ordinals = np.array(coeffs.dates)
        grid_ord_to_idx = {int(o): i for i, o in enumerate(grid_ordinals)}

        panel_dates = pd.to_datetime(pool_rows["date"])
        panel_ordinals = np.array([d.toordinal() for d in panel_dates])

        # Filter to dates present in both panel and grid
        valid_mask = np.array([int(o) in grid_ord_to_idx for o in panel_ordinals])
        pool_rows = pool_rows[valid_mask].copy()
        panel_ordinals = panel_ordinals[valid_mask]

        if len(pool_rows) == 0:
            continue

        day_indices = np.array([grid_ord_to_idx[int(o)] for o in panel_ordinals])

        row0 = pool_rows.iloc[0]
        weights = pools_meta.get(prefix, {}).get("weights", [0.5, 0.5])

        matched[prefix] = {
            "panel": pool_rows.reset_index(drop=True),
            "coeffs": coeffs,
            "day_indices": day_indices,
            "pool_id": row0["pool_id"],
            "chain": row0["chain"],
            "fee": float(np.exp(row0["log_fee"])) if "swap_fee" not in pool_rows.columns
                   else float(row0.get("swap_fee", np.exp(row0["log_fee"]))),
            "tokens": row0["tokens"],
            "weights": weights,
        }

    return matched


# Token canonicalization — map wrapped/LST variants to base tokens
_CANON_MAP = {
    "WETH": "ETH", "waBasWETH": "ETH", "waEthLidoWETH": "ETH",
    "waEthLidowstETH": "wstETH", "waGnowstETH": "wstETH",
    "waBasUSDC": "USDC", "scUSD": "USDC", "USDC.e": "USDC",
    "USDbC": "USDC", "waEthUSDC": "USDC",
    "sDAI": "DAI", "WXDAI": "DAI",
    "WBTC": "BTC", "cbBTC": "BTC",
    "stS": "S", "wS": "S",
    "waGnoGNO": "GNO", "osGNO": "GNO",
}


def _canonicalize_token(symbol: str) -> str:
    """Map wrapped/derivative token to its canonical base symbol."""
    return _CANON_MAP.get(symbol, symbol)


# Token classification for token-factored model
_ETH_DERIVATIVES = {
    "WETH", "ETH", "wstETH", "stETH", "rETH", "cbETH",
    "waEthLidoWETH", "waEthLidowstETH", "waBasWETH", "waGnowstETH",
}
_L1_NATIVE = {
    "WETH", "ETH", "WMATIC", "MATIC", "POL", "wPOL",
    "WAVAX", "AVAX", "GNO", "S", "wS", "stS",
}

D_TOKEN = 5  # [intercept, log_mcap, is_stable, is_eth_derivative, is_L1_native]


def _classify_token(symbol: str, mcaps: dict) -> dict:
    """Classify a token into binary feature flags."""
    return {
        "is_stable": 1.0 if symbol in _STABLECOINS else 0.0,
        "is_eth_derivative": 1.0 if symbol in _ETH_DERIVATIVES else 0.0,
        "is_L1_native": 1.0 if symbol in _L1_NATIVE else 0.0,
        "log_mcap": np.log(max(mcaps.get(symbol, {}).get("mcap_usd", 1e6), 1.0)),
    }


def encode_tokens(
    matched: Dict[str, dict],
    mcap_path: str = None,
    canonicalize: bool = True,
) -> dict:
    """Build token index, per-pool token assignments, and token covariate matrix.

    Iterates over pools in sorted key order (same ordering as build_pool_attributes).

    When canonicalize=True (default), wrappd/derivative tokens are mapped to
    their canonical base symbol via _CANON_MAP before building the index.
    Raw symbols are still used for market cap lookup.

    Returns dict with:
        token_index: dict[str, int] — symbol -> integer index (sorted alphabetically)
        token_a_idx: np.ndarray (n_pools,) — index of token A for each pool
        token_b_idx: np.ndarray (n_pools,) — index of token B for each pool
        x_token: np.ndarray (n_tokens, D_TOKEN) — token covariate matrix
        chain_idx: np.ndarray (n_pools,) — chain integer index per pool
        chain_index: dict[str, int] — chain name -> integer index (sorted)
        log_fees: np.ndarray (n_pools,) — log(fee) per pool
        n_tokens: int
        n_chains: int
    """
    mcaps = _load_token_mcaps(mcap_path)
    pool_ids = sorted(matched.keys())
    n_pools = len(pool_ids)

    # Collect all tokens and chains; store per-pool canonical pairs
    all_tokens = set()
    all_chains = set()
    pool_canon_toks = []  # (canon_a, canon_b) per pool in sorted order
    for pid in pool_ids:
        entry = matched[pid]
        toks = _parse_tokens(entry["tokens"])
        raw_a, raw_b = toks[0], toks[1]
        canon_a = _canonicalize_token(raw_a) if canonicalize else raw_a
        canon_b = _canonicalize_token(raw_b) if canonicalize else raw_b
        all_tokens.update([canon_a, canon_b])
        all_chains.add(entry["chain"])
        pool_canon_toks.append((canon_a, canon_b))

    # Build sorted indices
    token_list = sorted(all_tokens)
    token_index = {t: i for i, t in enumerate(token_list)}
    n_tokens = len(token_list)

    chain_list = sorted(all_chains)
    chain_index = {c: i for i, c in enumerate(chain_list)}
    n_chains = len(chain_list)

    # Build per-pool arrays
    token_a_idx = np.zeros(n_pools, dtype=np.int32)
    token_b_idx = np.zeros(n_pools, dtype=np.int32)
    chain_idx = np.zeros(n_pools, dtype=np.int32)
    log_fees = np.zeros(n_pools, dtype=np.float64)

    for i, pid in enumerate(pool_ids):
        entry = matched[pid]
        canon_a, canon_b = pool_canon_toks[i]
        token_a_idx[i] = token_index[canon_a]
        token_b_idx[i] = token_index[canon_b]
        chain_idx[i] = chain_index[entry["chain"]]
        log_fees[i] = np.log(entry["fee"])

    # Build token covariate matrix: (n_tokens, D_TOKEN)
    # Columns: [intercept, log_mcap, is_stable, is_eth_derivative, is_L1_native]
    x_token = np.zeros((n_tokens, D_TOKEN), dtype=np.float64)
    for t, idx in token_index.items():
        cls = _classify_token(t, mcaps)
        x_token[idx, 0] = 1.0  # intercept
        x_token[idx, 1] = cls["log_mcap"]
        x_token[idx, 2] = cls["is_stable"]
        x_token[idx, 3] = cls["is_eth_derivative"]
        x_token[idx, 4] = cls["is_L1_native"]

    return {
        "token_index": token_index,
        "token_a_idx": token_a_idx,
        "token_b_idx": token_b_idx,
        "x_token": x_token,
        "chain_idx": chain_idx,
        "chain_index": chain_index,
        "log_fees": log_fees,
        "n_tokens": n_tokens,
        "n_chains": n_chains,
    }


def build_x_obs(panel_rows: pd.DataFrame, reduced: bool = False) -> np.ndarray:
    """Build observation covariate matrix from panel rows.

    Full (reduced=False): (n_obs, 8)
        [1, log_tvl_lag1, log_sigma, tvl*sigma, tvl*fee, sigma*fee, dow_sin, dow_cos]

    Reduced (reduced=True): (n_obs, 4)
        [1, log_tvl_lag1, dow_sin, dow_cos]
        Removes sigma- and fee-dependent terms so the arb channel is the only
        path for volatility-driven volume variation.

    Where:
        log_sigma = log(max(volatility, 1e-6))
        tvl = log_tvl_lag1
        fee = log_fee
        dow_sin = sin(2*pi*weekday/7), dow_cos = cos(2*pi*weekday/7)
        weekday: Monday=0, ..., Sunday=6
    """
    n = len(panel_rows)
    weekdays = pd.to_datetime(panel_rows["date"]).dt.weekday.values.astype(float)

    if reduced:
        x = np.zeros((n, K_OBS_REDUCED))
        x[:, 0] = 1.0
        x[:, 1] = panel_rows["log_tvl_lag1"].values.astype(float)
        x[:, 2] = np.sin(2 * np.pi * weekdays / 7)
        x[:, 3] = np.cos(2 * np.pi * weekdays / 7)
        return x

    x = np.zeros((n, K_OBS))

    tvl = panel_rows["log_tvl_lag1"].values.astype(float)
    sigma = np.log(np.maximum(panel_rows["volatility"].values.astype(float), 1e-6))
    fee = panel_rows["log_fee"].values.astype(float)

    x[:, 0] = 1.0                              # intercept
    x[:, 1] = tvl                               # log_tvl_lag1
    x[:, 2] = sigma                             # log_sigma
    x[:, 3] = tvl * sigma                       # tvl × sigma
    x[:, 4] = tvl * fee                         # tvl × fee
    x[:, 5] = sigma * fee                       # sigma × fee
    x[:, 6] = np.sin(2 * np.pi * weekdays / 7)  # dow_sin
    x[:, 7] = np.cos(2 * np.pi * weekdays / 7)  # dow_cos

    return x


def build_cross_pool_x_obs(
    panel_rows: pd.DataFrame,
    matched: Dict[str, dict],
    pool_id: str,
    exclude_pool: Optional[str] = None,
    canonicalize: bool = True,
) -> np.ndarray:
    """Build x_obs with cross-pool lagged volume features.

    Columns 0-3: same as build_x_obs(reduced=True)
    Column 4: mean log_volume at t-1 across pools sharing token A (excl self)
    Column 5: mean log_volume at t-1 across pools sharing token B (excl self)
    Column 6: mean log_volume at t-1 across pools on same chain (excl self)

    The first observation (day 0) is dropped because there is no lag available.

    Args:
        panel_rows: DataFrame for this pool
        matched: full matched dict (all pools)
        pool_id: this pool's key in matched (prefix)
        exclude_pool: optional pool to exclude from peer averages (for LOO)
        canonicalize: if True, canonicalize tokens before peer matching

    Returns:
        (n_obs - 1, K_OBS_CROSS) array
    """
    # Get this pool's tokens and chain
    entry = matched[pool_id]
    toks = _parse_tokens(entry["tokens"])
    tok_a_raw, tok_b_raw = toks[0], toks[1]
    tok_a = _canonicalize_token(tok_a_raw) if canonicalize else tok_a_raw
    tok_b = _canonicalize_token(tok_b_raw) if canonicalize else tok_b_raw
    this_chain = entry["chain"]

    # Build peer sets: token→set of pool_ids, chain→set of pool_ids
    token_peers = {}  # canonical_token → set of (prefix, panel_df)
    chain_peers = {}  # chain → set of (prefix, panel_df)
    all_pool_ids = sorted(matched.keys())

    for pid in all_pool_ids:
        if pid == pool_id:
            continue  # always exclude self
        if pid == exclude_pool:
            continue
        peer_entry = matched[pid]
        peer_toks = _parse_tokens(peer_entry["tokens"])
        peer_canonical = set()
        for t in peer_toks[:2]:
            ct = _canonicalize_token(t) if canonicalize else t
            peer_canonical.add(ct)

        for ct in peer_canonical:
            if ct not in token_peers:
                token_peers[ct] = []
            token_peers[ct].append(pid)

        peer_chain = peer_entry["chain"]
        if peer_chain not in chain_peers:
            chain_peers[peer_chain] = []
        chain_peers[peer_chain].append(pid)

    # Build (pool_id, date_ordinal) → log_volume lookup from all pools
    vol_lookup = {}  # (pid, date_ordinal) → log_volume
    for pid in all_pool_ids:
        if pid == pool_id or pid == exclude_pool:
            continue
        peer_panel = matched[pid]["panel"]
        peer_dates = pd.to_datetime(peer_panel["date"])
        peer_ords = np.array([d.toordinal() for d in peer_dates])
        peer_vols = peer_panel["log_volume"].values.astype(float)
        for ord_val, vol_val in zip(peer_ords, peer_vols):
            vol_lookup[(pid, int(ord_val))] = vol_val

    # Compute global lagged mean for fallback
    all_vols = list(vol_lookup.values())
    global_mean_vol = float(np.mean(all_vols)) if all_vols else 0.0

    # Get this pool's dates
    dates = pd.to_datetime(panel_rows["date"])
    date_ords = np.array([d.toordinal() for d in dates])
    n_obs = len(panel_rows)

    def _peer_mean_at_lag(peer_pids, date_ord_prev):
        """Mean log_volume of peer pools at date_ord_prev."""
        vals = []
        for pid in peer_pids:
            key = (pid, date_ord_prev)
            if key in vol_lookup:
                vals.append(vol_lookup[key])
        if vals:
            return float(np.mean(vals))
        return np.nan

    # Build cross-pool features for each obs (starting from day 1)
    cross_vol_a = np.full(n_obs, np.nan)
    cross_vol_b = np.full(n_obs, np.nan)
    cross_vol_chain = np.full(n_obs, np.nan)

    tok_a_peers = token_peers.get(tok_a, [])
    tok_b_peers = token_peers.get(tok_b, [])
    chain_peer_list = chain_peers.get(this_chain, [])

    for i in range(1, n_obs):
        prev_ord = int(date_ords[i - 1])

        if tok_a_peers:
            cross_vol_a[i] = _peer_mean_at_lag(tok_a_peers, prev_ord)
        if tok_b_peers:
            cross_vol_b[i] = _peer_mean_at_lag(tok_b_peers, prev_ord)
        if chain_peer_list:
            cross_vol_chain[i] = _peer_mean_at_lag(chain_peer_list, prev_ord)

    # Drop first day, fill NaN with global mean
    cross_vol_a = cross_vol_a[1:]
    cross_vol_b = cross_vol_b[1:]
    cross_vol_chain = cross_vol_chain[1:]

    cross_vol_a = np.where(np.isnan(cross_vol_a), global_mean_vol, cross_vol_a)
    cross_vol_b = np.where(np.isnan(cross_vol_b), global_mean_vol, cross_vol_b)
    cross_vol_chain = np.where(np.isnan(cross_vol_chain), global_mean_vol, cross_vol_chain)

    # Build base x_obs (reduced) and drop first row
    x_base = build_x_obs(panel_rows, reduced=True)
    x_base = x_base[1:]  # drop first day

    # Assemble
    x = np.zeros((n_obs - 1, K_OBS_CROSS))
    x[:, :4] = x_base
    x[:, 4] = cross_vol_a
    x[:, 5] = cross_vol_b
    x[:, 6] = cross_vol_chain

    return x


def build_pool_attributes(
    matched: Dict[str, dict],
    mcap_path: str = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build (n_pools, K_attr) pool attribute matrix.

    Columns:
        chain_dummies..., log_fee, mean_log_tvl,
        log_mcap_product, has_stable, same_asset_type, weight_imbalance

    No intercept column — bias terms are handled by the model internally.
    Chain dummies: one-hot with the first chain (alphabetically) as reference.
    Market caps loaded from cached JSON (run scripts/fetch_token_mcaps.py).

    Returns: (X_attr, attr_names, pool_ids)
    """
    mcaps = _load_token_mcaps(mcap_path)

    pool_ids = sorted(matched.keys())
    n_pools = len(pool_ids)

    # Collect per-pool attributes
    chains = []
    log_fees = []
    mean_tvls = []
    log_mcap_products = []
    has_stables = []
    same_asset_types = []
    weight_imbalances = []

    for pid in pool_ids:
        entry = matched[pid]
        chains.append(entry["chain"])
        log_fee = entry["panel"]["log_fee"].values[0]
        log_fees.append(float(log_fee))
        mean_tvls.append(float(entry["panel"]["log_tvl_lag1"].mean()))

        # Token-level features
        tokens = _parse_tokens(entry["tokens"])
        tok_a = tokens[0] if len(tokens) > 0 else "UNKNOWN"
        tok_b = tokens[1] if len(tokens) > 1 else "UNKNOWN"

        # Market cap product
        mcap_a = mcaps.get(tok_a, {}).get("mcap_usd", 1e6)  # $1M fallback
        mcap_b = mcaps.get(tok_b, {}).get("mcap_usd", 1e6)
        log_mcap_products.append(np.log(max(mcap_a, 1.0) * max(mcap_b, 1.0)))

        # Asset type: 0=stable, 1=native/LST, 2=volatile
        type_a = _get_asset_type(tok_a, mcaps)
        type_b = _get_asset_type(tok_b, mcaps)
        has_stables.append(1.0 if (type_a == 0 or type_b == 0) else 0.0)
        same_asset_types.append(1.0 if type_a == type_b else 0.0)

        # Weight imbalance: 0 for 50/50, 0.3 for 80/20
        weights = entry.get("weights", [0.5, 0.5])
        if len(weights) >= 2:
            weight_imbalances.append(abs(weights[0] - weights[1]))
        else:
            weight_imbalances.append(0.0)

    # Chain dummies (first alphabetically is reference)
    unique_chains = sorted(set(chains))
    chain_dummies = unique_chains[1:]  # drop reference

    attr_names = (
        [f"chain_{c}" for c in chain_dummies]
        + [
            "log_fee", "mean_log_tvl", "log_mcap_product",
            "has_stable", "same_asset_type", "weight_imbalance",
        ]
    )
    k_attr = len(attr_names)

    X = np.zeros((n_pools, k_attr))
    for i, pid in enumerate(pool_ids):
        chain = chains[i]
        for j, cd in enumerate(chain_dummies):
            if chain == cd:
                X[i, j] = 1.0
        base = len(chain_dummies)
        X[i, base] = log_fees[i]
        X[i, base + 1] = mean_tvls[i]
        X[i, base + 2] = log_mcap_products[i]
        X[i, base + 3] = has_stables[i]
        X[i, base + 4] = same_asset_types[i]
        X[i, base + 5] = weight_imbalances[i]

    return X, attr_names, pool_ids
