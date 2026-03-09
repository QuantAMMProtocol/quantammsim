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


def build_x_obs(panel_rows: pd.DataFrame) -> np.ndarray:
    """Build (n_obs, 8) observation covariate matrix from panel rows.

    Columns: [1, log_tvl_lag1, log_sigma, tvl*sigma, tvl*fee,
              sigma*fee, dow_sin, dow_cos]

    Where:
        log_sigma = log(max(volatility, 1e-6))
        tvl = log_tvl_lag1
        fee = log_fee
        dow_sin = sin(2*pi*weekday/7), dow_cos = cos(2*pi*weekday/7)
        weekday: Monday=0, ..., Sunday=6
    """
    n = len(panel_rows)
    x = np.zeros((n, K_OBS))

    tvl = panel_rows["log_tvl_lag1"].values.astype(float)
    sigma = np.log(np.maximum(panel_rows["volatility"].values.astype(float), 1e-6))
    fee = panel_rows["log_fee"].values.astype(float)
    weekdays = pd.to_datetime(panel_rows["date"]).dt.weekday.values.astype(float)

    x[:, 0] = 1.0                              # intercept
    x[:, 1] = tvl                               # log_tvl_lag1
    x[:, 2] = sigma                             # log_sigma
    x[:, 3] = tvl * sigma                       # tvl × sigma
    x[:, 4] = tvl * fee                         # tvl × fee
    x[:, 5] = sigma * fee                       # sigma × fee
    x[:, 6] = np.sin(2 * np.pi * weekdays / 7)  # dow_sin
    x[:, 7] = np.cos(2 * np.pi * weekdays / 7)  # dow_cos

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
