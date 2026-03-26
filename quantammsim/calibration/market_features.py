"""Market-level and token-level features for the noise volume model.

Derives daily features from Binance minute-level price data and pool metadata.
Features are grounded in market microstructure — what mechanistically drives
organic (non-arb) trading volume:

Market regime:
  - BTC log price, log return — crypto market regime proxy
  - BTC trend (rolling mean log return) — bull/bear at various horizons

Token-level (per pool token):
  - Token log price, daily log return
  - Token realized volatility — higher vol → more hedging/speculative flow
  - Token Binance volume — proxy for overall token trading interest
  - Token trend (rolling mean log return)

All features are computed daily and aligned to the panel date grid.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)

# Map wrapped/derivative tokens to their Binance underlying
TOKEN_MAP = {
    "WETH": "ETH",
    "WBTC": "BTC",
    "wstETH": "ETH",
    "waEthLidowstETH": "ETH",
    "waEthLidoWETH": "ETH",
    "waGnowstETH": "ETH",
    "waGnoGNO": "GNO",
    "waBasUSDC": "USDC",
    "waBasWETH": "ETH",
    "sDAI": "DAI",
    "scUSD": "USDC",
    "stS": "S",
    "JitoSOL": "SOL",
    "USDT": "USDC",  # treat as $1 stablecoin
}


def _load_binance_daily(symbol: str) -> pd.DataFrame:
    """Load Binance minute data and resample to daily OHLCV."""
    mapped = TOKEN_MAP.get(symbol, symbol)
    path = os.path.join(DATA_DIR, f"{mapped}_USD.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path, columns=["date", "close", "Volume USD"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    daily = df.resample("1D").agg({
        "close": "last",
        "Volume USD": "sum",
    }).dropna(subset=["close"])

    daily.columns = ["close", "volume_usd"]
    return daily


def _compute_token_features(
    daily: pd.DataFrame,
    trend_windows: List[int] = (7, 14, 30),
    is_market: bool = False,
) -> pd.DataFrame:
    """Compute daily features from a token's OHLCV.

    For market-level tokens (BTC): includes log_price as regime proxy.
    For pool tokens: only returns/vol/trends (comparable across tokens).

    Volume is normalised as z-score within each token: today's log-volume
    relative to a 30-day trailing mean/std. This captures "is this token
    unusually active today" without the cross-token scale problem.

    Returns DataFrame indexed by date.
    """
    out = pd.DataFrame(index=daily.index)
    log_price = np.log(daily["close"].clip(lower=1e-10))
    out["log_return"] = log_price.diff()

    if is_market:
        # BTC log_price is a market regime proxy (same for all pools)
        out["log_price"] = log_price

    # Realized volatility: std of log returns over trailing 7 days
    out["realized_vol_7d"] = out["log_return"].rolling(7, min_periods=3).std()

    # Volume: z-score relative to trailing 30d mean/std of log-volume
    # Captures "unusually active day for this token" — comparable across tokens
    log_vol = np.log(daily["volume_usd"].clip(lower=1.0))
    vol_mean_30d = log_vol.rolling(30, min_periods=10).mean()
    vol_std_30d = log_vol.rolling(30, min_periods=10).std().clip(lower=0.1)
    out["volume_zscore"] = (log_vol - vol_mean_30d) / vol_std_30d

    # Trend: rolling mean log return at various horizons
    for w in trend_windows:
        out[f"trend_{w}d"] = out["log_return"].rolling(w, min_periods=max(w // 2, 2)).mean()

    return out


def build_btc_daily_features(
    trend_windows: List[int] = (7, 14, 30),
) -> pd.DataFrame:
    """BTC daily features as market regime proxy.

    Returns DataFrame indexed by date with columns prefixed 'btc_'.
    """
    daily = _load_binance_daily("BTC")
    if daily is None:
        raise FileNotFoundError("BTC_USD.parquet not found")

    feat = _compute_token_features(daily, trend_windows, is_market=True)
    feat.columns = [f"btc_{c}" for c in feat.columns]
    return feat


def build_token_daily_features(
    symbol: str,
    trend_windows: List[int] = (7, 14, 30),
) -> Optional[pd.DataFrame]:
    """Daily features for a single token. Returns None if no data."""
    daily = _load_binance_daily(symbol)
    if daily is None:
        return None
    return _compute_token_features(daily, trend_windows)


def _compute_pair_volatility(
    symbol_a: str,
    symbol_b: str,
) -> Optional[pd.DataFrame]:
    """Compute realized volatility of the A/B price ratio.

    vol(log(price_A/price_B)) = vol(log(price_A) - log(price_B))
    Symmetric: A/B and B/A give identical volatility.

    Returns DataFrame indexed by date with 'pair_realized_vol_7d'.
    """
    daily_a = _load_binance_daily(symbol_a)
    daily_b = _load_binance_daily(symbol_b)
    if daily_a is None or daily_b is None:
        return None

    # Align on common dates
    log_a = np.log(daily_a["close"].clip(lower=1e-10))
    log_b = np.log(daily_b["close"].clip(lower=1e-10))
    common = log_a.index.intersection(log_b.index)
    if len(common) < 10:
        return None

    log_ratio = log_a.loc[common] - log_b.loc[common]
    log_ratio_return = log_ratio.diff()

    out = pd.DataFrame(index=common)
    out["pair_realized_vol_7d"] = log_ratio_return.rolling(7, min_periods=3).std()
    return out


def build_pool_market_features(
    matched_clean: Dict[str, dict],
    trend_windows: List[int] = (7, 14, 30),
) -> Dict[str, pd.DataFrame]:
    """Build per-pool market feature DataFrames.

    For each pool, produces a DataFrame aligned to the pool's panel dates with:
      - BTC features (market regime)
      - Token A features (vs USD)
      - Token B features (vs USD)
      - Pair volatility (A/B ratio)

    Returns dict: pool_id -> DataFrame with all features.
    """
    from quantammsim.calibration.pool_data import _parse_tokens

    # Load BTC features once
    btc_feat = build_btc_daily_features(trend_windows)

    # Cache token features and pair volatilities
    token_cache = {}
    pair_vol_cache = {}

    pool_features = {}
    for pid, entry in matched_clean.items():
        panel = entry["panel"]
        dates = pd.to_datetime(panel["date"])

        # Parse tokens
        toks = _parse_tokens(entry["tokens"])
        tok_a, tok_b = toks[0], toks[1] if len(toks) > 1 else toks[0]

        # Get token features
        for tok in [tok_a, tok_b]:
            mapped = TOKEN_MAP.get(tok, tok)
            if mapped not in token_cache:
                token_cache[mapped] = build_token_daily_features(mapped, trend_windows)

        feat_a = token_cache.get(TOKEN_MAP.get(tok_a, tok_a))
        feat_b = token_cache.get(TOKEN_MAP.get(tok_b, tok_b))

        # Pair volatility (cache by sorted token pair to avoid duplicates)
        mapped_a = TOKEN_MAP.get(tok_a, tok_a)
        mapped_b = TOKEN_MAP.get(tok_b, tok_b)
        pair_key = tuple(sorted([mapped_a, mapped_b]))
        if pair_key not in pair_vol_cache:
            pair_vol_cache[pair_key] = _compute_pair_volatility(mapped_a, mapped_b)
        pair_vol = pair_vol_cache[pair_key]

        # Build per-date feature vectors
        rows = []
        for date in dates:
            day = pd.Timestamp(date).normalize()
            row = {}

            # BTC features
            if day in btc_feat.index:
                for col in btc_feat.columns:
                    row[col] = btc_feat.loc[day, col]

            # Token A features
            if feat_a is not None and day in feat_a.index:
                for col in feat_a.columns:
                    row[f"tok_a_{col}"] = feat_a.loc[day, col]

            # Token B features
            if feat_b is not None and day in feat_b.index:
                for col in feat_b.columns:
                    row[f"tok_b_{col}"] = feat_b.loc[day, col]

            # Pair volatility
            if pair_vol is not None and day in pair_vol.index:
                row["pair_realized_vol_7d"] = pair_vol.loc[day, "pair_realized_vol_7d"]

            rows.append(row)

        df = pd.DataFrame(rows, index=dates)
        pool_features[pid] = df

    return pool_features


def pool_market_features_to_matrix(
    pool_features: Dict[str, pd.DataFrame],
    matched_clean: Dict[str, dict],
    date_to_idx: Dict,
    pool_ids: List[str],
    sample_pools: np.ndarray,
    sample_days: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Convert per-pool market features to a (n_samples, n_feat) matrix.

    Aligns features to the common date grid and sample indices.
    NaN-fills missing values, then imputes with column mean.

    Returns (feature_matrix, feature_names).
    """
    # Get feature columns from first pool
    first_pid = pool_ids[0]
    feat_cols = sorted(pool_features[first_pid].columns)
    n_feat = len(feat_cols)
    n_pools = len(pool_ids)

    # Collect all dates
    n_dates = max(date_to_idx.values()) + 1

    # Build (n_dates, n_pools, n_feat) grid
    feat_grid = np.full((n_dates, n_pools, n_feat), np.nan, dtype=np.float32)

    for j, pid in enumerate(pool_ids):
        if pid not in pool_features:
            continue
        pf = pool_features[pid]
        panel_dates = matched_clean[pid]["panel"]["date"].values
        for k, date in enumerate(panel_dates):
            t = date_to_idx.get(date)
            if t is None:
                continue
            for f, col in enumerate(feat_cols):
                if col in pf.columns and k < len(pf):
                    val = pf.iloc[k][col] if col in pf.columns else np.nan
                    if np.isfinite(val):
                        feat_grid[t, j, f] = val

    # Extract per-sample
    n_samples = len(sample_pools)
    X = np.zeros((n_samples, n_feat), dtype=np.float32)
    for s in range(n_samples):
        X[s] = feat_grid[sample_days[s], sample_pools[s]]

    # Impute NaN with column mean
    for f in range(n_feat):
        col = X[:, f]
        mask = np.isnan(col)
        if mask.all():
            col[:] = 0.0
        elif mask.any():
            col[mask] = np.nanmean(col)

    return X, feat_cols
