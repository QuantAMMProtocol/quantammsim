"""OLS calibration of the Tsoukalas noise volume model for reClAMM pools.

Fits the structural volume equation:
    V_daily/1e6 = a_0 + a_sigma*sigma + a_c*sqrt(c_eff/1e6)

where c_eff = (Ra+Va)*pA + (Rb+Vb)*pB is the effective TVL (real + virtual).

From daily pool snapshots (volume, TVL, volatility). Outputs a noise_params dict
compatible with run_fingerprint["reclamm_noise_params"].

Usage:
    # From a pre-assembled CSV
    python scripts/calibrate_reclamm_noise.py --csv daily_data.csv --base-fee 0.003

    # End-to-end from API + DB + parquets
    python scripts/calibrate_reclamm_noise.py --pool cbBTC_WETH
"""

import argparse
import json
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Balancer V3 API
# ---------------------------------------------------------------------------

BALANCER_API_URL = "https://api-v3.balancer.fi/"

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


def fetch_balancer_snapshots(chain, pool_address, start_ts, end_ts,
                             base_url=BALANCER_API_URL):
    """Fetch daily pool snapshots from Balancer V3 GraphQL API.

    Parameters
    ----------
    chain : str
        Chain name (e.g. 'base', 'ethereum').
    pool_address : str
        Pool contract address (hex, no 0x prefix).
    start_ts : int
        Start unix timestamp (seconds).
    end_ts : int
        End unix timestamp (seconds).
    base_url : str
        Balancer API base URL.

    Returns
    -------
    pd.DataFrame
        Columns: date, volume_usd, total_liquidity_usd. Indexed by date string.
    """
    api_chain = BALANCER_API_CHAIN.get(chain)
    if api_chain is None:
        raise ValueError(f"Unknown chain for Balancer API: {chain!r}")

    pool_id = f"0x{pool_address}" if not pool_address.startswith("0x") else pool_address

    # Paginate: API may limit results. Fetch in 90-day windows.
    all_snapshots = []
    window = 90 * 86400
    cursor = start_ts

    while cursor < end_ts:
        window_end = min(cursor + window, end_ts)
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
                "chain": api_chain,
                "range": "ALL_TIME",
            },
        }

        data = json.dumps(query).encode("utf-8")
        req = urllib.request.Request(
            base_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "quantammsim/1.0",
            },
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        snapshots = body.get("data", {}).get("poolGetSnapshots", [])
        if not snapshots:
            break

        for snap in snapshots:
            ts = int(snap["timestamp"])
            if start_ts <= ts <= end_ts:
                all_snapshots.append({
                    "timestamp": ts,
                    "volume_usd": float(snap["volume24h"]),
                    "total_liquidity_usd": float(snap["totalLiquidity"]),
                })

        # The API returns ALL_TIME, so no need to paginate further
        break

    if not all_snapshots:
        raise ValueError(
            f"No Balancer snapshots for {pool_id} on {chain} "
            f"between {start_ts} and {end_ts}"
        )

    df = pd.DataFrame(all_snapshots)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    # Deduplicate by date (keep last snapshot per day)
    df = df.sort_values("timestamp").drop_duplicates("date", keep="last")
    return df.set_index("date")


# ---------------------------------------------------------------------------
# DB-based daily pool state
# ---------------------------------------------------------------------------

def load_daily_pool_state(pool, db_path, data_root):
    """Load daily pool state from pools_history.db, compute effective TVL.

    Parameters
    ----------
    pool : PoolConfig
        Pool configuration (from pool_registry).
    db_path : str
        Path to pools_history.db.
    data_root : str
        Directory containing {TICKER}_USD.parquet files.

    Returns
    -------
    pd.DataFrame
        Indexed by date, columns: effective_tvl_usd, real_tvl_usd.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        f"""SELECT timestamp, balance_0, balance_1, virtual_0, virtual_1
            FROM {pool.db_label}
            ORDER BY timestamp"""
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No DB data for {pool.db_label}")

    df = pd.DataFrame(rows, columns=["timestamp", "bal_0", "bal_1", "virt_0", "virt_1"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date

    # Keep last snapshot per day
    daily = df.sort_values("timestamp").drop_duplicates("date", keep="last").set_index("date")

    # Load USD prices for each token
    if pool.reverse:
        tickers_in_db_order = [pool.tokens[1], pool.tokens[0]]
    else:
        tickers_in_db_order = [pool.tokens[0], pool.tokens[1]]

    price_dfs = {}
    for ticker in tickers_in_db_order:
        if ticker == "USDC":
            price_dfs[ticker] = None  # constant $1
        else:
            path = os.path.join(data_root, f"{ticker}_USD.parquet")
            pdf = pd.read_parquet(path)
            pdf["date"] = pd.to_datetime(pdf["unix"], unit="ms").dt.date
            # Daily close: last price per day
            price_dfs[ticker] = (
                pdf.sort_values("unix")
                .drop_duplicates("date", keep="last")
                .set_index("date")["close"]
            )

    # Compute USD prices at each daily snapshot
    records = []
    for date, row in daily.iterrows():
        b0, b1, v0, v1 = row["bal_0"], row["bal_1"], row["virt_0"], row["virt_1"]

        p0 = 1.0 if tickers_in_db_order[0] == "USDC" else price_dfs[tickers_in_db_order[0]].get(date, np.nan)
        p1 = 1.0 if tickers_in_db_order[1] == "USDC" else price_dfs[tickers_in_db_order[1]].get(date, np.nan)

        if np.isnan(p0) or np.isnan(p1):
            continue

        real_tvl = b0 * p0 + b1 * p1
        effective_tvl = (b0 + v0) * p0 + (b1 + v1) * p1

        records.append({
            "date": date,
            "real_tvl_usd": real_tvl,
            "effective_tvl_usd": effective_tvl,
        })

    result = pd.DataFrame(records).set_index("date")
    return result


# ---------------------------------------------------------------------------
# Daily volatility from price parquets
# ---------------------------------------------------------------------------

def compute_daily_volatility(tokens, data_root, start_ts, end_ts):
    """Compute daily annualised volatility of the price ratio.

    Uses 5-minute subsampled log returns within each day, then
    annualises with sqrt(365).

    Parameters
    ----------
    tokens : list
        Token tickers in quantammsim sorted order (e.g. ['BTC', 'ETH']).
    data_root : str
        Directory containing {TICKER}_USD.parquet files.
    start_ts : int
        Start unix timestamp (seconds).
    end_ts : int
        End unix timestamp (seconds).

    Returns
    -------
    pd.Series
        Indexed by date, values are annualised daily volatility.
    """
    # Load minute-level prices for both tokens
    prices = {}
    for ticker in tokens:
        if ticker == "USDC":
            prices[ticker] = None
        else:
            path = os.path.join(data_root, f"{ticker}_USD.parquet")
            df = pd.read_parquet(path)
            df = df[(df["unix"] >= start_ts * 1000) & (df["unix"] <= end_ts * 1000)]
            df["datetime"] = pd.to_datetime(df["unix"], unit="ms")
            df = df.set_index("datetime")["close"]
            prices[ticker] = df

    # Compute price ratio (token[0] / token[1])
    t0, t1 = tokens[0], tokens[1]
    if prices[t0] is not None and prices[t1] is not None:
        # Align on common timestamps
        combined = pd.DataFrame({"p0": prices[t0], "p1": prices[t1]}).dropna()
        ratio = combined["p0"] / combined["p1"]
    elif prices[t0] is not None:
        ratio = prices[t0]  # t1 is USDC ($1)
    elif prices[t1] is not None:
        ratio = 1.0 / prices[t1]  # t0 is USDC
    else:
        raise ValueError("Both tokens are USDC — cannot compute ratio")

    # Subsample to 5-min intervals
    ratio_5m = ratio.resample("5min").last().dropna()
    log_returns = np.log(ratio_5m / ratio_5m.shift(1)).dropna()

    # Group by date, compute daily vol
    log_returns_df = log_returns.to_frame("lr")
    log_returns_df["date"] = log_returns_df.index.date

    daily_vol = log_returns_df.groupby("date")["lr"].std()
    # Annualise: each day has ~288 5-min periods, scale by sqrt(288 * 365)
    daily_vol_ann = daily_vol * np.sqrt(288 * 365)

    return daily_vol_ann


# ---------------------------------------------------------------------------
# Calibration DataFrame assembly
# ---------------------------------------------------------------------------

def build_calibration_df(pool, data_root=None):
    """Build daily calibration DataFrame from Balancer API + price parquets.

    All pool state (volume, effective TVL) comes from the Balancer V3 API.
    The API's ``totalLiquidity`` is the effective TVL: for a reClAMM pool
    on Balancer V3, the router sees real + virtual reserves, and
    ``totalLiquidity`` reflects that full depth. Only the volatility
    computation requires price parquets.

    Parameters
    ----------
    pool : PoolConfig
        Pool configuration (must have pool_address field).
    data_root : str, optional
        Directory containing {TICKER}_USD.parquet price files.

    Returns
    -------
    pd.DataFrame
        Columns: volume_usd, effective_tvl_usd, volatility. Indexed by date.
    """
    from experiments.pool_registry import (
        get_data_end_date,
        _date_to_unix,
    )

    if data_root is None:
        data_root = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "quantammsim", "data",
        )

    start_ts = _date_to_unix(pool.plausible_start)
    end_str = get_data_end_date(pool.tokens, data_root)
    end_ts = _date_to_unix(end_str)

    print(f"  Fetching Balancer snapshots for {pool.label} "
          f"({pool.chain}, {pool.pool_address})...")
    api_df = fetch_balancer_snapshots(
        pool.chain, pool.pool_address, start_ts, end_ts,
    )
    print(f"    Got {len(api_df)} daily snapshots from API")
    print(f"    TVL range: ${api_df['total_liquidity_usd'].min():,.0f} — "
          f"${api_df['total_liquidity_usd'].max():,.0f}")

    print(f"  Computing daily volatility from price parquets...")
    vol_series = compute_daily_volatility(pool.tokens, data_root, start_ts, end_ts)
    print(f"    Got {len(vol_series)} daily volatility values")

    # Assemble: volume + TVL from API, volatility from parquets
    combined = api_df[["volume_usd", "total_liquidity_usd"]].copy()
    combined = combined.rename(columns={"total_liquidity_usd": "effective_tvl_usd"})
    combined["volatility"] = vol_series
    combined = combined.dropna()

    print(f"  Combined: {len(combined)} days after join")
    return combined


# ---------------------------------------------------------------------------
# OLS calibration
# ---------------------------------------------------------------------------

def run_ols_calibration(daily_df, base_fee, model="sqrt"):
    """OLS regression for Tsoukalas model params.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Must contain columns: volume_usd, volatility, effective_tvl_usd.
    base_fee : float
        Static swap fee (e.g. 0.003).
    model : str
        'sqrt' or 'log' — TVL regressor transformation.

    Returns
    -------
    noise_params : dict
        Coefficients for run_fingerprint["reclamm_noise_params"].
    diagnostics : dict
        Standard errors, R², residual summary.
    """
    if model == "loglinear":
        # Multiplicative model: log(V) = b_0 + b_sigma·σ + b_c·log(TVL)
        # Implies: V = exp(b_0) · TVL^b_c · exp(b_sigma·σ)
        mask = daily_df["volume_usd"].values > 0
        n_dropped = int((~mask).sum())
        df_fit = daily_df[mask]

        y_log = np.log(df_fit["volume_usd"].values)
        X = np.column_stack([
            np.ones(len(df_fit)),
            df_fit["volatility"].values,
            np.log(df_fit["effective_tvl_usd"].values),
        ])

        beta, _, _, _ = np.linalg.lstsq(X, y_log, rcond=None)
        b_0, b_sigma, b_c = beta

        residuals = y_log - X @ beta
        n, k = X.shape
        bread = np.linalg.inv(X.T @ X)
        hc1_scale = n / max(n - k, 1)
        meat = X.T @ np.diag(residuals**2 * hc1_scale) @ X
        robust_cov = bread @ meat @ bread
        se = np.sqrt(np.diag(robust_cov))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_log - y_log.mean())**2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

        # Pseudo-R² in levels (median predictor)
        y_pred_level = np.exp(X @ beta)
        y_actual_level = df_fit["volume_usd"].values
        res_level = y_actual_level - y_pred_level
        r_sq_level = 1.0 - np.sum(res_level**2) / max(
            np.sum((y_actual_level - y_actual_level.mean())**2), 1e-30)

        noise_params = {
            "b_0": float(b_0), "b_sigma": float(b_sigma),
            "b_c": float(b_c), "base_fee": float(base_fee),
        }
        diagnostics = {
            "se": {"b_0": float(se[0]), "b_sigma": float(se[1]),
                   "b_c": float(se[2])},
            "r_squared": float(r_squared),
            "r_squared_level": float(r_sq_level),
            "n_obs": int(n),
            "n_dropped_zero": n_dropped,
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "smearing_factor": float(np.exp(np.var(residuals, ddof=1) / 2)),
            "model": "loglinear",
        }
        return noise_params, diagnostics

    # --- Linear models (sqrt / log) ---
    y = daily_df["volume_usd"].values / 1e6

    if model == "sqrt":
        tvl_eff = np.sqrt(daily_df["effective_tvl_usd"].values / 1e6)
    elif model == "log":
        tvl_eff = np.log(np.maximum(daily_df["effective_tvl_usd"].values / 1e6, 1e-30))
    else:
        raise ValueError(f"Unknown model: {model!r}. Use 'sqrt', 'log', or 'loglinear'.")

    X = np.column_stack([
        np.ones(len(daily_df)),         # a_0
        daily_df["volatility"].values,  # a_sigma
        tvl_eff,                        # a_c
    ])

    beta, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    a_0, a_sigma, a_c = beta

    # Heteroskedasticity-robust standard errors (HC1)
    residuals = y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    hc1_scale = n / max(n - k, 1)
    meat = X.T @ np.diag(residuals**2 * hc1_scale) @ X
    robust_cov = bread @ meat @ bread
    se = np.sqrt(np.diag(robust_cov))

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

    noise_params = {
        "a_0_base": float(a_0),
        "a_f": 0.0,  # not identified with static fees
        "a_sigma": float(a_sigma),
        "a_c": float(a_c),
        "base_fee": float(base_fee),
    }

    diagnostics = {
        "se": dict(zip(
            ["a_0", "a_sigma", "a_c"],
            se.tolist(),
        )),
        "r_squared": float(r_squared),
        "n_obs": int(n),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "model": model,
    }

    return noise_params, diagnostics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration_diagnostics(daily_df, noise_params, diagnostics,
                                 pool_label="", model="sqrt",
                                 output_dir="results"):
    """Generate diagnostic plots for the noise volume calibration.

    Produces a 2×2 figure:
      Top-left:  Time series — real vs predicted daily volume + effective TVL
      Top-right: Scatter — predicted vs actual with 45° line
      Bot-left:  Residuals vs time + residuals vs fitted
      Bot-right: Component decomposition (stacked contributions)

    Parameters
    ----------
    daily_df : pd.DataFrame
        Calibration DataFrame (indexed by date).
    noise_params : dict
        Fitted coefficients from run_ols_calibration.
    diagnostics : dict
        Diagnostics dict from run_ols_calibration.
    pool_label : str
        Pool name for titles.
    model : str
        'sqrt' or 'log'.
    output_dir : str
        Directory for output PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import matplotlib.dates as mdates

    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    dates = pd.to_datetime(daily_df.index)
    y_actual = daily_df["volume_usd"].values / 1e6  # in $M
    eff_tvl = daily_df["effective_tvl_usd"].values
    vol = daily_df["volatility"].values

    r2 = diagnostics["r_squared"]
    n = diagnostics["n_obs"]

    if model == "loglinear":
        b_0 = noise_params["b_0"]
        b_sigma = noise_params["b_sigma"]
        b_c = noise_params["b_c"]
        log_tvl = np.log(np.maximum(eff_tvl, 1.0))
        y_pred_log = b_0 + b_sigma * vol + b_c * log_tvl
        y_pred = np.exp(y_pred_log) / 1e6  # median prediction in $M

        # Log-space residuals
        mask_pos = daily_df["volume_usd"].values > 0
        residuals = np.full(len(dates), np.nan)
        residuals[mask_pos] = (
            np.log(daily_df["volume_usd"].values[mask_pos])
            - y_pred_log[mask_pos]
        )
        resid_unit = "log scale"
        r2_level = diagnostics.get("r_squared_level")
    else:
        a_0 = noise_params["a_0_base"]
        a_sigma = noise_params["a_sigma"]
        a_c = noise_params["a_c"]

        if model == "sqrt":
            tvl_term = a_c * np.sqrt(eff_tvl / 1e6)
        else:
            tvl_term = a_c * np.log(np.maximum(eff_tvl / 1e6, 1e-30))

        y_pred = a_0 + a_sigma * vol + tvl_term
        residuals = y_actual - y_pred
        resid_unit = "$M"
        r2_level = None

    # --- Figure 1: Main diagnostics (2×2) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Time series: real vs predicted + TVL on secondary axis
    ax = axes[0, 0]
    ax.plot(dates, y_actual, color="steelblue", alpha=0.7, linewidth=1,
            label="Actual volume")
    ax.plot(dates, y_pred, color="crimson", linewidth=1.5,
            label="Predicted volume")
    ax.set_ylabel("Daily volume ($M)", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.legend(loc="upper left", fontsize=8)
    r2_str = (f"R²(log)={r2:.3f}, R²(level)={r2_level:.3f}"
              if r2_level is not None else f"R²={r2:.3f}")
    ax.set_title(f"Daily volume: actual vs predicted  ({r2_str}, n={n})")
    ax.xaxis.set_major_formatter(DateFormatter("%b %y"))

    ax2 = ax.twinx()
    ax2.fill_between(dates, eff_tvl / 1e6, alpha=0.15, color="green",
                     label="Effective TVL ($M)")
    ax2.set_ylabel("Effective TVL ($M)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="upper right", fontsize=8)

    # (0,1) Scatter: predicted vs actual
    ax = axes[0, 1]
    ax.scatter(y_pred, y_actual, alpha=0.5, s=15, color="steelblue",
               edgecolors="none")
    lims = [min(y_pred.min(), y_actual.min()), max(y_pred.max(), y_actual.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="45° line")
    ax.set_xlabel("Predicted ($M)")
    ax.set_ylabel("Actual ($M)")
    ax.set_title("Predicted vs actual")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    # (1,0) Residuals: vs time (top) and vs fitted (bottom)
    ax = axes[1, 0]
    ax.scatter(dates, residuals, alpha=0.5, s=12, color="steelblue",
               edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    # 7-day rolling mean of residuals
    res_series = pd.Series(residuals, index=dates)
    rolling_mean = res_series.rolling(7, min_periods=1).mean()
    ax.plot(dates, rolling_mean, color="crimson", linewidth=1.5,
            label="7-day rolling mean")
    ax.set_ylabel(f"Residual ({resid_unit})")
    ax.set_title("Residuals vs time")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(DateFormatter("%b %y"))

    # (1,1) Component decomposition
    ax = axes[1, 1]
    if model == "loglinear":
        # Log-space additive decomposition as line plots
        log_tvl_plot = np.log(np.maximum(eff_tvl, 1.0))
        comp_base = np.full(len(dates), b_0)
        comp_tvl = b_c * log_tvl_plot
        comp_vol = b_sigma * vol
        total_log = comp_base + comp_tvl + comp_vol
        vol_usd = daily_df["volume_usd"].values.copy()
        vol_usd[vol_usd <= 0] = np.nan
        actual_log = np.log(vol_usd)
        ax.plot(dates, comp_base, color="grey", linestyle="--", linewidth=1,
                label=f"b_0 = {b_0:.2f}")
        ax.plot(dates, comp_base + comp_tvl, color="green", linewidth=1.5,
                label=f"b_0 + b_c·log(TVL) (b_c={b_c:.4f})")
        ax.plot(dates, total_log, color="crimson", linewidth=1.5,
                label="Full prediction")
        ax.scatter(dates, actual_log, color="steelblue", s=10, alpha=0.5,
                   label="Actual log(V)", zorder=5)
        ax.set_ylabel("log(Volume, USD)")
        ax.set_title("Component decomposition (log space)")

        fig.suptitle(
            f"{pool_label} — noise calibration ({model})\n"
            f"log(V) = {b_0:.2f} + {b_sigma:.4f}·σ + {b_c:.4f}·log(TVL)",
            fontsize=11,
        )
    else:
        intercept_contrib = np.full(len(dates), a_0)
        vol_contrib = a_sigma * vol
        tvl_contrib = tvl_term

        ax.fill_between(dates, 0, intercept_contrib, alpha=0.3, color="grey",
                        label=f"a_0 = {a_0:.4f}")
        ax.fill_between(dates, intercept_contrib, intercept_contrib + vol_contrib,
                        alpha=0.3, color="orange",
                        label=f"a_σ·σ (a_σ={a_sigma:.4f})")
        ax.fill_between(dates, intercept_contrib + vol_contrib,
                        intercept_contrib + vol_contrib + tvl_contrib,
                        alpha=0.3, color="green",
                        label=f"a_c·{model}(TVL) (a_c={a_c:.4f})")
        ax.plot(dates, y_actual, color="steelblue", linewidth=1, alpha=0.7,
                label="Actual")
        ax.set_ylabel("Volume ($M)")
        ax.set_title("Component decomposition")

        fig.suptitle(
            f"{pool_label} — Tsoukalas noise calibration ({model})\n"
            f"V/1e6 = {a_0:.4f} + {a_sigma:.4f}·σ + {a_c:.4f}·{model}(TVL_eff/1e6)",
            fontsize=11,
        )
    ax.legend(fontsize=7, loc="upper left")
    ax.xaxis.set_major_formatter(DateFormatter("%b %y"))
    plt.tight_layout()

    fname = f"noise_calibration_{pool_label}_{model}.png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 2: Residuals vs each regressor ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    # Residuals vs volatility
    ax = axes2[0]
    ax.scatter(vol, residuals, alpha=0.5, s=12, color="orange", edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Volatility (annualised)")
    ax.set_ylabel(f"Residual ({resid_unit})")
    ax.set_title("Residuals vs volatility")

    # Residuals vs effective TVL
    ax = axes2[1]
    ax.scatter(eff_tvl / 1e6, residuals, alpha=0.5, s=12, color="green",
               edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effective TVL ($M)")
    ax.set_ylabel(f"Residual ({resid_unit})")
    ax.set_title("Residuals vs effective TVL")

    # Residuals vs fitted
    ax = axes2[2]
    ax.scatter(y_pred, residuals, alpha=0.5, s=12, color="steelblue",
               edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Fitted ($M)")
    ax.set_ylabel(f"Residual ({resid_unit})")
    ax.set_title("Residuals vs fitted")

    fig2.suptitle(f"{pool_label} — Residual diagnostics ({model})", fontsize=11)
    plt.tight_layout()

    fname2 = f"noise_residuals_{pool_label}_{model}.png"
    path2 = os.path.join(output_dir, fname2)
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    return path, path2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Tsoukalas noise volume model for reClAMM"
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to CSV with columns: volume_usd, volatility, effective_tvl_usd",
    )
    parser.add_argument(
        "--pool", default=None,
        help="Pool label from pool_registry (e.g. cbBTC_WETH) for end-to-end calibration",
    )
    parser.add_argument("--base-fee", type=float, default=None,
                        help="Override base fee (default: use pool's swap_fee)")
    parser.add_argument("--model", choices=["sqrt", "log", "loglinear"],
                        default="sqrt")
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate diagnostic plots (saved to --output-dir)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for diagnostic plots (default: results)",
    )
    args = parser.parse_args()

    if args.csv is None and args.pool is None:
        parser.error("One of --csv or --pool is required")

    if args.pool is not None:
        # End-to-end mode: fetch data, assemble, calibrate
        from experiments.pool_registry import POOL_REGISTRY

        if args.pool not in POOL_REGISTRY:
            print(f"Unknown pool: {args.pool}", file=sys.stderr)
            print(f"Available: {list(POOL_REGISTRY.keys())}", file=sys.stderr)
            sys.exit(1)

        pool = POOL_REGISTRY[args.pool]
        base_fee = args.base_fee if args.base_fee is not None else pool.swap_fee

        print(f"Calibrating noise model for {pool.label} ({pool.chain})")
        print(f"  Swap fee: {base_fee}")
        print(f"  Model: {args.model}")

        df = build_calibration_df(pool)
    else:
        # CSV mode
        df = pd.read_csv(args.csv)
        required_cols = {"volume_usd", "volatility", "effective_tvl_usd"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"Error: missing columns: {missing}", file=sys.stderr)
            sys.exit(1)
        base_fee = args.base_fee if args.base_fee is not None else 0.003

    noise_params, diagnostics = run_ols_calibration(df, base_fee, args.model)

    # Print diagnostics
    print(f"\n  OLS Results ({args.model} model):")
    print(f"    R² = {diagnostics['r_squared']:.4f}")
    if "r_squared_level" in diagnostics:
        print(f"    R²(level) = {diagnostics['r_squared_level']:.4f}")
    if "n_dropped_zero" in diagnostics and diagnostics["n_dropped_zero"] > 0:
        print(f"    Dropped {diagnostics['n_dropped_zero']} zero-volume days")
    if "smearing_factor" in diagnostics:
        print(f"    Smearing factor = {diagnostics['smearing_factor']:.4f}  "
              f"(E[V]/median[V])")
    print(f"    n  = {diagnostics['n_obs']}")
    print(f"    Coefficients:")
    if args.model == "loglinear":
        coef_keys = ["b_0", "b_sigma", "b_c"]
    else:
        coef_keys = ["a_0", "a_sigma", "a_c"]
    for key in coef_keys:
        param_key = "a_0_base" if key == "a_0" else key
        val = noise_params[param_key]
        se = diagnostics["se"][key]
        t_stat = val / se if se > 0 else float("inf")
        print(f"      {key:>8} = {val:>10.4f}  (SE={se:.4f}, t={t_stat:.2f})")
    print(f"    Residual: mean={diagnostics['residual_mean']:.6f}, "
          f"std={diagnostics['residual_std']:.4f}")

    # Plot diagnostics
    if args.plot:
        label = args.pool if args.pool else "custom"
        plot_calibration_diagnostics(
            df, noise_params, diagnostics,
            pool_label=label, model=args.model,
            output_dir=args.output_dir,
        )

    result = {
        "noise_params": noise_params,
        "diagnostics": diagnostics,
    }

    output_str = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str + "\n")
        print(f"\nWrote calibration to {args.output}", file=sys.stderr)
    else:
        print(f"\n{output_str}")


if __name__ == "__main__":
    main()
