"""Fixtures for calibration pipeline tests."""

import numpy as np
import pandas as pd
import pytest


# ── Constants ──────────────────────────────────────────────────────────────

N_CADENCES = 3
N_GAS = 3
N_DAYS = 15
N_POOLS = 2
K_OBS = 8

CADENCES = np.array([1.0, 12.0, 60.0])
GAS_COSTS = np.array([0.0, 1.0, 5.0])

# Pool ID prefixes (16 chars) that map to full 66-char pool IDs
POOL_PREFIXES = ["0xaaaa11112222aa", "0xbbbb33334444bb"]
POOL_IDS_FULL = [
    "0xaaaa11112222aa63ae5d458857e731c129069f29000200000000000000000588",
    "0xbbbb33334444bb9c8ef030ab642b10820db8f56000200000000000000000014",
]


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_daily_grid():
    """Small per-day grid DataFrame: 3 cadences x 3 gas_costs x 15 days.

    V_arb decreasing in cadence and gas, with daily sinusoidal variation.
    """
    np.random.seed(42)
    dates = pd.date_range("2025-12-01", periods=N_DAYS, freq="D")

    rows = []
    for ci, cad in enumerate(CADENCES):
        for gi, gas in enumerate(GAS_COSTS):
            base = 10000.0 / (1 + 0.3 * ci) / (1 + 0.5 * gi)
            for di, date in enumerate(dates):
                daily_var = 1 + 0.1 * np.sin(2 * np.pi * di / 7)
                vol = base * daily_var + np.random.normal(0, base * 0.01)
                rows.append({
                    "cadence": cad,
                    "gas_cost": gas,
                    "date": date,
                    "daily_arb_volume": max(vol, 0),
                })

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_panel():
    """Minimal panel DataFrame: 2 pools x 15 days.

    Columns match the real panel.parquet schema.
    """
    np.random.seed(42)
    dates = pd.date_range("2025-12-01", periods=N_DAYS, freq="D")

    rows = []
    for pi, (prefix, full_id) in enumerate(zip(POOL_PREFIXES, POOL_IDS_FULL)):
        chain = "MAINNET" if pi == 0 else "ARBITRUM"
        base_tvl = 12.0 + pi  # log TVL
        base_vol = 9.0 + pi
        fee = 0.003 if pi == 0 else 0.01

        for di, date in enumerate(dates):
            tvl = base_tvl + 0.05 * np.sin(2 * np.pi * di / 30)
            vol = base_vol + 0.3 * np.random.randn()
            sigma = 0.4 + 0.1 * np.random.randn()
            rows.append({
                "pool_id": full_id,
                "chain": chain,
                "date": date,
                "log_volume": vol,
                "log_tvl": tvl,
                "log_tvl_lag1": tvl - 0.01 if di > 0 else np.nan,
                "volatility": max(sigma, 0.01),
                "weekend": 1 if date.weekday() >= 5 else 0,
                "log_fee": np.log(fee),
                "swap_fee": fee,
                "tier_A": "major" if pi == 0 else "mid",
                "tier_B": "major",
                "tokens": "BTC,ETH" if pi == 0 else "AAVE,ETH",
                "total_shares": 1e6 * (1 + 0.01 * di),
            })

    df = pd.DataFrame(rows)
    # Drop rows where log_tvl_lag1 is NaN (first day per pool)
    df = df.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)
    return df


@pytest.fixture
def synthetic_pool_coeffs(synthetic_daily_grid):
    """PoolCoeffsDaily built from synthetic_daily_grid."""
    from quantammsim.calibration.grid_interpolation import precompute_pool_coeffs_daily
    return precompute_pool_coeffs_daily(synthetic_daily_grid)


@pytest.fixture
def synthetic_x_obs(synthetic_panel):
    """NumPy array (n_obs, K_OBS) from synthetic panel for one pool."""
    from quantammsim.calibration.pool_data import build_x_obs
    pool0 = synthetic_panel[
        synthetic_panel["pool_id"] == POOL_IDS_FULL[0]
    ]
    return build_x_obs(pool0)


@pytest.fixture
def synthetic_pool_fit_result():
    """Dict with per-pool fitted params for testing learned mapping."""
    np.random.seed(42)
    results = {}
    for prefix in POOL_PREFIXES:
        results[prefix] = {
            "log_cadence": np.log(12.0) + 0.1 * np.random.randn(),
            "log_gas": np.log(1.0) + 0.1 * np.random.randn(),
            "noise_coeffs": np.random.randn(K_OBS) * 0.1,
            "loss": 0.5 + 0.1 * np.random.rand(),
            "converged": True,
            "cadence_minutes": 12.0,
            "gas_usd": 1.0,
            "chain": "MAINNET" if prefix == POOL_PREFIXES[0] else "ARBITRUM",
            "fee": 0.003 if prefix == POOL_PREFIXES[0] else 0.01,
            "tokens": "BTC/ETH" if prefix == POOL_PREFIXES[0] else "AAVE/ETH",
        }
    return results
