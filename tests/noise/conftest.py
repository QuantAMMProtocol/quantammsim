"""Shared fixtures for noise calibration tests."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# synthetic_panel: 3 pools × 10 days = 30 obs (before lag drop → 27 obs)
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_panel() -> pd.DataFrame:
    """3 pools × 10 days with known structure.

    Pool A: MAINNET, WETH/USDC, tier_A=0, tier_B=0, fee=0.003
    Pool B: ARBITRUM, BAL/WETH, tier_A=0, tier_B=1, fee=0.01
    Pool C: BASE, RATS/WETH, tier_A=0, tier_B=2, fee=0.005

    Dates: 2026-01-01 to 2026-01-10
    Weekend flags: Sat 2026-01-03 and Sun 2026-01-04 are weekends.
    (2026-01-01 is Thursday, ..., 01-03 Sat, 01-04 Sun, 01-05 Mon, ...)
    """
    np.random.seed(42)

    pools = [
        ("pool_A", "MAINNET", "WETH,USDC", 0.003, 0, 0),
        ("pool_B", "ARBITRUM", "BAL,WETH", 0.01, 0, 1),
        ("pool_C", "BASE", "RATS,WETH", 0.005, 0, 2),
    ]
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(10)]

    records = []
    for pool_id, chain, tokens, fee, tier_a, tier_b in pools:
        log_tvl_base = 14.0 + np.random.randn() * 0.5
        for d in dates:
            log_tvl = log_tvl_base + np.random.randn() * 0.1
            log_vol = log_tvl - 2.0 + np.random.randn() * 0.3
            vol = 0.3 + np.random.rand() * 0.2
            is_weekend = 1.0 if d.weekday() >= 5 else 0.0

            records.append({
                "pool_id": pool_id,
                "chain": chain,
                "date": d,
                "log_volume": log_vol,
                "log_tvl": log_tvl,
                "volatility": vol,
                "weekend": is_weekend,
                "log_fee": np.log(max(fee, 1e-6)),
                "swap_fee": fee,
                "tier_A": tier_a,
                "tier_B": tier_b,
                "tokens": tokens,
            })

    panel = pd.DataFrame(records)
    panel = panel.sort_values(["pool_id", "date"]).reset_index(drop=True)
    panel["log_tvl_lag1"] = panel.groupby("pool_id")["log_tvl"].shift(1)
    panel = panel.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)

    # Structural model covariates
    panel["log_sigma"] = np.log(np.maximum(panel["volatility"].values, 1e-6))
    dow = panel["date"].apply(
        lambda d: d.weekday() if hasattr(d, "weekday") else pd.Timestamp(d).weekday()
    )
    panel["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    panel["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    panel["tvl_x_sigma"] = panel["log_tvl_lag1"] * panel["log_sigma"]
    panel["tvl_x_fee"] = panel["log_tvl_lag1"] * panel["log_fee"]
    panel["sigma_x_fee"] = panel["log_sigma"] * panel["log_fee"]

    return panel


# ---------------------------------------------------------------------------
# synthetic_encoded_data: output of encode_covariates(synthetic_panel)
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_encoded_data(synthetic_panel):
    from quantammsim.noise_calibration import encode_covariates
    return encode_covariates(synthetic_panel)


# ---------------------------------------------------------------------------
# synthetic_samples: deterministic posterior-like dict
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_samples(synthetic_encoded_data):
    """Deterministic posterior samples with eta=0, L_Omega=I.

    With this structure: theta = X_pool @ B^T exactly.
    """
    data = synthetic_encoded_data
    N_pools = data["N_pools"]
    K_cov = data["K_cov"]
    K_coeff = 4
    S = 10

    np.random.seed(99)
    B = np.random.randn(S, K_coeff, K_cov) * 0.5
    sigma_theta = np.ones((S, K_coeff))
    L_Omega = np.tile(np.eye(K_coeff), (S, 1, 1))
    eta = np.zeros((S, N_pools, K_coeff))
    df = np.full((S,), 5.0)
    sigma_eps = np.tile([0.5, 0.8, 0.6], (S, 1))

    return {
        "B": B,
        "sigma_theta": sigma_theta,
        "L_Omega": L_Omega,
        "eta": eta,
        "df": df,
        "sigma_eps": sigma_eps,
    }


# ---------------------------------------------------------------------------
# synthetic_pools_df: matches enumerate_balancer_pools output schema
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_pools_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "pool_id": "pool_A",
            "chain": "MAINNET",
            "pool_type": "WEIGHTED",
            "tokens": ["WETH", "USDC"],
            "token_addresses": ["0xweth", "0xusdc"],
            "swap_fee": 0.003,
            "current_tvl": 1_000_000,
        },
        {
            "pool_id": "pool_B",
            "chain": "ARBITRUM",
            "pool_type": "WEIGHTED",
            "tokens": ["BAL", "WETH"],
            "token_addresses": ["0xbal", "0xweth"],
            "swap_fee": 0.01,
            "current_tvl": 500_000,
        },
        {
            "pool_id": "pool_C",
            "chain": "BASE",
            "pool_type": "WEIGHTED",
            "tokens": ["RATS", "WETH"],
            "token_addresses": ["0xrats", "0xweth"],
            "swap_fee": 0.005,
            "current_tvl": 100_000,
        },
    ])


# ---------------------------------------------------------------------------
# synthetic_ibp_samples: IBP posterior-like dict (no eta, L_Omega, sigma_theta)
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_ibp_samples(synthetic_encoded_data):
    """Deterministic IBP posterior samples (marginalized model).

    Contains B, W, v_ibp, alpha_ibp — no z_logit, eta, L_Omega, sigma_theta.
    Z is analytically marginalized; MAP assignments are computed from data.
    """
    data = synthetic_encoded_data
    K_cov = data["K_cov"]
    K_coeff = 4
    K_features = 6
    S = 10

    np.random.seed(99)
    B = np.random.randn(S, K_coeff, K_cov) * 0.5
    W = np.random.randn(S, K_features, K_coeff) * 0.3
    v_ibp = np.random.beta(2, 1, size=(S, K_features))
    alpha_ibp = np.full((S,), 2.0)
    sigma_w = np.full((S,), 1.0)
    df = np.full((S,), 5.0)
    sigma_eps = np.full((S,), 0.5)

    return {
        "B": B,
        "W": W,
        "v_ibp": v_ibp,
        "alpha_ibp": alpha_ibp,
        "sigma_w": sigma_w,
        "df": df,
        "sigma_eps": sigma_eps,
    }


# ---------------------------------------------------------------------------
# synthetic_ibp_dp_samples: hybrid IBP+DP posterior-like dict
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_ibp_dp_samples(synthetic_encoded_data):
    """Deterministic hybrid IBP+DP posterior samples.

    Contains both IBP keys (B, W, v_ibp, alpha_ibp, sigma_w) and
    DP keys (v, alpha_dp, sigma_eps as vector). No z_logit, eta, L_Omega,
    sigma_theta.
    """
    data = synthetic_encoded_data
    K_cov = data["K_cov"]
    K_coeff = 4
    K_features = 6
    K_clusters = 6
    S = 10

    np.random.seed(99)
    B = np.random.randn(S, K_coeff, K_cov) * 0.5
    W = np.random.randn(S, K_features, K_coeff) * 0.3
    v_ibp = np.random.beta(2, 1, size=(S, K_features))
    alpha_ibp = np.full((S,), 2.0)
    sigma_w = np.full((S,), 1.0)
    v = np.random.beta(1, 2, size=(S, K_clusters - 1))
    alpha_dp = np.full((S,), 1.0)
    df = np.full((S,), 5.0)
    sigma_eps = np.abs(np.random.randn(S, K_clusters)) + 0.1

    return {
        "B": B,
        "W": W,
        "v_ibp": v_ibp,
        "alpha_ibp": alpha_ibp,
        "sigma_w": sigma_w,
        "v": v,
        "alpha_dp": alpha_dp,
        "df": df,
        "sigma_eps": sigma_eps,
    }


# ---------------------------------------------------------------------------
# synthetic_snapshots_df: matches fetch_all_snapshots output schema
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# synthetic_structural_data: output of encode_covariates_structural()
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_structural_data(synthetic_panel):
    from quantammsim.noise_calibration.covariate_encoding import (
        encode_covariates_structural,
    )
    return encode_covariates_structural(synthetic_panel)


# ---------------------------------------------------------------------------
# synthetic_snapshots_df: matches fetch_all_snapshots output schema
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_snapshots_df() -> pd.DataFrame:
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(10)]
    records = []
    np.random.seed(42)
    for pool_id, chain in [("pool_A", "MAINNET"), ("pool_B", "ARBITRUM"),
                            ("pool_C", "BASE")]:
        for d in dates:
            records.append({
                "pool_id": pool_id,
                "chain": chain,
                "date": d,
                "volume_usd": np.exp(10.0 + np.random.randn() * 0.5),
                "total_liquidity_usd": np.exp(14.0 + np.random.randn() * 0.3),
            })
    return pd.DataFrame(records)
