"""Covariate encoding for the hierarchical noise model."""

import numpy as np
import pandas as pd

from .constants import K_COEFF, K_OBS_COEFF, GAS_COSTS


def encode_covariates(panel: pd.DataFrame, include_tiers: bool = True) -> dict:
    """Build NumPyro-ready arrays from the panel DataFrame.

    Returns dict with arrays for the model plus metadata for output/prediction.
    Key difference from hierarchical script: x_obs uses log_tvl_lag1 not log_tvl.
    """
    pool_meta = panel.drop_duplicates("pool_id").reset_index(drop=True)
    pool_ids = pool_meta["pool_id"].values
    pool_id_to_idx = {pid: i for i, pid in enumerate(pool_ids)}
    N_pools = len(pool_ids)

    pool_idx = panel["pool_id"].map(pool_id_to_idx).values

    # --- Build X_pool (pool-level covariates, data-driven) ---
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
    if include_tiers:
        for t in tier_a_vals[1:]:
            tier_a_cols.append(
                (pool_meta["tier_A"].astype(str) == t).astype(float).values
            )
            tier_a_names.append(f"tier_A_{t}")

    tier_b_vals = sorted(pool_meta["tier_B"].astype(str).unique())
    ref_tier_b = tier_b_vals[0]
    tier_b_cols = []
    tier_b_names = []
    if include_tiers:
        for t in tier_b_vals[1:]:
            tier_b_cols.append(
                (pool_meta["tier_B"].astype(str) == t).astype(float).values
            )
            tier_b_names.append(f"tier_B_{t}")

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
    K_cov = X_pool.shape[1]

    # --- Observation-level arrays (uses LAGGED TVL) ---
    x_obs = np.column_stack([
        np.ones(len(panel)),
        panel["log_tvl_lag1"].values,
        panel["volatility"].values,
        panel["weekend"].values,
    ]).astype(np.float64)

    y_obs = panel["log_volume"].values.astype(np.float64)

    # --- Per-pool tier_A index for per-tier sigma_eps ---
    tier_A_per_pool = pool_meta["tier_A"].values.astype(np.int32)

    print(f"  Encoded: N_obs={len(y_obs)}, N_pools={N_pools}, "
          f"K_coeff={K_COEFF}, K_cov={K_cov}")
    print(f"  Covariates: {col_names}")
    print(f"  Tier distribution: "
          f"T0={np.sum(tier_A_per_pool == 0)}, "
          f"T1={np.sum(tier_A_per_pool == 1)}, "
          f"T2={np.sum(tier_A_per_pool == 2)}")

    return {
        "pool_idx": pool_idx.astype(np.int32),
        "X_pool": X_pool.astype(np.float64),
        "x_obs": x_obs,
        "y_obs": y_obs,
        "pool_ids": list(pool_ids),
        "pool_meta": pool_meta,
        "covariate_names": col_names,
        "tier_A_per_pool": tier_A_per_pool,
        "N_pools": N_pools,
        "K_cov": K_cov,
        "ref_chain": ref_chain,
        "ref_tier_a": ref_tier_a,
        "ref_tier_b": ref_tier_b,
        "chains": chains,
    }


def _tier_pair_idx(a: int, b: int) -> int:
    """Encode (tier_A, tier_B) pair as a single index.

    Upper triangle of 3x3 grid:
    (0,0)->0, (0,1)->1, (0,2)->2, (1,1)->3, (1,2)->4, (2,2)->5.
    """
    return a * (5 - a) // 2 + b - a


def encode_covariates_structural(
    panel: pd.DataFrame,
    gas: np.ndarray = None,
) -> dict:
    """Build NumPyro-ready arrays for the structural mixture model.

    Extends encode_covariates with:
    - x_obs: 8 columns (intercept, tvl, log_sigma, interactions, DOW harmonics)
    - Additional arrays: sigma_daily, fee, gas, chain_idx, tier_idx, lag_log_tvl
    - n_chains, n_tiers computed from panel

    Parameters
    ----------
    panel : pd.DataFrame
        Output of assemble_panel(), must have log_sigma, dow_sin, dow_cos,
        tvl_x_sigma, tvl_x_fee, sigma_x_fee columns.
    gas : np.ndarray, optional
        Per-observation gas costs in USD. If None, uses default (0.01 for all).
    """
    # Ensure structural columns exist (compute from base columns if missing)
    if "log_sigma" not in panel.columns:
        panel = panel.copy()
        panel["log_sigma"] = np.log(np.maximum(panel["volatility"].values, 1e-6))
        dow = panel["date"].apply(
            lambda d: d.weekday() if hasattr(d, "weekday")
            else pd.Timestamp(d).weekday()
        )
        panel["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        panel["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        panel["tvl_x_sigma"] = panel["log_tvl_lag1"] * panel["log_sigma"]
        panel["tvl_x_fee"] = panel["log_tvl_lag1"] * panel["log_fee"]
        panel["sigma_x_fee"] = panel["log_sigma"] * panel["log_fee"]

    # Reuse X_pool construction from encode_covariates (with tiers for gating)
    base = encode_covariates(panel, include_tiers=True)

    # --- Observation-level x_obs: 8 columns ---
    x_obs = np.column_stack([
        np.ones(len(panel)),                    # intercept
        panel["log_tvl_lag1"].values,           # lagged TVL
        panel["log_sigma"].values,              # log(volatility)
        panel["tvl_x_sigma"].values,            # tvl × sigma interaction
        panel["tvl_x_fee"].values,              # tvl × fee interaction
        panel["sigma_x_fee"].values,            # sigma × fee interaction
        panel["dow_sin"].values,                # DOW harmonic sin
        panel["dow_cos"].values,                # DOW harmonic cos
    ]).astype(np.float64)

    # --- Additional arrays for the structural model ---
    sigma_daily = (panel["volatility"] / np.sqrt(365.0)).values.astype(np.float64)
    fee_per_obs = np.exp(panel["log_fee"].values).astype(np.float64)
    lag_log_tvl = panel["log_tvl_lag1"].values.astype(np.float64)

    # Gas: per-observation
    if gas is not None:
        gas_arr = np.asarray(gas, dtype=np.float64)
    else:
        gas_arr = np.full(len(panel), 0.01, dtype=np.float64)

    # Chain index: integer per pool
    pool_meta = base["pool_meta"]
    chains = base["chains"]
    chain_to_idx = {c: i for i, c in enumerate(chains)}
    chain_idx_per_pool = np.array(
        [chain_to_idx[c] for c in pool_meta["chain"]], dtype=np.int32,
    )

    # Tier pair index: per pool
    tier_idx_per_pool = np.array(
        [_tier_pair_idx(int(row["tier_A"]), int(row["tier_B"]))
         for _, row in pool_meta.iterrows()],
        dtype=np.int32,
    )

    # Count unique tier pairs and chains
    n_chains = len(chains)
    tier_pairs = set()
    for _, row in pool_meta.iterrows():
        tier_pairs.add((int(row["tier_A"]), int(row["tier_B"])))
    n_tiers = 6  # fixed: upper triangle of 3x3

    print(f"  Structural encoding: N_obs={len(panel)}, "
          f"N_pools={base['N_pools']}, n_chains={n_chains}, n_tiers={n_tiers}")

    return {
        # Base arrays (same as encode_covariates)
        "pool_idx": base["pool_idx"],
        "X_pool": base["X_pool"],
        "x_obs": x_obs,
        "y_obs": base["y_obs"],
        "pool_ids": base["pool_ids"],
        "pool_meta": pool_meta,
        "covariate_names": base["covariate_names"],
        "tier_A_per_pool": base["tier_A_per_pool"],
        "N_pools": base["N_pools"],
        "K_cov": base["K_cov"],
        "ref_chain": base["ref_chain"],
        "ref_tier_a": base["ref_tier_a"],
        "ref_tier_b": base["ref_tier_b"],
        "chains": chains,
        # Structural model extras
        "sigma_daily": sigma_daily,
        "fee": fee_per_obs,
        "gas": gas_arr,
        "chain_idx": chain_idx_per_pool,
        "tier_idx": tier_idx_per_pool,
        "lag_log_tvl": lag_log_tvl,
        "n_chains": n_chains,
        "n_tiers": n_tiers,
    }
