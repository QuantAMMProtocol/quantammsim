"""Bayesian hierarchical noise volume model across Balancer pools.

Full Bayesian version of the noise calibration: ALL K=4 per-pool
coefficients (intercept, TVL elasticity, volatility response, weekend
effect) vary per pool with pool-level covariates modulating their priors,
and an LKJ-decomposed covariance capturing correlations between
coefficients.

Generative model:
    For pool i with pool-level covariates z_i, day t:

        mu_i = B . z_i                         # K-vector population mean
        eta_i ~ N(0, I_K)                      # non-centered offsets
        theta_i = mu_i + diag(sigma) . L . eta_i   # per-pool coefficients

        log(V_{i,t}) ~ N(theta_i . x_{i,t}, sigma_eps^2)

    theta_i = [intercept_i, b_tvl_i, b_sigma_i, b_weekend_i]
    x_{i,t}  = [1, log_tvl, volatility, weekend]
    z_i      = [1, chain_dummies(6), tier_A_dummies(2), tier_B_dummies(2), log_fee]

Priors:
    B_{k,d}  ~ N(0, 5^2)
    sigma_k  ~ HalfNormal(2.0)
    L        ~ LKJCholesky(K=4, eta=2)
    sigma_eps ~ HalfNormal(3.0)

Usage:
    # Full pipeline: fit + output + diagnostics
    python scripts/calibrate_noise_bayesian.py \\
        --fit --output results/bayesian_noise_params.json --plot

    # Predict for an unseen pool
    python scripts/calibrate_noise_bayesian.py \\
        --predict --chain BASE --tokens ETH USDC --fee 0.003

    # Custom NUTS settings
    python scripts/calibrate_noise_bayesian.py \\
        --fit --num-warmup 2000 --num-samples 4000 --num-chains 4
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# arviz 0.17.x imports scipy.signal.gaussian which was removed in scipy 1.13+.
# Patch it back from scipy.signal.windows before any arviz import.
try:
    from scipy.signal import gaussian as _  # noqa: F401
except ImportError:
    from scipy.signal.windows import gaussian as _gauss
    import scipy.signal
    scipy.signal.gaussian = _gauss

# ---------------------------------------------------------------------------
# Reuse constants and helpers from the frequentist script
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "local_data", "noise_calibration"
)

# Reference levels for dummy coding (dropped categories)
REF_CHAIN = "ARBITRUM"
REF_TIER = 0

# Ordered non-reference chains (alphabetical excluding REF_CHAIN)
CHAIN_ORDER = ["BASE", "GNOSIS", "MAINNET", "OPTIMISM", "POLYGON", "SONIC"]

K = 4   # number of per-pool coefficients
D = 12  # pool-level covariate dimension: 1 + 6 chains + 2 tier_A + 2 tier_B + 1 log_fee

COEFF_NAMES = ["intercept", "b_tvl", "b_sigma", "b_weekend"]


# ---------------------------------------------------------------------------
# Token tier helpers (duplicated to avoid import fragility)
# ---------------------------------------------------------------------------

_TIER_0 = {
    "ETH", "WETH", "BTC", "WBTC", "cbBTC", "USDC", "USDT", "DAI",
    "wstETH", "stETH", "rETH", "cbETH", "WMATIC", "MATIC", "POL",
    "WAVAX", "AVAX", "GNO", "WXDAI", "xDAI",
    "S", "wS",
}

_TIER_1 = {
    "AAVE", "LINK", "UNI", "BAL", "MKR", "CRV", "COMP", "SNX",
    "LDO", "RPL", "SUSHI", "YFI", "1INCH", "ENS", "DYDX",
    "FXS", "FRAX", "LUSD", "sDAI", "GHO", "crvUSD",
    "ARB", "OP", "PENDLE", "ENA", "EIGEN",
    "SAFE", "COW",
}


def classify_token_tier(symbol: str) -> int:
    s = symbol.strip()
    if s in _TIER_0:
        return 0
    if s in _TIER_1:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_panel(cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Load the cached panel parquet produced by calibrate_noise_hierarchical.py --fetch."""
    panel_path = os.path.join(cache_dir, "panel.parquet")
    if not os.path.exists(panel_path):
        print(f"ERROR: Panel cache not found at {panel_path}", file=sys.stderr)
        print("Run: python scripts/calibrate_noise_hierarchical.py --fetch", file=sys.stderr)
        sys.exit(1)

    panel = pd.read_parquet(panel_path)

    # Filter pools with < 10 observations
    pool_counts = panel.groupby("pool_id").size()
    valid_pools = pool_counts[pool_counts >= 10].index
    panel = panel[panel["pool_id"].isin(valid_pools)].copy()

    print(f"  Loaded panel: {len(panel)} obs, "
          f"{panel['pool_id'].nunique()} pools, "
          f"{panel['chain'].nunique()} chains")
    return panel


def _build_z_pool(pool_meta: pd.DataFrame) -> np.ndarray:
    """Build (N_pools, D) pool-level covariate matrix.

    Columns: [1, chain_BASE, ..., chain_SONIC (6),
              tier_A_1, tier_A_2, tier_B_1, tier_B_2, log_fee]
    """
    N = len(pool_meta)
    z = np.zeros((N, D), dtype=np.float64)

    # Intercept
    z[:, 0] = 1.0

    # Chain dummies (columns 1-6)
    for j, chain in enumerate(CHAIN_ORDER):
        z[:, 1 + j] = (pool_meta["chain"].values == chain).astype(float)

    # tier_A dummies (columns 7-8): tiers 1 and 2, reference = 0
    tier_a = pool_meta["tier_A"].values.astype(int)
    z[:, 7] = (tier_a == 1).astype(float)
    z[:, 8] = (tier_a == 2).astype(float)

    # tier_B dummies (columns 9-10): tiers 1 and 2, reference = 0
    tier_b = pool_meta["tier_B"].values.astype(int)
    z[:, 9] = (tier_b == 1).astype(float)
    z[:, 10] = (tier_b == 2).astype(float)

    # log_fee (column 11)
    z[:, 11] = np.log(np.maximum(pool_meta["swap_fee"].values.astype(float), 1e-6))

    return z


def prepare_data(panel: pd.DataFrame) -> dict:
    """Construct JAX-ready arrays from the panel DataFrame.

    Returns dict with:
        pool_idx  : (N_obs,) int32 — pool index per observation
        z_pool    : (N_pools, D) float64 — pool-level covariates
        x_obs     : (N_obs, K) float64 — within-day regressors
        y_obs     : (N_obs,) float64 — log_volume
        pool_ids  : list — ordered pool IDs
        pool_meta : DataFrame — per-pool metadata (indexed same as z_pool rows)
    """
    # Stable pool ordering
    pool_ids = sorted(panel["pool_id"].unique())
    pool_id_to_idx = {pid: i for i, pid in enumerate(pool_ids)}
    N_pools = len(pool_ids)

    # Pool-level metadata (one row per pool)
    pool_meta = panel.drop_duplicates("pool_id").set_index("pool_id").loc[pool_ids].reset_index()
    z_pool = _build_z_pool(pool_meta)

    # Observation-level arrays
    pool_idx = panel["pool_id"].map(pool_id_to_idx).values.astype(np.int32)
    x_obs = np.column_stack([
        np.ones(len(panel)),
        panel["log_tvl"].values,
        panel["volatility"].values,
        panel["weekend"].values,
    ]).astype(np.float64)
    y_obs = panel["log_volume"].values.astype(np.float64)

    print(f"  Prepared: N_obs={len(y_obs)}, N_pools={N_pools}, K={K}, D={D}")
    print(f"  z_pool range check — log_fee: [{z_pool[:, 11].min():.2f}, {z_pool[:, 11].max():.2f}]")

    return {
        "pool_idx": pool_idx,
        "z_pool": z_pool,
        "x_obs": x_obs,
        "y_obs": y_obs,
        "pool_ids": pool_ids,
        "pool_meta": pool_meta,
        "N_pools": N_pools,
    }


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

def hierarchical_noise_model(pool_idx, z_pool, x_obs, y_obs=None,
                             N_pools=None, K=4, D=12):
    """Bayesian hierarchical noise volume model.

    Non-centered parameterization with LKJ correlation prior.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    N_obs = pool_idx.shape[0]

    # --- Population coefficient matrix B: (K, D) ---
    B = numpyro.sample("B", dist.Normal(0.0, 5.0).expand([K, D]).to_event(2))

    # --- Per-pool scale and correlation ---
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0).expand([K]).to_event(1))
    L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(K, concentration=2.0))

    # Cholesky factor of covariance: diag(sigma) @ L_Omega
    L_Sigma = jnp.diag(sigma) @ L_Omega  # (K, K)

    # --- Non-centered pool effects ---
    with numpyro.plate("pools", N_pools):
        eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([K]).to_event(1))

    # theta_i = B @ z_i + L_Sigma @ eta_i   for each pool i
    # mu: (N_pools, K) = z_pool @ B^T
    mu = z_pool @ B.T  # (N_pools, K)
    theta = mu + eta @ L_Sigma.T  # (N_pools, K)

    # --- Observation model ---
    sigma_eps = numpyro.sample("sigma_eps", dist.HalfNormal(3.0))

    # Predicted log-volume: theta[pool_idx] . x_obs (dot product per obs)
    theta_obs = theta[pool_idx]  # (N_obs, K)
    mu_obs = jnp.sum(theta_obs * x_obs, axis=1)  # (N_obs,)

    with numpyro.plate("obs", N_obs):
        numpyro.sample("y", dist.Normal(mu_obs, sigma_eps), obs=y_obs)

    # Deterministic: store theta for extraction
    numpyro.deterministic("theta", theta)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(data, num_warmup=1000, num_samples=2000, num_chains=4,
                  target_accept=0.85, max_tree_depth=10, seed=42):
    """Run NUTS on the hierarchical model.

    Returns the MCMC object with samples.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    # Use all available CPU cores for chains
    numpyro.set_host_device_count(min(num_chains, len(jax.devices("cpu"))))

    kernel = NUTS(
        hierarchical_noise_model,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(seed)

    print(f"\n  Running NUTS: {num_chains} chains x "
          f"({num_warmup} warmup + {num_samples} samples)")
    print(f"  target_accept={target_accept}, max_tree_depth={max_tree_depth}")

    mcmc.run(
        rng_key,
        pool_idx=jnp.array(data["pool_idx"]),
        z_pool=jnp.array(data["z_pool"]),
        x_obs=jnp.array(data["x_obs"]),
        y_obs=jnp.array(data["y_obs"]),
        N_pools=data["N_pools"],
        K=K,
        D=D,
    )

    mcmc.print_summary(exclude_deterministic=True)
    return mcmc


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def extract_noise_params(mcmc, data) -> list:
    """Extract per-pool noise params from MCMC posterior.

    Reconstructs theta from the non-centered parameterization,
    takes posterior medians, and applies weekend absorption:
        b_0_effective = b_0_raw + b_weekend * (2/7)

    Returns list of dicts compatible with reclamm_loglinear_noise_volume.
    """
    samples = mcmc.get_samples()
    theta_samples = samples["theta"]  # (n_samples, N_pools, K)

    # Posterior median per pool
    theta_median = np.median(theta_samples, axis=0)  # (N_pools, K)
    theta_std = np.std(theta_samples, axis=0)         # (N_pools, K)

    pool_ids = data["pool_ids"]
    pool_meta = data["pool_meta"]

    results = []
    for i, pool_id in enumerate(pool_ids):
        meta = pool_meta.iloc[i]
        b_0_raw, b_tvl, b_sigma, b_weekend = theta_median[i]
        std_vals = theta_std[i]

        # Weekend absorption: simulator has no weekend indicator,
        # so fold the expected weekend effect into the intercept.
        # Weekend days = 2/7 of all days.
        b_0_effective = b_0_raw + b_weekend * (2.0 / 7.0)

        tokens = meta["tokens"]
        if isinstance(tokens, str):
            tokens = tokens.split(",")

        results.append({
            "pool_id": pool_id,
            "chain": str(meta["chain"]),
            "tokens": tokens,
            "theta_median": [float(x) for x in theta_median[i]],
            "theta_std": [float(x) for x in std_vals],
            "noise_params": {
                "b_0": float(b_0_effective),
                "b_sigma": float(b_sigma),
                "b_c": float(b_tvl),
                "b_weekend": float(b_weekend),
                "base_fee": float(meta["swap_fee"]),
            },
        })

    return results


def predict_new_pool(mcmc, data, chain: str, tokens: list, fee: float) -> dict:
    """Predict noise params for an unseen pool using population effects.

    Constructs z_new, computes mu_new = B @ z_new across posterior samples,
    and returns median + 90% credible intervals.
    """
    # Build z_new
    z_new = np.zeros(D, dtype=np.float64)
    z_new[0] = 1.0  # intercept

    # Chain dummies
    if chain in CHAIN_ORDER:
        j = CHAIN_ORDER.index(chain)
        z_new[1 + j] = 1.0

    # Tier dummies
    tiers = sorted([classify_token_tier(t) for t in tokens])
    tier_a = tiers[0]
    tier_b = tiers[1] if len(tiers) > 1 else tiers[0]
    if tier_a == 1:
        z_new[7] = 1.0
    elif tier_a == 2:
        z_new[8] = 1.0
    if tier_b == 1:
        z_new[9] = 1.0
    elif tier_b == 2:
        z_new[10] = 1.0

    # log_fee
    z_new[11] = np.log(max(fee, 1e-6))

    # Compute mu_new = B @ z_new across all posterior samples
    B_samples = np.array(mcmc.get_samples()["B"])  # (n_samples, K, D)
    mu_samples = np.einsum("skd,d->sk", B_samples, z_new)  # (n_samples, K)

    mu_median = np.median(mu_samples, axis=0)
    mu_q05 = np.percentile(mu_samples, 5, axis=0)
    mu_q95 = np.percentile(mu_samples, 95, axis=0)

    # Weekend absorption
    b_0_raw, b_tvl, b_sigma, b_weekend = mu_median
    b_0_effective = b_0_raw + b_weekend * (2.0 / 7.0)

    result = {
        "chain": chain,
        "tokens": tokens,
        "fee": fee,
        "prediction_source": "population_level",
        "noise_params": {
            "b_0": float(b_0_effective),
            "b_sigma": float(b_sigma),
            "b_c": float(b_tvl),
            "b_weekend": float(b_weekend),
            "base_fee": float(fee),
        },
        "credible_intervals_90": {
            name: {
                "median": float(mu_median[k]),
                "q05": float(mu_q05[k]),
                "q95": float(mu_q95[k]),
            }
            for k, name in enumerate(COEFF_NAMES)
        },
    }

    print(f"\n  Predicted noise_params for {chain} {tokens} (fee={fee}):")
    for name, ci in result["credible_intervals_90"].items():
        print(f"    {name:12s}: {ci['median']:+.3f}  "
              f"[{ci['q05']:+.3f}, {ci['q95']:+.3f}]")
    print(f"\n  Effective b_0 (weekend-absorbed): {b_0_effective:.3f}")

    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_convergence(mcmc) -> dict:
    """Compute convergence diagnostics: R-hat, ESS, divergences."""
    import arviz as az

    idata = az.from_numpyro(mcmc)

    # R-hat and ESS for non-deterministic parameters
    n_chains = idata.posterior.sizes.get("chain", 1)

    rhat_max = float("nan")
    if n_chains >= 2:
        rhat = az.rhat(idata)
        rhat_vals = []
        for var in rhat.data_vars:
            if var == "theta":
                continue  # deterministic
            vals = rhat[var].values
            rhat_vals.extend(vals.flatten())
        rhat_max = float(np.nanmax(rhat_vals)) if rhat_vals else float("nan")

    ess = az.ess(idata)
    ess_vals = []
    for var in ess.data_vars:
        if var == "theta":
            continue
        vals = ess[var].values
        ess_vals.extend(vals.flatten())
    ess_min = float(np.nanmin(ess_vals)) if ess_vals else float("nan")

    # Divergences
    divergences = int(idata.sample_stats["diverging"].sum().values)

    print(f"\n  Convergence diagnostics:")
    if n_chains >= 2:
        print(f"    R-hat max:   {rhat_max:.4f}  {'OK' if rhat_max < 1.05 else 'WARNING'}")
    else:
        print(f"    R-hat max:   N/A (need >= 2 chains)")
    print(f"    ESS min:     {ess_min:.0f}    {'OK' if ess_min > 400 else 'WARNING'}")
    print(f"    Divergences: {divergences}     {'OK' if divergences == 0 else 'WARNING'}")

    return {
        "r_hat_max": rhat_max,
        "ess_min": ess_min,
        "divergences": divergences,
    }


def plot_bayesian_diagnostics(mcmc, data, output_dir="results"):
    """Generate ArviZ diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import arviz as az

    os.makedirs(output_dir, exist_ok=True)
    idata = az.from_numpyro(mcmc)
    samples = mcmc.get_samples()

    # --- 1. Trace plots for sigma, sigma_eps ---
    axes = az.plot_trace(idata, var_names=["sigma", "sigma_eps"], compact=True)
    fig1 = axes.ravel()[0].figure
    fig1.set_size_inches(14, 8)
    path1 = os.path.join(output_dir, "bayesian_trace_sigma.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # --- 2. Posterior predictive: predicted vs observed ---
    theta_samples = samples["theta"]  # (S, N_pools, K)
    sigma_eps_samples = np.array(samples["sigma_eps"])  # (S,)
    theta_median = np.median(theta_samples, axis=0)  # (N_pools, K)

    pool_idx = data["pool_idx"]
    x_obs = data["x_obs"]
    y_obs = data["y_obs"]

    theta_obs = theta_median[pool_idx]
    y_pred = np.sum(theta_obs * x_obs, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(y_obs, y_pred, alpha=0.1, s=4, color="steelblue")
    lims = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Observed log(volume)")
    ax.set_ylabel("Predicted log(volume)")
    ax.set_title("Posterior predictive check")
    r2 = 1 - np.var(y_obs - y_pred) / np.var(y_obs)
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
            fontsize=11, verticalalignment="top")

    ax = axes[1]
    residuals = y_obs - y_pred
    ax.hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_title(f"Residual distribution (σ_ε ≈ {np.median(sigma_eps_samples):.2f})")

    plt.tight_layout()
    path2 = os.path.join(output_dir, "bayesian_posterior_predictive.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    # --- 3. Per-pool b_c (TVL elasticity) by chain/tier ---
    pool_meta = data["pool_meta"]
    b_tvl_all = theta_median[:, 1]  # index 1 = b_tvl

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    chains_present = sorted(pool_meta["chain"].unique())
    chain_data = []
    chain_labels = []
    for c in chains_present:
        mask = pool_meta["chain"].values == c
        if mask.sum() > 0:
            chain_data.append(b_tvl_all[mask])
            chain_labels.append(f"{c}\n(n={mask.sum()})")
    ax.boxplot(chain_data, tick_labels=chain_labels, vert=True)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Per-pool b_c (TVL elasticity)")
    ax.set_title("TVL elasticity by chain")

    ax = axes[1]
    # By tier_A
    tier_a_vals = pool_meta["tier_A"].values.astype(int)
    tier_labels_map = {0: "Blue-chip", 1: "Mid-cap", 2: "Long-tail"}
    tier_data = []
    tier_labels = []
    for t in [0, 1, 2]:
        mask = tier_a_vals == t
        if mask.sum() > 0:
            tier_data.append(b_tvl_all[mask])
            tier_labels.append(f"{tier_labels_map[t]}\n(n={mask.sum()})")
    ax.boxplot(tier_data, tick_labels=tier_labels, vert=True)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Per-pool b_c (TVL elasticity)")
    ax.set_title("TVL elasticity by token tier (best token)")

    plt.tight_layout()
    path3 = os.path.join(output_dir, "bayesian_per_pool_b_c.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path3}")

    # --- 4. Correlation matrix posterior ---
    L_Omega_samples = np.array(samples["L_Omega"])  # (S, K, K)
    # Correlation = L @ L^T
    Omega_samples = np.einsum("sij,skj->sik", L_Omega_samples, L_Omega_samples)
    Omega_median = np.median(Omega_samples, axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Omega_median, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(COEFF_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(COEFF_NAMES)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{Omega_median[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="white" if abs(Omega_median[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Posterior median correlation matrix (Ω)")
    plt.tight_layout()
    path4 = os.path.join(output_dir, "bayesian_correlation_matrix.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path4}")

    # --- 5. Shrinkage plot: OLS b_c vs hierarchical b_c ---
    # Compute per-pool OLS b_c for comparison
    panel_meta = data["pool_meta"]
    pool_idx_arr = data["pool_idx"]
    pool_ids = data["pool_ids"]

    ols_b_c = np.zeros(len(pool_ids))
    for i, pid in enumerate(pool_ids):
        mask = pool_idx_arr == i
        if mask.sum() < 5:
            ols_b_c[i] = np.nan
            continue
        x_i = data["x_obs"][mask]
        y_i = data["y_obs"][mask]
        # Simple OLS: y = X @ beta
        try:
            beta, _, _, _ = np.linalg.lstsq(x_i, y_i, rcond=None)
            ols_b_c[i] = beta[1]  # TVL coefficient
        except np.linalg.LinAlgError:
            ols_b_c[i] = np.nan

    hier_b_c = theta_median[:, 1]
    valid = np.isfinite(ols_b_c)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ols_b_c[valid], hier_b_c[valid], alpha=0.6, s=20, color="steelblue")

    # Population mean line
    pop_b_c = np.median(hier_b_c)
    ax.axhline(pop_b_c, color="red", linestyle="--", linewidth=0.8,
               label=f"Population median = {pop_b_c:.3f}")

    # 45-degree line
    lims = [min(np.nanmin(ols_b_c[valid]), hier_b_c[valid].min()) - 0.2,
            max(np.nanmax(ols_b_c[valid]), hier_b_c[valid].max()) + 0.2]
    ax.plot(lims, lims, "k:", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Per-pool OLS b_c")
    ax.set_ylabel("Hierarchical posterior median b_c")
    ax.set_title("Shrinkage: OLS vs hierarchical TVL elasticity")
    ax.legend()
    plt.tight_layout()
    path5 = os.path.join(output_dir, "bayesian_shrinkage_b_c.png")
    plt.savefig(path5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path5}")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def generate_output_json(pool_params, mcmc, data, convergence, output_path,
                         num_warmup, num_samples, num_chains, target_accept):
    """Write structured JSON output with population effects and per-pool params."""
    samples = mcmc.get_samples()

    B_median = np.median(np.array(samples["B"]), axis=0).tolist()
    sigma_median = np.median(np.array(samples["sigma"]), axis=0).tolist()
    sigma_eps_median = float(np.median(np.array(samples["sigma_eps"])))

    output = {
        "model": "bayesian_hierarchical_loglinear",
        "inference": {
            "method": "NUTS",
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "target_accept_prob": target_accept,
        },
        "population_effects": {
            "B": B_median,
            "sigma": sigma_median,
            "sigma_eps": sigma_eps_median,
            "coeff_names": COEFF_NAMES,
            "covariate_names": (
                ["intercept"] + [f"chain_{c}" for c in CHAIN_ORDER]
                + ["tier_A_1", "tier_A_2", "tier_B_1", "tier_B_2", "log_fee"]
            ),
        },
        "convergence": convergence,
        "pools": {
            p["pool_id"]: {
                "chain": p["chain"],
                "tokens": p["tokens"],
                "theta_median": p["theta_median"],
                "theta_std": p["theta_std"],
                "noise_params": p["noise_params"],
            }
            for p in pool_params
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Wrote {len(pool_params)} pool params -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hierarchical noise volume model for Balancer pools"
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch pool data (delegates to calibrate_noise_hierarchical.py --fetch)",
    )
    parser.add_argument("--fit", action="store_true", help="Run NUTS inference")
    parser.add_argument("--plot", action="store_true", help="Generate diagnostic plots")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--output-dir", default="results", help="Plot output directory")
    parser.add_argument("--predict", action="store_true", help="Predict for a new pool")
    parser.add_argument("--chain", default=None, help="Chain for --predict")
    parser.add_argument("--tokens", nargs="+", default=None, help="Tokens for --predict")
    parser.add_argument("--fee", type=float, default=0.003, help="Fee for --predict")
    parser.add_argument("--cache-dir", default=None, help="Cache directory")

    # NUTS hyperparameters
    parser.add_argument("--num-warmup", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.85)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cache_dir = args.cache_dir or CACHE_DIR

    if not any([args.fetch, args.fit, args.predict]):
        parser.error("At least one of --fetch, --fit, --predict is required")

    # --- Fetch (delegate to existing script) ---
    if args.fetch:
        import subprocess
        cmd = [
            sys.executable, "scripts/calibrate_noise_hierarchical.py",
            "--fetch", "--cache-dir", cache_dir,
        ]
        print("Delegating data fetch to calibrate_noise_hierarchical.py...")
        subprocess.run(cmd, check=True)

    # --- Fit ---
    if args.fit:
        print("\nBayesian Hierarchical Noise Volume Model")
        print("=" * 60)

        panel = load_panel(cache_dir)
        data = prepare_data(panel)

        mcmc = run_inference(
            data,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            target_accept=args.target_accept,
            max_tree_depth=args.max_tree_depth,
            seed=args.seed,
        )

        convergence = check_convergence(mcmc)
        pool_params = extract_noise_params(mcmc, data)

        # Print summary statistics
        b_c_vals = [p["noise_params"]["b_c"] for p in pool_params]
        b_0_vals = [p["noise_params"]["b_0"] for p in pool_params]
        print(f"\n  Per-pool b_c: mean={np.mean(b_c_vals):.3f}, "
              f"std={np.std(b_c_vals):.3f}, "
              f"range=[{np.min(b_c_vals):.3f}, {np.max(b_c_vals):.3f}]")
        print(f"  Per-pool b_0: mean={np.mean(b_0_vals):.3f}, "
              f"std={np.std(b_0_vals):.3f}")

        if args.output:
            generate_output_json(
                pool_params, mcmc, data, convergence, args.output,
                args.num_warmup, args.num_samples, args.num_chains,
                args.target_accept,
            )

        if args.plot:
            print("\nGenerating diagnostic plots...")
            plot_bayesian_diagnostics(mcmc, data, output_dir=args.output_dir)

        # Save MCMC samples for --predict reuse
        mcmc_cache = os.path.join(cache_dir, "bayesian_mcmc_samples.npz")
        samples = mcmc.get_samples()
        np.savez_compressed(
            mcmc_cache,
            **{k: np.array(v) for k, v in samples.items()},
        )
        # Also save data arrays for predict
        data_cache = os.path.join(cache_dir, "bayesian_data.npz")
        np.savez_compressed(
            data_cache,
            pool_idx=data["pool_idx"],
            z_pool=data["z_pool"],
            x_obs=data["x_obs"],
            y_obs=data["y_obs"],
        )
        # Save pool_ids list
        with open(os.path.join(cache_dir, "bayesian_pool_ids.json"), "w") as f:
            json.dump(data["pool_ids"], f)
        print(f"  Saved MCMC samples -> {mcmc_cache}")

    # --- Predict ---
    if args.predict:
        if args.chain is None or args.tokens is None:
            parser.error("--predict requires --chain and --tokens")

        # Load cached MCMC samples
        mcmc_cache = os.path.join(cache_dir, "bayesian_mcmc_samples.npz")
        if not os.path.exists(mcmc_cache):
            print(f"ERROR: MCMC cache not found at {mcmc_cache}", file=sys.stderr)
            print("Run with --fit first.", file=sys.stderr)
            sys.exit(1)

        # For predict, we only need B samples — create a minimal mock
        cached = np.load(mcmc_cache)

        class _MockMCMC:
            """Minimal interface to reuse predict_new_pool with cached samples."""
            def __init__(self, samples_dict):
                self._samples = samples_dict
            def get_samples(self):
                return self._samples

        samples_dict = {k: cached[k] for k in cached.files}
        mock_mcmc = _MockMCMC(samples_dict)

        result = predict_new_pool(mock_mcmc, None, args.chain, args.tokens, args.fee)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
