"""Diagnostic plots for noise calibration."""

import os

import numpy as np
import pandas as pd

from .constants import K_COEFF, COEFF_NAMES
from .inference import _get_theta_samples


def plot_diagnostics(samples, data, output_dir, elbo_losses=None,
                     mcmc=None, prior_samples=None):
    """Generate up to 9 diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    theta_samples = _get_theta_samples(
        sample_dict, np.array(data["X_pool"]), data=data
    )  # (S, N_pools, K_coeff)
    theta_median = np.median(theta_samples, axis=0)

    pool_idx = data["pool_idx"]
    x_obs = data["x_obs"]
    y_obs = data["y_obs"]
    pool_meta = data["pool_meta"]
    pool_ids = data["pool_ids"]

    # --- 1. Prior predictive check ---
    if prior_samples is not None:
        y_prior = prior_samples.get("y", None)
        if y_prior is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            # Flatten a subsample of prior draws
            y_prior_flat = y_prior.flatten()
            # Clip for display
            clip_lo, clip_hi = np.percentile(y_prior_flat, [0.5, 99.5])
            y_prior_clipped = y_prior_flat[
                (y_prior_flat >= clip_lo) & (y_prior_flat <= clip_hi)
            ]
            ax.hist(y_prior_clipped, bins=100, alpha=0.5, density=True,
                    color="steelblue", label="Prior predictive")
            ax.hist(y_obs, bins=100, alpha=0.5, density=True,
                    color="coral", label="Observed")
            ax.set_xlabel("log(volume)")
            ax.set_ylabel("Density")
            ax.set_title("Prior predictive check: log-volume")
            ax.legend()
            plt.tight_layout()
            path = os.path.join(output_dir, "prior_predictive.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}")

    # --- 2. ELBO loss curve (SVI only) ---
    if elbo_losses is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(elbo_losses, alpha=0.3, color="steelblue", linewidth=0.5)
        # Smoothed
        window = min(100, len(elbo_losses) // 10)
        if window > 1:
            smoothed = pd.Series(elbo_losses).rolling(window).mean().values
            ax.plot(smoothed, color="red", linewidth=1.5, label=f"Rolling {window}")
            ax.legend()
        ax.set_xlabel("Step")
        ax.set_ylabel("ELBO loss")
        ax.set_title("ELBO convergence")

        ax = axes[1]
        # Last 20% of training
        start = len(elbo_losses) * 4 // 5
        ax.plot(range(start, len(elbo_losses)), elbo_losses[start:],
                color="steelblue", linewidth=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("ELBO loss")
        ax.set_title("ELBO convergence (last 20%)")

        plt.tight_layout()
        path = os.path.join(output_dir, "elbo_convergence.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # --- 3. Trace plots (NUTS only) ---
    if mcmc is not None:
        try:
            import arviz as az
            idata = az.from_numpyro(mcmc)
            var_names = ["sigma_theta", "sigma_eps", "df"]
            available = [v for v in var_names if v in idata.posterior]
            if available:
                axes = az.plot_trace(idata, var_names=available, compact=True)
                fig = axes.ravel()[0].figure
                fig.set_size_inches(14, 3 * len(available))
                path = os.path.join(output_dir, "trace_plots.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {path}")
        except Exception as e:
            print(f"  WARNING: Trace plots failed: {e}")

    # --- 4. Posterior predictive: predicted vs observed ---
    # Compute y_pred and r2 here; r2 is reused in plot 9 (model summary).
    theta_obs = theta_median[pool_idx]
    y_pred = np.sum(theta_obs * x_obs, axis=1)
    r2 = 1 - np.var(y_obs - y_pred) / np.var(y_obs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(y_obs, y_pred, alpha=0.1, s=4, color="steelblue")
    lims = [min(y_obs.min(), y_pred.min()), max(y_obs.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Observed log(volume)")
    ax.set_ylabel("Predicted log(volume)")
    ax.set_title("Posterior predictive check")
    ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
            fontsize=11, verticalalignment="top")

    ax = axes[1]
    residuals = y_obs - y_pred
    ax.hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual")

    sigma_eps_samples = np.array(sample_dict.get("sigma_eps", [0]))
    if sigma_eps_samples.ndim > 1:
        sigma_str = ", ".join(f"{np.median(sigma_eps_samples[:, i]):.2f}"
                              for i in range(sigma_eps_samples.shape[1]))
    else:
        sigma_str = f"{np.median(sigma_eps_samples):.2f}"
    ax.set_title(f"Residuals (sigma_eps ~ [{sigma_str}])")

    plt.tight_layout()
    path = os.path.join(output_dir, "posterior_predictive.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- 5. Per-pool b_c by chain/tier ---
    b_tvl_all = theta_median[:, 1]

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
    if chain_data:
        ax.boxplot(chain_data, tick_labels=chain_labels, vert=True)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Per-pool b_c (TVL elasticity)")
    ax.set_title("TVL elasticity by chain")

    ax = axes[1]
    tier_a_vals = pool_meta["tier_A"].values.astype(int)
    tier_labels_map = {0: "Blue-chip", 1: "Mid-cap", 2: "Long-tail"}
    tier_data = []
    tier_labels = []
    for t in [0, 1, 2]:
        mask = tier_a_vals == t
        if mask.sum() > 0:
            tier_data.append(b_tvl_all[mask])
            tier_labels.append(f"{tier_labels_map[t]}\n(n={mask.sum()})")
    if tier_data:
        ax.boxplot(tier_data, tick_labels=tier_labels, vert=True)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Per-pool b_c (TVL elasticity)")
    ax.set_title("TVL elasticity by token tier (best token)")

    plt.tight_layout()
    path = os.path.join(output_dir, "per_pool_b_c.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- 6. Correlation matrix posterior ---
    L_Omega_samples = np.array(sample_dict["L_Omega"])  # (S, K, K)
    Omega_samples = np.einsum("sij,skj->sik", L_Omega_samples, L_Omega_samples)
    Omega_median = np.median(Omega_samples, axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(Omega_median, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(K_COEFF))
    ax.set_yticks(range(K_COEFF))
    ax.set_xticklabels(COEFF_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(COEFF_NAMES)
    for i in range(K_COEFF):
        for j in range(K_COEFF):
            ax.text(j, i, f"{Omega_median[i, j]:.2f}", ha="center",
                    va="center", fontsize=10,
                    color="white" if abs(Omega_median[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Posterior median correlation matrix (Omega)")
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- 7. Shrinkage plot: OLS b_c vs hierarchical b_c ---
    ols_b_c = np.zeros(len(pool_ids))
    for i, pid in enumerate(pool_ids):
        mask = pool_idx == i
        if mask.sum() < 5:
            ols_b_c[i] = np.nan
            continue
        x_i = x_obs[mask]
        y_i = y_obs[mask]
        try:
            beta, _, _, _ = np.linalg.lstsq(x_i, y_i, rcond=None)
            ols_b_c[i] = beta[1]  # TVL coefficient
        except np.linalg.LinAlgError:
            ols_b_c[i] = np.nan

    hier_b_c = theta_median[:, 1]
    valid = np.isfinite(ols_b_c)

    if valid.sum() > 2:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(ols_b_c[valid], hier_b_c[valid], alpha=0.6, s=20,
                   color="steelblue")

        pop_b_c = np.median(hier_b_c)
        ax.axhline(pop_b_c, color="red", linestyle="--", linewidth=0.8,
                   label=f"Population median = {pop_b_c:.3f}")

        lims = [min(np.nanmin(ols_b_c[valid]), hier_b_c[valid].min()) - 0.2,
                max(np.nanmax(ols_b_c[valid]), hier_b_c[valid].max()) + 0.2]
        ax.plot(lims, lims, "k:", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Per-pool OLS b_c (lagged TVL)")
        ax.set_ylabel("Hierarchical posterior median b_c")
        ax.set_title("Shrinkage: OLS vs hierarchical TVL elasticity")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, "shrinkage_b_c.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # --- 8. beta_tvl vs beta_vol scatter colored by chain ---
    fig, ax = plt.subplots(figsize=(10, 7))
    pool_id_to_chain = dict(zip(pool_meta["pool_id"], pool_meta["chain"]))
    chain_colors = {}
    cmap = plt.cm.tab10
    unique_chains = sorted(pool_meta["chain"].unique())
    for i, c in enumerate(unique_chains):
        chain_colors[c] = cmap(i % 10)

    beta_tvl_arr = theta_median[:, 1]
    beta_vol_arr = theta_median[:, 2]
    for i, pid in enumerate(pool_ids):
        c = pool_id_to_chain.get(pid, "?")
        ax.scatter(beta_tvl_arr[i], beta_vol_arr[i],
                   color=chain_colors.get(c, "gray"), alpha=0.6, s=20,
                   edgecolors="white", linewidths=0.3)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=chain_colors[c], markersize=8,
                       label=c)
               for c in unique_chains if c in chain_colors]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_xlabel("b_tvl (TVL elasticity)")
    ax.set_ylabel("b_sigma (volatility sensitivity)")
    ax.set_title("Pool-specific coefficients by chain")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    path = os.path.join(output_dir, "beta_tvl_vs_beta_vol.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # --- 9. Model summary panel ---
    B_samples = np.array(sample_dict["B"])
    B_median = np.median(B_samples, axis=0)  # (K_coeff, K_cov)
    sigma_theta_med = np.median(np.array(sample_dict["sigma_theta"]), axis=0)
    df_med = np.median(np.array(sample_dict["df"]))
    sigma_eps_med = np.median(np.array(sample_dict["sigma_eps"]), axis=0)

    col_names = data["covariate_names"]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    summary = "Group-level regression B (posterior median):\n"
    header = f"  {'covariate':<20s}"
    for cn in COEFF_NAMES:
        header += f" {cn:>10s}"
    summary += header + "\n"
    summary += "  " + "-" * (20 + 11 * K_COEFF) + "\n"
    for j, name in enumerate(col_names):
        line = f"  {name:<20s}"
        for k in range(K_COEFF):
            line += f" {B_median[k, j]:>10.3f}"
        summary += line + "\n"

    summary += f"\nsigma_theta: [{', '.join(f'{v:.3f}' for v in sigma_theta_med)}]\n"
    summary += f"\nCorrelation matrix (Omega):\n"
    for i in range(K_COEFF):
        row = "  [" + " ".join(f"{Omega_median[i, j]:>6.3f}"
                               for j in range(K_COEFF)) + "]\n"
        summary += row

    tier_names = ["blue-chip", "mid-cap", "long-tail"]
    sigma_eps_str = ", ".join(f"{tier_names[i]}={sigma_eps_med[i]:.3f}"
                              for i in range(len(sigma_eps_med)))
    summary += f"\nsigma_eps: [{sigma_eps_str}]\n"
    summary += f"df (Student-t): {df_med:.1f}\n"
    summary += f"R^2: {r2:.3f}\n"

    ax.text(0.02, 0.98, summary, transform=ax.transAxes,
            fontsize=7, verticalalignment="top", fontfamily="monospace")
    ax.set_title("Model Summary")
    plt.tight_layout()
    path = os.path.join(output_dir, "model_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
