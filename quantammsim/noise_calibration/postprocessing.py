"""Post-processing: extract params, predict, convergence, prior predictive."""

import numpy as np

from .constants import K_COEFF, COEFF_NAMES, K_OBS_COEFF, OBS_COEFF_NAMES
from .token_classification import classify_token_tier
from .covariate_encoding import _tier_pair_idx
from .inference import _get_theta_samples, _build_model_kwargs
from .model import noise_model


def extract_noise_params(samples, data, use_median=True) -> list:
    """Extract per-pool noise params from posterior samples.

    Handles both MCMC.get_samples() and SVI samples dict.
    Applies weekend absorption: b_0_eff = b_0_raw + b_weekend * (2/7).
    """
    # Get theta samples
    if hasattr(samples, "get_samples"):
        # MCMC object
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    theta_samples = _get_theta_samples(
        sample_dict, np.array(data["X_pool"]), data=data
    )  # (S, N_pools, K_coeff)

    agg_fn = np.median if use_median else np.mean
    theta_agg = agg_fn(theta_samples, axis=0)  # (N_pools, K_coeff)
    theta_std = np.std(theta_samples, axis=0)

    pool_ids = data["pool_ids"]
    pool_meta = data["pool_meta"]

    results = []
    for i, pool_id in enumerate(pool_ids):
        meta = pool_meta.iloc[i]
        b_0_raw, b_tvl, b_sigma, b_weekend = theta_agg[i]
        std_vals = theta_std[i]

        # Weekend absorption: simulator has no weekend indicator,
        # so fold the expected weekend effect into the intercept.
        b_0_effective = b_0_raw + b_weekend * (2.0 / 7.0)

        tokens = meta["tokens"]
        if isinstance(tokens, str):
            tokens = tokens.split(",")

        results.append({
            "pool_id": pool_id,
            "chain": str(meta["chain"]),
            "tokens": tokens,
            "theta_median": [float(x) for x in theta_agg[i]],
            "theta_std": [float(x) for x in std_vals],
            "b_weekend": float(b_weekend),
            "noise_params": {
                "b_0": float(b_0_effective),
                "b_sigma": float(b_sigma),
                "b_c": float(b_tvl),
                "b_weekend": float(b_weekend),
                "base_fee": float(meta["swap_fee"]),
            },
        })

    return results


def predict_new_pool(samples, data, chain: str, tokens: list,
                     fee: float, feature_assignments=None) -> dict:
    """Predict noise params for an unseen pool using population effects.

    Constructs z_new, computes mu_new = B @ z_new across all posterior samples,
    returns point estimate + 90% credible intervals with weekend absorption.
    """
    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    # Build z_new using data-driven column names
    col_names = data["covariate_names"]
    z_new = np.zeros(len(col_names), dtype=np.float64)

    tiers = sorted([classify_token_tier(t) for t in tokens])
    tier_a = str(tiers[0])
    tier_b = str(tiers[1]) if len(tiers) > 1 else tier_a

    for i, name in enumerate(col_names):
        if name == "intercept":
            z_new[i] = 1.0
        elif name == "log_fee":
            z_new[i] = np.log(max(fee, 1e-6))
        elif name == f"chain_{chain}":
            z_new[i] = 1.0
        elif name == f"tier_A_{tier_a}":
            z_new[i] = 1.0
        elif name == f"tier_B_{tier_b}":
            z_new[i] = 1.0

    # mu_new = B @ z_new across all posterior samples
    B_samples = np.array(sample_dict["B"])  # (S, K_coeff, K_cov)
    mu_samples = np.einsum("skd,d->sk", B_samples, z_new)  # (S, K_coeff)

    # IBP path: add feature effects
    is_ibp = "W" in sample_dict
    if is_ibp:
        W_samples = np.array(sample_dict["W"])  # (S, K_features, K_coeff)
        if feature_assignments is not None:
            # User-specified binary features
            Z = np.array(feature_assignments)  # (K_features,)
            feature_effect = np.einsum("skj,k->sj", W_samples, Z)
            prediction_source = "ibp_user_features"
        else:
            # Marginal: weight by prevalences pi = cumprod(v_ibp)
            v_ibp = np.array(sample_dict["v_ibp"])  # (S, K_features)
            pi = np.cumprod(v_ibp, axis=1)           # (S, K_features)
            feature_effect = np.einsum("skj,sk->sj", W_samples, pi)
            prediction_source = "ibp_marginal"
        mu_samples = mu_samples + feature_effect
    else:
        prediction_source = "population_level"

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
        "prediction_source": prediction_source,
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


def check_convergence(mcmc_or_losses, method="nuts") -> dict:
    """Compute convergence diagnostics.

    For NUTS: R-hat, ESS, divergences.
    For SVI: final ELBO, ELBO stability.
    """
    if method == "svi":
        losses = np.array(mcmc_or_losses)
        return {
            "method": "svi",
            "final_elbo": float(losses[-1]),
            "elbo_last_100_std": float(np.std(losses[-100:])),
            "elbo_last_100_mean": float(np.mean(losses[-100:])),
        }

    # NUTS diagnostics
    import arviz as az

    mcmc = mcmc_or_losses
    idata = az.from_numpyro(mcmc)

    n_chains = idata.posterior.sizes.get("chain", 1)

    rhat_max = float("nan")
    if n_chains >= 2:
        rhat = az.rhat(idata)
        rhat_vals = []
        for var in rhat.data_vars:
            if var == "theta":
                continue
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

    divergences = int(idata.sample_stats["diverging"].sum().values)

    print(f"\n  Convergence diagnostics:")
    if n_chains >= 2:
        print(f"    R-hat max:   {rhat_max:.4f}  "
              f"{'OK' if rhat_max < 1.05 else 'WARNING'}")
    else:
        print(f"    R-hat max:   N/A (need >= 2 chains)")
    print(f"    ESS min:     {ess_min:.0f}    "
          f"{'OK' if ess_min > 400 else 'WARNING'}")
    print(f"    Divergences: {divergences}     "
          f"{'OK' if divergences == 0 else 'WARNING'}")

    return {
        "method": "nuts",
        "r_hat_max": rhat_max,
        "ess_min": ess_min,
        "divergences": divergences,
    }


def assign_dp_clusters(samples, data) -> np.ndarray:
    """Compute posterior MAP cluster assignments for DP mixture model.

    Uses median posterior samples for v->w, sigma_eps, df, theta to compute
    per-pool-per-cluster log-likelihoods, then returns argmax assignments.
    """
    from scipy.special import logsumexp
    from .model import stick_breaking_weights

    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    # Median posterior parameters
    v_med = np.median(np.array(sample_dict["v"]), axis=0)
    sigma_eps_med = np.median(np.array(sample_dict["sigma_eps"]), axis=0)
    df_med = float(np.median(np.array(sample_dict["df"])))

    # Compute w from v via stick-breaking (using numpy)
    import jax.numpy as jnp
    w = np.array(stick_breaking_weights(jnp.array(v_med)))

    # Reconstruct theta
    theta_samples = _get_theta_samples(
        sample_dict, np.array(data["X_pool"]), data=data
    )
    theta_med = np.median(theta_samples, axis=0)  # (N_pools, K_coeff)

    # Per-observation predicted means
    pool_idx = np.array(data["pool_idx"])
    x_obs = np.array(data["x_obs"])
    y_obs = np.array(data["y_obs"])
    N_pools = data["N_pools"]
    K_clusters = len(sigma_eps_med)

    theta_obs = theta_med[pool_idx]
    mu_obs = np.sum(theta_obs * x_obs, axis=1)  # (N_obs,)

    # Log-likelihood per observation per cluster
    from scipy.stats import t as t_dist
    log_lik_per_k = np.zeros((len(y_obs), K_clusters))
    for k in range(K_clusters):
        log_lik_per_k[:, k] = t_dist.logpdf(
            y_obs, df_med, loc=mu_obs, scale=sigma_eps_med[k]
        )

    # Sum within pools
    pool_log_liks = np.zeros((N_pools, K_clusters))
    for i in range(len(y_obs)):
        pool_log_liks[pool_idx[i]] += log_lik_per_k[i]

    # Posterior cluster probabilities: log p(z=k|data) = log w_k + sum log p(y|k)
    log_posterior = np.log(w + 1e-30)[None, :] + pool_log_liks
    # MAP assignment
    assignments = np.argmax(log_posterior, axis=1).astype(np.int64)
    return assignments


def assign_ibp_features(samples, data) -> np.ndarray:
    """Compute MAP feature assignments for marginalized IBP model.

    Enumerates all 2^K binary feature configurations per pool, evaluates
    per-pool log-posterior (log-prior + log-likelihood), returns argmax
    config as (N_pools, K_features) binary ndarray.

    Uses median posterior parameters for B, W, v_ibp, sigma_eps, df.
    """
    from scipy.stats import t as t_dist

    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    B_med = np.median(np.array(sample_dict["B"]), axis=0)        # (K_coeff, K_cov)
    W_med = np.median(np.array(sample_dict["W"]), axis=0)        # (K_features, K_coeff)
    v_ibp_med = np.median(np.array(sample_dict["v_ibp"]), axis=0)  # (K_features,)
    sigma_eps_med = float(np.median(np.array(sample_dict["sigma_eps"])))
    df_med = float(np.median(np.array(sample_dict["df"])))

    pi = np.cumprod(v_ibp_med)  # (K_features,)
    K_features = len(pi)

    pool_idx = np.array(data["pool_idx"])
    X_pool = np.array(data["X_pool"])
    x_obs = np.array(data["x_obs"])
    y_obs = np.array(data["y_obs"])
    N_pools = data["N_pools"]

    # Enumerate all 2^K configs
    n_configs = 2 ** K_features
    configs = (
        (np.arange(n_configs)[:, None] >> np.arange(K_features)[None, :]) & 1
    ).astype(float)  # (n_configs, K_features)

    # Log-prior per config
    log_pi = np.log(pi + 1e-30)
    log_1mpi = np.log(1.0 - pi + 1e-30)
    log_prior = configs @ log_pi + (1.0 - configs) @ log_1mpi  # (n_configs,)

    # Population mean
    mu_pop = X_pool @ B_med.T  # (N_pools, K_coeff)

    # Feature effects per config
    feature_effects = configs @ W_med  # (n_configs, K_coeff)

    # Per-obs means: mu_pop_obs + feature_mu
    mu_pop_obs = np.sum(mu_pop[pool_idx] * x_obs, axis=1)  # (N_obs,)
    feature_mu = x_obs @ feature_effects.T  # (N_obs, n_configs)
    mu_obs = mu_pop_obs[:, None] + feature_mu  # (N_obs, n_configs)

    # Log-likelihood per obs per config
    log_lik = t_dist.logpdf(
        y_obs[:, None], df_med, loc=mu_obs, scale=sigma_eps_med
    )  # (N_obs, n_configs)

    # Sum within pools
    pool_log_liks = np.zeros((N_pools, n_configs))
    for i in range(len(y_obs)):
        pool_log_liks[pool_idx[i]] += log_lik[i]

    # Posterior = log_prior + pool_log_liks; MAP config per pool
    log_posterior = log_prior[None, :] + pool_log_liks  # (N_pools, n_configs)
    best_config_idx = np.argmax(log_posterior, axis=1)  # (N_pools,)

    return configs[best_config_idx].astype(int)  # (N_pools, K_features)


def assign_ibp_dp_joint(samples, data) -> tuple:
    """Compute MAP joint (feature, cluster) assignments for hybrid IBP+DP model.

    Enumerates all (2^K_features × K_clusters) joint configurations per pool,
    evaluates joint log-posterior, returns argmax assignments.

    Returns:
        (feature_assignments, cluster_assignments):
            feature_assignments: (N_pools, K_features) binary ndarray
            cluster_assignments: (N_pools,) int ndarray
    """
    from scipy.stats import t as t_dist
    from .model import stick_breaking_weights

    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    B_med = np.median(np.array(sample_dict["B"]), axis=0)          # (K_coeff, K_cov)
    W_med = np.median(np.array(sample_dict["W"]), axis=0)          # (K_features, K_coeff)
    v_ibp_med = np.median(np.array(sample_dict["v_ibp"]), axis=0)  # (K_features,)
    v_med = np.median(np.array(sample_dict["v"]), axis=0)          # (K_clusters-1,)
    sigma_eps_med = np.median(np.array(sample_dict["sigma_eps"]), axis=0)  # (K_clusters,)
    df_med = float(np.median(np.array(sample_dict["df"])))

    pi = np.cumprod(v_ibp_med)  # (K_features,)
    K_features = len(pi)
    K_clusters = len(sigma_eps_med)

    import jax.numpy as jnp
    w = np.array(stick_breaking_weights(jnp.array(v_med)))

    pool_idx = np.array(data["pool_idx"])
    X_pool = np.array(data["X_pool"])
    x_obs = np.array(data["x_obs"])
    y_obs = np.array(data["y_obs"])
    N_pools = data["N_pools"]

    # Enumerate all 2^K configs
    n_configs = 2 ** K_features
    configs = (
        (np.arange(n_configs)[:, None] >> np.arange(K_features)[None, :]) & 1
    ).astype(float)  # (n_configs, K_features)

    # IBP log-prior per config
    log_pi = np.log(pi + 1e-30)
    log_1mpi = np.log(1.0 - pi + 1e-30)
    log_ibp_prior = configs @ log_pi + (1.0 - configs) @ log_1mpi  # (n_configs,)

    # Joint log-prior: IBP config × DP cluster
    log_joint_prior = log_ibp_prior[:, None] + np.log(w + 1e-30)[None, :]  # (n_configs, K_clusters)

    # Population mean
    mu_pop = X_pool @ B_med.T  # (N_pools, K_coeff)
    feature_effects = configs @ W_med  # (n_configs, K_coeff)

    # Per-obs means
    mu_pop_obs = np.sum(mu_pop[pool_idx] * x_obs, axis=1)  # (N_obs,)
    feature_mu = x_obs @ feature_effects.T  # (N_obs, n_configs)
    mu_obs = mu_pop_obs[:, None] + feature_mu  # (N_obs, n_configs)

    # Log-likelihood per obs per config per cluster
    log_lik = np.zeros((len(y_obs), n_configs, K_clusters))
    for k in range(K_clusters):
        log_lik[:, :, k] = t_dist.logpdf(
            y_obs[:, None], df_med, loc=mu_obs, scale=sigma_eps_med[k]
        )

    # Sum within pools
    pool_log_liks = np.zeros((N_pools, n_configs, K_clusters))
    for i in range(len(y_obs)):
        pool_log_liks[pool_idx[i]] += log_lik[i]

    # Joint posterior: log_joint_prior + pool_log_liks
    log_posterior = log_joint_prior[None, :, :] + pool_log_liks  # (N_pools, n_configs, K_clusters)

    # Flatten to (N_pools, n_configs * K_clusters), argmax, unravel
    flat = log_posterior.reshape(N_pools, -1)
    best_flat_idx = np.argmax(flat, axis=1)
    best_config_idx = best_flat_idx // K_clusters
    best_cluster_idx = best_flat_idx % K_clusters

    feature_assignments = configs[best_config_idx].astype(int)  # (N_pools, K_features)
    cluster_assignments = best_cluster_idx.astype(np.int64)      # (N_pools,)

    return feature_assignments, cluster_assignments


def extract_structural_params(samples, data, use_median=True) -> list:
    """Extract per-pool arb frequency and noise coefficients from structural model.

    Parameters
    ----------
    samples : dict
        Posterior samples from SVI/NUTS with structural_noise_model.
    data : dict
        Output of encode_covariates_structural().

    Returns
    -------
    list of dict
        Per-pool dicts with: pool_id, chain, tokens, arb_frequency, noise_params.
    """
    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    import jax
    import jax.numpy as jnp
    from .model import _pad_with_ref

    agg_fn = np.median if use_median else np.mean

    # Cadence parameters
    alpha_0 = agg_fn(np.array(sample_dict["alpha_0"]))
    alpha_chain = agg_fn(np.array(sample_dict["alpha_chain"]), axis=0)
    alpha_tier = agg_fn(np.array(sample_dict["alpha_tier"]), axis=0)
    alpha_tvl = agg_fn(np.array(sample_dict["alpha_tvl"]))

    # MoE parameters
    W_gate = agg_fn(np.array(sample_dict["W_gate"]), axis=0)
    beta = agg_fn(np.array(sample_dict["beta"]), axis=0)

    chain_idx = np.array(data["chain_idx"])
    tier_idx = np.array(data["tier_idx"])
    X_pool = np.array(data["X_pool"])
    pool_meta = data["pool_meta"]
    pool_ids = data["pool_ids"]

    # Per-pool cadence
    padded_chain = np.concatenate([[0.0], alpha_chain])
    padded_tier = np.concatenate([[0.0], alpha_tier])

    # Per-pool log_tvl (median across observations)
    pool_idx_arr = np.array(data["pool_idx"])
    lag_log_tvl = np.array(data["lag_log_tvl"])
    N_pools = data["N_pools"]

    pool_tvl_median = np.zeros(N_pools)
    for p in range(N_pools):
        mask = pool_idx_arr == p
        if mask.any():
            pool_tvl_median[p] = np.median(lag_log_tvl[mask])

    # Per-pool noise coefficients via MoE gating
    logits = X_pool @ W_gate                       # (N_pools, K_archetypes)
    w = np.exp(logits - logits.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)           # softmax
    beta_pool = w @ beta                           # (N_pools, K_obs_coeff)

    results = []
    for i, pool_id in enumerate(pool_ids):
        meta = pool_meta.iloc[i]
        log_cadence = (
            alpha_0
            + padded_chain[chain_idx[i]]
            + padded_tier[tier_idx[i]]
            + alpha_tvl * pool_tvl_median[i]
        )
        cadence = np.exp(np.clip(log_cadence, -2.0, 6.0))
        arb_freq = int(np.clip(np.round(cadence), 1, 60))

        noise_coeffs = {
            name: float(beta_pool[i, k])
            for k, name in enumerate(OBS_COEFF_NAMES)
        }

        tokens = meta["tokens"]
        if isinstance(tokens, str):
            tokens = tokens.split(",")

        results.append({
            "pool_id": pool_id,
            "chain": str(meta["chain"]),
            "tokens": tokens,
            "arb_frequency": arb_freq,
            "noise_params": noise_coeffs,
        })

    return results


def predict_new_pool_structural(
    samples, data, chain: str, tokens: list, fee: float, tvl_est: float,
) -> dict:
    """Predict cadence and noise coefficients for a hypothetical pool.

    Uses the structural model's arb cadence parameters and MoE gating.

    Parameters
    ----------
    samples : dict
        Posterior samples from structural_noise_model.
    data : dict
        Output of encode_covariates_structural().
    chain : str
        Chain name.
    tokens : list of str
        Token symbols.
    fee : float
        Swap fee (fraction).
    tvl_est : float
        Estimated TVL in USD.
    """
    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    agg_fn = np.median

    # Cadence parameters
    alpha_0 = agg_fn(np.array(sample_dict["alpha_0"]))
    alpha_chain = agg_fn(np.array(sample_dict["alpha_chain"]), axis=0)
    alpha_tier = agg_fn(np.array(sample_dict["alpha_tier"]), axis=0)
    alpha_tvl = agg_fn(np.array(sample_dict["alpha_tvl"]))

    # MoE parameters
    W_gate = agg_fn(np.array(sample_dict["W_gate"]), axis=0)
    beta = agg_fn(np.array(sample_dict["beta"]), axis=0)

    # Construct chain and tier indices for the new pool
    chains = data["chains"]
    chain_to_idx = {c: i for i, c in enumerate(chains)}
    c_idx = chain_to_idx.get(chain, 0)  # fallback to reference

    tiers = sorted([classify_token_tier(t) for t in tokens])
    tier_a = tiers[0]
    tier_b = tiers[1] if len(tiers) > 1 else tier_a
    t_idx = _tier_pair_idx(tier_a, tier_b)

    padded_chain = np.concatenate([[0.0], alpha_chain])
    padded_tier = np.concatenate([[0.0], alpha_tier])

    log_tvl = np.log(max(tvl_est, 1.0))
    log_cadence = (
        alpha_0
        + padded_chain[c_idx]
        + padded_tier[t_idx]
        + alpha_tvl * log_tvl
    )
    cadence = np.exp(np.clip(log_cadence, -2.0, 6.0))
    arb_freq = int(np.clip(np.round(cadence), 1, 60))

    # Construct X_pool_new for gating
    col_names = data["covariate_names"]
    z_new = np.zeros(len(col_names), dtype=np.float64)
    tier_a_str = str(tier_a)
    tier_b_str = str(tier_b)

    for i, name in enumerate(col_names):
        if name == "intercept":
            z_new[i] = 1.0
        elif name == "log_fee":
            z_new[i] = np.log(max(fee, 1e-6))
        elif name == f"chain_{chain}":
            z_new[i] = 1.0
        elif name == f"tier_A_{tier_a_str}":
            z_new[i] = 1.0
        elif name == f"tier_B_{tier_b_str}":
            z_new[i] = 1.0

    # MoE gating for new pool
    logits = z_new @ W_gate  # (K_archetypes,)
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    beta_new = w @ beta  # (K_obs_coeff,)

    noise_coeffs = {
        name: float(beta_new[k])
        for k, name in enumerate(OBS_COEFF_NAMES)
    }

    return {
        "chain": chain,
        "tokens": tokens,
        "fee": fee,
        "tvl_est": tvl_est,
        "arb_frequency": arb_freq,
        "noise_params": noise_coeffs,
        "archetype_weights": w.tolist(),
    }


def run_prior_predictive(data, num_samples=500, model_fn=None) -> dict:
    """Run prior predictive check (no observations)."""
    import jax
    from numpyro.infer import Predictive

    if model_fn is None:
        model_fn = noise_model

    model_kwargs = _build_model_kwargs(data, model_fn=model_fn)
    model_kwargs["y_obs"] = None  # no observations

    predictive = Predictive(model_fn, num_samples=num_samples)
    rng_key = jax.random.PRNGKey(99)
    prior_samples = predictive(rng_key, **model_kwargs)
    prior_samples = {k: np.array(v) for k, v in prior_samples.items()}

    print(f"  Prior predictive: drew {num_samples} samples")
    y_prior = prior_samples.get("y", None)
    if y_prior is not None:
        print(f"    Prior log-volume range: "
              f"[{np.percentile(y_prior, 1):.1f}, "
              f"{np.percentile(y_prior, 99):.1f}]")
        print(f"    Observed log-volume range: "
              f"[{data['y_obs'].min():.1f}, {data['y_obs'].max():.1f}]")

    return prior_samples
