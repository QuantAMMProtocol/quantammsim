"""NumPyro noise volume models."""

import jax
import jax.numpy as jnp

from .formula_arb import formula_arb_volume_daily_jax


def _pad_with_ref(alpha):
    """Prepend a zero for the reference category."""
    return jnp.concatenate([jnp.zeros(1), alpha])


def stick_breaking_weights(v):
    """Convert Beta stick-breaking fractions to K-simplex weights.

    v: array of shape (K-1,) with values in (0, 1).
    Returns weights of shape (K,) summing to 1.

    w_1 = v_1
    w_k = v_k * prod_{j<k}(1 - v_j)   for k = 2..K-1
    w_K = prod_{j=1..K-1}(1 - v_j)
    """
    one_minus_v = 1.0 - v
    # cumprod of (1-v): [1-v_1, (1-v_1)(1-v_2), ...]
    cumprod = jnp.cumprod(one_minus_v)
    # Shift right and prepend 1.0: [1, 1-v_1, (1-v_1)(1-v_2), ...]
    remaining = jnp.concatenate([jnp.ones(1), cumprod])
    # w_k = v_k * remaining_k for k=1..K-1, w_K = remaining_{K-1}
    w_head = v * remaining[:-1]
    w_last = remaining[-1:]
    return jnp.concatenate([w_head, w_last])


def noise_model(pool_idx, X_pool, x_obs, y_obs=None,
                N_pools=None, K_coeff=4, K_cov=None,
                tier_A_per_pool=None):
    """Unified Bayesian hierarchical noise volume model.

    Student-t likelihood, per-tier sigma_eps, non-centered parameterization.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    # --- Hyperpriors ---
    B = numpyro.sample(
        "B", dist.Normal(0.0, 5.0).expand([K_coeff, K_cov]).to_event(2)
    )
    sigma_theta = numpyro.sample(
        "sigma_theta", dist.HalfNormal(2.0).expand([K_coeff]).to_event(1)
    )
    L_Omega = numpyro.sample(
        "L_Omega", dist.LKJCholesky(K_coeff, concentration=2.0)
    )

    # Student-t degrees of freedom
    df = numpyro.sample("df", dist.Gamma(2.0, 0.1))

    # Per-tier observation noise (3 tiers: blue-chip / mid-cap / long-tail)
    sigma_eps = numpyro.sample(
        "sigma_eps", dist.HalfNormal(3.0).expand([3]).to_event(1)
    )

    # --- Non-centered pool effects ---
    L_Sigma = jnp.diag(sigma_theta) @ L_Omega  # (K_coeff, K_coeff)

    with numpyro.plate("pools", N_pools):
        eta = numpyro.sample(
            "eta", dist.Normal(0.0, 1.0).expand([K_coeff]).to_event(1)
        )

    mu = X_pool @ B.T  # (N_pools, K_coeff)
    theta = mu + eta @ L_Sigma.T  # (N_pools, K_coeff)
    numpyro.deterministic("theta", theta)

    # --- Observation model ---
    theta_obs = theta[pool_idx]  # (N_obs, K_coeff)
    mu_obs = jnp.sum(theta_obs * x_obs, axis=1)  # (N_obs,)

    # Per-observation sigma from tier_A of its pool
    sigma_obs = sigma_eps[tier_A_per_pool[pool_idx]]  # (N_obs,)

    with numpyro.plate("obs", pool_idx.shape[0]):
        numpyro.sample(
            "y", dist.StudentT(df, mu_obs, sigma_obs), obs=y_obs
        )


def noise_model_dp_sigma(pool_idx, X_pool, x_obs, y_obs=None,
                          N_pools=None, K_coeff=4, K_cov=None,
                          K_clusters=6):
    """Hierarchical noise model with DP mixture on sigma_eps.

    Replaces per-tier sigma_eps with a Dirichlet Process mixture that discovers
    noise classes from data. Mean structure (B, eta, theta) is identical to
    noise_model. Observation likelihood is marginalized over cluster assignments
    via numpyro.factor when y_obs is provided.
    """
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import logsumexp

    # --- Hyperpriors (identical to noise_model) ---
    B = numpyro.sample(
        "B", dist.Normal(0.0, 5.0).expand([K_coeff, K_cov]).to_event(2)
    )
    sigma_theta = numpyro.sample(
        "sigma_theta", dist.HalfNormal(2.0).expand([K_coeff]).to_event(1)
    )
    L_Omega = numpyro.sample(
        "L_Omega", dist.LKJCholesky(K_coeff, concentration=2.0)
    )
    df = numpyro.sample("df", dist.Gamma(2.0, 0.1))

    # --- DP mixture on sigma_eps ---
    alpha_dp = numpyro.sample("alpha_dp", dist.Gamma(1.0, 1.0))
    with numpyro.plate("sticks", K_clusters - 1):
        v = numpyro.sample("v", dist.Beta(1.0, alpha_dp))
    w = numpyro.deterministic("w", stick_breaking_weights(v))

    sigma_eps = numpyro.sample(
        "sigma_eps",
        dist.HalfNormal(2.0).expand([K_clusters]).to_event(1),
    )

    # --- Non-centered pool effects (identical to noise_model) ---
    L_Sigma = jnp.diag(sigma_theta) @ L_Omega

    with numpyro.plate("pools", N_pools):
        eta = numpyro.sample(
            "eta", dist.Normal(0.0, 1.0).expand([K_coeff]).to_event(1)
        )

    mu = X_pool @ B.T
    theta = mu + eta @ L_Sigma.T
    numpyro.deterministic("theta", theta)

    # --- Observation model ---
    theta_obs = theta[pool_idx]
    mu_obs = jnp.sum(theta_obs * x_obs, axis=1)  # (N_obs,)

    if y_obs is not None:
        # Marginalize over cluster assignments per pool.
        # log p(y_pool | k) for each observation, then sum within pools.
        log_lik_per_k = dist.StudentT(
            df, mu_obs[:, None], sigma_eps[None, :]
        ).log_prob(y_obs[:, None])  # (N_obs, K_clusters)

        # Sum log-likelihoods within each pool for each cluster
        pool_log_liks = jnp.zeros((N_pools, K_clusters))
        pool_log_liks = pool_log_liks.at[pool_idx].add(log_lik_per_k)

        # log p(pool | mixture) = logsumexp_k(log w_k + sum_obs log p(y|k))
        log_marginal = logsumexp(
            jnp.log(w)[None, :] + pool_log_liks, axis=1
        )  # (N_pools,)
        numpyro.factor("log_lik", log_marginal.sum())
    else:
        # Prior predictive: sample explicit cluster assignments
        with numpyro.plate("pool_clusters", N_pools):
            z = numpyro.sample("z", dist.Categorical(probs=w))
        sigma_obs = sigma_eps[z[pool_idx]]
        with numpyro.plate("obs", pool_idx.shape[0]):
            numpyro.sample("y", dist.StudentT(df, mu_obs, sigma_obs))


def noise_model_ibp(pool_idx, X_pool, x_obs, y_obs=None,
                     N_pools=None, K_coeff=4, K_cov=None,
                     K_features=8):
    """Hierarchical noise model with IBP latent feature assignments.

    Replaces per-pool random effects (eta, sigma_theta, L_Omega) with
    compositional binary latent features via an Indian Buffet Process prior.
    Z is analytically marginalized over all 2^K binary configurations when
    y_obs is provided (logsumexp pattern). For prior predictive (y_obs=None),
    explicit z_features are sampled.

    theta_pool = X_pool @ B.T + z @ W
    """
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import logsumexp

    # --- Population effects ---
    B = numpyro.sample(
        "B", dist.Normal(0.0, 5.0).expand([K_coeff, K_cov]).to_event(2)
    )

    # Student-t degrees of freedom
    df = numpyro.sample("df", dist.Gamma(2.0, 0.1))

    # Scalar observation noise
    sigma_eps = numpyro.sample("sigma_eps", dist.HalfNormal(3.0))

    # --- IBP prior on feature prevalences ---
    alpha_ibp = numpyro.sample("alpha_ibp", dist.Gamma(2.0, 1.0))
    with numpyro.plate("features", K_features):
        v_ibp = numpyro.sample("v_ibp", dist.Beta(alpha_ibp, 1.0))
    pi = jnp.cumprod(v_ibp)  # decreasing prevalences

    # --- Feature effect matrix ---
    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(2.0))
    W = numpyro.sample(
        "W", dist.Normal(0.0, sigma_w).expand([K_features, K_coeff]).to_event(2)
    )

    # --- Population mean per pool ---
    mu_pop = X_pool @ B.T  # (N_pools, K_coeff)

    if y_obs is not None:
        # === Marginalize Z analytically ===
        # Enumerate all 2^K binary feature configurations
        n_configs = 2 ** K_features
        configs = (
            (jnp.arange(n_configs)[:, None] >> jnp.arange(K_features)[None, :]) & 1
        ).astype(jnp.float32)  # (n_configs, K_features)

        # Log-prior for each config from IBP stick-breaking
        log_pi = jnp.log(pi + 1e-30)
        log_1mpi = jnp.log(1.0 - pi + 1e-30)
        log_prior = configs @ log_pi + (1.0 - configs) @ log_1mpi  # (n_configs,)

        # Per-config feature effect
        feature_effects = configs @ W  # (n_configs, K_coeff)

        # Per-observation means
        mu_pop_obs = jnp.sum(mu_pop[pool_idx] * x_obs, axis=1)  # (N_obs,)
        feature_mu = x_obs @ feature_effects.T  # (N_obs, n_configs)
        mu_obs = mu_pop_obs[:, None] + feature_mu  # (N_obs, n_configs)

        # Log-likelihood per obs per config
        log_lik = dist.StudentT(df, mu_obs, sigma_eps).log_prob(
            y_obs[:, None]
        )  # (N_obs, n_configs)

        # Sum log-likelihoods within each pool
        pool_log_liks = jnp.zeros((N_pools, n_configs))
        pool_log_liks = pool_log_liks.at[pool_idx].add(log_lik)

        # Marginal: logsumexp over configs per pool
        log_marginal = logsumexp(
            log_prior[None, :] + pool_log_liks, axis=1
        )  # (N_pools,)
        numpyro.factor("log_lik", log_marginal.sum())
    else:
        # === Prior predictive: sample explicit assignments ===
        with numpyro.plate("pools", N_pools):
            z_features = numpyro.sample(
                "z_features",
                dist.Bernoulli(probs=pi).expand([K_features]).to_event(1),
            )

        theta = mu_pop + z_features @ W  # (N_pools, K_coeff)
        numpyro.deterministic("theta", theta)

        theta_obs = theta[pool_idx]
        mu_obs = jnp.sum(theta_obs * x_obs, axis=1)

        with numpyro.plate("obs", pool_idx.shape[0]):
            numpyro.sample(
                "y", dist.StudentT(df, mu_obs, sigma_eps),
            )


def noise_model_ibp_dp(pool_idx, X_pool, x_obs, y_obs=None,
                         N_pools=None, K_coeff=4, K_cov=None,
                         K_features=6, K_clusters=6):
    """Hybrid IBP+DP noise model.

    IBP latent features for mean heterogeneity (theta = X_pool @ B.T + z @ W),
    DP mixture for noise heterogeneity (per-cluster sigma_eps). Joint
    marginalization over (2^K_features × K_clusters) configurations when
    y_obs is provided.
    """
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import logsumexp

    # --- Population effects ---
    B = numpyro.sample(
        "B", dist.Normal(0.0, 5.0).expand([K_coeff, K_cov]).to_event(2)
    )

    # Student-t degrees of freedom
    df = numpyro.sample("df", dist.Gamma(2.0, 0.1))

    # --- IBP prior on feature prevalences ---
    alpha_ibp = numpyro.sample("alpha_ibp", dist.Gamma(2.0, 1.0))
    with numpyro.plate("features", K_features):
        v_ibp = numpyro.sample("v_ibp", dist.Beta(alpha_ibp, 1.0))
    pi = jnp.cumprod(v_ibp)  # decreasing prevalences

    # --- Feature effect matrix ---
    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(2.0))
    W = numpyro.sample(
        "W", dist.Normal(0.0, sigma_w).expand([K_features, K_coeff]).to_event(2)
    )

    # --- DP mixture on sigma_eps ---
    alpha_dp = numpyro.sample("alpha_dp", dist.Gamma(1.0, 1.0))
    with numpyro.plate("sticks", K_clusters - 1):
        v = numpyro.sample("v", dist.Beta(1.0, alpha_dp))
    w = numpyro.deterministic("w", stick_breaking_weights(v))

    sigma_eps = numpyro.sample(
        "sigma_eps",
        dist.HalfNormal(2.0).expand([K_clusters]).to_event(1),
    )

    # --- Population mean per pool ---
    mu_pop = X_pool @ B.T  # (N_pools, K_coeff)

    if y_obs is not None:
        # === Joint marginalization over IBP configs × DP clusters ===

        # Enumerate all 2^K binary feature configurations
        n_configs = 2 ** K_features
        configs = (
            (jnp.arange(n_configs)[:, None] >> jnp.arange(K_features)[None, :]) & 1
        ).astype(jnp.float32)  # (n_configs, K_features)

        # Log-prior for each IBP config
        log_pi = jnp.log(pi + 1e-30)
        log_1mpi = jnp.log(1.0 - pi + 1e-30)
        log_ibp_prior = configs @ log_pi + (1.0 - configs) @ log_1mpi  # (n_configs,)

        # Per-config feature effect
        feature_effects = configs @ W  # (n_configs, K_coeff)

        # Per-observation means for each IBP config
        mu_pop_obs = jnp.sum(mu_pop[pool_idx] * x_obs, axis=1)  # (N_obs,)
        feature_mu = x_obs @ feature_effects.T  # (N_obs, n_configs)
        mu_obs = mu_pop_obs[:, None] + feature_mu  # (N_obs, n_configs)

        # Log-likelihood per obs per IBP config per DP cluster
        log_lik = dist.StudentT(
            df, mu_obs[:, :, None], sigma_eps[None, None, :]
        ).log_prob(y_obs[:, None, None])  # (N_obs, n_configs, K_clusters)

        # Sum log-likelihoods within each pool
        pool_log_liks = jnp.zeros((N_pools, n_configs, K_clusters))
        pool_log_liks = pool_log_liks.at[pool_idx].add(log_lik)

        # Joint prior: IBP config prior × DP cluster weight
        log_joint_prior = log_ibp_prior[:, None] + jnp.log(w + 1e-30)[None, :]  # (n_configs, K_clusters)

        # Marginal log-likelihood per pool: logsumexp over (configs, clusters)
        log_marginal = logsumexp(
            log_joint_prior[None, :, :] + pool_log_liks, axis=(1, 2)
        )  # (N_pools,)
        numpyro.factor("log_lik", log_marginal.sum())
    else:
        # === Prior predictive: sample explicit assignments ===
        with numpyro.plate("pools", N_pools):
            z_features = numpyro.sample(
                "z_features",
                dist.Bernoulli(probs=pi).expand([K_features]).to_event(1),
            )
            z_cluster = numpyro.sample("z_cluster", dist.Categorical(probs=w))

        theta = mu_pop + z_features @ W  # (N_pools, K_coeff)
        numpyro.deterministic("theta", theta)

        theta_obs = theta[pool_idx]
        mu_obs = jnp.sum(theta_obs * x_obs, axis=1)
        sigma_obs = sigma_eps[z_cluster[pool_idx]]

        with numpyro.plate("obs", pool_idx.shape[0]):
            numpyro.sample("y", dist.StudentT(df, mu_obs, sigma_obs))


def structural_noise_model(pool_idx, X_pool, x_obs, y_obs=None,
                           sigma_daily=None, lag_log_tvl=None,
                           fee=None, gas=None,
                           chain_idx=None, tier_idx=None,
                           N_pools=None, K_archetypes=3,
                           n_chains=8, n_tiers=6,
                           **kwargs):
    """Structural mixture model: LVR arb + mixture-of-experts noise.

    Decomposes observed total volume into arb (structurally restricted to
    LVR formula) and noise (flexible MoE). All continuous — no discrete
    latent variables, so AutoNormal guide works directly.
    """
    import numpyro
    import numpyro.distributions as dist

    # --- Arb cadence parameters ---
    alpha_0 = numpyro.sample("alpha_0", dist.Normal(1.0, 2.0))
    alpha_chain = numpyro.sample(
        "alpha_chain",
        dist.Normal(0, 1).expand([n_chains - 1]).to_event(1),
    )
    alpha_tier = numpyro.sample(
        "alpha_tier",
        dist.Normal(0, 1).expand([n_tiers - 1]).to_event(1),
    )
    alpha_tvl = numpyro.sample("alpha_tvl", dist.Normal(0, 0.5))

    # Per-observation cadence (broadcast pool-level indices to obs)
    log_cadence = (
        alpha_0
        + _pad_with_ref(alpha_chain)[chain_idx[pool_idx]]
        + _pad_with_ref(alpha_tier)[tier_idx[pool_idx]]
        + alpha_tvl * lag_log_tvl
    )
    cadence = jnp.exp(jnp.clip(log_cadence, -2.0, 6.0))  # 0.1 to 400 min

    # V_arb per obs (deterministic given cadence + observables)
    V_arb = formula_arb_volume_daily_jax(
        sigma_daily, jnp.exp(lag_log_tvl), fee, gas, cadence,
    )

    # --- Noise MoE parameters ---
    n_obs_coeff = x_obs.shape[1]
    K_pool_cov = X_pool.shape[1]

    W_gate = numpyro.sample(
        "W_gate",
        dist.Normal(0, 1).expand([K_pool_cov, K_archetypes]).to_event(2),
    )
    beta = numpyro.sample(
        "beta",
        dist.Normal(0, 2).expand([K_archetypes, n_obs_coeff]).to_event(2),
    )

    # Per-pool soft assignment and coefficient blend
    logits = X_pool @ W_gate                            # (N_pools, K)
    w = jax.nn.softmax(logits, axis=-1)                 # (N_pools, K)
    beta_pool = jnp.einsum("pk,kc->pc", w, beta)       # (N_pools, n_obs_coeff)
    log_V_noise = jnp.sum(beta_pool[pool_idx] * x_obs, axis=1)
    V_noise = jnp.exp(log_V_noise)

    # --- Observation model ---
    df = numpyro.sample("df", dist.Gamma(2.0, 0.1))
    sigma_eps = numpyro.sample("sigma_eps", dist.HalfNormal(1.0))

    mu = jnp.log(jnp.maximum(V_arb + V_noise, 1e-6))

    if y_obs is not None:
        numpyro.sample("y", dist.StudentT(df, mu, sigma_eps), obs=y_obs)
    else:
        numpyro.sample("y", dist.StudentT(df, mu, sigma_eps))
