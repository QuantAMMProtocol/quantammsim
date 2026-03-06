"""Inference runners: SVI, NUTS, SVI-initialized NUTS."""

import numpy as np

from .constants import K_COEFF
from .model import noise_model


def _get_theta_samples(sample_dict: dict, X_pool: np.ndarray,
                       data: dict = None) -> np.ndarray:
    """Get theta samples, reconstructing from non-centered params if needed.

    MCMC.get_samples() includes the deterministic "theta" site.
    SVI's Predictive(guide, ...) does NOT — the guide only samples latent
    variables.  In that case, reconstruct theta manually:
        mu = X_pool @ B^T
        L_Sigma = diag(sigma_theta) @ L_Omega
        theta = mu + eta @ L_Sigma^T

    For marginalized IBP (W present, z_logit absent): compute MAP feature
    assignments from data, then theta = X_pool @ B.T + Z_MAP @ W.
    Requires data dict for pool_idx, x_obs, y_obs.

    For legacy STE IBP (z_logit present): theta = X_pool @ B.T + Z_hard @ W.
    """
    if "theta" in sample_dict:
        return np.array(sample_dict["theta"])

    # Marginalized IBP path: compute MAP assignments from data
    if "W" in sample_dict and "z_logit" not in sample_dict:
        B = np.array(sample_dict["B"])    # (S, K_coeff, K_cov)
        W = np.array(sample_dict["W"])    # (S, K_features, K_coeff)

        # MAP assignments: (N_pools, K_features) binary
        if "v" in sample_dict:
            # Hybrid IBP+DP: joint MAP over (features, clusters)
            from .postprocessing import assign_ibp_dp_joint
            Z_map, _ = assign_ibp_dp_joint(sample_dict, data)
        else:
            from .postprocessing import assign_ibp_features
            Z_map = assign_ibp_features(sample_dict, data)

        mu = np.einsum("pd,sjd->spj", X_pool, B)
        # Z_map doesn't vary across samples — broadcast
        feature_effect = np.einsum("pk,skj->spj", Z_map.astype(float), W)
        return mu + feature_effect

    # Legacy STE IBP path: theta = X_pool @ B.T + Z_hard @ W
    if "z_logit" in sample_dict:
        B = np.array(sample_dict["B"])                   # (S, K_coeff, K_cov)
        W = np.array(sample_dict["W"])                   # (S, K_features, K_coeff)
        z_logit = np.array(sample_dict["z_logit"])       # (S, N_pools, K_features)
        Z_hard = (z_logit > 0).astype(float)

        mu = np.einsum("pd,sjd->spj", X_pool, B)        # (S, N_pools, K_coeff)
        feature_effect = np.einsum("spk,skj->spj", Z_hard, W)
        return mu + feature_effect

    B = np.array(sample_dict["B"])                      # (S, K_coeff, K_cov)
    sigma_theta = np.array(sample_dict["sigma_theta"])   # (S, K_coeff)
    L_Omega = np.array(sample_dict["L_Omega"])           # (S, K_coeff, K_coeff)
    eta = np.array(sample_dict["eta"])                   # (S, N_pools, K_coeff)

    # mu[s, p, j] = sum_d X_pool[p, d] * B[s, j, d]   -> (S, N_pools, K_coeff)
    mu = np.einsum("pd,sjd->spj", X_pool, B)

    # L_Sigma = diag(sigma_theta) @ L_Omega  -> (S, K_coeff, K_coeff)
    L_Sigma = sigma_theta[:, :, None] * L_Omega

    # offset = eta @ L_Sigma^T  -> (S, N_pools, K_coeff)
    offset = np.einsum("spi,sji->spj", eta, L_Sigma)

    return mu + offset


def _build_model_kwargs(data: dict, model_fn=None) -> dict:
    """Convert data dict to jnp arrays for the model.

    Uses inspect.signature on model_fn to decide which kwargs to include:
    - tier_A_per_pool: only if model_fn accepts it
    - K_clusters: only if model_fn accepts it and data has it
    """
    import inspect
    import jax.numpy as jnp

    if model_fn is None:
        model_fn = noise_model

    params = set(inspect.signature(model_fn).parameters.keys())

    kwargs = dict(
        pool_idx=jnp.array(data["pool_idx"]),
        X_pool=jnp.array(data["X_pool"]),
        x_obs=jnp.array(data["x_obs"]),
        y_obs=jnp.array(data["y_obs"]),
        N_pools=data["N_pools"],
        K_coeff=K_COEFF,
        K_cov=data["K_cov"],
    )

    if "tier_A_per_pool" in params:
        kwargs["tier_A_per_pool"] = jnp.array(data["tier_A_per_pool"])

    if "K_clusters" in params and "K_clusters" in data:
        kwargs["K_clusters"] = data["K_clusters"]

    if "K_features" in params and "K_features" in data:
        kwargs["K_features"] = data["K_features"]

    # Structural model parameters
    if "sigma_daily" in params and "sigma_daily" in data:
        kwargs["sigma_daily"] = jnp.array(data["sigma_daily"])
    if "lag_log_tvl" in params and "lag_log_tvl" in data:
        kwargs["lag_log_tvl"] = jnp.array(data["lag_log_tvl"])
    if "fee" in params and "fee" in data:
        kwargs["fee"] = jnp.array(data["fee"])
    if "gas" in params and "gas" in data:
        kwargs["gas"] = jnp.array(data["gas"])
    if "chain_idx" in params and "chain_idx" in data:
        kwargs["chain_idx"] = jnp.array(data["chain_idx"])
    if "tier_idx" in params and "tier_idx" in data:
        kwargs["tier_idx"] = jnp.array(data["tier_idx"])
    if "n_chains" in params and "n_chains" in data:
        kwargs["n_chains"] = data["n_chains"]
    if "n_tiers" in params and "n_tiers" in data:
        kwargs["n_tiers"] = data["n_tiers"]
    if "K_archetypes" in params and "K_archetypes" in data:
        kwargs["K_archetypes"] = data["K_archetypes"]

    return kwargs


def run_svi(data, num_steps=20000, lr=1e-3, seed=0,
            num_samples=1000, model_fn=None) -> tuple:
    """Run SVI with AutoNormal guide.

    Returns (samples_dict, elbo_losses).
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoNormal

    if model_fn is None:
        model_fn = noise_model

    model_kwargs = _build_model_kwargs(data, model_fn=model_fn)

    print(f"\n  Running SVI: {num_steps} steps, lr={lr}")
    guide = AutoNormal(model_fn)
    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(seed)
    svi_result = svi.run(rng_key, num_steps, **model_kwargs)

    elbo_losses = np.array(svi_result.losses)
    print(f"  SVI complete. Final ELBO: {elbo_losses[-1]:.2f}")
    print(f"  ELBO last 100 std: {np.std(elbo_losses[-100:]):.2f}")

    # Draw posterior samples
    predictive = Predictive(
        guide, params=svi_result.params, num_samples=num_samples,
    )
    samples = predictive(jax.random.PRNGKey(seed + 1), **model_kwargs)
    samples = {k: np.array(v) for k, v in samples.items()}

    print(f"  Drew {num_samples} posterior samples.")
    return samples, elbo_losses


def run_nuts(data, num_warmup=1000, num_samples=2000, num_chains=4,
             target_accept=0.85, max_tree_depth=10, seed=42,
             init_values=None, model_fn=None):
    """Run NUTS MCMC.

    Uses init_to_value if init_values provided (for SVI-initialized NUTS).
    Returns the MCMC object.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS, init_to_value

    if model_fn is None:
        model_fn = noise_model

    # Note: set_host_device_count must be called before JAX init.
    # We handle this in main(). Here we just verify device count.
    n_devices = len(jax.devices("cpu"))
    if n_devices < num_chains:
        print(f"  WARNING: Only {n_devices} CPU devices available for "
              f"{num_chains} chains. Chains will run sequentially.")

    init_strategy = None
    if init_values is not None:
        init_strategy = init_to_value(
            values={k: jnp.array(v) for k, v in init_values.items()}
        )

    kernel = NUTS(
        model_fn,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    model_kwargs = _build_model_kwargs(data, model_fn=model_fn)
    rng_key = jax.random.PRNGKey(seed)

    print(f"\n  Running NUTS: {num_chains} chains x "
          f"({num_warmup} warmup + {num_samples} samples)")
    print(f"  target_accept={target_accept}, max_tree_depth={max_tree_depth}")
    if init_values is not None:
        print("  Using SVI-initialized starting values.")

    mcmc.run(rng_key, **model_kwargs)
    mcmc.print_summary(exclude_deterministic=True)
    return mcmc


def run_svi_then_nuts(data, svi_steps=5000, svi_lr=1e-3,
                      num_warmup=500, num_samples=2000, num_chains=4,
                      target_accept=0.85, max_tree_depth=10, seed=42,
                      model_fn=None):
    """Run SVI first, then use posterior means as NUTS init.

    Returns (MCMC, elbo_losses).
    """
    if model_fn is None:
        model_fn = noise_model

    # Phase 1: SVI
    print("  Phase 1: SVI warm-start")
    samples, elbo_losses = run_svi(
        data, num_steps=svi_steps, lr=svi_lr, seed=seed, num_samples=100,
        model_fn=model_fn,
    )

    # Extract posterior means for init
    init_values = {}
    skip_keys = {"y", "theta", "w"}
    for k, v in samples.items():
        if k in skip_keys:
            continue
        init_values[k] = np.mean(v, axis=0)

    # Phase 2: NUTS from SVI init
    print("\n  Phase 2: NUTS from SVI-initialized values")
    mcmc = run_nuts(
        data,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        target_accept=target_accept,
        max_tree_depth=max_tree_depth,
        seed=seed,
        init_values=init_values,
        model_fn=model_fn,
    )

    return mcmc, elbo_losses
