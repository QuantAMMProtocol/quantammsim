"""JSON output and sample caching."""

import json
import os

import numpy as np

from .constants import K_COEFF, COEFF_NAMES, K_OBS_COEFF, OBS_COEFF_NAMES


def generate_output_json(pool_params, samples, data, convergence,
                         output_path, inference_config):
    """Write structured JSON output.

    Dispatches format based on whether samples contain DP mixture parameters
    (detected via "v" in sample_dict), or structural model parameters
    (detected via "W_gate" in sample_dict).
    """
    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    # Structural model path: has arb cadence params + hierarchical noise
    is_structural = "alpha_0" in sample_dict and "B" in sample_dict
    if is_structural:
        _generate_structural_output(
            pool_params, sample_dict, data, convergence,
            output_path, inference_config,
        )
        return

    # Detection priority: check hybrid first, then pure IBP, then DP
    is_ibp_dp = ("W" in sample_dict and "v" in sample_dict
                 and "z_logit" not in sample_dict)
    is_ibp = ("W" in sample_dict and "v" not in sample_dict
              and "z_logit" not in sample_dict)
    is_ibp_ste = "z_logit" in sample_dict  # legacy STE artifacts
    is_dp = "v" in sample_dict and "W" not in sample_dict

    B_median = np.median(np.array(sample_dict["B"]), axis=0).tolist()
    sigma_eps_median = np.median(
        np.array(sample_dict["sigma_eps"]), axis=0
    )
    df_median = float(np.median(np.array(sample_dict["df"])))

    if is_ibp_dp:
        model_name = "hierarchical_student_t_ibp_dp"
        sigma_eps_structure = "dp_mixture"

        W_median = np.median(np.array(sample_dict["W"]), axis=0).tolist()
        v_ibp_median = np.median(np.array(sample_dict["v_ibp"]), axis=0)
        pi = np.cumprod(v_ibp_median).tolist()

        from .model import stick_breaking_weights
        import jax.numpy as jnp
        v_median = np.median(np.array(sample_dict["v"]), axis=0)
        cluster_weights = np.array(
            stick_breaking_weights(jnp.array(v_median))
        ).tolist()

        population_effects = {
            "B": B_median,
            "sigma_eps": sigma_eps_median.tolist() if hasattr(sigma_eps_median, 'tolist') else sigma_eps_median,
            "df": df_median,
            "W": W_median,
            "feature_prevalences": pi,
            "alpha_ibp": float(
                np.median(np.array(sample_dict["alpha_ibp"]))
            ),
            "cluster_weights": cluster_weights,
            "alpha_dp": float(
                np.median(np.array(sample_dict["alpha_dp"]))
            ),
        }

        # Joint MAP assignments
        from .postprocessing import assign_ibp_dp_joint
        feat_assignments, cluster_assignments = assign_ibp_dp_joint(
            sample_dict, data
        )

        pool_entries = {}
        for i, p in enumerate(pool_params):
            entry = {
                "chain": p["chain"],
                "tokens": p["tokens"],
                "theta_median": p["theta_median"],
                "theta_std": p["theta_std"],
                "noise_params": p["noise_params"],
                "feature_assignments": feat_assignments[i].tolist(),
                "cluster_assignment": int(cluster_assignments[i]),
            }
            pool_entries[p["pool_id"]] = entry

    elif is_ibp or is_ibp_ste:
        model_name = "hierarchical_student_t_ibp"
        sigma_eps_structure = "scalar"

        W_median = np.median(np.array(sample_dict["W"]), axis=0).tolist()
        v_ibp_median = np.median(np.array(sample_dict["v_ibp"]), axis=0)
        pi = np.cumprod(v_ibp_median).tolist()

        population_effects = {
            "B": B_median,
            "sigma_eps": float(sigma_eps_median),
            "df": df_median,
            "W": W_median,
            "feature_prevalences": pi,
            "alpha_ibp": float(
                np.median(np.array(sample_dict["alpha_ibp"]))
            ),
        }

        # Per-pool feature assignments
        if is_ibp_ste:
            # Legacy STE path: threshold z_logit
            z_logit = np.array(sample_dict["z_logit"])
            z_logit_median = np.median(z_logit, axis=0)
            feature_assignments = (z_logit_median > 0).astype(int).tolist()
        else:
            # Marginalized path: MAP assignments from data
            from .postprocessing import assign_ibp_features
            feature_assignments = assign_ibp_features(
                sample_dict, data
            ).tolist()

        pool_entries = {}
        for i, p in enumerate(pool_params):
            entry = {
                "chain": p["chain"],
                "tokens": p["tokens"],
                "theta_median": p["theta_median"],
                "theta_std": p["theta_std"],
                "noise_params": p["noise_params"],
                "feature_assignments": feature_assignments[i],
            }
            pool_entries[p["pool_id"]] = entry

    elif is_dp:
        model_name = "hierarchical_student_t_dp_sigma"
        sigma_eps_structure = "dp_mixture"
    else:
        model_name = "unified_hierarchical_student_t"
        sigma_eps_structure = "per_tier"

    if not is_ibp and not is_ibp_dp:
        sigma_theta_median = np.median(
            np.array(sample_dict["sigma_theta"]), axis=0
        ).tolist()

        # Correlation matrix
        L_Omega = np.array(sample_dict["L_Omega"])
        Omega = np.einsum("sij,skj->sik", L_Omega, L_Omega)
        Omega_median = np.median(Omega, axis=0).tolist()

        population_effects = {
            "B": B_median,
            "sigma_theta": sigma_theta_median,
            "sigma_eps": sigma_eps_median.tolist() if hasattr(sigma_eps_median, 'tolist') else sigma_eps_median,
            "df": df_median,
            "correlation_matrix": Omega_median,
        }

        if is_dp:
            from .model import stick_breaking_weights
            import jax.numpy as jnp
            v_median = np.median(np.array(sample_dict["v"]), axis=0)
            w = stick_breaking_weights(jnp.array(v_median))
            population_effects["cluster_weights"] = np.array(w).tolist()
            population_effects["alpha_dp"] = float(
                np.median(np.array(sample_dict["alpha_dp"]))
            )

        pool_entries = {
            p["pool_id"]: {
                "chain": p["chain"],
                "tokens": p["tokens"],
                "theta_median": p["theta_median"],
                "theta_std": p["theta_std"],
                "noise_params": p["noise_params"],
            }
            for p in pool_params
        }

    output = {
        "model": model_name,
        "model_spec": {
            "K_coeff": K_COEFF,
            "K_cov": data["K_cov"],
            "coeff_names": COEFF_NAMES,
            "covariate_names": data["covariate_names"],
            "likelihood": "StudentT",
            "tvl_lag": "log_tvl_lag1",
            "sigma_eps_structure": sigma_eps_structure,
        },
        "inference": inference_config,
        "population_effects": population_effects,
        "convergence": convergence,
        "n_pools": len(pool_params),
        "n_obs": len(data["y_obs"]),
        "pools": pool_entries,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Wrote {len(pool_params)} pool params -> {output_path}")


def _generate_structural_output(pool_params, sample_dict, data, convergence,
                                output_path, inference_config):
    """Write structural model JSON output (LVR arb + hierarchical noise)."""
    alpha_0 = float(np.median(np.array(sample_dict["alpha_0"])))
    alpha_chain = np.median(np.array(sample_dict["alpha_chain"]), axis=0).tolist()
    alpha_tier = np.median(np.array(sample_dict["alpha_tier"]), axis=0).tolist()
    alpha_tvl = float(np.median(np.array(sample_dict["alpha_tvl"])))

    B_median = np.median(np.array(sample_dict["B"]), axis=0).tolist()
    sigma_theta_median = np.median(
        np.array(sample_dict["sigma_theta"]), axis=0
    ).tolist()

    # Correlation matrix
    L_Omega = np.array(sample_dict["L_Omega"])
    Omega = np.einsum("sij,skj->sik", L_Omega, L_Omega)
    Omega_median = np.median(Omega, axis=0).tolist()

    df_median = float(np.median(np.array(sample_dict["df"])))
    sigma_eps_median = np.median(
        np.array(sample_dict["sigma_eps"]), axis=0
    ).tolist()

    population_effects = {
        "alpha_0": alpha_0,
        "alpha_chain": alpha_chain,
        "alpha_tier": alpha_tier,
        "alpha_tvl": alpha_tvl,
        "B": B_median,
        "sigma_theta": sigma_theta_median,
        "correlation_matrix": Omega_median,
        "df": df_median,
        "sigma_eps": sigma_eps_median,
    }

    pool_entries = {}
    for p in pool_params:
        pool_entries[p["pool_id"]] = {
            "chain": p["chain"],
            "tokens": p["tokens"],
            "arb_frequency": p["arb_frequency"],
            "noise_params": p["noise_params"],
        }

    output = {
        "model": "structural_mixture",
        "model_spec": {
            "K_obs_coeff": K_OBS_COEFF,
            "obs_coeff_names": OBS_COEFF_NAMES,
            "K_cov": data["K_cov"],
            "covariate_names": data["covariate_names"],
            "likelihood": "StudentT",
        },
        "inference": inference_config,
        "population_effects": population_effects,
        "convergence": convergence,
        "n_pools": len(pool_params),
        "n_obs": len(data["y_obs"]),
        "pools": pool_entries,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Wrote {len(pool_params)} pool params -> {output_path}")


def _save_sample_cache(samples, data, cache_dir):
    """Cache posterior samples and data arrays for --predict reuse."""
    if hasattr(samples, "get_samples"):
        sample_dict = samples.get_samples()
    else:
        sample_dict = samples

    os.makedirs(cache_dir, exist_ok=True)

    # Save only the samples needed for prediction and diagnostics.
    # Skip "y" (S x N_obs, can be >1GB) and "theta" (S x N_pools x K,
    # reconstructible from B, eta, sigma_theta, L_Omega).
    skip_keys = {"y", "theta"}
    sample_cache = os.path.join(cache_dir, "unified_samples.npz")
    np.savez_compressed(
        sample_cache,
        **{k: np.array(v) for k, v in sample_dict.items()
           if k not in skip_keys},
    )

    # Data arrays for predict
    data_cache = os.path.join(cache_dir, "unified_data.json")
    cache_data = {
        "pool_ids": data["pool_ids"],
        "covariate_names": data["covariate_names"],
        "K_cov": data["K_cov"],
        "N_pools": data["N_pools"],
        "ref_chain": data["ref_chain"],
        "ref_tier_a": data["ref_tier_a"],
        "ref_tier_b": data["ref_tier_b"],
        "chains": data["chains"],
    }
    with open(data_cache, "w") as f:
        json.dump(cache_data, f, indent=2)

    print(f"  Cached samples -> {sample_cache}")
    print(f"  Cached data metadata -> {data_cache}")
