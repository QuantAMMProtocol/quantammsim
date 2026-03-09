"""CLI entry point for noise calibration."""

import argparse
import json
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .constants import CACHE_DIR
from .data_pipeline import (
    enumerate_balancer_pools, fetch_all_snapshots,
    fetch_token_prices, assemble_panel,
)
from .data_validation import validate_panel
from .covariate_encoding import encode_covariates
from .inference import run_svi, run_nuts, run_svi_then_nuts
from .postprocessing import (
    extract_noise_params, predict_new_pool,
    check_convergence, run_prior_predictive,
)
from .plotting import plot_diagnostics
from .output import generate_output_json, _save_sample_cache


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Bayesian hierarchical noise volume model "
                    "for Balancer pools (gold standard)"
    )

    # Actions
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch data from Balancer API")
    parser.add_argument("--fit", action="store_true",
                        help="Run inference (SVI default)")
    parser.add_argument("--nuts", action="store_true",
                        help="Use NUTS instead of SVI")
    parser.add_argument("--svi-init-nuts", action="store_true",
                        help="SVI-initialized NUTS (fast warmup)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate diagnostic plots")
    parser.add_argument("--prior-predictive", action="store_true",
                        help="Include prior predictive check")
    parser.add_argument("--validate", action="store_true",
                        help="Run data validation pass")
    parser.add_argument("--predict", action="store_true",
                        help="Predict for unseen pool")

    # Output
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--output-dir", default="results",
                        help="Plot output directory (default: results)")

    # Predict args
    parser.add_argument("--chain", default=None,
                        help="Chain for --predict")
    parser.add_argument("--tokens", nargs="+", default=None,
                        help="Tokens for --predict")
    parser.add_argument("--fee", type=float, default=0.003,
                        help="Fee for --predict")

    # NUTS hyperparameters
    parser.add_argument("--num-warmup", type=int, default=1000,
                        help="NUTS warmup iterations (default: 1000)")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="NUTS/SVI samples (default: 2000)")
    parser.add_argument("--num-chains", type=int, default=4,
                        help="NUTS chains (default: 4)")
    parser.add_argument("--target-accept", type=float, default=0.85,
                        help="NUTS target accept prob (default: 0.85)")
    parser.add_argument("--max-tree-depth", type=int, default=10,
                        help="NUTS max tree depth (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # SVI hyperparameters
    parser.add_argument("--svi-steps", type=int, default=20000,
                        help="SVI optimization steps (default: 20000)")
    parser.add_argument("--svi-lr", type=float, default=1e-3,
                        help="SVI learning rate (default: 1e-3)")

    # Model variant
    parser.add_argument("--model", choices=["tier", "dp_sigma", "ibp", "ibp_dp",
                                             "structural"],
                        default="tier",
                        help="Noise model variant: 'tier' (per-tier sigma_eps), "
                        "'dp_sigma' (DP mixture on sigma_eps), "
                        "'ibp' (IBP latent features), "
                        "'ibp_dp' (IBP features + DP noise clusters), or "
                        "'structural' (structural mixture: arb + MoE noise)")
    parser.add_argument("--k-clusters", type=int, default=6,
                        help="Number of DP mixture components "
                        "(capacity ceiling, default: 6)")
    parser.add_argument("--k-features", type=int, default=6,
                        help="Number of IBP latent features "
                        "(default: 6)")

    # Data
    parser.add_argument("--train-days", type=int, default=90,
                        help="Use only the last N days of data for fitting "
                        "(default: 90). Aligns with Balancer API hourly price "
                        "coverage window. Set to 0 to use all data.")
    parser.add_argument("--min-tvl", type=float, default=10000.0,
                        help="Pool enumeration TVL filter")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache directory")
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"],
                        default="auto",
                        help="JAX device (default: auto)")

    return parser.parse_args()


def main():
    args = _parse_args()

    if not any([args.fetch, args.fit, args.predict, args.validate]):
        print("ERROR: At least one of --fetch, --fit, --predict, --validate "
              "is required", file=sys.stderr)
        sys.exit(1)

    cache_dir = args.cache_dir or CACHE_DIR

    # --- JAX setup (BEFORE any JAX ops / imports) ---
    if args.fit or args.predict or args.prior_predictive:
        # Set device before importing JAX
        if args.device == "cpu":
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        elif args.device == "gpu":
            os.environ.setdefault("JAX_PLATFORMS", "cuda")
        # auto: don't touch JAX_PLATFORMS, let JAX pick

        # Set host device count for NUTS multi-chain BEFORE JAX init
        if args.nuts or args.svi_init_nuts:
            import numpyro as _np_pre
            _np_pre.set_host_device_count(
                min(args.num_chains, os.cpu_count() or 4)
            )

        import jax
        import numpyro
        numpyro.enable_x64()

    # --- File paths ---
    pools_cache = os.path.join(cache_dir, "pools.parquet")
    snaps_cache = os.path.join(cache_dir, "pool_snapshots.parquet")
    prices_cache = os.path.join(cache_dir, "token_prices")
    panel_cache = os.path.join(cache_dir, "panel.parquet")

    # --- Fetch ---
    if args.fetch:
        print("Phase 1: Fetching data from Balancer API")
        print("=" * 60)

        print("\n1. Enumerating pools...")
        pools_df = enumerate_balancer_pools(min_tvl=args.min_tvl)
        os.makedirs(cache_dir, exist_ok=True)
        pools_df.to_parquet(pools_cache, index=False)
        print(f"   Saved {len(pools_df)} pools -> {pools_cache}")

        print("\n2. Fetching daily snapshots...")
        snapshots_df = fetch_all_snapshots(pools_df, cache_path=snaps_cache)

        print("\n3. Fetching token prices...")
        token_addr_by_chain = {}
        for _, pool in pools_df.iterrows():
            chain = pool["chain"]
            tokens = pool["tokens"]
            addresses = pool["token_addresses"]
            if chain not in token_addr_by_chain:
                token_addr_by_chain[chain] = {}
            for sym, addr in zip(tokens, addresses):
                if sym and addr:
                    token_addr_by_chain[chain][sym] = addr

        token_prices = fetch_token_prices(
            token_addr_by_chain, cache_dir=prices_cache
        )

        print("\n4. Assembling panel (with lagged TVL)...")
        panel = assemble_panel(pools_df, snapshots_df, token_prices)
        panel.to_parquet(panel_cache, index=False)
        print(f"   Saved panel -> {panel_cache}")

        print(f"\nFetch complete. Panel: {len(panel)} obs, "
              f"{panel['pool_id'].nunique()} pools")

    # --- Validate ---
    if args.validate:
        if not os.path.exists(panel_cache):
            print(f"ERROR: Panel cache not found at {panel_cache}",
                  file=sys.stderr)
            print("Run with --fetch first.", file=sys.stderr)
            sys.exit(1)
        panel = pd.read_parquet(panel_cache)
        validate_panel(panel)

    # --- Fit ---
    if args.fit:
        print("\nUnified Noise Volume Model")
        print("=" * 60)

        # Load panel
        if not os.path.exists(panel_cache):
            print(f"ERROR: Panel cache not found at {panel_cache}",
                  file=sys.stderr)
            print("Run with --fetch first.", file=sys.stderr)
            sys.exit(1)

        panel = pd.read_parquet(panel_cache)
        print(f"  Loaded panel: {len(panel)} obs, "
              f"{panel['pool_id'].nunique()} pools, "
              f"{panel['chain'].nunique()} chains")

        # Filter to recent window for training
        if args.train_days > 0:
            max_date = panel["date"].max()
            if not isinstance(max_date, date):
                max_date = pd.Timestamp(max_date).date()
            cutoff = max_date - timedelta(days=args.train_days)
            n_before = len(panel)
            panel = panel[
                panel["date"].apply(
                    lambda d: d >= cutoff if isinstance(d, date)
                    else pd.Timestamp(d).date() >= cutoff
                )
            ].copy()
            print(f"  Filtered to last {args.train_days} days "
                  f"(>= {cutoff}): {len(panel)} obs "
                  f"(dropped {n_before - len(panel)})")

        # Ensure lagged TVL exists (in case loaded from old cache)
        if "log_tvl_lag1" not in panel.columns:
            print("  Adding lagged TVL to cached panel...")
            panel = panel.sort_values(["pool_id", "date"]).reset_index(drop=True)
            panel["log_tvl_lag1"] = panel.groupby("pool_id")["log_tvl"].shift(1)
            panel = panel.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)

        # Filter: need at least 10 days per pool
        pool_counts = panel.groupby("pool_id").size()
        valid_pools = pool_counts[pool_counts >= 10].index
        panel = panel[panel["pool_id"].isin(valid_pools)].copy()
        print(f"  After filtering (>= 10 days): {len(panel)} obs, "
              f"{panel['pool_id'].nunique()} pools")

        # Select model variant
        if args.model == "structural":
            from .model import structural_noise_model
            from .covariate_encoding import encode_covariates_structural
            model_fn = structural_noise_model
            data = encode_covariates_structural(panel)
            print(f"  Model: structural mixture (arb + MoE noise)")
        elif args.model == "ibp_dp":
            from .model import noise_model_ibp_dp
            model_fn = noise_model_ibp_dp
            data = encode_covariates(panel, include_tiers=False)
            data["K_features"] = args.k_features
            data["K_clusters"] = args.k_clusters
            print(f"  Model: IBP+DP hybrid "
                  f"(K_features={args.k_features}, "
                  f"K_clusters={args.k_clusters})")
        elif args.model == "ibp":
            from .model import noise_model_ibp
            model_fn = noise_model_ibp
            data = encode_covariates(panel, include_tiers=False)
            data["K_features"] = args.k_features
            print(f"  Model: IBP latent features "
                  f"(K_features={args.k_features})")
        elif args.model == "dp_sigma":
            from .model import noise_model_dp_sigma
            model_fn = noise_model_dp_sigma
            data = encode_covariates(panel, include_tiers=False)
            data["K_clusters"] = args.k_clusters
            print(f"  Model: DP mixture on sigma_eps "
                  f"(K_clusters={args.k_clusters})")
        else:
            model_fn = None  # default = noise_model
            data = encode_covariates(panel)

        # Prior predictive
        prior_samples = None
        if args.prior_predictive:
            print("\n  Running prior predictive check...")
            prior_samples = run_prior_predictive(data, model_fn=model_fn)

        # Inference
        mcmc_obj = None
        elbo_losses = None
        inference_config = {"seed": args.seed}

        if args.svi_init_nuts:
            inference_config["method"] = "svi_init_nuts"
            inference_config["svi_steps"] = args.svi_steps
            inference_config["svi_lr"] = args.svi_lr
            inference_config["num_warmup"] = args.num_warmup
            inference_config["num_samples"] = args.num_samples
            inference_config["num_chains"] = args.num_chains
            inference_config["target_accept"] = args.target_accept
            inference_config["max_tree_depth"] = args.max_tree_depth

            mcmc_obj, elbo_losses = run_svi_then_nuts(
                data,
                svi_steps=args.svi_steps,
                svi_lr=args.svi_lr,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=args.num_chains,
                target_accept=args.target_accept,
                max_tree_depth=args.max_tree_depth,
                seed=args.seed,
                model_fn=model_fn,
            )
            samples = mcmc_obj
            convergence = check_convergence(mcmc_obj, method="nuts")

        elif args.nuts:
            inference_config["method"] = "nuts"
            inference_config["num_warmup"] = args.num_warmup
            inference_config["num_samples"] = args.num_samples
            inference_config["num_chains"] = args.num_chains
            inference_config["target_accept"] = args.target_accept
            inference_config["max_tree_depth"] = args.max_tree_depth

            mcmc_obj = run_nuts(
                data,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                num_chains=args.num_chains,
                target_accept=args.target_accept,
                max_tree_depth=args.max_tree_depth,
                seed=args.seed,
                model_fn=model_fn,
            )
            samples = mcmc_obj
            convergence = check_convergence(mcmc_obj, method="nuts")

        else:
            inference_config["method"] = "svi"
            inference_config["svi_steps"] = args.svi_steps
            inference_config["svi_lr"] = args.svi_lr
            inference_config["num_samples"] = args.num_samples

            samples, elbo_losses = run_svi(
                data,
                num_steps=args.svi_steps,
                lr=args.svi_lr,
                seed=args.seed,
                num_samples=args.num_samples,
                model_fn=model_fn,
            )
            convergence = check_convergence(elbo_losses, method="svi")

        if args.model == "structural":
            from .postprocessing import extract_structural_params
            pool_params = extract_structural_params(samples, data)
            arb_freqs = [p["arb_frequency"] for p in pool_params]
            print(f"\n  Per-pool arb_frequency: "
                  f"mean={np.mean(arb_freqs):.1f}, "
                  f"range=[{np.min(arb_freqs)}, {np.max(arb_freqs)}]")
        else:
            pool_params = extract_noise_params(samples, data)
            b_c_vals = [p["noise_params"]["b_c"] for p in pool_params]
            b_0_vals = [p["noise_params"]["b_0"] for p in pool_params]
            print(f"\n  Per-pool b_c: mean={np.mean(b_c_vals):.3f}, "
                  f"std={np.std(b_c_vals):.3f}, "
                  f"range=[{np.min(b_c_vals):.3f}, {np.max(b_c_vals):.3f}]")
            print(f"  Per-pool b_0: mean={np.mean(b_0_vals):.3f}, "
                  f"std={np.std(b_0_vals):.3f}")

        if args.output:
            generate_output_json(
                pool_params, samples, data, convergence,
                args.output, inference_config,
            )

        if args.plot:
            print("\nGenerating diagnostic plots...")
            plot_diagnostics(
                samples, data, output_dir=args.output_dir,
                elbo_losses=elbo_losses, mcmc=mcmc_obj,
                prior_samples=prior_samples,
            )

        # Cache samples for --predict
        _save_sample_cache(samples, data, cache_dir)

    # --- Predict ---
    if args.predict:
        if args.chain is None or args.tokens is None:
            print("ERROR: --predict requires --chain and --tokens",
                  file=sys.stderr)
            sys.exit(1)

        # Load cached samples
        sample_cache = os.path.join(cache_dir, "unified_samples.npz")
        data_cache = os.path.join(cache_dir, "unified_data.json")

        if not os.path.exists(sample_cache):
            print(f"ERROR: Sample cache not found at {sample_cache}",
                  file=sys.stderr)
            print("Run with --fit first.", file=sys.stderr)
            sys.exit(1)

        cached = np.load(sample_cache)
        sample_dict = {k: cached[k] for k in cached.files}

        with open(data_cache) as f:
            data_meta = json.load(f)

        result = predict_new_pool(
            sample_dict, data_meta, args.chain, args.tokens, args.fee
        )
        print(json.dumps(result, indent=2))
