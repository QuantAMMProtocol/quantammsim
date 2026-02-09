"""SDE-GAN training optimized for GPU (RTX 4090, 24GB VRAM).

Unlocks what CPU couldn't do:
- Larger batch (1024): better gradient estimates, more stable GAN training
- Larger model (hidden=64, depth=2): more expressive drift/diffusion functions
- More noise dims (noise=8): richer stochastic dynamics
- ReversibleHeun solver: O(1) memory backprop through SDE solve

CPU results (hidden=16, depth=1, batch=64) for reference:
  lambda=0:   ETH 3.1x, BTC 5.1x drift (10d)
  lambda=1.0: ETH 1.8x, BTC 0.9x drift (10d)  <-- best

Usage:
    # ---- Default: single run with recommended GPU config ----
    python scripts/train_sde_gan_gpu.py

    # ---- Sweep drift_lambda to find optimal value ----
    python scripts/train_sde_gan_gpu.py --drift-lambda 0.0 0.1 0.5 1.0 2.0

    # ---- Full fat: big model + lambda sweep + save results ----
    python scripts/train_sde_gan_gpu.py \
        --hidden 128 --width 128 --depth 3 \
        --noise 12 --initial-noise 12 \
        --batch 1024 --steps 30000 --window 50 \
        --drift-lambda 0.0 0.5 1.0 2.0 \
        --output-dir results/sde_gan_sweep

    # ---- Longer windows for multi-month generation ----
    python scripts/train_sde_gan_gpu.py \
        --window 100 --steps 30000 \
        --drift-lambda 0.5 1.0

    # ---- Quick sanity check (smaller, faster) ----
    python scripts/train_sde_gan_gpu.py \
        --hidden 32 --depth 1 --batch 512 --steps 5000

    # ---- Force Euler solver (if ReversibleHeun has issues) ----
    python scripts/train_sde_gan_gpu.py --solver euler

    # ---- Different token set ----
    python scripts/train_sde_gan_gpu.py --tokens ETH BTC SOL USDC

    # ---- Auto CPU fallback ----
    # If no GPU detected, automatically reduces to CPU-safe config:
    # hidden=16, depth=1, batch=64, Euler solver
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from quantammsim.synthetic.sde_gan import (
    train_sde_gan,
    generate_paths,
    compute_daily_log_prices,
)
from quantammsim.utils.data_processing.historic_data_utils import get_historic_parquet_data


def load_data(tokens, data_root):
    """Load and prepare minute-price data."""
    price_df = get_historic_parquet_data(tokens, cols=["close"], root=data_root)
    close_cols = [f"close_{t}" for t in tokens]
    minute_prices = price_df[close_cols].values.astype(np.float64)
    valid_mask = ~np.any(np.isnan(minute_prices), axis=1)
    first_valid = np.argmax(valid_mask)
    last_valid = len(valid_mask) - np.argmax(valid_mask[::-1])
    minute_prices = minute_prices[first_valid:last_valid]
    return jnp.array(minute_prices)


def evaluate(generator, vol_scale, daily_log, tokens, real_drift, real_vol, key,
             horizons=(10, 30, 50, 100, 200), n_paths=1000):
    """Evaluate generated paths at multiple horizons."""
    n_assets = len(tokens)
    y0 = daily_log[0]
    results = {}

    for horizon in horizons:
        paths = generate_paths(generator, vol_scale, y0,
                               n_days=horizon, n_paths=n_paths, key=key)
        y0_bc = jnp.broadcast_to(y0[:, None], (n_assets, n_paths))[None, ...]
        full = jnp.concatenate([y0_bc, paths], axis=0)
        returns = jnp.diff(full, axis=0)
        drift = jnp.mean(returns, axis=(0, 2))
        vol = jnp.mean(jnp.std(returns, axis=2), axis=0)

        horizon_results = {}
        for i, t in enumerate(tokens):
            rd, rv = float(real_drift[i]), float(real_vol[i])
            d, v = float(drift[i]), float(vol[i])
            drift_ratio = d / rd if abs(rd) > 1e-8 else float('inf')
            vol_ratio = v / rv if abs(rv) > 1e-8 else float('inf')
            horizon_results[t] = {
                'drift': d, 'drift_ratio': drift_ratio,
                'vol': v, 'vol_ratio': vol_ratio,
            }
        results[horizon] = horizon_results

    return results


def print_results(results, horizons):
    """Pretty-print evaluation results."""
    tokens = list(results[horizons[0]].keys())

    # Header
    print(f"\n{'':>6}", end="")
    for h in horizons:
        print(f" | {h:>3}d drift  {h:>3}d vol ", end="")
    print()
    print("-" * (8 + len(horizons) * 22))

    for t in tokens:
        print(f"  {t:>4}", end="")
        for h in horizons:
            r = results[h][t]
            dr = r['drift_ratio']
            vr = r['vol_ratio']
            # Color-code: bold if within 0.5x of target
            dr_str = f"{dr:>5.1f}x" if abs(dr) < 100 else f"{'inf':>5}x"
            vr_str = f"{vr:>5.2f}x"
            print(f" | {dr_str}  {vr_str}", end="")
        print()


def run_experiment(minute_prices, n_assets, tokens, daily_log, real_drift, real_vol,
                   config, seed=42):
    """Run a single training + evaluation experiment."""
    print(f"\n{'='*60}")
    print(f"Config: hidden={config['hidden_size']}, depth={config['depth']}, "
          f"window={config['window_len']}, batch={config['batch_size']}, "
          f"drift_lambda={config['drift_lambda']}, steps={config['n_steps']}")
    print(f"{'='*60}")

    key = jax.random.PRNGKey(seed)
    t0 = time.time()

    generator, vol_scale, history = train_sde_gan(
        minute_prices,
        n_assets=n_assets,
        key=key,
        window_len=config['window_len'],
        initial_noise_size=config['initial_noise_size'],
        noise_size=config['noise_size'],
        hidden_size=config['hidden_size'],
        width_size=config['width_size'],
        depth=config['depth'],
        generator_lr=config['generator_lr'],
        discriminator_lr=config['discriminator_lr'],
        batch_size=config['batch_size'],
        n_steps=config['n_steps'],
        drift_lambda=config['drift_lambda'],
        use_reversible_heun=config.get('use_reversible_heun', False),
        verbose=True,
    )

    train_time = time.time() - t0
    print(f"\nTraining time: {train_time:.0f}s ({train_time/config['n_steps']*1000:.1f}ms/step)")

    # Evaluate
    key_eval = jax.random.PRNGKey(99)
    horizons = [10, 30, 50, 100, 200]
    results = evaluate(generator, vol_scale, daily_log, tokens, real_drift, real_vol,
                       key_eval, horizons=horizons, n_paths=2000)
    print_results(results, horizons)

    return generator, vol_scale, history, results, train_time


def main():
    parser = argparse.ArgumentParser(description="Train SDE-GAN on GPU")
    parser.add_argument("--tokens", nargs="+", default=["ETH", "BTC", "USDC", "PAXG"])
    parser.add_argument("--data-root", default="/Users/matthew/Projects/quantammsim/quantammsim/data")

    # Model architecture
    parser.add_argument("--hidden", type=int, default=64, help="Hidden state dim")
    parser.add_argument("--width", type=int, default=64, help="MLP width")
    parser.add_argument("--depth", type=int, default=2, help="MLP depth")
    parser.add_argument("--noise", type=int, default=8, help="Brownian noise dim")
    parser.add_argument("--initial-noise", type=int, default=8, help="Initial noise dim")

    # Training
    parser.add_argument("--window", type=int, default=50, help="Window length (days)")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--g-lr", type=float, default=2e-5, help="Generator LR")
    parser.add_argument("--d-lr", type=float, default=1e-4, help="Discriminator LR")

    # Drift penalty and solver
    parser.add_argument("--drift-lambda", type=float, nargs="+", default=[1.0],
                        help="Drift penalty weight(s). Multiple values = sweep.")
    parser.add_argument("--solver", choices=["euler", "reversible_heun"], default="reversible_heun",
                        help="SDE solver. reversible_heun = O(1) memory (GPU), euler = simpler (CPU)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results JSON")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Check GPU
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    has_gpu = any(d.platform == "gpu" for d in devices)
    if has_gpu:
        print(f"GPU detected — using full config")
    else:
        print(f"WARNING: No GPU detected. Reducing batch/model size for CPU.")
        if args.batch > 128:
            args.batch = 64
        if args.hidden > 16:
            args.hidden = 16
            args.width = 16
        if args.depth > 1:
            args.depth = 1
        if args.noise > 3:
            args.noise = 3
            args.initial_noise = 5
        args.solver = "euler"

    # Load data
    print(f"\nLoading data for {args.tokens}...")
    minute_prices = load_data(args.tokens, args.data_root)
    n_assets = len(args.tokens)
    daily_log = compute_daily_log_prices(minute_prices)

    real_daily_returns = jnp.diff(daily_log, axis=0)
    real_drift = jnp.mean(real_daily_returns, axis=0)
    real_vol = jnp.std(real_daily_returns, axis=0)
    n_days = daily_log.shape[0]

    print(f"Data: {n_days} days, {n_assets} assets")
    for i, t in enumerate(args.tokens):
        print(f"  {t}: drift={float(real_drift[i]):.6f}/day, vol={float(real_vol[i]):.6f}/day")

    # Run experiments
    all_results = []
    for drift_lambda in args.drift_lambda:
        config = {
            'hidden_size': args.hidden,
            'width_size': args.width,
            'depth': args.depth,
            'noise_size': args.noise,
            'initial_noise_size': args.initial_noise,
            'window_len': args.window,
            'batch_size': args.batch,
            'n_steps': args.steps,
            'generator_lr': args.g_lr,
            'discriminator_lr': args.d_lr,
            'drift_lambda': drift_lambda,
            'use_reversible_heun': args.solver == "reversible_heun",
        }

        gen, vol_scale, history, results, train_time = run_experiment(
            minute_prices, n_assets, args.tokens, daily_log, real_drift, real_vol,
            config, seed=args.seed,
        )

        all_results.append({
            'config': config,
            'results': {str(k): v for k, v in results.items()},
            'train_time': train_time,
            'final_wgan_loss': history[-1] if history else None,
        })

    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"sde_gan_results_{int(time.time())}.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    # Summary
    if len(args.drift_lambda) > 1:
        print(f"\n{'='*60}")
        print("SWEEP SUMMARY (10-day drift ratios)")
        print(f"{'='*60}")
        print(f"{'lambda':>8}", end="")
        for t in args.tokens:
            print(f" | {t:>8}", end="")
        print()
        for entry in all_results:
            lam = entry['config']['drift_lambda']
            print(f"  {lam:>6.2f}", end="")
            for t in args.tokens:
                dr = entry['results']['10'][t]['drift_ratio']
                print(f" | {dr:>7.1f}x", end="")
            print()


if __name__ == "__main__":
    main()
