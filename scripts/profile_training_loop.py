#!/usr/bin/env python3
"""
Training loop profiler for quantammsim.

Standalone script -- scp to remote GPU box and run against existing codebase.
Produces:
  1. Coarse wall-clock: JIT compile time vs steady-state per-iteration cost
  2. Optional JAX profiler trace (viewable in tensorboard / perfetto.dev)
  3. Python cProfile of the full training run

Usage:
    # Coarse timing (JIT compile vs amortized iteration cost)
    python scripts/profile_training_loop.py

    # With JAX profiler trace
    python scripts/profile_training_loop.py --trace

    # Match your tuning config (ETH/USDC, mean_reversion_channel, fees=0)
    python scripts/profile_training_loop.py --tuning-config

    # More iterations, bigger batch
    python scripts/profile_training_loop.py -n 100 --batch-size 32

    # Python cProfile (shows where Python/host time goes)
    python scripts/profile_training_loop.py --cprofile

    # View JAX trace:
    #   tensorboard --logdir /tmp/jax-trace-quantammsim
    #   or upload to https://ui.perfetto.dev/
"""

import sys
import os
import time
import argparse
import json
from copy import deepcopy
from contextlib import contextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
import numpy as np


# ── Config ─────────────────────────────────────────────────────────────────────

def make_tuning_config(n_iterations=50, n_parameter_sets=8, batch_size=16):
    """Config matching tune_training_hyperparams.py (ETH/USDC, mean_reversion_channel, fees=0)."""
    return {
        "tokens": ["ETH", "USDC"],
        "rule": "mean_reversion_channel",
        "startDateString": "2021-01-01 00:00:00",
        "endDateString": "2022-01-01 00:00:00",  # 1 year for profiling
        "endTestDateString": "2022-06-01 00:00:00",
        "freq": "minute",
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1_000_000.0,
        "fees": 0.0,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "do_arb": True,
        "arb_frequency": 1,
        "arb_quality": 1.0,
        "minimum_weight": 0.01,
        "max_memory_days": 365,
        "bout_offset": 24 * 60 * 7 * 4,  # 4 weeks
        "return_val": "daily_log_sharpe",
        "optimisation_settings": {
            "method": "gradient_descent",
            "base_lr": 0.05,
            "optimiser": "adam",
            "batch_size": batch_size,
            "n_iterations": n_iterations,
            "n_parameter_sets": n_parameter_sets,
            "use_gradient_clipping": True,
        },
    }


def make_simple_config(n_iterations=50, n_parameter_sets=4, batch_size=8):
    """Simpler config for quick profiling (BTC/ETH momentum, shorter period)."""
    return {
        "tokens": ["BTC", "ETH"],
        "rule": "momentum",
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-06-01 00:00:00",
        "endTestDateString": "2023-09-01 00:00:00",
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1_000_000.0,
        "do_arb": True,
        "arb_quality": 1.0,
        "bout_offset": 24 * 60 * 7,  # 1 week
        "return_val": "daily_log_sharpe",
        "optimisation_settings": {
            "method": "gradient_descent",
            "base_lr": 0.05,
            "optimiser": "adam",
            "batch_size": batch_size,
            "n_iterations": n_iterations,
            "n_parameter_sets": n_parameter_sets,
        },
        "learnable_bounds_settings": {
            "min_weights_per_asset": [0.2, 0.2],
            "max_weights_per_asset": [0.8, 0.8],
            "freeze_bounds": False,
        },
    }


# ── Profiling ──────────────────────────────────────────────────────────────────

def profile_coarse(fp, trace_dir=None, root=None):
    """
    Coarse profiling: time JIT compilation vs steady-state iteration cost.

    Strategy: run with n_iterations=1 (forces JIT + 1 step), then run the full
    config. The difference reveals per-iteration amortized cost.
    """
    from quantammsim.runners.jax_runners import train_on_historic_data

    # ── Warmup: 1 iteration (JIT compile dominates) ───────────────────────
    warmup_fp = deepcopy(fp)
    warmup_fp["optimisation_settings"]["n_iterations"] = 1

    print("\nPhase 1: JIT warmup (1 iteration)...")
    jax.effects_barrier()
    t0 = time.perf_counter()
    train_on_historic_data(
        warmup_fp, verbose=False, force_init=True,
        return_training_metadata=True, iterations_per_print=999999,
        root=root,
    )
    jax.effects_barrier()
    t_warmup = time.perf_counter() - t0
    print(f"  Warmup: {t_warmup:.2f}s (includes data load + JIT compile + 1 iteration)")

    # ── Main run ──────────────────────────────────────────────────────────
    n_its = fp["optimisation_settings"]["n_iterations"]
    print(f"\nPhase 2: Main run ({n_its} iterations)...")

    ctx = jax.profiler.trace(trace_dir) if trace_dir else contextmanager(lambda: (yield))()

    jax.effects_barrier()
    with ctx:
        t0 = time.perf_counter()
        _, metadata = train_on_historic_data(
            fp, verbose=False, force_init=True,
            return_training_metadata=True, iterations_per_print=999999,
            root=root,
        )
        jax.effects_barrier()
    t_main = time.perf_counter() - t0

    actual_its = metadata["epochs_trained"]

    # ── Results ───────────────────────────────────────────────────────────
    # Warmup primes JAX's JIT cache. Main run benefits from cached compilation
    # but re-loads data. The difference between warmup and main run tells us
    # how much was JIT compile vs data+iteration cost.
    amortized_ms = t_main / max(actual_its, 1) * 1000

    # Better estimate: if JIT cache hit, main run = data_load + N*iteration.
    # Warmup = data_load + JIT_compile + 1*iteration.
    # JIT_compile_est = warmup - main/actual_its*(actual_its) ... can't cleanly separate.
    # Just report both and let user interpret.
    print(f"\n{'=' * 60}")
    print(f"  COARSE TIMING RESULTS")
    print(f"{'=' * 60}")
    print(f"  1st call (data+JIT+1it):   {t_warmup:.2f}s")
    print(f"  2nd call (data+{actual_its}its):    {t_main:.2f}s")
    if t_main < t_warmup:
        jit_est = t_warmup - t_main / actual_its
        print(f"  JIT cache hit on 2nd call: yes (saved ~{jit_est:.1f}s)")
    print(f"  Amortized per-it (2nd):    {amortized_ms:.1f}ms")
    if actual_its > 1:
        # Very rough: assume data load is similar for both calls
        # Then: warmup - main/actual_its ≈ jit_compile_time
        data_plus_one_it = t_main / actual_its  # rough per-it incl data amortization
        print(f"  Rough per-it estimate:     {data_plus_one_it*1000:.0f}ms (incl data amortization)")
    print(f"{'=' * 60}")

    return metadata


def profile_cprofile(fp, root=None):
    """
    Python cProfile: shows where host/Python time goes.
    Reveals whether the bottleneck is in JAX kernels vs Python overhead.
    """
    import cProfile
    import pstats
    from io import StringIO

    from quantammsim.runners.jax_runners import train_on_historic_data

    print("\nRunning with cProfile...")
    profiler = cProfile.Profile()
    profiler.enable()

    train_on_historic_data(
        fp, verbose=False, force_init=True,
        return_training_metadata=True, iterations_per_print=999999,
        root=root,
    )

    profiler.disable()

    # Print top functions by cumulative time
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")

    print(f"\n{'=' * 60}")
    print(f"  cPROFILE: TOP 40 BY CUMULATIVE TIME")
    print(f"{'=' * 60}")
    stats.print_stats(40)
    print(stream.getvalue())

    # Also show callers of the most expensive functions
    stream2 = StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.sort_stats("tottime")
    print(f"\n{'=' * 60}")
    print(f"  cPROFILE: TOP 20 BY SELF TIME (where CPU actually spends time)")
    print(f"{'=' * 60}")
    stats2.print_stats(20)
    print(stream2.getvalue())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile quantammsim training loop")
    parser.add_argument("-n", "--n-iterations", type=int, default=50,
                        help="Number of training iterations to profile")
    parser.add_argument("--n-param-sets", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tuning-config", action="store_true",
                        help="Use full tuning config (ETH/USDC, mean_reversion_channel)")
    parser.add_argument("--trace", action="store_true",
                        help="Generate JAX profiler trace (viewable in TensorBoard/perfetto)")
    parser.add_argument("--trace-dir", default="/tmp/jax-trace-quantammsim",
                        help="Directory for JAX profiler trace output")
    parser.add_argument("--cprofile", action="store_true",
                        help="Run Python cProfile to see host-side bottlenecks")
    parser.add_argument("--root", type=str, default=None,
                        help="Root directory for parquet data files (default: use package data)")
    args = parser.parse_args()

    if args.tuning_config:
        fp = make_tuning_config(args.n_iterations, args.n_param_sets, args.batch_size)
        label = "tuning (ETH/USDC, mean_reversion_channel, fees=0)"
    else:
        fp = make_simple_config(args.n_iterations, args.n_param_sets, args.batch_size)
        label = "simple (BTC/ETH, momentum)"

    print(f"{'=' * 60}")
    print(f"  quantammsim Training Loop Profiler")
    print(f"{'=' * 60}")
    print(f"  Config:           {label}")
    print(f"  Iterations:       {args.n_iterations}")
    print(f"  n_parameter_sets: {args.n_param_sets}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  JAX backend:      {jax.default_backend()}")
    print(f"  JAX devices:      {jax.devices()}")
    print(f"  x64 enabled:      {jax.config.jax_enable_x64}")
    if args.root:
        print(f"  Data root:        {args.root}")
    print(f"{'=' * 60}")

    print("\n--- COARSE PROFILING ---")
    trace_dir = args.trace_dir if args.trace else None
    metadata = profile_coarse(fp, trace_dir=trace_dir, root=args.root)

    if args.trace:
        print(f"\n  JAX trace written to: {trace_dir}")
        print(f"  View: tensorboard --logdir {trace_dir}")
        print(f"  Or upload to: https://ui.perfetto.dev/")

    if args.cprofile:
        print("\n--- cPROFILE ---")
        profile_cprofile(fp, root=args.root)


if __name__ == "__main__":
    main()
