#!/usr/bin/env python3
"""
CMA-ES memory profiler.

Uses XLA's compiled memory_analysis() to measure the actual temp memory
XLA allocates for the CMA-ES population evaluation in float32 vs float64.

With --execute, also runs the compiled computation and measures wall-clock
time, effective throughput (GFLOP/s), and speedup ratio.

We compile:
  1. eval_population = jit(vmap(eval_single)) — the per-generation fitness evaluation
     This is the dominant cost: pop_size × n_eval_points forward passes per generation.

Unlike BFGS, CMA-ES never computes gradients, so there's no value_and_grad to
profile. The eigendecomposition (10×10 matrix) is negligible.

Usage:
    # Quick comparison: float32 vs float64 (compile-time only)
    python scripts/profile_cmaes_memory.py

    # With wall-clock execution timing
    python scripts/profile_cmaes_memory.py --execute

    # Sweep population sizes with execution timing
    python scripts/profile_cmaes_memory.py --sweep --execute --max-pop 32

    # Save results
    python scripts/profile_cmaes_memory.py --sweep --execute --json results.json
"""
from __future__ import annotations

import sys
import os
import io
import time
import argparse
import json
import gc
from contextlib import redirect_stdout
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, clear_caches
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

from dateutil.relativedelta import relativedelta

from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set
from quantammsim.utils.data_processing.historic_data_utils import get_data_dict
from quantammsim.pools.creator import create_pool
from quantammsim.core_simulator.forward_pass import forward_pass
from quantammsim.runners.jax_runner_utils import (
    Hashabledict,
    get_unique_tokens,
    generate_evaluation_points,
    create_static_dict,
    get_sig_variations,
)
from quantammsim.training.backpropagation import (
    batched_partial_training_step_factory,
    batched_objective_factory,
)
from quantammsim.training.cma_es import (
    default_params as cma_default_params,
    init_cmaes,
    ask as cma_ask,
    tell as cma_tell,
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class MemoryResult:
    pop_size: int
    n_eval_points: int
    compute_dtype: str
    n_flat_params: int = 0
    # From compiled.memory_analysis()
    temp_bytes: int = 0
    argument_bytes: int = 0
    output_bytes: int = 0
    # From compiled.cost_analysis()
    flops: int = 0
    transcendentals: int = 0
    # Timing
    compile_time_s: float = 0.0
    # Execution timing (--execute mode)
    eval_wall_ms: float = 0.0       # median wall-clock per eval_population call
    eval_gflops: float = 0.0        # effective GFLOP/s
    gen_wall_ms: float = 0.0        # wall-clock per full generation (ask+eval+tell)
    error: str = ""

    @property
    def temp_mb(self) -> float:
        return self.temp_bytes / (1024 * 1024)

    @property
    def argument_mb(self) -> float:
        return self.argument_bytes / (1024 * 1024)


# ── Setup ─────────────────────────────────────────────────────────────────────

def build_fingerprint(
    n_eval_points: int,
    compute_dtype: str,
    months: int,
    fees: float,
) -> dict:
    start = datetime(2021, 6, 1)
    end_train = start + relativedelta(months=months)
    end_test = end_train + relativedelta(months=1)

    fp = {
        "tokens": ["ETH", "USDC"],
        "rule": "mean_reversion_channel",
        "startDateString": start.strftime("%Y-%m-%d %H:%M:%S"),
        "endDateString": end_train.strftime("%Y-%m-%d %H:%M:%S"),
        "endTestDateString": end_test.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1_000_000.0,
        "fees": fees,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "do_arb": True,
        "arb_frequency": 1,
        "minimum_weight": 0.01,
        "max_memory_days": 365,
        "bout_offset": 0,
        "return_val": "daily_log_sharpe",
        "optimisation_settings": {
            "method": "cma_es",
            "n_parameter_sets": 1,
            "noise_scale": 0.3,
            "val_fraction": 0.0,
            "cma_es_settings": {
                "n_generations": 300,
                "sigma0": 0.5,
                "tol": 1e-8,
                "n_evaluation_points": n_eval_points,
                "compute_dtype": compute_dtype,
            },
        },
    }
    recursive_default_set(fp, run_fingerprint_defaults)
    return fp


def setup_cmaes_computation(fp, pop_size=None, root=None):
    """
    Replicate the CMA-ES setup from jax_runners.train_on_historic_data,
    returning all the pieces needed to build the compiled evaluation.
    """
    cma_settings = fp["optimisation_settings"]["cma_es_settings"]
    compute_dtype_str = cma_settings.get("compute_dtype", "float32")
    use_x64 = compute_dtype_str != "float32"
    jax.config.update("jax_enable_x64", use_x64)

    unique_tokens = get_unique_tokens(fp)
    n_tokens = len(unique_tokens)
    n_assets = n_tokens
    all_sig_variations = get_sig_variations(n_assets)
    n_parameter_sets = 1  # Single set for profiling

    np.random.seed(0)

    data_dict = get_data_dict(
        unique_tokens,
        fp,
        data_kind=fp["optimisation_settings"]["training_data_kind"],
        root=root,
        max_memory_days=fp["max_memory_days"],
        start_date_string=fp["startDateString"],
        end_time_string=fp["endDateString"],
        start_time_test_string=fp["endDateString"],
        end_time_test_string=fp["endTestDateString"],
        max_mc_version=fp["optimisation_settings"]["max_mc_version"],
        do_test_period=True,
    )

    bout_length_window = data_dict["bout_length"] - fp["bout_offset"]
    sampling_end_idx = data_dict["end_idx"]

    pool = create_pool(fp["rule"])
    initial_params = {
        "initial_memory_length": fp["initial_memory_length"],
        "initial_memory_length_delta": fp["initial_memory_length_delta"],
        "initial_k_per_day": fp["initial_k_per_day"],
        "initial_weights_logits": fp["initial_weights_logits"],
        "initial_log_amplitude": fp["initial_log_amplitude"],
        "initial_raw_width": fp["initial_raw_width"],
        "initial_raw_exponents": fp["initial_raw_exponents"],
        "initial_pre_exp_scaling": fp["initial_pre_exp_scaling"],
        "min_weights_per_asset": fp.get("learnable_bounds_settings", {}).get("min_weights_per_asset"),
        "max_weights_per_asset": fp.get("learnable_bounds_settings", {}).get("max_weights_per_asset"),
    }
    params = pool.init_parameters(
        initial_params, fp, n_tokens, n_parameter_sets, noise="gaussian",
    )

    base_static_dict = create_static_dict(
        fp,
        bout_length=bout_length_window,
        all_sig_variations=all_sig_variations,
        overrides={
            "n_assets": n_assets,
            "training_data_kind": fp["optimisation_settings"]["training_data_kind"],
            "do_trades": False,
        },
    )

    n_eval_points = cma_settings["n_evaluation_points"]

    partial_training_step = Partial(
        forward_pass,
        prices=data_dict["prices"],
        static_dict=Hashabledict(base_static_dict),
        pool=pool,
    )

    min_spacing = data_dict["bout_length"] // 2
    evaluation_starts = generate_evaluation_points(
        data_dict["start_idx"],
        sampling_end_idx,
        bout_length_window,
        n_eval_points,
        min_spacing,
        fp["optimisation_settings"]["initial_random_key"],
    )
    fixed_start_indexes = jnp.array(
        [(s, 0) for s in evaluation_starts], dtype=jnp.int32
    )

    # Build objective and flatten
    batched_pts = batched_partial_training_step_factory(partial_training_step)
    batched_obj = batched_objective_factory(batched_pts)

    params_single = {}
    for k, v in params.items():
        if k == "subsidary_params":
            params_single[k] = v
        elif hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == n_parameter_sets:
            params_single[k] = v[0]
        else:
            params_single[k] = v

    flat_x0, unravel_fn = ravel_pytree(params_single)
    n_flat = flat_x0.shape[0]

    # Determine population size
    cma_params = cma_default_params(n_flat)
    if pop_size is not None:
        lam = pop_size
    else:
        lam = cma_params["lam"]

    return (
        batched_obj,
        unravel_fn,
        fixed_start_indexes,
        flat_x0,
        n_flat,
        lam,
        cma_params,
    )


def compile_cmaes_eval(
    batched_obj,
    unravel_fn,
    fixed_start_indexes,
    flat_x0,
    n_flat: int,
    pop_size: int,
) -> tuple:
    """
    Build and compile the CMA-ES population evaluation.
    Returns (compiled_eval, sample_pop, compile_time_s).
    """
    def eval_single(flat_x):
        p = unravel_fn(flat_x)
        return -batched_obj(p, fixed_start_indexes)

    eval_population = jit(vmap(eval_single))

    # Create a sample population for compilation
    key = random.key(0)
    sample_pop = flat_x0[None, :] + 0.5 * random.normal(key, shape=(pop_size, n_flat))

    t0 = time.perf_counter()
    lowered = eval_population.lower(sample_pop)
    compiled = lowered.compile()
    compile_time = time.perf_counter() - t0

    return compiled, eval_population, sample_pop, compile_time


def extract_stats(compiled) -> dict:
    """Extract memory_analysis and cost_analysis from a compiled object."""
    stats = {}

    try:
        mem = compiled.memory_analysis()
        stats["temp_bytes"] = mem.temp_size_in_bytes
        stats["argument_bytes"] = mem.argument_size_in_bytes
        stats["output_bytes"] = mem.output_size_in_bytes
    except Exception as e:
        stats["error"] = f"memory_analysis: {e}"

    try:
        cost = compiled.cost_analysis()
        if isinstance(cost, list):
            cost = cost[0]
        if cost:
            stats["flops"] = int(cost.get("flops", 0))
            stats["transcendentals"] = int(cost.get("transcendentals", 0))
    except Exception:
        pass

    return stats


# ── Execution timing ──────────────────────────────────────────────────────

def time_execution(compiled_eval, eval_fn, sample_pop, flat_x0, cma_params,
                   pop_size, n_flat, eval_flops, reps=5):
    """
    Run the compiled evaluation and measure wall-clock time.
    Returns (eval_wall_ms, eval_gflops, gen_wall_ms).
    """
    # Warm up eval
    out = compiled_eval(sample_pop)
    jax.block_until_ready(out)

    # Time eval_population over multiple reps
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = compiled_eval(sample_pop)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    eval_wall_s = float(np.median(times))
    eval_wall_ms = eval_wall_s * 1000
    eval_gflops = (eval_flops / 1e9) / eval_wall_s if eval_wall_s > 0 else 0

    # Time a full generation: ask + eval + tell
    state = init_cmaes(flat_x0, sigma=0.5)
    key = random.key(42)

    # Warm up full generation
    key, subkey = random.split(key)
    pop = cma_ask(state, subkey, pop_size)
    fitness = eval_fn(pop)
    jax.block_until_ready(fitness)
    state = cma_tell(state, pop, fitness, cma_params)

    gen_times = []
    for _ in range(reps):
        key, subkey = random.split(key)
        t0 = time.perf_counter()
        pop = cma_ask(state, subkey, pop_size)
        fitness = eval_fn(pop)
        jax.block_until_ready(fitness)
        state = cma_tell(state, pop, fitness, cma_params)
        gen_times.append(time.perf_counter() - t0)
    gen_wall_ms = float(np.median(gen_times)) * 1000

    return eval_wall_ms, eval_gflops, gen_wall_ms


# ── Display ───────────────────────────────────────────────────────────────────

def print_header(execute=False):
    hdr = (f"{'dtype':>7} {'pop':>5} {'n_eval':>6} {'n_flat':>6} "
           f"{'temp_MB':>10} {'arg_MB':>10} "
           f"{'GFLOP':>10} {'compile_s':>10}")
    if execute:
        hdr += f" {'eval_ms':>10} {'GFLOP/s':>10} {'gen_ms':>10}"
    hdr += f" {'status':>8}"
    print(hdr)
    print("-" * (82 + (32 if execute else 0)))


def print_row(r: MemoryResult, execute=False):
    if not r.error:
        gflop = r.flops / 1e9 if r.flops else 0
        row = (f"{r.compute_dtype:>7} {r.pop_size:>5} {r.n_eval_points:>6} "
               f"{r.n_flat_params:>6} "
               f"{r.temp_mb:>10.1f} {r.argument_mb:>10.1f} "
               f"{gflop:>10.2f} {r.compile_time_s:>10.1f}")
        if execute:
            row += (f" {r.eval_wall_ms:>10.1f} {r.eval_gflops:>10.2f}"
                    f" {r.gen_wall_ms:>10.1f}")
        row += f" {'OK':>8}"
        print(row)
    else:
        row = (f"{r.compute_dtype:>7} {r.pop_size:>5} {r.n_eval_points:>6} "
               f"{r.n_flat_params:>6} "
               f"{'':>10} {'':>10} "
               f"{'':>10} {r.compile_time_s:>10.1f}")
        if execute:
            row += f" {'':>10} {'':>10} {'':>10}"
        row += f" {'ERR':>8}"
        print(row)
        print(f"  error: {r.error}")


def print_comparison(results: List[MemoryResult]):
    f64 = [r for r in results if r.compute_dtype == "float64" and not r.error]
    f32 = [r for r in results if r.compute_dtype == "float32" and not r.error]

    if not (f64 and f32):
        return

    r64, r32 = f64[0], f32[0]

    print(f"\n  {'metric':<25} {'float64':>12} {'float32':>12} {'delta':>12}")
    print(f"  {'-'*61}")

    # Temp memory
    t64, t32 = r64.temp_mb, r32.temp_mb
    if t64 > 0:
        delta = (t32 / t64 - 1) * 100
        print(f"  {'temp memory (MB)':<25} {t64:>12.1f} {t32:>12.1f} {delta:>+11.1f}%")

    # Argument memory
    a64, a32 = r64.argument_mb, r32.argument_mb
    if a64 > 0:
        delta = (a32 / a64 - 1) * 100
        print(f"  {'argument memory (MB)':<25} {a64:>12.1f} {a32:>12.1f} {delta:>+11.1f}%")

    # FLOPs
    f_64, f_32 = r64.flops / 1e9, r32.flops / 1e9
    if f_64 > 0:
        delta = (f_32 / f_64 - 1) * 100
        print(f"  {'GFLOP':<25} {f_64:>12.2f} {f_32:>12.2f} {delta:>+11.1f}%")

    # Compile time
    c64, c32 = r64.compile_time_s, r32.compile_time_s
    print(f"  {'compile time (s)':<25} {c64:>12.1f} {c32:>12.1f}")

    # Execution timing (if available)
    if r64.eval_wall_ms > 0 and r32.eval_wall_ms > 0:
        print()
        w64, w32 = r64.eval_wall_ms, r32.eval_wall_ms
        speedup = w64 / w32 if w32 > 0 else 0
        print(f"  {'eval wall-clock (ms)':<25} {w64:>12.1f} {w32:>12.1f} {speedup:>11.1f}x")
        g64, g32 = r64.eval_gflops, r32.eval_gflops
        print(f"  {'eval throughput (GFLOP/s)':<25} {g64:>12.2f} {g32:>12.2f}")
        if r64.gen_wall_ms > 0 and r32.gen_wall_ms > 0:
            gen64, gen32 = r64.gen_wall_ms, r32.gen_wall_ms
            speedup_g = gen64 / gen32 if gen32 > 0 else 0
            print(f"  {'full generation (ms)':<25} {gen64:>12.1f} {gen32:>12.1f} {speedup_g:>11.1f}x")


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_config(
    pop_size: Optional[int],
    n_eval_points: int,
    compute_dtype: str,
    months: int,
    fees: float,
    root: Optional[str],
    execute: bool = False,
    execute_reps: int = 5,
) -> MemoryResult:
    """Profile a single configuration. Returns MemoryResult."""
    result = MemoryResult(
        pop_size=pop_size or 0,
        n_eval_points=n_eval_points,
        compute_dtype=compute_dtype,
    )

    try:
        fp = build_fingerprint(n_eval_points, compute_dtype, months, fees)

        with redirect_stdout(io.StringIO()):
            setup = setup_cmaes_computation(fp, pop_size=pop_size, root=root)

        (batched_obj, unravel_fn, fixed_start_indexes,
         flat_x0, n_flat, lam, cma_params) = setup

        result.pop_size = lam
        result.n_flat_params = n_flat

        # Clear JIT cache to get independent compilation
        clear_caches()
        gc.collect()

        compiled, eval_fn, sample_pop, compile_time = compile_cmaes_eval(
            batched_obj, unravel_fn, fixed_start_indexes,
            flat_x0, n_flat, lam,
        )

        result.compile_time_s = compile_time

        stats = extract_stats(compiled)
        result.temp_bytes = stats.get("temp_bytes", 0)
        result.argument_bytes = stats.get("argument_bytes", 0)
        result.output_bytes = stats.get("output_bytes", 0)
        result.flops = stats.get("flops", 0)
        result.transcendentals = stats.get("transcendentals", 0)

        if "error" in stats:
            result.error = stats["error"]

        eval_gflop = result.flops / 1e9
        print(f"  [eval_population] temp={result.temp_mb:.1f} MB, "
              f"flops={eval_gflop:.2f} GFLOP, "
              f"pop={lam}, n_flat={n_flat}  ({compute_dtype})")

        # Execution timing
        if execute and not result.error:
            print(f"  [executing] {execute_reps} reps eval + {execute_reps} full generations ...")
            result.eval_wall_ms, result.eval_gflops, result.gen_wall_ms = (
                time_execution(
                    compiled, eval_fn, sample_pop, flat_x0,
                    cma_params, lam, n_flat,
                    result.flops, reps=execute_reps,
                )
            )
            print(f"  [eval] {result.eval_wall_ms:.1f} ms/call, "
                  f"{result.eval_gflops:.2f} GFLOP/s")
            print(f"  [gen]  {result.gen_wall_ms:.1f} ms/gen "
                  f"(ask + eval + tell, pop={lam})")

    except Exception as e:
        result.error = str(e)[:300]
        import traceback
        traceback.print_exc()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Profile CMA-ES memory: float32 vs float64 via XLA compile-time analysis"
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep population sizes")
    parser.add_argument("--min-pop", type=int, default=None,
                        help="Min population size for sweep (default: auto from dimension)")
    parser.add_argument("--max-pop", type=int, default=32)
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Population size (default: auto from dimension)")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="n_evaluation_points (default: 20)")
    parser.add_argument("--months", type=int, default=12,
                        help="Training window in months (default: 12)")
    parser.add_argument("--fees", type=float, default=0.0,
                        help="Pool fees (0.0 = analytical, >0 = scan reserves)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually run the compiled computation and measure wall-clock time")
    parser.add_argument("--execute-reps", type=int, default=5,
                        help="Number of reps for timing (default: 5)")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    w = 82 + (32 if args.execute else 0)
    print(f"{'=' * w}")
    print(f"  CMA-ES Dtype Comparison — XLA Memory Analysis"
          + (" + Execution Timing" if args.execute else ""))
    print(f"{'=' * w}")
    print(f"  JAX:        {jax.__version__}")
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  Method:     compiled.memory_analysis() — XLA's planned allocation")
    if args.execute:
        print(f"  Execution:  wall-clock timing with block_until_ready ({args.execute_reps} reps)")
    print(f"  n_eval:     {args.n_eval}")
    print(f"  pop_size:   {args.pop_size or 'auto'}")
    print(f"  months:     {args.months}")
    print(f"  fees:       {args.fees}")
    if args.root:
        print(f"  data root:  {args.root}")
    print(f"{'=' * w}")

    results = []

    if args.sweep:
        for dtype in ["float64", "float32"]:
            print(f"\n--- Sweep: {dtype} ---")
            print_header(execute=args.execute)

            pop = args.min_pop
            while True:
                actual_pop = pop  # None on first pass = auto
                r = profile_config(
                    pop_size=actual_pop,
                    n_eval_points=args.n_eval,
                    compute_dtype=dtype,
                    months=args.months,
                    fees=args.fees,
                    root=args.root,
                    execute=args.execute,
                    execute_reps=args.execute_reps,
                )
                results.append(r)
                print_row(r, execute=args.execute)

                if r.error:
                    break

                if pop is None:
                    # First pass was auto; now start doubling from there
                    pop = r.pop_size * 2
                else:
                    pop *= 2

                if pop > args.max_pop:
                    break

        # Summary
        print(f"\n{'=' * w}")
        print(f"  SWEEP COMPARISON")
        print(f"{'=' * w}")
        f64_results = {r.pop_size: r for r in results
                       if r.compute_dtype == "float64" and not r.error}
        f32_results = {r.pop_size: r for r in results
                       if r.compute_dtype == "float32" and not r.error}
        common = sorted(set(f64_results) & set(f32_results))
        if common:
            hdr = (f"  {'pop':>5} {'temp_f64_MB':>12} {'temp_f32_MB':>12} "
                   f"{'mem_reduce':>10} {'flop_ratio':>10}")
            if args.execute:
                hdr += f" {'eval_f64':>10} {'eval_f32':>10} {'speedup':>10}"
                hdr += f" {'gen_f64':>10} {'gen_f32':>10} {'speedup':>10}"
            print(f"\n{hdr}")
            print(f"  {'-'*(len(hdr) - 2)}")
            for p in common:
                r64, r32 = f64_results[p], f32_results[p]
                t64, t32 = r64.temp_mb, r32.temp_mb
                pct = (1 - t32 / t64) * 100 if t64 > 0 else 0
                flop_r = r32.flops / r64.flops if r64.flops > 0 else 0
                row = (f"  {p:>5} {t64:>12.1f} {t32:>12.1f} "
                       f"{pct:>+9.1f}% {flop_r:>10.2f}x")
                if args.execute:
                    w64, w32 = r64.eval_wall_ms, r32.eval_wall_ms
                    eval_su = w64 / w32 if w32 > 0 else 0
                    row += f" {w64:>9.1f}ms {w32:>9.1f}ms {eval_su:>9.1f}x"
                    g64, g32 = r64.gen_wall_ms, r32.gen_wall_ms
                    gen_su = g64 / g32 if g32 > 0 else 0
                    row += f" {g64:>8.1f}ms {g32:>8.1f}ms {gen_su:>9.1f}x"
                print(row)

    else:
        pop_label = args.pop_size or "auto"
        print(f"\n--- Comparison at pop_size={pop_label} ---")
        print_header(execute=args.execute)

        for dtype in ["float64", "float32"]:
            r = profile_config(
                pop_size=args.pop_size,
                n_eval_points=args.n_eval,
                compute_dtype=dtype,
                months=args.months,
                fees=args.fees,
                root=args.root,
                execute=args.execute,
                execute_reps=args.execute_reps,
            )
            results.append(r)
            print_row(r, execute=args.execute)

        print_comparison(results)

    if args.json:
        out = []
        for r in results:
            d = {
                "pop_size": r.pop_size,
                "n_eval_points": r.n_eval_points,
                "n_flat_params": r.n_flat_params,
                "compute_dtype": r.compute_dtype,
                "temp_bytes": r.temp_bytes,
                "temp_mb": r.temp_mb,
                "argument_bytes": r.argument_bytes,
                "argument_mb": r.argument_mb,
                "output_bytes": r.output_bytes,
                "flops": r.flops,
                "transcendentals": r.transcendentals,
                "compile_time_s": r.compile_time_s,
                "error": r.error,
            }
            if args.execute:
                d["eval_wall_ms"] = r.eval_wall_ms
                d["eval_gflops"] = r.eval_gflops
                d["gen_wall_ms"] = r.gen_wall_ms
            out.append(d)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
