#!/usr/bin/env python3
"""
BFGS dtype memory profiler.

Uses XLA's compiled memory_analysis() to measure the actual temp memory
XLA allocates for the BFGS computation in float32 vs float64.
Deterministic and accurate — no runtime measurement noise, no nvidia-smi
polling, no subprocess isolation needed.

We compile two things:
  1. value_and_grad(neg_objective) — the inner BFGS step
  2. jit(vmap(solve_single)) — the full vmapped BFGS solve

Usage:
    # Quick comparison: float32 vs float64
    python scripts/profile_bfgs_memory.py

    # Sweep n_parameter_sets
    python scripts/profile_bfgs_memory.py --sweep

    # More eval points / longer window
    python scripts/profile_bfgs_memory.py --n-eval 20 --months 12

    # Save results
    python scripts/profile_bfgs_memory.py --sweep --json results.json
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import json
import gc
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, clear_caches
from jax.flatten_util import ravel_pytree
from jax.scipy.optimize import minimize as jax_minimize
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


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class MemoryResult:
    n_parameter_sets: int
    n_eval_points: int
    compute_dtype: str
    # From compiled.memory_analysis()
    temp_bytes: int = 0
    argument_bytes: int = 0
    output_bytes: int = 0
    # From compiled.cost_analysis()
    flops: int = 0
    transcendentals: int = 0
    # Timing
    compile_time_s: float = 0.0
    error: str = ""

    @property
    def temp_mb(self) -> float:
        return self.temp_bytes / (1024 * 1024)

    @property
    def argument_mb(self) -> float:
        return self.argument_bytes / (1024 * 1024)


# ── Setup ─────────────────────────────────────────────────────────────────────

def build_fingerprint(
    n_parameter_sets: int,
    n_eval_points: int,
    compute_dtype: str,
    maxiter: int,
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
            "method": "bfgs",
            "n_parameter_sets": n_parameter_sets,
            "noise_scale": 0.3,
            "val_fraction": 0.0,
            "bfgs_settings": {
                "maxiter": maxiter,
                "tol": 1e-6,
                "n_evaluation_points": n_eval_points,
                "compute_dtype": compute_dtype,
            },
        },
    }
    recursive_default_set(fp, run_fingerprint_defaults)
    return fp


def setup_bfgs_computation(fp, root=None):
    """
    Replicate the BFGS setup from jax_runners.train_on_historic_data,
    returning all the pieces needed to build the compiled solve.
    """
    unique_tokens = get_unique_tokens(fp)
    n_tokens = len(unique_tokens)
    n_assets = n_tokens
    all_sig_variations = get_sig_variations(n_assets)
    n_parameter_sets = fp["optimisation_settings"]["n_parameter_sets"]

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

    bfgs_settings = fp["optimisation_settings"]["bfgs_settings"]
    compute_dtype_str = bfgs_settings.get("compute_dtype", "float64")
    compute_dtype = jnp.float32 if compute_dtype_str == "float32" else jnp.float64
    n_eval_points = bfgs_settings["n_evaluation_points"]
    maxiter = bfgs_settings["maxiter"]
    tol = bfgs_settings["tol"]

    # Cast prices to compute dtype if needed
    if compute_dtype != jnp.float64:
        prices = data_dict["prices"].astype(compute_dtype)
        partial_training_step = Partial(
            forward_pass,
            prices=prices,
            static_dict=Hashabledict(base_static_dict),
            pool=pool,
        )
    else:
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

    return (
        partial_training_step,
        params,
        fixed_start_indexes,
        n_parameter_sets,
        maxiter,
        tol,
        compute_dtype,
    )


def compile_bfgs(
    partial_training_step,
    params,
    fixed_start_indexes,
    n_parameter_sets: int,
    maxiter: int,
    tol: float,
    compute_dtype,
) -> tuple:
    """
    Build and compile the BFGS computation.
    Returns (compiled_solve, compiled_inner, compile_time_s).
    """
    batched_pts = batched_partial_training_step_factory(partial_training_step)
    batched_obj = batched_objective_factory(batched_pts)

    # Build single-set params for ravel_pytree
    params_single = {}
    for k, v in params.items():
        if k == "subsidary_params":
            params_single[k] = v
        elif hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == n_parameter_sets:
            params_single[k] = v[0]
        else:
            params_single[k] = v

    flat_x0_template, unravel_fn = ravel_pytree(params_single)

    def neg_objective(flat_x):
        p = unravel_fn(flat_x)
        if compute_dtype != jnp.float64:
            p = jax.tree.map(lambda x: x.astype(compute_dtype), p)
        obj = -batched_obj(p, fixed_start_indexes)
        return obj.astype(jnp.float64) if compute_dtype != jnp.float64 else obj

    # Flatten all parameter sets
    all_flat_x0 = []
    for i in range(n_parameter_sets):
        ps = {}
        for k, v in params.items():
            if k == "subsidary_params":
                ps[k] = v
            elif hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == n_parameter_sets:
                ps[k] = v[i]
            else:
                ps[k] = v
        flat_xi, _ = ravel_pytree(ps)
        all_flat_x0.append(flat_xi)
    all_flat_x0 = jnp.stack(all_flat_x0)

    # Compile the inner value_and_grad (one BFGS step)
    inner_fn = jit(value_and_grad(neg_objective))

    # Compile the full vmapped solve
    def solve_single(flat_x0):
        result = jax_minimize(
            neg_objective, flat_x0, method="BFGS",
            options={"maxiter": maxiter}, tol=tol,
        )
        return result.x, result.fun, result.status

    vmapped_solve = jit(vmap(solve_single))

    t0 = time.perf_counter()

    # Lower and compile both
    lowered_inner = inner_fn.lower(all_flat_x0[0])
    compiled_inner = lowered_inner.compile()

    lowered_solve = vmapped_solve.lower(all_flat_x0)
    compiled_solve = lowered_solve.compile()

    compile_time = time.perf_counter() - t0

    return compiled_solve, compiled_inner, compile_time


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


# ── Display ───────────────────────────────────────────────────────────────────

def print_header():
    print(f"{'dtype':>7} {'n_sets':>6} {'n_eval':>6} "
          f"{'temp_MB':>10} {'arg_MB':>10} "
          f"{'GFLOP':>10} {'compile_s':>10} {'status':>8}")
    print("-" * 76)


def print_row(r: MemoryResult):
    if not r.error:
        gflop = r.flops / 1e9 if r.flops else 0
        print(f"{r.compute_dtype:>7} {r.n_parameter_sets:>6} {r.n_eval_points:>6} "
              f"{r.temp_mb:>10.1f} {r.argument_mb:>10.1f} "
              f"{gflop:>10.2f} {r.compile_time_s:>10.1f} {'OK':>8}")
    else:
        print(f"{r.compute_dtype:>7} {r.n_parameter_sets:>6} {r.n_eval_points:>6} "
              f"{'':>10} {'':>10} "
              f"{'':>10} {r.compile_time_s:>10.1f} {'ERR':>8}")
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


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_config(
    n_parameter_sets: int,
    n_eval_points: int,
    compute_dtype: str,
    maxiter: int,
    months: int,
    fees: float,
    root: Optional[str],
) -> MemoryResult:
    """Profile a single configuration. Returns MemoryResult."""
    result = MemoryResult(
        n_parameter_sets=n_parameter_sets,
        n_eval_points=n_eval_points,
        compute_dtype=compute_dtype,
    )

    try:
        fp = build_fingerprint(
            n_parameter_sets, n_eval_points, compute_dtype,
            maxiter, months, fees,
        )
        setup = setup_bfgs_computation(fp, root=root)

        (partial_training_step, params, fixed_start_indexes,
         n_sets, max_it, tol, dtype) = setup

        # Clear JIT cache to get independent compilation
        clear_caches()
        gc.collect()

        compiled_solve, compiled_inner, compile_time = compile_bfgs(
            partial_training_step, params, fixed_start_indexes,
            n_sets, max_it, tol, dtype,
        )

        result.compile_time_s = compile_time

        # Use the full vmapped_solve stats (includes BFGS loop + all inner steps)
        solve_stats = extract_stats(compiled_solve)
        result.temp_bytes = solve_stats.get("temp_bytes", 0)
        result.argument_bytes = solve_stats.get("argument_bytes", 0)
        result.output_bytes = solve_stats.get("output_bytes", 0)
        result.flops = solve_stats.get("flops", 0)
        result.transcendentals = solve_stats.get("transcendentals", 0)

        if "error" in solve_stats:
            result.error = solve_stats["error"]

        # Also print inner (value_and_grad) stats for reference
        inner_stats = extract_stats(compiled_inner)
        inner_temp_mb = inner_stats.get("temp_bytes", 0) / (1024 * 1024)
        inner_flops = inner_stats.get("flops", 0) / 1e9
        print(f"  [inner value_and_grad] temp={inner_temp_mb:.1f} MB, "
              f"flops={inner_flops:.2f} GFLOP  ({compute_dtype})")

    except Exception as e:
        result.error = str(e)[:300]
        import traceback
        traceback.print_exc()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Profile BFGS memory: float32 vs float64 via XLA compile-time analysis"
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep n_parameter_sets")
    parser.add_argument("--min-sets", type=int, default=1)
    parser.add_argument("--max-sets", type=int, default=32)
    parser.add_argument("--n-sets", type=int, default=4,
                        help="n_parameter_sets for single comparison (default: 4)")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="n_evaluation_points (default: 20)")
    parser.add_argument("--maxiter", type=int, default=3,
                        help="BFGS maxiter (default: 3)")
    parser.add_argument("--months", type=int, default=12,
                        help="Training window in months (default: 12)")
    parser.add_argument("--fees", type=float, default=0.0,
                        help="Pool fees (0.0 = analytical, >0 = scan reserves)")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print(f"{'=' * 76}")
    print(f"  BFGS Dtype Comparison — XLA Memory Analysis")
    print(f"{'=' * 76}")
    print(f"  JAX:        {jax.__version__}")
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  Method:     compiled.memory_analysis() — XLA's planned allocation")
    print(f"  n_eval:     {args.n_eval}")
    print(f"  maxiter:    {args.maxiter}")
    print(f"  months:     {args.months}")
    print(f"  fees:       {args.fees}")
    if args.root:
        print(f"  data root:  {args.root}")
    print(f"{'=' * 76}")

    results = []

    if args.sweep:
        for dtype in ["float64", "float32"]:
            print(f"\n--- Sweep: {dtype} ---")
            print_header()

            n = args.min_sets
            while n <= args.max_sets:
                r = profile_config(
                    n_parameter_sets=n,
                    n_eval_points=args.n_eval,
                    compute_dtype=dtype,
                    maxiter=args.maxiter,
                    months=args.months,
                    fees=args.fees,
                    root=args.root,
                )
                results.append(r)
                print_row(r)

                if r.error:
                    break

                n *= 2

        # Summary: compare matching rows
        print(f"\n{'=' * 76}")
        print(f"  SWEEP COMPARISON")
        print(f"{'=' * 76}")
        f64_results = {r.n_parameter_sets: r for r in results
                       if r.compute_dtype == "float64" and not r.error}
        f32_results = {r.n_parameter_sets: r for r in results
                       if r.compute_dtype == "float32" and not r.error}
        common = sorted(set(f64_results) & set(f32_results))
        if common:
            print(f"\n  {'n_sets':>6} {'temp_f64_MB':>12} {'temp_f32_MB':>12} "
                  f"{'reduction':>10} {'flop_ratio':>10}")
            print(f"  {'-'*56}")
            for n in common:
                r64, r32 = f64_results[n], f32_results[n]
                t64, t32 = r64.temp_mb, r32.temp_mb
                pct = (1 - t32 / t64) * 100 if t64 > 0 else 0
                flop_r = r32.flops / r64.flops if r64.flops > 0 else 0
                print(f"  {n:>6} {t64:>12.1f} {t32:>12.1f} "
                      f"{pct:>+9.1f}% {flop_r:>10.2f}x")

    else:
        print(f"\n--- Comparison at n_parameter_sets={args.n_sets} ---")
        print_header()

        for dtype in ["float64", "float32"]:
            r = profile_config(
                n_parameter_sets=args.n_sets,
                n_eval_points=args.n_eval,
                compute_dtype=dtype,
                maxiter=args.maxiter,
                months=args.months,
                fees=args.fees,
                root=args.root,
            )
            results.append(r)
            print_row(r)

        print_comparison(results)

    if args.json:
        out = []
        for r in results:
            out.append({
                "n_parameter_sets": r.n_parameter_sets,
                "n_eval_points": r.n_eval_points,
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
            })
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
