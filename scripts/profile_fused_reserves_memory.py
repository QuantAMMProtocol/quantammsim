#!/usr/bin/env python3
"""
Fused reserves memory profiler.

Uses XLA's compiled memory_analysis() to measure the temp memory XLA allocates
for the forward pass with use_fused_reserves=True vs False.

The fused path avoids materialising full (T_fine, n_assets) weight and reserve
arrays by computing per-chunk ratio products inline.  This script quantifies
the memory saving and optional wall-clock speedup on GPU.

We compile value_and_grad(batched_objective) — the inner training step that
dominates both BFGS and CMA-ES memory.

Usage:
    # Quick comparison (compile-time only, 6-month window)
    python scripts/profile_fused_reserves_memory.py

    # With wall-clock execution timing
    python scripts/profile_fused_reserves_memory.py --execute

    # Sweep training window length
    python scripts/profile_fused_reserves_memory.py --sweep --execute

    # Different n_parameter_sets (vmapped over param sets)
    python scripts/profile_fused_reserves_memory.py --n-sets 8 --execute

    # Save results
    python scripts/profile_fused_reserves_memory.py --sweep --execute --json results.json
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
from jax import jit, vmap, value_and_grad, clear_caches
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


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class MemoryResult:
    use_fused: bool
    n_parameter_sets: int
    n_eval_points: int
    actual_n_eval: int = 0
    months: int = 0
    bout_length: int = 0
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
    vg_wall_ms: float = 0.0       # median wall-clock per value_and_grad call
    vg_gflops: float = 0.0        # effective GFLOP/s
    error: str = ""

    @property
    def temp_mb(self) -> float:
        return self.temp_bytes / (1024 * 1024)

    @property
    def argument_mb(self) -> float:
        return self.argument_bytes / (1024 * 1024)

    @property
    def fused_label(self) -> str:
        return "fused" if self.use_fused else "full"


# ── Setup ─────────────────────────────────────────────────────────────────────

def build_fingerprint(
    n_parameter_sets: int,
    n_eval_points: int,
    months: int,
    rule: str,
) -> dict:
    start = datetime(2021, 6, 1)
    end_train = start + relativedelta(months=months)
    end_test = end_train + relativedelta(months=1)

    fp = {
        "tokens": ["ETH", "USDC"],
        "rule": rule,
        "startDateString": start.strftime("%Y-%m-%d %H:%M:%S"),
        "endDateString": end_train.strftime("%Y-%m-%d %H:%M:%S"),
        "endTestDateString": end_test.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "initial_pool_value": 1_000_000.0,
        # Fused path requires zero fees
        "fees": 0.0,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "do_arb": True,
        "arb_frequency": 1,
        "minimum_weight": 0.01,
        "max_memory_days": 365,
        # bout_offset must be > 0 so generate_evaluation_points has room
        # for multiple distinct eval windows (available_range = bout_offset)
        "bout_offset": 2 * n_eval_points,
        "return_val": "daily_log_sharpe",
        "optimisation_settings": {
            "method": "bfgs",
            "n_parameter_sets": n_parameter_sets,
            "noise_scale": 0.3,
            "val_fraction": 0.0,
            "bfgs_settings": {
                "maxiter": 3,
                "tol": 1e-6,
                "n_evaluation_points": n_eval_points,
                "compute_dtype": "float32",
            },
        },
    }
    recursive_default_set(fp, run_fingerprint_defaults)
    return fp


def setup_computation(fp, use_fused: bool, root=None):
    """
    Build the batched objective and flatten params, returning all pieces
    needed to compile value_and_grad.
    """
    jax.config.update("jax_enable_x64", False)

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
            "use_fused_reserves": use_fused,
        },
    )

    n_eval_points = fp["optimisation_settings"]["bfgs_settings"]["n_evaluation_points"]

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
        bout_length_window,
    )


def compile_vg(
    partial_training_step,
    params,
    fixed_start_indexes,
    n_parameter_sets: int,
) -> tuple:
    """
    Build and compile value_and_grad(neg_batched_objective).
    Returns (compiled_vg, flat_x0, compile_time_s).
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

    flat_x0, unravel_fn = ravel_pytree(params_single)

    def neg_objective(flat_x):
        p = unravel_fn(flat_x)
        return -batched_obj(p, fixed_start_indexes)

    vg_fn = jit(value_and_grad(neg_objective))

    t0 = time.perf_counter()
    lowered = vg_fn.lower(flat_x0)
    compiled = lowered.compile()
    compile_time = time.perf_counter() - t0

    return compiled, flat_x0, compile_time


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

def time_execution(compiled_vg, flat_x0, flops, reps=5):
    """
    Run the compiled value_and_grad and measure wall-clock time.
    Returns (vg_wall_ms, vg_gflops).
    """
    # Warm up
    out = compiled_vg(flat_x0)
    jax.block_until_ready(out)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = compiled_vg(flat_x0)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    vg_wall_s = float(np.median(times))
    vg_wall_ms = vg_wall_s * 1000
    vg_gflops = (flops / 1e9) / vg_wall_s if vg_wall_s > 0 else 0

    return vg_wall_ms, vg_gflops


# ── Display ───────────────────────────────────────────────────────────────────

def print_header(execute=False):
    hdr = (f"{'mode':>7} {'months':>6} {'n_sets':>6} {'n_eval':>6} {'actual':>6} {'bout':>7} "
           f"{'temp_MB':>10} {'arg_MB':>10} "
           f"{'GFLOP':>10} {'compile_s':>10}")
    if execute:
        hdr += f" {'vg_ms':>10} {'GFLOP/s':>10}"
    hdr += f" {'status':>8}"
    print(hdr)
    print("-" * (90 + (22 if execute else 0)))


def print_row(r: MemoryResult, execute=False):
    if not r.error:
        gflop = r.flops / 1e9 if r.flops else 0
        row = (f"{r.fused_label:>7} {r.months:>6} {r.n_parameter_sets:>6} "
               f"{r.n_eval_points:>6} {r.actual_n_eval:>6} {r.bout_length:>7} "
               f"{r.temp_mb:>10.1f} {r.argument_mb:>10.1f} "
               f"{gflop:>10.2f} {r.compile_time_s:>10.1f}")
        if execute:
            row += f" {r.vg_wall_ms:>10.1f} {r.vg_gflops:>10.2f}"
        row += f" {'OK':>8}"
        print(row)
    else:
        row = (f"{r.fused_label:>7} {r.months:>6} {r.n_parameter_sets:>6} "
               f"{r.n_eval_points:>6} {r.actual_n_eval:>6} {r.bout_length:>7} "
               f"{'':>10} {'':>10} "
               f"{'':>10} {r.compile_time_s:>10.1f}")
        if execute:
            row += f" {'':>10} {'':>10}"
        row += f" {'ERR':>8}"
        print(row)
        print(f"  error: {r.error}")


def print_comparison(r_full: MemoryResult, r_fused: MemoryResult, execute=False):
    if r_full.error or r_fused.error:
        return

    print(f"\n  {'metric':<25} {'full':>12} {'fused':>12} {'delta':>12}")
    print(f"  {'-'*61}")

    # Temp memory
    tf, tu = r_full.temp_mb, r_fused.temp_mb
    if tf > 0:
        delta = (tu / tf - 1) * 100
        print(f"  {'temp memory (MB)':<25} {tf:>12.1f} {tu:>12.1f} {delta:>+11.1f}%")

    # Argument memory
    af, au = r_full.argument_mb, r_fused.argument_mb
    if af > 0:
        delta = (au / af - 1) * 100
        print(f"  {'argument memory (MB)':<25} {af:>12.1f} {au:>12.1f} {delta:>+11.1f}%")

    # FLOPs
    ff, fu = r_full.flops / 1e9, r_fused.flops / 1e9
    if ff > 0:
        delta = (fu / ff - 1) * 100
        print(f"  {'GFLOP':<25} {ff:>12.2f} {fu:>12.2f} {delta:>+11.1f}%")

    # Compile time
    cf, cu = r_full.compile_time_s, r_fused.compile_time_s
    print(f"  {'compile time (s)':<25} {cf:>12.1f} {cu:>12.1f}")

    # Execution timing
    if execute and r_full.vg_wall_ms > 0 and r_fused.vg_wall_ms > 0:
        print()
        wf, wu = r_full.vg_wall_ms, r_fused.vg_wall_ms
        speedup = wf / wu if wu > 0 else 0
        print(f"  {'value_and_grad (ms)':<25} {wf:>12.1f} {wu:>12.1f} {speedup:>11.1f}x")
        gf, gu = r_full.vg_gflops, r_fused.vg_gflops
        print(f"  {'throughput (GFLOP/s)':<25} {gf:>12.2f} {gu:>12.2f}")


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_config(
    use_fused: bool,
    n_parameter_sets: int,
    n_eval_points: int,
    months: int,
    rule: str,
    root: Optional[str],
    execute: bool = False,
    execute_reps: int = 5,
) -> MemoryResult:
    """Profile a single configuration. Returns MemoryResult."""
    result = MemoryResult(
        use_fused=use_fused,
        n_parameter_sets=n_parameter_sets,
        n_eval_points=n_eval_points,
        months=months,
    )

    try:
        fp = build_fingerprint(n_parameter_sets, n_eval_points, months, rule)

        with redirect_stdout(io.StringIO()):
            setup = setup_computation(fp, use_fused=use_fused, root=root)

        (partial_training_step, params, fixed_start_indexes,
         n_sets, bout_length_window) = setup

        result.bout_length = bout_length_window
        result.actual_n_eval = fixed_start_indexes.shape[0]
        actual_n_eval = result.actual_n_eval

        # Clear JIT cache to get independent compilation
        clear_caches()
        gc.collect()

        compiled_vg, flat_x0, compile_time = compile_vg(
            partial_training_step, params, fixed_start_indexes, n_sets,
        )

        result.compile_time_s = compile_time

        stats = extract_stats(compiled_vg)
        result.temp_bytes = stats.get("temp_bytes", 0)
        result.argument_bytes = stats.get("argument_bytes", 0)
        result.output_bytes = stats.get("output_bytes", 0)
        result.flops = stats.get("flops", 0)
        result.transcendentals = stats.get("transcendentals", 0)

        if "error" in stats:
            result.error = stats["error"]

        mode = "fused" if use_fused else "full"
        gflop = result.flops / 1e9
        print(f"  [{mode}] temp={result.temp_mb:.1f} MB, "
              f"flops={gflop:.2f} GFLOP, "
              f"bout={bout_length_window}, "
              f"actual_n_eval={actual_n_eval}")

        # Execution timing
        if execute and not result.error:
            print(f"  [executing] {execute_reps} reps value_and_grad ...")
            result.vg_wall_ms, result.vg_gflops = time_execution(
                compiled_vg, flat_x0, result.flops, reps=execute_reps,
            )
            print(f"  [{mode}] {result.vg_wall_ms:.1f} ms/call, "
                  f"{result.vg_gflops:.2f} GFLOP/s")

    except Exception as e:
        result.error = str(e)[:300]
        import traceback
        traceback.print_exc()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Profile fused vs full-resolution reserve computation via XLA memory analysis"
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep training window length (months)")
    parser.add_argument("--min-months", type=int, default=3)
    parser.add_argument("--max-months", type=int, default=12)
    parser.add_argument("--months", type=int, default=6,
                        help="Training window in months for single comparison (default: 6)")
    parser.add_argument("--n-sets", type=int, default=1,
                        help="n_parameter_sets (default: 1)")
    parser.add_argument("--n-eval", type=int, default=5,
                        help="n_evaluation_points (default: 5)")
    parser.add_argument("--rule", type=str, default="momentum",
                        help="Pool rule (default: momentum)")
    parser.add_argument("--execute", action="store_true",
                        help="Run compiled computation and measure wall-clock time")
    parser.add_argument("--execute-reps", type=int, default=5,
                        help="Number of reps for timing (default: 5)")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    w = 83 + (22 if args.execute else 0)
    print(f"{'=' * w}")
    print(f"  Fused Reserves Memory Comparison — XLA Memory Analysis"
          + (" + Execution Timing" if args.execute else ""))
    print(f"{'=' * w}")
    print(f"  JAX:        {jax.__version__}")
    print(f"  Backend:    {jax.default_backend()}")
    print(f"  Method:     compiled.memory_analysis() — XLA's planned allocation")
    if args.execute:
        print(f"  Execution:  wall-clock timing with block_until_ready ({args.execute_reps} reps)")
    print(f"  Rule:       {args.rule}")
    print(f"  n_sets:     {args.n_sets}")
    print(f"  n_eval:     {args.n_eval}")
    if not args.sweep:
        print(f"  months:     {args.months}")
    if args.root:
        print(f"  data root:  {args.root}")
    print(f"{'=' * w}")

    results = []

    if args.sweep:
        month_values = list(range(args.min_months, args.max_months + 1, 3))
        if args.max_months not in month_values:
            month_values.append(args.max_months)

        for months in month_values:
            print(f"\n--- {months} months ---")
            print_header(execute=args.execute)

            r_full = profile_config(
                use_fused=False,
                n_parameter_sets=args.n_sets,
                n_eval_points=args.n_eval,
                months=months,
                rule=args.rule,
                root=args.root,
                execute=args.execute,
                execute_reps=args.execute_reps,
            )
            results.append(r_full)
            print_row(r_full, execute=args.execute)

            r_fused = profile_config(
                use_fused=True,
                n_parameter_sets=args.n_sets,
                n_eval_points=args.n_eval,
                months=months,
                rule=args.rule,
                root=args.root,
                execute=args.execute,
                execute_reps=args.execute_reps,
            )
            results.append(r_fused)
            print_row(r_fused, execute=args.execute)

            print_comparison(r_full, r_fused, execute=args.execute)

        # Sweep summary table
        print(f"\n{'=' * w}")
        print(f"  SWEEP SUMMARY")
        print(f"{'=' * w}")
        hdr = (f"  {'months':>6} {'bout':>7} "
               f"{'temp_full':>10} {'temp_fused':>10} {'saving':>10}")
        if args.execute:
            hdr += f" {'ms_full':>10} {'ms_fused':>10} {'speedup':>10}"
        print(f"\n{hdr}")
        print(f"  {'-'*(len(hdr) - 2)}")
        for i in range(0, len(results), 2):
            rf, ru = results[i], results[i + 1]
            if rf.error or ru.error:
                continue
            tf, tu = rf.temp_mb, ru.temp_mb
            saving = (1 - tu / tf) * 100 if tf > 0 else 0
            row = (f"  {rf.months:>6} {rf.bout_length:>7} "
                   f"{tf:>10.1f} {tu:>10.1f} {saving:>+9.1f}%")
            if args.execute:
                wf, wu = rf.vg_wall_ms, ru.vg_wall_ms
                speedup = wf / wu if wu > 0 else 0
                row += f" {wf:>9.1f}ms {wu:>9.1f}ms {speedup:>9.1f}x"
            print(row)

    else:
        print(f"\n--- Comparison at {args.months} months ---")
        print_header(execute=args.execute)

        r_full = profile_config(
            use_fused=False,
            n_parameter_sets=args.n_sets,
            n_eval_points=args.n_eval,
            months=args.months,
            rule=args.rule,
            root=args.root,
            execute=args.execute,
            execute_reps=args.execute_reps,
        )
        results.append(r_full)
        print_row(r_full, execute=args.execute)

        r_fused = profile_config(
            use_fused=True,
            n_parameter_sets=args.n_sets,
            n_eval_points=args.n_eval,
            months=args.months,
            rule=args.rule,
            root=args.root,
            execute=args.execute,
            execute_reps=args.execute_reps,
        )
        results.append(r_fused)
        print_row(r_fused, execute=args.execute)

        print_comparison(r_full, r_fused, execute=args.execute)

    if args.json:
        out = []
        for r in results:
            d = {
                "use_fused": r.use_fused,
                "n_parameter_sets": r.n_parameter_sets,
                "n_eval_points": r.n_eval_points,
                "months": r.months,
                "bout_length": r.bout_length,
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
                d["vg_wall_ms"] = r.vg_wall_ms
                d["vg_gflops"] = r.vg_gflops
            out.append(d)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
