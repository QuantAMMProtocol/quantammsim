#!/usr/bin/env python3
"""
BFGS gradient checkpointing memory profiler.

Measures GPU memory, power draw, and utilisation during BFGS optimisation
with and without jax.checkpoint. Designed to validate that checkpointing
trades compute for memory and to find the new parallelism ceiling.

Approach:
  - Background thread polls nvidia-smi at ~200ms intervals
  - Each trial: clear caches → run BFGS for a few iterations → record peak stats
  - Sweep n_parameter_sets with checkpoint on/off to map the frontier

Usage (on GPU box):
    # Quick comparison: checkpoint on vs off at default size
    python scripts/profile_bfgs_memory.py

    # Sweep n_parameter_sets to find the OOM ceiling
    python scripts/profile_bfgs_memory.py --sweep

    # Custom sweep range
    python scripts/profile_bfgs_memory.py --sweep --min-sets 1 --max-sets 32

    # Longer window (more memory pressure from larger arrays)
    python scripts/profile_bfgs_memory.py --months 6

    # Use data from a non-default root
    python scripts/profile_bfgs_memory.py --root /path/to/data
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import gc
import json
import subprocess
import threading
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp


# ── nvidia-smi poller ─────────────────────────────────────────────────────────

@dataclass
class GpuSnapshot:
    timestamp: float
    memory_used_mb: float
    memory_total_mb: float
    power_draw_w: float
    utilisation_pct: float  # "gpu utilisation" (SM activity)


@dataclass
class GpuStats:
    """Aggregated stats from a monitoring window."""
    peak_memory_mb: float = 0.0
    memory_total_mb: float = 0.0
    mean_power_w: float = 0.0
    peak_power_w: float = 0.0
    mean_utilisation_pct: float = 0.0
    peak_utilisation_pct: float = 0.0
    n_samples: int = 0
    snapshots: List[GpuSnapshot] = field(default_factory=list)


def query_nvidia_smi() -> Optional[GpuSnapshot]:
    """Single nvidia-smi query. Returns None if nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        # Parse first GPU line
        parts = result.stdout.strip().split("\n")[0].split(",")
        return GpuSnapshot(
            timestamp=time.monotonic(),
            memory_used_mb=float(parts[0].strip()),
            memory_total_mb=float(parts[1].strip()),
            power_draw_w=float(parts[2].strip()),
            utilisation_pct=float(parts[3].strip()),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


class GpuMonitor:
    """Background thread that polls nvidia-smi and records snapshots."""

    def __init__(self, poll_interval_s: float = 0.2):
        self.poll_interval = poll_interval_s
        self._snapshots: List[GpuSnapshot] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        return query_nvidia_smi() is not None

    def start(self):
        self._snapshots.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GpuStats:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self._aggregate()

    def _poll_loop(self):
        while not self._stop.is_set():
            snap = query_nvidia_smi()
            if snap:
                self._snapshots.append(snap)
            self._stop.wait(self.poll_interval)

    def _aggregate(self) -> GpuStats:
        if not self._snapshots:
            return GpuStats()
        mems = [s.memory_used_mb for s in self._snapshots]
        pows = [s.power_draw_w for s in self._snapshots]
        utils = [s.utilisation_pct for s in self._snapshots]
        return GpuStats(
            peak_memory_mb=max(mems),
            memory_total_mb=self._snapshots[0].memory_total_mb,
            mean_power_w=sum(pows) / len(pows),
            peak_power_w=max(pows),
            mean_utilisation_pct=sum(utils) / len(utils),
            peak_utilisation_pct=max(utils),
            n_samples=len(self._snapshots),
            snapshots=self._snapshots,
        )


# ── BFGS run config ──────────────────────────────────────────────────────────

def make_bfgs_fingerprint(
    n_parameter_sets: int = 2,
    n_eval_points: int = 5,
    maxiter: int = 3,
    gradient_checkpointing: bool = True,
    months: int = 3,
):
    """Create a BFGS fingerprint sized for profiling.

    Uses fees=0 (analytical cumprod path on GPU) and mean_reversion_channel
    to match the real experiment config. Window length controls array sizes
    and hence memory pressure.
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
    from quantammsim.core_simulator.param_utils import recursive_default_set

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
        "fees": 0.0,
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
                "gradient_checkpointing": gradient_checkpointing,
            },
        },
    }

    recursive_default_set(fp, run_fingerprint_defaults)
    return fp


# ── Trial runner ──────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    n_parameter_sets: int
    n_eval_points: int
    gradient_checkpointing: bool
    success: bool
    wall_time_s: float = 0.0
    gpu: GpuStats = field(default_factory=GpuStats)
    error: str = ""


def run_trial(
    n_parameter_sets: int,
    n_eval_points: int,
    gradient_checkpointing: bool,
    maxiter: int,
    months: int,
    gpu_monitor: GpuMonitor,
    root: Optional[str] = None,
) -> TrialResult:
    """Run a single BFGS trial with GPU monitoring."""
    from jax import clear_caches
    from quantammsim.runners.jax_runners import train_on_historic_data

    fp = make_bfgs_fingerprint(
        n_parameter_sets=n_parameter_sets,
        n_eval_points=n_eval_points,
        maxiter=maxiter,
        gradient_checkpointing=gradient_checkpointing,
        months=months,
    )

    # Clear before run
    clear_caches()
    gc.collect()

    result = TrialResult(
        n_parameter_sets=n_parameter_sets,
        n_eval_points=n_eval_points,
        gradient_checkpointing=gradient_checkpointing,
        success=False,
    )

    gpu_monitor.start()
    try:
        jax.effects_barrier()
        t0 = time.perf_counter()

        train_on_historic_data(
            fp,
            verbose=False,
            force_init=True,
            return_training_metadata=True,
            root=root,
        )

        jax.effects_barrier()
        result.wall_time_s = time.perf_counter() - t0
        result.success = True

    except Exception as e:
        result.wall_time_s = time.perf_counter() - t0
        error_str = str(e).lower()
        if "resource" in error_str or "memory" in error_str or "oom" in error_str:
            result.error = "OOM"
        else:
            result.error = str(e)[:200]
    finally:
        result.gpu = gpu_monitor.stop()

    # Clear after run
    clear_caches()
    gc.collect()

    return result


# ── Display ───────────────────────────────────────────────────────────────────

def print_result_row(r: Optional[TrialResult] = None, header: bool = False):
    """Print one row of results."""
    if header:
        print(f"{'ckpt':>5} {'n_sets':>6} {'n_eval':>6} "
              f"{'peak_MB':>8} {'mean_W':>7} {'peak_W':>7} "
              f"{'mean_%':>7} {'peak_%':>7} {'time_s':>7} {'status':>8}")
        print("-" * 80)
        return

    ckpt = "ON" if r.gradient_checkpointing else "OFF"
    if r.success:
        print(f"{ckpt:>5} {r.n_parameter_sets:>6} {r.n_eval_points:>6} "
              f"{r.gpu.peak_memory_mb:>8.0f} {r.gpu.mean_power_w:>7.1f} "
              f"{r.gpu.peak_power_w:>7.1f} {r.gpu.mean_utilisation_pct:>7.1f} "
              f"{r.gpu.peak_utilisation_pct:>7.1f} {r.wall_time_s:>7.1f} {'OK':>8}")
    else:
        print(f"{ckpt:>5} {r.n_parameter_sets:>6} {r.n_eval_points:>6} "
              f"{r.gpu.peak_memory_mb:>8.0f} {'':>7} {'':>7} "
              f"{'':>7} {'':>7} {r.wall_time_s:>7.1f} {r.error:>8}")


def print_comparison(results: List[TrialResult]):
    """Print side-by-side comparison of checkpoint on vs off."""
    on = [r for r in results if r.gradient_checkpointing]
    off = [r for r in results if not r.gradient_checkpointing]

    if on and off and on[0].success and off[0].success:
        mem_on = on[0].gpu.peak_memory_mb
        mem_off = off[0].gpu.peak_memory_mb
        if mem_off > 0:
            reduction = (1 - mem_on / mem_off) * 100
            print(f"\n  Memory reduction: {mem_off:.0f} MB → {mem_on:.0f} MB "
                  f"({reduction:+.1f}%)")

        time_on = on[0].wall_time_s
        time_off = off[0].wall_time_s
        if time_off > 0:
            slowdown = (time_on / time_off - 1) * 100
            print(f"  Time change:      {time_off:.1f}s → {time_on:.1f}s "
                  f"({slowdown:+.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Profile BFGS GPU memory with/without gradient checkpointing"
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep n_parameter_sets to find OOM ceiling")
    parser.add_argument("--min-sets", type=int, default=1,
                        help="Min n_parameter_sets for sweep (default: 1)")
    parser.add_argument("--max-sets", type=int, default=32,
                        help="Max n_parameter_sets for sweep (default: 32)")
    parser.add_argument("--n-sets", type=int, default=4,
                        help="n_parameter_sets for single comparison (default: 4)")
    parser.add_argument("--n-eval", type=int, default=5,
                        help="n_evaluation_points (default: 5)")
    parser.add_argument("--maxiter", type=int, default=3,
                        help="BFGS iterations per trial (default: 3)")
    parser.add_argument("--months", type=int, default=3,
                        help="Training window in months (default: 3)")
    parser.add_argument("--root", type=str, default=None,
                        help="Data root directory")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Setup
    gpu_monitor = GpuMonitor()
    has_gpu = gpu_monitor.available

    print(f"{'=' * 80}")
    print(f"  BFGS Gradient Checkpointing Memory Profiler")
    print(f"{'=' * 80}")
    print(f"  JAX backend:   {jax.default_backend()}")
    print(f"  JAX devices:   {jax.devices()}")
    print(f"  nvidia-smi:    {'available' if has_gpu else 'NOT FOUND (no GPU stats)'}")
    print(f"  n_eval_points: {args.n_eval}")
    print(f"  maxiter:       {args.maxiter}")
    print(f"  months:        {args.months}")
    if args.root:
        print(f"  data root:     {args.root}")
    print(f"{'=' * 80}")

    if not has_gpu:
        print("\n  WARNING: nvidia-smi not available. GPU memory/power stats will be zeros.")
        print("  The script will still run and measure wall-clock time.\n")

    results = []

    if args.sweep:
        # Sweep n_parameter_sets with checkpoint on, find OOM ceiling
        # Then do the same with checkpoint off for comparison
        for ckpt in [False, True]:
            label = "checkpoint ON" if ckpt else "checkpoint OFF"
            print(f"\n--- Sweep: {label} ---")
            print_result_row(None, header=True)

            n = args.min_sets
            while n <= args.max_sets:
                r = run_trial(
                    n_parameter_sets=n,
                    n_eval_points=args.n_eval,
                    gradient_checkpointing=ckpt,
                    maxiter=args.maxiter,
                    months=args.months,
                    gpu_monitor=gpu_monitor,
                    root=args.root,
                )
                results.append(r)
                print_result_row(r)

                if not r.success:
                    print(f"  → OOM at n_parameter_sets={n}, stopping sweep")
                    break

                # Double until we hit ceiling, then we've bracketed it
                n *= 2

            # Find max successful
            successes = [r.n_parameter_sets for r in results
                         if r.gradient_checkpointing == ckpt and r.success]
            if successes:
                print(f"  → Max successful: n_parameter_sets={max(successes)}")

        # Summary
        on_max = max(
            (r.n_parameter_sets for r in results
             if r.gradient_checkpointing and r.success),
            default=0,
        )
        off_max = max(
            (r.n_parameter_sets for r in results
             if not r.gradient_checkpointing and r.success),
            default=0,
        )
        print(f"\n{'=' * 80}")
        print(f"  SWEEP SUMMARY")
        print(f"{'=' * 80}")
        print(f"  Max n_parameter_sets (checkpoint OFF): {off_max}")
        print(f"  Max n_parameter_sets (checkpoint ON):  {on_max}")
        if off_max > 0:
            print(f"  Parallelism gain: {on_max / off_max:.1f}×")
        print(f"{'=' * 80}")

    else:
        # Single comparison: same n_parameter_sets, checkpoint on vs off
        print(f"\n--- Comparison at n_parameter_sets={args.n_sets} ---")
        print_result_row(None, header=True)

        for ckpt in [False, True]:
            r = run_trial(
                n_parameter_sets=args.n_sets,
                n_eval_points=args.n_eval,
                gradient_checkpointing=ckpt,
                maxiter=args.maxiter,
                months=args.months,
                gpu_monitor=gpu_monitor,
                root=args.root,
            )
            results.append(r)
            print_result_row(r)

        print_comparison(results)

    # Save JSON if requested
    if args.json:
        out = []
        for r in results:
            d = {
                "n_parameter_sets": r.n_parameter_sets,
                "n_eval_points": r.n_eval_points,
                "gradient_checkpointing": r.gradient_checkpointing,
                "success": r.success,
                "wall_time_s": r.wall_time_s,
                "error": r.error,
                "peak_memory_mb": r.gpu.peak_memory_mb,
                "memory_total_mb": r.gpu.memory_total_mb,
                "mean_power_w": r.gpu.mean_power_w,
                "peak_power_w": r.gpu.peak_power_w,
                "mean_utilisation_pct": r.gpu.mean_utilisation_pct,
                "peak_utilisation_pct": r.gpu.peak_utilisation_pct,
                "n_gpu_samples": r.gpu.n_samples,
            }
            out.append(d)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
