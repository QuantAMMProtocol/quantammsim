#!/usr/bin/env python3
"""
BFGS gradient checkpointing memory profiler.

Measures GPU memory, power draw, and utilisation during BFGS optimisation
with and without jax.checkpoint. Designed to validate that checkpointing
trades compute for memory and to find the new parallelism ceiling.

Approach:
  - Queries JAX's internal memory_stats() for peak_bytes_in_use (actual
    allocation peaks inside the memory pool — not the pool size itself)
  - Background thread polls nvidia-smi for power/utilisation
  - Each trial: clear caches → run BFGS for a few iterations → record stats
  - Sweep n_parameter_sets with checkpoint on/off to map the frontier

Note on memory measurement:
  JAX pre-allocates a GPU memory pool (typically 75%+ of VRAM), so nvidia-smi
  always shows ~the same number regardless of actual usage. We use two
  complementary approaches:

  1. JAX memory_stats()["peak_bytes_in_use"] — actual peak within the pool.
     Available without any env vars. This is the primary metric.

  2. --no-pool mode (XLA_PYTHON_CLIENT_ALLOCATOR=platform) — disables JAX's
     pool allocator so nvidia-smi shows true allocation. Slower but lets you
     see real nvidia-smi numbers. Must be set BEFORE jax import, so the script
     re-execs itself with the env var.

Usage (on GPU box):
    # Quick comparison: checkpoint on vs off at default size
    python scripts/profile_bfgs_memory.py

    # Sweep n_parameter_sets to find the OOM ceiling
    python scripts/profile_bfgs_memory.py --sweep

    # Custom sweep range
    python scripts/profile_bfgs_memory.py --sweep --min-sets 1 --max-sets 32

    # Disable JAX memory pool for accurate nvidia-smi readings
    python scripts/profile_bfgs_memory.py --no-pool

    # Longer window (more memory pressure from larger arrays)
    python scripts/profile_bfgs_memory.py --months 6

    # Use data from a non-default root
    python scripts/profile_bfgs_memory.py --root /path/to/data
"""
from __future__ import annotations

import sys
import os

# ── --no-pool handling: must set env var BEFORE importing jax ─────────────────
# Parse just this flag early, re-exec if needed.
if "--no-pool" in sys.argv and "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import time
import argparse
import gc
import json
import subprocess
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp


# ── JAX memory stats ─────────────────────────────────────────────────────────

def get_jax_memory_stats() -> dict:
    """Query JAX's internal memory tracking for the first GPU device.

    Returns dict with keys like peak_bytes_in_use, bytes_in_use, etc.
    Returns empty dict on CPU or if stats are unavailable.
    """
    try:
        device = jax.local_devices()[0]
        stats = device.memory_stats()
        if stats is None:
            return {}
        return stats
    except Exception:
        return {}


def reset_jax_peak_memory():
    """Clear JAX caches and force GC to get a clean baseline.

    Note: JAX doesn't expose a "reset peak counter" API. The best we can do
    is clear caches + GC so that the peak from here forward is meaningful.
    We record baseline bytes_in_use so we can report delta.
    """
    jax.clear_caches()
    gc.collect()
    # Force a sync to ensure all pending frees are processed
    jnp.zeros(1).block_until_ready()
    return get_jax_memory_stats()


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
    # nvidia-smi stats (pool-level; only meaningful with --no-pool)
    smi_peak_memory_mb: float = 0.0
    smi_min_memory_mb: float = 0.0
    memory_total_mb: float = 0.0
    mean_power_w: float = 0.0
    peak_power_w: float = 0.0
    mean_utilisation_pct: float = 0.0
    peak_utilisation_pct: float = 0.0
    n_smi_samples: int = 0
    # JAX-internal stats (actual allocations within the pool)
    jax_peak_bytes: int = 0
    jax_baseline_bytes: int = 0
    jax_peak_delta_mb: float = 0.0


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
            smi_peak_memory_mb=max(mems),
            smi_min_memory_mb=min(mems),
            memory_total_mb=self._snapshots[0].memory_total_mb,
            mean_power_w=sum(pows) / len(pows),
            peak_power_w=max(pows),
            mean_utilisation_pct=sum(utils) / len(utils),
            peak_utilisation_pct=max(utils),
            n_smi_samples=len(self._snapshots),
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

    # Reset memory and get baseline
    baseline_stats = reset_jax_peak_memory()
    baseline_bytes = baseline_stats.get("bytes_in_use", 0)

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

    # Read JAX peak memory (cumulative peak since process start, unfortunately)
    post_stats = get_jax_memory_stats()
    peak_bytes = post_stats.get("peak_bytes_in_use", 0)
    result.gpu.jax_peak_bytes = peak_bytes
    result.gpu.jax_baseline_bytes = baseline_bytes
    result.gpu.jax_peak_delta_mb = (peak_bytes - baseline_bytes) / (1024 * 1024)

    # Clear after run
    clear_caches()
    gc.collect()

    return result


# ── Display ───────────────────────────────────────────────────────────────────

NO_POOL_MODE = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") == "platform"


def print_result_row(r: Optional[TrialResult] = None, header: bool = False):
    """Print one row of results."""
    if header:
        mem_label = "smi_pk" if NO_POOL_MODE else "jax_pk"
        print(f"{'ckpt':>5} {'n_sets':>6} {'n_eval':>6} "
              f"{mem_label + '_MB':>10} "
              f"{'mean_W':>7} {'peak_W':>7} "
              f"{'mean_%':>7} {'peak_%':>7} {'time_s':>7} {'status':>8}")
        print("-" * 84)
        return

    ckpt = "ON" if r.gradient_checkpointing else "OFF"
    # Use nvidia-smi peak in no-pool mode, JAX peak otherwise
    if NO_POOL_MODE:
        mem_mb = r.gpu.smi_peak_memory_mb
    else:
        mem_mb = r.gpu.jax_peak_bytes / (1024 * 1024) if r.gpu.jax_peak_bytes else 0

    if r.success:
        print(f"{ckpt:>5} {r.n_parameter_sets:>6} {r.n_eval_points:>6} "
              f"{mem_mb:>10.0f} {r.gpu.mean_power_w:>7.1f} "
              f"{r.gpu.peak_power_w:>7.1f} {r.gpu.mean_utilisation_pct:>7.1f} "
              f"{r.gpu.peak_utilisation_pct:>7.1f} {r.wall_time_s:>7.1f} {'OK':>8}")
    else:
        print(f"{ckpt:>5} {r.n_parameter_sets:>6} {r.n_eval_points:>6} "
              f"{mem_mb:>10.0f} {'':>7} {'':>7} "
              f"{'':>7} {'':>7} {r.wall_time_s:>7.1f} {r.error:>8}")


def print_comparison(results: List[TrialResult]):
    """Print side-by-side comparison of checkpoint on vs off."""
    on = [r for r in results if r.gradient_checkpointing]
    off = [r for r in results if not r.gradient_checkpointing]

    if not (on and off and on[0].success and off[0].success):
        return

    # Use JAX peak_bytes_in_use as primary metric
    mem_on = on[0].gpu.jax_peak_bytes / (1024 * 1024)
    mem_off = off[0].gpu.jax_peak_bytes / (1024 * 1024)

    if mem_off > 0 and mem_on > 0:
        reduction = (1 - mem_on / mem_off) * 100
        print(f"\n  JAX peak memory:  {mem_off:.0f} MB → {mem_on:.0f} MB "
              f"({reduction:+.1f}%)")
    elif mem_off == 0 and mem_on == 0:
        print(f"\n  JAX memory_stats not available (CPU backend?)")

    if NO_POOL_MODE:
        smi_on = on[0].gpu.smi_peak_memory_mb
        smi_off = off[0].gpu.smi_peak_memory_mb
        if smi_off > 0:
            reduction = (1 - smi_on / smi_off) * 100
            print(f"  nvidia-smi peak:  {smi_off:.0f} MB → {smi_on:.0f} MB "
                  f"({reduction:+.1f}%)")

    time_on = on[0].wall_time_s
    time_off = off[0].wall_time_s
    if time_off > 0:
        slowdown = (time_on / time_off - 1) * 100
        print(f"  Wall time:        {time_off:.1f}s → {time_on:.1f}s "
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
    parser.add_argument("--no-pool", action="store_true",
                        help="Disable JAX memory pool (XLA_PYTHON_CLIENT_ALLOCATOR=platform). "
                             "Slower but nvidia-smi shows true allocations.")
    parser.add_argument("--root", type=str, default=None,
                        help="Data root directory")
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    # Setup
    gpu_monitor = GpuMonitor()
    has_smi = gpu_monitor.available
    has_jax_stats = bool(get_jax_memory_stats())

    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default (pool)")

    print(f"{'=' * 84}")
    print(f"  BFGS Gradient Checkpointing Memory Profiler")
    print(f"{'=' * 84}")
    print(f"  JAX backend:    {jax.default_backend()}")
    print(f"  JAX devices:    {jax.devices()}")
    print(f"  Allocator:      {allocator}")
    print(f"  JAX mem stats:  {'available' if has_jax_stats else 'NOT AVAILABLE'}")
    print(f"  nvidia-smi:     {'available' if has_smi else 'NOT FOUND'}")
    print(f"  n_eval_points:  {args.n_eval}")
    print(f"  maxiter:        {args.maxiter}")
    print(f"  months:         {args.months}")
    if args.root:
        print(f"  data root:      {args.root}")
    print(f"{'=' * 84}")

    if not has_jax_stats and not NO_POOL_MODE:
        print("\n  NOTE: JAX memory_stats not available. For accurate memory measurement,")
        print("  use --no-pool to disable JAX's memory pool allocator.\n")

    results = []

    if args.sweep:
        for ckpt in [False, True]:
            label = "checkpoint ON" if ckpt else "checkpoint OFF"
            print(f"\n--- Sweep: {label} ---")
            print_result_row(header=True)

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

                n *= 2

            successes = [r.n_parameter_sets for r in results
                         if r.gradient_checkpointing == ckpt and r.success]
            if successes:
                print(f"  → Max successful: n_parameter_sets={max(successes)}")

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
        print(f"\n{'=' * 84}")
        print(f"  SWEEP SUMMARY")
        print(f"{'=' * 84}")
        print(f"  Max n_parameter_sets (checkpoint OFF): {off_max}")
        print(f"  Max n_parameter_sets (checkpoint ON):  {on_max}")
        if off_max > 0:
            print(f"  Parallelism gain: {on_max / off_max:.1f}x")
        print(f"{'=' * 84}")

    else:
        print(f"\n--- Comparison at n_parameter_sets={args.n_sets} ---")
        print_result_row(header=True)

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
                "smi_peak_memory_mb": r.gpu.smi_peak_memory_mb,
                "memory_total_mb": r.gpu.memory_total_mb,
                "jax_peak_bytes": r.gpu.jax_peak_bytes,
                "jax_baseline_bytes": r.gpu.jax_baseline_bytes,
                "jax_peak_delta_mb": r.gpu.jax_peak_delta_mb,
                "mean_power_w": r.gpu.mean_power_w,
                "peak_power_w": r.gpu.peak_power_w,
                "mean_utilisation_pct": r.gpu.mean_utilisation_pct,
                "peak_utilisation_pct": r.gpu.peak_utilisation_pct,
                "n_smi_samples": r.gpu.n_smi_samples,
                "allocator": allocator,
            }
            out.append(d)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
