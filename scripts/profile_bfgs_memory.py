#!/usr/bin/env python3
"""
BFGS gradient checkpointing memory profiler.

Measures GPU memory, power draw, and utilisation during BFGS optimisation
with and without jax.checkpoint.

Each trial runs in a **separate subprocess** so that JAX's peak_bytes_in_use
counter resets between trials. The parent process polls nvidia-smi for
power/utilisation while the child runs.

Usage (on GPU box):
    # Quick comparison: checkpoint on vs off at default size
    python scripts/profile_bfgs_memory.py

    # Sweep n_parameter_sets to find the OOM ceiling
    python scripts/profile_bfgs_memory.py --sweep

    # Custom sweep range
    python scripts/profile_bfgs_memory.py --sweep --min-sets 1 --max-sets 64

    # Disable JAX memory pool for true nvidia-smi readings
    python scripts/profile_bfgs_memory.py --no-pool

    # Longer window (more memory pressure)
    python scripts/profile_bfgs_memory.py --months 6

    # Higher eval points / BFGS iterations
    python scripts/profile_bfgs_memory.py --n-eval 20 --maxiter 10

    # Save results
    python scripts/profile_bfgs_memory.py --sweep --json results.json
"""
from __future__ import annotations

import sys
import os

# ── --no-pool: set env var BEFORE importing jax, then re-exec ────────────────
if "--no-pool" in sys.argv and "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import time
import argparse
import json
import subprocess
import threading
from dataclasses import dataclass, field
from typing import List, Optional


# ── nvidia-smi poller (runs in parent process) ───────────────────────────────

@dataclass
class GpuSnapshot:
    timestamp: float
    memory_used_mb: float
    memory_total_mb: float
    power_draw_w: float
    utilisation_pct: float


@dataclass
class GpuStats:
    smi_peak_memory_mb: float = 0.0
    smi_min_memory_mb: float = 0.0
    memory_total_mb: float = 0.0
    mean_power_w: float = 0.0
    peak_power_w: float = 0.0
    mean_utilisation_pct: float = 0.0
    peak_utilisation_pct: float = 0.0
    n_smi_samples: int = 0
    jax_peak_bytes: int = 0


def query_nvidia_smi() -> Optional[GpuSnapshot]:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
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


# ── Trial result ──────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    n_parameter_sets: int
    n_eval_points: int
    gradient_checkpointing: bool
    success: bool
    wall_time_s: float = 0.0
    gpu: GpuStats = field(default_factory=GpuStats)
    error: str = ""


# ── Subprocess worker ─────────────────────────────────────────────────────────
# Each trial runs in a fresh process so peak_bytes_in_use resets.
# Config is passed via a temp JSON file to avoid template escaping issues.

WORKER_SCRIPT = '''
import sys, os, json, time

config_path = sys.argv[1]
repo_root = sys.argv[2]
sys.path.insert(0, repo_root)

import jax
import jax.numpy as jnp

def get_peak_bytes():
    try:
        stats = jax.local_devices()[0].memory_stats()
        return stats.get("peak_bytes_in_use", 0) if stats else 0
    except Exception:
        return 0

with open(config_path) as f:
    config = json.load(f)

from datetime import datetime
from dateutil.relativedelta import relativedelta
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set
from quantammsim.runners.jax_runners import train_on_historic_data

months = config["months"]
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
        "n_parameter_sets": config["n_parameter_sets"],
        "noise_scale": 0.3,
        "val_fraction": 0.0,
        "bfgs_settings": {
            "maxiter": config["maxiter"],
            "tol": 1e-6,
            "n_evaluation_points": config["n_eval_points"],
            "gradient_checkpointing": config["gradient_checkpointing"],
        },
    },
}
recursive_default_set(fp, run_fingerprint_defaults)

result = {"success": False, "wall_time_s": 0.0, "jax_peak_bytes": 0, "error": ""}

try:
    jax.effects_barrier()
    t0 = time.perf_counter()
    root = config.get("root")
    kwargs = {"verbose": False, "force_init": True, "return_training_metadata": True}
    if root:
        kwargs["root"] = root
    train_on_historic_data(fp, **kwargs)
    jax.effects_barrier()
    result["wall_time_s"] = time.perf_counter() - t0
    result["success"] = True
except Exception as e:
    result["wall_time_s"] = time.perf_counter() - t0
    err = str(e).lower()
    if "resource" in err or "memory" in err or "oom" in err:
        result["error"] = "OOM"
    else:
        result["error"] = str(e)[:200]

result["jax_peak_bytes"] = get_peak_bytes()
print("RESULT_JSON:" + json.dumps(result))
'''


def run_trial_subprocess(
    n_parameter_sets: int,
    n_eval_points: int,
    gradient_checkpointing: bool,
    maxiter: int,
    months: int,
    gpu_monitor: GpuMonitor,
    root: Optional[str] = None,
) -> TrialResult:
    """Run a single BFGS trial in a fresh subprocess."""
    import tempfile

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config = {
        "n_parameter_sets": n_parameter_sets,
        "n_eval_points": n_eval_points,
        "gradient_checkpointing": gradient_checkpointing,
        "maxiter": maxiter,
        "months": months,
    }
    if root:
        config["root"] = root

    # Write config and worker script to temp files
    config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="bfgs_profile_cfg_"
    )
    json.dump(config, config_file)
    config_file.close()

    worker_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="bfgs_profile_worker_"
    )
    worker_file.write(WORKER_SCRIPT)
    worker_file.close()

    env = os.environ.copy()

    result = TrialResult(
        n_parameter_sets=n_parameter_sets,
        n_eval_points=n_eval_points,
        gradient_checkpointing=gradient_checkpointing,
        success=False,
    )

    gpu_monitor.start()
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, worker_file.name, config_file.name, repo_root],
            capture_output=True, text=True, timeout=600, env=env,
        )
        wall_time = time.perf_counter() - t0

        # Parse result from stdout
        worker_result = None
        for line in proc.stdout.split("\n"):
            if line.startswith("RESULT_JSON:"):
                worker_result = json.loads(line[len("RESULT_JSON:"):])
                break

        if worker_result is None:
            stderr_tail = proc.stderr[-500:] if proc.stderr else "(empty)"
            result.error = f"No result. stderr: {stderr_tail}"
            result.wall_time_s = wall_time
        else:
            result.success = worker_result["success"]
            result.wall_time_s = worker_result["wall_time_s"]
            result.error = worker_result.get("error", "")
            result.gpu.jax_peak_bytes = worker_result.get("jax_peak_bytes", 0)

    except subprocess.TimeoutExpired:
        result.error = "TIMEOUT"
        result.wall_time_s = time.perf_counter() - t0
    except Exception as e:
        result.error = str(e)[:200]
        result.wall_time_s = time.perf_counter() - t0
    finally:
        jax_peak = result.gpu.jax_peak_bytes
        smi_stats = gpu_monitor.stop()
        result.gpu = smi_stats
        result.gpu.jax_peak_bytes = jax_peak

        # Clean up temp files
        for path in [config_file.name, worker_file.name]:
            try:
                os.unlink(path)
            except OSError:
                pass

    return result


# ── Display ───────────────────────────────────────────────────────────────────

NO_POOL_MODE = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") == "platform"


def print_result_row(r: Optional[TrialResult] = None, header: bool = False):
    if header:
        print(f"{'ckpt':>5} {'n_sets':>6} {'n_eval':>6} "
              f"{'jax_pk_MB':>10} {'smi_pk_MB':>10} "
              f"{'mean_W':>7} {'peak_W':>7} "
              f"{'mean_%':>7} {'peak_%':>7} {'time_s':>7} {'status':>8}")
        print("-" * 96)
        return

    ckpt = "ON" if r.gradient_checkpointing else "OFF"
    jax_mb = r.gpu.jax_peak_bytes / (1024 * 1024) if r.gpu.jax_peak_bytes else 0
    smi_mb = r.gpu.smi_peak_memory_mb

    if r.success:
        print(f"{ckpt:>5} {r.n_parameter_sets:>6} {r.n_eval_points:>6} "
              f"{jax_mb:>10.0f} {smi_mb:>10.0f} "
              f"{r.gpu.mean_power_w:>7.1f} {r.gpu.peak_power_w:>7.1f} "
              f"{r.gpu.mean_utilisation_pct:>7.1f} {r.gpu.peak_utilisation_pct:>7.1f} "
              f"{r.wall_time_s:>7.1f} {'OK':>8}")
    else:
        print(f"{ckpt:>5} {r.n_parameter_sets:>6} {r.n_eval_points:>6} "
              f"{jax_mb:>10.0f} {smi_mb:>10.0f} "
              f"{'':>7} {'':>7} {'':>7} {'':>7} "
              f"{r.wall_time_s:>7.1f} {r.error:>8}")


def print_comparison(results: List[TrialResult]):
    on = [r for r in results if r.gradient_checkpointing]
    off = [r for r in results if not r.gradient_checkpointing]

    if not (on and off and on[0].success and off[0].success):
        return

    mem_on = on[0].gpu.jax_peak_bytes / (1024 * 1024)
    mem_off = off[0].gpu.jax_peak_bytes / (1024 * 1024)

    if mem_off > 0 and mem_on > 0:
        reduction = (1 - mem_on / mem_off) * 100
        print(f"\n  JAX peak memory:  {mem_off:.0f} MB -> {mem_on:.0f} MB "
              f"({reduction:+.1f}%)")

    if NO_POOL_MODE:
        smi_on = on[0].gpu.smi_peak_memory_mb
        smi_off = off[0].gpu.smi_peak_memory_mb
        if smi_off > 0:
            reduction = (1 - smi_on / smi_off) * 100
            print(f"  nvidia-smi peak:  {smi_off:.0f} MB -> {smi_on:.0f} MB "
                  f"({reduction:+.1f}%)")

    time_on = on[0].wall_time_s
    time_off = off[0].wall_time_s
    if time_off > 0:
        slowdown = (time_on / time_off - 1) * 100
        print(f"  Wall time:        {time_off:.1f}s -> {time_on:.1f}s "
              f"({slowdown:+.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Profile BFGS GPU memory with/without gradient checkpointing"
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep n_parameter_sets to find OOM ceiling")
    parser.add_argument("--min-sets", type=int, default=1)
    parser.add_argument("--max-sets", type=int, default=32)
    parser.add_argument("--n-sets", type=int, default=4,
                        help="n_parameter_sets for single comparison (default: 4)")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="n_evaluation_points (default: 20, matching production)")
    parser.add_argument("--maxiter", type=int, default=3,
                        help="BFGS iterations per trial (default: 3, enough for peak memory)")
    parser.add_argument("--months", type=int, default=12,
                        help="Training window in months (default: 12, production uses 12-48)")
    parser.add_argument("--no-pool", action="store_true",
                        help="Disable JAX memory pool for true nvidia-smi readings")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    gpu_monitor = GpuMonitor()
    has_smi = gpu_monitor.available
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default (pool)")

    print(f"{'=' * 96}")
    print(f"  BFGS Gradient Checkpointing Memory Profiler")
    print(f"{'=' * 96}")
    print(f"  Allocator:      {allocator}")
    print(f"  nvidia-smi:     {'available' if has_smi else 'NOT FOUND'}")
    print(f"  Subprocess:     each trial runs in a fresh process (peak counter resets)")
    print(f"  n_eval_points:  {args.n_eval}")
    print(f"  maxiter:        {args.maxiter}")
    print(f"  months:         {args.months}")
    if args.root:
        print(f"  data root:      {args.root}")
    print(f"{'=' * 96}")

    results = []

    if args.sweep:
        for ckpt in [False, True]:
            label = "checkpoint ON" if ckpt else "checkpoint OFF"
            print(f"\n--- Sweep: {label} ---")
            print_result_row(header=True)

            n = args.min_sets
            while n <= args.max_sets:
                r = run_trial_subprocess(
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
                    print(f"  -> OOM at n_parameter_sets={n}, stopping sweep")
                    break

                n *= 2

            successes = [r.n_parameter_sets for r in results
                         if r.gradient_checkpointing == ckpt and r.success]
            if successes:
                print(f"  -> Max successful: n_parameter_sets={max(successes)}")

        on_max = max(
            (r.n_parameter_sets for r in results
             if r.gradient_checkpointing and r.success), default=0)
        off_max = max(
            (r.n_parameter_sets for r in results
             if not r.gradient_checkpointing and r.success), default=0)
        print(f"\n{'=' * 96}")
        print(f"  SWEEP SUMMARY")
        print(f"{'=' * 96}")
        print(f"  Max n_parameter_sets (checkpoint OFF): {off_max}")
        print(f"  Max n_parameter_sets (checkpoint ON):  {on_max}")
        if off_max > 0:
            print(f"  Parallelism gain: {on_max / off_max:.1f}x")
        print(f"{'=' * 96}")

    else:
        print(f"\n--- Comparison at n_parameter_sets={args.n_sets} ---")
        print_result_row(header=True)

        for ckpt in [False, True]:
            r = run_trial_subprocess(
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
                "jax_peak_bytes": r.gpu.jax_peak_bytes,
                "jax_peak_mb": r.gpu.jax_peak_bytes / (1024 * 1024),
                "smi_peak_memory_mb": r.gpu.smi_peak_memory_mb,
                "memory_total_mb": r.gpu.memory_total_mb,
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
