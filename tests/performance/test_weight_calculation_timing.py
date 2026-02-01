"""
Performance tests comparing vectorized and scan weight calculation paths.

These tests benchmark execution time differences between the two paths.
Run with: pytest tests/performance/test_weight_calculation_timing.py -v -s
"""

import pytest
import time
import jax.numpy as jnp
from jax import block_until_ready

from quantammsim.runners.jax_runners import do_run_on_historic_data
from tests.conftest import TEST_DATA_DIR
from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb


@pytest.mark.slow
class TestWeightCalculationTiming:
    """Benchmark vectorized vs scan paths."""

    @pytest.fixture
    def base_fingerprint(self):
        return {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
            "arb_quality": 1.0,
            "maximum_change": 0.001,
        }

    @pytest.fixture
    def momentum_params(self):
        return {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
            ]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

    def _time_run(self, fingerprint, params, n_warmup=1, n_runs=3):
        """Time a run with warmup iterations."""
        # Warmup runs (for JIT compilation)
        for _ in range(n_warmup):
            result = do_run_on_historic_data(
                run_fingerprint=fingerprint,
                params=params,
                root=TEST_DATA_DIR,
            )
            block_until_ready(result["final_value"])

        # Timed runs
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            result = do_run_on_historic_data(
                run_fingerprint=fingerprint,
                params=params,
                root=TEST_DATA_DIR,
            )
            block_until_ready(result["final_value"])
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "final_value": float(result["final_value"]),
        }

    def test_momentum_timing_comparison(self, base_fingerprint, momentum_params):
        """Compare timing between vectorized and scan paths for momentum."""
        fingerprint_vectorized = {
            **base_fingerprint,
            "rule": "momentum",
            "weight_calculation_method": "vectorized",
        }
        fingerprint_scan = {
            **base_fingerprint,
            "rule": "momentum",
            "weight_calculation_method": "scan",
        }

        print("\n" + "=" * 60)
        print("Momentum Pool Timing Comparison (2 assets, 6 months)")
        print("=" * 60)

        vectorized_timing = self._time_run(fingerprint_vectorized, momentum_params)
        print(f"Vectorized path: {vectorized_timing['mean_time']:.4f}s "
              f"(min: {vectorized_timing['min_time']:.4f}s, max: {vectorized_timing['max_time']:.4f}s)")

        scan_timing = self._time_run(fingerprint_scan, momentum_params)
        print(f"Scan path:       {scan_timing['mean_time']:.4f}s "
              f"(min: {scan_timing['min_time']:.4f}s, max: {scan_timing['max_time']:.4f}s)")

        ratio = scan_timing['mean_time'] / vectorized_timing['mean_time']
        print(f"Scan/Vectorized ratio: {ratio:.2f}x")

        # Verify results match
        assert abs(vectorized_timing['final_value'] - scan_timing['final_value']) < 1.0, \
            "Final values should match between paths"

    def test_three_asset_timing_comparison(self):
        """Compare timing with 3 assets."""
        base_fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH", "USDC"],
            "rule": "momentum",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
            "maximum_change": 0.001,
        }
        params = {
            "log_k": jnp.array([3.0, 3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0, 0.0]),
        }

        fingerprint_vectorized = {**base_fingerprint, "weight_calculation_method": "vectorized"}
        fingerprint_scan = {**base_fingerprint, "weight_calculation_method": "scan"}

        print("\n" + "=" * 60)
        print("Momentum Pool Timing Comparison (3 assets, 6 months)")
        print("=" * 60)

        vectorized_timing = self._time_run(fingerprint_vectorized, params)
        print(f"Vectorized path: {vectorized_timing['mean_time']:.4f}s")

        scan_timing = self._time_run(fingerprint_scan, params)
        print(f"Scan path:       {scan_timing['mean_time']:.4f}s")

        ratio = scan_timing['mean_time'] / vectorized_timing['mean_time']
        print(f"Scan/Vectorized ratio: {ratio:.2f}x")

    def test_shorter_bout_timing(self):
        """Compare timing with shorter bout (1 month)."""
        base_fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-03-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
            "maximum_change": 0.001,
        }
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

        fingerprint_vectorized = {**base_fingerprint, "weight_calculation_method": "vectorized"}
        fingerprint_scan = {**base_fingerprint, "weight_calculation_method": "scan"}

        print("\n" + "=" * 60)
        print("Momentum Pool Timing Comparison (2 assets, 1 month)")
        print("=" * 60)

        vectorized_timing = self._time_run(fingerprint_vectorized, params)
        print(f"Vectorized path: {vectorized_timing['mean_time']:.4f}s")

        scan_timing = self._time_run(fingerprint_scan, params)
        print(f"Scan path:       {scan_timing['mean_time']:.4f}s")

        ratio = scan_timing['mean_time'] / vectorized_timing['mean_time']
        print(f"Scan/Vectorized ratio: {ratio:.2f}x")

    def test_hourly_chunk_timing(self):
        """Compare timing with hourly chunks (more iterations)."""
        base_fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-03-01 00:00:00",  # 1 month with hourly = ~720 chunks
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "chunk_period": 60,  # Hourly
            "weight_interpolation_period": 60,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
            "maximum_change": 0.001,
        }
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([
                memory_days_to_logit_lamb(10.0, chunk_period=60),
                memory_days_to_logit_lamb(10.0, chunk_period=60),
            ]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

        fingerprint_vectorized = {**base_fingerprint, "weight_calculation_method": "vectorized"}
        fingerprint_scan = {**base_fingerprint, "weight_calculation_method": "scan"}

        print("\n" + "=" * 60)
        print("Momentum Pool Timing Comparison (2 assets, 1 month, hourly chunks)")
        print("=" * 60)

        vectorized_timing = self._time_run(fingerprint_vectorized, params)
        print(f"Vectorized path: {vectorized_timing['mean_time']:.4f}s")

        scan_timing = self._time_run(fingerprint_scan, params)
        print(f"Scan path:       {scan_timing['mean_time']:.4f}s")

        ratio = scan_timing['mean_time'] / vectorized_timing['mean_time']
        print(f"Scan/Vectorized ratio: {ratio:.2f}x")


@pytest.mark.slow
class TestJITCompilationTiming:
    """Test JIT compilation overhead."""

    def test_first_run_vs_subsequent(self):
        """Measure JIT compilation overhead."""
        fingerprint = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-03-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
            "weight_calculation_method": "scan",
        }
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

        print("\n" + "=" * 60)
        print("JIT Compilation Overhead (scan path)")
        print("=" * 60)

        # First run (includes JIT compilation)
        start = time.perf_counter()
        result = do_run_on_historic_data(run_fingerprint=fingerprint, params=params, root=TEST_DATA_DIR)
        block_until_ready(result["final_value"])
        first_run_time = time.perf_counter() - start
        print(f"First run (with JIT):  {first_run_time:.4f}s")

        # Subsequent runs (JIT cached)
        times = []
        for i in range(3):
            start = time.perf_counter()
            result = do_run_on_historic_data(run_fingerprint=fingerprint, params=params, root=TEST_DATA_DIR)
            block_until_ready(result["final_value"])
            times.append(time.perf_counter() - start)

        avg_subsequent = sum(times) / len(times)
        print(f"Subsequent runs (avg): {avg_subsequent:.4f}s")
        print(f"JIT overhead:          {first_run_time - avg_subsequent:.4f}s")
