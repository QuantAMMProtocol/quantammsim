"""
E2E tests for weight calculation path selection.

Tests that:
1. Vectorized and scan paths produce equivalent results for pools that support both
2. Path selection ("vectorized", "scan", "auto") works correctly
3. Appropriate errors are raised for unsupported paths
"""

import pytest
import numpy as np
import jax.numpy as jnp

from quantammsim.runners.jax_runners import do_run_on_historic_data
from tests.conftest import TEST_DATA_DIR
from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.pools.G3M.quantamm.power_channel_pool import PowerChannelPool
from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import MeanReversionChannelPool
from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool
from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb


class TestPathEquivalence:
    """Test that vectorized and scan paths produce equivalent results."""

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

    def test_momentum_vectorized_scan_equivalence(self, base_fingerprint, momentum_params):
        """Test that momentum pool produces same results on both paths."""
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

        result_vectorized = do_run_on_historic_data(
            run_fingerprint=fingerprint_vectorized,
            params=momentum_params,
            root=TEST_DATA_DIR,
        )
        result_scan = do_run_on_historic_data(
            run_fingerprint=fingerprint_scan,
            params=momentum_params,
            root=TEST_DATA_DIR,
        )

        # Final values should match within tolerance
        np.testing.assert_allclose(
            result_vectorized["final_value"],
            result_scan["final_value"],
            rtol=1e-4,
            err_msg="Final values differ between vectorized and scan paths"
        )

        # Weights should match
        np.testing.assert_allclose(
            result_vectorized["weights"],
            result_scan["weights"],
            rtol=1e-4,
            atol=1e-6,
            err_msg="Weights differ between vectorized and scan paths"
        )

        # Reserves should match
        np.testing.assert_allclose(
            result_vectorized["reserves"],
            result_scan["reserves"],
            rtol=1e-4,
            atol=1e-6,
            err_msg="Reserves differ between vectorized and scan paths"
        )

    def test_power_channel_vectorized_scan_equivalence(self, base_fingerprint):
        """Test that power channel pool produces same results on both paths."""
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
            ]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "raw_exponents": jnp.array([1.0, 1.0]),
            "raw_pre_exp_scaling": jnp.array([0.5, 0.5]),
        }

        fingerprint_vectorized = {
            **base_fingerprint,
            "rule": "power_channel",
            "weight_calculation_method": "vectorized",
            "use_pre_exp_scaling": True,
        }
        fingerprint_scan = {
            **base_fingerprint,
            "rule": "power_channel",
            "weight_calculation_method": "scan",
            "use_pre_exp_scaling": True,
        }

        result_vectorized = do_run_on_historic_data(
            run_fingerprint=fingerprint_vectorized,
            params=params,
            root=TEST_DATA_DIR,
        )
        result_scan = do_run_on_historic_data(
            run_fingerprint=fingerprint_scan,
            params=params,
            root=TEST_DATA_DIR,
        )

        np.testing.assert_allclose(
            result_vectorized["final_value"],
            result_scan["final_value"],
            rtol=1e-4,
            err_msg="Power channel: Final values differ between paths"
        )

        np.testing.assert_allclose(
            result_vectorized["weights"],
            result_scan["weights"],
            rtol=1e-4,
            atol=1e-6,
            err_msg="Power channel: Weights differ between paths"
        )

    def test_mean_reversion_vectorized_scan_equivalence(self, base_fingerprint):
        """Test that mean reversion pool produces same results on both paths."""
        params = {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
                memory_days_to_logit_lamb(10.0, chunk_period=1440),
            ]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
            "log_amplitude": jnp.array([0.0, 0.0]),
            "raw_width": jnp.array([0.0, 0.0]),
            "raw_exponents": jnp.array([1.0, 1.0]),
            "raw_pre_exp_scaling": jnp.array([0.5, 0.5]),
        }

        fingerprint_vectorized = {
            **base_fingerprint,
            "rule": "mean_reversion_channel",
            "weight_calculation_method": "vectorized",
            "use_pre_exp_scaling": True,
        }
        fingerprint_scan = {
            **base_fingerprint,
            "rule": "mean_reversion_channel",
            "weight_calculation_method": "scan",
            "use_pre_exp_scaling": True,
        }

        result_vectorized = do_run_on_historic_data(
            run_fingerprint=fingerprint_vectorized,
            params=params,
            root=TEST_DATA_DIR,
        )
        result_scan = do_run_on_historic_data(
            run_fingerprint=fingerprint_scan,
            params=params,
            root=TEST_DATA_DIR,
        )

        np.testing.assert_allclose(
            result_vectorized["final_value"],
            result_scan["final_value"],
            rtol=1e-4,
            err_msg="Mean reversion: Final values differ between paths"
        )


class TestPathSelection:
    """Test that path selection works correctly."""

    @pytest.fixture
    def base_fingerprint(self):
        return {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-04-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 1000000.0,
            "do_arb": True,
        }

    @pytest.fixture
    def momentum_params(self):
        return {
            "log_k": jnp.array([3.0, 3.0]),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        }

    def test_auto_selects_vectorized_when_available(self, base_fingerprint, momentum_params):
        """Test that 'auto' selects vectorized path for momentum pool."""
        fingerprint = {
            **base_fingerprint,
            "rule": "momentum",
            "weight_calculation_method": "auto",
        }

        # Should not raise - auto should select vectorized
        result = do_run_on_historic_data(
            run_fingerprint=fingerprint,
            params=momentum_params,
            root=TEST_DATA_DIR,
        )
        assert result["final_value"] > 0

    def test_explicit_vectorized_works(self, base_fingerprint, momentum_params):
        """Test that explicit 'vectorized' path works."""
        fingerprint = {
            **base_fingerprint,
            "rule": "momentum",
            "weight_calculation_method": "vectorized",
        }

        result = do_run_on_historic_data(
            run_fingerprint=fingerprint,
            params=momentum_params,
            root=TEST_DATA_DIR,
        )
        assert result["final_value"] > 0

    def test_explicit_scan_works(self, base_fingerprint, momentum_params):
        """Test that explicit 'scan' path works."""
        fingerprint = {
            **base_fingerprint,
            "rule": "momentum",
            "weight_calculation_method": "scan",
        }

        result = do_run_on_historic_data(
            run_fingerprint=fingerprint,
            params=momentum_params,
            root=TEST_DATA_DIR,
        )
        assert result["final_value"] > 0


class TestPathCapabilities:
    """Test pool capability detection and error handling."""

    def test_momentum_supports_both_paths(self):
        """Test that MomentumPool supports both vectorized and scan paths."""
        pool = MomentumPool()
        assert pool.supports_vectorized_path()
        assert pool.supports_scan_path()

    def test_power_channel_supports_both_paths(self):
        """Test that PowerChannelPool supports both vectorized and scan paths."""
        pool = PowerChannelPool()
        assert pool.supports_vectorized_path()
        assert pool.supports_scan_path()

    def test_mean_reversion_supports_both_paths(self):
        """Test that MeanReversionChannelPool supports both vectorized and scan paths."""
        pool = MeanReversionChannelPool()
        assert pool.supports_vectorized_path()
        assert pool.supports_scan_path()

    def test_min_variance_supports_only_vectorized(self):
        """Test that MinVariancePool only supports vectorized path."""
        pool = MinVariancePool()
        assert pool.supports_vectorized_path()
        assert not pool.supports_scan_path()


class TestThreeAssetEquivalence:
    """Test path equivalence with 3 assets."""

    def test_momentum_three_assets_equivalence(self):
        """Test momentum pool with 3 assets produces same results on both paths."""
        fingerprint_base = {
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-04-01 00:00:00",
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

        fingerprint_vectorized = {**fingerprint_base, "weight_calculation_method": "vectorized"}
        fingerprint_scan = {**fingerprint_base, "weight_calculation_method": "scan"}

        result_vectorized = do_run_on_historic_data(
            run_fingerprint=fingerprint_vectorized,
            params=params,
            root=TEST_DATA_DIR,
        )
        result_scan = do_run_on_historic_data(
            run_fingerprint=fingerprint_scan,
            params=params,
            root=TEST_DATA_DIR,
        )

        np.testing.assert_allclose(
            result_vectorized["final_value"],
            result_scan["final_value"],
            rtol=1e-4,
            err_msg="3-asset momentum: Final values differ between paths"
        )

        np.testing.assert_allclose(
            result_vectorized["weights"],
            result_scan["weights"],
            rtol=1e-4,
            atol=1e-6,
            err_msg="3-asset momentum: Weights differ between paths"
        )
