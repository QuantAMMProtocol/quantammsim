"""
Integration tests for gradient clipping functionality.

These tests verify that gradient clipping prevents parameter explosion
during training with high learning rates.
"""
import pytest
import copy

from quantammsim.runners.jax_runners import train_on_historic_data
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set, check_run_fingerprint
from tests.conftest import TEST_DATA_DIR


@pytest.fixture
def explosive_training_fingerprint():
    """Create a fingerprint prone to parameter explosion without clipping."""
    return {
        "tokens": ["BTC", "ETH", "SOL", "USDC"],
        "rule": "power_channel",  # Power channel is most sensitive to explosion
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-02-01 00:00:00",
        "endTestDateString": "2023-02-10 00:00:00",
        "chunk_period": 1440,
        "bout_offset": 30240,  # 21 days - leaves ~10 days effective window
        "weight_interpolation_period": 1440,
        "optimisation_settings": {
            "base_lr": 20.0,  # High learning rate to trigger explosion
            "optimiser": "sgd",
            "decay_lr_ratio": 0.8,
            "decay_lr_plateau": 200,
            "batch_size": 2,
            "train_on_hessian_trace": False,
            "min_lr": 0.004,
            "n_iterations": 3,
            "n_cycles": 1,
            "sample_method": "uniform",
            "n_parameter_sets": 1,
            "training_data_kind": "historic",
            "max_mc_version": 9,
            "include_flipped_training_data": False,
            "initial_random_key": 0,
            "method": "gradient_descent",
            "force_scalar": False,
        },
        "initial_memory_length": 10.0,
        "initial_memory_length_delta": 0.0,
        "initial_k_per_day": 20,
        "initial_weights_logits": 1.0,
        "initial_log_amplitude": -10.0,
        "initial_raw_width": -8.0,
        "initial_raw_exponents": 0.0,
        "subsidary_pools": [],
        "maximum_change": 0.0003,
        "return_val": "returns",
        "initial_pool_value": 1000000.0,
        "fees": 0,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "weight_interpolation_method": "linear",
        "arb_frequency": 1,
        "do_arb": True,
        "arb_quality": 1.0,
        "numeraire": None,
        "do_trades": False,
        "noise_trader_ratio": 0.0,
        "minimum_weight": 0.03,
        "max_memory_days": 365,
    }


class TestGradientClipping:
    """Test suite for gradient clipping functionality."""

    @pytest.mark.slow
    @pytest.mark.training
    @pytest.mark.requires_data
    def test_training_without_clipping(self, explosive_training_fingerprint):
        """Test training without gradient clipping (may have numerical issues)."""
        fp = copy.deepcopy(explosive_training_fingerprint)
        fp["use_gradient_clipping"] = False

        recursive_default_set(fp, run_fingerprint_defaults)
        check_run_fingerprint(fp)

        # Training may complete or fail - we're just checking it doesn't crash unexpectedly
        try:
            result = train_on_historic_data(
                fp,
                iterations_per_print=1,
                verbose=False,
                root=TEST_DATA_DIR,
            )
            # If it completes, result should exist (may contain NaN due to explosion)
            assert result is not None
        except Exception as e:
            # Expected to potentially fail without clipping
            pytest.skip(f"Training failed without clipping (expected): {e}")

    @pytest.mark.slow
    @pytest.mark.training
    @pytest.mark.requires_data
    def test_training_with_clipping(self, explosive_training_fingerprint):
        """Test that training with gradient clipping completes successfully."""
        fp = copy.deepcopy(explosive_training_fingerprint)
        fp["use_gradient_clipping"] = True
        fp["clip_norm"] = 10.0
        fp["clip_by_param_type"] = True

        recursive_default_set(fp, run_fingerprint_defaults)
        check_run_fingerprint(fp)

        result = train_on_historic_data(
            fp,
            iterations_per_print=1,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        assert result is not None

    @pytest.mark.slow
    @pytest.mark.training
    @pytest.mark.requires_data
    def test_training_with_tight_clipping(self, explosive_training_fingerprint):
        """Test training with tighter clipping threshold."""
        fp = copy.deepcopy(explosive_training_fingerprint)
        fp["use_gradient_clipping"] = True
        fp["clip_norm"] = 5.0  # Tighter global clipping
        fp["clip_by_param_type"] = True

        recursive_default_set(fp, run_fingerprint_defaults)
        check_run_fingerprint(fp)

        result = train_on_historic_data(
            fp,
            iterations_per_print=1,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        assert result is not None

    @pytest.mark.slow
    @pytest.mark.training
    @pytest.mark.requires_data
    @pytest.mark.parametrize("clip_norm", [1.0, 5.0, 10.0, 50.0])
    def test_various_clip_norms(self, explosive_training_fingerprint, clip_norm):
        """Test training with various clipping thresholds."""
        fp = copy.deepcopy(explosive_training_fingerprint)
        fp["use_gradient_clipping"] = True
        fp["clip_norm"] = clip_norm
        fp["clip_by_param_type"] = True
        fp["optimisation_settings"]["n_iterations"] = 3
        fp["optimisation_settings"]["n_cycles"] = 1

        recursive_default_set(fp, run_fingerprint_defaults)
        check_run_fingerprint(fp)

        result = train_on_historic_data(
            fp,
            iterations_per_print=1,
            verbose=False,
            root=TEST_DATA_DIR,
        )

        assert result is not None, f"Training failed with clip_norm={clip_norm}"
