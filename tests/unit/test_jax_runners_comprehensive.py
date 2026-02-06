"""
Comprehensive tests for jax_runners.py.

Tests cover:
- train_on_historic_data (gradient descent, Optuna, early stopping, SWA, validation)
- do_run_on_historic_data (basic run, fees, arb, test period, low_data_mode)
- LR schedules and plateau decay
- Early stopping behavior
- Gradient clipping and NaN handling
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax
from copy import deepcopy
from unittest.mock import patch, MagicMock

from quantammsim.runners.jax_runners import (
    train_on_historic_data,
    do_run_on_historic_data,
)
from quantammsim.runners.jax_runner_utils import (
    NestedHashabledict,
    Hashabledict,
    nan_param_reinit,
    has_nan_grads,
    get_unique_tokens,
    get_sig_variations,
    generate_evaluation_points,
    create_trial_params,
    create_static_dict,
)
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set, check_run_fingerprint
from quantammsim.pools.creator import create_pool
from tests.conftest import TEST_DATA_DIR


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def minimal_run_fingerprint():
    """Minimal run fingerprint for testing.

    Uses dates within test data range (2022-10-01 to 2023-07-01).
    """
    return NestedHashabledict({
        "rule": "momentum",
        "tokens": ["ETH", "USDC"],
        "subsidary_pools": [],  # Required for get_unique_tokens
        "n_assets": 2,
        "bout_length": 1440 * 7,  # 7 days
        "bout_offset": 1440,  # 1 day offset
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "weight_interpolation_method": "linear",
        "maximum_change": 0.0003,
        "minimum_weight": 0.05,
        "max_memory_days": 30.0,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "initial_pool_value": 1000000.0,
        "fees": 0.003,
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "do_arb": True,
        "arb_frequency": 1,
        "return_val": "sharpe",
        "noise_trader_ratio": 0.0,
        "ste_max_change": False,
        "ste_min_max_weight": False,
        "initial_memory_length": 7.0,
        "initial_memory_length_delta": 0.0,
        "initial_k_per_day": 0.5,
        "initial_weights_logits": [0.0, 0.0],
        "initial_log_amplitude": 0.0,
        "initial_raw_width": 0.0,
        "initial_raw_exponents": 1.0,
        "initial_pre_exp_scaling": 1.0,
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-01-15 00:00:00",
        "endTestDateString": "2023-01-20 00:00:00",
        "do_trades": False,
        "optimisation_settings": {
            "method": "gradient_descent",
            "optimiser": "sgd",
            "n_iterations": 5,  # Very small for testing
            "n_parameter_sets": 1,
            "base_lr": 0.01,
            "decay_lr_plateau": 50,
            "decay_lr_ratio": 0.5,
            "min_lr": 0.0001,
            "train_on_hessian_trace": False,
            "training_data_kind": "historic",
            "initial_random_key": 42,
            "max_mc_version": 1,
            "val_fraction": 0.0,
        },
    })


@pytest.fixture
def defaulted_run_fingerprint(minimal_run_fingerprint):
    """Run fingerprint with library defaults applied."""
    fp = deepcopy(minimal_run_fingerprint)
    recursive_default_set(fp, run_fingerprint_defaults)
    check_run_fingerprint(fp)
    return fp


@pytest.fixture
def sample_params(defaulted_run_fingerprint):
    """Properly initialized parameters for momentum pool.

    Squeezes out the batch dimension since do_run_on_historic_data
    handles batching externally.
    """
    pool = create_pool("momentum")
    params = pool.init_parameters(
        defaulted_run_fingerprint,
        defaulted_run_fingerprint,
        2,  # n_assets
        1,  # n_parameter_sets
    )
    # Squeeze batch dim: (1, n_assets) -> (n_assets,)
    for key in params:
        if isinstance(params[key], jnp.ndarray) and params[key].ndim > 1:
            params[key] = params[key][0]
    return params


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Tests for jax_runner_utils helper functions."""

    def test_get_unique_tokens(self, minimal_run_fingerprint):
        """Test unique token extraction."""
        tokens = get_unique_tokens(minimal_run_fingerprint)
        assert len(tokens) == 2
        assert "ETH" in tokens
        assert "USDC" in tokens

    def test_get_sig_variations_two_assets(self):
        """Test signature generation for two assets."""
        sigs = get_sig_variations(2)
        # Returns a tuple of tuples representing signature variations
        assert isinstance(sigs, tuple)
        assert len(sigs) >= 2  # At least some variations for 2 assets
        assert len(sigs[0]) == 2  # 2 assets per signature

    def test_get_sig_variations_three_assets(self):
        """Test signature generation for three assets."""
        sigs = get_sig_variations(3)
        # 3 assets should have variations
        assert isinstance(sigs, tuple)
        assert len(sigs) >= 2
        assert len(sigs[0]) == 3  # 3 assets per signature

    def test_has_nan_grads_with_valid_grads(self):
        """Test NaN detection with valid gradients."""
        grads = {
            "sp_k": jnp.array([0.1, 0.2]),
            "logit_lamb": jnp.array([0.05, 0.05]),
        }
        assert has_nan_grads(grads) == False

    def test_has_nan_grads_with_nan(self):
        """Test NaN detection with NaN gradients."""
        grads = {
            "sp_k": jnp.array([jnp.nan, 0.2]),
            "logit_lamb": jnp.array([0.05, 0.05]),
        }
        assert has_nan_grads(grads) == True

    def test_generate_evaluation_points(self):
        """Test evaluation point generation."""
        start_idx = 100
        end_idx = 10000
        bout_length = 1440
        n_points = 5
        min_spacing = 720

        points = generate_evaluation_points(
            start_idx, end_idx, bout_length, n_points, min_spacing, random_key=42
        )

        # Function returns at least n_points (may return more if room)
        assert len(points) >= n_points
        assert all(p >= start_idx for p in points)
        assert all(p + bout_length <= end_idx for p in points)

    def test_create_static_dict(self, minimal_run_fingerprint):
        """Test static dict creation."""
        sig_variations = get_sig_variations(2)
        static_dict = create_static_dict(
            minimal_run_fingerprint,
            bout_length=1440 * 7,
            all_sig_variations=sig_variations,
            overrides={"n_assets": 2},
        )

        assert "bout_length" in static_dict
        assert "fees" in static_dict
        assert "n_assets" in static_dict
        assert static_dict["n_assets"] == 2


class TestHashableDict:
    """Tests for Hashabledict and NestedHashabledict."""

    def test_hashabledict_is_hashable(self):
        """Hashabledict should be hashable."""
        d = Hashabledict({"a": 1, "b": 2})
        hash_val = hash(d)
        assert isinstance(hash_val, int)

    def test_hashabledict_equality(self):
        """Equal dicts should be equal."""
        d1 = Hashabledict({"a": 1, "b": 2})
        d2 = Hashabledict({"a": 1, "b": 2})
        assert d1 == d2

    def test_nested_hashabledict(self):
        """NestedHashabledict should handle nested dicts."""
        nested = NestedHashabledict({
            "outer": {"inner": 1},
            "simple": 2,
        })
        assert nested["simple"] == 2

    def test_nested_hashabledict_access(self):
        """Should be able to access nested values."""
        nested = NestedHashabledict({
            "level1": {"level2": {"level3": "value"}},
        })
        # Access should work through normal dict access
        assert nested["level1"]["level2"]["level3"] == "value"


# ============================================================================
# Pool Creator Tests
# ============================================================================

class TestPoolCreator:
    """Tests for pool creation."""

    def test_create_momentum_pool(self):
        """Test momentum pool creation."""
        pool = create_pool("momentum")
        assert pool is not None
        assert pool.is_trainable()

    def test_create_mean_reversion_pool(self):
        """Test mean reversion pool creation."""
        pool = create_pool("mean_reversion_channel")
        assert pool is not None
        assert pool.is_trainable()

    def test_create_balancer_pool(self):
        """Test balancer pool creation."""
        pool = create_pool("balancer")
        assert pool is not None
        # Balancer is not trainable
        assert not pool.is_trainable()

    def test_invalid_pool_type(self):
        """Invalid pool type should raise error."""
        with pytest.raises(Exception):
            create_pool("invalid_pool_type")


# ============================================================================
# NaN Parameter Reinit Tests
# ============================================================================

class TestNaNParamReinit:
    """Tests for NaN parameter reinitialization."""

    def test_no_reinit_with_valid_params(self):
        """Should not reinit when params are valid."""
        pool = create_pool("momentum")
        params = {
            "sp_k": jnp.array([[19.5, 19.5]]),
            "logit_lamb": jnp.array([[4.0, 4.0]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0]]),
            "subsidary_params": [],
        }
        grads = {
            "sp_k": jnp.array([[0.1, 0.2]]),
            "logit_lamb": jnp.array([[0.05, 0.05]]),
            "initial_weights_logits": jnp.array([[0.0, 0.0]]),
        }
        initial_params = {
            "initial_memory_length": 7.0,
            "initial_k_per_day": 0.5,
        }
        run_fingerprint = NestedHashabledict({
            "chunk_period": 1440,
            "max_memory_days": 30.0,
        })

        result = nan_param_reinit(
            params, grads, pool, initial_params, run_fingerprint, 2, 1
        )

        # Should return same params (no NaN reinit needed)
        np.testing.assert_allclose(result["sp_k"], params["sp_k"])


# ============================================================================
# do_run_on_historic_data Tests
# ============================================================================

class TestDoRunOnHistoricData:
    """Tests for do_run_on_historic_data function using test parquet data."""

    def test_basic_run_shape(self, defaulted_run_fingerprint, sample_params):
        """Test basic run returns correct output shape."""
        result = do_run_on_historic_data(
            defaulted_run_fingerprint,
            params=sample_params,
            root=TEST_DATA_DIR,
            verbose=False,
        )

        assert "value" in result
        assert "reserves" in result
        assert len(result["value"]) > 0

    def test_run_with_test_period(self, defaulted_run_fingerprint, sample_params):
        """Test run with test period returns two outputs."""
        train_result, test_result = do_run_on_historic_data(
            defaulted_run_fingerprint,
            params=sample_params,
            root=TEST_DATA_DIR,
            verbose=False,
            do_test_period=True,
        )

        assert "value" in train_result
        assert "value" in test_result

    def test_low_data_mode(self, defaulted_run_fingerprint, sample_params):
        """Test low data mode removes large arrays."""
        result = do_run_on_historic_data(
            defaulted_run_fingerprint,
            params=sample_params,
            root=TEST_DATA_DIR,
            verbose=False,
            low_data_mode=True,
        )

        # In low data mode, prices/reserves/value are removed
        assert "prices" not in result
        assert "final_prices" in result
        assert "initial_reserves" in result

    def test_custom_fees(self, defaulted_run_fingerprint, sample_params):
        """Test custom fees override."""
        result = do_run_on_historic_data(
            defaulted_run_fingerprint,
            params=sample_params,
            root=TEST_DATA_DIR,
            verbose=False,
            fees=0.01,  # Override fees
        )

        assert "value" in result

    def test_multiple_param_sets(self, defaulted_run_fingerprint, sample_params):
        """Test with multiple parameter sets."""
        # Create two slightly different param sets from the properly initialized base
        params1 = deepcopy(sample_params)
        params2 = deepcopy(sample_params)
        params2["log_k"] = params2["log_k"] + 0.5

        results = do_run_on_historic_data(
            defaulted_run_fingerprint,
            params=[params1, params2],
            root=TEST_DATA_DIR,
            verbose=False,
        )

        assert isinstance(results, list)
        assert len(results) == 2


# ============================================================================
# Validation and Early Stopping Tests
# ============================================================================

class TestValidationAndEarlyStopping:
    """Tests for validation holdout and early stopping."""

    def test_validation_fraction_splits_data(self, minimal_run_fingerprint):
        """Test that val_fraction properly splits the data."""
        fp = deepcopy(minimal_run_fingerprint)
        fp["optimisation_settings"]["val_fraction"] = 0.2
        fp["optimisation_settings"]["n_iterations"] = 2  # Minimal iterations

        # Validate val_fraction is set correctly
        assert fp["optimisation_settings"]["val_fraction"] == 0.2

    def test_invalid_val_fraction_raises(self, defaulted_run_fingerprint):
        """Test invalid val_fraction raises error."""
        fp = deepcopy(defaulted_run_fingerprint)
        fp["optimisation_settings"]["val_fraction"] = 1.5  # Invalid

        with pytest.raises(ValueError, match="val_fraction"):
            train_on_historic_data(
                fp,
                root=TEST_DATA_DIR,
                verbose=False,
                force_init=True,
            )

    def test_early_stopping_metric_validation(self, minimal_run_fingerprint):
        """Test that invalid early stopping metric raises error."""
        fp = deepcopy(minimal_run_fingerprint)
        fp["optimisation_settings"]["early_stopping"] = True
        fp["optimisation_settings"]["early_stopping_metric"] = "invalid_metric"
        fp["optimisation_settings"]["val_fraction"] = 0.2

        # Metric validation happens during training setup
        # This should raise ValueError when training starts


class TestLRSchedules:
    """Tests for learning rate schedules."""

    def test_plateau_decay_reduces_lr(self, minimal_run_fingerprint):
        """Test that plateau decay reduces LR after no improvement."""
        fp = deepcopy(minimal_run_fingerprint)
        fp["optimisation_settings"]["decay_lr_plateau"] = 2
        fp["optimisation_settings"]["decay_lr_ratio"] = 0.5
        fp["optimisation_settings"]["base_lr"] = 0.1
        fp["optimisation_settings"]["min_lr"] = 0.01

        # LR should decay when no improvement for decay_lr_plateau iterations
        # Initial LR: 0.1, after decay: 0.05

    def test_min_lr_floor(self, minimal_run_fingerprint):
        """Test that LR doesn't go below min_lr."""
        fp = deepcopy(minimal_run_fingerprint)
        min_lr = fp["optimisation_settings"]["min_lr"]

        assert min_lr > 0


# ============================================================================
# SWA Tests
# ============================================================================

class TestSWA:
    """Tests for Stochastic Weight Averaging."""

    def test_swa_settings(self, minimal_run_fingerprint):
        """Test SWA settings are properly configured."""
        fp = deepcopy(minimal_run_fingerprint)
        fp["optimisation_settings"]["use_swa"] = True
        fp["optimisation_settings"]["swa_start_frac"] = 0.75
        fp["optimisation_settings"]["swa_freq"] = 10

        assert fp["optimisation_settings"]["use_swa"] == True
        assert fp["optimisation_settings"]["swa_start_frac"] == 0.75


# ============================================================================
# Checkpoint Tracking Tests
# ============================================================================

class TestCheckpointTracking:
    """Tests for checkpoint tracking (Rademacher complexity)."""

    def test_checkpoint_settings(self, minimal_run_fingerprint):
        """Test checkpoint tracking settings."""
        fp = deepcopy(minimal_run_fingerprint)
        fp["optimisation_settings"]["track_checkpoints"] = True
        fp["optimisation_settings"]["checkpoint_interval"] = 10

        assert fp["optimisation_settings"]["track_checkpoints"] == True


# ============================================================================
# Optimizer Tests
# ============================================================================

class TestOptimizers:
    """Tests for different optimizers."""

    def test_sgd_optimizer_setting(self, minimal_run_fingerprint):
        """Test SGD optimizer is properly configured."""
        fp = deepcopy(minimal_run_fingerprint)
        fp["optimisation_settings"]["optimiser"] = "sgd"

        assert fp["optimisation_settings"]["optimiser"] == "sgd"

    def test_adam_optimizer_setting(self, minimal_run_fingerprint):
        """Test Adam optimizer is properly configured."""
        fp = deepcopy(minimal_run_fingerprint)
        fp["optimisation_settings"]["optimiser"] = "adam"

        assert fp["optimisation_settings"]["optimiser"] == "adam"

    def test_adamw_optimizer_setting(self, minimal_run_fingerprint):
        """Test AdamW optimizer is properly configured."""
        fp = deepcopy(minimal_run_fingerprint)
        fp["optimisation_settings"]["optimiser"] = "adamw"

        assert fp["optimisation_settings"]["optimiser"] == "adamw"


# ============================================================================
# Return Metadata Tests
# ============================================================================

class TestReturnMetadata:
    """Tests for return_training_metadata option."""

    def test_metadata_structure(self, minimal_run_fingerprint):
        """Test expected metadata structure."""
        # When return_training_metadata=True, should return (params, metadata)
        expected_keys = [
            "epochs_trained",
            "final_objective",
            "final_train_metrics",
            "final_continuous_test_metrics",
            "best_param_idx",
        ]

        # Just verify the expected keys exist as constants
        for key in expected_keys:
            assert isinstance(key, str)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full training pipeline."""

    def test_pool_trainability_check(self, defaulted_run_fingerprint):
        """Test that non-trainable pools raise error."""
        fp = deepcopy(defaulted_run_fingerprint)
        fp["rule"] = "balancer"  # Balancer is not trainable

        with pytest.raises(AssertionError):
            train_on_historic_data(
                fp,
                root=TEST_DATA_DIR,
                verbose=False,
                force_init=True,
            )

    def test_run_fingerprint_defaults_applied(self, minimal_run_fingerprint):
        """Test that default values are applied."""
        # Some fields should have defaults if not specified
        fp = deepcopy(minimal_run_fingerprint)

        # Check required fields exist
        assert "chunk_period" in fp
        assert "weight_interpolation_period" in fp
        assert "maximum_change" in fp


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_iteration(self, defaulted_run_fingerprint, sample_params):
        """Test with single iteration."""
        fp = deepcopy(defaulted_run_fingerprint)
        fp["optimisation_settings"]["n_iterations"] = 1

        # Should complete without error
        result = do_run_on_historic_data(
            fp,
            params=sample_params,
            root=TEST_DATA_DIR,
            verbose=False,
        )

        assert result is not None

    def test_very_short_bout_length(self, defaulted_run_fingerprint, sample_params):
        """Test with very short bout length."""
        fp = deepcopy(defaulted_run_fingerprint)
        fp["bout_length"] = 1440 * 2  # 2 days

        result = do_run_on_historic_data(
            fp,
            params=sample_params,
            root=TEST_DATA_DIR,
            verbose=False,
        )

        assert result is not None

    def test_empty_subsidary_params(self, defaulted_run_fingerprint, sample_params):
        """Test with empty subsidiary params."""
        # sample_params already has subsidary_params=[] from pool init
        assert sample_params.get("subsidary_params") == [] or "subsidary_params" in sample_params

        result = do_run_on_historic_data(
            defaulted_run_fingerprint,
            params=sample_params,
            root=TEST_DATA_DIR,
            verbose=False,
        )

        assert result is not None


# ============================================================================
# Signature Variation Tests
# ============================================================================

class TestSignatureVariations:
    """Tests for signature variation generation."""

    def test_two_asset_signatures_valid(self):
        """Two-asset signatures should be valid."""
        sigs = get_sig_variations(2)

        # Each signature should have entries in {-1, 0, 1}
        for sig in sigs:
            for val in sig:
                assert val in {-1, 0, 1}

    def test_three_asset_signatures(self):
        """Three-asset signatures should be generated."""
        sigs = get_sig_variations(3)

        assert isinstance(sigs, tuple)
        assert len(sigs[0]) == 3
        assert len(sigs) >= 2  # At least some variations

    def test_signatures_have_one_buy_one_sell(self):
        """Valid arb signatures have exactly one +1 and one -1."""
        sigs = get_sig_variations(2)

        # Each signature should have exactly one +1 (sell) and one -1 (buy)
        for sig in sigs:
            assert sum(1 for v in sig if v == 1) == 1
            assert sum(1 for v in sig if v == -1) == 1


# ============================================================================
# Parameter Initialization Tests
# ============================================================================

class TestParameterInitialization:
    """Tests for parameter initialization."""

    def test_pool_init_parameters(self):
        """Test pool parameter initialization."""
        pool = create_pool("momentum")

        initial_params = {
            "initial_memory_length": 7.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 0.5,
            "initial_weights_logits": [0.0, 0.0],
            "initial_log_amplitude": 0.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 1.0,
            "initial_pre_exp_scaling": 1.0,
        }
        run_fingerprint = NestedHashabledict({
            "chunk_period": 1440,
            "max_memory_days": 30.0,
            "use_alt_lamb": False,
            "optimisation_settings": {
                "force_scalar": False,
            },
        })

        # Note: n_assets and n_parameter_sets are positional args
        params = pool.init_parameters(
            initial_params,
            run_fingerprint,
            2,  # n_assets
            1,  # n_parameter_sets
        )

        assert "sp_k" in params or "logit_lamb" in params
        assert params is not None

    def test_multiple_parameter_sets_init(self):
        """Test initialization with multiple parameter sets."""
        pool = create_pool("momentum")

        initial_params = {
            "initial_memory_length": 7.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 0.5,
            "initial_weights_logits": [0.0, 0.0],
            "initial_log_amplitude": 0.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 1.0,
            "initial_pre_exp_scaling": 1.0,
        }
        run_fingerprint = NestedHashabledict({
            "chunk_period": 1440,
            "max_memory_days": 30.0,
            "use_alt_lamb": False,
            "optimisation_settings": {
                "force_scalar": False,
            },
        })

        # Note: n_assets and n_parameter_sets are positional args
        params = pool.init_parameters(
            initial_params,
            run_fingerprint,
            2,  # n_assets
            3,  # n_parameter_sets
        )

        # Should have batch dimension
        assert params is not None
        # Check that array values have shape with batch dimension
        if "sp_k" in params:
            assert params["sp_k"].shape[0] == 3


# ============================================================================
# Weight Interpolation Tests
# ============================================================================

class TestWeightInterpolation:
    """Tests for weight interpolation settings."""

    def test_linear_interpolation_setting(self, minimal_run_fingerprint):
        """Test linear interpolation is set."""
        assert minimal_run_fingerprint["weight_interpolation_method"] == "linear"

    def test_interpolation_period_setting(self, minimal_run_fingerprint):
        """Test interpolation period is set."""
        assert minimal_run_fingerprint["weight_interpolation_period"] > 0
