"""Tests for BFGS optimizer integration in train_on_historic_data.

Tests follow the same fixture/pattern as test_jax_runners_comprehensive.py.
Uses minimal data windows and iteration counts to keep tests fast.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from copy import deepcopy

from quantammsim.runners.jax_runners import train_on_historic_data
from quantammsim.runners.jax_runner_utils import NestedHashabledict
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set, check_run_fingerprint
from tests.conftest import TEST_DATA_DIR


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def bfgs_run_fingerprint():
    """Minimal run fingerprint for fast BFGS tests.

    Uses 3-day train + 2-day test windows within test data range.
    """
    return {
        "rule": "momentum",
        "tokens": ["ETH", "USDC"],
        "subsidary_pools": [],
        "n_assets": 2,
        "bout_offset": 0,
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "weight_interpolation_method": "linear",
        "maximum_change": 0.0003,
        "minimum_weight": 0.05,
        "max_memory_days": 5.0,
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
        "initial_memory_length": 3.0,
        "initial_memory_length_delta": 0.0,
        "initial_k_per_day": 0.5,
        "initial_weights_logits": [0.0, 0.0],
        "initial_log_amplitude": 0.0,
        "initial_raw_width": 0.0,
        "initial_raw_exponents": 1.0,
        "initial_pre_exp_scaling": 1.0,
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-01-04 00:00:00",
        "endTestDateString": "2023-01-06 00:00:00",
        "do_trades": False,
        "optimisation_settings": {
            "method": "bfgs",
            "n_parameter_sets": 1,
            "noise_scale": 0.1,
            "training_data_kind": "historic",
            "initial_random_key": 42,
            "max_mc_version": 1,
            "val_fraction": 0.0,
            "base_lr": 0.01,
            "optimiser": "adam",
            "decay_lr_plateau": 50,
            "decay_lr_ratio": 0.5,
            "min_lr": 0.0001,
            "train_on_hessian_trace": False,
            "n_iterations": 10,
            "bfgs_settings": {
                "maxiter": 5,
                "tol": 1e-6,
                "n_evaluation_points": 2,
            },
        },
    }


@pytest.fixture
def defaulted_bfgs_fingerprint(bfgs_run_fingerprint):
    """BFGS fingerprint with library defaults applied."""
    fp = deepcopy(bfgs_run_fingerprint)
    recursive_default_set(fp, run_fingerprint_defaults)
    check_run_fingerprint(fp)
    return fp


# ============================================================================
# Tests
# ============================================================================

class TestBFGSOptimizer:
    """Tests for the BFGS optimization branch."""

    def test_bfgs_runs_end_to_end(self, bfgs_run_fingerprint):
        """BFGS with n_parameter_sets=1 returns a params dict with correct keys."""
        fp = deepcopy(bfgs_run_fingerprint)

        result = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
        )

        assert result is not None
        assert isinstance(result, dict)
        # Momentum pool params should be present
        assert "log_k" in result
        assert "logit_lamb" in result
        # Params should be 1-D (n_assets,) — batch dim selected out
        for k, v in result.items():
            if k == "subsidary_params":
                continue
            if hasattr(v, "shape"):
                assert v.ndim == 1, f"{k} has ndim={v.ndim}, expected 1"

    def test_bfgs_multiple_parameter_sets(self, bfgs_run_fingerprint):
        """Multi-start BFGS with n_parameter_sets=2 returns correct shapes."""
        fp = deepcopy(bfgs_run_fingerprint)
        fp["optimisation_settings"]["n_parameter_sets"] = 2

        result = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
        )

        assert result is not None
        assert isinstance(result, dict)
        # Result should be a single param set (best selected)
        for k, v in result.items():
            if k == "subsidary_params":
                continue
            if hasattr(v, "shape"):
                assert v.ndim == 1, f"{k} has ndim={v.ndim}, expected 1 (selected)"

    def test_bfgs_improves_objective(self, bfgs_run_fingerprint):
        """Optimized params should have non-degenerate objective (not NaN/zero)."""
        fp = deepcopy(bfgs_run_fingerprint)
        fp["optimisation_settings"]["bfgs_settings"]["maxiter"] = 10

        _, metadata = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
            return_training_metadata=True,
        )

        # Objective should be finite and non-zero
        obj = metadata["final_objective"]
        assert np.isfinite(obj), f"Objective is not finite: {obj}"
        assert obj != 0.0, "Objective is exactly zero (degenerate)"

    def test_bfgs_returns_metadata(self, bfgs_run_fingerprint):
        """return_training_metadata=True returns (params, metadata) with correct structure."""
        fp = deepcopy(bfgs_run_fingerprint)
        fp["optimisation_settings"]["n_parameter_sets"] = 2

        result = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
            return_training_metadata=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        params, metadata = result
        assert isinstance(params, dict)
        assert isinstance(metadata, dict)

        # Check method tag
        assert metadata["method"] == "bfgs"

        # Check required metadata keys
        required_keys = [
            "epochs_trained",
            "best_train_metrics",
            "best_continuous_test_metrics",
            "best_param_idx",
            "best_final_reserves",
            "best_final_weights",
            "run_fingerprint",
            "checkpoint_returns",
            "selection_method",
            "selection_metric",
        ]
        for key in required_keys:
            assert key in metadata, f"Missing metadata key: {key}"

        # BFGS-specific keys
        assert "status_per_set" in metadata
        assert "objective_per_set" in metadata
        assert len(metadata["status_per_set"]) == 2
        assert len(metadata["objective_per_set"]) == 2

        # Checkpoint returns should be None (BFGS doesn't checkpoint)
        assert metadata["checkpoint_returns"] is None

        # best_train_metrics should be a list (one per param set)
        assert isinstance(metadata["best_train_metrics"], list)

    def test_bfgs_with_validation_fraction(self, bfgs_run_fingerprint):
        """BFGS with val_fraction > 0 uses best_val selection."""
        fp = deepcopy(bfgs_run_fingerprint)
        # Need longer window so val split exceeds 1 chunk_period (1440 min)
        fp["endDateString"] = "2023-01-15 00:00:00"
        fp["endTestDateString"] = "2023-01-20 00:00:00"
        fp["optimisation_settings"]["val_fraction"] = 0.2
        fp["optimisation_settings"]["n_parameter_sets"] = 2

        params, metadata = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
            return_training_metadata=True,
        )

        assert params is not None
        assert metadata["method"] == "bfgs"
        assert metadata["selection_method"] == "best_val"
        assert metadata["best_val_metrics"] is not None
        assert isinstance(metadata["best_val_metrics"], list)
        assert len(metadata["best_val_metrics"]) == 2

    def test_bfgs_config_defaults(self):
        """bfgs_settings defaults are applied via recursive_default_set."""
        fp = {
            "optimisation_settings": {
                "method": "bfgs",
            }
        }
        recursive_default_set(fp, run_fingerprint_defaults)

        bfgs = fp["optimisation_settings"]["bfgs_settings"]
        assert bfgs["maxiter"] == 100
        assert bfgs["tol"] == 1e-6
        assert bfgs["n_evaluation_points"] == 20

    def test_bfgs_memory_budget_caps_param_sets(self, bfgs_run_fingerprint):
        """memory_budget in bfgs_settings caps n_parameter_sets."""
        fp = deepcopy(bfgs_run_fingerprint)
        fp["optimisation_settings"]["n_parameter_sets"] = 4
        fp["optimisation_settings"]["bfgs_settings"]["n_evaluation_points"] = 2
        # Budget of 4 with 2 eval points → max 2 param sets
        fp["optimisation_settings"]["bfgs_settings"]["memory_budget"] = 4

        _, metadata = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
            return_training_metadata=True,
        )

        # Should have been capped to 2 param sets (budget=4 // n_eval=2)
        assert len(metadata["status_per_set"]) == 2
        assert len(metadata["objective_per_set"]) == 2
