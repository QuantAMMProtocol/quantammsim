"""Tests for BFGS optimizer integration in train_on_historic_data.

Tests follow the same fixture/pattern as test_jax_runners_comprehensive.py.
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
    """Run fingerprint configured for BFGS optimization.

    Uses dates within test data range (2022-10-01 to 2023-07-01).
    """
    return {
        "rule": "momentum",
        "tokens": ["ETH", "USDC"],
        "subsidary_pools": [],
        "n_assets": 2,
        "bout_offset": 1440,
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
                "maxiter": 10,
                "tol": 1e-6,
                "n_evaluation_points": 5,
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
        fp["optimisation_settings"]["n_parameter_sets"] = 1

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
        """Multi-start BFGS with n_parameter_sets=3 returns correct shapes."""
        fp = deepcopy(bfgs_run_fingerprint)
        fp["optimisation_settings"]["n_parameter_sets"] = 3

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
        """Optimized params should have better objective than initial."""
        from quantammsim.training.backpropagation import (
            batched_partial_training_step_factory,
            batched_objective_factory,
        )
        from quantammsim.runners.jax_runner_utils import generate_evaluation_points
        from quantammsim.pools.creator import create_pool
        from quantammsim.utils.data_processing.historic_data_utils import get_data_dict
        from quantammsim.runners.jax_runner_utils import (
            get_unique_tokens,
            create_static_dict,
            get_sig_variations,
            Hashabledict,
        )
        from quantammsim.core_simulator.forward_pass import forward_pass
        from jax.tree_util import Partial
        from jax import jit, vmap

        fp = deepcopy(bfgs_run_fingerprint)
        fp["optimisation_settings"]["n_parameter_sets"] = 1
        fp["optimisation_settings"]["bfgs_settings"]["maxiter"] = 20
        recursive_default_set(fp, run_fingerprint_defaults)

        unique_tokens = get_unique_tokens(fp)
        n_tokens = len(unique_tokens)
        data_dict = get_data_dict(
            unique_tokens, fp,
            data_kind="historic",
            root=TEST_DATA_DIR,
            max_memory_days=fp["max_memory_days"],
            start_date_string=fp["startDateString"],
            end_time_string=fp["endDateString"],
            start_time_test_string=fp["endDateString"],
            end_time_test_string=fp["endTestDateString"],
            do_test_period=True,
        )
        bout_length_window = data_dict["bout_length"] - fp["bout_offset"]

        pool = create_pool("momentum")
        initial_params_spec = {
            "initial_memory_length": fp["initial_memory_length"],
            "initial_memory_length_delta": fp["initial_memory_length_delta"],
            "initial_k_per_day": fp["initial_k_per_day"],
            "initial_weights_logits": fp["initial_weights_logits"],
            "initial_log_amplitude": fp["initial_log_amplitude"],
            "initial_raw_width": fp["initial_raw_width"],
            "initial_raw_exponents": fp["initial_raw_exponents"],
            "initial_pre_exp_scaling": fp["initial_pre_exp_scaling"],
            "min_weights_per_asset": None,
            "max_weights_per_asset": None,
        }
        params = pool.init_parameters(initial_params_spec, fp, n_tokens, 1)
        all_sig_variations = get_sig_variations(n_tokens)
        static_dict = create_static_dict(
            fp,
            bout_length=bout_length_window,
            all_sig_variations=all_sig_variations,
            overrides={"n_assets": n_tokens, "training_data_kind": "historic", "do_trades": False},
        )
        partial_training_step = Partial(
            forward_pass,
            prices=data_dict["prices"],
            static_dict=Hashabledict(static_dict),
            pool=pool,
        )
        batched_pts = batched_partial_training_step_factory(partial_training_step)
        batched_obj = batched_objective_factory(batched_pts)

        eval_starts = generate_evaluation_points(
            data_dict["start_idx"], data_dict["end_idx"],
            bout_length_window, 5, bout_length_window // 2, 42,
        )
        fixed_starts = jnp.array([(s, 0) for s in eval_starts], dtype=jnp.int32)

        # Squeeze batch dim for single param set
        params_single = {}
        for k, v in params.items():
            if k == "subsidary_params":
                params_single[k] = v
            elif hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == 1:
                params_single[k] = v[0]
            else:
                params_single[k] = v

        initial_obj = float(batched_obj(params_single, fixed_starts))

        # Now run BFGS
        result = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
        )

        optimized_obj = float(batched_obj(result, fixed_starts))

        # BFGS should improve (or at least not worsen) the objective
        assert optimized_obj >= initial_obj - 1e-6, (
            f"BFGS did not improve: initial={initial_obj:.6f}, optimized={optimized_obj:.6f}"
        )

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
        """BFGS with val_fraction > 0 produces validation metrics and uses best_val selection."""
        fp = deepcopy(bfgs_run_fingerprint)
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
