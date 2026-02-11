"""Tests for scan-based training loop components.

Tests for:
- init_tracker_state: Pure-function tracker initialization
- update_tracker_state: Pure-function tracker update (scan-compatible)
- nan_reinit_from_bank: Bank-based NaN reinit for scan compatibility
- Scan vs Python loop equivalence

These tests are written BEFORE implementation (TDD).
They will FAIL until the scan functions are implemented.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from copy import deepcopy
from pathlib import Path

from tests.conftest import TEST_DATA_DIR

# Tolerance for pinned value comparisons.
RTOL = 1e-6

# Re-use the training fingerprint factory from the regression tests
# to guarantee identical config.
from tests.unit.test_training_loop_regression import (
    _make_training_fingerprint,
    PINNED_TRAINING_OBJECTIVE,
    PINNED_MR_TRAINING_OBJECTIVE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mock_params():
    """Batched params: (n_parameter_sets=4, n_tokens=2)."""
    return {
        "log_k": jnp.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [2.0, 2.0]]),
        "logit_lamb": jnp.array([[-0.22, -0.22], [-0.5, -0.5], [0.1, 0.1], [-1.0, -1.0]]),
        "initial_weights_logits": jnp.array([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2], [0.0, 0.0]]),
        "log_amplitude": jnp.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1], [0.5, 0.5]]),
        "raw_width": jnp.array([[0.0, 0.0], [-0.5, -0.5], [0.5, 0.5], [0.0, 0.0]]),
        "raw_exponents": jnp.array([[1.0, 1.0], [1.5, 1.5], [0.5, 0.5], [2.0, 2.0]]),
        "raw_pre_exp_scaling": jnp.array([[0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [0.5, 0.5]]),
    }


@pytest.fixture(scope="module")
def mock_params_with_nans():
    """Params with NaN in sets 1 and 3."""
    return {
        "log_k": jnp.array([[3.0, 3.0], [float("nan"), 4.0], [5.0, 5.0], [2.0, 2.0]]),
        "logit_lamb": jnp.array([[-0.22, -0.22], [-0.5, -0.5], [0.1, 0.1], [-1.0, -1.0]]),
        "initial_weights_logits": jnp.array([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2], [0.0, 0.0]]),
        "log_amplitude": jnp.array([[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1], [0.5, 0.5]]),
        "raw_width": jnp.array([[0.0, 0.0], [-0.5, -0.5], [0.5, 0.5], [0.0, 0.0]]),
        "raw_exponents": jnp.array([[1.0, 1.0], [1.5, 1.5], [0.5, 0.5], [float("nan"), float("nan")]]),
        "raw_pre_exp_scaling": jnp.array([[0.5, 0.5], [0.3, 0.3], [0.7, 0.7], [0.5, 0.5]]),
    }


# ── init_tracker_state tests ─────────────────────────────────────────────────


class TestInitTrackerState:
    """Test init_tracker_state returns correct structure for scan carry."""

    def test_returns_dict_with_expected_keys(self, mock_params):
        from quantammsim.runners.jax_runner_utils import init_tracker_state
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        expected_keys = {
            "best_metric_value", "best_iteration", "best_param_idx",
            "best_params", "best_train_metrics", "best_val_metrics",
            "best_test_metrics",
        }
        assert set(state.keys()) == expected_keys

    def test_all_values_are_jax_arrays(self, mock_params):
        from quantammsim.runners.jax_runner_utils import init_tracker_state
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        for key, val in state.items():
            if key == "best_params":
                # best_params is a pytree of arrays
                for k, v in val.items():
                    assert isinstance(v, jnp.ndarray), f"best_params[{k}] is not a jax array"
            else:
                assert isinstance(val, jnp.ndarray), f"{key} is not a jax array"

    def test_no_none_values(self, mock_params):
        from quantammsim.runners.jax_runner_utils import init_tracker_state
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        for key, val in state.items():
            assert val is not None, f"{key} is None"

    def test_best_metric_value_is_neg_inf(self, mock_params):
        from quantammsim.runners.jax_runner_utils import init_tracker_state
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        assert state["best_metric_value"] == -jnp.inf

    def test_best_params_matches_input_structure(self, mock_params):
        from quantammsim.runners.jax_runner_utils import init_tracker_state
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        assert set(state["best_params"].keys()) == set(mock_params.keys())
        for key in mock_params:
            assert state["best_params"][key].shape == mock_params[key].shape

    def test_metric_arrays_shape(self, mock_params):
        from quantammsim.runners.jax_runner_utils import init_tracker_state
        state = init_tracker_state(mock_params, n_parameter_sets=4, n_metrics=12)
        assert state["best_train_metrics"].shape == (4, 12)
        assert state["best_val_metrics"].shape == (4, 12)
        assert state["best_test_metrics"].shape == (4, 12)


# ── update_tracker_state tests ───────────────────────────────────────────────


class TestUpdateTrackerState:
    """Test pure-function tracker matches BestParamsTracker class behavior."""

    def test_first_update_always_improves(self, mock_params):
        from quantammsim.runners.jax_runner_utils import (
            init_tracker_state, update_tracker_state,
        )
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        train_arr = jnp.ones((4, 12))  # All 1s
        val_arr = jnp.zeros((4, 12))
        test_arr = jnp.zeros((4, 12))
        new_state, improved = update_tracker_state(
            state, jnp.int32(0), mock_params,
            train_arr, val_arr, test_arr,
            sel_metric_idx=0, use_val=False, use_test=False,
        )
        assert bool(improved), "First update should always improve from -inf"
        assert float(new_state["best_metric_value"]) > float(-jnp.inf)

    def test_better_iteration_updates_best(self, mock_params):
        from quantammsim.runners.jax_runner_utils import (
            init_tracker_state, update_tracker_state,
        )
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        # First update: metric=1.0
        train1 = jnp.ones((4, 12))
        state, _ = update_tracker_state(
            state, jnp.int32(0), mock_params,
            train1, jnp.zeros((4, 12)), jnp.zeros((4, 12)),
            sel_metric_idx=0, use_val=False, use_test=False,
        )
        # Second update: metric=5.0 (better)
        params2 = tree_map(lambda x: x + 1.0, mock_params)
        train2 = jnp.full((4, 12), 5.0)
        state, improved = update_tracker_state(
            state, jnp.int32(1), params2,
            train2, jnp.zeros((4, 12)), jnp.zeros((4, 12)),
            sel_metric_idx=0, use_val=False, use_test=False,
        )
        assert bool(improved)
        assert int(state["best_iteration"]) == 1

    def test_worse_iteration_preserves_best(self, mock_params):
        from quantammsim.runners.jax_runner_utils import (
            init_tracker_state, update_tracker_state,
        )
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        # Good update
        train_good = jnp.full((4, 12), 10.0)
        state, _ = update_tracker_state(
            state, jnp.int32(0), mock_params,
            train_good, jnp.zeros((4, 12)), jnp.zeros((4, 12)),
            sel_metric_idx=0, use_val=False, use_test=False,
        )
        best_val = float(state["best_metric_value"])
        # Worse update
        params2 = tree_map(lambda x: x + 99.0, mock_params)
        train_bad = jnp.full((4, 12), 1.0)
        state, improved = update_tracker_state(
            state, jnp.int32(1), params2,
            train_bad, jnp.zeros((4, 12)), jnp.zeros((4, 12)),
            sel_metric_idx=0, use_val=False, use_test=False,
        )
        assert not bool(improved)
        np.testing.assert_allclose(
            float(state["best_metric_value"]), best_val, rtol=1e-10,
        )
        assert int(state["best_iteration"]) == 0

    def test_all_nan_metrics_no_crash(self, mock_params):
        from quantammsim.runners.jax_runner_utils import (
            init_tracker_state, update_tracker_state,
        )
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        nan_arr = jnp.full((4, 12), jnp.nan)
        # Should not crash
        new_state, improved = update_tracker_state(
            state, jnp.int32(0), mock_params,
            nan_arr, nan_arr, nan_arr,
            sel_metric_idx=0, use_val=False, use_test=False,
        )
        # NaN mean → doesn't improve from -inf (or is NaN)
        # Just verify no crash

    def test_matches_class_tracker_for_10_iterations(self, mock_params):
        """Run both pure-fn and class tracker with identical inputs,
        assert identical best_metric_value and best_param_idx."""
        from quantammsim.runners.jax_runner_utils import (
            init_tracker_state, update_tracker_state,
            BestParamsTracker, _nanargmax_jnp,
        )
        from quantammsim.utils.post_train_analysis import (
            _METRIC_KEYS, metrics_arr_to_dicts,
        )

        n_param_sets = 4
        rng = np.random.RandomState(42)

        # Pure-function tracker
        state = init_tracker_state(mock_params, n_parameter_sets=n_param_sets)

        # Class tracker (for comparison)
        class_tracker = BestParamsTracker(
            selection_method="best_train",
            metric="sharpe",
            min_threshold=0.0,
        )

        sel_idx = _METRIC_KEYS.index("sharpe")

        for i in range(10):
            # Generate random metrics and params
            train_arr = jnp.array(rng.randn(n_param_sets, 12))
            val_arr = jnp.zeros((n_param_sets, 12))
            test_arr = jnp.array(rng.randn(n_param_sets, 12))
            noise = rng.randn(*mock_params["log_k"].shape) * 0.1
            iter_params = tree_map(
                lambda x: x + jnp.array(noise[:x.shape[0], :x.shape[1]] if x.ndim == 2 else noise[:x.shape[0]]),
                mock_params,
            )

            # Update pure-function tracker
            state, _ = update_tracker_state(
                state, jnp.int32(i), iter_params,
                train_arr, val_arr, test_arr,
                sel_metric_idx=sel_idx, use_val=False, use_test=False,
            )

            # Update class tracker — needs dicts and continuous_outputs
            train_dicts = metrics_arr_to_dicts(train_arr)
            test_dicts = metrics_arr_to_dicts(test_arr)
            # Fake continuous_outputs with minimal structure
            fake_continuous = {
                "reserves": jnp.zeros((n_param_sets, 10, 2)),
                "weights": jnp.zeros((n_param_sets, 10, 2)),
            }
            class_tracker.update(
                iteration=i,
                params=iter_params,
                continuous_outputs=fake_continuous,
                train_metrics_list=train_dicts,
                continuous_test_metrics_list=test_dicts,
            )

        # Compare results
        np.testing.assert_allclose(
            float(state["best_metric_value"]),
            float(class_tracker.best_metric_value),
            rtol=1e-6,
            err_msg="Pure-fn and class tracker best_metric_value differ",
        )
        assert int(state["best_param_idx"]) == int(class_tracker.best_param_idx), (
            f"best_param_idx: pure-fn={int(state['best_param_idx'])}, "
            f"class={int(class_tracker.best_param_idx)}"
        )

    def test_val_selection_mode(self, mock_params):
        """When use_val=True, selection should use val_metrics not train."""
        from quantammsim.runners.jax_runner_utils import (
            init_tracker_state, update_tracker_state,
        )
        state = init_tracker_state(mock_params, n_parameter_sets=4)
        # Train metrics high, val metrics low
        train_arr = jnp.full((4, 12), 100.0)
        val_arr = jnp.full((4, 12), 1.0)
        state, _ = update_tracker_state(
            state, jnp.int32(0), mock_params,
            train_arr, val_arr, jnp.zeros((4, 12)),
            sel_metric_idx=0, use_val=True, use_test=False,
        )
        # best_metric_value should be ~1.0 (val mean), not ~100.0 (train mean)
        assert float(state["best_metric_value"]) < 50.0


# ── NaN reinit from bank tests ───────────────────────────────────────────────


class TestNanReinitFromBank:
    """Test bank-based NaN reinit for scan compatibility."""

    @pytest.fixture(scope="class")
    def nan_bank(self, mock_params):
        """Pre-generated replacement bank."""
        from quantammsim.runners.jax_runner_utils import (
            NAN_EXCLUDED_PARAM_KEYS, generate_nan_bank,
        )
        # Use generate_nan_bank if available, else build manually
        bank = {}
        bank_size = 4
        for i in range(bank_size):
            for key in mock_params:
                if key not in NAN_EXCLUDED_PARAM_KEYS:
                    replacement = jnp.ones_like(mock_params[key]) * (100.0 + i)
                    bank.setdefault(key, []).append(replacement)
        return {k: jnp.stack(v) for k, v in bank.items()}

    def test_replaces_nan_param_sets_only(self, mock_params_with_nans, nan_bank):
        from quantammsim.runners.jax_runner_utils import nan_reinit_from_bank
        has_nan = jnp.array([False, True, False, True])  # sets 1 and 3
        new_params, _ = nan_reinit_from_bank(
            mock_params_with_nans, has_nan, nan_bank, jnp.int32(0),
        )
        # Set 0 unchanged
        np.testing.assert_array_equal(
            new_params["log_k"][0], mock_params_with_nans["log_k"][0],
        )
        # Set 1 replaced (was NaN)
        assert not jnp.any(jnp.isnan(new_params["log_k"][1]))

    def test_clean_params_unchanged(self, mock_params, nan_bank):
        from quantammsim.runners.jax_runner_utils import nan_reinit_from_bank
        has_nan = jnp.array([False, False, False, False])
        new_params, count = nan_reinit_from_bank(
            mock_params, has_nan, nan_bank, jnp.int32(0),
        )
        for key in mock_params:
            np.testing.assert_array_equal(new_params[key], mock_params[key])

    def test_excluded_keys_preserved(self, mock_params_with_nans, nan_bank):
        from quantammsim.runners.jax_runner_utils import nan_reinit_from_bank
        has_nan = jnp.array([False, True, False, True])
        new_params, _ = nan_reinit_from_bank(
            mock_params_with_nans, has_nan, nan_bank, jnp.int32(0),
        )
        # initial_weights_logits is excluded — should be unchanged even for NaN sets
        np.testing.assert_array_equal(
            new_params["initial_weights_logits"],
            mock_params_with_nans["initial_weights_logits"],
        )

    def test_bank_wraps_around_at_size(self, mock_params_with_nans, nan_bank):
        from quantammsim.runners.jax_runner_utils import nan_reinit_from_bank
        has_nan = jnp.array([False, True, False, False])
        bank_size = list(nan_bank.values())[0].shape[0]
        # Use counter >= bank_size to trigger wraparound
        _, count = nan_reinit_from_bank(
            mock_params_with_nans, has_nan, nan_bank, jnp.int32(bank_size + 1),
        )
        assert int(count) == bank_size + 2  # incremented by 1

    def test_counter_increments_on_nan(self, mock_params_with_nans, nan_bank):
        from quantammsim.runners.jax_runner_utils import nan_reinit_from_bank
        has_nan = jnp.array([False, True, False, False])
        _, count = nan_reinit_from_bank(
            mock_params_with_nans, has_nan, nan_bank, jnp.int32(5),
        )
        assert int(count) == 6

    def test_counter_unchanged_when_clean(self, mock_params, nan_bank):
        from quantammsim.runners.jax_runner_utils import nan_reinit_from_bank
        has_nan = jnp.array([False, False, False, False])
        _, count = nan_reinit_from_bank(
            mock_params, has_nan, nan_bank, jnp.int32(5),
        )
        assert int(count) == 5

    def test_is_jit_compatible(self, mock_params_with_nans, nan_bank):
        """Must compile under jax.jit without error."""
        from quantammsim.runners.jax_runner_utils import nan_reinit_from_bank

        @jax.jit
        def _jit_reinit(params, has_nan, bank, count):
            return nan_reinit_from_bank(params, has_nan, bank, count)

        has_nan = jnp.array([False, True, False, True])
        new_params, count = _jit_reinit(
            mock_params_with_nans, has_nan, nan_bank, jnp.int32(0),
        )
        assert not jnp.any(jnp.isnan(new_params["log_k"][1]))


# ── Scan equivalence tests ───────────────────────────────────────────────────


class TestScanEquivalence:
    """CRITICAL: Scan path must match Python loop exactly.

    These tests run both the current Python loop and the new scan-based
    loop with identical configs and assert numeric equivalence.
    """

    def test_momentum_objective_matches_pinned(self):
        """Scan path must reproduce PINNED_TRAINING_OBJECTIVE."""
        from quantammsim.runners.jax_runners import train_on_historic_data
        fp = _make_training_fingerprint()
        _, metadata = train_on_historic_data(
            fp, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        np.testing.assert_allclose(
            metadata["final_objective"],
            PINNED_TRAINING_OBJECTIVE,
            rtol=RTOL,
            err_msg="Scan path: momentum objective drifted from pin",
        )

    def test_mean_reversion_objective_matches_pinned(self):
        """Scan path must reproduce PINNED_MR_TRAINING_OBJECTIVE."""
        from quantammsim.runners.jax_runners import train_on_historic_data
        fp = _make_training_fingerprint(rule="mean_reversion_channel")
        _, metadata = train_on_historic_data(
            fp, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        np.testing.assert_allclose(
            metadata["final_objective"],
            PINNED_MR_TRAINING_OBJECTIVE,
            rtol=RTOL,
            err_msg="Scan path: mean reversion objective drifted from pin",
        )

    def test_metadata_structure_matches(self):
        """Metadata from scan path has same keys as Python path."""
        from quantammsim.runners.jax_runners import train_on_historic_data
        fp = _make_training_fingerprint()
        _, metadata = train_on_historic_data(
            fp, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        required = {
            "epochs_trained", "final_objective",
            "best_train_metrics", "last_train_metrics",
            "best_continuous_test_metrics", "last_continuous_test_metrics",
            "best_iteration", "best_param_idx", "best_metric_value",
            "selection_method", "selection_metric",
            "best_final_reserves", "best_final_weights",
            "last_final_reserves", "last_final_weights",
            "checkpoint_returns",
        }
        missing = required - set(metadata.keys())
        assert not missing, f"Missing metadata keys: {missing}"

    def test_best_iteration_and_idx_match(self):
        """Scan must pick the same best iteration and param set."""
        from quantammsim.runners.jax_runners import train_on_historic_data
        from tests.unit.test_training_loop_regression import (
            TestTrainingMetadataRegression as TMR,
        )
        fp = _make_training_fingerprint()
        _, metadata = train_on_historic_data(
            fp, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        assert int(metadata["best_iteration"]) == TMR.PINNED_BEST_ITERATION
        assert int(metadata["best_param_idx"]) == TMR.PINNED_BEST_PARAM_IDX
        np.testing.assert_allclose(
            float(metadata["best_metric_value"]),
            TMR.PINNED_BEST_METRIC_VALUE,
            rtol=RTOL,
        )

    def test_scan_infrastructure_cached_across_calls(self):
        """Second call with same config must reuse cached scan infra."""
        from quantammsim.runners.jax_runners import (
            train_on_historic_data,
            _scan_infra_cache,
        )
        _scan_infra_cache.clear()
        fp = _make_training_fingerprint()
        train_on_historic_data(
            fp, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        assert len(_scan_infra_cache) == 1
        cached_key = list(_scan_infra_cache.keys())[0]
        cached_fn = _scan_infra_cache[cached_key][0]

        # Second call — same config, fresh fingerprint copy
        fp2 = _make_training_fingerprint()
        train_on_historic_data(
            fp2, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        # Cache should still have exactly 1 entry (same key hit)
        assert len(_scan_infra_cache) == 1
        assert _scan_infra_cache[cached_key][0] is cached_fn

    def test_cache_hit_produces_identical_results(self):
        """Cache-hit (2nd call) must produce bit-identical output to cache-miss (1st call).

        This catches subtle bugs where cached closures silently reference stale state.
        """
        from quantammsim.runners.jax_runners import (
            train_on_historic_data,
            _scan_infra_cache,
        )
        _scan_infra_cache.clear()

        fp1 = _make_training_fingerprint()
        _, meta1 = train_on_historic_data(
            fp1, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        fp2 = _make_training_fingerprint()
        _, meta2 = train_on_historic_data(
            fp2, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        # Numerical output must be identical
        np.testing.assert_equal(
            meta1["final_objective"], meta2["final_objective"],
        )
        np.testing.assert_equal(
            meta1["best_metric_value"], meta2["best_metric_value"],
        )
        assert meta1["best_iteration"] == meta2["best_iteration"]
        assert meta1["best_param_idx"] == meta2["best_param_idx"]
        assert meta1["epochs_trained"] == meta2["epochs_trained"]
        # Reserves and weights
        np.testing.assert_array_equal(
            np.asarray(meta1["best_final_reserves"]),
            np.asarray(meta2["best_final_reserves"]),
        )
        np.testing.assert_array_equal(
            np.asarray(meta1["best_final_weights"]),
            np.asarray(meta2["best_final_weights"]),
        )

    def test_multi_chunk_matches_single_chunk(self):
        """Multi-chunk (iterations_per_print=2) must match single-chunk (999999).

        This exercises:
        1. The outer while loop with multiple scan chunks
        2. The partial-chunk Python fallback (4 iterations, chunk_size=2 → no remainder,
           but with chunk_size=3 → 1 remaining → partial fallback)
        3. Per-chunk save accumulation and reset
        """
        from quantammsim.runners.jax_runners import train_on_historic_data

        # Single chunk baseline
        fp_single = _make_training_fingerprint()
        _, meta_single = train_on_historic_data(
            fp_single, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )
        # Multi-chunk: iterations_per_print=2, n_iterations=3 → total=4
        # chunk_size=2 → 2 full chunks of 2 each
        fp_multi = _make_training_fingerprint()
        _, meta_multi = train_on_historic_data(
            fp_multi, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=2,
        )

        np.testing.assert_allclose(
            meta_multi["final_objective"],
            meta_single["final_objective"],
            rtol=RTOL,
            err_msg="Multi-chunk vs single-chunk: final objective diverged",
        )
        np.testing.assert_allclose(
            meta_multi["best_metric_value"],
            meta_single["best_metric_value"],
            rtol=RTOL,
        )
        assert meta_multi["best_iteration"] == meta_single["best_iteration"]
        assert meta_multi["best_param_idx"] == meta_single["best_param_idx"]
        assert meta_multi["epochs_trained"] == meta_single["epochs_trained"]

    def test_partial_last_chunk_matches(self):
        """Partial last chunk (Python fallback) must match full scan.

        iterations_per_print=3, n_iterations=3 → total=4
        chunk_size=3 → first chunk: 3 via lax.scan, second chunk: 1 via Python loop
        """
        from quantammsim.runners.jax_runners import train_on_historic_data

        fp_single = _make_training_fingerprint()
        _, meta_single = train_on_historic_data(
            fp_single, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )

        fp_partial = _make_training_fingerprint()
        _, meta_partial = train_on_historic_data(
            fp_partial, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=3,  # 4 total: chunk of 3 + partial chunk of 1
        )

        np.testing.assert_allclose(
            meta_partial["final_objective"],
            meta_single["final_objective"],
            rtol=RTOL,
            err_msg="Partial-chunk vs single-chunk: final objective diverged",
        )
        np.testing.assert_allclose(
            meta_partial["best_metric_value"],
            meta_single["best_metric_value"],
            rtol=RTOL,
        )
        assert meta_partial["best_iteration"] == meta_single["best_iteration"]
        assert meta_partial["best_param_idx"] == meta_single["best_param_idx"]

    def test_chunk_size_1_matches(self):
        """iterations_per_print=1: every iteration is a separate scan chunk.

        This is the most stressful multi-chunk test — 4 separate scan(length=1)
        calls with Python accumulation between each. Also exercises the
        save_multi_params accumulator reset path 4 times.
        """
        from quantammsim.runners.jax_runners import train_on_historic_data

        fp_single = _make_training_fingerprint()
        _, meta_single = train_on_historic_data(
            fp_single, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=999999,
        )

        fp_chunk1 = _make_training_fingerprint()
        _, meta_chunk1 = train_on_historic_data(
            fp_chunk1, root=str(TEST_DATA_DIR), verbose=False,
            force_init=True, return_training_metadata=True,
            iterations_per_print=1,
        )

        np.testing.assert_allclose(
            meta_chunk1["final_objective"],
            meta_single["final_objective"],
            rtol=RTOL,
            err_msg="chunk_size=1 vs single-chunk: final objective diverged",
        )
        np.testing.assert_allclose(
            meta_chunk1["best_metric_value"],
            meta_single["best_metric_value"],
            rtol=RTOL,
        )
        assert meta_chunk1["best_iteration"] == meta_single["best_iteration"]
        assert meta_chunk1["best_param_idx"] == meta_single["best_param_idx"]
