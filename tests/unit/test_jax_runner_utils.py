"""
Tests for the jax_runner_utils module.

Covers:
- Hashabledict and NestedHashabledict classes
- Utility functions: get_sig_variations, split_list, get_unique_tokens
- Gradient utilities: has_nan_grads, nan_rollback
- Permutation utilities: invert_permutation, permute_list_of_params
"""

import warnings

import pytest
import numpy as np
import jax.numpy as jnp


class TestHashabledict:
    """Tests for Hashabledict class."""

    def test_basic_hash(self):
        """Test that Hashabledict can be hashed and hash is stable."""
        from quantammsim.runners.jax_runner_utils import Hashabledict

        d = Hashabledict({"a": 1, "b": 2})
        h1 = hash(d)
        h2 = hash(d)
        assert isinstance(h1, int)
        assert h1 == h2, "Hash should be stable across calls"

    def test_equal_dicts_same_hash(self):
        """Test that equal dicts have same hash regardless of insertion order."""
        from quantammsim.runners.jax_runner_utils import Hashabledict

        d1 = Hashabledict({"a": 1, "b": 2})
        d2 = Hashabledict({"b": 2, "a": 1})
        assert hash(d1) == hash(d2)
        assert d1 == d2

    def test_different_dicts_not_equal(self):
        """Test that different dicts are not equal (hash collision allowed)."""
        from quantammsim.runners.jax_runner_utils import Hashabledict

        d1 = Hashabledict({"a": 1, "b": 2})
        d2 = Hashabledict({"a": 1, "b": 3})
        # Note: hash collision is allowed, so we only test equality
        assert d1 != d2

    def test_can_use_as_dict_key(self):
        """Test that Hashabledict can be used as a dictionary key."""
        from quantammsim.runners.jax_runner_utils import Hashabledict

        d1 = Hashabledict({"a": 1, "b": 2})
        outer = {d1: "value"}
        assert outer[d1] == "value"

    def test_nested_list_values(self):
        """Test hashing with nested list values."""
        from quantammsim.runners.jax_runner_utils import Hashabledict

        d1 = Hashabledict({"a": [1, 2, 3], "b": [[4, 5], [6, 7]]})
        d2 = Hashabledict({"a": [1, 2, 3], "b": [[4, 5], [6, 7]]})
        assert hash(d1) == hash(d2)

    def test_nested_dict_values(self):
        """Test hashing with nested dict values."""
        from quantammsim.runners.jax_runner_utils import Hashabledict

        d1 = Hashabledict({"a": 1, "nested": {"x": 10, "y": 20}})
        d2 = Hashabledict({"a": 1, "nested": {"y": 20, "x": 10}})
        assert hash(d1) == hash(d2)


class TestCreateStaticDictExcludesParamInitFields:
    """Regression tests: create_static_dict must exclude parameter-init fields.

    The flask /api/runSimulation endpoint crashed with
    'unhashable type: jaxlib.xla_extension.ArrayImpl' because
    initial_weights_logits (a JAX array) leaked through create_static_dict
    into a Hashabledict. These fields are only used to build initial_params,
    never read from static_dict during forward passes.
    """

    def _make_fingerprint(self, **overrides):
        """Minimal run_fingerprint with param-init fields set to JAX arrays."""
        fp = {
            "tokens": ["ETH", "USDC"],
            "n_assets": 2,
            "rule": "balancer",
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "initial_pool_value": 60000000,
            "fees": 0.0,
            "arb_fees": 0.0,
            "gas_cost": 0.0,
            "maximum_change": 0.0003,
            "do_arb": True,
            "arb_frequency": 1,
            "use_alt_lamb": False,
            "use_pre_exp_scaling": True,
            "max_memory_days": 365,
            # param-init fields that should NOT appear in static_dict
            "initial_weights_logits": jnp.array([-0.69314718, -0.69314718]),
            "initial_memory_length": 10.0,
            "initial_memory_length_delta": 0.0,
            "initial_k_per_day": 20,
            "initial_log_amplitude": 0.0,
            "initial_raw_width": 0.0,
            "initial_raw_exponents": 0.0,
            "initial_pre_exp_scaling": 0.5,
        }
        fp.update(overrides)
        return fp

    def test_initial_weights_logits_excluded(self):
        """initial_weights_logits must not leak into static_dict."""
        from quantammsim.runners.jax_runner_utils import create_static_dict

        fp = self._make_fingerprint()
        static = create_static_dict(fp, bout_length=10080)
        assert "initial_weights_logits" not in static

    def test_all_param_init_fields_excluded(self):
        """All initial_* param-init fields must be excluded from static_dict."""
        from quantammsim.runners.jax_runner_utils import create_static_dict

        fp = self._make_fingerprint()
        static = create_static_dict(fp, bout_length=10080)

        param_init_fields = [
            "initial_weights_logits",
            "initial_memory_length",
            "initial_memory_length_delta",
            "initial_k_per_day",
            "initial_log_amplitude",
            "initial_raw_width",
            "initial_raw_exponents",
            "initial_pre_exp_scaling",
        ]
        for field in param_init_fields:
            assert field not in static, f"{field} should be excluded from static_dict"

    def test_initial_pool_value_preserved(self):
        """initial_pool_value IS used by pool classes and must stay."""
        from quantammsim.runners.jax_runner_utils import create_static_dict

        fp = self._make_fingerprint()
        static = create_static_dict(fp, bout_length=10080)
        assert "initial_pool_value" in static
        assert static["initial_pool_value"] == 60000000

    def test_static_dict_is_hashable_with_real_fingerprint(self):
        """End-to-end: static_dict from a realistic fingerprint must be hashable."""
        from quantammsim.runners.jax_runner_utils import create_static_dict, Hashabledict

        fp = self._make_fingerprint()
        static = create_static_dict(fp, bout_length=10080)
        hd = Hashabledict(static)
        h = hash(hd)
        assert isinstance(h, int)

    def test_unknown_array_fields_dropped_with_warning(self):
        """Arrays not in _TRAINING_ONLY_FIELDS are dropped with a warning.

        Ensures the guard in create_static_dict catches future array-typed
        fields that aren't yet in the exclusion list.
        """
        from quantammsim.runners.jax_runner_utils import create_static_dict

        fp = self._make_fingerprint()
        fp["surprise_array"] = jnp.array([1.0, 2.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            static = create_static_dict(fp, bout_length=10080)

        assert "surprise_array" not in static
        assert any("surprise_array" in str(warning.message) for warning in w)


class TestNestedHashabledict:
    """Tests for NestedHashabledict class."""

    def test_basic_hash(self):
        """Test that NestedHashabledict can be hashed."""
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        d = NestedHashabledict({"a": 1, "b": 2})
        h = hash(d)
        assert isinstance(h, int)

    def test_nested_dicts_converted(self):
        """Test that nested dicts are automatically converted and values preserved."""
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        d = NestedHashabledict({"outer": {"inner": {"deep": 1}}})
        assert isinstance(d["outer"], NestedHashabledict)
        assert isinstance(d["outer"]["inner"], NestedHashabledict)
        # Verify values are preserved through conversion
        assert d["outer"]["inner"]["deep"] == 1

    def test_nested_list_of_dicts_converted(self):
        """Test that dicts in lists are converted and values preserved."""
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        d = NestedHashabledict({"items": [{"a": 1}, {"b": 2}]})
        assert isinstance(d["items"][0], NestedHashabledict)
        assert isinstance(d["items"][1], NestedHashabledict)
        # Verify values are preserved
        assert d["items"][0]["a"] == 1
        assert d["items"][1]["b"] == 2

    def test_equal_nested_dicts_same_hash(self):
        """Test that equal nested dicts have same hash."""
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        d1 = NestedHashabledict({"outer": {"a": 1, "b": 2}})
        d2 = NestedHashabledict({"outer": {"b": 2, "a": 1}})
        assert hash(d1) == hash(d2)
        assert d1 == d2

    def test_equality_with_regular_dict(self):
        """Test equality comparison with regular dict."""
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        d1 = NestedHashabledict({"a": 1, "b": 2})
        d2 = {"a": 1, "b": 2}
        assert d1 == d2

    def test_equality_with_non_dict_returns_false(self):
        """Test that equality with non-dict returns False."""
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        d = NestedHashabledict({"a": 1})
        assert d != "not a dict"
        assert d != 123
        assert d != [1, 2, 3]


class TestGetSigVariations:
    """Tests for get_sig_variations function."""

    def test_two_assets(self):
        """Test sig variations for 2 assets."""
        from quantammsim.runners.jax_runner_utils import get_sig_variations

        result = get_sig_variations(2)
        # For 2 assets, should have exactly 2 variations: (1,-1) and (-1,1)
        assert len(result) == 2
        assert (1, -1) in result
        assert (-1, 1) in result

    def test_three_assets(self):
        """Test sig variations for 3 assets."""
        from quantammsim.runners.jax_runner_utils import get_sig_variations

        result = get_sig_variations(3)
        # For 3 assets: 3 choose 2 * 2 = 6 variations
        assert len(result) == 6
        # Each should have exactly one +1 and one -1, rest zeros
        for sig in result:
            assert sum(1 for x in sig if x == 1) == 1
            assert sum(1 for x in sig if x == -1) == 1
            assert sum(1 for x in sig if x == 0) == 1

    def test_three_assets_exact_values(self):
        """Verify exact sig variations for 3 assets."""
        from quantammsim.runners.jax_runner_utils import get_sig_variations

        result = set(get_sig_variations(3))
        # All valid pairs of (source, dest) for 3 assets
        expected = {
            (1, -1, 0), (-1, 1, 0),  # assets 0 and 1
            (1, 0, -1), (-1, 0, 1),  # assets 0 and 2
            (0, 1, -1), (0, -1, 1),  # assets 1 and 2
        }
        assert result == expected

    def test_four_assets(self):
        """Test sig variations for 4 assets."""
        from quantammsim.runners.jax_runner_utils import get_sig_variations

        result = get_sig_variations(4)
        # For 4 assets: 4 choose 2 * 2 = 12 variations
        assert len(result) == 12
        # Verify all unique
        assert len(set(result)) == 12

    def test_result_is_tuple_of_tuples(self):
        """Test that result is tuple of tuples (hashable)."""
        from quantammsim.runners.jax_runner_utils import get_sig_variations

        result = get_sig_variations(3)
        assert isinstance(result, tuple)
        assert all(isinstance(sig, tuple) for sig in result)
        # Should be hashable
        hash(result)


class TestSplitList:
    """Tests for split_list function."""

    def test_even_split(self):
        """Test splitting list evenly."""
        from quantammsim.runners.jax_runner_utils import split_list

        result = split_list([1, 2, 3, 4, 5, 6], 3)
        assert len(result) == 3
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_uneven_split(self):
        """Test splitting list with remainder."""
        from quantammsim.runners.jax_runner_utils import split_list

        result = split_list([1, 2, 3, 4, 5], 2)
        assert len(result) == 2
        # Remainder distributed to first sublists
        assert len(result[0]) == 3
        assert len(result[1]) == 2

    def test_single_split(self):
        """Test splitting into single list."""
        from quantammsim.runners.jax_runner_utils import split_list

        result = split_list([1, 2, 3], 1)
        assert result == [[1, 2, 3]]

    def test_more_splits_than_elements(self):
        """Test splitting with more splits than elements."""
        from quantammsim.runners.jax_runner_utils import split_list

        result = split_list([1, 2], 5)
        assert len(result) == 5
        # First 2 have elements, rest are empty
        assert sum(len(x) for x in result) == 2
        # Verify all original elements are present
        all_elements = [item for sublist in result for item in sublist]
        assert sorted(all_elements) == [1, 2]

    def test_split_preserves_order(self):
        """Test that split_list preserves element order."""
        from quantammsim.runners.jax_runner_utils import split_list

        original = [1, 2, 3, 4, 5, 6, 7, 8]
        result = split_list(original, 3)
        # Flatten and verify order is preserved
        flattened = [item for sublist in result for item in sublist]
        assert flattened == original


class TestGetUniqueTokens:
    """Tests for get_unique_tokens function."""

    def test_simple_tokens(self):
        """Test with simple token list."""
        from quantammsim.runners.jax_runner_utils import get_unique_tokens

        fp = {"tokens": ["BTC", "ETH"], "subsidary_pools": []}
        result = get_unique_tokens(fp)
        assert result == ["BTC", "ETH"]

    def test_with_subsidiary_pools(self):
        """Test with subsidiary pools."""
        from quantammsim.runners.jax_runner_utils import get_unique_tokens

        fp = {
            "tokens": ["BTC", "ETH"],
            "subsidary_pools": [{"tokens": ["ETH", "DAI"]}],
        }
        result = get_unique_tokens(fp)
        assert result == ["BTC", "DAI", "ETH"]  # Sorted

    def test_duplicates_removed(self):
        """Test that duplicates are removed."""
        from quantammsim.runners.jax_runner_utils import get_unique_tokens

        fp = {
            "tokens": ["BTC", "ETH", "BTC"],
            "subsidary_pools": [{"tokens": ["ETH", "ETH"]}],
        }
        result = get_unique_tokens(fp)
        assert result == ["BTC", "ETH"]


class TestHasNanGrads:
    """Tests for has_nan_grads function."""

    def test_no_nans(self):
        """Test with no NaN values."""
        from quantammsim.runners.jax_runner_utils import has_nan_grads

        grads = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
        assert not has_nan_grads(grads)

    def test_with_nans(self):
        """Test with NaN values."""
        from quantammsim.runners.jax_runner_utils import has_nan_grads

        grads = {"a": jnp.array([1.0, jnp.nan]), "b": jnp.array([3.0, 4.0])}
        assert has_nan_grads(grads)

    def test_nested_structure(self):
        """Test with nested gradient structure."""
        from quantammsim.runners.jax_runner_utils import has_nan_grads

        grads = {
            "a": jnp.array([[1.0, 2.0], [3.0, jnp.nan]]),
        }
        assert has_nan_grads(grads)

    def test_with_inf(self):
        """Test that inf is not considered NaN (inf is valid, just large)."""
        from quantammsim.runners.jax_runner_utils import has_nan_grads

        grads = {"a": jnp.array([1.0, jnp.inf]), "b": jnp.array([3.0, -jnp.inf])}
        # inf is not NaN - function should return False
        assert not has_nan_grads(grads)

    def test_empty_grads(self):
        """Test with empty gradient dict."""
        from quantammsim.runners.jax_runner_utils import has_nan_grads

        grads = {}
        assert not has_nan_grads(grads)


class TestNanRollback:
    """Tests for nan_rollback function."""

    def test_rollback_on_nan(self):
        """Test that parameters are rolled back when gradients have NaN."""
        from quantammsim.runners.jax_runner_utils import nan_rollback

        grads = {"log_k": jnp.array([[1.0, jnp.nan]])}
        params = {"log_k": jnp.array([[0.1, 0.2]])}
        old_params = {"log_k": jnp.array([[0.05, 0.15]])}

        result = nan_rollback(grads, params, old_params)
        # Should rollback to old_params because of NaN
        np.testing.assert_array_equal(result["log_k"], old_params["log_k"])

    def test_no_rollback_without_nan(self):
        """Test that parameters are unchanged when no NaN in gradients."""
        from quantammsim.runners.jax_runner_utils import nan_rollback

        grads = {"log_k": jnp.array([[1.0, 2.0]])}
        params = {"log_k": jnp.array([[0.1, 0.2]])}
        old_params = {"log_k": jnp.array([[0.05, 0.15]])}

        result = nan_rollback(grads, params, old_params)
        np.testing.assert_array_equal(result["log_k"], params["log_k"])

    def test_rollback_multiple_keys(self):
        """Test rollback with multiple parameter keys."""
        from quantammsim.runners.jax_runner_utils import nan_rollback

        grads = {
            "log_k": jnp.array([[1.0, jnp.nan]]),  # Has NaN
            "logit_lamb": jnp.array([[2.0, 3.0]]),  # No NaN
        }
        params = {
            "log_k": jnp.array([[0.1, 0.2]]),
            "logit_lamb": jnp.array([[0.5, 0.6]]),
        }
        old_params = {
            "log_k": jnp.array([[0.05, 0.15]]),
            "logit_lamb": jnp.array([[0.4, 0.5]]),
        }

        result = nan_rollback(grads, params, old_params)
        # Both should rollback because any NaN triggers full rollback
        np.testing.assert_array_equal(result["log_k"], old_params["log_k"])
        # Note: nan_rollback rolls back ALL params if ANY have NaN
        np.testing.assert_array_equal(result["logit_lamb"], old_params["logit_lamb"])


class TestInvertPermutation:
    """Tests for invert_permutation function."""

    def test_simple_permutation(self):
        """Test inverting a simple permutation."""
        from quantammsim.runners.jax_runner_utils import invert_permutation

        perm = np.array([2, 0, 1])  # 0->2, 1->0, 2->1
        inv = invert_permutation(perm)
        # Applying perm then inv should give identity
        np.testing.assert_array_equal(inv, [1, 2, 0])

    def test_identity_permutation(self):
        """Test inverting identity permutation."""
        from quantammsim.runners.jax_runner_utils import invert_permutation

        perm = np.array([0, 1, 2])
        inv = invert_permutation(perm)
        np.testing.assert_array_equal(inv, [0, 1, 2])

    def test_roundtrip(self):
        """Test that double inversion gives original."""
        from quantammsim.runners.jax_runner_utils import invert_permutation

        perm = np.array([3, 1, 0, 2])
        inv = invert_permutation(perm)
        double_inv = invert_permutation(inv)
        np.testing.assert_array_equal(double_inv, perm)

    def test_inverse_property(self):
        """Test the mathematical property: applying perm then inv gives identity."""
        from quantammsim.runners.jax_runner_utils import invert_permutation

        perm = np.array([2, 0, 3, 1])
        inv = invert_permutation(perm)

        # Applying perm then inv should give identity: inv[perm[i]] == i
        n = len(perm)
        for i in range(n):
            assert inv[perm[i]] == i, f"inv[perm[{i}]] = inv[{perm[i]}] = {inv[perm[i]]} != {i}"

        # Also verify: perm[inv[i]] == i
        for i in range(n):
            assert perm[inv[i]] == i, f"perm[inv[{i}]] = perm[{inv[i]}] = {perm[inv[i]]} != {i}"


class TestCreateStaticDict:
    """Tests for create_static_dict function."""

    def test_basic_creation(self):
        """Test basic static_dict creation."""
        from quantammsim.runners.jax_runner_utils import create_static_dict

        fp = {
            "tokens": ["BTC", "ETH"],
            "n_assets": 2,
            "rule": "momentum",
            "chunk_period": 60,
            "weight_interpolation_period": 60,
            "initial_pool_value": 1000000.0,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "maximum_change": 0.01,
            "do_arb": True,
            "arb_frequency": 1,
            "subsidary_pools": [],
        }

        result = create_static_dict(fp, bout_length=1440)

        assert result["bout_length"] == 1440
        assert result["n_assets"] == 2
        assert isinstance(result["tokens"], tuple)
        assert result["tokens"] == ("BTC", "ETH")

    def test_excludes_training_fields(self):
        """Test that training-only fields are excluded."""
        from quantammsim.runners.jax_runner_utils import create_static_dict

        fp = {
            "tokens": ["BTC", "ETH"],
            "optimisation_settings": {"lr": 0.01},  # Should be excluded
            "startDateString": "2023-01-01",  # Should be excluded
            "chunk_period": 60,
            "weight_interpolation_period": 60,
            "initial_pool_value": 1000000.0,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "maximum_change": 0.01,
            "do_arb": True,
            "arb_frequency": 1,
            "subsidary_pools": [],
        }

        result = create_static_dict(fp, bout_length=1440)

        assert "optimisation_settings" not in result
        assert "startDateString" not in result

    def test_with_overrides(self):
        """Test that overrides are applied."""
        from quantammsim.runners.jax_runner_utils import create_static_dict

        fp = {
            "tokens": ["BTC", "ETH"],
            "chunk_period": 60,
            "weight_interpolation_period": 60,
            "initial_pool_value": 1000000.0,
            "fees": 0.0,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "maximum_change": 0.01,
            "do_arb": True,
            "arb_frequency": 1,
            "subsidary_pools": [],
        }

        result = create_static_dict(fp, bout_length=1440, overrides={"fees": 0.003})

        assert result["fees"] == 0.003


class TestGetRunLocation:
    """Tests for get_run_location function."""

    def test_produces_hash(self):
        """Test that get_run_location produces a hash string."""
        from quantammsim.runners.jax_runner_utils import get_run_location

        fp = {"a": 1, "b": 2}
        result = get_run_location(fp)

        assert result.startswith("run_")
        assert len(result) > 10

    def test_deterministic(self):
        """Test that same fingerprint produces same hash."""
        from quantammsim.runners.jax_runner_utils import get_run_location

        fp = {"a": 1, "b": 2}
        result1 = get_run_location(fp)
        result2 = get_run_location(fp)

        assert result1 == result2

    def test_different_fingerprints_different_hash(self):
        """Test that different fingerprints produce different hashes."""
        from quantammsim.runners.jax_runner_utils import get_run_location

        fp1 = {"a": 1, "b": 2}
        fp2 = {"a": 1, "b": 3}

        assert get_run_location(fp1) != get_run_location(fp2)
