"""Tests for EnsembleAveragingHook."""

import pytest
import jax
import jax.numpy as jnp
from jax import vmap, grad

from quantammsim.pools.creator import create_pool
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.runners.jax_runner_utils import NestedHashabledict
from quantammsim.core_simulator.param_utils import recursive_default_set


@pytest.fixture
def base_run_fingerprint():
    """Create a basic run fingerprint for testing.

    Returns a NestedHashabledict so it can be used with jitted functions.
    Uses chunk_period=10 (10-minute chunks) for reasonable test data sizes.
    """
    rf = {}
    recursive_default_set(rf, run_fingerprint_defaults)
    rf["tokens"] = ["BTC", "ETH", "USDC"]
    rf["rule"] = "momentum"
    rf["n_ensemble_members"] = 1  # Default, will be overridden in tests
    rf["chunk_period"] = 10  # 10-minute chunks for tests
    rf["weight_interpolation_period"] = 10  # Match chunk_period
    return NestedHashabledict(rf)


@pytest.fixture
def initial_values():
    """Create initial values dict for parameter initialization."""
    return {
        "initial_weights_logits": 1.0,
        "initial_memory_length": 10.0,
        "initial_memory_length_delta": 0.0,
        "initial_log_amplitude": 0.0,
        "initial_k_per_day": 20,
    }


class TestEnsembleAveragingHook:
    """Tests for the EnsembleAveragingHook class."""

    def test_create_ensemble_pool(self):
        """Test that ensemble pool can be created."""
        pool = create_pool("ensemble__momentum")
        assert pool is not None
        assert hasattr(pool, "calculate_rule_outputs")
        assert hasattr(pool, "init_base_parameters")

    def test_params_shape_no_ensemble(self, base_run_fingerprint, initial_values):
        """Test params shape when n_ensemble_members=1 (no ensembling)."""
        pool = create_pool("ensemble__momentum")
        base_run_fingerprint["n_ensemble_members"] = 1
        n_parameter_sets = 4
        n_assets = 3

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=n_parameter_sets,
        )

        # With n_ensemble_members=1, params should have shape (n_parameter_sets, ...)
        for key, value in params.items():
            if key != "subsidary_params" and hasattr(value, "shape") and len(value.shape) > 0:
                assert value.shape[0] == n_parameter_sets, (
                    f"Expected first dim {n_parameter_sets} for {key}, got {value.shape}"
                )

    def test_params_shape_with_ensemble(self, base_run_fingerprint, initial_values):
        """Test params shape when n_ensemble_members > 1."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 3
        n_parameter_sets = 2
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=n_parameter_sets,
        )

        # With n_ensemble_members > 1, most params have shape (n_parameter_sets, n_ensemble_members, ...)
        # Exception: initial_weights_logits is shared across members (n_parameter_sets, n_assets)
        for key, value in params.items():
            if key == "subsidary_params":
                continue
            if not hasattr(value, "shape") or len(value.shape) == 0:
                continue

            assert value.shape[0] == n_parameter_sets, (
                f"Expected first dim {n_parameter_sets} for {key}, got {value.shape}"
            )

            if key == "initial_weights_logits":
                # initial_weights_logits is shared - no ensemble dimension
                assert len(value.shape) == 2, (
                    f"initial_weights_logits should have 2 dims (n_param_sets, n_assets), got {value.shape}"
                )
                assert value.shape[1] == n_assets, (
                    f"Expected second dim {n_assets} for initial_weights_logits, got {value.shape}"
                )
            else:
                # Other params have ensemble dimension
                assert value.shape[1] == n_ensemble_members, (
                    f"Expected second dim {n_ensemble_members} for {key}, got {value.shape}"
                )

    def test_calculate_rule_outputs_no_ensemble(self, base_run_fingerprint, initial_values):
        """Test calculate_rule_outputs with n_ensemble_members=1."""
        pool = create_pool("ensemble__momentum")
        base_run_fingerprint["n_ensemble_members"] = 1
        n_assets = 3
        # 1000 timesteps with chunk_period=10 gives 100 chunks, 99 outputs after gradient
        time_steps = 1000
        n_chunks_output = 99  # 100 chunks - 1 for gradient calculation

        # Create single param set (as if after outer vmap slices)
        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )
        # Remove the n_parameter_sets dimension to simulate post-vmap
        params_single = {
            k: v[0] if hasattr(v, "shape") and len(v.shape) > 0 else v
            for k, v in params.items()
        }

        prices = jnp.ones((time_steps, n_assets))

        rule_outputs = pool.calculate_rule_outputs(
            params_single, base_run_fingerprint, prices
        )

        # Output shape is (n_chunks - 1, n_assets) due to gradient calculation
        assert rule_outputs.shape == (n_chunks_output, n_assets), (
            f"Expected shape {(n_chunks_output, n_assets)}, got {rule_outputs.shape}"
        )

    def test_calculate_rule_outputs_with_ensemble(self, base_run_fingerprint, initial_values):
        """Test calculate_rule_outputs with n_ensemble_members > 1."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 3
        n_assets = 3
        time_steps = 1000
        n_chunks_output = 99
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members

        # Create params for one parameter set with multiple ensemble members
        # (as if after outer vmap slices the n_parameter_sets dimension)
        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )
        # After init with n_parameter_sets=1 and n_ensemble_members=3,
        # params have shape (1, 3, ...). Slice to get (3, ...) for one param set.
        params_single = {
            k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 else v
            for k, v in params.items()
        }

        prices = jnp.ones((time_steps, n_assets))

        rule_outputs = pool.calculate_rule_outputs(
            params_single, base_run_fingerprint, prices
        )

        # Output should be (n_chunks - 1, n_assets) - averaged across ensemble members
        assert rule_outputs.shape == (n_chunks_output, n_assets), (
            f"Expected shape {(n_chunks_output, n_assets)}, got {rule_outputs.shape}"
        )

    def test_vmap_over_parameter_sets_with_ensemble(self, base_run_fingerprint, initial_values):
        """Test that outer vmap over n_parameter_sets works correctly."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 3
        n_parameter_sets = 2
        n_assets = 3
        time_steps = 1000
        n_chunks_output = 99
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=n_parameter_sets,
        )

        prices = jnp.ones((time_steps, n_assets))

        # Get vmap axes from pool
        params_in_axes = pool.make_vmap_in_axes(params)

        # vmap calculate_rule_outputs over parameter sets
        vmapped_outputs = vmap(
            lambda p: pool.calculate_rule_outputs(p, base_run_fingerprint, prices),
            in_axes=[params_in_axes],
        )(params)

        # Output should be (n_parameter_sets, n_chunks - 1, n_assets)
        assert vmapped_outputs.shape == (n_parameter_sets, n_chunks_output, n_assets), (
            f"Expected shape {(n_parameter_sets, n_chunks_output, n_assets)}, got {vmapped_outputs.shape}"
        )

    def test_gradient_flow_through_ensemble(self, base_run_fingerprint, initial_values):
        """Test that gradients flow back to all ensemble members.

        Note: Only params that affect rule_outputs will have non-zero gradients.
        For momentum pools, memory_length and k_per_day affect rule_outputs,
        but initial_weights_logits only affects fine_weights (not tested here).
        """
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 3
        n_assets = 3
        time_steps = 1000  # 100 chunks with chunk_period=10
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members

        # Create params for one parameter set with ensemble members
        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )
        # Slice to get single param set's ensemble members
        params_single = {
            k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 else v
            for k, v in params.items()
        }

        # Use varying prices to generate non-zero gradients
        prices = jnp.ones((time_steps, n_assets)) * jnp.arange(1, n_assets + 1)
        # Add some price variation over time
        prices = prices * (1 + 0.01 * jnp.arange(time_steps)[:, None])

        def loss_fn(p):
            rule_outputs = pool.calculate_rule_outputs(p, base_run_fingerprint, prices)
            return jnp.sum(rule_outputs ** 2)

        # Compute gradients
        grads = grad(loss_fn)(params_single)

        # Check params that actually affect rule_outputs (memory_length affects gradients)
        # initial_weights_logits doesn't affect rule_outputs, only fine_weights
        params_affecting_rule_outputs = ["memory_length", "k_per_day"]
        for key in params_affecting_rule_outputs:
            if key in grads:
                grad_value = grads[key]
                if hasattr(grad_value, "shape") and len(grad_value.shape) > 0:
                    # Gradients should have shape (n_ensemble_members, ...)
                    assert grad_value.shape[0] == n_ensemble_members, (
                        f"Expected gradient first dim {n_ensemble_members} for {key}, got {grad_value.shape}"
                    )
                    # At least some members should have non-zero gradients
                    has_nonzero = any(
                        not jnp.allclose(grad_value[i], 0)
                        for i in range(n_ensemble_members)
                    )
                    assert has_nonzero, f"All gradients for {key} are zero"

    def test_ensemble_members_get_different_init(self, base_run_fingerprint, initial_values):
        """Test that ensemble members get different random initializations.

        Note: add_noise() explicitly skips 'initial_weights_logits' and only adds
        noise to the first parameter set row when n_parameter_sets > 1. So we check
        'memory_length' which does receive noise.
        """
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
            noise="gaussian",
        )

        # Check that ensemble members have different values (due to noise)
        # Note: add_noise() skips 'initial_weights_logits', so check memory_length
        # Also, the first row (member 0) keeps the original value, noise is added to rows 1+
        params_to_check = ["memory_length"]  # These parameters receive noise
        for key in params_to_check:
            if key in params:
                value = params[key]
                if hasattr(value, "shape") and len(value.shape) > 1:
                    # value shape: (1, n_ensemble_members, ...)
                    members = value[0]  # shape: (n_ensemble_members, ...)
                    # Member 0 keeps original value, members 1+ have noise
                    # Check that at least some members are different from member 0
                    first_member = members[0]
                    has_difference = any(
                        not jnp.allclose(members[i], first_member)
                        for i in range(1, n_ensemble_members)
                    )
                    assert has_difference, (
                        f"No ensemble members have different values from member 0 for {key}"
                    )

    def test_backwards_compatibility_standard_pool(self, base_run_fingerprint, initial_values):
        """Test that non-ensemble pool still works normally."""
        pool = create_pool("momentum")  # No ensemble__ prefix
        n_parameter_sets = 4
        n_assets = 3
        time_steps = 1000
        n_chunks_output = 99

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=n_parameter_sets,
        )

        # Params should have shape (n_parameter_sets, ...) - no ensemble dimension
        for key, value in params.items():
            if key != "subsidary_params" and hasattr(value, "shape") and len(value.shape) > 0:
                assert value.shape[0] == n_parameter_sets

        prices = jnp.ones((time_steps, n_assets))

        # vmap over param sets
        params_in_axes = pool.make_vmap_in_axes(params)
        vmapped_outputs = vmap(
            lambda p: pool.calculate_rule_outputs(p, base_run_fingerprint, prices),
            in_axes=[params_in_axes],
        )(params)

        # Output should be (n_parameter_sets, n_chunks - 1, n_assets)
        assert vmapped_outputs.shape == (n_parameter_sets, n_chunks_output, n_assets)


class TestEnsembleWithBoundedWeights:
    """Test ensemble hook combined with bounded weights hook."""

    def test_ensemble_bounded_pool_creation(self):
        """Test that ensemble__bounded__pool can be created."""
        pool = create_pool("ensemble__bounded__momentum")
        assert pool is not None

    def test_ensemble_bounded_params_shape(self, base_run_fingerprint, initial_values):
        """Test params shape with both ensemble and bounded hooks."""
        pool = create_pool("ensemble__bounded__momentum")
        n_ensemble_members = 2
        n_parameter_sets = 2
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members

        # Bounded weights hook requires min/max bounds - add to initial_values
        initial_values["min_weights_per_asset"] = 0.1
        initial_values["max_weights_per_asset"] = 0.6

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=n_parameter_sets,
        )

        # Check that params have ensemble dimension
        for key, value in params.items():
            if key not in ["subsidary_params"] and hasattr(value, "shape") and len(value.shape) > 1:
                # Should have (n_parameter_sets, n_ensemble_members, ...) or similar
                assert value.shape[0] == n_parameter_sets, (
                    f"Expected first dim {n_parameter_sets} for {key}, got {value.shape}"
                )


class TestEnsembleStructuredInitialization:
    """Tests for structured ensemble initialization methods."""

    def test_lhs_initialization(self, base_run_fingerprint, initial_values):
        """Test Latin Hypercube Sampling initialization."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "lhs"
        base_run_fingerprint["ensemble_init_scale"] = 0.3

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        # Check shape
        assert params["log_k"].shape == (1, n_ensemble_members, n_assets)

        # With LHS, members should have different values
        log_k_members = params["log_k"][0]  # (n_ensemble_members, n_assets)
        for i in range(n_ensemble_members):
            for j in range(i + 1, n_ensemble_members):
                assert not jnp.allclose(log_k_members[i], log_k_members[j]), (
                    f"LHS members {i} and {j} should have different values"
                )

    def test_sobol_initialization(self, base_run_fingerprint, initial_values):
        """Test Sobol quasi-random initialization."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "sobol"
        base_run_fingerprint["ensemble_init_scale"] = 0.3

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        # Check shape
        assert params["log_k"].shape == (1, n_ensemble_members, n_assets)

        # With Sobol, members should have different values
        log_k_members = params["log_k"][0]
        for i in range(n_ensemble_members):
            for j in range(i + 1, n_ensemble_members):
                assert not jnp.allclose(log_k_members[i], log_k_members[j]), (
                    f"Sobol members {i} and {j} should have different values"
                )

    def test_grid_initialization(self, base_run_fingerprint, initial_values):
        """Test grid-based initialization."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "grid"
        base_run_fingerprint["ensemble_init_scale"] = 0.3

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        # Check shape
        assert params["log_k"].shape == (1, n_ensemble_members, n_assets)

    def test_centered_lhs_initialization(self, base_run_fingerprint, initial_values):
        """Test centered Latin Hypercube Sampling initialization."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "centered_lhs"
        base_run_fingerprint["ensemble_init_scale"] = 0.3

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        # Check shape
        assert params["log_k"].shape == (1, n_ensemble_members, n_assets)

    def test_schema_based_sampling_uses_schema_ranges(self, base_run_fingerprint, initial_values):
        """Test that schema-based sampling uses schema ranges, ignoring ensemble_init_scale.

        When a pool has PARAM_SCHEMA with optuna ranges, the LHS/Sobol sampling
        uses those ranges directly rather than scaling around base values.
        """
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "lhs"
        base_run_fingerprint["ensemble_init_seed"] = 42

        # Different scales should give same results when using schema ranges
        base_run_fingerprint["ensemble_init_scale"] = 0.1
        params_small_scale = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        base_run_fingerprint["ensemble_init_scale"] = 0.5
        params_large_scale = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        # With schema-based sampling, scale is ignored - results should be identical
        # (same seed, same schema ranges)
        assert jnp.allclose(params_small_scale["log_k"], params_large_scale["log_k"]), (
            "Schema-based sampling should produce identical results regardless of "
            "ensemble_init_scale (scale is only used for fallback without schema)"
        )

        # Verify params are within schema range
        schema = pool.get_param_schema()
        log_k_range = schema["log_k"].optuna
        assert params_small_scale["log_k"].min() >= log_k_range.low
        assert params_small_scale["log_k"].max() <= log_k_range.high

    def test_gaussian_initialization(self, base_run_fingerprint, initial_values):
        """Test Gaussian (default) initialization method."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "gaussian"

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=2,
        )

        # Check shape
        assert params["log_k"].shape == (2, n_ensemble_members, n_assets)

        # Members should have different values
        log_k_members = params["log_k"][0]
        has_difference = False
        for i in range(n_ensemble_members):
            for j in range(i + 1, n_ensemble_members):
                if not jnp.allclose(log_k_members[i], log_k_members[j]):
                    has_difference = True
                    break
        assert has_difference, "Gaussian init should produce different member values"

    def test_different_param_sets_get_different_ensembles(self, base_run_fingerprint, initial_values):
        """Test that each parameter set gets a unique ensemble configuration."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        n_parameter_sets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "lhs"
        base_run_fingerprint["ensemble_init_scale"] = 0.3

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=n_parameter_sets,
        )

        # Each parameter set should have different ensemble values
        for i in range(n_parameter_sets):
            for j in range(i + 1, n_parameter_sets):
                assert not jnp.allclose(params["log_k"][i], params["log_k"][j]), (
                    f"Parameter sets {i} and {j} should have different ensembles"
                )

    def test_seed_reproducibility(self, base_run_fingerprint, initial_values):
        """Test that the same seed produces identical results."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "lhs"
        base_run_fingerprint["ensemble_init_scale"] = 0.3
        base_run_fingerprint["ensemble_init_seed"] = 123

        params1 = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        params2 = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        assert jnp.allclose(params1["log_k"], params2["log_k"]), (
            "Same seed should produce identical results"
        )

    def test_different_seeds_produce_different_results(self, base_run_fingerprint, initial_values):
        """Test that different seeds produce different results."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = "lhs"
        base_run_fingerprint["ensemble_init_scale"] = 0.3

        base_run_fingerprint["ensemble_init_seed"] = 123
        params1 = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        base_run_fingerprint["ensemble_init_seed"] = 456
        params2 = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=1,
        )

        assert not jnp.allclose(params1["log_k"], params2["log_k"]), (
            "Different seeds should produce different results"
        )

    def test_invalid_init_method_raises_error(self, base_run_fingerprint, initial_values):
        """Test that invalid init method raises ValueError."""
        pool = create_pool("ensemble__momentum")
        base_run_fingerprint["n_ensemble_members"] = 4
        base_run_fingerprint["ensemble_init_method"] = "invalid_method"

        with pytest.raises(ValueError, match="Unknown ensemble init method"):
            pool.init_base_parameters(
                initial_values,
                base_run_fingerprint,
                n_assets=3,
                n_parameter_sets=1,
            )

    @pytest.mark.parametrize("method", ["gaussian", "lhs", "centered_lhs", "sobol", "grid"])
    def test_all_methods_work_with_multiple_param_sets(
        self, base_run_fingerprint, initial_values, method
    ):
        """Test all init methods work correctly with multiple parameter sets."""
        pool = create_pool("ensemble__momentum")
        n_ensemble_members = 4
        n_assets = 3
        n_parameter_sets = 3
        base_run_fingerprint["n_ensemble_members"] = n_ensemble_members
        base_run_fingerprint["ensemble_init_method"] = method
        base_run_fingerprint["ensemble_init_scale"] = 0.3

        params = pool.init_base_parameters(
            initial_values,
            base_run_fingerprint,
            n_assets,
            n_parameter_sets=n_parameter_sets,
        )

        # Check shapes
        assert params["log_k"].shape == (n_parameter_sets, n_ensemble_members, n_assets), (
            f"Method {method}: Expected shape {(n_parameter_sets, n_ensemble_members, n_assets)}, "
            f"got {params['log_k'].shape}"
        )
        # initial_weights_logits should be shared (no ensemble dim)
        assert params["initial_weights_logits"].shape == (n_parameter_sets, n_assets), (
            f"Method {method}: initial_weights_logits should have shape "
            f"{(n_parameter_sets, n_assets)}, got {params['initial_weights_logits'].shape}"
        )
