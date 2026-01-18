"""
Unit tests for learnable weight bounds reparameterization.

These tests verify:
1. All feasibility constraints are satisfied by construction
2. Gradient flow through the reparameterization
3. Round-trip conversion between explicit bounds and raw parameters
4. Edge cases (small gaps requiring scaling, extreme parameter values)

The constraints that must hold:
- sum(min_i) < 1  (can satisfy all minimums)
- sum(max_i) >= 1 (can reach total weight of 1)
- min_i < max_i for all i (valid range per asset)
- min_i >= 0 for all i (non-negative)
- max_i <= 1 for all i (bounded above)
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax

from quantammsim.hooks.bounded_weights_hook import (
    reparameterize_bounds,
    init_learnable_bounds_params,
    BoundedWeightsHook,
)


class TestReparameterizeBounds:
    """Test the core reparameterize_bounds function."""

    def test_basic_constraints_satisfied(self):
        """Test that basic feasibility constraints are satisfied."""
        raw_min_budget = jnp.array([0.0])  # sigmoid(0) = 0.5
        raw_min_logits = jnp.zeros((1, 4))
        raw_gap_logits = jnp.zeros((1, 4))

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        # Check all constraints
        assert jnp.sum(min_w) < 1.0, "sum(min) should be < 1"
        assert jnp.sum(max_w) >= 1.0 - 1e-5, "sum(max) should be >= 1 (within tolerance)"
        assert jnp.all(min_w >= 0), "all min should be >= 0"
        assert jnp.all(max_w <= 1.0), "all max should be <= 1"
        assert jnp.all(max_w > min_w), "all max should be > min"

    def test_constraints_with_various_n_assets(self):
        """Test constraints hold for different numbers of assets."""
        for n_assets in [2, 3, 4, 8]:
            raw_min_budget = jnp.array([0.0])
            raw_min_logits = jnp.zeros((1, n_assets))
            raw_gap_logits = jnp.zeros((1, n_assets))

            min_w, max_w = reparameterize_bounds(
                raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=n_assets
            )

            assert jnp.sum(min_w) < 1.0, f"sum(min) < 1 failed for n_assets={n_assets}"
            assert jnp.sum(max_w) >= 1.0 - 1e-5, f"sum(max) >= 1 failed for n_assets={n_assets}"
            assert jnp.all(max_w > min_w), f"max > min failed for n_assets={n_assets}"

    def test_small_gaps_trigger_scaling(self):
        """Test that small gaps trigger the scaling mechanism."""
        raw_min_budget = jnp.array([0.0])  # sum(min) = 0.5
        raw_min_logits = jnp.zeros((1, 4))
        raw_gap_logits = jnp.full((1, 4), -5.0)  # very small gaps

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        # With very small gaps, initial_max would be close to min
        # Scaling should ensure sum(max) >= 1
        sum_max = float(jnp.sum(max_w))
        assert sum_max >= 1.0 - 1e-5, f"sum(max) = {sum_max} should be >= 1 after scaling"
        assert jnp.all(max_w > min_w), "max > min must hold even with small gaps"
        assert jnp.all(max_w <= 1.0), "max <= 1 must hold after scaling"

    def test_large_gaps_no_scaling_needed(self):
        """Test that large gaps don't trigger unnecessary scaling."""
        raw_min_budget = jnp.array([-2.0])  # small sum(min)
        raw_min_logits = jnp.zeros((1, 4))
        raw_gap_logits = jnp.full((1, 4), 2.0)  # large gaps

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        # With large gaps and small mins, sum(initial_max) > 1
        # No scaling should occur (scale = 1)
        sum_max = float(jnp.sum(max_w))
        assert sum_max > 1.0, f"sum(max) = {sum_max} should be > 1 with large gaps"

    def test_extreme_min_budget_low(self):
        """Test with very small min budget (mins close to 0)."""
        raw_min_budget = jnp.array([-10.0])  # sigmoid(-10) ≈ 0
        raw_min_logits = jnp.zeros((1, 4))
        raw_gap_logits = jnp.zeros((1, 4))

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        assert jnp.sum(min_w) < 0.01, "sum(min) should be very small"
        assert jnp.all(min_w >= 0), "min should still be non-negative"
        assert jnp.all(max_w > min_w), "max > min should hold"

    def test_extreme_min_budget_high(self):
        """Test with large min budget (mins close to sum = 1)."""
        raw_min_budget = jnp.array([5.0])  # sigmoid(5) ≈ 0.99
        raw_min_logits = jnp.zeros((1, 4))
        raw_gap_logits = jnp.zeros((1, 4))

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        sum_min = float(jnp.sum(min_w))
        assert sum_min > 0.9, f"sum(min) = {sum_min} should be close to 1"
        assert sum_min < 1.0, "sum(min) should still be < 1"
        assert jnp.all(max_w > min_w), "max > min should hold"

    def test_asymmetric_min_distribution(self):
        """Test with non-uniform min distribution."""
        raw_min_budget = jnp.array([0.0])
        # Asymmetric logits: first asset gets more
        raw_min_logits = jnp.array([[2.0, 0.0, 0.0, 0.0]])
        raw_gap_logits = jnp.zeros((1, 4))

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        # First asset should have larger min
        assert min_w[0, 0] > min_w[0, 1], "First asset should have larger min"
        # All constraints should still hold
        assert jnp.all(max_w > min_w), "max > min should hold"
        assert jnp.sum(max_w) >= 1.0 - 1e-5, "sum(max) >= 1 should hold"

    def test_multiple_parameter_sets(self):
        """Test with multiple parallel parameter sets."""
        n_sets = 5
        n_assets = 4
        raw_min_budget = jnp.zeros((n_sets,))
        raw_min_logits = jnp.zeros((n_sets, n_assets))
        raw_gap_logits = jnp.zeros((n_sets, n_assets))

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=n_assets
        )

        assert min_w.shape == (n_sets, n_assets)
        assert max_w.shape == (n_sets, n_assets)

        for i in range(n_sets):
            assert jnp.sum(min_w[i]) < 1.0, f"sum(min) < 1 failed for set {i}"
            assert jnp.sum(max_w[i]) >= 1.0 - 1e-5, f"sum(max) >= 1 failed for set {i}"
            assert jnp.all(max_w[i] > min_w[i]), f"max > min failed for set {i}"


class TestGradientFlow:
    """Test gradient flow through the reparameterization."""

    def test_gradients_are_finite(self):
        """Test that gradients are finite for all parameters."""
        def loss_fn(raw_min_budget, raw_min_logits, raw_gap_logits):
            min_w, max_w = reparameterize_bounds(
                raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
            )
            # Dummy loss: want tight bounds (small gaps)
            return jnp.sum(max_w - min_w)

        raw_min_budget = jnp.array([0.0])
        raw_min_logits = jnp.zeros((1, 4))
        raw_gap_logits = jnp.zeros((1, 4))

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(
            raw_min_budget, raw_min_logits, raw_gap_logits
        )

        for i, g in enumerate(grads):
            assert jnp.all(jnp.isfinite(g)), f"Gradient {i} contains non-finite values"

    def test_gradients_are_nonzero(self):
        """Test that gradients are non-zero (information flows)."""
        def loss_fn(raw_min_budget, raw_min_logits, raw_gap_logits):
            min_w, max_w = reparameterize_bounds(
                raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
            )
            return jnp.sum(min_w) - jnp.sum(max_w)

        raw_min_budget = jnp.array([0.0])
        raw_min_logits = jnp.zeros((1, 4))
        raw_gap_logits = jnp.zeros((1, 4))

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(
            raw_min_budget, raw_min_logits, raw_gap_logits
        )

        # min_budget gradient should be non-zero (affects sum of mins)
        assert jnp.any(grads[0] != 0), "Gradient w.r.t. raw_min_budget should be non-zero"
        # gap_logits gradient should be non-zero (affects max)
        assert jnp.any(grads[2] != 0), "Gradient w.r.t. raw_gap_logits should be non-zero"

    def test_gradients_at_extremes(self):
        """Test gradient stability at extreme parameter values."""
        def loss_fn(raw_min_budget, raw_min_logits, raw_gap_logits):
            min_w, max_w = reparameterize_bounds(
                raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
            )
            return jnp.sum(max_w)

        # Test at extremes
        for raw_min_budget_val in [-5.0, 0.0, 5.0]:
            for raw_gap_val in [-5.0, 0.0, 5.0]:
                raw_min_budget = jnp.array([raw_min_budget_val])
                raw_min_logits = jnp.zeros((1, 4))
                raw_gap_logits = jnp.full((1, 4), raw_gap_val)

                grads = jax.grad(loss_fn, argnums=(0, 1, 2))(
                    raw_min_budget, raw_min_logits, raw_gap_logits
                )

                for g in grads:
                    assert jnp.all(jnp.isfinite(g)), \
                        f"Gradient not finite at min_budget={raw_min_budget_val}, gap={raw_gap_val}"

    def test_jit_compilation(self):
        """Test that the function can be JIT compiled."""
        @jax.jit
        def jitted_reparameterize(raw_min_budget, raw_min_logits, raw_gap_logits):
            return reparameterize_bounds(
                raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
            )

        raw_min_budget = jnp.array([0.0])
        raw_min_logits = jnp.zeros((1, 4))
        raw_gap_logits = jnp.zeros((1, 4))

        # Should not raise
        min_w, max_w = jitted_reparameterize(raw_min_budget, raw_min_logits, raw_gap_logits)
        assert min_w.shape == (1, 4)
        assert max_w.shape == (1, 4)


class TestInitLearnableBoundsParams:
    """Test the init_learnable_bounds_params function."""

    def test_returns_correct_keys(self):
        """Test that the function returns the expected parameter keys."""
        params = init_learnable_bounds_params(n_assets=4, n_parameter_sets=1)

        expected_keys = {"raw_min_budget", "raw_min_logits", "raw_gap_logits"}
        assert set(params.keys()) == expected_keys

    def test_correct_shapes(self):
        """Test that returned arrays have correct shapes."""
        n_assets = 4
        n_sets = 3
        params = init_learnable_bounds_params(n_assets=n_assets, n_parameter_sets=n_sets)

        assert params["raw_min_budget"].shape == (n_sets,)
        assert params["raw_min_logits"].shape == (n_sets, n_assets)
        assert params["raw_gap_logits"].shape == (n_sets, n_assets)

    def test_target_min_sum_respected(self):
        """Test that target_min_sum is approximately achieved."""
        for target in [0.1, 0.25, 0.5, 0.75]:
            params = init_learnable_bounds_params(
                n_assets=4, n_parameter_sets=1, target_min_sum=target
            )

            min_w, _ = reparameterize_bounds(
                params["raw_min_budget"],
                params["raw_min_logits"],
                params["raw_gap_logits"],
                n_assets=4,
            )

            actual_sum = float(jnp.sum(min_w))
            assert abs(actual_sum - target) < 0.01, \
                f"Expected sum(min) ≈ {target}, got {actual_sum}"

    def test_target_gap_fraction_respected(self):
        """Test that target_gap_fraction is approximately achieved."""
        for target_gap in [0.25, 0.5, 0.75]:
            params = init_learnable_bounds_params(
                n_assets=4, n_parameter_sets=1,
                target_min_sum=0.25, target_gap_fraction=target_gap
            )

            min_w, max_w = reparameterize_bounds(
                params["raw_min_budget"],
                params["raw_min_logits"],
                params["raw_gap_logits"],
                n_assets=4,
            )

            # Gap = max - min, available = 1 - min
            # gap_fraction = gap / available
            gaps = max_w - min_w
            available = 1.0 - min_w
            actual_fractions = gaps / available

            # Check average gap fraction (scaling may distort individual fractions)
            mean_fraction = float(jnp.mean(actual_fractions))
            # Allow larger tolerance due to potential scaling effects
            assert abs(mean_fraction - target_gap) < 0.2, \
                f"Expected gap_fraction ≈ {target_gap}, got {mean_fraction}"


class TestBoundsToRawParams:
    """Test the _bounds_to_raw_params conversion."""

    def test_round_trip_uniform_bounds(self):
        """Test round-trip conversion with uniform bounds."""
        target_min = np.array([0.1, 0.1, 0.1, 0.1])
        target_max = np.array([0.4, 0.4, 0.4, 0.4])

        raw_params = BoundedWeightsHook._bounds_to_raw_params(target_min, target_max)
        recovered_min, recovered_max = BoundedWeightsHook.raw_params_to_bounds(
            raw_params, n_assets=4
        )

        # Check mins are close
        np.testing.assert_array_almost_equal(
            np.array(recovered_min[0]), target_min, decimal=2,
            err_msg="Recovered mins don't match target"
        )

        # Check maxes are close (may not be exact due to scaling)
        np.testing.assert_array_almost_equal(
            np.array(recovered_max[0]), target_max, decimal=2,
            err_msg="Recovered maxes don't match target"
        )

    def test_round_trip_asymmetric_bounds(self):
        """Test round-trip with asymmetric bounds."""
        target_min = np.array([0.05, 0.1, 0.15, 0.2])
        target_max = np.array([0.3, 0.4, 0.5, 0.6])

        raw_params = BoundedWeightsHook._bounds_to_raw_params(target_min, target_max)
        recovered_min, recovered_max = BoundedWeightsHook.raw_params_to_bounds(
            raw_params, n_assets=4
        )

        # Check relative ordering is preserved
        recovered_min_arr = np.array(recovered_min[0])
        recovered_max_arr = np.array(recovered_max[0])

        # Min ordering should be preserved
        assert np.all(np.diff(recovered_min_arr) > 0) or \
               np.allclose(np.diff(recovered_min_arr), np.diff(target_min), rtol=0.1), \
            "Min ordering not preserved"

    def test_constraints_hold_after_round_trip(self):
        """Test that constraints hold after round-trip conversion."""
        target_min = np.array([0.1, 0.1, 0.1, 0.1])
        target_max = np.array([0.5, 0.5, 0.5, 0.5])

        raw_params = BoundedWeightsHook._bounds_to_raw_params(target_min, target_max)
        recovered_min, recovered_max = BoundedWeightsHook.raw_params_to_bounds(
            raw_params, n_assets=4
        )

        assert jnp.sum(recovered_min) < 1.0, "sum(min) < 1 should hold"
        assert jnp.sum(recovered_max) >= 1.0 - 1e-5, "sum(max) >= 1 should hold"
        assert jnp.all(recovered_max > recovered_min), "max > min should hold"
        assert jnp.all(recovered_min >= 0), "min >= 0 should hold"
        assert jnp.all(recovered_max <= 1.0), "max <= 1 should hold"


class TestValidateBounds:
    """Test the validate_bounds static method."""

    def test_valid_bounds_pass(self):
        """Test that valid bounds pass validation."""
        min_w = np.array([0.1, 0.1, 0.1, 0.1])
        max_w = np.array([0.4, 0.4, 0.4, 0.4])

        # Should not raise
        BoundedWeightsHook.validate_bounds(min_w, max_w, n_assets=4)

    def test_sum_min_exceeds_one_fails(self):
        """Test that sum(min) > 1 fails validation."""
        min_w = np.array([0.3, 0.3, 0.3, 0.3])  # sum = 1.2 > 1
        max_w = np.array([0.5, 0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match="Sum of min_weights.*exceeds 1.0"):
            BoundedWeightsHook.validate_bounds(min_w, max_w, n_assets=4)

    def test_sum_max_below_one_fails(self):
        """Test that sum(max) < 1 fails validation."""
        min_w = np.array([0.1, 0.1, 0.1, 0.1])
        max_w = np.array([0.2, 0.2, 0.2, 0.2])  # sum = 0.8 < 1

        with pytest.raises(ValueError, match="Sum of max_weights.*less than 1.0"):
            BoundedWeightsHook.validate_bounds(min_w, max_w, n_assets=4)

    def test_min_exceeds_max_fails(self):
        """Test that min >= max fails validation."""
        min_w = np.array([0.1, 0.5, 0.1, 0.1])  # second asset: min > max
        max_w = np.array([0.4, 0.4, 0.4, 0.4])

        with pytest.raises(ValueError, match="min_weights must be strictly less than max_weights"):
            BoundedWeightsHook.validate_bounds(min_w, max_w, n_assets=4)

    def test_negative_min_fails(self):
        """Test that negative min fails validation."""
        min_w = np.array([-0.1, 0.1, 0.1, 0.1])
        max_w = np.array([0.4, 0.4, 0.4, 0.4])

        with pytest.raises(ValueError, match="min_weights must be non-negative"):
            BoundedWeightsHook.validate_bounds(min_w, max_w, n_assets=4)

    def test_max_exceeds_one_fails(self):
        """Test that max > 1 fails validation."""
        min_w = np.array([0.1, 0.1, 0.1, 0.1])
        max_w = np.array([0.4, 0.4, 1.1, 0.4])

        with pytest.raises(ValueError, match="max_weights must not exceed 1.0"):
            BoundedWeightsHook.validate_bounds(min_w, max_w, n_assets=4)

    def test_shape_mismatch_fails(self):
        """Test that shape mismatch fails validation."""
        min_w = np.array([0.1, 0.1, 0.1])
        max_w = np.array([0.4, 0.4, 0.4, 0.4])

        with pytest.raises(ValueError, match=r"min_weights shape \(3,\) != max_weights shape \(4,\)"):
            BoundedWeightsHook.validate_bounds(min_w, max_w, n_assets=4)

    def test_wrong_n_assets_fails(self):
        """Test that wrong n_assets fails validation."""
        min_w = np.array([0.1, 0.1, 0.1, 0.1])
        max_w = np.array([0.4, 0.4, 0.4, 0.4])

        with pytest.raises(ValueError, match="length.*!= n_assets"):
            BoundedWeightsHook.validate_bounds(min_w, max_w, n_assets=3)
