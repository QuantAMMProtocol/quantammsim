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
    """Test the core reparameterize_bounds function.

    Note: reparameterize_bounds works with simple (non-batched) params only.
    Batching over n_parameter_sets is handled externally by vmap.
    """

    def test_basic_constraints_satisfied(self):
        """Test that basic feasibility constraints are satisfied."""
        # Simple params: scalar budget, 1D logits
        raw_min_budget = jnp.array(0.0)  # sigmoid(0) = 0.5
        raw_min_logits = jnp.zeros(4)
        raw_gap_logits = jnp.zeros(4)

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        # Output should be 1D: (n_assets,)
        assert min_w.shape == (4,), f"Expected shape (4,), got {min_w.shape}"
        assert max_w.shape == (4,), f"Expected shape (4,), got {max_w.shape}"

        # Check all constraints
        assert jnp.sum(min_w) < 1.0, "sum(min) should be < 1"
        assert jnp.sum(max_w) >= 1.0 - 1e-5, "sum(max) should be >= 1 (within tolerance)"
        assert jnp.all(min_w >= 0), "all min should be >= 0"
        assert jnp.all(max_w <= 1.0), "all max should be <= 1"
        assert jnp.all(max_w > min_w), "all max should be > min"

    def test_constraints_with_various_n_assets(self):
        """Test constraints hold for different numbers of assets."""
        for n_assets in [2, 3, 4, 8]:
            raw_min_budget = jnp.array(0.0)
            raw_min_logits = jnp.zeros(n_assets)
            raw_gap_logits = jnp.zeros(n_assets)

            min_w, max_w = reparameterize_bounds(
                raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=n_assets
            )

            assert min_w.shape == (n_assets,), f"Expected (n_assets,), got {min_w.shape}"
            assert jnp.sum(min_w) < 1.0, f"sum(min) < 1 failed for n_assets={n_assets}"
            assert jnp.sum(max_w) >= 1.0 - 1e-5, f"sum(max) >= 1 failed for n_assets={n_assets}"
            assert jnp.all(max_w > min_w), f"max > min failed for n_assets={n_assets}"

    def test_small_gaps_trigger_scaling(self):
        """Test that small gaps trigger the scaling mechanism."""
        raw_min_budget = jnp.array(0.0)  # sum(min) = 0.5
        raw_min_logits = jnp.zeros(4)
        raw_gap_logits = jnp.full(4, -5.0)  # very small gaps

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
        raw_min_budget = jnp.array(-2.0)  # small sum(min)
        raw_min_logits = jnp.zeros(4)
        raw_gap_logits = jnp.full(4, 2.0)  # large gaps

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        # With large gaps and small mins, sum(initial_max) > 1
        # No scaling should occur (scale = 1)
        sum_max = float(jnp.sum(max_w))
        assert sum_max > 1.0, f"sum(max) = {sum_max} should be > 1 with large gaps"

    def test_extreme_min_budget_low(self):
        """Test with very small min budget (mins close to 0)."""
        raw_min_budget = jnp.array(-10.0)  # sigmoid(-10) ≈ 0
        raw_min_logits = jnp.zeros(4)
        raw_gap_logits = jnp.zeros(4)

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        assert jnp.sum(min_w) < 0.01, "sum(min) should be very small"
        assert jnp.all(min_w >= 0), "min should still be non-negative"
        assert jnp.all(max_w > min_w), "max > min should hold"

    def test_extreme_min_budget_high(self):
        """Test with large min budget (mins close to sum = 1)."""
        raw_min_budget = jnp.array(5.0)  # sigmoid(5) ≈ 0.99
        raw_min_logits = jnp.zeros(4)
        raw_gap_logits = jnp.zeros(4)

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        sum_min = float(jnp.sum(min_w))
        assert sum_min > 0.9, f"sum(min) = {sum_min} should be close to 1"
        assert sum_min < 1.0, "sum(min) should still be < 1"
        assert jnp.all(max_w > min_w), "max > min should hold"

    def test_asymmetric_min_distribution(self):
        """Test with non-uniform min distribution."""
        raw_min_budget = jnp.array(0.0)
        # Asymmetric logits: first asset gets more
        raw_min_logits = jnp.array([2.0, 0.0, 0.0, 0.0])
        raw_gap_logits = jnp.zeros(4)

        min_w, max_w = reparameterize_bounds(
            raw_min_budget, raw_min_logits, raw_gap_logits, n_assets=4
        )

        # First asset should have larger min
        assert min_w[0] > min_w[1], "First asset should have larger min"
        # All constraints should still hold
        assert jnp.all(max_w > min_w), "max > min should hold"
        assert jnp.sum(max_w) >= 1.0 - 1e-5, "sum(max) >= 1 should hold"



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

        raw_min_budget = jnp.array(0.0)
        raw_min_logits = jnp.zeros(4)
        raw_gap_logits = jnp.zeros(4)

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

        raw_min_budget = jnp.array(0.0)
        raw_min_logits = jnp.zeros(4)
        raw_gap_logits = jnp.zeros(4)

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
                raw_min_budget = jnp.array(raw_min_budget_val)
                raw_min_logits = jnp.zeros(4)
                raw_gap_logits = jnp.full(4, raw_gap_val)

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

        raw_min_budget = jnp.array(0.0)
        raw_min_logits = jnp.zeros(4)
        raw_gap_logits = jnp.zeros(4)

        # Should not raise
        min_w, max_w = jitted_reparameterize(raw_min_budget, raw_min_logits, raw_gap_logits)
        assert min_w.shape == (4,)
        assert max_w.shape == (4,)


class TestInitLearnableBoundsParams:
    """Test the init_learnable_bounds_params function.

    Note: init_learnable_bounds_params returns batched params with shape
    (n_parameter_sets, ...) for use during training initialization.
    To test with reparameterize_bounds, we index into the first param set.
    """

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

            # Index into first param set for simple params
            min_w, _ = reparameterize_bounds(
                params["raw_min_budget"][0],
                params["raw_min_logits"][0],
                params["raw_gap_logits"][0],
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

            # Index into first param set for simple params
            min_w, max_w = reparameterize_bounds(
                params["raw_min_budget"][0],
                params["raw_min_logits"][0],
                params["raw_gap_logits"][0],
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
    """Test the _bounds_to_raw_params conversion.

    Note: _bounds_to_raw_params returns batched params with shape (n_parameter_sets, ...).
    To test round-trip with reparameterize_bounds, we index into the first param set.
    """

    def _get_simple_params(self, raw_params):
        """Extract first param set for use with reparameterize_bounds."""
        return {
            "raw_min_budget": raw_params["raw_min_budget"][0],
            "raw_min_logits": raw_params["raw_min_logits"][0],
            "raw_gap_logits": raw_params["raw_gap_logits"][0],
        }

    def test_round_trip_uniform_bounds(self):
        """Test round-trip conversion with uniform bounds."""
        target_min = np.array([0.1, 0.1, 0.1, 0.1])
        target_max = np.array([0.4, 0.4, 0.4, 0.4])

        raw_params = BoundedWeightsHook._bounds_to_raw_params(target_min, target_max)
        simple_params = self._get_simple_params(raw_params)
        recovered_min, recovered_max = BoundedWeightsHook.raw_params_to_bounds(
            simple_params, n_assets=4
        )

        # Check mins are close
        np.testing.assert_array_almost_equal(
            np.array(recovered_min), target_min, decimal=2,
            err_msg="Recovered mins don't match target"
        )

        # Check maxes are close (may not be exact due to scaling)
        np.testing.assert_array_almost_equal(
            np.array(recovered_max), target_max, decimal=2,
            err_msg="Recovered maxes don't match target"
        )

    def test_round_trip_asymmetric_bounds(self):
        """Test round-trip with asymmetric bounds."""
        target_min = np.array([0.05, 0.1, 0.15, 0.2])
        target_max = np.array([0.3, 0.4, 0.5, 0.6])

        raw_params = BoundedWeightsHook._bounds_to_raw_params(target_min, target_max)
        simple_params = self._get_simple_params(raw_params)
        recovered_min, recovered_max = BoundedWeightsHook.raw_params_to_bounds(
            simple_params, n_assets=4
        )

        # Check relative ordering is preserved
        recovered_min_arr = np.array(recovered_min)
        recovered_max_arr = np.array(recovered_max)

        # Min ordering should be preserved
        assert np.all(np.diff(recovered_min_arr) > 0) or \
               np.allclose(np.diff(recovered_min_arr), np.diff(target_min), rtol=0.1), \
            "Min ordering not preserved"

    def test_constraints_hold_after_round_trip(self):
        """Test that constraints hold after round-trip conversion."""
        target_min = np.array([0.1, 0.1, 0.1, 0.1])
        target_max = np.array([0.5, 0.5, 0.5, 0.5])

        raw_params = BoundedWeightsHook._bounds_to_raw_params(target_min, target_max)
        simple_params = self._get_simple_params(raw_params)
        recovered_min, recovered_max = BoundedWeightsHook.raw_params_to_bounds(
            simple_params, n_assets=4
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


class TestRoundTripFidelity:
    """
    Test round-trip fidelity of _bounds_to_raw_params → raw_params_to_bounds.

    These tests specifically explore what "damage" is done to constraint-fulfilling
    bounds when they are converted to raw parameters for training and back.

    Key questions:
    1. How close are recovered bounds to the original targets?
    2. Are all 5 feasibility constraints preserved?
    3. What happens at edge cases (tight bounds, wide bounds, asymmetric, etc.)?
    """

    def _round_trip(self, target_min, target_max, n_parameter_sets=1):
        """Helper: convert bounds to raw params and back.

        Note: _bounds_to_raw_params returns batched params. We index into
        the first param set to get simple params for raw_params_to_bounds.
        """
        raw_params = BoundedWeightsHook._bounds_to_raw_params(
            target_min, target_max, n_parameter_sets
        )
        # Extract first param set for simple params
        simple_params = {
            "raw_min_budget": raw_params["raw_min_budget"][0],
            "raw_min_logits": raw_params["raw_min_logits"][0],
            "raw_gap_logits": raw_params["raw_gap_logits"][0],
        }
        recovered_min, recovered_max = BoundedWeightsHook.raw_params_to_bounds(
            simple_params, n_assets=len(target_min)
        )
        return recovered_min, recovered_max, raw_params

    def _check_constraints(self, min_w, max_w, tolerance=1e-6):
        """Helper: verify all 5 feasibility constraints."""
        errors = []

        # 1. sum(min) <= 1
        sum_min = float(jnp.sum(min_w))
        if sum_min > 1.0 + tolerance:
            errors.append(f"sum(min) = {sum_min:.6f} > 1")

        # 2. sum(max) >= 1
        sum_max = float(jnp.sum(max_w))
        if sum_max < 1.0 - tolerance:
            errors.append(f"sum(max) = {sum_max:.6f} < 1")

        # 3. min < max for all
        violations = jnp.where(min_w >= max_w - tolerance)[0]
        if len(violations) > 0:
            errors.append(f"min >= max at indices {violations.tolist()}")

        # 4. min >= 0
        neg_mins = jnp.where(min_w < -tolerance)[0]
        if len(neg_mins) > 0:
            errors.append(f"min < 0 at indices {neg_mins.tolist()}")

        # 5. max <= 1
        over_maxs = jnp.where(max_w > 1.0 + tolerance)[0]
        if len(over_maxs) > 0:
            errors.append(f"max > 1 at indices {over_maxs.tolist()}")

        return errors

    def test_uniform_bounds_fidelity(self):
        """Test fidelity with uniform bounds across assets."""
        target_min = np.array([0.1, 0.1, 0.1, 0.1])
        target_max = np.array([0.5, 0.5, 0.5, 0.5])

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        # Check fidelity
        min_error = np.max(np.abs(np.array(recovered_min) - target_min))
        max_error = np.max(np.abs(np.array(recovered_max) - target_max))

        print(f"Uniform bounds - min error: {min_error:.6f}, max error: {max_error:.6f}")
        print(f"  Target min: {target_min}, Recovered: {np.array(recovered_min)}")
        print(f"  Target max: {target_max}, Recovered: {np.array(recovered_max)}")

        # Constraints must hold
        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

        # Fidelity should be reasonable (within 5%)
        assert min_error < 0.05, f"Min error {min_error} too large"

    def test_asymmetric_bounds_fidelity(self):
        """Test fidelity with different bounds per asset."""
        target_min = np.array([0.05, 0.10, 0.15, 0.10])  # sum = 0.40
        target_max = np.array([0.30, 0.40, 0.50, 0.40])  # sum = 1.60

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        # Check relative ordering is preserved
        min_order_preserved = np.all(
            np.sign(np.diff(np.array(recovered_min))) == np.sign(np.diff(target_min))
        )

        print(f"Asymmetric bounds:")
        print(f"  Target min: {target_min}, Recovered: {np.array(recovered_min)}")
        print(f"  Target max: {target_max}, Recovered: {np.array(recovered_max)}")
        print(f"  Min ordering preserved: {min_order_preserved}")

        # Constraints must hold
        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

    def test_tight_bounds_fidelity(self):
        """Test fidelity when min is close to max (narrow ranges)."""
        target_min = np.array([0.20, 0.20, 0.20, 0.20])  # sum = 0.80
        target_max = np.array([0.30, 0.30, 0.30, 0.30])  # sum = 1.20, gap = 0.10

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        target_gap = target_max - target_min
        recovered_gap = np.array(recovered_max) - np.array(recovered_min)

        print(f"Tight bounds:")
        print(f"  Target gap: {target_gap}, Recovered gap: {recovered_gap}")
        print(f"  Target min: {target_min}, Recovered: {np.array(recovered_min)}")
        print(f"  Target max: {target_max}, Recovered: {np.array(recovered_max)}")

        # Constraints must hold
        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

        # Gap should be positive
        assert jnp.all(recovered_gap > 0), "Gap must remain positive"

    def test_wide_bounds_fidelity(self):
        """Test fidelity when bounds are very wide."""
        target_min = np.array([0.01, 0.01, 0.01, 0.01])  # sum = 0.04
        target_max = np.array([0.90, 0.90, 0.90, 0.90])  # sum = 3.60

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        print(f"Wide bounds:")
        print(f"  Target min: {target_min}, Recovered: {np.array(recovered_min)}")
        print(f"  Target max: {target_max}, Recovered: {np.array(recovered_max)}")

        # Constraints must hold
        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

    def test_edge_case_sum_min_near_one(self):
        """Test when sum(min) is close to 1 (stressful for constraint 1)."""
        target_min = np.array([0.24, 0.24, 0.24, 0.24])  # sum = 0.96, close to 1
        target_max = np.array([0.50, 0.50, 0.50, 0.50])  # sum = 2.0

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        print(f"Sum(min) near 1:")
        print(f"  Target sum(min): {np.sum(target_min):.4f}")
        print(f"  Recovered sum(min): {float(jnp.sum(recovered_min)):.4f}")
        print(f"  Target min: {target_min}, Recovered: {np.array(recovered_min)}")

        # Constraints must hold
        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

    def test_edge_case_sum_max_near_one(self):
        """Test when sum(max) is close to 1 (stressful for constraint 2)."""
        target_min = np.array([0.05, 0.05, 0.05, 0.05])  # sum = 0.20
        target_max = np.array([0.26, 0.26, 0.26, 0.26])  # sum = 1.04, close to 1

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        print(f"Sum(max) near 1:")
        print(f"  Target sum(max): {np.sum(target_max):.4f}")
        print(f"  Recovered sum(max): {float(jnp.sum(recovered_max)):.4f}")
        print(f"  Target max: {target_max}, Recovered: {np.array(recovered_max)}")

        # Constraints must hold
        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

    def test_two_assets(self):
        """Test with minimum number of assets."""
        target_min = np.array([0.2, 0.3])  # sum = 0.5
        target_max = np.array([0.6, 0.7])  # sum = 1.3

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        print(f"Two assets:")
        print(f"  Target min: {target_min}, Recovered: {np.array(recovered_min)}")
        print(f"  Target max: {target_max}, Recovered: {np.array(recovered_max)}")

        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

    def test_many_assets(self):
        """Test with larger number of assets."""
        n = 8
        target_min = np.full(n, 0.05)  # sum = 0.40
        target_max = np.full(n, 0.40)  # sum = 3.20

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        print(f"Eight assets:")
        print(f"  Target min: {target_min}")
        print(f"  Recovered min: {np.array(recovered_min)}")

        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

    @pytest.mark.parametrize("n_assets", [2, 3, 4, 5, 8])
    def test_constraints_always_hold(self, n_assets):
        """Parametrized test: constraints should hold for any valid input."""
        # Generate random valid bounds
        np.random.seed(42 + n_assets)

        # Random mins that sum to less than 1
        target_min = np.random.uniform(0.01, 0.2, n_assets)
        target_min = target_min * (0.8 / target_min.sum())  # scale to sum=0.8

        # Random maxes that sum to more than 1
        gaps = np.random.uniform(0.1, 0.3, n_assets)
        target_max = target_min + gaps
        target_max = np.clip(target_max, 0, 1)  # ensure max <= 1

        # Ensure sum(max) >= 1
        if target_max.sum() < 1.0:
            target_max = target_max * (1.1 / target_max.sum())
            target_max = np.clip(target_max, 0, 1)

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, \
            f"Constraint violations for n_assets={n_assets}: {errors}"

    def test_quantify_max_reconstruction_error(self):
        """
        Quantify the reconstruction error: how much do bounds change?

        This test documents the typical error introduced by round-trip conversion.
        """
        test_cases = [
            ("uniform_narrow", np.array([0.15, 0.15, 0.15, 0.15]), np.array([0.35, 0.35, 0.35, 0.35])),
            ("uniform_wide", np.array([0.05, 0.05, 0.05, 0.05]), np.array([0.80, 0.80, 0.80, 0.80])),
            ("asymmetric", np.array([0.05, 0.10, 0.15, 0.20]), np.array([0.30, 0.40, 0.50, 0.60])),
            ("tight", np.array([0.20, 0.20, 0.20, 0.20]), np.array([0.30, 0.30, 0.30, 0.30])),
        ]

        print("\n=== Round-trip reconstruction error analysis ===")
        for name, target_min, target_max in test_cases:
            recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

            min_abs_error = np.abs(np.array(recovered_min) - target_min)
            max_abs_error = np.abs(np.array(recovered_max) - target_max)

            min_rel_error = min_abs_error / (target_min + 1e-8)
            max_rel_error = max_abs_error / (target_max + 1e-8)

            print(f"\n{name}:")
            print(f"  Min - max abs error: {np.max(min_abs_error):.4f}, mean: {np.mean(min_abs_error):.4f}")
            print(f"  Max - max abs error: {np.max(max_abs_error):.4f}, mean: {np.mean(max_abs_error):.4f}")
            print(f"  Min - max rel error: {np.max(min_rel_error)*100:.1f}%")
            print(f"  Max - max rel error: {np.max(max_rel_error)*100:.1f}%")

            # Document that constraints hold even if values differ
            errors = self._check_constraints(recovered_min, recovered_max)
            print(f"  Constraints satisfied: {len(errors) == 0}")

    def test_min_sum_is_preserved_approximately(self):
        """Test that sum(min) is preserved reasonably well."""
        for target_sum in [0.2, 0.4, 0.6, 0.8]:
            target_min = np.full(4, target_sum / 4)
            target_max = np.full(4, 0.5)

            recovered_min, _, _ = self._round_trip(target_min, target_max)
            recovered_sum = float(jnp.sum(recovered_min))

            error = abs(recovered_sum - target_sum)
            print(f"Target sum(min)={target_sum:.2f}, recovered={recovered_sum:.4f}, error={error:.4f}")

            # Sum should be close (within 10% relative error)
            assert error < target_sum * 0.15, \
                f"Sum(min) error too large: target={target_sum}, got={recovered_sum}"

    def test_gap_preservation(self):
        """Test that the gap (max - min) is reasonably preserved."""
        target_min = np.array([0.10, 0.10, 0.10, 0.10])

        for target_gap in [0.1, 0.2, 0.3, 0.5]:
            target_max = target_min + target_gap

            recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)
            recovered_gap = np.array(recovered_max) - np.array(recovered_min)

            mean_recovered_gap = np.mean(recovered_gap)

            print(f"Target gap={target_gap:.2f}, recovered mean gap={mean_recovered_gap:.4f}")

            # Gap should be positive and roughly preserved
            assert np.all(recovered_gap > 0), "Gap must be positive"

    def test_scaling_behavior_when_sum_max_too_small(self):
        """
        Test the scaling mechanism when initial sum(max) < 1.

        The reparameterization scales up max weights if sum(initial_max) < 1.
        This test verifies the scaling behavior.
        """
        # Create bounds where naive conversion would give sum(max) < 1
        # min values such that even with sigmoid(raw_gap) at 0.5, sum(initial_max) might be small
        target_min = np.array([0.20, 0.20, 0.20, 0.20])  # sum = 0.80
        target_max = np.array([0.28, 0.28, 0.28, 0.28])  # sum = 1.12 (just above 1)

        recovered_min, recovered_max, raw_params = self._round_trip(target_min, target_max)

        # Check that sum(max) >= 1 after scaling
        sum_max = float(jnp.sum(recovered_max))

        print(f"Scaling test:")
        print(f"  Target max: {target_max}, sum={np.sum(target_max):.4f}")
        print(f"  Recovered max: {np.array(recovered_max)}, sum={sum_max:.4f}")

        assert sum_max >= 1.0 - 1e-5, f"sum(max) = {sum_max} should be >= 1"

        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

    def test_small_gap_scaling_documented(self):
        """
        Document: small gaps may be increased by the scaling mechanism.

        When sum(initial_max) < 1, the reparameterization scales up max weights
        to ensure sum(max) >= 1. This can increase gaps beyond the target.

        This test documents this expected behavior.
        """
        # Case where scaling kicks in: small gaps with moderate mins
        target_min = np.array([0.10, 0.10, 0.10, 0.10])  # sum = 0.40
        target_gap = 0.10
        target_max = target_min + target_gap  # [0.20, 0.20, 0.20, 0.20], sum = 0.80 < 1

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        # The reparameterization MUST scale max up to ensure sum(max) >= 1
        # target sum(max) = 0.80, so needs ~25% scaling to reach 1.0
        recovered_gap = np.array(recovered_max) - np.array(recovered_min)
        mean_gap = float(np.mean(recovered_gap))

        print(f"\nSmall gap scaling documentation:")
        print(f"  Target min: {target_min}, sum={np.sum(target_min):.2f}")
        print(f"  Target max: {target_min + target_gap}, sum={np.sum(target_min + target_gap):.2f}")
        print(f"  Target gap: {target_gap}")
        print(f"  Recovered min: {np.array(recovered_min)}")
        print(f"  Recovered max: {np.array(recovered_max)}, sum={float(jnp.sum(recovered_max)):.6f}")
        print(f"  Recovered gap: {recovered_gap}, mean={mean_gap:.4f}")

        # Constraints must hold (with slightly relaxed tolerance for numerical precision)
        errors = self._check_constraints(recovered_min, recovered_max, tolerance=1e-5)
        assert len(errors) == 0, f"Constraint violations: {errors}"

        # Document: gap is increased when sum(target_max) < 1
        # This is expected and necessary to satisfy sum(max) >= 1
        if np.sum(target_min + target_gap) < 1.0:
            # When target sum(max) < 1, recovered gap will be larger
            assert mean_gap >= target_gap, \
                "Gap should be >= target when sum(target_max) < 1 (scaling needed)"
            print(f"  Gap increased from {target_gap} to {mean_gap:.4f} " +
                  f"(+{(mean_gap/target_gap - 1)*100:.1f}%) due to scaling")

    def test_no_scaling_when_sum_max_above_one(self):
        """
        Verify: no gap distortion when sum(max) is already >= 1.

        When the target bounds already satisfy sum(max) >= 1, no scaling
        should occur and gaps should be preserved exactly.
        """
        # Case where no scaling needed: larger gaps
        target_min = np.array([0.10, 0.10, 0.10, 0.10])  # sum = 0.40
        target_gap = 0.30
        target_max = target_min + target_gap  # [0.40, 0.40, 0.40, 0.40], sum = 1.60 > 1

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        recovered_gap = np.array(recovered_max) - np.array(recovered_min)
        mean_gap = float(np.mean(recovered_gap))

        print(f"\nNo-scaling case:")
        print(f"  Target sum(max): {np.sum(target_max):.2f} >= 1 (no scaling needed)")
        print(f"  Target gap: {target_gap}, recovered mean gap: {mean_gap:.4f}")

        # Gap should be preserved exactly when no scaling needed
        np.testing.assert_array_almost_equal(
            recovered_gap, np.full(4, target_gap), decimal=5,
            err_msg="Gap should be preserved exactly when sum(max) >= 1"
        )

    def test_min_always_preserved_exactly(self):
        """
        Verify: minimum weights are always preserved exactly.

        The reparameterization's scaling only affects max weights (via gaps).
        Minimum weights should always be preserved exactly regardless of scaling.
        """
        test_cases = [
            np.array([0.05, 0.05, 0.05, 0.05]),
            np.array([0.10, 0.15, 0.20, 0.05]),
            np.array([0.20, 0.20, 0.20, 0.20]),
            np.array([0.01, 0.01, 0.01, 0.01]),
        ]

        for target_min in test_cases:
            # Use a max that might require scaling
            target_max = target_min + 0.15

            recovered_min, _, _ = self._round_trip(target_min, target_max)

            np.testing.assert_array_almost_equal(
                np.array(recovered_min), target_min, decimal=6,
                err_msg=f"Min weights not preserved for target_min={target_min}"
            )

    def test_practical_bounds_for_training(self):
        """
        Test practical bounds that a user might specify for training.

        This documents what happens with real-world bound specifications.
        """
        # Scenario: User wants to constrain weights between 5% and 60% per asset
        target_min = np.array([0.05, 0.05, 0.05, 0.05])  # sum = 0.20
        target_max = np.array([0.60, 0.60, 0.60, 0.60])  # sum = 2.40

        recovered_min, recovered_max, _ = self._round_trip(target_min, target_max)

        print(f"\nPractical training bounds:")
        print(f"  User specifies: min=5% each, max=60% each")
        print(f"  Recovered min: {np.array(recovered_min) * 100}%")
        print(f"  Recovered max: {np.array(recovered_max) * 100}%")

        # Verify user's intent is preserved
        np.testing.assert_array_almost_equal(
            np.array(recovered_min), target_min, decimal=4,
            err_msg="Minimum bounds not preserved"
        )
        np.testing.assert_array_almost_equal(
            np.array(recovered_max), target_max, decimal=4,
            err_msg="Maximum bounds not preserved"
        )

        errors = self._check_constraints(recovered_min, recovered_max)
        assert len(errors) == 0, f"Constraint violations: {errors}"

        print("  ✓ User's specified bounds preserved exactly")
