"""
Tests for weight guardrail functions.

These tests correspond to tests in:
QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/base/MathGuard.t.sol

The guardrails ensure weights:
1. Sum to 1
2. Are >= absolute_weight_guardrail (minimum weight per asset)
3. Are <= 1 - (n-1)*absolute_weight_guardrail (maximum weight per asset)
4. Change by at most epsilon_max per step (speed limit)

These tests use the actual simulator function `_jax_calc_coarse_weight_scan_function`
which implements the guardrail logic.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit

from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    _jax_calc_coarse_weight_scan_function,
    scale_diff,
)

# Protocol source file path
PROTOCOL_MATHGUARD_TEST_PATH = (
    "QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/base/MathGuard.t.sol"
)


def apply_simulator_guardrails(
    new_weights: jnp.ndarray,
    prev_weights: jnp.ndarray,
    minimum_weight: float,
    maximum_change: float = 1.0,
) -> tuple:
    """
    Apply the simulator's guardrail logic using the actual production function.

    This wraps _jax_calc_coarse_weight_scan_function for easier testing.

    Args:
        new_weights: Desired new weights (target)
        prev_weights: Previous weights
        minimum_weight: Minimum allowed weight per asset
        maximum_change: Maximum allowed change per step (default 1.0 = no limit)

    Returns:
        (target_weights, actual_weights): Target after clamping, actual after speed limit
    """
    n_assets = len(new_weights)

    # The function expects weight changes, not target weights
    # When rule_outputs_are_weights=False: raw_weights = prev + rule_outputs
    # So rule_outputs = new_weights - prev_weights
    weight_changes = new_weights - prev_weights

    result, (_, _, target_weights) = _jax_calc_coarse_weight_scan_function(
        carry_list=[prev_weights],
        rule_outputs=weight_changes,
        minimum_weight=minimum_weight,
        asset_arange=jnp.arange(n_assets),
        n_assets=n_assets,
        alt_lamb=None,
        interpol_num=2,  # interpol_num=2 means diff = target - prev (single step)
        maximum_change=maximum_change,
        rule_outputs_are_weights=False,
        ste_max_change=False,
        ste_min_max_weight=False,
        max_weights_per_asset=None,
        min_weights_per_asset=None,
        use_per_asset_bounds=False,
    )

    actual_weights = result[0]
    return target_weights, actual_weights


class TestClampWeightsSimulator:
    """
    Tests for weight clamping using the actual simulator function.

    These test the clamping behavior (lines 663-691 of fine_weights.py)
    by using a large maximum_change to disable speed limiting.
    """

    def test_clamp_weights_all_within_guardrail(self):
        """
        MathGuard.t.sol:28 - testClampWeights_AllWeightsWithinGuardRail

        Weights already within bounds should remain unchanged.
        """
        new_weights = jnp.array([0.3, 0.3, 0.4])
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        expected = jnp.array([0.3, 0.3, 0.4])
        np.testing.assert_array_almost_equal(
            target, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:28"
        )

    def test_clamp_weights_weight_below_guardrail(self):
        """
        MathGuard.t.sol:43 - testClampWeights_WeightBelowGuardRail

        Weight below minimum should be clamped up, others adjusted proportionally.
        Protocol expected: [0.1, 0.426315789473684210, 0.473684210526315789]
        """
        new_weights = jnp.array([0.05, 0.45, 0.5])
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        expected = jnp.array([0.1, 0.426315789473684210, 0.473684210526315789])
        np.testing.assert_array_almost_equal(
            target, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:43"
        )

    def test_clamp_weights_weight_above_guardrail(self):
        """
        MathGuard.t.sol:58 - testClampWeights_WeightAboveGuardRail

        Weights at valid maximum should remain unchanged.
        For 3 assets with guard=0.1, max = 1 - 2*0.1 = 0.8
        """
        new_weights = jnp.array([0.8, 0.1, 0.1])
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        expected = jnp.array([0.8, 0.1, 0.1])
        np.testing.assert_array_almost_equal(
            target, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:58"
        )

    def test_clamp_weights_weight_exceeds_max(self):
        """
        MathGuard.t.sol:73 - testClampWeights_WeightExceedsMax

        Weight above maximum should be clamped down, others adjusted up.
        """
        new_weights = jnp.array([0.95, 0.025, 0.025])
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        expected = jnp.array([0.8, 0.1, 0.1])
        np.testing.assert_array_almost_equal(
            target, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:73"
        )

    def test_clamp_weights_proportional_adjustment(self):
        """
        MathGuard.t.sol:88 - testClampWeights_ProportionalAdjustment

        Valid weights should remain unchanged.
        """
        new_weights = jnp.array([0.2, 0.2, 0.6])
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        expected = jnp.array([0.2, 0.2, 0.6])
        np.testing.assert_array_almost_equal(
            target, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:88"
        )

    def test_clamp_weights_negative_weights(self):
        """
        MathGuard.t.sol:118 - testClampWeights_NegativeWeights

        Negative weights should be clamped to minimum.
        Protocol expected: [0.1, 0.409090909090909091, 0.490909090909090909]
        """
        new_weights = jnp.array([-0.1, 0.5, 0.6])
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        expected = jnp.array([0.1, 0.409090909090909091, 0.490909090909090909])
        np.testing.assert_array_almost_equal(
            target, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:118"
        )

    def test_clamp_weights_sum_exceeds_one(self):
        """
        MathGuard.t.sol:134 - testClampWeights_SumExceedsOne

        Weights summing to more than 1 should be normalized first.
        """
        new_weights = jnp.array([0.5, 0.5, 0.5])
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        expected = jnp.array([1/3, 1/3, 1/3])
        np.testing.assert_array_almost_equal(
            target, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:134"
        )

    @pytest.mark.xfail(
        reason="Simulator produces NaN for all-zero weights due to division by zero. "
               "Protocol clamps to minimum first (MathGuard.t.sol:102).",
        strict=True  # Will fail if the test unexpectedly passes (i.e., bug is fixed)
    )
    def test_clamp_weights_all_zeros(self):
        """
        MathGuard.t.sol:102 - testClampWeights_ZeroWeights

        All zero weights should be clamped to minimum guardrail and normalized.
        Protocol expected: [1/3, 1/3, 1/3] after clamping to 0.1 each and normalizing.
        """
        new_weights = jnp.array([0.0, 0.0, 0.0])
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        # Protocol expects equal weights after clamping all to minimum
        expected = jnp.array([1/3, 1/3, 1/3])

        assert jnp.isclose(jnp.sum(target), 1.0, rtol=1e-6), \
            "Weights should sum to 1"
        assert jnp.all(target >= minimum_weight - 1e-10), \
            f"All weights should be >= minimum ({minimum_weight})"
        np.testing.assert_array_almost_equal(
            target, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:102"
        )


class TestNormalizeWeightUpdatesEdgeCases:
    """
    Tests for edge cases in weight update normalization.

    These correspond to testNormalizeWeightUpdates_* in MathGuard.t.sol
    """

    def test_normalize_updates_zero_epsilon_max(self):
        """
        MathGuard.t.sol:270 - testNormalizeWeightUpdates_ZeroEpsilonMax

        With epsilon_max = 0, no weight changes should be allowed.
        The weights should stay at previous values.
        """
        prev_weights = jnp.array([0.4, 0.3, 0.3])
        new_weights = jnp.array([0.45, 0.25, 0.3])
        minimum_weight = 0.1
        epsilon_max = 0.0  # No changes allowed

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        # With zero epsilon, actual weights should equal previous weights
        np.testing.assert_array_almost_equal(
            actual, prev_weights, decimal=6,
            err_msg="With epsilon_max=0, weights should not change"
        )
        # And they should still sum to 1
        assert jnp.isclose(jnp.sum(actual), 1.0, rtol=1e-6), \
            "Weights should sum to 1"

    def test_normalize_updates_max_epsilon_max(self):
        """
        MathGuard.t.sol:290 - testNormalizeWeightUpdates_MaxEpsilonMax

        With epsilon_max = 1.0, any valid change should be allowed.
        """
        prev_weights = jnp.array([0.4, 0.3, 0.3])
        new_weights = jnp.array([0.45, 0.25, 0.3])
        minimum_weight = 0.1
        epsilon_max = 1.0  # All changes allowed

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        # With max epsilon, target and actual should be the same
        np.testing.assert_array_almost_equal(
            actual, target, decimal=6,
            err_msg="With epsilon_max=1.0, actual should equal target"
        )
        # And they should match the requested new_weights
        np.testing.assert_array_almost_equal(
            actual, new_weights, decimal=6,
            err_msg="With epsilon_max=1.0, weights should reach target"
        )
        assert jnp.isclose(jnp.sum(actual), 1.0, rtol=1e-6), \
            "Weights should sum to 1"


class TestProtocolKnownEdgeCases:
    """
    Tests for specific edge cases identified in protocol testing.

    These are exact values from fuzz testing failures in the protocol.
    """

    @pytest.mark.xfail(
        reason="Simulator uses single-pass proportional clamping which can leave "
               "borderline weights below minimum. Protocol uses iterative clamping.",
        strict=True
    )
    def test_4_tokens_proportional_clamping_edge_case(self):
        """
        MathGuard.t.sol:764 - testWeightGuards4TokensKnownGuardrailEdgeCase

        Tests a 4-token scenario that exercises the interaction between
        clamping and proportional rescaling.

        When weight 3 (at 0.0) is clamped up to minimum 0.1, the other
        weights are scaled by 0.9. Weight 2 at 0.101 becomes 0.0909,
        which is below minimum.

        Protocol uses iterative clamping to prevent this; simulator does not.
        """
        # Use original edge case values
        new_weights = jnp.array([0.699, 0.2, 0.101, 0.0])
        prev_weights = jnp.array([0.7, 0.15, 0.1, 0.05])
        minimum_weight = 0.1
        epsilon_max = 0.5

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        # ALL weights should be >= minimum after proper iterative clamping
        assert jnp.isclose(jnp.sum(target), 1.0, rtol=1e-6), \
            "Target weights should sum to 1"
        assert jnp.all(target >= minimum_weight - 1e-10), \
            f"All target weights should be >= minimum ({minimum_weight})"

        assert jnp.isclose(jnp.sum(actual), 1.0, rtol=1e-6), \
            "Actual weights should sum to 1"
        assert jnp.all(actual >= minimum_weight - 1e-10), \
            f"All actual weights should be >= minimum ({minimum_weight})"

    def test_large_weight_reduction_with_guardrails(self):
        """
        Test attempting to reduce a weight from 0.7 to below minimum.

        This exercises the interaction between clamping and speed limiting.
        """
        prev_weights = jnp.array([0.7, 0.2, 0.1])
        new_weights = jnp.array([0.05, 0.475, 0.475])  # Try to make first weight tiny
        minimum_weight = 0.1
        epsilon_max = 0.1  # 10% max change

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        # Verify guardrails
        assert jnp.isclose(jnp.sum(actual), 1.0, rtol=1e-6), \
            "Weights should sum to 1"
        # Target should respect minimum weight
        assert jnp.all(target >= minimum_weight - 1e-10), \
            f"Target weights should be >= minimum ({minimum_weight})"
        # Actual should also respect minimum weight
        assert jnp.all(actual >= minimum_weight - 1e-10), \
            f"Actual weights should be >= minimum ({minimum_weight})"

        # Speed limit should prevent large change
        max_change = jnp.max(jnp.abs(actual - prev_weights))
        assert max_change <= epsilon_max + 1e-6, \
            f"Max change ({max_change}) should be <= epsilon_max ({epsilon_max})"


class TestGuardQuantAMMWeightsSimulator:
    """
    Tests for combined guardrails (speed limit + clamping) using simulator.

    These test the full guardrail behavior including speed limiting
    (lines 727-736 of fine_weights.py).
    """

    def test_weight_guards_2_tokens_below_epsilon_max(self):
        """
        MathGuard.t.sol:470 - testWeightGuards2TokensBelowEpsilonMax

        Changes within epsilon should pass through unchanged.
        """
        prev_weights = jnp.array([0.5, 0.5])
        new_weights = jnp.array([0.55, 0.45])
        minimum_weight = 0.1
        epsilon_max = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        expected = jnp.array([0.55, 0.45])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:470"
        )

    def test_weight_guards_2_tokens_above_epsilon_max(self):
        """
        MathGuard.t.sol:493 - testWeightGuards2TokensAboveEpsilonMax

        Changes exceeding epsilon should be scaled down.
        prev=[0.5, 0.5], new=[0.7, 0.3], eps=0.1
        Change is 0.2, exceeds 0.1, so scaled to [0.6, 0.4]
        """
        prev_weights = jnp.array([0.5, 0.5])
        new_weights = jnp.array([0.7, 0.3])
        minimum_weight = 0.1
        epsilon_max = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        expected = jnp.array([0.6, 0.4])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:493"
        )

    def test_weight_guards_2_tokens_clamped(self):
        """
        MathGuard.t.sol:516 - testWeightGuards2TokensClamped

        Both epsilon and clamping applied.
        prev=[0.5, 0.5], new=[0.95, 0.05], eps=0.1
        Speed limited first, then clamped.
        """
        prev_weights = jnp.array([0.5, 0.5])
        new_weights = jnp.array([0.95, 0.05])
        minimum_weight = 0.1
        epsilon_max = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        expected = jnp.array([0.6, 0.4])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:516"
        )

    def test_weight_guards_3_tokens_below_epsilon_max(self):
        """
        MathGuard.t.sol:539 - testWeightGuards3TokensBelowEpsilonMax

        3 tokens with changes within epsilon.
        """
        prev_weights = jnp.array([0.3, 0.3, 0.4])
        new_weights = jnp.array([0.35, 0.24, 0.41])
        minimum_weight = 0.1
        epsilon_max = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        expected = jnp.array([0.35, 0.24, 0.41])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:539"
        )

    def test_weight_guards_3_tokens_above_epsilon_max(self):
        """
        MathGuard.t.sol:565 - testWeightGuards3TokensAboveEpsilonMax

        3 tokens with changes exceeding epsilon.
        prev=[0.3, 0.3, 0.4], new=[0.5, 0.1, 0.4]
        Max change is 0.2 (on first two), scale by 0.1/0.2 = 0.5
        Scaled: [0.4, 0.2, 0.4]
        """
        prev_weights = jnp.array([0.3, 0.3, 0.4])
        new_weights = jnp.array([0.5, 0.1, 0.4])
        minimum_weight = 0.1
        epsilon_max = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        expected = jnp.array([0.4, 0.2, 0.4])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:565"
        )

    def test_weight_guards_3_tokens_clamped(self):
        """
        MathGuard.t.sol:591 - testWeightGuards3TokensClamped

        3 tokens with extreme changes requiring both speed limit and clamping.
        prev=[0.3, 0.3, 0.4], new=[0.9, 0.06, 0.04]

        Note: The simulator applies clamping BEFORE speed limiting,
        while the protocol applies speed limiting then clamping.
        This can lead to different results in edge cases.
        We verify the key invariants hold.
        """
        prev_weights = jnp.array([0.3, 0.3, 0.4])
        new_weights = jnp.array([0.9, 0.06, 0.04])
        minimum_weight = 0.1
        epsilon_max = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        # Verify key invariants
        assert jnp.isclose(jnp.sum(actual), 1.0, rtol=1e-6), \
            f"Weights should sum to 1, got {jnp.sum(actual)}"
        assert jnp.all(actual >= minimum_weight - 1e-10), \
            f"All weights should be >= {minimum_weight}, got min: {jnp.min(actual)}"
        changes = jnp.abs(actual - prev_weights)
        assert jnp.all(changes <= epsilon_max + 1e-10), \
            f"Changes should be <= {epsilon_max}, got max: {jnp.max(changes)}"

    def test_weight_guards_4_tokens_below_epsilon_max(self):
        """
        MathGuard.t.sol:617 - testWeightGuards4TokensBelowEpsilonMax
        """
        prev_weights = jnp.array([0.3, 0.3, 0.2, 0.2])
        new_weights = jnp.array([0.35, 0.25, 0.25, 0.15])
        minimum_weight = 0.1
        epsilon_max = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        expected = jnp.array([0.35, 0.25, 0.25, 0.15])
        np.testing.assert_array_almost_equal(
            actual, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MATHGUARD_TEST_PATH}:617"
        )

    def test_weight_guards_4_tokens_above_epsilon_max(self):
        """
        MathGuard.t.sol:646 - testWeightGuards4TokensAboveEpsilonMax

        4 tokens with changes exceeding epsilon.
        Verify key invariants hold.
        """
        prev_weights = jnp.array([0.3, 0.3, 0.2, 0.2])
        new_weights = jnp.array([0.15, 0.45, 0.05, 0.35])
        minimum_weight = 0.1
        epsilon_max = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=epsilon_max
        )

        # Verify key invariants
        assert jnp.isclose(jnp.sum(actual), 1.0, rtol=1e-6), \
            f"Weights should sum to 1, got {jnp.sum(actual)}"
        assert jnp.all(actual >= minimum_weight - 1e-10), \
            f"All weights should be >= {minimum_weight}, got min: {jnp.min(actual)}"
        changes = jnp.abs(actual - prev_weights)
        assert jnp.all(changes <= epsilon_max + 1e-10), \
            f"Changes should be <= {epsilon_max}, got max: {jnp.max(changes)}"


class TestScaleDiff:
    """Tests for the scale_diff helper function (actual simulator code)."""

    def test_scale_diff_no_scaling_needed(self):
        """Diff within limit should not be scaled."""
        diff = jnp.array([0.05, -0.05])
        maximum_change = 0.1

        result = scale_diff(diff, maximum_change)

        np.testing.assert_array_almost_equal(
            result, diff, decimal=10,
            err_msg="Diff within limit should not be scaled"
        )

    def test_scale_diff_scaling_required(self):
        """Diff exceeding limit should be scaled down."""
        diff = jnp.array([0.2, -0.2])
        maximum_change = 0.1

        result = scale_diff(diff, maximum_change)

        expected = jnp.array([0.1, -0.1])
        np.testing.assert_array_almost_equal(
            result, expected, decimal=10,
            err_msg="Diff should be scaled to maximum_change"
        )

    def test_scale_diff_preserves_proportions(self):
        """Scaling should preserve relative proportions."""
        diff = jnp.array([0.3, -0.1, 0.2])
        maximum_change = 0.15

        result = scale_diff(diff, maximum_change)

        # Max is 0.3, scale by 0.15/0.3 = 0.5
        expected = diff * 0.5
        np.testing.assert_array_almost_equal(
            result, expected, decimal=10,
            err_msg="Scaling should preserve proportions"
        )


class TestGuardrailInvariants:
    """
    Tests that verify key invariants of the guardrail system.

    These tests ensure the simulator's guardrails maintain the required
    properties regardless of edge cases.
    """

    @pytest.mark.parametrize("n_assets", [2, 3, 4, 5, 8])
    def test_output_sums_to_one(self, n_assets):
        """Output weights should always sum to 1."""
        # Random weights
        np.random.seed(42)
        new_weights = np.random.dirichlet(np.ones(n_assets))
        prev_weights = np.random.dirichlet(np.ones(n_assets))
        minimum_weight = 0.1 / n_assets

        target, actual = apply_simulator_guardrails(
            jnp.array(new_weights), jnp.array(prev_weights), minimum_weight
        )

        assert jnp.isclose(jnp.sum(target), 1.0, rtol=1e-6), \
            f"Target weights should sum to 1, got {jnp.sum(target)}"
        assert jnp.isclose(jnp.sum(actual), 1.0, rtol=1e-6), \
            f"Actual weights should sum to 1, got {jnp.sum(actual)}"

    @pytest.mark.parametrize("n_assets", [2, 3, 4, 5, 8])
    def test_output_respects_minimum(self, n_assets):
        """Output weights should all be >= minimum_weight."""
        np.random.seed(42)
        new_weights = np.random.dirichlet(np.ones(n_assets))
        prev_weights = np.random.dirichlet(np.ones(n_assets))
        minimum_weight = 0.1 / n_assets

        target, actual = apply_simulator_guardrails(
            jnp.array(new_weights), jnp.array(prev_weights), minimum_weight
        )

        assert jnp.all(target >= minimum_weight - 1e-10), \
            f"Target weights should be >= {minimum_weight}, got min: {jnp.min(target)}"
        assert jnp.all(actual >= minimum_weight - 1e-10), \
            f"Actual weights should be >= {minimum_weight}, got min: {jnp.min(actual)}"

    @pytest.mark.parametrize("n_assets", [2, 3, 4, 5, 8])
    def test_output_respects_maximum(self, n_assets):
        """Output weights should all be <= maximum_weight."""
        np.random.seed(42)
        new_weights = np.random.dirichlet(np.ones(n_assets))
        prev_weights = np.random.dirichlet(np.ones(n_assets))
        minimum_weight = 0.1 / n_assets
        maximum_weight = 1.0 - (n_assets - 1) * minimum_weight

        target, actual = apply_simulator_guardrails(
            jnp.array(new_weights), jnp.array(prev_weights), minimum_weight
        )

        assert jnp.all(target <= maximum_weight + 1e-10), \
            f"Target weights should be <= {maximum_weight}, got max: {jnp.max(target)}"
        assert jnp.all(actual <= maximum_weight + 1e-10), \
            f"Actual weights should be <= {maximum_weight}, got max: {jnp.max(actual)}"

    def test_speed_limit_respected(self):
        """Weight changes should be limited by maximum_change."""
        prev_weights = jnp.array([0.5, 0.5])
        new_weights = jnp.array([0.9, 0.1])  # Large change
        minimum_weight = 0.1
        maximum_change = 0.05

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=maximum_change
        )

        changes = jnp.abs(actual - prev_weights)
        assert jnp.all(changes <= maximum_change + 1e-10), \
            f"Changes should be <= {maximum_change}, got max: {jnp.max(changes)}"

    def test_extreme_weights_clamped(self):
        """Extreme weights should be clamped to valid range."""
        prev_weights = jnp.array([0.33, 0.33, 0.34])
        new_weights = jnp.array([0.99, 0.005, 0.005])  # Extreme
        minimum_weight = 0.1

        target, actual = apply_simulator_guardrails(
            new_weights, prev_weights, minimum_weight, maximum_change=1.0
        )

        # For 3 assets, max = 1 - 2*0.1 = 0.8
        maximum_weight = 1.0 - 2 * minimum_weight

        assert jnp.all(target >= minimum_weight - 1e-10), \
            f"Weights should be >= {minimum_weight}"
        assert jnp.all(target <= maximum_weight + 1e-10), \
            f"Weights should be <= {maximum_weight}"


class TestWeightGuardrailsJIT:
    """Tests for JIT compilation of guardrail functions."""

    def test_scale_diff_jittable(self):
        """scale_diff should be JIT-compilable."""
        @jit
        def jitted_scale_diff(diff, max_change):
            return scale_diff(diff, max_change)

        diff = jnp.array([0.2, -0.2])
        max_change = 0.1

        result = jitted_scale_diff(diff, max_change)
        expected = scale_diff(diff, max_change)

        np.testing.assert_array_almost_equal(
            result, expected, decimal=10,
            err_msg="JIT version should match non-JIT"
        )

    def test_coarse_weight_scan_jittable(self):
        """The coarse weight scan function should be JIT-compilable."""
        min_weight = 0.1
        n_assets = 3

        @jit
        def jitted_scan(prev_weights, weight_changes):
            return _jax_calc_coarse_weight_scan_function(
                carry_list=[prev_weights],
                rule_outputs=weight_changes,
                minimum_weight=min_weight,
                asset_arange=jnp.arange(n_assets),
                n_assets=n_assets,
                alt_lamb=None,
                interpol_num=2,
                maximum_change=0.1,
                rule_outputs_are_weights=False,
                ste_max_change=False,
                ste_min_max_weight=False,
                max_weights_per_asset=None,
                min_weights_per_asset=None,
                use_per_asset_bounds=False,
            )

        prev_weights = jnp.array([0.33, 0.33, 0.34])
        weight_changes = jnp.array([0.1, -0.05, -0.05])

        result, _ = jitted_scan(prev_weights, weight_changes)

        assert jnp.isclose(jnp.sum(result[0]), 1.0, rtol=1e-6), \
            "JIT-compiled function should produce valid weights"
