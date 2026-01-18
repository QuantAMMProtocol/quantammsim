"""
Tests for the anti-momentum trading rule.

Anti-momentum is a mean-reversion strategy that uses negative k values,
causing weight updates in the opposite direction to momentum.
When price is above the moving average, weights DECREASE (sell high).
When price is below the moving average, weights INCREASE (buy low).

These tests correspond to tests in:
QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMAntiMomentum.t.sol
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit
from typing import Union

from quantammsim.pools.G3M.quantamm.momentum_pool import _jax_momentum_weight_update
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_gradient_scan_function,
)

# Protocol source file path
PROTOCOL_ANTIMOMENTUM_TEST_PATH = (
    "QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMAntiMomentum.t.sol"
)


def compute_antimomentum_update(
    prev_ewma: jnp.ndarray,
    prev_alpha: jnp.ndarray,
    price: jnp.ndarray,
    lamb: jnp.ndarray,
    k: jnp.ndarray,
    prev_weights: jnp.ndarray,
    use_raw_price: bool = False,
) -> tuple:
    """
    Compute anti-momentum weight update using simulator functions.

    This is an integration test helper that uses the actual simulator code:
    - _jax_gradient_scan_function for ewma/alpha calculation
    - _jax_momentum_weight_update for weight updates (with negative k)

    Returns (new_weights, new_ewma, new_alpha, gradient) for verification.
    """
    G_inf = 1.0 / (1.0 - lamb)
    saturated_b = lamb / ((1 - lamb) ** 3)

    # Use simulator's scan function to compute new ewma and alpha
    new_carry, _ = _jax_gradient_scan_function(
        [prev_ewma, prev_alpha], price, G_inf, lamb, saturated_b
    )
    new_ewma, new_alpha = new_carry

    # Compute gradient with appropriate denominator
    denom = price if use_raw_price else new_ewma
    gradient = new_alpha / (saturated_b * denom)

    # Apply anti-momentum weight update (negative k)
    weight_updates = _jax_momentum_weight_update(gradient, -k)
    new_weights = prev_weights + weight_updates

    return new_weights, new_ewma, new_alpha, gradient


class TestAntiMomentumWeightUpdate:
    """Tests for the anti-momentum weight update mechanism."""

    def test_antimomentum_inverts_momentum_direction(self):
        """Anti-momentum should produce opposite weight changes to momentum."""
        gradient = jnp.array([0.1, -0.1])
        k = 1.0

        k_arr = jnp.broadcast_to(jnp.asarray(k), gradient.shape)
        momentum_update = _jax_momentum_weight_update(gradient, k_arr)
        antimomentum_update = _jax_momentum_weight_update(gradient, -k_arr)

        assert jnp.allclose(momentum_update, -antimomentum_update), \
            "Anti-momentum should be exact opposite of momentum"

    def test_antimomentum_sells_high(self):
        """When price > ewma (positive gradient), anti-momentum should reduce weight."""
        gradient = jnp.array([0.1, 0.0])
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        weight_updates = _jax_momentum_weight_update(gradient, -k)
        new_weights = prev_weights + weight_updates

        assert new_weights[0] < prev_weights[0], \
            "Anti-momentum should decrease weight when gradient is positive (sell high)"

    def test_antimomentum_buys_low(self):
        """When price < ewma (negative gradient), anti-momentum should increase weight."""
        gradient = jnp.array([-0.1, 0.0])
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        weight_updates = _jax_momentum_weight_update(gradient, -k)
        new_weights = prev_weights + weight_updates

        assert new_weights[0] > prev_weights[0], \
            "Anti-momentum should increase weight when gradient is negative (buy low)"

    def test_antimomentum_weight_sum_preserved(self):
        """Weight updates should sum to zero (weights remain normalized)."""
        gradient = jnp.array([0.1, -0.05, 0.02])
        k = jnp.array([1.0, 1.0, 1.0])

        weight_updates = _jax_momentum_weight_update(gradient, -k)

        assert jnp.abs(jnp.sum(weight_updates)) < 1e-10, \
            "Anti-momentum weight updates should sum to zero"


class TestAntiMomentumRuleProtocolCases:
    """
    Integration tests matching protocol's QuantAMMAntiMomentum.t.sol test cases.

    These tests use the actual simulator functions:
    - _jax_gradient_scan_function for gradient calculation
    - _jax_momentum_weight_update for weight updates
    """

    def testCorrectUpdateWithHigherPrices(self):
        """
        QuantAMMAntiMomentum.t.sol:241 - testCorrectUpdateWithHigherPrices

        Protocol values (useRawPrice=False - no parameters[1] in protocol):
        - parameters[0] (kappa): [1e18] -> 1.0
        - NO parameters[1] -> useRawPrice defaults to False
        - previousAlphas: [1e18, 2e18] -> [1.0, 2.0]
        - prevMovingAverages: [0, 0] -> [0.0, 0.0]
        - movingAverages: [0.9e18, 1.2e18] -> [0.9, 1.2] (computed)
        - lambdas: [0.7e18] -> 0.7
        - prevWeights: [0.5e18, 0.5e18] -> [0.5, 0.5]
        - data: [3e18, 4e18] -> [3.0, 4.0]
        - expected: [0.507499999999999999e18, 0.4925e18]
        """
        prev_ewma = jnp.array([0.0, 0.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        new_weights, new_ewma, _, _ = compute_antimomentum_update(
            prev_ewma, prev_alpha, price, lamb, k, prev_weights, use_raw_price=False
        )

        # Verify intermediate ewma matches protocol
        expected_ewma = jnp.array([0.9, 1.2])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Computed ewma should match protocol's movingAverages"
        )

        # Verify final weights
        expected_weights = jnp.array([0.507499999999999999, 0.4925])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_ANTIMOMENTUM_TEST_PATH}:241"
        )

    def testCorrectUpdateWithLowerPrices(self):
        """
        QuantAMMAntiMomentum.t.sol:296 - testCorrectUpdateWithLowerPrices

        Protocol values (useRawPrice=False):
        - previousAlphas: [1e18, 2e18] -> [1.0, 2.0]
        - prevMovingAverages: [3e18, 4e18] -> [3.0, 4.0]
        - movingAverages: [2.7e18, 4e18] -> [2.7, 4.0] (computed)
        - data: [2e18, 4e18] -> [2.0, 4.0]
        - expected: [0.518416666666666667e18, 0.481583333333333334e18]
        """
        prev_ewma = jnp.array([3.0, 4.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        new_weights, new_ewma, _, _ = compute_antimomentum_update(
            prev_ewma, prev_alpha, price, lamb, k, prev_weights, use_raw_price=False
        )

        # Verify intermediate ewma
        expected_ewma = jnp.array([2.7, 4.0])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Computed ewma should match protocol's movingAverages"
        )

        # Verify final weights
        expected_weights = jnp.array([0.518416666666666667, 0.481583333333333334])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_ANTIMOMENTUM_TEST_PATH}:296"
        )

    def testCorrectUpdateWithHigherPrices_VectorParams(self):
        """
        QuantAMMAntiMomentum.t.sol:350 - testCorrectUpdateWithHigherPrices_VectorParams

        Protocol values with per-asset k values and useRawPrice=True:
        - parameters[0] (kappa): [1e18, 1.5e18] -> [1.0, 1.5]
        - parameters[1][0] = 1e18 -> useRawPrice=True
        - previousAlphas: [1e18, 2e18] -> [1.0, 2.0]
        - prevMovingAverages: [3e18, 4e18] -> [3.0, 4.0]
        - movingAverages: [3e18, 4e18] -> [3.0, 4.0] (computed)
        - data: [3e18, 4e18] -> [3.0, 4.0]
        - expected: [0.5027e18, 0.4973e18]
        """
        prev_ewma = jnp.array([3.0, 4.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.5])
        prev_weights = jnp.array([0.5, 0.5])

        new_weights, new_ewma, _, _ = compute_antimomentum_update(
            prev_ewma, prev_alpha, price, lamb, k, prev_weights, use_raw_price=True
        )

        # Verify intermediate ewma
        expected_ewma = jnp.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Computed ewma should match protocol's movingAverages"
        )

        # Verify final weights
        expected_weights = jnp.array([0.5027, 0.4973])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=4,
            err_msg=f"Protocol test - {PROTOCOL_ANTIMOMENTUM_TEST_PATH}:350"
        )

    def testCorrectUpdateWithLowerPrices_VectorParams(self):
        """
        QuantAMMAntiMomentum.t.sol:401 - testCorrectUpdateWithLowerPrices_VectorParams

        Protocol values with useRawPrice=True:
        - parameters[0] (kappa): [1e18, 1.5e18] -> [1.0, 1.5]
        - useRawPrice=True
        - previousAlphas: [1e18, 2e18] -> [1.0, 2.0]
        - prevMovingAverages: [3e18, 4e18] -> [3.0, 4.0]
        - movingAverages: [2.7e18, 4e18] -> [2.7, 4.0] (computed)
        - data: [2e18, 4e18] -> [2.0, 4.0]
        - expected: [0.527e18, 0.473e18]
        """
        prev_ewma = jnp.array([3.0, 4.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.5])
        prev_weights = jnp.array([0.5, 0.5])

        new_weights, new_ewma, _, _ = compute_antimomentum_update(
            prev_ewma, prev_alpha, price, lamb, k, prev_weights, use_raw_price=True
        )

        # Verify intermediate ewma
        expected_ewma = jnp.array([2.7, 4.0])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Computed ewma should match protocol's movingAverages"
        )

        # Verify final weights
        expected_weights = jnp.array([0.527, 0.473])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=4,
            err_msg=f"Protocol test - {PROTOCOL_ANTIMOMENTUM_TEST_PATH}:401"
        )

    def testCorrectUpdateWithHigherPricesAverageDenominator(self):
        """
        QuantAMMAntiMomentum.t.sol:453 - testCorrectUpdateWithHigherPricesAverageDenominator

        Same setup as testCorrectUpdateWithHigherPrices (useRawPrice=False).
        Protocol has NO parameters[1], so useRawPrice defaults to False.
        """
        prev_ewma = jnp.array([0.0, 0.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        new_weights, new_ewma, _, _ = compute_antimomentum_update(
            prev_ewma, prev_alpha, price, lamb, k, prev_weights, use_raw_price=False
        )

        expected_ewma = jnp.array([0.9, 1.2])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Computed ewma should match protocol's movingAverages"
        )

        expected_weights = jnp.array([0.507499999999999999, 0.4925])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_ANTIMOMENTUM_TEST_PATH}:453"
        )

    def testCorrectUpdateWithLowerPricesAverageDenominator(self):
        """
        QuantAMMAntiMomentum.t.sol:508 - testCorrectUpdateWithLowerPricesAverageDenominator

        Same setup as testCorrectUpdateWithLowerPrices (useRawPrice=False).
        """
        prev_ewma = jnp.array([3.0, 4.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        new_weights, new_ewma, _, _ = compute_antimomentum_update(
            prev_ewma, prev_alpha, price, lamb, k, prev_weights, use_raw_price=False
        )

        expected_ewma = jnp.array([2.7, 4.0])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Computed ewma should match protocol's movingAverages"
        )

        expected_weights = jnp.array([0.518416666666666667, 0.481583333333333334])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_ANTIMOMENTUM_TEST_PATH}:508"
        )

    def testCorrectUpdateWithHigherPricesAverageDenominator_VectorParams(self):
        """
        QuantAMMAntiMomentum.t.sol:562 - testCorrectUpdateWithHigherPricesAverageDenominator_VectorParams

        Vector k with useRawPrice=False (no parameters[1] in protocol).
        """
        prev_ewma = jnp.array([3.0, 4.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.5])
        prev_weights = jnp.array([0.5, 0.5])

        new_weights, new_ewma, _, _ = compute_antimomentum_update(
            prev_ewma, prev_alpha, price, lamb, k, prev_weights, use_raw_price=False
        )

        expected_ewma = jnp.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Computed ewma should match protocol's movingAverages"
        )

        expected_weights = jnp.array([0.5027, 0.4973])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=4,
            err_msg=f"Protocol test - {PROTOCOL_ANTIMOMENTUM_TEST_PATH}:562"
        )

    def testCorrectUpdateWithLowerPricesAverageDenominator_VectorParams(self):
        """
        QuantAMMAntiMomentum.t.sol:611 - testCorrectUpdateWithLowerPricesAverageDenominator_VectorParams

        Vector k with useRawPrice=False.
        """
        prev_ewma = jnp.array([3.0, 4.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.5])
        prev_weights = jnp.array([0.5, 0.5])

        new_weights, new_ewma, _, _ = compute_antimomentum_update(
            prev_ewma, prev_alpha, price, lamb, k, prev_weights, use_raw_price=False
        )

        expected_ewma = jnp.array([2.7, 4.0])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Computed ewma should match protocol's movingAverages"
        )

        expected_weights = jnp.array([0.5221, 0.4779])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=4,
            err_msg=f"Protocol test - {PROTOCOL_ANTIMOMENTUM_TEST_PATH}:611"
        )


class TestAntiMomentumEdgeCases:
    """Edge case tests for anti-momentum behavior."""

    def test_zero_gradient_no_update(self):
        """Zero gradient should produce zero weight update."""
        gradient = jnp.array([0.0, 0.0])
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        weight_updates = _jax_momentum_weight_update(gradient, -k)
        new_weights = prev_weights + weight_updates

        assert jnp.allclose(new_weights, prev_weights), \
            "Zero gradient should not change weights"

    def test_uniform_gradient_no_update(self):
        """Uniform gradient across all assets should produce no relative weight change."""
        gradient = jnp.array([0.1, 0.1, 0.1])
        k = jnp.array([1.0, 1.0, 1.0])
        prev_weights = jnp.array([0.33, 0.33, 0.34])

        weight_updates = _jax_momentum_weight_update(gradient, -k)
        new_weights = prev_weights + weight_updates

        assert jnp.allclose(new_weights, prev_weights, atol=1e-10), \
            "Uniform gradient should not change relative weights"

    def test_large_k_amplifies_update(self):
        """Larger k should produce larger weight changes."""
        gradient = jnp.array([0.1, -0.1])
        prev_weights = jnp.array([0.5, 0.5])

        k1 = jnp.array([1.0, 1.0])
        k2 = jnp.array([2.0, 2.0])

        weights_k1 = prev_weights + _jax_momentum_weight_update(gradient, -k1)
        weights_k2 = prev_weights + _jax_momentum_weight_update(gradient, -k2)

        delta_k1 = jnp.abs(weights_k1 - prev_weights)
        delta_k2 = jnp.abs(weights_k2 - prev_weights)

        assert jnp.all(delta_k2 > delta_k1), \
            "Larger k should produce larger weight changes"

    def test_multiasset_antimomentum(self):
        """Test anti-momentum with more than 2 assets."""
        gradient = jnp.array([0.1, -0.05, 0.02, -0.03])
        k = jnp.array([1.0, 1.0, 1.0, 1.0])
        prev_weights = jnp.array([0.25, 0.25, 0.25, 0.25])

        weight_updates = _jax_momentum_weight_update(gradient, -k)
        new_weights = prev_weights + weight_updates

        assert jnp.abs(jnp.sum(new_weights) - 1.0) < 1e-10, \
            "Weights should still sum to 1"

        assert new_weights[0] < prev_weights[0], \
            "Asset with positive gradient should lose weight"
        assert new_weights[1] > prev_weights[1], \
            "Asset with negative gradient should gain weight"


class TestAntiMomentumJIT:
    """Tests for JIT compilation of anti-momentum functions."""

    def test_antimomentum_update_jittable(self):
        """The anti-momentum weight update should be JIT-compilable."""
        @jit
        def jitted_antimomentum_update(prev_weights, gradient, k):
            weight_updates = _jax_momentum_weight_update(gradient, -k)
            return prev_weights + weight_updates

        gradient = jnp.array([0.1, -0.1])
        prev_weights = jnp.array([0.5, 0.5])
        k = jnp.array([1.0, 1.0])

        result = jitted_antimomentum_update(prev_weights, gradient, k)
        expected = prev_weights + _jax_momentum_weight_update(gradient, -k)

        assert jnp.allclose(result, expected), \
            "JIT-compiled version should match non-JIT version"

    def test_full_pipeline_jittable(self):
        """The full anti-momentum pipeline should be JIT-compilable."""
        @jit
        def jitted_full_pipeline(prev_ewma, prev_alpha, price, lamb, k, prev_weights):
            G_inf = 1.0 / (1.0 - lamb)
            saturated_b = lamb / ((1 - lamb) ** 3)

            new_carry, _ = _jax_gradient_scan_function(
                [prev_ewma, prev_alpha], price, G_inf, lamb, saturated_b
            )
            new_ewma, new_alpha = new_carry

            gradient = new_alpha / (saturated_b * new_ewma)
            weight_updates = _jax_momentum_weight_update(gradient, -k)
            return prev_weights + weight_updates

        prev_ewma = jnp.array([0.0, 0.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        result = jitted_full_pipeline(prev_ewma, prev_alpha, price, lamb, k, prev_weights)

        # Should match protocol expected value
        expected = jnp.array([0.5075, 0.4925])
        assert jnp.allclose(result, expected, rtol=1e-4), \
            f"JIT-compiled full pipeline should produce correct result, got {result}"

    def test_antimomentum_vectorized_k(self):
        """Anti-momentum should work with per-asset k values."""
        gradient = jnp.array([0.1, -0.1, 0.05])
        k = jnp.array([1.0, 2.0, 0.5])
        prev_weights = jnp.array([0.33, 0.33, 0.34])

        weight_updates = _jax_momentum_weight_update(gradient, -k)
        new_weights = prev_weights + weight_updates

        assert jnp.abs(jnp.sum(new_weights) - 1.0) < 1e-10, \
            "Weights should sum to 1 with vectorized k"
