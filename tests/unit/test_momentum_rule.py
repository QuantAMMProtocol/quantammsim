"""
Unit tests for the momentum strategy rule.

These tests verify that the simulator's momentum rule calculations match the protocol's
Solidity implementation.

Protocol source:
QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMMomentum.t.sol

The protocol tests use fixed-point math with 18 decimals (1e18 = 1.0).
The simulator uses standard floats, so we compare with appropriate tolerances.
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax
from typing import Union, Tuple

from quantammsim.pools.G3M.quantamm.momentum_pool import (
    MomentumPool,
    _jax_momentum_weight_update,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_gradient_scan_function,
)


# Protocol source file path
PROTOCOL_MOMENTUM_TEST_PATH = (
    "QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMMomentum.t.sol"
)


# =============================================================================
# Helper functions for momentum rule calculations
# =============================================================================

def single_step_gradient_update(
    previous_alpha: jnp.ndarray,
    moving_average: jnp.ndarray,
    lamb: Union[float, jnp.ndarray],
    data: jnp.ndarray,
    use_raw_price: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute a single step of the gradient estimator.

    Args:
        previous_alpha: Previous alpha (running_a) values
        moving_average: Current EWMA values
        lamb: Lambda decay parameter
        data: Current price data
        use_raw_price: If True, divide by price; if False, divide by EWMA

    Returns:
        Tuple of (new_alpha, gradient)
    """
    n_assets = len(data)
    if isinstance(lamb, (float, int)):
        lamb = jnp.array([lamb] * n_assets)
    else:
        lamb = jnp.array(lamb)

    saturated_b = lamb / ((1 - lamb) ** 3)
    G_inf = 1.0 / (1.0 - lamb)

    new_alpha = lamb * previous_alpha + G_inf * (data - moving_average)
    denom = data if use_raw_price else moving_average
    gradient = new_alpha / (saturated_b * denom)

    return new_alpha, gradient


def single_step_weight_update(
    prev_weights: jnp.ndarray,
    gradient: jnp.ndarray,
    k: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute weight update from gradient using momentum formula.

    Args:
        prev_weights: Previous weights
        gradient: Price gradient
        k: Kappa parameter

    Returns:
        New weights
    """
    k_arr = jnp.broadcast_to(jnp.asarray(k), gradient.shape)
    weight_updates = _jax_momentum_weight_update(gradient, k_arr)
    return prev_weights + weight_updates


def full_single_step_update(
    previous_alpha: jnp.ndarray,
    moving_average: jnp.ndarray,
    lamb: Union[float, jnp.ndarray],
    prev_weights: jnp.ndarray,
    data: jnp.ndarray,
    k: Union[float, jnp.ndarray],
    use_raw_price: bool = False,
) -> jnp.ndarray:
    """
    Complete single-step momentum update matching the protocol's runInitialUpdate.

    Args:
        previous_alpha: Previous alpha values
        moving_average: Current EWMA values
        lamb: Lambda decay parameter
        prev_weights: Previous weights
        data: Current price data
        k: Kappa parameter
        use_raw_price: Whether to use raw price as denominator

    Returns:
        New weights
    """
    _, gradient = single_step_gradient_update(
        previous_alpha=previous_alpha,
        moving_average=moving_average,
        lamb=lamb,
        data=data,
        use_raw_price=use_raw_price,
    )
    return single_step_weight_update(prev_weights, gradient, k)


# =============================================================================
# Test Classes - Momentum Weight Update Function
# =============================================================================

class TestMomentumWeightUpdate:
    """Test the _jax_momentum_weight_update function directly."""

    def test_weight_updates_sum_to_zero(self):
        """Weight updates should sum to zero (portfolio weight preserved)."""
        price_gradient = jnp.array([0.1, -0.05, 0.02])
        k = jnp.array([1.0, 1.0, 1.0])

        updates = _jax_momentum_weight_update(price_gradient, k)

        assert jnp.abs(jnp.sum(updates)) < 1e-10, \
            f"Weight updates should sum to zero, got {jnp.sum(updates)}"

    def test_weight_updates_with_scalar_k(self):
        """Test with scalar k (same for all assets)."""
        price_gradient = jnp.array([0.1, -0.1])
        k = jnp.array([2.0, 2.0])

        updates = _jax_momentum_weight_update(price_gradient, k)

        assert jnp.abs(updates[0] + updates[1]) < 1e-10
        assert jnp.abs(jnp.sum(updates)) < 1e-10

    def test_weight_updates_with_vector_k(self):
        """Test with different k per asset."""
        price_gradient = jnp.array([0.1, 0.1])
        k = jnp.array([1.0, 2.0])

        updates = _jax_momentum_weight_update(price_gradient, k)

        assert jnp.abs(jnp.sum(updates)) < 1e-10

    def test_zero_k_gives_zero_update(self):
        """Assets with k=0 should have zero weight update."""
        price_gradient = jnp.array([0.1, 0.2, 0.3])
        k = jnp.array([1.0, 0.0, 1.0])

        updates = _jax_momentum_weight_update(price_gradient, k)

        assert updates[1] == 0.0, "Asset with k=0 should have zero update"
        assert jnp.abs(jnp.sum(updates)) < 1e-10

    def test_all_same_gradient_gives_zero_updates(self):
        """If all gradients are the same, updates should be zero."""
        price_gradient = jnp.array([0.5, 0.5, 0.5])
        k = jnp.array([1.0, 1.0, 1.0])

        updates = _jax_momentum_weight_update(price_gradient, k)

        assert jnp.allclose(updates, 0.0, atol=1e-10), \
            "Equal gradients should give zero updates"


# =============================================================================
# Test Classes - Protocol Test Cases (Momentum Rule)
# Ported from: QuantAMMMomentum.t.sol
# =============================================================================

class TestMomentumRuleProtocolCases:
    """
    Tests ported from QuantAMMMomentum.t.sol.

    These test cases use exact values from the Solidity tests to verify
    the simulator matches the protocol implementation.
    """

    def testCorrectUpdateWithHigherPrices(self):
        """
        QuantAMMMomentum.t.sol:229 - testCorrectUpdateWithHigherPrices

        Protocol values (useRawPrice=True):
        - parameters[0] (kappa): [1e18] -> 1.0
        - previousAlphas: [1e18, 2e18] -> [1.0, 2.0]
        - movingAverages: [0.9e18, 1.2e18] -> [0.9, 1.2]
        - lambdas: [0.7e18] -> 0.7
        - prevWeights: [0.5e18, 0.5e18] -> [0.5, 0.5]
        - data: [3e18, 4e18] -> [3.0, 4.0]
        - expected: [0.49775e18, 0.50225e18]
        """
        new_weights = full_single_step_update(
            previous_alpha=jnp.array([1.0, 2.0]),
            moving_average=jnp.array([0.9, 1.2]),
            lamb=0.7,
            prev_weights=jnp.array([0.5, 0.5]),
            data=jnp.array([3.0, 4.0]),
            k=1.0,
            use_raw_price=True,
        )

        expected = jnp.array([0.49775, 0.50225])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=4,
            err_msg=f"Mismatch with protocol test at {PROTOCOL_MOMENTUM_TEST_PATH}:229"
        )

    def testCorrectUpdateWithHigherPricesAverageDenominator(self):
        """
        QuantAMMMomentum.t.sol:445 - testCorrectUpdateWithHigherPricesAverageDenominator

        Protocol values (useRawPrice=False):
        - Same inputs as testCorrectUpdateWithHigherPrices
        - expected: [0.4925e18, 0.5075e18]
        """
        new_weights = full_single_step_update(
            previous_alpha=jnp.array([1.0, 2.0]),
            moving_average=jnp.array([0.9, 1.2]),
            lamb=0.7,
            prev_weights=jnp.array([0.5, 0.5]),
            data=jnp.array([3.0, 4.0]),
            k=1.0,
            use_raw_price=False,
        )

        expected = jnp.array([0.4925, 0.5075])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=4,
            err_msg=f"Mismatch with protocol test at {PROTOCOL_MOMENTUM_TEST_PATH}:445"
        )

    def testCorrectUpdateWithLowerPrices(self):
        """
        QuantAMMMomentum.t.sol:286 - testCorrectUpdateWithLowerPrices

        Protocol values:
        - previousAlphas: [1e18, 2e18]
        - prevMovingAverages: [3e18, 4e18]
        - movingAverages: [2.7e18, 4e18]
        - lambdas: [0.7e18]
        - prevWeights: [0.5e18, 0.5e18]
        - data: [2e18, 4e18]
        - expected: [0.4775e18, 0.5225e18]
        """
        new_weights = full_single_step_update(
            previous_alpha=jnp.array([1.0, 2.0]),
            moving_average=jnp.array([2.7, 4.0]),
            lamb=0.7,
            prev_weights=jnp.array([0.5, 0.5]),
            data=jnp.array([2.0, 4.0]),
            k=1.0,
            use_raw_price=True,
        )

        expected = jnp.array([0.4775, 0.5225])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=4,
            err_msg=f"Mismatch with protocol test at {PROTOCOL_MOMENTUM_TEST_PATH}:286"
        )

    def testCorrectUpdateWithLowerPricesAverageDenominator(self):
        """
        QuantAMMMomentum.t.sol:501 - testCorrectUpdateWithLowerPricesAverageDenominator

        Protocol values (useRawPrice=False):
        - Same inputs as testCorrectUpdateWithLowerPrices
        - expected: [0.481583e18, 0.518417e18]
        """
        new_weights = full_single_step_update(
            previous_alpha=jnp.array([1.0, 2.0]),
            moving_average=jnp.array([2.7, 4.0]),
            lamb=0.7,
            prev_weights=jnp.array([0.5, 0.5]),
            data=jnp.array([2.0, 4.0]),
            k=1.0,
            use_raw_price=False,
        )

        expected = jnp.array([0.481583333333333333, 0.518416666666666666])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=4,
            err_msg=f"Mismatch with protocol test at {PROTOCOL_MOMENTUM_TEST_PATH}:501"
        )

    def testCorrectUpdateWithHigherPrices_VectorParams(self):
        """
        QuantAMMMomentum.t.sol:342 - testCorrectUpdateWithHigherPrices_VectorParams

        Protocol values:
        - parameters[0] (kappa): [1e18, 1.5e18] -> [1.0, 1.5]
        - previousAlphas: [1e18, 2e18]
        - movingAverages: [3e18, 4e18]
        - lambdas: [0.7e18, 0.7e18]
        - data: [3e18, 4e18]
        - expected: [0.4973e18, 0.5027e18]
        """
        new_weights = full_single_step_update(
            previous_alpha=jnp.array([1.0, 2.0]),
            moving_average=jnp.array([3.0, 4.0]),
            lamb=jnp.array([0.7, 0.7]),
            prev_weights=jnp.array([0.5, 0.5]),
            data=jnp.array([3.0, 4.0]),
            k=jnp.array([1.0, 1.5]),
            use_raw_price=True,
        )

        expected = jnp.array([0.4973, 0.5027])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=4,
            err_msg=f"Mismatch with protocol test at {PROTOCOL_MOMENTUM_TEST_PATH}:342"
        )

    def testCorrectUpdateWithHigherPricesAverageDenominator_VectorParams(self):
        """
        QuantAMMMomentum.t.sol:555 - testCorrectUpdateWithHigherPricesAverageDenominator_VectorParams

        Protocol values (useRawPrice=False, vector params):
        - Same inputs as testCorrectUpdateWithHigherPrices_VectorParams
        - expected: [0.4973e18, 0.5027e18]
        """
        new_weights = full_single_step_update(
            previous_alpha=jnp.array([1.0, 2.0]),
            moving_average=jnp.array([3.0, 4.0]),
            lamb=jnp.array([0.7, 0.7]),
            prev_weights=jnp.array([0.5, 0.5]),
            data=jnp.array([3.0, 4.0]),
            k=jnp.array([1.0, 1.5]),
            use_raw_price=False,
        )

        expected = jnp.array([0.4973, 0.5027])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=4,
            err_msg=f"Mismatch with protocol test at {PROTOCOL_MOMENTUM_TEST_PATH}:555"
        )

    def testCorrectUpdateWithLowerPrices_VectorParams(self):
        """
        QuantAMMMomentum.t.sol:393 - testCorrectUpdateWithLowerPrices_VectorParams

        Protocol values:
        - parameters[0] (kappa): [1e18, 1.5e18]
        - previousAlphas: [1e18, 2e18]
        - movingAverages: [2.7e18, 4e18]
        - lambdas: [0.7e18, 0.7e18]
        - data: [2e18, 4e18]
        - expected: [0.473e18, 0.527e18]
        """
        new_weights = full_single_step_update(
            previous_alpha=jnp.array([1.0, 2.0]),
            moving_average=jnp.array([2.7, 4.0]),
            lamb=jnp.array([0.7, 0.7]),
            prev_weights=jnp.array([0.5, 0.5]),
            data=jnp.array([2.0, 4.0]),
            k=jnp.array([1.0, 1.5]),
            use_raw_price=True,
        )

        expected = jnp.array([0.473, 0.527])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=3,
            err_msg=f"Mismatch with protocol test at {PROTOCOL_MOMENTUM_TEST_PATH}:393"
        )

    def testCorrectUpdateWithLowerPricesAverageDenominator_VectorParams(self):
        """
        QuantAMMMomentum.t.sol:604 - testCorrectUpdateWithLowerPricesAverageDenominator_VectorParams

        Protocol values (useRawPrice=False):
        - Same inputs as testCorrectUpdateWithLowerPrices_VectorParams
        - expected: [0.4779e18, 0.5221e18]
        """
        new_weights = full_single_step_update(
            previous_alpha=jnp.array([1.0, 2.0]),
            moving_average=jnp.array([2.7, 4.0]),
            lamb=jnp.array([0.7, 0.7]),
            prev_weights=jnp.array([0.5, 0.5]),
            data=jnp.array([2.0, 4.0]),
            k=jnp.array([1.0, 1.5]),
            use_raw_price=False,
        )

        expected = jnp.array([0.4779, 0.5221])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=3,
            err_msg=f"Mismatch with protocol test at {PROTOCOL_MOMENTUM_TEST_PATH}:604"
        )


# =============================================================================
# Test Classes - Protocol Cases via Estimator Primitives
# =============================================================================

class TestMomentumViaPrimitives:
    """
    Protocol test case verification using estimator primitives.

    These tests verify that the primitive scan functions produce correct
    intermediate values (ewma, alpha) before applying the momentum weight update.
    """

    def testCorrectUpdateWithHigherPrices_via_primitives(self):
        """
        QuantAMMMomentum.t.sol:229 - testCorrectUpdateWithHigherPrices
        Verified via _jax_gradient_scan_function.

        Protocol provides:
        - prevMovingAverages: [0, 0]
        - movingAverages: [0.9e18, 1.2e18] (computed)
        - previousAlphas: [1e18, 2e18]
        - data: [3e18, 4e18]
        """
        lamb = jnp.array([0.7, 0.7])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        prev_ewma = jnp.array([0.0, 0.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([3.0, 4.0])

        new_carry, _ = _jax_gradient_scan_function(
            [prev_ewma, prev_alpha], price, G_inf, lamb, saturated_b
        )
        new_ewma, new_alpha = new_carry

        # Verify computed ewma matches protocol's movingAverages
        expected_new_ewma = jnp.array([0.9, 1.2])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_new_ewma, decimal=10,
            err_msg="Scan function should compute correct new ewma"
        )

        # Apply momentum weight update (useRawPrice=True)
        gradient = new_alpha / (saturated_b * price)
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        weight_updates = _jax_momentum_weight_update(gradient, k)
        new_weights = prev_weights + weight_updates

        expected = jnp.array([0.49775, 0.50225])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=4,
            err_msg=f"Protocol test via primitives - {PROTOCOL_MOMENTUM_TEST_PATH}:229"
        )

    def testCorrectUpdateWithLowerPrices_via_primitives(self):
        """
        QuantAMMMomentum.t.sol:286 - testCorrectUpdateWithLowerPrices
        Verified via _jax_gradient_scan_function.

        Protocol provides:
        - prevMovingAverages: [3e18, 4e18]
        - movingAverages: [2.7e18, 4e18]
        - previousAlphas: [1e18, 2e18]
        - data: [2e18, 4e18]
        """
        lamb = jnp.array([0.7, 0.7])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        prev_ewma = jnp.array([3.0, 4.0])
        prev_alpha = jnp.array([1.0, 2.0])
        price = jnp.array([2.0, 4.0])

        new_carry, _ = _jax_gradient_scan_function(
            [prev_ewma, prev_alpha], price, G_inf, lamb, saturated_b
        )
        new_ewma, new_alpha = new_carry

        expected_new_ewma = jnp.array([2.7, 4.0])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_new_ewma, decimal=10,
            err_msg="Scan function should compute correct new ewma"
        )

        gradient = new_alpha / (saturated_b * price)
        k = jnp.array([1.0, 1.0])
        prev_weights = jnp.array([0.5, 0.5])

        weight_updates = _jax_momentum_weight_update(gradient, k)
        new_weights = prev_weights + weight_updates

        expected = jnp.array([0.4775, 0.5225])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=4,
            err_msg=f"Protocol test via primitives - {PROTOCOL_MOMENTUM_TEST_PATH}:286"
        )


# =============================================================================
# Test Classes - Weight Normalization
# =============================================================================

class TestWeightNormalization:
    """Test that weights are properly normalized."""

    def test_weights_sum_preserved(self):
        """New weights should sum to the same as previous weights."""
        prev_weights = jnp.array([0.5, 0.5])
        gradient = jnp.array([0.1, -0.05])
        k = jnp.array([1.0, 1.0])

        new_weights = single_step_weight_update(prev_weights, gradient, k)

        np.testing.assert_almost_equal(
            jnp.sum(new_weights), jnp.sum(prev_weights), decimal=10,
            err_msg="Weight sum should be preserved"
        )

    def test_weights_sum_preserved_unequal_initial(self):
        """Test with unequal initial weights."""
        prev_weights = jnp.array([0.7, 0.3])
        gradient = jnp.array([0.2, 0.1])
        k = jnp.array([2.0, 2.0])

        new_weights = single_step_weight_update(prev_weights, gradient, k)

        np.testing.assert_almost_equal(
            jnp.sum(new_weights), jnp.sum(prev_weights), decimal=10,
            err_msg="Weight sum should be preserved with unequal initial weights"
        )

    def test_weights_sum_preserved_three_assets(self):
        """Test with three assets."""
        prev_weights = jnp.array([0.4, 0.35, 0.25])
        gradient = jnp.array([0.1, -0.05, 0.02])
        k = jnp.array([1.0, 1.5, 2.0])

        new_weights = single_step_weight_update(prev_weights, gradient, k)

        np.testing.assert_almost_equal(
            jnp.sum(new_weights), jnp.sum(prev_weights), decimal=10,
            err_msg="Weight sum should be preserved with 3 assets"
        )


# =============================================================================
# Test Classes - Edge Cases
# =============================================================================

class TestMomentumEdgeCases:
    """Test edge cases and boundary conditions for momentum rule."""

    def test_very_small_gradient(self):
        """Very small gradients should give very small weight changes."""
        prev_weights = jnp.array([0.5, 0.5])
        gradient = jnp.array([1e-10, -1e-10])
        k = jnp.array([1.0, 1.0])

        new_weights = single_step_weight_update(prev_weights, gradient, k)

        max_change = jnp.max(jnp.abs(new_weights - prev_weights))
        assert max_change < 1e-9, f"Expected tiny change, got {max_change}"

    def test_large_k_scales_updates(self):
        """Larger k should produce larger weight changes."""
        prev_weights = jnp.array([0.5, 0.5])
        gradient = jnp.array([0.1, -0.1])

        k_small = jnp.array([1.0, 1.0])
        k_large = jnp.array([10.0, 10.0])

        weights_small_k = single_step_weight_update(prev_weights, gradient, k_small)
        weights_large_k = single_step_weight_update(prev_weights, gradient, k_large)

        change_small = jnp.max(jnp.abs(weights_small_k - prev_weights))
        change_large = jnp.max(jnp.abs(weights_large_k - prev_weights))

        np.testing.assert_almost_equal(
            change_large / change_small, 10.0, decimal=5,
            err_msg="k should scale weight changes linearly"
        )

    def test_opposite_gradients_symmetric_updates(self):
        """Opposite gradients with equal k should give symmetric updates."""
        prev_weights = jnp.array([0.5, 0.5])
        gradient = jnp.array([0.1, -0.1])
        k = jnp.array([1.0, 1.0])

        new_weights = single_step_weight_update(prev_weights, gradient, k)

        update_0 = new_weights[0] - prev_weights[0]
        update_1 = new_weights[1] - prev_weights[1]

        np.testing.assert_almost_equal(
            update_0, -update_1, decimal=10,
            err_msg="Opposite gradients should give opposite updates"
        )

    def test_many_assets(self):
        """Test with many assets (8, matching protocol max)."""
        n_assets = 8
        prev_weights = jnp.ones(n_assets) / n_assets
        gradient = jnp.array([0.1, -0.1, 0.05, -0.05, 0.02, -0.02, 0.01, -0.01])
        k = jnp.ones(n_assets)

        new_weights = single_step_weight_update(prev_weights, gradient, k)

        np.testing.assert_almost_equal(
            jnp.sum(new_weights), 1.0, decimal=10,
            err_msg="Weight sum should be 1.0 with 8 assets"
        )

        np.testing.assert_almost_equal(
            jnp.sum(new_weights - prev_weights), 0.0, decimal=10,
            err_msg="Weight updates should sum to zero with 8 assets"
        )


# =============================================================================
# Test Classes - JIT Compilation
# =============================================================================

class TestMomentumJIT:
    """Test that momentum functions work correctly when JIT compiled."""

    def test_weight_update_jit(self):
        """Test _jax_momentum_weight_update works with JIT."""
        @jax.jit
        def jitted_update(gradient, k):
            return _jax_momentum_weight_update(gradient, k)

        gradient = jnp.array([0.1, -0.05])
        k = jnp.array([1.0, 1.0])

        result = jitted_update(gradient, k)

        assert jnp.all(jnp.isfinite(result))
        assert jnp.abs(jnp.sum(result)) < 1e-10

    def test_full_update_jit(self):
        """Test full_single_step_update works with JIT."""
        @jax.jit
        def jitted_full_update(alpha, ewma, lamb, weights, data, k):
            return full_single_step_update(
                previous_alpha=alpha,
                moving_average=ewma,
                lamb=lamb,
                prev_weights=weights,
                data=data,
                k=k,
                use_raw_price=True,
            )

        result = jitted_full_update(
            jnp.array([1.0, 2.0]),
            jnp.array([0.9, 1.2]),
            0.7,
            jnp.array([0.5, 0.5]),
            jnp.array([3.0, 4.0]),
            1.0,
        )

        assert jnp.all(jnp.isfinite(result))
        np.testing.assert_almost_equal(jnp.sum(result), 1.0, decimal=10)

    def test_gradient_flow(self):
        """Test that gradients can be computed through the functions."""
        def loss_fn(k):
            weights = full_single_step_update(
                previous_alpha=jnp.array([1.0, 2.0]),
                moving_average=jnp.array([0.9, 1.2]),
                lamb=0.7,
                prev_weights=jnp.array([0.5, 0.5]),
                data=jnp.array([3.0, 4.0]),
                k=k,
                use_raw_price=True,
            )
            return jnp.sum(weights ** 2)

        grad = jax.grad(loss_fn)(jnp.array([1.0, 1.0]))

        assert jnp.all(jnp.isfinite(grad))
