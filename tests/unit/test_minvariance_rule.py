"""
Tests for the minimum variance trading rule.

Min-variance is a risk-parity style strategy that allocates weights inversely
proportional to asset variances (inverse variance weighting). Assets with
lower variance get higher weights.

These tests correspond to tests in:
QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMMinVariance.t.sol
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit

from quantammsim.pools.G3M.quantamm.min_variance_pool import _jax_min_variance_weights
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_variance_scan_function,
)

# Protocol source file path
PROTOCOL_MINVARIANCE_TEST_PATH = (
    "QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMMinVariance.t.sol"
)


def compute_protocol_minvariance_single_step(
    prev_ewma: jnp.ndarray,
    new_ewma: jnp.ndarray,
    prev_intermediate_var: jnp.ndarray,
    price: jnp.ndarray,
    lamb: jnp.ndarray,
    prev_weights: jnp.ndarray,
    mixing_lambda: float,
) -> tuple:
    """
    Single-step min-variance update matching protocol behavior exactly.

    The protocol's variance calculation (from QuantammVarianceBasedRule.sol):
      A(t) = lambda * A(t-1) + (price - prev_ewma) * (price - new_ewma)
      variance(t) = (1 - lambda) * A(t)

    In protocol tests, prev_ewma == new_ewma, so this becomes (price - ewma)^2.

    Weight mixing (from MinimumVarianceUpdateRule.sol):
      new_weight = mixing_lambda * prev_weight + (1 - mixing_lambda) * raw_weight
      where raw_weight = (1/variance) / sum(1/variance)

    Args:
        prev_ewma: Previous EWMA (movingAverage[n:2n] in protocol)
        new_ewma: New EWMA (movingAverage[0:n] in protocol) - often equals prev_ewma in tests
        prev_intermediate_var: Previous A(t-1) intermediate variance state
        price: Current price data
        lamb: Lambda decay parameter for variance calculation
        prev_weights: Previous portfolio weights
        mixing_lambda: Mixing parameter (0 = all new weights, 1 = all old weights)

    Returns:
        (new_weights, new_intermediate_var, variance, raw_weights)
    """
    # Variance calculation matching protocol exactly
    diff_old = price - prev_ewma
    diff_new = price - new_ewma
    new_intermediate_var = lamb * prev_intermediate_var + diff_old * diff_new
    variance = (1 - lamb) * new_intermediate_var

    # Min-variance weights using simulator's function
    raw_weights = _jax_min_variance_weights(variance)

    # Weight mixing - protocol formula
    new_weights = mixing_lambda * prev_weights + (1 - mixing_lambda) * raw_weights

    return new_weights, new_intermediate_var, variance, raw_weights


def compute_minvariance_via_scan(
    prev_ewma: jnp.ndarray,
    prev_running_var: jnp.ndarray,
    price: jnp.ndarray,
    lamb: jnp.ndarray,
    prev_weights: jnp.ndarray,
    mixing_lambda: float = 0.0,
) -> tuple:
    """
    Compute min-variance weight update using simulator's scan function.

    This tests the simulator's actual implementation which updates EWMA
    during the variance calculation (differs from protocol's single-step tests).

    Args:
        prev_ewma: Previous EWMA of prices
        prev_running_var: Previous running variance state
        price: Current price data
        lamb: Lambda decay parameter
        prev_weights: Previous weights
        mixing_lambda: Weight mixing parameter (0 = all new, 1 = all old)

    Returns:
        (new_weights, new_ewma, new_running_var, variance, raw_weights)
    """
    G_inf = 1.0 / (1.0 - lamb)

    # Use simulator's variance scan function
    new_carry, variance = _jax_variance_scan_function(
        [prev_ewma, prev_running_var], price, G_inf, lamb
    )
    new_ewma, new_running_var = new_carry

    # Compute inverse variance weights
    raw_weights = _jax_min_variance_weights(variance)

    # Weight mixing - same formula as protocol
    new_weights = mixing_lambda * prev_weights + (1 - mixing_lambda) * raw_weights

    return new_weights, new_ewma, new_running_var, variance, raw_weights


class TestMinVarianceWeights:
    """Tests for the min-variance weight calculation mechanism."""

    def test_minvariance_weights_sum_to_one(self):
        """Min-variance weights should sum to 1."""
        variances = jnp.array([0.5, 1.0, 2.0])

        weights = _jax_min_variance_weights(variances)

        assert jnp.abs(jnp.sum(weights) - 1.0) < 1e-10, \
            "Min-variance weights should sum to 1"

    def test_minvariance_lower_variance_higher_weight(self):
        """Assets with lower variance should get higher weight."""
        variances = jnp.array([0.5, 2.0])

        weights = _jax_min_variance_weights(variances)

        assert weights[0] > weights[1], \
            "Asset with lower variance should have higher weight"

    def test_minvariance_equal_variance_equal_weight(self):
        """Equal variances should produce equal weights."""
        variances = jnp.array([1.0, 1.0, 1.0])

        weights = _jax_min_variance_weights(variances)

        expected = jnp.array([1/3, 1/3, 1/3])
        assert jnp.allclose(weights, expected, rtol=1e-10), \
            "Equal variances should produce equal weights"

    def test_minvariance_inverse_variance_weighting(self):
        """Weights should be proportional to 1/variance."""
        variances = jnp.array([1.0, 2.0, 4.0])

        weights = _jax_min_variance_weights(variances)

        # Expected: 1/1 : 1/2 : 1/4 = 4 : 2 : 1, normalized
        # Sum of precisions = 1 + 0.5 + 0.25 = 1.75
        expected = jnp.array([1/1.75, 0.5/1.75, 0.25/1.75])
        assert jnp.allclose(weights, expected, rtol=1e-10), \
            f"Expected {expected}, got {weights}"


class TestVarianceScanFunction:
    """Tests for the variance scan function primitive."""

    def test_variance_scan_updates_ewma(self):
        """Variance scan should update ewma correctly."""
        lamb = jnp.array([0.7, 0.7])
        G_inf = 1.0 / (1.0 - lamb)

        prev_ewma = jnp.array([1.0, 1.5])
        prev_running_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])

        new_carry, variance = _jax_variance_scan_function(
            [prev_ewma, prev_running_var], price, G_inf, lamb
        )
        new_ewma, new_running_var = new_carry

        # EWMA update: ewma = ewma + (price - ewma) / G_inf
        # G_inf = 1 / 0.3 = 3.333...
        # new_ewma[0] = 1.0 + (3.0 - 1.0) / 3.333 = 1.0 + 0.6 = 1.6
        # new_ewma[1] = 1.5 + (4.0 - 1.5) / 3.333 = 1.5 + 0.75 = 2.25
        expected_ewma = jnp.array([1.6, 2.25])
        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="Variance scan should update ewma correctly"
        )

    def test_variance_scan_computes_variance(self):
        """Variance scan should compute variance correctly."""
        lamb = jnp.array([0.7, 0.7])
        G_inf = 1.0 / (1.0 - lamb)

        prev_ewma = jnp.array([1.0, 1.5])
        prev_running_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])

        new_carry, variance = _jax_variance_scan_function(
            [prev_ewma, prev_running_var], price, G_inf, lamb
        )

        # diff_old = price - prev_ewma = [2.0, 2.5]
        # diff_new = diff_old * (1 - 1/G_inf) = diff_old * 0.7 = [1.4, 1.75]
        # new_running_var = lamb * prev_running_var + diff_old * diff_new
        #                 = 0.7 * 1.0 + 2.0 * 1.4 = 0.7 + 2.8 = 3.5 (asset 0)
        #                 = 0.7 * 1.0 + 2.5 * 1.75 = 0.7 + 4.375 = 5.075 (asset 1)
        # variance = new_running_var * (1 - lamb) = [3.5 * 0.3, 5.075 * 0.3]
        #          = [1.05, 1.5225]
        expected_variance = jnp.array([1.05, 1.5225])
        np.testing.assert_array_almost_equal(
            variance, expected_variance, decimal=10,
            err_msg="Variance scan should compute variance correctly"
        )

    def test_variance_positive(self):
        """Variance should always be positive."""
        lamb = jnp.array([0.7, 0.7])
        G_inf = 1.0 / (1.0 - lamb)

        prev_ewma = jnp.array([1.0, 1.5])
        prev_running_var = jnp.array([0.1, 0.1])
        price = jnp.array([1.1, 1.6])  # Small price change

        _, variance = _jax_variance_scan_function(
            [prev_ewma, prev_running_var], price, G_inf, lamb
        )

        assert jnp.all(variance > 0), "Variance should be positive"


class TestMinVarianceRuleProtocolCases:
    """
    Integration tests matching protocol's QuantAMMMinVariance.t.sol test cases.

    These tests use compute_protocol_minvariance_single_step which exactly
    matches the protocol's single-step behavior where prev_ewma == new_ewma.
    """

    def testCorrectUpdateWithLambdaPointFiveAndTwoWeights(self):
        """
        QuantAMMMinVariance.t.sol:129 - testCorrectUpdateWithLambdaPointFiveAndTwoWeights

        Protocol values:
        - parameters[0][0] = 0.5e18 -> mixing_lambda = 0.5
        - prevWeights: [0.5e18, 0.5e18] -> [0.5, 0.5]
        - data: [3e18, 4e18] -> [3.0, 4.0]
        - variances (initial intermediate state): [1e18, 1e18] -> [1.0, 1.0]
        - prevMovingAverages: [1, 1.5, 1, 1.5] -> new_ewma=[1,1.5], prev_ewma=[1,1.5]
        - lambda: [0.7e18] -> 0.7
        - expected: [0.548283261802575107e18, 0.451716738197424892e18]
        """
        # In protocol test, new_ewma == prev_ewma
        prev_ewma = jnp.array([1.0, 1.5])
        new_ewma = jnp.array([1.0, 1.5])
        prev_intermediate_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        prev_weights = jnp.array([0.5, 0.5])
        mixing_lambda = 0.5

        new_weights, _, _, _ = compute_protocol_minvariance_single_step(
            prev_ewma, new_ewma, prev_intermediate_var, price, lamb, prev_weights, mixing_lambda
        )

        expected_weights = jnp.array([0.548283261802575107, 0.451716738197424892])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MINVARIANCE_TEST_PATH}:129"
        )

    def testCorrectUpdateWithLambdaPointNineAndTwoWeights(self):
        """
        QuantAMMMinVariance.t.sol:185 - testCorrectUpdateWithLambdaPointNineAndTwoWeights

        Protocol values:
        - parameters[0][0] = 0.9e18 -> mixing_lambda = 0.9
        - Same other parameters as previous test
        - expected: [0.509656652360515021e18, 0.490343347639484978e18]
        """
        prev_ewma = jnp.array([1.0, 1.5])
        new_ewma = jnp.array([1.0, 1.5])
        prev_intermediate_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        prev_weights = jnp.array([0.5, 0.5])
        mixing_lambda = 0.9

        new_weights, _, _, _ = compute_protocol_minvariance_single_step(
            prev_ewma, new_ewma, prev_intermediate_var, price, lamb, prev_weights, mixing_lambda
        )

        expected_weights = jnp.array([0.509656652360515021, 0.490343347639484978])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MINVARIANCE_TEST_PATH}:185"
        )

    def testCorrectUpdateWithScalarParameterVectorLambdaTwoWeights(self):
        """
        QuantAMMMinVariance.t.sol:359 - testCorrectUpdateWithScalarParameterVectorLambdaTwoWeights

        Protocol values:
        - parameters[0][0] = 0.9e18 -> mixing_lambda = 0.9
        - lambda: [0.95e18, 0.5e18] -> [0.95, 0.5] (per-asset lambda)
        - expected: [0.543167701863354037e18, 0.456832298136645962e18]
        """
        prev_ewma = jnp.array([1.0, 1.5])
        new_ewma = jnp.array([1.0, 1.5])
        prev_intermediate_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.95, 0.5])  # Per-asset lambda
        prev_weights = jnp.array([0.5, 0.5])
        mixing_lambda = 0.9

        new_weights, _, _, _ = compute_protocol_minvariance_single_step(
            prev_ewma, new_ewma, prev_intermediate_var, price, lamb, prev_weights, mixing_lambda
        )

        expected_weights = jnp.array([0.543167701863354037, 0.456832298136645962])
        np.testing.assert_array_almost_equal(
            new_weights, expected_weights, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_MINVARIANCE_TEST_PATH}:359"
        )


class TestMinVarianceScanBehavior:
    """
    Tests for the simulator's scan-based variance calculation.

    These test the actual simulator functions which update EWMA during
    the scan (different from protocol's single-step tests where prev_ewma == new_ewma).
    """

    def test_scan_variance_differs_from_protocol_single_step(self):
        """
        The scan function updates EWMA internally, giving different results
        from protocol's single-step where prev_ewma == new_ewma.
        """
        lamb = jnp.array([0.7, 0.7])
        prev_ewma = jnp.array([1.0, 1.5])
        prev_running_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])

        # Scan-based calculation
        _, _, _, scan_variance, _ = compute_minvariance_via_scan(
            prev_ewma, prev_running_var, price, lamb, jnp.array([0.5, 0.5]), 0.0
        )

        # Protocol single-step calculation (prev_ewma == new_ewma)
        _, _, protocol_variance, _ = compute_protocol_minvariance_single_step(
            prev_ewma, prev_ewma, prev_running_var, price, lamb, jnp.array([0.5, 0.5]), 0.0
        )

        # They should be different because scan updates ewma internally
        assert not jnp.allclose(scan_variance, protocol_variance), \
            "Scan variance should differ from protocol single-step (ewma update difference)"

    def test_scan_mixing_formula_correct(self):
        """Test that the mixing formula is correctly applied."""
        lamb = jnp.array([0.7, 0.7])
        prev_ewma = jnp.array([1.0, 1.5])
        prev_running_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])
        prev_weights = jnp.array([0.3, 0.7])

        # mixing_lambda = 0 means all new weights
        weights_0, _, _, _, raw_weights = compute_minvariance_via_scan(
            prev_ewma, prev_running_var, price, lamb, prev_weights, mixing_lambda=0.0
        )
        assert jnp.allclose(weights_0, raw_weights), \
            "mixing_lambda=0 should give raw weights"

        # mixing_lambda = 1 means all old weights
        weights_1, _, _, _, _ = compute_minvariance_via_scan(
            prev_ewma, prev_running_var, price, lamb, prev_weights, mixing_lambda=1.0
        )
        assert jnp.allclose(weights_1, prev_weights), \
            "mixing_lambda=1 should give previous weights"

        # mixing_lambda = 0.5 means 50/50
        weights_half, _, _, _, _ = compute_minvariance_via_scan(
            prev_ewma, prev_running_var, price, lamb, prev_weights, mixing_lambda=0.5
        )
        expected_half = 0.5 * prev_weights + 0.5 * raw_weights
        assert jnp.allclose(weights_half, expected_half), \
            "mixing_lambda=0.5 should give 50/50 blend"


class TestMinVarianceEdgeCases:
    """Edge case tests for min-variance behavior."""

    def test_zero_mixing_returns_raw_weights(self):
        """Zero mixing (mixing_lambda=0) should return raw min-variance weights."""
        prev_ewma = jnp.array([1.0, 1.5])
        new_ewma = jnp.array([1.0, 1.5])
        prev_intermediate_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        prev_weights = jnp.array([0.5, 0.5])
        mixing_lambda = 0.0

        new_weights, _, _, raw_weights = compute_protocol_minvariance_single_step(
            prev_ewma, new_ewma, prev_intermediate_var, price, lamb, prev_weights, mixing_lambda
        )

        assert jnp.allclose(new_weights, raw_weights), \
            "Zero mixing should return raw min-variance weights"

    def test_full_mixing_returns_previous_weights(self):
        """Full mixing (mixing_lambda=1) should return previous weights."""
        prev_ewma = jnp.array([1.0, 1.5])
        new_ewma = jnp.array([1.0, 1.5])
        prev_intermediate_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        prev_weights = jnp.array([0.5, 0.5])
        mixing_lambda = 1.0

        new_weights, _, _, _ = compute_protocol_minvariance_single_step(
            prev_ewma, new_ewma, prev_intermediate_var, price, lamb, prev_weights, mixing_lambda
        )

        assert jnp.allclose(new_weights, prev_weights), \
            "Full mixing should return previous weights"

    def test_multiasset_minvariance(self):
        """Test min-variance with more than 2 assets."""
        variances = jnp.array([1.0, 2.0, 4.0, 8.0])

        weights = _jax_min_variance_weights(variances)

        assert jnp.abs(jnp.sum(weights) - 1.0) < 1e-10, \
            "Weights should sum to 1"

        # Weights should be in decreasing order (lower variance = higher weight)
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1], \
                "Lower variance assets should have higher weight"


class TestMinVarianceJIT:
    """Tests for JIT compilation of min-variance functions."""

    def test_minvariance_weights_jittable(self):
        """The min-variance weight function should be JIT-compilable."""
        @jit
        def jitted_minvariance(variances):
            return _jax_min_variance_weights(variances)

        variances = jnp.array([1.0, 2.0, 3.0])

        result = jitted_minvariance(variances)
        expected = _jax_min_variance_weights(variances)

        assert jnp.allclose(result, expected), \
            "JIT-compiled version should match non-JIT version"

    def test_protocol_pipeline_jittable(self):
        """The protocol-matching min-variance pipeline should be JIT-compilable."""
        @jit
        def jitted_protocol_pipeline(prev_ewma, new_ewma, prev_intermediate_var, price, lamb, prev_weights, mixing_lambda):
            diff_old = price - prev_ewma
            diff_new = price - new_ewma
            new_intermediate_var = lamb * prev_intermediate_var + diff_old * diff_new
            variance = (1 - lamb) * new_intermediate_var
            raw_weights = _jax_min_variance_weights(variance)
            new_weights = mixing_lambda * prev_weights + (1 - mixing_lambda) * raw_weights
            return new_weights

        prev_ewma = jnp.array([1.0, 1.5])
        new_ewma = jnp.array([1.0, 1.5])
        prev_intermediate_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        prev_weights = jnp.array([0.5, 0.5])
        mixing_lambda = 0.5

        result = jitted_protocol_pipeline(prev_ewma, new_ewma, prev_intermediate_var, price, lamb, prev_weights, mixing_lambda)

        # Should match protocol expected value
        expected = jnp.array([0.548283261802575107, 0.451716738197424892])
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6,
            err_msg="JIT-compiled protocol pipeline should produce correct result"
        )

    def test_scan_pipeline_jittable(self):
        """The scan-based pipeline should be JIT-compilable."""
        @jit
        def jitted_scan_pipeline(prev_ewma, prev_running_var, price, lamb, prev_weights, mixing_lambda):
            G_inf = 1.0 / (1.0 - lamb)
            new_carry, variance = _jax_variance_scan_function(
                [prev_ewma, prev_running_var], price, G_inf, lamb
            )
            raw_weights = _jax_min_variance_weights(variance)
            new_weights = mixing_lambda * prev_weights + (1 - mixing_lambda) * raw_weights
            return new_weights

        prev_ewma = jnp.array([1.0, 1.5])
        prev_running_var = jnp.array([1.0, 1.0])
        price = jnp.array([3.0, 4.0])
        lamb = jnp.array([0.7, 0.7])
        prev_weights = jnp.array([0.5, 0.5])
        mixing_lambda = 0.5

        # Just verify it compiles and runs without error
        result = jitted_scan_pipeline(prev_ewma, prev_running_var, price, lamb, prev_weights, mixing_lambda)

        assert jnp.abs(jnp.sum(result) - 1.0) < 1e-6, \
            "JIT-compiled scan pipeline should produce weights summing to 1"
