"""
Tests for the power-channel trading rule.

The power-channel rule applies a power law transformation to price gradients:
  signal = sign(g) * |g / (2*pre_exp_scaling)|^exponents

This creates a trend-following strategy with nonlinear (power law) scaling.

These tests correspond to tests in:
QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMPowerChannel.t.sol
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit

from quantammsim.pools.G3M.quantamm.power_channel_pool import (
    _jax_power_channel_weight_update,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_gradient_scan_function,
)

# Protocol source file path
PROTOCOL_POWER_CHANNEL_TEST_PATH = (
    "QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMPowerChannel.t.sol"
)


def compute_protocol_power_channel_single_step(
    prev_alpha: jnp.ndarray,
    prev_ewma: jnp.ndarray,
    new_ewma: jnp.ndarray,
    price: jnp.ndarray,
    lamb: jnp.ndarray,
    prev_weights: jnp.ndarray,
    kappa: jnp.ndarray,
    exponents: jnp.ndarray,
    use_raw_price: bool = True,
    pre_exp_scaling: float = 0.5,
) -> tuple:
    """
    Single-step power-channel update matching protocol behavior exactly.

    The protocol's power channel calculation (from PowerChannelUpdateRule.sol):
    1. From _calculateQuantAMMGradient in QuantammGradientBasedRule.sol:
       - alpha_new = λ * alpha_old + (price - ewma) / (1 - λ)
       - gradient_output = mulFactor * alpha_new, where mulFactor = (1-λ)^3 / λ
    2. From PowerChannelUpdateRule.sol:
       - price_gradient = gradient_output / denominator
       - where denominator = price (if use_raw_price) or ewma (otherwise)
    3. signal = sign(g) * |g / (2*pre_exp_scaling)|^exponents
    4. normalization = sum(kappa * signal) / sum(kappa) (for vector kappa)
       or sum(signal) / n (for scalar kappa)
    5. new_weight = prev_weight + kappa * (signal - normalization)

    Args:
        prev_alpha: Previous running gradient (alpha) state
        prev_ewma: Previous EWMA (used in _calculateQuantAMMGradient for price - ewma)
        new_ewma: New EWMA (movingAverage in protocol) - passed to rule, used as denominator if not use_raw_price
        price: Current price data
        lamb: Lambda decay parameter
        prev_weights: Previous portfolio weights
        kappa: Scaling factor for weight updates
        exponents: Exponents for the power law transformation
        use_raw_price: If True, use price as denominator; if False, use ewma
        pre_exp_scaling: Scaling factor before exponentiation, default 0.5

    Returns:
        (new_weights, new_alpha, price_gradient)
    """
    # Compute constants (same as other gradient-based rules)
    G_inf = 1.0 / (1.0 - lamb)
    saturated_b = lamb / ((1 - lamb) ** 3)

    # Step 1: Compute new alpha (running gradient state)
    # Protocol: alpha_new = λ * alpha_old + (price - ewma) / (1 - λ)
    #         = λ * alpha_old + G_inf * (price - new_ewma)
    # Note: Protocol uses movingAverage (new_ewma) passed to the rule
    new_alpha = lamb * prev_alpha + G_inf * (price - new_ewma)

    # Step 2: Compute price gradient
    # Protocol: gradient_output = mulFactor * alpha_new
    #           where mulFactor = (1-λ)^3 / λ = 1 / saturated_b
    # Then: price_gradient = gradient_output / denominator
    #                      = alpha_new / (saturated_b * denominator)
    denominator = price if use_raw_price else new_ewma
    price_gradient = new_alpha / (saturated_b * denominator)

    # Steps 3-5: Use the simulator's weight update function which implements
    # the power law transformation
    raw_weight_updates = _jax_power_channel_weight_update(
        price_gradient,
        kappa,
        exponents,
        pre_exp_scaling=pre_exp_scaling,
    )

    new_weights = prev_weights + raw_weight_updates

    return new_weights, new_alpha, price_gradient


class TestPowerChannelWeightUpdate:
    """Tests for the power-channel weight update mechanism."""

    def test_zero_gradient_no_update(self):
        """Zero gradient should produce no weight update."""
        gradient = jnp.array([0.0, 0.0])
        k = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0])

        updates = _jax_power_channel_weight_update(gradient, k, exponents)

        assert jnp.allclose(updates, jnp.zeros(2), atol=1e-10), \
            "Zero gradient should produce zero weight updates"

    def test_weight_updates_sum_to_zero(self):
        """Weight updates should sum to zero (normalization)."""
        gradient = jnp.array([0.1, -0.2, 0.15])
        k = jnp.array([2048.0, 2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0, 3.0])

        updates = _jax_power_channel_weight_update(gradient, k, exponents)

        assert jnp.abs(jnp.sum(updates)) < 1e-10, \
            "Weight updates should sum to zero"

    def test_power_channel_follows_trend(self):
        """Positive gradient should produce positive weight update (trend following)."""
        gradient = jnp.array([0.1, -0.1])
        k = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0])
        prev_weights = jnp.array([0.5, 0.5])

        updates = _jax_power_channel_weight_update(gradient, k, exponents)
        new_weights = prev_weights + updates

        # Power channel follows trend: positive gradient -> positive update
        assert updates[0] > 0, \
            "Asset with positive gradient should get positive update (trend following)"
        assert updates[1] < 0, \
            "Asset with negative gradient should get negative update (trend following)"

    def test_higher_exponent_amplifies_small_gradients_less(self):
        """Higher exponents should amplify small gradients less (|x|^q < |x| for |x| < 1 and q > 1)."""
        gradient = jnp.array([0.1, -0.1])
        k = jnp.array([2048.0, 2048.0])
        exponents_low = jnp.array([2.0, 2.0])
        exponents_high = jnp.array([4.0, 4.0])

        updates_low = _jax_power_channel_weight_update(gradient, k, exponents_low)
        updates_high = _jax_power_channel_weight_update(gradient, k, exponents_high)

        # Higher exponent should produce smaller updates for small gradients
        assert jnp.abs(updates_high[0]) < jnp.abs(updates_low[0]), \
            "Higher exponent should produce smaller updates for small gradients"


class TestPowerChannelProtocolCases:
    """
    Integration tests matching protocol's QuantAMMPowerChannel.t.sol test cases.

    These tests use compute_protocol_power_channel_single_step which matches
    the protocol's single-step behavior.
    """

    def testCorrectWeightsWithHigherPrices(self):
        """
        QuantAMMPowerChannel.t.sol:261 - testCorrectWeightsWithHigherPrices

        Protocol values:
        - kappa=2048, exponents=3, use_raw_price=1
        - prevAlphas=[1,2], prevMovingAverages=[3,4], movingAverages=[3,4]
        - lambda=0.9, prevWeights=[0.5,0.5], data=[5,6]
        - expected: [0.500035215760277504, 0.499964784239722496]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=True
        )

        expected = jnp.array([0.500035215760277504, 0.499964784239722496])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:261"
        )

    def testCorrectWeightsWithLowerPrices(self):
        """
        QuantAMMPowerChannel.t.sol:314 - testCorrectWeightsWithLowerPrices

        Protocol values:
        - Same params but data=[2,4]
        - expected: [0.499867557750343680, 0.500132442249656320]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=True
        )

        expected = jnp.array([0.499867557750343680, 0.500132442249656320])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:314"
        )

    def testCorrectWeightsWithVectorParamsHigherPrices(self):
        """
        QuantAMMPowerChannel.t.sol:373 - testCorrectWeightsWithVectorParamsHigherPrices

        Protocol values (per-asset kappa):
        - kappa=[2048,32768], exponents=[3,3], use_raw_price=1
        - data=[5,6], expected: [0.500066288489934848, 0.499933711510077440]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 32768.0])
        exponents = jnp.array([3.0, 3.0])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=True
        )

        expected = jnp.array([0.500066288489934848, 0.499933711510077440])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:373"
        )

    def testCorrectWeightsWithVectorParamsLowerPrices(self):
        """
        QuantAMMPowerChannel.t.sol:429 - testCorrectWeightsWithVectorParamsLowerPrices

        Protocol values (per-asset kappa):
        - kappa=[2048,32768], exponents=[3,3], use_raw_price=1
        - data=[2,4], expected: [0.499750696941821952, 0.500249303058153472]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 32768.0])
        exponents = jnp.array([3.0, 3.0])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=True
        )

        expected = jnp.array([0.499750696941821952, 0.500249303058153472])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:429"
        )

    def testCorrectWeightsWithVectorParamsVectorqLowerPrices(self):
        """
        QuantAMMPowerChannel.t.sol:492 - testCorrectWeightsWithVectorParamsVectorqLowerPrices

        Protocol values (per-asset kappa and exponents):
        - kappa=[2048,32768], exponents=[2.5,3.5], use_raw_price=1
        - data=[2,4], expected: [0.496497130970298368, 0.503502869029683200]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 32768.0])
        exponents = jnp.array([2.5, 3.5])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=True
        )

        expected = jnp.array([0.496497130970298368, 0.503502869029683200])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:492"
        )

    def testCorrectWeightsWithVectorParamsVectorqHigherPrices(self):
        """
        QuantAMMPowerChannel.t.sol:552 - testCorrectWeightsWithVectorParamsVectorqHigherPrices

        Protocol values (per-asset kappa and exponents):
        - kappa=[2048,32768], exponents=[2.5,3.5], use_raw_price=1
        - data=[5,6], expected: [0.502825521901510656, 0.497174478098497536]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 32768.0])
        exponents = jnp.array([2.5, 3.5])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=True
        )

        expected = jnp.array([0.502825521901510656, 0.497174478098497536])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:552"
        )

    def testCorrectWeightsWithVectorqLowerPrices(self):
        """
        QuantAMMPowerChannel.t.sol:608 - testCorrectWeightsWithVectorqLowerPrices

        Protocol values (scalar kappa, per-asset exponents):
        - kappa=2048, exponents=[2.5,3.5], use_raw_price=1
        - data=[2,4], expected: [0.498139100827971584, 0.501860899172028416]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 2048.0])  # Scalar kappa expanded
        exponents = jnp.array([2.5, 3.5])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=True
        )

        expected = jnp.array([0.498139100827971584, 0.501860899172028416])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:608"
        )

    def testCorrectWeightsWithVectorqHigherPrices(self):
        """
        QuantAMMPowerChannel.t.sol:667 - testCorrectWeightsWithVectorqHigherPrices

        Protocol values (scalar kappa, per-asset exponents):
        - kappa=2048, exponents=[2.5,3.5], use_raw_price=1
        - data=[3,4] (same as ewma), expected: [0.500002074426347520, 0.499997925573652480]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([3.0, 4.0])  # Same as ewma
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([2.5, 3.5])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=True
        )

        expected = jnp.array([0.500002074426347520, 0.499997925573652480])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:667"
        )

    def testCorrectWeightsWithUseMovingAverageHigherPrices(self):
        """
        QuantAMMPowerChannel.t.sol:720 - testCorrectWeightsWithUseMovingAverageHigherPrices

        Protocol values:
        - kappa=2048, exponents=3, use_raw_price=0 (use moving average)
        - data=[5,6], expected: [0.500247564531423232, 0.499752435468576768]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0])

        new_weights, _, _ = compute_protocol_power_channel_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, exponents, use_raw_price=False  # Use moving average as denominator
        )

        expected = jnp.array([0.500247564531423232, 0.499752435468576768])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_POWER_CHANNEL_TEST_PATH}:720"
        )


class TestPowerChannelEdgeCases:
    """Edge case tests for power-channel behavior."""

    def test_uniform_gradient_no_update(self):
        """Uniform gradient across assets should produce no relative weight change."""
        gradient = jnp.array([0.1, 0.1, 0.1])
        k = jnp.array([2048.0, 2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0, 3.0])

        updates = _jax_power_channel_weight_update(gradient, k, exponents)

        # All updates should be equal (and sum to zero)
        assert jnp.allclose(updates[0], updates[1], atol=1e-10), \
            "Uniform gradient should produce equal weight updates"
        assert jnp.allclose(updates[1], updates[2], atol=1e-10), \
            "Uniform gradient should produce equal weight updates"

    def test_symmetric_opposite_gradients(self):
        """Opposite gradients should produce opposite weight updates."""
        gradient_pos = jnp.array([0.1, -0.1])
        gradient_neg = jnp.array([-0.1, 0.1])
        k = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0])

        updates_pos = _jax_power_channel_weight_update(gradient_pos, k, exponents)
        updates_neg = _jax_power_channel_weight_update(gradient_neg, k, exponents)

        # Updates should be opposite
        assert jnp.allclose(updates_pos, -updates_neg, atol=1e-10), \
            "Opposite gradients should produce opposite weight updates"

    def test_multiasset_power_channel(self):
        """Test power-channel with more than 2 assets."""
        gradient = jnp.array([0.1, -0.2, 0.05, -0.15])
        k = jnp.array([2048.0, 2048.0, 2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0, 3.0, 3.0])

        updates = _jax_power_channel_weight_update(gradient, k, exponents)

        assert jnp.abs(jnp.sum(updates)) < 1e-10, \
            "Weight updates should sum to zero for multiple assets"


class TestPowerChannelJIT:
    """Tests for JIT compilation of power-channel functions."""

    def test_weight_update_jittable(self):
        """The power-channel weight update function should be JIT-compilable."""
        @jit
        def jitted_update(gradient, k, exponents):
            return _jax_power_channel_weight_update(gradient, k, exponents)

        gradient = jnp.array([0.1, -0.2])
        k = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0])

        result = jitted_update(gradient, k, exponents)
        expected = _jax_power_channel_weight_update(gradient, k, exponents)

        assert jnp.allclose(result, expected), \
            "JIT-compiled version should match non-JIT version"

    def test_full_pipeline_jittable(self):
        """The full power-channel pipeline should be JIT-compilable."""
        @jit
        def jitted_pipeline(prev_alpha, new_ewma, price, lamb, prev_weights,
                           kappa, exponents):
            # Compute constants
            G_inf = 1.0 / (1.0 - lamb)
            saturated_b = lamb / ((1 - lamb) ** 3)

            # Compute alpha and gradient matching protocol
            new_alpha = lamb * prev_alpha + G_inf * (price - new_ewma)
            gradient = new_alpha / (saturated_b * price)

            raw_updates = _jax_power_channel_weight_update(
                gradient, kappa, exponents
            )

            return prev_weights + raw_updates

        prev_alpha = jnp.array([1.0, 2.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([2048.0, 2048.0])
        exponents = jnp.array([3.0, 3.0])

        result = jitted_pipeline(
            prev_alpha, new_ewma, price, lamb, prev_weights,
            kappa, exponents
        )

        # Should match expected from protocol test
        expected = jnp.array([0.500035215760277504, 0.499964784239722496])
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6,
            err_msg="JIT-compiled pipeline should produce correct result"
        )
