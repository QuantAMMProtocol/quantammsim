"""
Tests for the channel-following (mean reversion channel) trading rule.

The channel-following rule combines mean reversion within a channel with trend
following outside the channel:
- Within channel: mean-revert towards the moving average
- Outside channel: follow the trend with power-law scaling

Formula:
  signal = channel_portion + trend_portion
  channel_portion = -amplitude * envelope * (scaled_g - scaled_g^3/6) / inverse_scaling
  trend_portion = (1-envelope) * sign(g) * |g / (2*pre_exp_scaling)|^exponents
  envelope = exp(-g^2 / (2*width^2))
  where g = price_gradient

These tests correspond to tests in:
QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMChannelFollowing.t.sol
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import jit

from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import (
    _jax_mean_reversion_channel_weight_update,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_gradient_scan_function,
)

# Protocol source file path
PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH = (
    "QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/QuantAMMChannelFollowing.t.sol"
)


def compute_protocol_channel_following_single_step(
    prev_alpha: jnp.ndarray,
    prev_ewma: jnp.ndarray,
    new_ewma: jnp.ndarray,
    price: jnp.ndarray,
    lamb: jnp.ndarray,
    prev_weights: jnp.ndarray,
    kappa: jnp.ndarray,
    width: jnp.ndarray,
    amplitude: jnp.ndarray,
    exponents: jnp.ndarray,
    inverse_scaling: jnp.ndarray,
    pre_exp_scaling: jnp.ndarray,
    use_raw_price: bool = True,
) -> tuple:
    """
    Single-step channel-following update matching protocol behavior exactly.

    The protocol's channel following calculation (from ChannelFollowingUpdateRule.sol):
    1. From _calculateQuantAMMGradient in QuantammGradientBasedRule.sol:
       - alpha_new = λ * alpha_old + (price - ewma) / (1 - λ)
       - gradient_output = mulFactor * alpha_new, where mulFactor = (1-λ)^3 / λ
    2. From ChannelFollowingUpdateRule.sol:
       - price_gradient = gradient_output / denominator
       - where denominator = price (if use_raw_price) or ewma (otherwise)
    3. Compute envelope: envelope = exp(-g^2 / (2*width^2))
    4. Compute scaled gradient: s = pi * g / (3 * width)
    5. Compute channel portion: -amplitude * envelope * (s - s^3/6) / inverse_scaling
    6. Compute trend portion: (1-envelope) * sign(g) * |g/(2*pre_exp_scaling)|^exponents
    7. signal = channel + trend
    8. normalization = sum(signal) / n (for scalar kappa) or sum(kappa*signal) / sum(kappa)
    9. new_weight = prev_weight + kappa * (signal - normalization)

    Args:
        prev_alpha: Previous running gradient (alpha) state
        prev_ewma: Previous EWMA (used in _calculateQuantAMMGradient for price - ewma)
        new_ewma: New EWMA (movingAverage in protocol) - passed to rule, used as denominator if not use_raw_price
        price: Current price data
        lamb: Lambda decay parameter
        prev_weights: Previous portfolio weights
        kappa: Scaling factor for weight updates
        width: Width parameter for the channel
        amplitude: Amplitude of the mean reversion effect
        exponents: Exponents for the trend following portion
        inverse_scaling: Scaling factor for channel portion
        pre_exp_scaling: Scaling factor before exponentiation
        use_raw_price: If True, use price as denominator; if False, use ewma

    Returns:
        (new_weights, new_alpha, price_gradient)
    """
    # Compute constants (same as anti-momentum)
    G_inf = 1.0 / (1.0 - lamb)
    saturated_b = lamb / ((1 - lamb) ** 3)

    # Step 1: Compute new alpha (running gradient state)
    # Protocol: alpha_new = λ * alpha_old + (price - ewma) / (1 - λ)
    #         = λ * alpha_old + G_inf * (price - new_ewma)
    # Note: Protocol uses movingAverage (new_ewma) passed to the rule, not prev_ewma
    new_alpha = lamb * prev_alpha + G_inf * (price - new_ewma)

    # Step 2: Compute price gradient
    # Protocol: gradient_output = mulFactor * alpha_new
    #           where mulFactor = (1-λ)^3 / λ = 1 / saturated_b
    # Then: price_gradient = gradient_output / denominator
    #                      = alpha_new / (saturated_b * denominator)
    denominator = price if use_raw_price else new_ewma
    price_gradient = new_alpha / (saturated_b * denominator)

    # Steps 3-9: Use the simulator's weight update function which implements
    # the same channel/trend formula
    raw_weight_updates = _jax_mean_reversion_channel_weight_update(
        price_gradient,
        kappa,
        width,
        amplitude,
        exponents,
        inverse_scaling=inverse_scaling,
        pre_exp_scaling=pre_exp_scaling,
    )

    new_weights = prev_weights + raw_weight_updates

    return new_weights, new_alpha, price_gradient


class TestChannelFollowingWeightUpdate:
    """Tests for the channel-following weight update mechanism."""

    def test_zero_gradient_no_update(self):
        """Zero gradient should produce no weight update."""
        gradient = jnp.array([0.0, 0.0])
        k = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])

        updates = _jax_mean_reversion_channel_weight_update(
            gradient, k, width, amplitude, exponents
        )

        assert jnp.allclose(updates, jnp.zeros(2), atol=1e-10), \
            "Zero gradient should produce zero weight updates"

    def test_weight_updates_sum_to_zero(self):
        """Weight updates should sum to zero (normalization)."""
        gradient = jnp.array([0.1, -0.2, 0.15])
        k = jnp.array([200.0, 200.0, 200.0])
        width = jnp.array([0.5, 0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1, 0.1])
        exponents = jnp.array([3.0, 3.0, 3.0])

        updates = _jax_mean_reversion_channel_weight_update(
            gradient, k, width, amplitude, exponents
        )

        assert jnp.abs(jnp.sum(updates)) < 1e-10, \
            "Weight updates should sum to zero"

    def test_channel_mean_reverts_small_gradient(self):
        """Small gradient within channel should trigger mean reversion (sell high)."""
        # Small positive gradient (price going up slightly)
        gradient = jnp.array([0.05, -0.05])
        k = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])  # Wide channel
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])

        updates = _jax_mean_reversion_channel_weight_update(
            gradient, k, width, amplitude, exponents
        )

        # Within channel, should mean-revert: positive gradient -> negative update (sell)
        # Due to normalization, we check relative behavior
        assert updates[0] < updates[1], \
            "Asset with positive gradient should get lower weight (mean reversion)"

    def test_trend_follows_large_gradient(self):
        """Large gradient outside channel should trigger trend following."""
        # Large positive gradient (price going up a lot)
        gradient = jnp.array([2.0, 0.0])
        k = jnp.array([200.0, 200.0])
        width = jnp.array([0.1, 0.1])  # Narrow channel
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([1.0, 1.0])  # Linear trend following

        updates = _jax_mean_reversion_channel_weight_update(
            gradient, k, width, amplitude, exponents
        )

        # Outside channel, should trend-follow: positive gradient -> positive update
        assert updates[0] > 0, \
            "Asset with large positive gradient should get positive update (trend following)"


class TestChannelFollowingProtocolCases:
    """
    Integration tests matching protocol's QuantAMMChannelFollowing.t.sol test cases.

    These tests use compute_protocol_channel_following_single_step which matches
    the protocol's single-step behavior.
    """

    def testCorrectWeightsWithHigherPrices(self):
        """
        QuantAMMChannelFollowing.t.sol:221 - testCorrectWeightsWithHigherPrices

        Protocol values:
        - kappa=200, width=0.5, amplitude=0.1, exponents=3
        - inverse_scaling=1, pre_exp_scaling=0.5, use_raw_price=1
        - prevAlphas=[1,2], prevMovingAverages=[3,4], movingAverages=[3,4]
        - lambda=0.9, prevWeights=[0.5,0.5], data=[5,6]
        - expected: [0.487280456264591400, 0.512719543735408600]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])
        inverse_scaling = jnp.array([1.0, 1.0])
        pre_exp_scaling = jnp.array([0.5, 0.5])

        new_weights, _, _ = compute_protocol_channel_following_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inverse_scaling, pre_exp_scaling,
            use_raw_price=True
        )

        expected = jnp.array([0.487280456264591400, 0.512719543735408600])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH}:221"
        )

    def testCorrectWeightsWithLowerPrices(self):
        """
        QuantAMMChannelFollowing.t.sol:282 - testCorrectWeightsWithLowerPrices

        Protocol values:
        - Same params as above but data=[2,4]
        - expected: [0.616347884950068200, 0.383652115049931800]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])
        inverse_scaling = jnp.array([1.0, 1.0])
        pre_exp_scaling = jnp.array([0.5, 0.5])

        new_weights, _, _ = compute_protocol_channel_following_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inverse_scaling, pre_exp_scaling,
            use_raw_price=True
        )

        expected = jnp.array([0.616347884950068200, 0.383652115049931800])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH}:282"
        )

    def testCorrectWeightsWithVectorParamsHigherPrices(self):
        """
        QuantAMMChannelFollowing.t.sol:349 - testCorrectWeightsWithVectorParamsHigherPrices

        Protocol values (per-asset):
        - kappa=[200,400], width=[0.5,1.5], amplitude=[0.1,0.2]
        - exponents=[2.5,3.5], inverse_scaling=[0.8,1], pre_exp_scaling=[0.5,0.3]
        - data=[5,6], expected: [0.413044386615075800, 0.586955613384924000]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 400.0])
        width = jnp.array([0.5, 1.5])
        amplitude = jnp.array([0.1, 0.2])
        exponents = jnp.array([2.5, 3.5])
        inverse_scaling = jnp.array([0.8, 1.0])
        pre_exp_scaling = jnp.array([0.5, 0.3])

        new_weights, _, _ = compute_protocol_channel_following_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inverse_scaling, pre_exp_scaling,
            use_raw_price=True
        )

        expected = jnp.array([0.413044386615075800, 0.586955613384924000])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH}:349"
        )

    def testCorrectWeightsWithVectorParamsLowerPrices(self):
        """
        QuantAMMChannelFollowing.t.sol:416 - testCorrectWeightsWithVectorParamsLowerPrices

        Protocol values (per-asset):
        - kappa=[200,400], width=[0.5,1.5], amplitude=[0.1,0.2]
        - exponents=[2.5,3.5], inverse_scaling=[0.8,1], pre_exp_scaling=[0.5,0.3]
        - data=[2,4], expected: [0.685768271666430200, 0.314231728333570000]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 400.0])
        width = jnp.array([0.5, 1.5])
        amplitude = jnp.array([0.1, 0.2])
        exponents = jnp.array([2.5, 3.5])
        inverse_scaling = jnp.array([0.8, 1.0])
        pre_exp_scaling = jnp.array([0.5, 0.3])

        new_weights, _, _ = compute_protocol_channel_following_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inverse_scaling, pre_exp_scaling,
            use_raw_price=True
        )

        expected = jnp.array([0.685768271666430200, 0.314231728333570000])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH}:416"
        )

    def testCorrectWeightsWithUseMovingAverageHigherPrices(self):
        """
        QuantAMMChannelFollowing.t.sol:489 - testCorrectWeightsWithUseMovingAverageHigherPrices

        Protocol values:
        - Same as testCorrectWeightsWithHigherPrices but use_raw_price=0
        - data=[5,6], expected: [0.464719395233870000, 0.535280604766130000]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])
        inverse_scaling = jnp.array([1.0, 1.0])
        pre_exp_scaling = jnp.array([0.5, 0.5])

        new_weights, _, _ = compute_protocol_channel_following_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inverse_scaling, pre_exp_scaling,
            use_raw_price=False  # Use moving average as denominator
        )

        expected = jnp.array([0.464719395233870000, 0.535280604766130000])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH}:489"
        )

    def testCorrectWeightsWithUseMovingAverageLowerPrice(self):
        """
        QuantAMMChannelFollowing.t.sol:551 - testCorrectWeightsWithUseMovingAverageLowerPrice

        Protocol values:
        - data=[2,4], use_raw_price=0
        - expected: [0.581058650366410200, 0.418941349633589800]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])
        inverse_scaling = jnp.array([1.0, 1.0])
        pre_exp_scaling = jnp.array([0.5, 0.5])

        new_weights, _, _ = compute_protocol_channel_following_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inverse_scaling, pre_exp_scaling,
            use_raw_price=False
        )

        expected = jnp.array([0.581058650366410200, 0.418941349633589800])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH}:551"
        )

    def testCorrectWeightsWithVectorParamsWithUseMovingAverageHigherPrices(self):
        """
        QuantAMMChannelFollowing.t.sol:618 - testCorrectWeightsWithVectorParamsWithUseMovingAverageHigherPrices

        Protocol values (per-asset, use_raw_price=0):
        - data=[3,4] (same as ewma, so gradient is prev_alpha)
        - expected: [0.497672897155407600, 0.502327102844592400]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([3.0, 4.0])  # Same as ewma
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 400.0])
        width = jnp.array([0.5, 1.5])
        amplitude = jnp.array([0.1, 0.2])
        exponents = jnp.array([2.5, 3.5])
        inverse_scaling = jnp.array([0.8, 1.0])
        pre_exp_scaling = jnp.array([0.5, 0.3])

        new_weights, _, _ = compute_protocol_channel_following_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inverse_scaling, pre_exp_scaling,
            use_raw_price=False
        )

        expected = jnp.array([0.497672897155407600, 0.502327102844592400])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH}:618"
        )

    def testCorrectWeightsWithVectorParamsUseMovingAverageLowerPrice(self):
        """
        QuantAMMChannelFollowing.t.sol:685 - testCorrectWeightsWithVectorParamsUseMovingAverageLowerPrice

        Protocol values (per-asset, use_raw_price=0):
        - data=[2,4], expected: [0.626952890125817400, 0.373047109874182800]
        """
        prev_alpha = jnp.array([1.0, 2.0])
        prev_ewma = jnp.array([3.0, 4.0])
        new_ewma = jnp.array([3.0, 4.0])
        price = jnp.array([2.0, 4.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 400.0])
        width = jnp.array([0.5, 1.5])
        amplitude = jnp.array([0.1, 0.2])
        exponents = jnp.array([2.5, 3.5])
        inverse_scaling = jnp.array([0.8, 1.0])
        pre_exp_scaling = jnp.array([0.5, 0.3])

        new_weights, _, _ = compute_protocol_channel_following_single_step(
            prev_alpha, prev_ewma, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inverse_scaling, pre_exp_scaling,
            use_raw_price=False
        )

        expected = jnp.array([0.626952890125817400, 0.373047109874182800])
        np.testing.assert_array_almost_equal(
            new_weights, expected, decimal=6,
            err_msg=f"Protocol test - {PROTOCOL_CHANNEL_FOLLOWING_TEST_PATH}:685"
        )


class TestChannelFollowingEdgeCases:
    """Edge case tests for channel-following behavior."""

    def test_uniform_gradient_no_update(self):
        """Uniform gradient across assets should produce no relative weight change."""
        gradient = jnp.array([0.1, 0.1, 0.1])
        k = jnp.array([200.0, 200.0, 200.0])
        width = jnp.array([0.5, 0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1, 0.1])
        exponents = jnp.array([3.0, 3.0, 3.0])

        updates = _jax_mean_reversion_channel_weight_update(
            gradient, k, width, amplitude, exponents
        )

        # All updates should be equal (and sum to zero)
        assert jnp.allclose(updates[0], updates[1], atol=1e-10), \
            "Uniform gradient should produce equal weight updates"
        assert jnp.allclose(updates[1], updates[2], atol=1e-10), \
            "Uniform gradient should produce equal weight updates"

    def test_symmetric_opposite_gradients(self):
        """Opposite gradients should produce opposite weight updates."""
        gradient_pos = jnp.array([0.1, -0.1])
        gradient_neg = jnp.array([-0.1, 0.1])
        k = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])

        updates_pos = _jax_mean_reversion_channel_weight_update(
            gradient_pos, k, width, amplitude, exponents
        )
        updates_neg = _jax_mean_reversion_channel_weight_update(
            gradient_neg, k, width, amplitude, exponents
        )

        # Updates should be opposite
        assert jnp.allclose(updates_pos, -updates_neg, atol=1e-10), \
            "Opposite gradients should produce opposite weight updates"

    def test_multiasset_channel_following(self):
        """Test channel-following with more than 2 assets."""
        gradient = jnp.array([0.1, -0.2, 0.05, -0.15])
        k = jnp.array([200.0, 200.0, 200.0, 200.0])
        width = jnp.array([0.5, 0.5, 0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1, 0.1, 0.1])
        exponents = jnp.array([3.0, 3.0, 3.0, 3.0])

        updates = _jax_mean_reversion_channel_weight_update(
            gradient, k, width, amplitude, exponents
        )

        assert jnp.abs(jnp.sum(updates)) < 1e-10, \
            "Weight updates should sum to zero for multiple assets"


class TestChannelFollowingJIT:
    """Tests for JIT compilation of channel-following functions."""

    def test_weight_update_jittable(self):
        """The channel-following weight update function should be JIT-compilable."""
        @jit
        def jitted_update(gradient, k, width, amplitude, exponents):
            return _jax_mean_reversion_channel_weight_update(
                gradient, k, width, amplitude, exponents
            )

        gradient = jnp.array([0.1, -0.2])
        k = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])

        result = jitted_update(gradient, k, width, amplitude, exponents)
        expected = _jax_mean_reversion_channel_weight_update(
            gradient, k, width, amplitude, exponents
        )

        assert jnp.allclose(result, expected), \
            "JIT-compiled version should match non-JIT version"

    def test_full_pipeline_jittable(self):
        """The full channel-following pipeline should be JIT-compilable."""
        @jit
        def jitted_pipeline(prev_alpha, new_ewma, price, lamb, prev_weights,
                           kappa, width, amplitude, exponents, inv_scaling, pre_exp):
            # Compute constants
            G_inf = 1.0 / (1.0 - lamb)
            saturated_b = lamb / ((1 - lamb) ** 3)

            # Compute alpha and gradient matching protocol
            new_alpha = lamb * prev_alpha + G_inf * (price - new_ewma)
            gradient = new_alpha / (saturated_b * price)

            raw_updates = _jax_mean_reversion_channel_weight_update(
                gradient, kappa, width, amplitude, exponents,
                inverse_scaling=inv_scaling, pre_exp_scaling=pre_exp
            )

            return prev_weights + raw_updates

        prev_alpha = jnp.array([1.0, 2.0])
        new_ewma = jnp.array([3.0, 4.0])  # Same as protocol test
        price = jnp.array([5.0, 6.0])
        lamb = jnp.array([0.9, 0.9])
        prev_weights = jnp.array([0.5, 0.5])

        kappa = jnp.array([200.0, 200.0])
        width = jnp.array([0.5, 0.5])
        amplitude = jnp.array([0.1, 0.1])
        exponents = jnp.array([3.0, 3.0])
        inv_scaling = jnp.array([1.0, 1.0])
        pre_exp = jnp.array([0.5, 0.5])

        result = jitted_pipeline(
            prev_alpha, new_ewma, price, lamb, prev_weights,
            kappa, width, amplitude, exponents, inv_scaling, pre_exp
        )

        # Should match expected from protocol test
        expected = jnp.array([0.487280456264591400, 0.512719543735408600])
        np.testing.assert_array_almost_equal(
            result, expected, decimal=6,
            err_msg="JIT-compiled pipeline should produce correct result"
        )
