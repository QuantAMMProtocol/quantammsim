"""
Unit tests for estimator primitives - the low-level scan functions for gradient calculation.

These tests verify the core estimator functions that are shared across all strategy rules:
- EWMA (exponential weighted moving average) calculation
- Alpha (running_a) accumulation
- Gradient calculation

The estimator primitives are used by momentum, anti-momentum, min-variance, and other rules.

Protocol reference:
QuantAMM-V1/pkg/pool-quantamm/test/foundry/rules/base/Gradient.t.sol
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax
from typing import Union, Tuple

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_gradient_scan_function,
    _jax_gradients_at_infinity_via_scan,
    _jax_gradients_at_infinity_via_scan_with_readout,
)


# =============================================================================
# Helper functions for gradient calculations (shared across rules)
# =============================================================================

def single_step_gradient_update(
    previous_alpha: jnp.ndarray,
    prev_moving_average: jnp.ndarray,
    moving_average: jnp.ndarray,
    lamb: Union[float, jnp.ndarray],
    data: jnp.ndarray,
    use_raw_price: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute a single step of the gradient estimator, matching the protocol.

    This mirrors the Solidity CalculateUnguardedWeights logic for a single timestep.

    Args:
        previous_alpha: Previous alpha (running_a) values, shape (n_assets,)
        prev_moving_average: Previous EWMA values (not used in this formula, but kept for API)
        moving_average: Current EWMA values, shape (n_assets,)
        lamb: Lambda decay parameter, scalar or shape (n_assets,)
        data: Current price data, shape (n_assets,)
        use_raw_price: If True, divide by price; if False, divide by EWMA

    Returns:
        Tuple of (new_alpha, gradient, denom) where:
        - new_alpha: Updated alpha values
        - gradient: Price gradient (beta / denom)
        - denom: Denominator used (price or EWMA)
    """
    n_assets = len(data)
    if isinstance(lamb, (float, int)):
        lamb = jnp.array([lamb] * n_assets)
    else:
        lamb = jnp.array(lamb)

    # Protocol formula for saturated_b and G_inf
    saturated_b = lamb / ((1 - lamb) ** 3)
    G_inf = 1.0 / (1.0 - lamb)

    # Update running_a (alpha)
    new_alpha = lamb * previous_alpha + G_inf * (data - moving_average)

    # Compute gradient (beta / denom)
    if use_raw_price:
        denom = data
    else:
        denom = moving_average

    gradient = new_alpha / (saturated_b * denom)

    return new_alpha, gradient, denom


# =============================================================================
# Test Classes - Estimator Primitives (Single-Step Scan Functions)
# =============================================================================

class TestEstimatorPrimitives:
    """
    Tests for the low-level estimator primitive scan functions.

    These tests directly verify the _jax_gradient_scan_function which is
    the core single-step update matching the protocol's UpdateWeightRunner logic.

    Key mappings between protocol and simulator:
    - Protocol's `alpha` (previousAlphas) -> Simulator's `running_a`
    - Protocol's `movingAverage` -> Simulator's `ewma`
    - Protocol's `beta` -> Simulator's `gradient` (gradient = alpha / (saturated_b * denom))
    """

    def test_single_step_scan_function_basic(self):
        """
        Test _jax_gradient_scan_function with basic inputs.

        The scan function takes:
        - carry_list: [ewma, running_a]
        - arr_in: current price
        - G_inf, lamb, saturated_b: constants

        And returns:
        - new_carry: [new_ewma, new_running_a]
        - gradient
        """
        lamb = jnp.array([0.7, 0.7])
        G_inf = 1.0 / (1.0 - lamb)  # [3.333..., 3.333...]
        saturated_b = lamb / ((1 - lamb) ** 3)  # [25.926..., 25.926...]

        # Initial state
        ewma_init = jnp.array([0.9, 1.2])
        running_a_init = jnp.array([1.0, 2.0])
        carry_list_init = [ewma_init, running_a_init]

        # Current price
        price = jnp.array([3.0, 4.0])

        # Run single step
        new_carry, gradient = _jax_gradient_scan_function(
            carry_list_init, price, G_inf, lamb, saturated_b
        )

        new_ewma, new_running_a = new_carry

        # Manual calculation:
        # ewma_new = ewma + (price - ewma) / G_inf
        expected_ewma = ewma_init + (price - ewma_init) / G_inf

        # running_a_new = lamb * running_a + G_inf * (price - ewma_new)
        expected_running_a = lamb * running_a_init + G_inf * (price - expected_ewma)

        # gradient = running_a / (saturated_b * ewma)
        expected_gradient = expected_running_a / (saturated_b * expected_ewma)

        np.testing.assert_array_almost_equal(
            new_ewma, expected_ewma, decimal=10,
            err_msg="EWMA update mismatch in scan function"
        )
        np.testing.assert_array_almost_equal(
            new_running_a, expected_running_a, decimal=10,
            err_msg="running_a update mismatch in scan function"
        )
        np.testing.assert_array_almost_equal(
            gradient, expected_gradient, decimal=10,
            err_msg="gradient mismatch in scan function"
        )

    def test_scan_function_matches_protocol_formula(self):
        """
        Verify the scan function exactly matches the protocol's formula.

        Protocol UpdateWeightRunner.sol calculates:
        - movingAverage[i] = prevMA[i] + (data[i] - prevMA[i]) / Ginf
        - alpha[i] = lambda * prevAlpha[i] + Ginf * (data[i] - movingAverage[i])
        - beta[i] = alpha[i] / (saturatedB * denominator)
        """
        lamb = jnp.array([0.7])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        prev_ewma = jnp.array([0.9])
        prev_alpha = jnp.array([1.0])
        price = jnp.array([3.0])

        new_carry, gradient = _jax_gradient_scan_function(
            [prev_ewma, prev_alpha], price, G_inf, lamb, saturated_b
        )
        new_ewma, new_alpha = new_carry

        # Protocol formulas
        expected_new_ewma = prev_ewma + (price - prev_ewma) / G_inf
        expected_new_alpha = lamb * prev_alpha + G_inf * (price - expected_new_ewma)
        expected_gradient = expected_new_alpha / (saturated_b * expected_new_ewma)

        np.testing.assert_array_almost_equal(new_ewma, expected_new_ewma, decimal=12)
        np.testing.assert_array_almost_equal(new_alpha, expected_new_alpha, decimal=12)
        np.testing.assert_array_almost_equal(gradient, expected_gradient, decimal=12)

    def test_scan_with_custom_initial_state(self):
        """
        Test that custom initial state can be used via manual single-step iteration.

        The _jax_gradients_at_infinity_via_scan function's carry_list_init is
        marked as static (for JIT), so we can't pass JAX arrays directly.
        Instead, we verify the capability via manual single-step iteration.
        """
        lamb = jnp.array([0.7, 0.7])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        # Price sequence (3 timesteps)
        prices = jnp.array([
            [1.0, 1.5],  # t=0
            [1.1, 1.6],  # t=1
            [1.2, 1.4],  # t=2
        ])

        # Custom initial state
        custom_ewma = jnp.array([0.9, 1.4])
        custom_running_a = jnp.array([0.5, 0.8])

        # Use manual iteration with custom initial state
        ewma = custom_ewma
        running_a = custom_running_a
        gradients = [jnp.zeros(2)]

        for price in prices[1:]:
            new_carry, gradient = _jax_gradient_scan_function(
                [ewma, running_a], price, G_inf, lamb, saturated_b
            )
            ewma, running_a = new_carry
            gradients.append(gradient)

        gradients = jnp.stack(gradients)

        assert gradients.shape == (3, 2), f"Expected shape (3, 2), got {gradients.shape}"
        np.testing.assert_array_equal(gradients[0], jnp.zeros(2))
        assert not jnp.allclose(gradients[1:], 0.0), "Gradients should be non-zero"

    def test_scan_with_readout_returns_full_state(self):
        """
        Test _jax_gradients_at_infinity_via_scan_with_readout returns all state.

        This function returns a dict with gradients, ewma, and running_a,
        allowing verification of intermediate state.
        """
        lamb = jnp.array([0.7, 0.7])

        prices = jnp.array([
            [1.0, 1.5],
            [1.1, 1.6],
            [1.2, 1.4],
            [1.3, 1.5],
        ])

        result = _jax_gradients_at_infinity_via_scan_with_readout(prices, lamb)

        assert "gradients" in result
        assert "ewma" in result
        assert "running_a" in result

        # gradients has shape (n_timesteps, n_assets) with zeros prepended
        assert result["gradients"].shape == (4, 2)

        # ewma and running_a have shape (n_timesteps-1, n_assets) (scan output)
        assert result["ewma"].shape == (3, 2)
        assert result["running_a"].shape == (3, 2)

    def test_readout_state_progression(self):
        """
        Verify ewma and running_a progress correctly through the scan.
        """
        lamb = jnp.array([0.8])  # Single asset for simplicity
        G_inf = 1.0 / (1.0 - lamb)

        prices = jnp.array([
            [100.0],  # t=0
            [105.0],  # t=1
            [102.0],  # t=2
        ])

        result = _jax_gradients_at_infinity_via_scan_with_readout(prices, lamb)

        ewma_series = result["ewma"]

        # Initial ewma is prices[0] = 100
        # After t=1: ewma = 100 + (105 - 100) / 5 = 101
        expected_ewma_t1 = 100.0 + (105.0 - 100.0) / G_inf[0]
        np.testing.assert_almost_equal(ewma_series[0, 0], expected_ewma_t1, decimal=10)

        # After t=2: ewma = 101 + (102 - 101) / 5 = 101.2
        expected_ewma_t2 = expected_ewma_t1 + (102.0 - expected_ewma_t1) / G_inf[0]
        np.testing.assert_almost_equal(ewma_series[1, 0], expected_ewma_t2, decimal=10)

    def test_multi_step_consistency(self):
        """
        Verify that multiple single steps match one scan call.
        """
        lamb = jnp.array([0.7, 0.7])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        prices = jnp.array([
            [1.0, 2.0],  # t=0
            [1.1, 2.1],  # t=1
            [1.2, 1.9],  # t=2
            [1.15, 2.0], # t=3
        ])

        # Method 1: Use the scan function
        result_scan = _jax_gradients_at_infinity_via_scan_with_readout(prices, lamb)

        # Method 2: Manual single steps
        # Note: running_a must be initialized to zeros to match the scan function
        # (zeros is the steady-state for constant input)
        ewma = prices[0]
        running_a = jnp.zeros(2)
        gradients_manual = [jnp.zeros(2)]

        for price in prices[1:]:
            new_carry, gradient = _jax_gradient_scan_function(
                [ewma, running_a], price, G_inf, lamb, saturated_b
            )
            ewma, running_a = new_carry
            gradients_manual.append(gradient)

        gradients_manual = jnp.stack(gradients_manual)

        np.testing.assert_array_almost_equal(
            result_scan["gradients"], gradients_manual, decimal=10,
            err_msg="Scan and manual single-step should match"
        )


# =============================================================================
# Test Classes - Gradient Calculation
# =============================================================================

class TestGradientCalculation:
    """Test the gradient (beta/denom) calculation."""

    def test_gradient_formula_components(self):
        """Verify the individual components of the gradient formula."""
        lamb = 0.7
        previous_alpha = jnp.array([1.0, 2.0])
        moving_average = jnp.array([0.9, 1.2])
        data = jnp.array([3.0, 4.0])

        # Compute expected values manually
        saturated_b = lamb / ((1 - lamb) ** 3)  # 0.7 / 0.027 = 25.926...
        G_inf = 1.0 / (1.0 - lamb)  # 1 / 0.3 = 3.333...

        expected_new_alpha = lamb * previous_alpha + G_inf * (data - moving_average)

        new_alpha, gradient, denom = single_step_gradient_update(
            previous_alpha=previous_alpha,
            prev_moving_average=jnp.zeros(2),
            moving_average=moving_average,
            lamb=lamb,
            data=data,
            use_raw_price=True,
        )

        np.testing.assert_array_almost_equal(
            new_alpha, expected_new_alpha, decimal=5,
            err_msg="Alpha calculation mismatch"
        )

        # Gradient = alpha / (saturated_b * denom)
        expected_gradient = expected_new_alpha / (saturated_b * data)
        np.testing.assert_array_almost_equal(
            gradient, expected_gradient, decimal=5,
            err_msg="Gradient calculation mismatch"
        )

    def test_use_raw_price_vs_ewma_denominator(self):
        """Verify the difference between useRawPrice=True and False."""
        lamb = 0.7
        previous_alpha = jnp.array([1.0, 2.0])
        moving_average = jnp.array([0.9, 1.2])
        data = jnp.array([3.0, 4.0])

        _, gradient_raw, denom_raw = single_step_gradient_update(
            previous_alpha=previous_alpha,
            prev_moving_average=jnp.zeros(2),
            moving_average=moving_average,
            lamb=lamb,
            data=data,
            use_raw_price=True,
        )

        _, gradient_ewma, denom_ewma = single_step_gradient_update(
            previous_alpha=previous_alpha,
            prev_moving_average=jnp.zeros(2),
            moving_average=moving_average,
            lamb=lamb,
            data=data,
            use_raw_price=False,
        )

        # Denominators should be different
        np.testing.assert_array_equal(denom_raw, data)
        np.testing.assert_array_equal(denom_ewma, moving_average)

        # Since gradient = alpha / (saturated_b * denom), we have:
        # gradient_ewma / gradient_raw = denom_raw / denom_ewma
        ratio = denom_raw / denom_ewma
        np.testing.assert_array_almost_equal(
            gradient_raw * ratio, gradient_ewma, decimal=10,
            err_msg="Gradient ratio should match inverse denominator ratio"
        )

    def test_gradient_sign_follows_price_vs_ewma(self):
        """Gradient sign should reflect whether price is above or below EWMA."""
        lamb = jnp.array([0.7])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        prev_ewma = jnp.array([100.0])
        prev_alpha = jnp.array([0.0])  # Start with zero alpha

        # Price above EWMA
        price_high = jnp.array([110.0])
        new_carry_high, grad_high = _jax_gradient_scan_function(
            [prev_ewma, prev_alpha], price_high, G_inf, lamb, saturated_b
        )

        # Price below EWMA
        price_low = jnp.array([90.0])
        new_carry_low, grad_low = _jax_gradient_scan_function(
            [prev_ewma, prev_alpha], price_low, G_inf, lamb, saturated_b
        )

        # High price should give positive gradient, low price negative
        assert grad_high[0] > 0, "Price above EWMA should give positive gradient"
        assert grad_low[0] < 0, "Price below EWMA should give negative gradient"

    def test_ewma_converges_to_constant_price(self):
        """EWMA should converge to price when price is constant."""
        lamb = jnp.array([0.9])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        constant_price = 100.0
        n_steps = 100

        ewma = jnp.array([50.0])  # Start far from price
        running_a = jnp.array([0.0])

        for _ in range(n_steps):
            new_carry, _ = _jax_gradient_scan_function(
                [ewma, running_a], jnp.array([constant_price]), G_inf, lamb, saturated_b
            )
            ewma, running_a = new_carry

        # EWMA should be very close to constant price
        np.testing.assert_almost_equal(
            ewma[0], constant_price, decimal=1,
            err_msg="EWMA should converge to constant price"
        )


# =============================================================================
# Test Classes - JIT Compilation for Primitives
# =============================================================================

class TestPrimitivesJIT:
    """Test that primitive functions work correctly when JIT compiled."""

    def test_scan_function_jit(self):
        """Test _jax_gradient_scan_function works with JIT."""
        @jax.jit
        def jitted_scan(carry, price, G_inf, lamb, saturated_b):
            return _jax_gradient_scan_function(carry, price, G_inf, lamb, saturated_b)

        lamb = jnp.array([0.7])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        carry = [jnp.array([1.0]), jnp.array([0.5])]
        price = jnp.array([1.1])

        new_carry, gradient = jitted_scan(carry, price, G_inf, lamb, saturated_b)

        assert jnp.all(jnp.isfinite(new_carry[0]))
        assert jnp.all(jnp.isfinite(new_carry[1]))
        assert jnp.all(jnp.isfinite(gradient))

    def test_gradient_flow_through_scan(self):
        """Test that gradients can be computed through the scan function."""
        lamb = jnp.array([0.7])
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        def loss_fn(prev_alpha):
            carry = [jnp.array([1.0]), prev_alpha]
            price = jnp.array([1.1])
            new_carry, gradient = _jax_gradient_scan_function(
                carry, price, G_inf, lamb, saturated_b
            )
            return jnp.sum(gradient ** 2)

        grad = jax.grad(loss_fn)(jnp.array([0.5]))
        assert jnp.all(jnp.isfinite(grad))
