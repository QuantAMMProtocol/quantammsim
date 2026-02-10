"""
Tests for update rule estimators and primitives.

Tests cover:
- EMA calculation (EWMA)
- Gradient estimator (scan and convolution paths)
- Variance estimator
- Covariance estimator
- Squareplus activation
- Lambda/memory days conversion
"""
import pytest
import numpy as np
import jax.numpy as jnp
import jax
from jax.lax import scan
from jax.tree_util import Partial

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    squareplus,
    inverse_squareplus,
    _jax_ewma_scan_function,
    _jax_ewma_at_infinity_via_scan,
    _jax_gradient_scan_function,
    _jax_variance_at_infinity_via_conv_1D,
    make_ewma_kernel,
    make_a_kernel,
    lamb_to_memory,
    lamb_to_memory_days,
)
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
)


# ============================================================================
# Squareplus Activation Tests
# ============================================================================

class TestSquareplus:
    """Tests for squareplus activation function."""

    def test_squareplus_positive_output(self):
        """Test that squareplus always produces positive output."""
        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        y = squareplus(x)

        assert jnp.all(y > 0), "Squareplus should always be positive"

    def test_squareplus_monotonic(self):
        """Test that squareplus is monotonically increasing."""
        x = jnp.linspace(-10, 10, 100)
        y = squareplus(x)

        diffs = jnp.diff(y)
        assert jnp.all(diffs > 0), "Squareplus should be monotonically increasing"

    def test_squareplus_at_zero(self):
        """Test squareplus value at zero."""
        x = jnp.array([0.0])
        y = squareplus(x)

        # squareplus(0) = 0.5 * (0 + sqrt(0 + 4)) = 0.5 * 2 = 1.0
        np.testing.assert_allclose(
            y, 1.0, rtol=1e-6,
            err_msg="squareplus(0) should be 1.0"
        )

    def test_squareplus_large_input(self):
        """Test squareplus behavior for large inputs."""
        x = jnp.array([100.0])
        y = squareplus(x)

        # For large x, squareplus(x) ≈ x
        np.testing.assert_allclose(
            y, 100.0, rtol=0.01,
            err_msg="squareplus(x) ≈ x for large x"
        )

    def test_squareplus_inverse(self):
        """Test that inverse_squareplus is the inverse of squareplus."""
        x = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        y = squareplus(x)
        x_recovered = inverse_squareplus(y)

        np.testing.assert_allclose(
            x_recovered, x, rtol=1e-6,
            err_msg="inverse_squareplus should recover original value"
        )


# ============================================================================
# EWMA Tests
# ============================================================================

class TestEWMA:
    """Tests for Exponentially Weighted Moving Average."""

    def test_ewma_scan_function_basic(self):
        """Test basic EWMA scan function."""
        G_inf = 10.0  # 1 / (1 - lamb) where lamb = 0.9
        ewma_init = jnp.array([100.0])
        new_value = jnp.array([110.0])

        carry = [ewma_init]
        new_carry, ewma_out = _jax_ewma_scan_function(carry, new_value, G_inf)

        # EWMA update: ewma_new = ewma + (value - ewma) / G_inf
        expected = ewma_init + (new_value - ewma_init) / G_inf
        np.testing.assert_allclose(
            new_carry[0], expected, rtol=1e-6,
            err_msg="EWMA update should follow formula"
        )

    def test_ewma_converges_to_constant(self):
        """Test that EWMA converges to constant input."""
        lamb = 0.9
        constant_value = 100.0
        n = 100

        arr = jnp.ones((n, 1)) * constant_value
        ewma = _jax_ewma_at_infinity_via_scan(arr, jnp.array([lamb]))

        # After many steps, EWMA should be close to constant
        np.testing.assert_allclose(
            ewma[-1], constant_value, rtol=0.01,
            err_msg="EWMA should converge to constant input"
        )

    def test_ewma_shape(self):
        """Test EWMA output shape."""
        lamb = jnp.array([0.9, 0.95])
        arr = jnp.ones((50, 2)) * 100.0

        ewma = _jax_ewma_at_infinity_via_scan(arr, lamb)

        assert ewma.shape == (49, 2), f"Expected shape (49, 2), got {ewma.shape}"


# ============================================================================
# Gradient Estimator Tests
# ============================================================================

class TestGradientEstimator:
    """Tests for gradient estimator."""

    def test_gradient_scan_function_basic(self):
        """Test basic gradient scan function."""
        lamb = 0.9
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        ewma_init = jnp.array([100.0])
        running_a_init = jnp.array([0.0])
        new_value = jnp.array([101.0])

        carry = [ewma_init, running_a_init]
        new_carry, gradient = _jax_gradient_scan_function(
            carry, new_value, G_inf, lamb, saturated_b
        )

        # Gradient should be computed
        assert gradient.shape == (1,)
        assert not jnp.isnan(gradient[0]), "Gradient should not be NaN"

    def test_gradient_zero_for_constant_price(self):
        """Test that gradient is zero for constant prices."""
        lamb = 0.9
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        ewma = jnp.array([100.0])
        running_a = jnp.array([0.0])
        constant_price = jnp.array([100.0])

        # After many steps with constant price
        carry = [ewma, running_a]
        for _ in range(100):
            carry, gradient = _jax_gradient_scan_function(
                carry, constant_price, G_inf, lamb, saturated_b
            )

        # Gradient should be very close to zero
        np.testing.assert_allclose(
            gradient, 0.0, atol=1e-6,
            err_msg="Gradient should be zero for constant price"
        )

    def test_gradient_positive_for_uptrend(self):
        """Test that gradient is positive for uptrend."""
        lamb = 0.9
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        ewma = jnp.array([100.0])
        running_a = jnp.array([0.0])
        carry = [ewma, running_a]

        # Simulate uptrend
        for i in range(50):
            price = jnp.array([100.0 + i * 0.5])
            carry, gradient = _jax_gradient_scan_function(
                carry, price, G_inf, lamb, saturated_b
            )

        # Gradient should be positive for uptrend
        assert gradient[0] > 0, "Gradient should be positive for uptrend"

    def test_gradient_negative_for_downtrend(self):
        """Test that gradient is negative for downtrend."""
        lamb = 0.9
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        ewma = jnp.array([100.0])
        running_a = jnp.array([0.0])
        carry = [ewma, running_a]

        # Simulate downtrend
        for i in range(50):
            price = jnp.array([100.0 - i * 0.5])
            carry, gradient = _jax_gradient_scan_function(
                carry, price, G_inf, lamb, saturated_b
            )

        # Gradient should be negative for downtrend
        assert gradient[0] < 0, "Gradient should be negative for downtrend"


# ============================================================================
# Variance Estimator Tests
# ============================================================================

class TestVarianceEstimator:
    """Tests for variance estimator."""

    def test_variance_non_negative(self):
        """Test that variance is always non-negative."""
        np.random.seed(42)
        n = 100
        lamb = 0.9

        # Random price series
        arr_in = jnp.array(100.0 + np.random.normal(0, 1, n))

        # Compute EWMA first
        ewma = _jax_ewma_at_infinity_via_scan(
            arr_in.reshape(-1, 1), jnp.array([lamb])
        ).flatten()

        # Compute kernel
        kernel = jnp.array([lamb ** i for i in range(10)])

        variance = _jax_variance_at_infinity_via_conv_1D(
            arr_in, ewma, kernel, lamb
        )

        assert jnp.all(variance >= -1e-10), "Variance should be non-negative"

    def test_variance_zero_for_constant(self):
        """Test that variance is zero for constant prices."""
        n = 100
        lamb = 0.9
        constant = 100.0

        arr_in = jnp.ones(n) * constant

        # Compute EWMA (should be constant)
        ewma = jnp.ones(n - 1) * constant

        kernel = jnp.array([lamb ** i for i in range(10)])

        variance = _jax_variance_at_infinity_via_conv_1D(
            arr_in, ewma, kernel, lamb
        )

        # After initial transient, variance should be near zero
        np.testing.assert_allclose(
            variance[-1], 0.0, atol=1e-10,
            err_msg="Variance should be zero for constant prices"
        )


# ============================================================================
# Lambda/Memory Conversion Tests
# ============================================================================

class TestLambdaMemoryConversion:
    """Tests for lambda to memory days conversion."""

    def test_memory_days_to_lamb_basic(self):
        """Test basic memory days to lambda conversion."""
        memory_days = 10.0
        chunk_period = 1440  # 1 day

        lamb = memory_days_to_lamb(memory_days, chunk_period)

        assert 0 < lamb < 1, "Lambda should be between 0 and 1"

    def test_memory_days_to_lamb_longer_memory(self):
        """Test that longer memory gives higher lambda."""
        chunk_period = 1440

        lamb_short = memory_days_to_lamb(5.0, chunk_period)
        lamb_long = memory_days_to_lamb(20.0, chunk_period)

        assert lamb_long > lamb_short, \
            "Longer memory should give higher lambda"

    def test_lamb_to_memory_days_inverse(self):
        """Test that lamb_to_memory_days is inverse of memory_days_to_lamb."""
        chunk_period = 1440
        original_memory = 10.0

        lamb = memory_days_to_lamb(original_memory, chunk_period)
        recovered_memory = lamb_to_memory_days(lamb, chunk_period)

        np.testing.assert_allclose(
            recovered_memory, original_memory, rtol=0.01,
            err_msg="Should recover original memory days"
        )

    def test_lamb_to_memory_clipped(self):
        """Test memory clipping function."""
        chunk_period = 1440
        max_memory = 30.0

        # Very high lambda should be clipped
        lamb = 0.9999  # Very high memory

        memory = lamb_to_memory_days_clipped(lamb, chunk_period, max_memory)

        assert memory <= max_memory, "Memory should be clipped to max"


# ============================================================================
# Kernel Generation Tests
# ============================================================================

class TestKernelGeneration:
    """Tests for EWMA kernel generation."""

    def test_ewma_kernel_shape(self):
        """Test EWMA kernel has correct shape."""
        lamb = jnp.array([0.9, 0.95])
        max_memory_days = 30
        chunk_period = 1440

        kernel = make_ewma_kernel(lamb, max_memory_days, chunk_period)

        expected_len = int(max_memory_days * 1440 / chunk_period)
        assert kernel.shape == (expected_len, 2), \
            f"Expected shape ({expected_len}, 2), got {kernel.shape}"

    def test_ewma_kernel_decays(self):
        """Test that EWMA kernel decays."""
        lamb = jnp.array([0.9])
        max_memory_days = 30
        chunk_period = 1440

        kernel = make_ewma_kernel(lamb, max_memory_days, chunk_period)

        # Kernel should decay (first value > last value)
        assert kernel[0, 0] > kernel[-1, 0], \
            "Kernel should decay over time"

    def test_ewma_kernel_normalized(self):
        """Test EWMA kernel normalization properties."""
        lamb = jnp.array([0.9])
        max_memory_days = 100  # Long enough to capture most of the weight
        chunk_period = 1440

        kernel = make_ewma_kernel(lamb, max_memory_days, chunk_period)

        # Kernel is multiplied by (1 - lamb), so sum should be close to 1
        # for large enough max_memory_days
        kernel_sum = jnp.sum(kernel[:, 0])
        np.testing.assert_allclose(
            kernel_sum, 1.0, rtol=0.01,
            err_msg="EWMA kernel should sum to approximately 1"
        )


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases in estimators."""

    def test_very_small_lambda(self):
        """Test behavior with very small lambda (short memory)."""
        lamb = 0.1  # Very short memory
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        ewma = jnp.array([100.0])
        running_a = jnp.array([0.0])
        carry = [ewma, running_a]

        # Simulate a few steps
        for i in range(10):
            price = jnp.array([100.0 + i])
            carry, gradient = _jax_gradient_scan_function(
                carry, price, G_inf, lamb, saturated_b
            )

        # Should not produce NaN
        assert not jnp.isnan(gradient[0]), \
            "Small lambda should not produce NaN"

    def test_lambda_near_one(self):
        """Test behavior with lambda near 1 (long memory)."""
        lamb = 0.999  # Very long memory
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        ewma = jnp.array([100.0])
        running_a = jnp.array([0.0])
        carry = [ewma, running_a]

        # Simulate a few steps
        for i in range(10):
            price = jnp.array([100.0 + i * 0.01])
            carry, gradient = _jax_gradient_scan_function(
                carry, price, G_inf, lamb, saturated_b
            )

        # Should not produce NaN or Inf
        assert not jnp.isnan(gradient[0]), \
            "Lambda near 1 should not produce NaN"
        assert not jnp.isinf(gradient[0]), \
            "Lambda near 1 should not produce Inf"

    def test_multi_asset_gradient(self):
        """Test gradient computation with multiple assets."""
        lamb = 0.9
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        n_assets = 4
        ewma = jnp.array([100.0, 200.0, 50.0, 1000.0])
        running_a = jnp.zeros(n_assets)
        carry = [ewma, running_a]

        # Different trends for each asset
        prices = jnp.array([101.0, 198.0, 51.0, 1005.0])
        carry, gradient = _jax_gradient_scan_function(
            carry, prices, G_inf, lamb, saturated_b
        )

        assert gradient.shape == (n_assets,), \
            f"Expected shape ({n_assets},), got {gradient.shape}"
        assert not jnp.any(jnp.isnan(gradient)), \
            "Multi-asset gradient should not have NaN"


# ============================================================================
# JIT Compilation Tests
# ============================================================================

class TestJITCompilation:
    """Tests for JIT compilation of estimators."""

    def test_gradient_scan_jits(self):
        """Test that gradient scan function JITs properly."""
        lamb = 0.9
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)

        @jax.jit
        def run_scan(ewma_init, running_a_init, prices):
            scan_fn = Partial(
                _jax_gradient_scan_function,
                G_inf=G_inf, lamb=lamb, saturated_b=saturated_b
            )
            carry_init = [ewma_init, running_a_init]
            final_carry, gradients = scan(scan_fn, carry_init, prices)
            return gradients

        ewma_init = jnp.array([100.0])
        running_a_init = jnp.array([0.0])
        prices = jnp.array([[101.0], [102.0], [103.0]])

        # First call (compilation)
        _ = run_scan(ewma_init, running_a_init, prices)

        # Second call (should use cache)
        gradients = run_scan(ewma_init, running_a_init, prices)

        assert gradients.shape == (3, 1)

    def test_ewma_kernel_jits(self):
        """Test that EWMA kernel generation JITs properly."""
        lamb = jnp.array([0.9])
        max_memory_days = 30
        chunk_period = 1440

        # First call (compilation)
        _ = make_ewma_kernel(lamb, max_memory_days, chunk_period)

        # Second call (should use cache)
        kernel = make_ewma_kernel(lamb, max_memory_days, chunk_period)

        assert kernel is not None

    def test_squareplus_jits(self):
        """Test that squareplus JITs properly."""
        x = jnp.array([0.0, 1.0, 2.0])

        # First call (compilation)
        _ = squareplus(x)

        # Second call (should use cache)
        result = squareplus(x)

        assert result is not None
