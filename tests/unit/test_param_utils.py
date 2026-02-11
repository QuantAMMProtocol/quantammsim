"""
Tests for param_utils conversion functions.

Covers edge cases in _to_bd18_string_list and _to_float64_list,
particularly scalar / 0-d array inputs that arise when
base_pool.calculate_weights returns a 1-D array for static pools.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from quantammsim.core_simulator.param_utils import (
    _to_bd18_string_list,
    _to_float64_list,
)


class TestToBd18StringList:
    """Tests for _to_bd18_string_list."""

    def test_list_input(self):
        """Standard list of floats should convert correctly."""
        result = _to_bd18_string_list([0.5, 0.5])
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_1d_array_input(self):
        """1-D JAX array should convert correctly."""
        result = _to_bd18_string_list(jnp.array([0.5, 0.5]))
        assert len(result) == 2

    def test_1d_numpy_array_input(self):
        """1-D numpy array should convert correctly."""
        result = _to_bd18_string_list(np.array([0.5, 0.5]))
        assert len(result) == 2

    def test_0d_jax_array_input(self):
        """0-D JAX array (scalar) should not raise TypeError.

        This is the exact failure mode from run_pool_simulation with
        Balancer pools where weights[-1] produces a 0-d array.
        """
        scalar = jnp.array(0.5)
        assert scalar.ndim == 0, "Precondition: input must be 0-d"
        result = _to_bd18_string_list(scalar)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_0d_numpy_array_input(self):
        """0-D numpy array should not raise TypeError."""
        scalar = np.float64(0.5)
        result = _to_bd18_string_list(scalar)
        assert len(result) == 1

    def test_python_float_input(self):
        """Plain Python float should be handled."""
        result = _to_bd18_string_list(0.5)
        assert len(result) == 1

    def test_known_value(self):
        """Check BD18 conversion produces expected string for 0.5."""
        result = _to_bd18_string_list([0.5])
        # 0.5 * 1e18 = 500000000000000000
        assert result[0] == "500000000000000000"

    def test_zero(self):
        """Zero should produce '0'."""
        result = _to_bd18_string_list([0.0])
        assert result[0] == "0"


class TestToFloat64List:
    """Tests for _to_float64_list — verify it handles edge cases for comparison."""

    def test_0d_jax_array(self):
        """0-D JAX array should produce a single-element list."""
        result = _to_float64_list(jnp.array(0.5))
        assert result == [0.5]

    def test_1d_array(self):
        """1-D array should produce matching list."""
        result = _to_float64_list(jnp.array([0.3, 0.7]))
        assert len(result) == 2
        assert abs(result[0] - 0.3) < 1e-6
        assert abs(result[1] - 0.7) < 1e-6
