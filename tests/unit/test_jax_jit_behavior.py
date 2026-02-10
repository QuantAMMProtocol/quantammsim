"""
Unit tests for JAX JIT behavior with dictionary mutations.

Tests to verify that JAX JIT correctly handles pytree mutations
and maintains immutability semantics.
"""
import pytest
import jax.numpy as jnp
from jax import jit


class TestJITImmutability:
    """Test JAX JIT immutability behavior with dict mutations."""

    def test_basic_modification_preserves_original(self):
        """Test that JIT function doesn't modify original params."""
        @jit
        def modify_params(params):
            new_array = params['key'].at[0].set(5.0)
            params['key'] = jnp.zeros_like(params['key'])
            return params

        original = {'key': jnp.array([1.0, 2.0, 3.0])}
        original_copy = jnp.array(original['key'])

        modified = modify_params(original)

        # Original should be unchanged
        assert jnp.allclose(original['key'], original_copy), \
            "Original params should not be modified by JIT function"

        # Modified should have changes
        assert jnp.allclose(modified['key'], jnp.zeros(3)), \
            "Returned params should have modifications"

    def test_multiple_modifications(self):
        """Test multiple sequential modifications within JIT."""
        @jit
        def modify_multiple(params):
            new_array = params['key'].at[0].set(5.0)
            params['key'] = new_array
            newer_array = params['key'].at[1].set(10.0)
            params['key'] = newer_array
            return params

        original = {'key': jnp.array([1.0, 2.0, 3.0])}
        original_copy = jnp.array(original['key'])

        modified = modify_multiple(original)

        # Original should be unchanged
        assert jnp.allclose(original['key'], original_copy), \
            "Original params should not be modified"

        # Modified should have both changes
        expected = jnp.array([5.0, 10.0, 3.0])
        assert jnp.allclose(modified['key'], expected), \
            f"Expected {expected}, got {modified['key']}"

    def test_nested_structure_modification(self):
        """Test modifications to nested dict structures."""
        @jit
        def modify_nested(params):
            new_array = params['outer']['key'].at[0].set(5.0)
            params['outer']['key'] = new_array
            return params

        original = {
            'outer': {
                'key': jnp.array([1.0, 2.0, 3.0])
            }
        }
        original_copy = jnp.array(original['outer']['key'])

        modified = modify_nested(original)

        # Original should be unchanged
        assert jnp.allclose(original['outer']['key'], original_copy), \
            "Original nested params should not be modified"

        # Modified should have changes
        expected = jnp.array([5.0, 2.0, 3.0])
        assert jnp.allclose(modified['outer']['key'], expected), \
            f"Expected {expected}, got {modified['outer']['key']}"

    def test_array_identity_changes(self):
        """Test that returned arrays have different identity."""
        @jit
        def modify_params(params):
            new_array = params['key'].at[0].set(5.0)
            params['key'] = new_array
            return params

        original = {'key': jnp.array([1.0, 2.0, 3.0])}
        modified = modify_params(original)

        # The arrays should have different values after modification
        # This tests that JAX's functional semantics preserve original
        assert not jnp.array_equal(original['key'], modified['key']), \
            "Original and modified arrays should have different values"

    def test_in_place_update_syntax(self):
        """Test JAX's .at[].set() syntax for in-place-style updates."""
        @jit
        def update_element(arr, idx, val):
            return arr.at[idx].set(val)

        original = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        updated = update_element(original, 2, 99.0)

        # Original unchanged
        assert jnp.allclose(original, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        # Updated has change
        assert jnp.allclose(updated, jnp.array([1.0, 2.0, 99.0, 4.0, 5.0]))

    def test_multiple_keys_modification(self):
        """Test modifying multiple keys in a dict."""
        @jit
        def modify_both_keys(params):
            params['a'] = params['a'].at[0].set(100.0)
            params['b'] = params['b'].at[1].set(200.0)
            return params

        original = {
            'a': jnp.array([1.0, 2.0]),
            'b': jnp.array([3.0, 4.0])
        }

        modified = modify_both_keys(original)

        # Check modifications
        assert jnp.allclose(modified['a'], jnp.array([100.0, 2.0]))
        assert jnp.allclose(modified['b'], jnp.array([3.0, 200.0]))
