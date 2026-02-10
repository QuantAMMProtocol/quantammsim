import jax.numpy as jnp
from jax import jit
import numpy as np

def print_separator():
    print("\n" + "="*50 + "\n")

@jit
def test_modify(params):
    # Creates new array and new pytree
    new_array = params['key'].at[0].set(5.0)
    params['key'] = jnp.zeros_like(params['key'])
    return params

@jit
def test_modify_multiple(params):
    # Multiple modifications
    new_array = params['key'].at[0].set(5.0)
    params['key'] = new_array
    newer_array = params['key'].at[1].set(10.0)
    params['key'] = newer_array
    return params

def run_tests():
    # Test 1: Basic modification
    print("Test 1: Basic Modification")
    params = {'key': jnp.array([1.0, 2.0, 3.0])}
    
    print("Original params:", params)
    modified_params = test_modify(params)
    print("Returned params:", modified_params)
    print("Original params after call:", params)
    print("Original array id:", id(params['key']))
    print("Returned array id:", id(modified_params['key']))
    
    print_separator()
    
    # Test 2: Multiple modifications
    print("Test 2: Multiple Modifications")
    params = {'key': jnp.array([1.0, 2.0, 3.0])}
    
    print("Original params:", params)
    modified_params = test_modify_multiple(params)
    print("Returned params:", modified_params)
    print("Original params after call:", params)
    
    print_separator()
    
    # Test 3: Nested structure
    print("Test 3: Nested Structure")
    params = {
        'outer': {
            'key': jnp.array([1.0, 2.0, 3.0])
        }
    }
    
    @jit
    def test_modify_nested(p):
        new_array = p['outer']['key'].at[0].set(5.0)
        p['outer']['key'] = new_array
        return p
    
    print("Original params:", params)
    modified_params = test_modify_nested(params)
    print("Returned params:", modified_params)
    print("Original params after call:", params)

if __name__ == "__main__":
    run_tests() 