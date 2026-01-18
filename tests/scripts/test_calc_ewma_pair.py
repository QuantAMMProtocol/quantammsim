import numpy as np
import jax.numpy as jnp
from jax import random
from contextlib import contextmanager

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_ewma_pair,
    DEFAULT_BACKEND
)
import debug

@contextmanager
def override_backend(backend):
    """Temporarily override the DEFAULT_BACKEND"""
    from quantammsim.pools.G3M.quantamm.update_rule_estimators import estimators
    original = estimators.DEFAULT_BACKEND
    estimators.DEFAULT_BACKEND = backend
    try:
        yield
    finally:
        estimators.DEFAULT_BACKEND = original

def generate_test_data(key, n_timesteps=100, n_assets=3):
    """Generate test price data with known properties"""
    key1, key2 = random.split(key)
    returns = random.normal(key1, (n_timesteps, n_assets)) * 0.01
    prices = jnp.exp(jnp.cumsum(returns, axis=0))
    prices = prices - jnp.min(prices) + 1.0
    return prices

def run_ewma_pair_test(prices, memory_days_1=5, memory_days_2=10, chunk_period=1440, max_memory_days=30):
    """Run EWMA pair calculation with both CPU and GPU implementations"""
    
    # Broadcast memory days to match number of assets
    memory_days_1 = jnp.full(prices.shape[1], memory_days_1)
    memory_days_2 = jnp.full(prices.shape[1], memory_days_2)
    # Run CPU version
    with override_backend("cpu"):
        cpu_ewma1, cpu_ewma2 = calc_ewma_pair(
            memory_days_1,
            memory_days_2,
            prices,
            chunk_period,
            max_memory_days,
            cap_lamb=True
        )
    
    # Run GPU version
    with override_backend("gpu"):
        gpu_ewma1, gpu_ewma2 = calc_ewma_pair(
            memory_days_1,
            memory_days_2,
            prices,
            chunk_period,
            max_memory_days,
            cap_lamb=True
        )
    
    return (cpu_ewma1, cpu_ewma2), (gpu_ewma1, gpu_ewma2)

def test_ewma_pair_implementations():
    """Test EWMA pair calculations across different scenarios"""
    
    key = random.PRNGKey(0)
    
    test_cases = [
        # (n_timesteps, n_assets, memory_days_1, memory_days_2, chunk_period, max_memory_days)
        (100, 3, 5, 10, 1440, 30),    # Standard case
        (1000, 5, 10, 20, 1440, 60),  # Longer series
        (50, 2, 3, 6, 1440, 15),      # Short series
        (200, 4, 15, 15, 1440, 45),   # Equal memory lengths
    ]

    print("Testing EWMA pair calculations...")
    print("-" * 50)

    for i, (n_timesteps, n_assets, mem1, mem2, chunk_period, max_mem) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Timesteps: {n_timesteps}, Assets: {n_assets}")
        print(f"Memory days: {mem1}/{mem2}, Max memory: {max_mem}")

        # Generate test data
        key, subkey = random.split(key)
        prices = generate_test_data(subkey, n_timesteps, n_assets)

        # Run test
        cpu_results, gpu_results = run_ewma_pair_test(
            prices, mem1, mem2, chunk_period, max_mem
        )

        # Check shapes
        print(f"Input shape: {prices.shape}")
        print(f"CPU EWMA shapes: {cpu_results[0].shape}, {cpu_results[1].shape}")
        print(f"GPU EWMA shapes: {gpu_results[0].shape}, {gpu_results[1].shape}")

        # Compare results
        for j, (cpu_ewma, gpu_ewma) in enumerate(zip(cpu_results, gpu_results)):
            max_diff = jnp.max(jnp.abs(cpu_ewma - gpu_ewma))
            is_close = jnp.allclose(cpu_ewma, gpu_ewma, rtol=1e-10, atol=1e-10)
            
            print(f"\nEWMA {j+1}:")
            print(f"Maximum absolute difference: {max_diff:.2e}")
            print(f"Arrays match within tolerance: {is_close}")

            if not is_close:
                print("WARNING: Implementations produce different results!")
                print(f"First few values CPU: {cpu_ewma[:3, 0]}")
                print(f"First few values GPU: {gpu_ewma[:3, 0]}")

if __name__ == "__main__":
    test_ewma_pair_implementations()