import numpy as np
import jax.numpy as jnp
from jax import random
import sys
from contextlib import contextmanager
import os
import debug
# Add path to quantammsim if needed
# sys.path.append("path/to/quantammsim")

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_return_variances,
    OG_calc_return_variances,
    calc_return_precision_based_weights,
    DEFAULT_BACKEND
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    make_ewma_kernel,
    make_a_kernel,
    make_cov_kernel,
    _jax_ewma_at_infinity_via_conv_padded,
    _jax_ewma_at_infinity_via_scan,
    _jax_gradients_at_infinity_via_conv_padded,
    _jax_gradients_at_infinity_via_conv_padded_with_alt_ewma,
    _jax_variance_at_infinity_via_conv,
    _jax_variance_at_infinity_via_scan,
    squareplus,
    _jax_gradients_at_infinity_via_scan,
    _jax_gradients_at_infinity_via_scan_with_alt_ewma,
)
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    jax_memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
)

def alt_OG_calc_return_variances(
    update_rule_parameter_dict,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    cap_lamb,
):
    lamb = calc_lamb(update_rule_parameter_dict)
    if cap_lamb:
        max_lamb = memory_days_to_lamb(max_memory_days, chunk_period)
        lamb = jnp.clip(lamb, a_min=0.0, a_max=max_lamb)
    safety_margin_max_memory_days = max_memory_days * 5.0
    cov_kernel = make_cov_kernel(lamb, safety_margin_max_memory_days, chunk_period)

    returns = jnp.diff(chunkwise_price_values, axis=0) / chunkwise_price_values[:-1]

    padded_returns = jnp.vstack(
        [
            jnp.ones(
                (
                    int(safety_margin_max_memory_days * 1440 / chunk_period),
                    returns.shape[1],
                )
            )
            * returns[0],
            returns,
        ]
    )

    # ewma_returns = _ewma_at_infinity(returns, lamb)

    ewma_kernel = make_ewma_kernel(lamb, safety_margin_max_memory_days, chunk_period)

    # ewma_padded_ = jnp.convolve(
    #     padded_chunkwise_price_values[:, 0], ewma_kernel[:, 0], mode="full"
    # )[(return_slice_index) : len(padded_chunkwise_price_values)]
    ewma_returns_padded = _jax_ewma_at_infinity_via_conv_padded(
        padded_returns, ewma_kernel
    )
    variances = _jax_variance_at_infinity_via_conv(
        padded_returns, ewma_returns_padded[1:], cov_kernel, lamb
    )

    return variances


def OG_calc_return_precision_based_weights(
    update_rule_parameter_dict,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    cap_lamb,
):
    variances = calc_return_variances(
        update_rule_parameter_dict,
        chunkwise_price_values,
        chunk_period,
        max_memory_days,
        cap_lamb,
    )
    # ewma_padded = calc_ewma_padded(update_rule_parameter_dict, chunkwise_price_values, chunk_period, max_memory_days, cap_lamb)
    # ewma = ewma_padded[-(len(chunkwise_price_values) - 1):]
    # variances = _jax_variance_at_infinity_via_conv(chunkwise_price_values, ewma, cov_kernel, lamb)
    precision_based_weights = 1.0 / variances
    precision_based_weights = precision_based_weights / precision_based_weights.sum(
        axis=-1, keepdims=True
    )
    return precision_based_weights


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
    
    # Generate random walk with known volatility
    returns = random.normal(key1, (n_timesteps, n_assets)) * 0.01
    prices = jnp.exp(jnp.cumsum(returns, axis=0))
    # Ensure positive prices
    prices = prices - jnp.min(prices) + 1.0
    return prices

def run_variance_test(prices, memory_days=10, chunk_period=1440, max_memory_days=30):
    """Run variance calculation with both CPU and GPU implementations"""
    
    # Create parameter dict
    params = {
        "logit_lamb": jnp.array([-2.0]*prices.shape[1]),  # Example lambda value
    }
    
    # Run CPU version
    with override_backend("cpu"):
        cpu_variances = calc_return_variances(
            params, prices, chunk_period, max_memory_days, cap_lamb=True
        )
    
    # Run GPU version
    with override_backend("gpu"):
        gpu_variances = calc_return_variances(
            params, prices, chunk_period, max_memory_days, cap_lamb=True
        )
    
    alt_gpu_variances = calc_return_precision_based_weights(
            params, prices, chunk_period, max_memory_days, cap_lamb=True
        )
    
    return cpu_variances, gpu_variances, alt_gpu_variances

def test_variance_implementations():
    """Test variance calculations across different scenarios"""

    # Initialize random key
    key = random.PRNGKey(0)

    test_cases = [
        # (n_timesteps, n_assets, memory_days, chunk_period, max_memory_days)
        (100, 3, 10, 1440, 30),  # Standard case
        (1000, 5, 20, 1440, 60),  # Longer series
        (50, 2, 5, 1440, 15),     # Short series
    ]

    print("Testing variance calculations...")
    print("-" * 50)

    for i, (n_timesteps, n_assets, memory_days, chunk_period, max_memory_days) in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Timesteps: {n_timesteps}, Assets: {n_assets}, Memory days: {memory_days}, Chunk period: {chunk_period}, Max memory days: {max_memory_days}")

        # Generate test data
        key, subkey = random.split(key)
        prices = generate_test_data(subkey, n_timesteps, n_assets)

        # Run test
        cpu_vars, gpu_vars, alt_gpu_vars = run_variance_test(
            prices, memory_days, chunk_period, max_memory_days
        )
        params = {
            "logit_lamb": jnp.array([-2.0] * prices.shape[1]),  # Example lambda value
        }
        variances = calc_return_variances(
            params,
            prices,
            chunk_period,
            max_memory_days,
            cap_lamb=True,
        )
        OG_variances = OG_calc_return_variances(
            params,
            prices,
            chunk_period,
            max_memory_days,
            cap_lamb=True,
        )
        print(f"CPU variances: {cpu_vars.shape}")
        print(f"GPU variances: {gpu_vars.shape}")
        print(f"Alt GPU variances: {alt_gpu_vars.shape}")
        raise Exception("Stop")
        # Compare results
        max_diff = jnp.max(jnp.abs(cpu_vars - gpu_vars))
        is_close = jnp.allclose(cpu_vars, gpu_vars, rtol=1e-10, atol=1e-10)

        print(f"Maximum absolute difference: {max_diff:.2e}")
        print(f"Arrays match within tolerance: {is_close}")

        if not is_close:
            print("WARNING: Implementations produce different results!")
            print(f"CPU shape: {cpu_vars.shape}, GPU shape: {gpu_vars.shape}")
            print(f"First few values CPU: {cpu_vars[:3, 0]}")
            print(f"First few values GPU: {gpu_vars[:3, 0]}")

if __name__ == "__main__":
    test_variance_implementations() 
