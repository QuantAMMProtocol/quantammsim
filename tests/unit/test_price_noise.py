"""Tests for Feature 3: Price noise augmentation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def test_sigma_zero_identical_results():
    """Forward pass with price_noise_sigma=0.0 is deterministic."""
    from quantammsim.core_simulator.forward_pass import _apply_price_noise

    prices = jnp.ones((100, 2)) * jnp.array([50000.0, 3000.0])
    noised = _apply_price_noise(prices, 0.0, 42)
    assert jnp.allclose(prices, noised), "Sigma=0 should not change prices"


def test_noise_changes_prices():
    """sigma=0.005 with different keys → different prices."""
    from quantammsim.core_simulator.forward_pass import _apply_price_noise

    prices = jnp.ones((100, 2)) * jnp.array([50000.0, 3000.0])
    noised1 = _apply_price_noise(prices, 0.005, 1)
    noised2 = _apply_price_noise(prices, 0.005, 2)
    assert not jnp.allclose(noised1, noised2), "Different keys should give different noise"


def test_noise_preserves_gradients():
    """value_and_grad with noisy prices → finite non-zero grads."""
    from quantammsim.core_simulator.forward_pass import _apply_price_noise

    prices = jnp.ones((100, 2)) * jnp.array([50000.0, 3000.0])

    def loss_fn(prices_):
        noised = _apply_price_noise(prices_, 0.005, 42)
        return jnp.sum(noised)

    val, grads = jax.value_and_grad(loss_fn)(prices)
    assert jnp.isfinite(val), f"Value not finite: {val}"
    assert jnp.all(jnp.isfinite(grads)), "Some gradients not finite"
    assert jnp.any(grads != 0.0), "All gradients zero"


def test_noised_prices_positive():
    """Log-normal noise guarantees positive prices for any sigma."""
    from quantammsim.core_simulator.forward_pass import _apply_price_noise

    prices = jnp.ones((1000, 2)) * jnp.array([50000.0, 3000.0])
    for sigma in [0.001, 0.005, 0.01, 0.1, 0.5]:
        noised = _apply_price_noise(prices, sigma, 42)
        assert jnp.all(noised > 0), f"Negative prices with sigma={sigma}"


def test_noise_is_log_normal():
    """Verify noise is multiplicative log-normal: log(noised/prices) ~ N(0, sigma)."""
    from quantammsim.core_simulator.forward_pass import _apply_price_noise

    sigma = 0.01
    prices = jnp.ones((10000, 1)) * 100.0
    noised = _apply_price_noise(prices, sigma, 42)
    log_ratio = jnp.log(noised / prices)
    # Should have mean ≈ 0 and std ≈ sigma
    assert jnp.abs(log_ratio.mean()) < 0.01, f"Mean log ratio {log_ratio.mean()} not near 0"
    assert jnp.abs(log_ratio.std() - sigma) < 0.005, f"Std log ratio {log_ratio.std()} not near {sigma}"


def test_default_sigma_zero():
    """run_fingerprint_defaults has 'price_noise_sigma': 0.0."""
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults

    sigma = run_fingerprint_defaults["price_noise_sigma"]
    assert sigma == 0.0, f"Expected 0.0, got {sigma}"
