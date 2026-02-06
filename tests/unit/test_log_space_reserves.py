"""Tests for Feature 9: Log-space reserve products."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


def _reserve_ratio_direct(prev_weights, prev_prices, weights, prices):
    """Reference implementation using direct prod(X**W)."""
    price_change_ratio = prices / prev_prices
    price_prod = jnp.prod(price_change_ratio**prev_weights)
    reserves_from_price = price_prod / price_change_ratio

    weight_change_ratio = weights / prev_weights
    weight_prod = jnp.prod(weight_change_ratio**weights)
    reserves_from_weight = weight_change_ratio / weight_prod

    return reserves_from_price * reserves_from_weight


def _reserve_ratio_log_space(prev_weights, prev_prices, weights, prices):
    """Log-space implementation: exp(sum(W * log(X)))."""
    price_change_ratio = prices / prev_prices
    price_prod = jnp.exp(jnp.sum(prev_weights * jnp.log(price_change_ratio)))
    reserves_from_price = price_prod / price_change_ratio

    weight_change_ratio = weights / prev_weights
    weight_prod = jnp.exp(jnp.sum(weights * jnp.log(weight_change_ratio)))
    reserves_from_weight = weight_change_ratio / weight_prod

    return reserves_from_price * reserves_from_weight


def test_log_space_matches_direct_2_assets():
    """Log-space and direct prod match within 1e-10 for 2 assets."""
    prev_weights = jnp.array([0.5, 0.5])
    prev_prices = jnp.array([100.0, 2000.0])
    weights = jnp.array([0.6, 0.4])
    prices = jnp.array([105.0, 1950.0])

    direct = _reserve_ratio_direct(prev_weights, prev_prices, weights, prices)
    log_space = _reserve_ratio_log_space(prev_weights, prev_prices, weights, prices)

    np.testing.assert_allclose(direct, log_space, atol=1e-10)


def test_log_space_matches_direct_4_assets():
    """Log-space and direct prod match within 1e-10 for 4 assets."""
    prev_weights = jnp.array([0.25, 0.25, 0.25, 0.25])
    prev_prices = jnp.array([100.0, 2000.0, 50.0, 1.0])
    weights = jnp.array([0.3, 0.2, 0.3, 0.2])
    prices = jnp.array([110.0, 1900.0, 55.0, 0.95])

    direct = _reserve_ratio_direct(prev_weights, prev_prices, weights, prices)
    log_space = _reserve_ratio_log_space(prev_weights, prev_prices, weights, prices)

    np.testing.assert_allclose(direct, log_space, atol=1e-10)


def test_log_space_stable_10_assets_extreme():
    """10 assets with 100x price ratios produce finite results."""
    n = 10
    prev_weights = jnp.ones(n) / n
    weights = jnp.ones(n) / n
    # Extreme price ratios: 0.01x to 100x
    prev_prices = jnp.ones(n)
    prices = jnp.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 0.05])

    result = _reserve_ratio_log_space(prev_weights, prev_prices, weights, prices)
    assert jnp.all(jnp.isfinite(result)), f"Non-finite values: {result}"


def test_actual_reserve_ratio_unchanged():
    """_jax_calc_quantAMM_reserve_ratio produces same output after log-space change."""
    from quantammsim.pools.G3M.quantamm.quantamm_reserves import (
        _jax_calc_quantAMM_reserve_ratio,
    )

    prev_weights = jnp.array([0.5, 0.5])
    prev_prices = jnp.array([100.0, 2000.0])
    weights = jnp.array([0.6, 0.4])
    prices = jnp.array([105.0, 1950.0])

    result = _jax_calc_quantAMM_reserve_ratio(prev_weights, prev_prices, weights, prices)
    expected = _reserve_ratio_log_space(prev_weights, prev_prices, weights, prices)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_gradient_through_log_space_finite():
    """Gradients through log-space reserve calculation are finite."""
    from quantammsim.pools.G3M.quantamm.quantamm_reserves import (
        _jax_calc_quantAMM_reserve_ratio,
    )

    prev_weights = jnp.array([0.5, 0.5])
    prev_prices = jnp.array([100.0, 2000.0])
    prices = jnp.array([105.0, 1950.0])

    def loss(weights):
        ratios = _jax_calc_quantAMM_reserve_ratio(prev_weights, prev_prices, weights, prices)
        return jnp.sum(ratios)

    grads = jax.grad(loss)(jnp.array([0.6, 0.4]))
    assert jnp.all(jnp.isfinite(grads)), f"Non-finite gradients: {grads}"
    assert not jnp.allclose(grads, 0.0), "Gradients are all zero"
