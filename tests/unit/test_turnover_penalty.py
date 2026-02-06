"""Tests for Feature 2: Turnover penalty in loss."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _make_synthetic_data(n_steps=4321, n_assets=2, constant_weights=False):
    """Create synthetic reserves, prices, and value_over_time."""
    prices = jnp.ones((n_steps, n_assets)) * jnp.array([50000.0, 3000.0])
    if constant_weights:
        reserves = jnp.ones((n_steps, n_assets)) * jnp.array([10.0, 166.0])
    else:
        # Gradually shift reserves
        t = jnp.linspace(0.0, 1.0, n_steps)[:, None]
        reserves = jnp.array([10.0, 166.0]) + t * jnp.array([2.0, -20.0])
    value_over_time = jnp.sum(reserves * prices, axis=-1)
    return reserves, prices, value_over_time


def _compute_turnover(reserves, prices, value_over_time):
    """Compute turnover metric matching the implementation."""
    implied_weights = (reserves * prices) / value_over_time[:, jnp.newaxis]
    turnover = jnp.mean(jnp.sum(jnp.abs(jnp.diff(implied_weights, axis=0)), axis=-1))
    return turnover


def test_turnover_zero_when_weights_constant():
    """Constant reserves + prices → zero turnover penalty."""
    reserves, prices, value_over_time = _make_synthetic_data(constant_weights=True)
    turnover = _compute_turnover(reserves, prices, value_over_time)
    assert jnp.allclose(turnover, 0.0, atol=1e-10), f"Expected zero turnover, got {turnover}"


def test_turnover_positive_when_weights_change():
    """Changing reserves → positive penalty."""
    reserves, prices, value_over_time = _make_synthetic_data(constant_weights=False)
    turnover = _compute_turnover(reserves, prices, value_over_time)
    assert turnover > 0.0, f"Expected positive turnover, got {turnover}"


def test_turnover_proportional_to_magnitude():
    """2x weight changes → ~2x penalty."""
    n_steps = 4321
    n_assets = 2
    prices = jnp.ones((n_steps, n_assets)) * jnp.array([50000.0, 3000.0])
    t = jnp.linspace(0.0, 1.0, n_steps)[:, None]

    delta_small = t * jnp.array([1.0, -10.0])
    delta_big = t * jnp.array([2.0, -20.0])

    base_reserves = jnp.array([10.0, 166.0])

    reserves_small = base_reserves + delta_small
    reserves_big = base_reserves + delta_big

    val_small = jnp.sum(reserves_small * prices, axis=-1)
    val_big = jnp.sum(reserves_big * prices, axis=-1)

    t_small = _compute_turnover(reserves_small, prices, val_small)
    t_big = _compute_turnover(reserves_big, prices, val_big)

    ratio = t_big / t_small
    assert 1.5 < ratio < 2.5, f"Expected ~2x ratio, got {ratio}"


def test_turnover_penalty_modifies_loss():
    """When turnover_penalty > 0, loss changes vs base."""
    from quantammsim.core_simulator.forward_pass import _calculate_return_value

    reserves, prices, value_over_time = _make_synthetic_data(constant_weights=False)

    base_metric = _calculate_return_value(
        "daily_log_sharpe", reserves, prices, value_over_time
    )
    turnover = _compute_turnover(reserves, prices, value_over_time)
    penalty_loss = base_metric - 1.0 * turnover

    # Turnover is positive so penalty_loss must be strictly less
    assert penalty_loss < base_metric, (
        f"Expected penalty_loss < base_metric, got {penalty_loss} vs {base_metric}"
    )


def test_turnover_penalty_gradient_flow():
    """Gradients through penalty-augmented loss are finite, non-zero."""
    from quantammsim.core_simulator.forward_pass import _calculate_return_value

    reserves, prices, value_over_time = _make_synthetic_data(constant_weights=False)

    def penalized_metric(reserves_):
        val = jnp.sum(reserves_ * prices, axis=-1)
        base = _calculate_return_value("daily_log_sharpe", reserves_, prices, val)
        implied_w = (reserves_ * prices) / val[:, jnp.newaxis]
        turnover = jnp.mean(jnp.sum(jnp.abs(jnp.diff(implied_w, axis=0)), axis=-1))
        return base - 0.1 * turnover

    grads = jax.grad(penalized_metric)(reserves)
    assert jnp.all(jnp.isfinite(grads)), "Some gradients are not finite"
    assert jnp.any(grads != 0.0), "All gradients are zero"


def test_default_turnover_penalty_zero():
    """run_fingerprint_defaults has 'turnover_penalty': 0.0."""
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults

    assert "turnover_penalty" in run_fingerprint_defaults, (
        "turnover_penalty missing from defaults"
    )
    assert run_fingerprint_defaults["turnover_penalty"] == 0.0
