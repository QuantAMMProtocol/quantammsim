"""Tests for Feature 1: daily_log_sharpe as default objective."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def test_daily_log_sharpe_returns_scalar():
    """_daily_log_sharpe() on synthetic values returns shape ()."""
    from quantammsim.core_simulator.forward_pass import _daily_log_sharpe

    # 4321 minutes ~ 3 days of data
    values = jnp.linspace(1000.0, 1100.0, 4321)
    result = _daily_log_sharpe(values)
    assert result.shape == (), f"Expected scalar, got shape {result.shape}"


def test_daily_log_sharpe_produces_gradients():
    """jax.grad(_daily_log_sharpe) yields finite non-zero gradients."""
    from quantammsim.core_simulator.forward_pass import _daily_log_sharpe

    values = jnp.linspace(1000.0, 1100.0, 4321)

    # grad w.r.t. the values array directly
    grad_fn = jax.grad(lambda v: _daily_log_sharpe(v))
    g = grad_fn(values)
    assert jnp.all(jnp.isfinite(g)), f"Some gradients are not finite"
    assert jnp.any(g != 0.0), f"All gradients are zero"


def test_daily_log_sharpe_correct_on_constant_growth():
    """Linearly growing values → analytically predictable Sharpe."""
    from quantammsim.core_simulator.forward_pass import _daily_log_sharpe

    # 10 days of data at minute resolution, constant daily growth of 1%
    n_days = 10
    n_minutes = n_days * 1440
    daily_growth = 1.01
    # Build prices that grow exactly 1% per day
    t = jnp.arange(n_minutes, dtype=jnp.float64)
    values = 1000.0 * daily_growth ** (t / 1440.0)

    result = _daily_log_sharpe(values)
    # With constant daily growth, daily log return = log(1.01) every day
    # std of constant returns → 0, so sharpe → very large
    # But numerically std is small but not zero, so just check it's large positive
    assert result > 10.0, f"Expected large positive Sharpe for constant growth, got {result}"


def test_daily_log_sharpe_negative_for_declining():
    """Declining values → negative Sharpe."""
    from quantammsim.core_simulator.forward_pass import _daily_log_sharpe

    n_days = 10
    n_minutes = n_days * 1440
    daily_decline = 0.99
    t = jnp.arange(n_minutes, dtype=jnp.float64)
    values = 1000.0 * daily_decline ** (t / 1440.0)

    result = _daily_log_sharpe(values)
    assert result < 0.0, f"Expected negative Sharpe for declining values, got {result}"


def test_calculate_return_value_daily_log_sharpe():
    """_calculate_return_value('daily_log_sharpe', ...) returns a finite scalar."""
    from quantammsim.core_simulator.forward_pass import _calculate_return_value

    n_minutes = 4321
    n_assets = 2
    reserves = jnp.ones((n_minutes, n_assets)) * 500.0
    local_prices = jnp.ones((n_minutes, n_assets)) * jnp.array([50000.0, 3000.0])
    value_over_time = jnp.sum(reserves * local_prices, axis=-1)

    result = _calculate_return_value(
        "daily_log_sharpe", reserves, local_prices, value_over_time
    )
    assert jnp.isfinite(result), f"Expected finite result, got {result}"
    assert result.shape == (), f"Expected scalar, got shape {result.shape}"


def test_calculate_period_metrics_includes_daily_log_sharpe():
    """calculate_period_metrics returns dict with 'daily_log_sharpe' key."""
    from quantammsim.utils.post_train_analysis import calculate_period_metrics

    n_minutes = 4321
    n_assets = 2
    reserves = jnp.ones((n_minutes, n_assets)) * 500.0
    prices = jnp.ones((n_minutes, n_assets)) * jnp.array([50000.0, 3000.0])
    # Add slight upward trend to values
    trend = jnp.linspace(1.0, 1.05, n_minutes)
    value = jnp.sum(reserves * prices, axis=-1) * trend

    results_dict = {"reserves": reserves, "value": value}
    metrics = calculate_period_metrics(results_dict, prices=prices)
    assert "daily_log_sharpe" in metrics, f"daily_log_sharpe missing from metrics: {list(metrics.keys())}"
    assert np.isfinite(metrics["daily_log_sharpe"]), f"daily_log_sharpe is not finite: {metrics['daily_log_sharpe']}"


def test_default_return_val_is_daily_log_sharpe():
    """run_fingerprint_defaults['return_val'] == 'daily_log_sharpe'."""
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults

    assert run_fingerprint_defaults["return_val"] == "daily_log_sharpe", (
        f"Expected 'daily_log_sharpe', got '{run_fingerprint_defaults['return_val']}'"
    )


def test_default_early_stopping_metric_is_daily_log_sharpe():
    """run_fingerprint_defaults early_stopping_metric == 'daily_log_sharpe'."""
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults

    metric = run_fingerprint_defaults["optimisation_settings"]["early_stopping_metric"]
    assert metric == "daily_log_sharpe", (
        f"Expected 'daily_log_sharpe', got '{metric}'"
    )


def test_outer_to_inner_metric_has_daily_log_sharpe():
    """OUTER_TO_INNER_METRIC has daily_log_sharpe entries."""
    from quantammsim.runners.hyperparam_tuner import OUTER_TO_INNER_METRIC

    assert "mean_oos_daily_log_sharpe" in OUTER_TO_INNER_METRIC
    assert OUTER_TO_INNER_METRIC["mean_oos_daily_log_sharpe"] == "daily_log_sharpe"
    assert "worst_oos_daily_log_sharpe" in OUTER_TO_INNER_METRIC
    assert OUTER_TO_INNER_METRIC["worst_oos_daily_log_sharpe"] == "daily_log_sharpe"
