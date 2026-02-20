"""Tests for fused chunked reserve computation.

The fused path processes one coarse chunk at a time: interpolate weights →
compute reserve ratios → take product → return a single (n_assets,) chunk ratio.
This avoids materialising full minute-resolution arrays during training.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool
from quantammsim.pools.G3M.balancer.balancer import BalancerPool
from quantammsim.core_simulator.param_utils import memory_days_to_lamb
from quantammsim.runners.jax_runner_utils import NestedHashabledict
from quantammsim.core_simulator.forward_pass import forward_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_momentum_params(n_assets, memory_days=30.0, k_per_day=1.0, chunk_period=60):
    """Create momentum pool parameters."""
    initial_lamb = memory_days_to_lamb(memory_days, chunk_period)
    logit_lamb = np.log(initial_lamb / (1.0 - initial_lamb))
    return {
        "log_k": jnp.array([np.log2(k_per_day)] * n_assets),
        "logit_lamb": jnp.array([logit_lamb] * n_assets),
        "initial_weights_logits": jnp.array([0.0] * n_assets),
    }


def _make_static_dict(
    bout_length,
    n_assets=2,
    chunk_period=60,
    return_val="daily_log_sharpe",
    use_fused_reserves=False,
    fees=0.0,
    gas_cost=0.0,
    arb_fees=0.0,
):
    return NestedHashabledict({
        "bout_length": bout_length,
        "maximum_change": 0.0003,
        "n_assets": n_assets,
        "chunk_period": chunk_period,
        "weight_interpolation_period": chunk_period,
        "return_val": return_val,
        "rule": "momentum",
        "run_type": "normal",
        "max_memory_days": 365.0,
        "initial_pool_value": 1_000_000.0,
        "fees": fees,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "arb_fees": arb_fees,
        "gas_cost": gas_cost,
        "all_sig_variations": None,
        "noise_trader_ratio": 0.0,
        "weight_interpolation_method": "linear",
        "training_data_kind": "historic",
        "arb_frequency": 1,
        "do_trades": False,
        "do_arb": True,
        "minimum_weight": 0.05,
        "ste_max_change": False,
        "ste_min_max_weight": False,
        "use_fused_reserves": use_fused_reserves,
    })


def _make_test_prices(n_timesteps, n_assets=2, seed=42):
    """Synthetic minute-level prices with GBM dynamics."""
    rng = np.random.RandomState(seed)
    base_prices = np.array([100.0, 50.0])[:n_assets]
    log_rets = rng.randn(n_timesteps, n_assets) * 0.0005
    prices = base_prices * np.exp(np.cumsum(log_rets, axis=0))
    return jnp.array(prices)


# ---------------------------------------------------------------------------
# Test: Pool capability flag
# ---------------------------------------------------------------------------


def test_supports_fused_reserves_flag():
    """MomentumPool has supports_fused_reserves=True, BalancerPool has False."""
    assert MomentumPool().supports_fused_reserves is True
    assert BalancerPool().supports_fused_reserves is False


# ---------------------------------------------------------------------------
# Test: Coarse weight output matches internal state
# ---------------------------------------------------------------------------


def test_calc_coarse_weight_output_matches():
    """calc_coarse_weight_output returns (actual_starts, scaled_diffs)
    that match the internal coarse weights from the full pipeline."""
    from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
        calc_coarse_weight_output_from_weight_changes,
        calc_fine_weight_output_from_weight_changes,
    )

    n_assets = 2
    chunk_period = 60
    pool = MomentumPool()
    params = _make_momentum_params(n_assets, chunk_period=chunk_period)

    fp = NestedHashabledict({
        "chunk_period": chunk_period,
        "weight_interpolation_period": chunk_period,
        "max_memory_days": 365.0,
        "n_assets": n_assets,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "maximum_change": 0.0003,
        "weight_interpolation_method": "linear",
        "ste_max_change": False,
        "ste_min_max_weight": False,
        "minimum_weight": 0.05,
    })

    # Generate rule outputs
    n_timesteps = 1440 * 10 + chunk_period  # 10 days + burn-in
    prices = _make_test_prices(n_timesteps, n_assets)
    rule_outputs = pool.calculate_rule_outputs(params, fp, prices)
    initial_weights = pool.calculate_initial_weights(params)

    # Coarse-only path
    actual_starts_c, scaled_diffs_c = calc_coarse_weight_output_from_weight_changes(
        rule_outputs, initial_weights, fp, params
    )

    # Full fine pipeline (for reference — extract coarse weights internally)
    fine_weights = calc_fine_weight_output_from_weight_changes(
        rule_outputs, initial_weights, fp, params
    )

    # The coarse path's actual_starts should match fine weights at chunk boundaries
    # For delta-based pools, fine_weights has chunk_period initial-weight rows prepended
    # So chunk boundary k corresponds to fine_weights[chunk_period + k * chunk_period]
    for k in range(min(5, actual_starts_c.shape[0])):
        fine_idx = chunk_period + k * chunk_period
        np.testing.assert_allclose(
            actual_starts_c[k],
            fine_weights[fine_idx],
            atol=1e-10,
            err_msg=f"Mismatch at chunk {k}",
        )


# ---------------------------------------------------------------------------
# Test: Fused reserves match full resolution
# ---------------------------------------------------------------------------


def test_fused_reserves_matches_full_resolution():
    """Daily boundary values from fused path match values[::1440] from full path."""
    n_assets = 2
    chunk_period = 1440
    bout_length = 10 * 1440  # 10 days
    n_timesteps = bout_length + chunk_period  # +burn-in
    pool = MomentumPool()
    params = _make_momentum_params(n_assets, chunk_period=chunk_period)
    prices = _make_test_prices(n_timesteps, n_assets)
    start_index = jnp.array([chunk_period, 0])

    # Full-resolution path
    sd_full = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="reserves_and_values", use_fused_reserves=False,
    )
    result_full = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_full,
    )
    full_values = result_full["value"]
    daily_values_full = full_values[::1440]

    # Fused path
    sd_fused = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=True,
    )
    # The fused path is internal — we test via the forward_pass metric output
    # But let's also test the pool method directly
    fused_result = pool.calculate_fused_reserves_zero_fees(
        params, sd_fused, prices, start_index,
    )
    boundary_values = fused_result["boundary_values"]

    # boundary_values[0] should be value at t=0 (initial)
    # boundary_values[k] should match daily_values_full[k]
    np.testing.assert_allclose(
        boundary_values[:len(daily_values_full)],
        daily_values_full,
        atol=1e-6,
        err_msg="Fused boundary values don't match full-resolution daily subsampling",
    )


# ---------------------------------------------------------------------------
# Test: Gradients match between paths
# ---------------------------------------------------------------------------


def test_fused_reserves_gradient_matches():
    """Gradients of daily_log_sharpe through both paths should agree."""
    n_assets = 2
    chunk_period = 1440
    bout_length = 10 * 1440
    n_timesteps = bout_length + chunk_period
    pool = MomentumPool()
    params = _make_momentum_params(n_assets, chunk_period=chunk_period)
    prices = _make_test_prices(n_timesteps, n_assets)
    start_index = jnp.array([chunk_period, 0])

    def loss_full(p):
        sd = _make_static_dict(
            bout_length, n_assets=n_assets, chunk_period=chunk_period,
            return_val="daily_log_sharpe", use_fused_reserves=False,
        )
        return forward_pass(p, start_index, prices, pool=pool, static_dict=sd)

    def loss_fused(p):
        sd = _make_static_dict(
            bout_length, n_assets=n_assets, chunk_period=chunk_period,
            return_val="daily_log_sharpe", use_fused_reserves=True,
        )
        return forward_pass(p, start_index, prices, pool=pool, static_dict=sd)

    g_full = jax.grad(loss_full)(params)
    g_fused = jax.grad(loss_fused)(params)

    for key in g_full:
        np.testing.assert_allclose(
            g_full[key], g_fused[key], atol=1e-5, rtol=1e-4,
            err_msg=f"Gradient mismatch for {key}",
        )


# ---------------------------------------------------------------------------
# Test: Forward pass with fused flag matches without
# ---------------------------------------------------------------------------


def test_fused_forward_pass_matches_full():
    """forward_pass() with use_fused_reserves=True matches without for daily_log_sharpe."""
    n_assets = 2
    chunk_period = 1440
    bout_length = 10 * 1440
    n_timesteps = bout_length + chunk_period
    pool = MomentumPool()
    params = _make_momentum_params(n_assets, chunk_period=chunk_period)
    prices = _make_test_prices(n_timesteps, n_assets)
    start_index = jnp.array([chunk_period, 0])

    sd_full = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=False,
    )
    sd_fused = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=True,
    )

    val_full = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_full,
    )
    val_fused = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_fused,
    )

    np.testing.assert_allclose(
        val_fused, val_full, atol=1e-6,
        err_msg="Fused forward pass doesn't match full-resolution forward pass",
    )


# ---------------------------------------------------------------------------
# Test: Fallback for minute-level metrics
# ---------------------------------------------------------------------------


def test_fused_path_fallback_for_minute_metrics():
    """return_val='sharpe' (minute-level) + use_fused_reserves → falls back, same result."""
    n_assets = 2
    chunk_period = 1440
    bout_length = 10 * 1440
    n_timesteps = bout_length + chunk_period
    pool = MomentumPool()
    params = _make_momentum_params(n_assets, chunk_period=chunk_period)
    prices = _make_test_prices(n_timesteps, n_assets)
    start_index = jnp.array([chunk_period, 0])

    sd_without = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="sharpe", use_fused_reserves=False,
    )
    sd_with = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="sharpe", use_fused_reserves=True,
    )

    val_without = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_without,
    )
    val_with = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_with,
    )

    # Should be exactly equal — both take the full-resolution path
    np.testing.assert_allclose(val_with, val_without, atol=0.0)


# ---------------------------------------------------------------------------
# Test: chunk_period=60 aggregation
# ---------------------------------------------------------------------------


def test_chunk_period_60_aggregation():
    """chunk_period=60, fused daily values match full-resolution daily subsampling."""
    n_assets = 2
    chunk_period = 60
    bout_length = 5 * 1440  # 5 days
    n_timesteps = bout_length + chunk_period
    pool = MomentumPool()
    params = _make_momentum_params(n_assets, chunk_period=chunk_period)
    prices = _make_test_prices(n_timesteps, n_assets)
    start_index = jnp.array([chunk_period, 0])

    sd_full = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=False,
    )
    sd_fused = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=True,
    )

    val_full = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_full,
    )
    val_fused = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_fused,
    )

    np.testing.assert_allclose(
        val_fused, val_full, atol=1e-6,
        err_msg="chunk_period=60 fused path doesn't match full path",
    )


# ---------------------------------------------------------------------------
# Test: Fees cause fallback
# ---------------------------------------------------------------------------


def test_fused_path_with_fees_falls_back():
    """fees > 0 + use_fused_reserves → falls back to full path, same result."""
    from quantammsim.runners.jax_runner_utils import get_sig_variations

    n_assets = 2
    chunk_period = 1440
    bout_length = 10 * 1440
    n_timesteps = bout_length + chunk_period
    pool = MomentumPool()
    params = _make_momentum_params(n_assets, chunk_period=chunk_period)
    prices = _make_test_prices(n_timesteps, n_assets)
    start_index = jnp.array([chunk_period, 0])

    sig_vars = get_sig_variations(n_assets)

    sd_without = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=False,
        fees=0.003, gas_cost=0.01,
    )
    sd_without["all_sig_variations"] = sig_vars
    sd_with = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=True,
        fees=0.003, gas_cost=0.01,
    )
    sd_with["all_sig_variations"] = sig_vars

    val_without = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_without,
    )
    val_with = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_with,
    )

    # Should be exactly equal — both take the fees path
    np.testing.assert_allclose(val_with, val_without, atol=0.0)


# ---------------------------------------------------------------------------
# Test: Checkpoint produces identical results and gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("checkpoint_mode", ["vmap", "scan"])
def test_checkpoint_matches_fused(checkpoint_mode):
    """checkpoint_fused modes produce identical value and gradients to plain fused."""
    n_assets = 2
    chunk_period = 1440
    bout_length = 10 * 1440
    n_timesteps = bout_length + chunk_period
    pool = MomentumPool()
    params = _make_momentum_params(n_assets, chunk_period=chunk_period)
    prices = _make_test_prices(n_timesteps, n_assets)
    start_index = jnp.array([chunk_period, 0])

    sd_fused = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=True,
    )
    sd_ckpt = _make_static_dict(
        bout_length, n_assets=n_assets, chunk_period=chunk_period,
        return_val="daily_log_sharpe", use_fused_reserves=True,
    )
    sd_ckpt["checkpoint_fused"] = checkpoint_mode

    val_fused = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_fused,
    )
    val_ckpt = forward_pass(
        params, start_index, prices, pool=pool, static_dict=sd_ckpt,
    )

    # Values should be bitwise identical
    np.testing.assert_allclose(val_ckpt, val_fused, atol=0.0)

    # Gradients should also match
    def loss_fused(p):
        return forward_pass(p, start_index, prices, pool=pool, static_dict=sd_fused)

    def loss_ckpt(p):
        return forward_pass(p, start_index, prices, pool=pool, static_dict=sd_ckpt)

    g_fused = jax.grad(loss_fused)(params)
    g_ckpt = jax.grad(loss_ckpt)(params)

    for key in g_fused:
        np.testing.assert_allclose(
            g_ckpt[key], g_fused[key], atol=0.0,
            err_msg=f"Gradient mismatch for {key} with checkpoint_mode={checkpoint_mode}",
        )
