"""Differentiability tests for reCLAMM STE-gated training path behavior."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from quantammsim.pools.creator import create_pool
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    initialise_reclamm_reserves,
    _jax_calc_reclamm_reserves_zero_fees,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs,
)
from quantammsim.runners.jax_runner_utils import Hashabledict


ALL_SIG_VARIATIONS_2 = jnp.array([[1, -1], [-1, 1]])
DEFAULT_POOL_VALUE = 1_000_000.0
DEFAULT_INITIAL_PRICES = jnp.array([2500.0, 1.0], dtype=jnp.float64)
DEFAULT_PRICE_RATIO = 4.0
DEFAULT_SHIFT_BASE = 1.0 - 1.0 / 124000.0
DEFAULT_SECONDS_PER_STEP = 60.0


def _init_pool_state():
    return initialise_reclamm_reserves(
        DEFAULT_POOL_VALUE,
        DEFAULT_INITIAL_PRICES,
        DEFAULT_PRICE_RATIO,
    )


def _trending_prices(n_steps):
    return jnp.stack(
        [jnp.linspace(DEFAULT_INITIAL_PRICES[0], 4200.0, n_steps), jnp.ones((n_steps,))],
        axis=1,
    )


def test_ste_forward_outputs_are_temperature_invariant():
    """STE hard-forward path should be invariant to STE temperature."""
    reserves, Va, Vb = _init_pool_state()
    n_steps = 12
    prices = _trending_prices(n_steps)
    fees = jnp.full((n_steps,), 0.003, dtype=jnp.float64)
    arb_thresh = jnp.zeros((n_steps,), dtype=jnp.float64)
    arb_fees = jnp.full((n_steps,), 0.0005, dtype=jnp.float64)

    schedule = np.zeros((n_steps, 4), dtype=np.float64)
    schedule[:, 3] = np.nan
    schedule[2] = np.array([1.0, 6.0, 7.0, DEFAULT_PRICE_RATIO], dtype=np.float64)
    schedule = jnp.asarray(schedule)

    low_temp_reserves, low_temp_fee_revenue = (
        _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs(
            reserves,
            Va,
            Vb,
            prices,
            centeredness_margin=0.2,
            daily_price_shift_base=DEFAULT_SHIFT_BASE,
            seconds_per_step=DEFAULT_SECONDS_PER_STEP,
            fees=fees,
            arb_thresh=arb_thresh,
            arb_fees=arb_fees,
            price_ratio_updates=schedule,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            ste_temperature=3.0,
        )
    )
    high_temp_reserves, high_temp_fee_revenue = (
        _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs(
            reserves,
            Va,
            Vb,
            prices,
            centeredness_margin=0.2,
            daily_price_shift_base=DEFAULT_SHIFT_BASE,
            seconds_per_step=DEFAULT_SECONDS_PER_STEP,
            fees=fees,
            arb_thresh=arb_thresh,
            arb_fees=arb_fees,
            price_ratio_updates=schedule,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
            ste_temperature=50.0,
        )
    )
    npt.assert_allclose(high_temp_reserves, low_temp_reserves, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(
        high_temp_fee_revenue, low_temp_fee_revenue, rtol=1e-10, atol=1e-10
    )


def test_margin_gradient_is_finite_and_nonzero_in_zero_fee_kernel():
    """Centeredness-margin gradient should flow through always-on STE gates."""
    reserves, Va, Vb = _init_pool_state()
    n_steps = 6
    prices = jnp.tile(DEFAULT_INITIAL_PRICES, (n_steps, 1))
    margin = jnp.float64(1.0)

    def _loss(centeredness_margin):
        reserves_out = _jax_calc_reclamm_reserves_zero_fees(
            reserves,
            Va,
            Vb,
            prices,
            centeredness_margin=centeredness_margin,
            daily_price_shift_base=DEFAULT_SHIFT_BASE,
            seconds_per_step=DEFAULT_SECONDS_PER_STEP,
            ste_temperature=25.0,
        )
        return jnp.sum(reserves_out[-1])

    grad_val = jax.grad(_loss)(margin)

    assert jnp.isfinite(grad_val)
    assert jnp.abs(grad_val) > 1e-9


def test_pool_zero_fee_path_uses_configured_ste_temperature():
    """Pool-level path should pass STE temperature through to kernel gradients."""
    pool = create_pool("reclamm")
    n_steps = 6
    prices = jnp.tile(DEFAULT_INITIAL_PRICES, (n_steps, 1))
    start_index = jnp.array([0, 0], dtype=jnp.int32)

    run_fp_low_temp = Hashabledict(
        {
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": DEFAULT_POOL_VALUE,
            "arb_frequency": 1,
            "do_arb": True,
            "ste_temperature": 2.0,
        }
    )
    run_fp_high_temp = Hashabledict(
        {
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": DEFAULT_POOL_VALUE,
            "arb_frequency": 1,
            "do_arb": True,
            "ste_temperature": 50.0,
        }
    )

    def _loss(centeredness_margin, run_fingerprint):
        params = {
            "price_ratio": jnp.float64(DEFAULT_PRICE_RATIO),
            "centeredness_margin": centeredness_margin,
            "daily_price_shift_base": jnp.float64(DEFAULT_SHIFT_BASE),
        }
        reserves_out = pool.calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index
        )
        return jnp.sum(reserves_out[-1])

    margin = jnp.float64(1.0)
    low_temp_grad = jax.grad(lambda m: _loss(m, run_fp_low_temp))(margin)
    high_temp_grad = jax.grad(lambda m: _loss(m, run_fp_high_temp))(margin)

    assert jnp.isfinite(low_temp_grad)
    assert jnp.isfinite(high_temp_grad)
    assert jnp.abs(low_temp_grad) > 1e-9
    assert jnp.abs(high_temp_grad) > 1e-9
    assert jnp.abs(high_temp_grad) > jnp.abs(low_temp_grad) * 1.5
