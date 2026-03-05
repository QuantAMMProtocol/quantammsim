"""Tests for manual reCLAMM price-ratio schedule updates."""

import numpy as np
import numpy.testing as npt
import jax.numpy as jnp
import pytest

from quantammsim.pools.reCLAMM.reclamm_reserves import (
    compute_price_ratio,
    initialise_reclamm_reserves,
    _jax_calc_reclamm_reserves_with_dynamic_inputs,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs,
)
from tests.pools.reCLAMM.helpers import (
    _jax_calc_reclamm_reserves_with_dynamic_inputs_full_state,
)


DEFAULT_INITIAL_POOL_VALUE = 1_000_000.0
DEFAULT_INITIAL_PRICES = jnp.array([2500.0, 1.0], dtype=jnp.float64)
DEFAULT_PRICE_RATIO = 4.0
DEFAULT_DAILY_PRICE_SHIFT_BASE = 1.0 - 1.0 / 124000.0
DEFAULT_SECONDS_PER_STEP = 60.0
ALL_SIG_VARIATIONS_2 = tuple(map(tuple, [[1, -1], [-1, 1]]))


def _init_pool(price_ratio=DEFAULT_PRICE_RATIO):
    reserves, Va, Vb = initialise_reclamm_reserves(
        DEFAULT_INITIAL_POOL_VALUE,
        DEFAULT_INITIAL_PRICES,
        price_ratio,
    )
    return reserves, Va, Vb


def _flat_prices(n_steps):
    return jnp.stack(
        [jnp.full((n_steps,), DEFAULT_INITIAL_PRICES[0]), jnp.ones((n_steps,))],
        axis=1,
    )


def _empty_schedule(n_steps):
    schedule = np.zeros((n_steps, 4), dtype=np.float64)
    schedule[:, 3] = np.nan
    return jnp.asarray(schedule)


def _single_event_schedule(
    n_steps,
    start_step,
    end_step,
    target_price_ratio,
    start_price_ratio_override=np.nan,
):
    schedule = np.zeros((n_steps, 4), dtype=np.float64)
    schedule[:, 3] = np.nan
    schedule[start_step, 0] = 1.0
    schedule[start_step, 1] = target_price_ratio
    schedule[start_step, 2] = float(end_step)
    schedule[start_step, 3] = start_price_ratio_override
    return jnp.asarray(schedule)


class TestReclammPriceRatioUpdates:
    def test_schedule_off_matches_baseline_dynamic_kernel(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 8
        prices = _flat_prices(n_steps)
        fees = jnp.zeros((n_steps,), dtype=jnp.float64)
        arb_thresh = jnp.zeros((n_steps,), dtype=jnp.float64)
        arb_fees = jnp.zeros((n_steps,), dtype=jnp.float64)

        baseline = _jax_calc_reclamm_reserves_with_dynamic_inputs(
            reserves,
            Va,
            Vb,
            prices,
            centeredness_margin=0.2,
            daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
            seconds_per_step=DEFAULT_SECONDS_PER_STEP,
            fees=fees,
            arb_thresh=arb_thresh,
            arb_fees=arb_fees,
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )
        with_schedule = _jax_calc_reclamm_reserves_with_dynamic_inputs(
            reserves,
            Va,
            Vb,
            prices,
            centeredness_margin=0.2,
            daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
            seconds_per_step=DEFAULT_SECONDS_PER_STEP,
            fees=fees,
            arb_thresh=arb_thresh,
            arb_fees=arb_fees,
            price_ratio_updates=_empty_schedule(n_steps),
            all_sig_variations=ALL_SIG_VARIATIONS_2,
        )
        npt.assert_allclose(with_schedule, baseline, rtol=1e-10, atol=1e-10)

    def test_single_schedule_reaches_target_ratio_at_end_step(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 8
        prices = _flat_prices(n_steps)
        fees = jnp.zeros((n_steps,), dtype=jnp.float64)
        arb_thresh = jnp.zeros((n_steps,), dtype=jnp.float64)
        arb_fees = jnp.zeros((n_steps,), dtype=jnp.float64)

        end_step = 4
        schedule = _single_event_schedule(
            n_steps,
            start_step=1,
            end_step=end_step,
            target_price_ratio=9.0,
            start_price_ratio_override=DEFAULT_PRICE_RATIO,
        )
        reserves_out, Va_history, Vb_history = (
            _jax_calc_reclamm_reserves_with_dynamic_inputs_full_state(
                reserves,
                Va,
                Vb,
                prices,
                centeredness_margin=0.0,
                daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
                seconds_per_step=DEFAULT_SECONDS_PER_STEP,
                fees=fees,
                arb_thresh=arb_thresh,
                arb_fees=arb_fees,
                price_ratio_updates=schedule,
                all_sig_variations=ALL_SIG_VARIATIONS_2,
            )
        )

        ratio_at_end = float(
            compute_price_ratio(
                reserves_out[end_step, 0],
                reserves_out[end_step, 1],
                Va_history[end_step],
                Vb_history[end_step],
            )
        )
        assert ratio_at_end == pytest.approx(9.0, rel=1e-5, abs=1e-5)

    def test_replacement_event_supersedes_active_event(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 9
        prices = _flat_prices(n_steps)
        fees = jnp.zeros((n_steps,), dtype=jnp.float64)
        arb_thresh = jnp.zeros((n_steps,), dtype=jnp.float64)
        arb_fees = jnp.zeros((n_steps,), dtype=jnp.float64)

        schedule = np.zeros((n_steps, 4), dtype=np.float64)
        schedule[:, 3] = np.nan
        # Event 1: interpolate toward 8.0 until step 6.
        schedule[1] = np.array([1.0, 8.0, 6.0, DEFAULT_PRICE_RATIO], dtype=np.float64)
        # Event 2 replaces at step 3 and targets 2.0 by step 4.
        schedule[3] = np.array([1.0, 2.0, 4.0, np.nan], dtype=np.float64)

        reserves_out, Va_history, Vb_history = (
            _jax_calc_reclamm_reserves_with_dynamic_inputs_full_state(
                reserves,
                Va,
                Vb,
                prices,
                centeredness_margin=0.0,
                daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
                seconds_per_step=DEFAULT_SECONDS_PER_STEP,
                fees=fees,
                arb_thresh=arb_thresh,
                arb_fees=arb_fees,
                price_ratio_updates=jnp.asarray(schedule),
                all_sig_variations=ALL_SIG_VARIATIONS_2,
            )
        )

        ratio_after_replacement = float(
            compute_price_ratio(
                reserves_out[4, 0],
                reserves_out[4, 1],
                Va_history[4],
                Vb_history[4],
            )
        )
        assert ratio_after_replacement == pytest.approx(2.0, rel=1e-4, abs=1e-4)

    def test_dynamic_fee_revenue_path_with_schedule(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 10
        prices = _flat_prices(n_steps)
        fees = jnp.full((n_steps,), 0.003, dtype=jnp.float64)
        arb_thresh = jnp.zeros((n_steps,), dtype=jnp.float64)
        arb_fees = jnp.zeros((n_steps,), dtype=jnp.float64)

        schedule = _single_event_schedule(
            n_steps,
            start_step=2,
            end_step=6,
            target_price_ratio=5.5,
            start_price_ratio_override=DEFAULT_PRICE_RATIO,
        )
        reserves_out, fee_revenue = (
            _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs(
                reserves,
                Va,
                Vb,
                prices,
                centeredness_margin=0.2,
                daily_price_shift_base=DEFAULT_DAILY_PRICE_SHIFT_BASE,
                seconds_per_step=DEFAULT_SECONDS_PER_STEP,
                fees=fees,
                arb_thresh=arb_thresh,
                arb_fees=arb_fees,
                price_ratio_updates=schedule,
                all_sig_variations=ALL_SIG_VARIATIONS_2,
            )
        )
        assert reserves_out.shape == (n_steps, 2)
        assert fee_revenue.shape == (n_steps,)
        assert jnp.all(fee_revenue >= 0.0)
