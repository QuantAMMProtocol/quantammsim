from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.lax import cond, scan
from jax.tree_util import Partial

from quantammsim.pools.G3M.G3M_trades import (
    _jax_calc_G3M_trade_from_exact_in_given_out,
)
from quantammsim.pools.G3M.optimal_n_pool_arb import (
    parallelised_optimal_trade_sifter,
    precalc_components_of_optimal_trade_across_signatures,
    precalc_shared_values_for_all_signatures,
)
from quantammsim.pools.hypersurge_utils import (
    _EPS,
    broadcast_scan_vector,
    fee_to_gamma,
    max_pair_deviation,
    oracle_pair_is_valid,
    oracle_vector_is_valid,
    pair_deviation,
    ramp_fee,
    safe_positive,
)
from quantammsim.pools.noise_trades import (
    calculate_reserves_after_noise_trade,
    reclamm_calibrated_noise_volume,
    reclamm_loglinear_noise_volume,
    reclamm_market_linear_noise_volume,
    reclamm_mm_observed_noise_volume,
    reclamm_tsoukalas_log_noise_volume,
    reclamm_tsoukalas_sqrt_noise_volume,
)
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    _DUST_USD,
    _ste_greater_equal,
    _ste_less_than,
    _ste_select,
    apply_target_price_ratio_to_virtual_balances,
    compute_centeredness,
    compute_invariant,
    compute_price_ratio,
    compute_virtual_balances_constant_arc_length,
    compute_virtual_balances_updating_price_range,
)


_WEIGHTS = jnp.array([0.5, 0.5])


def _broadcast_oracle_prices(oracle_prices, prices):
    oracle_prices = jnp.asarray(oracle_prices)
    if oracle_prices.ndim == 1:
        oracle_prices = oracle_prices.reshape((1, oracle_prices.shape[0]))
    if oracle_prices.shape[-1] != prices.shape[-1]:
        oracle_prices = prices
    elif oracle_prices.shape[0] == 1:
        oracle_prices = jnp.broadcast_to(oracle_prices, prices.shape)
    return oracle_prices


def _broadcast_schedule_array(price_ratio_updates, prices):
    if price_ratio_updates is None:
        updates = jnp.zeros((prices.shape[0], 4), dtype=prices.dtype)
        return updates.at[:, 3].set(jnp.nan)

    updates = jnp.asarray(price_ratio_updates, dtype=prices.dtype)
    if updates.ndim == 1:
        updates = jnp.broadcast_to(updates, (prices.shape[0], updates.shape[0]))
    elif updates.shape[0] == 1 and prices.shape[0] != 1:
        updates = jnp.broadcast_to(updates, (prices.shape[0], updates.shape[1]))
    return updates


def _zero_fee_optimal_trade(Ra, Rb, Va, Vb, prices):
    market_price = prices[0] / prices[1]
    L = compute_invariant(Ra, Rb, Va, Vb)
    Ea_new = jnp.sqrt(L / market_price)
    Eb_new = jnp.sqrt(L * market_price)
    return jnp.array([Ea_new - (Ra + Va), Eb_new - (Rb + Vb)])


def _effective_reserves(real_reserves, Va, Vb):
    return jnp.array([real_reserves[0] + Va, real_reserves[1] + Vb])


def _reclamm_hypersurge_fee_for_trade(
    real_reserves,
    Va,
    Vb,
    candidate_trade,
    oracle_prices,
    base_fee,
    hypersurge_params,
):
    token_in = jnp.argmax(candidate_trade)
    token_out = jnp.argmin(candidate_trade)
    trade_active = jnp.logical_and(
        candidate_trade[token_in] > 0.0,
        candidate_trade[token_out] < 0.0,
    )
    pair_has_oracle = oracle_pair_is_valid(oracle_prices, token_in, token_out)

    effective_before = safe_positive(_effective_reserves(real_reserves, Va, Vb))
    effective_after = safe_positive(effective_before + candidate_trade)
    dev_before = pair_deviation(
        effective_before,
        _WEIGHTS,
        oracle_prices,
        token_in,
        token_out,
    )
    dev_after = pair_deviation(
        effective_after,
        _WEIGHTS,
        oracle_prices,
        token_in,
        token_out,
    )
    worsens = dev_after > dev_before

    arb_fee = ramp_fee(
        base_fee,
        hypersurge_params[0],
        hypersurge_params[1],
        hypersurge_params[2],
        dev_before,
    )
    noise_fee = ramp_fee(
        base_fee,
        hypersurge_params[3],
        hypersurge_params[4],
        hypersurge_params[5],
        dev_after,
    )
    fee = jnp.where(worsens, noise_fee, arb_fee)
    return jnp.where(jnp.logical_and(trade_active, pair_has_oracle), fee, base_fee)


def _reclamm_hypersurge_noise_fee(
    real_reserves,
    Va,
    Vb,
    oracle_prices,
    base_fee,
    hypersurge_params,
):
    effective = safe_positive(_effective_reserves(real_reserves, Va, Vb))
    deviation = max_pair_deviation(effective, _WEIGHTS, oracle_prices)
    fee = ramp_fee(
        base_fee,
        hypersurge_params[3],
        hypersurge_params[4],
        hypersurge_params[5],
        deviation,
    )
    return jnp.where(oracle_vector_is_valid(oracle_prices), fee, base_fee)


def _optimal_arb_trade_with_gamma(
    reserves,
    prices,
    gamma,
    tokens_to_drop,
    active_trade_directions,
    leave_one_out_idxs,
    n,
):
    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_signatures(
            _WEIGHTS,
            prices,
            gamma,
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idxs,
        )
    )
    return parallelised_optimal_trade_sifter(
        reserves,
        _WEIGHTS,
        prices,
        active_initial_weights,
        active_trade_directions,
        per_asset_ratios,
        all_other_assets_ratios,
        tokens_to_drop,
        gamma,
        n,
        0,
    )


def _apply_protocol_fee(reserves_after_trade, trade, fee, protocol_fee_split):
    inbound = jnp.maximum(trade, 0.0)
    protocol_fee = inbound * fee * protocol_fee_split
    return jnp.maximum(reserves_after_trade - protocol_fee, _EPS)


def _reclamm_hypersurge_scan_step_with_fee_revenue(
    carry_list,
    input_list,
    tokens_to_drop,
    active_trade_directions,
    leave_one_out_idxs,
    n,
    hypersurge_params,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    ste_temperature=10.0,
    noise_trader_ratio=0.0,
    noise_model="ratio",
    noise_params=None,
):
    prev_reserves = carry_list[0]
    Va = carry_list[1]
    Vb = carry_list[2]
    step_idx = carry_list[3]
    active_start_ratio = carry_list[4]
    active_target_ratio = carry_list[5]
    active_start_step = carry_list[6]
    active_end_step = carry_list[7]
    active_enabled = carry_list[8]
    prev_lp_supply = carry_list[9]

    prices = input_list[0]
    oracle_prices = input_list[1]
    base_fee = input_list[2]
    arb_thresh = input_list[3]
    arb_fees = input_list[4]
    price_ratio_update = input_list[5]
    lp_supply = input_list[6]
    volatility = input_list[7]
    dow_sin = input_list[8]
    dow_cos = input_list[9]
    noise_base = input_list[10]
    noise_tvl_coeff = input_list[11]
    competitor_tvl = input_list[12]

    scale = lp_supply / prev_lp_supply
    lp_supply_change = lp_supply != prev_lp_supply
    prev_reserves = jnp.where(lp_supply_change, prev_reserves * scale, prev_reserves)
    Va = jnp.where(lp_supply_change, Va * scale, Va)
    Vb = jnp.where(lp_supply_change, Vb * scale, Vb)

    Ra = prev_reserves[0]
    Rb = prev_reserves[1]

    event_has = price_ratio_update[0] > 0.5
    event_target_ratio = jnp.maximum(
        jnp.where(jnp.isfinite(price_ratio_update[1]), price_ratio_update[1], 1.0),
        1.0 + 1e-12,
    )
    event_end_step = jnp.where(
        jnp.isfinite(price_ratio_update[2]), price_ratio_update[2], step_idx
    )
    event_start_override = price_ratio_update[3]

    def _apply_schedule_state(_):
        current_price_ratio = compute_price_ratio(Ra, Rb, Va, Vb)
        start_ratio_from_event = jnp.where(
            jnp.isfinite(event_start_override),
            event_start_override,
            current_price_ratio,
        )
        next_active_start_ratio = jnp.where(
            event_has, start_ratio_from_event, active_start_ratio
        )
        next_active_target_ratio = jnp.where(
            event_has, event_target_ratio, active_target_ratio
        )
        next_active_start_step = jnp.where(event_has, step_idx, active_start_step)
        next_active_end_step = jnp.where(
            event_has, jnp.maximum(event_end_step, step_idx), active_end_step
        )
        next_active_enabled = jnp.where(event_has, True, active_enabled)
        next_active_enabled = jnp.logical_and(
            next_active_enabled, step_idx <= next_active_end_step
        )

        schedule_duration = next_active_end_step - next_active_start_step
        schedule_progress = jnp.where(
            schedule_duration <= 0.0,
            1.0,
            jnp.clip((step_idx - next_active_start_step) / schedule_duration, 0.0, 1.0),
        )
        safe_start_ratio = jnp.maximum(next_active_start_ratio, 1.0 + 1e-12)
        safe_target_ratio = jnp.maximum(next_active_target_ratio, 1.0 + 1e-12)
        scheduled_price_ratio = safe_start_ratio * (
            safe_target_ratio / safe_start_ratio
        ) ** schedule_progress
        scheduled_price_ratio = jnp.where(
            next_active_enabled, scheduled_price_ratio, current_price_ratio
        )
        Va_scheduled, Vb_scheduled = apply_target_price_ratio_to_virtual_balances(
            Ra, Rb, Va, Vb, scheduled_price_ratio
        )
        next_Va = jnp.where(next_active_enabled, Va_scheduled, Va)
        next_Vb = jnp.where(next_active_enabled, Vb_scheduled, Vb)
        return (
            next_Va,
            next_Vb,
            next_active_start_ratio,
            next_active_target_ratio,
            next_active_start_step,
            next_active_end_step,
            next_active_enabled,
        )

    def _skip_schedule_state(_):
        retained_active_enabled = jnp.logical_and(
            active_enabled, step_idx <= active_end_step
        )
        return (
            Va,
            Vb,
            active_start_ratio,
            active_target_ratio,
            active_start_step,
            active_end_step,
            retained_active_enabled,
        )

    active_not_expired = jnp.logical_and(active_enabled, step_idx <= active_end_step)
    schedule_active = jnp.logical_or(event_has, active_not_expired)
    (
        Va,
        Vb,
        active_start_ratio,
        active_target_ratio,
        active_start_step,
        active_end_step,
        active_enabled,
    ) = cond(
        schedule_active,
        _apply_schedule_state,
        _skip_schedule_state,
        operand=None,
    )

    centeredness, is_above = compute_centeredness(Ra, Rb, Va, Vb)
    sqrt_Q = jnp.sqrt(compute_price_ratio(Ra, Rb, Va, Vb))
    market_price = prices[0] / prices[1]

    speed_multiplier = jnp.where(
        centeredness_scaling,
        centeredness_margin / jnp.maximum(centeredness, 1e-10),
        1.0,
    )

    Va_geo, Vb_geo = compute_virtual_balances_updating_price_range(
        Ra,
        Rb,
        Va,
        Vb,
        is_pool_above_center=is_above,
        daily_price_shift_base=daily_price_shift_base,
        seconds_elapsed=seconds_per_step * speed_multiplier,
        sqrt_price_ratio=sqrt_Q,
    )
    Va_cal, Vb_cal = compute_virtual_balances_constant_arc_length(
        Ra,
        Rb,
        Va,
        Vb,
        is_pool_above_center=is_above,
        arc_length_speed=arc_length_speed * speed_multiplier,
        seconds_elapsed=seconds_per_step,
        sqrt_price_ratio=sqrt_Q,
        market_price=market_price,
    )
    use_cal = arc_length_speed > 0.0
    Va_updated = jnp.where(use_cal, Va_cal, Va_geo)
    Vb_updated = jnp.where(use_cal, Vb_cal, Vb_geo)

    out_of_range_gate = _ste_less_than(
        centeredness, centeredness_margin, ste_temperature
    )
    Va = _ste_select(out_of_range_gate, Va_updated, Va)
    Vb = _ste_select(out_of_range_gate, Vb_updated, Vb)

    effective_reserves = _effective_reserves(prev_reserves, Va, Vb)
    zero_fee_trade = _zero_fee_optimal_trade(Ra, Rb, Va, Vb, prices)

    preview_fee = _reclamm_hypersurge_fee_for_trade(
        prev_reserves,
        Va,
        Vb,
        zero_fee_trade,
        oracle_prices,
        base_fee,
        hypersurge_params,
    )
    preview_trade = _optimal_arb_trade_with_gamma(
        effective_reserves,
        prices,
        fee_to_gamma(preview_fee),
        tokens_to_drop,
        active_trade_directions,
        leave_one_out_idxs,
        n,
    )
    arb_fee = _reclamm_hypersurge_fee_for_trade(
        prev_reserves,
        Va,
        Vb,
        preview_trade,
        oracle_prices,
        base_fee,
        hypersurge_params,
    )
    arb_gamma = fee_to_gamma(arb_fee)
    optimal_arb_trade = _optimal_arb_trade_with_gamma(
        effective_reserves,
        prices,
        arb_gamma,
        tokens_to_drop,
        active_trade_directions,
        leave_one_out_idxs,
        n,
    )

    profit_to_arb = -(optimal_arb_trade * prices).sum() - arb_thresh
    arb_external_cost = 0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
    trade_gate = _ste_greater_equal(
        profit_to_arb, arb_external_cost, ste_temperature
    )
    applied_trade = _ste_select(
        trade_gate, optimal_arb_trade, jnp.zeros_like(optimal_arb_trade)
    )

    Ra_trade = Ra + applied_trade[0]
    Rb_trade = Rb + applied_trade[1]

    dust_a = _DUST_USD / prices[0]
    dust_b = _DUST_USD / prices[1]
    drain_a = jnp.maximum(Ra - dust_a, 0.0)
    drain_b = jnp.maximum(Rb - dust_b, 0.0)
    edge_a = _jax_calc_G3M_trade_from_exact_in_given_out(
        effective_reserves,
        _WEIGHTS,
        token_in=1,
        token_out=0,
        amount_out=drain_a,
        gamma=arb_gamma,
    )
    edge_b = _jax_calc_G3M_trade_from_exact_in_given_out(
        effective_reserves,
        _WEIGHTS,
        token_in=0,
        token_out=1,
        amount_out=drain_b,
        gamma=arb_gamma,
    )

    clamp_a = Ra_trade < 0
    clamp_b = Rb_trade < 0
    final_arb_trade = jnp.where(
        clamp_a,
        edge_a,
        jnp.where(clamp_b, edge_b, applied_trade),
    )

    reserves_after_arb = prev_reserves + final_arb_trade
    reserves_after_arb = _apply_protocol_fee(
        reserves_after_arb,
        final_arb_trade,
        arb_fee,
        protocol_fee_split,
    )
    arb_lp_fee_income = (
        jnp.maximum(final_arb_trade, 0.0) * arb_fee * (1.0 - protocol_fee_split)
    )
    lp_fee_revenue_usd = (arb_lp_fee_income * prices).sum()

    noise_fee = _reclamm_hypersurge_noise_fee(
        reserves_after_arb,
        Va,
        Vb,
        oracle_prices,
        base_fee,
        hypersurge_params,
    )

    Ra_new = reserves_after_arb[0]
    Rb_new = reserves_after_arb[1]

    if noise_model == "ratio":
        lp_noise_gamma = fee_to_gamma(noise_fee * (1.0 - protocol_fee_split))
        noisy_reserves = calculate_reserves_after_noise_trade(
            final_arb_trade,
            reserves_after_arb,
            prices,
            noise_trader_ratio,
            lp_noise_gamma,
        )
        noise_lp_fee_income_usd = (
            noise_trader_ratio
            * noise_fee
            * (1.0 - protocol_fee_split)
            * jnp.sum(jnp.maximum(final_arb_trade, 0.0) * prices)
        )
        Ra_new = jnp.where(noise_trader_ratio > 0.0, noisy_reserves[0], Ra_new)
        Rb_new = jnp.where(noise_trader_ratio > 0.0, noisy_reserves[1], Rb_new)
        lp_fee_revenue_usd = jnp.where(
            noise_trader_ratio > 0.0,
            lp_fee_revenue_usd + noise_lp_fee_income_usd,
            lp_fee_revenue_usd,
        )
    elif noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
        arb_volume = 0.5 * jnp.sum(jnp.abs(final_arb_trade) * prices)
        effective_value = (Ra_new + Va) * prices[0] + (Rb_new + Vb) * prices[1]
        noise_cfg = noise_params if noise_params is not None else {}
        if noise_model == "tsoukalas_sqrt":
            noise_vol = reclamm_tsoukalas_sqrt_noise_volume(
                effective_value, arb_gamma, volatility, arb_volume, noise_cfg
            )
        elif noise_model == "tsoukalas_log":
            noise_vol = reclamm_tsoukalas_log_noise_volume(
                effective_value, arb_gamma, volatility, arb_volume, noise_cfg
            )
        else:
            noise_vol = reclamm_loglinear_noise_volume(
                effective_value, arb_gamma, volatility, arb_volume, noise_cfg
            )
        minutes_per_step = seconds_per_step / 60.0
        noise_lp_fee_income_usd = (
            noise_fee * (1.0 - protocol_fee_split) * noise_vol * minutes_per_step
        )
        scale = 1.0 + noise_lp_fee_income_usd / jnp.maximum(effective_value, 1e-8)
        Ra_new = (Ra_new + Va) * scale - Va
        Rb_new = (Rb_new + Vb) * scale - Vb
        lp_fee_revenue_usd = lp_fee_revenue_usd + noise_lp_fee_income_usd
    elif noise_model == "calibrated":
        arb_volume = 0.5 * jnp.sum(jnp.abs(final_arb_trade) * prices)
        effective_value = (Ra_new + Va) * prices[0] + (Rb_new + Vb) * prices[1]
        noise_cfg = noise_params if noise_params is not None else {}
        noise_vol = reclamm_calibrated_noise_volume(
            effective_value,
            arb_gamma,
            volatility,
            arb_volume,
            dow_sin,
            dow_cos,
            noise_cfg,
        )
        minutes_per_step = seconds_per_step / 60.0
        noise_lp_fee_income_usd = (
            noise_fee * (1.0 - protocol_fee_split) * noise_vol * minutes_per_step
        )
        scale = 1.0 + noise_lp_fee_income_usd / jnp.maximum(effective_value, 1e-8)
        Ra_new = (Ra_new + Va) * scale - Va
        Rb_new = (Rb_new + Vb) * scale - Vb
        lp_fee_revenue_usd = lp_fee_revenue_usd + noise_lp_fee_income_usd
    elif noise_model == "market_linear":
        effective_value = (Ra_new + Va) * prices[0] + (Rb_new + Vb) * prices[1]
        noise_cfg = noise_params if noise_params is not None else {}
        noise_vol = reclamm_market_linear_noise_volume(
            effective_value,
            noise_base,
            noise_tvl_coeff,
            tvl_mean=noise_cfg.get("tvl_mean", 0.0),
            tvl_std=noise_cfg.get("tvl_std", 1.0),
        )
        minutes_per_step = seconds_per_step / 60.0
        noise_lp_fee_income_usd = (
            noise_fee * (1.0 - protocol_fee_split) * noise_vol * minutes_per_step
        )
        scale = 1.0 + noise_lp_fee_income_usd / jnp.maximum(effective_value, 1e-8)
        Ra_new = (Ra_new + Va) * scale - Va
        Rb_new = (Rb_new + Vb) * scale - Vb
        lp_fee_revenue_usd = lp_fee_revenue_usd + noise_lp_fee_income_usd
    elif noise_model == "mm_observed":
        effective_value = (Ra_new + Va) * prices[0] + (Rb_new + Vb) * prices[1]
        noise_vol = reclamm_mm_observed_noise_volume(
            effective_value, noise_base, competitor_tvl
        )
        minutes_per_step = seconds_per_step / 60.0
        noise_lp_fee_income_usd = (
            noise_fee * (1.0 - protocol_fee_split) * noise_vol * minutes_per_step
        )
        scale = 1.0 + noise_lp_fee_income_usd / jnp.maximum(effective_value, 1e-8)
        Ra_new = (Ra_new + Va) * scale - Va
        Rb_new = (Rb_new + Vb) * scale - Vb
        lp_fee_revenue_usd = lp_fee_revenue_usd + noise_lp_fee_income_usd

    new_reserves = jnp.array([Ra_new, Rb_new])
    return [
        new_reserves,
        Va,
        Vb,
        step_idx + 1.0,
        active_start_ratio,
        active_target_ratio,
        active_start_step,
        active_end_step,
        active_enabled,
        lp_supply,
    ], (new_reserves, lp_fee_revenue_usd)


def _reclamm_hypersurge_scan_step(
    carry_list,
    input_list,
    tokens_to_drop,
    active_trade_directions,
    leave_one_out_idxs,
    n,
    hypersurge_params,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    ste_temperature=10.0,
    noise_trader_ratio=0.0,
    noise_model="ratio",
    noise_params=None,
):
    new_carry, (new_reserves, _fee_revenue) = (
        _reclamm_hypersurge_scan_step_with_fee_revenue(
            carry_list,
            input_list,
            tokens_to_drop=tokens_to_drop,
            active_trade_directions=active_trade_directions,
            leave_one_out_idxs=leave_one_out_idxs,
            n=n,
            hypersurge_params=hypersurge_params,
            centeredness_margin=centeredness_margin,
            daily_price_shift_base=daily_price_shift_base,
            seconds_per_step=seconds_per_step,
            arc_length_speed=arc_length_speed,
            centeredness_scaling=centeredness_scaling,
            protocol_fee_split=protocol_fee_split,
            ste_temperature=ste_temperature,
            noise_trader_ratio=noise_trader_ratio,
            noise_model=noise_model,
            noise_params=noise_params,
        )
    )
    return new_carry, new_reserves


@partial(jit, static_argnames=("noise_model",))
def _jax_calc_reclamm_hypersurge_reserves(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    oracle_prices,
    hypersurge_params,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    price_ratio_updates=None,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    ste_temperature=10.0,
    noise_trader_ratio=0.0,
    lp_supply_array=None,
    noise_model="ratio",
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
    competitor_tvl_array=None,
):
    if lp_supply_array is None:
        lp_supply_array = jnp.ones((prices.shape[0],), dtype=prices.dtype)
    else:
        lp_supply_array = broadcast_scan_vector(lp_supply_array, prices.shape[0])

    fees = broadcast_scan_vector(fees, prices.shape[0])
    arb_thresh = broadcast_scan_vector(arb_thresh, prices.shape[0])
    arb_fees = broadcast_scan_vector(arb_fees, prices.shape[0])
    oracle_prices = _broadcast_oracle_prices(oracle_prices, prices)
    price_ratio_updates = _broadcast_schedule_array(price_ratio_updates, prices)
    volatility_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if volatility_array is None
        else volatility_array,
        prices.shape[0],
    )
    dow_sin_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if dow_sin_array is None
        else dow_sin_array,
        prices.shape[0],
    )
    dow_cos_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if dow_cos_array is None
        else dow_cos_array,
        prices.shape[0],
    )
    noise_base_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if noise_base_array is None
        else noise_base_array,
        prices.shape[0],
    )
    noise_tvl_coeff_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if noise_tvl_coeff_array is None
        else noise_tvl_coeff_array,
        prices.shape[0],
    )
    competitor_tvl_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if competitor_tvl_array is None
        else competitor_tvl_array,
        prices.shape[0],
    )

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, 2)
    )

    scan_fn = Partial(
        _reclamm_hypersurge_scan_step,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        leave_one_out_idxs=leave_one_out_idxs,
        n=2,
        hypersurge_params=jnp.asarray(hypersurge_params, dtype=prices.dtype),
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        ste_temperature=ste_temperature,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        noise_params=noise_params if noise_params is not None else {},
    )

    carry_init = [
        initial_reserves,
        initial_Va,
        initial_Vb,
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.array(False),
        lp_supply_array[0],
    ]
    _, reserves = scan(
        scan_fn,
        carry_init,
        [
            prices,
            oracle_prices,
            fees,
            arb_thresh,
            arb_fees,
            price_ratio_updates,
            lp_supply_array,
            volatility_array,
            dow_sin_array,
            dow_cos_array,
            noise_base_array,
            noise_tvl_coeff_array,
            competitor_tvl_array,
        ],
    )
    return reserves


@partial(jit, static_argnames=("noise_model",))
def _jax_calc_reclamm_hypersurge_reserves_and_fee_revenue(
    initial_reserves,
    initial_Va,
    initial_Vb,
    prices,
    oracle_prices,
    hypersurge_params,
    centeredness_margin,
    daily_price_shift_base,
    seconds_per_step,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    price_ratio_updates=None,
    all_sig_variations=None,
    arc_length_speed=0.0,
    centeredness_scaling=False,
    protocol_fee_split=0.0,
    ste_temperature=10.0,
    noise_trader_ratio=0.0,
    lp_supply_array=None,
    noise_model="ratio",
    noise_params=None,
    volatility_array=None,
    dow_sin_array=None,
    dow_cos_array=None,
    noise_base_array=None,
    noise_tvl_coeff_array=None,
    competitor_tvl_array=None,
):
    if lp_supply_array is None:
        lp_supply_array = jnp.ones((prices.shape[0],), dtype=prices.dtype)
    else:
        lp_supply_array = broadcast_scan_vector(lp_supply_array, prices.shape[0])

    fees = broadcast_scan_vector(fees, prices.shape[0])
    arb_thresh = broadcast_scan_vector(arb_thresh, prices.shape[0])
    arb_fees = broadcast_scan_vector(arb_fees, prices.shape[0])
    oracle_prices = _broadcast_oracle_prices(oracle_prices, prices)
    price_ratio_updates = _broadcast_schedule_array(price_ratio_updates, prices)
    volatility_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if volatility_array is None
        else volatility_array,
        prices.shape[0],
    )
    dow_sin_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if dow_sin_array is None
        else dow_sin_array,
        prices.shape[0],
    )
    dow_cos_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if dow_cos_array is None
        else dow_cos_array,
        prices.shape[0],
    )
    noise_base_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if noise_base_array is None
        else noise_base_array,
        prices.shape[0],
    )
    noise_tvl_coeff_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if noise_tvl_coeff_array is None
        else noise_tvl_coeff_array,
        prices.shape[0],
    )
    competitor_tvl_array = broadcast_scan_vector(
        jnp.zeros((1,), dtype=prices.dtype)
        if competitor_tvl_array is None
        else competitor_tvl_array,
        prices.shape[0],
    )

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, 2)
    )

    scan_fn = Partial(
        _reclamm_hypersurge_scan_step_with_fee_revenue,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        leave_one_out_idxs=leave_one_out_idxs,
        n=2,
        hypersurge_params=jnp.asarray(hypersurge_params, dtype=prices.dtype),
        centeredness_margin=centeredness_margin,
        daily_price_shift_base=daily_price_shift_base,
        seconds_per_step=seconds_per_step,
        arc_length_speed=arc_length_speed,
        centeredness_scaling=centeredness_scaling,
        protocol_fee_split=protocol_fee_split,
        ste_temperature=ste_temperature,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        noise_params=noise_params if noise_params is not None else {},
    )

    carry_init = [
        initial_reserves,
        initial_Va,
        initial_Vb,
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.float64(0.0),
        jnp.array(False),
        lp_supply_array[0],
    ]
    _, (reserves, fee_revenue) = scan(
        scan_fn,
        carry_init,
        [
            prices,
            oracle_prices,
            fees,
            arb_thresh,
            arb_fees,
            price_ratio_updates,
            lp_supply_array,
            volatility_array,
            dow_sin_array,
            dow_cos_array,
            noise_base_array,
            noise_tvl_coeff_array,
            competitor_tvl_array,
        ],
    )
    return reserves, fee_revenue
