from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.lax import scan
from jax.tree_util import Partial

from quantammsim.pools.G3M.G3M_trades import (
    _jax_calc_G3M_trade_from_exact_out_given_in,
)
from quantammsim.pools.G3M.optimal_n_pool_arb import (
    precalc_components_of_optimal_trade_across_signatures,
    precalc_shared_values_for_all_signatures,
    parallelised_optimal_trade_sifter,
)
from quantammsim.pools.noise_trades import (
    calculate_reserves_after_noise_trade,
    reclamm_market_linear_noise_volume,
)


_EPS = 1e-18
_MAX_FEE = 0.999999


def _safe_positive(values):
    values = jnp.asarray(values)
    values = jnp.where(jnp.isfinite(values), values, 1.0)
    return jnp.maximum(values, _EPS)


def _fee_to_gamma(fee):
    return jnp.maximum(1.0 - jnp.clip(fee, 0.0, _MAX_FEE), _EPS)


def _pair_pool_price(reserves, weights, token_in, token_out):
    """Pool spot amount-out-per-amount-in for a weighted-pool token pair."""
    token_in = jnp.int32(token_in)
    token_out = jnp.int32(token_out)
    reserves = _safe_positive(reserves)
    weights = _safe_positive(weights)
    numerator = reserves[token_out] * weights[token_in]
    denominator = reserves[token_in] * weights[token_out]
    return numerator / jnp.maximum(denominator, _EPS)


def _pair_oracle_price(oracle_prices, token_in, token_out):
    """External amount-out-per-amount-in, assuming common-numeraire prices."""
    token_in = jnp.int32(token_in)
    token_out = jnp.int32(token_out)
    oracle_prices = _safe_positive(oracle_prices)
    return oracle_prices[token_in] / jnp.maximum(oracle_prices[token_out], _EPS)


def _pair_deviation(reserves, weights, oracle_prices, token_in, token_out):
    pool_price = _pair_pool_price(reserves, weights, token_in, token_out)
    oracle_price = _pair_oracle_price(oracle_prices, token_in, token_out)
    ratio = pool_price / jnp.maximum(oracle_price, _EPS)
    ratio = jnp.where(jnp.isfinite(ratio), ratio, 1.0)
    return jnp.abs(ratio - 1.0)


def _max_pair_deviation(reserves, weights, oracle_prices):
    reserves = _safe_positive(reserves)
    weights = _safe_positive(weights)
    oracle_prices = _safe_positive(oracle_prices)

    pool_prices = (reserves[None, :] * weights[:, None]) / jnp.maximum(
        reserves[:, None] * weights[None, :],
        _EPS,
    )
    oracle_pair_prices = oracle_prices[:, None] / jnp.maximum(
        oracle_prices[None, :],
        _EPS,
    )
    ratios = pool_prices / jnp.maximum(oracle_pair_prices, _EPS)
    ratios = jnp.where(jnp.isfinite(ratios), ratios, 1.0)
    deviations = jnp.abs(ratios - 1.0)
    off_diagonal = ~jnp.eye(reserves.shape[0], dtype=bool)
    return jnp.max(jnp.where(off_diagonal, deviations, 0.0))


def _ramp_fee(base_fee, max_fee, threshold, cap, deviation):
    max_fee = jnp.maximum(max_fee, base_fee)
    threshold = jnp.maximum(threshold, 0.0)
    cap = jnp.maximum(cap, threshold + _EPS)
    span = jnp.maximum(cap - threshold, _EPS)
    ramp = jnp.clip((deviation - threshold) / span, 0.0, 1.0)
    fee = base_fee + (max_fee - base_fee) * ramp
    fee = jnp.where(deviation <= threshold, base_fee, fee)
    return jnp.clip(fee, 0.0, _MAX_FEE)


def _hypersurge_fee_for_trade(
    reserves,
    candidate_trade,
    weights,
    oracle_prices,
    token_in,
    token_out,
    base_fee,
    hypersurge_params,
):
    """Select arb/noise fee params from whether a candidate trade worsens peg deviation."""
    candidate_reserves = _safe_positive(reserves + candidate_trade)
    dev_before = _pair_deviation(reserves, weights, oracle_prices, token_in, token_out)
    dev_after = _pair_deviation(
        candidate_reserves, weights, oracle_prices, token_in, token_out
    )
    trade_active = jnp.logical_and(
        candidate_trade[token_in] > 0.0,
        candidate_trade[token_out] < 0.0,
    )

    worsens = dev_after > dev_before
    arb_fee = _ramp_fee(
        base_fee,
        hypersurge_params[0],
        hypersurge_params[1],
        hypersurge_params[2],
        dev_before,
    )
    noise_fee = _ramp_fee(
        base_fee,
        hypersurge_params[3],
        hypersurge_params[4],
        hypersurge_params[5],
        dev_after,
    )
    fee = jnp.where(worsens, noise_fee, arb_fee)
    return jnp.where(trade_active, fee, base_fee)


def _hypersurge_noise_fee(reserves, weights, oracle_prices, base_fee, hypersurge_params):
    deviation = _max_pair_deviation(reserves, weights, oracle_prices)
    return _ramp_fee(
        base_fee,
        hypersurge_params[3],
        hypersurge_params[4],
        hypersurge_params[5],
        deviation,
    )


def _zero_fee_optimal_trade(reserves, weights, prices):
    current_value = jnp.sum(reserves * prices)
    quoted_prices = current_value * weights / jnp.maximum(reserves, _EPS)
    price_change_ratio = prices / jnp.maximum(quoted_prices, _EPS)
    price_product_change_ratio = jnp.prod(price_change_ratio**weights)
    reserves_ratios_from_price_change = (
        price_product_change_ratio / jnp.maximum(price_change_ratio, _EPS)
    )
    return reserves * reserves_ratios_from_price_change - reserves


def _trade_pair_from_delta(trade):
    token_in = jnp.argmax(trade)
    token_out = jnp.argmin(trade)
    return token_in, token_out


def _apply_protocol_fee(reserves_after_trade, trade, fee, protocol_fee_split):
    inbound = jnp.maximum(trade, 0.0)
    protocol_fee = inbound * fee * protocol_fee_split
    return jnp.maximum(reserves_after_trade - protocol_fee, _EPS)


def _optimal_arb_trade_with_gamma(
    reserves,
    weights,
    prices,
    gamma,
    tokens_to_drop,
    active_trade_directions,
    leave_one_out_idxs,
    n,
):
    active_initial_weights, per_asset_ratios, all_other_assets_ratios = (
        precalc_components_of_optimal_trade_across_signatures(
            weights,
            prices,
            gamma,
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idxs,
        )
    )
    return parallelised_optimal_trade_sifter(
        reserves,
        weights,
        prices,
        active_initial_weights,
        active_trade_directions,
        per_asset_ratios,
        all_other_assets_ratios,
        tokens_to_drop,
        gamma,
        n,
        -1e-15,
    )


def _broadcast_scan_vector(values, scan_len):
    values = jnp.asarray(values)
    if values.ndim == 0:
        values = values.reshape((1,))
    values = jnp.ravel(values)
    return jnp.where(values.size == 1, jnp.full((scan_len,), values[0]), values)


def _broadcast_oracle_prices(oracle_prices, prices):
    oracle_prices = jnp.asarray(oracle_prices)
    if oracle_prices.ndim == 1:
        oracle_prices = oracle_prices.reshape((1, oracle_prices.shape[0]))
    if oracle_prices.shape[-1] != prices.shape[-1]:
        oracle_prices = prices
    elif oracle_prices.shape[0] == 1:
        oracle_prices = jnp.broadcast_to(oracle_prices, prices.shape)
    return oracle_prices


def _hypersurge_scan_step(
    carry_list,
    input_list,
    weights,
    tokens_to_drop,
    active_trade_directions,
    leave_one_out_idxs,
    n,
    do_trades,
    do_arb,
    hypersurge_params,
    protocol_fee_split,
    noise_trader_ratio,
    noise_model,
    tvl_mean,
    tvl_std,
    minutes_per_step,
):
    reserves = carry_list[1]
    prev_lp_supply = carry_list[2]

    prices = input_list[0]
    oracle_prices = input_list[1]
    base_fee = input_list[2]
    arb_thresh = input_list[3]
    arb_fees = input_list[4]
    trade = input_list[5]
    lp_supply = input_list[6]
    noise_base = input_list[7]
    noise_tvl_coeff = input_list[8]

    reserves = jnp.where(
        lp_supply != prev_lp_supply,
        reserves * lp_supply / jnp.maximum(prev_lp_supply, _EPS),
        reserves,
    )

    applied_arb_trade = jnp.zeros_like(reserves)
    if do_arb:
        preview_trade = _zero_fee_optimal_trade(reserves, weights, prices)
        token_in, token_out = _trade_pair_from_delta(preview_trade)
        preview_fee = _hypersurge_fee_for_trade(
            reserves,
            preview_trade,
            weights,
            oracle_prices,
            token_in,
            token_out,
            base_fee,
            hypersurge_params,
        )
        preview_trade = _optimal_arb_trade_with_gamma(
            reserves,
            weights,
            prices,
            _fee_to_gamma(preview_fee),
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idxs,
            n,
        )
        token_in, token_out = _trade_pair_from_delta(preview_trade)
        arb_fee = _hypersurge_fee_for_trade(
            reserves,
            preview_trade,
            weights,
            oracle_prices,
            token_in,
            token_out,
            base_fee,
            hypersurge_params,
        )
        optimal_arb_trade = _optimal_arb_trade_with_gamma(
            reserves,
            weights,
            prices,
            _fee_to_gamma(arb_fee),
            tokens_to_drop,
            active_trade_directions,
            leave_one_out_idxs,
            n,
        )
        profit_to_arb = -(optimal_arb_trade * prices).sum() - arb_thresh
        arb_external_rebalance_cost = (
            0.5 * arb_fees * (jnp.abs(optimal_arb_trade) * prices).sum()
        )
        arb_profitable = profit_to_arb >= arb_external_rebalance_cost
        applied_arb_trade = jnp.where(
            arb_profitable,
            optimal_arb_trade,
            applied_arb_trade,
        )
        reserves = _apply_protocol_fee(
            reserves + applied_arb_trade,
            applied_arb_trade,
            arb_fee,
            protocol_fee_split,
        )

    if do_trades:
        token_in = jnp.int32(trade[0])
        token_out = jnp.int32(trade[1])
        amount_in = trade[2]
        preview_trade = _jax_calc_G3M_trade_from_exact_out_given_in(
            reserves,
            weights,
            token_in,
            token_out,
            amount_in,
            gamma=_fee_to_gamma(base_fee),
        )
        trade_fee = _hypersurge_fee_for_trade(
            reserves,
            preview_trade,
            weights,
            oracle_prices,
            token_in,
            token_out,
            base_fee,
            hypersurge_params,
        )
        applied_user_trade = _jax_calc_G3M_trade_from_exact_out_given_in(
            reserves,
            weights,
            token_in,
            token_out,
            amount_in,
            gamma=_fee_to_gamma(trade_fee),
        )
        reserves = _apply_protocol_fee(
            reserves + applied_user_trade,
            applied_user_trade,
            trade_fee,
            protocol_fee_split,
        )

    if noise_model == "ratio":
        noise_fee = _hypersurge_noise_fee(
            reserves, weights, oracle_prices, base_fee, hypersurge_params
        )
        lp_noise_gamma = _fee_to_gamma(noise_fee * (1.0 - protocol_fee_split))
        noisy_reserves = calculate_reserves_after_noise_trade(
            applied_arb_trade,
            reserves,
            prices,
            noise_trader_ratio,
            lp_noise_gamma,
        )
        reserves = jnp.where(noise_trader_ratio > 0.0, noisy_reserves, reserves)
    elif noise_model == "market_linear":
        noise_fee = _hypersurge_noise_fee(
            reserves, weights, oracle_prices, base_fee, hypersurge_params
        )
        pool_value = jnp.sum(reserves * prices)
        noise_volume = reclamm_market_linear_noise_volume(
            pool_value,
            noise_base,
            noise_tvl_coeff,
            tvl_mean=tvl_mean,
            tvl_std=tvl_std,
        )
        lp_fee_income = (
            noise_fee * (1.0 - protocol_fee_split) * noise_volume * minutes_per_step
        )
        reserves = reserves * (1.0 + lp_fee_income / jnp.maximum(pool_value, 1e-8))

    return [
        prices,
        reserves,
        lp_supply,
    ], reserves


@partial(jit, static_argnames=("do_trades", "do_arb", "noise_model"))
def _jax_calc_hypersurge_balancer_reserves(
    initial_reserves,
    weights,
    prices,
    oracle_prices,
    fees=0.003,
    arb_thresh=0.0,
    arb_fees=0.0,
    all_sig_variations=None,
    trades=None,
    do_trades=False,
    do_arb=True,
    lp_supply_array=None,
    hypersurge_params=None,
    noise_trader_ratio=0.0,
    protocol_fee_split=0.0,
    noise_model="ratio",
    noise_base_array=None,
    noise_tvl_coeff_array=None,
    tvl_mean=0.0,
    tvl_std=1.0,
    minutes_per_step=1.0,
):
    n_assets = weights.shape[0]
    scan_len = prices.shape[0]

    fees = _broadcast_scan_vector(fees, scan_len)
    arb_thresh = _broadcast_scan_vector(arb_thresh, scan_len)
    arb_fees = _broadcast_scan_vector(arb_fees, scan_len)
    oracle_prices = _broadcast_oracle_prices(oracle_prices, prices)

    if trades is None:
        if do_trades:
            raise ValueError("Trades must be provided when do_trades=True.")
        trades = jnp.zeros((scan_len, 3), dtype=prices.dtype)

    if lp_supply_array is None:
        lp_supply_array = jnp.ones((scan_len,), dtype=prices.dtype)
    else:
        lp_supply_array = _broadcast_scan_vector(lp_supply_array, scan_len)

    if hypersurge_params is None:
        hypersurge_params = jnp.array([fees[0], 0.0, 1.0, fees[0], 0.0, 1.0])
    else:
        hypersurge_params = jnp.asarray(hypersurge_params, dtype=prices.dtype)

    if noise_base_array is None:
        noise_base_array = jnp.zeros((scan_len,), dtype=prices.dtype)
    else:
        noise_base_array = _broadcast_scan_vector(noise_base_array, scan_len)
    if noise_tvl_coeff_array is None:
        noise_tvl_coeff_array = jnp.zeros((scan_len,), dtype=prices.dtype)
    else:
        noise_tvl_coeff_array = _broadcast_scan_vector(
            noise_tvl_coeff_array, scan_len
        )

    _, active_trade_directions, tokens_to_drop, leave_one_out_idxs = (
        precalc_shared_values_for_all_signatures(all_sig_variations, n_assets)
    )

    scan_fn = Partial(
        _hypersurge_scan_step,
        weights=weights,
        tokens_to_drop=tokens_to_drop,
        active_trade_directions=active_trade_directions,
        leave_one_out_idxs=leave_one_out_idxs,
        n=n_assets,
        do_trades=do_trades,
        do_arb=do_arb,
        hypersurge_params=hypersurge_params,
        protocol_fee_split=protocol_fee_split,
        noise_trader_ratio=noise_trader_ratio,
        noise_model=noise_model,
        tvl_mean=tvl_mean,
        tvl_std=tvl_std,
        minutes_per_step=minutes_per_step,
    )

    carry_list_init = [
        prices[0],
        initial_reserves,
        lp_supply_array[0],
    ]
    _, reserves = scan(
        scan_fn,
        carry_list_init,
        [
            prices,
            oracle_prices,
            fees,
            arb_thresh,
            arb_fees,
            trades,
            lp_supply_array,
            noise_base_array,
            noise_tvl_coeff_array,
        ],
    )

    return reserves
