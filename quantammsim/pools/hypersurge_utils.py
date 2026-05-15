from typing import Any, Dict

import numpy as np

import jax.numpy as jnp


HYPERSURGE_PARAM_KEYS = (
    "hypersurge_arb_max_fee",
    "hypersurge_arb_threshold",
    "hypersurge_arb_cap_deviation",
    "hypersurge_noise_max_fee",
    "hypersurge_noise_threshold",
    "hypersurge_noise_cap_deviation",
)

_EPS = 1e-18
_MAX_FEE = 0.999999


def _coalesce(value, default):
    return default if value is None else value


def _scalar_like(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        flat = np.asarray(value).reshape(-1)
        return flat[0] if flat.size else default
    return value


def run_fingerprint_hypersurge_defaults(run_fingerprint: Dict[str, Any]):
    base_fee = _scalar_like(run_fingerprint.get("fees", 0.0), default=0.0)

    raw_params = run_fingerprint.get("hypersurge_params")
    if raw_params is not None:
        if isinstance(raw_params, dict):
            shared_max = raw_params.get("max_surge_fee", base_fee)
            shared_threshold = raw_params.get("threshold", 0.0)
            shared_cap = raw_params.get("cap_deviation", 1.0)
            return {
                "hypersurge_arb_max_fee": raw_params.get("arb_max_fee", shared_max),
                "hypersurge_arb_threshold": raw_params.get(
                    "arb_threshold", shared_threshold
                ),
                "hypersurge_arb_cap_deviation": raw_params.get(
                    "arb_cap_deviation", shared_cap
                ),
                "hypersurge_noise_max_fee": raw_params.get(
                    "noise_max_fee", shared_max
                ),
                "hypersurge_noise_threshold": raw_params.get(
                    "noise_threshold", shared_threshold
                ),
                "hypersurge_noise_cap_deviation": raw_params.get(
                    "noise_cap_deviation", shared_cap
                ),
            }

        raw_params = np.asarray(raw_params, dtype=np.float64).reshape(-1)
        if raw_params.size != len(HYPERSURGE_PARAM_KEYS):
            raise ValueError(
                "hypersurge_params must contain exactly six values: "
                + ", ".join(HYPERSURGE_PARAM_KEYS)
            )
        return dict(zip(HYPERSURGE_PARAM_KEYS, raw_params))

    shared_max = _coalesce(
        run_fingerprint.get("hypersurge_max_surge_fee"),
        _coalesce(run_fingerprint.get("hypersurge_max_fee"), base_fee),
    )
    shared_threshold = _coalesce(
        run_fingerprint.get("hypersurge_threshold"),
        0.0,
    )
    shared_cap = _coalesce(
        run_fingerprint.get("hypersurge_cap_deviation"),
        1.0,
    )
    return {
        "hypersurge_arb_max_fee": _coalesce(
            run_fingerprint.get("hypersurge_arb_max_fee"), shared_max
        ),
        "hypersurge_arb_threshold": _coalesce(
            run_fingerprint.get("hypersurge_arb_threshold"), shared_threshold
        ),
        "hypersurge_arb_cap_deviation": _coalesce(
            run_fingerprint.get("hypersurge_arb_cap_deviation"), shared_cap
        ),
        "hypersurge_noise_max_fee": _coalesce(
            run_fingerprint.get("hypersurge_noise_max_fee"), shared_max
        ),
        "hypersurge_noise_threshold": _coalesce(
            run_fingerprint.get("hypersurge_noise_threshold"), shared_threshold
        ),
        "hypersurge_noise_cap_deviation": _coalesce(
            run_fingerprint.get("hypersurge_noise_cap_deviation"), shared_cap
        ),
    }


def hypersurge_params_from_params(params: Dict[str, Any], run_fingerprint: Dict[str, Any]):
    if "hypersurge_params" in params:
        return jnp.ravel(params["hypersurge_params"])

    if all(key in params for key in HYPERSURGE_PARAM_KEYS):
        return jnp.asarray(
            [jnp.squeeze(params[key]) for key in HYPERSURGE_PARAM_KEYS],
            dtype=jnp.float64,
        )

    defaults = run_fingerprint_hypersurge_defaults(run_fingerprint)
    return jnp.asarray(
        [defaults[key] for key in HYPERSURGE_PARAM_KEYS],
        dtype=jnp.float64,
    )


def safe_positive(values):
    values = jnp.asarray(values)
    values = jnp.where(jnp.isfinite(values), values, 1.0)
    return jnp.maximum(values, _EPS)


def fee_to_gamma(fee):
    return jnp.maximum(1.0 - jnp.clip(fee, 0.0, _MAX_FEE), _EPS)


def ramp_fee(base_fee, max_fee, threshold, cap, deviation):
    max_fee = jnp.maximum(max_fee, base_fee)
    threshold = jnp.maximum(threshold, 0.0)
    cap = jnp.maximum(cap, threshold + _EPS)
    span = jnp.maximum(cap - threshold, _EPS)
    ramp = jnp.clip((deviation - threshold) / span, 0.0, 1.0)
    fee = base_fee + (max_fee - base_fee) * ramp
    fee = jnp.where(deviation <= threshold, base_fee, fee)
    return jnp.clip(fee, 0.0, _MAX_FEE)


def oracle_pair_is_valid(oracle_prices, token_in, token_out):
    oracle_prices = jnp.asarray(oracle_prices)
    token_in = jnp.int32(token_in)
    token_out = jnp.int32(token_out)
    pair_prices = jnp.asarray([oracle_prices[token_in], oracle_prices[token_out]])
    return jnp.all(jnp.isfinite(pair_prices) & (pair_prices > 0.0))


def oracle_vector_is_valid(oracle_prices):
    oracle_prices = jnp.asarray(oracle_prices)
    return jnp.all(jnp.isfinite(oracle_prices) & (oracle_prices > 0.0))


def pair_pool_price(reserves, weights, token_in, token_out):
    token_in = jnp.int32(token_in)
    token_out = jnp.int32(token_out)
    reserves = safe_positive(reserves)
    weights = safe_positive(weights)
    numerator = reserves[token_out] * weights[token_in]
    denominator = reserves[token_in] * weights[token_out]
    return numerator / jnp.maximum(denominator, _EPS)


def pair_oracle_price(oracle_prices, token_in, token_out):
    token_in = jnp.int32(token_in)
    token_out = jnp.int32(token_out)
    oracle_prices = jnp.asarray(oracle_prices)
    return oracle_prices[token_in] / jnp.maximum(oracle_prices[token_out], _EPS)


def pair_deviation(reserves, weights, oracle_prices, token_in, token_out):
    pool_price = pair_pool_price(reserves, weights, token_in, token_out)
    oracle_price = pair_oracle_price(oracle_prices, token_in, token_out)
    ratio = pool_price / jnp.maximum(oracle_price, _EPS)
    ratio = jnp.where(jnp.isfinite(ratio), ratio, 1.0)
    return jnp.abs(ratio - 1.0)


def max_pair_deviation(reserves, weights, oracle_prices):
    reserves = safe_positive(reserves)
    weights = safe_positive(weights)
    oracle_prices = jnp.asarray(oracle_prices)

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


def broadcast_scan_vector(values, scan_len):
    values = jnp.asarray(values)
    if values.ndim == 0:
        values = values.reshape((1,))
    values = jnp.ravel(values)
    return jnp.where(values.size == 1, jnp.full((scan_len,), values[0]), values)
