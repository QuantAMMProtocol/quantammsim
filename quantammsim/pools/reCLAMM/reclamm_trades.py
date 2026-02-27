"""Trade execution for reClAMM pools.

Thin wrappers around G3M constant-product trade functions, operating on
effective reserves (real + virtual) with clamp-to-edge semantics: when a
trade would push a real reserve below zero, output is clamped to the
real balance of the output token.

reClAMM is a 2-token equal-weight constant-product AMM on effective
reserves E_i = R_i + V_i, so all G3M calls use weights = [0.5, 0.5].
"""

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit

from quantammsim.pools.G3M.G3M_trades import (
    _jax_calc_G3M_trade_from_exact_out_given_in,
    _jax_calc_G3M_trade_from_exact_in_given_out,
)

_WEIGHTS = jnp.array([0.5, 0.5])


@jit
def reclamm_out_given_in(Ra, Rb, Va, Vb, token_in, token_out, amount_in, gamma=1.0):
    """Compute swap output for a given input, with clamp-to-edge.

    Wraps the G3M trade function on effective reserves with equal weights.
    Output is clamped to the real balance of the output token.

    Returns
    -------
    amount_out : scalar
    """
    effective = jnp.array([Ra + Va, Rb + Vb])
    trade = _jax_calc_G3M_trade_from_exact_out_given_in(
        effective, _WEIGHTS, token_in, token_out, amount_in, gamma,
    )
    amount_out = -trade[token_out]
    max_out = jnp.array([Ra, Rb])[token_out]
    return jnp.minimum(amount_out, max_out)


@jit
def reclamm_in_given_out(Ra, Rb, Va, Vb, token_in, token_out, amount_out, gamma=1.0):
    """Compute required input for desired output, with clamp-to-edge.

    Output is clamped to the real balance of the output token; the
    returned ``amount_in`` corresponds to the (possibly clamped) output.

    Returns
    -------
    amount_in : scalar
    amount_out_actual : scalar
    """
    max_out = jnp.array([Ra, Rb])[token_out]
    amount_out_actual = jnp.minimum(amount_out, max_out)

    effective = jnp.array([Ra + Va, Rb + Vb])
    trade = _jax_calc_G3M_trade_from_exact_in_given_out(
        effective, _WEIGHTS, token_in, token_out, amount_out_actual, gamma,
    )
    amount_in = trade[token_in]
    return amount_in, amount_out_actual
