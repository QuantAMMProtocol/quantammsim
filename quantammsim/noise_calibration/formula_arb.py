"""JAX-differentiable LVR formula for arb volume.

Based on arXiv:2305.14604v2 §6 with gas costs and discrete-time correction.
Reference: scripts/plot_formula_arb_vs_real.py:formula_arb_volume_daily (line 58).
"""

import jax.numpy as jnp


def formula_arb_volume_daily_jax(sigma_daily, tvl, fee, gas_usd, cadence_minutes):
    """Analytical arb volume per day for a CPMM with gas costs.

    All inputs are JAX scalars or arrays (must be broadcastable).

    Parameters
    ----------
    sigma_daily : float
        Daily volatility of the log price ratio (NOT annualised).
    tvl : float
        Pool TVL in USD.
    fee : float
        Swap fee as fraction (e.g. 0.003 for 30bp).
    gas_usd : float
        All-in gas cost per arb tx in USD.
    cadence_minutes : float
        Effective arb cadence in minutes (= simulator's arb_frequency).
    """
    block_time_s = cadence_minutes * 60.0
    delta = 2.0 * jnp.sqrt(2.0 * jnp.maximum(gas_usd, 0.0) / jnp.maximum(tvl, 1e-6))
    bLVR = sigma_daily**2 * tvl / 8.0
    sqrt_term = sigma_daily * jnp.sqrt(block_time_s / (2.0 * 86400.0))
    correction = jnp.maximum(
        1.0 - delta / (2.0 * fee) - sqrt_term / (fee + delta / 2.0),
        0.0,
    )
    return bLVR * correction / fee
