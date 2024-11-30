# again, this only works on startup!
from jax import config, jit, devices

import jax.numpy as jnp
from jax.lib.xla_bridge import default_backend

config.update("jax_enable_x64", True)

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
else:
    GPU_DEVICE = devices("cpu")[0]


@jit
def calculate_reserves_after_noise_trade(
    arb_trade,
    current_reserves,
    prices,
    noise_trader_ratio,
    gamma,
):
    """Calculate pool reserves after accounting for noise trader 
    fee income following a simple model.

    Parameters
    ----------
    arb_trade : jnp.ndarray
        The arbitrage trade vector
    current_reserves : jnp.ndarray
        Current pool reserves
    prices : jnp.ndarray
        Asset prices
    noise_trader_ratio : float
        Ratio of noise trader to arbitrageur volume (e.g. 1.5 means noise traders contribute 
        60% of volume)
    gamma : float
        Fee parameter (1 - fee rate)

    Returns
    -------
    jnp.ndarray
        Updated pool reserves after accounting for noise trader fee income

    Notes
    -----
    This function estimates how pool reserves change after an arbitrage trade by modeling the
    additional fee income generated from noise traders. The model is based on academic research
    showing that noise traders contribute significantly to AMM trading volume. This approach allows
    us to model pools that have not yet been created. That is because in this approach, where
    noise trades are imputed as having an impact calculated for the arbitrage trade present, avoids
    us having to require real historic trade data for a pool, as is typically done when modelling
    the impact of noise trades. Freeing us from being able to model only pools that have existed
    in the past is of particular important for time-varying pools, for example TFMM pools
    where the weights change with time, as even if there is a CFMM pool with the same 
    constituents as the TFMM pool, one cannot expect that the noise trading behaviour will 
    be the same as for a CFMM pool (i.e. with static weights), say. 
    This approach also works for pools with more than two assets, and does not
    require the pool to have a particular trading function.

    The calculation follows these steps:
    1. Extract the inbound leg of the arbitrage trade
    2. Calculate fee income from noise trades as a proportion of trade value
    3. Scale up reserves uniformly to reflect the additional fee income

    The approach assumes noise traders take no directional view and trade in equal and opposite
    directions across assets, preserving relative prices.

    References
    ----------
    .. [1] "When does the tail wag the dog? Curvature and market making"
           Appendix B
           https://arxiv.org/pdf/2012.08040
    .. [2] "Arbitrageurs' profits, LVR, and sandwich attacks: 
            batch trading as an AMM design response"
           section 5.3
           https://arxiv.org/pdf/2307.02074
    """
    # First we extract the inbound leg of the arbitrage trade.
    # We model the fee income from the noise trades as a proportion of the value of the trade
    # following the results in Appendix B of the paper
    # "When does the tail wag the dog? Curvature and market making"
    # available @ https://arxiv.org/pdf/2012.08040.
    estimated_trade_in = jnp.where(
        arb_trade > 0,
        arb_trade,
        0.0,
    )
    # We scale the fee income from the arb trades up by the noise trader ratio.
    # In "Arbitrageurs' profits, LVR, and sandwich attacks:
    # batch trading as an AMM design response."
    # (https://arxiv.org/pdf/2307.02074) the authors describe in section 5.3
    # that the volume from noise traders
    # on uniswap v3 is about 60% of all volume compared to about 40%
    # of volume being from arbitrageurs.
    # this would correspond to a noise trader ratio of 0.6 / 0.4 = 1.5.
    estimated_arb_fee_income_from_inbound_trade = (
        noise_trader_ratio * (1.0 - gamma) * jnp.sum(estimated_trade_in * prices)
    )
    # We then add this fee income to the reserves, which we do by scaling up all reserves
    # which means a) that the quoted prices of the pool are not changed and b) that we are assuming
    # that noise traders are taking no directional view on the market,
    # but trading in equal and opposite directions across all assets.
    ratio_of_value_of_trade_to_reserves = (
        1.0
        + estimated_arb_fee_income_from_inbound_trade
        / jnp.sum(current_reserves * prices)
    )
    reserves = current_reserves * ratio_of_value_of_trade_to_reserves
    return reserves
