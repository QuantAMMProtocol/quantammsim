# again, this only works on startup!
from jax import jit, devices

import jax.numpy as jnp
from jax import default_backend

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


@jit
def reclamm_tsoukalas_sqrt_noise_volume(
    effective_value_usd,
    gamma,
    volatility,
    arb_volume_this_period,
    noise_params=None,
):
    """reClAMM Tsoukalas sqrt model: effective TVL regressor.

    Predicts per-minute noise trader volume using:
        V_daily = (a_0 - a_f*fee + a_sigma*sigma
                   + a_c*sqrt(c_eff/1e6)) * 1e6
        V_noise = max(0, V_daily/1440 - arb_volume_this_period)

    where c_eff = (Ra+Va)*pA + (Rb+Vb)*pB is the effective TVL (real +
    virtual reserves valued in USD). For a concentrated liquidity pool,
    effective reserves determine execution quality and routing decisions,
    so they are the natural driver of noise volume.

    Parameters
    ----------
    effective_value_usd : float
        Effective TVL in USD: (Ra+Va)*pA + (Rb+Vb)*pB.
    gamma : float
        Fee parameter (1 - fee_rate).
    volatility : float
        Annualised daily realised volatility of the price ratio.
    arb_volume_this_period : float
        Arb volume already accounted for this time step (USD).
    noise_params : dict, optional
        Regression coefficients. Keys: a_0_base, a_f, a_sigma,
        a_c, base_fee.

    Returns
    -------
    float
        Per-minute noise volume (USD), floored at zero.
    """
    if noise_params is None:
        noise_params = {}
    a_0_base = noise_params.get("a_0_base", 0.5)
    a_f = noise_params.get("a_f", 0.0)
    a_sigma = noise_params.get("a_sigma", 2.0)
    a_c = noise_params.get("a_c", 1.0)
    base_fee = noise_params.get("base_fee", 0.003)

    fee = 1.0 - gamma
    a_0 = a_0_base + base_fee * a_f
    daily_vol = (
        a_0 - a_f * fee
        + a_sigma * volatility
        + a_c * jnp.sqrt(effective_value_usd / 1e6)
    ) * 1e6
    return jnp.maximum(0.0, daily_vol / 1440.0 - arb_volume_this_period)


@jit
def reclamm_tsoukalas_log_noise_volume(
    effective_value_usd,
    gamma,
    volatility,
    arb_volume_this_period,
    noise_params=None,
):
    """reClAMM Tsoukalas log model: log(c_eff/1e6) instead of sqrt.

    Same specification as the sqrt variant but uses log regressor,
    which may fit better for pools spanning a wide TVL range.

    Parameters
    ----------
    effective_value_usd : float
        Effective TVL in USD: (Ra+Va)*pA + (Rb+Vb)*pB.
    gamma : float
        Fee parameter (1 - fee_rate).
    volatility : float
        Annualised daily realised volatility of the price ratio.
    arb_volume_this_period : float
        Arb volume already accounted for this time step (USD).
    noise_params : dict, optional
        Regression coefficients (same keys as sqrt variant).

    Returns
    -------
    float
        Per-minute noise volume (USD), floored at zero.
    """
    if noise_params is None:
        noise_params = {}
    a_0_base = noise_params.get("a_0_base", 0.5)
    a_f = noise_params.get("a_f", 0.0)
    a_sigma = noise_params.get("a_sigma", 2.0)
    a_c = noise_params.get("a_c", 1.0)
    base_fee = noise_params.get("base_fee", 0.003)

    fee = 1.0 - gamma
    a_0 = a_0_base + base_fee * a_f
    daily_vol = (
        a_0 - a_f * fee
        + a_sigma * volatility
        + a_c * jnp.log(jnp.maximum(effective_value_usd / 1e6, 1e-30))
    ) * 1e6
    return jnp.maximum(0.0, daily_vol / 1440.0 - arb_volume_this_period)


@jit
def reclamm_loglinear_noise_volume(
    effective_value_usd,
    gamma,
    volatility,
    arb_volume_this_period,
    noise_params=None,
):
    """Loglinear noise volume from hierarchical cross-pool calibration.

    Predicts per-minute noise volume using:
        log(V_daily) = b_0 + b_sigma * volatility + b_c * log(TVL)
        V_noise = max(0, exp(log_daily_vol) / 1440 - arb_volume)

    where b_0 is a pool-specific intercept (BLUP from the hierarchical
    model, absorbing chain, token tier, and fee effects), and b_sigma,
    b_c are shared fixed effects estimated from cross-pool variation.

    Note: ``gamma`` is accepted for interface compatibility with the
    other noise volume functions but is not used; fee effects are
    absorbed into ``b_0`` via the hierarchical model's BLUP.

    Parameters
    ----------
    effective_value_usd : float
        Effective TVL in USD: (Ra+Va)*pA + (Rb+Vb)*pB.
    gamma : float
        Fee parameter (1 - fee_rate).  Unused — kept for uniform
        calling convention across noise models.
    volatility : float
        Annualised daily realised volatility of the price ratio.
    arb_volume_this_period : float
        Arb volume already accounted for this time step (USD).
    noise_params : dict, optional
        Hierarchical model coefficients. Keys: b_0, b_sigma, b_c.

    Returns
    -------
    float
        Per-minute noise volume (USD), floored at zero.
    """
    if noise_params is None:
        noise_params = {}
    b_0 = noise_params.get("b_0", -6.7)
    b_sigma = noise_params.get("b_sigma", -0.0007)
    b_c = noise_params.get("b_c", 1.04)

    log_daily_vol = (
        b_0
        + b_sigma * volatility
        + b_c * jnp.log(jnp.maximum(effective_value_usd, 1.0))
    )
    daily_vol = jnp.exp(log_daily_vol)
    return jnp.maximum(0.0, daily_vol / 1440.0 - arb_volume_this_period)


@jit
def reclamm_calibrated_noise_volume(
    effective_value_usd,
    gamma,
    volatility,
    arb_volume_this_period,
    dow_sin,
    dow_cos,
    noise_params=None,
):
    """8-covariate calibrated noise volume from cross-pool log-linear model.

    Predicts per-minute noise volume using::

        log(V_daily) = c_0 + c_1*log(TVL) + c_2*log(sigma)
                        + c_3*log(TVL)*log(sigma) + c_4*log(TVL)*fee
                        + c_5*log(sigma)*fee + c_6*dow_sin + c_7*dow_cos
        V_noise = max(0, exp(log_daily_vol) / 1440 - arb_volume)

    where sigma is annualised daily realised volatility, fee = 1 - gamma,
    and dow_sin/dow_cos encode day-of-week seasonality.

    Parameters
    ----------
    effective_value_usd : float
        Effective TVL in USD: (Ra+Va)*pA + (Rb+Vb)*pB.
    gamma : float
        Fee parameter (1 - fee_rate).
    volatility : float
        Annualised daily realised volatility of the price ratio.
    arb_volume_this_period : float
        Arb volume already accounted for this time step (USD).
    dow_sin : float
        sin(2*pi*weekday/7) for the current day.
    dow_cos : float
        cos(2*pi*weekday/7) for the current day.
    noise_params : dict, optional
        Calibrated coefficients: c_0 .. c_7.

    Returns
    -------
    float
        Per-minute noise volume (USD), floored at zero.
    """
    if noise_params is None:
        noise_params = {}
    c_0 = noise_params.get("c_0", 0.0)
    c_1 = noise_params.get("c_1", 1.0)
    c_2 = noise_params.get("c_2", 0.0)
    c_3 = noise_params.get("c_3", 0.0)
    c_4 = noise_params.get("c_4", 0.0)
    c_5 = noise_params.get("c_5", 0.0)
    c_6 = noise_params.get("c_6", 0.0)
    c_7 = noise_params.get("c_7", 0.0)

    fee = 1.0 - gamma
    log_tvl = jnp.log(jnp.maximum(effective_value_usd, 1.0))
    log_sigma = jnp.log(jnp.maximum(volatility, 1e-10))

    log_daily_vol = (
        c_0
        + c_1 * log_tvl
        + c_2 * log_sigma
        + c_3 * log_tvl * log_sigma
        + c_4 * log_tvl * fee
        + c_5 * log_sigma * fee
        + c_6 * dow_sin
        + c_7 * dow_cos
    )
    daily_vol = jnp.exp(log_daily_vol)
    # noise_coeffs predict V_noise directly (not V_total), so no need to
    # subtract arb volume — that would double-count the arb subtraction.
    return jnp.maximum(0.0, daily_vol / 1440.0)


@jit
def reclamm_market_linear_noise_volume(
    effective_value_usd,
    noise_base,
    noise_tvl_coeff,
    tvl_mean=0.0,
    tvl_std=1.0,
):
    """Market-feature linear noise model with precomputed daily coefficients.

    The full model is::

        log(V_daily_noise) = base_t + tvl_coeff_t * standardized_log_tvl

    where ``base_t`` absorbs all non-TVL terms (intercept, market regime,
    token volatility, pair volatility, day-of-week, cross-pool volumes)
    and ``tvl_coeff_t`` is the effective TVL coefficient including
    interaction terms (tvl×btc_vol, tvl×tok_a_vol, tvl×pair_vol).

    The log(TVL) is standardized using the same mean/std from training
    to ensure the coefficient scale matches.

    Both base_t and tvl_coeff_t are precomputed daily from the per-pool
    calibrated noise model and passed in as dynamic input arrays.

    Under counterfactual (varying reClAMM concentration), only
    ``effective_value_usd`` changes — all market/peer features are held
    at observed values via the precomputed arrays.

    Parameters
    ----------
    effective_value_usd : float
        Effective TVL in USD: (Ra+Va)*pA + (Rb+Vb)*pB.
    noise_base : float
        Precomputed non-TVL component of log(V_daily_noise) for this step.
    noise_tvl_coeff : float
        Precomputed effective coefficient on log(TVL) for this step.
    tvl_mean : float
        Mean of log(TVL) from training data standardization.
    tvl_std : float
        Std of log(TVL) from training data standardization.

    Returns
    -------
    float
        Per-minute noise volume (USD), floored at zero.
    """
    log_tvl = jnp.log(jnp.maximum(effective_value_usd, 1.0))
    standardized_log_tvl = (log_tvl - tvl_mean) / tvl_std
    log_daily_noise = noise_base + noise_tvl_coeff * standardized_log_tvl
    daily_noise = jnp.exp(log_daily_noise)
    return jnp.maximum(0.0, daily_noise / 1440.0)


