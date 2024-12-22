from jax import config, jit
from jax.lax import scan
from jax.tree_util import Partial
import jax.numpy as jnp
import numpy as np

from functools import partial

config.update("jax_enable_x64", True)

np.seterr(all="raise")
np.seterr(under="print")


def calculate_A_matrix(c, s, lam):
    return jnp.array([[c / lam, -s / lam], [s, c]])


def calculate_A_matrix_inv(c, s, lam):
    return jnp.array([[c * lam, s], [-s * lam, c]])


def calculate_tau(in_value, A_matrix):
    # This implements the definition of Tau in Definition 3 of the E-CLP paper cf.
    # the Defn. 2 and 1.

    ap = A_matrix @ jnp.array([-1, in_value])

    zeta = (jnp.array([0, 1]) @ ap) / (-jnp.array([1, 0]) @ ap)
    # tau is eta(zeta) from Defn. 3
    tau = 1 / jnp.sqrt(1 + zeta**2) * jnp.array([zeta, 1])
    return tau


def calculate_chi(A_matrix, A_matrix_inv, alpha, beta):
    # This implements the definition of Chi in Proposition 8 of the E-CLP paper ("Initialisation
    # from real reserves").
    tau_beta = calculate_tau(beta, A_matrix)
    tau_alpha = calculate_tau(alpha, A_matrix)

    chi = jnp.array([(A_matrix_inv @ tau_beta)[0], (A_matrix_inv @ tau_alpha)[1]])
    return chi


def _jax_calc_gyroscope_invariant(reserves, alpha, beta, A_matrix, A_matrix_inv):
    # This implements the expression in Proposition 8 of the E-CLP paper ("Initialisation
    # from real reserves"), taking the positive root of the quadratic equation.
    t = reserves

    chi = calculate_chi(A_matrix, A_matrix_inv, alpha, beta)
    A_chi = A_matrix @ chi
    A_t = A_matrix @ t

    A_chi_norm_sq = jnp.dot(A_chi, A_chi)
    A_t_norm_sq = jnp.dot(A_t, A_t)

    A_t_dot_A_chi = jnp.dot(A_t, A_chi)

    invariant = (
        A_t_dot_A_chi
        + jnp.sqrt(A_t_dot_A_chi**2.0 - (A_chi_norm_sq - 1.0) * A_t_norm_sq)
    ) / (A_chi_norm_sq - 1.0)
    return invariant


def _jax_calc_gyroscope_inner_price(
    reserves, alpha, beta, A_matrix, A_matrix_inv, invariant
):
    # This implements the expression for p_x^g(t) in Proposition 12 of the E-CLP paper.abs

    tau_beta = calculate_tau(beta, A_matrix)
    tau_alpha = calculate_tau(alpha, A_matrix)

    # First calculate the transformed (shifted) reserves t'=(x',y')=(x-a,y-b)

    a = invariant * (A_matrix_inv @ tau_beta)[0]
    b = invariant * (A_matrix_inv @ tau_alpha)[1]

    x_prime = reserves[0] - a
    y_prime = reserves[1] - b

    t_prime = jnp.array([x_prime, y_prime])

    # Second calculate the transformed (linear operator) reserves t''=(x'',y'')=A (x',y')

    t_double_prime = A_matrix @ t_prime

    # Third, calculate the vector p^c(t'') = (x''/y'', 1)

    p_c_of_t_double_prime = jnp.array([t_double_prime[0] / t_double_prime[1], 1.0])

    # Finally compute price p_x^g(t) using prop 12
    A_ex = A_matrix @ jnp.array([1.0, 0.0])
    A_ey = A_matrix @ jnp.array([0.0, 1.0])
    price = jnp.dot(p_c_of_t_double_prime, A_ex) / jnp.dot(p_c_of_t_double_prime, A_ey)
    return price


def _jax_calc_gyroscope_reserves_zero_fees_scan_function(
    carry_list, prices, alpha, beta, A_matrix, A_matrix_inv
):
    # We perform the calculation in Appendix A of the E-CLP paper for
    # a single timestep, carrying forward the previous timestep's reserves.
    prev_reserves = carry_list[0]
    current_prices = prices[0] / prices[1]  # use scalar prices

    invariant = _jax_calc_gyroscope_invariant(
        prev_reserves, alpha, beta, A_matrix, A_matrix_inv
    )

    chi = calculate_chi(A_matrix, A_matrix_inv, alpha, beta)

    tau_of_prices = calculate_tau(current_prices, A_matrix)
    overall_trade = invariant * (chi - A_matrix_inv @ tau_of_prices) - prev_reserves

    reserves = prev_reserves + overall_trade
    return [reserves], reserves


@jit
def _jax_calc_gyroscope_reserves_zero_fees(
    initial_reserves,
    alpha,
    beta,
    sin,
    cos,
    lam,
    prices):
    """
    Calculate AMM reserves for an ECLP pool

    This function computes the changes in reserves for an automated market maker (AMM) model,
    following Appendix A of "The Elliptic Concentrated Liquidity Pool (E-CLP)".
    It uses a scan operation to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    alpha : float
        Lower bound of price range.
    beta : float
        Upper bound of price range.
    sin : float
        Sine of rotation angle phi.
    cos : float
        Cosine of rotation angle phi.
    lam : float
        Lambda parameter controlling ellipse shape.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.

    Returns
    -------
    jnp.ndarray
        The reserves array, showing how reserves change over time.
    """

    initial_prices = prices[0]

    # pre-calculate some values that are repeatedly used in optimal arb calculations
    A_matrix = calculate_A_matrix(cos, sin, lam)
    A_matrix_inv = calculate_A_matrix_inv(cos, sin, lam)

    scan_fn = Partial(
        _jax_calc_gyroscope_reserves_zero_fees_scan_function,
        alpha=alpha,
        beta=beta,
        A_matrix=A_matrix,
        A_matrix_inv=A_matrix_inv,
    )

    carry_list_init = [
        initial_reserves,
    ]

    carry_list_end, reserves = scan(
        scan_fn,
        carry_list_init,
        prices,
    )

    return reserves


def _jax_calc_gyroscope_reserves_with_fees_scan_function(
    carry_list,
    prices,
    alpha,
    beta,
    A_matrix,
    A_matrix_inv,
    gamma,
    arb_fees=0.0,
    arb_thresh=0.0,
):
    # We perform the calculation in Appendix A of the E-CLP paper for
    # a single timestep, carrying forward the previous timestep's reserves.
    prev_reserves = carry_list[0]
    invariant = _jax_calc_gyroscope_invariant(
        prev_reserves, alpha, beta, A_matrix, A_matrix_inv
    )
    chi = calculate_chi(A_matrix, A_matrix_inv, alpha, beta)

    # find quoted 'inner' price

    current_price = _jax_calc_gyroscope_inner_price(
        prev_reserves, alpha, beta, A_matrix, A_matrix_inv, invariant
    )

    current_price = prev_reserves[1] / prev_reserves[0]

    # envelope of no arb region
    ### see if prices are out of envelope

    bid_price = ask_price = prices[0] / prices[1]
    # now handle fee logic (Remark 5 in Appendix A)
    effective_price_for_price_below_bid = bid_price * gamma
    effective_price_for_price_above_ask = ask_price / gamma
    current_price_below_bid = current_price <= effective_price_for_price_below_bid
    current_price_above_ask = effective_price_for_price_above_ask <= current_price

    effective_price = jnp.where(
        current_price_below_bid,
        effective_price_for_price_below_bid,
        current_price,
    )
    effective_price = jnp.where(
        current_price_above_ask,
        effective_price_for_price_above_ask,
        effective_price,
    )
    # now construct new values of reserves, profit, for IF
    # we are out of region -- and then use jnp where function to 'paste in'
    # values if outside_no_arb_region is True and trade is profitable to arb
    # We have to handle two cases here, for if prices are above or below the
    # no-arb region
    tau_of_prices = calculate_tau(effective_price, A_matrix)
    overall_trade_if_arbed = (
        invariant * (chi - A_matrix_inv @ tau_of_prices) - prev_reserves
    )

    # if we are outside arb regio, no trade is done so overall trade is zeros
    overall_trade = jnp.array([0.0, 0.0])
    overall_trade_if_current_price_below_bid = overall_trade_if_arbed * jnp.array(
        [1.0, 1.0 / gamma]
    )
    overall_trade_if_current_price_above_ask = overall_trade_if_arbed * jnp.array(
        [1.0 / gamma, 1.0]
    )
    overall_trade = jnp.where(
        current_price_below_bid, overall_trade_if_current_price_below_bid, overall_trade
    )
    overall_trade = jnp.where(
        current_price_above_ask, overall_trade_if_current_price_above_ask, overall_trade
    )

    # only perform trade if trade would be profitable to arb

    # check if this is worth the cost to arbs
    # is this trade a good deal for the arb?
    profit_to_arb = -(overall_trade * prices).sum() - arb_thresh
    profit_prices = profit_to_arb

    arb_external_rebalance_cost = (
        0.5 * arb_fees * (jnp.abs(overall_trade) * prices).sum()
    )

    arb_profitable = profit_to_arb >= arb_external_rebalance_cost

    # if arb trade IS profitable
    # then reserves is equal to post_price_reserves, otherwise equal to prev_reserves
    do_arb_trade = arb_profitable

    reserves = jnp.where(do_arb_trade, prev_reserves + overall_trade, prev_reserves)

    return [reserves], reserves


@jit
def _jax_calc_gyroscope_reserves_with_fees(
    initial_reserves,
    alpha,
    beta,
    sin,
    cos,
    lam,
    prices,
    fees=0.0,
    arb_thresh=0.0,
    arb_fees=0.0,
):
    """
    Calculate AMM reserves for an ECLP pool

    This function computes the changes in reserves for an automated market maker (AMM) model,
    following Appendix A of "The Elliptic Concentrated Liquidity Pool (E-CLP)".
    It uses a scan operation to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    alpha : float
        Lower bound of price range.
    beta : float
        Upper bound of price range.
    sin : float
        Sine of rotation angle phi.
    cos : float
        Cosine of rotation angle phi.
    lam : float
        Lambda parameter controlling ellipse shape.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    fees : float, optional
        Swap fee charged on transactions, by default 0.003.
    arb_thresh : float, optional
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : float, optional
        Fees associated with arbitrage, by default 0.0.

    Returns
    -------
    jnp.ndarray
        The reserves array, showing how reserves change over time.
    """

    initial_prices = prices[0]
    gamma = 1.0 - fees
    # pre-calculate some values that are repeatedly used in optimal arb calculations
    A_matrix = calculate_A_matrix(cos, sin, lam)
    A_matrix_inv = calculate_A_matrix_inv(cos, sin, lam)

    scan_fn = Partial(
        _jax_calc_gyroscope_reserves_with_fees_scan_function,
        alpha=alpha,
        beta=beta,
        A_matrix=A_matrix,
        A_matrix_inv=A_matrix_inv,
        gamma=gamma,
        arb_thresh=arb_thresh,
        arb_fees=arb_fees,
    )

    carry_list_init = [
        initial_reserves,
    ]

    carry_list_end, reserves = scan(
        scan_fn,
        carry_list_init,
        prices,
    )

    return reserves


# @jit
def _jax_calc_ECLP_trade_from_exact_out_given_in(
    reserves,
    token_in,
    token_out,
    amount_in,
    alpha,
    beta,
    A_matrix,
    A_matrix_inv,
    lam,
    s,
    c,
    gamma=0.997,
):
    """
    Calculate the post-trade reserves from a trade of exact amount
    of token_out given an input amount of token_in for an ECLP pool.
    Uses Proposition 14 in "The Elliptic Concentrated Liquidity Pool (E-CLP)".

    Parameters:
    -----------
    reserves : jnp.ndarray
        Current reserves of all tokens in the AMM.
    token_in : int
        Index of the input token.
    token_out : int
        Index of the output token.
    amount_in : float
        Amount of token_in to be swapped.
    alpha : float
        Lower bound of price range.
    beta : float
        Upper bound of price range.
    A_matrix : jnp.ndarray
        Rotation matrix.
    A_matrix_inv : jnp.ndarray
        Inverse rotation matrix.
    lam : float
        Lambda parameter controlling ellipse shape.
    s : float
        Sine of rotation angle phi.
    c : float
        Cosine of rotation angle phi.
    gamma : float, optional
        Fee parameter, where (1 - gamma) is the fee percentage. Default is 0.997.

    Returns:
    --------
    jnp.ndarray
        The reserves post trade.
    """
    # Calculate the ratio of weights
    token_in = jnp.int32(token_in)
    token_out = jnp.int32(token_out)

    # some functions of parameters we need to have

    tau_beta = calculate_tau(beta, A_matrix)
    tau_alpha = calculate_tau(alpha, A_matrix)

    lam_underscore = 1.0 - 1.0 / (lam**2.0)

    # we need the invariant too

    invariant = _jax_calc_gyroscope_invariant(
        reserves, alpha, beta, A_matrix, A_matrix_inv
    )

    # Now calculate the transformed (shifted) reserves t'=(x',y')=(x-a,y-b)

    a = invariant * (A_matrix_inv @ tau_beta)[0]
    b = invariant * (A_matrix_inv @ tau_alpha)[1]

    translation_vector = jnp.array([a, b])
    x_prime = reserves[0] - a
    y_prime = reserves[1] - b

    t_prime = jnp.array([x_prime, y_prime])

    post_trade_in_reserves_with_fees_applied = t_prime[token_in] + amount_in * gamma
    s_c_lam_unsc_post_t_in_r = (
        s * c * lam_underscore * post_trade_in_reserves_with_fees_applied
    )
    one_minus_lam_underscore_times_trig_squared = jnp.array(
        [1 - lam_underscore * s**2.0, 1 - lam_underscore * c**2.0]
    )
    post_trade_t_prime_for_token_out = (
        -s_c_lam_unsc_post_t_in_r
        - jnp.sqrt(
            s_c_lam_unsc_post_t_in_r * 2.0
            - (1 - lam_underscore * s**2.0)
            * (
                (1 - lam_underscore * c**2.0) * post_trade_in_reserves_with_fees_applied
                - invariant**2.0
            )
        )
    ) / one_minus_lam_underscore_times_trig_squared[token_in]
    amount_out = post_trade_in_reserves_with_fees_applied - t_prime[token_out]
    overall_trade = jnp.zeros(2)
    overall_trade = overall_trade.at[token_in].set(amount_in)
    overall_trade = overall_trade.at[token_out].set(-amount_out)
    return jnp.where(amount_in != 0, overall_trade, 0)


# version of _jax_calc_ECLP_trade_from_exact_out_given_in that
# in 'trade' as one single input. Useful for lazy evaluation
def wrapped_ECLP_trade_function(
    reserves, trade, alpha, beta, A_matrix, A_matrix_inv, lam, s, c, gamma
):
    return _jax_calc_ECLP_trade_from_exact_out_given_in(
        reserves,
        trade[0],
        trade[1],
        trade[2],
        alpha,
        beta,
        A_matrix,
        A_matrix_inv,
        lam,
        s,
        c,
        gamma,
    )


def zero_trade_function_ECLP(reserves, trade, gamma):
    return jnp.zeros(reserves.shape)


# Create a jitted function that includes the cond, for lazy evaluation
@jit
def jitted_ECLP_cond_trade(
    condition, reserves, trade, alpha, beta, A_matrix, A_matrix_inv, lam, s, c, gamma
):
    return jax.lax.cond(
        condition,
        wrapped_ECLP_trade_function,
        zero_trade_function_ECLP,
        reserves,
        trade,
        alpha,
        beta,
        A_matrix,
        A_matrix_inv,
        lam,
        s,
        c,
        gamma,
    )


@partial(jit, static_argnums=(10))
def _jax_calc_gyroscopr_reserves_with_dynamic_fees_and_trades_scan_function_using_precalcs(
    carry_list,
    input_list,
    alpha,
    beta,
    A_matrix,
    A_matrix_inv,
    lam,
    s,
    c,
    gamma,
    do_trades,
):
    """
    Calculate changes in AMM reserves considering fees and arbitrage opportunities.

    This function extends the basic reserve calculation by incorporating transaction fees
    and potential arbitrage opportunities.

    Parameters
    ----------
    carry_list : list
        List containing the previous weights, prices, and reserves.
    input_list : list
        List containing:
        prices : jnp.ndarray
            Array containing the current prices.
        gamma: jnp.ndarray
            Array containing the AMM pool's 1-fees over time.
        arb_thresh: jnp.ndarray
            Array containing the arb's threshold for profitable arbitrage over time.
        arb_fees: jnp.ndarray
            Array containing the fees associated with arbitrage.
        trades: jnp.ndarray
            Array containing the indexs of the in and out tokens and the in amount for trades at each time.
    alpha : float
        The alpha parameter of the ECLP curve.
    beta : float
        The beta parameter of the ECLP curve.
    A_matrix : jnp.ndarray
        The A matrix used in ECLP calculations.
    A_matrix_inv : jnp.ndarray
        The inverse of the A matrix.
    lam : float
        The lambda parameter controlling curve shape.
    s : float
        Sine of the rotation angle
    c : float
        Cosine of the rotation angle
    gamma : float
        The gamma parameter is 1 minus fees.
    do_trades : bool
        Whether to execute trades or not.

    Returns
    -------
    list
        Updated list containing the new weights, prices, and reserves.
    jnp.ndarray
        Array of reserves changes.
    """

    # carry_list[0] is previous prices
    prev_prices = carry_list[0]

    # carry_list[1] is previous reserves
    prev_reserves = carry_list[1]

    counter = carry_list[2]

    # input_list contains weights, prices, precalcs and fee/arb amounts
    prices = input_list[0]
    gamma = input_list[1]
    arb_thresh = input_list[2]
    arb_fees = input_list[3]
    trade = input_list[4]

    fees_are_being_charged = gamma != 1.0

    current_value = (prev_reserves * prices).sum()

    _, reserves = _jax_calc_gyroscope_reserves_with_fees_scan_function(
        [prev_reserves],
        prices,
        alpha,
        beta,
        A_matrix,
        A_matrix_inv,
        gamma,
        arb_fees=arb_fees,
        arb_thresh=arb_thresh,
    )

    # apply trade if trade is present
    if do_trades:
        reserves += jitted_ECLP_cond_trade(
            do_trades,
            reserves,
            trade,
            alpha,
            beta,
            A_matrix,
            A_matrix_inv,
            lam,
            s,
            c,
            gamma,
        )

    counter += 1
    return [
        prices,
        reserves,
        counter,
    ], reserves


@partial(jit, static_argnums=(8,))
def _jax_calc_gyroscope_reserves_with_dynamic_inputs(
    initial_reserves,
    alpha,
    beta,
    sin,
    cos,
    lam,
    prices,
    fees,
    arb_thresh,
    arb_fees,
    trades=None,
    do_trades=False,
):
    """
    Calculate AMM reserves considering fees and arbitrage opportunities using signature variations,
    using the approach given in https://arxiv.org/abs/2402.06731.

    This function computes the changes in reserves for an automated market maker (AMM) model
    considering dynamic transaction fees, dynamic arbitrage costs, external trades and
    potential arbitrage opportunities.
    It uses a scan operation to apply these calculations over multiple timesteps.

    Parameters
    ----------
    initial_reserves : jnp.ndarray
        Initial reserves at the start of the calculation.
    alpha : float
        Lower bound of price range.
    beta : float
        Upper bound of price range.
    sin : float
        Sine of rotation angle phi.
    cos : float
        Cosine of rotation angle phi.
    lam : float
        Lambda parameter controlling ellipse shape.
    prices : jnp.ndarray
        Two-dimensional array of asset prices over time.
    fees : float / jnp.ndarray
        Swap fee charged on transactions, by default 0.003.
    arb_thresh : jnp.ndarray
        Threshold for profitable arbitrage, by default 0.0.
    arb_fees : jnp.ndarray
        Fees associated with arbitrage, by default 0.0.
    trades :  jnp.ndarray, optional
        array of trades for each timestep.
        format, for each row:
            trades[0] = index of token to trade in to pool
            trades[1] = index of token to trade out to pool
            trades[2] = amount of 'token in' to trade
    do_trades : bool, optional
        Whether or not to apply the trades, by default False

    Returns
    -------
    jnp.ndarray
        The reserves array, indicating the changes in reserves over time.
    """
    n_assets = weights.shape[0]

    n = prices.shape[0]

    initial_prices = prices[0]

    # pre-calculate some values that are repeatedly used in optimal arb calculations
    A_matrix = calculate_A_matrix(cos, sin, lam)
    A_matrix_inv = calculate_A_matrix_inv(cos, sin, lam)

    gamma = jnp.where(fees.size == 1, jnp.full(prices.shape[0], 1.0 - fees), 1.0 - fees)

    arb_thresh = jnp.where(
        arb_thresh.size == 1, jnp.full(prices.shape[0], arb_thresh), arb_thresh
    )

    arb_fees = jnp.where(
        arb_fees.size == 1, jnp.full(prices.shape[0], arb_fees), arb_fees
    )

    scan_fn = Partial(
        _jax_calc_gyroscopr_reserves_with_dynamic_fees_and_trades_scan_function_using_precalcs,
        alpha=alpha,
        beta=beta,
        A_matrix=A_matrix,
        A_matrix_inv=A_matrix_inv,
        lam=lam,
        s=sin,
        c=cos,
        do_trades=do_trades,
    )

    carry_list_init = [
        initial_prices,
        initial_reserves,
    ]
    carry_list_end, reserves = scan(
        scan_fn,
        carry_list_init,
        [
            prices,
            gamma,
            arb_thresh,
            arb_fees,
            trades,
        ],
    )

    return reserves


if __name__ == "__main__":
    # # Test tau calculation
    c = 1 / np.sqrt(2)
    s = 1 / np.sqrt(2)
    lam = 2
    p = 1
    A_matrix = np.array([[c / lam, -s / lam], [s, c]])
    test_tau = calculate_tau(p, A_matrix)
    np.testing.assert_array_almost_equal(test_tau, np.array([0, 1]))

    # # Test chi calculation
    alpha = 1 / 2
    beta = 2
    A_matrix_inv = np.array([[c * lam, s], [-s * lam, c]])
    test_chi = calculate_chi(A_matrix, A_matrix_inv, alpha, beta)
    expected_chi = np.array([1.37281294596729, 1.37281294596729])
    np.testing.assert_array_almost_equal(test_chi, expected_chi)

    alpha = 0.5
    beta = 4.0

    phi = np.pi / 4.0
    sin = np.sin(phi)
    cos = np.cos(phi)
    lam = 2.0
    prices = jnp.array([[1.0, 2.0, 3.0, 2.0], [1.0, 1.0, 1.0, 1.0]]).T
    initial_reserves = jnp.array([1.0, 1.0])
    test_reserves = _jax_calc_gyroscope_reserves_using_precalcs(
        initial_reserves, alpha, beta, sin, cos, lam, prices
    )
    print("test_reserves", test_reserves)
    for gamma in [1.0, 0.99, 0.9]:
        test_reserves = _jax_calc_gyroscope_reserves_using_precalcs(
            initial_reserves, alpha, beta, sin, cos, lam, prices, gamma
        )
        print("gamma", gamma)
        print("test_reserves", test_reserves)
    print("-------------------")
    for gamma in [1.0, 0.99, 0.9]:
        test_reserves = _jax_calc_gyroscope_reserves_using_precalcs(
            initial_reserves, alpha, beta, sin, cos, lam, prices, gamma, arb_thresh=0.1
        )
        print("gamma", gamma)
        print("test_reserves", test_reserves)

        # Test trade execution
    print("\nTesting trades...")
    initial_reserves = jnp.array([100.0, 100.0])
    prices = jnp.array([[1.0, 1.0], [1.0, 1.0]]).T  # Single timestep with equal prices

    # Test parameters
    trade = jnp.array([0, 1, 10.0])  # Trade 10 units from token 0 to token 1
    do_trades = True
    gamma = 0.997  # 0.3% fee

    print("initial_reserves")
    print(initial_reserves)
    test_reserves = _jax_calc_gyroscope_reserves_using_precalcs(
        initial_reserves, alpha, beta, sin, cos, lam, prices, gamma=1.0, arb_thresh=0.0
    )
    print("test_reserves")
    print(test_reserves)
    # Execute trade
    overall_trade = _jax_calc_ECLP_trade_from_exact_out_given_in(
        test_reserves[0],
        trade[0],
        trade[1],
        trade[2],
        alpha,
        beta,
        A_matrix,
        A_matrix_inv,
        lam,
        s,
        c,
        gamma=0.997,
    )

    print(f"Initial reserves: {test_reserves}")
    print(
        f"Trade: {trade[2]} units from token {int(trade[0])} to token {int(trade[1])}"
    )
    print(f"Final reserves: {test_reserves + overall_trade}")
    print(f"Reserve changes: {overall_trade }")
