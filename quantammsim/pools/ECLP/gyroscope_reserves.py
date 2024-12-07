from jax import config, jit
from jax.lax import scan
from jax.tree_util import Partial
import jax.numpy as jnp
import numpy as np

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


def _jax_calc_gyroscope_reserves_with_fees_scan_function(
    carry_list, prices, alpha, beta, A_matrix, A_matrix_inv
):
    # We perform the calculation in Appendix A of the E-CLP paper for
    # a single timestep, carrying forward the previous timestep's reserves.
    prev_reserves = carry_list[0]
    current_prices = prices[0] / prices[1] # use scalar prices

    invariant = _jax_calc_gyroscope_invariant(
        prev_reserves, alpha, beta, A_matrix, A_matrix_inv
    )

    chi = calculate_chi(A_matrix, A_matrix_inv, alpha, beta)

    tau_of_prices = calculate_tau(current_prices, A_matrix)
    overall_trade = invariant * (chi - A_matrix_inv @ tau_of_prices) - prev_reserves

    reserves = prev_reserves + overall_trade
    return [reserves], reserves


@jit
def _jax_calc_gyroscope_reserves_using_precalcs(
    initial_reserves,
    alpha,
    beta,
    sin,
    cos,
    lam,
    prices,
):
    """
    Calculate AMM reserves for an ECLP pool

    This function computes the changes in reserves for an automated market maker (AMM) model,
    following Appendix A of "The Elliptic Concentrated Liquidity Pool (E-CLP)".
    It is a WIP. It does not yet consider transaction fees nor the size of the arbitrage opportunity.
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
        _jax_calc_gyroscope_reserves_with_fees_scan_function,
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
