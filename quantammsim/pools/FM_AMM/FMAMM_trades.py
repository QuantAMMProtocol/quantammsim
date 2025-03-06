# again, this only works on startup!
from jax import config, jit,devices
from jax import default_backend
from jax.lax import cond
import jax.numpy as jnp

config.update("jax_enable_x64", True)

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
else:
    GPU_DEVICE = devices("cpu")[0]


# def zero_trade_function_FMAMM(reserves, _unused_trade, _unused_gamma):
#     """
#     Generates a zero-filled array with the same shape as the input reserves.

#     Args:
#         reserves (jnp.ndarray): The current reserves of the AMM.
#         trade (Any): The trade details (not used in this function).
#         gamma (Any): A parameter (not used in this function).

#     Returns:
#         jnp.ndarray: An array of zeros with the same shape as the input reserves.
#     """
#     return jnp.zeros(reserves.shape)


def _jax_calc_FMAMM_trade_from_exact_out_given_in(
    reserves,
    weights,
    token_in,
    token_out,
    amount_in,
    gamma=0.997,
):
    """
    Calculate the post-trade reserves from a trade of exact amount
    of token_out given an input amount of token_in for a FM AMM pool.

    Parameters:
    -----------
    reserves : jnp.ndarray
        Current reserves of all tokens in the AMM.
    weights : jnp.ndarray
        Array containing the weights of the pool, assumed to be of shape (2,) and sum to 1.
    token_in : int
        Index of the input token.
    token_out : int
        Index of the output token.
    amount_in : float
        Amount of token_in to be swapped.
    gamma : float, optional
        Fee parameter, where (1 - gamma) is the fee percentage. Default is 0.997.

    Returns:
    --------
    jnp.ndarray
        The reserves post trade.
    """

    # Note are using non-pythonic indexing in this comment, in keeping with the TFMM papers.
    # So 'token 1' in the text below means the 0-indexed token in the reserves array in code and
    # 'token 2' means the 1-indexed token in the reserves array in code. The FM-AMM paper
    # associates 'x' with 'token 1' and 'y' with 'token 2'.

    # We use Eq 3 in FM AMM paper.
    # x < 0 in Eq 3 corresponds to the pool sending out token 2 and receiving token 1
    # in an 'exactoutgivenin' trade.

    # We also need an equation for token 1 going into the pool and token 2 coming out.
    # I.e., if instead the exactoutgivenin trade has the pool receiving token 2 and
    # sending out token 1.
    # Then we need to swap the input and output tokens from the above equation
    # and get get a new equation for the trade that is equivalent to Eq 3 with x<0 but
    # swapping X->Y, Y->X, x->y, y->x

    # The effective price for a trade in of token 1 and out of token 2 is
    # p-tilde(amount_in_token_1) = R_2 / ((R_1 / gamma) + (1/weight) * amount_in_token_1)
    # which is the same as Eq 5 with x<0, so x is negative (which is why we now have
    # a + instead of a - in the denominator as amount_in_token_1 is always positive).
    # Amount_out of token 2 is then p-tilde(amount_in_token_1) * amount_in_token_1

    # effective price for a trade in of token 2 and out of token 1 is
    # p-tilde(amount_in_token_2) = R_1 / ((R_2 / gamma) - (1/weight) * amount_in_token_2)
    # which is the same as Eq 5 with x<0 and swapping X->Y, Y->X, x->y, y->X
    # so y is negative.
    # Amount_out of token 1 here is then p-tilde(amount_in_token_2) * amount_in_token_2

    token_in = jnp.int32(token_in)
    token_out = jnp.int32(token_out)

    # TODO: MW check this
    # Calculate the new reserves for token_out
    amount_out = (
        reserves[token_out]
        * amount_in
        / (
            (reserves[token_in] / gamma)
            * ((1.0 - weights[token_in]) / weights[token_in])
            + amount_in / weights[token_in]
        )
    )
    overall_trade = jnp.zeros(len(reserves))
    overall_trade = overall_trade.at[token_in].set(amount_in)
    overall_trade = overall_trade.at[token_out].set(-amount_out)
    return jnp.where(amount_in != 0, overall_trade, 0)


# version of _jax_calc_FMAMM_trade_from_exact_out_given_in that
# in 'trade' as one single input. Useful for lazy evaluation
def wrapped_FMAMM_trade_function(reserves, weights, trade, gamma):
    return _jax_calc_FMAMM_trade_from_exact_out_given_in(
        reserves, weights, trade[0], trade[1], trade[2], gamma
    )


def zero_trade_function_FMAMM(reserves, weights, trade, gamma):
    return jnp.zeros(reserves.shape)


# Create a jitted function that includes the cond, for lazy evaluation
@jit
def jitted_FMAMM_cond_trade(condition, reserves, weights, trade, gamma):
    return cond(
        condition,
        wrapped_FMAMM_trade_function,
        zero_trade_function_FMAMM,
        reserves,
        weights,
        trade,
        gamma,
    )


if __name__ == "__main__":
    print("testing")
    weights = jnp.array([0.5, 0.5])
    print("weights", weights)
    reserves = jnp.array([10000000, 20000000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 1.0
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 1.0
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    print("---")
    weights = jnp.array([0.2, 0.8])
    print("weights", weights)
    reserves = jnp.array([10000000, 20000000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 1.0
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 1.0
    print(wrapped_FMAMM_trade_function(reserves, weights, trade, gamma))
    print("G3M")
    from quantammsim.pools.G3M.G3M_trades import wrapped_G3M_trade_function
    print("---")
    weights = jnp.array([0.5, 0.5])
    print("weights", weights)
    reserves = jnp.array([10000000, 20000000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 1.0
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 1.0
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))

    print("---")
    weights = jnp.array([0.2, 0.8])
    print("weights", weights)
    reserves = jnp.array([10000000, 20000000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 1.0
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 1.0
    print(wrapped_G3M_trade_function(reserves, weights, trade, gamma))
