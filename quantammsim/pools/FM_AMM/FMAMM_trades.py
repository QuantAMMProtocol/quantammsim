# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jax import jit, vmap
from jax import devices
from jax.tree_util import Partial
from jax.lax import scan
from jax.lib.xla_bridge import default_backend
from jax import local_device_count, devices

from functools import partial

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
else:
    GPU_DEVICE = devices("cpu")[0]


def zero_trade_function_FMAMM(reserves, trade, gamma):
    return jnp.zeros(reserves.shape)


def _jax_calc_FMAMM_trade_from_exact_out_given_in(
    reserves, token_in, token_out, amount_in, gamma=0.997
):
    """
    Calculate the post-trade reserves from a trade of exact amount
    of token_out given an input amount of token_in for a FM AMM pool.

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
    gamma : float, optional
        Fee parameter, where (1 - gamma) is the fee percentage. Default is 0.997.

    Returns:
    --------
    jnp.ndarray
        The reserves post trade.
    """

    # We use Eq 5 in FM AMM paper
    # x < 0 in Eq 5 corresponds the pool is trading in token 1 and receiving token 0
    # in an 'exactoutgivenin' trade
    # So that is what we need for token 1 going into the pool and token 2 coming out
    # If instead the exactoutgivenin trade is token 2 going into the pool and token 1 coming out
    # then we need to swap the input and output tokens
    # and get get a new equation for the trade that is equivalent to Eq 5 with x<0 but
    # swapping X->Y, Y->X, x->y, y->x

    # effective price for a trade in of token 1 and out of token 2 is
    # p-tilde (amount_in_token_1) = R_2 / ((R_1 / gamma) - 2 * amount_in_token_1)
    # which is the same as Eq 5 with x<0, so x is negative.
    # Anount_out here is then p-tilde (amount_in_token_1) * R_2

    # effective price for a trade in of token 2 and out of token 1 is
    # p-tilde (amount_in_token_2) = R_1 / ((R_2 / gamma) - 2 * amount_in_token_2)
    # which is the same as Eq 5 with x<0 and swapping X->Y, Y->X, x->y, y->X
    # so y is negative.
    # Anount_out here is then p-tilde (amount_in_token_2) * R_1

    # Note are using non-pythonic indexing here, in keeping with the TFMM papers.

    # Calculate the new reserves for token_out
    amount_out = reserves[token_out] * amount_in / ((reserves[token_in] / gamma) + 2 * amount_in)
    overall_trade = jnp.zeros(len(reserves))
    overall_trade = overall_trade.at[token_in].set(amount_in)
    overall_trade = overall_trade.at[token_out].set(-amount_out)
    return jnp.where(amount_in != 0, overall_trade, 0)


# version of _jax_calc_FMAMM_trade_from_exact_out_given_in that
# in 'trade' as one single input. Useful for lazy evaluation
def wrapped_FMAMM_trade_function(reserves, trade, gamma):
    return _jax_calc_FMAMM_trade_from_exact_out_given_in(
        reserves, trade[0], trade[1], trade[2], gamma
    )


# Create a jitted function that includes the cond, for lazy evaluation
@jit
def jitted_FMAMM_cond_trade(condition, reserves, trade, gamma):
    return jax.lax.cond(
        condition,
        wrapped_FMAMM_trade_function,
        zero_trade_function_FMAMM,
        reserves,
        trade,
        gamma,
    )


if __name__ == "__main__":
    print("testing")

    reserves = jnp.array([10000000, 20000000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 0.99
    print(wrapped_FMAMM_trade_function(reserves, trade, gamma))
    reserves = jnp.array([1000, 2000])
    trade = jnp.array([0, 1, 100])
    gamma = 1.0
    print(wrapped_FMAMM_trade_function(reserves, trade, gamma))
    trade = jnp.array([1, 0, 100])
    gamma = 1.0
    print(wrapped_FMAMM_trade_function(reserves, trade, gamma))
