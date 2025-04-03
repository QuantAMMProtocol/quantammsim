from jax import config, jit, devices
import jax.numpy as jnp
from jax.lax import cond
from jax import default_backend

config.update("jax_enable_x64", True)

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
else:
    GPU_DEVICE = devices("cpu")[0]


@jit
def _jax_calc_G3M_trade_from_exact_out_given_in(
    reserves, weights, token_in, token_out, amount_in, gamma=0.997
):
    """
    Calculate the post-trade reserves from a trade of exact amount
    of token_out given an input amount of token_in for a G3M pool.

    Parameters:
    -----------
    reserves : jnp.ndarray
        Current reserves of all tokens in the AMM.
    weights : jnp.ndarray
        Current weights of all tokens in the AMM.
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
    # Calculate the ratio of weights
    token_in = jnp.int32(token_in)
    token_out = jnp.int32(token_out)

    weights_ratio = weights[token_in] / weights[token_out]
    # Calculate the new reserves for token_out
    amount_out = reserves[token_out] * (
        1.0 - (1.0 / (1 + gamma * (amount_in / reserves[token_in]))) ** weights_ratio
    )
    overall_trade = jnp.zeros(len(weights))
    overall_trade = overall_trade.at[token_in].set(amount_in)
    overall_trade = overall_trade.at[token_out].set(-amount_out)
    return jnp.where(amount_in != 0, overall_trade, 0)


# version of _jax_calc_G3M_trade_from_exact_out_given_in that
# in 'trade' as one single input. Useful for lazy evaluation
def wrapped_G3M_trade_function(reserves, weights, trade, gamma):
    return _jax_calc_G3M_trade_from_exact_out_given_in(
        reserves, weights, trade[0], trade[1], trade[2], gamma
    )


def zero_trade_function_G3M(reserves, weights, trade, gamma):
    return jnp.zeros(reserves.shape)


# Create a jitted function that includes the cond, for lazy evaluation
@jit
def jitted_G3M_cond_trade(condition, reserves, weights, trade, gamma):
    return cond(
        condition,
        wrapped_G3M_trade_function,
        zero_trade_function_G3M,
        reserves,
        weights,
        trade,
        gamma,
    )
