# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax import lax
from jax.tree_util import Partial
from jax.lax import scan, dynamic_slice


def squareplus(x):
    # algebraic (so non-trancendental) replacement for softplus
    # see https://arxiv.org/abs/2112.11687 for detail
    return lax.mul(0.5, lax.add(x, lax.sqrt(lax.add(lax.square(x), 4.0))))


def inverse_squareplus(y):
    return lax.div(lax.sub(lax.square(y), 1.0), y)


def inverse_squareplus_np(y):
    return (y**2 - 1.0) / y


def inverse_logit(y):
    return jnp.log(y / (1.0 - y))


def _jax_ewma_scan_function(carry_list, arr_in, G_inf):
    # ewma is carry[0]
    # arr_in is the array value
    ewma = carry_list[0]
    ewma = ewma + (arr_in - ewma) / G_inf
    return [ewma], ewma


def _jax_ewma_at_infinity_via_scan(arr_in, lamb):
    r"""Exponentialy weighted moving average of PROPORTIONAL gradients
    at infinity. ie for input p, returns (on-line estimate of) 1/p dp/dt
    Parameters
    ----------
    arr_in : np.ndarray, float64
        A two-dimenisional numpy array
    lamb : np.ndarray, float64
        The decay constants

    Returns
    -------
    np.ndarray
        The vector of gradients, same length / shape as ``arr_in``

    """
    n = arr_in.shape[0]
    n_grads = arr_in.shape[1]

    G_inf = 1.0 / (1.0 - lamb)
    saturated_b = lamb / ((1 - lamb) ** 3)
    # scan_fn = Partial(_jax_gradient_scan_function, {'G_inf': G_inf, 'lamb': lamb})
    scan_fn = Partial(_jax_ewma_scan_function, G_inf=G_inf)

    carry_list_init = [arr_in[0]]
    carry_list_end, ewma = scan(scan_fn, carry_list_init, arr_in[1:])

    # gradients = jnp.row_stack([jnp.zeros((n_grads,), dtype=jnp.float64), gradients])

    return ewma


@jit
def lamb_to_memory(lamb):
    memory = jnp.cbrt(6 * lamb / ((1 - lamb) ** 3.0)) * 4.0
    # memory_days = np.clip(memory_days, a_min=0.0, a_max=365.0)
    return memory


@jit
def lamb_to_memory_days(lamb, chunk_period):
    memory_days = jnp.cbrt(6 * lamb / ((1 - lamb) ** 3.0)) * 2 * chunk_period / 1440
    # memory_days = np.clip(memory_days, a_min=0.0, a_max=365.0)
    return memory_days


@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def make_ewma_kernel(lamb, max_memory_days, chunk_period):
    """
    Generate an exponentially weighted moving average (EWMA) kernel.

    This function creates a kernel for computing the EWMA of a time series. The kernel is
    constructed based on the decay factor `lamb`, the maximum memory in days, and the chunk
    period which defines the granularity of the time series.

    Parameters
    ----------
    lamb : jnp.ndarray
        Decay factors for the EWMA, typically between 0 and 1.
    max_memory_days : int
        The number of days over which the memory decays.
    chunk_period : int
        The period of each chunk in minutes, defining the granularity of the time series.

    Returns
    -------
    jnp.ndarray
        The EWMA kernel matrix, with each row corresponding to a different decay factor.
    """
    memory = lamb_to_memory(lamb)
    kernel = lamb[:, jnp.newaxis] ** jnp.arange(
        int(max_memory_days * 1440 / chunk_period)
    )
    # kernel = kernel / kernel.sum(-1, keepdims=True)
    kernel = kernel * (1 - lamb[:, jnp.newaxis])
    kernel = kernel.T
    return kernel


@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def make_a_kernel(lamb, max_memory_days, chunk_period):
    memory = lamb_to_memory(lamb)
    kernel = lamb[:, jnp.newaxis] ** jnp.arange(
        int(max_memory_days * 1440 / chunk_period)
    )
    # kernel = kernel / kernel.sum(-1, keepdims=True)
    kernel = kernel / (1 - lamb[:, jnp.newaxis])
    kernel = kernel.T
    return kernel


@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def make_cov_kernel(lamb, max_memory_days, chunk_period):
    memory = lamb_to_memory(lamb)
    kernel = lamb[:, jnp.newaxis] ** jnp.arange(
        int(max_memory_days * 1440 / chunk_period)
    )
    # kernel = kernel / kernel.sum(-1, keepdims=True)
    # kernel = kernel / (1 - lamb[:, jnp.newaxis])
    kernel = kernel.T
    return kernel


@partial(
    jit,
    static_argnums=(2,),
)
def _jax_ewma_at_infinity_via_conv_1D(arr_in, kernel, return_slice_index=1):
    return jnp.convolve(arr_in, kernel, mode="full")[return_slice_index : len(arr_in)]


_jax_ewma_at_infinity_via_conv = vmap(
    _jax_ewma_at_infinity_via_conv_1D, in_axes=[-1, -1], out_axes=-1
)


@partial(
    jit,
    static_argnums=(2,),
)
def _jax_ewma_at_infinity_via_conv_1D_padded(arr_in, kernel, return_slice_index=0):
    return jnp.convolve(arr_in, kernel, mode="full")[return_slice_index : len(arr_in)]


_jax_ewma_at_infinity_via_conv_padded = vmap(
    _jax_ewma_at_infinity_via_conv_1D_padded, in_axes=[-1, -1], out_axes=-1
)


# NOTE THE [1: ] slice here maybe should be a [0: ] slice BUT THIS SHOULD BE TESTED
@jit
def _jax_gradients_at_infinity_via_conv_1D_padded_with_alt_ewma(
    arr_in, ewma, alt_ewma, kernel, saturated_b
):
    ewma_diff = arr_in - ewma
    a = jnp.convolve(ewma_diff, kernel, mode="valid")
    # grad_conv = a[:98] / (saturated_b * ewma_conv.T[:,0])
    grad = a[1:] / (saturated_b * alt_ewma[-len(a) + 1 :])
    return grad[1:]


_jax_gradients_at_infinity_via_conv_padded_with_alt_ewma = vmap(
    _jax_gradients_at_infinity_via_conv_1D_padded_with_alt_ewma,
    in_axes=[-1, -1, -1, -1, -1],
    out_axes=-1,
)


# NOTE THE [1: ] slice here maybe should be a [0: ] slice BUT THIS SHOULD BE TESTED
@jit
def _jax_gradients_at_infinity_via_conv_1D(arr_in, ewma, kernel, saturated_b):
    ewma_diff = arr_in[1:] - ewma
    a = jnp.convolve(ewma_diff, kernel, mode="full")
    # grad_conv = a[:98] / (saturated_b * ewma_conv.T[:,0])
    grad = a[: len(ewma)] / (saturated_b * ewma)
    return grad


_jax_gradients_at_infinity_via_conv = vmap(
    _jax_gradients_at_infinity_via_conv_1D, in_axes=[-1, -1, -1, -1], out_axes=-1
)


# NOTE THE [1: ] slice here maybe should be a [0: ] slice BUT THIS SHOULD BE TESTED
@jit
def _jax_gradients_at_infinity_via_conv_1D_padded(arr_in, ewma, kernel, saturated_b):
    ewma_diff = arr_in - ewma
    a = jnp.convolve(ewma_diff, kernel, mode="valid")
    # grad_conv = a[:98] / (saturated_b * ewma_conv.T[:,0])
    grad = a[1:] / (saturated_b * ewma[-len(a) + 1 :])
    return grad[1:]


_jax_gradients_at_infinity_via_conv_padded = vmap(
    _jax_gradients_at_infinity_via_conv_1D_padded, in_axes=[-1, -1, -1, -1], out_axes=-1
)


# NOTE THE [1: ] slice here maybe should be a [0: ] slice BUT THIS SHOULD BE TESTED
@jit
def _jax_gradients_at_infinity_via_conv_1D_with_alt_ewma(
    arr_in, ewma, alt_ewma, kernel, saturated_b
):
    ewma_diff = arr_in[1:] - ewma
    a = jnp.convolve(ewma_diff, kernel, mode="full")
    # grad_conv = a[:98] / (saturated_b * ewma_conv.T[:,0])
    grad = a[: len(ewma)] / (saturated_b * alt_ewma)
    return grad


_jax_gradients_at_infinity_via_conv_with_alt_ewma = vmap(
    _jax_gradients_at_infinity_via_conv_1D_with_alt_ewma,
    in_axes=[-1, -1, -1, -1, -1],
    out_axes=-1,
)


# NOTE THE [1: ] slice here maybe should be a [0: ] slice BUT THIS SHOULD BE TESTED
@jit
def _jax_gradients_at_infinity_via_conv_1D_padded_with_alt_ewma(
    arr_in, ewma, alt_ewma, kernel, saturated_b
):
    ewma_diff = arr_in - ewma
    a = jnp.convolve(ewma_diff, kernel, mode="valid")
    # grad_conv = a[:98] / (saturated_b * ewma_conv.T[:,0])
    grad = a[1:] / (saturated_b * alt_ewma[-len(a) + 1 :])
    return grad[1:]


_jax_gradients_at_infinity_via_conv_padded_with_alt_ewma = vmap(
    _jax_gradients_at_infinity_via_conv_1D_padded_with_alt_ewma,
    in_axes=[-1, -1, -1, -1, -1],
    out_axes=-1,
)


@partial(
    jit,
    static_argnums=(4,),
)
def _jax_variance_at_infinity_via_sums_1D(
    arr_in, ewma, kernel, control_idx, kernel_len
):
    # ewma_diff = (arr_in[2:] - ewma[:-1])
    # ewma_diff = arr_in - dynamic_slice(ewma,(control_idx,),(1,))
    # ewma_diff = dynamic_slice(arr_in,(2,),(int(control_idx),)) - ewma[control_idx]
    # ewma_diff = (arr_in[2:] - ewma[:-1]) * (arr_in[2:] - ewma[1:])
    # ewma_diff = (arr_in[1:] - ewma) * (arr_in[:-1] - ewma)
    # a = jnp.convolve(ewma_diff ** 2, kernel, mode="full")
    # a = jnp.zeros_like(ewma_diff)
    # a = a.at[:control_idx].set(ewma_diff[:control_idx])
    a = dynamic_slice(arr_in, (control_idx,), (kernel_len,))
    ewma_diff = a - dynamic_slice(ewma, (control_idx,), (1,))
    # ZERO PAD THE ARR IN, EWMA, ETC TO HAVE LEN(Kernel)
    # LEADING ZEROS, THEN WE CAN ALWAYS DYNAMICALLy SLICE THE
    # SAME SIZE ARRAY, AND DELETE THE CONV
    #  NEED TO DO SOME SMART BOOLEAN INDEXING/SLICING, AS ALL THE ZERO_PAD ZEROS - EWMA SHOULD NOT BE COUNTED IN THE SUM!
    # return ((a ** 2.0) * kernel)
    return jnp.nansum((ewma_diff**2.0) * kernel, 0)


@jit
def _jax_variance_at_infinity_via_conv_1D(arr_in, ewma, kernel, lamb):
    diff_old = arr_in[1:] - jnp.concat([arr_in[jnp.newaxis, 0], ewma])[:-1]
    diff_new = arr_in[1:] - ewma

    outer = diff_old * diff_new
    a = jnp.convolve(outer, kernel, mode="full")
    cov = a[: len(outer)] * (1 - lamb)
    return jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), cov], axis=0)


conv_intermediate = vmap(
    Partial(jnp.convolve, mode="full"), in_axes=[-1, -1], out_axes=-1
)

conv_vmap = vmap(conv_intermediate, in_axes=[1, None], out_axes=1)


@jit
def _jax_covariance_at_infinity_via_conv(arr_in, ewma, kernel, lamb):
    n = arr_in.shape[1]
    diff_old = arr_in[1:] - jnp.concat([arr_in[jnp.newaxis, 0], ewma])[:-1]
    diff_new = arr_in[1:] - ewma

    outer = jnp.einsum("...i,...j->...ij", diff_old, diff_new)
    a = conv_vmap(outer, kernel)
    cov = a[: len(outer)] * (1 - lamb)
    return jnp.concatenate([np.zeros((1, n, n), dtype=jnp.float64), cov], axis=0)


# _jax_covariance_at_infinity_via_conv = vmap(
#     _jax_covariance_at_infinity_via_conv_1D, in_axes=[-1, -1, -1, -1], out_axes=-1
# )

# _jax_variance_at_infinity_via_sums_1D(jnp.pad(chunkwise_price_values[:,0],((len(kernel)-1,0)),constant_values=jnp.nan),ewma[:,0],kernel[:,0][::-1],1,len(kernel))

# _jax_variance_at_infinity_via_sums_1D_ = Partial(_jax_variance_at_infinity_via_sums_1D,kernel_len=len(kernel))

# _jax_covariance_at_infinity_via_conv_intermediate = jit(vmap(
#     _jax_variance_at_infinity_via_conv_1D,
#     in_axes=[None,None,0,0],
#     out_axes=-1,
# ))

_jax_variance_at_infinity_via_conv = jit(
    vmap(
        _jax_variance_at_infinity_via_conv_1D,
        in_axes=[-1, -1, -1, -1],
        out_axes=-1,
    )
)


@jit
def _jax_gradient_scan_function(carry_list, arr_in, G_inf, lamb, saturated_b):
    # ewma is carry[0]
    # running_a is carry[1]
    # arr_in is the array value
    ewma = carry_list[0]
    running_a = carry_list[1]
    ewma = ewma + (arr_in - ewma) / G_inf
    running_a = lamb * running_a + G_inf * (arr_in - ewma)
    running_a = jnp.where(jnp.abs(running_a) < 1e-10, 0.0, running_a)
    gradient = running_a / (saturated_b * ewma)
    return [ewma, running_a], gradient


@jit
def _jax_gradient_scan_function_with_alt_ewma(
    carry_list, arr_in, G_inf, alt_G_inf, lamb, saturated_b
):
    # ewma is carry[0]
    # alt_ewma is carry[1]
    # running_a is carry[2]
    # arr_in is the array value
    ewma = carry_list[0]
    alt_ewma = carry_list[1]
    running_a = carry_list[2]
    ewma = ewma + (arr_in - ewma) / G_inf
    alt_ewma = alt_ewma + (arr_in - alt_ewma) / alt_G_inf
    running_a = lamb * running_a + G_inf * (arr_in - ewma)
    running_a = jnp.where(jnp.abs(running_a) < 1e-10, 0.0, running_a)
    gradient = running_a / (saturated_b * alt_ewma)
    return [ewma, alt_ewma, running_a], gradient


@jit
def _jax_gradients_at_infinity_via_scan(arr_in, lamb):
    r"""Exponentialy weighted moving average of PROPORTIONAL gradients
    at infinity. ie for input p, returns (on-line estimate of) 1/p dp/dt
    Parameters
    ----------
    arr_in : np.ndarray, float64
        A two-dimenisional numpy array
    lamb : np.ndarray, float64
        The decay constants

    Returns
    -------
    np.ndarray
        The vector of gradients, same length / shape as ``arr_in``

    """
    n = arr_in.shape[0]
    n_grads = arr_in.shape[1]

    G_inf = 1.0 / (1.0 - lamb)
    saturated_b = lamb / ((1 - lamb) ** 3)
    # scan_fn = Partial(_jax_gradient_scan_function, {'G_inf': G_inf, 'lamb': lamb})
    scan_fn = Partial(
        _jax_gradient_scan_function, G_inf=G_inf, lamb=lamb, saturated_b=saturated_b
    )

    carry_list_init = [arr_in[0], jnp.ones((n_grads,), dtype=jnp.float64)]
    carry_list_end, gradients = scan(scan_fn, carry_list_init, arr_in[1:])

    gradients = jnp.vstack([jnp.zeros((n_grads,), dtype=jnp.float64), gradients])

    return gradients


@jit
def _jax_gradients_at_infinity_via_scan_with_alt_ewma(arr_in, lamb, alt_lamb):
    r"""Exponentialy weighted moving average of PROPORTIONAL gradients
    at infinity. ie for input p, returns (on-line estimate of) 1/p dp/dt
    Parameters
    ----------
    arr_in : np.ndarray, float64
        A two-dimenisional numpy array
    lamb : np.ndarray, float64
        The decay constants

    Returns
    -------
    np.ndarray
        The vector of gradients, same length / shape as ``arr_in``

    """
    n = arr_in.shape[0]
    n_grads = arr_in.shape[1]

    G_inf = 1.0 / (1.0 - lamb)
    alt_G_inf = 1.0 / (1.0 - alt_lamb)

    saturated_b = lamb / ((1 - lamb) ** 3)
    # scan_fn = Partial(_jax_gradient_scan_function, {'G_inf': G_inf, 'lamb': lamb})
    scan_fn = Partial(
        _jax_gradient_scan_function_with_alt_ewma,
        G_inf=G_inf,
        alt_G_inf=alt_G_inf,
        lamb=lamb,
        saturated_b=saturated_b,
    )

    carry_list_init = [arr_in[0], arr_in[0], jnp.ones((n_grads,), dtype=jnp.float64)]
    carry_list_end, gradients = scan(scan_fn, carry_list_init, arr_in[1:])

    gradients = jnp.vstack([jnp.zeros((n_grads,), dtype=jnp.float64), gradients])

    return gradients


@jit
def _jax_gradients_at_infinity_via_scan_alt1(arr_in, lamb):
    r"""Exponentialy weighted moving average of PROPORTIONAL gradients
    at infinity. ie for input p, returns (on-line estimate of) 1/p dp/dt
    Parameters
    ----------
    arr_in : np.ndarray, float64
        A two-dimenisional numpy array
    lamb : np.ndarray, float64
        The decay constants

    Returns
    -------
    np.ndarray
        The vector of gradients, same length / shape as ``arr_in``

    """
    n = arr_in.shape[0]
    n_grads = arr_in.shape[1]

    G_inf = 1.0 / (1.0 - lamb)
    saturated_b = lamb / ((1 - lamb) ** 3)
    # scan_fn = Partial(_jax_gradient_scan_function, {'G_inf': G_inf, 'lamb': lamb})
    scan_fn = Partial(
        _jax_gradient_scan_function, G_inf=G_inf, lamb=lamb, saturated_b=saturated_b
    )

    carry_list_init = [arr_in[0], jnp.ones((n_grads,), dtype=jnp.float64)]

    gradients = jnp.vstack(
        [
            jnp.zeros((n_grads,), dtype=jnp.float64),
            scan(scan_fn, carry_list_init, arr_in[1:])[1],
        ]
    )

    return gradients


@jit
def _jax_gradients_at_infinity_via_scan_alt2(arr_in, lamb):
    r"""Exponentialy weighted moving average of PROPORTIONAL gradients
    at infinity. ie for input p, returns (on-line estimate of) 1/p dp/dt
    Parameters
    ----------
    arr_in : np.ndarray, float64
        A two-dimenisional numpy array
    lamb : np.ndarray, float64
        The decay constants

    Returns
    -------
    np.ndarray
        The vector of gradients, same length / shape as ``arr_in``

    """
    n = arr_in.shape[0]
    n_grads = arr_in.shape[1]

    G_inf = 1.0 / (1.0 - lamb)
    saturated_b = lamb / ((1 - lamb) ** 3)
    # scan_fn = Partial(_jax_gradient_scan_function, {'G_inf': G_inf, 'lamb': lamb})
    scan_fn = Partial(
        _jax_gradient_scan_function, G_inf=G_inf, lamb=lamb, saturated_b=saturated_b
    )

    carry_list_init = [arr_in[0], jnp.ones((n_grads,), dtype=jnp.float64)]

    gradients = jnp.zeros((n, n_grads), dtype=jnp.float64)
    gradients = gradients.at[1:].set(scan(scan_fn, carry_list_init, arr_in[1:])[1])

    return gradients


@jit
def _jax_covariance_scan_function(carry_list, arr_in, G_inf, lamb):
    ewma = carry_list[0]
    running_a = carry_list[1]

    diff_old = arr_in - ewma
    ewma = ewma + (arr_in - ewma) / G_inf
    diff_new = arr_in - ewma
    running_a = lamb * running_a + jnp.outer(diff_old, diff_new)
    # if np.sum(np.abs(running_a) < 1e-10) > 0:
    # running_a[np.abs(running_a) < 1e-10] = 0
    covariance = running_a / G_inf
    return [ewma, running_a], covariance


@jit
def _jax_covariance_matrix_at_infinity_via_scan(arr_in, lamb):
    r"""Exponentialy weighted moving average of covariance matrix between
    signals at infinity
    ----------
    arr_in : np.ndarray, float64
        A two-dimenisional numpy array
    lamb : np.ndarray, float64
        The decay constants

    Returns
    -------
    np.ndarray
        The series of covariance matrices, same length as ``arr_in``

    """
    n = arr_in.shape[0]
    dim = arr_in.shape[1]
    # covariance = np.empty(
    #     (
    #         n,
    #         dim,
    #         dim,
    #     ),
    #     dtype=float64,
    # )
    G_inf = 1.0 / (1.0 - lamb)
    ewma = arr_in[0]
    running_a = jnp.eye(dim)
    # , dtype=jnp.float64)

    scan_fn = Partial(_jax_covariance_scan_function, G_inf=G_inf, lamb=lamb)

    carry_list_init = [ewma, running_a]

    covariances = jnp.vstack(
        [
            jnp.zeros((1, dim, dim)),
            scan(scan_fn, carry_list_init, arr_in[1:])[1],
        ]
    )

    return covariances


@jit
def _jax_variance_scan_function(carry_list, arr_in, G_inf, lamb):
    ewma = carry_list[0]
    running_var = carry_list[1]

    diff_old = arr_in - ewma
    ewma = ewma + diff_old / G_inf
    diff_new = diff_old * (1 - 1 / G_inf)  # Equivalent to arr_in - ewma_new
    running_var = lamb * running_var + diff_old * diff_new
    variance = running_var * (1 - lamb)

    return [ewma, running_var], variance


@jit
def _jax_variance_at_infinity_via_scan(arr_in, lamb):
    """Calculate exponentially weighted variance using scan.

    Parameters
    ----------
    arr_in : jnp.ndarray
        Input array of shape (time, features)
    lamb : jnp.ndarray
        Decay factor for each feature

    Returns
    -------
    jnp.ndarray
        Variance estimates of shape (time, features)
    """
    n = arr_in.shape[0]
    n_features = arr_in.shape[1]

    G_inf = 1.0 / (1.0 - lamb)
    scan_fn = Partial(_jax_variance_scan_function, G_inf=G_inf, lamb=lamb)

    # Initialize with first value
    carry_list_init = [arr_in[0], jnp.zeros((n_features,), dtype=jnp.float64)]

    # Run scan and prepend ones for first timestep
    _, variances = scan(scan_fn, carry_list_init, arr_in[1:])
    variances = jnp.vstack([jnp.ones((1, n_features), dtype=jnp.float64), variances])

    return variances
