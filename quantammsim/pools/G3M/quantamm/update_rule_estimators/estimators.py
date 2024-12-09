# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
from jax.lib.xla_bridge import default_backend
from jax import local_device_count, devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

# DEFAULT_BACKEND = 'gpu'
import jax.numpy as jnp
from jax import jit, vmap
from jax import devices, device_put
from jax.tree_util import Partial
from jax.lax import scan, stop_gradient

import numpy as np

from functools import partial

np.seterr(all="raise")
np.seterr(under="print")

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    make_ewma_kernel,
    make_a_kernel,
    make_cov_kernel,
    _jax_ewma_at_infinity_via_conv_padded,
    _jax_gradients_at_infinity_via_conv_padded,
    _jax_gradients_at_infinity_via_conv_padded_with_alt_ewma,
    _jax_variance_at_infinity_via_conv,
    squareplus,
    _jax_gradients_at_infinity_via_scan,
    _jax_gradients_at_infinity_via_scan_with_alt_ewma,
)
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
)


def calc_gradients(
    update_rule_parameter_dict,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    use_alt_lamb,
    cap_lamb=True,
):

    lamb = calc_lamb(update_rule_parameter_dict)
    max_lamb = memory_days_to_lamb(max_memory_days, chunk_period)
    # Apply max_memory_days restriction to lamb and alt_lamb
    if cap_lamb:
        capped_lamb = jnp.clip(lamb, a_min=0.0, a_max=max_lamb)
        lamb = capped_lamb
    safety_margin_max_memory_days = max_memory_days * 5.0
    # we can use alt lamb / alt memory days to allow different parts of
    # update rules to act over different memory lengths
    if use_alt_lamb:
        if update_rule_parameter_dict.get("logit_delta_lamb") is not None:
            logit_delta_lamb = update_rule_parameter_dict["logit_delta_lamb"]
            logit_alt_lamb = logit_delta_lamb + update_rule_parameter_dict["logit_lamb"]
            alt_lamb = jnp.exp(logit_alt_lamb) / (1 + jnp.exp(logit_alt_lamb))
            if cap_lamb:
                capped_alt_lamb = jnp.clip(alt_lamb, a_min=0.0, a_max=max_lamb)
                alt_lamb = capped_alt_lamb
        else:
            raise Exception

        alt_memory_days = (
            jnp.cbrt(6 * alt_lamb / ((1 - alt_lamb) ** 3)) * 2 * chunk_period / 1440
        )
        alt_memory_days = jnp.clip(alt_memory_days, a_min=0.0, a_max=max_memory_days)
    else:
        capped_alt_lamb = lamb
        alt_lamb = lamb

    ## NOTE THAT PROBABLY WE WONT TAKE GRADIENTS WRT chunk_period
    ## as that would lead to changing shapes of input data
    ## which means we have to re-jit every step, which we probably dont want to do!

    if DEFAULT_BACKEND != "cpu":
        ewma_kernel = make_ewma_kernel(
            lamb, safety_margin_max_memory_days, chunk_period
        )
        a_kernel = make_a_kernel(lamb, safety_margin_max_memory_days, chunk_period)
        padded_chunkwise_price_values = jnp.vstack(
            [
                jnp.ones(
                    (
                        int(safety_margin_max_memory_days * 1440 / chunk_period),
                        chunkwise_price_values.shape[1],
                    )
                )
                * chunkwise_price_values[0],
                chunkwise_price_values,
            ]
        )
        ewma_padded = _jax_ewma_at_infinity_via_conv_padded(
            padded_chunkwise_price_values, ewma_kernel
        )
        saturated_b = lamb / ((1 - lamb) ** 3)

        if use_alt_lamb:
            alt_ewma_kernel = make_ewma_kernel(
                alt_lamb, safety_margin_max_memory_days, chunk_period
            )
            alt_ewma_padded = _jax_ewma_at_infinity_via_conv_padded(
                padded_chunkwise_price_values, alt_ewma_kernel
            )
            gradients = _jax_gradients_at_infinity_via_conv_padded_with_alt_ewma(
                padded_chunkwise_price_values,
                ewma_padded,
                alt_ewma_padded,
                a_kernel,
                saturated_b,
            )
        else:
            gradients = _jax_gradients_at_infinity_via_conv_padded(
                padded_chunkwise_price_values, ewma_padded, a_kernel, saturated_b
            )
    else:
        if use_alt_lamb:
            gradients = _jax_gradients_at_infinity_via_scan_with_alt_ewma(
                chunkwise_price_values, lamb, alt_lamb
            )[1:]
        else:
            gradients = _jax_gradients_at_infinity_via_scan(
                chunkwise_price_values, lamb
            )[1:]
    return gradients


def calc_ewma_padded(
    update_rule_parameter_dict,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    cap_lamb=True,
):

    lamb = calc_lamb(update_rule_parameter_dict)
    max_lamb = memory_days_to_lamb(max_memory_days, chunk_period)
    # Apply max_memory_days restriction to lamb and alt_lamb
    # og_lamb = lamb.copy()
    if cap_lamb:
        capped_lamb = jnp.clip(lamb, a_min=0.0, a_max=max_lamb)
        lamb = capped_lamb
    safety_margin_max_memory_days = max_memory_days * 5.0

    ewma_kernel = make_ewma_kernel(lamb, safety_margin_max_memory_days, chunk_period)
    padded_chunkwise_price_values = jnp.vstack(
        [
            jnp.ones(
                (
                    int(safety_margin_max_memory_days * 1440 / chunk_period),
                    chunkwise_price_values.shape[1],
                )
            )
            * chunkwise_price_values[0],
            chunkwise_price_values,
        ]
    )
    ewma_padded = _jax_ewma_at_infinity_via_conv_padded(
        padded_chunkwise_price_values, ewma_kernel
    )

    return ewma_padded


def calc_alt_ewma_padded(
    update_rule_parameter_dict,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    cap_lamb=True,
):

    lamb = calc_lamb(update_rule_parameter_dict)
    max_lamb = memory_days_to_lamb(max_memory_days, chunk_period)
    # Apply max_memory_days restriction to lamb and alt_lamb
    # og_lamb = lamb.copy()
    if cap_lamb:
        capped_lamb = jnp.clip(lamb, a_min=0.0, a_max=max_lamb)
        lamb = capped_lamb
    safety_margin_max_memory_days = max_memory_days * 5.0
    # we can use alt lamb / alt memory days to allow different parts of

    if update_rule_parameter_dict.get("logit_delta_lamb") is not None:
        logit_delta_lamb = update_rule_parameter_dict["logit_delta_lamb"]
        logit_alt_lamb = logit_delta_lamb + update_rule_parameter_dict["logit_lamb"]
        alt_lamb = jnp.exp(logit_alt_lamb) / (1 + jnp.exp(logit_alt_lamb))
        # og_alt_lamb = alt_lamb.copy()
        if cap_lamb:
            capped_alt_lamb = jnp.clip(alt_lamb, a_min=0.0, a_max=max_lamb)
            alt_lamb = capped_alt_lamb
    else:
        raise Exception

    alt_memory_days = (
        jnp.cbrt(6 * alt_lamb / ((1 - alt_lamb) ** 3)) * 2 * chunk_period / 1440
    )
    alt_memory_days = jnp.clip(alt_memory_days, a_min=0.0, a_max=max_memory_days)

    alt_ewma_kernel = make_ewma_kernel(
        alt_lamb, safety_margin_max_memory_days, chunk_period
    )
    padded_chunkwise_price_values = jnp.vstack(
        [
            jnp.ones(
                (
                    int(safety_margin_max_memory_days * 1440 / chunk_period),
                    chunkwise_price_values.shape[1],
                )
            )
            * chunkwise_price_values[0],
            chunkwise_price_values,
        ]
    )
    alt_ewma_padded = _jax_ewma_at_infinity_via_conv_padded(
        padded_chunkwise_price_values, alt_ewma_kernel
    )

    return alt_ewma_padded


def calc_return_variances(
    update_rule_parameter_dict,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    cap_lamb,
):
    lamb = calc_lamb(update_rule_parameter_dict)
    if cap_lamb:
        max_lamb = memory_days_to_lamb(max_memory_days, chunk_period)
        lamb = jnp.clip(lamb, a_min=0.0, a_max=max_lamb)
    safety_margin_max_memory_days = max_memory_days * 5.0
    cov_kernel = make_cov_kernel(lamb, safety_margin_max_memory_days, chunk_period)
    ewma_kernel = make_ewma_kernel(lamb, safety_margin_max_memory_days, chunk_period)
    returns = jnp.diff(chunkwise_price_values, axis=0) / chunkwise_price_values[:-1]

    padded_returns = jnp.vstack(
        [
            jnp.ones(
                (
                    int(safety_margin_max_memory_days * 1440 / chunk_period),
                    returns.shape[1],
                )
            )
            * returns[0],
            returns,
        ]
    )

    # ewma_returns = _ewma_at_infinity(returns, lamb)

    # ewma_padded_ = jnp.convolve(padded_returns[:, 0], ewma_kernel[:, 0], mode="full")[
    #     (return_slice_index) : len(padded_returns)
    # ]
    ewma_returns_padded = _jax_ewma_at_infinity_via_conv_padded(
        padded_returns, ewma_kernel
    )
    variances = _jax_variance_at_infinity_via_conv(
        padded_returns, ewma_returns_padded[1:], cov_kernel, lamb
    )

    return variances


def calc_return_precision_based_weights(
    update_rule_parameter_dict,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    cap_lamb,
):
    variances = calc_return_variances(
        update_rule_parameter_dict,
        chunkwise_price_values,
        chunk_period,
        max_memory_days,
        cap_lamb,
    )
    # ewma_padded = calc_ewma_padded(update_rule_parameter_dict, chunkwise_price_values, chunk_period, max_memory_days, cap_lamb)
    # ewma = ewma_padded[-(len(chunkwise_price_values) - 1):]
    # variances = _jax_variance_at_infinity_via_conv(chunkwise_price_values, ewma, cov_kernel, lamb)
    precision_based_weights = 1.0 / variances
    precision_based_weights = precision_based_weights / precision_based_weights.sum(
        axis=-1, keepdims=True
    )
    return precision_based_weights

    # cov_kernel = make_cov_kernel(lamb, safety_margin_max_memory_days, chunk_period)
    # variances = _jax_variance_at_infinity_via_conv(
    #     chunkwise_price_values, ewma, cov_kernel, lamb
    # )

    # diag_precisions = diag_numba(precisions)
    # reshape_sum = np.reshape(np.sum(diag_precisions, axis=-1), (n - 1, 1))
    # precision_based_weights = 1.0 / variances
    # precision_based_weights = precision_based_weights / precision_based_weights.sum(
    #     axis=-1, keepdims=True
    # )


def calc_k(update_rule_parameter_dict, memory_days):
    if update_rule_parameter_dict.get("log_k") is not None:
        log_k = update_rule_parameter_dict.get("log_k")
        k = (2**log_k) * memory_days
    else:
        k = update_rule_parameter_dict.get("k") * memory_days
    return k
