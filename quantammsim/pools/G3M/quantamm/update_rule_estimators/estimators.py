# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
from jax import default_backend
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
    _jax_ewma_at_infinity_via_scan,
    _jax_gradients_at_infinity_via_conv_padded,
    _jax_gradients_at_infinity_via_conv_padded_with_alt_ewma,
    _jax_variance_at_infinity_via_conv,
    _jax_variance_at_infinity_via_scan,
    squareplus,
    _jax_gradients_at_infinity_via_scan,
    _jax_gradients_at_infinity_via_scan_with_alt_ewma,
)
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    jax_memory_days_to_lamb,
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
    """Calculate time-weighted price gradients for TFMM strategy implementation.

    Computes gradients of price movements using exponentially weighted moving averages (EWMA),
    with support for both CPU and GPU acceleration via JAX. Implements the gradient calculation
    described in the TFMM litepaper, with additional features for alternative memory lengths.

    Parameters
    ----------
    update_rule_parameter_dict : dict
        Dictionary containing strategy parameters including:
        - 'logit_lamb': Controls the base memory length
        - 'logit_delta_lamb': Optional, controls alternative memory length if use_alt_lamb=True
    chunkwise_price_values : ndarray
        Array of shape (time_steps, n_assets) containing price values for each asset
        over time
    chunk_period : float
        Time period between chunks in minutes
    max_memory_days : float
        Maximum allowed memory length in days, used to cap lambda parameters
    use_alt_lamb : bool
        Whether to use an alternative lambda parameter for part of the calculation.
        Enables different parts of update rules to act over different memory lengths
    cap_lamb : bool, optional
        Whether to apply maximum memory day restriction to lambda parameters.
        Defaults to True

    Returns
    -------
    ndarray
        Array of shape (time_steps-1, n_assets) containing calculated gradients.
        For each asset, represents the time-weighted rate of change of prices.

    Notes
    -----
    The function implements two calculation paths:
    
    1. GPU path: Uses convolution operations for efficient parallel computation
       - Pads input data to handle initialization
       - Creates specialized kernels for EWMA calculation
       - Leverages GPU parallelism for efficient computation
    
    2. CPU path: Uses scan operations for sequential computation
       - More memory efficient
       - Uses a scan operation, so is fundamentally sequential

    The gradient calculation follows the methodology described in the TFMM litepaper,
    using exponential weighting to estimate price trends while avoiding look-ahead bias.

    """

    lamb = calc_lamb(update_rule_parameter_dict)
    max_lamb = jax_memory_days_to_lamb(max_memory_days, chunk_period)
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


def calc_ewma_pair(
    memory_days_1,
    memory_days_2,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    cap_lamb=True,
):
    """Calculate two exponentially weighted moving averages (EWMAs) with different memory lengths.

    Core estimator for Difference Momentum strategies, implementing the MACD-like comparison:

    .. math::
       1 - \\frac{E_2(\\mathbf{p}(t))}{E_1(\\mathbf{p}(t))}

    where E₁ and E₂ are EWMAs with memory lengths m₁ and m₂ respectively.
    Typically m₂ > m₁ for trend comparison.

    This function outputs the EWMAs for the two memory lengths used in the above formula.
    It supports both CPU and GPU paths, with GPU being more efficient for larger datasets.

    Parameters
    ----------
    memory_days_1 : float
        Memory length for first EWMA in days. Typically shorter period (m₁),
        making it more responsive to recent price changes.
    memory_days_2 : float
        Memory length for second EWMA in days. Typically longer period (m₂),
        providing baseline for trend comparison.
    chunkwise_price_values : jnp.ndarray
        Price/oracle values of shape (time, assets). Can be any oracle value,
        not just prices (see Notes).
    chunk_period : float
        Time period between chunks in minutes. Used to convert between
        memory_days and λ parameters.
    max_memory_days : float
        Maximum allowed memory length in days. Prevents numerical instability
        from extremely long memory periods.
    cap_lamb : bool, optional
        Whether to cap λ values to prevent numerical issues, by default True.
        Recommended for numerical stability.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        Two EWMA arrays of shape (time - 1, assets) (each same shape as calc_gradients).
     Notes
    -----
    Implementation details:
    1. Converts memory_days to λ values using:
       λ = memory_days_to_lamb(memory_days, chunk_period)
    
    2. GPU acceleration:
       - Uses convolution for parallel computation
       - Adds 5 * max_memory_days padding for initialization
       - Returns padded arrays for consistent calculations
    
    3. CPU computation:
       - Uses sequential scan operations
       - More memory efficient for smaller datasets

    The function ensures:
    - Non-negative memory lengths
    - Numerical stability through λ capping

    See Also
    --------
    DifferenceMomentumPool : Primary user of this function
    calc_gradients : Alternative trend estimation approach

    """

    # Ensure non-negative memory days
    memory_days_1 = jnp.maximum(memory_days_1, 0.0)
    memory_days_2 = jnp.maximum(memory_days_2, 0.0)

    # Convert to lambda values
    lamb_1 = jax_memory_days_to_lamb(memory_days_1, chunk_period)
    lamb_2 = jax_memory_days_to_lamb(memory_days_2, chunk_period)

    if cap_lamb:
        max_lamb = jax_memory_days_to_lamb(max_memory_days, chunk_period)
        lamb_1 = jnp.clip(lamb_1, a_min=0.0, a_max=max_lamb)
        lamb_2 = jnp.clip(lamb_2, a_min=0.0, a_max=max_lamb)

    # Ensure input is 2D
    chunkwise_price_values = jnp.atleast_2d(chunkwise_price_values)

    if DEFAULT_BACKEND != "cpu":
        safety_margin_max_memory_days = max_memory_days * 5.0

        # Create padding exactly as in calc_ewma_padded
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
        ewma_kernel_1 = make_ewma_kernel(lamb_1, safety_margin_max_memory_days, chunk_period)
        ewma_kernel_2 = make_ewma_kernel(lamb_2, safety_margin_max_memory_days, chunk_period)
        # Calculate EWMAs using same function as calc_ewma_padded
        ewma_1 = _jax_ewma_at_infinity_via_conv_padded(
            padded_chunkwise_price_values, ewma_kernel_1
        )
        ewma_2 = _jax_ewma_at_infinity_via_conv_padded(
            padded_chunkwise_price_values, ewma_kernel_2
        )
        ewma_1 = ewma_1[-(len(chunkwise_price_values) - 1):]
        ewma_2 = ewma_2[-(len(chunkwise_price_values) - 1):]

    else:
        # CPU path - no padding needed
        ewma_1 = _jax_ewma_at_infinity_via_scan(chunkwise_price_values, lamb_1)
        ewma_2 = _jax_ewma_at_infinity_via_scan(chunkwise_price_values, lamb_2)

    return ewma_1, ewma_2


def calc_return_variances(
    update_rule_parameter_dict,
    chunkwise_price_values,
    chunk_period,
    max_memory_days,
    cap_lamb,
):
    """Calculate time-weighted return variances for TFMM strategy implementation.

    Computes the variance of asset returns using exponentially weighted moving averages (EWMA),
    with support for both CPU and GPU acceleration via JAX. Essential for minimum variance
    portfolio calculations and related risk estimation.

    Parameters
    ----------
    update_rule_parameter_dict : dict
        Dictionary containing strategy parameters, supporting two parameterizations:
        1. Direct memory specification:
        - 'memory_days_1': Direct specification of memory length in days
        2. Logit parameterization:
        - 'logit_lamb': Controls memory length through logit transform
    chunkwise_price_values : ndarray
        Array of shape (time_steps, n_assets) containing price values for each asset
        over time
    chunk_period : float
        Time period between chunks in minutes
    max_memory_days : float
        Maximum allowed memory length in days, used to cap lambda parameters
    cap_lamb : bool
        Whether to apply maximum memory day restriction to lambda parameters

    Returns
    -------
    ndarray
        Array of shape (time_steps-1, n_assets) containing calculated variances.
        For each asset, represents the time-weighted variance of returns.

    Notes
    -----
    The function implements two calculation paths:

    1. GPU path: Uses convolution operations for efficient parallel computation
    - Pads input data to handle initialization
    - Creates specialized kernels for variance calculation
    - Combines EWMA and covariance kernels for efficient computation

    2. CPU path: Uses scan operations for sequential computation
    - More memory efficient for smaller datasets
    - Direct implementation of variance formula

    The variance calculation follows the methodology described in the TFMM litepaper,
    using exponential weighting to estimate return variances while avoiding look-ahead bias.

    See Also
    --------
    MinVariancePool : Primary user of this function
    calc_gradients : Calculates price gradients using similar EWMA methodology
    calc_ewma_pair : Calculates EWMA pairs for difference momentum strategies
    """

    # Determine which parameterization is being used
    if "memory_days_1" in update_rule_parameter_dict:
        # Direct memory_days parameterization
        memory_days = update_rule_parameter_dict["memory_days_1"]
        lamb = jax_memory_days_to_lamb(memory_days, chunk_period)
    else:
        # Original logit_lamb parameterization
        lamb = calc_lamb(update_rule_parameter_dict)

    if cap_lamb:
        max_lamb = memory_days_to_lamb(max_memory_days, chunk_period)
        lamb = jnp.clip(lamb, a_min=0.0, a_max=max_lamb)
    returns = jnp.diff(chunkwise_price_values, axis=0) / chunkwise_price_values[:-1]

    if DEFAULT_BACKEND != "cpu":
        safety_margin_max_memory_days = max_memory_days * 5.0
        cov_kernel = make_cov_kernel(lamb, safety_margin_max_memory_days, chunk_period)
        ewma_kernel = make_ewma_kernel(
            lamb, safety_margin_max_memory_days, chunk_period
        )

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

        ewma_returns_padded = _jax_ewma_at_infinity_via_conv_padded(
            padded_returns, ewma_kernel
        )
        variances = _jax_variance_at_infinity_via_conv(
            padded_returns, ewma_returns_padded[1:], cov_kernel, lamb
        )
        variances = variances[-(len(chunkwise_price_values) - 1) :]
    else:
        variances = _jax_variance_at_infinity_via_scan(returns, lamb)

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
