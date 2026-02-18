"""Parameter utilities for strategy parameterization, serialization, and loading.

This module handles the full lifecycle of strategy parameters:

- **Initialization**: ``init_params`` / ``init_params_singleton`` create parameter dicts
  from human-readable initial values (memory days, k per day) by converting to the
  internal reparameterized form (logit_lamb, log_k, etc.).
- **Reparameterization**: Functions like ``calc_lamb``, ``calc_alt_lamb``, ``squareplus``,
  and their inverses convert between human-interpretable values and the unconstrained
  spaces used for gradient-based optimization.
- **Serialization**: ``NumpyEncoder``, ``dict_of_jnp_to_np``, ``dict_of_jnp_to_list``,
  ``dict_of_np_to_jnp`` handle conversion between JAX arrays, NumPy arrays, and
  JSON-serializable Python types.
- **Loading**: ``load_or_init``, ``load``, ``load_manually``, ``retrieve_best`` load
  saved training checkpoints with various selection strategies (best train, best test,
  best-train-above-test-threshold, etc.).
- **Grid generation**: ``create_product_of_linspaces``, ``generate_params_combinations``
  produce parameter grids for heatmap evaluations.

The key reparameterizations are:

- **lambda (λ)**: EWMA decay factor in [0, 1], stored as ``logit_lamb = log(λ/(1-λ))``.
  Converted to/from human-readable ``memory_days`` via cubic-root inversion.
- **k**: Weight update aggressiveness, stored as ``log_k = log2(k / memory_days)``.
  This decouples scale from memory length.
- **squareplus**: Smooth, non-negative activation ``(x + sqrt(x² + 4)) / 2``,
  an algebraic (non-transcendental) replacement for softplus. Used for exponent params.

Notes
-----
The ``memory_days ↔ lambda`` conversion involves solving a cubic equation analytically.
Both NumPy (``memory_days_to_lamb``) and JAX (``jax_memory_days_to_lamb``) versions
exist; the NumPy version includes safe division guards for zero memory days, while the
JAX version relies on ``jnp.where`` for the zero case.
"""
import os
import json
import hashlib
from copy import deepcopy
from itertools import product

import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from jax import config

from quantammsim.training.hessian_trace import hessian_trace


def squareplus(x):
    """Algebraic (non-transcendental) replacement for softplus.

    Computes ``(x + sqrt(x² + 4)) / 2``, which maps R → R⁺ smoothly. Unlike softplus
    (``log(1 + exp(x))``), squareplus avoids transcendental functions and is thus
    cheaper to differentiate through and more JIT-friendly.

    Parameters
    ----------
    x : jnp.ndarray or float
        Input value(s).

    Returns
    -------
    jnp.ndarray or float
        Non-negative output(s), always > 0.

    References
    ----------
    Barron, J.T. (2021). "Squareplus: A Softplus-Like Algebraic Rectifier."
    arXiv:2112.11687.

    See Also
    --------
    inverse_squareplus : Inverse mapping R⁺ → R.
    """
    return lax.mul(0.5, lax.add(x, lax.sqrt(lax.add(lax.square(x), 4.0))))


# again, this only works on startup!
config.update("jax_enable_x64", True)

np.seterr(all="raise")
np.seterr(under="print")


def check_run_fingerprint(run_fingerprint):
    """
    Check that the run fingerprint is not malformed.

    Parameters
    ----------
    run_fingerprint : dict
        The run fingerprint to validate.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If weight_interpolation_period is greater than chunk_period.
    """
    assert (
        run_fingerprint["weight_interpolation_period"]
        <= run_fingerprint["chunk_period"]
    )
    protocol_fee_split = run_fingerprint.get("protocol_fee_split", 0.0)
    assert 0.0 <= protocol_fee_split <= 1.0

def default_set_or_get(dictionary, key, default, augment=True):
    """
    Retrieves the value for a given key from a dictionary. If the key does not exist,
    it sets the key to a default value and returns the default value.

    Parameters
    ----------
    dictionary : dict
        The dictionary to search for the key.
    key : str
        The key to look up in the dictionary.
    default : Any
        The default value to set and return if the key is not found.
    augment : bool, optional
        If True, the default value is added to the dictionary if the key is not found.
        Default is True.

    Returns
    -------
    Any
        The value associated with the key if it exists, otherwise the default value.
    """
    value = dictionary.get(key)
    if value is None:
        if augment:
            dictionary[key] = default
        return default

    return value


def default_set(dictionary, key, default):
    """
    Sets a default value for a given key in a dictionary if the key does not already exist.

    Parameters
    ----------
    dictionary : dict
        The dictionary to update.
    key : str
        The key to check in the dictionary.
    default : Any
        The default value to set if the key is not present.

    Returns
    -------
    None
    """
    value = dictionary.get(key)
    if value is None:
        dictionary[key] = default


def recursive_default_set(target_dict, default_dict):
    """
    Recursively sets default values in a target dictionary based on a default dictionary.

    Parameters
    ----------
    target_dict : dict
        The dictionary to update with default values.
    default_dict : dict
        The dictionary containing the default values.

    Returns
    -------
    None
    """
    for key, value in default_dict.items():
        if isinstance(value, dict):
            if key not in target_dict:
                target_dict[key] = {}
            recursive_default_set(target_dict[key], value)
        else:
            default_set(target_dict, key, value)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalar and array types.

    Extends ``json.JSONEncoder`` to serialize ``np.integer`` as ``int``,
    ``np.floating`` as ``float``, and ``np.ndarray`` as nested lists.
    Used when saving training checkpoints and run fingerprints to JSON.

    Examples
    --------
    >>> import json, numpy as np
    >>> json.dumps({"val": np.float64(0.5)}, cls=NumpyEncoder)
    '{"val": 0.5}'
    """

    def default(self, o):
        """
        Convert numpy types to Python native types.

        Parameters
        ----------
        o : Any
            The object to encode.

        Returns
        -------
        Any
            The encoded object.
        """
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def get_run_location(run_fingerprint):
    """
    Get the run location based on the run fingerprint.

    Parameters
    ----------
    run_fingerprint : dict
        The run fingerprint.

    Returns
    -------
    str
        The run location.
    """
    run_location = "run_" + str(
        hashlib.sha256(
            json.dumps(run_fingerprint, sort_keys=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
    )
    return run_location


def dict_of_jnp_to_np(dictionary):
    """
    Convert dictionary values from jax numpy arrays to numpy arrays.

    Parameters
    ----------
    dictionary : dict
        The dictionary to convert.

    Returns
    -------
    dict
        The converted dictionary.
    """
    for key in dictionary:
        if key != "subsidary_params":
            dictionary[key] = np.array(dictionary[key])
    return dictionary


def dict_of_jnp_to_list(dictionary):
    """
    Convert dictionary values from jax numpy arrays to lists.

    Parameters
    ----------
    dictionary : dict
        The dictionary to convert.

    Returns
    -------
    dict
        The converted dictionary.
    """
    for key in dictionary:
        if key != "subsidary_params":
            dictionary[key] = np.array(dictionary[key]).tolist()
    return dictionary


def dict_of_np_to_jnp(dictionary):
    """
    Convert dictionary values from numpy arrays to jax numpy arrays.

    Parameters
    ----------
    dictionary : dict
        The dictionary to convert.

    Returns
    -------
    dict
        The converted dictionary.
    """
    for key in dictionary:
        if key != "subsidary_params":
            dictionary[key] = jnp.array(dictionary[key])
    return dictionary


@jit
def lamb_to_memory(lamb):
    """Convert EWMA decay factor lambda to the effective memory length (unitless).

    The EWMA weighting kernel ``w_t = lambda^t * (1 - lambda)`` has a
    characteristic memory scale that grows with lambda.  This function
    inverts the cubic relationship used in quantammsim's parameterisation:

    .. math::

        \\text{memory} = 4 \\cdot \\sqrt[3]{\\frac{6 \\lambda}{(1 - \\lambda)^3}}

    To convert to days, use :func:`lamb_to_memory_days` which divides by
    ``2 * chunk_period / 1440``.

    Parameters
    ----------
    lamb : float or jnp.ndarray
        EWMA decay factor in (0, 1).

    Returns
    -------
    float or jnp.ndarray
        Unitless memory scale.

    See Also
    --------
    lamb_to_memory_days : Returns memory in days.
    memory_days_to_lamb : Inverse mapping (days -> lambda).
    """
    memory = jnp.cbrt(6 * lamb / ((1 - lamb) ** 3.0)) * 4.0
    return memory


def memory_days_to_lamb(memory_days, chunk_period=60):
    """
    Convert memory days to lambda value.

    Parameters
    ----------
    memory_days : float
        The memory days value.
    chunk_period : int, optional
        The chunk period. Default is 60.

    Returns
    -------
    float
        The lambda value.
    """

    scaled_memory_days = (1440.0 * memory_days / (2.0 * chunk_period)) ** 3 / 6.0

    smd = scaled_memory_days
    smd2 = scaled_memory_days**2
    smd3 = scaled_memory_days**3
    smd4 = scaled_memory_days**4

    numerator_1 = np.cbrt((np.sqrt(3 * (27 * smd4 + 4 * smd3)) - 9 * smd2))
    denominator_1 = np.cbrt(2) * 3 ** (2.0 / 3.0) * smd

    numerator_2 = np.cbrt((2 / 3))
    denominator_2 = numerator_1

    # Handle division by zero by checking denominator
    safe_div1 = np.divide(
        numerator_1,
        denominator_1,
        out=np.zeros_like(numerator_1),
        where=denominator_1 != 0,
    )
    safe_div2 = np.divide(
        np.broadcast_to(numerator_2, denominator_2.shape),
        denominator_2,
        out=np.zeros_like(np.broadcast_to(numerator_2, denominator_2.shape)),
        where=denominator_2 != 0,
    )
    lamb = np.nan_to_num(safe_div1 - safe_div2 + 1.0, nan=0.0, posinf=1.0, neginf=0.0)
    return np.where(memory_days == 0.0, 0.0, lamb)


def jax_memory_days_to_lamb(memory_days, chunk_period=60):
    """
    Convert memory days to lambda value using JAX operations.

    Parameters
    ----------
    memory_days : float
        The memory days value.
    chunk_period : int, optional
        The chunk period. Default is 60.

    Returns
    -------
    float
        The lambda value.
    """
    scaled_memory_days = (1440.0 * memory_days / (2.0 * chunk_period)) ** 3 / 6.0

    smd = scaled_memory_days
    smd2 = scaled_memory_days**2
    smd3 = scaled_memory_days**3
    smd4 = scaled_memory_days**4

    numerator_1 = jnp.cbrt((jnp.sqrt(3 * (27 * smd4 + 4 * smd3)) - 9 * smd2))
    denominator_1 = jnp.cbrt(2) * 3 ** (2.0 / 3.0) * smd

    numerator_2 = jnp.cbrt((2 / 3))
    denominator_2 = numerator_1

    lamb = numerator_1 / denominator_1 - numerator_2 / denominator_2 + 1.0

    return jnp.where(memory_days==0.0, 0.0, lamb)

def memory_days_to_logit_lamb(memory_days, chunk_period=60):
    """
    Convert memory days to logit lambda value.

    Parameters
    ----------
    memory_days : float
        The memory days value.
    chunk_period : int, optional
        The chunk period. Default is 60.

    Returns
    -------
    float
        The logit lambda value.
    """
    lamb = memory_days_to_lamb(memory_days, chunk_period)
    logit_lamb = jnp.log(lamb / (1 - lamb))
    return logit_lamb


@jit
def lamb_to_memory_days(lamb, chunk_period):
    """Convert EWMA decay factor lambda to effective memory in days.

    Applies :func:`lamb_to_memory` then rescales by ``2 * chunk_period / 1440``
    to convert from unitless memory to calendar days, accounting for the
    observation frequency.

    Parameters
    ----------
    lamb : float or jnp.ndarray
        EWMA decay factor in (0, 1).
    chunk_period : int
        Time between observations in minutes (e.g., 1440 for daily, 60 for hourly).

    Returns
    -------
    float or jnp.ndarray
        Effective memory in days.

    See Also
    --------
    lamb_to_memory : Unitless version.
    memory_days_to_lamb : Inverse mapping.
    lamb_to_memory_days_clipped : Clipped version with max_memory_days bound.
    """
    memory_days = jnp.cbrt(6 * lamb / ((1 - lamb) ** 3.0)) * 2 * chunk_period / 1440
    return memory_days

@jit
def logistic_func(x):
    """Standard logistic sigmoid: ``sigma(x) = exp(x) / (1 + exp(x))``.

    Maps R -> (0, 1).  Used to convert the unconstrained ``logit_lamb``
    parameter to the EWMA decay factor ``lambda`` in (0, 1).

    Parameters
    ----------
    x : float or jnp.ndarray
        Unconstrained input value(s).

    Returns
    -------
    float or jnp.ndarray
        Output in (0, 1).
    """
    return jnp.exp(x) / (1 + jnp.exp(x))


@jit
def jax_logit_lamb_to_lamb(logit_lamb):
    """
    Convert logit lambda to lambda value using JAX operations.

    Parameters
    ----------
    logit_lamb : float
        The logit lambda value.

    Returns
    -------
    float
        The lambda value between 0 and 1.
    """
    lamb = logistic_func(logit_lamb)
    return lamb


@jit
def lamb_to_memory_days_clipped(lamb, chunk_period, max_memory_days):
    """
    Convert lambda value to memory days, clipped to a maximum value.

    Parameters
    ----------
    lamb : float
        The lambda value.
    chunk_period : int
        The chunk period in minutes.
    max_memory_days : float
        The maximum allowed memory days.

    Returns
    -------
    float
        The clipped memory value in days.
    """
    memory_days = jnp.clip(
        lamb_to_memory_days(lamb, chunk_period), min=0.0, max=max_memory_days
    )
    return memory_days


def calc_lamb(update_rule_parameter_dict):
    """
    Calculate the lambda value from the given update rule parameter dictionary.

    Parameters
    ----------
    update_rule_parameter_dict : dict
        A dictionary containing the update rule parameters.
        Must include the key "logit_lamb".

    Returns
    -------
    float
        The calculated lambda value.

    Raises
    ------
    KeyError
        If "logit_lamb" key is not found in update_rule_parameter_dict.
    """
    if update_rule_parameter_dict.get("logit_lamb") is not None:
        logit_lamb = update_rule_parameter_dict["logit_lamb"]
        lamb = logistic_func(logit_lamb)
    else:
        raise KeyError("logit_lamb key not found in update_rule_parameter_dict")
    return lamb

def calc_lamb_from_index(update_rule_parameter_dict, logit_lamb_index):
    """
    Calculate the lambda value from the given update rule parameter dictionary and index.

    Parameters
    ----------
    update_rule_parameter_dict : dict
        A dictionary containing the update rule parameters.
        Must include the key "logit_lamb".
    logit_lamb_index : int
        The index of the logit lambda value to calculate.

    Returns
    -------
    float
        The calculated lambda value.

    Raises
    ------
    KeyError
        If "logit_lamb" key is not found in update_rule_parameter_dict.
    """
    if update_rule_parameter_dict.get("logit_lamb") is not None:
        logit_lamb = update_rule_parameter_dict["logit_lamb"][logit_lamb_index]
        lamb = logistic_func(logit_lamb)
    else:
        raise KeyError("logit_lamb key not found in update_rule_parameter_dict")
    return lamb

def calc_alt_lamb(update_rule_parameter_dict):
    """
    Calculate the alternative lambda value based on the provided update rule parameters.

    Parameters
    ----------
    update_rule_parameter_dict : dict
        A dictionary containing the update rule parameters.
        Must include keys:
        - "logit_lamb": The logit lambda value
        - "logit_delta_lamb": The logit delta lambda value

    Returns
    -------
    float
        The calculated alternative lambda value.

    Raises
    ------
    KeyError
        If "logit_lamb" or "logit_delta_lamb" is not found in update_rule_parameter_dict.
    """
    if update_rule_parameter_dict.get("logit_lamb") is not None:
        logit_lamb = update_rule_parameter_dict["logit_lamb"]
    else:
        raise KeyError("logit_lamb key not found in update_rule_parameter_dict")

    if update_rule_parameter_dict.get("logit_delta_lamb") is not None:
        logit_delta_lamb = update_rule_parameter_dict["logit_delta_lamb"]
    else:
        raise KeyError("logit_delta_lamb key not found in update_rule_parameter_dict")
    logit_alt_lamb = logit_delta_lamb + logit_lamb
    alt_lamb = logistic_func(logit_alt_lamb)
    return alt_lamb


def inverse_squareplus(y):
    """Inverse of the squareplus activation (JAX version).

    Given ``y = squareplus(x)``, recovers ``x = (y² - 1) / y``. Used to convert
    from a desired positive parameter value back to the unconstrained raw parameter
    for initialization.

    Parameters
    ----------
    y : float or jnp.ndarray
        Positive input value(s). Must be > 0 (domain of inverse squareplus).

    Returns
    -------
    jnp.ndarray
        Unconstrained value(s) that map to ``y`` under squareplus.

    See Also
    --------
    squareplus : Forward mapping R → R⁺.
    inverse_squareplus_np : NumPy version for non-JAX contexts.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    return lax.div(lax.sub(lax.square(y), 1.0), y)


def inverse_squareplus_np(y):
    """Inverse of the squareplus activation (NumPy version).

    Identical to ``inverse_squareplus`` but uses NumPy operations, suitable for
    use outside JAX-traced contexts (e.g., initialization, post-processing).

    Parameters
    ----------
    y : float or np.ndarray
        Positive input value(s).

    Returns
    -------
    float or np.ndarray
        Unconstrained value(s) that map to ``y`` under squareplus.

    See Also
    --------
    inverse_squareplus : JAX version.
    """
    return (y**2 - 1.0) / y

def get_raw_value(value):
    """Convert a desired parameter value to raw (log2) space.

    Many parameters (k, width, amplitude) use ``2^raw`` reparameterization so that
    the raw parameter can take any real value while the effective value is always
    positive. This function inverts that: ``raw = log2(value)``.

    Parameters
    ----------
    value : float
        Desired positive parameter value.

    Returns
    -------
    float
        Log2 of the input, for use as the raw parameter.

    See Also
    --------
    get_log_amplitude : Similar but divides by memory_days first.
    """
    return np.log2(value)


def get_log_amplitude(amplitude, memory_days):
    """Convert desired amplitude to raw log_amplitude parameter.

    The effective amplitude is ``2^log_amplitude * memory_days``, so to achieve a
    target amplitude: ``log_amplitude = log2(amplitude / memory_days)``.

    Parameters
    ----------
    amplitude : float
        Desired amplitude value.
    memory_days : float
        Memory length in days (used to decouple amplitude from memory scale).

    Returns
    -------
    float
        Raw log_amplitude parameter value.
    """
    return np.log2(amplitude / memory_days)


def init_params_singleton(
    initial_values_dict, n_tokens, n_subsidary_rules=0, chunk_period=60, log_for_k=True
):
    """Initialize a single parameter set from human-readable initial values.

    Converts intuitive values (memory_days, k_per_day, etc.) into the internal
    reparameterized form (logit_lamb, log_k, etc.) as 1-D JAX arrays of length
    ``n_tokens + n_subsidary_rules``.

    Parameters
    ----------
    initial_values_dict : dict
        Human-readable initial values. Required keys:
        - ``'initial_k_per_day'``: Weight update aggressiveness
        - ``'initial_memory_length'``: EWMA memory in days
        Optional keys:
        - ``'initial_memory_length_delta'``: Additional memory for alt lambda
        - ``'initial_weights_logits'``: Starting weight logits
        - ``'initial_log_amplitude'``: Channel amplitude (log2 scale)
        - ``'initial_raw_width'``: Channel width (log2 scale)
        - ``'initial_raw_exponents'``: Power exponents (squareplus space)
        - ``'initial_pre_exp_scaling'``: Pre-exponential scaling (logit space)
    n_tokens : int
        Number of assets in the pool.
    n_subsidary_rules : int, optional
        Number of subsidiary rules (for composite pools). Default is 0.
    chunk_period : int, optional
        Time between price observations in minutes. Default is 60.
    log_for_k : bool, optional
        If True, use ``log_k`` parameterization; if False, use linear ``k``.
        Default is True.

    Returns
    -------
    dict
        Parameter dict with keys: ``'log_k'`` (or ``'k'``), ``'logit_lamb'``,
        ``'logit_delta_lamb'``, ``'initial_weights_logits'``, ``'log_amplitude'``,
        ``'raw_width'``, ``'raw_exponents'``, ``'logit_pre_exp_scaling'``,
        ``'subsidary_params'``. All values are 1-D ``jnp.ndarray``.

    See Also
    --------
    init_params : Multi-set version with noise injection.
    """
    n_pool_members = n_tokens + n_subsidary_rules
    if log_for_k:
        log_k = jnp.array(
            [np.log2(initial_values_dict["initial_k_per_day"])] * n_pool_members
        )

    initial_lamb = memory_days_to_lamb(
        initial_values_dict["initial_memory_length"], chunk_period
    )

    logit_lamb_np = np.log(initial_lamb / (1.0 - initial_lamb))
    logit_lamb = jnp.array([logit_lamb_np] * n_pool_members)

    # lamb delta is the difference in lamb needed for
    # lamb + delta lamb to give a final memory length
    # of  initial_memory_length + initial_memory_length_delta
    if initial_values_dict.get("initial_memory_length_delta") is not None:
        initial_lamb_plus_delta_lamb = memory_days_to_lamb(
            initial_values_dict["initial_memory_length"]
            + initial_values_dict["initial_memory_length_delta"],
            chunk_period,
        )
        logit_lamb_plus_delta_lamb_np = np.log(
            initial_lamb_plus_delta_lamb / (1.0 - initial_lamb_plus_delta_lamb)
        )
        logit_delta_lamb_np = logit_lamb_plus_delta_lamb_np - logit_lamb_np
        logit_delta_lamb = jnp.array([logit_delta_lamb_np] * n_pool_members)
    else:
        logit_delta_lamb = jnp.array([0.0] * n_pool_members)

    if initial_values_dict.get("initial_weights_logits") is not None:
        if type(initial_values_dict.get("initial_weights_logits")) not in [
            np.array,
            jnp.array,
            list,
        ]:
            initial_weights_logits = jnp.array(
                [initial_values_dict["initial_weights_logits"]] * n_pool_members
            )
        else:
            initial_weights_logits = jnp.array(
                initial_values_dict["initial_weights_logits"]
            )
    else:
        initial_weights_logits = jnp.array([0.0] * n_pool_members)

    if initial_values_dict.get("initial_log_amplitude") is not None:
        log_amplitude = jnp.array(
            [initial_values_dict["initial_log_amplitude"]] * n_pool_members
        )
    else:
        log_amplitude = jnp.array([0.0] * n_pool_members)

    if initial_values_dict.get("initial_raw_width") is not None:
        raw_width = jnp.array([initial_values_dict["initial_raw_width"]] * n_pool_members)
    else:
        raw_width = jnp.array([0.0] * n_pool_members)

    if initial_values_dict.get("initial_raw_exponents") is not None:
        raw_exponents = jnp.array(
            [initial_values_dict["initial_raw_exponents"]] * n_pool_members
        )
    else:
        raw_exponents = jnp.array([0.0] * n_pool_members)

    if initial_values_dict.get("initial_pre_exp_scaling") is not None:
        logit_pre_exp_scaling_np = np.log(
            initial_values_dict["initial_pre_exp_scaling"]
            / (1.0 - initial_values_dict["initial_pre_exp_scaling"])
        )
        logit_pre_exp_scaling = jnp.array([[logit_pre_exp_scaling_np] * n_pool_members])
    else:
        logit_pre_exp_scaling = jnp.array([[0.0] * n_pool_members])

    if log_for_k:
        params = {
            "log_k": log_k,
            "logit_lamb": logit_lamb,
            "logit_delta_lamb": logit_delta_lamb,
            "initial_weights_logits": initial_weights_logits,
            "log_amplitude": log_amplitude,
            "raw_width": raw_width,
            "raw_exponents": raw_exponents,
            "logit_pre_exp_scaling": logit_pre_exp_scaling,
            "subsidary_params": [],
        }
    else:
        params = {
            "k": jnp.array([initial_values_dict["initial_k_per_day"]] * n_pool_members),
            "logit_lamb": logit_lamb,
            "logit_delta_lamb": logit_delta_lamb,
            "initial_weights_logits": initial_weights_logits,
            "log_amplitude": log_amplitude,
            "raw_width": raw_width,
            "raw_exponents": raw_exponents,
            "logit_pre_exp_scaling": logit_pre_exp_scaling,
            "subsidary_params": [],
        }
    return params


def fill_in_missing_values_from_init_singleton(
    params,
    initial_values_dict,
    n_tokens,
    n_subsidary_rules=0,
    chunk_period=60,
    log_for_k=True,
):
    """
    Fill in missing values in parameters from initial values.

    Parameters
    ----------
    params : dict
        The parameters dictionary to update.
    initial_values_dict : dict
        The initial values dictionary.
    n_tokens : int
        The number of tokens.
    n_subsidary_rules : int, optional
        The number of subsidary rules. Default is 0.
    chunk_period : int, optional
        The chunk period. Default is 60.
    log_for_k : bool, optional
        Whether to use log scale for k parameter. Default is True.

    Returns
    -------
    dict
        The updated parameters dictionary.
    """
    initial_params = init_params_singleton(
        initial_values_dict, n_tokens, n_subsidary_rules, chunk_period, log_for_k
    )
    for key, value in initial_params.items():
        if params.get(key) is None:
            params[key] = value
    return params


def init_params(
    initial_values_dict,
    n_tokens,
    n_subsidary_rules=0,
    chunk_period=60,
    n_parameter_sets=1,
    noise="gaussian",
):
    """Initialize multiple parameter sets from human-readable initial values.

    Creates ``n_parameter_sets`` copies of the base parameters. When
    ``n_parameter_sets > 1``, Gaussian noise is added to all rows except
    the first (which remains at the exact initial values). This is the
    legacy ensemble initialization method; for more control, see
    ``EnsembleAveragingHook``.

    Parameters
    ----------
    initial_values_dict : dict
        Human-readable initial values (same format as ``init_params_singleton``).
    n_tokens : int
        Number of assets in the pool.
    n_subsidary_rules : int, optional
        Number of subsidiary rules. Default is 0.
    chunk_period : int, optional
        Time between price observations in minutes. Default is 60.
    n_parameter_sets : int, optional
        Number of parameter sets (ensemble members). Default is 1.
    noise : str, optional
        Noise type for diversification. Only ``'gaussian'`` is supported.
        Default is ``'gaussian'``.

    Returns
    -------
    dict
        Parameter dict with 2-D arrays of shape ``(n_parameter_sets, n_pool_members)``
        for each parameter key.

    See Also
    --------
    init_params_singleton : Single parameter set initialization.
    """
    n_pool_members = n_tokens + n_subsidary_rules
    log_k = np.array(
        [[np.log2(initial_values_dict["initial_k_per_day"])] * n_pool_members]
        * n_parameter_sets
    )

    initial_lamb = memory_days_to_lamb(
        initial_values_dict["initial_memory_length"], chunk_period
    )

    logit_lamb_np = np.log(initial_lamb / (1.0 - initial_lamb))
    logit_lamb = np.array([[logit_lamb_np] * n_pool_members] * n_parameter_sets)

    # lamb delta is the difference in lamb needed for
    # lamb + delta lamb to give a final memory length
    # of  initial_memory_length + initial_memory_length_delta
    initial_lamb_plus_delta_lamb = memory_days_to_lamb(
        initial_values_dict["initial_memory_length"]
        + initial_values_dict["initial_memory_length_delta"],
        chunk_period,
    )

    logit_lamb_plus_delta_lamb_np = np.log(
        initial_lamb_plus_delta_lamb / (1.0 - initial_lamb_plus_delta_lamb)
    )
    logit_delta_lamb_np = logit_lamb_plus_delta_lamb_np - logit_lamb_np
    logit_delta_lamb = np.array(
        [[logit_delta_lamb_np] * n_pool_members] * n_parameter_sets
    )

    if type(initial_values_dict["initial_weights_logits"]) not in [
        np.array,
        jnp.array,
        list,
    ]:
        initial_weights_logits = np.array(
            [[initial_values_dict["initial_weights_logits"]] * n_pool_members]
            * n_parameter_sets
        )
    else:
        initial_weights_logits = np.array(
            [initial_values_dict["initial_weights_logits"]] * n_parameter_sets
        )
    log_amplitude = np.array(
        [[initial_values_dict["initial_log_amplitude"]] * n_pool_members]
        * n_parameter_sets
    )

    raw_width = np.array(
        [[initial_values_dict["initial_raw_width"]] * n_pool_members] * n_parameter_sets
    )

    raw_exponents = np.array(
        [[initial_values_dict["initial_raw_exponents"]] * n_pool_members]
        * n_parameter_sets
    )

    logit_pre_exp_scaling_np = np.log(
        initial_values_dict["initial_pre_exp_scaling"]
        / (1.0 - initial_values_dict["initial_pre_exp_scaling"])
    )
    logit_pre_exp_scaling = np.array(
        [[logit_pre_exp_scaling_np] * n_pool_members] * n_parameter_sets
    )

    params = {
        "log_k": log_k,
        "logit_lamb": logit_lamb,
        "logit_delta_lamb": logit_delta_lamb,
        "initial_weights_logits": initial_weights_logits,
        "log_amplitude": log_amplitude,
        "raw_width": raw_width,
        "raw_exponents": raw_exponents,
        "logit_pre_exp_scaling": logit_pre_exp_scaling,
        "subsidary_params": [],
    }

    if n_parameter_sets > 1:
        if noise == "gaussian":
            for key, value in params.items():
                if key != "subsidary_params":
                    # Leave first row of each jax parameter unaltered, add
                    # gaussian noise to subsequent rows.
                    value[1:] = value[1:] + np.random.randn(*value[1:].shape)
    for key, value in params.items():
        if key != "subsidary_params":
            params[key] = jnp.array(value)
    return params


def fill_in_missing_values_from_init(
    params,
    initial_values_dict,
    n_tokens,
    n_subsidary_rules=0,
    chunk_period=60,
    n_parameter_sets=1,
):
    """
    Fill in missing values in parameters from initial values.

    Parameters
    ----------
    params : dict
        The parameters dictionary to update.
    initial_values_dict : dict
        The initial values dictionary.
    n_tokens : int
        The number of tokens.
    n_subsidary_rules : int, optional
        The number of subsidary rules. Default is 0.
    chunk_period : int, optional
        The chunk period. Default is 60.
    n_parameter_sets : int, optional
        The number of parameter sets. Default is 1.

    Returns
    -------
    dict
        The updated parameters dictionary.
    """
    initial_params = init_params(
        initial_values_dict,
        n_tokens,
        n_subsidary_rules,
        chunk_period,
        n_parameter_sets=n_parameter_sets,
    )
    for key, value in initial_params.items():
        if params.get(key) is None:
            params[key] = value
    return params


def calc_hessian_from_loaded_params(params, partial_fixed_training_step):
    """
    Calculate the Hessian matrix from the loaded parameters.

    Parameters
    ----------
    params : dict
        A dictionary of parameters.
    partial_fixed_training_step : callable
        A function representing the partial fixed training step.

    Returns
    -------
    numpy.ndarray
        The Hessian matrix calculated from the loaded parameters.
    """
    params_local = deepcopy(params)
    if params_local.get("step") is not None:
        params_local.pop("step")
    if params_local.get("test_objective") is not None:
        params_local.pop("test_objective")
    if params_local.get("train_objective") is not None:
        params_local.pop("train_objective")
    return np.array(
        hessian_trace(
            dict_of_np_to_jnp(params_local), partial_fixed_training_step
        ).copy()
    )


def load_result_array(run_location, key="objective", recalc_hess=False):
    """
    Load simulation results from a JSON file and return run fingerprint and results array.

    Parameters
    ----------
    run_location : str
        Path to the JSON results file.
    key : str, optional
        Which value to extract from results. Default is "objective".
    recalc_hess : bool, optional
        Whether to recalculate Hessian trace values. Default is False.

    Returns
    -------
    tuple
        A tuple containing:
            run_fingerprint : dict
                Configuration details and metadata for the simulation run
            values : list
                Array of values extracted from results based on specified key
    """
    if os.path.isfile(run_location):
        with open(run_location, encoding='utf-8') as json_file:
            params = json.load(json_file)
            params = json.loads(params)
        if recalc_hess is True:
            if "hessian_trace" not in params[0].keys():
                for i, param in enumerate(params):
                    params[i]["hessian_trace"] = calc_hessian_from_loaded_params(
                        params[i]
                    )
                    print(
                        i,
                        "/",
                        len(params),
                        "  ",
                        i / len(params),
                        "htr: ",
                        params[i]["hessian_trace"],
                    )

        return params[0], [p[key] for p in params[1:]]


def _extract_objective_values(objectives, metric_key="returns_over_uniform_hodl"):
    """Extract numeric values from objectives that may be dicts or numbers.

    Handles both old format (list of numbers) and new format (list of metric dicts).

    Parameters
    ----------
    objectives : list
        List of objectives, where each objective is either:
        - A list of numbers (old format)
        - A list of dicts with metric keys (new format)
    metric_key : str
        The key to extract from dict objectives. Defaults to "return".
        For continuous_test_metrics, use keys like "continuous_test_return".

    Returns
    -------
    np.ndarray
        2D array of numeric objective values
    """
    if not objectives:
        return np.array([[float("-inf")]])

    result = []
    for obj_list in objectives:
        if isinstance(obj_list, (list, tuple)) and len(obj_list) > 0:
            if isinstance(obj_list[0], dict):
                # New format: list of metric dicts
                result.append([d.get(metric_key, float("-inf")) for d in obj_list])
            else:
                # Old format: list of numbers
                result.append(list(obj_list))
        else:
            # Single value or empty
            if isinstance(obj_list, dict):
                result.append([obj_list.get(metric_key, float("-inf"))])
            elif obj_list is not None:
                try:
                    result.append([float(obj_list)])
                except (TypeError, ValueError):
                    result.append([float("-inf")])
            else:
                result.append([float("-inf")])
    return np.array(result)


def _is_new_format(params):
    """Check if params use the new format with metric dicts.

    Returns True if train_objective contains dicts, False if it contains numbers.
    """
    if len(params) < 2:
        return False
    train_obj = params[1].get("train_objective")
    if isinstance(train_obj, (list, tuple)) and len(train_obj) > 0:
        return isinstance(train_obj[0], dict)
    return isinstance(train_obj, dict)


def get_objective_scalar(obj, metric_key="returns_over_uniform_hodl"):
    """Extract a scalar value from an objective that may be a dict or number.

    Use this when you have a single objective value (after retrieve_best has
    indexed into the parameter sets) and need a float.

    Parameters
    ----------
    obj : float, int, or dict
        The objective value - either a scalar (old format) or a dict of metrics (new format)
    metric_key : str
        The key to extract from dict objectives. Defaults to "returns_over_uniform_hodl".

    Returns
    -------
    float
        The scalar objective value

    Examples
    --------
    >>> get_objective_scalar(0.1)  # old format
    0.1
    >>> get_objective_scalar({"return": 0.1, "sharpe": 0.5})  # new format
    0.1
    """
    if isinstance(obj, dict):
        return float(obj.get(metric_key, float("-inf")))
    try:
        return float(obj)
    except (TypeError, ValueError):
        return float("-inf")


def _get_test_objectives(params, use_continuous=True, metric_key="returns_over_uniform_hodl"):
    """Get test objectives, preferring continuous_test_metrics if available.

    Parameters
    ----------
    params : list
        List of parameter dicts (including fingerprint at index 0)
    use_continuous : bool
        If True and continuous_test_metrics exists, use it instead of test_objective
    metric_key : str
        The metric key to extract (e.g., "return", "sharpe")

    Returns
    -------
    list
        Raw objectives list from params
    str
        The actual metric key to use (may be prefixed with "continuous_test_")
    """
    # Check if continuous_test_metrics is available
    if use_continuous and len(params) > 1:
        first_param = params[1]
        if "continuous_test_metrics" in first_param and first_param["continuous_test_metrics"]:
            # Use continuous test metrics - keys are prefixed with "continuous_test_"
            continuous_key = f"continuous_test_{metric_key}"
            return [p.get("continuous_test_metrics", []) for p in params[1:]], continuous_key

    # Fall back to test_objective
    return [p["test_objective"] for p in params[1:]], metric_key


def load_manually(
    run_location,
    load_method="last",
    recalc_hess=False,
    min_test=0.0,
    return_as_iterables=False,
    metric_key="returns_over_uniform_hodl",
    use_continuous_test=True,
):
    """Load and process parameter sets from a JSON results file with custom loading methods.

    Parameters
    ----------
    run_location : str
        Path to the JSON results file.
    load_method : str, optional
        Method for selecting parameter sets. One of:
        'last' - Returns the last parameter set
        'best_objective' - Returns set with highest overall objective
        'best_train_objective' - Returns set with highest training objective
        'best_test_objective' - Returns set with highest test objective
        'best_train_min_test_objective' - Returns set with highest training objective
        that meets minimum test threshold.
        Defaults to 'last'.
    recalc_hess : bool, optional
        Whether to recalculate Hessian trace values. Defaults to False.
    min_test : float, optional
        Minimum test objective threshold for 'best_train_min_test_objective' method.
        Defaults to 0.0.
    metric_key : str, optional
        For new format files with metric dicts, specifies which metric to use.
        Options include: "return", "sharpe", "jax_sharpe", "returns_over_hodl",
        "returns_over_uniform_hodl", "annualised_returns", "calmar", "sterling", "ulcer".
        Ignored for old format files with simple numeric objectives.
        Defaults to "returns_over_uniform_hodl".
    use_continuous_test : bool, optional
        If True and continuous_test_metrics is available, use it instead of
        test_objective for test-related load methods. Defaults to True.

    Returns
    -------
    tuple
        Two-element tuple containing:
        - dict: Loaded parameters
        - int: The index of the selected parameter set
    """
    if os.path.isfile(run_location):
        with open(run_location, encoding="utf-8") as json_file:
            params = json.load(json_file)
            params = json.loads(params)

        # Check if params length exceeds 1.5x the number of iterations
        if len(params) > 1.5 * params[0]["optimisation_settings"]["n_iterations"]:
            # Find last index where step == 0
            last_step_zero_idx = -1
            for i in range(len(params) - 1, 0, -1):
                if params[i].get("step", -1) == 0:
                    last_step_zero_idx = i
                    break

            # Keep only 0th row and rows from last step==0 onwards
            if last_step_zero_idx != -1:
                params = [params[0]] + params[last_step_zero_idx:]
        if recalc_hess is True:
            if "hessian_trace" not in params[0].keys():
                for i in range(len(params)):
                    params[i]["hessian_trace"] = calc_hessian_from_loaded_params(
                        params[i]
                    )

                dumped = json.dumps(params, cls=NumpyEncoder)

                with open(run_location, "w", encoding="utf-8") as json_file:
                    json.dump(dumped, json_file)

        # Helper to extract a single numeric value from an objective (handles old/new format)
        def _get_objective_value(obj, key=metric_key):
            if isinstance(obj, dict):
                return obj.get(key, float("-inf"))
            try:
                return float(obj)
            except (TypeError, ValueError):
                return float("-inf")

        if load_method == "last":
            index = -1
            context = None
        elif load_method == "best_objective":
            objectives = [p["objective"] for p in params[1:]]
            index = np.argmax(np.nanmax(objectives, axis=1)) + 1
            context = np.nanargmax(np.nanmax(objectives, axis=0))
        elif load_method == "best_train_objective":
            raw_objectives = [p["train_objective"] for p in params[1:]]
            objectives = _extract_objective_values(raw_objectives, metric_key)
            index = np.argmax(np.nanmax(objectives, axis=1)) + 1
            context = np.nanargmax(np.nanmax(objectives, axis=0))
        elif load_method == "best_train_objective_for_each_parameter_set":
            raw_objectives = [p["train_objective"] for p in params[1:]]
            objectives = _extract_objective_values(raw_objectives, metric_key)
            index = (np.nanargmax(objectives, axis=0) + 1).tolist()
            context = np.arange(len(objectives[0])).tolist()
        elif load_method == "best_test_objective":
            raw_objectives, actual_key = _get_test_objectives(params, use_continuous_test, metric_key)
            objectives = _extract_objective_values(raw_objectives, actual_key)
            index = np.argmax(np.nanmax(objectives, axis=1)) + 1
            context = np.nanargmax(np.nanmax(objectives, axis=0))
        elif load_method == "best_objective_of_last":
            objectives = [params[-1]["objective"]]
            index = -1
            context = np.nanargmax(np.nanmax(objectives))
        elif load_method == "best_train_objective_of_last":
            raw_objectives = [params[-1]["train_objective"]]
            objectives = _extract_objective_values(raw_objectives, metric_key)
            index = -1
            context = np.nanargmax(np.nanmax(objectives))
        elif load_method == "best_test_objective_of_last":
            raw_objectives, actual_key = _get_test_objectives(
                [params[0], params[-1]], use_continuous_test, metric_key
            )
            objectives = _extract_objective_values(raw_objectives, actual_key)
            index = -1
            context = np.nanargmax(np.nanmax(objectives))
        elif load_method == "best_train_min_test_objective":
            # Get test objectives (prefer continuous if available)
            raw_test_objs, test_key = _get_test_objectives(params, use_continuous_test, metric_key)

            # Filter params where best test objective meets threshold
            objectives = []
            for idx, p in enumerate(params[1:]):
                test_vals = _extract_objective_values([raw_test_objs[idx]], test_key)[0]
                if np.nanmax(test_vals) >= min_test:
                    objectives.append(p)

            train_objective_max = float("-inf")
            if len(objectives) == 0:
                objectives = params[1:]

            best_objective = objectives[0]
            set_with_best_test_index = 0
            num_param_sets = len(_extract_objective_values(
                [params[1]["train_objective"]], metric_key
            )[0])

            for p in objectives:
                train_vals = _extract_objective_values([p["train_objective"]], metric_key)[0]
                p_idx = params[1:].index(p) if p in params[1:] else 0
                test_vals = _extract_objective_values([raw_test_objs[p_idx]], test_key)[0]

                for i in range(num_param_sets):
                    test_val = test_vals[i] if i < len(test_vals) else float("-inf")
                    train_val = train_vals[i] if i < len(train_vals) else float("-inf")
                    if test_val >= min_test and train_val >= train_objective_max:
                        best_objective = p
                        set_with_best_test_index = i
                        train_objective_max = train_val

            if return_as_iterables:
                return [best_objective], [set_with_best_test_index]
            else:
                return best_objective, set_with_best_test_index
        elif load_method == "best_test_min_train_objective":
            # Get test objectives (prefer continuous if available)
            raw_test_objs, test_key = _get_test_objectives(params, use_continuous_test, metric_key)

            # Filter params where best train objective meets threshold
            objectives = []
            for p in params[1:]:
                train_vals = _extract_objective_values([p["train_objective"]], metric_key)[0]
                if np.nanmax(train_vals) >= min_test:
                    objectives.append(p)

            test_objective_max = float("-inf")
            if len(objectives) == 0:
                objectives = params[1:]

            best_objective = objectives[0]
            set_with_best_test_index = 0
            num_param_sets = len(_extract_objective_values(
                [params[1]["test_objective"]], metric_key
            )[0])

            for p in objectives:
                train_vals = _extract_objective_values([p["train_objective"]], metric_key)[0]
                p_idx = params[1:].index(p) if p in params[1:] else 0
                test_vals = _extract_objective_values([raw_test_objs[p_idx]], test_key)[0]

                for i in range(num_param_sets):
                    train_val = train_vals[i] if i < len(train_vals) else float("-inf")
                    test_val = test_vals[i] if i < len(test_vals) else float("-inf")
                    if train_val >= min_test and test_val >= test_objective_max:
                        best_objective = p
                        set_with_best_test_index = i
                        test_objective_max = test_val

            if return_as_iterables:
                return [best_objective], [set_with_best_test_index]
            else:
                return best_objective, set_with_best_test_index
            return best_objective, set_with_best_test_index
        else:
            raise NotImplementedError
        if return_as_iterables:
            if "for_each_parameter_set" not in load_method:
                return [params[index]], [context]
            else:
                return [params[i] for i in index], context
        else:
            return params[index], context


def retrieve_best(data_location, load_method, re_calc_hess, min_alt_obj=0.0, return_as_iterables=False):
    """Retrieve the best parameters from a training run.

    Loads parameters using the specified method and extracts the best
    parameter set based on the context (index of best performing parameters).
    Removes training metadata (step, hessian_trace, etc.) from the returned params.

    Parameters
    ----------
    data_location : str
        Path to the directory containing saved training results.
    load_method : str
        Method for loading parameters. Options include:
        - 'last': Load the most recent checkpoint
        - 'best_train_objective': Load checkpoint with best training objective
        - 'best_test_objective': Load checkpoint with best test objective
    re_calc_hess : bool
        Whether to recalculate hessian information when loading.
    min_alt_obj : float, optional
        Minimum alternative objective threshold. Defaults to 0.0.
    return_as_iterables : bool, optional
        If True, returns lists of all loaded params and steps.
        If False, returns only the first (best) params and step.
        Defaults to False.

    Returns
    -------
    params : dict or list of dict
        Best parameter dictionary (or list if return_as_iterables=True).
        Training metadata fields are removed.
    steps : int or list of int
        Training step(s) at which the parameters were saved.
    """
    params, contexts = load_manually(data_location, load_method, re_calc_hess, min_alt_obj, return_as_iterables=True)
    steps = []
    params_list = []
    for param, context in zip(params, contexts):
        steps.append(param["step"])
        params_list.append(param.copy())
        params_list[-1].pop("step")
        params_list[-1].pop("hessian_trace")
        params_list[-1].pop("local_learning_rate")
        params_list[-1].pop("iterations_since_improvement")
        for key in params_list[-1].keys():
            if key != "subsidary_params":
                params_list[-1][key] = params_list[-1][key][context]
    if return_as_iterables:
        return params_list, steps
    else:
        return params_list[0], steps[0]


def load_or_init(
    run_fingerprint,
    initial_values_dict,
    n_tokens,
    n_subsidary_rules,
    recalc_hess=False,
    chunk_period=60,
    force_init=False,
    load_method="last",
    n_parameter_sets=1,
    results_dir="./results/",
    partial_fixed_training_step=None,
):
    """
    Load or initialize parameters for the AMM simulator.

    Parameters
    ----------
    run_fingerprint : str
        The fingerprint of the run.
    initial_values_dict : dict
        The initial values dictionary.
    n_tokens : int
        The number of tokens.
    n_subsidary_rules : int
        The number of subsidiary rules.
    recalc_hess : bool, optional
        Whether to recalculate the Hessian. Default is False.
    chunk_period : int, optional
        The chunk period. Default is 60.
    force_init : bool, optional
        Whether to force initialization. Default is False.
    load_method : str, optional
        The method to use for loading. Default is "last".
    n_parameter_sets : int, optional
        The number of parameter sets. Default is 1.
    results_dir : str, optional
        The directory for results. Default is "./results/".
    partial_fixed_training_step : callable, optional
        The partial fixed training step. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
            params : dict
                The loaded or initialized parameters
            loaded : bool
                Whether the parameters were loaded (True) or initialized (False)
    """

    run_location = results_dir + get_run_location(run_fingerprint) + ".json"
    if force_init:
        print("force init")
        params = init_params(
            initial_values_dict,
            n_tokens,
            n_subsidary_rules,
            chunk_period,
            n_parameter_sets=n_parameter_sets,
        )
        loaded = False
    elif os.path.isfile(run_location):
        print("Loading from: ", run_location)
        print("found file")
        with open(run_location, encoding='utf-8') as json_file:
            params = json.load(json_file)
            # if params:
            #     calc()
            params = json.loads(params)
            print("params")
            print(params)
        if recalc_hess is True:
            if "hessian_trace" not in params[0].keys():
                for i in range(len(params)):
                    params[i]["hessian_trace"] = calc_hessian_from_loaded_params(
                        params[i], partial_fixed_training_step
                    )
                    print(
                        i,
                        "/",
                        len(params),
                        "  ",
                        i / len(params),
                        "htr: ",
                        params[i]["hessian_trace"],
                    )
                dumped = json.dumps(params, cls=NumpyEncoder)
                with open(run_location, "w", encoding='utf-8') as json_file:
                    json.dump(dumped, json_file, indent=4)

        if isinstance(params, list):
            params = [
                fill_in_missing_values_from_init(
                    p,
                    initial_values_dict,
                    n_tokens,
                    n_subsidary_rules,
                    chunk_period,
                    n_parameter_sets=n_parameter_sets,
                )
                for p in params
            ]
        else:
            params = fill_in_missing_values_from_init(
                params,
                initial_values_dict,
                n_tokens,
                n_subsidary_rules,
                chunk_period,
                n_parameter_sets=n_parameter_sets,
            )

        if load_method == "last":
            index = -1
        elif load_method == "best_objective":
            objectives = [p["objective"] for p in params[1:]]
            index = np.argmax(np.max(objectives, axis=0)) + 1
        else:
            raise NotImplementedError
        params = dict_of_np_to_jnp(params[index])
        params["subsidary_params"] = [
            dict_of_np_to_jnp(sp) for sp in params["subsidary_params"]
        ]
        loaded = True
    else:
        # if n_parameter_sets == 1:
        #     params = init_params_singleton(
        #         initial_values_dict,
        #         n_tokens,
        #         n_subsidary_rules,
        #         chunk_period)
        # else:
        params = init_params(
            initial_values_dict,
            n_tokens,
            n_subsidary_rules,
            chunk_period,
            n_parameter_sets=n_parameter_sets,
        )
        loaded = False
    return params, loaded


def load(
    run_location,
    initial_values_dict,
    n_tokens,
    n_subsidary_rules,
    chunk_period=60,
    load_method="last",
    n_parameter_sets=1,
):
    """
    Load parameters from a file and fill in missing values based on initial values.

    Parameters
    ----------
    run_location : str
        The location of the file containing the parameters.
    initial_values_dict : dict
        A dictionary of initial values.
    n_tokens : int
        The number of tokens.
    n_subsidary_rules : int
        The number of subsidiary rules.
    chunk_period : int, optional
        The chunk period. Default is 60.
    load_method : {'last', 'best_objective', 'best_train_objective'}, optional
        The method to use for loading parameters. Default is 'last'.
    n_parameter_sets : int, optional
        The number of parameter sets. Default is 1.

    Returns
    -------
    tuple
        A tuple containing:
            params : dict
                The loaded parameters
            context : int or None
                The context index for the loaded parameters

    Raises
    ------
    FileNotFoundError
        If the run_location file does not exist.
    NotImplementedError
        If an unsupported load_method is specified.
    """

    if os.path.isfile(run_location):
        with open(run_location, encoding='utf-8') as json_file:
            params = json.load(json_file)
            params = json.loads(params)
        if isinstance(params, list):
            params = [
                fill_in_missing_values_from_init(
                    p,
                    initial_values_dict,
                    n_tokens,
                    n_subsidary_rules,
                    chunk_period,
                    n_parameter_sets=n_parameter_sets,
                )
                for p in params
            ]
        else:
            params = fill_in_missing_values_from_init(
                params,
                initial_values_dict,
                n_tokens,
                n_subsidary_rules,
                chunk_period,
                n_parameter_sets=n_parameter_sets,
            )
        if load_method == "last":
            index = -1
            context = None
        elif load_method == "best_objective":
            objectives = [p["objective"] for p in params[1:]]
            index = np.argmax(np.nanmax(objectives, axis=1)) + 1
            context = np.argmax(np.nanmax(objectives, axis=0))
        elif load_method == "best_train_objective":
            objectives = [p["train_objective"] for p in params[1:]]
            index = np.argmax(np.nanmax(objectives, axis=1)) + 1
            context = np.argmax(np.nanmax(objectives, axis=0))
        else:
            raise NotImplementedError
        params = dict_of_np_to_jnp(params[index])
        params["subsidary_params"] = [
            dict_of_np_to_jnp(sp) for sp in params["subsidary_params"]
        ]
    else:
        raise FileNotFoundError(f"File not found: {run_location}")
    return params, context


def make_composite_run_params(
    composite_params,
    list_of_subsidary_pool_run_dicts,
    initial_values_dict,
    n_parameter_sets,
):
    """
    Create composite run parameters for the AMM simulator.

    Parameters
    ----------
    composite_params : dict
        The composite parameters for the AMM simulator.
    list_of_subsidary_pool_run_dicts : list
        A list of dictionaries containing the parameters for each subsidiary pool run.
    initial_values_dict : dict
        The initial values dictionary for the AMM simulator.
    n_parameter_sets : int
        The number of parameter sets.

    Returns
    -------
    dict
        The composite run parameters for the AMM simulator.
    """
    params = deepcopy(composite_params)
    params["subsidary_params"] = []
    for sub in list_of_subsidary_pool_run_dicts:
        local_initial_values_dict = deepcopy(initial_values_dict)
        local_initial_values_dict["initial_memory_length"] = sub[
            "initial_memory_length"
        ]
        local_initial_values_dict["initial_k_per_day"] = sub["initial_k_per_day"]
        local_n_tokens = len(sub["tokens"])
        params["subsidary_params"].append(
            init_params(
                local_initial_values_dict,
                local_n_tokens,
                n_parameter_sets=n_parameter_sets,
            )
        )
    return params


def create_product_of_linspaces(
    params, keys_ranges, num_points_per_key, inverse_funcs=None
):
    """
    Create a product of linspaces for chosen keys in the params dict.

    Parameters
    ----------
    params : dict
        The dictionary containing initial parameter values.
    keys_ranges : dict
        The dictionary containing high and low values for each key.
    num_points_per_key : dict
        The dictionary containing the number of points for each key.
    inverse_funcs : dict, optional
        A dictionary of inverse functions for each key.

    Returns
    -------
    list
        A list of dictionaries with all combinations of linspace values for the chosen keys.
    """

    # Create linspaces for each key
    linspaces = {}
    for key, (low, high) in keys_ranges.items():
        num_points = num_points_per_key.get(
            key, 10
        )  # Default to 10 points if not specified
        linspace = np.linspace(low, high, num_points)
        if inverse_funcs and key in inverse_funcs:
            linspace = inverse_funcs[key](linspace)
        linspaces[key] = linspace

    # Create the product of linspaces
    linspace_product = list(product(*linspaces.values()))

    # Create a list of dictionaries with all combinations of linspace values
    param_combinations = []
    for values in linspace_product:
        param_combination = params.copy()
        for i, key in enumerate(keys_ranges.keys()):
            param_combination[key] = values[i]
        param_combinations.append(param_combination)

    return param_combinations


def create_product_of_arrays(params, keys_arrays):
    """
    Create a product of arrays for chosen keys in the params dict.

    Parameters
    ----------
    params : dict
        The dictionary containing initial parameter values.
    keys_arrays : dict
        The dictionary containing the points for each key.

    Returns
    -------
    list
        A list of dictionaries with all combinations of linspace values for the chosen keys.
    """

    # Create the product of linspaces
    key_product = list(product(*keys_arrays.values()))

    # Create a list of dictionaries with all combinations of linspace values
    param_combinations = []
    for values in key_product:
        param_combination = params.copy()
        for i, key in enumerate(keys_arrays.keys()):
            param_combination[key] = values[i]
        param_combinations.append(param_combination)

    return param_combinations


def generate_run_fingerprint_combinations(
    run_fingerprint,
    keys_ranges=None,
    num_points_per_key=None,
    inverse_funcs=None,
):
    """
    Generate run fingerprint combinations with specified ranges and scaling.

    Parameters
    ----------
    run_fingerprint : dict
        The base run fingerprint.
    keys_ranges : dict, optional
        The dictionary containing high and low values for each key.
        Defaults to logarithmic ranges for 'arb_frequency'.
    num_points_per_key : dict, optional
        The dictionary containing the number of points for each key.
        Defaults to 10 points for each key.
    inverse_funcs : dict, optional
        A dictionary of inverse functions for each key.
        Defaults to logarithmic scaling for 'arb_frequency'.

    Returns
    -------
    list
        A list of dictionaries with all combinations of run fingerprint values.
    """

    # Default keys ranges
    if keys_ranges is None:
        keys_ranges = {
            # "arb_fees": (np.log(0.001), np.log(0.1)),  # Example logarithmic range
            "arb_frequency": (np.log(1), np.log(100)),  # Example logarithmic range
        }

    # Default number of points for each key
    if num_points_per_key is None:
        num_points_per_key = {key: 10 for key in keys_ranges}

    # # Default inverse functions for logarithmic scaling
    # if inverse_funcs is None:
    #     inverse_funcs = {key: log_scale for key in keys_ranges}

    # Generate run fingerprint combinations
    run_fingerprint_combinations = create_product_of_linspaces(
        run_fingerprint.copy(), keys_ranges, num_points_per_key, inverse_funcs
    )

    return run_fingerprint_combinations


def make_log_range_with_zero(x):
    """
    Compute the exponential of a given value, with a special case for zero.

    Parameters
    ----------
    x : float
        The input value for which the exponential is to be computed.

    Returns
    -------
    float
        The exponential of the input value `x`, or zero if `x` is zero.
    """
    if x == 0:
        return 0
    else:
        return np.exp(x)


def combine_param_combinations(param_combinations, n_parameter_sets):
    """
    Combine single-row jnp arrays in param_combinations into multi-row jnp arrays.

    Parameters
    ----------
    param_combinations : list
        List of dictionaries with single-row jnp arrays.
    n_parameter_sets : int
        Number of parameter sets to combine into each dictionary.

    Returns
    -------
    list
        List of dictionaries with multi-row jnp arrays.
    """

    def combine_subsidary_params(subsidary_params_list):
        combined_subsidary_params = []
        for i in range(0, len(subsidary_params_list), n_parameter_sets):
            batch = subsidary_params_list[i : i + n_parameter_sets]
            combined_params = {}
            if len(batch[0]) > 0:
                for key in batch[0].keys():
                    combined_params[key] = jnp.stack([params[key] for params in batch])
                combined_subsidary_params.append(combined_params)
        return combined_subsidary_params

    combined_params_list = []

    for i in range(0, len(param_combinations), n_parameter_sets):
        batch = param_combinations[i : i + n_parameter_sets]
        combined_params = {}

        for key in batch[0].keys():
            if key == "subsidary_params":
                combined_params[key] = combine_subsidary_params(
                    [params[key] for params in batch]
                )
            else:
                combined_params[key] = jnp.stack([params[key] for params in batch])

        combined_params_list.append(combined_params)

    return combined_params_list


def split_param_combinations(param_combinations):
    """
    Split multi-row jnp arrays in param_combinations into single-row jnp arrays.

    Parameters
    ----------
    param_combinations : list
        List of dictionaries with multi-row jnp arrays.

    Returns
    -------
    list
        List of dictionaries with single-row jnp arrays.
    """

    def split_subsidary_params(subsidary_params_dict):
        split_subsidary_params = []
        keys = [k for k in subsidary_params_dict.keys()]
        for i in range(len(subsidary_params_dict[keys[0]])):
            split_params = {k: subsidary_params_dict[k][i] for k in keys}
            split_subsidary_params.append(split_params)
        return split_subsidary_params

    split_params_list = []

    for dict_ in param_combinations:
        keys = [k for k in dict_.keys()]
        for i in range(len(dict_[keys[0]])):
            split_dict = {}
            for key in keys:
                if key == "subsidary_params" or key == "rule_outputs_dict":
                    split_dict[key] = split_subsidary_params(dict_[key])
                else:
                    split_dict[key] = dict_[key][i]
            split_params_list.append(split_dict)

    return split_params_list


def make_vmap_in_axes_dict(
    input_dict, in_axes, keys_to_recur_on, keys_with_no_vamp=[], n_repeats_of_recurred=0
):
    """Create a ``vmap`` in_axes specification dict matching a parameter dict structure.

    Constructs the nested dict/list structure that ``jax.vmap`` expects for its
    ``in_axes`` argument when vectorizing over a dict of parameters. Handles
    recursive structure for subsidiary parameters.

    Parameters
    ----------
    input_dict : dict
        Parameter dictionary whose structure to mirror.
    in_axes : int
        Axis to vectorize over (typically 0 for the parameter-set dimension).
    keys_to_recur_on : list of str
        Keys (e.g., ``'subsidary_params'``) that contain nested parameter dicts
        requiring recursive axis specification.
    keys_with_no_vamp : list of str, optional
        Keys that should not be vectorized (axis set to None). Default is ``[]``.
    n_repeats_of_recurred : int, optional
        Number of subsidiary parameter dicts. Default is 0.

    Returns
    -------
    dict
        Nested dict matching the structure of ``input_dict`` with integer axes
        or None for each leaf.
    """

    in_axes_dict = dict()
    for key, _ in input_dict.items():
        in_axes_dict[key] = in_axes
    for key in keys_to_recur_on:
        in_axes_dict[key] = [
            make_vmap_in_axes_dict(
                input_dict,
                in_axes,
                [],
                keys_with_no_vamp=["subsidary_params"],
                n_repeats_of_recurred=0,
            )
        ] * n_repeats_of_recurred
    for key in keys_with_no_vamp:
        in_axes_dict[key] = None
    return in_axes_dict


def generate_params_combinations(
    initial_values_dict,
    n_tokens,
    n_subsidary_rules,
    chunk_period,
    n_parameter_sets,
    k_per_day_range,
    memory_days_range,
    num_points_k_per_day=10,
    num_points_memory_days=10,
):
    """
    Generate parameter combinations with linearly-spaced values of k_per_day and memory_days.

    Args:
        initial_values_dict (dict): The initial values dictionary.
        n_tokens (int): The number of tokens.
        n_subsidary_rules (int): The number of subsidary rules.
        chunk_period (int): The chunk period.
        n_parameter_sets (int): The number of parameter sets.
        k_per_day_range (tuple): The range (low, high) for k_per_day.
        memory_days_range (tuple): The range (low, high) for memory_days.
        num_points_k_per_day (int, optional): The number of points for k_per_day linspace. Defaults to 10.
        num_points_memory_days (int, optional): The number of points for memory_days linspace. Defaults to 10.

    Returns:
        list: A list of dictionaries with all combinations of parameter values.
    """
    # Initialize base params
    # base_params = init_params_singleton(
    #     initial_values_dict, n_tokens, n_subsidary_rules, chunk_period
    # )

    # Define keys ranges for linspace generation
    keys_ranges = {
        "initial_k_per_day": k_per_day_range,
        "initial_memory_length": memory_days_range,
    }

    # Define number of points for each key
    num_points_per_key = {
        "initial_k_per_day": num_points_k_per_day,
        "initial_memory_length": num_points_memory_days,
    }

    # Generate param combinations
    initial_values_dict_combinations = create_product_of_linspaces(
        initial_values_dict.copy(), keys_ranges, num_points_per_key
    )

    # Fill in missing values from initial values
    filled_param_combinations = [
        fill_in_missing_values_from_init_singleton(
            {},
            i_v_d,
            n_tokens,
            n_subsidary_rules,
            chunk_period,
            n_parameter_sets,
        )
        for i_v_d in initial_values_dict_combinations
    ]
    return filled_param_combinations, initial_values_dict_combinations


def process_initial_values(
    initial_values_dict, key, n_assets, n_parameter_sets, force_scalar=False
):
    """Extract and broadcast a parameter value to the correct shape.

    Handles flexible input formats: scalar (broadcast to all assets and sets),
    per-asset vector (broadcast across sets), or full matrix. Used by the
    schema-aware initialization path.

    Parameters
    ----------
    initial_values_dict : dict
        Dictionary containing initial parameter values.
    key : str
        Parameter name to extract.
    n_assets : int
        Number of assets (columns).
    n_parameter_sets : int
        Number of parameter sets / ensemble members (rows).
    force_scalar : bool, optional
        If True, treat value as a scalar even if it's array-like, producing
        shape ``(n_parameter_sets,)`` instead of ``(n_parameter_sets, n_assets)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_parameter_sets, n_assets)`` or ``(n_parameter_sets,)``
        if ``force_scalar=True``.

    Raises
    ------
    ValueError
        If ``key`` is not in ``initial_values_dict`` or has incompatible shape.
    """
    if key in initial_values_dict:
        initial_value = initial_values_dict[key]
        if isinstance(initial_value, (np.ndarray, jnp.ndarray, list)):
            initial_value = np.array(initial_value)
            if force_scalar:
                return np.array([initial_value] * n_parameter_sets)
            elif initial_value.size == n_assets:
                return np.array([initial_value] * n_parameter_sets)
            elif initial_value.size == 1:
                return np.array([[initial_value] * n_assets] * n_parameter_sets)
            elif initial_value.shape == (n_parameter_sets, n_assets):
                return initial_value
            else:
                raise ValueError(
                    f"{key} must be a singleton or a vector of length n_assets or a matrix of shape (n_parameter_sets, n_assets)"
                )
        else:
            if force_scalar:
                return np.array([initial_value] * n_parameter_sets)
            else:
                return np.array([[initial_value] * n_assets] * n_parameter_sets)
    else:
        raise ValueError(f"initial_values_dict must contain {key}")


def _to_float64_list(value):
    """Convert JAX/numpy array to list of float64."""
    if isinstance(value, (jnp.ndarray, np.ndarray)):
        return [float(x) for x in np.array(value).flatten()]
    elif isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    else:
        return [float(value)]


def _to_bd18_string_list(values):
    """Convert list of floats to list of 18 fixed point integer strings.

    Uses string manipulation to avoid overflow from multiplication by 1e18.
    Formats each value with 18 decimal places, then removes the decimal point
    and strips leading zeros.

    Accepts arrays (including 0-d), scalars, lists, and tuples — consistent
    with :func:`_to_float64_list`.
    """
    # Normalise to an iterable of scalars (handles 0-d arrays and bare floats)
    if isinstance(values, (jnp.ndarray, np.ndarray)):
        values = np.array(values).flatten()
    elif not isinstance(values, (list, tuple)):
        values = [values]
    result = []
    for x in values:
        # Format with 18 decimal places, then remove decimal point
        formatted = f"{x:.18f}"
        # Split on decimal point
        if '.' in formatted:
            int_part, frac_part = formatted.split('.')
            # Pad fractional part to exactly 18 digits if needed
            frac_part = frac_part.ljust(18, '0')[:18]
            combined = int_part + frac_part
        else:
            # No decimal point, just append 18 zeros
            combined = formatted + '0' * 18
        # Strip leading zeros, but keep at least one digit (handle zero case)
        stripped = combined.lstrip('0')
        result.append(stripped if stripped else '0')
    return result


def convert_parameter_values(params, run_fingerprint, max_memory_days=None):
    """Convert raw (reparameterized) parameters to human-readable and on-chain formats.

    Applies the inverse reparameterizations (logit → lambda → memory_days, log2 → k,
    squareplus → exponents, etc.) and produces both float64 values and BD18 fixed-point
    string representations suitable for on-chain deployment.

    Parameters
    ----------
    params : dict
        Raw parameter dictionary (e.g., ``'logit_lamb'``, ``'log_k'``, ``'raw_exponents'``).
    run_fingerprint : dict
        Run configuration, must include ``'chunk_period'``.
    max_memory_days : float, optional
        Maximum memory days for lambda clipping. If None, uses
        ``run_fingerprint['max_memory_days']`` (default 365).

    Returns
    -------
    dict
        ``{'values': {...}, 'strings': {...}}`` where each inner dict maps
        human-readable parameter names (``'lamb'``, ``'k'``, ``'exponents'``,
        ``'width'``, ``'amplitude'``, ``'pre_exp_scaling'``) to lists.
        ``'values'`` contains float64 lists; ``'strings'`` contains BD18
        (18-decimal fixed-point integer) string representations.

    Notes
    -----
    BD18 format multiplies the float value by 10^18 and represents as an integer
    string, matching the Solidity ``uint256`` representation used by the on-chain
    QuantAMM contracts. The conversion uses string manipulation to avoid float64
    overflow from direct multiplication by 1e18.
    """
    result = {"values": {}, "strings": {}}
    memory_days = None  # Keep track of computed memory_days for reuse
    if max_memory_days is None:
        max_memory_days = run_fingerprint.get("max_memory_days", 365)

    if "logit_lamb" in params:
        memory_days = lamb_to_memory_days_clipped(
            calc_lamb(params),
            chunk_period=run_fingerprint["chunk_period"],
            max_memory_days=max_memory_days,
        )

        lamb = calc_lamb(params)
        lamb_list = _to_float64_list(lamb)
        result["values"]["lamb"] = lamb_list
        result["strings"]["lamb"] = _to_bd18_string_list(lamb_list)

        if "log_k" in params:
            k = 2 ** params["log_k"] * memory_days
            k_list = _to_float64_list(k)
            result["values"]["k"] = k_list
            result["strings"]["k"] = _to_bd18_string_list(k_list)
        elif "k" in params:
            k = params["k"] * memory_days
            k_list = _to_float64_list(k)
            result["values"]["k"] = k_list
            result["strings"]["k"] = _to_bd18_string_list(k_list)

    if "raw_exponents" in params:
        exponents = squareplus(params["raw_exponents"])
        exponents_list = _to_float64_list(exponents)
        result["values"]["exponents"] = exponents_list
        result["strings"]["exponents"] = _to_bd18_string_list(exponents_list)

    if "raw_width" in params:
        width = 2 ** params["raw_width"]
        width_list = _to_float64_list(width)
        result["values"]["width"] = width_list
        result["strings"]["width"] = _to_bd18_string_list(width_list)

    if "log_amplitude" in params:
        # Recompute memory_days if not already computed
        if memory_days is None:
            memory_days = lamb_to_memory_days_clipped(
                calc_lamb(params),
                chunk_period=run_fingerprint["chunk_period"],
                max_memory_days=max_memory_days,
            )
        amplitude = (2 ** params["log_amplitude"]) * memory_days
        amplitude_list = _to_float64_list(amplitude)
        result["values"]["amplitude"] = amplitude_list
        result["strings"]["amplitude"] = _to_bd18_string_list(amplitude_list)

    if "logit_pre_exp_scaling" in params:
        pre_exp_scaling = jnp.exp(params["logit_pre_exp_scaling"]) / (
            1 + jnp.exp(params["logit_pre_exp_scaling"])
        )
        pes_list = _to_float64_list(pre_exp_scaling)
        result["values"]["pre_exp_scaling"] = pes_list
        result["strings"]["pre_exp_scaling"] = _to_bd18_string_list(pes_list)

    if "raw_pre_exp_scaling" in params:
        pre_exp_scaling = 2 ** params["raw_pre_exp_scaling"]
        pes_list = _to_float64_list(pre_exp_scaling)
        result["values"]["pre_exp_scaling"] = pes_list
        result["strings"]["pre_exp_scaling"] = _to_bd18_string_list(pes_list)

    return result

    # print("-" * 80)
    # print("final readouts")
    # for readout in result["readouts"]:
    #     print(
    #         f"{readout}: { jnp.array_str(result['readouts'][readout][-1], precision=16, suppress_small=False)}"
    #     )
    # print("-" * 80)
    # print("final weights")
    # print(
    #     f"{jnp.array_str(result['weights'][-1], precision=16, suppress_small=False)}"
    # )
    # print("-" * 80)
    # print("final prices")
    # print(
    #     f"{jnp.array_str(result['prices'][-1], precision=16, suppress_small=False)}"
    # )
    # print("=" * 80)
