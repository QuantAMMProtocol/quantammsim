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
    """
    Special json encoder for numpy types.

    Methods
    -------
    default(obj)
        Convert numpy types to Python native types.
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
    """
    Convert lambda value to memory.

    Parameters
    ----------
    lamb : float
        The lambda value.

    Returns
    -------
    float
        The memory value.
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
    """
    Convert lambda value to memory days.

    Parameters
    ----------
    lamb : float
        The lambda value.
    chunk_period : int
        The chunk period in minutes.

    Returns
    -------
    float
        The memory value in days.
    """
    memory_days = jnp.cbrt(6 * lamb / ((1 - lamb) ** 3.0)) * 2 * chunk_period / 1440
    return memory_days


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
    lamb = jnp.exp(logit_lamb) / (1 + jnp.exp(logit_lamb))
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
        lamb_to_memory_days(lamb, chunk_period), a_min=0.0, a_max=max_memory_days
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
        lamb = jnp.exp(logit_lamb) / (1 + jnp.exp(logit_lamb))
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
    alt_lamb = jnp.exp(logit_alt_lamb) / (1 + jnp.exp(logit_alt_lamb))
    return alt_lamb


def inverse_squareplus(y):
    """
    Calculate the inverse of the squareplus function.

    Parameters
    ----------
    y : float or array_like
        Input value(s).

    Returns
    -------
    float or array_like
        The inverse squareplus of the input.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    return lax.div(lax.sub(lax.square(y), 1.0), y)


def inverse_squareplus_np(y):
    """
    Calculate the inverse of the squareplus function using numpy.

    Parameters
    ----------
    y : float or array_like
        Input value(s).

    Returns
    -------
    float or array_like
        The inverse squareplus of the input.
    """
    return (y**2 - 1.0) / y

def get_raw_value(value):
    """
    Get raw_value parameter from desired value.

    Parameters
    ----------
    value : float
        The input value.

    Returns
    -------
    float
        The log2 of the input value.
    """
    return np.log2(value)


def get_log_amplitude(amplitude, memory_days):
    """
    Get log_amplitude parameter from desired amplitude and memory_days.

    Parameters
    ----------
    amplitude : float
        The amplitude value.
    memory_days : float
        The memory days value.

    Returns
    -------
    float
        The log2 of amplitude divided by memory_days.
    """
    return np.log2(amplitude / memory_days)


def init_params_singleton(
    initial_values_dict, n_tokens, n_subsidary_rules=0, chunk_period=60, log_for_k=True
):
    """
    Initialize parameters for a singleton.

    Parameters
    ----------
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
        The initialized parameters.
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
    """
    Initialize parameters.

    Parameters
    ----------
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
    noise : str, optional
        The type of noise to add. Default is "gaussian".

    Returns
    -------
    dict
        The initialized parameters.
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


def load_manually(run_location, load_method="last", recalc_hess=False, min_test=0.0):
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

    Returns
    -------
    tuple
        Two-element tuple containing:
        - dict: Loaded parameters
        - int: The index of the selected parameter set
    """
    if os.path.isfile(run_location):
        with open(run_location, encoding='utf-8') as json_file:
            params = json.load(json_file)
            params = json.loads(params)
        if recalc_hess is True:
            if "hessian_trace" not in params[0].keys():
                for i in range(len(params)):
                    params[i]["hessian_trace"] = calc_hessian_from_loaded_params(
                        params[i]
                    )

                dumped = json.dumps(params, cls=NumpyEncoder)

                with open(run_location, "w", encoding='utf-8') as json_file:
                    json.dump(dumped, json_file)

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
        elif load_method == "best_test_objective":
            objectives = [p["test_objective"] for p in params[1:]]
            index = np.argmax(np.nanmax(objectives, axis=1)) + 1
            context = np.argmax(np.nanmax(objectives, axis=0))
        elif load_method == "best_train_min_test_objective":
            objectives = []
            for p in params[1:]:
                if p["test_objective"][np.argmax(p["test_objective"])] >= min_test:
                    objectives.append(p)
            train_objective_max = 0.0
            if len(objectives) == 0:
                objectives = params[1:]

            best_objective = objectives[0]
            set_with_best_test_index = 0
            num_param_sets = len(params[1]["train_objective"])

            for p in objectives:
                for i in range(num_param_sets):
                    if (
                        p["test_objective"][i] >= min_test
                        and p["train_objective"][i] >= train_objective_max
                    ):
                        best_objective = p
                        set_with_best_test_index = i
                        train_objective_max = p["train_objective"][i]
            return best_objective, set_with_best_test_index
        elif load_method == "best_test_min_train_objective":
            objectives = []
            for p in params[1:]:
                if p["train_objective"][np.argmax(p["train_objective"])] >= min_test:
                    objectives.append(p)
            test_objective_max = 0.0
            best_objective = objectives[0]
            set_with_best_test_index = 0
            num_param_sets = len(params[1]["test_objective"])

            for p in objectives:
                for i in range(num_param_sets):
                    if (
                        p["train_objective"][i] >= min_test
                        and p["test_objective"][i] >= test_objective_max
                    ):
                        best_objective = p
                        set_with_best_test_index = i
                        test_objective_max = p["test_objective"][i]
            return best_objective, set_with_best_test_index
        else:
            raise NotImplementedError
        return params[index], context


def retrieve_best(data_location, load_method, re_calc_hess, min_alt_obj = 0.0):
    param, context = load_manually(data_location,load_method, re_calc_hess, min_alt_obj)
    step = param["step"]
    param.pop("step")
    param.pop("hessian_trace")
    param.pop("local_learning_rate")
    param.pop("iterations_since_improvement")
    
    for key in param.keys():
        if key != "subsidary_params":
            param[key] = param[key][context]

    return param, step


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
        params = init_params(
            initial_values_dict,
            n_tokens,
            n_subsidary_rules,
            chunk_period,
            n_parameter_sets=n_parameter_sets,
        )
        loaded = False
    elif os.path.isfile(run_location):
        with open(run_location, encoding='utf-8') as json_file:
            params = json.load(json_file)
            # if params:
            #     calc()
            params = json.loads(params)
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
                if key == "subsidary_params" or key == "raw_weight_outputs_dict":
                    split_dict[key] = split_subsidary_params(dict_[key])
                else:
                    split_dict[key] = dict_[key][i]
            split_params_list.append(split_dict)

    return split_params_list


def make_vmap_in_axes_dict(
    input_dict, in_axes, keys_to_recur_on, keys_with_no_vamp=[], n_repeats_of_recurred=0
):
    """
    Create a dictionary specifying vmap axes for input parameters.

    Parameters
    ----------
    input_dict : dict
        Dictionary of input parameters.
    in_axes : int
        The axis to vectorize over.
    keys_to_recur_on : list
        Keys in input_dict that should be recursively processed.
    keys_with_no_vamp : list, optional
        Keys that should not be vectorized. Defaults to [].
    n_repeats_of_recurred : int, optional
        Number of times to repeat recursion. Defaults to 0.

    Returns
    -------
    dict
        Dictionary mapping parameter keys to their vmap axes specifications.
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
