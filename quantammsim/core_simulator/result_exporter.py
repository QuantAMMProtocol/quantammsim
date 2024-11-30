import hashlib
import json
import os

import numpy as np
from jax import config

from quantammsim.core_simulator.param_utils import NumpyEncoder, dict_of_jnp_to_np

# again, this only works on startup!
config.update("jax_enable_x64", True)

np.seterr(all="raise")
np.seterr(under="print")

# TODO above is all from jax utils, tidy up required


def get_run_location(run_fingerprint):
    """
    Generates a unique run location string based on the provided run fingerprint.

    The function takes a dictionary representing the run fingerprint, converts it to a JSON string,
    and then computes its SHA-256 hash. The resulting hash is used to create a unique run location
    string prefixed with "run_".

    Args:
        run_fingerprint (dict): A dictionary representing the run fingerprint.

    Returns:
        str: A unique run location string based on the SHA-256 hash of the run fingerprint.
    """
    run_location = "run_" + str(
        hashlib.sha256(
            json.dumps(run_fingerprint, sort_keys=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
    )
    return run_location


def append_json(new_data, filename):
    """
    Append new data to a JSON file.

    This function reads the existing data from a JSON file, appends the new data to it,
    and then writes the updated data back to the file.

    Args:
        new_data (dict): The new data to be appended to the JSON file.
        filename (str): The path to the JSON file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    """
    with open(filename, "r+", encoding="utf-8") as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        file_data = json.loads(file_data)
        # Join new_data with file_data inside emp_details
        file_data.append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        dumped = json.dumps(file_data, cls=NumpyEncoder, sort_keys=True)
        json.dump(dumped, file, indent=4)


def append_list_json(new_data, filename):
    """
    Append new data to a JSON file.

    This function reads the existing data from a JSON file, appends the new data to it,
    and then writes the updated data back to the file.

    Args:
        new_data (list): The new data to be appended to the JSON file.
        filename (str): The path to the JSON file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.

    """
    with open(filename, "r+", encoding="utf-8") as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        file_data = json.loads(file_data)
        # Join new_data with file_data inside emp_details
        file_data += new_data
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        dumped = json.dumps(file_data, cls=NumpyEncoder, sort_keys=True)
        json.dump(dumped, file, indent=4)


def save_multi_params(
    run_fingerprint,
    params,
    test_objective,
    train_objective,
    objective,
    local_learning_rate,
    iterations_since_improvement,
    steps,
    sorted_tokens=True,
):
    """
    Save multiple parameter sets along with their associated metrics to a JSON file.

    Parameters
    ----------
    run_fingerprint : dict
        Dictionary containing run configuration details used to generate unique run location
    params : list
        List of parameter dictionaries to save
    test_objective : list
        List of objective values on test set for each parameter set
    train_objective : list
        List of objective values on training set for each parameter set
    objective : list
        List of overall objective values for each parameter set
    local_learning_rate : list
        List of learning rates used for each parameter set
    iterations_since_improvement : list
        List tracking iterations without improvement for each parameter set
    steps : list
        List of step counts for each parameter set
    sorted_tokens : bool, optional
        Whether tokens are sorted alphabetically, by default True

    Notes
    -----
    Saves the data to a JSON file at ./results/<run_hash>.json
    If file exists, appends new parameter sets to existing data
    Converts JAX arrays to numpy arrays before saving
    """
    run_location = "./results/" + get_run_location(run_fingerprint) + ".json"
    for i, param in enumerate(params):
        if param.get("subsidary_params") is not None:
            param["subsidary_params"] = [
                dict_of_jnp_to_np(sp) for sp in param["subsidary_params"]
            ]
        param["step"] = steps[i]
        param["test_objective"] = test_objective[i]
        param["train_objective"] = train_objective[i]
        param["objective"] = objective[i]
        param["hessian_trace"] = 0
        param["local_learning_rate"] = local_learning_rate[i]
        param["iterations_since_improvement"] = iterations_since_improvement[i]
        params[i] = dict_of_jnp_to_np(param)
    if sorted_tokens:
        run_fingerprint["alphabetic"] = True
    if os.path.isfile(run_location) is False:
        results = [run_fingerprint] + params
        dumped = json.dumps(results, cls=NumpyEncoder, sort_keys=True)
        with open(run_location, "w", encoding="utf-8") as json_file:
            json.dump(dumped, json_file, indent=4)
    else:
        append_list_json(params, run_location)


def save_params(
    run_fingerprint,
    params,
    step,
    test_objective,
    train_objective,
    objective,
    hess,
    local_learning_rate,
    iterations_since_improvement,
    sorted_tokens=True,
):
    """
    Save optimization parameters and results to a JSON file.

    Parameters
    ----------
    run_fingerprint : dict
        Dictionary containing run configuration details
    params : dict
        Dictionary of optimization parameters
    step : int
        Current optimization step count
    test_objective : float
        Objective function value on test data
    train_objective : float
        Objective function value on training data
    objective : float
        Overall objective function value
    hess : float
        Trace of the Hessian matrix
    local_learning_rate : float
        Current learning rate
    iterations_since_improvement : int
        Number of iterations without improvement
    sorted_tokens : bool, optional
        Whether tokens are sorted alphabetically, by default True

    Notes
    -----
    Saves the data to a JSON file at ./results/<run_hash>.json
    If file exists, appends new parameter set to existing data
    Converts JAX arrays to numpy arrays before saving
    """

    run_location = "./results/" + get_run_location(run_fingerprint) + ".json"
    params["subsidary_params"] = [
        dict_of_jnp_to_np(sp) for sp in params["subsidary_params"]
    ]
    params = dict_of_jnp_to_np(params)

    params["step"] = step
    params["test_objective"] = test_objective
    params["train_objective"] = train_objective
    params["objective"] = objective
    params["hessian_trace"] = hess
    params["local_learning_rate"] = local_learning_rate
    params["iterations_since_improvement"] = iterations_since_improvement
    if sorted_tokens:
        run_fingerprint["alphabetic"] = True
    if os.path.isfile(run_location) is False:
        dumped = json.dumps([run_fingerprint, params], cls=NumpyEncoder, sort_keys=True)
        with open(run_location, "w", encoding="utf-8") as json_file:
            json.dump(dumped, json_file)
    else:
        append_json(params, run_location)
