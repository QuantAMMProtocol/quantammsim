import os

# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

import numpy as np

import json
import hashlib

from quantammsim.core_simulator.param_utils import dict_of_jnp_to_np, NumpyEncoder

np.seterr(all="raise")
np.seterr(under="print")

# TODO above is all from jax utils, tidy up required


def get_run_location(run_fingerprint):
    run_location = "run_" + str(
        hashlib.sha256(
            json.dumps(run_fingerprint, sort_keys=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
    )
    return run_location


def append_json(new_data, filename):
    with open(filename, "r+") as file:
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
    with open(filename, "r+") as file:
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
    for i in range(len(params)):
        if params[i].get("subsidary_params") is not None:
            params[i]["subsidary_params"] = [
                dict_of_jnp_to_np(sp) for sp in params[i]["subsidary_params"]
            ]
        params[i]["step"] = steps[i]
        params[i]["test_objective"] = test_objective[i]
        params[i]["train_objective"] = train_objective[i]
        params[i]["objective"] = objective[i]
        params[i]["hessian_trace"] = 0
        params[i]["local_learning_rate"] = local_learning_rate[i]
        params[i]["iterations_since_improvement"] = iterations_since_improvement[i]
        params[i] = dict_of_jnp_to_np(params[i])
    if sorted_tokens:
        run_fingerprint["alphabetic"] = True
    if os.path.isfile(run_location) is False:
        results = [run_fingerprint] + params
        dumped = json.dumps(results, cls=NumpyEncoder, sort_keys=True)
        with open(run_location, "w") as json_file:
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
        with open(run_location, "w") as json_file:
            json.dump(dumped, json_file)
    else:
        append_json(params, run_location)
