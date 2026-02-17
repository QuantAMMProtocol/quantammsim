import hashlib
import json
import os

import numpy as np

from quantammsim.core_simulator.param_utils import NumpyEncoder, dict_of_jnp_to_np

np.seterr(all="raise")
np.seterr(under="print")

# TODO above is all from jax utils, tidy up required


def get_run_location(run_fingerprint):
    """
    Generates a unique identifier string based on the provided run fingerprint.

    The function takes a dictionary representing the run fingerprint, converts it to a JSON string,
    and then computes its SHA-256 hash. The resulting hash is used to create a unique identifier
    string with a "run\_" prefix.

    Parameters
    ----------
    run_fingerprint : dict
        A dictionary representing the run fingerprint.

    Returns
    -------
    str
        A unique identifier string formatted as "run\_" followed by a SHA-256 hash
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
    continuous_test_metrics=None,
    validation_metrics=None,
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
        List of objective values/metrics on test set for each parameter set
    train_objective : list
        List of objective values/metrics on training set for each parameter set
    objective : list
        List of overall objective values for each parameter set
    local_learning_rate : list
        List of learning rates used for each parameter set
    iterations_since_improvement : list
        List tracking iterations without improvement for each parameter set
    steps : list
        List of step counts for each parameter set
    continuous_test_metrics : list, optional
        List of continuous test metrics for each parameter set
    validation_metrics : list, optional
        List of validation metrics for each parameter set (when using val_fraction > 0)
    sorted_tokens : bool, optional
        Whether tokens are sorted alphabetically, by default True

    Notes
    -----
    Saves the data to a JSON file at ``./results/run_<sha256_hash>.json`` where the hash
    is generated from the run_fingerprint using SHA-256.
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
        if continuous_test_metrics is not None:
            param["continuous_test_metrics"] = continuous_test_metrics[i]
        if validation_metrics is not None:
            param["validation_metrics"] = validation_metrics[i]
        params[i] = dict_of_jnp_to_np(param)
    if sorted_tokens:
        run_fingerprint["alphabetic"] = True
    if os.path.isfile(run_location) is False:
        results = [run_fingerprint] + params
        dumped = json.dumps(results, cls=NumpyEncoder, sort_keys=True)
        os.makedirs(os.path.dirname(run_location), exist_ok=True)
        with open(run_location, "w", encoding="utf-8") as json_file:
            json.dump(dumped, json_file, indent=4)
    else:
        append_list_json(params, run_location)


def save_optuna_results_sgd_format(
    run_fingerprint,
    study,
    n_assets,
    sorted_tokens=True,
):
    """
    Save optuna study results in the same format as SGD training results.

    This allows optuna-optimized parameters to be loaded and analyzed with
    the same tools used for SGD-trained parameters.

    Parameters
    ----------
    run_fingerprint : dict
        Dictionary containing run configuration details
    study : optuna.Study
        Completed optuna study object
    n_assets : int
        Number of assets in the pool (needed to reconstruct array params)
    sorted_tokens : bool, optional
        Whether tokens are sorted alphabetically, by default True

    Notes
    -----
    Saves to ``./results/run_<sha256_hash>.json`` in the same format as
    save_multi_params, allowing unified result analysis.
    """
    import optuna

    run_location = "./results/" + get_run_location(run_fingerprint) + ".json"

    # Get all completed trials
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        return  # Nothing to save

    params_list = []
    for trial in completed_trials:
        # Convert flattened optuna params (log_k_0, log_k_1) to arrays (log_k: [v0, v1])
        param_dict = _optuna_params_to_arrays(trial.params, n_assets)

        # Add metadata in SGD format
        param_dict["step"] = trial.number
        param_dict["test_objective"] = float(trial.user_attrs.get("validation_value", float("-inf")))
        param_dict["train_objective"] = float(trial.user_attrs.get("train_value", float("-inf")))
        param_dict["objective"] = float(trial.value) if trial.value is not None else float("-inf")
        param_dict["hessian_trace"] = 0  # Not applicable for optuna
        param_dict["local_learning_rate"] = 0  # Not applicable for optuna
        param_dict["iterations_since_improvement"] = 0  # Not applicable for optuna

        # Add additional optuna-specific metrics
        param_dict["optuna_trial_number"] = trial.number
        param_dict["validation_sharpe"] = float(trial.user_attrs.get("validation_sharpe", float("-inf")))
        param_dict["validation_return"] = float(trial.user_attrs.get("validation_return", float("-inf")))
        param_dict["train_sharpe"] = float(trial.user_attrs.get("train_sharpe", float("-inf")))
        param_dict["train_return"] = float(trial.user_attrs.get("train_return", float("-inf")))
        param_dict["validation_returns_over_hodl"] = float(
            trial.user_attrs.get("validation_returns_over_hodl", float("-inf"))
        )
        param_dict["train_returns_over_hodl"] = float(
            trial.user_attrs.get("train_returns_over_hodl", float("-inf"))
        )

        # Convert any remaining jax arrays to numpy
        param_dict = dict_of_jnp_to_np(param_dict)
        params_list.append(param_dict)

    if sorted_tokens:
        run_fingerprint["alphabetic"] = True

    # Mark as optuna-trained for downstream analysis
    run_fingerprint["training_method"] = "optuna"

    if os.path.isfile(run_location) is False:
        results = [run_fingerprint] + params_list
        dumped = json.dumps(results, cls=NumpyEncoder, sort_keys=True)
        os.makedirs(os.path.dirname(run_location), exist_ok=True)
        with open(run_location, "w", encoding="utf-8") as json_file:
            json.dump(dumped, json_file, indent=4)
    else:
        append_list_json(params_list, run_location)

    return run_location


def _optuna_params_to_arrays(optuna_params, n_assets):
    """
    Convert flattened optuna params to array format.

    Optuna stores params as: {'log_k_0': v0, 'log_k_1': v1, ...}
    This converts to: {'log_k': [v0, v1, ...]}

    Parameters
    ----------
    optuna_params : dict
        Flattened parameter dictionary from optuna trial
    n_assets : int
        Number of assets to expect

    Returns
    -------
    dict
        Parameter dictionary with arrays
    """
    import jax.numpy as jnp
    import re

    result = {}
    # Group params by base name
    base_names = set()
    for key in optuna_params.keys():
        # Match patterns like "log_k_0", "logit_lamb_1", etc.
        match = re.match(r"^(.+)_(\d+)$", key)
        if match:
            base_names.add(match.group(1))
        else:
            # Scalar param - keep as is
            result[key] = optuna_params[key]

    # Convert indexed params to arrays
    for base_name in base_names:
        values = []
        for i in range(n_assets):
            key = f"{base_name}_{i}"
            if key in optuna_params:
                values.append(float(optuna_params[key]))
            else:
                # Missing index - this shouldn't happen but handle gracefully
                break

        if values:
            result[base_name] = jnp.array(values)

    return result


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
    Saves the data to a JSON file at ``./results/run_<sha256_hash>.json`` where the hash
    is generated from the run_fingerprint using SHA-256.
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
