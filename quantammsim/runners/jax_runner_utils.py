import numpy as np
import pandas as pd

import json
import hashlib

# again, this only works on startup!
from jax import config, jit
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp

from quantammsim.core_simulator.windowing_utils import (
    raw_fee_like_amounts_to_fee_like_array,
    raw_trades_to_trade_array,
)

from quantammsim.apis.rest_apis.simulator_dtos.simulation_run_dto import (
    LiquidityPoolCoinDto,
    SimulationResultTimestepDto,
)

config.update("jax_enable_x64", True)

import os
import optuna
import logging
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from optuna.visualization import plot_optimization_history, plot_param_importances
import numpy as np


from typing import Dict, Any, Generic, TypeVar
T = TypeVar('T')      # Declare type variable

def create_trial_params(
    trial: Any,  # optuna.Trial, but avoid direct dependency
    param_config: Dict,
    params: Dict,
    run_fingerprint: Dict,
    n_assets: int,
    expand_around=False
) -> Dict:
    """
    Create trial parameters for Optuna optimization.
    
    Parameters:
    -----------
    trial : optuna.Trial
        The Optuna trial object
    param_config : dict
        Configuration for parameter optimization. Each parameter can have:
        - low: float, lower bound
        - high: float, upper bound
        - log_scale: bool, whether to use log scale
        - scalar: bool, whether to use same value for all assets
    params : dict
        Current parameter values, used for shape information
    run_fingerprint : dict
        Run configuration
    n_assets : int
        Number of assets
        
    Returns:
    --------
    dict
        Trial parameters dictionary
    
    Raises:
    -------
    ValueError
        If parameter shapes are invalid or required config is missing
    """
    trial_params = {}
    for key, value in params.items():
        if key == "subsidary_params":
            continue

        # Verify value has correct shape
        if not hasattr(value, 'shape') or len(value.shape) < 2:
            raise ValueError(f"Parameter {key} must have at least 2 dimensions")

        param_length = value.shape[1]

        config = param_config.get(key, {})
        # Set defaults while preserving any existing config
        if expand_around:
            default_config = {
                    "low": 0.1,
                    "high": 0.1,
                    "log_scale": False,
                    "scalar": False
                }
        else:
            default_config = {
                "low": -10.0,
                "high": 10.0,
                "log_scale": False,
                "scalar": False
            }
        config = {**default_config, **config}
        # Handle logit_delta_lamb parameters
        if key.startswith("logit_delta_lamb") and not run_fingerprint.get(
            "use_alt_lamb", False
        ):
            trial_params[key] = jnp.zeros(param_length)
            continue

        # Handle initial_weights_logits specially
        if key == "initial_weights_logits":
            trial_params[key] = jnp.zeros(n_assets)
            continue
        if key == "initial_weights":
            trial_params[key] = value
            continue

        # Handle scalar vs vector parameters
        if config["scalar"]:
            # Create single value and repeat
            param_value = trial.suggest_float(
                key,  # Use key directly for scalar params
                config["low"],
                config["high"],
                log=config["log_scale"],
            )
            trial_params[key] = jnp.full(param_length, param_value)
        else:
            # Create array of different values
            trial_params[key] = jnp.array(
                [
                    trial.suggest_float(
                        f"{key}_{i}",
                        (
                            config["low"]
                            if not expand_around
                            else float(params[key][0][i]) - config["low"]
                        ),
                        (
                            config["high"]
                            if not expand_around
                            else float(params[key][0][i]) + config["high"]
                        ),
                        log=config["log_scale"],
                    )
                    for i in range(param_length)
                ]
            )
    return trial_params

def generate_evaluation_points(
    start_idx, end_idx, bout_length, n_points, min_spacing, random_key=0
):
    """Generate evaluation start points for optuna-style hyperparameter search.

    If the training period is exactly equal to bout_length (no room for multiple
    windows), returns just the start_idx as a single evaluation point.

    Parameters
    ----------
    start_idx : int
        Start index of the training period
    end_idx : int
        End index of the training period
    bout_length : int
        Length of each evaluation window
    n_points : int
        Desired number of evaluation points
    min_spacing : int
        Minimum spacing between evaluation points (currently unused)
    random_key : int
        Random seed for reproducibility

    Returns
    -------
    list
        List of evaluation start indices
    """
    np.random.seed(random_key)
    available_range = end_idx - start_idx - bout_length

    # Handle edge case where training period equals bout_length
    if available_range <= 0:
        # Only one evaluation point possible: the start of the training period
        return [start_idx]

    # Generate random points
    points = np.random.randint(0, available_range, n_points)
    points = np.sort(points)  # Sort for better coverage

    # Generate equally spaced points
    equal_points = np.linspace(0, available_range, n_points, dtype=int)

    # Combine with random points and sort
    all_points = np.concatenate([points, equal_points])
    all_points = np.unique(all_points)

    # Convert to absolute indices
    evaluation_starts = [start_idx + p for p in all_points]
    return evaluation_starts


def find_best_balanced_solution(values_array, n_objectives=None):
    """Find the solution closest to the ideal point after normalizing objectives.

    Args:
        values_array: Either a numpy array of shape (n_trials, n_objectives) or
                     a list of optuna trials with values attribute
        n_objectives: Number of objectives. Only needed if using list of trials.

    Returns:
        int: Index of the best balanced solution
    """
    if not isinstance(values_array, np.ndarray):
        # Convert list of trials to numpy array
        values_array = np.array([t.values for t in values_array])

    if n_objectives is None:
        n_objectives = values_array.shape[1]

    normalized = (values_array - values_array.min(axis=0)) / (
        values_array.max(axis=0) - values_array.min(axis=0)
    )

    # Find solution closest to ideal point
    ideal_point = np.ones(n_objectives)
    distances = np.linalg.norm(normalized - ideal_point, axis=1)
    best_idx = np.argmin(distances)

    return best_idx


def get_best_balanced_solution(study):
    trials = study.best_trials
    # Normalize each objective to [0,1]
    
    # Use the helper function
    if len(trials) > 1:
        best_idx = find_best_balanced_solution(trials, len(study.directions))
    else:
        best_idx = 0
    return trials[best_idx].params, trials[best_idx].values, best_idx


class OptunaManager:
    def __init__(self, run_fingerprint):
        self.run_fingerprint = run_fingerprint
        self.optuna_settings = run_fingerprint["optimisation_settings"]["optuna_settings"]
        self.output_dir = Path(run_fingerprint["optimisation_settings"]["optuna_settings"].get("output_dir", "optuna_studies"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.study = None
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Configure logging for the optimization process."""
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        logger = logging.getLogger(f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.output_dir / "optimization.log")
        fh.setLevel(logging.INFO)

        # Console handler
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)

        logger.addHandler(fh)

        return logger

    def setup_study(self, multi_objective=False):
        """Create and configure the Optuna study."""
        self.multi_objective = multi_objective
        study_name = (
            self.optuna_settings["study_name"]
            or f"quantamm_{self.run_fingerprint['rule']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Setup storage
        storage = None
        if self.optuna_settings["storage"]["url"]:
            storage = optuna.storages.RDBStorage(
                url=self.optuna_settings["storage"]["url"]
            )

        # Custom sampler with multivariate TPE
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.optuna_settings["n_startup_trials"], multivariate=True
        )

        # Custom pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=1,
        )

        if multi_objective:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                pruner=pruner,
                sampler=sampler,
                directions=["maximize", "maximize", "maximize"],
            )
        else:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
                direction="maximize",
            )

    def early_stopping_callback(self, study, trial):
        """Enhanced callback to implement early stopping using both training and validation metrics."""
        if not self.optuna_settings["early_stopping"]["enabled"]:
            return

        patience = self.optuna_settings["early_stopping"]["patience"]
        min_improvement = self.optuna_settings["early_stopping"]["min_improvement"]

        if len(study.trials) < patience:
            return

        # Get best validation value up to current trial
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            return

        validation_values = [
            t.user_attrs.get("validation_value", float("-inf"))
            for t in completed_trials
        ]
        best_validation = max(validation_values)

        recent_trials = completed_trials[-patience:]
        recent_best_validation = max(
            t.user_attrs.get("validation_value", float("-inf")) for t in recent_trials
        )

        relative_improvement = (recent_best_validation - best_validation) / abs(
            best_validation
        )

        if relative_improvement < min_improvement:
            self.logger.info(
                f"Stopping study: No validation improvement > {min_improvement} "
                f"in last {patience} trials"
            )
            study.stop()

    def save_results(self):
        """Enhanced save_results to include validation metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = self.output_dir / f"study_{timestamp}"
        study_dir.mkdir(parents=True, exist_ok=True)

        # Check if any trials completed
        completed_trials = [
            t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        has_completed_trials = len(completed_trials) > 0

        # Save study statistics
        if self.multi_objective:
            if has_completed_trials:
                pareto_front_trials = self.study.best_trials  # Returns list of all non-dominated trials
                best_balanced_params, best_balanced_values, best_balanced_idx = get_best_balanced_solution(self.study)
                stats = {
                    "best_params": [trial.params for trial in pareto_front_trials],
                    "best_values": [trial.values for trial in pareto_front_trials],
                    "n_trials": len(self.study.trials),
                    "n_completed_trials": len(completed_trials),
                    "datetime": timestamp,
                    "run_fingerprint": self.run_fingerprint,
                    "best_balanced_params": best_balanced_params,
                    "best_balanced_values": best_balanced_values,
                    "best_balanced_idx": int(best_balanced_idx),
                }
            else:
                stats = {
                    "best_params": None,
                    "best_values": None,
                    "n_trials": len(self.study.trials),
                    "n_completed_trials": 0,
                    "datetime": timestamp,
                    "run_fingerprint": self.run_fingerprint,
                    "error": "No trials completed successfully",
                }
        else:
            if has_completed_trials:
                stats = {
                    "best_value": float(self.study.best_value),  # Convert to Python float
                    "best_params": {
                        k: float(v)
                        for k, v in self.study.best_params.items()  # Convert to Python float
                    },
                    "n_trials": len(self.study.trials),
                    "n_completed_trials": len(completed_trials),
                    "datetime": timestamp,
                    "run_fingerprint": self.run_fingerprint,
                }
            else:
                stats = {
                    "best_value": None,
                    "best_params": None,
                    "n_trials": len(self.study.trials),
                    "n_completed_trials": 0,
                    "datetime": timestamp,
                    "run_fingerprint": self.run_fingerprint,
                    "error": "No trials completed successfully",
                }

        with open(study_dir / "study_results.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Save visualization plots
        # fig_history = plot_optimization_history(self.study)
        # fig_history.write_html(str(study_dir / "optimization_history.html"))

        # fig_importance = plot_param_importances(self.study)
        # fig_importance.write_html(str(study_dir / "param_importance.html"))

        # Save trial data with validation metrics
        trial_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data_entry = {
                    "number": trial.number,
                    "datetime_start": trial.datetime_start.isoformat(),
                    "datetime_complete": trial.datetime_complete.isoformat(),
                    "params": {k: float(v) for k, v in trial.params.items()},
                }

                # Handle multi-objective values
                if self.multi_objective:
                    trial_data_entry.update(
                        {
                            "mean_return": float(trial.values[0]),
                            "worst_case": float(trial.values[1]),
                            "stability": float(trial.values[2]),
                        }
                    )
                else:
                    trial_data_entry["train_value"] = float(trial.value)

                # Add user attributes
                for attr in [
                    "validation_value",
                    "validation_returns_over_hodl",
                    "validation_sharpe",
                    "validation_return",
                    "train_returns_over_hodl",
                    "train_sharpe",
                    "train_return",
                ]:
                    trial_data_entry[attr] = float(
                        trial.user_attrs.get(attr, float("-inf"))
                    )

                trial_data.append(trial_data_entry)

        with open(study_dir / "trial_data.json", "w") as f:
            json.dump(trial_data, f, indent=2)

        # Create and save training vs validation plot
        # self._plot_train_vs_validation(trial_data, study_dir)
        # # Create and save training vs validation plot
        # self._plot_train_vs_validation(trial_data, study_dir)

    def optimize(self, objective):
        """Run the optimization process with error handling and parallel execution."""
        try:
            self.study.optimize(
                objective,
                n_trials=self.optuna_settings["n_trials"],
                timeout=self.optuna_settings["timeout"],
                n_jobs=self.optuna_settings["n_jobs"],
                callbacks=[self.early_stopping_callback],
                catch=(Exception,),
            )
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
        finally:
            self.save_results()


class Hashabledict(dict):
    """A hashable dictionary class that enables using dictionaries as dictionary keys.

    This class extends the built-in dict class to make dictionaries hashable by
    implementing the __hash__ and __eq__ methods. The hash is computed based on a
    sorted tuple of key-value pairs.

    Methods
    -------
    __key()
        Returns a tuple of sorted key-value pairs representing the dictionary.
    __hash__()
        Returns an integer hash value for the dictionary.
    __eq__(other)
        Checks equality between this dictionary and another by comparing their sorted
        key-value pairs.

    Examples
    --------
    >>> d1 = Hashabledict({'a': 1, 'b': 2})
    >>> d2 = Hashabledict({'b': 2, 'a': 1})
    >>> hash(d1) == hash(d2)
    True
    >>> d1 == d2
    True
    >>> d3 = {d1: 'value'}  # Can use as dictionary key
    """

    def __key(self):
        def make_hashable(v):
            if isinstance(v, list):
                return tuple(make_hashable(x) for x in v)
            elif isinstance(v, dict):
                return tuple(sorted((k, make_hashable(val)) for k, val in v.items()))
            return v
        return tuple((k, make_hashable(self[k])) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


class NestedHashabledict(dict):
    """A hashable dictionary class that enables using dictionaries as dictionary keys.
    Handles deeply nested dictionaries by recursively converting all nested dicts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Recursively convert all nested dictionaries to NestedHashabledict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = NestedHashabledict(
                    value
                )  # Use NestedHashabledict instead of Hashabledict
            elif isinstance(value, list):
                self[key] = [
                    NestedHashabledict(item) if isinstance(item, dict) else item
                    for item in value
                ]

    def __key(self):
        def make_hashable(v):
            if isinstance(v, list):
                return tuple(make_hashable(x) for x in v)
            elif isinstance(v, dict):
                return tuple(sorted((k, make_hashable(val)) for k, val in v.items()))
            return v
        return tuple((k, make_hashable(v)) for k, v in sorted(self.items()))

    def __hash__(self):
        try:
            return hash(self.__key())
        except TypeError as e:
            # Debug info to help identify unhashable items
            for k, v in self.items():
                try:
                    hash((k, v))
                except TypeError:
                    print(f"Unhashable item found - Key: {k}, Value type: {type(v)}")
            raise e

    def __eq__(self, other):
        if not isinstance(other, dict):
            return False
        return self.__key() == NestedHashabledict(other).__key()


class HashableArrayWrapper(Generic[T]):
    def __init__(self, val: T):
        self.val = val

    def __getattribute__(self, prop):
        if prop == "val" or prop == "__hash__" or prop == "__eq__":
            return super(HashableArrayWrapper, self).__getattribute__(prop)
        return getattr(self.val, prop)

    def __getitem__(self, key):
        return self.val[key]

    def __setitem__(self, key, val):
        self.val[key] = val

    def __hash__(self):
        return hash(self.val.tobytes())

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return self.__hash__() == other.__hash__()

        f = getattr(self.val, "__eq__")
        return f(self, other)


def get_run_location(run_fingerprint):
    """Generate a unique run location identifier based on the run fingerprint.

    This function creates a unique identifier for a simulation run by hashing the
    run_fingerprint dictionary. The run_fingerprint contains configuration parameters
    that define the simulation run.

    Parameters
    ----------
    run_fingerprint : dict
        A dictionary containing the configuration parameters for the simulation run.
        This typically includes parameters like start/end dates, tokens, rules, etc.

    Returns
    -------
    str
        A string identifier in the format "run_<sha256_hash>" where the hash is
        generated from the sorted JSON representation of the run_fingerprint.

    Examples
    --------
    >>> fingerprint = {"startDate": "2023-01-01", "tokens": ["BTC", "ETH"]}
    >>> get_run_location(fingerprint)
    'run_8d147a1f8b8...'
    """
    run_location = "run_" + str(
        hashlib.sha256(
            json.dumps(run_fingerprint, sort_keys=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
    )
    return run_location


def nan_rollback(grads, params, old_params):
    """Handles NaN values in gradients by rolling back to previous parameter values.

    This function checks for NaN values in gradients and reverts the corresponding
    parameters back to their previous values when NaNs are detected. This helps
    maintain numerical stability during optimization.

    Parameters
    ----------
    grads : dict
        Dictionary containing the current gradients
    params : dict
        Dictionary containing the current parameter values
    old_params : dict
        Dictionary containing the previous parameter values

    Returns
    -------
    dict
        Updated parameters with NaN values rolled back to previous values

    Examples
    --------
    >>> grads = {"log_k": jnp.array([[1.0, jnp.nan], [3.0, 4.0]])}
    >>> params = {"log_k": jnp.array([[0.1, 0.2], [0.3, 0.4]])}
    >>> old_params = {"log_k": jnp.array([[0.05, 0.15], [0.25, 0.35]])}
    >>> rolled_back = nan_rollback(grads, params, old_params)
    """
    for key in["log_k", "logit_lamb"]:
        if key in grads:
            bool_idx = jnp.sum(jnp.isnan(grads[key]), axis=-1, keepdims=True) > 0
            params = tree_map(
                lambda p, old_p: jnp.where(bool_idx, old_p, p), params, old_params
            )

    return params


@jit
def has_nan_grads(grad_tree):
    """Check if any gradients contain NaN values."""
    return tree_reduce(
        lambda acc, x: jnp.logical_or(acc, jnp.any(jnp.isnan(x))),
        grad_tree,
        initializer=False,
    )


def nan_param_reinit(
    params, grads, pool, initial_params, run_fingerprint, n_tokens, n_parameter_sets
):
    """Reinitialize parameters with previous values when gradients contain NaNs."""
    if has_nan_grads(grads):
        new_noised_params = pool.init_parameters(
            initial_params, run_fingerprint, n_tokens, n_parameter_sets
        )
        # For each parameter set index
        for i in range(len(next(iter(grads.values())))):
            # Check if any key has NaNs for this parameter set
            has_nans = False
            for key in grads:
                if key not in ["initial_weights", "initial_weights_logits", "subsidary_params"]:
                    if jnp.any(jnp.isnan(grads[key][i])):
                        has_nans = True
                        break

            # If NaNs found, replace all params for this index
            if has_nans:
                for key in params:
                    if key not in ["initial_weights", "initial_weights_logits", "subsidary_params"]:
                        params[key] = params[key].at[i].set(new_noised_params[key][i])
    return params


def get_unique_tokens(run_fingerprint):
    """Gets unique tokens from run fingerprint including subsidiary pools.

    Extracts all tokens from the main pool and subsidiary pools in the run fingerprint,
    removes duplicates, and returns a sorted list of unique tokens.

    Parameters
    ----------
    run_fingerprint : dict
        Dictionary containing run configuration including tokens and subsidiary pools

    Returns
    -------
    list
        Sorted list of unique token symbols

    Examples
    --------
    >>> fingerprint = {
    ...     "tokens": ["BTC", "ETH"],
    ...     "subsidary_pools": [{"tokens": ["ETH", "DAI"]}]
    ... }
    >>> get_unique_tokens(fingerprint)
    ['BTC', 'DAI', 'ETH']
    """
    all_tokens = [run_fingerprint["tokens"]] + [
        cprd["tokens"] for cprd in run_fingerprint["subsidary_pools"]
    ]
    all_tokens = [item for sublist in all_tokens for item in sublist]
    unique_tokens = list(set(all_tokens))
    unique_tokens.sort()
    return unique_tokens


def split_list(lst, num_splits):
    """Splits a list into a specified number of roughly equal sublists.

    Divides a list into num_splits sublists, distributing any remainder elements
    evenly among the first sublists.

    Parameters
    ----------
    lst : list
        The input list to split
    num_splits : int
        Number of sublists to create

    Returns
    -------
    list
        List of sublists

    Examples
    --------
    >>> split_list([1,2,3,4,5], 2)
    [[1,2,3], [4,5]]
    >>> split_list([1,2,3,4,5,6], 3)
    [[1,2], [3,4], [5,6]]
    """
    # Calculate the length of each sublist
    sub_len = len(lst) // num_splits

    # Determine the number of sublists that should be one element longer
    num_longer = len(lst) % num_splits

    # Initialize variables
    result = []
    start_idx = 0

    # Iterate over the number of sublists
    for _ in range(num_splits):
        # Calculate the end index of the sublist
        end_idx = start_idx + sub_len

        # If there are remaining elements to distribute, add one to the sublist length
        if num_longer > 0:
            end_idx += 1
            num_longer -= 1

        # Add the sublist to the result
        result.append(lst[start_idx:end_idx])

        # Update the start index for the next sublist
        start_idx = end_idx

    return result


def invert_permutation(perm):
    """
    Compute the inverse of a permutation.

    Given a permutation array that maps indices to their new positions,
    returns the inverse permutation that maps the new positions back to
    their original indices.

    Parameters
    ----------
    perm : numpy.ndarray
        Array representing a permutation of indices

    Returns
    -------
    numpy.ndarray
        The inverse permutation array

    Examples
    --------
    >>> perm = np.array([2,0,1])
    >>> invert_permutation(perm)
    array([1, 2, 0])
    """
    s = np.zeros(perm.size, perm.dtype)
    s[perm] = range(perm.size)
    return s


def permute_list_of_params(list_of_params, seed=0):
    """
    Randomly permute a list of parameters using a fixed random seed.

    This function takes a list of parameters and returns a new list with the same elements
    in a randomly permuted order. The permutation is deterministic based on the provided
    random seed.

    Parameters
    ----------
    list_of_params : list
        The list of parameters to permute
    seed : int, optional
        Random seed to use for reproducible permutations (default: 0)

    Returns
    -------
    list
        A new list containing the same elements as the input list but in a randomly
        permuted order

    Examples
    --------
    >>> params = [1, 2, 3, 4]
    >>> permute_list_of_params(params, seed=42)
    [3, 1, 4, 2]
    >>> permute_list_of_params(params, seed=42)  # Same seed gives same permutation
    [3, 1, 4, 2]
    """
    np.random.seed(seed)
    # permute
    idx = np.random.permutation(len(list_of_params))
    list_of_params_to_return = [list_of_params[i] for i in idx]
    return list_of_params_to_return


def unpermute_list_of_params(list_of_params):
    """
    Restore the original order of a previously permuted list of parameters.

    This function takes a list that was permuted using permute_list_of_params() and
    restores it to its original order by applying the inverse permutation with the
    same random seed.

    Parameters
    ----------
    list_of_params : list
        The permuted list of parameters to restore to original order

    Returns
    -------
    list
        A new list containing the same elements as the input list but restored to
        their original order before permutation

    Examples
    --------
    >>> params = [1, 2, 3, 4]
    >>> permuted = permute_list_of_params(params)  # [3, 1, 4, 2]
    >>> unpermute_list_of_params(permuted)  # Restores original order
    [1, 2, 3, 4]
    """
    # unpermute
    idx = np.random.permutation(len(list_of_params))
    idx_unpermute = invert_permutation(idx)
    list_of_params_to_return = [list_of_params[i] for i in idx_unpermute]
    return list_of_params_to_return


def get_trades_and_fees(
    run_fingerprint, raw_trades, fees_df, gas_cost_df, arb_fees_df, lp_supply_df, do_test_period=False
):
    """
    Process trade and fee data for a simulation run.

    Takes raw trades, fees, gas costs and arbitrage fees and converts them into arrays
    suitable for simulation. Handles both training and test periods if specified.

    Parameters
    ----------
    run_fingerprint : dict
        Dictionary containing run configuration including start/end dates and tokens
    raw_trades : pd.DataFrame, optional
        DataFrame containing raw trade data
    fees_df : pd.DataFrame, optional
        DataFrame containing fee data
    gas_cost_df : pd.DataFrame, optional
        DataFrame containing gas cost data
    arb_fees_df : pd.DataFrame, optional
        DataFrame containing arbitrage fee data
    lp_supply_df : pd.DataFrame, optional
        DataFrame containing LP supply data
    do_test_period : bool, optional
        Whether to process data for a test period after training period (default False)

    Returns
    -------
    dict
        Contains processed arrays for trades, fees, gas costs and arb fees for both
        training and test periods as applicable
    """
    # Process raw trades if provided
    if raw_trades is not None:
        train_period_trades = raw_trades_to_trade_array(
            raw_trades,
            start_date_string=run_fingerprint["startDateString"],
            end_date_string=run_fingerprint["endDateString"],
            tokens=get_unique_tokens(run_fingerprint),
        )
        if do_test_period:
            test_period_trades = raw_trades_to_trade_array(
                raw_trades,
                start_date_string=run_fingerprint["endDateString"],
                end_date_string=run_fingerprint["endTestDateString"],
                tokens=get_unique_tokens(run_fingerprint),
            )
    else:
        train_period_trades = None
        test_period_trades = None
    # Process fees, gas costs, and arb fees if provided
    fees_array = (
        raw_fee_like_amounts_to_fee_like_array(
            fees_df,
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
            names=["fees"],
            fill_method="ffill",
        )
        if fees_df is not None
        else None
    )
    if do_test_period:
        test_fees_array = (
            raw_fee_like_amounts_to_fee_like_array(
                fees_df,
                run_fingerprint["startDateString"],
                run_fingerprint["endDateString"],
                names=["fees"],
                fill_method="ffill",
            )
            if fees_df is not None
            else None
        )

    gas_cost_array = (
        raw_fee_like_amounts_to_fee_like_array(
            gas_cost_df,
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
            names=["trade_gas_cost_usd"],
            fill_method="ffill",
        )
        if gas_cost_df is not None
        else None
    )
    if do_test_period:
        test_gas_cost_array = (
            raw_fee_like_amounts_to_fee_like_array(
                gas_cost_df,
                run_fingerprint["endDateString"],
                run_fingerprint["endTestDateString"],
                names=["trade_gas_cost_usd"],
                fill_method="ffill",
            )
            if gas_cost_df is not None
            else None
        )

    arb_fees_array = (
        raw_fee_like_amounts_to_fee_like_array(
            arb_fees_df,
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
            names=["arb_fees"],
            fill_method="ffill",
        )
        if arb_fees_df is not None
        else None
    )
    if do_test_period:
        test_arb_fees_array = (
            raw_fee_like_amounts_to_fee_like_array(
                arb_fees_df,
                run_fingerprint["endDateString"],
                run_fingerprint["endTestDateString"],
                names=["arb_fees"],
                fill_method="ffill",
            )
            if arb_fees_df is not None
            else None
        )
    lp_supply_array = (
        raw_fee_like_amounts_to_fee_like_array(
            lp_supply_df,
            run_fingerprint["startDateString"],
            run_fingerprint["endDateString"],
            names=["lp_supply"],
            fill_method="ffill",
        )
        if lp_supply_df is not None
        else None
    )
    if do_test_period:
        test_lp_supply_array = (
            raw_fee_like_amounts_to_fee_like_array(
                lp_supply_df,
                run_fingerprint["endDateString"],
                run_fingerprint["endTestDateString"],
                names=["lp_supply"],
                fill_method="ffill",
            )
            if lp_supply_df is not None
            else None
        )
        return {
            "train_period_trades": train_period_trades,
            "test_period_trades": test_period_trades,
            "fees_array": fees_array,
            "gas_cost_array": gas_cost_array,
            "arb_fees_array": arb_fees_array,
            "lp_supply_array": lp_supply_array,
            "test_fees_array": test_fees_array,
            "test_gas_cost_array": test_gas_cost_array,
            "test_arb_fees_array": test_arb_fees_array,
            "test_lp_supply_array": test_lp_supply_array,
        }
    else:
        return {
            "train_period_trades": train_period_trades,
            "fees_array": fees_array,
            "gas_cost_array": gas_cost_array,
            "arb_fees_array": arb_fees_array,
            "lp_supply_array": lp_supply_array,
        }


def create_daily_unix_array(start_date_str, end_date_str):
    """
    Creates an array of daily Unix timestamps in milliseconds between two dates.

    Args:
        start_date_str (str): Start date string in format 'YYYY-MM-DD HH:MM:SS'
        end_date_str (str): End date string in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        list: Array of Unix timestamps in milliseconds for each day between start and end dates
    """
    end_date = pd.to_datetime(end_date_str)
    # Create a date range ending the day before the end_date
    date_range = pd.date_range(start=start_date_str, end=end_date, freq="D")
    daily_unix_values = date_range.view("int64") // 10**6
    return daily_unix_values.tolist()


def create_time_step(row, unix_values, tokens, index):
    """
    Creates a SimulationResultTimestepDto object for a single time step.

    Args:
        row (pd.Series): Row containing prices, reserves and weights data for this timestep
        unix_values (list): List of Unix timestamps in milliseconds
        tokens (list): List of token symbols
        index (int): Index of current timestep

    Returns:
        SimulationResultTimestepDto: Object containing timestamp and coin data for this timestep
    """
    timeStep = SimulationResultTimestepDto(unix_values[index], [], 0)

    for coinIndex, token in enumerate(tokens):
        coin = LiquidityPoolCoinDto()
        coin.coinCode = token
        coin.currentPrice = row["prices"][coinIndex].item()
        coin.amount = row["reserves"][coinIndex].item()
        coin.weight = row["weights"][coinIndex].item()
        coin.marketValue = coin.currentPrice * coin.amount
        timeStep.coinsHeld.append(coin)

    return timeStep


def optimized_output_conversion(simulationRunDto, outputDict, tokens):
    """
    Converts simulation output dictionary to a list of SimulationResultTimestepDto objects.

    Args:
        simulationRunDto (SimulationRunDto): Object containing simulation run parameters
        outputDict (dict): Dictionary containing simulation output data including prices, reserves, and values
        tokens (list): List of token symbols used in simulation

    Returns:
        list: List of SimulationResultTimestepDto objects containing timestep data

    The function:
    1. Creates Unix timestamps for each day between start and end dates
    2. Downsamples simulation data from minutes to daily frequency
    3. Calculates token weights from reserves, prices and total value
    4. Combines data into timestep DTOs with coin holdings and values
    """
    print(simulationRunDto.startDateString)
    print(simulationRunDto.endDateString)
    print(tokens)
    # Create a date range with daily frequency and convert to Unix timestamps in milliseconds
    unix_values = create_daily_unix_array(
        simulationRunDto.startDateString, simulationRunDto.endDateString
    )

    # Convert outputDict data to pandas DataFrame for efficient slicing
    prices_df = pd.DataFrame(outputDict["prices"])[::1440]
    reserves_df = pd.DataFrame(outputDict["reserves"])[::1440]
    values_df = pd.DataFrame(outputDict["value"])[::1440]
    
    # note that the returned weights are empirical weights, not calculated weights
    # this is because the calculated weights are not returned in the outputDict as
    # they are not guaranteed to exist for all possible pool types
    weights_df = pd.DataFrame(
        outputDict["reserves"]
        * outputDict["prices"]
        / outputDict["value"][:, np.newaxis]
    )[::1440]

    print("prices_df: ", len(prices_df))
    print("reserves_df: ", len(reserves_df))
    print("weights_df: ", len(weights_df))
    print("unix_values: ", len(unix_values))

    # Combine DataFrames
    combined_df = pd.concat(
        [prices_df, reserves_df, weights_df],
        axis=1,
        keys=["prices", "reserves", "weights"],
    )

    print(len(unix_values))
    print(len(combined_df))

    # Check if the length of unix_values matches the number of rows in combined_df
    if len(unix_values) != len(combined_df):
        print(len(unix_values))
        print(len(combined_df))
        raise ValueError(
            "The length of unix_values does not match the number of rows in combined_df"
        )

    # Ensure index alignment by resetting index
    combined_df = combined_df.reset_index(drop=True)

    # Convert DataFrame to list of DTO objects using apply
    resultTimeSteps = combined_df.apply(
        lambda row: create_time_step(row, unix_values, tokens, row.name), axis=1
    ).tolist()

    return resultTimeSteps
