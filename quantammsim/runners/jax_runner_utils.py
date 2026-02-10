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


from typing import Dict, Any, Generic, TypeVar, List, Optional, Tuple
from copy import deepcopy
T = TypeVar('T')      # Declare type variable

# Parameter keys excluded from NaN checking and selective reinitialization.
# These keys are either non-differentiable (subsidary_params), derived from
# other params (initial_weights), or handled separately (initial_weights_logits).
#
# This set is the single source of truth — import it rather than duplicating.
# Currently also hardcoded in:
#   - backpropagation.py (_NAN_EXCLUDED_KEYS)
#   - base_pool.py (add_noise)
#   - sampling.py (_DEFAULT_EXCLUDE_KEYS, subset)
# TODO: consolidate all those sites to import from here.
NAN_EXCLUDED_PARAM_KEYS = frozenset([
    "initial_weights", "initial_weights_logits", "subsidary_params",
])


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
    # Copy subsidary_params if present (required by forward pass)
    if "subsidary_params" in params:
        trial_params["subsidary_params"] = params["subsidary_params"]

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
                    "validation_returns_over_uniform_hodl",
                    "validation_sharpe",
                    "validation_return",
                    "train_returns_over_hodl",
                    "train_returns_over_uniform_hodl",
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


# Fields that are only used during training setup, not in forward passes
# These are excluded when creating static_dict from run_fingerprint
_TRAINING_ONLY_FIELDS = frozenset({
    "optimisation_settings",  # Contains lr, optimizer, etc.
    "startDateString",  # Data loading dates
    "endDateString",
    "endTestDateString",
    "subsidary_pools",  # Handled separately
    "bout_offset",  # Training sampling config
    "freq",  # Data frequency string
})


def get_sig_variations(n_assets: int) -> tuple:
    """
    Compute signature variations for arbitrage.

    Returns all possible (asset_in, asset_out) pairs encoded as a tuple of tuples,
    where each inner tuple has exactly one +1 (asset out) and one -1 (asset in),
    with zeros elsewhere.

    Parameters
    ----------
    n_assets : int
        Number of assets in the pool.

    Returns
    -------
    tuple
        Tuple of tuples representing valid arbitrage directions.
        Each inner tuple has shape (n_assets,) with values in {-1, 0, 1}.

    Example
    -------
    >>> get_sig_variations(3)
    ((1, -1, 0), (1, 0, -1), (-1, 1, 0), (0, 1, -1), (-1, 0, 1), (0, -1, 1))
    """
    from itertools import product

    all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    # Keep only variations with exactly one +1 and one -1
    all_sig_variations = all_sig_variations[(all_sig_variations == 1).sum(-1) == 1]
    all_sig_variations = all_sig_variations[(all_sig_variations == -1).sum(-1) == 1]
    return tuple(map(tuple, all_sig_variations))


def create_static_dict(
    run_fingerprint: dict,
    bout_length: int,
    all_sig_variations: list = None,
    overrides: dict = None,
) -> NestedHashabledict:
    """Create a static_dict from run_fingerprint for use in forward passes.

    This simplifies the previous pattern of manually picking ~30 fields
    from run_fingerprint to create static_dict. Instead, we start with
    the full run_fingerprint and:
    1. Exclude training-only fields
    2. Apply necessary transformations (e.g., tokens -> tuple)
    3. Add computed fields (bout_length, all_sig_variations)
    4. Apply any overrides

    Parameters
    ----------
    run_fingerprint : dict
        The full run configuration dictionary
    bout_length : int
        Bout length to use (varies between train/test)
    all_sig_variations : list, optional
        Pre-computed signature variations for arbitrage
    overrides : dict, optional
        Additional key-value pairs to override/add

    Returns
    -------
    NestedHashabledict
        Hashable static dictionary for use in JAX forward passes

    Example
    -------
    >>> static_dict = create_static_dict(run_fingerprint, bout_length=10080)
    >>> # Instead of manually building:
    >>> # static_dict = {"chunk_period": rf["chunk_period"], "bout_length": ..., ...}
    """
    # Start with filtered copy
    static = {k: v for k, v in run_fingerprint.items() if k not in _TRAINING_ONLY_FIELDS}

    # Apply transformations
    if "tokens" in static:
        static["tokens"] = tuple(static["tokens"])

    # Add computed fields
    static["bout_length"] = bout_length

    # Compute all_sig_variations if not provided but n_assets is available
    if all_sig_variations is not None:
        static["all_sig_variations"] = all_sig_variations
    elif "n_assets" in static or (overrides and "n_assets" in overrides):
        n_assets = overrides.get("n_assets") if overrides and "n_assets" in overrides else static.get("n_assets")
        if n_assets is not None:
            static["all_sig_variations"] = get_sig_variations(n_assets)

    # Default run_type if not present
    if "run_type" not in static:
        static["run_type"] = "normal"

    # Apply overrides
    if overrides:
        static.update(overrides)

    return NestedHashabledict(static)


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


def has_nan_params(params):
    """Check if any parameters contain NaN values."""
    for key in params:
        if key not in NAN_EXCLUDED_PARAM_KEYS:
            if hasattr(params[key], 'shape') and jnp.any(jnp.isnan(params[key])):
                return True
    return False


def nan_param_reinit(
    params, grads, pool, initial_params, run_fingerprint, n_tokens, n_parameter_sets
):
    """Reinitialize parameter sets that contain NaN values.

    Checks params directly (not just grads) since params can become NaN
    from bad update steps even when grads were finite.
    """
    # Check if any param set has NaN params (the actual problem)
    if has_nan_params(params):
        new_noised_params = pool.init_parameters(
            initial_params, run_fingerprint, n_tokens, n_parameter_sets
        )
        # For each parameter set index
        n_param_sets = len(next(iter(params.values())))
        for i in range(n_param_sets):
            # Check if any key has NaNs for this parameter set
            has_nans = False
            for key in params:
                if key not in NAN_EXCLUDED_PARAM_KEYS:
                    if hasattr(params[key], 'shape') and len(params[key].shape) > 0:
                        if jnp.any(jnp.isnan(params[key][i])):
                            has_nans = True
                            break

            # If NaNs found, replace all params for this index
            if has_nans:
                for key in params:
                    if key not in NAN_EXCLUDED_PARAM_KEYS:
                        if hasattr(params[key], 'shape') and len(params[key].shape) > 0:
                            params[key] = params[key].at[i].set(new_noised_params[key][i])
    return params


def nan_param_reinit_vectorized(
    params, has_nan_per_set, pool, initial_params, run_fingerprint, n_tokens, n_parameter_sets
):
    """Vectorized reinitialization of NaN parameter sets.

    Uses a pre-computed boolean mask (from the JIT'd update function) instead of
    re-checking every param key with device syncs.

    Parameters
    ----------
    params : dict
        Batched params, shape (n_parameter_sets, ...) per key.
    has_nan_per_set : jnp.ndarray
        Boolean array of shape (n_parameter_sets,) from the JIT'd update.
    pool : AbstractPool
        Pool instance for generating replacement params.
    initial_params, run_fingerprint : dict
        Passed through to pool.init_parameters.
    n_tokens, n_parameter_sets : int
        Shape info.
    """
    if not jnp.any(has_nan_per_set):
        return params

    new_noised_params = pool.init_parameters(
        initial_params, run_fingerprint, n_tokens, n_parameter_sets
    )

    for key in params:
        if key not in NAN_EXCLUDED_PARAM_KEYS:
            if hasattr(params[key], 'shape') and len(params[key].shape) > 0:
                # mask shape: (n_parameter_sets,) -> broadcast with (n_parameter_sets, n_tokens)
                mask = has_nan_per_set.reshape(-1, *([1] * (len(params[key].shape) - 1)))
                params[key] = jnp.where(mask, new_noised_params[key], params[key])

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


# =============================================================================
# Memory Probing Utilities
# =============================================================================


def probe_max_n_parameter_sets(
    run_fingerprint: dict,
    min_sets: int = 1,
    max_sets: int = 64,
    safety_margin: float = 0.9,
    verbose: bool = True,
) -> dict:
    """
    Probe to find the maximum n_parameter_sets that fits in GPU memory.

    Uses binary search to find the largest n_parameter_sets value that can
    complete a forward pass without OOM. Returns a dict with the recommended
    value and diagnostic info.

    Parameters
    ----------
    run_fingerprint : dict
        The run fingerprint configuration. Will be modified temporarily during probing.
    min_sets : int
        Minimum n_parameter_sets to try (default 1).
    max_sets : int
        Maximum n_parameter_sets to try (default 64).
    safety_margin : float
        Fraction of max found to use as recommendation (default 0.9).
        This provides headroom for gradient computation which uses more memory.
    verbose : bool
        Whether to print progress information.

    Returns
    -------
    dict
        Keys: ``max_n_parameter_sets`` (int), ``recommended_n_parameter_sets``
        (int, with safety margin applied), ``probed_values`` (list),
        ``success_values`` (list), ``failed_values`` (list).

    Notes
    -----
    - This function temporarily modifies run_fingerprint during probing.
    - JAX caches are cleared between attempts.
    - The forward pass (without gradients) is used for probing, so gradient
      computation may require ~2x more memory. Hence the safety_margin.
    """
    from copy import deepcopy
    from jax import clear_caches
    from jax.tree_util import Partial
    import jax.numpy as jnp

    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
    from quantammsim.core_simulator.param_utils import recursive_default_set
    from quantammsim.pools.creator import create_pool
    from quantammsim.utils.data_processing.historic_data_utils import get_data_dict
    from quantammsim.core_simulator.forward_pass import forward_pass_nograd
    from jax import jit, vmap

    # Work with a copy to avoid side effects
    probe_fingerprint = deepcopy(run_fingerprint)
    recursive_default_set(probe_fingerprint, run_fingerprint_defaults)

    probed_values = []
    success_values = []
    failed_values = []

    def try_forward_pass(n_sets: int) -> bool:
        """Attempt a forward pass with n_sets parameter sets. Returns True if successful."""
        probe_fingerprint["optimisation_settings"]["n_parameter_sets"] = n_sets

        try:
            # Get tokens and setup
            unique_tokens = get_unique_tokens(probe_fingerprint)
            n_tokens = len(unique_tokens)

            # Load minimal data
            data_dict = get_data_dict(
                unique_tokens,
                probe_fingerprint,
                data_kind=probe_fingerprint["optimisation_settings"]["training_data_kind"],
                max_memory_days=probe_fingerprint["max_memory_days"],
                start_date_string=probe_fingerprint["startDateString"],
                end_time_string=probe_fingerprint["endDateString"],
                start_time_test_string=probe_fingerprint["endDateString"],
                end_time_test_string=probe_fingerprint.get("endTestDateString"),
                do_test_period=False,
            )

            bout_length_window = data_dict["bout_length"] - probe_fingerprint["bout_offset"]
            if bout_length_window <= 0:
                bout_length_window = data_dict["bout_length"] // 2

            # Create pool and params
            rule = probe_fingerprint["rule"]
            pool = create_pool(rule)

            learnable_bounds = probe_fingerprint.get("learnable_bounds_settings", {})
            initial_params = {
                "initial_memory_length": probe_fingerprint["initial_memory_length"],
                "initial_memory_length_delta": probe_fingerprint["initial_memory_length_delta"],
                "initial_k_per_day": probe_fingerprint["initial_k_per_day"],
                "initial_weights_logits": probe_fingerprint["initial_weights_logits"],
                "initial_log_amplitude": probe_fingerprint["initial_log_amplitude"],
                "initial_raw_width": probe_fingerprint["initial_raw_width"],
                "initial_raw_exponents": probe_fingerprint["initial_raw_exponents"],
                "initial_pre_exp_scaling": probe_fingerprint["initial_pre_exp_scaling"],
                "min_weights_per_asset": learnable_bounds.get("min_weights_per_asset"),
                "max_weights_per_asset": learnable_bounds.get("max_weights_per_asset"),
            }

            params = pool.init_parameters(initial_params, probe_fingerprint, n_tokens, n_sets)
            params_in_axes_dict = pool.make_vmap_in_axes(params)

            # Setup static dict using encapsulated helper
            # all_sig_variations is auto-computed from n_assets
            static_dict = create_static_dict(
                probe_fingerprint,
                bout_length=bout_length_window,
                overrides={
                    "n_assets": n_tokens,
                    "training_data_kind": probe_fingerprint["optimisation_settings"]["training_data_kind"],
                    "do_trades": False,
                },
            )

            # Create vmapped forward pass
            partial_forward = Partial(
                forward_pass_nograd,
                prices=data_dict["prices"],
                static_dict=static_dict,
                pool=pool,
            )

            vmapped_forward = jit(
                vmap(partial_forward, in_axes=[params_in_axes_dict, None, None])
            )

            # Run forward pass
            start_index = (data_dict["start_idx"], 0)
            _ = vmapped_forward(params, start_index, None)

            # Force computation to complete
            jnp.zeros(1).block_until_ready()

            return True

        except Exception as e:
            error_str = str(e).lower()
            if "resource" in error_str or "memory" in error_str or "oom" in error_str:
                return False
            # Re-raise non-memory errors
            raise

    # Binary search for max n_parameter_sets
    low, high = min_sets, max_sets
    best = min_sets

    while low <= high:
        mid = (low + high) // 2
        probed_values.append(mid)

        if verbose:
            print(f"Probing n_parameter_sets={mid}...", end=" ")

        # Clear caches before each attempt
        clear_caches()
        import gc
        gc.collect()

        try:
            success = try_forward_pass(mid)
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            success = False

        if success:
            if verbose:
                print("OK")
            success_values.append(mid)
            best = mid
            low = mid + 1
        else:
            if verbose:
                print("OOM")
            failed_values.append(mid)
            high = mid - 1

        # Clear caches after attempt
        clear_caches()
        gc.collect()

    recommended = max(min_sets, int(best * safety_margin))

    result = {
        "max_n_parameter_sets": best,
        "recommended_n_parameter_sets": recommended,
        "probed_values": sorted(probed_values),
        "success_values": sorted(success_values),
        "failed_values": sorted(failed_values),
    }

    if verbose:
        print(f"\nMemory probe results:")
        print(f"  Max n_parameter_sets: {best}")
        print(f"  Recommended (with {safety_margin:.0%} margin): {recommended}")

    return result


def allocate_memory_budget(
    run_fingerprint: dict,
    available_memory_gb: float = None,
    priority: str = "exploration",
    probe_if_needed: bool = True,
    max_ensemble_members: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Allocate memory budget across hyperparameters based on priority.

    Parameters
    ----------
    run_fingerprint : dict
        The run fingerprint configuration.
    available_memory_gb : float, optional
        Available GPU memory in GB. If None and probe_if_needed=True,
        will probe to determine capacity.
    priority : str
        How to allocate memory budget:
        - "exploration": Maximize n_parameter_sets (find diverse solutions)
        - "robustness": Balance n_parameter_sets and n_ensemble_members
        - "variance_reduction": Maximize batch_size (stable gradients)
    probe_if_needed : bool
        Whether to probe memory if available_memory_gb is not provided.
    max_ensemble_members : int
        Maximum ensemble members to allocate (default 1 = no ensembling).
        Set higher (e.g., 4) if you want the "robustness" priority to use ensembles.
    verbose : bool
        Whether to print allocation info.

    Returns
    -------
    dict
        Recommended settings with keys: ``n_parameter_sets`` (int),
        ``n_ensemble_members`` (int), ``batch_size`` (int),
        ``priority_used`` (str), ``probe_result`` (dict or None).
    """
    probe_result = None

    if available_memory_gb is None and probe_if_needed:
        # Probe to find capacity
        probe_result = probe_max_n_parameter_sets(
            run_fingerprint,
            verbose=verbose,
            safety_margin=0.85,  # More conservative for allocation
        )
        max_units = probe_result["recommended_n_parameter_sets"]
    elif available_memory_gb is not None:
        # Rough estimate: assume ~0.5-2 GB per parameter set depending on config
        # This is a very rough heuristic
        max_units = int(available_memory_gb * 4)  # ~4 param sets per GB as rough estimate
    else:
        # Default conservative estimate
        max_units = 8

    # Allocate based on priority
    if priority == "exploration":
        # Maximize exploration with independent param sets
        n_parameter_sets = max(1, max_units)
        n_ensemble_members = 1
        batch_size = 1  # Small batch, rely on param diversity

    elif priority == "robustness":
        # Balance exploration and ensembling (if allowed)
        if max_ensemble_members > 1:
            n_parameter_sets = max(1, max_units // 2)
            n_ensemble_members = min(max_ensemble_members, max(1, max_units // n_parameter_sets // 2))
            batch_size = max(1, max_units // (n_parameter_sets * n_ensemble_members))
        else:
            # No ensembling allowed, fall back to exploration-like allocation
            n_parameter_sets = max(1, max_units)
            n_ensemble_members = 1
            batch_size = 1

    elif priority == "variance_reduction":
        # Fewer param sets, larger batches for stable gradients
        n_parameter_sets = min(4, max_units)
        n_ensemble_members = 1
        batch_size = max(1, max_units // n_parameter_sets)

    else:
        raise ValueError(f"Unknown priority: {priority}. Use 'exploration', 'robustness', or 'variance_reduction'")

    result = {
        "n_parameter_sets": n_parameter_sets,
        "n_ensemble_members": n_ensemble_members,
        "batch_size": batch_size,
        "priority_used": priority,
        "probe_result": probe_result,
    }

    if verbose:
        print(f"\nMemory allocation ({priority} priority):")
        print(f"  n_parameter_sets: {n_parameter_sets}")
        print(f"  n_ensemble_members: {n_ensemble_members}")
        print(f"  batch_size: {batch_size}")
        if probe_result:
            print(f"  (based on probed max: {probe_result['max_n_parameter_sets']})")

    return result


def apply_memory_allocation(run_fingerprint: dict, allocation: dict) -> dict:
    """
    Apply memory allocation results to a run_fingerprint.

    Parameters
    ----------
    run_fingerprint : dict
        The run fingerprint to modify (will be modified in place).
    allocation : dict
        Result from allocate_memory_budget().

    Returns
    -------
    dict
        The modified run_fingerprint.
    """
    run_fingerprint["optimisation_settings"]["n_parameter_sets"] = allocation["n_parameter_sets"]
    run_fingerprint["optimisation_settings"]["batch_size"] = allocation["batch_size"]
    run_fingerprint["n_ensemble_members"] = allocation["n_ensemble_members"]

    return run_fingerprint


def auto_configure_memory_params(
    run_fingerprint: dict,
    priority: str = "exploration",
    max_ensemble_members: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Convenience function: probe memory and apply allocation in one step.

    Parameters
    ----------
    run_fingerprint : dict
        The run fingerprint to configure (will be modified in place).
    priority : str
        Allocation priority ("exploration", "robustness", "variance_reduction").
    max_ensemble_members : int
        Maximum ensemble members to allocate (default 1 = no ensembling).
    verbose : bool
        Whether to print progress info.

    Returns
    -------
    dict
        The modified run_fingerprint with optimal memory settings.

    Example
    -------
    >>> run = {...}  # your run_fingerprint
    >>> auto_configure_memory_params(run, priority="exploration")
    >>> train_on_historic_data(run)
    """
    allocation = allocate_memory_budget(
        run_fingerprint,
        priority=priority,
        probe_if_needed=True,
        max_ensemble_members=max_ensemble_members,
        verbose=verbose,
    )
    return apply_memory_allocation(run_fingerprint, allocation)


# =============================================================================
# Best Params Selection and Tracking
# =============================================================================

# Valid selection methods - must match load_manually methods where applicable
SELECTION_METHODS = [
    "last",                    # Always return last iteration/trial
    "best_train",              # Best training metric
    "best_val",                # Best validation metric (requires val_fraction > 0)
    "best_continuous_test",    # Best continuous test metric (NOT RECOMMENDED - data leakage)
    "best_train_min_test",     # Best train meeting test threshold
]


def _nanargmax_jnp(arr):
    """jnp.argmax ignoring NaNs (no jnp.nanargmax in JAX)."""
    return jnp.argmax(jnp.where(jnp.isnan(arr), -jnp.inf, arr))


# =============================================================================
# Scan-compatible pure-function tracker
# =============================================================================


def init_tracker_state(params, n_parameter_sets, n_metrics=12):
    """Create initial tracker carry state for jax.lax.scan.

    All fields are JAX arrays (no None) so the pytree structure is fixed
    from iteration 0.  ``best_params`` is initialised to ``params`` so
    the leaf structure matches on every iteration.

    Parameters
    ----------
    params : dict
        Batched strategy parameters, shape ``(n_parameter_sets, ...)``
        per key.
    n_parameter_sets : int
        Number of parallel parameter sets.
    n_metrics : int
        Number of scalar metrics per param set (default 12, matching
        ``_METRIC_KEYS``).

    Returns
    -------
    dict
        Carry-compatible tracker state.
    """
    return {
        "best_metric_value": jnp.array(-jnp.inf),
        "best_iteration": jnp.array(0, dtype=jnp.int32),
        "best_param_idx": jnp.array(0, dtype=jnp.int32),
        "best_params": tree_map(lambda x: x.copy(), params),
        "best_train_metrics": jnp.full((n_parameter_sets, n_metrics), jnp.nan),
        "best_val_metrics": jnp.full((n_parameter_sets, n_metrics), jnp.nan),
        "best_test_metrics": jnp.full((n_parameter_sets, n_metrics), jnp.nan),
    }


def update_tracker_state(
    tracker_state, iteration, params,
    train_metrics_arr, val_metrics_arr, test_metrics_arr,
    sel_metric_idx, use_val, use_test,
):
    """Pure-function tracker update for jax.lax.scan.

    Replaces ``BestParamsTracker.update()`` inside the scan body.
    All operations are JAX — no Python control flow that depends on
    runtime values.

    The selection source (train vs val vs test) is controlled by
    ``use_val`` and ``use_test`` which are Python bools resolved at
    trace time.  Inside the scan body they are constants baked into
    the closure by the factory.

    Parameters
    ----------
    tracker_state : dict
        Current tracker carry (from ``init_tracker_state``).
    iteration : jnp.int32
        Current iteration index.
    params : dict
        Current batched params.
    train_metrics_arr : jnp.ndarray
        Shape ``(n_parameter_sets, n_metrics)``.
    val_metrics_arr : jnp.ndarray
        Shape ``(n_parameter_sets, n_metrics)``.  Zeros if unused.
    test_metrics_arr : jnp.ndarray
        Shape ``(n_parameter_sets, n_metrics)``.  Zeros if unused.
    sel_metric_idx : int
        Column index into the metrics array for selection (Python int,
        resolved at trace time).
    use_val : bool
        If True, select on ``val_metrics_arr``.  Python bool.
    use_test : bool
        If True, select on ``test_metrics_arr``.  Python bool.
        ``use_val`` takes precedence if both are True.

    Returns
    -------
    (new_tracker_state, improved) : tuple
        ``improved`` is a JAX boolean scalar.
    """
    # Pick selection source — Python if at trace time
    if use_val:
        sel_arr = val_metrics_arr[:, sel_metric_idx]
    elif use_test:
        sel_arr = test_metrics_arr[:, sel_metric_idx]
    else:
        sel_arr = train_metrics_arr[:, sel_metric_idx]

    selection_value = jnp.nanmean(sel_arr)
    param_idx = _nanargmax_jnp(sel_arr).astype(jnp.int32)

    improved = selection_value > tracker_state["best_metric_value"]

    new_state = {
        "best_metric_value": jnp.where(
            improved, selection_value, tracker_state["best_metric_value"],
        ),
        "best_iteration": jnp.where(
            improved, iteration, tracker_state["best_iteration"],
        ),
        "best_param_idx": jnp.where(
            improved, param_idx, tracker_state["best_param_idx"],
        ),
        "best_params": tree_map(
            lambda new, old: jnp.where(improved, new, old),
            params, tracker_state["best_params"],
        ),
        "best_train_metrics": jnp.where(
            improved, train_metrics_arr, tracker_state["best_train_metrics"],
        ),
        "best_val_metrics": jnp.where(
            improved, val_metrics_arr, tracker_state["best_val_metrics"],
        ),
        "best_test_metrics": jnp.where(
            improved, test_metrics_arr, tracker_state["best_test_metrics"],
        ),
    }
    return new_state, improved


# =============================================================================
# Scan-compatible NaN reinit from pre-generated bank
# =============================================================================

NAN_BANK_SIZE = 32


def generate_nan_bank(pool, initial_params, run_fingerprint, n_tokens,
                      n_parameter_sets, bank_size=NAN_BANK_SIZE):
    """Pre-generate a bank of replacement params for NaN reinit.

    Called once before the scan loop.  Each bank entry has shape
    ``(n_parameter_sets, ...)``, so simultaneous NaN across param sets
    still gets distinct replacements.

    Parameters
    ----------
    pool : AbstractPool
        Pool instance (for ``init_parameters``).
    initial_params : dict
        Initial param config passed to ``pool.init_parameters``.
    run_fingerprint : dict
        Run configuration.
    n_tokens : int
        Number of tokens/assets.
    n_parameter_sets : int
        Batch size for parameter sets.
    bank_size : int
        Number of replacement entries in the bank.

    Returns
    -------
    dict
        ``{key: jnp.ndarray of shape (bank_size, n_parameter_sets, ...)}``
        for each non-excluded param key.
    """
    bank = {}
    for _ in range(bank_size):
        new_params = pool.init_parameters(
            initial_params, run_fingerprint, n_tokens, n_parameter_sets,
        )
        for key in new_params:
            if key not in NAN_EXCLUDED_PARAM_KEYS:
                if hasattr(new_params[key], 'shape'):
                    bank.setdefault(key, []).append(new_params[key])
    return {k: jnp.stack(v) for k, v in bank.items()}


def nan_reinit_from_bank(params, has_nan, nan_bank, nan_count):
    """Replace NaN param sets using a pre-generated bank.

    Scan-compatible: no Python calls to ``pool.init_parameters``,
    only ``jnp.where`` indexing into the bank.

    Parameters
    ----------
    params : dict
        Current batched params.
    has_nan : jnp.ndarray
        Boolean ``(n_parameter_sets,)`` from the update function.
    nan_bank : dict
        Pre-generated bank from ``generate_nan_bank``.
    nan_count : jnp.int32
        Running counter of NaN events (for bank index cycling).

    Returns
    -------
    (new_params, new_count) : tuple
    """
    any_nan = jnp.any(has_nan)
    bank_size = jnp.array(list(nan_bank.values())[0].shape[0])
    bank_idx = nan_count % bank_size
    new_count = nan_count + any_nan.astype(jnp.int32)

    new_params = {}
    for key in params:
        if key in nan_bank:
            replacement = nan_bank[key][bank_idx]
            mask = has_nan.reshape(-1, *([1] * (params[key].ndim - 1)))
            new_params[key] = jnp.where(mask, replacement, params[key])
        else:
            new_params[key] = params[key]
    return new_params, new_count


def compute_selection_metric(
    train_metrics: List[Dict],
    val_metrics: Optional[List[Dict]] = None,
    continuous_test_metrics: Optional[List[Dict]] = None,
    method: str = "best_val",
    metric: str = "sharpe",
    min_threshold: float = 0.0,
) -> Tuple:
    """
    Compute selection metric value for a single iteration/trial.

    This is the shared core logic used by both BestParamsTracker (during training)
    and load_manually (post-training). Returns a value for comparison and the
    index of the best param set.

    Returns JAX scalars to avoid device→host sync. Callers that need host values
    should explicitly convert (e.g., ``float(selection_value)``).

    Parameters
    ----------
    train_metrics : list of dict
        Training metrics for each param set. Values may be JAX scalars or floats.
    val_metrics : list of dict, optional
        Validation metrics for each param set. Required if method="best_val".
    continuous_test_metrics : list of dict, optional
        Continuous test metrics for each param set.
    method : str
        Selection method. One of SELECTION_METHODS.
    metric : str
        Which metric to use for comparison (e.g., "sharpe", "returns_over_uniform_hodl").
    min_threshold : float
        Minimum threshold for "best_train_min_test" method.

    Returns
    -------
    tuple of (scalar, scalar)
        (selection_value, best_param_idx). May be JAX or Python scalars.
        Higher selection_value is always better.
    """
    if method not in SELECTION_METHODS:
        raise ValueError(f"Unknown selection method: {method}. Must be one of {SELECTION_METHODS}")

    if method == "last":
        return jnp.inf, 0

    elif method == "best_train":
        if not train_metrics:
            return -jnp.inf, 0
        metrics_per_set = jnp.array([m.get(metric, jnp.nan) for m in train_metrics])
        has_valid = jnp.any(~jnp.isnan(metrics_per_set))
        best_idx = jnp.where(has_valid, _nanargmax_jnp(metrics_per_set), 0)
        selection_val = jnp.where(has_valid, jnp.nanmean(metrics_per_set), -jnp.inf)
        return selection_val, best_idx

    elif method == "best_val":
        if not val_metrics:
            raise ValueError("best_val method requires val_metrics (set val_fraction > 0)")
        metrics_per_set = jnp.array([m.get(metric, jnp.nan) for m in val_metrics])
        has_valid = jnp.any(~jnp.isnan(metrics_per_set))
        best_idx = jnp.where(has_valid, _nanargmax_jnp(metrics_per_set), 0)
        selection_val = jnp.where(has_valid, jnp.nanmean(metrics_per_set), -jnp.inf)
        return selection_val, best_idx

    elif method == "best_continuous_test":
        if not continuous_test_metrics:
            return -jnp.inf, 0
        metrics_per_set = jnp.array([m.get(metric, jnp.nan) for m in continuous_test_metrics])
        has_valid = jnp.any(~jnp.isnan(metrics_per_set))
        best_idx = jnp.where(has_valid, _nanargmax_jnp(metrics_per_set), 0)
        selection_val = jnp.where(has_valid, jnp.nanmean(metrics_per_set), -jnp.inf)
        return selection_val, best_idx

    elif method == "best_train_min_test":
        if not train_metrics:
            return -jnp.inf, 0

        train_per_set = jnp.array([m.get(metric, jnp.nan) for m in train_metrics])

        if continuous_test_metrics:
            test_per_set = jnp.array([m.get(metric, jnp.nan) for m in continuous_test_metrics])
            # Mask: valid train AND test >= threshold
            valid = ~jnp.isnan(train_per_set) & ~jnp.isnan(test_per_set) & (test_per_set >= min_threshold)
            masked_train = jnp.where(valid, train_per_set, -jnp.inf)
            any_valid = jnp.any(valid)
            best_idx = jnp.where(any_valid, jnp.argmax(masked_train), _nanargmax_jnp(train_per_set))
            selection_val = jnp.where(any_valid, masked_train[jnp.argmax(masked_train)], jnp.nanmean(train_per_set))
            return selection_val, best_idx
        else:
            best_idx = _nanargmax_jnp(train_per_set)
            return jnp.nanmean(train_per_set), best_idx

    else:
        raise ValueError(f"Unknown selection method: {method}")


class BestParamsTracker:
    """
    Unified tracking of params across training iterations/trials.

    Tracks both "last" (most recent iteration) and "best" (by selection method)
    params along with their associated metrics and continuous outputs.

    Used by both SGD and Optuna paths to ensure consistent param selection logic.

    Parameters
    ----------
    selection_method : str
        Method for selecting best params. One of SELECTION_METHODS.
    metric : str
        Which metric to use for selection (e.g., "sharpe", "returns_over_uniform_hodl").
    min_threshold : float
        Minimum threshold for "best_train_min_test" method.

    Attributes
    ----------
    last_* : Various
        State from the most recent update() call.
    best_* : Various
        State from when selection metric was highest.
    """

    def __init__(
        self,
        selection_method: str = "best_val",
        metric: str = "sharpe",
        min_threshold: float = 0.0,
    ):
        if selection_method not in SELECTION_METHODS:
            raise ValueError(f"Unknown selection method: {selection_method}. Must be one of {SELECTION_METHODS}")

        self.selection_method = selection_method
        self.metric = metric
        self.min_threshold = min_threshold

        # "Last" state - always most recent iteration
        self.last_iteration = 0
        self.last_params = None
        self.last_param_idx = 0
        self.last_train_metrics = None
        self.last_val_metrics = None
        self.last_continuous_test_metrics = None
        self.last_continuous_outputs = None  # {"reserves": ..., "weights": ...}

        # "Best" state - based on selection_method
        self.best_iteration = 0
        self.best_params = None
        self.best_param_idx = 0
        self.best_train_metrics = None
        self.best_val_metrics = None
        self.best_continuous_test_metrics = None
        self.best_continuous_outputs = None
        self.best_metric_value = -float("inf")

    def update(
        self,
        iteration: int,
        params: Dict,
        continuous_outputs: Dict,
        train_metrics_list: List[Dict],
        val_metrics_list: Optional[List[Dict]] = None,
        continuous_test_metrics_list: Optional[List[Dict]] = None,
    ) -> bool:
        """
        Update tracker with current iteration's state.

        Parameters
        ----------
        iteration : int
            Current iteration/trial number.
        params : dict
            Current parameters (batched over param sets).
        continuous_outputs : dict
            Output from continuous forward pass. Must have "reserves" and "weights"
            with shape (n_param_sets, time_steps, ...).
        train_metrics_list : list of dict
            Training metrics for each param set.
        val_metrics_list : list of dict, optional
            Validation metrics for each param set.
        continuous_test_metrics_list : list of dict, optional
            Continuous test metrics for each param set.

        Returns
        -------
        bool
            True if this iteration improved the best metric, False otherwise.
        """
        # Always update "last" state — no copies needed, JAX arrays are immutable
        self.last_iteration = iteration
        self.last_params = params
        self.last_train_metrics = train_metrics_list
        self.last_val_metrics = val_metrics_list
        self.last_continuous_test_metrics = continuous_test_metrics_list
        self.last_continuous_outputs = {
            "reserves": continuous_outputs["reserves"],
            "weights": continuous_outputs["weights"],
        }

        # Compute selection value and param_idx (JAX ops, no device→host sync)
        selection_value, param_idx = compute_selection_metric(
            train_metrics_list,
            val_metrics_list,
            continuous_test_metrics_list,
            method=self.selection_method,
            metric=self.metric,
            min_threshold=self.min_threshold,
        )
        self.last_param_idx = param_idx

        # Branchless update of "best" state via jnp.where — avoids device→host sync.
        # The comparison produces a JAX boolean that stays on device; jnp.where
        # dispatches conditional copies without materialising the boolean to Python.
        improved = selection_value > self.best_metric_value

        if self.best_params is None:
            # First call: initialise all best state directly
            self.best_metric_value = selection_value
            self.best_iteration = iteration
            self.best_param_idx = param_idx
            self.best_params = params
            self.best_train_metrics = train_metrics_list
            self.best_val_metrics = val_metrics_list
            self.best_continuous_test_metrics = continuous_test_metrics_list
            self.best_continuous_outputs = {
                "reserves": continuous_outputs["reserves"],
                "weights": continuous_outputs["weights"],
            }
        else:
            # Branchless conditional update — all ops stay on device
            self.best_metric_value = jnp.where(improved, selection_value, self.best_metric_value)
            self.best_iteration = jnp.where(improved, iteration, self.best_iteration)
            self.best_param_idx = jnp.where(improved, param_idx, self.best_param_idx)
            self.best_params = tree_map(
                lambda new, old: jnp.where(improved, new, old),
                params, self.best_params,
            )
            self.best_train_metrics = [
                {k: jnp.where(improved, new[k], old[k]) for k in new}
                for new, old in zip(train_metrics_list, self.best_train_metrics)
            ]
            if val_metrics_list is not None and self.best_val_metrics is not None:
                self.best_val_metrics = [
                    {k: jnp.where(improved, new[k], old[k]) for k in new}
                    for new, old in zip(val_metrics_list, self.best_val_metrics)
                ]
            elif val_metrics_list is not None:
                self.best_val_metrics = val_metrics_list
            if continuous_test_metrics_list is not None and self.best_continuous_test_metrics is not None:
                self.best_continuous_test_metrics = [
                    {k: jnp.where(improved, new[k], old[k]) for k in new}
                    for new, old in zip(continuous_test_metrics_list, self.best_continuous_test_metrics)
                ]
            elif continuous_test_metrics_list is not None:
                self.best_continuous_test_metrics = continuous_test_metrics_list
            self.best_continuous_outputs = {
                k: jnp.where(improved, continuous_outputs[k], self.best_continuous_outputs[k])
                for k in ["reserves", "weights"]
            }

        return improved

    def select_param_set(self, params_dict: Dict, idx: int, n_param_sets: int) -> Dict:
        """
        Extract single param set from batched params.

        Parameters
        ----------
        params_dict : dict
            Batched parameters with shape (n_param_sets, ...) for each key.
        idx : int
            Index of param set to extract.
        n_param_sets : int
            Total number of param sets.

        Returns
        -------
        dict
            Parameters for single param set with shape (...) for each key.
        """
        if n_param_sets == 1:
            # Already single param set, just squeeze
            selected = {}
            for k, v in params_dict.items():
                if k == "subsidary_params":
                    selected[k] = v
                elif hasattr(v, 'shape') and len(v.shape) >= 1 and v.shape[0] == 1:
                    selected[k] = np.squeeze(v, axis=0) if isinstance(v, np.ndarray) else v[0]
                else:
                    selected[k] = v
            return selected
        else:
            # Select the param set at idx
            selected = {}
            for k, v in params_dict.items():
                if k == "subsidary_params":
                    selected[k] = v
                elif hasattr(v, 'shape') and len(v.shape) >= 1 and v.shape[0] == n_param_sets:
                    selected[k] = v[idx]
                else:
                    selected[k] = v
            return selected

    def get_results(self, n_param_sets: int, train_bout_length: int) -> Dict:
        """
        Get comprehensive results with both last and best state.

        Parameters
        ----------
        n_param_sets : int
            Number of parameter sets (for extracting correct shapes).
        train_bout_length : int
            Length of training period. Used to extract final reserves/weights
            at end of training (not end of test) for warm-starting.

        Returns
        -------
        dict
            Comprehensive results including:
            - last_* fields: State from most recent iteration
            - best_* fields: State from when selection metric was best
            - Selection metadata
        """
        # Extract final state at end of training period (for warm-starting next cycle)
        # continuous_outputs has shape (n_param_sets, total_time_steps, ...)
        # where total_time_steps = train + val + test
        # We want the state at train_bout_length (end of training)

        last_final_reserves = None
        last_final_weights = None
        best_final_reserves = None
        best_final_weights = None

        if self.last_continuous_outputs is not None:
            # Index at train_bout_length gives state at END of training period
            last_final_reserves = self.last_continuous_outputs["reserves"][:, train_bout_length - 1, :]
            last_final_weights = self.last_continuous_outputs["weights"][:, train_bout_length - 1, :]

        if self.best_continuous_outputs is not None:
            best_final_reserves = self.best_continuous_outputs["reserves"][:, train_bout_length - 1, :]
            best_final_weights = self.best_continuous_outputs["weights"][:, train_bout_length - 1, :]

        return {
            # Last iteration results
            "last_iteration": self.last_iteration,
            "last_params": self.last_params,
            "last_param_idx": self.last_param_idx,
            "last_train_metrics": self.last_train_metrics,
            "last_val_metrics": self.last_val_metrics,
            "last_continuous_test_metrics": self.last_continuous_test_metrics,
            "last_final_reserves": last_final_reserves,
            "last_final_weights": last_final_weights,

            # Best iteration results
            "best_iteration": self.best_iteration,
            "best_params": self.best_params,
            "best_param_idx": self.best_param_idx,
            "best_metric_value": self.best_metric_value,
            "best_train_metrics": self.best_train_metrics,
            "best_val_metrics": self.best_val_metrics,
            "best_continuous_test_metrics": self.best_continuous_test_metrics,
            "best_final_reserves": best_final_reserves,
            "best_final_weights": best_final_weights,

            # Selection info
            "selection_method": self.selection_method,
            "selection_metric": self.metric,
        }
