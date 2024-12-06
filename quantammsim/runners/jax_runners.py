import pickle
import os
import numpy as np

from jax.tree_util import Partial
from jax import jit, vmap
from jax import random

from quantammsim.utils.data_processing.historic_data_utils import (
    get_data_dict,
)

from quantammsim.utils.data_processing.price_data_fingerprint_utils import (
    load_run_fingerprints,
    load_price_data_if_fingerprints_in_dir_match,
)


from quantammsim.core_simulator.forward_pass import (
    forward_pass,
    forward_pass_nograd,
)
from quantammsim.core_simulator.windowing_utils import (
    get_indices,
    raw_trades_to_trade_array,
    raw_fee_like_amounts_to_fee_like_array,
)

from quantammsim.training.backpropagation import (
    update_from_partial_training_step_factory,
)
from quantammsim.core_simulator.param_utils import (
    load_or_init,
    load,
    default_set,
    dict_of_jnp_to_np,
    NumpyEncoder,
)

from quantammsim.core_simulator.result_exporter import (
    save_params,
    save_multi_params,
)
from quantammsim.core_simulator.param_utils import (
    dict_of_jnp_to_np,
    NumpyEncoder,
)

from quantammsim.runners.jax_runner_utils import (
    nan_rollback,
    Hashabledict,
    get_trades_and_fees,
    get_unique_tokens,
)

from quantammsim.pools.creator import create_pool
from functools import partial

from copy import deepcopy

import hashlib
from itertools import product
from tqdm import tqdm
import math

from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults


def train_on_historic_data(
    run_fingerprint,
    root=None,
    iterations_per_print=1,
    force_init=False,
    price_data=None,
    verbose=True,
):
    """
    Train a model on historical price data using JAX.

    This function trains a model on historical price data using JAX for optimization. It supports various
    hyperparameters and training configurations specified in the run_fingerprint.

    Parameters:
    -----------
    run_fingerprint : dict
        A dictionary containing all the configuration settings for the training run.
    root : str, optional
        The root directory for data and output files.
    iterations_per_print : int, optional
        The number of iterations between progress prints (default is 1).
    force_init : bool, optional
        If True, force reinitialization of parameters (default is False).
    price_data : array-like, optional
        The historical price data to train on. If None, data will be loaded from a file.
    verbose : bool, optional
        If True, print detailed progress information (default is True).

    Returns:
    --------
    None

    Notes:
    ------
    This function performs the following steps:
    1. Initializes or loads model parameters
    2. Prepares the training and test data
    3. Sets up the optimization process (SGD or Adam)
    4. Iteratively trains the model, updating parameters and learning rate
    5. Periodically saves the model state and prints progress
    6. Optionally returns the final model state and performance metrics

    The function uses JAX for efficient computation and supports various advanced features
    such as custom fee structures and different training data configurations.
    """

    for key, value in run_fingerprint_defaults.items():
        default_set(run_fingerprint, key, value)
    for key, value in run_fingerprint_defaults["optimisation_settings"].items():
        default_set(run_fingerprint["optimisation_settings"], key, value)
    if verbose:
        print("Run Fingerprint: ", run_fingerprint)

    rule = run_fingerprint["rule"]
    n_subsidary_rules = len(run_fingerprint["subsidary_pools"])
    chunk_period = run_fingerprint["chunk_period"]
    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    use_alt_lamb = run_fingerprint["use_alt_lamb"]
    use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
    fees = run_fingerprint["fees"]
    arb_fees = run_fingerprint["arb_fees"]
    gas_cost = run_fingerprint["gas_cost"]
    n_parameter_sets = run_fingerprint["optimisation_settings"]["n_parameter_sets"]
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    training_data_kind = run_fingerprint["optimisation_settings"]["training_data_kind"]
    include_flipped_training_data = run_fingerprint["optimisation_settings"][
        "include_flipped_training_data"
    ]
    arb_frequency = run_fingerprint["arb_frequency"]
    random_key = random.key(
        run_fingerprint["optimisation_settings"]["initial_random_key"]
    )

    inital_params = {
        "initial_memory_length": run_fingerprint["initial_memory_length"],
        "initial_memory_length_delta": run_fingerprint["initial_memory_length_delta"],
        "initial_k_per_day": run_fingerprint["initial_k_per_day"],
        "initial_weights_logits": run_fingerprint["initial_weights_logits"],
        "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
        "initial_raw_width": run_fingerprint["initial_raw_width"],
        "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
        "initial_pre_exp_scaling": run_fingerprint["maximum_change"],
    }

    unique_tokens = get_unique_tokens(run_fingerprint)
    n_tokens = len(unique_tokens)
    n_assets = n_tokens

    all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    all_sig_variations = all_sig_variations[(all_sig_variations != 0).sum(-1) > 1]
    all_sig_variations = all_sig_variations[np.any(all_sig_variations == 1, -1)]
    all_sig_variations = all_sig_variations[np.any(all_sig_variations == -1, -1)]
    all_sig_variations = tuple(map(tuple, all_sig_variations))

    np.random.seed(0)

    max_memory_days = 365.0

    if price_data is None:
        if verbose:
            print("loading data")
    data_dict = get_data_dict(
        unique_tokens,
        run_fingerprint,
        data_kind=run_fingerprint["optimisation_settings"]["training_data_kind"],
        root=root,
        max_memory_days=max_memory_days,
        start_date_string=run_fingerprint["startDateString"],
        end_time_string=run_fingerprint["endDateString"],
        start_time_test_string=run_fingerprint["endDateString"],
        end_time_test_string=run_fingerprint["endTestDateString"],
        max_mc_version=run_fingerprint["optimisation_settings"]["max_mc_version"],
        price_data=price_data,
    )

    if verbose:
        print("max_memory_days: ", max_memory_days)

    bout_length_window = data_dict["bout_length"] - run_fingerprint["bout_offset"]

    assert bout_length_window > 0

    # params, loaded = load_or_init(
    #     run_fingerprint,
    #     inital_params,
    #     n_tokens,
    #     n_subsidary_rules,
    #     chunk_period=chunk_period,
    #     force_init=force_init,
    #     load_method="last",
    #     n_parameter_sets=n_parameter_sets,
    # )

    # Create pool
    pool = create_pool(rule)

    # pool must be trainable
    assert pool.is_trainable(), "The selected pool must be trainable for this operation"
    params = pool.init_parameters(
        inital_params, run_fingerprint, n_tokens, n_parameter_sets
    )
    loaded = False

    if verbose:
        print("Using Loaded Params?: ", loaded)

    if loaded:
        offset = params["step"] + 1
        if verbose:
            print("loaded params ", params)
            print("starting at step ", offset)
        best_train_objective = np.array(params["objective"])
        params.pop("step")
        params.pop("test_objective")
        params.pop("train_objective")
        params.pop("hessian_trace")
        local_learning_rate = np.array(params["local_learning_rate"])
        params.pop("local_learning_rate")
        iterations_since_improvement = np.array(params["iterations_since_improvement"])
        params.pop("iterations_since_improvement")
    else:
        offset = 0

    params_in_axes_dict = pool.make_vmap_in_axes(
        params
    )
    base_static_dict = {
        "chunk_period": chunk_period,
        "bout_length": bout_length_window,
        "n_assets": n_assets,
        "maximum_change": run_fingerprint["maximum_change"],
        "weight_interpolation_period": weight_interpolation_period,
        "return_val": run_fingerprint["return_val"],
        "rule": run_fingerprint["rule"],
        "initial_pool_value": run_fingerprint["initial_pool_value"],
        "fees": fees,
        "arb_fees": arb_fees,
        "gas_cost": gas_cost,
        "run_type": "normal",
        "max_memory_days": 365.0,
        "initial_pool_value": 1000000.0,
        "training_data_kind": run_fingerprint["optimisation_settings"][
            "training_data_kind"
        ],
        "use_alt_lamb": use_alt_lamb,
        "use_pre_exp_scaling": use_pre_exp_scaling,
        "all_sig_variations": all_sig_variations,
        "weight_interpolation_method": weight_interpolation_method,
        "arb_frequency": arb_frequency,
    }

    partial_training_step = Partial(
        forward_pass,
        prices=data_dict["prices"],
        static_dict=Hashabledict(base_static_dict),
        pool=pool
    )
    partial_forward_pass_nograd_batch = Partial(
        forward_pass_nograd,
        prices=data_dict["prices"],
        static_dict=Hashabledict(base_static_dict),
        pool=pool
    )

    returns_train_static_dict = base_static_dict.copy()
    returns_train_static_dict["return_val"] = "returns_over_hodl"
    returns_train_static_dict["bout_length"] = data_dict["bout_length"]
    partial_forward_pass_nograd_batch_returns_train = Partial(
        forward_pass_nograd, static_dict=Hashabledict(returns_train_static_dict), pool=pool
    )

    returns_test_static_dict = base_static_dict.copy()
    returns_test_static_dict["return_val"] = "returns"
    returns_test_static_dict["bout_length"] = data_dict["bout_length_test"]
    partial_forward_pass_nograd_batch_returns_test = Partial(
        forward_pass_nograd,
        static_dict=Hashabledict(returns_test_static_dict),
        pool=pool
    )

    nograd_in_axes = [params_in_axes_dict, None, None]

    partial_forward_pass_nograd_returns_train = jit(
        vmap(
            partial_forward_pass_nograd_batch_returns_train,
            in_axes=nograd_in_axes,
        )
    )
    partial_forward_pass_nograd_returns_test = jit(
        vmap(
            partial_forward_pass_nograd_batch_returns_test,
            in_axes=nograd_in_axes,
        )
    )

    partial_fixed_training_step = Partial(
        partial_training_step, start_index=(data_dict["start_idx"], 0)
    )

    update_batch = update_from_partial_training_step_factory(
        partial_training_step,
        run_fingerprint["optimisation_settings"]["train_on_hessian_trace"],
        partial_fixed_training_step,
    )

    update = jit(
        vmap(
            update_batch,
            in_axes=[params_in_axes_dict, None, None],
        )
    )

    best_train_objective = -100.0
    best_test_objective = -100.0
    local_learning_rate = run_fingerprint["optimisation_settings"]["base_lr"]
    iterations_since_improvement = 0

    max_iterations_with_no_improvement = run_fingerprint["optimisation_settings"][
        "decay_lr_plateau"
    ]
    decay_lr_ratio = run_fingerprint["optimisation_settings"]["decay_lr_ratio"]
    min_lr = run_fingerprint["optimisation_settings"]["min_lr"]

    if run_fingerprint["optimisation_settings"]["method"] == "gradient_descent":
        if run_fingerprint["optimisation_settings"]["optimiser"] == "adam":
            import optax
            opt = optax.inject_hyperparams(optax.adam)(learning_rate=local_learning_rate)
            raise NotImplementedError
        elif run_fingerprint["optimisation_settings"]["optimiser"] != "sgd":
            raise NotImplementedError

        paramSteps = []
        trainingSteps = []
        testSteps = []
        objectiveSteps = []
        learningRateSteps = []
        interationsSinceImprovementSteps = []
        stepSteps = []
        for i in range(run_fingerprint["optimisation_settings"]["n_iterations"] + 1):
            step = i + offset
            start_indexes, random_key = get_indices(
                start_index=data_dict["start_idx"],
                bout_length=bout_length_window,
                len_prices=data_dict["end_idx"],
                key=random_key,
                optimisation_settings=run_fingerprint["optimisation_settings"],
            )

            params, objective_value, old_params, grads = update(
                params, start_indexes, local_learning_rate
            )

            params = nan_rollback(grads, params, old_params)

            train_objective = partial_forward_pass_nograd_returns_train(
                params,
                (data_dict["start_idx"], 0),
                data_dict["prices"],
            )

            test_objective = partial_forward_pass_nograd_returns_test(
                params,
                (data_dict["start_idx_test"], 0),
                data_dict["prices_test"],
            )
            paramSteps.append(deepcopy(params))
            trainingSteps.append(np.array(train_objective.copy()))
            testSteps.append(np.array(test_objective.copy()))
            objectiveSteps.append(np.array(objective_value.copy()))
            learningRateSteps.append(deepcopy(local_learning_rate))
            interationsSinceImprovementSteps.append(iterations_since_improvement)
            stepSteps.append(step)

            if (objective_value > best_train_objective).any():
                best_train_objective = np.array(objective_value.max())
                best_train_params = deepcopy(params)
                iterations_since_improvement = 0
            else:
                iterations_since_improvement += 1
            if iterations_since_improvement > max_iterations_with_no_improvement:
                local_learning_rate = local_learning_rate * decay_lr_ratio
                iterations_since_improvement = 0
                if local_learning_rate < min_lr:
                    local_learning_rate = min_lr
            if step % iterations_per_print == 0:
                if verbose:
                    print(step, "Objective: ", objective_value)
                    print(step, "train_objective", train_objective)
                    print(step, "test_objective", test_objective)
                    print(step, "local_learning_rate", local_learning_rate)
                save_multi_params(
                    deepcopy(run_fingerprint),
                    paramSteps,
                    testSteps,
                    trainingSteps,
                    objectiveSteps,
                    learningRateSteps,
                    interationsSinceImprovementSteps,
                    stepSteps,
                    sorted_tokens=True,
                )

                paramSteps = []
                trainingSteps = []
                testSteps = []
                objectiveSteps = []
                learningRateSteps = []
                interationsSinceImprovementSteps = []
                stepSteps = []
        if verbose:
            print("final objective value: ", objective_value)
            print("best train params", best_train_params)
    elif run_fingerprint["optimisation_settings"]["method"] == "optuna":
        import optuna
        import jax.numpy as jnp
        # define optuna study
        assert run_fingerprint["optimisation_settings"]["n_parameter_sets"] == 1, "Optuna only supports single parameter sets"
        # create objective function
        def objective(trial):
            trial_params = {}
            for key, value in params.items():
                if key != "subsidary_params":
                    trial_params[key] = jnp.array(
                            [trial.suggest_float(key + f"_{i}", -10, 10) for i in range(value.shape[1])]
                    )
                #     trial_params[key] = jnp.array(
                #         [trial.suggest_float(key, -10, 10)] * value.shape[1]
                # )
            print("return over hodl: ",
                partial_forward_pass_nograd_batch_returns_train(
                    trial_params,
                    (data_dict["start_idx"], 0),
                    data_dict["prices"],
                )
            )
            value = partial_forward_pass_nograd_batch(
                trial_params, (data_dict["start_idx"], 0)
            )
            print(
                run_fingerprint["return_val"] + ": ",
                value,
            )
            return -value
        study = optuna.create_study(sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=2000) 
    else:
        raise NotImplementedError

def do_run_on_historic_data(
    run_fingerprint,
    params={},
    root=None,
    price_data=None,
    verbose=False,
    raw_trades=None,
    fees=None,
    gas_cost=None,
    arb_fees=None,
    fees_df=None,
    gas_cost_df=None,
    arb_fees_df=None,
    do_test_period=False,
):
    """
    Execute a simulation run on historic data using specified parameters and settings.

    This function performs a simulation run on historical price data using the provided
    run fingerprint and parameters. It supports various options including multiple parameter sets,
    incorporating trades and fees, and running test periods.

    Parameters:
    -----------
    run_fingerprint : dict
        A dictionary containing the configuration and settings for the run.
    params : dict or list
        The parameters for the model. Can be a single set (dict) or multiple sets (list of dicts).
    root : str, optional
        The root directory for data files.
    price_data : array-like, optional
        Pre-loaded price data. If None, data will be loaded based on the run_fingerprint.
    verbose : bool, optional
        Whether to print detailed output (default is True).
    raw_trades : pd.DataFrame, optional
        Trade data to incorporate into the simulation. Each row should contain:
        unix timestamp of the trade (minute), token in (str), token out (str), and amount in (float).
    fees : float, optional
        Transaction fees to apply (overrides run_fingerprint value if provided).
    gas_cost : float, optional
        Gas costs for transactions (overrides run_fingerprint value if provided).
    arb_fees : float, optional
        Arbitrage fees to apply (overrides run_fingerprint value if provided).
    fees_df : pd.DataFrame, optional
        Transaction fees to apply over time. Each row should contain the unix timestamp and fee to be charged.
    gas_cost_df : pd.DataFrame, optional
        Gas costs for transactions over time. Each row should contain the unix timestamp and gas cost.
    arb_fees_df : pd.DataFrame, optional
        Arbitrage fees to apply over time. Each row should contain the unix timestamp and arb fee to be charged.
    do_test_period : bool, optional
        Whether to run the test period (default is False).

    Returns:
    --------
    dict or tuple of dicts
        If do_test_period is False:
            A dictionary containing the results of the simulation run for the training period.
        If do_test_period is True:
            A tuple of two dictionaries (train_results, test_results), containing the results
            for both the training and test periods.

        Results include final reserves, values, weights, and other relevant metrics.

    Notes:
    ------
    - This function is a core component of the quantamm system, integrating various aspects
      of the simulation including data handling, parameter optimization, and result calculation.
    - It supports both single and multi-parameter set runs, processing them in batches for efficiency.
    - The function creates a pool object based on the specified rule in the run_fingerprint.
    - Dynamic inputs (trades, fees, gas costs) are processed using the get_trades_and_fees function.
    - For multiple parameter sets, the function returns lists of output dictionaries instead of single dictionaries.
    """

    # Set default values for run_fingerprint and its optimisation_settings
    for key, value in run_fingerprint_defaults.items():
        default_set(run_fingerprint, key, value)
    for key, value in run_fingerprint_defaults["optimisation_settings"].items():
        default_set(run_fingerprint["optimisation_settings"], key, value)

    # Extract various settings from run_fingerprint
    chunk_period = run_fingerprint["chunk_period"]
    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    use_alt_lamb = run_fingerprint["use_alt_lamb"]
    use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
    n_parameter_sets = run_fingerprint["optimisation_settings"]["n_parameter_sets"]
    n_parameter_sets = 1
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    training_data_kind = run_fingerprint["optimisation_settings"]["training_data_kind"]
    arb_frequency = run_fingerprint["arb_frequency"]
    rule = run_fingerprint["rule"]

    # Create a list of unique tokens
    unique_tokens = get_unique_tokens(run_fingerprint)

    n_tokens = len(run_fingerprint["tokens"])
    n_assets = n_tokens

    # Generate all possible signature variations
    all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    all_sig_variations = all_sig_variations[(all_sig_variations != 0).sum(-1) > 1]
    all_sig_variations = all_sig_variations[np.any(all_sig_variations == 1, -1)]
    all_sig_variations = all_sig_variations[np.any(all_sig_variations == -1, -1)]
    all_sig_variations = tuple(map(tuple, all_sig_variations))

    max_memory_days = 365.0

    np.random.seed(0)

    dynamic_inputs_dict = get_trades_and_fees(
        run_fingerprint,
        raw_trades,
        fees_df,
        gas_cost_df,
        arb_fees_df,
        do_test_period=do_test_period,
    )

    # Load price data if not provided
    if price_data is None:
        if verbose:
            print("loading data")
    data_dict = get_data_dict(
        unique_tokens,
        run_fingerprint,
        data_kind=run_fingerprint["optimisation_settings"]["training_data_kind"],
        root=root,
        max_memory_days=max_memory_days,
        start_date_string=run_fingerprint["startDateString"],
        end_time_string=run_fingerprint["endDateString"],
        start_time_test_string=run_fingerprint["endDateString"],
        end_time_test_string=run_fingerprint["endTestDateString"],
        max_mc_version=run_fingerprint["optimisation_settings"]["max_mc_version"],
        price_data=price_data,
        do_test_period=do_test_period,
    )
    max_memory_days = data_dict["max_memory_days"]
    if verbose:
        print("max_memory_days: ", max_memory_days)

    if run_fingerprint["optimisation_settings"]["training_data_kind"] == "mc":
        # TODO: Handle MC data for post-training analysis
        raise NotImplementedError

    # create pool
    pool = create_pool(rule)

    inital_params = {
        "initial_memory_length": run_fingerprint["initial_memory_length"],
        "initial_memory_length_delta": run_fingerprint["initial_memory_length_delta"],
        "initial_k_per_day": run_fingerprint["initial_k_per_day"],
        "initial_weights_logits": run_fingerprint["initial_weights_logits"],
        "initial_log_amplitude": run_fingerprint["initial_log_amplitude"],
        "initial_raw_width": run_fingerprint["initial_raw_width"],
        "initial_raw_exponents": run_fingerprint["initial_raw_exponents"],
        "initial_pre_exp_scaling": run_fingerprint["maximum_change"],
    }

    base_static_dict = {
        "chunk_period": chunk_period,
        "bout_length": data_dict["bout_length"],
        "n_assets": n_assets,
        "maximum_change": run_fingerprint["maximum_change"],
        "weight_interpolation_period": weight_interpolation_period,
        "return_val": run_fingerprint["return_val"],
        "rule": run_fingerprint["rule"],
        "initial_pool_value": run_fingerprint["initial_pool_value"],
        "fees": fees if fees is not None else run_fingerprint["fees"],
        "arb_fees": arb_fees if arb_fees is not None else run_fingerprint["arb_fees"],
        "gas_cost": gas_cost if gas_cost is not None else run_fingerprint["gas_cost"],
        "run_type": "normal",
        "max_memory_days": 365.0,
        "training_data_kind": run_fingerprint["optimisation_settings"][
            "training_data_kind"
        ],
        "use_alt_lamb": use_alt_lamb,
        "use_pre_exp_scaling": use_pre_exp_scaling,
        "all_sig_variations": all_sig_variations,
        "weight_interpolation_method": weight_interpolation_method,
        "arb_frequency": arb_frequency,
        "do_trades": False if raw_trades is None else run_fingerprint["do_trades"],
        "tokens": tuple(run_fingerprint["tokens"]),
        "startDateString": run_fingerprint["startDateString"],
        "endDateString": run_fingerprint["endDateString"],
        "endTestDateString": run_fingerprint["endTestDateString"],
        "do_arb": run_fingerprint["do_arb"],
        "arb_quality": run_fingerprint["arb_quality"],
    }

    # Create static dictionaries for training and testing
    reserves_values_train_static_dict = base_static_dict.copy()
    reserves_values_train_static_dict["return_val"] = "reserves_and_values"
    reserves_values_train_static_dict["bout_length"] = data_dict["bout_length"]
    partial_forward_pass_nograd_batch_reserves_values_train = Partial(
        forward_pass_nograd,
        static_dict=Hashabledict(reserves_values_train_static_dict),
        pool=pool,
    )

    if do_test_period:
        reserves_values_test_static_dict = base_static_dict.copy()
        reserves_values_test_static_dict["return_val"] = "reserves_and_values"
        reserves_values_test_static_dict["bout_length"] = data_dict["bout_length_test"]
        partial_forward_pass_nograd_batch_reserves_values_test = (
            Partial(
                forward_pass_nograd,
                static_dict=Hashabledict(reserves_values_test_static_dict),
                pool=pool,
            )
        )

    # Ensure params is a list
    if isinstance(params, dict):
        params = [params]

    total_params = len(params)
    update_every = max(
        math.floor(total_params / 10), 1
    )  # Update every 10% of the way through the number of param sets
    output_dicts = []
    if do_test_period:
        output_dicts_test = []

    # Process each set of parameters
    for i in range(total_params):
        param = params[i]
        if i % update_every == 0:
            if verbose:
                tqdm.write(f"Processed {i+1} out of {total_params} parameters.")

        # Run forward pass for training data
        output_dict = partial_forward_pass_nograd_batch_reserves_values_train(
            param,
            (data_dict["start_idx"], 0),
            data_dict["prices"],
            dynamic_inputs_dict["train_period_trades"],
            dynamic_inputs_dict["fees_array"],
            dynamic_inputs_dict["gas_cost_array"],
            dynamic_inputs_dict["arb_fees_array"],
        )
        output_dicts.append(output_dict)

        # Run forward pass for test data if required
        if do_test_period:
            output_dict_test = partial_forward_pass_nograd_batch_reserves_values_test(
                param,
                (data_dict["start_idx_test"], 0),
                data_dict["prices"],
                dynamic_inputs_dict["test_period_trades"],
                dynamic_inputs_dict["test_fees_array"],
                dynamic_inputs_dict["test_gas_cost_array"],
                dynamic_inputs_dict["test_arb_fees_array"],
            )
            output_dicts_test.append(output_dict_test)

    # If only one set of parameters, return as single dict instead of list
    if len(output_dicts) == 1:
        output_dicts = output_dicts[0]
        if do_test_period:
            output_dicts_test = output_dicts_test[0]
    # Return results
    if do_test_period:
        return output_dicts, output_dicts_test
    else:
        return output_dicts
