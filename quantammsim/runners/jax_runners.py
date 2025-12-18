import numpy as np
from copy import deepcopy

from itertools import product
from tqdm import tqdm
import math
import gc
import os
import hashlib, json
from jax.tree_util import Partial
from jax import jit, vmap, random
from jax import clear_caches
from jax.tree_util import tree_map

from quantammsim.utils.data_processing.historic_data_utils import (
    get_data_dict,
)

from quantammsim.core_simulator.forward_pass import (
    forward_pass,
    forward_pass_nograd,
    _calculate_return_value,
)
from quantammsim.core_simulator.windowing_utils import get_indices, filter_coarse_weights_by_data_indices

from quantammsim.training.backpropagation import (
    update_from_partial_training_step_factory,
    update_from_partial_training_step_factory_with_optax,
    create_opt_state_in_axes_dict,
    create_optimizer_chain,
)
from quantammsim.core_simulator.param_utils import (
    recursive_default_set,
    check_run_fingerprint,
    memory_days_to_logit_lamb,
    retrieve_best,
    process_initial_values,
    get_run_location,
)

from quantammsim.core_simulator.result_exporter import (
    save_multi_params,
)

from quantammsim.runners.jax_runner_utils import (
    nan_param_reinit,
    has_nan_grads,
    Hashabledict,
    NestedHashabledict,
    HashableArrayWrapper,
    get_trades_and_fees,
    get_unique_tokens,
    OptunaManager,
    generate_evaluation_points,
    create_trial_params,
)

from quantammsim.pools.creator import create_pool

from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
import jax.numpy as jnp


def train_on_historic_data(
    run_fingerprint,
    root=None,
    iterations_per_print=1,
    force_init=False,
    price_data=None,
    verbose=True,
    run_location=None,
):
    """
    Train a model on historical price data using JAX.

    This function trains a model on historical price data using JAX for optimization.
    It supports various hyperparameters and training configurations specified in the run_fingerprint.

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
    run_location : str, optional
        The location of the run to load from. If None, the run will be initialized.

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

    recursive_default_set(run_fingerprint, run_fingerprint_defaults)
    check_run_fingerprint(run_fingerprint)
    if verbose:
        print("Run Fingerprint: ", run_fingerprint)
    rule = run_fingerprint["rule"]
    chunk_period = run_fingerprint["chunk_period"]
    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    use_alt_lamb = run_fingerprint["use_alt_lamb"]
    use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
    fees = run_fingerprint["fees"]
    arb_fees = run_fingerprint["arb_fees"]
    gas_cost = run_fingerprint["gas_cost"]
    n_parameter_sets = run_fingerprint["optimisation_settings"]["n_parameter_sets"]
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    arb_frequency = run_fingerprint["arb_frequency"]
    random_key = random.key(
        run_fingerprint["optimisation_settings"]["initial_random_key"]
    )

    initial_params = {
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

    # all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    # all_sig_variations = all_sig_variations[(all_sig_variations != 0).sum(-1) > 1]
    # all_sig_variations = all_sig_variations[np.any(all_sig_variations == 1, -1)]
    # all_sig_variations = all_sig_variations[np.any(all_sig_variations == -1, -1)]
    # all_sig_variations = tuple(map(tuple, all_sig_variations))

    all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    # Keep only variations with exactly one +1 and one -1
    all_sig_variations = all_sig_variations[(all_sig_variations == 1).sum(-1) == 1]
    all_sig_variations = all_sig_variations[(all_sig_variations == -1).sum(-1) == 1]
    all_sig_variations = tuple(map(tuple, all_sig_variations))

    np.random.seed(int(hashlib.sha256(
            json.dumps(run_fingerprint, sort_keys=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest(), 16) % (2**32)
    )

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
        do_test_period=True,
    )
    max_memory_days = data_dict["max_memory_days"]
    print("max_memory_days: ", max_memory_days)

    bout_length_window = data_dict["bout_length"] - run_fingerprint["bout_offset"]

    assert bout_length_window > 0

    if run_location is None:
        run_location = './results/' + get_run_location(run_fingerprint) + ".json"

    if os.path.isfile(run_location):
        print("Loading from: ", run_location)
        print("found file")
        params, step = retrieve_best(run_location, "best_train_objective", False, None)
        loaded = True
    else:
        loaded = False
    # Create pool
    pool = create_pool(rule)

    # pool must be trainable
    assert pool.is_trainable(), "The selected pool must be trainable for this operation"

    if not loaded:
        params = pool.init_parameters(
            initial_params, run_fingerprint, n_tokens, n_parameter_sets, prices=data_dict["prices"], noise="spectral"
        )
        offset = 0
    else:
        if verbose:
            print("Using Loaded Params?: ", loaded)
        offset = step + 1
        if verbose:
            print("loaded params ", params)
            print("starting at step ", offset)
        best_train_objective = np.array(params["objective"])
        for key in ["step", "test_objective", "train_objective", "hessian_trace", "local_learning_rate", "iterations_since_improvement", "objective"]:
            if key in params:
                params.pop(key)
        if run_fingerprint["optimisation_settings"]["method"] == "optuna":
            n_parameter_sets = 1
        for key, value in params.items():
            params[key] = process_initial_values(
                params, key, n_assets, n_parameter_sets, force_scalar=True
            )
        params["subsidary_params"] = []
        params = pool.add_noise(params, "gaussian", n_parameter_sets, noise_scale=0.1)

    params_in_axes_dict = pool.make_vmap_in_axes(params)
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
        "max_memory_days": run_fingerprint["max_memory_days"],
        "training_data_kind": run_fingerprint["optimisation_settings"][
            "training_data_kind"
        ],
        "tokens": tuple(run_fingerprint["tokens"]),
        "use_alt_lamb": use_alt_lamb,
        "use_pre_exp_scaling": use_pre_exp_scaling,
        "all_sig_variations": all_sig_variations,
        "weight_interpolation_method": weight_interpolation_method,
        "arb_frequency": arb_frequency,
        "do_arb": run_fingerprint["do_arb"],
        "arb_quality": run_fingerprint["arb_quality"],
        "numeraire": run_fingerprint["numeraire"],
        "do_trades": False,
        "noise_trader_ratio": run_fingerprint["noise_trader_ratio"],
        "minimum_weight": run_fingerprint["minimum_weight"],
        "ste_max_change": run_fingerprint["ste_max_change"],
        "ste_min_max_weight": run_fingerprint["ste_min_max_weight"],
    }

    partial_training_step = Partial(
        forward_pass,
        prices=data_dict["prices"],
        static_dict=Hashabledict(base_static_dict),
        pool=pool,
    )
    partial_forward_pass_nograd_batch = Partial(
        forward_pass_nograd,
        prices=data_dict["prices"],
        static_dict=Hashabledict(base_static_dict),
        pool=pool,
    )

    base_static_dict_test = base_static_dict.copy()
    base_static_dict_test["bout_length"] = data_dict["bout_length_test"]
    partial_forward_pass_nograd_batch_test = Partial(
        forward_pass_nograd,
        prices=data_dict["prices"],
        static_dict=Hashabledict(base_static_dict_test),
        pool=pool,
    )

    returns_train_static_dict = base_static_dict.copy()
    returns_train_static_dict["return_val"] = "returns_over_hodl"
    returns_train_static_dict["bout_length"] = data_dict["bout_length"]
    partial_forward_pass_nograd_batch_returns_train = Partial(
        forward_pass_nograd,
        static_dict=Hashabledict(returns_train_static_dict),
        pool=pool,
    )

    returns_test_static_dict = base_static_dict.copy()
    returns_test_static_dict["return_val"] = "returns_over_hodl"
    returns_test_static_dict["bout_length"] = data_dict["bout_length_test"]
    partial_forward_pass_nograd_batch_returns_test = Partial(
        forward_pass_nograd,
        static_dict=Hashabledict(returns_test_static_dict),
        pool=pool,
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

    # Setup regularization if requested
    regularisation_step = None
    regularisation_weight = run_fingerprint["optimisation_settings"].get("regularisation_weight", 1e-4)
    regularisation_type = run_fingerprint["optimisation_settings"].get("regularisation_type", None)

    if regularisation_type == "entropy":
        entropy_static_dict = base_static_dict.copy()
        entropy_static_dict["return_val"] = "reserves_and_values"
        
        # Helper to calculate negative entropy for a single sample
        def calculate_neg_entropy_single(params, start_index):
            # We bind dynamic inputs as None implicitly by not passing them
            output = forward_pass(
                params, 
                start_index, 
                prices=data_dict["prices"],
                pool=pool, 
                static_dict=Hashabledict(entropy_static_dict)
            )
            weights = output["weights"]
            # Clip for numerical stability
            w = jnp.clip(weights, 1e-10, 1.0)
            entropy = -jnp.sum(w * jnp.log(w), axis=-1)
            # Return negative entropy (since we minimize objective)
            return -jnp.mean(entropy)

        # Vectorize over batch (params shared, start_indexes batched)
        batched_neg_entropy_step = jit(vmap(calculate_neg_entropy_single, in_axes=(None, 0)))

        # Define the final step function that averages over batch
        def entropy_regularisation_step_fn(params, start_indexes):
            neg_entropies = batched_neg_entropy_step(params, start_indexes)
            return jnp.mean(neg_entropies)
            
        regularisation_step = entropy_regularisation_step_fn

    best_train_objective = -100.0
    local_learning_rate = run_fingerprint["optimisation_settings"]["base_lr"]
    iterations_since_improvement = 0

    max_iterations_with_no_improvement = run_fingerprint["optimisation_settings"][
        "decay_lr_plateau"
    ]
    decay_lr_ratio = run_fingerprint["optimisation_settings"]["decay_lr_ratio"]
    min_lr = run_fingerprint["optimisation_settings"]["min_lr"]

    if run_fingerprint["optimisation_settings"]["method"] == "gradient_descent":
        if run_fingerprint["optimisation_settings"]["optimiser"] in ["adam", "adamw"]:
            import optax

            # Create Adam optimizer with the specified learning rate
            optimizer = create_optimizer_chain(run_fingerprint)

            # Initialize optimizer state for each parameter set
            # For multiple parameter sets, each needs its own optimizer state
            # if n_parameter_sets > 1:
            # Use vmap to vectorize optimizer initialization over parameter sets
            init_optimizer = lambda params: optimizer.init(params)
            batched_init = vmap(init_optimizer, in_axes=[params_in_axes_dict])
            opt_state = batched_init(params)
            # else:
            # opt_state = optimizer.init(params)

            opt_state_in_axes_dict = create_opt_state_in_axes_dict(opt_state)
            # Use optax-based update function
            update_batch = update_from_partial_training_step_factory_with_optax(
                partial_training_step,
                optimizer,
                run_fingerprint["optimisation_settings"]["train_on_hessian_trace"],
                partial_fixed_training_step,
                regularisation_step=regularisation_step,
                regularisation_weight=regularisation_weight,
            )
            update = jit(
                vmap(
                    update_batch,
                    in_axes=[params_in_axes_dict, None, None, opt_state_in_axes_dict],
                )
            )

        elif run_fingerprint["optimisation_settings"]["optimiser"] == "sgd":

            update_batch = update_from_partial_training_step_factory(
                partial_training_step,
                run_fingerprint["optimisation_settings"]["train_on_hessian_trace"],
                partial_fixed_training_step,
                regularisation_step=regularisation_step,
                regularisation_weight=regularisation_weight,
            )

            update = jit(
                vmap(
                    update_batch,
                    in_axes=[params_in_axes_dict, None, None],
                )
            )

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
            if run_fingerprint["optimisation_settings"]["optimiser"] in ["adam", "adamw"]:
                # Adam update with state maintenance
                params, objective_value, old_params, grads, opt_state = update(
                    params, start_indexes, local_learning_rate, opt_state
                )
            else:
                # Regular SGD update
                params, objective_value, old_params, grads = update(
                    params, start_indexes, local_learning_rate
                )

            params = nan_param_reinit(
                params,
                grads,
                pool,
                initial_params,
                run_fingerprint,
                n_tokens,
                n_parameter_sets,
            )


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
                    print(step, "train (returns over hodl)", train_objective)
                    print(step, "test (returns over hodl)", test_objective)
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
        return best_train_params
    elif run_fingerprint["optimisation_settings"]["method"] == "optuna":

        n_evaluation_points = 20
        min_spacing = data_dict["bout_length"] // 2  # E

        run_fingerprint["optimisation_settings"]["n_parameter_sets"] = 1
        # assert run_fingerprint["optimisation_settings"]["n_parameter_sets"] == 1, \
        #     "Optuna only supports single parameter sets"

        # Generate and store evaluation points
        if "evaluation_starts" not in run_fingerprint:
            evaluation_starts = generate_evaluation_points(
                data_dict["start_idx"],
                data_dict["end_idx"],
                bout_length_window,
                n_evaluation_points,
                min_spacing,
                run_fingerprint["optimisation_settings"]["initial_random_key"],
            )
            run_fingerprint["evaluation_starts"] = [int(e) for e in evaluation_starts]
        else:
            evaluation_starts = run_fingerprint["evaluation_starts"]

        reserves_values_train_static_dict = base_static_dict.copy()
        reserves_values_train_static_dict["return_val"] = "reserves_and_values"
        reserves_values_train_static_dict["bout_length"] = data_dict["bout_length"]
        partial_forward_pass_nograd_batch_reserves_values_train = jit(
            Partial(
                forward_pass_nograd,
                static_dict=Hashabledict(reserves_values_train_static_dict),
                pool=pool,
            )
        )

        reserves_values_test_static_dict = base_static_dict.copy()
        reserves_values_test_static_dict["return_val"] = "reserves_and_values"
        reserves_values_test_static_dict["bout_length"] = data_dict["bout_length_test"]
        partial_forward_pass_nograd_batch_reserves_values_test = jit(
            Partial(
                forward_pass_nograd,
                static_dict=Hashabledict(reserves_values_test_static_dict),
                pool=pool,
            )
        )

        # Initialize Optuna manager
        optuna_manager = OptunaManager(run_fingerprint)
        optuna_manager.setup_study(
            multi_objective=run_fingerprint["optimisation_settings"]["optuna_settings"][
                "multi_objective"
            ]
        )

        run_fingerprint["optimisation_settings"]["optuna_settings"]["parameter_config"][
            "logit_lamb"
        ] = {
            "low": float(
                memory_days_to_logit_lamb(
                    0.5, chunk_period=base_static_dict["chunk_period"]
                )
            ),
            "high": float(
                memory_days_to_logit_lamb(
                    base_static_dict["max_memory_days"],
                    chunk_period=base_static_dict["chunk_period"],
                )
            ),
            "log_scale": False,
        }

        # Create objective with parameter configuration and validation
        def objective(trial):
            try:
                param_config = run_fingerprint["optimisation_settings"][
                    "optuna_settings"
                ]["parameter_config"]

                if run_fingerprint["optimisation_settings"]["optuna_settings"][
                    "make_scalar"
                ]:
                    # Set scalar=True for all parameter configurations
                    for param_key in param_config:
                        param_config[param_key]["scalar"] = True

                # param_config["log_k"]["scalar"] = False
                # param_config["k_per_day"]["scalar"] = False
                trial_params = create_trial_params(
                    trial, {}, params, run_fingerprint, n_assets, expand_around=True
                )
                # Training evaluation
                train_outputs = partial_forward_pass_nograd_batch_reserves_values_train(
                    trial_params,
                    (data_dict["start_idx"], 0),
                    data_dict["prices"],
                )
                # Calculate objectives for each evaluation point through slicing
                train_objectives = []
                for start_offset in evaluation_starts:
                    # Calculate relative indices for slicing
                    start_idx = start_offset - data_dict["start_idx"]
                    end_idx = start_idx + data_dict["bout_length"]

                    # Slice the relevant portions of the full trajectory
                    train_value = _calculate_return_value(
                        run_fingerprint["return_val"],
                        train_outputs["reserves"][start_idx:end_idx],
                        data_dict["prices"][start_idx:end_idx],
                        train_outputs["value"][start_idx:end_idx],
                        initial_reserves=train_outputs["reserves"][start_idx],
                    )
                    train_objectives.append(train_value)

                mean_train_value = jnp.sum(train_objectives) / len(train_objectives)
                train_value = _calculate_return_value(
                    run_fingerprint["return_val"],
                    train_outputs["reserves"],
                    train_outputs["prices"],
                    train_outputs["value"],
                )

                train_sharpe = _calculate_return_value(
                    "sharpe",
                    train_outputs["reserves"],
                    train_outputs["prices"],
                    train_outputs["value"],
                )

                train_return = (
                    train_outputs["value"][-1] / train_outputs["value"][0] - 1.0
                )

                train_returns_over_hodl = _calculate_return_value(
                    "returns_over_hodl",
                    train_outputs["reserves"],
                    train_outputs["prices"],
                    train_outputs["value"],
                    initial_reserves=train_outputs["reserves"][0],
                )

                # Validation (test period) evaluation
                validation_outputs = (
                    partial_forward_pass_nograd_batch_reserves_values_test(
                        trial_params,
                        (data_dict["start_idx_test"], 0),
                        data_dict["prices"],
                    )
                )

                validation_value = _calculate_return_value(
                    run_fingerprint["return_val"],
                    validation_outputs["reserves"],
                    validation_outputs["prices"],
                    validation_outputs["value"],
                )

                validation_sharpe = _calculate_return_value(
                    "sharpe",
                    validation_outputs["reserves"],
                    validation_outputs["prices"],
                    validation_outputs["value"],
                )

                validation_return = (
                    validation_outputs["value"][-1] / validation_outputs["value"][0]
                    - 1.0
                )

                validation_returns_over_hodl = _calculate_return_value(
                    "returns_over_hodl",
                    validation_outputs["reserves"],
                    validation_outputs["prices"],
                    validation_outputs["value"],
                    initial_reserves=validation_outputs["reserves"][0],
                )
                # Log both training and validation metrics
                # optuna_manager.logger.info(f"Trial {trial.number}:")
                optuna_manager.logger.info(
                    f"Training {trial.number}, Return over HODL: {train_returns_over_hodl}"
                )
                optuna_manager.logger.info(
                    f"Training {trial.number}, Return: {train_return}"
                )
                optuna_manager.logger.info(
                    f"Training {trial.number}, Sharpe: {train_sharpe}"
                )
                optuna_manager.logger.info(
                    f"Training {trial.number}, {run_fingerprint['return_val']}: {train_value}"
                )
                optuna_manager.logger.info(
                    f"Validation {trial.number}, Return over HODL: {validation_returns_over_hodl}"
                )
                optuna_manager.logger.info(
                    f"Validation {trial.number}, Return: {validation_return}"
                )
                optuna_manager.logger.info(
                    f"Validation {trial.number}, Sharpe: {validation_sharpe}"
                )
                optuna_manager.logger.info(
                    f"Validation {trial.number}, {run_fingerprint['return_val']}: {validation_value}"
                )
                for i, value in enumerate(train_objectives):
                    optuna_manager.logger.info(
                        f"Training {trial.number},  Evaluation point {i}: {value}"
                    )
                optuna_manager.logger.info(
                    f"Training {trial.number},  Mean value: {mean_train_value}"
                )
                # Store validation value as a trial attribute
                trial.set_user_attr("validation_value", validation_value)
                trial.set_user_attr(
                    "validation_returns_over_hodl", validation_returns_over_hodl
                )
                trial.set_user_attr("validation_sharpe", validation_sharpe)
                trial.set_user_attr("validation_return", validation_return)
                trial.set_user_attr("train_value", train_value)
                trial.set_user_attr("train_returns_over_hodl", train_returns_over_hodl)
                trial.set_user_attr("train_sharpe", train_sharpe)
                trial.set_user_attr("train_return", train_return)
                trial.set_user_attr("train_objectives", train_objectives)
                trial.set_user_attr("mean_train_value", mean_train_value)

                if run_fingerprint["optimisation_settings"]["optuna_settings"][
                    "multi_objective"
                ]:
                    return (
                        np.mean(train_objectives),  # mean_return
                        np.min(train_objectives),  # worst_case
                        -np.std(train_objectives),  # stability
                    )
                else:
                    return mean_train_value  # Still optimize on training value

            except Exception as e:
                optuna_manager.logger.error(f"Trial {trial.number} failed: {str(e)}")
                raise e

        # Run optimization
        optuna_manager.optimize(objective)
        if verbose:
            if run_fingerprint["optimisation_settings"]["optuna_settings"][
                "multi_objective"
            ]:
                print("Best trials:")
                print(f"  Training Value: {optuna_manager.study.best_trials}")
                print(
                    f"  Validation Value: {[trial.user_attrs['validation_value'] for trial in optuna_manager.study.best_trials]}"
                )
                print(
                    f"  Params: {[trial.params for trial in optuna_manager.study.best_trials]}"
                )
            else:
                print("Best trial:")
                print(f"  Training Value: {optuna_manager.study.best_value}")
                print(
                    f"  Validation Value: {optuna_manager.study.best_trial.user_attrs['validation_value']}"
                )
                print(f"  Params: {optuna_manager.study.best_params}")
        return optuna_manager.study.best_trials
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
    lp_supply_df=None,
    do_test_period=False,
    low_data_mode=False,
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
        Transaction fees to apply over time.
        Each row should contain the unix timestamp and fee to be charged.
    gas_cost_df : pd.DataFrame, optional
        Gas costs for transactions over time.
        Each row should contain the unix timestamp and gas cost.
    arb_fees_df : pd.DataFrame, optional
        Arbitrage fees to apply over time.
        Each row should contain the unix timestamp and arb fee to be charged.
    do_test_period : bool, optional
        Whether to run the test period (default is False).
    low_data_mode : bool, optional
        Whether to delete the prices from the output dictionary (default is False).

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
    recursive_default_set(run_fingerprint, run_fingerprint_defaults)
    # Extract various settings from run_fingerprint
    chunk_period = run_fingerprint["chunk_period"]
    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    use_alt_lamb = run_fingerprint["use_alt_lamb"]
    use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    arb_frequency = run_fingerprint["arb_frequency"]
    rule = run_fingerprint["rule"]

    # Create a list of unique tokens
    unique_tokens = get_unique_tokens(run_fingerprint)

    n_tokens = len(run_fingerprint["tokens"])
    n_assets = n_tokens

    # Generate all possible signature variations
    # all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    # all_sig_variations = all_sig_variations[(all_sig_variations != 0).sum(-1) > 1]
    # all_sig_variations = all_sig_variations[np.any(all_sig_variations == 1, -1)]
    # all_sig_variations = all_sig_variations[np.any(all_sig_variations == -1, -1)]
    # all_sig_variations = tuple(map(tuple, all_sig_variations))

    all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    # Keep only variations with exactly one +1 and one -1
    all_sig_variations = all_sig_variations[(all_sig_variations == 1).sum(-1) == 1]
    all_sig_variations = all_sig_variations[(all_sig_variations == -1).sum(-1) == 1]
    all_sig_variations = tuple(map(tuple, all_sig_variations))

    max_memory_days = run_fingerprint["max_memory_days"]

    np.random.seed(0)

    dynamic_inputs_dict = get_trades_and_fees(
        run_fingerprint,
        raw_trades,
        fees_df,
        gas_cost_df,
        arb_fees_df,
        lp_supply_df,
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
        "numeraire": run_fingerprint["numeraire"],
        "noise_trader_ratio": run_fingerprint["noise_trader_ratio"],
        "minimum_weight": run_fingerprint["minimum_weight"],
        "ste_max_change": run_fingerprint["ste_max_change"],
        "ste_min_max_weight": run_fingerprint["ste_min_max_weight"],
    }

    # Create static dictionaries for training and testing
    reserves_values_train_static_dict = base_static_dict.copy()
    reserves_values_train_static_dict["return_val"] = "reserves_and_values"
    reserves_values_train_static_dict["bout_length"] = data_dict["bout_length"]
    partial_forward_pass_nograd_batch_reserves_values_train = jit(
        Partial(
            forward_pass_nograd,
            static_dict=Hashabledict(reserves_values_train_static_dict),
            pool=pool,
        )
    )

    if do_test_period:
        reserves_values_test_static_dict = base_static_dict.copy()
        reserves_values_test_static_dict["return_val"] = "reserves_and_values"
        reserves_values_test_static_dict["bout_length"] = data_dict["bout_length_test"]
        partial_forward_pass_nograd_batch_reserves_values_test = jit(
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
        if low_data_mode:
            output_dict["final_prices"] = output_dict["prices"][-1]
            output_dict["initial_reserves"] = output_dict["reserves"][0]
            output_dict["initial_prices"] = output_dict["prices"][0]
            del output_dict["prices"]
            del output_dict["reserves"]
            del output_dict["value"]
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
            if low_data_mode:
                output_dict_test["final_prices"] = output_dict_test["prices"][-1]
                output_dict_test["initial_reserves"] = output_dict_test["reserves"][0]
                output_dict_test["initial_prices"] = output_dict_test["prices"][0]
                del output_dict_test["prices"]
                del output_dict_test["reserves"]
                del output_dict_test["value"]
            output_dicts_test.append(output_dict_test)

    # out = partial_forward_pass_nograd_batch(
    #     params[0],
    #     (data_dict["start_idx"], 0),
    # )
    # raise Exception("stop")
    # If only one set of parameters, return as single dict instead of list
    if len(output_dicts) == 1:
        output_dicts = output_dicts[0]
        output_dicts["data_dict"] = data_dict
        if do_test_period:
            output_dicts_test = output_dicts_test[0]
    # Return results
    gc.collect()
    gc.collect()
    # Clear any cached JAX computations to free memory
    clear_caches()
    if do_test_period:
        return output_dicts, output_dicts_test
    else:
        return output_dicts

def do_run_on_historic_data_with_provided_coarse_weights(
    run_fingerprint,
    coarse_weights,
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
    lp_supply_df=None,
    do_test_period=False,
    low_data_mode=False,
):
    """
    Execute a simulation run on historic data using specified parameters and settings, including provided coarse weights.

    This function performs a simulation run on historical price data using the provided
    run fingerprint, weights and parameters. It supports various options including multiple parameter sets,
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
        Transaction fees to apply over time.
        Each row should contain the unix timestamp and fee to be charged.
    gas_cost_df : pd.DataFrame, optional
        Gas costs for transactions over time.
        Each row should contain the unix timestamp and gas cost.
    arb_fees_df : pd.DataFrame, optional
        Arbitrage fees to apply over time.
        Each row should contain the unix timestamp and arb fee to be charged.
    lp_supply_df : pd.DataFrame, optional
        LP supply over time.
        Each row should contain the unix timestamp and LP supply.
    do_test_period : bool, optional
        Whether to run the test period (default is False).
    low_data_mode : bool, optional
        Whether to delete the prices from the output dictionary (default is False).

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
    from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
        _jax_calc_coarse_weights,
        _jax_fine_weights_from_actual_starts_and_diffs,
    )
    from quantammsim.pools.G3M.quantamm.quantamm_reserves import (
        _jax_calc_quantAMM_reserves_with_dynamic_inputs,
    )

    # Set default values for run_fingerprint and its optimisation_settings
    recursive_default_set(run_fingerprint, run_fingerprint_defaults)
    # Extract various settings from run_fingerprint
    chunk_period = run_fingerprint["chunk_period"]
    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    use_alt_lamb = run_fingerprint["use_alt_lamb"]
    use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    arb_frequency = run_fingerprint["arb_frequency"]
    rule = run_fingerprint["rule"]

    # Create a list of unique tokens
    unique_tokens = get_unique_tokens(run_fingerprint)

    n_tokens = len(run_fingerprint["tokens"])
    n_assets = n_tokens

    # Generate all possible signature variations
    # all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    # all_sig_variations = all_sig_variations[(all_sig_variations != 0).sum(-1) > 1]
    # all_sig_variations = all_sig_variations[np.any(all_sig_variations == 1, -1)]
    # all_sig_variations = all_sig_variations[np.any(all_sig_variations == -1, -1)]
    # all_sig_variations = tuple(map(tuple, all_sig_variations))

    all_sig_variations = np.array(list(product([1, 0, -1], repeat=n_assets)))
    # Keep only variations with exactly one +1 and one -1
    all_sig_variations = all_sig_variations[(all_sig_variations == 1).sum(-1) == 1]
    all_sig_variations = all_sig_variations[(all_sig_variations == -1).sum(-1) == 1]
    all_sig_variations = tuple(map(tuple, all_sig_variations))

    max_memory_days = run_fingerprint["max_memory_days"]

    np.random.seed(0)

    dynamic_inputs_dict = get_trades_and_fees(
        run_fingerprint,
        raw_trades,
        fees_df,
        gas_cost_df,
        arb_fees_df,
        lp_supply_df,
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
        "numeraire": run_fingerprint["numeraire"],
        "noise_trader_ratio": run_fingerprint["noise_trader_ratio"],
        "minimum_weight": run_fingerprint["minimum_weight"],
        "ste_max_change": run_fingerprint["ste_max_change"],
        "ste_min_max_weight": run_fingerprint["ste_min_max_weight"],
    }

    # Create static dictionaries for training and testing
    static_dict = base_static_dict.copy()
    static_dict["return_val"] = "reserves_and_values"
    static_dict["bout_length"] = data_dict["bout_length"]

    training_data_kind = static_dict["training_data_kind"]
    minimum_weight = static_dict.get("minimum_weight")
    n_assets = static_dict["n_assets"]
    return_val = static_dict["return_val"]
    bout_length = static_dict["bout_length"]

    # filter coarse weights using the start and end indices
    coarse_weights = filter_coarse_weights_by_data_indices(coarse_weights, data_dict)
    # take coarse weights and convert to array of fine weights
    initial_weights = coarse_weights["weights"][0]
    # Repeat the last row of coarse weights
    coarse_weights_padded = jnp.vstack(
        [coarse_weights["weights"], coarse_weights["weights"][-1]]
    )
    raw_weight_outputs = jnp.diff(coarse_weights_padded, axis=0)
    actual_starts_cpu, scaled_diffs_cpu, target_weights_cpu = _jax_calc_coarse_weights(
        raw_weight_outputs,
        initial_weights,
        minimum_weight,
        params,
        run_fingerprint["max_memory_days"],
        chunk_period,
        chunk_period,
        1.0,
        False,
    )

    weights = _jax_fine_weights_from_actual_starts_and_diffs(
        actual_starts_cpu,
        scaled_diffs_cpu,
        initial_weights,
        interpol_num=chunk_period + 1,
        num=chunk_period + 1,
        maximum_change=1.0,
        method="linear",
    )
    # undo padding
    weights = weights[: (-1 * chunk_period + 1)]

    # Check that weights[::chunk_period] matches coarse_weights["weights"]
    # Get weights at coarse timesteps
    coarse_timestep_weights = weights[::chunk_period]
    weights = weights[:-1]

    # Compare with original coarse weights
    weights_match = jnp.allclose(
        coarse_timestep_weights, coarse_weights["weights"], rtol=1e-10
    )

    start_index = data_dict["start_idx"]
    end_index = data_dict["end_idx"] - 1

    local_prices = data_dict["prices"][start_index:end_index]
    local_unix_values = data_dict["unix_values"][start_index:end_index]

    # reserves = pool.calculate_reserves_with_fees(
    #     params,
    #     NestedHashabledict(static_dict),
    #     data_dict["prices"],
    #     start_index=None,
    #     local_prices=HashableArrayWrapper(local_prices),
    #     weights=HashableArrayWrapper(weights),
    #     initial_reserves=HashableArrayWrapper(params["initial_reserves"]),
    # )
    fees_array = dynamic_inputs_dict.get("fees_array")
    arb_thresh_array = dynamic_inputs_dict.get("gas_cost_array")
    arb_fees_array = dynamic_inputs_dict.get("arb_fees_array")
    trade_array = dynamic_inputs_dict.get("trades")
    lp_supply_array = dynamic_inputs_dict.get("lp_supply_array")

    if fees_array is None:
        fees_array = jnp.array([static_dict["fees"]])
    if arb_thresh_array is None:
        arb_thresh_array = jnp.array([static_dict["gas_cost"]])
    if arb_fees_array is None:
        arb_fees_array = jnp.array([static_dict["arb_fees"]])

        # initial_pool_value = run_fingerprint["initial_pool_value"]
        # initial_value_per_token = arb_acted_upon_weights[0] * initial_pool_value
        # initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]

    initial_reserves = params["initial_reserves"]

    # any of fees_array, arb_thresh_array, arb_fees_array, trade_array, and lp_supply_array
    # can be singletons, in which case we repeat them for the length of the bout.

    # Determine the maximum leading dimension
    max_len = bout_length - 1

    if run_fingerprint["arb_frequency"] != 1:
        max_len = max_len // run_fingerprint["arb_frequency"]

    fees_array = fees_array[:max_len]
    arb_thresh_array = arb_thresh_array[:max_len]
    arb_thresh_array = arb_thresh_array * 0.0
    arb_fees_array = arb_fees_array[:max_len]
    if lp_supply_array is not None:
        lp_supply_array = lp_supply_array[:max_len]
    if trade_array is not None:
        trade_array = trade_array[:max_len]
    # Broadcast input arrays to match the maximum leading dimension.
    # If they are singletons, this will just repeat them for the length of the bout.
    # If they are arrays of length bout_length, this will cause no change.
    fees_array_broadcast = jnp.broadcast_to(
        fees_array, (max_len,) + fees_array.shape[1:]
    )
    arb_thresh_array_broadcast = jnp.broadcast_to(
        arb_thresh_array, (max_len,) + arb_thresh_array.shape[1:]
    )
    arb_fees_array_broadcast = jnp.broadcast_to(
        arb_fees_array, (max_len,) + arb_fees_array.shape[1:]
    )
    # if lp_supply_array is not provided, we set it to a constant of 1.0
    if lp_supply_array is None:
        lp_supply_array = jnp.array(1.0)

    lp_supply_array_broadcast = jnp.broadcast_to(
        lp_supply_array, (max_len,) + lp_supply_array.shape[1:]
    )
    # if we are doing trades, the trades array must be of the same length as the other arrays
    if run_fingerprint["do_trades"]:
        assert trade_array.shape[0] == max_len
    reserves = _jax_calc_quantAMM_reserves_with_dynamic_inputs(
        initial_reserves,
        weights,
        local_prices,
        fees_array_broadcast,
        arb_thresh_array_broadcast,
        arb_fees_array_broadcast,
        jnp.array(static_dict["all_sig_variations"]),
        None,
        run_fingerprint["do_trades"],
        run_fingerprint["do_arb"],
        run_fingerprint["noise_trader_ratio"],
        lp_supply_array_broadcast,
    )

    value_over_time = jnp.sum(jnp.multiply(reserves, local_prices), axis=-1)
    return_dict = {
        "final_reserves": reserves[-1],
        "final_value": (reserves[-1] * local_prices[-1]).sum(),
        "value": value_over_time,
        "prices": local_prices,
        "reserves": reserves,
        "weights": weights,
        "raw_weight_outputs": raw_weight_outputs,
        "data_dict": data_dict,
        "unix_values": local_unix_values,
    }
    return return_dict
