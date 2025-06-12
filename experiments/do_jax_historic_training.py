import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
import hashlib
import jax.numpy as jnp
if __name__ == "__main__":
    # from jax.config import config

    # config.update("jax_enable_x64", True)
    # config.update("jax_debug_nans", True)
    # config.update('jax_disable_jit', True)
    from quantammsim.runners.jax_runners import (
        train_on_historic_data,
        do_run_on_historic_data,
    )
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
    from quantammsim.core_simulator.param_utils import (
        recursive_default_set,
        check_run_fingerprint,
    )

    # train_on_historic_data({}, iterations_per_print=1)
    # rez=iterative_train_on_historic_data({"bout_offset": 30 * 24 * 60})
    # # fingerprint = {"rule":"difference_channel", "chunk_period": 60, "weight_interpolation_period": 60, "return_val": "returns", "bout_offset": 14400, "startDateString": "2022-02-03 00:00:00", "endDateString": "2022-05-22 00:00:00", "endTestDateString": "2023-07-03 00:00:00", "optimisation_settings": {"base_lr": 1.5, "optimiser": "sgd", "decay_lr_ratio": 0.8, "decay_lr_plateau": 200, "batch_size": 200, "train_on_hessian_trace": False, "min_lr": 1e-06, "n_iterations": 10000, "n_cycles": 5}}
    # # train_on_historic_data(fingerprint,iterations_per_print=1)
    # train_on_historic_data({}, iterations_per_print=1)
    # np.seterr(under="print")
    # print(os.getcwd())
    # print(__file__)

    list_of_run_fingerprints = []
    with open("prod_design_0321_sgd_sonic_.jsonl") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            # Pure JSON file
            list_of_run_fingerprints = json.load(f)
        else:
            # JSONL file
            list_of_run_fingerprints = [json.loads(line) for line in f]
    i = 0
    for run in list_of_run_fingerprints:

        run_location = "run_" + str(
        hashlib.sha256(
            json.dumps(run, sort_keys=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
    )
        file = open('log_training_products.txt', 'a')
        file.write(str(i) + run_location + "\n") # Write some text
        file.close() # Close the file
        params = {
            "initial_weights_logits": jnp.array([0.0, 10.0, 10.0]),
            "k": jnp.array([10.0, 10.0, 10.0]),
            "logit_lamb": jnp.array([10.0, 10.0, 10.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0, 0.0]),
        }
        # if run["rule"] == "difference_momentum":
        #     recursive_default_set(run, run_fingerprint_defaults)
        #     check_run_fingerprint(run)
        #     scalar = run["optimisation_settings"]["optuna_settings"]["make_scalar"]
        #     run["optimisation_settings"]["optuna_settings"]["parameter_config"]["k_per_day"] = {
        #         "low": -1000,
        #         "high": 1000,
        #         "log_scale": False,
        #         "scalar": scalar
        #     }
        #     run["optimisation_settings"]["optuna_settings"]["parameter_config"]["memory_days_1"] = {
        #         "low": 0.1,
        #         "high": 100,
        #         "log_scale": True,
        #         "scalar": scalar
        #     }
        #     run["optimisation_settings"]["optuna_settings"]["parameter_config"]["memory_days_2"] = {
        #         "low": 0.1,
        #         "high": 100,
        #         "log_scale": True,
        #         "scalar": scalar
        #     }
        train_on_historic_data(run, iterations_per_print=100)

        i = i + 1

    # train_on_historic_data(
    #     {
    #         "chunk_period": 60,
    #         "weight_interpolation_period": 60,
    #         "return_val": "sharpe",
    #         "bout_offset": 24 * 60 * 10,
    #         "optimisation_settings": {"batch_size": 4, "train_on_hessian_trace": False},
    #         # "subsidary_pools": [
    #         #     {
    #         #         "update_rule": "momentum",
    #         #         "initial_memory_length": 40,
    #         #         "initial_k": 20,
    #         #         "tokens": ["BTC", "ETH", "DAI"],
    #         #     },
    #         #     {
    #         #         "update_rule": "anti_momentum",
    #         #         "initial_memory_length": 10,
    #         #         "initial_k": 60,
    #         #         "tokens": ["BTC", "ETH", "DAI"],
    #         #     },
    #         # ],
    #     }
    # )
