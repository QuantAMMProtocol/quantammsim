import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
import hashlib
import jax.numpy as jnp
import debug
if __name__ == "__main__":
    from quantammsim.runners.jax_runners import (
        train_on_historic_data,
    )

    list_of_run_fingerprints = []
    list_of_run_locations = []

    run_location_directory = './sonic_release_candidate_run_results/'
    for file in os.listdir(run_location_directory):
        if file.endswith(".json"):
            list_of_run_locations.append(run_location_directory + file)

    for run_location in list_of_run_locations:
        # load the run fingerprint
        with open(run_location, "r") as f:
            run_fingerprint = json.loads(json.load(f))[0]
            list_of_run_fingerprints.append(run_fingerprint)
    i = 0
    for base_lr in [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
        for run,run_location in zip(list_of_run_fingerprints, list_of_run_locations):
            file = open("log_training_products.txt", "a")
            file.write(str(i) + run_location + "\n")  # Write some text
            file.close()  # Close the file
            run["optimisation_settings"]["optimiser"] = "adam"
            run["optimisation_settings"]["base_lr"] = base_lr
            train_on_historic_data(
                run,
                iterations_per_print=100,
                run_location=run_location,
            )

            i = i + 1
