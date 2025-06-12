import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
import hashlib
import jax.numpy as jnp

if __name__ == "__main__":
    from quantammsim.runners.jax_runners import (
        train_on_historic_data,
    )

    list_of_run_fingerprints = []
    with open("prod_design_0321_sgd_sonic_.jsonl") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
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
        file = open("log_training_products.txt", "a")
        file.write(str(i) + run_location + "\n")  # Write some text
        file.close()  # Close the file
        params = {
            "initial_weights_logits": jnp.array([0.0, 10.0, 10.0]),
            "k": jnp.array([10.0, 10.0, 10.0]),
            "logit_lamb": jnp.array([10.0, 10.0, 10.0]),
            "logit_delta_lamb": jnp.array([0.0, 0.0, 0.0]),
        }

        train_on_historic_data(run, iterations_per_print=100)

        i = i + 1
