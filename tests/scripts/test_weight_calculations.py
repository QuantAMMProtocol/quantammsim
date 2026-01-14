from jax import config
config.update("jax_enable_x64", True)
config.update("jax_disable_jit", True)
import jax.numpy as jnp
from jax import random
import numpy as np
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output,
)
calc_fine_weight_output_old = calc_fine_weight_output
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights_new import (
    calc_fine_weight_output as calc_fine_weight_output_new,
)
from quantammsim.runners.jax_runner_utils import Hashabledict
from jax.nn import softmax
# Set random seed for reproducibility
key = random.PRNGKey(0)
np.random.seed(0)
# Test parameters
test_cases = [
    {
        "n_assets": 3,
        "n_timesteps": 10,
        "mvpt": True,
        "name": "Small portfolio with MVPT"
    },
    {
        "n_assets": 5,
        "n_timesteps": 20,
        "mvpt": False,
        "name": "Medium portfolio without MVPT"
    },
    {
        "n_assets": 10,
        "n_timesteps": 15,
        "mvpt": True,
        "name": "Large portfolio with MVPT"
    }
]

for case in test_cases:
    print(f"\nTesting: {case['name']}")

    n_assets = case["n_assets"]
    n_timesteps = case["n_timesteps"]
    mvpt = True

    # Generate test data
    key, subkey = random.split(key)
    raw_weight_outputs = softmax(random.normal(
        subkey, 
        shape=(n_timesteps, n_assets),
        dtype=jnp.float64
    ),axis=1)
    if mvpt is False:
        raw_weight_outputs = raw_weight_outputs - jnp.mean(raw_weight_outputs,axis=0)
    # Initial weights that sum to 1
    initial_weights = jnp.ones(n_assets, dtype=jnp.float64) / n_assets

    # Run parameters
    run_fingerprint = {
        "weight_interpolation_period": 1,
        "chunk_period": 1,
        # "maximum_change": float(jnp.finfo(jnp.float64).max),  # Vacuous maximum_change
        "maximum_change": 0.01,  # Vacuous maximum_change
        "weight_interpolation_method": "linear",

        "n_assets": n_assets,
        "minimum_weight": 0.1 / n_assets,
    }

    # Update rule parameters
    params = {
        "logit_lamb": jnp.array(np.random.randn(n_assets)),
        "logit_delta_lamb": jnp.array(np.random.randn(n_assets)),
        "log_k": jnp.array(np.random.randn(n_assets))
    }

    # Calculate weights using both methods
    weights_new = calc_fine_weight_output_new(
        raw_weight_outputs,
        initial_weights,
        Hashabledict(run_fingerprint),
        params,
        mvpt,
    )
    weights_new = jnp.vstack([jnp.ones((run_fingerprint["chunk_period"],n_assets),dtype=jnp.float64)*initial_weights, weights_new])

    weights_old = calc_fine_weight_output_old(
        raw_weight_outputs,
        initial_weights,
        Hashabledict(run_fingerprint),
        params,
        mvpt,
    )

    # Compare results
    are_equal = jnp.allclose(weights_new, weights_old, rtol=1e-10, atol=1e-10)
    print(f"Outputs match: {are_equal}")

    if not are_equal:
        max_diff = jnp.max(jnp.abs(weights_new - weights_old))
        print(f"Maximum difference: {max_diff}")

        # Print first mismatch
        mismatch_idx = jnp.where(~jnp.allclose(
            weights_new, 
            weights_old, 
            rtol=1e-10, 
            atol=1e-10, 
            equal_nan=True
        ))
        if len(mismatch_idx[0]) > 0:
            i, j = mismatch_idx[0][0], mismatch_idx[1][0]
            print(f"\nFirst mismatch at position ({i}, {j}):")
            print(f"New value: {weights_new[i, j]}")
            print(f"Old value: {weights_old[i, j]}")

            # Print full weights at mismatch position
            print(f"\nFull weights at position {i}:")
            print(f"New weights: {weights_new[i]}")
            print(f"Old weights: {weights_old[i]}")
            print(f"Sum new: {jnp.sum(weights_new[i])}")
            print(f"Sum old: {jnp.sum(weights_old[i])}")

            # Check if weights sum to 1 throughout
            sum_diffs = jnp.abs(jnp.sum(weights_new, axis=1) - 1.0)
            max_sum_diff = jnp.max(sum_diffs)
            print(f"\nMaximum deviation from sum=1 (new): {max_sum_diff}")

            sum_diffs_old = jnp.abs(jnp.sum(weights_old, axis=1) - 1.0)
            max_sum_diff_old = jnp.max(sum_diffs_old)
            print(f"Maximum deviation from sum=1 (old): {max_sum_diff_old}")
