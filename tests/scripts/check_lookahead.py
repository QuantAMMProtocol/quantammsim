import jax.numpy as jnp
from jax import random
import numpy as np
import debug
# Import the necessary functions
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import calc_gradients
from quantammsim.core_simulator.param_utils import memory_days_to_lamb

def test_no_lookahead_bias():
    # Set random seed for reproducibility
    key = random.PRNGKey(0)

    # Test parameters
    n_assets = 2
    n_timesteps = 1000
    chunk_period = 60
    max_memory_days = 30.0
    use_alt_lamb = False

    # Generate random price data
    prices = jnp.ones((n_timesteps, n_assets))

    # Create parameter dict with a reasonable lambda value
    update_rule_parameter_dict = {
        "logit_lamb": jnp.array(0.0),  # This gives lambda ≈ 0.5
    }

    # Calculate gradients for full dataset
    full_gradients = calc_gradients(
        update_rule_parameter_dict,
        prices,
        chunk_period,
        max_memory_days,
        use_alt_lamb
    )
    # Test at different cutoff points
    for cutoff in [int(n_timesteps * 0.25), int(n_timesteps * 0.5), int(n_timesteps * 0.75)]:
        # Calculate gradients with truncated future data
        truncated_prices = jnp.concatenate([
            prices[:cutoff],
            jnp.ones((n_timesteps - cutoff, n_assets)) * 1000.0  # Dramatically different future prices
        ])

        truncated_gradients = calc_gradients(
            update_rule_parameter_dict,
            truncated_prices,
            chunk_period,
            max_memory_days,
            use_alt_lamb,
        )
        # Check if gradients are identical up to the cutoff point
        # Allow for small numerical differences
        max_diff = jnp.max(jnp.abs(full_gradients[:cutoff-1] - truncated_gradients[:cutoff-1]))

        print(f"Testing cutoff at t={cutoff}")
        print(f"Maximum difference in gradients before cutoff: {max_diff}")

        # Assert that differences are negligible
        assert max_diff < 1e-10, f"Look-ahead bias detected! Max difference: {max_diff}"

        # Optional: verify that gradients after cutoff are different
        post_cutoff_diff = jnp.max(jnp.abs(full_gradients[cutoff:] - truncated_gradients[cutoff:]))

        print(f"Maximum difference in gradients after cutoff: {post_cutoff_diff}")
        print("---")
        print("prices near cutoff:")
        print(truncated_prices[cutoff - 2 : cutoff + 2])
        print("gradients near cutoff:")
        print(full_gradients[cutoff-2:cutoff+2])
        print(truncated_gradients[cutoff-2:cutoff+2])
        print("gradient size:")
        print(full_gradients.shape)
        print(truncated_gradients.shape)
        print("prices size:")
        print(prices.shape)
        print("truncated prices size:")
        print(truncated_prices.shape)
    raise Exception("Stop here")
        

if __name__ == "__main__":
    test_no_lookahead_bias()
    print("All tests passed! No look-ahead bias detected.")

