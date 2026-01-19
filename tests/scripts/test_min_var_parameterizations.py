import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.nn import softmax
from quantammsim.pools.G3M.quantamm.min_variance_pool import MinVariancePool
from quantammsim.runners.jax_runner_utils import NestedHashabledict
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    jax_memory_days_to_lamb,
    memory_days_to_logit_lamb,
)

def create_test_prices(n_timesteps=1000, n_assets=2):
    """Create some test price data with some volatility"""
    np.random.seed(0)
    base_prices = np.exp(np.random.randn(n_timesteps, n_assets) * 0.1)
    # Ensure first price is 1.0 for easy comparison
    base_prices = base_prices / base_prices[0]
    return jnp.array(base_prices)

def create_params_logit_lamb(memory_days_1=10.0, memory_days_2=20.0, chunk_period=60):
    """Create params using logit_lamb parameterization"""
    logit_lamb = memory_days_to_logit_lamb(jnp.array([memory_days_1]), chunk_period)
    
    # Calculate logit_delta_lamb to achieve memory_days_2
    lamb_1 = jax_memory_days_to_lamb(memory_days_1, chunk_period)
    lamb_2 = jax_memory_days_to_lamb(memory_days_2, chunk_period)
    logit_lamb_2 = jnp.log(lamb_2 / (1.0 - lamb_2))
    logit_delta_lamb = logit_lamb_2 - logit_lamb
    
    return {
        'logit_lamb': logit_lamb,
        'logit_delta_lamb': logit_delta_lamb,
        'initial_weights_logits': jnp.array([0.0, 0.0])
    }

def create_params_memory_days(memory_days_1=10.0, memory_days_2=20.0):
    """Create params using direct memory_days parameterization"""
    return {
        'memory_days_1': jnp.array([memory_days_1]),
        'memory_days_2': jnp.array([memory_days_2]),
        'initial_weights_logits': jnp.array([0.0, 0.0])
    }

def create_run_fingerprint():
    """Create basic run fingerprint for testing"""
    return {
        'chunk_period': 60,
        'weight_interpolation_period': 60,
        'max_memory_days': 365,
        'minimum_weight': 0.1,
        'use_alt_lamb': True
    }

def plot_comparison(weights1, weights2, title1, title2):
    """Plot two sets of weights for comparison"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    for i in range(weights1.shape[1]):
        ax1.plot(weights1[:, i], label=f'Asset {i+1}')
    ax1.set_title(title1)
    ax1.legend()
    ax1.grid(True)
    
    for i in range(weights2.shape[1]):
        ax2.plot(weights2[:, i], label=f'Asset {i+1}')
    ax2.set_title(title2)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# def main():
# Create test data
prices = create_test_prices()

# Setup run fingerprint
rf = NestedHashabledict(create_run_fingerprint())
rf["bout_length"] = len(prices)
rf["n_assets"] = len(prices[0])
rf["rule"] = "min_variance"
rf["return_val"] = "final_reserves_value_and_weights"
rf["maximum_change"] = 0.0003
rf["arb_fees"] = 0.0
rf["gas_cost"] = 0.0
rf["arb_quality"] = 0.0
rf["do_arb"] = True
rf["weight_interpolation_method"] = "linear"
rf["arb_frequency"] = 1
rf["do_trades"] = False
rf["chunk_period"] = 60
rf["weight_interpolation_period"] = 60
rf["max_memory_days"] = 365
rf["minimum_weight"] = 0.1
rf["use_alt_lamb"] = True

# Create params in both formats
params_logit = create_params_logit_lamb(memory_days_1=10.0, memory_days_2=2.0, chunk_period=rf["chunk_period"])
params_memory = create_params_memory_days(memory_days_1=10.0, memory_days_2=2.0)

# Initialize pool
pool = MinVariancePool()

# Calculate weights using both parameterizations
weights_logit = pool.calculate_weights(params_logit, rf, prices, [0, 0])
weights_memory = pool.calculate_weights(params_memory, rf, prices, [0, 0])

# Calculate raw weights for both
raw_weights_logit = pool.calculate_rule_outputs(params_logit, rf, prices, None)
raw_weights_memory = pool.calculate_rule_outputs(params_memory, rf, prices, None)

# Plot comparisons
plot_comparison(
    raw_weights_logit[:100], 
    raw_weights_memory[:100],
    "Raw Weights (logit_lamb parameterization)",
    "Raw Weights (memory_days parameterization)"
)

plot_comparison(
    weights_logit[100:300], 
    weights_memory[100:300],
    "Fine Weights (logit_lamb parameterization)",
    "Fine Weights (memory_days parameterization)"
)

# Print max difference to verify they're equivalent
print("Maximum difference in raw weights:", 
        jnp.max(jnp.abs(raw_weights_logit - raw_weights_memory)))
print("Maximum difference in fine weights:", 
        jnp.max(jnp.abs(weights_logit - weights_memory)))

# if __name__ == "__main__":
#     main() 
