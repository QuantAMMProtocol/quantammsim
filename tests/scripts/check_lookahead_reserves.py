import jax.numpy as jnp
from jax import random
import numpy as np
from jax import config
config.update("jax_disable_jit", True)
import debug

# Import necessary components
from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool
from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_gradients,
    calc_k,
)
from quantammsim.pools.G3M.quantamm.momentum_pool import _jax_momentum_weight_update
from quantammsim.pools.creator import create_pool

class Hashabledict(dict):
    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


# class SimpleTFMMPool(TFMMBasePool):
#     """Simple TFMM pool implementation for testing"""
#     def calculate_rule_outputs(self, params, run_fingerprint, prices, additional_oracle_input=None):
#         # Return constant weights for testing
#         n_chunks = prices.shape[0] // run_fingerprint["chunk_period"]
#         return jnp.ones((n_chunks, 2)) * 0.5

#     def calculate_fine_weights(self, rule_output, initial_weights, run_fingerprint, params):
#         # Simple interpolation for testing
#         n_fine = run_fingerprint["bout_length"]
#         return jnp.ones((n_fine, 2)) * 0.5

#     def init_base_parameters(
#         self):
#         pass


def test_no_lookahead_bias_tfmm():
    # Set random seed for reproducibility
    key = random.PRNGKey(0)

    # Test parameters
    n_assets = 2
    n_timesteps = 1000

    for rule in ["momentum"]:

        pool = create_pool(rule)

        for chunk_period in [2]:
            max_memory_days = 30.0

            # Generate random price data
            prices = jnp.ones((n_timesteps, n_assets))

            # Create parameter dict
            params = {
                "initial_weights_logits": jnp.zeros(n_assets),
                "logit_lamb": jnp.array([0.0, 0.0]),
                "log_k": jnp.array([10.0, 10.0]),
                "memory_days_1": jnp.array([1.0, 1.0]),
                "memory_days_2": jnp.array([1.0, 1.0]),
            }

            # Create run fingerprint
            run_fingerprint = {
                "bout_length": n_timesteps-2*chunk_period+1,
                "chunk_period": chunk_period,
                "n_assets": n_assets,
                "initial_pool_value": 1000.0,
                "fees": 0.00003,
                "gas_cost": 0.0,
                "arb_fees": 0.0,
                "arb_frequency": 1,
                "all_sig_variations": [[1, -1], [-1, 1]],
                "do_trades": False,
                "max_memory_days": max_memory_days,
                "use_alt_lamb": False,
                "weight_interpolation_period": chunk_period,
                "weight_interpolation_method": "linear",
                "maximum_change": 1.0,
                "minimum_weight": 0.0,
                "do_arb": True,
                "noise_trader_ratio": 0.0,
                "ste_max_change": False,
                "ste_min_max_weight": False,
            }

            run_fingerprint = Hashabledict(run_fingerprint)

            # Test both zero fees and with fees cases
            for fees_case in ["zero_fees", "with_fees"]:
                print(f"\nTesting {fees_case} case:")

                # Calculate reserves for full dataset
                if fees_case == "zero_fees":
                    full_reserves = pool.calculate_reserves_zero_fees(
                        params, run_fingerprint, prices, jnp.array([0,0])
                    )
                else:
                    full_reserves = pool.calculate_reserves_with_fees(
                        params, run_fingerprint, prices, jnp.array([0,0])
                    )

                # Test at different cutoff points
                for cutoff in [int(n_timesteps * 0.25), int(n_timesteps * 0.5), int(n_timesteps * 0.75)]:
                    # Calculate reserves with truncated future data
                    truncated_prices = jnp.concatenate([
                        prices[:cutoff],
                        jnp.ones((n_timesteps - cutoff, n_assets)).at[:,0].set(1000.0)  # Dramatically different future price for first asset only
                    ])

                    if fees_case == "zero_fees":
                        truncated_reserves = pool.calculate_reserves_zero_fees(
                            params, run_fingerprint, truncated_prices, jnp.array([0,0])
                        )
                    else:
                        truncated_reserves = pool.calculate_reserves_with_fees(
                            params, run_fingerprint, truncated_prices, jnp.array([0,0])
                        )

                    # Check if reserves are identical up to the cutoff point
                    max_diff = jnp.max(jnp.abs(full_reserves[:cutoff-1] - truncated_reserves[:cutoff-1]))

                    print(f"Testing cutoff at t={cutoff}")
                    print(f"Maximum difference in reserves before cutoff: {max_diff}")

                    # Verify that reserves after cutoff are different
                    post_cutoff_diff = jnp.max(jnp.abs(full_reserves[cutoff:] - truncated_reserves[cutoff:]))
                    print(f"Maximum difference in reserves after cutoff: {post_cutoff_diff}")
                    print("---")
                    # Plot prices and weights around the cutoff point
                    import matplotlib.pyplot as plt

                    # Define window size around cutoff to plot
                    window = 100
                    start_idx = max(0, cutoff - window)
                    end_idx = min(cutoff + window, len(prices))

                    # Create time axis for x-coordinates
                    time_window = np.arange(start_idx, end_idx)

                    # Create subplots for prices and weights
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                    # Plot prices
                    for i in range(n_assets):
                        ax1.plot(time_window, truncated_prices[start_idx:end_idx, i], 
                                label=f'Asset {i+1}')
                        ax1.axvline(x=cutoff, color='r', linestyle='--', alpha=0.5)

                    ax1.set_ylabel('Prices')
                    ax1.legend()
                    ax1.grid(True)
                    ax1.set_title(f'Prices Around Cutoff Point (t={cutoff})')

                    # Plot weights
                    weights = pool.calculate_weights(
                        params, run_fingerprint, truncated_prices, jnp.array([0,0])
                    )
                    print("len(weights)", len(weights))
                    rule_outputs_ = pool.calculate_rule_outputs(
                        params, run_fingerprint, truncated_prices)

                    memory_days = lamb_to_memory_days_clipped(
                    calc_lamb(params),
                        run_fingerprint["chunk_period"],
                        run_fingerprint["max_memory_days"],
                    )
                    k = calc_k(params, memory_days)
                    chunkwise_price_values = truncated_prices[:: run_fingerprint["chunk_period"]]
                    gradients = calc_gradients(
                        params,
                        chunkwise_price_values,
                        run_fingerprint["chunk_period"],
                        run_fingerprint["max_memory_days"],
                        run_fingerprint["use_alt_lamb"],
                        cap_lamb=True,
                    )
                    rule_outputs = _jax_momentum_weight_update(gradients, k)
                    if cutoff == 500:
                        raise Exception("Stop here")

                    for i in range(n_assets):
                        ax2.plot(time_window, weights[start_idx:end_idx, i],
                                label=f'Asset {i+1}')
                        ax2.axvline(x=cutoff, color='r', linestyle='--', alpha=0.5)

                    ax2.set_xlabel('Timestep')
                    ax2.set_ylabel('Weights')
                    ax2.legend()
                    ax2.grid(True)
                    ax2.set_title('Asset Weights Around Cutoff Point')

                    plt.tight_layout()
                    plt.savefig(f'prices_weights_cutoff_{cutoff}_{fees_case}_{chunk_period}_{rule}.png')
                    plt.close()
                    # Plot prices and reserves around the cutoff point
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                    # Plot prices
                    for i in range(n_assets):
                        ax1.plot(
                            time_window,
                            truncated_prices[start_idx:end_idx, i],
                            label=f"Asset {i+1}",
                        )
                        ax1.axvline(x=cutoff, color='r', linestyle='--', alpha=0.5)

                    ax1.set_ylabel('Prices')
                    ax1.legend()
                    ax1.grid(True)
                    ax1.set_title(f'Prices Around Cutoff Point (t={cutoff})')

                    # Plot reserves
                    for i in range(n_assets):
                        ax2.plot(time_window, truncated_reserves[start_idx:end_idx, i],
                                label=f'Asset {i+1}')
                        ax2.axvline(x=cutoff, color='r', linestyle='--', alpha=0.5)

                    ax2.set_xlabel('Timestep')
                    ax2.set_ylabel('Reserves')
                    ax2.legend()
                    ax2.grid(True)
                    ax2.set_title('Asset Reserves Around Cutoff Point')

                    plt.tight_layout()
                    plt.savefig(f'prices_reserves_cutoff_{cutoff}_{fees_case}_{chunk_period}_{rule}.png')
                    plt.close()
                    # Assert that differences are negligible
                    assert (
                        max_diff < 1e-10
                    ), f"Look-ahead bias detected! Max difference: {max_diff}"


if __name__ == "__main__":
    test_no_lookahead_bias_tfmm()
    print("All tests passed! No look-ahead bias detected in TFMM pools.")
