# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
from jax.lib.xla_bridge import default_backend
from jax import local_device_count, devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import jit, vmap
from jax import devices, device_put
from jax import tree_util
from jax.lax import stop_gradient, dynamic_slice
from jax.nn import softmax

from quantammsim.pools.G3M.quantamm.momentum_pool import (
    MomentumPool,
    _jax_momentum_weight_update,
)
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
    calc_alt_lamb,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_gradients,
    calc_k,
    calc_alt_ewma_padded,
    calc_ewma_padded,
    calc_ewma_pair,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_ewma_at_infinity_via_scan,
)

from typing import Dict, Any, Optional
from functools import partial
import numpy as np

# import the fine weight output function which has pre-set argument raw_weight_outputs_are_themselves_weights
# as this is False for momentum pools --- the strategy outputs weight _changes_
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output_from_weight_changes,
)

class DifferenceMomentumPool(MomentumPool):
    """
    A class for difference-momentum strategies run as TFMM liquidity pools.

    This class implements a moving average convergence divergence (MACD) strategy for asset allocation within a TFMM framework.
    It uses price data to generate MACD signals, which are then translated into weight adjustments.

    Parameters
    ----------
    None

    Methods
    -------
    calculate_raw_weights_outputs(params, run_fingerprint, prices, additional_oracle_input)
        Calculate the raw weight outputs based on mean-reversion signals.

    Notes
    -----
    The class implements a mean-reversion-based strategy for asset allocation within a TFMM framework.
    It uses price data to generate mean-reversion signals, which are then translated into weight adjustments.
    The class provides methods to calculate raw weight outputs based on these signals and refine them
    into final asset weights, taking into account various parameters and constraints defined in the pool setup.
    """

    def __init__(self):
        """
        Initialize a new AntiMomentumPool instance.

        Parameters
        ----------
        None
        """
        super().__init__()

    @partial(jit, static_argnums=(2))
    def calculate_raw_weights_outputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Calculate raw weight outputs based on difference between two EWMAs.

        Parameters
        ----------
        params : Dict[str, Any]
            Either:
            - memory_days_1: float (memory period for first EWMA)
            - memory_days_2: float (memory period for second EWMA)
            - k: float (scaling factor)
            Or:
            - logit_lamb: array (for base lambda calculation)
            - logit_delta_lamb: array (for delta lambda calculation)
            - log_k: array (for k calculation)
        """
        chunkwise_price_values = prices[:: run_fingerprint["chunk_period"]]

        # Determine which parameterization is being used
        if "memory_days_1" in params and "memory_days_2" in params:
            # New parameterization
            memory_days_1 = params["memory_days_1"]
            memory_days_2 = params["memory_days_2"]
            k = params["k"] * jnp.maximum(memory_days_1, memory_days_2)
        else:
            # Original parameterization
            lamb = calc_lamb(params)
            alt_lamb = calc_alt_lamb(params)

            # Convert lambdas to memory days
            memory_days_1 = lamb_to_memory_days_clipped(
                lamb,
                run_fingerprint["chunk_period"],
                run_fingerprint["max_memory_days"],
            )
            memory_days_2 = lamb_to_memory_days_clipped(
                alt_lamb,
                run_fingerprint["chunk_period"],
                run_fingerprint["max_memory_days"],
            )
            # Original k calculation
            k = calc_k(params, memory_days_1)  # Uses original memory days for scaling

        # Calculate EWMAs
        ewma_1, ewma_2 = calc_ewma_pair(
            memory_days_1,
            memory_days_2,
            chunkwise_price_values,
            run_fingerprint["chunk_period"],
            run_fingerprint["max_memory_days"],
        )

        # Calculate signal
        ewma_proportional_difference = 1.0 - ewma_2 / ewma_1
        raw_weight_outputs = _jax_momentum_weight_update(
            ewma_proportional_difference, k
        )
        return raw_weight_outputs
        # def calculate_raw_weights_outputs(
        #     self,
        #     params: Dict[str, Any],
        #     run_fingerprint: Dict[str, Any],
        #     prices: jnp.ndarray,
        #     additional_oracle_input: Optional[jnp.ndarray] = None,
        # ) -> jnp.ndarray:
        # """
        # Calculate the raw weight outputs based on antimomentum signals.
        # This method computes the raw weight adjustments for the antimomentum strategy. It processes
        # the input prices to calculate gradients, which are then used to determine weight updates.

        # Parameters
        # ----------
        # params : Dict[str, Any]
        #     A dictionary of strategy parameters.
        # run_fingerprint : Dict[str, Any]
        #     A dictionary containing run-specific settings.
        # prices : jnp.ndarray
        #     An array of asset prices over time.
        # additional_oracle_input : Optional[jnp.ndarray], optional
        #     Additional input data, if any.

        # Returns
        # -------
        # jnp.ndarray
        #     Raw weight outputs representing the suggested weight adjustments.

        # Notes
        # -----
        # The method performs the following steps:
        # 1. Calculates the memory days based on the lambda parameter.
        # 2. Computes the 'k' factor which scales the weight updates.
        # 3. Extracts chunkwise price values from the input prices.
        # 4. Calculates two difference EWMAs and the proportional difference between them.
        # 5. Applies the momentum weight update formula using the proportional difference as signal to get raw weight outputs.

        # The raw weight outputs are not the final weights, but rather the changes
        # to be applied to the previous weights. These will be refined in subsequent steps.
        # """

        # chunkwise_price_values = prices[:: run_fingerprint["chunk_period"]]
        # lamb = calc_lamb(params)
        # cap_lamb = True
        # memory_days = lamb_to_memory_days_clipped(
        #     lamb, run_fingerprint["chunk_period"], run_fingerprint["max_memory_days"]
        # )
        # k = calc_k(params, memory_days)
        # if params.get("long_memory_days") is not None:
        #     long_memory_days = params["long_memory_days"]
        # else:
        #     long_memory_days = memory_days
        # if DEFAULT_BACKEND != "cpu":
        #     alt_ewma_padded = calc_alt_ewma_padded(
        #         params,
        #         chunkwise_price_values,
        #     run_fingerprint["chunk_period"],
        #     run_fingerprint["max_memory_days"],
        #     cap_lamb=cap_lamb,
        #     )
        #     alt_ewma = alt_ewma_padded[-(len(chunkwise_price_values) - 1) :]
        #     ewma_padded = calc_ewma_padded(
        #         params,
        #         chunkwise_price_values,
        #         run_fingerprint["chunk_period"],
        #         run_fingerprint["max_memory_days"],
        #         cap_lamb=cap_lamb,
        #     )
        #     ewma = ewma_padded[-(len(chunkwise_price_values) - 1) :]
        # else:
        #     alt_lamb = calc_alt_lamb(params)
        #     if cap_lamb:
        #         max_lamb = memory_days_to_lamb(
        #             run_fingerprint["max_memory_days"], run_fingerprint["chunk_period"]
        #         )
        #         capped_alt_lamb = jnp.clip(alt_lamb, a_min=0.0, a_max=max_lamb)
        #         alt_lamb = capped_alt_lamb
        #         capped_lamb = jnp.clip(lamb, a_min=0.0, a_max=max_lamb)
        #         lamb = capped_lamb
        #     alt_ewma = _jax_ewma_at_infinity_via_scan(chunkwise_price_values, alt_lamb)
        #     ewma = _jax_ewma_at_infinity_via_scan(chunkwise_price_values, lamb)
        # ewma_proportional_difference = 1.0 - alt_ewma / ewma
        # raw_weight_outputs = _jax_momentum_weight_update(
        #     ewma_proportional_difference, k
        # )
        # return raw_weight_outputs

    def _init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        """
        Initialize parameters for the momentum pool.

        This method sets up the initial parameters for the momentum pool strategy, including
        weights, memory length (lambda), and the momentum factor (k).

        Parameters
        ----------
        initial_values_dict : Dict[str, Any]
            Dictionary containing initial values for various parameters.
        run_fingerprint : Dict[str, Any]
            Dictionary containing run-specific settings and parameters.
        n_assets : int
            The number of assets in the pool.
        n_parameter_sets : int, optional
            The number of parameter sets to initialize, by default 1.
        noise : str, optional
            The type of noise to apply during initialization, by default "gaussian".

        Returns
        -------
        Dict[str, jnp.array]
            Dictionary containing the initialized parameters for the momentum pool.

        Raises
        ------
        ValueError
            If required initial values are missing or in an incorrect format.

        Notes
        -----
        This method handles the initialization of parameters for initial weights, lambda
        (memory length parameter), and k (momentum factor) for each asset and parameter set.
        It processes the initial values to ensure they are in the correct format and applies
        any necessary transformations (e.g., logit transformations for lambda).
        """
        np.random.seed(0)

        # We need to initialise the weights for each parameter set
        # If a vector is provided in the inital values dict, we use
        # that, if only a singleton array is provided we expand it
        # to n_assets and use that vlaue for all assets.
        def process_initial_values(
            initial_values_dict, key, n_assets, n_parameter_sets
        ):
            if key in initial_values_dict:
                initial_value = initial_values_dict[key]
                if isinstance(initial_value, (np.ndarray, jnp.ndarray, list)):
                    initial_value = np.array(initial_value)
                    if initial_value.size == n_assets:
                        return np.array([initial_value] * n_parameter_sets)
                    elif initial_value.size == 1:
                        return np.array([[initial_value] * n_assets] * n_parameter_sets)
                    elif initial_value.shape == (n_parameter_sets, n_assets):
                        return initial_value
                    else:
                        raise ValueError(
                            f"{key} must be a singleton or a vector of length n_assets or a matrix of shape (n_parameter_sets, n_assets)"
                        )
                else:
                    return np.array([[initial_value] * n_assets] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        initial_weights_logits = process_initial_values(
            initial_values_dict, "initial_weights_logits", n_assets, n_parameter_sets
        )
        k = process_initial_values(
                initial_values_dict, "initial_k_per_day", n_assets, n_parameter_sets
            )

        memory_days_1 = process_initial_values(
            initial_values_dict, "initial_memory_length", n_assets, n_parameter_sets
        )

        initial_values_dict["initial_memory_length_alt"] = np.array(
            memory_days_1
        ) + np.array(initial_values_dict["initial_memory_length_delta"])

        memory_days_2 = process_initial_values(
            initial_values_dict, "initial_memory_length_alt", n_assets, n_parameter_sets
        )

        params = {
            "k": k,
            "memory_days_1": memory_days_1,
            "memory_days_2": memory_days_2,
            "initial_weights_logits": initial_weights_logits,
            "subsidary_params": [],
        }

        params = self.add_noise(params, noise, n_parameter_sets)
        return params


tree_util.register_pytree_node(
    DifferenceMomentumPool, DifferenceMomentumPool._tree_flatten, DifferenceMomentumPool._tree_unflatten
)


# def test_backends_match():
#     """Quick test that GPU and CPU backends give same results"""
#     from jax import config
#     from jax.lib.xla_bridge import get_backend
#     import numpy as np
#     from quantammsim.runners.jax_runner_utils import Hashabledict
#     # Save original backend
#     original_backend = get_backend().platform

#     # Test data
#     run_fingerprint = {
#         "chunk_period": 1,
#         "max_memory_days": 30
#     }
#     params = {
#         "logit_lamb": jnp.array([0.0, 0.1]),
#         "logit_delta_lamb": jnp.array([0.5, 0.2]),
#         "log_k": jnp.array([1.0, 1.0])
#     }
#     prices = jnp.array([[100.0, 200.0], [101.0, 202.0], [99.0, 198.0]])

#     # Force CPU backend
#     config.update("jax_platform_name", "cpu")
#     pool_cpu = DifferenceMomentumPool()
#     cpu_output = pool_cpu.calculate_raw_weights_outputs(
#         params, Hashabledict(run_fingerprint), prices
#     )
#     print(cpu_output)
#     # Try GPU if available
#     try:
#         config.update("jax_platform_name", "gpu")
#         pool_gpu = DifferenceMomentumPool()
#         gpu_output = pool_gpu.calculate_raw_weights_outputs(
#             params, Hashabledict(run_fingerprint), prices
#         )
#         print(gpu_output)
#         # Compare outputs
#         assert np.allclose(cpu_output, gpu_output, rtol=1e-5), "CPU and GPU outputs differ!"
#         print("CPU and GPU backends produce matching outputs")
#     except:
#         print("GPU backend not available for testing")

#     # Restore original backend
#     config.update("jax_platform_name", original_backend)

# if __name__ == "__main__":
#     test_backends_match()
