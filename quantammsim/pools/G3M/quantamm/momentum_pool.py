# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
from jax import default_backend
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

from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_gradients,
    calc_k,
)

from typing import Dict, Any, Optional
from functools import partial
from abc import abstractmethod
import numpy as np

# import the fine weight output function which has pre-set argument raw_weight_outputs_are_themselves_weights
# as this is False for momentum pools --- the strategy outputs weight _changes_
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output_from_weight_changes,
)


@jit
def _jax_momentum_weight_update(price_gradient, k):
    """
    Calculate momentum-based weight updates for assets.

    This function computes weight updates for a set of assets based on their price gradients
    and a momentum factor k. It ensures that the sum of weight updates across all assets is zero,
    maintaining the total portfolio weight.

    Parameters
    ----------
    price_gradient : jnp.ndarray
        Array of price gradients for each asset.
    k : float or jnp.ndarray
        Momentum factor or array of momentum factors for each asset.

    Returns
    -------
    jnp.ndarray
        Array of weight updates for each asset.

    Notes
    -----
    The function performs the following steps:
    1. Calculates an offset to ensure the sum of weight updates is zero.
    2. Computes weight updates using the price gradients, k, and the offset.
    3. Sets weight updates to zero for any asset where k is zero.

    The function assumes that inputs are JAX arrays for efficient computation.
    """
    offset_constants = -(k * price_gradient).sum(axis=-1, keepdims=True) / (jnp.sum(k))
    weight_updates = k * (price_gradient + offset_constants)
    weight_updates = jnp.where(k == 0.0, 0.0, weight_updates)
    return weight_updates


class MomentumPool(TFMMBasePool):
    """
    A class for momentum strategies run as TFMM (Temporal Function Market Making) liquidity pools,
    extending the TFMMBasePool class.

    This class implements a momentum-based strategy for asset allocation within a TFMM framework.
    It uses price data to generate momentum signals, which are then translated into weight adjustments.

    Parameters
    ----------
    None

    Methods
    -------
    calculate_raw_weights_outputs(params, run_fingerprint, prices, additional_oracle_input)
        Calculate the raw weight outputs based on momentum signals.
    fine_weight_output(raw_weight_output, initial_weights, run_fingerprint, params)
        Refine the raw weight outputs to produce final weights.
    calculate_weights(params, run_fingerprint, prices, additional_oracle_input)
        Orchestrate the weight calculation process.

    Notes
    -----
    The class provides methods to calculate raw weight outputs based on momentum signals and refine them
    into final asset weights, taking into account various parameters and constraints defined in the pool setup.
    """

    def __init__(self):
        """
        Initialize a new MomentumPool instance.

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
        """
        Calculate the raw weight outputs based on momentum signals.

        This method computes the raw weight adjustments for the momentum strategy. It processes
        the input prices to calculate gradients, which are then used to determine weight updates.

        Parameters
        ----------
        params : Dict[str, Any]
            A dictionary of strategy parameters.
        run_fingerprint : Dict[str, Any]
            A dictionary containing run-specific settings.
        prices : jnp.ndarray
            An array of asset prices over time.
        additional_oracle_input : Optional[jnp.ndarray], optional
            Additional input data, if any.

        Returns
        -------
        jnp.ndarray
            Raw weight outputs representing the suggested weight adjustments.

        Notes
        -----
        The method performs the following steps:
        1. Calculates the memory days based on the lambda parameter.
        2. Computes the 'k' factor which scales the weight updates.
        3. Extracts chunkwise price values from the input prices.
        4. Calculates price gradients using the calc_gradients function.
        5. Applies the momentum weight update formula to get raw weight outputs.

        The raw weight outputs are not the final weights, but rather the changes
        to be applied to the previous weights. These will be refined in subsequent steps.
        """
        memory_days = lamb_to_memory_days_clipped(
            calc_lamb(params),
            run_fingerprint["chunk_period"],
            run_fingerprint["max_memory_days"],
        )
        k = calc_k(params, memory_days)
        chunkwise_price_values = prices[:: run_fingerprint["chunk_period"]]
        gradients = calc_gradients(
            params,
            chunkwise_price_values,
            run_fingerprint["chunk_period"],
            run_fingerprint["max_memory_days"],
            run_fingerprint["use_alt_lamb"],
            cap_lamb=True,
        )
        raw_weight_outputs = _jax_momentum_weight_update(gradients, k)
        return raw_weight_outputs

    @partial(jit, static_argnums=(3))
    def fine_weight_output(
        self,
        raw_weight_output: jnp.ndarray,
        initial_weights: jnp.ndarray,
        run_fingerprint: Dict[str, Any],
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Refine raw weight outputs to produce final weights for the momentum pool.

        This method takes the raw weight outputs calculated from momentum signals and refines
        them into final asset weights. It applies various constraints and adjustments defined
        in the pool parameters and run fingerprint.

        Parameters
        ----------
        raw_weight_output : jnp.ndarray
            Raw weight changes or outputs from momentum calculations.
        initial_weights : jnp.ndarray
            Initial weights of assets in the pool.
        run_fingerprint : Dict[str, Any]
            Dictionary containing run-specific parameters and settings.
        params : Dict[str, Any]
            Pool parameters.

        Returns
        -------
        jnp.ndarray
            Refined weights for each asset in the pool over the specified time period.

        Notes
        -----
        Uses the `calc_fine_weight_output_from_weight_changes` function to perform the actual
        refinement. The implementation of this function should handle details such as weight
        interpolation, maximum change limits, and ensuring weights sum to 1.
        """
        return calc_fine_weight_output_from_weight_changes(
            raw_weight_output, initial_weights, run_fingerprint, params
        )

    def init_base_parameters(
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
        log_k = np.log2(
            process_initial_values(
                initial_values_dict, "initial_k_per_day", n_assets, n_parameter_sets
            )
        )

        initial_lamb = memory_days_to_lamb(
            initial_values_dict["initial_memory_length"],
            run_fingerprint["chunk_period"],
        )

        logit_lamb_np = np.log(initial_lamb / (1.0 - initial_lamb))
        logit_lamb = np.array([[logit_lamb_np] * n_assets] * n_parameter_sets)

        # lamb delta is the difference in lamb needed for
        # lamb + delta lamb to give a final memory length
        # of  initial_memory_length + initial_memory_length_delta
        initial_lamb_plus_delta_lamb = memory_days_to_lamb(
            initial_values_dict["initial_memory_length"]
            + initial_values_dict["initial_memory_length_delta"],
            run_fingerprint["chunk_period"],
        )

        logit_lamb_plus_delta_lamb_np = np.log(
            initial_lamb_plus_delta_lamb / (1.0 - initial_lamb_plus_delta_lamb)
        )
        logit_delta_lamb_np = logit_lamb_plus_delta_lamb_np - logit_lamb_np
        logit_delta_lamb = np.array(
            [[logit_delta_lamb_np] * n_assets] * n_parameter_sets
        )

        params = {
            "log_k": log_k,
            "logit_lamb": logit_lamb,
            "logit_delta_lamb": logit_delta_lamb,
            "initial_weights_logits": initial_weights_logits,
            "subsidary_params": [],
        }

        params = self.add_noise(params, noise, n_parameter_sets)
        return params


tree_util.register_pytree_node(
    MomentumPool, MomentumPool._tree_flatten, MomentumPool._tree_unflatten
)
