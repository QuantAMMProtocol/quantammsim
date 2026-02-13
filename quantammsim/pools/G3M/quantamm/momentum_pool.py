"""Trend-following (momentum) pool for QuantAMM.

Implements an EWMA-based momentum strategy that computes exponentially weighted
price gradients and converts them into zero-sum weight changes via a learnable
sensitivity factor ``k``. Overweights assets with positive recent price trends
and underweights those with negative trends.

Key parameters: ``log_k`` (momentum sensitivity), ``logit_lamb`` (EWMA decay /
memory length), ``logit_delta_lamb`` (alternative memory offset).
"""
# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
from jax import default_backend
from jax import devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import jit
from jax import devices
from jax import tree_util
from jax.lax import dynamic_slice

from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_gradients,
    calc_gradients_with_readout,
    calc_k,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_gradient_scan_function,
)
from quantammsim.core_simulator.param_utils import jax_memory_days_to_lamb
from quantammsim.core_simulator.param_schema import (
    ParamSpec,
    OptunaRange,
    COMMON_PARAM_SCHEMA,
)

from typing import Dict, Any, Optional
from functools import partial
import numpy as np

# import the fine weight output function which has pre-set argument rule_outputs_are_themselves_weights
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
    calculate_rule_outputs(params, run_fingerprint, prices, additional_oracle_input)
        Calculate the raw weight outputs based on momentum signals.
    calculate_fine_weights(rule_output, initial_weights, run_fingerprint, params)
        Refine the raw weight outputs to produce final weights.
    calculate_weights(params, run_fingerprint, prices, additional_oracle_input)
        Orchestrate the weight calculation process.

    Notes
    -----
    The class provides methods to calculate raw weight outputs based on momentum signals and refine them
    into final asset weights, taking into account various parameters and constraints defined in the pool setup.
    """

    # Pool-owned parameter schema: defines all parameters this pool uses,
    # their defaults, and Optuna search ranges.
    #
    # IMPORTANT: Ranges are defined in INTERNAL param space (after transforms).
    # This ensures ensemble sampling produces correctly-transformed values.
    #
    # Internal param mappings:
    #   log_k = log2(k_per_day)  -> k_per_day=0.01 gives log_k≈-6.6, k_per_day=4096 gives log_k=12
    #   logit_lamb = logit(lamb) -> depends on chunk_period, but roughly:
    #     memory_length=0.5 days (very fast) -> low logit
    #     memory_length=365 days (slow) -> high logit
    PARAM_SCHEMA = {
        # log_k: log2(k_per_day)
        # k_per_day in [0.01, 4096] -> log_k in [-6.6, 12]
        "log_k": ParamSpec(
            initial=4.32,  # log2(20) ≈ 4.32
            optuna=OptunaRange(low=-6.6, high=12.0, log_scale=False, scalar=False),
            description="Log2 of momentum sensitivity factor (k) per day",
        ),
        # logit_lamb: logit of decay parameter lambda
        # Wide range to accommodate memory_length from 0.5 to 365 days
        # The exact mapping depends on chunk_period
        "logit_lamb": ParamSpec(
            initial=4.0,  # Corresponds to ~10 day memory at chunk_period=1440
            optuna=OptunaRange(low=-4.0, high=8.0, log_scale=False, scalar=False),
            description="Logit of decay parameter lambda (memory length)",
        ),
        # logit_delta_lamb: delta in logit space for alternative lambda
        "logit_delta_lamb": ParamSpec(
            initial=0.0,
            optuna=OptunaRange(low=-5.0, high=5.0, log_scale=False, scalar=False),
            description="Delta in logit space for alternative lambda calculation",
        ),
        # initial_weights_logits: no transform needed
        "initial_weights_logits": ParamSpec(
            initial=1.0,
            optuna=OptunaRange(low=-10, high=10, log_scale=False, scalar=False),
            description="Logit-space initial portfolio weights",
            trainable=False,
        ),
    }

    @classmethod
    def get_param_schema(cls) -> Dict[str, ParamSpec]:
        """Get the full parameter schema for this pool.

        Returns the pool-specific schema merged with common parameters
        from the base class.

        Returns
        -------
        Dict[str, ParamSpec]
            Complete parameter schema for this pool
        """
        return {**COMMON_PARAM_SCHEMA, **cls.PARAM_SCHEMA}

    def __init__(self):
        """
        Initialize a new MomentumPool instance.

        Parameters
        ----------
        None
        """
        super().__init__()

    @partial(jit, static_argnums=(2))
    def calculate_rule_outputs(
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
        rule_outputs = _jax_momentum_weight_update(gradients, k)
        return rule_outputs

    def calculate_readouts(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Calculate readouts (internal gradient estimator variables) for the pool,
        based on price history.

        This method gives the readout values for the gradient estimator (the ewma of
        prices and  the running a), sliced in the same way that the raw weight outputs
        are sliced.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters for weight calculation
        run_fingerprint : Dict[str, Any]
            Simulation settings
        prices : jnp.ndarray
            Historical price data
        additional_oracle_input : Optional[jnp.ndarray]
            Extra data for weight calculation

        Returns
        -------
        dict
            Dict containing readout value from the gradient estimator
        """
        chunk_period = run_fingerprint["chunk_period"]
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        memory_days = lamb_to_memory_days_clipped(
            calc_lamb(params),
            run_fingerprint["chunk_period"],
            run_fingerprint["max_memory_days"],
        )
        chunkwise_price_values = prices[:: run_fingerprint["chunk_period"]]
        gradients_dict = calc_gradients_with_readout(
            params,
            chunkwise_price_values,
            run_fingerprint["chunk_period"],
            run_fingerprint["max_memory_days"],
            run_fingerprint["use_alt_lamb"],
            cap_lamb=True,
        )
        # we have a sequence now of readout values and gradients, but if we are doing
        # a burnin operation, we need to cut off the changes associated
        # with the burnin period, ie everything before the start of the sequence

        start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)

        # if the chunk period is not a divisor of bout_length, we need to pad the readout values.
        # this can require more data to be available, potentially beyond the end of the bout.
        if bout_length % chunk_period != 0:
            additional_offset = 1
        else:
            additional_offset = 0

        # slice gradients
        gradients = dynamic_slice(
            gradients_dict["gradients"],
            start_index_coarse,
            (int((bout_length) / chunk_period) + additional_offset, n_assets),
        )
        # slice running a
        running_a = dynamic_slice(
            gradients_dict["running_a"],
            start_index_coarse,
            (int((bout_length) / chunk_period) + additional_offset, n_assets),
        )
        # slice ewma
        ewma = dynamic_slice(
            gradients_dict["ewma"],
            start_index_coarse,
            (int((bout_length) / chunk_period) + additional_offset, n_assets),
        )
        return {"gradients": gradients, "running_a": running_a, "ewma": ewma}

    def _get_estimator_constants(self, lamb: jnp.ndarray) -> tuple:
        """
        Compute G_inf and saturated_b from lambda.

        These are the standard constants needed for the gradient estimator:
        - G_inf = 1 / (1 - lamb)
        - saturated_b = lamb / ((1 - lamb) ** 3)

        Parameters
        ----------
        lamb : jnp.ndarray
            Lambda (decay) parameter

        Returns
        -------
        tuple
            (G_inf, saturated_b)
        """
        G_inf = 1.0 / (1.0 - lamb)
        saturated_b = lamb / ((1 - lamb) ** 3)
        return G_inf, saturated_b

    def get_initial_rule_state(
        self,
        initial_price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> Dict[str, jnp.ndarray]:
        """
        Initialize the carry state for scanning.

        For MomentumPool, the carry consists of:
        - ewma: initialized to the first price
        - running_a: initialized to zeros (steady-state for constant input)

        Parameters
        ----------
        initial_price : jnp.ndarray
            First price observation (shape: n_assets,)
        params : Dict[str, Any]
            Pool parameters
        run_fingerprint : Dict[str, Any]
            Simulation settings

        Returns
        -------
        Dict[str, jnp.ndarray]
            Initial carry state with 'ewma' and 'running_a' keys.
        """
        n_assets = initial_price.shape[0]
        return {
            "ewma": initial_price,
            "running_a": jnp.zeros((n_assets,), dtype=jnp.float64),
        }

    def calculate_rule_output_step(
        self,
        carry: Dict[str, jnp.ndarray],
        price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> tuple:
        """
        Calculate a single step of momentum weight update.

        This mirrors the production implementation where we:
        1. Update the gradient estimator state (ewma, running_a)
        2. Compute the gradient from the updated state
        3. Apply the momentum weight update formula

        Parameters
        ----------
        carry : Dict[str, jnp.ndarray]
            Current state with 'ewma' and 'running_a'
        price : jnp.ndarray
            Current price observation (shape: n_assets,)
        params : Dict[str, Any]
            Pool parameters (logit_lamb, log_k, etc.)
        run_fingerprint : Dict[str, Any]
            Simulation settings (chunk_period, max_memory_days, etc.)

        Returns
        -------
        tuple
            (new_carry, rule_output)
        """
        # Compute lambda with max_memory_days capping
        lamb = calc_lamb(params)
        max_lamb = jax_memory_days_to_lamb(
            run_fingerprint["max_memory_days"], run_fingerprint["chunk_period"]
        )
        lamb = jnp.clip(lamb, min=0.0, max=max_lamb)

        # Get estimator constants
        G_inf, saturated_b = self._get_estimator_constants(lamb)

        # Use the estimator primitive for gradient calculation
        carry_list = [carry["ewma"], carry["running_a"]]
        new_carry_list, gradient = _jax_gradient_scan_function(
            carry_list, price, G_inf, lamb, saturated_b
        )

        # Compute memory days and k for weight update
        memory_days = lamb_to_memory_days_clipped(
            lamb, run_fingerprint["chunk_period"], run_fingerprint["max_memory_days"]
        )
        k = calc_k(params, memory_days)

        # Apply momentum weight update
        rule_output = _jax_momentum_weight_update(gradient, k)

        new_carry = {
            "ewma": new_carry_list[0],
            "running_a": new_carry_list[1],
        }

        return new_carry, rule_output

    @partial(jit, static_argnums=(3))
    def calculate_fine_weights(
        self,
        rule_output: jnp.ndarray,
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
        rule_output : jnp.ndarray
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
            rule_output, initial_weights, run_fingerprint, params
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
        # np.random.seed(0)

        # We need to initialise the weights for each parameter set
        # If a vector is provided in the inital values dict, we use
        # that, if only a singleton array is provided we expand it
        # to n_assets and use that vlaue for all assets.
        def process_initial_values(
            initial_values_dict, key, n_assets, n_parameter_sets, force_scalar=False
        ):
            if key in initial_values_dict:
                initial_value = initial_values_dict[key]
                if isinstance(initial_value, (np.ndarray, jnp.ndarray, list)):
                    initial_value = np.array(initial_value)
                    if force_scalar:
                        return np.array([initial_value] * n_parameter_sets)
                    elif initial_value.size == n_assets:
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
                    if force_scalar:
                        return np.array([initial_value] * n_parameter_sets)
                    else:
                        return np.array([[initial_value] * n_assets] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        initial_weights_logits = process_initial_values(
            initial_values_dict, "initial_weights_logits", n_assets, n_parameter_sets, force_scalar=False
        )
        log_k = np.log2(
            process_initial_values(
                initial_values_dict, "initial_k_per_day", n_assets, n_parameter_sets, force_scalar=run_fingerprint["optimisation_settings"]["force_scalar"]
            )
        )

        initial_lamb = memory_days_to_lamb(
            initial_values_dict["initial_memory_length"],
            run_fingerprint["chunk_period"],
        )

        logit_lamb_np = np.log(initial_lamb / (1.0 - initial_lamb))
        if run_fingerprint["optimisation_settings"]["force_scalar"]:
            logit_lamb = np.array([[logit_lamb_np]] * n_parameter_sets)
        else:
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
        if run_fingerprint["optimisation_settings"]["force_scalar"]:
            logit_delta_lamb = np.array(
                [[logit_delta_lamb_np]] * n_parameter_sets
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
