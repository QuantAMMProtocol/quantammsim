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

from quantammsim.pools.G3M.quantamm.momentum_pool import (
    MomentumPool,
    _jax_momentum_weight_update,
)
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
    inverse_squareplus_np,
    get_raw_value,
    jax_memory_days_to_lamb,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_gradients,
    calc_k,
    squareplus,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
    _jax_gradient_scan_function,
)

from typing import Dict, Any, Optional
from functools import partial
import numpy as np

# import the fine weight output function which has pre-set argument raw_weight_outputs_are_themselves_weights
# as this is False for momentum pools --- the strategy outputs weight _changes_
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output_from_weight_changes,
)


@jit
def _jax_power_channel_weight_update(price_gradient, k, exponents, pre_exp_scaling=0.5):
    """
    Calculate weight updates using power channel strategy.

    Parameters
    ----------
    price_gradient : jnp.ndarray
        Array of price gradients for each asset.
    k : float or jnp.ndarray
        Scaling factor for weight updates.
    exponents : jnp.ndarray
        Exponents for the power law scaling.
    pre_exp_scaling : float, optional
        Scaling factor applied before exponentiation, by default 0.5.

    Returns
    -------
    jnp.ndarray
        Array of weight updates for each asset.

    Notes
    -----
    Applies a power law transformation to price gradients with:
    1. Pre-scaling of gradients
    2. Power law transformation with specified exponents
    3. Offset calculation to ensure zero sum weight updates
    """
    signal = jnp.sign(price_gradient) * jnp.power(
        jnp.abs(price_gradient / (2.0 * pre_exp_scaling)), exponents
    )
    sum_k = jnp.sum(k)
    offset_constants = -(k * signal).sum(axis=-1, keepdims=True) / sum_k
    weight_updates = k * (signal + offset_constants)
    return weight_updates


class PowerChannelPool(MomentumPool):
    """
    A class for power channel strategies run as TFMM liquidity pools.

    This class implements a "power channel" strategy for asset allocation within a TFMM framework.
    It uses price data to generate power channel signals, which are then translated into weight adjustments.

    Parameters
    ----------
    None

    Methods
    -------
    calculate_raw_weights_outputs(params, run_fingerprint, prices, additional_oracle_input)
        Calculate the raw weight outputs based on power channel signals.

    Notes
    -----
    The PowerChannelPool implements a "power channel" strategy for asset allocation within a TFMM framework.
    It uses price data to generate power channel signals, which are then translated into weight adjustments.
    The class provides methods to calculate raw weight outputs based on these signals and refine them
    into final asset weights, taking into account various parameters and constraints defined in the pool setup.
    """

    def __init__(self):
        """
        Initialize a new PowerChannelPool instance.

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
        Calculate the raw weight outputs based on power channel signals.

        This method computes the raw weight adjustments for the power channel strategy. It processes
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
        5. Applies the power channel weight update formula to get raw weight outputs.

        The raw weight outputs are not the final weights, but rather the changes
        to be applied to the previous weights. These will be refined in subsequent steps.
        """
        use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
        # pre_exp_scaling: prefer sp_ (squareplus), fall back to logit_ (sigmoid), then raw_ (2^x)
        if use_pre_exp_scaling and params.get("sp_pre_exp_scaling") is not None:
            pre_exp_scaling = squareplus(params.get("sp_pre_exp_scaling"))
        elif use_pre_exp_scaling and params.get("logit_pre_exp_scaling") is not None:
            logit_pre_exp_scaling = params.get("logit_pre_exp_scaling")
            pre_exp_scaling = jnp.exp(logit_pre_exp_scaling) / (
                1 + jnp.exp(logit_pre_exp_scaling)
            )
        elif use_pre_exp_scaling and params.get("raw_pre_exp_scaling") is not None:
            pre_exp_scaling = 2 ** params.get("raw_pre_exp_scaling")
        else:
            pre_exp_scaling = 0.5
        memory_days = lamb_to_memory_days_clipped(
            calc_lamb(params),
            run_fingerprint["chunk_period"],
            run_fingerprint["max_memory_days"],
        )
        # k: prefer sp_k (squareplus), fall back to log_k (2^x)
        if params.get("sp_k") is not None:
            k = squareplus(params.get("sp_k")) * memory_days
        else:
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

        # exponents: prefer sp_exponents, fall back to raw_exponents (both use squareplus)
        if params.get("sp_exponents") is not None:
            exponents = jnp.clip(squareplus(params.get("sp_exponents")), a_min=1.0, a_max=None)
        else:
            exponents = jnp.clip(squareplus(params.get("raw_exponents")), a_min=1.0, a_max=None)

        raw_weight_outputs = _jax_power_channel_weight_update(
            gradients, k, exponents, pre_exp_scaling=pre_exp_scaling
        )

        return raw_weight_outputs

    def calculate_single_step_weight_update(
        self,
        carry: Dict[str, jnp.ndarray],
        price: jnp.ndarray,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
    ) -> tuple:
        """
        Calculate a single step of power channel weight update.

        This mirrors the production implementation where we:
        1. Update the gradient estimator state (ewma, running_a)
        2. Compute the gradient from the updated state
        3. Apply the power channel weight update formula

        Parameters
        ----------
        carry : Dict[str, jnp.ndarray]
            Current state with 'ewma' and 'running_a'
        price : jnp.ndarray
            Current price observation (shape: n_assets,)
        params : Dict[str, Any]
            Pool parameters (logit_lamb, sp_k, sp_exponents, etc.)
        run_fingerprint : Dict[str, Any]
            Simulation settings (chunk_period, max_memory_days, use_pre_exp_scaling, etc.)

        Returns
        -------
        tuple
            (new_carry, raw_weight_output)
        """
        # Compute lambda with max_memory_days capping
        lamb = calc_lamb(params)
        max_lamb = jax_memory_days_to_lamb(
            run_fingerprint["max_memory_days"], run_fingerprint["chunk_period"]
        )
        lamb = jnp.clip(lamb, a_min=0.0, a_max=max_lamb)

        # Get estimator constants (inherited from MomentumPool)
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

        # k: prefer sp_k (squareplus), fall back to log_k (2^x)
        if params.get("sp_k") is not None:
            k = squareplus(params.get("sp_k")) * memory_days
        else:
            k = calc_k(params, memory_days)

        # pre_exp_scaling: prefer sp_ (squareplus), fall back to logit_ (sigmoid), then raw_ (2^x)
        use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
        if use_pre_exp_scaling and params.get("sp_pre_exp_scaling") is not None:
            pre_exp_scaling = squareplus(params.get("sp_pre_exp_scaling"))
        elif use_pre_exp_scaling and params.get("logit_pre_exp_scaling") is not None:
            logit_pre_exp_scaling = params.get("logit_pre_exp_scaling")
            pre_exp_scaling = jnp.exp(logit_pre_exp_scaling) / (
                1 + jnp.exp(logit_pre_exp_scaling)
            )
        elif use_pre_exp_scaling and params.get("raw_pre_exp_scaling") is not None:
            pre_exp_scaling = 2 ** params.get("raw_pre_exp_scaling")
        else:
            pre_exp_scaling = 0.5

        # exponents: prefer sp_exponents, fall back to raw_exponents (both use squareplus)
        if params.get("sp_exponents") is not None:
            exponents = jnp.clip(squareplus(params.get("sp_exponents")), a_min=1.0, a_max=None)
        else:
            exponents = jnp.clip(squareplus(params.get("raw_exponents")), a_min=1.0, a_max=None)

        # Apply power channel weight update
        raw_weight_output = _jax_power_channel_weight_update(
            gradient, k, exponents, pre_exp_scaling=pre_exp_scaling
        )

        new_carry = {
            "ewma": new_carry_list[0],
            "running_a": new_carry_list[1],
        }

        return new_carry, raw_weight_output

    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        """
        Initialize parameters for a power channel pool.

        This method sets up the initial parameters for the power channel pool strategy, including
        weights, memory length (lambda), the update agressiveness (k) and the exponents.

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
        (memory length parameter), the update agressiveness (k) and the exponents for each asset and parameter set.
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
        # sp_k: use inverse_squareplus to get param that squareplus maps to initial_k_per_day
        sp_k = inverse_squareplus_np(
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
        if run_fingerprint["optimisation_settings"]["force_scalar"]:
            logit_delta_lamb = np.array([[logit_delta_lamb_np]] * n_parameter_sets)
        else:
            logit_delta_lamb = np.array(
                [[logit_delta_lamb_np] * n_assets] * n_parameter_sets
            )

        # sp_pre_exp_scaling: use inverse_squareplus to get param that squareplus maps to initial_pre_exp_scaling
        sp_pre_exp_scaling_np = inverse_squareplus_np(
            initial_values_dict["initial_pre_exp_scaling"]
        )
        if run_fingerprint["optimisation_settings"]["force_scalar"]:
            sp_pre_exp_scaling = np.array([[sp_pre_exp_scaling_np]] * n_parameter_sets)
        else:
            sp_pre_exp_scaling = np.array(
                [[sp_pre_exp_scaling_np] * n_assets] * n_parameter_sets
            )

        # sp_exponents: the initial_raw_exponents value is already in the right form for squareplus
        if run_fingerprint["optimisation_settings"]["force_scalar"]:
            sp_exponents = np.array([[initial_values_dict["initial_raw_exponents"]]] * n_parameter_sets)
        else:
            sp_exponents = np.array(
                [[initial_values_dict["initial_raw_exponents"]] * n_assets]
                * n_parameter_sets
            )

        params = {
            "sp_k": sp_k,
            "logit_lamb": logit_lamb,
            "logit_delta_lamb": logit_delta_lamb,
            "initial_weights_logits": initial_weights_logits,
            "sp_exponents": sp_exponents,
            "sp_pre_exp_scaling": sp_pre_exp_scaling,
            "subsidary_params": [],
        }

        params = self.add_noise(params, noise, n_parameter_sets)
        return params

    @classmethod
    def _process_specific_parameters(cls, update_rule_parameters, run_fingerprint):
        """Process power channel specific parameters."""
        result = {}

        # Process specific parameters
        for urp in update_rule_parameters:
            if urp.name == "exponent":
                # Use inverse_squareplus to get sp_exponents param
                sp_exponents = [float(inverse_squareplus_np(val)) for val in urp.value]
                if len(sp_exponents) != len(run_fingerprint["tokens"]):
                    sp_exponents = [sp_exponents[0]] * len(run_fingerprint["tokens"])
                result["sp_exponents"] = np.array(sp_exponents)
            elif urp.name == "pre_exp_scaling":
                # Use inverse_squareplus to get sp_pre_exp_scaling param
                sp_pre_exp_scaling = [float(inverse_squareplus_np(val)) for val in urp.value]
                if len(sp_pre_exp_scaling) != len(run_fingerprint["tokens"]):
                    sp_pre_exp_scaling = [sp_pre_exp_scaling[0]] * len(run_fingerprint["tokens"])
                result["sp_pre_exp_scaling"] = np.array(sp_pre_exp_scaling)

        return result


tree_util.register_pytree_node(
    PowerChannelPool, PowerChannelPool._tree_flatten, PowerChannelPool._tree_unflatten
)
