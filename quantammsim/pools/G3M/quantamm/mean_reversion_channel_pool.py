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
    get_log_amplitude,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_gradients,
    calc_k,
    squareplus,
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
def _jax_mean_reversion_channel_weight_update(
    price_gradient,
    k,
    width,
    amplitude,
    exponents,
    inverse_scaling=0.5415,
    pre_exp_scaling=0.5,
):
    """
    Calculate weight updates using mean reversion channel strategy.

    Parameters
    ----------
    price_gradient : jnp.ndarray
        Array of price gradients for each asset.
    k : float or jnp.ndarray
        Scaling factor for weight updates.
    width : float or jnp.ndarray
        Width parameter for the mean reversion channel.
    amplitude : float or jnp.ndarray
        Amplitude of the mean reversion effect.
    exponents : jnp.ndarray
        Exponents for the trend following portion.
    inverse_scaling : float, optional
        Scaling factor for the channel portion, by default 0.5415.
    pre_exp_scaling : float, optional
        Scaling factor applied before exponentiation, by default 0.5.

    Returns
    -------
    jnp.ndarray
        Array of weight updates for each asset.

    Notes
    -----
    Combines a mean reversion channel component with a trend following component:
    1. Channel portion uses a Gaussian envelope and cubic function
    2. Trend portion uses power law scaling outside the channel
    """
    envelope = jnp.exp(-(price_gradient**2) / (2 * width**2))
    scaled_price_gradient = jnp.pi * price_gradient / (3 * width)

    channel_portion = (
        -amplitude
        * envelope
        * (scaled_price_gradient - (scaled_price_gradient**3) / 6)
        / inverse_scaling
    )

    trend_portion = (
        (1 - envelope)
        * jnp.sign(price_gradient)
        * jnp.power(jnp.abs(price_gradient / (2.0 * pre_exp_scaling)), exponents)
    )
    signal = channel_portion + trend_portion

    offset_constants = -(k * signal).sum(axis=-1, keepdims=True) / (jnp.sum(k))
    weight_updates = k * (signal + offset_constants)
    return weight_updates


class MeanReversionChannelPool(MomentumPool):
    """
    A class for mean reversion channel strategies run as TFMM liquidity pools.

    This class implements a "mean reversion channel" strategy for asset allocation within a TFMM framework.
    It uses price data to generate mean reversion channel signals, which are then translated into weight adjustments.

    Parameters
    ----------
    None

    Methods
    -------
    calculate_raw_weights_outputs(params, run_fingerprint, prices, additional_oracle_input)
        Calculate the raw weight outputs based on mean reversion channel signals.

    Notes
    -----
    The MeanReversionChannelPool implements a mean-reversion-based channel following strategy for asset allocation within a TFMM framework.
    It uses price data to generate mean-reversion signals, which are then translated into weight adjustments.
    The class provides methods to calculate raw weight outputs based on these signals and refine them
    into final asset weights, taking into account various parameters and constraints defined in the pool setup.
    """

    def __init__(self):
        """
        Initialize a new MeanReversionChannelPool instance.

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
        Calculate the raw weight outputs based on mean reversion channel signals.

        This method computes the raw weight adjustments for the mean reversion channel strategy. It processes
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
        5. Applies the mean reversion channel weight update formula to get raw weight outputs.

        The raw weight outputs are not the final weights, but rather the changes
        to be applied to the previous weights. These will be refined in subsequent steps.
        """

        use_pre_exp_scaling = run_fingerprint["use_pre_exp_scaling"]
        if use_pre_exp_scaling and params.get("logit_pre_exp_scaling") is not None:
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

        exponents = squareplus(params.get("raw_exponents"))
        amplitude = (2 ** params.get("log_amplitude")) * memory_days
        width = 2 ** params.get("raw_width")
        raw_weight_outputs = _jax_mean_reversion_channel_weight_update(
            gradients,
            k,
            width,
            amplitude,
            exponents,
            pre_exp_scaling=pre_exp_scaling,
        )

        return raw_weight_outputs

    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
        prices: Optional[jnp.ndarray] = None,
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
        (memory length parameter), the update agressiveness (k), the exponents and the width for each asset and parameter set.
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
                        return np.array([[initial_value]] * n_parameter_sets)
                    else:
                        return np.array([[initial_value] * n_assets] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")


        initial_weights_logits = process_initial_values(
            initial_values_dict, "initial_weights_logits", n_assets, n_parameter_sets, force_scalar=False
        )

        # If using spectral initialization, we spread memory and scale k differently
        if noise == "spectral" and n_parameter_sets > 1:
            # 0. Calculate Volatility if prices are available
            volatility = 1.0 # Default fallback
            if prices is not None:
                # Subsample prices to match the chunk period (timeframe of the strategy)
                chunk_period = run_fingerprint["chunk_period"]
                chunked_prices = prices[::chunk_period]
                
                # Calculate log returns on chunked data
                log_prices = jnp.log(chunked_prices + 1e-12)
                returns = jnp.diff(log_prices, axis=0)
                # Volatility per asset
                volatility = jnp.std(returns, axis=0)
                # Ensure volatility is at least something small to avoid log(0)
                volatility = jnp.maximum(volatility, 1e-6)
                
            # 1. Spread Memory Logarithmically (Filter Bank)
            min_mem = run_fingerprint["chunk_period"] / 1440.0  # Min memory is 1 chunk
            max_mem = run_fingerprint["max_memory_days"]
            # Generate log-spaced memory days for each parameter set (head)
            memory_spread = np.geomspace(min_mem, max_mem, n_parameter_sets)
            
            # Broadcast to assets (shape: [n_sets, n_assets])
            initial_memory_days = np.tile(memory_spread[:, None], (1, n_assets))
            
            # Convert to lambda
            initial_lamb = memory_days_to_lamb(
                initial_memory_days,
                run_fingerprint["chunk_period"],
            )
            
            # scaling factor for volatility-dependent parameters (1/sqrt(T))
            # We assume the provided initial values are calibrated for T=1 (Daily)
            # or simply that they need to shrink as T grows to match the smoother signal.
            vol_scaling = 1.0 / np.sqrt(initial_memory_days)

            # 2. Scale Speed (k)
            target_k = initial_values_dict["initial_k_per_day"]
            # We scale k by 1/sqrt(T) to match signal strength decay over time.
            k_param = target_k * vol_scaling
            log_k = np.log2(k_param)
            
            # 3. Scale Width
            # Width is compared to signal. Signal ~ 1/sqrt(T).
            # So Width should scale as 1/sqrt(T).
            # AND Width should scale with Volatility.
            
            base_width_raw = initial_values_dict["initial_raw_width"]

            # If prices are provided, we use volatility as the base width
            if prices is not None:
                 # Increase safety factor to 4.0 sigma to ensure stability (1/N weights) initially
                 safety_factor = 4.0
                 log_vol = np.log2(volatility * safety_factor)
                 # Broadcast log_vol to (n_sets, n_assets)
                 log_vol_broadcast = np.tile(log_vol, (n_parameter_sets, 1))
                 
                 raw_width = log_vol_broadcast + base_width_raw + np.log2(vol_scaling)
            else:
                 # Fallback: assume base_width_raw is the full width
                 raw_width = base_width_raw + np.log2(vol_scaling)
                 if np.ndim(raw_width) == 0:
                      raw_width = np.tile(raw_width, (n_parameter_sets, n_assets))
                 elif np.ndim(raw_width) == 1:
                      raw_width = np.tile(raw_width, (n_parameter_sets, 1))

            # 4. Scale Pre-Exp Scaling
            # Same logic as Width. Normalizes the signal.
            base_scaling_val = initial_values_dict.get("initial_pre_exp_scaling", 0.5)
            
            if prices is not None:
                # Target Scaling = Volatility * base_scaling_val
                target_scaling = volatility * base_scaling_val
                # Apply 1/sqrt(T) scaling
                target_scaling = target_scaling * vol_scaling
                # Broadcast
                # volatility is (n_assets,), vol_scaling is (n_sets, n_assets)
                # We need (n_sets, n_assets)
                target_scaling = np.tile(volatility, (n_parameter_sets, 1)) * base_scaling_val * vol_scaling
                raw_pre_exp_scaling = np.log2(target_scaling)
            else:
                scaled_scaling_val = base_scaling_val * vol_scaling
                raw_pre_exp_scaling = np.log2(scaled_scaling_val)

            # 5. Scale Amplitude
            base_amp_log = initial_values_dict["initial_log_amplitude"]
            
            # To achieve a "Do Nothing" baseline structurally, we suppress the mean-reversion component.
            # We subtract a constant (e.g. 3.0 -> factor of 8) to make the channel reaction weak by default.
            # This relies on the "Trend" component (outside channel) to drive major moves,
            # which is gated by the wide channel width.
            amplitude_suppression = 3.0
            log_amplitude = base_amp_log + np.log2(vol_scaling) - amplitude_suppression
            
            if np.ndim(log_amplitude) == 0:
                 log_amplitude = np.tile(log_amplitude, (n_parameter_sets, n_assets))
            
            # 6. Exponents (Constant)
            base_exp = initial_values_dict["initial_raw_exponents"]
            raw_exponents = np.tile(base_exp, (n_parameter_sets, n_assets))
            
        else:
            # Standard initialization
            log_k = np.log2(
                process_initial_values(
                    initial_values_dict, "initial_k_per_day", n_assets, n_parameter_sets, force_scalar=run_fingerprint["optimisation_settings"]["force_scalar"]
                )
            )

            initial_lamb = memory_days_to_lamb(
                initial_values_dict["initial_memory_length"],
                run_fingerprint["chunk_period"],
            )
            # Broadcast standard lamb
            if run_fingerprint["optimisation_settings"]["force_scalar"]:
                initial_lamb = np.array([[initial_lamb]] * n_parameter_sets)
            else:
                initial_lamb = np.array([[initial_lamb] * n_assets] * n_parameter_sets)
            
            # Standard params processing
            raw_pre_exp_scaling_np = np.log2(
                initial_values_dict["initial_pre_exp_scaling"]
            )
            if run_fingerprint["optimisation_settings"]["force_scalar"]:
                raw_pre_exp_scaling = np.array([[raw_pre_exp_scaling_np]] * n_parameter_sets)
            else:
                raw_pre_exp_scaling = np.array(
                    [[raw_pre_exp_scaling_np] * n_assets] * n_parameter_sets
                )

            if run_fingerprint["optimisation_settings"]["force_scalar"]:
                log_amplitude = np.array([[initial_values_dict["initial_log_amplitude"]]] * n_parameter_sets)
            else:
                log_amplitude = np.array(
                    [[initial_values_dict["initial_log_amplitude"]] * n_assets]
                    * n_parameter_sets
                )

            if run_fingerprint["optimisation_settings"]["force_scalar"]:
                raw_width = np.array([[initial_values_dict["initial_raw_width"]]] * n_parameter_sets)
            else:
                raw_width = np.array(
                    [[initial_values_dict["initial_raw_width"]] * n_assets] * n_parameter_sets
                )

            if run_fingerprint["optimisation_settings"]["force_scalar"]:
                raw_exponents = np.array([[initial_values_dict["initial_raw_exponents"]]] * n_parameter_sets)
            else:
                raw_exponents = np.array(
                    [[initial_values_dict["initial_raw_exponents"]] * n_assets]
                    * n_parameter_sets
                )

        logit_lamb = np.log(initial_lamb / (1.0 - initial_lamb))

        # lamb delta is the difference in lamb needed for
        # lamb + delta lamb to give a final memory length
        # of  initial_memory_length + initial_memory_length_delta
        # Note: For spectral init, we keep delta_lamb at 0 or scaled? 
        # Standard approach: keep it based on the input delta, but relative to the new spread lamb?
        # Simpler to just zero it or use the standard logic which might shift the spread uniformly.
        # Let's stick to standard logic but using the potentially spread initial_lamb.
        
        # Calculate target 'plus delta' memory
        # If spectral, this delta shift applies to the whole bank.
        initial_lamb_plus_delta_lamb = memory_days_to_lamb(
            lamb_to_memory_days_clipped(initial_lamb, run_fingerprint["chunk_period"], run_fingerprint["max_memory_days"])
            + initial_values_dict["initial_memory_length_delta"],
            run_fingerprint["chunk_period"],
        )

        logit_lamb_plus_delta_lamb_np = np.log(
            initial_lamb_plus_delta_lamb / (1.0 - initial_lamb_plus_delta_lamb)
        )
        logit_delta_lamb = logit_lamb_plus_delta_lamb_np - logit_lamb # broadcasting works
        
        # Force scalar if needed (though spectral init implies non-scalar across parameter sets)
        if run_fingerprint["optimisation_settings"]["force_scalar"] and noise != "spectral":
             logit_lamb = logit_lamb[:, 0:1] # Keep shape (n_sets, 1)
             logit_delta_lamb = logit_delta_lamb[:, 0:1]

        params = {
            "log_k": log_k,
            "logit_lamb": logit_lamb,
            "logit_delta_lamb": logit_delta_lamb,
            "initial_weights_logits": initial_weights_logits,
            "log_amplitude": log_amplitude,
            "raw_width": raw_width,
            "raw_exponents": raw_exponents,
            "raw_pre_exp_scaling": raw_pre_exp_scaling,
            "subsidary_params": [],
        }

        # Apply noise (jitter) on top of the structured initialization
        # Note: If noise="spectral", we still want the random jitter from "gaussian" logic?
        # The add_noise method likely handles "gaussian" string. 
        # We can pass "gaussian" to add_noise if noise=="spectral" to get the jitter.
        noise_type = "gaussian" if noise == "spectral" else noise
        params = self.add_noise(params, noise_type, n_parameter_sets)
        return params

    @classmethod
    def _process_specific_parameters(cls, update_rule_parameters, run_fingerprint):
        """Process mean reversion channel specific parameters."""
        result = {}
        amplitude_values = None
        memory_days = None

        # Get memory_days value for amplitude calculation
        for urp in update_rule_parameters:
            if urp.name == "memory_days":
                memory_days = urp.value
                break

        # Process specific parameters
        for urp in update_rule_parameters:
            if urp.name == "amplitude":
                amplitude_values = urp.value
            elif urp.name == "exponent":
                raw_exponents = [float(inverse_squareplus_np(val)) for val in urp.value]
                if len(raw_exponents) != len(run_fingerprint["tokens"]):
                    raw_exponents = [raw_exponents[0]] * len(run_fingerprint["tokens"])
                result["raw_exponents"] = np.array(raw_exponents)
            elif urp.name == "width":
                raw_width = [float(get_raw_value(val)) for val in urp.value]
                result["raw_width"] = np.array(raw_width)
                if len(raw_width) != len(run_fingerprint["tokens"]):
                    raw_width = [raw_width[0]] * len(run_fingerprint["tokens"])
                result["raw_width"] = np.array(raw_width)
            elif urp.name == "pre_exp_scaling":
                raw_pre_exp_scaling = [float(get_raw_value(val)) for val in urp.value]
                if len(raw_pre_exp_scaling) != len(run_fingerprint["tokens"]):
                    raw_pre_exp_scaling = [raw_pre_exp_scaling[0]] * len(run_fingerprint["tokens"])
                result["raw_pre_exp_scaling"] = np.array(raw_pre_exp_scaling)

        # Process amplitude last
        if amplitude_values is not None:
            if memory_days is None:
                raise ValueError("memory_days parameter is required for amplitude calculation")
            log_amplitude = [
                get_log_amplitude(float(amp), float(mem)) 
                for amp, mem in zip(amplitude_values, memory_days)
            ]
            if len(log_amplitude) != len(run_fingerprint["tokens"]):
                log_amplitude = [log_amplitude[0]] * len(run_fingerprint["tokens"])
            result["log_amplitude"] = np.array(log_amplitude)
        return result

tree_util.register_pytree_node(
    MeanReversionChannelPool,
    MeanReversionChannelPool._tree_flatten,
    MeanReversionChannelPool._tree_unflatten,
)
