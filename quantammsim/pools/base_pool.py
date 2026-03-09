from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from functools import partial
import numpy as np

import jax.numpy as jnp
from jax.nn import softmax
from jax.lax import stop_gradient, dynamic_slice
from jax import tree_util, jit, vmap

from quantammsim.core_simulator.param_utils import make_vmap_in_axes_dict

class AbstractPool(ABC):
    """
    Abstract base class for implementing various types of liquidity pools.

    This class defines the basic structure and interface for different pool implementations
    in the quantammsim simulator. It provides abstract methods that must be implemented by
    concrete subclasses to define specific pool behaviors.

    Methods
    -------
    calculate_reserves_with_fees(params, run_fingerprint, prices, start_index, additional_oracle_input)
        Calculate reserve changes with fees and arbitrage enabled.
        
        Used when fees are non-zero and arbitrage is enabled. Handles arbitrage thresholds,
        trading costs, and fee calculations. Less performant than zero-fees case.

    calculate_reserves_zero_fees(params, run_fingerprint, prices, start_index, additional_oracle_input) 
        Calculate reserve changes assuming zero fees.

        Fast, vectorized implementation for the zero-fees case. Uses parallel computation
        since arbitrageurs will always trade to exactly match external market prices.
        Should be overridden with fees=0 version of calculate_reserves_with_fees if no
        faster implementation exists.

    calculate_reserves_with_dynamic_inputs(params, run_fingerprint, prices, start_index, additional_oracle_input)
        Calculate reserve changes with time-varying parameters.
        
        Handles cases where pool properties like fees, arbitrage thresholds, or weights
        can change over time. Required for pools with dynamic parameters.

    Notes
    -----
    Subclasses of AbstractPool should implement the abstract methods to define
    specific behaviors for different types of liquidity pools.
    """

    @property
    def supports_fused_reserves(self) -> bool:
        """Whether this pool supports the fused chunked reserve computation path."""
        return False

    def __init__(self):
        pass

    def extend_parameters(
        self,
        base_params: Dict[str, Any],
        initial_values_dict: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int,
    ) -> Dict[str, Any]:
        """Default null implementation of parameter extension."""
        return base_params

    @abstractmethod
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    # In almost all cases it is possible to write a fast (i.e. embarrassingly-
    # parallelisable and easily vmap-able) function for how reserves
    # change when there are no fees. If there is not a faster way
    # to do it, then this method 'calculate_reserves_zero_fees'
    # should be overridden with the concrete version of
    # calculate_reserve_changes with fees set to zero.
    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def calculate_reserves_with_dynamic_inputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        dynamic_inputs: Any,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        pass

    def init_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        """Initialize pool parameters and apply any extensions from mixins."""
        # Get base parameters
        params = self.init_base_parameters(
            initial_values_dict, run_fingerprint, n_assets, n_parameter_sets, noise
        )

        # Apply any parameter extensions from mixins
        params = self.extend_parameters(
            params, initial_values_dict, n_assets, n_parameter_sets
        )

        return params

    def calculate_initial_weights(
        self, params: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """
        Calculate initial pool weights from initial logits or from directly-provided weights.
        If both are provided, the weights calculated from logits take precedence.

        Uses softmax with stop_gradient to ensure weights remain constant
        during any optimization.

        Parameters
        ----------
        params : Dict[str, jnp.ndarray]
            Must contain 'initial_weights_logits' key or 'initial_weights' key
        *args, **kwargs
            Not used, kept for interface compatibility

        Returns
        -------
        jnp.ndarray
            Fixed normalized weights

        Notes
        -----
        Using 'initial_weights_logits' means that the calculated initial weights
        have +ve entries and sum to one by construction. If 'initial_weights' is
        used the values are used unchecked.
        """
        initial_weights_logits = params.get("initial_weights_logits", None)
        initial_weights = params.get("initial_weights", None)
        if initial_weights_logits is not None:
            # we don't want to change the initial weights during any training
            # so wrap them in a stop_grad
            weights = softmax(stop_gradient(initial_weights_logits))
        elif initial_weights is not None:
            # we don't want to change the initial weights during any training
            # so wrap them in a stop_grad
            weights = stop_gradient(initial_weights)
        else:
            raise ValueError(
                "At least one of 'initial_weights_logits' and 'initial_weights' must be provided"
            )
        return weights

    def calculate_weights(
        self, params: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        """
        This function will be overridden for any pools that a) have weights and b) have weights that vary.
        As so many of the pools modelled in this package have weights (Balancer [G3M], Cow [FM-AMM], QuantAMM [TFMM])
        this is helpful to have here (though this method is overriden for QuantAMM [TFMM] pools).

        This method is used by some hooks that rely on having access to a pools weights over time. If a pool is to work
        with all hooks, this method should be ensured to implement the correct logic for that pool. See GyroscopePool
        for an example where a custom implementation was needed for the sake of hook compatibility.

        Parameters
        ----------
        params : Dict[str, jnp.ndarray]
            Must contain 'initial_weights_logits' key or 'initial_weights' key
        *args, **kwargs
            Not used, kept for interface compatibility

        Returns
        -------
        jnp.ndarray
            Fixed normalized weights

        Notes
        -----
        Using 'initial_weights_logits' means that the calculated initial weights
        have +ve entries and sum to one by construction. If 'initial_weights' is used the
        values are used unchecked.
        """
        return self.calculate_initial_weights(params, *args, **kwargs)

    @abstractmethod
    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        """Initialize base parameters specific to this pool type."""
        pass

    def add_noise(
        self,
        params: Dict[str, np.ndarray],
        noise: str,
        n_parameter_sets: int,
        noise_scale: float = 1.0,
        per_param_noise_scale: Optional[Dict[str, float]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Add noise to parameter sets for initialization diversity.

        Parameters
        ----------
        params : dict
            Parameter arrays, each with shape (n_parameter_sets, ...).
        noise : str
            Noise method: "gaussian", "sobol", "lhs", "centered_lhs".
        n_parameter_sets : int
            Number of parameter sets. First set is left unaltered.
        noise_scale : float
            Global noise scale (default 1.0).
        per_param_noise_scale : dict, optional
            Per-parameter noise scale overrides. Keys are param names,
            values are scale multipliers. Params not in this dict use
            the global noise_scale. Useful for wider exploration of
            under-studied params (e.g., raw_width, pre_exp_scaling).
        """
        if n_parameter_sets > 1:
            if noise == "gaussian":
                for key in params.keys():
                    if key != "subsidary_params" and key != "initial_weights_logits":
                        scale = noise_scale
                        if per_param_noise_scale and key in per_param_noise_scale:
                            scale = per_param_noise_scale[key]
                        params[key][1:] = params[key][1:] + scale * np.random.randn(
                            *params[key][1:].shape
                        )
            elif noise in ("sobol", "lhs", "centered_lhs"):
                from scipy.stats import norm
                from quantammsim.utils.sampling import generate_param_space_samples

                n_new = n_parameter_sets - 1
                samples, trainable_keys, dim_map = generate_param_space_samples(
                    params, n_new, method=noise, seed=0,
                )

                # Transform [0,1] → normal offsets, clip to avoid inf at boundaries
                samples = np.clip(samples, 1e-6, 1.0 - 1e-6)

                if per_param_noise_scale:
                    # Apply per-param scales: build a scale vector matching sample columns
                    col_scales = np.full(samples.shape[1], noise_scale)
                    for key in trainable_keys:
                        if key in per_param_noise_scale:
                            start_col, n_dims, _ = dim_map[key]
                            col_scales[start_col:start_col + n_dims] = per_param_noise_scale[key]
                    normal_offsets = norm.ppf(samples) * col_scales[np.newaxis, :]
                else:
                    normal_offsets = norm.ppf(samples) * noise_scale

                # Distribute offsets back to each param array
                for key in trainable_keys:
                    start_col, n_dims, shape_per_sample = dim_map[key]
                    shape = params[key][1:].shape
                    offsets = normal_offsets[:, start_col:start_col + n_dims].reshape(shape)
                    params[key][1:] = params[key][1:] + offsets

        for key in params.keys():
            if key != "subsidary_params":
                params[key] = jnp.array(params[key])
        return params

    @partial(jit, static_argnums=(2, 3))
    def calculate_volatility_array(self, prices, run_fingerprint, subsample_freq=5):
        """Annualised daily realised volatility broadcast to minute-level array.

        Pure-JAX implementation (vmap + dynamic_slice) — JIT-compatible and
        callable from within traced contexts (e.g. forward_pass).

        Parameters
        ----------
        prices : jnp.ndarray, shape (T, 2)
            Minute-level prices for two tokens.
        run_fingerprint : dict
            Must contain ``tokens`` and ``numeraire`` for ordering.
        subsample_freq : int
            Subsample within each day to reduce microstructure noise.

        Returns
        -------
        jnp.ndarray, shape (T,)
            Annualised volatility, constant within each day.
        """
        ordered_prices, needs_swap = self._handle_numeraire_ordering(
            prices, run_fingerprint,
        )
        asset_prices = ordered_prices[:, 0] / ordered_prices[:, 1]
        n_minutes = len(asset_prices)

        # Guard: need at least one full day for vmap + dynamic_slice
        if n_minutes < 1440:
            return jnp.full(n_minutes, 0.1) * jnp.sqrt(365.0)

        n_days = n_minutes // 1440

        def calculate_daily_volatility(day_idx):
            start_idx = day_idx * 1440
            window_prices = dynamic_slice(asset_prices, [start_idx], [1440])
            subsampled_prices = window_prices[::subsample_freq]
            log_prices = jnp.log(jnp.maximum(subsampled_prices, 1e-8))
            returns = jnp.diff(log_prices)
            num_nonzero_returns = jnp.sum(returns != 0)
            total_returns = len(returns)
            adjusted_variance = (
                num_nonzero_returns * jnp.var(returns) / total_returns
            )
            dt = subsample_freq / 1440
            vol = jnp.sqrt(adjusted_variance) / jnp.sqrt(dt)
            return vol

        daily_volatilities = vmap(calculate_daily_volatility)(jnp.arange(n_days))
        volatility_array = jnp.repeat(daily_volatilities, 1440)

        remaining_minutes = n_minutes - len(volatility_array)
        if remaining_minutes > 0:
            last_vol = (
                daily_volatilities[-1] if len(daily_volatilities) > 0 else 0.1
            )
            volatility_array = jnp.concatenate(
                [volatility_array, jnp.full(remaining_minutes, last_vol)]
            )

        return volatility_array * jnp.sqrt(365.0)

    @partial(jit, static_argnums=(2,))
    def _handle_numeraire_ordering(
        self,
        prices: jnp.ndarray,
        run_fingerprint: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, bool]:
        """Reorder prices so numeraire token is in second position.

        Parameters
        ----------
        prices : jnp.ndarray, shape (..., 2)
            Price array with two tokens.
        run_fingerprint : dict
            Must contain ``tokens`` (sorted) and ``numeraire``.

        Returns
        -------
        (ordered_prices, needs_swap) : (jnp.ndarray, bool)
        """
        tokens = sorted(run_fingerprint["tokens"])
        numeraire = run_fingerprint["numeraire"]
        if numeraire is None or numeraire not in tokens:
            numeraire = tokens[-1]
        needs_swap = tokens.index(numeraire) == 0

        if needs_swap:
            ordered_prices = prices[..., ::-1]
        else:
            ordered_prices = prices
        return ordered_prices, needs_swap

    def _tree_flatten(self):
        children = ()
        aux_data = dict()  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def make_vmap_in_axes(self, params: Dict[str, Any], n_repeats_of_recurred: int = 0):
        """
        Configure JAX vectorization axes for pool parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters
        n_repeats_of_recurred : int
            Number of times to repeat recurrent parameters

        Returns
        -------
        Dict[str, Any]
            vmap axes configuration
        """
        return make_vmap_in_axes_dict(params, 0, [], [], n_repeats_of_recurred)

    def get_initial_values(self, run_fingerprint):
        """Extract initial parameter values from run_fingerprint.

        Override in subclasses to define pool-specific initial values.
        """
        return {}

    @abstractmethod
    def is_trainable(self):
        pass

    @classmethod
    def process_parameters(cls, update_rule_parameters, run_fingerprint):
        """
        Default implementation for processing pool parameters from web interface input.

        Performs simple conversion of parameter values to numpy arrays while preserving names.
        Override this method in subclasses that need custom parameter processing.

        Parameters
        ----------
        update_rule_parameters : Dict[str, Any]
            Dict of parameters from the web interface
        run_fingerprint : Dict[str, Any]
            Run fingerprint dictionary
        Returns
        -------
        Dict[str, np.ndarray]
            Processed parameters ready for pool initialization
        """
        result = {}
        for urp in update_rule_parameters:
            result[urp.name] = np.array(urp.value)
        return result

    def weights_needs_original_methods(self) -> bool:
        """Indicates if calculate_weights needs access to original pool methods.
        
        Returns
        -------
        bool
            False by default - most pools don't need original methods. Override in subclasses
            if they do.
        """
        return False


tree_util.register_pytree_node(
    AbstractPool, AbstractPool._tree_flatten, AbstractPool._tree_unflatten
)
