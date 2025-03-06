from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

import jax.numpy as jnp
from jax.nn import softmax
from jax.lax import stop_gradient
from jax import tree_util

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
        fees_array: jnp.ndarray,
        arb_thresh_array: jnp.ndarray,
        arb_fees_array: jnp.ndarray,
        trade_array: jnp.ndarray,
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
        If both are provided, the weights calculated from logits take precendence.

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
            # we dont't want to change the initial weights during any training
            # so wrap them in a stop_grad
            weights = softmax(stop_gradient(initial_weights_logits))
        elif initial_weights is not None:
            # we dont't want to change the initial weights during any training
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
        self, params: Dict[str, np.ndarray], noise: str, n_parameter_sets: int
    ) -> Dict[str, jnp.ndarray]:
        if n_parameter_sets > 1:
            if noise == "gaussian":
                for key in params.keys():
                    if key != "subsidary_params":
                        # Leave first row of each jax parameter unaltered, add
                        # gaussian noise to subsequent rows.
                        params[key][1:] = params[key][1:] + np.random.randn(
                            *params[key][1:].shape
                        )
        for key in params.keys():
            if key != "subsidary_params":
                params[key] = jnp.array(params[key])
        return params

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

    @abstractmethod
    def is_trainable(self):
        pass

    @classmethod
    def process_parameters(cls, update_rule_parameters, n_assets):
        """
        Default implementation for processing pool parameters from web interface input.

        Performs simple conversion of parameter values to numpy arrays while preserving names.
        Override this method in subclasses that need custom parameter processing.

        Parameters
        ----------
        update_rule_parameters : List[UpdateRuleParameter]
            List of parameters from the web interface

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
