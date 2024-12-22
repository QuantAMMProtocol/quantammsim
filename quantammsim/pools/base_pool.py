from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

import jax.numpy as jnp
from jax import tree_util


class AbstractPool(ABC):
    """
    Abstract base class for implementing various types of liquidity pools.

    This class defines the basic structure and interface for different pool implementations
    in the quantammsim simulator. It provides abstract methods that must be implemented by
    concrete subclasses to define specific pool behaviors.

    Methods
    -------
    calculate_reserve_changes(params, run_fingerprint, prices, start_index, additional_oracle_input)
        Calculate changes in reserves based on weights, prices, and parameters.

    calculate_reserve_changes_zero_fees(params, run_fingerprint, prices, 
    start_index, additional_oracle_input)
        Calculate reserve changes assuming zero fees, based on weights, prices, and parameters.

    initialize_parameters(initial_values_dict, run_fingerprint, n_assets, n_parameter_sets, noise)
        Initialize the pool's parameters.

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
        params = self._init_base_parameters(
            initial_values_dict, run_fingerprint, n_assets, n_parameter_sets, noise
        )

        # Apply any parameter extensions from mixins
        params = self.extend_parameters(
            params, initial_values_dict, n_assets, n_parameter_sets
        )

        return params

    @abstractmethod
    def _init_base_parameters(
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

    @abstractmethod
    def make_vmap_in_axes(self, input_dict: Dict[str, Any], n_repeats_of_recurred: int):
        pass

    @abstractmethod
    def is_trainable(self):
        pass


tree_util.register_pytree_node(
    AbstractPool, AbstractPool._tree_flatten, AbstractPool._tree_unflatten
)
