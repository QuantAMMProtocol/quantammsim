from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import jax.numpy as jnp


class BaseDynamicFeeHook(ABC):
    """Mixin class to add dynamic fee calculation capabilities to pools.

    Parameters
    ----------
    params_init_dict
        Dict that gives the names of the parameters needed by the hook and their initialisatio method

    Attributes
    ----------
    None

    Methods
    -------
    calculate_dynamic_fees(params, run_fingerprint, prices, start_index, additional_oracle_input)
        Abstract method that must be implemented to define fee calculation logic
    calculate_reserves_with_fees(params, run_fingerprint, prices, start_index, additional_oracle_input)
        Combines dynamic fee calculation with reserve updates

    Notes
    -----
    This mixin provides functionality for pools to dynamically adjust their fees based on market conditions,
    oracle inputs, and other parameters. It is designed to be used with AMM (Automated Market Maker) pool
    implementations.

    The mixin assumes the existence of calculate_reserves_with_dynamic_inputs() in the pool class it will 
    be used with. Fee calculations should only use current and historical data to avoid look-ahead bias.
    All calculations should be vectorized (i.e. use jax.vmap) where possible for performance.

    Features:
    - Supports time-varying fees based on market conditions
    - Integrates with existing pool infrastructure for reserve calculations  
    - Handles gas costs and arbitrage fees

    Examples
    --------
    >>> class MyPool(BasePool, DynamicFeesMixin):
    ...     def calculate_dynamic_fees(self, params, run_fingerprint, prices, start_index):
    ...         # Custom fee calculation logic here
    ...         return computed_fees
    """
    def __init__(self, params_init_dict):
        """
        Initialize a new MomentumPool instance.

        Parameters
        ----------
        None
        """
        super().__init__()

    @abstractmethod
    def calculate_dynamic_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Calculate dynamic fees based on price/oracle input.

        Take care when implementing this method to ensure that there is no look ahead bias.
        """
        pass

    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        # Calculate dynamic fees based on price/oracle input
        dynamic_fees = self.calculate_dynamic_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )

        # Use existing dynamic inputs infrastructure
        return self.calculate_reserves_with_dynamic_inputs(
            params,
            run_fingerprint,
            prices,
            start_index,
            dynamic_fees,
            run_fingerprint["gas_cost"],
            run_fingerprint["arb_fees"],
            jnp.zeros((dynamic_fees.shape[0], run_fingerprint["n_assets"])),
            additional_oracle_input,
        )
