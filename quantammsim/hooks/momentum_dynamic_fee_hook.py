from typing import Dict, Any, Optional
import jax.numpy as jnp
from jax.nn import sigmoid, softplus

from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_gradients
)
from quantammsim.hooks.dynamic_fee_base_hook import BaseDynamicFeeHook


class MomentumDynamicFeeHook(BaseDynamicFeeHook):
    """
    MomentumDynamicFeeHook is a class that extends BaseDynamicFeeHook 
    to implement a dynamic fee mechanism based on momentum.

    Methods:
        __init__():
            Initialize a new MomentumDynamicFeeHook instance.

        calculate_dynamic_fees(params: Dict[str, Any], run_fingerprint: 
        Dict[str, Any], prices: jnp.ndarray, start_index: jnp.ndarray, additional_oracle_input: 
        Optional[jnp.ndarray] = None) -> jnp.ndarray:
            Calculate dynamic fees based on price gradients and momentum parameters.

        extend_parameters(base_params: Dict[str, Any], initial_values_dict: Dict[str, Any], 
        n_assets: int, n_parameter_sets: int) -> Dict[str, Any]:
            Extend base parameters with dynamic fee parameters.

        is_trainable() -> bool:
            Check if the momentum fee hook is trainable.
    """

    def __init__(self):
        """
        Initialize a new MomentumDynamicFeeHook instance.
        """
        super().__init__()


    def calculate_dynamic_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Calculate price gradients
        params["momentum_fee_params"] = {}
        params["momentum_fee_params"]["fee_logit"] = jnp.array([10.0])
        params["momentum_fee_params"]["fee_scaling_factor"] = jnp.array([1.0])
        params["momentum_fee_params"]["logit_lamb"] = jnp.array([0.5])
        max_fees = softplus(params["momentum_fee_params"]["fee_logit"]).clip(0.0, 0.3)
        chunkwise_price_values = prices[:: run_fingerprint["chunk_period"]]
        gradients = calc_gradients(
            params["momentum_fee_params"],
            chunkwise_price_values,
            run_fingerprint["chunk_period"],
            run_fingerprint["max_memory_days"],
            False,
            cap_lamb=True,
        )
        dynamic_fees = max_fees * sigmoid(
            params["momentum_fee_params"]["fee_scaling_factor"] * gradients
        )
        return dynamic_fees

    def extend_parameters(
        self,
        base_params: Dict[str, Any],
        initial_values_dict: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int,
    ) -> Dict[str, Any]:
        """Extend base parameters with dynamic fee parameters."""
        base_params["momentum_fee_params"] = {}
        base_params["momentum_fee_params"]["fee_logit"] = jnp.array([10.0])
        base_params["momentum_fee_params"]["fee_scaling_factor"] = jnp.array([1.0])
        base_params["momentum_fee_params"]["logit_lamb"] = jnp.array([0.5])
        return base_params

    def is_trainable(self):
        """Momentum fee hook is trainable."""
        return True
