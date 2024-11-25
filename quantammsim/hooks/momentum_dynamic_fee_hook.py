from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import jax.numpy as jnp
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import calc_gradients, calc_k
from quantammsim.hooks.dynamic_fee_base_hook import BaseDynamicFeeHook
from jax.nn import sigmoid, softplus


class MomentumDynamicFeeHook(BaseDynamicFeeHook):

    def __init__(self):
        """
        Initialize a new TFMMBasePool instance.
        """
        super().__init__()

    def __init__(self):
        BaseDynamicFeeHook.__init__(self)

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
        dynamic_fees = max_fees * sigmoid(params["momentum_fee_params"]["fee_scaling_factor"] * gradients)
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
