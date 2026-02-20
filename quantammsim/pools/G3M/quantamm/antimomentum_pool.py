"""Contrarian (anti-momentum) pool for QuantAMM.

Extends :class:`MomentumPool` by negating the momentum sensitivity factor ``k``,
producing a mean-reversion strategy that overweights recently declining assets
and underweights recently appreciating ones. Shares all parameters and EWMA
estimator infrastructure with the momentum pool.
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

from quantammsim.pools.G3M.quantamm.momentum_pool import (
    MomentumPool,
    _jax_momentum_weight_update,
)
from quantammsim.core_simulator.param_utils import (
    lamb_to_memory_days_clipped,
    calc_lamb,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import calc_gradients, calc_k

from typing import Dict, Any, Optional
from functools import partial


# import the fine weight output function which has pre-set argument rule_outputs_are_themselves_weights
# as this is False for momentum pools --- the strategy outputs weight _changes_

class AntiMomentumPool(MomentumPool):
    """
    A class for anti-momentum strategies run as TFMM liquidity pools.

    This class implements a mean-reversion-based strategy for asset allocation within a TFMM framework.
    It uses price data to generate mean-reversion signals, which are then translated into weight adjustments.

    Parameters
    ----------
    None

    Methods
    -------
    calculate_rule_outputs(params, run_fingerprint, prices, additional_oracle_input)
        Calculate the raw weight outputs based on mean-reversion signals.

    Notes
    -----
    The class implements a mean-reversion-based strategy for asset allocation within a TFMM framework.
    It uses price data to generate mean-reversion signals, which are then translated into weight adjustments.
    The class provides methods to calculate raw weight outputs based on these signals and refine them
    into final asset weights, taking into account various parameters and constraints defined in the pool setup.
    """

    # AntiMomentum uses same params as Momentum (just negates k internally)
    # Inherits PARAM_SCHEMA from MomentumPool

    def __init__(self):
        """
        Initialize a new AntiMomentumPool instance.

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
        Calculate the raw weight outputs based on antimomentum signals.
        This method computes the raw weight adjustments for the antimomentum strategy. It processes
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
        5. Applies the antimomentum weight update (momentum with negative k) formula to get raw weight outputs.

        The raw weight outputs are not the final weights, but rather the changes
        to be applied to the previous weights. These will be refined in subsequent steps.
        """
        memory_days = lamb_to_memory_days_clipped(
            calc_lamb(params), run_fingerprint["chunk_period"], run_fingerprint["max_memory_days"]
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
        rule_outputs = _jax_momentum_weight_update(gradients, -k)
        return rule_outputs

tree_util.register_pytree_node(
    AntiMomentumPool, AntiMomentumPool._tree_flatten, AntiMomentumPool._tree_unflatten
)
