"""Market-capitalisation weighted index pool for QuantAMM.

Computes pool weights proportional to each asset's market capitalisation,
loaded from historical data via the data pipeline. This is the base index
strategy: weights are derived directly from market-cap ratios (optionally
softmax-normalised) and output as absolute weights rather than weight changes.
Serves as the parent class for :class:`HodlingIndexPool` and
:class:`TradHodlingIndexPool`.
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
from jax.lax import stop_gradient, while_loop

from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
)

from typing import Dict, Any, Optional
from functools import partial
import numpy as np



# import the fine weight output function which has pre-set argument rule_outputs_are_themselves_weights
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output_from_weights,
)
from quantammsim.utils.data_processing.historic_data_utils import get_data_dict


class IndexMarketCapPool(TFMMBasePool):
    """
    A class for an index strategy run as TFMM (Temporal Function Market Making) liquidity pools,
    extending the TFMMBasePool class.

    This class implements a market cap-based strategy for asset allocation within a TFMM framework.
    It uses price data to generate market cap signals, which are then translated into weight adjustments.

    Parameters
    ----------
    None

    Methods
    -------
    calculate_rule_outputs(params, run_fingerprint, prices, additional_oracle_input)
        Calculate the raw weight outputs based on market cap signals.
    calculate_fine_weights(rule_output, initial_weights, run_fingerprint, params)
        Refine the raw weight outputs to produce final weights.
    calculate_weights(params, run_fingerprint, prices, additional_oracle_input)
        Orchestrate the weight calculation process.

    Notes
    -----
    The class provides methods to calculate raw weight outputs based on market cap signals and refine them
    into final asset weights, taking into account various parameters and constraints defined in the pool setup.
    """

    def __init__(self):
        """
        Initialize a new IndexMarketCapPool instance.

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

        # Calculate the proportional weights of the market caps
        # return self.calculate_rule_outputs_nojit(params, run_fingerprint, prices, additional_oracle_input)
        chunkwise_price_values = prices[:: run_fingerprint["chunk_period"]]
        sorted_tokens = sorted(run_fingerprint["tokens"])

        # # Match circulating supply values for tokens in run_fingerprint
        # # JAX array version
        # circulating_supply = jnp.array(
        #     [
        #         ALL_TOKENS_CIRCULATING_SUPPLY[ALL_TOKENS_SYMBOLS.index(token)]
        #         for token in sorted_tokens
        #     ]
        # )
        all_tokens = [run_fingerprint["tokens"]]
        all_tokens = [item for sublist in all_tokens for item in sublist]
        unique_tokens = list(set(all_tokens))
        unique_tokens.sort()

        data_dict = get_data_dict(
            unique_tokens,
            run_fingerprint,
            data_kind="historic",
            root=None,
            max_memory_days=run_fingerprint["max_memory_days"],
            start_date_string=run_fingerprint["startDateString"],
            end_time_string=run_fingerprint["endDateString"],
            start_time_test_string=run_fingerprint["endDateString"],
            end_time_test_string=run_fingerprint["endTestDateString"],
            max_mc_version=None,
            return_slippage=False,
            return_supply=True,
        )

        supply_data = data_dict["supply"][:len(prices)]
        # # Pure Python version for comparison
        # python_circulating_supply = []
        # for token in sorted_tokens:
        #     token_index = ALL_TOKENS_SYMBOLS.index(token)
        #     supply = ALL_TOKENS_CIRCULATING_SUPPLY[token_index]
        #     python_circulating_supply.append(supply)
        chunkwise_supply_values = supply_data[::run_fingerprint["chunk_period"]]

        market_cap = chunkwise_price_values * chunkwise_supply_values
        total_market_cap = jnp.sum(market_cap, axis=-1, keepdims=True)
        rule_outputs = market_cap / total_market_cap

        if run_fingerprint.get("cap_weights") is not None:
            rule_outputs = self._cap_and_redistribute_weights(
                rule_outputs, run_fingerprint["cap_weights"]
            )

        return rule_outputs

    @partial(jit, static_argnums=(0))
    def _cap_and_redistribute_weights(
        self, weights: jnp.ndarray, max_weight: float = 0.25
    ) -> jnp.ndarray:
        """
        Cap weights at max_weight and redistribute excess proportionally.
        Iterates until no weights exceed the cap or max iterations reached.
        """

        def _redistribute_step(carry):
            weights, iteration = carry

            # Identify which weights exceed the cap
            excess_mask = weights > max_weight
            # Calculate how much excess weight needs to be redistributed
            excess = jnp.sum(
                jnp.where(excess_mask, weights - max_weight, 0.0),
                axis=-1,
                keepdims=True,
            )
            # Set exceeding weights to max
            capped = jnp.where(excess_mask, max_weight, weights)
            # Calculate redistribution weights for uncapped tokens
            uncapped_sum = jnp.sum(
                jnp.where(~excess_mask, capped, 0.0), axis=-1, keepdims=True
            )
            redistribution_weights = jnp.where(
                ~excess_mask, capped / (uncapped_sum + 1e-10), 0.0
            )
            # Redistribute excess
            redistributed = capped + redistribution_weights * excess

            return (redistributed, iteration + 1)

        def _cond_fn(carry):
            weights, iteration = carry
            # Continue if any weight exceeds cap and we haven't hit max iterations
            return jnp.any(weights > max_weight + 1e-6) & (iteration < 10)

        # Initialize
        init_carry = (weights, 0)

        # Run redistribution loop
        final_carry = while_loop(_cond_fn, _redistribute_step, init_carry)
        final_weights = final_carry[0]

        # Ensure weights sum to 1 (handle numerical precision issues)
        final_weights = final_weights / jnp.sum(final_weights, axis=-1, keepdims=True)

        return final_weights

    def calculate_rule_outputs_nojit(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        # Calculate the proportional weights of the market caps
        chunkwise_price_values = prices[:: run_fingerprint["chunk_period"]]
        market_cap = chunkwise_price_values
        total_market_cap = jnp.sum(market_cap, axis=-1, keepdims=True)
        rule_outputs = market_cap / total_market_cap

        return rule_outputs

    @partial(jit, static_argnums=(3))
    def calculate_fine_weights(
        self,
        rule_output: jnp.ndarray,
        initial_weights: jnp.ndarray,
        run_fingerprint: Dict[str, Any],
        params: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Refine raw weight outputs to produce final weights for the index pool.

        This method takes the raw weight outputs calculated from index mcap signals and refines
        them into final asset weights. It applies various constraints and adjustments defined
        in the pool parameters and run fingerprint.

        Parameters
        ----------
        rule_output : jnp.ndarray
            Raw weight changes or outputs from market cap calculations.
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
        Uses the `calc_fine_weight_output_from_weights` function to perform the actual
        refinement. The implementation of this function should handle details such as weight
        interpolation, maximum change limits, and ensuring weights sum to 1.
        """
        if (
            params.get("logit_delta_lamb") is None
            and params.get("logit_lamb") is not None
        ):
            params["logit_delta_lamb"] = jnp.zeros_like(params["logit_lamb"])
        return calc_fine_weight_output_from_weights(
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
        Initialize parameters for the index pool.

        This method sets up the initial parameters for the index pool strategy, including
        weights and memory length (lambda).

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
            Dictionary containing the initialized parameters for the index pool.

        Raises
        ------
        ValueError
            If required initial values are missing or in an incorrect format.

        Notes
        -----
        This method handles the initialization of parameters for initial weights, lambda
        (memory length parameter) for each asset and parameter set.
        It processes the initial values to ensure they are in the correct format and applies
        any necessary transformations (e.g., logit transformations for lambda).
        """
        np.random.seed(0)

        # We need to initialise the weights for each parameter set
        # If a vector is provided in the inital values dict, we use
        # that, if only a singleton array is provided we expand it
        # to n_assets and use that vlaue for all assets.
        def process_initial_values(
            initial_values_dict, key, n_assets, n_parameter_sets
        ):
            if key in initial_values_dict:
                initial_value = initial_values_dict[key]
                if isinstance(initial_value, (np.ndarray, jnp.ndarray, list)):
                    initial_value = np.array(initial_value)
                    if initial_value.size == n_assets:
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
                    return np.array([[initial_value] * n_assets] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        initial_lamb = memory_days_to_lamb(
            initial_values_dict["initial_memory_length"],
            run_fingerprint["chunk_period"],
        )

        logit_lamb_np = np.log(initial_lamb / (1.0 - initial_lamb))
        logit_lamb = np.array([[logit_lamb_np] * n_assets] * n_parameter_sets)

        initial_weights_logits = process_initial_values(
            initial_values_dict, "initial_weights_logits", n_assets, n_parameter_sets
        )

        logit_delta_lamb = stop_gradient(jnp.zeros_like(logit_lamb))

        params = {
            "logit_lamb": logit_lamb,
            "logit_delta_lamb": logit_delta_lamb,
            "initial_weights_logits": initial_weights_logits,
            "subsidary_params": [],
        }

        params = self.add_noise(params, noise, n_parameter_sets)
        return params


tree_util.register_pytree_node(
    IndexMarketCapPool,
    IndexMarketCapPool._tree_flatten,
    IndexMarketCapPool._tree_unflatten,
)
