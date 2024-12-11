from typing import Dict, Any, Optional
from functools import partial
import numpy as np

# again, this only works on startup!
from jax import config, devices, jit, tree_util
from jax.lib.xla_bridge import default_backend
import jax.numpy as jnp
from jax.lax import stop_gradient, dynamic_slice
from jax.nn import softmax

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.G3M.balancer.balancer_reserves import (
    _jax_calc_balancer_reserve_ratios,
    _jax_calc_balancer_reserves_with_fees_using_precalcs,
    _jax_calc_balancer_reserves_with_dynamic_inputs,
)
from quantammsim.core_simulator.param_utils import make_vmap_in_axes_dict

config.update("jax_enable_x64", True)

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")


class BalancerPool(AbstractPool):
    def __init__(self):
        super().__init__()

    @partial(jit, static_argnums=2)
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        weights = self.calculate_weights(params)
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        reserves = _jax_calc_balancer_reserves_with_fees_using_precalcs(
            initial_reserves,
            weights,
            arb_acted_upon_local_prices,
            fees=run_fingerprint["fees"],
            arb_thresh=run_fingerprint["gas_cost"],
            arb_fees=run_fingerprint["arb_fees"],
            all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
        )
        return reserves

    @partial(jit, static_argnums=2)
    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        weights = self.calculate_weights(params)
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        reserve_ratios = _jax_calc_balancer_reserve_ratios(
            arb_acted_upon_local_prices[:-1],
            weights,
            arb_acted_upon_local_prices[1:],
        )

        # calculate the reserves by cumprod of reserve ratios
        reserves = jnp.vstack(
            [
                initial_reserves,
                initial_reserves * jnp.cumprod(reserve_ratios, axis=0),
            ]
        )
        return reserves

    @partial(jit, static_argnums=2)
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
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        weights = self.calculate_weights(params)

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]

        # any of fees_array, arb_thresh_array, arb_fees_array, trade_array
        # can be singletons, in which case we repeat them for the length of the bout

        # Determine the maximum leading dimension
        max_len = bout_length - 1
        if run_fingerprint["arb_frequency"] != 1:
            max_len = max_len // run_fingerprint["arb_frequency"]
        # Broadcast input arrays to match the maximum leading dimension.
        # If they are singletons, this will just repeat them for the length of the bout.
        # If they are arrays of length bout_length, this will cause no change.
        fees_array_broadcast = jnp.broadcast_to(
            fees_array, (max_len,) + fees_array.shape[1:]
        )
        arb_thresh_array_broadcast = jnp.broadcast_to(
            arb_thresh_array, (max_len,) + arb_thresh_array.shape[1:]
        )
        arb_fees_array_broadcast = jnp.broadcast_to(
            arb_fees_array, (max_len,) + arb_fees_array.shape[1:]
        )
        # if we are doing trades, the trades array must be of the same length as the other arrays
        if run_fingerprint["do_trades"]:
            assert trade_array.shape[0] == max_len
        reserves = _jax_calc_balancer_reserves_with_dynamic_inputs(
            initial_reserves,
            weights,
            arb_acted_upon_local_prices,
            fees_array_broadcast,
            arb_thresh_array_broadcast,
            arb_fees_array_broadcast,
            jnp.array(run_fingerprint["all_sig_variations"]),
            trade_array,
            run_fingerprint["do_trades"],
        )
        return reserves

    def _init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
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
                            f"{key} must be a singleton or a vector of length n_assets"
                             +  "or a matrix of shape (n_parameter_sets, n_assets)"
                        )
                else:
                    return np.array([[initial_value] * n_assets] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        initial_weights_logits = process_initial_values(
            initial_values_dict, "initial_weights_logits", n_assets, n_parameter_sets
        )
        params = {
            "initial_weights_logits": initial_weights_logits,
        }
        params = self.add_noise(params, noise, n_parameter_sets)
        return params

    def calculate_weights(
        self, params: Dict[str, jnp.ndarray], *args, **kwargs
    ) -> jnp.ndarray:
        initial_weights_logits = params.get("initial_weights_logits")
        # we dont't want to change the weights during any training
        # so wrap them in a stop_grad
        weights = softmax(stop_gradient(initial_weights_logits))
        return weights

    def make_vmap_in_axes(self, params: Dict[str, Any], n_repeats_of_recurred: int = 0):
        return make_vmap_in_axes_dict(params, 0, [], [], n_repeats_of_recurred)

    def is_trainable(self):
        return False


tree_util.register_pytree_node(
    BalancerPool,
    BalancerPool._tree_flatten,
    BalancerPool._tree_unflatten,
)
