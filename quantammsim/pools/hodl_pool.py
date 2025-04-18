from typing import Dict, Any, Optional
import numpy as np

# again, this only works on startup!
from jax import config, devices, tree_util
import jax.numpy as jnp
from jax.lax import stop_gradient, dynamic_slice
from jax import default_backend

from quantammsim.pools.base_pool import AbstractPool

config.update("jax_enable_x64", True)

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

class HODLPool(AbstractPool):
    """
    HODLPool is a subclass of AbstractPool that represents a pool with no activity.
    This class provides methods to calculate reserves assuming no trading activity occurs.

    Methods
    -------
    __init__():
        Initializes the HODLPool instance.

    calculate_reserves_with_fees(params, run_fingerprint, prices, start_index, 
    additional_oracle_input=None):
        Calculates the reserves with fees, which in this case is the same as reserves 
        without fees due to no activity.

    calculate_reserves_zero_fees(params, run_fingerprint, prices, start_index, 
    additional_oracle_input=None):
        Calculates the reserves without fees, assuming no trading activity.

    calculate_reserves_with_dynamic_inputs(params, run_fingerprint, prices, start_index, 
    fees_array, arb_thresh_array, arb_fees_array, trade_array, additional_oracle_input=None):
        Calculates the reserves with dynamic inputs, which in this case is 
        the same as reserves without fees due to no activity.

    init_base_parameters(initial_values_dict, run_fingerprint, n_assets, 
        n_parameter_sets=1, noise="gaussian"):
    Initializes the base parameters for the pool, including weights and other initial values.

    calculate_initial_weights(params):
        Calculates the initial weights for the assets in the pool based on the initial weights logits.

    make_vmap_in_axes(params, n_repeats_of_recurred=0):
        Creates a dictionary for vectorized mapping of input axes.

    is_trainable():
        Indicates whether the pool is trainable. Always returns False for HODLPool.
    """

    def __init__(self):
        super().__init__()

    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # hodl means no activity, so reserves are just the initial reserves
        return self.calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )

    def _calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # hodl means no activity, so reserves are just the initial reserves
        weights = self.calculate_initial_weights(params)
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        # calculate the reserves by cumprod of reserve ratios
        reserves = jnp.repeat(initial_reserves[jnp.newaxis, :], bout_length - 1, axis=0)
        return reserves

    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Public interface for zero-fee reserve calculation.

        This method can be safely overridden by hooks (e.g., LVR) while still
        allowing access to the original implementation through the protected
        _calculate_reserves_zero_fees method.
        """
        return self._calculate_reserves_zero_fees(
            params,
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
        )

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
        # hodl means no activity, so reserves are just the initial reserves
        return self.calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )

    def init_base_parameters(
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
                             + " or a matrix of shape (n_parameter_sets, n_assets)"
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
        for k in params.keys():
            if k != "subsidary_params":
                params[k] = params[k][0]
        return params

    def is_trainable(self):
        return False

    def calculate_weights(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Calculate empirical weights for HODL.

        HODL does not have weights in the same way as other pools,
        such as G3M pools or FM-AMM pools. Therefore, the weights are
        calculated based on the reserves that the pool has when run in
        the zero-fees case, and the empirical weights are derived from
        the empirical division of value between reserve over time.
        This method:
        1. Calculates zero-fee reserves
        2. Computes value distribution using prices
        3. Returns normalized weights

        Parameters
        ----------
        params : Dict[str, Any]
            The parameters for the pool.
        run_fingerprint : Dict[str, Any]
            The fingerprint of the current run.
        prices : jnp.ndarray
            The prices of the assets.
        start_index : jnp.ndarray
            The starting index for the prices.
        additional_oracle_input : Optional[jnp.ndarray]
            Additional input from the oracle, if any.

        Returns
        -------
        jnp.ndarray
            The calculated weights for the ECLP pool.
        Notes
        -----
        This method uses the protected _calculate_reserves_zero_fees
        implementation to ensure consistent weight calculation even
        when hooks override the public interface. It is only called
        in the 'versus rebalancing' hooks.

        """
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        reserves = self._calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )
        value = reserves * local_prices
        weights = value / jnp.sum(value, axis=-1, keepdims=True)
        return weights


tree_util.register_pytree_node(
    HODLPool,
    HODLPool._tree_flatten,
    HODLPool._tree_unflatten,
)
