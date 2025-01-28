# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
from jax.lib.xla_bridge import default_backend
from jax import local_device_count, devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import jit, vmap
from jax import device_put
from jax import tree_util
from jax.lax import stop_gradient, dynamic_slice
from jax.nn import softmax

from functools import partial

from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np

from quantammsim.pools.base_pool import AbstractPool

from quantammsim.core_simulator.param_utils import make_vmap_in_axes_dict
from quantammsim.pools.ECLP.gyroscope_reserves import (
    _jax_calc_gyroscope_reserves_with_fees,
    _jax_calc_gyroscope_reserves_zero_fees,
    _jax_calc_gyroscope_reserves_with_dynamic_inputs,
    initialise_gyroscope_reserves_given_value,
)
from quantammsim.pools.ECLP.gyroscope_weight_conversion import (
    optimize_lambda_and_tan_phi,
)


class GyroscopePool(AbstractPool):
    """Elliptical Concentrated Liquidity Pool (ECLP) implementation.
    
    The ECLP is an automated market maker (AMM) design that uses elliptical 
    geometry to define the trading curve. It provides concentrated liquidity
    within a specified price range while maintaining smooth price discovery.
    
    Key Features:
    - Price bounds (alpha, beta) define valid trading range
    - Rotation angle (phi) and scaling factor (lambda) control curve shape
    - Supports fee-based and zero-fee trading
    - Optimizes parameters to achieve target weights
    - Compatible with LVR (Liquidity Value at Risk) hooks
    
    The pool is defined by:
    - Elliptical trading curve rotated by phi
    - Price range [alpha, beta] for valid trades
    - Lambda parameter controlling curve eccentricity
    - Optional trading fees and arbitrage thresholds
    
    The implementation follows the E-CLP paper, using JAX for
    efficient computation of reserves and weights. The pool maintains both
    public interfaces for normal operation and protected implementations for
    use by hooks and internal calculations.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Pool parameters including alpha, beta, lambda, and phi
    run_fingerprint : Dict[str, Any]
        Configuration settings for the simulation run
    prices : jnp.ndarray
        Asset prices over time
    start_index : jnp.ndarray
        Starting indices for price windows
    additional_oracle_input : Optional[jnp.ndarray]
        Additional input data from oracles
    
    Notes
    -----
    - Only supports exactly 2 assets
    - Uses JAX for efficient computation
    - Implements equations from "The Elliptic Concentrated Liquidity Pool" paper
    - Maintains original implementation access for hooks via protected methods
    - Weights are derived empirically from zero-fee reserve calculations
    """
    def __init__(self):
        super().__init__()

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Calculate reserves for ECLP pool including trading fees.
        
        This method computes pool reserves over time considering trading fees and
        arbitrage thresholds. It follows Appendix A of the E-CLP paper, applying
        the calculations at each timestep.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters including:
            - alpha : float
                Lower price bound
            - beta : float
                Upper price bound
            - phi : float
                Rotation angle
            - lam : float
                Scaling factor
        run_fingerprint : Dict[str, Any]
            Run configuration including:
            - fees : float
                Trading fee percentage
            - gas_cost : float
                Arbitrage threshold
            - arb_fees : float
                Additional arbitrage fees
        prices : jnp.ndarray
            Asset prices over time, shape (T, 2)
        start_index : jnp.ndarray
            Starting indices for price windows
        additional_oracle_input : Optional[jnp.ndarray]
            Additional oracle data if needed
        
        Returns
        -------
        jnp.ndarray
            Calculated reserves over time, shape (T, 2)
        
        Notes
        -----
        The implementation handles numeraire ordering internally and
        restores the original order before returning.
        """
        # Gyroscope ECLP pools are only defined for 2 assets
        assert run_fingerprint["n_assets"] == 2
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        lam, phi = self._get_lam_and_phi(params, run_fingerprint, local_prices[0])

        # Handle numeraire ordering for prices
        prices, needs_swap = self._handle_numeraire_ordering(prices, run_fingerprint)

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_reserves = initialise_gyroscope_reserves_given_value(
            initial_pool_value,
            local_prices[0],
            alpha=params["alpha"],
            beta=params["beta"],
            lam=lam,
            sin=jnp.sin(phi),
            cos=jnp.cos(phi),
        )
        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_gyroscope_reserves_with_fees(
                initial_reserves,
                prices=arb_acted_upon_local_prices,
                alpha=params["alpha"],
                beta=params["beta"],
                sin=jnp.sin(phi),
                cos=jnp.cos(phi),
                lam=lam,
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves, arb_acted_upon_local_prices.shape
            )

        # Restore original order if we swapped
        reserves = jnp.where(needs_swap, jnp.flip(reserves, axis=-1), reserves)
        return reserves

    @partial(jit, static_argnums=(2,))
    def _calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Protected implementation for zero-fee reserve calculation.

               This protected method preserves the original implementation for use by:
               1. The public calculate_reserves_zero_fees interface
               2. Internal weight calculations that need the zero-fee behavior
               3. Hooks that need access to the original logic

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters (see calculate_reserves_with_fees)
        run_fingerprint : Dict[str, Any]
            Run configuration (see calculate_reserves_with_fees)
        prices : jnp.ndarray
            Asset prices over time
        start_index : jnp.ndarray
            Starting indices for price windows
        additional_oracle_input : Optional[jnp.ndarray]
            Additional oracle data if needed

        Returns
        -------
        jnp.ndarray
            Calculated reserves over time

        Notes
        -----
        The implementation is jitted for performance, with run_fingerprint
        marked as static to allow JAX to optimize the computation.
        """
        # Gyroscope ECLP pools are only defined for 2 assets
        assert run_fingerprint["n_assets"] == 2
        # Handle numeraire ordering for prices
        prices, needs_swap = self._handle_numeraire_ordering(prices, run_fingerprint)

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        lam, phi = self._get_lam_and_phi(params, run_fingerprint, local_prices[0])

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_reserves = initialise_gyroscope_reserves_given_value(
            initial_pool_value,
            local_prices[0],
            alpha=params["alpha"],
            beta=params["beta"],
            lam=lam,
            sin=jnp.sin(phi),
            cos=jnp.cos(phi),
        )
        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_gyroscope_reserves_zero_fees(
                initial_reserves,
                prices=arb_acted_upon_local_prices,
                alpha=params["alpha"],
                beta=params["beta"],
                sin=jnp.sin(phi),
                cos=jnp.cos(phi),
                lam=lam,
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves, arb_acted_upon_local_prices.shape
            )

        # Restore original order if we swapped
        reserves = jnp.where(needs_swap, jnp.flip(reserves, axis=-1), reserves)
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

    @partial(jit, static_argnums=(2,))
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
        # Gyroscope ECLP pools are only defined for 2 assets
        assert run_fingerprint["n_assets"] == 2
        # Handle numeraire ordering for prices
        prices, needs_swap = self._handle_numeraire_ordering(prices, run_fingerprint)

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        lam, phi = self._get_lam_and_phi(params, run_fingerprint, local_prices[0])

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_reserves = initialise_gyroscope_reserves_given_value(
            initial_pool_value,
            local_prices[0],
            alpha=params["alpha"],
            beta=params["beta"],
            lam=lam,
            sin=jnp.sin(phi),
            cos=jnp.cos(phi),
        )
        # any of fees_array, arb_thresh_array, arb_fees_array, trade_array
        # can be singletons, in which case we repeat them for the length of the bout

        # Determine the maximum leading dimension
        max_len = bout_length - 1
        if run_fingerprint["arb_frequency"] != 1:
            max_len = max_len // run_fingerprint["arb_frequency"]

        # Handle trade array reordering if needed
        if run_fingerprint["do_trades"]:
            # if we are doing trades, the trades array must be of the same length as the other arrays
            assert trade_array.shape[0] == max_len
            if needs_swap:
                # Swap trade indices (0->1, 1->0) but keep amounts unchanged
                trade_array = trade_array.at[:, :2].set(1 - trade_array[:, :2])

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

        # Calculate reserves
        reserves = _jax_calc_gyroscope_reserves_with_dynamic_inputs(
            initial_reserves,
            prices=arb_acted_upon_local_prices,
            alpha=params["alpha"],
            beta=params["beta"],
            sin=jnp.sin(phi),
            cos=jnp.cos(phi),
            lam=lam,
            fees=fees_array_broadcast,
            arb_thresh=arb_thresh_array_broadcast,
            arb_fees=arb_fees_array_broadcast,
            trades=trade_array,
            do_trades=run_fingerprint["do_trades"],
        )
        # Restore original order if we swapped
        reserves = jnp.where(needs_swap, jnp.flip(reserves, axis=-1), reserves)
        return reserves

    def _init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        """
        Initialize parameters for an ECLP pool.

        ECLP pools have four base parameters:
        - rotation angle Phi: Controls the rotation of the ellipse
        - scaling factor Lambda: Controls the eccentricity of the ellipse
        - Lower price bound alpha: Minimum price ratio between assets
        - Upper price bound beta: Maximum price ratio between assets

        Parameters
        ----------
        initial_values_dict : Dict[str, Any]
            Dictionary containing initial values for the parameters
        run_fingerprint : Dict[str, Any]
            Dictionary containing run configuration settings
        n_assets : int
            Number of assets in the pool (must be 2 for ECLP)
        n_parameter_sets : int, optional
            Number of parameter sets to initialize, by default 1
        noise : str, optional
            Type of noise to apply during initialization, by default "gaussian"

        Returns
        -------
        Dict[str, Any]
            Dictionary containing initialized parameters:
            - phi: Rotation angle
            - lambda: Scaling factor
            - alpha: Lower price bound
            - beta: Upper price bound

        Raises
        ------
        ValueError
            If n_assets is not 2 or if required initial values are missing
        """

        # We need to initialise the weights for each parameter set
        # If a vector is provided in the inital values dict, we use
        # that, if only a singleton array is provided we expand it
        # to n_assets and use that vlaue for all assets.
        def process_initial_values(initial_values_dict, key, n_parameter_sets):
            if key in initial_values_dict:
                initial_value = initial_values_dict[key]
                if isinstance(initial_value, (np.ndarray, jnp.ndarray, list)):
                    initial_value = np.array(initial_value)
                    if initial_value.size == 1:
                        return np.array([initial_value] * n_parameter_sets)
                    elif initial_value.shape == (n_parameter_sets,):
                        return initial_value
                    else:
                        raise ValueError(
                            f"{key} must be a singleton or a vector of length n_parameter_sets"
                        )
                else:
                    return np.array([initial_value] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        phi = process_initial_values(
            initial_values_dict, "rotation_angle", n_parameter_sets
        )
        alpha = process_initial_values(initial_values_dict, "alpha", n_parameter_sets)
        beta = process_initial_values(initial_values_dict, "beta", n_parameter_sets)
        lam = process_initial_values(initial_values_dict, "lam", n_parameter_sets)

        params = {
            "phi": phi,
            "alpha": alpha,
            "beta": beta,
            "lam": lam,
            "subsidary_params": [],
        }

        params = self.add_noise(params, noise, n_parameter_sets)
        return params

    def calculate_weights(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        return jnp.array([0.5, 0.5])

    def make_vmap_in_axes(self, params: Dict[str, Any], n_repeats_of_recurred: int = 0):
        return make_vmap_in_axes_dict(params, 0, [], [], n_repeats_of_recurred)

    def is_trainable(self):
        return False

    def weights_needs_original_methods(self) -> bool:
        """ECLP pools need original methods for weight calculation.
        
        Returns
        -------
        bool
            True - ECLP weight calculation requires original pool methods.

        Notes
        -----
        This is because the weights are calculated based on the reserves that the pool has when run in
        the zero-fees case, and the empirical weights are derived from the empirical division of value between reserve over time.
        This also means that we need to preserve the original reserve calculation method in the original pool class as a
        classmethod.
        """
        return True

    def calculate_weights(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Calculate empirical weights for ECLP pool.

        ECLP pools do not have weights in the same way as other pools,
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

    @partial(jit, static_argnums=(2,))
    def _handle_numeraire_ordering(
        self,
        prices: jnp.ndarray,
        run_fingerprint: Dict[str, Any],
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], bool]:
        """Reorders prices and reserves to ensure numeraire token is in second position for ECLP calculations.

        In ECLP pools, the order of assets matters because parameters (alpha, beta) define the price
        range of the first asset in terms of the second (numeraire) asset.

        Parameters
        ----------
        prices : jnp.ndarray
            - Array of prices with shape (..., n_assets) where n_assets=2
            - For multi-timestep inputs, leading dimensions can vary
        reserves : jnp.ndarray, optional
            - Array of reserves with shape (..., n_assets)
            - Can be None if only reordering prices
        run_fingerprint : Dict[str, Any]
            - Must contain "token_list" (List[str]) and "numeraire" (str)
            - token_list assumed to be in alphabetical order

        Returns
        -------
        tuple
            - jnp.ndarray: Reordered prices (same shape as input)
            - jnp.ndarray or None: Reordered reserves if provided
            - bool: Whether a swap was performed (True if numeraire was in first position)

        Raises
        ------
        KeyError
            If required keys missing from run_fingerprint
        ValueError
            If prices or reserves last dimension != 2

        Notes
        -----
        The swap operation uses jnp.flip along the last axis and is transparent to core ECLP
        calculations which assume numeraire is always in second position. The function assumes
        tokens in token_list are in alphabetical order. The swap operation uses jnp.flip along
        the last axis for efficiency. This reordering is transparent to the core ECLP calculations
        which assume the numeraire is always in the second position.
        """
        # Get token tickers in current order
        tokens = sorted(run_fingerprint["tokens"])
        numeraire = run_fingerprint["numeraire"]
        if numeraire is None:
            numeraire = tokens[-1]

        # Check if numeraire is already in second position
        needs_swap = tokens.index(numeraire) == 0

        if needs_swap:
            # Swap the order of prices and reserves
            prices = prices[..., ::-1]

        return prices, needs_swap

    @partial(jit, static_argnums=(2,))
    def _get_lam_and_phi(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        initial_prices: jnp.ndarray,
    ) -> jnp.ndarray:
        # We assume that the prices are in the correct order for ECLP calculations
        # so we don't need to swap them using _handle_numeraire_ordering
        # BUT, we do need to know if we need to swap them for the weight conversion
        # so we pass in the initial prices to _handle_numeraire_ordering
        # and then use the result to determine if we need to swap
        _, needs_swap = self._handle_numeraire_ordering(initial_prices, run_fingerprint)

        if "initial_weights_logits" in params:
            # Calculate target weight from logits
            weights = stop_gradient(softmax(params["initial_weights_logits"]))
            target_weight = jnp.where(needs_swap, weights[1], weights[0])

            # Optimize lambda and phi to match target weight
            lam, tan_phi = optimize_lambda_and_tan_phi(
                target_weight=target_weight,
                initial_pool_value=run_fingerprint["initial_pool_value"],
                initial_prices=initial_prices,
                alpha=params["alpha"],
                beta=params["beta"],
            )
            phi = jnp.arctan(tan_phi)
        else:
            lam = params["lam"]
            phi = params["phi"]

        return lam, phi

    @classmethod
    def process_parameters(cls, update_rule_parameters, n_assets):
        """Process gyroscope pool parameters from web interface input."""
        result = {}
        # Process any remaining parameters in a default way
        for urp in update_rule_parameters:
            result[urp.name] = np.squeeze(np.array(urp.value))
        return result


tree_util.register_pytree_node(
    GyroscopePool,
    GyroscopePool._tree_flatten,
    GyroscopePool._tree_unflatten,
)
