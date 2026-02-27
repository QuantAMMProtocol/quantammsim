"""reClAMM pool implementation.

Rebalancing Concentrated Liquidity AMM — a 2-token constant-product pool
with dynamic virtual reserves that track market price. Extends AbstractPool
following the GyroscopePool pattern (scan-based, not trainable).
"""

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, tree_util
from jax.lax import dynamic_slice
from functools import partial

from typing import Dict, Any, Optional
import numpy as np

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.reClAMM.reclamm_reserves import (
    initialise_reclamm_reserves,
    _jax_calc_reclamm_reserves_zero_fees,
    _jax_calc_reclamm_reserves_with_fees,
    _jax_calc_reclamm_reserves_with_dynamic_inputs,
)


class ReClammPool(AbstractPool):
    """Rebalancing Concentrated Liquidity AMM pool.

    A 2-token constant-product AMM with dynamic virtual reserves that track
    market price. The invariant is L = (Ra + Va) * (Rb + Vb), equivalent to
    standard xy=k on effective reserves (real + virtual).

    Virtual balances evolve over time (path-dependent) when the pool drifts
    outside its target price range, making this inherently scan-based.

    Parameters
    ----------
    price_ratio : float
        Desired max_price / min_price for the pool's price range.
    centeredness_margin : float
        Threshold [0, 1] below which virtual balance updates are triggered.
    daily_price_shift_base : float
        Decay rate for virtual balance updates, typically 1 - 1/124000.

    Notes
    -----
    Not trainable — parameters define pool geometry, not a learned strategy.
    Weights are empirical (derived from reserves * prices / total value).
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
        assert run_fingerprint["n_assets"] == 2

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            arb_prices = local_prices[:: run_fingerprint["arb_frequency"]]
        else:
            arb_prices = local_prices

        price_ratio = params["price_ratio"]
        centeredness_margin = params["centeredness_margin"]
        daily_price_shift_base = params["daily_price_shift_base"]

        initial_pool_value = run_fingerprint["initial_pool_value"]
        seconds_per_step = run_fingerprint["arb_frequency"] * 60.0

        initial_reserves, Va, Vb = initialise_reclamm_reserves(
            initial_pool_value, local_prices[0], price_ratio
        )

        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_reclamm_reserves_with_fees(
                initial_reserves, Va, Vb,
                arb_prices,
                centeredness_margin,
                daily_price_shift_base,
                seconds_per_step,
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
            )
        else:
            reserves = jnp.broadcast_to(initial_reserves, arb_prices.shape)

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
        """Protected zero-fee implementation for hooks and weight calculation."""
        assert run_fingerprint["n_assets"] == 2

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            arb_prices = local_prices[:: run_fingerprint["arb_frequency"]]
        else:
            arb_prices = local_prices

        price_ratio = params["price_ratio"]
        centeredness_margin = params["centeredness_margin"]
        daily_price_shift_base = params["daily_price_shift_base"]

        initial_pool_value = run_fingerprint["initial_pool_value"]
        seconds_per_step = run_fingerprint["arb_frequency"] * 60.0

        initial_reserves, Va, Vb = initialise_reclamm_reserves(
            initial_pool_value, local_prices[0], price_ratio
        )

        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_reclamm_reserves_zero_fees(
                initial_reserves, Va, Vb,
                arb_prices,
                centeredness_margin,
                daily_price_shift_base,
                seconds_per_step,
            )
        else:
            reserves = jnp.broadcast_to(initial_reserves, arb_prices.shape)

        return reserves

    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
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
        lp_supply_array: jnp.ndarray = None,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        assert run_fingerprint["n_assets"] == 2

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            arb_prices = local_prices[:: run_fingerprint["arb_frequency"]]
        else:
            arb_prices = local_prices

        price_ratio = params["price_ratio"]
        centeredness_margin = params["centeredness_margin"]
        daily_price_shift_base = params["daily_price_shift_base"]

        initial_pool_value = run_fingerprint["initial_pool_value"]
        seconds_per_step = run_fingerprint["arb_frequency"] * 60.0

        initial_reserves, Va, Vb = initialise_reclamm_reserves(
            initial_pool_value, local_prices[0], price_ratio
        )

        max_len = bout_length - 1
        if run_fingerprint["arb_frequency"] != 1:
            max_len = max_len // run_fingerprint["arb_frequency"]

        fees_array_broadcast = jnp.broadcast_to(
            fees_array, (max_len,) + fees_array.shape[1:]
        )
        arb_thresh_array_broadcast = jnp.broadcast_to(
            arb_thresh_array, (max_len,) + arb_thresh_array.shape[1:]
        )
        arb_fees_array_broadcast = jnp.broadcast_to(
            arb_fees_array, (max_len,) + arb_fees_array.shape[1:]
        )

        reserves = _jax_calc_reclamm_reserves_with_dynamic_inputs(
            initial_reserves, Va, Vb,
            arb_prices,
            centeredness_margin,
            daily_price_shift_base,
            seconds_per_step,
            fees=fees_array_broadcast,
            arb_thresh=arb_thresh_array_broadcast,
            arb_fees=arb_fees_array_broadcast,
            all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
        )
        return reserves

    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        """Initialize reClAMM pool parameters.

        Required keys in initial_values_dict:
        - price_ratio: max_price / min_price
        - centeredness_margin: threshold for virtual balance updates
        - daily_price_shift_base: decay rate for virtual balances
        """
        def process(key, default=None):
            if key in initial_values_dict:
                val = initial_values_dict[key]
                if isinstance(val, (np.ndarray, jnp.ndarray, list)):
                    val = np.array(val)
                    if val.size == 1:
                        return np.array([float(val)] * n_parameter_sets)
                    elif val.shape == (n_parameter_sets,):
                        return val
                    else:
                        raise ValueError(f"{key} shape mismatch")
                else:
                    return np.array([float(val)] * n_parameter_sets)
            elif default is not None:
                return np.array([default] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        params = {
            "price_ratio": process("price_ratio", 4.0),
            "centeredness_margin": process("centeredness_margin", 0.2),
            "daily_price_shift_base": process(
                "daily_price_shift_base", 1.0 - 1.0 / 124000.0
            ),
            "subsidary_params": [],
        }

        # No noise for non-trainable params, but keep interface consistent
        params = self.add_noise(params, noise, n_parameter_sets)
        return params

    def is_trainable(self):
        return False

    def weights_needs_original_methods(self) -> bool:
        return True

    def calculate_weights(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Calculate empirical weights from zero-fee reserves.

        Same pattern as GyroscopePool: weights = value_per_asset / total_value.
        """
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            local_prices = local_prices[:: run_fingerprint["arb_frequency"]]

        reserves = self._calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )
        value = reserves * local_prices
        weights = value / jnp.sum(value, axis=-1, keepdims=True)
        return weights


tree_util.register_pytree_node(
    ReClammPool,
    ReClammPool._tree_flatten,
    ReClammPool._tree_unflatten,
)
