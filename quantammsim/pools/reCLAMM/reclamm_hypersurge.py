from functools import partial
from typing import Any, Dict, Optional

import numpy as np

import jax.numpy as jnp
from jax import jit, tree_util
from jax.lax import dynamic_slice

from quantammsim.core_simulator.dynamic_inputs import materialize_dynamic_inputs
from quantammsim.pools.hypersurge_utils import (
    HYPERSURGE_PARAM_KEYS,
    hypersurge_params_from_params,
    run_fingerprint_hypersurge_defaults,
)
from quantammsim.pools.reCLAMM.reclamm import ReClammPool
from quantammsim.pools.reCLAMM.reclamm_hypersurge_reserves import (
    _jax_calc_reclamm_hypersurge_reserves,
    _jax_calc_reclamm_hypersurge_reserves_and_fee_revenue,
)


class ReClammHyperSurgePool(ReClammPool):
    """reCLAMM pool with HyperSurge state-dependent swap fees."""

    @staticmethod
    def _run_fingerprint_hypersurge_defaults(run_fingerprint: Dict[str, Any]):
        return run_fingerprint_hypersurge_defaults(run_fingerprint)

    def _hypersurge_params(self, params: Dict[str, Any], run_fingerprint: Dict[str, Any]):
        return hypersurge_params_from_params(params, run_fingerprint)

    def _oracle_price_window(
        self,
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray],
    ):
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        if additional_oracle_input is None:
            local_oracle_prices = dynamic_slice(
                prices, start_index, (bout_length - 1, n_assets)
            )
        else:
            local_oracle_prices = dynamic_slice(
                additional_oracle_input,
                start_index,
                (bout_length - 1, n_assets),
            )

        arb_frequency = run_fingerprint["arb_frequency"]
        if arb_frequency != 1:
            return local_oracle_prices[::arb_frequency]
        return local_oracle_prices

    def _run_hypersurge_reserves(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        fees,
        gas_cost,
        arb_fees,
        lp_supply_array,
        price_ratio_updates,
        oracle_prices,
        lp_supply_already_prepared: bool = False,
        return_fee_revenue: bool = False,
    ):
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)
        ste_temperature = self._resolve_ste_temperature(run_fingerprint)
        noise_lp_supply = None if lp_supply_already_prepared else lp_supply_array
        lp_prepared, noise_model, noise_params, noise_arrays = self._resolve_noise_inputs(
            run_fingerprint,
            prices,
            start_index,
            s.arb_prices.shape[0],
            lp_supply_array=noise_lp_supply,
        )
        if lp_supply_already_prepared:
            lp_prepared = lp_supply_array

        if not run_fingerprint["do_arb"]:
            reserves = jnp.broadcast_to(s.initial_reserves, s.arb_prices.shape)
            if return_fee_revenue:
                return reserves, jnp.zeros(s.arb_prices.shape[0], dtype=s.arb_prices.dtype)
            return reserves

        kernel = (
            _jax_calc_reclamm_hypersurge_reserves_and_fee_revenue
            if return_fee_revenue
            else _jax_calc_reclamm_hypersurge_reserves
        )
        return kernel(
            s.initial_reserves,
            s.Va,
            s.Vb,
            s.arb_prices,
            oracle_prices,
            self._hypersurge_params(params, run_fingerprint),
            s.centeredness_margin,
            s.daily_price_shift_base,
            s.seconds_per_step,
            fees=fees,
            arb_thresh=gas_cost,
            arb_fees=arb_fees,
            price_ratio_updates=price_ratio_updates,
            all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
            arc_length_speed=s.arc_length_speed,
            centeredness_scaling=s.centeredness_scaling,
            protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
            ste_temperature=ste_temperature,
            noise_trader_ratio=run_fingerprint.get("noise_trader_ratio", 0.0),
            lp_supply_array=lp_prepared,
            noise_model=noise_model,
            noise_params=noise_params,
            volatility_array=noise_arrays["volatility"],
            dow_sin_array=noise_arrays["dow_sin"],
            dow_cos_array=noise_arrays["dow_cos"],
            noise_base_array=noise_arrays["noise_base"],
            noise_tvl_coeff_array=noise_arrays["noise_tvl_coeff"],
            competitor_tvl_array=noise_arrays["competitor_tvl"],
        )

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        lp_supply_array: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        oracle_prices = self._oracle_price_window(
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
        )
        return self._run_hypersurge_reserves(
            params,
            run_fingerprint,
            prices,
            start_index,
            fees=self._resolve_fees(params, run_fingerprint),
            gas_cost=run_fingerprint["gas_cost"],
            arb_fees=run_fingerprint["arb_fees"],
            lp_supply_array=lp_supply_array,
            price_ratio_updates=None,
            oracle_prices=oracle_prices,
        )

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_and_fee_revenue_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        lp_supply_array: Optional[jnp.ndarray] = None,
    ):
        oracle_prices = self._oracle_price_window(
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
        )
        return self._run_hypersurge_reserves(
            params,
            run_fingerprint,
            prices,
            start_index,
            fees=self._resolve_fees(params, run_fingerprint),
            gas_cost=run_fingerprint["gas_cost"],
            arb_fees=run_fingerprint["arb_fees"],
            lp_supply_array=lp_supply_array,
            price_ratio_updates=None,
            oracle_prices=oracle_prices,
            return_fee_revenue=True,
        )

    @partial(jit, static_argnums=(2,))
    def _calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        lp_supply_array: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        oracle_prices = self._oracle_price_window(
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
        )
        return self._run_hypersurge_reserves(
            params,
            run_fingerprint,
            prices,
            start_index,
            fees=0.0,
            gas_cost=run_fingerprint["gas_cost"],
            arb_fees=run_fingerprint["arb_fees"],
            lp_supply_array=lp_supply_array,
            price_ratio_updates=None,
            oracle_prices=oracle_prices,
        )

    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        lp_supply_array: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._calculate_reserves_zero_fees(
            params,
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
            lp_supply_array,
        )

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_with_dynamic_inputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        dynamic_inputs,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)
        materialized_inputs = materialize_dynamic_inputs(
            dynamic_inputs,
            run_fingerprint.get("dynamic_input_flags"),
            run_fingerprint,
            scan_len=s.arb_prices.shape[0],
            do_trades=False,
            dtype=s.arb_prices.dtype,
        )

        oracle_prices = self._oracle_price_window(
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
        )
        if materialized_inputs.oracle_prices.shape[-1] == run_fingerprint["n_assets"]:
            oracle_prices = materialized_inputs.oracle_prices

        return self._run_hypersurge_reserves(
            params,
            run_fingerprint,
            prices,
            start_index,
            fees=materialized_inputs.fees,
            gas_cost=materialized_inputs.gas_cost,
            arb_fees=materialized_inputs.arb_fees,
            lp_supply_array=materialized_inputs.lp_supply,
            price_ratio_updates=materialized_inputs.reclamm_price_ratio_updates,
            oracle_prices=oracle_prices,
            lp_supply_already_prepared=True,
        )

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_and_fee_revenue_with_dynamic_inputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        dynamic_inputs,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ):
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)
        materialized_inputs = materialize_dynamic_inputs(
            dynamic_inputs,
            run_fingerprint.get("dynamic_input_flags"),
            run_fingerprint,
            scan_len=s.arb_prices.shape[0],
            do_trades=False,
            dtype=s.arb_prices.dtype,
        )

        oracle_prices = self._oracle_price_window(
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
        )
        if materialized_inputs.oracle_prices.shape[-1] == run_fingerprint["n_assets"]:
            oracle_prices = materialized_inputs.oracle_prices

        return self._run_hypersurge_reserves(
            params,
            run_fingerprint,
            prices,
            start_index,
            fees=materialized_inputs.fees,
            gas_cost=materialized_inputs.gas_cost,
            arb_fees=materialized_inputs.arb_fees,
            lp_supply_array=materialized_inputs.lp_supply,
            price_ratio_updates=materialized_inputs.reclamm_price_ratio_updates,
            oracle_prices=oracle_prices,
            lp_supply_already_prepared=True,
            return_fee_revenue=True,
        )

    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        params = super().init_base_parameters(
            initial_values_dict,
            run_fingerprint,
            n_assets,
            n_parameter_sets=n_parameter_sets,
            noise=noise,
        )

        def process_scalar(key, default):
            value = initial_values_dict.get(key, default)
            if value is None:
                value = default
            value = np.asarray(value, dtype=np.float64)
            if value.size == 1:
                return np.array([[float(value.reshape(-1)[0])]] * n_parameter_sets)
            if value.shape == (n_parameter_sets,):
                return value.reshape(n_parameter_sets, 1)
            if value.shape == (n_parameter_sets, 1):
                return value
            raise ValueError(
                f"{key} must be a scalar or a matrix of shape (n_parameter_sets, 1)"
            )

        defaults = self._run_fingerprint_hypersurge_defaults(run_fingerprint)
        hypersurge_params = {
            key: process_scalar(key, defaults[key]) for key in HYPERSURGE_PARAM_KEYS
        }
        hypersurge_params = self.add_noise(hypersurge_params, noise, n_parameter_sets)
        params.update(hypersurge_params)
        return params

    def get_initial_values(self, run_fingerprint):
        values = super().get_initial_values(run_fingerprint)
        defaults = self._run_fingerprint_hypersurge_defaults(run_fingerprint)
        for key, value in defaults.items():
            values[key] = run_fingerprint.get(f"initial_{key}", value)
        return values


tree_util.register_pytree_node(
    ReClammHyperSurgePool,
    ReClammHyperSurgePool._tree_flatten,
    ReClammHyperSurgePool._tree_unflatten,
)
