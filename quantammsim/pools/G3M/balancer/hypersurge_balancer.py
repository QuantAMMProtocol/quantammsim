from functools import partial
from typing import Any, Dict, Optional

import numpy as np

import jax.numpy as jnp
from jax import jit, tree_util
from jax.lax import dynamic_slice

from quantammsim.core_simulator.dynamic_inputs import materialize_dynamic_inputs
from quantammsim.pools.G3M.balancer.balancer import BalancerPool
from quantammsim.pools.G3M.balancer.hypersurge_balancer_reserves import (
    _jax_calc_hypersurge_balancer_reserves,
)


def _prepare_dynamic_array(arr, start_index, bout_length, arb_frequency, max_len):
    """Slice and decimate a dynamic input array to match the arb-price scan."""
    arr = jnp.asarray(arr)
    if arr.ndim == 0:
        return jnp.full((max_len,), arr, dtype=arr.dtype)
    if arr.shape[0] <= 1:
        return jnp.broadcast_to(arr, (max_len,) + arr.shape[1:])

    start = (start_index[0],) + (0,) * (arr.ndim - 1)
    slice_sizes = (bout_length - 1,) + arr.shape[1:]
    sliced = dynamic_slice(arr, start, slice_sizes)
    if arb_frequency != 1:
        sliced = sliced[::arb_frequency]
    return sliced


def _coalesce(value, default):
    return default if value is None else value


HYPERSURGE_PARAM_KEYS = (
    "hypersurge_arb_max_fee",
    "hypersurge_arb_threshold",
    "hypersurge_arb_cap_deviation",
    "hypersurge_noise_max_fee",
    "hypersurge_noise_threshold",
    "hypersurge_noise_cap_deviation",
)


class HyperSurgeBalancerPool(BalancerPool):
    """Balancer weighted pool with HyperSurge-style state-dependent swap fees."""

    @staticmethod
    def _run_fingerprint_hypersurge_defaults(run_fingerprint: Dict[str, Any]):
        base_fee = run_fingerprint.get("fees", 0.0)
        if isinstance(base_fee, (list, tuple)):
            base_fee = base_fee[0]

        raw_params = run_fingerprint.get("hypersurge_params")
        if raw_params is not None:
            if isinstance(raw_params, dict):
                shared_max = raw_params.get("max_surge_fee", base_fee)
                shared_threshold = raw_params.get("threshold", 0.0)
                shared_cap = raw_params.get("cap_deviation", 1.0)
                return {
                    "hypersurge_arb_max_fee": raw_params.get(
                        "arb_max_fee", shared_max
                    ),
                    "hypersurge_arb_threshold": raw_params.get(
                        "arb_threshold", shared_threshold
                    ),
                    "hypersurge_arb_cap_deviation": raw_params.get(
                        "arb_cap_deviation", shared_cap
                    ),
                    "hypersurge_noise_max_fee": raw_params.get(
                        "noise_max_fee", shared_max
                    ),
                    "hypersurge_noise_threshold": raw_params.get(
                        "noise_threshold", shared_threshold
                    ),
                    "hypersurge_noise_cap_deviation": raw_params.get(
                        "noise_cap_deviation", shared_cap
                    ),
                }

            raw_params = np.asarray(raw_params, dtype=np.float64).reshape(-1)
            if raw_params.size != len(HYPERSURGE_PARAM_KEYS):
                raise ValueError(
                    "hypersurge_params must contain exactly six values: "
                    + ", ".join(HYPERSURGE_PARAM_KEYS)
                )
            return dict(zip(HYPERSURGE_PARAM_KEYS, raw_params))

        shared_max = _coalesce(
            run_fingerprint.get("hypersurge_max_surge_fee"),
            _coalesce(run_fingerprint.get("hypersurge_max_fee"), base_fee),
        )
        shared_threshold = _coalesce(
            run_fingerprint.get("hypersurge_threshold"),
            0.0,
        )
        shared_cap = _coalesce(
            run_fingerprint.get("hypersurge_cap_deviation"),
            1.0,
        )
        return {
            "hypersurge_arb_max_fee": _coalesce(
                run_fingerprint.get("hypersurge_arb_max_fee"), shared_max
            ),
            "hypersurge_arb_threshold": _coalesce(
                run_fingerprint.get("hypersurge_arb_threshold"), shared_threshold
            ),
            "hypersurge_arb_cap_deviation": _coalesce(
                run_fingerprint.get("hypersurge_arb_cap_deviation"), shared_cap
            ),
            "hypersurge_noise_max_fee": _coalesce(
                run_fingerprint.get("hypersurge_noise_max_fee"), shared_max
            ),
            "hypersurge_noise_threshold": _coalesce(
                run_fingerprint.get("hypersurge_noise_threshold"), shared_threshold
            ),
            "hypersurge_noise_cap_deviation": _coalesce(
                run_fingerprint.get("hypersurge_noise_cap_deviation"), shared_cap
            ),
        }

    def _hypersurge_params(self, params: Dict[str, Any], run_fingerprint: Dict[str, Any]):
        if "hypersurge_params" in params:
            return jnp.ravel(params["hypersurge_params"])

        if all(key in params for key in HYPERSURGE_PARAM_KEYS):
            return jnp.asarray(
                [jnp.squeeze(params[key]) for key in HYPERSURGE_PARAM_KEYS],
                dtype=jnp.float64,
            )

        defaults = self._run_fingerprint_hypersurge_defaults(run_fingerprint)
        return jnp.asarray(
            [defaults[key] for key in HYPERSURGE_PARAM_KEYS],
            dtype=jnp.float64,
        )

    def _price_windows(
        self,
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray],
    ):
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if additional_oracle_input is None:
            local_oracle_prices = local_prices
        else:
            local_oracle_prices = dynamic_slice(
                additional_oracle_input,
                start_index,
                (bout_length - 1, n_assets),
            )

        arb_frequency = run_fingerprint["arb_frequency"]
        if arb_frequency != 1:
            return local_prices[::arb_frequency], local_oracle_prices[::arb_frequency]
        return local_prices, local_oracle_prices

    def _noise_inputs(
        self,
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        max_len: int,
    ):
        noise_model = run_fingerprint.get("noise_model", "ratio")
        noise_base = None
        noise_tvl_coeff = None

        if noise_model != "market_linear":
            return noise_model, noise_base, noise_tvl_coeff

        noise_base = run_fingerprint.get("noise_base_array")
        noise_tvl_coeff = run_fingerprint.get("noise_tvl_coeff_array")
        if (noise_base is None or noise_tvl_coeff is None) and "noise_arrays_path" in run_fingerprint:
            path = run_fingerprint["noise_arrays_path"]
            if (
                not hasattr(self, "_market_linear_cache")
                or self._market_linear_cache[0] != path
            ):
                arrays = np.load(path)
                self._market_linear_cache = (
                    path,
                    arrays["noise_base"],
                    arrays["noise_tvl_coeff"],
                )
            noise_base = self._market_linear_cache[1]
            noise_tvl_coeff = self._market_linear_cache[2]

        if noise_base is None or noise_tvl_coeff is None:
            raise ValueError(
                "noise_model='market_linear' requires noise_base_array and "
                "noise_tvl_coeff_array, or noise_arrays_path."
            )

        noise_base = _prepare_dynamic_array(
            jnp.asarray(noise_base),
            start_index=start_index,
            bout_length=run_fingerprint["bout_length"],
            arb_frequency=run_fingerprint["arb_frequency"],
            max_len=max_len,
        )
        noise_tvl_coeff = _prepare_dynamic_array(
            jnp.asarray(noise_tvl_coeff),
            start_index=start_index,
            bout_length=run_fingerprint["bout_length"],
            arb_frequency=run_fingerprint["arb_frequency"],
            max_len=max_len,
        )
        return noise_model, noise_base, noise_tvl_coeff

    def _run_hypersurge_reserves(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        fees,
        gas_cost,
        arb_fees,
        trades,
        do_trades: bool,
        lp_supply_array,
        additional_oracle_input: Optional[jnp.ndarray] = None,
        oracle_prices_override: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        weights = self.calculate_initial_weights(params)
        arb_prices, oracle_prices = self._price_windows(
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
        )
        if oracle_prices_override is not None:
            oracle_prices = oracle_prices_override

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / arb_prices[0]

        noise_model, noise_base, noise_tvl_coeff = self._noise_inputs(
            run_fingerprint,
            prices,
            start_index,
            arb_prices.shape[0],
        )

        return _jax_calc_hypersurge_balancer_reserves(
            initial_reserves,
            weights,
            arb_prices,
            oracle_prices,
            fees=fees,
            arb_thresh=gas_cost,
            arb_fees=arb_fees,
            all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
            trades=trades,
            do_trades=do_trades,
            do_arb=run_fingerprint["do_arb"],
            lp_supply_array=lp_supply_array,
            hypersurge_params=self._hypersurge_params(params, run_fingerprint),
            noise_trader_ratio=run_fingerprint.get("noise_trader_ratio", 0.0),
            protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
            noise_model=noise_model,
            noise_base_array=noise_base,
            noise_tvl_coeff_array=noise_tvl_coeff,
            tvl_mean=run_fingerprint.get("noise_tvl_mean", 0.0),
            tvl_std=run_fingerprint.get("noise_tvl_std", 1.0),
            minutes_per_step=run_fingerprint.get(
                "seconds_per_step",
                60.0 * run_fingerprint["arb_frequency"],
            )
            / 60.0,
        )

    @partial(jit, static_argnums=2)
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._run_hypersurge_reserves(
            params,
            run_fingerprint,
            prices,
            start_index,
            fees=run_fingerprint["fees"],
            gas_cost=run_fingerprint["gas_cost"],
            arb_fees=run_fingerprint["arb_fees"],
            trades=None,
            do_trades=False,
            lp_supply_array=None,
            additional_oracle_input=additional_oracle_input,
        )

    @partial(jit, static_argnums=2)
    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._run_hypersurge_reserves(
            params,
            run_fingerprint,
            prices,
            start_index,
            fees=0.0,
            gas_cost=run_fingerprint["gas_cost"],
            arb_fees=run_fingerprint["arb_fees"],
            trades=None,
            do_trades=False,
            lp_supply_array=None,
            additional_oracle_input=additional_oracle_input,
        )

    @partial(jit, static_argnums=2)
    def calculate_reserves_with_dynamic_inputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        dynamic_inputs,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        arb_prices, fallback_oracle_prices = self._price_windows(
            run_fingerprint,
            prices,
            start_index,
            additional_oracle_input,
        )
        materialized_inputs = materialize_dynamic_inputs(
            dynamic_inputs,
            run_fingerprint.get("dynamic_input_flags"),
            run_fingerprint,
            scan_len=arb_prices.shape[0],
            do_trades=run_fingerprint["do_trades"],
            dtype=arb_prices.dtype,
        )

        oracle_prices = fallback_oracle_prices
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
            trades=materialized_inputs.trades,
            do_trades=run_fingerprint["do_trades"],
            lp_supply_array=materialized_inputs.lp_supply,
            additional_oracle_input=additional_oracle_input,
            oracle_prices_override=oracle_prices,
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

        def process_weights(key):
            if key not in initial_values_dict:
                raise ValueError(f"initial_values_dict must contain {key}")
            initial_value = initial_values_dict[key]
            if isinstance(initial_value, (np.ndarray, jnp.ndarray, list)):
                initial_value = np.array(initial_value)
                if initial_value.size == n_assets:
                    return np.array([initial_value] * n_parameter_sets)
                if initial_value.size == 1:
                    return np.array([[initial_value] * n_assets] * n_parameter_sets)
                if initial_value.shape == (n_parameter_sets, n_assets):
                    return initial_value
                raise ValueError(
                    f"{key} must be a singleton or a vector of length n_assets "
                    "or a matrix of shape (n_parameter_sets, n_assets)"
                )
            return np.array([[initial_value] * n_assets] * n_parameter_sets)

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
                f"{key} must be a scalar or a matrix of shape "
                "(n_parameter_sets, 1)"
            )

        hypersurge_defaults = self._run_fingerprint_hypersurge_defaults(
            run_fingerprint
        )
        params = {
            "initial_weights_logits": process_weights("initial_weights_logits"),
            "subsidary_params": [],
        }
        for key in HYPERSURGE_PARAM_KEYS:
            params[key] = process_scalar(key, hypersurge_defaults[key])

        return self.add_noise(params, noise, n_parameter_sets)

    def get_initial_values(self, run_fingerprint):
        values = {
            "initial_weights_logits": run_fingerprint.get(
                "initial_weights_logits", 1.0
            ),
        }
        defaults = self._run_fingerprint_hypersurge_defaults(run_fingerprint)
        for key, value in defaults.items():
            values[key] = run_fingerprint.get(f"initial_{key}", value)
        return values

    def is_trainable(self):
        return True


tree_util.register_pytree_node(
    HyperSurgeBalancerPool,
    HyperSurgeBalancerPool._tree_flatten,
    HyperSurgeBalancerPool._tree_unflatten,
)
