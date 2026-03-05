"""reClAMM pool implementation.

Rebalancing Concentrated Liquidity AMM — a 2-token constant-product pool
with dynamic virtual reserves that track market price. Extends AbstractPool
following the GyroscopePool pattern (scan-based). Trainable via Optuna.
"""

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, tree_util
from jax.lax import dynamic_slice
from functools import partial
from typing import Dict, Any, Optional, NamedTuple
import numpy as np

from quantammsim.pools.base_pool import AbstractPool
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    initialise_reclamm_reserves,
    calibrate_arc_length_speed,
    compute_price_ratio,
    _jax_calc_reclamm_reserves_zero_fees,
    _jax_calc_reclamm_reserves_with_fees,
    _jax_calc_reclamm_reserves_with_dynamic_inputs,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_fees,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs,
)


# Solidity constant: daily_price_shift_base = 1 - shift_exponent / DIVISOR
SHIFT_EXPONENT_DIVISOR = 124649.0


class _PoolState(NamedTuple):
    """Intermediate state produced by _init_pool_state.

    All fields are JAX arrays (or Python scalars for seconds_per_step /
    centeredness_scaling). JAX treats NamedTuples as pytree nodes, so this
    works inside JIT-traced code without special registration.
    """
    local_prices: jnp.ndarray
    arb_prices: jnp.ndarray
    initial_reserves: jnp.ndarray
    Va: jnp.ndarray
    Vb: jnp.ndarray
    centeredness_margin: jnp.ndarray
    daily_price_shift_base: jnp.ndarray
    seconds_per_step: float
    arc_length_speed: jnp.ndarray
    centeredness_scaling: bool


def _resolve_arc_length_speed(
    params, run_fingerprint, initial_reserves, Va, Vb,
    local_prices, centeredness_margin, daily_price_shift_base, seconds_per_step,
):
    """Three-level priority for arc_length_speed resolution.

    1. Learnable: ``"arc_length_speed" in params`` — use the param value.
    2. Fingerprint override: ``reclamm_arc_length_speed is not None``.
    3. Auto-calibrate from geometric onset.

    This is a Python-level if/elif/else evaluated at JIT trace time.
    Different param structures produce different compiled functions.
    """
    interpolation_method = run_fingerprint.get(
        "reclamm_interpolation_method", "geometric"
    )
    if interpolation_method != "constant_arc_length":
        return jnp.float64(0.0)

    # Priority 1: learnable param
    if "arc_length_speed" in params:
        return jnp.squeeze(params["arc_length_speed"])

    # Priority 2: fingerprint override
    speed_override = run_fingerprint.get("reclamm_arc_length_speed", None)
    if speed_override is not None:
        return jnp.float64(speed_override)

    # Priority 3: auto-calibrate
    market_price_0 = local_prices[0, 0] / local_prices[0, 1]
    sqrt_Q = jnp.sqrt(compute_price_ratio(
        initial_reserves[0], initial_reserves[1], Va, Vb,
    ))
    return calibrate_arc_length_speed(
        initial_reserves[0], initial_reserves[1], Va, Vb,
        daily_price_shift_base, seconds_per_step, sqrt_Q, market_price_0,
        centeredness_margin=centeredness_margin,
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
    Trainable via Optuna (hyperparameter search over pool geometry).
    Weights are empirical (derived from reserves * prices / total value).
    """

    def __init__(self):
        super().__init__()

    def _init_pool_state(self, params, run_fingerprint, prices, start_index):
        """Centralised setup: price slicing, param extraction, reserve init,
        arc_length_speed resolution.

        Called by all reserve/weight methods. Not @jit itself — inlined
        during tracing of the calling method.
        """
        assert run_fingerprint["n_assets"] == 2

        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(
            prices, start_index, (bout_length - 1, n_assets)
        )

        if run_fingerprint["arb_frequency"] != 1:
            arb_prices = local_prices[:: run_fingerprint["arb_frequency"]]
        else:
            arb_prices = local_prices

        price_ratio = jnp.squeeze(params["price_ratio"])
        centeredness_margin = jnp.squeeze(params["centeredness_margin"])
        if "shift_exponent" in params:
            daily_price_shift_base = (
                1.0 - jnp.squeeze(params["shift_exponent"]) / SHIFT_EXPONENT_DIVISOR
            )
        else:
            daily_price_shift_base = jnp.squeeze(params["daily_price_shift_base"])

        seconds_per_step = run_fingerprint["arb_frequency"] * 60.0

        # On-chain state override: use actual reserves/virtuals instead of
        # computing a fresh centered pool.  Python-level branch — different
        # fingerprint structures produce different compiled functions.
        onchain = run_fingerprint.get("reclamm_initial_state", None)
        if onchain is not None:
            initial_reserves = jnp.array(
                [onchain["Ra"], onchain["Rb"]], dtype=jnp.float64,
            )
            Va = jnp.float64(onchain["Va"])
            Vb = jnp.float64(onchain["Vb"])
        else:
            initial_pool_value = run_fingerprint["initial_pool_value"]
            initial_reserves, Va, Vb = initialise_reclamm_reserves(
                initial_pool_value, local_prices[0], price_ratio
            )

        arc_length_speed = _resolve_arc_length_speed(
            params, run_fingerprint, initial_reserves, Va, Vb,
            local_prices, centeredness_margin, daily_price_shift_base,
            seconds_per_step,
        )

        centeredness_scaling = run_fingerprint.get(
            "reclamm_centeredness_scaling", False
        )

        return _PoolState(
            local_prices=local_prices,
            arb_prices=arb_prices,
            initial_reserves=initial_reserves,
            Va=Va,
            Vb=Vb,
            centeredness_margin=centeredness_margin,
            daily_price_shift_base=daily_price_shift_base,
            seconds_per_step=seconds_per_step,
            arc_length_speed=arc_length_speed,
            centeredness_scaling=centeredness_scaling,
        )

    @staticmethod
    def _resolve_fees(params, run_fingerprint):
        """Use learnable fees from params if present, else fingerprint value."""
        if "fees" in params:
            return jnp.squeeze(params["fees"])
        return run_fingerprint["fees"]

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)

        if run_fingerprint["do_arb"]:
            return _jax_calc_reclamm_reserves_with_fees(
                s.initial_reserves, s.Va, s.Vb,
                s.arb_prices,
                s.centeredness_margin,
                s.daily_price_shift_base,
                s.seconds_per_step,
                fees=self._resolve_fees(params, run_fingerprint),
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(
                    run_fingerprint["all_sig_variations"]
                ),
                arc_length_speed=s.arc_length_speed,
                centeredness_scaling=s.centeredness_scaling,
                protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
            )
        return jnp.broadcast_to(s.initial_reserves, s.arb_prices.shape)

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_and_fee_revenue_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ):
        """Calculate reserves and LP fee revenue with fees.

        Returns
        -------
        reserves : jnp.ndarray, shape (T, 2)
        fee_revenue : jnp.ndarray, shape (T,)
            LP fee revenue per timestep in USD.
        """
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)

        if run_fingerprint["do_arb"]:
            return _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
                s.initial_reserves, s.Va, s.Vb,
                s.arb_prices,
                s.centeredness_margin,
                s.daily_price_shift_base,
                s.seconds_per_step,
                fees=self._resolve_fees(params, run_fingerprint),
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(
                    run_fingerprint["all_sig_variations"]
                ),
                arc_length_speed=s.arc_length_speed,
                centeredness_scaling=s.centeredness_scaling,
                protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
            )
        return (
            jnp.broadcast_to(s.initial_reserves, s.arb_prices.shape),
            jnp.zeros(s.arb_prices.shape[0]),
        )

    @partial(jit, static_argnums=(2,))
    def calculate_reserves_and_fee_revenue_with_dynamic_inputs(
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
    ):
        """Calculate reserves and LP fee revenue with time-varying inputs.

        Returns
        -------
        reserves : jnp.ndarray, shape (T, 2)
        fee_revenue : jnp.ndarray, shape (T,)
            LP fee revenue per timestep in USD.
        """
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)

        bout_length = run_fingerprint["bout_length"]
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

        return _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs(
            s.initial_reserves, s.Va, s.Vb,
            s.arb_prices,
            s.centeredness_margin,
            s.daily_price_shift_base,
            s.seconds_per_step,
            fees=fees_array_broadcast,
            arb_thresh=arb_thresh_array_broadcast,
            arb_fees=arb_fees_array_broadcast,
            all_sig_variations=jnp.array(
                run_fingerprint["all_sig_variations"]
            ),
            arc_length_speed=s.arc_length_speed,
            centeredness_scaling=s.centeredness_scaling,
            protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
        )

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
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)

        if run_fingerprint["do_arb"]:
            return _jax_calc_reclamm_reserves_zero_fees(
                s.initial_reserves, s.Va, s.Vb,
                s.arb_prices,
                s.centeredness_margin,
                s.daily_price_shift_base,
                s.seconds_per_step,
                arc_length_speed=s.arc_length_speed,
                centeredness_scaling=s.centeredness_scaling,
            )
        return jnp.broadcast_to(s.initial_reserves, s.arb_prices.shape)

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
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)

        bout_length = run_fingerprint["bout_length"]
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

        return _jax_calc_reclamm_reserves_with_dynamic_inputs(
            s.initial_reserves, s.Va, s.Vb,
            s.arb_prices,
            s.centeredness_margin,
            s.daily_price_shift_base,
            s.seconds_per_step,
            fees=fees_array_broadcast,
            arb_thresh=arb_thresh_array_broadcast,
            arb_fees=arb_fees_array_broadcast,
            all_sig_variations=jnp.array(
                run_fingerprint["all_sig_variations"]
            ),
            arc_length_speed=s.arc_length_speed,
            centeredness_scaling=s.centeredness_scaling,
            protocol_fee_split=run_fingerprint.get("protocol_fee_split", 0.0),
        )

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

        Optional (when reclamm_learn_arc_length_speed is True):
        - arc_length_speed: thermostat speed for constant-arc-length interpolation
        """
        def process(key, default=None):
            if key in initial_values_dict:
                val = initial_values_dict[key]
                if isinstance(val, (np.ndarray, jnp.ndarray, list)):
                    val = np.array(val)
                    if val.size == 1:
                        return np.array([[float(val)]] * n_parameter_sets)
                    elif val.shape == (n_parameter_sets,):
                        return val.reshape(n_parameter_sets, 1)
                    elif val.shape == (n_parameter_sets, 1):
                        return val
                    else:
                        raise ValueError(f"{key} shape mismatch")
                else:
                    return np.array([[float(val)]] * n_parameter_sets)
            elif default is not None:
                return np.array([[default]] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        use_shift_exp = run_fingerprint.get("reclamm_use_shift_exponent", False)
        params = {
            "price_ratio": process("price_ratio", 4.0),
            "centeredness_margin": process("centeredness_margin", 0.2),
            "subsidary_params": [],
        }
        if use_shift_exp:
            params["shift_exponent"] = process("shift_exponent", 1.0)
        else:
            params["daily_price_shift_base"] = process(
                "daily_price_shift_base", 1.0 - 1.0 / 124000.0
            )

        learn_speed = (
            run_fingerprint.get("reclamm_learn_arc_length_speed", False)
            and run_fingerprint.get("reclamm_interpolation_method", "geometric")
            == "constant_arc_length"
        )
        if learn_speed:
            params["arc_length_speed"] = process(
                "arc_length_speed",
                run_fingerprint.get("initial_arc_length_speed", 1e-4),
            )

        if run_fingerprint.get("reclamm_learn_fees", False):
            init_fees = run_fingerprint.get("fees", 0.0025)
            assert init_fees > 0, (
                "reclamm_learn_fees requires fees > 0 in run_fingerprint "
                "(needed for forward-pass dispatch to with-fees path). "
                f"Got fees={init_fees}"
            )
            params["fees"] = process("fees", init_fees)

        params = self.add_noise(params, noise, n_parameter_sets)
        return params

    def is_trainable(self):
        return True

    def get_initial_values(self, run_fingerprint):
        """Extract initial reClAMM parameter values from run_fingerprint."""
        use_shift_exp = run_fingerprint.get("reclamm_use_shift_exponent", False)
        vals = {
            "price_ratio": run_fingerprint.get("initial_price_ratio", 4.0),
            "centeredness_margin": run_fingerprint.get(
                "initial_centeredness_margin", 0.2
            ),
        }
        if use_shift_exp:
            vals["shift_exponent"] = run_fingerprint.get(
                "initial_shift_exponent", 1.0
            )
        else:
            vals["daily_price_shift_base"] = run_fingerprint.get(
                "initial_daily_price_shift_base", 1.0 - 1.0 / 124000.0
            )

        learn_speed = (
            run_fingerprint.get("reclamm_learn_arc_length_speed", False)
            and run_fingerprint.get("reclamm_interpolation_method", "geometric")
            == "constant_arc_length"
        )
        if learn_speed:
            vals["arc_length_speed"] = run_fingerprint.get(
                "initial_arc_length_speed", 1e-4
            )

        if run_fingerprint.get("reclamm_learn_fees", False):
            vals["fees"] = run_fingerprint.get("fees", 0.0025)

        return vals

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
        s = self._init_pool_state(params, run_fingerprint, prices, start_index)

        reserves = self._calculate_reserves_zero_fees(
            params, run_fingerprint, prices, start_index, additional_oracle_input
        )
        value = reserves * s.arb_prices
        weights = value / jnp.sum(value, axis=-1, keepdims=True)
        return weights


tree_util.register_pytree_node(
    ReClammPool,
    ReClammPool._tree_flatten,
    ReClammPool._tree_unflatten,
)
