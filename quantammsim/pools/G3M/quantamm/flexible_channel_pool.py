# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
from jax import default_backend
from jax import local_device_count, devices
import jax.nn as jnn

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from jax import jit, vmap, lax
from jax import devices, device_put
from jax import tree_util
from jax.lax import stop_gradient, dynamic_slice

from quantammsim.pools.G3M.quantamm.momentum_pool import (
    MomentumPool,
    _jax_momentum_weight_update,
)
from quantammsim.core_simulator.param_utils import (
    memory_days_to_lamb,
    lamb_to_memory_days_clipped,
    calc_lamb,
    inverse_squareplus_np,
    get_raw_value,
    get_log_amplitude,
)
from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import (
    calc_gradients,
    calc_k,
    squareplus,
)

from typing import Dict, Any, Optional
from functools import partial
import numpy as np

# import the fine weight output function which has pre-set argument raw_weight_outputs_are_themselves_weights
# as this is False for momentum pools --- the strategy outputs weight _changes_
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import (
    calc_fine_weight_output_from_weight_changes,
)


@jit
def _jax_flexible_channel_weight_update(
    price_gradient: jnp.ndarray,  # shape (S, N)
    k: jnp.ndarray,  # shape (S, N)
    width_env: jnp.ndarray,  # σ_env
    amplitude: jnp.ndarray,
    alpha: jnp.ndarray,
    exponents_up: jnp.ndarray,
    exponents_down: jnp.ndarray,
    risk_off: jnp.ndarray,  # ρ_off  (S, N)
    risk_on: jnp.ndarray,  # ρ_on   (S, N)
    profit_pos: jnp.ndarray,  # Π⁺     (S, 1)
    drawdown_neg: jnp.ndarray,  # Π⁻     (S, 1)
    inverse_scaling: float = 0.5415,
    pre_exp_scaling: float = 0.5,
) -> jnp.ndarray:
    """
    Mean-reversion + trend-following + profit-harvest + risk-on amplifier.
    """
    # 1) Envelope ------------------------------------------------------
    envelope = jnp.exp(-(price_gradient**2) / (2.0 * width_env**2))

    # 2) Cubic channel -------------------------------------------------
    width_poly = width_env / alpha
    s = (jnp.pi * price_gradient) / (3.0 * width_poly)
    channel = -amplitude * envelope * (s - s**3 / 6.0) / inverse_scaling

    # 3) Bare trend-following -----------------------------------------
    base = jnp.abs(price_gradient / (2.0 * pre_exp_scaling))
    exp_choose = jnp.where(price_gradient > 0, exponents_up, exponents_down)
    trend_bare = (
        (1.0 - envelope) * jnp.sign(price_gradient) * jnp.power(base, exp_choose)
    )

    # 3b) Apply risk-on amplifier (only when Π⁻>0) ---------------------
    trend = trend_bare * (1.0 + risk_on * drawdown_neg)

    # 4) Profit-to-stable term ----------------------------------------
    profit_term = risk_off * jnp.broadcast_to(profit_pos, price_gradient.shape)

    # 5) Combine, offset, scale ---------------------------------------
    signal = channel + trend + profit_term
    offset = -(k * signal).sum(axis=-1, keepdims=True) / jnp.sum(k)
    return k * (signal + offset)


class FlexibleChannelPool(MomentumPool):
    """
    A class for mean reversion channel strategies run as TFMM liquidity pools.

    This class implements a "mean reversion channel" strategy for asset allocation within a TFMM framework.
    It uses price data to generate mean reversion channel signals, which are then translated into weight adjustments.

    Parameters
    ----------
    None

    Methods
    -------
    calculate_raw_weights_outputs(params, run_fingerprint, prices, additional_oracle_input)
        Calculate the raw weight outputs based on mean reversion channel signals.

    Notes
    -----
    The FlexibleChannelPool implements a mean-reversion-based channel following strategy for asset allocation within a TFMM framework.
    It uses price data to generate mean-reversion signals, which are then translated into weight adjustments.
    The class provides methods to calculate raw weight outputs based on these signals and refine them
    into final asset weights, taking into account various parameters and constraints defined in the pool setup.
    """

    def __init__(self):
        """
        Initialize a new FlexibleChannelPool instance.

        Parameters
        ----------
        None
        """
        super().__init__()

    # ─────────────────────────────────────────────────────────────────
    #  calculate_raw_weights_outputs  (final, mutation-safe)
    # ─────────────────────────────────────────────────────────────────

    @partial(jit, static_argnums=(2,))
    def calculate_raw_weights_outputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,  # (T_total, N)
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        fp = run_fingerprint
        S, N = params["log_k"].shape

        # —— helper to compute 2**x with overflow guard ——————————————
        ln2 = jnp.log(2.0)

        def pow2_safe(x):
            return jnp.exp(jnp.clip(squareplus(x) * ln2, a_min=-60.0, a_max=60.0))

        # 0) realised σ̂_i(t)  (EWMA with trainable λσ)
        lookback = fp.get("vol_lookback", 90)
        chunk_prices = prices[:: fp["chunk_period"]]
        window = chunk_prices[-(lookback + 1) :]  # (L+1,N)
        log_ret_sq = jnp.square(jnp.log(window[1:] / window[:-1]))

        lambda_vol = jnn.sigmoid(params["logit_lamb_vol"])  # (S,N)

        def ewma_1d(x, lam):
            def step(c, xi):
                return c * lam + xi * (1 - lam), None

            last, _ = lax.scan(step, x[0], x[1:])
            return jnp.sqrt(last + 1e-12)

        sigma_hat = vmap(vmap(ewma_1d, in_axes=(1, 0), out_axes=1), in_axes=(None, 0))(
            log_ret_sq, lambda_vol
        )  # (S,N)

        # 1) adaptive width & Kelly-scaled k
        width_env = pow2_safe(params["raw_width"]) * sigma_hat

        memory_days = lamb_to_memory_days_clipped(
            calc_lamb(params), fp["chunk_period"], fp["max_memory_days"]
        )  # (S,)
        k_plain = calc_k(params, memory_days)  # (S,N)

        kappa = pow2_safe(params["raw_kelly_kappa"])  # (S,N)
        k = k_plain * kappa / jnp.clip(sigma_hat, 1e-9, None)  # avoid Inf

        # 2) β  (pre-exp scaling)
        if (
            fp["use_pre_exp_scaling"]
            and params.get("logit_pre_exp_scaling") is not None
        ):
            beta = jnn.sigmoid(params["logit_pre_exp_scaling"])
        elif (
            fp["use_pre_exp_scaling"] and params.get("raw_pre_exp_scaling") is not None
        ):
            beta = pow2_safe(params["raw_pre_exp_scaling"])
        else:
            beta = 0.5
        beta = jnp.broadcast_to(beta, (S, N))

        # 3) main-λ gradients
        grads = calc_gradients(
            params,
            chunk_prices,
            fp["chunk_period"],
            fp["max_memory_days"],
            fp["use_alt_lamb"],
            cap_lamb=True,
        )  # (S,N)

        # 4) Π⁺  (profit skim)
        port_prices = jnp.mean(prices, axis=1).reshape(-1, 1)
        profit_pos = jnp.maximum(
            0.0,
            calc_gradients(
                params,
                port_prices,
                fp["chunk_period"],
                fp["max_memory_days"],
                fp["use_alt_lamb"],
                cap_lamb=True,
            ),
        )  # (S,1)

        # 5) Πᴅ⁻  (draw-down, λᴅ)  – build shallow copy (no mutation)
        params_dd = dict(params, logit_lamb=params["logit_lamb_drawdown"])
        drawdown_neg = jnp.maximum(
            0.0,
            -calc_gradients(
                params_dd,
                port_prices,
                fp["chunk_period"],
                fp["max_memory_days"],
                fp["use_alt_lamb"],
                cap_lamb=True,
            ),
        )  # (S,1)

        # 6) deterministic transforms
        alpha = pow2_safe(params["raw_alpha"])
        exp_up = squareplus(params["raw_exponents_up"])
        exp_dn = squareplus(params["raw_exponents_down"])
        amplitude = pow2_safe(params["log_amplitude"]) * memory_days[:, None]
        risk_off = jnn.sigmoid(params["logit_risk_off"])
        risk_on = jnn.sigmoid(params["logit_risk_on"])

        # 7) kernel
        raw = _jax_flexible_channel_weight_update(
            price_gradient=grads,
            k=k,
            width_env=width_env,
            amplitude=amplitude,
            alpha=alpha,
            exponents_up=exp_up,
            exponents_down=exp_dn,
            risk_off=risk_off,
            risk_on=risk_on,
            profit_pos=profit_pos,
            drawdown_neg=drawdown_neg,
            pre_exp_scaling=beta,
        )

        # 8) entropy guard-rail (√-shrink, branch-free)
        prev_w = fp["prev_weights"]  # (S,N)
        new_w = jnp.clip(prev_w + raw, 1e-12)
        new_w /= new_w.sum(axis=-1, keepdims=True)
        ent = -jnp.sum(new_w * jnp.log(new_w), axis=-1, keepdims=True)
        H_min = jnn.softplus(params["raw_entropy_floor"])  # (S,1)
        gamma = jnp.sqrt(jnp.clip(H_min / ent, a_min=0.0))
        gamma = jnp.minimum(gamma, 1.0)
        return raw * gamma

        # ─────────────────────────────────────────────────────────────────

    #  init_base_parameters   (fully fixed)
    # ─────────────────────────────────────────────────────────────────
    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        """
        Build the parameter dictionary in *internal* representation:

            • log_k
            • logit_lamb, logit_delta_lamb
            • logit_lamb_drawdown      (NEW: λᴅ)
            • raw_width, raw_alpha
            • raw_exponents_up / down
            • raw_pre_exp_scaling
            • log_amplitude
            • logit_risk_off / logit_risk_on
            • initial_weights_logits
            • subsidary_params

        The helper _expand() makes every key accept:
            scalar, length-n_assets vector, or
            (n_parameter_sets, n_assets) matrix.
        """
        np.random.seed(0)

        # ── helper for broadcasting & validation ──────────────────────
        def _expand(key: str, *, force_scalar: bool = False) -> np.ndarray:
            if key not in initial_values_dict:
                raise ValueError(f"initial_values_dict missing '{key}'")
            val = np.asarray(initial_values_dict[key])
            if force_scalar:
                return np.broadcast_to(val, (n_parameter_sets, 1))
            if val.size == 1:
                val = np.full((n_assets,), val.item())
            if val.shape == (n_assets,):
                val = np.stack([val] * n_parameter_sets, axis=0)
            if val.shape != (n_parameter_sets, n_assets):
                raise ValueError(
                    f"'{key}' must be scalar, len-{n_assets} vector, or "
                    f"({n_parameter_sets},{n_assets}) array"
                )
            return val

        force_scalar = run_fingerprint["optimisation_settings"]["force_scalar"]

        # ── 1) Basic aggressiveness k ─────────────────────────────────
        initial_k_per_day = _expand("initial_k_per_day", force_scalar=force_scalar)
        log_k = np.log2(initial_k_per_day)  # (S,N)

        # ── 2) Main λ and Δλ ──────────────────────────────────────────
        initial_mem_len = initial_values_dict["initial_memory_length"]
        lamb_main = memory_days_to_lamb(
            initial_mem_len, run_fingerprint["chunk_period"]
        )
        logit_lamb = np.log(lamb_main / (1.0 - lamb_main))
        logit_lamb = np.broadcast_to(
            logit_lamb, (n_parameter_sets, 1 if force_scalar else n_assets)
        )

        delta_mem = initial_values_dict["initial_memory_length_delta"]
        lamb_plus_delta = memory_days_to_lamb(
            initial_mem_len + delta_mem, run_fingerprint["chunk_period"]
        )
        logit_lamb_plus_delta = np.log(lamb_plus_delta / (1.0 - lamb_plus_delta))
        logit_delta_lamb = logit_lamb_plus_delta - logit_lamb

        # ── 3) Draw-down λᴅ  (separate memory) ────────────────────────
        initial_mem_len_dd = initial_values_dict["initial_memory_length_drawdown"]
        lamb_dd = memory_days_to_lamb(
            initial_mem_len_dd, run_fingerprint["chunk_period"]
        )
        logit_lamb_drawdown = np.log(lamb_dd / (1.0 - lamb_dd))
        logit_lamb_drawdown = np.broadcast_to(
            logit_lamb_drawdown, (n_parameter_sets, 1 if force_scalar else n_assets)
        )

        # ── 4) Shape parameters (per-token) ───────────────────────────
        log_amplitude = _expand("initial_log_amplitude", force_scalar=force_scalar)
        raw_width = _expand("initial_raw_width", force_scalar=force_scalar)
        raw_alpha = _expand("initial_raw_alpha", force_scalar=force_scalar)
        raw_exp_up = _expand("initial_raw_exponents_up", force_scalar=force_scalar)
        raw_exp_down = _expand("initial_raw_exponents_down", force_scalar=force_scalar)

        # pre-exp scaling β  (log₂ space, may be scalar or vector)
        raw_pre_exp_scaling = _expand(
            "initial_raw_pre_exp_scaling", force_scalar=force_scalar
        )

        # ── 5) Risk routing knobs (probabilities → logits) ────────────
        risk_off_prob = _expand("initial_risk_off")  # already prob
        risk_on_prob = _expand("initial_risk_on")
        logit_risk_off = np.log(risk_off_prob / (1.0 - risk_off_prob))
        logit_risk_on = np.log(risk_on_prob / (1.0 - risk_on_prob))

        # ── 6) Initial weight logits (softmax space) ──────────────────
        initial_weights_logits = _expand("initial_weights_logits", force_scalar=False)

        # ── inside init_base_parameters (add after other per-token knobs) ──
        raw_kelly_kappa = _expand("initial_raw_kelly_kappa", force_scalar=force_scalar)
        logit_lamb_vol = _expand("initial_logit_lamb_vol", force_scalar=force_scalar)
        raw_entropy_floor = _expand(
            "initial_raw_entropy_floor", force_scalar=True
        )  # one per set

        # ── 7) Pack dictionary ────────────────────────────────────────
        params = {
            "log_k": log_k,
            "logit_lamb": logit_lamb,
            "logit_delta_lamb": logit_delta_lamb,
            "logit_lamb_drawdown": logit_lamb_drawdown,  # NEW
            "initial_weights_logits": initial_weights_logits,
            "log_amplitude": log_amplitude,
            "raw_width": raw_width,
            "raw_alpha": raw_alpha,
            "raw_exponents_up": raw_exp_up,
            "raw_exponents_down": raw_exp_down,
            "raw_pre_exp_scaling": raw_pre_exp_scaling,
            "logit_risk_off": logit_risk_off,
            "logit_risk_on": logit_risk_on,
            "raw_kelly_kappa": raw_kelly_kappa,
            "logit_lamb_vol": logit_lamb_vol,
            "raw_entropy_floor": raw_entropy_floor,  # one per set
            "subsidary_params": [],
        }

        params = self.add_noise(params, noise, n_parameter_sets)
        return params

    # ─────────────────────────────────────────────────────────────────
    #  _process_specific_parameters  (entropy-floor shape fixed)
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def _process_specific_parameters(
        cls,
        update_rule_parameters,
        run_fingerprint: Dict[str, Any],
    ) -> Dict[str, Any]:
        tokens  = run_fingerprint["tokens"]
        n_tok   = len(tokens)
        cp      = run_fingerprint["chunk_period"]
        S_sets  = run_fingerprint["n_parameter_sets"]

        def _broadcast(vals, fn=lambda x: x):
            out = [fn(v) for v in vals]
            if len(out) != n_tok:
                out = [out[0]] * n_tok
            return out

        res, md_main, md_dd, tmp_amp = {}, None, None, None

        # collect memory_days first
        for p in update_rule_parameters:
            if p.name == "memory_days":
                md_main = p.value
            elif p.name == "memory_days_drawdown":
                md_dd = p.value

        # second pass
        for p in update_rule_parameters:
            n, v = p.name, p.value
            if n == "width":
                res["raw_width"] = jnp.array(_broadcast(v, get_raw_value))
            elif n == "alpha":
                res["raw_alpha"] = jnp.array(_broadcast(v, get_raw_value))
            elif n == "exponent_up":
                res["raw_exponents_up"] = jnp.array(_broadcast(v, inverse_squareplus_np))
            elif n == "exponent_down":
                res["raw_exponents_down"] = jnp.array(_broadcast(v, inverse_squareplus_np))
            elif n == "pre_exp_scaling":
                res["raw_pre_exp_scaling"] = jnp.array(_broadcast(v, get_raw_value))
            elif n == "risk_off":
                probs = _broadcast(v, float)
                res["logit_risk_off"] = jnp.array([np.log(p/(1-p)) for p in probs])
            elif n == "risk_on":
                probs = _broadcast(v, float)
                res["logit_risk_on"] = jnp.array([np.log(p/(1-p)) for p in probs])
            elif n == "kelly_kappa":
                res["raw_kelly_kappa"] = jnp.array(_broadcast(v, get_raw_value))
            elif n == "vol_memory_days":
                lambs = [memory_days_to_lamb(d, cp) for d in _broadcast(v, float)]
                res["logit_lamb_vol"] = jnp.array([np.log(l/(1-l)) for l in lambs])
            elif n == "entropy_floor":
                val = float(get_raw_value(v[0] if isinstance(v, (list, tuple)) else v))
                res["raw_entropy_floor"] = jnp.full((S_sets, 1), val)    # (S,1)
            elif n == "amplitude":
                tmp_amp = _broadcast(v, float)

        # draw-down λᴅ
        if md_dd is not None:
            lambs_dd = [memory_days_to_lamb(d, cp) for d in _broadcast(md_dd, float)]
            res["logit_lamb_drawdown"] = jnp.array([np.log(l/(1-l)) for l in lambs_dd])

        # amplitude
        if tmp_amp is not None:
            if md_main is None:
                raise ValueError("`amplitude` requires `memory_days`.")
            md_main_b = _broadcast(md_main, float)
            res["log_amplitude"] = jnp.array([
                get_log_amplitude(a, m) for a, m in zip(tmp_amp, md_main_b)
            ])

        return res


tree_util.register_pytree_node(
    FlexibleChannelPool,
    FlexibleChannelPool._tree_flatten,
    FlexibleChannelPool._tree_unflatten,
)
