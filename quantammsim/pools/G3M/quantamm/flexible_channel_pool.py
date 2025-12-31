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
    k: jnp.ndarray,               # shape (S, N)
    width_env: jnp.ndarray,       # σ_env
    amplitude: jnp.ndarray,       # (S, N) or (N,)
    alpha: jnp.ndarray,           # (S, N) or (N,)
    exponents_up: jnp.ndarray,    # (S, N) or (N,)
    exponents_down: jnp.ndarray,  # (S, N) or (N,)
    risk_off: jnp.ndarray,        # ρ_off  (S, N) or (N,)
    risk_on: jnp.ndarray,         # ρ_on   (S, N) or (N,)
    profit_pos: jnp.ndarray,      # Π⁺     (S, 1)
    drawdown_neg: jnp.ndarray,    # Π⁻     (S, 1)
    inverse_scaling: float = 0.5415,
    pre_exp_scaling: jnp.ndarray = 0.5,  # (S, N) or (N,)
) -> jnp.ndarray:
    """
    Mean-reversion + trend-following + profit-harvest + risk-on amplifier.
    S is typically 1 in this kernel usage.
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
    A mean-reversion channel strategy with causal, time-varying risk scaling and
    portfolio-aware gating.

    Optional run_fingerprint keys:
      - "pi_mode": one of {"weights", "risk", "kelly_proxy", "equal"}; default "weights".
      - "use_entropy_shrink": bool; default True.
    """

    def __init__(self):
        super().__init__()

    # ─────────────────────────────────────────────────────────────────────────────
    #  calculate_raw_weights_outputs  (history-aware & causal)
    # ─────────────────────────────────────────────────────────────────────────────
    @partial(jit, static_argnums=(2,))
    def calculate_raw_weights_outputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,  # (T ,) or (T ,N)
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        fp, chunk = run_fingerprint, run_fingerprint["chunk_period"]
        pi_mode = fp.get("pi_mode", "weights")
        use_entropy_shrink = fp.get("use_entropy_shrink", True)

        # ── prices → (T , N) ----------------------------------------------------
        prices = prices[:, None] if prices.ndim == 1 else prices
        T, N = prices.shape

        # helper ─ 2**x with overflow guard
        ln2 = jnp.log(2.0)
        pow2 = lambda x: jnp.exp(jnp.clip(squareplus(x) * ln2, -60.0, 60.0))
        eps = 1e-12

        # ── single parameter-set (S == 1) --------------------------------------
        log_k = jnp.squeeze(jnp.asarray(params["log_k"]))  # (N,)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 1. realised σ̂ᵢ(t)  as *causal* EWMA time series                │
        # │    (no look-ahead; used for width_envₜ and kₜ)                  │
        # ╰──────────────────────────────────────────────────────────────────╯
        chunk_p = prices[::chunk]                  # (Tʹ , N)
        ret_sq = jnp.square(jnp.log(chunk_p[1:] / chunk_p[:-1]))  # (Tʹ-1 , N)

        λσ = jnn.sigmoid(params["logit_lamb_vol"]).reshape(N)  # (N,)

        def ewma_ts(series_1d, lam):
            # series_1d: (T'-1,)
            def step(c, x):
                c_new = c * lam + x * (1.0 - lam)
                return c_new, c_new
            init = series_1d[0]
            _, out = lax.scan(step, init, series_1d[1:])
            out_full = jnp.concatenate([jnp.asarray([init]), out], axis=0)  # (T'-1,)
            return jnp.sqrt(out_full + 1e-12)

        # vmap across assets → (N, T'-1) → transpose to (T'-1, N)
        sigma_ts = vmap(ewma_ts)(ret_sq.T, λσ).T

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 2. per-token memory_daysᵢ & time-varying kₜ (Kelly scaled)      │
        # ╰──────────────────────────────────────────────────────────────────╯
        mem_days = jnp.squeeze(
            lamb_to_memory_days_clipped(calc_lamb(params), chunk, fp["max_memory_days"])
        )  # (N,)

        k_plain = (2.0**log_k) / jnp.clip(mem_days, 1e-3, None)  # (N,)
        κ = pow2(params["raw_kelly_kappa"]).reshape(N)           # (N,)
        # time-varying Kelly due to σ̂ₜ
        k_ts = (k_plain * κ)[None, :] / jnp.clip(sigma_ts, 1e-9, None)  # (T'-1, N)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 3. deterministic per-token knobs                                │
        # │    width_envₜ depends on σ̂ₜ; others remain per-asset constants │
        # ╰──────────────────────────────────────────────────────────────────╯
        width_env_ts = pow2(params["raw_width"]).reshape(1, N) * sigma_ts  # (T'-1, N)
        alpha = pow2(params["raw_alpha"]).reshape(N)                        # (N,)
        exp_up = squareplus(params["raw_exponents_up"]).reshape(N)          # (N,)
        exp_dn = squareplus(params["raw_exponents_down"]).reshape(N)        # (N,)

        amp_raw = pow2(params["log_amplitude"]).reshape(-1)
        amp_raw = jnp.full((N,), amp_raw.item()) if amp_raw.size == 1 else amp_raw
        amplitude = amp_raw * mem_days  # (N,)

        # pre-exp scaling, broadcast to time
        β = (
            jnn.sigmoid(params["logit_pre_exp_scaling"])
            if fp.get("use_pre_exp_scaling", False)
            and params.get("logit_pre_exp_scaling") is not None
            else (
                pow2(params["raw_pre_exp_scaling"])
                if fp.get("use_pre_exp_scaling", False)
                and params.get("raw_pre_exp_scaling") is not None
                else 0.5
            )
        )
        β = jnp.broadcast_to(jnp.asarray(β).reshape(1, -1 if jnp.ndim(β) > 0 else 1), (1, N))
        β_ts = jnp.broadcast_to(β, width_env_ts.shape)  # (T'-1, N)

        
        ρ_off = jnn.sigmoid(stop_gradient(params["logit_risk_off"])).reshape(N,)
        ρ_on  = jnn.sigmoid(stop_gradient(params["logit_risk_on"])).reshape(N,)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 4. full *time-series* EWMA-gradients (one per chunk)            │
        # ╰──────────────────────────────────────────────────────────────────╯
        grad_ts = calc_gradients(
            params,
            chunk_p,
            chunk,
            fp["max_memory_days"],
            fp["use_alt_lamb"],
            cap_lamb=True,
        )  # (Tʹ-1 , N)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 5. Portfolio gating Π from portfolio-weighted gradient          │
        # │    Modes: "weights" | "risk" | "kelly_proxy" | "equal"          │
        # ╰──────────────────────────────────────────────────────────────────╯
        w_prev = fp.get("prev_weights", jnp.full((N,), 1.0 / N, dtype=grad_ts.dtype))
        w_prev = w_prev / jnp.clip(jnp.sum(w_prev), eps, None)

        if pi_mode == "equal":
            w_for_pi_ts = jnp.broadcast_to(jnp.full((1, N), 1.0 / N, dtype=grad_ts.dtype), grad_ts.shape)
            port_grad_ts = jnp.sum(grad_ts * w_for_pi_ts, axis=1)
        elif pi_mode == "risk":
            # risk-weighted by contemporaneous σ̂ₜ, using lagged weights for exposure
            w_risk_ts = w_prev[None, :] * sigma_ts
            w_risk_ts = w_risk_ts / jnp.clip(jnp.sum(w_risk_ts, axis=1, keepdims=True), eps, None)
            port_grad_ts = jnp.sum(grad_ts * w_risk_ts, axis=1)
        elif pi_mode == "kelly_proxy":
            # contemporaneous Kelly vector proxy (causal)
            k_norm_ts = k_ts / jnp.clip(jnp.sum(k_ts, axis=1, keepdims=True), eps, None)
            port_grad_ts = jnp.sum(grad_ts * k_norm_ts, axis=1)
        else:  # "weights" (default)
            port_grad_ts = grad_ts @ w_prev

        Π_pos_ts = jnp.maximum(0.0, port_grad_ts)[:, None]  # (T'-1, 1)
        Π_dd_ts  = jnp.maximum(0.0, -port_grad_ts)[:, None] # (T'-1, 1)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 6. kernel over the *whole* history with time-varying inputs     │
        # ╰──────────────────────────────────────────────────────────────────╯
        def kernel_t(pg_t, k_t, width_t, Πp_t, Πd_t, β_t):
            return _jax_flexible_channel_weight_update(
                pg_t[None, :],
                k_t[None, :],
                width_t[None, :],
                amplitude[None, :],
                alpha[None, :],
                exp_up[None, :],
                exp_dn[None, :],
                ρ_off[None, :],
                ρ_on[None, :],
                Πp_t[None, :],
                Πd_t[None, :],
                pre_exp_scaling=β_t[None, :],
            )[0]

        raw_ts = vmap(kernel_t)(grad_ts, k_ts, width_env_ts, Π_pos_ts, Π_dd_ts, β_ts)  # (Tʹ-1 , N)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 7. entropy guard-rail (causal): shrink when ent < H_min         │
        # ╰──────────────────────────────────────────────────────────────────╯
        H_min = jnn.softplus(params["raw_entropy_floor"]).mean()  # scalar > 0
        prev0 = fp.get("prev_weights", jnp.full((N,), 1.0 / N, dtype=raw_ts.dtype))

        def shrink(prev, dw):
            w = jnp.clip(prev + dw, 1e-12)
            w = w / jnp.sum(w)
            ent = -jnp.sum(w * jnp.log(w + eps))
            if use_entropy_shrink:
                gamma = jnp.minimum(jnp.sqrt(jnp.clip(ent / (H_min + eps), 0.0)), 1.0)
                dw_shrunk = dw * gamma
            else:
                dw_shrunk = dw
            return w, dw_shrunk

        _, shrunk = lax.scan(shrink, prev0, raw_ts)  # (Tʹ-1 , N)
        return shrunk

    # ─────────────────────────────────────────────────────────────────
    #  init_base_parameters   (fully fixed; unchanged)
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
        Build the parameter dictionary in internal (trainable) form.
        All broadcasted arrays are copied to guarantee they are write-able.
        """
        np.random.seed(0)
        eps = 1e-6  # for prob→logit clipping
        force_scalar = run_fingerprint["optimisation_settings"]["force_scalar"]

        # -----------------------------------------------------------------
        # inside init_base_parameters  – replace _expand
        # -----------------------------------------------------------------
        def _expand(key: str, *, force_scalar: bool = False) -> np.ndarray:
            if key not in initial_values_dict:
                raise ValueError(f"initial_values_dict missing '{key}'")
            val = np.asarray(initial_values_dict[key])

            # ── scalar-per-set case → replicate across assets ───────────
            if force_scalar:
                scalar = val.item() if val.size == 1 else float(val.squeeze())
                return np.full(
                    (n_parameter_sets, n_assets), scalar, dtype=val.dtype
                ).copy()

            # ── normal broadcasting rules ───────────────────────────────
            if val.size == 1:
                val = np.full((n_assets,), val.item())
            if val.shape == (n_assets,):
                val = np.stack([val] * n_parameter_sets, axis=0)
            if val.shape != (n_parameter_sets, n_assets):
                raise ValueError(
                    f"'{key}' must be scalar, len-{n_assets} vector, or "
                    f"({n_parameter_sets},{n_assets}) array"
                )
            return val.copy()  # ensure write-able

        # 1) k
        log_k = np.log2(_expand("initial_k_per_day", force_scalar=force_scalar))

        # 2) main λ + Δλ
        chunk = run_fingerprint["chunk_period"]

        def _logit_lambda(mem_days):
            l = memory_days_to_lamb(mem_days, chunk)
            return np.log(l / (1 - l))

        init_mem = initial_values_dict["initial_memory_length"]
        logit_lamb = _logit_lambda(init_mem)
        logit_lamb = np.broadcast_to(
            logit_lamb, (n_parameter_sets, 1 if force_scalar else n_assets)
        ).copy()

        delta = initial_values_dict["initial_memory_length_delta"]
        logit_delta_lamb = _logit_lambda(init_mem + delta) - _logit_lambda(init_mem)
        logit_delta_lamb = np.broadcast_to(logit_delta_lamb, logit_lamb.shape).copy()

        # 3) draw-down λᴅ  (fallback to main λ)
        mem_dd = initial_values_dict.get("initial_memory_length_drawdown", init_mem)
        logit_lamb_drawdown = _logit_lambda(mem_dd)
        logit_lamb_drawdown = np.broadcast_to(
            logit_lamb_drawdown, logit_lamb.shape
        ).copy()

        # 4) per-token raw params
        log_amplitude = _expand("initial_log_amplitude", force_scalar=force_scalar)
        raw_width = _expand("initial_raw_width", force_scalar=force_scalar)
        raw_alpha = _expand("initial_raw_alpha", force_scalar=force_scalar)
        raw_exp_up = _expand("initial_raw_exponents_up", force_scalar=force_scalar)
        raw_exp_down = _expand("initial_raw_exponents_down", force_scalar=force_scalar)

        raw_pre_beta_np = np.log2(initial_values_dict["initial_pre_exp_scaling"])
        if force_scalar:
            raw_pre_exp_scaling = np.full((n_parameter_sets, 1), raw_pre_beta_np)
        else:
            raw_pre_exp_scaling = np.full((n_parameter_sets, n_assets), raw_pre_beta_np)
        raw_pre_exp_scaling = raw_pre_exp_scaling.copy()

        # 5) risk prob → logit  (clipped, mutable)
        risk_off_prob = np.clip(_expand("initial_risk_off"), eps, 1 - eps)
        risk_on_prob = np.clip(_expand("initial_risk_on"), eps, 1 - eps)
        logit_risk_off = np.log(risk_off_prob / (1 - risk_off_prob))
        logit_risk_on = np.log(risk_on_prob / (1 - risk_on_prob))

        # 6) weight logits
        initial_weights_logits = _expand("initial_weights_logits")

        # 7) new trainables
        raw_kelly_kappa = _expand("initial_raw_kelly_kappa", force_scalar=force_scalar)
        logit_lamb_vol = _expand("initial_logit_lamb_vol", force_scalar=force_scalar)
        raw_entropy_floor = _expand("initial_raw_entropy_floor", force_scalar=True)

        # 8) pack dict
        params = {
            "log_k": log_k,
            "logit_lamb": logit_lamb,
            "logit_delta_lamb": logit_delta_lamb,
            "logit_lamb_drawdown": logit_lamb_drawdown,
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
            "raw_entropy_floor": raw_entropy_floor,
            "subsidary_params": [],
        }

        # add Gaussian noise (expects arrays to be write-able)
        params = self.add_noise(params, noise, n_parameter_sets)
        return params

    # ─────────────────────────────────────────────────────────────────
    #  _process_specific_parameters  (entropy-floor shape fixed; unchanged)
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def _process_specific_parameters(
        cls,
        update_rule_parameters,
        run_fingerprint: Dict[str, Any],
    ) -> Dict[str, Any]:
        tokens = run_fingerprint["tokens"]
        n_tok = len(tokens)
        cp = run_fingerprint["chunk_period"]
        S_sets = run_fingerprint["n_parameter_sets"]

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
                res["raw_exponents_up"] = jnp.array(
                    _broadcast(v, inverse_squareplus_np)
                )
            elif n == "exponent_down":
                res["raw_exponents_down"] = jnp.array(
                    _broadcast(v, inverse_squareplus_np)
                )
            elif n == "pre_exp_scaling":
                res["raw_pre_exp_scaling"] = jnp.array(_broadcast(v, get_raw_value))
            elif n == "risk_off":
                probs = _broadcast(v, float)
                res["logit_risk_off"] = jnp.array([np.log(p / (1 - p)) for p in probs])
            elif n == "risk_on":
                probs = _broadcast(v, float)
                res["logit_risk_on"] = jnp.array([np.log(p / (1 - p)) for p in probs])
            elif n == "kelly_kappa":
                res["raw_kelly_kappa"] = jnp.array(_broadcast(v, get_raw_value))
            elif n == "vol_memory_days":
                lambs = [memory_days_to_lamb(d, cp) for d in _broadcast(v, float)]
                res["logit_lamb_vol"] = jnp.array([np.log(l / (1 - l)) for l in lambs])
            elif n == "entropy_floor":
                val = float(get_raw_value(v[0] if isinstance(v, (list, tuple)) else v))
                res["raw_entropy_floor"] = jnp.full((S_sets, 1), val)  # (S,1)
            elif n == "amplitude":
                tmp_amp = _broadcast(v, float)

        # draw-down λᴅ
        if md_dd is not None:
            lambs_dd = [memory_days_to_lamb(d, cp) for d in _broadcast(md_dd, float)]
            res["logit_lamb_drawdown"] = jnp.array(
                [np.log(l / (1 - l)) for l in lambs_dd]
            )

        # amplitude
        if tmp_amp is not None:
            if md_main is None:
                raise ValueError("`amplitude` requires `memory_days`.")
            md_main_b = _broadcast(md_main, float)
            res["log_amplitude"] = jnp.array(
                [get_log_amplitude(a, m) for a, m in zip(tmp_amp, md_main_b)]
            )

        return res


tree_util.register_pytree_node(
    FlexibleChannelPool,
    FlexibleChannelPool._tree_flatten,
    FlexibleChannelPool._tree_unflatten,
)
