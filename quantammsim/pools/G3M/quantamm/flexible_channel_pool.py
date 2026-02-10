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
    price_gradient: jnp.ndarray,  # (S, N)
    k: jnp.ndarray,  # (S, N)
    width_env: jnp.ndarray,  # (S, N)  σ_env
    amplitude: jnp.ndarray,  # (S, N)
    alpha: jnp.ndarray,  # (S, N)
    exponents_up: jnp.ndarray,  # (S, N)
    exponents_down: jnp.ndarray,  # (S, N)
    risk_off: jnp.ndarray,  # (S, N)  ρ_off
    risk_on: jnp.ndarray,  # (S, N)   ρ_on
    profit_pos: jnp.ndarray,  # (S, 1) Π⁺
    drawdown_neg: jnp.ndarray,  # (S, 1) Π⁻
    inverse_scaling: float = 0.5415,
    pre_exp_scaling: jnp.ndarray = 0.5,
    trend_base_cap: float = 50.0,
) -> jnp.ndarray:
    """
    Mean-reversion + trend-following + profit-harvest + risk-on amplifier.
    Returns Δw (weight changes), not absolute weights.
    """
    # 1) Envelope ------------------------------------------------------
    envelope = jnp.exp(-(price_gradient**2) / (2.0 * width_env**2))

    # 2) Cubic channel -------------------------------------------------
    width_poly = width_env / alpha
    s = (jnp.pi * price_gradient) / (3.0 * width_poly)
    channel = -amplitude * envelope * (s - s**3 / 6.0) / inverse_scaling

    # 3) Bare trend-following -----------------------------------------
    base = jnp.abs(price_gradient / (2.0 * pre_exp_scaling))
    base = jnp.minimum(base, trend_base_cap)  # risk control against power blow-ups
    exp_choose = jnp.where(price_gradient > 0, exponents_up, exponents_down)
    trend_bare = (1.0 - envelope) * jnp.sign(price_gradient) * jnp.power(base, exp_choose)

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
    FlexibleChannelPool: mean-reversion channel + trend + portfolio-gated risk controls.

    This implementation is designed to be:
      - causal (no "future data" leakage),
      - dimensionally consistent for portfolio-level Π signals,
      - robust in the presence of a stablecoin leg (e.g., USDC),
      - bounded via optional risk controls (k caps, vol floors, Π caps, Δw caps),
      - compatible with the existing MomentumPool / TFMM interface (returns Δw).
    """

    def __init__(self):
        super().__init__()

    # ─────────────────────────────────────────────────────────────────────────────
    #  calculate_raw_weights_outputs (history-aware, causal version)
    # ─────────────────────────────────────────────────────────────────────────────
    @partial(jit, static_argnums=(2,))
    def calculate_raw_weights_outputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,  # (T,) or (T,N)
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        fp, chunk = run_fingerprint, run_fingerprint["chunk_period"]

        # ── prices → (T, N) -----------------------------------------------------
        prices = prices[:, None] if prices.ndim == 1 else prices
        T, N = prices.shape
        if T < 2:
            return jnp.zeros((0, N), dtype=prices.dtype)

        # ── single parameter-set (S == 1) --------------------------------------
        log_k = jnp.squeeze(jnp.asarray(params["log_k"]))  # (N,)

        ln2 = jnp.log(2.0)

        # positive-only 2**squareplus(x)
        pow2_pos = lambda x: jnp.exp(jnp.clip(squareplus(x) * ln2, -60.0, 60.0))
        # literal 2**x (allows <1)
        pow2 = lambda x: jnp.exp(jnp.clip(x * ln2, -60.0, 60.0))

        # ── Optional risk controls (safe defaults) ------------------------------
        rc = fp.get("risk_controls", fp.get("strategy_risk_controls", {})) or {}
        sigma_floor = float(rc.get("sigma_floor", 1e-4))   # floor on σ̂ for stablecoin robustness
        sigma_cap = rc.get("sigma_cap", None)              # optional hard cap on σ̂
        k_max = rc.get("k_max", 10.0)                      # soft cap on k to avoid domination of offset by low-vol assets
        dw_max = rc.get("dw_max", None)                    # optional smooth cap on |Δw|
        pi_scale = float(rc.get("pi_scale", 1.0))          # scale Π signals
        pi_cap = rc.get("pi_cap", 0.10)                    # smooth cap Π signals
        trend_base_cap = float(rc.get("trend_base_cap", 50.0))
        freeze_risk_logits = bool(
            rc.get(
                "freeze_risk_logits",
                fp.get("freeze_risk_logits", fp.get("optimisation_settings", {}).get("freeze_risk_logits", False)),
            )
        )
        use_entropy_shrink = bool(rc.get("use_entropy_shrink", True))

        def soft_cap(x, cap):
            if cap is None:
                return x
            cap = jnp.asarray(cap, dtype=x.dtype)
            return cap * jnp.tanh(x / cap)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 1) Realised σ̂ᵢ(t): causal EWMA volatility (trainable λσᵢ)       │
        # ╰──────────────────────────────────────────────────────────────────╯
        chunk_p = prices[::chunk]  # (Tʹ, N)
        if chunk_p.shape[0] < 2:
            return jnp.zeros((0, N), dtype=prices.dtype)
        ret = jnp.log(chunk_p[1:] / chunk_p[:-1])          # (Tʹ-1, N)
        ret_sq = jnp.square(ret)

        λσ = jnn.sigmoid(params["logit_lamb_vol"]).reshape(N,)  # (N,)

        def ewma_var_ts(series: jnp.ndarray, lam: float) -> jnp.ndarray:
            # series: (Tʹ-1,)
            def step(c, x):
                c_new = c * lam + x * (1.0 - lam)
                return c_new, c_new

            init = series[0]
            _, out = lax.scan(step, init, series[1:])
            var_post = jnp.concatenate([init[None], out], axis=0)  # (Tʹ-1,)
            # causal shift: for decision at t, use variance up to t-1 (except t=0)
            var_causal = jnp.concatenate([var_post[:1], var_post[:-1]], axis=0)
            return var_causal

        var_ts = vmap(ewma_var_ts, in_axes=(1, 0))(ret_sq, λσ).T  # (Tʹ-1, N)
        σ̂_ts = jnp.sqrt(var_ts + 1e-12)

        # vol floors/caps (risk control)
        σ_max = jnp.asarray(sigma_cap, dtype=σ̂_ts.dtype) if sigma_cap is not None else jnp.inf
        σ̂_ts = jnp.clip(σ̂_ts, sigma_floor, σ_max)
        σ̂ = σ̂_ts[-1]  # last available σ̂ (N,)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 2) memory_daysᵢ & kᵢ(t): Kelly-scaled, vol-normalised, capped     │
        # ╰──────────────────────────────────────────────────────────────────╯
        mem_days = jnp.squeeze(
            lamb_to_memory_days_clipped(calc_lamb(params), chunk, fp["max_memory_days"])
        )  # (N,)

        k_plain = (2.0**log_k) / jnp.clip(mem_days, 1e-3, None)  # (N,)
        κ = pow2_pos(params["raw_kelly_kappa"]).reshape(N,)        # (N,)
        # time-varying k(t) to account for changing σ̂(t)
        k_ts = (k_plain[None, :] * κ[None, :]) / σ̂_ts             # (Tʹ-1, N)
        k_ts = jnp.clip(k_ts, 0.0, None)
        k_ts = soft_cap(k_ts, k_max)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 3) Per-token knobs (width, alpha, exponents, amplitude, β)       │
        # ╰──────────────────────────────────────────────────────────────────╯
        width_env_ts = pow2_pos(params["raw_width"]).reshape(N,) * σ̂_ts  # (Tʹ-1, N)
        alpha = pow2_pos(params["raw_alpha"]).reshape(N,)                # (N,)
        exp_up = squareplus(params["raw_exponents_up"]).reshape(N,)       # (N,)
        exp_dn = squareplus(params["raw_exponents_down"]).reshape(N,)     # (N,)

        # Step 5 fix: `log_amplitude` is literal log2(A), allowing A < 1.
        amp_raw = pow2(params["log_amplitude"]).reshape(-1)
        amp_raw = jnp.full((N,), amp_raw.item()) if amp_raw.size == 1 else amp_raw
        amplitude = amp_raw * mem_days  # (N,)

        # pre-exp scaling β
        if fp.get("use_pre_exp_scaling", True) and params.get("raw_pre_exp_scaling") is not None:
            β = pow2_pos(params["raw_pre_exp_scaling"]).reshape(-1)
            β = jnp.full((N,), β.item()) if β.size == 1 else β
        else:
            β = jnp.full((N,), 0.5, dtype=prices.dtype)

        ρ_off = jnn.sigmoid(stop_gradient(params["logit_risk_off"])).reshape(N,)
        ρ_on = jnn.sigmoid(params["logit_risk_on"]).reshape(N,)
        if freeze_risk_logits:
            ρ_on = stop_gradient(ρ_on)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 4) Full time-series EWMA-gradients (one per chunk)               │
        # ╰──────────────────────────────────────────────────────────────────╯
        grad_ts = calc_gradients(
            params,
            chunk_p,
            chunk,
            fp["max_memory_days"],
            fp["use_alt_lamb"],
            cap_lamb=True,
        )  # (Tʹ-1, N)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 5) Portfolio gating Π: portfolio-return based, bounded            │
        # ╰──────────────────────────────────────────────────────────────────╯
        pi_mode = fp.get("pi_mode", rc.get("pi_mode", "weighted_prev"))

        w_prev = jnp.asarray(fp.get("prev_weights", jnp.full((N,), 1.0 / N, dtype=prices.dtype)))
        w_prev = jnp.clip(w_prev, 0.0)
        w_prev_sum = jnp.sum(w_prev)
        w_prev = jnp.where(
            w_prev_sum > 0.0,
            w_prev / w_prev_sum,
            jnp.full((N,), 1.0 / N, dtype=prices.dtype),
        )

        w_ts = None
        if pi_mode in ("weights_ts", "prev_weights_ts"):
            w_ts = fp.get("weights_ts", None)
            if w_ts is None:
                w_ts = fp.get("prev_weights_ts", None)

        if w_ts is not None:
            w_ts = jnp.asarray(w_ts)
            # Accept (T,N), (Tʹ,N), (Tʹ-1,N), or (N,)
            if w_ts.ndim == 1:
                w_ts = jnp.broadcast_to(w_ts[None, :], (ret.shape[0], N))
            elif w_ts.shape[0] == prices.shape[0]:
                w_ts = w_ts[::chunk][:-1]
            elif w_ts.shape[0] == chunk_p.shape[0]:
                w_ts = w_ts[:-1]
            elif w_ts.shape[0] != ret.shape[0]:
                w_ts = jnp.broadcast_to(w_prev[None, :], (ret.shape[0], N))
            w_ts = jnp.clip(w_ts, 0.0)
            w_ts_den = jnp.sum(w_ts, axis=-1, keepdims=True)
            w_ts = jnp.where(w_ts_den > 0.0, w_ts / w_ts_den, jnp.full_like(w_ts, 1.0 / N))
        else:
            if pi_mode == "equal":
                w_ts = jnp.full((ret.shape[0], N), 1.0 / N, dtype=prices.dtype)
            else:  # "weighted_prev" default
                w_ts = jnp.broadcast_to(w_prev[None, :], (ret.shape[0], N))
        # Strictly causal portfolio weights for return interval t: use weights from t-1.
        w_ts = jnp.concatenate([w_prev[None, :], w_ts[:-1]], axis=0)

        # portfolio log-return and synthetic price series
        r_p = jnp.sum(w_ts * ret, axis=-1, keepdims=True)  # (Tʹ-1, 1)
        port_price = jnp.concatenate(
            [jnp.ones((1, 1), dtype=prices.dtype), jnp.exp(jnp.cumsum(r_p, axis=0))],
            axis=0,
        )  # (Tʹ, 1)
        port_rep = jnp.repeat(port_price, N, axis=1)  # (Tʹ, N) for estimator API

        # Π⁺ (profit) under main memory
        port_grad = calc_gradients(
            params,
            port_rep,
            chunk,
            fp["max_memory_days"],
            fp["use_alt_lamb"],
            cap_lamb=True,
        )  # (Tʹ-1, N)
        Π_pos_ts = jnp.maximum(0.0, port_grad.mean(axis=1, keepdims=True))  # (Tʹ-1, 1)

        # Π⁻ (drawdown) under drawdown memory
        params_dd = dict(params, logit_lamb=params["logit_lamb_drawdown"])
        dd_grad = calc_gradients(
            params_dd,
            port_rep,
            chunk,
            fp["max_memory_days"],
            fp["use_alt_lamb"],
            cap_lamb=True,
        )  # (Tʹ-1, N)
        Π_dd_ts = jnp.maximum(0.0, -dd_grad.mean(axis=1, keepdims=True))  # (Tʹ-1, 1)

        # bound Π terms (risk control)
        Π_pos_ts = soft_cap(pi_scale * Π_pos_ts, pi_cap)
        Π_dd_ts = soft_cap(pi_scale * Π_dd_ts, pi_cap)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 6) Kernel over the whole history                                 │
        # ╰──────────────────────────────────────────────────────────────────╯
        def kernel_t(pg, k_row, width_row, Πp, Πd):
            return _jax_flexible_channel_weight_update(
                pg[None, :],
                k_row[None, :],
                width_row[None, :],
                amplitude[None, :],
                alpha[None, :],
                exp_up[None, :],
                exp_dn[None, :],
                ρ_off[None, :],
                ρ_on[None, :],
                Πp[None, :],
                Πd[None, :],
                pre_exp_scaling=β[None, :],
                trend_base_cap=trend_base_cap,
            )[0]

        raw_ts = vmap(kernel_t)(grad_ts, k_ts, width_env_ts, Π_pos_ts, Π_dd_ts)  # (Tʹ-1, N)
        if dw_max is not None:
            raw_ts = soft_cap(raw_ts, dw_max)

        # ╭──────────────────────────────────────────────────────────────────╮
        # │ 7) Entropy guard-rail (step 9 fix: consistent shrink + state)     │
        # ╰──────────────────────────────────────────────────────────────────╯
        eps = 1e-12
        H_min = jnn.softplus(params["raw_entropy_floor"]).mean()  # scalar
        prev0 = fp.get("prev_weights", jnp.full((N,), 1.0 / N, dtype=raw_ts.dtype))

        def shrink(prev, dw):
            # propose
            w_prop = jnp.clip(prev + dw, eps)
            w_prop = w_prop / jnp.sum(w_prop)
            ent = -jnp.sum(w_prop * jnp.log(w_prop + eps))

            if use_entropy_shrink:
                gamma = jnp.minimum(jnp.sqrt(jnp.clip(ent / (H_min + eps), 0.0)), 1.0)
            else:
                gamma = 1.0

            dw_adj = dw * gamma
            w_new = jnp.clip(prev + dw_adj, eps)
            w_new = w_new / jnp.sum(w_new)
            return w_new, dw_adj

        _, shrunk = lax.scan(shrink, prev0, raw_ts)  # (Tʹ-1, N)
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
        """
        np.random.seed(0)
        eps = 1e-6  # for prob→logit clipping

        opt = run_fingerprint.get("optimisation_settings", {}) or {}
        force_scalar = bool(opt.get("force_scalar", False))

        # -----------------------------------------------------------------
        # Helpers
        # -----------------------------------------------------------------
        def _require(key: str):
            if key not in initial_values_dict:
                raise ValueError(f"initial_values_dict missing '{key}'")
            return initial_values_dict[key]

        def _expand(key: str, *, force_scalar: bool = False) -> np.ndarray:
            """Broadcast scalar / (N,) / (S,N) to (S,N)."""
            val = np.asarray(_require(key))

            if force_scalar:
                scalar = val.item() if val.size == 1 else float(np.asarray(val).squeeze())
                return np.full((n_parameter_sets, n_assets), scalar, dtype=float).copy()

            if val.size == 1:
                val = np.full((n_assets,), float(val.item()), dtype=float)
            if val.shape == (n_assets,):
                val = np.stack([val] * n_parameter_sets, axis=0)
            if val.shape != (n_parameter_sets, n_assets):
                raise ValueError(
                    f"'{key}' must be scalar, len-{n_assets} vector, or "
                    f"({n_parameter_sets},{n_assets}) array"
                )
            return np.asarray(val, dtype=float).copy()

        def _expand_memory_days(key: str, *, default=None) -> np.ndarray:
            """
            Broadcast memory-day inputs to:
            - (S,1) if force_scalar
            - (S,N) otherwise
            Accepts scalar, (N,), (S,), (S,1), (S,N).
            """
            if key in initial_values_dict:
                raw = initial_values_dict[key]
            else:
                if default is None:
                    raise ValueError(f"initial_values_dict missing '{key}'")
                raw = default

            arr = np.asarray(raw, dtype=float)

            # scalar
            if arr.ndim == 0:
                if force_scalar:
                    return np.full((n_parameter_sets, 1), float(arr), dtype=float)
                return np.full((n_parameter_sets, n_assets), float(arr), dtype=float)

            # 1D
            if arr.ndim == 1:
                if arr.shape == (n_assets,):
                    if force_scalar:
                        raise ValueError(
                            f"'{key}' provided per-asset, but force_scalar=True requires scalar or (S,) only."
                        )
                    return np.stack([arr] * n_parameter_sets, axis=0)

                if arr.shape == (n_parameter_sets,):
                    # per-set scalar
                    if force_scalar:
                        return arr.reshape(n_parameter_sets, 1)
                    return np.repeat(arr.reshape(n_parameter_sets, 1), n_assets, axis=1)

                if arr.size == 1:
                    # degenerate scalar-in-list
                    val = float(arr.item())
                    if force_scalar:
                        return np.full((n_parameter_sets, 1), val, dtype=float)
                    return np.full((n_parameter_sets, n_assets), val, dtype=float)

                raise ValueError(
                    f"'{key}' 1D input must be scalar, (N,) or (S,). Got shape {arr.shape}."
                )

            # 2D
            if arr.ndim == 2:
                if arr.shape == (n_parameter_sets, 1):
                    if force_scalar:
                        return arr
                    return np.repeat(arr, n_assets, axis=1)

                if arr.shape == (n_parameter_sets, n_assets):
                    if force_scalar:
                        raise ValueError(
                            f"'{key}' provided as (S,N), but force_scalar=True requires scalar or (S,1)/(S,)."
                        )
                    return arr

                raise ValueError(
                    f"'{key}' 2D input must be (S,1) or (S,N). Got shape {arr.shape}."
                )

            raise ValueError(f"'{key}' has invalid ndim={arr.ndim} (expected 0,1,2).")

        def _logit_lambda_from_memdays(mem_days_arr: np.ndarray, chunk: int) -> np.ndarray:
            """
            Convert memory_days -> λ -> logit(λ), safely for arrays.
            """
            l = memory_days_to_lamb(mem_days_arr, chunk)
            l = np.clip(l, eps, 1.0 - eps)
            return np.log(l / (1.0 - l))

        # -----------------------------------------------------------------
        # 1) k
        # -----------------------------------------------------------------
        log_k = np.log2(_expand("initial_k_per_day", force_scalar=force_scalar))

        # -----------------------------------------------------------------
        # 2) main λ + Δλ  (supports scalar or vector memory lengths & deltas)
        # -----------------------------------------------------------------
        chunk = int(run_fingerprint["chunk_period"])

        init_mem_days = _expand_memory_days("initial_memory_length")
        delta_days = _expand_memory_days("initial_memory_length_delta")

        # ensure strictly positive mem-days after delta
        mem_plus = np.maximum(init_mem_days + delta_days, 1e-9)

        logit_lamb = _logit_lambda_from_memdays(init_mem_days, chunk)
        logit_lamb_plus = _logit_lambda_from_memdays(mem_plus, chunk)
        logit_delta_lamb = (logit_lamb_plus - logit_lamb).copy()

        # final shape for lamb params (match prior convention)
        if force_scalar:
            # keep (S,1)
            logit_lamb = np.broadcast_to(logit_lamb, (n_parameter_sets, 1)).copy()
            logit_delta_lamb = np.broadcast_to(logit_delta_lamb, (n_parameter_sets, 1)).copy()
        else:
            logit_lamb = np.broadcast_to(logit_lamb, (n_parameter_sets, n_assets)).copy()
            logit_delta_lamb = np.broadcast_to(logit_delta_lamb, (n_parameter_sets, n_assets)).copy()

        # -----------------------------------------------------------------
        # 3) draw-down λᴅ  (fallback to main memory_days)
        # -----------------------------------------------------------------
        mem_dd_days = _expand_memory_days("initial_memory_length_drawdown", default=init_mem_days)
        logit_lamb_drawdown = _logit_lambda_from_memdays(mem_dd_days, chunk)

        if force_scalar:
            logit_lamb_drawdown = np.broadcast_to(logit_lamb_drawdown, (n_parameter_sets, 1)).copy()
        else:
            logit_lamb_drawdown = np.broadcast_to(logit_lamb_drawdown, (n_parameter_sets, n_assets)).copy()

        # -----------------------------------------------------------------
        # 4) per-token raw params (already internal/raw in your config naming)
        # -----------------------------------------------------------------
        log_amplitude = _expand("initial_log_amplitude", force_scalar=force_scalar)
        raw_width = _expand("initial_raw_width", force_scalar=force_scalar)
        raw_alpha = _expand("initial_raw_alpha", force_scalar=force_scalar)
        raw_exp_up = _expand("initial_raw_exponents_up", force_scalar=force_scalar)
        raw_exp_down = _expand("initial_raw_exponents_down", force_scalar=force_scalar)

        # pre-exp scaling β is an external positive knob; store in raw space consistently
        pre_exp = float(np.asarray(_require("initial_pre_exp_scaling"), dtype=float).squeeze())
        pre_exp = max(pre_exp, 1e-12)
        raw_pre = inverse_squareplus_np(np.log2(pre_exp))  # consistent with pow2_pos(squareplus(raw))
        raw_pre_exp_scaling = np.full(
            (n_parameter_sets, 1 if force_scalar else n_assets),
            raw_pre,
            dtype=float,
        ).copy()

        # -----------------------------------------------------------------
        # 5) risk prob → logit (clipped)
        # -----------------------------------------------------------------
        risk_off_prob = np.clip(_expand("initial_risk_off"), eps, 1.0 - eps)
        risk_on_prob = np.clip(_expand("initial_risk_on"), eps, 1.0 - eps)
        logit_risk_off = np.log(risk_off_prob / (1.0 - risk_off_prob))
        logit_risk_on = np.log(risk_on_prob / (1.0 - risk_on_prob))

        # -----------------------------------------------------------------
        # 6) weight logits
        # -----------------------------------------------------------------
        initial_weights_logits = _expand("initial_weights_logits")

        # -----------------------------------------------------------------
        # 7) additional trainables
        # -----------------------------------------------------------------
        raw_kelly_kappa = _expand("initial_raw_kelly_kappa", force_scalar=force_scalar)
        logit_lamb_vol = _expand("initial_logit_lamb_vol", force_scalar=force_scalar)
        raw_entropy_floor = _expand("initial_raw_entropy_floor", force_scalar=True)

        # -----------------------------------------------------------------
        # 8) pack dict
        # -----------------------------------------------------------------
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
    #  _process_specific_parameters
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

        for p in update_rule_parameters:
            if p.name == "memory_days":
                md_main = p.value
            elif p.name == "memory_days_drawdown":
                md_dd = p.value

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
                res["raw_entropy_floor"] = jnp.full((S_sets, 1), val)
            elif n == "amplitude":
                tmp_amp = _broadcast(v, float)

        if md_dd is not None:
            lambs_dd = [memory_days_to_lamb(d, cp) for d in _broadcast(md_dd, float)]
            res["logit_lamb_drawdown"] = jnp.array([np.log(l / (1 - l)) for l in lambs_dd])

        if tmp_amp is not None:
            if md_main is None:
                raise ValueError("`amplitude` requires `memory_days`.")
            md_main_b = _broadcast(md_main, float)
            res["log_amplitude"] = jnp.array([get_log_amplitude(a, m) for a, m in zip(tmp_amp, md_main_b)])

        return res


tree_util.register_pytree_node(
    FlexibleChannelPool,
    FlexibleChannelPool._tree_flatten,
    FlexibleChannelPool._tree_unflatten,
)
