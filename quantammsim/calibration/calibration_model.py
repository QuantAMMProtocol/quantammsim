"""Composable CalibrationModel with pluggable Head components.

The CalibrationModel coordinates three heads (cadence, gas, noise) and
provides:
  - Parameter packing/unpacking across all heads
  - Per-pool JIT-compiled loss closures (same pattern as existing code)
  - Joint loss aggregation with head regularization
  - scipy L-BFGS-B fitting for both per-pool and joint modes
  - Prediction for new pools

All heads are concatenated in order [cadence | gas | noise] in the flat
parameter vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
from quantammsim.calibration.heads import Head
from quantammsim.calibration.loss import K_OBS, noise_volume


@dataclass
class CalibrationModel:
    """Composable calibration model with pluggable heads.

    Coordinates cadence_head, gas_head, and noise_head to build a single
    flat parameter vector and produce per-pool JIT-compiled loss functions.
    """

    cadence_head: Head
    gas_head: Head
    noise_head: Head
    loss_type: str = "l2"
    huber_delta: float = 1.5

    # ── Parameter geometry ─────────────────────────────────────────────

    def n_params(self, n_pools: int, k_attr: int) -> int:
        """Total parameter count across all heads."""
        return (
            self.cadence_head.n_params(n_pools, k_attr)
            + self.gas_head.n_params(n_pools, k_attr)
            + self.noise_head.n_params(n_pools, k_attr)
        )

    def _head_slices(self, n_pools: int, k_attr: int):
        """Return (start, end) index pairs for each head's param slice."""
        n_cad = self.cadence_head.n_params(n_pools, k_attr)
        n_gas = self.gas_head.n_params(n_pools, k_attr)
        n_noise = self.noise_head.n_params(n_pools, k_attr)
        cad_end = n_cad
        gas_end = cad_end + n_gas
        noise_end = gas_end + n_noise
        return (0, cad_end), (cad_end, gas_end), (gas_end, noise_end)

    # ── Initialization ─────────────────────────────────────────────────

    def pack_init(self, jdata, warm_start=None) -> np.ndarray:
        """Concatenate head inits into a single flat NumPy vector."""
        cad_init = self.cadence_head.init(jdata, warm_start)
        gas_init = self.gas_head.init(jdata, warm_start)
        noise_init = self.noise_head.init(jdata, warm_start)
        return np.concatenate([cad_init, gas_init, noise_init])

    # ── Bounds ─────────────────────────────────────────────────────────

    def make_bounds(self, n_pools: int, k_attr: int) -> list:
        """Concatenate per-head scipy bounds."""
        return (
            self.cadence_head.make_bounds(n_pools, k_attr)
            + self.gas_head.make_bounds(n_pools, k_attr)
            + self.noise_head.make_bounds(n_pools, k_attr)
        )

    # ── Loss functions ─────────────────────────────────────────────────

    def _compute_loss(self, residuals: jnp.ndarray) -> jnp.ndarray:
        """Compute loss from residuals based on loss_type."""
        if self.loss_type == "huber":
            delta = self.huber_delta
            abs_r = jnp.abs(residuals)
            huber = jnp.where(
                abs_r <= delta,
                0.5 * residuals ** 2,
                delta * (abs_r - 0.5 * delta),
            )
            return jnp.mean(huber)
        return jnp.mean(residuals ** 2)

    def make_pool_loss_fn(
        self,
        pool_idx: int,
        pool_data_i: dict,
        x_attr_i: jnp.ndarray,
        n_pools: int,
        k_attr: int,
    ) -> Callable:
        """Create a JIT-compiled loss function for a single pool.

        Closes over pool-specific data.  Takes params_flat as sole argument.
        Returns scalar loss (no regularization — that's added at aggregate level).
        """
        coeffs = pool_data_i["coeffs"]
        x_obs = pool_data_i["x_obs"]
        y_obs = pool_data_i["y_obs"]
        day_indices = pool_data_i["day_indices"]

        (cad_s, cad_e), (gas_s, gas_e), (noise_s, noise_e) = \
            self._head_slices(n_pools, k_attr)

        cad_head = self.cadence_head
        gas_head = self.gas_head
        noise_head = self.noise_head
        compute_loss = self._compute_loss
        i = pool_idx

        @jax.jit
        def pool_loss_fn(params_flat):
            cad_slice = params_flat[cad_s:cad_e]
            gas_slice = params_flat[gas_s:gas_e]
            noise_slice = params_flat[noise_s:noise_e]

            log_cad = cad_head.predict(cad_slice, i, x_attr_i)
            log_gas = gas_head.predict(gas_slice, i, x_attr_i)
            noise_c = noise_head.predict(noise_slice, i, x_attr_i)

            v_arb_all = interpolate_pool_daily(
                coeffs, log_cad, jnp.exp(log_gas)
            )
            v_arb = v_arb_all[day_indices]
            v_noise = jnp.exp(x_obs @ noise_c)
            log_v_pred = jnp.log(jnp.maximum(v_arb + v_noise, 1e-6))

            return compute_loss(log_v_pred - y_obs)

        return pool_loss_fn

    def make_joint_loss_fn(self, jdata) -> Callable:
        """Create the joint loss function over all pools.

        Returns loss_fn(params_flat) -> scalar.  Also attaches helper
        attributes for the scipy wrapper (_pool_val_and_grad_fns, etc.).
        """
        n_pools = len(jdata.pool_data)
        k_attr = jdata.x_attr.shape[1]

        pool_loss_fns = []
        pool_val_and_grad_fns = []
        for i in range(n_pools):
            fn = self.make_pool_loss_fn(
                i, jdata.pool_data[i], jdata.x_attr[i], n_pools, k_attr
            )
            pool_loss_fns.append(fn)
            pool_val_and_grad_fns.append(jax.value_and_grad(fn))

        (cad_s, cad_e), (gas_s, gas_e), (noise_s, noise_e) = \
            self._head_slices(n_pools, k_attr)

        cad_head = self.cadence_head
        gas_head = self.gas_head
        noise_head = self.noise_head

        def loss_fn(params_flat):
            total = sum(fn(params_flat) for fn in pool_loss_fns)
            data_loss = total / n_pools

            reg = cad_head.regularization(params_flat[cad_s:cad_e])
            reg = reg + gas_head.regularization(params_flat[gas_s:gas_e])
            reg = reg + noise_head.regularization(params_flat[noise_s:noise_e])

            return data_loss + reg

        # Attach for the scipy wrapper
        loss_fn._pool_val_and_grad_fns = pool_val_and_grad_fns
        loss_fn._n_pools = n_pools
        loss_fn._head_slices = (cad_s, cad_e), (gas_s, gas_e), (noise_s, noise_e)
        loss_fn._cad_head = cad_head
        loss_fn._gas_head = gas_head
        loss_fn._noise_head = noise_head

        return loss_fn

    # ── Fitting ────────────────────────────────────────────────────────

    def fit(
        self,
        jdata,
        maxiter: int = 500,
        warm_start: Optional[Dict[str, dict]] = None,
    ) -> dict:
        """Fit the model on joint data via L-BFGS-B.

        Returns a result dict with fitted parameters and diagnostics.
        """
        n_pools = len(jdata.pool_data)
        k_attr = jdata.x_attr.shape[1]

        loss_fn = self.make_joint_loss_fn(jdata)
        init = self.pack_init(jdata, warm_start)
        bounds = self.make_bounds(n_pools, k_attr)

        pool_vg_fns = loss_fn._pool_val_and_grad_fns
        (cad_s, cad_e), (gas_s, gas_e), (noise_s, noise_e) = \
            loss_fn._head_slices

        cad_head = self.cadence_head
        gas_head = self.gas_head
        noise_head = self.noise_head

        def scipy_wrapper(params_np):
            params_j = jnp.array(params_np)

            total_val = 0.0
            total_grad = jnp.zeros_like(params_j)
            for vg_fn in pool_vg_fns:
                v, g = vg_fn(params_j)
                total_val += float(v)
                total_grad = total_grad + g

            data_loss = total_val / n_pools
            data_grad = total_grad / n_pools

            # Regularization — compute value and gradient
            reg_val = 0.0
            reg_grad = jnp.zeros_like(params_j)

            # Cadence head regularization
            cad_slice = params_j[cad_s:cad_e]
            if cad_e > cad_s:
                cad_reg_fn = lambda p: cad_head.regularization(p)
                cr = float(cad_reg_fn(cad_slice))
                if cr != 0.0:
                    cad_rg = jax.grad(cad_reg_fn)(cad_slice)
                    reg_val += cr
                    reg_grad = reg_grad.at[cad_s:cad_e].set(cad_rg)

            # Gas head regularization
            gas_slice = params_j[gas_s:gas_e]
            if gas_e > gas_s:
                gas_reg_fn = lambda p: gas_head.regularization(p)
                gr = float(gas_reg_fn(gas_slice))
                if gr != 0.0:
                    gas_rg = jax.grad(gas_reg_fn)(gas_slice)
                    reg_val += gr
                    reg_grad = reg_grad.at[gas_s:gas_e].set(gas_rg)

            # Noise head regularization
            noise_slice = params_j[noise_s:noise_e]
            if noise_e > noise_s:
                noise_reg_fn = lambda p: noise_head.regularization(p)
                nr = float(noise_reg_fn(noise_slice))
                if nr != 0.0:
                    noise_rg = jax.grad(noise_reg_fn)(noise_slice)
                    reg_val += nr
                    reg_grad = reg_grad.at[noise_s:noise_e].set(noise_rg)

            val = data_loss + reg_val
            grad = data_grad + reg_grad
            return val, np.array(grad, dtype=np.float64)

        init_np = np.array(init, dtype=np.float64)
        init_loss = float(loss_fn(jnp.array(init_np)))

        result = scipy.optimize.minimize(
            scipy_wrapper,
            init_np,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8},
        )

        fitted = jnp.array(result.x)
        (cad_s, cad_e), (gas_s, gas_e), (noise_s, noise_e) = \
            self._head_slices(n_pools, k_attr)

        out = {
            "init_loss": init_loss,
            "loss": float(result.fun),
            "converged": result.success,
            "params_flat": np.array(result.x),
        }

        # Unpack each head's result
        out.update(self.cadence_head.unpack_result(
            np.array(fitted[cad_s:cad_e]), n_pools, k_attr))
        out.update(self.gas_head.unpack_result(
            np.array(fitted[gas_s:gas_e]), n_pools, k_attr))
        out.update(self.noise_head.unpack_result(
            np.array(fitted[noise_s:noise_e]), n_pools, k_attr))

        out["pool_ids"] = jdata.pool_ids
        out["attr_names"] = jdata.attr_names
        out["k_attr"] = k_attr
        out["n_pools"] = n_pools

        return out

    # ── Prediction ─────────────────────────────────────────────────────

    def predict_new_pool(
        self,
        result: dict,
        x_attr: np.ndarray,
    ) -> dict:
        """Predict simulator settings for a new pool.

        Delegates to each head's predict_new.  Heads that can't
        generalize (PerPoolHead, FixedHead) will raise ValueError.
        """
        n_pools = result["n_pools"]
        k_attr = result["k_attr"]
        params = result["params_flat"]

        (cad_s, cad_e), (gas_s, gas_e), (noise_s, noise_e) = \
            self._head_slices(n_pools, k_attr)

        log_cadence = self.cadence_head.predict_new(
            params[cad_s:cad_e], x_attr
        )
        log_gas = self.gas_head.predict_new(
            params[gas_s:gas_e], x_attr
        )

        out = {
            "log_cadence": float(log_cadence),
            "log_gas": float(log_gas),
            "cadence_minutes": float(np.exp(log_cadence)),
            "gas_usd": float(np.exp(log_gas)),
        }

        try:
            noise_coeffs = self.noise_head.predict_new(
                params[noise_s:noise_e], x_attr
            )
            out["noise_coeffs"] = np.array(noise_coeffs)
        except ValueError:
            pass  # PerPoolNoiseHead can't generalize

        return out
