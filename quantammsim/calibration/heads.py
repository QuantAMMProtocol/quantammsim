"""Pluggable Head components for the composable CalibrationModel.

Each Head encapsulates a specific parameterization strategy (per-pool,
fixed, linear) for one of the three model components: cadence, gas, or noise.

Heads define how many parameters they need, how to predict from a parameter
slice, and how to compute regularization.  The CalibrationModel concatenates
head parameter slices into a single flat vector for scipy L-BFGS-B.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np

from quantammsim.calibration.loss import K_OBS


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Head(Protocol):
    """Protocol that all head implementations must satisfy."""

    name: str

    def n_params(self, n_pools: int, k_attr: int) -> int:
        """Number of scalar parameters this head contributes."""
        ...

    def predict(
        self,
        params_slice: jnp.ndarray,
        pool_idx: int,
        x_attr_i: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict value(s) for *pool_idx* given its attribute vector.

        Called inside a JIT-compiled per-pool closure, so this must be
        JAX-traceable.  Returns a scalar for cadence/gas heads, or a
        (K_OBS,) vector for noise heads.
        """
        ...

    def regularization(self, params_slice: jnp.ndarray) -> jnp.ndarray:
        """Scalar regularization penalty added to the joint loss."""
        ...

    def init(
        self,
        jdata,
        warm_start: Optional[dict] = None,
    ) -> np.ndarray:
        """Return initial NumPy parameter vector (flat)."""
        ...

    def predict_new(
        self,
        params_slice: np.ndarray,
        x_attr: np.ndarray,
    ) -> np.ndarray:
        """Predict for a *new* pool not seen during training (NumPy)."""
        ...

    def unpack_result(
        self,
        params_slice: np.ndarray,
        n_pools: int,
        k_attr: int,
    ) -> dict:
        """Convert the optimized parameter slice to human-readable dict."""
        ...

    def make_bounds(self, n_pools: int, k_attr: int) -> list:
        """Scipy (lo, hi) bounds for each parameter."""
        ...


# ---------------------------------------------------------------------------
# PerPoolHead — one free scalar per pool (Option C cadence / gas)
# ---------------------------------------------------------------------------


class PerPoolHead:
    """One free scalar parameter per pool.

    Used for Option C per-pool cadence or gas.
    """

    def __init__(self, name: str, default: float = 0.0):
        self.name = name
        self._default = default

    def n_params(self, n_pools: int, k_attr: int) -> int:
        return n_pools

    def predict(
        self,
        params_slice: jnp.ndarray,
        pool_idx: int,
        x_attr_i: jnp.ndarray,
    ) -> jnp.ndarray:
        return params_slice[pool_idx]

    def regularization(self, params_slice: jnp.ndarray) -> jnp.ndarray:
        return jnp.float32(0.0)

    def init(self, jdata, warm_start=None) -> np.ndarray:
        n_pools = len(jdata.pool_data)
        if warm_start is not None:
            vals = []
            for pid in jdata.pool_ids:
                if pid in warm_start and self.name in warm_start[pid]:
                    vals.append(warm_start[pid][self.name])
                else:
                    vals.append(self._default)
            return np.array(vals, dtype=np.float64)
        return np.full(n_pools, self._default, dtype=np.float64)

    def predict_new(self, params_slice, x_attr):
        raise ValueError(
            f"PerPoolHead('{self.name}') cannot predict for unseen pools"
        )

    def unpack_result(self, params_slice, n_pools, k_attr):
        return {f"{self.name}_per_pool": np.array(params_slice)}

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * n_pools


# ---------------------------------------------------------------------------
# FixedHead — zero parameters, returns pre-set values
# ---------------------------------------------------------------------------


class FixedHead:
    """Zero-parameter head that returns pre-set per-pool values.

    Used when gas is fixed to known chain-level costs.
    """

    def __init__(self, name: str, values: np.ndarray):
        self.name = name
        self._values = np.asarray(values, dtype=np.float64)
        self._values_jax = jnp.array(self._values)

    def n_params(self, n_pools: int, k_attr: int) -> int:
        return 0

    def predict(self, params_slice, pool_idx, x_attr_i):
        return self._values_jax[pool_idx]

    def regularization(self, params_slice):
        return jnp.float32(0.0)

    def init(self, jdata, warm_start=None):
        return np.array([], dtype=np.float64)

    def predict_new(self, params_slice, x_attr):
        raise ValueError(
            f"FixedHead('{self.name}') cannot predict for unseen pools — "
            "values are pool-specific"
        )

    def unpack_result(self, params_slice, n_pools, k_attr):
        return {f"{self.name}_fixed": np.array(self._values)}

    def make_bounds(self, n_pools, k_attr):
        return []


# ---------------------------------------------------------------------------
# LinearHead — bias + x_attr @ W  (Option A cadence / gas)
# ---------------------------------------------------------------------------


class LinearHead:
    """Linear mapping from pool attributes: bias + x_attr @ W.

    L2 regularization on W (not bias) with strength ``alpha``.
    """

    def __init__(self, name: str, alpha: float = 0.01):
        self.name = name
        self.alpha = alpha

    def n_params(self, n_pools: int, k_attr: int) -> int:
        return 1 + k_attr  # bias + W

    def predict(self, params_slice, pool_idx, x_attr_i):
        bias = params_slice[0]
        W = params_slice[1:]
        return bias + jnp.dot(x_attr_i, W)

    def regularization(self, params_slice):
        W = params_slice[1:]
        return self.alpha * jnp.sum(W ** 2)

    def init(self, jdata, warm_start=None):
        k_attr = jdata.x_attr.shape[1]
        n_pools = len(jdata.pool_data)

        if warm_start is not None:
            # Fit linear regression from per-pool values
            vals = []
            for pid in jdata.pool_ids:
                if pid in warm_start and self.name in warm_start[pid]:
                    vals.append(warm_start[pid][self.name])
                else:
                    vals.append(self._default_bias())
            y = np.array(vals)
            X_aug = np.column_stack([np.ones(n_pools), np.array(jdata.x_attr)])
            params, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            return params.astype(np.float64)

        init = np.zeros(1 + k_attr, dtype=np.float64)
        init[0] = self._default_bias()
        return init

    def _default_bias(self):
        if "cad" in self.name:
            return np.log(12.0)
        elif "gas" in self.name:
            return np.log(1.0)
        return 0.0

    def predict_new(self, params_slice, x_attr):
        bias = params_slice[0]
        W = params_slice[1:]
        return bias + np.dot(x_attr, W)

    def unpack_result(self, params_slice, n_pools, k_attr):
        return {
            f"bias_{self.name}": float(params_slice[0]),
            f"W_{self.name}": np.array(params_slice[1:]),
        }

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * (1 + k_attr)


# ---------------------------------------------------------------------------
# PerPoolNoiseHead — K_OBS free coefficients per pool
# ---------------------------------------------------------------------------


class PerPoolNoiseHead:
    """Per-pool noise coefficients: each pool has K_OBS free parameters.

    Used for Option C noise or Option A with per-pool noise.
    """

    def __init__(self, alpha: float = 0.0):
        self.name = "noise"
        self.alpha = alpha

    def n_params(self, n_pools: int, k_attr: int) -> int:
        return n_pools * K_OBS

    def predict(self, params_slice, pool_idx, x_attr_i):
        start = pool_idx * K_OBS
        return params_slice[start:start + K_OBS]

    def regularization(self, params_slice):
        if self.alpha == 0.0:
            return jnp.float32(0.0)
        return self.alpha * jnp.sum(params_slice ** 2)

    def init(self, jdata, warm_start=None):
        n_pools = len(jdata.pool_data)

        if warm_start is not None:
            noise_all = np.zeros((n_pools, K_OBS), dtype=np.float64)
            for i, pid in enumerate(jdata.pool_ids):
                if pid in warm_start and "noise_coeffs" in warm_start[pid]:
                    noise_all[i] = warm_start[pid]["noise_coeffs"]
            return noise_all.ravel()

        noise_all = np.zeros((n_pools, K_OBS), dtype=np.float64)
        for i, pd in enumerate(jdata.pool_data):
            x_obs_np = np.array(pd["x_obs"])
            y_obs_np = np.array(pd["y_obs"])
            c, _, _, _ = np.linalg.lstsq(x_obs_np, y_obs_np, rcond=None)
            noise_all[i] = c
        return noise_all.ravel()

    def predict_new(self, params_slice, x_attr):
        raise ValueError(
            "PerPoolNoiseHead cannot predict noise for unseen pools"
        )

    def unpack_result(self, params_slice, n_pools, k_attr):
        return {
            "noise_coeffs": np.array(params_slice).reshape(n_pools, K_OBS),
        }

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * (n_pools * K_OBS)


# ---------------------------------------------------------------------------
# SharedLinearNoiseHead — bias_noise + x_attr @ W_noise
# ---------------------------------------------------------------------------


class SharedLinearNoiseHead:
    """Shared linear mapping for noise: bias_noise + x_attr @ W_noise.

    Output is (K_OBS,) noise coefficients, predicted from pool attributes.
    L2 regularization on W_noise (not bias_noise).
    """

    def __init__(self, alpha: float = 0.01):
        self.name = "noise"
        self.alpha = alpha

    def n_params(self, n_pools: int, k_attr: int) -> int:
        return (1 + k_attr) * K_OBS

    def predict(self, params_slice, pool_idx, x_attr_i):
        # params_slice is ((1+k_attr) * K_OBS,)
        k_attr = x_attr_i.shape[0]
        W_full = params_slice.reshape(1 + k_attr, K_OBS)
        bias_noise = W_full[0]
        W_noise = W_full[1:]
        return bias_noise + jnp.dot(x_attr_i, W_noise)

    def regularization(self, params_slice):
        # Regularize W_noise only, not bias_noise
        W_full = params_slice.reshape(-1, K_OBS)
        W_noise = W_full[1:]
        return self.alpha * jnp.sum(W_noise ** 2)

    def init(self, jdata, warm_start=None):
        k_attr = jdata.x_attr.shape[1]
        n_pools = len(jdata.pool_data)

        if warm_start is not None:
            # Collect per-pool noise, regress on attributes
            noise_all = np.zeros((n_pools, K_OBS), dtype=np.float64)
            for i, pid in enumerate(jdata.pool_ids):
                if pid in warm_start and "noise_coeffs" in warm_start[pid]:
                    noise_all[i] = warm_start[pid]["noise_coeffs"]
            X_aug = np.column_stack([np.ones(n_pools), np.array(jdata.x_attr)])
            params, _, _, _ = np.linalg.lstsq(X_aug, noise_all, rcond=None)
            return params.ravel().astype(np.float64)

        # Pool OLS noise as shared bias, W_noise = 0
        all_x = np.vstack([np.array(pd["x_obs"]) for pd in jdata.pool_data])
        all_y = np.concatenate([np.array(pd["y_obs"]) for pd in jdata.pool_data])
        c, _, _, _ = np.linalg.lstsq(all_x, all_y, rcond=None)
        params = np.zeros((1 + k_attr, K_OBS), dtype=np.float64)
        params[0, :] = c
        return params.ravel()

    def predict_new(self, params_slice, x_attr):
        k_attr = len(x_attr)
        W_full = np.array(params_slice).reshape(1 + k_attr, K_OBS)
        bias_noise = W_full[0]
        W_noise = W_full[1:]
        return bias_noise + x_attr @ W_noise

    def unpack_result(self, params_slice, n_pools, k_attr):
        W_full = np.array(params_slice).reshape(1 + k_attr, K_OBS)
        return {
            "bias_noise": W_full[0],
            "W_noise": W_full[1:],
        }

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * ((1 + k_attr) * K_OBS)


# ---------------------------------------------------------------------------
# MLPHead — x_attr → Dense(hidden, relu) → Dense(1)
# ---------------------------------------------------------------------------


class MLPHead:
    """Two-layer MLP mapping from pool attributes to a scalar.

    Architecture: x_attr → Dense(hidden, ReLU) → Dense(1) → scalar

    Parameter layout (flat):
        [W1(k_attr * hidden), b1(hidden), W2(hidden), b2(1)]

    L2 regularization on W1 and W2 (not biases).

    Initialization:
      - W1: He (scaled normal), b1: zeros
      - W2: zeros (so initial output ≈ b2 = default bias)
      - b2: sensible default (log(12) for cadence, log(1) for gas)
    """

    def __init__(
        self,
        name: str,
        hidden: int = 16,
        alpha: float = 0.01,
        seed: int = 0,
    ):
        self.name = name
        self.hidden = hidden
        self.alpha = alpha
        self._seed = seed

    def n_params(self, n_pools: int, k_attr: int) -> int:
        h = self.hidden
        return k_attr * h + h + h + 1  # W1 + b1 + W2 + b2

    def _unpack_weights(self, params_slice, k_attr):
        """Unpack flat slice → (W1, b1, W2, b2) as JAX arrays."""
        h = self.hidden
        idx = 0
        W1 = params_slice[idx:idx + k_attr * h].reshape(k_attr, h)
        idx += k_attr * h
        b1 = params_slice[idx:idx + h]
        idx += h
        W2 = params_slice[idx:idx + h]
        idx += h
        b2 = params_slice[idx]
        return W1, b1, W2, b2

    def predict(self, params_slice, pool_idx, x_attr_i):
        k_attr = x_attr_i.shape[0]
        W1, b1, W2, b2 = self._unpack_weights(params_slice, k_attr)
        hidden = jnp.maximum(x_attr_i @ W1 + b1, 0.0)  # ReLU
        return hidden @ W2 + b2

    def regularization(self, params_slice):
        # Regularize W1 and W2, not biases
        # We can't call _unpack_weights without k_attr, so compute
        # the total weight norm from the full slice minus biases.
        # Layout: [W1(k*h), b1(h), W2(h), b2(1)]
        # But we don't know k_attr here. Use a simpler approach:
        # regularize the entire slice — biases are small relative to
        # weights and the approximation error is negligible.
        # Actually, let's extract properly by computing h from params.
        h = self.hidden
        total = params_slice.shape[0]
        k_attr = (total - 2 * h - 1) // h
        W1 = params_slice[:k_attr * h]
        # b1 = params_slice[k_attr*h : k_attr*h + h]  # skip
        W2 = params_slice[k_attr * h + h:k_attr * h + 2 * h]
        # b2 = params_slice[-1]  # skip
        return self.alpha * (jnp.sum(W1 ** 2) + jnp.sum(W2 ** 2))

    def init(self, jdata, warm_start=None):
        k_attr = jdata.x_attr.shape[1]
        n_pools = len(jdata.pool_data)
        h = self.hidden
        rng = np.random.RandomState(self._seed)

        # He initialization for W1
        std = np.sqrt(2.0 / k_attr)
        W1 = rng.randn(k_attr, h).astype(np.float64) * std
        b1 = np.zeros(h, dtype=np.float64)

        # W2 = 0 so initial output = b2 (warm-start friendly)
        W2 = np.zeros(h, dtype=np.float64)
        b2 = np.array([self._default_bias()], dtype=np.float64)

        if warm_start is not None:
            # Fit linear mapping from per-pool values, use as last-layer init
            vals = []
            for pid in jdata.pool_ids:
                if pid in warm_start and self.name in warm_start[pid]:
                    vals.append(warm_start[pid][self.name])
                else:
                    vals.append(self._default_bias())
            y = np.array(vals)
            # Use mean as b2 (since W2=0, output = b2)
            b2 = np.array([np.mean(y)], dtype=np.float64)

        return np.concatenate([W1.ravel(), b1, W2, b2])

    def _default_bias(self):
        if "cad" in self.name:
            return np.log(12.0)
        elif "gas" in self.name:
            return np.log(1.0)
        return 0.0

    def predict_new(self, params_slice, x_attr):
        k_attr = len(x_attr)
        W1, b1, W2, b2 = self._unpack_weights(
            np.asarray(params_slice), k_attr
        )
        hidden = np.maximum(x_attr @ W1 + b1, 0.0)
        return float(hidden @ W2 + b2)

    def unpack_result(self, params_slice, n_pools, k_attr):
        params_np = np.array(params_slice)
        W1, b1, W2, b2 = self._unpack_weights(params_np, k_attr)
        return {
            f"mlp_{self.name}_W1": np.array(W1),
            f"mlp_{self.name}_b1": np.array(b1),
            f"mlp_{self.name}_W2": np.array(W2),
            f"mlp_{self.name}_b2": float(b2),
        }

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * self.n_params(n_pools, k_attr)


# ---------------------------------------------------------------------------
# MLPNoiseHead — x_attr → Dense(hidden, relu) → Dense(K_OBS)
# ---------------------------------------------------------------------------


class MLPNoiseHead:
    """Two-layer MLP mapping from pool attributes to noise coefficients.

    Architecture: x_attr → Dense(hidden, ReLU) → Dense(K_OBS)

    Parameter layout (flat):
        [W1(k_attr * hidden), b1(hidden), W2(hidden * K_OBS), b2(K_OBS)]

    L2 regularization on W1 and W2 (not biases).

    Initialization:
      - W1: He (scaled normal), b1: zeros
      - W2: zeros (so initial output = b2 = pooled OLS noise coefficients)
      - b2: pooled OLS noise from training data
    """

    def __init__(
        self,
        hidden: int = 16,
        alpha: float = 0.01,
        seed: int = 0,
    ):
        self.name = "noise"
        self.hidden = hidden
        self.alpha = alpha
        self._seed = seed

    def n_params(self, n_pools: int, k_attr: int) -> int:
        h = self.hidden
        # W1(k_attr*h) + b1(h) + W2(h*K_OBS) + b2(K_OBS)
        return k_attr * h + h + h * K_OBS + K_OBS

    def _unpack_weights(self, params_slice, k_attr):
        """Unpack flat slice → (W1, b1, W2, b2)."""
        h = self.hidden
        idx = 0
        W1 = params_slice[idx:idx + k_attr * h].reshape(k_attr, h)
        idx += k_attr * h
        b1 = params_slice[idx:idx + h]
        idx += h
        W2 = params_slice[idx:idx + h * K_OBS].reshape(h, K_OBS)
        idx += h * K_OBS
        b2 = params_slice[idx:idx + K_OBS]
        return W1, b1, W2, b2

    def predict(self, params_slice, pool_idx, x_attr_i):
        k_attr = x_attr_i.shape[0]
        W1, b1, W2, b2 = self._unpack_weights(params_slice, k_attr)
        hidden = jnp.maximum(x_attr_i @ W1 + b1, 0.0)  # ReLU
        return hidden @ W2 + b2  # (K_OBS,)

    def regularization(self, params_slice):
        h = self.hidden
        total = params_slice.shape[0]
        # Solve for k_attr: total = k*h + h + h*K_OBS + K_OBS
        # k*h = total - h - h*K_OBS - K_OBS
        k_attr = (total - h - h * K_OBS - K_OBS) // h
        W1 = params_slice[:k_attr * h]
        W2 = params_slice[k_attr * h + h:k_attr * h + h + h * K_OBS]
        return self.alpha * (jnp.sum(W1 ** 2) + jnp.sum(W2 ** 2))

    def init(self, jdata, warm_start=None):
        k_attr = jdata.x_attr.shape[1]
        n_pools = len(jdata.pool_data)
        h = self.hidden
        rng = np.random.RandomState(self._seed)

        # He initialization for W1
        std = np.sqrt(2.0 / k_attr)
        W1 = rng.randn(k_attr, h).astype(np.float64) * std
        b1 = np.zeros(h, dtype=np.float64)

        # W2 = 0 so initial output = b2
        W2 = np.zeros((h, K_OBS), dtype=np.float64)

        if warm_start is not None:
            # Use mean of per-pool noise as b2
            noise_all = np.zeros((n_pools, K_OBS), dtype=np.float64)
            for i, pid in enumerate(jdata.pool_ids):
                if pid in warm_start and "noise_coeffs" in warm_start[pid]:
                    noise_all[i] = warm_start[pid]["noise_coeffs"]
            b2 = np.mean(noise_all, axis=0)
        else:
            # Pooled OLS noise as b2
            all_x = np.vstack([np.array(pd["x_obs"]) for pd in jdata.pool_data])
            all_y = np.concatenate([np.array(pd["y_obs"]) for pd in jdata.pool_data])
            b2, _, _, _ = np.linalg.lstsq(all_x, all_y, rcond=None)

        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def predict_new(self, params_slice, x_attr):
        k_attr = len(x_attr)
        W1, b1, W2, b2 = self._unpack_weights(np.asarray(params_slice), k_attr)
        hidden = np.maximum(x_attr @ W1 + b1, 0.0)
        return hidden @ W2 + b2  # (K_OBS,)

    def unpack_result(self, params_slice, n_pools, k_attr):
        params_np = np.array(params_slice)
        W1, b1, W2, b2 = self._unpack_weights(params_np, k_attr)
        return {
            "mlp_noise_W1": np.array(W1),
            "mlp_noise_b1": np.array(b1),
            "mlp_noise_W2": np.array(W2),
            "mlp_noise_b2": np.array(b2),
        }

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * self.n_params(n_pools, k_attr)
