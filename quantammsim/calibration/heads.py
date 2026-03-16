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
from quantammsim.calibration.pool_data import (
    D_TOKEN, K_OBS_REDUCED, _canonicalize_token, _classify_token,
    _load_token_mcaps,
)


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

    def __init__(self, name: str, alpha: float = 0.01,
                 output_lo: float = None, output_hi: float = None):
        self.name = name
        self.alpha = alpha
        self.output_lo = output_lo
        self.output_hi = output_hi

    def n_params(self, n_pools: int, k_attr: int) -> int:
        return 1 + k_attr  # bias + W

    def predict(self, params_slice, pool_idx, x_attr_i):
        bias = params_slice[0]
        W = params_slice[1:]
        out = bias + jnp.dot(x_attr_i, W)
        if self.output_lo is not None or self.output_hi is not None:
            out = jnp.clip(out, self.output_lo, self.output_hi)
        return out

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
    """Per-pool noise coefficients: each pool has k_obs free parameters.

    Used for Option C noise or Option A with per-pool noise.
    """

    def __init__(self, alpha: float = 0.0, k_obs: int = None):
        self.name = "noise"
        self.alpha = alpha
        self.k_obs = k_obs if k_obs is not None else K_OBS

    def n_params(self, n_pools: int, k_attr: int) -> int:
        return n_pools * self.k_obs

    def predict(self, params_slice, pool_idx, x_attr_i):
        start = pool_idx * self.k_obs
        return params_slice[start:start + self.k_obs]

    def regularization(self, params_slice):
        if self.alpha == 0.0:
            return jnp.float32(0.0)
        return self.alpha * jnp.sum(params_slice ** 2)

    def init(self, jdata, warm_start=None):
        n_pools = len(jdata.pool_data)

        if warm_start is not None:
            noise_all = np.zeros((n_pools, self.k_obs), dtype=np.float64)
            for i, pid in enumerate(jdata.pool_ids):
                if pid in warm_start and "noise_coeffs" in warm_start[pid]:
                    noise_all[i] = warm_start[pid]["noise_coeffs"]
            return noise_all.ravel()

        noise_all = np.zeros((n_pools, self.k_obs), dtype=np.float64)
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
            "noise_coeffs": np.array(params_slice).reshape(n_pools, self.k_obs),
        }

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * (n_pools * self.k_obs)


# ---------------------------------------------------------------------------
# SharedLinearNoiseHead — bias_noise + x_attr @ W_noise
# ---------------------------------------------------------------------------


class SharedLinearNoiseHead:
    """Shared linear mapping for noise: bias_noise + x_attr @ W_noise.

    Output is (k_obs,) noise coefficients, predicted from pool attributes.
    L2 regularization on W_noise (not bias_noise).
    """

    def __init__(self, alpha: float = 0.01, k_obs: int = None):
        self.name = "noise"
        self.alpha = alpha
        self.k_obs = k_obs if k_obs is not None else K_OBS

    def n_params(self, n_pools: int, k_attr: int) -> int:
        return (1 + k_attr) * self.k_obs

    def predict(self, params_slice, pool_idx, x_attr_i):
        k_attr = x_attr_i.shape[0]
        W_full = params_slice.reshape(1 + k_attr, self.k_obs)
        bias_noise = W_full[0]
        W_noise = W_full[1:]
        return bias_noise + jnp.dot(x_attr_i, W_noise)

    def regularization(self, params_slice):
        W_full = params_slice.reshape(-1, self.k_obs)
        W_noise = W_full[1:]
        return self.alpha * jnp.sum(W_noise ** 2)

    def init(self, jdata, warm_start=None):
        k_attr = jdata.x_attr.shape[1]
        n_pools = len(jdata.pool_data)

        if warm_start is not None:
            noise_all = np.zeros((n_pools, self.k_obs), dtype=np.float64)
            for i, pid in enumerate(jdata.pool_ids):
                if pid in warm_start and "noise_coeffs" in warm_start[pid]:
                    noise_all[i] = warm_start[pid]["noise_coeffs"]
            X_aug = np.column_stack([np.ones(n_pools), np.array(jdata.x_attr)])
            params, _, _, _ = np.linalg.lstsq(X_aug, noise_all, rcond=None)
            return params.ravel().astype(np.float64)

        all_x = np.vstack([np.array(pd["x_obs"]) for pd in jdata.pool_data])
        all_y = np.concatenate([np.array(pd["y_obs"]) for pd in jdata.pool_data])
        c, _, _, _ = np.linalg.lstsq(all_x, all_y, rcond=None)
        params = np.zeros((1 + k_attr, self.k_obs), dtype=np.float64)
        params[0, :] = c
        return params.ravel()

    def predict_new(self, params_slice, x_attr):
        k_attr = len(x_attr)
        W_full = np.array(params_slice).reshape(1 + k_attr, self.k_obs)
        bias_noise = W_full[0]
        W_noise = W_full[1:]
        return bias_noise + x_attr @ W_noise

    def unpack_result(self, params_slice, n_pools, k_attr):
        W_full = np.array(params_slice).reshape(1 + k_attr, self.k_obs)
        return {
            "bias_noise": W_full[0],
            "W_noise": W_full[1:],
        }

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * ((1 + k_attr) * self.k_obs)


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
        output_lo: float = None,
        output_hi: float = None,
    ):
        self.name = name
        self.hidden = hidden
        self.alpha = alpha
        self._seed = seed
        self.output_lo = output_lo
        self.output_hi = output_hi

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
        out = hidden @ W2 + b2
        if self.output_lo is not None or self.output_hi is not None:
            out = jnp.clip(out, self.output_lo, self.output_hi)
        return out

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

        b2 = np.array([self._default_bias()], dtype=np.float64)
        W2 = np.zeros(h, dtype=np.float64)

        if warm_start is not None:
            vals = []
            for pid in jdata.pool_ids:
                if pid in warm_start and self.name in warm_start[pid]:
                    vals.append(warm_start[pid][self.name])
                else:
                    vals.append(self._default_bias())
            y = np.array(vals)
            b2 = np.array([np.mean(y)], dtype=np.float64)

            # Warm-start W2 by least-squares through hidden activations
            # so the MLP init approximates the per-pool warm-start values
            x_attr = np.array(jdata.x_attr)
            H = np.maximum(x_attr @ W1 + b1, 0.0)  # (n_pools, h)
            residuals = y - float(b2)  # what W2 needs to produce
            W2, _, _, _ = np.linalg.lstsq(H, residuals, rcond=None)

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
# TokenFactoredNoiseHead — additive token + chain + fee composition
# ---------------------------------------------------------------------------


class TokenFactoredNoiseHead:
    """Noise coefficients from additive token + chain + fee composition.

    noise_coeffs_i = u[token_a_i] + u[token_b_i] + alpha[chain_i]
                     + beta_fee * log(fee_i) + delta_i

    Token effects u_t are regularized toward x_token_t @ Gamma (population
    prediction from token covariates). Per-pool deltas are L2-regularized,
    controlling the shrinkage between per-pool and population estimates.

    Parameter layout (flat):
        [u (n_tokens * k_obs),
         Gamma (d_token * k_obs),
         alpha (n_chains * k_obs),
         beta_fee (k_obs),
         delta (n_pools * k_obs)]
    """

    def __init__(
        self,
        token_a_idx: np.ndarray,
        token_b_idx: np.ndarray,
        chain_idx: np.ndarray,
        log_fees: np.ndarray,
        x_token: np.ndarray,
        n_tokens: int,
        n_chains: int,
        token_index: dict,
        chain_index: dict,
        k_obs: int = K_OBS_REDUCED,
        lambda_delta: float = 1.0,
        lambda_token: float = 0.1,
        lambda_chain: float = 0.1,
        lambda_fee: float = 0.01,
        mcap_path: str = None,
    ):
        self.name = "noise"
        self.token_a_idx = np.asarray(token_a_idx, dtype=np.int32)
        self.token_b_idx = np.asarray(token_b_idx, dtype=np.int32)
        self.chain_idx = np.asarray(chain_idx, dtype=np.int32)
        self.log_fees = np.asarray(log_fees, dtype=np.float64)
        self.x_token = np.asarray(x_token, dtype=np.float64)
        self.n_tokens = n_tokens
        self.n_chains = n_chains
        self.d_token = x_token.shape[1]
        self.k_obs = k_obs
        self.token_index = dict(token_index)
        self.chain_index = dict(chain_index)
        self.lambda_delta = lambda_delta
        self.lambda_token = lambda_token
        self.lambda_chain = lambda_chain
        self.lambda_fee = lambda_fee
        self._mcap_path = mcap_path
        # Pre-convert to JAX for predict()
        self._token_a_jax = jnp.array(self.token_a_idx)
        self._token_b_jax = jnp.array(self.token_b_idx)
        self._chain_jax = jnp.array(self.chain_idx)
        self._log_fees_jax = jnp.array(self.log_fees)
        self._x_token_jax = jnp.array(self.x_token)

    def n_params(self, n_pools: int, k_attr: int) -> int:
        k = self.k_obs
        return (self.n_tokens * k       # u
                + self.d_token * k       # Gamma
                + self.n_chains * k      # alpha
                + k                      # beta_fee
                + n_pools * k)           # delta

    def _unpack(self, params_slice, n_pools):
        k = self.k_obs
        idx = 0
        u = params_slice[idx:idx + self.n_tokens * k].reshape(self.n_tokens, k)
        idx += self.n_tokens * k
        Gamma = params_slice[idx:idx + self.d_token * k].reshape(self.d_token, k)
        idx += self.d_token * k
        alpha = params_slice[idx:idx + self.n_chains * k].reshape(self.n_chains, k)
        idx += self.n_chains * k
        beta_fee = params_slice[idx:idx + k]
        idx += k
        delta = params_slice[idx:idx + n_pools * k].reshape(n_pools, k)
        return u, Gamma, alpha, beta_fee, delta

    def _infer_n_pools(self, params_slice):
        k = self.k_obs
        n_shared = self.n_tokens * k + self.d_token * k + self.n_chains * k + k
        return (params_slice.shape[0] - n_shared) // k

    def predict(self, params_slice, pool_idx, x_attr_i):
        n_pools = self._infer_n_pools(params_slice)
        u, Gamma, alpha, beta_fee, delta = self._unpack(params_slice, n_pools)
        ta = self._token_a_jax[pool_idx]
        tb = self._token_b_jax[pool_idx]
        ch = self._chain_jax[pool_idx]
        lf = self._log_fees_jax[pool_idx]
        return u[ta] + u[tb] + alpha[ch] + beta_fee * lf + delta[pool_idx]

    def regularization(self, params_slice):
        n_pools = self._infer_n_pools(params_slice)
        u, Gamma, alpha, beta_fee, delta = self._unpack(params_slice, n_pools)
        u_pred = self._x_token_jax @ Gamma
        reg_token = self.lambda_token * jnp.sum((u - u_pred) ** 2)
        reg_chain = self.lambda_chain * jnp.sum(alpha ** 2)
        reg_fee = self.lambda_fee * jnp.sum(beta_fee ** 2)
        reg_delta = self.lambda_delta * jnp.sum(delta ** 2)
        return reg_token + reg_chain + reg_fee + reg_delta

    def init(self, jdata, warm_start=None):
        n_pools = len(jdata.pool_data)
        k = self.k_obs

        if warm_start is not None:
            # Collect per-pool noise_coeffs from warm_start
            noise_all = np.zeros((n_pools, k), dtype=np.float64)
            for i, pid in enumerate(jdata.pool_ids):
                if pid in warm_start and "noise_coeffs" in warm_start[pid]:
                    nc = warm_start[pid]["noise_coeffs"]
                    noise_all[i] = nc[:k]

            # Solve: u[ta_i] + u[tb_i] + alpha[ch_i] + beta_fee * lf_i ≈ noise_all[i]
            n_cols = self.n_tokens + self.n_chains + 1
            A = np.zeros((n_pools, n_cols), dtype=np.float64)
            for i in range(n_pools):
                A[i, self.token_a_idx[i]] = 1.0
                A[i, self.token_b_idx[i]] += 1.0
                A[i, self.n_tokens + self.chain_idx[i]] = 1.0
                A[i, -1] = self.log_fees[i]

            lam_reg = 0.1
            AtA = A.T @ A + lam_reg * np.eye(n_cols)
            u_init = np.zeros((self.n_tokens, k))
            alpha_init = np.zeros((self.n_chains, k))
            beta_fee_init = np.zeros(k)

            for j in range(k):
                sol = np.linalg.solve(AtA, A.T @ noise_all[:, j])
                u_init[:, j] = sol[:self.n_tokens]
                alpha_init[:, j] = sol[self.n_tokens:self.n_tokens + self.n_chains]
                beta_fee_init[j] = sol[-1]

            # Delta = residuals
            predicted = np.zeros_like(noise_all)
            for i in range(n_pools):
                predicted[i] = (u_init[self.token_a_idx[i]]
                               + u_init[self.token_b_idx[i]]
                               + alpha_init[self.chain_idx[i]]
                               + beta_fee_init * self.log_fees[i])
            delta_init = noise_all - predicted

            # Gamma from post-hoc regression of u on x_token
            Gamma_init, _, _, _ = np.linalg.lstsq(
                self.x_token, u_init, rcond=None
            )
        else:
            # Cold start: pooled OLS for baseline, then decompose
            all_x = np.vstack([np.array(pd["x_obs"]) for pd in jdata.pool_data])
            all_y = np.concatenate([np.array(pd["y_obs"]) for pd in jdata.pool_data])
            pooled_coeffs, _, _, _ = np.linalg.lstsq(all_x, all_y, rcond=None)
            pooled_coeffs = pooled_coeffs[:k]

            u_init = np.tile(pooled_coeffs / 2.0, (self.n_tokens, 1))
            Gamma_init, _, _, _ = np.linalg.lstsq(
                self.x_token, u_init, rcond=None
            )
            alpha_init = np.zeros((self.n_chains, k))
            beta_fee_init = np.zeros(k)
            delta_init = np.zeros((n_pools, k))

        return np.concatenate([
            u_init.ravel(),
            Gamma_init.ravel(),
            alpha_init.ravel(),
            beta_fee_init,
            delta_init.ravel(),
        ]).astype(np.float64)

    def predict_new(self, params_slice, x_attr):
        raise ValueError(
            "TokenFactoredNoiseHead.predict_new() requires token identifiers. "
            "Use predict_new_pool(params, token_a, token_b, chain, fee) instead."
        )

    def predict_new_pool(
        self, params_slice, token_a, token_b, chain, fee, n_pools,
    ) -> dict:
        """Predict noise coefficients for a new pool from token composition.

        Seen tokens use learned u_t. Unseen tokens fall back to x_t @ Gamma.
        Unseen chains use alpha = zeros. No delta for new pools.
        Input token names are canonicalized before lookup.
        """
        params_np = np.asarray(params_slice)
        u, Gamma, alpha, beta_fee, delta = self._unpack(params_np, n_pools)
        u, Gamma, alpha, beta_fee = (
            np.array(u), np.array(Gamma), np.array(alpha), np.array(beta_fee)
        )
        mcaps = _load_token_mcaps(self._mcap_path)

        # Canonicalize input tokens
        token_a = _canonicalize_token(token_a)
        token_b = _canonicalize_token(token_b)

        def _get_token_effect(token):
            if token in self.token_index:
                return u[self.token_index[token]]
            x_t = np.zeros(self.d_token)
            x_t[0] = 1.0
            cls = _classify_token(token, mcaps)
            x_t[1] = cls["log_mcap"]
            x_t[2] = cls["is_stable"]
            x_t[3] = cls["is_eth_derivative"]
            x_t[4] = cls["is_L1_native"]
            return x_t @ Gamma

        u_a = _get_token_effect(token_a)
        u_b = _get_token_effect(token_b)

        if chain in self.chain_index:
            alpha_c = alpha[self.chain_index[chain]]
        else:
            alpha_c = np.zeros(self.k_obs)

        fee_effect = beta_fee * np.log(fee)
        noise_coeffs = u_a + u_b + alpha_c + fee_effect

        return {
            "noise_coeffs": noise_coeffs,
            "components": {
                "token_a": u_a,
                "token_b": u_b,
                "chain": alpha_c,
                "fee": fee_effect,
            },
        }

    def unpack_result(self, params_slice, n_pools, k_attr):
        params_np = np.asarray(params_slice)
        u, Gamma, alpha, beta_fee, delta = self._unpack(params_np, n_pools)
        u, Gamma, alpha, beta_fee, delta = (
            np.array(u), np.array(Gamma), np.array(alpha),
            np.array(beta_fee), np.array(delta),
        )
        # Reconstruct per-pool noise_coeffs
        noise_coeffs = np.zeros((n_pools, self.k_obs))
        for i in range(n_pools):
            noise_coeffs[i] = (u[self.token_a_idx[i]] + u[self.token_b_idx[i]]
                              + alpha[self.chain_idx[i]]
                              + beta_fee * self.log_fees[i]
                              + delta[i])
        return {
            "token_effects": u,
            "Gamma": Gamma,
            "chain_effects": alpha,
            "beta_fee": beta_fee,
            "noise_deltas": delta,
            "noise_coeffs": noise_coeffs,
        }

    def make_bounds(self, n_pools, k_attr):
        return [(None, None)] * self.n_params(n_pools, k_attr)


# ---------------------------------------------------------------------------
# MLPNoiseHead — x_attr → Dense(hidden, relu) → Dense(K_OBS)
# ---------------------------------------------------------------------------


class MLPNoiseHead:
    """Two-layer MLP mapping from pool attributes to noise coefficients.

    Architecture: x_attr → Dense(hidden, ReLU) → Dense(k_obs)

    Parameter layout (flat):
        [W1(k_attr * hidden), b1(hidden), W2(hidden * k_obs), b2(k_obs)]

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
        k_obs: int = None,
    ):
        self.name = "noise"
        self.hidden = hidden
        self.alpha = alpha
        self._seed = seed
        self.k_obs = k_obs if k_obs is not None else K_OBS

    def n_params(self, n_pools: int, k_attr: int) -> int:
        h = self.hidden
        return k_attr * h + h + h * self.k_obs + self.k_obs

    def _unpack_weights(self, params_slice, k_attr):
        """Unpack flat slice → (W1, b1, W2, b2)."""
        h = self.hidden
        ko = self.k_obs
        idx = 0
        W1 = params_slice[idx:idx + k_attr * h].reshape(k_attr, h)
        idx += k_attr * h
        b1 = params_slice[idx:idx + h]
        idx += h
        W2 = params_slice[idx:idx + h * ko].reshape(h, ko)
        idx += h * ko
        b2 = params_slice[idx:idx + ko]
        return W1, b1, W2, b2

    def predict(self, params_slice, pool_idx, x_attr_i):
        k_attr = x_attr_i.shape[0]
        W1, b1, W2, b2 = self._unpack_weights(params_slice, k_attr)
        hidden = jnp.maximum(x_attr_i @ W1 + b1, 0.0)  # ReLU
        return hidden @ W2 + b2  # (k_obs,)

    def regularization(self, params_slice):
        h = self.hidden
        ko = self.k_obs
        total = params_slice.shape[0]
        # Solve for k_attr: total = k*h + h + h*ko + ko
        k_attr = (total - h - h * ko - ko) // h
        W1 = params_slice[:k_attr * h]
        W2 = params_slice[k_attr * h + h:k_attr * h + h + h * ko]
        return self.alpha * (jnp.sum(W1 ** 2) + jnp.sum(W2 ** 2))

    def init(self, jdata, warm_start=None):
        k_attr = jdata.x_attr.shape[1]
        n_pools = len(jdata.pool_data)
        h = self.hidden
        ko = self.k_obs
        rng = np.random.RandomState(self._seed)

        # He initialization for W1
        std = np.sqrt(2.0 / k_attr)
        W1 = rng.randn(k_attr, h).astype(np.float64) * std
        b1 = np.zeros(h, dtype=np.float64)

        W2 = np.zeros((h, ko), dtype=np.float64)

        if warm_start is not None:
            noise_all = np.zeros((n_pools, ko), dtype=np.float64)
            for i, pid in enumerate(jdata.pool_ids):
                if pid in warm_start and "noise_coeffs" in warm_start[pid]:
                    noise_all[i] = warm_start[pid]["noise_coeffs"]
            b2 = np.mean(noise_all, axis=0)

            # Warm-start W2 by least-squares through hidden activations
            x_attr = np.array(jdata.x_attr)
            H = np.maximum(x_attr @ W1 + b1, 0.0)  # (n_pools, h)
            residuals = noise_all - b2  # (n_pools, ko)
            W2, _, _, _ = np.linalg.lstsq(H, residuals, rcond=None)
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
        return hidden @ W2 + b2  # (k_obs,)

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
