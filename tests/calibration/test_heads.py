"""Tests for quantammsim.calibration.heads — pluggable Head components."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, POOL_PREFIXES

from quantammsim.calibration.heads import (
    FixedHead,
    Head,
    LinearHead,
    PerPoolHead,
    PerPoolNoiseHead,
    SharedLinearNoiseHead,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

N_POOLS = 2
K_ATTR = 5


def _make_fake_jdata():
    """Minimal JointData-like object for init() testing."""
    from quantammsim.calibration.joint_fit import JointData

    pool_data = []
    for _ in range(N_POOLS):
        n_obs = 14
        x_obs = np.random.randn(n_obs, K_OBS)
        x_obs[:, 0] = 1.0  # intercept column
        y_obs = np.random.randn(n_obs) * 0.5 + 9.0
        pool_data.append({
            "x_obs": jnp.array(x_obs),
            "y_obs": jnp.array(y_obs),
            "day_indices": jnp.arange(n_obs) % 10,
        })

    x_attr = jnp.array(np.random.randn(N_POOLS, K_ATTR))
    return JointData(
        pool_data=pool_data,
        x_attr=x_attr,
        pool_ids=POOL_PREFIXES[:N_POOLS],
        attr_names=[f"attr_{i}" for i in range(K_ATTR)],
    )


# ── Protocol compliance ────────────────────────────────────────────────────


class TestProtocol:
    def test_per_pool_head_is_head(self):
        assert isinstance(PerPoolHead("cad"), Head)

    def test_fixed_head_is_head(self):
        assert isinstance(FixedHead("gas", np.array([1.0, 2.0])), Head)

    def test_linear_head_is_head(self):
        assert isinstance(LinearHead("cad"), Head)

    def test_per_pool_noise_head_is_head(self):
        assert isinstance(PerPoolNoiseHead(), Head)

    def test_shared_linear_noise_head_is_head(self):
        assert isinstance(SharedLinearNoiseHead(), Head)


# ── PerPoolHead ─────────────────────────────────────────────────────────────


class TestPerPoolHead:
    def test_n_params(self):
        h = PerPoolHead("cad")
        assert h.n_params(3, 5) == 3
        assert h.n_params(10, 7) == 10

    def test_predict_returns_indexed_value(self):
        h = PerPoolHead("cad")
        params = jnp.array([1.0, 2.0, 3.0])
        x_attr_i = jnp.zeros(5)
        assert float(h.predict(params, 0, x_attr_i)) == 1.0
        assert float(h.predict(params, 1, x_attr_i)) == 2.0
        assert float(h.predict(params, 2, x_attr_i)) == 3.0

    def test_regularization_is_zero(self):
        h = PerPoolHead("cad")
        params = jnp.array([1.0, 2.0, 3.0])
        assert float(h.regularization(params)) == 0.0

    def test_init_default(self):
        h = PerPoolHead("cad", default=np.log(12.0))
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        assert init.shape == (N_POOLS,)
        np.testing.assert_allclose(init, np.log(12.0))

    def test_init_warm_start(self):
        h = PerPoolHead("log_cadence")
        jdata = _make_fake_jdata()
        warm = {
            POOL_PREFIXES[0]: {"log_cadence": 2.5},
            POOL_PREFIXES[1]: {"log_cadence": 3.0},
        }
        init = h.init(jdata, warm_start=warm)
        np.testing.assert_allclose(init, [2.5, 3.0])

    def test_predict_new_raises(self):
        h = PerPoolHead("cad")
        with pytest.raises(ValueError, match="cannot predict"):
            h.predict_new(np.array([1.0]), np.zeros(5))

    def test_make_bounds(self):
        h = PerPoolHead("cad")
        bounds = h.make_bounds(3, 5)
        assert len(bounds) == 3
        assert all(b == (None, None) for b in bounds)


# ── FixedHead ───────────────────────────────────────────────────────────────


class TestFixedHead:
    def test_n_params_is_zero(self):
        h = FixedHead("gas", np.array([0.0, -4.6]))
        assert h.n_params(2, 5) == 0

    def test_predict_returns_fixed_value(self):
        vals = np.array([0.0, -4.6, 1.5])
        h = FixedHead("gas", vals)
        empty_slice = jnp.array([])
        x_attr_i = jnp.zeros(5)
        assert float(h.predict(empty_slice, 0, x_attr_i)) == 0.0
        np.testing.assert_allclose(
            float(h.predict(empty_slice, 1, x_attr_i)), -4.6
        )
        assert float(h.predict(empty_slice, 2, x_attr_i)) == 1.5

    def test_regularization_is_zero(self):
        h = FixedHead("gas", np.array([1.0]))
        assert float(h.regularization(jnp.array([]))) == 0.0

    def test_init_returns_empty(self):
        h = FixedHead("gas", np.array([1.0, 2.0]))
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        assert init.shape == (0,)

    def test_predict_new_raises(self):
        h = FixedHead("gas", np.array([1.0]))
        with pytest.raises(ValueError, match="cannot predict"):
            h.predict_new(np.array([]), np.zeros(5))

    def test_make_bounds_empty(self):
        h = FixedHead("gas", np.array([1.0]))
        assert h.make_bounds(1, 5) == []


# ── LinearHead ──────────────────────────────────────────────────────────────


class TestLinearHead:
    def test_n_params(self):
        h = LinearHead("cad")
        assert h.n_params(3, 5) == 6  # 1 + 5
        assert h.n_params(10, 7) == 8  # 1 + 7

    def test_predict_bias_plus_dot(self):
        h = LinearHead("cad")
        # params_slice = [bias, W0, W1, W2]
        params = jnp.array([2.0, 0.5, -1.0, 0.3])
        x_attr_i = jnp.array([1.0, 2.0, 3.0])
        # expected = 2.0 + (0.5*1.0 + (-1.0)*2.0 + 0.3*3.0)
        #          = 2.0 + 0.5 - 2.0 + 0.9 = 1.4
        result = float(h.predict(params, 0, x_attr_i))
        np.testing.assert_allclose(result, 1.4)

    def test_predict_ignores_pool_idx(self):
        h = LinearHead("cad")
        params = jnp.array([2.0, 0.5, -1.0])
        x = jnp.array([1.0, 2.0])
        v0 = float(h.predict(params, 0, x))
        v1 = float(h.predict(params, 5, x))
        assert v0 == v1

    def test_regularization_on_W_not_bias(self):
        h = LinearHead("cad", alpha=1.0)
        params = jnp.array([100.0, 3.0, 4.0])
        # reg = 1.0 * (3^2 + 4^2) = 25.0 (bias ignored)
        np.testing.assert_allclose(float(h.regularization(params)), 25.0)

    def test_regularization_alpha_scaling(self):
        h = LinearHead("cad", alpha=0.5)
        params = jnp.array([0.0, 2.0, 0.0])
        # reg = 0.5 * 4.0 = 2.0
        np.testing.assert_allclose(float(h.regularization(params)), 2.0)

    def test_init_default_cadence(self):
        h = LinearHead("cad")
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        assert init.shape == (1 + K_ATTR,)
        np.testing.assert_allclose(init[0], np.log(12.0))
        np.testing.assert_allclose(init[1:], 0.0)

    def test_init_default_gas(self):
        h = LinearHead("gas")
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        np.testing.assert_allclose(init[0], np.log(1.0))

    def test_init_warm_start(self):
        h = LinearHead("log_cadence")
        jdata = _make_fake_jdata()
        warm = {
            POOL_PREFIXES[0]: {"log_cadence": 2.0},
            POOL_PREFIXES[1]: {"log_cadence": 3.0},
        }
        init = h.init(jdata, warm_start=warm)
        assert init.shape == (1 + K_ATTR,)
        # Should have fitted OLS to recover bias/W

    def test_predict_new(self):
        h = LinearHead("cad")
        params = np.array([2.0, 0.5, -1.0])
        x_attr = np.array([1.0, 2.0])
        result = h.predict_new(params, x_attr)
        np.testing.assert_allclose(result, 2.0 + 0.5 - 2.0)

    def test_unpack_result(self):
        h = LinearHead("cad")
        params = np.array([2.0, 0.5, -1.0])
        result = h.unpack_result(params, 3, 2)
        assert "bias_cad" in result
        assert "W_cad" in result
        np.testing.assert_allclose(result["bias_cad"], 2.0)
        np.testing.assert_allclose(result["W_cad"], [0.5, -1.0])

    def test_make_bounds(self):
        h = LinearHead("cad")
        bounds = h.make_bounds(3, 5)
        assert len(bounds) == 6  # 1 + 5


# ── PerPoolNoiseHead ────────────────────────────────────────────────────────


class TestPerPoolNoiseHead:
    def test_n_params(self):
        h = PerPoolNoiseHead()
        assert h.n_params(3, 5) == 3 * K_OBS
        assert h.n_params(2, 7) == 2 * K_OBS

    def test_predict_correct_slice(self):
        h = PerPoolNoiseHead()
        n_pools = 3
        params = jnp.arange(n_pools * K_OBS, dtype=float)
        x_attr_i = jnp.zeros(5)

        for i in range(n_pools):
            result = h.predict(params, i, x_attr_i)
            expected = params[i * K_OBS:(i + 1) * K_OBS]
            np.testing.assert_allclose(result, expected)

    def test_regularization_zero_by_default(self):
        h = PerPoolNoiseHead()
        params = jnp.ones(16)
        assert float(h.regularization(params)) == 0.0

    def test_regularization_with_alpha(self):
        h = PerPoolNoiseHead(alpha=1.0)
        params = jnp.array([3.0, 4.0])
        np.testing.assert_allclose(float(h.regularization(params)), 25.0)

    def test_init_from_ols(self):
        np.random.seed(42)
        h = PerPoolNoiseHead()
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        assert init.shape == (N_POOLS * K_OBS,)
        assert np.all(np.isfinite(init))

    def test_init_warm_start(self):
        h = PerPoolNoiseHead()
        jdata = _make_fake_jdata()
        warm = {
            POOL_PREFIXES[0]: {"noise_coeffs": np.ones(K_OBS) * 5.0},
            POOL_PREFIXES[1]: {"noise_coeffs": np.ones(K_OBS) * 7.0},
        }
        init = h.init(jdata, warm_start=warm)
        assert init.shape == (N_POOLS * K_OBS,)
        np.testing.assert_allclose(init[:K_OBS], 5.0)
        np.testing.assert_allclose(init[K_OBS:], 7.0)

    def test_predict_new_raises(self):
        h = PerPoolNoiseHead()
        with pytest.raises(ValueError, match="cannot predict"):
            h.predict_new(np.zeros(K_OBS * 2), np.zeros(5))

    def test_unpack_result(self):
        h = PerPoolNoiseHead()
        params = np.arange(N_POOLS * K_OBS, dtype=float)
        result = h.unpack_result(params, N_POOLS, K_ATTR)
        assert result["noise_coeffs"].shape == (N_POOLS, K_OBS)


# ── SharedLinearNoiseHead ───────────────────────────────────────────────────


class TestSharedLinearNoiseHead:
    def test_n_params(self):
        h = SharedLinearNoiseHead()
        assert h.n_params(3, 5) == (1 + 5) * K_OBS
        assert h.n_params(10, 7) == (1 + 7) * K_OBS

    def test_predict_bias_plus_dot(self):
        k_attr = 3
        h = SharedLinearNoiseHead()
        W_full = np.zeros((1 + k_attr, K_OBS))
        W_full[0, :] = 1.0  # bias_noise = [1, 1, ..., 1]
        W_full[1, 0] = 2.0  # first feature maps to first noise coeff
        params = jnp.array(W_full.ravel())
        x_attr_i = jnp.array([1.0, 0.0, 0.0])
        result = h.predict(params, 0, x_attr_i)
        assert result.shape == (K_OBS,)
        np.testing.assert_allclose(float(result[0]), 3.0)  # 1 + 2*1
        np.testing.assert_allclose(float(result[1]), 1.0)  # 1 + 0

    def test_predict_ignores_pool_idx(self):
        k_attr = 2
        h = SharedLinearNoiseHead()
        params = jnp.ones((1 + k_attr) * K_OBS)
        x = jnp.array([1.0, 2.0])
        r0 = h.predict(params, 0, x)
        r5 = h.predict(params, 5, x)
        np.testing.assert_allclose(r0, r5)

    def test_regularization_on_W_not_bias(self):
        k_attr = 2
        h = SharedLinearNoiseHead(alpha=1.0)
        W_full = np.zeros((1 + k_attr, K_OBS))
        W_full[0, :] = 100.0  # bias — not regularized
        W_full[1, 0] = 3.0
        W_full[2, 0] = 4.0
        params = jnp.array(W_full.ravel())
        # reg = 1.0 * (9 + 16) = 25.0
        np.testing.assert_allclose(float(h.regularization(params)), 25.0)

    def test_init_default(self):
        np.random.seed(42)
        h = SharedLinearNoiseHead()
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        assert init.shape == ((1 + K_ATTR) * K_OBS,)
        assert np.all(np.isfinite(init))

    def test_predict_new(self):
        k_attr = 2
        h = SharedLinearNoiseHead()
        W_full = np.zeros((1 + k_attr, K_OBS))
        W_full[0, :] = 5.0
        W_full[1, 0] = 1.0
        params = W_full.ravel()
        x_attr = np.array([2.0, 0.0])
        result = h.predict_new(params, x_attr)
        assert result.shape == (K_OBS,)
        np.testing.assert_allclose(result[0], 7.0)
        np.testing.assert_allclose(result[1], 5.0)

    def test_unpack_result(self):
        h = SharedLinearNoiseHead()
        k_attr = 3
        W_full = np.arange((1 + k_attr) * K_OBS, dtype=float)
        result = h.unpack_result(W_full, 2, k_attr)
        assert "bias_noise" in result
        assert "W_noise" in result
        assert result["bias_noise"].shape == (K_OBS,)
        assert result["W_noise"].shape == (k_attr, K_OBS)
