"""Tests for quantammsim.calibration.heads — pluggable Head components."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, POOL_PREFIXES

from quantammsim.calibration.heads import (
    FixedHead,
    Head,
    LinearHead,
    MLPHead,
    MLPNoiseHead,
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

    def test_mlp_head_is_head(self):
        assert isinstance(MLPHead("cad"), Head)

    def test_mlp_noise_head_is_head(self):
        assert isinstance(MLPNoiseHead(), Head)


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


# ── MLPHead ─────────────────────────────────────────────────────────────────


class TestMLPHead:
    def test_n_params(self):
        h = MLPHead("cad", hidden=16)
        # k_attr=5: 5*16 + 16 + 16 + 1 = 113
        assert h.n_params(3, 5) == 113
        # k_attr=7: 7*16 + 16 + 16 + 1 = 145
        assert h.n_params(3, 7) == 145

    def test_n_params_custom_hidden(self):
        h = MLPHead("cad", hidden=8)
        # k_attr=5: 5*8 + 8 + 8 + 1 = 57
        assert h.n_params(3, 5) == 57

    def test_predict_with_zero_W2_equals_b2(self):
        """With W2=0, output should be b2 regardless of input."""
        k_attr = 3
        h = MLPHead("cad", hidden=4)
        n_p = h.n_params(1, k_attr)
        params = np.zeros(n_p)
        params[-1] = 2.5  # b2
        x_attr_i = jnp.array([1.0, 2.0, 3.0])
        result = float(h.predict(jnp.array(params), 0, x_attr_i))
        np.testing.assert_allclose(result, 2.5)

    def test_predict_nonlinear(self):
        """MLP should produce different outputs for different inputs."""
        k_attr = 3
        h = MLPHead("cad", hidden=4, seed=42)
        jdata = _make_fake_jdata()
        # Override x_attr to have k_attr=3
        from quantammsim.calibration.joint_fit import JointData
        jdata = JointData(
            pool_data=jdata.pool_data,
            x_attr=jnp.array(np.random.randn(N_POOLS, k_attr)),
            pool_ids=jdata.pool_ids,
            attr_names=[f"a{i}" for i in range(k_attr)],
        )
        init = jnp.array(h.init(jdata))
        # Set W1 to nonzero so ReLU activations vary
        np.random.seed(42)
        W1 = np.random.randn(k_attr * 4) * 0.5
        init = init.at[:k_attr * 4].set(jnp.array(W1))
        # Set W2 to nonzero so output varies
        init = init.at[k_attr * 4 + 4:k_attr * 4 + 8].set(jnp.ones(4) * 0.1)

        x1 = jnp.array([1.0, 0.0, 0.0])
        x2 = jnp.array([0.0, 1.0, 0.0])
        v1 = float(h.predict(init, 0, x1))
        v2 = float(h.predict(init, 0, x2))
        assert v1 != v2, "MLP should produce different outputs for different inputs"

    def test_predict_ignores_pool_idx(self):
        """MLP output depends only on x_attr, not pool_idx."""
        k_attr = 3
        h = MLPHead("cad", hidden=4)
        params = jnp.ones(h.n_params(5, k_attr)) * 0.1
        x = jnp.array([1.0, 2.0, 3.0])
        v0 = float(h.predict(params, 0, x))
        v3 = float(h.predict(params, 3, x))
        assert v0 == v3

    def test_regularization_on_weights_not_biases(self):
        k_attr = 2
        h_alpha1 = MLPHead("cad", hidden=2, alpha=1.0)
        # Layout: W1(2*2=4), b1(2), W2(2), b2(1) = 9 params
        params = np.zeros(9)
        params[0] = 3.0  # W1[0,0]
        params[1] = 4.0  # W1[0,1]
        # b1 = 0 (indices 4,5)
        params[6] = 1.0  # W2[0]
        params[7] = 2.0  # W2[1]
        params[8] = 999.0  # b2 — should not be regularized
        # reg = 1.0 * (9 + 16 + 1 + 4) = 30.0
        result = float(h_alpha1.regularization(jnp.array(params)))
        np.testing.assert_allclose(result, 30.0)

    def test_regularization_alpha_scaling(self):
        k_attr = 2
        h = MLPHead("cad", hidden=2, alpha=0.5)
        params = np.zeros(9)
        params[0] = 2.0  # W1 weight
        # reg = 0.5 * 4.0 = 2.0
        np.testing.assert_allclose(float(h.regularization(jnp.array(params))), 2.0)

    def test_init_default_cadence(self):
        h = MLPHead("cad", hidden=4)
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        n_p = h.n_params(N_POOLS, K_ATTR)
        assert init.shape == (n_p,)
        assert np.all(np.isfinite(init))
        # b2 should be log(12)
        np.testing.assert_allclose(init[-1], np.log(12.0))

    def test_init_default_gas(self):
        h = MLPHead("gas", hidden=4)
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        # b2 should be log(1) = 0
        np.testing.assert_allclose(init[-1], 0.0)

    def test_init_W2_is_small(self):
        """W2 should be small random at init (not zero) for L-BFGS conditioning."""
        h = MLPHead("cad", hidden=4)
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        # W2 is at [k_attr*hidden + hidden : k_attr*hidden + 2*hidden]
        w2_start = K_ATTR * 4 + 4
        w2_end = w2_start + 4
        w2 = init[w2_start:w2_end]
        assert np.all(np.abs(w2) < 0.1), "W2 should be small"
        assert np.any(w2 != 0.0), "W2 should not be exactly zero"

    def test_init_warm_start(self):
        h = MLPHead("log_cadence", hidden=4)
        jdata = _make_fake_jdata()
        warm = {
            POOL_PREFIXES[0]: {"log_cadence": 2.0},
            POOL_PREFIXES[1]: {"log_cadence": 3.0},
        }
        init = h.init(jdata, warm_start=warm)
        # b2 should be mean of warm-start values
        np.testing.assert_allclose(init[-1], 2.5)

    def test_predict_new(self):
        k_attr = 3
        h = MLPHead("cad", hidden=4)
        n_p = h.n_params(1, k_attr)
        params = np.zeros(n_p)
        params[-1] = 2.5  # b2
        x_attr = np.array([1.0, 2.0, 3.0])
        result = h.predict_new(params, x_attr)
        np.testing.assert_allclose(result, 2.5)

    def test_predict_new_matches_predict(self):
        """predict_new should give same result as predict for same input."""
        k_attr = 3
        h = MLPHead("cad", hidden=4, seed=42)
        n_p = h.n_params(1, k_attr)
        np.random.seed(99)
        params = np.random.randn(n_p) * 0.1
        x_attr = np.array([0.5, -1.0, 2.0])

        jax_result = float(h.predict(jnp.array(params), 0, jnp.array(x_attr)))
        np_result = h.predict_new(params, x_attr)
        np.testing.assert_allclose(jax_result, np_result, rtol=1e-6)

    def test_unpack_result(self):
        k_attr = 3
        h = MLPHead("cad", hidden=4)
        n_p = h.n_params(1, k_attr)
        params = np.arange(n_p, dtype=float)
        result = h.unpack_result(params, 2, k_attr)
        assert f"mlp_cad_W1" in result
        assert f"mlp_cad_b1" in result
        assert f"mlp_cad_W2" in result
        assert f"mlp_cad_b2" in result
        assert result["mlp_cad_W1"].shape == (k_attr, 4)
        assert result["mlp_cad_b1"].shape == (4,)
        assert result["mlp_cad_W2"].shape == (4,)

    def test_make_bounds(self):
        h = MLPHead("cad", hidden=4)
        bounds = h.make_bounds(3, 5)
        assert len(bounds) == h.n_params(3, 5)

    def test_jax_differentiable(self):
        """MLP predict should be JAX-differentiable."""
        import jax
        k_attr = 3
        h = MLPHead("cad", hidden=4)
        n_p = h.n_params(1, k_attr)
        np.random.seed(42)
        params = jnp.array(np.random.randn(n_p) * 0.1)
        x_attr_i = jnp.array([1.0, 2.0, 3.0])

        def loss(p):
            return h.predict(p, 0, x_attr_i) ** 2

        grad = jax.grad(loss)(params)
        assert grad.shape == params.shape
        assert jnp.all(jnp.isfinite(grad))


# ── MLPNoiseHead ────────────────────────────────────────────────────────────


class TestMLPNoiseHead:
    def test_n_params(self):
        h = MLPNoiseHead(hidden=16)
        # k_attr=5: 5*16 + 16 + 16*8 + 8 = 80+16+128+8 = 232
        assert h.n_params(3, 5) == 232
        # k_attr=7: 7*16 + 16 + 16*8 + 8 = 112+16+128+8 = 264
        assert h.n_params(3, 7) == 264

    def test_n_params_custom_hidden(self):
        h = MLPNoiseHead(hidden=8)
        # k_attr=5: 5*8 + 8 + 8*8 + 8 = 40+8+64+8 = 120
        assert h.n_params(3, 5) == 120

    def test_predict_output_shape(self):
        k_attr = 3
        h = MLPNoiseHead(hidden=4)
        n_p = h.n_params(1, k_attr)
        params = jnp.zeros(n_p)
        x_attr_i = jnp.array([1.0, 2.0, 3.0])
        result = h.predict(params, 0, x_attr_i)
        assert result.shape == (K_OBS,)

    def test_predict_with_zero_W2_equals_b2(self):
        """With W2=0, output should be b2 regardless of input."""
        k_attr = 3
        h = MLPNoiseHead(hidden=4)
        n_p = h.n_params(1, k_attr)
        params = np.zeros(n_p)
        # b2 is the last K_OBS elements
        params[-K_OBS:] = np.arange(K_OBS) + 1.0
        x_attr_i = jnp.array([1.0, 2.0, 3.0])
        result = h.predict(jnp.array(params), 0, x_attr_i)
        np.testing.assert_allclose(result, np.arange(K_OBS) + 1.0)

    def test_predict_nonlinear(self):
        """MLP should produce different outputs for different inputs."""
        k_attr = 3
        h = MLPNoiseHead(hidden=4, seed=42)
        n_p = h.n_params(1, k_attr)
        np.random.seed(42)
        params = jnp.array(np.random.randn(n_p) * 0.1)
        x1 = jnp.array([1.0, 0.0, 0.0])
        x2 = jnp.array([0.0, 1.0, 0.0])
        v1 = h.predict(params, 0, x1)
        v2 = h.predict(params, 0, x2)
        assert not jnp.allclose(v1, v2), "Should produce different outputs"

    def test_predict_ignores_pool_idx(self):
        k_attr = 3
        h = MLPNoiseHead(hidden=4)
        params = jnp.ones(h.n_params(5, k_attr)) * 0.1
        x = jnp.array([1.0, 2.0, 3.0])
        v0 = h.predict(params, 0, x)
        v3 = h.predict(params, 3, x)
        np.testing.assert_allclose(v0, v3)

    def test_regularization_on_weights_not_biases(self):
        k_attr = 2
        h = MLPNoiseHead(hidden=2, alpha=1.0)
        # Layout: W1(2*2=4), b1(2), W2(2*8=16), b2(8) = 30 params
        n_p = h.n_params(1, k_attr)
        assert n_p == 30
        params = np.zeros(n_p)
        params[0] = 3.0  # W1[0,0]
        params[1] = 4.0  # W1[0,1]
        # b1 at indices 4,5 — not regularized
        params[6] = 1.0  # W2[0,0]
        params[7] = 2.0  # W2[0,1]
        params[-1] = 999.0  # b2[-1] — not regularized
        # reg = 1.0 * (9 + 16 + 1 + 4) = 30.0
        result = float(h.regularization(jnp.array(params)))
        np.testing.assert_allclose(result, 30.0)

    def test_init_default(self):
        np.random.seed(42)
        h = MLPNoiseHead(hidden=4)
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        n_p = h.n_params(N_POOLS, K_ATTR)
        assert init.shape == (n_p,)
        assert np.all(np.isfinite(init))

    def test_init_W2_is_small(self):
        """W2 should be small random at init (not zero) for L-BFGS conditioning."""
        h = MLPNoiseHead(hidden=4)
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        # W2 starts at k_attr*hidden + hidden
        w2_start = K_ATTR * 4 + 4
        w2_end = w2_start + 4 * K_OBS
        w2 = init[w2_start:w2_end]
        assert np.all(np.abs(w2) < 0.1), "W2 should be small"
        assert np.any(w2 != 0.0), "W2 should not be exactly zero"

    def test_init_b2_from_ols(self):
        """b2 should be pooled OLS noise coefficients."""
        np.random.seed(42)
        h = MLPNoiseHead(hidden=4)
        jdata = _make_fake_jdata()
        init = h.init(jdata)
        b2 = init[-K_OBS:]
        assert np.all(np.isfinite(b2))
        # Should be nonzero (OLS on random data)
        assert np.any(b2 != 0.0)

    def test_init_warm_start(self):
        h = MLPNoiseHead(hidden=4)
        jdata = _make_fake_jdata()
        warm = {
            POOL_PREFIXES[0]: {"noise_coeffs": np.ones(K_OBS) * 5.0},
            POOL_PREFIXES[1]: {"noise_coeffs": np.ones(K_OBS) * 7.0},
        }
        init = h.init(jdata, warm_start=warm)
        b2 = init[-K_OBS:]
        # b2 should be mean of warm-start noise: (5+7)/2 = 6
        np.testing.assert_allclose(b2, 6.0)

    def test_predict_new(self):
        k_attr = 3
        h = MLPNoiseHead(hidden=4)
        n_p = h.n_params(1, k_attr)
        params = np.zeros(n_p)
        params[-K_OBS:] = np.arange(K_OBS) + 1.0  # b2
        x_attr = np.array([1.0, 2.0, 3.0])
        result = h.predict_new(params, x_attr)
        assert result.shape == (K_OBS,)
        np.testing.assert_allclose(result, np.arange(K_OBS) + 1.0)

    def test_predict_new_matches_predict(self):
        k_attr = 3
        h = MLPNoiseHead(hidden=4, seed=42)
        n_p = h.n_params(1, k_attr)
        np.random.seed(99)
        params = np.random.randn(n_p) * 0.1
        x_attr = np.array([0.5, -1.0, 2.0])

        jax_result = np.array(h.predict(jnp.array(params), 0, jnp.array(x_attr)))
        np_result = h.predict_new(params, x_attr)
        np.testing.assert_allclose(jax_result, np_result, rtol=1e-6)

    def test_unpack_result(self):
        k_attr = 3
        h = MLPNoiseHead(hidden=4)
        n_p = h.n_params(1, k_attr)
        params = np.arange(n_p, dtype=float)
        result = h.unpack_result(params, 2, k_attr)
        assert "mlp_noise_W1" in result
        assert "mlp_noise_b1" in result
        assert "mlp_noise_W2" in result
        assert "mlp_noise_b2" in result
        assert result["mlp_noise_W1"].shape == (k_attr, 4)
        assert result["mlp_noise_b1"].shape == (4,)
        assert result["mlp_noise_W2"].shape == (4, K_OBS)
        assert result["mlp_noise_b2"].shape == (K_OBS,)

    def test_make_bounds(self):
        h = MLPNoiseHead(hidden=4)
        bounds = h.make_bounds(3, 5)
        assert len(bounds) == h.n_params(3, 5)

    def test_jax_differentiable(self):
        import jax
        k_attr = 3
        h = MLPNoiseHead(hidden=4)
        n_p = h.n_params(1, k_attr)
        np.random.seed(42)
        params = jnp.array(np.random.randn(n_p) * 0.1)
        x_attr_i = jnp.array([1.0, 2.0, 3.0])

        def loss(p):
            return jnp.sum(h.predict(p, 0, x_attr_i) ** 2)

        grad = jax.grad(loss)(params)
        assert grad.shape == params.shape
        assert jnp.all(jnp.isfinite(grad))
