"""Tests for quantammsim.calibration.loss — per-pool loss function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, N_DAYS


class TestNoiseVolume:
    """Test noise_volume: V_noise = exp(x_obs @ noise_coeffs)."""

    def test_noise_volume_shape(self, synthetic_x_obs):
        from quantammsim.calibration.loss import noise_volume

        coeffs = jnp.zeros(K_OBS)
        v = noise_volume(coeffs, jnp.array(synthetic_x_obs))
        assert v.shape == (synthetic_x_obs.shape[0],)

    def test_noise_volume_positive(self, synthetic_x_obs):
        from quantammsim.calibration.loss import noise_volume

        coeffs = jnp.ones(K_OBS) * 0.1
        v = noise_volume(coeffs, jnp.array(synthetic_x_obs))
        assert jnp.all(v > 0)

    def test_noise_volume_intercept_only(self, synthetic_x_obs):
        from quantammsim.calibration.loss import noise_volume

        c = 5.0
        coeffs = jnp.zeros(K_OBS).at[0].set(c)
        v = noise_volume(coeffs, jnp.array(synthetic_x_obs))
        # x_obs[:, 0] is all 1s, so V_noise should be close to exp(c)
        # (not exactly, because other columns are nonzero and contribute 0*x)
        np.testing.assert_allclose(v, jnp.exp(c), rtol=1e-5)

    def test_noise_volume_tvl_effect(self, synthetic_x_obs):
        from quantammsim.calibration.loss import noise_volume

        # Positive TVL coefficient: higher TVL → higher noise
        coeffs = jnp.zeros(K_OBS).at[1].set(1.0)
        v = noise_volume(coeffs, jnp.array(synthetic_x_obs))
        tvl_col = synthetic_x_obs[:, 1]
        # Sort by TVL, check volume is monotone
        order = np.argsort(tvl_col)
        assert np.all(np.diff(np.array(v[order])) >= -1e-6)


class TestPoolLoss:
    """Test pool_loss: per-pool log-space L2 loss with per-day V_arb."""

    def _make_params(self, log_cad=None, log_gas=None, noise_coeffs=None):
        from quantammsim.calibration.loss import pack_params

        if log_cad is None:
            log_cad = float(jnp.log(jnp.array(12.0)))
        if log_gas is None:
            log_gas = float(jnp.log(jnp.array(1.0)))
        if noise_coeffs is None:
            noise_coeffs = jnp.zeros(K_OBS).at[0].set(8.0)
        return pack_params(log_cad, log_gas, noise_coeffs)

    def _make_inputs(self, synthetic_pool_coeffs, synthetic_x_obs):
        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        day_indices = jnp.array(np.arange(n_obs) % n_days)
        y_obs = jnp.ones(n_obs) * 9.0
        return jnp.array(synthetic_x_obs), y_obs, day_indices

    def test_loss_scalar_output(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        loss = pool_loss(params, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        assert loss.shape == ()

    def test_loss_zero_when_perfect(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
        from quantammsim.calibration.loss import noise_volume, pack_params, pool_loss

        log_cad = jnp.log(jnp.array(12.0))
        log_gas = jnp.log(jnp.array(1.0))
        noise_coeffs = jnp.zeros(K_OBS).at[0].set(8.0)

        # Compute what V_pred would be
        v_arb_all = interpolate_pool_daily(synthetic_pool_coeffs, log_cad, jnp.exp(log_gas))
        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        day_indices = jnp.array(np.arange(n_obs) % n_days)
        v_arb = v_arb_all[day_indices]
        v_noise = noise_volume(noise_coeffs, jnp.array(synthetic_x_obs))
        y_obs = jnp.log(jnp.maximum(v_arb + v_noise, 1e-6))

        params = pack_params(float(log_cad), float(log_gas), noise_coeffs)
        loss = pool_loss(params, synthetic_pool_coeffs, jnp.array(synthetic_x_obs),
                         y_obs, day_indices)
        assert float(loss) < 1e-6

    def test_loss_positive(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        loss = pool_loss(params, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        assert float(loss) >= 0

    def test_loss_increases_with_error(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        params_ok = self._make_params(noise_coeffs=jnp.zeros(K_OBS).at[0].set(8.0))
        params_bad = self._make_params(noise_coeffs=jnp.zeros(K_OBS).at[0].set(20.0))

        loss_ok = pool_loss(params_ok, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        loss_bad = pool_loss(params_bad, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        assert float(loss_bad) > float(loss_ok)

    def test_loss_differentiable(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        grad_fn = jax.grad(pool_loss, argnums=0)
        g = grad_fn(params, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        assert g.shape == params.shape

    def test_loss_grad_nonzero(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        grad_fn = jax.grad(pool_loss, argnums=0)
        g = grad_fn(params, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        assert float(jnp.sum(jnp.abs(g))) > 0

    def test_loss_grad_wrt_log_cadence(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        grad_fn = jax.grad(pool_loss, argnums=0)
        g = grad_fn(params, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        assert jnp.isfinite(g[0])  # log_cadence gradient

    def test_loss_grad_wrt_log_gas(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        grad_fn = jax.grad(pool_loss, argnums=0)
        g = grad_fn(params, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        assert jnp.isfinite(g[1])  # log_gas gradient

    def test_loss_grad_wrt_noise_coeffs(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        grad_fn = jax.grad(pool_loss, argnums=0)
        g = grad_fn(params, synthetic_pool_coeffs, x_obs, y_obs, day_indices)
        assert jnp.all(jnp.isfinite(g[2:]))  # noise_coeffs gradients

    def test_loss_uses_per_day_varb(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
        from quantammsim.calibration.loss import pool_loss, unpack_params

        params = self._make_params()
        log_cad, log_gas, _ = unpack_params(params)
        v_arb = interpolate_pool_daily(
            synthetic_pool_coeffs, jnp.array(log_cad), jnp.exp(jnp.array(log_gas))
        )
        # Per-day V_arb should vary across days
        assert float(jnp.std(v_arb)) > 0

    def test_day_indices_alignment(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss

        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        params = self._make_params()
        x_obs = jnp.array(synthetic_x_obs)
        y_obs = jnp.ones(n_obs) * 9.0

        # Different day_indices should give different loss
        day_idx_a = jnp.zeros(n_obs, dtype=jnp.int32)  # all same day
        day_idx_b = jnp.array(np.arange(n_obs) % n_days)  # varying days

        loss_a = pool_loss(params, synthetic_pool_coeffs, x_obs, y_obs, day_idx_a)
        loss_b = pool_loss(params, synthetic_pool_coeffs, x_obs, y_obs, day_idx_b)
        # Different day mappings → different losses
        assert float(loss_a) != float(loss_b)


class TestPackUnpack:
    """Test pack/unpack parameter roundtrip."""

    def test_roundtrip(self):
        from quantammsim.calibration.loss import pack_params, unpack_params

        log_cad = 2.5
        log_gas = -0.3
        noise_coeffs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        flat = pack_params(log_cad, log_gas, noise_coeffs)
        lc, lg, nc = unpack_params(flat)

        np.testing.assert_allclose(lc, log_cad)
        np.testing.assert_allclose(lg, log_gas)
        np.testing.assert_allclose(nc, noise_coeffs)

    def test_pack_shape(self):
        from quantammsim.calibration.loss import pack_params

        flat = pack_params(1.0, 2.0, jnp.zeros(K_OBS))
        assert flat.shape == (2 + K_OBS,)
