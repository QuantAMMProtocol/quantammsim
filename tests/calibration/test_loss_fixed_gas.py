"""Tests for fixed-gas extensions in quantammsim.calibration.loss."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, N_DAYS


class TestChainGasUSD:
    """Test CHAIN_GAS_USD constants are correct and complete."""

    def test_known_chains(self):
        from quantammsim.calibration.loss import CHAIN_GAS_USD

        assert CHAIN_GAS_USD["MAINNET"] == 1.0
        assert CHAIN_GAS_USD["POLYGON"] == 0.005
        assert CHAIN_GAS_USD["GNOSIS"] == 0.001
        assert CHAIN_GAS_USD["ARBITRUM"] == 0.01
        assert CHAIN_GAS_USD["BASE"] == 0.005
        assert CHAIN_GAS_USD["SONIC"] == 0.005

    def test_all_values_positive(self):
        from quantammsim.calibration.loss import CHAIN_GAS_USD

        for chain, cost in CHAIN_GAS_USD.items():
            assert cost > 0, f"{chain} gas cost must be positive"

    def test_mainnet_most_expensive(self):
        from quantammsim.calibration.loss import CHAIN_GAS_USD

        mainnet = CHAIN_GAS_USD["MAINNET"]
        for chain, cost in CHAIN_GAS_USD.items():
            if chain != "MAINNET":
                assert cost < mainnet, f"{chain} should be cheaper than MAINNET"

    def test_six_chains(self):
        from quantammsim.calibration.loss import CHAIN_GAS_USD

        assert len(CHAIN_GAS_USD) == 6


class TestPackUnpackFixedGas:
    """Test pack/unpack for fixed-gas param vectors."""

    def test_pack_shape(self):
        from quantammsim.calibration.loss import pack_params_fixed_gas

        flat = pack_params_fixed_gas(2.5, jnp.zeros(K_OBS))
        assert flat.shape == (1 + K_OBS,)

    def test_pack_shape_is_one_shorter_than_free(self):
        from quantammsim.calibration.loss import pack_params, pack_params_fixed_gas

        free = pack_params(2.5, 0.0, jnp.zeros(K_OBS))
        fixed = pack_params_fixed_gas(2.5, jnp.zeros(K_OBS))
        assert free.shape[0] == fixed.shape[0] + 1

    def test_roundtrip(self):
        from quantammsim.calibration.loss import (
            pack_params_fixed_gas,
            unpack_params_fixed_gas,
        )

        log_cad = 2.5
        noise_coeffs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        flat = pack_params_fixed_gas(log_cad, noise_coeffs)
        lc, nc = unpack_params_fixed_gas(flat)

        np.testing.assert_allclose(lc, log_cad)
        np.testing.assert_allclose(nc, noise_coeffs)

    def test_unpack_log_cadence_position(self):
        """log_cadence is the first element."""
        from quantammsim.calibration.loss import pack_params_fixed_gas

        flat = pack_params_fixed_gas(3.14, jnp.ones(K_OBS) * 99.0)
        np.testing.assert_allclose(flat[0], 3.14)

    def test_unpack_noise_coeffs_position(self):
        """noise_coeffs are elements [1:]."""
        from quantammsim.calibration.loss import pack_params_fixed_gas

        nc = jnp.arange(1, K_OBS + 1, dtype=float)
        flat = pack_params_fixed_gas(0.0, nc)
        np.testing.assert_allclose(flat[1:], nc)


class TestPoolLossFixedGas:
    """Test pool_loss_fixed_gas with pinned numerical values."""

    def _make_params(self, log_cad=None, noise_coeffs=None):
        from quantammsim.calibration.loss import pack_params_fixed_gas

        if log_cad is None:
            log_cad = float(jnp.log(jnp.array(12.0)))
        if noise_coeffs is None:
            noise_coeffs = jnp.zeros(K_OBS).at[0].set(8.0)
        return pack_params_fixed_gas(log_cad, noise_coeffs)

    def _make_inputs(self, synthetic_pool_coeffs, synthetic_x_obs):
        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        day_indices = jnp.array(np.arange(n_obs) % n_days)
        y_obs = jnp.ones(n_obs) * 9.0
        return jnp.array(synthetic_x_obs), y_obs, day_indices

    def test_scalar_output(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        fixed_log_gas = jnp.log(jnp.array(1.0))
        loss = pool_loss_fixed_gas(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices
        )
        assert loss.shape == ()

    def test_positive(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        fixed_log_gas = jnp.log(jnp.array(1.0))
        loss = pool_loss_fixed_gas(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices
        )
        assert float(loss) >= 0

    def test_zero_when_perfect(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Construct y_obs = log(V_arb + V_noise) exactly, verify loss ≈ 0."""
        from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
        from quantammsim.calibration.loss import (
            noise_volume,
            pack_params_fixed_gas,
            pool_loss_fixed_gas,
        )

        log_cad = jnp.log(jnp.array(12.0))
        fixed_log_gas = jnp.log(jnp.array(1.0))
        noise_coeffs = jnp.zeros(K_OBS).at[0].set(8.0)

        v_arb_all = interpolate_pool_daily(
            synthetic_pool_coeffs, log_cad, jnp.exp(fixed_log_gas)
        )
        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        day_indices = jnp.array(np.arange(n_obs) % n_days)
        v_arb = v_arb_all[day_indices]
        v_noise = noise_volume(noise_coeffs, jnp.array(synthetic_x_obs))
        y_obs = jnp.log(jnp.maximum(v_arb + v_noise, 1e-6))

        params = pack_params_fixed_gas(float(log_cad), noise_coeffs)
        loss = pool_loss_fixed_gas(
            params, fixed_log_gas, synthetic_pool_coeffs,
            jnp.array(synthetic_x_obs), y_obs, day_indices,
        )
        assert float(loss) < 1e-6

    def test_matches_free_gas_at_same_value(
        self, synthetic_pool_coeffs, synthetic_x_obs
    ):
        """Fixed-gas loss should equal free-gas loss when gas matches."""
        from quantammsim.calibration.loss import (
            pack_params,
            pack_params_fixed_gas,
            pool_loss,
            pool_loss_fixed_gas,
        )

        log_cad = float(jnp.log(jnp.array(12.0)))
        log_gas = float(jnp.log(jnp.array(1.0)))
        noise_coeffs = jnp.zeros(K_OBS).at[0].set(8.0)

        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        free_params = pack_params(log_cad, log_gas, noise_coeffs)
        fixed_params = pack_params_fixed_gas(log_cad, noise_coeffs)

        loss_free = pool_loss(
            free_params, synthetic_pool_coeffs, x_obs, y_obs, day_indices
        )
        loss_fixed = pool_loss_fixed_gas(
            fixed_params, jnp.array(log_gas), synthetic_pool_coeffs,
            x_obs, y_obs, day_indices,
        )
        np.testing.assert_allclose(float(loss_free), float(loss_fixed), rtol=1e-6)

    def test_different_fixed_gas_different_loss(
        self, synthetic_pool_coeffs, synthetic_x_obs
    ):
        """Different gas values should give different losses."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        loss_low = pool_loss_fixed_gas(
            params, jnp.log(jnp.array(0.01)),
            synthetic_pool_coeffs, x_obs, y_obs, day_indices,
        )
        loss_high = pool_loss_fixed_gas(
            params, jnp.log(jnp.array(10.0)),
            synthetic_pool_coeffs, x_obs, y_obs, day_indices,
        )
        assert float(loss_low) != float(loss_high)

    def test_grad_wrt_params_only(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Gradient is only w.r.t. params_flat (argnums=0), not fixed_log_gas."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        fixed_log_gas = jnp.log(jnp.array(1.0))

        grad = jax.grad(pool_loss_fixed_gas, argnums=0)(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices,
        )
        assert grad.shape == (1 + K_OBS,)
        assert jnp.all(jnp.isfinite(grad))

    def test_no_grad_wrt_fixed_gas(self, synthetic_pool_coeffs, synthetic_x_obs):
        """fixed_log_gas should not be in the gradient (it's a constant)."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        fixed_log_gas = jnp.log(jnp.array(1.0))

        # Gradient w.r.t. argnums=0 has shape (1+K_OBS,) — no gas element
        grad = jax.grad(pool_loss_fixed_gas, argnums=0)(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices,
        )
        # If there were a gas gradient, shape would be (2+K_OBS,)
        assert grad.shape[0] == 1 + K_OBS

    def test_grad_wrt_log_cadence_finite(
        self, synthetic_pool_coeffs, synthetic_x_obs
    ):
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        fixed_log_gas = jnp.log(jnp.array(1.0))

        grad = jax.grad(pool_loss_fixed_gas, argnums=0)(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices,
        )
        assert jnp.isfinite(grad[0])  # log_cadence gradient

    def test_grad_wrt_noise_coeffs_finite(
        self, synthetic_pool_coeffs, synthetic_x_obs
    ):
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        fixed_log_gas = jnp.log(jnp.array(1.0))

        grad = jax.grad(pool_loss_fixed_gas, argnums=0)(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices,
        )
        assert jnp.all(jnp.isfinite(grad[1:]))  # noise_coeffs gradients
