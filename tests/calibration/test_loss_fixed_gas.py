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

    def test_pinned_loss_value(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Loss at known params must match precomputed value."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        fixed_log_gas = jnp.log(jnp.array(1.0))
        loss = pool_loss_fixed_gas(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices
        )
        # cadence=12, gas=1.0, noise=[8,0..0], y=9.0 → pinned
        np.testing.assert_allclose(float(loss), 0.001727, atol=1e-4)

    def test_pinned_loss_at_multiple_gas_values(
        self, synthetic_pool_coeffs, synthetic_x_obs
    ):
        """Pinned loss values at gas=0.01, 1.0, 5.0."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        # Precomputed with _make_params defaults: log_cad=log(12), noise=[8,0..0]
        expected = {0.01: 0.0763, 1.0: 0.00173, 5.0: 0.0313}
        for gas_val, exp_loss in expected.items():
            lg = jnp.log(jnp.array(gas_val))
            loss = float(pool_loss_fixed_gas(
                params, lg, synthetic_pool_coeffs, x_obs, y_obs, day_indices
            ))
            np.testing.assert_allclose(loss, exp_loss, atol=1e-3,
                                       err_msg=f"gas={gas_val}")

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
        assert float(loss) < 1e-10

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

    def test_loss_varies_with_gas_within_grid(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Gas values within grid range [0, 5] should produce distinct losses."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        # Stay within grid range (gas_costs=[0, 1, 5]) to avoid extrapolation plateau
        losses = []
        for gas in [0.01, 0.1, 1.0, 3.0]:
            lg = jnp.log(jnp.array(gas))
            loss = float(pool_loss_fixed_gas(
                params, lg, synthetic_pool_coeffs, x_obs, y_obs, day_indices
            ))
            losses.append(loss)
        # All 4 within-grid gas values should give distinct losses
        assert len(set(f"{l:.8f}" for l in losses)) == 4

    def test_day_indices_affect_loss(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Different day_indices must produce different loss — verifies per-day V_arb is used."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        x_obs = jnp.array(synthetic_x_obs)
        y_obs = jnp.ones(n_obs) * 9.0
        fixed_log_gas = jnp.log(jnp.array(1.0))

        day_idx_all_zero = jnp.zeros(n_obs, dtype=jnp.int32)
        day_idx_varying = jnp.array(np.arange(n_obs) % n_days)

        loss_same = pool_loss_fixed_gas(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_idx_all_zero
        )
        loss_vary = pool_loss_fixed_gas(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_idx_varying
        )
        assert float(loss_same) != float(loss_vary)

    def test_grad_wrt_params(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Gradient w.r.t. params_flat has correct shape and is finite."""
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
        # Gradient should be nonzero (we're not at the optimum)
        assert float(jnp.sum(jnp.abs(grad))) > 1e-10

    def test_grad_changes_with_gas(
        self, synthetic_pool_coeffs, synthetic_x_obs
    ):
        """Gradient w.r.t. params should differ at different fixed gas values.

        fixed_log_gas affects V_arb through grid interpolation, which shifts
        the loss landscape and thus the gradient.
        """
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        grad_fn = jax.grad(pool_loss_fixed_gas, argnums=0)
        grad_low = grad_fn(
            params, jnp.log(jnp.array(0.01)),
            synthetic_pool_coeffs, x_obs, y_obs, day_indices
        )
        grad_high = grad_fn(
            params, jnp.log(jnp.array(10.0)),
            synthetic_pool_coeffs, x_obs, y_obs, day_indices
        )
        # Gradients should differ because V_arb differs
        assert not jnp.allclose(grad_low, grad_high, atol=1e-6)

    def test_extreme_negative_noise_finite(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Loss should remain finite with very negative noise intercept."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas, pack_params_fixed_gas

        # Very negative noise intercept → V_noise ≈ 0, but V_arb still positive
        nc = jnp.zeros(K_OBS).at[0].set(-100.0)
        params = pack_params_fixed_gas(float(jnp.log(jnp.array(12.0))), nc)

        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        day_indices = jnp.array(np.arange(n_obs) % n_days)
        x_obs = jnp.array(synthetic_x_obs)
        y_obs = jnp.ones(n_obs) * 9.0
        fixed_log_gas = jnp.log(jnp.array(1.0))

        loss = pool_loss_fixed_gas(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices
        )
        assert jnp.isfinite(loss)
        # V_noise ≈ 0, so log(V_arb) ≈ 8.5 vs y=9.0 → nonzero loss
        assert float(loss) > 0.01
        # Gradient should also be finite
        grad = jax.grad(pool_loss_fixed_gas, argnums=0)(
            params, fixed_log_gas, synthetic_pool_coeffs, x_obs, y_obs, day_indices
        )
        assert jnp.all(jnp.isfinite(grad))

    def test_boundary_clamp_cadence(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Cadence below grid min should clamp — loss at cad=0.5 equals cad=1.0."""
        from quantammsim.calibration.loss import pack_params_fixed_gas, pool_loss_fixed_gas

        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        nc = jnp.zeros(K_OBS).at[0].set(8.0)
        fixed_log_gas = jnp.log(jnp.array(1.0))

        params_below = pack_params_fixed_gas(float(jnp.log(jnp.array(0.5))), nc)
        params_at_min = pack_params_fixed_gas(float(jnp.log(jnp.array(1.0))), nc)

        loss_below = pool_loss_fixed_gas(
            params_below, fixed_log_gas, synthetic_pool_coeffs,
            x_obs, y_obs, day_indices,
        )
        loss_at_min = pool_loss_fixed_gas(
            params_at_min, fixed_log_gas, synthetic_pool_coeffs,
            x_obs, y_obs, day_indices,
        )
        np.testing.assert_allclose(float(loss_below), float(loss_at_min), rtol=1e-6)

    def test_boundary_clamp_gas(self, synthetic_pool_coeffs, synthetic_x_obs):
        """Gas above grid max should clamp — loss at gas=10 equals gas=5."""
        from quantammsim.calibration.loss import pool_loss_fixed_gas

        params = self._make_params()
        x_obs, y_obs, day_indices = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )

        loss_above = pool_loss_fixed_gas(
            params, jnp.log(jnp.array(10.0)), synthetic_pool_coeffs,
            x_obs, y_obs, day_indices,
        )
        loss_at_max = pool_loss_fixed_gas(
            params, jnp.log(jnp.array(5.0)), synthetic_pool_coeffs,
            x_obs, y_obs, day_indices,
        )
        np.testing.assert_allclose(float(loss_above), float(loss_at_max), rtol=1e-6)

    def test_k_obs_matches_loss_module(self):
        """K_OBS in conftest must match K_OBS in loss.py."""
        from quantammsim.calibration.loss import K_OBS as K_OBS_IMPL
        assert K_OBS == K_OBS_IMPL
