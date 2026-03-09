"""Tests for fixed-gas mode in quantammsim.calibration.per_pool_fit."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, N_DAYS, POOL_IDS_FULL, POOL_PREFIXES


class TestInitialGuessFixedGas:
    """Test make_initial_guess_fixed_gas."""

    def test_shape(self, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import make_initial_guess_fixed_gas

        n_obs = synthetic_x_obs.shape[0]
        y_obs = np.ones(n_obs) * 9.0
        init = make_initial_guess_fixed_gas(synthetic_x_obs, y_obs)
        assert init.shape == (1 + K_OBS,)

    def test_one_shorter_than_free(self, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import (
            make_initial_guess,
            make_initial_guess_fixed_gas,
        )

        n_obs = synthetic_x_obs.shape[0]
        y_obs = np.ones(n_obs) * 9.0
        free = make_initial_guess(synthetic_x_obs, y_obs)
        fixed = make_initial_guess_fixed_gas(synthetic_x_obs, y_obs)
        assert free.shape[0] == fixed.shape[0] + 1

    def test_log_cadence_default(self, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import make_initial_guess_fixed_gas

        n_obs = synthetic_x_obs.shape[0]
        y_obs = np.ones(n_obs) * 9.0
        init = make_initial_guess_fixed_gas(synthetic_x_obs, y_obs)
        np.testing.assert_allclose(init[0], np.log(12.0), atol=0.01)

    def test_noise_from_ols(self, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import make_initial_guess_fixed_gas

        n_obs = synthetic_x_obs.shape[0]
        y_obs = np.ones(n_obs) * 9.0
        init = make_initial_guess_fixed_gas(synthetic_x_obs, y_obs)
        noise_coeffs = init[1:]
        assert len(noise_coeffs) == K_OBS
        assert np.all(np.isfinite(noise_coeffs))

    def test_noise_matches_free_gas_noise(self, synthetic_x_obs):
        """OLS noise coeffs should be identical for free and fixed-gas init."""
        from quantammsim.calibration.per_pool_fit import (
            make_initial_guess,
            make_initial_guess_fixed_gas,
        )

        n_obs = synthetic_x_obs.shape[0]
        y_obs = np.ones(n_obs) * 9.0
        free = make_initial_guess(synthetic_x_obs, y_obs)
        fixed = make_initial_guess_fixed_gas(synthetic_x_obs, y_obs)
        np.testing.assert_allclose(free[2:], fixed[1:])


class TestFitSinglePoolFixedGas:
    """Test fit_single_pool with fixed_gas_usd."""

    def _make_inputs(self, synthetic_pool_coeffs, synthetic_x_obs):
        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        day_indices = np.arange(n_obs) % n_days
        y_obs = np.ones(n_obs) * 9.0
        return synthetic_x_obs, y_obs, day_indices

    def test_returns_result_dict(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx, fixed_gas_usd=1.0
        )
        for key in [
            "log_cadence", "log_gas", "noise_coeffs", "loss",
            "converged", "gas_fixed",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_gas_fixed_flag(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx, fixed_gas_usd=1.0
        )
        assert result["gas_fixed"] is True

    def test_free_gas_flag(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx
        )
        assert result["gas_fixed"] is False

    def test_gas_usd_pinned(self, synthetic_pool_coeffs, synthetic_x_obs):
        """gas_usd in result must exactly match the fixed value."""
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        for gas_val in [0.001, 0.01, 0.5, 1.0, 5.0]:
            result = fit_single_pool(
                synthetic_pool_coeffs, x_obs, y_obs, day_idx,
                fixed_gas_usd=gas_val,
            )
            assert result["gas_usd"] == gas_val

    def test_log_gas_pinned(self, synthetic_pool_coeffs, synthetic_x_obs):
        """log_gas must equal log(fixed_gas_usd)."""
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx, fixed_gas_usd=2.5,
        )
        np.testing.assert_allclose(
            result["log_gas"], np.log(2.5), rtol=1e-6,
        )

    def test_cadence_in_range(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx, fixed_gas_usd=1.0,
        )
        cadence = result["cadence_minutes"]
        assert 1.0 <= cadence <= 60.0

    def test_noise_coeffs_length(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx, fixed_gas_usd=1.0,
        )
        assert len(result["noise_coeffs"]) == K_OBS

    def test_loss_decreases_from_init(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pack_params_fixed_gas, pool_loss_fixed_gas
        from quantammsim.calibration.per_pool_fit import (
            fit_single_pool,
            make_initial_guess_fixed_gas,
        )

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        init = make_initial_guess_fixed_gas(x_obs, y_obs)
        fixed_log_gas = jnp.float64(np.log(1.0))
        init_loss = float(pool_loss_fixed_gas(
            jnp.array(init), fixed_log_gas, synthetic_pool_coeffs,
            jnp.array(x_obs), jnp.array(y_obs), jnp.array(day_idx),
        ))

        result = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx, fixed_gas_usd=1.0,
        )
        assert result["loss"] <= init_loss

    def test_different_fixed_gas_different_cadence(
        self, synthetic_pool_coeffs, synthetic_x_obs
    ):
        """Different gas values should (generally) lead to different fitted cadences."""
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        r_low = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx, fixed_gas_usd=0.001,
        )
        r_high = fit_single_pool(
            synthetic_pool_coeffs, x_obs, y_obs, day_idx, fixed_gas_usd=5.0,
        )
        # Cadences should differ (gas-cadence tradeoff)
        assert abs(r_low["log_cadence"] - r_high["log_cadence"]) > 0.01


class TestFitAllPoolsFixedGas:
    """Test fit_all_pools with fix_gas_to_chain=True."""

    def _make_matched(self, synthetic_daily_grid, synthetic_panel, tmp_path):
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )
        return match_grids_to_panel(str(grid_dir), synthetic_panel)

    def test_all_gas_fixed(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.per_pool_fit import fit_all_pools

        matched = self._make_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        results = fit_all_pools(matched, fix_gas_to_chain=True)
        for prefix, res in results.items():
            assert res["gas_fixed"] is True

    def test_gas_matches_chain(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        """Each pool's gas_usd should match CHAIN_GAS_USD[chain]."""
        from quantammsim.calibration.loss import CHAIN_GAS_USD
        from quantammsim.calibration.per_pool_fit import fit_all_pools

        matched = self._make_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        results = fit_all_pools(matched, fix_gas_to_chain=True)
        for prefix, res in results.items():
            chain = res["chain"]
            expected = CHAIN_GAS_USD.get(chain, 1.0)
            assert res["gas_usd"] == expected, (
                f"{prefix} ({chain}): gas_usd={res['gas_usd']} != {expected}"
            )

    def test_free_gas_not_fixed(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.per_pool_fit import fit_all_pools

        matched = self._make_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        results = fit_all_pools(matched, fix_gas_to_chain=False)
        for prefix, res in results.items():
            assert res["gas_fixed"] is False
