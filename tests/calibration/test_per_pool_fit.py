"""Tests for quantammsim.calibration.per_pool_fit — L-BFGS-B per-pool fitting."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, N_DAYS, POOL_IDS_FULL, POOL_PREFIXES


class TestFitSinglePool:
    """Test fit_single_pool: L-BFGS-B optimization for one pool."""

    def _make_inputs(self, synthetic_pool_coeffs, synthetic_x_obs):
        n_obs = synthetic_x_obs.shape[0]
        n_days = int(synthetic_pool_coeffs.values.shape[2])
        day_indices = np.arange(n_obs) % n_days
        y_obs = np.ones(n_obs) * 9.0  # log(V_obs)
        return synthetic_x_obs, y_obs, day_indices

    def test_returns_result_dict(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(synthetic_pool_coeffs, x_obs, y_obs, day_idx)
        assert isinstance(result, dict)
        for key in ["log_cadence", "log_gas", "noise_coeffs", "loss", "converged"]:
            assert key in result, f"Missing key: {key}"

    def test_cadence_in_range(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(synthetic_pool_coeffs, x_obs, y_obs, day_idx)
        cadence = np.exp(result["log_cadence"])
        assert 1.0 <= cadence <= 60.0

    def test_gas_positive(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(synthetic_pool_coeffs, x_obs, y_obs, day_idx)
        assert np.exp(result["log_gas"]) > 0

    def test_noise_coeffs_length(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(synthetic_pool_coeffs, x_obs, y_obs, day_idx)
        assert len(result["noise_coeffs"]) == K_OBS

    def test_loss_decreases_from_init(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.loss import pack_params, pool_loss
        from quantammsim.calibration.per_pool_fit import (
            fit_single_pool,
            make_initial_guess,
        )

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        init = make_initial_guess(x_obs, y_obs)
        init_loss = float(pool_loss(
            jnp.array(init), synthetic_pool_coeffs,
            jnp.array(x_obs), jnp.array(y_obs), jnp.array(day_idx),
        ))

        result = fit_single_pool(synthetic_pool_coeffs, x_obs, y_obs, day_idx)
        assert result["loss"] <= init_loss

    def test_converged_flag(self, synthetic_pool_coeffs, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import fit_single_pool

        x_obs, y_obs, day_idx = self._make_inputs(
            synthetic_pool_coeffs, synthetic_x_obs
        )
        result = fit_single_pool(synthetic_pool_coeffs, x_obs, y_obs, day_idx)
        assert isinstance(result["converged"], bool)


class TestFitAllPools:
    """Test fit_all_pools: fit all matched pools."""

    def test_returns_dict_per_pool(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.per_pool_fit import fit_all_pools
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        results = fit_all_pools(matched)
        assert isinstance(results, dict)

    def test_all_pools_have_results(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.per_pool_fit import fit_all_pools
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        results = fit_all_pools(matched)
        for prefix in matched:
            assert prefix in results

    def test_results_have_metadata(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.per_pool_fit import fit_all_pools
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        results = fit_all_pools(matched)
        for prefix, res in results.items():
            assert "chain" in res
            assert "fee" in res
            assert "tokens" in res


class TestInitialGuess:
    """Test make_initial_guess: reasonable starting point."""

    def test_default_init_reasonable(self, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import make_initial_guess

        n_obs = synthetic_x_obs.shape[0]
        y_obs = np.ones(n_obs) * 9.0
        init = make_initial_guess(synthetic_x_obs, y_obs)
        assert len(init) == 2 + K_OBS
        # log_cadence ~ log(12)
        np.testing.assert_allclose(init[0], np.log(12.0), atol=0.1)
        # log_gas ~ log(1.0)
        np.testing.assert_allclose(init[1], np.log(1.0), atol=0.1)

    def test_init_noise_from_ols(self, synthetic_x_obs):
        from quantammsim.calibration.per_pool_fit import make_initial_guess

        n_obs = synthetic_x_obs.shape[0]
        y_obs = np.ones(n_obs) * 9.0
        init = make_initial_guess(synthetic_x_obs, y_obs)
        noise_coeffs = init[2:]
        # OLS should give finite values
        assert np.all(np.isfinite(noise_coeffs))
