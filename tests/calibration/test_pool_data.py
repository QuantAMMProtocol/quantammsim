"""Tests for quantammsim.calibration.pool_data — data assembly."""

import numpy as np
import pandas as pd
import pytest

from tests.calibration.conftest import (
    K_OBS,
    N_DAYS,
    POOL_IDS_FULL,
    POOL_PREFIXES,
)


class TestEncodeTokens:
    """Test encode_tokens: token index, assignments, and covariate matrix."""

    def _get_matched(self, synthetic_daily_grid, synthetic_panel, tmp_path):
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )
        return match_grids_to_panel(str(grid_dir), synthetic_panel)

    def test_returns_expected_keys(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import encode_tokens

        matched = self._get_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        result = encode_tokens(matched)
        expected_keys = {
            "token_index", "token_a_idx", "token_b_idx",
            "x_token", "chain_idx", "chain_index",
            "log_fees", "n_tokens", "n_chains",
        }
        assert expected_keys.issubset(result.keys())

    def test_unique_tokens_discovered(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import encode_tokens

        matched = self._get_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        result = encode_tokens(matched)
        # Synthetic panel has tokens: BTC, ETH (pool 0) and AAVE, ETH (pool 1)
        # Unique tokens: AAVE, BTC, ETH (sorted)
        assert result["n_tokens"] == 3
        assert set(result["token_index"].keys()) == {"AAVE", "BTC", "ETH"}
        # Indices should be contiguous 0..2
        assert set(result["token_index"].values()) == {0, 1, 2}

    def test_x_token_shape_and_intercept(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import encode_tokens

        matched = self._get_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        result = encode_tokens(matched)
        x_token = result["x_token"]
        assert x_token.shape[0] == result["n_tokens"]  # 3 tokens
        assert x_token.shape[1] >= 4  # at least intercept + 3 binary flags
        # Intercept column is all 1s
        np.testing.assert_array_equal(x_token[:, 0], 1.0)

    def test_token_classifications(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import encode_tokens

        matched = self._get_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        result = encode_tokens(matched)
        ti = result["token_index"]
        x_tok = result["x_token"]
        # Column layout: [intercept, log_mcap, is_stable, is_eth_deriv, is_L1_native]
        # ETH: is_eth_derivative=1, is_L1_native=1
        assert x_tok[ti["ETH"], 3] == 1.0  # is_eth_derivative
        assert x_tok[ti["ETH"], 4] == 1.0  # is_L1_native
        # AAVE: none of the binary flags
        assert x_tok[ti["AAVE"], 2] == 0.0  # not stable
        assert x_tok[ti["AAVE"], 3] == 0.0  # not eth_deriv
        assert x_tok[ti["AAVE"], 4] == 0.0  # not L1_native

    def test_pool_token_mapping(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import encode_tokens

        matched = self._get_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        result = encode_tokens(matched)
        ti = result["token_index"]

        # Pool 0 (first sorted prefix): tokens = "BTC,ETH"
        assert result["token_a_idx"][0] == ti["BTC"]
        assert result["token_b_idx"][0] == ti["ETH"]

        # Pool 1 (second sorted prefix): tokens = "AAVE,ETH"
        assert result["token_a_idx"][1] == ti["AAVE"]
        assert result["token_b_idx"][1] == ti["ETH"]

    def test_chain_index(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import encode_tokens

        matched = self._get_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        result = encode_tokens(matched)
        assert result["n_chains"] == 2
        assert set(result["chain_index"].keys()) == {"ARBITRUM", "MAINNET"}
        assert set(result["chain_index"].values()) == {0, 1}

    def test_chain_idx_mapping(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import encode_tokens

        matched = self._get_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        result = encode_tokens(matched)
        pool_ids = sorted(matched.keys())
        ci = result["chain_index"]
        # Pool 0 is MAINNET, pool 1 is ARBITRUM
        assert result["chain_idx"][0] == ci[matched[pool_ids[0]]["chain"]]
        assert result["chain_idx"][1] == ci[matched[pool_ids[1]]["chain"]]

    def test_log_fees(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import encode_tokens

        matched = self._get_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        result = encode_tokens(matched)
        pool_ids = sorted(matched.keys())
        for i, pid in enumerate(pool_ids):
            expected_fee = matched[pid]["fee"]
            np.testing.assert_allclose(
                result["log_fees"][i], np.log(expected_fee), rtol=1e-6
            )


class TestMatchGridsToPanel:
    """Test match_grids_to_panel: match grid parquets to panel rows."""

    def test_match_returns_dict_per_pool(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import match_grids_to_panel

        # Write grid for pool 0
        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        synthetic_daily_grid.to_parquet(
            grid_dir / f"{POOL_PREFIXES[0]}_daily.parquet", index=False
        )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        assert isinstance(matched, dict)
        assert POOL_PREFIXES[0] in matched

    def test_match_filters_to_grid_pools_only(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        # Only write grid for pool 0 — pool 1 should be excluded
        synthetic_daily_grid.to_parquet(
            grid_dir / f"{POOL_PREFIXES[0]}_daily.parquet", index=False
        )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        assert POOL_PREFIXES[0] in matched
        assert POOL_PREFIXES[1] not in matched

    def test_match_includes_panel_obs(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        synthetic_daily_grid.to_parquet(
            grid_dir / f"{POOL_PREFIXES[0]}_daily.parquet", index=False
        )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        entry = matched[POOL_PREFIXES[0]]
        assert "panel" in entry
        assert isinstance(entry["panel"], pd.DataFrame)
        assert len(entry["panel"]) > 0

    def test_match_includes_coeffs(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.grid_interpolation import PoolCoeffsDaily
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        synthetic_daily_grid.to_parquet(
            grid_dir / f"{POOL_PREFIXES[0]}_daily.parquet", index=False
        )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        assert "coeffs" in matched[POOL_PREFIXES[0]]
        assert isinstance(matched[POOL_PREFIXES[0]]["coeffs"], PoolCoeffsDaily)

    def test_pool_id_prefix_matching(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        synthetic_daily_grid.to_parquet(
            grid_dir / f"{POOL_PREFIXES[0]}_daily.parquet", index=False
        )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        entry = matched[POOL_PREFIXES[0]]
        assert entry["pool_id"] == POOL_IDS_FULL[0]

    def test_match_includes_day_indices(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        synthetic_daily_grid.to_parquet(
            grid_dir / f"{POOL_PREFIXES[0]}_daily.parquet", index=False
        )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        entry = matched[POOL_PREFIXES[0]]
        assert "day_indices" in entry
        assert len(entry["day_indices"]) == len(entry["panel"])

    def test_day_indices_align_dates(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        synthetic_daily_grid.to_parquet(
            grid_dir / f"{POOL_PREFIXES[0]}_daily.parquet", index=False
        )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        entry = matched[POOL_PREFIXES[0]]
        coeffs = entry["coeffs"]
        day_indices = entry["day_indices"]

        # Panel dates should map to grid dates via ordinals
        panel_dates = pd.to_datetime(entry["panel"]["date"])
        panel_ordinals = np.array([d.toordinal() for d in panel_dates])
        grid_ordinals = np.array(coeffs.dates)

        for i, panel_ord in enumerate(panel_ordinals):
            grid_idx = day_indices[i]
            assert grid_ordinals[grid_idx] == panel_ord


class TestBuildXObs:
    """Test build_x_obs: observation covariate matrix."""

    def test_x_obs_shape(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0)
        assert x.shape == (len(pool0), K_OBS)

    def test_x_obs_intercept_column(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0)
        np.testing.assert_array_equal(x[:, 0], 1.0)

    def test_x_obs_lagged_tvl(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0)
        np.testing.assert_allclose(x[:, 1], pool0["log_tvl_lag1"].values)

    def test_x_obs_log_sigma(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0)
        expected = np.log(np.maximum(pool0["volatility"].values, 1e-6))
        np.testing.assert_allclose(x[:, 2], expected)

    def test_x_obs_interactions(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0)
        tvl = pool0["log_tvl_lag1"].values
        sigma = np.log(np.maximum(pool0["volatility"].values, 1e-6))
        fee = pool0["log_fee"].values
        np.testing.assert_allclose(x[:, 3], tvl * sigma)
        np.testing.assert_allclose(x[:, 4], tvl * fee)
        np.testing.assert_allclose(x[:, 5], sigma * fee)

    def test_x_obs_dow_harmonics(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0)
        weekdays = pd.to_datetime(pool0["date"]).dt.weekday.values
        expected_sin = np.sin(2 * np.pi * weekdays / 7)
        expected_cos = np.cos(2 * np.pi * weekdays / 7)
        np.testing.assert_allclose(x[:, 6], expected_sin, atol=1e-10)
        np.testing.assert_allclose(x[:, 7], expected_cos, atol=1e-10)

    def test_x_obs_no_nans(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0)
        assert not np.any(np.isnan(x))


class TestBuildXObsReduced:
    """Test build_x_obs with reduced=True: 4-column pruned covariates."""

    def test_reduced_shape(self, synthetic_panel):
        from quantammsim.calibration.pool_data import K_OBS_REDUCED, build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0, reduced=True)
        assert x.shape == (len(pool0), K_OBS_REDUCED)
        assert K_OBS_REDUCED == 4

    def test_reduced_columns(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x_full = build_x_obs(pool0)
        x_red = build_x_obs(pool0, reduced=True)

        # col 0: intercept
        np.testing.assert_array_equal(x_red[:, 0], 1.0)
        # col 1: log_tvl_lag1 (same as full col 1)
        np.testing.assert_allclose(x_red[:, 1], x_full[:, 1])
        # col 2: dow_sin (same as full col 6)
        np.testing.assert_allclose(x_red[:, 2], x_full[:, 6])
        # col 3: dow_cos (same as full col 7)
        np.testing.assert_allclose(x_red[:, 3], x_full[:, 7])

    def test_reduced_no_sigma(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x_full = build_x_obs(pool0)
        x_red = build_x_obs(pool0, reduced=True)

        # Sigma-dependent columns from full (2,3,5) should not appear
        sigma_cols = x_full[:, [2, 3, 5]]
        for col in range(x_red.shape[1]):
            for scol in range(sigma_cols.shape[1]):
                if not np.allclose(sigma_cols[:, scol], 0.0):
                    assert not np.allclose(x_red[:, col], sigma_cols[:, scol])

    def test_default_unchanged(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0)
        assert x.shape == (len(pool0), K_OBS)

    def test_reduced_no_nans(self, synthetic_panel):
        from quantammsim.calibration.pool_data import build_x_obs

        pool0 = synthetic_panel[synthetic_panel["pool_id"] == POOL_IDS_FULL[0]]
        x = build_x_obs(pool0, reduced=True)
        assert not np.any(np.isnan(x))


class TestBuildPoolAttributes:
    """Test build_pool_attributes: pool-level feature matrix."""

    def test_attributes_shape(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import (
            build_pool_attributes,
            match_grids_to_panel,
        )

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)
        assert X_attr.shape[0] == len(matched)
        assert X_attr.shape[1] == len(attr_names)

    def test_attributes_has_chain(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import (
            build_pool_attributes,
            match_grids_to_panel,
        )

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)
        chain_cols = [n for n in attr_names if n.startswith("chain_")]
        assert len(chain_cols) > 0

    def test_attributes_has_log_fee(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import (
            build_pool_attributes,
            match_grids_to_panel,
        )

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)
        assert "log_fee" in attr_names

    def test_attributes_has_log_tvl(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import (
            build_pool_attributes,
            match_grids_to_panel,
        )

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)
        assert "mean_log_tvl" in attr_names

    def test_attributes_returns_pool_order(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import (
            build_pool_attributes,
            match_grids_to_panel,
        )

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )

        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)
        assert isinstance(pool_ids, list)
        assert len(pool_ids) == len(matched)
        assert set(pool_ids) == set(matched.keys())
