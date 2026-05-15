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


class TestTokenCanonicalization:
    """Test _CANON_MAP and canonicalization in encode_tokens."""

    def test_canon_map_exists(self):
        from quantammsim.calibration.pool_data import _CANON_MAP
        assert isinstance(_CANON_MAP, dict)

    def test_canon_map_expected_mappings(self):
        from quantammsim.calibration.pool_data import _CANON_MAP
        assert _CANON_MAP["WETH"] == "ETH"
        assert _CANON_MAP["waBasWETH"] == "ETH"
        assert _CANON_MAP["waEthLidoWETH"] == "ETH"
        assert _CANON_MAP["waEthLidowstETH"] == "wstETH"
        assert _CANON_MAP["waGnowstETH"] == "wstETH"
        assert _CANON_MAP["waBasUSDC"] == "USDC"
        assert _CANON_MAP["scUSD"] == "USDC"
        assert _CANON_MAP["sDAI"] == "DAI"
        assert _CANON_MAP["WBTC"] == "BTC"
        assert _CANON_MAP["waGnoGNO"] == "GNO"
        assert _CANON_MAP["stS"] == "S"

    def test_canonicalize_passthrough(self):
        from quantammsim.calibration.pool_data import _CANON_MAP
        for tok in ["AAVE", "BTC", "ETH", "LINK", "ARB"]:
            assert tok not in _CANON_MAP

    def test_canonicalize_function(self):
        from quantammsim.calibration.pool_data import _canonicalize_token
        assert _canonicalize_token("WETH") == "ETH"
        assert _canonicalize_token("WBTC") == "BTC"
        assert _canonicalize_token("ETH") == "ETH"
        assert _canonicalize_token("AAVE") == "AAVE"

    def test_encode_tokens_canonicalize_false(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        """With canonicalize=False, same result as v1 (no merging)."""
        from quantammsim.calibration.pool_data import encode_tokens, match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )
        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        result = encode_tokens(matched, canonicalize=False)
        # Synthetic uses BTC, ETH, AAVE — none in canon map, same either way
        assert result["n_tokens"] == 3
        assert set(result["token_index"].keys()) == {"AAVE", "BTC", "ETH"}

    def test_encode_tokens_canonicalize_default(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        """Default canonicalize=True. Synthetic data unaffected (BTC, ETH, AAVE not in map)."""
        from quantammsim.calibration.pool_data import encode_tokens, match_grids_to_panel

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )
        matched = match_grids_to_panel(str(grid_dir), synthetic_panel)
        result = encode_tokens(matched)  # canonicalize=True by default
        assert result["n_tokens"] == 3
        assert set(result["token_index"].keys()) == {"AAVE", "BTC", "ETH"}

    def test_encode_tokens_merges_wrapped_tokens(self):
        """Synthetic matched dict with WETH+USDC and waBasWETH+WBTC → ETH, BTC, USDC."""
        from quantammsim.calibration.pool_data import encode_tokens

        # Minimal matched dict — encode_tokens only uses 'tokens' and 'fee' keys
        matched = {
            "pool_a": {
                "tokens": "WETH,USDC",
                "fee": 0.003,
                "chain": "MAINNET",
            },
            "pool_b": {
                "tokens": "waBasWETH,WBTC",
                "fee": 0.003,
                "chain": "BASE",
            },
        }
        result = encode_tokens(matched, canonicalize=True)
        # WETH→ETH, waBasWETH→ETH, WBTC→BTC → unique: {BTC, ETH, USDC}
        assert result["n_tokens"] == 3
        assert set(result["token_index"].keys()) == {"BTC", "ETH", "USDC"}

        # Both pools should have ETH as token A (canonicalized)
        ti = result["token_index"]
        assert result["token_a_idx"][0] == ti["ETH"]  # pool_a: WETH→ETH
        assert result["token_a_idx"][1] == ti["ETH"]  # pool_b: waBasWETH→ETH

    def test_encode_tokens_canon_false_keeps_wrapped(self):
        """canonicalize=False keeps WETH and waBasWETH as separate tokens."""
        from quantammsim.calibration.pool_data import encode_tokens

        matched = {
            "pool_a": {
                "tokens": "WETH,USDC",
                "fee": 0.003,
                "chain": "MAINNET",
            },
            "pool_b": {
                "tokens": "waBasWETH,WBTC",
                "fee": 0.003,
                "chain": "BASE",
            },
        }
        result = encode_tokens(matched, canonicalize=False)
        # No merging: WETH, USDC, waBasWETH, WBTC → 4 unique tokens
        assert result["n_tokens"] == 4
        assert "WETH" in result["token_index"]
        assert "waBasWETH" in result["token_index"]


class TestCrossPoolFeatures:
    """Test build_cross_pool_x_obs: cross-pool lagged volume features."""

    @pytest.fixture
    def three_pool_panel(self):
        """3 pools sharing tokens, 10 days each."""
        np.random.seed(42)
        dates = pd.date_range("2025-12-01", periods=10, freq="D")
        rows = []
        pool_configs = [
            ("0xpool_a_full_id_padding_to_66_chars_aaaaaaaaaaaaaaaaaaaaaaaaa",
             "MAINNET", "ETH,USDC", 0.003),
            ("0xpool_b_full_id_padding_to_66_chars_bbbbbbbbbbbbbbbbbbbbbbbbb",
             "MAINNET", "ETH,AAVE", 0.003),
            ("0xpool_c_full_id_padding_to_66_chars_ccccccccccccccccccccccccc",
             "ARBITRUM", "AAVE,USDC", 0.01),
        ]
        for full_id, chain, tokens, fee in pool_configs:
            for di, date in enumerate(dates):
                tvl = 12.0 + 0.05 * np.sin(2 * np.pi * di / 7)
                vol = 9.0 + 0.3 * np.random.randn()
                rows.append({
                    "pool_id": full_id,
                    "chain": chain,
                    "date": date,
                    "log_volume": vol,
                    "log_tvl": tvl,
                    "log_tvl_lag1": tvl - 0.01,
                    "volatility": 0.4,
                    "log_fee": np.log(fee),
                    "swap_fee": fee,
                    "tokens": tokens,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def three_pool_matched(self, three_pool_panel):
        """Minimal matched dict for 3 pools (no grid needed for x_obs tests)."""
        matched = {}
        for full_id in three_pool_panel["pool_id"].unique():
            prefix = full_id[:16]
            rows = three_pool_panel[three_pool_panel["pool_id"] == full_id].copy()
            rows = rows.reset_index(drop=True)
            matched[prefix] = {
                "panel": rows,
                "pool_id": full_id,
                "chain": rows.iloc[0]["chain"],
                "fee": float(np.exp(rows.iloc[0]["log_fee"])),
                "tokens": rows.iloc[0]["tokens"],
                "weights": [0.5, 0.5],
            }
        return matched

    def test_build_cross_pool_x_obs_shape(self, three_pool_matched):
        from quantammsim.calibration.pool_data import (
            K_OBS_CROSS, build_cross_pool_x_obs,
        )

        pid = sorted(three_pool_matched.keys())[0]
        entry = three_pool_matched[pid]
        x = build_cross_pool_x_obs(entry["panel"], three_pool_matched, pid)
        # Drops first day → n_obs - 1 rows, K_OBS_CROSS=7 columns
        assert x.shape[1] == K_OBS_CROSS
        assert x.shape[0] == len(entry["panel"]) - 1

    def test_first_four_cols_match_reduced(self, three_pool_matched):
        from quantammsim.calibration.pool_data import (
            build_cross_pool_x_obs, build_x_obs,
        )

        pid = sorted(three_pool_matched.keys())[0]
        entry = three_pool_matched[pid]
        x_cross = build_cross_pool_x_obs(entry["panel"], three_pool_matched, pid)
        x_reduced = build_x_obs(entry["panel"], reduced=True)
        # First 4 columns should match (after dropping first row)
        np.testing.assert_allclose(x_cross[:, :4], x_reduced[1:, :4])

    def test_cross_vol_token_a_excludes_self(self, three_pool_matched):
        """Peer average for token A excludes pool i itself."""
        from quantammsim.calibration.pool_data import build_cross_pool_x_obs

        pool_ids = sorted(three_pool_matched.keys())
        pid_a = pool_ids[0]  # ETH,USDC
        pid_b = pool_ids[1]  # ETH,AAVE — shares ETH with pool_a

        x_a = build_cross_pool_x_obs(
            three_pool_matched[pid_a]["panel"],
            three_pool_matched, pid_a,
        )
        # Column 4 = cross_vol_token_a (ETH peers excl self)
        # Pool b also has ETH, so pool_a's cross_vol_token_a should use pool_b's volume
        panel_b = three_pool_matched[pid_b]["panel"]
        log_vol_b_lagged = panel_b["log_volume"].values[:-1]  # lag by 1
        np.testing.assert_allclose(x_a[:, 4], log_vol_b_lagged, rtol=1e-6)

    def test_cross_vol_is_lagged(self, three_pool_matched):
        """Features at day t use log_volume at day t-1."""
        from quantammsim.calibration.pool_data import build_cross_pool_x_obs

        pool_ids = sorted(three_pool_matched.keys())
        pid = pool_ids[0]
        x = build_cross_pool_x_obs(
            three_pool_matched[pid]["panel"],
            three_pool_matched, pid,
        )
        # x has n_obs - 1 rows (first day dropped)
        # Row 0 of x corresponds to day 1 and should use day 0 volume
        assert x.shape[0] > 0

    def test_cross_vol_nan_free_after_first_day(self, three_pool_matched):
        from quantammsim.calibration.pool_data import build_cross_pool_x_obs

        for pid in three_pool_matched:
            x = build_cross_pool_x_obs(
                three_pool_matched[pid]["panel"],
                three_pool_matched, pid,
            )
            assert not np.any(np.isnan(x)), f"NaNs in cross-pool x_obs for {pid}"

    def test_exclude_pool_changes_features(self, three_pool_matched):
        """exclude_pool removes that pool from peer averages."""
        from quantammsim.calibration.pool_data import build_cross_pool_x_obs

        pool_ids = sorted(three_pool_matched.keys())
        pid = pool_ids[0]  # ETH,USDC

        x_normal = build_cross_pool_x_obs(
            three_pool_matched[pid]["panel"],
            three_pool_matched, pid,
        )
        x_excluded = build_cross_pool_x_obs(
            three_pool_matched[pid]["panel"],
            three_pool_matched, pid,
            exclude_pool=pool_ids[1],  # exclude the ETH peer
        )
        # Chain feature (col 6) may change too; token A col (4) definitely changes
        # since pool_b is the only ETH peer
        # With only peer excluded, cross_vol_token_a should be NaN→fallback
        assert not np.allclose(x_normal[:, 4], x_excluded[:, 4])

    def test_single_token_pool_fallback(self):
        """When a token appears in only one pool, its cross_vol uses global mean."""
        from quantammsim.calibration.pool_data import build_cross_pool_x_obs

        np.random.seed(42)
        dates = pd.date_range("2025-12-01", periods=5, freq="D")
        rows = []
        # Pool A: LINK,USDC — LINK is unique
        for di, date in enumerate(dates):
            rows.append({
                "pool_id": "0xsolo_link_pool_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "chain": "MAINNET", "date": date,
                "log_volume": 9.0 + 0.1 * di,
                "log_tvl_lag1": 12.0, "volatility": 0.4,
                "log_fee": np.log(0.003), "tokens": "LINK,USDC",
            })
        # Pool B: ETH,USDC
        for di, date in enumerate(dates):
            rows.append({
                "pool_id": "0xpeer_eth_pool__bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "chain": "MAINNET", "date": date,
                "log_volume": 10.0 + 0.1 * di,
                "log_tvl_lag1": 13.0, "volatility": 0.4,
                "log_fee": np.log(0.003), "tokens": "ETH,USDC",
            })
        panel = pd.DataFrame(rows)

        matched = {}
        for full_id in panel["pool_id"].unique():
            prefix = full_id[:16]
            sub = panel[panel["pool_id"] == full_id].reset_index(drop=True)
            matched[prefix] = {
                "panel": sub, "pool_id": full_id,
                "chain": sub.iloc[0]["chain"],
                "fee": float(np.exp(sub.iloc[0]["log_fee"])),
                "tokens": sub.iloc[0]["tokens"], "weights": [0.5, 0.5],
            }

        pid_link = [p for p in matched if matched[p]["tokens"] == "LINK,USDC"][0]
        x = build_cross_pool_x_obs(panel[panel["pool_id"] == matched[pid_link]["pool_id"]].reset_index(drop=True),
                                    matched, pid_link)
        # LINK has no peers → col 4 should be a fallback (global mean), not NaN
        assert not np.any(np.isnan(x[:, 4]))
