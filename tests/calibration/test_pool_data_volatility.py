"""Tests for Binance volatility and TOKEN_MAP in quantammsim.calibration.pool_data."""

import numpy as np
import pandas as pd
import pytest

from tests.calibration.conftest import POOL_IDS_FULL


class TestTokenMap:
    """Test TOKEN_MAP resolves Balancer tokens to Binance symbols correctly."""

    def test_wrapped_native(self):
        from quantammsim.calibration.pool_data import _resolve_binance_symbol

        assert _resolve_binance_symbol("WBTC") == "BTC"
        assert _resolve_binance_symbol("WETH") == "ETH"
        assert _resolve_binance_symbol("cbBTC") == "BTC"

    def test_lst_to_underlying(self):
        from quantammsim.calibration.pool_data import _resolve_binance_symbol

        assert _resolve_binance_symbol("wstETH") == "ETH"
        assert _resolve_binance_symbol("stETH") == "ETH"
        assert _resolve_binance_symbol("rETH") == "ETH"
        assert _resolve_binance_symbol("cbETH") == "ETH"

    def test_vault_tokens(self):
        from quantammsim.calibration.pool_data import _resolve_binance_symbol

        assert _resolve_binance_symbol("waEthLidoWETH") == "ETH"
        assert _resolve_binance_symbol("waEthLidowstETH") == "ETH"
        assert _resolve_binance_symbol("waBasWETH") == "ETH"
        assert _resolve_binance_symbol("waGnowstETH") == "ETH"
        assert _resolve_binance_symbol("waGnoGNO") == "GNO"

    def test_stablecoins_map_to_usdc(self):
        from quantammsim.calibration.pool_data import _resolve_binance_symbol

        for stable in ["DAI", "WXDAI", "sDAI", "USDT", "DOLA", "scUSD",
                        "USDC.e", "USDbC", "waBasUSDC"]:
            assert _resolve_binance_symbol(stable) == "USDC", (
                f"{stable} should map to USDC"
            )

    def test_matic_variants(self):
        from quantammsim.calibration.pool_data import _resolve_binance_symbol

        assert _resolve_binance_symbol("wPOL") == "POL"
        assert _resolve_binance_symbol("WMATIC") == "POL"
        assert _resolve_binance_symbol("MATIC") == "POL"

    def test_sonic_variants(self):
        from quantammsim.calibration.pool_data import _resolve_binance_symbol

        assert _resolve_binance_symbol("wS") == "S"
        assert _resolve_binance_symbol("stS") == "S"

    def test_passthrough_unknown(self):
        from quantammsim.calibration.pool_data import _resolve_binance_symbol

        assert _resolve_binance_symbol("AAVE") == "AAVE"
        assert _resolve_binance_symbol("LINK") == "LINK"
        assert _resolve_binance_symbol("SNX") == "SNX"

    def test_jitosol(self):
        from quantammsim.calibration.pool_data import _resolve_binance_symbol

        assert _resolve_binance_symbol("JitoSOL") == "SOL"


class TestGetAssetType:
    """Test _get_asset_type classification."""

    def test_stablecoins(self):
        from quantammsim.calibration.pool_data import _get_asset_type

        for tok in ["USDC", "USDT", "DAI", "WXDAI", "sDAI", "DOLA", "scUSD"]:
            assert _get_asset_type(tok, {}) == 0, f"{tok} should be stable (0)"

    def test_native_lst(self):
        from quantammsim.calibration.pool_data import _get_asset_type

        for tok in ["WETH", "ETH", "wstETH", "WBTC", "BTC", "GNO", "S", "wS"]:
            assert _get_asset_type(tok, {}) == 1, f"{tok} should be native/LST (1)"

    def test_volatile(self):
        from quantammsim.calibration.pool_data import _get_asset_type

        for tok in ["AAVE", "LINK", "SNX", "CRV", "COMP"]:
            assert _get_asset_type(tok, {}) == 2, f"{tok} should be volatile (2)"

    def test_mcap_override(self):
        from quantammsim.calibration.pool_data import _get_asset_type

        mcaps = {"AAVE": {"asset_type": "stable", "mcap_usd": 1e9}}
        assert _get_asset_type("AAVE", mcaps) == 0  # overridden to stable


class TestComputeBinancePairVolatility:
    """Test compute_binance_pair_volatility with synthetic Binance-like data."""

    @pytest.fixture
    def fake_binance_dir(self, tmp_path):
        """Create fake Binance minute parquets for ETH and AAVE."""
        np.random.seed(42)
        n_minutes = 24 * 60 * 7  # 7 days of minute data
        base_ts = int(pd.Timestamp("2025-01-01").timestamp() * 1000)
        unix = base_ts + np.arange(n_minutes) * 60_000

        # ETH: geometric brownian motion starting at 3000
        eth_log_returns = np.random.normal(0, 0.0005, n_minutes)
        eth_prices = 3000.0 * np.exp(np.cumsum(eth_log_returns))
        eth_df = pd.DataFrame({"unix": unix, "close": eth_prices})
        eth_df.to_parquet(tmp_path / "ETH_USD.parquet", index=False)

        # AAVE: correlated with ETH but with higher vol
        aave_log_returns = 0.6 * eth_log_returns + 0.4 * np.random.normal(
            0, 0.001, n_minutes
        )
        aave_prices = 200.0 * np.exp(np.cumsum(aave_log_returns))
        aave_df = pd.DataFrame({"unix": unix, "close": aave_prices})
        aave_df.to_parquet(tmp_path / "AAVE_USD.parquet", index=False)

        # USDC: constant at $1 (stablecoin proxy)
        usdc_df = pd.DataFrame({
            "unix": unix, "close": np.ones(n_minutes),
        })
        usdc_df.to_parquet(tmp_path / "USDC_USD.parquet", index=False)

        return str(tmp_path)

    def test_pinned_volatility_values(self, fake_binance_dir):
        """Exact pinned values with seed(42)."""
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        expected = np.array([
            0.27103676, 0.23068148, 0.43763073, 0.35174542,
            0.26827274, 0.35256874, 0.27833725,
        ])
        np.testing.assert_allclose(vol.values, expected, rtol=1e-4)

    def test_exactly_seven_days(self, fake_binance_dir):
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        assert len(vol) == 7

    def test_values_positive(self, fake_binance_dir):
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        assert (vol > 0).all()

    def test_pinned_median(self, fake_binance_dir):
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        np.testing.assert_allclose(vol.median(), 0.2783, atol=0.001)

    def test_token_order_invariance(self, fake_binance_dir):
        """vol(A,B) should equal vol(B,A) — log returns of reciprocal have same std."""
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol_ab = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        vol_ba = compute_binance_pair_volatility("AAVE", "WETH", fake_binance_dir)
        np.testing.assert_allclose(vol_ab.values, vol_ba.values, rtol=1e-5)

    def test_stable_vs_volatile_uses_single_asset(self, fake_binance_dir):
        """ETH/USDC should use just ETH price — verify against hand-computed ETH vol."""
        import os

        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol_pair = compute_binance_pair_volatility("WETH", "USDC", fake_binance_dir)

        # Hand-compute ETH-only vol for ground-truth comparison
        eth = pd.read_parquet(os.path.join(fake_binance_dir, "ETH_USD.parquet"))
        eth_ts = pd.DataFrame(
            {"ratio": eth["close"].values},
            index=pd.to_datetime(eth["unix"].values, unit="ms", utc=True),
        )
        hourly = eth_ts.resample("1h").last().dropna()
        hourly["log_return"] = np.log(hourly["ratio"] / hourly["ratio"].shift(1))
        hourly = hourly.dropna()
        hourly["date"] = hourly.index.date
        daily_std = hourly.groupby("date")["log_return"].std()
        expected = (daily_std * np.sqrt(24 * 365)).dropna()
        expected = expected[expected > 0]

        np.testing.assert_allclose(vol_pair.values, expected.values, rtol=1e-5)

    def test_stable_stable_returns_none(self, fake_binance_dir):
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("USDC", "DAI", fake_binance_dir)
        assert vol is None

    def test_same_underlying_returns_none(self, fake_binance_dir):
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        # WETH and wstETH both map to ETH
        vol = compute_binance_pair_volatility("WETH", "wstETH", fake_binance_dir)
        assert vol is None

    def test_missing_data_returns_none(self, fake_binance_dir):
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "MAGIC", fake_binance_dir)
        assert vol is None

    def test_daily_index_type(self, fake_binance_dir):
        """Index should be datetime.date objects."""
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility
        import datetime

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        for d in vol.index:
            assert isinstance(d, datetime.date)

    def test_stable_a_volatile_b_uses_reciprocal(self, fake_binance_dir):
        """DAI/WETH should use 1/ETH, giving same vol as WETH/DAI."""
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol_forward = compute_binance_pair_volatility("WETH", "USDC", fake_binance_dir)
        vol_reverse = compute_binance_pair_volatility("DAI", "WETH", fake_binance_dir)
        # Both should be ETH vol (log returns of X and 1/X have same std)
        np.testing.assert_allclose(
            vol_forward.values, vol_reverse.values, rtol=1e-5,
        )


class TestReplacePanelVolatility:
    """Test replace_panel_volatility_with_binance."""

    @pytest.fixture
    def fake_binance_dir(self, tmp_path):
        """Minimal fake Binance data for 3 days."""
        np.random.seed(42)
        n_minutes = 24 * 60 * 3
        base_ts = int(pd.Timestamp("2025-12-01").timestamp() * 1000)
        unix = base_ts + np.arange(n_minutes) * 60_000

        eth_prices = 3000.0 + np.cumsum(np.random.normal(0, 1.0, n_minutes))
        eth_df = pd.DataFrame({"unix": unix, "close": eth_prices})
        eth_df.to_parquet(tmp_path / "ETH_USD.parquet", index=False)

        btc_prices = 60000.0 + np.cumsum(np.random.normal(0, 5.0, n_minutes))
        btc_df = pd.DataFrame({"unix": unix, "close": btc_prices})
        btc_df.to_parquet(tmp_path / "BTC_USD.parquet", index=False)

        aave_prices = 200.0 + np.cumsum(np.random.normal(0, 0.5, n_minutes))
        aave_df = pd.DataFrame({"unix": unix, "close": aave_prices})
        aave_df.to_parquet(tmp_path / "AAVE_USD.parquet", index=False)

        return str(tmp_path)

    def test_does_not_modify_input(self, synthetic_panel, fake_binance_dir):
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        original_vol = synthetic_panel["volatility"].copy()
        replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        pd.testing.assert_series_equal(synthetic_panel["volatility"], original_vol)

    def test_no_nans_introduced(self, synthetic_panel, fake_binance_dir):
        """Pools without Binance data should keep original volatility, not NaN."""
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        assert result["volatility"].notna().all()

    def test_volatility_actually_changes(self, synthetic_panel, fake_binance_dir):
        """At least some volatility values should differ after replacement."""
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        original_vol = synthetic_panel["volatility"].values.copy()
        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        # At least one pool has BTC,ETH or AAVE,ETH — both have Binance data,
        # and dates overlap (panel starts 2025-12-01, fake data starts 2025-12-01).
        n_changed = (result["volatility"].values != original_vol).sum()
        assert n_changed > 0, "No volatility values were replaced"

    def test_replaced_values_are_positive(self, synthetic_panel, fake_binance_dir):
        """Replaced volatility values must be positive."""
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        assert (result["volatility"] > 0).all()

    def test_all_columns_preserved(self, synthetic_panel, fake_binance_dir):
        """Output should have all original columns."""
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        for col in synthetic_panel.columns:
            assert col in result.columns

    def test_row_count_preserved(self, synthetic_panel, fake_binance_dir):
        """Output should have the same number of rows."""
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        assert len(result) == len(synthetic_panel)

    def test_replaced_values_match_binance_computation(
        self, synthetic_panel, fake_binance_dir
    ):
        """Replaced vol values must equal compute_binance_pair_volatility exactly."""
        from quantammsim.calibration.pool_data import (
            compute_binance_pair_volatility,
            replace_panel_volatility_with_binance,
        )

        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )

        # BTC/ETH pool — compute expected vol independently
        vol_btc_eth = compute_binance_pair_volatility("BTC", "ETH", fake_binance_dir)
        assert vol_btc_eth is not None, "BTC/ETH vol should be computable"
        vol_dict = vol_btc_eth.to_dict()

        pool0 = result[result["tokens"] == "BTC,ETH"].copy()
        pool0_dates = pd.to_datetime(pool0["date"]).dt.date
        matched_mask = pool0_dates.isin(vol_dict.keys()).values
        matched = pool0[matched_mask]
        assert len(matched) > 0, "No date overlap between panel and Binance data"

        for _, row in matched.iterrows():
            d = pd.to_datetime(row["date"]).date()
            np.testing.assert_allclose(
                row["volatility"], vol_dict[d], rtol=1e-6,
                err_msg=f"BTC/ETH vol mismatch on {d}",
            )


class TestBuildPoolAttributeValues:
    """Test build_pool_attributes returns correct numerical values, not just names."""

    def _make_matched(self, synthetic_daily_grid, synthetic_panel, tmp_path):
        from quantammsim.calibration.pool_data import match_grids_to_panel
        from tests.calibration.conftest import POOL_PREFIXES

        grid_dir = tmp_path / "grids"
        grid_dir.mkdir()
        for prefix in POOL_PREFIXES:
            synthetic_daily_grid.to_parquet(
                grid_dir / f"{prefix}_daily.parquet", index=False
            )
        return match_grids_to_panel(str(grid_dir), synthetic_panel)

    def test_chain_dummy_values(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        """MAINNET pool has chain_MAINNET=1, ARBITRUM pool has chain_MAINNET=0."""
        from quantammsim.calibration.pool_data import build_pool_attributes

        matched = self._make_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)

        # ARBITRUM is reference (alphabetically first), MAINNET gets a dummy
        chain_idx = attr_names.index("chain_MAINNET")
        for i, pid in enumerate(pool_ids):
            if matched[pid]["chain"] == "MAINNET":
                assert X_attr[i, chain_idx] == 1.0
            else:
                assert X_attr[i, chain_idx] == 0.0

    def test_log_fee_values(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        """log_fee should match panel values."""
        from quantammsim.calibration.pool_data import build_pool_attributes

        matched = self._make_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)

        fee_idx = attr_names.index("log_fee")
        for i, pid in enumerate(pool_ids):
            expected = np.log(matched[pid]["fee"])
            np.testing.assert_allclose(X_attr[i, fee_idx], expected, rtol=1e-3)

    def test_same_asset_type_for_btc_eth(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        """BTC,ETH pool: both native/LST → same_asset_type=1."""
        from quantammsim.calibration.pool_data import build_pool_attributes

        matched = self._make_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)

        sat_idx = attr_names.index("same_asset_type")
        for i, pid in enumerate(pool_ids):
            if matched[pid]["tokens"] == "BTC,ETH":
                assert X_attr[i, sat_idx] == 1.0
            elif matched[pid]["tokens"] == "AAVE,ETH":
                # AAVE=volatile(2), ETH=native(1) → different
                assert X_attr[i, sat_idx] == 0.0

    def test_pinned_attribute_values(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        """Pinned X_attr values for the two synthetic pools."""
        from quantammsim.calibration.pool_data import build_pool_attributes

        matched = self._make_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        X_attr, attr_names, pool_ids = build_pool_attributes(matched)

        # Pool 0 (0xaaaa = MAINNET, BTC/ETH, fee=0.003)
        p0_idx = pool_ids.index("0xaaaa11112222aa")
        np.testing.assert_allclose(X_attr[p0_idx, 0], 1.0)  # chain_MAINNET
        np.testing.assert_allclose(
            X_attr[p0_idx, attr_names.index("log_fee")], np.log(0.003), rtol=1e-3
        )

        # Pool 1 (0xbbbb = ARBITRUM, AAVE/ETH, fee=0.01)
        p1_idx = pool_ids.index("0xbbbb33334444bb")
        np.testing.assert_allclose(X_attr[p1_idx, 0], 0.0)  # chain_MAINNET=0
        np.testing.assert_allclose(
            X_attr[p1_idx, attr_names.index("log_fee")], np.log(0.01), rtol=1e-3
        )

    def test_no_nans(
        self, synthetic_daily_grid, synthetic_panel, tmp_path
    ):
        from quantammsim.calibration.pool_data import build_pool_attributes

        matched = self._make_matched(
            synthetic_daily_grid, synthetic_panel, tmp_path
        )
        X_attr, _, _ = build_pool_attributes(matched)
        assert not np.any(np.isnan(X_attr))
