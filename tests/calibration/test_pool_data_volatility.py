"""Tests for Binance volatility and TOKEN_MAP in quantammsim.calibration.pool_data."""

import numpy as np
import pandas as pd
import pytest
import os

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

    def test_returns_series(self, fake_binance_dir):
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        assert isinstance(vol, pd.Series)
        assert len(vol) > 0

    def test_values_positive(self, fake_binance_dir):
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        assert (vol > 0).all()

    def test_annualized_magnitude(self, fake_binance_dir):
        """Annualized vol should be in [0.01, 10.0] range for typical assets."""
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        assert vol.median() > 0.01
        assert vol.median() < 10.0

    def test_stable_vs_volatile(self, fake_binance_dir):
        """ETH/USDC should just use ETH price (one-sided)."""
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "USDC", fake_binance_dir)
        assert isinstance(vol, pd.Series)
        assert len(vol) > 0

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

    def test_seven_days_of_data(self, fake_binance_dir):
        """7 days of minute data → ~6 days of vol (first day partial)."""
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("WETH", "AAVE", fake_binance_dir)
        assert 5 <= len(vol) <= 7

    def test_stable_a_volatile_b(self, fake_binance_dir):
        """When token_a is stable, should use 1/price_b."""
        from quantammsim.calibration.pool_data import compute_binance_pair_volatility

        vol = compute_binance_pair_volatility("DAI", "WETH", fake_binance_dir)
        assert isinstance(vol, pd.Series)
        assert len(vol) > 0


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

        return str(tmp_path)

    def test_returns_dataframe(self, synthetic_panel, fake_binance_dir):
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_input(self, synthetic_panel, fake_binance_dir):
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        original_vol = synthetic_panel["volatility"].copy()
        replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        pd.testing.assert_series_equal(synthetic_panel["volatility"], original_vol)

    def test_volatility_column_exists(self, synthetic_panel, fake_binance_dir):
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        assert "volatility" in result.columns

    def test_no_nans_introduced(self, synthetic_panel, fake_binance_dir):
        """Pools without Binance data should keep original volatility, not NaN."""
        from quantammsim.calibration.pool_data import replace_panel_volatility_with_binance

        result = replace_panel_volatility_with_binance(
            synthetic_panel, fake_binance_dir
        )
        assert result["volatility"].notna().all()
