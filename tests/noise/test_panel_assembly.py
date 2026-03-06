"""Tests for compute_pair_volatility, assemble_panel, validate_panel."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from quantammsim.noise_calibration import (
    compute_pair_volatility,
    assemble_panel,
    validate_panel,
)


# ===========================================================================
# TestComputePairVolatility
# ===========================================================================


class TestComputePairVolatility:
    @pytest.fixture()
    def _snap_dates(self):
        """10 unique dates for snapshot stub."""
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(10)]
        return pd.DataFrame({"date": dates})

    def test_stablecoin_pair_returns_001(self, _snap_dates):
        pool_row = pd.Series({
            "tokens": ["USDC", "DAI"],
            "chain": "MAINNET",
        })
        vol = compute_pair_volatility(_snap_dates, pool_row, {})
        assert (vol == 0.01).all()

    def test_both_missing_non_stable(self, _snap_dates):
        pool_row = pd.Series({
            "tokens": ["FOO", "BAR"],
            "chain": "MAINNET",
        })
        vol = compute_pair_volatility(_snap_dates, pool_row, {})
        assert (vol == 0.5).all()

    def test_one_missing_non_stable(self, _snap_dates):
        np.random.seed(77)
        pool_row = pd.Series({
            "tokens": ["WETH", "BAR"],
            "chain": "MAINNET",
        })
        prices = {
            ("MAINNET", "WETH"): pd.DataFrame({
                "timestamp": [1735689600 + i * 3600 for i in range(48)],
                "price": [3000.0 + np.random.randn() * 10 for _ in range(48)],
            }),
        }
        vol = compute_pair_volatility(_snap_dates, pool_row, prices)
        assert (vol == 0.5).all()

    def test_synthetic_hourly_prices_positive_finite(self, _snap_dates):
        np.random.seed(42)
        n_hours = 240  # 10 days x 24 hours
        ts_base = 1735689600
        timestamps = [ts_base + i * 3600 for i in range(n_hours)]
        prices_a = np.exp(np.cumsum(np.random.randn(n_hours) * 0.01) + 8)
        prices_b = np.exp(np.cumsum(np.random.randn(n_hours) * 0.01) + 7)

        pool_row = pd.Series({
            "tokens": ["WETH", "LINK"],
            "chain": "MAINNET",
        })
        token_prices = {
            ("MAINNET", "WETH"): pd.DataFrame({
                "timestamp": timestamps, "price": prices_a,
            }),
            ("MAINNET", "LINK"): pd.DataFrame({
                "timestamp": timestamps, "price": prices_b,
            }),
        }
        vol = compute_pair_volatility(_snap_dates, pool_row, token_prices)
        assert len(vol) > 0
        assert (vol > 0).all()
        assert np.all(np.isfinite(vol))

    def test_annualisation_uses_sqrt_24x365(self, _snap_dates):
        """Verify the annualisation factor is sqrt(24*365)."""
        np.random.seed(7)
        n_hours = 240
        ts_base = 1735689600
        timestamps = [ts_base + i * 3600 for i in range(n_hours)]
        raw_prices = np.exp(np.cumsum(np.random.randn(n_hours) * 0.005) + 8)

        pool_row = pd.Series({
            "tokens": ["WETH", "USDC"],
            "chain": "MAINNET",
        })
        token_prices = {
            ("MAINNET", "WETH"): pd.DataFrame({
                "timestamp": timestamps, "price": raw_prices,
            }),
        }
        vol = compute_pair_volatility(_snap_dates, pool_row, token_prices)

        # Reconstruct manually
        df = pd.DataFrame({"timestamp": timestamps, "price": raw_prices})
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["date"] = df["datetime"].dt.date
        df["ratio"] = df["price"]  # WETH vs stable => ratio = price
        df["log_return"] = np.log(df["ratio"] / df["ratio"].shift(1))
        df = df.dropna(subset=["log_return"])
        daily_std = df.groupby("date")["log_return"].std()
        expected = daily_std * np.sqrt(24 * 365)

        common = vol.index.intersection(expected.index)
        assert len(common) > 0
        np.testing.assert_allclose(
            vol.loc[common].values, expected.loc[common].values, rtol=1e-10,
        )

    def test_single_token_pool_returns_empty(self, _snap_dates):
        pool_row = pd.Series({"tokens": ["WETH"], "chain": "MAINNET"})
        vol = compute_pair_volatility(_snap_dates, pool_row, {})
        assert len(vol) == 0

    def test_no_overlapping_price_dates(self, _snap_dates):
        """Two tokens with non-overlapping timestamps -> fallback 0.5."""
        pool_row = pd.Series({
            "tokens": ["WETH", "LINK"],
            "chain": "MAINNET",
        })
        token_prices = {
            ("MAINNET", "WETH"): pd.DataFrame({
                "timestamp": [1000000 + i for i in range(10)],
                "price": [3000.0] * 10,
            }),
            ("MAINNET", "LINK"): pd.DataFrame({
                "timestamp": [9000000 + i for i in range(10)],
                "price": [15.0] * 10,
            }),
        }
        vol = compute_pair_volatility(_snap_dates, pool_row, token_prices)
        assert (vol == 0.5).all()

    def test_cross_chain_price_fallback(self, _snap_dates):
        """Prices keyed to a different chain SHOULD be used as fallback.

        Token prices are chain-agnostic (WETH is WETH regardless of chain),
        so an ARBITRUM pool should use MAINNET WETH prices if ARBITRUM
        prices aren't available.
        """
        np.random.seed(55)
        n_hours = 240
        ts_base = 1735689600
        timestamps = [ts_base + i * 3600 for i in range(n_hours)]
        prices = np.exp(np.cumsum(np.random.randn(n_hours) * 0.01) + 8)

        pool_row = pd.Series({
            "tokens": ["WETH", "USDC"],
            "chain": "ARBITRUM",
        })
        token_prices = {
            # Only MAINNET prices, but ARBITRUM pool should still use them
            ("MAINNET", "WETH"): pd.DataFrame({
                "timestamp": timestamps, "price": prices.tolist(),
            }),
        }
        vol = compute_pair_volatility(_snap_dates, pool_row, token_prices)
        # Should get real volatility, NOT the 0.5 fallback
        assert len(vol) > 0
        assert not (vol == 0.5).all(), (
            "Cross-chain price fallback should have produced real volatility"
        )


# ===========================================================================
# TestAssemblePanel
# ===========================================================================


class TestAssemblePanel:
    def test_lagged_tvl_drops_first_obs_per_pool(
        self, synthetic_pools_df, synthetic_snapshots_df
    ):
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        counts = panel.groupby("pool_id").size()
        assert (counts == 9).all()

    def test_lagged_tvl_exact_values(
        self, synthetic_pools_df, synthetic_snapshots_df
    ):
        """log_tvl_lag1[t] must equal log_tvl[t-1] for same pool (shift(1))."""
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        for pid in panel["pool_id"].unique():
            pool = panel[panel["pool_id"] == pid].sort_values("date")
            tvl_vals = pool["log_tvl"].values
            lag_vals = pool["log_tvl_lag1"].values
            # After dropping the first obs, lag[i] = tvl[i-1] in the original
            # pre-drop series. Since the panel is sorted by date, each
            # lag value should equal the log_tvl of the chronologically
            # preceding observation. We verify consecutive pairs: for rows
            # i and i+1, lag[i+1] == tvl[i].
            for i in range(len(tvl_vals) - 1):
                np.testing.assert_allclose(
                    lag_vals[i + 1], tvl_vals[i], rtol=1e-14,
                    err_msg=f"Pool {pid} row {i+1}: lag should equal previous tvl",
                )

    def test_weekend_flag(self, synthetic_pools_df, synthetic_snapshots_df):
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        for _, row in panel.iterrows():
            d = row["date"]
            if not isinstance(d, date):
                d = pd.Timestamp(d).date()
            expected = 1.0 if d.weekday() >= 5 else 0.0
            assert row["weekend"] == expected, f"Wrong weekend flag for {d}"

    def test_log_volume_is_natural_log(
        self, synthetic_pools_df, synthetic_snapshots_df
    ):
        """log_volume must equal ln(volume_usd), not log10 or log2."""
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        snaps = synthetic_snapshots_df.copy()
        # Join snapshots to panel by pool_id + date to verify exact log values
        for _, row in panel.iterrows():
            pid = row["pool_id"]
            d = row["date"]
            snap_match = snaps[
                (snaps["pool_id"] == pid) & (snaps["date"] == d)
            ]
            assert len(snap_match) == 1, f"No snapshot for {pid} on {d}"
            expected = np.log(snap_match.iloc[0]["volume_usd"])
            np.testing.assert_allclose(
                row["log_volume"], expected, rtol=1e-14,
                err_msg=f"log_volume for {pid} on {d} should be ln(volume_usd)",
            )

    def test_tier_assignment_min_first(
        self, synthetic_pools_df, synthetic_snapshots_df
    ):
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        # Pool A: WETH(0), USDC(0) -> tier_A=0, tier_B=0
        pool_a = panel[panel["pool_id"] == "pool_A"].iloc[0]
        assert pool_a["tier_A"] == 0
        assert pool_a["tier_B"] == 0

        # Pool B: BAL(1), WETH(0) -> tier_A=0, tier_B=1
        pool_b = panel[panel["pool_id"] == "pool_B"].iloc[0]
        assert pool_b["tier_A"] == 0
        assert pool_b["tier_B"] == 1

        # Pool C: RATS(2), WETH(0) -> tier_A=0, tier_B=2
        pool_c = panel[panel["pool_id"] == "pool_C"].iloc[0]
        assert pool_c["tier_A"] == 0
        assert pool_c["tier_B"] == 2

    def test_log_fee(self, synthetic_pools_df, synthetic_snapshots_df):
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        pool_a = panel[panel["pool_id"] == "pool_A"].iloc[0]
        assert np.isclose(pool_a["log_fee"], np.log(0.003))

    def test_all_expected_columns(
        self, synthetic_pools_df, synthetic_snapshots_df
    ):
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        expected_cols = {
            "pool_id", "chain", "date", "log_volume", "log_tvl",
            "log_tvl_lag1", "volatility", "weekend", "log_fee",
            "swap_fee", "tier_A", "tier_B", "tokens",
        }
        assert expected_cols.issubset(set(panel.columns))

    def test_zero_volume_rows_dropped(self, synthetic_pools_df):
        """Rows with volume_usd <= 0 must be excluded from the panel."""
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(5)]
        records = []
        for d in dates:
            records.append({
                "pool_id": "pool_A", "chain": "MAINNET", "date": d,
                "volume_usd": 1000.0,
                "total_liquidity_usd": 100000.0,
            })
        # Add a zero-volume row
        records.append({
            "pool_id": "pool_A", "chain": "MAINNET",
            "date": date(2026, 1, 6),
            "volume_usd": 0.0,
            "total_liquidity_usd": 100000.0,
        })
        snaps = pd.DataFrame(records)
        panel = assemble_panel(synthetic_pools_df, snaps, {})
        pool_a = panel[panel["pool_id"] == "pool_A"]
        # 5 valid rows, minus 1 for lag = 4 (the zero-volume row is skipped)
        assert len(pool_a) == 4

    def test_zero_tvl_rows_dropped(self, synthetic_pools_df):
        """Rows with total_liquidity_usd <= 0 must be excluded."""
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(5)]
        records = []
        for d in dates:
            records.append({
                "pool_id": "pool_A", "chain": "MAINNET", "date": d,
                "volume_usd": 1000.0,
                "total_liquidity_usd": 100000.0,
            })
        # Add a zero-TVL row
        records.append({
            "pool_id": "pool_A", "chain": "MAINNET",
            "date": date(2026, 1, 6),
            "volume_usd": 1000.0,
            "total_liquidity_usd": 0.0,
        })
        snaps = pd.DataFrame(records)
        panel = assemble_panel(synthetic_pools_df, snaps, {})
        pool_a = panel[panel["pool_id"] == "pool_A"]
        assert len(pool_a) == 4


# ===========================================================================
# TestSyntheticPanelColumns — structural model covariates in the fixture
# ===========================================================================


class TestSyntheticPanelColumns:
    """Tests that the synthetic_panel fixture has new structural columns."""

    def test_panel_has_log_sigma(self, synthetic_panel):
        assert "log_sigma" in synthetic_panel.columns
        expected = np.log(np.maximum(synthetic_panel["volatility"].values, 1e-6))
        np.testing.assert_allclose(
            synthetic_panel["log_sigma"].values, expected, rtol=1e-12,
        )

    def test_panel_has_dow_harmonics(self, synthetic_panel):
        assert "dow_sin" in synthetic_panel.columns
        assert "dow_cos" in synthetic_panel.columns
        assert (synthetic_panel["dow_sin"] >= -1.0).all()
        assert (synthetic_panel["dow_sin"] <= 1.0).all()
        assert (synthetic_panel["dow_cos"] >= -1.0).all()
        assert (synthetic_panel["dow_cos"] <= 1.0).all()

    def test_panel_has_interactions(self, synthetic_panel):
        assert "tvl_x_sigma" in synthetic_panel.columns
        assert "tvl_x_fee" in synthetic_panel.columns
        assert "sigma_x_fee" in synthetic_panel.columns

    def test_dow_harmonics_correct_for_known_date(self, synthetic_panel):
        """2026-01-03 is Saturday (weekday=5), so dow=5."""
        sat_rows = synthetic_panel[
            synthetic_panel["date"] == date(2026, 1, 3)
        ]
        if len(sat_rows) == 0:
            pytest.skip("No Saturday rows in fixture")
        expected_sin = np.sin(2 * np.pi * 5 / 7)
        expected_cos = np.cos(2 * np.pi * 5 / 7)
        np.testing.assert_allclose(
            sat_rows["dow_sin"].values[0], expected_sin, atol=1e-12,
        )
        np.testing.assert_allclose(
            sat_rows["dow_cos"].values[0], expected_cos, atol=1e-12,
        )

    def test_interactions_use_lagged_tvl(self, synthetic_panel):
        """tvl_x_sigma must use log_tvl_lag1, not log_tvl."""
        expected = (
            synthetic_panel["log_tvl_lag1"].values
            * synthetic_panel["log_sigma"].values
        )
        np.testing.assert_allclose(
            synthetic_panel["tvl_x_sigma"].values, expected, rtol=1e-12,
        )


# ===========================================================================
# TestAssemblePanelStructuralColumns — new columns from real pipeline
# ===========================================================================


class TestAssemblePanelStructuralColumns:
    """Tests that assemble_panel() produces new structural columns."""

    def test_assemble_panel_has_log_sigma(
        self, synthetic_pools_df, synthetic_snapshots_df
    ):
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        assert "log_sigma" in panel.columns
        expected = np.log(np.maximum(panel["volatility"].values, 1e-6))
        np.testing.assert_allclose(
            panel["log_sigma"].values, expected, rtol=1e-12,
        )

    def test_assemble_panel_has_dow_harmonics(
        self, synthetic_pools_df, synthetic_snapshots_df
    ):
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        assert "dow_sin" in panel.columns
        assert "dow_cos" in panel.columns

    def test_assemble_panel_has_interactions(
        self, synthetic_pools_df, synthetic_snapshots_df
    ):
        panel = assemble_panel(synthetic_pools_df, synthetic_snapshots_df, {})
        assert "tvl_x_sigma" in panel.columns
        assert "tvl_x_fee" in panel.columns
        assert "sigma_x_fee" in panel.columns
        # Interactions use lagged TVL
        expected = panel["log_tvl_lag1"].values * panel["log_sigma"].values
        np.testing.assert_allclose(
            panel["tvl_x_sigma"].values, expected, rtol=1e-12,
        )


# ===========================================================================
# TestValidatePanel
# ===========================================================================


class TestValidatePanel:
    def test_flags_constant_volume(self, synthetic_panel, capsys):
        panel = synthetic_panel.copy()
        mask = panel["pool_id"] == "pool_A"
        panel.loc[mask, "log_volume"] = 10.0
        validate_panel(panel)
        captured = capsys.readouterr()
        assert "near-constant" in captured.out or "constant" in captured.out.lower()

    def test_flags_tvl_jumps(self, synthetic_panel, capsys):
        panel = synthetic_panel.copy()
        idx = panel[panel["pool_id"] == "pool_A"].index[1]
        panel.loc[idx, "log_tvl"] = panel.loc[idx, "log_tvl"] + 5.0
        validate_panel(panel)
        captured = capsys.readouterr()
        assert "TVL jumps" in captured.out

    def test_flags_volume_exceeds_tvl(self, synthetic_panel, capsys):
        panel = synthetic_panel.copy()
        panel["log_volume"] = panel["log_tvl"] + 1.0
        validate_panel(panel)
        captured = capsys.readouterr()
        assert "volume > TVL" in captured.out

    def test_returns_dataframe_unchanged(self, synthetic_panel):
        result = validate_panel(synthetic_panel)
        pd.testing.assert_frame_equal(result, synthetic_panel)
