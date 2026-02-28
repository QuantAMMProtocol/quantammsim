"""
Tests for post_train_analysis.py statistical functions:
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
- Block bootstrap confidence intervals
- Impermanent loss decomposition
"""

import pytest
import numpy as np

from quantammsim.utils.post_train_analysis import (
    deflated_sharpe_ratio,
    block_bootstrap_sharpe_ci,
    decompose_pool_returns,
)


# =============================================================================
# Deflated Sharpe Ratio
# =============================================================================

class TestDeflatedSharpeRatio:
    """Tests for deflated_sharpe_ratio()."""

    def test_basic_output_structure(self):
        result = deflated_sharpe_ratio(observed_sr=1.5, n_trials=100, T=500)
        assert "sr0" in result
        assert "dsr" in result
        assert "significant" in result
        assert "observed_sr" in result
        assert "n_trials" in result
        assert "T" in result

    def test_dsr_between_0_and_1(self):
        result = deflated_sharpe_ratio(observed_sr=1.0, n_trials=50, T=250)
        assert 0.0 <= result["dsr"] <= 1.0

    def test_high_sr_many_observations_is_significant(self):
        """A genuinely high SR with many observations should pass DSR."""
        result = deflated_sharpe_ratio(observed_sr=3.0, n_trials=10, T=1000)
        assert result["significant"] is True
        assert result["dsr"] > 0.95

    def test_low_sr_many_trials_is_not_significant(self):
        """A mediocre SR found after many trials is likely noise."""
        result = deflated_sharpe_ratio(observed_sr=0.3, n_trials=500, T=100)
        assert result["significant"] is False

    def test_more_trials_increases_sr0(self):
        """Expected max SR under null increases with more trials."""
        result_few = deflated_sharpe_ratio(observed_sr=1.0, n_trials=10, T=250)
        result_many = deflated_sharpe_ratio(observed_sr=1.0, n_trials=500, T=250)
        assert result_many["sr0"] > result_few["sr0"]

    def test_more_observations_increases_dsr(self):
        """More observations → tighter SR estimate → higher DSR for same SR."""
        # Use SR close to the sr0 threshold so neither saturates at 1.0
        result_few = deflated_sharpe_ratio(observed_sr=0.4, n_trials=100, T=50)
        result_many = deflated_sharpe_ratio(observed_sr=0.4, n_trials=100, T=100)
        assert result_many["dsr"] > result_few["dsr"]

    def test_negative_sr_not_significant(self):
        result = deflated_sharpe_ratio(observed_sr=-0.5, n_trials=10, T=250)
        assert result["significant"] is False
        assert result["dsr"] < 0.5

    def test_single_trial(self):
        """With 1 trial, sr0 should be 0 (no multiple testing penalty)."""
        result = deflated_sharpe_ratio(observed_sr=1.0, n_trials=1, T=250)
        assert result["sr0"] == 0.0

    def test_zero_trials(self):
        result = deflated_sharpe_ratio(observed_sr=1.0, n_trials=0, T=250)
        assert result["sr0"] == 0.0

    def test_T_equals_1(self):
        """Edge case: only 1 observation."""
        result = deflated_sharpe_ratio(observed_sr=1.0, n_trials=10, T=1)
        assert result["sr0"] == 0.0

    def test_non_normal_returns_fat_tails(self):
        """Fat-tailed returns (positive excess kurtosis) should make DSR harder to pass."""
        result_normal = deflated_sharpe_ratio(observed_sr=1.5, n_trials=50, T=250, kurt=0.0)
        result_fat = deflated_sharpe_ratio(observed_sr=1.5, n_trials=50, T=250, kurt=6.0)
        # Fat tails inflate SR variance → harder to distinguish from noise
        assert result_fat["dsr"] <= result_normal["dsr"]

    def test_skewed_returns(self):
        """Negative skew should make the DSR computation run without error."""
        result = deflated_sharpe_ratio(observed_sr=1.0, n_trials=50, T=250, skew=-1.0)
        assert 0.0 <= result["dsr"] <= 1.0

    def test_observed_sr_preserved(self):
        result = deflated_sharpe_ratio(observed_sr=2.34, n_trials=10, T=100)
        assert result["observed_sr"] == pytest.approx(2.34)


# =============================================================================
# Block Bootstrap Confidence Intervals
# =============================================================================

class TestBlockBootstrapSharpeCi:
    """Tests for block_bootstrap_sharpe_ci()."""

    def test_basic_output_structure(self):
        rng = np.random.RandomState(0)
        returns = rng.normal(0.001, 0.02, size=200)
        result = block_bootstrap_sharpe_ci(returns)
        assert "point_estimate" in result
        assert "lower" in result
        assert "upper" in result
        assert "std" in result
        assert "confidence_level" in result

    def test_lower_less_than_upper(self):
        rng = np.random.RandomState(0)
        returns = rng.normal(0.001, 0.02, size=200)
        result = block_bootstrap_sharpe_ci(returns)
        assert result["lower"] < result["upper"]

    def test_point_estimate_within_ci(self):
        rng = np.random.RandomState(0)
        returns = rng.normal(0.001, 0.02, size=200)
        result = block_bootstrap_sharpe_ci(returns)
        assert result["lower"] <= result["point_estimate"] <= result["upper"]

    def test_too_few_observations_returns_warning(self):
        returns = np.array([0.01, 0.02, 0.03])
        result = block_bootstrap_sharpe_ci(returns, block_length=10)
        assert "warning" in result
        assert np.isnan(result["point_estimate"])

    def test_confidence_level_preserved(self):
        rng = np.random.RandomState(0)
        returns = rng.normal(0.001, 0.02, size=200)
        result = block_bootstrap_sharpe_ci(returns, confidence=0.90)
        assert result["confidence_level"] == 0.90

    def test_wider_ci_for_higher_confidence(self):
        rng = np.random.RandomState(0)
        returns = rng.normal(0.001, 0.02, size=200)
        ci_90 = block_bootstrap_sharpe_ci(returns, confidence=0.90, seed=42)
        ci_99 = block_bootstrap_sharpe_ci(returns, confidence=0.99, seed=42)
        width_90 = ci_90["upper"] - ci_90["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]
        assert width_99 > width_90

    def test_deterministic_with_seed(self):
        rng = np.random.RandomState(0)
        returns = rng.normal(0.001, 0.02, size=200)
        result1 = block_bootstrap_sharpe_ci(returns, seed=123)
        result2 = block_bootstrap_sharpe_ci(returns, seed=123)
        assert result1["lower"] == result2["lower"]
        assert result1["upper"] == result2["upper"]

    def test_zero_variance_returns(self):
        """All identical returns → SR undefined, should return 0."""
        returns = np.ones(100) * 0.001
        result = block_bootstrap_sharpe_ci(returns)
        assert result["point_estimate"] == 0.0
        assert result["lower"] == 0.0
        assert result["upper"] == 0.0
        assert result["std"] == 0.0

    def test_strongly_positive_returns_give_positive_ci(self):
        """Strong positive drift should give CI entirely above zero."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.01, 0.005, size=500)  # Very high SR
        result = block_bootstrap_sharpe_ci(returns)
        assert result["lower"] > 0

    def test_annualisation_factor(self):
        """Point estimate should be annualised (√365 scaling)."""
        rng = np.random.RandomState(0)
        returns = rng.normal(0.001, 0.02, size=200)
        result = block_bootstrap_sharpe_ci(returns)
        # Raw daily SR
        daily_sr = returns.mean() / returns.std()
        expected_annualised = daily_sr * np.sqrt(365)
        assert result["point_estimate"] == pytest.approx(expected_annualised, rel=1e-6)


# =============================================================================
# Impermanent Loss Decomposition
# =============================================================================

class TestDecomposePoolReturns:
    """Tests for decompose_pool_returns()."""

    @pytest.fixture
    def simple_2asset_scenario(self):
        """Simple 2-asset scenario: asset 0 doubles, asset 1 stays flat."""
        T = 100
        prices = np.ones((T, 2))
        prices[:, 0] = np.linspace(100, 200, T)  # doubles
        prices[:, 1] = 100.0  # flat

        # Equal-weight pool: 50/50
        initial_value = 10000.0
        initial_reserves = np.array([50.0, 50.0])  # 50 of each at price 100

        # HODL: hold initial reserves, revalue at final prices
        # hodl_final = 50 * 200 + 50 * 100 = 15000
        # hodl_return = 15000 / 10000 - 1 = 0.5

        # CW AMM with 50/50 weights:
        # V_T/V_0 = (200/100)^0.5 * (100/100)^0.5 = √2 ≈ 1.4142
        # cw_amm_return ≈ 0.4142
        # divergence = 0.4142 - 0.5 = -0.0858

        values = np.linspace(initial_value, initial_value * 1.45, T)
        reserves = np.column_stack([
            np.linspace(50, 36.2, T),  # less of the appreciating asset
            np.linspace(50, 72.4, T),  # more of the flat asset
        ])

        return values, reserves, prices

    def test_basic_output_structure(self, simple_2asset_scenario):
        values, reserves, prices = simple_2asset_scenario
        result = decompose_pool_returns(values, reserves, prices)
        assert "pool_return" in result
        assert "hodl_return" in result
        assert "cw_amm_return" in result
        assert "divergence_loss" in result
        assert "fee_income" in result
        assert "strategy_alpha" in result
        assert "initial_weights" in result

    def test_decomposition_sums_correctly(self, simple_2asset_scenario):
        """pool_return = hodl_return + divergence_loss + fee_income + strategy_alpha"""
        values, reserves, prices = simple_2asset_scenario
        result = decompose_pool_returns(values, reserves, prices)
        reconstructed = (
            result["hodl_return"]
            + result["divergence_loss"]
            + result["fee_income"]
            + result["strategy_alpha"]
        )
        assert result["pool_return"] == pytest.approx(reconstructed, abs=1e-10)

    def test_hodl_return_correct(self, simple_2asset_scenario):
        """HODL: hold initial reserves, revalue at final prices."""
        values, reserves, prices = simple_2asset_scenario
        result = decompose_pool_returns(values, reserves, prices)
        # 50 * 200 + 50 * 100 = 15000, initial = 10000
        assert result["hodl_return"] == pytest.approx(0.5, abs=1e-10)

    def test_cw_amm_return_correct(self, simple_2asset_scenario):
        """CW AMM return: prod(price_ratio_i ^ w_i) - 1."""
        values, reserves, prices = simple_2asset_scenario
        result = decompose_pool_returns(values, reserves, prices)
        # 50/50 weights → (2.0)^0.5 * (1.0)^0.5 - 1 = √2 - 1
        assert result["cw_amm_return"] == pytest.approx(np.sqrt(2) - 1, abs=1e-10)

    def test_divergence_loss_is_negative(self, simple_2asset_scenario):
        """Divergence loss should be non-positive for unequal price moves."""
        values, reserves, prices = simple_2asset_scenario
        result = decompose_pool_returns(values, reserves, prices)
        assert result["divergence_loss"] <= 0

    def test_no_price_change_means_no_divergence(self):
        """If prices don't change, divergence loss = 0."""
        T = 50
        prices = np.ones((T, 2)) * 100.0
        values = np.ones(T) * 10000.0
        reserves = np.ones((T, 2)) * 50.0
        result = decompose_pool_returns(values, reserves, prices)
        assert result["divergence_loss"] == pytest.approx(0.0, abs=1e-10)
        assert result["hodl_return"] == pytest.approx(0.0, abs=1e-10)
        assert result["cw_amm_return"] == pytest.approx(0.0, abs=1e-10)

    def test_fee_income_passthrough(self, simple_2asset_scenario):
        values, reserves, prices = simple_2asset_scenario
        result = decompose_pool_returns(values, reserves, prices, fees_earned=0.05)
        assert result["fee_income"] == 0.05
        # Alpha should decrease by fee amount
        result_nofee = decompose_pool_returns(values, reserves, prices, fees_earned=0.0)
        assert result["strategy_alpha"] == pytest.approx(
            result_nofee["strategy_alpha"] - 0.05, abs=1e-10
        )

    def test_explicit_weights(self, simple_2asset_scenario):
        values, reserves, prices = simple_2asset_scenario
        weights = np.ones((len(values), 2)) * 0.5  # explicit 50/50
        result = decompose_pool_returns(values, reserves, prices, weights=weights)
        # Should use explicit weights, giving same result as inferred
        result_inferred = decompose_pool_returns(values, reserves, prices)
        assert result["cw_amm_return"] == pytest.approx(
            result_inferred["cw_amm_return"], abs=1e-10
        )

    def test_unequal_weights(self):
        """Non-equal weights should change the CW AMM benchmark."""
        T = 50
        prices = np.ones((T, 2))
        prices[:, 0] = np.linspace(100, 200, T)
        prices[:, 1] = 100.0

        # 80/20 pool
        initial_value = 10000.0
        values = np.ones(T) * initial_value
        reserves = np.column_stack([
            np.ones(T) * 80.0,  # 80 units of asset 0
            np.ones(T) * 20.0,  # 20 units of asset 1
        ])
        # Initial value = 80*100 + 20*100 = 10000 ✓
        weights = np.ones((T, 2))
        weights[:, 0] = 0.8
        weights[:, 1] = 0.2

        result = decompose_pool_returns(values, reserves, prices, weights=weights)
        # CW AMM: (2.0)^0.8 * (1.0)^0.2 - 1 = 2^0.8 - 1 ≈ 0.7411
        assert result["cw_amm_return"] == pytest.approx(2**0.8 - 1, abs=1e-10)


# =============================================================================
# _compute_regime_tags (via TrainingEvaluator)
# =============================================================================

class TestComputeRegimeTags:
    """Tests for TrainingEvaluator._compute_regime_tags()."""

    def _make_evaluator(self):
        """Create a minimal TrainingEvaluator for testing _compute_regime_tags."""
        from unittest.mock import Mock
        from quantammsim.runners.training_evaluator import TrainingEvaluator

        trainer = Mock()
        trainer.name = "test"
        trainer.config = {}
        evaluator = TrainingEvaluator.__new__(TrainingEvaluator)
        evaluator.trainer = trainer
        evaluator.verbose = False
        return evaluator

    def _make_daily_prices(self, n_days, daily_returns_mean, daily_returns_std, n_assets=2):
        """Generate minute-level prices from daily return parameters.

        Creates 1440*n_days minute-level steps, but the underlying return
        distribution is set at the daily level.
        """
        steps_per_day = 1440
        T = n_days * steps_per_day
        rng = np.random.RandomState(42)

        # Generate daily log returns, then interpolate to minutes
        daily_log_returns = rng.normal(daily_returns_mean, daily_returns_std, size=(n_days, n_assets))
        # Spread each day's return evenly across 1440 minutes
        minute_log_returns = np.repeat(daily_log_returns / steps_per_day, steps_per_day, axis=0)
        log_prices = np.cumsum(minute_log_returns, axis=0)
        prices = 100.0 * np.exp(log_prices)

        # Prepend the starting price
        start = np.ones((1, n_assets)) * 100.0
        prices = np.vstack([start, prices])
        return prices

    def test_low_vol_regime(self):
        evaluator = self._make_evaluator()
        # Daily vol ~10% annualised → low_vol
        prices = self._make_daily_prices(180, 0.0, 0.005)
        data_dict = {"prices": prices}
        vol, trend = evaluator._compute_regime_tags(data_dict, 0, len(prices))
        assert vol == "low_vol"

    def test_high_vol_regime(self):
        evaluator = self._make_evaluator()
        # Daily vol ~115% annualised → high_vol (use 1 asset to avoid √n reduction)
        prices = self._make_daily_prices(180, 0.0, 0.06, n_assets=1)
        data_dict = {"prices": prices}
        vol, trend = evaluator._compute_regime_tags(data_dict, 0, len(prices))
        assert vol == "high_vol"

    def test_medium_vol_regime(self):
        evaluator = self._make_evaluator()
        # Daily vol ~57% annualised → medium_vol (use 1 asset)
        prices = self._make_daily_prices(180, 0.0, 0.03, n_assets=1)
        data_dict = {"prices": prices}
        vol, trend = evaluator._compute_regime_tags(data_dict, 0, len(prices))
        assert vol == "medium_vol"

    def test_bull_regime(self):
        evaluator = self._make_evaluator()
        # Strong positive drift
        prices = self._make_daily_prices(180, 0.003, 0.02)
        data_dict = {"prices": prices}
        _, trend = evaluator._compute_regime_tags(data_dict, 0, len(prices))
        assert trend == "bull"

    def test_bear_regime(self):
        evaluator = self._make_evaluator()
        # Strong negative drift
        prices = self._make_daily_prices(180, -0.003, 0.02)
        data_dict = {"prices": prices}
        _, trend = evaluator._compute_regime_tags(data_dict, 0, len(prices))
        assert trend == "bear"

    def test_sideways_regime(self):
        evaluator = self._make_evaluator()
        # Zero drift
        prices = self._make_daily_prices(180, 0.0, 0.02)
        data_dict = {"prices": prices}
        _, trend = evaluator._compute_regime_tags(data_dict, 0, len(prices))
        assert trend == "sideways"

    def test_too_short_returns_unknown(self):
        evaluator = self._make_evaluator()
        data_dict = {"prices": np.array([[100.0, 100.0]])}
        vol, trend = evaluator._compute_regime_tags(data_dict, 0, 1)
        assert vol == "unknown"
        assert trend == "unknown"

    def test_uses_daily_not_minute_vol(self):
        """Verify that vol is computed from daily-sampled prices, not minute prices.

        Minute-level vol * sqrt(1440*365) >> daily vol * sqrt(365) due to
        microstructure noise. If we accidentally use minute returns, the
        annualised vol will be much higher.
        """
        evaluator = self._make_evaluator()
        # Construct prices where daily vol is clearly low (~10% annualised)
        # but minute vol would be different due to noise pattern
        n_days = 90
        steps_per_day = 1440
        T = n_days * steps_per_day

        # Deterministic daily prices: gentle uptrend
        daily_prices = 100.0 * np.exp(np.linspace(0, 0.05, n_days + 1))
        # Interpolate to minute-level (no extra noise)
        minute_prices = np.interp(
            np.linspace(0, n_days, T + 1),
            np.arange(n_days + 1),
            daily_prices,
        )
        prices = minute_prices.reshape(-1, 1)  # single asset

        data_dict = {"prices": prices}
        vol, _ = evaluator._compute_regime_tags(data_dict, 0, len(prices))
        # Daily vol of a smooth exponential is ~0, so should be low_vol
        assert vol == "low_vol"

    def test_slice_indices_respected(self):
        """Only the [start:end] slice should be used for classification."""
        evaluator = self._make_evaluator()
        # First half: calm. Second half: volatile. Use 1 asset to avoid √n reduction.
        calm = self._make_daily_prices(90, 0.0, 0.005, n_assets=1)
        volatile = self._make_daily_prices(90, 0.0, 0.06, n_assets=1)
        prices = np.vstack([calm, volatile[1:]])  # skip duplicate start row

        data_dict = {"prices": prices}
        mid = len(calm)

        vol_first, _ = evaluator._compute_regime_tags(data_dict, 0, mid)
        vol_second, _ = evaluator._compute_regime_tags(data_dict, mid, len(prices))

        assert vol_first == "low_vol"
        assert vol_second == "high_vol"
