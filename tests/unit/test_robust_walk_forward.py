"""
Unit tests for robust walk-forward training utilities.

Tests:
1. Rademacher complexity computation
2. Rademacher haircut calculation
3. Walk-Forward Efficiency (WFE) computation
4. Cycle generation
"""

import pytest
import numpy as np
from quantammsim.runners.robust_walk_forward import (
    compute_empirical_rademacher,
    compute_rademacher_haircut,
    compute_walk_forward_efficiency,
    generate_walk_forward_cycles,
    WalkForwardCycle,
)


class TestRademacherComplexity:
    """Tests for Rademacher complexity computation."""

    def test_rademacher_single_strategy(self):
        """Single strategy should have low complexity."""
        returns = np.random.randn(1, 100)
        r_hat = compute_empirical_rademacher(returns)
        # Single strategy can't "fit" noise much - should be near zero
        # (can be slightly negative due to random sampling)
        assert abs(r_hat) < 0.5  # Should be relatively low

    def test_rademacher_many_strategies(self):
        """More strategies should generally have higher complexity."""
        np.random.seed(42)
        returns_few = np.random.randn(5, 100)
        returns_many = np.random.randn(50, 100)

        r_hat_few = compute_empirical_rademacher(returns_few, seed=42)
        r_hat_many = compute_empirical_rademacher(returns_many, seed=42)

        # More strategies = more ability to fit noise
        assert r_hat_many > r_hat_few

    def test_rademacher_correlated_strategies(self):
        """Highly correlated strategies should have similar complexity to one."""
        np.random.seed(42)
        base_returns = np.random.randn(100)

        # Create highly correlated strategies (small noise)
        correlated = np.array([
            base_returns + np.random.randn(100) * 0.01
            for _ in range(10)
        ])

        # Create independent strategies
        independent = np.random.randn(10, 100)

        r_corr = compute_empirical_rademacher(correlated, seed=42)
        r_ind = compute_empirical_rademacher(independent, seed=42)

        # Correlated strategies should have lower complexity
        # (they span less of the strategy space)
        assert r_corr < r_ind

    def test_rademacher_empty_input(self):
        """Empty input should return 0."""
        r_hat = compute_empirical_rademacher(np.array([]))
        assert r_hat == 0.0

    def test_rademacher_reproducibility(self):
        """Same seed should give same result."""
        returns = np.random.randn(10, 100)
        r1 = compute_empirical_rademacher(returns, seed=123)
        r2 = compute_empirical_rademacher(returns, seed=123)
        assert r1 == r2


class TestRademacherHaircut:
    """Tests for Rademacher haircut calculation."""

    def test_haircut_positive(self):
        """Haircut should always be positive."""
        _, haircut = compute_rademacher_haircut(
            observed_sharpe=1.5,
            rademacher_complexity=0.1,
            T=100,
        )
        assert haircut > 0

    def test_haircut_increases_with_complexity(self):
        """Higher complexity = larger haircut."""
        _, haircut_low = compute_rademacher_haircut(1.5, 0.05, 100)
        _, haircut_high = compute_rademacher_haircut(1.5, 0.20, 100)
        assert haircut_high > haircut_low

    def test_haircut_decreases_with_more_data(self):
        """More data = smaller haircut (more confidence)."""
        _, haircut_small = compute_rademacher_haircut(1.5, 0.1, 50)
        _, haircut_large = compute_rademacher_haircut(1.5, 0.1, 500)
        assert haircut_large < haircut_small

    def test_adjusted_sharpe_less_than_observed(self):
        """Adjusted Sharpe should be less than observed (haircut is penalty)."""
        adj_sharpe, _ = compute_rademacher_haircut(1.5, 0.1, 100)
        assert adj_sharpe < 1.5

    def test_zero_complexity_still_has_estimation_error(self):
        """Even with R̂=0, there's estimation error."""
        adj_sharpe, haircut = compute_rademacher_haircut(1.5, 0.0, 100)
        assert haircut > 0  # Estimation error term
        assert adj_sharpe < 1.5

    def test_haircut_returns_nan_for_zero_T(self):
        """T=0 should return NaN, not divide by zero."""
        adj_sharpe, haircut = compute_rademacher_haircut(1.5, 0.1, 0)
        assert np.isnan(adj_sharpe)
        assert np.isnan(haircut)

    def test_haircut_returns_nan_for_negative_T(self):
        """Negative T should return NaN."""
        adj_sharpe, haircut = compute_rademacher_haircut(1.5, 0.1, -10)
        assert np.isnan(adj_sharpe)
        assert np.isnan(haircut)


class TestWalkForwardEfficiency:
    """Tests for WFE computation."""

    def test_wfe_perfect(self):
        """OOS = IS should give WFE = 1."""
        wfe = compute_walk_forward_efficiency(1.0, 1.0, 365, 90)
        assert wfe == 1.0

    def test_wfe_typical(self):
        """Typical case: OOS < IS, WFE < 1."""
        wfe = compute_walk_forward_efficiency(1.2, 0.8, 365, 90)
        assert 0 < wfe < 1
        assert abs(wfe - 0.8/1.2) < 1e-6

    def test_wfe_outperformance(self):
        """OOS > IS gives WFE > 1 (unusual but possible)."""
        wfe = compute_walk_forward_efficiency(0.8, 1.0, 365, 90)
        assert wfe > 1.0

    def test_wfe_zero_is_returns_nan(self):
        """Zero IS Sharpe should return NaN (undefined ratio)."""
        wfe = compute_walk_forward_efficiency(0.0, 0.5, 365, 90)
        assert np.isnan(wfe)

    def test_wfe_negative_is_returns_nan(self):
        """Negative IS Sharpe returns NaN (undefined ratio)."""
        # When IS is negative, WFE is undefined/meaningless
        wfe = compute_walk_forward_efficiency(-1.0, -0.5, 365, 90)
        assert np.isnan(wfe)

    def test_wfe_zero_is_negative_oos_returns_nan(self):
        """Zero IS with negative OOS returns NaN."""
        wfe = compute_walk_forward_efficiency(0.0, -0.5, 365, 90)
        assert np.isnan(wfe)


class TestCycleGeneration:
    """Tests for walk-forward cycle generation."""

    def test_cycle_count(self):
        """Should generate correct number of cycles."""
        cycles = generate_walk_forward_cycles(
            "2022-01-01 00:00:00",
            "2023-01-01 00:00:00",
            n_cycles=4,
        )
        assert len(cycles) == 4

    def test_cycle_ordering(self):
        """Cycles should be in chronological order."""
        cycles = generate_walk_forward_cycles(
            "2022-01-01 00:00:00",
            "2023-01-01 00:00:00",
            n_cycles=4,
        )

        for i in range(len(cycles) - 1):
            # Each cycle's test should start where training ends
            assert cycles[i].test_start_date == cycles[i].train_end_date

    def test_expanding_window(self):
        """Expanding window should always start from beginning."""
        cycles = generate_walk_forward_cycles(
            "2022-01-01 00:00:00",
            "2023-01-01 00:00:00",
            n_cycles=4,
            keep_fixed_start=True,
        )

        for cycle in cycles:
            # All training starts from beginning
            assert cycle.train_start_date == cycles[0].train_start_date

    def test_rolling_window(self):
        """Rolling window should move forward each cycle."""
        cycles = generate_walk_forward_cycles(
            "2022-01-01 00:00:00",
            "2023-01-01 00:00:00",
            n_cycles=4,
            keep_fixed_start=False,
        )

        # Training start should move forward
        train_starts = [c.train_start_date for c in cycles]
        assert len(set(train_starts)) == len(train_starts)  # All different


class TestSoftminGradientFlow:
    """Tests for softmin aggregation gradient flow."""

    def test_softmin_vs_min_gradients(self):
        """Softmin should have gradients for all elements, min only for one."""
        import jax.numpy as jnp
        from jax import grad
        from jax.nn import softmax

        def hard_min(x):
            return jnp.min(x)

        def soft_min(x, temp=1.0):
            weights = softmax(-x / temp)
            return jnp.sum(x * weights)

        x = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Hard min gradients
        hard_grads = grad(hard_min)(x)
        # Only minimum element gets gradient
        assert jnp.sum(hard_grads != 0) == 1

        # Soft min gradients
        soft_grads = grad(soft_min)(x)
        # All elements should have non-zero gradients
        assert jnp.all(soft_grads != 0)

    def test_softmin_temperature_effect(self):
        """Lower temperature should concentrate gradients more."""
        import jax.numpy as jnp
        from jax import grad
        from jax.nn import softmax

        def soft_min(x, temp):
            weights = softmax(-x / temp)
            return jnp.sum(x * weights)

        x = jnp.array([1.0, 2.0, 3.0, 4.0])

        grads_high_temp = grad(lambda x: soft_min(x, 10.0))(x)
        grads_low_temp = grad(lambda x: soft_min(x, 0.1))(x)

        # Low temp should have more concentrated gradients on minimum
        # (higher gradient magnitude for min element)
        assert jnp.abs(grads_low_temp[0]) > jnp.abs(grads_high_temp[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
