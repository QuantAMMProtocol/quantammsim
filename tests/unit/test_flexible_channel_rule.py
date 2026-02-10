"""
Unit tests for the flexible channel strategy rule.

These tests cover:
- low-level kernel invariants for _jax_flexible_channel_weight_update
- pool-level output behavior for FlexibleChannelPool.calculate_rule_outputs
- causality/no-lookahead for rule outputs
"""

import numpy as np
import jax.numpy as jnp

from quantammsim.pools.G3M.quantamm.flexible_channel_pool import (
    FlexibleChannelPool,
    _jax_flexible_channel_weight_update,
)


def _make_params(n_assets: int = 2):
    """Create a minimal flexible-channel parameter set for tests."""
    return {
        "log_k": jnp.full((n_assets,), 4.0),
        "logit_lamb": jnp.full((n_assets,), 2.0),
        "logit_delta_lamb": jnp.zeros((n_assets,)),
        "logit_lamb_drawdown": jnp.full((n_assets,), 1.0),
        "log_amplitude": jnp.full((n_assets,), -3.0),
        "raw_width": jnp.full((n_assets,), 0.0),
        "raw_alpha": jnp.full((n_assets,), 0.0),
        "raw_exponents_up": jnp.full((n_assets,), 1.0),
        "raw_exponents_down": jnp.full((n_assets,), 1.0),
        "raw_pre_exp_scaling": jnp.full((n_assets,), 0.0),
        "logit_risk_off": jnp.zeros((n_assets,)),
        "logit_risk_on": jnp.zeros((n_assets,)),
        "raw_kelly_kappa": jnp.full((n_assets,), 0.0),
        "logit_lamb_vol": jnp.full((n_assets,), 0.0),
        "raw_entropy_floor": jnp.array([0.0]),
        "initial_weights_logits": jnp.zeros((n_assets,)),
        "subsidary_params": [],
    }


def _make_run_fingerprint(n_assets: int = 2, chunk_period: int = 2):
    """Create a minimal run_fingerprint for flexible-channel rule-output tests."""
    return {
        "n_assets": n_assets,
        "chunk_period": chunk_period,
        "max_memory_days": 365.0,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "risk_controls": {
            # Disable adaptive shrink in unit tests so behavior stays transparent.
            "use_entropy_shrink": False,
        },
        "optimisation_settings": {
            "freeze_risk_logits": False,
        },
    }


class TestFlexibleChannelKernel:
    """Tests for _jax_flexible_channel_weight_update."""

    def test_zero_gradient_and_zero_pi_gives_zero_update(self):
        """With no gradient/profit/drawdown signal, updates should be zero."""
        updates = _jax_flexible_channel_weight_update(
            price_gradient=jnp.zeros((1, 2)),
            k=jnp.array([[2.0, 2.0]]),
            width_env=jnp.array([[0.4, 0.4]]),
            amplitude=jnp.array([[0.1, 0.1]]),
            alpha=jnp.array([[1.0, 1.0]]),
            exponents_up=jnp.array([[2.0, 2.0]]),
            exponents_down=jnp.array([[2.0, 2.0]]),
            risk_off=jnp.array([[0.5, 0.5]]),
            risk_on=jnp.array([[0.5, 0.5]]),
            profit_pos=jnp.array([[0.0]]),
            drawdown_neg=jnp.array([[0.0]]),
            pre_exp_scaling=jnp.array([[0.5, 0.5]]),
        )

        assert jnp.allclose(updates, 0.0, atol=1e-12)

    def test_updates_sum_to_zero(self):
        """Weight updates must conserve total weight at each timestep."""
        updates = _jax_flexible_channel_weight_update(
            price_gradient=jnp.array([[0.15, -0.10, 0.05], [-0.20, 0.10, 0.08]]),
            k=jnp.array([[2.0, 1.5, 1.0], [2.0, 1.5, 1.0]]),
            width_env=jnp.array([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]),
            amplitude=jnp.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]),
            alpha=jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            exponents_up=jnp.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            exponents_down=jnp.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
            risk_off=jnp.array([[0.3, 0.4, 0.5], [0.3, 0.4, 0.5]]),
            risk_on=jnp.array([[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]),
            profit_pos=jnp.array([[0.05], [0.02]]),
            drawdown_neg=jnp.array([[0.01], [0.03]]),
            pre_exp_scaling=jnp.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        )

        row_sums = jnp.sum(updates, axis=1)
        assert jnp.allclose(row_sums, 0.0, atol=1e-10)

    def test_risk_off_profit_term_tilts_toward_high_risk_off_assets(self):
        """When profit signal is positive, larger risk_off should attract weight."""
        updates = _jax_flexible_channel_weight_update(
            price_gradient=jnp.zeros((1, 2)),
            k=jnp.array([[1.0, 1.0]]),
            width_env=jnp.array([[0.2, 0.2]]),
            amplitude=jnp.array([[0.1, 0.1]]),
            alpha=jnp.array([[1.0, 1.0]]),
            exponents_up=jnp.array([[2.0, 2.0]]),
            exponents_down=jnp.array([[2.0, 2.0]]),
            risk_off=jnp.array([[0.9, 0.1]]),
            risk_on=jnp.array([[0.0, 0.0]]),
            profit_pos=jnp.array([[0.20]]),
            drawdown_neg=jnp.array([[0.0]]),
            pre_exp_scaling=jnp.array([[0.5, 0.5]]),
        )

        assert updates[0, 0] > 0.0
        assert updates[0, 1] < 0.0

    def test_risk_on_drawdown_amplifies_trend_component(self):
        """Higher risk_on should increase trend-driven update magnitude under drawdown."""
        common_kwargs = {
            "price_gradient": jnp.array([[0.30, -0.30]]),
            "k": jnp.array([[1.0, 1.0]]),
            "width_env": jnp.array([[0.05, 0.05]]),  # narrow envelope -> trend dominates
            "amplitude": jnp.array([[0.0, 0.0]]),  # isolate trend component
            "alpha": jnp.array([[1.0, 1.0]]),
            "exponents_up": jnp.array([[1.0, 1.0]]),
            "exponents_down": jnp.array([[1.0, 1.0]]),
            "risk_off": jnp.array([[0.0, 0.0]]),
            "profit_pos": jnp.array([[0.0]]),
            "drawdown_neg": jnp.array([[0.2]]),
            "pre_exp_scaling": jnp.array([[0.5, 0.5]]),
        }
        updates_low = _jax_flexible_channel_weight_update(
            risk_on=jnp.array([[0.0, 0.0]]),
            **common_kwargs,
        )
        updates_high = _jax_flexible_channel_weight_update(
            risk_on=jnp.array([[1.0, 1.0]]),
            **common_kwargs,
        )

        assert jnp.abs(updates_high[0, 0]) > jnp.abs(updates_low[0, 0])
        assert jnp.sign(updates_high[0, 0]) == jnp.sign(updates_low[0, 0])


class TestFlexibleChannelPoolRuleOutputs:
    """Tests for FlexibleChannelPool.calculate_rule_outputs."""

    def test_rule_outputs_shape_finite_and_zero_sum(self):
        """Rule outputs should have expected shape, finite values, and zero-sum rows."""
        pool = FlexibleChannelPool()
        params = _make_params(n_assets=2)
        rf = _make_run_fingerprint(n_assets=2, chunk_period=2)

        n_timesteps = 240
        t = jnp.arange(n_timesteps)
        prices = jnp.column_stack(
            [
                100.0 * jnp.exp(0.0005 * t),
                80.0 * jnp.exp(-0.0002 * t),
            ]
        )

        outputs = pool.calculate_rule_outputs(params, rf, prices, None)

        expected_steps = prices[:: rf["chunk_period"]].shape[0] - 1
        assert outputs.shape == (expected_steps, 2)
        assert jnp.all(jnp.isfinite(outputs))

        row_sums = jnp.sum(outputs, axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-9)

    def test_risk_on_parameter_changes_outputs_under_drawdown(self):
        """Changing risk_on should materially change outputs when drawdown exists."""
        pool = FlexibleChannelPool()
        rf = _make_run_fingerprint(n_assets=2, chunk_period=2)

        n_timesteps = 300
        up = jnp.linspace(100.0, 140.0, n_timesteps // 2)
        down = jnp.linspace(140.0, 70.0, n_timesteps // 2)
        asset_0 = jnp.concatenate([up, down], axis=0)
        asset_1 = jnp.linspace(100.0, 103.0, n_timesteps)
        prices = jnp.column_stack([asset_0, asset_1])

        params_low = _make_params(n_assets=2)
        params_high = _make_params(n_assets=2)
        params_low["logit_risk_on"] = jnp.full((2,), -8.0)
        params_high["logit_risk_on"] = jnp.full((2,), 8.0)

        outputs_low = pool.calculate_rule_outputs(params_low, rf, prices, None)
        outputs_high = pool.calculate_rule_outputs(params_high, rf, prices, None)

        max_diff = jnp.max(jnp.abs(outputs_high - outputs_low))
        assert max_diff > 1e-8, "risk_on should influence flexible-channel outputs"

    def test_rule_outputs_no_lookahead(self):
        """Outputs before cutoff should not change when only future prices are modified."""
        pool = FlexibleChannelPool()
        params = _make_params(n_assets=2)
        rf = _make_run_fingerprint(n_assets=2, chunk_period=2)

        n_timesteps = 260
        t = jnp.arange(n_timesteps)
        prices = jnp.column_stack(
            [
                120.0 * jnp.exp(0.0004 * t),
                90.0 * jnp.exp(-0.0001 * t),
            ]
        )
        full_outputs = pool.calculate_rule_outputs(params, rf, prices, None)

        cutoff = 160
        truncated_prices = jnp.concatenate(
            [
                prices[:cutoff],
                prices[cutoff:].at[:, 0].set(1000.0),
            ],
            axis=0,
        )
        truncated_outputs = pool.calculate_rule_outputs(params, rf, truncated_prices, None)

        coarse_cutoff = cutoff // rf["chunk_period"]
        pre_diff = jnp.max(
            jnp.abs(
                full_outputs[: coarse_cutoff - 1] - truncated_outputs[: coarse_cutoff - 1]
            )
        )
        post_diff = jnp.max(
            jnp.abs(full_outputs[coarse_cutoff:] - truncated_outputs[coarse_cutoff:])
        )

        assert pre_diff < 1e-10, f"Lookahead detected in flexible-channel outputs: {pre_diff}"
        assert post_diff > 1e-8, "Sanity check failed: outputs should differ after cutoff"
