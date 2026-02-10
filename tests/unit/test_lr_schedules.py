"""
Tests for learning rate schedules in backpropagation.py.

Verifies that:
1. All schedule types work correctly (constant, cosine, exponential, warmup_cosine)
2. lr_decay_ratio properly computes min_lr = base_lr / lr_decay_ratio
3. Safety check triggers when min_lr >= base_lr
4. Schedules are properly wired through the optimizer chain
"""

import pytest
import jax.numpy as jnp

from quantammsim.training.backpropagation import _create_lr_schedule, create_optimizer_chain


class TestLRScheduleTypes:
    """Test each LR schedule type works correctly."""

    @pytest.fixture
    def base_settings(self):
        return {
            "base_lr": 0.01,
            "lr_decay_ratio": 1000,  # min_lr = 0.01 / 1000 = 1e-5
            "n_iterations": 1000,
        }

    def test_constant_schedule(self, base_settings):
        """Constant schedule should maintain base_lr throughout."""
        settings = {**base_settings, "lr_schedule_type": "constant"}
        schedule = _create_lr_schedule(settings)

        assert float(schedule(0)) == pytest.approx(0.01)
        assert float(schedule(500)) == pytest.approx(0.01)
        assert float(schedule(1000)) == pytest.approx(0.01)
        assert float(schedule(2000)) == pytest.approx(0.01)  # Beyond n_iterations

    def test_cosine_schedule(self, base_settings):
        """Cosine schedule should decay from base_lr to min_lr."""
        settings = {**base_settings, "lr_schedule_type": "cosine"}
        schedule = _create_lr_schedule(settings)

        start_lr = float(schedule(0))
        mid_lr = float(schedule(500))
        end_lr = float(schedule(1000))

        assert start_lr == pytest.approx(0.01)
        assert mid_lr < start_lr  # Should be decaying
        assert mid_lr > end_lr
        assert end_lr == pytest.approx(1e-5, rel=0.01)  # Should reach min_lr

    def test_exponential_schedule(self, base_settings):
        """Exponential schedule should decay from base_lr to min_lr."""
        settings = {**base_settings, "lr_schedule_type": "exponential"}
        schedule = _create_lr_schedule(settings)

        start_lr = float(schedule(0))
        mid_lr = float(schedule(500))
        end_lr = float(schedule(1000))

        assert start_lr == pytest.approx(0.01)
        assert mid_lr < start_lr
        assert mid_lr > end_lr
        assert end_lr == pytest.approx(1e-5, rel=0.01)

        # Exponential should be linear on log scale (geometric sequence)
        # mid_lr should be sqrt(start_lr * end_lr)
        expected_mid = (start_lr * end_lr) ** 0.5
        assert mid_lr == pytest.approx(expected_mid, rel=0.01)

    def test_warmup_cosine_schedule(self, base_settings):
        """Warmup cosine should warm up then decay."""
        settings = {
            **base_settings,
            "lr_schedule_type": "warmup_cosine",
            "warmup_steps": 100,
        }
        schedule = _create_lr_schedule(settings)

        # Should start at min_lr
        start_lr = float(schedule(0))
        assert start_lr == pytest.approx(1e-5, rel=0.1)

        # Should reach peak at warmup_steps
        peak_lr = float(schedule(100))
        assert peak_lr == pytest.approx(0.01, rel=0.01)

        # Should decay after warmup
        mid_lr = float(schedule(500))
        assert mid_lr < peak_lr
        assert mid_lr > 1e-5

        # Should reach min_lr at end
        end_lr = float(schedule(1000))
        assert end_lr == pytest.approx(1e-5, rel=0.01)


class TestLRDecayRatio:
    """Test lr_decay_ratio parameter handling."""

    def test_lr_decay_ratio_computes_min_lr(self):
        """lr_decay_ratio should compute min_lr = base_lr / ratio."""
        settings = {
            "base_lr": 0.1,
            "lr_decay_ratio": 100,
            "n_iterations": 100,
            "lr_schedule_type": "cosine",
        }
        schedule = _create_lr_schedule(settings)

        end_lr = float(schedule(100))
        expected_min_lr = 0.1 / 100  # 0.001
        assert end_lr == pytest.approx(expected_min_lr, rel=0.01)

    def test_lr_decay_ratio_takes_precedence_over_min_lr(self):
        """When both are provided, lr_decay_ratio should take precedence."""
        settings = {
            "base_lr": 0.01,
            "min_lr": 1e-8,  # Would give different result
            "lr_decay_ratio": 10,  # min_lr = 0.01 / 10 = 0.001
            "n_iterations": 100,
            "lr_schedule_type": "cosine",
        }
        schedule = _create_lr_schedule(settings)

        end_lr = float(schedule(100))
        # Should use lr_decay_ratio (0.001), not min_lr (1e-8)
        assert end_lr == pytest.approx(0.001, rel=0.01)

    def test_min_lr_fallback_when_no_decay_ratio(self):
        """Without lr_decay_ratio, should use min_lr directly."""
        settings = {
            "base_lr": 0.01,
            "min_lr": 0.0001,
            "n_iterations": 100,
            "lr_schedule_type": "cosine",
        }
        schedule = _create_lr_schedule(settings)

        end_lr = float(schedule(100))
        assert end_lr == pytest.approx(0.0001, rel=0.01)


class TestMinLRSafetyCheck:
    """Test safety check when min_lr >= base_lr."""

    def test_safety_check_triggers_when_min_lr_exceeds_base_lr(self):
        """When min_lr >= base_lr, should fallback to 100:1 ratio."""
        settings = {
            "base_lr": 1e-5,
            "min_lr": 1e-4,  # Greater than base_lr!
            "n_iterations": 100,
            "lr_schedule_type": "cosine",
        }
        schedule = _create_lr_schedule(settings)

        start_lr = float(schedule(0))
        end_lr = float(schedule(100))

        assert start_lr == pytest.approx(1e-5)
        # Should fallback to base_lr / 100
        expected_min = 1e-5 / 100
        assert end_lr == pytest.approx(expected_min, rel=0.01)
        assert end_lr < start_lr  # Must decay, not grow

    def test_safety_check_triggers_when_min_lr_equals_base_lr(self):
        """When min_lr == base_lr, should also trigger fallback."""
        settings = {
            "base_lr": 0.001,
            "min_lr": 0.001,  # Equal to base_lr
            "n_iterations": 100,
            "lr_schedule_type": "exponential",
        }
        schedule = _create_lr_schedule(settings)

        start_lr = float(schedule(0))
        end_lr = float(schedule(100))

        assert start_lr == pytest.approx(0.001)
        assert end_lr < start_lr  # Must decay

    def test_no_safety_check_for_constant_schedule(self):
        """Constant schedule shouldn't trigger safety check."""
        settings = {
            "base_lr": 1e-5,
            "min_lr": 1e-4,  # Greater than base_lr
            "n_iterations": 100,
            "lr_schedule_type": "constant",
        }
        # Should not raise error
        schedule = _create_lr_schedule(settings)
        assert float(schedule(0)) == pytest.approx(1e-5)


class TestOptimizerChainIntegration:
    """Test LR schedules integrate correctly with optimizer chain."""

    @pytest.fixture
    def base_fingerprint(self):
        return {
            "optimisation_settings": {
                "optimiser": "adam",
                "base_lr": 0.01,
                "lr_decay_ratio": 100,
                "n_iterations": 100,
                "use_plateau_decay": False,
                "use_gradient_clipping": False,
            }
        }

    @pytest.mark.parametrize("schedule_type", ["constant", "cosine", "exponential"])
    def test_schedule_wired_through_optimizer(self, base_fingerprint, schedule_type):
        """Each schedule type should work through create_optimizer_chain."""
        fp = base_fingerprint.copy()
        fp["optimisation_settings"] = {
            **fp["optimisation_settings"],
            "lr_schedule_type": schedule_type,
        }

        optimizer = create_optimizer_chain(fp)

        # Initialize with dummy params
        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        opt_state = optimizer.init(params)

        # Simulate gradient update
        grads = {"w": jnp.array([0.1, 0.1, 0.1])}
        updates, _ = optimizer.update(grads, opt_state, params)

        # Should produce valid updates
        assert not jnp.any(jnp.isnan(updates["w"]))
        assert jnp.all(jnp.isfinite(updates["w"]))

    def test_warmup_cosine_requires_warmup_steps(self, base_fingerprint):
        """warmup_cosine should require warmup_steps parameter."""
        fp = base_fingerprint.copy()
        fp["optimisation_settings"] = {
            **fp["optimisation_settings"],
            "lr_schedule_type": "warmup_cosine",
            "warmup_steps": 20,
        }

        optimizer = create_optimizer_chain(fp)
        params = {"w": jnp.array([1.0])}
        opt_state = optimizer.init(params)

        grads = {"w": jnp.array([0.1])}
        updates, _ = optimizer.update(grads, opt_state, params)

        assert not jnp.any(jnp.isnan(updates["w"]))
