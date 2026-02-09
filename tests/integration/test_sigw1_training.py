"""Integration tests for Sig-W1 training pipeline."""

import jax
import jax.numpy as jnp
import pytest

from quantammsim.synthetic.generation import generate_synthetic_price_array_daily
from quantammsim.synthetic.model import NeuralSDE
from quantammsim.synthetic.training import fit_neural_sde_sigw1


@pytest.fixture
def fake_minute_prices():
    """Fake minute prices: 60 days, 2 assets.

    Needs enough days for train/val split + windowing.
    """
    key = jax.random.PRNGKey(0)
    n_days = 60
    n_assets = 2
    T = n_days * 1440
    log_rets = jax.random.normal(key, (T, n_assets)) * 0.0001
    log_prices = jnp.cumsum(log_rets, axis=0) + jnp.array([7.0, 8.5])
    return jnp.exp(log_prices)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.training
class TestSigW1Training:
    def test_sigw1_training_reduces_loss(self, fake_minute_prices):
        """A few training steps should reduce Sig-W1 loss from random init."""
        key = jax.random.PRNGKey(42)
        sde, loss_history = fit_neural_sde_sigw1(
            fake_minute_prices,
            n_assets=2,
            key=key,
            hidden_dim=16,
            window_lens=[5, 10],
            depth=2,
            mc_samples=8,
            batch_size=4,
            n_steps=30,
            lr=1e-3,
            patience=100,
            antithetic=True,
            verbose=True,
        )

        assert len(loss_history) == 30
        # Training loss should decrease over 30 steps
        first_losses = [l[0] for l in loss_history[:5]]
        last_losses = [l[0] for l in loss_history[-5:]]
        avg_first = sum(first_losses) / len(first_losses)
        avg_last = sum(last_losses) / len(last_losses)
        assert avg_last < avg_first, (
            f"Sig-W1 loss did not decrease: {avg_first:.6f} -> {avg_last:.6f}"
        )

    def test_sigw1_model_has_drift(self, fake_minute_prices):
        """Sig-W1 trained model should have learn_drift=True."""
        key = jax.random.PRNGKey(0)
        sde, _ = fit_neural_sde_sigw1(
            fake_minute_prices,
            n_assets=2,
            key=key,
            hidden_dim=16,
            window_lens=10,
            depth=2,
            mc_samples=4,
            batch_size=4,
            n_steps=5,
            patience=100,
            verbose=False,
        )
        assert sde.learn_drift is True


@pytest.mark.slow
@pytest.mark.integration
class TestSigW1DailyGenerationPipeline:
    def test_end_to_end(self, fake_minute_prices):
        """Train -> generate daily paths -> Brownian bridge -> minute prices."""
        key = jax.random.PRNGKey(99)

        # Train (tiny config)
        sde, _ = fit_neural_sde_sigw1(
            fake_minute_prices,
            n_assets=2,
            key=key,
            hidden_dim=16,
            window_lens=[5, 10],
            depth=2,
            mc_samples=4,
            batch_size=4,
            n_steps=5,
            patience=100,
            verbose=False,
        )

        # Generate
        key_gen = jax.random.PRNGKey(123)
        synthetic = generate_synthetic_price_array_daily(
            sde, fake_minute_prices, n_paths=2, key=key_gen
        )

        assert synthetic.shape == (fake_minute_prices.shape[0], 2, 2)
        assert jnp.all(synthetic > 0)
        assert jnp.all(jnp.isfinite(synthetic))
