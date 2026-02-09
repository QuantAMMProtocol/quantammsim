"""Unit tests for the Neural SDE synthetic price path generation."""

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from quantammsim.synthetic.model import (
    DiffusionNetwork,
    DriftNetwork,
    NeuralSDE,
    ZeroDrift,
    load_sde,
    save_sde,
)
from quantammsim.synthetic.training import (
    compute_daily_log_prices,
    fit_neural_sde,
    gaussian_nll,
    total_nll,
)
from quantammsim.synthetic.generation import (
    generate_minute_paths,
    generate_synthetic_price_array,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def sde(key):
    """Diffusion-only SDE (default)."""
    return NeuralSDE(n_assets=3, hidden_dim=32, key=key)


@pytest.fixture
def sde_with_drift(key):
    """SDE with learned drift."""
    return NeuralSDE(n_assets=3, hidden_dim=32, learn_drift=True, key=key)


@pytest.fixture
def sde_diagonal(key):
    return NeuralSDE(n_assets=3, hidden_dim=32, diagonal_only=True, key=key)


@pytest.fixture
def fake_minute_prices():
    """Fake minute-resolution price data: 10 days, 3 assets."""
    key = jax.random.PRNGKey(0)
    n_days = 10
    n_assets = 3
    T = n_days * 1440
    # Geometric Brownian motion-ish
    log_rets = jax.random.normal(key, (T, n_assets)) * 0.0001
    log_prices = jnp.cumsum(log_rets, axis=0) + jnp.array([7.0, 8.5, 0.0])
    return jnp.exp(log_prices)


# --- Model tests ---


class TestDriftNetwork:
    def test_output_shape(self, key):
        drift = DriftNetwork(n_assets=3, hidden_dim=32, key=key)
        y = jnp.ones(3)
        out = drift(y)
        assert out.shape == (3,)

    def test_output_finite(self, key):
        drift = DriftNetwork(n_assets=5, hidden_dim=16, key=key)
        y = jnp.array([7.0, 8.0, 0.0, 6.5, 9.0])
        out = drift(y)
        assert jnp.all(jnp.isfinite(out))


class TestZeroDrift:
    def test_output_shape(self):
        drift = ZeroDrift(n_assets=3)
        y = jnp.ones(3)
        out = drift(y)
        assert out.shape == (3,)

    def test_always_zero(self):
        drift = ZeroDrift(n_assets=4)
        y = jnp.array([7.0, 8.0, 0.0, 5.0])
        out = drift(y)
        assert jnp.allclose(out, 0.0)

    def test_different_inputs_same_zero(self):
        drift = ZeroDrift(n_assets=2)
        out1 = drift(jnp.array([1.0, 2.0]))
        out2 = drift(jnp.array([100.0, -50.0]))
        assert jnp.allclose(out1, out2)
        assert jnp.allclose(out1, 0.0)


class TestDiffusionNetwork:
    def test_output_shape(self, key):
        diff = DiffusionNetwork(n_assets=3, hidden_dim=32, key=key)
        y = jnp.ones(3)
        L = diff(y)
        assert L.shape == (3, 3)

    def test_positive_definite(self, key):
        diff = DiffusionNetwork(n_assets=3, hidden_dim=32, key=key)
        y = jnp.array([7.0, 8.0, 0.0])
        L = diff(y)
        Sigma = L @ L.T
        eigvals = jnp.linalg.eigvalsh(Sigma)
        assert jnp.all(eigvals > 0), f"Non-PD eigenvalues: {eigvals}"

    def test_lower_triangular(self, key):
        diff = DiffusionNetwork(n_assets=4, hidden_dim=32, key=key)
        y = jnp.ones(4)
        L = diff(y)
        # Upper triangle (excluding diagonal) should be zero
        assert jnp.allclose(jnp.triu(L, k=1), 0.0)

    def test_diagonal_only_mode(self, key):
        diff = DiffusionNetwork(n_assets=3, hidden_dim=32, diagonal_only=True, key=key)
        y = jnp.ones(3)
        L = diff(y)
        assert L.shape == (3, 3)
        # Off-diagonal should be zero
        off_diag = L - jnp.diag(jnp.diag(L))
        assert jnp.allclose(off_diag, 0.0)
        # Diagonal should be positive
        assert jnp.all(jnp.diag(L) > 0)

    def test_positive_definite_diagonal(self, key):
        diff = DiffusionNetwork(n_assets=3, hidden_dim=32, diagonal_only=True, key=key)
        y = jnp.array([7.0, 8.0, 0.0])
        L = diff(y)
        Sigma = L @ L.T
        eigvals = jnp.linalg.eigvalsh(Sigma)
        assert jnp.all(eigvals > 0)

    def test_init_scale_sets_magnitude(self, key):
        """With init_scale, L diagonal should be within an order of magnitude of target."""
        target_scale = jnp.array([0.0008, 0.0003, 0.0005])
        diff = DiffusionNetwork(n_assets=3, hidden_dim=32, init_scale=target_scale, key=key)
        y = jnp.array([7.0, 8.0, 0.0])
        L = diff(y)
        diag = jnp.abs(jnp.diag(L))
        # Within 10x of target — the MLP contrib is O(1) so diag ≈ scale * softplus(O(1))
        assert jnp.all(diag < target_scale * 10)
        assert jnp.all(diag > target_scale * 0.01)

    def test_no_init_scale_defaults_to_unit(self, key):
        """Without init_scale, log_scale should be zeros (scale = 1)."""
        diff = DiffusionNetwork(n_assets=2, hidden_dim=16, key=key)
        assert jnp.allclose(diff.log_scale, 0.0)


class TestNeuralSDE:
    def test_construction_diffusion_only(self, key):
        sde = NeuralSDE(n_assets=3, hidden_dim=32, key=key)
        assert sde.n_assets == 3
        assert sde.learn_drift is False
        assert isinstance(sde.drift, ZeroDrift)

    def test_construction_with_drift(self, key):
        sde = NeuralSDE(n_assets=3, hidden_dim=32, learn_drift=True, key=key)
        assert sde.learn_drift is True
        assert isinstance(sde.drift, DriftNetwork)

    def test_drift_diffusion_shapes(self, sde):
        y = jnp.ones(3)
        assert sde.drift(y).shape == (3,)
        assert sde.diffusion(y).shape == (3, 3)

    def test_diffusion_only_zero_drift(self, sde):
        y = jnp.array([7.0, 8.0, 0.0])
        assert jnp.allclose(sde.drift(y), 0.0)

    def test_with_drift_nonzero(self, sde_with_drift):
        y = jnp.array([7.0, 8.0, 0.0])
        mu = sde_with_drift.drift(y)
        # With random init, drift should generally not be exactly zero
        assert not jnp.allclose(mu, 0.0, atol=1e-8)


# --- Training tests ---


class TestGaussianNLL:
    def test_nll_finite(self, sde):
        y_t = jnp.array([7.0, 8.0, 0.0])
        y_tp1 = jnp.array([7.01, 7.99, 0.001])
        nll = gaussian_nll(sde, y_t, y_tp1)
        assert jnp.isfinite(nll), f"NLL is not finite: {nll}"

    def test_nll_finite_with_drift(self, sde_with_drift):
        y_t = jnp.array([7.0, 8.0, 0.0])
        y_tp1 = jnp.array([7.01, 7.99, 0.001])
        nll = gaussian_nll(sde_with_drift, y_t, y_tp1)
        assert jnp.isfinite(nll)

    def test_nll_differentiable(self, sde):
        y_t = jnp.array([7.0, 8.0, 0.0])
        y_tp1 = jnp.array([7.01, 7.99, 0.001])
        loss, grads = eqx.filter_value_and_grad(
            lambda s: gaussian_nll(s, y_t, y_tp1)
        )(sde)
        assert jnp.isfinite(loss)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)

    def test_nll_positive_for_unlikely_transition(self, sde):
        """A huge jump should have high NLL."""
        y_t = jnp.array([7.0, 8.0, 0.0])
        y_tp1_small = jnp.array([7.001, 8.001, 0.001])
        y_tp1_large = jnp.array([17.0, 18.0, 10.0])
        nll_small = gaussian_nll(sde, y_t, y_tp1_small)
        nll_large = gaussian_nll(sde, y_t, y_tp1_large)
        assert nll_large > nll_small


class TestTotalNLL:
    def test_total_nll_shape(self, sde):
        daily = jnp.stack([jnp.array([7.0, 8.0, 0.0]) + i * 0.01 for i in range(5)])
        nll = total_nll(sde, daily)
        assert nll.shape == ()
        assert jnp.isfinite(nll)

    def test_total_nll_sums_transitions(self, sde):
        """Total NLL over 3 points should equal sum of 2 individual NLLs."""
        data = jnp.array([[7.0, 8.0, 0.0], [7.01, 8.01, 0.01], [7.02, 8.02, 0.02]])
        total = total_nll(sde, data)
        individual = gaussian_nll(sde, data[0], data[1]) + gaussian_nll(sde, data[1], data[2])
        assert jnp.allclose(total, individual, atol=1e-5)


class TestComputeDailyLogPrices:
    def test_shape(self, fake_minute_prices):
        daily = compute_daily_log_prices(fake_minute_prices)
        n_days = fake_minute_prices.shape[0] // 1440
        assert daily.shape == (n_days, fake_minute_prices.shape[1])

    def test_values_are_log(self, fake_minute_prices):
        daily = compute_daily_log_prices(fake_minute_prices)
        assert jnp.all(jnp.isfinite(daily))


class TestFitNeuralSDE:
    @pytest.mark.slow
    def test_fit_reduces_nll(self, fake_minute_prices, key):
        """A few training epochs should reduce NLL from random init."""
        sde, loss_history = fit_neural_sde(
            fake_minute_prices,
            n_assets=3,
            key=key,
            n_epochs=50,
            lr=1e-3,
            patience=100,
            batch_size=4096,
            verbose=False,
        )
        assert loss_history[-1][0] < loss_history[0][0], (
            f"Training NLL did not decrease: {loss_history[0][0]:.4f} -> {loss_history[-1][0]:.4f}"
        )

    @pytest.mark.slow
    def test_fit_with_drift(self, fake_minute_prices, key):
        """Training with learn_drift=True should also work."""
        sde, loss_history = fit_neural_sde(
            fake_minute_prices,
            n_assets=3,
            key=key,
            n_epochs=10,
            lr=1e-3,
            learn_drift=True,
            patience=100,
            batch_size=4096,
            verbose=False,
        )
        assert len(loss_history) == 10
        assert sde.learn_drift is True


# --- Generation tests ---


class TestGenerateMinutePaths:
    def test_output_shape(self, sde, key):
        y0 = jnp.array([7.0, 8.0, 0.0])
        paths = generate_minute_paths(sde, y0, n_steps=100, n_paths=3, key=key)
        assert paths.shape == (100, 3, 3)  # n_steps, n_assets, n_paths

    def test_all_finite(self, sde, key):
        y0 = jnp.array([7.0, 8.0, 0.0])
        paths = generate_minute_paths(sde, y0, n_steps=500, n_paths=4, key=key)
        assert jnp.all(jnp.isfinite(paths))

    def test_paths_differ(self, sde, key):
        """Different PRNG keys should produce different paths."""
        y0 = jnp.array([7.0, 8.0, 0.0])
        paths = generate_minute_paths(sde, y0, n_steps=100, n_paths=2, key=key)
        assert not jnp.allclose(paths[:, :, 0], paths[:, :, 1])

    def test_with_drift(self, sde_with_drift, key):
        y0 = jnp.array([7.0, 8.0, 0.0])
        paths = generate_minute_paths(sde_with_drift, y0, n_steps=100, n_paths=2, key=key)
        assert paths.shape == (100, 3, 2)
        assert jnp.all(jnp.isfinite(paths))


class TestGenerateSyntheticPriceArray:
    def test_output_shape(self, sde, fake_minute_prices, key):
        synthetic = generate_synthetic_price_array(
            sde, fake_minute_prices, n_paths=2, key=key
        )
        assert synthetic.shape == (fake_minute_prices.shape[0], 3, 2)

    def test_prices_positive(self, sde, fake_minute_prices, key):
        synthetic = generate_synthetic_price_array(
            sde, fake_minute_prices, n_paths=2, key=key
        )
        assert jnp.all(synthetic > 0)

    def test_all_finite(self, sde, fake_minute_prices, key):
        synthetic = generate_synthetic_price_array(
            sde, fake_minute_prices, n_paths=2, key=key
        )
        assert jnp.all(jnp.isfinite(synthetic))

    def test_initial_prices_match(self, sde, fake_minute_prices, key):
        """All synthetic paths should start at the same initial price."""
        synthetic = generate_synthetic_price_array(
            sde, fake_minute_prices, n_paths=3, key=key
        )
        initial_real = fake_minute_prices[0]
        for j in range(3):
            assert jnp.allclose(synthetic[0, :, j], initial_real, atol=1e-5)


# --- Save/load tests ---


class TestSaveLoad:
    def test_roundtrip_diffusion_only(self, sde):
        y = jnp.ones(3)
        L_orig = sde.diffusion(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_sde.eqx")
            save_sde(sde, path)
            loaded = load_sde(path, n_assets=3, hidden_dim=32, learn_drift=False)

        L_loaded = loaded.diffusion(y)
        assert jnp.allclose(L_orig, L_loaded)
        assert isinstance(loaded.drift, ZeroDrift)

    def test_roundtrip_with_drift(self, sde_with_drift):
        y = jnp.ones(3)
        drift_orig = sde_with_drift.drift(y)
        L_orig = sde_with_drift.diffusion(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_sde.eqx")
            save_sde(sde_with_drift, path)
            loaded = load_sde(path, n_assets=3, hidden_dim=32, learn_drift=True)

        assert jnp.allclose(drift_orig, loaded.drift(y))
        assert jnp.allclose(L_orig, loaded.diffusion(y))

    def test_roundtrip_diagonal(self, sde_diagonal):
        y = jnp.ones(3)
        L_orig = sde_diagonal.diffusion(y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_sde_diag.eqx")
            save_sde(sde_diagonal, path)
            loaded = load_sde(path, n_assets=3, hidden_dim=32, diagonal_only=True)

        L_loaded = loaded.diffusion(y)
        assert jnp.allclose(L_orig, L_loaded)

    def test_creates_parent_dirs(self, sde):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "dirs" / "model.eqx")
            save_sde(sde, path)
            assert Path(path).exists()
