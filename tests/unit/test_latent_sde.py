"""Unit tests for the Latent Neural SDE architecture."""

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from quantammsim.synthetic.model import (
    Encoder,
    LatentDiffusionNetwork,
    LatentDriftNetwork,
    LatentNeuralSDE,
    Readout,
    load_latent_sde,
    save_latent_sde,
)
from quantammsim.synthetic.generation import (
    generate_latent_daily_paths,
)
from quantammsim.synthetic.training import (
    sigw1_loss_multiscale_latent,
    compute_daily_log_prices,
    compute_real_expected_signature_multiscale,
    compute_drift_field,
    drifting_loss_latent,
    precompute_real_window_signatures,
)
from quantammsim.synthetic.augmentations import get_minimal_augmentation


N_ASSETS = 3
N_HIDDEN = 4
LATENT_DIM = N_ASSETS + N_HIDDEN
HIDDEN_DIM = 16


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def encoder(key):
    return Encoder(N_ASSETS, N_HIDDEN, HIDDEN_DIM, key=key)


@pytest.fixture
def readout():
    return Readout(N_ASSETS, LATENT_DIM)


@pytest.fixture
def latent_drift(key):
    return LatentDriftNetwork(LATENT_DIM, HIDDEN_DIM, key=key)


@pytest.fixture
def latent_diffusion(key):
    return LatentDiffusionNetwork(N_ASSETS, N_HIDDEN, HIDDEN_DIM, key=key)


@pytest.fixture
def latent_sde(key):
    return LatentNeuralSDE(N_ASSETS, N_HIDDEN, HIDDEN_DIM, key=key)


@pytest.fixture
def fake_minute_prices():
    """Fake minute prices: 30 days, 3 assets."""
    key = jax.random.PRNGKey(0)
    n_days = 30
    T = n_days * 1440
    log_rets = jax.random.normal(key, (T, N_ASSETS)) * 0.0001
    log_prices = jnp.cumsum(log_rets, axis=0) + jnp.array([7.0, 8.5, 0.0])
    return jnp.exp(log_prices)


# --- Encoder tests ---


class TestEncoder:
    def test_output_shape(self, encoder):
        y = jnp.ones(N_ASSETS)
        z = encoder(y)
        assert z.shape == (LATENT_DIM,)

    def test_first_n_assets_match_input(self, encoder):
        y = jnp.array([7.0, 8.5, 0.0])
        z = encoder(y)
        assert jnp.allclose(z[:N_ASSETS], y)

    def test_hidden_dims_nonzero(self, encoder):
        y = jnp.array([7.0, 8.5, 0.0])
        z = encoder(y)
        # With random MLP init, hidden dims should generally be non-zero
        assert not jnp.allclose(z[N_ASSETS:], 0.0, atol=1e-8)

    def test_finite(self, encoder):
        y = jnp.array([7.0, 8.5, 0.0])
        z = encoder(y)
        assert jnp.all(jnp.isfinite(z))

    def test_differentiable(self, encoder):
        y = jnp.array([7.0, 8.5, 0.0])
        _, grad = eqx.filter_value_and_grad(lambda enc: jnp.sum(enc(y)))(encoder)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


# --- Readout tests ---


class TestReadout:
    def test_output_shape(self, readout):
        z = jnp.ones(LATENT_DIM)
        y = readout(z)
        assert y.shape == (N_ASSETS,)

    def test_identity_init_recovers_asset_dims(self, readout):
        """At init, readout(Z) = Z[:n_assets] exactly."""
        z = jnp.array([7.0, 8.5, 0.0, 1.0, 2.0, 3.0, 4.0])
        y = readout(z)
        assert jnp.allclose(y, z[:N_ASSETS])

    def test_hidden_dims_ignored_at_init(self, readout):
        """Changing hidden dims shouldn't affect readout at init."""
        z1 = jnp.array([7.0, 8.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        z2 = jnp.array([7.0, 8.5, 0.0, 99.0, -50.0, 3.14, 2.71])
        assert jnp.allclose(readout(z1), readout(z2))

    def test_weight_shape(self, readout):
        assert readout.weight.shape == (N_ASSETS, LATENT_DIM)


# --- LatentDriftNetwork tests ---


class TestLatentDriftNetwork:
    def test_output_shape(self, latent_drift):
        z = jnp.ones(LATENT_DIM)
        out = latent_drift(z)
        assert out.shape == (LATENT_DIM,)

    def test_finite(self, latent_drift):
        z = jnp.array([7.0, 8.5, 0.0, 1.0, 2.0, 3.0, 4.0])
        out = latent_drift(z)
        assert jnp.all(jnp.isfinite(out))

    def test_differentiable(self, latent_drift):
        z = jnp.array([7.0, 8.5, 0.0, 1.0, 2.0, 3.0, 4.0])
        _, grad = eqx.filter_value_and_grad(lambda d: jnp.sum(d(z)))(latent_drift)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grad, eqx.is_array))
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)

    def test_no_init_drift_defaults_to_zero_bias(self, latent_drift):
        assert jnp.allclose(latent_drift.bias, 0.0)

    def test_init_drift_sets_bias_and_max_dev(self, key):
        """With init_drift, bias should match and max_dev should be 0.5 * |bias|."""
        init_drift = jnp.array([0.001, 0.001, 0.00001, 0.0005, 0.0, 0.0, 0.0])
        drift = LatentDriftNetwork(LATENT_DIM, HIDDEN_DIM, init_drift=init_drift, key=key)
        assert jnp.allclose(drift.bias, init_drift)
        expected_max_dev = jnp.maximum(jnp.abs(init_drift) * 0.5, 1e-4)
        assert jnp.allclose(drift.max_dev, expected_max_dev)

    def test_output_bounded_by_bias_plus_max_dev(self, key):
        """Output should be within [bias - max_dev, bias + max_dev] due to tanh."""
        init_drift = jnp.array([0.001, 0.001, 0.00001, 0.0005, 0.0, 0.0, 0.0])
        drift = LatentDriftNetwork(LATENT_DIM, HIDDEN_DIM, init_drift=init_drift, key=key)
        # Test with several different z values
        for z in [jnp.ones(LATENT_DIM), jnp.zeros(LATENT_DIM),
                  jnp.array([7.0, 8.5, 0.0, 1.0, 2.0, 3.0, 4.0]),
                  jnp.ones(LATENT_DIM) * 100.0]:
            out = drift(z)
            assert jnp.all(out >= drift.bias - drift.max_dev - 1e-7), f"Below lower bound: {out}"
            assert jnp.all(out <= drift.bias + drift.max_dev + 1e-7), f"Above upper bound: {out}"

    def test_output_near_bias_at_init(self, key):
        """At init, MLP output is small so drift should be close to bias."""
        init_drift = jnp.array([0.001, 0.001, 0.00001, 0.0005, 0.0, 0.0, 0.0])
        drift = LatentDriftNetwork(LATENT_DIM, HIDDEN_DIM, init_drift=init_drift, key=key)
        z = jnp.array([7.0, 8.5, 0.0, 1.0, 2.0, 3.0, 4.0])
        out = drift(z)
        deviation = jnp.abs(out - init_drift)
        # With 0.1x final layer and tanh, deviation should be small fraction of max_dev
        assert jnp.all(deviation < drift.max_dev), f"Deviation exceeds max_dev at init: {deviation}"

    def test_mlp_gradients_not_crushed(self, key):
        """Verify MLP parameters receive non-trivial gradients."""
        init_drift = jnp.array([0.001, 0.001, 0.00001, 0.0005, 0.0, 0.0, 0.0])
        drift = LatentDriftNetwork(LATENT_DIM, HIDDEN_DIM, init_drift=init_drift, key=key)
        z = jnp.array([7.0, 8.5, 0.0, 1.0, 2.0, 3.0, 4.0])

        def loss_fn(model):
            return jnp.sum(model(z) ** 2)

        grad = eqx.filter_grad(loss_fn)(drift)
        mlp_grads = jax.tree_util.tree_leaves(eqx.filter(grad.mlp, eqx.is_array))
        max_grad = max(float(jnp.max(jnp.abs(g))) for g in mlp_grads)
        assert max_grad > 1e-6, f"MLP gradients too small: {max_grad}"


# --- LatentDiffusionNetwork tests ---


class TestLatentDiffusionNetwork:
    def test_output_shape(self, latent_diffusion):
        z = jnp.ones(LATENT_DIM)
        L = latent_diffusion(z)
        assert L.shape == (LATENT_DIM, LATENT_DIM)

    def test_positive_definite(self, latent_diffusion):
        z = jnp.array([7.0, 8.5, 0.0, 1.0, 2.0, 3.0, 4.0])
        L = latent_diffusion(z)
        Sigma = L @ L.T
        eigvals = jnp.linalg.eigvalsh(Sigma)
        assert jnp.all(eigvals > 0), f"Non-PD eigenvalues: {eigvals}"

    def test_block_diagonal_structure(self, latent_diffusion):
        """Off-diagonal blocks should be zero: L[assets, hidden] = 0 and L[hidden, assets] = 0."""
        z = jnp.ones(LATENT_DIM)
        L = latent_diffusion(z)
        # Upper-right block: asset rows, hidden cols
        assert jnp.allclose(L[:N_ASSETS, N_ASSETS:], 0.0)
        # Lower-left block: hidden rows, asset cols
        assert jnp.allclose(L[N_ASSETS:, :N_ASSETS], 0.0)

    def test_asset_block_lower_triangular(self, latent_diffusion):
        z = jnp.ones(LATENT_DIM)
        L = latent_diffusion(z)
        L_asset = L[:N_ASSETS, :N_ASSETS]
        assert jnp.allclose(jnp.triu(L_asset, k=1), 0.0)

    def test_hidden_block_diagonal(self, latent_diffusion):
        z = jnp.ones(LATENT_DIM)
        L = latent_diffusion(z)
        L_hidden = L[N_ASSETS:, N_ASSETS:]
        off_diag = L_hidden - jnp.diag(jnp.diag(L_hidden))
        assert jnp.allclose(off_diag, 0.0)

    def test_all_diagonals_positive(self, latent_diffusion):
        z = jnp.ones(LATENT_DIM)
        L = latent_diffusion(z)
        diag = jnp.diag(L)
        assert jnp.all(diag > 0), f"Non-positive diag: {diag}"

    def test_init_scale_sets_magnitude(self, key):
        target_scale = jnp.array([0.01, 0.008, 0.005])
        diff = LatentDiffusionNetwork(
            N_ASSETS, N_HIDDEN, HIDDEN_DIM, init_scale=target_scale, key=key,
        )
        z = jnp.ones(LATENT_DIM)
        L = diff(z)
        # Asset diagonal should be within order of magnitude of target
        asset_diag = jnp.abs(jnp.diag(L[:N_ASSETS, :N_ASSETS]))
        assert jnp.all(asset_diag < target_scale * 10)
        assert jnp.all(asset_diag > target_scale * 0.01)

    def test_no_init_scale_defaults_to_unit(self, key):
        diff = LatentDiffusionNetwork(N_ASSETS, N_HIDDEN, HIDDEN_DIM, key=key)
        assert jnp.allclose(diff.log_scale, 0.0)

    def test_finite(self, latent_diffusion):
        z = jnp.array([7.0, 8.5, 0.0, 1.0, 2.0, 3.0, 4.0])
        L = latent_diffusion(z)
        assert jnp.all(jnp.isfinite(L))


# --- LatentNeuralSDE tests ---


class TestLatentNeuralSDE:
    def test_construction(self, latent_sde):
        assert latent_sde.n_assets == N_ASSETS
        assert latent_sde.n_hidden == N_HIDDEN
        assert latent_sde.latent_dim == LATENT_DIM

    def test_encode_drift_diffusion_readout(self, latent_sde):
        """End-to-end: encode -> drift/diff -> readout produces n_assets."""
        y = jnp.array([7.0, 8.5, 0.0])
        z = latent_sde.encoder(y)
        assert z.shape == (LATENT_DIM,)

        f = latent_sde.drift(z)
        assert f.shape == (LATENT_DIM,)

        L = latent_sde.diffusion(z)
        assert L.shape == (LATENT_DIM, LATENT_DIM)

        y_out = latent_sde.readout(z)
        assert y_out.shape == (N_ASSETS,)

    def test_identity_init_equivalence(self, latent_sde):
        """At init, readout(encoder(Y)) ≈ Y (identity on asset dims)."""
        y = jnp.array([7.0, 8.5, 0.0])
        z = latent_sde.encoder(y)
        y_out = latent_sde.readout(z)
        assert jnp.allclose(y_out, y)


# --- Save/Load tests ---


class TestSaveLoadLatentSDE:
    def test_roundtrip(self, latent_sde):
        y = jnp.array([7.0, 8.5, 0.0])
        z_orig = latent_sde.encoder(y)
        L_orig = latent_sde.diffusion(z_orig)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_latent_sde.eqx")
            save_latent_sde(latent_sde, path)
            loaded = load_latent_sde(
                path, n_assets=N_ASSETS, n_hidden=N_HIDDEN, hidden_dim=HIDDEN_DIM,
            )

        z_loaded = loaded.encoder(y)
        L_loaded = loaded.diffusion(z_loaded)
        assert jnp.allclose(z_orig, z_loaded)
        assert jnp.allclose(L_orig, L_loaded)

    def test_creates_parent_dirs(self, latent_sde):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "nested" / "dirs" / "model.eqx")
            save_latent_sde(latent_sde, path)
            assert Path(path).exists()


# --- Generation tests ---


class TestGenerateLatentDailyPaths:
    def test_output_shape(self, latent_sde, key):
        y0 = jnp.array([7.0, 8.5, 0.0])
        paths = generate_latent_daily_paths(
            latent_sde, y0, n_days=10, n_paths=3, key=key,
        )
        assert paths.shape == (10, N_ASSETS, 3)

    def test_all_finite(self, latent_sde, key):
        y0 = jnp.array([7.0, 8.5, 0.0])
        paths = generate_latent_daily_paths(
            latent_sde, y0, n_days=20, n_paths=4, key=key,
        )
        assert jnp.all(jnp.isfinite(paths))

    def test_readout_produces_n_assets(self, latent_sde, key):
        """Output should be n_assets, not latent_dim."""
        y0 = jnp.array([7.0, 8.5, 0.0])
        paths = generate_latent_daily_paths(
            latent_sde, y0, n_days=5, n_paths=2, key=key,
        )
        assert paths.shape[1] == N_ASSETS

    def test_paths_differ(self, latent_sde, key):
        y0 = jnp.array([7.0, 8.5, 0.0])
        paths = generate_latent_daily_paths(
            latent_sde, y0, n_days=10, n_paths=2, key=key,
        )
        assert not jnp.allclose(paths[:, :, 0], paths[:, :, 1])

    def test_antithetic(self, latent_sde, key):
        y0 = jnp.array([7.0, 8.5, 0.0])
        paths = generate_latent_daily_paths(
            latent_sde, y0, n_days=10, n_paths=6, key=key, antithetic=True,
        )
        assert paths.shape == (10, N_ASSETS, 6)
        assert jnp.all(jnp.isfinite(paths))

    def test_antithetic_odd_n_paths(self, latent_sde, key):
        y0 = jnp.array([7.0, 8.5, 0.0])
        paths = generate_latent_daily_paths(
            latent_sde, y0, n_days=10, n_paths=5, key=key, antithetic=True,
        )
        assert paths.shape == (10, N_ASSETS, 5)


# --- Sig-W1 loss tests ---


class TestSigW1LossLatent:
    def test_finite(self, latent_sde, fake_minute_prices, key):
        daily_log_prices = compute_daily_log_prices(fake_minute_prices)
        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * N_ASSETS  # lead-lag doubles dim
        real_sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5, 10], depth=2,
            augment_fn=augment_fn, aug_dim=aug_dim,
        )
        y0_batch = daily_log_prices[:3]
        loss = sigw1_loss_multiscale_latent(
            latent_sde, real_sigs, y0_batch,
            window_lens=[5, 10], depth=2, aug_dim=aug_dim,
            augment_fn=augment_fn, mc_samples=4, key=key,
        )
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_differentiable(self, latent_sde, fake_minute_prices, key):
        daily_log_prices = compute_daily_log_prices(fake_minute_prices)
        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * N_ASSETS
        real_sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5, 10], depth=2,
            augment_fn=augment_fn, aug_dim=aug_dim,
        )
        y0_batch = daily_log_prices[:2]

        loss, grads = eqx.filter_value_and_grad(
            lambda model: sigw1_loss_multiscale_latent(
                model, real_sigs, y0_batch,
                window_lens=[5, 10], depth=2, aug_dim=aug_dim,
                augment_fn=augment_fn, mc_samples=4, key=key,
            )
        )(latent_sde)

        assert jnp.isfinite(loss)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)

    def test_gradient_flows_through_encoder_and_readout(self, latent_sde, fake_minute_prices, key):
        """Verify gradient is non-zero for encoder and readout weights."""
        daily_log_prices = compute_daily_log_prices(fake_minute_prices)
        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * N_ASSETS
        real_sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5], depth=2,
            augment_fn=augment_fn, aug_dim=aug_dim,
        )
        y0_batch = daily_log_prices[:2]

        _, grads = eqx.filter_value_and_grad(
            lambda model: sigw1_loss_multiscale_latent(
                model, real_sigs, y0_batch,
                window_lens=[5], depth=2, aug_dim=aug_dim,
                augment_fn=augment_fn, mc_samples=4, key=key,
            )
        )(latent_sde)

        # Encoder MLP should have non-zero gradients
        enc_grads = jax.tree_util.tree_leaves(
            eqx.filter(grads.encoder, eqx.is_array)
        )
        assert any(jnp.any(g != 0) for g in enc_grads), "Encoder grads all zero"

        # Readout weight should have non-zero gradient
        assert jnp.any(grads.readout.weight != 0), "Readout grads all zero"


# ==========================================================================
# Drifting loss tests
# ==========================================================================


class TestComputeDriftField:
    """Tests for the kernel-based mean-shift drift field."""

    def test_output_shape(self):
        """V has same shape as gen: (G, D)."""
        gen = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        pos = jnp.array([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        V = compute_drift_field(gen, pos, temp=1.0)
        assert V.shape == (2, 2)

    def test_finite_output(self):
        """V is finite for reasonable inputs."""
        key = jax.random.PRNGKey(0)
        gen = jax.random.normal(key, (10, 5))
        pos = jax.random.normal(jax.random.PRNGKey(1), (20, 5))
        V = compute_drift_field(gen, pos, temp=0.5)
        assert jnp.all(jnp.isfinite(V))

    def test_vanishes_when_distributions_match(self):
        """When gen == pos, the drift field should be approximately zero."""
        key = jax.random.PRNGKey(42)
        data = jax.random.normal(key, (50, 4))
        # Use same data as both gen and pos
        V = compute_drift_field(data, data, temp=0.5)
        # Should be near zero (not exactly due to self-exclusion + finite sample)
        assert float(jnp.max(jnp.abs(V))) < 1.0, (
            f"Drift field too large when distributions match: max |V| = {float(jnp.max(jnp.abs(V)))}"
        )

    def test_drift_points_toward_data(self):
        """When gen is displaced from pos, V should point back toward pos."""
        pos = jnp.zeros((20, 2))  # Cluster at origin
        gen = jnp.ones((10, 2)) * 5.0  # Cluster far away
        V = compute_drift_field(gen, pos, temp=2.0)
        # V should have negative components (pointing from gen toward pos)
        mean_V = jnp.mean(V, axis=0)
        assert float(mean_V[0]) < 0, f"Drift should point toward data, got {float(mean_V[0])}"
        assert float(mean_V[1]) < 0, f"Drift should point toward data, got {float(mean_V[1])}"


class TestPrecomputeRealWindowSignatures:
    """Tests for individual per-window signature precomputation."""

    def test_output_shape(self, key):
        """Shape is (n_windows, n_scales * sig_dim)."""
        n_days = 30
        n_assets = 3
        daily_log = jax.random.normal(key, (n_days, n_assets)) * 0.01
        daily_log = jnp.cumsum(daily_log, axis=0)  # cumulative for realism

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        window_lens = [5, 10]
        depth = 2

        sigs = precompute_real_window_signatures(
            daily_log, window_lens, depth, augment_fn, aug_dim
        )

        max_window = 10
        n_returns = n_days - 1
        expected_n_windows = n_returns - max_window + 1

        # sig_dim for lead-lag with 3 assets, depth 2:
        # aug_dim = 6, sig_dim = 6 + 36 = 42
        sig_dim = aug_dim + aug_dim ** 2  # level 1 + level 2
        expected_total_dim = len(window_lens) * sig_dim

        assert sigs.shape == (expected_n_windows, expected_total_dim)

    def test_output_shape_with_drift_features(self, key):
        """With drift_weight > 0, each scale adds n_assets extra dims."""
        n_days = 30
        n_assets = 3
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        window_lens = [5, 10]
        depth = 2

        sigs = precompute_real_window_signatures(
            daily_log, window_lens, depth, augment_fn, aug_dim,
            drift_weight=10.0,
        )

        max_window = 10
        n_returns = n_days - 1
        expected_n_windows = n_returns - max_window + 1
        sig_dim = aug_dim + aug_dim ** 2
        # Each scale: sig_dim + n_assets
        expected_total_dim = len(window_lens) * (sig_dim + n_assets)

        assert sigs.shape == (expected_n_windows, expected_total_dim)

    def test_drift_weight_zero_matches_default(self, key):
        """drift_weight=0 should produce identical output to no drift_weight."""
        n_days = 25
        n_assets = 2
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets

        sigs_default = precompute_real_window_signatures(
            daily_log, [5], 2, augment_fn, aug_dim
        )
        sigs_zero = precompute_real_window_signatures(
            daily_log, [5], 2, augment_fn, aug_dim, drift_weight=0.0,
        )
        assert jnp.allclose(sigs_default, sigs_zero)

    def test_finite_values(self, key):
        """All signatures should be finite."""
        n_days = 25
        n_assets = 2
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        sigs = precompute_real_window_signatures(
            daily_log, [5], 2, augment_fn, aug_dim
        )
        assert jnp.all(jnp.isfinite(sigs))

    def test_finite_values_with_drift_features(self, key):
        """Drift features should also be finite."""
        n_days = 25
        n_assets = 2
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        sigs = precompute_real_window_signatures(
            daily_log, [5], 2, augment_fn, aug_dim, drift_weight=100.0,
        )
        assert jnp.all(jnp.isfinite(sigs))

    def test_consistent_with_multiscale_mean(self, key):
        """Mean of individual sigs should match compute_real_expected_signature_multiscale."""
        n_days = 40
        n_assets = 2
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        window_lens = [5]
        depth = 2

        # Individual signatures (no drift features for comparison)
        individual_sigs = precompute_real_window_signatures(
            daily_log, window_lens, depth, augment_fn, aug_dim
        )
        mean_from_individual = jnp.mean(individual_sigs, axis=0)

        # Expected signature (population mean)
        expected_sigs = compute_real_expected_signature_multiscale(
            daily_log, window_lens, depth, augment_fn, aug_dim
        )

        # Note: different number of windows (individual uses max_window constraint,
        # expected uses per-scale constraint), but for single scale they should be close
        # Actually for single window_lens, the windows are exactly the same set minus
        # the longest-window constraint. With only one scale, they're identical.
        atol = 1e-5
        assert jnp.allclose(mean_from_individual, expected_sigs[0], atol=atol), (
            f"Max diff: {float(jnp.max(jnp.abs(mean_from_individual - expected_sigs[0])))}"
        )


class TestDriftingLossLatent:
    """Tests for the per-sample drifting loss."""

    def test_finite_loss(self, key, latent_sde):
        """Loss should be finite."""
        n_days = 25
        n_assets = N_ASSETS
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        window_lens = [5]
        depth = 2

        real_sigs = precompute_real_window_signatures(
            daily_log, window_lens, depth, augment_fn, aug_dim
        )
        y0_batch = daily_log[:2]

        loss = drifting_loss_latent(
            latent_sde, real_sigs, y0_batch,
            window_lens=window_lens, depth=depth, aug_dim=aug_dim,
            augment_fn=augment_fn, key=key, temp=0.5,
        )
        assert jnp.isfinite(loss)
        assert float(loss) >= 0.0

    def test_finite_loss_with_drift_features(self, key, latent_sde):
        """Loss with drift features should be finite."""
        n_days = 25
        n_assets = N_ASSETS
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        window_lens = [5]
        depth = 2
        dw = 50.0

        real_sigs = precompute_real_window_signatures(
            daily_log, window_lens, depth, augment_fn, aug_dim,
            drift_weight=dw,
        )
        y0_batch = daily_log[:2]

        loss = drifting_loss_latent(
            latent_sde, real_sigs, y0_batch,
            window_lens=window_lens, depth=depth, aug_dim=aug_dim,
            augment_fn=augment_fn, key=key, temp=0.5,
            drift_weight=dw,
        )
        assert jnp.isfinite(loss)
        assert float(loss) >= 0.0

    def test_differentiable(self, key, latent_sde):
        """Gradients should flow through to all model components."""
        n_days = 25
        n_assets = N_ASSETS
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        window_lens = [5]
        depth = 2

        real_sigs = precompute_real_window_signatures(
            daily_log, window_lens, depth, augment_fn, aug_dim
        )
        y0_batch = daily_log[:2]

        _, grads = eqx.filter_value_and_grad(
            lambda model: drifting_loss_latent(
                model, real_sigs, y0_batch,
                window_lens=window_lens, depth=depth, aug_dim=aug_dim,
                augment_fn=augment_fn, key=key, temp=0.5,
            )
        )(latent_sde)

        # Drift network should have non-zero gradients
        drift_grads = jax.tree_util.tree_leaves(
            eqx.filter(grads.drift, eqx.is_array)
        )
        assert any(jnp.any(g != 0) for g in drift_grads), "Drift grads all zero"

        # Diffusion network should have non-zero gradients
        diff_grads = jax.tree_util.tree_leaves(
            eqx.filter(grads.diffusion, eqx.is_array)
        )
        assert any(jnp.any(g != 0) for g in diff_grads), "Diffusion grads all zero"

    def test_differentiable_with_drift_features(self, key, latent_sde):
        """Gradients should flow through drift features."""
        n_days = 25
        n_assets = N_ASSETS
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        window_lens = [5]
        depth = 2
        dw = 5.0

        real_sigs = precompute_real_window_signatures(
            daily_log, window_lens, depth, augment_fn, aug_dim,
            drift_weight=dw,
        )
        y0_batch = daily_log[:2]

        # Use large temp to ensure kernel overlap between untrained model
        # and real data. An untrained SDE produces features far from the
        # training data — with small temp, kernel values underflow to 0,
        # V vanishes, and all gradients are zero. This is a real limitation
        # of kernel methods (no signal without distribution overlap), not a
        # bug. In practice, auto-temp handles this.
        _, grads = eqx.filter_value_and_grad(
            lambda model: drifting_loss_latent(
                model, real_sigs, y0_batch,
                window_lens=window_lens, depth=depth, aug_dim=aug_dim,
                augment_fn=augment_fn, key=key, temp=10.0,
                drift_weight=dw,
            )
        )(latent_sde)

        # At least some component should have non-zero gradients
        all_grads = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert any(jnp.any(g != 0) for g in all_grads), "All grads zero"

    def test_gradient_through_encoder_and_readout(self, key, latent_sde):
        """Encoder and readout should receive gradients from drifting loss."""
        n_days = 25
        n_assets = N_ASSETS
        daily_log = jnp.cumsum(jax.random.normal(key, (n_days, n_assets)) * 0.01, axis=0)

        augment_fn = get_minimal_augmentation()
        aug_dim = 2 * n_assets
        window_lens = [5]
        depth = 2

        real_sigs = precompute_real_window_signatures(
            daily_log, window_lens, depth, augment_fn, aug_dim
        )
        y0_batch = daily_log[:2]

        _, grads = eqx.filter_value_and_grad(
            lambda model: drifting_loss_latent(
                model, real_sigs, y0_batch,
                window_lens=window_lens, depth=depth, aug_dim=aug_dim,
                augment_fn=augment_fn, key=key, temp=0.5,
            )
        )(latent_sde)

        # Encoder MLP should have non-zero gradients
        enc_grads = jax.tree_util.tree_leaves(
            eqx.filter(grads.encoder, eqx.is_array)
        )
        assert any(jnp.any(g != 0) for g in enc_grads), "Encoder grads all zero"

        # Readout weight should have non-zero gradient
        assert jnp.any(grads.readout.weight != 0), "Readout grads all zero"
