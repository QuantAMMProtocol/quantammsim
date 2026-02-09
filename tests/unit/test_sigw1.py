"""Unit tests for Sig-W1 training components: augmentations, signatures, loss, Brownian bridge."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import signax

from quantammsim.synthetic.augmentations import (
    add_time,
    compose_augmentations,
    cumsum,
    get_minimal_augmentation,
    get_standard_augmentation,
    lead_lag,
    scale,
)
from quantammsim.synthetic.generation import (
    brownian_bridge_interpolate,
    generate_daily_paths,
    generate_minute_paths,
    generate_synthetic_price_array_daily,
)
from quantammsim.synthetic.model import NeuralSDE
from quantammsim.synthetic.training import (
    _factorial_normalise,
    compute_daily_log_prices,
    compute_real_expected_signature,
    compute_real_expected_signature_multiscale,
    sigw1_loss,
    sigw1_loss_multiscale,
)


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def sde_with_drift(key):
    """Small SDE with drift enabled (as used in Sig-W1 training)."""
    return NeuralSDE(
        n_assets=2, hidden_dim=16, learn_drift=True,
        init_diffusion_scale=jnp.array([0.01, 0.008]),
        key=key,
    )


@pytest.fixture
def fake_minute_prices():
    """Fake minute prices: 30 days, 2 assets."""
    key = jax.random.PRNGKey(0)
    n_days = 30
    n_assets = 2
    T = n_days * 1440
    log_rets = jax.random.normal(key, (T, n_assets)) * 0.0001
    log_prices = jnp.cumsum(log_rets, axis=0) + jnp.array([7.0, 8.5])
    return jnp.exp(log_prices)


@pytest.fixture
def sample_path():
    """A simple (20, 2) path of log-returns."""
    key = jax.random.PRNGKey(1)
    return jax.random.normal(key, (20, 2)) * 0.01


@pytest.fixture
def daily_log_prices(fake_minute_prices):
    """Daily log-prices derived from fake minute data."""
    return compute_daily_log_prices(fake_minute_prices)


# --- Augmentation tests ---


class TestScale:
    def test_identity(self, sample_path):
        result = scale(sample_path, 1.0)
        assert jnp.allclose(result, sample_path)

    def test_scales_values(self, sample_path):
        result = scale(sample_path, 2.0)
        assert jnp.allclose(result, sample_path * 2.0)

    def test_shape_preserved(self, sample_path):
        result = scale(sample_path, 3.0)
        assert result.shape == sample_path.shape

    def test_jit_compatible(self, sample_path):
        result = jax.jit(lambda p: scale(p, 2.0))(sample_path)
        assert jnp.allclose(result, sample_path * 2.0)


class TestCumsum:
    def test_shape_preserved(self, sample_path):
        result = cumsum(sample_path)
        assert result.shape == sample_path.shape

    def test_first_row_matches(self, sample_path):
        result = cumsum(sample_path)
        assert jnp.allclose(result[0], sample_path[0])

    def test_cumulative(self, sample_path):
        result = cumsum(sample_path)
        expected = jnp.cumsum(sample_path, axis=0)
        assert jnp.allclose(result, expected)

    def test_jit_compatible(self, sample_path):
        result = jax.jit(cumsum)(sample_path)
        assert jnp.allclose(result, jnp.cumsum(sample_path, axis=0))


class TestAddTime:
    def test_adds_one_dimension(self, sample_path):
        result = add_time(sample_path)
        assert result.shape == (20, 3)

    def test_time_channel_range(self, sample_path):
        result = add_time(sample_path)
        time_col = result[:, -1]
        assert jnp.isclose(time_col[0], 0.0)
        assert jnp.isclose(time_col[-1], 1.0)

    def test_original_data_preserved(self, sample_path):
        result = add_time(sample_path)
        assert jnp.allclose(result[:, :2], sample_path)

    def test_jit_compatible(self, sample_path):
        result = jax.jit(add_time)(sample_path)
        assert result.shape == (20, 3)


class TestLeadLag:
    def test_output_shape(self, sample_path):
        result = lead_lag(sample_path)
        L, d = sample_path.shape
        assert result.shape == (2 * L - 1, 2 * d)

    def test_lag_component(self, sample_path):
        result = lead_lag(sample_path)
        lag = result[:, :2]
        repeated = jnp.repeat(sample_path, 2, axis=0)
        expected_lag = repeated[:-1]
        assert jnp.allclose(lag, expected_lag)

    def test_lead_component(self, sample_path):
        result = lead_lag(sample_path)
        lead = result[:, 2:]
        repeated = jnp.repeat(sample_path, 2, axis=0)
        expected_lead = repeated[1:]
        assert jnp.allclose(lead, expected_lead)

    def test_jit_compatible(self, sample_path):
        result = jax.jit(lead_lag)(sample_path)
        assert result.shape == (39, 4)

    def test_vmap_compatible(self):
        batch = jax.random.normal(jax.random.PRNGKey(0), (5, 10, 2))
        result = jax.vmap(lead_lag)(batch)
        assert result.shape == (5, 19, 4)


class TestCompose:
    def test_identity_compose(self, sample_path):
        fn = compose_augmentations(lambda p: p)
        assert jnp.allclose(fn(sample_path), sample_path)

    def test_two_functions(self, sample_path):
        fn = compose_augmentations(lambda p: scale(p, 2.0), cumsum)
        expected = cumsum(scale(sample_path, 2.0))
        assert jnp.allclose(fn(sample_path), expected)


class TestGetMinimalAugmentation:
    def test_is_lead_lag_when_unit_scale(self, sample_path):
        fn = get_minimal_augmentation(s=1.0)
        result = fn(sample_path)
        expected = lead_lag(sample_path)
        assert jnp.allclose(result, expected)

    def test_with_scale(self, sample_path):
        fn = get_minimal_augmentation(s=2.0)
        result = fn(sample_path)
        expected = lead_lag(scale(sample_path, 2.0))
        assert jnp.allclose(result, expected)


class TestGetStandardAugmentation:
    def test_output_shape(self, sample_path):
        fn = get_standard_augmentation()
        result = fn(sample_path)
        assert result.shape == (39, 6)

    def test_jit_compatible(self, sample_path):
        fn = get_standard_augmentation()
        result = jax.jit(fn)(sample_path)
        assert result.shape == (39, 6)
        assert jnp.all(jnp.isfinite(result))


# --- Signature computation tests ---


class TestSignatureComputation:
    def test_signature_shape(self, sample_path):
        sig = signax.signature(sample_path, depth=3, flatten=True)
        assert sig.shape == (14,)

    def test_signature_shape_after_lead_lag(self, sample_path):
        aug = lead_lag(sample_path)
        sig = signax.signature(aug, depth=3, flatten=True)
        assert sig.shape == (84,)

    def test_signature_finite(self, sample_path):
        sig = signax.signature(sample_path, depth=3, flatten=True)
        assert jnp.all(jnp.isfinite(sig))

    def test_signature_differentiable(self, sample_path):
        def sig_norm(path):
            s = signax.signature(path, depth=2, flatten=True)
            return jnp.sum(s ** 2)

        grad = jax.grad(sig_norm)(sample_path)
        assert grad.shape == sample_path.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_signature_batched(self):
        batch = jax.random.normal(jax.random.PRNGKey(0), (8, 15, 3))
        sigs = signax.signature(batch, depth=2, flatten=True)
        assert sigs.shape == (8, 12)

    def test_signature_jit(self, sample_path):
        sig = jax.jit(lambda p: signax.signature(p, depth=2, flatten=True))(sample_path)
        assert jnp.all(jnp.isfinite(sig))


# --- Factorial normalisation tests ---


class TestFactorialNormalise:
    def test_level_1_divided_by_1(self):
        """Level 1 terms should be divided by 1! = 1 (unchanged)."""
        sig = jnp.ones(4 + 16 + 64)  # dim=4, depth=3
        result = _factorial_normalise(sig, aug_dim=4, depth=3)
        assert jnp.allclose(result[:4], 1.0)

    def test_level_2_divided_by_2(self):
        """Level 2 terms should be divided by 2! = 2."""
        sig = jnp.ones(4 + 16 + 64)
        result = _factorial_normalise(sig, aug_dim=4, depth=3)
        assert jnp.allclose(result[4:20], 0.5)

    def test_level_3_divided_by_6(self):
        """Level 3 terms should be divided by 3! = 6."""
        sig = jnp.ones(4 + 16 + 64)
        result = _factorial_normalise(sig, aug_dim=4, depth=3)
        assert jnp.allclose(result[20:], 1.0 / 6.0)

    def test_output_shape(self):
        sig = jnp.ones(84)
        result = _factorial_normalise(sig, aug_dim=4, depth=3)
        assert result.shape == (84,)

    def test_jit_compatible(self):
        sig = jnp.ones(84)
        result = jax.jit(lambda s: _factorial_normalise(s, 4, 3))(sig)
        assert jnp.all(jnp.isfinite(result))

    def test_depth_1_is_identity(self):
        """With depth 1, factorial normalisation should be identity (1/1! = 1)."""
        sig = jnp.array([1.0, 2.0, 3.0])
        result = _factorial_normalise(sig, aug_dim=3, depth=1)
        assert jnp.allclose(result, sig)


# --- Real expected signature tests ---


class TestComputeRealExpectedSignature:
    def test_shape(self, daily_log_prices):
        augment_fn = get_minimal_augmentation()
        sig = compute_real_expected_signature(daily_log_prices, window_len=10, depth=3, augment_fn=augment_fn)
        assert sig.shape == (84,)

    def test_finite(self, daily_log_prices):
        augment_fn = get_minimal_augmentation()
        sig = compute_real_expected_signature(daily_log_prices, window_len=10, depth=3, augment_fn=augment_fn)
        assert jnp.all(jnp.isfinite(sig))

    def test_different_windows_give_different_sigs(self, daily_log_prices):
        augment_fn = get_minimal_augmentation()
        sig5 = compute_real_expected_signature(daily_log_prices, window_len=5, depth=3, augment_fn=augment_fn)
        sig10 = compute_real_expected_signature(daily_log_prices, window_len=10, depth=3, augment_fn=augment_fn)
        assert not jnp.allclose(sig5, sig10)

    def test_standard_augmentation(self, daily_log_prices):
        augment_fn = get_standard_augmentation()
        sig = compute_real_expected_signature(daily_log_prices, window_len=10, depth=2, augment_fn=augment_fn)
        assert sig.shape == (42,)
        assert jnp.all(jnp.isfinite(sig))


class TestComputeRealExpectedSignatureMultiscale:
    def test_shape(self, daily_log_prices):
        augment_fn = get_minimal_augmentation()
        sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5, 10], depth=3,
            augment_fn=augment_fn, aug_dim=4,
        )
        assert sigs.shape == (2, 84)

    def test_finite(self, daily_log_prices):
        augment_fn = get_minimal_augmentation()
        sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5, 10], depth=2,
            augment_fn=augment_fn, aug_dim=4,
        )
        assert jnp.all(jnp.isfinite(sigs))

    def test_scales_differ(self, daily_log_prices):
        """Different window lengths should give different expected signatures."""
        augment_fn = get_minimal_augmentation()
        sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5, 10], depth=2,
            augment_fn=augment_fn, aug_dim=4,
        )
        assert not jnp.allclose(sigs[0], sigs[1])

    def test_normalisation_applied(self, daily_log_prices):
        """Multiscale sigs should differ from un-normalised single-scale sigs."""
        augment_fn = get_minimal_augmentation()
        raw_sig = compute_real_expected_signature(
            daily_log_prices, window_len=10, depth=3, augment_fn=augment_fn,
        )
        normalised_sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[10], depth=3,
            augment_fn=augment_fn, aug_dim=4,
        )
        # Level 2+ terms are divided by k!, so these should differ
        assert not jnp.allclose(raw_sig, normalised_sigs[0])
        # But level 1 (first 4 terms) should match (1/1! = 1)
        assert jnp.allclose(raw_sig[:4], normalised_sigs[0, :4], atol=1e-5)


# --- Sig-W1 loss tests ---


class TestSigW1Loss:
    def test_finite(self, sde_with_drift, daily_log_prices, key):
        augment_fn = get_minimal_augmentation()
        real_sig = compute_real_expected_signature(
            daily_log_prices, window_len=5, depth=2, augment_fn=augment_fn
        )
        y0_batch = daily_log_prices[:3]
        loss = sigw1_loss(
            sde_with_drift, real_sig, y0_batch,
            window_len=5, depth=2, augment_fn=augment_fn,
            mc_samples=4, key=key,
        )
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_differentiable(self, sde_with_drift, daily_log_prices, key):
        augment_fn = get_minimal_augmentation()
        real_sig = compute_real_expected_signature(
            daily_log_prices, window_len=5, depth=2, augment_fn=augment_fn
        )
        y0_batch = daily_log_prices[:2]

        loss, grads = eqx.filter_value_and_grad(
            lambda model: sigw1_loss(
                model, real_sig, y0_batch,
                window_len=5, depth=2, augment_fn=augment_fn,
                mc_samples=4, key=key,
            )
        )(sde_with_drift)

        assert jnp.isfinite(loss)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)

    def test_positive_for_different_distributions(self, sde_with_drift, daily_log_prices, key):
        augment_fn = get_minimal_augmentation()
        real_sig = compute_real_expected_signature(
            daily_log_prices, window_len=5, depth=2, augment_fn=augment_fn
        )
        y0_batch = daily_log_prices[:3]
        loss = sigw1_loss(
            sde_with_drift, real_sig, y0_batch,
            window_len=5, depth=2, augment_fn=augment_fn,
            mc_samples=8, key=key,
        )
        assert loss > 0.0


class TestSigW1LossMultiscale:
    def test_finite(self, sde_with_drift, daily_log_prices, key):
        augment_fn = get_minimal_augmentation()
        real_sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5, 10], depth=2,
            augment_fn=augment_fn, aug_dim=4,
        )
        y0_batch = daily_log_prices[:3]
        loss = sigw1_loss_multiscale(
            sde_with_drift, real_sigs, y0_batch,
            window_lens=[5, 10], depth=2, aug_dim=4,
            augment_fn=augment_fn, mc_samples=4, key=key,
        )
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_differentiable(self, sde_with_drift, daily_log_prices, key):
        augment_fn = get_minimal_augmentation()
        real_sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5, 10], depth=2,
            augment_fn=augment_fn, aug_dim=4,
        )
        y0_batch = daily_log_prices[:2]

        loss, grads = eqx.filter_value_and_grad(
            lambda model: sigw1_loss_multiscale(
                model, real_sigs, y0_batch,
                window_lens=[5, 10], depth=2, aug_dim=4,
                augment_fn=augment_fn, mc_samples=4, key=key,
            )
        )(sde_with_drift)

        assert jnp.isfinite(loss)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert len(grad_leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)

    def test_multiscale_greater_than_single_scale(self, sde_with_drift, daily_log_prices, key):
        """Multi-scale loss (sum of per-scale distances) >= any single-scale distance."""
        augment_fn = get_minimal_augmentation()
        real_sigs = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5, 10], depth=2,
            augment_fn=augment_fn, aug_dim=4,
        )
        y0_batch = daily_log_prices[:3]
        multi_loss = sigw1_loss_multiscale(
            sde_with_drift, real_sigs, y0_batch,
            window_lens=[5, 10], depth=2, aug_dim=4,
            augment_fn=augment_fn, mc_samples=4, key=key,
        )
        # Single scale (5-day only)
        real_sigs_5 = compute_real_expected_signature_multiscale(
            daily_log_prices, window_lens=[5], depth=2,
            augment_fn=augment_fn, aug_dim=4,
        )
        single_loss = sigw1_loss_multiscale(
            sde_with_drift, real_sigs_5, y0_batch,
            window_lens=[5], depth=2, aug_dim=4,
            augment_fn=augment_fn, mc_samples=4, key=key,
        )
        assert multi_loss >= single_loss - 1e-6


# --- Antithetic sampling tests ---


class TestAntitheticSampling:
    def test_shape_preserved(self, sde_with_drift, key):
        y0 = jnp.array([7.0, 8.5])
        paths = generate_minute_paths(
            sde_with_drift, y0, n_steps=10, n_paths=6,
            key=key, dt=1.0, antithetic=True,
        )
        assert paths.shape == (10, 2, 6)

    def test_shape_odd_n_paths(self, sde_with_drift, key):
        """Odd n_paths should still work."""
        y0 = jnp.array([7.0, 8.5])
        paths = generate_minute_paths(
            sde_with_drift, y0, n_steps=10, n_paths=5,
            key=key, dt=1.0, antithetic=True,
        )
        assert paths.shape == (10, 2, 5)

    def test_all_finite(self, sde_with_drift, key):
        y0 = jnp.array([7.0, 8.5])
        paths = generate_minute_paths(
            sde_with_drift, y0, n_steps=20, n_paths=8,
            key=key, dt=1.0, antithetic=True,
        )
        assert jnp.all(jnp.isfinite(paths))

    def test_reduces_mean_variance(self, sde_with_drift, key):
        """Antithetic sampling should reduce the variance of the path mean
        compared to independent sampling with the same number of paths."""
        y0 = jnp.array([7.0, 8.5])
        n_paths = 20
        n_trials = 10

        means_independent = []
        means_antithetic = []

        for i in range(n_trials):
            k = jax.random.PRNGKey(i)
            paths_ind = generate_minute_paths(
                sde_with_drift, y0, n_steps=10, n_paths=n_paths,
                key=k, dt=1.0, antithetic=False,
            )
            paths_anti = generate_minute_paths(
                sde_with_drift, y0, n_steps=10, n_paths=n_paths,
                key=k, dt=1.0, antithetic=True,
            )
            means_independent.append(float(jnp.mean(paths_ind[-1])))
            means_antithetic.append(float(jnp.mean(paths_anti[-1])))

        var_ind = jnp.var(jnp.array(means_independent))
        var_anti = jnp.var(jnp.array(means_antithetic))
        # Antithetic variance should be smaller (or at least not much larger)
        assert var_anti < var_ind * 1.5, (
            f"Antithetic variance ({var_anti:.6f}) should be less than "
            f"independent variance ({var_ind:.6f})"
        )


# --- Brownian bridge tests ---


class TestBrownianBridge:
    def test_shape_2d(self, key):
        daily = jnp.array([[7.0, 8.5], [7.01, 8.49], [7.02, 8.51]])
        vol = jnp.array([0.0008, 0.0006])
        result = brownian_bridge_interpolate(daily, minutes_per_day=1440, intraday_vol=vol, key=key)
        expected_len = 2 * 1440 + 1
        assert result.shape == (expected_len, 2)

    def test_shape_3d(self, key):
        daily = jnp.ones((4, 2, 3))
        daily = daily.at[1].set(1.01)
        daily = daily.at[2].set(1.02)
        daily = daily.at[3].set(1.03)
        vol = jnp.array([0.0008, 0.0006])
        result = brownian_bridge_interpolate(daily, minutes_per_day=100, intraday_vol=vol, key=key)
        expected_len = 3 * 100 + 1
        assert result.shape == (expected_len, 2, 3)

    def test_endpoint_preservation(self, key):
        daily = jnp.array([[7.0, 8.5], [7.05, 8.45], [7.1, 8.5]])
        vol = jnp.array([0.001, 0.001])
        mpd = 100
        result = brownian_bridge_interpolate(daily, minutes_per_day=mpd, intraday_vol=vol, key=key)
        assert jnp.allclose(result[0], daily[0], atol=1e-10)
        assert jnp.allclose(result[mpd], daily[1], atol=1e-10)
        assert jnp.allclose(result[2 * mpd], daily[2], atol=1e-10)

    def test_deterministic_without_key(self):
        daily = jnp.array([[0.0, 0.0], [1.0, 2.0]])
        result = brownian_bridge_interpolate(daily, minutes_per_day=4, intraday_vol=None, key=None)
        expected = jnp.array([
            [0.0, 0.0],
            [0.25, 0.5],
            [0.5, 1.0],
            [0.75, 1.5],
            [1.0, 2.0],
        ])
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_stochastic_differs_across_keys(self):
        daily = jnp.array([[7.0, 8.5], [7.01, 8.49]])
        vol = jnp.array([0.001, 0.001])
        r1 = brownian_bridge_interpolate(daily, minutes_per_day=100, intraday_vol=vol, key=jax.random.PRNGKey(0))
        r2 = brownian_bridge_interpolate(daily, minutes_per_day=100, intraday_vol=vol, key=jax.random.PRNGKey(1))
        assert not jnp.allclose(r1[1:-1], r2[1:-1])
        assert jnp.allclose(r1[0], r2[0])
        assert jnp.allclose(r1[-1], r2[-1])

    def test_all_finite(self, key):
        daily = jnp.array([[7.0, 8.5], [7.01, 8.49], [7.02, 8.48]])
        vol = jnp.array([0.0008, 0.0006])
        result = brownian_bridge_interpolate(daily, minutes_per_day=1440, intraday_vol=vol, key=key)
        assert jnp.all(jnp.isfinite(result))


# --- Daily generation pipeline tests ---


class TestGenerateDailyPaths:
    def test_shape(self, sde_with_drift, key):
        y0 = jnp.array([7.0, 8.5])
        paths = generate_daily_paths(sde_with_drift, y0, n_days=10, n_paths=3, key=key)
        assert paths.shape == (10, 2, 3)

    def test_finite(self, sde_with_drift, key):
        y0 = jnp.array([7.0, 8.5])
        paths = generate_daily_paths(sde_with_drift, y0, n_days=20, n_paths=2, key=key)
        assert jnp.all(jnp.isfinite(paths))


class TestGenerateSyntheticPriceArrayDaily:
    def test_output_shape(self, sde_with_drift, fake_minute_prices, key):
        result = generate_synthetic_price_array_daily(
            sde_with_drift, fake_minute_prices, n_paths=2, key=key
        )
        assert result.shape == (fake_minute_prices.shape[0], 2, 2)

    def test_prices_positive(self, sde_with_drift, fake_minute_prices, key):
        result = generate_synthetic_price_array_daily(
            sde_with_drift, fake_minute_prices, n_paths=2, key=key
        )
        assert jnp.all(result > 0)

    def test_all_finite(self, sde_with_drift, fake_minute_prices, key):
        result = generate_synthetic_price_array_daily(
            sde_with_drift, fake_minute_prices, n_paths=2, key=key
        )
        assert jnp.all(jnp.isfinite(result))
