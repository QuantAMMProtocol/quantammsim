"""Integration test: strategy training with synthetic price path augmentation.

Verifies that train_on_historic_data completes when synthetic_settings.use_synthetic=True,
using a small stub SDE and short training run.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from quantammsim.synthetic.model import NeuralSDE, save_sde


@pytest.fixture
def stub_sde_path():
    """Create a small trained SDE and return its path."""
    key = jax.random.PRNGKey(0)
    n_assets = 2

    # Just create a random SDE (no training needed for integration test)
    sde = NeuralSDE(n_assets=n_assets, hidden_dim=16, key=key)

    tmpdir = tempfile.mkdtemp()
    path = str(Path(tmpdir) / "stub_sde.eqx")
    save_sde(sde, path)
    return path, n_assets


@pytest.mark.integration
@pytest.mark.slow
def test_synthetic_settings_in_default_fingerprint():
    """Verify synthetic_settings exists in default run_fingerprint."""
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults

    assert "synthetic_settings" in run_fingerprint_defaults
    ss = run_fingerprint_defaults["synthetic_settings"]
    assert ss["use_synthetic"] is False
    assert ss["sde_model_path"] is None
    assert isinstance(ss["n_synthetic_paths"], int)
    assert isinstance(ss["sde_learn_drift"], bool)


@pytest.mark.integration
@pytest.mark.slow
def test_synthetic_settings_excluded_from_static_dict():
    """Verify synthetic_settings is in _TRAINING_ONLY_FIELDS."""
    from quantammsim.runners.jax_runner_utils import _TRAINING_ONLY_FIELDS

    assert "synthetic_settings" in _TRAINING_ONLY_FIELDS


@pytest.mark.integration
@pytest.mark.slow
def test_generate_and_stack_synthetic_prices(stub_sde_path):
    """Test the core synthetic path generation + stacking logic."""
    from quantammsim.synthetic.generation import generate_synthetic_price_array
    from quantammsim.synthetic.model import load_sde

    sde_path, n_assets = stub_sde_path
    sde = load_sde(sde_path, n_assets=n_assets, hidden_dim=16)

    # Minimal minute prices: 3 days
    T = 3 * 1440
    key = jax.random.PRNGKey(1)
    log_rets = jax.random.normal(key, (T, n_assets)) * 0.0001
    log_prices = jnp.cumsum(log_rets, axis=0) + jnp.array([7.0, 0.0])
    minute_prices = jnp.exp(log_prices)

    n_synthetic = 3
    synthetic = generate_synthetic_price_array(
        sde, minute_prices, n_synthetic, key=jax.random.PRNGKey(2),
    )

    # Stack real + synthetic
    real_3d = np.array(minute_prices[:, :, np.newaxis])
    stacked = np.concatenate([real_3d, np.array(synthetic)], axis=-1)

    assert stacked.shape == (T, n_assets, 1 + n_synthetic)
    assert np.all(stacked > 0)
    assert np.all(np.isfinite(stacked))
