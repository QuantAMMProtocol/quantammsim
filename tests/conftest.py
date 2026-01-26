"""
Pytest configuration and shared fixtures for quantammsim tests.
"""
import os
from pathlib import Path

# Configure JAX compilation cache BEFORE importing JAX
# This speeds up repeated test runs by caching compiled XLA computations
_jax_cache_dir = Path.home() / ".cache" / "jax_compilations" / "quantammsim_tests"
_jax_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_jax_cache_dir))

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import config
from pathlib import Path


# Test data directory - use this for all tests that need historical price data
# This ensures tests use stable test data rather than production data that may change
TEST_DATA_DIR = Path(__file__).parent / "data"

# Configure JAX for testing - enable float64 for numerical precision
config.update("jax_enable_x64", True)


@pytest.fixture(scope="session", autouse=True)
def configure_jax():
    """Configure JAX settings for the test session."""
    config.update("jax_enable_x64", True)
    yield


@pytest.fixture
def rng_key():
    """Provide a reproducible JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def basic_pool_params():
    """Basic pool parameters for ECLP and similar pools."""
    return {
        "alpha": 0.25,
        "beta": 5.0,
        "phi": jnp.pi / 4,
        "lam": 2.0,
    }


@pytest.fixture
def basic_fingerprint():
    """Basic run fingerprint for simple tests."""
    return {
        "n_assets": 2,
        "bout_length": 4,
        "initial_pool_value": 1000.0,
        "arb_frequency": 1,
        "token_list": ["ETH", "USDC"],
    }


@pytest.fixture
def training_fingerprint():
    """Run fingerprint configured for training tests."""
    return {
        "tokens": ["BTC", "ETH", "USDC"],
        "rule": "momentum",
        "startDateString": "2023-03-01 00:00:00",
        "endDateString": "2023-06-01 00:00:00",
        "endTestDateString": "2023-07-01 00:00:00",
        "chunk_period": 1440,
        "bout_offset": 37800,
        "weight_interpolation_period": 1440,
        "optimisation_settings": {
            "base_lr": 0.01,
            "optimiser": "sgd",
            "decay_lr_ratio": 0.8,
            "decay_lr_plateau": 200,
            "batch_size": 4,
            "train_on_hessian_trace": False,
            "min_lr": 1e-6,
            "n_iterations": 3,
            "n_cycles": 5,
            "sample_method": "uniform",
            "n_parameter_sets": 2,
            "training_data_kind": "historic",
            "max_mc_version": 9,
            "include_flipped_training_data": False,
            "initial_random_key": 0,
            "method": "gradient_descent",
            "force_scalar": False,
        },
        "initial_memory_length": 10.0,
        "initial_memory_length_delta": 0.0,
        "initial_k_per_day": 20,
        "initial_weights_logits": 1.0,
        "initial_log_amplitude": -10.0,
        "initial_raw_width": -8.0,
        "initial_raw_exponents": 0.0,
        "subsidary_pools": [],
        "maximum_change": 0.0003,
        "return_val": "returns",
        "initial_pool_value": 1000000.0,
        "fees": 0,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "weight_interpolation_method": "linear",
        "arb_frequency": 1,
        "do_arb": True,
        "arb_quality": 1.0,
        "numeraire": None,
        "do_trades": False,
        "noise_trader_ratio": 0.0,
        "minimum_weight": 0.03,
        "max_memory_days": 365,
    }


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n_timesteps = 100
    n_assets = 3

    # Generate realistic price returns
    returns = np.random.normal(0.0001, 0.02, (n_timesteps, n_assets))

    # Convert to prices starting from 100
    prices = np.zeros((n_timesteps, n_assets))
    prices[0] = [100.0, 100.0, 100.0]

    for t in range(1, n_timesteps):
        prices[t] = prices[t - 1] * (1 + returns[t])

    return jnp.array(prices)


# Markers for categorizing tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "training: marks tests that involve model training")
    config.addinivalue_line("markers", "requires_data: marks tests that require external data files")
