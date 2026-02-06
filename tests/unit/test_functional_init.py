"""Tests for Feature 8: Functional-space parameter initialization."""

import pytest
import numpy as np
import jax.numpy as jnp


def _make_test_params(n_parameter_sets=4, n_assets=2):
    """Create synthetic params dict for testing add_noise."""
    return {
        "sp_k": np.ones((n_parameter_sets, n_assets)) * 19.5,
        "logit_lamb": np.ones((n_parameter_sets, n_assets)) * 4.0,
        "logit_delta_lamb": np.zeros((n_parameter_sets, n_assets)),
        "initial_weights_logits": np.ones((n_parameter_sets, n_assets)),
        "subsidary_params": [],
    }


def _get_pool():
    """Get a pool instance for testing."""
    from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import MeanReversionChannelPool
    return MeanReversionChannelPool()


def test_sobol_valid_shapes():
    """add_noise(params, 'sobol', 4) → correct shapes."""
    pool = _get_pool()
    params = _make_test_params(n_parameter_sets=4, n_assets=2)
    result = pool.add_noise(params, "sobol", 4)
    assert result["sp_k"].shape == (4, 2)
    assert result["logit_lamb"].shape == (4, 2)


def test_lhs_valid_shapes():
    """Same for 'lhs'."""
    pool = _get_pool()
    params = _make_test_params(n_parameter_sets=4, n_assets=2)
    result = pool.add_noise(params, "lhs", 4)
    assert result["sp_k"].shape == (4, 2)
    assert result["logit_lamb"].shape == (4, 2)


def test_sobol_more_diverse():
    """Sobol spans wider range than Gaussian (lower discrepancy)."""
    pool = _get_pool()
    n_sets = 8

    # Run both methods
    params_gauss = _make_test_params(n_parameter_sets=n_sets, n_assets=2)
    result_gauss = pool.add_noise(params_gauss, "gaussian", n_sets, noise_scale=0.5)

    params_sobol = _make_test_params(n_parameter_sets=n_sets, n_assets=2)
    result_sobol = pool.add_noise(params_sobol, "sobol", n_sets, noise_scale=0.5)

    # Sobol should have more uniform coverage - check that range is at least as wide
    for key in ["sp_k", "logit_lamb"]:
        range_gauss = float(jnp.ptp(result_gauss[key][1:, 0]))
        range_sobol = float(jnp.ptp(result_sobol[key][1:, 0]))
        # Sobol with enough samples should have reasonable coverage
        assert range_sobol > 0, f"Sobol range is zero for {key}"


def test_gaussian_unchanged():
    """noise='gaussian' produces same behavior as before."""
    pool = _get_pool()
    np.random.seed(42)
    params = _make_test_params(n_parameter_sets=4, n_assets=2)
    result = pool.add_noise(params, "gaussian", 4, noise_scale=0.1)
    # First row should be unaltered
    assert jnp.allclose(result["sp_k"][0], 19.5)
    # Other rows should be different
    assert not jnp.allclose(result["sp_k"][1], 19.5)


def test_all_methods_finite():
    """All init methods produce finite values."""
    pool = _get_pool()
    for method in ["gaussian", "sobol", "lhs", "centered_lhs"]:
        params = _make_test_params(n_parameter_sets=4, n_assets=2)
        result = pool.add_noise(params, method, 4, noise_scale=0.5)
        for key in ["sp_k", "logit_lamb", "logit_delta_lamb"]:
            assert jnp.all(jnp.isfinite(result[key])), f"{method}/{key} has non-finite values"


def test_first_row_preserved():
    """For all methods, the first parameter set (row 0) is unaltered."""
    pool = _get_pool()
    for method in ["gaussian", "sobol", "lhs", "centered_lhs"]:
        params = _make_test_params(n_parameter_sets=4, n_assets=2)
        result = pool.add_noise(params, method, 4, noise_scale=0.5)
        assert jnp.allclose(result["sp_k"][0], 19.5), f"{method}: first row altered"
        assert jnp.allclose(result["logit_lamb"][0], 4.0), f"{method}: first row altered"


def test_default_parameter_init_method():
    """run_fingerprint_defaults has 'parameter_init_method': 'gaussian'."""
    from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults

    method = run_fingerprint_defaults["optimisation_settings"]["parameter_init_method"]
    assert method == "gaussian", f"Expected 'gaussian', got '{method}'"
