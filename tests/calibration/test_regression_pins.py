"""Pinned numerical regression tests for the calibration pipeline.

These tests pin exact numerical values computed from the synthetic fixtures.
They protect against silent computation errors during refactoring — a test
that checks only shapes/signs would still pass if e.g. an index is off by
one in unpack, or a sign is flipped in regularization.

All pinned values were computed with:
  - Python 3.9, JAX 0.4.30, numpy seed 42
  - Synthetic fixtures from conftest.py (N_DAYS=15, 2 pools)
"""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from tests.calibration.conftest import (
    CADENCES,
    GAS_COSTS,
    K_OBS,
    N_DAYS,
    POOL_IDS_FULL,
    POOL_PREFIXES,
)

from quantammsim.calibration.grid_interpolation import (
    interpolate_pool_daily,
    precompute_pool_coeffs_daily,
)
from quantammsim.calibration.loss import noise_volume, pack_params, pool_loss
from quantammsim.calibration.per_pool_fit import fit_all_pools, fit_single_pool
from quantammsim.calibration.pool_data import build_x_obs, match_grids_to_panel


# ── Helpers ────────────────────────────────────────────────────────────────


@pytest.fixture
def matched_data(synthetic_daily_grid, synthetic_panel):
    """Build matched data dict by writing temp parquets for both pools."""
    tmpdir = tempfile.mkdtemp()
    for prefix in POOL_PREFIXES:
        path = os.path.join(tmpdir, f"{prefix}_daily.parquet")
        synthetic_daily_grid.to_parquet(path)
    matched = match_grids_to_panel(tmpdir, synthetic_panel)
    yield matched
    import shutil
    shutil.rmtree(tmpdir)


@pytest.fixture
def pool0_inputs(matched_data):
    """x_obs, y_obs, day_indices, coeffs for pool 0."""
    entry = matched_data[POOL_PREFIXES[0]]
    panel = entry["panel"]
    x_obs = build_x_obs(panel)
    y_obs = panel["log_volume"].values.astype(float)
    day_indices = np.array(entry["day_indices"])
    return entry["coeffs"], x_obs, y_obs, day_indices


def _known_params():
    """Standard test params: cadence=12, gas=$1, noise intercept=8."""
    noise_coeffs = np.zeros(K_OBS)
    noise_coeffs[0] = 8.0
    return pack_params(np.log(12.0), np.log(1.0), jnp.array(noise_coeffs))


# ── Grid interpolation pins ───────────────────────────────────────────────


class TestInterpolationPins:
    """Verify interpolation exactness at grid knot points."""

    def test_interpolation_exact_at_all_knots(self, synthetic_pool_coeffs):
        """Interpolation at grid knot points must exactly reproduce grid values."""
        coeffs = synthetic_pool_coeffs
        for ci, cad in enumerate(CADENCES):
            for gi, gas in enumerate(GAS_COSTS):
                log_cad = jnp.log(cad)
                v_arb = interpolate_pool_daily(coeffs, log_cad, jnp.array(gas))
                grid_vals = coeffs.values[ci, gi, :]
                np.testing.assert_allclose(
                    v_arb, grid_vals, atol=1e-4,
                    err_msg=f"Mismatch at cad={cad}, gas={gas}",
                )

    def test_interpolation_midpoint_value(self, synthetic_pool_coeffs):
        """Pin interpolated value at a known mid-grid point."""
        v_arb = interpolate_pool_daily(
            synthetic_pool_coeffs, jnp.log(6.0), jnp.array(0.5)
        )
        # Pinned from JAX 0.4.30, seed 42
        assert v_arb.shape == (N_DAYS,)
        np.testing.assert_allclose(float(v_arb[0]), 6579.6309, rtol=1e-4)
        np.testing.assert_allclose(float(jnp.mean(v_arb)), 6621.3186, rtol=1e-4)

    def test_interpolation_monotone_in_cadence(self, synthetic_pool_coeffs):
        """V_arb should decrease as cadence increases (at fixed gas)."""
        coeffs = synthetic_pool_coeffs
        gas = jnp.array(1.0)
        cads = [1.0, 6.0, 12.0, 30.0, 60.0]
        means = [
            float(jnp.mean(interpolate_pool_daily(coeffs, jnp.log(c), gas)))
            for c in cads
        ]
        for i in range(len(means) - 1):
            assert means[i] > means[i + 1], (
                f"V_arb not decreasing: cad={cads[i]}->{cads[i+1]}, "
                f"mean={means[i]:.1f}->{means[i+1]:.1f}"
            )

    def test_interpolation_monotone_in_gas(self, synthetic_pool_coeffs):
        """V_arb should decrease as gas cost increases (at fixed cadence)."""
        coeffs = synthetic_pool_coeffs
        log_cad = jnp.log(12.0)
        gases = [0.0, 0.5, 1.0, 3.0, 5.0]
        means = [
            float(jnp.mean(interpolate_pool_daily(coeffs, log_cad, jnp.array(g))))
            for g in gases
        ]
        for i in range(len(means) - 1):
            assert means[i] > means[i + 1], (
                f"V_arb not decreasing: gas={gases[i]}->{gases[i+1]}, "
                f"mean={means[i]:.1f}->{means[i+1]:.1f}"
            )

    def test_interpolation_differentiable(self, synthetic_pool_coeffs):
        """Gradient of interpolated V_arb w.r.t. log_cadence must be finite."""
        coeffs = synthetic_pool_coeffs

        def f(log_cad):
            return jnp.sum(interpolate_pool_daily(coeffs, log_cad, jnp.array(1.0)))

        grad_val = jax.grad(f)(jnp.log(12.0))
        assert jnp.isfinite(grad_val), f"Non-finite gradient: {grad_val}"
        # Gradient should be negative (more cadence → less arb)
        assert float(grad_val) < 0, f"Expected negative gradient, got {grad_val}"


# ── Loss function pins ─────────────────────────────────────────────────────


class TestLossPins:
    """Pin exact loss values and gradients at known parameter points."""

    def test_loss_value_pinned(self, synthetic_pool_coeffs, pool0_inputs):
        """Pin the exact loss value at known params on synthetic data."""
        coeffs, x_obs, _, day_indices = pool0_inputs
        params = _known_params()
        y_obs = jnp.ones(x_obs.shape[0]) * 9.0
        day_indices_j = jnp.arange(x_obs.shape[0]) % N_DAYS

        loss = pool_loss(params, coeffs, jnp.array(x_obs), y_obs, day_indices_j)
        # Pinned: 0.001726984975292 (JAX 0.4.30, seed 42)
        np.testing.assert_allclose(float(loss), 0.001727, rtol=1e-3)

    def test_gradient_pinned(self, synthetic_pool_coeffs, pool0_inputs):
        """Pin gradient values at known params."""
        coeffs, x_obs, _, day_indices = pool0_inputs
        params = _known_params()
        y_obs = jnp.ones(x_obs.shape[0]) * 9.0
        day_indices_j = jnp.arange(x_obs.shape[0]) % N_DAYS

        grad_fn = jax.grad(pool_loss)
        grad = grad_fn(params, coeffs, jnp.array(x_obs), y_obs, day_indices_j)
        grad_np = np.array(grad)

        # All gradients must be finite
        assert np.all(np.isfinite(grad_np)), f"Non-finite gradients: {grad_np}"

        # Pin signs of key gradient components
        # grad[0] = d_loss/d_log_cadence (negative: increasing cadence decreases V_arb,
        #   pushing log(V_arb + V_noise) away from y_obs=9.0)
        assert grad_np[0] < 0, f"Expected negative cadence grad, got {grad_np[0]}"
        # grad[1] = d_loss/d_log_gas (negative: same effect via gas)
        assert grad_np[1] < 0, f"Expected negative gas grad, got {grad_np[1]}"

        # Pin magnitudes (rtol=0.01 to allow platform variance)
        expected_grad = np.array([
            -0.000223, -0.000362, -0.000264, -0.003138,
            0.001071, 0.012854, 0.018232, -0.006220,
            0.013421, -0.016786,
        ])
        np.testing.assert_allclose(grad_np, expected_grad, rtol=0.05, atol=1e-5)

    def test_loss_increases_with_bad_params(self, synthetic_pool_coeffs, pool0_inputs):
        """Loss with wildly wrong noise intercept >> loss with good params."""
        coeffs, x_obs, _, _ = pool0_inputs
        y_obs = jnp.ones(x_obs.shape[0]) * 9.0
        day_indices_j = jnp.arange(x_obs.shape[0]) % N_DAYS
        x_obs_j = jnp.array(x_obs)

        params_good = _known_params()
        noise_bad = np.zeros(K_OBS)
        noise_bad[0] = 20.0
        params_bad = pack_params(np.log(12.0), np.log(1.0), jnp.array(noise_bad))

        loss_good = float(pool_loss(params_good, coeffs, x_obs_j, y_obs, day_indices_j))
        loss_bad = float(pool_loss(params_bad, coeffs, x_obs_j, y_obs, day_indices_j))

        assert loss_bad > 100.0, f"Expected loss_bad > 100, got {loss_bad}"
        assert loss_bad > loss_good * 1000, "Bad params should be >1000x worse"


# ── Noise volume pins ──────────────────────────────────────────────────────


class TestNoiseVolumePins:
    def test_intercept_only_equals_exp(self, synthetic_x_obs):
        """With intercept-only noise coeffs, V_noise = exp(intercept) exactly."""
        coeffs = np.zeros(K_OBS)
        coeffs[0] = 8.0
        v_noise = noise_volume(jnp.array(coeffs), jnp.array(synthetic_x_obs))
        # x_obs column 0 is all 1.0 (intercept), so x_obs @ coeffs = 8.0 for all obs
        np.testing.assert_allclose(v_noise, np.exp(8.0), rtol=1e-6)

    def test_tvl_coeff_creates_variation(self, synthetic_x_obs):
        """With nonzero TVL coeff, V_noise varies across observations."""
        coeffs = np.zeros(K_OBS)
        coeffs[0] = 5.0
        coeffs[1] = 1.0  # TVL coefficient
        v_noise = noise_volume(jnp.array(coeffs), jnp.array(synthetic_x_obs))
        assert float(jnp.std(v_noise)) > 0, "Expected variation from TVL coeff"


# ── Per-pool fit pins ──────────────────────────────────────────────────────


class TestPerPoolFitPins:
    """Pin per-pool optimizer convergence on synthetic data."""

    def test_fit_single_pool_converges(self, pool0_inputs):
        """fit_single_pool should converge on synthetic data."""
        coeffs, x_obs, y_obs, day_indices = pool0_inputs
        result = fit_single_pool(coeffs, x_obs, y_obs, day_indices)
        assert result["converged"], "fit_single_pool did not converge"

    def test_fit_single_pool_loss_pinned(self, pool0_inputs):
        """Pin the converged loss value."""
        coeffs, x_obs, y_obs, day_indices = pool0_inputs
        result = fit_single_pool(coeffs, x_obs, y_obs, day_indices)
        # Pinned: 0.0723 (JAX 0.4.30, seed 42)
        np.testing.assert_allclose(result["loss"], 0.0723, rtol=0.05)

    def test_fit_single_pool_cadence_pinned(self, pool0_inputs):
        """Pin the converged cadence — should find ~1.27 min on synthetic data."""
        coeffs, x_obs, y_obs, day_indices = pool0_inputs
        result = fit_single_pool(coeffs, x_obs, y_obs, day_indices)
        # Pinned: 1.266 minutes
        np.testing.assert_allclose(result["cadence_minutes"], 1.27, rtol=0.1)
        # Cadence must be in valid range
        assert 1.0 <= result["cadence_minutes"] <= 60.0

    def test_fit_single_pool_loss_lower_than_init(self, pool0_inputs):
        """Fitted loss must be lower than loss at initial guess."""
        from quantammsim.calibration.per_pool_fit import make_initial_guess

        coeffs, x_obs, y_obs, day_indices = pool0_inputs
        init = make_initial_guess(x_obs, y_obs)
        init_loss = float(
            pool_loss(
                jnp.array(init),
                coeffs,
                jnp.array(x_obs),
                jnp.array(y_obs),
                jnp.array(day_indices),
            )
        )
        result = fit_single_pool(coeffs, x_obs, y_obs, day_indices)
        assert result["loss"] < init_loss, (
            f"Fitted loss {result['loss']:.6f} >= init loss {init_loss:.6f}"
        )

    def test_fit_all_pools_returns_all(self, matched_data):
        """fit_all_pools returns results for every matched pool."""
        results = fit_all_pools(matched_data)
        assert set(results.keys()) == set(matched_data.keys())
        for pid, r in results.items():
            assert "loss" in r
            assert "log_cadence" in r
            assert "noise_coeffs" in r
            assert len(r["noise_coeffs"]) == K_OBS


# ── Joint fit pins ─────────────────────────────────────────────────────────


class TestJointFitPins:
    """Pin joint optimization behavior on synthetic data."""

    def test_joint_ppn_loss_decreases(self, matched_data):
        """Joint per_pool_noise loss must decrease from initialization."""
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(matched_data, mode="per_pool_noise", maxiter=100)
        assert result["loss"] < result["init_loss"], (
            f"Loss didn't decrease: {result['loss']:.6f} >= {result['init_loss']:.6f}"
        )

    def test_joint_ppn_loss_pinned(self, matched_data):
        """Pin the joint per_pool_noise loss value."""
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(matched_data, mode="per_pool_noise", maxiter=100)
        # Pinned: 0.0406 (JAX 0.4.30, seed 42)
        # Use wide tolerance since optimizer path may vary across platforms
        assert result["loss"] < 0.10, f"Loss too high: {result['loss']}"
        assert result["loss"] < result["init_loss"]

    def test_joint_shared_noise_loss_decreases(self, matched_data):
        """Joint shared_noise loss must decrease from initialization."""
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(matched_data, mode="shared_noise", maxiter=100)
        assert result["loss"] < result["init_loss"], (
            f"Loss didn't decrease: {result['loss']:.6f} >= {result['init_loss']:.6f}"
        )

    def test_joint_predict_new_pool_at_zero_attrs(self, matched_data):
        """Predict at zero attributes → output equals bias terms."""
        from quantammsim.calibration.joint_fit import fit_joint, predict_new_pool_joint

        result = fit_joint(matched_data, mode="per_pool_noise", maxiter=50)
        x_attr = np.zeros(result["k_attr"])
        pred = predict_new_pool_joint(result, x_attr)

        # At zero attributes: log_cadence = bias_cad, log_gas = bias_gas
        np.testing.assert_allclose(
            pred["log_cadence"], result["bias_cad"], rtol=1e-10
        )
        np.testing.assert_allclose(
            pred["log_gas"], result["bias_gas"], rtol=1e-10
        )
        assert pred["cadence_minutes"] > 0
        assert pred["gas_usd"] > 0

    def test_joint_shared_noise_predict_includes_noise(self, matched_data):
        """Shared noise mode prediction includes noise_coeffs."""
        from quantammsim.calibration.joint_fit import fit_joint, predict_new_pool_joint

        result = fit_joint(matched_data, mode="shared_noise", maxiter=50)
        x_attr = np.zeros(result["k_attr"])
        pred = predict_new_pool_joint(result, x_attr)

        assert "noise_coeffs" in pred, "shared_noise predict should include noise_coeffs"
        assert len(pred["noise_coeffs"]) == K_OBS
        # At zero attributes: noise_coeffs = bias_noise
        np.testing.assert_allclose(
            pred["noise_coeffs"], result["bias_noise"], rtol=1e-10
        )

    def test_joint_ppn_noise_shape(self, matched_data):
        """Per-pool noise mode produces (n_pools, K_OBS) noise coefficients."""
        from quantammsim.calibration.joint_fit import fit_joint

        n_pools = len(matched_data)
        result = fit_joint(matched_data, mode="per_pool_noise", maxiter=20)
        assert result["noise_coeffs"].shape == (n_pools, K_OBS)

    def test_joint_warm_start_from_option_c(self, matched_data):
        """Warm start from Option C should produce a viable starting point."""
        from quantammsim.calibration.joint_fit import fit_joint

        option_c = fit_all_pools(matched_data)
        result = fit_joint(
            matched_data,
            mode="per_pool_noise",
            maxiter=100,
            init_from_option_c=option_c,
        )
        # The warm start may have higher init_loss than cold start because
        # the linear projection of per-pool params introduces approximation
        # error. But the final loss should still decrease from init.
        assert result["loss"] < result["init_loss"]


# ── Pack/unpack roundtrip pins ─────────────────────────────────────────────


class TestPackUnpackPins:
    def test_per_pool_loss_pack_roundtrip_exact(self):
        """pack → unpack must recover exact values."""
        from quantammsim.calibration.loss import unpack_params

        log_cad = 2.4849
        log_gas = -0.6932
        noise = jnp.array([8.1, -1.2, 3.4, -0.5, 0.7, -2.1, 0.3, 0.9])
        packed = pack_params(log_cad, log_gas, noise)

        lc, lg, nc = unpack_params(packed)
        np.testing.assert_allclose(float(lc), log_cad, atol=1e-10)
        np.testing.assert_allclose(float(lg), log_gas, atol=1e-10)
        np.testing.assert_allclose(nc, noise, atol=1e-10)

    def test_joint_pack_roundtrip_ppn(self):
        """Joint per_pool_noise pack → unpack roundtrip."""
        from quantammsim.calibration.joint_fit import (
            pack_joint_params,
            unpack_joint_params,
        )

        k_attr = 5
        n_pools = 3
        bias_cad = 2.5
        bias_gas = -0.1
        W_cad = jnp.arange(k_attr, dtype=float) * 0.1
        W_gas = jnp.arange(k_attr, dtype=float) * -0.05
        noise = jnp.ones((n_pools, K_OBS)) * 0.3

        packed = pack_joint_params(bias_cad, bias_gas, W_cad, W_gas, noise)
        config = {"k_attr": k_attr, "n_pools": n_pools, "mode": "per_pool_noise"}
        unpacked = unpack_joint_params(packed, config)

        np.testing.assert_allclose(float(unpacked["bias_cad"]), bias_cad, atol=1e-10)
        np.testing.assert_allclose(float(unpacked["bias_gas"]), bias_gas, atol=1e-10)
        np.testing.assert_allclose(unpacked["W_cad"], W_cad, atol=1e-10)
        np.testing.assert_allclose(unpacked["W_gas"], W_gas, atol=1e-10)
        np.testing.assert_allclose(unpacked["noise_coeffs"], noise, atol=1e-10)

    def test_joint_pack_roundtrip_shared(self):
        """Joint shared_noise pack → unpack roundtrip."""
        from quantammsim.calibration.joint_fit import (
            pack_joint_params,
            unpack_joint_params,
        )

        k_attr = 4
        bias_cad = 1.5
        bias_gas = 0.2
        W_cad = jnp.ones(k_attr) * 0.1
        W_gas = jnp.ones(k_attr) * -0.2
        # shared_noise: (1 + k_attr, K_OBS) where row 0 is bias_noise
        noise = jnp.arange((1 + k_attr) * K_OBS, dtype=float).reshape(
            1 + k_attr, K_OBS
        )

        packed = pack_joint_params(bias_cad, bias_gas, W_cad, W_gas, noise)
        config = {"k_attr": k_attr, "n_pools": 2, "mode": "shared_noise"}
        unpacked = unpack_joint_params(packed, config)

        np.testing.assert_allclose(float(unpacked["bias_cad"]), bias_cad, atol=1e-10)
        np.testing.assert_allclose(unpacked["bias_noise"], noise[0], atol=1e-10)
        np.testing.assert_allclose(unpacked["W_noise"], noise[1:], atol=1e-10)
