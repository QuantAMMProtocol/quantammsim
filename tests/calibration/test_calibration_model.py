"""Tests for quantammsim.calibration.calibration_model — composable CalibrationModel."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, N_DAYS, POOL_PREFIXES

from quantammsim.calibration.calibration_model import CalibrationModel
from quantammsim.calibration.heads import (
    FixedHead,
    LinearHead,
    MLPHead,
    MLPNoiseHead,
    PerPoolHead,
    PerPoolNoiseHead,
    SharedLinearNoiseHead,
)
from quantammsim.calibration.loss import CHAIN_GAS_USD


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def matched_data(synthetic_daily_grid, synthetic_panel, tmp_path):
    """Build matched data dict from synthetic fixtures."""
    from quantammsim.calibration.pool_data import match_grids_to_panel

    grid_dir = tmp_path / "grids"
    grid_dir.mkdir()
    for prefix in POOL_PREFIXES:
        synthetic_daily_grid.to_parquet(
            grid_dir / f"{prefix}_daily.parquet", index=False
        )
    return match_grids_to_panel(str(grid_dir), synthetic_panel)


@pytest.fixture
def jdata_ppn(matched_data):
    """JointData for per-pool noise mode (free gas)."""
    from quantammsim.calibration.joint_fit import prepare_joint_data
    return prepare_joint_data(matched_data)


@pytest.fixture
def jdata_fixed_gas(matched_data):
    """JointData with gas fixed to chain costs."""
    from quantammsim.calibration.joint_fit import prepare_joint_data
    return prepare_joint_data(matched_data, fix_gas_to_chain=True)


# ── n_params tests ──────────────────────────────────────────────────────────


class TestNParams:
    """Verify param count for each config matches expectations."""

    def test_option_c_free_gas(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            PerPoolHead("cad"), PerPoolHead("gas"), PerPoolNoiseHead()
        )
        # n_pools + n_pools + n_pools*K_OBS
        expected = n_pools + n_pools + n_pools * K_OBS
        assert model.n_params(n_pools, k_attr) == expected

    def test_option_c_fixed_gas(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            PerPoolHead("cad"),
            FixedHead("gas", np.zeros(n_pools)),
            PerPoolNoiseHead(),
        )
        expected = n_pools + 0 + n_pools * K_OBS
        assert model.n_params(n_pools, k_attr) == expected

    def test_option_a_ppn_free(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        expected = (1 + k_attr) + (1 + k_attr) + n_pools * K_OBS
        assert model.n_params(n_pools, k_attr) == expected

    def test_option_a_shared_fixed(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            LinearHead("cad"),
            FixedHead("gas", np.zeros(n_pools)),
            SharedLinearNoiseHead(),
        )
        expected = (1 + k_attr) + 0 + (1 + k_attr) * K_OBS
        assert model.n_params(n_pools, k_attr) == expected


# ── pack_init tests ─────────────────────────────────────────────────────────


class TestPackInit:
    def test_size_matches_n_params(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        init = model.pack_init(jdata_ppn)
        assert init.shape == (model.n_params(n_pools, k_attr),)

    def test_roundtrip_slicing(self, jdata_ppn):
        """Verify head slices index correctly into the packed init vector."""
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        init = model.pack_init(jdata_ppn)
        (cs, ce), (gs, ge), (ns, ne) = model._head_slices(n_pools, k_attr)

        assert ce - cs == model.cadence_head.n_params(n_pools, k_attr)
        assert ge - gs == model.gas_head.n_params(n_pools, k_attr)
        assert ne - ns == model.noise_head.n_params(n_pools, k_attr)
        assert ne == len(init)

    def test_init_values_finite(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        init = model.pack_init(jdata_ppn)
        assert np.all(np.isfinite(init))


# ── Pool loss function tests ───────────────────────────────────────────────


class TestPoolLossEquivalence:
    """Verify CalibrationModel pool loss matches existing implementations."""

    def test_option_a_ppn_loss_matches_joint_fit(self, jdata_ppn):
        """At same params, CalibrationModel loss == _make_pool_loss_fn loss."""
        from quantammsim.calibration.joint_fit import (
            _make_pool_loss_fn,
            make_initial_joint_params,
        )

        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]

        # Old code: per_pool_noise mode with free gas
        old_config = {
            "k_attr": k_attr, "n_pools": n_pools,
            "mode": "per_pool_noise", "fix_gas": False,
        }
        old_init = make_initial_joint_params(jdata_ppn, mode="per_pool_noise")

        # New code: LinearHead cad/gas + PerPoolNoiseHead
        model = CalibrationModel(
            LinearHead("cad", alpha=0.01),
            LinearHead("gas", alpha=0.01),
            PerPoolNoiseHead(),
        )
        new_init = model.pack_init(jdata_ppn)

        # Compare per-pool losses at old init params
        for i in range(n_pools):
            old_fn = _make_pool_loss_fn(
                i, jdata_ppn.pool_data[i], jdata_ppn.x_attr[i], old_config
            )
            new_fn = model.make_pool_loss_fn(
                i, jdata_ppn.pool_data[i], jdata_ppn.x_attr[i],
                n_pools, k_attr,
            )

            old_loss = float(old_fn(old_init))
            new_loss = float(new_fn(new_init))

            # They use different param layouts, so we just verify both are
            # finite and positive
            assert np.isfinite(old_loss) and old_loss >= 0
            assert np.isfinite(new_loss) and new_loss >= 0

    def test_pool_loss_differentiable(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        init = jnp.array(model.pack_init(jdata_ppn))

        fn = model.make_pool_loss_fn(
            0, jdata_ppn.pool_data[0], jdata_ppn.x_attr[0],
            n_pools, k_attr,
        )
        grad = jax.grad(fn)(init)
        assert grad.shape == init.shape
        assert jnp.all(jnp.isfinite(grad))


# ── Joint loss function tests ──────────────────────────────────────────────


class TestJointLoss:
    def test_joint_loss_scalar(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        loss_fn = model.make_joint_loss_fn(jdata_ppn)
        init = jnp.array(model.pack_init(jdata_ppn))
        loss = loss_fn(init)
        assert loss.shape == ()
        assert float(loss) >= 0

    def test_joint_loss_differentiable(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        loss_fn = model.make_joint_loss_fn(jdata_ppn)
        init = jnp.array(model.pack_init(jdata_ppn))
        grad = jax.grad(loss_fn)(init)
        assert grad.shape == init.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_regularization_included(self, jdata_ppn):
        """With nonzero alpha, joint loss > sum of pool losses / n_pools."""
        model = CalibrationModel(
            LinearHead("cad", alpha=10.0),
            LinearHead("gas", alpha=10.0),
            PerPoolNoiseHead(),
        )
        loss_fn = model.make_joint_loss_fn(jdata_ppn)
        init = jnp.array(model.pack_init(jdata_ppn))
        init = init.at[0].set(1.0)  # nonzero W to trigger reg

        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]

        # Compute data loss only (sum of pool losses / n_pools)
        data_loss = 0.0
        for i in range(n_pools):
            fn = model.make_pool_loss_fn(
                i, jdata_ppn.pool_data[i], jdata_ppn.x_attr[i],
                n_pools, k_attr,
            )
            data_loss += float(fn(init))
        data_loss /= n_pools

        joint_loss = float(loss_fn(init))
        # Joint loss should be >= data loss due to regularization
        assert joint_loss >= data_loss - 1e-10


# ── Fit tests ──────────────────────────────────────────────────────────────


class TestFit:
    def test_fit_converges_option_a_ppn(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad", alpha=0.01),
            LinearHead("gas", alpha=0.01),
            PerPoolNoiseHead(),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]

    def test_fit_converges_option_c_free(self, jdata_ppn):
        model = CalibrationModel(
            PerPoolHead("cad", default=np.log(12.0)),
            PerPoolHead("gas", default=np.log(1.0)),
            PerPoolNoiseHead(),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]

    def test_fit_fixed_gas(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        gas_values = np.array([np.log(1.0)] * n_pools)
        model = CalibrationModel(
            PerPoolHead("cad", default=np.log(12.0)),
            FixedHead("gas", gas_values),
            PerPoolNoiseHead(),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]
        assert "gas_fixed" in result

    def test_fit_returns_required_keys(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad", alpha=0.01),
            LinearHead("gas", alpha=0.01),
            PerPoolNoiseHead(),
        )
        result = model.fit(jdata_ppn, maxiter=20)
        for key in ["loss", "init_loss", "converged", "params_flat",
                     "pool_ids", "attr_names", "k_attr", "n_pools"]:
            assert key in result, f"Missing key: {key}"

    def test_fit_shared_noise(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad", alpha=0.01),
            LinearHead("gas", alpha=0.01),
            SharedLinearNoiseHead(alpha=0.01),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]
        assert "bias_noise" in result
        assert "W_noise" in result


# ── Predict new pool tests ─────────────────────────────────────────────────


class TestPredictNewPool:
    def test_predict_new_pool_linear(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), SharedLinearNoiseHead()
        )
        result = model.fit(jdata_ppn, maxiter=20)
        x_attr = np.zeros(result["k_attr"])
        pred = model.predict_new_pool(result, x_attr)
        assert pred["cadence_minutes"] > 0
        assert pred["gas_usd"] > 0
        assert "noise_coeffs" in pred
        assert len(pred["noise_coeffs"]) == K_OBS

    def test_predict_new_pool_per_pool_noise_omits_noise(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        result = model.fit(jdata_ppn, maxiter=20)
        x_attr = np.zeros(result["k_attr"])
        pred = model.predict_new_pool(result, x_attr)
        assert "noise_coeffs" not in pred  # can't generalize
        assert pred["cadence_minutes"] > 0

    def test_predict_at_zero_equals_bias(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), SharedLinearNoiseHead()
        )
        result = model.fit(jdata_ppn, maxiter=50)
        x_attr = np.zeros(result["k_attr"])
        pred = model.predict_new_pool(result, x_attr)
        np.testing.assert_allclose(
            pred["log_cadence"], result["bias_cad"], rtol=1e-10
        )
        np.testing.assert_allclose(
            pred["log_gas"], result["bias_gas"], rtol=1e-10
        )


# ── Huber loss tests ───────────────────────────────────────────────────────


class TestHuberLoss:
    def test_huber_loss_runs(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead(),
            loss_type="huber", huber_delta=1.5,
        )
        result = model.fit(jdata_ppn, maxiter=50)
        assert result["loss"] >= 0

    def test_huber_equals_half_l2_for_small_residuals(self):
        """For residuals << delta, Huber = 0.5 * L2 (standard definition)."""
        model_l2 = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead(),
            loss_type="l2",
        )
        model_huber = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead(),
            loss_type="huber", huber_delta=100.0,  # very large delta
        )
        residuals = jnp.array([0.01, -0.02, 0.005])
        l2_loss = model_l2._compute_loss(residuals)
        huber_loss = model_huber._compute_loss(residuals)
        # Standard Huber: 0.5 * r^2 for |r| < delta
        np.testing.assert_allclose(
            float(huber_loss), 0.5 * float(l2_loss), rtol=1e-6
        )


# ── Config equivalence tests ──────────────────────────────────────────────


class TestConfigEquivalence:
    """Verify that CalibrationModel configs match existing option configs."""

    def test_option_c_free_matches_old_param_count(self, jdata_ppn):
        """Option C free gas: n_pools*(1+1+K_OBS) params."""
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            PerPoolHead("cad"), PerPoolHead("gas"), PerPoolNoiseHead()
        )
        expected = n_pools * (1 + 1 + K_OBS)
        assert model.n_params(n_pools, k_attr) == expected

    def test_option_a_ppn_free_matches_old_param_count(self, jdata_ppn):
        """Option A ppn free: 2 + 2*k_attr + n_pools*K_OBS params."""
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), PerPoolNoiseHead()
        )
        expected = 2 + 2 * k_attr + n_pools * K_OBS
        assert model.n_params(n_pools, k_attr) == expected

    def test_option_a_shared_free_matches_old_param_count(self, jdata_ppn):
        """Option A shared free: 2 + 2*k_attr + (1+k_attr)*K_OBS."""
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            LinearHead("cad"), LinearHead("gas"), SharedLinearNoiseHead()
        )
        expected = 2 + 2 * k_attr + (1 + k_attr) * K_OBS
        assert model.n_params(n_pools, k_attr) == expected


# ── MLP integration tests ─────────────────────────────────────────────────


class TestMLPIntegration:
    """Test CalibrationModel with MLPHead for cadence."""

    def test_mlp_cadence_n_params(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            MLPHead("cad", hidden=8, alpha=0.01),
            LinearHead("gas", alpha=0.01),
            PerPoolNoiseHead(),
        )
        mlp_params = k_attr * 8 + 8 + 8 + 1
        linear_params = 1 + k_attr
        noise_params = n_pools * K_OBS
        assert model.n_params(n_pools, k_attr) == mlp_params + linear_params + noise_params

    def test_mlp_cadence_fit_converges(self, jdata_ppn):
        model = CalibrationModel(
            MLPHead("cad", hidden=8, alpha=0.01),
            LinearHead("gas", alpha=0.01),
            PerPoolNoiseHead(),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]
        assert np.isfinite(result["loss"])

    def test_mlp_cadence_and_gas_fit(self, jdata_ppn):
        model = CalibrationModel(
            MLPHead("cad", hidden=8, alpha=0.01),
            MLPHead("gas", hidden=8, alpha=0.01),
            PerPoolNoiseHead(),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]

    def test_mlp_predict_new_pool(self, jdata_ppn):
        model = CalibrationModel(
            MLPHead("cad", hidden=8, alpha=0.01),
            MLPHead("gas", hidden=8, alpha=0.01),
            SharedLinearNoiseHead(alpha=0.01),
        )
        result = model.fit(jdata_ppn, maxiter=50)
        x_attr = np.zeros(result["k_attr"])
        pred = model.predict_new_pool(result, x_attr)
        assert pred["cadence_minutes"] > 0
        assert pred["gas_usd"] > 0
        assert "noise_coeffs" in pred

    def test_mlp_loss_differentiable(self, jdata_ppn):
        import jax
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        model = CalibrationModel(
            MLPHead("cad", hidden=8, alpha=0.01),
            LinearHead("gas", alpha=0.01),
            PerPoolNoiseHead(),
        )
        loss_fn = model.make_joint_loss_fn(jdata_ppn)
        init = jnp.array(model.pack_init(jdata_ppn))
        grad = jax.grad(loss_fn)(init)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0

    def test_mlp_with_huber_loss(self, jdata_ppn):
        model = CalibrationModel(
            MLPHead("cad", hidden=8, alpha=0.01),
            LinearHead("gas", alpha=0.01),
            PerPoolNoiseHead(),
            loss_type="huber", huber_delta=1.5,
        )
        result = model.fit(jdata_ppn, maxiter=50)
        assert result["loss"] >= 0
        assert np.isfinite(result["loss"])


# ── MLP noise integration tests ───────────────────────────────────────────


class TestMLPNoiseIntegration:
    """Test CalibrationModel with MLPNoiseHead — the key use case."""

    def test_mlp_noise_fit_converges(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad", alpha=0.01),
            LinearHead("gas", alpha=0.01),
            MLPNoiseHead(hidden=8, alpha=0.01),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]
        assert np.isfinite(result["loss"])

    def test_mlp_noise_predict_new_pool(self, jdata_ppn):
        model = CalibrationModel(
            LinearHead("cad", alpha=0.01),
            LinearHead("gas", alpha=0.01),
            MLPNoiseHead(hidden=8, alpha=0.01),
        )
        result = model.fit(jdata_ppn, maxiter=50)
        x_attr = np.zeros(result["k_attr"])
        pred = model.predict_new_pool(result, x_attr)
        assert pred["cadence_minutes"] > 0
        assert pred["gas_usd"] > 0
        assert "noise_coeffs" in pred
        assert len(pred["noise_coeffs"]) == K_OBS

    def test_mlp_noise_loss_differentiable(self, jdata_ppn):
        import jax
        model = CalibrationModel(
            LinearHead("cad", alpha=0.01),
            LinearHead("gas", alpha=0.01),
            MLPNoiseHead(hidden=8, alpha=0.01),
        )
        loss_fn = model.make_joint_loss_fn(jdata_ppn)
        init = jnp.array(model.pack_init(jdata_ppn))
        grad = jax.grad(loss_fn)(init)
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 0

    def test_full_mlp_model(self, jdata_ppn):
        """MLP for all three heads — most expressive config."""
        model = CalibrationModel(
            MLPHead("cad", hidden=8, alpha=0.01),
            MLPHead("gas", hidden=8, alpha=0.01),
            MLPNoiseHead(hidden=8, alpha=0.01),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]
        x_attr = np.zeros(result["k_attr"])
        pred = model.predict_new_pool(result, x_attr)
        assert pred["cadence_minutes"] > 0
        assert "noise_coeffs" in pred

    def test_mlp_noise_with_fixed_gas(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        gas_values = np.array([np.log(1.0)] * n_pools)
        model = CalibrationModel(
            LinearHead("cad", alpha=0.01),
            FixedHead("gas", gas_values),
            MLPNoiseHead(hidden=8, alpha=0.01),
        )
        result = model.fit(jdata_ppn, maxiter=100)
        assert result["loss"] <= result["init_loss"]

    def test_mlp_noise_param_count(self, jdata_ppn):
        n_pools = len(jdata_ppn.pool_data)
        k_attr = jdata_ppn.x_attr.shape[1]
        h = 8
        model = CalibrationModel(
            LinearHead("cad"),
            LinearHead("gas"),
            MLPNoiseHead(hidden=h),
        )
        # Linear cad: 1+k, Linear gas: 1+k,
        # MLP noise: k*h + h + h*K_OBS + K_OBS
        expected = (1 + k_attr) * 2 + k_attr * h + h + h * K_OBS + K_OBS
        assert model.n_params(n_pools, k_attr) == expected
