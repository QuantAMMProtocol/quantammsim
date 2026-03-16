"""Tests for quantammsim.calibration.joint_fit — joint end-to-end optimization (Option A)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, POOL_IDS_FULL, POOL_PREFIXES


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


class TestPrepareJointData:
    """Test prepare_joint_data: build batched arrays for joint optimization."""

    def test_returns_expected_structure(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data)
        assert hasattr(jdata, "pool_data")
        assert hasattr(jdata, "x_attr")
        assert hasattr(jdata, "pool_ids")

    def test_pool_count(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data)
        assert len(jdata.pool_data) == len(matched_data)

    def test_pool_data_has_jax_arrays(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data)
        for pd in jdata.pool_data:
            assert isinstance(pd["x_obs"], jnp.ndarray)
            assert isinstance(pd["y_obs"], jnp.ndarray)
            assert isinstance(pd["day_indices"], jnp.ndarray)

    def test_x_attr_shape(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data)
        n_pools = len(matched_data)
        assert jdata.x_attr.shape[0] == n_pools
        assert jdata.x_attr.shape[1] > 0  # K_attr


class TestJointLoss:
    """Test joint_loss: end-to-end loss over all pools."""

    def _make_loss_fn(self, matched_data):
        from quantammsim.calibration.joint_fit import (
            make_joint_loss_fn,
            make_initial_joint_params,
            prepare_joint_data,
        )

        jdata = prepare_joint_data(matched_data)
        init = make_initial_joint_params(jdata, mode="per_pool_noise")
        loss_fn = make_joint_loss_fn(jdata, mode="per_pool_noise")
        return loss_fn, init, jdata

    def test_loss_scalar(self, matched_data):
        loss_fn, init, _ = self._make_loss_fn(matched_data)
        loss = loss_fn(init)
        assert loss.shape == ()

    def test_loss_positive(self, matched_data):
        loss_fn, init, _ = self._make_loss_fn(matched_data)
        loss = loss_fn(init)
        assert float(loss) >= 0

    def test_loss_differentiable(self, matched_data):
        loss_fn, init, _ = self._make_loss_fn(matched_data)
        grad = jax.grad(loss_fn)(init)
        assert grad.shape == init.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_loss_grad_nonzero(self, matched_data):
        loss_fn, init, _ = self._make_loss_fn(matched_data)
        grad = jax.grad(loss_fn)(init)
        assert float(jnp.sum(jnp.abs(grad))) > 0

    def test_shared_cadence_gas_affects_all_pools(self, matched_data):
        """Changing W_cad should affect losses from all pools."""
        from quantammsim.calibration.joint_fit import (
            make_joint_loss_fn,
            make_initial_joint_params,
            prepare_joint_data,
            unpack_joint_params,
        )

        jdata = prepare_joint_data(matched_data)
        init = make_initial_joint_params(jdata, mode="per_pool_noise")
        loss_fn = make_joint_loss_fn(jdata, mode="per_pool_noise")

        # Gradient w.r.t. W_cad should be nonzero
        grad = jax.grad(loss_fn)(init)
        config = {"n_pools": len(jdata.pool_data),
                  "k_attr": jdata.x_attr.shape[1],
                  "mode": "per_pool_noise"}
        params = unpack_joint_params(grad, config)
        assert float(jnp.sum(jnp.abs(params["W_cad"]))) > 0


class TestFitJoint:
    """Test fit_joint: L-BFGS-B joint optimization."""

    def test_returns_result(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(matched_data, mode="per_pool_noise", maxiter=20)
        assert isinstance(result, dict)
        for key in ["bias_cad", "bias_gas", "W_cad", "W_gas", "loss", "converged"]:
            assert key in result, f"Missing key: {key}"

    def test_loss_decreases(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(matched_data, mode="per_pool_noise", maxiter=50)
        assert result["loss"] <= result["init_loss"]

    def test_predict_new_pool(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint, predict_new_pool_joint

        result = fit_joint(matched_data, mode="per_pool_noise", maxiter=20)
        # Predict for a new pool — k_attr must match training
        k_attr = result["W_cad"].shape[0]
        x_attr_new = np.zeros(k_attr)
        x_attr_new[0] = 1.0  # intercept
        pred = predict_new_pool_joint(result, x_attr_new)
        assert "cadence_minutes" in pred
        assert "gas_usd" in pred
        assert pred["cadence_minutes"] > 0
        assert pred["gas_usd"] > 0

    def test_init_from_option_c(self, matched_data):
        """Warm-starting from Option C per-pool fits should work."""
        from quantammsim.calibration.joint_fit import fit_joint
        from quantammsim.calibration.per_pool_fit import fit_all_pools

        option_c = fit_all_pools(matched_data)
        result = fit_joint(
            matched_data, mode="per_pool_noise",
            init_from_option_c=option_c, maxiter=20,
        )
        assert result["loss"] >= 0


class TestModes:
    """Test per_pool_noise vs shared_noise modes."""

    def test_per_pool_noise_mode(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(matched_data, mode="per_pool_noise", maxiter=20)
        assert "noise_coeffs" in result
        n_pools = len(matched_data)
        assert result["noise_coeffs"].shape == (n_pools, K_OBS)

    def test_shared_noise_mode(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(matched_data, mode="shared_noise", maxiter=20)
        assert "W_noise" in result
        k_attr = result["W_cad"].shape[0]
        assert result["W_noise"].shape == (k_attr, K_OBS)

    def test_shared_noise_predict(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint, predict_new_pool_joint

        result = fit_joint(matched_data, mode="shared_noise", maxiter=20)
        k_attr = result["W_cad"].shape[0]
        x_attr_new = np.zeros(k_attr)
        x_attr_new[0] = 1.0  # intercept
        pred = predict_new_pool_joint(result, x_attr_new)
        assert "noise_coeffs" in pred
        assert len(pred["noise_coeffs"]) == K_OBS


class TestPrepareTokenFactoredData:
    def test_returns_jdata_and_encoding(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_token_factored_data

        jdata, token_enc = prepare_token_factored_data(matched_data)
        assert hasattr(jdata, "pool_data")
        assert hasattr(jdata, "x_attr")
        assert "token_index" in token_enc
        assert "token_a_idx" in token_enc

    def test_reduced_x_obs_by_default(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_token_factored_data
        from quantammsim.calibration.pool_data import K_OBS_REDUCED

        jdata, _ = prepare_token_factored_data(matched_data)
        for pd in jdata.pool_data:
            assert pd["x_obs"].shape[1] == K_OBS_REDUCED

    def test_encoding_matches_pools(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_token_factored_data

        jdata, enc = prepare_token_factored_data(matched_data)
        n_pools = len(jdata.pool_data)
        assert len(enc["token_a_idx"]) == n_pools
        assert len(enc["token_b_idx"]) == n_pools
        assert len(enc["chain_idx"]) == n_pools
        assert len(enc["log_fees"]) == n_pools


class TestTokenFactoredEndToEnd:
    """Full pipeline: TokenFactoredNoiseHead + CalibrationModel + fit."""

    def test_model_fits_and_converges(self, matched_data):
        from quantammsim.calibration.calibration_model import CalibrationModel
        from quantammsim.calibration.heads import (
            FixedHead, PerPoolHead, TokenFactoredNoiseHead,
        )
        from quantammsim.calibration.joint_fit import prepare_token_factored_data
        from quantammsim.calibration.loss import CHAIN_GAS_USD
        from quantammsim.calibration.pool_data import K_OBS_REDUCED

        jdata, enc = prepare_token_factored_data(matched_data)
        n_pools = len(jdata.pool_data)

        # Gas: fixed to chain defaults
        gas_values = []
        for pid in jdata.pool_ids:
            chain = matched_data[pid]["chain"]
            gas_values.append(np.log(max(CHAIN_GAS_USD.get(chain, 1.0), 1e-6)))
        gas_head = FixedHead("log_gas", np.array(gas_values))

        # Cadence: per-pool
        cad_head = PerPoolHead("log_cadence", default=np.log(12.0))

        # Noise: token-factored
        noise_head = TokenFactoredNoiseHead(k_obs=K_OBS_REDUCED, **enc)

        model = CalibrationModel(cad_head, gas_head, noise_head)
        result = model.fit(jdata, maxiter=50)

        assert result["loss"] <= result["init_loss"]
        assert "token_effects" in result
        assert "noise_coeffs" in result
        assert result["noise_coeffs"].shape == (n_pools, K_OBS_REDUCED)

    def test_model_with_warm_start(self, matched_data):
        from quantammsim.calibration.calibration_model import CalibrationModel
        from quantammsim.calibration.heads import (
            FixedHead, PerPoolHead, TokenFactoredNoiseHead,
        )
        from quantammsim.calibration.joint_fit import prepare_token_factored_data
        from quantammsim.calibration.loss import CHAIN_GAS_USD
        from quantammsim.calibration.per_pool_fit import fit_all_pools
        from quantammsim.calibration.pool_data import K_OBS_REDUCED

        # Run Option C first
        option_c = fit_all_pools(
            matched_data, fix_gas_to_chain=True, reduced=True
        )

        jdata, enc = prepare_token_factored_data(matched_data)
        n_pools = len(jdata.pool_data)

        gas_values = []
        for pid in jdata.pool_ids:
            chain = matched_data[pid]["chain"]
            gas_values.append(np.log(max(CHAIN_GAS_USD.get(chain, 1.0), 1e-6)))

        noise_head = TokenFactoredNoiseHead(k_obs=K_OBS_REDUCED, **enc)
        model = CalibrationModel(
            PerPoolHead("log_cadence", default=np.log(12.0)),
            FixedHead("log_gas", np.array(gas_values)),
            noise_head,
        )
        result = model.fit(jdata, maxiter=100, warm_start=option_c)
        assert result["loss"] <= result["init_loss"]


class TestPrepareJointDataReduced:
    """Test prepare_joint_data with reduced_x_obs=True."""

    def test_reduced_x_obs_shape(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data
        from quantammsim.calibration.pool_data import K_OBS_REDUCED

        jdata = prepare_joint_data(matched_data, reduced_x_obs=True)
        for pd in jdata.pool_data:
            assert pd["x_obs"].shape[1] == K_OBS_REDUCED

    def test_default_unchanged(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data)
        for pd in jdata.pool_data:
            assert pd["x_obs"].shape[1] == K_OBS
