"""Tests for fixed-gas mode in quantammsim.calibration.joint_fit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.calibration.conftest import K_OBS, POOL_PREFIXES


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


class TestPrepareJointDataFixedGas:
    """Test prepare_joint_data with fix_gas_to_chain=True."""

    def test_pool_data_has_fixed_log_gas(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)
        for pd_i in jdata.pool_data:
            assert "fixed_log_gas" in pd_i

    def test_pool_data_no_fixed_log_gas_by_default(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=False)
        for pd_i in jdata.pool_data:
            assert "fixed_log_gas" not in pd_i

    def test_fixed_log_gas_values_match_chain(self, matched_data):
        """fixed_log_gas should be log(CHAIN_GAS_USD[chain])."""
        from quantammsim.calibration.joint_fit import prepare_joint_data
        from quantammsim.calibration.loss import CHAIN_GAS_USD

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)

        for i, pid in enumerate(jdata.pool_ids):
            chain = matched_data[pid]["chain"]
            expected_gas = CHAIN_GAS_USD.get(chain, 1.0)
            expected_log_gas = np.log(max(expected_gas, 1e-6))
            np.testing.assert_allclose(
                float(jdata.pool_data[i]["fixed_log_gas"]),
                expected_log_gas,
                rtol=1e-6,
            )

    def test_mainnet_gas_is_log_1(self, matched_data):
        """MAINNET pools should have fixed_log_gas = log(1.0) = 0.0."""
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)
        found = False
        for i, pid in enumerate(jdata.pool_ids):
            if matched_data[pid]["chain"] == "MAINNET":
                np.testing.assert_allclose(
                    float(jdata.pool_data[i]["fixed_log_gas"]),
                    0.0,
                    atol=1e-6,
                )
                found = True
        assert found, "No MAINNET pool found in test data"

    def test_arbitrum_gas_is_log_001(self, matched_data):
        """ARBITRUM pools should have fixed_log_gas = log(0.01)."""
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)
        found = False
        for i, pid in enumerate(jdata.pool_ids):
            if matched_data[pid]["chain"] == "ARBITRUM":
                np.testing.assert_allclose(
                    float(jdata.pool_data[i]["fixed_log_gas"]),
                    np.log(0.01),
                    rtol=1e-6,
                )
                found = True
        assert found, "No ARBITRUM pool found in test data"

    def test_x_attr_has_correct_shape(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)
        assert jdata.x_attr.shape[0] == len(jdata.pool_ids)
        assert jdata.x_attr.shape[1] == len(jdata.attr_names)

    def test_pool_data_has_obs_arrays(self, matched_data):
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)
        for pd_i in jdata.pool_data:
            assert pd_i["x_obs"].ndim == 2
            assert pd_i["y_obs"].ndim == 1
            assert pd_i["day_indices"].ndim == 1
            assert pd_i["x_obs"].shape[0] == pd_i["y_obs"].shape[0]


class TestPackUnpackJointFixedGas:
    """Test packing/unpacking joint params with fix_gas=True."""

    def test_pack_fixed_gas_shape(self):
        from quantammsim.calibration.joint_fit import pack_joint_params_fixed_gas

        k_attr = 6
        n_pools = 2
        noise_params = jnp.zeros((n_pools, K_OBS))
        flat = pack_joint_params_fixed_gas(
            1.0, jnp.zeros(k_attr), noise_params
        )
        # Layout: [bias_cad, W_cad(6), noise(2*8)] = 1+6+16 = 23
        assert flat.shape == (1 + k_attr + n_pools * K_OBS,)

    def test_pack_fixed_gas_shorter_than_free(self):
        from quantammsim.calibration.joint_fit import (
            pack_joint_params,
            pack_joint_params_fixed_gas,
        )

        k_attr = 6
        n_pools = 2
        noise = jnp.zeros((n_pools, K_OBS))

        free = pack_joint_params(1.0, 2.0, jnp.zeros(k_attr),
                                  jnp.zeros(k_attr), noise)
        fixed = pack_joint_params_fixed_gas(1.0, jnp.zeros(k_attr), noise)
        # Fixed is shorter by: 1 (bias_gas) + k_attr (W_gas)
        assert free.shape[0] - fixed.shape[0] == 1 + k_attr

    def test_unpack_fixed_gas_no_gas_keys(self):
        from quantammsim.calibration.joint_fit import (
            pack_joint_params_fixed_gas,
            unpack_joint_params,
        )

        k_attr = 6
        n_pools = 2
        noise = jnp.zeros((n_pools, K_OBS))
        flat = pack_joint_params_fixed_gas(
            1.0, jnp.ones(k_attr) * 0.5, noise
        )

        config = {"k_attr": k_attr, "n_pools": n_pools,
                  "mode": "per_pool_noise", "fix_gas": True}
        params = unpack_joint_params(flat, config)

        assert "bias_cad" in params
        assert "W_cad" in params
        assert "noise_coeffs" in params
        assert "bias_gas" not in params
        assert "W_gas" not in params

    def test_unpack_roundtrip_per_pool_noise(self):
        from quantammsim.calibration.joint_fit import (
            pack_joint_params_fixed_gas,
            unpack_joint_params,
        )

        k_attr = 4
        n_pools = 3
        bias_cad = 2.5
        W_cad = jnp.array([0.1, -0.2, 0.3, -0.4])
        noise = jnp.arange(n_pools * K_OBS, dtype=float).reshape(n_pools, K_OBS)

        flat = pack_joint_params_fixed_gas(bias_cad, W_cad, noise)
        config = {"k_attr": k_attr, "n_pools": n_pools,
                  "mode": "per_pool_noise", "fix_gas": True}
        params = unpack_joint_params(flat, config)

        np.testing.assert_allclose(params["bias_cad"], bias_cad)
        np.testing.assert_allclose(params["W_cad"], W_cad)
        np.testing.assert_allclose(params["noise_coeffs"], noise)

    def test_unpack_roundtrip_shared_noise(self):
        from quantammsim.calibration.joint_fit import (
            pack_joint_params_fixed_gas,
            unpack_joint_params,
        )

        k_attr = 4
        bias_cad = 1.5
        W_cad = jnp.array([0.5, -0.5, 0.1, -0.1])
        # shared_noise: (1+k_attr, K_OBS) = (5, 8)
        noise = jnp.arange((1 + k_attr) * K_OBS, dtype=float).reshape(
            1 + k_attr, K_OBS
        )

        flat = pack_joint_params_fixed_gas(bias_cad, W_cad, noise)
        config = {"k_attr": k_attr, "n_pools": 99,
                  "mode": "shared_noise", "fix_gas": True}
        params = unpack_joint_params(flat, config)

        np.testing.assert_allclose(params["bias_cad"], bias_cad)
        np.testing.assert_allclose(params["W_cad"], W_cad)
        np.testing.assert_allclose(params["bias_noise"], noise[0])
        np.testing.assert_allclose(params["W_noise"], noise[1:])


class TestJointLossFixedGas:
    """Test joint loss function with fix_gas=True."""

    def _make_loss_fn(self, matched_data, mode="per_pool_noise"):
        from quantammsim.calibration.joint_fit import (
            make_initial_joint_params,
            make_joint_loss_fn,
            prepare_joint_data,
        )

        jdata = prepare_joint_data(
            matched_data, fix_gas_to_chain=True
        )
        init = make_initial_joint_params(
            jdata, mode=mode, fix_gas=True
        )
        loss_fn = make_joint_loss_fn(
            jdata, mode=mode, fix_gas=True
        )
        return loss_fn, init, jdata

    def test_loss_differentiable_and_nonzero_grad(self, matched_data):
        loss_fn, init, _ = self._make_loss_fn(matched_data)
        grad = jax.grad(loss_fn)(init)
        assert grad.shape == init.shape
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.sum(jnp.abs(grad))) > 1e-10

    def test_no_gas_regularization_but_cad_regularization_works(self, matched_data):
        """alpha_gas has no effect, but alpha_cad DOES."""
        from quantammsim.calibration.joint_fit import (
            make_initial_joint_params,
            make_joint_loss_fn,
            prepare_joint_data,
        )

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)
        init = make_initial_joint_params(jdata, mode="per_pool_noise", fix_gas=True)

        # alpha_gas shouldn't matter
        loss_fn_a = make_joint_loss_fn(
            jdata, mode="per_pool_noise", fix_gas=True, alpha_gas=0.0
        )
        loss_fn_b = make_joint_loss_fn(
            jdata, mode="per_pool_noise", fix_gas=True, alpha_gas=100.0
        )
        np.testing.assert_allclose(
            float(loss_fn_a(init)), float(loss_fn_b(init)), rtol=1e-6
        )

        # alpha_cad SHOULD matter (positive control)
        loss_fn_no_reg = make_joint_loss_fn(
            jdata, mode="per_pool_noise", fix_gas=True, alpha_cad=0.0
        )
        loss_fn_big_reg = make_joint_loss_fn(
            jdata, mode="per_pool_noise", fix_gas=True, alpha_cad=100.0
        )
        # With W_cad initialized to non-zero by warm start, these should differ.
        # Even with default init (W_cad=0), perturbation test:
        init_perturbed = init.at[1].set(1.0)  # perturb first W_cad element
        loss_no = float(loss_fn_no_reg(init_perturbed))
        loss_big = float(loss_fn_big_reg(init_perturbed))
        assert loss_big > loss_no, "alpha_cad regularization has no effect"

    def test_shared_noise_mode(self, matched_data):
        loss_fn, init, _ = self._make_loss_fn(
            matched_data, mode="shared_noise"
        )
        loss = loss_fn(init)
        assert loss.shape == ()
        assert float(loss) >= 0
        # Verify gradient works for shared_noise too
        grad = jax.grad(loss_fn)(init)
        assert jnp.all(jnp.isfinite(grad))

    def test_init_param_count_per_pool_noise(self, matched_data):
        """Verify parameter count: 1(bias_cad) + k_attr(W_cad) + n_pools*K_OBS."""
        _, init, jdata = self._make_loss_fn(matched_data)
        k_attr = jdata.x_attr.shape[1]
        n_pools = len(jdata.pool_data)
        expected = 1 + k_attr + n_pools * K_OBS
        assert init.shape[0] == expected

    def test_init_param_count_shared_noise(self, matched_data):
        """Verify: 1(bias_cad) + k_attr(W_cad) + (1+k_attr)*K_OBS."""
        _, init, jdata = self._make_loss_fn(
            matched_data, mode="shared_noise"
        )
        k_attr = jdata.x_attr.shape[1]
        expected = 1 + k_attr + (1 + k_attr) * K_OBS
        assert init.shape[0] == expected


class TestFitJointFixedGas:
    """Test fit_joint with fix_gas_to_chain=True."""

    def test_returns_result(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=20,
        )
        assert isinstance(result, dict)
        for key in ["bias_cad", "W_cad", "loss", "converged", "fix_gas"]:
            assert key in result, f"Missing key: {key}"

    def test_fix_gas_flag_stored(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=10,
        )
        assert result["fix_gas"] is True

    def test_gas_per_pool_stored_with_correct_values(self, matched_data):
        """gas_per_pool has right length and chain-level values."""
        from quantammsim.calibration.joint_fit import fit_joint
        from quantammsim.calibration.loss import CHAIN_GAS_USD

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=10,
        )
        assert "gas_per_pool" in result
        assert len(result["gas_per_pool"]) == len(result["pool_ids"])
        for i, pid in enumerate(result["pool_ids"]):
            chain = matched_data[pid]["chain"]
            expected = CHAIN_GAS_USD.get(chain, 1.0)
            assert result["gas_per_pool"][i] == expected

    def test_loss_decreases_substantially(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=50,
        )
        # Must decrease, not just by epsilon
        assert result["loss"] < result["init_loss"] * 0.999

    def test_w_gas_and_bias_gas_are_zeros(self, matched_data):
        """With fixed gas, W_gas and bias_gas should be zero placeholders."""
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=10,
        )
        np.testing.assert_allclose(result["W_gas"], 0.0)
        assert result["bias_gas"] == 0.0

    def test_warm_start_from_option_c_runs(self, matched_data):
        """Warm start from Option C should run without error and reduce loss."""
        from quantammsim.calibration.joint_fit import fit_joint
        from quantammsim.calibration.per_pool_fit import fit_all_pools

        option_c = fit_all_pools(matched_data, fix_gas_to_chain=True)
        result_warm = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True,
            init_from_option_c=option_c,
            maxiter=50,
        )
        # Should at least decrease from its own init
        assert result_warm["loss"] <= result_warm["init_loss"]
        assert result_warm["loss"] >= 0

    def test_shared_noise_fixed_gas(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="shared_noise",
            fix_gas_to_chain=True, maxiter=20,
        )
        assert "W_noise" in result
        assert "bias_noise" in result
        assert result["fix_gas"] is True
        assert result["loss"] >= 0

    def test_predict_new_pool_fixed_gas_pinned(self, matched_data):
        """predict_new_pool_joint with zero attrs → gas_usd=1.0 exactly."""
        from quantammsim.calibration.joint_fit import fit_joint, predict_new_pool_joint

        result = fit_joint(
            matched_data, mode="shared_noise",
            fix_gas_to_chain=True, maxiter=20,
        )
        k_attr = result["W_cad"].shape[0]
        x_attr_new = np.zeros(k_attr)
        pred = predict_new_pool_joint(result, x_attr_new)

        assert "cadence_minutes" in pred
        assert pred["cadence_minutes"] > 0
        # bias_gas=0, W_gas=zeros → log_gas=0 → gas_usd=exp(0)=1.0
        np.testing.assert_allclose(pred["gas_usd"], 1.0, rtol=1e-6)
        # cadence = exp(bias_cad + 0) = exp(bias_cad)
        np.testing.assert_allclose(
            pred["cadence_minutes"], np.exp(result["bias_cad"]), rtol=1e-6
        )
        # shared_noise mode should include noise_coeffs
        assert "noise_coeffs" in pred
        assert len(pred["noise_coeffs"]) == K_OBS

    def test_predict_matches_linear_model(self, matched_data):
        """predict_new_pool_joint computes cadence = exp(bias_cad + W_cad @ x)."""
        from quantammsim.calibration.joint_fit import fit_joint, predict_new_pool_joint

        result = fit_joint(
            matched_data, mode="shared_noise",
            fix_gas_to_chain=True, maxiter=20,
        )
        k_attr = result["W_cad"].shape[0]
        x_test = np.random.RandomState(42).randn(k_attr)

        pred = predict_new_pool_joint(result, x_test)

        # Verify cadence prediction directly against the linear model
        expected_cadence = float(np.exp(
            result["bias_cad"] + result["W_cad"] @ x_test
        ))
        np.testing.assert_allclose(
            pred["cadence_minutes"], expected_cadence, rtol=1e-6,
        )
        # Gas is fixed → always exp(0) = 1.0
        np.testing.assert_allclose(pred["gas_usd"], 1.0, rtol=1e-6)


class TestMakeBoundsFixedGas:
    """Test _make_bounds with fix_gas=True."""

    def test_bounds_count_per_pool_noise(self):
        from quantammsim.calibration.joint_fit import _make_bounds

        k_attr = 6
        n_pools = 3
        bounds = _make_bounds(k_attr, n_pools, "per_pool_noise", fix_gas=True)
        # 1(bias_cad) + 6(W_cad) + 3*8(noise) = 31
        assert len(bounds) == 1 + k_attr + n_pools * K_OBS

    def test_bounds_count_shared_noise(self):
        from quantammsim.calibration.joint_fit import _make_bounds

        k_attr = 6
        n_pools = 3
        bounds = _make_bounds(k_attr, n_pools, "shared_noise", fix_gas=True)
        # 1(bias_cad) + 6(W_cad) + (1+6)*8(noise) = 63
        assert len(bounds) == 1 + k_attr + (1 + k_attr) * K_OBS

    def test_bounds_fewer_with_fixed_gas(self):
        from quantammsim.calibration.joint_fit import _make_bounds

        k_attr = 6
        n_pools = 3
        free = _make_bounds(k_attr, n_pools, "per_pool_noise", fix_gas=False)
        fixed = _make_bounds(k_attr, n_pools, "per_pool_noise", fix_gas=True)
        # Difference: 1(bias_gas) + k_attr(W_gas)
        assert len(free) - len(fixed) == 1 + k_attr
