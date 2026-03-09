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
        for i, pid in enumerate(jdata.pool_ids):
            if matched_data[pid]["chain"] == "MAINNET":
                np.testing.assert_allclose(
                    float(jdata.pool_data[i]["fixed_log_gas"]),
                    0.0,
                    atol=1e-6,
                )

    def test_arbitrum_gas_is_log_001(self, matched_data):
        """ARBITRUM pools should have fixed_log_gas = log(0.01)."""
        from quantammsim.calibration.joint_fit import prepare_joint_data

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)
        for i, pid in enumerate(jdata.pool_ids):
            if matched_data[pid]["chain"] == "ARBITRUM":
                np.testing.assert_allclose(
                    float(jdata.pool_data[i]["fixed_log_gas"]),
                    np.log(0.01),
                    rtol=1e-6,
                )


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

    def test_no_gas_regularization(self, matched_data):
        """With fix_gas=True, alpha_gas should have no effect."""
        from quantammsim.calibration.joint_fit import (
            make_initial_joint_params,
            make_joint_loss_fn,
            prepare_joint_data,
        )

        jdata = prepare_joint_data(matched_data, fix_gas_to_chain=True)
        init = make_initial_joint_params(jdata, mode="per_pool_noise", fix_gas=True)

        loss_fn_a = make_joint_loss_fn(
            jdata, mode="per_pool_noise", fix_gas=True, alpha_gas=0.0
        )
        loss_fn_b = make_joint_loss_fn(
            jdata, mode="per_pool_noise", fix_gas=True, alpha_gas=100.0
        )

        np.testing.assert_allclose(
            float(loss_fn_a(init)), float(loss_fn_b(init)), rtol=1e-6
        )

    def test_shared_noise_mode(self, matched_data):
        loss_fn, init, _ = self._make_loss_fn(
            matched_data, mode="shared_noise"
        )
        loss = loss_fn(init)
        assert loss.shape == ()
        assert float(loss) >= 0

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

    def test_gas_per_pool_stored(self, matched_data):
        """Result should contain gas_per_pool with chain-level values."""
        from quantammsim.calibration.joint_fit import fit_joint
        from quantammsim.calibration.loss import CHAIN_GAS_USD

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=10,
        )
        assert "gas_per_pool" in result
        for i, pid in enumerate(result["pool_ids"]):
            chain = matched_data[pid]["chain"]
            expected = CHAIN_GAS_USD.get(chain, 1.0)
            assert result["gas_per_pool"][i] == expected

    def test_loss_decreases(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=50,
        )
        assert result["loss"] <= result["init_loss"]

    def test_w_gas_is_zeros(self, matched_data):
        """With fixed gas, W_gas should be zeros (placeholder)."""
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=10,
        )
        np.testing.assert_allclose(result["W_gas"], 0.0)

    def test_bias_gas_is_zero(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True, maxiter=10,
        )
        assert result["bias_gas"] == 0.0

    def test_warm_start_from_option_c(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint
        from quantammsim.calibration.per_pool_fit import fit_all_pools

        option_c = fit_all_pools(matched_data, fix_gas_to_chain=True)
        result = fit_joint(
            matched_data, mode="per_pool_noise",
            fix_gas_to_chain=True,
            init_from_option_c=option_c,
            maxiter=20,
        )
        assert result["loss"] >= 0

    def test_shared_noise_fixed_gas(self, matched_data):
        from quantammsim.calibration.joint_fit import fit_joint

        result = fit_joint(
            matched_data, mode="shared_noise",
            fix_gas_to_chain=True, maxiter=20,
        )
        assert "W_noise" in result
        assert result["fix_gas"] is True

    def test_predict_new_pool_fixed_gas(self, matched_data):
        """predict_new_pool_joint should still work with fixed-gas results."""
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
        # gas_usd comes from bias_gas=0 + W_gas=0 → exp(0)=1.0
        assert pred["gas_usd"] > 0


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
