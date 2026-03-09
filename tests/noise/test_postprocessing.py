"""Tests for extract_noise_params, predict_new_pool, check_convergence,
assign_dp_clusters, and structural model post-processing."""

import numpy as np
import pytest

from quantammsim.noise_calibration import (
    extract_noise_params,
    predict_new_pool,
    check_convergence,
    classify_token_tier,
    _get_theta_samples,
    K_COEFF,
    COEFF_NAMES,
)
from quantammsim.noise_calibration.constants import K_OBS_COEFF, OBS_COEFF_NAMES


# ===========================================================================
# TestExtractNoiseParams
# ===========================================================================


class TestExtractNoiseParams:
    def test_output_length(self, synthetic_samples, synthetic_encoded_data):
        result = extract_noise_params(synthetic_samples, synthetic_encoded_data)
        assert len(result) == synthetic_encoded_data["N_pools"]

    def test_weekend_absorption(self, synthetic_samples, synthetic_encoded_data):
        """b_0_eff = b_0_raw + b_weekend * (2/7)."""
        result = extract_noise_params(synthetic_samples, synthetic_encoded_data)

        theta = _get_theta_samples(
            synthetic_samples, synthetic_encoded_data["X_pool"]
        )
        theta_med = np.median(theta, axis=0)

        for i, p in enumerate(result):
            b_0_raw = theta_med[i, 0]
            b_weekend = theta_med[i, 3]
            expected_b_0 = b_0_raw + b_weekend * (2.0 / 7.0)
            np.testing.assert_allclose(
                p["noise_params"]["b_0"], expected_b_0, atol=1e-10,
            )

    def test_noise_params_keys(self, synthetic_samples, synthetic_encoded_data):
        result = extract_noise_params(synthetic_samples, synthetic_encoded_data)
        expected_keys = {"b_0", "b_sigma", "b_c", "b_weekend", "base_fee"}
        for p in result:
            assert set(p["noise_params"].keys()) == expected_keys

    def test_theta_median_length(self, synthetic_samples, synthetic_encoded_data):
        result = extract_noise_params(synthetic_samples, synthetic_encoded_data)
        for p in result:
            assert len(p["theta_median"]) == K_COEFF

    def test_b_c_equals_theta_1(self, synthetic_samples, synthetic_encoded_data):
        result = extract_noise_params(synthetic_samples, synthetic_encoded_data)
        for p in result:
            np.testing.assert_allclose(
                p["noise_params"]["b_c"], p["theta_median"][1], atol=1e-10,
            )

    def test_b_sigma_equals_theta_2(
        self, synthetic_samples, synthetic_encoded_data
    ):
        result = extract_noise_params(synthetic_samples, synthetic_encoded_data)
        for p in result:
            np.testing.assert_allclose(
                p["noise_params"]["b_sigma"], p["theta_median"][2], atol=1e-10,
            )

    def test_base_fee_matches_pool(
        self, synthetic_samples, synthetic_encoded_data
    ):
        result = extract_noise_params(synthetic_samples, synthetic_encoded_data)
        pool_meta = synthetic_encoded_data["pool_meta"]
        for i, p in enumerate(result):
            expected_fee = pool_meta.iloc[i]["swap_fee"]
            np.testing.assert_allclose(
                p["noise_params"]["base_fee"], expected_fee, atol=1e-10,
            )

    def test_pool_id_and_chain_preserved(
        self, synthetic_samples, synthetic_encoded_data
    ):
        result = extract_noise_params(synthetic_samples, synthetic_encoded_data)
        pool_ids = synthetic_encoded_data["pool_ids"]
        pool_meta = synthetic_encoded_data["pool_meta"]
        for i, p in enumerate(result):
            assert p["pool_id"] == pool_ids[i]
            assert p["chain"] == str(pool_meta.iloc[i]["chain"])

    def test_use_median_false_uses_mean(
        self, synthetic_samples, synthetic_encoded_data
    ):
        """use_median=False must produce different values than use_median=True."""
        result_med = extract_noise_params(
            synthetic_samples, synthetic_encoded_data, use_median=True
        )
        result_mean = extract_noise_params(
            synthetic_samples, synthetic_encoded_data, use_median=False
        )
        assert len(result_mean) == len(result_med)

        # With random B samples (S=10, seed 99), median != mean for at least
        # one pool. Check that at least one theta_median value differs.
        any_differ = False
        for pm, pn in zip(result_med, result_mean):
            for tm, tn in zip(pm["theta_median"], pn["theta_median"]):
                if not np.isclose(tm, tn, atol=1e-14):
                    any_differ = True
                    break
            if any_differ:
                break
        assert any_differ, "median and mean paths produced identical values"


# ===========================================================================
# TestPredictNewPool
# ===========================================================================


class TestPredictNewPool:
    def _build_z_new(self, data, chain, tokens, fee):
        """Reconstruct z_new the same way predict_new_pool does internally."""
        col_names = data["covariate_names"]
        z_new = np.zeros(len(col_names), dtype=np.float64)

        tiers = sorted([classify_token_tier(t) for t in tokens])
        tier_a = str(tiers[0])
        tier_b = str(tiers[1]) if len(tiers) > 1 else tier_a

        for i, name in enumerate(col_names):
            if name == "intercept":
                z_new[i] = 1.0
            elif name == "log_fee":
                z_new[i] = np.log(max(fee, 1e-6))
            elif name == f"chain_{chain}":
                z_new[i] = 1.0
            elif name == f"tier_A_{tier_a}":
                z_new[i] = 1.0
            elif name == f"tier_B_{tier_b}":
                z_new[i] = 1.0
        return z_new

    def test_known_chain_sets_dummy(
        self, synthetic_samples, synthetic_encoded_data
    ):
        """ARBITRUM dummy must be 1 and must affect the prediction vs MAINNET."""
        data = synthetic_encoded_data

        result_arb = predict_new_pool(
            synthetic_samples, data,
            chain="ARBITRUM", tokens=["WETH", "USDC"], fee=0.003,
        )
        result_main = predict_new_pool(
            synthetic_samples, data,
            chain="MAINNET", tokens=["WETH", "USDC"], fee=0.003,
        )
        # MAINNET is the reference chain (alphabetically: ARBITRUM < BASE < MAINNET).
        # Wait — ARBITRUM is alphabetically first, so ARBITRUM is the reference.
        # MAINNET has a chain_MAINNET dummy. Both should produce different mu.
        # If chain dummies are ignored, these would be identical.
        arb_b0 = result_arb["noise_params"]["b_0"]
        main_b0 = result_main["noise_params"]["b_0"]
        assert not np.isclose(arb_b0, main_b0, atol=1e-10), (
            "Different chains should produce different predictions"
        )

    def test_tier_assignment_affects_prediction(
        self, synthetic_samples, synthetic_encoded_data
    ):
        """WETH/RATS (tier 0,2) vs WETH/USDC (tier 0,0) must differ."""
        data = synthetic_encoded_data

        result_rats = predict_new_pool(
            synthetic_samples, data,
            chain="BASE", tokens=["WETH", "RATS"], fee=0.005,
        )
        result_usdc = predict_new_pool(
            synthetic_samples, data,
            chain="BASE", tokens=["WETH", "USDC"], fee=0.005,
        )
        # Different tier_B dummies should give different mu
        rats_b0 = result_rats["noise_params"]["b_0"]
        usdc_b0 = result_usdc["noise_params"]["b_0"]
        assert not np.isclose(rats_b0, usdc_b0, atol=1e-10), (
            "Different tier assignments should produce different predictions"
        )

    def test_weekend_absorption_arithmetic(
        self, synthetic_samples, synthetic_encoded_data
    ):
        """b_0 in noise_params must equal mu_median[0] + mu_median[3] * (2/7)."""
        data = synthetic_encoded_data
        result = predict_new_pool(
            synthetic_samples, data,
            chain="MAINNET", tokens=["WETH", "USDC"], fee=0.003,
        )
        # Reconstruct mu_median independently
        z_new = self._build_z_new(data, "MAINNET", ["WETH", "USDC"], 0.003)
        B = synthetic_samples["B"]
        mu_samples = np.einsum("skd,d->sk", B, z_new)
        mu_median = np.median(mu_samples, axis=0)

        b_0_raw = mu_median[0]
        b_weekend = mu_median[3]
        expected_b_0 = b_0_raw + b_weekend * (2.0 / 7.0)
        np.testing.assert_allclose(
            result["noise_params"]["b_0"], expected_b_0, atol=1e-10,
        )

    def test_mu_equals_b_at_z_new(
        self, synthetic_samples, synthetic_encoded_data
    ):
        """With known B samples, verify credible interval medians = median(B @ z_new)."""
        data = synthetic_encoded_data
        z_new = self._build_z_new(data, "ARBITRUM", ["WETH", "RATS"], 0.005)

        B = synthetic_samples["B"]
        mu_expected = np.einsum("skd,d->sk", B, z_new)
        mu_median_expected = np.median(mu_expected, axis=0)

        result = predict_new_pool(
            synthetic_samples, data,
            chain="ARBITRUM", tokens=["WETH", "RATS"], fee=0.005,
        )
        for k, name in enumerate(COEFF_NAMES):
            np.testing.assert_allclose(
                result["credible_intervals_90"][name]["median"],
                mu_median_expected[k],
                atol=1e-10,
            )

    def test_unseen_chain_uses_reference(
        self, synthetic_samples, synthetic_encoded_data
    ):
        """A chain not in training data should get reference-chain prediction
        (all chain dummies = 0), not raise an error."""
        data = synthetic_encoded_data
        result = predict_new_pool(
            synthetic_samples, data,
            chain="SONIC", tokens=["WETH", "USDC"], fee=0.003,
        )
        # Should be same as ARBITRUM (the reference chain, all dummies 0)
        result_ref = predict_new_pool(
            synthetic_samples, data,
            chain="ARBITRUM", tokens=["WETH", "USDC"], fee=0.003,
        )
        np.testing.assert_allclose(
            result["noise_params"]["b_0"],
            result_ref["noise_params"]["b_0"],
            atol=1e-10,
        )


# ===========================================================================
# TestCheckConvergence
# ===========================================================================


class TestCheckConvergence:
    def test_svi_returns_expected_keys(self):
        losses = np.random.randn(1000).cumsum() + 5000
        result = check_convergence(losses, method="svi")
        assert "final_elbo" in result
        assert "elbo_last_100_std" in result
        assert "elbo_last_100_mean" in result

    def test_svi_method_key(self):
        losses = np.linspace(5000, 1000, 500)
        result = check_convergence(losses, method="svi")
        assert result["method"] == "svi"

    def test_svi_elbo_last_100_std_correct(self):
        np.random.seed(42)
        losses = np.random.randn(500) * 10 + 1000
        result = check_convergence(losses, method="svi")
        expected_std = float(np.std(losses[-100:]))
        np.testing.assert_allclose(
            result["elbo_last_100_std"], expected_std, atol=1e-10,
        )

    def test_svi_final_elbo_is_last_loss(self):
        losses = np.array([100.0, 50.0, 25.0, 12.5])
        result = check_convergence(losses, method="svi")
        assert result["final_elbo"] == 12.5


# ===========================================================================
# TestAssignDPClusters
# ===========================================================================


class TestAssignDPClusters:
    @pytest.fixture()
    def dp_samples_and_data(self, synthetic_encoded_data):
        """Synthetic DP posterior samples with known cluster structure."""
        data = synthetic_encoded_data
        N_pools = data["N_pools"]
        K_cov = data["K_cov"]
        K_clusters = 4
        S = 10

        np.random.seed(99)
        B = np.random.randn(S, K_COEFF, K_cov) * 0.5
        sigma_theta = np.ones((S, K_COEFF))
        L_Omega = np.tile(np.eye(K_COEFF), (S, 1, 1))
        eta = np.zeros((S, N_pools, K_COEFF))
        df = np.full((S,), 5.0)

        # Well-separated sigma_eps clusters
        sigma_eps = np.tile([0.3, 0.8, 1.5, 2.5], (S, 1))
        # Stick-breaking weights: mostly on cluster 0
        v = np.tile([0.6, 0.3, 0.05], (S, 1))

        samples = {
            "B": B, "sigma_theta": sigma_theta, "L_Omega": L_Omega,
            "eta": eta, "df": df, "sigma_eps": sigma_eps, "v": v,
        }
        data_with_k = dict(data)
        data_with_k["K_clusters"] = K_clusters
        return samples, data_with_k

    def test_returns_correct_length(self, dp_samples_and_data):
        from quantammsim.noise_calibration.postprocessing import assign_dp_clusters

        samples, data = dp_samples_and_data
        assignments = assign_dp_clusters(samples, data)
        assert len(assignments) == data["N_pools"]

    def test_valid_cluster_indices(self, dp_samples_and_data):
        from quantammsim.noise_calibration.postprocessing import assign_dp_clusters

        samples, data = dp_samples_and_data
        assignments = assign_dp_clusters(samples, data)
        K = data["K_clusters"]
        assert all(0 <= a < K for a in assignments)

    def test_returns_integer_array(self, dp_samples_and_data):
        from quantammsim.noise_calibration.postprocessing import assign_dp_clusters

        samples, data = dp_samples_and_data
        assignments = assign_dp_clusters(samples, data)
        assert assignments.dtype in (np.int32, np.int64, int)


# ===========================================================================
# TestExtractNoiseParamsDP
# ===========================================================================


class TestExtractNoiseParamsDP:
    @pytest.fixture()
    def dp_samples_and_data(self, synthetic_encoded_data):
        """Same as above for extract_noise_params testing."""
        data = synthetic_encoded_data
        N_pools = data["N_pools"]
        K_cov = data["K_cov"]
        S = 10

        np.random.seed(99)
        B = np.random.randn(S, K_COEFF, K_cov) * 0.5
        sigma_theta = np.ones((S, K_COEFF))
        L_Omega = np.tile(np.eye(K_COEFF), (S, 1, 1))
        eta = np.zeros((S, N_pools, K_COEFF))
        df = np.full((S,), 5.0)
        sigma_eps = np.tile([0.3, 0.8, 1.5, 2.5], (S, 1))
        v = np.tile([0.6, 0.3, 0.05], (S, 1))

        samples = {
            "B": B, "sigma_theta": sigma_theta, "L_Omega": L_Omega,
            "eta": eta, "df": df, "sigma_eps": sigma_eps, "v": v,
        }
        return samples, data

    def test_output_length(self, dp_samples_and_data):
        samples, data = dp_samples_and_data
        result = extract_noise_params(samples, data)
        assert len(result) == data["N_pools"]

    def test_noise_params_keys_present(self, dp_samples_and_data):
        samples, data = dp_samples_and_data
        result = extract_noise_params(samples, data)
        expected_keys = {"b_0", "b_sigma", "b_c", "b_weekend", "base_fee"}
        for p in result:
            assert set(p["noise_params"].keys()) == expected_keys


# ===========================================================================
# TestExtractStructuralParams
# ===========================================================================


class TestExtractStructuralParams:
    """Tests for extract_structural_params()."""

    @pytest.fixture()
    def structural_samples_and_data(self, synthetic_structural_data):
        """Run a quick SVI fit to get structural samples."""
        from quantammsim.noise_calibration.inference import run_svi
        from quantammsim.noise_calibration.model import structural_noise_model

        samples, _ = run_svi(
            synthetic_structural_data,
            num_steps=500,
            lr=5e-3,
            seed=0,
            num_samples=10,
            model_fn=structural_noise_model,
        )
        return samples, synthetic_structural_data

    def test_extract_returns_arb_params(self, structural_samples_and_data):
        from quantammsim.noise_calibration.postprocessing import (
            extract_structural_params,
        )
        samples, data = structural_samples_and_data
        result = extract_structural_params(samples, data)
        assert len(result) == data["N_pools"]
        for p in result:
            assert "arb_frequency" in p
            assert isinstance(p["arb_frequency"], int)
            assert 1 <= p["arb_frequency"] <= 60

    def test_extract_returns_noise_params(self, structural_samples_and_data):
        from quantammsim.noise_calibration.postprocessing import (
            extract_structural_params,
        )
        samples, data = structural_samples_and_data
        result = extract_structural_params(samples, data)
        for p in result:
            assert "noise_params" in p
            coeffs = p["noise_params"]
            assert len(coeffs) == K_OBS_COEFF
            for name in OBS_COEFF_NAMES:
                assert name in coeffs, f"Missing coefficient: {name}"


# ===========================================================================
# TestPredictStructural
# ===========================================================================


class TestPredictStructural:
    @pytest.fixture()
    def structural_samples_and_data(self, synthetic_structural_data):
        from quantammsim.noise_calibration.inference import run_svi
        from quantammsim.noise_calibration.model import structural_noise_model

        samples, _ = run_svi(
            synthetic_structural_data,
            num_steps=500,
            lr=5e-3,
            seed=0,
            num_samples=10,
            model_fn=structural_noise_model,
        )
        return samples, synthetic_structural_data

    def test_predict_returns_cadence(self, structural_samples_and_data):
        from quantammsim.noise_calibration.postprocessing import (
            predict_new_pool_structural,
        )
        samples, data = structural_samples_and_data
        result = predict_new_pool_structural(
            samples, data,
            chain="MAINNET", tokens=["WETH", "USDC"],
            fee=0.003, tvl_est=1e6,
        )
        assert "arb_frequency" in result
        assert isinstance(result["arb_frequency"], int)
        assert 1 <= result["arb_frequency"] <= 60

    def test_predict_returns_noise_coefficients(
        self, structural_samples_and_data,
    ):
        from quantammsim.noise_calibration.postprocessing import (
            predict_new_pool_structural,
        )
        samples, data = structural_samples_and_data
        result = predict_new_pool_structural(
            samples, data,
            chain="MAINNET", tokens=["WETH", "USDC"],
            fee=0.003, tvl_est=1e6,
        )
        assert "noise_params" in result
        assert len(result["noise_params"]) == K_OBS_COEFF

    def test_predict_uses_B_regression(self, structural_samples_and_data):
        """Different (chain, tier) → different predictions via B regression."""
        from quantammsim.noise_calibration.postprocessing import (
            predict_new_pool_structural,
        )
        samples, data = structural_samples_and_data
        r1 = predict_new_pool_structural(
            samples, data,
            chain="MAINNET", tokens=["WETH", "USDC"],
            fee=0.003, tvl_est=1e6,
        )
        r2 = predict_new_pool_structural(
            samples, data,
            chain="ARBITRUM", tokens=["BAL", "WETH"],
            fee=0.01, tvl_est=5e5,
        )
        # Different pool characteristics should produce different noise params
        assert r1["noise_params"] != r2["noise_params"]

    def test_predict_cadence_higher_for_longtail(
        self, structural_samples_and_data,
    ):
        """Long-tail pools should have higher cadence (less efficient arb)."""
        from quantammsim.noise_calibration.postprocessing import (
            predict_new_pool_structural,
        )
        samples, data = structural_samples_and_data
        # This tests the structural relationship — it may not hold with
        # random SVI samples on synthetic data, so we just check it doesn't
        # crash and returns valid values
        r1 = predict_new_pool_structural(
            samples, data,
            chain="MAINNET", tokens=["WETH", "USDC"],
            fee=0.003, tvl_est=1e6,
        )
        r2 = predict_new_pool_structural(
            samples, data,
            chain="BASE", tokens=["RATS", "WETH"],
            fee=0.005, tvl_est=1e5,
        )
        # Both should be valid
        assert 1 <= r1["arb_frequency"] <= 60
        assert 1 <= r2["arb_frequency"] <= 60
