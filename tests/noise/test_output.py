"""Tests for generate_output_json and _save_sample_cache."""

import json
import os

import numpy as np
import pytest

from quantammsim.noise_calibration import (
    extract_noise_params,
    generate_output_json,
    _save_sample_cache,
    K_COEFF,
)


# ===========================================================================
# TestGenerateOutputJSON
# ===========================================================================


class TestGenerateOutputJSON:
    @pytest.fixture()
    def _output_setup(self, tmp_path, synthetic_samples, synthetic_encoded_data):
        """Produce the JSON file and return (path, data dict)."""
        pool_params = extract_noise_params(
            synthetic_samples, synthetic_encoded_data
        )
        output_path = str(tmp_path / "test_output.json")
        convergence = {"method": "svi", "final_elbo": 1234.0}
        inference_config = {"method": "svi", "svi_steps": 1000}

        generate_output_json(
            pool_params, synthetic_samples, synthetic_encoded_data,
            convergence, output_path, inference_config,
        )
        with open(output_path) as f:
            data = json.load(f)
        return output_path, data

    def test_writes_valid_json(self, _output_setup):
        path, data = _output_setup
        assert isinstance(data, dict)

    def test_top_level_keys(self, _output_setup):
        _, data = _output_setup
        expected = {
            "model", "model_spec", "inference", "population_effects",
            "convergence", "n_pools", "n_obs", "pools",
        }
        assert expected.issubset(data.keys())

    def test_model_spec_fields(self, _output_setup):
        _, data = _output_setup
        spec = data["model_spec"]
        assert "K_coeff" in spec
        assert "K_cov" in spec
        assert "coeff_names" in spec
        assert "covariate_names" in spec
        assert "likelihood" in spec
        assert spec["likelihood"] == "StudentT"
        assert "tvl_lag" in spec
        assert spec["tvl_lag"] == "log_tvl_lag1"

    def test_population_effects_fields(self, _output_setup):
        _, data = _output_setup
        pe = data["population_effects"]
        assert "B" in pe
        assert "sigma_theta" in pe
        assert "sigma_eps" in pe
        assert "df" in pe
        assert "correlation_matrix" in pe

    def test_pool_entries(self, _output_setup, synthetic_encoded_data):
        _, data = _output_setup
        pools = data["pools"]
        for pid in synthetic_encoded_data["pool_ids"]:
            assert pid in pools
            entry = pools[pid]
            assert "chain" in entry
            assert "tokens" in entry
            assert "theta_median" in entry
            assert "noise_params" in entry

    def test_correlation_matrix_symmetric_unit_diagonal(self, _output_setup):
        _, data = _output_setup
        Omega = np.array(data["population_effects"]["correlation_matrix"])
        np.testing.assert_allclose(Omega, Omega.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(Omega), 1.0, atol=1e-10)

    def test_n_pools_and_n_obs(self, _output_setup, synthetic_encoded_data):
        _, data = _output_setup
        assert data["n_pools"] == synthetic_encoded_data["N_pools"]
        assert data["n_obs"] == len(synthetic_encoded_data["y_obs"])

    def test_model_name(self, _output_setup):
        _, data = _output_setup
        assert data["model"] == "unified_hierarchical_student_t"


# ===========================================================================
# TestSaveSampleCache
# ===========================================================================


class TestSaveSampleCache:
    def test_creates_npz_and_json(
        self, tmp_path, synthetic_samples, synthetic_encoded_data
    ):
        cache_dir = str(tmp_path / "cache")
        _save_sample_cache(synthetic_samples, synthetic_encoded_data, cache_dir)

        assert os.path.exists(os.path.join(cache_dir, "unified_samples.npz"))
        assert os.path.exists(os.path.join(cache_dir, "unified_data.json"))

    def test_npz_excludes_y_and_theta(
        self, tmp_path, synthetic_samples, synthetic_encoded_data
    ):
        """Both 'y' and 'theta' must be excluded from the npz cache."""
        samples_with_extras = dict(synthetic_samples)
        samples_with_extras["y"] = np.random.randn(10, 27)
        samples_with_extras["theta"] = np.random.randn(10, 3, 4)

        cache_dir = str(tmp_path / "cache2")
        _save_sample_cache(samples_with_extras, synthetic_encoded_data, cache_dir)

        npz_path = os.path.join(cache_dir, "unified_samples.npz")
        loaded = np.load(npz_path)
        assert "y" not in loaded.files
        assert "theta" not in loaded.files

    def test_npz_contains_required_keys(
        self, tmp_path, synthetic_samples, synthetic_encoded_data
    ):
        """The npz must contain B, sigma_theta, L_Omega, eta, df, sigma_eps."""
        cache_dir = str(tmp_path / "cache3")
        _save_sample_cache(synthetic_samples, synthetic_encoded_data, cache_dir)

        npz_path = os.path.join(cache_dir, "unified_samples.npz")
        loaded = np.load(npz_path)
        required = {"B", "sigma_theta", "L_Omega", "eta", "df", "sigma_eps"}
        assert required.issubset(set(loaded.files))

    def test_json_contains_metadata_keys(
        self, tmp_path, synthetic_samples, synthetic_encoded_data
    ):
        """The JSON cache must contain all keys needed by --predict."""
        cache_dir = str(tmp_path / "cache4")
        _save_sample_cache(synthetic_samples, synthetic_encoded_data, cache_dir)

        json_path = os.path.join(cache_dir, "unified_data.json")
        with open(json_path) as f:
            meta = json.load(f)
        required = {
            "pool_ids", "covariate_names", "K_cov", "N_pools",
            "ref_chain", "ref_tier_a", "ref_tier_b", "chains",
        }
        assert required.issubset(meta.keys())


# ===========================================================================
# TestGenerateOutputJSONDP
# ===========================================================================


class TestGenerateOutputJSONDP:
    @pytest.fixture()
    def _dp_output_setup(self, tmp_path, synthetic_encoded_data):
        """Produce DP model JSON output and return (path, data dict)."""
        data = synthetic_encoded_data
        N_pools = data["N_pools"]
        K_cov = data["K_cov"]
        K_clusters = 4
        S = 10

        np.random.seed(99)
        dp_samples = {
            "B": np.random.randn(S, K_COEFF, K_cov) * 0.5,
            "sigma_theta": np.ones((S, K_COEFF)),
            "L_Omega": np.tile(np.eye(K_COEFF), (S, 1, 1)),
            "eta": np.zeros((S, N_pools, K_COEFF)),
            "df": np.full((S,), 5.0),
            "sigma_eps": np.tile([0.3, 0.8, 1.5, 2.5], (S, 1)),
            "v": np.tile([0.6, 0.3, 0.05], (S, 1)),
            "alpha_dp": np.full((S,), 1.5),
        }

        pool_params = extract_noise_params(dp_samples, data)
        output_path = str(tmp_path / "dp_output.json")
        convergence = {"method": "svi", "final_elbo": 1234.0}
        inference_config = {"method": "svi", "svi_steps": 1000}

        generate_output_json(
            pool_params, dp_samples, data,
            convergence, output_path, inference_config,
        )
        with open(output_path) as f:
            result = json.load(f)
        return output_path, result

    def test_sigma_eps_structure_is_dp_mixture(self, _dp_output_setup):
        _, data = _dp_output_setup
        assert data["model_spec"]["sigma_eps_structure"] == "dp_mixture"

    def test_model_name_includes_dp(self, _dp_output_setup):
        _, data = _dp_output_setup
        assert "dp_sigma" in data["model"]

    def test_has_cluster_weights(self, _dp_output_setup):
        _, data = _dp_output_setup
        assert "cluster_weights" in data["population_effects"]

    def test_sigma_eps_length_equals_k_clusters(self, _dp_output_setup):
        _, data = _dp_output_setup
        sigma_eps = data["population_effects"]["sigma_eps"]
        assert len(sigma_eps) == 4  # K_clusters = 4

    def test_cluster_weights_sum_to_one(self, _dp_output_setup):
        _, data = _dp_output_setup
        w = data["population_effects"]["cluster_weights"]
        np.testing.assert_allclose(sum(w), 1.0, atol=1e-4)

    def test_still_has_standard_fields(self, _dp_output_setup):
        _, data = _dp_output_setup
        expected = {
            "model", "model_spec", "inference", "population_effects",
            "convergence", "n_pools", "n_obs", "pools",
        }
        assert expected.issubset(data.keys())


# ===========================================================================
# TestGenerateOutputJSONStructural
# ===========================================================================


class TestGenerateOutputJSONStructural:
    @pytest.fixture()
    def _structural_output_setup(self, tmp_path, synthetic_structural_data):
        """Produce structural model JSON output and return (path, data dict)."""
        from quantammsim.noise_calibration.postprocessing import (
            extract_structural_params,
        )
        from quantammsim.noise_calibration.constants import K_OBS_COEFF

        data = synthetic_structural_data
        K_cov = data["K_cov"]
        N_pools = data["N_pools"]
        n_chains = data["n_chains"]
        n_tiers = data["n_tiers"]
        S = 10

        np.random.seed(77)
        structural_samples = {
            "alpha_0": np.random.randn(S) * 0.1 + 2.0,
            "alpha_chain": np.random.randn(S, n_chains - 1) * 0.1,
            "alpha_tier": np.random.randn(S, n_tiers - 1) * 0.1,
            "alpha_tvl": np.random.randn(S) * 0.01,
            "B": np.random.randn(S, K_OBS_COEFF, K_cov) * 0.5,
            "sigma_theta": np.ones((S, K_OBS_COEFF)),
            "L_Omega": np.tile(np.eye(K_OBS_COEFF), (S, 1, 1)),
            "eta": np.zeros((S, N_pools, K_OBS_COEFF)),
            "df": np.full((S,), 5.0),
            "sigma_eps": np.tile([0.5, 0.8, 0.6], (S, 1)),
        }

        pool_params = extract_structural_params(structural_samples, data)
        output_path = str(tmp_path / "structural_output.json")
        convergence = {"method": "svi", "final_elbo": 999.0}
        inference_config = {"method": "svi", "svi_steps": 2000}

        generate_output_json(
            pool_params, structural_samples, data,
            convergence, output_path, inference_config,
        )
        with open(output_path) as f:
            result = json.load(f)
        return output_path, result

    def test_output_model_name(self, _structural_output_setup):
        _, data = _structural_output_setup
        assert data["model"] == "structural_mixture"

    def test_output_has_arb_params(self, _structural_output_setup):
        _, data = _structural_output_setup
        pe = data["population_effects"]
        assert "alpha_0" in pe
        assert "alpha_chain" in pe
        assert "alpha_tier" in pe
        assert "alpha_tvl" in pe

    def test_output_has_hierarchical_noise_params(self, _structural_output_setup):
        _, data = _structural_output_setup
        pe = data["population_effects"]
        assert "B" in pe
        assert "sigma_theta" in pe
        assert "correlation_matrix" in pe

    def test_output_pools_have_arb_frequency(self, _structural_output_setup):
        _, data = _structural_output_setup
        pools = data["pools"]
        for pid, entry in pools.items():
            assert "arb_frequency" in entry
            assert isinstance(entry["arb_frequency"], int)
            assert 1 <= entry["arb_frequency"] <= 60
