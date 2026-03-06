"""Tests for encode_covariates and encode_covariates_structural."""

import numpy as np
import pytest

from quantammsim.noise_calibration import encode_covariates
from quantammsim.noise_calibration.constants import K_OBS_COEFF, OBS_COEFF_NAMES


class TestEncodeCovariates:
    def test_x_pool_shape(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        N_pools = data["N_pools"]
        K_cov = data["K_cov"]
        assert data["X_pool"].shape == (N_pools, K_cov)

    def test_x_obs_shape(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        N_obs = len(synthetic_panel)
        assert data["x_obs"].shape == (N_obs, 4)

    def test_x_obs_column_0_is_intercept(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        np.testing.assert_array_equal(data["x_obs"][:, 0], 1.0)

    def test_x_obs_column_1_is_lagged_tvl(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        np.testing.assert_array_equal(
            data["x_obs"][:, 1],
            synthetic_panel["log_tvl_lag1"].values,
        )

    def test_x_obs_column_2_is_volatility(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        np.testing.assert_array_equal(
            data["x_obs"][:, 2],
            synthetic_panel["volatility"].values,
        )

    def test_x_obs_column_3_is_weekend(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        np.testing.assert_array_equal(
            data["x_obs"][:, 3],
            synthetic_panel["weekend"].values,
        )

    def test_intercept_column_all_ones(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        np.testing.assert_array_equal(data["X_pool"][:, 0], 1.0)

    def test_chain_dummies_one_hot(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        col_names = data["covariate_names"]
        chain_cols = [i for i, n in enumerate(col_names) if n.startswith("chain_")]
        X = data["X_pool"]

        for row in range(X.shape[0]):
            chain_vals = X[row, chain_cols]
            assert chain_vals.sum() <= 1.0

    def test_reference_chain_is_alphabetically_first(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        chains = sorted(synthetic_panel["chain"].unique())
        ref_chain = chains[0]  # ARBITRUM

        pool_meta = data["pool_meta"]
        ref_idx = pool_meta[pool_meta["chain"] == ref_chain].index[0]
        col_names = data["covariate_names"]
        chain_cols = [i for i, n in enumerate(col_names) if n.startswith("chain_")]
        X = data["X_pool"]
        assert all(X[ref_idx, c] == 0.0 for c in chain_cols)

    def test_tier_a_dummies_match(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        col_names = data["covariate_names"]
        tier_a_cols = [
            (i, n) for i, n in enumerate(col_names) if n.startswith("tier_A_")
        ]
        pool_meta = data["pool_meta"]
        X = data["X_pool"]

        for idx, row in pool_meta.iterrows():
            tier_a_str = str(row["tier_A"])
            for col_idx, col_name in tier_a_cols:
                expected_tier = col_name.split("_")[-1]
                expected = 1.0 if tier_a_str == expected_tier else 0.0
                assert X[idx, col_idx] == expected, (
                    f"Pool {idx} tier_A={tier_a_str}, col {col_name}: "
                    f"expected {expected}, got {X[idx, col_idx]}"
                )

    def test_tier_b_dummies_match(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        col_names = data["covariate_names"]
        tier_b_cols = [
            (i, n) for i, n in enumerate(col_names) if n.startswith("tier_B_")
        ]
        pool_meta = data["pool_meta"]
        X = data["X_pool"]

        for idx, row in pool_meta.iterrows():
            tier_b_str = str(row["tier_B"])
            for col_idx, col_name in tier_b_cols:
                expected_tier = col_name.split("_")[-1]
                expected = 1.0 if tier_b_str == expected_tier else 0.0
                assert X[idx, col_idx] == expected

    def test_log_fee_column(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        col_names = data["covariate_names"]
        fee_idx = col_names.index("log_fee")
        pool_meta = data["pool_meta"]

        for idx, row in pool_meta.iterrows():
            expected = np.log(max(row["swap_fee"], 1e-6))
            np.testing.assert_allclose(
                data["X_pool"][idx, fee_idx], expected, rtol=1e-10,
            )

    def test_pool_idx_maps_observations(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        pool_ids = data["pool_ids"]
        pool_idx = data["pool_idx"]

        assert pool_idx.min() >= 0
        assert pool_idx.max() < len(pool_ids)

        for i, pid in enumerate(pool_ids):
            mask = synthetic_panel["pool_id"] == pid
            obs_indices = np.where(mask.values)[0]
            assert (pool_idx[obs_indices] == i).all()

    def test_y_obs_matches_panel(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        np.testing.assert_array_equal(
            data["y_obs"],
            synthetic_panel["log_volume"].values,
        )

    def test_covariate_names_length(self, synthetic_panel):
        data = encode_covariates(synthetic_panel)
        assert len(data["covariate_names"]) == data["K_cov"]

    def test_covariate_column_ordering(self, synthetic_panel):
        """X_pool columns must follow: intercept, chain dummies, tier_A dummies,
        tier_B dummies, log_fee. This ordering is load-bearing because B is
        indexed by column position."""
        data = encode_covariates(synthetic_panel)
        names = data["covariate_names"]

        assert names[0] == "intercept"
        assert names[-1] == "log_fee"

        # Find boundaries
        chain_start = None
        tier_a_start = None
        tier_b_start = None
        fee_idx = len(names) - 1

        for i, n in enumerate(names):
            if n.startswith("chain_") and chain_start is None:
                chain_start = i
            if n.startswith("tier_A_") and tier_a_start is None:
                tier_a_start = i
            if n.startswith("tier_B_") and tier_b_start is None:
                tier_b_start = i

        # Verify ordering: intercept < chains < tier_A < tier_B < log_fee
        if chain_start is not None:
            assert chain_start > 0  # after intercept
        if tier_a_start is not None and chain_start is not None:
            assert tier_a_start > chain_start
        if tier_b_start is not None and tier_a_start is not None:
            assert tier_b_start > tier_a_start
        if tier_b_start is not None:
            assert fee_idx > tier_b_start

        # All chain dummies are contiguous
        chain_names = [n for n in names if n.startswith("chain_")]
        if chain_names:
            chain_indices = [names.index(n) for n in chain_names]
            assert chain_indices == list(range(min(chain_indices),
                                               max(chain_indices) + 1))

    def test_output_dict_has_all_required_keys(self, synthetic_panel):
        """encode_covariates must return all keys consumed by downstream
        functions (predict_new_pool, generate_output_json, _save_sample_cache)."""
        data = encode_covariates(synthetic_panel)
        required_keys = {
            "pool_idx", "X_pool", "x_obs", "y_obs", "pool_ids", "pool_meta",
            "covariate_names", "tier_A_per_pool", "N_pools", "K_cov",
            "ref_chain", "ref_tier_a", "ref_tier_b", "chains",
        }
        assert required_keys.issubset(data.keys()), (
            f"Missing keys: {required_keys - data.keys()}"
        )


class TestEncodeCovariatesNoTiers:
    """Tests for encode_covariates(include_tiers=False)."""

    def test_no_tier_columns_in_covariate_names(self, synthetic_panel):
        data = encode_covariates(synthetic_panel, include_tiers=False)
        for name in data["covariate_names"]:
            assert not name.startswith("tier_A_"), (
                f"Found tier_A column {name} with include_tiers=False"
            )
            assert not name.startswith("tier_B_"), (
                f"Found tier_B column {name} with include_tiers=False"
            )

    def test_still_has_intercept_and_log_fee(self, synthetic_panel):
        data = encode_covariates(synthetic_panel, include_tiers=False)
        assert "intercept" in data["covariate_names"]
        assert "log_fee" in data["covariate_names"]

    def test_still_has_chain_dummies(self, synthetic_panel):
        data = encode_covariates(synthetic_panel, include_tiers=False)
        chain_cols = [n for n in data["covariate_names"]
                      if n.startswith("chain_")]
        assert len(chain_cols) > 0

    def test_k_cov_smaller_than_with_tiers(self, synthetic_panel):
        data_tiers = encode_covariates(synthetic_panel, include_tiers=True)
        data_no_tiers = encode_covariates(synthetic_panel, include_tiers=False)
        assert data_no_tiers["K_cov"] < data_tiers["K_cov"]

    def test_default_is_include_tiers_true(self, synthetic_panel):
        data_default = encode_covariates(synthetic_panel)
        data_explicit = encode_covariates(synthetic_panel, include_tiers=True)
        assert data_default["K_cov"] == data_explicit["K_cov"]

    def test_tier_A_per_pool_still_present(self, synthetic_panel):
        """tier_A_per_pool is still returned (for other downstream uses)."""
        data = encode_covariates(synthetic_panel, include_tiers=False)
        assert "tier_A_per_pool" in data

    def test_x_pool_shape_matches_k_cov(self, synthetic_panel):
        data = encode_covariates(synthetic_panel, include_tiers=False)
        assert data["X_pool"].shape == (data["N_pools"], data["K_cov"])


class TestEncodeStructuralCovariates:
    """Tests for encode_covariates_structural()."""

    @pytest.fixture()
    def struct_data(self, synthetic_panel):
        from quantammsim.noise_calibration.covariate_encoding import (
            encode_covariates_structural,
        )
        return encode_covariates_structural(synthetic_panel)

    def test_encode_structural_x_obs_shape(self, struct_data, synthetic_panel):
        N_obs = len(synthetic_panel)
        assert struct_data["x_obs"].shape == (N_obs, K_OBS_COEFF)

    def test_encode_structural_x_obs_columns(self, struct_data, synthetic_panel):
        """Columns must match OBS_COEFF_NAMES ordering:
        [1, lag_log_tvl, log_sigma, tvl_x_sigma, tvl_x_fee, sigma_x_fee,
         dow_sin, dow_cos]."""
        x = struct_data["x_obs"]
        np.testing.assert_array_equal(x[:, 0], 1.0)  # intercept
        np.testing.assert_array_equal(
            x[:, 1], synthetic_panel["log_tvl_lag1"].values,
        )
        np.testing.assert_array_equal(
            x[:, 2], synthetic_panel["log_sigma"].values,
        )
        np.testing.assert_allclose(
            x[:, 3], synthetic_panel["tvl_x_sigma"].values, rtol=1e-12,
        )
        np.testing.assert_allclose(
            x[:, 4], synthetic_panel["tvl_x_fee"].values, rtol=1e-12,
        )
        np.testing.assert_allclose(
            x[:, 5], synthetic_panel["sigma_x_fee"].values, rtol=1e-12,
        )
        np.testing.assert_allclose(
            x[:, 6], synthetic_panel["dow_sin"].values, rtol=1e-12,
        )
        np.testing.assert_allclose(
            x[:, 7], synthetic_panel["dow_cos"].values, rtol=1e-12,
        )

    def test_encode_structural_has_sigma_daily(self, struct_data):
        """sigma_daily = volatility / sqrt(365), de-annualised."""
        assert "sigma_daily" in struct_data
        assert len(struct_data["sigma_daily"]) > 0

    def test_encode_structural_has_gas(self, struct_data):
        assert "gas" in struct_data
        assert len(struct_data["gas"]) > 0
        assert (struct_data["gas"] >= 0).all()

    def test_encode_structural_has_chain_idx_tier_idx(self, struct_data):
        assert "chain_idx" in struct_data
        assert "tier_idx" in struct_data
        assert struct_data["chain_idx"].dtype in (np.int32, np.int64)
        assert struct_data["tier_idx"].dtype in (np.int32, np.int64)

    def test_encode_structural_has_fee(self, struct_data):
        """fee array is raw (not log), for the formula."""
        assert "fee" in struct_data
        assert (struct_data["fee"] > 0).all()
        assert (struct_data["fee"] < 1).all()  # fees are fractions

    def test_encode_structural_tier_idx_is_pair(self, struct_data):
        """tier_idx encodes the (tier_A, tier_B) PAIR, not individual tokens.
        (0,0)->0, (0,1)->1, (0,2)->2, (1,1)->3, (1,2)->4, (2,2)->5."""
        tier_idx = struct_data["tier_idx"]
        pool_meta = struct_data["pool_meta"]

        for i, row in pool_meta.iterrows():
            a, b = int(row["tier_A"]), int(row["tier_B"])
            expected = a * (5 - a) // 2 + b - a
            assert tier_idx[i] == expected, (
                f"Pool {i}: tier ({a},{b}) expected idx {expected}, "
                f"got {tier_idx[i]}"
            )

    def test_encode_structural_n_chains_n_tiers(self, struct_data):
        """n_chains and n_tiers computed from data."""
        assert "n_chains" in struct_data
        assert "n_tiers" in struct_data
        assert struct_data["n_chains"] >= 1
        assert struct_data["n_tiers"] >= 1
