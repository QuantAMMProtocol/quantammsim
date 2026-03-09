"""Tests for noise_model, _get_theta_samples, _build_model_kwargs, and SVI smoke."""

import numpy as np
import pytest

from quantammsim.noise_calibration import (
    noise_model,
    _get_theta_samples,
    _build_model_kwargs,
    K_COEFF,
)


# ===========================================================================
# TestNoiseModelDefinition
# ===========================================================================


class TestNoiseModelDefinition:
    def test_model_traces_without_error(self, synthetic_encoded_data):
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.handlers as handlers

        data = synthetic_encoded_data
        kwargs = _build_model_kwargs(data)
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(handlers.seed(noise_model, rng_key)).get_trace(
            **kwargs
        )
        assert trace is not None

    def test_required_sites_present(self, synthetic_encoded_data):
        import jax
        import numpyro.handlers as handlers

        data = synthetic_encoded_data
        kwargs = _build_model_kwargs(data)
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(handlers.seed(noise_model, rng_key)).get_trace(
            **kwargs
        )
        required = {"B", "sigma_theta", "L_Omega", "df", "sigma_eps", "eta", "y"}
        assert required.issubset(trace.keys())

    def test_site_shapes(self, synthetic_encoded_data):
        import jax
        import numpyro.handlers as handlers

        data = synthetic_encoded_data
        kwargs = _build_model_kwargs(data)
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(handlers.seed(noise_model, rng_key)).get_trace(
            **kwargs
        )
        N_pools = data["N_pools"]
        K_cov = data["K_cov"]

        assert trace["B"]["value"].shape == (K_COEFF, K_cov)
        assert trace["sigma_theta"]["value"].shape == (K_COEFF,)
        assert trace["L_Omega"]["value"].shape == (K_COEFF, K_COEFF)
        assert trace["df"]["value"].shape == ()
        assert trace["sigma_eps"]["value"].shape == (3,)
        assert trace["eta"]["value"].shape == (N_pools, K_COEFF)

    def test_theta_deterministic_site(self, synthetic_encoded_data):
        import jax
        import numpyro.handlers as handlers

        data = synthetic_encoded_data
        kwargs = _build_model_kwargs(data)
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(handlers.seed(noise_model, rng_key)).get_trace(
            **kwargs
        )
        assert "theta" in trace
        assert trace["theta"]["value"].shape == (data["N_pools"], K_COEFF)

    def test_prior_predictive_produces_y(self, synthetic_encoded_data):
        import jax
        from numpyro.infer import Predictive

        data = synthetic_encoded_data
        kwargs = _build_model_kwargs(data)
        kwargs["y_obs"] = None

        predictive = Predictive(noise_model, num_samples=5)
        rng_key = jax.random.PRNGKey(42)
        samples = predictive(rng_key, **kwargs)
        assert "y" in samples
        assert samples["y"].shape[0] == 5


# ===========================================================================
# TestGetThetaSamples
# ===========================================================================


class TestGetThetaSamples:
    def test_returns_theta_directly_when_present(self, synthetic_encoded_data):
        data = synthetic_encoded_data
        theta_direct = np.random.randn(10, data["N_pools"], K_COEFF)
        sample_dict = {"theta": theta_direct}
        result = _get_theta_samples(sample_dict, data["X_pool"])
        np.testing.assert_array_equal(result, theta_direct)

    def test_reconstructs_from_non_centered(
        self, synthetic_encoded_data, synthetic_samples
    ):
        data = synthetic_encoded_data
        result = _get_theta_samples(synthetic_samples, data["X_pool"])
        assert result.shape == (10, data["N_pools"], K_COEFF)

    def test_eta_zero_identity_gives_mu(
        self, synthetic_encoded_data, synthetic_samples
    ):
        """With eta=0 and L_Omega=I, theta = X_pool @ B^T."""
        data = synthetic_encoded_data
        X_pool = data["X_pool"]
        B = synthetic_samples["B"]

        result = _get_theta_samples(synthetic_samples, X_pool)

        # Expected: mu[s,p,j] = sum_d X_pool[p,d] * B[s,j,d]
        expected = np.einsum("pd,sjd->spj", X_pool, B)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_output_shape(self, synthetic_encoded_data, synthetic_samples):
        data = synthetic_encoded_data
        result = _get_theta_samples(synthetic_samples, data["X_pool"])
        S = synthetic_samples["B"].shape[0]
        assert result.shape == (S, data["N_pools"], K_COEFF)

    def test_reconstructed_matches_direct(self, synthetic_encoded_data):
        """When both theta and raw params are present, reconstruction matches."""
        data = synthetic_encoded_data
        N_pools = data["N_pools"]
        K_cov = data["K_cov"]
        S = 8
        X_pool = data["X_pool"]

        np.random.seed(123)
        B = np.random.randn(S, K_COEFF, K_cov) * 0.3
        sigma_theta = np.abs(np.random.randn(S, K_COEFF)) + 0.1
        # Random lower-triangular L
        L_raw = np.zeros((S, K_COEFF, K_COEFF))
        for s in range(S):
            A = np.random.randn(K_COEFF, K_COEFF)
            L_raw[s] = np.linalg.cholesky(A @ A.T + np.eye(K_COEFF))
        eta = np.random.randn(S, N_pools, K_COEFF)

        sample_dict = {
            "B": B, "sigma_theta": sigma_theta,
            "L_Omega": L_raw, "eta": eta,
        }

        theta_recon = _get_theta_samples(sample_dict, X_pool)

        # Compute expected directly
        mu = np.einsum("pd,sjd->spj", X_pool, B)
        L_Sigma = sigma_theta[:, :, None] * L_raw
        offset = np.einsum("spi,sji->spj", eta, L_Sigma)
        expected = mu + offset

        np.testing.assert_allclose(theta_recon, expected, atol=1e-10)


# ===========================================================================
# TestBuildModelKwargs
# ===========================================================================


class TestBuildModelKwargs:
    def test_all_outputs_are_jnp(self, synthetic_encoded_data):
        import jax.numpy as jnp

        kwargs = _build_model_kwargs(synthetic_encoded_data)
        for key in ["pool_idx", "X_pool", "x_obs", "y_obs", "tier_A_per_pool"]:
            assert isinstance(kwargs[key], jnp.ndarray), (
                f"{key} should be jnp array"
            )

    def test_shapes_preserved(self, synthetic_encoded_data):
        data = synthetic_encoded_data
        kwargs = _build_model_kwargs(data)
        assert kwargs["pool_idx"].shape == data["pool_idx"].shape
        assert kwargs["X_pool"].shape == data["X_pool"].shape
        assert kwargs["x_obs"].shape == data["x_obs"].shape
        assert kwargs["y_obs"].shape == data["y_obs"].shape
        assert kwargs["N_pools"] == data["N_pools"]
        assert kwargs["K_cov"] == data["K_cov"]

    def test_dp_model_kwargs_exclude_tier_A(self, synthetic_encoded_data):
        """When model_fn is noise_model_dp_sigma, tier_A_per_pool is excluded
        and K_clusters is included."""
        from quantammsim.noise_calibration.model import noise_model_dp_sigma

        data = dict(synthetic_encoded_data)
        data["K_clusters"] = 6
        kwargs = _build_model_kwargs(data, model_fn=noise_model_dp_sigma)
        assert "tier_A_per_pool" not in kwargs
        assert kwargs["K_clusters"] == 6

    def test_tier_model_kwargs_include_tier_A(self, synthetic_encoded_data):
        """When model_fn is noise_model (default), tier_A_per_pool is included
        and K_clusters is not."""
        kwargs = _build_model_kwargs(synthetic_encoded_data)
        assert "tier_A_per_pool" in kwargs
        assert "K_clusters" not in kwargs

    def test_default_model_fn_is_noise_model(self, synthetic_encoded_data):
        """Calling without model_fn should behave identically to model_fn=noise_model."""
        kwargs_default = _build_model_kwargs(synthetic_encoded_data)
        kwargs_explicit = _build_model_kwargs(
            synthetic_encoded_data, model_fn=noise_model
        )
        assert set(kwargs_default.keys()) == set(kwargs_explicit.keys())


# ===========================================================================
# TestSVISmoke
# ===========================================================================


class TestSVISmoke:
    @pytest.mark.slow
    def test_svi_converges_small_data(self):
        """SVI on tiny synthetic data: ELBO should decrease."""
        import jax
        import numpyro

        numpyro.enable_x64()

        from quantammsim.noise_calibration import encode_covariates, run_svi

        # Build minimal panel: 5 pools × 20 days
        np.random.seed(42)
        from datetime import date, timedelta

        pools_spec = [
            ("p0", "MAINNET", "WETH,USDC", 0.003, 0, 0),
            ("p1", "ARBITRUM", "BAL,WETH", 0.01, 0, 1),
            ("p2", "BASE", "RATS,WETH", 0.005, 0, 2),
            ("p3", "MAINNET", "LINK,WETH", 0.005, 0, 1),
            ("p4", "ARBITRUM", "AAVE,USDC", 0.003, 0, 1),
        ]
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(21)]
        records = []
        for pid, chain, tokens, fee, ta, tb in pools_spec:
            base_tvl = 14 + np.random.randn() * 0.5
            for d in dates:
                tvl = base_tvl + np.random.randn() * 0.1
                vol = tvl - 2 + np.random.randn() * 0.3
                records.append({
                    "pool_id": pid, "chain": chain, "date": d,
                    "log_volume": vol, "log_tvl": tvl,
                    "volatility": 0.3 + np.random.rand() * 0.2,
                    "weekend": 1.0 if d.weekday() >= 5 else 0.0,
                    "log_fee": np.log(max(fee, 1e-6)),
                    "swap_fee": fee, "tier_A": ta, "tier_B": tb,
                    "tokens": tokens,
                })

        import pandas as pd

        panel = pd.DataFrame(records)
        panel = panel.sort_values(["pool_id", "date"]).reset_index(drop=True)
        panel["log_tvl_lag1"] = panel.groupby("pool_id")["log_tvl"].shift(1)
        panel = panel.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)

        data = encode_covariates(panel)
        samples, losses = run_svi(data, num_steps=2000, lr=1e-3, seed=0,
                                  num_samples=50)

        # ELBO should decrease: mean of last 100 < mean of first 100
        assert np.mean(losses[-100:]) < np.mean(losses[:100])

    @pytest.mark.slow
    def test_svi_samples_have_required_keys(self):
        """SVI output dict has all expected latent variable keys."""
        import jax
        import numpyro

        numpyro.enable_x64()

        from quantammsim.noise_calibration import encode_covariates, run_svi

        np.random.seed(42)
        from datetime import date, timedelta
        import pandas as pd

        pools_spec = [
            ("p0", "MAINNET", "WETH,USDC", 0.003, 0, 0),
            ("p1", "ARBITRUM", "BAL,WETH", 0.01, 0, 1),
        ]
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(15)]
        records = []
        for pid, chain, tokens, fee, ta, tb in pools_spec:
            base_tvl = 14 + np.random.randn() * 0.5
            for d in dates:
                tvl = base_tvl + np.random.randn() * 0.1
                vol = tvl - 2 + np.random.randn() * 0.3
                records.append({
                    "pool_id": pid, "chain": chain, "date": d,
                    "log_volume": vol, "log_tvl": tvl,
                    "volatility": 0.3 + np.random.rand() * 0.2,
                    "weekend": 1.0 if d.weekday() >= 5 else 0.0,
                    "log_fee": np.log(max(fee, 1e-6)),
                    "swap_fee": fee, "tier_A": ta, "tier_B": tb,
                    "tokens": tokens,
                })

        panel = pd.DataFrame(records)
        panel = panel.sort_values(["pool_id", "date"]).reset_index(drop=True)
        panel["log_tvl_lag1"] = panel.groupby("pool_id")["log_tvl"].shift(1)
        panel = panel.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)

        data = encode_covariates(panel)
        samples, _ = run_svi(data, num_steps=500, lr=1e-3, seed=0,
                             num_samples=10)

        required = {"B", "sigma_theta", "L_Omega", "eta", "df", "sigma_eps"}
        assert required.issubset(samples.keys())
