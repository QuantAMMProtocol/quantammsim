"""Tests for stick_breaking_weights and noise_model_dp_sigma."""

import numpy as np
import pytest

from quantammsim.noise_calibration import K_COEFF
from quantammsim.noise_calibration.constants import K_CLUSTERS_DEFAULT


# ===========================================================================
# TestStickBreakingWeights
# ===========================================================================


class TestStickBreakingWeights:
    def test_sums_to_one(self):
        from quantammsim.noise_calibration.model import stick_breaking_weights
        import jax.numpy as jnp

        v = jnp.array([0.5, 0.3, 0.4, 0.6, 0.2])
        w = stick_breaking_weights(v)
        np.testing.assert_allclose(float(jnp.sum(w)), 1.0, atol=1e-6)

    def test_correct_length(self):
        from quantammsim.noise_calibration.model import stick_breaking_weights
        import jax.numpy as jnp

        K = 7
        v = jnp.ones(K - 1) * 0.3
        w = stick_breaking_weights(v)
        assert w.shape == (K,)

    def test_non_negative(self):
        from quantammsim.noise_calibration.model import stick_breaking_weights
        import jax.numpy as jnp

        v = jnp.array([0.1, 0.9, 0.5, 0.7, 0.3])
        w = stick_breaking_weights(v)
        assert jnp.all(w >= 0.0)

    def test_first_weight_equals_first_v(self):
        from quantammsim.noise_calibration.model import stick_breaking_weights
        import jax.numpy as jnp

        v = jnp.array([0.7, 0.4, 0.2])
        w = stick_breaking_weights(v)
        np.testing.assert_allclose(float(w[0]), 0.7, atol=1e-6)

    def test_jit_compatible(self):
        from quantammsim.noise_calibration.model import stick_breaking_weights
        import jax
        import jax.numpy as jnp

        v = jnp.array([0.5, 0.3, 0.4])
        w_eager = stick_breaking_weights(v)
        w_jit = jax.jit(stick_breaking_weights)(v)
        np.testing.assert_allclose(
            np.array(w_eager), np.array(w_jit), atol=1e-6
        )

    def test_all_v_one_concentrates_on_first(self):
        """If v = [1, 1, ...], all mass goes to first component."""
        from quantammsim.noise_calibration.model import stick_breaking_weights
        import jax.numpy as jnp

        v = jnp.ones(5)
        w = stick_breaking_weights(v)
        np.testing.assert_allclose(float(w[0]), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(jnp.sum(w[1:])), 0.0, atol=1e-6)


# ===========================================================================
# TestDPModelDefinition
# ===========================================================================


class TestDPModelDefinition:
    def _get_dp_model_kwargs(self, data, K_clusters=6):
        """Build kwargs for noise_model_dp_sigma from encoded data."""
        import jax.numpy as jnp

        return dict(
            pool_idx=jnp.array(data["pool_idx"]),
            X_pool=jnp.array(data["X_pool"]),
            x_obs=jnp.array(data["x_obs"]),
            y_obs=jnp.array(data["y_obs"]),
            N_pools=data["N_pools"],
            K_coeff=K_COEFF,
            K_cov=data["K_cov"],
            K_clusters=K_clusters,
        )

    def test_model_traces_without_error(self, synthetic_encoded_data):
        from quantammsim.noise_calibration.model import noise_model_dp_sigma
        import jax
        import numpyro.handlers as handlers

        kwargs = self._get_dp_model_kwargs(synthetic_encoded_data)
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(
            handlers.seed(noise_model_dp_sigma, rng_key)
        ).get_trace(**kwargs)
        assert trace is not None

    def test_has_dp_sites(self, synthetic_encoded_data):
        from quantammsim.noise_calibration.model import noise_model_dp_sigma
        import jax
        import numpyro.handlers as handlers

        kwargs = self._get_dp_model_kwargs(synthetic_encoded_data)
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(
            handlers.seed(noise_model_dp_sigma, rng_key)
        ).get_trace(**kwargs)

        dp_sites = {"alpha_dp", "v", "sigma_eps", "log_lik"}
        assert dp_sites.issubset(trace.keys()), (
            f"Missing DP sites: {dp_sites - trace.keys()}"
        )

    def test_no_y_site_when_obs_provided(self, synthetic_encoded_data):
        """With y_obs provided, the marginalized model uses factor, not obs."""
        from quantammsim.noise_calibration.model import noise_model_dp_sigma
        import jax
        import numpyro.handlers as handlers

        kwargs = self._get_dp_model_kwargs(synthetic_encoded_data)
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(
            handlers.seed(noise_model_dp_sigma, rng_key)
        ).get_trace(**kwargs)

        assert "y" not in trace, (
            "DP model should use numpyro.factor, not obs=y_obs"
        )

    def test_shared_mean_structure_sites(self, synthetic_encoded_data):
        """The DP model must share the same mean structure as the tier model."""
        from quantammsim.noise_calibration.model import noise_model_dp_sigma
        import jax
        import numpyro.handlers as handlers

        kwargs = self._get_dp_model_kwargs(synthetic_encoded_data)
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(
            handlers.seed(noise_model_dp_sigma, rng_key)
        ).get_trace(**kwargs)

        shared_sites = {"B", "sigma_theta", "L_Omega", "df", "eta", "theta"}
        assert shared_sites.issubset(trace.keys()), (
            f"Missing shared sites: {shared_sites - trace.keys()}"
        )

    def test_correct_shapes(self, synthetic_encoded_data):
        from quantammsim.noise_calibration.model import noise_model_dp_sigma
        import jax
        import numpyro.handlers as handlers

        K_clusters = 6
        kwargs = self._get_dp_model_kwargs(
            synthetic_encoded_data, K_clusters=K_clusters
        )
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(
            handlers.seed(noise_model_dp_sigma, rng_key)
        ).get_trace(**kwargs)

        N_pools = synthetic_encoded_data["N_pools"]
        K_cov = synthetic_encoded_data["K_cov"]

        assert trace["B"]["value"].shape == (K_COEFF, K_cov)
        assert trace["sigma_theta"]["value"].shape == (K_COEFF,)
        assert trace["L_Omega"]["value"].shape == (K_COEFF, K_COEFF)
        assert trace["df"]["value"].shape == ()
        assert trace["sigma_eps"]["value"].shape == (K_clusters,)
        assert trace["eta"]["value"].shape == (N_pools, K_COEFF)
        assert trace["v"]["value"].shape == (K_clusters - 1,)
        assert trace["alpha_dp"]["value"].shape == ()

    def test_no_tier_A_per_pool_in_signature(self):
        """The DP model should not accept tier_A_per_pool."""
        from quantammsim.noise_calibration.model import noise_model_dp_sigma
        import inspect

        sig = inspect.signature(noise_model_dp_sigma)
        assert "tier_A_per_pool" not in sig.parameters

    def test_prior_predictive_produces_y(self, synthetic_encoded_data):
        """With y_obs=None, the model should sample y explicitly."""
        from quantammsim.noise_calibration.model import noise_model_dp_sigma
        import jax
        from numpyro.infer import Predictive

        kwargs = self._get_dp_model_kwargs(synthetic_encoded_data)
        kwargs["y_obs"] = None

        predictive = Predictive(noise_model_dp_sigma, num_samples=5)
        rng_key = jax.random.PRNGKey(42)
        samples = predictive(rng_key, **kwargs)
        assert "y" in samples
        assert samples["y"].shape[0] == 5

    def test_k_clusters_configurable(self, synthetic_encoded_data):
        """K_clusters=4 should produce different sigma_eps shape."""
        from quantammsim.noise_calibration.model import noise_model_dp_sigma
        import jax
        import numpyro.handlers as handlers

        K_clusters = 4
        kwargs = self._get_dp_model_kwargs(
            synthetic_encoded_data, K_clusters=K_clusters
        )
        rng_key = jax.random.PRNGKey(0)

        trace = handlers.trace(
            handlers.seed(noise_model_dp_sigma, rng_key)
        ).get_trace(**kwargs)

        assert trace["sigma_eps"]["value"].shape == (K_clusters,)
        assert trace["v"]["value"].shape == (K_clusters - 1,)


# ===========================================================================
# TestDPModelSVISmoke
# ===========================================================================


class TestDPModelSVISmoke:
    def _build_dp_panel(self):
        """Build a minimal panel for DP SVI testing."""
        from datetime import date, timedelta
        import pandas as pd

        np.random.seed(42)
        pools_spec = [
            ("p0", "MAINNET", "WETH,USDC", 0.003, 0, 0),
            ("p1", "ARBITRUM", "BAL,WETH", 0.01, 0, 1),
            ("p2", "BASE", "RATS,WETH", 0.005, 0, 2),
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
        return panel

    def test_svi_converges_dp_model(self):
        """SVI on DP model with tiny data: ELBO should decrease."""
        import numpyro
        numpyro.enable_x64()

        from quantammsim.noise_calibration import encode_covariates, run_svi
        from quantammsim.noise_calibration.model import noise_model_dp_sigma

        panel = self._build_dp_panel()
        data = encode_covariates(panel, include_tiers=False)
        data["K_clusters"] = 4

        samples, losses = run_svi(
            data, num_steps=2000, lr=1e-3, seed=0,
            num_samples=50, model_fn=noise_model_dp_sigma,
        )
        assert np.mean(losses[-100:]) < np.mean(losses[:100])

    def test_svi_samples_have_dp_keys(self):
        """SVI output for DP model has v, alpha_dp, sigma_eps."""
        import numpyro
        numpyro.enable_x64()

        from quantammsim.noise_calibration import encode_covariates, run_svi
        from quantammsim.noise_calibration.model import noise_model_dp_sigma

        panel = self._build_dp_panel()
        data = encode_covariates(panel, include_tiers=False)
        data["K_clusters"] = 4

        samples, _ = run_svi(
            data, num_steps=500, lr=1e-3, seed=0,
            num_samples=10, model_fn=noise_model_dp_sigma,
        )
        required = {"B", "sigma_theta", "L_Omega", "eta", "df",
                     "sigma_eps", "v", "alpha_dp"}
        assert required.issubset(samples.keys()), (
            f"Missing keys: {required - samples.keys()}"
        )
