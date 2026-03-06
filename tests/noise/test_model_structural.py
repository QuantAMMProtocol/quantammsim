"""Tests for structural_noise_model definition and SVI integration."""

import numpy as np
import pytest


class TestStructuralModelDefinition:
    """Tests for the structural_noise_model numpyro model."""

    def test_model_traces_without_error(self, synthetic_structural_data):
        import jax
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=2)
        rng_key = jax.random.PRNGKey(0)
        samples = predictive(rng_key, **kwargs)
        assert "y" in samples

    def test_required_sites_present(self, synthetic_structural_data):
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=2)
        samples = predictive(jax.random.PRNGKey(0), **kwargs)
        required = {
            "alpha_0", "alpha_chain", "alpha_tier", "alpha_tvl",
            "W_gate", "beta", "df", "sigma_eps", "y",
        }
        assert required.issubset(samples.keys()), (
            f"Missing sites: {required - samples.keys()}"
        )

    def test_no_eta_sigma_theta_L_Omega(self, synthetic_structural_data):
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=2)
        samples = predictive(jax.random.PRNGKey(0), **kwargs)
        old_sites = {"eta", "sigma_theta", "L_Omega", "theta", "B"}
        for site in old_sites:
            assert site not in samples, f"Old site '{site}' should not be present"

    def test_W_gate_shape(self, synthetic_structural_data):
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=3)
        samples = predictive(jax.random.PRNGKey(0), **kwargs)

        K_pool_cov = synthetic_structural_data["K_cov"]
        K_archetypes = 3  # default
        assert samples["W_gate"].shape == (3, K_pool_cov, K_archetypes)

    def test_beta_shape(self, synthetic_structural_data):
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=3)
        samples = predictive(jax.random.PRNGKey(0), **kwargs)

        from quantammsim.noise_calibration.constants import K_OBS_COEFF
        K_archetypes = 3
        assert samples["beta"].shape == (3, K_archetypes, K_OBS_COEFF)

    def test_alpha_chain_shape(self, synthetic_structural_data):
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=3)
        samples = predictive(jax.random.PRNGKey(0), **kwargs)

        n_chains = synthetic_structural_data["n_chains"]
        assert samples["alpha_chain"].shape == (3, n_chains - 1)

    def test_prior_predictive_produces_y(self, synthetic_structural_data):
        """y_obs=None path produces y samples."""
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        kwargs["y_obs"] = None
        predictive = Predictive(structural_noise_model, num_samples=10)
        samples = predictive(jax.random.PRNGKey(42), **kwargs)
        assert "y" in samples
        assert samples["y"].shape[0] == 10

    def test_prior_predictive_range_reasonable(self, synthetic_structural_data):
        """Prior y median should be in a plausible range for log(volume).

        The tails can be wide (V_noise = exp(beta @ x_obs) with large
        covariates), but the central mass should be reasonable.
        """
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        kwargs["y_obs"] = None
        predictive = Predictive(structural_noise_model, num_samples=200)
        samples = predictive(jax.random.PRNGKey(42), **kwargs)
        y = np.array(samples["y"])
        median = np.median(y)
        p25 = np.percentile(y, 25)
        p75 = np.percentile(y, 75)
        # Median should be in a plausible log-volume range
        assert median > -20, f"Prior y median too low: {median}"
        assert median < 50, f"Prior y median too high: {median}"
        # IQR should not be enormous
        iqr = p75 - p25
        assert iqr < 500, f"Prior y IQR too wide: {iqr}"

    def test_K_archetypes_configurable(self, synthetic_structural_data):
        """K=2 and K=4 both work."""
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        for K in [2, 4]:
            kwargs = _build_model_kwargs(synthetic_structural_data,
                                         model_fn=structural_noise_model)
            kwargs["K_archetypes"] = K
            predictive = Predictive(structural_noise_model, num_samples=2)
            samples = predictive(jax.random.PRNGKey(0), **kwargs)
            assert samples["W_gate"].shape[-1] == K
            assert samples["beta"].shape[1] == K


class TestSVIStructural:
    """SVI convergence tests for structural model."""

    @pytest.mark.slow
    def test_svi_structural_converges(self, synthetic_structural_data):
        """2000 SVI steps, ELBO should decrease."""
        from quantammsim.noise_calibration.inference import run_svi
        from quantammsim.noise_calibration.model import structural_noise_model

        samples, losses = run_svi(
            synthetic_structural_data,
            num_steps=2000,
            lr=5e-3,
            seed=0,
            num_samples=10,
            model_fn=structural_noise_model,
        )
        assert losses[-100:].mean() < losses[:100].mean()

    @pytest.mark.slow
    def test_svi_structural_samples_have_required_keys(
        self, synthetic_structural_data,
    ):
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
        required = {"W_gate", "beta", "alpha_0", "alpha_chain",
                    "alpha_tier", "alpha_tvl", "df", "sigma_eps"}
        assert required.issubset(samples.keys()), (
            f"Missing keys: {required - samples.keys()}"
        )

    @pytest.mark.slow
    def test_svi_structural_no_eta_keys(self, synthetic_structural_data):
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
        old_keys = {"eta", "sigma_theta", "L_Omega", "B"}
        for key in old_keys:
            assert key not in samples, f"Old key '{key}' should not be in samples"
