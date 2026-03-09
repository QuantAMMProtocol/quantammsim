"""Tests for structural_noise_model definition and SVI integration."""

import numpy as np
import pytest


class TestStructuralModelDefinition:
    """Tests for the structural_noise_model numpyro model."""

    def test_model_traces_without_error(self, synthetic_structural_data):
        import jax
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
            "B", "sigma_theta", "L_Omega", "eta", "theta",
            "df", "sigma_eps", "y",
        }
        assert required.issubset(samples.keys()), (
            f"Missing sites: {required - samples.keys()}"
        )

    def test_no_moe_sites(self, synthetic_structural_data):
        """MoE sites (W_gate, beta) should not be present."""
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=2)
        samples = predictive(jax.random.PRNGKey(0), **kwargs)
        for site in ("W_gate", "beta"):
            assert site not in samples, f"MoE site '{site}' should not be present"

    def test_theta_shape(self, synthetic_structural_data):
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=3)
        samples = predictive(jax.random.PRNGKey(0), **kwargs)

        N_pools = synthetic_structural_data["N_pools"]
        K_obs = synthetic_structural_data["x_obs"].shape[1]
        assert samples["theta"].shape == (3, N_pools, K_obs)

    def test_B_shape(self, synthetic_structural_data):
        import jax
        from numpyro.infer import Predictive
        from quantammsim.noise_calibration.model import structural_noise_model
        from quantammsim.noise_calibration.inference import _build_model_kwargs

        kwargs = _build_model_kwargs(synthetic_structural_data,
                                     model_fn=structural_noise_model)
        predictive = Predictive(structural_noise_model, num_samples=3)
        samples = predictive(jax.random.PRNGKey(0), **kwargs)

        K_obs = synthetic_structural_data["x_obs"].shape[1]
        K_cov = synthetic_structural_data["K_cov"]
        assert samples["B"].shape == (3, K_obs, K_cov)

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
        """Prior y should contain finite values in a plausible range."""
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
        finite = y[np.isfinite(y)]
        assert len(finite) > 0.5 * y.size, "Too many non-finite prior samples"
        median = np.median(finite)
        assert median > -50, f"Prior y median too low: {median}"
        assert median < 100, f"Prior y median too high: {median}"


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
        required = {"B", "sigma_theta", "L_Omega", "eta",
                    "alpha_0", "alpha_chain",
                    "alpha_tier", "alpha_tvl", "df", "sigma_eps"}
        assert required.issubset(samples.keys()), (
            f"Missing keys: {required - samples.keys()}"
        )

    @pytest.mark.slow
    def test_svi_structural_no_moe_keys(self, synthetic_structural_data):
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
        for key in ("W_gate", "beta"):
            assert key not in samples, f"MoE key '{key}' should not be in samples"
