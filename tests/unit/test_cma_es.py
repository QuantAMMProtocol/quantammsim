"""Tests for CMA-ES optimizer — unit tests for the algorithm and integration tests
for the train_on_historic_data pipeline.

Unit tests validate the pure CMA-ES implementation on standard benchmarks.
Integration tests follow the same fixture/pattern as test_bfgs_optimizer.py.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from copy import deepcopy

from quantammsim.training.cma_es import (
    CMAESState,
    default_params,
    init_cmaes,
    ask,
    tell,
    should_stop,
    run_cmaes,
)
from quantammsim.runners.jax_runner_utils import compute_cmaes_population_size
from quantammsim.runners.jax_runners import train_on_historic_data
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.core_simulator.param_utils import recursive_default_set, check_run_fingerprint
from tests.conftest import TEST_DATA_DIR


# ============================================================================
# Unit Tests — Pure CMA-ES Algorithm
# ============================================================================


class TestCMAESAlgorithm:
    """Tests for the CMA-ES core algorithm on standard benchmarks."""

    def test_sphere_convergence(self):
        """Minimise f(x) = sum(x^2) from random init. Should reach < 1e-6."""
        n = 5
        params = default_params(n)
        key = jax.random.key(0)
        key, init_key = jax.random.split(key)
        x0 = jax.random.normal(init_key, shape=(n,)) * 2.0
        state = init_cmaes(x0, sigma=1.0)

        for gen in range(300):
            key, subkey = jax.random.split(key)
            pop = ask(state, subkey, params["lam"])
            fitness = jnp.sum(pop ** 2, axis=1)
            state = tell(state, pop, fitness, params)
            if should_stop(state, tol=1e-12):
                break

        assert state.best_f < 1e-6, f"Sphere: best_f={state.best_f:.2e}, expected < 1e-6"

    def test_rosenbrock_convergence(self):
        """2D Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2. Optimum at (1,1)."""
        n = 2
        params = default_params(n)
        key = jax.random.key(42)
        x0 = jnp.array([-1.0, -1.0])
        state = init_cmaes(x0, sigma=1.0)

        def rosenbrock(pop):
            x, y = pop[:, 0], pop[:, 1]
            return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

        for gen in range(1000):
            key, subkey = jax.random.split(key)
            pop = ask(state, subkey, params["lam"])
            fitness = rosenbrock(pop)
            state = tell(state, pop, fitness, params)
            if should_stop(state, tol=1e-12):
                break

        assert jnp.allclose(state.best_x, jnp.array([1.0, 1.0]), atol=0.1), (
            f"Rosenbrock: best_x={state.best_x}, expected near (1, 1)"
        )

    def test_init_state_shapes(self):
        """init_cmaes returns state with correct shapes."""
        n = 7
        x0 = jnp.zeros(n)
        state = init_cmaes(x0, sigma=0.5)

        assert state.mean.shape == (n,)
        assert state.C.shape == (n, n)
        assert state.p_sigma.shape == (n,)
        assert state.p_c.shape == (n,)
        assert state.eigenvalues.shape == (n,)
        assert state.eigenvectors.shape == (n, n)
        assert state.invsqrt_C.shape == (n, n)
        assert state.gen == 0
        assert state.best_f == jnp.inf

    def test_ask_population_shape(self):
        """ask() returns population with shape (lam, n)."""
        n = 10
        params = default_params(n)
        state = init_cmaes(jnp.zeros(n), sigma=1.0)
        key = jax.random.key(0)

        pop = ask(state, key, params["lam"])
        assert pop.shape == (params["lam"], n)

    def test_tell_updates_state(self):
        """tell() returns a new state with incremented generation."""
        n = 4
        params = default_params(n)
        state = init_cmaes(jnp.ones(n), sigma=1.0)
        key = jax.random.key(0)

        pop = ask(state, key, params["lam"])
        fitness = jnp.sum(pop ** 2, axis=1)
        new_state = tell(state, pop, fitness, params)

        assert new_state.gen == 1
        # Mean should have moved (not identical to initial)
        assert not jnp.allclose(new_state.mean, state.mean)

    def test_default_params_n10(self):
        """Verify default params for n=10: lam=11, mu=5, weights sum to 1."""
        params = default_params(10)
        assert params["lam"] == 4 + int(3 * np.log(10))  # 10
        # Actually: 4 + floor(3 * ln(10)) = 4 + floor(6.908) = 4 + 6 = 10
        assert params["mu"] == params["lam"] // 2
        assert jnp.allclose(jnp.sum(params["weights"]), 1.0, atol=1e-6)

    def test_should_stop_false_at_init(self):
        """A fresh state should not trigger stopping."""
        n = 10
        state = init_cmaes(jnp.zeros(n), sigma=1.0)
        assert not should_stop(state, tol=1e-8)

    def test_run_cmaes_sphere_convergence(self):
        """run_cmaes minimises f(x) = sum(x^2) via lax.while_loop."""
        n = 5
        params = default_params(n)
        key = jax.random.key(0)
        key, init_key = jax.random.split(key)
        x0 = jax.random.normal(init_key, shape=(n,))
        state = init_cmaes(x0, sigma=1.0)

        def eval_fn(pop):
            return jnp.sum(pop ** 2, axis=1)

        final = run_cmaes(state, key, eval_fn, params, n_generations=300, tol=1e-12)
        assert final.best_f < 1e-6, f"best_f={final.best_f:.2e}, expected < 1e-6"

    def test_run_cmaes_matches_python_loop(self):
        """run_cmaes produces identical results to the Python ask/eval/tell loop."""
        n = 5
        params = default_params(n)
        key = jax.random.key(7)
        x0 = jnp.ones(n) * 3.0
        n_gens = 50

        def eval_fn(pop):
            return jnp.sum(pop ** 2, axis=1)

        # Python loop
        state_py = init_cmaes(x0, sigma=1.0)
        key_py = key
        for gen in range(n_gens):
            key_py, subkey = jax.random.split(key_py)
            pop = ask(state_py, subkey, params["lam"])
            fitness = eval_fn(pop)
            state_py = tell(state_py, pop, fitness, params)
            if should_stop(state_py, tol=1e-12):
                break

        # Fused loop
        state_fused = init_cmaes(x0, sigma=1.0)
        state_fused = run_cmaes(state_fused, key, eval_fn, params, n_gens, tol=1e-12)

        assert jnp.allclose(state_py.best_x, state_fused.best_x, atol=1e-10), (
            f"best_x mismatch: py={state_py.best_x}, fused={state_fused.best_x}"
        )
        assert jnp.allclose(state_py.best_f, state_fused.best_f, atol=1e-10), (
            f"best_f mismatch: py={state_py.best_f}, fused={state_fused.best_f}"
        )
        assert int(state_py.gen) == int(state_fused.gen), (
            f"gen mismatch: py={state_py.gen}, fused={state_fused.gen}"
        )

    def test_run_cmaes_early_stop(self):
        """Starting near optimum with tiny sigma triggers early convergence."""
        n = 5
        params = default_params(n)
        key = jax.random.key(0)
        x0 = jnp.ones(n) * 1e-10
        state = init_cmaes(x0, sigma=1e-10)

        def eval_fn(pop):
            return jnp.sum(pop ** 2, axis=1)

        n_generations = 300
        final = run_cmaes(state, key, eval_fn, params, n_generations, tol=1e-8)
        assert int(final.gen) < n_generations, (
            f"Expected early stop but ran all {n_generations} generations"
        )

    def test_run_cmaes_float32_under_x64(self):
        """run_cmaes with float32 state works when x64 mode is enabled.

        Verifies that dtype hardening prevents float64 promotion inside
        lax.while_loop when the global x64 flag differs from state dtype.
        """
        prev = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", True)
            n = 5
            params = default_params(n)
            key = jax.random.key(0)
            x0 = jnp.ones(n, dtype=jnp.float32)
            state = init_cmaes(x0, sigma=1.0)

            # Verify init state is float32
            assert state.mean.dtype == jnp.float32

            def eval_fn(pop):
                return jnp.sum(pop ** 2, axis=1)

            final = run_cmaes(state, key, eval_fn, params, n_generations=50, tol=1e-8)

            # All float fields should remain float32
            assert final.mean.dtype == jnp.float32, f"mean dtype={final.mean.dtype}"
            assert final.sigma.dtype == jnp.float32, f"sigma dtype={final.sigma.dtype}"
            assert final.C.dtype == jnp.float32, f"C dtype={final.C.dtype}"
            assert final.best_f < 1e-2  # convergence check
        finally:
            jax.config.update("jax_enable_x64", prev)


# ============================================================================
# GPU-aware Population Sizing Tests
# ============================================================================


class TestCMAESPopulationSizing:
    """Tests for custom λ in default_params and compute_cmaes_population_size."""

    def test_default_params_custom_lambda(self):
        """default_params(10, lam=24) recomputes all dependent quantities."""
        params = default_params(10, lam=24)
        assert params["lam"] == 24
        assert params["mu"] == 12
        assert params["weights"].shape == (12,)
        assert jnp.allclose(jnp.sum(params["weights"]), 1.0, atol=1e-6)
        # Verify mu_eff is consistent with the new weights (not stale)
        expected_mu_eff = 1.0 / jnp.sum(params["weights"] ** 2)
        assert jnp.allclose(params["mu_eff"], expected_mu_eff, atol=1e-6)

    def test_compute_cmaes_population_size_small_budget(self):
        """Small budget: budget_max < hansen_default → clamp to hansen_default."""
        # budget=40, n_eval=20 → budget_max=2; hansen(14)=4+floor(3*ln(14))=4+7=11
        lam = compute_cmaes_population_size(
            memory_budget=40, n_eval_points=20, n_flat=14,
        )
        assert lam == 11  # Hansen default wins

    def test_compute_cmaes_population_size_large_budget(self):
        """Large budget: budget_max between hansen_default and 10n → use budget_max."""
        # budget=1000, n_eval=20 → budget_max=50; hansen(14)=11; cap=10*14=140
        lam = compute_cmaes_population_size(
            memory_budget=1000, n_eval_points=20, n_flat=14,
        )
        assert lam == 50

    def test_compute_cmaes_population_size_huge_budget(self):
        """Huge budget: fills VRAM (no artificial cap — GPU parallelism makes large λ free)."""
        # budget=50000, n_eval=10 → budget_max=5000; hansen(14)=11
        lam = compute_cmaes_population_size(
            memory_budget=50000, n_eval_points=10, n_flat=14,
        )
        assert lam == 5000  # use full budget

    def test_run_cmaes_with_custom_lambda(self):
        """run_cmaes converges on sphere with custom λ=20."""
        n = 5
        params = default_params(n, lam=20)
        assert params["lam"] == 20
        assert params["mu"] == 10

        key = jax.random.key(0)
        x0 = jnp.ones(n) * 3.0
        state = init_cmaes(x0, sigma=1.0)

        def eval_fn(pop):
            return jnp.sum(pop ** 2, axis=1)

        final = run_cmaes(state, key, eval_fn, params, n_generations=300, tol=1e-12)
        assert final.best_f < 1e-6, f"best_f={final.best_f:.2e}, expected < 1e-6"

    def test_cma_es_config_defaults_include_memory_budget(self):
        """memory_budget default is applied via recursive_default_set."""
        fp = {
            "optimisation_settings": {
                "method": "cma_es",
            }
        }
        recursive_default_set(fp, run_fingerprint_defaults)
        cma = fp["optimisation_settings"]["cma_es_settings"]
        assert "memory_budget" in cma
        assert cma["memory_budget"] is None


# ============================================================================
# Integration Tests — train_on_historic_data pipeline
# ============================================================================


@pytest.fixture
def cma_es_run_fingerprint():
    """Minimal run fingerprint for fast CMA-ES tests.

    Uses 3-day train + 2-day test windows within test data range.
    """
    return {
        "rule": "momentum",
        "tokens": ["ETH", "USDC"],
        "subsidary_pools": [],
        "n_assets": 2,
        "bout_offset": 0,
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "weight_interpolation_method": "linear",
        "maximum_change": 0.0003,
        "minimum_weight": 0.05,
        "max_memory_days": 5.0,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "initial_pool_value": 1000000.0,
        "fees": 0.003,
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "do_arb": True,
        "arb_frequency": 1,
        "return_val": "sharpe",
        "noise_trader_ratio": 0.0,
        "ste_max_change": False,
        "ste_min_max_weight": False,
        "initial_memory_length": 3.0,
        "initial_memory_length_delta": 0.0,
        "initial_k_per_day": 0.5,
        "initial_weights_logits": [0.0, 0.0],
        "initial_log_amplitude": 0.0,
        "initial_raw_width": 0.0,
        "initial_raw_exponents": 1.0,
        "initial_pre_exp_scaling": 1.0,
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-01-04 00:00:00",
        "endTestDateString": "2023-01-06 00:00:00",
        "do_trades": False,
        "optimisation_settings": {
            "method": "cma_es",
            "n_parameter_sets": 1,
            "noise_scale": 0.1,
            "training_data_kind": "historic",
            "initial_random_key": 42,
            "max_mc_version": 1,
            "val_fraction": 0.0,
            "base_lr": 0.01,
            "optimiser": "adam",
            "decay_lr_plateau": 50,
            "decay_lr_ratio": 0.5,
            "min_lr": 0.0001,
            "train_on_hessian_trace": False,
            "n_iterations": 10,
            "cma_es_settings": {
                "n_generations": 10,
                "sigma0": 0.5,
                "tol": 1e-8,
                "n_evaluation_points": 2,
            },
        },
    }


class TestCMAESIntegration:
    """Integration tests for CMA-ES through train_on_historic_data."""

    def test_cma_es_runs_end_to_end(self, cma_es_run_fingerprint):
        """CMA-ES with n_parameter_sets=1 returns a params dict with correct keys."""
        fp = deepcopy(cma_es_run_fingerprint)

        result = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
        )

        assert result is not None
        assert isinstance(result, dict)
        # Momentum pool params should be present
        assert "log_k" in result
        assert "logit_lamb" in result
        # Params should be 1-D (n_assets,) — batch dim selected out
        for k, v in result.items():
            if k == "subsidary_params":
                continue
            if hasattr(v, "shape"):
                assert v.ndim == 1, f"{k} has ndim={v.ndim}, expected 1"

    def test_cma_es_multiple_restarts(self, cma_es_run_fingerprint):
        """Multi-restart CMA-ES with n_parameter_sets=2 returns correct shapes."""
        fp = deepcopy(cma_es_run_fingerprint)
        fp["optimisation_settings"]["n_parameter_sets"] = 2

        result = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
        )

        assert result is not None
        assert isinstance(result, dict)
        # Result should be a single param set (best selected)
        for k, v in result.items():
            if k == "subsidary_params":
                continue
            if hasattr(v, "shape"):
                assert v.ndim == 1, f"{k} has ndim={v.ndim}, expected 1 (selected)"

    def test_cma_es_returns_metadata(self, cma_es_run_fingerprint):
        """return_training_metadata=True returns (params, metadata) with correct structure."""
        fp = deepcopy(cma_es_run_fingerprint)
        fp["optimisation_settings"]["n_parameter_sets"] = 2

        result = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
            return_training_metadata=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        params, metadata = result
        assert isinstance(params, dict)
        assert isinstance(metadata, dict)

        # Check method tag
        assert metadata["method"] == "cma_es"

        # Check required metadata keys (same as BFGS)
        required_keys = [
            "epochs_trained",
            "best_train_metrics",
            "best_continuous_test_metrics",
            "best_param_idx",
            "best_final_reserves",
            "best_final_weights",
            "run_fingerprint",
            "checkpoint_returns",
            "selection_method",
            "selection_metric",
        ]
        for key in required_keys:
            assert key in metadata, f"Missing metadata key: {key}"

        # CMA-ES-specific keys
        assert "generations_per_restart" in metadata
        assert "objective_per_restart" in metadata
        assert len(metadata["generations_per_restart"]) == 2
        assert len(metadata["objective_per_restart"]) == 2

        # Checkpoint returns should be None (CMA-ES doesn't checkpoint)
        assert metadata["checkpoint_returns"] is None

        # best_train_metrics should be a list (one per param set)
        assert isinstance(metadata["best_train_metrics"], list)

    def test_cma_es_config_defaults(self):
        """cma_es_settings defaults are applied via recursive_default_set."""
        fp = {
            "optimisation_settings": {
                "method": "cma_es",
            }
        }
        recursive_default_set(fp, run_fingerprint_defaults)

        cma = fp["optimisation_settings"]["cma_es_settings"]
        assert cma["n_generations"] == 300
        assert cma["sigma0"] == 0.5
        assert cma["tol"] == 1e-8
        assert cma["n_evaluation_points"] == 20
        assert cma["population_size"] is None  # Auto
        assert cma["compute_dtype"] == "float32"

    def test_cma_es_with_validation(self, cma_es_run_fingerprint):
        """CMA-ES with val_fraction > 0 uses best_val selection."""
        fp = deepcopy(cma_es_run_fingerprint)
        # Need longer window so val split exceeds 1 chunk_period (1440 min)
        fp["endDateString"] = "2023-01-15 00:00:00"
        fp["endTestDateString"] = "2023-01-20 00:00:00"
        fp["optimisation_settings"]["val_fraction"] = 0.2
        fp["optimisation_settings"]["n_parameter_sets"] = 2

        params, metadata = train_on_historic_data(
            fp,
            root=TEST_DATA_DIR,
            verbose=False,
            force_init=True,
            return_training_metadata=True,
        )

        assert params is not None
        assert metadata["method"] == "cma_es"
        assert metadata["selection_method"] == "best_val"
        assert metadata["best_val_metrics"] is not None
        assert isinstance(metadata["best_val_metrics"], list)
        assert len(metadata["best_val_metrics"]) == 2
