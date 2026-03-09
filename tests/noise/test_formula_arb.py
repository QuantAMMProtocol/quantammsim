"""Tests for JAX-differentiable LVR formula."""

import numpy as np
import pytest


class TestFormulaArbJax:
    @pytest.fixture(autouse=True)
    def _import(self):
        from quantammsim.noise_calibration.formula_arb import (
            formula_arb_volume_daily_jax,
        )
        self.formula = formula_arb_volume_daily_jax

    def test_formula_arb_zero_vol_returns_zero(self):
        import jax.numpy as jnp
        result = self.formula(
            sigma_daily=jnp.float64(0.0),
            tvl=jnp.float64(1e6),
            fee=jnp.float64(0.003),
            gas_usd=jnp.float64(1.0),
            cadence_minutes=jnp.float64(1.0),
        )
        assert float(result) == pytest.approx(0.0, abs=1e-10)

    def test_formula_arb_zero_tvl_returns_zero(self):
        import jax.numpy as jnp
        result = self.formula(
            sigma_daily=jnp.float64(0.03),
            tvl=jnp.float64(0.0),
            fee=jnp.float64(0.003),
            gas_usd=jnp.float64(1.0),
            cadence_minutes=jnp.float64(1.0),
        )
        assert float(result) == pytest.approx(0.0, abs=1e-10)

    def test_formula_arb_quadratic_in_sigma(self):
        """V_arb(2σ) / V_arb(σ) ≈ 4 for small gas (correction ≈ 1)."""
        import jax.numpy as jnp
        sigma = jnp.float64(0.01)
        tvl = jnp.float64(1e8)  # large TVL so gas is negligible
        fee = jnp.float64(0.003)
        gas = jnp.float64(0.001)  # tiny gas
        cadence = jnp.float64(0.01)  # very fast arb

        v1 = float(self.formula(sigma, tvl, fee, gas, cadence))
        v2 = float(self.formula(2.0 * sigma, tvl, fee, gas, cadence))
        assert v1 > 0
        ratio = v2 / v1
        assert ratio == pytest.approx(4.0, rel=0.1)

    def test_formula_arb_linear_in_tvl(self):
        """V_arb(2V) / V_arb(V) ≈ 2 for small gas."""
        import jax.numpy as jnp
        sigma = jnp.float64(0.02)
        tvl = jnp.float64(1e8)
        fee = jnp.float64(0.003)
        gas = jnp.float64(0.0001)
        cadence = jnp.float64(0.01)

        v1 = float(self.formula(sigma, tvl, fee, gas, cadence))
        v2 = float(self.formula(sigma, 2.0 * tvl, fee, gas, cadence))
        assert v1 > 0
        ratio = v2 / v1
        assert ratio == pytest.approx(2.0, rel=0.1)

    def test_formula_arb_gas_kills_small_pools(self):
        """High gas, small TVL → V_arb ≈ 0."""
        import jax.numpy as jnp
        result = self.formula(
            sigma_daily=jnp.float64(0.02),
            tvl=jnp.float64(1000.0),
            fee=jnp.float64(0.003),
            gas_usd=jnp.float64(1000.0),
            cadence_minutes=jnp.float64(1.0),
        )
        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_formula_arb_matches_numpy_reference(self):
        """Compare to the numpy formula in plot_formula_arb_vs_real.py."""
        import jax.numpy as jnp

        # Reference implementation (from plot_formula_arb_vs_real.py:58)
        def ref(sigma_daily, tvl, fee, block_time_s, gas_usd):
            if tvl <= 0 or fee <= 0 or sigma_daily <= 0:
                return 0.0
            gamma = fee
            delta = 2.0 * np.sqrt(2.0 * gas_usd / tvl) if gas_usd > 0 else 0.0
            bLVR = sigma_daily**2 * tvl / 8.0
            sqrt_s2_2l = sigma_daily * np.sqrt(block_time_s / (2.0 * 86400.0))
            bFEE = bLVR * max(
                1.0 - delta / (2.0 * gamma) - sqrt_s2_2l / (gamma + delta / 2.0),
                0.0,
            )
            return bFEE / gamma

        test_cases = [
            (0.03, 1e6, 0.003, 1.0, 1.0),
            (0.05, 5e5, 0.01, 0.5, 0.005),
            (0.01, 1e7, 0.005, 2.0, 0.01),
            (0.1, 1e4, 0.03, 10.0, 5.0),
            (0.02, 1e5, 0.001, 1.0, 0.001),
        ]

        for sigma, tvl, fee, cadence_min, gas in test_cases:
            block_time_s = cadence_min * 60.0
            expected = ref(sigma, tvl, fee, block_time_s, gas)
            actual = float(self.formula(
                jnp.float64(sigma), jnp.float64(tvl),
                jnp.float64(fee), jnp.float64(gas),
                jnp.float64(cadence_min),
            ))
            np.testing.assert_allclose(
                actual, expected, rtol=1e-10,
                err_msg=f"Mismatch for sigma={sigma}, tvl={tvl}, "
                        f"fee={fee}, cadence={cadence_min}, gas={gas}",
            )

    def test_formula_arb_is_jax_differentiable(self):
        """jax.grad w.r.t. sigma should run without error."""
        import jax
        import jax.numpy as jnp

        grad_fn = jax.grad(self.formula, argnums=0)
        result = grad_fn(
            jnp.float64(0.03), jnp.float64(1e6),
            jnp.float64(0.003), jnp.float64(1.0),
            jnp.float64(1.0),
        )
        assert np.isfinite(float(result))

    def test_formula_arb_cadence_reduces_volume(self):
        """Higher cadence → less frequent arb → lower volume."""
        import jax.numpy as jnp
        sigma = jnp.float64(0.03)
        tvl = jnp.float64(1e6)
        fee = jnp.float64(0.003)
        gas = jnp.float64(1.0)

        v_fast = float(self.formula(sigma, tvl, fee, gas, jnp.float64(1.0)))
        v_slow = float(self.formula(sigma, tvl, fee, gas, jnp.float64(10.0)))
        assert v_fast > v_slow
