"""Tests for reClAMM Tsoukalas noise volume model.

Tests noise volume functions (sqrt and log variants), volatility computation,
scan step integration, pool class plumbing, and OLS calibration.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from quantammsim.pools.noise_trades import (
    reclamm_tsoukalas_sqrt_noise_volume,
    reclamm_tsoukalas_log_noise_volume,
    reclamm_loglinear_noise_volume,
)

# Typical noise_params for a mid-cap pool.
# a_c=1.5 is roughly equivalent to the old a_c_real=1.0 + a_c_virt=0.5
# for pools with comparable real and virtual TVL: 1.5/sqrt(2) ≈ 1.06
# per-component, but we use 1.5 to ensure noise income dominates the
# arb-suppression side-effect in integration tests.
DEFAULT_NOISE_PARAMS = {
    "a_0_base": 0.5,
    "a_f": 0.0,
    "a_sigma": 2.0,
    "a_c": 1.5,
    "base_fee": 0.003,
}


# ---------------------------------------------------------------------------
# Tests 1-7: Unit tests for noise volume functions
# ---------------------------------------------------------------------------


class TestPositiveOutputReasonableInputs:
    """Test 1: Volume > 0 for typical inputs (both sqrt and log variants)."""

    def test_sqrt_positive_output(self):
        vol = reclamm_tsoukalas_sqrt_noise_volume(
            effective_value_usd=15_000_000.0,
            gamma=0.997,
            volatility=0.5,
            arb_volume_this_period=0.0,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        assert float(vol) > 0, f"Expected positive noise volume, got {float(vol)}"

    def test_log_positive_output(self):
        vol = reclamm_tsoukalas_log_noise_volume(
            effective_value_usd=15_000_000.0,
            gamma=0.997,
            volatility=0.5,
            arb_volume_this_period=0.0,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        assert float(vol) > 0, f"Expected positive noise volume, got {float(vol)}"


class TestZeroWhenArbExceedsPredicted:
    """Test 2: noise = max(0, daily/1440 - arb), so returns 0 when arb dominates."""

    def test_sqrt_zero_when_arb_large(self):
        vol = reclamm_tsoukalas_sqrt_noise_volume(
            effective_value_usd=3_000_000.0,
            gamma=0.997,
            volatility=0.3,
            arb_volume_this_period=1e12,  # Absurdly large arb
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        assert float(vol) == 0.0, f"Expected zero noise volume, got {float(vol)}"

    def test_log_zero_when_arb_large(self):
        vol = reclamm_tsoukalas_log_noise_volume(
            effective_value_usd=3_000_000.0,
            gamma=0.997,
            volatility=0.3,
            arb_volume_this_period=1e12,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        assert float(vol) == 0.0, f"Expected zero noise volume, got {float(vol)}"


class TestMonotonicInEffectiveTVL:
    """Test 3: Higher effective TVL -> more predicted volume."""

    def test_sqrt_monotonic_effective_tvl(self):
        kwargs = dict(
            gamma=0.997, volatility=0.5, arb_volume_this_period=0.0,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        vol_low = reclamm_tsoukalas_sqrt_noise_volume(effective_value_usd=3_000_000.0, **kwargs)
        vol_high = reclamm_tsoukalas_sqrt_noise_volume(effective_value_usd=20_000_000.0, **kwargs)
        assert float(vol_high) > float(vol_low)

    def test_log_monotonic_effective_tvl(self):
        kwargs = dict(
            gamma=0.997, volatility=0.5, arb_volume_this_period=0.0,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        vol_low = reclamm_tsoukalas_log_noise_volume(effective_value_usd=3_000_000.0, **kwargs)
        vol_high = reclamm_tsoukalas_log_noise_volume(effective_value_usd=20_000_000.0, **kwargs)
        assert float(vol_high) > float(vol_low)


class TestMonotonicInVolatility:
    """Test 4: Higher volatility -> more predicted volume."""

    def test_sqrt_monotonic_volatility(self):
        kwargs = dict(
            effective_value_usd=10_000_000.0,
            gamma=0.997, arb_volume_this_period=0.0,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        vol_low = reclamm_tsoukalas_sqrt_noise_volume(volatility=0.2, **kwargs)
        vol_high = reclamm_tsoukalas_sqrt_noise_volume(volatility=0.8, **kwargs)
        assert float(vol_high) > float(vol_low)

    def test_log_monotonic_volatility(self):
        kwargs = dict(
            effective_value_usd=10_000_000.0,
            gamma=0.997, arb_volume_this_period=0.0,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        vol_low = reclamm_tsoukalas_log_noise_volume(volatility=0.2, **kwargs)
        vol_high = reclamm_tsoukalas_log_noise_volume(volatility=0.8, **kwargs)
        assert float(vol_high) > float(vol_low)


class TestEffectiveTVLSensitivity:
    """Test 5: Changing effective TVL changes output."""

    def test_sqrt_tvl_sensitivity(self):
        base = dict(
            gamma=0.997, volatility=0.5, arb_volume_this_period=0.0,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        v1 = reclamm_tsoukalas_sqrt_noise_volume(
            effective_value_usd=7_000_000.0, **base)
        v2 = reclamm_tsoukalas_sqrt_noise_volume(
            effective_value_usd=13_000_000.0, **base)
        assert float(v1) != float(v2), "Effective TVL change should affect output"

    def test_log_tvl_sensitivity(self):
        base = dict(
            gamma=0.997, volatility=0.5, arb_volume_this_period=0.0,
            noise_params=DEFAULT_NOISE_PARAMS,
        )
        v1 = reclamm_tsoukalas_log_noise_volume(
            effective_value_usd=7_000_000.0, **base)
        v2 = reclamm_tsoukalas_log_noise_volume(
            effective_value_usd=13_000_000.0, **base)
        assert float(v1) != float(v2), "Effective TVL change should affect output"


class TestCustomParamsOverrideDefaults:
    """Test 6: noise_params dict values are actually used."""

    def test_sqrt_custom_params(self):
        # With a_sigma=0, volatility shouldn't matter
        zero_sigma_params = {**DEFAULT_NOISE_PARAMS, "a_sigma": 0.0}
        v1 = reclamm_tsoukalas_sqrt_noise_volume(
            effective_value_usd=10_000_000.0,
            gamma=0.997, volatility=0.2, arb_volume_this_period=0.0,
            noise_params=zero_sigma_params,
        )
        v2 = reclamm_tsoukalas_sqrt_noise_volume(
            effective_value_usd=10_000_000.0,
            gamma=0.997, volatility=0.8, arb_volume_this_period=0.0,
            noise_params=zero_sigma_params,
        )
        npt.assert_allclose(float(v1), float(v2), rtol=1e-10,
                            err_msg="With a_sigma=0, volatility should not affect output")

    def test_log_custom_params(self):
        zero_sigma_params = {**DEFAULT_NOISE_PARAMS, "a_sigma": 0.0}
        v1 = reclamm_tsoukalas_log_noise_volume(
            effective_value_usd=10_000_000.0,
            gamma=0.997, volatility=0.2, arb_volume_this_period=0.0,
            noise_params=zero_sigma_params,
        )
        v2 = reclamm_tsoukalas_log_noise_volume(
            effective_value_usd=10_000_000.0,
            gamma=0.997, volatility=0.8, arb_volume_this_period=0.0,
            noise_params=zero_sigma_params,
        )
        npt.assert_allclose(float(v1), float(v2), rtol=1e-10,
                            err_msg="With a_sigma=0, volatility should not affect output")


class TestCalculateVolatilityArray:
    """Test calculate_volatility_array on the base pool class (JIT'd, pure JAX)."""

    def _make_run_fp(self):
        from quantammsim.runners.jax_runner_utils import Hashabledict
        return Hashabledict({"tokens": ("ETH", "USDC"), "numeraire": "USDC"})

    def _make_pool(self):
        from quantammsim.pools.creator import create_pool
        return create_pool("reclamm")

    def test_output_shape_matches_input(self):
        """Volatility array length matches input price length."""
        pool = self._make_pool()
        n_minutes = 1440 * 3  # 3 days
        rng = np.random.default_rng(42)
        log_rets = rng.normal(0, 0.001, (n_minutes, 2))
        prices = jnp.array(
            np.exp(np.cumsum(log_rets, axis=0)) * np.array([2500.0, 1.0])
        )
        vol_array = pool.calculate_volatility_array(prices, self._make_run_fp())
        assert vol_array.shape == (n_minutes,), (
            f"Expected shape ({n_minutes},), got {vol_array.shape}"
        )

    def test_constant_prices_zero_vol(self):
        """Constant prices should give zero volatility."""
        pool = self._make_pool()
        n_minutes = 1440 * 2
        prices = jnp.tile(jnp.array([2500.0, 1.0]), (n_minutes, 1))
        vol_array = pool.calculate_volatility_array(prices, self._make_run_fp())
        npt.assert_allclose(np.array(vol_array), 0.0, atol=1e-10)

    def test_volatile_prices_positive_vol(self):
        """Volatile prices should give positive volatility."""
        pool = self._make_pool()
        n_minutes = 1440 * 2
        rng = np.random.default_rng(123)
        log_rets = rng.normal(0, 0.01, n_minutes)
        price_ratio = np.exp(np.cumsum(log_rets))
        prices = jnp.array(
            np.column_stack([price_ratio * 2500.0, np.ones(n_minutes)])
        )
        vol_array = pool.calculate_volatility_array(prices, self._make_run_fp())
        assert float(jnp.mean(vol_array)) > 0, "Volatile prices should give positive vol"

    def test_partial_last_day_handled(self):
        """Non-multiple-of-1440: correct shape and partial-day fill uses last day's vol."""
        pool = self._make_pool()
        n_full_days = 2
        n_partial = 500
        n_minutes = 1440 * n_full_days + n_partial

        # Use volatile prices so daily vol is nonzero
        rng = np.random.default_rng(77)
        log_rets = rng.normal(0, 0.005, n_minutes)
        price_ratio = np.exp(np.cumsum(log_rets))
        prices = jnp.array(
            np.column_stack([price_ratio * 2500.0, np.ones(n_minutes)])
        )
        vol_array = pool.calculate_volatility_array(prices, self._make_run_fp())

        assert vol_array.shape == (n_minutes,)

        # The partial-day region (last 500 minutes) should be filled
        # with the last full day's volatility value
        last_full_day_vol = vol_array[n_full_days * 1440 - 1]
        partial_region = vol_array[n_full_days * 1440:]
        npt.assert_allclose(
            np.array(partial_region),
            float(last_full_day_vol),
            rtol=1e-10,
            err_msg="Partial-day region should be filled with last full day's vol",
        )
        # And the fill value should be nonzero (volatile prices)
        assert float(last_full_day_vol) > 0, "Expected nonzero vol from volatile prices"


# ---------------------------------------------------------------------------
# Tests 8-12: Scan step integration tests
# ---------------------------------------------------------------------------

from quantammsim.pools.reCLAMM.reclamm_reserves import (
    initialise_reclamm_reserves,
    _jax_calc_reclamm_reserves_with_fees,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_fees,
    _jax_calc_reclamm_reserves_and_fee_revenue_with_dynamic_inputs,
)

ALL_SIG_VARIATIONS_2 = jnp.array([[1, -1], [-1, 1]])

# Pool config shared by integration tests
_CM = 0.2  # centeredness_margin
_DPSB = 1.0 - 1.0 / 124000.0  # daily_price_shift_base
_SPP = 60.0  # seconds_per_step (1-min arb)
_FEES = 0.003
_PRICE_RATIO = 4.0
_POOL_VALUE = 1_000_000.0


def _init_pool(pool_value=_POOL_VALUE, price_a=2500.0, price_b=1.0,
               price_ratio=_PRICE_RATIO):
    initial_prices = jnp.array([price_a, price_b])
    reserves, Va, Vb = initialise_reclamm_reserves(pool_value, initial_prices, price_ratio)
    return reserves, Va, Vb


def _make_trending_prices(start_a, end_a, price_b, n_steps):
    prices_a = jnp.linspace(start_a, end_a, n_steps)
    prices_b = jnp.full(n_steps, price_b)
    return jnp.stack([prices_a, prices_b], axis=1)


class TestRatioBackwardCompatible:
    """Test 8: noise_model='ratio' matches existing noise_trader_ratio path."""

    def test_ratio_model_matches_legacy(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 50
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        # Legacy path: just noise_trader_ratio
        res_legacy = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_trader_ratio=1.5,
        )

        # New path: noise_model="ratio" (default)
        res_new = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_trader_ratio=1.5,
            noise_model="ratio",
        )
        npt.assert_array_equal(res_legacy, res_new)


class TestArbOnlyEqualsZeroRatio:
    """Test 9: noise_model='arb_only' same as noise_trader_ratio=0."""

    def test_arb_only_matches_zero_ratio(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 50
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)

        res_zero = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_trader_ratio=0.0,
        )
        res_arb_only = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="arb_only",
        )
        npt.assert_array_equal(res_zero, res_arb_only)


class TestTsoukalasSqrtIncreasesReserves:
    """Test 10: Tsoukalas noise income grows real TVL vs arb-only."""

    def test_sqrt_reserves_grow(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 50
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)
        vol_array = jnp.full(n_steps, 0.5)  # Synthetic constant volatility

        res_arb_only = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="arb_only",
        )
        res_tsoukalas = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="tsoukalas_sqrt",
            noise_params=DEFAULT_NOISE_PARAMS,
            volatility_array=vol_array,
        )
        # Noise fee income should make total real value strictly greater than arb-only
        val_arb = float(jnp.sum(res_arb_only[-1] * prices[-1]))
        val_tsoukalas = float(jnp.sum(res_tsoukalas[-1] * prices[-1]))
        assert val_tsoukalas > val_arb, (
            f"Tsoukalas reserves ({val_tsoukalas:.2f}) should be strictly > "
            f"arb-only ({val_arb:.2f})"
        )


class TestTsoukalasDoesNotAffectVirtualBalances:
    """Test 11: Within a single scan step, noise modifies real reserves
    but does NOT modify Va/Vb in the carry."""

    def test_single_step_virtual_balances_identical(self):
        """Call the scan step directly for one step with arb_only and tsoukalas_sqrt.
        Assert that the carry's Va and Vb are bitwise identical."""
        from quantammsim.pools.reCLAMM.reclamm_reserves import (
            _reclamm_scan_step_with_fees_and_revenue,
        )
        from quantammsim.pools.G3M.optimal_n_pool_arb import (
            precalc_shared_values_for_all_signatures,
            precalc_components_of_optimal_trade_across_prices,
        )

        reserves, Va, Vb = _init_pool()
        # Small price shift so arb volume is small relative to predicted noise
        # (a 2500→3000 jump produces arb > predicted noise/min, zeroing noise_vol)
        prices_1 = jnp.array([[2510.0, 1.0]])

        weights = jnp.array([0.5, 0.5])
        gamma = 1.0 - _FEES

        _, active_trade_dirs, tokens_to_drop, leave_one_out_idxs = (
            precalc_shared_values_for_all_signatures(ALL_SIG_VARIATIONS_2, 2)
        )
        aiw, par, aoar = precalc_components_of_optimal_trade_across_prices(
            weights, prices_1, gamma, tokens_to_drop,
            active_trade_dirs, leave_one_out_idxs,
        )

        carry = [
            reserves, Va, Vb,
            jnp.float64(1.0),   # prev_lp_supply
            jnp.float64(0.0),   # step_idx
            jnp.float64(0.0),   # active_start_ratio
            jnp.float64(0.0),   # active_target_ratio
            jnp.float64(0.0),   # active_start_step
            jnp.float64(0.0),   # active_end_step
            jnp.array(False),   # active_enabled
        ]

        def run_step(noise_model, noise_params=None):
            input_list = [
                prices_1[0], aiw[0], par[0], aoar[0],
                jnp.float64(gamma), jnp.float64(0.0),
                jnp.float64(0.0),
                jnp.array([0.0, 0.0, 0.0]),  # price_ratio_update (no-op)
                jnp.float64(1.0),             # lp_supply
            ]
            if noise_model in ("tsoukalas_sqrt", "tsoukalas_log", "loglinear"):
                input_list.append(jnp.float64(0.5))  # volatility

            return _reclamm_scan_step_with_fees_and_revenue(
                carry, input_list,
                weights=weights,
                tokens_to_drop=tokens_to_drop,
                active_trade_directions=active_trade_dirs,
                n=2,
                centeredness_margin=_CM,
                daily_price_shift_base=_DPSB,
                seconds_per_step=_SPP,
                noise_model=noise_model,
                noise_params=noise_params if noise_params is not None else {},
            )

        carry_arb, (res_arb, _) = run_step("arb_only")
        carry_tsoukalas, (res_tsoukalas, _) = run_step(
            "tsoukalas_sqrt", DEFAULT_NOISE_PARAMS,
        )

        # Va and Vb in carry must be bitwise identical
        npt.assert_array_equal(carry_arb[1], carry_tsoukalas[1],
                               err_msg="Va should be unaffected by noise model")
        npt.assert_array_equal(carry_arb[2], carry_tsoukalas[2],
                               err_msg="Vb should be unaffected by noise model")

        # But real reserves SHOULD differ (noise adds fee income).
        # The noise effect is small relative to reserve magnitude, so use
        # exact bitwise comparison rather than allclose (whose default
        # rtol=1e-5 would mask the difference).
        assert not jnp.array_equal(res_arb, res_tsoukalas), (
            "Real reserves should differ between arb_only and tsoukalas_sqrt"
        )

        # Same invariant for loglinear path
        loglinear_params = {"b_0": -1.4, "b_sigma": 0.1, "b_c": 1.04}
        carry_loglinear, (res_loglinear, _) = run_step(
            "loglinear", loglinear_params,
        )
        npt.assert_array_equal(carry_arb[1], carry_loglinear[1],
                               err_msg="Va should be unaffected by loglinear noise model")
        npt.assert_array_equal(carry_arb[2], carry_loglinear[2],
                               err_msg="Vb should be unaffected by loglinear noise model")
        assert not jnp.array_equal(res_arb, res_loglinear), (
            "Real reserves should differ between arb_only and loglinear"
        )


class TestTsoukalasWithFeeRevenue:
    """Test 12: Fee revenue includes noise contribution."""

    def test_fee_revenue_includes_noise(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 50
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)
        vol_array = jnp.full(n_steps, 0.5)  # Synthetic constant volatility

        _, fee_rev_arb = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="arb_only",
        )
        _, fee_rev_tsoukalas = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="tsoukalas_sqrt",
            noise_params=DEFAULT_NOISE_PARAMS,
            volatility_array=vol_array,
        )
        # Tsoukalas should generate strictly more fee revenue due to noise volume
        total_arb = float(fee_rev_arb.sum())
        total_tsoukalas = float(fee_rev_tsoukalas.sum())
        assert total_tsoukalas > total_arb, (
            f"Tsoukalas fee revenue ({total_tsoukalas:.4f}) should exceed "
            f"arb-only ({total_arb:.4f})"
        )


# ---------------------------------------------------------------------------
# Tests 13-14: Pool class integration tests
# ---------------------------------------------------------------------------


class TestNoiseModelFromFingerprint:
    """Test 13: Pool reads noise_model from fingerprint."""

    def test_tsoukalas_sqrt_from_fingerprint(self):
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": _PRICE_RATIO,
            "centeredness_margin": _CM,
            "daily_price_shift_base": _DPSB,
        }

        n_steps = 50
        np.random.seed(42)
        price_a = 2500.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_steps)))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        # Fingerprint with Tsoukalas noise model
        run_fingerprint_tsoukalas = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": _POOL_VALUE,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": _FEES,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "noise_model": "tsoukalas_sqrt",
            "reclamm_noise_params": DEFAULT_NOISE_PARAMS,
        })

        # Fingerprint without noise
        run_fingerprint_arb_only = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": _POOL_VALUE,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": _FEES,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "noise_model": "arb_only",
        })

        start_index = jnp.array([0, 0])

        res_tsoukalas, fee_rev_tsoukalas = pool.calculate_reserves_and_fee_revenue_with_fees(
            params, run_fingerprint_tsoukalas, prices, start_index,
        )
        res_arb, fee_rev_arb = pool.calculate_reserves_and_fee_revenue_with_fees(
            params, run_fingerprint_arb_only, prices, start_index,
        )

        assert res_tsoukalas.shape == (n_steps, 2)
        assert fee_rev_tsoukalas.shape == (n_steps,)
        # Tsoukalas should produce more fee revenue
        assert float(fee_rev_tsoukalas.sum()) > float(fee_rev_arb.sum())


class TestVolatilityComputedForTsoukalas:
    """Test 14: Volatility array auto-computed when noise_model is tsoukalas_*.

    Uses >= 1440 minutes so the real vmap+dynamic_slice path is exercised
    (not just the <1440 fallback). Compares against arb_only to verify the
    auto-computed volatility feeds through to meaningfully different fee revenue.
    """

    def test_volatility_auto_computed_affects_fee_revenue(self):
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": _PRICE_RATIO,
            "centeredness_margin": _CM,
            "daily_price_shift_base": _DPSB,
        }

        # Need at least 1 day of data for real volatility computation
        n_steps = 1440 + 100  # Just over 1 day
        np.random.seed(42)
        price_a = 2500.0 * np.exp(np.cumsum(np.random.normal(0, 0.001, n_steps)))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        base_fp = {
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": _POOL_VALUE,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": _FEES,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
        }

        fp_tsoukalas = Hashabledict({
            **base_fp,
            "noise_model": "tsoukalas_sqrt",
            "reclamm_noise_params": DEFAULT_NOISE_PARAMS,
        })
        fp_arb_only = Hashabledict({
            **base_fp,
            "noise_model": "arb_only",
        })

        start_index = jnp.array([0, 0])

        res_tsoukalas, fee_rev_tsoukalas = pool.calculate_reserves_and_fee_revenue_with_fees(
            params, fp_tsoukalas, prices, start_index,
        )
        _, fee_rev_arb = pool.calculate_reserves_and_fee_revenue_with_fees(
            params, fp_arb_only, prices, start_index,
        )

        assert res_tsoukalas.shape == (n_steps, 2)
        assert fee_rev_tsoukalas.shape == (n_steps,)

        # The auto-computed volatility must feed through to produce
        # strictly more fee revenue than arb-only
        total_tsoukalas = float(fee_rev_tsoukalas.sum())
        total_arb = float(fee_rev_arb.sum())
        assert total_tsoukalas > total_arb, (
            f"Tsoukalas with auto-computed volatility ({total_tsoukalas:.4f}) "
            f"should exceed arb-only ({total_arb:.4f})"
        )


# ---------------------------------------------------------------------------
# Tests 15-16: Calibration pipeline tests
# ---------------------------------------------------------------------------

import pandas as pd
from scripts.calibrate_reclamm_noise import run_ols_calibration


class TestOLSRecoversKnownParams:
    """Test 15: Synthetic data with known coefficients -> OLS recovers them."""

    def test_ols_recovery_sqrt(self):
        rng = np.random.default_rng(42)
        n = 200

        true_a_0 = 0.8
        true_a_sigma = 1.5
        true_a_c = 0.6

        vol = rng.uniform(0.2, 1.0, n)
        eff_tvl = rng.uniform(3e6, 50e6, n)

        # Construct volume from known params (in $M units)
        volume_M = (
            true_a_0
            + true_a_sigma * vol
            + true_a_c * np.sqrt(eff_tvl / 1e6)
        )
        # Add small noise
        volume_M += rng.normal(0, 0.01, n)
        volume_usd = volume_M * 1e6

        df = pd.DataFrame({
            "volume_usd": volume_usd,
            "volatility": vol,
            "effective_tvl_usd": eff_tvl,
        })

        noise_params, diagnostics = run_ols_calibration(df, base_fee=0.003, model="sqrt")

        npt.assert_allclose(noise_params["a_0_base"], true_a_0, atol=0.05)
        npt.assert_allclose(noise_params["a_sigma"], true_a_sigma, atol=0.05)
        npt.assert_allclose(noise_params["a_c"], true_a_c, atol=0.05)
        assert diagnostics["r_squared"] > 0.99

    def test_ols_recovery_log(self):
        rng = np.random.default_rng(123)
        n = 200

        true_a_0 = 0.5
        true_a_sigma = 2.0
        true_a_c = 0.4

        vol = rng.uniform(0.2, 1.0, n)
        eff_tvl = rng.uniform(3e6, 50e6, n)

        volume_M = (
            true_a_0
            + true_a_sigma * vol
            + true_a_c * np.log(eff_tvl / 1e6)
        )
        volume_M += rng.normal(0, 0.01, n)
        volume_usd = volume_M * 1e6

        df = pd.DataFrame({
            "volume_usd": volume_usd,
            "volatility": vol,
            "effective_tvl_usd": eff_tvl,
        })

        noise_params, diagnostics = run_ols_calibration(df, base_fee=0.003, model="log")

        npt.assert_allclose(noise_params["a_0_base"], true_a_0, atol=0.05)
        npt.assert_allclose(noise_params["a_sigma"], true_a_sigma, atol=0.05)
        npt.assert_allclose(noise_params["a_c"], true_a_c, atol=0.05)
        assert diagnostics["r_squared"] > 0.99


class TestOutputFormatCompatible:
    """Test 16: Output dict has all required keys for run_fingerprint integration."""

    def test_output_keys(self):
        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame({
            "volume_usd": rng.uniform(1e6, 10e6, n),
            "volatility": rng.uniform(0.2, 0.8, n),
            "effective_tvl_usd": rng.uniform(3e6, 25e6, n),
        })

        noise_params, diagnostics = run_ols_calibration(df, base_fee=0.003)

        required_keys = {"a_0_base", "a_f", "a_sigma", "a_c", "base_fee"}
        assert set(noise_params.keys()) == required_keys

        # All values are float
        for k, v in noise_params.items():
            assert isinstance(v, float), f"{k} should be float, got {type(v)}"

        # a_f should be 0 for static fees
        assert noise_params["a_f"] == 0.0

        # Diagnostics should have standard errors
        assert "se" in diagnostics
        assert set(diagnostics["se"].keys()) == {"a_0", "a_sigma", "a_c"}

        # Can be used directly as noise_params for the noise functions
        vol = reclamm_tsoukalas_sqrt_noise_volume(
            effective_value_usd=15e6,
            gamma=0.997, volatility=0.5, arb_volume_this_period=0.0,
            noise_params=noise_params,
        )
        assert jnp.isfinite(vol)


# ---------------------------------------------------------------------------
# Tests 17-24: Loglinear (hierarchical) noise volume model
# ---------------------------------------------------------------------------

# Typical noise_params from the hierarchical model
LOGLINEAR_NOISE_PARAMS = {
    "b_0": -7.1,      # grand mean + BLUP
    "b_sigma": -0.003,  # shared volatility effect
    "b_c": 1.04,       # shared TVL elasticity
    "base_fee": 0.003,
}


class TestLoglinearPositiveOutput:
    """Test 17: Volume > 0 for typical inputs."""

    def test_loglinear_positive_output(self):
        vol = reclamm_loglinear_noise_volume(
            effective_value_usd=15_000_000.0,
            gamma=0.997,
            volatility=0.5,
            arb_volume_this_period=0.0,
            noise_params=LOGLINEAR_NOISE_PARAMS,
        )
        assert float(vol) > 0, f"Expected positive noise volume, got {float(vol)}"


class TestLoglinearZeroWhenArbDominates:
    """Test 18: noise = max(0, ...) so returns 0 when arb dominates."""

    def test_loglinear_zero_when_arb_large(self):
        vol = reclamm_loglinear_noise_volume(
            effective_value_usd=3_000_000.0,
            gamma=0.997,
            volatility=0.3,
            arb_volume_this_period=1e12,
            noise_params=LOGLINEAR_NOISE_PARAMS,
        )
        assert float(vol) == 0.0, f"Expected zero, got {float(vol)}"


class TestLoglinearMonotonic:
    """Test 19: Higher effective TVL -> more predicted volume."""

    def test_loglinear_monotonic_tvl(self):
        kwargs = dict(
            gamma=0.997, volatility=0.5, arb_volume_this_period=0.0,
            noise_params=LOGLINEAR_NOISE_PARAMS,
        )
        vol_low = reclamm_loglinear_noise_volume(
            effective_value_usd=3_000_000.0, **kwargs)
        vol_high = reclamm_loglinear_noise_volume(
            effective_value_usd=20_000_000.0, **kwargs)
        assert float(vol_high) > float(vol_low)


class TestLoglinearCustomParams:
    """Test 20: noise_params dict values are actually used."""

    def test_loglinear_custom_b_sigma(self):
        # With b_sigma=0, volatility shouldn't matter
        zero_sigma_params = {**LOGLINEAR_NOISE_PARAMS, "b_sigma": 0.0}
        v1 = reclamm_loglinear_noise_volume(
            effective_value_usd=10_000_000.0,
            gamma=0.997, volatility=0.2, arb_volume_this_period=0.0,
            noise_params=zero_sigma_params,
        )
        v2 = reclamm_loglinear_noise_volume(
            effective_value_usd=10_000_000.0,
            gamma=0.997, volatility=0.8, arb_volume_this_period=0.0,
            noise_params=zero_sigma_params,
        )
        npt.assert_allclose(float(v1), float(v2), rtol=1e-10,
                            err_msg="With b_sigma=0, volatility should not affect output")


class TestLoglinearScanStepIntegration:
    """Test 21: loglinear noise model works through the scan step."""

    def test_loglinear_increases_reserves(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 50
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)
        vol_array = jnp.full(n_steps, 0.5)

        # Use a b_0 that gives reasonable volume at this TVL
        # Pool TVL ~$1M → log(1e6) ≈ 13.8 → b_0 + 1.04*13.8 = b_0 + 14.4
        # Want log(V_daily) ≈ 13 (= ~$440k/day) → b_0 ≈ -1.4
        params = {"b_0": -1.4, "b_sigma": 0.1, "b_c": 1.04, "base_fee": 0.003}

        res_arb_only = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="arb_only",
        )
        res_loglinear = _jax_calc_reclamm_reserves_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="loglinear",
            noise_params=params,
            volatility_array=vol_array,
        )
        val_arb = float(jnp.sum(res_arb_only[-1] * prices[-1]))
        val_loglinear = float(jnp.sum(res_loglinear[-1] * prices[-1]))
        assert val_loglinear > val_arb, (
            f"Loglinear reserves ({val_loglinear:.2f}) should exceed "
            f"arb-only ({val_arb:.2f})"
        )


class TestLoglinearFeeRevenue:
    """Test 22: Fee revenue includes loglinear noise contribution."""

    def test_loglinear_fee_revenue(self):
        reserves, Va, Vb = _init_pool()
        n_steps = 50
        prices = _make_trending_prices(2500.0, 3500.0, 1.0, n_steps)
        vol_array = jnp.full(n_steps, 0.5)

        params = {"b_0": -1.4, "b_sigma": 0.1, "b_c": 1.04, "base_fee": 0.003}

        _, fee_rev_arb = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="arb_only",
        )
        _, fee_rev_loglinear = _jax_calc_reclamm_reserves_and_fee_revenue_with_fees(
            reserves, Va, Vb, prices, _CM, _DPSB, _SPP,
            fees=_FEES, all_sig_variations=ALL_SIG_VARIATIONS_2,
            noise_model="loglinear",
            noise_params=params,
            volatility_array=vol_array,
        )
        total_arb = float(fee_rev_arb.sum())
        total_loglinear = float(fee_rev_loglinear.sum())
        assert total_loglinear > total_arb, (
            f"Loglinear fee revenue ({total_loglinear:.4f}) should exceed "
            f"arb-only ({total_arb:.4f})"
        )


class TestLoglinearPoolClassIntegration:
    """Test 23: Pool reads loglinear noise_model from fingerprint."""

    def test_loglinear_from_fingerprint(self):
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import Hashabledict

        pool = create_pool("reclamm")

        params = {
            "price_ratio": _PRICE_RATIO,
            "centeredness_margin": _CM,
            "daily_price_shift_base": _DPSB,
        }

        n_steps = 50
        np.random.seed(42)
        price_a = 2500.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_steps)))
        prices = jnp.stack([jnp.array(price_a), jnp.ones(n_steps)], axis=1)

        loglinear_params = {
            "b_0": -1.4, "b_sigma": 0.1, "b_c": 1.04, "base_fee": 0.003,
        }

        fp_loglinear = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": _POOL_VALUE,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": _FEES,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "noise_model": "loglinear",
            "reclamm_noise_params": loglinear_params,
        })

        fp_arb_only = Hashabledict({
            "n_assets": 2,
            "bout_length": n_steps + 1,
            "initial_pool_value": _POOL_VALUE,
            "arb_frequency": 1,
            "do_arb": True,
            "fees": _FEES,
            "gas_cost": 0.0,
            "arb_fees": 0.0,
            "tokens": ("ETH", "USDC"),
            "numeraire": "USDC",
            "all_sig_variations": tuple(map(tuple, [[1, -1], [-1, 1]])),
            "noise_model": "arb_only",
        })

        start_index = jnp.array([0, 0])

        res_loglinear, fee_rev_loglinear = pool.calculate_reserves_and_fee_revenue_with_fees(
            params, fp_loglinear, prices, start_index,
        )
        _, fee_rev_arb = pool.calculate_reserves_and_fee_revenue_with_fees(
            params, fp_arb_only, prices, start_index,
        )

        assert res_loglinear.shape == (n_steps, 2)
        assert fee_rev_loglinear.shape == (n_steps,)
        assert float(fee_rev_loglinear.sum()) > float(fee_rev_arb.sum())


class TestLoglinearDefaultParams:
    """Test 24: Function works with default params (noise_params=None)."""

    def test_loglinear_defaults(self):
        vol = reclamm_loglinear_noise_volume(
            effective_value_usd=15_000_000.0,
            gamma=0.997,
            volatility=0.5,
            arb_volume_this_period=0.0,
        )
        assert jnp.isfinite(vol)
        assert float(vol) > 0
