"""
Tests for single-step weight update methods and scan-based calculation.

These tests verify that:
1. calculate_raw_weights_outputs_via_scan produces the same results as calculate_raw_weights_outputs
2. The single-step methods correctly implement the weight update logic
"""
import pytest
import numpy as np
import jax.numpy as jnp

from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool
from quantammsim.pools.G3M.quantamm.power_channel_pool import PowerChannelPool
from quantammsim.pools.G3M.quantamm.mean_reversion_channel_pool import MeanReversionChannelPool
from quantammsim.core_simulator.param_utils import memory_days_to_lamb
from quantammsim.runners.jax_runner_utils import NestedHashabledict


def make_run_fingerprint(n_assets, chunk_period=60, max_memory_days=365.0, hashable=True):
    """Create a minimal run fingerprint for testing.

    Note: chunk_period=60 (hourly) is used by default for numerical stability.
    With chunk_period=1 (minute-level), lambda values get very close to 1,
    causing numerical instability in the gradient calculations.
    """
    fp = {
        "chunk_period": chunk_period,
        "max_memory_days": max_memory_days,
        "n_assets": n_assets,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
    }
    if hashable:
        return NestedHashabledict(fp)
    return fp


def make_momentum_params(n_assets, memory_days=30.0, k_per_day=1.0, chunk_period=60):
    """Create momentum pool parameters."""
    initial_lamb = memory_days_to_lamb(memory_days, chunk_period)
    logit_lamb = np.log(initial_lamb / (1.0 - initial_lamb))

    return {
        "logit_lamb": jnp.array([logit_lamb] * n_assets),
        "logit_delta_lamb": jnp.array([0.0] * n_assets),
        "log_k": jnp.array([np.log2(k_per_day)] * n_assets),
        "initial_weights_logits": jnp.array([0.0] * n_assets),
    }


def make_power_channel_params(n_assets, memory_days=30.0, k_per_day=1.0,
                               exponents=2.0, pre_exp_scaling=0.5, chunk_period=60):
    """Create power channel pool parameters."""
    from quantammsim.core_simulator.param_utils import inverse_squareplus_np

    initial_lamb = memory_days_to_lamb(memory_days, chunk_period)
    logit_lamb = np.log(initial_lamb / (1.0 - initial_lamb))

    return {
        "logit_lamb": jnp.array([logit_lamb] * n_assets),
        "logit_delta_lamb": jnp.array([0.0] * n_assets),
        "sp_k": jnp.array([inverse_squareplus_np(k_per_day)] * n_assets),
        "sp_exponents": jnp.array([inverse_squareplus_np(exponents)] * n_assets),
        "sp_pre_exp_scaling": jnp.array([inverse_squareplus_np(pre_exp_scaling)] * n_assets),
        "initial_weights_logits": jnp.array([0.0] * n_assets),
    }


def make_mean_reversion_channel_params(n_assets, memory_days=30.0, k_per_day=1.0,
                                        exponents=2.0, amplitude=1.0, width=0.1,
                                        pre_exp_scaling=0.5, chunk_period=60):
    """Create mean reversion channel pool parameters."""
    from quantammsim.core_simulator.param_utils import inverse_squareplus_np

    initial_lamb = memory_days_to_lamb(memory_days, chunk_period)
    logit_lamb = np.log(initial_lamb / (1.0 - initial_lamb))

    return {
        "logit_lamb": jnp.array([logit_lamb] * n_assets),
        "logit_delta_lamb": jnp.array([0.0] * n_assets),
        "sp_k": jnp.array([inverse_squareplus_np(k_per_day)] * n_assets),
        "sp_exponents": jnp.array([inverse_squareplus_np(exponents)] * n_assets),
        "sp_amplitude": jnp.array([inverse_squareplus_np(amplitude)] * n_assets),
        "sp_width": jnp.array([inverse_squareplus_np(width)] * n_assets),
        "sp_pre_exp_scaling": jnp.array([inverse_squareplus_np(pre_exp_scaling)] * n_assets),
        "initial_weights_logits": jnp.array([0.0] * n_assets),
    }


def generate_price_data(n_timesteps, n_assets, seed=42):
    """Generate synthetic price data with some trend and noise."""
    np.random.seed(seed)

    # Start prices
    prices = np.zeros((n_timesteps, n_assets))
    prices[0] = 100.0 + np.random.randn(n_assets) * 10

    # Random walk with drift
    for t in range(1, n_timesteps):
        drift = 0.001 * np.random.randn(n_assets)
        noise = 0.02 * np.random.randn(n_assets)
        prices[t] = prices[t-1] * (1 + drift + noise)

    return jnp.array(prices)


class TestMomentumPoolScanEquivalence:
    """Test that scan-based calculation matches original for MomentumPool."""

    def test_scan_matches_original_two_assets(self):
        """Test with 2 assets."""
        n_assets = 2
        n_timesteps = 10000  # Need enough timesteps for hourly chunks

        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        # Original method
        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )

        # Scan-based method
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="Scan output should match original for MomentumPool"
        )

    def test_scan_matches_original_three_assets(self):
        """Test with 3 assets."""
        n_assets = 3
        n_timesteps = 10000

        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="Scan output should match original for MomentumPool (3 assets)"
        )

    def test_scan_matches_original_longer_series(self):
        """Test with longer price series."""
        n_assets = 2
        n_timesteps = 50000

        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="Scan output should match original for longer series"
        )

    def test_scan_matches_original_different_memory_days(self):
        """Test with different memory_days parameter."""
        n_assets = 2
        n_timesteps = 10000

        pool = MomentumPool()
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        for memory_days in [7.0, 30.0, 90.0]:
            params = make_momentum_params(n_assets, memory_days=memory_days)

            original_output = pool.calculate_raw_weights_outputs(
                params, run_fingerprint, prices, None
            )
            scan_output = pool.calculate_raw_weights_outputs_via_scan(
                params, run_fingerprint, prices, None
            )

            np.testing.assert_allclose(
                original_output, scan_output, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with memory_days={memory_days}"
            )

    def test_scan_matches_original_different_k(self):
        """Test with different k parameter."""
        n_assets = 2
        n_timesteps = 10000

        pool = MomentumPool()
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        for k_per_day in [0.5, 1.0, 2.0, 5.0]:
            params = make_momentum_params(n_assets, k_per_day=k_per_day)

            original_output = pool.calculate_raw_weights_outputs(
                params, run_fingerprint, prices, None
            )
            scan_output = pool.calculate_raw_weights_outputs_via_scan(
                params, run_fingerprint, prices, None
            )

            np.testing.assert_allclose(
                original_output, scan_output, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with k_per_day={k_per_day}"
            )


class TestPowerChannelPoolScanEquivalence:
    """Test that scan-based calculation matches original for PowerChannelPool."""

    def test_scan_matches_original_two_assets(self):
        """Test with 2 assets."""
        n_assets = 2
        n_timesteps = 10000

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="Scan output should match original for PowerChannelPool"
        )

    def test_scan_matches_original_three_assets(self):
        """Test with 3 assets."""
        n_assets = 3
        n_timesteps = 10000

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="Scan output should match original for PowerChannelPool (3 assets)"
        )

    def test_scan_matches_original_different_exponents(self):
        """Test with different exponent values."""
        n_assets = 2
        n_timesteps = 10000

        pool = PowerChannelPool()
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        for exponents in [1.0, 1.5, 2.0, 3.0]:
            params = make_power_channel_params(n_assets, exponents=exponents)

            original_output = pool.calculate_raw_weights_outputs(
                params, run_fingerprint, prices, None
            )
            scan_output = pool.calculate_raw_weights_outputs_via_scan(
                params, run_fingerprint, prices, None
            )

            np.testing.assert_allclose(
                original_output, scan_output, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with exponents={exponents}"
            )

    def test_scan_matches_original_different_pre_exp_scaling(self):
        """Test with different pre_exp_scaling values."""
        n_assets = 2
        n_timesteps = 10000

        pool = PowerChannelPool()
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        for pre_exp_scaling in [0.25, 0.5, 0.75, 1.0]:
            params = make_power_channel_params(n_assets, pre_exp_scaling=pre_exp_scaling)

            original_output = pool.calculate_raw_weights_outputs(
                params, run_fingerprint, prices, None
            )
            scan_output = pool.calculate_raw_weights_outputs_via_scan(
                params, run_fingerprint, prices, None
            )

            np.testing.assert_allclose(
                original_output, scan_output, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with pre_exp_scaling={pre_exp_scaling}"
            )


class TestMeanReversionChannelPoolScanEquivalence:
    """Test that scan-based calculation matches original for MeanReversionChannelPool."""

    def test_scan_matches_original_two_assets(self):
        """Test with 2 assets."""
        n_assets = 2
        n_timesteps = 10000

        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="Scan output should match original for MeanReversionChannelPool"
        )

    def test_scan_matches_original_three_assets(self):
        """Test with 3 assets."""
        n_assets = 3
        n_timesteps = 10000

        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="Scan output should match original for MeanReversionChannelPool (3 assets)"
        )

    def test_scan_matches_original_different_amplitude(self):
        """Test with different amplitude values."""
        n_assets = 2
        n_timesteps = 10000

        pool = MeanReversionChannelPool()
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        for amplitude in [0.5, 1.0, 2.0]:
            params = make_mean_reversion_channel_params(n_assets, amplitude=amplitude)

            original_output = pool.calculate_raw_weights_outputs(
                params, run_fingerprint, prices, None
            )
            scan_output = pool.calculate_raw_weights_outputs_via_scan(
                params, run_fingerprint, prices, None
            )

            np.testing.assert_allclose(
                original_output, scan_output, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with amplitude={amplitude}"
            )

    def test_scan_matches_original_different_width(self):
        """Test with different width values."""
        n_assets = 2
        n_timesteps = 10000

        pool = MeanReversionChannelPool()
        run_fingerprint = make_run_fingerprint(n_assets)
        prices = generate_price_data(n_timesteps, n_assets)

        for width in [0.05, 0.1, 0.2, 0.5]:
            params = make_mean_reversion_channel_params(n_assets, width=width)

            original_output = pool.calculate_raw_weights_outputs(
                params, run_fingerprint, prices, None
            )
            scan_output = pool.calculate_raw_weights_outputs_via_scan(
                params, run_fingerprint, prices, None
            )

            np.testing.assert_allclose(
                original_output, scan_output, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with width={width}"
            )


class TestSingleStepMethods:
    """Test the single-step methods directly."""

    def test_momentum_single_step_returns_correct_shape(self):
        """Test that single step returns correct shapes."""
        n_assets = 3
        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets, hashable=False)

        initial_price = jnp.array([100.0, 150.0, 200.0])
        next_price = jnp.array([101.0, 149.0, 202.0])

        initial_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)

        assert "ewma" in initial_carry
        assert "running_a" in initial_carry
        assert initial_carry["ewma"].shape == (n_assets,)
        assert initial_carry["running_a"].shape == (n_assets,)

        new_carry, output = pool.calculate_single_step_weight_update(
            initial_carry, next_price, params, run_fingerprint
        )

        assert new_carry["ewma"].shape == (n_assets,)
        assert new_carry["running_a"].shape == (n_assets,)
        assert output.shape == (n_assets,)

    def test_power_channel_single_step_returns_correct_shape(self):
        """Test that PowerChannel single step returns correct shapes."""
        n_assets = 3
        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets, hashable=False)

        initial_price = jnp.array([100.0, 150.0, 200.0])
        next_price = jnp.array([101.0, 149.0, 202.0])

        initial_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)
        new_carry, output = pool.calculate_single_step_weight_update(
            initial_carry, next_price, params, run_fingerprint
        )

        assert new_carry["ewma"].shape == (n_assets,)
        assert new_carry["running_a"].shape == (n_assets,)
        assert output.shape == (n_assets,)

    def test_mean_reversion_single_step_returns_correct_shape(self):
        """Test that MeanReversionChannel single step returns correct shapes."""
        n_assets = 3
        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets, hashable=False)

        initial_price = jnp.array([100.0, 150.0, 200.0])
        next_price = jnp.array([101.0, 149.0, 202.0])

        initial_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)
        new_carry, output = pool.calculate_single_step_weight_update(
            initial_carry, next_price, params, run_fingerprint
        )

        assert new_carry["ewma"].shape == (n_assets,)
        assert new_carry["running_a"].shape == (n_assets,)
        assert output.shape == (n_assets,)

    def test_weight_updates_sum_to_zero(self):
        """Test that weight updates sum to zero (preserves portfolio weight)."""
        n_assets = 3
        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets, hashable=False)

        initial_price = jnp.array([100.0, 150.0, 200.0])
        next_price = jnp.array([105.0, 145.0, 210.0])

        initial_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)
        _, output = pool.calculate_single_step_weight_update(
            initial_carry, next_price, params, run_fingerprint
        )

        np.testing.assert_almost_equal(
            jnp.sum(output), 0.0, decimal=10,
            err_msg="Weight updates should sum to zero"
        )

    def test_ewma_updates_correctly(self):
        """Test that EWMA updates according to the expected formula."""
        n_assets = 2
        pool = MomentumPool()
        params = make_momentum_params(n_assets, memory_days=30.0)
        run_fingerprint = make_run_fingerprint(n_assets, hashable=False)

        initial_price = jnp.array([100.0, 200.0])
        next_price = jnp.array([110.0, 190.0])

        initial_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)
        new_carry, _ = pool.calculate_single_step_weight_update(
            initial_carry, next_price, params, run_fingerprint
        )

        # EWMA should be between initial price and next price
        # (exponential average moves toward the new price)
        assert jnp.all(new_carry["ewma"] != initial_price)

        # For asset 0: price increased, so ewma should be > initial
        assert new_carry["ewma"][0] > initial_price[0]
        # For asset 1: price decreased, so ewma should be < initial
        assert new_carry["ewma"][1] < initial_price[1]


def make_fine_weight_run_fingerprint(n_assets, chunk_period=60, max_memory_days=365.0,
                                      weight_interpolation_period=None,
                                      maximum_change=0.01,
                                      weight_interpolation_method="linear",
                                      hashable=True):
    """Create a run fingerprint with all fields needed for fine weight calculation."""
    if weight_interpolation_period is None:
        weight_interpolation_period = chunk_period
    fp = {
        "chunk_period": chunk_period,
        "max_memory_days": max_memory_days,
        "n_assets": n_assets,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "weight_interpolation_period": weight_interpolation_period,
        "maximum_change": maximum_change,
        "weight_interpolation_method": weight_interpolation_method,
        "minimum_weight": 0.1 / n_assets,
        "ste_max_change": False,
        "ste_min_max_weight": False,
    }
    if hashable:
        return NestedHashabledict(fp)
    return fp


class TestFineWeightsScanEquivalence:
    """Test that calculate_fine_weights_via_scan matches calculate_weights.

    calculate_fine_weights_via_scan uses a truly sequential approach with
    single-step interpolation blocks, producing the same output as calculate_weights.
    """

    def test_momentum_fine_weights_match_two_assets(self):
        """Test MomentumPool fine weights via scan matches calculate_weights for 2 assets."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="Fine weights via scan should match calculate_weights for MomentumPool"
        )

    def test_momentum_fine_weights_match_three_assets(self):
        """Test MomentumPool fine weights via scan matches calculate_weights for 3 assets."""
        n_assets = 3
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="Fine weights via scan should match calculate_weights for MomentumPool (3 assets)"
        )

    def test_power_channel_fine_weights_match(self):
        """Test PowerChannelPool fine weights via scan matches calculate_weights."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="Fine weights via scan should match calculate_weights for PowerChannelPool"
        )

    def test_mean_reversion_fine_weights_match(self):
        """Test MeanReversionChannelPool fine weights via scan matches calculate_weights."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="Fine weights via scan should match calculate_weights for MeanReversionChannelPool"
        )

    def test_fine_weights_different_interpolation_periods(self):
        """Test with different weight_interpolation_period values.

        Note: interpolation period must be <= chunk_period, since you cannot
        interpolate over more timesteps than exist in a chunk.
        """
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        # Interpolation periods must be <= chunk_period
        for interp_period in [15, 30, 60]:
            run_fingerprint = make_calculate_weights_run_fingerprint(
                n_assets, chunk_period=chunk_period, bout_length=bout_length,
                weight_interpolation_period=interp_period
            )

            original_weights = pool.calculate_weights(
                params, run_fingerprint, prices, start_index, None
            )
            scan_weights = pool.calculate_fine_weights_via_scan(
                params, run_fingerprint, prices, start_index, None
            )

            np.testing.assert_allclose(
                original_weights, scan_weights, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with weight_interpolation_period={interp_period}"
            )

    def test_fine_weights_different_max_change(self):
        """Test with different maximum_change values."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        for max_change in [0.005, 0.01, 0.02, 0.05]:
            run_fingerprint = make_calculate_weights_run_fingerprint(
                n_assets, chunk_period=chunk_period, bout_length=bout_length,
                maximum_change=max_change
            )

            original_weights = pool.calculate_weights(
                params, run_fingerprint, prices, start_index, None
            )
            scan_weights = pool.calculate_fine_weights_via_scan(
                params, run_fingerprint, prices, start_index, None
            )

            np.testing.assert_allclose(
                original_weights, scan_weights, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with maximum_change={max_change}"
            )

    def test_fine_weights_sum_to_one(self):
        """Test that fine weights always sum to 1."""
        n_assets = 3
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        weight_sums = jnp.sum(scan_weights, axis=1)
        np.testing.assert_allclose(
            weight_sums, jnp.ones_like(weight_sums), rtol=1e-10, atol=1e-10,
            err_msg="Fine weights should sum to 1 at all timesteps"
        )

    def test_fine_weights_respect_min_weight(self):
        """Test that fine weights respect minimum weight constraint."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2
        min_weight = 0.1

        pool = MomentumPool()
        params = make_momentum_params(n_assets, k_per_day=5.0, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        # Override minimum_weight
        run_fingerprint_dict = dict(run_fingerprint)
        run_fingerprint_dict["minimum_weight"] = min_weight
        run_fingerprint = NestedHashabledict(run_fingerprint_dict)

        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        # All weights should be >= minimum_weight (with small tolerance for numerical error)
        assert jnp.all(scan_weights >= min_weight - 1e-10), \
            f"Some weights below minimum: min={jnp.min(scan_weights)}"

    def test_fine_weights_zero_burn_in(self):
        """Test calculate_fine_weights_via_scan with zero burn-in (start_index=0).

        With zero burn-in, the warm-up scan processes an empty array.
        This is an important edge case.
        """
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        n_timesteps = bout_length + chunk_period * 2  # No burn-in needed

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([0])  # Zero burn-in

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        # Use looser tolerance for accumulated floating-point differences
        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-6, atol=1e-6,
            err_msg="calculate_fine_weights_via_scan should match original with zero burn-in"
        )

    def test_fine_weights_approx_optimal_interpolation(self):
        """Test calculate_fine_weights_via_scan with approx_optimal interpolation method."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length,
            weight_interpolation_method="approx_optimal"
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        # Use looser tolerance for accumulated floating-point differences
        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-6, atol=1e-6,
            err_msg="approx_optimal interpolation should match between scan and original"
        )

    def test_fine_weights_approx_optimal_three_assets(self):
        """Test approx_optimal interpolation with 3 assets."""
        n_assets = 3
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length,
            weight_interpolation_method="approx_optimal"
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        # Use looser tolerance for accumulated floating-point differences
        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-6, atol=1e-6,
            err_msg="approx_optimal interpolation should match for 3 assets"
        )

    def test_fine_weights_large_k_forces_guardrails(self):
        """Test with very large k_per_day to ensure guardrails are actually hit.

        With k_per_day=100.0, the weight changes will be large and will
        definitely hit the maximum_change guardrail.
        """
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2
        max_change = 0.01  # Small max_change to ensure guardrail is hit
        min_weight = 0.1 / n_assets

        pool = MomentumPool()
        params = make_momentum_params(n_assets, k_per_day=100.0, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length,
            maximum_change=max_change
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        # Use looser tolerance for accumulated floating-point differences
        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-6, atol=1e-6,
            err_msg="Large k_per_day with guardrails should match between scan and original"
        )

        # Verify guardrails are respected
        min_weight_found = jnp.min(scan_weights)
        assert min_weight_found >= min_weight - 1e-6, \
            f"Min weight violated: found {min_weight_found}, expected >= {min_weight}"

        # Verify weights sum to 1
        weight_sums = jnp.sum(scan_weights, axis=1)
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-10, atol=1e-10)

    def test_fine_weights_large_k_power_channel(self):
        """Test PowerChannelPool with large k to force guardrails."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2
        max_change = 0.005  # Very small to really force guardrails
        min_weight = 0.1 / n_assets

        pool = PowerChannelPool()
        params = make_power_channel_params(
            n_assets, k_per_day=100.0, exponents=3.0, chunk_period=chunk_period
        )
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length,
            maximum_change=max_change
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        # Use looser tolerance for accumulated floating-point differences
        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-6, atol=1e-6,
            err_msg="PowerChannelPool with large k should match between scan and original"
        )

        # Verify guardrails are respected
        min_weight_found = jnp.min(scan_weights)
        assert min_weight_found >= min_weight - 1e-6, \
            f"Min weight violated: found {min_weight_found}, expected >= {min_weight}"

        # Verify weights sum to 1
        weight_sums = jnp.sum(scan_weights, axis=1)
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-10, atol=1e-10)

    def test_fine_weights_large_k_mean_reversion(self):
        """Test MeanReversionChannelPool with large k to force guardrails."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2
        max_change = 0.005
        min_weight = 0.1 / n_assets

        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(
            n_assets, k_per_day=100.0, amplitude=5.0, chunk_period=chunk_period
        )
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length,
            maximum_change=max_change
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        # Use looser tolerance for accumulated floating-point differences
        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-6, atol=1e-6,
            err_msg="MeanReversionChannelPool with large k should match between scan and original"
        )

        # Verify guardrails are respected
        min_weight_found = jnp.min(scan_weights)
        assert min_weight_found >= min_weight - 1e-6, \
            f"Min weight violated: found {min_weight_found}, expected >= {min_weight}"

        # Verify weights sum to 1
        weight_sums = jnp.sum(scan_weights, axis=1)
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-10, atol=1e-10)


class TestSingleStepGuardrailedWeight:
    """Test calculate_single_step_guardrailed_weight directly."""

    def test_guardrailed_weight_output_structure(self):
        """Test that guardrailed weight method returns correct structure."""
        n_assets = 2
        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_fine_weight_run_fingerprint(n_assets, hashable=False)

        initial_price = jnp.array([100.0, 200.0])
        next_price = jnp.array([105.0, 195.0])
        initial_weights = jnp.array([0.5, 0.5])

        estimator_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)
        weight_carry = pool.get_initial_weight_carry(initial_weights)

        new_est_carry, new_wt_carry, step_output = pool.calculate_single_step_guardrailed_weight(
            estimator_carry, weight_carry, next_price, params, run_fingerprint
        )

        # Check structure
        assert "ewma" in new_est_carry
        assert "running_a" in new_est_carry
        assert "prev_actual_weight" in new_wt_carry
        assert "actual_start" in step_output
        assert "scaled_diff" in step_output
        assert "target_weight" in step_output

        # Check shapes
        assert new_est_carry["ewma"].shape == (n_assets,)
        assert new_wt_carry["prev_actual_weight"].shape == (n_assets,)
        assert step_output["actual_start"].shape == (n_assets,)
        assert step_output["scaled_diff"].shape == (n_assets,)
        assert step_output["target_weight"].shape == (n_assets,)

    def test_guardrailed_weights_sum_to_one(self):
        """Test that target weights from guardrailed method sum to 1."""
        n_assets = 3
        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_fine_weight_run_fingerprint(n_assets, hashable=False)

        initial_price = jnp.array([100.0, 150.0, 200.0])
        next_price = jnp.array([105.0, 145.0, 210.0])
        initial_weights = jnp.array([0.4, 0.35, 0.25])

        estimator_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)
        weight_carry = pool.get_initial_weight_carry(initial_weights)

        _, new_wt_carry, step_output = pool.calculate_single_step_guardrailed_weight(
            estimator_carry, weight_carry, next_price, params, run_fingerprint
        )

        # Target weight should sum to 1
        np.testing.assert_almost_equal(
            jnp.sum(step_output["target_weight"]), 1.0, decimal=10,
            err_msg="Target weights should sum to 1"
        )

        # Actual position reached should also sum to 1
        np.testing.assert_almost_equal(
            jnp.sum(new_wt_carry["prev_actual_weight"]), 1.0, decimal=10,
            err_msg="Actual weight position should sum to 1"
        )


class TestSingleStepInterpolationBlock:
    """Test calculate_single_step_interpolation_block directly."""

    def test_interpolation_block_shape(self):
        """Test that interpolation block has correct shape."""
        n_assets = 2
        chunk_period = 60
        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_fine_weight_run_fingerprint(
            n_assets, chunk_period=chunk_period, hashable=False
        )

        initial_price = jnp.array([100.0, 200.0])
        next_price = jnp.array([105.0, 195.0])
        initial_weights = jnp.array([0.5, 0.5])

        estimator_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)
        weight_carry = pool.get_initial_weight_carry(initial_weights)

        _, _, interpolation_block = pool.calculate_single_step_interpolation_block(
            estimator_carry, weight_carry, next_price, params, run_fingerprint
        )

        # Block should have shape (chunk_period, n_assets)
        assert interpolation_block.shape == (chunk_period, n_assets), \
            f"Expected shape ({chunk_period}, {n_assets}), got {interpolation_block.shape}"

    def test_interpolation_block_weights_sum_to_one(self):
        """Test that all weights in interpolation block sum to 1."""
        n_assets = 3
        chunk_period = 60
        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_fine_weight_run_fingerprint(
            n_assets, chunk_period=chunk_period, hashable=False
        )

        initial_price = jnp.array([100.0, 150.0, 200.0])
        next_price = jnp.array([105.0, 145.0, 210.0])
        initial_weights = jnp.array([0.4, 0.35, 0.25])

        estimator_carry = pool.get_initial_carry(initial_price, params, run_fingerprint)
        weight_carry = pool.get_initial_weight_carry(initial_weights)

        _, _, interpolation_block = pool.calculate_single_step_interpolation_block(
            estimator_carry, weight_carry, next_price, params, run_fingerprint
        )

        # All rows should sum to 1
        row_sums = jnp.sum(interpolation_block, axis=1)
        np.testing.assert_allclose(
            row_sums, jnp.ones(chunk_period), rtol=1e-10, atol=1e-10,
            err_msg="All interpolation block rows should sum to 1"
        )


class TestHistoricDataScanEquivalence:
    """Test scan methods match original methods using real historic price data.

    These tests use get_data_dict to load actual BTC/ETH price data and verify
    that scan-based calculation produces identical results to original methods.
    """

    @pytest.fixture
    def historic_data(self):
        """Load real historic price data for BTC/ETH."""
        from quantammsim.utils.data_processing.historic_data_utils import get_data_dict

        n_assets = 2
        list_of_tickers = ["BTC", "ETH"]

        # Create minimal run fingerprint for data loading
        run_fingerprint = {
            "chunk_period": 60,
            "n_assets": n_assets,
        }

        # Load a 1-month window of data
        data_dict = get_data_dict(
            list_of_tickers,
            run_fingerprint,
            data_kind="historic",
            root=None,  # Use package data
            start_date_string="2023-06-01 00:00:00",
            end_time_string="2023-07-01 00:00:00",
        )

        return {
            "prices": jnp.array(data_dict["prices"]),
            "n_assets": n_assets,
            "list_of_tickers": list_of_tickers,
        }

    def test_momentum_raw_weights_historic_data(self, historic_data):
        """Test MomentumPool raw weight scan matches original with real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]

        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="MomentumPool scan should match original with real BTC/ETH data"
        )

    def test_power_channel_raw_weights_historic_data(self, historic_data):
        """Test PowerChannelPool raw weight scan matches original with real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="PowerChannelPool scan should match original with real BTC/ETH data"
        )

    def test_mean_reversion_raw_weights_historic_data(self, historic_data):
        """Test MeanReversionChannelPool raw weight scan matches original with real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]

        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="MeanReversionChannelPool scan should match original with real BTC/ETH data"
        )

    def test_momentum_fine_weights_historic_data(self, historic_data):
        """Test MomentumPool fine weight scan matches original with real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]

        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        chunk_period = 60
        bout_length = 2880  # 2 days of minutes
        burn_in = 1440  # 1 day burn-in
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_fine_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_fine_weights, rtol=1e-10, atol=1e-10,
            err_msg="MomentumPool fine weights scan should match original with real BTC/ETH data"
        )

    def test_power_channel_fine_weights_historic_data(self, historic_data):
        """Test PowerChannelPool fine weight scan matches original with real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets)
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_fine_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_fine_weights, rtol=1e-10, atol=1e-10,
            err_msg="PowerChannelPool fine weights scan should match original with real BTC/ETH data"
        )

    def test_mean_reversion_fine_weights_historic_data(self, historic_data):
        """Test MeanReversionChannelPool fine weight scan matches original with real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]

        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(n_assets)
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_fine_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_fine_weights, rtol=1e-10, atol=1e-10,
            err_msg="MeanReversionChannelPool fine weights scan should match original with real BTC/ETH data"
        )

    def test_momentum_multiple_parameter_sets_historic_data(self, historic_data):
        """Test MomentumPool with various parameter combinations on real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]

        pool = MomentumPool()
        run_fingerprint = make_run_fingerprint(n_assets)

        # Test different memory_days and k values
        test_cases = [
            {"memory_days": 7.0, "k_per_day": 0.5},
            {"memory_days": 14.0, "k_per_day": 1.0},
            {"memory_days": 30.0, "k_per_day": 2.0},
            {"memory_days": 60.0, "k_per_day": 5.0},
        ]

        for case in test_cases:
            params = make_momentum_params(n_assets, **case)

            original_output = pool.calculate_raw_weights_outputs(
                params, run_fingerprint, prices, None
            )
            scan_output = pool.calculate_raw_weights_outputs_via_scan(
                params, run_fingerprint, prices, None
            )

            np.testing.assert_allclose(
                original_output, scan_output, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with params: {case}"
            )

    def test_fine_weights_different_max_change_historic_data(self, historic_data):
        """Test fine weights with various maximum_change values on real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]

        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        start_index = jnp.array([burn_in])

        for max_change in [0.001, 0.005, 0.01, 0.02, 0.05]:
            run_fingerprint = make_calculate_weights_run_fingerprint(
                n_assets, chunk_period=chunk_period, bout_length=bout_length,
                maximum_change=max_change
            )

            original_weights = pool.calculate_weights(
                params, run_fingerprint, prices, start_index, None
            )
            scan_fine_weights = pool.calculate_fine_weights_via_scan(
                params, run_fingerprint, prices, start_index, None
            )

            np.testing.assert_allclose(
                original_weights, scan_fine_weights, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with maximum_change={max_change}"
            )


class TestHistoricDataThreeAssets:
    """Test scan equivalence with 3 assets using real historic data."""

    @pytest.fixture
    def historic_data_3_assets(self):
        """Load real historic price data for BTC/ETH/USDC."""
        from quantammsim.utils.data_processing.historic_data_utils import get_data_dict

        n_assets = 3
        list_of_tickers = ["BTC", "ETH", "USDC"]

        run_fingerprint = {
            "chunk_period": 60,
            "n_assets": n_assets,
        }

        data_dict = get_data_dict(
            list_of_tickers,
            run_fingerprint,
            data_kind="historic",
            root=None,
            start_date_string="2023-06-01 00:00:00",
            end_time_string="2023-07-01 00:00:00",
        )

        return {
            "prices": jnp.array(data_dict["prices"]),
            "n_assets": n_assets,
            "list_of_tickers": list_of_tickers,
        }

    def test_momentum_raw_weights_3_assets(self, historic_data_3_assets):
        """Test MomentumPool raw weight scan with 3 real assets."""
        n_assets = historic_data_3_assets["n_assets"]
        prices = historic_data_3_assets["prices"]

        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        run_fingerprint = make_run_fingerprint(n_assets)

        original_output = pool.calculate_raw_weights_outputs(
            params, run_fingerprint, prices, None
        )
        scan_output = pool.calculate_raw_weights_outputs_via_scan(
            params, run_fingerprint, prices, None
        )

        np.testing.assert_allclose(
            original_output, scan_output, rtol=1e-10, atol=1e-10,
            err_msg="MomentumPool scan should match original with 3 real assets"
        )

    def test_momentum_fine_weights_3_assets(self, historic_data_3_assets):
        """Test MomentumPool fine weight scan with 3 real assets."""
        n_assets = historic_data_3_assets["n_assets"]
        prices = historic_data_3_assets["prices"]

        pool = MomentumPool()
        params = make_momentum_params(n_assets)
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_fine_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_fine_weights, rtol=1e-10, atol=1e-10,
            err_msg="MomentumPool fine weights scan should match original with 3 real assets"
        )

    def test_power_channel_fine_weights_3_assets(self, historic_data_3_assets):
        """Test PowerChannelPool fine weight scan with 3 real assets."""
        n_assets = historic_data_3_assets["n_assets"]
        prices = historic_data_3_assets["prices"]

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets)
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_fine_weights = pool.calculate_fine_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_fine_weights, rtol=1e-10, atol=1e-10,
            err_msg="PowerChannelPool fine weights scan should match original with 3 real assets"
        )


def make_calculate_weights_run_fingerprint(n_assets, chunk_period=60, bout_length=5760,
                                            max_memory_days=365.0,
                                            weight_interpolation_period=None,
                                            maximum_change=0.01,
                                            weight_interpolation_method="linear",
                                            hashable=True):
    """Create a run fingerprint with all fields needed for calculate_weights."""
    if weight_interpolation_period is None:
        weight_interpolation_period = chunk_period
    fp = {
        "chunk_period": chunk_period,
        "bout_length": bout_length,
        "max_memory_days": max_memory_days,
        "n_assets": n_assets,
        "use_alt_lamb": False,
        "use_pre_exp_scaling": True,
        "weight_interpolation_period": weight_interpolation_period,
        "maximum_change": maximum_change,
        "weight_interpolation_method": weight_interpolation_method,
        "minimum_weight": 0.1 / n_assets,
        "ste_max_change": False,
        "ste_min_max_weight": False,
    }
    if hashable:
        return NestedHashabledict(fp)
    return fp


class TestCalculateWeightsViaScan:
    """Test that calculate_weights_via_scan matches calculate_weights."""

    def test_momentum_calculate_weights_match_two_assets(self):
        """Test MomentumPool calculate_weights_via_scan matches original for 2 assets."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880  # 2 days
        burn_in = 1440  # 1 day
        n_timesteps = burn_in + bout_length + chunk_period * 2  # Extra buffer

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="MomentumPool calculate_weights_via_scan should match original"
        )

    def test_momentum_calculate_weights_match_three_assets(self):
        """Test MomentumPool calculate_weights_via_scan matches original for 3 assets."""
        n_assets = 3
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="MomentumPool calculate_weights_via_scan should match original (3 assets)"
        )

    def test_power_channel_calculate_weights_match(self):
        """Test PowerChannelPool calculate_weights_via_scan matches original."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="PowerChannelPool calculate_weights_via_scan should match original"
        )

    def test_mean_reversion_calculate_weights_match(self):
        """Test MeanReversionChannelPool calculate_weights_via_scan matches original."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="MeanReversionChannelPool calculate_weights_via_scan should match original"
        )

    def test_calculate_weights_zero_burn_in(self):
        """Test calculate_weights_via_scan with zero burn-in (start_index=0)."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        n_timesteps = bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([0])  # No burn-in

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        # Use looser tolerance for accumulated floating-point differences
        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-6, atol=1e-6,
            err_msg="calculate_weights_via_scan should match original with zero burn-in"
        )

    def test_calculate_weights_long_burn_in(self):
        """Test calculate_weights_via_scan with longer burn-in period."""
        n_assets = 2
        chunk_period = 60
        bout_length = 2880
        burn_in = 4320  # 3 days
        n_timesteps = burn_in + bout_length + chunk_period * 2

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        prices = generate_price_data(n_timesteps, n_assets)
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="calculate_weights_via_scan should match original with long burn-in"
        )

    def test_calculate_weights_different_bout_lengths(self):
        """Test calculate_weights_via_scan with various bout lengths."""
        n_assets = 2
        chunk_period = 60
        burn_in = 1440

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)

        for bout_length in [1440, 2880, 4320, 5760]:
            n_timesteps = burn_in + bout_length + chunk_period * 2
            run_fingerprint = make_calculate_weights_run_fingerprint(
                n_assets, chunk_period=chunk_period, bout_length=bout_length
            )
            prices = generate_price_data(n_timesteps, n_assets)
            start_index = jnp.array([burn_in])

            original_weights = pool.calculate_weights(
                params, run_fingerprint, prices, start_index, None
            )
            scan_weights = pool.calculate_weights_via_scan(
                params, run_fingerprint, prices, start_index, None
            )

            np.testing.assert_allclose(
                original_weights, scan_weights, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with bout_length={bout_length}"
            )


class TestCalculateWeightsViaScanHistoric:
    """Test calculate_weights_via_scan with real historic data."""

    @pytest.fixture
    def historic_data(self):
        """Load real historic price data for BTC/ETH."""
        from quantammsim.utils.data_processing.historic_data_utils import get_data_dict

        n_assets = 2
        list_of_tickers = ["BTC", "ETH"]

        run_fingerprint = {
            "chunk_period": 60,
            "n_assets": n_assets,
        }

        data_dict = get_data_dict(
            list_of_tickers,
            run_fingerprint,
            data_kind="historic",
            root=None,
            start_date_string="2023-06-01 00:00:00",
            end_time_string="2023-07-01 00:00:00",
        )

        return {
            "prices": jnp.array(data_dict["prices"]),
            "n_assets": n_assets,
        }

    def test_momentum_calculate_weights_historic(self, historic_data):
        """Test MomentumPool calculate_weights_via_scan with real BTC/ETH data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440

        pool = MomentumPool()
        params = make_momentum_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="MomentumPool calculate_weights_via_scan should match with real BTC/ETH data"
        )

    def test_power_channel_calculate_weights_historic(self, historic_data):
        """Test PowerChannelPool calculate_weights_via_scan with real BTC/ETH data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440

        pool = PowerChannelPool()
        params = make_power_channel_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="PowerChannelPool calculate_weights_via_scan should match with real BTC/ETH data"
        )

    def test_mean_reversion_calculate_weights_historic(self, historic_data):
        """Test MeanReversionChannelPool calculate_weights_via_scan with real BTC/ETH data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440

        pool = MeanReversionChannelPool()
        params = make_mean_reversion_channel_params(n_assets, chunk_period=chunk_period)
        run_fingerprint = make_calculate_weights_run_fingerprint(
            n_assets, chunk_period=chunk_period, bout_length=bout_length
        )
        start_index = jnp.array([burn_in])

        original_weights = pool.calculate_weights(
            params, run_fingerprint, prices, start_index, None
        )
        scan_weights = pool.calculate_weights_via_scan(
            params, run_fingerprint, prices, start_index, None
        )

        np.testing.assert_allclose(
            original_weights, scan_weights, rtol=1e-10, atol=1e-10,
            err_msg="MeanReversionChannelPool calculate_weights_via_scan should match with real BTC/ETH data"
        )

    def test_calculate_weights_multiple_params_historic(self, historic_data):
        """Test calculate_weights_via_scan with various parameters on real data."""
        n_assets = historic_data["n_assets"]
        prices = historic_data["prices"]
        chunk_period = 60
        bout_length = 2880
        burn_in = 1440

        pool = MomentumPool()
        start_index = jnp.array([burn_in])

        test_cases = [
            {"memory_days": 7.0, "k_per_day": 0.5},
            {"memory_days": 14.0, "k_per_day": 1.0},
            {"memory_days": 30.0, "k_per_day": 2.0},
        ]

        for case in test_cases:
            params = make_momentum_params(n_assets, chunk_period=chunk_period, **case)
            run_fingerprint = make_calculate_weights_run_fingerprint(
                n_assets, chunk_period=chunk_period, bout_length=bout_length
            )

            original_weights = pool.calculate_weights(
                params, run_fingerprint, prices, start_index, None
            )
            scan_weights = pool.calculate_weights_via_scan(
                params, run_fingerprint, prices, start_index, None
            )

            np.testing.assert_allclose(
                original_weights, scan_weights, rtol=1e-10, atol=1e-10,
                err_msg=f"Mismatch with params: {case}"
            )
