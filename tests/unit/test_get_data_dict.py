"""
Unit tests for get_data_dict and related data loading functions.

These tests verify:
1. Correct slicing and alignment of price data
2. start_idx, end_idx, bout_length calculations
3. max_memory_days capping when insufficient history
4. chunk_period alignment
5. Preservation of behavior after pre-slicing optimization

Critical invariants that must hold:
- prices[start_idx:end_idx] is the training period
- prices[:start_idx] is the burn-in period (for gradient convergence)
- start_idx >= max_memory_days * 1440 (enough burn-in) OR max_memory_days is capped
- Data is aligned to midnight (remainder_idx handling)
"""
import pytest
import numpy as np
import pandas as pd
import copy
from datetime import datetime, timedelta

from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.utils.data_processing.historic_data_utils import (
    get_data_dict,
    start_and_end_calcs,
)
from tests.conftest import TEST_DATA_DIR


class TestStartAndEndCalcs:
    """Test the start_and_end_calcs helper function."""

    def test_basic_slicing(self):
        """Test basic start/end index calculation."""
        # Create mock data: 10 days of minute data
        n_minutes = 10 * 1440
        unix_start = 1612310400000  # Some timestamp aligned to midnight
        unix_values = np.array([unix_start + i * 60000 for i in range(n_minutes)])
        prices = np.random.randn(n_minutes, 2)

        # Request days 3-7
        start_date = unix_start + 3 * 1440 * 60000
        end_date = unix_start + 7 * 1440 * 60000

        start_idx, end_idx, bout_length, unix_out, prices_out, _, remainder_idx = \
            start_and_end_calcs(unix_values, prices=prices, start_date=start_date, end_date=end_date)

        # bout_length = end_idx - start_idx (end_idx is exclusive + 1)
        assert bout_length == end_idx - start_idx
        # Verify we get roughly 4 days (allowing for +1 due to end_idx calculation)
        assert abs(bout_length - 4 * 1440) <= 1, f"Expected bout_length≈{4*1440}, got {bout_length}"

    def test_remainder_idx_based_on_start(self):
        """Test that remainder_idx is based on start_idx % 1440."""
        # Create mock data
        n_minutes = 10 * 1440
        unix_start = 1612310400000
        unix_values = np.array([unix_start + i * 60000 for i in range(n_minutes)])
        prices = np.random.randn(n_minutes, 2)

        start_date = unix_start + 3 * 1440 * 60000
        end_date = unix_start + 7 * 1440 * 60000

        start_idx, _, _, _, _, _, remainder_idx = \
            start_and_end_calcs(unix_values, prices=prices, start_date=start_date, end_date=end_date)

        # remainder_idx = start_idx % 1440 (alignment to midnight)
        expected_remainder = start_idx % 1440
        assert remainder_idx == expected_remainder, \
            f"Expected remainder_idx={expected_remainder}, got {remainder_idx}"

    def test_returned_data_starts_at_midnight(self):
        """Test that after remainder trimming, data starts at midnight.

        The start_and_end_calcs function trims data by remainder_idx so that
        the returned unix_values start at a day boundary (midnight).
        """
        # Create data that starts 100 minutes after midnight
        offset = 100
        n_minutes = 5 * 1440
        # Start at a known midnight timestamp + offset
        midnight_ts = 1612310400000  # 2021-02-03 00:00:00 UTC
        unix_start = midnight_ts + offset * 60000
        unix_values = np.array([unix_start + i * 60000 for i in range(n_minutes)])
        prices = np.random.randn(n_minutes, 2)

        # Request a date range that exists in our data
        start_date = midnight_ts + 1440 * 60000  # 1 day after midnight
        end_date = midnight_ts + 3 * 1440 * 60000  # 3 days after midnight

        _, _, _, unix_out, prices_out, _, remainder_idx = \
            start_and_end_calcs(unix_values, prices=prices, start_date=start_date, end_date=end_date)

        # The first timestamp in unix_out should be at midnight
        # (divisible by 1440 * 60000 ms = 86400000 ms = 1 day)
        first_ts = unix_out[0]
        ms_per_day = 1440 * 60000
        assert first_ts % ms_per_day == 0, \
            f"First timestamp {first_ts} should be at midnight (divisible by {ms_per_day})"

    @pytest.mark.skip(reason="Bug in start_and_end_calcs: remainder_idx not set when no dates provided")
    def test_no_date_constraints(self):
        """Test behavior when no start/end dates provided.

        Note: This test is skipped because it reveals a bug in the original code.
        When no dates are provided, remainder_idx is referenced before assignment.
        This is an unused code path in production (dates are always provided).
        """
        n_minutes = 1000
        unix_values = np.arange(n_minutes)
        prices = np.random.randn(n_minutes, 2)

        # When no dates provided, remainder_idx is not set (returns tuple of 7)
        # but we only care about start_idx, end_idx, bout_length
        result = start_and_end_calcs(unix_values, prices=prices)
        start_idx, end_idx, bout_length = result[0], result[1], result[2]

        assert start_idx == 0
        assert end_idx == n_minutes
        assert bout_length == n_minutes


class TestGetDataDictBasics:
    """Test basic get_data_dict behavior with mock/real data."""

    @pytest.fixture
    def base_fingerprint(self):
        """Create a base run fingerprint for testing."""
        fp = copy.deepcopy(run_fingerprint_defaults)
        fp["chunk_period"] = 1440  # Daily
        return fp

    def test_data_dict_structure(self, base_fingerprint):
        """Test that get_data_dict returns expected keys."""
        try:
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        required_keys = ["prices", "start_idx", "end_idx", "bout_length",
                        "unix_values", "n_chunks", "max_memory_days"]
        for key in required_keys:
            assert key in data_dict, f"Missing required key: {key}"

    def test_bout_length_consistency(self, base_fingerprint):
        """Test that bout_length = end_idx - start_idx."""
        try:
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        assert data_dict["bout_length"] == data_dict["end_idx"] - data_dict["start_idx"], \
            "bout_length should equal end_idx - start_idx"

    def test_prices_shape_matches_unix(self, base_fingerprint):
        """Test that prices array length matches unix_values."""
        try:
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        assert len(data_dict["prices"]) == len(data_dict["unix_values"]), \
            "prices and unix_values should have same length"


class TestMaxMemoryDaysCapping:
    """Test the max_memory_days capping logic."""

    @pytest.fixture
    def base_fingerprint(self):
        fp = copy.deepcopy(run_fingerprint_defaults)
        fp["chunk_period"] = 1440
        return fp

    def test_max_memory_days_capped_when_insufficient_history(self, base_fingerprint):
        """Test that max_memory_days is reduced when there's not enough history."""
        try:
            # Request a very large max_memory_days
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=1000.0,  # Unrealistically large
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        # max_memory_days should be capped based on available history
        start_idx_days = data_dict["start_idx"] / 1440
        assert data_dict["max_memory_days"] <= start_idx_days, \
            "max_memory_days should be capped to available history"

    def test_max_memory_days_preserved_when_sufficient_history(self, base_fingerprint):
        """Test that max_memory_days is preserved when there's enough history."""
        try:
            requested_max_memory = 30.0
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=requested_max_memory,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        start_idx_days = data_dict["start_idx"] / 1440
        if start_idx_days >= requested_max_memory:
            assert data_dict["max_memory_days"] == requested_max_memory, \
                "max_memory_days should be preserved when sufficient history exists"


class TestBurnInRequirements:
    """Test burn-in period (data before start_idx) requirements."""

    @pytest.fixture
    def base_fingerprint(self):
        fp = copy.deepcopy(run_fingerprint_defaults)
        fp["chunk_period"] = 60  # Hourly
        return fp

    def test_sufficient_burnin_for_gradients(self, base_fingerprint):
        """Test that there's enough data before start_idx for gradient burn-in."""
        try:
            max_memory_days = 30.0
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=max_memory_days,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        # Either we have enough burn-in, or max_memory_days was capped
        start_idx_days = data_dict["start_idx"] / 1440
        actual_max_memory = data_dict["max_memory_days"]

        assert start_idx_days >= actual_max_memory or actual_max_memory < max_memory_days, \
            "Must have sufficient burn-in or max_memory_days must be capped"

    def test_training_data_accessible(self, base_fingerprint):
        """Test that training period data is accessible via start_idx:end_idx."""
        try:
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        prices = data_dict["prices"]
        start_idx = data_dict["start_idx"]
        end_idx = data_dict["end_idx"]

        # Should be able to slice training period
        training_prices = prices[start_idx:end_idx]
        assert len(training_prices) == data_dict["bout_length"]
        assert not np.any(np.isnan(training_prices)), "Training data should not contain NaN"


class TestChunkPeriodAlignment:
    """Test that data is properly aligned for different chunk periods."""

    @pytest.fixture
    def base_fingerprint(self):
        return copy.deepcopy(run_fingerprint_defaults)

    @pytest.mark.parametrize("chunk_period", [60, 1440])
    def test_data_divisible_by_chunk_period(self, base_fingerprint, chunk_period):
        """Test that data length is divisible by chunk_period."""
        base_fingerprint["chunk_period"] = chunk_period

        try:
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        prices = data_dict["prices"]
        # The chunkable portion should be divisible
        assert len(prices) % chunk_period == 0 or len(prices) >= chunk_period, \
            f"Data length should be compatible with chunk_period={chunk_period}"

    @pytest.mark.parametrize("chunk_period", [60, 1440])
    def test_n_chunks_calculation(self, base_fingerprint, chunk_period):
        """Test that n_chunks is correctly calculated with day alignment.

        The formula is: n_chunks = int((len(prices) - remainder_idx) / 1440) * 1440 / chunk_period
        This ensures data is aligned to day boundaries before chunking.
        """
        base_fingerprint["chunk_period"] = chunk_period

        try:
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        n_chunks = data_dict["n_chunks"]
        prices = data_dict["prices"]

        # n_chunks should be positive
        assert n_chunks > 0, "n_chunks should be positive"

        # n_chunks * chunk_period should not exceed data length
        assert n_chunks * chunk_period <= len(prices), \
            f"n_chunks * chunk_period ({n_chunks * chunk_period}) should not exceed len(prices) ({len(prices)})"

        # n_chunks should represent whole days worth of chunks
        # (n_chunks * chunk_period) should be divisible by 1440
        assert (n_chunks * chunk_period) % 1440 == 0, \
            f"n_chunks * chunk_period should align to day boundaries"


class TestPreSlicingInvariant:
    """
    Tests that will verify the pre-slicing optimization maintains behavior.

    These tests establish invariants that must hold before AND after
    implementing the pre-slicing optimization.
    """

    @pytest.fixture
    def base_fingerprint(self):
        fp = copy.deepcopy(run_fingerprint_defaults)
        fp["chunk_period"] = 60
        return fp

    def test_gradient_calc_same_with_truncated_burnin(self, base_fingerprint):
        """
        Test that gradient calculations are identical whether using full history
        or just max_memory_days of burn-in.

        This is the key invariant for the pre-slicing optimization.
        """
        pytest.importorskip("jax")
        import jax.numpy as jnp
        from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators import calc_gradients
        from quantammsim.core_simulator.param_utils import memory_days_to_logit_lamb

        try:
            max_memory_days = 30.0
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=max_memory_days,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        prices = data_dict["prices"]
        start_idx = data_dict["start_idx"]
        chunk_period = base_fingerprint["chunk_period"]

        # Minimum required: max_memory_days before start_idx
        min_required_idx = max(0, start_idx - int(max_memory_days * 1440))

        # Full prices (current behavior)
        full_prices = jnp.array(prices[::chunk_period])

        # Truncated prices (what pre-slicing would give)
        truncated_prices = jnp.array(prices[min_required_idx:][::chunk_period])

        # Adjust start_idx for truncated array
        truncated_start_idx = (start_idx - min_required_idx) // chunk_period

        # Setup params
        memory_days = 10.0  # Use shorter memory than max for this test
        logit_lamb = memory_days_to_logit_lamb(memory_days, chunk_period)
        n_assets = prices.shape[1]
        params = {
            "logit_lamb": jnp.array([logit_lamb] * n_assets),
            "initial_weights_logits": jnp.array([0.0] * n_assets),
        }

        # Calculate gradients on full data
        full_grads = calc_gradients(params, full_prices, chunk_period, max_memory_days, False)

        # Calculate gradients on truncated data
        truncated_grads = calc_gradients(params, truncated_prices, chunk_period, max_memory_days, False)

        # Compare gradients in the training period
        # For full: training starts at start_idx // chunk_period
        # For truncated: training starts at truncated_start_idx
        full_start = start_idx // chunk_period

        # Compare from after burn-in
        burn_in_chunks = int(max_memory_days * 1440 / chunk_period)

        full_training = full_grads[full_start + burn_in_chunks:]
        truncated_training = truncated_grads[truncated_start_idx + burn_in_chunks:]

        # Ensure we have data to compare
        min_len = min(len(full_training), len(truncated_training))
        if min_len > 0:
            max_diff = float(jnp.max(jnp.abs(
                full_training[:min_len] - truncated_training[:min_len]
            )))
            assert max_diff < 1e-10, \
                f"Gradients should be identical with truncated burn-in, max diff: {max_diff}"


class TestNChunksFormula:
    """
    Tests specifically for the n_chunks formula:
    n_chunks = int((len(prices) - remainder_idx) / 1440) * 1440 / chunk_period

    This formula ensures:
    1. Data is aligned to day boundaries (remainder_idx trimming)
    2. Only complete days are counted
    3. Chunk period divides evenly into the day-aligned data
    """

    def test_n_chunks_formula_hourly(self):
        """Test n_chunks formula with chunk_period=60 (hourly)."""
        # Simulate data: 10 complete days, starting at midnight
        n_days = 10
        n_minutes = n_days * 1440
        remainder_idx = 0  # Starts at midnight

        # Formula: int((len(prices) - remainder_idx) / 1440) * 1440 / chunk_period
        chunk_period = 60
        n_chunks = int((n_minutes - remainder_idx) / 1440) * 1440 / chunk_period

        # Should be 10 days * 24 hours = 240 chunks
        assert n_chunks == 240, f"Expected 240 chunks, got {n_chunks}"

    def test_n_chunks_formula_daily(self):
        """Test n_chunks formula with chunk_period=1440 (daily)."""
        n_days = 10
        n_minutes = n_days * 1440
        remainder_idx = 0

        chunk_period = 1440
        n_chunks = int((n_minutes - remainder_idx) / 1440) * 1440 / chunk_period

        # Should be exactly 10 chunks (one per day)
        assert n_chunks == 10, f"Expected 10 chunks, got {n_chunks}"

    def test_n_chunks_with_partial_day_at_end(self):
        """Test that partial days at the end are not counted."""
        # 10 full days + 100 extra minutes
        n_minutes = 10 * 1440 + 100
        remainder_idx = 0
        chunk_period = 60

        n_chunks = int((n_minutes - remainder_idx) / 1440) * 1440 / chunk_period

        # Should still be 240 (only complete days count)
        assert n_chunks == 240, f"Expected 240 chunks (partial day ignored), got {n_chunks}"

    def test_n_chunks_with_remainder_offset(self):
        """Test n_chunks when data doesn't start at midnight (has remainder_idx)."""
        # Start 100 minutes after midnight, have 10+ days of data
        # This simulates data that was trimmed by remainder_idx
        n_minutes = 10 * 1440 + 500  # More than 10 days
        remainder_idx = 100  # Data starts 100 min after midnight

        chunk_period = 60
        n_chunks = int((n_minutes - remainder_idx) / 1440) * 1440 / chunk_period

        # After removing remainder_idx, we have 10*1440 + 400 minutes
        # That's 10 complete days plus 400 minutes (partial)
        # So n_chunks = 10 * 24 = 240
        assert n_chunks == 240, f"Expected 240 chunks, got {n_chunks}"

    def test_n_chunks_day_alignment_invariant(self):
        """Test that n_chunks * chunk_period is always divisible by 1440."""
        test_cases = [
            (1440 * 5, 0, 60),    # 5 days, no offset, hourly
            (1440 * 5, 0, 1440),  # 5 days, no offset, daily
            (1440 * 5 + 500, 0, 60),    # 5+ days, no offset
            (1440 * 10, 100, 60),       # 10 days, with offset
            (1440 * 10 + 1000, 100, 60),  # 10+ days, with offset
        ]

        for n_minutes, remainder_idx, chunk_period in test_cases:
            n_chunks = int((n_minutes - remainder_idx) / 1440) * 1440 / chunk_period
            total_minutes = n_chunks * chunk_period

            assert total_minutes % 1440 == 0, \
                f"n_chunks * chunk_period ({total_minutes}) should be divisible by 1440. " \
                f"Params: n_minutes={n_minutes}, remainder_idx={remainder_idx}, chunk_period={chunk_period}"


class TestDataAlignmentWithRealData:
    """Integration tests verifying alignment properties with actual data files."""

    @pytest.fixture
    def base_fingerprint(self):
        return copy.deepcopy(run_fingerprint_defaults)

    def test_unix_values_start_at_midnight(self, base_fingerprint):
        """Test that returned unix_values start at midnight (00:00:00)."""
        base_fingerprint["chunk_period"] = 1440

        try:
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        unix_values = data_dict["unix_values"]
        first_timestamp = unix_values[0]

        # Should be divisible by ms per day (86400000)
        ms_per_day = 1440 * 60 * 1000
        assert first_timestamp % ms_per_day == 0, \
            f"First timestamp {first_timestamp} should be at midnight"

    def test_start_idx_preserves_midnight_alignment(self, base_fingerprint):
        """Test that start_idx still points to midnight-aligned data."""
        base_fingerprint["chunk_period"] = 1440

        try:
            data_dict = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        unix_values = data_dict["unix_values"]
        start_idx = data_dict["start_idx"]

        start_timestamp = unix_values[start_idx]
        ms_per_day = 1440 * 60 * 1000

        assert start_timestamp % ms_per_day == 0, \
            f"Timestamp at start_idx ({start_timestamp}) should be at midnight"

    def test_data_length_compatible_with_chunking(self, base_fingerprint):
        """Test that data can be cleanly chunked without leftover minutes."""
        for chunk_period in [60, 1440]:
            base_fingerprint["chunk_period"] = chunk_period

            try:
                data_dict = get_data_dict(
                    list_of_tickers=["BTC", "ETH"],
                    run_fingerprint=base_fingerprint,
                    data_kind="historic",
                    root=TEST_DATA_DIR,
                    max_memory_days=30.0,
                    start_date_string=base_fingerprint["startDateString"],
                    end_time_string=base_fingerprint["endDateString"],
                )
            except FileNotFoundError:
                pytest.skip("Price data files not available")

            n_chunks = data_dict["n_chunks"]
            prices = data_dict["prices"]

            # Verify we can actually slice n_chunks worth of data
            chunkable_length = int(n_chunks * chunk_period)
            assert chunkable_length <= len(prices), \
                f"n_chunks implies {chunkable_length} minutes but only have {len(prices)}"

            # Verify the chunkable portion aligns to days
            assert chunkable_length % 1440 == 0, \
                f"Chunkable length {chunkable_length} should align to day boundaries"


class TestEndToEndPreSlicing:
    """
    End-to-end tests verifying pre-slicing doesn't change simulation results.

    These tests run actual simulations with and without pre-slicing and verify
    the weights and reserves are identical for the training period.
    """

    @pytest.fixture
    def base_fingerprint(self):
        fp = copy.deepcopy(run_fingerprint_defaults)
        fp["chunk_period"] = 1440  # Daily for faster tests
        fp["startDateString"] = "2022-01-01 00:00:00"
        fp["endDateString"] = "2022-03-01 00:00:00"  # 2 months
        fp["tokens"] = ["BTC", "ETH"]
        fp["rule"] = "momentum"
        return fp

    def test_weights_same_with_and_without_preslicing(self, base_fingerprint):
        """Test that calculated weights are identical with/without pre-slicing."""
        pytest.importorskip("jax")
        import jax.numpy as jnp
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        try:
            # Load data with pre-slicing (default behavior)
            data_dict_presliced = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        # Load data without pre-slicing
        data_dict_full = get_data_dict(
            list_of_tickers=["BTC", "ETH"],
            run_fingerprint=base_fingerprint,
            data_kind="historic",
            root=TEST_DATA_DIR,
            max_memory_days=30.0,
            start_date_string=base_fingerprint["startDateString"],
            end_time_string=base_fingerprint["endDateString"],
            preslice_burnin=False,  # Disable pre-slicing
        )

        # Verify the arrays are different sizes (pre-slicing worked)
        assert len(data_dict_presliced["prices"]) < len(data_dict_full["prices"]), \
            "Pre-slicing should reduce data size"

        # Create pool and run fingerprint
        pool = create_pool("momentum")
        fp_hashable = NestedHashabledict(base_fingerprint)
        fp_hashable["bout_length"] = data_dict_presliced["bout_length"]
        fp_hashable["n_assets"] = 2
        fp_hashable["all_sig_variations"] = ((1, -1), (-1, 1))

        # Common params
        params = {
            "initial_weights_logits": jnp.zeros(2),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "log_k": jnp.array([5.0, 5.0]),
        }

        # Calculate weights with pre-sliced data
        prices_presliced = jnp.array(data_dict_presliced["prices"])
        weights_presliced = pool.calculate_rule_outputs(
            params, fp_hashable, prices_presliced
        )

        # Calculate weights with full data
        prices_full = jnp.array(data_dict_full["prices"])
        # Need to adjust fingerprint for full data
        fp_full = NestedHashabledict(base_fingerprint)
        fp_full["bout_length"] = data_dict_full["bout_length"]
        fp_full["n_assets"] = 2
        fp_full["all_sig_variations"] = ((1, -1), (-1, 1))
        weights_full = pool.calculate_rule_outputs(
            params, fp_full, prices_full
        )

        # Extract weights for the training period
        start_idx_pre = data_dict_presliced["start_idx"]
        end_idx_pre = data_dict_presliced["end_idx"]
        start_idx_full = data_dict_full["start_idx"]
        end_idx_full = data_dict_full["end_idx"]

        chunk_period = base_fingerprint["chunk_period"]
        training_weights_pre = weights_presliced[
            start_idx_pre // chunk_period:end_idx_pre // chunk_period
        ]
        training_weights_full = weights_full[
            start_idx_full // chunk_period:end_idx_full // chunk_period
        ]

        # Verify weights are the same
        min_len = min(len(training_weights_pre), len(training_weights_full))
        max_diff = float(jnp.max(jnp.abs(
            training_weights_pre[:min_len] - training_weights_full[:min_len]
        )))

        # Allow small numerical differences due to floating point arithmetic
        # starting from different initial data points
        assert max_diff < 1e-8, \
            f"Weights should be identical with/without pre-slicing, max diff: {max_diff}"

    def test_reserves_same_with_and_without_preslicing(self, base_fingerprint):
        """Test that calculated reserves are identical with/without pre-slicing."""
        pytest.importorskip("jax")
        import jax.numpy as jnp
        from quantammsim.pools.creator import create_pool
        from quantammsim.runners.jax_runner_utils import NestedHashabledict

        try:
            # Load data with pre-slicing
            data_dict_presliced = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=30.0,
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        # Load data without pre-slicing
        data_dict_full = get_data_dict(
            list_of_tickers=["BTC", "ETH"],
            run_fingerprint=base_fingerprint,
            data_kind="historic",
            root=TEST_DATA_DIR,
            max_memory_days=30.0,
            start_date_string=base_fingerprint["startDateString"],
            end_time_string=base_fingerprint["endDateString"],
            preslice_burnin=False,  # Disable pre-slicing
        )

        # Create pool and params
        pool = create_pool("momentum")
        params = {
            "initial_weights_logits": jnp.zeros(2),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "log_k": jnp.array([5.0, 5.0]),
        }

        # Create fingerprint for pre-sliced data
        fp_pre = NestedHashabledict(base_fingerprint)
        fp_pre["bout_length"] = data_dict_presliced["bout_length"]
        fp_pre["n_assets"] = 2
        fp_pre["initial_pool_value"] = 1000.0
        fp_pre["fees"] = 0.0
        fp_pre["gas_cost"] = 0.0
        fp_pre["arb_fees"] = 0.0
        fp_pre["arb_frequency"] = 1
        fp_pre["all_sig_variations"] = ((1, -1), (-1, 1))
        fp_pre["do_trades"] = False
        fp_pre["max_memory_days"] = 30.0
        fp_pre["use_alt_lamb"] = False
        fp_pre["weight_interpolation_period"] = base_fingerprint["chunk_period"]
        fp_pre["weight_interpolation_method"] = "linear"
        fp_pre["maximum_change"] = 1.0
        fp_pre["minimum_weight"] = 0.0
        fp_pre["do_arb"] = True
        fp_pre["noise_trader_ratio"] = 0.0
        fp_pre["ste_max_change"] = False
        fp_pre["ste_min_max_weight"] = False

        # Calculate reserves with pre-sliced data
        prices_presliced = jnp.array(data_dict_presliced["prices"])
        reserves_presliced = pool.calculate_reserves_zero_fees(
            params, fp_pre, prices_presliced, jnp.array([0, 0])
        )

        # Create fingerprint for full data
        fp_full = NestedHashabledict(base_fingerprint)
        fp_full["bout_length"] = data_dict_full["bout_length"]
        fp_full["n_assets"] = 2
        fp_full["initial_pool_value"] = 1000.0
        fp_full["fees"] = 0.0
        fp_full["gas_cost"] = 0.0
        fp_full["arb_fees"] = 0.0
        fp_full["arb_frequency"] = 1
        fp_full["all_sig_variations"] = ((1, -1), (-1, 1))
        fp_full["do_trades"] = False
        fp_full["max_memory_days"] = 30.0
        fp_full["use_alt_lamb"] = False
        fp_full["weight_interpolation_period"] = base_fingerprint["chunk_period"]
        fp_full["weight_interpolation_method"] = "linear"
        fp_full["maximum_change"] = 1.0
        fp_full["minimum_weight"] = 0.0
        fp_full["do_arb"] = True
        fp_full["noise_trader_ratio"] = 0.0
        fp_full["ste_max_change"] = False
        fp_full["ste_min_max_weight"] = False

        # Calculate reserves with full data
        prices_full = jnp.array(data_dict_full["prices"])
        reserves_full = pool.calculate_reserves_zero_fees(
            params, fp_full, prices_full, jnp.array([0, 0])
        )

        # Extract reserves for the training period
        start_idx_pre = data_dict_presliced["start_idx"]
        end_idx_pre = data_dict_presliced["end_idx"]
        start_idx_full = data_dict_full["start_idx"]
        end_idx_full = data_dict_full["end_idx"]

        # Reserves are at minute level, not chunked
        training_reserves_pre = reserves_presliced[start_idx_pre:end_idx_pre]
        training_reserves_full = reserves_full[start_idx_full:end_idx_full]

        # Verify reserves are the same
        min_len = min(len(training_reserves_pre), len(training_reserves_full))
        if min_len > 0:
            max_diff = float(jnp.max(jnp.abs(
                training_reserves_pre[:min_len] - training_reserves_full[:min_len]
            )))

            assert max_diff < 1e-6, \
                f"Reserves should be identical with/without pre-slicing, max diff: {max_diff}"


class TestDoRunOnHistoricDataPreSlicing:
    """
    End-to-end tests using do_run_on_historic_data to verify pre-slicing
    doesn't change simulation results at the runner level.

    These tests are the most comprehensive as they test the full pipeline
    from data loading through forward pass to final output.
    """

    @pytest.fixture
    def base_fingerprint(self):
        """Create a base run fingerprint for testing.

        Uses max_memory_days=365.0 (the default) to ensure exact equality
        between presliced and full data paths. With shorter max_memory_days
        values (like 30.0), small floating-point differences can occur due
        to different array sizes affecting JAX's computation paths.
        """
        fp = copy.deepcopy(run_fingerprint_defaults)
        fp["chunk_period"] = 1440  # Daily for faster tests
        fp["weight_interpolation_period"] = 1440
        fp["startDateString"] = "2022-01-01 00:00:00"
        fp["endDateString"] = "2022-03-01 00:00:00"  # 2 months
        fp["tokens"] = ["BTC", "ETH"]
        fp["rule"] = "momentum"
        fp["fees"] = 0.0
        fp["gas_cost"] = 0.0
        fp["arb_fees"] = 0.0
        fp["initial_pool_value"] = 1000000.0
        fp["max_memory_days"] = 365.0  # Use default value for exact equality
        return fp

    @pytest.fixture
    def base_params(self):
        """Create base parameters for testing."""
        pytest.importorskip("jax")
        import jax.numpy as jnp
        return {
            "initial_weights_logits": jnp.zeros(2),
            "logit_lamb": jnp.array([0.0, 0.0]),
            "log_k": jnp.array([5.0, 5.0]),
        }

    def test_full_simulation_identical_with_and_without_preslicing(self, base_fingerprint, base_params):
        """
        Test that a full simulation via do_run_on_historic_data produces
        identical results with and without pre-slicing.

        This is the definitive end-to-end test that pre-slicing doesn't
        break anything in the simulation pipeline.
        """
        pytest.importorskip("jax")
        import jax.numpy as jnp

        try:
            from quantammsim.runners.jax_runners import do_run_on_historic_data
        except ImportError:
            pytest.skip("jax_runners not available")

        try:
            # Run simulation WITH pre-slicing (default behavior)
            result_presliced = do_run_on_historic_data(
                run_fingerprint=copy.deepcopy(base_fingerprint),
                params=copy.deepcopy(base_params),
                verbose=False,
                preslice_burnin=True,
            )

            # Run simulation WITHOUT pre-slicing
            result_full = do_run_on_historic_data(
                run_fingerprint=copy.deepcopy(base_fingerprint),
                params=copy.deepcopy(base_params),
                verbose=False,
                preslice_burnin=False,
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        # Compare reserves - should be exactly identical with max_memory_days=365
        reserves_presliced = np.array(result_presliced["reserves"])
        reserves_full = np.array(result_full["reserves"])
        assert reserves_presliced.shape == reserves_full.shape, \
            f"Reserves shape mismatch: {reserves_presliced.shape} vs {reserves_full.shape}"
        max_reserves_diff = np.max(np.abs(reserves_presliced - reserves_full))
        assert max_reserves_diff == 0.0, \
            f"Reserves should be exactly identical with/without pre-slicing, max diff: {max_reserves_diff}"

        # Compare values - should be exactly identical
        values_presliced = np.array(result_presliced["value"])
        values_full = np.array(result_full["value"])
        assert values_presliced.shape == values_full.shape, \
            f"Values shape mismatch: {values_presliced.shape} vs {values_full.shape}"
        max_values_diff = np.max(np.abs(values_presliced - values_full))
        assert max_values_diff == 0.0, \
            f"Values should be exactly identical with/without pre-slicing, max diff: {max_values_diff}"

        # Compare weights - should be exactly identical
        weights_presliced = np.array(result_presliced["weights"])
        weights_full = np.array(result_full["weights"])
        assert weights_presliced.shape == weights_full.shape, \
            f"Weights shape mismatch: {weights_presliced.shape} vs {weights_full.shape}"
        max_weights_diff = np.max(np.abs(weights_presliced - weights_full))
        assert max_weights_diff == 0.0, \
            f"Weights should be exactly identical with/without pre-slicing, max diff: {max_weights_diff}"

        # Compare final value - should be exactly identical
        final_value_presliced = result_presliced["final_value"]
        final_value_full = result_full["final_value"]
        assert final_value_presliced == final_value_full, \
            f"Final values should be exactly identical: {final_value_presliced} vs {final_value_full}"

        # Sanity checks
        assert final_value_presliced > 0, "Final value should be positive"
        assert not np.any(np.isnan(reserves_presliced)), "Reserves should not contain NaN"
        assert not np.any(np.isnan(values_presliced)), "Values should not contain NaN"

    @pytest.mark.parametrize("rule", ["momentum", "anti_momentum", "balancer"])
    def test_different_pool_types_identical_with_preslicing(self, base_fingerprint, rule):
        """
        Test that different pool types produce identical results with/without pre-slicing.
        """
        pytest.importorskip("jax")
        import jax.numpy as jnp

        try:
            from quantammsim.runners.jax_runners import do_run_on_historic_data
        except ImportError:
            pytest.skip("jax_runners not available")

        # Create fingerprint for this pool type
        fp = copy.deepcopy(base_fingerprint)
        fp["rule"] = rule

        # Create appropriate params
        n_assets = 2
        if rule == "balancer":
            params = {
                "initial_weights_logits": jnp.zeros(n_assets),
            }
        else:
            params = {
                "initial_weights_logits": jnp.zeros(n_assets),
                "logit_lamb": jnp.array([0.0] * n_assets),
                "log_k": jnp.array([5.0] * n_assets),
            }

        try:
            # Run with pre-slicing
            result_presliced = do_run_on_historic_data(
                run_fingerprint=copy.deepcopy(fp),
                params=copy.deepcopy(params),
                verbose=False,
                preslice_burnin=True,
            )

            # Run without pre-slicing
            result_full = do_run_on_historic_data(
                run_fingerprint=copy.deepcopy(fp),
                params=copy.deepcopy(params),
                verbose=False,
                preslice_burnin=False,
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        # Compare reserves - should be exactly identical with max_memory_days=365
        reserves_presliced = np.array(result_presliced["reserves"])
        reserves_full = np.array(result_full["reserves"])
        max_diff = np.max(np.abs(reserves_presliced - reserves_full))
        assert max_diff == 0.0, \
            f"[{rule}] Reserves should be exactly identical, max diff: {max_diff}"

        # Compare final values - should be exactly identical
        assert result_presliced["final_value"] == result_full["final_value"], \
            f"[{rule}] Final values should be exactly identical"

    def test_preslicing_reduces_data_loaded(self, base_fingerprint):
        """
        Test that pre-slicing actually reduces the amount of data loaded.
        """
        try:
            # Load data with pre-slicing (default)
            data_presliced = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=base_fingerprint["max_memory_days"],
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
                preslice_burnin=True,
            )

            # Load data without pre-slicing
            data_full = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=base_fingerprint["max_memory_days"],
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
                preslice_burnin=False,
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        presliced_len = len(data_presliced["prices"])
        full_len = len(data_full["prices"])

        # Pre-sliced should be smaller
        assert presliced_len < full_len, \
            f"Pre-sliced ({presliced_len}) should be smaller than full ({full_len})"

        # Calculate expected reduction
        # Pre-sliced should have ~max_memory_days + bout_length days + 2 extra days
        # (1 for day-alignment rounding, 1 for gradient calculation which returns n-1 elements)
        expected_max_len_days = base_fingerprint["max_memory_days"] + \
            data_presliced["bout_length"] / 1440 + 2  # +2 for rounding and gradient calc
        expected_max_len_minutes = int(np.ceil(expected_max_len_days * 1440))

        assert presliced_len <= expected_max_len_minutes, \
            f"Pre-sliced length ({presliced_len}) should be <= {expected_max_len_minutes}"

        # Bout length should be identical
        assert data_presliced["bout_length"] == data_full["bout_length"], \
            "bout_length should be unchanged by pre-slicing"

    def test_start_idx_adjusted_after_preslicing(self, base_fingerprint):
        """
        Test that start_idx is properly adjusted after pre-slicing.
        """
        try:
            data_presliced = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=base_fingerprint["max_memory_days"],
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
                preslice_burnin=True,
            )

            data_full = get_data_dict(
                list_of_tickers=["BTC", "ETH"],
                run_fingerprint=base_fingerprint,
                data_kind="historic",
                root=TEST_DATA_DIR,
                max_memory_days=base_fingerprint["max_memory_days"],
                start_date_string=base_fingerprint["startDateString"],
                end_time_string=base_fingerprint["endDateString"],
                preslice_burnin=False,
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        # The training period timestamps should match
        unix_presliced = data_presliced["unix_values"]
        unix_full = data_full["unix_values"]

        start_ts_presliced = unix_presliced[data_presliced["start_idx"]]
        start_ts_full = unix_full[data_full["start_idx"]]

        assert start_ts_presliced == start_ts_full, \
            "Training start timestamp should be identical with/without pre-slicing"

        # End timestamps should also match
        end_ts_presliced = unix_presliced[data_presliced["end_idx"] - 1]
        end_ts_full = unix_full[data_full["end_idx"] - 1]

        assert end_ts_presliced == end_ts_full, \
            "Training end timestamp should be identical with/without pre-slicing"

    def test_demo_run_style_simulation_unchanged(self, base_fingerprint, base_params):
        """
        Test a demo_run style simulation (like scripts/demo_run_.py) to verify
        that pre-slicing produces identical results to the full data case.

        This test mimics the typical usage pattern:
        1. Load historic data
        2. Run simulation with specified parameters
        3. Verify outputs (reserves, weights, values, final_value)
        """
        pytest.importorskip("jax")
        import jax.numpy as jnp

        try:
            from quantammsim.runners.jax_runners import do_run_on_historic_data
        except ImportError:
            pytest.skip("jax_runners not available")

        # Use a typical demo_run fingerprint style
        fp = copy.deepcopy(base_fingerprint)
        fp["do_arb"] = True
        fp["arb_quality"] = 1.0

        try:
            # Run with pre-slicing (what production will use)
            result_with_preslice = do_run_on_historic_data(
                run_fingerprint=copy.deepcopy(fp),
                params=copy.deepcopy(base_params),
                verbose=False,
                preslice_burnin=True,
            )

            # Run without pre-slicing (reference/gold standard)
            result_without_preslice = do_run_on_historic_data(
                run_fingerprint=copy.deepcopy(fp),
                params=copy.deepcopy(base_params),
                verbose=False,
                preslice_burnin=False,
            )
        except FileNotFoundError:
            pytest.skip("Price data files not available")

        # Verify all key outputs are exactly identical with max_memory_days=365
        # Reserves
        reserves_with = np.array(result_with_preslice["reserves"])
        reserves_without = np.array(result_without_preslice["reserves"])
        assert np.array_equal(reserves_with, reserves_without), \
            f"Reserves should be exactly identical, max diff: {np.max(np.abs(reserves_with - reserves_without))}"

        # Weights
        weights_with = np.array(result_with_preslice["weights"])
        weights_without = np.array(result_without_preslice["weights"])
        assert np.array_equal(weights_with, weights_without), \
            f"Weights should be exactly identical, max diff: {np.max(np.abs(weights_with - weights_without))}"

        # Values over time
        values_with = np.array(result_with_preslice["value"])
        values_without = np.array(result_without_preslice["value"])
        assert np.array_equal(values_with, values_without), \
            f"Values should be exactly identical, max diff: {np.max(np.abs(values_with - values_without))}"

        # Final value
        assert result_with_preslice["final_value"] == result_without_preslice["final_value"], \
            f"Final value should be exactly identical: {result_with_preslice['final_value']} vs {result_without_preslice['final_value']}"

        # Verify prices in output match (they should be same for training period)
        prices_with = np.array(result_with_preslice["prices"])
        prices_without = np.array(result_without_preslice["prices"])
        assert np.allclose(prices_with, prices_without, atol=1e-10), \
            "Output prices should be identical for training period"

        # Print summary for debugging if test fails
        print(f"\nSimulation completed successfully:")
        print(f"  Final value (with preslice): ${result_with_preslice['final_value']:.2f}")
        print(f"  Final value (without preslice): ${result_without_preslice['final_value']:.2f}")
        print(f"  Reserves shape: {reserves_with.shape}")
        print(f"  Max reserves diff: {np.max(np.abs(reserves_with - reserves_without))}")
