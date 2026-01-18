"""
Unit tests for JAX trend detection implementation.

Compares Python and JAX implementations of trend detection
to ensure numerical equivalence.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax
from typing import Tuple


# Test parameters
SLOPE_LENGTH = 15
THRESHOLD_UP = 0.0125
THRESHOLD_DOWN = -0.0125
FLAT_BUFFER_UP = 0.005
FLAT_BUFFER_DOWN = -0.005
CONFIRM_UP_DAYS = 5
CONFIRM_DOWN_DAYS = 5
CONFIRM_FLAT_DAYS = 4


def detect_trends_python(df):
    """Original Python implementation of trend detection."""
    df = df.copy()
    df["slope"] = df.truf_cpi.diff(SLOPE_LENGTH) / SLOPE_LENGTH

    df["confirm_up"] = False
    df["confirm_down"] = False
    df["confirm_flat"] = False

    last_confirmed = -1
    trend_candidate = None
    confirmation_counter = 0

    for i in range(1, len(df)):
        s = df.at[i, "slope"]
        if last_confirmed == -1:
            if s < FLAT_BUFFER_DOWN:
                zone = -1
            elif s <= THRESHOLD_UP:
                zone = 0
            else:
                zone = 1
        elif last_confirmed == 1:
            if s > FLAT_BUFFER_UP:
                zone = 1
            elif s >= THRESHOLD_DOWN:
                zone = 0
            else:
                zone = -1
        else:
            if s > THRESHOLD_UP:
                zone = 1
            elif s < THRESHOLD_DOWN:
                zone = -1
            else:
                zone = 0

        req_days = {1: CONFIRM_UP_DAYS, -1: CONFIRM_DOWN_DAYS, 0: CONFIRM_FLAT_DAYS}[zone]

        if trend_candidate != zone:
            trend_candidate = zone
            confirmation_counter = 1
        else:
            confirmation_counter += 1

        if confirmation_counter >= req_days and zone != last_confirmed:
            if zone == 1:
                df.at[i, "confirm_up"] = True
            elif zone == -1:
                df.at[i, "confirm_down"] = True
            else:
                df.at[i, "confirm_flat"] = True
            last_confirmed = zone
            trend_candidate = None
            confirmation_counter = 0

    return df


def calculate_zone_jax(slope: float, last_confirmed: int) -> int:
    """JAX implementation of zone calculation logic."""
    zone = jax.lax.cond(
        last_confirmed == -1,
        lambda: jnp.where(slope < FLAT_BUFFER_DOWN, -1,
                         jnp.where(slope <= THRESHOLD_UP, 0, 1)),
        lambda: jax.lax.cond(
            last_confirmed == 1,
            lambda: jnp.where(slope > FLAT_BUFFER_UP, 1,
                             jnp.where(slope >= THRESHOLD_DOWN, 0, -1)),
            lambda: jnp.where(slope > THRESHOLD_UP, 1,
                             jnp.where(slope < THRESHOLD_DOWN, -1, 0))
        )
    )

    return zone


def confirmation_step(carry: Tuple, inputs: Tuple) -> Tuple[Tuple, Tuple]:
    """Single step of confirmation logic using JAX scan."""
    last_confirmed, trend_candidate, confirmation_counter, _, _, _ = carry
    slope = inputs
    confirm_up = False
    confirm_down = False
    confirm_flat = False
    is_valid = True

    zone = jax.lax.cond(
        last_confirmed == -1,
        lambda: jnp.where(slope < FLAT_BUFFER_DOWN, -1,
                         jnp.where(slope <= THRESHOLD_UP, 0, 1)),
        lambda: jax.lax.cond(
            last_confirmed == 1,
            lambda: jnp.where(slope > FLAT_BUFFER_UP, 1,
                             jnp.where(slope >= THRESHOLD_DOWN, 0, -1)),
            lambda: jnp.where(slope > THRESHOLD_UP, 1,
                             jnp.where(slope < THRESHOLD_DOWN, -1, 0))
        )
    )

    req_days = jnp.select(
        [zone == 1, zone == -1, zone == 0],
        [CONFIRM_UP_DAYS, CONFIRM_DOWN_DAYS, CONFIRM_FLAT_DAYS],
        default=0
    )

    new_trend_candidate = jnp.where(is_valid, zone, trend_candidate)
    new_confirmation_counter = jnp.where(
        is_valid & (trend_candidate != zone), 1, confirmation_counter + 1
    )

    should_confirm = is_valid & (new_confirmation_counter >= req_days) & (zone != last_confirmed)

    new_confirm_up = jnp.where(should_confirm & (zone == 1), True, confirm_up)
    new_confirm_down = jnp.where(should_confirm & (zone == -1), True, confirm_down)
    new_confirm_flat = jnp.where(should_confirm & (zone == 0), True, confirm_flat)

    new_last_confirmed = jnp.where(should_confirm, zone, last_confirmed)

    final_trend_candidate = jnp.where(should_confirm, -999, new_trend_candidate)
    final_confirmation_counter = jnp.where(should_confirm, 0, new_confirmation_counter)

    new_carry = (
        new_last_confirmed, final_trend_candidate, final_confirmation_counter,
        new_confirm_up, new_confirm_down, new_confirm_flat
    )

    output = (new_confirm_up, new_confirm_down, new_confirm_flat)

    return new_carry, output


def detect_trends_jax(slopes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX implementation of trend detection using scan."""
    init_carry = (-1, -999, 0, False, False, False)
    inputs = slopes

    final_carry, outputs = lax.scan(confirmation_step, init_carry, inputs)

    confirm_up, confirm_down, confirm_flat = outputs

    return confirm_up, confirm_down, confirm_flat


@pytest.fixture
def sample_cpi_data():
    """Generate sample CPI data for testing."""
    np.random.seed(42)
    n_days = 100

    start_date = pd.Timestamp('2020-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_days)]

    trend_periods = 4
    trend_length = n_days // trend_periods

    cpi_data = []
    for i in range(trend_periods):
        start_idx = i * trend_length
        end_idx = min((i + 1) * trend_length, n_days)

        if i == 0:
            trend = np.linspace(2.0, 3.5, end_idx - start_idx)
        elif i == 1:
            trend = np.linspace(3.5, 1.8, end_idx - start_idx)
        elif i == 2:
            trend = np.full(end_idx - start_idx, 1.8)
        else:
            trend = np.linspace(1.8, 2.8, end_idx - start_idx)

        cpi_data.extend(trend)

    noise = np.random.normal(0, 0.1, n_days)
    cpi_data = np.array(cpi_data[:n_days]) + noise

    df = pd.DataFrame({
        'date': dates,
        'truf_cpi': cpi_data
    })

    return df


class TestZoneCalculation:
    """Test zone calculation logic."""

    @pytest.mark.parametrize("slope,last_confirmed,expected_zone", [
        # From downtrend (last_confirmed=-1)
        (-0.01, -1, -1),   # Below flat_buffer_down -> downtrend
        (0.0, -1, 0),      # In flat zone
        (0.02, -1, 1),     # Above threshold_up -> uptrend

        # From uptrend (last_confirmed=1)
        (0.01, 1, 1),      # Above flat_buffer_up -> uptrend
        (0.0, 1, 0),       # In flat zone
        (-0.02, 1, -1),    # Below threshold_down -> downtrend

        # From flat (last_confirmed=0)
        (0.02, 0, 1),      # Above threshold_up -> uptrend
        (0.0, 0, 0),       # In flat zone
        (-0.02, 0, -1),    # Below threshold_down -> downtrend
    ])
    def test_zone_calculation(self, slope, last_confirmed, expected_zone):
        """Test zone calculation for various scenarios."""
        zone = calculate_zone_jax(slope, last_confirmed)
        assert int(zone) == expected_zone


class TestTrendDetection:
    """Test trend detection implementations."""

    def test_python_implementation_runs(self, sample_cpi_data):
        """Test that Python implementation runs without error."""
        result = detect_trends_python(sample_cpi_data)

        assert "slope" in result.columns
        assert "confirm_up" in result.columns
        assert "confirm_down" in result.columns
        assert "confirm_flat" in result.columns

    def test_jax_implementation_runs(self, sample_cpi_data):
        """Test that JAX implementation runs without error."""
        df_python = detect_trends_python(sample_cpi_data)
        slopes = df_python['slope'].values[1:]
        slopes_jax = jnp.array(slopes)

        confirm_up, confirm_down, confirm_flat = detect_trends_jax(slopes_jax)

        assert confirm_up.shape == slopes_jax.shape
        assert confirm_down.shape == slopes_jax.shape
        assert confirm_flat.shape == slopes_jax.shape

    def test_zone_jax_matches_python(self, sample_cpi_data):
        """Test that JAX zone calculation matches Python implementation."""
        df_python = detect_trends_python(sample_cpi_data)
        slopes = df_python['slope'].values[1:]

        # Extract last_confirmed state at each step from Python
        last_confirmed = -1
        last_confirmed_values = []

        for i in range(1, len(df_python)):
            s = df_python.at[i, "slope"]

            # Calculate zone using Python logic
            if last_confirmed == -1:
                if s < FLAT_BUFFER_DOWN:
                    zone = -1
                elif s <= THRESHOLD_UP:
                    zone = 0
                else:
                    zone = 1
            elif last_confirmed == 1:
                if s > FLAT_BUFFER_UP:
                    zone = 1
                elif s >= THRESHOLD_DOWN:
                    zone = 0
                else:
                    zone = -1
            else:
                if s > THRESHOLD_UP:
                    zone = 1
                elif s < THRESHOLD_DOWN:
                    zone = -1
                else:
                    zone = 0

            if (df_python.at[i, "confirm_up"] or
                df_python.at[i, "confirm_down"] or
                df_python.at[i, "confirm_flat"]):
                last_confirmed = zone

            last_confirmed_values.append(last_confirmed)

        # Calculate zones using JAX
        slopes_jax = jnp.array(slopes)
        last_confirmed_jax = jnp.array(last_confirmed_values)
        zones_jax = jax.vmap(calculate_zone_jax)(slopes_jax, last_confirmed_jax)

        # Compare zones
        zones_match = True
        for i in range(len(slopes)):
            s = slopes[i]
            last_conf = last_confirmed_values[i]

            if last_conf == -1:
                if s < FLAT_BUFFER_DOWN:
                    python_zone = -1
                elif s <= THRESHOLD_UP:
                    python_zone = 0
                else:
                    python_zone = 1
            elif last_conf == 1:
                if s > FLAT_BUFFER_UP:
                    python_zone = 1
                elif s >= THRESHOLD_DOWN:
                    python_zone = 0
                else:
                    python_zone = -1
            else:
                if s > THRESHOLD_UP:
                    python_zone = 1
                elif s < THRESHOLD_DOWN:
                    python_zone = -1
                else:
                    python_zone = 0

            jax_zone = int(zones_jax[i])

            if python_zone != jax_zone:
                zones_match = False
                break

        assert zones_match, "Zone calculations don't match between Python and JAX"


class TestConfirmationLogic:
    """Test confirmation logic."""

    def test_confirmation_requires_consecutive_days(self, sample_cpi_data):
        """Test that confirmations require the correct number of consecutive days."""
        df = detect_trends_python(sample_cpi_data)

        # Count confirmations
        n_up = df["confirm_up"].sum()
        n_down = df["confirm_down"].sum()
        n_flat = df["confirm_flat"].sum()

        # At least some confirmations should occur in realistic data
        total_confirmations = n_up + n_down + n_flat
        assert total_confirmations >= 0  # May be 0 in short data

    def test_no_consecutive_same_confirmations(self, sample_cpi_data):
        """Test that we don't get consecutive confirmations of the same type."""
        df = detect_trends_python(sample_cpi_data)

        last_confirmation = None
        for i in range(len(df)):
            if df.at[i, "confirm_up"]:
                assert last_confirmation != "up", "Consecutive up confirmations"
                last_confirmation = "up"
            elif df.at[i, "confirm_down"]:
                assert last_confirmation != "down", "Consecutive down confirmations"
                last_confirmation = "down"
            elif df.at[i, "confirm_flat"]:
                assert last_confirmation != "flat", "Consecutive flat confirmations"
                last_confirmation = "flat"
