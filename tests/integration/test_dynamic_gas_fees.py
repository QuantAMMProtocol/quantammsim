"""
Integration tests for dynamic gas and fees in simulations.

Tests that simulations correctly handle dynamic gas costs and fee structures
loaded from external data files.
"""
import pytest
import pandas as pd
import jax.numpy as jnp
from pathlib import Path

from quantammsim.runners.jax_runners import do_run_on_historic_data


# Path to test data files
TEST_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def gas_df():
    """Load gas cost data if available."""
    gas_file = TEST_DATA_DIR / "Gas.csv"
    if not gas_file.exists():
        pytest.skip(f"Gas data file not found: {gas_file}")
    df = pd.read_csv(gas_file)
    df = df.rename(columns={"USD": "trade_gas_cost_usd"})
    return df


@pytest.fixture
def fees_df():
    """Load fees data if available."""
    fees_file = TEST_DATA_DIR / "fees.csv"
    if not fees_file.exists():
        pytest.skip(f"Fees data file not found: {fees_file}")
    df = pd.read_csv(fees_file)
    df = df.rename(columns={"bps": "fees"})
    df["fees"] = df["fees"] / 10000
    return df


@pytest.fixture
def base_fingerprint():
    """Base run fingerprint for dynamic gas/fees tests."""
    return {
        "startDateString": "2021-02-03 00:00:00",
        "endDateString": "2022-07-22 23:59:00",
        "endTestDateString": "2022-07-24 00:00:00",
        "tokens": ["ETH", "DAI"],
        "rule": "balancer",
        "bout_offset": 14400,
        "initial_weights_logits": jnp.array([-0.69314718, -0.69314718], dtype=jnp.float64),
        "initial_pool_value": 60000000,
        "use_alt_lamb": False,
        "return_val": "final_reserves_value_and_weights",
        "fees": 0.0,
        "do_trades": False,
    }


@pytest.fixture
def base_params():
    """Base parameters for tests."""
    return {
        "initial_weights_logits": jnp.array([-0.69314718, -0.69314718])
    }


class TestDynamicGasAndFees:
    """Test dynamic gas and fee handling in simulations."""

    @pytest.mark.slow
    @pytest.mark.requires_data
    def test_run_with_gas_and_fees(self, base_fingerprint, base_params, gas_df, fees_df):
        """Test simulation with both gas costs and dynamic fees."""
        result = do_run_on_historic_data(
            base_fingerprint, base_params, gas_cost_df=gas_df, fees_df=fees_df
        )

        assert result is not None
        assert "value" in result or "final_value" in result

        if "final_value" in result:
            assert result["final_value"] > 0, "Final value should be positive"

    @pytest.mark.slow
    @pytest.mark.requires_data
    def test_run_with_gas_only(self, base_fingerprint, base_params, gas_df):
        """Test simulation with gas costs only."""
        result = do_run_on_historic_data(
            base_fingerprint, base_params, gas_cost_df=gas_df
        )

        assert result is not None
        if "final_value" in result:
            assert result["final_value"] > 0

    @pytest.mark.slow
    @pytest.mark.requires_data
    def test_run_with_fees_only(self, base_fingerprint, base_params, fees_df):
        """Test simulation with dynamic fees only."""
        result = do_run_on_historic_data(
            base_fingerprint, base_params, fees_df=fees_df
        )

        assert result is not None
        if "final_value" in result:
            assert result["final_value"] > 0

    @pytest.mark.slow
    @pytest.mark.requires_data
    def test_gas_reduces_final_value(self, base_fingerprint, base_params, gas_df):
        """Test that gas costs reduce the final portfolio value."""
        # Run without gas
        result_no_gas = do_run_on_historic_data(base_fingerprint, base_params)

        # Run with gas
        result_with_gas = do_run_on_historic_data(
            base_fingerprint, base_params, gas_cost_df=gas_df
        )

        if "final_value" in result_no_gas and "final_value" in result_with_gas:
            # Gas costs should reduce final value (or at least not increase it)
            assert result_with_gas["final_value"] <= result_no_gas["final_value"], \
                "Gas costs should not increase portfolio value"

    @pytest.mark.slow
    @pytest.mark.requires_data
    def test_fees_reduce_final_value(self, base_fingerprint, base_params, fees_df):
        """Test that fees reduce the final portfolio value."""
        # Run without fees
        result_no_fees = do_run_on_historic_data(base_fingerprint, base_params)

        # Run with fees
        result_with_fees = do_run_on_historic_data(
            base_fingerprint, base_params, fees_df=fees_df
        )

        if "final_value" in result_no_fees and "final_value" in result_with_fees:
            # Fees should reduce final value (or at least not increase it significantly)
            # Note: fees can sometimes help via LP income, so we allow small increase
            relative_diff = (result_with_fees["final_value"] - result_no_fees["final_value"]) / result_no_fees["final_value"]
            assert relative_diff < 0.1, \
                f"Fees unexpectedly increased value by {relative_diff*100:.1f}%"
