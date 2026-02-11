"""
Full e2e tests for the /api/runSimulation code path.

Two levels of coverage:
  1. DTO → run_pool_simulation  (tests the DTO parsing + full simulation)
  2. HTTP POST → Flask endpoint  (tests the complete HTTP round-trip)

Both levels patch `get_historic_parquet_data` to use test-data parquet files.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np

from quantammsim.apis.rest_apis.simulator_dtos.simulation_run_dto import (
    SimulationRunDto,
)
from quantammsim.simulator_analysis_tools.finance.param_financial_calculator import (
    run_pool_simulation,
)
from quantammsim.apis.rest_apis.flask_server.flask_index import app
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_parquet_data,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "data"

# Patch targets — the imports inside param_financial_calculator
_PARQUET_PATCH_TARGET = (
    "quantammsim.simulator_analysis_tools.finance"
    ".param_financial_calculator.get_historic_parquet_data"
)
_DTB3_PATCH_TARGET = (
    "quantammsim.simulator_analysis_tools.finance"
    ".param_financial_calculator.filter_dtb3_values"
)


def _get_test_parquet_data(list_of_tickers, cols=["close"], root=None, **kwargs):
    """Redirect get_historic_parquet_data to test data directory."""
    return get_historic_parquet_data(
        list_of_tickers, cols=cols, root=str(TEST_DATA_DIR), **kwargs
    )


def _mock_filter_dtb3_values(filename, start_date, end_date, **kwargs):
    """Return a constant risk-free rate array for the date range.

    DTB3.csv is not tracked in the repo so we mock it with a flat 5% annual rate.
    Mimics the real filter_dtb3_values: truncate to midnight, inclusive endpoints.
    """
    import pandas as pd

    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    n_days = (end - start).days + 1  # inclusive of both endpoints
    return np.full(n_days, 0.05)  # 5% annual rate


# ---------------------------------------------------------------------------
# JSON request body builders
# ---------------------------------------------------------------------------

def _make_momentum_json(tokens, start, end, memory_days=10.0, k_per_day=20.0):
    """Build a JSON dict for a momentum pool request."""
    n = len(tokens)
    equal_value = 1_000_000.0 / n
    return {
        "startUnix": 0,
        "endUnix": 0,
        "startDateString": start,
        "endDateString": end,
        "pool": {
            "id": "test-momentum",
            "poolConstituents": [
                {
                    "coinCode": tok,
                    "marketValue": equal_value,
                    "currentPrice": 1.0,
                    "amount": equal_value,
                    "weight": 1.0 / n,
                }
                for tok in tokens
            ],
            "updateRule": {
                "name": "momentum",
                "UpdateRuleParameters": [
                    {"name": "memory_days", "value": [memory_days] * n},
                    {"name": "k_per_day", "value": [k_per_day] * n},
                    {"name": "chunk_period", "value": [1440]},
                    {"name": "weight_interpolation_period", "value": [1440]},
                    {"name": "arb_quality", "value": [1.0]},
                    {"name": "noise_trader_ratio", "value": [0.0]},
                ],
            },
            "enableAutomaticArbBots": True,
            "poolNumeraireCoinCode": None,
        },
        "feeHooks": [],
        "swapImports": [],
        "gasPriceImports": [],
    }


def _make_balancer_json(tokens, start, end):
    """Build a JSON dict for a balancer pool request."""
    n = len(tokens)
    equal_value = 1_000_000.0 / n
    return {
        "startUnix": 0,
        "endUnix": 0,
        "startDateString": start,
        "endDateString": end,
        "pool": {
            "id": "test-balancer",
            "poolConstituents": [
                {
                    "coinCode": tok,
                    "marketValue": equal_value,
                    "currentPrice": 1.0,
                    "amount": equal_value,
                    "weight": 1.0 / n,
                }
                for tok in tokens
            ],
            "updateRule": {
                "name": "balancer",
                "UpdateRuleParameters": [
                    {"name": "chunk_period", "value": [1440]},
                    {"name": "weight_interpolation_period", "value": [1440]},
                    {"name": "arb_quality", "value": [1.0]},
                    {"name": "noise_trader_ratio", "value": [0.0]},
                ],
            },
            "enableAutomaticArbBots": True,
            "poolNumeraireCoinCode": None,
        },
        "feeHooks": [],
        "swapImports": [],
        "gasPriceImports": [],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_via_dto(json_body):
    """Parse JSON into a SimulationRunDto and call run_pool_simulation."""
    dto = SimulationRunDto(json_body)
    with patch(_PARQUET_PATCH_TARGET, side_effect=_get_test_parquet_data), \
         patch(_DTB3_PATCH_TARGET, side_effect=_mock_filter_dtb3_values):
        return run_pool_simulation(dto)


def _post_via_flask(client, json_body):
    """POST to the Flask endpoint and return parsed response."""
    with patch(_PARQUET_PATCH_TARGET, side_effect=_get_test_parquet_data), \
         patch(_DTB3_PATCH_TARGET, side_effect=_mock_filter_dtb3_values):
        resp = client.post(
            "/api/runSimulation",
            data=json.dumps(json_body),
            content_type="application/json",
        )
    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.data[:500]}"
    )
    # The endpoint double-encodes: json.dumps(jsonpickle.encode(...))
    outer = json.loads(resp.data)
    return json.loads(outer)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


START = "2023-01-01 00:00:00"
END = "2023-05-31 23:59:00"


# ===================================================================
# PART 1 — DTO-level tests (SimulationRunDto → run_pool_simulation)
# ===================================================================

class TestDtoMomentumPool:
    """Exercise the DTO parsing + run_pool_simulation for momentum."""

    def test_result_has_timesteps_and_analysis(self):
        result = _run_via_dto(_make_momentum_json(["BTC", "ETH"], START, END))
        assert "resultTimeSteps" in result
        assert "analysis" in result
        assert len(result["resultTimeSteps"]) > 0

    def test_analysis_final_weights(self):
        result = _run_via_dto(_make_momentum_json(["BTC", "ETH"], START, END))
        analysis = result["analysis"]

        assert "final_weights" in analysis
        assert "final_weights_strings" in analysis
        assert len(analysis["final_weights"]) == 2
        assert len(analysis["final_weights_strings"]) == 2
        assert abs(sum(analysis["final_weights"]) - 1.0) < 0.01

    def test_analysis_jax_parameters(self):
        result = _run_via_dto(_make_momentum_json(["BTC", "ETH"], START, END))
        jax_params = result["analysis"]["jax_parameters"]
        assert "logit_lamb" in jax_params

    def test_analysis_smart_contract_parameters(self):
        result = _run_via_dto(_make_momentum_json(["BTC", "ETH"], START, END))
        sc = result["analysis"]["smart_contract_parameters"]
        assert "values" in sc
        assert "strings" in sc

    def test_3_asset_pool(self):
        result = _run_via_dto(
            _make_momentum_json(["BTC", "ETH", "SOL"], START, END)
        )
        assert len(result["analysis"]["final_weights"]) == 3
        assert len(result["analysis"]["final_weights_strings"]) == 3

    def test_final_weights_strings_are_bd18(self):
        result = _run_via_dto(_make_momentum_json(["BTC", "ETH"], START, END))
        for s in result["analysis"]["final_weights_strings"]:
            assert isinstance(s, str)
            assert s.isdigit() or s == "0"


class TestDtoBalancerPool:
    """DTO-level tests for Balancer — the pool type that crashed with 0-d arrays."""

    def test_does_not_crash(self):
        """The 0-d array TypeError must not occur."""
        result = _run_via_dto(_make_balancer_json(["BTC", "ETH"], START, END))
        assert "analysis" in result

    def test_final_weights_equal(self):
        result = _run_via_dto(_make_balancer_json(["BTC", "ETH"], START, END))
        analysis = result["analysis"]
        assert len(analysis["final_weights"]) == 2
        np.testing.assert_allclose(
            analysis["final_weights"], [0.5, 0.5], atol=0.01,
        )

    def test_final_weights_strings_valid(self):
        result = _run_via_dto(_make_balancer_json(["BTC", "ETH"], START, END))
        for s in result["analysis"]["final_weights_strings"]:
            assert isinstance(s, str)
            assert s.isdigit() or s == "0"

    def test_3_asset_balancer(self):
        result = _run_via_dto(
            _make_balancer_json(["BTC", "ETH", "SOL"], START, END)
        )
        assert len(result["analysis"]["final_weights"]) == 3
        np.testing.assert_allclose(
            result["analysis"]["final_weights"],
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            atol=0.01,
        )

    def test_has_timesteps(self):
        result = _run_via_dto(_make_balancer_json(["BTC", "ETH"], START, END))
        assert len(result["resultTimeSteps"]) > 0


# ===================================================================
# PART 2 — Flask HTTP tests (POST /api/runSimulation)
# ===================================================================

class TestFlaskMomentumPool:
    """Full HTTP round-trip for momentum pool."""

    def test_returns_200(self, client):
        body = _make_momentum_json(["BTC", "ETH"], START, END)
        result = _post_via_flask(client, body)
        assert "timeSteps" in result
        assert "analysis" in result

    def test_analysis_has_final_weights(self, client):
        result = _post_via_flask(
            client, _make_momentum_json(["BTC", "ETH"], START, END)
        )
        analysis = result["analysis"]
        assert len(analysis["final_weights"]) == 2
        assert len(analysis["final_weights_strings"]) == 2
        assert abs(sum(analysis["final_weights"]) - 1.0) < 0.01

    def test_analysis_has_financial_metrics(self, client):
        result = _post_via_flask(
            client, _make_momentum_json(["BTC", "ETH"], START, END)
        )
        analysis = result["analysis"]
        assert "return_analysis" in analysis
        metric_names = {m["metricName"] for m in analysis["return_analysis"]}
        for name in ["Sharpe Ratio", "Absolute Return (%)", "Daily Returns Maximum Drawdown"]:
            assert name in metric_names, f"Missing metric: {name}"

    def test_3_asset_pool(self, client):
        result = _post_via_flask(
            client, _make_momentum_json(["BTC", "ETH", "SOL"], START, END)
        )
        assert len(result["analysis"]["final_weights"]) == 3


class TestFlaskBalancerPool:
    """Full HTTP round-trip for Balancer — the 0-d array crash scenario."""

    def test_returns_200(self, client):
        """Balancer pool must not 500."""
        result = _post_via_flask(
            client, _make_balancer_json(["BTC", "ETH"], START, END)
        )
        assert "analysis" in result

    def test_final_weights_equal(self, client):
        result = _post_via_flask(
            client, _make_balancer_json(["BTC", "ETH"], START, END)
        )
        np.testing.assert_allclose(
            result["analysis"]["final_weights"], [0.5, 0.5], atol=0.01,
        )

    def test_final_weights_strings_valid(self, client):
        result = _post_via_flask(
            client, _make_balancer_json(["BTC", "ETH"], START, END)
        )
        for s in result["analysis"]["final_weights_strings"]:
            assert isinstance(s, str)
            assert s.isdigit() or s == "0"

    def test_3_asset_balancer(self, client):
        result = _post_via_flask(
            client, _make_balancer_json(["BTC", "ETH", "SOL"], START, END)
        )
        assert len(result["analysis"]["final_weights"]) == 3
