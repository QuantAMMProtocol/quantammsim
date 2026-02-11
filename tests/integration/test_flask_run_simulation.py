"""
Full e2e tests for the /api/runSimulation code path.

Two levels of coverage:
  1. DTO -> run_pool_simulation  (tests the DTO parsing + full simulation)
  2. HTTP POST -> Flask endpoint (tests the complete HTTP round-trip)

Both levels patch ``get_historic_parquet_data`` to use test-data parquet
files and ``filter_dtb3_values`` to avoid needing the untracked DTB3.csv.

Note on end time
----------------
END uses 23:59 rather than midnight.  This matches the frontend behaviour
and sidesteps an off-by-one in ``create_daily_unix_array`` /
``filter_dtb3_values`` which both use inclusive-endpoint date ranges while
the simulation output has one fewer daily row.

Note on response keys
---------------------
``run_pool_simulation`` returns ``{"resultTimeSteps": ..., "analysis": ...}``.
``SimulationResult.__init__`` renames the key: ``self.timeSteps = result["resultTimeSteps"]``.
After jsonpickle serialisation the Flask response therefore contains
``"timeSteps"`` (not ``"resultTimeSteps"``).  The DTO-level tests assert on
``"resultTimeSteps"``; the Flask-level tests assert on ``"timeSteps"``.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd

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

START = "2023-01-01 00:00:00"
END = "2023-05-31 23:59:00"


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------

def _get_test_parquet_data(list_of_tickers, cols=None, root=None, **kwargs):
    """Redirect get_historic_parquet_data to test data directory."""
    if cols is None:
        cols = ["close"]
    return get_historic_parquet_data(
        list_of_tickers, cols=cols, root=str(TEST_DATA_DIR), **kwargs
    )


def _mock_filter_dtb3_values(filename, start_date, end_date, **kwargs):
    """Return a constant risk-free rate array for the date range.

    DTB3.csv is not tracked in the repo so we mock it with a flat 5%
    annual rate.  Must replicate the real ``filter_dtb3_values`` date
    semantics: truncate to midnight, inclusive of both endpoints.
    """
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    n_days = (end - start).days + 1
    return np.full(n_days, 0.05)


def _patches():
    """Context manager stacking both patches needed by run_pool_simulation."""
    return (
        patch(_PARQUET_PATCH_TARGET, side_effect=_get_test_parquet_data),
        patch(_DTB3_PATCH_TARGET, side_effect=_mock_filter_dtb3_values),
    )


# ---------------------------------------------------------------------------
# JSON request body builders
# ---------------------------------------------------------------------------

def _make_static_pool_json(rule, tokens, start, end):
    """Build a JSON dict for a static (constant-weight) pool request.

    Works for both ``"balancer"`` and ``"cow"``.
    """
    n = len(tokens)
    equal_value = 1_000_000.0 / n
    return {
        "startUnix": 0,
        "endUnix": 0,
        "startDateString": start,
        "endDateString": end,
        "pool": {
            "id": f"test-{rule}",
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
                "name": rule,
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


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _run_via_dto(json_body):
    """Parse JSON into a SimulationRunDto and call run_pool_simulation."""
    dto = SimulationRunDto(json_body)
    p1, p2 = _patches()
    with p1, p2:
        return run_pool_simulation(dto)


def _post_via_flask(client, json_body):
    """POST to the Flask endpoint and return parsed response."""
    p1, p2 = _patches()
    with p1, p2:
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
# Module-scoped fixtures — each simulation runs once, shared across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flask_client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# DTO-level results
@pytest.fixture(scope="module")
def dto_momentum_2():
    return _run_via_dto(_make_momentum_json(["BTC", "ETH"], START, END))


@pytest.fixture(scope="module")
def dto_momentum_3():
    return _run_via_dto(_make_momentum_json(["BTC", "ETH", "SOL"], START, END))


@pytest.fixture(scope="module")
def dto_balancer_2():
    return _run_via_dto(_make_static_pool_json("balancer", ["BTC", "ETH"], START, END))


@pytest.fixture(scope="module")
def dto_balancer_3():
    return _run_via_dto(
        _make_static_pool_json("balancer", ["BTC", "ETH", "SOL"], START, END)
    )


@pytest.fixture(scope="module")
def dto_cow_2():
    return _run_via_dto(_make_static_pool_json("cow", ["BTC", "ETH"], START, END))


# Flask-level results
@pytest.fixture(scope="module")
def flask_momentum_2(flask_client):
    return _post_via_flask(
        flask_client, _make_momentum_json(["BTC", "ETH"], START, END)
    )


@pytest.fixture(scope="module")
def flask_momentum_3(flask_client):
    return _post_via_flask(
        flask_client, _make_momentum_json(["BTC", "ETH", "SOL"], START, END)
    )


@pytest.fixture(scope="module")
def flask_balancer_2(flask_client):
    return _post_via_flask(
        flask_client,
        _make_static_pool_json("balancer", ["BTC", "ETH"], START, END),
    )


@pytest.fixture(scope="module")
def flask_balancer_3(flask_client):
    return _post_via_flask(
        flask_client,
        _make_static_pool_json("balancer", ["BTC", "ETH", "SOL"], START, END),
    )


@pytest.fixture(scope="module")
def flask_cow_2(flask_client):
    return _post_via_flask(
        flask_client,
        _make_static_pool_json("cow", ["BTC", "ETH"], START, END),
    )


# ===================================================================
# PART 1 — DTO-level tests (SimulationRunDto -> run_pool_simulation)
# ===================================================================

class TestDtoMomentumPool:
    """Exercise the DTO parsing + run_pool_simulation for momentum."""

    def test_result_has_timesteps_and_analysis(self, dto_momentum_2):
        assert "resultTimeSteps" in dto_momentum_2
        assert "analysis" in dto_momentum_2
        assert len(dto_momentum_2["resultTimeSteps"]) > 0

    def test_analysis_final_weights(self, dto_momentum_2):
        analysis = dto_momentum_2["analysis"]
        assert len(analysis["final_weights"]) == 2
        assert len(analysis["final_weights_strings"]) == 2
        assert abs(sum(analysis["final_weights"]) - 1.0) < 0.01

    def test_analysis_jax_parameters(self, dto_momentum_2):
        assert "logit_lamb" in dto_momentum_2["analysis"]["jax_parameters"]

    def test_analysis_smart_contract_parameters(self, dto_momentum_2):
        sc = dto_momentum_2["analysis"]["smart_contract_parameters"]
        assert "values" in sc
        assert "strings" in sc

    def test_3_asset_pool(self, dto_momentum_3):
        assert len(dto_momentum_3["analysis"]["final_weights"]) == 3
        assert len(dto_momentum_3["analysis"]["final_weights_strings"]) == 3

    def test_final_weights_strings_are_bd18(self, dto_momentum_2):
        for s in dto_momentum_2["analysis"]["final_weights_strings"]:
            assert isinstance(s, str)
            assert s.isdigit() or s == "0"


class TestDtoStaticPools:
    """DTO-level tests for static pools (Balancer, CowPool).

    These are the pool types that return 1-D weights from
    calculate_weights, which previously caused a 0-d array crash
    when run_pool_simulation indexed weights[-1].
    """

    def test_balancer_does_not_crash(self, dto_balancer_2):
        assert "analysis" in dto_balancer_2

    def test_balancer_final_weights_equal(self, dto_balancer_2):
        analysis = dto_balancer_2["analysis"]
        assert len(analysis["final_weights"]) == 2
        np.testing.assert_allclose(
            analysis["final_weights"], [0.5, 0.5], atol=0.01,
        )

    def test_balancer_final_weights_strings_valid(self, dto_balancer_2):
        for s in dto_balancer_2["analysis"]["final_weights_strings"]:
            assert isinstance(s, str)
            assert s.isdigit() or s == "0"

    def test_balancer_3_asset(self, dto_balancer_3):
        assert len(dto_balancer_3["analysis"]["final_weights"]) == 3
        np.testing.assert_allclose(
            dto_balancer_3["analysis"]["final_weights"],
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            atol=0.01,
        )

    def test_balancer_has_timesteps(self, dto_balancer_2):
        assert len(dto_balancer_2["resultTimeSteps"]) > 0

    def test_cow_does_not_crash(self, dto_cow_2):
        assert "analysis" in dto_cow_2

    def test_cow_final_weights_equal(self, dto_cow_2):
        analysis = dto_cow_2["analysis"]
        assert len(analysis["final_weights"]) == 2
        np.testing.assert_allclose(
            analysis["final_weights"], [0.5, 0.5], atol=0.01,
        )

    def test_cow_final_weights_strings_valid(self, dto_cow_2):
        for s in dto_cow_2["analysis"]["final_weights_strings"]:
            assert isinstance(s, str)
            assert s.isdigit() or s == "0"

    def test_cow_has_timesteps(self, dto_cow_2):
        assert len(dto_cow_2["resultTimeSteps"]) > 0


# ===================================================================
# PART 2 — Flask HTTP tests (POST /api/runSimulation)
# ===================================================================

class TestFlaskMomentumPool:
    """Full HTTP round-trip for momentum pool."""

    def test_returns_200(self, flask_momentum_2):
        # SimulationResult renames "resultTimeSteps" -> "timeSteps"
        assert "timeSteps" in flask_momentum_2
        assert "analysis" in flask_momentum_2

    def test_analysis_has_final_weights(self, flask_momentum_2):
        analysis = flask_momentum_2["analysis"]
        assert len(analysis["final_weights"]) == 2
        assert len(analysis["final_weights_strings"]) == 2
        assert abs(sum(analysis["final_weights"]) - 1.0) < 0.01

    def test_analysis_has_financial_metrics(self, flask_momentum_2):
        analysis = flask_momentum_2["analysis"]
        assert "return_analysis" in analysis
        metric_names = {m["metricName"] for m in analysis["return_analysis"]}
        for name in [
            "Sharpe Ratio",
            "Absolute Return (%)",
            "Daily Returns Maximum Drawdown",
        ]:
            assert name in metric_names, f"Missing metric: {name}"

    def test_3_asset_pool(self, flask_momentum_3):
        assert len(flask_momentum_3["analysis"]["final_weights"]) == 3


class TestFlaskStaticPools:
    """Full HTTP round-trip for static pools (Balancer, CowPool)."""

    def test_balancer_returns_200(self, flask_balancer_2):
        assert "analysis" in flask_balancer_2

    def test_balancer_final_weights_equal(self, flask_balancer_2):
        np.testing.assert_allclose(
            flask_balancer_2["analysis"]["final_weights"],
            [0.5, 0.5],
            atol=0.01,
        )

    def test_balancer_final_weights_strings_valid(self, flask_balancer_2):
        for s in flask_balancer_2["analysis"]["final_weights_strings"]:
            assert isinstance(s, str)
            assert s.isdigit() or s == "0"

    def test_balancer_3_asset(self, flask_balancer_3):
        assert len(flask_balancer_3["analysis"]["final_weights"]) == 3

    def test_cow_returns_200(self, flask_cow_2):
        assert "analysis" in flask_cow_2

    def test_cow_final_weights_equal(self, flask_cow_2):
        np.testing.assert_allclose(
            flask_cow_2["analysis"]["final_weights"],
            [0.5, 0.5],
            atol=0.01,
        )

    def test_cow_final_weights_strings_valid(self, flask_cow_2):
        for s in flask_cow_2["analysis"]["final_weights_strings"]:
            assert isinstance(s, str)
            assert s.isdigit() or s == "0"
