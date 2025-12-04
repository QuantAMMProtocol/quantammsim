import time
from datetime import timedelta, datetime

from flask import Flask, request
import json
import jsonpickle
from flask_cors import CORS
import pandas as pd
import os
import hashlib
import hmac
import ipaddress
from datetime import timezone as tz

from quantammsim.apis.rest_apis.simulator_dtos.simulation_run_dto import (
    FinancialAnalysisResult,
    LoadPriceHistoryRequestDto,
    SimulationRunDto,
    SimulationResult,
    FinancialAnalysisRequestDto,
)

from quantammsim.simulator_analysis_tools.finance.param_financial_calculator import (
    run_pool_simulation,
    run_financial_analysis,
    process_return_array,
)
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_daily_csv_data,
    get_coin_comparison_data,
)

app = Flask(__name__, static_url_path="", static_folder="frontend/quantAMMapp/build")

CORS(app)

app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["JWT_COOKIE_SECURE"] = False
app.config["JWT_ACCESS_CSRF_HEADER_NAME"] = "ROBODEX-X-CSRF-TOKEN"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=3)
app.config["JWT_SECRET_KEY"] = (
    "2b25014d8e591e91cc4e3bfc3a7561983e06bc7ff0a140bcecca3c0a15d31c5e"
)


@app.route("/api/runSimulation", methods=["POST"])
def runSimulation():
    """
    Handle the POST request to run a simulation.

    This function reads the request data, initializes a SimulationRunDto object,
    and runs the pool simulation. The results are then encoded into JSON format
    and returned as a response.

    Returns
    -------
    str
        A JSON string containing the simulation results.
    """
    request_data = request.get_json()

    dto = SimulationRunDto(request_data)
    result = run_pool_simulation(dto)

    print(result["analysis"])
    
    resultJSON = jsonpickle.encode(SimulationResult(result), unpicklable=False)
    jsonString = json.dumps(resultJSON, indent=4)

    return jsonString


PEPPER = os.environ.get(
    "IP_HASH_PEPPER", app.config["JWT_SECRET_KEY"]
).encode()  # 32 random bytes


def canonical_ip(raw_ip: str) -> str:
    """First hop, trimmed, canonical textual representation."""
    first = raw_ip.split(",")[0].strip()  # X-Forwarded-For support
    # strip :port on IPv4
    if first.count(":") == 1 and "." in first:
        first = first.split(":")[0]
    return str(ipaddress.ip_address(first))


def hash_ip(ip: str) -> str:
    """Deterministic, keyed, one-way hash of an IP address."""
    return hmac.new(PEPPER, ip.encode(), hashlib.sha256).hexdigest()


@app.route("/api/runAuditLog", methods=["POST"])
def runAuditLog():
    """
    Handle the POST request to log audit information.

    This function retrieves a parquet file labeled with today's Unix timestamp,
    updates the log with the provided audit information, and saves the updated file.

    Returns
    -------
    str
        A success message.
    """

    # Retrieve the request data
    request_data = request.get_json()

    requester_ip = canonical_ip(
        request.headers.get("X-Forwarded-For", request.remote_addr)
    )
    hashed_ip = hash_ip(requester_ip)

    timezone = (
        request_data["timestamp"].split(",")[-1].strip()
        if "," in request_data.get("timestamp", "")
        else "Unknown"
    )

    audit_info = {
        "timestamp": int(
            datetime.now(tz.utc).replace(second=0, microsecond=0).timestamp()
        ),
        "user": request_data["user"],  # FingerprintJS visitorId from front-end
        "page": request_data["page"],
        "tosAgreement": request_data["tosAgreement"],
        "isMobile": request_data["isMobile"],
        "timezone": timezone,
        "flask_user": hashed_ip,  # pseudonymised IP
    }

    today_unix_timestamp = int(
        datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    )
    file_name = f"{today_unix_timestamp}.parquet"
    file_path = os.path.join("./audit_logs", file_name)

    if os.path.exists(file_path):
        df = pd.read_parquet(file_path, engine="pyarrow")
    else:
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "user",
                "page",
                "tosAgreement",
                "isMobile",
                "timezone",
                "flask_user",
                "count",
            ]
        )

    row_filter = (
        (df["timestamp"] == audit_info["timestamp"])
        & (df["user"] == audit_info["user"])
        & (df["page"] == audit_info["page"])
        & (df["tosAgreement"] == audit_info["tosAgreement"])
        & (df["isMobile"] == audit_info["isMobile"])
        & (df["timezone"] == audit_info["timezone"])
        & (df["flask_user"] == audit_info["flask_user"])
    )

    if df[row_filter].empty:
        audit_info["count"] = 1
        df = pd.concat([df, pd.DataFrame([audit_info])], ignore_index=True)
    else:
        df.loc[row_filter, "count"] += 1

    os.makedirs("./audit_logs", exist_ok=True)
    df.to_parquet(file_path, engine="pyarrow")

    return json.dumps({"message": "Audit log updated successfully."})


@app.route("/api/runFinancialAnalysis", methods=["POST"])
def runFinancialAnalysis():
    """
    Handle the POST request to run a financial analysis.

    This function reads the request data, initializes a FinancialAnalysisRequestDto object,
    processes the return arrays, and runs the financial analysis. The results are then encoded
    into JSON format and returned as a response.

    Returns
    -------
    str
        A JSON string containing the financial analysis results.
    """
    request_data = request.get_json()
    dto = FinancialAnalysisRequestDto(request_data)

    portfolio_returns, benchmark_returns, _, _ = process_return_array(
        dto.returns, dto.benchmarks
    )

    start_timestamp = dto.returns[0][0]
    end_timestamp = dto.returns[-1][0]
    start_date = time.strftime(
        "%Y-%m-%d  %H:%M:%S", time.gmtime(start_timestamp / 1000)
    )
    end_date = time.strftime("%Y-%m-%d  %H:%M:%S", time.gmtime(end_timestamp / 1000))

    result = run_financial_analysis(
        portfolio_daily_returns=portfolio_returns,
        startDateString=start_date,
        endDateString=end_date,
        bechmark_names=dto.benchmarks,
        benchmarks_returns=benchmark_returns,
    )

    resultJSON = jsonpickle.encode(FinancialAnalysisResult(result), unpicklable=False)
    jsonString = json.dumps(resultJSON, indent=4)

    return jsonString


@app.route("/api/loadHistoricDailyPrices", methods=["POST"])
def loadHistoricDailyPrices():
    """
    Handle the POST request to load historic daily prices.

    This function reads the request data, initializes a LoadPriceHistoryRequestDto object,
    retrieves historic daily price data from CSV files, converts the data to JSON format,
    and returns the JSON string as a response.

    Returns
    -------
    str
        A JSON string containing the historic daily price data.
    """
    request_data = request.get_json()
    dto = LoadPriceHistoryRequestDto(request_data)
    root = "../../../../quantammsim/data/"
    historic = get_historic_daily_csv_data([dto.coinCode], root)
    result = historic.to_json(orient="records")  # Is this the right way to do this?
    parsed = json.loads(result)
    jsonString = json.dumps(parsed)
    return jsonString


@app.route("/api/loadCoinComparisonData", methods=["POST"])
def loadCoinComparisonData():
    """
    Handle the POST request to load coin comparison data.

    This function retrieves coin comparison data from CSV files, converts the data to JSON format,
    and returns the JSON string as a response.

    Returns
    -------
    str
        A JSON string containing the coin comparison data.
    """
    root = "../../../../quantammsim/data/"
    historic = get_coin_comparison_data(root)
    result = historic.to_json(orient="records")
    parsed = json.loads(result)
    jsonString = json.dumps(parsed)
    ##return result
    return jsonString


@app.route("/api/products", methods=["GET"])
def products():
    """
    Handle the GET request to retrieve product information.

    This function reads product data from a JSON file and returns it as a response.

    Returns
    -------
    dict
        A dictionary containing the product information.
    """
    file_path = "./stubs/productStubs.json"

    with open(file_path, "r", encoding="utf-8") as file:
        content = json.load(file)

    return content


@app.route("/api/filters", methods=["GET"])
def filters():
    """
    Handle the GET request to retrieve filter information.

    This function reads filter data from a JSON file and returns it as a response.

    Returns
    -------
    dict
        A dictionary containing the filter information.
    """
    file_path = "./stubs/filterStubs.json"

    with open(file_path, "r", encoding="utf-8") as file:
        content = json.load(file)

    return content


@app.route("/api/test", methods=["GET"])
def test():
    return "Hello World"


@app.route("/health", methods=["GET"])
def health():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5001")
