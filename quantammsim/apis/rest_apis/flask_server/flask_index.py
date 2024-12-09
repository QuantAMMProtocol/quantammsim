import time
from datetime import timedelta

from flask import Flask, jsonify, request, send_from_directory, redirect
import json
import jsonpickle
from json import JSONEncoder
from functools import wraps

from flask_jwt_extended import create_access_token, verify_jwt_in_request
from flask_jwt_extended import jwt_required
from flask_jwt_extended import JWTManager
from flask_jwt_extended import set_access_cookies
from flask_jwt_extended import unset_jwt_cookies

# from training_result import TrainingResult
from flask_cors import CORS
from quantammsim.runners.jax_runners import train_on_historic_data
from quantammsim.apis.rest_apis.simulator_dtos.simulation_run_dto import (
    FinancialAnalysisResult,
    LoadPriceHistoryRequestDto,
    SimulationRunDto,
    SimulationResult,
    TrainingDto,
    FinancialAnalysisRequestDto,
)

from quantammsim.core_simulator.param_utils import dict_of_jnp_to_np, NumpyEncoder


from quantammsim.simulator_analysis_tools.finance.param_financial_calculator import (
    run_bencharks_and_financial_analysis,
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

jwt = JWTManager(app)


def redirect_if_jwt_invalid():
    """
    Redirects to the login page if the JWT is invalid.

    This function is a decorator that wraps around other functions to ensure that
    the JWT (JSON Web Token) is valid. If the JWT is invalid, it redirects the user
    to the login page.

    Returns
    -------
    function
        A decorator function that wraps the input function and verifies the JWT.
    """
    def wrapper(fn):
        @wraps(fn)
        def decorator(*args, **kwargs):
            try:
                verify_jwt_in_request()
            except:
                return redirect("login", code=302)

            return fn(*args, **kwargs)

        return decorator

    return wrapper


@app.route("/runTraining", methods=["POST"])
def runTraining():
    """
    Endpoint to initiate the training process.

    This endpoint receives a POST request with training parameters in JSON format,
    processes the parameters, and starts the training process on historic data.

    Returns
    -------
    str
        A success message indicating that the training process has started.
    """
    request_data = request.get_json()
    dto = TrainingDto(request_data)
    run_fingerprint = dto.convert_to_run_fingerprint()
    null_keys = list()
    for key in run_fingerprint:
        if run_fingerprint[key] is None:
            null_keys.append(key)

    for key in null_keys:
        del run_fingerprint[key]

    dumped = json.dumps(run_fingerprint, cls=NumpyEncoder, sort_keys=True)
    with open("../../../../experiments/jax_training_runs.json", "w") as json_file:
        json.dump(dumped, json_file)
    train_on_historic_data(
        dto.convert_to_run_fingerprint(),
        "../../../../quantammsim/data/",
        iterations_per_print=1000,
    )
    return "success"


def load_from_file(file_path):
    """
    Load training results from a JSON file.

    This function reads a JSON file containing training results, parses the content,
    and returns a list of TrainingResult objects.

    Parameters
    ----------
    file_path : str
        The path to the JSON file containing the training results.

    Returns
    -------
    list of TrainingResult
        A list of TrainingResult objects parsed from the JSON file.
    """
    with open(file_path, "r") as file:
        content = json.load(file)
        return [TrainingResult(**item) for item in content]


@app.route("/getTrainingResult", methods=["POST"])
def getTrainingResult(request):
    """
    Handle the POST request to get training results.

    This function reads the request data, loads the training results from a JSON file,
    and prints the results.

    Parameters
    ----------
    request : flask.Request
        The request object containing the POST data.

    Returns
    -------
    None
    """
    request_data = request.get_json()
    dto = load_from_file("../../../../experiments/results/run_test.json")

    resultJSON = jsonpickle.encode(dto, unpicklable=False)
    jsonString = json.dumps(resultJSON, indent=4)

    return jsonString


@app.route("/runSimulation", methods=["POST"])
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

    resultJSON = jsonpickle.encode(SimulationResult(result), unpicklable=False)
    jsonString = json.dumps(resultJSON, indent=4)

    return jsonString


@app.route("/runFinancialAnalysis", methods=["POST"])
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

    portfolio_returns, benchmark_returns, filled_days_count, total_days_count = (
        process_return_array(dto.returns, dto.benchmarks)
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


@app.route("/loadHistoricDailyPrices", methods=["POST"])
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
    parsed = json.loads(result)
    jsonString = json.dumps(parsed)
    return jsonString

@app.route("/loadCoinComparisonData", methods=["POST"])
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


@app.route("/products", methods=["GET"])
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

    with open(file_path, "r") as file:
        content = json.load(file)

    return content


@app.route("/filters", methods=["GET"])
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

    with open(file_path, "r") as file:
        content = json.load(file)

    return content


if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5001")
