REST API Reference
==================

quantammsim includes a Flask-based REST API for running simulations and analysis programmatically.

Starting the Server
-------------------

.. code-block:: bash

    cd quantammsim/apis/rest_apis/flask_server
    python flask_index.py

The server runs on ``http://localhost:5001`` by default.

Endpoints
---------

POST /api/runSimulation
~~~~~~~~~~~~~~~~~~~~~~~

Run a pool simulation with the specified configuration.

**Request:**

.. code-block:: json

    {
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-12-01 00:00:00",
        "tokens": ["BTC", "ETH"],
        "rule": "momentum",
        "initial_pool_value": 1000000,
        "fees": 0.003,
        "params": {
            "log_k": [3.0, 3.0],
            "logit_lamb": [0.0, 0.0],
            "initial_weights_logits": [0.0, 0.0]
        }
    }

**Response:**

Returns simulation results including:

- Time series of pool value
- Reserve history
- Weight history
- Performance metrics

**Example (curl):**

.. code-block:: bash

    curl -X POST http://localhost:5001/api/runSimulation \
        -H "Content-Type: application/json" \
        -d '{
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "initial_pool_value": 1000000
        }'

**Example (Python):**

.. code-block:: python

    import requests

    response = requests.post(
        "http://localhost:5001/api/runSimulation",
        json={
            "startDateString": "2023-01-01 00:00:00",
            "endDateString": "2023-06-01 00:00:00",
            "tokens": ["BTC", "ETH"],
            "rule": "momentum",
            "initial_pool_value": 1000000,
        }
    )
    result = response.json()

POST /api/runFinancialAnalysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform financial analysis on return data.

**Request:**

.. code-block:: text

    {
        "returns": [
            [1672531200000, 1000000],
            [1672617600000, 1005000],
            ...
        ],
        "benchmarks": ["BTC", "ETH"]
    }

The ``returns`` array contains ``[timestamp_ms, value]`` pairs.

**Response:**

Returns comprehensive financial metrics:

- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Drawdown statistics
- Capture ratios
- Distribution statistics

**Example:**

.. code-block:: python

    import requests

    response = requests.post(
        "http://localhost:5001/api/runFinancialAnalysis",
        json={
            "returns": returns_data,
            "benchmarks": ["BTC"]
        }
    )
    analysis = response.json()

POST /api/loadHistoricDailyPrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load historical daily price data for a token.

**Request:**

.. code-block:: json

    {
        "coinCode": "BTC"
    }

**Response:**

Returns daily price history as JSON array:

.. code-block:: text

    [
        {"date": "2023-01-01", "price": 16500.0},
        {"date": "2023-01-02", "price": 16600.0},
        ...
    ]

POST /api/loadCoinComparisonData
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load comparison data for all available coins.

**Request:** No body required.

**Response:**

Returns comparison metrics for available tokens.

GET /api/products
~~~~~~~~~~~~~~~~~

Get available product configurations.

**Response:**

Returns list of pre-configured pool products.

GET /api/filters
~~~~~~~~~~~~~~~~

Get available filter options for the UI.

**Response:**

Returns filter configuration data.

GET /api/test
~~~~~~~~~~~~~

Health check endpoint.

**Response:** ``"Hello World"``

GET /health
~~~~~~~~~~~

Server health check.

**Response:** ``"OK"``

POST /api/runAuditLog
~~~~~~~~~~~~~~~~~~~~~

Log audit information (for frontend usage tracking).

**Request:**

.. code-block:: json

    {
        "user": "visitor_id",
        "page": "simulator",
        "timestamp": "2023-01-01 12:00:00, UTC",
        "tosAgreement": true,
        "isMobile": false
    }

**Response:**

.. code-block:: json

    {"message": "Audit log updated successfully."}

Error Handling
--------------

The API returns standard HTTP status codes:

- ``200`` - Success
- ``400`` - Bad request (invalid parameters)
- ``500`` - Internal server error

Errors include a JSON body with details:

.. code-block:: json

    {
        "error": "Error message description"
    }

CORS
----

Cross-Origin Resource Sharing (CORS) is enabled for all origins. For production deployments, configure appropriate CORS restrictions.

Configuration
-------------

Server configuration is in ``flask_index.py``:

.. code-block:: python

    app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=3)

The server listens on all interfaces (``0.0.0.0``) port ``5001`` by default.

Data Transfer Objects
---------------------

The API uses DTOs defined in ``quantammsim/apis/rest_apis/simulator_dtos/``:

- ``SimulationRunDto`` - Input for simulation
- ``SimulationResult`` - Output from simulation
- ``FinancialAnalysisRequestDto`` - Input for analysis
- ``FinancialAnalysisResult`` - Output from analysis
- ``LoadPriceHistoryRequestDto`` - Input for price loading
