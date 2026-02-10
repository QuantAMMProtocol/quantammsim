Data Sources
============

quantammsim supports loading price data from multiple exchanges and data providers,
with automatic gap-filling across sources to produce complete minute-level time series.
This guide covers the data pipeline, supported sources, format requirements, and how
to supply your own price data.

Data Pipeline Overview
----------------------

All data loading flows through
:func:`~quantammsim.utils.data_processing.historic_data_utils.get_data_dict`, called
by both :func:`~quantammsim.runners.jax_runners.train_on_historic_data` and
:func:`~quantammsim.runners.jax_runners.do_run_on_historic_data`.

.. code-block:: python

    from quantammsim.utils.data_processing.historic_data_utils import get_data_dict

    data_dict = get_data_dict(
        list_of_tickers=["BTC", "ETH", "USDC"],
        run_fingerprint=run_fingerprint,
        data_kind="historic",
        root="/path/to/data/",
        max_memory_days=365.0,
        start_date_string="2024-01-01 00:00:00",
        end_time_string="2024-06-01 00:00:00",
    )

The returned dictionary contains:

* ``prices`` -- Minute-level close prices, numpy array of shape ``(T, n_assets)``
* ``unix_values`` -- Millisecond unix timestamps, shape ``(T,)``
* ``start_idx`` / ``end_idx`` -- Indices bounding the simulation period
* ``bout_length`` -- Timesteps in the simulation period (``end_idx - start_idx``)
* ``max_memory_days`` -- Burn-in lookback before ``start_idx`` (clamped if data is short)
* ``n_chunks`` -- Number of ``chunk_period``-sized blocks in the price array

When a test period is specified, the dictionary also includes ``prices_test``,
``start_idx_test``, ``end_idx_test``, ``bout_length_test``, and ``unix_values_test``.

The ``data_kind`` parameter selects the loading strategy:

* ``"historic"`` -- Load from parquet via
  :func:`~quantammsim.utils.data_processing.historic_data_utils.get_historic_parquet_data`
  (default).
* ``"mc"`` -- Monte Carlo price path versions (BTC and ETH only).
* ``"step"`` -- Step-function price pattern for debugging strategy responses.

Supported Data Sources
----------------------

Historic Parquet (primary)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-asset parquet files (e.g. ``BTC_USD.parquet``) loaded and joined on their
``unix`` index. When ``root=None``, data is loaded from the bundled
``quantammsim/data/`` directory.
:func:`~quantammsim.utils.data_processing.historic_data_utils.update_historic_data`
builds these files by amalgamating all downstream sources into a gap-free
minute-level series per token.

Binance
~~~~~~~

:mod:`quantammsim.utils.data_processing.binance_data_utils` --
Handles yearly CSVs from CryptoDataDownload;
:func:`~quantammsim.utils.data_processing.binance_data_utils.concat_csv_files`
joins them into a single DataFrame.
:func:`~quantammsim.utils.data_processing.historic_data_utils.get_binance_vision_data`
downloads directly from ``binance.vision`` via the ``binance_historical_data`` package.

Coinbase
~~~~~~~~

:mod:`quantammsim.utils.data_processing.coinbase_data_utils` --
Uses the ``Historic_Crypto`` package.
:func:`~quantammsim.utils.data_processing.coinbase_data_utils.fill_missing_rows_with_coinbase_data`
fills gaps from pre-downloaded Coinbase Pro CSVs.

CoinMarketCap
~~~~~~~~~~~~~

:mod:`quantammsim.utils.data_processing.cmc_data_utils` --
3-hour interval data.
:func:`~quantammsim.utils.data_processing.cmc_data_utils.fill_missing_rows_with_cmc_historical_data`
fills gaps in the primary series.

Crypto Historical Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`quantammsim.utils.data_processing.amalgamated_data_utils` --
1-minute text files (``{TOKEN}_full_1min.txt``).
:func:`~quantammsim.utils.data_processing.amalgamated_data_utils.forward_fill_ohlcv_data`
creates a complete minute-level index, forward-filling close prices, setting
OHLC to previous close, and volume to zero for missing rows.

st0x
~~~~

:mod:`quantammsim.utils.data_processing.st0x_data_utils` --
Non-crypto assets (e.g. TSLA, JNJ).
:func:`~quantammsim.utils.data_processing.st0x_data_utils.fill_missing_rows_with_st0x_historical_data`
fills gaps with st0x data.

Aerodrome DEX
~~~~~~~~~~~~~

:mod:`quantammsim.utils.data_processing.aerodrome_data_utils` --
On-chain data from Aerodrome on Base. Last source in the gap-filling cascade;
useful for tokens with limited centralised exchange coverage.

Treasury Bill Rates
~~~~~~~~~~~~~~~~~~~

:mod:`quantammsim.utils.data_processing.dtb3_data_utils` --
3-month T-bill rates from FRED, used as a risk-free rate benchmark. Not part of
the gap-filling cascade. Returns daily rates as decimals (percentage / 100),
forward- then back-filled for missing dates.

.. code-block:: python

    from quantammsim.utils.data_processing.dtb3_data_utils import filter_dtb3_values
    rates = filter_dtb3_values("DTB3.csv", "2024-01-01", "2024-06-01")

Synthetic Data
~~~~~~~~~~~~~~

:mod:`quantammsim.utils.data_processing.synthetic_data_utils` --
Deterministic sinusoidal prices for testing, with no external data dependency.

.. code-block:: python

    from quantammsim.utils.data_processing.synthetic_data_utils import make_sinuisoid_data
    prices = make_sinuisoid_data(n_time_steps=2880, n_tokens=3, n_periods=3, noise=True)

The ``composite_run`` flag interleaves fast and slow cycles for multi-frequency tests.

Data Format Requirements
------------------------

All sources are normalised to a common DataFrame format before writing to parquet:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``unix``
     - ``int64``
     - Millisecond Unix timestamp (index or column)
   * - ``date``
     - ``str``
     - ``"YYYY-MM-DD HH:MM:SS"``
   * - ``symbol``
     - ``str``
     - Trading pair, e.g. ``"BTC/USD"``
   * - ``open`` / ``high`` / ``low`` / ``close``
     - ``float64``
     - OHLC minute candle prices
   * - ``Volume USD``
     - ``float64``
     - Dollar volume
   * - ``Volume {TOKEN}``
     - ``float64``
     - Token-denominated volume

Key constraints:

* **1-minute resolution** required (60000 ms between consecutive rows).
  ``update_historic_data`` validates this and raises on any remaining gap.
* Timestamps use **millisecond precision**. Source data in seconds is multiplied
  by 1000; nanosecond data is divided by 10\ :sup:`6`.
* Only ``close`` is consumed by the default simulation path (``cols=["close"]``).
  Full OHLCV is needed only with ``return_slippage=True``.

Using Custom Price Data
-----------------------

Bypass all file loading by passing ``price_data`` directly:

.. code-block:: python

    import pandas as pd, numpy as np

    unix_ms = np.arange(start_ms, end_ms, 60_000, dtype=np.int64)
    df = pd.DataFrame(
        {"close_BTC": btc_prices, "close_ETH": eth_prices, "close_USDC": usdc_prices},
        index=pd.Index(unix_ms, name="unix"),
    )

    data_dict = get_data_dict(
        list_of_tickers=["BTC", "ETH", "USDC"],
        run_fingerprint=run_fingerprint,
        price_data=df,
        start_date_string="2024-01-01 00:00:00",
        end_time_string="2024-06-01 00:00:00",
    )

The index must be named ``"unix"`` with millisecond timestamps. Columns must
follow the ``close_{TICKER}`` convention. Tickers are sorted alphabetically
internally, so column order is irrelevant.

Gap Filling and Amalgamation
----------------------------

:func:`~quantammsim.utils.data_processing.historic_data_utils.update_historic_data`
tries each source in sequence, filling only timestamps still missing:

1. **Binance Vision** -- primary, from ``binance.vision``
2. **Binance CDD** -- CryptoDataDownload yearly CSVs
3. **Coinbase** -- pre-downloaded Coinbase Pro CSVs
4. **Gemini** -- Gemini exchange CSVs
5. **Bitstamp** -- Bitstamp exchange CSVs
6. **Crypto Historical Dataset** -- 1-minute text files
7. **CoinMarketCap** -- 3-hour data (selected tokens)
8. **st0x** -- non-crypto assets (selected tokens)
9. **Candles parquet** -- DeFi tokens via Trading Strategy candles
10. **Aerodrome DEX** -- on-chain data from Base

At each step the fill function computes the index set difference, concatenates
missing rows, sorts, and deduplicates. After all sources,
:func:`~quantammsim.utils.data_processing.amalgamated_data_utils.forward_fill_ohlcv_data`
produces a gapless series. The pipeline validates that all consecutive timestamp
differences are exactly 60000 ms.

Frequency Conversion
--------------------

:mod:`quantammsim.utils.data_processing.minute_daily_conversion_utils` provides:

* :func:`~quantammsim.utils.data_processing.minute_daily_conversion_utils.expand_daily_to_minute_data`
  -- reindex daily data to minute frequency via forward-fill.
* :func:`~quantammsim.utils.data_processing.minute_daily_conversion_utils.resample_minute_level_OHLC_data_to_daily`
  -- aggregate minute OHLC into daily candles (first open, max high, min low,
  last close, summed volume).
* :func:`~quantammsim.utils.data_processing.minute_daily_conversion_utils.calculate_annualised_daily_volatility_from_minute_data`
  -- daily log-return std from minute prices, annualised by :math:`\sqrt{365.25}`.

:mod:`quantammsim.utils.data_processing.volume_data_utils` provides
:func:`~quantammsim.utils.data_processing.volume_data_utils.calculate_daily_volume_from_minute_data`,
computing daily token volume as summed dollar volume / daily mean close.

See Also
--------

* :doc:`run_fingerprints` -- ``startDateString``, ``endDateString``, and data-related run fingerprint settings
* :doc:`core_concepts` -- How price data feeds into weight update rules
* :doc:`../tutorials/training_pipeline` -- Training pipeline walkthrough
* :mod:`quantammsim.utils.data_processing.datetime_utils` -- Timestamp conversion helpers
* :mod:`quantammsim.utils.data_processing.price_data_fingerprint_utils` -- Comparing/loading run fingerprints to avoid redundant data loading
