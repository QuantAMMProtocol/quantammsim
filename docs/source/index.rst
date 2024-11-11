Welcome to quantammsim's Documentation
======================================

quantammsim is a Python library for modeling synthetic markets, enabling modelling of Balancer, CowAMM and QuantAMM protocol pools. It provides tools for:

* Automated Market Making (AMM) simulation
* Arbitrage opportunity detection
* Historical data backtesting
* Simulation of trading strategies
* Tuning of pool parameters/strategies

Quick Start
-----------

Install quantammsim:

.. code-block:: bash

   pip install quantammsim

Basic usage:

.. code-block:: python

   from quantammsim.runners.jax_runners import do_run_on_historic_data

   # Define experiment parameters
   run_fingerprint = {
       'tokens': ['BTC', 'DAI'],
       'rule': 'momentum',
       'initial_pool_value': 1000000.0
   }

   # Run simulation
   result = do_run_on_historic_data(run_fingerprint)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   tutorials/index
   user_guide/index
   api/index

.. include:: introduction.rst

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`