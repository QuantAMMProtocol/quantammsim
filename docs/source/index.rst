Welcome to quantammsim's Documentation
======================================

quantammsim is a Python library for modeling synthetic markets, enabling modelling of Balancer, CowAMM and QuantAMM protocol pools. It provides tools for:

* Automated Market Making (AMM) simulation
* Arbitrage opportunity detection
* Historical data backtesting
* Simulation of trading strategies
* Tuning of pool parameters/strategies

See our :doc:`installation guide <installation>` to get started.

Quick Start
-----------

Once installed, here's a basic usage example:

.. code-block:: python

   from quantammsim.runners.jax_runners import do_run_on_historic_data
   import jax.numpy as jnp
   # Define simulation parameters
   run_fingerprint = {
       'tokens': ['BTC', 'USDC'],
       'rule': 'balancer',
       'initial_pool_value': 1000000.0
   }

   # Initialise pool parameters, equal weights. Equivalent to a Uniswap v2 pool
   params = {
       "initial_weights": jnp.array([0.5, 0.5]),
   }

   # Run simulation
   result = do_run_on_historic_data(run_fingerprint, params, verbose=True)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   tutorials/index
   user_guide/index
   api/index

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`