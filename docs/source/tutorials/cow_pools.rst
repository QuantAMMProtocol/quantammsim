CoW Pools
=========

This tutorial explains CoW (Coincidence of Wants) pools, which implement the CoW Protocol AMM design where post-trade quoted prices must match execution prices.

CoW pools are a unique type of AMM that ensures the quoted price for a trade matches the execution price. This design helps prevent price manipulation and provides better price discovery.

Key Features
~~~~~~~~~~~~

* Fixed 50-50 weights for token pairs
* Price-responsive reserve adjustments modelled via the action of arbitrageurs
* Support for dynamic trading fees

Pool Variants
-------------

The simulator provides two implementations:

1. Standard CoW Pool (``CowPool``)
   - Models multiple simultaneous arbitrageurs
   - More closely matches real-world behavior
   - Suitable for production simulations

2. Single Arbitrageur CoW Pool (``CowPoolOneArb``)
   - Assumes only one arbitrageur acts at a time
   - Useful for theoretical analysis and research
   - Models a worst-case scenario for LPs

Basic Usage
-----------

Here's how to create and simulate a basic CoW pool:

.. code-block:: python

    from quantammsim.runners.jax_runners import do_run_on_historic_data

    # Configure the simulation
    run_fingerprint = {
        'tokens': ['ETH', 'DAI'],
        'pool_type': 'cow',
        'initial_pool_value': 1000000.0,  # $1M initial pool value
        'fees': 0.001,                    # 0.1% fee per trade
        'gas_cost': 0.0001,               # Minimum profit threshold for arbitrage
        'arb_frequency': 1                # How often arbitrageurs can act
    }

    # Run the simulation
    result = do_run_on_historic_data(run_fingerprint)

Advanced Configuration
----------------------

CoW pools support several advanced parameters:

1. Dynamic Fees
   - Fees can vary over time
   - Can be based on market conditions
   - Specified through the fees_array parameter

2. Arbitrage Thresholds
   - Control when arbitrage occurs
   - Can model different levels of market efficiency
   - Set via arb_thresh_array

3. Trade Execution
   - Support for user-specified trades
   - Can simulate real trading patterns
   - Enabled through trade_array

Implementation Details
----------------------

The logic can be found in the class documentation :ref:`here <cow pool>`.
