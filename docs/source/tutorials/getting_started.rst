Getting Started
===============

This tutorial will walk you through your first AMM simulation using quantammsim.

Your First Simulation
---------------------

Let's create a simple momentum-based AMM pool and run a simulation:

.. code-block:: python

    from quantammsim.runners.jax_runners import do_run_on_historic_data

    # Set up a basic simulation
    run_fingerprint = {
        'tokens': ['BTC', 'USDC'],
        'rule': 'balancer',
        'initial_pool_value': 1000000.0
    }

    params = {
        "initial_weights_logits": jnp.array([0.0, 0.0]),
    }

    # Run simulation
    result = do_run_on_historic_data(run_fingerprint, params, verbose=True)

Anything not set in the run_fingerprint will take on a default value

Understanding the Results
-------------------------

Let's examine what the simulation tells us:

.. code-block:: python

    # Access key metrics
    print(f"Final pool value: {result["value"][-1]}")


Now that you've run your first simulation, you might want to:

* Learn about the principles of dynamic pools (see :doc:`./introduction_to_dynamic_pools`)
* Read about how QuantAMM pools work (see :doc:`./quantamm_pools`)
* Explore Balancer, CowAMM, and Gyroscope pools (see :doc:`./balancer_pools`, :doc:`./cow_pools`, :doc:`./gyroscope_pools`)
* Learn about deeper mechanics and implementation of the pools (see :doc:`../api/core/pools`)

Basic Usage
-----------

Let's walk through a simple example of simulating a BTC/USDC QuantAMM pool with a momentum strategy:

.. code-block:: python

    from quantammsim.runners.jax_runners import do_run_on_historic_data

    # Define the basic parameters for our simulation
    run_fingerprint = {
        'tokens': ['BTC', 'USDC'],        # Token pair to simulate
        'rule': 'balancer',              # Weight update strategy
        'initial_pool_value': 1000000.0, # Starting liquidity
        'chunk_period': 60,              # Update frequency in minutes
        'startDateString': '2023-06-01 00:00:00',
        'endDateString': '2023-12-31 23:59:59'
    }

    params = {
        "initial_weights_logits": jnp.array([0.0, 0.0]),
    }

    # Run simulation
    result = do_run_on_historic_data(run_fingerprint, params, verbose=True)

    # The result contains various metrics and time series including:
    # - Token prices
    # - Pool weights
    # - Trading volumes
    # - Pool value over time

Advanced Configuration
----------------------

The run_fingerprint supports many additional parameters for fine-tuning the simulation:

.. code-block:: python

    run_fingerprint = {
        # ... basic parameters ...
        'fees': 0.003,                           # Trading fees (30 bps)
        'maximum_change': 0.0003                 # Max weight change per update
    }

Next Steps
----------

To learn more about:

* Different pool types and strategies, see :doc:`../user_guide/core_concepts`
* Detailed parameter configuration, see :doc:`../user_guide/run_fingerprints`
* Dive into the math and implementation details, see :doc:`../api/core/pools`