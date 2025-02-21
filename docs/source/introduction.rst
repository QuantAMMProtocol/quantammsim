Introduction
============

Overview
--------

``quantammsim`` is a Python library for simulating and analyzing Automated Market Maker (AMM) protocols, with a 
particular focus on dynamic AMMs that can adapt their behavior based on market conditions.

Core Concepts
-------------

Before diving into usage, let's understand some key concepts:

Pools
~~~~~

Pools are the fundamental building blocks in quantammsim. They represent AMM liquidity pools that hold 
tokens and facilitate trades. The library supports several types of pools:

* **QuantAMM Pools**: Dynamic pools that can adjust their weights based on market conditions. These pools are implementations of *Temporal Function Market Making* (TFMM) pools detailed
  in `the TFMM litepaper`_. As well as enabling custom strategies to be implemented by the user, this package provides a number of strategies already implemented that are supported by the QuantAMM DeFi protocol:

  * Momentum: Adjusts weights based on recent price trends.
  * Anti-Momentum: Counters recent price movements.
  * Power Channel:  Weights are adjusted based on recent price trends raised to a power (e.g. the square).
  * Mean Reversion Channel: For small price changes weights move to counter price movements, for larger price changes weights move with the price in a non-linear fashion.

* **Balancer Pools**: Traditional constant-weight geometric-mean AMM pools as implemented in `Balancer`_.
* **CowAMM Pools**: Pool that require post-trade quoted prices match the trade execution price, as
  described in `this paper`_ and implemented in the `CowAMM`_ DeFi protocol.
* **Gyroscope Pools**: Pools that use elliptical curves as the trading function, able to provide concentrated liquidity within price bounds while still having a smooth trading function and fungible liquidity.

Runners
~~~~~~~

Runners are high-level interfaces that handle the execution of simulations. They manage:

* Data loading and preprocessing
* Pool initialization
* Simulation execution
* Result collection and analysis

The can also handle a wide variety of different choices of pool fees, the gas cost paid by arbitrageurs,
and other simulation choices.

As well as simple backtest simulations, the package also supports the training of pool to improve
their performance on historic or synthetic data. This is most relevant for QuantAMM pools, as they
themselves *run* strategies and those strategies include parameters that must be chosen by a pool creator. For 
example, using the appropriate runner, a user could tune a QuantAMM pool that implements a momentum strategy
to maximize the Sharpe ratio (over a chosen price series) by varying the aggressiveness and lookback period of the strategy.

Hooks
~~~~~

Hooks are a way to customize the behavior of AMM pools. They are functions that are called at various points in the pools operation/in the simulator.
In this package, hooks are used to alter the fees charged by pools, for example tuning fees up or down as asset volatility changes.
These hooks can contain parameters that can then be tuned as part of the optimization process, for example aiming to minimise the drawdown of the pool.
This functionality works with pools that otherwise have no parameters that one would otherwise tune, such as base Balancer pools.

Run Fingerprints
~~~~~~~~~~~~~~~~

Run fingerprints are dictionaries that define the settings for a simulation. They specify:

* Tokens/assets to simulate
* Pool type and strategy
* Time period and frequency of oracle calls (where relevant)
* Pool fees
* Arbitrageur gas costs
* Initial conditions of the pool
* Any additional constraints

Key Features
------------

* Simulation of multiple AMM protocols (Balancer, QuantAMM, CowAMM and Gyroscope)
* Various weight-updating strategies for QuantAMM pools
* Historical data analysis
* Performance metrics calculation

Quick Start
~~~~~~~~~~~

.. code-block:: python

   from quantammsim.runners.jax_runners import do_run_on_historic_data
   
   run_fingerprint = {
       'tokens': ['BTC', 'USDC'],
       'rule': 'balancer',
       'initial_pool_value': 1000000.0,
       'startDateString': '2024-01-01 00:00:00',
       'endDateString': '2024-06-15 00:00:00',
   }
   
    params = {
       "initial_weights_logits": jnp.array([0.0, 0.0]),
    }

    # Run simulation
    result = do_run_on_historic_data(run_fingerprint, params, verbose=True)

.. _this paper: https://arxiv.org/abs/2307.02074
.. _the TFMM litepaper: https://cdn.prod.website-files.com/6616670ddddc931f1dd3aa73/6617c4c2381409947dc42c7a_TFMM_litepaper.pdf
.. _CowAMM: https://docs.cow.fi/cow-amm/concepts/how-cow-amms-work
.. _Balancer: https://balancer.fi