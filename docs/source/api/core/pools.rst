Pools
=====

Pool Architecture
-----------------

All pools in quantamm inherit from the ``AbstractPool`` base class, which defines the core interface that any pool must implement. This includes critical methods for:

* Calculating reserves with and without fees
* Handling dynamic inputs (like fees or arbitrage thresholds/gas costs)
* Parameter initialization
* JAX-optimized implementations for high performance

The following sections provide more detail on the different pool types implemented in the simulator.
All pool types in quantammsim inherit from the ``AbstractPool`` base class, which is documented here.

Base Pool Interface
~~~~~~~~~~~~~~~~~~~

.. autoclass:: quantammsim.pools.AbstractPool
   :members:
   :undoc-members:
   :show-inheritance:

Pool Types
----------

In order to implement a pool type, you need to implement the ``calculate_reserves_zero_fees`` and ``calculate_reserves_with_fees`` methods (as well as ``calculate_reserves_with_dynamic_inputs``, and some helper functions around that method, if you want to support time-varying fees or provide a sequence of trades to apply).
These functions all require the calculation of how arbitrageurs will act given pool state and external market prices.
For the pool types provided in this simulator, we (almost always) use externally-derived mathematical formulas for this process, which are then implemented in high-performance JAX code that can leverage GPU acceleration.
This means that years of simulation, at 1-minute price resolution, can take only the order of a few seconds on a basic laptop and fractions of a second on a GPU.
It is possible, however, to implement custom pool types using, for example, convex solvers, but this leads to a very noticeable slow-down in performance compared with the native JAX approach where the user implement hand-derived reserve update equations.

Out the box, the simulator implements three main types of pools:

Balancer Pools
~~~~~~~~~~~~~~
Traditional constant-weight geometric mean AMM pools, following the Balancer protocol design.

.. autoclass:: quantammsim.pools.BalancerPool
   :members: calculate_reserves_zero_fees, calculate_reserves_with_fees
   :noindex:

.. _cow pool:

CowAMM Pools
~~~~~~~~~~~~

Implements the CoW Protocol AMM design where post-trade quoted prices must match execution prices. Available in two variants:

.. autoclass:: quantammsim.pools.CowPool
   :members: calculate_reserves_zero_fees, calculate_reserves_with_fees
   :noindex:

.. autoclass:: quantammsim.pools.CowPoolOneArb
   :members: calculate_reserves_zero_fees, calculate_reserves_with_fees
   :noindex:

QuantAMM Pools
~~~~~~~~~~~~~~
Dynamic pools that can adjust their weights based on market conditions. All QuantAMM pools inherit from ``TFMMBasePool`` and implement different strategies:

.. autoclass:: quantammsim.pools.TFMMBasePool
   :members: calculate_weights, calculate_raw_weights_outputs
   :noindex:

Strategy Implementations:

* **Momentum Pool**: Adjusts weights based on recent price trends
* **Anti-Momentum Pool**: Counters recent price movements
* **Power Channel Pool**: Uses non-linear power functions of price changes
* **Mean Reversion Channel Pool**: Combines mean reversion for small moves with trend-following for large moves

For more detail, see the deep dive below.

Deeper look at QuantAMM Pools
-----------------------------

QuantAMM pools are dynamic AMMs that can adjust their weights based on market conditions. All QuantAMM pools inherit from ``TFMMBasePool`` and implement the Temporal Function Market Making (TFMM) framework.

Weight Update Mechanism
~~~~~~~~~~~~~~~~~~~~~~~

The weight update process follows these steps:

1. Calculate raw weight outputs based on oracle values (commonly prices) in accordance with a chosen strategy.
2. Apply guardrail constraints (weights must have values larger than a set minimum, and changes in weights have to be below a chosen speed limit) and normalization
3. Carry our weight interpolation over the coming chosen interpolation period. This can be linear interpolation (weights change by a fixed amount per unit time) or can be non-linear. An apporximately-optimal non-linear interpolation is provided, but users can develope and implement any other interpolation method they are interested in experimenting with.
4. Model how reserves change from the action of arbitrageurs interacting with the pool.

Step 4. can also include application of fees (including dynamic fees), gas costs paid by arbitrageurs, and also having the pool fulfill particular trades provided as an input to the simulation by the user.

Pre-implemented Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide here in the simulatorthe same strategies that are implemented in V1 of the QuantAMM protocol. Users can also experiment with custom strategies in this simulator, but support for custom strategies in the onchain protocol is not yet fully implemented. If you are interested in deploying a custom strategy as a smart contract, please get in touch.

Momentum Pool
^^^^^^^^^^^^^
.. autoclass:: quantammsim.pools.MomentumPool
   :members: calculate_weights, calculate_raw_weights_outputs
   :noindex:

The momentum strategy adjusts weights based on recent price trends. Key parameters:

* ``memory_days``: Lookback period for trend calculation
* ``aggressiveness``: How strongly to respond to trends
* ``min_weight``: Floor on individual asset weights

Anti-Momentum Pool
^^^^^^^^^^^^^^^^^^
.. autoclass:: quantammsim.pools.AntiMomentumPool
   :members: calculate_weights, calculate_raw_weights_outputs
   :noindex:

Implements a contrarian strategy that moves weights against recent price trends. Uses the same parameters as Momentum Pool but responds in the opposite direction.

Power Channel Pool
^^^^^^^^^^^^^^^^^^
.. autoclass:: quantammsim.pools.PowerChannelPool
   :members: calculate_weights, calculate_raw_weights_outputs
   :noindex:

Uses non-linear power functions of price changes. Additional parameters:

* ``power``: Exponent applied to price changes
* ``channel_width``: Width of the neutral zone

Mean Reversion Channel Pool
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: quantammsim.pools.MeanReversionChannelPool
   :members: calculate_weights, calculate_raw_weights_outputs
   :noindex:

Combines mean reversion for small moves with (power channel) trend-following for large moves:

* For price changes within the channel: Acts like Anti-Momentum
* For price changes outside the channel: Acts like Power Channel
* ``channel_width`` parameter controls this threshold

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

All strategies use JAX for efficient computation:

* Strategy and weight calculations are JIT-compiled
* Reserve calculations are parallelised where possible (commonly in the zero-fees case)
* Batch processing of parameters is supported for parallel simulations

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint = {
        'tokens': ['BTC', 'ETH', 'USDC'],
        'initial_pool_value': 1_000_000,  # $1M initial pool value
        'fees': 0.001,  # 0.1% fee per trade
    }


Performance Optimisation
------------------------

All pools use JAX-optimized implementations with:

* JIT-compiled core functions
* Vectorized operations using ``jax.vmap``
* GPU acceleration support
* Efficient parameter handling for parallel simulations