Pool Architecture
=================

All pools in quantamm inherit from the :class:`~quantammsim.pools.base_pool.AbstractPool` base class, which defines the core interface that any pool must implement.
This includes critical methods for:

* Calculating reserves with and without fees
* Handling dynamic inputs (like fees or arbitrage thresholds/gas costs)
* Parameter initialization
* JAX-optimized implementations for high performance

The following sections provide more detail on the different pool types implemented in the simulator.
All pool types in quantammsim inherit from the :class:`~quantammsim.pools.base_pool.AbstractPool` base class, which is documented here.



Implementation Requirements
---------------------------

In order to implement a pool type, you need to implement the ``calculate_reserves_zero_fees`` and ``calculate_reserves_with_fees`` methods (as well as ``calculate_reserves_with_dynamic_inputs``, and some helper functions around that method, if you want to support time-varying fees or provide a sequence of trades to apply).
These functions all require the calculation of how arbitrageurs will act given pool state and external market prices.
For the pool types provided in this simulator, we (almost always) use externally-derived mathematical formulas for this process, which are then implemented in high-performance JAX code that can leverage GPU acceleration.
This means that years of simulation, at 1-minute price resolution, can take only the order of a few seconds on a basic laptop and fractions of a second on a GPU.
It is possible, however, to implement custom pool types using, for example, convex solvers, but this leads to a very noticeable slow-down in performance compared with the native JAX approach where the user implements hand-derived reserve update equations.

Out the box, the simulator implements three main types of pools:

Balancer Pools
~~~~~~~~~~~~~~
Traditional constant-weight geometric mean AMM pools, following the Balancer protocol design.


CowAMM Pools
~~~~~~~~~~~~

Implements the CoW Protocol AMM design where post-trade quoted prices must match execution prices.
Available in two variants:


QuantAMM Pools
~~~~~~~~~~~~~~
Dynamic pools that can adjust their weights based on market conditions.
All QuantAMM pools inherit from :class:`~quantammsim.pools.tfmm.TFMMBasePool` and implement different strategies:

For more detail, see :doc:`../tutorials/introduction_to_dynamic_pools` and :doc:`../tutorials/quantamm_pools`.

Deeper look at QuantAMM Pools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QuantAMM pools are dynamic AMMs that can adjust their weights based on market conditions.
All QuantAMM pools inherit from :class:`~quantammsim.pools.tfmm.TFMMBasePool` and implement the Temporal Function Market Making (TFMM) framework.


JAX Details
-----------

All pools and all strategies that come with this package use JAX for efficient computation:

* Strategy and weight calculations are JIT-compiled
* Reserve calculations are parallelised where possible (commonly in the zero-fees case) using JAX's ``vmap`` function
* GPU acceleration support by construction
* Batch processing of parameters supported for parallel simulations

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    run_fingerprint = {
        'tokens': ['BTC', 'ETH', 'USDC'],
        'initial_pool_value': 1_000_000,  # $1M initial pool value
        'fees': 0.001,  # 0.1% fee per trade
    }


.. _the Temporal Function Market Making litepaper: https://cdn.prod.website-files.com/6616670ddddc931f1dd3aa73/6617c4c2381409947dc42c7a_TFMM_litepaper.pdf
.. _this paper by the team on optimal arbitrage trades in G3Ms in the presence of fees: https://arxiv.org/abs/2402.06731
