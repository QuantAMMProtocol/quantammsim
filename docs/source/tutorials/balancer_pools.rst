Balancer Pools
==============

This tutorial explains Balancer pools, which implement constant-weight automated market making with support for multiple tokens and efficient GPU-accelerated computation.

Key Features
~~~~~~~~~~~~

* Constant weight ratios between tokens
* Multi-token support (up to 8 tokens)
* GPU-accelerated calculations
* Sophisticated arbitrage modeling
* Dynamic parameter support

Core Mechanics
--------------

Balancer pools maintain constant weight ratios between tokens using the following key equation:

.. math::

   \frac{w_i}{w_j} \cdot \frac{R_j}{R_i} = P_{i,j}

where:
- :math:`w_i` is the weight of token i
- :math:`R_i` is the reserve of token i
- :math:`P_{i,j}` is the spot price of token i in terms of token j

The pool automatically maintains these ratios through arbitrage opportunities (in the prescense of fees, the exact details get complicated but are modellable using, for example, the results in `this paper from the team`_).

Basic Usage
-----------

Here's how to create and simulate a basic Balancer pool:

.. code-block:: python

    from quantammsim.runners.jax_runners import do_run_on_historic_data

    # Configure a 80/20 BTC/DAI pool
    run_fingerprint = {
        'tokens': ['BTC', 'DAI'],
        'pool_type': 'balancer',
        'initial_pool_value': 1000000.0,  # $1M initial pool value
        'weights': [0.8, 0.2],           # 80% BTC, 20% DAI
        'fees': 0.002,                   # 0.2% trading fee
        'gas_cost': 0.0001,              # Minimum profit threshold for arbitrage
        'arb_frequency': 1,              # How often arbitrageurs can act
        'do_arb': True                   # Enable arbitrage simulation
    }

    # Run the simulation
    result = do_run_on_historic_data(run_fingerprint)

Advanced Features
-----------------


Multi-Token Pools
~~~~~~~~~~~~~~~~~

Balancer supports pools with any number of tokens (the protocol supports up to 8 in the current implementation):

.. code-block:: python

    # Create a three-token pool
    run_fingerprint = {
        'tokens': ['ETH', 'BTC', 'DAI'],
        'pool_type': 'balancer',
        'initial_pool_value': 1000000.0,
        'weights': [0.4, 0.4, 0.2],      # 40/40/20 split
        'fees': 0.002,
        'do_arb': True
    }



Arbitrage Modeling
~~~~~~~~~~~~~~~~~~

The simulator provides two arbitrage models:

1. Perfect Arbitrage
   - Models multiple simultaneous arbitrageurs
   - Achieves optimal price alignment
   - Used for ideal market conditions

2. Single Arbitrageur
   - Models scenarios with limited arbitrage activity
   - More conservative price alignment
   - Represents real-world friction

Dynamic Parameters
~~~~~~~~~~~~~~~~~~

The pool supports time-varying parameters:

.. code-block:: python

    import numpy as np
    import pandas as pd

    # Create time-varying fees
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    fees_df = pd.DataFrame({
        'date': dates,
        'fee': np.linspace(0.001, 0.002, len(dates))  # Fees increasing over time
    })

    # Create custom trades
    trades_df = pd.DataFrame({
        'date': dates[:10],
        'token_in_idx': [0, 1] * 5,      # Alternating tokens
        'token_out_idx': [1, 0] * 5,
        'amount_in': [1000.0] * 10       # Constant trade size
    })

    result = do_run_on_historic_data(
        run_fingerprint,
        fees_df=fees_df,
        trades_df=trades_df
    )

Arbitrage Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune arbitrage behavior:

.. code-block:: python

    run_fingerprint.update({
        'gas_cost': 0.0001,              # Minimum profit threshold
        'arb_fees': 0.0002,              # External arbitrage costs
        'arb_frequency': 5,              # Check every 5 minutes
        'all_sig_variations': [...],     # Custom arbitrage patterns
    })

Implementation Details
----------------------

The pool implements three main calculation modes:

1. Standard Trading (``calculate_reserves_with_fees``)
   - Handles regular trading with fees
   - Maintains constant weight ratios
   - Supports arbitrage simulation

2. Zero Fee Trading (``calculate_reserves_zero_fees``)
   - Special case for fee-less trading
   - Useful for theoretical analysis
   - Maintains weight ratios

3. Dynamic Input Trading (``calculate_reserves_with_dynamic_inputs``)
   - Supports time-varying parameters
   - Handles custom trade sequences
   - Most flexible configuration

The implementation uses JAX for efficient computation and supports both CPU and GPU execution.

Performance Considerations
--------------------------

1. GPU Acceleration
   - All core calculations are JAX-accelerated
   - Supports parallel processing of trades
   - Efficient handling of large datasets

2. Memory Usage
   - Optimized for long simulations
   - Efficient precalculation of common values
   - Smart broadcasting of parameters

3. Numerical Stability
   - Uses 64-bit precision
   - Handles edge cases in weight calculations
   - Robust arbitrage detection

Next Steps
----------

To learn more about:

* Different pool types, see :doc:`../user_guide/core_concepts`
* Implementation details, see :doc:`../api/core/pools`
* Mathematical foundations, see the `Balancer whitepaper <https://balancer.fi/whitepaper.pdf>`_

.. note::
   For full API documentation, see :class:`quantammsim.pools.BalancerPool`

.. _this paper from the team: https://arxiv.org/abs/2402.06731
