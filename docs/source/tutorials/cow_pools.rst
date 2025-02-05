CoW Pools
=========

This tutorial explains CoW (Coincidence of Wants) pools, which implement the CoW Protocol AMM design where post-trade quoted prices must match execution prices.

CoW pools are a unique type of AMM that ensures the quoted price for a trade matches the execution price. This design helps prevent price manipulation and provides better price discovery.

Key Features
~~~~~~~~~~~~

* Fixed weights for token pairs (configurable in advanced usage)
* Price-responsive reserve adjustments through sophisticated arbitrage modeling
* Support for dynamic trading fees and parameters
* Flexible arbitrage quality modeling


Core Mechanics
--------------

CoW pools maintain price consistency using the following key equation:

.. math::

   P_{exec} = P_{quote}

where:
- :math:`P_{exec}` is the execution price
- :math:`P_{quote}` is the quoted price


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
        'arb_frequency': 1,               # How often arbitrageurs can act
        'arb_quality': 0.8                # Blend of perfect/imperfect arbitrage
    }

    # Run the simulation
    result = do_run_on_historic_data(run_fingerprint)


Advanced Features
-----------------

Arbitrage Modeling
~~~~~~~~~~~~~~~~~~

The simulator provides two arbitrage models that can be combined:

1. Perfect Arbitrage
   - Models multiple simultaneous arbitrageurs
   - Achieves optimal price alignment
   - Used for ideal market conditions

2. Single Arbitrageur
   - Models scenarios with limited arbitrage activity
   - More conservative price alignment
   - Represents real-world friction

The `arb_quality` parameter (0-1) allows blending between these models:
- 1.0 = perfect arbitrage
- 0.0 = single arbitrageur
- Values between represent partial market efficiency

Dynamic Parameters
~~~~~~~~~~~~~~~~~~

CoW pools support several dynamic parameters that can vary over time:

1. Trading Fees
   - Configurable per-minute fees or set as a single value
   - Can respond to market conditions
   - Per minute fees are specified through `fees_array` (but can be a single value `fees` in the run fingerprint)

2. Arbitrage Parameters
   - Threshold for profitable arbitrage (`arb_thresh_array`) often coterminous with gas costs (but can be a single value `arb_fees` in the run fingerprint)
   - Arbitrage fees (`arb_fees_array`), the fees the arbitrageur has to pay to liquidate the position they have after the arbitrage trade

3. Trade Execution
   - Support for user-specified trades via `trade_array`
   - Format: [token_in_idx, token_out_idx, amount_in]
   - Can model specific trading patterns


Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~

For sophisticated simulations with time-varying parameters:

.. code-block:: python

    from quantammsim.runners.jax_runners import do_run_on_historic_data
    import pandas as pd
    import numpy as np

    # Create time series data for dynamic parameters
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    fees_df = pd.DataFrame({
        'date': dates,
        'fee': np.linspace(0.001, 0.002, len(dates))  # Fees increasing over time
    })
    
    gas_cost_df = pd.DataFrame({
        'date': dates,
        'gas_cost': np.full(len(dates), 0.0001)  # Constant gas costs
    })
    
    arb_fees_df = pd.DataFrame({
        'date': dates,
        'arb_fee': np.full(len(dates), 0.0002)  # Constant arbitrage fees
    })

    # Define specific trades to execute
    raw_trades = pd.DataFrame({
        'date': dates[:10],  # First 10 days
        'token_in_idx': [0, 1, 0, 1] * 2 + [0, 1],  # Alternating tokens
        'token_out_idx': [1, 0, 1, 0] * 2 + [1, 0],
        'amount_in': [1000.0] * 10  # Constant trade size
    })

    # Run simulation with dynamic parameters
    result = do_run_on_historic_data(
        run_fingerprint,
        fees_df=fees_df,
        gas_cost_df=gas_cost_df,
        arb_fees_df=arb_fees_df,
        raw_trades=raw_trades,
        do_test_period=True  # Enable test period simulation
    )

Arbitrage Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune arbitrage behavior:

.. code-block:: python

    run_fingerprint.update({
        'gas_cost': 0.0001,              # Minimum profit threshold
        'arb_fees': 0.0002,              # External arbitrage costs
        'arb_frequency': 5,              # Check every 5 minutes
        'arb_quality': 0.8,              # Blend between perfect/imperfect arbitrage
        'all_sig_variations': [...],     # Custom arbitrage patterns
    })

The `arb_quality` parameter (0-1) controls how efficiently arbitrage opportunities are captured:
- 1.0 = perfect arbitrage (multiple simultaneous arbitrageurs perfectly compete, resulting in rebalancing at the true price)
- 0.0 = single arbitrageur (more pessimistic market conditions, the one arbitrageur is able to capture the arbitrage opportunity giving the pool less surplus)
- Values between represent partial market efficiency
- Default is 0.8 (mostly efficient but not perfect)

This allows simulating different market conditions, from ideal (perfect arbitrage) to more realistic scenarios with market frictions and limited arbitrageur participation.

Implementation Details
----------------------

The pool implements three main calculation modes:

1. Standard Fee Calculation (`calculate_reserves_with_fees`)
   - Handles regular trading with fees
   - Supports both perfect and imperfect arbitrage
   - Considers gas costs and arbitrage thresholds

2. Zero Fee Calculation (`calculate_reserves_zero_fees`)
   - Special case for fee-less trading
   - Useful for theoretical analysis
   - Maintains arbitrage modeling

3. Dynamic Input Calculation (`calculate_reserves_with_dynamic_inputs`)
   - Supports time-varying parameters
   - Handles custom trade sequences
   - Most flexible configuration

The implementation uses JAX for efficient computation and supports both CPU and GPU execution.
This functions are called by the `do_run_on_historic_data` function, which is the main entry point for running simulations.

For implementation details, see the source code in :mod:`quantammsim.pools.FM_AMM.cow_pool`.

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
   - Handles edge cases in calculations
   - Robust arbitrage detection

Next Steps
----------

To learn more about:

* Different pool types, see :doc:`../user_guide/core_concepts`
* Advanced features, see :doc:`./advanced_usage`
* Implementation details, see :doc:`../api/core/pools`
* Mathematical foundations, see the CoW Protocol documentation

