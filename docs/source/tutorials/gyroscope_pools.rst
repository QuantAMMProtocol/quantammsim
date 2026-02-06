Gyroscope Pools
===============

This tutorial explains Gyroscope pools, Elliptic Concentrated Liquidity Pools (E-CLP), with support for weight targeting and efficient GPU-accelerated computation.
These pools enable efficient concentrated liquidity within price bounds while still having fungible liquidity.

Key Features
~~~~~~~~~~~~

* E-CLP design with matrix-based price calculations
* Weight targeting through lambda/phi optimization
* Price bounds through alpha/beta parameters
* Dynamic fee support
* GPU-accelerated calculations

Core Mechanics
--------------

Gyroscope pools use matrix transformations to maintain price relationships and reserves using the following key equations:

.. math::

   A = \begin{bmatrix} 
   \cos(\phi)/\lambda & -\sin(\phi)/\lambda \\
   \sin(\phi) & \cos(\phi)
   \end{bmatrix}

where:

- :math:`\phi` is the rotation angle
- :math:`\lambda` is the scaling parameter
- The matrix :math:`A` determines price relationships

Basic Usage
-----------

Here's how to create and simulate a basic Gyroscope pool:

.. code-block:: python

    import jax.numpy as jnp

    run_fingerprint = {
        'tokens': ['ETH', 'USDC'],
        'rule': 'gyroscope',
        'initial_pool_value': 1000000.0,  # $1M initial pool value
        'alpha': 1500.0,                  # Lower price bound
        'beta': 4500.0,                   # Upper price bound
        'fees': 0.002,                    # 0.2% trading fee
        'gas_cost': 0.0001,              # Minimum profit threshold for arbitrage
        'arb_frequency': 1,              # How often arbitrageurs can act
        'do_arb': True                   # Enable arbitrage simulation
    }
    params = {
        'initial_weights': jnp.array([0.7, 0.3]),    # 70% ETH, 30% USDC
    }

    # Run the simulation
    result = do_run_on_historic_data(run_fingerprint, params)

Advanced Features
-----------------

Weight Targeting
~~~~~~~~~~~~~~~~

The pool can automatically optimize lambda and phi to achieve target weights if 'initial_weights' or 'initial_weights_logits' are provided in the params dict.
Otherwise 'lam' and 'phi' must be provided as singleton jnp arrays.

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
        'date': dates[:10],  # First 10 days
        'token_in_idx': [0, 1] * 5,  # Alternating tokens
        'token_out_idx': [1, 0] * 5,
        'amount_in': [1000.0] * 10  # Constant trade size
    })

    # Run simulation with dynamic parameters
    result = do_run_on_historic_data(
        run_fingerprint,
        fees_df=fees_df,
        raw_trades=trades_df
    )

Arbitrage Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune arbitrage behavior:

.. code-block:: python

    run_fingerprint.update({
        'gas_cost': 0.0001,              # Minimum profit threshold
        'arb_fees': 0.0002,              # External fees paid by arbitrageurs when they liquidate their positions
        'arb_frequency': 5,              # Check every 5 minutes
    })

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
   - Handles edge cases in matrix calculations
   - Robust arbitrage detection

Next Steps
----------

To learn more about:

* Different pool types, see :doc:`../user_guide/core_concepts`
* Implementation details, see :doc:`../api/core/pools`
* Mathematical foundations, see the `E-CLP paper <https://3407769812-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MU527HCtxlYaQoNazhF%2Fuploads%2FLK4MN8COTAR2EjAdQNlH%2FE-CLP%20Mathematics.pdf?alt=media&token=f77bc40b-9262-41de-bde1-55b000c7bd6e>`_ 