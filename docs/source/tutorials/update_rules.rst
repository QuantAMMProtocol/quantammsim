Update Rules and Pool Mechanics
===============================

This tutorial explains the fundamental concepts behind QuantAMM pools (which implement Temporal Function Market Making) and their update rules.

Core Concepts
-------------

Weight Vectors and Pool Composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At its heart, a TFMM pool is defined by its weight vector. For a pool with n tokens, the weight vector :math:`\mathbf{w} = (w_1, \ldots, w_n)` determines:

1. The desired ratio of value held in each token
2. The prices at which the pool will trade

For example, in a BTC/DAI pool:
- :math:`\mathbf{w} = (0.5, 0.5)` means the pool wants equal value in both tokens
- :math:`\mathbf{w} = (0.7, 0.3)` means the pool wants 70% of its value in BTC, 30% in DAI

.. note::
   Weights must always sum to 1.0, and each weight must stay above a minimum threshold (typically around 0.02 or 2%).

From Weights to Trading
~~~~~~~~~~~~~~~~~~~~~~~

When a pool's actual token composition differs from its weight vector, it creates arbitrage opportunities. Here's how:

1. The pool's spot price for trading token i for token j is given by:

   .. math::

      P_{i,j} = \frac{w_i}{w_j} \cdot \frac{R_j}{R_i}

   where :math:`R_i` and :math:`R_j` are the token reserves.

2. If this price differs from the market price, arbitrageurs can profit by trading with the pool.

3. These trades naturally move the pool's composition toward its target weights.

Example:
   If :math:`\mathbf{w} = (0.6, 0.4)` but the pool holds equal values of tokens, arbitrageurs will:
   - Buy token 1 from the pool (it's undervalued according to weights)
   - Sell token 2 to the pool (it's overvalued according to weights)
   Until the value ratio matches 60:40

Dynamic Weight Updates
~~~~~~~~~~~~~~~~~~~~~~

Update rules make pools dynamic by adjusting their weight vectors over time. The process:

1. Observe market conditions (usually via price oracles)
2. Calculate desired new weights
3. Apply constraints (min weights, max change speed)
4. Interpolate between old and new weights
5. Let arbitrageurs trade to match new weights

This creates pools that can adapt to market conditions while maintaining controlled, predictable behavior.

Creating Update Rules
---------------------

When designing an update rule, you're essentially creating a function that maps market observations to weight changes. Key considerations:

Weight Calculation
~~~~~~~~~~~~~~~~~~

The core of any update rule is the logic that converts market observations into desired weight changes. This typically involves:

.. code-block:: python

    from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool

    class MyCustomRule(TFMMBasePool):
        def calculate_raw_weights_outputs(self, state):
            # Your weight calculation logic here
            # Example: increase weight when price increases
            price_change = (state.current_price - state.previous_price) 
            weight_change = self.aggressiveness * price_change
            return current_weights + weight_change

Constraints and Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All weight changes must respect certain constraints:

1. Minimum weight bounds (e.g., no weight below 0.02)
2. Maximum change speed (to prevent extreme shifts)
3. Weights must sum to 1.0

The base TFMM implementation handles these automatically after your raw calculation.

Weight Interpolation
~~~~~~~~~~~~~~~~~~~~

Rather than jumping directly to new weights, changes are typically interpolated over time to:

- Reduce market impact
- Make arbitrage easier
- Improve price stability

Creating Custom Rules
---------------------

To create a custom update rule:

1. Inherit from TFMMBasePool
2. Implement method ``calculate_raw_weights_outputs()``
3. (Optional) Add custom parameters


Example Template
~~~~~~~~~~~~~~~~

.. code-block:: python

    class CustomUpdateRule(TFMMBasePool):
        def __init__(self, params):
            super().__init__(params)
            # Add any custom parameters
            self.sensitivity = params.get('sensitivity', 1.0)
            self.lookback = params.get('lookback', 10)
            
        def calculate_raw_weights_outputs(self, state):
            # Get relevant price history
            price_history = state.price_history[-self.lookback:]
            
            # Calculate your signals
            signal = self._calculate_signal(price_history)
            
            # Convert signal to weight changes
            weight_changes = self.sensitivity * signal
            
            # Return new weights
            return state.current_weights + weight_changes
            
        def _calculate_signal(self, prices):
            # Your custom signal logic
            return some_calculation(prices)

Using Custom Rules
------------------

To use your custom rule:

.. code-block:: python

    run_fingerprint = {
        'tokens': ['BTC', 'DAI'],
        'rule': 'custom',  # Must register your rule
        'sensitivity': 0.5,
        'lookback': 20,
        'initial_pool_value': 1000000.0
    }

Design Considerations
---------------------

When designing update rules, consider:

1. **Responsiveness**: How quickly should weights change?
2. **Stability**: Avoid oscillations or extreme changes
3. **Robustness**: Handle edge cases and unusual market conditions
4. **Computational Efficiency**: Rules run frequently, keep them fast
5. **Memory Usage**: Consider how much historical data you need

The base TFMM implementation handles many edge cases and constraints, allowing you to focus on the core strategy logic in your update rule.
Note of course that if the strategy is novel, for QuantAMM V1 it will have to be implemented as a smart contract. Contact the team if you'd like to discuss.

Practical Constraints
~~~~~~~~~~~~~~~~~~~~~

Real-world considerations:

1. Gas costs: Complex calculations are expensive on-chain
2. Numerical stability: Avoid potential overflows or division by zero
3. Gaming resistance: Consider how arbitrageurs might exploit your rule
4. Market impact: Sudden weight changes can cause large price moves


Monitor key metrics:
- Pool value (vs buy-and-hold)
- Trading volume and fees
- Weight trajectories
- Price impact of updates

Next Steps
----------
- Study the `TFMM paper <https://quantamm.fi/research>`_ for mathematical foundations
- Examine existing strategy implementations
- Start with simple rules and gradually add complexity
- Test across different market conditions