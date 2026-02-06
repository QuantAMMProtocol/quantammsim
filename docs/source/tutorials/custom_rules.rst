Implementing a Custom QuantAMM Strategy
=======================================

When designing a QuantAMM strategy, an *update rule*, you're essentially creating a function that maps market observations to weight changes. Key considerations:

Weight Calculation Paths
~~~~~~~~~~~~~~~~~~~~~~~~

There are two approaches to implementing weight calculation in a custom strategy:

1. **Vectorized Path** - Compute all weight changes at once using convolution operations (faster, good for simulation)
2. **Scan Path** - Compute weight changes sequentially, one step at a time (matches on-chain execution)

You can implement one or both paths. See :doc:`../user_guide/weight_calculation_paths` for detailed information.

Vectorized Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

The vectorized approach computes all weight outputs at once. This is typically faster for simulation and training.

.. code-block:: python

    from jax import jit
    from jax import numpy as jnp
    from typing import Dict, Any, Optional
    from functools import partial
    from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool

    class MyCustomRule(TFMMBasePool):
        @partial(jit, static_argnums=(2))
        def calculate_rule_outputs(
            self,
            params: Dict[str, Any],
            run_fingerprint: Dict[str, Any],
            prices: jnp.ndarray,
            additional_oracle_input: Optional[jnp.ndarray] = None,
        ) -> jnp.ndarray:
            # Your vectorized weight calculation logic here
            # Returns weight changes for all time steps at once
            ...

Scan-Based Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

The scan-based approach processes prices one at a time, maintaining state between steps.
This mirrors how weights are computed on-chain and is useful for verifying production behavior.

.. code-block:: python

    class MyCustomRule(TFMMBasePool):
        def get_initial_rule_state(
            self,
            initial_price: jnp.ndarray,
            params: Dict[str, Any],
            run_fingerprint: Dict[str, Any],
        ) -> Dict[str, jnp.ndarray]:
            """Initialize the estimator carry state from the first price."""
            # Return initial state dictionary
            return {
                "ewma": initial_price,
                # ... other state variables
            }

        def calculate_rule_output_step(
            self,
            carry: Dict[str, jnp.ndarray],
            price: jnp.ndarray,
            params: Dict[str, Any],
            run_fingerprint: Dict[str, Any],
        ) -> tuple:
            """Single-step weight update.

            Returns:
                (new_carry, rule_output) tuple
            """
            # Update state and compute weight change for this step
            new_carry = {
                "ewma": ...,  # Updated state
            }
            rule_output = ...  # Weight change for this step
            return new_carry, rule_output

Implementing Both Paths
~~~~~~~~~~~~~~~~~~~~~~~

For maximum flexibility, implement both paths. They should produce numerically equivalent results:

.. code-block:: python

    class MyCustomRule(TFMMBasePool):
        # Vectorized path
        def calculate_rule_outputs(self, params, run_fingerprint, prices, ...):
            ...

        # Scan path
        def get_initial_rule_state(self, initial_price, params, run_fingerprint):
            ...

        def calculate_rule_output_step(self, carry, price, params, run_fingerprint):
            ...

    # Check which paths are supported
    pool = MyCustomRule()
    print(pool.supports_vectorized_path())  # True
    print(pool.supports_scan_path())        # True

Creating a Custom Rule
~~~~~~~~~~~~~~~~~~~~~~

To create a custom update rule:

1. Inherit from :class:`~quantammsim.pools.TFMMBasePool`
2. Implement **either** the vectorized path (``calculate_rule_outputs``) **or** the scan path (``get_initial_rule_state`` + ``calculate_rule_output_step``), or both
3. (Optional) Provide the logic for any custom parameters in the pool's helper function :meth:`~quantammsim.pools.AbstractPool.init_base_parameters`
4. Register the pool with JAX as a pytree node using :func:`~jax.tree_util.register_pytree_node` (see note in :doc:`../tutorials/custom_pools`)

Note that the simulator does not enforce causality, so be careful to make sure no look-ahead bias is introduced in the raw weight calculation.
If you stick to using provided QuantAMM estimators, e.g. the gradient estimator :func:`~quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators.calc_gradients`, then you can be confident that no look-ahead bias is introduced.


Using Custom Rules
~~~~~~~~~~~~~~~~~~

To use your custom rule, add it to the function :func:`~quantammsim.pools.creator.create_pool` giving it a string name, and then pass this string name to the ``rule`` key in the ``run_fingerprint`` dictionary to use it.

.. code-block:: python

    run_fingerprint = {
        'tokens': ['BTC', 'USDC'],
        'rule': 'custom',  # Must register your rule in the creator.py file
        'initial_pool_value': 1000000.0
    }

Design Considerations
~~~~~~~~~~~~~~~~~~~~~

When designing update rules, consider:

1. **Responsiveness**: How quickly should weights change?
2. **Stability**: Avoid oscillations or extreme changes
3. **Robustness**: Handle edge cases and unusual market conditions
4. **Computational Efficiency**: Rules run frequently, keep them fast
5. **Memory Usage**: Consider how much historical data you need

The base TFMM implementation handles many edge cases and constraints, allowing you to focus on the core strategy logic in your update rule.
Note of course that if the strategy is novel, for QuantAMM V1 it will have to be implemented as a smart contract. Contact the team if you'd like to discuss.


.. _the Temporal Function Market Making litepaper: https://cdn.prod.website-files.com/6616670ddddc931f1dd3aa73/6617c4c2381409947dc42c7a_TFMM_litepaper.pdf
.. _this paper by the team on optimal arbitrage trades in G3Ms in the presence of fees: https://arxiv.org/abs/2402.06731
