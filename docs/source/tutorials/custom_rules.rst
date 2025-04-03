Implementing a Custom QuantAMM Strategy
=======================================

When designing a QuantAMM strategy, an *update rule*, you're essentially creating a function that maps market observations to weight changes. Key considerations:

Weight Calculation
~~~~~~~~~~~~~~~~~~

The core of any update rule is the logic that converts market observations into desired weight changes.

.. code-block:: python

    from jax import jit
    from jax import numpy as jnp
    from typing import Dict, Any, Optional
    from functools import partial
    from quantammsim.pools.G3M.quantamm.TFMM_base_pool import TFMMBasePool

    class MyCustomRule(TFMMBasePool):
        @partial(jit, static_argnums=(2))
        def calculate_raw_weights_outputs(
            self,
            params: Dict[str, Any],
            run_fingerprint: Dict[str, Any],
            prices: jnp.ndarray,
            additional_oracle_input: Optional[jnp.ndarray] = None,
        ) -> jnp.ndarray:
            # Your weight calculation logic here


To create a custom update rule:

1. Inherit from :class:`~quantammsim.pools.TFMMBasePool`
2. Implement method :meth:`~quantammsim.pools.TFMMBasePool.calculate_raw_weights_outputs`
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
