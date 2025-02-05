Advanced Usage
==============

This tutorial covers advanced features of quantamm.


Creating Update Rules
---------------------

When designing an update rule, you're essentially creating a function that maps market observations to weight changes. Key considerations:

Weight Calculation
~~~~~~~~~~~~~~~~~~

The core of any update rule is the logic that converts market observations into desired weight changes.

.. code-block:: python

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

1. Inherit from TFMMBasePool
2. Implement method ``calculate_raw_weights_outputs()``
3. (Optional) Add custom parameters

Note that the simulator does not enforce causality, so be careful to make sure no look-ahead bias is introduced in the raw weight calculation.
If you stick to using provided QuantAMM estimators, e.g. the gradient estimator :func:`~quantammsim.pools.G3M.quantamm.update_rule_estimators.estimators.calc_gradients`, then you can be confident that no look-ahead bias is introduced.

The base TFMM implementation handles these automatically after the raw weight calculation.


Using Custom Rules
------------------

To use your custom rule, add it to the function :func:`~quantammsim.pools.creator.create_pool`` giving it a string name, and then pass this string name to the ``rule`` key in the ``run_fingerprint`` dictionary to use it.

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


Implementing a new pool type
----------------------------

Above we have seen how to implement a custom update rule *for* a QuantAMM pool.

What if you have a totally new type of AMM?
To implement a new pool *type*, you need to create a new class that implements abstract methods from the :class:`~quantammsim.pools.base_pool.AbstractPool` interface.

Let's walk through an example implementation, looking at how :class:`~quantammsim.pools.G3M.balancer.balancer.BalancerPool` implements these requirements.

Note that pools do not hold any state, they only have methods.
This makes them much easier to make work with JAX, which as a semi-functional language is not well-suited to object-oriented programming.
The pool classes thus almost act (very informally speaking) as a namespace for methods.

Core Implementation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Looking at the :class:`~quantammsim.pools.base_pool.AbstractPool` interface, we need to implement:

i. Reserve Calculation Methods

   These methods determine how pool reserves change in response to market conditions:

   * ``calculate_reserves_zero_fees`` - Optimized calculation without fees
   * ``calculate_reserves_with_fees`` - Handles arbitrage with fees charged by the pool (and also gas costs paid by arbitrageurs and even the fees arbitrageurs may pay on a secondary market to liquidate their positions)
   * ``calculate_reserves_with_dynamic_inputs`` - Supports time-varying fees, gas costs, and more (including applying a sequence of [not necessarily arbitrage] trades to the pool).

For full funcionality, all three should be implemented, but for quick testing it is often sufficient to implement only the zero-fees case, which often is substantially faster and simpler than the other two.

ii. Configuration Methods

   These methods handle pool setup and parameters:

   * ``init_base_parameters`` - Initialize pool configuration
   * ``make_vmap_in_axes`` - Configure JAX vectorization
   * ``is_trainable`` - Determine if weights can be trained

See :class:`~quantammsim.pools.base_pool.AbstractPool` for the complete interface specification.

The following sections demonstrate how :class:`~quantammsim.pools.G3M.balancer.balancer.BalancerPool` implements these requirements.

Basic Structure
^^^^^^^^^^^^^^^

First, let's look at the class definition and initialization:

.. code-block:: python

    class BalancerPool(AbstractPool):
        def __init__(self):
            super().__init__()

        def calculate_weights(self, params):
            """Calculate fixed weights using softmax of initial logits."""
            return softmax(params["initial_weights_logits"])

Note the empty ``__init__`` method--pools do not hold any state, they only have methods.

Not all pools will have weights (though some hooks might require them) so the abstract class does not require this method.
Balancer pools, however, do have weights, so we need to implement this method for them.

Reserve Calculations
^^^^^^^^^^^^^^^^^^^^

The Balancer pool implements the three methods for reserve calculations:

1. Zero Fees Case
"""""""""""""""""

The zero fees implementation is the simplest and most performant:

.. code-block:: python

    def calculate_reserves_zero_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
            """
        Calculate reserves assuming zero fees and perfect arbitrage.

        Uses JAX-accelerated function _jax_calc_balancer_reserve_ratios for efficient
        computation in the theoretical zero-fee case. Simpler than TFMM implementation
        due to constant weights.

        Implementation Notes:
        ---------------------
        1. Uses dynamic_slice for price window
        2. Applies constant weights from calculate_weights
        3. Computes reserve ratios directly
        4. Uses cumprod for reserve calculation
        5. Handles no-arbitrage case via broadcasting

        Parameters
        ----------
        params : Dict[str, Any]
            Pool parameters containing initial_weights_logits
        run_fingerprint : Dict[str, Any]
            Simulation parameters
        prices : jnp.ndarray
            Price history array
        start_index : jnp.ndarray
            Starting index for the calculation window
        additional_oracle_input : Optional[jnp.ndarray]
            Not used in BalancerPool, kept for interface compatibility

        Returns
        -------
        jnp.ndarray
            Calculated reserves over time
        """
        
        # Get constant weights
        weights = self.calculate_weights(params)
        
        # Extract relevant price window
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        
        # Calculate initial reserves
        initial_value_per_token = weights * run_fingerprint["initial_pool_value"]
        initial_reserves = initial_value_per_token / local_prices[0]
        
        if run_fingerprint["do_arb"]:
            # Calculate reserve ratios using vectorized operation
            reserve_ratios = _jax_calc_balancer_reserve_ratios(
                local_prices[:-1], weights, local_prices[1:]
            )
            # Compute reserves through cumulative products
            reserves = jnp.vstack([
                initial_reserves,
                initial_reserves * jnp.cumprod(reserve_ratios, axis=0)
            ])
        else:
            reserves = jnp.broadcast_to(initial_reserves, local_prices.shape)
        
        return reserves

**Slicing the price window**

While it might be natural to consider passing in a price array that corresponds exactly the time period covered by the simulation, it can actually be neater for some use cases to pass in a price array that is longer than the simulation period, and then slice the price array to the relevant period within these functions.

This is particularly useful for pools that have dynamic properties that change over time, such as time-varying fees or dynamic weights, as these features very often will depend on earlier prices than those of the just the simulation period.

So in the ``calculate_reserves_zero_fees`` function, we see that we pass in a ``start_index`` parameter, which is used to slice the price array to the relevant period.
The length of the price array is given by ``bout_length``, which is a parameter of the ``run_fingerprint`` dictionary.

For a base Balancer pool with constant weights, however, we have no dynamic properties (the weights are constant, the fees are fixed at zero here).
This means that we could happily pass in a price array that is the length of the entire simulation, and then slice it to the relevant period within the ``calculate_reserves_zero_fees`` function.
But this is the structure required by the :class:`~quantammsim.pools.base_pool.AbstractPool` interface, and is the structure that enables time varying properties.

**Arbitrage control**


The ``run_fingerprint`` dictionary contains a ``do_arb`` parameter, which controls whether arbitrage is performed on the pool.
If arbitrage is not enabled, this function simply returns the initial reserves without any further calculation.
In practice, we would set ``do_arb`` to ``True``, as this is the only way to get a realistic simulation of the pool.
If one is performing a simulation, however, where a trade sequence is provided, it may be useful to set ``do_arb`` to ``False``, as this will allow one to see the effect of trades on the pool without the additional complexity of arbitrage.
See below the discussion of the ``calculate_reserves_with_dynamic_inputs`` function for more details.
The ``do_arb`` key is set to ``True`` by default.

**Understanding** :code:`_jax_calc_balancer_reserve_ratios`

Deriving the actual reserve calculations for a particular pool type can be a bit of a dark art.
For Balancer pools with fixed weights the core calculation of how reserves change in response to price movements is handled by ``_jax_calc_balancer_reserve_ratios``.

Here we will take a brief foray into the mathematics of the Balancer pool, and how give a gloss on where the logic in ``_jax_calc_balancer_reserve_ratios`` comes from.
Other pools will have different reserve calculations, but the general approach is the same: derive the reserve calculations from the pool's trading function by considering how arbitrageurs will act given pool state and external market prices.

The derivations tend to rely on two key ideas:

a. **Invariant Preservation**
    
For a Balancer pool containg :math:`N` assets, with weights :math:`w_1, w_2, ..., w_N`, (where :math:`w_i` sum to 1 and are in the range [0, 1]), and reserves :math:`R_1, R_2, ..., R_N`, the trading function is

.. math::

    k = \prod_i^N R_i^{w_i}

in the zero fees case. And the value :math:`k` of the trading function is invariant under allowed operations on the pool.

b. **Price Matching and Equilibrium**

After arbitrage, in the zero fees case, the pool's marginal prices exactly match the external market prices.
The pool's quoted price for a marginal trade of the :math:`i`\ :sup:`th` asset is proportional to  :math:`\frac{w_i}{R_i}`.
So we have that, after arbitrage,
.. math::

       \frac{\frac{w_i}{R_i}}{\frac{w_j}{R_j}} = \frac{p_i}{p_j},

where :math:`p_k` is the price of asset :math:`k` on the external market in a particular numeraire.

Combining these ideas, we can derive the reserve ratio formula for a Balancer pool with constant weights,

   .. math::

       \frac{R_i(t')}{R_i(t)} = \frac{p_i(t)}{p_i(t')} \prod_{j=1}^N \left(\frac{p_j(t')}{p_j(t)}\right)^{w_j}.

The full derivation is available in the `the Temporal Function Market Making litepaper`_, Appendix A1.

.. note::
   We have subtly used, under the hood, the result that geometric mean market maker pools hold *minimum value* when their quoted marginal prices are equal to the external market price.
   Proving *that* result is beyond the scope of this tutorial, but it is a well-known result in the AMM literature, and can be derived using the method of Lagrange multipliers.

.. note::
    For different pools and/or when handling the presence of fees and other time varying properties of pools (e.g. that arbitrageurs might have fixed costs and other constraints) the reserve derivations and resulting calculations will be different.
    The general approach is the same: derive the reserve calculations from the pool's trading function by considering how arbitrageurs will act given pool state and external market prices.

Now let's focus on the implementation, :func:`~quantammsim.pools.G3M.balancer.balancer_reserves._jax_calc_balancer_reserve_ratios`:

.. code-block:: python

    @jit
    def _jax_calc_balancer_reserve_ratios(prev_prices, weights, prices):
        """Calculate reserve ratio changes for constant-weight Balancer pools.
        
        Parameters
        ----------
        prev_prices : jnp.ndarray
            Previous asset prices
        weights : jnp.ndarray
            Pool weights (must sum to 1)
        prices : jnp.ndarray
            New asset prices
        
        Returns
        -------
        jnp.ndarray
            Ratio of new reserves to old reserves for each asset
        """
        # Calculate price ratios p'/p for each asset
        price_ratios = prices / prev_prices
        
        # Calculate the product term ∏(p'/p)^w
        price_product_ratio = jnp.prod(price_ratios**weights)
        
        # Calculate final reserve ratios
        reserve_ratios = price_product_ratio / price_ratios
        return reserve_ratios

This implementation is:
    - Fully vectorized for parallel computation, computing this for all assets and time steps simultaneously (as we have obtained the *ratio* between reserves at different times and the result only depends on the weights and the prices, not the prior reserves):
    - JIT-compiled for performance, via the :code:`@jit` decorator
    - Numerically stable through use of ratios rather than absolute values
    - Handles arbitrary numbers of assets

With no fees arbitrageurs will always trade to exactly match external market prices.
With fees, we need more complex calculations to account for the exact structure of the arbitrage trade, as well as other factors like gas costs, as we will see below.

2. With Fees Case
"""""""""""""""""
The implementation with fees requires more complex arbitrage calculations:

.. code-block:: python

    @partial(jit, static_argnums=2)
    def calculate_reserves_with_fees(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        weights = self.calculate_weights(params)
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]
        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        # calculate initial reserves
        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / local_prices[0]

        if run_fingerprint["do_arb"]:
            reserves = _jax_calc_balancer_reserves_with_fees_using_precalcs(
                initial_reserves,
                weights,
                arb_acted_upon_local_prices,
                fees=run_fingerprint["fees"],
                arb_thresh=run_fingerprint["gas_cost"],
                arb_fees=run_fingerprint["arb_fees"],
                all_sig_variations=jnp.array(run_fingerprint["all_sig_variations"]),
            )
        else:
            reserves = jnp.broadcast_to(
                initial_reserves, arb_acted_upon_local_prices.shape
            )

        return reserves

This implementation has a similar structure to the zero-fees case, but with the addition of the ``fees``, ``arb_thresh``, and ``arb_fees`` parameters.
These parameters are used to account for the exact structure of the arbitrage trade, as well as other factors like gas costs.
For a deep dive into this part of the codebase, see :func:`~quantammsim.pools.G3M.balancer.balancer_reserves._jax_calc_balancer_reserves_with_fees_using_precalcs`.
The underlying mathematics is provided in `this paper by the team on optimal arbitrage trades in G3Ms in the presence of fees`_.


3. Dynamic Inputs Case
""""""""""""""""""""""

For time-varying parameters:

.. code-block:: python

    @partial(jit, static_argnums=2)
    def calculate_reserves_with_dynamic_inputs(
        self,
        params: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        prices: jnp.ndarray,
        start_index: jnp.ndarray,
        fees_array: jnp.ndarray,
        arb_thresh_array: jnp.ndarray,
        arb_fees_array: jnp.ndarray,
        trade_array: jnp.ndarray,
        additional_oracle_input: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        bout_length = run_fingerprint["bout_length"]
        n_assets = run_fingerprint["n_assets"]

        local_prices = dynamic_slice(prices, start_index, (bout_length - 1, n_assets))
        weights = self.calculate_weights(params)

        if run_fingerprint["arb_frequency"] != 1:
            arb_acted_upon_local_prices = local_prices[
                :: run_fingerprint["arb_frequency"]
            ]
        else:
            arb_acted_upon_local_prices = local_prices

        initial_pool_value = run_fingerprint["initial_pool_value"]
        initial_value_per_token = weights * initial_pool_value
        initial_reserves = initial_value_per_token / arb_acted_upon_local_prices[0]

        # any of fees_array, arb_thresh_array, arb_fees_array, trade_array
        # can be singletons, in which case we repeat them for the length of the bout

        # Determine the maximum leading dimension
        max_len = bout_length - 1
        if run_fingerprint["arb_frequency"] != 1:
            max_len = max_len // run_fingerprint["arb_frequency"]
        # Broadcast input arrays to match the maximum leading dimension.
        # If they are singletons, this will just repeat them for the length of the bout.
        # If they are arrays of length bout_length, this will cause no change.
        fees_array_broadcast = jnp.broadcast_to(
            fees_array, (max_len,) + fees_array.shape[1:]
        )
        arb_thresh_array_broadcast = jnp.broadcast_to(
            arb_thresh_array, (max_len,) + arb_thresh_array.shape[1:]
        )
        arb_fees_array_broadcast = jnp.broadcast_to(
            arb_fees_array, (max_len,) + arb_fees_array.shape[1:]
        )
        # if we are doing trades, the trades array must be of the same length as the other arrays
        if run_fingerprint["do_trades"]:
            assert trade_array.shape[0] == max_len
        reserves = _jax_calc_balancer_reserves_with_dynamic_inputs(
            initial_reserves,
            weights,
            arb_acted_upon_local_prices,
            fees_array_broadcast,
            arb_thresh_array_broadcast,
            arb_fees_array_broadcast,
            jnp.array(run_fingerprint["all_sig_variations"]),
            trade_array,
            run_fingerprint["do_trades"],
            run_fingerprint["do_arb"],
        )
        return reserves

This method is more complex still, with the addition of the ``fees_array``, ``arb_thresh_array``, ``arb_fees_array``, and ``trade_array`` parameters.
The function :func:`~quantammsim.pools.G3M.balancer.balancer_reserves._jax_calc_balancer_reserves_with_dynamic_inputs` is doing the heavy lifting here.
It implements the same core logic as the fees case above, but also contains the logic for time varying fees, arbitrage thresholds, arbitrage fees, and so on, and enabling "exact out given in" trades to be done from the ``trade array`` input.

Helper Methods
~~~~~~~~~~~~~~

Finally, we implement required helper methods:

.. code-block:: python

    def init_base_parameters(
        self,
        initial_values_dict: Dict[str, Any],
        run_fingerprint: Dict[str, Any],
        n_assets: int,
        n_parameter_sets: int = 1,
        noise: str = "gaussian",
    ) -> Dict[str, Any]:
        np.random.seed(0)

        # We need to initialise the weights for each parameter set
        # If a vector is provided in the inital values dict, we use
        # that, if only a singleton array is provided we expand it
        # to n_assets and use that vlaue for all assets.
        def process_initial_values(
            initial_values_dict, key, n_assets, n_parameter_sets
        ):
            if key in initial_values_dict:
                initial_value = initial_values_dict[key]
                if isinstance(initial_value, (np.ndarray, jnp.ndarray, list)):
                    initial_value = np.array(initial_value)
                    if initial_value.size == n_assets:
                        return np.array([initial_value] * n_parameter_sets)
                    elif initial_value.size == 1:
                        return np.array([[initial_value] * n_assets] * n_parameter_sets)
                    elif initial_value.shape == (n_parameter_sets, n_assets):
                        return initial_value
                    else:
                        raise ValueError(
                            f"{key} must be a singleton or a vector of length n_assets"
                             +  "or a matrix of shape (n_parameter_sets, n_assets)"
                        )
                else:
                    return np.array([[initial_value] * n_assets] * n_parameter_sets)
            else:
                raise ValueError(f"initial_values_dict must contain {key}")

        initial_weights_logits = process_initial_values(
            initial_values_dict, "initial_weights_logits", n_assets, n_parameter_sets
        )
        params = {
            "initial_weights_logits": initial_weights_logits,
        }
        params = self.add_noise(params, noise, n_parameter_sets)
        return params
    
    def is_trainable(self):
        """Balancer pools have fixed weights and are not trainable."""
        return False

.. note::
    JAX enables very efficient vmapping over the parameters of a pool, and out the box this is enabled via the method :func:`~quantammsim.pools.base_pool.AbstractPool.make_vmap_in_axes` provided in the base class.
    If the pool has a particularly complex structure in its parameters, however, (e.g. dicts of dicts where different levels of the hierachy have different vmap axes, for example) it may be necessary to implement a custom method to enable vmapping over that pool's params dict.

.. note::
    After a pool class is created, it should be registered with JAX as a pytree.
    For the Balancer pool class, the call looks like this:

    .. code-block:: python

        jax.tree_util.register_pytree_node(
            BalancerPool, BalancerPool._tree_flatten, BalancerPool._tree_unflatten
        )

    This can be put directly below the class definition.
    The methods :func:`~quantammsim.pools.base_pool.AbstractPool._tree_flatten` and :func:`~quantammsim.pools.base_pool.AbstractPool._tree_unflatten` are provided in the base class.
    For custom pools that maintain internal state (breaking the standard design pattern for pool classes to be stateless) these methods would perhaps need to be overridden.

.. note::

    If you want to go further and access your pool via the frontend *and* if your pool has parameters that have both human-readable-but-contrained and hard-to-interpret-but-unconstrained representations, we recommend that you implement :func:`_process_specific_parameters` that takes the human-readable parameterisation and converts it into the underlying parameterisation.
    See the implementation of :func:`~quantammsim.pools.G3M.quantamm.power_channel_pool.PowerChannelPool._process_specific_parameters` for an example of this.
    See :ref:`constrained-vs-unconstrained` for more details on the use of constrained vs unconstrained parameters.

This implementation demonstrates how to create a pool type with:

* Efficient JAX-accelerated calculations
* Support for fees and arbitrage 
* Proper handling of dynamic parameters
* Clear separation of zero-fee and fee-based calculations

.. _the Temporal Function Market Making litepaper: https://cdn.prod.website-files.com/6616670ddddc931f1dd3aa73/6617c4c2381409947dc42c7a_TFMM_litepaper.pdf
.. _this paper by the team on optimal arbitrage trades in G3Ms in the presence of fees: https://arxiv.org/abs/2402.06731
