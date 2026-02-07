Pool Hooks
==========

Hooks are mixin classes that extend pool functionality by intercepting and modifying
pool behavior at key points. They enable features like custom fee logic, performance
tracking, and weight constraints without modifying the base pool implementations.

Using Hooks
-----------

Hooks are applied using the ``hookname__poolrule`` naming convention:

.. code-block:: python

    from quantammsim.pools.creator import create_pool

    # Create a momentum pool with LVR tracking
    pool = create_pool("lvr__momentum")

    # Create a balancer pool with rebalancing comparison
    pool = create_pool("rvr__balancer")

    # Create a momentum pool with per-asset weight bounds
    pool = create_pool("bounded__momentum")

Available Hooks
---------------

Loss Versus Rebalancing (LVR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``lvr`` hook tracks Loss-Versus-Rebalancing, measuring how much value the pool
loses to arbitrageurs compared to a continuously rebalancing portfolio.

.. code-block:: python

    pool = create_pool("lvr__mean_reversion_channel")

This is useful for:

* Measuring arbitrage extraction
* Comparing pool efficiency across strategies
* Understanding the cost of providing liquidity

Rebalancing Versus Rebalancing (RVR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``rvr`` hook compares pool performance against a periodically rebalancing
benchmark portfolio.

.. code-block:: python

    pool = create_pool("rvr__balancer")

This helps evaluate whether the pool's dynamic behavior adds value compared to
simple periodic rebalancing.

Bounded Weights
~~~~~~~~~~~~~~~

The ``bounded`` hook adds per-asset minimum and maximum weight constraints.
See :doc:`per_asset_bounds` for full documentation.

.. code-block:: python

    pool = create_pool("bounded__momentum")

    params = {
        "min_weights_per_asset": jnp.array([0.2, 0.2, 0.1]),
        "max_weights_per_asset": jnp.array([0.5, 0.5, 0.3]),
        # ... other parameters
    }

Ensemble Averaging
~~~~~~~~~~~~~~~~~~

The ``ensemble`` hook trains multiple parameter sets ("members") simultaneously
and averages their weight outputs, providing implicit regularisation through
diversity.  Members start from different initial parameters (using structured
sampling methods) and converge to different local optima.

.. code-block:: python

    pool = create_pool("ensemble__momentum")

    run_fingerprint["optimisation_settings"].update({
        "n_ensemble_members": 4,            # Number of members
        "ensemble_init_method": "lhs",      # Initialisation sampling method
        "ensemble_init_scale": 1.0,         # Perturbation scale
    })

Available initialisation methods:

* ``"lhs"`` — Latin Hypercube Sampling (default, good space coverage)
* ``"sobol"`` — Sobol quasi-random sequence (low discrepancy)
* ``"grid"`` — Regular grid (deterministic, evenly spaced)
* ``"gaussian"`` — Gaussian perturbations around initial values

The ensemble hook averages the *weight outputs* of all members, not the
parameters. This means each member produces its own weight trajectory, and the
final weights are the arithmetic mean.  Gradients flow through all members
during backpropagation.

Multi-Hook Chaining
~~~~~~~~~~~~~~~~~~~

Multiple hooks can be combined using the double-underscore syntax.  Hooks are
applied left-to-right (leftmost = highest priority in MRO):

.. code-block:: python

    # Ensemble + bounded weights + mean reversion channel
    pool = create_pool("ensemble__bounded__mean_reversion_channel")

    # Ensemble + LVR tracking + momentum
    pool = create_pool("ensemble__lvr__momentum")

This is equivalent to constructing the class manually:

.. code-block:: python

    from quantammsim.pools.creator import create_hooked_pool_instance
    from quantammsim.hooks.ensemble_averaging_hook import EnsembleAveragingHook
    from quantammsim.hooks.bounded_weights_hook import BoundedWeightsHook

    pool = create_hooked_pool_instance(
        MeanReversionChannelPool,
        BoundedWeightsHook,
        EnsembleAveragingHook,
    )

Dynamic Fee Hooks
~~~~~~~~~~~~~~~~~

Fee hooks allow pools to adjust their fees based on market conditions.

**Base Dynamic Fee Hook**

The ``dynamic_fee_base_hook`` provides infrastructure for volatility-adjusted fees:

.. code-block:: python

    from quantammsim.hooks.dynamic_fee_base_hook import DynamicFeeBaseHook

**Momentum Dynamic Fee Hook**

Adjusts fees based on recent price momentum:

.. code-block:: python

    from quantammsim.hooks.momentum_dynamic_fee_hook import MomentumDynamicFeeHook

Creating Custom Hooks
---------------------

Hooks are implemented as mixin classes that override specific pool methods.
The hook must appear before the base pool class in the inheritance order:

.. code-block:: python

    from quantammsim.pools.G3M.quantamm.momentum_pool import MomentumPool

    class MyCustomHook:
        """Custom hook that modifies pool behavior."""

        def some_pool_method(self, *args, **kwargs):
            # Custom logic before
            result = super().some_pool_method(*args, **kwargs)
            # Custom logic after
            return result

    # Create hooked pool class
    class HookedMomentumPool(MyCustomHook, MomentumPool):
        pass

    # Or use the creator utility
    from quantammsim.pools.creator import create_hooked_pool_instance

    pool = create_hooked_pool_instance(MomentumPool, MyCustomHook)

Key methods that can be overridden:

* ``calculate_fine_weights`` - Post-process calculated weights
* ``calculate_fees`` - Custom fee calculation
* ``post_trade_hook`` - Logic after each trade

Hook Ordering
-------------

When multiple hooks are applied, they are processed right-to-left in the
inheritance order. The first hook in the list has the highest priority
(its methods are called first).

.. code-block:: python

    # Hook1's methods take precedence over Hook2's
    class MultiHookedPool(Hook1, Hook2, BasePool):
        pass
