Per-Asset Weight Bounds
=======================

In addition to the uniform ``minimum_weight`` guardrail that applies equally to all assets,
quantammsim supports per-asset minimum and maximum weight constraints. This allows you to
specify different allocation limits for each asset in the pool.

Use Cases
---------

Per-asset bounds are useful when you want to:

* Ensure a stablecoin maintains a minimum allocation (e.g., USDC always >= 10%)
* Cap exposure to volatile assets (e.g., meme coins <= 30%)
* Implement risk management constraints specific to each asset
* Model regulatory or mandate-driven allocation limits

Usage
-----

To use per-asset bounds, create a pool with the ``bounded__`` prefix::

    from quantammsim.pools.creator import create_pool

    # Create a bounded momentum pool
    pool = create_pool("bounded__momentum")

    # Also works with other strategies
    pool = create_pool("bounded__mean_reversion_channel")

Then specify the bounds in your parameters::

    params = {
        # ... other strategy parameters ...
        "min_weights_per_asset": jnp.array([0.25, 0.20, 0.10]),  # BTC, ETH, USDC
        "max_weights_per_asset": jnp.array([0.60, 0.55, 0.30]),
    }

Algorithm
---------

The per-asset bounds use a **clip-and-redistribute** algorithm:

**Step 1: Initial Clip**

.. math::

    w'_i = \text{clip}(w_i, \min_i, \max_i)

**Step 2: Calculate Slack**

.. math::

    \text{slack\_up}_i = \max_i - w'_i \quad \text{(room to grow)}

    \text{slack\_down}_i = w'_i - \min_i \quad \text{(room to shrink)}

**Step 3: Redistribute Proportionally**

If :math:`\sum w'_i < 1` (deficit):

.. math::

    \text{adjustment}_i = \frac{(1 - \sum w'_j) \cdot \text{slack\_up}_i}{\sum \text{slack\_up}_j}

If :math:`\sum w'_i > 1` (surplus):

.. math::

    \text{adjustment}_i = -\frac{(\sum w'_j - 1) \cdot \text{slack\_down}_i}{\sum \text{slack\_down}_j}

**Step 4: Final Normalisation**

.. math::

    w''_i = \text{clip}(w'_i + \text{adjustment}_i, \min_i, \max_i)

    w_{\text{final},i} = \frac{w''_i}{\sum w''_j}

Restrictions on Bounds
----------------------

For a feasible solution to exist, the bounds must satisfy these constraints:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Constraint
     - Formula
     - Meaning
   * - Sum of minimums
     - :math:`\sum \min_i \leq 1`
     - Must be possible to satisfy all minimums
   * - Sum of maximums
     - :math:`\sum \max_i \geq 1`
     - Must be possible to reach total weight of 1
   * - Per-asset ordering
     - :math:`\min_i < \max_i \; \forall i`
     - Each asset must have a valid range
   * - Non-negative
     - :math:`\min_i \geq 0 \; \forall i`
     - Weights cannot be negative
   * - Upper bound
     - :math:`\max_i \leq 1 \; \forall i`
     - No single asset can exceed 100%

**Example: Valid vs Invalid Bounds**

For a 3-asset pool:

* **Valid**: ``min = [0.2, 0.2, 0.2]``, ``max = [0.5, 0.5, 0.5]``

  - :math:`\sum \min = 0.6 \leq 1` ✓
  - :math:`\sum \max = 1.5 \geq 1` ✓

* **Invalid**: ``min = [0.4, 0.4, 0.4]``

  - :math:`\sum \min = 1.2 > 1` ✗ (impossible to satisfy all minimums)

* **Invalid**: ``max = [0.3, 0.3, 0.3]``

  - :math:`\sum \max = 0.9 < 1` ✗ (impossible to reach total weight of 1)

Key Properties
--------------

1. **Guaranteed feasibility**: If the bounds satisfy the constraints above, a valid weight vector always exists.

2. **Proportional redistribution**: Slack is redistributed proportionally, so assets with more room to adjust absorb more of the deficit/surplus.

3. **Preserves relative ordering**: Assets closer to their bounds move less than those with more slack.

4. **Layered on existing guardrails**: Per-asset bounds are applied BEFORE the uniform ``minimum_weight`` guardrail, so both constraints must be satisfied.

Integration with Run Fingerprints
---------------------------------

When using per-asset bounds in a simulation, specify the rule with the ``bounded__`` prefix::

    run_fingerprint = {
        "tokens": ["BTC", "ETH", "USDC"],
        "rule": "bounded__momentum",
        "minimum_weight": 0.05,  # Uniform guardrail (still applies)
        # ... other settings ...
    }

    params = {
        "min_weights_per_asset": jnp.array([0.25, 0.20, 0.10]),
        "max_weights_per_asset": jnp.array([0.60, 0.55, 0.30]),
        # ... other strategy parameters ...
    }

    result = do_run_on_historic_data(run_fingerprint, params)
