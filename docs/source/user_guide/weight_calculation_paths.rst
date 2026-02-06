Weight Calculation Paths
========================

This guide explains the two weight calculation paths available in quantammsim and when to use each.

Overview
--------

QuantAMM pools compute portfolio weights from price data using estimators (e.g., EWMA for momentum detection). There are two computational approaches:

1. **Vectorized Path** - Computes all weights at once using convolution operations
2. **Scan Path** - Computes weights sequentially, one time step at a time

Both paths produce numerically equivalent results but have different performance characteristics.

Vectorized Path
---------------

The vectorized path uses JAX's convolution operations to compute estimator outputs (like EWMA values) for all time steps simultaneously. This is the traditional approach and is typically faster for simulation and training.

**How it works:**

1. Compute all estimator outputs at once using ``calculate_rule_outputs()``
2. Apply guardrails and interpolation using ``calculate_fine_weights()``
3. Slice to the relevant bout period

**Advantages:**

- Faster execution (typically 1.5-2x faster than scan)
- Better GPU utilization through parallelization
- Well-optimized by XLA compiler

**Disadvantages:**

- Higher memory usage (stores all intermediate values)
- Computation flow differs from production execution

Scan Path
---------

The scan path processes prices sequentially using ``jax.lax.scan`` and ``jax.lax.fori_loop``, updating estimator state one step at a time. This mirrors how weights are computed in production (on-chain).

**How it works:**

1. Initialize estimator state from first price
2. Warm up estimator over burn-in period (using ``fori_loop``)
3. Compute fine weights for bout period (using ``scan``)

**Advantages:**

- Matches production/on-chain execution exactly
- Lower memory footprint (only stores current state)
- Useful for verifying production behavior

**Disadvantages:**

- Slower execution due to sequential processing
- ``fori_loop`` has more overhead than vectorized operations

Selecting a Path
----------------

Use the ``weight_calculation_method`` parameter in your run fingerprint:

.. code-block:: python

    # Automatic selection (default) - uses vectorized if available
    run_fingerprint["weight_calculation_method"] = "auto"

    # Force vectorized path
    run_fingerprint["weight_calculation_method"] = "vectorized"

    # Force scan path (matches production)
    run_fingerprint["weight_calculation_method"] = "scan"

Pool Support
------------

Most QuantAMM pools support both paths:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Pool
     - Vectorized
     - Scan
   * - MomentumPool
     - Yes
     - Yes
   * - AntiMomentumPool
     - Yes
     - Yes
   * - PowerChannelPool
     - Yes
     - Yes
   * - MeanReversionChannelPool
     - Yes
     - Yes
   * - DifferenceMomentumPool
     - Yes
     - Yes
   * - MinVariancePool
     - Yes
     - No
   * - IndexMarketCapPool
     - Yes
     - No

You can check pool support programmatically:

.. code-block:: python

    from quantammsim.pools import MomentumPool

    pool = MomentumPool()
    print(pool.supports_vectorized_path())  # True
    print(pool.supports_scan_path())        # True

Numerical Equivalence
---------------------

For pools that support both paths, results are numerically equivalent within floating-point tolerance:

.. code-block:: python

    import numpy as np
    from quantammsim.runners.jax_runners import do_run_on_historic_data

    fingerprint = {
        "rule": "momentum",
        "tokens": ["BTC", "ETH"],
        # ... other settings ...
    }
    params = {
        "log_k": jnp.array([3.0, 3.0]),
        "logit_lamb": jnp.array([0.0, 0.0]),
        "initial_weights_logits": jnp.array([0.0, 0.0]),
    }

    # Run with vectorized path
    fingerprint["weight_calculation_method"] = "vectorized"
    result_vec = do_run_on_historic_data(fingerprint, params)

    # Run with scan path
    fingerprint["weight_calculation_method"] = "scan"
    result_scan = do_run_on_historic_data(fingerprint, params)

    # Results match
    np.testing.assert_allclose(
        result_vec["final_value"],
        result_scan["final_value"],
        rtol=1e-4
    )

Performance Comparison
----------------------

Typical performance ratios (scan time / vectorized time):

- **Daily chunks (1440 min)**: ~1.5x slower
- **Hourly chunks (60 min)**: ~2x slower
- **More assets**: Ratio increases slightly

Run the performance tests to measure on your hardware:

.. code-block:: bash

    pytest tests/performance/test_weight_calculation_timing.py -v -s

Implementation Details
----------------------

The scan path implementation uses:

- ``get_initial_rule_state()`` - Initialize estimator carry state
- ``calculate_rule_output_step()`` - Single-step estimator update
- ``get_initial_guardrail_state()`` - Initialize weight carry state
- ``calculate_coarse_weight_step()`` - Single-step guardrailed weight
- ``calculate_fine_weights_step()`` - Single-step interpolation block

The burn-in warm-up uses ``jax.lax.fori_loop`` which supports dynamic (traced) loop bounds, allowing the warm-up length to vary based on ``start_index``.

Implementing New Pools
----------------------

When implementing a new pool, you can choose to implement:

1. **Vectorized only** - Implement ``calculate_rule_outputs()``
2. **Scan only** - Implement ``calculate_rule_output_step()`` and ``get_initial_rule_state()``
3. **Both** - Implement all methods for maximum flexibility

The base class provides capability detection:

.. code-block:: python

    class MyPool(TFMMBasePool):
        def calculate_rule_outputs(self, params, run_fingerprint, prices, ...):
            # Vectorized implementation
            ...

        def calculate_rule_output_step(self, carry, price, params, run_fingerprint):
            # Single-step implementation
            ...

        def get_initial_rule_state(self, initial_price, params, run_fingerprint):
            # Initial carry state
            ...

    pool = MyPool()
    pool.supports_vectorized_path()  # True (has calculate_rule_outputs)
    pool.supports_scan_path()        # True (has calculate_rule_output_step)
