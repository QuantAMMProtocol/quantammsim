JAX Configuration
=================

quantammsim uses JAX for numerical computation. This guide explains how JAX is configured and how to optimize performance.

Default Configuration
---------------------

quantammsim automatically configures JAX at import time:

.. code-block:: python

    from jax import config
    config.update("jax_enable_x64", True)  # Enable 64-bit floats

This is required for numerical precision in financial calculations.

Backend Selection
-----------------

JAX automatically detects available hardware:

.. code-block:: python

    from jax import default_backend, devices

    # Check current backend
    print(default_backend())  # "cpu" or "gpu"

    # List available devices
    print(devices("cpu"))
    print(devices("gpu"))  # If GPU available

.. note::

    **TPU Support:** quantammsim does not currently support TPUs. The codebase assumes
    either CPU or GPU backends. If you're running on a TPU-equipped system, force
    the CPU or GPU backend using the environment variables described below.

Forcing CPU or GPU
~~~~~~~~~~~~~~~~~~

quantammsim detects the backend and configures accordingly:

.. code-block:: python

    # In quantammsim modules:
    DEFAULT_BACKEND = default_backend()

    if DEFAULT_BACKEND != "cpu":
        GPU_DEVICE = devices("gpu")[0]
        config.update("jax_platform_name", "gpu")
    else:
        GPU_DEVICE = devices("cpu")[0]
        config.update("jax_platform_name", "cpu")

To force a specific backend before importing quantammsim:

.. code-block:: python

    # Force CPU
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    # Then import
    import quantammsim

Or to force GPU:

.. code-block:: python

    os.environ["JAX_PLATFORM_NAME"] = "gpu"
    import quantammsim

Device Placement
----------------

Some computations are explicitly placed on CPU for efficiency:

.. code-block:: python

    from jax import device_put

    # Move data to CPU
    cpu_array = device_put(gpu_array, CPU_DEVICE)

This is used for operations where CPU is faster (e.g., scan operations with complex carry states).

Memory Management
-----------------

JAX pre-allocates GPU memory by default. To change this:

.. code-block:: python

    # Before importing JAX
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Or limit memory fraction
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Use 50% of GPU memory

Compilation and Caching
-----------------------

JIT Compilation
~~~~~~~~~~~~~~~

Most quantammsim functions are JIT-compiled:

.. code-block:: python

    from jax import jit

    @jit
    def my_function(x):
        return x * 2

The first call compiles the function; subsequent calls are fast.

Cache Directory
~~~~~~~~~~~~~~~

JAX caches compiled functions. To set a persistent cache:

.. code-block:: python

    import os
    os.environ["JAX_COMPILATION_CACHE_DIR"] = "/path/to/cache"

This speeds up startup when running the same code repeatedly.

Debugging
---------

Disable JIT for Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from jax import config
    config.update("jax_disable_jit", True)

This runs all operations eagerly, making debugging easier but much slower.

Enable Logging
~~~~~~~~~~~~~~

.. code-block:: python

    # Show XLA compilation logs
    os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_dump"

Full Traceback
~~~~~~~~~~~~~~

JAX simplifies tracebacks by default. For full tracebacks:

.. code-block:: python

    os.environ["JAX_TRACEBACK_FILTERING"] = "off"

Performance Tips
----------------

1. **Batch Operations**

   Use ``vmap`` for batched operations instead of Python loops:

   .. code-block:: python

       from jax import vmap

       # Instead of:
       results = [func(x) for x in batch]

       # Use:
       results = vmap(func)(batch)

2. **Avoid Python Control Flow**

   Use JAX primitives (``lax.cond``, ``lax.fori_loop``) instead of Python ``if``/``for``:

   .. code-block:: python

       from jax import lax

       # Instead of:
       for i in range(n):
           x = update(x)

       # Use:
       x = lax.fori_loop(0, n, lambda i, x: update(x), x)

3. **Static Arguments**

   Mark non-array arguments as static for better compilation:

   .. code-block:: python

       from functools import partial

       @partial(jit, static_argnums=(1,))
       def func(x, config_dict):
           ...

4. **Minimize Host-Device Transfers**

   Keep data on the accelerator; avoid frequent transfers to CPU.

Common Issues
-------------

NaN Values
~~~~~~~~~~

If you encounter NaN values, check:

- Division by zero in your calculations
- Log of negative numbers
- Overflow in exponentials

Enable NaN checking:

.. code-block:: python

    from jax import config
    config.update("jax_debug_nans", True)

Out of Memory
~~~~~~~~~~~~~

For large simulations:

1. Reduce batch size
2. Use gradient checkpointing
3. Process data in chunks
4. Limit GPU memory pre-allocation (see Memory Management above)

Slow Compilation
~~~~~~~~~~~~~~~~

First-time compilation can be slow. Solutions:

1. Use persistent compilation cache
2. Reduce function complexity
3. Use ``static_argnums`` for configuration dicts

Environment Variables Summary
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Description
   * - ``JAX_PLATFORM_NAME``
     - Force backend: "cpu", "gpu", "tpu"
   * - ``XLA_PYTHON_CLIENT_PREALLOCATE``
     - "false" to disable GPU memory pre-allocation
   * - ``XLA_PYTHON_CLIENT_MEM_FRACTION``
     - GPU memory fraction (0.0-1.0)
   * - ``JAX_COMPILATION_CACHE_DIR``
     - Path for persistent compilation cache
   * - ``JAX_TRACEBACK_FILTERING``
     - "off" for full tracebacks
   * - ``JAX_DISABLE_JIT``
     - "1" to disable JIT compilation
