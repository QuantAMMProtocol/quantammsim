Core Concepts
=============

AMM Mechanisms
--------------

quantammsim implements four main AMM mechanisms:

1. Balancer Protocol
2. QuantAMM Protocol
3. CowAMM Protocol
4. Gyroscope Protocol

Weight Update Rules
-------------------

Many pool types have static/preset weights, e.g. Balancer pools and CowAMM pools.
For modelling QuantAMM pools, the library supports multiple different strategies (also known as weight update rules):

* Momentum
* Anti-Momentum
* Power Channel
* Mean Reversion Channel

Understanding Pool Parameters
-----------------------------

Memory Days and Lambda
~~~~~~~~~~~~~~~~~~~~~~

Almost all weight update rules for quantamm pools in quantammsim take as parameters a memory parameter (:math:`\lambda`) and a :math:`k` parameter.
(N.B. some pools include additional parameters.)
Given their common use, we will focus on these parameters here.
The memory parameter (:math:`\lambda`) controls how much historical data influences the pool's behaviour, as takes a value between 0 and 1:

* Short memory (low :math:`\lambda`): Responds quickly to data changes
* Long memory (high :math:`\lambda`): More stable, slower response

The relationship is defined by::

    λ = memory_days_to_lamb(memory_days, chunk_period)

While :math:`\lambda` is the more fundamental parameter and is closer to how these strategies are mathematically defined, it is often easier to think about the strategy's memory length.
There is a mapping also::

    memory_days = lamb_to_memory_days(λ, chunk_period)

Note that the chunk period is the frequency at which the oracle data is observed.

K Parameter
~~~~~~~~~~~

The :math:`k` parameter controls the magnitude of weight adjustments:

* Higher :math:`k`: More aggressive rebalancing
* Lower :math:`k`: More conservative approach

In the mathematics of TFMMs, the parameter :math:`k` is denoted :math:`\tilde{k}`.

Example::

    run_fingerprint = {
        'initial_k_per_day': 20,  # Moderate rebalancing
        'initial_memory_length': 10.0,  # 10-day memory
    }


Vector vs Scalar (Universal) Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the examples above show scalar (universal) parameters, both :math:`k` and :math:`\lambda` can be specified as vectors to give different values for each asset in the pool:

* Scalar (universal) parameters: Same value applies to all assets
* Vector parameters: Different values for each asset

This allows fine-tuning of how aggressively each asset's weight responds to market conditions. Common use cases include:

* Different rebalancing speeds for volatile vs stable assets
* Varying memory lengths based on asset liquidity
* Custom parameter sets for different market regimes

Note that when using vector parameters, the length must match the number of assets in the pool.