QuantAMM: Pools as Portfolios
=============================

QuantAMM pools implement Temporal Function Market Making (TFMM) - a system where pool behavior is determined by dynamic weight vectors that evolve based on market conditions.
These changes in pool behavior lead to changes in the pool's composition.

If you're new to QuantAMM pools, you should start with :doc:`../tutorials/introduction_to_dynamic_pools`, which describes the overall structure of how weight changes are done in Temporal Function Market Making and in this simulator.

We provide here in the simulator the same strategies that are implemented in V1 of the QuantAMM protocol.
Users can also experiment with custom strategies in this simulator, but support for custom strategies in the onchain protocol is not yet fully implemented.
If you are interested in deploying a custom strategy as a smart contract, please get in touch.
See :doc:`../tutorials/advanced_usage` for more detail on how to implement a custom strategy in the simulator.

This page provides an overview of the actual strategies we have implemented in the simulator.


.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: none

Pre-implemented Strategies
--------------------------

Strategy Implementations:

* **Momentum Pool**: Adjusts weights based in alignment with recent price trends
* **Anti-Momentum Pool**: Adjusts weights counter to recent price movements
* **Difference Momentum Pool**: Implements a Moving Average Convergence Divergence (MACD) strategy
* **Power Channel Pool**: Uses non-linear power functions of price changes
* **Mean Reversion Channel Pool**: Combines mean reversion for small moves with (power channel-style)trend-following for large moves
* **Min Variance Pool**: Uses Mean Variance Portfolio Theory to calculate a simple minimum variance portfolio

All of these strategies use pre-defined *estimators* to extract signals from price/oracle data.
There are three core estimators:

* ``calc_gradients``: Calculates the gradient of a price/oracle value with respect to time
* ``calc_covariances``: Calculates the covariance of a price/oracle value with respect to time
* ``calc_ewma``: Calculates the exponentially-weighted moving average of a price/oracle value with respect to time

.. note::
   For the rest of this page, we will use the term "price" to refer to any oracle value, as all these estimators are used to calculate price-related signals, but non-price oracles can be used as well.

All these estimators use varieties of exponential smoothing.
They have a parameter :math:`\lambda` that controls the memory length of the underlying calculation, which takes values from 0 to 1.
The exact mapping between :math:`\lambda` and the memory length varies from estimator to estimator, but in general :math:`\lambda` is proportional to the memory length.

For a pool in the QuantAMM protocol (and in the hosted frontend for the simulator at `app.quantamm.fi <https://app.quantamm.fi>`_) :math:`\lambda` can be set per-asset or the pool creator can choose a single "universal" value.
Here in the backend, :math:`\lambda` is always per-asset, though of course it can contain repeated values giving it identical behaviour to a "universal" setting.

The estimators implement the same mathematics that is described in `the fronend simulator documentation <https://app.quantamm.fi/documentation>`_ (see pages "Estimating Gradients" and "Estimating Covariances").
They are written in JAX code that is optimized for performance.
So for running on CPU they run using loops and their implementation is straightforward, clearly matching line-for-line the mathematics.
For GPU acceleration, however, we perform different-looking but actually-equivalent calculations using convolution operations, thus avoiding having to run loops on the GPU.
The estimators are carefully written to avoid look-ahead bias.

.. _constrained-vs-unconstrained:

Constrained vs unconstrained parameters
"""""""""""""""""""""""""""""""""""""""

The behaviour of an update rule is controlled by the numerical values of its parameters.
One of the key purposes of this simulator is to allow users to optimise these parameters
But a given strategy can be parameterised in many different ways.
In the pre-implemented strategies we often parameterise a strategy not in terms that are most human readable but in terms that are most convenient for optimisation.
Why do this? Generally we choose underlying parameterisations that enable us to perform unconstrained optimisation.
For exaple, often we do not directly work with the memory length of a strategy, nor even its :math:`\lambda` parameter, but instead a logit representation of :math:`\lambda`.
:math:`\lambda` = :math:`\text{sigmoid}(\mathrm{logit\_lambda})`.
:math:`\mathrm{logit\_lambda}` is a free parameter that can take any value, and the sigmoid function ensures that :math:`\lambda` is always between 0 and 1 by constuction.

This means that we sometimes have to interconvert between the underlying parameterisation and the human-readable parameterisation.
For example, we have a method :func:`quantammsim.pools.TFMMBasePool.process_parameters` that takes the human-readable parameterisation (which is what is displayed in the simulator frontend) and converts it into the underlying parameterisation.

Momentum Pool
"""""""""""""
The momentum strategy adjusts weights based on recent price trends.

**Mathematical Model**

The update rule is:

.. math::

   \mathbf{w}(t) = \mathbf{w}(t-1) + \kappa \cdot \left(\frac{1}{\overline{\mathbf{p}}(t)}\frac{\partial \mathbf{p}(t)}{\partial t} - \ell_{\mathbf{p}(t)}\right)

where :math:`\ell_{\mathbf{p}(t)} = \frac{1}{N}\sum_{i=1}^N \left(\frac{1}{\overline{\mathbf{p}}(t)}\right)_i \left(\frac{\partial \mathbf{p}(t)}{\partial t}\right)_i` if we are using a "universal" value of :math:`\kappa` (it is a scalar).
If :math:`\kappa` is provided per-asset (it is a vector) then :math:`\ell_{\mathbf{p}(t)} = \frac{\sum_{i=1}^N \left(\kappa_i \frac{1}{\overline{\mathbf{p}}(t)}\right)_i \left(\frac{\partial \mathbf{p}(t)}{\partial t}\right)_i}{\sum_{i=1}^N \kappa_i}`.

:math:`\kappa` tunes the aggressiveness of the strategy.
For a given change in prices over a given time period, the larger :math:`\kappa` is, the larger the change in weights will be.

This rule is implemented in the function :func:`quantammsim.pools.MomentumPool._jax_momentum_weight_update`.
It uses the ``calc_gradients`` estimator, where the proportional gradient :math:`\frac{1}{p} \frac{\partial p}{\partial t}` is calculated.
The value of :math:`\lambda` tunes the memory of the strategy with respect to these price trends.

As a function of :math:`\lambda`, the memory length (in days) is given by :math:`2\cdot\sqrt[3]{\frac{6 \lambda}{(1 - \lambda)^3}} \cdot \frac{\mathrm{interpolation\_period}}{1440}` where :math:`\mathrm{interpolation\_period}` is the time between price samples and the time the pool takes to interpolate to new calculated values (in minutes).

The overall pool is implemented in :class:`quantammsim.pools.MomentumPool`.

.. note::
   The protocol enables pool creators to choose whether the scaling factor ":math:`\frac{1}{p}`" in ":math:`\frac{1}{p} \frac{\partial p}{\partial t}`" uses the current price or the smoothed EWMA prices that is calculated as part of the gradient calculation.
   Empirically, the EWMA version tends to perform better and is the default in the simulator.
   In the simulator this is controlled by the ``use_alt_lamb`` key in the run fingerprint, where a value of ``True`` enables a EWMA with different :math:`\lambda` to be used in the scaling factor, and setting this alternative :math:`\lambda` to zero gives the same behaviour as the non-EWMA version.

.. note::
   There is a sublety in how :math:`\kappa` is handled, which applies to all pool with a :math:`\kappa` parameter.
   Naturally as the memory length increases, the value of :math:`\kappa` should increase.
   The intuition for this is that the memory length gives a "bounding box" within which the price signal is analysed.
   For a price that remains within a range, the longer the "bounding box" the smaller the caculated gradient will be.
   The maximum possible gradient would be something like the maxmimum varation possible divided by the memory length.
   This means that it is natural to parameterise :math:`\kappa` in terms of :math:`\kappa` *per day*, and multiply by the memory length in days to get the actual value of :math:`\kappa`.
   This is what the simulator does, and all pools with a :math:`\kappa` parameter use this convention.

.. note::
   Other than the minimum variance pool, all the QuantAMM pool classes inherit from :class:`quantammsim.pools.MomentumPool` as the overall structure is the same.
   If you are implementing a custom strategy that uses the same structure, it might make sense to inherit from :class:`quantammsim.pools.MomentumPool` as well.

Anti-Momentum Pool
""""""""""""""""""

Implements a contrarian strategy that moves weights against recent price trends. Uses the same parameters as Momentum Pool but responds in the opposite direction.

For implementation, see :class:`quantammsim.pools.AntiMomentumPool`.

Power Channel Pool
""""""""""""""""""

A sophisticated strategy that applies non-linear transformations to price signals, enabling customized responses to market movements of different magnitudes.

**Mathematical Model**

The strategy transforms price signals through a power law function while preserving sign:

.. math::

   \mathbf{w}(t) = \mathbf{w}(t-1) + \kappa \cdot \left(\text{sign}(s) \cdot \left|s\right|^p - \ell\right)

where:
- :math:`s` is the price gradient signal (as in Momentum Pool)
- :math:`p` is the power parameter (> 1)
- :math:`\ell` is an auto-calculated offset ensuring zero-sum updates (as in Momentum Pool)
- :math:`\kappa` scales the overall response magnitude (as in Momentum Pool)

**Parameters**

Same as Momentum Pool, but with an additional ``power`` parameter.

* ``power`` (per-asset): Controls response curve shape

    - ``power > 1``: Amplifies large moves, dampens small ones


**Implementation Notes**

* Inherits core infrastructure from MomentumPool

For implementation, see :class:`quantammsim.pools.PowerChannelPool`.


Mean Reversion Channel Pool
"""""""""""""""""""""""""""
Combines mean reversion for small moves with (power channel) trend-following for large moves:

* For price changes within the channel: Acts like Anti-Momentum
* For price changes outside the channel: Acts like Power Channel

**Mathematical Model**

The strategy implements a smooth transition between mean reversion and trend following using a Gaussian envelope:

.. math::

   \mathbf{w}(t) = \mathbf{w}(t-1) + \kappa f(s)

where :math:`f(s)` combines channel and trend components:

.. math::

   f(s) = E(s) f_\text{channel}(s) + (1-E(s)) f_\text{trend}(s) - \ell

with:

.. math::

   E(s) &= \exp\left(-\frac{s^2}{2w^2}\right) \\
   f_\text{channel}(s) &= -A \cdot \left(\frac{\pi s}{3w} - \frac{1}{6}\left(\frac{\pi s}{3w}\right)^3\right) \\
   f_\text{trend}(s) &= \text{sign}(s) \cdot \left|\frac{s}{2\alpha}\right|^p

where:

- :math:`s` is the price gradient signal (as in Momentum Pool)
- :math:`w` is the channel width
- :math:`A` is the amplitude (scales with memory length)
- :math:`p` is the power parameter for trend following
- :math:`\alpha` is the pre-exponential scaling
- :math:`\ell` ensures zero-sum updates (as in Momentum Pool)
- :math:`\kappa` scales the overall response magnitude (as in Momentum Pool)

For implementation, see :class:`quantammsim.pools.MeanReversionChannelPool`.


Difference Momentum Pool
""""""""""""""""""""""""
A MACD-like strategy that uses the difference between two exponential moving averages to generate trading signals.

**Mathematical Model**

The strategy compares two moving averages with different memory lengths:

.. math::

   \mathbf{w}(t) = \mathbf{w}(t-1) + \kappa \cdot \left(1 - \frac{E_2(\mathbf{p}(t))}{E_1(\mathbf{p}(t))} - \ell\right)

where:

- :math:`E_1` is EWMA with memory length :math:`m_1`
- :math:`E_2` is EWMA with memory length :math:`m_2` (typically :math:`m_2 > m_1`)
- :math:`\kappa` scales with :math:`\max(m_1, m_2)` for consistent behavior (as in Momentum Pool)

This formulation ensures the signal is scale-invariant to price levels and produces proportional responses.

**Parameters**

*Core Parameters*

* ``logit_lamb``: Base lambda (controls short-term memory)

    - Transformed from memory_days_1 using logit function
    - Controls base EWMA calculation
    - Closer to 1 = longer memory

* ``logit_delta_lamb``: Lambda difference

    - Determines spread between short and long EWMAs
    - Added to base lambda for second EWMA
    - Controls signal generation sensitivity

* ``log_k``: Signal scaling (in log2 space)

    - Automatically scales with memory length
    - Controls update magnitude
    - Applied after EWMA difference calculation

*Human-Readable Equivalents*

* ``memory_days_1``: Short-term period (converted to logit_lamb)
* ``memory_days_2``: Long-term period (determines logit_delta_lamb)
* ``memory_length_delta``: Alternative way to specify logit_delta_lamb

**Implementation Notes**

* Inherits from MomentumPool
* Uses JAX-accelerated EWMA calculations
* Automatically maintains zero-sum updates
* Memory lengths affect both signal calculation and k_factor scaling

For implementation, see :class:`quantammsim.pools.DifferenceMomentumPool`.

Min Variance Pool
"""""""""""""""""

Implements a minimum variance portfolio strategy that aims to minimize the overall portfolio volatility. Unlike other strategies that output weight changes, this strategy directly outputs optimal portfolio weights.

**Mathematical Model**

The strategy calculates weights based on asset return variances:

.. math::

   \mathbf{w}(t+1) = \Lambda\mathbf{w}(t) + (1-\Lambda)\mathbf{w}_\text{target}(t)

where:

- :math:`\mathbf{w}_\text{target}(t)` is the estimated min-variance portfolio weights

The variance for each asset is calculated using an exponentially weighted moving average (EWMA) estimator, over returns. This estimator uses :math:`\lambda`, which may take a different value to the weight-smoothing parameter :math:`\Lambda`.

**Parameters**

Two memory lenghts, analogous to Difference Momentum Pool.

**Implementation Notes**

* Inherits from TFMMBasePool directly (not MomentumPool)
* Update rule outputs *weights* directly rather than *weight changes*
* Uses JAX-accelerated variance calculations

For implementation, see :class:`quantammsim.pools.MinVariancePool`.