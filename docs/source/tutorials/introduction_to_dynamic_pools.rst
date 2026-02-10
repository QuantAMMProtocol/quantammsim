Temporal Function Market Makers: Introduction to Dynamic AMMs
=============================================================

This tutorial explains the fundamental concepts of dynamic AMMs, Temporal Function Market Makers, which enable QuantAMM pools.

Core Concepts
-------------

Weight Vectors and Pool Composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geometric Mean Market Maker pool with :math:`N` tokens have the trading function :math:`k = \prod_{i=1}^N R_i^{w_i}` where :math:`R_i` are the reserves of token :math:`i`.
The weight vector :math:`\mathbf{w} = (w_1, \ldots, w_N)` determines the desired ratio of value held in each token.
In base Balancer pools the weight vector is constant over time.
(A Uniswap V2 pool is a special case of a Balancer pool where :math:`N=2` and the weights are equal to 1/2 for both tokens.)

For example, in a BTC/USDC pool:

- :math:`\mathbf{w} = (0.5, 0.5)` means the pool wants equal value in both tokens
- :math:`\mathbf{w} = (0.7, 0.3)` means the pool wants 70% of its value in BTC, 30% in USDC

.. note::
   Weights must always sum to 1.0 (to give them a clear interpretation as a percentage of the pool's value, see below), and for implementation reasons each weight must stay above a minimum threshold (Balancer V3 requires weights must be at least 1%, and QuantAMM pool creators can choose a higher minimum threshold, e.g. 3%).

From Weights to Asset Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The value of a pool is simply the sum of the values of all the assets it holds: :math:`V = \prod_{i=1}^N R_i p_i`, given prices :math:`p_i` in a chosen numeraire.
When a pool's actual token composition differs from its weight vector, it creates arbitrage opportunities. Here's how:

1. The pool's spot price for trading token i for token j is given by:

   .. math::

      P_{i,j} = \frac{w_i}{w_j} \cdot \frac{R_j}{R_i}

   where :math:`R_i` and :math:`R_j` are the token reserves.

2. So for asset :math:`i` prices increase as weights go up and as reserves go down.
3. This means that if the pool holds less of asset :math:`i` than the target weight, then the price at which the pool will buy that asset from traders will be higher than the market price. This will increase :math:`R_i`, decreasing the quoted price for that asset.
4. Conversely, if the pool holds more of asset :math:`i` than the target weight, then the price at which the pool will sell that asset to traders will be lower than the market price. This will decrease :math:`R_i`, increasing the quoted price for that asset.
5. These trades naturally move the pool's composition toward its target weights.

Example:
   If :math:`\mathbf{w} = (0.6, 0.4)` but the pool's actual holdings are equal values of tokens, arbitrageurs will:

   - sell token 1 to the pool, and
   - buy token 2 from the pool until the value ratio matches 60:40.

So by the action of arbs for a given asset :math:`R_i p_i = w_i V`: the value of the holdings of the :math:`i` :sup:`th` asset in the pool is equal to the weight of that asset times the total value of the pool.

We can interpret pools, even those with fixed weights, as pools that are always having their holdings adjusted to match the desired weights.
This is a form of *asset management*, but instead of "going out" to the market to buy and sell assets, the pool has its rebalancing done by arbitrageurs.

Dynamic Weight Updates
~~~~~~~~~~~~~~~~~~~~~~

So far we have described the operation of pools with constant weight vectors.
QuantAMM pools, Temporal Function Market Makers (TFMMs), enable pools to adjust their weight vectors over time.
The insight here is that as the weight vector sets the pool's desired composition, if you enable that to change over time you can create a pool whose target composition changes over time.
How the weight vector changes over time is the strategy of the pool.

We call the mathematical function that determines how the weight vector changes over time the *update rule*.
The path for a weight change is:

1. Pool observes market conditions (usually via price oracles)
2. Using store parameter values, the new oracle values, and some state (previous weights, running variables, etc), the pool calculates desired new weights in accordance to the pool's update rule.
3. Apply constraints (min weights, max change speed) on the weight change
4. Interpolate between old and new weights over a chosen period of time
5. Arbitrageurs trade to align pool holdings with new weights

This creates pools that can adapt to market conditions with controlled, predictable behavior.
The simulator models the entire process.

Step 5. can include application of fees (including dynamic fees), gas costs paid by arbitrageurs, and also having the pool perform particular trades provided as an input to the simulation by the user.

Update Rules
------------
The weight update process follows the general form:

.. math::
   \mathbf{w}_{t+1} = f(\mathbf{w}_t, \mathbf{p}_t, \theta)

where:

- :math:`\mathbf{w}_t` is the current weight vector
- :math:`\mathbf{p}_t` is a vector of oracle values (very commonly, prices)
- :math:`\theta` represents strategy parameters

Very often the update rule acts to add a weight change to the previous weights, with the weight *change* being a function of the current oracle values, some strategy parameters, and some running variables.

.. math::
   \mathbf{w}_{t+1} = \mathbf{w}_t + f(\mathbf{w}_t, \mathbf{p}_t, \theta)

For more details on the update rule, see :doc:`../tutorials/quantamm_pools` and the `TFMM litepaper <https://quantamm.fi/research>`_.

Weight Interpolation
~~~~~~~~~~~~~~~~~~~~

Rather than jumping directly to new weights, changes are typically interpolated over time to reduce the effective "slippage" paid by the pool to arbitrageurs.
The QuantAMM protocol itself implements linear interpolation, which is both simple to reason about and cheap to run.
See Appendix A.3. of the `TFMM litepaper <https://quantamm.fi/research>`_ for more details on the benefits of interpolation.

We include in the simulator a more advanced method based around approximations to the Lambert W function, which is more accurate but more expensive to run.
See this `paper <https://arxiv.org/abs/2403.18737>`_ for more details.

Guardrails
~~~~~~~~~~

Weigh changes have to respect some "guardrails".
The first is related to the weight interpolation: we allow pool creators to set a maximum weight change per block/per unit time.

The second is related to the weight range: pool creators can set a minimum weight that each token has to stay above.
For implementation reasons to do with the stability of the underlying math libraries in Balancer V3, this minimum weight has to be at least 1%.

There are reasons, however, why pool creators might want to set a higher minimum weight, and/or set a maximum weight change per block/per unit time.
On blockchains where neighbouring blocks might have the same block builder there can be opportunities for a multiblock MEV attack.
By setting a higher minimum weight and restricting the weight change per block, pool creators can make this potential attack uneconomical, for a given number of blocks under attack.

For more details on the potentical manipulation and on guardrails see `this paper <https://arxiv.org/abs/2404.15489>`_ and `TFMM litepaper <https://quantamm.fi/research>`_ Appendix C.

For chains with centralised, trusted block builders, these multiblock MEV attacks may be less of a concern.
The standard settings in the simulator are to set a minimum weight of 3% and a maximum weight change of 0.0003 per minute (which corresponds to cover for ~5 blocks on mainnet, see `this paper on multiblock MEV <https://arxiv.org/abs/2404.15489>`_).

Implementation
~~~~~~~~~~~~~~

The TFMM base class, :class:`~quantammsim.pools.TFMMBasePool` implements the logic needed for the application of guardrails and for weight change interpolation.
This means that QuantAMM pools can be created by simply subclassing :class:`~quantammsim.pools.TFMMBasePool` and implementing the update rule as the method :meth:`~quantammsim.pools.TFMMBasePool.calculate_rule_outputs` (plus a few helper methods for initialising/handling the particular parameters the pool's strategy needs).
