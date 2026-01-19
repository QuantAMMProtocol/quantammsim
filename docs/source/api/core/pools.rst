Pools
=====

.. currentmodule:: quantammsim.pools

Base Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AbstractPool
   TFMMBasePool

Constant Weight Pools
---------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BalancerPool
   CowPool
   HODLPool

QuantAMM Dynamic Weight Pools
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MomentumPool
   AntiMomentumPool
   PowerChannelPool
   MeanReversionChannelPool
   MinVariancePool
   DifferenceMomentumPool

Concentrated Pools
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GyroscopePool

Helper Functions
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   creator.create_pool
   creator.create_hooked_pool_instance