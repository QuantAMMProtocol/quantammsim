Core Concepts
=============

AMM Mechanisms
--------------

quantammsim implements two main AMM mechanisms:

1. Balancer Protocol
2. QuantAMM Protocol
3. CowAMM Protocol

Weight Update Rules
-------------------

Many pool types have static/preset weights, e.g. Balancer pools and CowAMM pools.
For modelling QuantAMM pools, the library supports multiple different strategies (also known as weight update rules):

* Momentum
* Anti-Momentum
* Power Channel
* Mean Reversion Channel
