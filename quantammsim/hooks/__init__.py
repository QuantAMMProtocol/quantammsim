"""Hooks for modifying pool behavior.

Available hooks:
- BoundedWeightsHook: Per-asset min/max weight constraints
- EnsembleAveragingHook: Average rule outputs across parameter sets
- CalculateLossVersusRebalancing: LVR calculation
- CalculateRebalancingVersusRebalancing: RVR calculation
"""
from quantammsim.hooks.bounded_weights_hook import BoundedWeightsHook
from quantammsim.hooks.ensemble_averaging_hook import EnsembleAveragingHook
from quantammsim.hooks.versus_rebalancing import (
    CalculateLossVersusRebalancing,
    CalculateRebalancingVersusRebalancing,
)

__all__ = [
    "BoundedWeightsHook",
    "EnsembleAveragingHook",
    "CalculateLossVersusRebalancing",
    "CalculateRebalancingVersusRebalancing",
]
