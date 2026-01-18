import jax.numpy as jnp
from jax.lax import stop_gradient
from quantammsim.pools.G3M.quantamm.weight_calculations.fine_weights import _jax_calc_coarse_weight_scan_function
jnp.set_printoptions(linewidth=float('inf'))
import numpy as np
min_weight = 0.1
midsize_weight_value = 0.2
n_assets = 4
eps=0.001
raw_in_weight = jnp.array(
    [
        1.0 - min_weight - eps - midsize_weight_value,
        midsize_weight_value,
        min_weight + eps,
        0.0,
    ]
)
out = _jax_calc_coarse_weight_scan_function(
    carry_list=[raw_in_weight],
    raw_weight_outputs=jnp.zeros(n_assets),
    minimum_weight=min_weight,
    asset_arange=jnp.arange(n_assets),
    n_assets=n_assets,
    alt_lamb=0.0,
    mvpt=False,
)[-1]


import numpy as np


def clamp_weights(weights, absolute_weight_guardrail):
    """
    Clamp weights to be within guardrail bounds while preserving relative proportions.

    Args:
        weights: Array of weights
        absolute_weight_guardrail: Minimum allowed weight

    Returns:
        Array of adjusted weights
    """
    weights = np.array(weights, dtype=float)

    # Early return for single weight
    if len(weights) == 1:
        return weights

    # Calculate bounds
    absolute_min = absolute_weight_guardrail
    absolute_max = 1.0 - ((len(weights) - 1) * absolute_weight_guardrail)

    # Calculate current stats
    w_min = np.min(weights)
    w_max = np.max(weights)
    aver = np.mean(weights)

    # If already within bounds, return unchanged
    if w_min >= absolute_min and w_max <= absolute_max:
        return weights

    # Calculate deltas from average
    d_min = aver - w_min
    d_max = w_max - aver

    # Calculate new target average and bounds
    new_aver = 1.0 / len(weights)
    n_min = new_aver - absolute_min
    n_max = absolute_max - new_aver

    # Calculate adjustment rate
    rate = float("inf")
    if d_min != 0:
        rate = n_min / d_min
    if d_max != 0:
        t_rate = n_max / d_max
        rate = min(rate, t_rate)

    # Apply adjustment
    return (weights - aver) * rate + new_aver


def _clampWeights_888(_weights, _absoluteWeightGuardRail):
    print("weights in _clampWeights: ", _weights)
    weightLength = len(_weights)
    if weightLength == 1:
        return _weights
    absoluteMin = _absoluteWeightGuardRail
    absoluteMax = 1.0 - (weightLength - 1) * _absoluteWeightGuardRail

    # Initialize tracking variables
    sumRemainerWeight = 1.0
    sumOtherWeights = 0.0
    isAtBound = [False] * weightLength

    # First pass - handle bounds and calculate sums
    for i in range(weightLength):
        if _weights[i] < absoluteMin:
            _weights[i] = absoluteMin
            sumRemainerWeight -= absoluteMin
            isAtBound[i] = True
        elif _weights[i] > absoluteMax:
            _weights[i] = absoluteMax
            sumOtherWeights += absoluteMax
            isAtBound[i] = True

    # Adjust non-bounded weights
    if sumOtherWeights != 0:
        proportionalRemainder = sumRemainerWeight / sumOtherWeights
        for i in range(weightLength):
            if not isAtBound[i]:
                _weights[i] = _weights[i] * proportionalRemainder

    print("final weights in _clampWeights: ", _weights, " sum: ", np.sum(_weights))
    return _weights


def _clampWeights_886(_weights, _absoluteWeightGuardRail):
    weightLength = len(_weights)
    if weightLength == 1:
        return _weights

    absoluteMin = _absoluteWeightGuardRail
    absoluteMax = 1.0 - (weightLength - 1) * _absoluteWeightGuardRail

    # Initialize tracking variables
    sumRemainerWeight = 1.0
    sumOtherWeights = 0.0

    # First pass - handle bounds and calculate sums
    for i in range(weightLength):
        if _weights[i] < absoluteMin:
            _weights[i] = absoluteMin
            sumRemainerWeight -= absoluteMin
        elif _weights[i] > absoluteMax:
            _weights[i] = absoluteMax
            sumRemainerWeight -= absoluteMax  # Subtract max-clamped weights
            sumOtherWeights += absoluteMax

    # Adjust non-min weights
    if sumOtherWeights != 0:
        proportionalRemainder = sumRemainerWeight / sumOtherWeights
        for i in range(weightLength):
            if _weights[i] != absoluteMin:
                _weights[i] = _weights[i] * proportionalRemainder

    return _weights


print("minimum allowed weight is:           ", min_weight)
print("intitial raw weight:                ", raw_in_weight)
print("post guardrail weight:              ", out)
print("number of values less than minimum:  ", jnp.sum(out < min_weight))
if jnp.sum(out < min_weight) > 0:
    print("weight(s) less than minimum: ", out[out < min_weight])

print(clamp_weights(raw_in_weight, min_weight))
print(sum(clamp_weights(raw_in_weight, min_weight)))
print(clamp_weights(out, min_weight))
print(sum(clamp_weights(out, min_weight)))

print("888")
min_weight = float(min_weight)
raw_in_weight = np.array([r for r in raw_in_weight])
out = np.array([o for o in out])
print(_clampWeights_888(raw_in_weight, min_weight))
print(sum(_clampWeights_888(raw_in_weight, min_weight)))
# print(_clampWeights_888(out, min_weight))
# print(sum(_clampWeights_888(out, min_weight)))

print("886")
print(_clampWeights_886(raw_in_weight, min_weight))
print(sum(_clampWeights_886(raw_in_weight, min_weight)))
# print(_clampWeights_886(out, min_weight))
# print(sum(_clampWeights_886(out, min_weight)))


raw_in_weight = np.array([0.5,0.4,0.1])
min_weight = 0.2
print("888")
min_weight = float(min_weight)
raw_in_weight = np.array([r for r in raw_in_weight])
out = np.array([o for o in out])
print(_clampWeights_888(raw_in_weight, min_weight))
print(sum(_clampWeights_888(raw_in_weight, min_weight)))
# print(_clampWeights_888(out, min_weight))
# print(sum(_clampWeights_888(out, min_weight)))

print("886")
print(_clampWeights_886(raw_in_weight, min_weight))
print(sum(_clampWeights_886(raw_in_weight, min_weight)))
