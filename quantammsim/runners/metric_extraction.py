"""
Metric Extraction: Registry-based lookup for cycle evaluation metrics.

This module provides unified metric extraction from CycleEvaluation objects,
replacing repetitive if/elif chains with a registry-based approach.

Usage:
------
```python
from quantammsim.runners.metric_extraction import extract_cycle_metric

# Extract aggregated metrics from cycle evaluations
value = extract_cycle_metric(cycle_evals, "mean_oos_sharpe")
value = extract_cycle_metric(cycle_evals, "worst_wfe")
value = extract_cycle_metric(cycle_evals, "neg_is_oos_gap")
```
"""

from typing import List, Dict, Callable, Any
import numpy as np


# Map metric names to CycleEvaluation attributes
CYCLE_METRICS: Dict[str, str] = {
    "oos_sharpe": "oos_sharpe",
    "is_sharpe": "is_sharpe",
    "wfe": "walk_forward_efficiency",
    "is_oos_gap": "is_oos_gap",
    "adjusted_oos_sharpe": "adjusted_oos_sharpe",
    # Risk metrics
    "oos_calmar": "oos_calmar",
    "is_calmar": "is_calmar",
    "oos_sterling": "oos_sterling",
    "is_sterling": "is_sterling",
    "oos_ulcer": "oos_ulcer",
    "is_ulcer": "is_ulcer",
    "oos_returns": "oos_returns",
    "is_returns": "is_returns",
    "oos_returns_over_hodl": "oos_returns_over_hodl",
    "is_returns_over_hodl": "is_returns_over_hodl",
}

# Aggregation functions
AGGREGATORS: Dict[str, Callable[[List[float]], float]] = {
    "mean": lambda v: np.mean([x for x in v if x is not None and np.isfinite(x)]) if v else float("-inf"),
    "worst": lambda v: np.min([x for x in v if x is not None and np.isfinite(x)]) if v else float("-inf"),
}


def extract_cycle_metric(cycle_evals: List[Any], metric_spec: str) -> float:
    """
    Extract aggregated metric from CycleEvaluation list.

    Supports metric specifications like:
    - "mean_oos_sharpe": mean of oos_sharpe across cycles
    - "worst_wfe": minimum walk_forward_efficiency
    - "neg_is_oos_gap": negated mean of is_oos_gap (for minimization)
    - "adjusted_mean_oos_sharpe": mean of adjusted_oos_sharpe

    Parameters
    ----------
    cycle_evals : List[CycleEvaluation]
        List of cycle evaluation results
    metric_spec : str
        Metric specification string

    Returns
    -------
    float
        Aggregated metric value

    Examples
    --------
    >>> value = extract_cycle_metric(cycle_evals, "mean_oos_sharpe")
    >>> value = extract_cycle_metric(cycle_evals, "worst_wfe")
    >>> value = extract_cycle_metric(cycle_evals, "neg_is_oos_gap")
    """
    if not cycle_evals:
        return float("-inf")

    # Parse "neg_mean_oos_sharpe" -> negate=True, remaining="mean_oos_sharpe"
    negate = metric_spec.startswith("neg_")
    if negate:
        metric_spec = metric_spec[4:]

    # Handle special case: "adjusted_mean_oos_sharpe" -> mean aggregation of adjusted_oos_sharpe
    # This is a common metric name that doesn't follow the standard agg_base pattern
    if metric_spec == "adjusted_mean_oos_sharpe":
        aggregator = AGGREGATORS["mean"]
        attr = "adjusted_oos_sharpe"
    elif metric_spec == "worst_adjusted_oos_sharpe":
        aggregator = AGGREGATORS["worst"]
        attr = "adjusted_oos_sharpe"
    else:
        # Parse aggregator prefix: "mean_oos_sharpe" -> agg="mean", base="oos_sharpe"
        aggregator = None
        base_metric = metric_spec

        for agg_name in AGGREGATORS:
            if metric_spec.startswith(agg_name + "_"):
                aggregator = AGGREGATORS[agg_name]
                base_metric = metric_spec[len(agg_name) + 1:]
                break

        if aggregator is None:
            # Default to mean aggregation
            aggregator = AGGREGATORS["mean"]

        # Map base metric to attribute name
        if base_metric in CYCLE_METRICS:
            attr = CYCLE_METRICS[base_metric]
        else:
            # Fall back to using base_metric as attribute name directly
            attr = base_metric

    # Extract values from cycle evaluations
    values = []
    for c in cycle_evals:
        val = getattr(c, attr, None)
        if val is not None:
            values.append(val)

    # Handle adjusted_oos_sharpe fallback to oos_sharpe
    if not values and attr == "adjusted_oos_sharpe":
        values = [getattr(c, "oos_sharpe", None) for c in cycle_evals]
        values = [v for v in values if v is not None]

    # Aggregate
    if not values:
        result = float("-inf")
    else:
        result = aggregator(values)

    return -result if negate else result


def get_metric_from_result(result: Any, metric_name: str) -> float:
    """
    Extract a metric from an EvaluationResult object.

    Parameters
    ----------
    result : EvaluationResult
        The evaluation result object
    metric_name : str
        Name of the metric to extract

    Returns
    -------
    float
        The metric value
    """
    metric_map = {
        "mean_oos_sharpe": "mean_oos_sharpe",
        "mean_wfe": "mean_wfe",
        "worst_oos_sharpe": "worst_oos_sharpe",
        "adjusted_mean_oos_sharpe": "adjusted_mean_oos_sharpe",
        "neg_is_oos_gap": "mean_is_oos_gap",  # Will be negated
    }

    attr = metric_map.get(metric_name, metric_name)
    value = getattr(result, attr, None)

    if value is None:
        return float("-inf")

    # Handle negation for gap metric
    if metric_name == "neg_is_oos_gap":
        return -value

    # Handle fallback for adjusted Sharpe
    if metric_name == "adjusted_mean_oos_sharpe" and value is None:
        return getattr(result, "mean_oos_sharpe", float("-inf"))

    return value
