"""
Robust Walk-Forward Training Utilities

This module provides core utilities for walk-forward analysis:

1. **Rademacher Complexity (Paleologo)**
   - Compute empirical Rademacher complexity from checkpoint returns
   - Apply haircut to OOS performance estimates

2. **Walk-Forward Efficiency (Pardo)**
   - Compute WFE = OOS performance / IS performance
   - Standard metric for assessing robustness

3. **Cycle Generation**
   - Generate walk-forward train/test splits
   - Support for rolling and expanding windows

Key References:
- Pardo, "The Evaluation and Optimization of Trading Strategies" (2008)
- Paleologo, "The Elements of Quantitative Investing" (2024), Ch. 6
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WalkForwardCycle:
    """Specification for a single walk-forward train/test cycle.

    Defines one segment of a walk-forward analysis: a contiguous training
    window followed by a contiguous test window.  Date fields are set at
    cycle-generation time; index fields are populated later once the price
    data has been loaded and aligned.

    Attributes
    ----------
    cycle_number : int
        Zero-based index of this cycle within the walk-forward sequence.
    train_start_date : str
        Training window start date (``"YYYY-MM-DD HH:MM:SS"``).
    train_end_date : str
        Training window end date (inclusive).
    test_start_date : str
        Test window start date, typically equal to ``train_end_date``.
    test_end_date : str
        Test window end date (inclusive).
    train_start_idx : int
        Row index into the price array for the start of training.
        Default 0; populated after data loading.
    train_end_idx : int
        Row index for the end of training. Default 0.
    test_start_idx : int
        Row index for the start of testing. Default 0.
    test_end_idx : int
        Row index for the end of testing. Default 0.
    """
    cycle_number: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    # Indices populated after data loading
    train_start_idx: int = 0
    train_end_idx: int = 0
    test_start_idx: int = 0
    test_end_idx: int = 0


# =============================================================================
# Rademacher Complexity Utilities
# =============================================================================

def compute_empirical_rademacher(
    returns_matrix: np.ndarray,
    n_samples: int = 1000,
    seed: int = 42,
) -> float:
    """
    Compute empirical Rademacher complexity of a set of strategies.

    The Rademacher complexity measures how well the strategy class can
    "fit" random noise. Higher complexity = more overfitting risk.

    Parameters
    ----------
    returns_matrix : ndarray of shape (n_strategies, T)
        Returns time series for each strategy (checkpoint)
    n_samples : int
        Number of random sign vectors to sample
    seed : int
        Random seed for reproducibility

    Returns
    -------
    float
        Empirical Rademacher complexity R̂

    Notes
    -----
    R̂ = E_σ[sup_s (1/T) Σ_t σ_t r_s(t)]

    where σ_t are random Rademacher variables (±1 with prob 0.5)
    """
    if returns_matrix.ndim == 1:
        returns_matrix = returns_matrix.reshape(1, -1)

    n_strategies, T = returns_matrix.shape

    if n_strategies == 0 or T == 0:
        return 0.0

    rng = np.random.RandomState(seed)
    suprema = []

    for _ in range(n_samples):
        # Random Rademacher signs
        sigma = rng.choice([-1, 1], size=T)

        # Correlation of each strategy with random signs
        correlations = returns_matrix @ sigma / T

        # Supremum over strategies
        sup = np.max(correlations)
        suprema.append(sup)

    return np.mean(suprema)


def compute_rademacher_haircut(
    observed_sharpe: float,
    rademacher_complexity: float,
    T: int,
    delta: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute Rademacher-adjusted performance bound.

    From Paleologo (2024):
    θ_n ≥ θ̂_n - 2R̂ - estimation_error

    Parameters
    ----------
    observed_sharpe : float
        Observed Sharpe ratio on test data
    rademacher_complexity : float
        Empirical Rademacher complexity R̂
    T : int
        Number of time periods in test data
    delta : float
        Confidence level (default 0.05 for 95% confidence)

    Returns
    -------
    Tuple[float, float]
        (adjusted_sharpe, haircut_magnitude)
    """
    # Guard against T=0 which would cause division by zero
    if T <= 0:
        return float('nan'), float('nan')

    # Estimation error term: 3√(2log(2/δ)/T)
    estimation_error = 3 * np.sqrt(2 * np.log(2 / delta) / T)

    # Total haircut
    haircut = 2 * rademacher_complexity + estimation_error

    adjusted_sharpe = observed_sharpe - haircut

    return adjusted_sharpe, haircut


# =============================================================================
# Walk-Forward Efficiency (Pardo)
# =============================================================================

def compute_walk_forward_efficiency(
    is_sharpe: float,
    oos_sharpe: float,
    is_days: int,
    oos_days: int,
) -> float:
    """
    Compute Walk-Forward Efficiency (WFE) as per Pardo.

    WFE = (Annualized OOS Performance) / (Annualized IS Performance)

    A WFE of 0.5 or higher suggests robustness.
    A WFE near 1.0 is ideal (OOS ≈ IS).
    A WFE > 1.0 means OOS outperformed IS (unusual but possible).

    Parameters
    ----------
    is_sharpe : float
        In-sample Sharpe ratio
    oos_sharpe : float
        Out-of-sample Sharpe ratio
    is_days : int
        Number of days in IS period
    oos_days : int
        Number of days in OOS period

    Returns
    -------
    float
        Walk-Forward Efficiency (returns NaN for undefined cases)
    """
    # Handle edge cases where WFE is undefined
    if is_sharpe <= 0:
        # Can't compute meaningful ratio when IS is non-positive
        # Return NaN to signal "undefined" rather than 0.0 which masks failure
        # Callers should filter with np.isfinite() - see _aggregate_results
        return float('nan')

    return oos_sharpe / is_sharpe


# =============================================================================
# Cycle Generation Utilities
# =============================================================================

def datetime_to_timestamp(date_string: str) -> float:
    """Convert a datetime string to a Unix timestamp.

    Parameters
    ----------
    date_string : str
        Date in ``"YYYY-MM-DD HH:MM:SS"`` format.

    Returns
    -------
    float
        Seconds since the Unix epoch, interpreted in the local timezone.
    """
    dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


def timestamp_to_datetime(timestamp: float) -> str:
    """Convert a Unix timestamp to a datetime string.

    Parameters
    ----------
    timestamp : float
        Seconds since the Unix epoch.

    Returns
    -------
    str
        Formatted date string in ``"YYYY-MM-DD HH:MM:SS"`` format.
    """
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def generate_walk_forward_cycles(
    start_date: str,
    end_date: str,
    n_cycles: int,
    keep_fixed_start: bool = False,
) -> List[WalkForwardCycle]:
    """
    Generate walk-forward cycle specifications with equal-length test periods.

    Divides [start_date, end_date] into (n_cycles + 1) equal segments.
    Each cycle trains on segment i and tests on segment i+1.

    Parameters
    ----------
    start_date : str
        Start date (format: "YYYY-MM-DD HH:MM:SS")
    end_date : str
        End date of walk-forward analysis (end of final test period)
    n_cycles : int
        Number of training/test cycles
    keep_fixed_start : bool
        If True, training always starts from start_date (expanding window).
        If False (default), training window rolls forward (rolling window).

    Returns
    -------
    List[WalkForwardCycle]
    """
    start_ts = datetime_to_timestamp(start_date)
    end_ts = datetime_to_timestamp(end_date)

    # Create n_cycles + 1 segment boundaries
    # n_cycles + 1 segments means n_cycles + 2 boundary points
    n_segments = n_cycles + 1
    times = np.linspace(start_ts, end_ts, n_segments + 1)

    # Round to midnight
    times = times - (times % (24 * 60 * 60))

    cycles = []
    for i in range(n_cycles):
        if keep_fixed_start:
            train_start = times[0]
        else:
            train_start = times[i]

        train_end = times[i + 1]
        test_start = times[i + 1]
        test_end = times[i + 2]

        cycles.append(WalkForwardCycle(
            cycle_number=i,
            train_start_date=timestamp_to_datetime(train_start),
            train_end_date=timestamp_to_datetime(train_end),
            test_start_date=timestamp_to_datetime(test_start),
            test_end_date=timestamp_to_datetime(test_end),
        ))

    return cycles
