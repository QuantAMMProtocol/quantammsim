"""
QuantAMMSim: Quantitative AMM Simulator
"""

try:
    import numpy as np  # noqa: F401
except ImportError as e:
    raise ImportError(
        "NumPy is required for QuantAMMSim. Please install numpy."
    ) from e

from . import runners
from . import pools
from . import core_simulator
from . import utils

__version__ = "0.1.0"

__all__ = [
    "runners",
    "pools",
    "core_simulator",
    "utils",
]
