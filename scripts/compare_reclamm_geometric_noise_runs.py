"""Wrapper for the canonical reCLAMM geometric noise comparison script."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("reclamm") / "compare_reclamm_geometric_noise_runs.py"),
        run_name="__main__",
    )
