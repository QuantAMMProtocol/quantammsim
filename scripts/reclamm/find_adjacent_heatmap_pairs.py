"""Scan cached reCLAMM heatmaps for adjacent cells with large value gaps.

This script reconstructs heatmap cells from the persisted scalar forward-value
cache written by ``compare_reclamm_thermostats.py``. It does not inspect PNG
pixels or rerun the simulator for cache-backed metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CACHE_ONLY_METRIC_SPECS = {
    "efficiency_pct": {
        "sources": ("noise_constant_arc", "noise_geometric"),
        "unit": "pct",
        "compute": lambda values: (
            values["noise_constant_arc"] / max(abs(values["noise_geometric"]), 1.0e-12)
            - 1.0
        )
        * 100.0,
    },
    "noise_geometric_final_value_musd": {
        "sources": ("noise_geometric",),
        "unit": "musd",
        "compute": lambda values: values["noise_geometric"] / 1.0e6,
    },
    "noise_constant_arc_final_value_musd": {
        "sources": ("noise_constant_arc",),
        "unit": "musd",
        "compute": lambda values: values["noise_constant_arc"] / 1.0e6,
    },
    "noise_vs_arb_geometric_improvement_pct": {
        "sources": ("noise_geometric", "arb_geometric"),
        "unit": "pct",
        "compute": lambda values: (
            values["noise_geometric"] / max(abs(values["arb_geometric"]), 1.0e-12) - 1.0
        )
        * 100.0,
    },
    "noise_vs_arb_constant_arc_improvement_pct": {
        "sources": ("noise_constant_arc", "arb_constant_arc"),
        "unit": "pct",
        "compute": lambda values: (
            values["noise_constant_arc"]
            / max(abs(values["arb_constant_arc"]), 1.0e-12)
            - 1.0
        )
        * 100.0,
    },
}

OUTPUT_COLUMNS = [
    "metric_key",
    "metric_unit",
    "source_noise_profile",
    "pair_slug",
    "slice_slug",
    "slice_label",
    "fixed_key",
    "fixed_value",
    "adjacency_axis",
    "heatmap_value_diff_abs",
    "heatmap_value_diff_signed_2_minus_1",
    "1_price_ratio",
    "1_centeredness_margin",
    "1_daily_price_shift_exponent",
    "1_tvl_usd",
    "1_heatmap_value",
    "1_x_index",
    "1_y_index",
    "2_price_ratio",
    "2_centeredness_margin",
    "2_daily_price_shift_exponent",
    "2_tvl_usd",
    "2_heatmap_value",
    "2_x_index",
    "2_y_index",
]


def build_inclusive_sweep(start: float, stop: float, step: float) -> np.ndarray:
    """Build a sweep that keeps the requested step and explicitly includes the stop."""
    values = np.arange(start, stop + 1.0e-12, step, dtype=float)
    if values.size == 0 or not np.isclose(values[-1], stop):
        values = np.append(values, float(stop))
    return values


class _LightweightCompareContext:
    """Small subset of compare_reclamm_thermostats usable without JAX."""

    RUN_CONSTANT_ARC_LENGTH = True
    DEFAULT_INITIAL_POOL_VALUE = 1_000_000.0
    TVL_SWEEP_VALUES = (
        1_000_000.0,
        5_000_000.0,
        20_000_000.0,
    )
    HEATMAP_PRICE_RATIOS = build_inclusive_sweep(1.01, 3.00, 0.025)
    HEATMAP_MARGINS = np.linspace(0.05, 0.90, 39)
    HEATMAP_SHIFT_EXPONENTS = build_inclusive_sweep(0.01, 0.50, 0.0125)
    FIXED_SLICE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
    FIXED_SLICE_LABELS = ("Q1", "Q2", "Q3", "Q4")
    HEATMAP_FORWARD_CACHE_ENABLED = True
    HEATMAP_FORWARD_CACHE_RUN_NAME = "aave_eth_thermostat_heatmaps_v1"
    HEATMAP_FORWARD_CACHE_ROOT = os.path.join(
        "results",
        "reclamm_heatmap_forward_cache",
    )
    AAVE_WETH_POOL_ID = "0x9d1fcf346ea1b0"
    DEFAULT_MARKET_LINEAR_ARTIFACT_DIR = "results/linear_market_noise"
    DEFAULT_NOISE_MODEL = "market_linear"
    DEFAULT_GAS_COST = 1.0
    DEFAULT_PROTOCOL_FEE_SPLIT = 0.25
    LEGACY_NOISE_COEFFS = [
        -0.453,
        0.025,
        -0.060,
        0.310,
        -0.149,
        0.359,
        0.061,
        0.060,
    ]
    LEGACY_LOG_CADENCE = 2.68
    LEGACY_ARB_FREQUENCY = max(1, round(math.exp(LEGACY_LOG_CADENCE)))
    FIXED_COMPARE_ARB_FREQUENCY = LEGACY_ARB_FREQUENCY
    AAVE_ETH_NOISE_SETTINGS = {
        "enable_noise_model": True,
        "noise_model": DEFAULT_NOISE_MODEL,
        "noise_artifact_dir": DEFAULT_MARKET_LINEAR_ARTIFACT_DIR,
        "noise_pool_id": AAVE_WETH_POOL_ID,
        "arb_frequency": FIXED_COMPARE_ARB_FREQUENCY,
        "gas_cost": DEFAULT_GAS_COST,
        "protocol_fee_split": DEFAULT_PROTOCOL_FEE_SPLIT,
    }
    CONFIGS = [
        {
            "name": "AAVE/ETH launch-style range (25bps, reference)",
            "tokens": ["AAVE", "ETH"],
            "start": "2024-06-01 00:00:00",
            "end": "2025-06-01 00:00:00",
            "fees": 0.0025,
            "price_ratio": 1.5014,
            "centeredness_margin": 0.5,
            "daily_price_shift_exponent": 0.1,
            "reason": "Original launch-style parameters.",
            **AAVE_ETH_NOISE_SETTINGS,
        },
        {
            "name": "AAVE/ETH aggressive tight range (25bps)",
            "tokens": ["AAVE", "ETH"],
            "start": "2024-06-01 00:00:00",
            "end": "2025-06-01 00:00:00",
            "fees": 0.0025,
            "price_ratio": 1.10,
            "centeredness_margin": 0.60,
            "daily_price_shift_exponent": 0.1,
            "reason": (
                "Aggressively tightened and moved to an earlier thermostat trigger. "
                "At fixed price_ratio=1.10, the shift_exponent sweep still favored "
                "0.1, while margin=0.60 widened the non-linear edge materially."
            ),
            **AAVE_ETH_NOISE_SETTINGS,
        },
    ]

    def __init__(self):
        self._noise_settings_cache = {}
        self.noise_profile = "market_linear"

    @classmethod
    def from_compare_module(cls, compare_module):
        """Build an analyzer-friendly context from an imported thermostat module."""
        context = cls()
        copied_attrs = (
            "RUN_CONSTANT_ARC_LENGTH",
            "DEFAULT_INITIAL_POOL_VALUE",
            "TVL_SWEEP_VALUES",
            "HEATMAP_PRICE_RATIOS",
            "HEATMAP_MARGINS",
            "HEATMAP_SHIFT_EXPONENTS",
            "FIXED_SLICE_FRACTIONS",
            "FIXED_SLICE_LABELS",
            "HEATMAP_FORWARD_CACHE_ENABLED",
            "HEATMAP_FORWARD_CACHE_RUN_NAME",
            "HEATMAP_FORWARD_CACHE_ROOT",
            "AAVE_WETH_POOL_ID",
            "DEFAULT_MARKET_LINEAR_ARTIFACT_DIR",
            "DEFAULT_NOISE_MODEL",
            "DEFAULT_GAS_COST",
            "DEFAULT_PROTOCOL_FEE_SPLIT",
            "LEGACY_NOISE_COEFFS",
            "LEGACY_LOG_CADENCE",
            "LEGACY_ARB_FREQUENCY",
            "FIXED_COMPARE_ARB_FREQUENCY",
            "AAVE_ETH_NOISE_SETTINGS",
            "CONFIGS",
        )
        for attr_name in copied_attrs:
            if hasattr(compare_module, attr_name):
                value = getattr(compare_module, attr_name)
                if attr_name == "CONFIGS":
                    value = [dict(cfg) for cfg in value]
                elif isinstance(value, dict):
                    value = dict(value)
                elif isinstance(value, np.ndarray):
                    value = np.asarray(value, dtype=float).copy()
                elif isinstance(value, tuple):
                    value = tuple(value)
                elif isinstance(value, list):
                    value = list(value)
                setattr(context, attr_name, value)
        context._noise_settings_cache.clear()
        return context

    def set_noise_profile(self, profile):
        if profile not in {"market_linear", "legacy_calibrated"}:
            raise ValueError(f"Unsupported lightweight noise profile: {profile}")
        if profile != self.noise_profile:
            self.noise_profile = profile
            self._noise_settings_cache.clear()

    def get_initial_pool_value(self, cfg):
        return float(cfg.get("initial_pool_value", self.DEFAULT_INITIAL_POOL_VALUE))

    def get_tvl_millions(self, cfg):
        return self.get_initial_pool_value(cfg) / 1_000_000.0

    def format_tvl_millions_slug(self, cfg):
        tvl_millions = self.get_tvl_millions(cfg)
        rounded = round(float(tvl_millions), 6)
        if np.isclose(rounded, round(rounded)):
            return f"{int(round(rounded))}m"
        return f"{rounded:.6f}".rstrip("0").rstrip(".").replace(".", "p") + "m"

    def format_tvl_millions_label(self, cfg):
        return f"{self.get_tvl_millions(cfg):.1f}M"

    def configs_for_tvl(self, base_configs, initial_pool_value):
        configs = []
        for cfg in base_configs:
            updated = dict(cfg)
            updated["initial_pool_value"] = float(initial_pool_value)
            configs.append(updated)
        return configs

    def _heatmap_forward_cache_scope_slug(self, cfg):
        if cfg is None:
            return "unspecified_tvl"
        return f"tvl_{self.format_tvl_millions_slug(cfg)}"

    def _heatmap_forward_cache_path(self, cfg):
        if not self.HEATMAP_FORWARD_CACHE_ENABLED:
            return None
        return os.path.join(
            self.HEATMAP_FORWARD_CACHE_ROOT,
            self.HEATMAP_FORWARD_CACHE_RUN_NAME,
            f"forward_values_{self._heatmap_forward_cache_scope_slug(cfg)}.parquet",
        )

    def build_fixed_slice_variants(self, values):
        values = np.asarray(values, dtype=float)
        if values.size < len(self.FIXED_SLICE_FRACTIONS):
            raise ValueError("Need at least four grid points to build fixed slices")

        variants = []
        used_indices = set()
        for idx, fraction in enumerate(self.FIXED_SLICE_FRACTIONS):
            target_index = int(round(fraction * (values.size - 1)))
            while target_index in used_indices and target_index + 1 < values.size:
                target_index += 1
            while target_index in used_indices and target_index - 1 >= 0:
                target_index -= 1
            if target_index in used_indices:
                raise ValueError(
                    "Could not build four unique fixed slices from sweep grid"
                )
            used_indices.add(target_index)
            variants.append(
                {
                    "index": target_index,
                    "fraction": fraction,
                    "label": self.FIXED_SLICE_LABELS[idx],
                    "slug": f"q{idx + 1}",
                    "value": float(values[target_index]),
                }
            )
        return variants

    def get_pair_heatmap_specs(self, _base_cfg):
        fixed_slice_variants = {
            "price_ratio": self.build_fixed_slice_variants(self.HEATMAP_PRICE_RATIOS),
            "centeredness_margin": self.build_fixed_slice_variants(self.HEATMAP_MARGINS),
            "daily_price_shift_exponent": self.build_fixed_slice_variants(
                self.HEATMAP_SHIFT_EXPONENTS
            ),
        }
        return [
            {
                "slug": "price_ratio_vs_margin",
                "x_values": self.HEATMAP_PRICE_RATIOS,
                "y_values": self.HEATMAP_MARGINS,
                "x_key": "price_ratio",
                "y_key": "centeredness_margin",
                "fixed_key": "daily_price_shift_exponent",
                "fixed_slices": fixed_slice_variants["daily_price_shift_exponent"],
            },
            {
                "slug": "shift_exp_vs_margin",
                "x_values": self.HEATMAP_SHIFT_EXPONENTS,
                "y_values": self.HEATMAP_MARGINS,
                "x_key": "daily_price_shift_exponent",
                "y_key": "centeredness_margin",
                "fixed_key": "price_ratio",
                "fixed_slices": fixed_slice_variants["price_ratio"],
            },
            {
                "slug": "price_ratio_vs_shift_exp",
                "x_values": self.HEATMAP_PRICE_RATIOS,
                "y_values": self.HEATMAP_SHIFT_EXPONENTS,
                "x_key": "price_ratio",
                "y_key": "daily_price_shift_exponent",
                "fixed_key": "centeredness_margin",
                "fixed_slices": fixed_slice_variants["centeredness_margin"],
            },
        ]

    @staticmethod
    def _hashable_noise_params(params):
        if params is None:
            return None
        return tuple(sorted((str(k), round(float(v), 12)) for k, v in params.items()))

    def _normalize_arb_frequency(self, value, default=None):
        if value is None:
            if default is None:
                default = self.FIXED_COMPARE_ARB_FREQUENCY
            value = default
        return max(int(round(float(value))), 1)

    def get_effective_arb_frequency(self, cfg, noise_cfg=None):
        del noise_cfg
        return self._normalize_arb_frequency(self.FIXED_COMPARE_ARB_FREQUENCY)

    def normalize_compare_run_cfg(self, cfg, enable_noise_model=None):
        updated = dict(cfg)
        updated["price_ratio"] = float(cfg["price_ratio"])
        updated["centeredness_margin"] = float(cfg["centeredness_margin"])
        updated["daily_price_shift_exponent"] = float(
            cfg["daily_price_shift_exponent"]
        )
        updated["initial_pool_value"] = float(self.get_initial_pool_value(cfg))
        updated["gas_cost"] = self.DEFAULT_GAS_COST
        updated["protocol_fee_split"] = self.DEFAULT_PROTOCOL_FEE_SPLIT
        updated["arb_fees"] = 0.0
        updated["arb_frequency"] = self.get_effective_arb_frequency(cfg)
        updated["noise_trader_ratio"] = 0.0

        arc_length_speed = cfg.get("arc_length_speed")
        if arc_length_speed is None:
            updated.pop("arc_length_speed", None)
        else:
            updated["arc_length_speed"] = float(arc_length_speed)

        use_noise = (
            bool(cfg.get("enable_noise_model", False))
            if enable_noise_model is None
            else bool(enable_noise_model)
        )
        updated["enable_noise_model"] = use_noise

        requested_mode = (
            cfg.get("noise_model", self.DEFAULT_NOISE_MODEL)
            or self.DEFAULT_NOISE_MODEL
        )
        if use_noise:
            canonical_noise_model = (
                requested_mode
                if requested_mode != "arb_only"
                else self.DEFAULT_NOISE_MODEL
            )
            updated["noise_model"] = canonical_noise_model
            if canonical_noise_model == "market_linear":
                updated["noise_artifact_dir"] = self.DEFAULT_MARKET_LINEAR_ARTIFACT_DIR
                updated["noise_pool_id"] = self.AAVE_WETH_POOL_ID
            else:
                updated.pop("noise_artifact_dir", None)
                updated.pop("noise_pool_id", None)
            updated.pop("reclamm_noise_params", None)
            updated.pop("noise_arrays_path", None)
        else:
            updated["noise_model"] = "arb_only"
            for key in (
                "reclamm_noise_params",
                "noise_arrays_path",
                "noise_artifact_dir",
                "noise_pool_id",
            ):
                updated.pop(key, None)

        return updated

    def _legacy_calibrated_noise_settings(self, arb_frequency=None):
        arb_frequency = self._normalize_arb_frequency(arb_frequency)
        return {
            "noise_model": "calibrated",
            "noise_trader_ratio": 0.0,
            "reclamm_noise_params": {
                f"c_{i}": self.LEGACY_NOISE_COEFFS[i]
                for i in range(len(self.LEGACY_NOISE_COEFFS))
            },
            "arb_frequency": arb_frequency,
            "noise_summary": (
                "calibrated legacy 8-covariate "
                f"(arb_frequency={arb_frequency})"
            ),
            "noise_cache_key": (
                "calibrated",
                tuple(round(float(c), 12) for c in self.LEGACY_NOISE_COEFFS),
                arb_frequency,
            ),
        }

    def resolve_reclamm_noise_settings(self, cfg):
        cfg = self.normalize_compare_run_cfg(cfg)
        enable_noise_model = cfg.get("enable_noise_model", False)
        requested_mode = cfg.get("noise_model", self.DEFAULT_NOISE_MODEL)
        requested_arb_frequency = self.get_effective_arb_frequency(cfg)
        cache_key = (
            tuple(cfg.get("tokens", [])),
            cfg.get("start"),
            cfg.get("end"),
            enable_noise_model,
            requested_mode,
            cfg.get("noise_artifact_dir", self.DEFAULT_MARKET_LINEAR_ARTIFACT_DIR),
            cfg.get("noise_pool_id", self.AAVE_WETH_POOL_ID),
            requested_arb_frequency,
            round(float(cfg.get("noise_trader_ratio", 0.0)), 12),
            self._hashable_noise_params(cfg.get("reclamm_noise_params")),
            cfg.get("noise_arrays_path"),
        )
        if cache_key in self._noise_settings_cache:
            return self._noise_settings_cache[cache_key]

        if not enable_noise_model:
            result = {
                "noise_model": "arb_only",
                "noise_trader_ratio": 0.0,
                "reclamm_noise_params": None,
                "noise_arrays_path": None,
                "arb_frequency": requested_arb_frequency,
                "noise_summary": "arb_only (noise disabled)",
                "noise_cache_key": ("disabled",),
            }
        elif requested_mode == "market_linear":
            if self.noise_profile == "legacy_calibrated":
                result = self._legacy_calibrated_noise_settings(
                    arb_frequency=requested_arb_frequency
                )
                self._noise_settings_cache[cache_key] = result
                return result
            artifact_dir = cfg.get(
                "noise_artifact_dir",
                self.DEFAULT_MARKET_LINEAR_ARTIFACT_DIR,
            )
            pool_id = cfg.get("noise_pool_id", self.AAVE_WETH_POOL_ID)
            start_date = str(cfg["start"]).split(" ")[0]
            end_date = str(cfg["end"]).split(" ")[0]
            arrays_path = cfg.get("noise_arrays_path") or os.path.join(
                artifact_dir,
                "_sim_arrays",
                f"{pool_id}_{start_date}_{end_date}.npz",
            )
            meta_path = os.path.join(artifact_dir, "meta.json")
            model_path = os.path.join(artifact_dir, "model.npz")
            if not (
                os.path.exists(arrays_path)
                and os.path.exists(meta_path)
                and os.path.exists(model_path)
            ):
                result = self._legacy_calibrated_noise_settings(
                    arb_frequency=requested_arb_frequency
                )
            else:
                with np.load(arrays_path) as arrays:
                    tvl_mean = float(arrays["tvl_mean"])
                    tvl_std = float(arrays["tvl_std"])
                arb_frequency = requested_arb_frequency
                result = {
                    "noise_model": "market_linear",
                    "noise_trader_ratio": 0.0,
                    "reclamm_noise_params": {
                        "tvl_mean": tvl_mean,
                        "tvl_std": tvl_std,
                    },
                    "noise_arrays_path": arrays_path,
                    "arb_frequency": arb_frequency,
                    "noise_summary": f"market_linear (arb_frequency={arb_frequency})",
                    "noise_cache_key": (
                        "market_linear",
                        arrays_path,
                        arb_frequency,
                        round(tvl_mean, 12),
                        round(tvl_std, 12),
                    ),
                }
        elif requested_mode == "calibrated":
            result = self._legacy_calibrated_noise_settings(
                arb_frequency=requested_arb_frequency
            )
        else:
            arb_frequency = requested_arb_frequency
            result = {
                "noise_model": requested_mode,
                "noise_trader_ratio": cfg.get("noise_trader_ratio", 0.0),
                "reclamm_noise_params": cfg.get("reclamm_noise_params"),
                "noise_arrays_path": cfg.get("noise_arrays_path"),
                "arb_frequency": arb_frequency,
                "noise_summary": f"{requested_mode} (arb_frequency={arb_frequency})",
                "noise_cache_key": (
                    requested_mode,
                    round(float(cfg.get("noise_trader_ratio", 0.0)), 12),
                    self._hashable_noise_params(cfg.get("reclamm_noise_params")),
                    cfg.get("noise_arrays_path"),
                    arb_frequency,
                ),
            }

        self._noise_settings_cache[cache_key] = result
        return result

    def make_noise_variant_cfg(self, cfg, enable_noise_model):
        return self.normalize_compare_run_cfg(
            cfg,
            enable_noise_model=enable_noise_model,
        )

    @staticmethod
    def _make_method_cache_hash(key):
        return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()

    def _make_method_cache_key(self, cfg, method):
        cfg = self.normalize_compare_run_cfg(cfg)
        noise_cfg = self.resolve_reclamm_noise_settings(cfg)
        arb_frequency = self.get_effective_arb_frequency(cfg, noise_cfg)
        key = (
            method,
            bool(cfg.get("enable_noise_model", False)),
            round(float(cfg["price_ratio"]), 6),
            round(float(cfg["centeredness_margin"]), 6),
            round(float(cfg["daily_price_shift_exponent"]), 6),
            round(self.get_initial_pool_value(cfg), 2),
            noise_cfg.get("noise_cache_key"),
            None if arb_frequency is None else int(arb_frequency),
            round(
                float(
                    cfg.get(
                        "gas_cost",
                        self.DEFAULT_GAS_COST
                        if cfg.get("enable_noise_model", False)
                        else 0.0,
                    )
                ),
                6,
            ),
            round(
                float(
                    cfg.get(
                        "protocol_fee_split",
                        self.DEFAULT_PROTOCOL_FEE_SPLIT
                        if cfg.get("enable_noise_model", False)
                        else 0.0,
                    )
                ),
                6,
            ),
        )
        if method == "constant_arc_length":
            speed = cfg.get("arc_length_speed")
            key += (None if speed is None else round(float(speed), 12),)
        return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Identify horizontally and vertically adjacent reCLAMM heatmap cells "
            "whose derived metric values differ by at least the requested threshold."
        )
    )
    parser.add_argument(
        "--metric-key",
        default="noise_vs_arb_geometric_improvement_pct",
        choices=sorted(CACHE_ONLY_METRIC_SPECS),
        help="Heatmap metric to reconstruct from the persisted forward-value cache.",
    )
    parser.add_argument(
        "--pair-slug",
        default="price_ratio_vs_margin",
        help="Pair heatmap family slug, or 'all' to scan every pair family.",
    )
    parser.add_argument(
        "--slice-slug",
        default="all",
        help="Quarter-slice slug (q1/q2/q3/q4), or 'all' to scan every slice.",
    )
    parser.add_argument(
        "--min-diff",
        type=float,
        default=30.0,
        help=(
            "Minimum absolute difference between adjacent heatmap values. "
            "For the default metric this is in percentage points."
        ),
    )
    parser.add_argument(
        "--adjacency-axis",
        default="both",
        choices=("both", "horizontal", "vertical"),
        help=(
            "Which adjacency direction to scan. "
            "'both' includes horizontal and vertical neighbors."
        ),
    )
    parser.add_argument(
        "--initial-pool-value",
        type=float,
        default=1_000_000.0,
        help="TVL in USD used for the cached heatmap sweep.",
    )
    parser.add_argument(
        "--config-index",
        type=int,
        default=1,
        help="Which compare_reclamm_thermostats.py base config to use.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional parquet cache override. Defaults to the compare script's TVL cache.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path. Defaults under scripts/results/.",
    )
    parser.add_argument(
        "--skip-top-row-geometric-comparison",
        action="store_true",
        help=(
            "Skip the follow-up geometric noise comparison for the top CSV row. "
            "By default the script attempts that comparison after writing the CSV."
        ),
    )
    parser.add_argument(
        "--top-row-geometric-comparison-output-file",
        default=None,
        help="Optional PNG output path override for the top-row geometric comparison.",
    )
    parser.add_argument(
        "--allow-partial-cache",
        action="store_true",
        help="Write output even if some heatmap cells are missing from the cache.",
    )
    return parser.parse_args()


def load_compare_module(module_path: Optional[Path] = None):
    compare_path = module_path or Path(__file__).with_name("compare_reclamm_thermostats.py")
    spec = importlib.util.spec_from_file_location(
        "reclamm_compare_reclamm_thermostats",
        compare_path,
    )
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            return _LightweightCompareContext.from_compare_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != "jax":
                raise
            print(
                "compare_reclamm_thermostats.py depends on jax in this environment; "
                "using the lightweight cache-key context instead."
            )
    return _LightweightCompareContext()


def get_metric_spec(metric_key: str) -> Mapping[str, object]:
    try:
        return CACHE_ONLY_METRIC_SPECS[metric_key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported metric_key={metric_key!r}. "
            f"Supported cache-only metrics: {sorted(CACHE_ONLY_METRIC_SPECS)}"
        ) from exc


def load_cache_lookup(cache_path: Path) -> Dict[str, float]:
    frame = pd.read_parquet(cache_path, columns=["cache_key_hash", "final_value"])
    return {
        str(row.cache_key_hash): float(row.final_value)
        for row in frame.itertuples(index=False)
    }


def resolve_existing_cache_path(cache_path: Path) -> Path:
    candidates = [Path(cache_path)]
    if not cache_path.is_absolute():
        candidates.append(Path("scripts") / cache_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(cache_path)


def load_geometric_compare_module(module_path: Optional[Path] = None):
    compare_path = module_path or Path(__file__).with_name(
        "compare_reclamm_geometric_noise_runs.py"
    )
    spec = importlib.util.spec_from_file_location(
        "reclamm_compare_reclamm_geometric_noise_runs",
        compare_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load geometric compare module from {compare_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_top_row_geometric_comparison(
    csv_path: Path,
    output_file: Optional[str] = None,
    row_index: int = 0,
):
    """Run the paired geometric-vs-arb comparison using the top adjacent CSV row."""
    module = load_geometric_compare_module()
    if not hasattr(module, "run_adjacent_csv_row_comparison"):
        raise RuntimeError(
            "compare_reclamm_geometric_noise_runs.py does not expose "
            "run_adjacent_csv_row_comparison"
        )
    return module.run_adjacent_csv_row_comparison(
        csv_path=csv_path,
        row_index=row_index,
        output_file=output_file,
    )


def resolve_pair_specs(compare_module, base_cfg: Mapping[str, object], pair_slug: str):
    pair_specs = compare_module.get_pair_heatmap_specs(base_cfg)
    if pair_slug == "all":
        return pair_specs

    matched = [pair for pair in pair_specs if pair["slug"] == pair_slug]
    if not matched:
        available = [pair["slug"] for pair in pair_specs]
        raise ValueError(
            f"Unknown pair slug {pair_slug!r}. Available pair slugs: {available}"
        )
    return matched


def resolve_slice_variants(pair_spec: Mapping[str, object], slice_slug: str):
    slice_variants = pair_spec["fixed_slices"]
    if slice_slug == "all":
        return list(slice_variants)

    matched = [variant for variant in slice_variants if variant["slug"] == slice_slug]
    if not matched:
        available = [variant["slug"] for variant in slice_variants]
        raise ValueError(
            f"Unknown slice slug {slice_slug!r}. Available slice slugs: {available}"
        )
    return matched


def build_default_output_path(
    compare_module,
    base_cfg: Mapping[str, object],
    metric_key: str,
    pair_slug: str,
    slice_slug: str,
    min_diff: float,
) -> Path:
    output_dir = Path("scripts/results/reclamm_heatmap_adjacency")
    output_dir.mkdir(parents=True, exist_ok=True)
    diff_token = str(float(min_diff)).rstrip("0").rstrip(".").replace(".", "p")
    filename = (
        f"reclamm_adjacent_pairs_{metric_key}_{pair_slug}_{slice_slug}"
        f"_mindiff_{diff_token}_tvl_{compare_module.format_tvl_millions_slug(base_cfg)}.csv"
    )
    return output_dir / filename


def autodetect_lightweight_noise_profile(
    compare_module,
    base_cfg: Mapping[str, object],
    pair_specs: Sequence[Mapping[str, object]],
    metric_key: str,
    slice_slug: str,
    cache_lookup: Mapping[str, float],
):
    if not hasattr(compare_module, "set_noise_profile"):
        return

    metric_spec = get_metric_spec(metric_key)
    if not pair_specs:
        return

    pair_spec = pair_specs[0]
    slice_variants = resolve_slice_variants(pair_spec, slice_slug)
    if not slice_variants:
        return

    slice_variant = slice_variants[0]
    x_values = list(pair_spec["x_values"])
    y_values = list(pair_spec["y_values"])
    sample_x_indices = sorted({0, len(x_values) // 2, len(x_values) - 1})
    sample_y_indices = sorted({0, len(y_values) // 2, len(y_values) - 1})

    scores = {}
    for profile in ("market_linear", "legacy_calibrated"):
        compare_module.set_noise_profile(profile)
        hit_count = 0
        probe_count = 0
        slice_cfg = dict(base_cfg)
        slice_cfg[pair_spec["fixed_key"]] = float(slice_variant["value"])
        for y_index in sample_y_indices:
            for x_index in sample_x_indices:
                cfg = dict(slice_cfg)
                cfg[pair_spec["x_key"]] = float(x_values[x_index])
                cfg[pair_spec["y_key"]] = float(y_values[y_index])
                for source_name in metric_spec["sources"]:
                    source_cfg, method = _source_variant(compare_module, cfg, source_name)
                    cache_key = compare_module._make_method_cache_key(source_cfg, method)
                    cache_key_hash = compare_module._make_method_cache_hash(cache_key)
                    probe_count += 1
                    if cache_key_hash in cache_lookup:
                        hit_count += 1
        scores[profile] = (hit_count, probe_count)

    best_profile = max(
        scores,
        key=lambda profile: (scores[profile][0], scores[profile][1], profile == "market_linear"),
    )
    compare_module.set_noise_profile(best_profile)
    hit_count, probe_count = scores[best_profile]
    print(
        f"Lightweight noise profile auto-detect chose {best_profile} "
        f"({hit_count}/{probe_count} sample cache hits)."
    )


def _source_variant(compare_module, cfg: Mapping[str, object], source_name: str):
    enable_noise_model = source_name.startswith("noise_")
    method = "geometric" if source_name.endswith("geometric") else "constant_arc_length"
    source_cfg = compare_module.make_noise_variant_cfg(cfg, enable_noise_model)
    source_cfg["noise_model"] = (
        getattr(compare_module, "DEFAULT_NOISE_MODEL", "market_linear")
        if enable_noise_model
        else "arb_only"
    )
    return source_cfg, method


def _compute_metric_value(metric_key: str, final_values: Mapping[str, float]) -> float:
    metric_spec = get_metric_spec(metric_key)
    return float(metric_spec["compute"](final_values))


def build_cell_record(
    compare_module,
    cfg: Mapping[str, object],
    pair_spec: Mapping[str, object],
    slice_variant: Mapping[str, object],
    metric_key: str,
    x_index: int,
    y_index: int,
    cache_lookup: Mapping[str, float],
):
    metric_spec = get_metric_spec(metric_key)
    final_values = {}
    missing_hashes = []
    for source_name in metric_spec["sources"]:
        source_cfg, method = _source_variant(compare_module, cfg, source_name)
        cache_key = compare_module._make_method_cache_key(source_cfg, method)
        cache_key_hash = compare_module._make_method_cache_hash(cache_key)
        cached_value = cache_lookup.get(cache_key_hash)
        if cached_value is None:
            missing_hashes.append(cache_key_hash)
            continue
        final_values[source_name] = float(cached_value)

    if missing_hashes:
        return None, missing_hashes

    return (
        {
            "metric_key": metric_key,
            "metric_unit": metric_spec["unit"],
            "source_noise_profile": str(
                getattr(compare_module, "noise_profile", "unknown")
            ),
            "pair_slug": pair_spec["slug"],
            "slice_slug": slice_variant["slug"],
            "slice_label": slice_variant["label"],
            "fixed_key": pair_spec["fixed_key"],
            "fixed_value": float(slice_variant["value"]),
            "price_ratio": float(cfg["price_ratio"]),
            "centeredness_margin": float(cfg["centeredness_margin"]),
            "daily_price_shift_exponent": float(cfg["daily_price_shift_exponent"]),
            "tvl_usd": float(compare_module.get_initial_pool_value(cfg)),
            "heatmap_value": _compute_metric_value(metric_key, final_values),
            "x_index": int(x_index),
            "y_index": int(y_index),
        },
        [],
    )


def build_slice_cell_grid(
    compare_module,
    base_cfg: Mapping[str, object],
    pair_spec: Mapping[str, object],
    slice_variant: Mapping[str, object],
    metric_key: str,
    cache_lookup: Mapping[str, float],
):
    records_by_coord: Dict[Tuple[int, int], MutableMapping[str, object]] = {}
    missing_hashes: List[str] = []
    x_values = pair_spec["x_values"]
    y_values = pair_spec["y_values"]
    slice_cfg = dict(base_cfg)
    slice_cfg[pair_spec["fixed_key"]] = float(slice_variant["value"])

    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            cfg = dict(slice_cfg)
            cfg[pair_spec["x_key"]] = float(x_value)
            cfg[pair_spec["y_key"]] = float(y_value)
            record, missing_for_cell = build_cell_record(
                compare_module=compare_module,
                cfg=cfg,
                pair_spec=pair_spec,
                slice_variant=slice_variant,
                metric_key=metric_key,
                x_index=x_index,
                y_index=y_index,
                cache_lookup=cache_lookup,
            )
            if record is not None:
                records_by_coord[(y_index, x_index)] = record
            missing_hashes.extend(missing_for_cell)

    expected_cell_count = len(x_values) * len(y_values)
    return {
        "records_by_coord": records_by_coord,
        "expected_cell_count": expected_cell_count,
        "resolved_cell_count": len(records_by_coord),
        "missing_hash_count": len(missing_hashes),
        "missing_hashes": missing_hashes,
    }


def build_adjacent_row(
    metric_key: str,
    metric_unit: str,
    axis: str,
    first_cell: Mapping[str, object],
    second_cell: Mapping[str, object],
) -> Dict[str, object]:
    signed_diff = float(second_cell["heatmap_value"]) - float(first_cell["heatmap_value"])
    abs_diff = abs(signed_diff)
    return {
        "metric_key": metric_key,
        "metric_unit": metric_unit,
        "source_noise_profile": first_cell.get("source_noise_profile", "unknown"),
        "pair_slug": first_cell["pair_slug"],
        "slice_slug": first_cell["slice_slug"],
        "slice_label": first_cell["slice_label"],
        "fixed_key": first_cell["fixed_key"],
        "fixed_value": float(first_cell["fixed_value"]),
        "adjacency_axis": axis,
        "heatmap_value_diff_abs": abs_diff,
        "heatmap_value_diff_signed_2_minus_1": signed_diff,
        "1_price_ratio": float(first_cell["price_ratio"]),
        "1_centeredness_margin": float(first_cell["centeredness_margin"]),
        "1_daily_price_shift_exponent": float(first_cell["daily_price_shift_exponent"]),
        "1_tvl_usd": float(first_cell["tvl_usd"]),
        "1_heatmap_value": float(first_cell["heatmap_value"]),
        "1_x_index": int(first_cell["x_index"]),
        "1_y_index": int(first_cell["y_index"]),
        "2_price_ratio": float(second_cell["price_ratio"]),
        "2_centeredness_margin": float(second_cell["centeredness_margin"]),
        "2_daily_price_shift_exponent": float(second_cell["daily_price_shift_exponent"]),
        "2_tvl_usd": float(second_cell["tvl_usd"]),
        "2_heatmap_value": float(second_cell["heatmap_value"]),
        "2_x_index": int(second_cell["x_index"]),
        "2_y_index": int(second_cell["y_index"]),
    }


def find_adjacent_rows_for_slice(
    metric_key: str,
    metric_unit: str,
    records_by_coord: Mapping[Tuple[int, int], Mapping[str, object]],
    x_count: int,
    y_count: int,
    min_diff: float,
    adjacency_axis: str = "both",
) -> List[Dict[str, object]]:
    rows = []

    if adjacency_axis not in {"both", "horizontal", "vertical"}:
        raise ValueError(
            f"Unsupported adjacency_axis={adjacency_axis!r}; expected both, horizontal, or vertical"
        )

    if adjacency_axis in {"both", "horizontal"}:
        for y_index in range(y_count):
            for x_index in range(x_count - 1):
                first_cell = records_by_coord.get((y_index, x_index))
                second_cell = records_by_coord.get((y_index, x_index + 1))
                if first_cell is None or second_cell is None:
                    continue
                row = build_adjacent_row(
                    metric_key=metric_key,
                    metric_unit=metric_unit,
                    axis="horizontal",
                    first_cell=first_cell,
                    second_cell=second_cell,
                )
                if row["heatmap_value_diff_abs"] >= min_diff:
                    rows.append(row)

    if adjacency_axis in {"both", "vertical"}:
        for y_index in range(y_count - 1):
            for x_index in range(x_count):
                first_cell = records_by_coord.get((y_index, x_index))
                second_cell = records_by_coord.get((y_index + 1, x_index))
                if first_cell is None or second_cell is None:
                    continue
                row = build_adjacent_row(
                    metric_key=metric_key,
                    metric_unit=metric_unit,
                    axis="vertical",
                    first_cell=first_cell,
                    second_cell=second_cell,
                )
                if row["heatmap_value_diff_abs"] >= min_diff:
                    rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row["heatmap_value_diff_abs"]),
            str(row["pair_slug"]),
            str(row["slice_slug"]),
            str(row["adjacency_axis"]),
            int(row["1_y_index"]),
            int(row["1_x_index"]),
        )
    )
    return rows


def scan_heatmap_pairs(
    compare_module,
    base_cfg: Mapping[str, object],
    metric_key: str,
    pair_specs: Sequence[Mapping[str, object]],
    slice_slug: str,
    min_diff: float,
    adjacency_axis: str,
    cache_lookup: Mapping[str, float],
):
    metric_spec = get_metric_spec(metric_key)
    all_rows: List[Dict[str, object]] = []
    diagnostics = []

    for pair_spec in pair_specs:
        slice_variants = resolve_slice_variants(pair_spec, slice_slug)
        for slice_variant in slice_variants:
            slice_scan = build_slice_cell_grid(
                compare_module=compare_module,
                base_cfg=base_cfg,
                pair_spec=pair_spec,
                slice_variant=slice_variant,
                metric_key=metric_key,
                cache_lookup=cache_lookup,
            )
            diagnostics.append(
                {
                    "pair_slug": pair_spec["slug"],
                    "slice_slug": slice_variant["slug"],
                    "resolved_cell_count": slice_scan["resolved_cell_count"],
                    "expected_cell_count": slice_scan["expected_cell_count"],
                    "missing_hash_count": slice_scan["missing_hash_count"],
                }
            )
            all_rows.extend(
                find_adjacent_rows_for_slice(
                    metric_key=metric_key,
                    metric_unit=metric_spec["unit"],
                    records_by_coord=slice_scan["records_by_coord"],
                    x_count=len(pair_spec["x_values"]),
                    y_count=len(pair_spec["y_values"]),
                    min_diff=min_diff,
                    adjacency_axis=adjacency_axis,
                )
            )

    return all_rows, diagnostics


def rows_to_frame(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = frame.loc[:, OUTPUT_COLUMNS]
    frame.sort_values(
        by=[
            "heatmap_value_diff_abs",
            "pair_slug",
            "slice_slug",
            "adjacency_axis",
            "1_y_index",
            "1_x_index",
        ],
        ascending=[False, True, True, True, True, True],
        inplace=True,
        ignore_index=True,
    )
    return frame


def main() -> int:
    args = parse_args()
    compare_module = load_compare_module()

    if not 0 <= args.config_index < len(compare_module.CONFIGS):
        raise ValueError(
            f"config-index {args.config_index} is out of range for "
            f"{len(compare_module.CONFIGS)} available configs"
        )

    base_cfg = compare_module.configs_for_tvl(
        compare_module.CONFIGS,
        initial_pool_value=args.initial_pool_value,
    )[args.config_index]
    pair_specs = resolve_pair_specs(compare_module, base_cfg, args.pair_slug)

    cache_path = (
        Path(args.cache_path)
        if args.cache_path is not None
        else Path(compare_module._heatmap_forward_cache_path(base_cfg))
    )
    cache_path = resolve_existing_cache_path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache parquet not found: {cache_path}")

    cache_lookup = load_cache_lookup(cache_path)
    autodetect_lightweight_noise_profile(
        compare_module=compare_module,
        base_cfg=base_cfg,
        pair_specs=pair_specs,
        metric_key=args.metric_key,
        slice_slug=args.slice_slug,
        cache_lookup=cache_lookup,
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else build_default_output_path(
            compare_module=compare_module,
            base_cfg=base_cfg,
            metric_key=args.metric_key,
            pair_slug=args.pair_slug,
            slice_slug=args.slice_slug,
            min_diff=args.min_diff,
        )
    )

    print(
        f"Loaded {len(cache_lookup):,} cached final values from {cache_path} "
        f"for {base_cfg['name']} at TVL {compare_module.format_tvl_millions_label(base_cfg)}."
    )
    print(
        f"Scanning metric={args.metric_key}, pair_slug={args.pair_slug}, "
        f"slice_slug={args.slice_slug}, adjacency_axis={args.adjacency_axis}, "
        f"min_diff={args.min_diff} "
        f"({get_metric_spec(args.metric_key)['unit']})."
    )

    rows, diagnostics = scan_heatmap_pairs(
        compare_module=compare_module,
        base_cfg=base_cfg,
        metric_key=args.metric_key,
        pair_specs=pair_specs,
        slice_slug=args.slice_slug,
        min_diff=args.min_diff,
        adjacency_axis=args.adjacency_axis,
        cache_lookup=cache_lookup,
    )

    for diagnostic in diagnostics:
        print(
            f"[{diagnostic['pair_slug']}:{diagnostic['slice_slug']}] "
            f"resolved {diagnostic['resolved_cell_count']}/"
            f"{diagnostic['expected_cell_count']} cells "
            f"({diagnostic['missing_hash_count']} missing cache hashes)"
        )

    missing_any = any(diagnostic["missing_hash_count"] > 0 for diagnostic in diagnostics)
    if missing_any and not args.allow_partial_cache:
        raise RuntimeError(
            "Cache was incomplete for at least one requested heatmap slice. "
            "Re-run with --allow-partial-cache to write the rows that were resolvable."
        )

    frame = rows_to_frame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    print(f"Wrote {len(frame):,} adjacent pairs to {output_csv}")

    if not args.skip_top_row_geometric_comparison:
        if frame.empty:
            print("Skipping top-row geometric comparison because the CSV is empty.")
        else:
            print(
                "Running geometric noise comparison for the top adjacent-pairs CSV row..."
            )
            try:
                comparison_output = run_top_row_geometric_comparison(
                    csv_path=output_csv,
                    output_file=args.top_row_geometric_comparison_output_file,
                    row_index=0,
                )
                print(
                    f"Completed top-row geometric comparison using {output_csv} row 0. "
                    f"Output: {comparison_output}"
                )
            except Exception as exc:  # pragma: no cover - depends on local runtime deps
                print(
                    "Top-row geometric comparison did not run successfully: "
                    f"{exc}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
