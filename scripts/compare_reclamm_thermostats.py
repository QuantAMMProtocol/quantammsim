"""Compare reCLAMM interpolation modes on historic AAVE/ETH data.

Runs the production geometric interpolation against the non-linear
constant-arc-length interpolation on:
1. The original launch-style range (price_ratio ~= 1.50)
2. A much tighter range (price_ratio = 1.10)

The aggressive case is deliberate. A local AAVE/ETH sweep showed:
price_ratio 1.15, margin 0.5, shift 0.1 -> about +$10k vs geometric
price_ratio 1.10, margin 0.5, shift 0.1 -> about +$31k vs geometric
price_ratio 1.10, margin 0.6, shift 0.1 -> about +$73k vs geometric

So the strongest clean demo setting came from tightening the band and
slightly raising the trigger margin, while keeping the launch-style shift
speed rather than pushing shift_exponent higher.
"""

import gc
import hashlib
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from quantammsim.pools.reCLAMM.reclamm_reserves import (
    calibrate_arc_length_speed,
    compute_price_ratio,
    initialise_reclamm_reserves,
)
from quantammsim.runners.jax_runners import do_run_on_historic_data
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_parquet_data,
)


def to_daily_price_shift_base(daily_price_shift_exponent):
    """Convert shift rate to daily price shift base (matches Solidity)."""
    return 1.0 - daily_price_shift_exponent / 124649.0


def build_inclusive_sweep(start, stop, step):
    """Build a sweep that keeps the requested step and explicitly includes the stop."""
    values = np.arange(start, stop + 1.0e-12, step, dtype=float)
    if values.size == 0 or not np.isclose(values[-1], stop):
        values = np.append(values, float(stop))
    return values


def _resolve_repo_root(script_path):
    """Locate the repository root from either scripts/ or scripts/reclamm/."""
    script_path = Path(script_path).resolve()
    for parent in script_path.parents:
        if (parent / "quantammsim").exists() and (parent / "scripts").exists():
            return parent
    return script_path.parents[1]


RUN_CONSTANT_ARC_LENGTH = True
INTERPOLATION_METHODS = (
    ("geometric", "constant_arc_length")
    if RUN_CONSTANT_ARC_LENGTH
    else ("geometric",)
)
HEATMAP_PRICE_RATIOS = build_inclusive_sweep(1.01, 3.00, 0.025)
HEATMAP_MARGINS = np.linspace(0.05, 0.90, 39)
HEATMAP_SHIFT_EXPONENTS = build_inclusive_sweep(0.01, 0.50, 0.0125)
HEATMAP_ARC_LENGTH_SPEEDS = np.geomspace(1.0e-6, 5.0e-4, 11)
PRICE_RATIO_TICKS = np.array([1.01, 1.25, 1.50, 2.00, 2.50, 3.00])
MARGIN_TICKS = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90])
SHIFT_EXPONENT_TICKS = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
ARC_LENGTH_SPEED_TICKS = np.array([
    1.0e-6,
    2.0e-6,
    5.0e-6,
    1.0e-5,
    2.0e-5,
    5.0e-5,
    1.0e-4,
    2.0e-4,
    5.0e-4,
])
SWEEP_LINE_WIDTH = 0.45
REFERENCE_LINE_WIDTH = 0.9
DEFAULT_INITIAL_POOL_VALUE = 1_000_000.0
TVL_SWEEP_VALUES = (
    1_000_000.0,
    5_000_000.0,
    20_000_000.0,
)
CENTER_ZERO_HEATMAP_COLOR_NORM = "symlog"
CENTER_ZERO_HEATMAP_COLOR_TAG = "symlog20"
CENTER_ZERO_HEATMAP_SYMLOG_LINTHRESH = 20.0
FIXED_SLICE_FRACTIONS = (0.125, 0.375, 0.625, 0.875)
FIXED_SLICE_LABELS = ("Q1", "Q2", "Q3", "Q4")
THREE_D_VIEW_ELEVATION = 22.0
THREE_D_VIEW_AZIMUTH = 140.0
HEATMAP_FORWARD_CACHE_ENABLED = True
HEATMAP_FORWARD_CACHE_RUN_NAME = "aave_eth_thermostat_heatmaps_market_linear_v2"
HEATMAP_FORWARD_CACHE_ROOT = os.path.join(
    "results",
    "reclamm_heatmap_forward_cache",
)
HEATMAP_FORWARD_CACHE_FLUSH_EVERY = 360

REPO_ROOT = _resolve_repo_root(__file__)
AAVE_WETH_POOL_ID = "0x9d1fcf346ea1b0"
DEFAULT_MARKET_LINEAR_ARTIFACT_DIR = "results/linear_market_noise"
DEFAULT_MARKET_LINEAR_NOISE_START_DATE = "2024-06-01"
DEFAULT_MARKET_LINEAR_NOISE_END_DATE = "2026-03-01"
DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH = str(
    REPO_ROOT
    / "results"
    / "linear_market_noise"
    / "_sim_arrays"
    / (
        f"{AAVE_WETH_POOL_ID}_{DEFAULT_MARKET_LINEAR_NOISE_START_DATE}_"
        f"{DEFAULT_MARKET_LINEAR_NOISE_END_DATE}.npz"
    )
)
DEFAULT_NOISE_MODEL = "market_linear"
DEFAULT_GAS_COST = 1.0
DEFAULT_PROTOCOL_FEE_SPLIT = 0.25
FIXED_COMPARE_ARB_FREQUENCY = 15
AAVE_ETH_NOISE_SETTINGS = {
    "enable_noise_model": True,
    "noise_model": DEFAULT_NOISE_MODEL,
    "noise_reference_model": DEFAULT_NOISE_MODEL,
    "noise_artifact_dir": DEFAULT_MARKET_LINEAR_ARTIFACT_DIR,
    "noise_pool_id": AAVE_WETH_POOL_ID,
    "noise_arrays_path": DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH,
    "arb_frequency": FIXED_COMPARE_ARB_FREQUENCY,
    "gas_cost": DEFAULT_GAS_COST,
    "protocol_fee_split": DEFAULT_PROTOCOL_FEE_SPLIT,
}
PERSISTED_FORWARD_VALUE_COLUMNS = (
    "cache_key_hash",
    "final_value",
    "method",
    "enable_noise_model",
    "noise_model",
    "price_ratio",
    "centeredness_margin",
    "daily_price_shift_exponent",
    "initial_pool_value",
    "arb_frequency",
)

GEOMETRIC_ONLY_HEATMAP_METRIC_KEYS = (
    "geometric_vs_launch_geometric_pct",
    "noise_geometric_final_value_musd",
    "noise_vs_arb_geometric_improvement_pct",
)
CONSTANT_ARC_HEATMAP_METRIC_KEYS = (
    "efficiency_pct",
    "launch_geometric_efficiency_pct",
    "constant_arc_vs_launch_constant_arc_pct",
    "noise_constant_arc_final_value_musd",
    "noise_vs_arb_constant_arc_improvement_pct",
)
HEATMAP_METRIC_DEPENDENCIES = {
    "efficiency_pct": ("noise_geometric", "noise_constant_arc"),
    "launch_geometric_efficiency_pct": ("noise_constant_arc",),
    "geometric_vs_launch_geometric_pct": ("noise_geometric",),
    "constant_arc_vs_launch_constant_arc_pct": ("noise_constant_arc",),
    "noise_geometric_final_value_musd": ("noise_geometric",),
    "noise_constant_arc_final_value_musd": ("noise_constant_arc",),
    "noise_vs_arb_geometric_improvement_pct": ("noise_geometric", "arb_geometric"),
    "noise_vs_arb_constant_arc_improvement_pct": (
        "noise_constant_arc",
        "arb_constant_arc",
    ),
}

_NOISE_SETTINGS_CACHE = {}
_MARKET_LINEAR_NOISE_DATA_CACHE = {}


def get_initial_pool_value(cfg):
    """Return the configured base pool TVL in USD."""
    return float(cfg.get("initial_pool_value", DEFAULT_INITIAL_POOL_VALUE))


def get_tvl_millions(cfg):
    """Return the configured base pool TVL in millions of USD."""
    return get_initial_pool_value(cfg) / 1_000_000.0


def format_tvl_millions_slug(cfg):
    """Format the TVL in millions for stable filenames."""
    tvl_millions = get_tvl_millions(cfg)
    rounded = round(float(tvl_millions), 6)
    if np.isclose(rounded, round(rounded)):
        return f"{int(round(rounded))}m"
    return f"{rounded:.6f}".rstrip("0").rstrip(".").replace(".", "p") + "m"


def format_tvl_millions_label(cfg):
    """Format the TVL in millions for plot titles and logs."""
    return f"{get_tvl_millions(cfg):.1f}M"


def tvl_artifact_filename(stem, cfg, suffix=None):
    """Append a TVL-in-millions suffix to a PNG artifact name."""
    parts = [stem]
    if suffix:
        parts.append(suffix)
    parts.append(f"tvl_{format_tvl_millions_slug(cfg)}")
    return "_".join(parts) + ".png"


def heatmap_artifact_filename(spec, cfg, suffix=None):
    """Build a heatmap filename, including any colour-style tag."""
    stem = f"reclamm_heatmap_{spec['slug']}"
    artifact_tag = spec.get("artifact_tag")
    if artifact_tag:
        stem = f"{stem}_{artifact_tag}"
    return tvl_artifact_filename(stem, cfg, suffix=suffix)


def three_d_heatmap_artifact_filename(spec, cfg, suffix=None):
    """Build a 3D heatmap filename, including any colour-style tag."""
    stem = f"reclamm_heatmap_3d_{spec['slug']}"
    artifact_tag = spec.get("artifact_tag")
    if artifact_tag:
        stem = f"{stem}_{artifact_tag}"
    return tvl_artifact_filename(stem, cfg, suffix=suffix)


def format_heatmap_param_value(value):
    """Format a sweep parameter compactly for titles and logs."""
    value = float(value)
    if abs(value) >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def configs_for_tvl(base_configs, initial_pool_value):
    """Attach a shared initial TVL to each compare configuration."""
    configs = []
    for cfg in base_configs:
        updated = dict(cfg)
        updated["initial_pool_value"] = float(initial_pool_value)
        configs.append(updated)
    return configs


def _normalize_arb_frequency(value, default=FIXED_COMPARE_ARB_FREQUENCY):
    """Return a stable integer arb cadence for thermostat comparisons."""
    if value is None:
        if default is None:
            return None
        value = default
    return max(int(round(float(value))), 1)


def get_effective_arb_frequency(cfg, noise_cfg=None):
    """Resolve the arb cadence used by a thermostat comparison run."""
    del noise_cfg
    return _normalize_arb_frequency(FIXED_COMPARE_ARB_FREQUENCY)


def _canonical_noise_reference_model(cfg):
    """Resolve the only supported thermostat noise parametrisation."""
    noise_model = cfg.get("noise_model", DEFAULT_NOISE_MODEL) or DEFAULT_NOISE_MODEL
    reference_model = cfg.get("noise_reference_model")
    if reference_model is None:
        reference_model = DEFAULT_NOISE_MODEL if noise_model == "arb_only" else noise_model
    noise_model = str(noise_model)
    reference_model = str(reference_model)
    if noise_model not in {DEFAULT_NOISE_MODEL, "arb_only"}:
        raise ValueError(
            "compare_reclamm_thermostats only supports "
            "'market_linear' noise and 'arb_only' baselines."
        )
    if reference_model != DEFAULT_NOISE_MODEL:
        raise ValueError(
            "compare_reclamm_thermostats only supports the "
            "'market_linear' noise parametrisation."
        )
    return reference_model


def normalize_compare_run_cfg(cfg, enable_noise_model=None):
    """Canonicalize the compare-run config so non-axis inputs stay fixed."""
    updated = dict(cfg)
    updated["price_ratio"] = float(cfg["price_ratio"])
    updated["centeredness_margin"] = float(cfg["centeredness_margin"])
    updated["daily_price_shift_exponent"] = float(cfg["daily_price_shift_exponent"])
    updated["initial_pool_value"] = float(get_initial_pool_value(cfg))
    updated["gas_cost"] = DEFAULT_GAS_COST
    updated["protocol_fee_split"] = DEFAULT_PROTOCOL_FEE_SPLIT
    updated["arb_fees"] = 0.0
    updated["arb_frequency"] = get_effective_arb_frequency(cfg)
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

    reference_mode = _canonical_noise_reference_model(cfg)
    if use_noise:
        updated["noise_model"] = reference_mode
        updated["noise_reference_model"] = reference_mode
    else:
        updated["noise_model"] = "arb_only"
        updated["noise_reference_model"] = reference_mode

    updated["noise_arrays_path"] = DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    updated.pop("reclamm_noise_params", None)
    updated["noise_artifact_dir"] = DEFAULT_MARKET_LINEAR_ARTIFACT_DIR
    updated["noise_pool_id"] = AAVE_WETH_POOL_ID

    return updated


def make_noise_variant_cfg(cfg, enable_noise_model):
    """Return a config with either noise modelling or pure arb-only enabled."""
    return normalize_compare_run_cfg(cfg, enable_noise_model=enable_noise_model)


def _hashable_noise_params(params):
    """Convert a noise-params dict into a stable cache key fragment."""
    if params is None:
        return None
    return tuple(sorted((str(k), round(float(v), 12)) for k, v in params.items()))


def load_shared_market_linear_noise_data(
    arrays_path=DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH,
):
    """Load the market_linear arrays once so compare runs can reuse them."""
    arrays_path = os.path.abspath(os.fspath(arrays_path))
    cached = _MARKET_LINEAR_NOISE_DATA_CACHE.get(arrays_path)
    if cached is not None:
        return cached

    if not os.path.exists(arrays_path):
        raise FileNotFoundError(f"market_linear arrays file not found: {arrays_path}")

    with np.load(arrays_path) as arrays:
        required_keys = {"noise_base", "noise_tvl_coeff", "tvl_mean", "tvl_std"}
        missing_keys = sorted(required_keys.difference(arrays.files))
        if missing_keys:
            raise KeyError(
                f"market_linear arrays file {arrays_path} is missing keys: {missing_keys}"
            )
        shared = {
            "arrays_path": arrays_path,
            "noise_base_array": np.asarray(arrays["noise_base"]),
            "noise_tvl_coeff_array": np.asarray(arrays["noise_tvl_coeff"]),
            "tvl_mean": float(arrays["tvl_mean"]),
            "tvl_std": float(arrays["tvl_std"]),
        }
    _MARKET_LINEAR_NOISE_DATA_CACHE[arrays_path] = shared
    return shared


def _load_market_linear_noise_stats(arrays_path=DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH):
    """Load the exact arrays file used by the market_linear run fingerprint.

    The simulator consumes ``noise_base`` and ``noise_tvl_coeff`` from
    ``run_fingerprint["noise_arrays_path"]`` and uses ``tvl_mean``/``tvl_std``
    from the same file for TVL standardization.
    """
    shared = load_shared_market_linear_noise_data(arrays_path=arrays_path)
    return shared["arrays_path"], shared["tvl_mean"], shared["tvl_std"]


def _market_linear_noise_settings(noise_model="market_linear", arb_frequency=None):
    """Build the tuned market_linear fingerprint block from the fixed arrays file."""
    arrays_path, tvl_mean, tvl_std = _load_market_linear_noise_stats()
    arb_frequency = _normalize_arb_frequency(arb_frequency)
    return {
        "noise_model": noise_model,
        "noise_trader_ratio": 0.0,
        "reclamm_noise_params": {
            "tvl_mean": tvl_mean,
            "tvl_std": tvl_std,
        },
        "noise_arrays_path": arrays_path,
        "arb_frequency": arb_frequency,
        "noise_summary": f"{noise_model} (arb_frequency={arb_frequency})",
        "noise_cache_key": (
            noise_model,
            arrays_path,
            arb_frequency,
            round(tvl_mean, 12),
            round(tvl_std, 12),
        ),
    }

def resolve_reclamm_noise_settings(cfg):
    """Resolve the active reCLAMM noise-model fingerprint block for a config."""
    cfg = normalize_compare_run_cfg(cfg)
    enable_noise_model = cfg.get("enable_noise_model", False)
    requested_mode = cfg.get("noise_model", DEFAULT_NOISE_MODEL)
    reference_mode = cfg.get("noise_reference_model", DEFAULT_NOISE_MODEL)
    requested_arb_frequency = get_effective_arb_frequency(cfg)
    cache_key = (
        tuple(cfg.get("tokens", [])),
        cfg.get("start"),
        cfg.get("end"),
        enable_noise_model,
        requested_mode,
        reference_mode,
        cfg.get("noise_artifact_dir", DEFAULT_MARKET_LINEAR_ARTIFACT_DIR),
        cfg.get("noise_pool_id", AAVE_WETH_POOL_ID),
        requested_arb_frequency,
        round(float(cfg.get("noise_trader_ratio", 0.0)), 12),
        _hashable_noise_params(cfg.get("reclamm_noise_params")),
        cfg.get("noise_arrays_path"),
    )
    if cache_key in _NOISE_SETTINGS_CACHE:
        return _NOISE_SETTINGS_CACHE[cache_key]

    if requested_mode == "arb_only":
        result = _market_linear_noise_settings(
            noise_model="arb_only",
            arb_frequency=requested_arb_frequency,
        )
    elif requested_mode == DEFAULT_NOISE_MODEL:
        result = _market_linear_noise_settings(
            noise_model=DEFAULT_NOISE_MODEL,
            arb_frequency=requested_arb_frequency,
        )
    else:
        raise ValueError(
            "compare_reclamm_thermostats only supports "
            "'market_linear' noise and 'arb_only' baselines."
        )

    _NOISE_SETTINGS_CACHE[cache_key] = result
    return result


# Pool configurations to compare
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


def _attach_market_linear_noise_arrays(
    fingerprint,
    noise_cfg,
    market_linear_noise_data,
):
    """Attach preloaded market_linear arrays when the compare flow has them."""
    if market_linear_noise_data is None:
        return
    expected_path = noise_cfg.get("noise_arrays_path")
    if expected_path is None:
        return
    shared_path = os.path.abspath(os.fspath(market_linear_noise_data["arrays_path"]))
    expected_path = os.path.abspath(os.fspath(expected_path))
    if shared_path != expected_path:
        raise ValueError(
            "Shared market_linear noise arrays path does not match "
            f"the resolved compare-run noise path: {shared_path} != {expected_path}"
        )
    fingerprint["noise_base_array"] = market_linear_noise_data["noise_base_array"]
    fingerprint["noise_tvl_coeff_array"] = market_linear_noise_data["noise_tvl_coeff_array"]


def make_fingerprint(cfg, interpolation_method, market_linear_noise_data=None):
    """Build run fingerprint for a given config and interpolation method."""
    cfg = normalize_compare_run_cfg(cfg)
    speed_override = (
        cfg.get("arc_length_speed")
        if interpolation_method == "constant_arc_length"
        else None
    )
    noise_cfg = resolve_reclamm_noise_settings(cfg)
    arb_frequency = get_effective_arb_frequency(cfg, noise_cfg)
    fingerprint = {
        "tokens": cfg["tokens"],
        "rule": "reclamm",
        "startDateString": cfg["start"],
        "endDateString": cfg["end"],
        "initial_pool_value": get_initial_pool_value(cfg),
        "do_arb": True,
        "fees": cfg["fees"],
        "gas_cost": cfg.get(
            "gas_cost",
            DEFAULT_GAS_COST if cfg.get("enable_noise_model", False) else 0.0,
        ),
        "arb_fees": cfg.get("arb_fees", 0.0),
        "protocol_fee_split": cfg.get(
            "protocol_fee_split",
            DEFAULT_PROTOCOL_FEE_SPLIT if cfg.get("enable_noise_model", False) else 0.0,
        ),
        "noise_trader_ratio": noise_cfg.get("noise_trader_ratio", 0.0),
        "reclamm_interpolation_method": interpolation_method,
        "reclamm_arc_length_speed": speed_override,
    }
    if noise_cfg.get("noise_model") is not None:
        fingerprint["noise_model"] = noise_cfg["noise_model"]
    if noise_cfg.get("reclamm_noise_params") is not None:
        fingerprint["reclamm_noise_params"] = noise_cfg["reclamm_noise_params"]
    if noise_cfg.get("noise_arrays_path") is not None:
        fingerprint["noise_arrays_path"] = noise_cfg["noise_arrays_path"]
        _attach_market_linear_noise_arrays(
            fingerprint,
            noise_cfg,
            market_linear_noise_data,
        )
    if arb_frequency is not None:
        fingerprint["arb_frequency"] = arb_frequency
    return fingerprint


def make_params(cfg):
    """Build pool params from config."""
    cfg = normalize_compare_run_cfg(cfg)
    return {
        "price_ratio": jnp.array(cfg["price_ratio"]),
        "centeredness_margin": jnp.array(cfg["centeredness_margin"]),
        "daily_price_shift_base": jnp.array(
            to_daily_price_shift_base(cfg["daily_price_shift_exponent"])
        ),
    }


def load_shared_price_data(configs, root=None):
    """Load the shared historic price panel once for all compare runs."""
    tokens = sorted({token for cfg in configs for token in cfg["tokens"]})
    return get_historic_parquet_data(tokens, cols=["close"], root=root)


def run_comparison(
    cfg,
    price_data=None,
    low_data_mode=False,
    market_linear_noise_data=None,
):
    """Run both interpolation variants, return results dict."""
    params = make_params(cfg)

    results = {}
    for method in INTERPOLATION_METHODS:
        fp = make_fingerprint(
            cfg,
            method,
            market_linear_noise_data=market_linear_noise_data,
        )
        results[method] = do_run_on_historic_data(
            run_fingerprint=fp,
            params=params,
            price_data=price_data,
            low_data_mode=low_data_mode,
        )

    return results


def _set_padded_ylim(ax, series_list, pad_ratio=0.04):
    """Fit the y-axis tightly around the plotted series."""
    flat = [
        np.asarray(series, dtype=float).ravel()
        for series in series_list
        if np.asarray(series).size > 0
    ]
    if not flat:
        return

    values = np.concatenate(flat)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    ymin = float(values.min())
    ymax = float(values.max())
    if np.isclose(ymin, ymax):
        pad = max(abs(ymin) * pad_ratio, 1e-6)
    else:
        pad = (ymax - ymin) * pad_ratio
    ax.set_ylim(ymin - pad, ymax + pad)


def _cache_size(cache):
    """Count memoized final-value cache entries materialised in memory."""
    return len(cache.get("_final_value_cache", {}))


def _comparison_cache_size(cache):
    """Count memoized scalar comparison bundles."""
    return len(cache.get("_comparison_cache", {}))


def _heatmap_forward_cache_scope_slug(cfg):
    """Build a compact cache scope slug for a shared-TVL heatmap run."""
    if cfg is None:
        return "unspecified_tvl"
    return f"tvl_{format_tvl_millions_slug(cfg)}"


def _heatmap_forward_cache_path(cfg):
    """Return the parquet path for persisted scalar forward values."""
    if not HEATMAP_FORWARD_CACHE_ENABLED:
        return None
    return os.path.join(
        HEATMAP_FORWARD_CACHE_ROOT,
        HEATMAP_FORWARD_CACHE_RUN_NAME,
        f"forward_values_{_heatmap_forward_cache_scope_slug(cfg)}.parquet",
    )


def _make_method_cache_hash(key):
    """Build a compact stable digest for a method cache key."""
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()


def _build_persistent_final_value_record(cfg, method, cache_key_hash, final_value):
    """Build one self-describing parquet row for a cached scalar run result."""
    cfg = normalize_compare_run_cfg(cfg)
    noise_cfg = resolve_reclamm_noise_settings(cfg)
    return {
        "cache_key_hash": str(cache_key_hash),
        "final_value": float(final_value),
        "method": str(method),
        "enable_noise_model": bool(cfg.get("enable_noise_model", False)),
        "noise_model": noise_cfg.get("noise_model"),
        "price_ratio": float(cfg["price_ratio"]),
        "centeredness_margin": float(cfg["centeredness_margin"]),
        "daily_price_shift_exponent": float(cfg["daily_price_shift_exponent"]),
        "initial_pool_value": float(get_initial_pool_value(cfg)),
        "arb_frequency": get_effective_arb_frequency(cfg, noise_cfg),
    }


def _load_persistent_final_value_cache(cache):
    """Load persisted scalar forward values from parquet once per sweep cache."""
    if cache.get("_persistent_final_value_cache_loaded"):
        return

    disk_cache = {}
    next_batch_id = 0
    cache_path = cache.get("_persistent_final_value_cache_path")
    if cache_path and os.path.exists(cache_path):
        parquet_files = []
        if os.path.isdir(cache_path):
            parquet_files = [
                os.path.join(cache_path, filename)
                for filename in sorted(os.listdir(cache_path))
                if filename.endswith(".parquet")
            ]
            batch_ids = []
            for filename in os.listdir(cache_path):
                if not (filename.startswith("batch_") and filename.endswith(".parquet")):
                    continue
                token = filename[len("batch_") : -len(".parquet")]
                if token.isdigit():
                    batch_ids.append(int(token))
            next_batch_id = (max(batch_ids) + 1) if batch_ids else 0
        else:
            parquet_files = [cache_path]

        for parquet_file in parquet_files:
            frame = pd.read_parquet(
                parquet_file,
                columns=["cache_key_hash", "final_value"],
            )
            if frame.empty:
                continue
            for row in frame.itertuples(index=False):
                cache_key_hash = str(row.cache_key_hash)
                final_value = float(row.final_value)
                disk_cache[cache_key_hash] = final_value
        print(
            f"Loaded {len(disk_cache)} persisted heatmap forward values from {cache_path}"
        )

    cache["_persistent_final_value_cache"] = disk_cache
    cache["_persistent_final_value_next_batch_id"] = next_batch_id
    cache["_persistent_final_value_cache_loaded"] = True


def flush_sweep_cache(cache, force=False):
    """Persist newly computed scalar forward values to parquet."""
    if not HEATMAP_FORWARD_CACHE_ENABLED:
        return

    pending = cache.get("_pending_persistent_final_values")
    if not pending:
        return
    if not force and len(pending) < HEATMAP_FORWARD_CACHE_FLUSH_EVERY:
        return

    _load_persistent_final_value_cache(cache)
    disk_cache = cache.setdefault("_persistent_final_value_cache", {})
    batch_records = []
    for cache_key_hash, record in pending.items():
        normalized = dict(record)
        normalized["cache_key_hash"] = str(cache_key_hash)
        normalized["final_value"] = float(normalized["final_value"])
        disk_cache[cache_key_hash] = normalized["final_value"]
        batch_records.append(normalized)

    cache_path = cache.get("_persistent_final_value_cache_path")
    if cache_path is None:
        pending.clear()
        return

    if os.path.exists(cache_path) and not os.path.isdir(cache_path):
        raise RuntimeError(
            f"Persistent cache path {cache_path} already exists as a file. "
            "Use a fresh cache namespace for append-only parquet shards."
        )

    os.makedirs(cache_path, exist_ok=True)
    batch_records.sort(key=lambda record: record["cache_key_hash"])
    payload = {
        column: [record.get(column) for record in batch_records]
        for column in PERSISTED_FORWARD_VALUE_COLUMNS
    }
    payload["final_value"] = np.asarray(payload["final_value"], dtype=np.float64)
    frame = pd.DataFrame(payload)
    batch_id = int(cache.setdefault("_persistent_final_value_next_batch_id", 0))
    batch_path = os.path.join(cache_path, f"batch_{batch_id:08d}.parquet")
    cache["_persistent_final_value_next_batch_id"] = batch_id + 1
    frame.to_parquet(batch_path, index=False, compression="zstd")
    print(
        f"Persisted {len(pending)} new heatmap forward values to {batch_path} "
        f"({len(disk_cache)} total cached values)."
    )
    pending.clear()


def make_sweep_cache(
    price_data,
    cache_scope_cfg=None,
    market_linear_noise_data=None,
):
    """Create a shared cache for heatmap and line sweeps."""
    cache = {
        "_shared_price_data": price_data,
        "_shared_market_linear_noise_data": market_linear_noise_data,
        "_final_value_cache": {},
        "_comparison_cache": {},
        "_pending_persistent_final_values": {},
        "_persistent_final_value_cache": {},
        "_persistent_final_value_next_batch_id": 0,
        "_persistent_final_value_cache_loaded": False,
        "_persistent_final_value_cache_path": _heatmap_forward_cache_path(
            cache_scope_cfg
        ),
    }
    return cache


def _missing_artifacts(progress_label, filenames):
    """Report which plot artifacts still need to be generated."""
    missing = [filename for filename in filenames if not os.path.exists(filename)]
    if not missing:
        print(f"[{progress_label}] skipping sweep: all artifacts already exist.")
        return set()

    existing_count = len(filenames) - len(missing)
    if existing_count:
        print(
            f"[{progress_label}] reusing {existing_count}/{len(filenames)} "
            "existing artifacts; generating the missing outputs."
        )
    return set(missing)


def _speed_cache_key(speed):
    """Stable cache token for optional arc-length speed."""
    if speed is None:
        return None
    return round(float(speed), 12)


def _make_method_cache_key(cfg, method):
    """Cache key for a single-method final-value run."""
    cfg = normalize_compare_run_cfg(cfg)
    noise_cfg = resolve_reclamm_noise_settings(cfg)
    arb_frequency = get_effective_arb_frequency(cfg, noise_cfg)
    key = (
        method,
        tuple(str(token) for token in cfg["tokens"]),
        str(cfg["start"]),
        str(cfg["end"]),
        round(float(cfg["fees"]), 12),
        bool(cfg.get("enable_noise_model", False)),
        round(float(cfg["price_ratio"]), 6),
        round(float(cfg["centeredness_margin"]), 6),
        round(float(cfg["daily_price_shift_exponent"]), 6),
        round(get_initial_pool_value(cfg), 2),
        noise_cfg.get("noise_cache_key"),
        None if arb_frequency is None else int(arb_frequency),
        round(
            float(
                cfg.get(
                    "gas_cost",
                    DEFAULT_GAS_COST if cfg.get("enable_noise_model", False) else 0.0,
                )
            ),
            6,
        ),
        round(
            float(
                cfg.get(
                    "protocol_fee_split",
                    DEFAULT_PROTOCOL_FEE_SPLIT if cfg.get("enable_noise_model", False) else 0.0,
                )
            ),
            6,
        ),
    )
    if method == "constant_arc_length":
        key += (_speed_cache_key(cfg.get("arc_length_speed")),)
    return key


def _nearest_price_row(price_data, start_ts):
    """Select the closest available price row to the requested start timestamp."""
    if len(price_data.index) == 0:
        raise ValueError("price_data is empty")

    if isinstance(price_data.index, pd.DatetimeIndex):
        target_ts = start_ts
        index_tz = getattr(price_data.index, "tz", None)
        if index_tz is not None and target_ts.tzinfo is None:
            target_ts = target_ts.tz_localize(index_tz)
        elif index_tz is None and target_ts.tzinfo is not None:
            target_ts = target_ts.tz_convert(None)
        target_value = int(target_ts.value)
        index_values = price_data.index.asi8
    else:
        target_value = int(start_ts.timestamp() * 1000.0)
        index_values = price_data.index.to_numpy(dtype=np.int64)

    row_idx = int(np.searchsorted(index_values, target_value, side="left"))
    if row_idx >= len(index_values):
        row_idx = len(index_values) - 1
    elif row_idx > 0 and index_values[row_idx] != target_value:
        prev_idx = row_idx - 1
        if abs(int(index_values[prev_idx]) - target_value) <= abs(
            int(index_values[row_idx]) - target_value
        ):
            row_idx = prev_idx

    row = price_data.iloc[row_idx]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _make_comparison_cache_key(cfg, launch_final_values):
    """Cache key for scalar heatmap metrics at a single parameter point."""
    noise_cfg = make_noise_variant_cfg(cfg, True)
    arb_only_cfg = make_noise_variant_cfg(cfg, False)
    key = [
        _make_method_cache_key(noise_cfg, "geometric"),
        _make_method_cache_key(arb_only_cfg, "geometric"),
        round(float(launch_final_values["geometric"]), 6),
    ]
    if RUN_CONSTANT_ARC_LENGTH:
        key.extend(
            [
                _make_method_cache_key(noise_cfg, "constant_arc_length"),
                _make_method_cache_key(arb_only_cfg, "constant_arc_length"),
                round(float(launch_final_values["constant_arc_length"]), 6),
            ]
        )
    return tuple(key)


def _run_method_final_value_cached(cfg, method, cache):
    """Memoize final value for a single interpolation method."""
    final_value_cache = cache.setdefault("_final_value_cache", {})
    key = _make_method_cache_key(cfg, method)
    if key in final_value_cache:
        return final_value_cache[key]

    _load_persistent_final_value_cache(cache)
    key_hash = _make_method_cache_hash(key)
    persisted_cache = cache.setdefault("_persistent_final_value_cache", {})
    if key_hash in persisted_cache:
        final_value_cache[key] = persisted_cache[key_hash]
        return final_value_cache[key]

    result = do_run_on_historic_data(
        run_fingerprint=make_fingerprint(
            cfg,
            method,
            market_linear_noise_data=cache.get("_shared_market_linear_noise_data"),
        ),
        params=make_params(cfg),
        price_data=cache["_shared_price_data"],
        low_data_mode=True,
    )
    final_value_cache[key] = float(result["final_value"])
    cache.setdefault("_pending_persistent_final_values", {})[key_hash] = (
        _build_persistent_final_value_record(
            cfg=cfg,
            method=method,
            cache_key_hash=key_hash,
            final_value=final_value_cache[key],
        )
    )
    flush_sweep_cache(cache, force=False)
    del result
    gc.collect()
    return final_value_cache[key]


def extract_comparison_metrics_from_final_values(
    geo_final, arc_final, launch_final_values
):
    """Summarize scalar comparison metrics from final values only."""
    return {
        "efficiency_pct": (arc_final / max(abs(geo_final), 1e-12) - 1.0) * 100.0,
        "launch_geometric_efficiency_pct": (
            arc_final / max(abs(launch_final_values["geometric"]), 1e-12) - 1.0
        )
        * 100.0,
        "geometric_vs_launch_geometric_pct": (
            geo_final / max(abs(launch_final_values["geometric"]), 1e-12) - 1.0
        )
        * 100.0,
        "constant_arc_vs_launch_constant_arc_pct": (
            arc_final
            / max(abs(launch_final_values["constant_arc_length"]), 1e-12)
            - 1.0
        )
        * 100.0,
    }


def _load_required_heatmap_final_values(cfg, cache, metric_keys):
    """Load only the cached final values needed for the requested heatmap metrics."""
    required_sources = set()
    for metric_key in metric_keys:
        required_sources.update(HEATMAP_METRIC_DEPENDENCIES[metric_key])

    if not RUN_CONSTANT_ARC_LENGTH and any(
        source.endswith("constant_arc") for source in required_sources
    ):
        raise ValueError(
            "Constant-arc heatmap metric requested while RUN_CONSTANT_ARC_LENGTH=False"
        )

    final_values = {}
    noise_cfg = None
    arb_only_cfg = None

    if any(source.startswith("noise_") for source in required_sources):
        noise_cfg = make_noise_variant_cfg(cfg, True)
    if any(source.startswith("arb_") for source in required_sources):
        arb_only_cfg = make_noise_variant_cfg(cfg, False)

    if "noise_geometric" in required_sources:
        final_values["noise_geometric"] = _run_method_final_value_cached(
            noise_cfg,
            "geometric",
            cache,
        )
    if "noise_constant_arc" in required_sources:
        final_values["noise_constant_arc"] = _run_method_final_value_cached(
            noise_cfg,
            "constant_arc_length",
            cache,
        )
    if "arb_geometric" in required_sources:
        final_values["arb_geometric"] = _run_method_final_value_cached(
            arb_only_cfg,
            "geometric",
            cache,
        )
    if "arb_constant_arc" in required_sources:
        final_values["arb_constant_arc"] = _run_method_final_value_cached(
            arb_only_cfg,
            "constant_arc_length",
            cache,
        )
    return final_values


def extract_heatmap_metrics_from_mode_final_values(
    metric_keys,
    final_values,
    launch_final_values,
):
    """Collect the requested scalar heatmap metrics from cached final values."""
    metrics = {}

    if "efficiency_pct" in metric_keys:
        metrics["efficiency_pct"] = (
            final_values["noise_constant_arc"]
            / max(abs(final_values["noise_geometric"]), 1e-12)
            - 1.0
        ) * 100.0

    if "launch_geometric_efficiency_pct" in metric_keys:
        metrics["launch_geometric_efficiency_pct"] = (
            final_values["noise_constant_arc"]
            / max(abs(launch_final_values["geometric"]), 1e-12)
            - 1.0
        ) * 100.0

    if "geometric_vs_launch_geometric_pct" in metric_keys:
        metrics["geometric_vs_launch_geometric_pct"] = (
            final_values["noise_geometric"]
            / max(abs(launch_final_values["geometric"]), 1e-12)
            - 1.0
        ) * 100.0

    if "constant_arc_vs_launch_constant_arc_pct" in metric_keys:
        metrics["constant_arc_vs_launch_constant_arc_pct"] = (
            final_values["noise_constant_arc"]
            / max(abs(launch_final_values["constant_arc_length"]), 1e-12)
            - 1.0
        ) * 100.0

    if "noise_geometric_final_value_musd" in metric_keys:
        metrics["noise_geometric_final_value_musd"] = (
            final_values["noise_geometric"] / 1e6
        )

    if "noise_constant_arc_final_value_musd" in metric_keys:
        metrics["noise_constant_arc_final_value_musd"] = (
            final_values["noise_constant_arc"] / 1e6
        )

    if "noise_vs_arb_geometric_improvement_pct" in metric_keys:
        metrics["noise_vs_arb_geometric_improvement_pct"] = (
            final_values["noise_geometric"]
            / max(abs(final_values["arb_geometric"]), 1e-12)
            - 1.0
        ) * 100.0

    if "noise_vs_arb_constant_arc_improvement_pct" in metric_keys:
        metrics["noise_vs_arb_constant_arc_improvement_pct"] = (
            final_values["noise_constant_arc"]
            / max(abs(final_values["arb_constant_arc"]), 1e-12)
            - 1.0
        ) * 100.0

    return metrics


def extract_comparison_metrics(results, launch_final_values):
    """Summarize scalar heatmap metrics for a pair of runs."""
    geo = results["geometric"]
    arc = results["constant_arc_length"]

    geo_final = float(geo["final_value"])
    arc_final = float(arc["final_value"])

    return extract_comparison_metrics_from_final_values(
        geo_final,
        arc_final,
        launch_final_values=launch_final_values,
    )


def run_comparison_cached(cfg, cache, launch_final_values, metric_keys):
    """Memoize scalar heatmap metrics across heatmap sweeps."""
    requested_metric_keys = tuple(dict.fromkeys(metric_keys))
    comparison_cache = cache.setdefault("_comparison_cache", {})
    cache_key = _make_comparison_cache_key(cfg, launch_final_values)
    cached_metrics = comparison_cache.setdefault(cache_key, {})
    missing_metric_keys = [
        metric_key for metric_key in requested_metric_keys if metric_key not in cached_metrics
    ]
    if missing_metric_keys:
        final_values = _load_required_heatmap_final_values(
            cfg,
            cache,
            missing_metric_keys,
        )
        cached_metrics.update(
            extract_heatmap_metrics_from_mode_final_values(
                missing_metric_keys,
                final_values,
                launch_final_values=launch_final_values,
            )
        )
    return {
        metric_key: cached_metrics[metric_key] for metric_key in requested_metric_keys
    }


def build_heatmap_matrices(
    x_values,
    y_values,
    x_key,
    y_key,
    base_cfg,
    metric_keys,
    cache,
    progress_label,
    launch_final_values,
):
    """Evaluate multiple metrics over a 2D parameter grid in one pass."""
    data = {
        metric_key: np.zeros((len(y_values), len(x_values)), dtype=float)
        for metric_key in metric_keys
    }
    total_points = len(y_values) * len(x_values)

    print(
        f"[{progress_label}] start: {len(y_values)} rows x {len(x_values)} cols "
        f"= {total_points} parameter points"
    )

    for yi, y_value in enumerate(y_values):
        final_cache_before_row = _cache_size(cache)
        comparison_cache_before_row = _comparison_cache_size(cache)
        for xi, x_value in enumerate(x_values):
            cfg = dict(base_cfg)
            cfg[x_key] = float(x_value)
            cfg[y_key] = float(y_value)
            metrics = run_comparison_cached(
                cfg,
                cache,
                launch_final_values=launch_final_values,
                metric_keys=metric_keys,
            )
            for metric_key in metric_keys:
                data[metric_key][yi, xi] = metrics[metric_key]

        completed_points = (yi + 1) * len(x_values)
        row_new_final_entries = _cache_size(cache) - final_cache_before_row
        row_new_comparisons = (
            _comparison_cache_size(cache) - comparison_cache_before_row
        )
        row_pct = completed_points / total_points * 100.0
        flush_sweep_cache(cache, force=True)
        print(
            f"[{progress_label}] row {yi + 1}/{len(y_values)} complete "
            f"({y_key}={float(y_value):.4f}, {completed_points}/{total_points} "
            f"points, {row_pct:.1f}%, {row_new_final_entries} new final-value cache entries, "
            f"{row_new_comparisons} new comparison bundles)"
        )

    print(
        f"[{progress_label}] done: "
        + ", ".join(
            (
                f"{metric_key} min={float(np.nanmin(data[metric_key])):.4f}, "
                f"max={float(np.nanmax(data[metric_key])):.4f}"
            )
            for metric_key in metric_keys
        )
        + (
            f", final_value_cache_size={_cache_size(cache)}, "
            f"comparison_cache_size={_comparison_cache_size(cache)}"
        )
    )

    return data


def build_metric_curve(
    x_values,
    x_key,
    base_cfg,
    metric_key,
    cache,
    launch_final_values,
):
    """Evaluate one metric over a 1D sweep."""
    data = np.zeros(len(x_values), dtype=float)
    for xi, x_value in enumerate(x_values):
        cfg = dict(base_cfg)
        cfg[x_key] = float(x_value)
        metrics = run_comparison_cached(
            cfg,
            cache,
            launch_final_values=launch_final_values,
            metric_keys=(metric_key,),
        )
        data[xi] = metrics[metric_key]
    flush_sweep_cache(cache, force=True)
    return data


def _compute_axis_edges(values, scale="linear"):
    """Convert axis centers to cell edges for pcolormesh."""
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        if scale == "log":
            return np.array([values[0] / np.sqrt(10.0), values[0] * np.sqrt(10.0)])
        pad = max(abs(values[0]) * 0.5, 1.0)
        return np.array([values[0] - pad, values[0] + pad])

    if scale == "log":
        log_values = np.log10(values)
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (log_values[:-1] + log_values[1:])
        edges[0] = log_values[0] - 0.5 * (log_values[1] - log_values[0])
        edges[-1] = log_values[-1] + 0.5 * (log_values[-1] - log_values[-2])
        return 10.0 ** edges

    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def build_fixed_slice_variants(values):
    """Pick four representative quarter-range slices from a sweep grid."""
    values = np.asarray(values, dtype=float)
    if values.size < len(FIXED_SLICE_FRACTIONS):
        raise ValueError("Need at least four grid points to build fixed slices")

    variants = []
    used_indices = set()
    for idx, fraction in enumerate(FIXED_SLICE_FRACTIONS):
        target_index = int(round(fraction * (values.size - 1)))
        while target_index in used_indices and target_index + 1 < values.size:
            target_index += 1
        while target_index in used_indices and target_index - 1 >= 0:
            target_index -= 1
        if target_index in used_indices:
            raise ValueError("Could not build four unique fixed slices from sweep grid")
        used_indices.add(target_index)
        variants.append(
            {
                "index": target_index,
                "fraction": fraction,
                "label": FIXED_SLICE_LABELS[idx],
                "slug": f"q{idx + 1}",
                "value": float(values[target_index]),
            }
        )
    return variants


def _pair_slice_suffix(pair, slice_variant):
    """Build a stable artifact suffix for a pairwise fixed-variable slice."""
    return f"{pair['slug']}_{pair['fixed_slug']}_{slice_variant['slug']}"


def _build_heatmap_norm(
    data_arrays,
    center_zero,
    color_norm=None,
    symlog_linthresh=None,
):
    """Build a color normalizer shared by 2D and 3D heatmaps."""
    finite_parts = []
    for data in data_arrays:
        finite = np.asarray(data, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            finite_parts.append(finite)
    finite = np.concatenate(finite_parts) if finite_parts else np.array([], dtype=float)

    if center_zero:
        if finite.size == 0:
            vmax = 1.0
        else:
            vmax = max(abs(float(finite.min())), abs(float(finite.max())), 1e-9)
        if (
            color_norm == "symlog"
            and symlog_linthresh is not None
            and vmax > symlog_linthresh
        ):
            return SymLogNorm(
                linthresh=symlog_linthresh,
                linscale=1.0,
                vmin=-vmax,
                vmax=vmax,
                base=10.0,
            )
        return TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if np.isclose(vmin, vmax):
            pad = max(abs(vmin) * 0.01, 1e-9)
            vmin -= pad
            vmax += pad
    return Normalize(vmin=vmin, vmax=vmax)


def get_pair_heatmap_metric_specs():
    """Return the standard thermostat pairwise heatmap metrics."""
    metric_specs = [
        {
            "key": "efficiency_pct",
            "title": "Efficiency vs heatmap geometric",
            "colorbar_label": "Const Arc - heatmap Geo (% of heatmap geometric final value)",
            "slug": "efficiency",
            "center_zero": True,
            "cmap": "RdYlGn",
        },
        {
            "key": "launch_geometric_efficiency_pct",
            "title": "Efficiency vs launch-style geometric",
            "colorbar_label": "Const Arc - launch Geo (% of launch geometric final value)",
            "slug": "launch_geometric_efficiency",
            "center_zero": True,
            "cmap": "RdYlGn",
        },
        {
            "key": "geometric_vs_launch_geometric_pct",
            "title": "Geometric tuning vs launch-style geometric",
            "colorbar_label": "Candidate Geo - launch Geo (% of launch geometric final value)",
            "slug": "geometric_vs_launch_geometric",
            "center_zero": True,
            "cmap": "RdYlGn",
        },
        {
            "key": "constant_arc_vs_launch_constant_arc_pct",
            "title": "Const arc tuning vs launch-style const arc",
            "colorbar_label": "Candidate Const Arc - launch Const Arc (% of launch const arc final value)",
            "slug": "constant_arc_vs_launch_constant_arc",
            "center_zero": True,
            "cmap": "RdYlGn",
        },
        {
            "key": "noise_geometric_final_value_musd",
            "title": "Geometric final value with noise model",
            "colorbar_label": "Geometric final value with noise model ($M)",
            "slug": "noise_geometric_final_value",
            "center_zero": False,
            "cmap": "viridis",
        },
        {
            "key": "noise_constant_arc_final_value_musd",
            "title": "Const arc final value with noise model",
            "colorbar_label": "Const Arc final value with noise model ($M)",
            "slug": "noise_constant_arc_final_value",
            "center_zero": False,
            "cmap": "viridis",
        },
        {
            "key": "noise_vs_arb_geometric_improvement_pct",
            "title": "Noise-model improvement over arb-only (geometric)",
            "colorbar_label": "Noise-model Geo - arb-only Geo (% of arb-only final value)",
            "slug": "noise_vs_arb_geometric_improvement",
            "center_zero": True,
            "cmap": "RdYlGn",
        },
        {
            "key": "noise_vs_arb_constant_arc_improvement_pct",
            "title": "Noise-model improvement over arb-only (const arc)",
            "colorbar_label": "Noise-model Const Arc - arb-only Const Arc (% of arb-only final value)",
            "slug": "noise_vs_arb_constant_arc_improvement",
            "center_zero": True,
            "cmap": "RdYlGn",
        },
    ]
    for spec in metric_specs:
        if spec["center_zero"]:
            spec["color_norm"] = CENTER_ZERO_HEATMAP_COLOR_NORM
            spec["symlog_linthresh"] = CENTER_ZERO_HEATMAP_SYMLOG_LINTHRESH
            spec["artifact_tag"] = CENTER_ZERO_HEATMAP_COLOR_TAG
    if not RUN_CONSTANT_ARC_LENGTH:
        metric_specs = [
            spec
            for spec in metric_specs
            if spec["key"] in GEOMETRIC_ONLY_HEATMAP_METRIC_KEYS
        ]
    return metric_specs


def get_pair_heatmap_specs(base_cfg):
    """Return the three pairwise thermostat heatmap families plus slice settings."""
    fixed_slice_variants = {
        "price_ratio": build_fixed_slice_variants(HEATMAP_PRICE_RATIOS),
        "centeredness_margin": build_fixed_slice_variants(HEATMAP_MARGINS),
        "daily_price_shift_exponent": build_fixed_slice_variants(
            HEATMAP_SHIFT_EXPONENTS
        ),
    }
    return [
        {
            "slug": "price_ratio_vs_margin",
            "x_values": HEATMAP_PRICE_RATIOS,
            "y_values": HEATMAP_MARGINS,
            "x_key": "price_ratio",
            "y_key": "centeredness_margin",
            "x_label": "Price ratio",
            "y_label": "Centeredness margin",
            "xticks": PRICE_RATIO_TICKS,
            "yticks": MARGIN_TICKS,
            "fixed_key": "daily_price_shift_exponent",
            "fixed_label": "Shift exponent",
            "fixed_slug": "shift_exp",
            "fixed_slices": fixed_slice_variants["daily_price_shift_exponent"],
        },
        {
            "slug": "shift_exp_vs_margin",
            "x_values": HEATMAP_SHIFT_EXPONENTS,
            "y_values": HEATMAP_MARGINS,
            "x_key": "daily_price_shift_exponent",
            "y_key": "centeredness_margin",
            "x_label": "Shift exponent",
            "y_label": "Centeredness margin",
            "xticks": SHIFT_EXPONENT_TICKS,
            "yticks": MARGIN_TICKS,
            "fixed_key": "price_ratio",
            "fixed_label": "Price ratio",
            "fixed_slug": "price_ratio",
            "fixed_slices": fixed_slice_variants["price_ratio"],
        },
        {
            "slug": "price_ratio_vs_shift_exp",
            "x_values": HEATMAP_PRICE_RATIOS,
            "y_values": HEATMAP_SHIFT_EXPONENTS,
            "x_key": "price_ratio",
            "y_key": "daily_price_shift_exponent",
            "x_label": "Price ratio",
            "y_label": "Shift exponent",
            "xticks": PRICE_RATIO_TICKS,
            "yticks": SHIFT_EXPONENT_TICKS,
            "fixed_key": "centeredness_margin",
            "fixed_label": "Centeredness margin",
            "fixed_slug": "margin",
            "fixed_slices": fixed_slice_variants["centeredness_margin"],
        },
    ]


def plot_heatmap(
    data,
    x_values,
    y_values,
    x_label,
    y_label,
    title,
    colorbar_label,
    filename,
    xticks=None,
    yticks=None,
    xscale="linear",
    center_zero=True,
    cmap=None,
    color_norm=None,
    symlog_linthresh=None,
):
    """Render and save a single heatmap."""
    norm = _build_heatmap_norm(
        [data],
        center_zero=center_zero,
        color_norm=color_norm,
        symlog_linthresh=symlog_linthresh,
    )
    cmap_name = cmap or ("RdYlGn" if center_zero else "viridis")

    x_edges = _compute_axis_edges(x_values, scale=xscale)
    y_edges = _compute_axis_edges(y_values, scale="linear")

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        data,
        cmap=cmap_name,
        norm=norm,
        shading="auto",
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xticks(np.asarray(xticks if xticks is not None else x_values, dtype=float))
    ax.set_yticks(np.asarray(yticks if yticks is not None else y_values, dtype=float))
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close(fig)


def plot_three_variable_heatmap_3d(
    price_margin_data,
    shift_margin_data,
    price_shift_data,
    fixed_price_ratio,
    fixed_margin,
    fixed_shift_exponent,
    title,
    colorbar_label,
    filename,
    center_zero=True,
    cmap=None,
    color_norm=None,
    symlog_linthresh=None,
):
    """Render orthogonal 3D heatmap surfaces across the three thermostat variables."""
    norm = _build_heatmap_norm(
        [price_margin_data, shift_margin_data, price_shift_data],
        center_zero=center_zero,
        color_norm=color_norm,
        symlog_linthresh=symlog_linthresh,
    )
    cmap_name = cmap or ("RdYlGn" if center_zero else "viridis")
    cmap_obj = plt.get_cmap(cmap_name)

    price_margin_x, price_margin_y = np.meshgrid(HEATMAP_PRICE_RATIOS, HEATMAP_MARGINS)
    price_margin_z = np.full_like(price_margin_x, fixed_shift_exponent, dtype=float)

    shift_margin_z, shift_margin_y = np.meshgrid(
        HEATMAP_SHIFT_EXPONENTS,
        HEATMAP_MARGINS,
    )
    shift_margin_x = np.full_like(shift_margin_z, fixed_price_ratio, dtype=float)

    price_shift_x, price_shift_z = np.meshgrid(
        HEATMAP_PRICE_RATIOS,
        HEATMAP_SHIFT_EXPONENTS,
    )
    price_shift_y = np.full_like(price_shift_x, fixed_margin, dtype=float)

    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.plot_surface(
        price_margin_x,
        price_margin_y,
        price_margin_z,
        facecolors=cmap_obj(norm(np.asarray(price_margin_data, dtype=float))),
        shade=False,
    )
    ax.plot_surface(
        shift_margin_x,
        shift_margin_y,
        shift_margin_z,
        facecolors=cmap_obj(norm(np.asarray(shift_margin_data, dtype=float))),
        shade=False,
    )
    ax.plot_surface(
        price_shift_x,
        price_shift_y,
        price_shift_z,
        facecolors=cmap_obj(norm(np.asarray(price_shift_data, dtype=float))),
        shade=False,
    )

    ax.set_xlim(float(HEATMAP_PRICE_RATIOS.min()), float(HEATMAP_PRICE_RATIOS.max()))
    ax.set_ylim(float(HEATMAP_MARGINS.min()), float(HEATMAP_MARGINS.max()))
    ax.set_zlim(
        float(HEATMAP_SHIFT_EXPONENTS.min()),
        float(HEATMAP_SHIFT_EXPONENTS.max()),
    )
    ax.set_xlabel("Price ratio")
    ax.set_ylabel("Centeredness margin")
    ax.set_zlabel("Shift exponent")
    ax.set_xticks(PRICE_RATIO_TICKS)
    ax.set_yticks(MARGIN_TICKS[::2])
    ax.set_zticks(SHIFT_EXPONENT_TICKS)
    ax.set_title(title)
    ax.grid(False)
    ax.view_init(elev=THREE_D_VIEW_ELEVATION, azim=THREE_D_VIEW_AZIMUTH)
    try:
        ax.set_box_aspect(
            (
                float(HEATMAP_PRICE_RATIOS.max() - HEATMAP_PRICE_RATIOS.min()),
                float(HEATMAP_MARGINS.max() - HEATMAP_MARGINS.min()),
                float(
                    HEATMAP_SHIFT_EXPONENTS.max() - HEATMAP_SHIFT_EXPONENTS.min()
                ),
            )
        )
    except AttributeError:
        pass

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1, shrink=0.82)
    cbar.set_label(colorbar_label)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close(fig)


def plot_arc_speed_line_chart(
    data,
    x_values,
    y_values,
    y_label,
    title,
    filename,
    launch_curve,
    launch_auto_speed=None,
):
    """Plot thin multi-series efficiency lines over the arc-speed sweep."""
    fig, ax = plt.subplots(figsize=(10.5, 5.75))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.0, 1.0, len(y_values)))
    plotted_series = []

    for yi, (y_value, color) in enumerate(zip(y_values, colors)):
        series = np.asarray(data[yi], dtype=float)
        plotted_series.append(series)
        ax.plot(
            x_values,
            series,
            color=color,
            linewidth=SWEEP_LINE_WIDTH,
            alpha=0.8,
        )

    launch_curve = np.asarray(launch_curve, dtype=float)
    plotted_series.append(launch_curve)
    ax.plot(
        x_values,
        launch_curve,
        color="black",
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=0.9,
        label="Current launch config",
    )
    if launch_auto_speed is not None:
        ax.axvline(
            float(launch_auto_speed),
            color="black",
            ls=":",
            linewidth=0.8,
            alpha=0.7,
            label="Launch auto-cal speed",
        )

    ax.axhline(0.0, color="gray", ls="--", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks(ARC_LENGTH_SPEED_TICKS)
    ax.set_xlabel("Arc-length speed")
    ax.set_ylabel("Efficiency vs geometric (%)")
    ax.set_title(title)
    _set_padded_ylim(ax, plotted_series, pad_ratio=0.08)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    sm = ScalarMappable(
        norm=Normalize(vmin=float(np.min(y_values)), vmax=float(np.max(y_values))),
        cmap=cmap,
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(y_label)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close(fig)


def generate_heatmaps(base_cfg, price_data, launch_final_values, cache=None):
    """Generate pairwise heatmaps for thermostat tuning and noise-vs-arb effects."""
    owns_cache = cache is None
    if cache is None:
        cache = make_sweep_cache(price_data, cache_scope_cfg=base_cfg)
    metric_specs = get_pair_heatmap_metric_specs()
    pair_specs = get_pair_heatmap_specs(base_cfg)
    metric_spec_map = {spec["key"]: spec for spec in metric_specs}
    slice_count = len(pair_specs[0]["fixed_slices"]) if pair_specs else 0

    if RUN_CONSTANT_ARC_LENGTH:
        print(
            "Using launch-style benchmarks "
            f"Geo=${launch_final_values['geometric']:,.0f}, "
            f"Const Arc=${launch_final_values['constant_arc_length']:,.0f}, "
            f"TVL={format_tvl_millions_label(base_cfg)}."
        )
        print(
            "Running {count} heatmap pair sweeps sequentially "
            "(3 pair grids x {slice_count} fixed-variable quarter slices; "
            "cached noise-model runs are reused across the absolute, launch, "
            "and arb-only comparison outputs).".format(
                count=len(pair_specs) * slice_count,
                slice_count=slice_count,
            )
        )
    else:
        print(
            "Using launch-style geometric benchmark "
            f"Geo=${launch_final_values['geometric']:,.0f}, "
            f"TVL={format_tvl_millions_label(base_cfg)}."
        )
        print(
            "RUN_CONSTANT_ARC_LENGTH=False, so only geometric heatmaps will be generated "
            f"across {len(pair_specs) * slice_count} fixed-variable pair sweeps."
        )

    for pair in pair_specs:
        for slice_variant in pair["fixed_slices"]:
            pair_suffix = _pair_slice_suffix(pair, slice_variant)
            slice_cfg = dict(base_cfg)
            slice_cfg[pair["fixed_key"]] = float(slice_variant["value"])
            output_files = {
                spec["key"]: heatmap_artifact_filename(
                    spec,
                    base_cfg,
                    suffix=pair_suffix,
                )
                for spec in metric_specs
            }
            missing_files = _missing_artifacts(
                pair_suffix,
                list(output_files.values()),
            )
            if not missing_files:
                continue

            missing_metric_keys = [
                spec["key"]
                for spec in metric_specs
                if output_files[spec["key"]] in missing_files
            ]
            data_by_metric = build_heatmap_matrices(
                x_values=pair["x_values"],
                y_values=pair["y_values"],
                x_key=pair["x_key"],
                y_key=pair["y_key"],
                base_cfg=slice_cfg,
                metric_keys=missing_metric_keys,
                cache=cache,
                progress_label=pair_suffix,
                launch_final_values=launch_final_values,
            )
            print(f"[{pair_suffix}] plotting missing heatmaps...")
            for metric_key in missing_metric_keys:
                spec = metric_spec_map[metric_key]
                plot_heatmap(
                    data=data_by_metric[metric_key],
                    x_values=pair["x_values"],
                    y_values=pair["y_values"],
                    x_label=pair["x_label"],
                    y_label=pair["y_label"],
                    title=(
                        f"{spec['title']}: {pair['fixed_label']} {slice_variant['label']} "
                        f"slice fixed at {format_heatmap_param_value(slice_variant['value'])} | "
                        f"TVL {format_tvl_millions_label(base_cfg)}"
                    ),
                    colorbar_label=spec["colorbar_label"],
                    filename=output_files[metric_key],
                    xticks=pair["xticks"],
                    yticks=pair["yticks"],
                    center_zero=spec["center_zero"],
                    cmap=spec["cmap"],
                    color_norm=spec.get("color_norm"),
                    symlog_linthresh=spec.get("symlog_linthresh"),
                )
            del data_by_metric
            gc.collect()

    if owns_cache:
        flush_sweep_cache(cache, force=True)
        cache.clear()
        gc.collect()
        print("Released heatmap metric cache.")


def generate_three_variable_3d_heatmaps(
    base_cfg,
    price_data,
    launch_final_values,
    cache=None,
):
    """Render 3D thermostat heatmaps from the three pairwise quarter slices."""
    owns_cache = cache is None
    if cache is None:
        cache = make_sweep_cache(price_data, cache_scope_cfg=base_cfg)

    metric_specs = get_pair_heatmap_metric_specs()
    metric_spec_map = {spec["key"]: spec for spec in metric_specs}
    pair_specs = get_pair_heatmap_specs(base_cfg)
    pair_by_fixed_key = {pair["fixed_key"]: pair for pair in pair_specs}
    price_margin_pair = pair_by_fixed_key["daily_price_shift_exponent"]
    shift_margin_pair = pair_by_fixed_key["price_ratio"]
    price_shift_pair = pair_by_fixed_key["centeredness_margin"]
    slice_count = len(price_margin_pair["fixed_slices"])

    def build_pair_slice_data(pair, slice_variant, metric_keys):
        pair_cfg = dict(base_cfg)
        pair_cfg[pair["fixed_key"]] = float(slice_variant["value"])
        return build_heatmap_matrices(
            x_values=pair["x_values"],
            y_values=pair["y_values"],
            x_key=pair["x_key"],
            y_key=pair["y_key"],
            base_cfg=pair_cfg,
            metric_keys=metric_keys,
            cache=cache,
            progress_label=f"3d_{_pair_slice_suffix(pair, slice_variant)}",
            launch_final_values=launch_final_values,
        )

    print(
        "\nGenerating 3D thermostat heatmaps "
        f"({slice_count} quarter-slice variants, TVL={format_tvl_millions_label(base_cfg)})..."
    )

    for slice_idx in range(slice_count):
        shift_slice = price_margin_pair["fixed_slices"][slice_idx]
        price_slice = shift_margin_pair["fixed_slices"][slice_idx]
        margin_slice = price_shift_pair["fixed_slices"][slice_idx]
        slice_slug = shift_slice["slug"]
        slice_label = shift_slice["label"]

        output_files = {
            spec["key"]: three_d_heatmap_artifact_filename(
                spec,
                base_cfg,
                suffix=f"slice_{slice_slug}",
            )
            for spec in metric_specs
        }
        missing_files = _missing_artifacts(
            f"3d_slice_{slice_slug}",
            list(output_files.values()),
        )
        if not missing_files:
            continue

        missing_metric_keys = [
            spec["key"]
            for spec in metric_specs
            if output_files[spec["key"]] in missing_files
        ]
        price_margin_data = build_pair_slice_data(
            price_margin_pair,
            shift_slice,
            missing_metric_keys,
        )
        shift_margin_data = build_pair_slice_data(
            shift_margin_pair,
            price_slice,
            missing_metric_keys,
        )
        price_shift_data = build_pair_slice_data(
            price_shift_pair,
            margin_slice,
            missing_metric_keys,
        )

        for metric_key in missing_metric_keys:
            spec = metric_spec_map[metric_key]
            plot_three_variable_heatmap_3d(
                price_margin_data=price_margin_data[metric_key],
                shift_margin_data=shift_margin_data[metric_key],
                price_shift_data=price_shift_data[metric_key],
                fixed_price_ratio=float(price_slice["value"]),
                fixed_margin=float(margin_slice["value"]),
                fixed_shift_exponent=float(shift_slice["value"]),
                title=(
                    f"{spec['title']} 3D {slice_label} slice | TVL {format_tvl_millions_label(base_cfg)}\n"
                    f"price_ratio={format_heatmap_param_value(price_slice['value'])}, "
                    f"margin={format_heatmap_param_value(margin_slice['value'])}, "
                    f"shift_exp={format_heatmap_param_value(shift_slice['value'])}"
                ),
                colorbar_label=spec["colorbar_label"],
                filename=output_files[metric_key],
                center_zero=spec["center_zero"],
                cmap=spec["cmap"],
                color_norm=spec.get("color_norm"),
                symlog_linthresh=spec.get("symlog_linthresh"),
            )

        del price_margin_data, shift_margin_data, price_shift_data
        gc.collect()

    if owns_cache:
        flush_sweep_cache(cache, force=True)
        cache.clear()
        gc.collect()
        print("Released 3D heatmap cache.")


def compute_auto_calibrated_arc_length_speed(cfg, price_data):
    """Compute the launch/reference auto-calibrated speed for a config."""
    start_ts = pd.Timestamp(cfg["start"])
    row = _nearest_price_row(price_data, start_ts)

    if isinstance(price_data.columns, pd.MultiIndex):
        initial_price_values = [
            float(row[(token, "close")])
            for token in cfg["tokens"]
        ]
    else:
        initial_price_values = [
            float(row[f"close_{token}"])
            for token in cfg["tokens"]
        ]

    initial_prices = jnp.array(initial_price_values, dtype=jnp.float64)
    initial_reserves, Va, Vb = initialise_reclamm_reserves(
        get_initial_pool_value(cfg),
        initial_prices,
        float(cfg["price_ratio"]),
    )
    market_price_0 = float(initial_prices[0] / initial_prices[1])
    sqrt_Q = jnp.sqrt(
        compute_price_ratio(
            initial_reserves[0],
            initial_reserves[1],
            Va,
            Vb,
        )
    )
    return float(
        calibrate_arc_length_speed(
            initial_reserves[0],
            initial_reserves[1],
            Va,
            Vb,
            to_daily_price_shift_base(float(cfg["daily_price_shift_exponent"])),
            60.0,
            sqrt_Q,
            market_price_0,
            centeredness_margin=float(cfg["centeredness_margin"]),
        )
    )


def generate_arc_speed_efficiency_artifacts(
    base_cfg,
    launch_cfg,
    price_data,
    launch_final_values,
    cache=None,
):
    """Generate arc-speed heatmaps plus the existing efficiency line charts."""
    if not RUN_CONSTANT_ARC_LENGTH:
        print("\nSkipping arc-speed heatmaps because RUN_CONSTANT_ARC_LENGTH=False.")
        return
    owns_cache = cache is None
    if cache is None:
        cache = make_sweep_cache(price_data, cache_scope_cfg=base_cfg)
    launch_auto_speed = compute_auto_calibrated_arc_length_speed(launch_cfg, price_data)
    heatmap_metric_specs = [
        {
            "key": "efficiency_pct",
            "title": "Efficiency vs geometric",
            "colorbar_label": "Const Arc - heatmap Geo (% of heatmap geometric final value)",
            "slug": "efficiency",
            "center_zero": True,
            "cmap": "RdYlGn",
        },
        {
            "key": "noise_constant_arc_final_value_musd",
            "title": "Const arc final value with noise model",
            "colorbar_label": "Const Arc final value with noise model ($M)",
            "slug": "noise_constant_arc_final_value",
            "center_zero": False,
            "cmap": "viridis",
        },
        {
            "key": "noise_vs_arb_constant_arc_improvement_pct",
            "title": "Noise-model improvement over arb-only (const arc)",
            "colorbar_label": "Noise-model Const Arc - arb-only Const Arc (% of arb-only final value)",
            "slug": "noise_vs_arb_constant_arc_improvement",
            "center_zero": True,
            "cmap": "RdYlGn",
        },
    ]
    for spec in heatmap_metric_specs:
        if spec["center_zero"]:
            spec["color_norm"] = CENTER_ZERO_HEATMAP_COLOR_NORM
            spec["symlog_linthresh"] = CENTER_ZERO_HEATMAP_SYMLOG_LINTHRESH
            spec["artifact_tag"] = CENTER_ZERO_HEATMAP_COLOR_TAG
    pair_specs = [
        {
            "slug": "arc_speed_vs_price_ratio",
            "x_values": HEATMAP_ARC_LENGTH_SPEEDS,
            "y_values": HEATMAP_PRICE_RATIOS,
            "x_key": "arc_length_speed",
            "y_key": "price_ratio",
            "x_label": "Arc-length speed",
            "y_label": "Price ratio",
            "title_suffix": (
                f"margin fixed at {base_cfg['centeredness_margin']:.2f}, "
                f"shift_exp fixed at {base_cfg['daily_price_shift_exponent']:.2f}"
            ),
            "xticks": ARC_LENGTH_SPEED_TICKS,
            "yticks": PRICE_RATIO_TICKS,
        },
        {
            "slug": "arc_speed_vs_margin",
            "x_values": HEATMAP_ARC_LENGTH_SPEEDS,
            "y_values": HEATMAP_MARGINS,
            "x_key": "arc_length_speed",
            "y_key": "centeredness_margin",
            "x_label": "Arc-length speed",
            "y_label": "Centeredness margin",
            "title_suffix": (
                f"price_ratio fixed at {base_cfg['price_ratio']:.2f}, "
                f"shift_exp fixed at {base_cfg['daily_price_shift_exponent']:.2f}"
            ),
            "xticks": ARC_LENGTH_SPEED_TICKS,
            "yticks": MARGIN_TICKS
        },
        {
            "slug": "arc_speed_vs_shift_exp",
            "x_values": HEATMAP_ARC_LENGTH_SPEEDS,
            "y_values": HEATMAP_SHIFT_EXPONENTS,
            "x_key": "arc_length_speed",
            "y_key": "daily_price_shift_exponent",
            "x_label": "Arc-length speed",
            "y_label": "Shift exponent",
            "title_suffix": (
                f"price_ratio fixed at {base_cfg['price_ratio']:.2f}, "
                f"margin fixed at {base_cfg['centeredness_margin']:.2f}"
            ),
            "xticks": ARC_LENGTH_SPEED_TICKS,
            "yticks": SHIFT_EXPONENT_TICKS,
        },
    ]
    metric_spec_map = {spec["key"]: spec for spec in heatmap_metric_specs}

    print(
        "\nGenerating arc-speed heatmaps and line charts "
        f"(launch auto-cal speed={launch_auto_speed:.3e}, TVL={format_tvl_millions_label(base_cfg)})..."
    )

    for pair in pair_specs:
        heatmap_files = {
            spec["key"]: heatmap_artifact_filename(
                spec,
                base_cfg,
                suffix=pair["slug"],
            )
            for spec in heatmap_metric_specs
        }
        line_filename = tvl_artifact_filename(
            "reclamm_line_efficiency",
            base_cfg,
            suffix=pair["slug"],
        )
        missing_files = _missing_artifacts(
            pair["slug"],
            list(heatmap_files.values()) + [line_filename],
        )
        if not missing_files:
            continue

        missing_metric_keys = [
            spec["key"]
            for spec in heatmap_metric_specs
            if heatmap_files[spec["key"]] in missing_files
        ]
        if line_filename in missing_files and "efficiency_pct" not in missing_metric_keys:
            missing_metric_keys.append("efficiency_pct")

        data_by_metric = build_heatmap_matrices(
            x_values=pair["x_values"],
            y_values=pair["y_values"],
            x_key=pair["x_key"],
            y_key=pair["y_key"],
            base_cfg=base_cfg,
            metric_keys=missing_metric_keys,
            cache=cache,
            progress_label=pair["slug"],
            launch_final_values=launch_final_values,
        )
        for metric_key in missing_metric_keys:
            if metric_key not in heatmap_files:
                continue
            if heatmap_files[metric_key] not in missing_files:
                continue
            spec = metric_spec_map[metric_key]
            plot_heatmap(
                data=data_by_metric[metric_key],
                x_values=pair["x_values"],
                y_values=pair["y_values"],
                x_label=pair["x_label"],
                y_label=pair["y_label"],
                title=(
                    f"{spec['title']}: {pair['title_suffix']} | "
                    f"TVL {format_tvl_millions_label(base_cfg)}"
                ),
                colorbar_label=spec["colorbar_label"],
                filename=heatmap_files[metric_key],
                xticks=pair["xticks"],
                yticks=pair["yticks"],
                xscale="log",
                center_zero=spec["center_zero"],
                cmap=spec["cmap"],
                color_norm=spec.get("color_norm"),
                symlog_linthresh=spec.get("symlog_linthresh"),
            )

        if line_filename in missing_files:
            efficiency_data = data_by_metric["efficiency_pct"]
            launch_curve = build_metric_curve(
                x_values=pair["x_values"],
                x_key=pair["x_key"],
                base_cfg=launch_cfg,
                metric_key="efficiency_pct",
                cache=cache,
                launch_final_values=launch_final_values,
            )
            plot_arc_speed_line_chart(
                data=efficiency_data,
                x_values=pair["x_values"],
                y_values=pair["y_values"],
                y_label=pair["y_label"],
                title=(
                    "Arc-speed efficiency sweep: "
                    f"{pair['title_suffix']} | TVL {format_tvl_millions_label(base_cfg)}"
                ),
                filename=line_filename,
                launch_curve=launch_curve,
                launch_auto_speed=launch_auto_speed,
            )
        del data_by_metric
        gc.collect()

    if owns_cache:
        flush_sweep_cache(cache, force=True)
        cache.clear()
        gc.collect()
        print("Released arc-speed sweep cache.")


def get_launch_final_values(
    all_results,
    launch_cfg,
    price_data,
    market_linear_noise_data=None,
):
    """Reuse launch-style runs when available; otherwise run them once."""
    for cfg, results in all_results:
        if cfg["name"] == launch_cfg["name"]:
            launch_final_values = {
                "geometric": float(results["geometric"]["final_value"]),
            }
            if "constant_arc_length" in results:
                launch_final_values["constant_arc_length"] = float(
                    results["constant_arc_length"]["final_value"]
                )
            return launch_final_values

    print("\nRunning launch-style benchmarks for heatmaps...")
    launch_results = run_comparison(
        launch_cfg,
        price_data=price_data,
        low_data_mode=True,
        market_linear_noise_data=market_linear_noise_data,
    )
    launch_final_values = {
        "geometric": float(launch_results["geometric"]["final_value"]),
    }
    if "constant_arc_length" in launch_results:
        launch_final_values["constant_arc_length"] = float(
            launch_results["constant_arc_length"]["final_value"]
        )
    del launch_results
    gc.collect()
    return launch_final_values



def print_comparison(cfg, results):
    """Print text summary table."""
    methods = [("Geometric", results["geometric"])]
    has_constant_arc = "constant_arc_length" in results
    if has_constant_arc:
        methods.append(("Const Arc", results["constant_arc_length"]))
    noise_cfg = resolve_reclamm_noise_settings(cfg)

    hodl_value = float((methods[0][1]["reserves"][0] * methods[0][1]["prices"][-1]).sum())

    print("=" * 105)
    print(f"  {cfg['name']}")
    print(f"  price_ratio={cfg['price_ratio']}, "
          f"margin={cfg['centeredness_margin']}, "
          f"shift_exp={cfg['daily_price_shift_exponent']}, "
          f"fees={cfg['fees']}")
    print(
        f"  base_tvl=${get_initial_pool_value(cfg):,.0f} "
        f"(TVL {format_tvl_millions_label(cfg)})"
    )
    print(f"  note={cfg['reason']}")
    print(
        f"  noise={noise_cfg['noise_summary']}, "
        f"gas={cfg.get('gas_cost', 0.0)}, "
        f"protocol_fee_split={cfg.get('protocol_fee_split', 0.0)}"
    )
    if not has_constant_arc:
        print("  constant_arc=disabled")
    print("-" * 105)
    header = "  {:20s}".format("")
    for name, _ in methods:
        header += f" {name:>14s}"
    print(header)

    row = "  {:20s}".format("Final value")
    for _, r in methods:
        row += f" ${float(r['final_value']):>13,.0f}"
    print(row)

    print(f"  {'HODL value':20s} ${hodl_value:>13,.0f}")

    row = "  {:20s}".format("LVR (HODL - final)")
    for _, r in methods:
        lvr = hodl_value - float(r["final_value"])
        row += f" ${lvr:>13,.0f}"
    print(row)

    row = "  {:20s}".format("Return")
    for _, r in methods:
        ret = (float(r["final_value"]) / float(r["value"][0]) - 1) * 100
        row += f" {ret:>13.2f}%"
    print(row)

    row = "  {:20s}".format("vs HODL")
    for _, r in methods:
        vs = (float(r["final_value"]) / hodl_value - 1) * 100
        row += f" {vs:>13.2f}%"
    print(row)

    if has_constant_arc:
        geo_final = float(results["geometric"]["final_value"])
        arc_final = float(results["constant_arc_length"]["final_value"])
        geo_lvr = hodl_value - geo_final
        arc_lvr = hodl_value - arc_final
        print(f"  {'Const Arc - Geo':20s} ${arc_final - geo_final:>13,.0f}")
        print(f"  {'LVR saved vs Geo':20s} ${geo_lvr - arc_lvr:>13,.0f}")
    print("=" * 105)



def plot_comparison(cfg, results, fig_idx):
    """Plot comparison diagnostics for one config."""
    tvl_label = format_tvl_millions_label(cfg)
    variants = {
        "Geometric": (results["geometric"], "C0", "-"),
    }
    has_constant_arc = "constant_arc_length" in results
    if has_constant_arc:
        variants["Const arc-len"] = (results["constant_arc_length"], "C2", "--")

    geo = results["geometric"]
    geo_prices = np.array(geo["prices"])
    geo_reserves = np.array(geo["reserves"])
    n_steps = len(np.array(geo["value"]))
    t_days = np.arange(n_steps) / (60 * 24)

    hodl_traj = (geo_reserves[0] * geo_prices[:n_steps]).sum(axis=-1)
    price_ratio_traj = geo_prices[:n_steps, 0] / geo_prices[:n_steps, 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{cfg['name']} — TVL {tvl_label}", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    plotted_values = []
    for name, (r, color, ls) in variants.items():
        vals = np.array(r["value"])
        plotted_values.append(vals / 1e6)
        ax.plot(t_days, vals / 1e6, color=color, ls=ls, label=name, alpha=0.9)
    _set_padded_ylim(ax, plotted_values, pad_ratio=0.03)
    ax.set_xlabel("Days")
    ax.set_ylabel("Pool value ($M)")
    ax.set_title("Pool value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name, (r, color, ls) in variants.items():
        vals = np.array(r["value"])
        lvr = np.array(hodl_traj) - vals
        ax.plot(t_days, lvr / 1e3, color=color, ls=ls, label=name, alpha=0.9)
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative LVR ($K)")
    ax.set_title("Cumulative LVR (HODL - pool value)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_days, price_ratio_traj, color="C4", alpha=0.7)
    ax.set_xlabel("Days")
    ax.set_ylabel(f"{cfg['tokens'][0]}/{cfg['tokens'][1]} price ratio")
    ax.set_title("Price path")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for name, (r, color, ls) in variants.items():
        w = np.array(r["weights"])
        n_w = min(len(w), n_steps)
        t_w = np.arange(n_w) / (60 * 24)
        ax.plot(t_w, w[:n_w, 0], color=color, ls=ls, label=name, alpha=0.9)
    ax.set_xlabel("Days")
    ax.set_ylabel(f"Weight ({cfg['tokens'][0]})")
    ax.set_title("Empirical weight (token 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = tvl_artifact_filename("reclamm_thermostat_comparison", cfg, suffix=str(fig_idx))
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close(fig)

    if not has_constant_arc:
        print("Skipping constant-arc comparison diagnostics because RUN_CONSTANT_ARC_LENGTH=False.")
        return

    geo_values = np.array(geo["value"])
    geo_lvr = np.array(hodl_traj) - geo_values

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(f"{cfg['name']} — diagnostics — TVL {tvl_label}", fontsize=13, fontweight="bold")

    ax = axes2[0]
    for name, (r, color, ls) in variants.items():
        if name == "Geometric":
            continue
        vals = np.array(r["value"])
        ax.plot(t_days, (vals - geo_values) / 1e3, color=color, ls=ls,
                label=name, alpha=0.9)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Days")
    ax.set_ylabel("Value difference ($K)")
    ax.set_title("Minus Geometric")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    mask = np.abs(geo_lvr) > 100
    if mask.any():
        for name, (r, color, ls) in variants.items():
            if name == "Geometric":
                continue
            vals = np.array(r["value"])
            method_lvr = np.array(hodl_traj) - vals
            ratio = np.full_like(geo_lvr, np.nan)
            ratio[mask] = method_lvr[mask] / geo_lvr[mask]
            ax.plot(t_days, ratio, color=color, ls=ls, alpha=0.7, label=name)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_ylabel("LVR ratio (method / geometric)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "LVR too small to compare",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Days")
    ax.set_title("Relative LVR")
    ax.grid(True, alpha=0.3)

    ax = axes2[2]
    all_pos = []
    for name, (r, color, ls) in variants.items():
        vals = np.array(r["value"])
        method_lvr = np.array(hodl_traj) - vals
        step_lvr = np.diff(method_lvr)
        pos = step_lvr[step_lvr > 0]
        all_pos.append((name, pos, color))
    has_data = [len(p) > 10 for _, p, _ in all_pos]
    if any(has_data):
        max_val = max(np.percentile(p, 99) for _, p, _ in all_pos if len(p) > 10)
        bins = np.linspace(0, max_val, 50)
        for name, pos, color in all_pos:
            if len(pos) > 10:
                ax.hist(pos, bins=bins, color=color, alpha=0.3, label=name,
                        density=True)
        ax.set_xlabel("Per-step LVR ($)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Too few thermostat steps",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_title("Per-step LVR distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname2 = tvl_artifact_filename("reclamm_thermostat_diff", cfg, suffix=str(fig_idx))
    plt.savefig(fname2, dpi=150)
    print(f"Saved {fname2}")
    plt.close(fig2)

    arc_values = np.array(results["constant_arc_length"]["value"])
    n_eff = min(len(geo_values), len(arc_values))
    t_eff = np.arange(n_eff) / (60 * 24)
    efficiency_pct = (
        (arc_values[:n_eff] - geo_values[:n_eff])
        / np.maximum(np.abs(geo_values[:n_eff]), 1e-12)
        * 100.0
    )

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4.5))
    fig3.suptitle(f"{cfg['name']} — efficiency — TVL {tvl_label}", fontsize=13, fontweight="bold")
    ax3.plot(
        t_eff,
        efficiency_pct,
        color="C2",
        linewidth=1.8,
        label="(Const Arc - Geo) / Geo",
    )
    ax3.axhline(0.0, color="gray", ls="--", alpha=0.6)
    _set_padded_ylim(ax3, [efficiency_pct], pad_ratio=0.08)
    ax3.set_xlabel("Days")
    ax3.set_ylabel("Efficiency vs geometric (%)")
    ax3.set_title("Efficiency")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fname3 = tvl_artifact_filename("reclamm_thermostat_efficiency", cfg, suffix=str(fig_idx))
    plt.savefig(fname3, dpi=150)
    print(f"Saved {fname3}")
    plt.close(fig3)



if __name__ == "__main__":
    shared_price_data = load_shared_price_data(CONFIGS)
    shared_market_linear_noise_data = load_shared_market_linear_noise_data()

    for initial_pool_value in TVL_SWEEP_VALUES:
        tvl_configs = configs_for_tvl(CONFIGS, initial_pool_value)
        tvl_label = format_tvl_millions_label(tvl_configs[0])
        print(f"\n=== TVL sweep: {tvl_label} ===")

        all_results = []
        for i, cfg in enumerate(tvl_configs):
            print(f"\n>>> Running {cfg['name']} at TVL {tvl_label}...")
            try:
                results = run_comparison(
                    cfg,
                    price_data=shared_price_data,
                    market_linear_noise_data=shared_market_linear_noise_data,
                )
                print_comparison(cfg, results)
                plot_comparison(cfg, results, i)
                all_results.append((cfg, results))
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback

                traceback.print_exc()

        if len(all_results) > 1:
            if RUN_CONSTANT_ARC_LENGTH:
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                fig.suptitle(
                    f"Cross-config comparison (normalised) — TVL {tvl_label}",
                    fontsize=13,
                    fontweight="bold",
                )

                method_keys = [
                    ("geometric", "geo", "-"),
                    ("constant_arc_length", "arc", "--"),
                ]

                for i, (cfg, results) in enumerate(all_results):
                    geo_v = np.array(results["geometric"]["value"])
                    t = np.arange(len(geo_v)) / (60 * 24)
                    short_name = cfg["name"].split("(")[0].strip()

                    for j, (key, suffix, ls) in enumerate(method_keys):
                        v = np.array(results[key]["value"])
                        color_idx = i * len(method_keys) + j

                        axes[0].plot(
                            t,
                            v / v[0],
                            ls=ls,
                            alpha=0.8,
                            label=f"{short_name} {suffix}",
                            color=f"C{color_idx % 10}",
                        )

                        if key != "geometric":
                            pct_diff = (v - geo_v) / geo_v * 100
                            axes[1].plot(
                                t,
                                pct_diff,
                                ls=ls,
                                alpha=0.8,
                                label=f"{short_name} {suffix}",
                                color=f"C{color_idx % 10}",
                            )

                axes[0].set_xlabel("Days")
                axes[0].set_ylabel("Normalised pool value")
                axes[0].set_title("Pool value (V/V0)")
                axes[0].legend(fontsize=6, ncol=2)
                axes[0].grid(True, alpha=0.3)

                axes[1].set_xlabel("Days")
                axes[1].set_ylabel("Efficiency vs geometric (%)")
                axes[1].set_title("Efficiency vs Geometric")
                axes[1].axhline(0, color="gray", ls="--", alpha=0.5)
                axes[1].legend(fontsize=6, ncol=2)
                axes[1].grid(True, alpha=0.3)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(9, 5))
                fig.suptitle(
                    f"Cross-config comparison (normalised geometric) — TVL {tvl_label}",
                    fontsize=13,
                    fontweight="bold",
                )

                for i, (cfg, results) in enumerate(all_results):
                    geo_v = np.array(results["geometric"]["value"])
                    t = np.arange(len(geo_v)) / (60 * 24)
                    short_name = cfg["name"].split("(")[0].strip()
                    ax.plot(
                        t,
                        geo_v / geo_v[0],
                        ls="-",
                        alpha=0.8,
                        label=f"{short_name} geo",
                        color=f"C{i % 10}",
                    )

                ax.set_xlabel("Days")
                ax.set_ylabel("Normalised pool value")
                ax.set_title("Geometric pool value (V/V0)")
                ax.legend(fontsize=6, ncol=2)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            summary_name = tvl_artifact_filename(
                "reclamm_thermostat_summary",
                tvl_configs[0],
            )
            plt.savefig(summary_name, dpi=150)
            print(f"\nSaved {summary_name}")
            plt.close(fig)

        launch_final_values = get_launch_final_values(
            all_results,
            launch_cfg=tvl_configs[0],
            price_data=shared_price_data,
            market_linear_noise_data=shared_market_linear_noise_data,
        )
        shared_sweep_cache = make_sweep_cache(
            shared_price_data,
            cache_scope_cfg=tvl_configs[1],
            market_linear_noise_data=shared_market_linear_noise_data,
        )

        print(f"\nGenerating thermostat heatmaps for TVL {tvl_label}...")
        generate_heatmaps(
            dict(tvl_configs[1]),
            shared_price_data,
            launch_final_values=launch_final_values,
            cache=shared_sweep_cache,
        )

        generate_arc_speed_efficiency_artifacts(
            dict(tvl_configs[1]),
            launch_cfg=dict(tvl_configs[0]),
            price_data=shared_price_data,
            launch_final_values=launch_final_values,
            cache=shared_sweep_cache,
        )
        generate_three_variable_3d_heatmaps(
            dict(tvl_configs[1]),
            price_data=shared_price_data,
            launch_final_values=launch_final_values,
            cache=shared_sweep_cache,
        )
        flush_sweep_cache(shared_sweep_cache, force=True)
        shared_sweep_cache.clear()
        gc.collect()
        print(f"Released shared sweep cache for TVL {tvl_label}.")
