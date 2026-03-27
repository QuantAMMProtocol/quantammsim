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
import math
import os

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


RUN_CONSTANT_ARC_LENGTH = False
INTERPOLATION_METHODS = (
    ("geometric", "constant_arc_length")
    if RUN_CONSTANT_ARC_LENGTH
    else ("geometric",)
)
HEATMAP_PRICE_RATIOS = np.arange(1.01, 1.50 + 1e-9, 0.025)
HEATMAP_MARGINS = np.linspace(0.05, 0.90, 20)
HEATMAP_SHIFT_EXPONENTS = np.arange(0.01, 0.50 + 1e-9, 0.025)
HEATMAP_ARC_LENGTH_SPEEDS = np.geomspace(1.0e-6, 5.0e-4, 11)
PRICE_RATIO_TICKS = np.array([1.01, 1.10, 1.20, 1.30, 1.40, 1.50])
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
AAVE_ETH_NOISE_SETTINGS = {
    "enable_noise_model": True,
    "noise_model": DEFAULT_NOISE_MODEL,
    "noise_artifact_dir": DEFAULT_MARKET_LINEAR_ARTIFACT_DIR,
    "noise_pool_id": AAVE_WETH_POOL_ID,
    "gas_cost": DEFAULT_GAS_COST,
    "protocol_fee_split": DEFAULT_PROTOCOL_FEE_SPLIT,
}

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
_WARNED_NOISE_FALLBACKS = set()


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


def configs_for_tvl(base_configs, initial_pool_value):
    """Attach a shared initial TVL to each compare configuration."""
    configs = []
    for cfg in base_configs:
        updated = dict(cfg)
        updated["initial_pool_value"] = float(initial_pool_value)
        configs.append(updated)
    return configs


def make_noise_variant_cfg(cfg, enable_noise_model):
    """Return a config with either noise modelling or pure arb-only enabled."""
    updated = dict(cfg)
    if enable_noise_model:
        updated["enable_noise_model"] = True
        return updated

    matched_noise = resolve_reclamm_noise_settings(cfg)

    updated["enable_noise_model"] = False
    updated["noise_model"] = None
    updated["gas_cost"] = cfg.get("gas_cost", DEFAULT_GAS_COST)
    updated["protocol_fee_split"] = cfg.get(
        "protocol_fee_split", DEFAULT_PROTOCOL_FEE_SPLIT
    )
    updated["noise_trader_ratio"] = 0.0
    matched_arb_frequency = matched_noise.get("arb_frequency")
    if matched_arb_frequency is not None:
        updated["arb_frequency"] = matched_arb_frequency
    for key in (
        "reclamm_noise_params",
        "noise_arrays_path",
        "noise_artifact_dir",
        "noise_pool_id",
    ):
        updated.pop(key, None)
    return updated


def _warn_noise_fallback(message):
    """Print a one-time message when the preferred noise setup is unavailable."""
    if message not in _WARNED_NOISE_FALLBACKS:
        print(message)
        _WARNED_NOISE_FALLBACKS.add(message)


def _hashable_noise_params(params):
    """Convert a noise-params dict into a stable cache key fragment."""
    if params is None:
        return None
    return tuple(sorted((str(k), round(float(v), 12)) for k, v in params.items()))


def _legacy_calibrated_noise_settings(reason=None):
    """Fallback calibrated noise config used when market-linear artifacts are absent."""
    if reason:
        _warn_noise_fallback(
            "market_linear noise unavailable for thermostat comparison; "
            f"falling back to calibrated legacy coefficients ({reason})."
        )
    return {
        "noise_model": "calibrated",
        "noise_trader_ratio": 0.0,
        "reclamm_noise_params": {
            f"c_{i}": LEGACY_NOISE_COEFFS[i] for i in range(len(LEGACY_NOISE_COEFFS))
        },
        "arb_frequency": LEGACY_ARB_FREQUENCY,
        "noise_summary": (
            "calibrated legacy 8-covariate "
            f"(arb_frequency={LEGACY_ARB_FREQUENCY})"
        ),
        "noise_cache_key": (
            "calibrated",
            tuple(round(float(c), 12) for c in LEGACY_NOISE_COEFFS),
            LEGACY_ARB_FREQUENCY,
        ),
    }


def resolve_reclamm_noise_settings(cfg):
    """Resolve the active reCLAMM noise-model fingerprint block for a config."""
    enable_noise_model = cfg.get("enable_noise_model", False)
    requested_mode = cfg.get("noise_model", DEFAULT_NOISE_MODEL)
    cache_key = (
        tuple(cfg.get("tokens", [])),
        cfg.get("start"),
        cfg.get("end"),
        enable_noise_model,
        requested_mode,
        cfg.get("noise_artifact_dir", DEFAULT_MARKET_LINEAR_ARTIFACT_DIR),
        cfg.get("noise_pool_id", AAVE_WETH_POOL_ID),
        cfg.get("arb_frequency"),
        round(float(cfg.get("noise_trader_ratio", 0.0)), 12),
        _hashable_noise_params(cfg.get("reclamm_noise_params")),
        cfg.get("noise_arrays_path"),
    )
    if cache_key in _NOISE_SETTINGS_CACHE:
        return _NOISE_SETTINGS_CACHE[cache_key]

    if not enable_noise_model:
        result = {
            "noise_model": None,
            "noise_trader_ratio": 0.0,
            "reclamm_noise_params": None,
            "noise_arrays_path": None,
            "arb_frequency": None,
            "noise_summary": "arb-only (noise disabled)",
            "noise_cache_key": ("disabled",),
        }
    elif requested_mode == "market_linear":
        artifact_dir = cfg.get("noise_artifact_dir", DEFAULT_MARKET_LINEAR_ARTIFACT_DIR)
        pool_id = cfg.get("noise_pool_id", AAVE_WETH_POOL_ID)
        start_date = str(cfg["start"]).split(" ")[0]
        end_date = str(cfg["end"]).split(" ")[0]
        try:
            from quantammsim.calibration.noise_model_arrays import (
                _find_pool_index,
                build_simulator_arrays,
                load_artifact,
            )

            model_path = os.path.join(artifact_dir, "model.npz")
            meta_path = os.path.join(artifact_dir, "meta.json")
            if not (os.path.exists(model_path) and os.path.exists(meta_path)):
                raise FileNotFoundError(
                    f"expected {model_path} and {meta_path}"
                )

            cache_dir = os.path.join(artifact_dir, "_sim_arrays")
            os.makedirs(cache_dir, exist_ok=True)
            arrays_path = os.path.join(
                cache_dir,
                f"{pool_id}_{start_date}_{end_date}.npz",
            )
            if not os.path.exists(arrays_path):
                arrays = build_simulator_arrays(
                    pool_id=pool_id,
                    start_date=start_date,
                    end_date=end_date,
                    artifact_dir=artifact_dir,
                )
                np.savez(
                    arrays_path,
                    noise_base=arrays["noise_base"],
                    noise_tvl_coeff=arrays["noise_tvl_coeff"],
                    tvl_mean=arrays["tvl_mean"],
                    tvl_std=arrays["tvl_std"],
                )

            with np.load(arrays_path) as arrays:
                tvl_mean = float(arrays["tvl_mean"])
                tvl_std = float(arrays["tvl_std"])

            art, meta = load_artifact(artifact_dir)
            pool_idx = _find_pool_index(pool_id, meta["pool_ids"])
            if pool_idx >= 0:
                learned_cadence = float(np.exp(art["log_cadence"][pool_idx]))
            else:
                learned_cadence = 5.0
            arb_frequency = max(1, round(learned_cadence))
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
        except Exception as exc:  # pragma: no cover - fallback path depends on local artifacts
            result = _legacy_calibrated_noise_settings(str(exc))
    elif requested_mode == "calibrated":
        params = cfg.get("reclamm_noise_params")
        if params is None:
            result = _legacy_calibrated_noise_settings()
        else:
            arb_frequency = cfg.get("arb_frequency", LEGACY_ARB_FREQUENCY)
            result = {
                "noise_model": "calibrated",
                "noise_trader_ratio": cfg.get("noise_trader_ratio", 0.0),
                "reclamm_noise_params": dict(params),
                "arb_frequency": arb_frequency,
                "noise_summary": f"calibrated (arb_frequency={arb_frequency})",
                "noise_cache_key": (
                    "calibrated",
                    _hashable_noise_params(params),
                    arb_frequency,
                ),
            }
    else:
        arb_frequency = cfg.get("arb_frequency")
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
                _hashable_noise_params(cfg.get("reclamm_noise_params")),
                cfg.get("noise_arrays_path"),
                arb_frequency,
            ),
        }

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


def make_fingerprint(cfg, interpolation_method):
    """Build run fingerprint for a given config and interpolation method."""
    speed_override = (
        cfg.get("arc_length_speed")
        if interpolation_method == "constant_arc_length"
        else None
    )
    noise_cfg = resolve_reclamm_noise_settings(cfg)
    arb_frequency = cfg.get("arb_frequency", noise_cfg.get("arb_frequency"))
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
    if arb_frequency is not None:
        fingerprint["arb_frequency"] = arb_frequency
    return fingerprint


def make_params(cfg):
    """Build pool params from config."""
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


def run_comparison(cfg, price_data=None, low_data_mode=False):
    """Run both interpolation variants, return results dict."""
    params = make_params(cfg)

    results = {}
    for method in INTERPOLATION_METHODS:
        fp = make_fingerprint(cfg, method)
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
    """Count memoized final-value runs."""
    return len(cache.get("_final_value_cache", {}))


def _comparison_cache_size(cache):
    """Count memoized scalar comparison bundles."""
    return len(cache.get("_comparison_cache", {}))


def make_sweep_cache(price_data):
    """Create a shared cache for heatmap and line sweeps."""
    return {
        "_shared_price_data": price_data,
        "_final_value_cache": {},
        "_comparison_cache": {},
    }


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
    noise_cfg = resolve_reclamm_noise_settings(cfg)
    arb_frequency = cfg.get("arb_frequency", noise_cfg.get("arb_frequency"))
    key = (
        method,
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
    if key not in final_value_cache:
        result = do_run_on_historic_data(
            run_fingerprint=make_fingerprint(cfg, method),
            params=make_params(cfg),
            price_data=cache["_shared_price_data"],
            low_data_mode=True,
        )
        final_value_cache[key] = float(result["final_value"])
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
        row_new_final_runs = _cache_size(cache) - final_cache_before_row
        row_new_comparisons = (
            _comparison_cache_size(cache) - comparison_cache_before_row
        )
        row_pct = completed_points / total_points * 100.0
        print(
            f"[{progress_label}] row {yi + 1}/{len(y_values)} complete "
            f"({y_key}={float(y_value):.4f}, {completed_points}/{total_points} "
            f"points, {row_pct:.1f}%, {row_new_final_runs} new final-value runs, "
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
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]

    if center_zero:
        vmax = max(abs(float(np.nanmin(data))), abs(float(np.nanmax(data))), 1e-9)
        if color_norm == "symlog" and symlog_linthresh is not None and vmax > symlog_linthresh:
            norm = SymLogNorm(
                linthresh=symlog_linthresh,
                linscale=1.0,
                vmin=-vmax,
                vmax=vmax,
                base=10.0,
            )
        else:
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        cmap_name = cmap or "RdYlGn"
    else:
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(finite.min())
            vmax = float(finite.max())
            if np.isclose(vmin, vmax):
                pad = max(abs(vmin) * 0.01, 1e-9)
                vmin -= pad
                vmax += pad
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_name = cmap or "viridis"

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
        cache = make_sweep_cache(price_data)
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
    pair_specs = [
        {
            "slug": "price_ratio_vs_margin",
            "x_values": HEATMAP_PRICE_RATIOS,
            "y_values": HEATMAP_MARGINS,
            "x_key": "price_ratio",
            "y_key": "centeredness_margin",
            "x_label": "Price ratio",
            "y_label": "Centeredness margin",
            "title_suffix": (
                f"shift_exp fixed at {base_cfg['daily_price_shift_exponent']:.2f}"
            ),
            "xticks": PRICE_RATIO_TICKS,
            "yticks": MARGIN_TICKS
        },
        {
            "slug": "shift_exp_vs_margin",
            "x_values": HEATMAP_SHIFT_EXPONENTS,
            "y_values": HEATMAP_MARGINS,
            "x_key": "daily_price_shift_exponent",
            "y_key": "centeredness_margin",
            "x_label": "Shift exponent",
            "y_label": "Centeredness margin",
            "title_suffix": f"price_ratio fixed at {base_cfg['price_ratio']:.2f}",
            "xticks": SHIFT_EXPONENT_TICKS,
            "yticks": MARGIN_TICKS
        },
        {
            "slug": "price_ratio_vs_shift_exp",
            "x_values": HEATMAP_PRICE_RATIOS,
            "y_values": HEATMAP_SHIFT_EXPONENTS,
            "x_key": "price_ratio",
            "y_key": "daily_price_shift_exponent",
            "x_label": "Price ratio",
            "y_label": "Shift exponent",
            "title_suffix": (
                f"margin fixed at {base_cfg['centeredness_margin']:.2f}"
            ),
            "xticks": PRICE_RATIO_TICKS,
            "yticks": SHIFT_EXPONENT_TICKS,
        },
    ]

    metric_spec_map = {spec["key"]: spec for spec in metric_specs}

    if RUN_CONSTANT_ARC_LENGTH:
        print(
            "Using launch-style benchmarks "
            f"Geo=${launch_final_values['geometric']:,.0f}, "
            f"Const Arc=${launch_final_values['constant_arc_length']:,.0f}, "
            f"TVL={format_tvl_millions_label(base_cfg)}."
        )
        print(
            "Running {count} heatmap pair sweeps sequentially "
            "(current outputs use cached noise-model runs; improvement heatmaps "
            "reuse those values and add cached arb-only runs).".format(
                count=len(pair_specs)
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
            "and only geometric/arb-only geometric runs will be scheduled."
        )

    for pair in pair_specs:
        output_files = {
            spec["key"]: heatmap_artifact_filename(
                spec,
                base_cfg,
                suffix=pair["slug"],
            )
            for spec in metric_specs
        }
        missing_files = _missing_artifacts(
            pair["slug"],
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
            base_cfg=base_cfg,
            metric_keys=missing_metric_keys,
            cache=cache,
            progress_label=pair["slug"],
            launch_final_values=launch_final_values,
        )
        print(f"[{pair['slug']}] plotting missing heatmaps...")
        for metric_key in missing_metric_keys:
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
        cache.clear()
        gc.collect()
        print("Released heatmap metric cache.")


def compute_auto_calibrated_arc_length_speed(cfg, price_data):
    """Compute the launch/reference auto-calibrated speed for a config."""
    start_ts = pd.Timestamp(cfg["start"])

    if isinstance(price_data.index, pd.DatetimeIndex):
        row = price_data.loc[start_ts]
    else:
        start_unix_ms = int(start_ts.timestamp() * 1000.0)
        index_values = price_data.index.to_numpy(dtype=np.int64)
        row_idx = int(np.searchsorted(index_values, start_unix_ms, side="left"))
        if row_idx >= len(index_values):
            row_idx = len(index_values) - 1
        if row_idx > 0 and index_values[row_idx] != start_unix_ms:
            prev_idx = row_idx - 1
            if abs(index_values[prev_idx] - start_unix_ms) <= abs(
                index_values[row_idx] - start_unix_ms
            ):
                row_idx = prev_idx
        row = price_data.iloc[row_idx]

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

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
        cache = make_sweep_cache(price_data)
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
        cache.clear()
        gc.collect()
        print("Released arc-speed sweep cache.")


def get_launch_final_values(all_results, launch_cfg, price_data):
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

    for initial_pool_value in TVL_SWEEP_VALUES:
        tvl_configs = configs_for_tvl(CONFIGS, initial_pool_value)
        tvl_label = format_tvl_millions_label(tvl_configs[0])
        print(f"\n=== TVL sweep: {tvl_label} ===")

        all_results = []
        for i, cfg in enumerate(tvl_configs):
            print(f"\n>>> Running {cfg['name']} at TVL {tvl_label}...")
            try:
                results = run_comparison(cfg, price_data=shared_price_data)
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
        )
        shared_sweep_cache = make_sweep_cache(shared_price_data)

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
        shared_sweep_cache.clear()
        gc.collect()
        print(f"Released shared sweep cache for TVL {tvl_label}.")
