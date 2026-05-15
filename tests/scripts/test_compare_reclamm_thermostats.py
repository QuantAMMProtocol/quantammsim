"""Tests for heatmap skip logic in compare_reclamm_thermostats.py."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "compare_reclamm_thermostats.py"
)


def _load_script_module():
    injected_modules = {}

    def inject_module(name, module):
        injected_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    # Minimal stubs so the script can be imported without the full runtime
    # stack present in this test environment.
    jax_module = types.ModuleType("jax")
    jax_module.numpy = np
    inject_module("jax", jax_module)
    inject_module("jax.numpy", np)

    pandas_module = types.ModuleType("pandas")
    pandas_module.Timestamp = lambda value: value
    pandas_module.DatetimeIndex = tuple
    pandas_module.DataFrame = type("DataFrame", (), {})
    pandas_module.read_parquet = lambda *args, **kwargs: None
    inject_module("pandas", pandas_module)

    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    pyplot_module.cm = types.SimpleNamespace(viridis=lambda values: values)
    colors_module = types.ModuleType("matplotlib.colors")
    colors_module.TwoSlopeNorm = object
    colors_module.Normalize = object
    colors_module.SymLogNorm = object
    cm_module = types.ModuleType("matplotlib.cm")
    cm_module.ScalarMappable = object
    inject_module("matplotlib", matplotlib_module)
    inject_module("matplotlib.pyplot", pyplot_module)
    inject_module("matplotlib.colors", colors_module)
    inject_module("matplotlib.cm", cm_module)

    quantammsim_module = types.ModuleType("quantammsim")
    runners_module = types.ModuleType("quantammsim.runners")
    jax_runners_module = types.ModuleType("quantammsim.runners.jax_runners")
    jax_runners_module.do_run_on_historic_data = lambda **kwargs: {
        "final_value": 0.0
    }
    runners_module.jax_runners = jax_runners_module

    pools_module = types.ModuleType("quantammsim.pools")
    reclamm_pkg_module = types.ModuleType("quantammsim.pools.reCLAMM")
    reserves_module = types.ModuleType(
        "quantammsim.pools.reCLAMM.reclamm_reserves"
    )
    reserves_module.calibrate_arc_length_speed = lambda *args, **kwargs: 0.0
    reserves_module.compute_price_ratio = lambda *args, **kwargs: 1.0
    reserves_module.initialise_reclamm_reserves = (
        lambda *args, **kwargs: (np.array([1.0, 1.0]), 1.0, 1.0)
    )
    reclamm_pkg_module.reclamm_reserves = reserves_module
    pools_module.reCLAMM = reclamm_pkg_module

    utils_module = types.ModuleType("quantammsim.utils")
    data_processing_module = types.ModuleType("quantammsim.utils.data_processing")
    historic_utils_module = types.ModuleType(
        "quantammsim.utils.data_processing.historic_data_utils"
    )
    historic_utils_module.get_historic_parquet_data = lambda *args, **kwargs: None
    data_processing_module.historic_data_utils = historic_utils_module
    utils_module.data_processing = data_processing_module

    quantammsim_module.runners = runners_module
    quantammsim_module.pools = pools_module
    quantammsim_module.utils = utils_module

    inject_module("quantammsim", quantammsim_module)
    inject_module("quantammsim.runners", runners_module)
    inject_module("quantammsim.runners.jax_runners", jax_runners_module)
    inject_module("quantammsim.pools", pools_module)
    inject_module("quantammsim.pools.reCLAMM", reclamm_pkg_module)
    inject_module("quantammsim.pools.reCLAMM.reclamm_reserves", reserves_module)
    inject_module("quantammsim.utils", utils_module)
    inject_module("quantammsim.utils.data_processing", data_processing_module)
    inject_module(
        "quantammsim.utils.data_processing.historic_data_utils",
        historic_utils_module,
    )

    spec = importlib.util.spec_from_file_location(
        "test_compare_reclamm_thermostats_module",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in injected_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture
def script_module():
    return _load_script_module()


@pytest.fixture
def base_cfg():
    return {
        "tokens": ["AAVE", "ETH"],
        "start": "2024-06-01 00:00:00",
        "end": "2025-06-01 00:00:00",
        "fees": 0.0025,
        "price_ratio": 1.10,
        "centeredness_margin": 0.60,
        "daily_price_shift_exponent": 0.1,
        "initial_pool_value": 5_000_000.0,
    }


@pytest.fixture
def launch_final_values():
    return {
        "geometric": 1_000_000.0,
        "constant_arc_length": 1_010_000.0,
    }


def test_make_noise_variant_cfg_disables_noise_fields(script_module, base_cfg):
    noisy_cfg = {
        **base_cfg,
        "enable_noise_model": True,
        "noise_model": "market_linear",
        "noise_artifact_dir": "results/linear_market_noise",
        "noise_pool_id": "0x9d1fcf346ea1b0",
        "gas_cost": 1.0,
        "protocol_fee_split": 0.25,
        "reclamm_noise_params": {"tvl_mean": 1.0, "tvl_std": 2.0},
        "noise_arrays_path": "results/linear_market_noise/_sim_arrays/aave_eth.npz",
        "arb_frequency": 6,
    }

    arb_only_cfg = script_module.make_noise_variant_cfg(
        noisy_cfg,
        enable_noise_model=False,
    )
    resolved = script_module.resolve_reclamm_noise_settings(arb_only_cfg)

    assert arb_only_cfg["enable_noise_model"] is False
    assert arb_only_cfg["noise_model"] == "arb_only"
    assert arb_only_cfg["noise_reference_model"] == "market_linear"
    assert arb_only_cfg["gas_cost"] == script_module.DEFAULT_GAS_COST
    assert arb_only_cfg["protocol_fee_split"] == script_module.DEFAULT_PROTOCOL_FEE_SPLIT
    assert arb_only_cfg["arb_frequency"] == script_module.FIXED_COMPARE_ARB_FREQUENCY
    assert arb_only_cfg["noise_artifact_dir"] == script_module.DEFAULT_MARKET_LINEAR_ARTIFACT_DIR
    assert arb_only_cfg["noise_pool_id"] == script_module.AAVE_WETH_POOL_ID
    assert arb_only_cfg["noise_arrays_path"] == script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    assert "reclamm_noise_params" not in arb_only_cfg
    assert resolved["noise_model"] == "arb_only"
    assert resolved["noise_arrays_path"] == script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    assert set(resolved["reclamm_noise_params"]) == {"tvl_mean", "tvl_std"}
    assert resolved["noise_summary"] == (
        f"arb_only (arb_frequency={script_module.FIXED_COMPARE_ARB_FREQUENCY})"
    )


def test_make_noise_variant_cfg_defaults_to_fixed_compare_arb_cadence(
    script_module,
    base_cfg,
):
    noisy_cfg = {
        **base_cfg,
        "enable_noise_model": True,
        "noise_model": "market_linear",
    }

    arb_only_cfg = script_module.make_noise_variant_cfg(
        noisy_cfg,
        enable_noise_model=False,
    )
    resolved_noise = script_module.resolve_reclamm_noise_settings(noisy_cfg)

    assert arb_only_cfg["arb_frequency"] == script_module.FIXED_COMPARE_ARB_FREQUENCY
    assert arb_only_cfg["noise_arrays_path"] == script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    assert resolved_noise["arb_frequency"] == script_module.FIXED_COMPARE_ARB_FREQUENCY


def test_make_fingerprint_ignores_non_axis_override_fields(script_module, base_cfg):
    canonical_cfg = {
        **base_cfg,
        "enable_noise_model": True,
        "noise_model": "market_linear",
    }
    noisy_override_cfg = {
        **canonical_cfg,
        "arb_frequency": 6,
        "gas_cost": 7.0,
        "protocol_fee_split": 0.9,
        "arb_fees": 3.0,
        "noise_artifact_dir": "custom/noise/dir",
        "noise_pool_id": "override-pool",
        "reclamm_noise_params": {"tvl_mean": 999.0},
        "noise_arrays_path": "custom/path.npz",
    }

    canonical_fingerprint = script_module.make_fingerprint(canonical_cfg, "geometric")
    overridden_fingerprint = script_module.make_fingerprint(
        noisy_override_cfg,
        "geometric",
    )
    canonical_key = script_module._make_method_cache_key(canonical_cfg, "geometric")
    overridden_key = script_module._make_method_cache_key(
        noisy_override_cfg,
        "geometric",
    )

    assert overridden_fingerprint == canonical_fingerprint
    assert overridden_key == canonical_key
    assert overridden_fingerprint["arb_frequency"] == script_module.FIXED_COMPARE_ARB_FREQUENCY
    assert overridden_fingerprint["gas_cost"] == script_module.DEFAULT_GAS_COST
    assert (
        overridden_fingerprint["noise_arrays_path"]
        == script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    )
    assert (
        overridden_fingerprint["protocol_fee_split"]
        == script_module.DEFAULT_PROTOCOL_FEE_SPLIT
    )


def test_arb_only_fingerprint_only_changes_noise_model(script_module, base_cfg):
    noisy_cfg = {
        **base_cfg,
        "enable_noise_model": True,
        "noise_model": "market_linear",
    }

    noise_fingerprint = script_module.make_fingerprint(noisy_cfg, "geometric")
    arb_only_cfg = script_module.make_noise_variant_cfg(noisy_cfg, enable_noise_model=False)
    arb_fingerprint = script_module.make_fingerprint(arb_only_cfg, "geometric")

    expected_arb = dict(noise_fingerprint)
    expected_arb["noise_model"] = "arb_only"

    assert noise_fingerprint["noise_model"] == "market_linear"
    assert noise_fingerprint["noise_arrays_path"] == script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    assert arb_fingerprint == expected_arb


def test_make_fingerprint_keeps_path_fallback_when_arrays_not_preloaded(
    script_module,
    base_cfg,
):
    noisy_cfg = {
        **base_cfg,
        "enable_noise_model": True,
        "noise_model": "market_linear",
    }

    fingerprint = script_module.make_fingerprint(noisy_cfg, "geometric")

    assert fingerprint["noise_arrays_path"] == script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    assert "noise_base_array" not in fingerprint
    assert "noise_tvl_coeff_array" not in fingerprint


def test_make_fingerprint_includes_preloaded_market_linear_arrays(
    script_module,
    base_cfg,
):
    noisy_cfg = {
        **base_cfg,
        "enable_noise_model": True,
        "noise_model": "market_linear",
    }
    shared_noise = {
        "arrays_path": script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH,
        "noise_base_array": np.array([1.0, 2.0]),
        "noise_tvl_coeff_array": np.array([3.0, 4.0]),
        "tvl_mean": 10.0,
        "tvl_std": 5.0,
    }

    fingerprint = script_module.make_fingerprint(
        noisy_cfg,
        "geometric",
        market_linear_noise_data=shared_noise,
    )

    assert fingerprint["noise_arrays_path"] == script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    assert np.array_equal(fingerprint["noise_base_array"], shared_noise["noise_base_array"])
    assert np.array_equal(
        fingerprint["noise_tvl_coeff_array"],
        shared_noise["noise_tvl_coeff_array"],
    )


@pytest.mark.parametrize(
    ("field", "updated_value"),
    [
        ("fees", 0.01),
        ("start", "2024-07-01 00:00:00"),
        ("end", "2025-07-01 00:00:00"),
        ("tokens", ["WBTC", "ETH"]),
    ],
)
def test_method_cache_key_includes_run_identity_fields(
    script_module,
    base_cfg,
    field,
    updated_value,
):
    canonical_cfg = {
        **base_cfg,
        "enable_noise_model": True,
        "noise_model": "market_linear",
    }
    updated_cfg = dict(canonical_cfg)
    updated_cfg[field] = updated_value

    canonical_key = script_module._make_method_cache_key(canonical_cfg, "geometric")
    updated_key = script_module._make_method_cache_key(updated_cfg, "geometric")

    assert updated_key != canonical_key


@pytest.mark.parametrize(
    "cfg_overrides",
    [
        {"enable_noise_model": True, "noise_model": "calibrated"},
        {
            "enable_noise_model": False,
            "noise_model": "arb_only",
            "noise_reference_model": "calibrated",
        },
    ],
)
def test_resolve_reclamm_noise_settings_rejects_legacy_modes(
    script_module,
    base_cfg,
    cfg_overrides,
):
    cfg = {
        **base_cfg,
        **cfg_overrides,
    }

    with pytest.raises(ValueError, match="market_linear"):
        script_module.resolve_reclamm_noise_settings(cfg)


def test_generate_heatmaps_skips_existing_pairs(
    monkeypatch,
    script_module,
    base_cfg,
    launch_final_values,
):
    monkeypatch.setattr(script_module.os.path, "exists", lambda filename: True)
    monkeypatch.setattr(
        script_module,
        "build_heatmap_matrices",
        lambda **kwargs: pytest.fail("heatmap sweep should have been skipped"),
    )
    monkeypatch.setattr(
        script_module,
        "plot_heatmap",
        lambda **kwargs: pytest.fail("plotting should have been skipped"),
    )

    script_module.generate_heatmaps(
        base_cfg,
        price_data=None,
        launch_final_values=launch_final_values,
        cache={},
    )


def test_generate_heatmaps_only_renders_missing_artifacts(
    monkeypatch,
    script_module,
    base_cfg,
    launch_final_values,
):
    pair = script_module.get_pair_heatmap_specs(base_cfg)[0]
    slice_variant = pair["fixed_slices"][0]
    pair_suffix = script_module._pair_slice_suffix(pair, slice_variant)
    missing_file = script_module.tvl_artifact_filename(
        "reclamm_heatmap_geometric_vs_launch_geometric_symlog20",
        base_cfg,
        suffix=pair_suffix,
    )

    def fake_exists(filename):
        if filename == missing_file:
            return False
        return filename.startswith("reclamm_heatmap_")

    build_calls = []
    plotted_files = []

    def fake_build_heatmap_matrices(**kwargs):
        build_calls.append(kwargs)
        return {
            "geometric_vs_launch_geometric_pct": np.zeros(
                (len(kwargs["y_values"]), len(kwargs["x_values"])),
                dtype=float,
            )
        }

    def fake_plot_heatmap(**kwargs):
        plotted_files.append(kwargs["filename"])

    monkeypatch.setattr(script_module.os.path, "exists", fake_exists)
    monkeypatch.setattr(
        script_module,
        "build_heatmap_matrices",
        fake_build_heatmap_matrices,
    )
    monkeypatch.setattr(script_module, "plot_heatmap", fake_plot_heatmap)

    script_module.generate_heatmaps(
        base_cfg,
        price_data=None,
        launch_final_values=launch_final_values,
        cache={},
    )

    assert len(build_calls) == 1
    assert build_calls[0]["progress_label"] == pair_suffix
    assert build_calls[0]["base_cfg"][pair["fixed_key"]] == pytest.approx(
        slice_variant["value"]
    )
    assert build_calls[0]["metric_keys"] == ["geometric_vs_launch_geometric_pct"]
    assert plotted_files == [missing_file]


def test_generate_heatmaps_only_renders_missing_improvement_artifacts(
    monkeypatch,
    script_module,
    base_cfg,
    launch_final_values,
):
    pair = script_module.get_pair_heatmap_specs(base_cfg)[0]
    slice_variant = pair["fixed_slices"][0]
    pair_suffix = script_module._pair_slice_suffix(pair, slice_variant)
    missing_file = script_module.tvl_artifact_filename(
        "reclamm_heatmap_noise_vs_arb_geometric_improvement_symlog20",
        base_cfg,
        suffix=pair_suffix,
    )

    def fake_exists(filename):
        if filename == missing_file:
            return False
        return filename.startswith("reclamm_heatmap_")

    build_calls = []
    plotted_files = []

    def fake_build_heatmap_matrices(**kwargs):
        build_calls.append(kwargs)
        return {
            "noise_vs_arb_geometric_improvement_pct": np.zeros(
                (len(kwargs["y_values"]), len(kwargs["x_values"])),
                dtype=float,
            )
        }

    def fake_plot_heatmap(**kwargs):
        plotted_files.append(kwargs["filename"])

    monkeypatch.setattr(script_module.os.path, "exists", fake_exists)
    monkeypatch.setattr(
        script_module,
        "build_heatmap_matrices",
        fake_build_heatmap_matrices,
    )
    monkeypatch.setattr(script_module, "plot_heatmap", fake_plot_heatmap)

    script_module.generate_heatmaps(
        base_cfg,
        price_data=None,
        launch_final_values=launch_final_values,
        cache={},
    )

    assert len(build_calls) == 1
    assert build_calls[0]["progress_label"] == pair_suffix
    assert build_calls[0]["base_cfg"][pair["fixed_key"]] == pytest.approx(
        slice_variant["value"]
    )
    assert build_calls[0]["metric_keys"] == [
        "noise_vs_arb_geometric_improvement_pct"
    ]
    assert plotted_files == [missing_file]


def test_generate_three_variable_3d_heatmaps_only_renders_missing_slice(
    monkeypatch,
    script_module,
    base_cfg,
    launch_final_values,
):
    missing_file = script_module.tvl_artifact_filename(
        "reclamm_heatmap_3d_geometric_vs_launch_geometric_symlog20",
        base_cfg,
        suffix="slice_q1",
    )

    def fake_exists(filename):
        if filename == missing_file:
            return False
        return filename.startswith("reclamm_heatmap_3d_")

    build_calls = []
    plotted_files = []

    def fake_build_heatmap_matrices(**kwargs):
        build_calls.append(kwargs)
        return {
            "geometric_vs_launch_geometric_pct": np.zeros(
                (len(kwargs["y_values"]), len(kwargs["x_values"])),
                dtype=float,
            )
        }

    def fake_plot_three_variable_heatmap_3d(**kwargs):
        plotted_files.append(kwargs["filename"])

    monkeypatch.setattr(script_module.os.path, "exists", fake_exists)
    monkeypatch.setattr(
        script_module,
        "build_heatmap_matrices",
        fake_build_heatmap_matrices,
    )
    monkeypatch.setattr(
        script_module,
        "plot_three_variable_heatmap_3d",
        fake_plot_three_variable_heatmap_3d,
    )

    script_module.generate_three_variable_3d_heatmaps(
        base_cfg,
        price_data=None,
        launch_final_values=launch_final_values,
        cache={},
    )

    assert len(build_calls) == 3
    assert {call["progress_label"] for call in build_calls} == {
        "3d_price_ratio_vs_margin_shift_exp_q1",
        "3d_shift_exp_vs_margin_price_ratio_q1",
        "3d_price_ratio_vs_shift_exp_margin_q1",
    }
    assert plotted_files == [missing_file]


def test_run_method_final_value_cached_reuses_persisted_parquet_value(
    monkeypatch,
    script_module,
    base_cfg,
):
    cfg = dict(base_cfg)
    cache_key = script_module._make_method_cache_key(cfg, "geometric")
    cache_key_hash = script_module._make_method_cache_hash(cache_key)

    class FakeFrame:
        empty = False

        def itertuples(self, index=False):
            return [
                types.SimpleNamespace(
                    cache_key_hash=cache_key_hash,
                    final_value=1_234_567.0,
                )
            ]

    monkeypatch.setattr(script_module.os.path, "exists", lambda filename: True)
    monkeypatch.setattr(script_module.pd, "read_parquet", lambda *args, **kwargs: FakeFrame())
    monkeypatch.setattr(
        script_module,
        "do_run_on_historic_data",
        lambda **kwargs: pytest.fail("persisted forward-value cache should be reused"),
    )

    cache = script_module.make_sweep_cache(price_data=None, cache_scope_cfg=cfg)
    value = script_module._run_method_final_value_cached(cfg, "geometric", cache)

    assert value == pytest.approx(1_234_567.0)


def test_run_method_final_value_cached_passes_preloaded_market_linear_arrays(
    monkeypatch,
    script_module,
    base_cfg,
):
    captured = {}
    shared_noise = {
        "arrays_path": script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH,
        "noise_base_array": np.array([11.0, 12.0]),
        "noise_tvl_coeff_array": np.array([21.0, 22.0]),
        "tvl_mean": 100.0,
        "tvl_std": 25.0,
    }

    monkeypatch.setattr(
        script_module,
        "_load_persistent_final_value_cache",
        lambda cache: cache.update(
            {
                "_persistent_final_value_cache_loaded": True,
                "_persistent_final_value_cache": {},
                "_persistent_final_value_next_batch_id": 0,
            }
        ),
    )
    monkeypatch.setattr(script_module, "flush_sweep_cache", lambda *args, **kwargs: None)

    def fake_do_run_on_historic_data(**kwargs):
        captured["run_fingerprint"] = kwargs["run_fingerprint"]
        return {"final_value": 1_111_111.0}

    monkeypatch.setattr(
        script_module,
        "do_run_on_historic_data",
        fake_do_run_on_historic_data,
    )

    cfg = {
        **base_cfg,
        "enable_noise_model": True,
        "noise_model": "market_linear",
    }
    cache = script_module.make_sweep_cache(
        price_data=None,
        cache_scope_cfg=cfg,
        market_linear_noise_data=shared_noise,
    )

    value = script_module._run_method_final_value_cached(cfg, "geometric", cache)

    assert value == pytest.approx(1_111_111.0)
    assert np.array_equal(
        captured["run_fingerprint"]["noise_base_array"],
        shared_noise["noise_base_array"],
    )
    assert np.array_equal(
        captured["run_fingerprint"]["noise_tvl_coeff_array"],
        shared_noise["noise_tvl_coeff_array"],
    )
    assert (
        captured["run_fingerprint"]["noise_arrays_path"]
        == script_module.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    )


def test_arc_speed_artifacts_only_build_missing_line_output(
    monkeypatch,
    script_module,
    base_cfg,
    launch_final_values,
):
    missing_line = script_module.tvl_artifact_filename("reclamm_line_efficiency", base_cfg, suffix="arc_speed_vs_price_ratio")

    def fake_exists(filename):
        if filename == missing_line:
            return False
        return filename.startswith("reclamm_")

    build_calls = []
    curve_calls = []
    plotted_lines = []

    def fake_build_heatmap_matrices(**kwargs):
        build_calls.append(kwargs)
        return {
            "efficiency_pct": np.zeros(
                (len(kwargs["y_values"]), len(kwargs["x_values"])),
                dtype=float,
            )
        }

    def fake_build_metric_curve(**kwargs):
        curve_calls.append(kwargs)
        return np.zeros(len(kwargs["x_values"]), dtype=float)

    monkeypatch.setattr(script_module.os.path, "exists", fake_exists)
    monkeypatch.setattr(script_module, "RUN_CONSTANT_ARC_LENGTH", True)
    monkeypatch.setattr(
        script_module,
        "compute_auto_calibrated_arc_length_speed",
        lambda cfg, price_data: 1.23e-4,
    )
    monkeypatch.setattr(
        script_module,
        "build_heatmap_matrices",
        fake_build_heatmap_matrices,
    )
    monkeypatch.setattr(
        script_module,
        "build_metric_curve",
        fake_build_metric_curve,
    )
    monkeypatch.setattr(
        script_module,
        "plot_heatmap",
        lambda **kwargs: pytest.fail("existing heatmap should not be redrawn"),
    )
    monkeypatch.setattr(
        script_module,
        "plot_arc_speed_line_chart",
        lambda **kwargs: plotted_lines.append(kwargs["filename"]),
    )

    script_module.generate_arc_speed_efficiency_artifacts(
        base_cfg=base_cfg,
        launch_cfg=dict(base_cfg),
        price_data=None,
        launch_final_values=launch_final_values,
        cache={},
    )

    assert len(build_calls) == 1
    assert build_calls[0]["progress_label"] == "arc_speed_vs_price_ratio"
    assert len(curve_calls) == 1
    assert curve_calls[0]["x_key"] == "arc_length_speed"
    assert plotted_lines == [missing_line]


def test_compute_auto_calibrated_arc_length_speed_uses_nearest_datetime_row(
    monkeypatch,
    script_module,
):
    real_pd = pytest.importorskip("pandas")
    monkeypatch.setattr(script_module.pd, "Timestamp", real_pd.Timestamp)
    monkeypatch.setattr(script_module.pd, "DatetimeIndex", real_pd.DatetimeIndex)
    monkeypatch.setattr(script_module.pd, "DataFrame", real_pd.DataFrame)
    monkeypatch.setattr(
        script_module.pd,
        "MultiIndex",
        real_pd.MultiIndex,
        raising=False,
    )

    price_data = real_pd.DataFrame(
        {
            "close_AAVE": [10.0, 30.0],
            "close_ETH": [1.0, 1.0],
        },
        index=real_pd.DatetimeIndex(
            ["2024-06-01 00:01:00", "2024-06-01 00:03:00"]
        ),
    )
    cfg = {
        "tokens": ["AAVE", "ETH"],
        "start": "2024-06-01 00:02:10",
        "price_ratio": 1.10,
        "centeredness_margin": 0.60,
        "daily_price_shift_exponent": 0.1,
        "initial_pool_value": 1_000_000.0,
    }
    captured = {}

    monkeypatch.setattr(
        script_module,
        "initialise_reclamm_reserves",
        lambda *args, **kwargs: (np.array([1.0, 1.0]), 1.0, 1.0),
    )
    monkeypatch.setattr(
        script_module,
        "compute_price_ratio",
        lambda *args, **kwargs: 1.0,
    )

    def fake_calibrate_arc_length_speed(*args, **kwargs):
        captured["market_price_0"] = args[7]
        return 123.0

    monkeypatch.setattr(
        script_module,
        "calibrate_arc_length_speed",
        fake_calibrate_arc_length_speed,
    )

    result = script_module.compute_auto_calibrated_arc_length_speed(cfg, price_data)

    assert result == pytest.approx(123.0)
    assert captured["market_price_0"] == pytest.approx(30.0)


def test_flush_sweep_cache_writes_compact_scalar_parquet(script_module):
    captured = {}

    class FakeFrame:
        def __init__(self, payload):
            captured["payload"] = payload

        def to_parquet(self, path, index=False, compression=None):
            captured["path"] = path
            captured["index"] = index
            captured["compression"] = compression

    script_module.pd.DataFrame = FakeFrame
    script_module.os.makedirs = lambda *args, **kwargs: captured.setdefault(
        "makedirs", args[0]
    )

    cache = {
        "_pending_persistent_final_values": {
            "abc123": {
                "cache_key_hash": "abc123",
                "final_value": 123.45,
                "method": "geometric",
                "enable_noise_model": True,
                "noise_model": "market_linear",
                "price_ratio": 1.1,
                "centeredness_margin": 0.6,
                "daily_price_shift_exponent": 0.1,
                "initial_pool_value": 5_000_000.0,
                "arb_frequency": 15,
            }
        },
        "_persistent_final_value_cache": {},
        "_persistent_final_value_next_batch_id": 0,
        "_persistent_final_value_cache_loaded": True,
        "_persistent_final_value_cache_path": "results/reclamm_heatmap_forward_cache/test/forward_values_tvl_5m.parquet",
    }

    script_module.flush_sweep_cache(cache, force=True)

    assert set(captured["payload"].keys()) == set(
        script_module.PERSISTED_FORWARD_VALUE_COLUMNS
    )
    assert captured["payload"]["cache_key_hash"] == ["abc123"]
    assert captured["payload"]["method"] == ["geometric"]
    assert captured["payload"]["arb_frequency"] == [15]
    assert captured["index"] is False
    assert captured["compression"] == "zstd"
    assert captured["makedirs"].endswith("forward_values_tvl_5m.parquet")
    assert captured["path"].endswith("forward_values_tvl_5m.parquet/batch_00000000.parquet")
    assert cache["_pending_persistent_final_values"] == {}
    assert cache["_persistent_final_value_cache"] == {"abc123": 123.45}
    assert cache["_persistent_final_value_next_batch_id"] == 1


def test_load_persistent_final_value_cache_reads_sharded_parquet_dir(
    monkeypatch,
    script_module,
):
    frames = {
        "results/reclamm_heatmap_forward_cache/test/forward_values_tvl_1m.parquet/batch_00000000.parquet": [
            types.SimpleNamespace(
                cache_key_hash="first123",
                final_value=111.0,
                method="geometric",
                enable_noise_model=True,
                noise_model="market_linear",
                price_ratio=1.1,
                centeredness_margin=0.6,
                daily_price_shift_exponent=0.1,
                initial_pool_value=1_000_000.0,
                arb_frequency=15,
            )
        ],
        "results/reclamm_heatmap_forward_cache/test/forward_values_tvl_1m.parquet/batch_00000001.parquet": [
            types.SimpleNamespace(
                cache_key_hash="second456",
                final_value=222.0,
                method="geometric",
                enable_noise_model=False,
                noise_model="arb_only",
                price_ratio=1.2,
                centeredness_margin=0.7,
                daily_price_shift_exponent=0.2,
                initial_pool_value=1_000_000.0,
                arb_frequency=15,
            )
        ],
    }

    class FakeFrame:
        def __init__(self, rows):
            self._rows = rows
            self.empty = not rows

        def itertuples(self, index=False):
            return list(self._rows)

    monkeypatch.setattr(script_module.os.path, "exists", lambda filename: True)
    monkeypatch.setattr(script_module.os.path, "isdir", lambda filename: True)
    monkeypatch.setattr(
        script_module.os,
        "listdir",
        lambda path: ["batch_00000001.parquet", "batch_00000000.parquet"],
    )
    monkeypatch.setattr(
        script_module.pd,
        "read_parquet",
        lambda path, *args, **kwargs: FakeFrame(frames[path]),
    )

    cache = {
        "_persistent_final_value_cache_loaded": False,
        "_persistent_final_value_cache_path": "results/reclamm_heatmap_forward_cache/test/forward_values_tvl_1m.parquet",
    }

    script_module._load_persistent_final_value_cache(cache)

    assert cache["_persistent_final_value_cache"] == {
        "first123": 111.0,
        "second456": 222.0,
    }
    assert cache["_persistent_final_value_next_batch_id"] == 2


def test_load_persistent_final_value_cache_supports_legacy_two_column_parquet(
    monkeypatch,
    script_module,
):
    class FakeFrame:
        empty = False

        def itertuples(self, index=False):
            return [
                types.SimpleNamespace(
                    cache_key_hash="legacy123",
                    final_value=999.0,
                )
            ]

    monkeypatch.setattr(script_module.os.path, "exists", lambda filename: True)
    monkeypatch.setattr(script_module.pd, "read_parquet", lambda *args, **kwargs: FakeFrame())

    cache = {
        "_persistent_final_value_cache_loaded": False,
        "_persistent_final_value_cache_path": "results/reclamm_heatmap_forward_cache/test/forward_values_tvl_1m.parquet",
    }

    script_module._load_persistent_final_value_cache(cache)

    assert cache["_persistent_final_value_cache"] == {"legacy123": 999.0}
    assert cache["_persistent_final_value_next_batch_id"] == 0


def test_make_sweep_cache_does_not_eagerly_load_persisted_parquet(
    monkeypatch,
    script_module,
):
    calls = []

    monkeypatch.setattr(
        script_module,
        "_load_persistent_final_value_cache",
        lambda cache: calls.append(dict(cache)),
    )

    cache = script_module.make_sweep_cache(price_data=None, cache_scope_cfg=None)

    assert calls == []
    assert cache["_persistent_final_value_cache"] == {}
    assert cache["_persistent_final_value_next_batch_id"] == 0
    assert cache["_persistent_final_value_cache_loaded"] is False


def test_run_comparison_cached_only_uses_geometric_runs_when_constant_arc_disabled(
    monkeypatch,
    script_module,
    base_cfg,
    launch_final_values,
):
    calls = []

    monkeypatch.setattr(
        script_module,
        "_make_comparison_cache_key",
        lambda cfg, launch_final_values: ("cache", round(float(cfg["price_ratio"]), 6)),
    )

    def fake_run_method_final_value_cached(cfg, method, cache):
        calls.append((cfg.get("enable_noise_model", False), method))
        return {
            (True, "geometric"): 1_050_000.0,
            (False, "geometric"): 1_000_000.0,
        }[(cfg.get("enable_noise_model", False), method)]

    monkeypatch.setattr(
        script_module,
        "_run_method_final_value_cached",
        fake_run_method_final_value_cached,
    )

    metrics = script_module.run_comparison_cached(
        base_cfg,
        cache={"_comparison_cache": {}, "_final_value_cache": {}, "_shared_price_data": None},
        launch_final_values=launch_final_values,
        metric_keys=(
            "geometric_vs_launch_geometric_pct",
            "noise_vs_arb_geometric_improvement_pct",
        ),
    )

    assert calls == [(True, "geometric"), (False, "geometric")]
    assert metrics == {
        "geometric_vs_launch_geometric_pct": pytest.approx(5.0),
        "noise_vs_arb_geometric_improvement_pct": pytest.approx(5.0),
    }
