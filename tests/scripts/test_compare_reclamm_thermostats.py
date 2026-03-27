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
    inject_module("pandas", pandas_module)

    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    pyplot_module.cm = types.SimpleNamespace(viridis=lambda values: values)
    colors_module = types.ModuleType("matplotlib.colors")
    colors_module.TwoSlopeNorm = object
    colors_module.Normalize = object
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
        "price_ratio": 1.10,
        "centeredness_margin": 0.60,
        "daily_price_shift_exponent": 0.1,
    }


@pytest.fixture
def launch_final_values():
    return {
        "geometric": 1_000_000.0,
        "constant_arc_length": 1_010_000.0,
    }


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
    missing_file = "reclamm_heatmap_efficiency_price_ratio_vs_margin.png"

    def fake_exists(filename):
        if filename == missing_file:
            return False
        return filename.startswith("reclamm_heatmap_")

    build_calls = []
    plotted_files = []

    def fake_build_heatmap_matrices(**kwargs):
        build_calls.append(kwargs)
        return {
            "efficiency_pct": np.zeros(
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
    assert build_calls[0]["progress_label"] == "price_ratio_vs_margin"
    assert build_calls[0]["metric_keys"] == ["efficiency_pct"]
    assert plotted_files == [missing_file]


def test_arc_speed_artifacts_only_build_missing_line_output(
    monkeypatch,
    script_module,
    base_cfg,
    launch_final_values,
):
    missing_line = "reclamm_line_efficiency_arc_speed_vs_price_ratio.png"

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
