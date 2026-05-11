"""Tests for cache-backed adjacent heatmap pair detection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "reclamm"
    / "find_adjacent_heatmap_pairs.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "test_find_adjacent_heatmap_pairs_module",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_cell(x_index, y_index, heatmap_value, **overrides):
    cell = {
        "metric_key": "noise_vs_arb_geometric_improvement_pct",
        "metric_unit": "pct",
        "pair_slug": "price_ratio_vs_margin",
        "slice_slug": "q2",
        "slice_label": "Q2",
        "fixed_key": "daily_price_shift_exponent",
        "fixed_value": 0.1975,
        "price_ratio": 1.01 + 0.1 * x_index,
        "centeredness_margin": 0.05 + 0.1 * y_index,
        "daily_price_shift_exponent": 0.1975,
        "tvl_usd": 1_000_000.0,
        "heatmap_value": float(heatmap_value),
        "x_index": int(x_index),
        "y_index": int(y_index),
    }
    cell.update(overrides)
    return cell


def test_find_adjacent_rows_for_slice_filters_and_sorts_descending():
    module = load_script_module()
    records_by_coord = {
        (0, 0): make_cell(0, 0, 0.0),
        (0, 1): make_cell(1, 0, 35.0),
        (0, 2): make_cell(2, 0, -10.0),
        (1, 0): make_cell(0, 1, 5.0),
        (1, 1): make_cell(1, 1, -40.0),
        (1, 2): make_cell(2, 1, -50.0),
    }

    rows = module.find_adjacent_rows_for_slice(
        metric_key="noise_vs_arb_geometric_improvement_pct",
        metric_unit="pct",
        records_by_coord=records_by_coord,
        x_count=3,
        y_count=2,
        min_diff=30.0,
    )

    assert [row["heatmap_value_diff_abs"] for row in rows] == [75.0, 45.0, 45.0, 40.0, 35.0]
    assert rows[0]["adjacency_axis"] == "vertical"
    assert rows[0]["1_x_index"] == 1
    assert rows[0]["1_y_index"] == 0
    assert rows[0]["2_x_index"] == 1
    assert rows[0]["2_y_index"] == 1

    horizontal_rows = module.find_adjacent_rows_for_slice(
        metric_key="noise_vs_arb_geometric_improvement_pct",
        metric_unit="pct",
        records_by_coord=records_by_coord,
        x_count=3,
        y_count=2,
        min_diff=30.0,
        adjacency_axis="horizontal",
    )
    assert {row["adjacency_axis"] for row in horizontal_rows} == {"horizontal"}

    vertical_rows = module.find_adjacent_rows_for_slice(
        metric_key="noise_vs_arb_geometric_improvement_pct",
        metric_unit="pct",
        records_by_coord=records_by_coord,
        x_count=3,
        y_count=2,
        min_diff=30.0,
        adjacency_axis="vertical",
    )
    assert {row["adjacency_axis"] for row in vertical_rows} == {"vertical"}


def test_build_slice_cell_grid_reconstructs_metric_values_from_cache_hashes():
    module = load_script_module()

    class FakeCompareModule:
        @staticmethod
        def make_noise_variant_cfg(cfg, enable_noise_model):
            updated = dict(cfg)
            updated["enable_noise_model"] = bool(enable_noise_model)
            return updated

        @staticmethod
        def _make_method_cache_key(cfg, method):
            return (
                method,
                bool(cfg["enable_noise_model"]),
                round(float(cfg["price_ratio"]), 6),
                round(float(cfg["centeredness_margin"]), 6),
                round(float(cfg["daily_price_shift_exponent"]), 6),
                round(float(cfg["initial_pool_value"]), 2),
            )

        @staticmethod
        def _make_method_cache_hash(key):
            return repr(key)

        @staticmethod
        def get_initial_pool_value(cfg):
            return float(cfg["initial_pool_value"])

    base_cfg = {
        "price_ratio": 1.1,
        "centeredness_margin": 0.3,
        "daily_price_shift_exponent": 0.2,
        "initial_pool_value": 1_000_000.0,
    }
    pair_spec = {
        "slug": "price_ratio_vs_margin",
        "x_values": [1.1, 1.2],
        "y_values": [0.3, 0.4],
        "x_key": "price_ratio",
        "y_key": "centeredness_margin",
        "fixed_key": "daily_price_shift_exponent",
    }
    slice_variant = {
        "slug": "q2",
        "label": "Q2",
        "value": 0.2,
    }

    heatmap_targets = {
        (0, 0): (130.0, 100.0),  # +30%
        (0, 1): (200.0, 100.0),  # +100%
        (1, 0): (70.0, 100.0),   # -30%
        (1, 1): (160.0, 100.0),  # +60%
    }
    cache_lookup = {}
    for (y_index, x_index), (noise_geo, arb_geo) in heatmap_targets.items():
        cfg = dict(base_cfg)
        cfg["price_ratio"] = pair_spec["x_values"][x_index]
        cfg["centeredness_margin"] = pair_spec["y_values"][y_index]
        noise_cfg, noise_method = FakeCompareModule.make_noise_variant_cfg(cfg, True), "geometric"
        arb_cfg, arb_method = FakeCompareModule.make_noise_variant_cfg(cfg, False), "geometric"

        noise_key = FakeCompareModule._make_method_cache_key(noise_cfg, noise_method)
        arb_key = FakeCompareModule._make_method_cache_key(arb_cfg, arb_method)
        cache_lookup[FakeCompareModule._make_method_cache_hash(noise_key)] = noise_geo
        cache_lookup[FakeCompareModule._make_method_cache_hash(arb_key)] = arb_geo

    slice_scan = module.build_slice_cell_grid(
        compare_module=FakeCompareModule,
        base_cfg=base_cfg,
        pair_spec=pair_spec,
        slice_variant=slice_variant,
        metric_key="noise_vs_arb_geometric_improvement_pct",
        cache_lookup=cache_lookup,
    )

    assert slice_scan["resolved_cell_count"] == 4
    assert slice_scan["missing_hash_count"] == 0
    assert slice_scan["records_by_coord"][(0, 0)]["heatmap_value"] == pytest.approx(30.0)
    assert slice_scan["records_by_coord"][(0, 1)]["heatmap_value"] == pytest.approx(100.0)
    assert slice_scan["records_by_coord"][(1, 0)]["heatmap_value"] == pytest.approx(-30.0)
    assert slice_scan["records_by_coord"][(1, 1)]["heatmap_value"] == pytest.approx(60.0)

    rows = module.find_adjacent_rows_for_slice(
        metric_key="noise_vs_arb_geometric_improvement_pct",
        metric_unit="pct",
        records_by_coord=slice_scan["records_by_coord"],
        x_count=2,
        y_count=2,
        min_diff=30.0,
    )

    assert [row["heatmap_value_diff_abs"] for row in rows] == pytest.approx([90.0, 70.0, 60.0, 40.0])
    assert rows[0]["1_heatmap_value"] == pytest.approx(-30.0)
    assert rows[0]["2_heatmap_value"] == pytest.approx(60.0)


def test_run_top_row_geometric_comparison_dispatches_to_compare_module(monkeypatch):
    module = load_script_module()
    captured = {}

    class FakeCompareModule:
        @staticmethod
        def run_adjacent_csv_row_comparison(csv_path, row_index=0, output_file=None):
            captured["csv_path"] = csv_path
            captured["row_index"] = row_index
            captured["output_file"] = output_file
            return "fake-output.png"

    monkeypatch.setattr(
        module,
        "load_geometric_compare_module",
        lambda module_path=None: FakeCompareModule,
    )

    output = module.run_top_row_geometric_comparison(
        Path("tmp_adjacent.csv"),
        output_file="custom.png",
        row_index=0,
    )

    assert output == "fake-output.png"
    assert captured == {
        "csv_path": Path("tmp_adjacent.csv"),
        "row_index": 0,
        "output_file": "custom.png",
    }


def test_autodetect_lightweight_noise_profile_keeps_market_linear():
    module = load_script_module()
    compare_context = module._LightweightCompareContext()
    base_cfg = compare_context.configs_for_tvl(compare_context.CONFIGS, 1_000_000.0)[1]
    pair_spec = compare_context.get_pair_heatmap_specs(base_cfg)[0]

    compare_context.set_noise_profile("market_linear")
    module.autodetect_lightweight_noise_profile(
        compare_module=compare_context,
        base_cfg=base_cfg,
        pair_specs=[pair_spec],
        metric_key="noise_vs_arb_geometric_improvement_pct",
        slice_slug="q2",
        cache_lookup={},
    )

    assert compare_context.noise_profile == "market_linear"


def test_lightweight_context_rejects_legacy_noise_profile():
    module = load_script_module()
    compare_context = module._LightweightCompareContext()

    with pytest.raises(ValueError):
        compare_context.set_noise_profile("legacy_calibrated")


def test_source_variant_sets_explicit_market_and_arb_only_noise_models():
    module = load_script_module()
    compare_context = module._LightweightCompareContext()
    cfg = compare_context.configs_for_tvl(compare_context.CONFIGS, 1_000_000.0)[1]

    noise_cfg, noise_method = module._source_variant(
        compare_context,
        cfg,
        "noise_geometric",
    )
    arb_cfg, arb_method = module._source_variant(
        compare_context,
        cfg,
        "arb_geometric",
    )

    assert noise_method == "geometric"
    assert noise_cfg["enable_noise_model"] is True
    assert noise_cfg["noise_model"] == "market_linear"
    assert noise_cfg["noise_arrays_path"] == compare_context.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH

    assert arb_method == "geometric"
    assert arb_cfg["enable_noise_model"] is False
    assert arb_cfg["noise_model"] == "arb_only"
    assert arb_cfg["noise_arrays_path"] == compare_context.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH

    resolved_arb = compare_context.resolve_reclamm_noise_settings(arb_cfg)
    assert resolved_arb["noise_model"] == "arb_only"
    assert resolved_arb["noise_arrays_path"] == compare_context.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    assert set(resolved_arb["reclamm_noise_params"]) == {"tvl_mean", "tvl_std"}
    assert resolved_arb["noise_cache_key"][0] == "arb_only"


def test_lightweight_context_defaults_to_fixed_compare_arb_cadence():
    module = load_script_module()
    compare_context = module._LightweightCompareContext()
    cfg = compare_context.configs_for_tvl(compare_context.CONFIGS, 1_000_000.0)[1]
    cfg["arb_frequency"] = 6
    cfg["gas_cost"] = 99.0
    cfg["protocol_fee_split"] = 0.9

    arb_cfg = compare_context.make_noise_variant_cfg(cfg, False)
    resolved_noise = compare_context.resolve_reclamm_noise_settings(cfg)

    assert arb_cfg["arb_frequency"] == compare_context.FIXED_COMPARE_ARB_FREQUENCY
    assert arb_cfg["noise_model"] == "arb_only"
    assert arb_cfg["noise_arrays_path"] == compare_context.DEFAULT_MARKET_LINEAR_NOISE_ARRAYS_PATH
    assert resolved_noise["arb_frequency"] == compare_context.FIXED_COMPARE_ARB_FREQUENCY
    assert arb_cfg["gas_cost"] == compare_context.DEFAULT_GAS_COST
    assert arb_cfg["protocol_fee_split"] == compare_context.DEFAULT_PROTOCOL_FEE_SPLIT
