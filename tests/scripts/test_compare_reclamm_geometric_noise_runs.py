"""Tests for adjacent-row sourcing in compare_reclamm_geometric_noise_runs.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "reclamm"
    / "compare_reclamm_geometric_noise_runs.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location(
        "test_compare_reclamm_geometric_noise_runs_module",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_run_specs_from_adjacent_row_maps_csv_cells_to_two_specs():
    module = load_script_module()
    row = {
        "metric_key": "noise_vs_arb_geometric_improvement_pct",
        "metric_unit": "pct",
        "source_noise_profile": "market_linear",
        "pair_slug": "price_ratio_vs_margin",
        "slice_slug": "q2",
        "adjacency_axis": "horizontal",
        "heatmap_value_diff_abs": 54.223889391076895,
        "1_price_ratio": 1.335,
        "1_centeredness_margin": 0.3184210526,
        "1_daily_price_shift_exponent": 0.1975,
        "1_tvl_usd": 1_000_000.0,
        "1_heatmap_value": -53.0543210862,
        "2_price_ratio": 1.36,
        "2_centeredness_margin": 0.3184210526,
        "2_daily_price_shift_exponent": 0.1975,
        "2_tvl_usd": 1_000_000.0,
        "2_heatmap_value": 1.1695683049,
    }

    description, run_specs = module.build_run_specs_from_adjacent_row(
        row,
        csv_path=Path("adjacent_pairs.csv"),
        row_index=0,
    )

    assert "adjacent_pairs.csv row 0" in description
    assert "price_ratio_vs_margin q2" in description
    assert "horizontal" in description
    assert "noise_profile=market_linear" in description
    assert len(run_specs) == 2
    assert run_specs[0]["name"] == "Top diff row cell 1"
    assert run_specs[0]["price_ratio"] == 1.335
    assert run_specs[0]["centeredness_margin"] == 0.3184210526
    assert run_specs[0]["daily_price_shift_exponent"] == 0.1975
    assert run_specs[0]["tvl_usd"] == 1_000_000.0
    assert run_specs[0]["color"] == "C0"
    assert run_specs[0]["source_noise_profile"] == "market_linear"
    assert "heatmap_value=-53.054321" in run_specs[0]["reason"]
    assert run_specs[1]["name"] == "Top diff row cell 2"
    assert run_specs[1]["price_ratio"] == 1.36
    assert run_specs[1]["color"] == "C1"


def test_default_output_file_for_adjacent_csv_uses_csv_stem_and_row_index():
    module = load_script_module()
    output = module.default_output_file_for_adjacent_csv(
        Path("scripts/results/reclamm_heatmap_adjacency/example.csv"),
        row_index=3,
    )

    assert (
        output.as_posix()
        == "scripts/results/reclamm_heatmap_adjacency/example_row_3_geometric_noise_compare.png"
    )


def test_build_run_config_rejects_legacy_calibrated_noise_profile():
    module = load_script_module()
    base_config = {
        "name": "base",
        "price_ratio": 1.1,
        "centeredness_margin": 0.6,
        "daily_price_shift_exponent": 0.1,
        "initial_pool_value": 1_000_000.0,
        "noise_model": "market_linear",
        "reclamm_noise_params": {"foo": 1.0},
        "noise_arrays_path": "path.npz",
    }
    spec = {
        "name": "cell",
        "price_ratio": 1.335,
        "centeredness_margin": 0.3184210526,
        "daily_price_shift_exponent": 0.1975,
        "tvl_usd": 1_000_000.0,
        "source_noise_profile": "legacy_calibrated",
    }

    import pytest

    with pytest.raises(ValueError):
        module.build_run_config(spec, base_config=base_config)


def test_build_run_variants_canonicalizes_noise_and_arb_only_configs():
    module = load_script_module()
    fixed_path = str(
        Path(__file__).resolve().parents[2]
        / "results"
        / "linear_market_noise"
        / "_sim_arrays"
        / "0x9d1fcf346ea1b0_2024-06-01_2026-03-01.npz"
    )
    base_config = {
        "name": "base",
        "price_ratio": 1.1,
        "centeredness_margin": 0.6,
        "daily_price_shift_exponent": 0.1,
        "initial_pool_value": 1_000_000.0,
        "noise_model": "market_linear",
        "noise_arrays_path": fixed_path,
    }
    spec = {
        "name": "cell",
        "price_ratio": 1.335,
        "centeredness_margin": 0.3184210526,
        "daily_price_shift_exponent": 0.1975,
        "tvl_usd": 1_000_000.0,
        "source_noise_profile": "market_linear",
    }

    class FakeThermostatCompare:
        @staticmethod
        def make_noise_variant_cfg(cfg, enable_noise_model):
            updated = dict(cfg)
            updated["enable_noise_model"] = bool(enable_noise_model)
            updated["noise_model"] = "market_linear" if enable_noise_model else "arb_only"
            updated["noise_arrays_path"] = fixed_path
            updated["reclamm_noise_params"] = {"tvl_mean": 1.0, "tvl_std": 2.0}
            return updated

    variants = module.build_run_variants(spec, base_config, FakeThermostatCompare)

    assert variants["noise"]["noise_model"] == "market_linear"
    assert variants["arb"]["noise_model"] == "arb_only"
    assert variants["noise"]["noise_arrays_path"] == fixed_path
    assert variants["arb"]["noise_arrays_path"] == fixed_path
    assert variants["noise"]["reclamm_noise_params"] == {"tvl_mean": 1.0, "tvl_std": 2.0}
    assert variants["arb"]["reclamm_noise_params"] == {"tvl_mean": 1.0, "tvl_std": 2.0}


def test_print_run_inputs_to_terminal_includes_fingerprint_and_update_params(capsys):
    module = load_script_module()
    cfg = {
        "name": "cell",
        "variant_label": "arb-only",
    }
    run_fingerprint = {
        "tokens": ["AAVE", "ETH"],
        "fees": np.float64(0.0025),
        "arb_frequency": np.int64(14),
    }
    update_params = {
        "price_ratio": np.array(1.335),
        "centeredness_margin": np.array(0.3184210526),
        "daily_price_shift_base": np.array(0.99999841596),
    }

    module.print_run_inputs_to_terminal(cfg, run_fingerprint, update_params)

    captured = capsys.readouterr().out
    assert "Run inputs for cell (arb-only):" in captured
    assert '"run_fingerprint"' in captured
    assert '"update_params"' in captured
    assert '"tokens": [' in captured
    assert '"AAVE"' in captured
    assert '"arb_frequency": 14' in captured
    assert '"price_ratio": 1.335' in captured
