"""Sweep reCLAMM arc speed jointly with price ratio and shift exponent.

This is a focused companion to ``compare_reclamm_thermostats.py``.  It reuses
that script's configs, cache, market-linear noise setup, and heatmap metrics,
then evaluates the 3-variable cube:

    daily_price_shift_exponent x price_ratio x arc_length_speed

The default run is the full compare-grid for the 1M TVL aggressive/tight-range
config, with the launch-style config used as the benchmark where required.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from pathlib import Path

# Keep background runs from depending on a writable user-level matplotlib cache.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DEFAULT_METRIC_KEYS = (
    "efficiency_pct",
    "noise_constant_arc_final_value_musd",
)
DEFAULT_FACET_SHIFT_VALUES = (0.01, 0.05, 0.10, 0.20, 0.35, 0.50)
SUPPORTED_METRIC_KEYS = (
    "constant_arc_vs_launch_constant_arc_pct",
    "efficiency_pct",
    "geometric_vs_launch_geometric_pct",
    "launch_geometric_efficiency_pct",
    "noise_constant_arc_final_value_musd",
    "noise_geometric_final_value_musd",
    "noise_vs_arb_constant_arc_improvement_pct",
    "noise_vs_arb_geometric_improvement_pct",
)


def load_compare_module():
    """Import the heavy compare module only after CLI parsing."""
    import compare_reclamm_thermostats as compare_module

    return compare_module


def parse_csv_floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""
    values = []
    for token in value.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for a long-running background sweep."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the reCLAMM 3-variable arc-speed/price-ratio/shift-exponent "
            "sweep using the compare_reclamm_thermostats cache and configs."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="results/reclamm_arc_speed_shift_price_sweep",
        help="Directory for cube parquet files and plots.",
    )
    parser.add_argument(
        "--tvl",
        nargs="+",
        type=float,
        default=None,
        help=(
            "One or more initial pool values to run. Defaults to 1,000,000. "
            "Use --all-tvls for the compare script's 1M/5M/20M sweep."
        ),
    )
    parser.add_argument(
        "--all-tvls",
        action="store_true",
        help="Run every TVL from compare_reclamm_thermostats.TVL_SWEEP_VALUES.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=SUPPORTED_METRIC_KEYS,
        default=None,
        help=(
            "Metric to compute. Can be repeated. Defaults to efficiency_pct and "
            "noise_constant_arc_final_value_musd."
        ),
    )
    parser.add_argument(
        "--facet-shift-values",
        type=parse_csv_floats,
        default=DEFAULT_FACET_SHIFT_VALUES,
        help=(
            "Comma-separated shift exponents to include in the facet overview. "
            "Nearest grid values are used."
        ),
    )
    parser.add_argument(
        "--plot-all-shift-slices",
        action="store_true",
        help="Also save one 2D arc-speed/price-ratio heatmap per shift exponent.",
    )
    parser.add_argument(
        "--no-orthogonal-3d",
        action="store_true",
        help="Skip the literal 3D orthogonal-slice plot.",
    )
    parser.add_argument(
        "--skip-cube-parquet",
        action="store_true",
        help="Do not save the evaluated parameter cube to parquet.",
    )
    return parser.parse_args()


def ensure_output_dir(path: str | os.PathLike[str]) -> Path:
    """Create and return the output directory."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def tvl_output_path(
    output_dir: Path,
    stem: str,
    cfg: dict,
    suffix: str | None = None,
    ext: str = "png",
) -> Path:
    """Build a stable output path with the compare script's TVL slug."""
    parts = [stem]
    if suffix:
        parts.append(suffix)
    parts.append(f"tvl_{ct.format_tvl_millions_slug(cfg)}")
    return output_dir / ("_".join(parts) + f".{ext}")


def metric_spec_map() -> dict[str, dict]:
    """Return the compare script's metric specs, keyed by metric name."""
    return {spec["key"]: spec for spec in ct.get_pair_heatmap_metric_specs()}


def nearest_indices(values: np.ndarray, targets: tuple[float, ...]) -> list[int]:
    """Resolve target values to unique nearest indices in a sweep grid."""
    values = np.asarray(values, dtype=float)
    indices: list[int] = []
    for target in targets:
        idx = int(np.argmin(np.abs(values - float(target))))
        if idx not in indices:
            indices.append(idx)
    return indices


def speed_label(value: float) -> str:
    """Format an arc speed for plot tick labels."""
    return f"{float(value):.0e}"


def heatmap_value_slug(value: float) -> str:
    """Format a sweep value for filenames."""
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def build_arc_speed_shift_price_cube(
    base_cfg: dict,
    arc_length_speeds: np.ndarray,
    price_ratios: np.ndarray,
    shift_exponents: np.ndarray,
    metric_keys: tuple[str, ...],
    cache: dict,
    launch_final_values: dict,
) -> dict[str, np.ndarray]:
    """Evaluate metric cubes as shift_exp x price_ratio x arc_length_speed."""
    data = {
        metric_key: np.zeros(
            (len(shift_exponents), len(price_ratios), len(arc_length_speeds)),
            dtype=float,
        )
        for metric_key in metric_keys
    }
    total_points = len(shift_exponents) * len(price_ratios) * len(arc_length_speeds)
    print(
        "\nStarting 3-variable arc-speed sweep: "
        f"{len(shift_exponents)} shift slices x {len(price_ratios)} price ratios x "
        f"{len(arc_length_speeds)} arc speeds = {total_points} parameter points."
    )

    for zi, shift_exp in enumerate(shift_exponents):
        slice_cfg = dict(base_cfg)
        slice_cfg["daily_price_shift_exponent"] = float(shift_exp)
        progress_label = (
            "arc_speed_shift_price_"
            f"shift_{zi + 1:03d}_of_{len(shift_exponents):03d}_"
            f"{float(shift_exp):.4f}"
        )
        slice_data = ct.build_heatmap_matrices(
            x_values=arc_length_speeds,
            y_values=price_ratios,
            x_key="arc_length_speed",
            y_key="price_ratio",
            base_cfg=slice_cfg,
            metric_keys=metric_keys,
            cache=cache,
            progress_label=progress_label,
            launch_final_values=launch_final_values,
        )
        for metric_key in metric_keys:
            data[metric_key][zi, :, :] = slice_data[metric_key]
        ct.flush_sweep_cache(cache, force=True)
        del slice_data
        gc.collect()

    return data


def cube_to_frame(
    cube: dict[str, np.ndarray],
    arc_length_speeds: np.ndarray,
    price_ratios: np.ndarray,
    shift_exponents: np.ndarray,
) -> pd.DataFrame:
    """Flatten the 3D cube into a tidy table for offline analysis."""
    records = []
    for zi, shift_exp in enumerate(shift_exponents):
        for yi, price_ratio in enumerate(price_ratios):
            for xi, arc_speed in enumerate(arc_length_speeds):
                record = {
                    "shift_index": zi,
                    "price_ratio_index": yi,
                    "arc_length_speed_index": xi,
                    "daily_price_shift_exponent": float(shift_exp),
                    "price_ratio": float(price_ratio),
                    "arc_length_speed": float(arc_speed),
                }
                for metric_key, values in cube.items():
                    record[metric_key] = float(values[zi, yi, xi])
                records.append(record)
    return pd.DataFrame.from_records(records)


def save_cube_parquet(
    output_dir: Path,
    cfg: dict,
    cube: dict[str, np.ndarray],
    arc_length_speeds: np.ndarray,
    price_ratios: np.ndarray,
    shift_exponents: np.ndarray,
) -> Path:
    """Persist the full parameter cube as parquet."""
    output_path = tvl_output_path(
        output_dir,
        "reclamm_arc_speed_shift_price_cube",
        cfg,
        ext="parquet",
    )
    frame = cube_to_frame(
        cube,
        arc_length_speeds=arc_length_speeds,
        price_ratios=price_ratios,
        shift_exponents=shift_exponents,
    )
    frame.to_parquet(output_path, index=False, compression="zstd")
    print(f"Saved {output_path}")
    return output_path


def plot_shift_slice_facets(
    data: np.ndarray,
    arc_length_speeds: np.ndarray,
    price_ratios: np.ndarray,
    shift_exponents: np.ndarray,
    shift_indices: list[int],
    spec: dict,
    cfg: dict,
    filename: Path,
) -> None:
    """Render selected shift-exponent slices as small-multiple heatmaps."""
    selected = [data[idx] for idx in shift_indices]
    norm = ct._build_heatmap_norm(
        selected,
        center_zero=spec["center_zero"],
        color_norm=spec.get("color_norm"),
        symlog_linthresh=spec.get("symlog_linthresh"),
    )
    cmap_name = spec["cmap"]
    x_edges = ct._compute_axis_edges(arc_length_speeds, scale="log")
    y_edges = ct._compute_axis_edges(price_ratios, scale="linear")

    col_count = min(3, len(shift_indices))
    row_count = int(math.ceil(len(shift_indices) / col_count))
    fig, axes = plt.subplots(
        row_count,
        col_count,
        figsize=(4.3 * col_count, 3.3 * row_count),
        squeeze=False,
    )
    active_axes = []
    im = None
    for plot_idx, shift_idx in enumerate(shift_indices):
        ax = axes[plot_idx // col_count][plot_idx % col_count]
        active_axes.append(ax)
        im = ax.pcolormesh(
            x_edges,
            y_edges,
            data[shift_idx],
            cmap=cmap_name,
            norm=norm,
            shading="auto",
        )
        ax.set_xscale("log")
        ax.set_xticks(ct.ARC_LENGTH_SPEED_TICKS)
        ax.set_yticks(ct.PRICE_RATIO_TICKS)
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_xlabel("Arc-length speed")
        ax.set_ylabel("Price ratio")
        ax.set_title(
            f"shift_exp={ct.format_heatmap_param_value(shift_exponents[shift_idx])}"
        )

    for plot_idx in range(len(shift_indices), row_count * col_count):
        axes[plot_idx // col_count][plot_idx % col_count].set_visible(False)

    fig.suptitle(
        f"{spec['title']}: arc_speed x price_ratio by shift_exp | "
        f"TVL {ct.format_tvl_millions_label(cfg)}",
        y=0.995,
    )
    if im is not None:
        cbar = fig.colorbar(im, ax=active_axes, shrink=0.88)
        cbar.set_label(spec["colorbar_label"])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close(fig)


def plot_all_shift_slices(
    output_dir: Path,
    cube: dict[str, np.ndarray],
    arc_length_speeds: np.ndarray,
    price_ratios: np.ndarray,
    shift_exponents: np.ndarray,
    specs: dict[str, dict],
    cfg: dict,
    metric_keys: tuple[str, ...],
) -> None:
    """Optionally save one 2D heatmap per shift exponent and metric."""
    for metric_key in metric_keys:
        spec = specs[metric_key]
        for shift_idx, shift_exp in enumerate(shift_exponents):
            filename = tvl_output_path(
                output_dir,
                f"reclamm_arc_speed_price_slice_{spec['slug']}",
                cfg,
                suffix=f"shift_exp_{heatmap_value_slug(float(shift_exp))}",
            )
            ct.plot_heatmap(
                data=cube[metric_key][shift_idx],
                x_values=arc_length_speeds,
                y_values=price_ratios,
                x_label="Arc-length speed",
                y_label="Price ratio",
                title=(
                    f"{spec['title']}: shift_exp fixed at "
                    f"{ct.format_heatmap_param_value(float(shift_exp))} | "
                    f"TVL {ct.format_tvl_millions_label(cfg)}"
                ),
                colorbar_label=spec["colorbar_label"],
                filename=filename,
                xticks=ct.ARC_LENGTH_SPEED_TICKS,
                yticks=ct.PRICE_RATIO_TICKS,
                xscale="log",
                center_zero=spec["center_zero"],
                cmap=spec["cmap"],
                color_norm=spec.get("color_norm"),
                symlog_linthresh=spec.get("symlog_linthresh"),
            )


def compute_argmax_over_arc_speed(
    data: np.ndarray,
    arc_length_speeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return best metric value and the arc speed that produced it."""
    best_idx = np.nanargmax(data, axis=2)
    best_values = np.take_along_axis(data, best_idx[:, :, None], axis=2)[:, :, 0]
    best_speeds = np.asarray(arc_length_speeds, dtype=float)[best_idx]
    return best_values, best_speeds


def plot_best_speed_heatmap(
    best_speeds: np.ndarray,
    arc_length_speeds: np.ndarray,
    price_ratios: np.ndarray,
    shift_exponents: np.ndarray,
    metric_label: str,
    cfg: dict,
    filename: Path,
) -> None:
    """Render a price_ratio x shift_exp heatmap of the selected arc speed."""
    x_edges = ct._compute_axis_edges(price_ratios, scale="linear")
    y_edges = ct._compute_axis_edges(shift_exponents, scale="linear")
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    norm = LogNorm(
        vmin=float(np.min(arc_length_speeds)),
        vmax=float(np.max(arc_length_speeds)),
    )
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        best_speeds,
        cmap="viridis",
        norm=norm,
        shading="auto",
    )
    ax.set_xlabel("Price ratio")
    ax.set_ylabel("Shift exponent")
    ax.set_title(
        f"Best arc-length speed by {metric_label} | "
        f"TVL {ct.format_tvl_millions_label(cfg)}"
    )
    ax.set_xticks(ct.PRICE_RATIO_TICKS)
    ax.set_yticks(ct.SHIFT_EXPONENT_TICKS)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best arc-length speed")
    speed_ticks = np.asarray(ct.ARC_LENGTH_SPEED_TICKS, dtype=float)
    speed_ticks = speed_ticks[
        (speed_ticks >= float(np.min(arc_length_speeds)))
        & (speed_ticks <= float(np.max(arc_length_speeds)))
    ]
    cbar.set_ticks(speed_ticks)
    cbar.set_ticklabels([speed_label(value) for value in speed_ticks])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close(fig)


def save_best_speed_summary(
    output_dir: Path,
    cfg: dict,
    metric_key: str,
    best_values: np.ndarray,
    best_speeds: np.ndarray,
    price_ratios: np.ndarray,
    shift_exponents: np.ndarray,
) -> Path:
    """Persist the price_ratio x shift_exp argmax summary."""
    records = []
    for zi, shift_exp in enumerate(shift_exponents):
        for yi, price_ratio in enumerate(price_ratios):
            records.append(
                {
                    "daily_price_shift_exponent": float(shift_exp),
                    "price_ratio": float(price_ratio),
                    f"best_{metric_key}": float(best_values[zi, yi]),
                    f"best_arc_length_speed_by_{metric_key}": float(
                        best_speeds[zi, yi]
                    ),
                }
            )
    frame = pd.DataFrame.from_records(records)
    output_path = tvl_output_path(
        output_dir,
        f"reclamm_arc_speed_shift_price_best_{metric_key}",
        cfg,
        ext="parquet",
    )
    frame.to_parquet(output_path, index=False, compression="zstd")
    print(f"Saved {output_path}")
    return output_path


def plot_orthogonal_3d_slices(
    data: np.ndarray,
    arc_length_speeds: np.ndarray,
    price_ratios: np.ndarray,
    shift_exponents: np.ndarray,
    spec: dict,
    cfg: dict,
    launch_auto_speed: float,
    filename: Path,
) -> None:
    """Render one literal 3D orthogonal-slice view of the metric cube."""
    shift_idx = int(
        np.argmin(
            np.abs(
                np.asarray(shift_exponents, dtype=float)
                - float(cfg["daily_price_shift_exponent"])
            )
        )
    )
    price_idx = int(
        np.argmin(
            np.abs(np.asarray(price_ratios, dtype=float) - float(cfg["price_ratio"]))
        )
    )
    speed_idx = int(
        np.argmin(
            np.abs(
                np.asarray(arc_length_speeds, dtype=float) - float(launch_auto_speed)
            )
        )
    )

    norm = ct._build_heatmap_norm(
        [data],
        center_zero=spec["center_zero"],
        color_norm=spec.get("color_norm"),
        symlog_linthresh=spec.get("symlog_linthresh"),
    )
    cmap_obj = plt.get_cmap(spec["cmap"])
    log_speeds = np.log10(np.asarray(arc_length_speeds, dtype=float))

    arc_price_x, arc_price_y = np.meshgrid(log_speeds, price_ratios)
    arc_price_z = np.full_like(
        arc_price_x,
        float(shift_exponents[shift_idx]),
        dtype=float,
    )

    arc_shift_x, arc_shift_z = np.meshgrid(log_speeds, shift_exponents)
    arc_shift_y = np.full_like(
        arc_shift_x,
        float(price_ratios[price_idx]),
        dtype=float,
    )

    price_shift_y, price_shift_z = np.meshgrid(price_ratios, shift_exponents)
    price_shift_x = np.full_like(
        price_shift_y,
        float(log_speeds[speed_idx]),
        dtype=float,
    )

    fig = plt.figure(figsize=(10.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        arc_price_x,
        arc_price_y,
        arc_price_z,
        facecolors=cmap_obj(norm(data[shift_idx, :, :])),
        shade=False,
    )
    ax.plot_surface(
        arc_shift_x,
        arc_shift_y,
        arc_shift_z,
        facecolors=cmap_obj(norm(data[:, price_idx, :])),
        shade=False,
    )
    ax.plot_surface(
        price_shift_x,
        price_shift_y,
        price_shift_z,
        facecolors=cmap_obj(norm(data[:, :, speed_idx])),
        shade=False,
    )

    ax.set_xlabel("Arc-length speed")
    ax.set_ylabel("Price ratio")
    ax.set_zlabel("Shift exponent")
    ax.set_yticks(ct.PRICE_RATIO_TICKS)
    ax.set_zticks(ct.SHIFT_EXPONENT_TICKS)
    speed_ticks = np.asarray(ct.ARC_LENGTH_SPEED_TICKS, dtype=float)
    speed_ticks = speed_ticks[
        (speed_ticks >= float(np.min(arc_length_speeds)))
        & (speed_ticks <= float(np.max(arc_length_speeds)))
    ]
    ax.set_xticks(np.log10(speed_ticks))
    ax.set_xticklabels([speed_label(value) for value in speed_ticks], rotation=20)
    ax.set_title(
        f"{spec['title']}: orthogonal 3D slices | "
        f"TVL {ct.format_tvl_millions_label(cfg)}\n"
        f"shift_exp={ct.format_heatmap_param_value(shift_exponents[shift_idx])}, "
        f"price_ratio={ct.format_heatmap_param_value(price_ratios[price_idx])}, "
        f"arc_speed={speed_label(arc_length_speeds[speed_idx])}"
    )
    ax.view_init(elev=ct.THREE_D_VIEW_ELEVATION, azim=ct.THREE_D_VIEW_AZIMUTH)
    try:
        ax.set_box_aspect((1.45, 1.5, 1.0))
    except AttributeError:
        pass

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1, shrink=0.82)
    cbar.set_label(spec["colorbar_label"])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close(fig)


def run_for_tvl(
    initial_pool_value: float,
    output_dir: Path,
    metric_keys: tuple[str, ...],
    facet_shift_values: tuple[float, ...],
    plot_all_slices: bool,
    plot_orthogonal_3d: bool,
    save_cube: bool,
    shared_price_data,
    shared_market_linear_noise_data,
) -> None:
    """Run the 3D sweep for one initial TVL."""
    launch_cfg, base_cfg = ct.configs_for_tvl(ct.CONFIGS, initial_pool_value)
    tvl_label = ct.format_tvl_millions_label(base_cfg)
    print(f"\n=== Arc-speed/shift/price sweep: TVL {tvl_label} ===")

    launch_final_values = ct.get_launch_final_values(
        [],
        launch_cfg=launch_cfg,
        price_data=shared_price_data,
        market_linear_noise_data=shared_market_linear_noise_data,
    )
    cache = ct.make_sweep_cache(
        shared_price_data,
        cache_scope_cfg=base_cfg,
        market_linear_noise_data=shared_market_linear_noise_data,
    )
    try:
        cube = build_arc_speed_shift_price_cube(
            base_cfg=dict(base_cfg),
            arc_length_speeds=ct.HEATMAP_ARC_LENGTH_SPEEDS,
            price_ratios=ct.HEATMAP_PRICE_RATIOS,
            shift_exponents=ct.HEATMAP_SHIFT_EXPONENTS,
            metric_keys=metric_keys,
            cache=cache,
            launch_final_values=launch_final_values,
        )
        if save_cube:
            save_cube_parquet(
                output_dir,
                base_cfg,
                cube,
                arc_length_speeds=ct.HEATMAP_ARC_LENGTH_SPEEDS,
                price_ratios=ct.HEATMAP_PRICE_RATIOS,
                shift_exponents=ct.HEATMAP_SHIFT_EXPONENTS,
            )

        specs = metric_spec_map()
        facet_indices = nearest_indices(
            ct.HEATMAP_SHIFT_EXPONENTS,
            facet_shift_values,
        )
        launch_auto_speed = ct.compute_auto_calibrated_arc_length_speed(
            launch_cfg,
            shared_price_data,
        )

        for metric_key in metric_keys:
            spec = specs[metric_key]
            facet_path = tvl_output_path(
                output_dir,
                f"reclamm_arc_speed_shift_price_facets_{spec['slug']}",
                base_cfg,
            )
            plot_shift_slice_facets(
                data=cube[metric_key],
                arc_length_speeds=ct.HEATMAP_ARC_LENGTH_SPEEDS,
                price_ratios=ct.HEATMAP_PRICE_RATIOS,
                shift_exponents=ct.HEATMAP_SHIFT_EXPONENTS,
                shift_indices=facet_indices,
                spec=spec,
                cfg=base_cfg,
                filename=facet_path,
            )

            best_values, best_speeds = compute_argmax_over_arc_speed(
                cube[metric_key],
                ct.HEATMAP_ARC_LENGTH_SPEEDS,
            )
            best_value_path = tvl_output_path(
                output_dir,
                f"reclamm_arc_speed_shift_price_best_{spec['slug']}",
                base_cfg,
            )
            ct.plot_heatmap(
                data=best_values,
                x_values=ct.HEATMAP_PRICE_RATIOS,
                y_values=ct.HEATMAP_SHIFT_EXPONENTS,
                x_label="Price ratio",
                y_label="Shift exponent",
                title=(
                    f"Best {spec['title']} over arc-length speed | "
                    f"TVL {ct.format_tvl_millions_label(base_cfg)}"
                ),
                colorbar_label=f"Best {spec['colorbar_label']}",
                filename=best_value_path,
                xticks=ct.PRICE_RATIO_TICKS,
                yticks=ct.SHIFT_EXPONENT_TICKS,
                center_zero=spec["center_zero"],
                cmap=spec["cmap"],
                color_norm=spec.get("color_norm"),
                symlog_linthresh=spec.get("symlog_linthresh"),
            )
            best_speed_path = tvl_output_path(
                output_dir,
                f"reclamm_arc_speed_shift_price_argmax_speed_by_{spec['slug']}",
                base_cfg,
            )
            plot_best_speed_heatmap(
                best_speeds=best_speeds,
                arc_length_speeds=ct.HEATMAP_ARC_LENGTH_SPEEDS,
                price_ratios=ct.HEATMAP_PRICE_RATIOS,
                shift_exponents=ct.HEATMAP_SHIFT_EXPONENTS,
                metric_label=spec["title"],
                cfg=base_cfg,
                filename=best_speed_path,
            )
            save_best_speed_summary(
                output_dir,
                base_cfg,
                metric_key,
                best_values=best_values,
                best_speeds=best_speeds,
                price_ratios=ct.HEATMAP_PRICE_RATIOS,
                shift_exponents=ct.HEATMAP_SHIFT_EXPONENTS,
            )

            if plot_orthogonal_3d:
                orthogonal_path = tvl_output_path(
                    output_dir,
                    f"reclamm_arc_speed_shift_price_orthogonal_3d_{spec['slug']}",
                    base_cfg,
                )
                plot_orthogonal_3d_slices(
                    data=cube[metric_key],
                    arc_length_speeds=ct.HEATMAP_ARC_LENGTH_SPEEDS,
                    price_ratios=ct.HEATMAP_PRICE_RATIOS,
                    shift_exponents=ct.HEATMAP_SHIFT_EXPONENTS,
                    spec=spec,
                    cfg=base_cfg,
                    launch_auto_speed=launch_auto_speed,
                    filename=orthogonal_path,
                )

        if plot_all_slices:
            plot_all_shift_slices(
                output_dir=output_dir,
                cube=cube,
                arc_length_speeds=ct.HEATMAP_ARC_LENGTH_SPEEDS,
                price_ratios=ct.HEATMAP_PRICE_RATIOS,
                shift_exponents=ct.HEATMAP_SHIFT_EXPONENTS,
                specs=specs,
                cfg=base_cfg,
                metric_keys=metric_keys,
            )
    finally:
        ct.flush_sweep_cache(cache, force=True)
        cache.clear()
        gc.collect()
        print(f"Released arc-speed/shift/price cache for TVL {tvl_label}.")


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    global ct
    ct = load_compare_module()
    output_dir = ensure_output_dir(args.output_dir)
    if args.all_tvls:
        tvl_values = tuple(float(value) for value in ct.TVL_SWEEP_VALUES)
    elif args.tvl is not None:
        tvl_values = tuple(float(value) for value in args.tvl)
    else:
        tvl_values = (float(ct.DEFAULT_INITIAL_POOL_VALUE),)

    metric_keys = tuple(args.metric or DEFAULT_METRIC_KEYS)
    unsupported = [
        metric_key
        for metric_key in metric_keys
        if metric_key not in ct.HEATMAP_METRIC_DEPENDENCIES
    ]
    if unsupported:
        raise ValueError(f"Unsupported metric keys: {unsupported}")

    print(f"Writing artifacts to {output_dir}")
    print(f"Running metrics: {', '.join(metric_keys)}")
    print("Loading shared price data and market-linear noise arrays...")
    shared_price_data = ct.load_shared_price_data(ct.CONFIGS)
    shared_market_linear_noise_data = ct.load_shared_market_linear_noise_data()

    for initial_pool_value in tvl_values:
        run_for_tvl(
            initial_pool_value=initial_pool_value,
            output_dir=output_dir,
            metric_keys=metric_keys,
            facet_shift_values=tuple(args.facet_shift_values),
            plot_all_slices=bool(args.plot_all_shift_slices),
            plot_orthogonal_3d=not bool(args.no_orthogonal_3d),
            save_cube=not bool(args.skip_cube_parquet),
            shared_price_data=shared_price_data,
            shared_market_linear_noise_data=shared_market_linear_noise_data,
        )


if __name__ == "__main__":
    main()
