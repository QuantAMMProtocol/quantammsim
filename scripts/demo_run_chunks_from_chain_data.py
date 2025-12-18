#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Windowed runner: iterate from one coarse-weight knot to the next.

Minimal changes to core running functionality:
- Load scraped data exactly as in the single-run script.
- Build windows from adjacent coarse weight unix_values: [u_i, u_{i+1}].
- For each window, truncate coarse_weights to those TWO rows only.
- Fees/gas/LP/arb are NOT truncated.
- Create start/end dates from the window's unix pair and run.
- Reproduce the same plots that the single-run emits, but per window.
- Collect all per-window results and plot a unified value curve at the end.

Optional: --dump-diagnostics writes JSON/CSV checkpoints.
NEW: --debug-window-index gates diagnostics INSIDE the loop; global dumps unchanged.

NOTE: All plots are plain Matplotlib (no LaTeX).
"""
import debug
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax.numpy as jnp

from quantammsim.runners.jax_runners import (
    do_run_on_historic_data_with_provided_coarse_weights,
)

# =======================
# Styling (no LaTeX)
# =======================
mpl.rcParams.update({
    "text.usetex": False,
    "axes.grid": False,
    "figure.facecolor": "#162536",
    "axes.facecolor": "#162536",
    "axes.edgecolor": "#E6CE97",
    "xtick.color": "#E6CE97",
    "ytick.color": "#E6CE97",
    "text.color": "#E6CE97",
})
COLOR = "#E6CE97"

# =======================
# Helpers
# =======================

def resample_to_minute_grid(df, timestamp_col='datetime'):
    """
    Resample a randomly sampled dataframe to uniform 1-minute grid using forward fill.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df_indexed = df.set_index(timestamp_col)
    resampled = df_indexed.resample('1T').ffill()
    return resampled.reset_index()

def ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def plot_reserves(result, fingerprint, run_name, plot_dir="./"):
    """
    Simple line plot of normalized reserves per token over the window.
    No LaTeX; one line per token.
    """
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    reserves = result["reserves"] / result["reserves"][0]
    tokens = sorted(fingerprint["tokens"])

    plt.figure(figsize=(12, 6))
    for i, token in enumerate(tokens):
        plt.plot(reserves[:, i], linewidth=0.8, label=token)

    plt.title("Reserves Over Time")
    plt.xlabel("Minute")
    plt.ylabel("Reserves (normalized to start)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR); ax.spines["bottom"].set_color(COLOR)
    plt.legend(frameon=False, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"reserves_{run_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_weights(output_dict, run_fingerprint, plot_prefix="weights", plot_dir="./plots/"):
    """
    Simple line plot of portfolio weights (sampled daily like the single-run).
    No LaTeX; one line per token.
    """
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    total_value = np.sum(output_dict["reserves"] * output_dict["prices"], axis=1, keepdims=True)
    weights = np.array(output_dict["reserves"] * output_dict["prices"] / total_value)
    weights_daily = weights[::1440] if weights.shape[0] >= 1440 else weights

    tokens = sorted(run_fingerprint["tokens"])
    plt.figure(figsize=(12, 6))
    for i, token in enumerate(tokens):
        plt.plot(weights_daily[:, i], linewidth=0.8, label=token)

    plt.ylim(0, 1)
    plt.title("Weights Over Time")
    plt.xlabel("Sample (daily if long enough)")
    plt.ylabel("Weight")
    ax = plt.gca()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR); ax.spines["bottom"].set_color(COLOR)
    plt.legend(frameon=False, fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(Path(plot_dir) / (plot_prefix + "_weights_over_time.png"), dpi=300, bbox_inches="tight")
    plt.close()

# =======================
# Loader (mirrors single-run; minimal edits; ensures weight columns align with tokens)
# =======================

def load_scraped_pool_data(data_dir="./sonic_macro"):
    """
    Load and process scraped pool data from CSV files.
    Returns dict with: coarse_weights, fees_df, gas_cost_df, lp_supply_df, arb_fees_df,
    tokens, full_data, ordered_balances, ordered_prices.

    Coarse weights are sampled daily (1440) and columns are reindexed using
    token_sort_indices to match alphabetical tokens (as in the single-run script).
    """
    import re
    from pathlib import Path

    data_path = Path(data_dir)
    csv_files = list(data_path.glob("reserves_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    def extract_block_number(filename):
        m = re.search(r'reserves_.*_(\d+)_\d+\.csv', filename.name)
        return int(m.group(1)) if m else 0
    csv_files.sort(key=extract_block_number)

    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    # Dump combined_df to CSV for diagnostics
    try:
        out_path = Path("./results_windows")
        out_path.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out_path / "combined_df.csv", index=False)
    except Exception as e:
        print(f"Warning: failed to dump combined_df CSV: {e}")

    # Tokens + sort indices (compute indices from the unsorted names, then sort labels)
    token_columns = [col for col in combined_df.columns if col.endswith('_balance')]
    tokens_unsorted = [col.replace('_balance', '') for col in token_columns]
    token_sort_indices = np.argsort(tokens_unsorted)
    tokens = [tokens_unsorted[i] for i in token_sort_indices]  # alphabetical

    # Parse arrays
    def parse_string_array(series, fallback_len):
        parsed_arrays = []
        for val in series:
            try:
                if isinstance(val, str):
                    clean_val = val.strip().strip('[]')
                    if clean_val == "":
                        parsed_arrays.append([0.0] * fallback_len)
                    else:
                        parsed_arrays.append([float(x.strip()) for x in clean_val.split(',')])
                else:
                    parsed_arrays.append([float(val)])
            except Exception:
                parsed_arrays.append([0.0] * fallback_len)
        return np.array(parsed_arrays, dtype=np.float64)

    balances_array = parse_string_array(combined_df['balances'], fallback_len=len(tokens))
    weights_first_four = parse_string_array(combined_df['weights_first_four'], fallback_len=4)
    weights_second_four = parse_string_array(combined_df['weights_second_four'], fallback_len=4)
    token_rates = parse_string_array(combined_df['token_rates'], fallback_len=len(tokens))

    combined_df["weights_first_four"] = [arr.tolist() for arr in weights_first_four]
    combined_df["weights_second_four"] = [arr.tolist() for arr in weights_second_four]
    combined_df["balances"] = [arr.tolist() for arr in balances_array]
    combined_df["token_rates"] = [arr.tolist() for arr in token_rates]

    # First entry with supply
    first_entry = combined_df[combined_df['total_supply'] > 0].iloc[0]
    combined_df = combined_df.iloc[first_entry.name:].reset_index(drop=True)

    # Resample to 1-minute grid
    combined_df["datetime"] = pd.to_datetime(combined_df["timestamp"], unit="ms")
    combined_df = resample_to_minute_grid(combined_df)[1:]

    # Recompute ms timestamps
    combined_df["timestamp"] = (combined_df["datetime"].astype("int64") // 10**6).astype("int64")

    # Clip to midnight bounds
    midnight_mask = combined_df["datetime"].dt.time == pd.Timestamp("00:00:00").time()
    if midnight_mask.any():
        first_midnight = combined_df.loc[midnight_mask].iloc[0]["datetime"]
        last_midnight  = combined_df.loc[midnight_mask].iloc[-1]["datetime"]
    else:
        first_midnight = combined_df["datetime"].iloc[0]
        last_midnight  = combined_df["datetime"].iloc[-1]
    combined_df = combined_df[(combined_df["datetime"] >= first_midnight) &
                              (combined_df["datetime"] <= last_midnight)].reset_index(drop=True)
    combined_df = combined_df.drop('datetime', axis=1)

    # Ordered balances/prices in alphabetical token order
    ordered_balances = np.zeros((len(combined_df), len(tokens)))
    ordered_prices = np.zeros((len(combined_df), len(tokens)))
    for i, token in enumerate(tokens):
        if f"{token}_balance" in combined_df.columns:
            ordered_balances[:, i] = combined_df[f"{token}_balance"].values
        if f"{token}_price" in combined_df.columns:
            ordered_prices[:, i] = combined_df[f"{token}_price"].values

    try:
        # Determine output directory from CLI args or default
        out_dir = "./results_windows"
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ts = combined_df["timestamp"].to_numpy(dtype=np.int64)

        # Dump ordered balances
        df_bal = pd.DataFrame({"timestamp": ts})
        for i, token in enumerate(tokens):
            df_bal[token] = ordered_balances[:, i]
        df_bal.to_csv(out_path / "ordered_balances.csv", index=False)

        # Dump ordered prices
        df_px = pd.DataFrame({"timestamp": ts})
        for i, token in enumerate(tokens):
            df_px[token] = ordered_prices[:, i]
        df_px.to_csv(out_path / "ordered_prices.csv", index=False)
    except Exception as e:
        print(f"Warning: failed to dump ordered balances/prices diagnostics: {e}")
    # Build weights array and sample daily; apply column reindex like single-run
    weights_array = np.concatenate(
        [[np.array(w) for w in combined_df['weights_first_four']],
         [np.array(w) for w in combined_df['weights_second_four']]],
        axis=1
    )
    weights_array = weights_array[:, :4]  # keep first 4
    weights_array_sorted = weights_array[:, np.array(token_sort_indices[:weights_array.shape[1]], dtype=int)]

    chunk_period = 1440
    cw_weights = weights_array_sorted[::chunk_period]
    cw_unix = combined_df["timestamp"].iloc[::chunk_period].values.astype(np.int64)
    # normalize rows
    row_sums = np.sum(cw_weights, axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    cw_weights = cw_weights / row_sums

    coarse_weights = {
        "weights": jnp.array(cw_weights),
        "unix_values": jnp.array(cw_unix),
    }
    # Save coarse weights as a 2-column CSV: [unix, weights_json]
    try:
        out_path = Path("./results_windows")
        out_path.mkdir(parents=True, exist_ok=True)
        df_cw_2col = pd.DataFrame({
            "unix": cw_unix.astype(np.int64),
            "weights": [json.dumps([float(x) for x in row]) for row in cw_weights],
        })
        df_cw_2col.to_csv(out_path / "coarse_weights_2col.csv", index=False)
    except Exception as e:
        print(f"Warning: failed to dump 2-column coarse weights: {e}")

    fees_df = pd.DataFrame({'unix': combined_df['timestamp'].values,
                            'fees': combined_df['fee_data'].values})
    gas_cost_df = pd.DataFrame({'unix': combined_df['timestamp'].values,
                                'trade_gas_cost_usd': combined_df['gas_price_50p_gwei'].values})
    lp_supply_df = pd.DataFrame({'unix': combined_df['timestamp'].values,
                                 'lp_supply': combined_df['total_supply'].values})
    arb_fees_df = pd.DataFrame({'unix': combined_df['timestamp'].values,
                                'arb_fees': combined_df['fee_data'].values})

    return {
        'coarse_weights': coarse_weights,
        'fees_df': fees_df,
        'gas_cost_df': gas_cost_df,
        'lp_supply_df': lp_supply_df,
        'arb_fees_df': arb_fees_df,
        'tokens': tokens,
        'full_data': combined_df,
        'ordered_balances': ordered_balances,
        'ordered_prices': ordered_prices,
    }

# =======================
# Window builder + runner
# =======================

DEFAULT_FINGERPRINT = {
    "chunk_period": 1440,
    "weight_interpolation_period": 1440,
    "fees": 0.0,
    "gas_cost": 0.0,
    "minimum_weight": 0.03,
    "rule": "power_channel",
}

def run_windows(data_dir: str, out_dir: Path, dump_diagnostics: bool, debug_window_index: int | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load scraped data + coarse weights
    scraped = load_scraped_pool_data(data_dir)
    tokens = scraped["tokens"]
    cw = scraped["coarse_weights"]
    cw_unix = np.asarray(cw["unix_values"]).astype(np.int64)
    cw_w = np.asarray(cw["weights"], dtype=np.float64)

    # Optional: dump the full coarse-weight table (pre-window) — global, unaffected by debug index
    if dump_diagnostics:
        df_cw = pd.DataFrame({"timestamp": cw_unix})
        _cw = cw_w if cw_w.ndim == 2 else cw_w[:, None]
        for j in range(_cw.shape[1]):
            df_cw[f"weight_{j}"] = _cw[:, j]
        df_cw.to_csv(out_dir / "coarse_weights_global.csv", index=False)

    # 2) Build windows from adjacent unix_values: [u_i, u_{i+1}]
    windows = [(int(cw_unix[i]), int(cw_unix[i+1])) for i in range(len(cw_unix)-1)]

    # 3) Loop windows, truncate coarse weights to two rows, set fingerprint, run
    unified_values = []

    actual_reserves_all = scraped["ordered_balances"]  # shape [T_minute, n]

    for idx, (u0, u1) in enumerate(windows):
        start_iso = ms_to_iso(u0)
        end_iso = ms_to_iso(u1)

        # Two-row coarse weights for the window
        w2 = np.stack([cw_w[idx], cw_w[idx+1]], axis=0)
        u2 = np.array([u0, u1], dtype=np.int64)
        cw_window = {"weights": jnp.array(w2), "unix_values": jnp.array(u2)}

        # Fingerprint for the window
        fingerprint = {
            **DEFAULT_FINGERPRINT,
            "tokens": tokens,
            "startDateString": start_iso,
            "endDateString": end_iso,
            "initial_pool_value": 1_000_000.0,
        }

        # Params: initial_weights = the first knot’s weights
        # Pick initial reserves at window start time u0
        full_ts = scraped["full_data"]["timestamp"].values.astype(np.int64)
        match_idx = np.where(full_ts == u0)[0]
        start_idx = int(match_idx[0]) if match_idx.size else int(np.clip(np.searchsorted(full_ts, u0), 0, len(full_ts) - 1))
        print(f"[Window {idx}] start_idx in full data: {start_idx}, timestamp: {full_ts[start_idx]} (target {u0})")
        params = {
            "initial_weights": jnp.array(w2[0]),
            "initial_reserves": jnp.array(scraped["ordered_balances"][start_idx]),
            "log_k": jnp.array([6.0] * len(tokens)),
            "logit_delta_lamb": jnp.array([0.0] * len(tokens)),
            "logit_lamb": jnp.array([0.0] * len(tokens)),
            "raw_exponents": jnp.array([0.0] * len(tokens)),
        }

        # -------------------- Diagnostics per window (gated by args) --------------------
        do_dump = bool(dump_diagnostics and (debug_window_index is None or debug_window_index == idx))
        if do_dump:
            di = {
                "window_index": idx,
                "start_ms": u0,
                "end_ms": u1,
                "start_iso": start_iso,
                "end_iso": end_iso,
                "tokens": tokens,
                "coarse_weights_window": {
                    "unix_values": u2.tolist(),
                    "weights": w2.tolist(),
                },
                "params": {
                    "initial_weights": params["initial_weights"].tolist(),
                    "initial_reserves": params["initial_reserves"].tolist(),
                    "log_k": params["log_k"].tolist(),
                    "logit_delta_lamb": params["logit_delta_lamb"].tolist(),
                    "logit_lamb": params["logit_lamb"].tolist(),
                    "raw_exponents": params["raw_exponents"].tolist(),
                },
                "fingerprint": fingerprint,
            }
            with open(out_dir / f"window_{idx:04d}_runner_inputs.json", "w") as f:
                json.dump(di, f, indent=2)
            # CSV of coarse weights for this window
            df_cw_win = pd.DataFrame({"timestamp": u2})
            for j in range(w2.shape[1]):
                df_cw_win[f"weight_{j}"] = w2[:, j]
            df_cw_win.to_csv(out_dir / f"window_{idx:04d}_coarse_weights.csv", index=False)

            # Print all arguments that will be passed into do_run
            print(f"[Window {idx}] do_run inputs:")

            # run_fingerprint (small dict)
            try:
                print(json.dumps({"run_fingerprint": fingerprint}, indent=2, default=str))
            except Exception as e:
                print(f"Failed to serialize fingerprint: {e}")

            # coarse_weights (2 x n weights + 2 unix timestamps)
            try:
                cw_ser = {
                    "coarse_weights": {
                        "unix_values": np.asarray(cw_window["unix_values"]).astype(int).tolist(),
                        "weights": np.asarray(cw_window["weights"]).tolist(),
                    }
                }
                print(json.dumps(cw_ser, indent=2))
            except Exception as e:
                print(f"Failed to serialize coarse_weights: {e}")

            # params (vectors per token)
            try:
                params_ser = {
                    "params": {
                        k: np.asarray(v).tolist()
                        for k, v in params.items()
                    }
                }
                print(json.dumps(params_ser, indent=2))
            except Exception as e:
                print(f"Failed to serialize params: {e}")

            # Helper to preview DataFrames
            def _df_meta_and_head(df, name, n=3):
                try:
                    meta = {
                        "name": name,
                        "rows": int(len(df)),
                        "columns": list(map(str, df.columns)),
                        "head": df.head(n).to_dict(orient="records"),
                    }
                    print(json.dumps(meta, indent=2, default=str))
                except Exception as e:
                    print(f"Failed to serialize DataFrame {name}: {e}")

            _df_meta_and_head(scraped["fees_df"], "fees_df")
            _df_meta_and_head(scraped["gas_cost_df"], "gas_cost_df")
            _df_meta_and_head(scraped["lp_supply_df"], "lp_supply_df")
            _df_meta_and_head(scraped["arb_fees_df"], "arb_fees_df")
        # -------------------------------------------------------------------------------

        print(f"[Window {idx}] Starting run from {start_iso} to {end_iso}...")
        print(f"  Initial weights: {w2[0].tolist()}")
        print(f"  Initial reserves: {scraped['ordered_balances'][start_idx].tolist()}")
        print(f"  Initial Market Value: {(scraped['ordered_balances'][start_idx] * scraped['ordered_prices'][start_idx]).tolist()}")
        print(f"  Initial pool value: {fingerprint['initial_pool_value']}")
        print(f"  Coarse weights unix: {u2.tolist()}")
        print(f"  Coarse weights (first row): {w2[0].tolist()}")
        print(f"  Coarse weights (second row): {w2[1].tolist()}")
        print(f"  Params: log_k={params['log_k'].tolist()}, logit_delta_lamb={params['logit_delta_lamb'].tolist()}, logit_lamb={params['logit_lamb'].tolist()}, raw_exponents={params['raw_exponents'].tolist()}")
        print("  Running...")

        # Run the simulation for this window
        result = do_run_on_historic_data_with_provided_coarse_weights(
            run_fingerprint=fingerprint,
            coarse_weights=cw_window,
            params=params,
            fees_df=scraped["fees_df"],
            gas_cost_df=scraped["gas_cost_df"],
            lp_supply_df=scraped["lp_supply_df"],
            arb_fees_df=scraped["arb_fees_df"],
        )

        # ---------------- Correct, window-aligned plotting block (time-aware + plain y) ----------------
        from matplotlib import dates as mdates, ticker as mticker
        

        print(f"[Window {idx}] Run complete. Output has {len(result['value'])} time points.")
        print(f"  Final pool value: {result['value'][-1]}")
        run_name = f"window_{idx:04d}"

        # Runner series (per-minute over this window)
        sim_value = np.asarray(result["value"], dtype=np.float64)
        prices = np.asarray(result["prices"], dtype=np.float64)
        nT = int(len(sim_value))

        # Window-aligned actual reserves from the full minute grid: [u0, u1)
        ts_all = scraped["full_data"]["timestamp"].to_numpy(dtype=np.int64)
        mask = (ts_all >= u0) & (ts_all < u1)
        actual_reserves_win = scraped["ordered_balances"][mask]

        # Length guard: trim both sides to the common length (should normally match)
        M = min(nT, actual_reserves_win.shape[0], prices.shape[0])
        if (nT != actual_reserves_win.shape[0]) or (nT != prices.shape[0]):
            print(f"[Window {idx}] length mismatch: sim={nT}, reserves={actual_reserves_win.shape[0]}, prices={prices.shape[0]} -> trimming to {M}")
        sim_value = sim_value[:M]
        prices = prices[:M]
        actual_reserves_win = actual_reserves_win[:M]

        # Build a wall-clock x-axis at 1-min cadence starting from u0 (UTC)
        time_index = pd.date_range(start=pd.to_datetime(u0, unit="ms", utc=True), periods=M, freq="T")

        # Actual value track for the window (reserves * runner prices)
        actual_value = (actual_reserves_win * prices).sum(axis=1)

        # Common axis formatters
        def _format_axes_time_y_plain(ax, title, y_label):
            ax.set_title(title)
            ax.set_xlabel("Time (HH:MM, UTC)")
            ax.set_ylabel(y_label)
            # X axis as HH:MM
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # Y axis: plain numbers (no scientific notation)
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(COLOR)
            ax.spines["bottom"].set_color(COLOR)

        # Value over time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_index, sim_value, linewidth=1.0)
        _format_axes_time_y_plain(ax, f"Pool Value Over Time - {run_name}", "Value (USD)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{run_name}_value.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Difference (Sim - Actual)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_index, sim_value - actual_value, linewidth=1.0)
        _format_axes_time_y_plain(ax, f"Pool Value Difference - {run_name}", "Sim - Actual (USD)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{run_name}_value_difference.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Ratio (Sim / Actual)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(sim_value, actual_value,
                              out=np.full_like(sim_value, np.nan, dtype=float),
                              where=actual_value != 0)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_index, ratio, linewidth=1.0)
        _format_axes_time_y_plain(ax, f"Pool Value Ratio - {run_name}", "Sim/Actual")
        fig.tight_layout()
        fig.savefig(out_dir / f"{run_name}_value_ratio.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Overlay (Sim vs Actual)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time_index, sim_value, label="Simulated", linewidth=1.0)
        ax.plot(time_index, actual_value, label="Actual", linewidth=1.0)
        _format_axes_time_y_plain(ax, f"Pool Value Overlay - {run_name}", "Value (USD)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"{run_name}_value_actual.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Reserves + Weights plots (unchanged)
        plot_reserves(result, fingerprint, run_name=run_name, plot_dir=str(out_dir))
        plot_weights(result, fingerprint, plot_prefix=run_name, plot_dir=str(out_dir))
        # ----------------------------------------------------------------------------------------------

        # Collect unified values (concatenate)
        unified_values.append(np.asarray(result["value"], dtype=np.float64))

    # 4) Unified plot over all windows (concatenated)
    if unified_values:
        uv = np.concatenate(unified_values, axis=0)
        plt.figure(figsize=(12, 6))
        plt.plot(uv, linewidth=1.0)
        plt.title("Unified Pool Value Over Time (All Windows)")
        plt.xlabel("Contiguous minutes across windows")
        plt.ylabel("Value (USD)")
        plt.tight_layout()
        plt.savefig(out_dir / "unified_value.png", dpi=300, bbox_inches="tight")
        plt.close()

# =======================
# Entrypoints (callable main + CLI main)
# =======================

def main(data_dir: str = "./sonic_macro",
         out_dir: str = "./results_windows",
         dump_diagnostics: bool = False,
         debug_window_index: int | None = None):
    """
    Programmatic entrypoint. Call this from Python.
    """
    run_windows(
        data_dir=data_dir,
        out_dir=Path(out_dir),
        dump_diagnostics=dump_diagnostics,
        debug_window_index=debug_window_index
    )

def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./sonic_macro")
    p.add_argument("--out-dir", type=str, default="./results_windows")
    p.add_argument("--dump-diagnostics", action="store_true")
    p.add_argument("--debug-window-index", type=int, default=None,
                   help="When set, emit diagnostics only for this window index. "
                        "If omitted, diagnostics (when enabled) are emitted for every window.")
    args = p.parse_args()
    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        dump_diagnostics=args.dump_diagnostics,
        debug_window_index=args.debug_window_index
    )

if __name__ == "__main__":
    cli_main()
