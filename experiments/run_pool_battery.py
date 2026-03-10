"""Sim-vs-world gas + arb-frequency calibration for on-chain reClAMM pools.

For each pool in the registry, runs quantammsim forward passes with exact
on-chain parameters across a 2D grid of (gas_cost, arb_frequency), then
compares the simulated pool value trajectory against the actual on-chain
trajectory.

arb_frequency is the period between arb trades in minutes (1 = every minute,
the most aggressive; higher = sparser arb).

Generalizes scripts/sim_vs_world_comparison.py to work for any pool.

Usage:
    cd /Users/matthew/Projects/quantammsim-reclamm
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim-reclamm

    # Single pool (default gas + arb_freq grid)
    python experiments/run_pool_battery.py cbBTC_WETH

    # All pools with data available
    python experiments/run_pool_battery.py --all

    # Custom grids
    python experiments/run_pool_battery.py cbBTC_WETH --gas-costs 0.0 0.5 1.0 --arb-freqs 1 5 15 60

    # Dry run (show config without running)
    python experiments/run_pool_battery.py cbBTC_WETH --dry-run

    # List available pools
    python experiments/run_pool_battery.py --list
"""

import argparse
import json
import os
import time

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timezone

from experiments.pool_registry import (
    POOL_REGISTRY,
    PoolConfig,
    extract_initial_state,
    extract_on_chain_state,
    get_data_end_date,
    get_gas_costs,
    load_bpt_supply_df,
    load_gas_csv,
    load_world_history,
    print_pool_summary,
)
from quantammsim.runners.jax_runners import do_run_on_historic_data

PROTOCOL_FEE_SPLIT = 0.5
DEFAULT_ARB_FREQS = [1, 2, 3, 5, 10, 15, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_at_timestamps(minute_vals, start_unix_sec, timestamps_sec):
    """Sample a minute-level array at specific Unix timestamps.

    For each target timestamp, finds the nearest minute index in the
    sim output and returns the corresponding value.
    """
    indices = np.round((timestamps_sec - start_unix_sec) / 60).astype(int)
    indices = np.clip(indices, 0, len(minute_vals) - 1)
    return minute_vals[indices]


def compute_log_rmse(sim_growth, world_growth):
    """RMSE of log(sim/world) across all trajectory points.

    Symmetric in over/under-estimation, natural for multiplicative processes.
    A score of 0.02 means typical 2% deviation at any point in time.
    """
    log_ratio = np.log(sim_growth / world_growth)
    return np.sqrt(np.mean(log_ratio ** 2))


def _start_str_from_pool(pool):
    """Derive sim start time from pool's plausible_start, rounded to minute."""
    ts = int(
        datetime.strptime(pool.plausible_start, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    ts_minute = (ts // 60) * 60
    return datetime.utcfromtimestamp(ts_minute).strftime("%Y-%m-%d %H:%M:%S")


def _onchain_params_to_sim(pool):
    """Map DB param names to quantammsim param dict (jnp arrays)."""
    p = pool.on_chain_params
    return {
        "price_ratio": jnp.array(p["price_ratio"]),
        "centeredness_margin": jnp.array(p["margin"]),
        "shift_exponent": jnp.array(p["shift_rate"]),
    }


# ---------------------------------------------------------------------------
# Core sim runner
# ---------------------------------------------------------------------------

def run_sim(pool, gas_cost, arb_frequency, initial_state, start, end,
            protocol_fee_split=PROTOCOL_FEE_SPLIT, lp_supply_df=None,
            noise_config=None):
    """Run a single forward pass with exact on-chain params.

    gas_cost can be:
      - float: flat gas cost in USD (e.g. 0.0, 0.5)
      - str: gas percentile label (e.g. "50p", "90p") — loads time-varying
        gas from CSV

    noise_config can be:
      - None: no noise model (arb-only, default)
      - dict with keys 'noise_model' and 'reclamm_noise_params': inject
        Tsoukalas noise model into the sim

    Returns dict with minute-level per-LP value (USD), prices (USD per token),
    and start_unix_sec.  When lp_supply_df is provided, value_usd is divided
    by the interpolated LP supply so it is comparable to BPT-normalized world
    balances.
    """
    params = _onchain_params_to_sim(pool)

    # Resolve gas: percentile string → DataFrame, float → scalar
    gas_cost_df = None
    if isinstance(gas_cost, str):
        gas_cost_df = load_gas_csv(gas_cost)
        flat_gas = 0.0  # placeholder; gas_cost_df overrides
    else:
        flat_gas = gas_cost

    fp = {
        "tokens": pool.tokens,
        "rule": "reclamm",
        "startDateString": start,
        "endDateString": end,
        "initial_pool_value": pool.initial_pool_value_usd,
        "fees": pool.swap_fee,
        "gas_cost": flat_gas,
        "arb_fees": 0.0,
        "do_arb": True,
        "arb_frequency": arb_frequency,
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "reclamm_use_shift_exponent": True,
        "reclamm_interpolation_method": "geometric",
        "reclamm_centeredness_scaling": False,
        "protocol_fee_split": protocol_fee_split,
        "reclamm_initial_state": initial_state,
    }

    if noise_config is not None:
        fp["noise_model"] = noise_config["noise_model"]
        fp["reclamm_noise_params"] = noise_config["reclamm_noise_params"]

    result = do_run_on_historic_data(
        run_fingerprint=fp, params=params, lp_supply_df=lp_supply_df,
        gas_cost_df=gas_cost_df,
    )

    start_unix_sec = datetime.strptime(
        start, "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc).timestamp()

    value_usd = np.array(result["value"])

    # Normalize to per-LP value so comparison with BPT-normalized world is valid.
    #
    # Subtlety: the scan applies lp_supply every arb_frequency minutes.
    # Between scan steps, reserves are constant (no arb), so the pool value
    # reflects the lp_supply from the LAST scan step.  If a BPT event occurs
    # between scan steps (e.g. at minute 3 when arb_frequency=5), the value
    # at minutes 3-4 still reflects the old lp_supply.  We must divide by
    # the scan-step-aligned lp_supply, not the current minute's lp_supply,
    # otherwise we get transient spikes.
    if lp_supply_df is not None:
        n_minutes = len(value_usd)
        # Map each minute to its most recent scan-step time
        scan_step_minutes = (
            np.arange(n_minutes) // arb_frequency * arb_frequency
        )
        scan_step_times_ms = (
            start_unix_sec * 1000 + scan_step_minutes * 60_000
        )
        lp_unix = np.array(lp_supply_df["unix"])
        lp_vals = np.array(lp_supply_df["lp_supply"])
        indices = np.searchsorted(lp_unix, scan_step_times_ms, side="right") - 1
        indices = np.clip(indices, 0, len(lp_vals) - 1)
        value_usd = value_usd / lp_vals[indices]

    return {
        "value_usd": value_usd,
        "prices": np.array(result["prices"]),  # (T, n_tokens) in USD
        "start_unix_sec": start_unix_sec,
    }


# ---------------------------------------------------------------------------
# Pool calibration (2D grid: gas_cost × arb_frequency)
# ---------------------------------------------------------------------------

def run_pool_calibration(pool, gas_costs, arb_freqs, verbose=True,
                         noise_config=None):
    """Run 2D gas × arb_frequency calibration for a single pool.

    Parameters
    ----------
    noise_config : dict, optional
        If provided, passed through to run_sim to inject noise model.
        Keys: 'noise_model', 'reclamm_noise_params'.

    Returns dict with:
      world_growth: array of world growth factors
      sim_growths: {(gas_cost, arb_freq): growth array}
      timestamps: world timestamps (seconds)
      governance_idx: index of first governance event (or n_points)
      n_points: number of comparison points
      days: array of days from start
      gas_costs: list of gas costs
      arb_freqs: list of arb frequencies
    """
    # Extract on-chain state + initial reserves
    extract_on_chain_state(pool)
    initial_state = extract_initial_state(pool)

    if verbose:
        print_pool_summary(pool)
        print(f"  Initial state: Ra={initial_state['Ra']:.4f}, "
              f"Rb={initial_state['Rb']:.4f}, "
              f"Va={initial_state['Va']:.4f}, Vb={initial_state['Vb']:.4f}")

    start_str = _start_str_from_pool(pool)
    end_str = get_data_end_date(pool.tokens)

    # Load BPT supply history for LP supply scaling
    lp_supply_df = load_bpt_supply_df(pool, end_date=end_str)

    if verbose:
        bpt_start = lp_supply_df["lp_supply"].iloc[0]
        bpt_end = lp_supply_df["lp_supply"].iloc[-1]
        print(f"  BPT supply: {bpt_start:.4f} → {bpt_end:.4f} "
              f"({(bpt_end/bpt_start - 1)*100:+.1f}%)")
        print(f"  Sim period: {start_str} to {end_str}")
        n_runs = len(gas_costs) * len(arb_freqs)
        print(f"  Grid: {len(gas_costs)} gas × {len(arb_freqs)} arb_freq = {n_runs} runs")

    # Load world history (BPT-normalized balances + governance events)
    # BPT-normalized is correct for the growth ratio metric since LP supply
    # cancels out of sim_growth/world_growth. Raw balances are available
    # in world["raw_bal_0/1"] for absolute trajectory comparison.
    world = load_world_history(pool, end_date=end_str)
    world_ts = world["timestamps"]
    world_bal_0 = world["bal_0"]
    world_bal_1 = world["bal_1"]
    gov_events = world["governance_events"]

    if verbose:
        print(f"  World points: {len(world_ts)}")
        if gov_events:
            for ts, field, old, new in gov_events:
                dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                print(f"  Governance: {field} {old:.6f} -> {new:.6f} on {dt}")
        else:
            print("  No governance events")

    # Governance cutoff index
    if gov_events:
        gov_idx = np.searchsorted(world_ts, gov_events[0][0])
    else:
        gov_idx = len(world_ts)

    # Run sims across the 2D grid
    sim_results = {}
    prices_min = None
    start_sec = None

    for gc in gas_costs:
        for af in arb_freqs:
            if verbose:
                print(f"\n  Running gas=${gc}, arb_freq={af}min...")
            t0 = time.time()
            result = run_sim(pool, gc, af, initial_state, start_str, end_str,
                            lp_supply_df=lp_supply_df,
                            noise_config=noise_config)
            elapsed = time.time() - t0
            if verbose:
                print(f"    Done in {elapsed:.1f}s")
            sim_results[(gc, af)] = result
            if prices_min is None:
                prices_min = result["prices"]
                start_sec = result["start_unix_sec"]

    # Truncate at governance
    n = min(gov_idx, len(world_ts))
    world_ts_trunc = world_ts[:n]

    # Sample USD prices at world timestamps for world valuation
    prices_at_world = np.stack([
        sample_at_timestamps(prices_min[:, i], start_sec, world_ts_trunc)
        for i in range(prices_min.shape[1])
    ], axis=1)

    # World value in USD = sum(bal_i * price_usd_i)
    world_value = (
        world_bal_0[:n] * prices_at_world[:, 0]
        + world_bal_1[:n] * prices_at_world[:, 1]
    )
    world_growth = world_value / world_value[0]

    # Sim growths at world timestamps
    sim_growths = {}
    for key, result in sim_results.items():
        sim_val = sample_at_timestamps(
            result["value_usd"], start_sec, world_ts_trunc,
        )
        sim_growths[key] = sim_val / sim_val[0]

    days = (world_ts_trunc - world_ts_trunc[0]) / 86400

    return {
        "world_growth": world_growth,
        "sim_growths": sim_growths,
        "timestamps": world_ts_trunc,
        "governance_idx": gov_idx,
        "n_points": n,
        "days": days,
        "gas_costs": list(gas_costs),
        "arb_freqs": list(arb_freqs),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pool_calibration(pool, calibration, output_dir="results", suffix=""):
    """Plot 2D gas × arb_freq calibration as heatmap + time series.

    Left: heatmap of final % deviation (gas_cost × arb_freq).
    Right: time series for each arb_freq at best gas cost.
    """
    os.makedirs(output_dir, exist_ok=True)

    world_growth = calibration["world_growth"]
    sim_growths = calibration["sim_growths"]
    days = calibration["days"]
    gas_costs = calibration["gas_costs"]
    arb_freqs = calibration["arb_freqs"]

    # Build RMSE matrix (log ratio, %)
    rmse_matrix = np.zeros((len(arb_freqs), len(gas_costs)))
    for i, af in enumerate(arb_freqs):
        for j, gc in enumerate(gas_costs):
            rmse_matrix[i, j] = compute_log_rmse(
                sim_growths[(gc, af)], world_growth
            ) * 100

    fig, (ax_heat, ax_ts) = plt.subplots(
        1, 2, figsize=(18, 7),
        gridspec_kw={"width_ratios": [1, 1.5]},
    )

    # Left: heatmap of trajectory RMSE
    im = ax_heat.imshow(
        rmse_matrix, aspect="auto", cmap="RdYlGn_r",
        vmin=0, vmax=rmse_matrix.max(), origin="lower",
    )
    ax_heat.set_xticks(range(len(gas_costs)))
    gas_labels = [
        f"gas {gc}" if isinstance(gc, str) else f"${gc}"
        for gc in gas_costs
    ]
    ax_heat.set_xticklabels(gas_labels, fontsize=9)
    ax_heat.set_yticks(range(len(arb_freqs)))
    ax_heat.set_yticklabels([f"{af}min" for af in arb_freqs], fontsize=9)
    ax_heat.set_xlabel("gas cost (USD)")
    ax_heat.set_ylabel("arb frequency (minutes)")
    ax_heat.set_title("Trajectory RMSE (log ratio, %)")

    # Annotate cells
    for i in range(len(arb_freqs)):
        for j in range(len(gas_costs)):
            val = rmse_matrix[i, j]
            color = "white" if val > rmse_matrix.max() * 0.6 else "black"
            ax_heat.text(j, i, f"{val:.2f}%", ha="center", va="center",
                         fontsize=8, color=color)

    # Mark cell with least-negative mean bias (closest to 0 from below).
    # If no cell is below world on average, fall back to lowest RMSE.
    bias_matrix = np.zeros_like(rmse_matrix)
    for i, af in enumerate(arb_freqs):
        for j, gc in enumerate(gas_costs):
            bias_matrix[i, j] = float(np.mean(
                np.log(sim_growths[(gc, af)] / world_growth)
            ))
    negative_mask = bias_matrix < 0
    if negative_mask.any():
        # Among negative cells, find the one closest to 0 (max value)
        masked = np.where(negative_mask, bias_matrix, -np.inf)
        best_idx = np.unravel_index(np.argmax(masked), masked.shape)
    else:
        best_idx = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
    ax_heat.add_patch(plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        fill=False, edgecolor="lime", linewidth=3,
    ))

    fig.colorbar(im, ax=ax_heat, label="RMSE (%)", shrink=0.8)

    # Right: 4 closest from below + 1 first above world
    # Rationale: sim should underestimate (can't capture organic swaps, MEV
    # rebates, etc.), so being below world is expected.  The one-above config
    # brackets where the sim crosses from conservative to optimistic.
    ax_ts.axhline(y=0.0, color="brown", linewidth=2, label="world (on-chain)")

    # Classify configs by mean log ratio (trajectory-average bias).
    # Using the mean rather than endpoint avoids a curve that's above
    # world for 80% of the trajectory being classified as "below"
    # just because it dips at the end.
    below = []  # (mean_bias, rmse, gc, af) where sim < world on average
    above = []  # (mean_bias, rmse, gc, af) where sim >= world on average
    for (gc, af), sg in sim_growths.items():
        mean_bias = float(np.mean(np.log(sg / world_growth)))
        rmse = compute_log_rmse(sg, world_growth)
        if mean_bias < 0:
            below.append((mean_bias, rmse, gc, af))
        else:
            above.append((mean_bias, rmse, gc, af))

    # Sort: below by mean_bias descending (closest to 0 first)
    below.sort(key=lambda x: x[0], reverse=True)
    # Sort: above by mean_bias ascending (closest to 0 first)
    above.sort(key=lambda x: x[0])

    # Select: up to 4 from below, 1 from above, fill if needed
    selected = []
    n_below = min(4, len(below))
    n_above = min(1, len(above))
    selected.extend(below[:n_below])
    selected.extend(above[:n_above])
    remaining = 5 - len(selected)
    if remaining > 0 and len(below) > n_below:
        selected.extend(below[n_below:n_below + remaining])
    remaining = 5 - len(selected)
    if remaining > 0 and len(above) > n_above:
        selected.extend(above[n_above:n_above + remaining])

    colors_below = plt.cm.Blues(np.linspace(0.4, 0.8, n_below))
    colors_above = np.array([[0.8, 0.2, 0.2, 1.0]])  # red for above
    plot_colors = list(colors_below) + list(colors_above[:n_above])
    # Fill remaining with grey
    while len(plot_colors) < len(selected):
        plot_colors.append([0.5, 0.5, 0.5, 1.0])

    for rank, (mean_bias, rmse, gc, af) in enumerate(selected):
        dev = (sim_growths[(gc, af)] / world_growth - 1) * 100
        gc_label = f"gas {gc}" if isinstance(gc, str) else f"gas=${gc}"
        marker = "\u25b2" if mean_bias >= 0 else "\u25bc"  # ▲ above, ▼ below
        ax_ts.plot(days, dev, color=plot_colors[rank], linewidth=2,
                   label=f"{marker} {gc_label}, arb={af}min  "
                         f"bias={mean_bias*100:+.2f}%  RMSE={rmse*100:.2f}%")

    ax_ts.set_xlabel("days")
    ax_ts.set_ylabel("% deviation from world")
    trunc = " (pre-governance)" if calibration["governance_idx"] < calibration["n_points"] + 1 else ""
    ax_ts.set_title(f"Best bracket: {n_below} below + {n_above} above world{trunc}")
    ax_ts.legend(fontsize=7, loc="best")
    ax_ts.grid(True, alpha=0.2)

    p = pool.on_chain_params
    fig.suptitle(
        f"{pool.label} ({pool.chain}) — {pool.tokens[0]}/{pool.tokens[1]}\n"
        f"PR={p['price_ratio']:.4f}  margin={p['margin']}  "
        f"shift={p['shift_rate']}  fee={pool.swap_fee}  "
        f"TVL=${pool.initial_pool_value_usd:,.0f}  "
        f"protocol_fee={PROTOCOL_FEE_SPLIT}",
        fontsize=10,
    )
    plt.tight_layout()

    out = os.path.join(output_dir, f"gas_calibration_{pool.label}{suffix}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return out


def plot_cross_pool_summary(all_results, output_dir="results"):
    """Plot cross-pool comparison: best (gas, arb_freq) and residual deviation."""
    os.makedirs(output_dir, exist_ok=True)

    pool_labels = []
    best_configs = []
    best_devs = []

    best_rmses = []
    best_biases = []
    for pool, cal in all_results:
        wg = cal["world_growth"]
        # Best = least-negative mean bias (closest from below)
        below_keys = [
            k for k in cal["sim_growths"]
            if np.mean(np.log(cal["sim_growths"][k] / wg)) < 0
        ]
        if below_keys:
            best_key = max(
                below_keys,
                key=lambda k: np.mean(np.log(cal["sim_growths"][k] / wg)),
            )
        else:
            best_key = min(
                cal["sim_growths"].keys(),
                key=lambda k: compute_log_rmse(cal["sim_growths"][k], wg),
            )
        best_rmse = compute_log_rmse(cal["sim_growths"][best_key], wg) * 100
        best_bias = float(np.mean(np.log(cal["sim_growths"][best_key] / wg))) * 100
        pool_labels.append(f"{pool.label}\n({pool.chain})")
        best_configs.append(best_key)
        best_rmses.append(best_rmse)
        best_biases.append(best_bias)

    fig, (ax_cfg, ax_dev) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(pool_labels))

    # Left: RMSE at best config
    config_strs = [
        f"gas {gc}\narb={af}min" if isinstance(gc, str)
        else f"gas=${gc}\narb={af}min"
        for gc, af in best_configs
    ]
    ax_cfg.barh(x, best_rmses, color="steelblue")
    ax_cfg.set_yticks(x)
    ax_cfg.set_yticklabels(pool_labels, fontsize=9)
    ax_cfg.set_xlabel("Trajectory RMSE (%)")
    ax_cfg.set_title("RMSE at best config")
    for i, (cs, rmse) in enumerate(zip(config_strs, best_rmses)):
        ax_cfg.text(rmse + 0.05, i, f"{cs}  (RMSE={rmse:.2f}%)", va="center", fontsize=8)
    ax_cfg.grid(True, alpha=0.2, axis="x")

    # Right: mean bias at best config (negative = conservative)
    colors = ["green" if d < 0 else "orange" if d < 1 else "red"
              for d in best_biases]
    ax_dev.bar(x, best_biases, color=colors)
    ax_dev.axhline(y=0, color="brown", linewidth=1)
    ax_dev.set_xticks(x)
    ax_dev.set_xticklabels(pool_labels, fontsize=8)
    ax_dev.set_ylabel("Mean bias (%)")
    ax_dev.set_title("Mean trajectory bias at best config")
    ax_dev.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Cross-pool gas + arb frequency calibration", fontsize=12)
    plt.tight_layout()

    out = os.path.join(output_dir, "gas_calibration_cross_pool.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved cross-pool summary: {out}")
    return out


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_calibration_summary(pool, calibration):
    """Print a summary table of the 2D grid results (trajectory RMSE)."""
    world_growth = calibration["world_growth"]
    sim_growths = calibration["sim_growths"]
    n_days = calibration["days"][-1]
    gas_costs = calibration["gas_costs"]
    arb_freqs = calibration["arb_freqs"]

    print(f"\n  {pool.label} ({pool.chain}) — {n_days:.0f} days")
    print(f"  World growth: {world_growth[-1]:.4f}")

    # Print as table: rows=arb_freq, cols=gas_cost (values = trajectory RMSE %)
    col_label = "arb\\gas"
    gas_labels = [
        f"gas {gc}" if isinstance(gc, str) else f"${gc}"
        for gc in gas_costs
    ]
    header = f"  {col_label:<10}" + "".join(f"{gl:<10}" for gl in gas_labels)
    print(header)
    print(f"  {'-'*len(header)}")
    for af in arb_freqs:
        row = f"  {af:>4}min   "
        for gc in gas_costs:
            rmse = compute_log_rmse(
                sim_growths[(gc, af)], world_growth
            ) * 100
            row += f"{rmse:>8.2f}% "
        print(row)


# ---------------------------------------------------------------------------
# Data availability check
# ---------------------------------------------------------------------------

def check_data_available(pool, data_root=None):
    """Check that all required parquet files exist for a pool."""
    if data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..", "quantammsim", "data"
        )
    for ticker in pool.tokens:
        if ticker == "USDC":
            continue
        path = os.path.join(data_root, f"{ticker}_USD.parquet")
        if not os.path.exists(path):
            return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sim-vs-world gas + arb-frequency calibration for on-chain reClAMM pools"
    )
    parser.add_argument("pool", nargs="?",
                        help="Pool label (e.g. cbBTC_WETH)")
    parser.add_argument("--all", action="store_true",
                        help="Run all pools with available data")
    parser.add_argument("--list", action="store_true",
                        help="List available pools and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pool config without running")
    parser.add_argument("--gas-costs", nargs="+", type=float, default=None,
                        help="Override gas cost battery (flat USD values)")
    parser.add_argument("--arb-freqs", nargs="+", type=int,
                        default=DEFAULT_ARB_FREQS,
                        help="Arb frequency values in minutes (default: 1 2 3 5 10 15 20)")
    parser.add_argument("--protocol-fee", type=float, default=PROTOCOL_FEE_SPLIT,
                        help="Protocol fee split (default 0.5)")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for output plots and JSON")
    parser.add_argument("--calibrate-noise", action="store_true",
                        help="Calibrate Tsoukalas noise model from Balancer API + DB "
                             "and inject into sim fingerprints")
    parser.add_argument("--noise-model", choices=["sqrt", "log", "loglinear"],
                        default="sqrt",
                        help="Noise model variant (default: sqrt)")
    parser.add_argument("--noise-params-json", default=None,
                        help="Path to hierarchical noise params JSON "
                             "(from calibrate_noise_hierarchical.py). "
                             "Looks up pool by address or uses --predict.")
    args = parser.parse_args()

    # --list mode
    if args.list:
        print("\nAvailable pools:\n")
        for label, pool in POOL_REGISTRY.items():
            has_data = check_data_available(pool)
            status = "READY" if has_data else "MISSING DATA"
            try:
                extract_on_chain_state(pool)
                print_pool_summary(pool)
            except Exception as e:
                print(f"  {label}: {e}")
            print(f"  Data: {status}")
        return

    # Determine which pools to run
    if args.all:
        pool_labels = [
            label for label, pool in POOL_REGISTRY.items()
            if check_data_available(pool)
        ]
        if not pool_labels:
            print("No pools have all required data files.")
            return
    elif args.pool:
        if args.pool not in POOL_REGISTRY:
            print(f"Unknown pool: {args.pool}")
            print(f"Available: {list(POOL_REGISTRY.keys())}")
            return
        if not check_data_available(POOL_REGISTRY[args.pool]):
            missing = [
                f"{t}_USD.parquet" for t in POOL_REGISTRY[args.pool].tokens
                if t != "USDC" and not os.path.exists(
                    os.path.join(
                        os.path.dirname(__file__), "..", "quantammsim",
                        "data", f"{t}_USD.parquet"
                    )
                )
            ]
            print(f"Missing data for {args.pool}: {missing}")
            return
        pool_labels = [args.pool]
    else:
        parser.print_help()
        return

    # Collect runs
    runs = []
    for label in pool_labels:
        pool = POOL_REGISTRY[label]
        gas_costs = get_gas_costs(pool, args.gas_costs)
        runs.append((pool, gas_costs))

    arb_freqs = args.arb_freqs
    n_total = sum(len(gcs) * len(arb_freqs) for _, gcs in runs)

    print(f"\n{'='*60}")
    print(f"GAS + ARB CALIBRATION: {len(runs)} pool(s), {n_total} total runs")
    for pool, gcs in runs:
        print(f"  {pool.label:15s} ({pool.chain:10s})  "
              f"gas={gcs}  arb_freq={arb_freqs}")
    print(f"  Protocol fee split: {args.protocol_fee}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n--- DRY RUN ---\n")
        for pool, gas_costs in runs:
            extract_on_chain_state(pool)
            initial_state = extract_initial_state(pool)
            print_pool_summary(pool)
            print(f"  Initial state: {initial_state}")
            start = _start_str_from_pool(pool)
            end = get_data_end_date(pool.tokens)
            print(f"  Sim period: {start} to {end}")
            world = load_world_history(pool, end_date=end)
            n_gov = len(world["governance_events"])
            print(f"  World points: {len(world['timestamps'])}, "
                  f"governance events: {n_gov}")
            if world["governance_events"]:
                for ts, field, old, new in world["governance_events"]:
                    dt = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                    print(f"    {field} {old:.6f} -> {new:.6f} on {dt}")
            print(f"  Gas battery: {gas_costs}")
            print(f"  Arb freqs: {arb_freqs}")
            n_runs = len(gas_costs) * len(arb_freqs)
            print(f"  Total runs: {n_runs}")
        return

    # Execute calibration for each pool
    all_results = []
    for pool, gas_costs in runs:
        print(f"\n{'#'*60}")
        print(f"POOL: {pool.label}")
        print(f"{'#'*60}")

        # Noise calibration (if requested)
        noise_config = None
        if args.noise_params_json and args.noise_model == "loglinear":
            # Load hierarchical noise params from JSON
            with open(args.noise_params_json) as f:
                hier_data = json.load(f)
            # Look up pool by address
            addr = pool.pool_address.lower()
            pool_params = None
            for p in hier_data["pools"]:
                pid = p["pool_id"].lower().replace("0x", "")
                if pid.startswith(addr) or addr.startswith(pid):
                    pool_params = p["noise_params"]
                    break
            if pool_params is None:
                # Fall back to population-level prediction
                from scripts.calibrate_noise_hierarchical import predict_new_pool
                chain_map = {"ethereum": "MAINNET", "base": "BASE",
                             "gnosis": "GNOSIS", "arbitrum": "ARBITRUM",
                             "polygon": "POLYGON", "optimism": "OPTIMISM",
                             "sonic": "SONIC", "avalanche": "AVALANCHE"}
                api_chain = chain_map.get(pool.chain, pool.chain.upper())
                # Reconstruct posteriors + encoding from the JSON
                posteriors_from_json = {
                    "Phi_mean": np.array(hier_data["Phi"]),
                }
                encoding_from_json = {
                    "covariate_names": hier_data["covariate_names"],
                }
                pool_params = predict_new_pool(
                    posteriors_from_json, encoding_from_json,
                    api_chain, pool.tokens, pool.swap_fee,
                )
                print(f"\n  Using population-level loglinear params (pool not in JSON)")
            else:
                print(f"\n  Using hierarchical loglinear params for {pool.label}")
            print(f"    b_0 = {pool_params['b_0']:.4f}, "
                  f"b_sigma = {pool_params['b_sigma']:.6f}, "
                  f"b_c = {pool_params['b_c']:.4f}")
            # Strip metadata keys (prefixed with _) — JAX can't trace strings
            sim_params = {k: v for k, v in pool_params.items()
                          if not k.startswith("_")}
            noise_config = {
                "noise_model": "loglinear",
                "reclamm_noise_params": sim_params,
            }
        elif args.calibrate_noise:
            from scripts.calibrate_reclamm_noise import (
                build_calibration_df,
                run_ols_calibration,
            )
            noise_model_name = (
                "tsoukalas_sqrt" if args.noise_model == "sqrt"
                else "tsoukalas_log"
            )
            print(f"\n  Calibrating noise model ({args.noise_model})...")
            cal_df = build_calibration_df(pool)
            noise_params, diag = run_ols_calibration(
                cal_df, pool.swap_fee, args.noise_model,
            )
            print(f"    R² = {diag['r_squared']:.4f}, n = {diag['n_obs']}")
            for key in ["a_0", "a_sigma", "a_c"]:
                param_key = "a_0_base" if key == "a_0" else key
                val = noise_params[param_key]
                se = diag["se"][key]
                t_stat = val / se if se > 0 else float("inf")
                print(f"    {key:>8} = {val:>10.4f}  (SE={se:.4f}, t={t_stat:.2f})")
            noise_config = {
                "noise_model": noise_model_name,
                "reclamm_noise_params": noise_params,
            }

        t0 = time.time()
        calibration = run_pool_calibration(
            pool, gas_costs, arb_freqs, noise_config=noise_config,
        )
        elapsed = time.time() - t0

        print_calibration_summary(pool, calibration)
        plot_pool_calibration(pool, calibration, output_dir=args.output_dir)
        all_results.append((pool, calibration))

        print(f"\n  Total time for {pool.label}: {elapsed:.0f}s")

    # Cross-pool summary
    if len(all_results) > 1:
        plot_cross_pool_summary(all_results, output_dir=args.output_dir)

    # Final summary table
    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n{'Pool':<16} {'Chain':<10} {'Best Gas':>9} {'Best Arb':>9} "
          f"{'Bias':>8} {'RMSE':>8} {'Days':>6}")
    print("-" * 70)
    for pool, cal in all_results:
        wg = cal["world_growth"]
        # Best = least-negative mean bias (closest from below)
        below_keys = [
            k for k in cal["sim_growths"]
            if np.mean(np.log(cal["sim_growths"][k] / wg)) < 0
        ]
        if below_keys:
            best_key = max(
                below_keys,
                key=lambda k: np.mean(np.log(cal["sim_growths"][k] / wg)),
            )
        else:
            best_key = min(
                cal["sim_growths"].keys(),
                key=lambda k: compute_log_rmse(cal["sim_growths"][k], wg),
            )
        best_bias = float(np.mean(np.log(cal["sim_growths"][best_key] / wg))) * 100
        best_rmse = compute_log_rmse(cal["sim_growths"][best_key], wg) * 100
        n_days = cal["days"][-1]
        gc_label = f"gas {best_key[0]}" if isinstance(best_key[0], str) else f"${best_key[0]}"
        print(f"{pool.label:<16} {pool.chain:<10} {gc_label:<9} "
              f"{best_key[1]:>4}min {best_bias:>+7.2f}% {best_rmse:>7.2f}% {n_days:>5.0f}d")

    # Save JSON summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary = []
    for pool, cal in all_results:
        wg_arr = cal["world_growth"]
        wg_final = float(wg_arr[-1])
        pool_summary = {
            "label": pool.label,
            "chain": pool.chain,
            "tokens": pool.tokens,
            "swap_fee": pool.swap_fee,
            "tvl_usd": pool.initial_pool_value_usd,
            "on_chain_params": pool.on_chain_params,
            "n_days": float(cal["days"][-1]),
            "n_governance_events": 1 if cal["governance_idx"] < cal["n_points"] else 0,
            "world_growth": wg_final,
            "grid_results": {},
        }
        for (gc, af) in sorted(cal["sim_growths"].keys(), key=lambda k: (str(k[0]), k[1])):
            sg_arr = cal["sim_growths"][(gc, af)]
            rmse = compute_log_rmse(sg_arr, wg_arr) * 100
            pool_summary["grid_results"][f"gas={gc}_arb={af}"] = {
                "gas_cost": gc,
                "arb_frequency": af,
                "sim_growth": float(sg_arr[-1]),
                "pct_deviation": float((sg_arr[-1] / wg_final - 1) * 100),
                "trajectory_rmse_pct": float(rmse),
            }
        summary.append(pool_summary)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"gas_calibration_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {json_path}")


if __name__ == "__main__":
    main()
