#!/usr/bin/env python3
"""Compare quantammsim reClAMM / Balancer vs reclamm-simulations repo + on-chain.

Runs:
  1. Zero-fee Balancer pool (quantammsim) — the normalization baseline
  2. reClAMM pool with on-chain params (quantammsim)
  3. Loads reclamm-simulations results + world values from CSV
  4. Gas-experiment runs: time-varying gas from on-chain percentiles,
     50% protocol fee take, on-chain fees

All comparisons align quantammsim's minute-level output to the world state
CSV's actual Unix timestamps, eliminating timing drift from block-time
variability.

4-panel plot matching the reclamm-simulations format:
  Top-left:     Price (WETH/AAVE) — both repos overlaid
  Top-right:    (legend)
  Bottom-left:  Absolute value in WETH
  Bottom-right: Value relative to feeless weighted (Balancer = 1.0)

Usage:
    python scripts/sim_vs_world_comparison.py
    python scripts/sim_vs_world_comparison.py --csv /path/to/csv
    python scripts/sim_vs_world_comparison.py --gas-experiment
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
from pathlib import Path
from datetime import datetime, timezone

from quantammsim.core_simulator.dynamic_inputs import DynamicInputFrames
from quantammsim.runners.jax_runners import do_run_on_historic_data

# ── On-chain reClAMM params ───────────────────────────────────────────────────
ONCHAIN_FEES = 0.0025

ONCHAIN_LAUNCH_PARAMS = {  # deployment through 2025-12-18
    "price_ratio": 1.5014,
    "centeredness_margin": 0.5,
    "shift_exponent": 0.1,
}
ONCHAIN_CURRENT_PARAMS = {  # post 2025-12-18 governance
    "price_ratio": 4.0,
    "centeredness_margin": 0.1,
    "shift_exponent": 0.001,
}
GOVERNANCE_DATE = "2025-12-18"

# CSV starts at ~17.2 WETH ≈ $50k at $2900/ETH.
INITIAL_POOL_VALUE = 50_000.0

# Gas cost = arb profit threshold in USD.
# reclamm-simulations uses profit_threshold = 3e-4 WETH (in token1 units).
# quantammsim's arb_thresh is in USD: 3 * 3e-4 WETH × ~$3000/ETH ≈ $2.70.
ARB_GAS_COST = 2.7

DEFAULT_CSV = (
    "[old simulation project path]"
    "/data/sim_vs_world_values_AAVE_WETH.csv"
)
ZEROFEE_CSV = (
    "[old simulation project path]"
    "/data/sim_vs_world_zerofee_centered_AAVE_WETH.csv"
)
ZEROFEE_MINUTE_CSV = (
    "[old simulation project path]"
    "/data/sim_vs_world_zerofee_centered_minute_AAVE_WETH.csv"
)
WORLD_STATE_CSV = (
    "[old simulation project path]"
    "/data/sim_vs_world_world_AAVE_WETH.csv"
)
DEFAULT_START = "2025-08-16 00:00:00"
DEFAULT_END = "2026-01-04 00:00:00"
DEFAULT_TOKENS = ["AAVE", "ETH"]
HALF_DAY = 720  # minutes

# Gas experiment
GAS_CSV_DIR = Path(__file__).resolve().parent.parent / "gas_csvs"
GAS_PERCENTILES = ["50p", "75p", "90p", "95p"]
GAS_SCALE_FACTORS = [0.25, 0.5, 0.75, 1.0]
FLAT_GAS_USD = [0.0, 0.25, 0.50, 1.0, 2.0, 3.0, 5.0]
PROTOCOL_FEE_SPLIT = 0.5


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--tokens", nargs="+", default=DEFAULT_TOKENS)
    p.add_argument("--output", default="sim_vs_world_comparison.png")
    p.add_argument(
        "--gas-experiment", action="store_true",
        help="Run gas-experiment sweep (time-varying gas, 50%% protocol fee)",
    )
    p.add_argument(
        "--launch-params", action="store_true",
        help="Use launch params instead of current params in gas experiment",
    )
    p.add_argument(
        "--gas-scale-sweep", action="store_true",
        help="Sweep gas cost scale factors, rebase to world, truncate at governance",
    )
    p.add_argument(
        "--best-gas", action="store_true",
        help="Run the 3 best gas configs vs world (clean plot)",
    )
    return p.parse_args()


def load_onchain_initial_state():
    """Load the on-chain pool state at t=0 from the world state CSV.

    Returns (state_dict, start_time_str) where state_dict has
    Ra, Rb, Va, Vb (token units) and start_time_str is rounded
    to the nearest minute for alignment with minute-level price data.
    """
    df = pd.read_csv(WORLD_STATE_CSV)
    r = df.iloc[0]
    state = {
        "Ra": float(r.balance_0),
        "Rb": float(r.balance_1),
        "Va": float(r.virtual_0),
        "Vb": float(r.virtual_1),
    }
    # Round to nearest minute for price data alignment
    ts_sec = int(r.timestamp)
    ts_minute = (ts_sec // 60) * 60
    start_str = datetime.utcfromtimestamp(ts_minute).strftime("%Y-%m-%d %H:%M:%S")
    return state, start_str


def load_world_timestamps():
    """Load Unix timestamps (seconds) from the world state CSV."""
    df = pd.read_csv(WORLD_STATE_CSV)
    return df["timestamp"].values


def load_world_normalized_balances():
    """Load BPT-normalized on-chain balances and timestamps.

    Normalizes balances to initial BPT supply so that value tracks a
    fixed LP position (accounts for joins/exits changing BPT supply).

    Returns (norm_bal_0, norm_bal_1, timestamps_sec).
    """
    df = pd.read_csv(WORLD_STATE_CSV)
    bpt_0 = df["bpt_supply"].iloc[0]
    norm = bpt_0 / df["bpt_supply"].values
    return (
        df["balance_0"].values * norm,
        df["balance_1"].values * norm,
        df["timestamp"].values,
    )


def sample_at_timestamps(minute_vals, start_unix_sec, timestamps_sec):
    """Sample a minute-level array at specific Unix timestamps.

    For each target timestamp, finds the nearest minute index in the
    sim output and returns the corresponding value.

    Parameters
    ----------
    minute_vals : array, shape (N,)
        Minute-level sim output.
    start_unix_sec : float
        Unix timestamp (seconds) of minute_vals[0].
    timestamps_sec : array
        Unix timestamps (seconds) to sample at.

    Returns
    -------
    array : values at the nearest minute to each target timestamp.
    """
    indices = np.round((timestamps_sec - start_unix_sec) / 60).astype(int)
    indices = np.clip(indices, 0, len(minute_vals) - 1)
    return minute_vals[indices]


def run_pool(
    tokens,
    start,
    end,
    rule,
    fees,
    params,
    gas_cost=0.0,
    protocol_fee_split=0.0,
    dynamic_input_frames=None,
    onchain_initial_state=None,
):
    """Run a quantammsim pool and return minute-level results.

    Returns (val_eth, price_ratio, start_unix_sec) where val_eth and
    price_ratio are minute-level arrays and start_unix_sec is the Unix
    timestamp (seconds) of the first element.
    """
    fp = {
        "tokens": tokens,
        "rule": rule,
        "startDateString": start,
        "endDateString": end,
        "initial_pool_value": INITIAL_POOL_VALUE,
        "fees": fees,
        "gas_cost": gas_cost,
        "arb_fees": 0.0,
        "do_arb": True,
        "arb_frequency": 1,
        "chunk_period": 1440,
        "weight_interpolation_period": 1440,
    }
    if rule == "reclamm":
        fp["reclamm_use_shift_exponent"] = True
        fp["reclamm_interpolation_method"] = "geometric"
        fp["reclamm_centeredness_scaling"] = False
    if protocol_fee_split != 0.0:
        fp["protocol_fee_split"] = protocol_fee_split
    if onchain_initial_state is not None:
        fp["reclamm_initial_state"] = onchain_initial_state

    result = do_run_on_historic_data(
        run_fingerprint=fp,
        params=params,
        dynamic_input_frames=dynamic_input_frames,
    )

    # Prices: sorted tokens → [AAVE, ETH] in USD
    prices = np.array(result["prices"])
    eth_usd = prices[:, 1]
    price_ratio = prices[:, 0] / prices[:, 1]  # WETH/AAVE

    # Pool value in ETH
    val_eth = np.array(result["value"]) / eth_usd

    # Compute start timestamp from startDateString
    start_unix_sec = datetime.strptime(
        start, "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc).timestamp()

    return val_eth, price_ratio, start_unix_sec


def load_gas_csv(percentile):
    """Load a gas CSV and return a DataFrame with columns [unix, trade_gas_cost_usd].

    Gas CSV timestamps are offset by ~59s from exact minutes.  Round down
    to the nearest minute so they align with the simulator's minute-level index.
    """
    path = GAS_CSV_DIR / f"Gas_{percentile}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"USD": "trade_gas_cost_usd"})
    df["unix"] = (df["unix"] // 60000) * 60000  # floor to minute boundary
    return df


def run_gas_experiment(args):
    """Run gas-experiment sweep and produce comparison plot."""
    tokens = args.tokens
    start, end = args.start, args.end

    # ── Select params ─────────────────────────────────────────────────
    if args.launch_params:
        param_source = ONCHAIN_LAUNCH_PARAMS
        param_label = "launch"
    else:
        param_source = ONCHAIN_CURRENT_PARAMS
        param_label = "current"
    pool_params = {k: jnp.array(v) for k, v in param_source.items()}

    # ── Baselines ──────────────────────────────────────────────────────
    print("Running Balancer (zero-fee 50/50)...")
    bal_params = {"initial_weights_logits": jnp.array([0.0, 0.0])}
    bal_eth_min, qsim_price_min, start_sec = run_pool(
        tokens, start, end, "balancer", 0.0, bal_params,
    )

    print(f"Running reClAMM ({param_label} params, flat gas, no protocol fee)...")
    reclamm_flat_min, _, _ = run_pool(
        tokens, start, end, "reclamm", ONCHAIN_FEES, pool_params,
        gas_cost=ARB_GAS_COST,
    )

    # ── Load world values from CSV ─────────────────────────────────────
    print("Loading reclamm-simulations CSV...")
    df = pd.read_csv(args.csv)

    # ── Gas percentile runs ────────────────────────────────────────────
    gas_results_min = {}
    for pct in GAS_PERCENTILES:
        print(f"Running reClAMM ({param_label} params, gas={pct}, "
              f"protocol_fee={PROTOCOL_FEE_SPLIT})...")
        gas_df = load_gas_csv(pct)
        val_eth_min, _, _ = run_pool(
            tokens, start, end, "reclamm", ONCHAIN_FEES, pool_params,
            protocol_fee_split=PROTOCOL_FEE_SPLIT,
            dynamic_input_frames=DynamicInputFrames(gas_cost=gas_df),
        )
        gas_results_min[pct] = val_eth_min

    # ── Sample at world timestamps ────────────────────────────────────
    world_ts = load_world_timestamps()
    n = min(len(df), len(world_ts))
    world_ts = world_ts[:n]

    bal_eth = sample_at_timestamps(bal_eth_min, start_sec, world_ts)
    reclamm_flat_eth = sample_at_timestamps(reclamm_flat_min, start_sec, world_ts)
    qsim_price = sample_at_timestamps(qsim_price_min, start_sec, world_ts)
    gas_results = {
        pct: sample_at_timestamps(v, start_sec, world_ts)
        for pct, v in gas_results_min.items()
    }

    csv_world = df["world"].values[:n]
    csv_feeless = df["feeless weighted"].values[:n]
    print(f"  Aligned: {n} world-timestamp points")
    t = np.arange(n)

    # Governance half-day index
    gov_unix = datetime.strptime(
        GOVERNANCE_DATE, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc).timestamp()
    gov_idx = np.searchsorted(world_ts, gov_unix)

    # ── Plot: relative to feeless weighted ─────────────────────────────
    fig, (ax_price, ax_rel) = plt.subplots(2, 1, figsize=(14, 9),
                                            gridspec_kw={"height_ratios": [1, 2]})

    # Top: price
    ax_price.plot(t, qsim_price, color="gray", alpha=0.6, linewidth=1)
    ax_price.set_ylabel("AAVE/ETH")
    ax_price.set_title("Price")
    ax_price.set_ylim(bottom=0)
    if gov_idx < n:
        ax_price.axvline(x=gov_idx, color="gray", linestyle=":", alpha=0.6)

    # Bottom: relative values
    ax_rel.axhline(y=1.0, color="blue", linewidth=2, label="feeless weighted")

    # Flat-gas baseline (no protocol fee)
    flat_rel = reclamm_flat_eth / bal_eth
    ax_rel.plot(t, flat_rel, linewidth=2, color="gray", linestyle="--",
                label=f"flat gas ${ARB_GAS_COST}, no protocol fee")

    # Gas percentile runs
    colors = {"50p": "#2ca02c", "75p": "#ff7f0e", "90p": "#d62728", "95p": "#9467bd"}
    for pct in GAS_PERCENTILES:
        vals = gas_results[pct]
        rel = vals / bal_eth
        ax_rel.plot(t, rel, linewidth=1.5, color=colors[pct],
                    label=f"gas {pct}, {int(PROTOCOL_FEE_SPLIT*100)}% protocol fee")

    # World values
    world_rel = csv_world / csv_feeless
    ax_rel.plot(t, world_rel, linewidth=1.5, marker=".", markersize=2,
                color="brown", label="world (on-chain)")

    ax_rel.set_xlabel("half days")
    ax_rel.set_ylabel("value / feeless weighted")
    ax_rel.set_title("LP value relative to feeless weighted (Balancer 50/50)")
    ax_rel.legend(fontsize=8, loc="lower left")
    ax_rel.grid(True, alpha=0.2)
    if gov_idx < n:
        ax_rel.axvline(x=gov_idx, color="gray", linestyle=":", alpha=0.6)
        ax_rel.text(gov_idx + 1, ax_rel.get_ylim()[1] * 0.98,
                    "governance", fontsize=7, color="gray", va="top")

    tokens_str = "/".join(tokens)
    fig.suptitle(
        f"reClAMM gas experiment ({param_label} params) — {tokens_str}\n"
        f"params: {list(param_source.values())}, "
        f"fees: {ONCHAIN_FEES}, protocol fee: {PROTOCOL_FEE_SPLIT}",
        fontsize=10,
    )
    plt.tight_layout()
    out = args.output.replace(".png", f"_gas_experiment_{param_label}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'Scenario':<45} {'Final rel':>10} {'vs world':>10}")
    print("-" * 65)
    world_final_rel = world_rel[-1] if len(world_rel) > 0 else float("nan")
    print(f"{'Flat gas, no protocol fee':<45} {flat_rel[-1]:>10.4f} "
          f"{flat_rel[-1] - world_final_rel:>+10.4f}")
    for pct in GAS_PERCENTILES:
        rel = gas_results[pct] / bal_eth
        print(f"{'Gas ' + pct + f', {int(PROTOCOL_FEE_SPLIT*100)}% protocol fee':<45} "
              f"{rel[-1]:>10.4f} {rel[-1] - world_final_rel:>+10.4f}")
    print(f"{'World (on-chain)':<45} {world_final_rel:>10.4f}")


def run_gas_scale_experiment(args):
    """Sweep gas cost scale factors, rebase to world, truncate at governance."""
    tokens = args.tokens
    end = args.end

    if args.launch_params:
        param_source = ONCHAIN_LAUNCH_PARAMS
        param_label = "launch"
    else:
        param_source = ONCHAIN_CURRENT_PARAMS
        param_label = "current"
    pool_params = {k: jnp.array(v) for k, v in param_source.items()}

    # Load on-chain initial state and derive start time
    onchain_state, onchain_start = load_onchain_initial_state()
    start = onchain_start
    print(f"On-chain initial state: Ra={onchain_state['Ra']:.2f}, "
          f"Rb={onchain_state['Rb']:.2f}, Va={onchain_state['Va']:.2f}, "
          f"Vb={onchain_state['Vb']:.2f}")
    print(f"Sim start time (from on-chain): {start}")

    # Load world + reclamm-simulations values
    print("Loading reclamm-simulations CSV...")
    df = pd.read_csv(args.csv)

    # Load world timestamps and find governance cutoff
    world_ts = load_world_timestamps()
    gov_unix = datetime.strptime(
        GOVERNANCE_DATE, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc).timestamp()
    gov_idx = np.searchsorted(world_ts, gov_unix)

    # Run all (percentile, scale) combinations
    results_min = {}
    price_ratio_min = None
    for pct in GAS_PERCENTILES:
        gas_df_raw = load_gas_csv(pct)
        for scale in GAS_SCALE_FACTORS:
            label = f"{pct} × {scale}"
            print(f"Running reClAMM ({param_label}, gas={label}, "
                  f"protocol_fee={PROTOCOL_FEE_SPLIT})...")
            gas_df = gas_df_raw.copy()
            gas_df["trade_gas_cost_usd"] = gas_df_raw["trade_gas_cost_usd"] * scale
            val_eth_min, pr_min, start_sec = run_pool(
                tokens, start, end, "reclamm", ONCHAIN_FEES, pool_params,
                protocol_fee_split=PROTOCOL_FEE_SPLIT,
                dynamic_input_frames=DynamicInputFrames(gas_cost=gas_df),
                onchain_initial_state=onchain_state,
            )
            results_min[(pct, scale)] = (val_eth_min, start_sec)
            if price_ratio_min is None:
                price_ratio_min = pr_min

    # Flat gas cost runs
    flat_results_min = {}
    for gas_usd in FLAT_GAS_USD:
        print(f"Running reClAMM ({param_label}, flat gas=${gas_usd}, "
              f"protocol_fee={PROTOCOL_FEE_SPLIT})...")
        val_eth_min, _, start_sec = run_pool(
            tokens, start, end, "reclamm", ONCHAIN_FEES, pool_params,
            gas_cost=gas_usd, protocol_fee_split=PROTOCOL_FEE_SPLIT,
            onchain_initial_state=onchain_state,
        )
        flat_results_min[gas_usd] = (val_eth_min, start_sec)

    # ── World values: on-chain balances × quantammsim prices ──────────
    world_bal_0, world_bal_1, world_ts = load_world_normalized_balances()

    # reclamm-sim comparison uses its own CSV (self-consistent pricing)
    csv_world = df["world"].values
    csv_sim = df["simulation"].values

    n = min(gov_idx, len(world_bal_0), len(csv_world), len(csv_sim), len(world_ts))
    print(f"  Truncated at governance: {n} world-timestamp points")
    t = np.arange(n)
    world_ts_trunc = world_ts[:n]

    # Repriced world for quantammsim comparison
    price_at_world = sample_at_timestamps(
        price_ratio_min, start_sec, world_ts_trunc,
    )
    world_val = world_bal_0[:n] * price_at_world + world_bal_1[:n]
    world_growth = world_val / world_val[0]

    # CSV-based world for reclamm-sim comparison (self-consistent pricing)
    csv_world = csv_world[:n]
    csv_sim = csv_sim[:n]
    world_growth_csv = csv_world / csv_world[0]
    recsim_growth = csv_sim / csv_sim[0]

    # Sample all sim runs at world timestamps
    start_sec = flat_results_min[FLAT_GAS_USD[0]][1]

    results = {}
    for key, (val_min, _) in results_min.items():
        results[key] = sample_at_timestamps(val_min, start_sec, world_ts_trunc)

    flat_results = {}
    for gas_usd, (val_min, _) in flat_results_min.items():
        flat_results[gas_usd] = sample_at_timestamps(val_min, start_sec, world_ts_trunc)

    # Compute growth ratios
    flat_growths = {}
    for gas_usd in FLAT_GAS_USD:
        vals = flat_results[gas_usd]
        flat_growths[gas_usd] = vals / vals[0]

    # ── Plot (% deviation from world: positive = sim below world) ───
    fig, (ax_ts, ax_pct, ax_flat) = plt.subplots(
        1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [3, 1, 1]},
    )

    # Left: time series of % deviation from world
    ax_ts.axhline(y=0.0, color="brown", linewidth=2, label="world (on-chain)")

    # reclamm-simulations (uses CSV-based world for self-consistent pricing)
    recsim_dev = (1 - recsim_growth / world_growth_csv) * 100
    ax_ts.plot(t, recsim_dev, color="red", linewidth=2,
               linestyle="--", label="reclamm-sim")

    # Gas scale sweep (percentile-based)
    colors = {"50p": "#2ca02c", "75p": "#ff7f0e", "90p": "#d62728", "95p": "#9467bd"}
    for pct in GAS_PERCENTILES:
        for scale in GAS_SCALE_FACTORS:
            vals = results[(pct, scale)]
            sim_growth = vals / vals[0]
            dev = (1 - sim_growth / world_growth) * 100
            alpha = 0.3 + 0.7 * scale
            lw = 0.8 + 1.2 * scale
            if scale == 1.0:
                label = f"{pct} × {scale}"
            elif pct == "50p":
                label = f"50p × {scale}"
            else:
                label = None
            ax_ts.plot(t, dev, color=colors[pct], alpha=alpha,
                       linewidth=lw, label=label)

    # Flat gas runs
    flat_cmap = plt.cm.copper
    for i, gas_usd in enumerate(FLAT_GAS_USD):
        c = flat_cmap(i / max(len(FLAT_GAS_USD) - 1, 1))
        dev = (1 - flat_growths[gas_usd] / world_growth) * 100
        ax_ts.plot(t, dev, color=c, linewidth=1.5, linestyle="-.",
                   label=f"flat ${gas_usd}")

    ax_ts.set_xlabel("half days")
    ax_ts.set_ylabel("% deviation from world")
    ax_ts.set_title("LP value vs world (pre-governance)")
    ax_ts.legend(fontsize=6, loc="best", ncol=2)
    ax_ts.grid(True, alpha=0.2)

    # Reference lines for both summary panels (as % deviation)
    recsim_final_dev = (1 - recsim_growth[-1] / world_growth_csv[-1]) * 100

    # Middle: final % deviation vs percentile scale factor
    ax_pct.axhline(y=0.0, color="brown", linewidth=2, label="world")
    ax_pct.axhline(y=recsim_final_dev, color="red", linewidth=1.5,
                    linestyle="--", label=f"reclamm-sim ({recsim_final_dev:+.2f}%)")

    for pct in GAS_PERCENTILES:
        finals = []
        for scale in GAS_SCALE_FACTORS:
            vals = results[(pct, scale)]
            sim_growth = vals / vals[0]
            finals.append((1 - sim_growth[-1] / world_growth[-1]) * 100)
        ax_pct.plot(GAS_SCALE_FACTORS, finals, marker="o",
                    color=colors[pct], linewidth=2, label=pct)

    ax_pct.set_xlabel("gas scale factor\n(1.0 = 450k gas)")
    ax_pct.set_ylabel("% deviation from world")
    ax_pct.set_title("Percentile gas")
    ax_pct.legend(fontsize=6)
    ax_pct.grid(True, alpha=0.2)

    # Right: final % deviation vs flat gas cost
    ax_flat.axhline(y=0.0, color="brown", linewidth=2, label="world")
    ax_flat.axhline(y=recsim_final_dev, color="red", linewidth=1.5,
                     linestyle="--", label=f"reclamm-sim ({recsim_final_dev:+.2f}%)")

    flat_finals = []
    for gas_usd in FLAT_GAS_USD:
        flat_finals.append(
            (1 - flat_growths[gas_usd][-1] / world_growth[-1]) * 100
        )
    ax_flat.plot(FLAT_GAS_USD, flat_finals, marker="s", color="black",
                 linewidth=2, label="flat gas")

    ax_flat.set_xlabel("flat gas cost (USD)")
    ax_flat.set_ylabel("% deviation from world")
    ax_flat.set_title("Flat gas")
    ax_flat.legend(fontsize=6)
    ax_flat.grid(True, alpha=0.2)

    tokens_str = "/".join(tokens)
    fig.suptitle(
        f"reClAMM gas sweep ({param_label} params) — {tokens_str}\n"
        f"params: {list(param_source.values())}, "
        f"fees: {ONCHAIN_FEES}, protocol fee: {PROTOCOL_FEE_SPLIT}",
        fontsize=10,
    )
    plt.tight_layout()
    out = args.output.replace(".png", f"_gas_scale_{param_label}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()

    # ── Summary table (% deviation from world) ─────────────────────────
    print(f"\n{'Scenario':<35} {'% dev from world':>16}")
    print("-" * 52)
    print(f"{'reclamm-sim':<35} {recsim_final_dev:>+16.2f}%")
    print()
    for gas_usd in FLAT_GAS_USD:
        dev = (1 - flat_growths[gas_usd][-1] / world_growth[-1]) * 100
        print(f"{'Flat $' + f'{gas_usd}':<35} {dev:>+16.2f}%")
    print()
    for pct in GAS_PERCENTILES:
        for scale in GAS_SCALE_FACTORS:
            vals = results[(pct, scale)]
            sim_growth = vals / vals[0]
            dev = (1 - sim_growth[-1] / world_growth[-1]) * 100
            print(f"{'Gas ' + pct + f' × {scale}':<35} {dev:>+16.2f}%")


def run_best_gas_experiment(args):
    """Run the 3 best gas configs vs world on a clean single-panel plot."""
    tokens = args.tokens
    end = args.end

    if args.launch_params:
        param_source = ONCHAIN_LAUNCH_PARAMS
        param_label = "launch"
    else:
        param_source = ONCHAIN_CURRENT_PARAMS
        param_label = "current"
    pool_params = {k: jnp.array(v) for k, v in param_source.items()}

    # Load on-chain initial state and derive start time
    onchain_state, onchain_start = load_onchain_initial_state()
    start = onchain_start
    print(f"On-chain initial state: Ra={onchain_state['Ra']:.2f}, "
          f"Rb={onchain_state['Rb']:.2f}, Va={onchain_state['Va']:.2f}, "
          f"Vb={onchain_state['Vb']:.2f}")
    print(f"Sim start time (from on-chain): {start}")

    # Find governance cutoff from world timestamps
    world_ts_all = load_world_timestamps()
    gov_unix = datetime.strptime(
        GOVERNANCE_DATE, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc).timestamp()
    gov_idx = np.searchsorted(world_ts_all, gov_unix)

    # ── The best configs ─────────────────────────────────────────────
    configs = [
        ("Flat $1.00", "black", "-"),
        ("50p × 1.0", "#2ca02c", "-"),
        ("75p × 0.75", "#ff7f0e", "-"),
        ("90p × 0.25", "#d62728", "-"),
    ]

    # 1) Flat $1.00
    print(f"Running reClAMM ({param_label}, flat gas=$1.00, "
          f"protocol_fee={PROTOCOL_FEE_SPLIT})...")
    flat1_min, price_ratio_min, start_sec = run_pool(
        tokens, start, end, "reclamm", ONCHAIN_FEES, pool_params,
        gas_cost=1.0, protocol_fee_split=PROTOCOL_FEE_SPLIT,
        onchain_initial_state=onchain_state,
    )

    # 2) 50p × 1.0
    print(f"Running reClAMM ({param_label}, gas=50p × 1.0, "
          f"protocol_fee={PROTOCOL_FEE_SPLIT})...")
    gas_df_50p = load_gas_csv("50p")
    g50_min, _, _ = run_pool(
        tokens, start, end, "reclamm", ONCHAIN_FEES, pool_params,
        protocol_fee_split=PROTOCOL_FEE_SPLIT,
        dynamic_input_frames=DynamicInputFrames(gas_cost=gas_df_50p),
        onchain_initial_state=onchain_state,
    )

    # 3) 75p × 0.75
    print(f"Running reClAMM ({param_label}, gas=75p × 0.75, "
          f"protocol_fee={PROTOCOL_FEE_SPLIT})...")
    gas_df_75p = load_gas_csv("75p")
    gas_df_75p_scaled = gas_df_75p.copy()
    gas_df_75p_scaled["trade_gas_cost_usd"] *= 0.75
    g75_min, _, _ = run_pool(
        tokens, start, end, "reclamm", ONCHAIN_FEES, pool_params,
        protocol_fee_split=PROTOCOL_FEE_SPLIT,
        dynamic_input_frames=DynamicInputFrames(gas_cost=gas_df_75p_scaled),
        onchain_initial_state=onchain_state,
    )

    # 4) 90p × 0.25
    print(f"Running reClAMM ({param_label}, gas=90p × 0.25, "
          f"protocol_fee={PROTOCOL_FEE_SPLIT})...")
    gas_df_90p = load_gas_csv("90p")
    gas_df_90p_scaled = gas_df_90p.copy()
    gas_df_90p_scaled["trade_gas_cost_usd"] *= 0.25
    g90_min, _, _ = run_pool(
        tokens, start, end, "reclamm", ONCHAIN_FEES, pool_params,
        protocol_fee_split=PROTOCOL_FEE_SPLIT,
        dynamic_input_frames=DynamicInputFrames(gas_cost=gas_df_90p_scaled),
        onchain_initial_state=onchain_state,
    )

    # ── World values: on-chain balances × quantammsim prices ──────────
    # Both sim and world valued at the same price at each point,
    # so price fluctuations cancel in the growth ratio comparison.
    world_bal_0, world_bal_1, world_ts = load_world_normalized_balances()
    n = min(gov_idx, len(world_bal_0), len(world_ts))
    print(f"  Truncated at governance: {n} world-timestamp points")
    t = np.arange(n)
    world_ts_trunc = world_ts[:n]

    # Sample quantammsim price ratio at world timestamps
    price_at_world = sample_at_timestamps(
        price_ratio_min, start_sec, world_ts_trunc,
    )
    # World value in ETH = norm_AAVE * (AAVE/ETH) + norm_ETH
    world_val = world_bal_0[:n] * price_at_world + world_bal_1[:n]
    world_growth = world_val / world_val[0]

    run_vals = [
        sample_at_timestamps(flat1_min, start_sec, world_ts_trunc),
        sample_at_timestamps(g50_min, start_sec, world_ts_trunc),
        sample_at_timestamps(g75_min, start_sec, world_ts_trunc),
        sample_at_timestamps(g90_min, start_sec, world_ts_trunc),
    ]

    growths = [v / v[0] for v in run_vals]

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.axhline(y=0.0, color="brown", linewidth=2, label="world (on-chain)")

    # Best 3
    for (label, color, ls), g in zip(configs, growths):
        dev = (1 - g / world_growth) * 100
        final_dev = dev[-1]
        ax.plot(t, dev, color=color, linewidth=2, linestyle=ls,
                label=f"{label} (final {final_dev:+.2f}%)")

    ax.set_xlabel("half days")
    ax.set_ylabel("% deviation from world")
    ax.set_title(
        f"Best gas configs vs world ({param_label} params) — "
        f"{'/'.join(tokens)}\n"
        f"params: {list(param_source.values())}, "
        f"fees: {ONCHAIN_FEES}, protocol fee: {PROTOCOL_FEE_SPLIT}",
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = args.output.replace(".png", f"_best_gas_{param_label}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()

    # Summary
    labels = [c[0] for c in configs]
    print(f"\n{'Scenario':<25} {'% dev from world':>16}")
    print("-" * 42)
    for label, g in zip(labels, growths):
        dev = (1 - g[-1] / world_growth[-1]) * 100
        print(f"{label:<25} {dev:>+16.2f}%")


def main():
    args = parse_args()

    if args.best_gas:
        run_best_gas_experiment(args)
        return

    if args.gas_scale_sweep:
        run_gas_scale_experiment(args)
        return

    if args.gas_experiment:
        run_gas_experiment(args)
        return

    # ── Load CSVs ─────────────────────────────────────────────────────
    print("Loading reclamm-simulations CSV...")
    df = pd.read_csv(args.csv)
    n_csv = len(df)
    print(f"  {n_csv} half-day points")

    print("Loading zero-fee minute-level CSV...")
    df_zf_min = pd.read_csv(ZEROFEE_MINUTE_CSV)
    print(f"  {len(df_zf_min)} minute points")

    # Load world timestamps for alignment
    world_ts = load_world_timestamps()

    # ── Run quantammsim pools (minute-level) ──────────────────────────
    print("Running Balancer (zero-fee 50/50)...")
    bal_params = {"initial_weights_logits": jnp.array([0.0, 0.0])}
    bal_eth_min, qsim_price_min, start_sec = run_pool(
        args.tokens, args.start, args.end, "balancer", 0.0, bal_params,
    )

    print("Running reClAMM (launch, zero-fee, zero-gas)...")
    launch_params = {k: jnp.array(v) for k, v in ONCHAIN_LAUNCH_PARAMS.items()}
    reclamm_zerofee_min, _, _ = run_pool(
        args.tokens, args.start, args.end, "reclamm", 0.0, launch_params,
        gas_cost=0.0,
    )

    print(f"Running reClAMM (launch params, gas=${ARB_GAS_COST})...")
    reclamm_launch_min, _, _ = run_pool(
        args.tokens, args.start, args.end, "reclamm", ONCHAIN_FEES, launch_params,
        gas_cost=ARB_GAS_COST,
    )

    print(f"Running reClAMM (current params, gas=${ARB_GAS_COST})...")
    current_params = {k: jnp.array(v) for k, v in ONCHAIN_CURRENT_PARAMS.items()}
    reclamm_current_min, _, _ = run_pool(
        args.tokens, args.start, args.end, "reclamm", ONCHAIN_FEES, current_params,
        gas_cost=ARB_GAS_COST,
    )

    # ── Sample at world timestamps ────────────────────────────────────
    n = min(n_csv, len(world_ts))
    world_ts_trunc = world_ts[:n]

    bal_eth = sample_at_timestamps(bal_eth_min, start_sec, world_ts_trunc)
    reclamm_zerofee_eth = sample_at_timestamps(reclamm_zerofee_min, start_sec, world_ts_trunc)
    reclamm_launch_eth = sample_at_timestamps(reclamm_launch_min, start_sec, world_ts_trunc)
    reclamm_current_eth = sample_at_timestamps(reclamm_current_min, start_sec, world_ts_trunc)
    qsim_price = sample_at_timestamps(qsim_price_min, start_sec, world_ts_trunc)

    print(f"  Aligned: {n} world-timestamp points "
          f"(qsim minutes={len(bal_eth_min)}, csv={n_csv})")
    t = np.arange(n)

    csv_price = df["price"].values[:n]
    csv_feeless = df["feeless weighted"].values[:n]
    csv_sim = df["simulation"].values[:n]
    csv_hold = df["hold"].values[:n]
    csv_world = df["world"].values[:n]

    # Governance change index
    gov_unix = datetime.strptime(
        GOVERNANCE_DATE, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc).timestamp()
    gov_idx = np.searchsorted(world_ts_trunc, gov_unix)

    # Normalize quantammsim to same starting value as CSV
    v0 = csv_feeless[0]
    bal_norm = bal_eth * (v0 / bal_eth[0])
    zerofee_norm = reclamm_zerofee_eth * (v0 / reclamm_zerofee_eth[0])
    launch_norm = reclamm_launch_eth * (v0 / reclamm_launch_eth[0])
    current_norm = reclamm_current_eth * (v0 / reclamm_current_eth[0])

    # Relative values (÷ respective feeless weighted baseline)
    zerofee_rel = reclamm_zerofee_eth / bal_eth
    launch_rel = reclamm_launch_eth / bal_eth
    current_rel = reclamm_current_eth / bal_eth
    csv_sim_rel = csv_sim / csv_feeless
    csv_hold_rel = csv_hold / csv_feeless
    csv_world_rel = csv_world / csv_feeless

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    # Top-left: price
    axs[0][0].plot(t, csv_price, label="reclamm-sim", alpha=0.8)
    axs[0][0].plot(t, qsim_price, label="quantammsim", alpha=0.8, linestyle="--")
    axs[0][0].set_ylabel("WETH/AAVE")
    axs[0][0].set_title("Price")
    axs[0][0].set_ylim(bottom=0)
    axs[0][0].legend(fontsize=8)
    if gov_idx < n:
        axs[0][0].axvline(x=gov_idx, color="gray", linestyle=":", alpha=0.6)

    # Top-right: remove (legend is on other panels)
    axs[0][1].remove()

    # Bottom-left: absolute values in WETH
    axs[1][0].plot(t, bal_norm, label="qsim feeless weighted", linewidth=2, color="blue")
    axs[1][0].plot(t, launch_norm, label="qsim reClAMM (launch)", linewidth=2, color="orange")
    axs[1][0].plot(t, current_norm, label="qsim reClAMM (current)", linewidth=2,
                   color="purple", linestyle="-.")
    axs[1][0].plot(t, csv_sim, label="reclamm-sim simulation", linewidth=1.5,
                   linestyle="--", color="red")
    axs[1][0].plot(t, csv_hold, label="hold", linewidth=1.5, color="green")
    axs[1][0].plot(t, csv_world, label="world values", linewidth=1.5,
                   marker=".", markersize=2, color="brown")
    axs[1][0].set_title("Value histories")
    axs[1][0].set_xlabel("half days")
    axs[1][0].set_ylabel("Value in WETH")
    axs[1][0].set_ylim(bottom=0)
    axs[1][0].legend(fontsize=7, loc="upper right")
    if gov_idx < n:
        axs[1][0].axvline(x=gov_idx, color="gray", linestyle=":", alpha=0.6)
        axs[1][0].text(gov_idx + 1, axs[1][0].get_ylim()[1] * 0.95,
                       "governance", fontsize=7, color="gray", va="top")

    # Bottom-right: relative to feeless weighted
    axs[1][1].axhline(y=1.0, color="blue", linewidth=2, label="feeless weighted")
    axs[1][1].plot(t, launch_rel, label="qsim reClAMM (launch)", linewidth=2, color="orange")
    axs[1][1].plot(t, current_rel, label="qsim reClAMM (current)", linewidth=2,
                   color="purple", linestyle="-.")
    axs[1][1].plot(t, csv_sim_rel, label="reclamm-sim simulation", linewidth=1.5,
                   linestyle="--", color="red")
    axs[1][1].plot(t, csv_hold_rel, label="hold", linewidth=1.5, color="green")
    axs[1][1].plot(t, csv_world_rel, label="world values", linewidth=1.5,
                   marker=".", markersize=2, color="brown")
    axs[1][1].set_title("Value relative to feeless weighted")
    axs[1][1].set_xlabel("half days")
    axs[1][1].set_ylabel("relative value")
    axs[1][1].legend(fontsize=7, loc="lower left")
    if gov_idx < n:
        axs[1][1].axvline(x=gov_idx, color="gray", linestyle=":", alpha=0.6)

    tokens_str = "/".join(args.tokens)
    fig.suptitle(
        f"quantammsim vs reclamm-simulations — {tokens_str}\n"
        f"Launch: {list(ONCHAIN_LAUNCH_PARAMS.values())}, "
        f"Current: {list(ONCHAIN_CURRENT_PARAMS.values())}, "
        f"fees: {ONCHAIN_FEES}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.output}")

    # ── Zero-fee comparison plot (minute-level) ───────────────────────
    # Revalue reclamm-sim balances at quantammsim's price so both sides
    # use the same price and the comparison is purely about balances.
    # Skip row 0 of the CSV (initial state before first arb) to align
    # with quantammsim's reserves[0] which is post-first-step.
    ext_bal_0 = df_zf_min["balance_0"].values[1:]
    ext_bal_1 = df_zf_min["balance_1"].values[1:]
    n_zf = min(len(reclamm_zerofee_min), len(ext_bal_0), len(qsim_price_min))
    ext_val_repriced = (
        ext_bal_0[:n_zf] * qsim_price_min[:n_zf] + ext_bal_1[:n_zf]
    )
    qsim_growth = reclamm_zerofee_min[:n_zf] / reclamm_zerofee_min[0]
    ext_growth = ext_val_repriced[:n_zf] / ext_val_repriced[0]
    pct_dev = (qsim_growth / ext_growth - 1) * 100
    days = np.arange(n_zf) / 1440

    zerofee_title = (
        f"Zero-fee zero-gas reClAMM: quantammsim / reclamm-sim (minute-level) — {tokens_str}\n"
        f"params: {list(ONCHAIN_LAUNCH_PARAMS.values())}"
    )
    daily_smooth = pd.Series(pct_dev).rolling(1440, center=True, min_periods=720).mean()

    # Plot 1: with daily smoothing overlay
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(days, pct_dev, linewidth=0.5, color="teal", alpha=0.6)
    ax2.plot(days, daily_smooth, linewidth=2, color="darkblue", label="daily smoothed")
    ax2.axhline(y=0.0, color="gray", linestyle="--", alpha=0.6)
    ax2.set_xlabel("days")
    ax2.set_ylabel("deviation (%)")
    ax2.set_title(zerofee_title, fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    zerofee_path = args.output.replace(".png", "_zerofee_ratio.png")
    plt.savefig(zerofee_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {zerofee_path}")
    plt.close()

    # Plot 2: raw minute-level only (no smoothing)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(days, pct_dev, linewidth=0.5, color="teal", alpha=0.8)
    ax3.axhline(y=0.0, color="gray", linestyle="--", alpha=0.6)
    ax3.set_xlabel("days")
    ax3.set_ylabel("deviation (%)")
    ax3.set_title(zerofee_title, fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    zerofee_raw_path = args.output.replace(".png", "_zerofee_ratio_raw.png")
    plt.savefig(zerofee_raw_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {zerofee_raw_path}")
    plt.close()


if __name__ == "__main__":
    main()
