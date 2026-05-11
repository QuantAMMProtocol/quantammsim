"""Diagnostic: sim vs world absolute pool value for a single (gas=0, arb_freq=1) run.

Plots raw USD pool value over time for both sim and world, no per-LP normalization.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from experiments.pool_registry import (
    POOL_REGISTRY, extract_on_chain_state, extract_initial_state,
    get_data_end_date, load_world_history, load_bpt_supply_df,
)
from experiments.run_pool_battery import run_sim, sample_at_timestamps, _start_str_from_pool


def main():
    pool = POOL_REGISTRY["cbBTC_WETH"]
    extract_on_chain_state(pool)
    initial_state = extract_initial_state(pool)

    start_str = _start_str_from_pool(pool)
    end_str = get_data_end_date(pool.tokens)
    lp_supply_df = load_bpt_supply_df(pool, end_date=end_str)

    print(f"Pool: {pool.label}, TVL: ${pool.initial_pool_value_usd:,.0f}")
    print(f"BPT: {lp_supply_df['lp_supply'].iloc[0]:.4f} -> {lp_supply_df['lp_supply'].iloc[-1]:.4f}")
    print(f"Period: {start_str} to {end_str}")

    # Run sim WITH lp_supply (gas=0, arb_freq=1)
    result_lp = run_sim(pool, gas_cost=0.0, arb_frequency=1,
                        initial_state=initial_state,
                        start=start_str, end=end_str,
                        lp_supply_df=lp_supply_df)

    # Run sim WITHOUT lp_supply (gas=0, arb_freq=1)
    result_no_lp = run_sim(pool, gas_cost=0.0, arb_frequency=1,
                           initial_state=initial_state,
                           start=start_str, end=end_str,
                           lp_supply_df=None)

    # World
    world = load_world_history(pool, end_date=end_str)
    world_ts = world["timestamps"]
    raw_bal_0 = world["raw_bal_0"]
    raw_bal_1 = world["raw_bal_1"]

    start_sec = result_lp["start_unix_sec"]
    prices_min = result_lp["prices"]

    # World value at world timestamps (raw balances × USD prices)
    prices_at_world = np.stack([
        sample_at_timestamps(prices_min[:, i], start_sec, world_ts)
        for i in range(prices_min.shape[1])
    ], axis=1)
    world_value = raw_bal_0 * prices_at_world[:, 0] + raw_bal_1 * prices_at_world[:, 1]

    # Sim values (minute-resolution)
    sim_value_lp = np.array(result_lp["value_usd"])
    sim_value_no_lp = np.array(result_no_lp["value_usd"])
    n_minutes = len(sim_value_lp)
    sim_times_sec = start_sec + np.arange(n_minutes) * 60
    sim_days = (sim_times_sec - start_sec) / 86400
    world_days = (world_ts - start_sec) / 86400

    # BPT supply at world timestamps (for annotation)
    lp_at_world = np.interp(
        world_ts,
        np.array(lp_supply_df["unix"]) / 1000,
        np.array(lp_supply_df["lp_supply"]),
    )

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: absolute pool value
    ax = axes[0]
    ax.plot(world_days, world_value, "k-", linewidth=2, label="World (raw balances × prices)")
    ax.plot(sim_days, sim_value_lp, "b-", linewidth=1, alpha=0.8, label="Sim (with lp_supply)")
    ax.plot(sim_days, sim_value_no_lp, "r--", linewidth=1, alpha=0.8, label="Sim (no lp_supply)")
    ax.set_ylabel("Pool value (USD)")
    ax.set_title(f"{pool.label} — gas=0, arb_freq=1min — absolute pool value")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: growth factors
    ax = axes[1]
    world_growth = world_value / world_value[0]
    sim_growth_lp = sim_value_lp / sim_value_lp[0]
    sim_growth_no_lp = sim_value_no_lp / sim_value_no_lp[0]
    ax.plot(world_days, world_growth, "k-", linewidth=2, label="World growth")
    ax.plot(sim_days, sim_growth_lp, "b-", linewidth=1, alpha=0.8, label="Sim growth (with lp_supply)")
    ax.plot(sim_days, sim_growth_no_lp, "r--", linewidth=1, alpha=0.8, label="Sim growth (no lp_supply)")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Growth factor")
    ax.set_title("Growth factors (value / initial value)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 3: BPT supply
    ax = axes[2]
    ax.plot(world_days, lp_at_world, "g-", linewidth=2, label="BPT supply (normalized)")
    ax.set_ylabel("BPT / BPT₀")
    ax.set_xlabel("Days from start")
    ax.set_title("On-chain BPT supply")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"{pool.label} ({pool.chain}) — TVL=${pool.initial_pool_value_usd:,.0f} — "
        f"PR={pool.on_chain_params['price_ratio']:.4f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    out = "results/diagnostic_lp_supply_cbBTC_WETH.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    # Print key numbers
    print(f"\nWorld: {world_value[0]:.0f} -> {world_value[-1]:.0f} (growth={world_growth[-1]:.4f})")
    print(f"Sim (lp):   {sim_value_lp[0]:.0f} -> {sim_value_lp[-1]:.0f} (growth={sim_growth_lp[-1]:.4f})")
    print(f"Sim (no lp): {sim_value_no_lp[0]:.0f} -> {sim_value_no_lp[-1]:.0f} (growth={sim_growth_no_lp[-1]:.4f})")
    print(f"\nDeviation (lp):    {(sim_growth_lp[-1]/world_growth[-1] - 1)*100:+.2f}%")
    print(f"Deviation (no lp): {(sim_growth_no_lp[-1]/world_growth[-1] - 1)*100:+.2f}%")


if __name__ == "__main__":
    main()
