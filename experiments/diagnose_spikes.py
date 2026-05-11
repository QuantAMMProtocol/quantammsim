"""Diagnose spikes in sim-vs-world deviation for LP-supply-normalized runs.

Compares old (no LP supply) vs new (with LP supply + per-LP normalization)
to pinpoint what causes spikes in the deviation time series.
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
from experiments.run_pool_battery import (
    run_sim, sample_at_timestamps, _start_str_from_pool,
    _onchain_params_to_sim, PROTOCOL_FEE_SPLIT,
)
from quantammsim.runners.jax_runners import do_run_on_historic_data


POOL_LABEL = "WAVAX_USDC"  # Change to "cbBTC_WETH" etc.


def main():
    pool = POOL_REGISTRY[POOL_LABEL]
    extract_on_chain_state(pool)
    initial_state = extract_initial_state(pool)
    start_str = _start_str_from_pool(pool)
    end_str = get_data_end_date(pool.tokens)
    lp_supply_df = load_bpt_supply_df(pool, end_date=end_str)

    start_sec = datetime.strptime(
        start_str, "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc).timestamp()

    print(f"Pool: {pool.label}, TVL: ${pool.initial_pool_value_usd:,.0f}")
    print(f"Period: {start_str} to {end_str}")
    print(f"BPT range: {lp_supply_df['lp_supply'].min():.4f} to {lp_supply_df['lp_supply'].max():.4f}")

    # ---- Run sims ----
    # 1. Old way: no lp_supply at all
    result_old = run_sim(
        pool, gas_cost=0.0, arb_frequency=1,
        initial_state=initial_state, start=start_str, end=end_str,
        lp_supply_df=None,
    )

    # 2. New way: lp_supply in scan + per-LP normalization
    result_new = run_sim(
        pool, gas_cost=0.0, arb_frequency=1,
        initial_state=initial_state, start=start_str, end=end_str,
        lp_supply_df=lp_supply_df,
    )

    # 3. Raw lp run (scan has lp_supply, but we DON'T divide by it)
    params = _onchain_params_to_sim(pool)
    fp = {
        "tokens": pool.tokens, "rule": "reclamm",
        "startDateString": start_str, "endDateString": end_str,
        "initial_pool_value": pool.initial_pool_value_usd,
        "fees": pool.swap_fee, "gas_cost": 0.0, "arb_fees": 0.0,
        "do_arb": True, "arb_frequency": 1, "chunk_period": 1440,
        "weight_interpolation_period": 1440,
        "reclamm_use_shift_exponent": True,
        "reclamm_interpolation_method": "geometric",
        "reclamm_centeredness_scaling": False,
        "protocol_fee_split": PROTOCOL_FEE_SPLIT,
        "reclamm_initial_state": initial_state,
    }
    result_raw = do_run_on_historic_data(
        run_fingerprint=fp, params=params, lp_supply_df=lp_supply_df,
    )
    v_lp_raw = np.array(result_raw["value"])

    v_old = np.array(result_old["value_usd"])  # no LP, no normalization
    v_new = np.array(result_new["value_usd"])  # LP scan + divided by lp_supply

    # ---- World ----
    world = load_world_history(pool, end_date=end_str)
    world_ts = world["timestamps"]
    prices_min = result_old["prices"]

    prices_at_world = np.stack([
        sample_at_timestamps(prices_min[:, i], start_sec, world_ts)
        for i in range(prices_min.shape[1])
    ], axis=1)

    # BPT-normalized world value (same in both old and new)
    world_bpt_val = (
        world["bal_0"] * prices_at_world[:, 0]
        + world["bal_1"] * prices_at_world[:, 1]
    )
    world_growth = world_bpt_val / world_bpt_val[0]

    # Raw world value (absolute, un-normalized)
    world_raw_val = (
        world["raw_bal_0"] * prices_at_world[:, 0]
        + world["raw_bal_1"] * prices_at_world[:, 1]
    )

    # Sample sim at world timestamps
    old_at_world = sample_at_timestamps(v_old, start_sec, world_ts)
    new_at_world = sample_at_timestamps(v_new, start_sec, world_ts)
    raw_at_world = sample_at_timestamps(v_lp_raw, start_sec, world_ts)

    old_growth = old_at_world / old_at_world[0]
    new_growth = new_at_world / new_at_world[0]
    raw_growth = raw_at_world / raw_at_world[0]

    # Deviations
    dev_old = (old_growth / world_growth - 1) * 100
    dev_new = (new_growth / world_growth - 1) * 100

    # Raw vs raw-world comparison (both absolute)
    world_raw_growth = world_raw_val / world_raw_val[0]
    dev_raw = (raw_growth / world_raw_growth - 1) * 100

    days = (world_ts - world_ts[0]) / 86400

    # LP supply at world timestamps
    lp_unix = np.array(lp_supply_df["unix"])
    lp_vals = np.array(lp_supply_df["lp_supply"])
    lp_at_world = np.interp(world_ts, lp_unix / 1000, lp_vals)

    # ---- Print diagnostics ----
    print(f"\n--- Final deviations ---")
    print(f"Old (no LP):        {dev_old[-1]:+.4f}%")
    print(f"New (LP + per-LP):  {dev_new[-1]:+.4f}%")
    print(f"Raw (LP, absolute): {dev_raw[-1]:+.4f}%")

    # Spike analysis
    for label, dev in [("Old", dev_old), ("New", dev_new), ("Raw", dev_raw)]:
        diffs = np.abs(np.diff(dev))
        n_spikes_01 = np.sum(diffs > 0.1)
        n_spikes_05 = np.sum(diffs > 0.5)
        n_spikes_10 = np.sum(diffs > 1.0)
        print(f"\n{label} — step-to-step jumps in deviation:")
        print(f"  >0.1%: {n_spikes_01}, >0.5%: {n_spikes_05}, >1.0%: {n_spikes_10}")
        if n_spikes_10 > 0:
            spike_idx = np.where(diffs > 1.0)[0]
            for si in spike_idx[:5]:
                print(f"    day {days[si]:.1f}: dev {dev[si]:+.2f}% -> {dev[si+1]:+.2f}% "
                      f"(Δ={dev[si+1]-dev[si]:+.2f}%, lp={lp_at_world[si]:.4f}->{lp_at_world[si+1]:.4f})")

    # World growth spikes
    world_g_diffs = np.diff(world_growth)
    n_world_spikes = np.sum(np.abs(world_g_diffs) > 0.01)
    print(f"\nWorld BPT-normalized growth jumps > 1%: {n_world_spikes}")
    if n_world_spikes > 0:
        wsi = np.where(np.abs(world_g_diffs) > 0.01)[0]
        for si in wsi[:5]:
            print(f"  day {days[si]:.1f}: growth {world_growth[si]:.4f} -> {world_growth[si+1]:.4f} "
                  f"(Δ={world_g_diffs[si]:+.4f}, lp={lp_at_world[si]:.4f}->{lp_at_world[si+1]:.4f})")

    # ---- Plot ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    ax = axes[0]
    ax.plot(days, dev_old, "b-", linewidth=1.5, label=f"Old (no LP) → {dev_old[-1]:+.2f}%")
    ax.plot(days, dev_new, "r-", linewidth=1.5, label=f"New (LP + per-LP norm) → {dev_new[-1]:+.2f}%")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("% deviation from world")
    ax.set_title(f"{pool.label} — gas=0, arb=1min — old vs new deviation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(days, dev_raw, "g-", linewidth=1.5, label=f"Raw absolute (LP scan, raw world) → {dev_raw[-1]:+.2f}%")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("% deviation")
    ax.set_title("Alternative: raw absolute sim vs raw absolute world")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    ax.plot(days, world_growth, "k-", linewidth=2, label="World (BPT-normalized)")
    ax.plot(days, old_growth, "b-", linewidth=1, alpha=0.8, label="Old sim")
    ax.plot(days, new_growth, "r-", linewidth=1, alpha=0.8, label="New sim (LP + per-LP)")
    ax.set_ylabel("Growth factor")
    ax.set_title("Growth factors")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax = axes[3]
    ax.plot(days, lp_at_world, "g-", linewidth=2, label="LP supply (BPT/BPT₀)")
    # Mark large LP changes
    lp_diffs = np.abs(np.diff(lp_at_world))
    big_lp = np.where(lp_diffs > 0.05)[0]
    if len(big_lp):
        ax.scatter(days[big_lp], lp_at_world[big_lp], c="red", s=40, zorder=5,
                   label=f"Large LP events ({len(big_lp)})")
    ax.set_ylabel("BPT / BPT₀")
    ax.set_xlabel("Days from start")
    ax.set_title("On-chain BPT supply")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"{pool.label} ({pool.chain}) — spike diagnosis",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    out = f"results/diagnose_spikes_{POOL_LABEL}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
