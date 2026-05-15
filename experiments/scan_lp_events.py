"""Scan all pools for large LP deposit/withdrawal events and estimate TVL→noise elasticity.

Identifies "semi-exogenous" LP flow events — large share changes that represent
genuine deposit/withdrawal decisions, not pool creation or dust.

Filters:
  - |Δlog(shares)| > threshold (default 20%)
  - Pool must have been active for at least --min-age days before the event
  - Pre-event TVL must be above --min-tvl (filters out pool creation events
    where initial TVL is dust)
  - Enough pre/post data to estimate volume change

For each event, computes the volume response and implied elasticity.

Usage:
  python experiments/scan_lp_events.py
  python experiments/scan_lp_events.py --threshold 0.1 --window 7
  python experiments/scan_lp_events.py --use-api   # fetch fresh snapshots
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_panel_data(use_api=False):
    """Load pool panel data from calibration cache or API."""
    import pickle

    pools = {}

    # Stage1 calibration pools
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "token_factored_calibration", "_cache", "stage1.pkl",
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        for pid, entry in data["matched_clean"].items():
            panel = entry["panel"].copy()
            panel["pool_id"] = pid
            panel["chain"] = entry["chain"]
            panel["tokens"] = entry["tokens"]
            pools[pid] = panel

    # Noise calibration panel (broader set)
    noise_panel_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "local_data", "noise_calibration", "panel.parquet",
    )
    if os.path.exists(noise_panel_path):
        panel_all = pd.read_parquet(noise_panel_path)
        for pid in panel_all["pool_id"].unique():
            if pid[:16] not in pools:  # don't duplicate
                pp = panel_all[panel_all["pool_id"] == pid].copy()
                if len(pp) >= 30:
                    pools[pid[:16]] = pp

    # Top50 snapshots (even broader)
    snap_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "local_data", "noise_top50", "snapshots",
    )
    if os.path.exists(snap_dir):
        import glob
        for f in glob.glob(os.path.join(snap_dir, "*.parquet")):
            pid = os.path.basename(f).replace(".parquet", "")
            if pid[:16] not in pools:
                try:
                    df = pd.read_parquet(f)
                    if len(df) >= 30 and "total_shares" in df.columns:
                        df["pool_id"] = pid
                        pools[pid[:16]] = df
                except Exception:
                    pass

    if use_api:
        print("  Fetching fresh snapshots from Balancer API...")
        from quantammsim.noise_calibration import (
            fetch_pool_snapshots, BALANCER_API_CHAINS,
        )
        for pid_short, panel in list(pools.items()):
            if "chain" in panel.columns:
                chain = panel["chain"].iloc[0]
            else:
                chain = "MAINNET"
            full_pid = panel["pool_id"].iloc[0] if "pool_id" in panel.columns else pid_short
            try:
                fresh = fetch_pool_snapshots(full_pid, chain)
                if len(fresh) > len(panel):
                    fresh["pool_id"] = full_pid
                    fresh["chain"] = chain
                    if "tokens" in panel.columns:
                        fresh["tokens"] = panel["tokens"].iloc[0]
                    pools[pid_short] = fresh
                time.sleep(0.3)
            except Exception:
                pass

    print(f"  Loaded {len(pools)} pools")
    return pools


def find_lp_events(panel, threshold=0.2, min_age_days=30, min_tvl=10_000):
    """Find large LP deposit/withdrawal events in a single pool's panel.

    Returns list of event dicts.
    """
    dates = pd.to_datetime(panel["date"])

    # Need shares and TVL
    if "total_shares" not in panel.columns:
        return []
    shares = panel["total_shares"].values.astype(float)
    if np.all(shares <= 0) or np.all(np.isnan(shares)):
        return []

    # TVL
    if "total_liquidity_usd" in panel.columns:
        tvl = panel["total_liquidity_usd"].values.astype(float)
    elif "log_tvl" in panel.columns:
        tvl = np.exp(panel["log_tvl"].values.astype(float))
    elif "log_tvl_lag1" in panel.columns:
        tvl = np.exp(panel["log_tvl_lag1"].values.astype(float))
    else:
        return []

    # Volume
    if "volume_usd" in panel.columns:
        vol = panel["volume_usd"].values.astype(float)
    elif "log_volume" in panel.columns:
        vol = np.exp(panel["log_volume"].values.astype(float))
    else:
        return []

    log_shares = np.log(np.maximum(shares, 1e-10))
    d_log_shares = np.diff(log_shares)

    events = []
    for i in range(len(d_log_shares)):
        if abs(d_log_shares[i]) < np.log(1 + threshold):
            continue

        # Check min age: pool must have been active for min_age_days
        days_active = (dates.iloc[i + 1] - dates.iloc[0]).days
        if days_active < min_age_days:
            continue

        # Check min TVL before event
        if tvl[i] < min_tvl:
            continue

        # Check shares aren't near-zero before (not pool creation)
        if shares[i] < 1:
            continue

        pct_change = (np.exp(d_log_shares[i]) - 1) * 100
        event_type = "deposit" if d_log_shares[i] > 0 else "withdrawal"

        events.append({
            "date": dates.iloc[i + 1],
            "idx": i + 1,
            "type": event_type,
            "d_log_shares": float(d_log_shares[i]),
            "pct_change": float(pct_change),
            "shares_before": float(shares[i]),
            "shares_after": float(shares[i + 1]),
            "tvl_before": float(tvl[i]),
            "tvl_after": float(tvl[i + 1]),
            "vol_on_day": float(vol[i + 1]),
        })

    return events


def compute_event_elasticity(panel, event, window=7):
    """Compute volume response around an LP event.

    Compares median volume in [event-window, event) vs [event+1, event+window+1).
    """
    dates = pd.to_datetime(panel["date"])
    idx = event["idx"]

    if "volume_usd" in panel.columns:
        vol = panel["volume_usd"].values.astype(float)
    elif "log_volume" in panel.columns:
        vol = np.exp(panel["log_volume"].values.astype(float))
    else:
        return None

    if "total_liquidity_usd" in panel.columns:
        tvl = panel["total_liquidity_usd"].values.astype(float)
    elif "log_tvl" in panel.columns:
        tvl = np.exp(panel["log_tvl"].values.astype(float))
    elif "log_tvl_lag1" in panel.columns:
        tvl = np.exp(panel["log_tvl_lag1"].values.astype(float))
    else:
        return None

    pre_start = max(0, idx - window)
    post_end = min(len(vol), idx + 1 + window)

    if idx - pre_start < 3 or post_end - (idx + 1) < 3:
        return None

    vol_pre = np.median(vol[pre_start:idx])
    vol_post = np.median(vol[idx + 1:post_end])
    tvl_pre = np.median(tvl[pre_start:idx])
    tvl_post = np.median(tvl[idx + 1:post_end])

    if vol_pre <= 0 or tvl_pre <= 0 or tvl_post <= 0:
        return None

    vol_ratio = vol_post / vol_pre
    tvl_ratio = tvl_post / tvl_pre

    if abs(np.log(tvl_ratio)) < 0.05:  # TVL didn't actually change much
        return None

    elasticity = np.log(vol_ratio) / np.log(tvl_ratio)

    return {
        "vol_pre": vol_pre,
        "vol_post": vol_post,
        "tvl_pre": tvl_pre,
        "tvl_post": tvl_post,
        "vol_ratio": vol_ratio,
        "tvl_ratio": tvl_ratio,
        "elasticity": elasticity,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Min |share change| to count as event (0.2 = 20%%)")
    parser.add_argument("--window", type=int, default=7,
                        help="Days before/after event for volume comparison")
    parser.add_argument("--min-age", type=int, default=30,
                        help="Min days pool must be active before event")
    parser.add_argument("--min-tvl", type=float, default=10_000,
                        help="Min TVL before event (filters pool creation)")
    parser.add_argument("--use-api", action="store_true",
                        help="Fetch fresh snapshots from Balancer API")
    parser.add_argument("--output-dir", default="results/lp_events",
                        help="Output directory for CSV and plots")
    args = parser.parse_args()

    print("=" * 70)
    print("LP Event Scanner: Semi-Exogenous TVL Shocks")
    print(f"  threshold={args.threshold:.0%}, window={args.window}d,"
          f" min_age={args.min_age}d, min_tvl=${args.min_tvl:,.0f}")
    print("=" * 70)

    pools = load_panel_data(use_api=args.use_api)

    all_events = []
    print(f"\nScanning {len(pools)} pools for LP events...")

    for pid_short, panel in pools.items():
        tokens = (panel["tokens"].iloc[0] if "tokens" in panel.columns
                  else "?")
        chain = (panel["chain"].iloc[0] if "chain" in panel.columns
                 else "?")

        events = find_lp_events(
            panel, threshold=args.threshold,
            min_age_days=args.min_age, min_tvl=args.min_tvl)

        for ev in events:
            result = compute_event_elasticity(panel, ev, window=args.window)
            ev["pool_id"] = pid_short
            ev["tokens"] = tokens
            ev["chain"] = chain
            ev["result"] = result
            all_events.append(ev)

    # Sort by absolute share change
    all_events.sort(key=lambda e: abs(e["d_log_shares"]), reverse=True)

    print(f"\nFound {len(all_events)} LP events across {len(pools)} pools")
    events_with_elasticity = [e for e in all_events if e["result"] is not None]
    print(f"  {len(events_with_elasticity)} with computable elasticity")

    # Print event table
    print(f"\n{'Date':12s} {'Pool':16s} {'Tokens':18s} {'Type':10s}"
          f" {'Δshares':>8s} {'TVL before':>12s} {'TVL after':>12s}"
          f" {'VolPre':>10s} {'VolPost':>10s} {'Elast':>7s}")
    print("-" * 120)

    for ev in all_events:
        r = ev["result"]
        if r:
            elast_str = f"{r['elasticity']:+7.2f}"
            vol_pre_str = f"${r['vol_pre']:>9,.0f}"
            vol_post_str = f"${r['vol_post']:>9,.0f}"
        else:
            elast_str = "    n/a"
            vol_pre_str = "       n/a"
            vol_post_str = "       n/a"

        print(f"{str(ev['date'].date()):12s} {ev['pool_id'][:16]:16s}"
              f" {str(ev['tokens'])[:18]:18s} {ev['type']:10s}"
              f" {ev['pct_change']:+7.0f}%"
              f" ${ev['tvl_before']:>11,.0f} ${ev['tvl_after']:>11,.0f}"
              f" {vol_pre_str} {vol_post_str} {elast_str}")

    # Summary statistics
    if not events_with_elasticity:
        print("No events with computable elasticity.")
        return

    deposits = [e for e in events_with_elasticity if e["type"] == "deposit"]
    withdrawals = [e for e in events_with_elasticity if e["type"] == "withdrawal"]

    all_elast = [e["result"]["elasticity"] for e in events_with_elasticity]
    dep_elast = [e["result"]["elasticity"] for e in deposits]
    wth_elast = [e["result"]["elasticity"] for e in withdrawals]
    clean = [e for e in events_with_elasticity
             if -1 < e["result"]["elasticity"] < 5]
    clean_elast = [e["result"]["elasticity"] for e in clean]

    print(f"\n{'='*70}")
    print("Summary: Implied TVL→Volume Elasticity")
    print(f"{'='*70}")
    print(f"  All events ({len(all_elast)}):"
          f"  median={np.median(all_elast):+.2f}"
          f"  mean={np.mean(all_elast):+.2f}"
          f"  std={np.std(all_elast):.2f}")
    if dep_elast:
        print(f"  Deposits ({len(dep_elast)}):"
              f"   median={np.median(dep_elast):+.2f}"
              f"  mean={np.mean(dep_elast):+.2f}")
    if wth_elast:
        print(f"  Withdrawals ({len(wth_elast)}):"
              f" median={np.median(wth_elast):+.2f}"
              f"  mean={np.mean(wth_elast):+.2f}")
    if clean_elast:
        print(f"\n  Clean events (elasticity in [-1, 5], n={len(clean_elast)}):")
        print(f"    median={np.median(clean_elast):+.2f}"
              f"  mean={np.mean(clean_elast):+.2f}"
              f"  [Q25={np.percentile(clean_elast, 25):+.2f},"
              f" Q75={np.percentile(clean_elast, 75):+.2f}]")

    print(f"\n  For comparison:")
    print(f"    Per-pool observational b_tvl:  ~1.0")
    print(f"    Shared observational b_tvl:    ~2.5")
    print(f"    Daily Δ within-pool:           ~0.1")

    # ---- Save CSV ----
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for ev in all_events:
        r = ev.get("result") or {}
        rows.append({
            "date": ev["date"],
            "pool_id": ev["pool_id"],
            "tokens": str(ev["tokens"]),
            "chain": str(ev["chain"]),
            "type": ev["type"],
            "pct_change": ev["pct_change"],
            "tvl_before": ev["tvl_before"],
            "tvl_after": ev["tvl_after"],
            "shares_before": ev["shares_before"],
            "shares_after": ev["shares_after"],
            "vol_pre": r.get("vol_pre"),
            "vol_post": r.get("vol_post"),
            "tvl_ratio": r.get("tvl_ratio"),
            "vol_ratio": r.get("vol_ratio"),
            "elasticity": r.get("elasticity"),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "lp_events.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path} ({len(df)} events)")

    # ---- Plots ----
    # 1. Elasticity histogram (clean events, deposits vs withdrawals)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(clean_elast, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(np.median(clean_elast), color="red", linestyle="--", linewidth=2,
               label=f"median={np.median(clean_elast):+.2f}")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5, label="elasticity=1")
    ax.set_xlabel("Elasticity (Δlog vol / Δlog TVL)")
    ax.set_ylabel("Count")
    ax.set_title(f"All clean events (n={len(clean_elast)})")
    ax.legend(fontsize=8)

    ax = axes[1]
    dep_clean = [e["result"]["elasticity"] for e in clean if e["type"] == "deposit"]
    wth_clean = [e["result"]["elasticity"] for e in clean if e["type"] == "withdrawal"]
    ax.hist(dep_clean, bins=30, color="green", alpha=0.6, label=f"deposits (n={len(dep_clean)})", edgecolor="white")
    ax.hist(wth_clean, bins=30, color="coral", alpha=0.6, label=f"withdrawals (n={len(wth_clean)})", edgecolor="white")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Elasticity")
    ax.set_title("Deposits vs Withdrawals")
    ax.legend(fontsize=8)

    # 2. Elasticity vs event size (|Δlog shares|)
    ax = axes[2]
    sizes = [abs(e["d_log_shares"]) for e in clean]
    elasts = [e["result"]["elasticity"] for e in clean]
    colors = ["green" if e["type"] == "deposit" else "coral" for e in clean]
    ax.scatter(sizes, elasts, c=colors, alpha=0.4, s=15, edgecolors="none")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(np.median(elasts), color="red", linestyle="--", alpha=0.7,
               label=f"median={np.median(elasts):+.2f}")
    ax.set_xlabel("|Δlog(shares)| (event size)")
    ax.set_ylabel("Elasticity")
    ax.set_title("Elasticity vs Event Size")
    ax.legend(fontsize=8)

    fig.suptitle(f"LP Event Elasticity Analysis — {len(clean)} clean events"
                 f" from {len(pools)} pools", fontsize=11)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "elasticity_histograms.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # 3. Elasticity vs pre-event TVL (does pool size affect elasticity?)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    tvl_pre = [e["result"]["tvl_pre"] for e in clean]
    ax.scatter(tvl_pre, elasts, c=colors, alpha=0.4, s=15, edgecolors="none")
    ax.set_xscale("log")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Pre-event TVL (USD)")
    ax.set_ylabel("Elasticity")
    ax.set_title("Elasticity vs Pool Size")

    # Bin by TVL decile and show median elasticity
    tvl_arr = np.array(tvl_pre)
    el_arr = np.array(elasts)
    for q_lo, q_hi in [(0, 25), (25, 50), (50, 75), (75, 100)]:
        lo = np.percentile(tvl_arr, q_lo)
        hi = np.percentile(tvl_arr, q_hi)
        mask = (tvl_arr >= lo) & (tvl_arr < hi + 1)
        if mask.sum() > 5:
            med_tvl = np.median(tvl_arr[mask])
            med_el = np.median(el_arr[mask])
            ax.plot(med_tvl, med_el, "rs", markersize=10, zorder=5)
            ax.annotate(f"{med_el:.2f}", (med_tvl, med_el),
                        textcoords="offset points", xytext=(8, 5), fontsize=7)

    # 4. log(vol_post/vol_pre) vs log(tvl_post/tvl_pre) scatter
    ax = axes[1]
    log_tvl_ratio = [np.log(e["result"]["tvl_ratio"]) for e in clean]
    log_vol_ratio = [np.log(e["result"]["vol_ratio"]) for e in clean]
    ax.scatter(log_tvl_ratio, log_vol_ratio, c=colors, alpha=0.4, s=15,
               edgecolors="none")

    # OLS fit line
    x_fit = np.array(log_tvl_ratio)
    y_fit = np.array(log_vol_ratio)
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
            label=f"OLS slope={slope:.2f}")
    ax.plot(x_line, x_line, "k--", alpha=0.3, label="1:1 line")
    ax.set_xlabel("Δlog(TVL)")
    ax.set_ylabel("Δlog(Volume)")
    ax.set_title("Volume Response to TVL Shocks")
    ax.legend(fontsize=8)

    fig.suptitle("TVL→Volume Elasticity: Event Study", fontsize=11)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "elasticity_scatter.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # 5. Elasticity by chain
    fig, ax = plt.subplots(figsize=(10, 5))
    chain_data = {}
    for e in clean:
        ch = str(e["chain"])
        if ch not in chain_data:
            chain_data[ch] = []
        chain_data[ch].append(e["result"]["elasticity"])
    chains_sorted = sorted(chain_data.keys(),
                           key=lambda c: len(chain_data[c]), reverse=True)
    chains_plot = [c for c in chains_sorted if len(chain_data[c]) >= 5]
    if chains_plot:
        positions = range(len(chains_plot))
        bp = ax.boxplot([chain_data[c] for c in chains_plot],
                        positions=positions, widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{c}\n(n={len(chain_data[c])})" for c in chains_plot],
                           fontsize=8)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(np.median(clean_elast), color="red", linestyle="--", alpha=0.5,
                   label=f"overall median={np.median(clean_elast):.2f}")
        ax.set_ylabel("Elasticity")
        ax.set_title("Elasticity by Chain")
        ax.set_ylim(-2, 5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "elasticity_by_chain.png")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3}")


if __name__ == "__main__":
    main()
