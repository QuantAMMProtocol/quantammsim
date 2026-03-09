"""Plot predicted vs real volume for top 50 pools by TVL on Feb 1st 2026.

Enumerates WEIGHTED (min_tvl=1000) and RECLAMM (min_tvl=0) pools,
fetches their snapshots, filters to those with TVL >= $10k on Feb 1st 2026,
takes the top 50 by TVL, and plots predicted vs actual daily volume using
the inference artifact from calibrate_noise_unified.py.

For pools that were in the model's training set, uses their per-pool theta.
For pools not in the training set, uses population-level prediction from B.
"""

import ast
import json
import os
import sys
import time
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse functions from the noise calibration package
from quantammsim.noise_calibration import (
    BALANCER_API_CHAINS,
    _graphql_request,
    assemble_panel,
    classify_token_tier,
    encode_covariates,
    fetch_pool_snapshots,
    fetch_token_prices,
)


CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "local_data", "noise_top50"
)
TVL_DATE = date(2026, 2, 1)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "top50_feb1"
)
# Inference artifact from the main unified model run
FITTED_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "unified_full_90d.json"
)


def enumerate_all_pools():
    """Enumerate WEIGHTED (min_tvl=1000) and RECLAMM (min_tvl=0) pools."""
    all_pools = []

    for chain in BALANCER_API_CHAINS:
        for pool_type, min_tvl in [("WEIGHTED", 1000), ("RECLAMM", 0)]:
            query = {
                "query": """
                query GetPools($chain: GqlChain!, $types: [GqlPoolType!],
                               $minTvl: Float) {
                  poolGetPools(
                    where: { chainIn: [$chain], poolTypeIn: $types, minTvl: $minTvl }
                  ) {
                    id chain type protocolVersion
                    poolTokens { symbol weight address }
                    dynamicData { totalLiquidity swapFee }
                  }
                }
                """,
                "variables": {
                    "chain": chain,
                    "types": [pool_type],
                    "minTvl": min_tvl,
                },
            }

            try:
                body = _graphql_request(query)
                pools = body.get("data", {}).get("poolGetPools", [])
            except Exception as e:
                print(f"  FAILED {chain} {pool_type}: {e}")
                continue

            for p in pools:
                tokens = [t["symbol"] for t in p.get("poolTokens", [])]
                addresses = [t.get("address", "") for t in p.get("poolTokens", [])]
                tvl = float(p.get("dynamicData", {}).get("totalLiquidity", 0))
                fee = float(p.get("dynamicData", {}).get("swapFee", 0))
                all_pools.append({
                    "pool_id": p["id"],
                    "chain": p["chain"],
                    "pool_type": p["type"],
                    "tokens": tokens,
                    "token_addresses": addresses,
                    "swap_fee": fee,
                    "current_tvl": tvl,
                })

            if pools:
                print(f"  {chain:>10} {pool_type:>10}: {len(pools)}")
            time.sleep(0.3)

    df = pd.DataFrame(all_pools)
    print(f"\n  Total: {len(df)} pools")
    return df


def fetch_all_snapshots_cached(pools_df, cache_dir):
    """Fetch snapshots for all pools, caching per-pool."""
    snap_dir = os.path.join(cache_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    all_snaps = []
    n = len(pools_df)

    for i, (_, pool) in enumerate(pools_df.iterrows()):
        pid = pool["pool_id"]
        chain = pool["chain"]
        cache_file = os.path.join(snap_dir, f"{pid}.parquet")

        if os.path.exists(cache_file):
            df = pd.read_parquet(cache_file)
        else:
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    Fetching snapshots {i+1}/{n}...", flush=True)
            try:
                df = fetch_pool_snapshots(pid, chain)
                if len(df) > 0:
                    df.to_parquet(cache_file, index=False)
                time.sleep(0.3)
            except Exception as e:
                print(f"    FAILED {pid[:20]}: {e}")
                continue

        if len(df) > 0:
            df["pool_id"] = pid
            df["chain"] = chain
            all_snaps.append(df)

    if all_snaps:
        return pd.concat(all_snaps, ignore_index=True)
    return pd.DataFrame()


def get_tvl_on_date(snapshots_df, target_date, window_days=3):
    """Get TVL for each pool on/near target_date."""
    results = []
    for pid in snapshots_df["pool_id"].unique():
        pool_snaps = snapshots_df[snapshots_df["pool_id"] == pid]

        best_row = None
        best_dist = float("inf")
        for _, row in pool_snaps.iterrows():
            d = row["date"]
            if isinstance(d, date):
                dist = abs((d - target_date).days)
            else:
                dist = abs((pd.Timestamp(d).date() - target_date).days)
            if dist < best_dist:
                best_dist = dist
                best_row = row

        if best_row is not None and best_dist <= window_days:
            results.append({
                "pool_id": pid,
                "tvl_feb1": float(best_row["total_liquidity_usd"]),
                "date_used": best_row["date"],
            })

    return pd.DataFrame(results)


def _get_theta_for_pool(pid, fitted, panel_90d, pop_B, pop_cov_names):
    """Get theta for a pool: from fitted artifact if available, else population.

    Returns (theta, source) where source is 'fitted' or 'population'.
    For IBP models, population fallback adds marginal feature effect (pi @ W).
    """
    if pid in fitted["pools"]:
        return np.array(fitted["pools"][pid]["theta_median"]), "fitted"

    # Population-level prediction: theta = B @ z_pool
    # Build z_pool from the pool's covariates
    pp = panel_90d[panel_90d["pool_id"] == pid]
    if len(pp) == 0:
        return None, "no_data"

    chain = pp["chain"].iloc[0]
    tokens = pp["tokens"].iloc[0]
    if isinstance(tokens, str):
        tokens = tokens.split(",")
    fee = pp["swap_fee"].iloc[0] if "swap_fee" in pp.columns else 0.003
    tiers = sorted([classify_token_tier(t) for t in tokens])
    tier_a = tiers[0]

    # Build covariate vector matching the model's encoding
    z = np.zeros(len(pop_cov_names))
    for i, name in enumerate(pop_cov_names):
        if name == "intercept":
            z[i] = 1.0
        elif name == f"chain_{chain}":
            z[i] = 1.0
        elif name == f"tier_A_{tier_a}":
            z[i] = 1.0
        elif name == "log_fee":
            z[i] = np.log(max(fee, 1e-6))

    # B is (K_coeff, K_cov), theta = B @ z
    B = np.array(pop_B)  # (K_coeff, K_cov)
    theta = B @ z

    # IBP: add marginal feature effect (pi @ W)
    pop = fitted["population_effects"]
    if "W" in pop and "feature_prevalences" in pop:
        W = np.array(pop["W"])                        # (K_features, K_coeff)
        pi = np.array(pop["feature_prevalences"])     # (K_features,)
        theta = theta + pi @ W

    return theta, "population"


def plot_pages(plot_pools, pool_idx_map_fitted, fitted, panel_90d,
               pools_df, tvl_lookup, pop_B, pop_cov_names,
               output_dir=OUTPUT_DIR):
    """Generate paginated plots, 10 pools per page."""
    n_pools = len(plot_pools)
    per_page = 10
    n_pages = (n_pools + per_page - 1) // per_page

    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, n_pools)
        page_pools = plot_pools[start:end]
        n_this = len(page_pools)

        ncols = 2
        nrows = (n_this + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)

        for idx, (pid, feb_tvl) in enumerate(page_pools):
            ax = axes[idx // ncols][idx % ncols]

            theta, source = _get_theta_for_pool(
                pid, fitted, panel_90d, pop_B, pop_cov_names
            )
            if theta is None:
                ax.set_visible(False)
                continue

            pp = panel_90d[panel_90d["pool_id"] == pid].sort_values("date")
            if len(pp) < 5:
                ax.set_visible(False)
                continue

            x_obs = np.column_stack([
                np.ones(len(pp)),
                pp["log_tvl_lag1"].values,
                pp["volatility"].values,
                pp["weekend"].values,
            ])

            pred_log = x_obs @ theta
            actual_log = pp["log_volume"].values
            pred_vol = np.exp(pred_log)
            actual_vol = np.exp(actual_log)
            dates = pd.to_datetime(pp["date"].values)

            ax.plot(dates, actual_vol, "o-", color="steelblue", markersize=2.5,
                    linewidth=0.9, alpha=0.7, label="Actual")
            ax.plot(dates, pred_vol, "s--", color="orangered", markersize=2.5,
                    linewidth=0.9, alpha=0.7, label="Predicted")
            ax.set_yscale("log")
            ax.set_ylabel("Daily volume (USD)", fontsize=8)

            ss_res = np.sum((actual_log - pred_log) ** 2)
            ss_tot = np.sum((actual_log - actual_log.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            meta = pools_df[pools_df["pool_id"] == pid]
            if len(meta) > 0:
                m = meta.iloc[0]
                tokens = m["tokens"]
                tok_str = "/".join(str(t)[:8] for t in tokens[:2])
                chain = str(m["chain"])
                ptype = str(m["pool_type"])
            else:
                tok_str = pid[:16]
                chain = "?"
                ptype = "?"

            type_tag = "R" if ptype == "RECLAMM" else "W"
            src_tag = "*" if source == "population" else ""
            ax.set_title(
                "{} ({}, {}){}\n"
                "TVL ${:,.0f} on Feb 1  |  "
                "R\u00b2={:.3f}  b_c={:.2f}  b_\u03c3={:.2f}  "
                "b_wknd={:.2f}  n={}".format(
                    tok_str, chain, type_tag, src_tag, feb_tvl,
                    r2, theta[1], theta[2], theta[3], len(pp)),
                fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="x", rotation=30)

        for idx in range(n_this, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            "Predicted vs actual daily volume \u2014 page {}/{} "
            "(sorted by TVL on {})  [* = population prediction]".format(
                page + 1, n_pages, TVL_DATE),
            fontsize=11)
        fig.tight_layout()
        out = os.path.join(output_dir, "pred_vs_real_page{}.png".format(page + 1))
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: {}".format(out))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=FITTED_JSON,
                        help="Path to inference artifact JSON")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: auto from model name)")
    args = parser.parse_args()

    artifact_path = args.artifact
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---- Load inference artifact ----
    print(f"Loading inference artifact: {artifact_path}")
    with open(artifact_path) as f:
        fitted = json.load(f)
    n_fitted = len(fitted["pools"])
    model_name = fitted.get("model", "unknown")
    print(f"  Model: {model_name}")
    print(f"  {n_fitted} pools with fitted theta")

    # Output dir: use CLI override or auto from model name
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", f"top50_feb1_{model_name}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Extract population-level B matrix for pools not in the model
    pop_cov_names = fitted["model_spec"]["covariate_names"]
    pop_B = np.array(fitted["population_effects"]["B"])  # (K_coeff, K_cov)
    print(f"  Population B: {pop_B.shape}, covariates: {pop_cov_names}")

    # ---- Step 1: Enumerate pools ----
    pools_cache = os.path.join(CACHE_DIR, "pools.parquet")
    if os.path.exists(pools_cache):
        pools_df = pd.read_parquet(pools_cache)
        if isinstance(pools_df["tokens"].iloc[0], str):
            pools_df["tokens"] = pools_df["tokens"].apply(ast.literal_eval)
            pools_df["token_addresses"] = pools_df["token_addresses"].apply(
                ast.literal_eval
            )
        print(f"\nLoaded {len(pools_df)} pools from cache")
    else:
        print("\n1. Enumerating pools...")
        pools_df = enumerate_all_pools()
        pools_df.to_parquet(pools_cache, index=False)

    # ---- Step 2: Fetch snapshots ----
    print("\n2. Fetching snapshots...")
    snapshots_df = fetch_all_snapshots_cached(pools_df, CACHE_DIR)
    print(f"   {len(snapshots_df)} pool-days")

    # ---- Step 3: TVL on Feb 1st ----
    print(f"\n3. Finding TVL on {TVL_DATE}...")
    tvl_df = get_tvl_on_date(snapshots_df, TVL_DATE)
    tvl_df = tvl_df[tvl_df["tvl_feb1"] >= 10_000].copy()
    tvl_df = tvl_df.sort_values("tvl_feb1", ascending=False).head(50)
    top50_ids = set(tvl_df["pool_id"])
    tvl_lookup = dict(zip(tvl_df["pool_id"], tvl_df["tvl_feb1"]))
    print(f"   {len(tvl_df)} pools with TVL >= $10k")

    # How many are in the fitted model?
    n_in_model = sum(1 for pid in top50_ids if pid in fitted["pools"])
    print(f"   {n_in_model} in fitted model, "
          f"{len(top50_ids) - n_in_model} will use population prediction")

    # ---- Step 4: Fetch token prices & assemble panel ----
    panel_cache = os.path.join(CACHE_DIR, "panel.parquet")
    if os.path.exists(panel_cache):
        panel = pd.read_parquet(panel_cache)
        print(f"\n4. Loaded panel from cache: {len(panel)} obs")
    else:
        top50_pools = pools_df[pools_df["pool_id"].isin(top50_ids)].copy()
        top50_snaps = snapshots_df[snapshots_df["pool_id"].isin(top50_ids)].copy()

        print("\n4. Fetching token prices...")
        prices_cache = os.path.join(CACHE_DIR, "token_prices")
        token_addr_by_chain = {}
        for _, pool in top50_pools.iterrows():
            chain = pool["chain"]
            tokens = pool["tokens"]
            addresses = pool["token_addresses"]
            if chain not in token_addr_by_chain:
                token_addr_by_chain[chain] = {}
            for sym, addr in zip(tokens, addresses):
                if sym and addr:
                    token_addr_by_chain[chain][sym] = addr

        token_prices = fetch_token_prices(
            token_addr_by_chain, cache_dir=prices_cache
        )

        print("\n   Assembling panel...")
        panel = assemble_panel(top50_pools, top50_snaps, token_prices)
        panel.to_parquet(panel_cache, index=False)

    # ---- Step 5: Filter to 90 days ----
    max_date = panel["date"].max()
    if not isinstance(max_date, date):
        max_date = pd.Timestamp(max_date).date()
    cutoff = max_date - timedelta(days=90)
    panel_90d = panel[
        panel["date"].apply(
            lambda d: d >= cutoff if isinstance(d, date)
            else pd.Timestamp(d).date() >= cutoff
        )
    ].copy()

    if "log_tvl_lag1" not in panel_90d.columns:
        panel_90d = panel_90d.sort_values(["pool_id", "date"]).reset_index(drop=True)
        panel_90d["log_tvl_lag1"] = panel_90d.groupby("pool_id")["log_tvl"].shift(1)
        panel_90d = panel_90d.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)

    pool_counts = panel_90d.groupby("pool_id").size()
    valid_pools = pool_counts[pool_counts >= 10].index
    panel_90d = panel_90d[panel_90d["pool_id"].isin(valid_pools)].copy()
    print(f"\n5. 90-day panel: {len(panel_90d)} obs, "
          f"{panel_90d['pool_id'].nunique()} pools")

    # ---- Step 6: Plot ----
    # Sort by Feb 1 TVL, only include pools with panel data
    plot_pools = []
    for pid in panel_90d["pool_id"].unique():
        if pid in tvl_lookup:
            plot_pools.append((pid, tvl_lookup[pid]))
    plot_pools.sort(key=lambda x: -x[1])
    print(f"\n6. Plotting {len(plot_pools)} pools...")

    plot_pages(plot_pools, fitted, fitted, panel_90d, pools_df,
               tvl_lookup, pop_B, pop_cov_names, output_dir=output_dir)

    # ---- Summary table ----
    summary = []
    for pid, feb_tvl in plot_pools:
        theta, source = _get_theta_for_pool(
            pid, fitted, panel_90d, pop_B, pop_cov_names
        )
        if theta is None:
            continue
        pp = panel_90d[panel_90d["pool_id"] == pid]
        x = np.column_stack([
            np.ones(len(pp)),
            pp["log_tvl_lag1"].values,
            pp["volatility"].values,
            pp["weekend"].values,
        ])
        pred = x @ theta
        actual = pp["log_volume"].values
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        meta = pools_df[pools_df["pool_id"] == pid]
        if len(meta) > 0:
            m = meta.iloc[0]
            tok_str = "/".join(str(t) for t in m["tokens"][:2])
            chain = str(m["chain"])
            ptype = str(m["pool_type"])
        else:
            tok_str = pid[:16]
            chain = "?"
            ptype = "?"

        summary.append({
            "pool_id": pid[:20],
            "tokens": tok_str,
            "chain": chain,
            "type": ptype,
            "tvl_feb1": feb_tvl,
            "n_obs": len(pp),
            "R2": r2,
            "b_c": theta[1],
            "b_sigma": theta[2],
            "b_weekend": theta[3],
            "source": source,
        })

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(output_dir, "top50_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    n_pools = len(summary_df)
    n_fitted_used = (summary_df["source"] == "fitted").sum()
    n_pop = (summary_df["source"] == "population").sum()
    n_reclamm = (summary_df["type"] == "RECLAMM").sum()
    print(f"\n{'='*70}")
    print(f"Summary: {n_pools} pools ({n_fitted_used} fitted, {n_pop} population)")
    print(f"  RECLAMM: {n_reclamm}  WEIGHTED: {n_pools - n_reclamm}")
    print(f"  Median R\u00b2: {summary_df['R2'].median():.3f}")
    print(f"  Mean b_c:  {summary_df['b_c'].mean():.3f}")


if __name__ == "__main__":
    main()
