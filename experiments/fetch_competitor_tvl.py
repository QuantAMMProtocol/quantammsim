"""Fetch competitor TVL for each token pair from DeFi Llama.

For each of our 36 calibration pools, finds all other DEX pools trading
the same token pair, sums their daily TVL, and saves as a time series.

K_i(t) = sum_{j != i} TVL_j(t)  for all pools trading pool i's pair

Output: results/competitor_tvl/competitor_tvl.npz
  - pool_ids: list of our pool IDs
  - dates: array of dates (days since epoch or ISO strings)
  - competitor_tvl: (n_dates, n_pools) array of daily competitor TVL in USD

Usage:
    python experiments/fetch_competitor_tvl.py
    python experiments/fetch_competitor_tvl.py --cache-dir results/competitor_tvl
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)

# Map Balancer token names to DeFi Llama symbol conventions
# DeFi Llama symbols are typically uppercase, no dots, no "W" prefix inconsistencies
SYMBOL_MAP = {
    "WETH": "WETH",
    "WBTC": "WBTC",
    "wstETH": "WSTETH",
    "waEthLidowstETH": "WSTETH",
    "waEthLidoWETH": "WETH",
    "waGnowstETH": "WSTETH",
    "waGnoGNO": "GNO",
    "waBasUSDC": "USDC",
    "waBasWETH": "WETH",
    "sDAI": "DAI",
    "scUSD": "USDC",
    "stS": "S",
    "JitoSOL": "JITOSOL",
    # Common DeFi Llama variants
    "USDC.e": "USDC",
    "USDT.e": "USDT",
    "WETH.e": "WETH",
    "WBTC.e": "WBTC",
}


def _normalize_symbol(token):
    """Normalize Balancer token name to DeFi Llama symbol."""
    return SYMBOL_MAP.get(token, token.upper())


def _fetch_json(url, retries=5, delay=3.0):
    """Fetch JSON from URL with exponential backoff."""
    import urllib.request
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "quantammsim/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            wait = delay * (2 ** attempt)  # 3, 6, 12, 24, 48s
            if attempt < retries - 1:
                print(f"    Retry {attempt+1} (wait {wait:.0f}s): {e}")
                time.sleep(wait)
            else:
                raise


def fetch_all_pools(local_path=None):
    """Load DeFi Llama yield pools from local file or API."""
    if local_path and os.path.exists(local_path):
        print(f"Loading DeFi Llama pools from {local_path}...")
        with open(local_path) as f:
            data = json.load(f)
    else:
        print("Fetching DeFi Llama pool list from API...")
        data = _fetch_json("https://yields.llama.fi/pools")
    pools = data.get("data", []) if isinstance(data, dict) else data
    print(f"  {len(pools)} pools")
    return pools


# Map Balancer chain names to DeFi Llama chain names
CHAIN_MAP = {
    "mainnet": "Ethereum",
    "ethereum": "Ethereum",
    "arbitrum": "Arbitrum",
    "polygon": "Polygon",
    "gnosis": "Gnosis",
    "base": "Base",
    "optimism": "Optimism",
    "avalanche": "Avalanche",
    "sonic": "Sonic",
}


def match_pools(our_pools, llama_pools):
    """Match our token pairs to DeFi Llama pools.

    Returns dict: pool_id -> {pair_key, chain, llama_pools_same_chain,
                               llama_pools_all_chains, tokens}
    """
    # Normalize DeFi Llama token symbols to match our convention
    LLAMA_NORMALIZE = {
        "ETH": "WETH",
        "BTC": "WBTC",
        "STETH": "WSTETH",
    }

    # Index llama pools by (pair, chain)
    pair_chain_to_llama = defaultdict(list)
    pair_to_llama = defaultdict(list)
    for p in llama_pools:
        symbol = p.get("symbol", "")
        if not symbol or "-" not in symbol:
            continue
        tokens = symbol.split("-")
        if len(tokens) != 2:
            continue
        normed = [LLAMA_NORMALIZE.get(t.upper(), t.upper()) for t in tokens]
        pair_key = tuple(sorted(normed))
        chain = p.get("chain", "")
        pair_to_llama[pair_key].append(p)
        pair_chain_to_llama[(pair_key, chain)].append(p)

    from quantammsim.calibration.pool_data import _parse_tokens

    matches = {}
    for pid, entry in our_pools.items():
        toks = _parse_tokens(entry["tokens"])
        tok_a = _normalize_symbol(toks[0])
        tok_b = _normalize_symbol(toks[1]) if len(toks) > 1 else tok_a
        pair_key = tuple(sorted([tok_a, tok_b]))

        our_chain = entry.get("chain", "mainnet")
        llama_chain = CHAIN_MAP.get(our_chain.lower(), our_chain)

        matches[pid] = {
            "pair_key": pair_key,
            "chain": llama_chain,
            "llama_pools_same_chain": pair_chain_to_llama.get(
                (pair_key, llama_chain), []),
            "llama_pools_all_chains": pair_to_llama.get(pair_key, []),
            "tokens": (tok_a, tok_b),
        }

    return matches, pair_chain_to_llama


def fetch_pool_history(pool_id):
    """Fetch daily TVL history for a DeFi Llama pool."""
    url = f"https://yields.llama.fi/chart/{pool_id}"
    data = _fetch_json(url)
    points = data.get("data", [])
    if not points:
        return None

    dates = []
    tvls = []
    for p in points:
        ts = p.get("timestamp", "")[:10]
        tvl = p.get("tvlUsd", 0)
        if ts and tvl is not None:
            dates.append(pd.Timestamp(ts))
            tvls.append(float(tvl))

    return pd.Series(tvls, index=pd.DatetimeIndex(dates), name="tvl")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="results/competitor_tvl")
    parser.add_argument("--max-pools-per-pair", type=int, default=30,
                        help="Max DeFi Llama pools to fetch per pair")
    parser.add_argument("--min-tvl", type=float, default=10000,
                        help="Skip pools with current TVL below this")
    parser.add_argument("--pools-json", default=None,
                        help="Local DeFi Llama pools.json (skip API fetch)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Load our pools
    with open(os.path.join(CACHE_DIR, "stage1.pkl"), "rb") as f:
        stage1 = pickle.load(f)
    matched_clean = stage1["matched_clean"]
    pool_ids = sorted(matched_clean.keys())
    print(f"Our pools: {len(pool_ids)}")

    # Fetch DeFi Llama pools
    llama_pools = fetch_all_pools(args.pools_json)

    # Match
    matches, pair_chain_to_llama = match_pools(matched_clean, llama_pools)

    # Summary
    print(f"\nPair matching (same-chain / all-chains):")
    seen = set()
    for pid in pool_ids:
        m = matches[pid]
        pair = m["pair_key"]
        chain = m["chain"]
        key = (pair, chain)
        if key in seen:
            continue
        seen.add(key)
        toks = matched_clean[pid].get("tokens", "?")
        n_same = len(m["llama_pools_same_chain"])
        n_all = len(m["llama_pools_all_chains"])
        tvl_same = sum(p.get("tvlUsd", 0) for p in m["llama_pools_same_chain"])
        tvl_all = sum(p.get("tvlUsd", 0) for p in m["llama_pools_all_chains"])
        print(f"  {'/'.join(pair):>20s}  {chain:>10s}"
              f"  same={n_same:>3d} (${tvl_same/1e6:>7.1f}M)"
              f"  all={n_all:>3d} (${tvl_all/1e6:>7.1f}M)"
              f"  [{toks}]")

    # Flag zero-match pairs — likely symbol mapping issues
    zero_matches = []
    for pid in pool_ids:
        m = matches[pid]
        if len(m["llama_pools_same_chain"]) == 0 and len(m["llama_pools_all_chains"]) == 0:
            toks = matched_clean[pid].get("tokens", "?")
            zero_matches.append((pid[:16], toks, m["pair_key"], m["chain"]))
    if zero_matches:
        print(f"\n  WARNING: {len(zero_matches)} pools with zero DeFi Llama matches"
              f" (check SYMBOL_MAP):")
        for pid, toks, pair, chain in zero_matches:
            print(f"    {pid}  {toks:>20s}  → {'/'.join(pair)} ({chain})")

    # Fetch historical TVL for each (pair, chain) combination
    print(f"\nFetching historical TVL (same-chain)...")
    # Key: (pair, chain) -> pd.Series of daily total TVL
    pair_chain_histories = {}

    fetched = set()
    for pid in pool_ids:
        m = matches[pid]
        pair = m["pair_key"]
        chain = m["chain"]
        key = (pair, chain)
        if key in fetched:
            continue
        fetched.add(key)

        # Same-chain pools, sorted by TVL
        llama = sorted(m["llama_pools_same_chain"],
                       key=lambda p: p.get("tvlUsd", 0), reverse=True)
        llama = [p for p in llama if p.get("tvlUsd", 0) >= args.min_tvl]
        llama = llama[:args.max_pools_per_pair]

        cache_name = f"{'_'.join(pair)}_{chain}_history.pkl"
        pair_cache = os.path.join(args.cache_dir, cache_name)

        if not llama:
            print(f"  {'/'.join(pair)} ({chain}): no qualifying pools")
            continue

        if os.path.exists(pair_cache):
            with open(pair_cache, "rb") as f:
                pair_chain_histories[key] = pickle.load(f)
            print(f"  {'/'.join(pair)} ({chain}): loaded from cache"
                  f" ({len(pair_chain_histories[key])} days)")
            continue

        print(f"  {'/'.join(pair)} ({chain}): fetching {len(llama)} pools...",
              end="", flush=True)
        pool_series = []
        for lp in llama:
            lid = lp["pool"]
            try:
                hist = fetch_pool_history(lid)
                if hist is not None and len(hist) > 10:
                    pool_series.append(hist)
            except Exception as e:
                print(f"\n    Skip {lid}: {e}", end="")
            time.sleep(3.0)  # rate limit — DeFi Llama allows ~1 req/3s
        print(f" got {len(pool_series)} histories")

        if pool_series:
            df = pd.concat(pool_series, axis=1).sort_index()
            pair_chain_histories[key] = df.sum(axis=1)  # skipna=True: pre-launch = 0

            with open(pair_cache, "wb") as f:
                pickle.dump(pair_chain_histories[key], f)

    # --- Network conductance: fetch hub-pair TVL for multi-hop K ---
    HUB_TOKENS = ["WETH", "WSTETH", "USDC", "USDT", "DAI", "WBTC"]

    # Identify all (token, hub) pairs we need across all pools
    hub_pairs_needed = set()
    for pid in pool_ids:
        m = matches[pid]
        tok_a, tok_b = m["tokens"]
        chain = m["chain"]
        for hub in HUB_TOKENS:
            if hub in (tok_a, tok_b):
                continue
            # Need L(tok_a, hub) and L(hub, tok_b) on same chain
            pair_ah = tuple(sorted([tok_a, hub]))
            pair_hb = tuple(sorted([hub, tok_b]))
            hub_pairs_needed.add((pair_ah, chain))
            hub_pairs_needed.add((pair_hb, chain))

    # Remove pairs we already have
    hub_pairs_to_fetch = hub_pairs_needed - fetched
    print(f"\nFetching hub-pair TVL for network conductance...")
    print(f"  {len(hub_pairs_needed)} hub pairs needed,"
          f" {len(hub_pairs_to_fetch)} to fetch")

    for pair, chain in sorted(hub_pairs_to_fetch):
        key = (pair, chain)
        cache_name = f"{'_'.join(pair)}_{chain}_history.pkl"
        pair_cache = os.path.join(args.cache_dir, cache_name)

        if os.path.exists(pair_cache):
            with open(pair_cache, "rb") as f:
                pair_chain_histories[key] = pickle.load(f)
            continue

        # Find matching DeFi Llama pools
        llama = pair_chain_to_llama.get((pair, chain), [])
        llama = sorted(llama, key=lambda p: p.get("tvlUsd", 0), reverse=True)
        llama = [p for p in llama if p.get("tvlUsd", 0) >= args.min_tvl]
        llama = llama[:args.max_pools_per_pair]

        if not llama:
            continue

        print(f"  {'/'.join(pair)} ({chain}): fetching {len(llama)} pools...",
              end="", flush=True)
        pool_series = []
        for lp in llama:
            lid = lp["pool"]
            try:
                hist = fetch_pool_history(lid)
                if hist is not None and len(hist) > 10:
                    pool_series.append(hist)
            except Exception as e:
                print(f"\n    Skip {lid}: {e}", end="")
            time.sleep(3.0)
        print(f" got {len(pool_series)} histories")

        if pool_series:
            df = pd.concat(pool_series, axis=1).sort_index()
            pair_chain_histories[key] = df.sum(axis=1)  # skipna=True: pre-launch = 0
            with open(pair_cache, "wb") as f:
                pickle.dump(pair_chain_histories[key], f)

    # Build per-pool competitor TVL arrays aligned to our panel dates
    print(f"\nBuilding competitor TVL arrays...")

    # Common date grid
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)
    n_dates = len(date_list)
    date_to_idx = {d: i for i, d in enumerate(date_list)}
    n_pools = len(pool_ids)

    competitor_tvl = np.full((n_dates, n_pools), np.nan)

    for j, pid in enumerate(pool_ids):
        m = matches[pid]
        pair = m["pair_key"]
        chain = m["chain"]
        key = (pair, chain)
        if key not in pair_chain_histories:
            continue

        hist = pair_chain_histories[key]
        panel = matched_clean[pid]["panel"]
        panel_dates = panel["date"].values
        # Own TVL for self-exclusion: K_i = total_pair_tvl - own_tvl
        own_tvl = np.exp(panel["log_tvl_lag1"].values.astype(float))

        for k, date in enumerate(panel_dates):
            t = date_to_idx[date]
            day = pd.Timestamp(date).normalize()
            if day in hist.index:
                total = hist.loc[day]
                own = own_tvl[k] if k < len(own_tvl) else 0
                # Competitor TVL = total pair TVL - own TVL (floor at 0)
                competitor_tvl[t, j] = max(total - own, 0)

    # Forward-fill then back-fill gaps
    for j in range(n_pools):
        col = competitor_tvl[:, j]
        mask = np.isfinite(col)
        if mask.any() and not mask.all():
            s = pd.Series(col, index=date_list).ffill().bfill()
            competitor_tvl[:, j] = s.values

    valid = np.isfinite(competitor_tvl)
    n_valid = valid.sum()
    n_total = n_dates * n_pools
    print(f"  Coverage: {n_valid}/{n_total} ({100*n_valid/n_total:.0f}%)")

    # Warn about pools with surprisingly low coverage despite having pair data
    for j, pid in enumerate(pool_ids):
        m = matches[pid]
        key = (m["pair_key"], m["chain"])
        if key in pair_chain_histories and len(pair_chain_histories[key]) > 100:
            col = competitor_tvl[:, j]
            cov = np.isfinite(col).sum() / n_dates
            if cov < 0.5:
                print(f"  WARNING: {pid[:16]} has pair data but only"
                      f" {cov*100:.0f}% coverage — possible date mismatch")

    # --- Compute K_eff = K_direct + multi-hop contributions ---
    print(f"\nComputing network K_eff (direct + multi-hop)...")
    # Compute multi-hop contribution over ALL dates in the grid
    # (not just panel dates — so forward-fill works correctly)
    k_eff = np.full((n_dates, n_pools), np.nan)

    def _get_pair_tvl_on_date(pair_key, chain, day):
        """Get total TVL for a pair on a given date."""
        key = (pair_key, chain)
        if key not in pair_chain_histories:
            return 0.0
        hist = pair_chain_histories[key]
        if day in hist.index:
            return float(hist.loc[day])
        return 0.0

    for j, pid in enumerate(pool_ids):
        m = matches[pid]
        tok_a, tok_b = m["tokens"]
        chain = m["chain"]

        for t, date in enumerate(date_list):
            day = pd.Timestamp(date).normalize()

            # Direct (from already-computed competitor_tvl)
            direct = competitor_tvl[t, j] if np.isfinite(competitor_tvl[t, j]) else 0.0

            # Multi-hop through hub tokens
            multihop = 0.0
            for hub in HUB_TOKENS:
                if hub in (tok_a, tok_b):
                    continue
                pair_ah = tuple(sorted([tok_a, hub]))
                pair_hb = tuple(sorted([hub, tok_b]))
                L_ah = _get_pair_tvl_on_date(pair_ah, chain, day)
                L_hb = _get_pair_tvl_on_date(pair_hb, chain, day)
                if L_ah > 0 and L_hb > 0:
                    multihop += L_ah * L_hb / (L_ah + L_hb)

            total = direct + multihop
            if total > 0:
                k_eff[t, j] = total

    # Forward-fill / back-fill K_eff gaps
    for j in range(n_pools):
        col = k_eff[:, j]
        mask = np.isfinite(col) & (col > 0)
        if mask.any() and not mask.all():
            s = pd.Series(col, index=date_list).ffill().bfill()
            k_eff[:, j] = s.values

    # Per-pool stats
    print(f"\n  {'Pool':>16s}  {'Tokens':>20s}  {'Pair':>20s}"
          f"  {'K_direct med':>14s}  {'K_eff med':>14s}  {'Multi/Dir':>10s}")
    for j, pid in enumerate(pool_ids):
        toks = matched_clean[pid].get("tokens", "?")
        pair = matches[pid]["pair_key"]
        # Use only panel dates for display (not forward-filled grid dates)
        panel_dates = matched_clean[pid]["panel"]["date"].values
        panel_t = [date_to_idx[d] for d in panel_dates if d in date_to_idx]
        if panel_t:
            d_vals = competitor_tvl[panel_t, j]
            e_vals = k_eff[panel_t, j]
            valid_d = d_vals[np.isfinite(d_vals)]
            valid_e = e_vals[np.isfinite(e_vals)]
        else:
            valid_d = valid_e = np.array([])
        med_d = np.median(valid_d) if len(valid_d) > 0 else 0
        med_e = np.median(valid_e) if len(valid_e) > 0 else 0
        ratio = med_e / med_d if med_d > 0 else float("inf")
        print(f"  {pid[:16]}  {toks:>20s}  {'/'.join(pair):>20s}"
              f"  ${med_d:>13,.0f}  ${med_e:>13,.0f}  {ratio:>9.1f}x")

    # Save
    out_path = os.path.join(args.cache_dir, "competitor_tvl.npz")
    np.savez(out_path,
             pool_ids=pool_ids,
             date_list=np.array([str(d) for d in date_list]),
             competitor_tvl=competitor_tvl,
             k_eff=k_eff)
    print(f"\nSaved: {out_path}")

    # Also save raw pair-chain histories for inspection
    pair_path = os.path.join(args.cache_dir, "pair_chain_histories.pkl")
    with open(pair_path, "wb") as f:
        pickle.dump(pair_chain_histories, f)
    print(f"Saved: {pair_path}")


if __name__ == "__main__":
    main()
