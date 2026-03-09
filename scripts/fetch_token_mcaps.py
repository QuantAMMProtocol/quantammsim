"""Fetch token market caps from CoinGecko and cache locally.

Usage:
    python scripts/fetch_token_mcaps.py

Output:
    local_data/noise_calibration/token_mcaps.json
"""

import json
import os
import sys
import time

import requests

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "local_data", "noise_calibration", "token_mcaps.json",
)

# Token symbol -> CoinGecko ID mapping
# Covers all tokens appearing in our 26 matched pools + common Balancer tokens
COINGECKO_IDS = {
    # Blue-chip / wrapped natives
    "WETH": "ethereum",
    "ETH": "ethereum",
    "WBTC": "wrapped-bitcoin",
    "BTC": "bitcoin",
    "cbBTC": "bitcoin",   # Coinbase wrapped BTC — use BTC mcap
    "USDC": "usd-coin",
    "USDT": "tether",
    "DAI": "dai",
    "wstETH": "wrapped-steth",
    "stETH": "staked-ether",
    "rETH": "rocket-pool-eth",
    "cbETH": "coinbase-wrapped-staked-eth",
    "WMATIC": "polygon-ecosystem-token",
    "MATIC": "polygon-ecosystem-token",
    "POL": "polygon-ecosystem-token",
    "WAVAX": "avalanche-2",
    "AVAX": "avalanche-2",
    "GNO": "gnosis",
    "WXDAI": "dai",       # Wrapped xDAI ≈ DAI
    "xDAI": "dai",
    "S": "sonic-3",
    "wS": "sonic-3",
    # Mid-cap DeFi
    "AAVE": "aave",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "BAL": "balancer",
    "MKR": "maker",
    "CRV": "curve-dao-token",
    "COMP": "compound-governance-token",
    "SNX": "havven",
    "LDO": "lido-dao",
    "RPL": "rocket-pool",
    "SUSHI": "sushi",
    "YFI": "yearn-finance",
    "1INCH": "1inch",
    "ENS": "ethereum-name-service",
    "ARB": "arbitrum",
    "OP": "optimism",
    "PENDLE": "pendle",
    "ENA": "ethena",
    "EIGEN": "eigenlayer",
    "COW": "cow-protocol",
    "SAFE": "safe",
    # Smaller / specific tokens in our pools
    "ACX": "across-protocol",
    "ALCX": "alchemix",
    "QI": "benqi",
    "QNT": "quant-network",
    "RDNT": "radiant-capital",
    # TREE not on CoinGecko — handled as fallback below
    "XAI": "xai-blockchain",
    # Wrapped aTokens — use underlying
    "waEthLidoWETH": "ethereum",
    "waEthLidowstETH": "wrapped-steth",
    "waBasWETH": "ethereum",
    "waBasUSDC": "usd-coin",
    "waEthUSDC": "usd-coin",
    "waGnoGNO": "gnosis",
    "waGnowstETH": "wrapped-steth",
    # Additional tokens from expanded pool set
    "wPOL": "polygon-ecosystem-token",
    "stS": "sonic-3",        # Staked Sonic — use S mcap
    "JitoSOL": "jito-governance-token",
    "scUSD": "usd-coin",     # Rings scUSD stablecoin — use USDC mcap as proxy
    "DOLA": "dola-usd",
}

# Asset type classification
STABLECOINS = {
    "USDC", "USDT", "DAI", "WXDAI", "xDAI", "GHO", "LUSD", "crvUSD",
    "FRAX", "sDAI", "scUSD", "DOLA",
    "waBasUSDC", "waEthUSDC",
}
NATIVE_LST = {
    "WETH", "ETH", "wstETH", "stETH", "rETH", "cbETH",
    "WBTC", "BTC", "cbBTC",
    "WMATIC", "MATIC", "POL", "wPOL",
    "WAVAX", "AVAX",
    "GNO", "S", "wS", "stS",
    "JitoSOL",
    "waEthLidoWETH", "waEthLidowstETH",
    "waBasWETH", "waGnoGNO", "waGnowstETH",
}
# Everything else is VOLATILE (asset_type=2)


def fetch_mcaps():
    """Fetch market caps from CoinGecko in batches."""
    unique_ids = sorted(set(COINGECKO_IDS.values()))
    print(f"Fetching market caps for {len(unique_ids)} unique CoinGecko IDs...")

    # CoinGecko allows up to 250 IDs per request
    batch_size = 100
    all_data = {}

    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i:i + batch_size]
        ids_str = ",".join(batch)
        url = (
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={ids_str}&vs_currencies=usd&include_market_cap=true"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        all_data.update(data)
        print(f"  Batch {i // batch_size + 1}: {len(data)} tokens")
        if i + batch_size < len(unique_ids):
            time.sleep(1)  # rate limit

    # Build symbol -> mcap mapping
    mcaps = {}
    missing = []
    for symbol, gecko_id in COINGECKO_IDS.items():
        if gecko_id in all_data and "usd_market_cap" in all_data[gecko_id]:
            mcaps[symbol] = {
                "mcap_usd": all_data[gecko_id]["usd_market_cap"],
                "price_usd": all_data[gecko_id]["usd"],
                "coingecko_id": gecko_id,
            }
        else:
            missing.append((symbol, gecko_id))

    if missing:
        print(f"\n  Missing from CoinGecko: {missing}")

    # Fallback for tokens not on CoinGecko (very small tokens)
    FALLBACK_MCAPS = {
        "TREE": 1_000_000,  # ~$1M estimate for small governance token
    }
    for symbol, mcap_est in FALLBACK_MCAPS.items():
        if symbol not in mcaps:
            mcaps[symbol] = {
                "mcap_usd": mcap_est,
                "price_usd": 0.0,
                "coingecko_id": "fallback",
            }
            print(f"  Fallback: {symbol} -> ${mcap_est:,.0f}")

    # Add asset type classification
    for symbol in mcaps:
        if symbol in STABLECOINS:
            mcaps[symbol]["asset_type"] = "stable"
        elif symbol in NATIVE_LST:
            mcaps[symbol]["asset_type"] = "native_lst"
        else:
            mcaps[symbol]["asset_type"] = "volatile"

    return mcaps


def main():
    mcaps = fetch_mcaps()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(mcaps, f, indent=2)

    print(f"\nSaved {len(mcaps)} tokens to {OUTPUT_PATH}")

    # Summary
    print("\nSample entries:")
    for sym in ["WETH", "AAVE", "USDC", "QI", "TREE"]:
        if sym in mcaps:
            m = mcaps[sym]
            print(f"  {sym}: ${m['mcap_usd']:,.0f} ({m['asset_type']})")


if __name__ == "__main__":
    main()
