"""Constants for noise calibration."""

import os

K_COEFF = 4
COEFF_NAMES = ["intercept", "b_tvl", "b_sigma", "b_weekend"]

BALANCER_API_URL = "https://api-v3.balancer.fi/"

BALANCER_API_CHAINS = [
    "MAINNET", "POLYGON", "ARBITRUM", "GNOSIS", "BASE", "SONIC", "OPTIMISM",
    "AVALANCHE",
]

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "local_data", "noise_calibration",
)

# Tier 0: blue-chip — top by volume, wrapped native, major stables
_TIER_0 = {
    "ETH", "WETH", "BTC", "WBTC", "cbBTC", "USDC", "USDT", "DAI",
    "wstETH", "stETH", "rETH", "cbETH", "WMATIC", "MATIC", "POL",
    "WAVAX", "AVAX", "GNO", "WXDAI", "xDAI",
    "S", "wS",
}

K_CLUSTERS_DEFAULT = 6
K_FEATURES_DEFAULT = 6

# Structural model: observation-level covariates (expanded from K_COEFF=4)
K_OBS_COEFF = 8
OBS_COEFF_NAMES = [
    "intercept", "b_tvl", "b_sigma",
    "b_tvl_sigma", "b_tvl_fee", "b_sigma_fee",
    "b_dow_sin", "b_dow_cos",
]

# Gas costs per arb transaction (USD) by chain
GAS_COSTS = {
    "MAINNET": None,       # time-varying, loaded from CSV
    "POLYGON": 0.005,
    "ARBITRUM": 0.005,
    "BASE": 0.005,
    "GNOSIS": 0.01,
    "OPTIMISM": 0.005,
    "SONIC": 0.005,
    "AVALANCHE": 0.005,
    "MODE": 0.005,
    "FRAXTAL": 0.005,
}

# Tier 1: mid-cap DeFi blue-chips (approx CoinGecko rank < 200)
_TIER_1 = {
    "AAVE", "LINK", "UNI", "BAL", "MKR", "CRV", "COMP", "SNX",
    "LDO", "RPL", "SUSHI", "YFI", "1INCH", "ENS", "DYDX",
    "FXS", "FRAX", "LUSD", "sDAI", "GHO", "crvUSD",
    "ARB", "OP", "PENDLE", "ENA", "EIGEN",
    "SAFE", "COW",
}
