"""Token tier classification."""

from .constants import _TIER_0, _TIER_1


def _normalise_symbol(symbol: str) -> str:
    """Normalise wrapped/bridged variants to canonical form."""
    s = symbol.strip()
    mapping = {
        "WETH": "WETH", "WBTC": "WBTC", "cbBTC": "cbBTC",
        "WMATIC": "WMATIC", "WAVAX": "WAVAX", "WXDAI": "WXDAI", "wS": "wS",
    }
    return mapping.get(s, s)


def classify_token_tier(symbol: str) -> int:
    """Classify a token symbol into tier 0/1/2."""
    s = _normalise_symbol(symbol)
    if s in _TIER_0:
        return 0
    if s in _TIER_1:
        return 1
    return 2
