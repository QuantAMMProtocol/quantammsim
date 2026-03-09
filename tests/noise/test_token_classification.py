"""Tests for _normalise_symbol and classify_token_tier."""

import pytest
from quantammsim.noise_calibration import _normalise_symbol, classify_token_tier


# ===========================================================================
# TestNormaliseSymbol
# ===========================================================================


class TestNormaliseSymbol:
    def test_passthrough_unknown(self):
        assert _normalise_symbol("FOO") == "FOO"

    def test_known_mapping_preserved(self):
        assert _normalise_symbol("WETH") == "WETH"
        assert _normalise_symbol("WBTC") == "WBTC"
        assert _normalise_symbol("cbBTC") == "cbBTC"

    def test_whitespace_stripped(self):
        assert _normalise_symbol(" ETH ") == "ETH"
        assert _normalise_symbol("  WETH  ") == "WETH"

    def test_case_sensitivity_preserved(self):
        # Lowercase is NOT normalised to uppercase
        assert _normalise_symbol("weth") == "weth"


# ===========================================================================
# TestClassifyTokenTier
# ===========================================================================


class TestClassifyTokenTier:
    def test_tier0_native_tokens(self):
        for sym in ["ETH", "BTC"]:
            assert classify_token_tier(sym) == 0, f"{sym} should be tier 0"

    def test_tier0_wrapped(self):
        for sym in ["WETH", "WBTC", "cbBTC"]:
            assert classify_token_tier(sym) == 0, f"{sym} should be tier 0"

    def test_tier0_stablecoins(self):
        for sym in ["USDC", "USDT", "DAI"]:
            assert classify_token_tier(sym) == 0, f"{sym} should be tier 0"

    def test_tier0_chain_natives(self):
        for sym in ["MATIC", "AVAX", "GNO", "S", "wS"]:
            assert classify_token_tier(sym) == 0, f"{sym} should be tier 0"

    def test_tier1_defi_bluechips(self):
        for sym in ["AAVE", "BAL", "COW", "LINK", "ARB"]:
            assert classify_token_tier(sym) == 1, f"{sym} should be tier 1"

    def test_tier2_unknown_tokens(self):
        for sym in ["RATS", "PEPE"]:
            assert classify_token_tier(sym) == 2, f"{sym} should be tier 2"

    def test_tier2_empty_string(self):
        assert classify_token_tier("") == 2

    def test_wrapped_variant_normalisation(self):
        # "wS" is in the mapping table AND in _TIER_0
        assert classify_token_tier("wS") == 0
        assert classify_token_tier(" wS ") == 0
