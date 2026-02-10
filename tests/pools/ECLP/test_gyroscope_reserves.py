import jax.numpy as jnp
import numpy as np
import pytest
from quantammsim.pools.ECLP.gyroscope import GyroscopePool
from quantammsim.pools.ECLP.gyroscope_reserves import (
    _jax_calc_gyroscope_inner_price,
    _jax_calc_gyroscope_invariant,
    calculate_A_matrix,
    calculate_A_matrix_inv,
)
from quantammsim.runners.jax_runner_utils import Hashabledict
from jax import jit
from functools import partial

def test_numeraire_price_relationships():
    """Test that pool behaves correctly with different numeraire tokens.

    Note: Invariants with different alpha/beta bounds are NOT expected to be equal.
    The invariant is internal to each curve. What we verify is that:
    1. Each curve has a valid positive invariant
    2. Inner prices stay within their respective bounds

    Important: Market prices fed to the pool must have ratios WITHIN the
    alpha-beta bounds. If ratio is outside bounds, the pool goes to extreme
    positions and the invariant calculation fails (negative under sqrt).
    """
    pool = GyroscopePool()

    # Setup base parameters for USDC numeraire
    # Price ratio (token0/token1) must be in [alpha, beta] = [0.25, 5.0]
    usdc_params = {
        "alpha": 0.25,  # price ratio >= 0.25
        "beta": 5.0,    # price ratio <= 5.0
        "phi": jnp.pi / 4,
        "lam": 2.0,
    }

    # Reciprocal parameters for ETH numeraire
    # When numeraire flips, bounds become [1/5.0, 1/0.25] = [0.2, 4.0]
    eth_params = {
        "alpha": 0.2,   # USDC/ETH >= 0.2  (1/5.0)
        "beta": 4.0,    # USDC/ETH <= 4.0  (1/0.25)
        "phi": jnp.pi / 4,
        "lam": 2.0,
    }

    # Test prices - ratios must be within [0.25, 5.0]
    # Using token0=ETH-like, token1=USDC-like with ratio = token0_price/token1_price
    prices = jnp.array(
        [
            [2.0, 1.0],   # ratio = 2.0, within [0.25, 5.0] ✓
            [1.0, 1.0],   # ratio = 1.0, within [0.25, 5.0] ✓
            [0.5, 1.0],   # ratio = 0.5, within [0.25, 5.0] ✓
        ]
    )

    # Reciprocal prices for ETH numeraire - ratios should be reciprocals
    # ratio = 1/2 = 0.5, 1/1 = 1.0, 1/0.5 = 2.0, all within [0.2, 4.0]
    eth_prices = jnp.array(
        [
            [1.0, 2.0],   # ratio = 0.5, within [0.2, 4.0] ✓
            [1.0, 1.0],   # ratio = 1.0, within [0.2, 4.0] ✓
            [1.0, 0.5],   # ratio = 2.0, within [0.2, 4.0] ✓
        ]
    )

    base_fingerprint = {
        "n_assets": 2,
        "bout_length": 4,
        "initial_pool_value": 1000.0,
        "arb_frequency": 1,
        "do_arb": True,
    }

    # Test with USDC numeraire
    usdc_fingerprint = {
        **base_fingerprint,
        "numeraire": "USDC",
        "tokens": ["ETH", "USDC"],
    }

    # Test with ETH numeraire
    eth_fingerprint = {
        **base_fingerprint,
        "numeraire": "ETH",
        "tokens": ["ETH", "USDC"],
    }

    start_index = jnp.array([0, 0])  # Start indices for 2D prices array

    # Calculate reserves for both configurations
    usdc_reserves = pool.calculate_reserves_zero_fees(
        usdc_params, Hashabledict(usdc_fingerprint), prices, start_index
    )

    eth_reserves = pool.calculate_reserves_zero_fees(
        eth_params, Hashabledict(eth_fingerprint), eth_prices, start_index
    )

    # Get matrices for price calculations
    usdc_A = calculate_A_matrix(
        jnp.cos(usdc_params["phi"]), jnp.sin(usdc_params["phi"]), usdc_params["lam"]
    )
    usdc_A_inv = calculate_A_matrix_inv(
        jnp.cos(usdc_params["phi"]), jnp.sin(usdc_params["phi"]), usdc_params["lam"]
    )

    eth_A = calculate_A_matrix(
        jnp.cos(eth_params["phi"]), jnp.sin(eth_params["phi"]), eth_params["lam"]
    )
    eth_A_inv = calculate_A_matrix_inv(
        jnp.cos(eth_params["phi"]), jnp.sin(eth_params["phi"]), eth_params["lam"]
    )

    # Calculate invariants (each is internal to its curve, they don't need to be equal)
    usdc_invariant = _jax_calc_gyroscope_invariant(
        usdc_reserves[0], usdc_params["alpha"], usdc_params["beta"], usdc_A, usdc_A_inv
    )

    eth_invariant = _jax_calc_gyroscope_invariant(
        eth_reserves[0], eth_params["alpha"], eth_params["beta"], eth_A, eth_A_inv
    )

    # Verify both invariants are positive (valid curves)
    assert usdc_invariant > 0, f"USDC invariant should be positive, got {usdc_invariant}"
    assert eth_invariant > 0, f"ETH invariant should be positive, got {eth_invariant}"

    # Verify inner prices are within bounds for each curve
    for usdc_reserve, eth_reserve in zip(usdc_reserves, eth_reserves):
        usdc_price = _jax_calc_gyroscope_inner_price(
            usdc_reserve,
            usdc_params["alpha"],
            usdc_params["beta"],
            usdc_A,
            usdc_A_inv,
            usdc_invariant,
        )

        eth_price = _jax_calc_gyroscope_inner_price(
            eth_reserve,
            eth_params["alpha"],
            eth_params["beta"],
            eth_A,
            eth_A_inv,
            eth_invariant,
        )

        # Verify price bounds - USDC numeraire
        assert (
            usdc_params["alpha"] <= usdc_price <= usdc_params["beta"]
        ), f"USDC price {usdc_price} outside bounds [{usdc_params['alpha']}, {usdc_params['beta']}]"

        # Verify price bounds - ETH numeraire
        assert (
            eth_params["alpha"] <= eth_price <= eth_params["beta"]
        ), f"ETH price {eth_price} outside bounds [{eth_params['alpha']}, {eth_params['beta']}]"


def test_edge_cases():
    """Test behavior near price bounds with different numeraire tokens"""
    pool = GyroscopePool()

    # USDC numeraire parameters
    usdc_params = {"alpha": 0.25, "beta": 5.0, "phi": jnp.pi / 4, "lam": 2.0}

    # Test prices near bounds
    near_bounds_prices = jnp.array(
        [
            [0.26, 1.0],  # Just above alpha
            [4.99, 1.0],  # Just below beta
            [0.251, 1.0],  # Very close to alpha
            [4.999, 1.0],  # Very close to beta
        ]
    )

    # TODO: Add corresponding ETH numeraire tests


def test_trade_execution():
    """Test trade execution with different numeraire tokens"""
    # TODO: Implement trade execution tests with clear price relationships
    pass


def test_lam_phi_symmetry():
    """Test how lambda and phi behave under numeraire swap"""
    pool = GyroscopePool()

    # Test multiple parameter combinations
    test_cases = [
        {"phi": 0.0, "lam": 2.0},
        {"phi": jnp.pi / 4, "lam": 2.0},
        {"phi": jnp.pi / 2, "lam": 2.0},
        {"phi": jnp.pi / 4, "lam": 0.5},
        {"phi": jnp.pi / 4, "lam": 1.0},
    ]

    base_params = {
        "alpha": 0.25,
        "beta": 5.0,
    }

    base_fingerprint = {
        "n_assets": 2,
        "bout_length": 4,
        "initial_pool_value": 1000.0,
        "arb_frequency": 1,
        "do_arb": True,
    }

    # Test prices that explore the range
    prices = jnp.array(
        [
            [1000.0, 1.0],  # Middle of range
            [2000.0, 1.0],  # Higher
            [500.0, 1.0],  # Lower
        ]
    )

    eth_prices = jnp.array(
        [
            [1.0, 0.001],
            [1.0, 0.0005],
            [1.0, 0.002],
        ]
    )

    for case in test_cases:
        # USDC numeraire parameters
        usdc_params = {**base_params, **case}

        # Test different transformations for ETH numeraire
        eth_params_tests = [
            # Test 1: Keep same phi and lam
            {**base_params, "phi": case["phi"], "lam": case["lam"]},
            # Test 2: Negate phi
            {**base_params, "phi": -case["phi"], "lam": case["lam"]},
            # Test 3: Invert lam
            {**base_params, "phi": case["phi"], "lam": 1 / case["lam"]},
            # Test 4: Both negate phi and invert lam
            {**base_params, "phi": -case["phi"], "lam": 1 / case["lam"]},
        ]

        # Calculate USDC numeraire case
        usdc_fingerprint = {
            **base_fingerprint,
            "numeraire": "USDC",
            "tokens": ["ETH", "USDC"],
        }
        usdc_reserves = pool.calculate_reserves_zero_fees(
            usdc_params, Hashabledict(usdc_fingerprint), prices, jnp.array([0, 0])
        )

        # Get USDC price curve
        usdc_A = calculate_A_matrix(
            jnp.cos(usdc_params["phi"]), jnp.sin(usdc_params["phi"]), usdc_params["lam"]
        )
        usdc_A_inv = calculate_A_matrix_inv(
            jnp.cos(usdc_params["phi"]), jnp.sin(usdc_params["phi"]), usdc_params["lam"]
        )
        usdc_invariant = _jax_calc_gyroscope_invariant(
            usdc_reserves[0],
            usdc_params["alpha"],
            usdc_params["beta"],
            usdc_A,
            usdc_A_inv,
        )

        # Test each ETH numeraire transformation
        for i, eth_params in enumerate(eth_params_tests):
            eth_fingerprint = {
                **base_fingerprint,
                "numeraire": "ETH",
                "tokens": ["ETH", "USDC"],
            }
            eth_reserves = pool.calculate_reserves_zero_fees(
                eth_params, Hashabledict(eth_fingerprint), eth_prices, jnp.array([0, 0])
            )

            eth_A = calculate_A_matrix(
                jnp.cos(eth_params["phi"]),
                jnp.sin(eth_params["phi"]),
                eth_params["lam"],
            )
            eth_A_inv = calculate_A_matrix_inv(
                jnp.cos(eth_params["phi"]),
                jnp.sin(eth_params["phi"]),
                eth_params["lam"],
            )
            eth_invariant = _jax_calc_gyroscope_invariant(
                eth_reserves[0],
                eth_params["alpha"],
                eth_params["beta"],
                eth_A,
                eth_A_inv,
            )

            # Compare price curves
            for usdc_reserve, eth_reserve in zip(usdc_reserves, eth_reserves):
                usdc_price = _jax_calc_gyroscope_inner_price(
                    usdc_reserve,
                    usdc_params["alpha"],
                    usdc_params["beta"],
                    usdc_A,
                    usdc_A_inv,
                    usdc_invariant,
                )

                eth_price = _jax_calc_gyroscope_inner_price(
                    eth_reserve,
                    eth_params["alpha"],
                    eth_params["beta"],
                    eth_A,
                    eth_A_inv,
                    eth_invariant,
                )

                print(f"\nTest case phi={case['phi']}, lam={case['lam']}")
                print(
                    f"ETH params test {i}: phi={eth_params['phi']}, lam={eth_params['lam']}"
                )
                print(f"USDC price: {usdc_price}, ETH price: {eth_price}")
                print(f"Product: {usdc_price * eth_price}")
