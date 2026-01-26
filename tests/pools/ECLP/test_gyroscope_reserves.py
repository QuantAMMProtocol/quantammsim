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

@pytest.mark.skip(
    reason="Test has incorrect expectation: invariants with different alpha/beta "
           "bounds are not expected to be equal. Test logic needs review."
)
def test_numeraire_price_relationships():
    """Test that pool behaves correctly with different numeraire tokens"""
    pool = GyroscopePool()

    # Setup base parameters for USDC numeraire
    usdc_params = {
        "alpha": 0.25,  # ETH/USDC >= 0.25
        "beta": 5.0,  # ETH/USDC <= 5.0
        "phi": jnp.pi / 4,
        "lam": 2.0,
    }

    # Calculate reciprocal parameters for ETH numeraire
    eth_params = {
        "alpha": 0.2,  # USDC/ETH >= 0.2  (1/5.0)
        "beta": 4.0,  # USDC/ETH <= 4.0  (1/0.25)
        "phi": jnp.pi / 4,
        "lam": 2.0,
    }

    # Test prices (ETH/USDC pairs)
    prices = jnp.array(
        [
            [1000.0, 1.0],  # ETH/USDC = 1000
            [2000.0, 1.0],  # ETH/USDC = 2000
            [500.0, 1.0],  # ETH/USDC = 500
        ]
    )

    # Reciprocal prices (USDC/ETH pairs)
    eth_prices = jnp.array(
        [
            [1.0, 0.001],  # USDC/ETH = 0.001
            [1.0, 0.0005],  # USDC/ETH = 0.0005
            [1.0, 0.002],  # USDC/ETH = 0.002
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

    # Test invariants are maintained
    usdc_invariant = _jax_calc_gyroscope_invariant(
        usdc_reserves[0], usdc_params["alpha"], usdc_params["beta"], usdc_A, usdc_A_inv
    )

    eth_invariant = _jax_calc_gyroscope_invariant(
        eth_reserves[0], eth_params["alpha"], eth_params["beta"], eth_A, eth_A_inv
    )

    # Invariants should be equal
    np.testing.assert_almost_equal(usdc_invariant, eth_invariant)

    # Test quoted prices are reciprocals
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

        # Verify prices are reciprocals (within numerical tolerance)
        np.testing.assert_almost_equal(usdc_price * eth_price, 1.0)

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
