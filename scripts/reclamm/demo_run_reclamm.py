"""Demo runs for reClAMM pools vs Balancer 50/50 baseline.

Runs reClAMM pool simulations with parameters pulled from on-chain pools
(AAVE/ETH) and hypothetical configurations, each paired with a Balancer
50/50 constant-weight pool at the same fee level for comparison.

Usage:
    cd <repo-root>
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim-reclamm
    python scripts/demo_run_reclamm.py
"""

import jax.numpy as jnp
from quantammsim.runners.jax_runners import do_run_on_historic_data


def to_daily_price_shift_base(daily_price_shift_exponent):
    """Convert shift rate to daily price shift base (matches Solidity)."""
    return 1.0 - daily_price_shift_exponent / 124649.0


def balancer_fingerprint(tokens, start, end, fees):
    """Build a Balancer 50/50 fingerprint matching the given reclamm config."""
    return {
        "tokens": tokens,
        "rule": "balancer",
        "startDateString": start,
        "endDateString": end,
        "initial_pool_value": 1000000.0,
        "do_arb": True,
        "fees": fees,
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "chunk_period": 60,
        "weight_interpolation_period": 60,
    }


SCENARIOS = [
    {
        "name": "AAVE/ETH on-chain (25bps)",
        "reclamm": {
            "fingerprint": {
                "tokens": ["AAVE", "ETH"],
                "rule": "reclamm",
                "startDateString": "2024-06-01 00:00:00",
                "endDateString": "2025-06-01 00:00:00",
                "initial_pool_value": 1000000.0,
                "do_arb": True,
                "fees": 0.0025,
                "gas_cost": 0.0,
                "arb_fees": 0.0,
                "chunk_period": 60,
                "weight_interpolation_period": 60,
            },
            "params": {
                "price_ratio": jnp.array(1.5),
                "centeredness_margin": jnp.array(0.5),
                "daily_price_shift_base": jnp.array(
                    to_daily_price_shift_base(0.1)
                ),
            },
        },
    },
    {
        "name": "AAVE/ETH zero fees",
        "reclamm": {
            "fingerprint": {
                "tokens": ["AAVE", "ETH"],
                "rule": "reclamm",
                "startDateString": "2024-06-01 00:00:00",
                "endDateString": "2025-06-01 00:00:00",
                "initial_pool_value": 1000000.0,
                "do_arb": True,
                "fees": 0.0,
                "gas_cost": 0.0,
                "arb_fees": 0.0,
                "chunk_period": 60,
                "weight_interpolation_period": 60,
            },
            "params": {
                "price_ratio": jnp.array(1.5),
                "centeredness_margin": jnp.array(0.5),
                "daily_price_shift_base": jnp.array(
                    to_daily_price_shift_base(0.1)
                ),
            },
        },
    },
    {
        "name": "AAVE/ETH wide range (25bps)",
        "reclamm": {
            "fingerprint": {
                "tokens": ["AAVE", "ETH"],
                "rule": "reclamm",
                "startDateString": "2024-06-01 00:00:00",
                "endDateString": "2025-06-01 00:00:00",
                "initial_pool_value": 1000000.0,
                "do_arb": True,
                "fees": 0.0025,
                "gas_cost": 0.0,
                "arb_fees": 0.0,
                "chunk_period": 60,
                "weight_interpolation_period": 60,
            },
            "params": {
                "price_ratio": jnp.array(4.0),
                "centeredness_margin": jnp.array(0.2),
                "daily_price_shift_base": jnp.array(
                    to_daily_price_shift_base(1.0)
                ),
            },
        },
    },
    {
        "name": "BTC/ETH (10bps)",
        "reclamm": {
            "fingerprint": {
                "tokens": ["BTC", "ETH"],
                "rule": "reclamm",
                "startDateString": "2024-01-01 00:00:00",
                "endDateString": "2025-06-01 00:00:00",
                "initial_pool_value": 1000000.0,
                "do_arb": True,
                "fees": 0.001,
                "gas_cost": 0.0,
                "arb_fees": 0.0,
                "chunk_period": 60,
                "weight_interpolation_period": 60,
            },
            "params": {
                "price_ratio": jnp.array(2.0),
                "centeredness_margin": jnp.array(0.3),
                "daily_price_shift_base": jnp.array(
                    to_daily_price_shift_base(0.5)
                ),
            },
        },
    },
]


def run_scenario(scenario):
    """Run a reClAMM config and its Balancer 50/50 baseline, print comparison."""
    rc = scenario["reclamm"]
    fp = rc["fingerprint"]

    # Run reClAMM
    reclamm_result = do_run_on_historic_data(
        run_fingerprint=fp, params=rc["params"]
    )

    # Run Balancer 50/50 with same tokens, dates, fees
    bal_fp = balancer_fingerprint(
        fp["tokens"], fp["startDateString"], fp["endDateString"], fp["fees"]
    )
    bal_params = {
        "initial_weights_logits": jnp.zeros(len(fp["tokens"])),
    }
    balancer_result = do_run_on_historic_data(
        run_fingerprint=bal_fp, params=bal_params
    )

    # HODL value (from reClAMM initial reserves at final prices)
    hodl_value = float(
        (reclamm_result["reserves"][0] * reclamm_result["prices"][-1]).sum()
    )

    rc_final = float(reclamm_result["final_value"])
    bal_final = float(balancer_result["final_value"])
    rc_init = float(reclamm_result["value"][0])
    bal_init = float(balancer_result["value"][0])

    print("=" * 80)
    print(f"  {scenario['name']}")
    print(f"  Tokens: {', '.join(fp['tokens'])}  |  Fees: {fp['fees']}")
    print("-" * 80)
    print(f"  {'':30s} {'reClAMM':>14s} {'Balancer 50/50':>14s}")
    print(f"  {'Initial value':30s} ${rc_init:>13,.0f} ${bal_init:>13,.0f}")
    print(f"  {'Final value':30s} ${rc_final:>13,.0f} ${bal_final:>13,.0f}")
    print(
        f"  {'Return':30s} "
        f"{(rc_final / rc_init - 1) * 100:>13.2f}% "
        f"{(bal_final / bal_init - 1) * 100:>13.2f}%"
    )
    print(
        f"  {'vs HODL':30s} "
        f"{(rc_final / hodl_value - 1) * 100:>13.2f}% "
        f"{(bal_final / hodl_value - 1) * 100:>13.2f}%"
    )
    print(
        f"  {'reClAMM vs Balancer':30s} "
        f"{(rc_final / bal_final - 1) * 100:>13.2f}%"
    )
    print("=" * 80)


if __name__ == "__main__":
    for scenario in SCENARIOS:
        print(f"\n>>> {scenario['name']}...")
        try:
            run_scenario(scenario)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
