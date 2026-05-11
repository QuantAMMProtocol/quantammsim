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


def reclamm_fingerprint(tokens, start, end, fees, interpolation_method="geometric"):
    """Build a reCLAMM fingerprint for a demo scenario."""
    return {
        "tokens": tokens,
        "rule": "reclamm",
        "startDateString": start,
        "endDateString": end,
        "initial_pool_value": 1000000.0,
        "do_arb": True,
        "fees": fees,
        "gas_cost": 0.0,
        "arb_fees": 0.0,
        "chunk_period": 60,
        "weight_interpolation_period": 60,
        "reclamm_interpolation_method": interpolation_method,
        "reclamm_arc_length_speed": None,
    }


def reclamm_params(price_ratio, centeredness_margin, daily_price_shift_exponent):
    """Build reCLAMM params from a concise config."""
    return {
        "price_ratio": jnp.array(price_ratio),
        "centeredness_margin": jnp.array(centeredness_margin),
        "daily_price_shift_base": jnp.array(
            to_daily_price_shift_base(daily_price_shift_exponent)
        ),
    }


def _apply_active_noise_settings(fp):
    """Enable the active AAVE/ETH reCLAMM noise model for demo runs."""
    if fp.get("rule") != "reclamm" or list(fp.get("tokens", [])) != ["AAVE", "ETH"]:
        return fp, "disabled"

    from compare_reclamm_thermostats import (
        AAVE_ETH_NOISE_SETTINGS,
        resolve_reclamm_noise_settings,
    )

    cfg = {
        "tokens": fp["tokens"],
        "start": fp["startDateString"],
        "end": fp["endDateString"],
        "enable_noise_model": True,
        "noise_model": AAVE_ETH_NOISE_SETTINGS["noise_model"],
        "noise_artifact_dir": AAVE_ETH_NOISE_SETTINGS["noise_artifact_dir"],
        "noise_pool_id": AAVE_ETH_NOISE_SETTINGS["noise_pool_id"],
        "gas_cost": fp.get("gas_cost", AAVE_ETH_NOISE_SETTINGS["gas_cost"]),
        "protocol_fee_split": fp.get(
            "protocol_fee_split",
            AAVE_ETH_NOISE_SETTINGS["protocol_fee_split"],
        ),
        "arb_frequency": fp.get("arb_frequency"),
        "noise_trader_ratio": fp.get("noise_trader_ratio", 0.0),
        "reclamm_noise_params": fp.get("reclamm_noise_params"),
        "noise_arrays_path": fp.get("noise_arrays_path"),
    }
    noise_cfg = resolve_reclamm_noise_settings(cfg)

    updated = dict(fp)
    updated["gas_cost"] = cfg["gas_cost"]
    updated["protocol_fee_split"] = cfg["protocol_fee_split"]
    updated["noise_trader_ratio"] = noise_cfg.get("noise_trader_ratio", 0.0)
    for key in ("noise_model", "reclamm_noise_params", "noise_arrays_path", "arb_frequency"):
        if noise_cfg.get(key) is not None:
            updated[key] = noise_cfg[key]
    return updated, noise_cfg["noise_summary"]


SCENARIOS = [
    {
        "name": "AAVE/ETH launch-style range (25bps, geometric)",
        "reclamm": {
            "fingerprint": reclamm_fingerprint(
                ["AAVE", "ETH"],
                "2024-06-01 00:00:00",
                "2025-06-01 00:00:00",
                0.0025,
                interpolation_method="geometric",
            ),
            "params": reclamm_params(1.5014, 0.5, 0.1),
        },
    },
    {
        "name": "AAVE/ETH tighter launch-style range (25bps, geometric)",
        "reclamm": {
            "fingerprint": reclamm_fingerprint(
                ["AAVE", "ETH"],
                "2024-06-01 00:00:00",
                "2025-06-01 00:00:00",
                0.0025,
                interpolation_method="geometric",
            ),
            "params": reclamm_params(1.15, 0.5, 0.1),
        },
    },
    {
        "name": "AAVE/ETH tighter launch-style range (25bps, constant arc)",
        "reclamm": {
            "fingerprint": reclamm_fingerprint(
                ["AAVE", "ETH"],
                "2024-06-01 00:00:00",
                "2025-06-01 00:00:00",
                0.0025,
                interpolation_method="constant_arc_length",
            ),
            "params": reclamm_params(1.15, 0.5, 0.1),
        },
    },
    {
        "name": "BTC/ETH (10bps)",
        "reclamm": {
            "fingerprint": reclamm_fingerprint(
                ["BTC", "ETH"],
                "2024-01-01 00:00:00",
                "2025-06-01 00:00:00",
                0.001,
                interpolation_method="geometric",
            ),
            "params": reclamm_params(2.0, 0.3, 0.5),
        },
    },
]


def run_scenario(scenario):
    """Run a reClAMM config and its Balancer 50/50 baseline, print comparison."""
    rc = scenario["reclamm"]
    fp, noise_summary = _apply_active_noise_settings(dict(rc["fingerprint"]))

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
    print(
        f"  Tokens: {', '.join(fp['tokens'])}  |  Fees: {fp['fees']}  |  "
        f"Interpolation: {fp.get('reclamm_interpolation_method', 'geometric')}"
    )
    print(
        f"  Noise: {noise_summary}  |  Gas: {fp.get('gas_cost', 0.0)}  |  "
        f"Protocol fee split: {fp.get('protocol_fee_split', 0.0)}"
    )
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
