import jax.numpy as jnp
from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
)
from quantammsim.runners.jax_runners import do_run_on_historic_data
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import warnings

warnings.filterwarnings("ignore")
from jax import config

# Default fingerprint used as base for all pools
DEFAULT_FINGERPRINT = {
    "startDateString": "2021-01-01 00:00:00",
    "endDateString": "2024-06-01 00:00:00",
    "endTestDateString": "2024-11-30 00:00:00",
    "chunk_period": 60,
    "weight_interpolation_period": 60,
    "fees": 0.0,
    "gas_cost": 0.0,
    "use_alt_lamb": False,
}

EXAMPLE_CONFIGS = {
    "reclamm_1": {
        "fingerprint": {
        "arb_fees": 0.0,
        "arb_frequency": 15,
        "do_arb": True,
        "endDateString": "2025-06-01 00:00:00",
        "fees": 0.0025,
        "gas_cost": 1.0,
        "initial_pool_value": 1000000.0,
        "noise_trader_ratio": 0.0,
        "protocol_fee_split": 0.25,
        "reclamm_arc_length_speed": None,
        "reclamm_interpolation_method": "geometric",
        "rule": "reclamm",
        "startDateString": "2024-06-01 00:00:00",
        "tokens": [
        "AAVE",
        "ETH"
        ]
    },
    "params": {
        "centeredness_margin": 0.3184210526315789,
        "daily_price_shift_base": 0.9999984155508669,
        "price_ratio": 1.3349999999999989
    }
    },
    "reclamm_2":{
    "fingerprint": {
    "arb_fees": 0.0,
    "arb_frequency": 15,
    "do_arb": True,
    "endDateString": "2025-06-01 00:00:00",
    "fees": 0.0025,
    "gas_cost": 1.0,
    "initial_pool_value": 1000000.0,
    "noise_model": "calibrated",
    "noise_trader_ratio": 0.0,
    "protocol_fee_split": 0.25,
    "reclamm_arc_length_speed": None,
    "reclamm_interpolation_method": "geometric",
    "reclamm_noise_params": {
      "c_0": -0.453,
      "c_1": 0.025,
      "c_2": -0.06,
      "c_3": 0.31,
      "c_4": -0.149,
      "c_5": 0.359,
      "c_6": 0.061,
      "c_7": 0.06
    },
    "rule": "reclamm",
    "startDateString": "2024-06-01 00:00:00",
    "tokens": [
      "AAVE",
      "ETH"
    ]
  },
  "params": {
    "centeredness_margin": 0.3184210526315789,
    "daily_price_shift_base": 0.9999984155508669,
    "price_ratio": 1.3349999999999989
  }
    }
}


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from quantammsim.core_simulator.param_utils import (
        generate_params_combinations,
        jax_logit_lamb_to_lamb,
        lamb_to_memory_days,
        lamb_to_memory_days_clipped,
        calc_lamb,
    )
    from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
        squareplus,
        inverse_squareplus,
        inverse_squareplus_np,
    )

    for name, config in EXAMPLE_CONFIGS.items():
        print(name)
        if 'reclamm' not in name:
            continue
        print(f"\nRunning {name}...")
        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
        )
        print("-" * 80)
        print(f"Pool Type: {config['fingerprint']['rule']}")
        print(f"Tokens: {', '.join(config['fingerprint']['tokens'])}")
        print(f"Fees: {config['fingerprint'].get('fees', 0.0)}")
        if "arb_quality" in config["fingerprint"]:
            print(f"Arb Quality: {config['fingerprint']['arb_quality']}")
        print(f"Initial Pool Value: ${result['value'][0]:.2f}")
        print(f"Final Pool Value: ${result['final_value']:.2f}")
        print(f"Return: {(result['final_value']/result['value'][0]-1)*100:.2f}%")
        print(
            f"Return over hodl: {(result['final_value']/(result['reserves'][0]*result['prices'][-1]).sum()-1)*100:.2f}%"
        )
        print("-" * 80)
        # memory_days = lamb_to_memory_days(jax_logit_lamb_to_lamb(config["params"]["logit_lamb"]), config["fingerprint"]["chunk_period"])
        # print("memory days: ", memory_days)
        if "logit_lamb" in config["params"]:
            memory_days = lamb_to_memory_days_clipped(
                calc_lamb(config["params"]),
                chunk_period=config["fingerprint"]["chunk_period"],
                max_memory_days=365,
            )
            print(f"{'memory days':<20} {str(memory_days)}")
            lamb = calc_lamb(config["params"])
            print(
                f"{'lamb':<20} {jnp.array_str(lamb, precision=16, suppress_small=False)}"
            )
            if "log_k" in config["params"]:
                k = 2 ** config["params"]["log_k"] * memory_days
                k_str = " ".join(f"{x:.16e}" for x in k)
                print(f"{'k':<20} [{k_str}]")
                k_per_day_str = " ".join(
                    f"{x:.16e}" for x in 2 ** config["params"]["log_k"]
                )
                print(f"{'k per day':<20} [{k_per_day_str}]")
        if "raw_exponents" in config["params"]:
            exponents = squareplus(config["params"]["raw_exponents"])
            exp_str = " ".join(f"{x:.16f}" for x in exponents)
            print(f"{'exponents':<20} [{exp_str}]")
        if "raw_width" in config["params"]:
            width = 2 ** config["params"]["raw_width"]
            width_str = " ".join(f"{x:.16e}" for x in width)
            print(f"{'width':<20} [{width_str}]")
        if "log_amplitude" in config["params"]:
            memory_days = lamb_to_memory_days_clipped(
                calc_lamb(config["params"]),
                chunk_period=config["fingerprint"]["chunk_period"],
                max_memory_days=365,
            )
            amplitude = (2 ** config["params"]["log_amplitude"]) * memory_days
            amp_str = " ".join(f"{x:.16e}" for x in amplitude)
            print(f"{'amplitude':<20} [{amp_str}]")
        if "logit_pre_exp_scaling" in config["params"]:
            pre_exp_scaling = jnp.exp(config["params"]["logit_pre_exp_scaling"]) / (
                1 + jnp.exp(config["params"]["logit_pre_exp_scaling"])
            )
            pes_str = " ".join(f"{x:.16f}" for x in pre_exp_scaling)
            print(f"{'pre_exp_scaling':<20} [{pes_str}]")
        if "raw_pre_exp_scaling" in config["params"]:
            pre_exp_scaling = 2 ** config["params"]["raw_pre_exp_scaling"]
            pes_str = " ".join(f"{x:.16f}" for x in pre_exp_scaling)
            print(f"{'pre_exp_scaling':<20} [{pes_str}]")

        print("-" * 80)
        print("final readouts")
        if result.get("readouts") is not None:
            for readout in result["readouts"]:
                print(f"{readout}: { jnp.array_str(result['readouts'][readout][-1], precision=16, suppress_small=False)}")
            print("-" * 80)
            print("final weights")
            print(f"{jnp.array_str(result['weights'][-1], precision=16, suppress_small=False)}")
            print("-" * 80)
            print("final prices")
            print(f"{jnp.array_str(result['prices'][-1], precision=16, suppress_small=False)}")
        print("=" * 80)
        