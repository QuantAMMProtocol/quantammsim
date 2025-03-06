import jax.numpy as jnp
from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
)
from quantammsim.runners.jax_runners import do_run_on_historic_data
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
    "momentum_btc_usdc": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["BTC", "USDC"],
            "rule": "momentum",
        },
        "params": {
            "k": jnp.array([120.0, 2000.0]),
            "logit_lamb": memory_days_to_logit_lamb(
                jnp.array([50.0, 50.0]), chunk_period=60
            ),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        },
    },
    "anti_momentum_btc_usdc": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["BTC", "USDC"],
            "rule": "anti_momentum",
        },
        "params": {
            "log_k": jnp.array([-7.922600426298057] * 2),
            "logit_lamb": jnp.array([7.782979622032561] * 2),
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        },
    },
    "balancer_btc_usdc": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["BTC", "USDC"],
            "rule": "balancer",
        },
        "params": {
            "initial_weights_logits": jnp.array([0.0, 0.0]),
        },
    },
}

if __name__ == "__main__":
    for name, config in EXAMPLE_CONFIGS.items():
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
        print("=" * 80)
