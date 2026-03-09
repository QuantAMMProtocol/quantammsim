"""Compare effective weight/exposure trajectories: reClAMM vs Balancer 50/50.

Prints weight stats and saves a plot of weight[AAVE] over time for both pools.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from quantammsim.runners.jax_runners import do_run_on_historic_data


def to_daily_price_shift_base(exponent):
    return 1.0 - exponent / 124649.0


TOKENS = ["AAVE", "ETH"]
START = "2024-06-01 00:00:00"
END = "2025-06-01 00:00:00"

CONFIGS = {
    "reClAMM on-chain (pr=1.5)": {
        "fingerprint": {
            "tokens": TOKENS, "rule": "reclamm",
            "startDateString": START, "endDateString": END,
            "initial_pool_value": 1_000_000.0, "do_arb": True,
            "fees": 0.0, "gas_cost": 0.0, "arb_fees": 0.0,
            "chunk_period": 60, "weight_interpolation_period": 60,
        },
        "params": {
            "price_ratio": jnp.array(1.5),
            "centeredness_margin": jnp.array(0.5),
            "daily_price_shift_base": jnp.array(to_daily_price_shift_base(0.1)),
        },
    },
    "reClAMM wide (pr=4)": {
        "fingerprint": {
            "tokens": TOKENS, "rule": "reclamm",
            "startDateString": START, "endDateString": END,
            "initial_pool_value": 1_000_000.0, "do_arb": True,
            "fees": 0.0, "gas_cost": 0.0, "arb_fees": 0.0,
            "chunk_period": 60, "weight_interpolation_period": 60,
        },
        "params": {
            "price_ratio": jnp.array(4.0),
            "centeredness_margin": jnp.array(0.2),
            "daily_price_shift_base": jnp.array(to_daily_price_shift_base(1.0)),
        },
    },
    "reClAMM Phase 2 (pr=4, m=0.1)": {
        "fingerprint": {
            "tokens": TOKENS, "rule": "reclamm",
            "startDateString": START, "endDateString": END,
            "initial_pool_value": 1_000_000.0, "do_arb": True,
            "fees": 0.0, "gas_cost": 0.0, "arb_fees": 0.0,
            "chunk_period": 60, "weight_interpolation_period": 60,
        },
        "params": {
            "price_ratio": jnp.array(4.0),
            "centeredness_margin": jnp.array(0.1),
            "daily_price_shift_base": jnp.array(to_daily_price_shift_base(0.001)),
        },
    },
    "Balancer 50/50": {
        "fingerprint": {
            "tokens": TOKENS, "rule": "balancer",
            "startDateString": START, "endDateString": END,
            "initial_pool_value": 1_000_000.0, "do_arb": True,
            "fees": 0.0, "gas_cost": 0.0, "arb_fees": 0.0,
            "chunk_period": 60, "weight_interpolation_period": 60,
        },
        "params": {
            "initial_weights_logits": jnp.zeros(2),
        },
    },
}

results = {}
for name, cfg in CONFIGS.items():
    print(f"Running {name}...")
    r = do_run_on_historic_data(
        run_fingerprint=cfg["fingerprint"], params=cfg["params"]
    )
    results[name] = r

# Compute effective weights (value fraction in token 0 = AAVE)
print("\n" + "=" * 90)
print(f"  {'Config':<35s} {'w_AAVE mean':>10s} {'w_AAVE std':>10s} "
      f"{'w_AAVE min':>10s} {'w_AAVE max':>10s} {'vs HODL':>10s}")
print("-" * 90)

daily = 1440  # subsample to daily for stats and plotting
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

for name, r in results.items():
    reserves = np.array(r["reserves"])
    prices = np.array(r["prices"])
    values = reserves * prices  # (T, 2)
    total = values.sum(axis=1, keepdims=True)
    weights = values / np.clip(total, 1e-10, None)  # (T, 2)
    w_aave = weights[::daily, 0]

    hodl_value = float((reserves[0] * prices[-1]).sum())
    vs_hodl = r["final_value"] / hodl_value - 1.0

    print(f"  {name:<35s} {w_aave.mean():>10.4f} {w_aave.std():>10.4f} "
          f"{w_aave.min():>10.4f} {w_aave.max():>10.4f} {vs_hodl * 100:>9.2f}%")

    days = np.arange(len(w_aave))
    axes[0].plot(days, w_aave, label=name, alpha=0.8)

    # Pool value over time
    pool_val = np.array(r["value"])[::daily]
    axes[1].plot(days[:len(pool_val)], pool_val / 1e6, label=name, alpha=0.8)

print("=" * 90)

# HODL line
r0 = results[list(results.keys())[0]]
prices_daily = np.array(r0["prices"])[::daily]
reserves_0 = np.array(r0["reserves"])[0]
hodl_val = (reserves_0 * prices_daily).sum(axis=1) / 1e6
axes[1].plot(np.arange(len(hodl_val)), hodl_val, label="HODL", ls="--", color="gray", alpha=0.7)

# Price ratio (AAVE/ETH) on third axis
price_ratio_series = prices_daily[:, 0] / prices_daily[:, 1]
axes[2].plot(np.arange(len(price_ratio_series)), price_ratio_series, color="black", alpha=0.7)
axes[2].set_ylabel("AAVE/ETH price")
axes[2].set_xlabel("Days")

axes[0].set_ylabel("AAVE weight (value fraction)")
axes[0].axhline(0.5, ls="--", color="gray", alpha=0.5)
axes[0].legend(fontsize=8)
axes[0].set_title("Effective AAVE exposure over time")

axes[1].set_ylabel("Pool value ($M)")
axes[1].legend(fontsize=8)
axes[1].set_title("Pool value over time")

plt.tight_layout()
plt.savefig("reclamm_exposure_comparison.png", dpi=150)
print("\nSaved reclamm_exposure_comparison.png")
