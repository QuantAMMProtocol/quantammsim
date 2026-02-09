"""Ablation A: drift init + drifting loss, NO drift features (drift_weight=0)."""

import sys
sys.path.insert(0, "/Users/matthew/Projects/quantammsim-synthetic-paths")

import jax
import jax.numpy as jnp
import numpy as np

from quantammsim.synthetic.training import (
    fit_latent_sde_drifting,
    compute_daily_log_prices,
)
from quantammsim.synthetic.generation import generate_latent_daily_paths
from quantammsim.utils.data_processing.historic_data_utils import get_historic_parquet_data

# --- Load data ---
tokens = ["ETH", "BTC", "USDC", "PAXG"]
data_root = "/Users/matthew/Projects/quantammsim/quantammsim/data"
print(f"Loading data for {tokens}...")
price_df = get_historic_parquet_data(tokens, cols=["close"], root=data_root)
close_cols = [f"close_{t}" for t in tokens]
minute_prices = price_df[close_cols].values.astype(np.float64)
valid_mask = ~np.any(np.isnan(minute_prices), axis=1)
first_valid = np.argmax(valid_mask)
last_valid = len(valid_mask) - np.argmax(valid_mask[::-1])
minute_prices = minute_prices[first_valid:last_valid]
n_assets = len(tokens)
n_days = minute_prices.shape[0] // 1440
minute_prices_jnp = jnp.array(minute_prices)
daily_log = compute_daily_log_prices(minute_prices_jnp)

real_daily_returns = jnp.diff(daily_log, axis=0)
real_drift = jnp.mean(real_daily_returns, axis=0)
real_vol = jnp.std(real_daily_returns, axis=0)
print(f"Data: {n_days} days, {n_assets} assets")
for i, t in enumerate(tokens):
    print(f"  {t}: drift={float(real_drift[i]):.6f}/day, vol={float(real_vol[i]):.6f}/day")

# --- Ablation A: drift init, NO drift features ---
print(f"\n=== ABLATION A: Drift init + drifting loss, NO drift features ===")
print(f"    (drift_weight=0.0, mc_samples=64, 500 steps)")
key = jax.random.PRNGKey(42)
latent_sde, history = fit_latent_sde_drifting(
    minute_prices_jnp,
    n_assets=n_assets,
    key=key,
    n_hidden=4,
    hidden_dim=32,
    window_lens=[10, 20, 50],
    depth=2,
    mc_samples=64,
    gen_batch_size=16,
    data_batch_size=256,
    antithetic=True,
    n_steps=500,
    lr=1e-3,
    weight_decay=1e-4,
    drift_weight=0.0,  # NO drift features
    patience=300,
    verbose=True,
)

# --- Generate paths and evaluate ---
print(f"\n=== Path evaluation ===")
y0 = daily_log[0]
key_eval = jax.random.PRNGKey(99)

for horizon in [10, 50, 100]:
    paths = generate_latent_daily_paths(
        latent_sde, y0, n_days=horizon, n_paths=500, key=key_eval,
    )
    y0_bc = jnp.broadcast_to(y0[:, None], (n_assets, 500))[None, ...]
    full = jnp.concatenate([y0_bc, paths], axis=0)
    returns = jnp.diff(full, axis=0)
    drift = jnp.mean(returns, axis=(0, 2))
    vol = jnp.mean(jnp.std(returns, axis=2), axis=0)

    print(f"\n  {horizon}-day paths:")
    for i, t in enumerate(tokens):
        rd = float(real_drift[i])
        d = float(drift[i])
        v = float(vol[i])
        rv = float(real_vol[i])
        ratio = d / rd if abs(rd) > 1e-8 else float('inf')
        print(f"    {t}: drift={d:.6f} ({ratio:.1f}x real), vol={v:.6f} ({v/rv:.2f}x real)")

# Loss curve
print(f"\n=== Loss curve ===")
for step_i in [0, 50, 100, 200, 300]:
    if step_i < len(history):
        print(f"  Step {step_i:4d}: train={history[step_i][0]:.6f}, val={history[step_i][1]:.6f}")
print(f"  Final:    train={history[-1][0]:.6f}, val={history[-1][1]:.6f}")
print(f"  Best val: {min(v for _, v in history):.6f}")
best_step = min(range(len(history)), key=lambda i: history[i][1])
print(f"  Best val at step: {best_step}")

# Drift diagnostic
print(f"\n=== Drift network output diagnostic ===")
test_points = daily_log[::100]
for idx in range(min(5, test_points.shape[0])):
    y = test_points[idx]
    z = latent_sde.encoder(y)
    f = latent_sde.drift(z)
    observed_drift = latent_sde.readout.weight @ f
    print(f"\n  Day ~{idx*100}:")
    print(f"    Observed drift (W @ f(Z)): {[f'{float(observed_drift[j]):.6f}' for j in range(n_assets)]}")
    print(f"    Real daily drift:          {[f'{float(real_drift[j]):.6f}' for j in range(n_assets)]}")
    print(f"    Ratio (model/real):        {[f'{float(observed_drift[j]/real_drift[j]):.1f}' if abs(float(real_drift[j])) > 1e-8 else 'N/A' for j in range(n_assets)]}")
