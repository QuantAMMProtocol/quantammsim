"""Ablation D: drift init only, NO training (just evaluate the initialized model).

Tests how much training actually improves over the correctly-initialized model.
The drift init sets the drift network's output scale to match empirical drift,
so the initial model should already have O(0.001) drift. This measures whether
training meaningfully adjusts drift direction/magnitude or just refines vol.
"""

import sys
sys.path.insert(0, "/Users/matthew/Projects/quantammsim-synthetic-paths")

import jax
import jax.numpy as jnp
import numpy as np

from quantammsim.synthetic.model import LatentNeuralSDE
from quantammsim.synthetic.training import compute_daily_log_prices
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

# --- Build initialized (untrained) model ---
print(f"\n=== ABLATION D: Drift init, NO training ===")
print(f"    (just evaluate the initialized model)")

# Compute init scales exactly as fit_latent_sde_drifting does
train_daily = daily_log[:int(len(daily_log) * 0.8)]
daily_returns = jnp.diff(train_daily, axis=0)
daily_std = jnp.std(daily_returns, axis=0)
daily_drift_emp = jnp.mean(daily_returns, axis=0)
drift_mag = jnp.maximum(jnp.abs(daily_drift_emp), daily_std * 0.01)
n_hidden = 4
hidden_drift_scale = jnp.full(n_hidden, float(jnp.mean(drift_mag)))
init_drift_scale = jnp.concatenate([drift_mag, hidden_drift_scale])

print(f"  Diffusion init scale: {[f'{float(daily_std[j]):.6f}' for j in range(n_assets)]}")
print(f"  Drift init scale:     {[f'{float(init_drift_scale[j]):.6f}' for j in range(n_assets)]}")

key = jax.random.PRNGKey(42)
latent_sde = LatentNeuralSDE(
    n_assets, n_hidden, hidden_dim=32,
    init_diffusion_scale=daily_std,
    init_drift_scale=init_drift_scale,
    key=key,
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
