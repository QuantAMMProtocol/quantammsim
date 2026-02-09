"""SDE-GAN v2: drift moment-matching penalty for better drift calibration.

Experimental results (4 assets, 1977 days, window=50, batch=64, hidden=16):

  drift_lambda=0.0 (10k steps, baseline):
    10d: ETH 3.1x, BTC 5.1x, PAXG -0.6x
    Vol: ETH 0.94x, BTC 1.24x, PAXG 0.91x

  drift_lambda=1.0 (15k steps, BEST at 10d):
    10d: ETH 1.8x, BTC 0.9x, PAXG 0.5x    <-- best mean error
    Vol: ETH 0.98x, BTC 0.95x, PAXG 0.78x

  drift_lambda=0.1 (20k steps):
    10d: ETH 1.6x, BTC 1.7x, PAXG 0.1x
    Vol: ETH 0.99x, BTC 1.00x, PAXG 0.88x  <-- best vol matching

  drift_lambda=0.5 (20k steps):
    10d: ETH 1.2x, BTC 3.2x, PAXG 1.4x
    50d: ETH -2.2x, BTC 1.8x, PAXG 1.0x    <-- best PAXG
"""

import sys
sys.path.insert(0, "/Users/matthew/Projects/quantammsim-synthetic-paths")

import jax
import jax.numpy as jnp
import numpy as np

from quantammsim.synthetic.sde_gan import (
    train_sde_gan,
    generate_paths,
    compute_daily_log_prices,
)
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

# --- Train SDE-GAN v2 ---
# drift_lambda=1.0 gives best mean drift error at 10d across assets.
# Vol matching is good (0.78-0.98x) but slightly worse than lambda=0.1.
# Training loss is volatile but converges to good results.
DRIFT_LAMBDA = 1.0
N_STEPS = 15000

print(f"\n=== Training SDE-GAN v2 (drift_lambda={DRIFT_LAMBDA}) ===")
key = jax.random.PRNGKey(42)
generator, vol_scale, history = train_sde_gan(
    minute_prices_jnp,
    n_assets=n_assets,
    key=key,
    window_len=50,
    initial_noise_size=5,
    noise_size=3,
    hidden_size=16,
    width_size=16,
    depth=1,
    generator_lr=2e-5,
    discriminator_lr=1e-4,
    batch_size=64,
    n_steps=N_STEPS,
    drift_lambda=DRIFT_LAMBDA,
    verbose=True,
)

# --- Generate paths and evaluate ---
print(f"\n=== Path evaluation ===")
y0 = daily_log[0]
key_eval = jax.random.PRNGKey(99)

for horizon in [10, 50, 100]:
    paths = generate_paths(generator, vol_scale, y0, n_days=horizon, n_paths=500, key=key_eval)
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
