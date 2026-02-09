"""Synthetic price path generation via trained Neural SDE.

Two generation pipelines:

1. **Minute-resolution** (MLE-trained models): Euler-Maruyama scan at minute dt.
   No interpolation needed — generates natively at minute resolution.

2. **Daily-resolution + Brownian bridge** (Sig-W1-trained models): Euler-Maruyama
   at daily dt, then Brownian bridge interpolation to fill 1440 minutes between
   each pair of daily observations. Endpoints are preserved exactly.

Output shape (T_minutes, n_assets, n_paths) matches the existing MC convention
used by windowing_utils and forward_pass.
"""

import jax
import jax.numpy as jnp

from .model import NeuralSDE, LatentNeuralSDE


def generate_minute_paths(
    sde: NeuralSDE,
    y0: jnp.ndarray,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
    dt: float = 1.0,
    antithetic: bool = False,
) -> jnp.ndarray:
    """Euler-Maruyama simulation via jax.lax.scan.

    Y_{t+1} = Y_t + mu(Y_t)*dt + L(Y_t) @ (sqrt(dt) * Z_t)

    For diffusion-only SDEs (learn_drift=False), mu(Y_t) = 0.

    Args:
        sde: Trained Neural SDE.
        y0: (n_assets,) initial log-prices.
        n_steps: Number of steps to simulate.
        n_paths: Number of independent paths.
        key: JAX PRNG key.
        dt: Time step size (1.0 = per-minute or per-day depending on context).
        antithetic: If True, use antithetic sampling: generate ceil(n_paths/2)
            noise draws and simulate with both +Z and -Z. The odd-order MC
            noise cancels exactly in the path mean, halving the variance of
            drift (level-1 signature) estimates.

    Returns:
        (n_steps, n_assets, n_paths) array of simulated log-prices
        (excludes the initial condition y0).
    """
    sqrt_dt = jnp.sqrt(dt)
    n_assets = y0.shape[0]

    def _step(y, z):
        mu = sde.drift(y)
        L = sde.diffusion(y)
        y_next = y + mu * dt + L @ (sqrt_dt * z)
        return y_next, y_next

    def _simulate_with_noise(noise):
        _, path = jax.lax.scan(_step, y0, noise)
        return path  # (n_steps, n_assets)

    if antithetic:
        n_base = (n_paths + 1) // 2
        keys = jax.random.split(key, n_base)
        base_noise = jax.vmap(
            lambda k: jax.random.normal(k, (n_steps, n_assets))
        )(keys)  # (n_base, n_steps, n_assets)
        # Concatenate +Z and -Z, truncate to exactly n_paths
        all_noise = jnp.concatenate([base_noise, -base_noise], axis=0)[:n_paths]
    else:
        keys = jax.random.split(key, n_paths)
        all_noise = jax.vmap(
            lambda k: jax.random.normal(k, (n_steps, n_assets))
        )(keys)  # (n_paths, n_steps, n_assets)

    all_paths = jax.vmap(_simulate_with_noise)(all_noise)  # (n_paths, n_steps, n_assets)
    return jnp.transpose(all_paths, (1, 2, 0))  # (n_steps, n_assets, n_paths)


def generate_synthetic_price_array(
    sde: NeuralSDE,
    historical_minute_prices: jnp.ndarray,
    n_paths: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Full pipeline: historical prices -> synthetic (T_minutes, n_assets, n_paths).

    Steps:
    1. Convert historical prices to log-prices.
    2. Use first minute as initial condition.
    3. Simulate via Euler-Maruyama at minute resolution.
    4. exp() back to price space.

    Args:
        sde: Trained Neural SDE.
        historical_minute_prices: (T_minutes, n_assets) historical prices.
        n_paths: Number of synthetic paths to generate.
        key: JAX PRNG key.

    Returns:
        (T_minutes, n_assets, n_paths) synthetic price array matching
        the length of historical_minute_prices.
    """
    log_prices = jnp.log(historical_minute_prices)
    y0 = log_prices[0]  # (n_assets,)
    n_steps = log_prices.shape[0] - 1

    # Simulate: (n_steps, n_assets, n_paths)
    log_paths = generate_minute_paths(sde, y0, n_steps, n_paths, key)

    # Prepend initial condition: (1, n_assets, n_paths)
    y0_broadcast = jnp.broadcast_to(y0[:, None], (y0.shape[0], n_paths))[None, ...]
    log_paths = jnp.concatenate([y0_broadcast, log_paths], axis=0)

    return jnp.exp(log_paths)


def generate_daily_paths(
    sde: NeuralSDE,
    y0: jnp.ndarray,
    n_days: int,
    n_paths: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Generate daily-resolution log-price paths via Euler-Maruyama with dt=1 day.

    Thin wrapper around generate_minute_paths with dt=1.0.

    Args:
        sde: Trained Neural SDE (typically Sig-W1-trained with drift).
        y0: (n_assets,) initial daily log-prices.
        n_days: Number of daily steps to simulate.
        n_paths: Number of independent paths.
        key: JAX PRNG key.

    Returns:
        (n_days, n_assets, n_paths) array of simulated daily log-prices
        (excludes the initial condition y0).
    """
    return generate_minute_paths(sde, y0, n_steps=n_days, n_paths=n_paths, key=key, dt=1.0)


def brownian_bridge_interpolate(
    daily_log_prices: jnp.ndarray,
    minutes_per_day: int = 1440,
    intraday_vol: jnp.ndarray = None,
    key: jax.Array = None,
) -> jnp.ndarray:
    """Interpolate daily log-prices to minute resolution via Brownian bridge.

    Given daily log-price endpoints y_d and y_{d+1}, the Brownian bridge
    fills in minute-resolution values:

        y(t) = (1 - t/T) * y_d + (t/T) * y_{d+1} + sigma * sqrt(t*(T-t)/T) * Z

    where T = minutes_per_day, t = 1, ..., T-1, and Z ~ N(0, 1).

    Endpoints y_d and y_{d+1} are preserved exactly.

    Args:
        daily_log_prices: (n_days, n_assets) or (n_days, n_assets, n_paths).
        minutes_per_day: Number of intraday points (default 1440).
        intraday_vol: (n_assets,) per-asset intraday volatility (std of minute
            log-returns). If None, gives deterministic linear interpolation.
        key: JAX PRNG key for stochastic bridge. If None, deterministic.

    Returns:
        (n_minutes, ...) minute-resolution log-prices. Shape:
        - If input is (n_days, n_assets): output is (n_minutes, n_assets)
        - If input is (n_days, n_assets, n_paths): output is (n_minutes, n_assets, n_paths)
        where n_minutes = (n_days - 1) * minutes_per_day + 1.
    """
    has_paths = daily_log_prices.ndim == 3
    if not has_paths:
        # Add a dummy path dimension for uniform handling
        daily_log_prices = daily_log_prices[..., None]  # (n_days, n_assets, 1)

    n_days, n_assets, n_paths = daily_log_prices.shape
    T = minutes_per_day

    # Fractional times within a day: t/T for t = 0, 1, ..., T
    # We need T+1 points per day (inclusive of both endpoints), then stitch
    frac = jnp.arange(T + 1) / T  # (T+1,)

    # Linear interpolation component for each day gap
    # Shape broadcasting: (n_days-1, T+1, n_assets, n_paths)
    y_start = daily_log_prices[:-1]  # (n_days-1, n_assets, n_paths)
    y_end = daily_log_prices[1:]     # (n_days-1, n_assets, n_paths)

    # (n_days-1, T+1, n_assets, n_paths)
    linear = (
        y_start[:, None, :, :] * (1 - frac[None, :, None, None])
        + y_end[:, None, :, :] * frac[None, :, None, None]
    )

    # Stochastic component
    if key is not None and intraday_vol is not None:
        # Bridge variance at fractional time f: sigma^2 * f * (1 - f)
        # (but we use sigma * sqrt(f*(1-f)) * Z)
        bridge_std = jnp.sqrt(
            frac * (1 - frac)
        )  # (T+1,) — zero at endpoints

        # Scale by per-asset vol: (T+1, n_assets)
        bridge_std_scaled = bridge_std[:, None] * intraday_vol[None, :]  # (T+1, n_assets)

        # Sample noise: (n_days-1, T+1, n_assets, n_paths)
        noise = jax.random.normal(
            key, (n_days - 1, T + 1, n_assets, n_paths)
        )

        stochastic = bridge_std_scaled[None, :, :, None] * noise
    else:
        stochastic = 0.0

    minute_log_prices = linear + stochastic  # (n_days-1, T+1, n_assets, n_paths)

    # Stitch days together: take all T+1 points from day 0, then points 1..T
    # from subsequent days (to avoid duplicating endpoints)
    first_day = minute_log_prices[0]  # (T+1, n_assets, n_paths)
    rest = minute_log_prices[1:, 1:]  # (n_days-2, T, n_assets, n_paths)

    if n_days > 2:
        rest_flat = rest.reshape(-1, n_assets, n_paths)  # ((n_days-2)*T, n_assets, n_paths)
        result = jnp.concatenate([first_day, rest_flat], axis=0)
    else:
        result = first_day  # Only one day gap

    if not has_paths:
        result = result[..., 0]  # Remove dummy path dimension

    return result


def generate_synthetic_price_array_daily(
    sde: NeuralSDE,
    historical_minute_prices: jnp.ndarray,
    n_paths: int,
    key: jax.Array,
    intraday_vol: jnp.ndarray = None,
    chunk_period: int = 1440,
    generation_chunk_days: int = 90,
) -> jnp.ndarray:
    """Full pipeline: daily SDE -> daily paths -> Brownian bridge -> minute prices.

    Generates in chunks of ``generation_chunk_days``, resetting to real daily
    prices at each chunk boundary. This prevents drift accumulation over long
    horizons — the SDE is trained on 10-50 day windows and should only be asked
    to extrapolate over comparable timescales.

    Each chunk:
    1. Starts from the real daily log-price at the chunk's first day.
    2. Simulates ``generation_chunk_days`` daily steps via Euler-Maruyama.
    3. Brownian-bridge interpolates to minute resolution.

    Chunks are stitched at their boundaries. The synthetic path deviates from
    reality within each chunk (different per-path) but snaps back to the real
    level at chunk boundaries, giving realistic local dynamics without
    compounding drift errors over years.

    If intraday_vol is None, it is estimated from the historical minute data
    as the mean std of minute log-returns across days.

    Args:
        sde: Trained Neural SDE (Sig-W1 trained with drift).
        historical_minute_prices: (T_minutes, n_assets) historical prices.
        n_paths: Number of synthetic paths.
        key: JAX PRNG key.
        intraday_vol: (n_assets,) per-asset intraday vol. Estimated if None.
        chunk_period: Minutes per day (default 1440).
        generation_chunk_days: Days per generation chunk (default 90).
            Longer = more drift freedom, shorter = more anchored to real prices.

    Returns:
        (T_minutes, n_assets, n_paths) synthetic price array.
    """
    from .training import compute_daily_log_prices

    daily_log_prices = compute_daily_log_prices(historical_minute_prices, chunk_period)
    n_days = daily_log_prices.shape[0]
    n_assets = daily_log_prices.shape[1]

    # Estimate intraday vol if not provided
    if intraday_vol is None:
        log_prices = jnp.log(historical_minute_prices)
        n_full_minutes = n_days * chunk_period
        log_reshaped = log_prices[:n_full_minutes].reshape(n_days, chunk_period, n_assets)
        minute_returns = jnp.diff(log_reshaped, axis=1)  # (n_days, chunk_period-1, n_assets)
        intraday_vol = jnp.mean(jnp.std(minute_returns, axis=1), axis=0)  # (n_assets,)

    # Generate chunk by chunk
    chunk_minute_pieces = []
    chunk_start = 0

    while chunk_start < n_days - 1:
        chunk_end = min(chunk_start + generation_chunk_days, n_days - 1)
        chunk_len = chunk_end - chunk_start  # daily steps in this chunk

        # Initial condition: real daily log-price at chunk start
        y0 = daily_log_prices[chunk_start]  # (n_assets,)

        key, key_daily, key_bridge = jax.random.split(key, 3)
        daily_log_paths = generate_daily_paths(
            sde, y0, n_days=chunk_len, n_paths=n_paths, key=key_daily
        )  # (chunk_len, n_assets, n_paths)

        # Prepend initial condition
        y0_broadcast = jnp.broadcast_to(y0[:, None], (n_assets, n_paths))[None, ...]
        daily_with_y0 = jnp.concatenate(
            [y0_broadcast, daily_log_paths], axis=0
        )  # (chunk_len+1, n_assets, n_paths)

        # Brownian bridge interpolation
        chunk_minutes = brownian_bridge_interpolate(
            daily_with_y0, minutes_per_day=chunk_period,
            intraday_vol=intraday_vol, key=key_bridge,
        )  # (chunk_len*chunk_period + 1, n_assets, n_paths)

        # For stitching: first chunk keeps all points, subsequent chunks
        # drop the first point (which duplicates the previous chunk's last)
        if chunk_start == 0:
            chunk_minute_pieces.append(chunk_minutes)
        else:
            chunk_minute_pieces.append(chunk_minutes[1:])

        chunk_start = chunk_end

    minute_log_prices = jnp.concatenate(chunk_minute_pieces, axis=0)

    # Truncate or pad to match historical length
    T_hist = historical_minute_prices.shape[0]
    T_gen = minute_log_prices.shape[0]

    if T_gen >= T_hist:
        minute_log_prices = minute_log_prices[:T_hist]
    else:
        pad_len = T_hist - T_gen
        last_val = minute_log_prices[-1:].repeat(pad_len, axis=0)
        minute_log_prices = jnp.concatenate([minute_log_prices, last_val], axis=0)

    return jnp.exp(minute_log_prices)


# ---------------------------------------------------------------------------
# Latent Neural SDE generation
# ---------------------------------------------------------------------------


def generate_latent_daily_paths(
    sde: LatentNeuralSDE,
    y0: jnp.ndarray,
    n_days: int,
    n_paths: int,
    key: jax.Array,
    antithetic: bool = False,
) -> jnp.ndarray:
    """Euler-Maruyama in latent space, readout to observed space.

    1. Encode: Z_0 = sde.encoder(Y_0)
    2. Scan: Z_{t+1} = Z_t + f(Z_t)*dt + L(Z_t) @ (sqrt(dt) * noise)
    3. Readout: Y_t = sde.readout(Z_t) for each step

    Args:
        sde: Trained LatentNeuralSDE.
        y0: (n_assets,) observed initial log-prices.
        n_days: Number of daily steps to simulate.
        n_paths: Number of independent paths.
        key: JAX PRNG key.
        antithetic: If True, use antithetic sampling.

    Returns:
        (n_days, n_assets, n_paths) observed log-prices (excludes initial condition).
    """
    latent_dim = sde.latent_dim
    n_assets = sde.n_assets
    dt = 1.0
    sqrt_dt = jnp.sqrt(dt)

    # Encode Y_0 -> Z_0
    z0 = sde.encoder(y0)  # (latent_dim,)

    def _step(z, noise):
        f = sde.drift(z)
        L = sde.diffusion(z)
        z_next = z + f * dt + L @ (sqrt_dt * noise)
        y = sde.readout(z_next)
        return z_next, y

    def _simulate_with_noise(noise):
        # noise: (n_days, latent_dim)
        _, y_path = jax.lax.scan(_step, z0, noise)
        return y_path  # (n_days, n_assets)

    if antithetic:
        n_base = (n_paths + 1) // 2
        keys = jax.random.split(key, n_base)
        base_noise = jax.vmap(
            lambda k: jax.random.normal(k, (n_days, latent_dim))
        )(keys)
        all_noise = jnp.concatenate([base_noise, -base_noise], axis=0)[:n_paths]
    else:
        keys = jax.random.split(key, n_paths)
        all_noise = jax.vmap(
            lambda k: jax.random.normal(k, (n_days, latent_dim))
        )(keys)

    all_paths = jax.vmap(_simulate_with_noise)(all_noise)  # (n_paths, n_days, n_assets)
    return jnp.transpose(all_paths, (1, 2, 0))  # (n_days, n_assets, n_paths)


def generate_synthetic_price_array_latent(
    sde: LatentNeuralSDE,
    historical_minute_prices: jnp.ndarray,
    n_paths: int,
    key: jax.Array,
    intraday_vol: jnp.ndarray = None,
    chunk_period: int = 1440,
    generation_chunk_days: int = 90,
) -> jnp.ndarray:
    """Full pipeline: latent SDE -> daily paths -> Brownian bridge -> minute prices.

    Identical chunking strategy to generate_synthetic_price_array_daily, but uses
    the latent SDE for daily-step generation. Resets to real daily prices at chunk
    boundaries to prevent drift accumulation.

    Args:
        sde: Trained LatentNeuralSDE.
        historical_minute_prices: (T_minutes, n_assets) historical prices.
        n_paths: Number of synthetic paths.
        key: JAX PRNG key.
        intraday_vol: (n_assets,) per-asset intraday vol. Estimated if None.
        chunk_period: Minutes per day (default 1440).
        generation_chunk_days: Days per generation chunk (default 90).

    Returns:
        (T_minutes, n_assets, n_paths) synthetic price array.
    """
    from .training import compute_daily_log_prices

    daily_log_prices = compute_daily_log_prices(historical_minute_prices, chunk_period)
    n_days = daily_log_prices.shape[0]
    n_assets = daily_log_prices.shape[1]

    # Estimate intraday vol if not provided
    if intraday_vol is None:
        log_prices = jnp.log(historical_minute_prices)
        n_full_minutes = n_days * chunk_period
        log_reshaped = log_prices[:n_full_minutes].reshape(n_days, chunk_period, n_assets)
        minute_returns = jnp.diff(log_reshaped, axis=1)
        intraday_vol = jnp.mean(jnp.std(minute_returns, axis=1), axis=0)

    # Generate chunk by chunk
    chunk_minute_pieces = []
    chunk_start = 0

    while chunk_start < n_days - 1:
        chunk_end = min(chunk_start + generation_chunk_days, n_days - 1)
        chunk_len = chunk_end - chunk_start

        y0 = daily_log_prices[chunk_start]

        key, key_daily, key_bridge = jax.random.split(key, 3)
        daily_log_paths = generate_latent_daily_paths(
            sde, y0, n_days=chunk_len, n_paths=n_paths, key=key_daily
        )

        # Prepend initial condition
        y0_broadcast = jnp.broadcast_to(y0[:, None], (n_assets, n_paths))[None, ...]
        daily_with_y0 = jnp.concatenate(
            [y0_broadcast, daily_log_paths], axis=0
        )

        # Brownian bridge interpolation
        chunk_minutes = brownian_bridge_interpolate(
            daily_with_y0, minutes_per_day=chunk_period,
            intraday_vol=intraday_vol, key=key_bridge,
        )

        if chunk_start == 0:
            chunk_minute_pieces.append(chunk_minutes)
        else:
            chunk_minute_pieces.append(chunk_minutes[1:])

        chunk_start = chunk_end

    minute_log_prices = jnp.concatenate(chunk_minute_pieces, axis=0)

    # Truncate or pad to match historical length
    T_hist = historical_minute_prices.shape[0]
    T_gen = minute_log_prices.shape[0]

    if T_gen >= T_hist:
        minute_log_prices = minute_log_prices[:T_hist]
    else:
        pad_len = T_hist - T_gen
        last_val = minute_log_prices[-1:].repeat(pad_len, axis=0)
        minute_log_prices = jnp.concatenate([minute_log_prices, last_val], axis=0)

    return jnp.exp(minute_log_prices)
