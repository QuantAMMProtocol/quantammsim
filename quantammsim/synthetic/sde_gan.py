"""Neural SDE-GAN for synthetic financial time series generation.

Implements the approach from Kidger et al. (2021) "Neural SDEs as
Infinite-Dimensional GANs" with minimal adaptations for multi-asset
daily log-price data.

Architecture follows the diffrax reference example almost verbatim:
- Generator: Neural SDE (drift + diffusion MLPs, VirtualBrownianTree)
- Discriminator: Neural CDE (reads paths via controlled differential equation)
- Training: WGAN with weight clipping, RMSprop, 10x initial updates

The generator produces cumulative log-return paths starting from 0.
At generation time, condition by adding observed log-prices: Y = y0 + gen(ts).

References:
    Kidger et al., "Neural SDEs as Infinite-Dimensional GANs", ICML 2021.
    Kidger et al., "Efficient and Accurate Gradients for Neural SDEs", NeurIPS 2021.
"""

from typing import Sequence, Tuple

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax


# ---------------------------------------------------------------------------
# Activation (Lipschitz-1, from Kidger)
# ---------------------------------------------------------------------------

def lipswish(x):
    return 0.909 * jnn.silu(x)


# ---------------------------------------------------------------------------
# Vector fields (from Kidger, verbatim)
# ---------------------------------------------------------------------------

class VectorField(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __init__(self, hidden_size, width_size, depth, scale, *, key, **kwargs):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(scale_key, (hidden_size,), minval=0.9, maxval=1.1)
        else:
            self.scale = jnp.ones(hidden_size)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )

    def __call__(self, t, y, args):
        return self.scale * self.mlp(jnp.concatenate([jnp.asarray(t)[None], y]))


class ControlledVectorField(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP
    control_size: int
    hidden_size: int

    def __init__(
        self, control_size, hidden_size, width_size, depth, scale, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        scale_key, mlp_key = jr.split(key)
        if scale:
            self.scale = jr.uniform(
                scale_key, (hidden_size, control_size), minval=0.9, maxval=1.1
            )
        else:
            self.scale = jnp.ones((hidden_size, control_size))
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size + 1,
            out_size=hidden_size * control_size,
            width_size=width_size,
            depth=depth,
            activation=lipswish,
            final_activation=jnn.tanh,
            key=mlp_key,
        )
        self.control_size = control_size
        self.hidden_size = hidden_size

    def __call__(self, t, y, args):
        return self.scale * self.mlp(jnp.concatenate([jnp.asarray(t)[None], y])).reshape(
            self.hidden_size, self.control_size
        )


# ---------------------------------------------------------------------------
# Generator: Neural SDE (adapted from Kidger for multi-asset)
# ---------------------------------------------------------------------------

class Generator(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField
    cvf: ControlledVectorField
    readout: eqx.nn.Linear
    initial_noise_size: int
    noise_size: int
    use_reversible_heun: bool

    def __init__(
        self,
        data_size: int,
        initial_noise_size: int = 5,
        noise_size: int = 3,
        hidden_size: int = 16,
        width_size: int = 16,
        depth: int = 1,
        use_reversible_heun: bool = False,
        *,
        key,
    ):
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)
        self.initial = eqx.nn.MLP(
            initial_noise_size, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=True, key=vf_key)
        self.cvf = ControlledVectorField(
            noise_size, hidden_size, width_size, depth, scale=True, key=cvf_key
        )
        self.readout = eqx.nn.Linear(hidden_size, data_size, key=readout_key)
        self.initial_noise_size = initial_noise_size
        self.noise_size = noise_size
        self.use_reversible_heun = use_reversible_heun

    def __call__(self, ts, *, key):
        t0, t1 = ts[0], ts[-1]
        dt0 = 1.0
        init_key, bm_key = jr.split(key)
        init = jr.normal(init_key, (self.initial_noise_size,))
        control = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=dt0 / 2, shape=(self.noise_size,), key=bm_key
        )
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        # ReversibleHeun: O(1) memory backprop (Kidger 2021), ideal for GPU.
        # Euler: simpler, faster JIT compile, fine for CPU.
        solver = diffrax.ReversibleHeun() if self.use_reversible_heun else diffrax.Euler()
        y0 = self.initial(init)
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, saveat=saveat)
        return jax.vmap(self.readout)(sol.ys)


# ---------------------------------------------------------------------------
# Discriminator: Neural CDE (from Kidger, verbatim)
# ---------------------------------------------------------------------------

class Discriminator(eqx.Module):
    initial: eqx.nn.MLP
    vf: VectorField
    cvf: ControlledVectorField
    readout: eqx.nn.Linear

    def __init__(
        self,
        data_size: int,
        hidden_size: int = 16,
        width_size: int = 16,
        depth: int = 1,
        *,
        key,
    ):
        initial_key, vf_key, cvf_key, readout_key = jr.split(key, 4)
        self.initial = eqx.nn.MLP(
            data_size + 1, hidden_size, width_size, depth, key=initial_key
        )
        self.vf = VectorField(hidden_size, width_size, depth, scale=False, key=vf_key)
        self.cvf = ControlledVectorField(
            data_size, hidden_size, width_size, depth, scale=False, key=cvf_key
        )
        self.readout = eqx.nn.Linear(hidden_size, 1, key=readout_key)

    def __call__(self, ts, ys):
        ys = diffrax.linear_interpolation(
            ts, ys, replace_nans_at_start=0.0, fill_forward_nans_at_end=True
        )
        init = jnp.concatenate([ts[0, None], ys[0]])
        control = diffrax.LinearInterpolation(ts, ys)
        vf = diffrax.ODETerm(self.vf)
        cvf = diffrax.ControlTerm(self.cvf, control)
        terms = diffrax.MultiTerm(vf, cvf)
        solver = diffrax.Euler()
        y0 = self.initial(init)
        saveat = diffrax.SaveAt(t0=True, t1=True)
        sol = diffrax.diffeqsolve(
            terms, solver, ts[0], ts[-1], 1.0, y0, saveat=saveat
        )
        return jax.vmap(self.readout)(sol.ys)

    @eqx.filter_jit
    def clip_weights(self):
        leaves, treedef = jax.tree_util.tree_flatten(
            self, is_leaf=lambda x: isinstance(x, eqx.nn.Linear)
        )
        new_leaves = []
        for leaf in leaves:
            if isinstance(leaf, eqx.nn.Linear):
                lim = 1 / leaf.out_features
                leaf = eqx.tree_at(
                    lambda x: x.weight, leaf, leaf.weight.clip(-lim, lim)
                )
            new_leaves.append(leaf)
        return jax.tree_util.tree_unflatten(treedef, new_leaves)


# ---------------------------------------------------------------------------
# Training (from Kidger, adapted for our data)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _wgan_loss(generator, discriminator, ts_i, ys_i, key, step):
    """WGAN loss for evaluation: E[D(real)] - E[D(fake)]."""
    batch_size = ts_i.shape[0]
    key = jr.fold_in(key, step)
    keys = jr.split(key, batch_size)
    fake_ys = jax.vmap(generator)(ts_i, key=keys)
    real_score = jax.vmap(discriminator)(ts_i, ys_i)
    fake_score = jax.vmap(discriminator)(ts_i, fake_ys)
    return jnp.mean(real_score - fake_score)


def _increase_update_initial(updates):
    get_initial_leaves = lambda u: jax.tree_util.tree_leaves(u.initial)
    return eqx.tree_at(get_initial_leaves, updates, replace_fn=lambda x: x * 10)


@eqx.filter_grad
def _grad_loss(g_d, ts_i, ys_i, key, step, real_mean_return, drift_lambda):
    """Joint gradient: WGAN loss + drift penalty on generator mean return.

    The drift penalty only depends on generator output, so
    d(penalty)/d(discriminator) = 0 — it adds gradient to G only.
    """
    generator, discriminator = g_d
    batch_size = ts_i.shape[0]
    key = jr.fold_in(key, step)
    keys = jr.split(key, batch_size)
    fake_ys = jax.vmap(generator)(ts_i, key=keys)
    real_score = jax.vmap(discriminator)(ts_i, ys_i)
    fake_score = jax.vmap(discriminator)(ts_i, fake_ys)
    wgan = jnp.mean(real_score - fake_score)
    # Drift moment-matching: penalize generator's mean return deviating from real
    gen_returns = jnp.diff(fake_ys, axis=1)
    gen_mean = jnp.mean(gen_returns, axis=(0, 1))
    drift_pen = drift_lambda * jnp.sum((gen_mean - real_mean_return) ** 2)
    return wgan + drift_pen


@eqx.filter_jit
def _make_step(
    generator, discriminator, g_opt_state, d_opt_state,
    g_optim, d_optim, ts_i, ys_i, key, step,
    real_mean_return, drift_lambda,
):
    g_grad, d_grad = _grad_loss(
        (generator, discriminator), ts_i, ys_i, key, step,
        real_mean_return, drift_lambda,
    )
    g_updates, g_opt_state = g_optim.update(g_grad, g_opt_state)
    d_updates, d_opt_state = d_optim.update(d_grad, d_opt_state)
    g_updates = _increase_update_initial(g_updates)
    d_updates = _increase_update_initial(d_updates)
    generator = eqx.apply_updates(generator, g_updates)
    discriminator = eqx.apply_updates(discriminator, d_updates)
    discriminator = discriminator.clip_weights()
    return generator, discriminator, g_opt_state, d_opt_state


def _extract_windows(daily_log_prices: jnp.ndarray, window_len: int) -> jnp.ndarray:
    """Extract cumulative-return windows from daily log-prices.

    Returns (n_windows, window_len, n_assets) array where each window
    starts at 0 (cumulative return from window start).
    """
    n_days, n_assets = daily_log_prices.shape
    n_windows = n_days - window_len + 1
    # Build windows using lax.dynamic_slice for JIT compatibility
    def get_window(i):
        w = jax.lax.dynamic_slice(daily_log_prices, (i, 0), (window_len, n_assets))
        return w - w[0]  # cumulative returns from 0
    return jax.vmap(get_window)(jnp.arange(n_windows))


def compute_daily_log_prices(minute_prices: jnp.ndarray) -> jnp.ndarray:
    """Convert minute prices to daily log-prices (close of each day)."""
    n_minutes = minute_prices.shape[0]
    n_days = n_minutes // 1440
    daily = minute_prices[1439::1440][:n_days]
    return jnp.log(daily)


def train_sde_gan(
    minute_prices: jnp.ndarray = None,
    n_assets: int = None,
    key: jax.Array = None,
    *,
    daily_log_prices: jnp.ndarray = None,
    window_len: int = 50,
    initial_noise_size: int = 5,
    noise_size: int = 3,
    hidden_size: int = 16,
    width_size: int = 16,
    depth: int = 1,
    generator_lr: float = 2e-5,
    discriminator_lr: float = 1e-4,
    batch_size: int = 1024,
    n_steps: int = 10000,
    n_critic: int = 1,
    drift_lambda: float = 0.0,
    use_reversible_heun: bool = False,
    val_fraction: float = 0.2,
    verbose: bool = True,
    checkpoint_fn=None,
    checkpoint_every: int = 2000,
) -> Tuple[Generator, jnp.ndarray, list]:
    """Train a Neural SDE generator via WGAN with a Neural CDE discriminator.

    Args:
        minute_prices: (T_minutes, n_assets) raw price array. Converted to
            daily log-prices internally. Provide this OR daily_log_prices.
        n_assets: Number of assets. Inferred from data if not provided.
        key: JAX PRNG key.
        daily_log_prices: (T_days, n_assets) pre-computed daily log-prices.
            Use this when you already have daily data (e.g. from yfinance)
            to avoid needing minute-resolution parquets.
        window_len: Length of daily windows for training.
        initial_noise_size: Dimension of initial random noise.
        noise_size: Dimension of Brownian motion driving noise.
        hidden_size: Hidden state dimension for generator and discriminator.
        width_size: MLP hidden layer width.
        depth: MLP depth.
        generator_lr: Generator learning rate (RMSprop).
        discriminator_lr: Discriminator learning rate (RMSprop).
        batch_size: Batch size for training.
        n_steps: Number of generator training steps.
        n_critic: Discriminator steps per generator step.
        drift_lambda: Weight on drift moment-matching penalty (0=pure WGAN).
        use_reversible_heun: Use ReversibleHeun solver (O(1) memory, GPU).
        val_fraction: Fraction of data held out for validation.
        verbose: Print training progress.
        checkpoint_fn: Optional callback called every checkpoint_every steps
            with (generator, vol_scale, step). Use for saving to Drive etc.
        checkpoint_every: Steps between checkpoint_fn calls (default 2000).

    Returns:
        (trained_generator, vol_scale, loss_history) where vol_scale is
        (n_assets,) per-asset normalization factor, and loss_history is
        list of (train_loss, val_loss) tuples.
    """
    # Prepare daily log-price windows
    if daily_log_prices is not None:
        daily_log = daily_log_prices
    elif minute_prices is not None:
        daily_log = compute_daily_log_prices(minute_prices)
    else:
        raise ValueError("Provide either minute_prices or daily_log_prices")
    if n_assets is None:
        n_assets = daily_log.shape[1]
    n_days = daily_log.shape[0]
    windows = _extract_windows(daily_log, window_len)
    n_windows = windows.shape[0]

    # Normalize by per-asset daily vol so all assets have similar scale.
    # This helps the GAN — without it, USDC (vol 0.001) and ETH (vol 0.04)
    # differ by 40x, making training unstable.
    daily_returns = jnp.diff(daily_log, axis=0)
    vol_scale = jnp.std(daily_returns, axis=0)  # (n_assets,)
    vol_scale = jnp.maximum(vol_scale, 1e-8)  # avoid division by zero
    windows = windows / vol_scale[None, None, :]  # normalize

    # Train/val split (temporal)
    n_val = max(int(n_windows * val_fraction), 1)
    train_windows = windows[:-n_val]
    val_windows = windows[-n_val:]
    n_train = train_windows.shape[0]

    # Timestamps for windows (0, 1, ..., window_len-1)
    ts = jnp.arange(window_len, dtype=jnp.float32)
    # Broadcast ts to batch: (batch, window_len)
    ts_batch_template = jnp.broadcast_to(ts, (batch_size, window_len))

    # Precompute real mean return in normalized space for drift penalty
    real_returns_norm = jnp.diff(train_windows, axis=1)
    real_mean_return = jnp.mean(real_returns_norm, axis=(0, 1))  # (n_assets,)

    if verbose:
        print(f"[SDE-GAN] {n_days} days -> {n_windows} windows (len={window_len})")
        print(f"[SDE-GAN] train: {n_train}, val: {n_val}")
        print(f"[SDE-GAN] vol_scale: {[f'{float(v):.6f}' for v in vol_scale]}")
        print(f"[SDE-GAN] data_size={n_assets}, hidden={hidden_size}, noise={noise_size}")
        print(f"[SDE-GAN] G lr={generator_lr}, D lr={discriminator_lr}, batch={batch_size}")
        print(f"[SDE-GAN] n_critic={n_critic}, drift_lambda={drift_lambda}")
        if drift_lambda > 0:
            print(f"[SDE-GAN] real mean return (norm): {[f'{float(v):.6f}' for v in real_mean_return]}")

    # Build models
    key, g_key, d_key = jr.split(key, 3)
    generator = Generator(
        n_assets, initial_noise_size, noise_size,
        hidden_size, width_size, depth,
        use_reversible_heun=use_reversible_heun, key=g_key,
    )
    discriminator = Discriminator(
        n_assets, hidden_size, width_size, depth, key=d_key,
    )

    # Optimizers (note: negative lr for discriminator in WGAN — gradient ascent)
    g_optim = optax.rmsprop(generator_lr)
    d_optim = optax.rmsprop(-discriminator_lr)
    g_opt_state = g_optim.init(eqx.filter(generator, eqx.is_inexact_array))
    d_opt_state = d_optim.init(eqx.filter(discriminator, eqx.is_inexact_array))

    history = []
    steps_per_print = max(n_steps // 20, 1)

    for step in range(n_steps):
        # Sample batch of real windows
        key, batch_key, step_key = jr.split(key, 3)
        idx = jr.randint(batch_key, (batch_size,), 0, n_train)
        ys_batch = train_windows[idx]

        generator, discriminator, g_opt_state, d_opt_state = _make_step(
            generator, discriminator, g_opt_state, d_opt_state,
            g_optim, d_optim, ts_batch_template, ys_batch, step_key,
            jnp.asarray(step), real_mean_return, drift_lambda,
        )

        if step % steps_per_print == 0 or step == n_steps - 1:
            # Eval on val
            key, val_key = jr.split(key)
            val_idx = jr.randint(val_key, (min(batch_size, n_val),), 0, n_val)
            val_ys = val_windows[val_idx]
            val_ts = jnp.broadcast_to(ts, val_ys.shape[:2])
            key, eval_key = jr.split(key)
            train_loss = float(_wgan_loss(
                generator, discriminator, ts_batch_template, ys_batch,
                eval_key, jnp.asarray(step),
            ))
            key, eval_key2 = jr.split(key)
            val_loss = float(_wgan_loss(
                generator, discriminator, val_ts, val_ys,
                eval_key2, jnp.asarray(step),
            ))
            history.append((train_loss, val_loss))
            if verbose:
                print(f"  step {step:5d} | train: {train_loss:.6f} | val: {val_loss:.6f}")

        if checkpoint_fn is not None and step > 0 and step % checkpoint_every == 0:
            checkpoint_fn(generator, vol_scale, step)

    return generator, vol_scale, history


# ---------------------------------------------------------------------------
# Generation (post-training)
# ---------------------------------------------------------------------------

def generate_paths(
    generator: Generator,
    vol_scale: jnp.ndarray,
    y0: jnp.ndarray,
    n_days: int,
    n_paths: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Generate synthetic daily log-price paths from a trained generator.

    The generator produces normalized cumulative return paths (unit vol).
    We denormalize by vol_scale and add y0 to get absolute log-prices.

    Args:
        generator: Trained Generator.
        vol_scale: (n_assets,) per-asset daily vol used for normalization.
        y0: (n_assets,) starting log-prices.
        n_days: Number of days to generate.
        n_paths: Number of independent paths.
        key: JAX PRNG key.

    Returns:
        (n_days, n_assets, n_paths) array of daily log-prices.
    """
    ts = jnp.arange(n_days, dtype=jnp.float32)
    ts_batch = jnp.broadcast_to(ts, (n_paths, n_days))
    keys = jr.split(key, n_paths)
    # (n_paths, n_days, n_assets) normalized cumulative returns
    cum_returns_norm = jax.vmap(generator)(ts_batch, key=keys)
    # Denormalize and add y0
    cum_returns = cum_returns_norm * vol_scale[None, None, :]
    log_prices = cum_returns + y0[None, :]
    return jnp.transpose(log_prices, (1, 2, 0))
