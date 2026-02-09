"""Training a Neural SDE via MLE or Sig-W1 loss.

Two training modes:

1. **MLE (minute resolution)**: Euler-Maruyama maximum likelihood.
   At minute resolution, the SDE transition (with zero drift) is:
       Y_{t+1} | Y_t ~ N(Y_t, L(Y_t) @ L(Y_t)^T * dt)
   This is a heteroscedastic Gaussian regression — no SDE solver needed.
   ~2M observations → excellent diffusion estimates.

2. **Sig-W1 (daily resolution)**: Signature Wasserstein-1 loss.
   Matches expected signatures of real vs generated windows of daily
   log-returns. Captures multi-step dependencies (momentum, mean reversion,
   vol clustering) that MLE cannot, since MLE fits per-transition distributions
   independently.

   Reference: Ni, Szpruch, Wiese, Liao (2021) — "Sig-Wasserstein GANs for
   Time Series Generation".
"""

import math
from typing import Callable, Sequence, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import signax

from .model import NeuralSDE, LatentNeuralSDE
from .augmentations import get_minimal_augmentation, get_standard_augmentation
from .generation import generate_minute_paths, generate_latent_daily_paths


def compute_daily_log_prices(minute_prices: jnp.ndarray, chunk_period: int = 1440) -> jnp.ndarray:
    """Convert minute-resolution prices to daily log-prices.

    Takes the last observation in each chunk_period window (daily close).

    Args:
        minute_prices: (T_minutes, n_assets) price array.
        chunk_period: Minutes per day (1440 for standard data).

    Returns:
        (T_days, n_assets) array of log-prices.
    """
    n_minutes = minute_prices.shape[0]
    n_days = n_minutes // chunk_period
    # Truncate to whole days, take last minute of each day
    truncated = minute_prices[: n_days * chunk_period]
    daily = truncated.reshape(n_days, chunk_period, -1)[:, -1, :]
    return jnp.log(daily)


def gaussian_nll(
    sde: NeuralSDE, y_t: jnp.ndarray, y_tp1: jnp.ndarray, dt: float = 1.0
) -> jnp.ndarray:
    """Negative log-likelihood for a single Euler-Maruyama transition.

    Y_{t+1} ~ N(Y_t + mu(Y_t) * dt, L(Y_t) @ L(Y_t)^T * dt)

    For diffusion-only SDEs (learn_drift=False), mu = 0 and this simplifies to
    fitting the state-dependent covariance.

    Args:
        sde: Neural SDE model.
        y_t: (n_assets,) log-prices at time t.
        y_tp1: (n_assets,) log-prices at time t+1.
        dt: Time step (1.0 for per-minute).

    Returns:
        Scalar negative log-likelihood.
    """
    mu = sde.drift(y_t)
    L = sde.diffusion(y_t)

    # Predicted mean and residual
    mean = y_t + mu * dt
    residual = y_tp1 - mean

    # Scale Cholesky by sqrt(dt)
    L_scaled = L * jnp.sqrt(dt)

    # Solve L_scaled @ z = residual for z (whitened residual)
    z = jax.scipy.linalg.solve_triangular(L_scaled, residual, lower=True)

    # NLL = 0.5 * ||z||^2 + sum(log(diag(L_scaled))) + 0.5 * n * log(2*pi)
    n = y_t.shape[0]
    nll = (
        0.5 * jnp.sum(z**2)
        + jnp.sum(jnp.log(jnp.diag(L_scaled)))
        + 0.5 * n * jnp.log(2.0 * jnp.pi)
    )
    return nll


def total_nll(
    sde: NeuralSDE,
    log_prices: jnp.ndarray,
    dt: float = 1.0,
) -> jnp.ndarray:
    """Sum NLL over all consecutive transitions.

    Args:
        sde: Neural SDE model.
        log_prices: (T, n_assets) log-price array (minute or daily).
        dt: Time step.

    Returns:
        Scalar total NLL (not averaged — caller can normalise if desired).
    """
    y_t = log_prices[:-1]
    y_tp1 = log_prices[1:]

    nll_per_step = jax.vmap(lambda yt, ytp1: gaussian_nll(sde, yt, ytp1, dt))(y_t, y_tp1)
    return jnp.sum(nll_per_step)


def fit_neural_sde(
    minute_prices: jnp.ndarray,
    n_assets: int,
    key: jax.Array,
    n_epochs: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 32,
    diagonal_only: bool = False,
    learn_drift: bool = False,
    val_fraction: float = 0.2,
    patience: int = 200,
    batch_size: int = 8192,
    max_train_transitions: int = 50_000,
    max_val_transitions: int = 10_000,
    verbose: bool = True,
) -> Tuple[NeuralSDE, list]:
    """Train a Neural SDE on historical minute-resolution price data.

    Converts to log-prices, splits into train/val temporally, fits via
    minibatch Adam + weight decay, and uses early stopping on the validation NLL.

    At minute resolution, consecutive observations are highly correlated —
    we don't need every single transition. max_train_transitions controls
    how many randomly sampled transitions to use per epoch. This makes
    training tractable on long series without loss of statistical power.

    Args:
        minute_prices: (T_minutes, n_assets) price array.
        n_assets: Number of assets.
        key: JAX PRNG key.
        n_epochs: Maximum training epochs.
        lr: Learning rate.
        weight_decay: L2 regularisation coefficient.
        hidden_dim: Hidden layer width.
        diagonal_only: If True, use diagonal diffusion (fewer params).
        learn_drift: If True, learn a drift network. Default False (diffusion-only).
        val_fraction: Fraction of minute data held out for early stopping.
        patience: Epochs without val improvement before stopping.
        batch_size: Minibatch size (number of transitions per gradient step).
        max_train_transitions: Cap on training transitions used per epoch.
            At minute resolution, 50K transitions gives excellent state-space
            coverage. Set to 0 for no cap.
        max_val_transitions: Cap on validation transitions. 10K is more than
            sufficient for a stable NLL estimate. Set to 0 for no cap.
        verbose: Print training progress.

    Returns:
        (trained_sde, loss_history) where loss_history is list of
        (train_nll, val_nll) tuples per epoch.
    """
    log_prices = jnp.log(minute_prices)
    n_minutes = log_prices.shape[0]

    # Temporal train/val split
    val_size = max(int(n_minutes * val_fraction), 2)
    train_log = log_prices[:-val_size]
    val_log = log_prices[-val_size:]

    n_train_transitions = train_log.shape[0] - 1
    n_val_transitions = val_log.shape[0] - 1

    # Subsample if dataset is large — minute-scale transitions are highly
    # redundant and we get equally good diffusion estimates from a subset
    use_n_train = n_train_transitions
    if max_train_transitions > 0 and n_train_transitions > max_train_transitions:
        use_n_train = max_train_transitions
    use_n_val = n_val_transitions
    if max_val_transitions > 0 and n_val_transitions > max_val_transitions:
        use_n_val = max_val_transitions

    n_batches_per_epoch = max(use_n_train // batch_size, 1)
    effective_batch_size = min(batch_size, use_n_train)

    if verbose:
        print(f"[SDE Training] {n_minutes} minute obs -> train: {train_log.shape[0]}, val: {val_log.shape[0]}")
        subsample_msg = ""
        if use_n_train < n_train_transitions:
            subsample_msg = f" (subsampled from {n_train_transitions})"
        print(f"[SDE Training] {use_n_train} train transitions{subsample_msg}, "
              f"batch_size={effective_batch_size}, {n_batches_per_epoch} batches/epoch")
        print(f"[SDE Training] learn_drift={learn_drift}, diagonal_only={diagonal_only}, hidden_dim={hidden_dim}")

    # Compute empirical minute-return vol for diffusion scale initialisation.
    # The MLP operates in O(1) space; this scale brings outputs to the right
    # magnitude so the optimiser starts near the correct solution.
    log_returns = jnp.diff(train_log, axis=0)
    empirical_std = jnp.std(log_returns, axis=0)

    # Initialise model
    key, subkey = jax.random.split(key)
    sde = NeuralSDE(
        n_assets, hidden_dim, diagonal_only=diagonal_only,
        learn_drift=learn_drift, init_diffusion_scale=empirical_std, key=subkey,
    )

    # Optimiser: Adam + weight decay
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    # Pre-extract transition pairs for training
    train_y_t = train_log[:-1]    # (n_train_transitions, n_assets)
    train_y_tp1 = train_log[1:]   # (n_train_transitions, n_assets)

    # JIT-compiled batch loss+grad and eval
    @eqx.filter_jit
    def batch_loss_and_grad(sde, y_t_batch, y_tp1_batch):
        def loss_fn(model):
            nll = jax.vmap(lambda yt, ytp1: gaussian_nll(model, yt, ytp1, dt=1.0))(
                y_t_batch, y_tp1_batch
            )
            return jnp.mean(nll)
        return eqx.filter_value_and_grad(loss_fn)(sde)

    @eqx.filter_jit
    def eval_loss(sde, y_t, y_tp1):
        nll = jax.vmap(lambda yt, ytp1: gaussian_nll(sde, yt, ytp1, dt=1.0))(y_t, y_tp1)
        return jnp.mean(nll)

    # Validation transition pairs — subsample if needed
    val_y_t = val_log[:-1]
    val_y_tp1 = val_log[1:]
    if use_n_val < n_val_transitions:
        key, subkey = jax.random.split(key)
        val_idx = jax.random.permutation(subkey, n_val_transitions)[:use_n_val]
        val_y_t = val_y_t[val_idx]
        val_y_tp1 = val_y_tp1[val_idx]

    # Training loop
    loss_history = []
    best_val_nll = jnp.inf
    best_sde = sde
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        # Shuffle and subsample transition indices
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train_transitions)[:use_n_train]

        epoch_loss = 0.0
        n_batches_done = 0

        for b in range(n_batches_per_epoch):
            start = b * effective_batch_size
            idx = perm[start : start + effective_batch_size]
            y_t_batch = train_y_t[idx]
            y_tp1_batch = train_y_tp1[idx]

            loss, grads = batch_loss_and_grad(sde, y_t_batch, y_tp1_batch)
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(sde, eqx.is_array)
            )
            sde = eqx.apply_updates(sde, updates)

            epoch_loss += float(loss)
            n_batches_done += 1

        train_nll = epoch_loss / max(n_batches_done, 1)
        val_nll = float(eval_loss(sde, val_y_t, val_y_tp1))
        loss_history.append((train_nll, val_nll))

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_sde = sde
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            print(
                f"  epoch {epoch:4d} | train NLL: {train_nll:.4f} | "
                f"val NLL: {val_nll:.4f} | best val: {best_val_nll:.4f}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    return best_sde, loss_history


# ---------------------------------------------------------------------------
# Sig-W1 training
# ---------------------------------------------------------------------------


def compute_real_expected_signature(
    daily_log_prices: jnp.ndarray,
    window_len: int,
    depth: int,
    augment_fn: Callable,
) -> jnp.ndarray:
    """Precompute E[Sig(augment(real_windows))] from overlapping windows of daily log-returns.

    Extracts all overlapping windows of length ``window_len`` from the daily
    log-return series, applies the augmentation, computes the truncated path
    signature of each, and returns the mean signature vector.

    Args:
        daily_log_prices: (T_days, n_assets) daily log-price series.
        window_len: Window length in days.
        depth: Signature truncation depth.
        augment_fn: Augmentation applied to each window before signature.

    Returns:
        (sig_dim,) mean signature vector.
    """
    # Daily log-returns: (T_days - 1, n_assets)
    log_returns = jnp.diff(daily_log_prices, axis=0)
    n_returns = log_returns.shape[0]

    # Extract overlapping windows: (n_windows, window_len, n_assets)
    n_windows = n_returns - window_len + 1
    indices = jnp.arange(window_len)[None, :] + jnp.arange(n_windows)[:, None]
    windows = log_returns[indices]  # (n_windows, window_len, n_assets)

    # Augment each window
    augmented = jax.vmap(augment_fn)(windows)  # (n_windows, L_aug, d_aug)

    # Compute signatures: signax handles batched (n_windows, L_aug, d_aug) directly
    sigs = signax.signature(augmented, depth, flatten=True)  # (n_windows, sig_dim)

    return jnp.mean(sigs, axis=0)  # (sig_dim,)


def sigw1_loss(
    sde: NeuralSDE,
    real_expected_sig: jnp.ndarray,
    y0_batch: jnp.ndarray,
    window_len: int,
    depth: int,
    augment_fn: Callable,
    mc_samples: int,
    key: jax.Array,
) -> jnp.ndarray:
    """Sig-W1 loss: ||E[Sig(real)] - E[Sig(fake)]||_2.

    For each initial condition in y0_batch, generates mc_samples windows
    via Euler-Maruyama, computes log-returns, augments, takes the signature,
    and compares the average to the precomputed real expected signature.

    Args:
        sde: Neural SDE model.
        real_expected_sig: (sig_dim,) precomputed target.
        y0_batch: (batch_size, n_assets) initial log-prices for each window.
        window_len: Number of daily steps to simulate per window.
        depth: Signature truncation depth.
        augment_fn: Augmentation applied before signature.
        mc_samples: Number of Monte Carlo paths per initial condition.
        key: JAX PRNG key.

    Returns:
        Scalar Sig-W1 loss.
    """
    batch_size = y0_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    def _generate_sigs_for_y0(y0, subkey):
        # Generate mc_samples paths of window_len daily steps
        # generate_minute_paths returns (n_steps, n_assets, n_paths)
        log_paths = generate_minute_paths(
            sde, y0, n_steps=window_len, n_paths=mc_samples, key=subkey, dt=1.0
        )
        # log_paths: (window_len, n_assets, mc_samples)
        # Prepend y0
        y0_broadcast = y0[:, None]  # (n_assets, 1)
        y0_tiled = jnp.broadcast_to(y0_broadcast, (y0.shape[0], mc_samples))
        full_paths = jnp.concatenate(
            [y0_tiled[None, :, :], log_paths], axis=0
        )  # (window_len + 1, n_assets, mc_samples)

        # Transpose to (mc_samples, window_len + 1, n_assets)
        full_paths = jnp.transpose(full_paths, (2, 0, 1))

        # Log-returns: (mc_samples, window_len, n_assets)
        log_rets = jnp.diff(full_paths, axis=1)

        # Augment each sample
        augmented = jax.vmap(augment_fn)(log_rets)  # (mc_samples, L_aug, d_aug)

        # Compute signatures
        sigs = signax.signature(augmented, depth, flatten=True)  # (mc_samples, sig_dim)
        return jnp.mean(sigs, axis=0)  # (sig_dim,)

    # Average over batch of initial conditions
    fake_sigs = jax.vmap(_generate_sigs_for_y0)(y0_batch, keys)  # (batch_size, sig_dim)
    fake_expected_sig = jnp.mean(fake_sigs, axis=0)  # (sig_dim,)

    # L2 distance
    return jnp.sqrt(jnp.sum((real_expected_sig - fake_expected_sig) ** 2))


def _factorial_normalise(flat_sig: jnp.ndarray, aug_dim: int, depth: int) -> jnp.ndarray:
    """Normalise each level of a flattened signature by 1/k!.

    Raw signature terms at level k have magnitude O(||path||^k), growing
    factorially with depth. Without normalisation, deeper levels dominate
    the L2 loss, drowning out the level-1 drift signal and level-2 vol
    structure that we actually need to match.

    Standard in the signature kernel literature (Kiraly & Oberhauser,
    Chevyrev & Oberhauser).

    Args:
        flat_sig: (sig_dim,) flattened signature.
        aug_dim: Dimension of the augmented path (determines level sizes).
        depth: Signature truncation depth.

    Returns:
        (sig_dim,) normalised signature.
    """
    parts = []
    offset = 0
    for k in range(1, depth + 1):
        size = aug_dim ** k
        parts.append(flat_sig[offset:offset + size] / math.factorial(k))
        offset += size
    return jnp.concatenate(parts)


def compute_real_expected_signature_multiscale(
    daily_log_prices: jnp.ndarray,
    window_lens: Sequence[int],
    depth: int,
    augment_fn: Callable,
    aug_dim: int,
) -> jnp.ndarray:
    """Precompute normalised E[Sig(augment(real_windows))] at multiple timescales.

    For each window length, extracts all overlapping windows from the daily
    log-return series, applies augmentation, computes the factorial-normalised
    path signature, and returns the mean.

    Args:
        daily_log_prices: (T_days, n_assets) daily log-price series.
        window_lens: Sequence of window lengths in days (e.g. [10, 20, 50]).
        depth: Signature truncation depth.
        augment_fn: Augmentation applied to each window before signature.
        aug_dim: Dimension of the augmented path.

    Returns:
        (n_scales, sig_dim) array of mean normalised signatures, one per scale.
    """
    log_returns = jnp.diff(daily_log_prices, axis=0)
    n_returns = log_returns.shape[0]

    scale_sigs = []
    for w in window_lens:
        n_windows = n_returns - w + 1
        indices = jnp.arange(w)[None, :] + jnp.arange(n_windows)[:, None]
        windows = log_returns[indices]
        augmented = jax.vmap(augment_fn)(windows)
        sigs = signax.signature(augmented, depth, flatten=True)
        normalised = jax.vmap(
            lambda s: _factorial_normalise(s, aug_dim, depth)
        )(sigs)
        scale_sigs.append(jnp.mean(normalised, axis=0))

    return jnp.stack(scale_sigs)


def sigw1_loss_multiscale(
    sde: NeuralSDE,
    real_expected_sigs: jnp.ndarray,
    y0_batch: jnp.ndarray,
    window_lens: Sequence[int],
    depth: int,
    aug_dim: int,
    augment_fn: Callable,
    mc_samples: int,
    key: jax.Array,
    antithetic: bool = True,
) -> jnp.ndarray:
    """Multi-scale Sig-W1 loss with factorial normalisation and antithetic sampling.

    L = sum_w ||E[Sig_w(real)] - E[Sig_w(fake)]||_2

    Generates paths of length max(window_lens), then extracts prefixes for
    each scale. Shorter scales constrain local dynamics (vol clustering);
    longer scales constrain drift accumulation.

    Args:
        sde: Neural SDE model.
        real_expected_sigs: (n_scales, sig_dim) precomputed targets.
        y0_batch: (batch_size, n_assets) initial log-prices.
        window_lens: Sequence of window lengths.
        depth: Signature truncation depth.
        aug_dim: Dimension of augmented path.
        augment_fn: Augmentation applied before signature.
        mc_samples: Monte Carlo paths per initial condition.
        key: JAX PRNG key.
        antithetic: Use antithetic sampling for variance reduction.

    Returns:
        Scalar multi-scale Sig-W1 loss.
    """
    max_window = max(window_lens)
    batch_size = y0_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    def _compute_fake_sigs_for_y0(y0, subkey):
        # Generate max-length paths
        log_paths = generate_minute_paths(
            sde, y0, n_steps=max_window, n_paths=mc_samples,
            key=subkey, dt=1.0, antithetic=antithetic,
        )
        # (max_window, n_assets, mc_samples) -> prepend y0
        y0_tiled = jnp.broadcast_to(y0[:, None], (y0.shape[0], mc_samples))
        full = jnp.concatenate([y0_tiled[None, :, :], log_paths], axis=0)
        # (max_window+1, n_assets, mc_samples) -> (mc_samples, max_window+1, n_assets)
        full = jnp.transpose(full, (2, 0, 1))

        # Compute normalised signature for each scale
        scale_sigs = []
        for w in window_lens:
            prefix = full[:, :w + 1, :]
            log_rets = jnp.diff(prefix, axis=1)
            augmented = jax.vmap(augment_fn)(log_rets)
            sigs = signax.signature(augmented, depth, flatten=True)
            normalised = jax.vmap(
                lambda s: _factorial_normalise(s, aug_dim, depth)
            )(sigs)
            scale_sigs.append(jnp.mean(normalised, axis=0))

        return jnp.stack(scale_sigs)  # (n_scales, sig_dim)

    # (batch_size, n_scales, sig_dim)
    fake_sigs = jax.vmap(_compute_fake_sigs_for_y0)(y0_batch, keys)
    fake_expected = jnp.mean(fake_sigs, axis=0)  # (n_scales, sig_dim)

    # Sum of per-scale L2 distances
    diffs = real_expected_sigs - fake_expected
    per_scale = jnp.sqrt(jnp.sum(diffs ** 2, axis=1))
    return jnp.sum(per_scale)


def fit_neural_sde_sigw1(
    minute_prices: jnp.ndarray,
    n_assets: int,
    key: jax.Array,
    *,
    hidden_dim: int = 32,
    diagonal_only: bool = False,
    window_lens: Union[Sequence[int], int] = (10, 20, 50),
    depth: int = 3,
    mc_samples: int = 200,
    batch_size: int = 16,
    augmentation: str = "minimal",
    antithetic: bool = True,
    n_steps: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    patience: int = 300,
    verbose: bool = True,
) -> Tuple[NeuralSDE, list]:
    """Train a Neural SDE via multi-scale Sig-W1 loss at daily scale.

    Matches expected factorial-normalised signatures of real vs generated
    windows at multiple timescales simultaneously. Short windows capture local
    dynamics (vol clustering, short-term dependence). Longer windows constrain
    drift accumulation — a process with drift 0.025/day is indistinguishable
    from drift 0.001/day in 10-day windows, but over 50 days the drift
    accumulates 25x vs 1x, clearly separable in the level-1 signature.

    Three design choices address the drift identification problem:
    1. **Multi-scale windows** — provides the information to identify drift
    2. **Factorial normalisation (1/k!)** — prevents deep signature levels from
       drowning out drift signal in the L2 norm
    3. **Antithetic sampling** — halves MC variance of odd-order statistics
       (including drift), improving gradient signal-to-noise

    Args:
        minute_prices: (T_minutes, n_assets) price array.
        n_assets: Number of assets.
        key: JAX PRNG key.
        hidden_dim: Hidden layer width for drift/diffusion networks.
        diagonal_only: If True, use diagonal diffusion.
        window_lens: Window lengths in days. Can be a single int or a sequence.
            Default (10, 20, 50) provides fine-to-coarse resolution.
        depth: Signature truncation depth.
        mc_samples: Monte Carlo samples per initial condition.
        batch_size: Number of initial conditions per gradient step.
        augmentation: "minimal" (lead-lag only) or "standard" (scale+cumsum+time+lead-lag).
        antithetic: Use antithetic sampling for MC variance reduction.
        n_steps: Maximum optimisation steps.
        lr: Learning rate.
        weight_decay: L2 regularisation coefficient.
        val_fraction: Fraction of daily data held out for validation.
        patience: Steps without val improvement before stopping.
        verbose: Print training progress.

    Returns:
        (trained_sde, loss_history) where loss_history is list of
        (train_loss, val_loss) tuples.
    """
    # Normalise window_lens to a sorted tuple
    if isinstance(window_lens, int):
        window_lens = (window_lens,)
    window_lens = tuple(sorted(window_lens))
    max_window = max(window_lens)

    # Convert to daily log-prices
    daily_log_prices = compute_daily_log_prices(minute_prices)
    n_days = daily_log_prices.shape[0]

    if n_days < max_window + 2:
        raise ValueError(
            f"Need at least {max_window + 2} days of data, got {n_days}"
        )

    # Temporal train/val split on daily data
    val_days = max(int(n_days * val_fraction), max_window + 1)
    train_daily = daily_log_prices[:-val_days]
    val_daily = daily_log_prices[-val_days:]

    n_train_days = train_daily.shape[0]
    n_val_days = val_daily.shape[0]

    if n_train_days < max_window + 1:
        raise ValueError(
            f"Training split too small: {n_train_days} days, need >= {max_window + 1}"
        )

    # Select augmentation
    if augmentation == "minimal":
        augment_fn = get_minimal_augmentation()
    elif augmentation == "standard":
        augment_fn = get_standard_augmentation()
    else:
        raise ValueError(f"Unknown augmentation: {augmentation!r}")

    # Determine augmented dimension (static, needed for factorial normalisation)
    dummy_window = jnp.zeros((min(window_lens), n_assets))
    aug_dim = augment_fn(dummy_window).shape[1]

    # Precompute normalised real expected signatures at all scales
    real_train_sigs = compute_real_expected_signature_multiscale(
        train_daily, window_lens, depth, augment_fn, aug_dim
    )
    real_val_sigs = compute_real_expected_signature_multiscale(
        val_daily, window_lens, depth, augment_fn, aug_dim
    )

    sig_dim = real_train_sigs.shape[1]

    if verbose:
        print(f"[Sig-W1] {n_days} daily obs -> train: {n_train_days}, val: {n_val_days}")
        print(f"[Sig-W1] window_lens={window_lens}, depth={depth}, sig_dim={sig_dim}")
        print(f"[Sig-W1] mc_samples={mc_samples}, batch_size={batch_size}, antithetic={antithetic}")
        print(f"[Sig-W1] augmentation={augmentation}, aug_dim={aug_dim}")

    # Compute empirical daily-return vol for diffusion scale initialisation
    daily_returns = jnp.diff(train_daily, axis=0)
    daily_std = jnp.std(daily_returns, axis=0)

    # Initialise model — always learn drift for Sig-W1
    key, subkey = jax.random.split(key)
    sde = NeuralSDE(
        n_assets, hidden_dim, diagonal_only=diagonal_only,
        learn_drift=True, init_diffusion_scale=daily_std, key=subkey,
    )

    # Optimizer
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    # Candidate initial conditions: constrained by longest window
    n_train_windows = n_train_days - max_window
    train_y0_candidates = train_daily[:n_train_windows]

    n_val_windows = n_val_days - max_window
    val_y0_candidates = val_daily[:n_val_windows]

    @eqx.filter_jit
    def train_step(sde, y0_batch, key):
        loss, grads = eqx.filter_value_and_grad(
            lambda model: sigw1_loss_multiscale(
                model, real_train_sigs, y0_batch,
                window_lens, depth, aug_dim, augment_fn,
                mc_samples, key, antithetic,
            )
        )(sde)
        return loss, grads

    @eqx.filter_jit
    def eval_step(sde, y0_batch, real_sigs, key):
        return sigw1_loss_multiscale(
            sde, real_sigs, y0_batch,
            window_lens, depth, aug_dim, augment_fn,
            mc_samples, key, antithetic,
        )

    # Training loop
    loss_history = []
    best_val_loss = jnp.inf
    best_sde = sde
    steps_without_improvement = 0

    for step in range(n_steps):
        key, key_batch, key_loss, key_val = jax.random.split(key, 4)

        # Sample batch of initial conditions
        actual_batch_size = min(batch_size, n_train_windows)
        idx = jax.random.choice(
            key_batch, n_train_windows, shape=(actual_batch_size,), replace=False
        )
        y0_batch = train_y0_candidates[idx]

        # Gradient step
        train_loss, grads = train_step(sde, y0_batch, key_loss)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(sde, eqx.is_array)
        )
        sde = eqx.apply_updates(sde, updates)

        # Validation
        val_batch_size = min(batch_size, n_val_windows)
        val_idx = jax.random.choice(
            key_val, n_val_windows, shape=(val_batch_size,), replace=False
        )
        val_y0_batch = val_y0_candidates[val_idx]
        val_loss = eval_step(sde, val_y0_batch, real_val_sigs, key_val)

        train_loss_f = float(train_loss)
        val_loss_f = float(val_loss)
        loss_history.append((train_loss_f, val_loss_f))

        if val_loss_f < best_val_loss:
            best_val_loss = val_loss_f
            best_sde = sde
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if verbose and (step % 50 == 0 or step == n_steps - 1):
            print(
                f"  step {step:4d} | train Sig-W1: {train_loss_f:.6f} | "
                f"val Sig-W1: {val_loss_f:.6f} | best val: {best_val_loss:.6f}"
            )

        if steps_without_improvement >= patience:
            if verbose:
                print(f"  Early stopping at step {step} (patience={patience})")
            break

    return best_sde, loss_history


# ---------------------------------------------------------------------------
# Latent Neural SDE Sig-W1 training
# ---------------------------------------------------------------------------


def sigw1_loss_multiscale_latent(
    sde: LatentNeuralSDE,
    real_expected_sigs: jnp.ndarray,
    y0_batch: jnp.ndarray,
    window_lens: Sequence[int],
    depth: int,
    aug_dim: int,
    augment_fn: Callable,
    mc_samples: int,
    key: jax.Array,
    antithetic: bool = True,
) -> jnp.ndarray:
    """Multi-scale Sig-W1 loss for LatentNeuralSDE.

    Same as sigw1_loss_multiscale but uses generate_latent_daily_paths
    and readout for observed-space signatures.

    Args:
        sde: LatentNeuralSDE model.
        real_expected_sigs: (n_scales, sig_dim) precomputed targets.
        y0_batch: (batch_size, n_assets) initial observed log-prices.
        window_lens: Sequence of window lengths.
        depth: Signature truncation depth.
        aug_dim: Dimension of augmented path.
        augment_fn: Augmentation applied before signature.
        mc_samples: Monte Carlo paths per initial condition.
        key: JAX PRNG key.
        antithetic: Use antithetic sampling for variance reduction.

    Returns:
        Scalar multi-scale Sig-W1 loss.
    """
    max_window = max(window_lens)
    batch_size = y0_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    def _compute_fake_sigs_for_y0(y0, subkey):
        # Generate max-length paths via latent SDE
        log_paths = generate_latent_daily_paths(
            sde, y0, n_days=max_window, n_paths=mc_samples,
            key=subkey, antithetic=antithetic,
        )
        # (max_window, n_assets, mc_samples) -> prepend y0
        y0_tiled = jnp.broadcast_to(y0[:, None], (y0.shape[0], mc_samples))
        full = jnp.concatenate([y0_tiled[None, :, :], log_paths], axis=0)
        # (max_window+1, n_assets, mc_samples) -> (mc_samples, max_window+1, n_assets)
        full = jnp.transpose(full, (2, 0, 1))

        # Compute normalised signature for each scale
        scale_sigs = []
        for w in window_lens:
            prefix = full[:, :w + 1, :]
            log_rets = jnp.diff(prefix, axis=1)
            augmented = jax.vmap(augment_fn)(log_rets)
            sigs = signax.signature(augmented, depth, flatten=True)
            normalised = jax.vmap(
                lambda s: _factorial_normalise(s, aug_dim, depth)
            )(sigs)
            scale_sigs.append(jnp.mean(normalised, axis=0))

        return jnp.stack(scale_sigs)  # (n_scales, sig_dim)

    # (batch_size, n_scales, sig_dim)
    fake_sigs = jax.vmap(_compute_fake_sigs_for_y0)(y0_batch, keys)
    fake_expected = jnp.mean(fake_sigs, axis=0)  # (n_scales, sig_dim)

    # Sum of per-scale L2 distances
    diffs = real_expected_sigs - fake_expected
    per_scale = jnp.sqrt(jnp.sum(diffs ** 2, axis=1))
    return jnp.sum(per_scale)


def fit_latent_sde_sigw1(
    minute_prices: jnp.ndarray,
    n_assets: int,
    key: jax.Array,
    *,
    n_hidden: int = 4,
    hidden_dim: int = 32,
    diagonal_only: bool = False,
    window_lens: Union[Sequence[int], int] = (10, 20, 50),
    depth: int = 2,
    mc_samples: int = 200,
    batch_size: int = 16,
    augmentation: str = "minimal",
    antithetic: bool = True,
    n_steps: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    patience: int = 300,
    verbose: bool = True,
) -> Tuple[LatentNeuralSDE, list]:
    """Train a Latent Neural SDE via multi-scale Sig-W1 loss at daily scale.

    Same training loop as fit_neural_sde_sigw1 but constructs a
    LatentNeuralSDE and uses sigw1_loss_multiscale_latent.

    Default depth=2 (not 3) — the latent augmentation reduces overfitting risk.

    Args:
        minute_prices: (T_minutes, n_assets) price array.
        n_assets: Number of observed assets.
        key: JAX PRNG key.
        n_hidden: Number of hidden latent dimensions (default 4).
        hidden_dim: Hidden layer width for encoder/drift/diffusion networks.
        diagonal_only: Unused (kept for API consistency; latent SDE uses
            block-diagonal diffusion by design).
        window_lens: Window lengths in days.
        depth: Signature truncation depth (default 2).
        mc_samples: Monte Carlo samples per initial condition.
        batch_size: Number of initial conditions per gradient step.
        augmentation: "minimal" (lead-lag only) or "standard".
        antithetic: Use antithetic sampling for MC variance reduction.
        n_steps: Maximum optimisation steps.
        lr: Learning rate.
        weight_decay: L2 regularisation coefficient.
        val_fraction: Fraction of daily data held out for validation.
        patience: Steps without val improvement before stopping.
        verbose: Print training progress.

    Returns:
        (trained_sde, loss_history) where loss_history is list of
        (train_loss, val_loss) tuples.
    """
    # Normalise window_lens to a sorted tuple
    if isinstance(window_lens, int):
        window_lens = (window_lens,)
    window_lens = tuple(sorted(window_lens))
    max_window = max(window_lens)

    # Convert to daily log-prices
    daily_log_prices = compute_daily_log_prices(minute_prices)
    n_days = daily_log_prices.shape[0]

    if n_days < max_window + 2:
        raise ValueError(
            f"Need at least {max_window + 2} days of data, got {n_days}"
        )

    # Temporal train/val split on daily data
    val_days = max(int(n_days * val_fraction), max_window + 1)
    train_daily = daily_log_prices[:-val_days]
    val_daily = daily_log_prices[-val_days:]

    n_train_days = train_daily.shape[0]
    n_val_days = val_daily.shape[0]

    if n_train_days < max_window + 1:
        raise ValueError(
            f"Training split too small: {n_train_days} days, need >= {max_window + 1}"
        )

    # Select augmentation
    if augmentation == "minimal":
        augment_fn = get_minimal_augmentation()
    elif augmentation == "standard":
        augment_fn = get_standard_augmentation()
    else:
        raise ValueError(f"Unknown augmentation: {augmentation!r}")

    # Determine augmented dimension
    dummy_window = jnp.zeros((min(window_lens), n_assets))
    aug_dim = augment_fn(dummy_window).shape[1]

    # Precompute normalised real expected signatures at all scales
    real_train_sigs = compute_real_expected_signature_multiscale(
        train_daily, window_lens, depth, augment_fn, aug_dim
    )
    real_val_sigs = compute_real_expected_signature_multiscale(
        val_daily, window_lens, depth, augment_fn, aug_dim
    )

    sig_dim = real_train_sigs.shape[1]
    latent_dim = n_assets + n_hidden

    if verbose:
        print(f"[Latent Sig-W1] {n_days} daily obs -> train: {n_train_days}, val: {n_val_days}")
        print(f"[Latent Sig-W1] n_hidden={n_hidden}, latent_dim={latent_dim}")
        print(f"[Latent Sig-W1] window_lens={window_lens}, depth={depth}, sig_dim={sig_dim}")
        print(f"[Latent Sig-W1] mc_samples={mc_samples}, batch_size={batch_size}, antithetic={antithetic}")
        print(f"[Latent Sig-W1] augmentation={augmentation}, aug_dim={aug_dim}")

    # Compute empirical daily-return statistics for initialisation
    daily_returns = jnp.diff(train_daily, axis=0)
    daily_std = jnp.std(daily_returns, axis=0)

    # Drift init: signed empirical drift for asset dims, zero for hidden dims.
    # The bias provides the constant baseline; the MLP learns state-dependent
    # deviations with O(1) gradient flow.
    daily_drift = jnp.mean(daily_returns, axis=0)
    hidden_drift_init = jnp.zeros(n_hidden)
    init_drift_scale = jnp.concatenate([daily_drift, hidden_drift_init])

    # Initialise model
    key, subkey = jax.random.split(key)
    sde = LatentNeuralSDE(
        n_assets, n_hidden, hidden_dim,
        init_diffusion_scale=daily_std,
        init_drift_scale=init_drift_scale,
        key=subkey,
    )

    # Optimizer
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    # Candidate initial conditions
    n_train_windows = n_train_days - max_window
    train_y0_candidates = train_daily[:n_train_windows]

    n_val_windows = n_val_days - max_window
    val_y0_candidates = val_daily[:n_val_windows]

    @eqx.filter_jit
    def train_step(sde, y0_batch, key):
        loss, grads = eqx.filter_value_and_grad(
            lambda model: sigw1_loss_multiscale_latent(
                model, real_train_sigs, y0_batch,
                window_lens, depth, aug_dim, augment_fn,
                mc_samples, key, antithetic,
            )
        )(sde)
        return loss, grads

    @eqx.filter_jit
    def eval_step(sde, y0_batch, real_sigs, key):
        return sigw1_loss_multiscale_latent(
            sde, real_sigs, y0_batch,
            window_lens, depth, aug_dim, augment_fn,
            mc_samples, key, antithetic,
        )

    # Training loop
    loss_history = []
    best_val_loss = jnp.inf
    best_sde = sde
    steps_without_improvement = 0

    for step in range(n_steps):
        key, key_batch, key_loss, key_val = jax.random.split(key, 4)

        # Sample batch of initial conditions
        actual_batch_size = min(batch_size, n_train_windows)
        idx = jax.random.choice(
            key_batch, n_train_windows, shape=(actual_batch_size,), replace=False
        )
        y0_batch = train_y0_candidates[idx]

        # Gradient step
        train_loss, grads = train_step(sde, y0_batch, key_loss)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(sde, eqx.is_array)
        )
        sde = eqx.apply_updates(sde, updates)

        # Validation
        val_batch_size = min(batch_size, n_val_windows)
        val_idx = jax.random.choice(
            key_val, n_val_windows, shape=(val_batch_size,), replace=False
        )
        val_y0_batch = val_y0_candidates[val_idx]
        val_loss = eval_step(sde, val_y0_batch, real_val_sigs, key_val)

        train_loss_f = float(train_loss)
        val_loss_f = float(val_loss)
        loss_history.append((train_loss_f, val_loss_f))

        if val_loss_f < best_val_loss:
            best_val_loss = val_loss_f
            best_sde = sde
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if verbose and (step % 50 == 0 or step == n_steps - 1):
            print(
                f"  step {step:4d} | train Sig-W1: {train_loss_f:.6f} | "
                f"val Sig-W1: {val_loss_f:.6f} | best val: {best_val_loss:.6f}"
            )

        if steps_without_improvement >= patience:
            if verbose:
                print(f"  Early stopping at step {step} (patience={patience})")
            break

    return best_sde, loss_history


# ---------------------------------------------------------------------------
# Drifting model training (Deng et al., "Generative Modeling via Drifting")
# ---------------------------------------------------------------------------


def compute_drift_field(
    gen: jnp.ndarray,
    pos: jnp.ndarray,
    temp: float = 0.05,
) -> jnp.ndarray:
    """Kernel-based mean-shift drift field (Deng et al. 2026).

    Computes V(x) = V+(attract to data) - V-(repel from generated) using a
    Laplace kernel k(x,y) = exp(-||x-y||/temp) with batch normalization along
    both dimensions.

    The drift field satisfies V_{p,p} = 0: when the generated distribution
    matches the data, the field vanishes.

    Args:
        gen: (G, D) generated samples (no gradient needed).
        pos: (P, D) data samples.
        temp: Temperature (bandwidth) for the Laplace kernel. Should be
            on the order of the median pairwise distance in the data.

    Returns:
        (G, D) drift vectors — one per generated sample.
    """
    G = gen.shape[0]
    targets = jnp.concatenate([gen, pos], axis=0)  # (G+P, D)

    # Pairwise L2 distances: gen to all targets
    diff = gen[:, None, :] - targets[None, :, :]  # (G, G+P, D)
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)  # (G, G+P)

    # Mask self-distances (gen[i] vs gen[i] in the gen block of targets)
    dist = dist.at[:, :G].add(jnp.eye(G) * 1e6)

    # Laplace kernel
    kernel = jnp.exp(-dist / temp)  # (G, G+P)

    # Batch normalization along both dimensions (Sinkhorn-style, one step)
    row_sum = kernel.sum(axis=-1, keepdims=True)   # (G, 1)
    col_sum = kernel.sum(axis=-2, keepdims=True)   # (1, G+P)
    normalizer = jnp.sqrt(jnp.maximum(row_sum * col_sum, 1e-12))
    nk = kernel / normalizer  # (G, G+P)

    # Positive drift: attraction from data (targets[G:])
    nk_pos = nk[:, G:]   # (G, P)
    nk_neg = nk[:, :G]   # (G, G)
    pos_coeff = nk_pos * nk_neg.sum(axis=-1, keepdims=True)  # (G, P)
    pos_V = pos_coeff @ targets[G:]  # (G, D)

    # Negative drift: repulsion from generated (targets[:G])
    neg_coeff = nk_neg * nk_pos.sum(axis=-1, keepdims=True)  # (G, G)
    neg_V = neg_coeff @ targets[:G]  # (G, D)

    return pos_V - neg_V


def precompute_real_window_signatures(
    daily_log_prices: jnp.ndarray,
    window_lens: Sequence[int],
    depth: int,
    augment_fn: Callable,
    aug_dim: int,
    drift_weight: float = 0.0,
) -> jnp.ndarray:
    """Precompute individual factorial-normalised signatures for all overlapping windows.

    Unlike ``compute_real_expected_signature_multiscale`` which returns the *mean*
    signature per scale, this returns the full set of per-window signatures
    concatenated across scales — needed for the per-sample drifting loss.

    For each starting point t (where t + max(window_lens) <= n_returns), computes
    signatures at all scales and concatenates them into a single vector.

    When ``drift_weight > 0``, appends weighted mean-return features per scale.
    These make drift visible in the kernel metric space: signatures capture
    vol/correlation structure but are insensitive to drift at daily timescales
    (vol >> drift). Mean returns are directly proportional to drift, so
    weighting them into the feature vector gives the drift field V a signal
    to correct.

    Args:
        daily_log_prices: (T_days, n_assets) daily log-price series.
        window_lens: Sequence of window lengths in days.
        depth: Signature truncation depth.
        augment_fn: Augmentation applied to each window before signature.
        aug_dim: Dimension of the augmented path.
        drift_weight: Scaling factor for mean-return features. When 0 (default),
            no drift features are appended (backward compatible). When > 0,
            each scale gets n_assets extra dimensions: mean(returns) * drift_weight.

    Returns:
        (n_windows, total_dim) array where total_dim = n_scales * sig_dim when
        drift_weight=0, or n_scales * (sig_dim + n_assets) when drift_weight > 0.
    """
    log_returns = jnp.diff(daily_log_prices, axis=0)
    n_returns = log_returns.shape[0]
    max_window = max(window_lens)
    n_windows = n_returns - max_window + 1

    scale_features = []
    for w in window_lens:
        # All windows of length w starting from the same set of points
        indices = jnp.arange(w)[None, :] + jnp.arange(n_windows)[:, None]
        windows = log_returns[indices]  # (n_windows, w, n_assets)
        augmented = jax.vmap(augment_fn)(windows)
        sigs = signax.signature(augmented, depth, flatten=True)
        normalised = jax.vmap(
            lambda s: _factorial_normalise(s, aug_dim, depth)
        )(sigs)  # (n_windows, sig_dim)

        if drift_weight > 0:
            # Mean return per window: (n_windows, n_assets)
            mean_returns = jnp.mean(windows, axis=1) * drift_weight
            normalised = jnp.concatenate([normalised, mean_returns], axis=1)

        scale_features.append(normalised)

    return jnp.concatenate(scale_features, axis=1)


def drifting_loss_latent(
    sde: LatentNeuralSDE,
    real_sigs: jnp.ndarray,
    y0_batch: jnp.ndarray,
    window_lens: Sequence[int],
    depth: int,
    aug_dim: int,
    augment_fn: Callable,
    key: jax.Array,
    temp: float,
    mc_samples: int = 1,
    antithetic: bool = True,
    drift_weight: float = 0.0,
) -> jnp.ndarray:
    """Drifting loss for LatentNeuralSDE in multi-scale feature space.

    For each y0, generates ``mc_samples`` paths, computes their feature vectors
    (signatures + optional drift features), and averages to get one MC-estimated
    expected feature per y0. The drift field V then operates on these features.

    When ``drift_weight > 0``, weighted mean-return features are concatenated
    with signatures at each scale. This makes the kernel metric space sensitive
    to drift: signatures handle vol/correlation, mean returns handle drift.
    The drift field V then provides meaningful corrections for both.

    Args:
        sde: LatentNeuralSDE model.
        real_sigs: (n_real, total_feature_dim) precomputed per-window features.
            Must include drift features if drift_weight > 0.
        y0_batch: (batch_size, n_assets) initial observed log-prices.
        window_lens: Sequence of window lengths (must match real_sigs scales).
        depth: Signature truncation depth.
        aug_dim: Dimension of augmented path.
        augment_fn: Augmentation applied before signature.
        key: JAX PRNG key.
        temp: Kernel temperature for the drift field.
        mc_samples: Monte Carlo paths per y0 for signature averaging.
            Higher = less noisy drift field but more compute. Default 1.
        antithetic: Use antithetic sampling for MC variance reduction.
        drift_weight: Scaling factor for mean-return features. Must match
            the value used in precompute_real_window_signatures.

    Returns:
        Scalar drifting loss.
    """
    max_window = max(window_lens)
    batch_size = y0_batch.shape[0]
    keys = jax.random.split(key, batch_size)

    def _gen_mc_averaged_features(y0, subkey):
        # Generate mc_samples paths of max_window days
        paths = generate_latent_daily_paths(
            sde, y0, n_days=max_window, n_paths=mc_samples,
            key=subkey, antithetic=antithetic,
        )
        # paths: (max_window, n_assets, mc_samples)
        # Prepend y0
        y0_tiled = jnp.broadcast_to(y0[:, None], (y0.shape[0], mc_samples))
        full = jnp.concatenate([y0_tiled[None, :, :], paths], axis=0)
        # full: (max_window+1, n_assets, mc_samples)
        # Transpose to (mc_samples, max_window+1, n_assets)
        full = jnp.transpose(full, (2, 0, 1))

        # Compute feature vector (sig + optional drift features) per path per scale
        def _features_for_one_path(path):
            scale_features = []
            for w in window_lens:
                prefix = path[:w + 1, :]
                rets = jnp.diff(prefix, axis=0)
                aug = augment_fn(rets)
                sig = signax.signature(aug, depth, flatten=True)
                sig = _factorial_normalise(sig, aug_dim, depth)
                if drift_weight > 0:
                    mean_ret = jnp.mean(rets, axis=0) * drift_weight
                    sig = jnp.concatenate([sig, mean_ret])
                scale_features.append(sig)
            return jnp.concatenate(scale_features)

        all_features = jax.vmap(_features_for_one_path)(full)  # (mc_samples, total_dim)
        return jnp.mean(all_features, axis=0)  # (total_dim,) — MC average

    # Generate MC-averaged features (differentiable w.r.t. sde params)
    gen_features = jax.vmap(_gen_mc_averaged_features)(y0_batch, keys)  # (batch_size, total_dim)

    # Compute drift field in feature space (no gradient)
    gen_sg = jax.lax.stop_gradient(gen_features)
    V = compute_drift_field(gen_sg, real_sigs, temp=temp)

    # Drifting loss: MSE(gen, stopgrad(gen + V))
    target = jax.lax.stop_gradient(gen_features + V)
    return jnp.mean(jnp.sum((gen_features - target) ** 2, axis=1))


def fit_latent_sde_drifting(
    minute_prices: jnp.ndarray,
    n_assets: int,
    key: jax.Array,
    *,
    n_hidden: int = 4,
    hidden_dim: int = 32,
    window_lens: Union[Sequence[int], int] = (10, 20, 50),
    depth: int = 2,
    mc_samples: int = 64,
    gen_batch_size: int = 32,
    data_batch_size: int = 256,
    augmentation: str = "minimal",
    antithetic: bool = True,
    n_steps: int = 5000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temp: float = None,
    drift_weight: float = None,
    val_fraction: float = 0.2,
    patience: int = 500,
    verbose: bool = True,
) -> Tuple[LatentNeuralSDE, list]:
    """Train a Latent Neural SDE via the drifting model approach.

    Instead of matching population-average signatures (Sig-W1), this uses
    per-sample kernel-based drift in feature space (Deng et al. 2026,
    "Generative Modeling via Drifting"). Each generated path gets a personalised
    correction vector V that points toward the data distribution. The loss
    trains the SDE to produce paths whose features match the data.

    When ``drift_weight`` is set (default: auto), weighted mean-return features
    are concatenated with signatures. This addresses the core drift identification
    failure: signatures are insensitive to drift at daily timescales because vol
    terms dominate the L2 metric. Mean returns are directly proportional to drift,
    making them visible to the kernel and drift field V.

    The auto drift_weight equalises the per-component variance of signature
    entries and mean-return entries in the real data, so neither dominates.

    Args:
        minute_prices: (T_minutes, n_assets) price array.
        n_assets: Number of observed assets.
        key: JAX PRNG key.
        n_hidden: Number of hidden latent dimensions.
        hidden_dim: Hidden layer width.
        window_lens: Window lengths in days for multi-scale signatures.
        depth: Signature truncation depth (default 2).
        mc_samples: Monte Carlo paths per y0 for signature averaging.
            Higher values reduce vol noise in the drift field, improving
            drift estimation at the cost of more compute per step.
        gen_batch_size: Number of y0 initial conditions per training step.
            Each gets mc_samples paths, so total paths = gen_batch_size * mc_samples.
        data_batch_size: Real signature samples per step for drift computation.
        augmentation: "minimal" or "standard".
        antithetic: Use antithetic sampling for MC variance reduction.
        n_steps: Maximum training steps.
        lr: Learning rate.
        weight_decay: L2 regularisation.
        temp: Kernel temperature. If None, auto-computed from median pairwise
            distance in real features.
        drift_weight: Scaling for mean-return features. If None (default),
            auto-computed to equalise per-component variance with signatures.
            Set to 0.0 to disable drift features (signature-only mode).
        val_fraction: Fraction of daily data held out for validation.
        patience: Steps without val improvement before stopping.
        verbose: Print training progress.

    Returns:
        (trained_sde, loss_history) where loss_history is list of
        (train_loss, val_loss) tuples.
    """
    # Normalise window_lens
    if isinstance(window_lens, int):
        window_lens = (window_lens,)
    window_lens = tuple(sorted(window_lens))
    max_window = max(window_lens)

    # Convert to daily log-prices
    daily_log_prices = compute_daily_log_prices(minute_prices)
    n_days = daily_log_prices.shape[0]

    if n_days < max_window + 2:
        raise ValueError(
            f"Need at least {max_window + 2} days of data, got {n_days}"
        )

    # Temporal train/val split
    val_days = max(int(n_days * val_fraction), max_window + 1)
    train_daily = daily_log_prices[:-val_days]
    val_daily = daily_log_prices[-val_days:]

    n_train_days = train_daily.shape[0]
    n_val_days = val_daily.shape[0]

    if n_train_days < max_window + 1:
        raise ValueError(
            f"Training split too small: {n_train_days} days, need >= {max_window + 1}"
        )

    # Select augmentation
    if augmentation == "minimal":
        augment_fn = get_minimal_augmentation()
    elif augmentation == "standard":
        augment_fn = get_standard_augmentation()
    else:
        raise ValueError(f"Unknown augmentation: {augmentation!r}")

    # Determine augmented dimension
    dummy_window = jnp.zeros((min(window_lens), n_assets))
    aug_dim = augment_fn(dummy_window).shape[1]

    # Auto-compute drift_weight from real data statistics
    if drift_weight is None:
        # Compute signatures without drift features to calibrate
        sigs_only = precompute_real_window_signatures(
            train_daily, window_lens, depth, augment_fn, aug_dim, drift_weight=0.0
        )
        sig_component_std = float(jnp.std(sigs_only))

        # Compute mean returns for the same windows to calibrate
        log_returns_cal = jnp.diff(train_daily, axis=0)
        n_ret_cal = log_returns_cal.shape[0]
        n_win_cal = n_ret_cal - max_window + 1
        drift_parts = []
        for w in window_lens:
            idx_cal = jnp.arange(w)[None, :] + jnp.arange(n_win_cal)[:, None]
            wins_cal = log_returns_cal[idx_cal]
            drift_parts.append(jnp.mean(wins_cal, axis=1))  # (n_windows, n_assets)
        all_drift_feats = jnp.concatenate(drift_parts, axis=1)
        drift_component_std = float(jnp.std(all_drift_feats))

        # Equalise per-component variance
        drift_weight = sig_component_std / (drift_component_std + 1e-12)
        if verbose:
            print(f"[Drifting] Auto drift_weight = {drift_weight:.2f} "
                  f"(sig_std={sig_component_std:.6f}, drift_std={drift_component_std:.6f})")

    # Precompute individual per-window features (signatures + drift features)
    train_sigs = precompute_real_window_signatures(
        train_daily, window_lens, depth, augment_fn, aug_dim,
        drift_weight=drift_weight,
    )
    val_sigs = precompute_real_window_signatures(
        val_daily, window_lens, depth, augment_fn, aug_dim,
        drift_weight=drift_weight,
    )

    n_train_windows = train_sigs.shape[0]
    n_val_windows = val_sigs.shape[0]
    total_feature_dim = train_sigs.shape[1]

    # Auto-compute temperature from median pairwise distance in feature space
    if temp is None:
        n_sample = min(200, n_train_windows)
        key, key_temp = jax.random.split(key)
        sample_idx = jax.random.choice(
            key_temp, n_train_windows, shape=(n_sample,), replace=False
        )
        sample_sigs = train_sigs[sample_idx]
        sig_diff = sample_sigs[:, None, :] - sample_sigs[None, :, :]
        dists = jnp.sqrt(jnp.sum(sig_diff ** 2, axis=-1) + 1e-12)
        # Median of non-zero distances
        mask = dists > 1e-8
        temp = float(jnp.median(dists[mask]))
        if verbose:
            print(f"[Drifting] Auto temp = {temp:.6f} (median pairwise feature distance)")

    if verbose:
        print(f"[Drifting] {n_days} daily obs -> train: {n_train_days}, val: {n_val_days}")
        print(f"[Drifting] n_hidden={n_hidden}, latent_dim={n_assets + n_hidden}")
        print(f"[Drifting] window_lens={window_lens}, depth={depth}, total_feature_dim={total_feature_dim}")
        print(f"[Drifting] mc_samples={mc_samples}, antithetic={antithetic}")
        print(f"[Drifting] gen_batch={gen_batch_size}, data_batch={data_batch_size}, temp={temp:.6f}")
        print(f"[Drifting] drift_weight={drift_weight:.2f}")
        print(f"[Drifting] augmentation={augmentation}, aug_dim={aug_dim}")

    # Compute empirical daily-return statistics for initialisation
    daily_returns = jnp.diff(train_daily, axis=0)
    daily_std = jnp.std(daily_returns, axis=0)

    # Drift init: signed empirical drift for asset dims, zero for hidden dims.
    # The bias provides the constant baseline; the MLP learns state-dependent
    # deviations with O(1) gradient flow (additive residual form).
    daily_drift = jnp.mean(daily_returns, axis=0)
    hidden_drift_init = jnp.zeros(n_hidden)
    init_drift_scale = jnp.concatenate([daily_drift, hidden_drift_init])

    if verbose:
        print(f"[Drifting] drift init bias: {[f'{float(x):.6f}' for x in daily_drift]}")

    # Initialise model
    key, subkey = jax.random.split(key)
    sde = LatentNeuralSDE(
        n_assets, n_hidden, hidden_dim,
        init_diffusion_scale=daily_std,
        init_drift_scale=init_drift_scale,
        key=subkey,
    )

    # Optimizer
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(eqx.filter(sde, eqx.is_array))

    # Candidate initial conditions
    n_train_y0 = n_train_days - max_window
    train_y0_candidates = train_daily[:n_train_y0]

    n_val_y0 = n_val_days - max_window
    val_y0_candidates = val_daily[:n_val_y0]

    @eqx.filter_jit
    def train_step(sde, y0_batch, real_sig_batch, key):
        loss, grads = eqx.filter_value_and_grad(
            lambda model: drifting_loss_latent(
                model, real_sig_batch, y0_batch,
                window_lens, depth, aug_dim, augment_fn,
                key, temp, mc_samples, antithetic,
                drift_weight=drift_weight,
            )
        )(sde)
        return loss, grads

    @eqx.filter_jit
    def eval_step(sde, y0_batch, real_sig_batch, key):
        return drifting_loss_latent(
            sde, real_sig_batch, y0_batch,
            window_lens, depth, aug_dim, augment_fn,
            key, temp, mc_samples, antithetic,
            drift_weight=drift_weight,
        )

    # Training loop
    loss_history = []
    best_val_loss = jnp.inf
    best_sde = sde
    steps_without_improvement = 0

    for step in range(n_steps):
        key, key_y0, key_data, key_loss, key_val_y0, key_val_data, key_val_loss = (
            jax.random.split(key, 7)
        )

        # Sample initial conditions for generated paths
        actual_gen_batch = min(gen_batch_size, n_train_y0)
        y0_idx = jax.random.choice(
            key_y0, n_train_y0, shape=(actual_gen_batch,), replace=False
        )
        y0_batch = train_y0_candidates[y0_idx]

        # Sample real signatures for drift computation
        actual_data_batch = min(data_batch_size, n_train_windows)
        data_idx = jax.random.choice(
            key_data, n_train_windows, shape=(actual_data_batch,), replace=False
        )
        real_sig_batch = train_sigs[data_idx]

        # Gradient step
        train_loss, grads = train_step(sde, y0_batch, real_sig_batch, key_loss)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(sde, eqx.is_array)
        )
        sde = eqx.apply_updates(sde, updates)

        # Validation
        actual_val_gen = min(gen_batch_size, n_val_y0)
        val_y0_idx = jax.random.choice(
            key_val_y0, n_val_y0, shape=(actual_val_gen,), replace=False
        )
        val_y0_batch = val_y0_candidates[val_y0_idx]

        actual_val_data = min(data_batch_size, n_val_windows)
        val_data_idx = jax.random.choice(
            key_val_data, n_val_windows, shape=(actual_val_data,), replace=False
        )
        val_sig_batch = val_sigs[val_data_idx]

        val_loss = eval_step(sde, val_y0_batch, val_sig_batch, key_val_loss)

        train_loss_f = float(train_loss)
        val_loss_f = float(val_loss)
        loss_history.append((train_loss_f, val_loss_f))

        if val_loss_f < best_val_loss:
            best_val_loss = val_loss_f
            best_sde = sde
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if verbose and (step % 50 == 0 or step == n_steps - 1):
            print(
                f"  step {step:4d} | train drift: {train_loss_f:.6f} | "
                f"val drift: {val_loss_f:.6f} | best val: {best_val_loss:.6f}"
            )

        if steps_without_improvement >= patience:
            if verbose:
                print(f"  Early stopping at step {step} (patience={patience})")
            break

    return best_sde, loss_history
