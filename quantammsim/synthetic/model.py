"""Neural SDE model for synthetic price path generation.

Equinox modules defining a neural SDE in log-price space:
    dY_t = mu(Y_t) dt + L(Y_t) dW_t

where Y_t = log(prices_t), mu is an (optionally learned) drift, and L is a
learned Cholesky factor of the diffusion covariance.

Design choices:
- Log-price space ensures positivity of generated prices.
- Small networks (1 hidden layer, 32 units) — regularised by architecture.
- Cholesky parameterisation guarantees positive-definite diffusion covariance.
- Diagonal-only diffusion mode for n_assets > 4 to reduce parameters.
- Default: diffusion-only (zero drift). Drift is statistically unidentifiable
  from short time series and causes path blowup when compounded forward.
"""

from pathlib import Path
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp


class DriftNetwork(eqx.Module):
    """MLP mapping log-prices to drift vector.

    Input:  (n_assets,) log-prices
    Output: (n_assets,) drift
    """

    mlp: eqx.nn.MLP

    def __init__(self, n_assets: int, hidden_dim: int = 32, *, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=n_assets,
            out_size=n_assets,
            width_size=hidden_dim,
            depth=1,
            activation=jnp.tanh,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        return self.mlp(y)


class ZeroDrift(eqx.Module):
    """Constant-zero drift for diffusion-only SDEs.

    No learnable parameters. The SDE reduces to a pure volatility model:
        dY_t = L(Y_t) dW_t

    This is well-justified at minute timescale where drift is dominated by
    noise and statistically unidentifiable from finite data.
    """

    n_assets: int

    def __call__(self, y: jax.Array) -> jax.Array:
        return jnp.zeros(self.n_assets)


class DiffusionNetwork(eqx.Module):
    """MLP mapping log-prices to a Cholesky factor of the diffusion covariance.

    Input:  (n_assets,) log-prices
    Output: (n_assets, n_assets) lower-triangular L such that Sigma = L @ L^T is PD.

    The raw MLP output has n_assets*(n_assets+1)//2 entries (lower triangle).
    Diagonal elements are passed through softplus to ensure positivity.

    A learnable per-asset ``log_scale`` handles the magnitude: the MLP operates
    in its natural O(1) output range while ``exp(log_scale)`` brings values to the
    correct minute-return scale.  This avoids the floor problem where
    ``softplus(min_mlp_output)`` exceeds the target diffusion magnitude.

    Initialise ``init_scale`` from the empirical std of minute log-returns so
    the model starts in the right ballpark.  Correlations are preserved:
    ``L_final = diag(exp(log_scale)) @ L_raw`` scales each asset's vol
    independently without affecting the correlation structure.
    """

    mlp: eqx.nn.MLP
    n_assets: int
    diagonal_only: bool
    log_scale: jax.Array  # (n_assets,) learnable per-asset output scale

    def __init__(
        self,
        n_assets: int,
        hidden_dim: int = 32,
        diagonal_only: bool = False,
        init_scale: jax.Array = None,
        *,
        key: jax.Array,
    ):
        self.n_assets = n_assets
        self.diagonal_only = diagonal_only

        if init_scale is not None:
            self.log_scale = jnp.log(jnp.asarray(init_scale))
        else:
            self.log_scale = jnp.zeros(n_assets)

        if diagonal_only:
            out_size = n_assets
        else:
            out_size = n_assets * (n_assets + 1) // 2

        self.mlp = eqx.nn.MLP(
            in_size=n_assets,
            out_size=out_size,
            width_size=hidden_dim,
            depth=1,
            activation=jnp.tanh,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        """Returns (n_assets, n_assets) lower-triangular Cholesky factor."""
        raw = self.mlp(y)
        n = self.n_assets
        scale = jnp.exp(self.log_scale)  # (n_assets,)

        if self.diagonal_only:
            # Diagonal diffusion: L = diag(softplus(raw) * scale)
            return jnp.diag((jax.nn.softplus(raw) + 1e-6) * scale)

        # Fill lower triangular matrix
        L = jnp.zeros((n, n))
        tril_indices = jnp.tril_indices(n)
        L = L.at[tril_indices].set(raw)

        # Make diagonal positive via softplus
        diag_indices = jnp.diag_indices(n)
        L = L.at[diag_indices].set(jax.nn.softplus(L[diag_indices]) + 1e-6)

        # Scale each row by per-asset scale factor
        # L_final = diag(scale) @ L_raw
        L = L * scale[:, None]

        return L


class NeuralSDE(eqx.Module):
    """Neural SDE in log-price space.

    dY_t = drift(Y_t) dt + diffusion(Y_t) dW_t

    Default is diffusion-only (learn_drift=False): drift returns zeros,
    only the state-dependent volatility structure is learned. This avoids
    the drift blowup problem on short time series.
    """

    drift: Union[DriftNetwork, ZeroDrift]
    diffusion: DiffusionNetwork
    n_assets: int
    learn_drift: bool

    def __init__(
        self,
        n_assets: int,
        hidden_dim: int = 32,
        diagonal_only: bool = False,
        learn_drift: bool = False,
        init_diffusion_scale: jax.Array = None,
        *,
        key: jax.Array,
    ):
        key_drift, key_diff = jax.random.split(key)
        self.n_assets = n_assets
        self.learn_drift = learn_drift

        if learn_drift:
            self.drift = DriftNetwork(n_assets, hidden_dim, key=key_drift)
        else:
            self.drift = ZeroDrift(n_assets)

        self.diffusion = DiffusionNetwork(
            n_assets, hidden_dim, diagonal_only=diagonal_only,
            init_scale=init_diffusion_scale, key=key_diff,
        )


class Encoder(eqx.Module):
    """Maps observed Y_0 to latent Z_0 = [Y_0, MLP(Y_0)].

    The first ``n_assets`` dims of Z_0 are the observed log-prices (identity),
    and the remaining ``n_hidden`` dims are a learned nonlinear function of Y_0.
    This ensures that at initialisation (before training moves the hidden dims),
    the readout is a no-op: Y_t ≈ Z_t[:n_assets].
    """

    mlp: eqx.nn.MLP
    n_assets: int
    n_hidden: int

    def __init__(self, n_assets: int, n_hidden: int, hidden_dim: int = 32, *, key: jax.Array):
        self.n_assets = n_assets
        self.n_hidden = n_hidden
        self.mlp = eqx.nn.MLP(
            in_size=n_assets,
            out_size=n_hidden,
            width_size=hidden_dim,
            depth=1,
            activation=jnp.tanh,
            key=key,
        )

    def __call__(self, y: jax.Array) -> jax.Array:
        """Encode observed (n_assets,) -> latent (n_assets + n_hidden,)."""
        hidden = self.mlp(y)
        return jnp.concatenate([y, hidden])


class Readout(eqx.Module):
    """Linear projection Z_t -> Y_t. Identity-initialized on asset dims.

    weight is (n_assets, latent_dim). At init:
        weight = [I_{n_assets} | 0_{n_assets x n_hidden}]
    so readout(Z_t) = Z_t[:n_assets] exactly, making the latent SDE
    equivalent to the standard SDE before training starts.
    """

    weight: jax.Array  # (n_assets, latent_dim)

    def __init__(self, n_assets: int, latent_dim: int):
        # Identity on asset dims, zero on hidden dims
        self.weight = jnp.concatenate([
            jnp.eye(n_assets),
            jnp.zeros((n_assets, latent_dim - n_assets)),
        ], axis=1)

    def __call__(self, z: jax.Array) -> jax.Array:
        """Project (latent_dim,) -> (n_assets,)."""
        return self.weight @ z


class LatentDriftNetwork(eqx.Module):
    """MLP drift on full latent state: (latent_dim,) -> (latent_dim,).

    Unlike the observed-space DriftNetwork which only sees Y_t, this operates
    on the full latent state Z_t = [Y_t, H_t], giving the drift access to
    accumulated trajectory history through the hidden dimensions.

    Uses bounded additive residual form::

        drift(z) = bias + max_dev * tanh(MLP(z))

    ``bias`` is a learnable per-dim constant initialized from empirical daily
    drift. ``max_dev`` is a learnable per-dim scale that bounds the MLP's
    state-dependent contribution to ``[-max_dev, +max_dev]``. This means:

    - At init: drift ≈ bias (MLP output ≈ 0 due to small final-layer weights)
    - tanh prevents the MLP from overpowering the bias (no sign flips)
    - Gradients through the MLP are O(1) in tanh's linear regime
    - max_dev can grow during training if the loss benefits from larger deviations
    - weight_decay on max_dev provides natural regularization pressure
    """

    mlp: eqx.nn.MLP
    bias: jax.Array     # (latent_dim,) learnable constant drift
    max_dev: jax.Array  # (latent_dim,) learnable bound on MLP deviation

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 32,
        init_drift: jax.Array = None,
        *,
        key: jax.Array,
    ):
        if init_drift is not None:
            init_drift = jnp.asarray(init_drift)
            self.bias = init_drift
            # max_dev = 0.5 * |bias|, floored to avoid zero for stablecoins/hidden dims
            self.max_dev = jnp.maximum(jnp.abs(init_drift) * 0.5, 1e-4)
        else:
            self.bias = jnp.zeros(latent_dim)
            self.max_dev = jnp.full(latent_dim, 1e-4)

        key_mlp, _ = jax.random.split(key)

        self.mlp = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=latent_dim,
            width_size=hidden_dim,
            depth=1,
            activation=jnp.tanh,
            key=key_mlp,
        )
        # Scale final layer to 0.1x so initial MLP output is in [-0.2, 0.2],
        # keeping tanh in its linear regime (good gradients).
        old_final = self.mlp.layers[-1]
        small_w = old_final.weight * 0.1
        small_b = old_final.bias * 0.1 if old_final.bias is not None else None
        new_final = eqx.tree_at(
            lambda l: l.weight, old_final, small_w,
        )
        if small_b is not None:
            new_final = eqx.tree_at(
                lambda l: l.bias, new_final, small_b,
            )
        self.mlp = eqx.tree_at(
            lambda m: m.layers[-1], self.mlp, new_final,
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        return self.bias + self.max_dev * jnp.tanh(self.mlp(z))


class LatentDiffusionNetwork(eqx.Module):
    """Block-diagonal Cholesky diffusion on latent state.

    Asset block: full n_assets x n_assets lower-triangular (captures cross-asset
    correlations). Hidden block: diagonal n_hidden entries (independent noise for
    hidden dims, cross-coupling with observed dims via drift).

    Total MLP output size: n_assets*(n_assets+1)/2 + n_hidden
    (vs latent_dim*(latent_dim+1)/2 for a full Cholesky).

    A learnable ``log_scale`` (one per latent dim) handles magnitude: the MLP
    operates in O(1) range, log_scale brings outputs to the correct scale.
    """

    mlp: eqx.nn.MLP
    n_assets: int
    n_hidden: int
    log_scale: jax.Array  # (latent_dim,) per-dim output scale

    def __init__(
        self,
        n_assets: int,
        n_hidden: int,
        hidden_dim: int = 32,
        init_scale: jax.Array = None,
        *,
        key: jax.Array,
    ):
        self.n_assets = n_assets
        self.n_hidden = n_hidden
        latent_dim = n_assets + n_hidden

        # MLP output: lower-tri entries for asset block + diagonal for hidden block
        asset_tril_size = n_assets * (n_assets + 1) // 2
        out_size = asset_tril_size + n_hidden

        if init_scale is not None:
            # init_scale is (n_assets,) for asset dims; use mean for hidden dims
            asset_scale = jnp.asarray(init_scale)
            hidden_scale = jnp.full(n_hidden, jnp.mean(asset_scale))
            self.log_scale = jnp.log(jnp.concatenate([asset_scale, hidden_scale]))
        else:
            self.log_scale = jnp.zeros(latent_dim)

        self.mlp = eqx.nn.MLP(
            in_size=latent_dim,
            out_size=out_size,
            width_size=hidden_dim,
            depth=1,
            activation=jnp.tanh,
            key=key,
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        """Returns (latent_dim, latent_dim) block-diagonal lower-triangular Cholesky."""
        raw = self.mlp(z)
        n = self.n_assets
        n_h = self.n_hidden
        latent_dim = n + n_h
        scale = jnp.exp(self.log_scale)

        # Split output into asset block and hidden block
        asset_tril_size = n * (n + 1) // 2
        asset_raw = raw[:asset_tril_size]
        hidden_raw = raw[asset_tril_size:]

        # Build asset block: full lower-triangular
        L_asset = jnp.zeros((n, n))
        tril_indices = jnp.tril_indices(n)
        L_asset = L_asset.at[tril_indices].set(asset_raw)
        diag_idx = jnp.diag_indices(n)
        L_asset = L_asset.at[diag_idx].set(jax.nn.softplus(L_asset[diag_idx]) + 1e-6)

        # Build full block-diagonal matrix
        L = jnp.zeros((latent_dim, latent_dim))
        L = L.at[:n, :n].set(L_asset)
        # Hidden block: diagonal
        hidden_diag = jax.nn.softplus(hidden_raw) + 1e-6
        L = L.at[jnp.arange(n, latent_dim), jnp.arange(n, latent_dim)].set(hidden_diag)

        # Scale each row by per-dim scale factor
        L = L * scale[:, None]

        return L


class LatentNeuralSDE(eqx.Module):
    """Latent Neural SDE: dZ = f(Z)dt + g(Z)dW, Y = readout(Z).

    Evolves in extended latent space Z_t ∈ R^(n_assets + n_hidden), where
    extra hidden dimensions serve as implicit memory — they accumulate
    trajectory history through the dynamics, enabling regime-aware drift
    without explicit windowing or feature engineering.

    At initialisation (identity readout, small hidden init), this is
    equivalent to the standard NeuralSDE. Training gradually activates
    the hidden dimensions as needed.
    """

    encoder: Encoder
    drift: LatentDriftNetwork
    diffusion: LatentDiffusionNetwork
    readout: Readout
    n_assets: int
    n_hidden: int
    latent_dim: int  # = n_assets + n_hidden

    def __init__(
        self,
        n_assets: int,
        n_hidden: int = 4,
        hidden_dim: int = 32,
        init_diffusion_scale: jax.Array = None,
        init_drift_scale: jax.Array = None,
        *,
        key: jax.Array,
    ):
        self.n_assets = n_assets
        self.n_hidden = n_hidden
        self.latent_dim = n_assets + n_hidden

        key_enc, key_drift, key_diff = jax.random.split(key, 3)
        self.encoder = Encoder(n_assets, n_hidden, hidden_dim, key=key_enc)
        self.drift = LatentDriftNetwork(
            self.latent_dim, hidden_dim,
            init_drift=init_drift_scale, key=key_drift,
        )
        self.diffusion = LatentDiffusionNetwork(
            n_assets, n_hidden, hidden_dim,
            init_scale=init_diffusion_scale, key=key_diff,
        )
        self.readout = Readout(n_assets, self.latent_dim)


def save_latent_sde(sde: "LatentNeuralSDE", path: str) -> None:
    """Serialize a LatentNeuralSDE to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), sde)


def load_latent_sde(
    path: str,
    n_assets: int,
    n_hidden: int = 4,
    hidden_dim: int = 32,
) -> "LatentNeuralSDE":
    """Deserialize a LatentNeuralSDE from disk.

    Must provide architecture hyperparameters to reconstruct the pytree skeleton.
    """
    skeleton = LatentNeuralSDE(
        n_assets, n_hidden, hidden_dim,
        key=jax.random.PRNGKey(0),
    )
    return eqx.tree_deserialise_leaves(str(path), skeleton)


def save_sde(sde: NeuralSDE, path: str) -> None:
    """Serialize a NeuralSDE to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(path), sde)


def load_sde(
    path: str,
    n_assets: int,
    hidden_dim: int = 32,
    diagonal_only: bool = False,
    learn_drift: bool = False,
) -> NeuralSDE:
    """Deserialize a NeuralSDE from disk.

    Must provide architecture hyperparameters to reconstruct the pytree skeleton.
    """
    skeleton = NeuralSDE(
        n_assets,
        hidden_dim,
        diagonal_only=diagonal_only,
        learn_drift=learn_drift,
        key=jax.random.PRNGKey(0),
    )
    return eqx.tree_deserialise_leaves(str(path), skeleton)
