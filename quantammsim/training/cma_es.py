"""Pure-JAX CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

Follows Hansen's tutorial (arXiv:1604.00772). All functions are pure
and JIT-compatible. The ask/tell interface lets the caller control
evaluation (e.g. via vmap).

Typical usage::

    params = default_params(n)
    state = init_cmaes(x0, sigma0)
    for gen in range(max_gens):
        key, subkey = jax.random.split(key)
        pop = ask(state, subkey, params["lam"])
        fitness = evaluate(pop)          # caller's responsibility
        state = tell(state, pop, fitness, params)
        if should_stop(state, tol):
            break
    best = state.best_x
"""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random


class CMAESState(NamedTuple):
    """Immutable state of a CMA-ES run."""
    mean: jnp.ndarray        # (n,) distribution mean
    sigma: float              # step size (scalar)
    C: jnp.ndarray            # (n, n) covariance matrix
    p_sigma: jnp.ndarray      # (n,) conjugate evolution path (step-size)
    p_c: jnp.ndarray          # (n,) evolution path (covariance)
    gen: int                   # generation counter
    best_x: jnp.ndarray       # (n,) best solution found so far
    best_f: float              # best fitness value (minimization)
    eigenvalues: jnp.ndarray   # (n,) cached eigenvalues of C
    eigenvectors: jnp.ndarray  # (n, n) cached eigenvectors of C
    invsqrt_C: jnp.ndarray     # (n, n) C^{-1/2}


def default_params(n: int, lam: int = None) -> dict:
    """Return default CMA-ES hyper-parameters for problem dimension *n*.

    Population size λ = 4 + floor(3 · ln(n)), parent count μ = λ // 2.
    Weights, learning rates, and damping follow Hansen's defaults.

    Parameters
    ----------
    n : int
        Problem dimension.
    lam : int, optional
        Override population size. If None, uses Hansen's default.
        All dependent quantities (μ, weights, learning rates, damping)
        are recomputed from the given λ.
    """
    if lam is None:
        lam = 4 + int(math.floor(3 * math.log(n)))
    mu = lam // 2

    # Recombination weights (log-linear, normalised)
    raw_weights = jnp.array(
        [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
    )
    weights = raw_weights / jnp.sum(raw_weights)
    mu_eff = 1.0 / jnp.sum(weights ** 2)

    # Step-size adaptation
    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * jnp.maximum(0.0, jnp.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma

    # Covariance adaptation
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1.0 - float(c1),
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
    )

    # Expected length of N(0, I) vector
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

    return {
        "lam": lam,
        "mu": mu,
        "weights": weights,
        "mu_eff": float(mu_eff),
        "c_sigma": float(c_sigma),
        "d_sigma": float(d_sigma),
        "c_c": float(c_c),
        "c1": float(c1),
        "c_mu": float(c_mu),
        "chi_n": chi_n,
    }


def init_cmaes(mean: jnp.ndarray, sigma: float) -> CMAESState:
    """Initialise CMA-ES state from an initial mean and step size.

    All fields are explicit JAX arrays with dtypes derived from ``mean.dtype``,
    so the returned state is safe to use as ``lax.while_loop`` carry.
    """
    n = mean.shape[0]
    dtype = mean.dtype
    return CMAESState(
        mean=mean,
        sigma=jnp.asarray(sigma, dtype=dtype),
        C=jnp.eye(n, dtype=dtype),
        p_sigma=jnp.zeros(n, dtype=dtype),
        p_c=jnp.zeros(n, dtype=dtype),
        gen=jnp.int32(0),
        best_x=mean.copy(),
        best_f=jnp.asarray(jnp.inf, dtype=dtype),
        eigenvalues=jnp.ones(n, dtype=dtype),
        eigenvectors=jnp.eye(n, dtype=dtype),
        invsqrt_C=jnp.eye(n, dtype=dtype),
    )


def ask(state: CMAESState, key: jnp.ndarray, lam: int) -> jnp.ndarray:
    """Sample *lam* candidate solutions from the current distribution.

    Returns array of shape ``(lam, n)`` with the same dtype as ``state.mean``.
    """
    n = state.mean.shape[0]
    dtype = state.mean.dtype
    # Sample z ~ N(0, I), transform via C^{1/2}
    z = random.normal(key, shape=(lam, n), dtype=dtype)
    # C = B D^2 B^T  =>  C^{1/2} = B D B^T
    # population = mean + sigma * B D z^T
    D = jnp.sqrt(state.eigenvalues)  # (n,)
    # Transform: y_i = B @ diag(D) @ z_i
    y = z @ jnp.diag(D) @ state.eigenvectors.T  # (lam, n)
    population = state.mean + state.sigma * y
    return population


def tell(
    state: CMAESState,
    population: jnp.ndarray,
    fitness: jnp.ndarray,
    params: dict,
) -> CMAESState:
    """Update the CMA-ES state given the population and their fitness values.

    *fitness* should have shape ``(lam,)`` — lower is better (minimization).
    All arithmetic preserves ``state.mean.dtype`` to stay compatible with
    ``lax.while_loop`` carry constraints.
    """
    n = state.mean.shape[0]
    dtype = state.mean.dtype
    mu = params["mu"]
    # Cast weights to state dtype — default_params creates a JAX array whose
    # dtype follows the global x64 flag, which may differ from the state dtype.
    weights = params["weights"].astype(dtype)
    mu_eff = params["mu_eff"]
    c_sigma = params["c_sigma"]
    d_sigma = params["d_sigma"]
    c_c = params["c_c"]
    c1 = params["c1"]
    c_mu = params["c_mu"]
    chi_n = params["chi_n"]

    # Sort by fitness (ascending = best first for minimization)
    order = jnp.argsort(fitness)
    sorted_pop = population[order]

    # Best of this generation
    gen_best_x = sorted_pop[0]
    gen_best_f = fitness[order[0]]

    # Update elitist best
    improved = gen_best_f < state.best_f
    best_x = jnp.where(improved, gen_best_x, state.best_x)
    best_f = jnp.where(improved, gen_best_f, state.best_f)

    # Weighted recombination of top-μ
    selected = sorted_pop[:mu]  # (mu, n)
    new_mean = jnp.sum(weights[:, None] * selected, axis=0)

    # Evolution paths
    mean_diff = new_mean - state.mean
    invsqrt_C = state.invsqrt_C

    # Coefficients computed via Python math to stay weakly-typed and avoid
    # jnp.sqrt promoting to the default float dtype under x64.
    sqrt_csig = math.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
    sqrt_cc = math.sqrt(c_c * (2 - c_c) * mu_eff)

    # p_sigma = (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^{-1/2} * (mean_diff / sigma)
    p_sigma = (
        (1 - c_sigma) * state.p_sigma
        + sqrt_csig * invsqrt_C @ (mean_diff / state.sigma)
    )

    # Heaviside function for stalling detection
    p_sigma_norm = jnp.linalg.norm(p_sigma)
    gen_plus_1 = state.gen + 1
    threshold = (1.4 + 2.0 / (n + 1)) * chi_n * jnp.sqrt(
        1 - (1 - c_sigma) ** (2 * gen_plus_1)
    )
    # Cast bool→dtype instead of jnp.where with float literals (which would
    # default to float64 under x64, promoting downstream arrays).
    h_sigma = (p_sigma_norm < threshold).astype(dtype)

    # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (mean_diff / sigma)
    p_c = (
        (1 - c_c) * state.p_c
        + h_sigma * sqrt_cc * (mean_diff / state.sigma)
    )

    # Covariance matrix update
    # Rank-1 update
    rank1 = c1 * jnp.outer(p_c, p_c)
    # Correction for h_sigma = 0 case
    rank1_correction = c1 * (1 - h_sigma) * c_c * (2 - c_c) * state.C

    # Rank-μ update
    diff_scaled = (selected - state.mean) / state.sigma  # (mu, n)
    rank_mu = c_mu * jnp.sum(
        weights[:, None, None] * (diff_scaled[:, :, None] * diff_scaled[:, None, :]),
        axis=0,
    )

    new_C = (
        (1 - c1 - c_mu) * state.C
        + rank1
        + rank1_correction
        + rank_mu
    )

    # Step-size update (CSA)
    new_sigma = state.sigma * jnp.exp(
        (c_sigma / d_sigma) * (p_sigma_norm / chi_n - 1)
    )

    # Eigendecomposition of C (for next generation's sampling and C^{-1/2})
    # Force symmetry to avoid numerical drift
    new_C = (new_C + new_C.T) / 2
    eigenvalues, eigenvectors = jnp.linalg.eigh(new_C)
    # Clamp eigenvalues to avoid numerical issues
    eigenvalues = jnp.maximum(eigenvalues, 1e-20)
    # C^{-1/2} = B @ diag(1/sqrt(D)) @ B^T
    new_invsqrt_C = eigenvectors @ jnp.diag(1.0 / jnp.sqrt(eigenvalues)) @ eigenvectors.T

    return CMAESState(
        mean=new_mean,
        sigma=new_sigma,
        C=new_C,
        p_sigma=p_sigma,
        p_c=p_c,
        gen=gen_plus_1,
        best_x=best_x,
        best_f=best_f,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        invsqrt_C=new_invsqrt_C,
    )


def _should_stop_jax(state: CMAESState, tol: float = 1e-8) -> jnp.ndarray:
    """Check termination criteria, returning a JAX bool (for use in ``lax.while_loop``).

    Stops when:
    - Step size × max eigenvalue < tol (distribution has collapsed)
    - Condition number of C exceeds 1e14
    """
    max_eigval = jnp.max(state.eigenvalues)
    min_eigval = jnp.min(state.eigenvalues)
    cond = max_eigval / jnp.maximum(min_eigval, 1e-30)

    size_converged = state.sigma * jnp.sqrt(max_eigval) < tol
    ill_conditioned = cond > 1e14

    return size_converged | ill_conditioned


def should_stop(state: CMAESState, tol: float = 1e-8) -> bool:
    """Check termination criteria (Python bool for use in Python loops)."""
    return bool(_should_stop_jax(state, tol))


def run_cmaes(
    init_state: CMAESState,
    rng_key: jnp.ndarray,
    eval_fn,
    params: dict,
    n_generations: int,
    tol: float = 1e-8,
) -> CMAESState:
    """Run CMA-ES via ``lax.while_loop``.  JIT-compatible.

    Fuses the ask → eval → tell loop into a single XLA program, eliminating
    per-generation Python dispatch overhead.

    Parameters
    ----------
    init_state : CMAESState
        Initial state from :func:`init_cmaes`.
    rng_key : jax.Array
        PRNG key; split internally each generation.
    eval_fn : callable
        ``(lam, n) -> (lam,)`` fitness function (lower is better).
    params : dict
        CMA-ES hyper-parameters from :func:`default_params`.
    n_generations : int
        Maximum number of generations.
    tol : float
        Convergence tolerance passed to :func:`_should_stop_jax`.

    Returns
    -------
    CMAESState
        Final state after convergence or ``n_generations``.
    """
    lam = params["lam"]

    def cond_fn(carry):
        state, _key = carry
        return (~_should_stop_jax(state, tol)) & (state.gen < n_generations)

    def body_fn(carry):
        state, key = carry
        key, subkey = random.split(key)
        pop = ask(state, subkey, lam)
        fitness = eval_fn(pop)
        state = tell(state, pop, fitness, params)
        return (state, key)

    final_state, _ = jax.lax.while_loop(cond_fn, body_fn, (init_state, rng_key))
    return final_state
