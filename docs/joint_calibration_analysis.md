# Joint Calibration Analysis: MLP Capacity vs Identification

## Training results (2026-03-10)

### R2 progression across model architectures

| Attempt | Architecture | Noise params | Total params | Median R2 | Joint loss |
|---|---|---|---|---|---|
| Structural MoE (numpyro) | 3 archetypes x 8 coeffs | 24 | ~40 | -0.70 | - |
| Linear joint | SharedLinearNoiseHead | 63 | 63 | -0.15 | 9.62 |
| MLP noise joint | MLPNoiseHead(hidden=16) | 255 | 255 | 0.01 | 9.39 |
| Full MLP | MLPHead(cad,16) + MLPNoiseHead(16) | 255 | 377 | -0.02 | 8.59 |
| Option C (per-pool) | PerPoolNoiseHead | 37x8=296 | 37x9=333 | 0.61 | 1.25 (median) |

The direction is clear: more capacity on the noise side helps substantially
(-0.70 -> -0.15 -> 0.01). Adding cadence capacity (MLP noise -> full MLP)
improved joint loss (9.39 -> 8.59) but not per-pool R2 (-0.02), and didn't
converge within 500 iterations.

### Convergence concern

The full MLP explicitly failed to converge (scipy `success=False`). The MLP
noise model converged but only reduced loss from 12.70 to 9.39 — a 26%
reduction vs the linear baseline's 99.5% reduction (2011.77 -> 9.62). This
suggests the MLPs are undertraining.

Current optimizer settings:
- L-BFGS-B with maxiter=500
- ftol=1e-10, gtol=1e-8
- maxcor=10 (L-BFGS memory, scipy default)
- alpha=0.01 for all heads (L2 regularization on weights)
- hidden=16 for all MLPs
- He init for W1, W2=0, b2=pooled OLS / mean of Option C

### Why the MLPs may not be converging

1. **maxiter=500 is low for 255-377 params.** L-BFGS-B typically needs
   O(1000-5000) iterations for MLP-scale problems. The linear model with
   63 params converges easily in 500; the MLP with 377 params does not.

2. **maxcor=10 may be too small.** The default L-BFGS memory of 10 past
   gradients may not provide a good enough Hessian approximation for 377
   parameters. Increasing to 20-50 can help.

3. **Regularization alpha=0.01 may be wrong.** With 37 pools and 255 noise
   params, the model is overparameterized (255/37 ≈ 7 params per pool).
   alpha=0.01 might be too weak (overfitting some pools, underfitting
   others) or too strong (preventing the MLP from expressing the necessary
   nonlinearity). This is the most important hyperparameter to sweep.

4. **W2=0 initialization creates a flat starting surface.** Since the MLP
   starts as a constant function (output = b2 everywhere), L-BFGS-B must
   first learn to differentiate between pools. The initial gradients
   through W1 are informative (He init + backprop through ReLU), but the
   first few iterations may be slow compared to the linear model which
   starts from an OLS warm-start.

5. **Dead ReLU units.** With He init and k_attr=6 features, some hidden
   units may have all-negative pre-activations across the 37 pool
   attribute vectors, making them permanently dead with zero gradient.

6. **Per-pool loss weighting.** All observations contribute equally.
   USDC/WETH (1757 obs) dominates RDNT/WETH (89 obs) by 20x. The
   optimizer may be fitting a few high-obs pools at the expense of many
   low-obs ones.

## Diagnosis: identification vs convergence

Two distinct problems:

1. **Convergence problem** (addressable via hyperparameters):
   The MLP isn't reaching its minimum. Fix: more iterations, better
   hyperparameters, multiple restarts.

2. **Identification problem** (addressable via architecture):
   Even at the minimum, the shared mapping can't match per-pool R2.
   37 pools is tiny for a nonlinear model. Cadence is idiosyncratic.
   Fix: DeltaHead (per-pool residuals with shrinkage), better features.

These are **independent** problems that compound. We should fix convergence
first (hyperparameter sweep) to understand the true capacity of the current
architecture before adding structural complexity.

## Hyperparameter sweep design

### Parameters to sweep

| Parameter | Current | Sweep values | Rationale |
|---|---|---|---|
| maxiter | 500 | 500, 2000, 5000 | Primary convergence bottleneck |
| alpha (noise) | 0.01 | 0.0001, 0.001, 0.01, 0.1 | Controls overfitting vs underfitting |
| alpha (cadence) | 0.01 | 0.001, 0.01, 0.1 | Separate from noise reg |
| hidden | 16 | 8, 16, 32 | Capacity vs overfitting |
| maxcor | 10 | 10, 30 | L-BFGS Hessian quality |
| loss_type | l2 | l2, huber | Outlier robustness |

### Sweep strategy

Full grid is 3 x 4 x 3 x 3 x 2 x 2 = 432 runs. Too many.

**Phase 1: Fix convergence (1D sweeps)**
- Sweep maxiter = [500, 2000, 5000] with defaults. Cheapest diagnostic.
- If 5000 converges, use that going forward.

**Phase 2: Regularization (most important)**
- alpha_noise x alpha_cad grid: 4 x 3 = 12 runs at converged maxiter.
- Evaluate both joint loss AND per-pool median R2.

**Phase 3: Architecture**
- hidden = [8, 16, 32] at best alpha settings: 3 runs.
- loss_type = [l2, huber] at best settings: 2 runs.
- maxcor = [10, 30] at best settings: 2 runs.

Total: ~22 runs, each ~2-5 min = ~1-2 hours.

### Metrics to track per run

- Joint loss (final)
- Joint loss (init) — sanity check
- Converged (bool)
- Number of L-BFGS iterations used
- Per-pool median R2
- Per-pool mean R2
- Per-pool R2 distribution (10th, 25th, 50th, 75th, 90th percentiles)
- Wall time

### What success looks like

- Converged = True for the full MLP
- Joint loss < 8.0 (below current 8.59)
- Per-pool median R2 > 0.3 (closing the gap toward Option C's 0.61)
- The R2 improvement should be spread across pools, not concentrated

## Features / data that would help

### Missing pool attributes (from docs)

Current features (k_attr=6 after chain dummy removal):
log_fee, mean_log_tvl, log_mcap_product, has_stable, same_asset_type,
weight_imbalance.

These describe what the pool IS but not the market around it. Cadence is
driven by arbitrage frequency, which depends on:

| Missing feature | Why it matters | Source | Effort |
|---|---|---|---|
| Block time | Directly limits minimum cadence. Arb=0.25s vs Main=12s | Static per chain | Trivial |
| Mean pair volatility | Pool-level (not obs-level) vol predicts arb intensity | Binance minute data (loaded) | Small |
| CEX daily volume | More CEX vol = more arb opportunities | Binance API | Medium |
| Competing DEX pools | More pools for same pair = faster arb | Balancer subgraph | Medium |
| Pool routing share | Dominant pool gets arbitraged first | DEX aggregator data | Hard |
| Mean daily swap count | Direct proxy for pool activity | Panel data | Small |

The pair-intrinsic formula bias (1.26-2.22x) documented in
noise_calibration_review.md is the largest unexplained variance source.
It varies with pair liquidity characteristics in ways that the current
token classification doesn't capture. CEX volume/depth would help.

### Observation-level features (x_obs, K_OBS=8)

Current: [1, log_tvl_lag1, log_sigma, tvl*sigma, tvl*fee, sigma*fee,
dow_sin, dow_cos]

Missing:
- Rolling CEX volume (daily) — high volume days have more noise/organic flow
- Gas price that day (mainnet) — affects whether arbs execute
- Market regime (rolling momentum) — trending vs mean-reverting
- Number of swaps that day — direct activity measure

### Time-varying dynamics

Panel spans 2021-2026. MEV dynamics changed dramatically:
- Flashbots launched mid-2021
- L2s matured 2023-2024
- EIP-4844 (March 2024) dropped L2 gas costs
The current model assumes constant cadence per pool over this period.

## Structural improvements (post-sweep)

### DeltaHead (per-pool residuals with shrinkage)

Most important structural change. For cadence:
```
log_cadence_i = f(x_attr_i) + delta_i
regularization: alpha_shared * ||W||^2 + alpha_delta * sum(delta_i^2)
```

At alpha_delta=0: pure per-pool (Option C)
At alpha_delta=inf: pure shared (current joint)
Cross-validate alpha_delta.

For new pools: predict f(x_attr_new) with delta=0.

This is essentially a mixed-effects model fitted end-to-end through the
grid interpolation loss.

### Per-pool loss weighting

Weight each pool's contribution by 1/sqrt(n_obs_i) to equalize pool-level
influence. Currently USDC/WETH (1757 obs) has 20x the influence of any
Sonic pool (89 obs).

### Hybrid: per-pool cadence + shared noise

Cadence is idiosyncratic (LOO R2 = 0.24 at best). Noise structure is
more regular (hierarchical model R2 = 0.71 on total volume). Natural split:
- Cadence: per-pool (Option C)
- Noise: shared MLP (generalizable)
- Gas: fixed to chain values

### Sensitivity analysis (the decision point)

Before investing more in mapping improvement: does reCLAMM optimal
concentration change materially when cadence varies +/-50%? This is
recommendation #1 in calibration_results.md, noise_calibration_review.md,
and joint_calibration_design.md. Still not done.

If the optimum is robust, the current pipeline (Option C + Ridge LOO) is
already sufficient and further mapping improvement is nice-to-have.

## Priority order

1. **Hyperparameter sweep** — fix convergence before changing architecture
2. **DeltaHead** — if R2 gap persists post-sweep, this is the minimal
   structural change
3. **Per-pool loss weighting** — simple fix, helps all joint models
4. **Add block_time and mean_pair_volatility** — high-signal, low-effort
   features
5. **Sensitivity analysis** — the real decision point for whether any of
   this matters for the downstream task
