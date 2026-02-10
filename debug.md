# JAX TFMM AMM Simulator — Gradient-Based Training Review, Failure Modes, and Fixes

This document consolidates the findings from my earlier reviews of your attached simulator/training stack and **corrects or updates** any points where the first pass was necessarily speculative. It is intentionally long and detailed, per request, and is written as an engineering / research diagnosis note rather than a short summary.

**Scope of files reviewed (attached):**
- `jax_runners.py`
- `backpropagation.py`
- `forward_pass.py`
- `windowing_utils.py`
- `historic_data_utils.py`
- `creator.py`
- `base_pool.py`
- `TFMM_base_pool.py`
- `momentum_pool.py`
- `fine_weights.py`
- `linear_interpolation.py`
- `quantamm_reserves.py`
- `param_utils.py` (already reviewed earlier)

> Notes:
> - You requested to ignore two previously listed fixes (initial pre-exp scaling init and the arb-frequency != 1 forward-pass issue). They are therefore **not included** in the “Fixes” sections below and are not relied upon for conclusions here.
> - This document focuses on **issues that can plausibly explain training instability / sporadic degradation** even when the overall simulator logic is conceptually correct.

---



**Priority issues (impact order): Optax LR scaling bug; Ignored dynamic learning-rate input; Hard constraints + STE mismatch; Missing/optional global grad clipping; Hard max signature selection in optimal arb; Hard arb profitability gating; Coarse/bout misalignment risk; Geometric product numerics; Objective edge cases; Dtype/precision mismatch; Feasibility hard clamps in arb helpers; External trade branching.**


## 0) Working mental model (TFMM + CFMM per timestep)

Your stated model is consistent with the code:

1. At each timestep, the pool behaves as a standard CFMM/weighted invariant AMM with **fixed weights**.
2. Between timesteps, “coarse” strategy weights update using a rule (MomentumPool etc).
3. “Fine” weights are computed by **linear interpolation** of coarse outputs over each `chunk_period`.
4. Reserves evolve due to **price changes + arbitrage** (and optional fees/noise trades).
5. Training optimizes strategy parameters using JAX autodiff + Optax.

This is the correct conceptual abstraction for TFMM-style AMMs and is reflected across `TFMM_base_pool.py`, `fine_weights.py`, and `quantamm_reserves.py`.

---

## 1) Optax LR scaling bug (double-applied learning rate / schedule)

### 6.2 Confirmed double-scaling of LR in the optimizer chain
Your optimizer chain is created as:

```python
base_optimizer = _create_base_optimizer(settings["optimiser"], base_lr)
lr_schedule = _create_lr_schedule(settings)
optimizer_chain = optax.chain(base_optimizer, optax.scale_by_schedule(lr_schedule))
```

But `_create_base_optimizer(..., base_lr)` already embeds the learning rate (for Adam/SGD), and `_create_lr_schedule` returns an absolute learning rate schedule that includes `base_lr`.

So effective scaling can become roughly:

- Adam uses `base_lr`
- schedule scales again by `base_lr` (for constant schedule) → `base_lr^2`

This can cause:
- extremely small updates if base_lr < 1 (squared is smaller)
- surprisingly large updates if base_lr > 1
- confusing hyperparameter tuning because actual LR differs from what settings suggest

**Fix (one clean choice):**
- Either set Adam’s LR to 1.0 and use `scale_by_schedule`,
- OR use Adam with schedule directly and remove `scale_by_schedule`,
- BUT avoid both at the same time.

Example clean approach:

```python
optimizer = optax.adam(learning_rate=lr_schedule)
```

Optionally add:
- `optax.clip_by_global_norm(clip_norm)`
- `optax.contrib.reduce_on_plateau(...)`

## 2) Ignored dynamic learning-rate input (Python LR decay has no effect)

### 6.1 Your external “learning_rate” argument is ignored
The optax update function signature is:

```python
def update(params, start_indexes, learning_rate, opt_state=None):
    objective_value, grads = value_and_grad(batched_objective)(params, start_indexes)
    updates, new_opt_state = optimizer.update(...)
    new_params = optax.apply_updates(params, updates)
    return new_params, objective_value, params, grads, new_opt_state
```

The `learning_rate` argument is never used.

**Consequence:** any learning-rate decay you implement at the Python training loop level (e.g., plateau-based `local_learning_rate` changes) does nothing when you are using Optax schedules and this update function.

This can produce the exact “sporadic” perception:
- you believe LR changed but training still diverges,
- you attribute behavior to randomness rather than the fact LR control is inert.

**Fix options:**
- Remove the unused LR plumbing and rely only on optax schedule + reduce_on_plateau.
- Or use `optax.inject_hyperparams(...)` so LR can be passed dynamically.

## 3) Hard constraints + STE mismatch (weights pinned at min/max, surrogate-gradient drift)

### 3.3 Coarse weights: constraints and normalization (fine_weights.py)

The coarse weight evolution is built by scanning coarse outputs through:

- `_jax_calc_coarse_weights(...)`
- `_jax_calc_coarse_weight_scan_function(...)`

This stage applies multiple guardrails:
- sum-to-one normalization
- min/max bounds
- max change per step
- correction by adjusting one index (argmax)

#### Important gradient property: constraint mismatch
Two patterns create a forward/gradient mismatch:

1. **Argmax-based “sum to one” correction**
   - you choose the max-weight component and adjust it to absorb residual error,
   - the argmax selection is non-differentiable,
   - you apply `stop_gradient` to keep forward stable but gradient consistent.

This keeps weights summing exactly to 1 in forward simulation but introduces an approximation in gradient flow.

2. **Hard clipping and “STE” modes**
   - clipping min/max and max-change creates piecewise linear or flat gradients,
   - straight-through estimators propagate gradients as-if unclipped.

This can produce training dynamics like:
- gradients push parameters beyond feasible bounds,
- forward clips and does not improve objective,
- Adam’s momentum accumulators become “out of sync” with the true constrained mapping,
- parameter drift / oscillation / eventual NaNs.

This is a classic “optimizer believes a different model than the forward pass” failure mode.

### 8.2 Hard constraints + STE mismatch
You rely on hard constraints to keep weights feasible:
- min/max
- max change

This is essential in a live AMM, but it creates a non-smooth constrained optimization problem.

If you use STEs, the optimization is no longer optimizing the true constrained function; it is optimizing a surrogate.

Mitigation:
- smooth parameterization (e.g., softmax weights + bounded deltas)
- reduced reliance on STE
- smaller LR + stronger clipping

## 4) Missing/optional global gradient clipping (unbounded spikes corrupt Adam state)

### 6.4 Gradient clipping is not always on
Gradient clipping is optional in settings.

Given the non-smoothness of arbitrage and the “ratio objective”, clipping should be treated as an essential stabilizer, not an optional feature.

---

## 5) Hard max signature selection in optimal arb (argmax regime switching)

### 4.1b Additional high-impact discontinuity: signature / direction selection by hard max (optimal_n_pool_arb.py)

After reviewing `optimal_n_pool_arb.py`, the earlier conclusion that “arb introduces hard discontinuities” is **stronger than originally stated**.
There is not only the *profitability gate* (trade vs no trade), but also a second, often-dominant discontinuity inside the computation of the “optimal” arbitrage trade itself.

#### What the code is doing conceptually
For the fee-aware “optimal” arb, you build a large set of candidate trade directions (“signatures”), compute the candidate trade vector for each signature, compute each candidate profit, and then select the single best candidate.

In code, the *selection step* is effectively:

- compute `profits` for each candidate signature
- choose the max-profit candidate
- output the corresponding trade vector

A simplified form of the selection logic is:

```python
profits = -(overall_trades * local_prices).sum(-1)
mask = jnp.where(profits == jnp.max(profits), 1.0, 0.0)
optimal_trade = mask @ overall_trades
```

This is functionally an **argmax policy reveal** over discrete regimes. Small changes in weights/params/prices can change which signature is best,
causing a **discrete jump** in the chosen trade direction and magnitude.

#### Why this matters for gradient-based training
Even if you completely remove the outer “arb profitable?” gate, this signature-selection step is itself discontinuous:

- The map `profits -> argmax(profits)` is not differentiable.
- The selected trade can change abruptly when the profit ordering changes.
- The gradient field becomes high-variance and “spiky,” especially early in training when parameters are moving quickly.

This is fully consistent with “sporadic success” (some runs find a basin where the best signature is stable; others repeatedly cross signature boundaries).

#### Additional numerical brittleness: equality-to-max mask
Using:

```python
profits == jnp.max(profits)
```

introduces additional edge cases:
- ties can yield multi-hot masks (multiple candidates selected)
- float comparisons can behave unexpectedly near equality due to precision
- gradients through equality comparisons are effectively zero almost everywhere

This can add instability beyond the usual argmax issue.

#### Training-only stabilizations (recommended)
If your goal is *training stability* rather than exact discrete economics during optimization, there are established fixes:

**Option 1 — Softmax mixture (“expected trade”)**
Instead of choosing a single signature, form a convex mixture of candidate trades:

- `p = softmax(profits / temperature)`
- `trade = p @ overall_trades`

This yields a smooth function of profits and therefore a smooth function of params (through profits).

**Option 2 — Straight-through argmax (hard forward, smooth backward)**
If you want the forward pass to remain “choose one signature,” but still want smoother gradients:

- `hard = one_hot(argmax(profits))`
- `soft = softmax(profits / temperature)`
- `p = hard + stop_gradient(soft - hard)`
- `trade = p @ overall_trades`

Forward simulation behaves like a true argmax selection, but gradients follow the softmax surrogate.

**Option 3 — Use argmax/one_hot instead of equality-to-max**
If you refuse any smoothing, at minimum replace `profits == max(profits)` with:

- `idx = argmax(profits)`
- `mask = one_hot(idx)`

This does not fix the discontinuity, but removes multi-hot tie hazards.

---

## 6) Hard arb profitability gating (trade vs no-trade discontinuity)

### 4.1 Core non-smoothness: “do arb trade or not”

You compute profitability:

```python
profit_to_arb = -(optimal_arb_trade * prices).sum() - arb_thresh
arb_external_rebalance_cost = 0.5 * arb_fees * (abs(optimal_arb_trade) * prices).sum()
arb_profitable = profit_to_arb >= arb_external_rebalance_cost
reserves = jnp.where(arb_profitable, post_price_reserves, prev_reserves)
```

This is a *hard discontinuity* in the forward function.

Even if `optimal_arb_trade` is smooth, the final reserves jump between two branches.

**Training implication:**
- gradient is effectively “all or nothing” across the profitability boundary,
- small parameter changes can flip the branch,
- objective surface contains cliffs → high-variance gradients,
- Adam steps can bounce between regimes.

This is one of the most plausible reasons you observe sporadic learning: some parameter regimes sit inside smooth basins where arb decisions are stable; others sit near boundary cliffs.

#### Fix (training-only smoothing)
A common strategy is “soft gating” during training:

```python
k = 50.0  # tune
gate = sigmoid(k * (profit_to_arb - arb_external_rebalance_cost))
reserves = gate * post_price_reserves + (1 - gate) * prev_reserves
```

Then switch back to hard gating for evaluation.


> **Additional context (from later discussion):** even if “arbitrage is always on” in the sense that the arb module runs every step, the *trade* is often near-zero when prices and weights are stable.  
> The training-relevant non-smoothness is therefore not “arb is sometimes called and sometimes not,” but the **regime boundaries** around (a) whether the computed trade magnitude is exactly zero, (b) which trade direction/signature is chosen, and (c) feasibility/validity clamping.  
> Long chunk periods reduce how often weights change, but they do not remove the minute-level arb regime boundaries induced by price movement (and by the signature-selection mechanism).


## 7) Coarse/bout misalignment risk (chunk slicing vs minute start indexes)

### 3.2 TFMMBasePool coarse window slicing and alignment

In `TFMM_base_pool.py`, you slice coarse outputs corresponding to the selected training bout. This is where the “coarse alignment” risk emerges.

You compute:

```python
start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)
```

This is float division then truncation.

**Risk:** If `start_index[0]` is not guaranteed to be a multiple of `chunk_period`, then coarse alignment is ambiguous:
- `start_index=100`, `chunk_period=60` → 100/60 = 1.66 → trunc → 1  
  Your coarse slice starts at time 60, but the bout starts at time 100.

This can cause:
- inconsistent coarse context for the same real bout location,
- noisy learning because the sampled windows do not correspond to consistent coarse episodes,
- “random” good/bad behavior depending on where the batch starts inside a chunk.

#### Does your sampling enforce alignment?
No. `windowing_utils.get_indices(...)` samples uniformly/exponentially in minute space and does not enforce chunk alignment.

So this issue is **not ruled out** and is still a high-likelihood contributor to gradient noise.

**Fix options:**
1. Change to integer floor-div:
   ```python
   start_index_coarse = (jnp.floor_divide(start_index[0], chunk_period).astype(jnp.int64), 0)
   ```
2. Enforce `start_index` sampled in multiples of `chunk_period` in `get_indices(...)`.
3. If you intentionally want “within-chunk random starts”, then you need to explicitly offset fine weight generation to match within-chunk position (otherwise your interpolation schedule is implicitly anchored to chunk boundaries, not the true bout start).

## 8) Geometric product numerics (product-of-powers under/overflow; noisy grads)

## 1) Targeted re-check: “product-of-powers → log-domain” suspicion

### 1.1 Where it appears in your code

In `quantamm_reserves.py` the simulator computes the geometric price product across assets with expressions of the form:

```python
price_change_ratio = prices / prev_prices
price_product_change_ratio = jnp.prod(price_change_ratio ** prev_weights)
```

and similarly later:

```python
price_change_ratio = prices / quoted_prices
price_product_change_ratio = jnp.prod(price_change_ratio ** weights)
```

These are used to derive “reserve ratios” and the post-arb reserve updates.

### 1.2 Is this a real issue?

**Yes, it is a legitimate and still-unruled-out numerical stability issue.**  
It may or may not be the dominant instability driver depending on:
- number of assets (2 vs many),
- magnitude of price moves,
- whether any price ratios are near 0,
- how often the simulation hits edge-case states.

Even if you mostly run with 2 assets, the operation has two important numerical characteristics:

#### (A) Underflow / overflow risk
`jnp.prod(x_i)` can underflow to 0 or overflow to inf if the product is extreme.

Even with 2 assets, this can happen if:
- `price_change_ratio` is extremely large/small (market jumps, or data contamination),
- weights push exponents in the wrong direction,
- you get near-zero intermediate reserves → quoted prices become huge → ratios explode.

#### (B) The gradient is inherently “log-sensitive”
The derivative w.r.t. weights is proportional to `log(price_change_ratio)`.  
In the power-product form, the gradient can become numerically noisy because:
- `x**w` itself can lose precision for extreme `x`,
- and the product amplifies relative error.

**In log-domain, both the forward and gradient are substantially more stable.**

### 1.3 Recommended fix (log-domain computation)

Replace:

```python
price_product_change_ratio = jnp.prod(price_change_ratio ** weights)
```

with the log-domain stable form:

```python
eps = 1e-12
log_price_product = jnp.sum(weights * jnp.log(price_change_ratio + eps))
price_product_change_ratio = jnp.exp(log_price_product)
```

**Why this is superior:**
- avoids product underflow/overflow,
- turns multiplicative structure into additive,
- stabilizes gradients (especially for extreme price ratios),
- behaves better for many assets.

### 1.4 Correctness preservation
Mathematically:

\[
\prod_i (x_i^{w_i}) \equiv \exp\left(\sum_i w_i \log x_i\right)
\]

So this fix is a pure numerical improvement: the forward pass is equivalent (up to epsilon handling).

---

### 4.3 Numerically sensitive invariants
Your reserve updates rely on geometric price products and reserve ratios. These are mathematically fine, but in practice sensitive to:
- near-zero reserves (division spikes),
- extreme price moves (ratio spikes),
- weights near bounds (exponent sensitivity),
- products across many assets (under/overflow).

This is why the log-domain product fix is still recommended.

---

## 9) Objective edge cases (daily log sharpe empty/low-std sensitivity)

## 5) Objective: daily log sharpe specifics and failure modes

You indicated the training objective is `daily_log_sharpe` (legacy sharpe variants ignored). This objective is more stable than classic sharpe in some ways, but it still has sharp edges.

### 5.1 Common failure mode: insufficient daily points
If `values` contains fewer than ~2 daily points (or fewer than the minimum required for non-empty daily returns), then:

- `log_rets` can be empty,
- mean/std become NaN,
- objective becomes NaN,
- gradients become NaN,
- NaN reinit logic triggers and training appears unstable.

Even if you “usually” use long bouts, it is worth hardening.

### 5.2 Gradient explosion mode: small std
Even with epsilon, if daily returns are nearly constant, std is tiny and ratio can become large. This can be a major source of exploding gradients.

Two mitigations:
1. Use a smoother denominator: `sqrt(std^2 + c)`
2. Clip the objective magnitude during training

### 5.3 Practical advice: stabilize objective before changing simulator
For gradient-based learning, objective stability usually has higher ROI than simulator micro-optimizations:
- the simulator can be correct but harsh,
- objectives are where NaNs typically appear first.

---

## 10) Dtype/precision mismatch (float64 simulator + optimizer state interactions)

## 7) Precision and dtype interactions: float64 simulator + float32 optimizer state

You explicitly enable x64 in some modules. That means:
- many intermediate values, weights, reserves, grads are float64
- Optax’s internal state and certain operations are typically float32 oriented

You cast only the `value` argument to float32 but do not cast grads.

This may cause:
- silent dtype promotion or casting inside optax
- slower compilation and larger memory
- subtle numeric differences that can make training “touchier”

Two stable patterns:
1. **Fully float32 training** (if simulator tolerates it)
2. **True float64 optimizer state** (more expensive; sometimes necessary)

If you keep float64 for simulator correctness:
- ensure optimizer is configured to handle float64 cleanly (or explicitly cast updates and state).

---

## 11) Feasibility hard clamps in arb helpers (invalid-trade → zero discontinuities)

### 4.1c Additional discontinuities: validity checks and hard clamping inside arb helper functions

Within `optimal_n_pool_arb.py`, candidate trades are also filtered by hard validity conditions such as:

- post-trade reserves must remain strictly positive
- invariants / constants must not degrade past a slack threshold

Invalid candidates are then hard-clamped to **zero trade** via `jnp.where(...)`.

These hard validity masks are correct from a “never break the AMM state” standpoint, but they add further kinks:

- trade outputs can jump between “some vector” and “exactly zero”
- gradients can vanish abruptly at feasibility boundaries

In practice, these are usually secondary compared to the signature argmax and profitability gate, but if your training frequently drives into feasibility limits,
they can become dominant instability sources.

---

### 4.1d Smaller kink: absolute value in arb external cost calculation

Your arb external rebalancing cost uses `abs(trade)` as part of the cost proxy:

```python
cost ~ (abs(optimal_trade) * prices).sum()
```

This introduces a non-smooth kink at zero (the derivative of `abs` changes sign). This is generally less damaging than `argmax` and hard `where` gates,
but it can still contribute to noisy gradients around “almost no arb” conditions.

---

## 12) External trade branching (additional regime switches via `lax.cond`)

### 4.1e External trade path uses conditional branching (G3M_trades.py)

In `G3M_trades.py` you use `lax.cond(...)` to decide whether to apply an external trade computation at a step.

This is the correct JAX tool for conditional control flow, but it also introduces a piecewise function boundary.
If external trade mode is active during training, be aware that it adds yet another regime switch similar in character to the arb gate.

In most pipelines, this is not the primary issue (because the condition is often static or rarely toggled), but it is part of the cumulative “piecewise surface”
that makes this training problem intrinsically harder than typical smooth ML models.

## 13) System stack: forward pass and gradient flow end-to-end

## 2) System stack: forward pass and gradient flow end-to-end

This section walks through the training-time forward pass as it is executed under JIT + vmap + optax.

### 2.1 High-level training call graph

**Training step:**
- `jax_runners.py` constructs a “partial” forward:
  - `partial_training_step = Partial(forward_pass, prices=..., static_dict=..., pool=...)`
- `backpropagation.py` wraps this into:
  - `batched_partial_training_step = jit(vmap(partial_training_step, in_axes=(None, 0)))`
  - `batched_objective = jit(lambda params, start_indexes: mean(batched_partial_training_step(params, start_indexes)))`
  - `update = update_factory_with_optax(batched_objective, optimizer)`

So, gradient flows as:

`params → forward_pass → reserves → value series → objective scalar → grad(params) → optax.update → apply_updates`

### 2.2 Shapes and axes of vectorization

Let:
- `A = n_assets`
- `B = bout_length`
- `T = total dataset length`

Your forward pass uses minute-level prices and returns a scalar objective per window (“bout”).  
At training time:
- `start_indexes` is shape `(batch_size, 2)` for historic data
- `batched_partial_training_step(params, start_indexes)` returns shape `(batch_size,)`
- objective is `mean(...)` producing a scalar.

So:
- `grad` is over params only, aggregated across the sampled batch.

### 2.3 JAX features used correctly
You are using several advanced JAX patterns appropriately:

- `vmap` over window start indices (data windows) while sharing params.
- `lax.scan` inside weight builders and reserve evolution (where appropriate).
- `jit` on pure functions to compile the graph.
- `static_argnums` in forward pass to keep pool and static config static.

The overall structure is idiomatic JAX.

---

## 14) Pool weight pipeline details (remaining)

### 3.4 Fine weights via linear interpolation (linear_interpolation.py)

Once you compute:
- `actual_starts[t]` (start-of-chunk weight)
- `scaled_diffs[t]` (minute-step increments)

You build fine weights with per-chunk interpolation:

- interpolate over `weight_interpolation_period`
- then hold end state for the rest of the chunk

This matches your design intent and is consistent.

**Numerical note:** interpolation itself is smooth and differentiable, so it is rarely the instability source. The instability is more often caused by:
- coarse constraint operations,
- arbitrary chunk/bout alignment,
- downstream reserve/arbitrage non-smoothness.

---

## 15) Reserve simulation details (remaining)

### 4.2 Additional regime switches (fees vs no-fees)
You use:

```python
optimal_arb_trade = jnp.where(fees_are_being_charged, optimal_arb_trade_fees, optimal_arb_trade_zero_fees)
```

This is safe if `gamma` is static; it is essentially a compile-time constant branch.

## 16) Still-not-ruled-out instability mechanisms (complete list)

## 8) Still-not-ruled-out instability mechanisms (and why they matter)

This section lists all remaining plausible causes of unstable learning that are not “confirmed bugs”, but remain important because they are consistent with your symptoms.

### 8.1 Coarse-window misalignment and gradient variance
Because start indices are sampled at minute resolution and coarse outputs are chunked, the model can see inconsistent coarse context per bout. This is a gradient variance amplifier.

Mitigation:
- sample `start_index` aligned to `chunk_period`, or
- explicitly incorporate within-chunk offset in the fine-weight construction.

### 8.2 Hard constraints + STE mismatch
You rely on hard constraints to keep weights feasible:
- min/max
- max change

This is essential in a live AMM, but it creates a non-smooth constrained optimization problem.

If you use STEs, the optimization is no longer optimizing the true constrained function; it is optimizing a surrogate.

Mitigation:
- smooth parameterization (e.g., softmax weights + bounded deltas)
- reduced reliance on STE
- smaller LR + stronger clipping

### 8.3 Arbitrage gating discontinuity
Hard gating “arb or not” makes the simulator piecewise and can create cliffs.

Mitigation:
- soft gating during training
- or continuous trade-cost models where the arb decision is never a strict branch

### 8.4 Objective non-stationarity across windows
Even daily-log-sharpe can have different statistical regimes depending on which window is sampled:
- trending markets vs sideways,
- low-vol vs high-vol.

This can produce:
- non-stationary gradient estimates,
- optimizer chasing moving targets,
- sensitivity to batch sampling strategy.

Mitigations:
- larger batch size,
- stratified sampling,
- curriculum (train on stable regime first, then diversify),
- reduce objective variance by smoothing.

### 8.5 “NaN reinit” hides the real source
Reinitializing parameters after NaNs prevents total collapse but introduces randomness mid-training and can disguise the original root cause.

Mitigation:
- instrument NaN location and stop early
- harden computations to prevent NaNs
- clip grads and objective

---

## 17) Recommended stabilize-first fix set

## 9) Recommended “stabilize-first” fix set (excluding ignored items)

This section is a concrete action list targeted at stabilizing ML training **without changing the entire simulator**.

### Fix A: Log-domain geometric product (confirmed relevant)
Apply log-domain computation for all instances of `jnp.prod(ratio ** weights)`.

This is a strict numerical improvement and reduces both forward and gradient brittleness.

### Fix B: Correct Optax LR scaling (confirmed relevant)
Remove double LR scaling by choosing exactly one place where LR is applied:
- either inside the optimizer (Adam with schedule),
- or via schedule scaling (but not both).

### Fix C: Remove inert Python-level plateau LR (confirmed relevant)
Because optax update ignores `learning_rate`, remove `local_learning_rate` logic or implement injected hyperparams.

Prefer:
- optax plateau reduction within the optimizer chain,
- or injected LR.

### Fix D: Enable gradient clipping (high value)
Use global norm clipping early in the optimizer chain.

Given the objective and non-smooth forward pass, clipping is typically necessary.


### Fix E: Smooth the arbitrage gating during training (high value)
Replace hard `jnp.where(arb_profitable, ...)` with a soft gate (sigmoid) during training.

This reduces cliff edges in parameter space.

### Fix E2: Smooth the “best signature” selection inside optimal arb (very high value)
If your optimal arb implementation enumerates candidate signatures and selects the maximum-profit trade via a hard max/argmax rule,
then you have an additional discontinuity that can dominate training instability.

Training-focused options:
- replace hard argmax with a softmax mixture (smooth expected trade)
- or use straight-through argmax (hard forward, soft backward)

This change can be training-only; evaluation can revert to hard selection.

Replace hard `jnp.where(arb_profitable, ...)` with a soft gate (sigmoid) during training.

This reduces cliff edges in parameter space.

### Fix F: Hardening daily-log-sharpe edge cases (high value)
Ensure daily returns are non-empty and stabilize the denominator.

Even if bouts are long, guardrails reduce unexpected NaNs.

### Fix G: Make sampling chunk-consistent (high value)
Either:
- align start indices to chunk boundaries, or
- explicitly incorporate within-chunk offset.

This reduces gradient variance and is often a major improvement in practice.

---

## 18) Debugging plan

## 10) Debugging plan: how to conclusively isolate the main cause

When learning is “sporadic”, you need instrumentation to tell whether instability comes from:
- objective numerical issues,
- weight constraint regime switching,
- arbitrage gating,
- optax update scaling.

### 10.1 Turn on NaN/Inf debugging
During debugging runs:

```python
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
```

This will localize the first NaN site in the compiled graph.

### 10.2 Track gradient norms by parameter group
Log:
- global norm
- per-leaf max abs
- fraction of leaves that are NaN/Inf

If you see rare spikes, the cause is usually discontinuities (arb gating or constraints) or ratio objective issues.

### 10.3 Compare “smooth training mode” vs “true simulator mode”
A research-standard approach is:
- Train with softened gates and smoother constraints
- Evaluate with true gates and strict constraints

If training stabilizes in smooth mode, you have confirmed the “piecewise simulator surface” is the core problem.

### 10.4 Finite difference sanity checks on a single bout
Pick a single start window, a single param scalar, and compare:
- analytic grad vs finite-difference delta objective

This will tell you quickly whether gradients reflect the true forward mapping or a surrogate (e.g., due to stop_gradient/STE mismatch).

---

## 19) Opinionated diagnosis: why “sometimes a good parameter set is found”

## 11) Opinionated diagnosis: why “sometimes a good parameter set is found”

Given the attached code, the strongest model for your observed behavior is:

1. The simulator is **piecewise** due to:
   - arbitrage gating decisions,
   - clipping/constraints,
   - argmax-based corrections,
   - optional STE behavior.

2. The objective is a ratio-like statistic (daily log sharpe), which can be:
   - high-variance across windows,
   - sensitive to return volatility and window selection.

3. The optimizer configuration currently contains:
   - confusing/incorrect LR scaling (double LR),
   - an inert Python LR decay mechanism that does nothing,
   - optional clipping that may be off.

As a result:
- some runs / some parameter sets land in a stable basin where gates are consistent,
- some get pushed across discontinuity boundaries and “degrade”,
- some explode into NaNs and then recover by reinit.

This pattern is exactly what you would expect when optimizing a discontinuous, constrained, stochastic objective without strong step-size control and clipping.

---

## 20) Suggested “known stable” baseline configuration

## 12) Suggested “known stable” baseline configuration (for verification)

If the goal is: “prove training can be stable and monotonic”, even if slower, use:

- Objective: daily_log_sharpe with hardened denominator and empty-return guard
- Sampling: start indexes aligned to chunk boundaries
- Arb: soft gating during training
- LR: small and well-defined (single schedule application)
- Optax: Adam + global norm clipping
- Disable STE behaviors initially
- Train with fewer constraints first, then introduce them gradually (“constraint curriculum”)

This is a standard recipe for stabilizing highly non-linear differentiable simulators.

---

## Appendix A — Reference: key code hotspots

### A1) Geometric product in reserves
`quantamm_reserves.py`:
- `price_product_change_ratio = jnp.prod(price_change_ratio**weights)`  
Recommended log-domain rewrite.


### A2) Hard arb gating
`quantamm_reserves.py`:
- `arb_profitable = profit_to_arb >= cost`
- `reserves = jnp.where(arb_profitable, ...)`  
Recommended soft gating during training.

### A2b) Hard signature selection by max profit (optimal_n_pool_arb.py)
`optimal_n_pool_arb.py`:
- enumerate candidate signatures
- compute candidate profits
- select max-profit trade via equality-to-max mask or argmax rule

This is a major discontinuity source; softmax or straight-through argmax variants are recommended for training stability.


### A3) Coarse alignment
`TFMM_base_pool.py`:
- `start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)`  
Recommended integer div or aligned sampling.

### A4) Optax LR scaling
`backpropagation.py`:
- `optax.chain(adam(base_lr), scale_by_schedule(schedule(base_lr)))`  
Recommended single LR application.

### A5) External LR ignored
`backpropagation.py`:
- `update(params, start_indexes, learning_rate, opt_state)` but `learning_rate` unused  
Recommended injected hyperparams or remove outer LR decay.

---

## Appendix B — Minimal patch examples

### B1) Log-domain geometric product
```python
eps = 1e-12
log_price_product = jnp.sum(weights * jnp.log(price_change_ratio + eps))
price_product_change_ratio = jnp.exp(log_price_product)
```

### B2) Clean optax configuration (single schedule application)
```python
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=min_lr,
    peak_value=base_lr,
    warmup_steps=warmup_steps,
    decay_steps=n_iterations,
    end_value=min_lr,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.adam(learning_rate=lr_schedule),
)
```

### B3) Soft arb gating (training-only)
```python
k = 100.0
gate = jax.nn.sigmoid(k * (profit_to_arb - arb_external_rebalance_cost))
reserves = gate * post_price_reserves + (1.0 - gate) * prev_reserves
```

### B4) Bout start sampling aligned to chunk boundaries
```python
start = random.choice(subkey, range_, shape=sample_shape, replace=False)
start = (start // chunk_period) * chunk_period
```

### B5) Softmax mixture or straight-through argmax for signature selection
If `profits` is shape `(n_candidates,)` and `overall_trades` is shape `(n_candidates, n_assets)`, then:

**Softmax mixture:**
```python
temperature = 0.05  # tune
p = jax.nn.softmax(profits / temperature)
trade = p @ overall_trades
```

**Straight-through argmax:**
```python
temperature = 0.05
idx = jnp.argmax(profits)
hard = jax.nn.one_hot(idx, profits.shape[0])
soft = jax.nn.softmax(profits / temperature)
p = hard + jax.lax.stop_gradient(soft - hard)
trade = p @ overall_trades
```

Forward behavior can remain close to “choose one” while gradients become far more trainable.

```python
start = random.choice(subkey, range_, shape=sample_shape, replace=False)
start = (start // chunk_period) * chunk_period
```

---

## Final note
Nothing in the reviewed code indicates that the simulator concept is fundamentally wrong. The strongest explanation for instability is **optimization over a piecewise, constrained, regime-switching simulator** under an **inconsistent LR configuration** and without mandatory clipping.

Fixing numerical stability in the reserve products, controlling LR scaling, ensuring the LR actually does what you think it does, and smoothing the arb gating during training are the highest-value steps to turn “sporadic success” into stable convergence.

## Appendix A — Reference: key code hotspots

### A1) Geometric product in reserves
`quantamm_reserves.py`:
- `price_product_change_ratio = jnp.prod(price_change_ratio**weights)`  
Recommended log-domain rewrite.


### A2) Hard arb gating
`quantamm_reserves.py`:
- `arb_profitable = profit_to_arb >= cost`
- `reserves = jnp.where(arb_profitable, ...)`  
Recommended soft gating during training.

### A2b) Hard signature selection by max profit (optimal_n_pool_arb.py)
`optimal_n_pool_arb.py`:
- enumerate candidate signatures
- compute candidate profits
- select max-profit trade via equality-to-max mask or argmax rule

This is a major discontinuity source; softmax or straight-through argmax variants are recommended for training stability.


### A3) Coarse alignment
`TFMM_base_pool.py`:
- `start_index_coarse = ((start_index[0] / chunk_period).astype("int64"), 0)`  
Recommended integer div or aligned sampling.

### A4) Optax LR scaling
`backpropagation.py`:
- `optax.chain(adam(base_lr), scale_by_schedule(schedule(base_lr)))`  
Recommended single LR application.

### A5) External LR ignored
`backpropagation.py`:
- `update(params, start_indexes, learning_rate, opt_state)` but `learning_rate` unused  
Recommended injected hyperparams or remove outer LR decay.

---

## Appendix B — Minimal patch examples

### B1) Log-domain geometric product
```python
eps = 1e-12
log_price_product = jnp.sum(weights * jnp.log(price_change_ratio + eps))
price_product_change_ratio = jnp.exp(log_price_product)
```

### B2) Clean optax configuration (single schedule application)
```python
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=min_lr,
    peak_value=base_lr,
    warmup_steps=warmup_steps,
    decay_steps=n_iterations,
    end_value=min_lr,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.adam(learning_rate=lr_schedule),
)
```

### B3) Soft arb gating (training-only)
```python
k = 100.0
gate = jax.nn.sigmoid(k * (profit_to_arb - arb_external_rebalance_cost))
reserves = gate * post_price_reserves + (1.0 - gate) * prev_reserves
```

### B4) Bout start sampling aligned to chunk boundaries
```python
start = random.choice(subkey, range_, shape=sample_shape, replace=False)
start = (start // chunk_period) * chunk_period
```

### B5) Softmax mixture or straight-through argmax for signature selection
If `profits` is shape `(n_candidates,)` and `overall_trades` is shape `(n_candidates, n_assets)`, then:

**Softmax mixture:**
```python
temperature = 0.05  # tune
p = jax.nn.softmax(profits / temperature)
trade = p @ overall_trades
```

**Straight-through argmax:**
```python
temperature = 0.05
idx = jnp.argmax(profits)
hard = jax.nn.one_hot(idx, profits.shape[0])
soft = jax.nn.softmax(profits / temperature)
p = hard + jax.lax.stop_gradient(soft - hard)
trade = p @ overall_trades
```

Forward behavior can remain close to “choose one” while gradients become far more trainable.

```python
start = random.choice(subkey, range_, shape=sample_shape, replace=False)
start = (start // chunk_period) * chunk_period
```

---

## Final note
Nothing in the reviewed code indicates that the simulator concept is fundamentally wrong. The strongest explanation for instability is **optimization over a piecewise, constrained, regime-switching simulator** under an **inconsistent LR configuration** and without mandatory clipping.

Fixing numerical stability in the reserve products, controlling LR scaling, ensuring the LR actually does what you think it does, and smoothing the arb gating during training are the highest-value steps to turn “sporadic success” into stable convergence.