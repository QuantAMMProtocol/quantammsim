# Flexible Channel Pool — Strategy README (Rubber-Duck Walkthrough)

This document explains the Flexible Channel strategy step-by-step, with key equations (MathJax), and a critical review of the current JAX implementation (shapes, causality, scaling, and economic plausibility). It also proposes a focused test plan to validate correctness and prevent subtle “time travel” and cross-asset contamination failures.

Reference implementation reviewed: `flexible_channel_pool.py`. :contentReference[oaicite:0]{index=0}

---

## 0) Executive summary

The strategy converts **per-asset EWMA gradients** into **weight changes** using:
1) A **mean-reversion cubic channel** inside an exponential “envelope,”  
2) A **trend-following** term outside the envelope with asymmetric exponents,  
3) Portfolio-level gates for “risk-on” (drawdown amplifier) and “risk-off” (profit-to-stable),  
4) A **Kelly-like scaling** that reduces aggressiveness when realized vol is high,  
5) A **zero-sum offset** to keep total weights conserved,  
6) An optional **entropy shrink** to avoid concentration blow-ups.

**Asset mix matters**: BTC (high vol), PAXG (moderate, defensive), USDC (near-zero vol). USDC induces edge cases: volatility floors, division blow-ups in scaling, and envelope width collapse.

---

## 1) Inputs, shapes, and invariants

### Inputs
- `prices`: shape `(T, N)` where `N=3` assets `{BTC, PAXG, USDC}`
- `chunk_period = chunk`: downsampling stride for the strategy’s effective decision timescale
- `params`: model parameters, many shaped `(N,)` or broadcastable to `(T', N)`

### Output
- `raw_ts`: weight changes per chunk, shape `(T' - 1, N)`

### Required invariants
1) **No look-ahead**: any output at time index `t` must depend only on data `≤ t`.
2) **Conservation**: `sum_i Δw_i(t) = 0` (or very close numerically).
3) **Numerical stability**: no NaNs/Infs across reasonable market regimes.
4) **USDC stability**: near-zero vol should not cause infinite leverage or erratic flips.

---

## 2) Step-by-step strategy logic (what it does + why it exists)

### Step 1 — Chunk prices and compute squared returns
Implementation: chunking and return squares derived from log price ratios. :contentReference[oaicite:1]{index=1}

Let chunked prices be:
\[
p_i^{(c)}(t) = p_i(t \cdot \text{chunk})
\]
and chunk log returns:
\[
r_i(t) = \log\left(\frac{p_i^{(c)}(t)}{p_i^{(c)}(t-1)}\right)
\]
Squared returns:
\[
q_i(t) = r_i(t)^2
\]

**Reason (economic/quant):**
- Chunking approximates a lower-frequency decision process; reduces microstructure noise and overtrading sensitivity.

**Soundness assessment:** Good and standard. The main risk is that chunking can alias regimes if the chunk is poorly chosen.

---

### Step 2 — Realized volatility as causal EWMA time series
Implementation computes a volatility **time series** per asset, not a single “final” value. :contentReference[oaicite:2]{index=2}

For each asset \(i\):
\[
\hat{\sigma}_i^2(t) = \lambda_{\sigma,i}\,\hat{\sigma}_i^2(t-1) + (1-\lambda_{\sigma,i})\,q_i(t),
\qquad
\hat{\sigma}_i(t)=\sqrt{\hat{\sigma}_i^2(t)+\varepsilon}
\]

**Reason:**
- Volatility is needed for *risk scaling* and for the envelope width. It must be causal to avoid time travel.

**Soundness assessment:** Conceptually correct and modern enough.  
**However, USDC warning:** if \(\hat{\sigma}_{USDC}(t)\) is extremely small, any downstream division by \(\hat{\sigma}\) can explode, and any envelope width proportional to \(\hat{\sigma}\) can collapse to ~0.

**Implementation concern (important):**
- The code applies only a tiny floor (`1e-9`) in the Kelly scaling division, and **does not** floor the envelope width itself; that creates NaN risk via “0 * inf” patterns in the channel computations (details in Step 6).

---

### Step 3 — Memory length and base sensitivity
The strategy uses memory \(m_i\) (days) derived from a lambda parameter (via `calc_lamb`, `lamb_to_memory_days_clipped`). :contentReference[oaicite:3]{index=3}

Conceptually:
\[
k^{\text{plain}}_i = \frac{2^{\log k_i}}{m_i}
\]

**Reason:**
- Longer memory should imply slower trading (smaller daily adjustment).

**Soundness assessment:** Reasonable.

---

### Step 4 — Kelly-style risk scaling using volatility
Implementation (simplified):
\[
k_i(t) = \frac{k^{\text{plain}}_i \cdot \kappa_i}{\max(\hat{\sigma}_i(t), \sigma_{\min})}
\]
where \(\kappa_i\) is a trainable scaling (“Kelly kappa”). :contentReference[oaicite:4]{index=4}

**Reason:**
- Standard risk targeting: high vol → smaller step sizes.

**Soundness assessment:** Mostly sound.  
**Major issue for BTC/PAXG/USDC:**
- USDC vol can be *orders of magnitude smaller* than BTC vol, so this makes the optimizer try to “move” weights via USDC (cheap risk) too aggressively unless you cap or floor.

**Recommended risk controls (missing / worth adding):**
- Vol floor: \(\hat{\sigma}_i(t) \leftarrow \max(\hat{\sigma}_i(t), \sigma_{\text{floor}})\) with a realistic floor (not `1e-9`).
- Cap: \(k_i(t)\le k_{\max}\).
- Optional: risk budgeting so stablecoin does not dominate weight change degrees of freedom.

---

### Step 5 — Per-asset knobs (width, alpha, exponents, amplitude, pre-exp scaling)
These control the shape of the channel and trend.

#### 5.1 Envelope width
Implementation uses:
\[
\sigma_{\text{env},i}(t) = 2^{w_i}\,\hat{\sigma}_i(t)
\]
where \(w_i\) is derived from `raw_width` via `pow2(squareplus(.))`. :contentReference[oaicite:5]{index=5}

**Reason:**
- Envelope defines when the channel dominates (inside) versus trend dominates (outside).

**Soundness assessment:** Correct structure; scaling by volatility is sensible.

**Parameterization concern (subtle but important):**
- Because `pow2` uses `squareplus`, the scaling factor is effectively constrained to be **≥ 1**. That prevents envelope widths *tighter than* \(\hat{\sigma}\), which can remove an important degree of freedom (tight channels are common in MR designs).

#### 5.2 Alpha
\[
\alpha_i = 2^{a_i}
\]
controls the relationship between envelope width and cubic channel width.

**Soundness assessment:** Fine.

#### 5.3 Asymmetric exponents
Trend convexity:
\[
\gamma_i(t)=
\begin{cases}
\gamma^{up}_i, & g_i(t)>0 \\
\gamma^{down}_i, & g_i(t)\le 0
\end{cases}
\]
and trend uses \(|g|^{\gamma}\). :contentReference[oaicite:6]{index=6}

**Reason:**
- Asymmetry lets you treat uptrends vs downtrends differently (common in crypto, where down moves are sharper).

**Soundness assessment:** Good idea, but can be numerically brittle if you don’t clip bases or exponents.

#### 5.4 Amplitude mapping (potentially suspect)
Implementation does:
- `amp_raw = pow2(params["log_amplitude"])`
- with `pow2(x) = exp(clip(squareplus(x)*ln2, ...))` :contentReference[oaicite:7]{index=7}

This means amplitude cannot meaningfully go below ~1 (because `squareplus(x)` is always positive, and approaches 0 for large negative x, giving \(2^{\text{small positive}}\approx 1\)).

**Why this is suspect:**
- A parameter named `log_amplitude` strongly implies amplitude \(A=2^{\log A}\) which should allow values in \((0,\infty)\), including < 1.
- Forcing a lower bound near 1 removes the ability to create a genuinely “weak” channel. This can bias the rule toward over-responding even when it should be muted.

**Soundness assessment:** Channel amplitude design is fine; the **parameter transform** likely needs rethinking.

---

### Step 6 — Compute gradients (signal input)
`calc_gradients` produces `grad_ts` shape `(T'-1, N)`. :contentReference[oaicite:8]{index=8}

Interpretation: \(g_i(t)\) is an EWMA-ish proxy of recent trend/mean-reversion impulse.

**Soundness assessment:** Depends on what `calc_gradients` does internally; economically plausible if it is causal and stable.

**Primary risk:** verify `calc_gradients` is strictly causal and does not incorporate future data.

---

### Step 7 — Portfolio-level gates Π (profit / drawdown)
The implementation forms a scalar “portfolio gradient proxy” using one of several modes. :contentReference[oaicite:9]{index=9}

Default (“weights” mode):
\[
\Pi(t) = \sum_i w^{prev}_i \, g_i(t)
\]
Then:
\[
\Pi^+(t)=\max(\Pi(t),0),\qquad \Pi^-(t)=\max(-\Pi(t),0)
\]

These feed:
- **Risk-on amplifier** multiplies trend when drawdown is positive:
\[
T_i(t)=T^{bare}_i(t)\,\left(1+\rho^{on}_i\,\Pi^-(t)\right)
\]
- **Risk-off profit-to-stable** adds a term when profit proxy positive:
\[
P_i(t)=\rho^{off}_i\,\Pi^+(t)
\]

**Why this makes sense economically:**
- Portfolio-aware gating is consistent with utility maximization intuition: in drawdowns you may want to “ride winners” (trend) more aggressively; when doing well you siphon toward stability.

**Key implementation note:**
- `risk_on` and `risk_off` are wrapped in `stop_gradient(...)` in this version, meaning they are **effectively fixed (not trainable)** even though present in `params`. :contentReference[oaicite:10]{index=10}

**Soundness assessment:**
- The gating structure is reasonable.
- Using the **mean of prices across assets** (your earlier concern) would be economically nonsensical; this implementation does **not** do that anymore, which is a material improvement.

**Remaining concern:**
- If `w_prev` is static (single vector), Π does not reflect realized weight drift through time. That may be acceptable if this function is applied incrementally with correct `prev_weights`, but if used as an offline backtest over a full series, Π will be inconsistent with the evolving portfolio unless you explicitly roll weights forward.

---

### Step 8 — The flexible channel kernel (mean reversion + trend + gates)
Kernel computes:

#### 8.1 Envelope
\[
E_i(t)=\exp\left(-\frac{g_i(t)^2}{2\sigma_{\text{env},i}(t)^2}\right)
\]

#### 8.2 Cubic channel (mean reversion inside envelope)
Define
\[
s_i(t)=\frac{\pi g_i(t)}{3(\sigma_{\text{env},i}(t)/\alpha_i)}
\]
Channel term:
\[
C_i(t) = -A_i\,E_i(t)\,\frac{s_i(t) - \frac{s_i(t)^3}{6}}{c_{inv}}
\]

#### 8.3 Trend term (outside envelope)
Let:
\[
b_i(t)=\left|\frac{g_i(t)}{2\beta_i}\right|
\]
Bare trend:
\[
T^{bare}_i(t) = (1-E_i(t))\,\mathrm{sign}(g_i(t))\,b_i(t)^{\gamma_i(t)}
\]

Then apply risk-on:
\[
T_i(t)=T^{bare}_i(t)\,(1+\rho^{on}_i\,\Pi^-(t))
\]

#### 8.4 Profit-to-stable
\[
P_i(t)=\rho^{off}_i\,\Pi^+(t)
\]

#### 8.5 Combine and enforce zero-sum via offset
Signal:
\[
S_i(t)=C_i(t)+T_i(t)+P_i(t)
\]
Offset:
\[
\text{offset}(t) = -\frac{\sum_i k_i(t) S_i(t)}{\sum_i k_i(t)}
\]
Weight change:
\[
\Delta w_i(t)=k_i(t)\,(S_i(t)+\text{offset}(t))
\]

**Why this makes sense:**
- Combines MR and trend in a smooth regime-switching way (via envelope).
- Offset is essential to conserve total weight mass.

**Soundness assessment:** Strong conceptually.

**Critical numerical risk (BTC/PAXG/USDC):**
- If \(\sigma_{\text{env},USDC}(t)\) is tiny, \(s\) can be huge and \(E\) can underflow to 0. You then get products like `0 * (huge polynomial)` which can yield NaNs depending on floating-point path. This is where volatility floors and/or clipping \(s\) are not optional.

---

### Step 9 — Entropy shrink (guard-rail)
Code computes:
- Proposed weights \(w = \text{normalize}(\text{clip}(w_{prev} + \Delta w))\)
- Entropy \(H(w)\)
- Shrink factor:
\[
\gamma(t)=\min\left(1,\sqrt{\frac{H(w)}{H_{min}}}\right)
\]
Then outputs \(\gamma \Delta w\). :contentReference[oaicite:11]{index=11}

**Economic intent:**
- Avoid concentration and improve robustness against overfit signals.

**Soundness assessment:** The idea is financially sound.

**Implementation bug (important):**
- In the scan, the *state* `w` is updated using **unshrunk** `dw`, but the *output* is the **shrunk** `dw_shrunk`. :contentReference[oaicite:12]{index=12}  
This makes the returned sequence inconsistent with the internal recursion: the recursion “believes” the portfolio is more concentrated than the output implies, and the shrink does not compound correctly over time.

**Correct pattern:**
- Compute `dw_shrunk`, then update `w_next` using `prev + dw_shrunk`, not `prev + dw`.

---

## 3) Cross-token “pollination” and whether it’s good or bad

There are two kinds of cross-asset coupling:

1) **Offset coupling**: forces zero-sum across assets.  
   - This is necessary and correct.
2) **Portfolio gating Π**: uses a scalar portfolio proxy to modify each asset’s trend/profit response.  
   - This can be economically justified (portfolio-level risk state informs micro decisions).

**For BTC/PAXG/USDC**, portfolio gating is particularly valuable:
- In risk-off / profit states, shifting into USDC is reasonable.
- In drawdowns, amplifying trend can help if trends persist (common in BTC), but may worsen whipsaws.

**Main caution:** USDC should not dominate Π or \(k\) due to scaling artifacts.

---

## 4) What is financially “basic” vs “modern,” and what upgrades matter

### What is solid / modern enough
- Smooth MR↔trend blending via envelope.
- Volatility scaling (risk targeting).
- Portfolio-aware regime gating.
- Entropy / diversification guard-rails.

### What is “basic” or missing for production quant robustness
1) **Transaction costs & turnover penalty**: essential; otherwise strategies overtrade in backtest.
2) **Volatility model**: EWMA is fine, but for crypto you may want:
   - robust estimators (median absolute deviation of returns),
   - GARCH / realized vol with microstructure noise handling,
   - volatility-of-vol guards.
3) **Regime detection**:
   - trending vs mean-reverting regimes can be detected with Hurst-like proxies or trend strength indicators,
   - use regime-conditioned parameters (mixture-of-experts).
4) **Utility-based optimization**:
   - rather than hand-crafted gates, explicitly maximize expected utility with risk constraints:
     \[
     \max_w \; \mathbb{E}[r_p] - \frac{\lambda}{2}\mathrm{Var}(r_p) - c \cdot \text{turnover}(w)
     \]
5) **Stablecoin-specific modeling**:
   - treat stablecoin as cash numeraire; enforce bounds (e.g., min/max USDC weight) or use it as residual.

---

## 5) Test plan (to validate correctness and catch hidden failures)

Below are the minimum tests I would run before trusting training outcomes.

### A. Shape + broadcasting tests (unit)
- Assert shapes at each step:
  - `prices: (T,N)`
  - `chunk_p: (T',N)`
  - `grad_ts: (T'-1,N)`
  - `sigma_ts: (T'-1,N)`
  - `k_ts: (T'-1,N)`
  - `raw_ts: (T'-1,N)`

### B. Conservation tests (property)
For random inputs, verify:
\[
\left|\sum_i \Delta w_i(t)\right| < 10^{-10} \quad \forall t
\]

### C. Causality (“time travel”) tests
- Construct two price series identical up to time \(t_0\), diverging after \(t_0\).
- Verify outputs `raw_ts[:t0]` identical across both series.
- Do this specifically for:
  - sigma estimation,
  - gradients,
  - portfolio gating Π.

### D. USDC stress tests (edge-case)
- Use a price path where USDC is almost constant with tiny noise.
- Verify:
  - no NaNs/Infs,
  - `k_USDC` is capped or at least not orders of magnitude larger than BTC,
  - envelope width does not collapse to 0.

### E. Monotonicity / stability tests for trend term
- Sweep gradients \(g\) from small to large, ensure:
  - trend magnitude increases with \(|g|\) outside envelope,
  - no overflow (clip or log-domain power where needed).

### F. Entropy shrink correctness tests
- Verify state recursion matches output:
  - if you apply returned `Δw`, entropy should not violate the floor systematically.
- Specifically test the bug mentioned: state must update using shrunk `dw`.

### G. Parameter transform sanity tests
- Verify that intended parameter ranges are achievable:
  - can amplitude be < 1?
  - can width scaling be < 1?
If not, confirm this is deliberate.

---

## 6) Financial soundness scoring (qualitative)

| Step | Purpose | Soundness | Primary risks |
|---|---|---:|---|
| Chunking & returns | smooth timescale | High | aliasing regimes |
| Causal vol EWMA | risk targeting | High | USDC vol floor needed |
| Memory→k scaling | stability & turnover | Medium-High | USDC leverage blow-up |
| Envelope + cubic channel | MR behavior | High | width collapse → NaNs |
| Trend outside envelope | momentum behavior | Medium-High | power-law overflow |
| Portfolio gating Π | regime conditioning | Medium-High | static weights vs rolling |
| Offset | conservation | High | none (just test it) |
| Entropy shrink | concentration control | High in concept | **bug in recursion** |

---

## 7) Final conclusions (what to fix before trusting training)

1) **Volatility floors / caps are mandatory** for BTC/PAXG/USDC.
   - Floor \(\hat{\sigma}\) with a realistic threshold; cap \(k\); floor \(\sigma_{\text{env}}\).
2) **Entropy shrink scan should update state using the shrunk `dw`**, otherwise outputs and internal state diverge.
3) **Amplitude/width parameter transforms should be audited**:
   - If you want amplitude (or width) to be able to go below 1, do not use `squareplus` inside `pow2` for those parameters.
4) **Portfolio gating Π should be explicitly defined for backtests**:
   - If this function runs over a full history, either:
     - provide `prev_weights` per time step, or
     - roll weights forward inside the function (stateful), or
     - accept Π as an approximation and document it.

If you address the above, the strategy is structurally coherent and economically interpretable, and training results will be far less likely to be artifacts of time-travel, stablecoin scaling pathologies, or inconsistent recursion.
