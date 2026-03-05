# reClAMM Thermostat Design: Reducing LVR via Smart Re-centering

## Background

A reClAMM pool has a constant-product invariant L = (Ra+Va)(Rb+Vb), where Ra,Rb
are real reserves and Va,Vb are *virtual* reserves that define the pool's price
range. When the market price drifts, the pool becomes decentered — one real
balance grows while the other shrinks. The **thermostat** is the mechanism that
re-centers the pool by decaying virtual balances, which shifts the price range
to track the market.

Re-centering is necessary (it keeps the pool usable and earns fees), but it
creates **arb loss**: each virtual balance update changes the pool's spot price
relative to the market, and arbitrageurs extract value by trading the pool back
to equilibrium. This arb loss is the dominant cost of operating a reClAMM pool
and is closely related to the LVR (Loss-Versus-Rebalancing) framework.

The question: **can we reduce total arb loss by being smarter about how fast
the thermostat decays virtual balances?**

## Method 1: Geometric Decay (Baseline / On-chain)

The Solidity implementation uses exponential decay:

```
V_new = V * base^duration
```

where `base ≈ 1 - 1/124000` and `duration` is seconds elapsed. This is
front-loaded: the largest virtual balance changes (and therefore the largest
per-step arb losses) happen immediately after the thermostat fires, then decay
exponentially. Early steps are expensive; late steps are nearly free.

## Method 2: Constant Arc-Length Speed

The arb loss per thermostat step is proportional to (ΔZ)²/(4X), where
Z = √P·Va - Vb/√P is a geometry-aware thermostat coordinate and X = Ra+Va.
By Cauchy-Schwarz, for a fixed total displacement, total loss is minimised
when per-step loss is *constant* — i.e., when each step covers equal
arc-length in the (Z, X) metric space.

This requires stepping by ΔZ = 2·speed·√X·dt at each block, where `speed`
is calibrated to match the geometric decay rate at the onset state (the
moment centeredness first crosses the margin threshold). The implementation
solves a quadratic in VB-space to find the virtual balances that achieve
the target Z.

**Result**: Modest improvement over geometric. On AAVE/ETH (narrow range,
25bps fees, 1 year), constant arc-length saved ~$6,400 in LVR vs geometric
($372,927 vs $379,310).

## Method 3: Centeredness-Proportional Speed (the winner)

The key insight: re-centering urgency depends on *how far off-center the pool
is*. A deeply decentered pool accumulates arb losses faster between blocks
(larger price impact per trade), so it should re-center more aggressively.

The implementation scales the thermostat speed by `margin / centeredness`:

```
effective_speed = base_speed * margin / max(centeredness, 1e-10)
```

Properties:
- **At onset** (centeredness = margin): multiplier = 1.0. The calibration
  against geometric is preserved — the first step is identical.
- **Deeper off-center** (centeredness < margin): multiplier > 1. The pool
  re-centers faster, reducing the time spent in high-loss states.
- **No new state**: centeredness is already computed every block from
  (Ra, Rb, Va, Vb). No oracle, no price history, no additional storage.
  Just one extra division in the exponent.
- **Acts as an implicit vol proxy**: in high-vol regimes, the pool gets
  pushed further off-center between blocks → centeredness drops more →
  speed increases → faster re-centering. Low-vol → gentle re-centering.

This applies to **both** thermostat methods:
- Geometric: `decay = base ^ (duration * margin / centeredness)` — one
  extra multiply in the exponent
- Arc-length: `effective_speed = speed * margin / centeredness`

## Experimental Results

### Setup

- Pool: AAVE/ETH, 1-year simulation (Jun 2024 – Jun 2025), $1M initial
- Minute-resolution price data, minute-frequency arb
- Four variants: Geometric, Geo+Scaled, Const Arc-Length, Arc+Scaled

### Config 1: Narrow range (price_ratio=1.5, margin=0.5, 25bps fees)

This is the on-chain-realistic configuration where the thermostat fires
frequently.

```
                          Geometric     Geo+Scaled      Const Arc     Arc+Scaled
  Final value        $    1,144,275 $    1,155,637 $    1,150,658 $    1,155,509
  LVR (HODL-final)   $      379,310 $      367,948 $      372,927 $      368,077
  Return                     14.43%        15.56%         15.07%         15.55%
```

- Centeredness scaling saves ~$11,300 LVR regardless of base method
- Geo+Scaled ($1,155,637) ≈ Arc+Scaled ($1,155,509) — just $128 apart
- **The proportional controller dominates the base thermostat choice**

### Config 2: Wide range (price_ratio=4.0, margin=0.2, 25bps fees)

```
                          Geometric     Geo+Scaled      Const Arc     Arc+Scaled
  Final value        $    1,118,558 $    1,117,759 $    1,117,943 $    1,118,130
  LVR (HODL-final)   $      405,027 $      405,826 $      405,642 $      405,455
```

Negligible difference. With a wide range, the pool rarely decenters enough
for the thermostat to fire, so the scaling multiplier stays near 1.0.

### Config 3: Narrow range, zero fees

```
                          Geometric     Geo+Scaled      Const Arc     Arc+Scaled
  Final value        $      681,787 $      689,814 $      682,052 $      689,974
  LVR (HODL-final)   $      841,798 $      833,771 $      841,533 $      833,611
```

Same convergence pattern: Geo+Scaled ≈ Arc+Scaled. Without fees to dampen
arb, the LVR savings from scaling are ~$8,000.

## Conclusions

1. **Centeredness-proportional scaling is the dominant improvement.** It
   saves 3-4% of total LVR on narrow-range pools. The constant-arc-length
   thermostat adds negligible value on top of it.

2. **For on-chain implementation, Geometric + Scaling is optimal.** It
   achieves the same LVR reduction as the more complex arc-length approach,
   with far simpler math: just one extra multiply in the decay exponent.
   No Z-space coordinate, no quadratic solver.

3. **The benefit is concentrated in narrow-range, high-turnover pools.**
   Wide-range pools (price_ratio ≥ 4) see negligible effect because the
   thermostat fires rarely.

4. **The scaling acts as a free vol proxy.** High-vol → deeper decentering
   → faster re-centering. This is mechanistically correct and requires no
   external data.

## Implementation

The `reclamm_centeredness_scaling` flag in the run fingerprint enables the
proportional controller. It defaults to `False` for backward compatibility.
When enabled with geometric interpolation:

```python
run_fingerprint = {
    "reclamm_interpolation_method": "geometric",
    "reclamm_centeredness_scaling": True,
    ...
}
```

On-chain, the change is minimal: in the virtual balance update function,
replace `duration` with `duration * margin / centeredness` before computing
the decay. Centeredness is already available (computed from Ra, Rb, Va, Vb).
