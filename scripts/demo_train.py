import jax.numpy as jnp
from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
)
from quantammsim.runners.jax_runners import train_on_historic_data

# Default fingerprint used as base for all pools
DEFAULT_FINGERPRINT = {
    "startDateString": "2023-01-01 00:00:00",
    "endDateString": "2025-06-01 00:00:00",
    "endTestDateString": "2025-09-01 00:00:00",
    "chunk_period": 60,
    "weight_interpolation_period": 60,
    "fees": 0.0,
    "gas_cost": 0.0,
    "use_alt_lamb": False,
    "use_pre_exp_scaling": True,
    "maximum_change": 2.0,
    "bout_offset": 24 * 60 * 120,
    "optimisation_settings": {
        "method": "gradient_descent",
        "optimiser": "adam",
        "use_gradient_clipping": False,
        "warmup_steps": 1000,
        "batch_size": 6,
        "n_parameter_sets": 2,
        "base_lr": 0.05,
        "use_plateau_decay": True,
        "decay_lr_plateau": 30,
    },
}

EXAMPLE_CONFIGS = {
    "flexiblebtceth_usdc_cashleg": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["BTC", "ETH", "USDC"],
            "rule": "flexible_channel",
            "n_iterations": 5000,

    

  #  # Start very defensive
  #  "initial_weights_logits": [-4.5, -4.8, 5.2],
#
  #  # Moderate k, but channel does the work
  #  "initial_k_per_day": [
  #      0.015,  # BTC
  #      0.009,  # ETH
  #      1e-7,   # USDC
  #  ],
#
  #  # Huge channel amplitude
  #  "initial_log_amplitude": [
  #      -5.5, -5.8, -10.0
  #  ],
#
  #  # Wide envelope → strong mean reversion inside channel
  #  "initial_raw_width": [
  #      1.60,  # BTC
  #      2.20,  # ETH
  #      3.80,  # USDC
  #  ],
#
  #  # Very negative alpha → tight polynomial pullback
  #  "initial_raw_alpha": [
  #      -6.5, -6.8, -7.5
  #  ],
#
  #  # Trend heavily suppressed
  #  "initial_pre_exp_scaling": 3.5,
#
  #  # High exponents → trend dies quickly
  #  "initial_raw_exponents_up":   [2.8, 3.2, 3.5],
  #  "initial_raw_exponents_down": [3.2, 3.6, 3.8],
#
  #  # Kelly strongly constraining
  #  "initial_raw_kelly_kappa": [-16.0, -17.0, -20.0],
#
  #  # Entropy aggressively enforced
  #  "initial_raw_entropy_floor": -6.5,
#
  #  # Slow vol adaptation (regime-style)
  #  "initial_logit_lamb_vol": [
  #      3.6,  # BTC
  #      3.9,  # ETH
  #      6.8,  # USDC
  #  ],
#
  #  # Violent profit-to-cash routing
  #  "initial_risk_on":  [0.01, 0.01, 0.01],
  #  "initial_risk_off": [0.12, 0.12, 0.98],
#
  #  # Medium memory but long drawdown memory
  #  "initial_memory_length": 20.0,
  #  "initial_memory_length_delta": 6.0,
  #  "initial_memory_length_drawdown": [180.0, 180.0, 300.0],
#
  #  "initial_raw_exponents": [0.0, 0.0, 0.0],
  #  "pi_mode": "kelly_proxy",        # "weights" | "risk" | "kelly_proxy" | "equal"
  #  "use_entropy_shrink": True,  # True | False
  #  "sigma_floor": 1e-4,        # raise if USDC still dominates; lower if too restrictive
  #  "sigma_cap": None,          # e.g. 0.20 if you want to clip outliers
  #  "k_max": 50000.0,              # lower => more stable, less aggressive reallocations
  #  "dw_max": None,             # e.g. 0.05 to cap per-step |Δw| at ~5% (smoothly)
  #  "pi_scale": 1.0,
  #  "freeze_risk_logits": False,







#"initial_weights_logits": [
#    [0.00, 0.00, -0.70],   # ~ [0.40, 0.40, 0.20]
#    [0.30, 0.10, -0.40],   # ~ [0.43, 0.35, 0.22]
#],
#
#"initial_k_per_day": [
#    [0.030, 0.020, 0.0015],
#    [0.045, 0.030, 0.0010],
#],
#
## Because current code uses pow2(squareplus(.)) on log_amplitude,
## going very negative only moves you toward the floor (~1) — the real scale is mem_days.
## So the main way to reduce amplitude in practice is shortening memory (below),
## and tightening width/alpha.
#"initial_log_amplitude": [
#    [-18.0, -18.0, -25.0],
#    [-16.0, -16.0, -22.0],
#],
#
## Much narrower than your current widths (which are extremely wide after pow2(.)).
## This causes the envelope to fall off sooner, so trend can appear outside the channel.
#"initial_raw_width": [
#    [-0.60, -0.40, -2.00],
#    [-1.00, -0.80, -2.50],
#],
#
## Move alpha away from “~1” so it actually changes channel geometry.
#"initial_raw_alpha": [
#    [-1.80, -1.80, -3.00],
#    [-1.20, -1.20, -2.80],
#],
#
## NOTE: In your code path, init does raw_pre_exp_scaling = log2(initial_pre_exp_scaling)
## then calculate applies pow2(squareplus(raw_pre_exp_scaling)).
## So values < 1 here are reasonable; e.g. 0.45 maps to ~1.49 effective scaling.
#"initial_pre_exp_scaling": 0.45,
#
## Trend exponents: much lower than your current ~3+.
## This materially increases trend sensitivity without going extreme.
#"initial_raw_exponents_up": [
#    [0.20, 0.20, 0.00],    # exp ~ 1.10 / 1.10 / 1.00
#    [-0.40, -0.40, -0.20], # exp ~ 0.82 / 0.82 / 0.91
#],
#"initial_raw_exponents_down": [
#    [0.40, 0.40, 0.10],    # exp ~ 1.22 / 1.22 / 1.05
#    [-0.20, -0.20, 0.00],  # exp ~ 0.91 / 0.91 / 1.00
#],
#
## Your current transform cannot make kappa < 1, so keep it modestly > 1 (not huge).
## This becomes a mild aggressiveness scaler rather than a “strong constraint”.
#"initial_raw_kelly_kappa": [
#    [-3.0, -3.0, -6.0],
#    [-2.0, -2.0, -5.0],
#],
#
## Your current entropy floor of -6.5 is effectively ~0.
## This makes it a moderate stabiliser instead of near-no-op.
#"initial_raw_entropy_floor": -2.3,
#
## Faster vol adaptation (sigmoid(logit) ~ 0.70–0.90 instead of ~0.97–0.999)
#"initial_logit_lamb_vol": [
#    [1.4, 1.2, 3.8],
#    [0.8, 0.6, 3.2],
#],
#
## Less extreme profit-to-cash routing; risk-on no longer ~0.
#"initial_risk_on": [
#    [0.20, 0.20, 0.03],
#    [0.35, 0.35, 0.06],
#],
#"initial_risk_off": [
#    [0.10, 0.10, 0.70],
#    [0.06, 0.06, 0.55],
#],
#
## Shorter memory => more responsive gradients and also reduces the amplitude floor (amplitude ~ mem_days * amp_raw).
#"initial_memory_length": 12.0,
#"initial_memory_length_delta": 4.0,
#"initial_memory_length_drawdown": [90.0, 90.0, 120.0],
#
## Keep (even if unused) for compatibility with your runner config
#"initial_raw_exponents": [0.0, 0.0, 0.0],
#
## Risk controls: make these meaningfully active so training can’t exploit pathological reallocations
#"pi_mode": "kelly_proxy",
#"use_entropy_shrink": True,
#"sigma_floor": 2e-4,
#"sigma_cap": 0.20,
#"k_max": 250.0,
#"dw_max": 0.03,
#"pi_scale": 1.0,
#"freeze_risk_logits": False,


# Trend-forward starting points (n_parameter_sets = 2)
# Set 0: moderate trend-forward
# Set 1: more aggressive trend-forward

#"initial_weights_logits": [
#    [0.35, 0.25, -0.55],   # ~ [0.45, 0.41, 0.14]
#    [0.55, 0.35, -0.85],   # ~ [0.49, 0.40, 0.11]
#],
#
## Slightly higher k than your defensive setup, but still controlled via k_max & dw_max below
#"initial_k_per_day": [
#    [0.055, 0.035, 0.0020],
#    [0.080, 0.050, 0.0015],
#],
#
## Make amplitude as “low as possible” under your current pow2(squareplus(.)) transform.
## The real lever is memory length (below) because amplitude scales ~ mem_days.
#"initial_log_amplitude": [
#    [-22.0, -22.0, -30.0],
#    [-20.0, -20.0, -28.0],
#],
#
## Narrow envelope => envelope decays quickly => (1-envelope) trend term activates sooner.
## USDC gets very narrow (effectively “do nothing” asset) by keeping widths tiny.
#"initial_raw_width": [
#    [-1.30, -1.10, -3.00],
#    [-1.60, -1.40, -3.50],
#],
#
## Push alpha close to its floor (~1) so the channel does not become overly “springy”.
## (Under your transform, very negative raw_alpha => alpha ~ 1.xx.)
#"initial_raw_alpha": [
#    [-8.0, -8.0, -10.0],
#    [-8.0, -8.0, -10.0],
#],
#
## Lower effective pre-exp scaling => stronger trend response.
## Note: your transform makes effective pre_exp_scaling >= ~1, but <1 here still helps.
#"initial_pre_exp_scaling": 0.30,
#
## Trend exponents: keep ~0.6–1.2-ish effective (vs ~3+ previously).
## This is the biggest “trend-forward” switch.
#"initial_raw_exponents_up": [
#    [-1.40, -1.20, -1.80],
#    [-1.70, -1.50, -2.00],
#],
#"initial_raw_exponents_down": [
#    [-1.10, -0.90, -1.60],
#    [-1.40, -1.20, -1.80],
#],
#
## Your kappa transform cannot yield < 1, so treat kappa as “mild aggressiveness scaler”.
## Keep it near 1 for BTC/ETH; close to 1 for USDC as well.
#"initial_raw_kelly_kappa": [
#    [-5.0, -5.0, -8.0],
#    [-4.0, -4.0, -7.0],
#],
#
## Entropy floor: moderate (more active than -6.5, which tends to be near-no-op).
#"initial_raw_entropy_floor": -2.0,
#
## Faster vol adaptation than your current logit ~3.6–6.8 (which is extremely slow).
## BTC/ETH: responsive; USDC: still fairly slow (stable).
#"initial_logit_lamb_vol": [
#    [0.6, 0.4, 3.4],
#    [0.2, 0.0, 3.0],
#],
#
## Reduce profit-to-cash routing materially; keep some cash preference.
## (USDC still higher, but not ~0.98.)
#"initial_risk_off": [
#    [0.04, 0.04, 0.45],
#    [0.02, 0.02, 0.35],
#],
#
## Risk-on amplifier only applies when drawdown proxy > 0.
## Trend-forward variant: allow BTC/ETH to re-risk in drawdowns (optional behavior),
## but keep USDC near-zero.
#"initial_risk_on": [
#    [0.45, 0.45, 0.03],
#    [0.65, 0.65, 0.05],
#],
#
## Shorter signal memory to improve reactivity and reduce the amplitude floor (amp ~ mem_days * amp_raw).
#"initial_memory_length": 9.0,
#"initial_memory_length_delta": 3.0,
#"initial_memory_length_drawdown": [60.0, 60.0, 90.0],
#
## Keep (even if unused) for compatibility
#"initial_raw_exponents": [0.0, 0.0, 0.0],
#
## Risk controls: activate these meaningfully for training stability
#"pi_mode": "kelly_proxy",
#"use_entropy_shrink": True,
#"sigma_floor": 2e-4,
#"sigma_cap": 0.25,
#"k_max": 180.0,
#"dw_max": 0.035,
#"pi_scale": 1.0,
#"freeze_risk_logits": False,







# Variant B: Mean-reversion + controlled cash-sink (n_parameter_sets = 2)
# Set 0: conservative reversion
# Set 1: balanced reversion

#"initial_weights_logits": [
#    [-1.10, -1.10,  2.20],   # start cash-heavy but not extreme
#    [-0.55, -0.55,  1.10],   # more balanced, still USDC-tilted
#],
#
## k: non-zero BTC/ETH, very small USDC (USDC should not “trade itself”)
#"initial_k_per_day": [
#    [0.020, 0.015, 2.0e-5],
#    [0.030, 0.022, 1.5e-5],
#],
#
## Channel amplitude: moderate (and explicitly <1 if your transform is corrected)
#"initial_log_amplitude": [
#    [-1.60, -1.40, -3.50],
#    [-1.20, -1.00, -3.00],
#],
#
## Envelope widths: moderate-to-wide for BTC/ETH, wider still for USDC (effectively inert)
#"initial_raw_width": [
#    [0.35, 0.55, 1.40],
#    [0.15, 0.35, 1.20],
#],
#
## Alpha: keep polynomial pullback “present” but not extreme
#"initial_raw_alpha": [
#    [-2.2, -2.0, -3.0],
#    [-2.6, -2.4, -3.2],
#],
#
## Trend: not suppressed, not dominant (materially different from both prior regimes)
#"initial_pre_exp_scaling": 1.25,
#
## Exponents: moderate (avoid the “trend dies quickly” regime)
#"initial_raw_exponents_up": [
#    [0.20, 0.30, 0.60],
#    [0.05, 0.15, 0.45],
#],
#"initial_raw_exponents_down": [
#    [0.45, 0.55, 0.75],
#    [0.25, 0.35, 0.65],
#],
#
## Kelly kappa: mild constraint (not the ultra-restrictive -16/-20 regime)
#"initial_raw_kelly_kappa": [
#    [-8.0, -8.5, -12.0],
#    [-7.0, -7.5, -11.0],
#],
#
## Entropy: meaningful but not draconian
#"initial_raw_entropy_floor": -3.5,
#
## Vol memory: moderately responsive BTC/ETH, slow-ish USDC
#"initial_logit_lamb_vol": [
#    [1.40, 1.15, 4.20],
#    [1.05, 0.85, 3.90],
#],
#
## Risk knobs:
## - risk_off routes profit to stable leg (USDC) but not “always everything”
## - risk_on provides mild amplification in drawdown (mostly BTC/ETH)
#"initial_risk_on": [
#    [0.12, 0.12, 0.02],
#    [0.18, 0.18, 0.03],
#],
#"initial_risk_off": [
#    [0.10, 0.10, 0.70],
#    [0.08, 0.08, 0.60],
#],
#
## Memory: mid-range (different from 20/6 and different from the short 9/3 trend-forward)
#"initial_memory_length": 14.0,
#"initial_memory_length_delta": 4.0,
#"initial_memory_length_drawdown": [120.0, 120.0, 180.0],
#
## Compatibility
#"initial_raw_exponents": [0.0, 0.0, 0.0],
#
## Risk controls: keep on, but less tight than the trend-forward suggestion
#"pi_mode": "kelly_proxy",
#"use_entropy_shrink": True,
#"sigma_floor": 1.5e-4,
#"sigma_cap": 0.20,
#"k_max": 260.0,
#"dw_max": 0.030,
#"pi_scale": 1.0,
#"freeze_risk_logits": False,







# Variant C: Trend-first + drawdown amplifier + moderate profit-to-cash
# Two parameter sets (n_parameter_sets = 2)

#"initial_weights_logits": [
#    [-0.35, -0.55,  0.70],   # Set 0: slightly USDC-tilted
#    [ 0.15, -0.05, -0.05],   # Set 1: close to balanced
#],
#
## Higher k for BTC/ETH than prior variants; USDC still tiny
#"initial_k_per_day": [
#    [0.055, 0.038, 1.0e-5],
#    [0.070, 0.050, 8.0e-6],
#],
#
## Weak channel (expects corrected transform A = 2^(log_amplitude) allowing A<1)
#"initial_log_amplitude": [
#    [-2.60, -2.30, -4.20],
#    [-2.20, -1.90, -3.80],
#],
#
## Narrower envelope for BTC/ETH => trend engages sooner outside the channel
## USDC wide/inert
#"initial_raw_width": [
#    [0.10, 0.20, 1.10],
#    [0.00, 0.10, 0.95],
#],
#
## Alpha less extreme than your current (-6 to -7); keeps polynomial pullback present but not dominant
#"initial_raw_alpha": [
#    [-1.10, -1.00, -1.80],
#    [-1.40, -1.30, -2.00],
#],
#
## Trend activation earlier (smaller pre_exp_scaling than 1.25/3.5 regimes)
#"initial_pre_exp_scaling": 0.85,
#
## Exponents: momentum-friendly on the upside, harsher on the downside (crash-responsiveness)
#"initial_raw_exponents_up": [
#    [-0.10,  0.00,  0.40],
#    [-0.25, -0.10,  0.25],
#],
#"initial_raw_exponents_down": [
#    [0.55, 0.70, 0.95],
#    [0.40, 0.55, 0.85],
#],
#
## Kelly kappa: moderate constraint (not ultra-tight)
#"initial_raw_kelly_kappa": [
#    [-5.5, -6.0, -9.0],
#    [-5.0, -5.5, -8.5],
#],
#
## Entropy: allow concentration if signal is strong, but keep a floor
#"initial_raw_entropy_floor": -2.6,
#
## Vol memory: more responsive BTC/ETH, slower USDC
## (logits around ~0.2–0.8 => lambdas ~0.55–0.70-ish; USDC higher => slower)
#"initial_logit_lamb_vol": [
#    [0.65, 0.45, 3.50],
#    [0.35, 0.20, 3.20],
#],
#
## Risk knobs:
## - risk_on meaningfully higher: drawdown amplifier is intended to actually matter
## - risk_off moderate: profit harvest to USDC, but not “always dump everything”
#"initial_risk_on": [
#    [0.35, 0.35, 0.04],
#    [0.45, 0.45, 0.05],
#],
#"initial_risk_off": [
#    [0.18, 0.18, 0.55],
#    [0.15, 0.15, 0.50],
#],
#
## Shorter memory => faster reaction; drawdown memory still longer than main
#"initial_memory_length": 8.0,
#"initial_memory_length_delta": 2.0,
#"initial_memory_length_drawdown": [90.0, 90.0, 150.0],
#
## Compatibility
#"initial_raw_exponents": [0.0, 0.0, 0.0],
#
## Risk controls (set to allow the trend variant to express but still bounded)
#"pi_mode": "kelly_proxy",
#"use_entropy_shrink": True,
#"sigma_floor": 1.0e-4,
#"sigma_cap": 0.25,
#"k_max": 180.0,
#"dw_max": 0.040,
#"pi_scale": 1.0,
#"freeze_risk_logits": False,




## Variant D: Dual-regime ensemble (Crash-catcher vs Risk-on trend)
## n_parameter_sets = 2, assets = [BTC, ETH, USDC]
## Assumes your corrected transform: amplitude = 2^(log_amplitude) (so log_amplitude < 0 => amplitude < 1)
#
#"initial_weights_logits": [
#    [-1.25, -1.10,  2.05],   # Set 0: barbell / cash-heavy
#    [ 0.55,  0.35, -0.95],   # Set 1: risk-on (BTC/ETH overweight vs USDC)
#],
#
## k: Set 0 slower + safer, Set 1 faster + more reactive
#"initial_k_per_day": [
#    [0.020, 0.016, 1.0e-6],
#    [0.090, 0.070, 5.0e-6],
#],
#
## Channel amplitude: intentionally weak (so trend + risk controls matter more)
## USDC amplitude lower (more inert)
#"initial_log_amplitude": [
#    [-1.20, -1.10, -3.00],   # Set 0: A≈0.435/0.466/0.125
#    [-0.35, -0.45, -2.60],   # Set 1: A≈0.78/0.73/0.165
#],
#
## Envelope width: Set 0 narrow-ish (trend engages sooner), Set 1 wider (more structured channel region)
#"initial_raw_width": [
#    [-0.80, -0.75,  0.80],
#    [ 0.70,  0.90,  1.30],
#],
#
## Alpha: Set 0 modest curvature, Set 1 stronger curvature (steeper polynomial channel when inside envelope)
#"initial_raw_alpha": [
#    [0.30, 0.25, 0.00],
#    [1.20, 1.35, 0.40],
#],
#
## Earlier trend activation than your current 3.5 (but not “hyper twitchy”)
#"initial_pre_exp_scaling": 1.10,
#
## Trend exponents:
## - Set 0: downside exponent higher => more decisive “get out” on selloffs, upside more muted
## - Set 1: upside exponent lower => better participation; downside still > upside (risk asymmetry)
#"initial_raw_exponents_up": [
#    [0.60, 0.55, 1.20],
#    [-0.20, -0.10, 0.80],
#],
#"initial_raw_exponents_down": [
#    [1.40, 1.50, 1.80],
#    [0.95, 1.05, 1.55],
#],
#
## Kelly kappa: Set 0 closer to neutral; Set 1 somewhat more aggressive
## (Given current pow2(squareplus(.)) behaviour, negatives keep it near ~1; positives push >1.)
#"initial_raw_kelly_kappa": [
#    [-2.50, -2.70, -3.20],
#    [ 0.80,  0.65, -1.50],
#],
#
## Entropy floor: moderate (allow concentration, but not degenerate one-asset)
#"initial_raw_entropy_floor": -3.30,
#
## Vol memory logits:
## - Set 0: slower vol regime (stability)
## - Set 1: faster BTC/ETH vol adaptation (reactivity), USDC still slow
#"initial_logit_lamb_vol": [
#    [2.00, 2.10, 5.00],
#    [0.05, 0.30, 4.00],
#],
#
## Risk knobs:
## - Set 0: high risk_on => strong drawdown amplifier; high USDC risk_off => profit harvested to cash
## - Set 1: moderate risk_on/off => still some risk mgmt, but less “always cash”
#"initial_risk_on": [
#    [0.75, 0.75, 0.05],
#    [0.30, 0.30, 0.04],
#],
#"initial_risk_off": [
#    [0.12, 0.12, 0.94],
#    [0.08, 0.08, 0.70],
#],
#
## Memory: medium-fast core memory; longer drawdown memory
#"initial_memory_length": 12.0,
#"initial_memory_length_delta": 4.0,
#"initial_memory_length_drawdown": [120.0, 120.0, 180.0],
#
## Compatibility
#"initial_raw_exponents": [0.0, 0.0, 0.0],
#
## Risk controls / clamps (tuned to allow the ensemble to express itself safely)
#"pi_mode": "kelly_proxy",
#"use_entropy_shrink": True,
#"sigma_floor": 1.0e-4,
#"sigma_cap": 0.22,
#"k_max": 250.0,
#"dw_max": 0.050,
#"pi_scale": 1.0,
#"freeze_risk_logits": False,


# Variant E: USDC-numeraire relative-value (BTC/ETH RV with USDC funding)
# n_parameter_sets = 2, assets = [BTC, ETH, USDC]
# Assumes corrected transform: amplitude = 2^(log_amplitude) so log_amplitude < 0 => amplitude < 1

#"initial_weights_logits": [
#    [ 0.25,  0.25, -0.50],   # Set 0: roughly balanced BTC/ETH with meaningful USDC
#    [ 0.55,  0.40, -0.95],   # Set 1: slightly more risk-on tilt
#],
#
## k: moderate turnover (not ultra-slow like your current), USDC still near-inert
#"initial_k_per_day": [
#    [0.060, 0.055, 2.0e-6],
#    [0.095, 0.080, 5.0e-6],
#],
#
## Channel amplitude: moderate (not “huge channel”), USDC muted
#"initial_log_amplitude": [
#    [-0.80, -0.85, -3.20],   # Set 0: A≈0.57/0.55/0.11
#    [-0.35, -0.45, -3.00],   # Set 1: A≈0.78/0.73/0.125
#],
#
## Width: wide envelopes so “inside-channel” MR is meaningful and not hyper-twitchy
#"initial_raw_width": [
#    [1.20, 1.35, 1.80],      # Set 0: wider -> smoother MR region
#    [0.80, 1.00, 1.60],      # Set 1: slightly tighter -> more responsive
#],
#
## Alpha: curvature control
## Set 0 stronger curvature (more pullback within channel), Set 1 milder
#"initial_raw_alpha": [
#    [1.10, 1.20, 0.50],
#    [0.55, 0.65, 0.35],
#],
#
## Pre-exp scaling: keep trend “in the background” unless the move is meaningful
#"initial_pre_exp_scaling": 1.60,
#
## Trend exponents:
## Set 0: trend suppressed (higher exponents)
## Set 1: more breakout-aware (lower up exponent, moderate down exponent)
#"initial_raw_exponents_up": [
#    [1.05, 1.10, 1.60],
#    [0.35, 0.45, 1.20],
#],
#"initial_raw_exponents_down": [
#    [1.35, 1.45, 1.80],
#    [0.85, 0.95, 1.55],
#],
#
## Kelly kappa:
## Set 0 slightly constraining; Set 1 closer to neutral/slightly expansive on BTC/ETH
#"initial_raw_kelly_kappa": [
#    [-1.80, -1.90, -2.80],
#    [ 0.35,  0.25, -2.00],
#],
#
## Entropy floor: moderate — allows concentration during RV dislocations, but avoids degeneracy
#"initial_raw_entropy_floor": -2.80,
#
## Vol memory:
## Set 0 moderate regime speed; Set 1 faster adaptation on BTC/ETH (more responsive risk scaling)
#"initial_logit_lamb_vol": [
#    [0.90, 1.05, 4.80],
#    [-0.35, -0.15, 4.50],
#],
#
## Risk knobs:
## This is the “RV” signature: do NOT violently harvest everything to USDC,
## but DO have meaningful drawdown sensitivity.
#"initial_risk_on": [
#    [0.35, 0.35, 0.03],      # Set 0: drawdown amplifier is present
#    [0.55, 0.55, 0.04],      # Set 1: stronger drawdown response
#],
#"initial_risk_off": [
#    [0.06, 0.06, 0.55],      # Set 0: mild profit-to-USDC routing (not extreme)
#    [0.04, 0.04, 0.40],      # Set 1: even less “always cash”
#],
#
## Memory: shorter core memory (more RV), longer drawdown memory
#"initial_memory_length": 8.0,
#"initial_memory_length_delta": 3.0,
#"initial_memory_length_drawdown": [90.0, 90.0, 140.0],
#
## Compatibility
#"initial_raw_exponents": [0.0, 0.0, 0.0],
#
## Risk controls / clamps: tuned for “spread-ish” behaviour (avoid violent reallocations)
#"pi_mode": "risk",
#"use_entropy_shrink": True,
#"sigma_floor": 2.0e-4,
#"sigma_cap": 0.25,
#"k_max": 180.0,
#"dw_max": 0.035,
#"pi_scale": 0.85,
#"freeze_risk_logits": False,



"initial_weights_logits": [0.0, 0.0, 0.0],
    "initial_log_amplitude": [-10.412931541560809, -0.6277597628953144, -9.281580311670087],
    "initial_raw_width": [9.338773313364289, -1.1640024611713038, 7.1504453349797785],
    "initial_raw_alpha": [-10.134683387113698, -3.148046211000411, -10.034245559953128],
    "initial_raw_exponents_up": [-7.8081904578763925, -0.3911060553118604, 0.7227789671889656],
    "initial_raw_exponents_down": [0.005788727916650551, 0.343368569143285, -0.10645711023304505],
    "initial_raw_kelly_kappa": [-13.02054841035177, -12.088440940098666, -5.5369866264649525],
    "initial_logit_lamb_vol": [7.736376864436009, 1.5521250088338308, -3.1075196931846207],

    # translated from log_k -> k_per_day
    "initial_k_per_day": [
        0.0011790910,  # BTC  = 2**(-9.7282223)
        0.0022032692,  # ETH  = 2**(-8.8264410)
        0.0024319498,  # USDC = 2**(-8.6841286)
    ],

    # translated from logits -> probabilities
    "initial_risk_off": [
        0.1028862006,  # sigmoid(-2.1659178)
        0.0926079123,  # sigmoid(-2.2826342)
        0.6438341342,  # sigmoid(+0.5919989)
    ],
    "initial_risk_on": [
        0.1270719891,  # sigmoid(-1.9272415)
        0.1294285819,  # sigmoid(-1.9060881)
        0.0186260481,  # sigmoid(-3.9660356)
    ],

    # translated from logit_lamb -> memory days (requires Option B patch above if vectors)
    "initial_memory_length": [
        0.9741,  # BTC
        5.5558,  # ETH
        0.0686,  # USDC
    ],
    "initial_memory_length_delta": [0.0, 0.0, 0.0],
    "initial_memory_length_drawdown": [
        0.6950,  # BTC
        0.6816,  # ETH
        0.6890,  # USDC
    ],

    # your init currently expects scalar entropy floor; use mean of the 3 raw values
    "initial_raw_entropy_floor": -14.3188920970,

    # your init currently expects scalar pre_exp_scaling (positive); use a “compromise” value:
    # raw_pre_exp_scaling mean = -1.3553 -> pre_exp_scaling = 2**(-1.3553) ≈ 0.3916
    "initial_pre_exp_scaling": 0.3916,

    # present in your configs (even if unused in this class)
    "initial_raw_exponents": [0.0, 0.0, 0.0],

        },
    },
}

if __name__ == "__main__":
    for name, config in EXAMPLE_CONFIGS.items():
        print(f"\nTraining {name}...")
        train_on_historic_data(
            run_fingerprint=config["fingerprint"],
            verbose=True,
        )
