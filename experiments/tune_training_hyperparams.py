#!/usr/bin/env python3
"""
Hyperparameter Tuning for Robust Training
==========================================

This script finds optimal training hyperparameters using Optuna with
percentile-based pruning and walk-forward analysis.

Key Features:
-------------
1. Percentile pruning: Filter obvious disasters without over-predicting
2. Risk-aware objectives: Optimize for daily_log_sharpe, Calmar, Sterling, Ulcer, or Sharpe
3. Stability analysis: Identifies unreliable hyperparameters
4. Final holdout: never touched during tuning
5. Includes all robustness features: turnover penalty, price noise,
   stratified sampling, Sobol/LHS init, log-space reserves

Why Percentile Pruning (not Hyperband)?
---------------------------------------
WFA cycles are NOT true multi-fidelity: cycle 1 doesn't predict cycles 2-4
because they're different market regimes, not cheap proxies. Hyperband/ASHA
assume correlation between fidelity levels we don't have.

PercentilePruner (25%) is better for our case:
- Just filters obvious disasters (bottom 25%)
- Doesn't make false predictions about future cycles
- Simpler, more appropriate for independent regime evaluation

Search Space:
-------------
Focused ~7D search space (domain-knowledge defaults fix the rest):
- Tuned: base_lr, n_iterations, bout_offset_days, val_fraction,
         turnover_penalty, maximum_change, (training_objective if meaningful)
- Fixed: lr_schedule=cosine, clip_norm=10, noise_scale=0.3, weight_decay=0.01,
         sample_method=uniform, parameter_init_method=gaussian, price_noise=0
- Initial strategy params: conservative but learnable defaults (not tuned)

Usage:
------
# Standard run (150 trials, daily_log_sharpe objective)
python experiments/tune_training_hyperparams.py

# Fewer trials for faster results
python experiments/tune_training_hyperparams.py -n 50

# Optimize for Calmar ratio instead
python experiments/tune_training_hyperparams.py --objective mean_oos_calmar

# Conservative search ranges
python experiments/tune_training_hyperparams.py --conservative

# Quick test (5 trials, 2 cycles)
python experiments/tune_training_hyperparams.py --quick

# No pruning (run all cycles for all trials)
python experiments/tune_training_hyperparams.py --pruner none
"""

import sys
import os
import json
import argparse
import numpy as np
from scipy import stats as scipy_stats
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantammsim.runners.hyperparam_tuner import (
    HyperparamTuner,
    HyperparamSpace,
    TuningResult,
    OUTER_TO_INNER_METRIC,
)
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.utils.post_train_analysis import deflated_sharpe_ratio


# =============================================================================
# Configuration
# =============================================================================

TOKENS = ["ETH", "USDC"]

START_DATE = "2021-01-01 00:00:00"
WFA_END_DATE = "2025-01-01 00:00:00"        # End of walk-forward analysis
HOLDOUT_END_DATE = "2026-01-01 00:00:00"    # End of true holdout

RULE = "mean_reversion_channel"
INITIAL_POOL_VALUE = 1_000_000.0
FEES = 0.0
ARB_FEES = 0.0

STUDY_DIR = Path(__file__).parent / "hyperparam_studies"
STUDY_NAME = "eth_usdc_tuning_v3"


# =============================================================================
# Search Space
# =============================================================================

def create_search_space(
    cycle_days: int = 180,
    conservative: bool = False,
    objective_metric: str = "mean_oos_daily_log_sharpe",
) -> HyperparamSpace:
    """
    Create focused ~7D hyperparameter search space.

    Domain-knowledge defaults (LR schedule, weight decay, clip norm, noise,
    sampling, init method, price noise) are fixed on the base fingerprint —
    see create_base_fingerprint(). Initial strategy params use conservative
    defaults instead of being tuned (the inner optimizer handles them).

    Remaining tunable dimensions (~7):
      base_lr, n_iterations, bout_offset_days, val_fraction,
      turnover_penalty, maximum_change, (training_objective if meaningful)

    Parameters
    ----------
    cycle_days : int
        Approximate WFA cycle length in days
    conservative : bool
        If True, use tighter ranges for stability
    objective_metric : str
        Outer Optuna objective (e.g., "mean_oos_daily_log_sharpe", "mean_oos_calmar").
        Passed to library to conditionally include training_objective choice.
    """
    space = HyperparamSpace.create(cycle_days=cycle_days, objective_metric=objective_metric)

    if conservative:
        space.params["base_lr"] = {"low": 1e-5, "high": 1e-2, "log": True}
        space.params["n_iterations"] = {"low": 100, "high": 2000, "log": True, "type": "int"}

    return space


def create_base_fingerprint() -> dict:
    """Create the base run fingerprint with domain-knowledge fixed values.

    Fixed training params (from HyperparamSpace.FIXED_TRAINING_DEFAULTS):
      lr_schedule_type=cosine, clip_norm=10.0, noise_scale=0.3,
      price_noise_sigma=0.0, sample_method=uniform, parameter_init_method=gaussian,
      weight_decay=0.01 (adamw), early_stopping=True.

    Conservative initial strategy params (from HyperparamSpace.CONSERVATIVE_INITIAL_PARAMS):
      k_per_day=0.5, memory_length=30, log_amplitude=-1, raw_width=1.0,
      raw_exponents=1.0, pre_exp_scaling=0.01.
    """
    fp = deepcopy(run_fingerprint_defaults)

    fp["tokens"] = TOKENS
    fp["rule"] = RULE
    fp["startDateString"] = START_DATE
    fp["endDateString"] = WFA_END_DATE  # Per-cycle adapter overwrites this; set to WFA end as safe default
    fp["endTestDateString"] = WFA_END_DATE
    fp["holdoutEndDateString"] = HOLDOUT_END_DATE

    fp["freq"] = "minute"
    fp["chunk_period"] = 1440
    fp["weight_interpolation_period"] = 1440

    fp["initial_pool_value"] = INITIAL_POOL_VALUE
    fp["fees"] = FEES
    fp["arb_fees"] = ARB_FEES
    fp["gas_cost"] = 0.0

    fp["do_arb"] = True
    fp["arb_frequency"] = 1
    fp["arb_quality"] = 1.0

    fp["minimum_weight"] = 0.01
    fp["max_memory_days"] = 365

    # --- Fixed training params (domain knowledge, not worth searching) ---
    fixed = HyperparamSpace.FIXED_TRAINING_DEFAULTS
    fp["optimisation_settings"]["optimiser"] = "adamw"
    fp["optimisation_settings"]["method"] = "gradient_descent"
    fp["optimisation_settings"]["use_gradient_clipping"] = True
    fp["optimisation_settings"]["n_parameter_sets"] = 8
    fp["optimisation_settings"]["lr_schedule_type"] = fixed["lr_schedule_type"]
    fp["optimisation_settings"]["clip_norm"] = fixed["clip_norm"]
    fp["optimisation_settings"]["noise_scale"] = fixed["noise_scale"]
    # Wider exploration for under-studied params (raw_width, pre_exp_scaling)
    fp["optimisation_settings"]["per_param_noise_scale"] = {
        "raw_width": 0.5,
        "pre_exp_scaling": 0.5,
    }
    fp["optimisation_settings"]["sample_method"] = fixed["sample_method"]
    fp["optimisation_settings"]["parameter_init_method"] = fixed["parameter_init_method"]
    fp["optimisation_settings"]["weight_decay"] = fixed["weight_decay"]
    fp["optimisation_settings"]["early_stopping"] = fixed["early_stopping"]
    fp["price_noise_sigma"] = fixed["price_noise_sigma"]

    # --- Conservative initial strategy params ---
    # Nonzero enough for gradient signal. The inner optimizer discovers the right
    # values from here; we don't waste outer Optuna budget searching init space.
    init = HyperparamSpace.CONSERVATIVE_INITIAL_PARAMS
    fp["initial_k_per_day"] = init["initial_k_per_day"]
    fp["initial_memory_length"] = init["initial_memory_length"]
    fp["initial_log_amplitude"] = init["initial_log_amplitude"]
    fp["initial_raw_width"] = init["initial_raw_width"]
    fp["initial_raw_exponents"] = init["initial_raw_exponents"]
    fp["initial_pre_exp_scaling"] = init["initial_pre_exp_scaling"]

    # Training objective default: align with outer objective.
    # Overridden by search space if training_objective is in the space.
    fp["return_val"] = OUTER_TO_INNER_METRIC.get("mean_oos_daily_log_sharpe", "daily_log_sharpe")
    fp["optimisation_settings"]["early_stopping_metric"] = fp["return_val"]

    return fp


# =============================================================================
# Stability Analysis
# =============================================================================

def analyze_stability(result: TuningResult, study: Optional[Any] = None) -> Dict[str, Any]:
    """Analyze hyperparameter importance and stability using fANOVA.

    Uses Optuna's fANOVA importance evaluator when a study is available,
    falling back to coefficient-of-variation analysis otherwise.

    Parameters with high importance + high variability across top trials
    are dangerous (sensitive and unstable). Low importance = candidates
    for fixing in future runs.
    """
    if not result.all_trials:
        return {"error": "No trials to analyze"}

    completed = [t for t in result.all_trials if t["state"] == "TrialState.COMPLETE"]
    if len(completed) < 3:
        return {"error": "Need at least 3 completed trials"}

    # fANOVA importance (if study available and enough trials)
    fanova_importances = {}
    if study is not None and len(completed) >= 10:
        try:
            from optuna.importance import get_param_importances, FanovaImportanceEvaluator
            fanova_importances = get_param_importances(
                study, evaluator=FanovaImportanceEvaluator()
            )
        except Exception as e:
            print(f"  fANOVA failed (falling back to CV analysis): {e}")

    # CV-based stability analysis on top trials
    completed.sort(key=lambda t: t["value"] if t["value"] else float("-inf"), reverse=True)
    top_trials = completed[:min(10, len(completed))]

    param_distributions = {}
    for param in result.best_params.keys():
        values = [t["params"].get(param) for t in top_trials if param in t["params"]]
        if values and all(isinstance(v, (int, float)) for v in values):
            mean_val = np.mean(values)
            entry = {
                "mean": float(mean_val),
                "std": float(np.std(values)),
                "cv": float(np.std(values) / mean_val) if mean_val != 0 else float("inf"),
            }
            if param in fanova_importances:
                entry["fanova_importance"] = float(fanova_importances[param])
            param_distributions[param] = entry

    unstable = [p for p, s in param_distributions.items() if s.get("cv", 0) > 0.5]
    # High importance + high variability = dangerous
    dangerous = [
        p for p, s in param_distributions.items()
        if s.get("fanova_importance", 0) > 0.1 and s.get("cv", 0) > 0.3
    ]
    # Low importance = candidates for fixing
    fixable = [
        p for p, s in param_distributions.items()
        if s.get("fanova_importance", 0) < 0.05 and p in fanova_importances
    ]

    return {
        "param_distributions": param_distributions,
        "fanova_importances": {k: float(v) for k, v in fanova_importances.items()} if fanova_importances else None,
        "n_analyzed": len(top_trials),
        "unstable_params": unstable,
        "dangerous_params": dangerous,
        "fixable_params": fixable,
        "recommendation": (
            "All parameters stable." if not unstable
            else f"Unstable (high CV): {', '.join(unstable)}"
        ) + (
            f"\nDangerous (important + unstable): {', '.join(dangerous)}" if dangerous
            else ""
        ) + (
            f"\nCandidates for fixing (low importance): {', '.join(fixable)}" if fixable
            else ""
        ),
    }


# =============================================================================
# Regime Tagging
# =============================================================================


def analyze_regimes(result: TuningResult) -> Optional[Dict[str, Dict[str, float]]]:
    """Tag each WFA cycle with regime labels and report per-regime metrics.

    Examines the best trial's per-cycle data. Tags each test period with:
    - Volatility bucket (low/medium/high, based on annualised vol)
    - Trend direction (bull/bear/sideways, based on total return)

    Returns per-regime aggregated OOS metrics.
    """
    # Find best trial with cycle data
    best_eval = None
    for trial in result.all_trials:
        if trial.get("value") == result.best_value and trial.get("evaluation_result"):
            best_eval = trial["evaluation_result"]
            break

    if not best_eval or "cycles" not in best_eval:
        return None

    cycles = best_eval["cycles"]
    if not cycles:
        return None

    # Group cycles by regime
    regime_groups: Dict[str, List[dict]] = {}
    for cycle in cycles:
        oos_sharpe = cycle.get("oos_sharpe", 0)
        oos_roh = cycle.get("oos_returns_over_hodl", 0)
        wfe = cycle.get("walk_forward_efficiency", cycle.get("wfe", 0))

        # Use regime tags computed from actual price data if available
        vol_regime = cycle.get("volatility_regime")
        trend = cycle.get("trend_regime")

        if not vol_regime or vol_regime == "unknown":
            import warnings
            warnings.warn(
                f"Cycle {cycle.get('cycle_number', '?')}: no volatility_regime tag from evaluator. "
                f"This means _compute_regime_tags() didn't run — check TrainingEvaluator. "
                f"Skipping this cycle from regime analysis.",
                stacklevel=2,
            )
            continue

        if not trend or trend == "unknown":
            import warnings
            warnings.warn(
                f"Cycle {cycle.get('cycle_number', '?')}: no trend_regime tag from evaluator. "
                f"Skipping this cycle from regime analysis.",
                stacklevel=2,
            )
            continue

        regime = f"{vol_regime}/{trend}"

        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append({
            "oos_sharpe": oos_sharpe,
            "wfe": wfe,
            "oos_returns_over_hodl": oos_roh,
        })

    # Aggregate per regime
    regime_analysis = {}
    for regime, group in regime_groups.items():
        regime_analysis[regime] = {
            "n_cycles": len(group),
            "mean_oos_sharpe": float(np.mean([g["oos_sharpe"] for g in group])),
            "mean_wfe": float(np.mean([g["wfe"] for g in group])),
            "mean_oos_roh": float(np.mean([g["oos_returns_over_hodl"] for g in group])),
        }

    return regime_analysis


# =============================================================================
# Main Tuning
# =============================================================================

def run_tuning(
    n_trials: int = 150,
    n_wfa_cycles: int = 8,
    resume: bool = False,
    quick: bool = False,
    conservative: bool = False,
    pruner: str = "percentile",
    objective: str = "mean_oos_daily_log_sharpe",
    total_timeout: float = None,
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning.

    Parameters
    ----------
    pruner : str
        Pruning strategy: "percentile" (recommended), "median", "hyperband", "successive_halving", "none"
        Percentile (25%) filters obvious disasters without over-predicting future cycles.
        Hyperband/ASHA assume multi-fidelity correlation we don't have with WFA.
    """
    if quick:
        n_trials = 5
        n_wfa_cycles = 2
        print("\n*** QUICK MODE ***\n")

    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    training_days = 365 * 4  # START_DATE to WFA_END_DATE = 4 years
    cycle_days = int(training_days / n_wfa_cycles)

    base_fp = create_base_fingerprint()
    search_space = create_search_space(
        cycle_days=cycle_days,
        conservative=conservative,
        objective_metric=objective,
    )

    storage_path = STUDY_DIR / f"{STUDY_NAME}.db"
    storage = f"sqlite:///{storage_path}"

    print("=" * 70)
    print("HYPERPARAMETER TUNING v3")
    print("=" * 70)
    print(f"Basket: {TOKENS}")
    print(f"Strategy: {RULE}")
    print(f"WFA period: {START_DATE} to {WFA_END_DATE}")
    print(f"FINAL HOLDOUT (untouched): {WFA_END_DATE} to {HOLDOUT_END_DATE}")
    print(f"Objective: {objective}")
    print(f"Pruner: {pruner}")
    print(f"Conservative: {conservative}")
    print(f"Search space: {len(search_space.params)} dimensions")
    for name, spec in sorted(search_space.params.items()):
        if "choices" in spec:
            print(f"  {name}: {spec['choices']}")
        elif spec.get("type") == "int":
            print(f"  {name}: [{spec['low']}, {spec['high']}] (int, log={spec.get('log', False)})")
        else:
            print(f"  {name}: [{spec['low']}, {spec['high']}] (log={spec.get('log', False)})")
    print(f"Trials: {n_trials}")
    print(f"WFA cycles: {n_wfa_cycles} (~{cycle_days} days each)")
    print(f"Fixed training params: {list(HyperparamSpace.FIXED_TRAINING_DEFAULTS.keys())}")
    print(f"Conservative init params: {list(HyperparamSpace.CONSERVATIVE_INITIAL_PARAMS.keys())}")
    print("=" * 70)

    tuner = HyperparamTuner(
        runner_name="train_on_historic_data",
        n_trials=n_trials,
        n_wfa_cycles=n_wfa_cycles,
        objective=objective,
        hyperparam_space=search_space,
        pruner=pruner,
        enable_pruning=(pruner != "none"),
        total_timeout=total_timeout,
        verbose=True,
        study_name=f"{STUDY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        storage=storage,
    )

    result = tuner.tune(base_fp)

    # Load study back from storage for fANOVA analysis
    import optuna
    study = None
    try:
        studies = optuna.study.get_all_study_summaries(storage)
        if studies:
            study = optuna.load_study(study_name=studies[-1].study_name, storage=storage)
    except Exception as e:
        print(f"Could not load study for fANOVA: {e}")

    stability = analyze_stability(result, study=study)

    # --- Stability Analysis ---
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS (fANOVA + CV)")
    print("=" * 70)
    if "error" not in stability:
        if stability.get("fanova_importances"):
            print("\nfANOVA importance:")
            for param, imp in sorted(stability["fanova_importances"].items(), key=lambda x: -x[1]):
                print(f"  {param}: {imp:.3f}")
        print("\nTop-10 trial CV:")
        for param, st in stability.get("param_distributions", {}).items():
            cv = st.get("cv", 0)
            imp = st.get("fanova_importance", None)
            imp_str = f", importance={imp:.3f}" if imp is not None else ""
            label = "STABLE" if cv < 0.3 else "MODERATE" if cv < 0.5 else "UNSTABLE"
            print(f"  {param}: CV={cv:.2f} ({label}{imp_str})")
        print(f"\n{stability['recommendation']}")
    print("=" * 70)

    # --- Extract best trial's evaluation result ---
    best_eval = None
    for trial in result.all_trials:
        if trial.get("value") == result.best_value and trial.get("evaluation_result"):
            best_eval = trial["evaluation_result"]
            break

    # --- Bootstrap CIs for OOS Sharpe ---
    bootstrap_ci = None
    if best_eval:
        bootstrap_ci = best_eval.get("bootstrap_ci")
        if bootstrap_ci and "warning" not in bootstrap_ci:
            print("\n" + "=" * 70)
            print("BOOTSTRAP CI FOR OOS SHARPE (95%)")
            print("=" * 70)
            print(f"  Point estimate:  {bootstrap_ci['point_estimate']:.4f}")
            print(f"  95% CI:          [{bootstrap_ci['lower']:.4f}, {bootstrap_ci['upper']:.4f}]")
            print(f"  Bootstrap std:   {bootstrap_ci['std']:.4f}")
            if bootstrap_ci["lower"] <= 0:
                print("  WARNING: CI includes zero — OOS performance not significantly positive")
            print("=" * 70)
        elif bootstrap_ci and "warning" in bootstrap_ci:
            print(f"\nBootstrap CI skipped: {bootstrap_ci['warning']}")

    # --- Deflated Sharpe Ratio ---
    # Use concatenated OOS daily returns for correct T (not total_days which
    # conflates number of cycles with per-cycle precision)
    dsr_result = None
    if result.n_completed > 0 and best_eval:
        concat_returns = best_eval.get("concatenated_oos_daily_returns")
        observed_sr = best_eval.get("mean_oos_sharpe", result.best_value)

        if concat_returns and len(concat_returns) > 1:
            # T = number of daily return observations in the concatenated OOS series.
            # This correctly reflects the precision of the SR estimator.
            T_oos = len(concat_returns)
            returns_arr = np.array(concat_returns)
            skew_val = float(scipy_stats.skew(returns_arr))
            kurt_val = float(scipy_stats.kurtosis(returns_arr))
            if np.isnan(skew_val) or np.isnan(kurt_val):
                print(f"\nWARNING: skew/kurtosis is NaN (n={len(returns_arr)}) — "
                      "falling back to normal assumption for DSR")
                skew_val, kurt_val = 0.0, 0.0
            dsr_result = deflated_sharpe_ratio(
                observed_sr=observed_sr,
                n_trials=result.n_completed + result.n_pruned,
                T=T_oos,
                skew=skew_val,
                kurt=kurt_val,
            )
        else:
            # Fallback: use per-cycle day count (less precise but still usable)
            print("\nWARNING: No concatenated OOS returns available for DSR — "
                  "using per-cycle day estimate (less precise)")
            dsr_result = deflated_sharpe_ratio(
                observed_sr=observed_sr,
                n_trials=result.n_completed + result.n_pruned,
                T=cycle_days,  # per-cycle, not total
            )

        print("\n" + "=" * 70)
        print("DEFLATED SHARPE RATIO (Bailey & López de Prado)")
        print("=" * 70)
        print(f"  Observed SR:     {dsr_result['observed_sr']:.4f}")
        print(f"  Expected max SR: {dsr_result['sr0']:.4f} (under null, {dsr_result['n_trials']} trials)")
        print(f"  DSR:             {dsr_result['dsr']:.4f}")
        print(f"  T (observations):{dsr_result['T']}")
        print(f"  Significant:     {'YES' if dsr_result['significant'] else 'NO (DSR < 0.95 — result may be noise)'}")
        print("=" * 70)

    # --- Regime Tagging ---
    regime_analysis = analyze_regimes(result)
    if regime_analysis:
        print("\n" + "=" * 70)
        print("REGIME ANALYSIS")
        print("=" * 70)
        for regime, metrics in sorted(regime_analysis.items()):
            print(f"  {regime}: OOS Sharpe={metrics['mean_oos_sharpe']:.4f} "
                  f"(n={metrics['n_cycles']}, WFE={metrics['mean_wfe']:.4f})")
        print("=" * 70)

    # Save results
    save_results(
        result, stability, pruner,
        dsr_result=dsr_result, regime_analysis=regime_analysis,
        bootstrap_ci=bootstrap_ci,
    )

    return {
        "result": result, "stability": stability,
        "dsr": dsr_result, "regimes": regime_analysis,
        "bootstrap_ci": bootstrap_ci,
    }


def save_results(
    result,
    stability: Dict[str, Any],
    pruner: str,
    dsr_result: Optional[Dict] = None,
    regime_analysis: Optional[Dict] = None,
    bootstrap_ci: Optional[Dict] = None,
):
    """Save tuning results to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = STUDY_DIR / f"best_params_{timestamp}.json"

    output = {
        "version": "4.0",
        "timestamp": timestamp,
        "pruner": pruner,
        "basket": TOKENS,
        "rule": RULE,
        "training_period": {"start": START_DATE, "end": WFA_END_DATE},
        "holdout_end": HOLDOUT_END_DATE,
        "best_params": result.best_params,
        "best_value": result.best_value,
        "tuning_summary": {
            "n_completed": result.n_completed,
            "n_pruned": getattr(result, 'n_pruned', 0),
        },
        "fixed_training_defaults": HyperparamSpace.FIXED_TRAINING_DEFAULTS,
        "conservative_initial_params": HyperparamSpace.CONSERVATIVE_INITIAL_PARAMS,
        "stability_analysis": stability,
        "deflated_sharpe_ratio": dsr_result,
        "bootstrap_ci": bootstrap_ci,
        "regime_analysis": regime_analysis,
        "next_steps": [
            "1. Validate on final holdout (2025 - 2026)",
            "2. Review stability - fix unstable params if any",
            "3. Check DSR significance (>= 0.95)",
            "4. Review regime breakdown for regime-specific failures",
            "5. Run on additional asset pairs to check transferability",
            "6. Paper trade before production",
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Print copy-paste config
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    print("optimisation_settings = {")
    for key, value in sorted(result.best_params.items()):
        if isinstance(value, str):
            print(f'    "{key}": "{value}",')
        elif isinstance(value, float):
            print(f'    "{key}": {value:.6g},')
        else:
            print(f'    "{key}": {value},')
    print("}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for crypto baskets",
    )
    parser.add_argument("--n-trials", "-n", type=int, default=150)
    parser.add_argument("--n-wfa-cycles", "-c", type=int, default=8)
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--quick", "-q", action="store_true")
    parser.add_argument("--conservative", action="store_true")
    parser.add_argument("--pruner", "-p", default="percentile",
                        choices=["percentile", "median", "hyperband", "successive_halving", "none"],
                        help="Pruning strategy: percentile (recommended), median, hyperband, successive_halving, none")
    parser.add_argument("--objective", "-o", default="mean_oos_daily_log_sharpe",
                        choices=[
                            "mean_oos_daily_log_sharpe", "worst_oos_daily_log_sharpe",
                            "mean_oos_sharpe", "worst_oos_sharpe",
                            "mean_oos_calmar", "worst_oos_calmar",
                            "mean_oos_sterling", "worst_oos_sterling",
                            "mean_oos_ulcer", "worst_oos_ulcer",
                            "mean_oos_returns_over_hodl", "worst_oos_returns_over_hodl",
                            "mean_wfe", "worst_wfe",
                        ])
    parser.add_argument("--timeout", type=float, default=None, help="Max hours")

    args = parser.parse_args()

    run_tuning(
        n_trials=args.n_trials,
        n_wfa_cycles=args.n_wfa_cycles,
        resume=args.resume,
        quick=args.quick,
        conservative=args.conservative,
        pruner=args.pruner,
        objective=args.objective,
        total_timeout=args.timeout * 3600 if args.timeout else None,
    )


if __name__ == "__main__":
    main()
