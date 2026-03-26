"""Run calibration with MLPNoiseHead and compare against linear baselines.

Steps:
  1. Load panel, match to per-day grids
  2. Option C: per-pool L-BFGS-B fits (baseline, gas fixed to chain)
  3. Linear joint: SharedLinearNoiseHead baseline
  4. MLP noise joint: MLPNoiseHead (new)
  5. Full MLP joint: MLPHead cadence + MLPNoiseHead (new)
  6. Per-pool prediction, R², decomposition for each method
  7. Paginated plots, summary distributions, comparison scatter
  8. JSON export
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Config ----
PANEL_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "local_data", "noise_calibration", "panel.parquet",
)
GRID_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "pool_grids_v2",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "mlp_calibration",
)
OPTION_C_LOSS_CUTOFF = 5.0
OPTION_C_MAXITER = 500
JOINT_MAXITER = 5000
MLP_HIDDEN = 16
TOP_N = 50
# Best alpha settings from sweep (phase 2)
ALPHA_CAD = 0.001
ALPHA_NOISE = 0.1


# ---- Data loading ----


def load_and_match():
    """Load panel, match to grids."""
    from quantammsim.calibration.pool_data import (
        match_grids_to_panel,
        replace_panel_volatility_with_binance,
    )

    panel = pd.read_parquet(PANEL_CACHE)

    if "log_tvl_lag1" not in panel.columns:
        panel = panel.sort_values(["pool_id", "date"]).reset_index(drop=True)
        panel["log_tvl_lag1"] = panel.groupby("pool_id")["log_tvl"].shift(1)
        panel = panel.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)

    pool_counts = panel.groupby("pool_id").size()
    valid = pool_counts[pool_counts >= 10].index
    panel = panel[panel["pool_id"].isin(valid)].copy()

    print("Replacing volatility with Binance minute data...")
    panel = replace_panel_volatility_with_binance(panel)

    print(f"Panel: {len(panel)} obs, {panel['pool_id'].nunique()} pools, "
          f"{panel['date'].min()} to {panel['date'].max()}")

    matched = match_grids_to_panel(GRID_DIR, panel)
    print(f"Matched: {len(matched)} pools with grids")
    return panel, matched


def filter_pathological(matched, option_c):
    """Drop pools with high Option C loss."""
    good = {p: r for p, r in option_c.items() if r["loss"] <= OPTION_C_LOSS_CUTOFF}
    dropped = set(option_c) - set(good)
    matched_clean = {p: matched[p] for p in good if p in matched}
    if dropped:
        print(f"  Dropping {len(dropped)} pools (loss > {OPTION_C_LOSS_CUTOFF}):")
        for p in sorted(dropped):
            print(f"    {p} loss={option_c[p]['loss']:.1f}")
    return matched_clean, good


# ---- Fitting ----


def run_option_c(matched):
    """Per-pool fits with gas fixed to chain costs."""
    from quantammsim.calibration.per_pool_fit import fit_all_pools

    print(f"\n--- Option C: per-pool fits ({len(matched)} pools, gas fixed) ---")
    results = fit_all_pools(matched, fix_gas_to_chain=True)

    losses = [r["loss"] for r in results.values()]
    n_conv = sum(1 for r in results.values() if r["converged"])
    print(f"  Converged: {n_conv}/{len(results)}")
    print(f"  Loss: median={np.median(losses):.4f}, mean={np.mean(losses):.4f}")
    return results


def run_option_c_reduced(matched):
    """Per-pool fits with reduced x_obs (4 covariates) and gas fixed."""
    from quantammsim.calibration.per_pool_fit import fit_all_pools

    print(f"\n--- Option C Reduced: per-pool fits ({len(matched)} pools, "
          f"4-covariate x_obs, gas fixed) ---")
    results = fit_all_pools(matched, fix_gas_to_chain=True, reduced=True)

    losses = [r["loss"] for r in results.values()]
    n_conv = sum(1 for r in results.values() if r["converged"])
    print(f"  Converged: {n_conv}/{len(results)}")
    print(f"  Loss: median={np.median(losses):.4f}, mean={np.mean(losses):.4f}")
    return results


def _build_gas_values(jdata, matched_clean):
    """Build fixed gas values (log-space) from chain data."""
    from quantammsim.calibration.loss import CHAIN_GAS_USD
    gas_values = []
    for pid in jdata.pool_ids:
        chain = matched_clean[pid]["chain"]
        gas_usd = CHAIN_GAS_USD.get(chain, 1.0)
        gas_values.append(np.log(max(gas_usd, 1e-6)))
    return np.array(gas_values)


def run_linear_joint(matched_clean, option_c_clean):
    """Joint fit with LinearHead + SharedLinearNoiseHead (baseline)."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import FixedHead, LinearHead, SharedLinearNoiseHead
    from quantammsim.calibration.joint_fit import prepare_joint_data

    jdata = prepare_joint_data(
        matched_clean, drop_chain_dummies=True, fix_gas_to_chain=True)
    gas_values = _build_gas_values(jdata, matched_clean)

    model = CalibrationModel(
        cadence_head=LinearHead("cad", alpha=ALPHA_CAD),
        gas_head=FixedHead("gas", gas_values),
        noise_head=SharedLinearNoiseHead(alpha=ALPHA_NOISE),
    )

    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    n_p = model.n_params(n_pools, k_attr)
    print(f"\n--- Linear baseline: SharedLinearNoiseHead ({n_pools} pools, {n_p} params) ---")

    result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=option_c_clean)
    print(f"  Loss: {result['init_loss']:.4f} -> {result['loss']:.4f}")
    print(f"  Converged: {result['converged']}")
    return result, model, jdata


def run_mlp_noise_joint(matched_clean, option_c_clean, hidden=MLP_HIDDEN):
    """Joint fit with LinearHead cadence + FixedHead gas + MLPNoiseHead."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import FixedHead, LinearHead, MLPNoiseHead
    from quantammsim.calibration.joint_fit import prepare_joint_data

    jdata = prepare_joint_data(
        matched_clean, drop_chain_dummies=True, fix_gas_to_chain=True)
    gas_values = _build_gas_values(jdata, matched_clean)

    model = CalibrationModel(
        cadence_head=LinearHead("cad", alpha=ALPHA_CAD),
        gas_head=FixedHead("gas", gas_values),
        noise_head=MLPNoiseHead(hidden=hidden, alpha=ALPHA_NOISE),
    )

    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    n_p = model.n_params(n_pools, k_attr)
    print(f"\n--- MLP noise: MLPNoiseHead(hidden={hidden}) ({n_pools} pools, {n_p} params) ---")

    result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=option_c_clean)
    print(f"  Loss: {result['init_loss']:.4f} -> {result['loss']:.4f}")
    print(f"  Converged: {result['converged']}")
    return result, model, jdata


def run_mlp_full_joint(matched_clean, option_c_clean, hidden=MLP_HIDDEN):
    """Joint fit with MLPHead cadence + FixedHead gas + MLPNoiseHead."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import FixedHead, MLPHead, MLPNoiseHead
    from quantammsim.calibration.joint_fit import prepare_joint_data

    jdata = prepare_joint_data(
        matched_clean, drop_chain_dummies=True, fix_gas_to_chain=True)
    gas_values = _build_gas_values(jdata, matched_clean)

    model = CalibrationModel(
        cadence_head=MLPHead("cad", hidden=hidden, alpha=ALPHA_CAD),
        gas_head=FixedHead("gas", gas_values),
        noise_head=MLPNoiseHead(hidden=hidden, alpha=ALPHA_NOISE),
    )

    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    n_p = model.n_params(n_pools, k_attr)
    print(f"\n--- Full MLP: MLPHead(cad) + MLPNoiseHead ({n_pools} pools, {n_p} params) ---")

    result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=option_c_clean)
    print(f"  Loss: {result['init_loss']:.4f} -> {result['loss']:.4f}")
    print(f"  Converged: {result['converged']}")
    return result, model, jdata


def run_two_stage_joint(matched_clean, option_c_clean, hidden=MLP_HIDDEN):
    """Two-stage joint fit to identify cadence separately from noise.

    Stage 1: LinearHead(cad) + FixedHead(gas) + PerPoolNoiseHead
        Per-pool noise (8 coeffs/pool) can't fully absorb arb's daily
        volatility pattern, so cadence is identified.

    Stage 2: FixedHead(cad, stage1_values) + FixedHead(gas) + MLPNoiseHead
        Cadence frozen from stage 1, MLP learns shared noise mapping.

    For new-pool prediction: stage 1 linear coefficients give cadence,
    stage 2 MLP gives noise.
    """
    import jax.numpy as jnp
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import (
        FixedHead, LinearHead, MLPNoiseHead, PerPoolNoiseHead,
    )
    from quantammsim.calibration.joint_fit import prepare_joint_data

    jdata = prepare_joint_data(
        matched_clean, drop_chain_dummies=True, fix_gas_to_chain=True)
    gas_values = _build_gas_values(jdata, matched_clean)
    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]

    # ---- Stage 1: fit cadence with per-pool noise ----
    stage1_model = CalibrationModel(
        cadence_head=LinearHead("cad", alpha=ALPHA_CAD),
        gas_head=FixedHead("gas", gas_values),
        noise_head=PerPoolNoiseHead(),
    )
    n_p1 = stage1_model.n_params(n_pools, k_attr)
    print(f"\n--- Two-stage S1: LinearHead(cad) + PerPoolNoiseHead "
          f"({n_pools} pools, {n_p1} params) ---")

    stage1_result = stage1_model.fit(
        jdata, maxiter=JOINT_MAXITER, warm_start=option_c_clean)
    print(f"  Loss: {stage1_result['init_loss']:.4f} -> {stage1_result['loss']:.4f}")
    print(f"  Converged: {stage1_result['converged']}")

    # Extract per-pool cadences from stage 1
    params1 = jnp.array(stage1_result["params_flat"])
    (cs, ce), _, _ = stage1_model._head_slices(n_pools, k_attr)
    cad_slice = params1[cs:ce]
    stage1_cadences = np.array([
        float(stage1_model.cadence_head.predict(cad_slice, i, jdata.x_attr[i]))
        for i in range(n_pools)
    ])
    print(f"  Cadence range: {np.exp(stage1_cadences.min()):.1f} - "
          f"{np.exp(stage1_cadences.max()):.1f} min")

    # ---- Stage 2: fit MLP noise with frozen cadence ----
    stage2_model = CalibrationModel(
        cadence_head=FixedHead("cad", stage1_cadences),
        gas_head=FixedHead("gas", gas_values),
        noise_head=MLPNoiseHead(hidden=hidden, alpha=ALPHA_NOISE),
    )
    n_p2 = stage2_model.n_params(n_pools, k_attr)
    print(f"\n--- Two-stage S2: FixedHead(cad) + MLPNoiseHead(hidden={hidden}) "
          f"({n_pools} pools, {n_p2} params) ---")

    # Build warm-start for stage 2 noise from stage 1 per-pool noise
    (_, _), (_, _), (ns, ne) = stage1_model._head_slices(n_pools, k_attr)
    noise_params1 = np.array(params1[ns:ne])
    stage2_warm = {}
    for i, pid in enumerate(jdata.pool_ids):
        noise_c = np.array(stage1_model.noise_head.predict(
            jnp.array(noise_params1), i, jdata.x_attr[i]))
        stage2_warm[pid] = {"noise_coeffs": noise_c}

    stage2_result = stage2_model.fit(
        jdata, maxiter=JOINT_MAXITER, warm_start=stage2_warm)
    print(f"  Loss: {stage2_result['init_loss']:.4f} -> {stage2_result['loss']:.4f}")
    print(f"  Converged: {stage2_result['converged']}")

    # Build a composite result dict for downstream use
    # Cadence comes from stage 1 linear head, noise from stage 2 MLP
    result = {
        "stage1_result": stage1_result,
        "stage2_result": stage2_result,
        "loss": stage2_result["loss"],
        "init_loss": stage1_result["init_loss"],
        "converged": stage1_result["converged"] and stage2_result["converged"],
        "n_pools": n_pools,
        "k_attr": k_attr,
        "pool_ids": jdata.pool_ids,
        "attr_names": jdata.attr_names,
    }

    return result, stage1_model, stage2_model, jdata


def run_reduced_joint(matched_clean, option_c_clean, hidden=MLP_HIDDEN):
    """Joint fit with reduced x_obs (4 cols) to avoid noise-cadence confounding.

    Removes sigma- and fee-dependent features from the noise model's x_obs
    so the arb channel (grid + cadence) is the only path for volatility-driven
    volume variation. See docs/noise_covariate_design.md for theory.

    Uses LinearHead(cad) + FixedHead(gas) + MLPNoiseHead(k_obs=4).
    Cadence warm-started from Option C; noise cold-started (OLS on 4-col x_obs).
    """
    import jax.numpy as jnp
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import FixedHead, LinearHead, MLPNoiseHead
    from quantammsim.calibration.joint_fit import prepare_joint_data
    from quantammsim.calibration.pool_data import K_OBS_REDUCED

    jdata = prepare_joint_data(
        matched_clean, drop_chain_dummies=True,
        fix_gas_to_chain=True, reduced_x_obs=True)
    gas_values = _build_gas_values(jdata, matched_clean)
    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]

    model = CalibrationModel(
        cadence_head=LinearHead("cad", alpha=ALPHA_CAD),
        gas_head=FixedHead("gas", gas_values),
        noise_head=MLPNoiseHead(hidden=hidden, alpha=ALPHA_NOISE,
                                k_obs=K_OBS_REDUCED),
    )

    n_p = model.n_params(n_pools, k_attr)
    print(f"\n--- Reduced x_obs: LinearHead(cad) + MLPNoiseHead(k_obs={K_OBS_REDUCED}, "
          f"hidden={hidden}) ({n_pools} pools, {n_p} params) ---")

    # Only warm-start cadence (noise dimension changed 8→4, skip noise warm-start)
    warm_cad = {}
    for pid in jdata.pool_ids:
        if pid in option_c_clean:
            warm_cad[pid] = {"cad": option_c_clean[pid]["log_cadence"]}

    result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=warm_cad)
    print(f"  Loss: {result['init_loss']:.4f} -> {result['loss']:.4f}")
    print(f"  Converged: {result['converged']}")

    return result, model, jdata


def _extract_two_stage_per_pool(stage1_model, stage2_model, result, jdata):
    """Extract per-pool params from two-stage result."""
    import jax.numpy as jnp

    stage1_result = result["stage1_result"]
    stage2_result = result["stage2_result"]
    n_pools = result["n_pools"]
    k_attr = result["k_attr"]

    params1 = jnp.array(stage1_result["params_flat"])
    params2 = jnp.array(stage2_result["params_flat"])

    (cs1, ce1), _, (ns1, ne1) = stage1_model._head_slices(n_pools, k_attr)
    _, _, (ns2, ne2) = stage2_model._head_slices(n_pools, k_attr)

    cad_slice = params1[cs1:ce1]
    noise_slice = params2[ns2:ne2]

    per_pool = []
    for i in range(n_pools):
        x_attr_i = jdata.x_attr[i]
        log_cad = float(stage1_model.cadence_head.predict(cad_slice, i, x_attr_i))
        log_gas = float(stage2_model.gas_head.predict(
            jnp.array([]), i, x_attr_i))  # FixedHead ignores params
        noise_c = np.array(stage2_model.noise_head.predict(noise_slice, i, x_attr_i))
        per_pool.append({
            "log_cadence": log_cad,
            "log_gas": log_gas,
            "noise_coeffs": noise_c,
            "cadence_minutes": float(np.exp(log_cad)),
            "gas_usd": float(np.exp(log_gas)),
        })
    return per_pool


# ---- Per-pool predictions ----


def _extract_per_pool_params(model, result, jdata):
    """Extract per-pool (log_cadence, log_gas, noise_coeffs) from a CalibrationModel result."""
    import jax.numpy as jnp

    params = jnp.array(result["params_flat"])
    n_pools = result["n_pools"]
    k_attr = result["k_attr"]
    (cs, ce), (gs, ge), (ns, ne) = model._head_slices(n_pools, k_attr)

    cad_slice = params[cs:ce]
    gas_slice = params[gs:ge]
    noise_slice = params[ns:ne]

    per_pool = []
    for i in range(n_pools):
        x_attr_i = jdata.x_attr[i]
        log_cad = float(model.cadence_head.predict(cad_slice, i, x_attr_i))
        log_gas = float(model.gas_head.predict(gas_slice, i, x_attr_i))
        noise_c = np.array(model.noise_head.predict(noise_slice, i, x_attr_i))
        per_pool.append({
            "log_cadence": log_cad,
            "log_gas": log_gas,
            "noise_coeffs": noise_c,
            "cadence_minutes": float(np.exp(log_cad)),
            "gas_usd": float(np.exp(log_gas)),
        })
    return per_pool


def compute_per_pool_predictions(matched, option_c_results,
                                  model_results, reduced_models=None):
    """Compute V_arb, V_noise, R² per pool for Option C and each joint model.

    model_results: list of (label, per_pool_params, pool_ids) tuples,
    where per_pool_params[i] is a dict with log_cadence, log_gas, noise_coeffs.

    reduced_models: list of label strings whose noise_coeffs correspond to
    reduced x_obs (4 columns). For those, build_x_obs(reduced=True) is used.
    """
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import build_x_obs
    import jax.numpy as jnp

    if reduced_models is None:
        reduced_models = []

    pool_ids = sorted(matched.keys())

    # Build lookup for each model's per-pool params
    model_lookups = []
    for label, per_pool_params, m_pool_ids in model_results:
        lookup = {pid: per_pool_params[i] for i, pid in enumerate(m_pool_ids)}
        model_lookups.append((label, lookup))

    def r2(v_arb, v_noise, y):
        log_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
        ss_res = np.sum((log_pred - y) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / max(ss_tot, 1e-10)

    predictions = {}
    for pid in pool_ids:
        entry = matched[pid]
        panel = entry["panel"]
        coeffs = entry["coeffs"]
        day_indices = entry["day_indices"]

        x_obs_full = build_x_obs(panel)
        x_obs_red = None  # lazy-build only if needed
        y_obs = panel["log_volume"].values.astype(float)

        p = {
            "dates": pd.to_datetime(panel["date"].values),
            "y_obs": y_obs,
            "actual_vol": np.exp(y_obs),
            "chain": entry["chain"],
            "tokens": entry["tokens"],
            "fee": entry["fee"],
            "median_tvl": float(np.exp(panel["log_tvl_lag1"].median())),
            "n_obs": len(y_obs),
        }

        # Option C
        rc = option_c_results[pid]
        v_arb_all = np.array(interpolate_pool_daily(
            coeffs, jnp.float64(rc["log_cadence"]),
            jnp.float64(np.exp(rc["log_gas"]))))
        v_arb_c = v_arb_all[day_indices]
        v_noise_c = np.exp(x_obs_full @ rc["noise_coeffs"])
        p["v_arb_c"] = v_arb_c
        p["v_noise_c"] = v_noise_c
        p["r2_c"] = r2(v_arb_c, v_noise_c, y_obs)
        p["cadence_c"] = rc["cadence_minutes"]
        p["gas_c"] = rc["gas_usd"]

        # Each joint model
        for label, lookup in model_lookups:
            if pid in lookup:
                mp = lookup[pid]
                v_arb_all = np.array(interpolate_pool_daily(
                    coeffs, jnp.float64(mp["log_cadence"]),
                    jnp.float64(np.exp(mp["log_gas"]))))
                v_arb = v_arb_all[day_indices]

                # Use reduced x_obs for models that were trained with it
                if label in reduced_models:
                    if x_obs_red is None:
                        x_obs_red = build_x_obs(panel, reduced=True)
                    v_noise = np.exp(x_obs_red @ mp["noise_coeffs"])
                else:
                    v_noise = np.exp(x_obs_full @ mp["noise_coeffs"])

                p[f"v_arb_{label}"] = v_arb
                p[f"v_noise_{label}"] = v_noise
                p[f"r2_{label}"] = r2(v_arb, v_noise, y_obs)
                p[f"cadence_{label}"] = mp["cadence_minutes"]
                p[f"gas_{label}"] = mp["gas_usd"]
            else:
                n = len(y_obs)
                p[f"v_arb_{label}"] = np.full(n, np.nan)
                p[f"v_noise_{label}"] = np.full(n, np.nan)
                p[f"r2_{label}"] = np.nan
                p[f"cadence_{label}"] = np.nan
                p[f"gas_{label}"] = np.nan

        predictions[pid] = p

    return predictions


# ---- Tables ----


def print_pool_table(predictions, method_labels):
    """Print per-pool results ranked by TVL."""
    ranked = sorted(predictions.items(), key=lambda x: -x[1]["median_tvl"])

    header = f"{'Pool':<24} {'Chain':<10} {'TVL':>12} {'N':>4}"
    header += f"  {'Cad_C':>6} {'R2_C':>6}"
    for label in method_labels:
        short = label[:8]
        header += f"  {'Cad_'+short:>10} {'R2_'+short:>8}"
    header += f"  {'Arb%_C':>6}"

    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")
    for pid, p in ranked:
        tokens = p["tokens"]
        if isinstance(tokens, str):
            tok_str = "/".join(t.strip()[:6] for t in tokens.split(",")[:2])
        else:
            tok_str = pid[:16]
        arb_total = p["v_arb_c"] + p["v_noise_c"]
        arb_frac = np.median(p["v_arb_c"] / np.maximum(arb_total, 1.0))

        line = (f"{tok_str:<24} {p['chain']:<10} ${p['median_tvl']:>10,.0f} "
                f"{p['n_obs']:>4}")
        line += f"  {p['cadence_c']:>5.1f}m {p['r2_c']:>6.3f}"
        for label in method_labels:
            cad = p[f"cadence_{label}"]
            r2v = p[f"r2_{label}"]
            if np.isnan(cad):
                line += f"  {'---':>10} {'---':>8}"
            else:
                line += f"  {cad:>9.1f}m {r2v:>8.3f}"
        line += f"  {arb_frac:>5.1%}"
        print(line)


def print_r2_comparison(predictions, method_labels):
    """Print aggregate R² comparison."""
    pool_ids = sorted(predictions.keys())

    print(f"\n{'='*70}")
    print("R² comparison (per-pool, in-sample)")
    print(f"{'='*70}")

    r2_c = [predictions[p]["r2_c"] for p in pool_ids]
    print(f"  Option C (per-pool):  median={np.median(r2_c):.4f}  mean={np.mean(r2_c):.4f}")

    for label in method_labels:
        r2_vals = [predictions[p][f"r2_{label}"] for p in pool_ids
                   if np.isfinite(predictions[p][f"r2_{label}"])]
        if r2_vals:
            print(f"  {label:<22} median={np.median(r2_vals):.4f}  mean={np.mean(r2_vals):.4f}")


def print_loss_comparison(option_c, joint_results):
    """Print joint loss comparison."""
    print(f"\n{'='*70}")
    print("Joint loss comparison")
    print(f"{'='*70}")

    c_losses = [r["loss"] for r in option_c.values()]
    print(f"  Option C (per-pool):  median={np.median(c_losses):.4f}  mean={np.mean(c_losses):.4f}")

    for label, result in joint_results:
        print(f"  {label:<22} loss={result['loss']:.4f}  (from {result['init_loss']:.4f})")


# ---- Plots ----


def plot_decomposition_pages(predictions, method, method_label, output_dir):
    """Paginated V_arb + V_noise stacked area decomposition."""
    ranked = sorted(predictions.items(), key=lambda x: -x[1]["median_tvl"])[:TOP_N]

    per_page = 10
    n_pages = (len(ranked) + per_page - 1) // per_page

    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, len(ranked))
        page_pools = ranked[start:end]
        n_this = len(page_pools)

        ncols = 2
        nrows = (n_this + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (pid, p) in enumerate(page_pools):
            ax = axes[idx // ncols][idx % ncols]
            dates = p["dates"]

            v_arb_key = f"v_arb_{method}" if method != "c" else "v_arb_c"
            v_noise_key = f"v_noise_{method}" if method != "c" else "v_noise_c"
            r2_key = f"r2_{method}" if method != "c" else "r2_c"
            cad_key = f"cadence_{method}" if method != "c" else "cadence_c"
            gas_key = f"gas_{method}" if method != "c" else "gas_c"

            v_arb = p[v_arb_key]
            v_noise = p[v_noise_key]
            r2_val = p[r2_key]
            cad = p[cad_key]
            gas = p[gas_key]

            if np.any(np.isnan(v_arb)):
                ax.text(0.5, 0.5, f"Dropped from {method_label}", fontsize=12,
                        ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_title(f"{pid[:16]} — dropped", fontsize=8)
                continue

            v_total = v_arb + v_noise
            arb_frac = np.median(v_arb / np.maximum(v_total, 1.0))
            actual = p["actual_vol"]

            ax.fill_between(dates, 0, np.maximum(v_arb, 0),
                            alpha=0.3, color="orangered", label="V_arb (grid)")
            ax.fill_between(dates, np.maximum(v_arb, 0), np.maximum(v_total, 0),
                            alpha=0.3, color="steelblue", label="V_noise")
            ax.plot(dates, actual, "k-", linewidth=0.8, alpha=0.7, label="Actual")
            ax.plot(dates, np.maximum(v_total, 0), "--", color="purple",
                    linewidth=0.8, alpha=0.7, label="Predicted total")

            ax.set_yscale("log")
            ax.set_ylabel("Daily volume (USD)", fontsize=8)

            tokens = p["tokens"]
            if isinstance(tokens, str):
                tok_str = "/".join(t.strip()[:8] for t in tokens.split(",")[:2])
            else:
                tok_str = pid[:16]

            ax.set_title(
                f"{tok_str} ({p['chain']})\n"
                f"TVL ${p['median_tvl']:,.0f}  |  R\u00b2={r2_val:.3f}  "
                f"cad={cad:.1f}min  gas=${gas:.2f}  "
                f"arb_frac={arb_frac:.1%}  n={p['n_obs']}",
                fontsize=8,
            )
            ax.legend(fontsize=6, loc="upper right")
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="x", rotation=30)

        for idx in range(n_this, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            f"Calibration decomposition: {method_label}\n"
            f"page {page + 1}/{n_pages} (top {min(TOP_N, len(ranked))} by TVL)",
            fontsize=11,
        )
        fig.tight_layout()
        safe_method = method.replace(" ", "_")
        out = os.path.join(output_dir, f"{safe_method}_page{page + 1}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_summary_distributions(predictions, method_labels, output_dir):
    """Histograms of cadence, R², arb fraction for each method."""
    pool_ids = sorted(predictions.keys())
    methods = ["c"] + method_labels
    labels = ["Option C"] + method_labels
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, 3, figsize=(15, 4 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for row, (method, label) in enumerate(zip(methods, labels)):
        cad_key = f"cadence_{method}" if method != "c" else "cadence_c"
        r2_key = f"r2_{method}" if method != "c" else "r2_c"

        cads = [predictions[p][cad_key] for p in pool_ids
                if np.isfinite(predictions[p][cad_key])]
        r2s = [predictions[p][r2_key] for p in pool_ids
               if np.isfinite(predictions[p][r2_key])]
        arb_fracs = []
        for p in pool_ids:
            v_arb_key = f"v_arb_{method}" if method != "c" else "v_arb_c"
            v_noise_key = f"v_noise_{method}" if method != "c" else "v_noise_c"
            v_arb = predictions[p][v_arb_key]
            v_noise = predictions[p][v_noise_key]
            if not np.any(np.isnan(v_arb)):
                total = v_arb + v_noise
                arb_fracs.append(np.median(v_arb / np.maximum(total, 1.0)))

        ax = axes[row, 0]
        if cads:
            ax.hist(cads, bins=20, color="orangered", alpha=0.7, edgecolor="white")
            ax.axvline(np.median(cads), color="black", linestyle="--",
                       label=f"Median={np.median(cads):.1f}min")
        ax.set_xlabel("Cadence (minutes)")
        ax.set_title(f"{label}: Cadence")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        if r2s:
            ax.hist(r2s, bins=20, color="green", alpha=0.7, edgecolor="white")
            ax.axvline(np.median(r2s), color="black", linestyle="--",
                       label=f"Median={np.median(r2s):.3f}")
        ax.set_xlabel("R\u00b2")
        ax.set_title(f"{label}: R\u00b2")
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        if arb_fracs:
            ax.hist(arb_fracs, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(np.median(arb_fracs), color="black", linestyle="--",
                       label=f"Median={np.median(arb_fracs):.2f}")
        ax.set_xlabel("Arb fraction")
        ax.set_title(f"{label}: Arb fraction")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(output_dir, "summary_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_r2_scatter(predictions, method_labels, output_dir):
    """Scatter: Option C R² vs each joint method R²."""
    pool_ids = sorted(predictions.keys())
    n = len(method_labels)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, method_labels):
        r2_c = []
        r2_m = []
        for p in pool_ids:
            rc = predictions[p]["r2_c"]
            rm = predictions[p][f"r2_{label}"]
            if np.isfinite(rc) and np.isfinite(rm):
                r2_c.append(rc)
                r2_m.append(rm)

        ax.scatter(r2_c, r2_m, alpha=0.7, s=30, edgecolors="k", linewidth=0.5)
        lo = min(min(r2_c), min(r2_m)) if r2_c else 0
        hi = max(max(r2_c), max(r2_m)) if r2_c else 1
        margin = (hi - lo) * 0.05 + 0.01
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("Option C R\u00b2")
        ax.set_ylabel(f"{label} R\u00b2")
        ax.set_title(f"Option C vs {label}")

        # Count wins
        wins = sum(1 for c, m in zip(r2_c, r2_m) if m > c)
        ax.text(0.05, 0.95, f"{label} wins: {wins}/{len(r2_c)}",
                transform=ax.transAxes, fontsize=9, va="top")

    fig.tight_layout()
    out = os.path.join(output_dir, "r2_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_cadence_by_chain(predictions, method_labels, output_dir):
    """Cadence distributions by chain for Option C and each method."""
    pool_ids = sorted(predictions.keys())
    chains = sorted(set(predictions[p]["chain"] for p in pool_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(chains), 1)))
    chain_color = {c: colors[i] for i, c in enumerate(chains)}

    methods = ["c"] + method_labels
    labels = ["Option C"] + method_labels
    n = len(methods)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, method, label in zip(axes, methods, labels):
        cad_key = f"cadence_{method}" if method != "c" else "cadence_c"
        for chain in chains:
            cads = [predictions[p][cad_key] for p in pool_ids
                    if predictions[p]["chain"] == chain
                    and np.isfinite(predictions[p][cad_key])]
            if cads:
                ax.scatter([chain] * len(cads), cads, color=chain_color[chain],
                           alpha=0.7, s=40, edgecolors="k", linewidth=0.3)
        ax.set_ylabel("Cadence (minutes)")
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    out = os.path.join(output_dir, "cadence_by_chain.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- JSON export ----


def save_results_json(predictions, option_c_results, joint_results, output_dir):
    """Save all fitted parameters and diagnostics."""
    out = {"option_c": {}}
    for pid, r in option_c_results.items():
        out["option_c"][pid] = {
            "log_cadence": r["log_cadence"],
            "log_gas": r["log_gas"],
            "noise_coeffs": r["noise_coeffs"].tolist(),
            "loss": r["loss"],
            "converged": bool(r["converged"]),
            "cadence_minutes": r["cadence_minutes"],
            "gas_usd": r["gas_usd"],
            "chain": r.get("chain", ""),
            "fee": r.get("fee", 0),
            "tokens": r.get("tokens", ""),
        }

    for label, result in joint_results:
        entry = {
            "loss": result["loss"],
            "init_loss": result["init_loss"],
            "converged": bool(result["converged"]),
            "n_pools": result.get("n_pools", 0),
            "k_attr": result.get("k_attr", 0),
            "pool_ids": result.get("pool_ids", []),
            "attr_names": result.get("attr_names", []),
        }
        # Include any scalar/array results the heads produced
        for key in ["bias_cad", "W_cad", "bias_gas", "W_gas",
                     "bias_noise", "W_noise", "noise_coeffs"]:
            if key in result:
                val = result[key]
                entry[key] = val.tolist() if hasattr(val, "tolist") else val
        out[label] = entry

    # Per-pool R² for each method
    pool_ids = sorted(predictions.keys())
    method_labels = [label for label, _ in joint_results]
    per_pool_r2 = {}
    for pid in pool_ids:
        p = predictions[pid]
        row = {"r2_c": p["r2_c"]}
        for label in method_labels:
            row[f"r2_{label}"] = p[f"r2_{label}"]
        per_pool_r2[pid] = row
    out["per_pool_r2"] = per_pool_r2

    path = os.path.join(output_dir, "mlp_calibration_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ---- Main ----


def _save_per_pool_results(results, output_path, label="option_c_reduced"):
    """Save per-pool fit results to JSON immediately."""
    out = {}
    for pid, r in results.items():
        out[pid] = {
            "log_cadence": r["log_cadence"],
            "log_gas": r["log_gas"],
            "noise_coeffs": r["noise_coeffs"].tolist(),
            "loss": r["loss"],
            "converged": bool(r["converged"]),
            "cadence_minutes": r["cadence_minutes"],
            "gas_usd": r["gas_usd"],
            "chain": r.get("chain", ""),
            "fee": r.get("fee", 0),
            "tokens": r.get("tokens", ""),
        }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({label: out}, f, indent=2)
    print(f"  Saved {len(out)} pool results to {output_path}")


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("MLP Calibration: MLPNoiseHead vs Linear Baseline")
    print("=" * 70)

    panel, matched = load_and_match()

    # Step 0: 4-covariate per-pool fits (fast, save immediately)
    option_c_reduced = run_option_c_reduced(matched)
    _save_per_pool_results(
        option_c_reduced,
        os.path.join(OUTPUT_DIR, "option_c_reduced.json"),
        label="option_c_reduced",
    )

    # Step 1: Option C baseline (8-covariate)
    option_c = run_option_c(matched)

    # Step 2: Filter pathological pools
    matched_clean, option_c_clean = filter_pathological(matched, option_c)

    # Step 3: Fit each joint model
    linear_result, linear_model, jdata = run_linear_joint(
        matched_clean, option_c_clean)
    mlp_noise_result, mlp_noise_model, _ = run_mlp_noise_joint(
        matched_clean, option_c_clean)
    mlp_full_result, mlp_full_model, _ = run_mlp_full_joint(
        matched_clean, option_c_clean)

    # Two-stage: cadence identified with per-pool noise, then MLP noise
    two_stage_result, ts_s1_model, ts_s2_model, _ = run_two_stage_joint(
        matched_clean, option_c_clean)

    # Reduced x_obs: prune sigma/fee features from noise covariates
    reduced_result, reduced_model, jdata_reduced = run_reduced_joint(
        matched_clean, option_c_clean)

    # Step 4: Extract per-pool params from each model
    linear_pp = _extract_per_pool_params(linear_model, linear_result, jdata)
    mlp_noise_pp = _extract_per_pool_params(mlp_noise_model, mlp_noise_result, jdata)
    mlp_full_pp = _extract_per_pool_params(mlp_full_model, mlp_full_result, jdata)
    two_stage_pp = _extract_two_stage_per_pool(
        ts_s1_model, ts_s2_model, two_stage_result, jdata)
    reduced_pp = _extract_per_pool_params(
        reduced_model, reduced_result, jdata_reduced)

    method_labels = ["linear", "mlp_noise", "mlp_full", "two_stage", "reduced"]
    model_results_for_pred = [
        ("linear", linear_pp, jdata.pool_ids),
        ("mlp_noise", mlp_noise_pp, jdata.pool_ids),
        ("mlp_full", mlp_full_pp, jdata.pool_ids),
        ("two_stage", two_stage_pp, jdata.pool_ids),
        ("reduced", reduced_pp, jdata_reduced.pool_ids),
    ]

    # Step 5: Per-pool predictions
    print("\nComputing per-pool predictions...")
    predictions = compute_per_pool_predictions(
        matched_clean, option_c_clean, model_results_for_pred,
        reduced_models=["reduced"])

    # Step 6: Tables
    print_pool_table(predictions, method_labels)
    print_r2_comparison(predictions, method_labels)
    print_loss_comparison(option_c_clean, [
        ("linear", linear_result),
        ("mlp_noise", mlp_noise_result),
        ("mlp_full", mlp_full_result),
        ("two_stage", two_stage_result),
        ("reduced", reduced_result),
    ])

    # Step 7: Plots
    print("\nGenerating plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_decomposition_pages(predictions, "c", "Option C (per-pool)", OUTPUT_DIR)
    plot_decomposition_pages(predictions, "linear", "Linear shared noise", OUTPUT_DIR)
    plot_decomposition_pages(predictions, "mlp_noise", "MLP noise (linear cad)", OUTPUT_DIR)
    plot_decomposition_pages(predictions, "mlp_full", "Full MLP (MLP cad + MLP noise)", OUTPUT_DIR)
    plot_decomposition_pages(predictions, "two_stage", "Two-stage (linear cad -> MLP noise)", OUTPUT_DIR)
    plot_decomposition_pages(predictions, "reduced", "Reduced x_obs (k_obs=4)", OUTPUT_DIR)

    plot_summary_distributions(predictions, method_labels, OUTPUT_DIR)
    plot_r2_scatter(predictions, method_labels, OUTPUT_DIR)
    plot_cadence_by_chain(predictions, method_labels, OUTPUT_DIR)

    # Step 8: JSON export
    save_results_json(predictions, option_c_clean, [
        ("linear", linear_result),
        ("mlp_noise", mlp_noise_result),
        ("mlp_full", mlp_full_result),
        ("two_stage", two_stage_result),
        ("reduced", reduced_result),
    ], OUTPUT_DIR)

    print(f"\n{'='*70}")
    print(f"Done. Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
