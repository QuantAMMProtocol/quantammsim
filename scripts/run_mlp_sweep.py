"""Hyperparameter sweep for MLP calibration models.

Sweeps maxiter, alpha, hidden size, maxcor, and loss_type to find settings
where the MLP converges and achieves the best per-pool R2.

Usage:
    python scripts/run_mlp_sweep.py [--phase 1|2|3|all]
"""

import argparse
import json
import os
import time

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
    "results", "mlp_sweep",
)
OPTION_C_LOSS_CUTOFF = 5.0


def load_data():
    """Load panel, match grids, run Option C, return clean data."""
    from quantammsim.calibration.per_pool_fit import fit_all_pools
    from quantammsim.calibration.pool_data import (
        build_pool_attributes,
        build_x_obs,
        match_grids_to_panel,
        replace_panel_volatility_with_binance,
    )

    panel = pd.read_parquet(PANEL_CACHE)
    print("Replacing volatility with Binance minute data...")
    panel = replace_panel_volatility_with_binance(panel)
    matched = match_grids_to_panel(GRID_DIR, panel)
    print(f"Matched: {len(matched)} pools with grids")

    # Option C baseline
    print(f"\n--- Option C: per-pool fits ({len(matched)} pools, gas fixed) ---")
    option_c = fit_all_pools(matched, fix_gas_to_chain=True)
    n_conv = sum(1 for r in option_c.values() if r["converged"])
    losses = [r["loss"] for r in option_c.values()]
    print(f"  Converged: {n_conv}/{len(option_c)}")
    print(f"  Loss: median={np.median(losses):.4f}, mean={np.mean(losses):.4f}")

    # Drop pathological pools
    dropped = [p for p, r in option_c.items() if r["loss"] > OPTION_C_LOSS_CUTOFF]
    matched_clean = {k: v for k, v in matched.items() if k not in dropped}
    option_c_clean = {k: v for k, v in option_c.items() if k not in dropped}
    if dropped:
        print(f"  Dropping {len(dropped)} pools (loss > {OPTION_C_LOSS_CUTOFF})")

    return matched_clean, option_c_clean


def compute_per_pool_r2(model, result, jdata, matched):
    """Compute per-pool R2 for a fitted model."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import build_x_obs
    import jax.numpy as jnp

    params = jnp.array(result["params_flat"])
    n_pools = result["n_pools"]
    k_attr = result["k_attr"]
    (cs, ce), (gs, ge), (ns, ne) = model._head_slices(n_pools, k_attr)

    cad_slice = params[cs:ce]
    gas_slice = params[gs:ge]
    noise_slice = params[ns:ne]

    r2s = []
    for i, pid in enumerate(jdata.pool_ids):
        x_attr_i = jdata.x_attr[i]
        log_cad = float(model.cadence_head.predict(cad_slice, i, x_attr_i))
        log_gas = float(model.gas_head.predict(gas_slice, i, x_attr_i))
        noise_c = np.array(model.noise_head.predict(noise_slice, i, x_attr_i))

        entry = matched[pid]
        panel = entry["panel"]
        coeffs = entry["coeffs"]
        day_indices = entry["day_indices"]
        x_obs = build_x_obs(panel)
        y_obs = panel["log_volume"].values.astype(float)

        v_arb_all = np.array(interpolate_pool_daily(
            coeffs, jnp.float64(log_cad), jnp.float64(np.exp(log_gas))))
        v_arb = v_arb_all[day_indices]
        v_noise = np.exp(x_obs @ noise_c)
        log_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
        ss_res = np.sum((log_pred - y_obs) ** 2)
        ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
        r2s.append(1 - ss_res / max(ss_tot, 1e-10))

    return np.array(r2s)


def run_single(matched_clean, option_c_clean, config):
    """Run a single sweep configuration. Returns result dict."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import (
        FixedHead, LinearHead, MLPHead, MLPNoiseHead, SharedLinearNoiseHead,
    )
    from quantammsim.calibration.joint_fit import prepare_joint_data
    from quantammsim.calibration.loss import CHAIN_GAS_USD

    jdata = prepare_joint_data(
        matched_clean, drop_chain_dummies=True, fix_gas_to_chain=True)

    gas_values = []
    for pid in jdata.pool_ids:
        chain = matched_clean[pid]["chain"]
        gas_usd = CHAIN_GAS_USD.get(chain, 1.0)
        gas_values.append(np.log(max(gas_usd, 1e-6)))
    gas_values = np.array(gas_values)

    # Build model from config
    alpha_cad = config.get("alpha_cad", 0.01)
    alpha_noise = config.get("alpha_noise", 0.01)
    hidden = config.get("hidden", 16)
    maxiter = config.get("maxiter", 500)
    maxcor = config.get("maxcor", 10)
    loss_type = config.get("loss_type", "l2")
    cad_type = config.get("cad_type", "linear")  # "linear" or "mlp"
    noise_type = config.get("noise_type", "mlp")  # "mlp" or "linear"

    # Cadence head
    if cad_type == "mlp":
        cad_head = MLPHead("cad", hidden=hidden, alpha=alpha_cad)
    else:
        cad_head = LinearHead("cad", alpha=alpha_cad)

    # Noise head
    if noise_type == "mlp":
        noise_head = MLPNoiseHead(hidden=hidden, alpha=alpha_noise)
    else:
        noise_head = SharedLinearNoiseHead(alpha=alpha_noise)

    model = CalibrationModel(
        cadence_head=cad_head,
        gas_head=FixedHead("gas", gas_values),
        noise_head=noise_head,
        loss_type=loss_type,
    )

    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    n_params = model.n_params(n_pools, k_attr)

    # Override maxcor in the fit method by monkey-patching options
    import scipy.optimize
    _orig_minimize = scipy.optimize.minimize

    def patched_minimize(fun, x0, **kwargs):
        opts = kwargs.get("options", {})
        opts["maxcor"] = maxcor
        opts["maxiter"] = maxiter
        kwargs["options"] = opts
        return _orig_minimize(fun, x0, **kwargs)

    scipy.optimize.minimize = patched_minimize
    try:
        t0 = time.time()
        result = model.fit(jdata, maxiter=maxiter, warm_start=option_c_clean)
        wall_time = time.time() - t0
    finally:
        scipy.optimize.minimize = _orig_minimize

    # Compute per-pool R2
    r2s = compute_per_pool_r2(model, result, jdata, matched_clean)

    return {
        "config": config,
        "n_params": n_params,
        "n_pools": n_pools,
        "init_loss": result["init_loss"],
        "final_loss": result["loss"],
        "converged": result["converged"],
        "wall_time_s": round(wall_time, 1),
        "r2_median": round(float(np.median(r2s)), 4),
        "r2_mean": round(float(np.mean(r2s)), 4),
        "r2_p10": round(float(np.percentile(r2s, 10)), 4),
        "r2_p25": round(float(np.percentile(r2s, 25)), 4),
        "r2_p75": round(float(np.percentile(r2s, 75)), 4),
        "r2_p90": round(float(np.percentile(r2s, 90)), 4),
        "r2_min": round(float(np.min(r2s)), 4),
        "r2_max": round(float(np.max(r2s)), 4),
        "n_positive_r2": int(np.sum(r2s > 0)),
    }


def print_result(res, idx=None):
    """Print a single result row."""
    c = res["config"]
    prefix = f"[{idx}] " if idx is not None else ""
    label = (f"{c.get('cad_type','linear')}_cad + "
             f"{c.get('noise_type','mlp')}_noise")
    print(f"{prefix}{label}  "
          f"h={c.get('hidden',16):2d}  "
          f"a_c={c.get('alpha_cad',0.01):.4f}  "
          f"a_n={c.get('alpha_noise',0.01):.4f}  "
          f"maxiter={c.get('maxiter',500):5d}  "
          f"maxcor={c.get('maxcor',10):2d}  "
          f"loss={c.get('loss_type','l2'):5s}  |  "
          f"L={res['final_loss']:7.4f}  "
          f"conv={str(res['converged']):5s}  "
          f"R2_med={res['r2_median']:+.4f}  "
          f"R2_mean={res['r2_mean']:+.4f}  "
          f"R2+={res['n_positive_r2']:2d}/{res['n_pools']}  "
          f"{res['wall_time_s']:5.1f}s")


def run_phase_1(matched_clean, option_c_clean):
    """Phase 1: Sweep maxiter to diagnose convergence."""
    print("\n" + "=" * 80)
    print("Phase 1: maxiter sweep (MLP noise, linear cadence)")
    print("=" * 80)

    configs = []
    for maxiter in [500, 2000, 5000]:
        configs.append({
            "cad_type": "linear", "noise_type": "mlp",
            "maxiter": maxiter, "hidden": 16,
            "alpha_cad": 0.01, "alpha_noise": 0.01,
            "maxcor": 10, "loss_type": "l2",
            "label": f"maxiter={maxiter}",
        })
    # Also sweep maxiter for full MLP
    for maxiter in [500, 2000, 5000]:
        configs.append({
            "cad_type": "mlp", "noise_type": "mlp",
            "maxiter": maxiter, "hidden": 16,
            "alpha_cad": 0.01, "alpha_noise": 0.01,
            "maxcor": 10, "loss_type": "l2",
            "label": f"full_mlp_maxiter={maxiter}",
        })

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  Running {cfg['label']}...")
        res = run_single(matched_clean, option_c_clean, cfg)
        print_result(res, i)
        results.append(res)
    return results


def run_phase_2(matched_clean, option_c_clean, best_maxiter=5000):
    """Phase 2: Regularization grid (alpha_noise x alpha_cad)."""
    print("\n" + "=" * 80)
    print("Phase 2: regularization sweep (MLP noise, linear cadence)")
    print("=" * 80)

    configs = []
    for alpha_noise in [0.0001, 0.001, 0.01, 0.1]:
        for alpha_cad in [0.001, 0.01, 0.1]:
            configs.append({
                "cad_type": "linear", "noise_type": "mlp",
                "maxiter": best_maxiter, "hidden": 16,
                "alpha_cad": alpha_cad, "alpha_noise": alpha_noise,
                "maxcor": 10, "loss_type": "l2",
                "label": f"a_n={alpha_noise}, a_c={alpha_cad}",
            })

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  Running {cfg['label']}...")
        res = run_single(matched_clean, option_c_clean, cfg)
        print_result(res, i)
        results.append(res)
    return results


def run_phase_3(matched_clean, option_c_clean,
                best_maxiter=5000, best_alpha_cad=0.01,
                best_alpha_noise=0.01):
    """Phase 3: Architecture sweep (hidden, maxcor, loss_type)."""
    print("\n" + "=" * 80)
    print("Phase 3: architecture sweep")
    print("=" * 80)

    configs = []
    # Hidden size
    for hidden in [8, 16, 32]:
        configs.append({
            "cad_type": "linear", "noise_type": "mlp",
            "maxiter": best_maxiter, "hidden": hidden,
            "alpha_cad": best_alpha_cad, "alpha_noise": best_alpha_noise,
            "maxcor": 10, "loss_type": "l2",
            "label": f"hidden={hidden}",
        })
    # maxcor
    for maxcor in [10, 30, 50]:
        configs.append({
            "cad_type": "linear", "noise_type": "mlp",
            "maxiter": best_maxiter, "hidden": 16,
            "alpha_cad": best_alpha_cad, "alpha_noise": best_alpha_noise,
            "maxcor": maxcor, "loss_type": "l2",
            "label": f"maxcor={maxcor}",
        })
    # Loss type
    for loss_type in ["l2", "huber"]:
        configs.append({
            "cad_type": "linear", "noise_type": "mlp",
            "maxiter": best_maxiter, "hidden": 16,
            "alpha_cad": best_alpha_cad, "alpha_noise": best_alpha_noise,
            "maxcor": 10, "loss_type": loss_type,
            "label": f"loss={loss_type}",
        })
    # Full MLP with best settings
    configs.append({
        "cad_type": "mlp", "noise_type": "mlp",
        "maxiter": best_maxiter, "hidden": 16,
        "alpha_cad": best_alpha_cad, "alpha_noise": best_alpha_noise,
        "maxcor": 10, "loss_type": "l2",
        "label": "full_mlp_best",
    })

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  Running {cfg['label']}...")
        res = run_single(matched_clean, option_c_clean, cfg)
        print_result(res, i)
        results.append(res)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matched_clean, option_c_clean = load_data()

    # Compute Option C R2 for reference
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import build_x_obs
    import jax.numpy as jnp

    r2s_c = []
    for pid in sorted(matched_clean.keys()):
        entry = matched_clean[pid]
        rc = option_c_clean[pid]
        panel = entry["panel"]
        x_obs = build_x_obs(panel)
        y_obs = panel["log_volume"].values.astype(float)
        day_indices = entry["day_indices"]
        coeffs = entry["coeffs"]

        v_arb_all = np.array(interpolate_pool_daily(
            coeffs, jnp.float64(rc["log_cadence"]),
            jnp.float64(np.exp(rc["log_gas"]))))
        v_arb = v_arb_all[day_indices]
        v_noise = np.exp(x_obs @ rc["noise_coeffs"])
        log_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
        ss_res = np.sum((log_pred - y_obs) ** 2)
        ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
        r2s_c.append(1 - ss_res / max(ss_tot, 1e-10))

    r2s_c = np.array(r2s_c)
    print(f"\nOption C reference: R2 median={np.median(r2s_c):.4f}, "
          f"mean={np.mean(r2s_c):.4f}, "
          f"R2>0: {np.sum(r2s_c > 0)}/{len(r2s_c)}")

    all_results = {"option_c_r2_median": float(np.median(r2s_c)),
                   "option_c_r2_mean": float(np.mean(r2s_c)),
                   "phases": {}}

    if args.phase in ("1", "all"):
        r1 = run_phase_1(matched_clean, option_c_clean)
        all_results["phases"]["1"] = r1

        # Pick best maxiter from phase 1
        best_maxiter = max(r1, key=lambda r: r["r2_median"])["config"]["maxiter"]
        print(f"\n  Best maxiter from phase 1: {best_maxiter}")
    else:
        best_maxiter = 5000

    if args.phase in ("2", "all"):
        r2 = run_phase_2(matched_clean, option_c_clean, best_maxiter)
        all_results["phases"]["2"] = r2

        best_r = max(r2, key=lambda r: r["r2_median"])
        best_alpha_cad = best_r["config"]["alpha_cad"]
        best_alpha_noise = best_r["config"]["alpha_noise"]
        print(f"\n  Best from phase 2: alpha_cad={best_alpha_cad}, "
              f"alpha_noise={best_alpha_noise}, R2_med={best_r['r2_median']}")
    else:
        best_alpha_cad = 0.01
        best_alpha_noise = 0.01

    if args.phase in ("3", "all"):
        r3 = run_phase_3(matched_clean, option_c_clean,
                         best_maxiter, best_alpha_cad, best_alpha_noise)
        all_results["phases"]["3"] = r3

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    print(f"Option C reference: R2 median={np.median(r2s_c):.4f}")
    print()
    for phase, results in all_results["phases"].items():
        print(f"Phase {phase}:")
        for i, res in enumerate(results):
            print_result(res, i)
        print()


if __name__ == "__main__":
    main()
