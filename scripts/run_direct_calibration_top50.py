"""Run direct calibration pipeline and plot top-50 style decomposition.

Steps:
  1. Load panel, match to per-day grids in results/pool_grids_v2/
  2. Option C: per-pool L-BFGS-B fits
  3. Option A: joint end-to-end optimization (warm-started from C)
  4. Paginated plots: V_arb + V_noise decomposition per pool
  5. Summary plots: cadence, gas, R², arb fraction distributions
"""

import json
import os
import sys
from datetime import date, timedelta

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
    "results", "direct_calibration_top50",
)
TRAIN_DAYS = 0  # 0 = no filter, use all available data per pool
TOP_N = 50
OPTION_C_MAXITER = 500
JOINT_MAXITER = 500
OPTION_C_LOSS_CUTOFF = 5.0  # Drop pools with Option C loss above this from joint fit


def load_and_match():
    """Load panel, match to grids. No date filter — each pool uses all data."""
    from quantammsim.calibration.pool_data import (
        match_grids_to_panel,
        replace_panel_volatility_with_binance,
    )

    panel = pd.read_parquet(PANEL_CACHE)

    # Optional date filter (TRAIN_DAYS=0 means no filter)
    if TRAIN_DAYS > 0:
        max_date = panel["date"].max()
        if not isinstance(max_date, date):
            max_date = pd.Timestamp(max_date).date()
        cutoff = max_date - timedelta(days=TRAIN_DAYS)
        panel = panel[
            panel["date"].apply(
                lambda d: d >= cutoff if isinstance(d, date)
                else pd.Timestamp(d).date() >= cutoff
            )
        ].copy()
    else:
        panel = panel.copy()

    if "log_tvl_lag1" not in panel.columns:
        panel = panel.sort_values(["pool_id", "date"]).reset_index(drop=True)
        panel["log_tvl_lag1"] = panel.groupby("pool_id")["log_tvl"].shift(1)
        panel = panel.dropna(subset=["log_tvl_lag1"]).reset_index(drop=True)

    pool_counts = panel.groupby("pool_id").size()
    valid = pool_counts[pool_counts >= 10].index
    panel = panel[panel["pool_id"].isin(valid)].copy()

    # Replace Balancer-hourly volatility with Binance-minute volatility
    print("Replacing volatility with Binance minute data...")
    panel = replace_panel_volatility_with_binance(panel)

    min_date = panel["date"].min()
    max_date = panel["date"].max()
    print(f"Panel: {len(panel)} obs, {panel['pool_id'].nunique()} pools, "
          f"{min_date} to {max_date}")

    matched = match_grids_to_panel(GRID_DIR, panel)
    print(f"Matched: {len(matched)} pools with grids")

    return panel, matched


def run_option_c(matched, fix_gas_to_chain=False):
    """Per-pool L-BFGS-B fits."""
    from quantammsim.calibration.per_pool_fit import fit_all_pools
    gas_label = " (gas fixed to chain)" if fix_gas_to_chain else ""
    print(f"\n--- Option C: per-pool fits ({len(matched)} pools){gas_label} ---")
    results = fit_all_pools(matched, fix_gas_to_chain=fix_gas_to_chain)
    n_converged = sum(1 for r in results.values() if r["converged"])
    losses = [r["loss"] for r in results.values()]
    print(f"  Converged: {n_converged}/{len(results)}")
    print(f"  Loss: median={np.median(losses):.4f}, "
          f"mean={np.mean(losses):.4f}, "
          f"range=[{np.min(losses):.4f}, {np.max(losses):.4f}]")
    return results


def run_option_a(matched, option_c_results, fix_gas_to_chain=False):
    """Joint end-to-end optimization, warm-started from Option C.

    Drops pathological pools (Option C loss > OPTION_C_LOSS_CUTOFF) from the
    joint fit to prevent them from dominating the shared mapping.
    """
    from quantammsim.calibration.joint_fit import fit_joint

    # Filter out pathological pools
    good_pools = {p: r for p, r in option_c_results.items()
                  if r["loss"] <= OPTION_C_LOSS_CUTOFF}
    dropped = set(option_c_results) - set(good_pools)
    matched_clean = {p: matched[p] for p in good_pools if p in matched}

    if dropped:
        print(f"\n  Dropping {len(dropped)} pathological pools (Option C loss > {OPTION_C_LOSS_CUTOFF}):")
        for p in sorted(dropped):
            r = option_c_results[p]
            print(f"    {p} {r['tokens']:<16} loss={r['loss']:.1f}")

    gas_label = ", gas fixed" if fix_gas_to_chain else ""
    print(f"\n--- Option A: joint fit (per_pool_noise, {len(matched_clean)} pools, "
          f"warm-start from C, no chain dummies{gas_label}) ---")
    result_ppn = fit_joint(
        matched_clean,
        mode="per_pool_noise",
        init_from_option_c=good_pools,
        maxiter=JOINT_MAXITER,
        drop_chain_dummies=True,
        fix_gas_to_chain=fix_gas_to_chain,
    )
    print(f"  Loss: {result_ppn['init_loss']:.4f} -> {result_ppn['loss']:.4f}")
    print(f"  Converged: {result_ppn['converged']}")

    print(f"\n--- Option A: joint fit (shared_noise, {len(matched_clean)} pools, "
          f"warm-start from C, no chain dummies{gas_label}) ---")
    result_sn = fit_joint(
        matched_clean,
        mode="shared_noise",
        init_from_option_c=good_pools,
        maxiter=JOINT_MAXITER,
        drop_chain_dummies=True,
        fix_gas_to_chain=fix_gas_to_chain,
    )
    print(f"  Loss: {result_sn['init_loss']:.4f} -> {result_sn['loss']:.4f}")
    print(f"  Converged: {result_sn['converged']}")

    return result_ppn, result_sn


def run_option_rf(matched, option_c_results):
    """2-stage approach: Option C per-pool fits → Ridge/RF on pool attributes.

    Drops chain dummies (too sparse for n~30), keeps 6 continuous/binary features.
    Trains both Ridge regression and RF, reports both. LOO-CV for generalization.

    Noise coefficients are taken directly from Option C (per-pool).
    Drops pathological pools (Option C loss > OPTION_C_LOSS_CUTOFF).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from quantammsim.calibration.pool_data import build_pool_attributes

    # Filter pathological pools
    good_pools = {p: r for p, r in option_c_results.items()
                  if r["loss"] <= OPTION_C_LOSS_CUTOFF}
    dropped = set(option_c_results) - set(good_pools)
    matched_clean = {p: matched[p] for p in good_pools if p in matched}

    if dropped:
        print(f"\n  Dropping {len(dropped)} pathological pools (Option C loss > {OPTION_C_LOSS_CUTOFF}):")
        for p in sorted(dropped):
            r = option_c_results[p]
            print(f"    {p} {r['tokens']:<16} loss={r['loss']:.1f}")

    # Build attributes and targets
    X_attr_full, attr_names_full, pool_ids = build_pool_attributes(matched_clean)
    n_pools = len(pool_ids)

    # Drop chain dummies — too sparse for n~30. Keep only continuous/binary features.
    non_chain_mask = [i for i, name in enumerate(attr_names_full)
                      if not name.startswith("chain_")]
    X_attr = X_attr_full[:, non_chain_mask]
    attr_names = [attr_names_full[i] for i in non_chain_mask]
    k_attr = len(attr_names)

    # Detect if gas was fixed in Option C
    gas_fixed = any(good_pools[p].get("gas_fixed", False) for p in pool_ids)

    if gas_fixed:
        print(f"\n--- Option RF: 2-stage mapping ({n_pools} pools, {k_attr} features, "
              f"cadence only — gas fixed) ---")
    else:
        print(f"\n--- Option RF: 2-stage mapping ({n_pools} pools, {k_attr} features) ---")
    print(f"  Features: {', '.join(attr_names)}")

    Y_cad = np.array([good_pools[p]["log_cadence"] for p in pool_ids])
    Y_gas = np.array([good_pools[p]["log_gas"] for p in pool_ids])

    ss_tot_cad = np.sum((Y_cad - Y_cad.mean()) ** 2)

    def compute_r2(y_true, y_pred, ss_tot):
        return 1 - np.sum((y_true - y_pred) ** 2) / max(ss_tot, 1e-10)

    # ---- Ridge regression (cadence only when gas is fixed) ----
    alphas = np.logspace(-2, 4, 50)
    ridge_cad = RidgeCV(alphas=alphas, cv=None)  # GCV/LOO built-in
    ridge_cad.fit(X_attr, Y_cad)

    Y_ridge_cad_train = ridge_cad.predict(X_attr)
    r2_ridge_cad = compute_r2(Y_cad, Y_ridge_cad_train, ss_tot_cad)

    print(f"\n  Ridge (alpha_cad={ridge_cad.alpha_:.1f}):")
    print(f"    In-sample R² cadence: {r2_ridge_cad:.3f}")

    # Ridge LOO-CV (cadence only)
    loo = LeaveOneOut()
    Y_ridge_cad_loo = np.zeros_like(Y_cad)
    for train_idx, test_idx in loo.split(X_attr):
        rc = RidgeCV(alphas=alphas, cv=None).fit(X_attr[train_idx], Y_cad[train_idx])
        Y_ridge_cad_loo[test_idx] = rc.predict(X_attr[test_idx])

    r2_ridge_loo_cad = compute_r2(Y_cad, Y_ridge_cad_loo, ss_tot_cad)
    print(f"    LOO-CV R² cadence: {r2_ridge_loo_cad:.3f}")
    print(f"    LOO-CV MAE cadence: {np.mean(np.abs(np.exp(Y_cad) - np.exp(Y_ridge_cad_loo))):.1f} min")

    # Ridge coefficients
    print(f"    Coefficients (cadence):")
    print(f"      {'intercept':<20} {ridge_cad.intercept_:>7.3f}")
    for j, name in enumerate(attr_names):
        print(f"      {name:<20} {ridge_cad.coef_[j]:>7.3f}")

    # ---- Random Forest (cadence only) ----
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3,
        max_features=min(4, k_attr),
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_attr, Y_cad)
    Y_rf_cad_train = rf.predict(X_attr)

    r2_rf_cad = compute_r2(Y_cad, Y_rf_cad_train, ss_tot_cad)

    print(f"\n  Random Forest (min_leaf=3, max_feat=4):")
    print(f"    In-sample R² cadence: {r2_rf_cad:.3f}")

    # RF LOO-CV
    Y_rf_cad_loo = np.zeros_like(Y_cad)
    for train_idx, test_idx in loo.split(X_attr):
        rf_loo = RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=3,
            max_features=min(4, k_attr), random_state=42, n_jobs=-1,
        )
        rf_loo.fit(X_attr[train_idx], Y_cad[train_idx])
        Y_rf_cad_loo[test_idx] = rf_loo.predict(X_attr[test_idx])

    r2_rf_loo_cad = compute_r2(Y_cad, Y_rf_cad_loo, ss_tot_cad)
    print(f"    LOO-CV R² cadence: {r2_rf_loo_cad:.3f}")
    print(f"    LOO-CV MAE cadence: {np.mean(np.abs(np.exp(Y_cad) - np.exp(Y_rf_cad_loo))):.1f} min")

    print(f"\n    Feature importances:")
    for j, name in enumerate(attr_names):
        print(f"      {name:<20} {rf.feature_importances_[j]:.3f}")

    # ---- Pick best LOO model ----
    best = "ridge" if r2_ridge_loo_cad >= r2_rf_loo_cad else "rf"
    print(f"\n  Best LOO model: {best} (ridge={r2_ridge_loo_cad:.3f} vs rf={r2_rf_loo_cad:.3f})")

    if best == "ridge":
        Y_best_cad_train = Y_ridge_cad_train
        Y_best_cad_loo = Y_ridge_cad_loo
        r2_best_cad = r2_ridge_cad
        r2_best_loo_cad = r2_ridge_loo_cad
    else:
        Y_best_cad_train = Y_rf_cad_train
        Y_best_cad_loo = Y_rf_cad_loo
        r2_best_cad = r2_rf_cad
        r2_best_loo_cad = r2_rf_loo_cad

    # Build result dict — gas comes from Option C (which used chain-level values)
    noise_all = np.array([good_pools[p]["noise_coeffs"] for p in pool_ids])

    result = {
        "pool_ids": pool_ids,
        "attr_names": attr_names,
        "X_attr": X_attr,
        "best_model": best,
        "predictions": {},
        "loo_predictions": {},
        "noise_coeffs": noise_all,
        "r2_train_cad": r2_best_cad,
        "r2_loo_cad": r2_best_loo_cad,
    }

    for i, pid in enumerate(pool_ids):
        log_gas_fixed = good_pools[pid]["log_gas"]
        result["predictions"][pid] = {
            "log_cadence": float(Y_best_cad_train[i]),
            "log_gas": float(log_gas_fixed),
        }
        result["loo_predictions"][pid] = {
            "log_cadence": float(Y_best_cad_loo[i]),
            "log_gas": float(log_gas_fixed),
        }

    return result


def compute_per_pool_predictions(matched, option_c_results, joint_result, rf_result=None):
    """Compute per-observation V_arb, V_noise, V_total for each pool.

    Pools not in the joint result (dropped as pathological) get NaN for
    Option A predictions. Same for RF.
    """
    from quantammsim.calibration.pool_data import build_x_obs, build_pool_attributes
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.loss import K_OBS
    import jax.numpy as jnp

    pool_ids = sorted(matched.keys())

    # Build attributes for the joint-fitted pool subset
    joint_pool_ids = joint_result["pool_ids"]
    joint_matched = {p: matched[p] for p in joint_pool_ids if p in matched}
    X_attr_joint_full, attr_names_full, _ = build_pool_attributes(joint_matched)
    # Filter to the features actually used by the joint model
    joint_attr_names = joint_result["attr_names"]
    joint_feat_idx = [attr_names_full.index(n) for n in joint_attr_names
                      if n in attr_names_full]
    X_attr_joint = X_attr_joint_full[:, joint_feat_idx]
    joint_pid_to_idx = {p: i for i, p in enumerate(joint_pool_ids)}

    # RF predictions lookup
    rf_pool_ids = rf_result["pool_ids"] if rf_result else []
    rf_pid_to_idx = {p: i for i, p in enumerate(rf_pool_ids)}

    predictions = {}
    for pid in pool_ids:
        entry = matched[pid]
        panel = entry["panel"]
        coeffs = entry["coeffs"]
        day_indices = entry["day_indices"]

        x_obs = build_x_obs(panel)
        y_obs = panel["log_volume"].values.astype(float)

        def r2(v_arb, v_noise, y):
            log_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
            ss_res = np.sum((log_pred - y) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            return 1 - ss_res / max(ss_tot, 1e-10)

        # --- Option C predictions ---
        r = option_c_results[pid]
        log_cad_c = r["log_cadence"]
        log_gas_c = r["log_gas"]
        noise_c_c = r["noise_coeffs"]

        v_arb_all_c = np.array(interpolate_pool_daily(
            coeffs, jnp.float64(log_cad_c), jnp.float64(np.exp(log_gas_c)),
        ))
        v_arb_c = v_arb_all_c[day_indices]
        v_noise_c = np.exp(x_obs @ noise_c_c)

        # --- Option A (per_pool_noise) predictions ---
        if pid in joint_pid_to_idx:
            ji = joint_pid_to_idx[pid]
            x_attr = X_attr_joint[ji]
            log_cad_a = float(joint_result["bias_cad"]) + float(x_attr @ joint_result["W_cad"])
            if joint_result.get("fix_gas"):
                gas_a = float(joint_result["gas_per_pool"][ji])
                log_gas_a = np.log(max(gas_a, 1e-6))
            else:
                log_gas_a = float(joint_result["bias_gas"]) + float(x_attr @ joint_result["W_gas"])
                gas_a = np.exp(log_gas_a)
            noise_c_a = joint_result["noise_coeffs"][ji]

            v_arb_all_a = np.array(interpolate_pool_daily(
                coeffs, jnp.float64(log_cad_a), jnp.float64(np.exp(log_gas_a)),
            ))
            v_arb_a = v_arb_all_a[day_indices]
            v_noise_a = np.exp(x_obs @ noise_c_a)
            r2_a = r2(v_arb_a, v_noise_a, y_obs)
            cad_a = np.exp(log_cad_a)
        else:
            v_arb_a = np.full(len(y_obs), np.nan)
            v_noise_a = np.full(len(y_obs), np.nan)
            r2_a = np.nan
            cad_a = np.nan
            gas_a = np.nan

        # --- Option RF predictions ---
        if rf_result and pid in rf_pid_to_idx:
            ri = rf_pid_to_idx[pid]
            rf_pred = rf_result["predictions"][pid]
            log_cad_rf = rf_pred["log_cadence"]
            log_gas_rf = rf_pred["log_gas"]
            noise_c_rf = rf_result["noise_coeffs"][ri]  # from Option C

            v_arb_all_rf = np.array(interpolate_pool_daily(
                coeffs, jnp.float64(log_cad_rf), jnp.float64(np.exp(log_gas_rf)),
            ))
            v_arb_rf = v_arb_all_rf[day_indices]
            v_noise_rf = np.exp(x_obs @ noise_c_rf)
            r2_rf = r2(v_arb_rf, v_noise_rf, y_obs)
            cad_rf = np.exp(log_cad_rf)
            gas_rf = np.exp(log_gas_rf)

            # LOO predictions (out-of-sample)
            loo_pred = rf_result["loo_predictions"][pid]
            log_cad_loo = loo_pred["log_cadence"]
            log_gas_loo = loo_pred["log_gas"]
            v_arb_all_loo = np.array(interpolate_pool_daily(
                coeffs, jnp.float64(log_cad_loo), jnp.float64(np.exp(log_gas_loo)),
            ))
            v_arb_loo = v_arb_all_loo[day_indices]
            v_noise_loo = np.exp(x_obs @ noise_c_rf)  # same noise coeffs
            r2_loo = r2(v_arb_loo, v_noise_loo, y_obs)
            cad_loo = np.exp(log_cad_loo)
            gas_loo = np.exp(log_gas_loo)
        else:
            v_arb_rf = np.full(len(y_obs), np.nan)
            v_noise_rf = np.full(len(y_obs), np.nan)
            r2_rf = np.nan
            cad_rf = np.nan
            gas_rf = np.nan
            r2_loo = np.nan
            cad_loo = np.nan
            gas_loo = np.nan

        predictions[pid] = {
            "dates": pd.to_datetime(panel["date"].values),
            "y_obs": y_obs,
            "actual_vol": np.exp(y_obs),
            # Option C
            "v_arb_c": v_arb_c,
            "v_noise_c": v_noise_c,
            "r2_c": r2(v_arb_c, v_noise_c, y_obs),
            "cadence_c": np.exp(log_cad_c),
            "gas_c": np.exp(log_gas_c),
            "converged_c": r["converged"],
            # Option A
            "v_arb_a": v_arb_a,
            "v_noise_a": v_noise_a,
            "r2_a": r2_a,
            "cadence_a": cad_a,
            "gas_a": gas_a,
            # Option RF (in-sample)
            "v_arb_rf": v_arb_rf,
            "v_noise_rf": v_noise_rf,
            "r2_rf": r2_rf,
            "cadence_rf": cad_rf,
            "gas_rf": gas_rf,
            # Option RF LOO (out-of-sample)
            "r2_rf_loo": r2_loo,
            "cadence_rf_loo": cad_loo,
            "gas_rf_loo": gas_loo,
            # Metadata
            "chain": entry["chain"],
            "tokens": entry["tokens"],
            "fee": entry["fee"],
            "median_tvl": float(np.exp(panel["log_tvl_lag1"].median())),
            "n_obs": len(y_obs),
        }

    return predictions


def plot_top50_pages(predictions, method="c"):
    """Paginated plots: V_arb + V_noise decomposition."""
    # Rank by median TVL
    ranked = sorted(
        predictions.items(),
        key=lambda x: -x[1]["median_tvl"],
    )[:TOP_N]

    suffix = {"c": "option_c", "a": "option_a", "rf": "option_rf"}[method]
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

            if method == "c":
                v_arb = p["v_arb_c"]
                v_noise = p["v_noise_c"]
                r2_val = p["r2_c"]
                cad = p["cadence_c"]
                gas = p["gas_c"]
            elif method == "rf":
                v_arb = p["v_arb_rf"]
                v_noise = p["v_noise_rf"]
                r2_val = p["r2_rf"]
                cad = p["cadence_rf"]
                gas = p["gas_rf"]
            else:
                v_arb = p["v_arb_a"]
                v_noise = p["v_noise_a"]
                r2_val = p["r2_a"]
                cad = p["cadence_a"]
                gas = p["gas_a"]

            # Skip pools with NaN predictions (dropped from this method)
            if np.any(np.isnan(v_arb)):
                ax.text(0.5, 0.5, f"Dropped from {method.upper()}", fontsize=12,
                        ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_title(f"{pid[:16]} — dropped", fontsize=8)
                continue

            v_total = v_arb + v_noise
            arb_frac = np.median(v_arb / np.maximum(v_total, 1.0))
            actual = p["actual_vol"]

            # Stacked area: V_arb bottom, V_noise on top
            ax.fill_between(dates, 0, np.maximum(v_arb, 0),
                            alpha=0.3, color="orangered", label="V_arb (grid)")
            ax.fill_between(dates, np.maximum(v_arb, 0), np.maximum(v_total, 0),
                            alpha=0.3, color="steelblue", label="V_noise (covariates)")
            ax.plot(dates, actual, "k-", linewidth=0.8, alpha=0.7, label="Actual")
            ax.plot(dates, np.maximum(v_total, 0), "--", color="purple",
                    linewidth=0.8, alpha=0.7, label="Predicted total")

            ax.set_yscale("log")
            ax.set_ylabel("Daily volume (USD)", fontsize=8)

            tokens = p["tokens"]
            if isinstance(tokens, (list, tuple)):
                tok_str = "/".join(str(t)[:8] for t in tokens[:2])
            elif isinstance(tokens, str):
                tok_str = "/".join(t.strip()[:8] for t in tokens.split(",")[:2])
            else:
                tok_str = pid[:16]

            ax.set_title(
                f"{tok_str} ({p['chain']})\n"
                f"TVL ${p['median_tvl']:,.0f}  |  R²={r2_val:.3f}  "
                f"cad={cad:.1f}min  gas=${gas:.2f}  "
                f"arb_frac={arb_frac:.1%}  n={p['n_obs']}",
                fontsize=8,
            )
            ax.legend(fontsize=6, loc="upper right")
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="x", rotation=30)

        for idx in range(n_this, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        method_label = {"c": "Option C (per-pool)", "a": "Option A (linear)",
                        "rf": "Option RF (random forest)"}[method]
        fig.suptitle(
            f"Direct calibration: V_arb + V_noise — {method_label}\n"
            f"page {page + 1}/{n_pages} "
            f"(top {min(TOP_N, len(ranked))} by median TVL, {TRAIN_DAYS}d window)",
            fontsize=11,
        )
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, f"{suffix}_page{page + 1}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_summary(predictions, option_c_results, joint_result):
    """Summary: distributions of cadence, gas, R², arb fraction for both methods."""
    pool_ids = sorted(predictions.keys())
    n = len(pool_ids)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for row, (method, label) in enumerate([("c", "Option C (per-pool)"),
                                             ("a", "Option A (joint)")]):
        cads = [predictions[p][f"cadence_{method}"] for p in pool_ids]
        gases = [predictions[p][f"gas_{method}"] for p in pool_ids]
        r2s = [predictions[p][f"r2_{method}"] for p in pool_ids]
        arb_fracs = []
        for p in pool_ids:
            v_arb = predictions[p][f"v_arb_{method}"]
            v_noise = predictions[p][f"v_noise_{method}"]
            total = v_arb + v_noise
            arb_fracs.append(np.median(v_arb / np.maximum(total, 1.0)))

        # Cadence
        ax = axes[row, 0]
        ax.hist(cads, bins=20, color="orangered", alpha=0.7, edgecolor="white")
        ax.axvline(np.median(cads), color="black", linestyle="--",
                    label=f"Median={np.median(cads):.1f}min")
        ax.set_xlabel("Cadence (minutes)")
        ax.set_title(f"{label}: Cadence")
        ax.legend(fontsize=8)

        # Gas
        ax = axes[row, 1]
        ax.hist(gases, bins=20, color="goldenrod", alpha=0.7, edgecolor="white")
        ax.axvline(np.median(gases), color="black", linestyle="--",
                    label=f"Median=${np.median(gases):.2f}")
        ax.set_xlabel("Gas (USD)")
        ax.set_title(f"{label}: Gas cost")
        ax.legend(fontsize=8)

        # R²
        ax = axes[row, 2]
        r2arr = np.array(r2s)
        ax.hist(r2arr[np.isfinite(r2arr)], bins=20, color="green", alpha=0.7,
                edgecolor="white")
        ax.axvline(np.nanmedian(r2arr), color="black", linestyle="--",
                    label=f"Median={np.nanmedian(r2arr):.3f}")
        ax.set_xlabel("R²")
        ax.set_title(f"{label}: R²")
        ax.legend(fontsize=8)

        # Arb fraction
        ax = axes[row, 3]
        ax.hist(arb_fracs, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(np.median(arb_fracs), color="black", linestyle="--",
                    label=f"Median={np.median(arb_fracs):.2f}")
        ax.set_xlabel("Arb fraction")
        ax.set_title(f"{label}: Arb fraction")
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "summary_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_c_vs_a_scatter(predictions):
    """Scatter: Option C vs Option A parameters."""
    pool_ids = sorted(predictions.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, label in [
        (axes[0], "cadence", "Cadence (min)"),
        (axes[1], "gas", "Gas (USD)"),
        (axes[2], "r2", "R²"),
    ]:
        c_vals = [predictions[p][f"{metric}_c"] for p in pool_ids]
        a_vals = [predictions[p][f"{metric}_a"] for p in pool_ids]
        ax.scatter(c_vals, a_vals, alpha=0.7, s=30, edgecolors="k", linewidth=0.5)
        lo = min(min(c_vals), min(a_vals))
        hi = max(max(c_vals), max(a_vals))
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel(f"Option C: {label}")
        ax.set_ylabel(f"Option A: {label}")
        ax.set_title(f"{label}: C vs A")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "c_vs_a_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_cadence_gas_by_chain(predictions):
    """Scatter: cadence vs gas, colored by chain."""
    pool_ids = sorted(predictions.keys())
    chains = [predictions[p]["chain"] for p in pool_ids]
    unique_chains = sorted(set(chains))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_chains), 1)))
    chain_color = {c: colors[i] for i, c in enumerate(unique_chains)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, method, label in [(axes[0], "c", "Option C"), (axes[1], "a", "Option A")]:
        for c in unique_chains:
            mask = [i for i, p in enumerate(pool_ids) if predictions[p]["chain"] == c]
            cads = [predictions[pool_ids[i]][f"cadence_{method}"] for i in mask]
            gases = [predictions[pool_ids[i]][f"gas_{method}"] for i in mask]
            ax.scatter(cads, gases, label=c, color=chain_color[c],
                       alpha=0.7, s=50, edgecolors="k", linewidth=0.5)
        ax.set_xlabel("Cadence (minutes)")
        ax.set_ylabel("Gas cost (USD)")
        ax.set_title(f"{label}: Cadence vs Gas by chain")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "cadence_gas_by_chain.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def save_results_json(predictions, option_c_results, joint_ppn, joint_sn):
    """Save fitted params as JSON for later use."""
    out = {
        "option_c": {},
        "option_a_ppn": {
            "bias_cad": joint_ppn["bias_cad"],
            "bias_gas": joint_ppn["bias_gas"],
            "W_cad": joint_ppn["W_cad"].tolist(),
            "W_gas": joint_ppn["W_gas"].tolist(),
            "noise_coeffs": joint_ppn["noise_coeffs"].tolist(),
            "loss": joint_ppn["loss"],
            "init_loss": joint_ppn["init_loss"],
            "converged": bool(joint_ppn["converged"]),
            "attr_names": joint_ppn["attr_names"],
            "pool_ids": joint_ppn["pool_ids"],
        },
        "option_a_shared": {
            "bias_cad": joint_sn["bias_cad"],
            "bias_gas": joint_sn["bias_gas"],
            "W_cad": joint_sn["W_cad"].tolist(),
            "W_gas": joint_sn["W_gas"].tolist(),
            "bias_noise": joint_sn["bias_noise"].tolist(),
            "W_noise": joint_sn["W_noise"].tolist(),
            "loss": joint_sn["loss"],
            "init_loss": joint_sn["init_loss"],
            "converged": bool(joint_sn["converged"]),
            "attr_names": joint_sn["attr_names"],
            "pool_ids": joint_sn["pool_ids"],
        },
    }
    for pid, r in option_c_results.items():
        out["option_c"][pid] = {
            "log_cadence": r["log_cadence"],
            "log_gas": r["log_gas"],
            "noise_coeffs": r["noise_coeffs"].tolist(),
            "loss": r["loss"],
            "converged": bool(r["converged"]),
            "cadence_minutes": r["cadence_minutes"],
            "gas_usd": r["gas_usd"],
            "chain": r["chain"],
            "fee": r["fee"],
            "tokens": r["tokens"],
        }

    path = os.path.join(OUTPUT_DIR, "direct_calibration_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


def print_pool_table(predictions, option_c_results):
    """Print a summary table of per-pool results."""
    ranked = sorted(
        predictions.items(),
        key=lambda x: -x[1]["median_tvl"],
    )

    has_rf = any(not np.isnan(p["cadence_rf"]) for _, p in ranked)

    print(f"\n{'='*150}")
    header = (f"{'Pool':<24} {'Chain':<10} {'TVL':>12} {'N':>4}  "
              f"{'Cad_C':>6} {'Gas_C':>7} {'R2_C':>6}  "
              f"{'Cad_A':>6} {'Gas_A':>7} {'R2_A':>6}")
    if has_rf:
        header += f"  {'Cad_RF':>6} {'Gas_RF':>7} {'R2_RF':>6} {'R2_LOO':>6}"
    header += f"  {'Arb%_C':>6}"
    print(header)
    print(f"{'-'*150}")
    for pid, p in ranked:
        tokens = p["tokens"]
        if isinstance(tokens, str):
            tok_str = "/".join(t.strip()[:6] for t in tokens.split(",")[:2])
        else:
            tok_str = pid[:16]
        arb_total_c = p["v_arb_c"] + p["v_noise_c"]
        arb_frac = np.median(p["v_arb_c"] / np.maximum(arb_total_c, 1.0))
        if np.isnan(p["cadence_a"]):
            a_str = "  --- dropped ---       "
        else:
            a_str = f"{p['cadence_a']:>5.1f}m ${p['gas_a']:>5.2f} {p['r2_a']:>6.3f}"
        line = (f"{tok_str:<24} {p['chain']:<10} ${p['median_tvl']:>10,.0f} {p['n_obs']:>4}  "
                f"{p['cadence_c']:>5.1f}m ${p['gas_c']:>5.2f} {p['r2_c']:>6.3f}  "
                f"{a_str}")
        if has_rf:
            if np.isnan(p["cadence_rf"]):
                line += "    --- dropped ---              "
            else:
                line += (f"  {p['cadence_rf']:>5.1f}m ${p['gas_rf']:>5.2f} "
                         f"{p['r2_rf']:>6.3f} {p['r2_rf_loo']:>6.3f}")
        line += f"  {arb_frac:>5.1%}"
        print(line)


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Direct Calibration Pipeline: Training + Top 50 Plots")
    print("=" * 70)

    panel, matched = load_and_match()

    # Step 1: Option C (gas fixed to chain-level costs)
    option_c = run_option_c(matched, fix_gas_to_chain=True)

    # Step 2: Option A (linear mapping, gas fixed)
    joint_ppn, joint_sn = run_option_a(matched, option_c, fix_gas_to_chain=True)

    # Step 3: Option RF (random forest 2-stage)
    rf_result = run_option_rf(matched, option_c)

    # Step 4: Compute predictions
    print("\nComputing per-pool predictions...")
    predictions = compute_per_pool_predictions(matched, option_c, joint_ppn, rf_result)

    # Step 5: Print table
    print_pool_table(predictions, option_c)

    # Print RF vs A comparison summary
    rf_pools = [p for p in predictions if not np.isnan(predictions[p]["r2_rf"])]
    if rf_pools:
        r2_c = [predictions[p]["r2_c"] for p in rf_pools]
        r2_a = [predictions[p]["r2_a"] for p in rf_pools
                if not np.isnan(predictions[p]["r2_a"])]
        r2_rf = [predictions[p]["r2_rf"] for p in rf_pools]
        r2_loo = [predictions[p]["r2_rf_loo"] for p in rf_pools]
        print(f"\n--- R² comparison (non-dropped pools) ---")
        print(f"  Option C:         median={np.median(r2_c):.4f}  mean={np.mean(r2_c):.4f}")
        if r2_a:
            print(f"  Option A (linear): median={np.median(r2_a):.4f}  mean={np.mean(r2_a):.4f}")
        print(f"  Option RF (train): median={np.median(r2_rf):.4f}  mean={np.mean(r2_rf):.4f}")
        print(f"  Option RF (LOO):   median={np.median(r2_loo):.4f}  mean={np.mean(r2_loo):.4f}")

    # Step 6: Plots
    print("\nGenerating plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_top50_pages(predictions, method="c")
    plot_top50_pages(predictions, method="a")
    plot_top50_pages(predictions, method="rf")
    plot_summary(predictions, option_c, joint_ppn)
    plot_c_vs_a_scatter(predictions)
    plot_cadence_gas_by_chain(predictions)

    # Step 7: Save results
    save_results_json(predictions, option_c, joint_ppn, joint_sn)

    print(f"\n{'='*70}")
    print(f"Done. Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
