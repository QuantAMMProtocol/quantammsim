"""Token-factored noise calibration: pooled diagnostic + full pipeline.

Phase 0: Pooled Ridge diagnostic — does cross-pool signal exist?
Phase 1: Token-factored model with lambda_delta sweep
Phase 2: LOO cross-validation
Phase 3: Comparison plots and JSON export
"""

import json
import os
import sys

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
    "results", "token_factored_calibration",
)
OPTION_C_LOSS_CUTOFF = 5.0
JOINT_MAXITER = 5000
LAMBDA_DELTAS = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]


# ---- Data loading (shared with run_mlp_calibration.py) ----


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


# ---- Phase 0: Pooled Ridge Diagnostic ----


def run_phase0_diagnostic(matched, option_c):
    """Pooled Ridge + token-dummy Ridge — go/no-go gate."""
    from sklearn.linear_model import RidgeCV

    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import (
        build_pool_attributes, build_x_obs, encode_tokens, _parse_tokens,
    )
    import jax.numpy as jnp

    print("\n" + "=" * 70)
    print("Phase 0: Pooled Ridge Diagnostic — cross-pool signal?")
    print("=" * 70)

    pool_ids = sorted(matched.keys())
    X_attr, attr_names, _ = build_pool_attributes(matched)
    pool_idx_map = {pid: i for i, pid in enumerate(pool_ids)}
    enc = encode_tokens(matched)

    all_x, all_y, all_pool_attrs, all_token_dummies = [], [], [], []

    for pid in pool_ids:
        entry = matched[pid]
        oc = option_c[pid]
        coeffs = entry["coeffs"]

        v_arb_all = np.array(interpolate_pool_daily(
            coeffs, jnp.float64(oc["log_cadence"]),
            jnp.float64(np.exp(oc["log_gas"]))))
        v_arb = v_arb_all[entry["day_indices"]]

        x_obs = build_x_obs(entry["panel"], reduced=True)
        y_obs = entry["panel"]["log_volume"].values.astype(float)
        y_residual = y_obs - np.log(np.maximum(v_arb, 1e-6))

        all_x.append(x_obs)
        all_y.append(y_residual)

        # Broadcast pool attrs to each obs
        x_attr_row = X_attr[pool_idx_map[pid]]
        all_pool_attrs.append(np.tile(x_attr_row, (len(x_obs), 1)))

        # Token dummies: one-hot for each token in the pool
        n_obs = len(x_obs)
        dummies = np.zeros((n_obs, enc["n_tokens"]), dtype=np.float64)
        toks = _parse_tokens(entry["tokens"])
        for t in toks[:2]:
            if t in enc["token_index"]:
                dummies[:, enc["token_index"][t]] = 1.0
        all_token_dummies.append(dummies)

    X_obs = np.vstack(all_x)
    y_combined = np.concatenate(all_y)
    X_pool_attrs = np.vstack(all_pool_attrs)
    X_token_dummies = np.vstack(all_token_dummies)

    # Model 1: x_obs + pool_attrs → pooled Ridge
    X_combined = np.column_stack([X_obs, X_pool_attrs])
    model1 = RidgeCV(alphas=np.logspace(-2, 4, 50))
    model1.fit(X_combined, y_combined)
    r2_pooled = model1.score(X_combined, y_combined)
    print(f"  Pooled Ridge (x_obs + pool attrs): R² = {r2_pooled:.4f}")

    # Model 2: x_obs + token_dummies → token-dummy Ridge
    X_token = np.column_stack([X_obs, X_token_dummies])
    model2 = RidgeCV(alphas=np.logspace(-2, 4, 50))
    model2.fit(X_token, y_combined)
    r2_token = model2.score(X_token, y_combined)
    print(f"  Token-dummy Ridge (x_obs + token dummies): R² = {r2_token:.4f}")

    # Model 3: x_obs + token_dummies + chain_dummies + log_fee
    chain_dummies = np.zeros((len(y_combined), enc["n_chains"]), dtype=np.float64)
    log_fees = np.zeros((len(y_combined), 1), dtype=np.float64)
    offset = 0
    for pid in pool_ids:
        n_obs = len(matched[pid]["day_indices"])
        ci = enc["chain_idx"][pool_idx_map[pid]]
        chain_dummies[offset:offset + n_obs, ci] = 1.0
        log_fees[offset:offset + n_obs, 0] = enc["log_fees"][pool_idx_map[pid]]
        offset += n_obs

    X_full = np.column_stack([X_obs, X_token_dummies, chain_dummies, log_fees])
    model3 = RidgeCV(alphas=np.logspace(-2, 4, 50))
    model3.fit(X_full, y_combined)
    r2_full = model3.score(X_full, y_combined)
    print(f"  Full Ridge (x_obs + tokens + chains + fee): R² = {r2_full:.4f}")

    # Baseline: x_obs only
    model_base = RidgeCV(alphas=np.logspace(-2, 4, 50))
    model_base.fit(X_obs, y_combined)
    r2_base = model_base.score(X_obs, y_combined)
    print(f"  Baseline (x_obs only): R² = {r2_base:.4f}")

    print(f"\n  Signal above baseline:")
    print(f"    Pool attrs:     +{r2_pooled - r2_base:.4f}")
    print(f"    Token dummies:  +{r2_token - r2_base:.4f}")
    print(f"    Full (tok+ch+fee): +{r2_full - r2_base:.4f}")

    if r2_full - r2_base < 0.01:
        print("\n  WARNING: Very weak cross-pool signal. "
              "Token factoring may not improve over per-pool fits.")

    return {
        "r2_baseline": r2_base,
        "r2_pool_attrs": r2_pooled,
        "r2_token_dummies": r2_token,
        "r2_full": r2_full,
        "n_obs_total": len(y_combined),
        "n_pools": len(pool_ids),
        "n_tokens": enc["n_tokens"],
        "n_chains": enc["n_chains"],
    }


# ---- Phase 1: Token-Factored Model ----


def _build_gas_values(jdata, matched_clean):
    """Build fixed gas values (log-space) from chain data."""
    from quantammsim.calibration.loss import CHAIN_GAS_USD
    gas_values = []
    for pid in jdata.pool_ids:
        chain = matched_clean[pid]["chain"]
        gas_usd = CHAIN_GAS_USD.get(chain, 1.0)
        gas_values.append(np.log(max(gas_usd, 1e-6)))
    return np.array(gas_values)


def run_token_factored(matched_clean, option_c_clean, lambda_delta=1.0):
    """Fit TokenFactoredNoiseHead with PerPoolHead(cadence) + FixedHead(gas)."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import (
        FixedHead, PerPoolHead, TokenFactoredNoiseHead,
    )
    from quantammsim.calibration.joint_fit import prepare_token_factored_data
    from quantammsim.calibration.pool_data import K_OBS_REDUCED

    jdata, enc = prepare_token_factored_data(matched_clean)
    n_pools = len(jdata.pool_data)

    gas_values = _build_gas_values(jdata, matched_clean)
    gas_head = FixedHead("log_gas", gas_values)
    cad_head = PerPoolHead("log_cadence", default=np.log(12.0))
    noise_head = TokenFactoredNoiseHead(
        k_obs=K_OBS_REDUCED,
        lambda_delta=lambda_delta,
        **enc,
    )

    model = CalibrationModel(cad_head, gas_head, noise_head)
    n_p = model.n_params(n_pools, jdata.x_attr.shape[1])
    print(f"\n--- Token-factored (lambda_delta={lambda_delta}) ---")
    print(f"  {n_pools} pools, {enc['n_tokens']} tokens, "
          f"{enc['n_chains']} chains, {n_p} params")

    result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=option_c_clean)
    print(f"  Loss: {result['init_loss']:.4f} -> {result['loss']:.4f}")
    print(f"  Converged: {result['converged']}")

    return result, model, jdata, enc


# ---- Phase 2: Analysis & Visualization ----


def print_token_effects(result, enc):
    """Print token effect table."""
    u = result["token_effects"]
    Gamma = result["Gamma"]
    x_token = enc["x_token"]
    token_index = enc["token_index"]
    inv_index = {v: k for k, v in token_index.items()}

    u_pred = x_token @ Gamma  # population prediction

    print(f"\n{'='*70}")
    print("Token effects (u_t) vs population prediction (x_t @ Gamma)")
    print(f"{'='*70}")
    print(f"{'Token':<12} {'u[0]':>8} {'pred[0]':>8} {'delta[0]':>8} "
          f"{'u[1]':>8} {'pred[1]':>8}")
    print("-" * 60)
    for idx in range(len(inv_index)):
        name = inv_index[idx]
        print(f"{name:<12} {u[idx,0]:>8.3f} {u_pred[idx,0]:>8.3f} "
              f"{u[idx,0]-u_pred[idx,0]:>8.3f} "
              f"{u[idx,1]:>8.3f} {u_pred[idx,1]:>8.3f}")


def print_chain_effects(result, enc):
    """Print chain effect table."""
    alpha = result["chain_effects"]
    chain_index = enc["chain_index"]
    inv_index = {v: k for k, v in chain_index.items()}

    print(f"\n{'='*70}")
    print("Chain effects (alpha)")
    print(f"{'='*70}")
    k = alpha.shape[1]
    header = f"{'Chain':<12}" + "".join(f" {'a['+str(j)+']':>8}" for j in range(k))
    print(header)
    print("-" * (12 + 9 * k))
    for idx in range(len(inv_index)):
        name = inv_index[idx]
        vals = " ".join(f"{alpha[idx, j]:>8.3f}" for j in range(k))
        print(f"{name:<12} {vals}")


def print_delta_analysis(result, enc, jdata, matched_clean):
    """Print per-pool delta analysis."""
    delta = result["noise_deltas"]
    pool_ids = jdata.pool_ids
    token_index = enc["token_index"]
    inv_token = {v: k for k, v in token_index.items()}

    print(f"\n{'='*70}")
    print("Per-pool deltas (unexplained residual)")
    print(f"{'='*70}")
    print(f"{'Pool':<24} {'Tokens':<16} {'Chain':<10} "
          f"{'|delta|':>8} {'delta[0]':>8}")
    print("-" * 70)

    delta_norms = np.linalg.norm(delta, axis=1)
    order = np.argsort(-delta_norms)
    for i in order:
        pid = pool_ids[i]
        entry = matched_clean[pid]
        print(f"{pid[:24]:<24} {entry['tokens']:<16} {entry['chain']:<10} "
              f"{delta_norms[i]:>8.3f} {delta[i, 0]:>8.3f}")


def run_lambda_sweep(matched_clean, option_c_clean):
    """Sweep lambda_delta values and report loss + delta shrinkage."""
    print(f"\n{'='*70}")
    print("Lambda_delta sweep")
    print(f"{'='*70}")
    print(f"{'lambda':>10} {'loss':>10} {'delta_norm':>12} {'mean_|d|':>10}")
    print("-" * 45)

    results = []
    for lam in LAMBDA_DELTAS:
        result, model, jdata, enc = run_token_factored(
            matched_clean, option_c_clean, lambda_delta=lam)
        delta = result["noise_deltas"]
        delta_norm = float(np.linalg.norm(delta))
        mean_abs_d = float(np.mean(np.abs(delta)))
        print(f"{lam:>10.2f} {result['loss']:>10.4f} "
              f"{delta_norm:>12.4f} {mean_abs_d:>10.4f}")
        results.append({
            "lambda_delta": lam,
            "loss": result["loss"],
            "delta_norm": delta_norm,
            "mean_abs_delta": mean_abs_d,
            "converged": result["converged"],
        })

    return results


# ---- Phase 3: LOO Cross-Validation ----


def run_loo_validation(matched_clean, option_c_clean, lambda_delta=1.0):
    """Leave-one-pool-out cross-validation via predict_new_pool."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import (
        FixedHead, PerPoolHead, TokenFactoredNoiseHead,
    )
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.joint_fit import prepare_token_factored_data
    from quantammsim.calibration.pool_data import K_OBS_REDUCED, build_x_obs, _parse_tokens
    import jax.numpy as jnp

    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)

    print(f"\n{'='*70}")
    print(f"LOO Cross-Validation (lambda_delta={lambda_delta})")
    print(f"{'='*70}")

    loo_results = []
    for hold_out_pid in pool_ids:
        # Build training set without hold-out pool
        train_matched = {p: matched_clean[p] for p in pool_ids if p != hold_out_pid}
        train_oc = {p: option_c_clean[p] for p in pool_ids if p != hold_out_pid}

        if len(train_matched) < 3:
            continue

        # Fit on training set
        jdata, enc = prepare_token_factored_data(train_matched)
        gas_values = _build_gas_values(jdata, train_matched)

        noise_head = TokenFactoredNoiseHead(
            k_obs=K_OBS_REDUCED,
            lambda_delta=lambda_delta,
            **enc,
        )
        model = CalibrationModel(
            PerPoolHead("log_cadence", default=np.log(12.0)),
            FixedHead("log_gas", gas_values),
            noise_head,
        )
        result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=train_oc)

        # Extract noise params and predict for hold-out pool
        n_train = len(jdata.pool_data)
        k_attr = jdata.x_attr.shape[1]
        (_, _), (_, _), (ns, ne) = model._head_slices(n_train, k_attr)
        noise_params = result["params_flat"][ns:ne]

        ho_entry = matched_clean[hold_out_pid]
        toks = _parse_tokens(ho_entry["tokens"])
        ho_pred = noise_head.predict_new_pool(
            noise_params, toks[0], toks[1],
            ho_entry["chain"], ho_entry["fee"],
            n_pools=n_train,
        )

        # Evaluate hold-out R²
        ho_panel = ho_entry["panel"]
        x_obs_ho = build_x_obs(ho_panel, reduced=True)
        y_obs_ho = ho_panel["log_volume"].values.astype(float)

        # Use Option C cadence for the hold-out pool (not predicting cadence)
        oc_ho = option_c_clean[hold_out_pid]
        v_arb_all = np.array(interpolate_pool_daily(
            ho_entry["coeffs"],
            jnp.float64(oc_ho["log_cadence"]),
            jnp.float64(np.exp(oc_ho["log_gas"])),
        ))
        v_arb = v_arb_all[ho_entry["day_indices"]]
        v_noise = np.exp(x_obs_ho @ ho_pred["noise_coeffs"])
        log_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
        ss_res = np.sum((log_pred - y_obs_ho) ** 2)
        ss_tot = np.sum((y_obs_ho - y_obs_ho.mean()) ** 2)
        r2_loo = 1 - ss_res / max(ss_tot, 1e-10)

        # Compare with Option C in-sample R²
        v_noise_c = np.exp(build_x_obs(ho_panel, reduced=True) @ oc_ho["noise_coeffs"][:K_OBS_REDUCED])
        log_pred_c = np.log(np.maximum(v_arb + v_noise_c, 1e-6))
        ss_res_c = np.sum((log_pred_c - y_obs_ho) ** 2)
        r2_c = 1 - ss_res_c / max(ss_tot, 1e-10)

        loo_results.append({
            "pool_id": hold_out_pid,
            "r2_loo": r2_loo,
            "r2_option_c": r2_c,
            "tokens": ho_entry["tokens"],
            "chain": ho_entry["chain"],
        })

        print(f"  {hold_out_pid[:16]} ({ho_entry['tokens']:<14}) "
              f"R²_LOO={r2_loo:.3f}  R²_C={r2_c:.3f}  "
              f"{'BETTER' if r2_loo > r2_c else 'worse'}")

    if loo_results:
        r2s_loo = [r["r2_loo"] for r in loo_results]
        r2s_c = [r["r2_option_c"] for r in loo_results]
        n_better = sum(1 for r in loo_results if r["r2_loo"] > r["r2_option_c"])
        print(f"\n  LOO median R²: {np.median(r2s_loo):.4f} "
              f"(Option C: {np.median(r2s_c):.4f})")
        print(f"  LOO wins: {n_better}/{len(loo_results)}")

    return loo_results


# ---- Plots ----


def plot_lambda_sweep(sweep_results, output_dir):
    """Plot loss and delta norm vs lambda_delta."""
    lambdas = [r["lambda_delta"] for r in sweep_results]
    losses = [r["loss"] for r in sweep_results]
    delta_norms = [r["delta_norm"] for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogx(lambdas, losses, "o-", color="steelblue")
    ax1.set_xlabel("lambda_delta")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs lambda_delta")

    ax2.semilogx(lambdas, delta_norms, "o-", color="orangered")
    ax2.set_xlabel("lambda_delta")
    ax2.set_ylabel("||delta||")
    ax2.set_title("Delta norm vs lambda_delta")

    fig.tight_layout()
    out = os.path.join(output_dir, "lambda_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_token_effects(result, enc, output_dir):
    """Bar chart of token effects (intercept coefficient)."""
    u = result["token_effects"]
    token_index = enc["token_index"]
    inv_index = {v: k for k, v in token_index.items()}
    names = [inv_index[i] for i in range(len(inv_index))]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
    x = np.arange(len(names))
    ax.bar(x, u[:, 0], color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("u_t[0] (intercept effect)")
    ax.set_title("Token effects on noise intercept")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    out = os.path.join(output_dir, "token_effects.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_loo_scatter(loo_results, output_dir):
    """Scatter: Option C R² vs LOO R²."""
    if not loo_results:
        return

    r2_c = [r["r2_option_c"] for r in loo_results]
    r2_loo = [r["r2_loo"] for r in loo_results]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(r2_c, r2_loo, alpha=0.7, s=40, edgecolors="k", linewidth=0.5)
    lo = min(min(r2_c), min(r2_loo))
    hi = max(max(r2_c), max(r2_loo))
    margin = (hi - lo) * 0.05 + 0.01
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Option C R² (in-sample)")
    ax.set_ylabel("Token-factored R² (LOO)")
    ax.set_title("LOO Cross-Validation: Token-Factored vs Option C")

    n_better = sum(1 for c, l in zip(r2_c, r2_loo) if l > c)
    ax.text(0.05, 0.95, f"LOO wins: {n_better}/{len(r2_c)}",
            transform=ax.transAxes, fontsize=10, va="top")

    fig.tight_layout()
    out = os.path.join(output_dir, "loo_scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- Main ----


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Token-Factored Noise Calibration")
    print("=" * 70)

    panel, matched = load_and_match()

    # Step 1: Option C baseline (reduced x_obs)
    from quantammsim.calibration.per_pool_fit import fit_all_pools
    print(f"\n--- Option C Reduced: per-pool fits ({len(matched)} pools) ---")
    option_c = fit_all_pools(matched, fix_gas_to_chain=True, reduced=True)
    losses = [r["loss"] for r in option_c.values()]
    print(f"  Loss: median={np.median(losses):.4f}, mean={np.mean(losses):.4f}")

    # Step 2: Filter pathological pools
    matched_clean, option_c_clean = filter_pathological(matched, option_c)

    # Phase 0: Diagnostic
    diag = run_phase0_diagnostic(matched_clean, option_c_clean)

    # Phase 1: Token-factored model (default lambda)
    result, model, jdata, enc = run_token_factored(
        matched_clean, option_c_clean, lambda_delta=1.0)

    # Analysis
    print_token_effects(result, enc)
    print_chain_effects(result, enc)
    print_delta_analysis(result, enc, jdata, matched_clean)

    # Lambda sweep
    sweep_results = run_lambda_sweep(matched_clean, option_c_clean)

    # Phase 2: LOO cross-validation
    loo_results = run_loo_validation(
        matched_clean, option_c_clean, lambda_delta=1.0)

    # Phase 3: Plots & export
    print("\nGenerating plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_lambda_sweep(sweep_results, OUTPUT_DIR)
    plot_token_effects(result, enc, OUTPUT_DIR)
    plot_loo_scatter(loo_results, OUTPUT_DIR)

    # JSON export
    export = {
        "phase0_diagnostic": diag,
        "token_factored": {
            "loss": result["loss"],
            "init_loss": result["init_loss"],
            "converged": result["converged"],
            "n_pools": result["n_pools"],
            "n_tokens": enc["n_tokens"],
            "n_chains": enc["n_chains"],
            "token_index": enc["token_index"],
            "chain_index": enc["chain_index"],
            "token_effects": result["token_effects"].tolist(),
            "Gamma": result["Gamma"].tolist(),
            "chain_effects": result["chain_effects"].tolist(),
            "beta_fee": result["beta_fee"].tolist(),
            "noise_deltas": result["noise_deltas"].tolist(),
            "noise_coeffs": result["noise_coeffs"].tolist(),
        },
        "lambda_sweep": sweep_results,
        "loo_results": loo_results,
    }
    json_path = os.path.join(OUTPUT_DIR, "token_factored_results.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    print(f"\n{'='*70}")
    print(f"Done. Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
