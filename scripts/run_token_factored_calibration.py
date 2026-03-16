"""Token-factored noise calibration v2: canonicalization + cross-pool lag features.

Phase 0: Pooled Ridge diagnostic — does cross-pool signal exist?
Phase 1: Token-factored model with lambda_delta annealing sweep
Phase 2: LOO cross-validation (baseline vs cross-pool ablation)
Phase 3: Comparison plots and JSON export
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
    "results", "token_factored_calibration",
)
OPTION_C_LOSS_CUTOFF = 5.0
JOINT_MAXITER = 5000
# Sorted descending for warm-start annealing (highest regularization first)
LAMBDA_DELTAS = [10.0, 5.0, 1.0, 0.5, 0.1, 0.01]


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
    X_attr, _, _ = build_pool_attributes(matched)
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

    # Model 1: x_obs + pool_attrs
    X_combined = np.column_stack([X_obs, X_pool_attrs])
    model1 = RidgeCV(alphas=np.logspace(-2, 4, 50))
    model1.fit(X_combined, y_combined)
    r2_pooled = model1.score(X_combined, y_combined)
    print(f"  Pooled Ridge (x_obs + pool attrs): R² = {r2_pooled:.4f}")

    # Model 2: x_obs + token_dummies
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


def _result_to_warm_start(result):
    """Extract per-pool warm_start dict from a CalibrationModel fit result.

    Returns dict: pool_id -> {log_cadence, noise_coeffs} suitable for
    passing as warm_start to CalibrationModel.fit().
    """
    pool_ids = result["pool_ids"]
    warm = {}
    for i, pid in enumerate(pool_ids):
        entry = {}
        # Cadence: from PerPoolHead
        if "log_cadence_per_pool" in result:
            entry["log_cadence"] = float(result["log_cadence_per_pool"][i])
        # Noise: per-pool coefficients
        if "noise_coeffs" in result:
            entry["noise_coeffs"] = result["noise_coeffs"][i]
        warm[pid] = entry
    return warm


def run_token_factored(
    matched_clean, option_c_clean, lambda_delta=1.0,
    cross_pool=False, warm_start=None,
):
    """Fit TokenFactoredNoiseHead with PerPoolHead(cadence) + FixedHead(gas)."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import (
        FixedHead, PerPoolHead, TokenFactoredNoiseHead,
    )
    from quantammsim.calibration.joint_fit import prepare_token_factored_data
    from quantammsim.calibration.pool_data import K_OBS_CROSS, K_OBS_REDUCED

    k_obs = K_OBS_CROSS if cross_pool else K_OBS_REDUCED

    jdata, enc = prepare_token_factored_data(
        matched_clean, cross_pool=cross_pool,
    )
    n_pools = len(jdata.pool_data)

    gas_values = _build_gas_values(jdata, matched_clean)
    gas_head = FixedHead("log_gas", gas_values)
    cad_head = PerPoolHead("log_cadence", default=np.log(12.0))
    noise_head = TokenFactoredNoiseHead(
        k_obs=k_obs,
        lambda_delta=lambda_delta,
        **enc,
    )

    model = CalibrationModel(cad_head, gas_head, noise_head)
    n_p = model.n_params(n_pools, jdata.x_attr.shape[1])
    cp_tag = " [cross-pool]" if cross_pool else ""
    print(f"\n--- Token-factored (lambda_delta={lambda_delta}){cp_tag} ---")
    print(f"  {n_pools} pools, {enc['n_tokens']} tokens, "
          f"{enc['n_chains']} chains, {n_p} params, k_obs={k_obs}")

    ws = warm_start if warm_start is not None else option_c_clean
    result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=ws)
    print(f"  Loss: {result['init_loss']:.4f} -> {result['loss']:.4f}"
          f"  (data={result['data_loss']:.4f}, reg={result['reg_loss']:.4f})")
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


def run_lambda_sweep(matched_clean, option_c_clean, cross_pool=False):
    """Sweep lambda_delta with warm-start annealing (descending lambda).

    Each fit warm-starts from the previous result, so the sweep is
    effectively a continuation path from high to low regularization.
    """
    print(f"\n{'='*70}")
    cp_tag = " [cross-pool]" if cross_pool else ""
    print(f"Lambda_delta sweep{cp_tag}")
    print(f"{'='*70}")
    print(f"{'lambda':>10} {'loss':>10} {'data_loss':>10} {'reg_loss':>10} "
          f"{'delta_norm':>12} {'mean_|d|':>10}")
    print("-" * 65)

    results = []
    warm_start = option_c_clean
    for lam in LAMBDA_DELTAS:
        result, model, jdata, enc = run_token_factored(
            matched_clean, option_c_clean, lambda_delta=lam,
            cross_pool=cross_pool, warm_start=warm_start,
        )
        delta = result["noise_deltas"]
        delta_norm = float(np.linalg.norm(delta))
        mean_abs_d = float(np.mean(np.abs(delta)))
        print(f"{lam:>10.2f} {result['loss']:>10.4f} "
              f"{result['data_loss']:>10.4f} {result['reg_loss']:>10.4f} "
              f"{delta_norm:>12.4f} {mean_abs_d:>10.4f}")
        results.append({
            "lambda_delta": lam,
            "loss": result["loss"],
            "data_loss": result["data_loss"],
            "reg_loss": result["reg_loss"],
            "delta_norm": delta_norm,
            "mean_abs_delta": mean_abs_d,
            "converged": result["converged"],
        })
        # Warm-start next iteration from this result
        warm_start = _result_to_warm_start(result)

    return results


# ---- Phase 3: LOO Cross-Validation ----


def run_loo_validation(
    matched_clean, option_c_clean, lambda_delta=1.0, cross_pool=False,
):
    """Leave-one-pool-out cross-validation via predict_new_pool."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import (
        FixedHead, PerPoolHead, TokenFactoredNoiseHead,
    )
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.joint_fit import prepare_token_factored_data
    from quantammsim.calibration.pool_data import (
        K_OBS_CROSS, K_OBS_REDUCED, build_cross_pool_x_obs,
        build_x_obs, _parse_tokens,
    )
    import jax.numpy as jnp

    k_obs = K_OBS_CROSS if cross_pool else K_OBS_REDUCED
    pool_ids = sorted(matched_clean.keys())

    cp_tag = " [cross-pool]" if cross_pool else ""
    print(f"\n{'='*70}")
    print(f"LOO Cross-Validation (lambda_delta={lambda_delta}){cp_tag}")
    print(f"{'='*70}")

    loo_results = []
    for hold_out_pid in pool_ids:
        # Build training set without hold-out pool
        train_matched = {p: matched_clean[p] for p in pool_ids if p != hold_out_pid}
        train_oc = {p: option_c_clean[p] for p in pool_ids if p != hold_out_pid}

        if len(train_matched) < 3:
            continue

        # Fit on training set
        jdata, enc = prepare_token_factored_data(
            train_matched, cross_pool=cross_pool,
        )
        gas_values = _build_gas_values(jdata, train_matched)

        noise_head = TokenFactoredNoiseHead(
            k_obs=k_obs,
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
        y_obs_ho = ho_panel["log_volume"].values.astype(float)

        if cross_pool:
            # Build cross-pool x_obs for held-out pool.
            # Use matched_clean so pool's own entry is accessible;
            # build_cross_pool_x_obs auto-excludes pool_id from its own peers.
            x_obs_ho = build_cross_pool_x_obs(
                ho_panel, matched_clean, hold_out_pid,
            )
            # Trim y_obs to match (first day dropped)
            y_obs_ho = y_obs_ho[1:]
            day_indices_ho = ho_entry["day_indices"][1:]
        else:
            x_obs_ho = build_x_obs(ho_panel, reduced=True)
            day_indices_ho = ho_entry["day_indices"]

        # Use Option C cadence for the hold-out pool (not predicting cadence)
        oc_ho = option_c_clean[hold_out_pid]
        v_arb_all = np.array(interpolate_pool_daily(
            ho_entry["coeffs"],
            jnp.float64(oc_ho["log_cadence"]),
            jnp.float64(np.exp(oc_ho["log_gas"])),
        ))
        v_arb = v_arb_all[day_indices_ho]

        # Noise coefficients are k_obs-dimensional; x_obs_ho has k_obs columns
        noise_coeffs = ho_pred["noise_coeffs"][:k_obs]
        v_noise = np.exp(x_obs_ho @ noise_coeffs)
        log_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
        ss_res = np.sum((log_pred - y_obs_ho) ** 2)
        ss_tot = np.sum((y_obs_ho - y_obs_ho.mean()) ** 2)
        r2_loo = 1 - ss_res / max(ss_tot, 1e-10)

        # Compare with Option C in-sample R²
        x_obs_c = build_x_obs(ho_panel, reduced=True)
        v_noise_c = np.exp(x_obs_c @ oc_ho["noise_coeffs"][:K_OBS_REDUCED])
        v_arb_c = v_arb_all[ho_entry["day_indices"]]
        log_pred_c = np.log(np.maximum(v_arb_c + v_noise_c, 1e-6))
        y_obs_full = ho_panel["log_volume"].values.astype(float)
        ss_res_c = np.sum((log_pred_c - y_obs_full) ** 2)
        ss_tot_c = np.sum((y_obs_full - y_obs_full.mean()) ** 2)
        r2_c = 1 - ss_res_c / max(ss_tot_c, 1e-10)

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


def plot_lambda_sweep(sweep_results, output_dir, suffix=""):
    """Plot loss (data/reg separated) and delta norm vs lambda_delta."""
    lambdas = [r["lambda_delta"] for r in sweep_results]
    data_losses = [r["data_loss"] for r in sweep_results]
    reg_losses = [r["reg_loss"] for r in sweep_results]
    delta_norms = [r["delta_norm"] for r in sweep_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogx(lambdas, data_losses, "o-", color="steelblue", label="data_loss")
    ax1.semilogx(lambdas, reg_losses, "s--", color="orangered", label="reg_loss")
    ax1.set_xlabel("lambda_delta")
    ax1.set_ylabel("Loss")
    ax1.set_title("Data + Reg Loss vs lambda_delta")
    ax1.legend()

    ax2.semilogx(lambdas, delta_norms, "o-", color="orangered")
    ax2.set_xlabel("lambda_delta")
    ax2.set_ylabel("||delta||")
    ax2.set_title("Delta norm vs lambda_delta")

    fig.tight_layout()
    out = os.path.join(output_dir, f"lambda_sweep{suffix}.png")
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


def plot_loo_scatter(loo_results, output_dir, suffix=""):
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
    ax.set_title(f"LOO: Token-Factored vs Option C{suffix}")

    n_better = sum(1 for c, l in zip(r2_c, r2_loo) if l > c)
    ax.text(0.05, 0.95, f"LOO wins: {n_better}/{len(r2_c)}",
            transform=ax.transAxes, fontsize=10, va="top")

    fig.tight_layout()
    out = os.path.join(output_dir, f"loo_scatter{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- Main ----


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Token-Factored Noise Calibration v2")
    print("  Canonicalization + Cross-Pool Lag Features")
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

    # ---- Ablation: Baseline (K_OBS_REDUCED=4, no cross-pool) ----
    print("\n" + "=" * 70)
    print("ABLATION 1: Baseline (K_OBS_REDUCED=4, no cross-pool features)")
    print("=" * 70)

    result_base, model_base, jdata_base, enc_base = run_token_factored(
        matched_clean, option_c_clean, lambda_delta=1.0, cross_pool=False)

    # Analysis
    print_token_effects(result_base, enc_base)
    print_chain_effects(result_base, enc_base)
    print_delta_analysis(result_base, enc_base, jdata_base, matched_clean)

    # Lambda sweep with annealing
    sweep_baseline = run_lambda_sweep(matched_clean, option_c_clean, cross_pool=False)

    # LOO
    loo_baseline = run_loo_validation(
        matched_clean, option_c_clean, lambda_delta=1.0, cross_pool=False)

    # ---- Ablation: Cross-pool (K_OBS_CROSS=7) ----
    print("\n" + "=" * 70)
    print("ABLATION 2: Cross-pool lag features (K_OBS_CROSS=7)")
    print("=" * 70)

    result_cross, model_cross, jdata_cross, enc_cross = run_token_factored(
        matched_clean, option_c_clean, lambda_delta=1.0, cross_pool=True)

    print_token_effects(result_cross, enc_cross)
    print_chain_effects(result_cross, enc_cross)
    print_delta_analysis(result_cross, enc_cross, jdata_cross, matched_clean)

    sweep_cross = run_lambda_sweep(matched_clean, option_c_clean, cross_pool=True)

    loo_cross = run_loo_validation(
        matched_clean, option_c_clean, lambda_delta=1.0, cross_pool=True)

    # ---- Ablation summary ----
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    for label, loo in [("Baseline (k=4)", loo_baseline), ("Cross-pool (k=7)", loo_cross)]:
        if loo:
            r2s = [r["r2_loo"] for r in loo]
            r2s_c = [r["r2_option_c"] for r in loo]
            wins = sum(1 for r in loo if r["r2_loo"] > r["r2_option_c"])
            print(f"  {label}: median R²_LOO={np.median(r2s):.4f}, "
                  f"median R²_C={np.median(r2s_c):.4f}, "
                  f"wins={wins}/{len(loo)}")

    # Plots
    print("\nGenerating plots...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_lambda_sweep(sweep_baseline, OUTPUT_DIR, suffix="_baseline")
    plot_lambda_sweep(sweep_cross, OUTPUT_DIR, suffix="_crosspool")
    plot_token_effects(result_base, enc_base, OUTPUT_DIR)
    plot_loo_scatter(loo_baseline, OUTPUT_DIR, suffix="_baseline")
    plot_loo_scatter(loo_cross, OUTPUT_DIR, suffix="_crosspool")

    # JSON export
    export = {
        "phase0_diagnostic": diag,
        "baseline": {
            "loss": result_base["loss"],
            "data_loss": result_base["data_loss"],
            "reg_loss": result_base["reg_loss"],
            "init_loss": result_base["init_loss"],
            "converged": result_base["converged"],
            "n_pools": result_base["n_pools"],
            "n_tokens": enc_base["n_tokens"],
            "n_chains": enc_base["n_chains"],
            "token_index": enc_base["token_index"],
            "chain_index": enc_base["chain_index"],
            "token_effects": result_base["token_effects"].tolist(),
            "Gamma": result_base["Gamma"].tolist(),
            "chain_effects": result_base["chain_effects"].tolist(),
            "beta_fee": result_base["beta_fee"].tolist(),
            "noise_deltas": result_base["noise_deltas"].tolist(),
            "noise_coeffs": result_base["noise_coeffs"].tolist(),
        },
        "cross_pool": {
            "loss": result_cross["loss"],
            "data_loss": result_cross["data_loss"],
            "reg_loss": result_cross["reg_loss"],
            "init_loss": result_cross["init_loss"],
            "converged": result_cross["converged"],
            "n_pools": result_cross["n_pools"],
            "n_tokens": enc_cross["n_tokens"],
            "n_chains": enc_cross["n_chains"],
            "token_index": enc_cross["token_index"],
            "chain_index": enc_cross["chain_index"],
            "token_effects": result_cross["token_effects"].tolist(),
            "Gamma": result_cross["Gamma"].tolist(),
            "chain_effects": result_cross["chain_effects"].tolist(),
            "beta_fee": result_cross["beta_fee"].tolist(),
            "noise_deltas": result_cross["noise_deltas"].tolist(),
            "noise_coeffs": result_cross["noise_coeffs"].tolist(),
        },
        "lambda_sweep_baseline": sweep_baseline,
        "lambda_sweep_crosspool": sweep_cross,
        "loo_baseline": loo_baseline,
        "loo_crosspool": loo_cross,
    }
    json_path = os.path.join(OUTPUT_DIR, "token_factored_v2_results.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"  Saved: {json_path}")

    print(f"\n{'='*70}")
    print(f"Done. Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
