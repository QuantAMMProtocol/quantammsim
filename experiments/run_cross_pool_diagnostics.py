"""Diagnostic experiments for cross-pool noise calibration.

Runs cheap experiments to bound the value of learned cross-pool aggregation:
1. Lambda_token sweep — is the LOO failure due to overfitting token effects?
2. Leave-one-in — how much pool-specific data closes the gap?
3. Naive AR baseline — is the model barely beating lag-1?
4. Pool connectivity — which pools are predictable at all?

Uses cached stage1 data from run_token_factored_calibration.py.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)
JOINT_MAXITER = 3000  # reduced for sweep speed


def load_stage1():
    """Load cached stage 1 (matched_clean + option_c_clean)."""
    path = os.path.join(CACHE_DIR, "stage1.pkl")
    if not os.path.exists(path):
        print("ERROR: no stage1 cache. Run run_token_factored_calibration.py first.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['matched_clean'])} pools from cache")
    return data["matched_clean"], data["option_c_clean"]


# ---- Diagnostic 1: Lambda_token sweep ----


def run_lambda_token_sweep(matched_clean, option_c_clean):
    """LOO with varying lambda_token to test whether overfitting is the problem."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import (
        FixedHead, PerPoolHead, TokenFactoredNoiseHead,
    )
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.joint_fit import prepare_token_factored_data
    from quantammsim.calibration.loss import CHAIN_GAS_USD
    from quantammsim.calibration.pool_data import K_OBS_REDUCED, build_x_obs, _parse_tokens
    import jax.numpy as jnp

    pool_ids = sorted(matched_clean.keys())
    lambda_tokens = [0.1, 0.5, 1.0, 5.0, 10.0]

    print("\n" + "=" * 70)
    print("Diagnostic 1: Lambda_token sweep (LOO)")
    print("=" * 70)
    print(f"  lambda_delta=1.0 fixed, sweeping lambda_token")
    print(f"  maxiter={JOINT_MAXITER}")

    all_results = {}

    for lt in lambda_tokens:
        print(f"\n--- lambda_token={lt} ---")
        loo_r2s = []

        for hold_out_pid in pool_ids:
            train_matched = {p: matched_clean[p] for p in pool_ids if p != hold_out_pid}
            train_oc = {p: option_c_clean[p] for p in pool_ids if p != hold_out_pid}

            if len(train_matched) < 3:
                continue

            jdata, enc = prepare_token_factored_data(train_matched)

            gas_values = []
            for pid in jdata.pool_ids:
                chain = train_matched[pid]["chain"]
                gas_values.append(np.log(max(CHAIN_GAS_USD.get(chain, 1.0), 1e-6)))

            noise_head = TokenFactoredNoiseHead(
                k_obs=K_OBS_REDUCED,
                lambda_delta=1.0,
                lambda_token=lt,
                **enc,
            )
            model = CalibrationModel(
                PerPoolHead("log_cadence", default=np.log(12.0)),
                FixedHead("log_gas", np.array(gas_values)),
                noise_head,
            )
            result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=train_oc)

            # Predict for held-out pool
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

            # Evaluate
            ho_panel = ho_entry["panel"]
            x_obs_ho = build_x_obs(ho_panel, reduced=True)
            y_obs_ho = ho_panel["log_volume"].values.astype(float)

            oc_ho = option_c_clean[hold_out_pid]
            v_arb_all = np.array(interpolate_pool_daily(
                ho_entry["coeffs"],
                jnp.float64(oc_ho["log_cadence"]),
                jnp.float64(np.exp(oc_ho["log_gas"])),
            ))
            v_arb = v_arb_all[ho_entry["day_indices"]]
            v_noise = np.exp(x_obs_ho @ ho_pred["noise_coeffs"][:K_OBS_REDUCED])
            log_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
            ss_res = np.sum((log_pred - y_obs_ho) ** 2)
            ss_tot = np.sum((y_obs_ho - y_obs_ho.mean()) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            loo_r2s.append(r2)

            tag = "OK" if r2 > 0 else "NEG"
            print(f"    {hold_out_pid[:16]} R²={r2:.3f} [{tag}]")

        median_r2 = np.median(loo_r2s)
        wins = sum(1 for r2, pid in zip(loo_r2s, pool_ids)
                   if r2 > option_c_clean[pid].get("r2", 0))
        all_results[lt] = {
            "median_r2": median_r2,
            "mean_r2": np.mean(loo_r2s),
            "r2s": loo_r2s,
            "n_negative": sum(1 for r in loo_r2s if r < 0),
        }
        print(f"  lambda_token={lt}: median R²={median_r2:.4f}, "
              f"mean={np.mean(loo_r2s):.4f}, "
              f"n_negative={sum(1 for r in loo_r2s if r < 0)}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'lambda_token':>12} {'median_R²':>10} {'mean_R²':>10} {'n_neg':>6}")
    print("-" * 42)
    for lt in lambda_tokens:
        r = all_results[lt]
        print(f"{lt:>12.1f} {r['median_r2']:>10.4f} {r['mean_r2']:>10.4f} "
              f"{r['n_negative']:>6}")

    return all_results


# ---- Diagnostic 2: Leave-one-in ----


def run_leave_one_in(matched_clean, option_c_clean, n_days_in=30):
    """LOO but give held-out pool n_days_in days of data for adaptation."""
    from quantammsim.calibration.calibration_model import CalibrationModel
    from quantammsim.calibration.heads import (
        FixedHead, PerPoolHead, TokenFactoredNoiseHead,
    )
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.joint_fit import prepare_token_factored_data
    from quantammsim.calibration.loss import CHAIN_GAS_USD
    from quantammsim.calibration.pool_data import K_OBS_REDUCED, build_x_obs, _parse_tokens
    import jax.numpy as jnp

    pool_ids = sorted(matched_clean.keys())

    print("\n" + "=" * 70)
    print(f"Diagnostic 2: Leave-one-in ({n_days_in} days of held-out data)")
    print("=" * 70)

    results = []

    for hold_out_pid in pool_ids:
        ho_entry = matched_clean[hold_out_pid]
        ho_panel = ho_entry["panel"]
        n_obs = len(ho_panel)

        if n_obs <= n_days_in + 10:
            print(f"  {hold_out_pid[:16]} — too few obs ({n_obs}), skipping")
            continue

        # Split: first n_days_in for training, rest for evaluation
        train_panel = ho_panel.iloc[:n_days_in].copy()
        eval_panel = ho_panel.iloc[n_days_in:].copy()
        train_day_indices = ho_entry["day_indices"][:n_days_in]
        eval_day_indices = ho_entry["day_indices"][n_days_in:]

        # Build training matched: all other pools + truncated held-out pool
        train_matched = {}
        for p in pool_ids:
            if p != hold_out_pid:
                train_matched[p] = matched_clean[p]

        # Add truncated held-out pool
        ho_train_entry = dict(ho_entry)
        ho_train_entry["panel"] = train_panel.reset_index(drop=True)
        ho_train_entry["day_indices"] = train_day_indices
        train_matched[hold_out_pid] = ho_train_entry

        train_oc = dict(option_c_clean)  # all pools including held-out

        # Fit with held-out pool included (gets its own delta from 30 days)
        jdata, enc = prepare_token_factored_data(train_matched)

        gas_values = []
        for pid in jdata.pool_ids:
            chain = train_matched[pid]["chain"]
            gas_values.append(np.log(max(CHAIN_GAS_USD.get(chain, 1.0), 1e-6)))

        noise_head = TokenFactoredNoiseHead(
            k_obs=K_OBS_REDUCED,
            lambda_delta=1.0,
            lambda_token=0.1,
            **enc,
        )
        model = CalibrationModel(
            PerPoolHead("log_cadence", default=np.log(12.0)),
            FixedHead("log_gas", np.array(gas_values)),
            noise_head,
        )
        result = model.fit(jdata, maxiter=JOINT_MAXITER, warm_start=train_oc)

        # Find held-out pool's index in training set and extract noise_coeffs
        ho_idx = jdata.pool_ids.index(hold_out_pid)
        noise_coeffs = result["noise_coeffs"][ho_idx]

        # Evaluate on held-out days
        x_obs_eval = build_x_obs(eval_panel, reduced=True)
        y_obs_eval = eval_panel["log_volume"].values.astype(float)

        oc_ho = option_c_clean[hold_out_pid]
        v_arb_all = np.array(interpolate_pool_daily(
            ho_entry["coeffs"],
            jnp.float64(oc_ho["log_cadence"]),
            jnp.float64(np.exp(oc_ho["log_gas"])),
        ))
        v_arb = v_arb_all[eval_day_indices]
        v_noise = np.exp(x_obs_eval @ noise_coeffs[:K_OBS_REDUCED])
        log_pred = np.log(np.maximum(v_arb + v_noise, 1e-6))
        ss_res = np.sum((log_pred - y_obs_eval) ** 2)
        ss_tot = np.sum((y_obs_eval - y_obs_eval.mean()) ** 2)
        r2_in = 1 - ss_res / max(ss_tot, 1e-10)

        # Also compute Option C R² on eval days for comparison
        v_noise_c = np.exp(x_obs_eval @ oc_ho["noise_coeffs"][:K_OBS_REDUCED])
        log_pred_c = np.log(np.maximum(v_arb + v_noise_c, 1e-6))
        ss_res_c = np.sum((log_pred_c - y_obs_eval) ** 2)
        r2_c_eval = 1 - ss_res_c / max(ss_tot, 1e-10)

        results.append({
            "pool_id": hold_out_pid,
            "r2_leave_one_in": r2_in,
            "r2_option_c_eval": r2_c_eval,
            "n_train_days": n_days_in,
            "n_eval_days": len(eval_panel),
            "tokens": ho_entry["tokens"],
        })

        print(f"  {hold_out_pid[:16]} ({ho_entry['tokens']:<14}) "
              f"R²_in={r2_in:.3f}  R²_C_eval={r2_c_eval:.3f}  "
              f"n_eval={len(eval_panel)}")

    if results:
        r2s_in = [r["r2_leave_one_in"] for r in results]
        r2s_c = [r["r2_option_c_eval"] for r in results]
        print(f"\n  Leave-one-in ({n_days_in}d): median R²={np.median(r2s_in):.4f}")
        print(f"  Option C (eval days):       median R²={np.median(r2s_c):.4f}")
        print(f"  Recall: zero-shot LOO:      median R²=0.362")

    return results


# ---- Diagnostic 3: Naive AR baseline ----


def run_naive_ar_baseline(matched_clean):
    """Compute R² of vol_tomorrow = vol_today (no model, no cross-pool)."""
    print("\n" + "=" * 70)
    print("Diagnostic 3: Naive autoregressive baseline (lag-1 copy)")
    print("=" * 70)

    pool_r2s = []
    for pid in sorted(matched_clean.keys()):
        panel = matched_clean[pid]["panel"]
        y = panel["log_volume"].values.astype(float)

        if len(y) < 3:
            continue

        # Predict day t from day t-1
        y_true = y[1:]
        y_pred = y[:-1]

        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        pool_r2s.append(r2)

        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"R²_AR1={r2:.3f}  n_obs={len(y)}")

    print(f"\n  Naive AR1: median R²={np.median(pool_r2s):.4f}, "
          f"mean={np.mean(pool_r2s):.4f}")
    print(f"  Recall: zero-shot LOO = 0.362, Option C in-sample = 0.589")

    return pool_r2s


# ---- Diagnostic 4: Pool connectivity analysis ----


def run_connectivity_analysis(matched_clean, option_c_clean):
    """Analyze token overlap and partition LOO R² by connectivity."""
    from quantammsim.calibration.pool_data import _parse_tokens, _canonicalize_token

    print("\n" + "=" * 70)
    print("Diagnostic 4: Pool connectivity analysis")
    print("=" * 70)

    pool_ids = sorted(matched_clean.keys())

    # Build canonical token sets per pool
    pool_tokens = {}
    for pid in pool_ids:
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        canon = {_canonicalize_token(t) for t in toks[:2]}
        pool_tokens[pid] = canon

    # Count: for each pool, how many other pools share at least 1 token?
    # And how many share both tokens?
    print(f"\n{'Pool':<18} {'Tokens':<16} {'1+ shared':>10} {'2 shared':>10} "
          f"{'R²_C':>8}")
    print("-" * 66)

    connectivity = []
    for pid in pool_ids:
        my_toks = pool_tokens[pid]
        n_one_shared = 0
        n_both_shared = 0
        for other in pool_ids:
            if other == pid:
                continue
            overlap = len(my_toks & pool_tokens[other])
            if overlap >= 1:
                n_one_shared += 1
            if overlap >= 2:
                n_both_shared += 1

        oc = option_c_clean[pid]
        r2_c = 1 - oc["loss"] / max(
            np.var(matched_clean[pid]["panel"]["log_volume"].values) *
            len(matched_clean[pid]["panel"]) /
            max(len(matched_clean[pid]["panel"]) - 1, 1),
            1e-10,
        )

        connectivity.append({
            "pool_id": pid,
            "tokens": matched_clean[pid]["tokens"],
            "n_one_shared": n_one_shared,
            "n_both_shared": n_both_shared,
        })

        print(f"  {pid[:16]} {matched_clean[pid]['tokens']:<16} "
              f"{n_one_shared:>10} {n_both_shared:>10}")

    # Partition: well-connected (1+ shared ≥ 3) vs isolated
    well_connected = [c for c in connectivity if c["n_one_shared"] >= 3]
    isolated = [c for c in connectivity if c["n_one_shared"] < 3]

    print(f"\n  Well-connected (≥3 pools share a token): {len(well_connected)}")
    print(f"  Isolated (<3 pools share a token):        {len(isolated)}")

    if isolated:
        print(f"\n  Isolated pools:")
        for c in isolated:
            print(f"    {c['pool_id'][:16]} {c['tokens']}")

    return connectivity


# ---- Main ----


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Cross-Pool Calibration Diagnostics")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()

    # Run all diagnostics
    ar_results = run_naive_ar_baseline(matched_clean)
    connectivity = run_connectivity_analysis(matched_clean, option_c_clean)
    leave_one_in = run_leave_one_in(matched_clean, option_c_clean, n_days_in=30)
    lambda_sweep = run_lambda_token_sweep(matched_clean, option_c_clean)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Naive AR1 baseline:       median R² = {np.median(ar_results):.4f}")
    if leave_one_in:
        r2s_in = [r["r2_leave_one_in"] for r in leave_one_in]
        print(f"  Leave-one-in (30 days):   median R² = {np.median(r2s_in):.4f}")
    print(f"  Zero-shot LOO (current):  median R² = 0.362")
    print(f"  Option C in-sample:       median R² = 0.589")
    print(f"\n  Lambda_token sweep:")
    for lt, r in sorted(lambda_sweep.items()):
        print(f"    lambda_token={lt:>5.1f}: median R² = {r['median_r2']:.4f} "
              f"(n_neg={r['n_negative']})")


if __name__ == "__main__":
    main()
