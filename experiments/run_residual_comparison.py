"""Apples-to-apples R² comparison on noise residuals.

Target for all methods: r_it = log(V_total_it) - log(V_arb_it)

Methods:
  1. Option C: log(1 + exp(x_obs @ noise_coeffs) / V_arb)
  2. AR1 on residuals: r_{i, t-1}
  3. Ridge on residuals (peers only, in-sample)
  4. Ridge on residuals (peers + own lag, in-sample)
  5. Constant zero (predict r=0, i.e. V_total = V_arb)
"""

import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import RidgeCV

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)


def load_stage1():
    path = os.path.join(CACHE_DIR, "stage1.pkl")
    if not os.path.exists(path):
        print("ERROR: no stage1 cache.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["matched_clean"], data["option_c_clean"]


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-10)


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax.numpy as jnp
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import (
        K_OBS_REDUCED, build_x_obs, _parse_tokens, _canonicalize_token,
    )

    print("=" * 70)
    print("Apples-to-Apples: All methods on noise residual target")
    print("  target = log(V_total) - log(V_arb)")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()
    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)

    # ---- Build aligned data per pool ----
    # For each pool: residual, Option C prediction of residual, dates
    pool_data = {}
    all_dates = set()

    for pid in pool_ids:
        entry = matched_clean[pid]
        oc = option_c_clean[pid]
        panel = entry["panel"]

        # V_arb
        v_arb_all = np.array(interpolate_pool_daily(
            entry["coeffs"],
            jnp.float64(oc["log_cadence"]),
            jnp.float64(np.exp(oc["log_gas"])),
        ))
        v_arb = v_arb_all[entry["day_indices"]]
        log_v_arb = np.log(np.maximum(v_arb, 1e-6))

        # Observed
        log_vol = panel["log_volume"].values.astype(float)
        dates = panel["date"].values

        # Noise residual target
        resid = log_vol - log_v_arb

        # Option C noise prediction (in residual space)
        x_obs = build_x_obs(panel, reduced=True)
        noise_coeffs = oc["noise_coeffs"][:K_OBS_REDUCED]
        v_noise_oc = np.exp(x_obs @ noise_coeffs)
        resid_pred_oc = np.log(np.maximum(1.0 + v_noise_oc / np.maximum(v_arb, 1e-6), 1e-10))

        pool_data[pid] = {
            "dates": dates,
            "resid": resid,
            "resid_pred_oc": resid_pred_oc,
            "log_vol": log_vol,
            "v_arb": v_arb,
        }
        all_dates.update(dates)

    # ---- Build residual matrix for cross-pool methods ----
    date_list = sorted(all_dates)
    n_dates = len(date_list)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    resid_matrix = np.full((n_dates, n_pools), np.nan)
    for j, pid in enumerate(pool_ids):
        pd = pool_data[pid]
        for k, date in enumerate(pd["dates"]):
            resid_matrix[date_to_idx[date], j] = pd["resid"][k]

    # ---- Token overlap for peer identification ----
    pool_tokens = {}
    for i, pid in enumerate(pool_ids):
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        pool_tokens[i] = {_canonicalize_token(t) for t in toks[:2]}

    # ---- Compute R² for each method, per pool ----
    results = {m: [] for m in [
        "option_c", "ar1", "ridge_peers", "ridge_peers_own",
        "constant_zero", "peer_mean",
    ]}

    print(f"\n{'Pool':<18} {'Tokens':<14} {'OptC':>7} {'AR1':>7} "
          f"{'R_peer':>7} {'R_p+own':>7} {'zero':>7} {'pmean':>7} {'n':>5}")
    print("-" * 90)

    for i, pid in enumerate(pool_ids):
        pd = pool_data[pid]
        resid = pd["resid"]
        n_obs = len(resid)

        # --- Option C ---
        r2_oc = r2_score(resid, pd["resid_pred_oc"])

        # --- Constant zero (V_total = V_arb) ---
        r2_zero = r2_score(resid, np.zeros_like(resid))

        # --- AR1 on residuals ---
        if n_obs >= 3:
            r2_ar1 = r2_score(resid[1:], resid[:-1])
        else:
            r2_ar1 = np.nan

        # --- Ridge peers only (in-sample) ---
        X_lag = resid_matrix[:-1, :]
        y_cur = resid_matrix[1:, i]
        own_lag = X_lag[:, i]
        valid = ~np.isnan(y_cur)

        X_others = np.delete(X_lag, i, axis=1)
        X_filled = X_others.copy()
        for c in range(X_filled.shape[1]):
            col = X_filled[:, c]
            m = np.nanmean(col)
            col[np.isnan(col)] = m if np.isfinite(m) else 0.0
            X_filled[:, c] = col

        X_peers = X_filled[valid]
        y_i = y_cur[valid]

        if len(y_i) >= 10:
            model_p = RidgeCV(alphas=np.logspace(-2, 4, 50))
            model_p.fit(X_peers, y_i)
            r2_rp = r2_score(y_i, model_p.predict(X_peers))
        else:
            r2_rp = np.nan

        # --- Ridge peers + own lag (in-sample) ---
        valid_own = valid & ~np.isnan(own_lag)
        X_both = np.column_stack([X_filled, own_lag[:, None]])
        X_both_v = X_both[valid_own]
        y_both = y_cur[valid_own]

        if len(y_both) >= 10:
            model_po = RidgeCV(alphas=np.logspace(-2, 4, 50))
            model_po.fit(X_both_v, y_both)
            r2_rpo = r2_score(y_both, model_po.predict(X_both_v))
        else:
            r2_rpo = np.nan

        # --- Peer mean (zero parameter) ---
        peers = [j for j in range(n_pools) if j != i
                 and len(pool_tokens[i] & pool_tokens[j]) >= 1]
        if peers:
            peer_lag = resid_matrix[:-1, :][:, peers]
            peer_mean = np.nanmean(peer_lag, axis=1)
            y_pm = y_cur[valid]
            pm_pred = peer_mean[valid]
            pm_valid = ~np.isnan(pm_pred)
            if pm_valid.sum() >= 3:
                r2_pm = r2_score(y_pm[pm_valid], pm_pred[pm_valid])
            else:
                r2_pm = np.nan
        else:
            r2_pm = np.nan

        results["option_c"].append(r2_oc)
        results["ar1"].append(r2_ar1)
        results["ridge_peers"].append(r2_rp)
        results["ridge_peers_own"].append(r2_rpo)
        results["constant_zero"].append(r2_zero)
        results["peer_mean"].append(r2_pm)

        tokens = matched_clean[pid]["tokens"]
        print(f"  {pid[:16]} {tokens:<14} {r2_oc:>7.3f} {r2_ar1:>7.3f} "
              f"{r2_rp:>7.3f} {r2_rpo:>7.3f} {r2_zero:>7.3f} "
              f"{r2_pm:>7.3f} {n_obs:>5}")

    # ---- Summary ----
    def safe_median(xs):
        v = [x for x in xs if np.isfinite(x)]
        return np.median(v) if v else float("nan")

    print(f"\n{'='*70}")
    print("SUMMARY — all on noise residual target")
    print(f"{'='*70}")
    for name, label in [
        ("option_c", "Option C (per-pool fitted)"),
        ("ar1", "AR1 on residuals"),
        ("ridge_peers", "Ridge peers only (in-sample)"),
        ("ridge_peers_own", "Ridge peers + own lag (in-sample)"),
        ("peer_mean", "Peer mean (0 params)"),
        ("constant_zero", "Constant zero (V_total=V_arb)"),
    ]:
        vals = results[name]
        med = safe_median(vals)
        mean = np.nanmean([x for x in vals if np.isfinite(x)])
        n_neg = sum(1 for x in vals if np.isfinite(x) and x < 0)
        print(f"  {label:<35} median R² = {med:>7.4f}  "
              f"mean = {mean:>7.4f}  n_neg = {n_neg}")


if __name__ == "__main__":
    main()
