"""Cross-pool linear prediction of NOISE residuals.

Decomposes total volume into V_arb (from grid + Option C cadence/gas)
and noise residual, then tests whether peer pools' lagged noise
residuals predict this pool's noise residual.

1. Ridge in-sample: noise_resid_i_t = W_i @ noise_resid_{-i, t-1} [+ own_lag]
2. Ridge LOO with overlap transfer
3. Ridge LOO with 30-day burn-in
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
    print(f"Loaded {len(data['matched_clean'])} pools from cache")
    return data["matched_clean"], data["option_c_clean"]


def build_noise_residual_matrix(matched_clean, option_c_clean):
    """Build (n_dates, n_pools) noise residual matrix.

    noise_resid_i_t = log_volume_i_t - log(V_arb_i_t)

    V_arb computed from grid interpolation at Option C cadence/gas.
    Returns residual matrix (NaN where missing), date list, pool ids.
    """
    import jax.numpy as jnp
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)

    # Collect all dates
    all_dates = set()
    for pid in pool_ids:
        panel = matched_clean[pid]["panel"]
        all_dates.update(panel["date"].values)
    date_list = sorted(all_dates)
    n_dates = len(date_list)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    # Build matrices
    vol_matrix = np.full((n_dates, n_pools), np.nan)
    resid_matrix = np.full((n_dates, n_pools), np.nan)

    for j, pid in enumerate(pool_ids):
        entry = matched_clean[pid]
        oc = option_c_clean[pid]
        panel = entry["panel"]
        coeffs = entry["coeffs"]
        day_indices = entry["day_indices"]

        # Compute V_arb from grid
        v_arb_all = np.array(interpolate_pool_daily(
            coeffs,
            jnp.float64(oc["log_cadence"]),
            jnp.float64(np.exp(oc["log_gas"])),
        ))
        v_arb = v_arb_all[day_indices]
        log_v_arb = np.log(np.maximum(v_arb, 1e-6))

        # Fill matrices
        dates = panel["date"].values
        log_vols = panel["log_volume"].values.astype(float)

        for k, date in enumerate(dates):
            t = date_to_idx[date]
            vol_matrix[t, j] = log_vols[k]
            resid_matrix[t, j] = log_vols[k] - log_v_arb[k]

    print(f"  Built noise residual matrix: {n_dates} dates x {n_pools} pools")
    print(f"  Residual stats: mean={np.nanmean(resid_matrix):.3f}, "
          f"std={np.nanstd(resid_matrix):.3f}")

    return vol_matrix, resid_matrix, date_list, pool_ids


def build_token_overlap(matched_clean, pool_ids):
    from quantammsim.calibration.pool_data import _parse_tokens, _canonicalize_token
    n = len(pool_ids)
    overlap = np.zeros((n, n), dtype=np.int32)
    pool_tokens = {}
    for i, pid in enumerate(pool_ids):
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        pool_tokens[i] = {_canonicalize_token(t) for t in toks[:2]}
    for i in range(n):
        for j in range(n):
            overlap[i, j] = len(pool_tokens[i] & pool_tokens[j])
    return overlap


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-10)


# ---- In-sample ridge on noise residuals ----


def run_ridge_insample(resid_matrix, pool_ids, matched_clean):
    """Per-pool ridge: predict noise_resid from peers' lagged residuals."""
    n_dates, n_pools = resid_matrix.shape

    for include_own in [False, True]:
        tag = "peers + own_lag" if include_own else "peers only"
        print(f"\n{'='*70}")
        print(f"Ridge in-sample on noise residuals ({tag})")
        print(f"{'='*70}")

        pool_r2s = []
        for i, pid in enumerate(pool_ids):
            X_lag = resid_matrix[:-1, :]
            y_cur = resid_matrix[1:, i]
            own_lag = X_lag[:, i]

            valid = ~np.isnan(y_cur)
            if include_own:
                valid = valid & ~np.isnan(own_lag)

            X_others = np.delete(X_lag, i, axis=1)
            X_filled = X_others.copy()
            for c in range(X_filled.shape[1]):
                col = X_filled[:, c]
                m = np.nanmean(col)
                col[np.isnan(col)] = m if np.isfinite(m) else 0.0
                X_filled[:, c] = col

            if include_own:
                X_full = np.column_stack([X_filled, own_lag[:, None]])
            else:
                X_full = X_filled

            X_i = X_full[valid]
            y_i = y_cur[valid]

            if len(y_i) < 10:
                pool_r2s.append(np.nan)
                continue

            model = RidgeCV(alphas=np.logspace(-2, 4, 50))
            model.fit(X_i, y_i)
            y_pred = model.predict(X_i)
            r2 = r2_score(y_i, y_pred)
            pool_r2s.append(r2)

            print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
                  f"R²={r2:.3f}  n={len(y_i)}  alpha={model.alpha_:.1f}")

        valid_r2s = [r for r in pool_r2s if not np.isnan(r)]
        print(f"\n  In-sample ridge ({tag}): median R²={np.median(valid_r2s):.4f}, "
              f"mean={np.mean(valid_r2s):.4f}")

    return pool_r2s


def run_ar1_noise_baseline(resid_matrix, pool_ids, matched_clean):
    """Naive AR1 on noise residuals: resid_tomorrow = resid_today."""
    print(f"\n{'='*70}")
    print("AR1 baseline on noise residuals")
    print(f"{'='*70}")

    pool_r2s = []
    for i, pid in enumerate(pool_ids):
        y = resid_matrix[:, i]
        valid = ~np.isnan(y[:-1]) & ~np.isnan(y[1:])
        y_true = y[1:][valid]
        y_pred = y[:-1][valid]

        if len(y_true) < 3:
            pool_r2s.append(np.nan)
            continue

        r2 = r2_score(y_true, y_pred)
        pool_r2s.append(r2)
        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"R²={r2:.3f}  n={len(y_true)}")

    valid_r2s = [r for r in pool_r2s if not np.isnan(r)]
    print(f"\n  AR1 noise residual: median R²={np.median(valid_r2s):.4f}, "
          f"mean={np.mean(valid_r2s):.4f}")
    return pool_r2s


# ---- LOO with overlap transfer ----


def run_ridge_loo(resid_matrix, pool_ids, matched_clean):
    """LOO on noise residuals with token-overlap weight transfer."""
    print(f"\n{'='*70}")
    print("Ridge LOO on noise residuals (overlap transfer, peers + own_lag)")
    print(f"{'='*70}")

    n_dates, n_pools = resid_matrix.shape
    overlap = build_token_overlap(matched_clean, pool_ids)

    pool_r2s = []
    for i, pid in enumerate(pool_ids):
        train_idx = [j for j in range(n_pools) if j != i]

        # Fit ridge for each training pool
        train_weights = {}
        for k in train_idx:
            pred_idx = [j for j in range(n_pools) if j != k]
            X_lag = resid_matrix[:-1, pred_idx].copy()
            own_lag_k = resid_matrix[:-1, k]
            y_k = resid_matrix[1:, k]

            valid = ~np.isnan(y_k) & ~np.isnan(own_lag_k)
            for c in range(X_lag.shape[1]):
                col = X_lag[:, c]
                m = np.nanmean(col)
                col[np.isnan(col)] = m if np.isfinite(m) else 0.0
                X_lag[:, c] = col

            X_k = np.column_stack([X_lag[valid], own_lag_k[valid, None]])
            y_k = y_k[valid]

            if len(y_k) < 10:
                continue

            model = RidgeCV(alphas=np.logspace(-2, 4, 50))
            model.fit(X_k, y_k)

            # Store weights mapped to pool indices
            w = np.zeros(n_pools + 1)  # +1 for own_lag
            for widx, pidx in enumerate(pred_idx):
                w[pidx] = model.coef_[widx]
            w[-1] = model.coef_[-1]  # own_lag weight
            train_weights[k] = (w, model.intercept_)

        if not train_weights:
            pool_r2s.append(np.nan)
            continue

        # Transfer: overlap-weighted average of training pool weight vectors
        w_transfer = np.zeros(n_pools + 1)
        b_transfer = 0.0
        total_sim = 0.0
        for k in train_weights:
            sim = max(overlap[i, k], 0.1)
            w_k, b_k = train_weights[k]
            w_transfer += sim * w_k
            b_transfer += sim * b_k
            total_sim += sim
        w_transfer /= total_sim
        b_transfer /= total_sim
        w_transfer[i] = 0.0  # no self-prediction from peers

        # Predict held-out pool
        X_lag_all = resid_matrix[:-1, :].copy()
        own_lag_i = resid_matrix[:-1, i]
        y_true = resid_matrix[1:, i]
        valid = ~np.isnan(y_true) & ~np.isnan(own_lag_i)

        for c in range(X_lag_all.shape[1]):
            col = X_lag_all[:, c]
            m = np.nanmean(col)
            col[np.isnan(col)] = m if np.isfinite(m) else 0.0
            X_lag_all[:, c] = col

        X_i = np.column_stack([X_lag_all[valid], own_lag_i[valid, None]])
        y_pred = X_i @ w_transfer + b_transfer
        y_true = y_true[valid]

        r2 = r2_score(y_true, y_pred)
        pool_r2s.append(r2)
        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"R²={r2:.3f}  n={len(y_true)}")

    valid_r2s = [r for r in pool_r2s if not np.isnan(r)]
    print(f"\n  LOO ridge (overlap transfer): median R²={np.median(valid_r2s):.4f}, "
          f"mean={np.mean(valid_r2s):.4f}")
    return pool_r2s


# ---- LOO with burn-in ----


def run_ridge_burnin(resid_matrix, pool_ids, matched_clean, n_burnin=30):
    """LOO with burn-in: learn pool i's weights from first n_burnin days."""
    print(f"\n{'='*70}")
    print(f"Ridge LOO on noise residuals ({n_burnin}d burn-in, peers + own_lag)")
    print(f"{'='*70}")

    n_dates, n_pools = resid_matrix.shape

    pool_r2s = []
    for i, pid in enumerate(pool_ids):
        y_all = resid_matrix[:, i]
        own_lag_all = np.full(n_dates, np.nan)
        own_lag_all[1:] = y_all[:-1]

        valid_days = ~np.isnan(y_all) & ~np.isnan(own_lag_all)
        valid_indices = np.where(valid_days)[0]

        if len(valid_indices) < n_burnin + 10:
            pool_r2s.append(np.nan)
            continue

        burn_idx = valid_indices[:n_burnin]
        eval_idx = valid_indices[n_burnin:]

        pred_idx = [j for j in range(n_pools) if j != i]

        def build_X(indices):
            X_peers = resid_matrix[indices - 1][:, pred_idx].copy()
            for c in range(X_peers.shape[1]):
                col = X_peers[:, c]
                m = np.nanmean(col)
                col[np.isnan(col)] = m if np.isfinite(m) else 0.0
                X_peers[:, c] = col
            own = y_all[indices - 1]
            return np.column_stack([X_peers, own[:, None]])

        X_burn = build_X(burn_idx)
        y_burn = y_all[burn_idx]

        model = RidgeCV(alphas=np.logspace(-2, 4, 50))
        model.fit(X_burn, y_burn)

        X_eval = build_X(eval_idx)
        y_eval = y_all[eval_idx]
        y_pred = model.predict(X_eval)

        r2 = r2_score(y_eval, y_pred)
        pool_r2s.append(r2)
        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"R²={r2:.3f}  n_eval={len(eval_idx)}  alpha={model.alpha_:.1f}")

    valid_r2s = [r for r in pool_r2s if not np.isnan(r)]
    print(f"\n  LOO ridge ({n_burnin}d burn-in): "
          f"median R²={np.median(valid_r2s):.4f}, mean={np.mean(valid_r2s):.4f}")
    return pool_r2s


# ---- Main ----


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Cross-Pool Linear Prediction of Noise Residuals")
    print("  (total volume - grid arb volume)")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()
    vol_matrix, resid_matrix, date_list, pool_ids = build_noise_residual_matrix(
        matched_clean, option_c_clean)

    ar1_r2s = run_ar1_noise_baseline(resid_matrix, pool_ids, matched_clean)
    insample_r2s = run_ridge_insample(resid_matrix, pool_ids, matched_clean)
    loo_r2s = run_ridge_loo(resid_matrix, pool_ids, matched_clean)
    burnin_r2s = run_ridge_burnin(resid_matrix, pool_ids, matched_clean, n_burnin=30)

    def safe_median(xs):
        v = [x for x in xs if x is not None and not np.isnan(x)]
        return np.median(v) if v else float("nan")

    print("\n" + "=" * 70)
    print("SUMMARY (noise residuals)")
    print("=" * 70)
    print(f"  AR1 noise residual:         median R² = {safe_median(ar1_r2s):.4f}")
    print(f"  Ridge in-sample (+own):     median R² = {safe_median(insample_r2s):.4f}")
    print(f"  Ridge LOO (overlap xfer):   median R² = {safe_median(loo_r2s):.4f}")
    print(f"  Ridge LOO (30d burn-in):    median R² = {safe_median(burnin_r2s):.4f}")
    print(f"  ---")
    print(f"  (total vol) AR1:            median R² = 0.397")
    print(f"  (total vol) Ridge +own:     median R² = 0.599")
    print(f"  Option C in-sample:         median R² = 0.589")


if __name__ == "__main__":
    main()
