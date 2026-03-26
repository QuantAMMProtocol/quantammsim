"""Cross-pool linear volume prediction baselines.

1. Ridge cross-pool regression: log_vol_i_t = W_i @ log_vol_{-i, t-1}
   - In-sample: fit full 36x36 W, evaluate on training data
   - LOO: hold out pool i, fit W on 35 pools, predict pool i using
     token-overlap-weighted average of learned rows (transfer via similarity)
   - LOO with burn-in: use 30 days of pool i to learn its row of W directly

2. Zero-parameter peer-mean: predicted_vol_i_t = mean(log_vol_{j,t-1})
   for peers sharing a canonical token with pool i
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
        print("ERROR: no stage1 cache. Run run_token_factored_calibration.py first.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['matched_clean'])} pools from cache")
    return data["matched_clean"], data["option_c_clean"]


def build_volume_matrix(matched_clean):
    """Build (n_dates, n_pools) aligned volume matrix.

    Returns vol_matrix, date_list, pool_ids.
    Dates are the intersection of all pools' date ranges.
    Missing values filled with NaN.
    """
    pool_ids = sorted(matched_clean.keys())

    # Collect all (pool, date) -> log_volume
    pool_date_vol = {}
    all_dates = set()
    for pid in pool_ids:
        panel = matched_clean[pid]["panel"]
        dates = panel["date"].values
        vols = panel["log_volume"].values.astype(float)
        pool_date_vol[pid] = dict(zip(dates, vols))
        all_dates.update(dates)

    date_list = sorted(all_dates)
    n_dates = len(date_list)
    n_pools = len(pool_ids)

    vol_matrix = np.full((n_dates, n_pools), np.nan)
    for j, pid in enumerate(pool_ids):
        dv = pool_date_vol[pid]
        for t, date in enumerate(date_list):
            if date in dv:
                vol_matrix[t, j] = dv[date]

    return vol_matrix, date_list, pool_ids


def build_token_overlap(matched_clean, pool_ids):
    """Build (n_pools, n_pools) token overlap matrix (0, 1, or 2)."""
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


# ---- 1. Ridge cross-pool regression ----


def run_ridge_cross_pool(matched_clean):
    """Full cross-pool ridge: predict each pool from all others' lag-1."""
    vol_matrix, date_list, pool_ids = build_volume_matrix(matched_clean)
    n_dates, n_pools = vol_matrix.shape

    # Check if fully-observed rows exist
    X_lag = vol_matrix[:-1, :]
    Y_cur = vol_matrix[1:, :]
    valid = ~np.any(np.isnan(X_lag), axis=1) & ~np.any(np.isnan(Y_cur), axis=1)
    n_valid = int(valid.sum())

    # Peers only
    print("\n" + "=" * 70)
    print("1a. Ridge cross-pool regression (in-sample, peers only)")
    print("=" * 70)
    print(f"  {n_pools} pools, {n_valid} fully-observed day pairs "
          f"(of {n_dates-1} total)")
    print("  Using per-pool valid rows with NaN imputation.")
    r2_peers, _, _, _ = _run_ridge_per_pool_valid(
        vol_matrix, pool_ids, matched_clean, include_own_lag=False)

    # Peers + own lag
    print("\n" + "=" * 70)
    print("1a+. Ridge cross-pool regression (in-sample, peers + own lag)")
    print("=" * 70)
    r2_both, _, _, _ = _run_ridge_per_pool_valid(
        vol_matrix, pool_ids, matched_clean, include_own_lag=True)

    return r2_peers, r2_both, vol_matrix, date_list, pool_ids


def _run_ridge_per_pool_valid(vol_matrix, pool_ids, matched_clean,
                              include_own_lag=False):
    """Fallback: per-pool ridge using only rows where pool i AND predictors have data."""
    n_dates, n_pools = vol_matrix.shape
    tag = " + own_lag" if include_own_lag else ""

    pool_r2s = []
    for i, pid in enumerate(pool_ids):
        X_lag = vol_matrix[:-1, :]
        y_cur = vol_matrix[1:, i]
        own_lag = X_lag[:, i]  # pool i's own lag

        # Valid: pool i has data today AND own lag exists (if used)
        valid_y = ~np.isnan(y_cur)
        if include_own_lag:
            valid_y = valid_y & ~np.isnan(own_lag)

        X_others = np.delete(X_lag, i, axis=1)

        # For each predictor, fill NaN with that predictor's mean (simple imputation)
        X_filled = X_others.copy()
        for j in range(X_filled.shape[1]):
            col = X_filled[:, j]
            col_mean = np.nanmean(col)
            col[np.isnan(col)] = col_mean
            X_filled[:, j] = col

        if include_own_lag:
            X_full = np.column_stack([X_filled, own_lag[:, None]])
        else:
            X_full = X_filled

        X_i = X_full[valid_y]
        y_i = y_cur[valid_y]

        if len(y_i) < 10:
            pool_r2s.append(np.nan)
            continue

        model = RidgeCV(alphas=np.logspace(-2, 4, 50))
        model.fit(X_i, y_i)
        y_pred = model.predict(X_i)
        r2 = r2_score(y_i, y_pred)
        pool_r2s.append(r2)

        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"R²={r2:.3f}  n_obs={len(y_i)}  alpha={model.alpha_:.1f}")

    valid_r2s = [r for r in pool_r2s if not np.isnan(r)]
    print(f"\n  In-sample ridge{tag}: median R²={np.median(valid_r2s):.4f}, "
          f"mean={np.mean(valid_r2s):.4f}")

    return pool_r2s, vol_matrix, None, pool_ids


def run_ridge_loo(matched_clean):
    """LOO cross-pool ridge with token-overlap transfer."""
    print("\n" + "=" * 70)
    print("1b. Ridge cross-pool LOO (transfer via token overlap)")
    print("=" * 70)

    vol_matrix, date_list, pool_ids = build_volume_matrix(matched_clean)
    n_dates, n_pools = vol_matrix.shape
    overlap = build_token_overlap(matched_clean, pool_ids)

    pool_r2s = []
    for i, pid in enumerate(pool_ids):
        # Training pools: all except i
        train_idx = [j for j in range(n_pools) if j != i]
        n_train = len(train_idx)

        # Build training data: for each training pool k, predict from others' lag
        # Use per-pool valid rows with NaN imputation
        X_lag_all = vol_matrix[:-1, :]
        Y_cur_all = vol_matrix[1:, :]

        # Fit a ridge model for each training pool
        train_models = {}
        train_weights = {}  # weight vectors (excluding self)
        for k_pos, k in enumerate(train_idx):
            # Predictors: all pools except k (including pool i's historical data!)
            pred_idx = [j for j in range(n_pools) if j != k]
            X_k = X_lag_all[:, pred_idx].copy()
            y_k = Y_cur_all[:, k]

            valid = ~np.isnan(y_k)
            for c in range(X_k.shape[1]):
                col = X_k[:, c]
                m = np.nanmean(col)
                col[np.isnan(col)] = m if np.isfinite(m) else 0.0
                X_k[:, c] = col

            X_k = X_k[valid]
            y_k = y_k[valid]

            if len(y_k) < 10:
                continue

            model = RidgeCV(alphas=np.logspace(-2, 4, 50))
            model.fit(X_k, y_k)
            train_models[k] = model

            # Store full weight vector (n_pools-1,) with mapping to pool indices
            w = np.zeros(n_pools)
            for widx, pidx in enumerate(pred_idx):
                w[pidx] = model.coef_[widx]
            w_intercept = model.intercept_
            train_weights[k] = (w, w_intercept)

        if not train_weights:
            pool_r2s.append(np.nan)
            continue

        # Transfer to held-out pool i: weighted average of training pools' weight vectors
        # Weight by token overlap with pool i
        w_transfer = np.zeros(n_pools)
        intercept_transfer = 0.0
        total_sim = 0.0
        for k in train_weights:
            sim = overlap[i, k]
            if sim == 0:
                sim = 0.1  # small weight for unrelated pools
            w_k, b_k = train_weights[k]
            w_transfer += sim * w_k
            intercept_transfer += sim * b_k
            total_sim += sim

        w_transfer /= total_sim
        intercept_transfer /= total_sim

        # Zero out pool i's own weight (shouldn't predict from self)
        w_transfer[i] = 0.0

        # Predict pool i
        X_lag_i = vol_matrix[:-1, :].copy()
        y_true_i = vol_matrix[1:, i]
        valid = ~np.isnan(y_true_i)

        # Impute NaN predictors
        for c in range(X_lag_i.shape[1]):
            col = X_lag_i[:, c]
            m = np.nanmean(col)
            col[np.isnan(col)] = m if np.isfinite(m) else 0.0
            X_lag_i[:, c] = col

        y_pred_i = X_lag_i[valid] @ w_transfer + intercept_transfer
        y_true_i = y_true_i[valid]

        r2 = r2_score(y_true_i, y_pred_i)
        pool_r2s.append(r2)

        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"R²={r2:.3f}  n_eval={len(y_true_i)}")

    valid_r2s = [r for r in pool_r2s if not np.isnan(r)]
    print(f"\n  LOO ridge (overlap transfer): median R²={np.median(valid_r2s):.4f}, "
          f"mean={np.mean(valid_r2s):.4f}")
    print(f"  Recall: AR1={0.397:.3f}, zero-shot token-factored={0.362:.3f}")

    return pool_r2s


def run_ridge_loo_burnin(matched_clean, n_burnin=30):
    """LOO with burn-in: learn pool i's weight row from n_burnin days."""
    print("\n" + "=" * 70)
    print(f"1c. Ridge cross-pool LOO with {n_burnin}-day burn-in")
    print("=" * 70)

    vol_matrix, date_list, pool_ids = build_volume_matrix(matched_clean)
    n_dates, n_pools = vol_matrix.shape

    pool_r2s = []
    for i, pid in enumerate(pool_ids):
        # Pool i's data
        y_all = vol_matrix[:, i]
        valid_days = ~np.isnan(y_all)
        valid_indices = np.where(valid_days)[0]

        if len(valid_indices) < n_burnin + 10:
            print(f"  {pid[:16]} — too few obs, skipping")
            pool_r2s.append(np.nan)
            continue

        # Split: first n_burnin valid days for training, rest for eval
        burn_indices = valid_indices[:n_burnin]
        eval_indices = valid_indices[n_burnin:]

        # Training: predict pool i from all others' lag using burn-in days
        # Need (day, day-1) pairs where day is in burn_indices and day >= 1
        burn_pairs = burn_indices[burn_indices >= 1]

        pred_idx = [j for j in range(n_pools) if j != i]
        X_burn = vol_matrix[burn_pairs - 1][:, pred_idx].copy()
        y_burn = vol_matrix[burn_pairs, i]

        # Impute NaN
        for c in range(X_burn.shape[1]):
            col = X_burn[:, c]
            m = np.nanmean(col)
            col[np.isnan(col)] = m if np.isfinite(m) else 0.0
            X_burn[:, c] = col

        model = RidgeCV(alphas=np.logspace(-2, 4, 50))
        model.fit(X_burn, y_burn)

        # Evaluate on remaining days
        eval_pairs = eval_indices[eval_indices >= 1]
        X_eval = vol_matrix[eval_pairs - 1][:, pred_idx].copy()
        y_eval = vol_matrix[eval_pairs, i]

        for c in range(X_eval.shape[1]):
            col = X_eval[:, c]
            m = np.nanmean(col)
            col[np.isnan(col)] = m if np.isfinite(m) else 0.0
            X_eval[:, c] = col

        y_pred = model.predict(X_eval)
        r2 = r2_score(y_eval, y_pred)
        pool_r2s.append(r2)

        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"R²={r2:.3f}  n_burn={len(burn_pairs)}  n_eval={len(eval_pairs)}  "
              f"alpha={model.alpha_:.1f}")

    valid_r2s = [r for r in pool_r2s if not np.isnan(r)]
    print(f"\n  LOO ridge ({n_burnin}d burn-in): "
          f"median R²={np.median(valid_r2s):.4f}, "
          f"mean={np.mean(valid_r2s):.4f}")

    return pool_r2s


# ---- 2. Zero-parameter peer-mean ----


def run_peer_mean_baseline(matched_clean):
    """Predict pool i's volume as mean of token-peer lagged volumes."""
    from quantammsim.calibration.pool_data import _parse_tokens, _canonicalize_token

    print("\n" + "=" * 70)
    print("2. Zero-parameter peer-mean baseline")
    print("=" * 70)

    vol_matrix, date_list, pool_ids = build_volume_matrix(matched_clean)
    n_dates, n_pools = vol_matrix.shape
    overlap = build_token_overlap(matched_clean, pool_ids)

    pool_r2s = []
    for i, pid in enumerate(pool_ids):
        # Peers: pools sharing at least 1 token
        peers = [j for j in range(n_pools) if j != i and overlap[i, j] >= 1]

        if not peers:
            pool_r2s.append(np.nan)
            continue

        y_true = vol_matrix[1:, i]
        valid = ~np.isnan(y_true)

        # Peer mean at t-1
        peer_lag = vol_matrix[:-1, :][:, peers]
        peer_mean = np.nanmean(peer_lag, axis=1)

        y_pred = peer_mean[valid]
        y_true = y_true[valid]

        # Remove any remaining NaN
        both_valid = ~np.isnan(y_pred)
        y_pred = y_pred[both_valid]
        y_true = y_true[both_valid]

        r2 = r2_score(y_true, y_pred)
        pool_r2s.append(r2)

        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"R²={r2:.3f}  n_peers={len(peers)}  n_obs={len(y_true)}")

    valid_r2s = [r for r in pool_r2s if not np.isnan(r)]
    print(f"\n  Peer-mean baseline: median R²={np.median(valid_r2s):.4f}, "
          f"mean={np.mean(valid_r2s):.4f}")
    print(f"  Recall: AR1={0.397:.3f}, zero-shot token-factored={0.362:.3f}")

    return pool_r2s


# ---- Main ----


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Cross-Pool Linear Volume Prediction Baselines")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()

    # 1a. In-sample ridge (peers only + peers + own lag)
    r2_peers, r2_both, vol_matrix, date_list, pool_ids = run_ridge_cross_pool(matched_clean)

    # 1b. LOO ridge with token-overlap transfer
    loo_r2s = run_ridge_loo(matched_clean)

    # 1c. LOO ridge with 30-day burn-in
    burnin_r2s = run_ridge_loo_burnin(matched_clean, n_burnin=30)

    # 2. Zero-parameter peer mean
    peer_r2s = run_peer_mean_baseline(matched_clean)

    # Summary
    def safe_median(xs):
        v = [x for x in xs if not np.isnan(x)]
        return np.median(v) if v else float("nan")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Ridge in-sample (peers):   median R² = {safe_median(r2_peers):.4f}")
    print(f"  Ridge in-sample (+own):    median R² = {safe_median(r2_both):.4f}")
    print(f"  Ridge LOO (overlap xfer):  median R² = {safe_median(loo_r2s):.4f}")
    print(f"  Ridge LOO (30d burn-in):   median R² = {safe_median(burnin_r2s):.4f}")
    print(f"  Peer-mean (0 params):      median R² = {safe_median(peer_r2s):.4f}")
    print(f"  ---")
    print(f"  Naive AR1:                 median R² = 0.397")
    print(f"  Token-factored LOO:        median R² = 0.362")
    print(f"  Option C in-sample:        median R² = 0.589")


if __name__ == "__main__":
    main()
