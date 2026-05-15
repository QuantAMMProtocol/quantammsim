"""DeepSets cross-pool volume prediction.

Architecture:
  For pool i at day t:
    For each peer j != i with valid data at t-1:
      h_j = Encoder(attr_j, attr_i, vol_j_{t-1}, overlap_ij)
    peer_summary = masked_mean(h_j)
    pred_i_t = Decoder(peer_summary, attr_i, own_vol_{t-1})

Evaluation:
  1. In-sample R² (all data)
  2. Temporal split (70/30)
  3. LOO (hold out one pool, retrain, evaluate)
"""

import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "token_factored_calibration", "_cache",
)
HIDDEN = 8
D_EMBED = 4
LR = 1e-3
N_EPOCHS = 500
N_EPOCHS_LOO = 200
L2_ALPHA = 0.001


def load_stage1():
    path = os.path.join(CACHE_DIR, "stage1.pkl")
    if not os.path.exists(path):
        print("ERROR: no stage1 cache.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["matched_clean"], data["option_c_clean"]


def build_volume_matrix(matched_clean):
    """Build (n_dates, n_pools) volume matrix. NaN where missing."""
    pool_ids = sorted(matched_clean.keys())
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


def build_data(matched_clean, exclude_pool_idx=None):
    """Build all arrays for DeepSets training.

    If exclude_pool_idx is set, that pool is excluded from training
    samples but kept as a peer (its volume data is still available).
    """
    from quantammsim.calibration.pool_data import (
        build_pool_attributes, _parse_tokens, _canonicalize_token,
    )

    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)

    vol_matrix, date_list, _ = build_volume_matrix(matched_clean)
    X_attr, attr_names, _ = build_pool_attributes(matched_clean)

    # Standardize attributes
    attr_mean = np.mean(X_attr, axis=0)
    attr_std = np.std(X_attr, axis=0)
    attr_std[attr_std < 1e-6] = 1.0
    X_attr_norm = ((X_attr - attr_mean) / attr_std).astype(np.float32)

    # Standardize volumes
    vol_mean = float(np.nanmean(vol_matrix))
    vol_std = float(np.nanstd(vol_matrix))
    vol_norm = ((vol_matrix - vol_mean) / vol_std).astype(np.float32)

    # Token overlap
    k_attr = X_attr_norm.shape[1]
    pool_tokens = {}
    for i, pid in enumerate(pool_ids):
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        pool_tokens[i] = {_canonicalize_token(t) for t in toks[:2]}

    # Per-pool peer structures
    n_peers = n_pools - 1
    peer_attrs = np.zeros((n_pools, n_peers, k_attr), dtype=np.float32)
    peer_overlap = np.zeros((n_pools, n_peers), dtype=np.float32)
    peer_col_idx = np.zeros((n_pools, n_peers), dtype=np.int32)

    for i in range(n_pools):
        peers = [j for j in range(n_pools) if j != i]
        for p, j in enumerate(peers):
            peer_attrs[i, p] = X_attr_norm[j]
            peer_overlap[i, p] = len(pool_tokens[i] & pool_tokens[j])
            peer_col_idx[i, p] = j

    target_attrs = X_attr_norm

    # Build samples
    sample_pools, sample_days = [], []
    for i in range(n_pools):
        if i == exclude_pool_idx:
            continue
        for t in range(1, len(date_list)):
            if np.isnan(vol_matrix[t, i]) or np.isnan(vol_matrix[t - 1, i]):
                continue
            sample_pools.append(i)
            sample_days.append(t)

    sample_pools = np.array(sample_pools, dtype=np.int32)
    sample_days = np.array(sample_days, dtype=np.int32)
    n_samples = len(sample_pools)

    # Vectorized: gather peer volumes and masks
    peer_vols = np.zeros((n_samples, n_peers), dtype=np.float32)
    peer_mask = np.zeros((n_samples, n_peers), dtype=np.float32)
    own_lag = np.zeros(n_samples, dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for s in range(n_samples):
        i = sample_pools[s]
        t = sample_days[s]
        cols = peer_col_idx[i]
        pvols = vol_norm[t - 1, cols]
        valid = ~np.isnan(pvols)
        peer_vols[s] = np.where(valid, pvols, 0.0)
        peer_mask[s] = valid.astype(np.float32)
        own_lag[s] = vol_norm[t - 1, i]
        y[s] = vol_norm[t, i]

    return {
        "peer_attrs": jnp.array(peer_attrs),
        "target_attrs": jnp.array(target_attrs),
        "peer_overlap": jnp.array(peer_overlap),
        "peer_vols": jnp.array(peer_vols),
        "peer_mask": jnp.array(peer_mask),
        "own_lag": jnp.array(own_lag),
        "y": jnp.array(y),
        "pool_idx": jnp.array(sample_pools),
        "day_idx": sample_days,
        "n_pools": n_pools,
        "n_peers": n_peers,
        "k_attr": k_attr,
        "pool_ids": pool_ids,
        "vol_mean": vol_mean,
        "vol_std": vol_std,
    }


# ---- Model ----


def init_params(key, k_attr, hidden=HIDDEN, d=D_EMBED):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    enc_in = 2 * k_attr + 2  # peer_attr + target_attr + peer_vol + overlap
    dec_in = d + k_attr + 1   # summary + target_attr + own_lag
    return {
        "enc_W1": jax.random.normal(k1, (enc_in, hidden)) * np.sqrt(2.0 / enc_in),
        "enc_b1": jnp.zeros(hidden),
        "enc_W2": jax.random.normal(k2, (hidden, d)) * np.sqrt(2.0 / hidden),
        "enc_b2": jnp.zeros(d),
        "dec_W1": jax.random.normal(k3, (dec_in, hidden)) * np.sqrt(2.0 / dec_in),
        "dec_b1": jnp.zeros(hidden),
        "dec_W2": jax.random.normal(k4, (hidden, 1)) * 0.01,
        "dec_b2": jnp.zeros(1),
    }


def forward(params, peer_attrs_all, target_attrs_all, peer_overlap_all,
            peer_vols, peer_mask, own_lag, pool_idx):
    """Batched DeepSets forward pass."""
    batch = peer_vols.shape[0]
    n_peers = peer_vols.shape[1]

    pa = peer_attrs_all[pool_idx]       # (batch, n_peers, k_attr)
    ta = target_attrs_all[pool_idx]     # (batch, k_attr)
    ov = peer_overlap_all[pool_idx]     # (batch, n_peers)

    ta_broad = jnp.broadcast_to(ta[:, None, :], pa.shape)

    enc_in = jnp.concatenate([
        pa, ta_broad,
        peer_vols[:, :, None],
        ov[:, :, None],
    ], axis=-1)

    # Encoder MLP
    flat = enc_in.reshape(-1, enc_in.shape[-1])
    h = jnp.maximum(flat @ params["enc_W1"] + params["enc_b1"], 0.0)
    h = h @ params["enc_W2"] + params["enc_b2"]
    h = h.reshape(batch, n_peers, -1)

    # Masked mean
    h_masked = h * peer_mask[:, :, None]
    n_valid = jnp.maximum(jnp.sum(peer_mask, axis=1, keepdims=True), 1.0)
    summary = jnp.sum(h_masked, axis=1) / n_valid

    # Decoder MLP
    dec_in = jnp.concatenate([summary, ta, own_lag[:, None]], axis=-1)
    h_dec = jnp.maximum(dec_in @ params["dec_W1"] + params["dec_b1"], 0.0)
    return (h_dec @ params["dec_W2"] + params["dec_b2"])[:, 0]


def loss_fn(params, static, peer_vols, peer_mask, own_lag, pool_idx, y, alpha):
    pred = forward(params, static["peer_attrs"], static["target_attrs"],
                   static["peer_overlap"], peer_vols, peer_mask, own_lag, pool_idx)
    mse = jnp.mean((pred - y) ** 2)
    reg = sum(jnp.sum(v ** 2) for k, v in params.items() if "W" in k)
    return mse + alpha * reg


grad_fn = jax.jit(jax.value_and_grad(loss_fn))


# ---- Training ----


def train(params, data, n_epochs=N_EPOCHS, lr=LR, alpha=L2_ALPHA, verbose=True):
    """Full-batch Adam training."""
    static = {
        "peer_attrs": data["peer_attrs"],
        "target_attrs": data["target_attrs"],
        "peer_overlap": data["peer_overlap"],
    }

    # Adam state
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}

    for epoch in range(n_epochs):
        loss_val, grads = grad_fn(
            params, static, data["peer_vols"], data["peer_mask"],
            data["own_lag"], data["pool_idx"], data["y"], alpha,
        )

        # Adam update
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            m_hat = m[k] / (1.0 - 0.9 ** (epoch + 1))
            v_hat = v[k] / (1.0 - 0.999 ** (epoch + 1))
            params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:4d}  loss={float(loss_val):.6f}")

    return params


# ---- Evaluation ----


def per_pool_r2(params, data):
    """Compute per-pool R² from trained model."""
    static = {
        "peer_attrs": data["peer_attrs"],
        "target_attrs": data["target_attrs"],
        "peer_overlap": data["peer_overlap"],
    }
    pred = np.array(forward(
        params, static["peer_attrs"], static["target_attrs"],
        static["peer_overlap"], data["peer_vols"], data["peer_mask"],
        data["own_lag"], data["pool_idx"],
    ))
    y = np.array(data["y"])
    pool_idx = np.array(data["pool_idx"])

    r2s = {}
    for i in range(data["n_pools"]):
        mask = pool_idx == i
        if mask.sum() < 2:
            continue
        yi = y[mask]
        pi = pred[mask]
        ss_res = np.sum((yi - pi) ** 2)
        ss_tot = np.sum((yi - yi.mean()) ** 2)
        r2s[i] = 1 - ss_res / max(ss_tot, 1e-10)
    return r2s


# ---- Main experiments ----


def run_insample(matched_clean):
    print("\n" + "=" * 70)
    print("1. In-sample DeepSets")
    print("=" * 70)

    data = build_data(matched_clean)
    n_params = sum(v.size for v in init_params(jax.random.PRNGKey(0), data["k_attr"]).values())
    print(f"  {data['peer_vols'].shape[0]} samples, {data['n_pools']} pools, "
          f"{data['k_attr']} attrs, {n_params} params")

    params = init_params(jax.random.PRNGKey(42), data["k_attr"])
    t0 = time.time()
    params = train(params, data)
    print(f"  Training: {time.time() - t0:.1f}s")

    r2s = per_pool_r2(params, data)
    pool_ids = data["pool_ids"]
    for i, pid in enumerate(pool_ids):
        if i in r2s:
            print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) R²={r2s[i]:.3f}")

    vals = list(r2s.values())
    print(f"\n  In-sample: median R²={np.median(vals):.4f}, mean={np.mean(vals):.4f}")
    return params, data, r2s


def run_temporal_split(matched_clean, split_frac=0.7):
    print("\n" + "=" * 70)
    print(f"2. Temporal split ({int(split_frac*100)}/{int((1-split_frac)*100)})")
    print("=" * 70)

    data_all = build_data(matched_clean)
    day_idx = np.array(data_all["day_idx"])
    max_day = day_idx.max()
    split_day = int(max_day * split_frac)

    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day

    def subset(data, mask):
        jmask = jnp.array(mask)
        return {
            **{k: data[k] for k in ["peer_attrs", "target_attrs", "peer_overlap",
                                      "n_pools", "n_peers", "k_attr", "pool_ids",
                                      "vol_mean", "vol_std"]},
            "peer_vols": data["peer_vols"][jmask],
            "peer_mask": data["peer_mask"][jmask],
            "own_lag": data["own_lag"][jmask],
            "y": data["y"][jmask],
            "pool_idx": data["pool_idx"][jmask],
            "day_idx": data_all["day_idx"][mask],
        }

    train_data = subset(data_all, train_mask)
    eval_data = subset(data_all, eval_mask)

    print(f"  Train: {int(train_mask.sum())} samples, Eval: {int(eval_mask.sum())} samples")

    params = init_params(jax.random.PRNGKey(42), data_all["k_attr"])
    params = train(params, train_data)

    r2s_train = per_pool_r2(params, train_data)
    r2s_eval = per_pool_r2(params, eval_data)

    pool_ids = data_all["pool_ids"]
    for i, pid in enumerate(pool_ids):
        r_tr = r2s_train.get(i, float("nan"))
        r_ev = r2s_eval.get(i, float("nan"))
        print(f"  {pid[:16]} ({matched_clean[pid]['tokens']:<14}) "
              f"train={r_tr:.3f}  eval={r_ev:.3f}")

    vals_eval = [v for v in r2s_eval.values() if np.isfinite(v)]
    print(f"\n  Temporal eval: median R²={np.median(vals_eval):.4f}, "
          f"mean={np.mean(vals_eval):.4f}")
    return r2s_eval


def run_loo(matched_clean):
    print("\n" + "=" * 70)
    print("3. LOO DeepSets")
    print("=" * 70)

    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)
    loo_r2s = []

    for hold_out_idx in range(n_pools):
        hold_out_pid = pool_ids[hold_out_idx]

        # Build training data excluding held-out pool's samples
        # (but keeping its volume data for peers)
        train_data = build_data(matched_clean, exclude_pool_idx=hold_out_idx)

        params = init_params(jax.random.PRNGKey(42), train_data["k_attr"])
        params = train(params, train_data, n_epochs=N_EPOCHS_LOO, verbose=False)

        # Build eval data: only held-out pool's samples
        eval_data = build_data(matched_clean)
        ho_mask = np.array(eval_data["pool_idx"]) == hold_out_idx
        if ho_mask.sum() < 2:
            loo_r2s.append(float("nan"))
            continue

        jmask = jnp.array(ho_mask)
        eval_sub = {
            **{k: eval_data[k] for k in ["peer_attrs", "target_attrs", "peer_overlap",
                                           "n_pools", "n_peers", "k_attr", "pool_ids",
                                           "vol_mean", "vol_std"]},
            "peer_vols": eval_data["peer_vols"][jmask],
            "peer_mask": eval_data["peer_mask"][jmask],
            "own_lag": eval_data["own_lag"][jmask],
            "y": eval_data["y"][jmask],
            "pool_idx": eval_data["pool_idx"][jmask],
            "day_idx": np.array(eval_data["day_idx"])[ho_mask],
        }

        r2s = per_pool_r2(params, eval_sub)
        r2 = r2s.get(hold_out_idx, float("nan"))
        loo_r2s.append(r2)

        tag = "OK" if r2 > 0 else "NEG"
        print(f"  {hold_out_pid[:16]} ({matched_clean[hold_out_pid]['tokens']:<14}) "
              f"R²={r2:.3f} [{tag}]")

    valid = [r for r in loo_r2s if np.isfinite(r)]
    print(f"\n  LOO DeepSets: median R²={np.median(valid):.4f}, "
          f"mean={np.mean(valid):.4f}, "
          f"n_neg={sum(1 for r in valid if r < 0)}")
    return loo_r2s


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("DeepSets Cross-Pool Volume Prediction")
    print(f"  hidden={HIDDEN}, d={D_EMBED}, lr={LR}, "
          f"alpha={L2_ALPHA}, epochs={N_EPOCHS}")
    print("=" * 70)

    matched_clean, _ = load_stage1()

    params, data, r2_insample = run_insample(matched_clean)
    r2_temporal = run_temporal_split(matched_clean)
    r2_loo = run_loo(matched_clean)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    vals_in = list(r2_insample.values())
    vals_temp = [v for v in r2_temporal.values() if np.isfinite(v)]
    vals_loo = [r for r in r2_loo if np.isfinite(r)]
    print(f"  DeepSets in-sample:        median R² = {np.median(vals_in):.4f}")
    print(f"  DeepSets temporal (30%):   median R² = {np.median(vals_temp):.4f}")
    print(f"  DeepSets LOO:              median R² = {np.median(vals_loo):.4f}")
    print(f"  ---")
    print(f"  Ridge in-sample:           median R² = 0.441")
    print(f"  Naive AR1:                 median R² = 0.397")
    print(f"  Token-factored LOO:        median R² = 0.362")


if __name__ == "__main__":
    main()
