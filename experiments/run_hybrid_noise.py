"""Hybrid noise model: DeepSets peer encoder + linear noise model.

Architecture:
  peer_effect = DeepSets_encoder(peer_data, current_pool_attrs) → scalar
  log(V_noise) = [x_obs, market, peer_effect, peer_effect×tvl, ...] @ coeffs
  V_total = V_arb(cadence) + exp(log_v_noise)

The encoder learns how to aggregate peer information. The linear model
learns how that aggregate (plus market/pool features) drives noise volume.
Cadence is learnable per-pool via PCHIP.

Usage:
  python experiments/run_hybrid_noise.py
  python experiments/run_hybrid_noise.py --encoder-hidden 16 --epochs 2000
  python experiments/run_hybrid_noise.py --n-peer-outputs 3  # multi-dim peer effect
"""

import argparse
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


def load_stage1():
    path = os.path.join(CACHE_DIR, "stage1.pkl")
    if not os.path.exists(path):
        print("ERROR: no stage1 cache.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["matched_clean"], data["option_c_clean"]


def build_data(matched_clean, option_c_clean, trend_windows=(7, 14, 30)):
    """Build all features: x_obs, market, peer encoder inputs."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
    from quantammsim.calibration.pool_data import (
        build_x_obs, build_cross_pool_x_obs, build_pool_attributes,
        _parse_tokens, _canonicalize_token, K_OBS_CROSS,
    )
    from quantammsim.calibration.market_features import (
        build_pool_market_features, pool_market_features_to_matrix,
    )

    pool_ids = sorted(matched_clean.keys())
    n_pools = len(pool_ids)

    # Common date grid
    all_dates = set()
    for pid in pool_ids:
        all_dates.update(matched_clean[pid]["panel"]["date"].values)
    date_list = sorted(all_dates)
    n_dates = len(date_list)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    # Volume matrix and per-pool metadata
    vol_matrix = np.full((n_dates, n_pools), np.nan)
    pool_coeffs = []
    pool_gas = []
    init_log_cadences = np.zeros(n_pools, dtype=np.float32)
    common_to_grid = np.full((n_pools, n_dates), 0, dtype=np.int32)

    for j, pid in enumerate(pool_ids):
        entry = matched_clean[pid]
        oc = option_c_clean[pid]
        pool_coeffs.append(entry["coeffs"])
        pool_gas.append(jnp.float64(np.exp(oc["log_gas"])))
        init_log_cadences[j] = oc["log_cadence"]
        dates = entry["panel"]["date"].values
        log_vols = entry["panel"]["log_volume"].values.astype(float)
        for k, date in enumerate(dates):
            t = date_to_idx[date]
            vol_matrix[t, j] = log_vols[k]
            common_to_grid[j, t] = entry["day_indices"][k]

    # Pool attributes (static, normalized)
    X_attr, attr_names, _ = build_pool_attributes(matched_clean)
    attr_mean = np.mean(X_attr, axis=0)
    attr_std = np.std(X_attr, axis=0)
    attr_std[attr_std < 1e-6] = 1.0
    X_attr_norm = ((X_attr - attr_mean) / attr_std).astype(np.float32)
    k_attr = X_attr_norm.shape[1]

    # Token overlap matrix
    pool_tokens = {}
    for i, pid in enumerate(pool_ids):
        toks = _parse_tokens(matched_clean[pid]["tokens"])
        pool_tokens[i] = {_canonicalize_token(t) for t in toks[:2]}

    overlap = np.zeros((n_pools, n_pools), dtype=np.float32)
    for i in range(n_pools):
        for j in range(n_pools):
            if i != j:
                overlap[i, j] = len(pool_tokens[i] & pool_tokens[j])

    # Peer index mapping: for pool i, peers are all j != i
    n_peers = n_pools - 1
    peer_idx = np.zeros((n_pools, n_peers), dtype=np.int32)
    peer_overlap = np.zeros((n_pools, n_peers), dtype=np.float32)
    for i in range(n_pools):
        peers = [j for j in range(n_pools) if j != i]
        peer_idx[i] = peers
        peer_overlap[i] = overlap[i, peers]

    # Build samples
    sample_pools, sample_days = [], []
    for i in range(n_pools):
        for t in range(1, n_dates):
            if np.isnan(vol_matrix[t, i]) or np.isnan(vol_matrix[t - 1, i]):
                continue
            sample_pools.append(i)
            sample_days.append(t)
    sample_pools = np.array(sample_pools, dtype=np.int32)
    sample_days = np.array(sample_days, dtype=np.int32)
    n_samples = len(sample_pools)

    # ---- x_obs (cross-pool, 7 features) ----
    x_obs_grid = np.full((n_dates, n_pools, K_OBS_CROSS), np.nan)
    for j, pid in enumerate(pool_ids):
        panel = matched_clean[pid]["panel"]
        xc = build_cross_pool_x_obs(panel, matched_clean, pid)
        dates_j = panel["date"].values
        for k, date in enumerate(dates_j[1:]):
            x_obs_grid[date_to_idx[date], j] = xc[k]

    x_obs = np.zeros((n_samples, K_OBS_CROSS), dtype=np.float32)
    for s in range(n_samples):
        xval = x_obs_grid[sample_days[s], sample_pools[s]]
        if np.all(np.isfinite(xval)):
            x_obs[s] = xval

    # ---- Market features ----
    print("  Building market features...")
    pool_feat = build_pool_market_features(
        matched_clean, trend_windows=list(trend_windows))
    x_market, market_names = pool_market_features_to_matrix(
        pool_feat, matched_clean, date_to_idx, pool_ids,
        sample_pools, sample_days)
    print(f"  Market features: {len(market_names)} columns")

    # ---- Peer encoder inputs: (n_samples, n_peers, n_peer_feat) ----
    # Per peer: [peer_attrs, target_attrs, peer_vol_lag1, overlap]
    # peer_vol_lag1 is the peer's volume at t-1
    vol_mean = float(np.nanmean(vol_matrix))
    vol_std = max(float(np.nanstd(vol_matrix)), 1e-6)

    # Static peer features (per pool)
    peer_attrs = np.zeros((n_pools, n_peers, k_attr), dtype=np.float32)
    for i in range(n_pools):
        peer_attrs[i] = X_attr_norm[peer_idx[i]]

    # Per-sample peer features
    peer_vol_lag1 = np.zeros((n_samples, n_peers), dtype=np.float32)
    peer_mask = np.zeros((n_samples, n_peers), dtype=np.float32)

    for s in range(n_samples):
        i = sample_pools[s]
        t = sample_days[s]
        cols = peer_idx[i]
        pvols = vol_matrix[t - 1, cols]
        valid = ~np.isnan(pvols)
        peer_mask[s] = valid.astype(np.float32)
        peer_vol_lag1[s] = np.where(valid, (pvols - vol_mean) / vol_std, 0.0)

    # Assemble peer encoder input: (n_samples, n_peers, n_peer_feat)
    # [peer_attrs(k_attr), target_attrs(k_attr), vol_lag1(1), overlap(1)]
    target_attrs_broad = np.broadcast_to(
        X_attr_norm[sample_pools][:, None, :],
        (n_samples, n_peers, k_attr))
    peer_input = np.concatenate([
        peer_attrs[sample_pools],             # (n_samples, n_peers, k_attr)
        target_attrs_broad,                   # (n_samples, n_peers, k_attr)
        peer_vol_lag1[:, :, None],            # (n_samples, n_peers, 1)
        peer_overlap[sample_pools][:, :, None],  # (n_samples, n_peers, 1)
    ], axis=-1).astype(np.float32)
    n_peer_feat = peer_input.shape[-1]

    # Combine linear features (x_obs + market)
    x_linear = np.concatenate([x_obs, x_market], axis=1).astype(np.float32)
    linear_names = [f"xobs_{i}" for i in range(K_OBS_CROSS)] + market_names

    # Standardize linear features (except intercept)
    x_mean = np.mean(x_linear, axis=0)
    x_std_arr = np.std(x_linear, axis=0)
    x_std_arr[x_std_arr < 1e-6] = 1.0
    x_mean[0] = 0.0
    x_std_arr[0] = 1.0
    x_linear = ((x_linear - x_mean) / x_std_arr).astype(np.float32)

    # Targets and indices
    y_total = np.array([vol_matrix[sample_days[s], sample_pools[s]]
                        for s in range(n_samples)], dtype=np.float32)
    sample_grid_days = common_to_grid[sample_pools, sample_days]

    return {
        "x_linear": x_linear,         # (n_samples, n_linear_feat)
        "peer_input": peer_input,      # (n_samples, n_peers, n_peer_feat)
        "peer_mask": peer_mask,        # (n_samples, n_peers)
        "y_total": y_total,
        "pool_idx": sample_pools,
        "day_idx": sample_days,
        "sample_grid_days": sample_grid_days,
        "pool_coeffs": pool_coeffs,
        "pool_gas": pool_gas,
        "init_log_cadences": init_log_cadences,
        "n_pools": n_pools,
        "n_peers": n_peers,
        "n_linear_feat": x_linear.shape[1],
        "n_peer_feat": n_peer_feat,
        "pool_ids": pool_ids,
        "linear_names": linear_names,
    }


# ---- Model ----

_SAMPLE_KEYS = {
    "x_linear", "peer_input", "peer_mask", "y_total",
    "pool_idx", "day_idx", "sample_grid_days",
}


def _subset(d, mask):
    out = {}
    for k, v in d.items():
        if k in _SAMPLE_KEYS and isinstance(v, np.ndarray):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def init_params(key, n_peer_feat, n_linear_feat, encoder_hidden,
                n_peer_outputs, n_pools, init_log_cadences):
    """Initialize all parameters.

    Encoder: peer_input → hidden → n_peer_outputs (per peer, then mean-pooled)
    Linear: [x_linear, peer_outputs, peer_outputs × x_linear[1](tvl)] @ coeffs
    """
    k1, k2 = jax.random.split(key)

    # Encoder: single hidden layer → n_peer_outputs
    n_total_linear = n_linear_feat + n_peer_outputs + n_peer_outputs  # +interactions with tvl

    params = {
        "enc_W1": jax.random.normal(k1, (n_peer_feat, encoder_hidden)) * np.sqrt(2.0 / n_peer_feat),
        "enc_b1": jnp.zeros(encoder_hidden),
        "enc_W2": jax.random.normal(k2, (encoder_hidden, n_peer_outputs)) * 0.01,
        "enc_b2": jnp.zeros(n_peer_outputs),
        "noise_coeffs": jnp.zeros(n_total_linear),
        "log_cadence": jnp.array(init_log_cadences),
    }
    return params, n_total_linear


def forward_encoder(params, peer_input, peer_mask):
    """DeepSets encoder: per-peer MLP → masked mean → scalar(s).

    Returns (n_samples, n_peer_outputs).
    """
    batch, n_peers, _ = peer_input.shape
    flat = peer_input.reshape(-1, peer_input.shape[-1])

    h = jnp.maximum(flat @ params["enc_W1"] + params["enc_b1"], 0.0)
    h = h @ params["enc_W2"] + params["enc_b2"]
    h = h.reshape(batch, n_peers, -1)

    h_masked = h * peer_mask[:, :, None]
    n_valid = jnp.maximum(jnp.sum(peer_mask, axis=1, keepdims=True), 1.0)
    return jnp.sum(h_masked, axis=1) / n_valid  # (batch, n_peer_outputs)


def make_loss_fn(pool_coeffs, pool_gas, n_pools):
    """Loss with learnable cadence + encoder + linear noise model."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    def loss_fn(params, x_linear, peer_input, peer_mask, y_total,
                sample_grid_days, pool_idx, l2_alpha, huber_delta):

        # Encoder → peer effect scalar(s)
        peer_effect = forward_encoder(params, peer_input, peer_mask)

        # Build full linear input: [x_linear, peer_effect, peer_effect × tvl]
        # tvl is x_linear[:, 1] (xobs_1 = log_tvl_lag1, standardized)
        tvl = x_linear[:, 1:2]  # keep 2D
        peer_x_tvl = peer_effect * tvl  # interaction

        x_full = jnp.concatenate([x_linear, peer_effect, peer_x_tvl], axis=1)
        log_v_noise = x_full @ params["noise_coeffs"]

        # V_arb from PCHIP
        log_cadence = params["log_cadence"]
        n_samples = y_total.shape[0]
        v_arb = jnp.zeros(n_samples)
        for i in range(n_pools):
            v_arb_all = interpolate_pool_daily(
                pool_coeffs[i], log_cadence[i], pool_gas[i])
            safe_days = jnp.clip(sample_grid_days, 0, v_arb_all.shape[0] - 1)
            v_arb = jnp.where(pool_idx == i, v_arb_all[safe_days], v_arb)

        log_v_arb = jnp.log(jnp.maximum(v_arb, 1e-10))
        log_v_total = jnp.logaddexp(log_v_arb, log_v_noise)

        # Huber loss with per-pool weighting
        residuals = log_v_total - y_total
        abs_r = jnp.abs(residuals)
        huber_vals = jnp.where(abs_r <= huber_delta, 0.5 * residuals ** 2,
                               huber_delta * (abs_r - 0.5 * huber_delta))

        pool_counts = jnp.zeros(n_pools).at[pool_idx].add(
            jnp.ones_like(pool_idx, dtype=jnp.float32))
        active = (pool_counts > 0).astype(jnp.float32)
        n_active = jnp.maximum(jnp.sum(active), 1.0)
        pool_counts = jnp.maximum(pool_counts, 1.0)
        pool_sums = jnp.zeros(n_pools).at[pool_idx].add(huber_vals)
        data_loss = jnp.sum((pool_sums / pool_counts) * active) / n_active

        # L2 on encoder weights + noise coeffs
        reg = l2_alpha * (
            jnp.sum(params["enc_W1"] ** 2) +
            jnp.sum(params["enc_W2"] ** 2) +
            jnp.sum(params["noise_coeffs"] ** 2)
        )
        return data_loss + reg

    return jax.jit(jax.value_and_grad(loss_fn))


def train(params, data, grad_fn, n_epochs, lr, l2_alpha, huber_delta,
          verbose=True):
    m = {k: jnp.zeros_like(v) for k, v in params.items()}
    v = {k: jnp.zeros_like(v) for k, v in params.items()}

    xl = jnp.array(data["x_linear"])
    pi = jnp.array(data["peer_input"])
    pm = jnp.array(data["peer_mask"])
    yt = jnp.array(data["y_total"])
    sgd = jnp.array(data["sample_grid_days"])
    pidx = jnp.array(data["pool_idx"])

    for epoch in range(n_epochs):
        loss_val, grads = grad_fn(
            params, xl, pi, pm, yt, sgd, pidx, l2_alpha, huber_delta)
        loss_f = float(loss_val)

        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            m_hat = m[k] / (1.0 - 0.9 ** (epoch + 1))
            v_hat = v[k] / (1.0 - 0.999 ** (epoch + 1))
            params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            cads = np.exp(np.array(params["log_cadence"]))
            pe = np.array(forward_encoder(
                params, jnp.array(data["peer_input"][:100]),
                jnp.array(data["peer_mask"][:100])))
            print(f"  epoch {epoch:4d}  loss={loss_f:.6f}"
                  f"  cad=[{cads.min():.1f}-{np.median(cads):.1f}-{cads.max():.1f}]"
                  f"  peer_eff=[{pe.min():.2f},{pe.mean():.2f},{pe.max():.2f}]")

    return params


def evaluate(params, data, label=""):
    """Evaluate decomposition."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    x_linear = np.array(data["x_linear"])
    peer_input = data["peer_input"]
    peer_mask = data["peer_mask"]
    y_total = np.array(data["y_total"])
    pool_idx = np.array(data["pool_idx"])
    sgd = np.array(data["sample_grid_days"])
    log_cadence = np.array(params["log_cadence"])
    init_cads = data["init_log_cadences"]
    pool_ids = data["pool_ids"]
    n_pools = data["n_pools"]

    # Encoder
    peer_effect = np.array(forward_encoder(
        params, jnp.array(peer_input), jnp.array(peer_mask)))

    # Build full linear input
    tvl = x_linear[:, 1:2]
    peer_x_tvl = peer_effect * tvl
    x_full = np.concatenate([x_linear, peer_effect, peer_x_tvl], axis=1)

    noise_coeffs = np.array(params["noise_coeffs"])
    log_v_noise = x_full @ noise_coeffs
    v_noise = np.exp(log_v_noise)

    # V_arb
    v_arb = np.zeros(len(y_total))
    for i in range(n_pools):
        mask = pool_idx == i
        if not mask.any():
            continue
        v_arb_all = np.array(interpolate_pool_daily(
            data["pool_coeffs"][i], jnp.float64(log_cadence[i]),
            data["pool_gas"][i]))
        v_arb[mask] = v_arb_all[sgd[mask]]

    v_obs = np.exp(y_total)
    log_v_arb = np.log(np.maximum(v_arb, 1e-10))
    pred_total = np.logaddexp(log_v_arb, log_v_noise)

    if label:
        print(f"\n  {label}:")
    print(f"    {'Pool'[:16]:16s} {'R²':>6s} {'Cad':>5s} → {'learn':>5s}"
          f" {'Arb%':>6s} {'Noise%':>7s} {'PeerEff':>8s} {'Flag':>5s}")
    print(f"    {'-'*65}")

    r2s = {}
    pool_diag = []
    for i in range(n_pools):
        mask = pool_idx == i
        if mask.sum() < 2:
            continue
        yt = y_total[mask]
        pt = pred_total[mask]
        ss_res = np.sum((yt - pt) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2s[i] = 1 - ss_res / max(ss_tot, 1e-10)

        pid = pool_ids[i]
        ci = np.exp(init_cads[i])
        cl = np.exp(log_cadence[i])
        arb_pct = np.median(v_arb[mask] / v_obs[mask]) * 100
        noise_pct = np.median(v_noise[mask] / v_obs[mask]) * 100
        pe_mean = np.mean(peer_effect[mask])

        flags = []
        if arb_pct > 150:
            flags.append("A")
        if cl <= 1.01 or cl >= 59.9:
            flags.append("B")
        if r2s[i] < 0:
            flags.append("X")
        flag_str = "".join(flags)

        pool_diag.append({
            "pid": pid, "r2": r2s[i], "cad_init": ci, "cad_learned": cl,
            "arb_pct": arb_pct, "noise_pct": noise_pct,
            "peer_effect": pe_mean, "flags": flag_str,
        })

        print(f"    {pid[:16]:16s} {r2s[i]:6.3f} {ci:5.1f} → {cl:5.1f}"
              f" {arb_pct:6.0f}% {noise_pct:6.0f}% {pe_mean:+8.3f} {flag_str:>5s}")

    vals = [x for x in r2s.values() if np.isfinite(x)]
    med = np.median(vals) if vals else float("nan")
    healthy = [d for d in pool_diag if d["arb_pct"] <= 150 and d["r2"] > 0]
    med_h = np.median([d["r2"] for d in healthy]) if healthy else float("nan")
    n_path = sum(1 for d in pool_diag if d["arb_pct"] > 150)
    n_bound = sum(1 for d in pool_diag
                  if d["cad_learned"] <= 1.01 or d["cad_learned"] >= 59.9)

    # Print coefficient analysis
    nc = np.array(params["noise_coeffs"])
    n_linear = data["n_linear_feat"]
    n_po = len(nc) - n_linear
    n_each = n_po // 2

    print(f"\n    Median R²: {med:.4f} (healthy: {med_h:.4f})")
    print(f"    Healthy: {len(pool_diag) - n_path}/{len(pool_diag)},"
          f"  at bounds: {n_bound}")

    print(f"\n    Linear coefficients:")
    for j, name in enumerate(data["linear_names"]):
        print(f"      {name:30s}  {nc[j]:+8.4f}")
    for j in range(n_each):
        print(f"      {'peer_effect_' + str(j):30s}  {nc[n_linear + j]:+8.4f}")
    for j in range(n_each):
        print(f"      {'peer_eff_' + str(j) + '×tvl':30s}  {nc[n_linear + n_each + j]:+8.4f}")

    return med, r2s, pool_diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-alpha", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--encoder-hidden", type=int, default=16)
    parser.add_argument("--n-peer-outputs", type=int, default=1)
    parser.add_argument("--trend-windows", type=int, nargs="+", default=[7, 14, 30])
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("Hybrid: DeepSets Peer Encoder + Linear Noise Model")
    print(f"  encoder_hidden={args.encoder_hidden},"
          f" n_peer_outputs={args.n_peer_outputs}")
    print(f"  epochs={args.epochs}, lr={args.lr}, l2={args.l2_alpha}")
    print("=" * 70)

    matched_clean, option_c_clean = load_stage1()

    print("\nBuilding data...")
    t0 = time.time()
    data = build_data(matched_clean, option_c_clean,
                      trend_windows=tuple(args.trend_windows))
    n_pools = data["n_pools"]
    print(f"  {len(data['pool_idx'])} samples, {n_pools} pools")
    print(f"  Linear features: {data['n_linear_feat']}")
    print(f"  Peer encoder input: {data['n_peer_feat']} per peer,"
          f" {data['n_peers']} peers")
    print(f"  Build time: {time.time() - t0:.1f}s")

    # Temporal split
    day_idx = data["day_idx"]
    split_day = int(day_idx.max() * 0.7)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day

    train_data = _subset(data, train_mask)
    eval_data = _subset(data, eval_mask)

    # Init
    params, n_total_linear = init_params(
        jax.random.PRNGKey(42),
        data["n_peer_feat"], data["n_linear_feat"],
        args.encoder_hidden, args.n_peer_outputs,
        n_pools, data["init_log_cadences"],
    )

    # Warm-start linear coeffs via OLS (peer_effect = 0 initially)
    x_trn = data["x_linear"][train_mask]
    y_trn = data["y_total"][train_mask]
    # Pad with zeros for peer_effect columns
    x_trn_padded = np.concatenate([
        x_trn,
        np.zeros((x_trn.shape[0], args.n_peer_outputs * 2), dtype=np.float32)
    ], axis=1)
    sol, _, _, _ = np.linalg.lstsq(x_trn_padded, y_trn, rcond=None)
    params["noise_coeffs"] = jnp.array(sol.astype(np.float32))

    n_enc_params = (args.encoder_hidden * data["n_peer_feat"] +
                    args.encoder_hidden +
                    args.encoder_hidden * args.n_peer_outputs +
                    args.n_peer_outputs)
    print(f"\n  Params: {n_total_linear} linear + {n_enc_params} encoder"
          f" + {n_pools} cadences = {n_total_linear + n_enc_params + n_pools}")
    print(f"  Init cadence: {np.exp(data['init_log_cadences']).min():.1f}"
          f"-{np.median(np.exp(data['init_log_cadences'])):.1f}"
          f"-{np.exp(data['init_log_cadences']).max():.1f} min")

    # Train
    grad_fn = make_loss_fn(data["pool_coeffs"], data["pool_gas"], n_pools)

    print("\n  Compiling...")
    t0 = time.time()
    params = train(params, train_data, grad_fn, args.epochs, args.lr,
                   args.l2_alpha, args.huber_delta)
    print(f"  Training: {time.time() - t0:.1f}s")

    # Evaluate
    print("\n  --- Train ---")
    evaluate(params, train_data)
    print("\n  --- Eval ---")
    evaluate(params, eval_data)

    print(f"\n  Baselines (eval, total volume R²):")
    print(f"    V_arb only:       median R² = -0.33")
    print(f"    Linear shared:    median R² =  0.39")
    print(f"    Linear+intercept: median R² =  0.39")
    print(f"    DeepSets:         median R² =  0.43")


if __name__ == "__main__":
    main()
