"""MLP noise model with Binance market features and learnable cadence.

No cross-pool DEX dependency — only uses this pool's TVL + public market
data (Binance prices/volumes for BTC and the pool's tokens).

Architecture:
  log(V_noise) = MLP(x_market)
  V_total = V_arb(cadence) + exp(log_v_noise)

where x_market = [log_tvl, dow_sin, dow_cos, btc_features, tok_a_features,
tok_b_features, pair_vol, interactions].

Cadence is per-pool, learned jointly via Adam through PCHIP.

Usage:
  python experiments/run_mlp_noise.py
  python experiments/run_mlp_noise.py --hidden 64 32 --epochs 3000
  python experiments/run_mlp_noise.py --per-pool --hidden 32
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np


# ---- Model ----


def init_mlp_params(key, n_input, hidden_sizes, n_pools, init_log_cadences,
                    per_pool=False):
    """Initialize MLP parameters.

    MLP: input → hidden1 → ... → hiddenN → 1 (with ReLU activations).
    If per_pool: separate output bias per pool.
    """
    params = {}
    keys = jax.random.split(key, len(hidden_sizes) + 2)

    # Hidden layers
    in_dim = n_input
    for i, h in enumerate(hidden_sizes):
        params[f"W{i}"] = jax.random.normal(keys[i], (in_dim, h)) * np.sqrt(2.0 / in_dim)
        params[f"b{i}"] = jnp.zeros(h)
        in_dim = h

    # Output layer → scalar
    params["W_out"] = jax.random.normal(keys[-2], (in_dim, 1)) * 0.01
    params["b_out"] = jnp.zeros(1)

    if per_pool:
        params["pool_bias"] = jnp.zeros(n_pools)

    params["log_cadence"] = jnp.array(init_log_cadences)
    return params


def forward_mlp(params, x, pool_idx=None):
    """MLP forward pass. Returns (n_samples,) log_v_noise."""
    h = x
    i = 0
    while f"W{i}" in params:
        h = jnp.maximum(h @ params[f"W{i}"] + params[f"b{i}"], 0.0)
        i += 1
    out = (h @ params["W_out"] + params["b_out"])[:, 0]

    if "pool_bias" in params and pool_idx is not None:
        out = out + params["pool_bias"][pool_idx]

    return out


def make_loss_fn(pool_coeffs, pool_gas, n_pools):
    """Loss with learnable cadence + MLP noise model."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    def loss_fn(params, x, y_total, sample_grid_days, pool_idx,
                l2_alpha, huber_delta):
        log_v_noise = forward_mlp(params, x, pool_idx)

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

        reg = l2_alpha * sum(jnp.sum(v ** 2) for k, v in params.items()
                             if k.startswith("W"))
        return data_loss + reg

    return jax.jit(jax.value_and_grad(loss_fn))


def train(params, data, grad_fn, n_epochs, lr, l2_alpha, huber_delta,
          verbose=True, use_cosine=False, warmup_steps=100):
    x = jnp.array(data["x"])
    y = jnp.array(data["y_total"])
    sgd = jnp.array(data["sample_grid_days"])
    pidx = jnp.array(data["pool_idx"])

    if use_cosine:
        import optax
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.01,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=n_epochs,
            end_value=lr * 0.01,
        )
        optimizer = optax.adam(learning_rate=schedule)
        opt_state = optimizer.init(params)

        for epoch in range(n_epochs):
            loss_val, grads = grad_fn(
                params, x, y, sgd, pidx, l2_alpha, huber_delta)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
                cads = np.exp(np.array(params["log_cadence"]))
                cur_lr = float(schedule(epoch))
                print(f"  epoch {epoch:5d}  loss={float(loss_val):.6f}"
                      f"  lr={cur_lr:.2e}"
                      f"  cad=[{cads.min():.1f}-{np.median(cads):.1f}-{cads.max():.1f}]")
    else:
        m = {k: jnp.zeros_like(v) for k, v in params.items()}
        v = {k: jnp.zeros_like(v) for k, v in params.items()}

        for epoch in range(n_epochs):
            loss_val, grads = grad_fn(
                params, x, y, sgd, pidx, l2_alpha, huber_delta)

            for k in params:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                m_hat = m[k] / (1.0 - 0.9 ** (epoch + 1))
                v_hat = v[k] / (1.0 - 0.999 ** (epoch + 1))
                params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)

            if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
                cads = np.exp(np.array(params["log_cadence"]))
                print(f"  epoch {epoch:5d}  loss={float(loss_val):.6f}"
                      f"  cad=[{cads.min():.1f}-{np.median(cads):.1f}-{cads.max():.1f}]")

    return params


def evaluate(params, data, label=""):
    """Evaluate decomposition."""
    from quantammsim.calibration.grid_interpolation import interpolate_pool_daily

    x = np.array(data["x"])
    y_total = np.array(data["y_total"])
    pool_idx = np.array(data["pool_idx"])
    sgd = np.array(data["sample_grid_days"])
    log_cadence = np.array(params["log_cadence"])
    init_cads = data["init_log_cadences"]
    pool_ids = data["pool_ids"]
    n_pools = data["n_pools"]

    log_v_noise = np.array(forward_mlp(
        params, jnp.array(x),
        jnp.array(pool_idx) if "pool_bias" in params else None))
    v_noise = np.exp(log_v_noise)

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
          f" {'Arb%':>6s} {'Noise%':>7s} {'Flag':>5s}")
    print(f"    {'-'*55}")

    r2s = {}
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
        flags = []
        if arb_pct > 150: flags.append("A")
        if cl <= 1.01 or cl >= 59.9: flags.append("B")
        if r2s[i] < 0: flags.append("X")
        print(f"    {pid[:16]:16s} {r2s[i]:6.3f} {ci:5.1f} → {cl:5.1f}"
              f" {arb_pct:6.0f}% {noise_pct:6.0f}% {''.join(flags):>5s}")

    vals = [x for x in r2s.values() if np.isfinite(x)]
    med = np.median(vals) if vals else float("nan")
    healthy = [r for r in r2s.values() if r > 0 and np.isfinite(r)]
    med_h = np.median(healthy) if healthy else float("nan")
    print(f"\n    Median R²: {med:.4f} (healthy: {med_h:.4f})")
    return med, r2s


def run_optuna(data, n_trials):
    """Optuna sweep over MLP architecture and training hyperparameters."""
    import optuna

    day_idx = data["day_idx"]
    n_samples = len(day_idx)
    split_day = int(day_idx.max() * 0.7)
    train_mask = day_idx <= split_day
    eval_mask = day_idx > split_day
    train_data = {k: v[train_mask] if isinstance(v, np.ndarray)
                  and v.shape[0] == n_samples else v
                  for k, v in data.items()}
    eval_data = {k: v[eval_mask] if isinstance(v, np.ndarray)
                 and v.shape[0] == n_samples else v
                 for k, v in data.items()}

    n_pools = data["n_pools"]
    n_feat = data["n_feat"]

    def objective(trial):
        # Architecture
        n_layers = trial.suggest_int("n_layers", 1, 7)
        first_hidden = trial.suggest_categorical("first_hidden", [8, 16, 32, 64, 128, 256])
        # Bottleneck: each layer is half the previous (min 2)
        bottleneck_ratio = trial.suggest_categorical("bottleneck_ratio", [0.5, 0.75, 1.0])
        hidden = []
        h = first_hidden
        for _ in range(n_layers):
            hidden.append(h)
            h = max(int(h * bottleneck_ratio), 2)

        # Training
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        l2_alpha = trial.suggest_float("l2_alpha", 1e-5, 5e-1, log=True)
        huber_delta = trial.suggest_categorical("huber_delta", [0.5, 1.0, 1.5, 2.0])
        n_epochs = trial.suggest_categorical("n_epochs", [2000, 5000, 10000, 20000])
        use_cosine = trial.suggest_categorical("use_cosine", [True, False])
        per_pool = trial.suggest_categorical("per_pool", [True, False])

        params = init_mlp_params(
            jax.random.PRNGKey(42), n_feat, hidden, n_pools,
            data["init_log_cadences"], per_pool=per_pool)

        # OLS warm-start
        x_trn = jnp.array(train_data["x"])
        y_trn = np.array(train_data["y_total"])
        h_act = np.array(x_trn)
        i = 0
        while f"W{i}" in params:
            h_act = np.maximum(
                h_act @ np.array(params[f"W{i}"]) + np.array(params[f"b{i}"]), 0.0)
            i += 1
        h_bias = np.concatenate([h_act, np.ones((h_act.shape[0], 1))], axis=1)
        sol, _, _, _ = np.linalg.lstsq(h_bias, y_trn[:, None], rcond=None)
        params["W_out"] = jnp.array(sol[:-1].astype(np.float32))
        params["b_out"] = jnp.array(sol[-1:].astype(np.float32))

        grad_fn = make_loss_fn(data["pool_coeffs"], data["pool_gas"], n_pools)
        params = train(params, train_data, grad_fn, n_epochs, lr,
                       l2_alpha, huber_delta, verbose=False,
                       use_cosine=use_cosine)

        # Eval
        x_eval = np.array(eval_data["x"])
        y_eval = np.array(eval_data["y_total"])
        pool_idx_eval = np.array(eval_data["pool_idx"])
        sgd_eval = np.array(eval_data["sample_grid_days"])
        log_cadence = np.array(params["log_cadence"])

        log_v_noise = np.array(forward_mlp(
            params, jnp.array(x_eval),
            jnp.array(pool_idx_eval) if per_pool else None))

        from quantammsim.calibration.grid_interpolation import interpolate_pool_daily
        v_arb = np.zeros(len(y_eval))
        for i in range(n_pools):
            mask = pool_idx_eval == i
            if not mask.any():
                continue
            v_arb_all = np.array(interpolate_pool_daily(
                data["pool_coeffs"][i], jnp.float64(log_cadence[i]),
                data["pool_gas"][i]))
            v_arb[mask] = v_arb_all[sgd_eval[mask]]

        log_v_arb = np.log(np.maximum(v_arb, 1e-10))
        pred_total = np.logaddexp(log_v_arb, log_v_noise)

        r2s = []
        for i in range(n_pools):
            mask = pool_idx_eval == i
            if mask.sum() < 2:
                continue
            yt = y_eval[mask]
            pt = pred_total[mask]
            ss_res = np.sum((yt - pt) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2s.append(1 - ss_res / max(ss_tot, 1e-10))

        med_r2 = float(np.median(r2s)) if r2s else -10.0
        arch_str = "×".join(str(h) for h in hidden)
        print(f"  Trial {trial.number}: eval={med_r2:.4f}"
              f"  arch=[{arch_str}]"
              f"  {'cosine' if use_cosine else 'const'}"
              f"  {'per_pool' if per_pool else 'shared'}"
              f"  lr={lr:.1e} l2={l2_alpha:.1e}"
              f"  hub={huber_delta} ep={n_epochs}")

        # Save every trial's model
        trial_dir = os.path.join("results", "mlp_noise", "trials", f"trial_{trial.number:04d}")
        os.makedirs(trial_dir, exist_ok=True)
        save_dict = {k: np.array(v) for k, v in params.items()}
        save_dict["x_mean"] = data.get("x_mean", np.zeros(n_feat))
        save_dict["x_std"] = data.get("x_std", np.ones(n_feat))
        np.savez(os.path.join(trial_dir, "model.npz"), **save_dict)
        import json as _json
        _meta = {
            "pool_ids": data["pool_ids"],
            "feat_names": data["feat_names"],
            "n_feat": n_feat,
            "hidden": hidden,
            "per_pool": per_pool,
            "eval_r2": med_r2,
            "hparams": {
                "hidden": hidden, "lr": lr, "l2_alpha": l2_alpha,
                "huber_delta": huber_delta, "n_epochs": n_epochs,
                "use_cosine": use_cosine, "per_pool": per_pool,
                "bottleneck_ratio": bottleneck_ratio,
                "first_hidden": first_hidden, "n_layers": n_layers,
            },
        }
        with open(os.path.join(trial_dir, "meta.json"), "w") as _f:
            _json.dump(_meta, _f, indent=2)

        return med_r2

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\n{'='*70}")
    print(f"Optuna Results (MLP noise)")
    print(f"{'='*70}")
    print(f"  Best eval R²: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in sorted(study.best_params.items()):
        print(f"    {k}: {v}")

    trials = sorted(study.trials, key=lambda t: t.value if t.value else -999,
                    reverse=True)
    print(f"\n  Top 10:")
    for t in trials[:10]:
        if t.value is not None:
            n_l = t.params["n_layers"]
            fh = t.params["first_hidden"]
            h = fh
            arch = []
            for _ in range(n_l):
                arch.append(h)
                h = max(int(h * t.params.get("bottleneck_ratio", 0.5)), 2)
            print(f"    #{t.number}: eval={t.value:.4f}"
                  f"  arch={arch}"
                  f"  ep={t.params['n_epochs']}"
                  f"  {'cos' if t.params['use_cosine'] else 'cst'}"
                  f"  {'pp' if t.params['per_pool'] else 'sh'}")

    # Copy best trial to top-level artifact
    best_trial = study.best_trial
    best_trial_dir = os.path.join("results", "mlp_noise", "trials",
                                  f"trial_{best_trial.number:04d}")
    save_dir = "results/mlp_noise"
    if os.path.exists(os.path.join(best_trial_dir, "model.npz")):
        import shutil
        shutil.copy2(os.path.join(best_trial_dir, "model.npz"),
                     os.path.join(save_dir, "model.npz"))
        shutil.copy2(os.path.join(best_trial_dir, "meta.json"),
                     os.path.join(save_dir, "meta.json"))
        print(f"\n  Copied best trial ({best_trial.number}) to: {save_dir}")

        # TVL response check on best model
        import json as _json
        art = dict(np.load(os.path.join(save_dir, "model.npz"), allow_pickle=True))
        with open(os.path.join(save_dir, "meta.json")) as _f:
            meta = _json.load(_f)
        best_params = {k: jnp.array(art[k]) for k in art
                       if k.startswith("W") or k.startswith("b")
                       or k == "log_cadence" or k == "pool_bias"}
        per_pool = meta.get("per_pool", False)
        pool_idx_probe = jnp.array([0]) if per_pool else None

        print(f"\n  TVL Response Check (best model, trial {best_trial.number}):")
        x_probe = np.zeros((1, n_feat), dtype=np.float32)
        x_probe[0, 0] = 1.0
        x_probe[0, 4] = 10.5  # typical btc_log_price
        prev_noise = None
        for tvl in [1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8]:
            x_probe[0, 1] = np.log(tvl)
            out = np.array(forward_mlp(best_params, jnp.array(x_probe),
                                       pool_idx_probe))
            noise = np.exp(out[0])
            if prev_noise is not None and prev_noise > 0:
                ratio = noise / prev_noise
                print(f"    TVL=${tvl:>11,.0f}  noise=${noise:>12,.0f}/day"
                      f"  ({ratio:.2f}x prev)")
            else:
                print(f"    TVL=${tvl:>11,.0f}  noise=${noise:>12,.0f}/day")
            prev_noise = noise

    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, nargs="+", default=[32],
                        help="Hidden layer sizes (e.g. --hidden 64 32)")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-alpha", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--cosine", action="store_true",
                        help="Use optax Adam with cosine LR decay")
    parser.add_argument("--tune", type=int, default=0,
                        help="Optuna sweep (0 = single run)")
    parser.add_argument("--trend-windows", type=int, nargs="+", default=[7])
    parser.add_argument("--per-pool", action="store_true",
                        help="Per-pool output bias")
    parser.add_argument("--pool-attrs", action="store_true",
                        help="Append static pool attributes to input")
    parser.add_argument("--no-split", action="store_true",
                        help="Train on all data")
    parser.add_argument("--save-artifact", default=None,
                        help="Save model to this directory")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 70)
    print("MLP Noise Model (Binance features only, no cross-pool DEX)")
    print(f"  hidden={args.hidden}, per_pool={args.per_pool}")
    print(f"  epochs={args.epochs}, lr={args.lr}, l2={args.l2_alpha}")
    print("=" * 70)

    # Build data WITHOUT cross-pool features
    from experiments.run_linear_market_noise import load_stage1, build_data

    matched_clean, option_c_clean = load_stage1()

    print("\nBuilding data...")
    t0 = time.time()
    data = build_data(
        matched_clean, option_c_clean,
        trend_windows=tuple(args.trend_windows),
        include_market=True,
        include_cross_pool=False,  # No DEX peer features
    )
    n_pools = data["n_pools"]
    n_feat = data["n_feat"]
    print(f"  {len(data['pool_idx'])} samples, {n_pools} pools,"
          f" {n_feat} features, {time.time() - t0:.1f}s")

    # Append pool attributes if requested
    if args.pool_attrs:
        from quantammsim.calibration.pool_data import build_pool_attributes
        X_attr, attr_names, _ = build_pool_attributes(matched_clean)
        # Standardize
        attr_mean = X_attr.mean(axis=0)
        attr_std = X_attr.std(axis=0)
        attr_std[attr_std < 1e-6] = 1.0
        X_attr_norm = ((X_attr - attr_mean) / attr_std).astype(np.float32)

        # Broadcast to per-sample: each sample gets its pool's attributes
        pool_idx = data["pool_idx"]
        x_attr_samples = X_attr_norm[pool_idx]
        data["x"] = np.concatenate([data["x"], x_attr_samples], axis=1)
        data["n_feat"] = data["x"].shape[1]
        data["feat_names"] = data["feat_names"] + attr_names
        n_feat = data["n_feat"]
        print(f"  + {len(attr_names)} pool attributes → {n_feat} total features")

    print(f"  Features: {data['feat_names']}")

    if args.tune > 0:
        run_optuna(data, args.tune)
        return

    # Split
    if args.no_split:
        train_data = data
        eval_data = None
    else:
        day_idx = data["day_idx"]
        n_samples = len(day_idx)
        split_day = int(day_idx.max() * 0.7)
        train_mask = day_idx <= split_day
        eval_mask = day_idx > split_day
        train_data = {k: v[train_mask] if isinstance(v, np.ndarray)
                      and v.shape[0] == n_samples else v
                      for k, v in data.items()}
        eval_data = {k: v[eval_mask] if isinstance(v, np.ndarray)
                     and v.shape[0] == n_samples else v
                     for k, v in data.items()}

    # Init
    params = init_mlp_params(
        jax.random.PRNGKey(42), n_feat, args.hidden, n_pools,
        data["init_log_cadences"], per_pool=args.per_pool)

    n_params = sum(v.size for v in params.values())
    print(f"\n  Total params: {n_params}"
          f" (MLP: {n_params - n_pools - (n_pools if args.per_pool else 0)},"
          f" cadence: {n_pools}"
          f"{',' + str(n_pools) + ' pool biases' if args.per_pool else ''})")

    # Warm-start output layer via OLS through hidden activations
    x_trn = jnp.array(train_data["x"])
    y_trn = np.array(train_data["y_total"])
    h = np.array(x_trn)
    i = 0
    while f"W{i}" in params:
        h = np.maximum(h @ np.array(params[f"W{i}"]) + np.array(params[f"b{i}"]), 0.0)
        i += 1
    h_bias = np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)
    sol, _, _, _ = np.linalg.lstsq(h_bias, y_trn[:, None], rcond=None)
    params["W_out"] = jnp.array(sol[:-1].astype(np.float32))
    params["b_out"] = jnp.array(sol[-1:].astype(np.float32))
    print(f"  OLS warm-start on hidden activations")

    # Train
    grad_fn = make_loss_fn(data["pool_coeffs"], data["pool_gas"], n_pools)

    print(f"\n  Compiling + training...")
    t0 = time.time()
    params = train(params, train_data, grad_fn, args.epochs, args.lr,
                   args.l2_alpha, args.huber_delta,
                   use_cosine=args.cosine)
    print(f"  Training: {time.time() - t0:.1f}s")

    # Evaluate
    if eval_data is not None:
        print("\n  --- Train ---")
        evaluate(params, train_data)
        print("\n  --- Eval ---")
        evaluate(params, eval_data)
    else:
        print("\n  --- All data ---")
        evaluate(params, train_data)

    print(f"\n  Baselines:")
    print(f"    Linear (no cross-pool): median R² ≈ 0.48")
    print(f"    Linear (with cross-pool): median R² ≈ 0.53")
    print(f"    Per-pool linear: median R² ≈ 0.61")

    # Save artifact
    if args.save_artifact:
        import json
        os.makedirs(args.save_artifact, exist_ok=True)
        # Save params as npz
        save_dict = {k: np.array(v) for k, v in params.items()}
        save_dict["x_mean"] = data["x_mean"] if "x_mean" in data else np.zeros(n_feat)
        save_dict["x_std"] = data["x_std"] if "x_std" in data else np.ones(n_feat)
        np.savez(os.path.join(args.save_artifact, "model.npz"), **save_dict)
        # Save meta
        meta = {
            "pool_ids": data["pool_ids"],
            "feat_names": data["feat_names"],
            "n_feat": n_feat,
            "hidden": args.hidden,
            "per_pool": args.per_pool,
            "hparams": {
                "hidden": args.hidden,
                "lr": args.lr,
                "l2_alpha": args.l2_alpha,
                "huber_delta": args.huber_delta,
                "epochs": args.epochs,
                "trend_windows": args.trend_windows,
                "use_cosine": args.cosine,
                "per_pool": args.per_pool,
            },
        }
        with open(os.path.join(args.save_artifact, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  Saved artifact to: {args.save_artifact}")


if __name__ == "__main__":
    main()
