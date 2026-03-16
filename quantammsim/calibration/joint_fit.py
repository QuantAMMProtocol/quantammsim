"""Joint end-to-end optimization (Option A) for the direct calibration pipeline.

A parametric f_params maps pool_attributes → (cadence, gas, noise_coeffs),
optimized simultaneously across all pools through the grid interpolation loss.

Two noise modes:
  - "per_pool_noise": each pool has independent noise_coeffs (most flexible)
  - "shared_noise": noise_coeffs = bias_noise + x_attr @ W_noise (generalizes)

The cadence/gas mapping is always shared:
  log_cadence = bias_cad + x_attr @ W_cad
  log_gas     = bias_gas + x_attr @ W_gas
"""

from typing import Dict, List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from quantammsim.calibration.grid_interpolation import (
    PoolCoeffsDaily,
    interpolate_pool_daily,
)
from quantammsim.calibration.loss import K_OBS
from quantammsim.calibration.pool_data import build_pool_attributes, build_x_obs


class JointData(NamedTuple):
    """Batched data for joint optimization."""
    pool_data: list       # list of dicts with coeffs, x_obs, y_obs, day_indices
    x_attr: jnp.ndarray   # (n_pools, K_attr) pool attributes (no intercept)
    pool_ids: list         # list of pool_id prefixes
    attr_names: list       # attribute column names


def prepare_joint_data(
    matched: Dict[str, dict],
    drop_chain_dummies: bool = False,
    fix_gas_to_chain: bool = False,
    reduced_x_obs: bool = False,
) -> JointData:
    """Build batched JAX arrays from matched pool data.

    Args:
        matched: dict from match_grids_to_panel
        drop_chain_dummies: if True, remove chain_* columns from attributes
        fix_gas_to_chain: if True, store fixed_log_gas per pool from CHAIN_GAS_USD
        reduced_x_obs: if True, use 4-column reduced x_obs
            (removes sigma/fee terms to avoid identification problems)

    Returns:
        JointData with per-pool JAX arrays and shared attribute matrix.
    """
    from quantammsim.calibration.loss import CHAIN_GAS_USD

    X_attr, attr_names, pool_ids = build_pool_attributes(matched)

    if drop_chain_dummies:
        keep = [i for i, name in enumerate(attr_names)
                if not name.startswith("chain_")]
        X_attr = X_attr[:, keep]
        attr_names = [attr_names[i] for i in keep]

    pool_data = []
    for pid in pool_ids:
        entry = matched[pid]
        panel = entry["panel"]
        x_obs = build_x_obs(panel, reduced=reduced_x_obs)
        y_obs = panel["log_volume"].values.astype(float)

        d = {
            "coeffs": entry["coeffs"],
            "x_obs": jnp.array(x_obs),
            "y_obs": jnp.array(y_obs),
            "day_indices": jnp.array(entry["day_indices"]),
        }
        if fix_gas_to_chain:
            chain = entry["chain"]
            gas_usd = CHAIN_GAS_USD.get(chain, 1.0)
            d["fixed_log_gas"] = jnp.float64(np.log(max(gas_usd, 1e-6)))

        pool_data.append(d)

    return JointData(
        pool_data=pool_data,
        x_attr=jnp.array(X_attr),
        pool_ids=pool_ids,
        attr_names=attr_names,
    )


def prepare_token_factored_data(
    matched: Dict[str, dict],
    reduced_x_obs: bool = True,
    fix_gas_to_chain: bool = True,
) -> tuple:
    """Prepare JointData + token encoding for TokenFactoredNoiseHead.

    Returns (jdata, token_encoding) where token_encoding is the dict from
    encode_tokens() containing token/chain structure for constructing the head.
    """
    from quantammsim.calibration.pool_data import encode_tokens

    jdata = prepare_joint_data(
        matched,
        fix_gas_to_chain=fix_gas_to_chain,
        reduced_x_obs=reduced_x_obs,
    )

    token_encoding = encode_tokens(matched)

    return jdata, token_encoding


def pack_joint_params(
    bias_cad: float,
    bias_gas: float,
    W_cad: jnp.ndarray,
    W_gas: jnp.ndarray,
    noise_params: jnp.ndarray,
) -> jnp.ndarray:
    """Pack joint params into flat array.

    Layout: [bias_cad, bias_gas, W_cad(k_attr), W_gas(k_attr), noise_params...]

    noise_params is either:
      - (n_pools, K_OBS) for per_pool_noise mode
      - (1 + K_attr, K_OBS) for shared_noise mode (row 0 = noise bias)
    """
    return jnp.concatenate([
        jnp.array([bias_cad, bias_gas]),
        W_cad.ravel(),
        W_gas.ravel(),
        noise_params.ravel(),
    ])


def pack_joint_params_fixed_gas(
    bias_cad: float,
    W_cad: jnp.ndarray,
    noise_params: jnp.ndarray,
) -> jnp.ndarray:
    """Pack joint params with gas excluded.

    Layout: [bias_cad, W_cad(k_attr), noise_params...]
    """
    return jnp.concatenate([
        jnp.array([bias_cad]),
        W_cad.ravel(),
        noise_params.ravel(),
    ])


def unpack_joint_params(
    flat: jnp.ndarray, config: dict
) -> dict:
    """Unpack flat array to structured params.

    config must have: k_attr, n_pools, mode
    config may have: fix_gas (bool) — if True, no bias_gas/W_gas in flat array
    """
    k_attr = config["k_attr"]
    mode = config["mode"]
    fix_gas = config.get("fix_gas", False)

    if fix_gas:
        bias_cad = flat[0]
        W_cad = flat[1:1 + k_attr]
        rest = flat[1 + k_attr:]
    else:
        bias_cad = flat[0]
        bias_gas = flat[1]
        W_cad = flat[2:2 + k_attr]
        W_gas = flat[2 + k_attr:2 + 2 * k_attr]
        rest = flat[2 + 2 * k_attr:]

    if mode == "per_pool_noise":
        n_pools = config["n_pools"]
        noise_coeffs = rest.reshape(n_pools, K_OBS)
        if fix_gas:
            return {"bias_cad": bias_cad, "W_cad": W_cad,
                    "noise_coeffs": noise_coeffs}
        return {
            "bias_cad": bias_cad, "bias_gas": bias_gas,
            "W_cad": W_cad, "W_gas": W_gas,
            "noise_coeffs": noise_coeffs,
        }
    else:  # shared_noise
        W_noise_full = rest.reshape(1 + k_attr, K_OBS)
        if fix_gas:
            return {"bias_cad": bias_cad, "W_cad": W_cad,
                    "bias_noise": W_noise_full[0], "W_noise": W_noise_full[1:]}
        return {
            "bias_cad": bias_cad, "bias_gas": bias_gas,
            "W_cad": W_cad, "W_gas": W_gas,
            "bias_noise": W_noise_full[0],
            "W_noise": W_noise_full[1:],
        }


def _make_pool_loss_fn(
    pool_idx: int,
    pool_data_i: dict,
    x_attr_i: jnp.ndarray,
    config: dict,
):
    """Create a JIT'd loss function for a single pool.

    Closes over pool-specific data; takes only params_flat as input.
    Each pool gets its own small JIT'd computation graph.

    If config["fix_gas"] is True, gas comes from pool_data_i["fixed_log_gas"]
    instead of being predicted from attributes.
    """
    coeffs = pool_data_i["coeffs"]
    x_obs = pool_data_i["x_obs"]
    y_obs = pool_data_i["y_obs"]
    day_indices = pool_data_i["day_indices"]
    mode = config["mode"]
    fix_gas = config.get("fix_gas", False)
    i = pool_idx

    if fix_gas:
        fixed_log_gas = pool_data_i["fixed_log_gas"]

        @jax.jit
        def pool_loss_fn(params_flat):
            params = unpack_joint_params(params_flat, config)
            log_cad = params["bias_cad"] + jnp.dot(x_attr_i, params["W_cad"])

            if mode == "per_pool_noise":
                noise_c = params["noise_coeffs"][i]
            else:
                noise_c = params["bias_noise"] + jnp.dot(x_attr_i, params["W_noise"])

            v_arb_all = interpolate_pool_daily(coeffs, log_cad, jnp.exp(fixed_log_gas))
            v_arb = v_arb_all[day_indices]
            v_noise = jnp.exp(x_obs @ noise_c)
            log_v_pred = jnp.log(jnp.maximum(v_arb + v_noise, 1e-6))
            return jnp.mean((log_v_pred - y_obs) ** 2)
    else:
        @jax.jit
        def pool_loss_fn(params_flat):
            params = unpack_joint_params(params_flat, config)
            log_cad = params["bias_cad"] + jnp.dot(x_attr_i, params["W_cad"])
            log_gas = params["bias_gas"] + jnp.dot(x_attr_i, params["W_gas"])

            if mode == "per_pool_noise":
                noise_c = params["noise_coeffs"][i]
            else:
                noise_c = params["bias_noise"] + jnp.dot(x_attr_i, params["W_noise"])

            v_arb_all = interpolate_pool_daily(coeffs, log_cad, jnp.exp(log_gas))
            v_arb = v_arb_all[day_indices]
            v_noise = jnp.exp(x_obs @ noise_c)
            log_v_pred = jnp.log(jnp.maximum(v_arb + v_noise, 1e-6))
            return jnp.mean((log_v_pred - y_obs) ** 2)

    return pool_loss_fn


def make_joint_loss_fn(
    jdata: JointData,
    mode: str = "per_pool_noise",
    alpha_cad: float = 0.01,
    alpha_gas: float = 0.01,
    fix_gas: bool = False,
):
    """Create per-pool JIT'd loss functions and a Python-level aggregator.

    Each pool gets its own small JIT'd computation graph (compiled
    independently), avoiding a massive unrolled trace. The outer
    function sums per-pool losses in Python and adds regularization.

    Loss averages over pools (not observations), giving equal weight
    to each pool regardless of observation count.

    L2 regularization is applied to W_cad (and W_gas if not fixed).

    Args:
        jdata: JointData from prepare_joint_data
        mode: "per_pool_noise" or "shared_noise"
        alpha_cad: L2 regularization on W_cad
        alpha_gas: L2 regularization on W_gas (ignored if fix_gas=True)
        fix_gas: if True, gas is fixed per pool (no W_gas in params)

    Returns:
        loss_fn(params_flat) -> scalar loss
    """
    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    config = {"k_attr": k_attr, "n_pools": n_pools, "mode": mode,
              "fix_gas": fix_gas}

    # Build per-pool JIT'd loss functions
    pool_loss_fns = []
    pool_val_and_grad_fns = []
    for i in range(n_pools):
        fn = _make_pool_loss_fn(i, jdata.pool_data[i], jdata.x_attr[i], config)
        pool_loss_fns.append(fn)
        pool_val_and_grad_fns.append(jax.value_and_grad(fn))

    def loss_fn(params_flat):
        total = sum(fn(params_flat) for fn in pool_loss_fns)
        data_loss = total / n_pools

        params = unpack_joint_params(params_flat, config)
        reg = alpha_cad * jnp.sum(params["W_cad"] ** 2)
        if not fix_gas:
            reg = reg + alpha_gas * jnp.sum(params["W_gas"] ** 2)
        return data_loss + reg

    # Attach per-pool functions for the value_and_grad wrapper
    loss_fn._pool_val_and_grad_fns = pool_val_and_grad_fns
    loss_fn._n_pools = n_pools
    loss_fn._config = config
    loss_fn._alpha_cad = alpha_cad
    loss_fn._alpha_gas = alpha_gas

    return loss_fn


def make_initial_joint_params(
    jdata: JointData,
    mode: str = "per_pool_noise",
    init_from_option_c: Optional[Dict[str, dict]] = None,
    fix_gas: bool = False,
) -> jnp.ndarray:
    """Create initial parameter vector.

    If init_from_option_c is provided, warm-start from Option C per-pool fits.
    If fix_gas is True, excludes bias_gas and W_gas from the parameter vector.
    """
    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    x_attr_np = np.array(jdata.x_attr)

    if init_from_option_c is not None:
        pool_ids = jdata.pool_ids
        valid = {p: init_from_option_c[p] for p in pool_ids
                 if p in init_from_option_c
                 and np.isfinite(init_from_option_c[p].get("loss", float("nan")))}

        if len(valid) < len(pool_ids):
            default_lc = np.log(12.0)
            default_lg = np.log(1.0)
            for p in pool_ids:
                if p not in valid:
                    valid[p] = {
                        "log_cadence": default_lc,
                        "log_gas": default_lg,
                        "noise_coeffs": np.zeros(K_OBS),
                    }

        log_cads = np.array([valid[p]["log_cadence"] for p in pool_ids])
        noise_all = np.array([valid[p]["noise_coeffs"] for p in pool_ids])

        X_aug = np.column_stack([np.ones(n_pools), x_attr_np])
        cad_params, _, _, _ = np.linalg.lstsq(X_aug, log_cads, rcond=None)
        bias_cad, W_cad = cad_params[0], cad_params[1:]

        if not fix_gas:
            log_gases = np.array([valid[p]["log_gas"] for p in pool_ids])
            gas_params, _, _, _ = np.linalg.lstsq(X_aug, log_gases, rcond=None)
            bias_gas, W_gas = gas_params[0], gas_params[1:]

        if mode == "per_pool_noise":
            noise_params = noise_all
        else:
            noise_aug, _, _, _ = np.linalg.lstsq(X_aug, noise_all, rcond=None)
            noise_params = noise_aug
    else:
        bias_cad = np.log(12.0)
        W_cad = np.zeros(k_attr)

        if not fix_gas:
            bias_gas = np.log(1.0)
            W_gas = np.zeros(k_attr)

        if mode == "per_pool_noise":
            noise_params = np.zeros((n_pools, K_OBS))
            for i, pd in enumerate(jdata.pool_data):
                x_obs_np = np.array(pd["x_obs"])
                y_obs_np = np.array(pd["y_obs"])
                c, _, _, _ = np.linalg.lstsq(x_obs_np, y_obs_np, rcond=None)
                noise_params[i] = c
        else:
            all_x = np.vstack([np.array(pd["x_obs"]) for pd in jdata.pool_data])
            all_y = np.concatenate([np.array(pd["y_obs"]) for pd in jdata.pool_data])
            c, _, _, _ = np.linalg.lstsq(all_x, all_y, rcond=None)
            noise_params = np.zeros((1 + k_attr, K_OBS))
            noise_params[0, :] = c

    if fix_gas:
        return pack_joint_params_fixed_gas(
            float(bias_cad),
            jnp.array(W_cad),
            jnp.array(noise_params),
        )
    else:
        return pack_joint_params(
            float(bias_cad),
            float(bias_gas),
            jnp.array(W_cad),
            jnp.array(W_gas),
            jnp.array(noise_params),
        )


def _make_bounds(k_attr, n_pools, mode, fix_gas=False):
    """Build scipy bounds for joint params."""
    if fix_gas:
        # bias_cad only
        bounds = [(None, None)] * 1
        # W_cad only
        bounds += [(None, None)] * k_attr
    else:
        # bias_cad, bias_gas
        bounds = [(None, None)] * 2
        # W_cad, W_gas
        bounds += [(None, None)] * (2 * k_attr)

    if mode == "per_pool_noise":
        bounds += [(None, None)] * (n_pools * K_OBS)
    else:
        bounds += [(None, None)] * ((1 + k_attr) * K_OBS)

    return bounds


def fit_joint(
    matched: Dict[str, dict],
    mode: str = "per_pool_noise",
    init_from_option_c: Optional[Dict[str, dict]] = None,
    maxiter: int = 500,
    alpha_cad: float = 0.01,
    alpha_gas: float = 0.01,
    drop_chain_dummies: bool = False,
    fix_gas_to_chain: bool = False,
) -> dict:
    """Joint end-to-end optimization across all pools.

    Args:
        matched: dict from match_grids_to_panel
        mode: "per_pool_noise" or "shared_noise"
        init_from_option_c: Optional Option C results for warm start.
        maxiter: max L-BFGS-B iterations
        alpha_cad: L2 regularization on W_cad (not bias)
        alpha_gas: L2 regularization on W_gas (not bias, ignored if fix_gas)
        drop_chain_dummies: if True, remove chain_* columns from attributes
        fix_gas_to_chain: if True, gas is fixed to known chain-level costs

    Returns dict with fitted params and diagnostics.
    """
    jdata = prepare_joint_data(matched, drop_chain_dummies=drop_chain_dummies,
                               fix_gas_to_chain=fix_gas_to_chain)
    loss_fn = make_joint_loss_fn(jdata, mode=mode,
                                  alpha_cad=alpha_cad, alpha_gas=alpha_gas,
                                  fix_gas=fix_gas_to_chain)
    init = make_initial_joint_params(jdata, mode=mode,
                                     init_from_option_c=init_from_option_c,
                                     fix_gas=fix_gas_to_chain)

    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    config = {"k_attr": k_attr, "n_pools": n_pools, "mode": mode,
              "fix_gas": fix_gas_to_chain}
    bounds = _make_bounds(k_attr, n_pools, mode, fix_gas=fix_gas_to_chain)

    pool_vg_fns = loss_fn._pool_val_and_grad_fns

    if fix_gas_to_chain:
        w_cad_start = 1
        w_cad_end = 1 + k_attr
    else:
        w_cad_start = 2
        w_cad_end = 2 + k_attr
        w_gas_start = 2 + k_attr
        w_gas_end = 2 + 2 * k_attr

    def scipy_wrapper(params_np):
        params_j = jnp.array(params_np)

        total_val = 0.0
        total_grad = jnp.zeros_like(params_j)
        for vg_fn in pool_vg_fns:
            v, g = vg_fn(params_j)
            total_val += float(v)
            total_grad = total_grad + g

        data_loss = total_val / n_pools
        data_grad = total_grad / n_pools

        reg = alpha_cad * float(jnp.sum(params_j[w_cad_start:w_cad_end] ** 2))
        reg_grad = jnp.zeros_like(params_j)
        reg_grad = reg_grad.at[w_cad_start:w_cad_end].set(
            2 * alpha_cad * params_j[w_cad_start:w_cad_end])

        if not fix_gas_to_chain:
            reg += alpha_gas * float(jnp.sum(params_j[w_gas_start:w_gas_end] ** 2))
            reg_grad = reg_grad.at[w_gas_start:w_gas_end].set(
                2 * alpha_gas * params_j[w_gas_start:w_gas_end])

        val = data_loss + reg
        grad = data_grad + reg_grad
        return val, np.array(grad, dtype=np.float64)

    init_np = np.array(init, dtype=np.float64)
    init_loss = float(loss_fn(jnp.array(init_np)))

    result = scipy.optimize.minimize(
        scipy_wrapper,
        init_np,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8},
    )

    params = unpack_joint_params(jnp.array(result.x), config)

    out = {
        "init_loss": init_loss,
        "bias_cad": float(params["bias_cad"]),
        "W_cad": np.array(params["W_cad"]),
        "loss": float(result.fun),
        "converged": result.success,
        "mode": mode,
        "k_attr": k_attr,
        "pool_ids": jdata.pool_ids,
        "attr_names": jdata.attr_names,
        "fix_gas": fix_gas_to_chain,
    }

    if fix_gas_to_chain:
        # Store per-pool fixed gas values for downstream use
        from quantammsim.calibration.loss import CHAIN_GAS_USD
        gas_per_pool = []
        for pid in jdata.pool_ids:
            chain = matched[pid]["chain"]
            gas_per_pool.append(CHAIN_GAS_USD.get(chain, 1.0))
        out["gas_per_pool"] = np.array(gas_per_pool)
        out["bias_gas"] = 0.0
        out["W_gas"] = np.zeros(k_attr)
    else:
        out["bias_gas"] = float(params["bias_gas"])
        out["W_gas"] = np.array(params["W_gas"])

    if mode == "per_pool_noise":
        out["noise_coeffs"] = np.array(params["noise_coeffs"])
    else:
        out["bias_noise"] = np.array(params["bias_noise"])
        out["W_noise"] = np.array(params["W_noise"])

    return out


def predict_new_pool_joint(
    result: dict,
    x_attr: np.ndarray,
) -> dict:
    """Predict simulator settings for a new pool using joint-fitted mapping.

    In per_pool_noise mode, only cadence and gas are predicted (noise
    coefficients are per-pool and can't generalize). Use shared_noise
    mode for full deployment predictions including noise_coeffs.

    Args:
        result: dict from fit_joint
        x_attr: (K_attr,) pool attribute vector — must match the k_attr
            from training (check result['attr_names'] for the feature order).
            No intercept — just the real features.

    Returns dict with cadence_minutes, gas_usd, and noise_coeffs (shared_noise only).
    """
    x = np.asarray(x_attr)
    log_cadence = result["bias_cad"] + float(x @ result["W_cad"])
    log_gas = result["bias_gas"] + float(x @ result["W_gas"])

    out = {
        "log_cadence": log_cadence,
        "log_gas": log_gas,
        "cadence_minutes": float(np.exp(log_cadence)),
        "gas_usd": float(np.exp(log_gas)),
    }

    if result["mode"] == "shared_noise":
        out["noise_coeffs"] = np.array(
            result["bias_noise"] + x @ result["W_noise"]
        )

    return out
