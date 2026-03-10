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
) -> JointData:
    """Build batched JAX arrays from matched pool data.

    Args:
        matched: dict from match_grids_to_panel
        drop_chain_dummies: if True, remove chain_* columns from attributes
            (reduces feature count for small n)

    Returns:
        JointData with per-pool JAX arrays and shared attribute matrix.
    """
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
        x_obs = build_x_obs(panel)
        y_obs = panel["log_volume"].values.astype(float)

        pool_data.append({
            "coeffs": entry["coeffs"],
            "x_obs": jnp.array(x_obs),
            "y_obs": jnp.array(y_obs),
            "day_indices": jnp.array(entry["day_indices"]),
        })

    return JointData(
        pool_data=pool_data,
        x_attr=jnp.array(X_attr),
        pool_ids=pool_ids,
        attr_names=attr_names,
    )


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


def unpack_joint_params(
    flat: jnp.ndarray, config: dict
) -> dict:
    """Unpack flat array to structured params.

    config must have: k_attr, n_pools, mode
    """
    k_attr = config["k_attr"]
    mode = config["mode"]

    bias_cad = flat[0]
    bias_gas = flat[1]
    W_cad = flat[2:2 + k_attr]
    W_gas = flat[2 + k_attr:2 + 2 * k_attr]
    rest = flat[2 + 2 * k_attr:]

    if mode == "per_pool_noise":
        n_pools = config["n_pools"]
        noise_coeffs = rest.reshape(n_pools, K_OBS)
        return {
            "bias_cad": bias_cad, "bias_gas": bias_gas,
            "W_cad": W_cad, "W_gas": W_gas,
            "noise_coeffs": noise_coeffs,
        }
    else:  # shared_noise
        # noise_params: (1 + k_attr, K_OBS) — row 0 is bias
        W_noise_full = rest.reshape(1 + k_attr, K_OBS)
        return {
            "bias_cad": bias_cad, "bias_gas": bias_gas,
            "W_cad": W_cad, "W_gas": W_gas,
            "bias_noise": W_noise_full[0],    # (K_OBS,)
            "W_noise": W_noise_full[1:],      # (k_attr, K_OBS)
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
    """
    coeffs = pool_data_i["coeffs"]
    x_obs = pool_data_i["x_obs"]
    y_obs = pool_data_i["y_obs"]
    day_indices = pool_data_i["day_indices"]
    mode = config["mode"]
    i = pool_idx

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
):
    """Create per-pool JIT'd loss functions and a Python-level aggregator.

    Each pool gets its own small JIT'd computation graph (compiled
    independently), avoiding a massive unrolled trace. The outer
    function sums per-pool losses in Python and adds regularization.

    Loss averages over pools (not observations), giving equal weight
    to each pool regardless of observation count.

    L2 regularization is applied to W_cad and W_gas only (not biases).

    Args:
        jdata: JointData from prepare_joint_data
        mode: "per_pool_noise" or "shared_noise"
        alpha_cad: L2 regularization on W_cad
        alpha_gas: L2 regularization on W_gas

    Returns:
        loss_fn(params_flat) -> scalar loss
    """
    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    config = {"k_attr": k_attr, "n_pools": n_pools, "mode": mode}

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
        reg = alpha_cad * jnp.sum(params["W_cad"] ** 2) + \
              alpha_gas * jnp.sum(params["W_gas"] ** 2)
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
) -> jnp.ndarray:
    """Create initial parameter vector.

    If init_from_option_c is provided, warm-start from Option C per-pool fits:
    - bias_cad, W_cad from OLS on per-pool fitted log_cadence
    - bias_gas, W_gas from OLS on per-pool fitted log_gas
    - noise_coeffs from per-pool fits

    Otherwise, use defaults: cadence=12min, gas=$1 for all pools.
    """
    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    x_attr_np = np.array(jdata.x_attr)

    if init_from_option_c is not None:
        pool_ids = jdata.pool_ids
        # Filter out pools with NaN losses from warm start
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
        log_gases = np.array([valid[p]["log_gas"] for p in pool_ids])
        noise_all = np.array([valid[p]["noise_coeffs"] for p in pool_ids])

        # OLS with intercept: X_aug = [1, x_attr]; solve for [bias, W]
        X_aug = np.column_stack([np.ones(n_pools), x_attr_np])
        cad_params, _, _, _ = np.linalg.lstsq(X_aug, log_cads, rcond=None)
        gas_params, _, _, _ = np.linalg.lstsq(X_aug, log_gases, rcond=None)
        bias_cad, W_cad = cad_params[0], cad_params[1:]
        bias_gas, W_gas = gas_params[0], gas_params[1:]

        if mode == "per_pool_noise":
            noise_params = noise_all
        else:
            # OLS with intercept for noise mapping
            noise_aug, _, _, _ = np.linalg.lstsq(X_aug, noise_all, rcond=None)
            # noise_aug: (1+k_attr, K_OBS) — row 0 is bias
            noise_params = noise_aug
    else:
        # Default: all pools get cadence=12min, gas=$1
        bias_cad = np.log(12.0)
        bias_gas = np.log(1.0)  # = 0.0
        W_cad = np.zeros(k_attr)
        W_gas = np.zeros(k_attr)

        if mode == "per_pool_noise":
            # Initialize noise via OLS per pool
            noise_params = np.zeros((n_pools, K_OBS))
            for i, pd in enumerate(jdata.pool_data):
                x_obs_np = np.array(pd["x_obs"])
                y_obs_np = np.array(pd["y_obs"])
                c, _, _, _ = np.linalg.lstsq(x_obs_np, y_obs_np, rcond=None)
                noise_params[i] = c
        else:
            # Initialize shared noise from pooled OLS
            all_x = np.vstack([np.array(pd["x_obs"]) for pd in jdata.pool_data])
            all_y = np.concatenate([np.array(pd["y_obs"]) for pd in jdata.pool_data])
            c, _, _, _ = np.linalg.lstsq(all_x, all_y, rcond=None)
            # (1+k_attr, K_OBS): bias row + zero weight rows
            noise_params = np.zeros((1 + k_attr, K_OBS))
            noise_params[0, :] = c

    return pack_joint_params(
        float(bias_cad),
        float(bias_gas),
        jnp.array(W_cad),
        jnp.array(W_gas),
        jnp.array(noise_params),
    )


def _make_bounds(k_attr, n_pools, mode):
    """Build scipy bounds for joint params."""
    # bias_cad, bias_gas: unbounded
    bounds = [(None, None)] * 2
    # W_cad, W_gas: unbounded
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
) -> dict:
    """Joint end-to-end optimization across all pools.

    Args:
        matched: dict from match_grids_to_panel
        mode: "per_pool_noise" or "shared_noise"
        init_from_option_c: Optional Option C results for warm start.
            Pools with NaN losses are silently excluded from warm start.
        maxiter: max L-BFGS-B iterations
        alpha_cad: L2 regularization on W_cad (not bias)
        alpha_gas: L2 regularization on W_gas (not bias)
        drop_chain_dummies: if True, remove chain_* columns from attributes

    Returns dict with fitted params and diagnostics.
    """
    jdata = prepare_joint_data(matched, drop_chain_dummies=drop_chain_dummies)
    loss_fn = make_joint_loss_fn(jdata, mode=mode,
                                  alpha_cad=alpha_cad, alpha_gas=alpha_gas)
    init = make_initial_joint_params(jdata, mode=mode,
                                     init_from_option_c=init_from_option_c)

    n_pools = len(jdata.pool_data)
    k_attr = jdata.x_attr.shape[1]
    config = {"k_attr": k_attr, "n_pools": n_pools, "mode": mode}
    bounds = _make_bounds(k_attr, n_pools, mode)

    # Per-pool value_and_grad — each pool has its own small JIT graph
    pool_vg_fns = loss_fn._pool_val_and_grad_fns

    # Indices for W_cad and W_gas in the flat param vector (for reg gradient)
    w_cad_start = 2
    w_cad_end = 2 + k_attr
    w_gas_start = 2 + k_attr
    w_gas_end = 2 + 2 * k_attr

    def scipy_wrapper(params_np):
        params_j = jnp.array(params_np)

        # Sum per-pool losses and gradients
        total_val = 0.0
        total_grad = jnp.zeros_like(params_j)
        for vg_fn in pool_vg_fns:
            v, g = vg_fn(params_j)
            total_val += float(v)
            total_grad = total_grad + g

        data_loss = total_val / n_pools
        data_grad = total_grad / n_pools

        # Regularization on W_cad and W_gas (not biases)
        reg = (alpha_cad * float(jnp.sum(params_j[w_cad_start:w_cad_end] ** 2)) +
               alpha_gas * float(jnp.sum(params_j[w_gas_start:w_gas_end] ** 2)))

        reg_grad = jnp.zeros_like(params_j)
        reg_grad = reg_grad.at[w_cad_start:w_cad_end].set(
            2 * alpha_cad * params_j[w_cad_start:w_cad_end])
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
        "bias_gas": float(params["bias_gas"]),
        "W_cad": np.array(params["W_cad"]),
        "W_gas": np.array(params["W_gas"]),
        "loss": float(result.fun),
        "converged": result.success,
        "mode": mode,
        "k_attr": k_attr,
        "pool_ids": jdata.pool_ids,
        "attr_names": jdata.attr_names,
    }

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
