# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
from jax import default_backend
from jax import local_device_count, devices

DEFAULT_BACKEND = default_backend()
CPU_DEVICE = devices("cpu")[0]
if DEFAULT_BACKEND != "cpu":
    GPU_DEVICE = devices("gpu")[0]
    config.update("jax_platform_name", "gpu")
else:
    GPU_DEVICE = devices("cpu")[0]
    config.update("jax_platform_name", "cpu")


import jax.numpy as jnp
from jax import jit, vmap, lax
from jax import devices, device_put
from jax.tree_util import Partial
from jax.lax import scan, stop_gradient

import numpy as np

# from simulationRunDto import LiquidityPoolCoinDto, SimulationResultTimestepDto
from functools import partial

np.seterr(all="raise")
np.seterr(under="print")

# TODO above is all from jax utils, tidy up required

from quantammsim.pools.G3M.quantamm.weight_calculations.linear_interpolation import (
    _jax_calc_linear_interpolation_block,
)
from quantammsim.pools.G3M.quantamm.weight_calculations.non_linear_interpolation import (
    _jax_calc_approx_optimal_interpolation_block,
)
from quantammsim.core_simulator.param_utils import (
    calc_alt_lamb,
    memory_days_to_lamb,
    jax_memory_days_to_lamb,
)

def ste(x, y):
    # forward: y; backward: identity wrt x
    return x + lax.stop_gradient(y - x)

def ste_clip(x, lo, hi):
    y = jnp.clip(x, lo, hi)
    # forward: y; backward: identity wrt x
    return ste(x, y)

def scale_diff(diff, maximum_change):
    max_val = jnp.max(jnp.abs(diff))
    scale = maximum_change / (max_val + 1e-10)
    needs_scale = max_val > maximum_change
    scaled = jnp.where(needs_scale, diff * scale, diff)
    return scaled

@partial(
    jit,
    static_argnums=(4,5,6,7,8,),
)
def _jax_calc_coarse_weights(
    raw_weight_outputs,
    initial_weights,
    minimum_weight,
    update_rule_parameter_dict,
    max_memory_days,
    chunk_period,
    weight_interpolation_period,
    maximum_change,
    raw_weight_outputs_are_themselves_weights=False,
):
    r"""calc weights from raw weight outputs, and make sure they fall inside
    guard-rails --- sum to 1, are larger than minimum value
    ----------
    raw_weight_outputs : np.ndarray, float64
        A 2-dimenisional numpy array
    initial_weights: np.ndarray, float64
        A 1-dimenisional numpy array
    minimum_weight : float64
        The minimum value (between 0 and 1/n_cols)

    Returns
    -------
    np.ndarray
        The weight array, same length / shape as ``raw_weight_outputs``

    """
    n = raw_weight_outputs.shape[0] + 1
    n_assets = raw_weight_outputs.shape[1]
    asset_arange = jnp.arange(n_assets)

    cap_lamb = True
    if raw_weight_outputs_are_themselves_weights:
        # Determine which parameterization is being used
        # allow for direct memory_days parameterization
        if "memory_days_2" in update_rule_parameter_dict:
            # Direct memory_days parameterization
            memory_days = update_rule_parameter_dict["memory_days_2"]
            alt_lamb = jax_memory_days_to_lamb(memory_days, chunk_period)
        else:
            # Original logit_lamb parameterization
            alt_lamb = calc_alt_lamb(update_rule_parameter_dict)
        if cap_lamb:
            max_lamb = memory_days_to_lamb(max_memory_days, chunk_period)
            capped_alt_lamb = ste_clip(alt_lamb, lo=0.0, hi=max_lamb)
            alt_lamb = capped_alt_lamb
        # initial_weights = raw_weight_outputs[0]
    else:
        alt_lamb = None

    scan_fn = Partial(
        _jax_calc_coarse_weight_scan_function,
        minimum_weight=minimum_weight,
        asset_arange=asset_arange,
        n_assets=n_assets,
        alt_lamb=alt_lamb,
        interpol_num=weight_interpolation_period + 1,
        maximum_change=maximum_change,
        raw_weight_outputs_are_themselves_weights=raw_weight_outputs_are_themselves_weights,
    )

    if raw_weight_outputs_are_themselves_weights:
        # Apply guardrails to initial weights
        initial_carry = [raw_weight_outputs[0]]
        guardrailed_init, (actual_starts_init, scaled_diffs_init, target_weights_init) = (
            _jax_calc_coarse_weight_scan_function(
                initial_carry,
                raw_weight_outputs[0],
                minimum_weight=minimum_weight,
                asset_arange=asset_arange,
                n_assets=n_assets,
                alt_lamb=alt_lamb,
                interpol_num=2,  # interpol_num = 2 for immediate weight change
                maximum_change=maximum_change,
                raw_weight_outputs_are_themselves_weights=raw_weight_outputs_are_themselves_weights,
            )
        )
        carry_list_init = [target_weights_init]
    else:
        carry_list_init = [initial_weights]

    _, (actual_starts, scaled_diffs, target_weights) = scan(scan_fn, carry_list_init, raw_weight_outputs)
    return actual_starts, scaled_diffs, target_weights


@partial(jit, static_argnums=(2, 4))
def calc_fine_weight_output(
    raw_weight_outputs,
    initial_weights,
    run_fingerprint,
    params,
    raw_weight_outputs_are_themselves_weights,
):
    """
    Calculate fine weight outputs based on raw weight outputs and various parameters.

    This function performs the following steps:
    1. Determines if the calculation is for minimum variance or volatility targeting.
    2. Calculates coarse weights using the raw weight outputs.
    3. Computes actual starts and scaled differences for fine weight interpolation.
    4. Transfers data between CPU and GPU devices.
    5. Calculates fine weights using either linear or non-linear interpolation.

    Args:
        raw_weight_outputs (jnp.ndarray): Raw weight outputs from previous calculations.
        initial_weights (jnp.ndarray): Initial weights for the assets.
        run_fingerprint (dict): The settings for this run.
        params (dict): Dictionary containing parameters for the update rule.
        raw_weight_outputs_are_themselves_weights (bool): Whether the raw weight outputs are weights or weight changes.

    Returns:
        jnp.ndarray: Fine weights calculated based on the input parameters and chosen method.

    Note:
        This function uses JAX for GPU acceleration and supports both linear and non-linear
        interpolation methods for fine weight calculation.
    """

    weight_interpolation_period = run_fingerprint["weight_interpolation_period"]
    chunk_period = run_fingerprint["chunk_period"]
    maximum_change = run_fingerprint["maximum_change"]
    weight_interpolation_method = run_fingerprint["weight_interpolation_method"]
    minimum_weight = run_fingerprint.get("minimum_weight")
    n_assets = run_fingerprint["n_assets"]
    if minimum_weight == None:
        minimum_weight = 0.1 / n_assets

    actual_starts_cpu, scaled_diffs_cpu, target_weights_cpu = _jax_calc_coarse_weights(
        raw_weight_outputs,
        initial_weights,
        minimum_weight,
        params,
        run_fingerprint["max_memory_days"],
        chunk_period,
        weight_interpolation_period,
        maximum_change,
        raw_weight_outputs_are_themselves_weights,
    )

    scaled_diffs_gpu = device_put(scaled_diffs_cpu, GPU_DEVICE)
    actual_starts_gpu = device_put(actual_starts_cpu, GPU_DEVICE)

    weights = _jax_fine_weights_from_actual_starts_and_diffs(
        actual_starts_gpu,
        scaled_diffs_gpu,
        initial_weights,
        interpol_num=weight_interpolation_period + 1,
        num=chunk_period + 1,
        maximum_change=maximum_change,
        method=weight_interpolation_method,
    )
    if raw_weight_outputs_are_themselves_weights:
        return weights
    else:
        return jnp.vstack(
            [
                jnp.ones((chunk_period, n_assets), dtype=jnp.float64) * initial_weights,
                weights,
            ]
        )


calc_fine_weight_output_from_weight_changes = jit(
    Partial(calc_fine_weight_output, raw_weight_outputs_are_themselves_weights=False),
    static_argnums=(2,),
)
calc_fine_weight_output_from_weights = jit(
    Partial(calc_fine_weight_output, raw_weight_outputs_are_themselves_weights=True),
    static_argnums=(2,),
)


@partial(
    jit,
    static_argnums=(3, 4, 6),
)
def _jax_fine_weights_from_actual_starts_and_diffs(
    actual_starts,
    scaled_diffs,
    intial_weights,
    interpol_num,
    num,
    maximum_change,
    method="linear",
):
    r"""calc fine weights from coarse weight changes
    ----------
    coarse_weights : jnp.ndarray, float64
        A 2-dimenisional jax numpy array
    interpol_num : float64
        How many timesteps to interpolate over
    num : float64
        How many timesteps to map one coarse interval to
    maximum_change : float64
        maximum scalar change in w, for step sizes
    Returns
    -------
    jnp.ndarray
        The weight array, same length / shape as ``raw_weight_changes``

    """
    initial_weights = intial_weights
    # initial_i = 0
    n_assets = len(intial_weights)

    interpol_arange = jnp.expand_dims(jnp.arange(start=0, stop=interpol_num), 1)
    fine_ones = jnp.ones((num - 1, n_assets))
    array_of_trues = jnp.ones((n_assets,), dtype=bool)

    if method == "linear":
        partial_jax_calc_interpolation_block = Partial(
            _jax_calc_linear_interpolation_block,
            interpol_arange=interpol_arange,
            fine_ones=fine_ones,
            interpol_num=interpol_num,
        )
    elif method == "approx_optimal":
        partial_jax_calc_interpolation_block = Partial(
            _jax_calc_approx_optimal_interpolation_block,
            interpol_arange=interpol_arange,
            fine_ones=fine_ones,
            interpol_num=interpol_num,
        )
    else:
        raise ValueError(f"Invalid interpolation method: {method}")

    partial_jax_calc_interpolation_blocks = vmap(
        partial_jax_calc_interpolation_block, in_axes=[0, 0]
    )

    jit_jax_calc_interpolation_blocks = jit(partial_jax_calc_interpolation_blocks)

    fine_weights_array = jit_jax_calc_interpolation_blocks(actual_starts, scaled_diffs)

    return fine_weights_array.reshape(-1, n_assets)


@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def _jax_fine_weights_end_from_coarse_weights(
    coarse_weights, interpol_num, num, maximum_change
):
    r"""calc fine weights from coarse weight changes
    ----------
    coarse_weights : jnp.ndarray, float64
        A 2-dimenisional jax numpy array
    interpol_num : float64
        How many timesteps to interpolate over
    num : float64
        How many timesteps to map one coarse interval to
    maximum_change : float64
        maximum scalar change in w, for step sizes
    Returns
    -------
    jnp.ndarray
        The weight array, same length / shape as ``raw_weight_changes``

    """
    initial_weights = coarse_weights[0]
    # initial_i = 0
    n_assets = coarse_weights.shape[1]

    interpol_arange = jnp.expand_dims(jnp.arange(start=0, stop=interpol_num), 1)
    fine_ones = jnp.ones((num - 1, n_assets))

    array_of_trues = jnp.ones((n_assets,), dtype=bool)

    scan_fn = Partial(
        _jax_calc_fine_weight_ends_only_scan_function,
        num=num,
        interpol_num=interpol_num,
        interpol_arange=interpol_arange,
        maximum_change=maximum_change,
        n_assets=n_assets,
        fine_ones=fine_ones,
        array_of_trues=array_of_trues,
    )

    carry_list_init = [initial_weights]
    # carry_list_init = [initial_weights, initial_i]
    carry_list_end, [actual_starts, scaled_diffs] = scan(
        scan_fn, carry_list_init, coarse_weights
    )

    return actual_starts, scaled_diffs


@partial(
    jit,
    static_argnums=(
        2,
        3,
    ),
)
def _jax_calc_fine_weight_ends_only_scan_function(
    carry_list,
    coarse_weights,
    num,
    interpol_num,
    interpol_arange,
    maximum_change,
    n_assets,
    fine_ones,
    array_of_trues,
):
    """
    Calculate the fine weights using a scan function.

    Args:
        carry_list (list): List of carry variables needed to calculate the weights.
        coarse_weights (float): The coarse weight we are aiming to get to.
        num (int): Number of steps.
        interpol_num (int): Number of interpolation steps.
        interpol_arange (array): Array of interpolation steps.
        maximum_change (float): Maximum allowed change in weights.
        n_assets (int): Number of assets.
        fine_ones (array): Array of ones.
        array_of_trues (array): Array of boolean values.

    Returns:
        tuple: A tuple containing the actual stop weight and the updated carry variables.
    """

    # carry_list constains the needed 'state variables'
    # needed to calculate the weights

    # carry_list[0] is the actual value reached
    # last loop (if steps were too big last time,
    # such that we hit the 'maximum_change' limit
    # we won't have reached the actual goal)

    actual_start = carry_list[0]

    # carry_list[1] is the current loop variable
    # might be useful

    # stop is the coarse weight we are aiming to get to

    stop = coarse_weights

    diff = 1 / (interpol_num - 1) * (stop - actual_start)

    # STE max-change: forward caps; backward treats as identity for grads
    scaled_diff = scale_diff(diff, maximum_change)

    if run_fingerprint["ste_max_change"]:
        scaled_diff = ste(diff, scaled_diff)

    actual_stop = actual_start + scaled_diff * (interpol_num - 1)

    return [actual_stop], [actual_start, scaled_diff]


@partial(
    jit,
    static_argnums=(6,7,8),
)
def _jax_calc_coarse_weight_scan_function(
    carry_list,
    raw_weight_outputs,
    minimum_weight,
    asset_arange,
    n_assets,
    alt_lamb,
    interpol_num,
    maximum_change,
    raw_weight_outputs_are_themselves_weights,
):
    """
    Calculate the coarse weights for the AMM simulator.

    Args:
        carry_list (list): List of state variables needed to calculate the weights.
        raw_weight_changes (ndarray): Array of raw weight changes.
        minimum_weight (float): Minimum weight value.
        asset_arange (ndarray): Array of asset indices.
        n_assets (int): Number of assets.
        alt_lamb (float): Alternative lambda value.
        raw_weight_outputs_are_themselves_weights (bool): Whether raw weight outputs represent target weights (True) or weight changes (False).

    Returns:
        list: List containing the final weights.
        ndarray: Array of final weights.
    """

    # carry_list constains the needed 'state variables'
    # needed to calculate the weights

    # carry_list[0] is the previous weight value

    prev_actual_position = carry_list[0]

    ## calc raw weight, previous weight plus delta
    ## note that the ith-indexed raw_weight_change
    ## depends on the ith-indexed oracle/price value
    ## so to calculate the ith-raw weight, we take the
    ## (i-1)th weight and add the (i-1)th raw weight change
    ## it is at the (i-1)th moment in time we calculate the
    ## weights we WISH to have at the ith moment, using
    ## all information available at the (i-1)th moment
    if raw_weight_outputs_are_themselves_weights:
        raw_weights = alt_lamb * prev_actual_position + (1 - alt_lamb) * raw_weight_outputs
    else:
        raw_weights = prev_actual_position + raw_weight_outputs
    ## calc normed weights
    # if i > 5685:
    #     print(i, 'raw w', raw_weights)
    normed_weight_update = raw_weights / jnp.sum(raw_weights)

    maximum_weight = 1.0 - (n_assets - 1) * minimum_weight
    ## check values are all above minimum weight
    ## if any values are too small
    idx = normed_weight_update < minimum_weight
    n_less_than_min = jnp.sum(idx)
    idy = normed_weight_update > maximum_weight

    if run_fingerprint["ste_min_max_weight"]:
        normed_weight_update = ste_clip(normed_weight_update, minimum_weight, maximum_weight)
    else:
        normed_weight_update = jnp.clip(normed_weight_update, minimum_weight, maximum_weight)

    # calculate 'left over' weight, 1 - n * epsilon
    remaining_weight = 1 - n_less_than_min * minimum_weight
    ## now distribute this 'left over' weight to other weight-slots
    # in proportion to those other weights
    other_weights = jnp.where(~idx, normed_weight_update, 0.0)
    sum_of_other_weights = jnp.sum(other_weights)
    normed_weight_update = jnp.where(
        ~idx,
        normed_weight_update * remaining_weight / sum_of_other_weights,
        normed_weight_update,
    )
    target_weights = normed_weight_update

    ## if rounding means weights dont sum to one, do a little tweak:

    raw_idx = jnp.argmax(target_weights)
    idx = raw_idx == asset_arange
    corrected_weights = jnp.where(
        idx, target_weights - jnp.sum(target_weights) + 1.0, target_weights
    )

    # note that argmax is not differentiable, so we take the
    # gradients of the uncorrected values and the values from the
    # corrected values

    target_weights = target_weights + stop_gradient(corrected_weights - target_weights)

    diff = 1 / (interpol_num - 1) * (target_weights - prev_actual_position)

    # STE max-change: forward caps; backward passes gradients as if unscaled
    scaled_diff = scale_diff(diff, maximum_change)

    if run_fingerprint["ste_max_change"]:
        scaled_diff = ste(diff, scaled_diff)

    # Calculate actual position reached after applying both constraints
    actual_position = prev_actual_position + scaled_diff * (interpol_num - 1)

    return [actual_position], (prev_actual_position, scaled_diff, target_weights)
