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
from jax import jit, vmap
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
    return x + stop_gradient(y - x)


def ste_clip(x, lo, hi):
    y = jnp.clip(x, lo, hi)
    # forward: y; backward: identity wrt x
    return ste(x, y)


@partial(jit, static_argnums=(3,))
def _apply_per_asset_bounds(
    weights: jnp.ndarray,
    min_weights: jnp.ndarray,
    max_weights: jnp.ndarray,
    use_ste: bool = False,
) -> jnp.ndarray:
    """
    Apply per-asset min/max bounds and redistribute to ensure sum equals 1.

    This is a pre-processing step applied BEFORE the standard guardrails.
    The result will still go through the uniform minimum_weight guardrails.

    The algorithm:
    1. Clip weights to per-asset [min, max] bounds
    2. If total < 1: distribute deficit proportionally to assets with slack (can grow)
    3. If total > 1: remove surplus proportionally from assets with slack (can shrink)
    4. Final clip and normalise for numerical safety

    Args:
        weights: Input weights, shape (n_assets,).
        min_weights: Minimum weight per asset, shape (n_assets,).
        max_weights: Maximum weight per asset, shape (n_assets,).
        use_ste: Use straight-through estimator for clipping.

    Returns:
        Adjusted weights satisfying bounds and summing to 1.
    """
    # Initial clip to bounds
    if use_ste:
        clipped = ste_clip(weights, min_weights, max_weights)
    else:
        clipped = jnp.clip(weights, min_weights, max_weights)

    total = jnp.sum(clipped)

    # Calculate slack in each direction
    slack_up = max_weights - clipped  # how much each asset can grow
    slack_down = clipped - min_weights  # how much each asset can shrink

    total_slack_up = jnp.sum(slack_up)
    total_slack_down = jnp.sum(slack_down)

    deficit = 1.0 - total  # positive if we need to add weight
    surplus = total - 1.0  # positive if we need to remove weight

    # Redistribute: add to those with room to grow, or remove from those with room to shrink
    adjustment = jnp.where(
        total < 1.0,
        deficit * slack_up / (total_slack_up + 1e-10),
        jnp.where(total > 1.0, -surplus * slack_down / (total_slack_down + 1e-10), 0.0),
    )

    weights_adjusted = clipped + adjustment

    # Final clip for numerical safety
    if use_ste:
        weights_final = ste_clip(weights_adjusted, min_weights, max_weights)
    else:
        weights_final = jnp.clip(weights_adjusted, min_weights, max_weights)

    # Final normalisation (should be very close to 1 already)
    weights_final = weights_final / jnp.sum(weights_final)

    return weights_final


def scale_diff(diff, maximum_change):
    max_val = jnp.max(jnp.abs(diff))
    scale = maximum_change / (max_val + 1e-10)
    needs_scale = max_val > maximum_change
    scaled = jnp.where(needs_scale, diff * scale, diff)
    return scaled


@partial(
    jit,
    static_argnums=(6, 7, 8, 9, 10, 11, 12, 13),
)
def _jax_calc_coarse_weights(
    rule_outputs,
    initial_weights,
    minimum_weight,
    update_rule_parameter_dict,
    min_weights_per_asset,
    max_weights_per_asset,
    max_memory_days,
    chunk_period,
    weight_interpolation_period,
    maximum_change,
    rule_outputs_are_weights=False,
    ste_max_change=False,
    ste_min_max_weight=False,
    use_per_asset_bounds=False,
):
    r"""calc weights from raw weight outputs, and make sure they fall inside
    guard-rails --- sum to 1, are larger than minimum value
    ----------
    rule_outputs : np.ndarray, float64
        A 2-dimenisional numpy array
    initial_weights: np.ndarray, float64
        A 1-dimenisional numpy array
    minimum_weight : float64
        The minimum value (between 0 and 1/n_cols)
    update_rule_parameter_dict : dict
        The update rule parameters
    min_weights_per_asset : np.ndarray or None
        Per-asset minimum weights (applied before uniform guardrails)
    max_weights_per_asset : np.ndarray or None
        Per-asset maximum weights (applied before uniform guardrails)
    max_memory_days : float64
        The maximum memory days
    chunk_period : float64
        The chunk period
    weight_interpolation_period : float64
        The weight interpolation period
    maximum_change : float64
        The maximum change
    rule_outputs_are_weights : bool
        Whether the raw weight outputs are themselves weights
    ste_max_change : bool
        Whether to use ste max change
    ste_min_max_weight : bool
        Whether to use ste min max weight
    use_per_asset_bounds : bool
        Whether to apply per-asset bounds (static flag)

    Returns
    -------
    np.ndarray
        The weight array, same length / shape as ``rule_outputs``

    """
    n = rule_outputs.shape[0] + 1
    n_assets = rule_outputs.shape[1]
    asset_arange = jnp.arange(n_assets)

    cap_lamb = True
    if rule_outputs_are_weights:
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
        # initial_weights = rule_outputs[0]
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
        rule_outputs_are_weights=rule_outputs_are_weights,
        ste_max_change=ste_max_change,
        ste_min_max_weight=ste_min_max_weight,
        max_weights_per_asset=max_weights_per_asset,
        min_weights_per_asset=min_weights_per_asset,
        use_per_asset_bounds=use_per_asset_bounds,
    )

    if rule_outputs_are_weights:
        # Apply guardrails to initial weights
        initial_carry = [rule_outputs[0]]
        guardrailed_init, (
            actual_starts_init,
            scaled_diffs_init,
            target_weights_init,
        ) = _jax_calc_coarse_weight_scan_function(
            initial_carry,
            rule_outputs[0],
            minimum_weight=minimum_weight,
            asset_arange=asset_arange,
            n_assets=n_assets,
            alt_lamb=alt_lamb,
            interpol_num=2,  # interpol_num = 2 for immediate weight change
            maximum_change=maximum_change,
            rule_outputs_are_weights=rule_outputs_are_weights,
            ste_max_change=ste_max_change,
            ste_min_max_weight=ste_min_max_weight,
            max_weights_per_asset=max_weights_per_asset,
            min_weights_per_asset=min_weights_per_asset,
            use_per_asset_bounds=use_per_asset_bounds,
        )
        carry_list_init = [target_weights_init]
    else:
        carry_list_init = [initial_weights]

    _, (actual_starts, scaled_diffs, target_weights) = scan(
        scan_fn, carry_list_init, rule_outputs
    )
    return actual_starts, scaled_diffs, target_weights


@partial(jit, static_argnums=(2, 4, 5))
def calc_fine_weight_output(
    rule_outputs,
    initial_weights,
    run_fingerprint,
    params,
    rule_outputs_are_weights,
    use_per_asset_bounds=False,
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
        rule_outputs (jnp.ndarray): Raw weight outputs from previous calculations.
        initial_weights (jnp.ndarray): Initial weights for the assets.
        run_fingerprint (dict): The settings for this run.
        params (dict): Dictionary containing parameters for the update rule.
        rule_outputs_are_weights (bool): Whether the raw weight outputs are weights or weight changes.
        use_per_asset_bounds (bool): Whether to apply per-asset bounds from params.

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
    ste_max_change = run_fingerprint["ste_max_change"]
    ste_min_max_weight = run_fingerprint["ste_min_max_weight"]
    if minimum_weight == None:
        minimum_weight = 0.1 / n_assets

    # Get per-asset bounds from params (only used if use_per_asset_bounds=True)
    if use_per_asset_bounds:
        min_weights_per_asset = params["min_weights_per_asset"]
        max_weights_per_asset = params["max_weights_per_asset"]
    else:
        # Dummy values - won't be used since use_per_asset_bounds=False
        min_weights_per_asset = jnp.zeros(n_assets)
        max_weights_per_asset = jnp.ones(n_assets)

    actual_starts_cpu, scaled_diffs_cpu, target_weights_cpu = _jax_calc_coarse_weights(
        rule_outputs,
        initial_weights,
        minimum_weight,
        params,
        min_weights_per_asset,
        max_weights_per_asset,
        run_fingerprint["max_memory_days"],
        chunk_period,
        weight_interpolation_period,
        maximum_change,
        rule_outputs_are_weights,
        ste_max_change,
        ste_min_max_weight,
        use_per_asset_bounds,
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
    if rule_outputs_are_weights:
        return weights
    else:
        return jnp.vstack(
            [
                jnp.ones((chunk_period, n_assets), dtype=jnp.float64) * initial_weights,
                weights,
            ]
        )


calc_fine_weight_output_from_weight_changes = jit(
    Partial(
        calc_fine_weight_output,
        rule_outputs_are_weights=False,
        use_per_asset_bounds=False,
    ),
    static_argnums=(2,),
)
calc_fine_weight_output_from_weights = jit(
    Partial(
        calc_fine_weight_output,
        rule_outputs_are_weights=True,
        use_per_asset_bounds=False,
    ),
    static_argnums=(2,),
)

# Bounded versions with per-asset bounds enabled
calc_fine_weight_output_bounded_from_weight_changes = jit(
    Partial(
        calc_fine_weight_output,
        rule_outputs_are_weights=False,
        use_per_asset_bounds=True,
    ),
    static_argnums=(2,),
)
calc_fine_weight_output_bounded_from_weights = jit(
    Partial(
        calc_fine_weight_output,
        rule_outputs_are_weights=True,
        use_per_asset_bounds=True,
    ),
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
    coarse_weights, interpol_num, num, maximum_change, ste_max_change
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
    ste_max_change : bool
        Whether to use ste max change

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
        ste_max_change=ste_max_change,
    )

    carry_list_init = [initial_weights]
    # carry_list_init = [initial_weights, initial_i]
    carry_list_end, [actual_starts, scaled_diffs] = scan(
        scan_fn, carry_list_init, coarse_weights
    )

    return actual_starts, scaled_diffs


@partial(
    jit,
    static_argnums=(2, 3, 5, 6, 7, 8, 9),
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
    ste_max_change,
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

    if ste_max_change:
        scaled_diff = ste(diff, scaled_diff)

    actual_stop = actual_start + scaled_diff * (interpol_num - 1)

    return [actual_stop], [actual_start, scaled_diff]


@partial(
    jit,
    static_argnums=(6, 7, 8, 9, 10, 13),
)
def _jax_calc_coarse_weight_scan_function(
    carry_list,
    rule_outputs,
    minimum_weight,
    asset_arange,
    n_assets,
    alt_lamb,
    interpol_num,
    maximum_change,
    rule_outputs_are_weights=False,
    ste_max_change=False,
    ste_min_max_weight=False,
    max_weights_per_asset=None,
    min_weights_per_asset=None,
    use_per_asset_bounds=False,
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
        interpol_num (int): Number of interpolation steps.
        maximum_change (float): Maximum allowed weight change.
        rule_outputs_are_weights (bool, optional): Whether raw weight outputs represent target weights (True) or weight changes (False). Defaults to False.
        ste_max_change (bool, optional): Use straight-through estimator for max change. Defaults to False.
        ste_min_max_weight (bool, optional): Use straight-through estimator for min/max weight. Defaults to False.
        max_weights_per_asset (ndarray, optional): Per-asset maximum weights (applied before uniform guardrails). Defaults to None.
        min_weights_per_asset (ndarray, optional): Per-asset minimum weights (applied before uniform guardrails). Defaults to None.
        use_per_asset_bounds (bool, optional): Whether to apply per-asset bounds (static flag). Defaults to False.

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
    if rule_outputs_are_weights:
        raw_weights = (
            alt_lamb * prev_actual_position + (1 - alt_lamb) * rule_outputs
        )
    else:
        raw_weights = prev_actual_position + rule_outputs
    ## calc normed weights
    # if i > 5685:
    #     print(i, 'raw w', raw_weights)
    normed_weight_update = raw_weights / jnp.sum(raw_weights)

    # Apply per-asset bounds if enabled (BEFORE uniform guardrails)
    if use_per_asset_bounds:
        normed_weight_update = _apply_per_asset_bounds(
            normed_weight_update,
            min_weights_per_asset,
            max_weights_per_asset,
            ste_min_max_weight,
        )

    # Uniform guardrails (applied AFTER per-asset bounds)
    maximum_weight = 1.0 - (n_assets - 1) * minimum_weight
    ## check values are all above minimum weight
    ## if any values are too small
    idx = normed_weight_update < minimum_weight
    n_less_than_min = jnp.sum(idx)
    idy = normed_weight_update > maximum_weight

    if ste_min_max_weight:
        normed_weight_update = ste_clip(
            normed_weight_update, minimum_weight, maximum_weight
        )
    else:
        normed_weight_update = jnp.clip(
            normed_weight_update, minimum_weight, maximum_weight
        )

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

    # # Straight-through estimator: exact target weights in forward pass, original normed weights for gradients
    # # Forward pass: exact target weights as before
    # clipped_target_weights = target_weights

    # # Backward pass: use original normed weights for gradients
    # # This allows gradient flow even when target weights are constrained
    # target_weights = (
    #     stop_gradient(clipped_target_weights - og_normed_update) + og_normed_update
    # )

    # # Straight-through estimator: exact target weights in forward pass, original normed weights for gradients
    # # Forward pass: exact target weights as before
    # clipped_target_weights = target_weights

    # # Backward pass: use original normed weights for gradients
    # # This allows gradient flow even when target weights are constrained
    # target_weights = (
    #     stop_gradient(clipped_target_weights - og_normed_update) + og_normed_update
    # )

    diff = 1 / (interpol_num - 1) * (target_weights - prev_actual_position)

    # STE max-change: forward caps; backward passes gradients as if unscaled
    scaled_diff = scale_diff(diff, maximum_change)

    if ste_max_change:
        scaled_diff = ste(diff, scaled_diff)

    # Calculate actual position reached after applying both constraints
    actual_position = prev_actual_position + scaled_diff * (interpol_num - 1)

    return [actual_position], (prev_actual_position, scaled_diff, target_weights)
