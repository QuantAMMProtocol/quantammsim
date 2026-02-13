"""Approximately optimal (non-linear) weight interpolation.

Blends a linear ramp with a geometric interpolation path between coarse
weight updates, following the method in "Optimal Rebalancing in Dynamic AMMs"
(Willetts & Harrington, 2024, https://arxiv.org/abs/2403.18737). The combined
path reduces arbitrage loss compared to naive linear interpolation.
"""
import jax.numpy as jnp


def _jax_calc_approx_optimal_interpolation_block(
    actual_start, scaled_diff, interpol_arange, fine_ones, interpol_num
):
    """
    Calculate approximately optimal interpolation weights for a given range.
    This uses the method described in the paper "Optimal Rebalancing in Dynamic AMMs",
    Willetts & Harrington (2024), available at https://arxiv.org/abs/2403.18737.

    Args:
        actual_start (float): The starting value of the interpolation range.
        scaled_diff (float): The scaled difference between the starting and ending values of the interpolation range.
        interpol_arange (ndarray): The array of values representing the interpolation range.
        fine_ones (ndarray): An array of ones with the same shape as interpol_arange.
        interpol_num (int): The number of interpolation points.

    Returns:
        ndarray: The approximately optimal interpolation weights.

    """

    linear_interpolation = interpol_arange * scaled_diff + actual_start

    end_weights = linear_interpolation[-1]

    unnormalised_geometric_interpolation = actual_start * (
        (end_weights / actual_start) ** (interpol_arange / interpol_num)
    )

    approximately_optimal_weights = (
        linear_interpolation + unnormalised_geometric_interpolation
    )
    approximately_optimal_weights = approximately_optimal_weights / jnp.sum(
        approximately_optimal_weights, axis=-1, keepdims=True
    )
    fine_weights = end_weights * fine_ones

    fine_weights = fine_weights.at[0 : interpol_num - 1].set(
        approximately_optimal_weights[0 : interpol_num - 1]
    )
    return fine_weights
