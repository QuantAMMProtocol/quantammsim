from jax import jit
import jax.numpy as jnp

def _jax_calc_linear_interpolation_block(
    actual_start, scaled_diff, interpol_arange, fine_ones, interpol_num
):
    """
    Calculate linear interpolation weights.

    Args:
        actual_start (float): The starting value of the interpolation range.
        scaled_diff (float): The scaled difference between the start and end values.
        interpol_arange (ndarray): The array of values to interpolate.
        fine_ones (ndarray): An array of ones with the same shape as interpol_arange.
        interpol_num (int): The number of values to interpolate.

    Returns:
        ndarray: The calculated linear interpolation weights.
    """
    linear_interpolation = interpol_arange * scaled_diff + actual_start

    end_weights = linear_interpolation[-1]

    fine_weights = end_weights * fine_ones

    fine_weights = fine_weights.at[0 : interpol_num - 1].set(
        linear_interpolation[0 : interpol_num - 1]
    )
    return fine_weights
