import numpy as np
import math

def memory_days_to_lamb(memory_days, chunk_period):
    scaled_memory_days = (1440.0 * memory_days / (2.0 * chunk_period)) ** 3 / 6.0

    smd = scaled_memory_days
    smd2 = scaled_memory_days**2
    smd3 = scaled_memory_days**3
    smd4 = scaled_memory_days**4

    numerator_1 = np.cbrt((np.sqrt(3 * (27 * smd4 + 4 * smd3)) - 9 * smd2))
    denominator_1 = (np.cbrt(2) * 3 ** (2.0/3.0) * smd)

    numerator_2 = np.cbrt((2 / 3))
    denominator_2 = numerator_1

    lamb = numerator_1 / denominator_1 - numerator_2 / denominator_2 + 1.0
    return lamb

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1, array[idx - 1]
    else:
        return idx, array[idx]

def force_mem_to_lamb(memory_days, chunk_period):

    base_lamb_range = np.linspace(0.5, 0.9999, int(10e7))
    base_window_range = (
        np.cbrt(6 * base_lamb_range / ((1 - base_lamb_range) ** 3))
        * 2
        * chunk_period
        / 1440
    )
    # base_window_range = ((1 + base_lamb_range)**3/(1-base_lamb_range**2))* chunk_period / 1440
    lamb_idx = [find_nearest(base_window_range, md)[0] for md in memory_days]
    lamb = base_lamb_range[lamb_idx]
    return lamb

def lamb_to_memory_days(lamb, chunk_period=60):
    memory_days = np.cbrt(6 * lamb / ((1 - lamb) ** 3)) * 2 * chunk_period / 1440
    memory_days = np.clip(memory_days, a_min=0.0, a_max=365.0)
    return memory_days

if __name__ == "__main__":
    lamb = force_mem_to_lamb([5],60)
    lamb_ex = memory_days_to_lamb(5, 60)