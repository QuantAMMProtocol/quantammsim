"""Tests for Feature 4: Stratified sampling + batch_size floor."""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np


def _call_get_indices(sample_method, batch_size=8, len_prices=50000, bout_length=4321, start_index=0):
    """Helper to call get_indices with given sample_method."""
    from quantammsim.core_simulator.windowing_utils import get_indices

    key = random.PRNGKey(42)
    optimisation_settings = {
        "batch_size": batch_size,
        "training_data_kind": "historic",
        "sample_method": sample_method,
    }
    return get_indices(start_index, bout_length, len_prices, key, optimisation_settings)


def test_stratified_one_index_per_segment():
    """batch_size=8, range divided into 8 segments, one index in each."""
    batch_size = 8
    len_prices = 50000
    bout_length = 4321
    start_index = 0
    range_ = len_prices - bout_length - start_index

    start_indexes, _ = _call_get_indices(
        "stratified", batch_size=batch_size,
        len_prices=len_prices, bout_length=bout_length, start_index=start_index,
    )

    # Use the same linspace boundaries as the implementation
    seg_boundaries = np.linspace(0, range_, batch_size + 1).astype(np.int64)
    for i in range(batch_size):
        idx = int(start_indexes[i, 0])
        seg_start = start_index + int(seg_boundaries[i])
        seg_end = start_index + int(seg_boundaries[i + 1])
        assert seg_start <= idx < seg_end, (
            f"Index {i}: {idx} not in segment [{seg_start}, {seg_end})"
        )


def test_stratified_covers_full_period():
    """Min/max indices span ~full training range."""
    batch_size = 8
    len_prices = 50000
    bout_length = 4321
    start_index = 0
    range_ = len_prices - bout_length - start_index

    start_indexes, _ = _call_get_indices(
        "stratified", batch_size=batch_size,
        len_prices=len_prices, bout_length=bout_length, start_index=start_index,
    )

    min_idx = int(start_indexes[:, 0].min())
    max_idx = int(start_indexes[:, 0].max())
    span = max_idx - min_idx
    # Should span at least 70% of the range
    assert span >= 0.7 * range_, f"Span {span} too small vs range {range_}"


def test_stratified_respects_bounds():
    """No index exceeds len_prices - bout_length."""
    batch_size = 8
    len_prices = 50000
    bout_length = 4321
    start_index = 1000

    start_indexes, _ = _call_get_indices(
        "stratified", batch_size=batch_size,
        len_prices=len_prices, bout_length=bout_length, start_index=start_index,
    )

    max_valid = len_prices - bout_length
    assert jnp.all(start_indexes[:, 0] >= start_index), "Index below start_index"
    assert jnp.all(start_indexes[:, 0] < max_valid), f"Index exceeds max valid {max_valid}"


def test_stratified_correct_shape():
    """Shape (batch_size, 2) for historic."""
    batch_size = 8
    start_indexes, _ = _call_get_indices("stratified", batch_size=batch_size)
    assert start_indexes.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {start_indexes.shape}"


def test_uniform_unchanged():
    """sample_method='uniform' identical to before."""
    start_indexes, _ = _call_get_indices("uniform", batch_size=8)
    assert start_indexes.shape == (8, 2)
    # All indices should be in valid range
    assert jnp.all(start_indexes[:, 0] >= 0)


def test_batch_size_floor_in_tuner():
    """Tuner search space has batch_size low=2."""
    from quantammsim.runners.hyperparam_tuner import HyperparamSpace

    space = HyperparamSpace.create()
    assert "batch_size" in space.params
    assert space.params["batch_size"]["low"] == 2, (
        f"Expected batch_size low=2, got {space.params['batch_size']['low']}"
    )
