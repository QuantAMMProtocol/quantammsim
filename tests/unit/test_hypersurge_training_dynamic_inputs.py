import numpy as np
import pandas as pd

from quantammsim.core_simulator.dynamic_inputs import DynamicInputFrames
from quantammsim.pools.hypersurge_utils import HYPERSURGE_PARAM_KEYS
from quantammsim.runners.jax_runners import train_on_historic_data


def _synthetic_price_data(start="2023-01-01 00:00:00", periods=24 * 60 * 5):
    date_index = pd.date_range(start=start, periods=periods, freq="min")
    t = np.arange(periods, dtype=np.float64)
    price_data = pd.DataFrame(
        {
            "close_BTC": 20_000.0 * np.exp(0.00008 * t + 0.015 * np.sin(t / 240.0)),
            "close_ETH": 1_500.0 * np.exp(0.00005 * t + 0.02 * np.cos(t / 360.0)),
        },
        index=(date_index.view("int64") // 10**6),
    )
    oracle_prices = pd.DataFrame(
        {
            "unix": price_data.index.to_numpy(),
            "BTC": price_data["close_BTC"].to_numpy(),
            "ETH": price_data["close_ETH"].to_numpy(),
        }
    )
    return price_data, oracle_prices


def _training_fingerprint(rule):
    return {
        "tokens": ["BTC", "ETH"],
        "rule": rule,
        "startDateString": "2023-01-01 00:00:00",
        "endDateString": "2023-01-04 00:00:00",
        "endTestDateString": "2023-01-05 00:00:00",
        "chunk_period": 60,
        "weight_interpolation_period": 60,
        "bout_offset": 120,
        "initial_pool_value": 1_000_000.0,
        "fees": 0.003,
        "arb_fees": 0.0,
        "gas_cost": 0.0,
        "arb_frequency": 1,
        "do_arb": True,
        "return_val": "returns",
        "use_fused_reserves": False,
        "optimisation_settings": {
            "method": "gradient_descent",
            "base_lr": 0.01,
            "optimiser": "adam",
            "batch_size": 1,
            "n_iterations": 1,
            "n_parameter_sets": 1,
            "training_data_kind": "historic",
            "sample_method": "uniform",
            "initial_random_key": 0,
            "n_cycles": 1,
            "val_fraction": 0.0,
            "early_stopping": False,
            "decay_lr_ratio": 0.8,
            "decay_lr_plateau": 100,
            "min_lr": 1e-6,
        },
    }


def test_hypersurge_balancer_training_accepts_oracle_frames():
    price_data, oracle_prices = _synthetic_price_data()
    params, metadata = train_on_historic_data(
        _training_fingerprint("balancer_hypersurge"),
        price_data=price_data,
        dynamic_input_frames=DynamicInputFrames(oracle_prices=oracle_prices),
        verbose=False,
        force_init=True,
        return_training_metadata=True,
        iterations_per_print=999999,
    )

    assert np.isfinite(metadata["final_objective"])
    for key in HYPERSURGE_PARAM_KEYS:
        assert key in params


def test_hypersurge_reclamm_training_accepts_oracle_frames():
    price_data, oracle_prices = _synthetic_price_data()
    fingerprint = _training_fingerprint("reclamm_hypersurge")
    fingerprint.update(
        {
            "initial_price_ratio": 2.0,
            "initial_centeredness_margin": 0.25,
            "initial_daily_price_shift_base": 1.0 - 1.0 / 124000.0,
            "reclamm_interpolation_method": "geometric",
        }
    )

    params, metadata = train_on_historic_data(
        fingerprint,
        price_data=price_data,
        dynamic_input_frames=DynamicInputFrames(oracle_prices=oracle_prices),
        verbose=False,
        force_init=True,
        return_training_metadata=True,
        iterations_per_print=999999,
    )

    assert np.isfinite(metadata["final_objective"])
    assert "price_ratio" in params
    for key in HYPERSURGE_PARAM_KEYS:
        assert key in params
