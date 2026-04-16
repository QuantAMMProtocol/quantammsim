"""Demo training entry points for HyperSurge pool variants.

Compared with [demo_train.py](./demo_train.py), HyperSurge training needs:
1. A HyperSurge-enabled pool rule.
2. HyperSurge fee-curve initial values in the run fingerprint.
3. For reCLAMM, the usual range-shape parameters as well.
4. Optionally, a separate oracle-price frame. If omitted, both pool
   implementations fall back to the traded price series as the reference.

`price_data` and `oracle_prices` intentionally have different shapes:
- `price_data`: parquet-like price frame with `close_<TOKEN>` columns and a
  unix-ms or datetime index.
- `oracle_prices`: minute-level frame with a `unix` column plus one column per
  token in pool order.
"""

from copy import deepcopy

import numpy as np
import pandas as pd

from quantammsim.core_simulator.dynamic_inputs import DynamicInputFrames
from quantammsim.runners.jax_runners import train_on_historic_data


DEFAULT_FINGERPRINT = {
    "startDateString": "2023-01-01 00:00:00",
    "endDateString": "2023-03-01 00:00:00",
    "endTestDateString": "2023-04-01 00:00:00",
    "chunk_period": 1440,
    "weight_interpolation_period": 1440,
    "bout_offset": 24 * 60 * 7,
    "initial_pool_value": 1_000_000.0,
    "fees": 0.003,
    "gas_cost": 0.0,
    "arb_fees": 0.0,
    "arb_frequency": 1,
    "do_arb": True,
    "return_val": "daily_log_sharpe",
    "optimisation_settings": {
        "method": "gradient_descent",
        "base_lr": 0.05,
        "optimiser": "adam",
        "batch_size": 8,
        "n_iterations": 250,
        "n_parameter_sets": 4,
        "training_data_kind": "historic",
        "sample_method": "uniform",
        "initial_random_key": 0,
        "n_cycles": 1,
        "val_fraction": 0.0,
        "early_stopping": False,
    },
}


HYPERSURGE_INITIALS = {
    "initial_hypersurge_arb_max_fee": 0.02,
    "initial_hypersurge_arb_threshold": 0.10,
    "initial_hypersurge_arb_cap_deviation": 0.50,
    "initial_hypersurge_noise_max_fee": 0.10,
    "initial_hypersurge_noise_threshold": 0.10,
    "initial_hypersurge_noise_cap_deviation": 0.50,
}


def oracle_prices_from_price_data(price_data: pd.DataFrame, tokens):
    """Build a DynamicInputFrames-compatible oracle frame from price data."""
    index = pd.Index(price_data.index)
    if np.issubdtype(index.dtype, np.datetime64):
        unix = index.view("int64") // 10**6
    else:
        unix = pd.to_numeric(index, errors="raise").astype("int64")

    data = {"unix": unix}
    for token in tokens:
        column = f"close_{token}"
        if column not in price_data.columns:
            raise ValueError(
                f"price_data must contain {column} to derive oracle prices"
            )
        data[token] = price_data[column].to_numpy()
    return pd.DataFrame(data)


def _dynamic_input_frames(oracle_prices):
    if oracle_prices is None:
        return None
    return DynamicInputFrames(oracle_prices=oracle_prices)


def _common_training_fingerprint(rule, tokens):
    fingerprint = deepcopy(DEFAULT_FINGERPRINT)
    fingerprint["rule"] = rule
    fingerprint["tokens"] = list(tokens)
    fingerprint.update(HYPERSURGE_INITIALS)
    return fingerprint


def train_hypersurge_balancer(
    tokens=("BTC", "ETH"),
    *,
    root=None,
    price_data=None,
    oracle_prices=None,
    use_price_data_as_oracle=False,
    verbose=True,
):
    """Train the HyperSurge Balancer pool."""
    fingerprint = _common_training_fingerprint("balancer_hypersurge", tokens)
    if use_price_data_as_oracle:
        if price_data is None:
            raise ValueError("price_data is required when use_price_data_as_oracle=True")
        oracle_prices = oracle_prices_from_price_data(price_data, tokens)

    return train_on_historic_data(
        run_fingerprint=fingerprint,
        root=root,
        price_data=price_data,
        dynamic_input_frames=_dynamic_input_frames(oracle_prices),
        verbose=verbose,
        return_training_metadata=True,
        force_init=True,
    )


def train_hypersurge_reclamm(
    tokens=("BTC", "ETH"),
    *,
    root=None,
    price_data=None,
    oracle_prices=None,
    use_price_data_as_oracle=False,
    verbose=True,
):
    """Train the HyperSurge reCLAMM pool."""
    fingerprint = _common_training_fingerprint("reclamm_hypersurge", tokens)
    fingerprint.update(
        {
            "initial_price_ratio": 2.0,
            "initial_centeredness_margin": 0.25,
            "initial_daily_price_shift_base": 1.0 - 1.0 / 124000.0,
            "reclamm_interpolation_method": "geometric",
        }
    )
    if use_price_data_as_oracle:
        if price_data is None:
            raise ValueError("price_data is required when use_price_data_as_oracle=True")
        oracle_prices = oracle_prices_from_price_data(price_data, tokens)

    return train_on_historic_data(
        run_fingerprint=fingerprint,
        root=root,
        price_data=price_data,
        dynamic_input_frames=_dynamic_input_frames(oracle_prices),
        verbose=verbose,
        return_training_metadata=True,
        force_init=True,
    )


if __name__ == "__main__":
    examples = [
        ("balancer_hypersurge", train_hypersurge_balancer),
        ("reclamm_hypersurge", train_hypersurge_reclamm),
    ]
    for name, train_fn in examples:
        print(f"\nTraining {name}...")
        params, metadata = train_fn(verbose=True)
        print(
            f"{name}: objective={metadata['final_objective']:.4f}, "
            f"epochs={metadata['epochs_trained']}"
        )
