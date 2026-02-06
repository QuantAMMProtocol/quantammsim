import numpy as np
import pandas as pd

# again, this only works on startup!
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random


def get_indices(
    start_index,
    bout_length,
    len_prices,
    key,
    optimisation_settings,
):
    """
    Get indices for sampling data windows during training.

    Parameters
    ----------
    start_index : int
        Starting index position in the data
    bout_length : int
        Length of each training window/bout
    len_prices : int
        Total length of the price data
    key : jax.random.PRNGKey
        JAX random number generator key
    optimisation_settings : dict
        Dictionary containing optimization settings with keys:
        - batch_size: Number of windows to sample
        - training_data_kind: Type of training data ('historic' or 'mc')
        - sample_method: Method for sampling windows ('exponential' or 'uniform')
        - max_mc_version: Maximum MC version number (only used if training_data_kind='mc')

    Returns
    -------
    tuple
        - start_indexes : jnp.ndarray
            Array of sampled starting indices, shape (batch_size, 2) for historic data
            or (batch_size, 3) for MC data
        - key : jax.random.PRNGKey
            Updated random number generator key
    """

    # first split indices
    key, subkey = random.split(key)

    # we want to sample starting indices for training
    # simplest thing is simply anything before the bout is up

    batch_size = optimisation_settings["batch_size"]
    training_data_kind = optimisation_settings["training_data_kind"]
    sample_method = optimisation_settings["sample_method"]

    if training_data_kind == "historic":
        return_shape = (batch_size, 2)
        sample_shape = (batch_size,)
    elif training_data_kind == "mc":
        max_mc_version = optimisation_settings["max_mc_version"]
        # MC versions are indexed from zero
        return_shape = (batch_size, 3)
        sample_shape = (batch_size,)

    start_indexes = jnp.zeros(return_shape, dtype=jnp.int64)
    range_ = int(len_prices - bout_length - start_index)
    if sample_method == "exponential":
        probs = 5 * jnp.arange(range_) / range_
        probs = jnp.exp(probs)
        probs = probs / jnp.sum(probs)
        # start_indexes[:, 0] = start_index + np.random.choice(
        #     range_, size=batch_size, replace=False, p=probs
        # )
        start_indexes = start_indexes.at[:, 0].set(
            start_index
            + random.choice(subkey, range_, shape=sample_shape, replace=False, p=probs)
        )

    elif sample_method == "uniform":
        start_indexes = start_indexes.at[:, 0].set(
            start_index
            + random.choice(subkey, range_, shape=sample_shape, replace=False)
        )
    elif sample_method == "stratified":
        # Use float boundaries so the last segment extends to the full range
        seg_boundaries = jnp.linspace(0, range_, batch_size + 1).astype(jnp.int64)
        seg_starts = seg_boundaries[:-1]
        seg_sizes = seg_boundaries[1:] - seg_starts
        offsets = random.randint(subkey, shape=sample_shape, minval=0, maxval=jnp.maximum(seg_sizes, 1))
        start_indexes = start_indexes.at[:, 0].set(
            start_index + seg_starts + offsets
        )
    else:
        raise NotImplementedError
    if training_data_kind == "mc":
        key, subkey = random.split(key)
        start_indexes = start_indexes.at[:, -1].set(
            random.choice(subkey, max_mc_version + 1, shape=sample_shape, replace=True)
        )
    return start_indexes, key


def raw_trades_to_trade_array(raw_trades, start_date_string, end_date_string, tokens):
    """
    Convert raw trade data to a structured trade array.

    This function takes raw trade data and converts it into a pandas DataFrame
    with a continuous range of Unix timestamps. Each row in the DataFrame
    represents a minute, and trades are mapped to their corresponding timestamps.

    Parameters
    ----------
    raw_trades : pandas df
        Raw trades, where each trade is a row containing unix_timestamp, 
        token_in (str), token_out (str), amount_in).
    start_time : str
        The start date time in format "%Y-%m-%d %H:%M:%S".
    end_time : str
        The end date time in format "%Y-%m-%d %H:%M:%S".
    tokens : list of str
        The tokens of the run

    Returns
    -------
    numpy array:
        A numpy array with columns 'token in', 'token out', and 'amount in'.
        The index is a continuous range of Unix timestamps from start_unix
        to end_unix at minute intervals. Timestamps without trades are
        filled with zeros.
    """
    # Create a DataFrame with a continuous range of Unix timestamps
    full_index = (
        pd.date_range(
            start=pd.to_datetime(start_date_string, format="%Y-%m-%d %H:%M:%S"),
            end=pd.to_datetime(end_date_string, format="%Y-%m-%d %H:%M:%S"),
            freq="T",
        ).astype(int)
        // 10**6
    )
    full_index_df = pd.DataFrame(
        index=full_index, columns=["token_in", "token_out", "amount_in"], data=0
    )
    # Create a dictionary to map token strings to their numerical indexes
    token_to_index = {token: index for index, token in enumerate(tokens)}
    for index, row in raw_trades.iterrows():
        unix_timestamp = row["unix"]
        token_in = row["token_in"]
        token_out = row["token_out"]
        amount_in = row["amount_in"]
        if unix_timestamp in full_index_df.index:
            token_in_index = token_to_index.get(token_in, 0)  # Use 0 for unknown tokens
            token_out_index = token_to_index.get(
                token_out, 0
            )  # Use 0 for unknown tokens
            full_index_df.loc[unix_timestamp] = [
                token_in_index,
                token_out_index,
                amount_in,
            ]
    return np.array(full_index_df)[:-1]


def raw_fee_like_amounts_to_fee_like_array(
    raw_inputs, start_date_string, end_date_string, names, fill_method="base"
):
    """
    Convert raw fee-like data to a structured fee-like array.

    Takes raw fee-like data (fees, gas costs, arb fees) and converts it into a pandas
    DataFrame with a continuous range of Unix timestamps. Each row represents a minute,
    with trades mapped to their corresponding timestamps.

    Parameters
    ----------
    raw_inputs : pandas.DataFrame
        Raw fee-like data, where each row contains unix_timestamp and the fee-like
        amount with given column name
    start_time : str
        The start date time in format "%Y-%m-%d %H:%M:%S"
    end_time : str
        The end date time in format "%Y-%m-%d %H:%M:%S"
    names : list of str
        The names of columns in raw_inputs of the fee-like amount
    fill_method : str
        The method to fill in missing values. Options:
        - 'base': fills rows which have no values with 0
        - 'ffill': fills with the last non-zero value

    Returns
    -------
    numpy.ndarray
        Array giving the fee-like values over time. The index is a continuous range
        of Unix timestamps from start_unix to end_unix at minute intervals.
        Timestamps without values are filled with zeros.
    """
    # Create a DataFrame with a continuous range of Unix timestamps
    full_index = (
        pd.date_range(
            start=pd.to_datetime(start_date_string, format="%Y-%m-%d %H:%M:%S"),
            end=pd.to_datetime(end_date_string, format="%Y-%m-%d %H:%M:%S"),
            freq="min",
        ).astype(int)
        // 10**6
    )[:-1]
    full_index_df = pd.DataFrame(
        index=full_index, 
        columns=names, 
        data=0,
        dtype=np.float64
    )

    # Map raw data to the full index DataFrame
    for index, row in raw_inputs.iterrows():
        unix_timestamp = int(row["unix"])
        if unix_timestamp in full_index_df.index:
            for name in names:
                full_index_df.loc[unix_timestamp, name] = float(row[name])

    # Apply fill method
    if fill_method == "ffill":
        try:
            # Validate required columns exist
            if 'unix' not in raw_inputs.columns:
                raise KeyError("raw_inputs must contain 'unix' column")
            for name in names:
                if name not in raw_inputs.columns:
                    raise KeyError(f"raw_inputs missing required column: {name}")

            # Convert start_date_string to unix timestamp
            start_unix = pd.to_datetime(start_date_string, format="%Y-%m-%d %H:%M:%S").value // 10**6

            # Ensure unix values are valid
            valid_unix = pd.to_numeric(raw_inputs['unix'], errors='coerce')
            valid_mask = valid_unix.notna()

            for name in names:
                initial_value = None

                if valid_mask.any():
                    # Try to get the last value before our start date
                    previous_values = raw_inputs[
                        valid_mask & (valid_unix < start_unix)
                    ]

                    if not previous_values.empty:
                        try:
                            initial_value = pd.to_numeric(previous_values[name].iloc[-1])
                        except (ValueError, TypeError):
                            initial_value = None

                    if initial_value is None or pd.isna(initial_value):
                        # Try to get first value in our date range
                        in_range_values = raw_inputs[
                            valid_mask & (valid_unix >= start_unix)
                        ]
                        if not in_range_values.empty:
                            try:
                                initial_value = pd.to_numeric(in_range_values[name].iloc[0])
                            except (ValueError, TypeError):
                                initial_value = None

                if initial_value is not None and pd.notna(initial_value):
                    # this more complex logic is because of how we have started with prior-to-start values
                    # filled in, and then we want to ffill the rest
                    # Fill initial values
                    full_index_df[name] = full_index_df[name].mask(
                        full_index_df[name] == 0, 
                        initial_value
                    )
                    # Use ffill()
                    full_index_df[name] = full_index_df[name].where(
                        full_index_df[name] != 0
                    ).ffill()
        except (ValueError, KeyError, TypeError) as e:
            print(f"Warning: Error during ffill processing: {str(e)}")
            # On any error, return the original zero-filled DataFrame
            pass
    # If fill_method is 'base', we don't need to do anything as zeros are already in place
    elif fill_method == "base":
        pass
    else:
        raise NotImplementedError
    if len(names) == 1:
        return np.array(full_index_df, dtype=np.float64)[:,0]
    else:
        return np.array(full_index_df, dtype=np.float64)

def filter_coarse_weights_by_data_indices(coarse_weights, data_dict):
        """
        Filter coarse weights to match the start and end indices from data_dict.

        Parameters
        ----------
        coarse_weights : dict
            Dictionary containing 'unix_values' and 'weights' arrays
        data_dict : dict
            Dictionary containing 'unix_values', 'start_idx' and 'end_idx'

        Returns
        -------
        dict
            Copy of coarse_weights with filtered weights array
        """
        weights_start_index = np.where(
            coarse_weights["unix_values"]
            == data_dict["unix_values"][data_dict["start_idx"]]
        )[0][0]
        weights_end_index = np.where(
            coarse_weights["unix_values"]
            == data_dict["unix_values"][data_dict["end_idx"] - 1]
        )[0][0]

        filtered_weights = coarse_weights.copy()
        filtered_weights["weights"] = filtered_weights["weights"][
            weights_start_index : (weights_end_index + 1)
        ]
        return filtered_weights

def filter_reserves_by_data_indices(reserves, unix_values, data_dict):
    """
    Filter reserves to match the start and end indices from data_dict.
    """
    reserves_start_index = np.where(
        unix_values
        == data_dict["unix_values"][data_dict["start_idx"]]
    )[0][0]
    reserves_end_index = np.where(
        unix_values
        == data_dict["unix_values"][data_dict["end_idx"] - 1]
    )[0][0]

    filtered_reserves = reserves.copy()
    filtered_reserves = filtered_reserves[
        reserves_start_index : (reserves_end_index + 1)
    ]
    return filtered_reserves

def filter_reserves_by_given_timestamp(reserves, unix_values, timestamp):
    """
    Filter reserves to match the start and end indices from data_dict.
    """
    reserves_index = np.where(
        unix_values
        == timestamp
    )[0][0]
    return reserves[reserves_index].copy()