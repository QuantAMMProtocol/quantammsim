import numpy as np
import os.path
import os

import glob
import json
from quantammsim.runners.default_run_fingerprint import run_fingerprint_defaults
from quantammsim.utils.data_processing.historic_data_utils import (
    get_historic_csv_data,
    get_historic_csv_data_w_versions,
)
from quantammsim.core_simulator.param_utils import default_set


def check_if_run_fingerprints_have_same_values(list_of_run_fingerprints, keys_to_check):
    """
    Check if all run fingerprints in a list have the same values for specified keys.

    This function iterates through a list of run fingerprints (dictionaries) and checks if
    all dictionaries have the same values for the keys specified in `keys_to_check`.
    If a key is missing in any fingerprint or if any value differs, the function returns False.
    If all values match for the specified keys across all fingerprints, it returns True.

    Parameters:
    - list_of_run_fingerprints (list of dict): A list of dictionaries, each representing a run fingerprint.
    - keys_to_check (list of str): A list of keys to check for consistent values across the fingerprints.

    Returns:
    - bool: True if all specified keys have the same values in all fingerprints, False otherwise.
    """
    if not list_of_run_fingerprints:
        return True  # Empty list, trivially all values are the same

    # Initialize a dictionary to store the first occurrence of each key's value
    reference_values = {}

    # Iterate through each fingerprint in the list
    for fingerprint in list_of_run_fingerprints:
        # Check each key in the keys to check
        for key in keys_to_check:
            if key in fingerprint:
                if key not in reference_values:
                    # Store the first occurrence of the key's value
                    reference_values[key] = fingerprint[key]
                elif fingerprint[key] != reference_values[key]:
                    # If any value does not match the reference, return False
                    return False
            else:
                # If the key is not present in the fingerprint, consider it a match
                return True

    # If all checks passed, return True
    return True


def load_run_fingerprints(results_dir):
    """
    Load run fingerprints from JSON files in a specified directory.

    This function searches for all JSON files in the given directory, reads them,
    and extracts the first element (assumed to be the run fingerprint) from each file.
    It returns a list of these run fingerprints.

    Parameters:
    - results_dir (str): The directory path where the JSON files are stored.

    Returns:
    - list: A list of run fingerprints extracted from the JSON files.
    """
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    run_fingerprints = []
    n_files = len(json_files)
    counter = 0
    json_files.sort()
    for json_file in json_files:
        with open(json_file) as f:
            params = json.load(f)
            params = json.loads(params)
        local_run_fingerprint = params[0]
        run_fingerprints.append(local_run_fingerprint)
    return run_fingerprints


def load_price_data_if_fingerprints_match(
    run_fingerprints,
    base_keys_to_check=None,
    optimisation_keys_to_check=None,
    root=None,
    verbose=False,
    token_override=None,
):
    if base_keys_to_check is None:
        base_keys_to_check = ["tokens"]
    if optimisation_keys_to_check is None:
        optimisation_keys_to_check = ["training_data_kind", "max_mc_version"]
    if check_if_run_fingerprints_have_same_values(
        run_fingerprints, base_keys_to_check
    ) and check_if_run_fingerprints_have_same_values(
        [rf["optimisation_settings"] for rf in run_fingerprints],
        optimisation_keys_to_check,
    ):
        run_fingerprint = run_fingerprints[0]

        for key, value in run_fingerprint_defaults.items():
            default_set(run_fingerprint, key, value)
        for key, value in run_fingerprint_defaults["optimisation_settings"].items():
            default_set(run_fingerprint["optimisation_settings"], key, value)

        if token_override is not None:
            all_tokens = token_override
        else:
            # get extended list of assets
            all_tokens = [run_fingerprint["tokens"]] + [
                cprd["tokens"] for cprd in run_fingerprint["subsidary_pools"]
            ]
            all_tokens = [item for sublist in all_tokens for item in sublist]
        unique_tokens = list(set(all_tokens))
        unique_tokens.sort()

        max_memory_days = 365.0
        np.random.seed(0)
        if verbose:
            print("loading data for all run fingerprints")
        if run_fingerprint["optimisation_settings"]["training_data_kind"] == "historic":
            price_data = get_historic_csv_data(unique_tokens, cols=["close"], root=root)
            return price_data
        elif run_fingerprint["optimisation_settings"]["training_data_kind"] == "mc":
            list_of_tickers = unique_tokens
            mc_data_available_for = []  # TODO: populate from data directory scan
            cols = ["close"]
            mc_tokens = [
                value for value in list_of_tickers if value in mc_data_available_for
            ]
            non_mc_tokens = [
                value for value in list_of_tickers if value not in mc_data_available_for
            ]
            price_data_mc = get_historic_csv_data_w_versions(
                mc_tokens,
                cols,
                root,
                max_verion=run_fingerprint["optimisation_settings"]["max_mc_version"],
            )
            price_data_non_mc = get_historic_csv_data(non_mc_tokens, cols, root)
            return [price_data_mc, price_data_non_mc]

        else:
            raise NotImplementedError
    else:
        return None


def load_price_data_if_fingerprints_in_dir_match(
    results_dir,
    base_keys_to_check=None,
    optimisation_keys_to_check=None,
    root=None,
    verbose=False,
):
    if base_keys_to_check is None:
        base_keys_to_check = [
            "tokens",
            "startDateString",
            "endDateString",
            "endTestDateString",
        ]
    if optimisation_keys_to_check is None:
        optimisation_keys_to_check = ["training_data_kind", "max_mc_version"]
    run_fingerprints = load_run_fingerprints(results_dir)
    return load_price_data_if_fingerprints_match(
        run_fingerprints, base_keys_to_check, optimisation_keys_to_check, root, verbose
    )
