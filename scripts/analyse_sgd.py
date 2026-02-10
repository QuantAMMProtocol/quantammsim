import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so
from jax import  clear_caches

import gc
import ast
from datetime import datetime

from quantammsim.runners.jax_runners import do_run_on_historic_data
from quantammsim.pools.G3M.balancer.balancer import BalancerPool
from quantammsim.runners.jax_runner_utils import NestedHashabledict
from quantammsim.utils.data_processing.historic_data_utils import get_historic_parquet_data

from quantammsim.core_simulator.param_utils import retrieve_best, calc_lamb, lamb_to_memory_days_clipped, memory_days_to_logit_lamb
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import config

config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
config.update("jax_persistent_cache_min_entry_size_bytes", -1)
config.update("jax_persistent_cache_min_compile_time_secs", 0)

import warnings
warnings.filterwarnings('ignore')


"""Configuration for SGD analysis."""


from quantammsim.utils.post_train_analysis import (
    calculate_period_metrics,
    calculate_continuous_test_metrics,
)
# from quantammsim.utils.plot_utils import name_to_latex_name, plot_weights

# Environment setup
ENV_VARS = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREAD": "1",
}

# Directory configuration
BASE_DIR = Path("./")
PLOT_DIR = BASE_DIR / "product_training_runs/plots"
RESULTS_DIR = BASE_DIR / "analysis_results"

# Plot styling
PLOT_STYLE = {
    "COLOR": "#E6CE97",
    "BACKGROUND": "#162536",
    "SNS_CONFIG": {
        "text.color": "#E6CE97",
        "axes.labelcolor": "#E6CE97",
        "xtick.color": "#E6CE97",
        "ytick.color": "#E6CE97",
        "figure.facecolor": "#162536",
        "axes.facecolor": "#162536",
        "text.usetex": True,
        "axes.grid": False,
    },
}


# Run periods configuration
# alter this to be the additional run periods, besides that in
# the training fingerprint you want to analyse
RUN_PERIODS = [
    {
        "name": "March2021toDec2024toDec2025",
        "start_date": "2023-01-01 00:00:00",
        "end_date": "2025-06-01 00:00:00",
        "end_test_date": "2025-11-30 00:00:00",
    },

]

# Create required directories
for directory in [PLOT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# # Set environment variables
# for key, value in ENV_VARS.items():
#     os.environ[key] = value


def parse_date_range_column_name(column_name):
    """Parse column names containing date ranges and convert to readable formats.
    
    Parameters
    ----------
    column_name : str
        Column name containing date patterns like 'March2021toAug2024toJune2025test'
        
    Returns
    -------
    str
        Readable date range format like 'Aug 2024 -> June 2025' for test periods
        or '2021 -> Aug 2024' for train periods
    """
    import re
    
    # Pattern to match date ranges: [Month][Year]to[Month][Year]to[Month][Year]
    pattern = r'([A-Za-z]+)(\d{4})to([A-Za-z]+)(\d{4})to([A-Za-z]+)(\d{4})'
    match = re.search(pattern, column_name)
    
    if not match:
        return column_name  # Return original if no pattern match
    
    start_month, start_year, middle_month, middle_year, end_month, end_year = match.groups()
    
    # Convert month abbreviations to readable format
    month_map = {
        'Jan': 'Jan', 'Feb': 'Feb', 'Mar': 'Mar', 'Apr': 'Apr',
        'May': 'May', 'Jun': 'Jun', 'Jul': 'Jul', 'Aug': 'Aug',
        'Sep': 'Sep', 'Oct': 'Oct', 'Nov': 'Nov', 'Dec': 'Dec',
        'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
        'May': 'May', 'June': 'Jun', 'July': 'Jul', 'August': 'Aug',
        'September': 'Sep', 'October': 'Oct', 'November': 'Nov', 'December': 'Dec'
    }
    
    # Clean up month names
    start_month = month_map.get(start_month.title(), start_month)
    middle_month = month_map.get(middle_month.title(), middle_month)
    end_month = month_map.get(end_month.title(), end_month)
    
    # Determine context by checking if 'test' or 'train' is in the column name
    if 'test' in column_name.lower():
        # For test periods, show range from middle to end
        if middle_year == end_year:
            return f"{middle_month} {middle_year} -> {end_month} {end_year}"
        else:
            return f"{middle_month} {middle_year} -> {end_month} {end_year}"
    elif 'train' in column_name.lower():
        # For train periods, show range from start to middle
        if start_year == middle_year:
            return f"{start_month} {start_year} -> {middle_month} {middle_year}"
        else:
            return f"{start_year} -> {middle_month} {middle_year}"
    else:
        # For unknown context, return the full range
        return f"{start_month} {start_year} -> {end_month} {end_year}"

def rename_date_columns(df):
    """Rename columns containing date patterns to readable formats.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to rename
        
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
    """
    new_columns = {}
    
    for col in df.columns:
        if 'to' in col and any(month in col for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
                                                          'January', 'February', 'March', 'April',
                                                          'May', 'June', 'July', 'August',
                                                          'September', 'October', 'November', 'December']):
            new_columns[col] = col.split("(")[0] + "(" +parse_date_range_column_name(col) + ")"
    
    return df.rename(columns=new_columns)

def name_to_latex_name_OG(name):
    """Convert run name to LaTeX formatted name.

    Parameters
    ----------
    name : str
        Name of the run (e.g. 'index_market_cap', 'momentum')

    Returns
    -------
    str
        LaTeX formatted name
    """
    # Special case for index_market_cap since we want to shorten it
    if name == "index_market_cap":
        return "\\mathrm{Index}"

    # Split name into words
    words = name.split("_")

    # Capitalize first letter of each word
    words = [word.capitalize() for word in words]

    # Join with escaped spaces and wrap in \mathrm{}
    latex_name = "\\ ".join(words)
    return f"$\\mathrm{{{latex_name}}}$"


def name_to_latex_name(name):
    """Convert run name to clean LaTeX formatted name.

    Parameters
    ----------
    name : str
        Name of the run (e.g. 'Current_index_BTC-ETH_min_0.1_index_memory_day_30.0')

    Returns
    -------
    str
        LaTeX formatted name
    """
    # Handle different strategy types
    if name.startswith("Current_index"):
        return "$\\mathrm{Current\\ Index\\ Product}$"
    elif name.startswith("HODL"):
        return "$\\mathrm{HODL}$"
    elif name.startswith("QuantAMM_index"):
        return "$\\mathrm{QuantAMM\\ Index}$"
    elif name.startswith("Optimized_QuantAMM"):
        # Extract rule name from pattern "..._rule_RULENAME"
        rule = name.split("rule_")[-1]
        # Clean up rule name
        rule = rule.replace("_", " ").title()
        # Special case for specific rules
        if rule == "Mean Reversion Channel":
            rule = "Mean-Reversion\\ Channel"
        elif rule == "Anti Momentum":
            rule = "Anti-Momentum"
        elif rule == "Power Channel":
            rule = "Power-Channel"
        elif rule == "Difference Momentum":
            rule = "Difference\\ Momentum"
        elif rule == "Triple Threat Mean Reversion Channel":
            rule = "Triple\\ Threat\\ Mean\\ Reversion\\ Channel"
        else:
            raise Exception("Stop here")
        return f"$\\mathrm{{QuantAMM\\ {rule}}}$"

    return name_to_latex_name_OG(name)

def plot_weights(output_dict, run_fingerprint, plot_prefix="weights", verbose=True):
    plot_path = Path("./plots/")
    plot_path.mkdir(parents=True, exist_ok=True)
    plot_prefix = "./plots/" + plot_prefix

    # Calculate weights from reserves and prices
    total_value = np.sum(output_dict["reserves"] * output_dict["prices"], axis=1, keepdims=True)
    weights = np.array(output_dict["reserves"] * output_dict["prices"] / total_value)

    weights = weights[::1440]
    # Create DataFrame for plotting
    df_list = []
    tokens = sorted(run_fingerprint["tokens"])
    for i, token in enumerate(tokens):
        df_list.extend([
            {
                "Time": t,
                "Weight": w,
                "Token": token
            }
            for t, w in enumerate(weights[:, i])
        ])

    df = pd.DataFrame(df_list)
    start_date = datetime.strptime(run_fingerprint["startDateString"], "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(run_fingerprint["endDateString"], "%Y-%m-%d %H:%M:%S")

    # Create date range for x-axis
    date_range = pd.date_range(start=start_date, end=end_date, periods=len(df["Time"].unique()))
    df["Time"] = np.tile(date_range, weights.shape[1])

    # fig, ax = plt.subplots(figsize=(10, 6))
    f = mpl.figure.Figure()

    # Create stacked area plot
    pl = (
        so.Plot(df, "Time", "Weight", color="Token")
        .add(so.Area(alpha=0.7), so.Stack())
        .limit(y=(0, 1))
        .scale(color=sns.color_palette())
        .label(y="$\\mathrm{Weight}$", x="$\\mathrm{Date}$")
    )

    # Render the plot on our axis
    res = pl.on(f).plot()
    ax = f.axes[0]
    # Select sparse dates for x-axis (4 evenly spaced dates)
    unique_dates = df["Time"].unique()
    date_indices = np.linspace(0, len(unique_dates)-1, 4, dtype=int)
    selected_dates = unique_dates[date_indices]

    # Format dates as LaTeX strings
    date_labels = [f"$$\\mathrm{{{pd.Timestamp(date).strftime('%Y-%m-%d')}}}$$" for date in selected_dates]
    # Set the ticks and labels
    ax.set_xticks(date_indices,date_labels, rotation=45)
    # plt.xticks(date_indices, date_labels, rotation=45)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Save plot
    pl.save(
        plot_prefix + "_weights_over_time.png",
        dpi=700,
        bbox_inches="tight"
    )
    plt.close()

def load_sgd_results(base_dir, load_method="best_objective", min_test=None):
    """Load and parse SGD results files.

    Parameters
    ----------
    base_dir : str or Path
        Directory containing SGD result JSON files
    load_method : str, optional
        Method for selecting parameter sets. One of:
        'last', 'best_objective', 'best_train_objective', 'best_test_objective',
        'best_train_min_test_objective'
    min_test : float, optional
        Minimum test objective threshold for methods that use it

    Returns
    -------
    list
        List of trial info dictionaries matching Optuna format
    """
    trials_info = []
    base_path = Path(base_dir)

    for file_path in base_path.glob("run_*"):
        # Use retrieve_best to get cleaned parameters
        try:
            params, steps = retrieve_best(
                str(file_path), load_method, re_calc_hess=False, min_alt_obj=min_test, return_as_iterables=True
            )
            # Load run fingerprint from first entry
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                data = json.loads(data)
            run_fingerprint = data[0]
            # Format trial info to match Optuna structure
            for param, step in zip(params, steps):
                trial_info = {
                    "study_id": file_path.stem,
                    "trial_number": step,
                    # Parameters are already indexed by retrieve_best
                    "train_value": float(param["train_objective"]),
                    "test_value": float(param["test_objective"]),
                    "objective": float(param["objective"]),
                    # Configuration info from run fingerprint
                    "rule": run_fingerprint.get("rule"),
                    "tokens": tuple(sorted(run_fingerprint.get("tokens", []))),
                    "start_date": run_fingerprint.get("startDateString"),
                    "end_date": run_fingerprint.get("endDateString"),
                    "end_test_date": run_fingerprint.get("endTestDateString"),
                    "return_val": run_fingerprint.get("return_val", "sharpe"),
                    "chunk_period": run_fingerprint.get("chunk_period"),
                    "weight_interpolation_period": run_fingerprint.get("weight_interpolation_period"),
                    "minimum_weight": run_fingerprint.get("minimum_weight"),
                    "bout_offset": run_fingerprint.get("bout_offset"),
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "initial_pool_value": float(
                        run_fingerprint.get("initial_pool_value", 0.0)
                    ),
                    "bout_offset": int(run_fingerprint.get("bout_offset", 0)),
                }

                # Add parameters with param_ prefix
                # retrieve_best has already cleaned metadata and indexed parameters
                param_fields = {
                    k: jnp.array(v)
                    for k, v in param.items()
                    if k
                    not in [
                        "train_objective",
                        "test_objective",
                        "objective",
                        "subsidary_params",
                    ]
                }
                # Set logit_delta_lamb to zeros if present
                if "logit_delta_lamb" in param_fields:
                    param_fields["logit_delta_lamb"] = jnp.zeros_like(param_fields["logit_delta_lamb"])
                if "initial_weights_logits" in param_fields:
                    param_fields["initial_weights_logits"] = jnp.zeros_like(param_fields["initial_weights_logits"])
                trial_info.update({"params": param_fields})
                trial_info.update({"run_fingerprint": run_fingerprint})
                # Validate required fields
                required_fields = ["rule", "tokens", "train_value", "test_value"]
                if all(trial_info.get(f) is not None for f in required_fields):
                    trials_info.append(trial_info)
                else:
                    print(f"Skipping {file_path}: missing required fields")
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue

    return trials_info


def convert_daily_to_hourly_params(params, run_fingerprint, scale_k_by_frequency=False):
    """Convert parameters from daily (1440min) to hourly (60min) updates.
    
    Parameters
    ----------
    params : dict
        Original parameter dictionary
    run_fingerprint : dict
        Run fingerprint containing chunk_period info
    scale_k_by_frequency : bool
        If True, scale k down by 24 to maintain similar daily aggressiveness.
        If False, keep k unchanged (resulting in 24x more aggressive daily behavior).
    
    Returns
    -------
    tuple
        (converted_params, converted_fingerprint)
    """
    
    # Only convert if original chunk_period is 1440
    if run_fingerprint.get("chunk_period", 60) != 1440:
        return params, run_fingerprint
    
    converted_params = tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, params)
    converted_fingerprint = run_fingerprint.copy()
    
    original_chunk_period = 1440
    new_chunk_period = 60
    
    print(f"Converting daily to hourly parameters:")
    print(f"  Original chunk_period: {original_chunk_period}")
    print(f"  New chunk_period: {new_chunk_period}")
    
    # Adjust logit_lamb to preserve memory days
    if "logit_lamb" in params:
        current_lamb = calc_lamb(params)
        current_memory_days = lamb_to_memory_days_clipped(
            current_lamb, 
            original_chunk_period, 
            max_memory_days=365
        )
        
        # Calculate new logit_lamb to achieve same memory days with new chunk_period
        new_logit_lamb = memory_days_to_logit_lamb(current_memory_days, new_chunk_period)
        converted_params["logit_lamb"] = new_logit_lamb
        
        print(f"  Memory days preserved: {current_memory_days}")
        print(f"  Original logit_lamb: {params['logit_lamb']}")  
        print(f"  New logit_lamb: {new_logit_lamb}")
    else:
        print("  No logit_lamb found - memory days adjustment skipped")
    
    # Handle k parameter scaling
    if scale_k_by_frequency:
        frequency_ratio = original_chunk_period / new_chunk_period  # 24
        
        if "log_k" in params:
            converted_params["log_k"] = params["log_k"] - jnp.log2(frequency_ratio)
            print(f"  Scaling log_k: {params['log_k']} -> {converted_params['log_k']} (reduction by log2({frequency_ratio}))")
        elif "k" in params:
            converted_params["k"] = params["k"] / frequency_ratio
            print(f"  Scaling k: {params['k']} -> {converted_params['k']} (divided by {frequency_ratio})")
        else:
            print("  No k or log_k found - k scaling skipped")
        
        print("  K scaling: ON (maintaining daily aggressiveness)")
    else:
        print("  K scaling: OFF (24x more aggressive daily behavior)")
    
    # Update chunk_period in fingerprint
    converted_fingerprint["chunk_period"] = new_chunk_period
    
    # Scale other timing-related parameters
    frequency_ratio = original_chunk_period / new_chunk_period  # 24
    
    # Scale weight_interpolation_period if present
    if "weight_interpolation_period" in converted_fingerprint:
        original_wip = converted_fingerprint["weight_interpolation_period"]
        converted_fingerprint["weight_interpolation_period"] = int(original_wip / frequency_ratio)
        print(f"  Scaling weight_interpolation_period: {original_wip} -> {converted_fingerprint['weight_interpolation_period']}")
    
    # Scale bout_length if present (this controls the simulation length)
    if "bout_length" in converted_fingerprint:
        original_bout_length = converted_fingerprint["bout_length"] 
        converted_fingerprint["bout_length"] = int(original_bout_length / frequency_ratio)
        print(f"  Scaling bout_length: {original_bout_length} -> {converted_fingerprint['bout_length']}")
    
    print(f"  Conversion complete!")
    
    return converted_params, converted_fingerprint

def analyze_specific_trials(
    base_dir,
    trials_to_analyze,
    return_val="sharpe",
    load_method="best_objective",
    use_pareto_frontier=True,
    do_plots=False,
    keep_top=1.0,
    tokens='',
    convert_daily_to_hourly=False,
    scale_k_by_frequency=False,
    force_reload=True,
):
    """Analyze specific trials from run files.

    Parameters
    ----------
    base_dir : str
        Directory containing the run files
    trials_to_analyze : list[dict]
        List of dicts containing study_id and trial_number to analyze
        Example: [{"study_id": "run_XXXXX", "trial_number": 42}, ...]
    return_val : str, optional
        Metric to optimize for, by default "sharpe"
    load_method : str, optional
        Method for selecting parameter sets, by default "best_objective"
    use_pareto_frontier : bool, optional
        Whether to use pareto frontier analysis, by default True
    do_plots : bool, optional
        Whether to generate plots, by default False
    keep_top : float, optional
        Fraction of top trials to keep, by default 1.0
    tokens : str, optional
        Token filter string, by default ''
    convert_daily_to_hourly : bool, optional
        Whether to convert daily (1440min) runs to hourly (60min), by default False
    scale_k_by_frequency : bool, optional
        If convert_daily_to_hourly=True, whether to scale k by frequency ratio, by default True
    """
    base_path = Path(base_dir)
    all_trials = []

    filename = "sgd_analysis_result_" + load_method + "_keeptop_" + str(keep_top) + "_" + "_".join(tokens) + ".csv"

    if (Path(base_dir) / filename).exists() or force_reload == False:
        df = pd.read_csv(Path(base_dir) / filename)
        # Convert string columns to dicts
        if "params" in df.columns:
            df["params"] = df["params"].str.replace(", dtype=float64", "")
            df["params"] = df["params"].str.replace(",      dtype=float64", "")
            df["params"] = df["params"].str.replace(",\s*dtype=float64", "")
            df["params"] = df["params"].str.replace("Array(", "")
            df["params"] = df["params"].str.replace(")", "")
            # Drop rows where params contains 'nan'
            df = df[~df["params"].astype(str).str.contains('nan')]
            df["params"] = df["params"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df["params"] = df["params"].apply(lambda x: {k: jnp.array(v) for k, v in x.items()})
        if "tokens" in df.columns:
            df["tokens"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        if "run_fingerprint" in df.columns:
            df["run_fingerprint"] = df["run_fingerprint"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    else:
        # Load all study results
        trials_data = load_sgd_results(base_path, load_method=load_method, min_test=0.0)
        if tokens != '':
            trials_data = [trial for trial in trials_data if set(trial["tokens"]) == set(tokens)]
        # Group by tokens and keep only top fraction within each group
        if keep_top != 1.0:
            # Convert trials_data to DataFrame for easier filtering
            df_temp = pd.DataFrame(trials_data)

            # Determine which column to sort by based on load_method
            if load_method == "best_objective":
                sort_col = "objective"
            elif load_method == "best_train_objective":
                sort_col = "train_value"
            elif load_method == "best_test_objective":
                sort_col = "test_value"
            else:
                sort_col = "objective"  # Default to objective

            # Group by tokens and keep top fraction within each group
            filtered_trials = []
            groupby_cols = ["tokens", "return_val"] if load_method == "best_objective" else ["tokens"]
            for _, group in df_temp.groupby(groupby_cols):
                # Sort descending since we want highest values
                sorted_group = group.sort_values(sort_col, ascending=False)
                # Calculate number of rows to keep
                # Determine number of rows to keep based on keep_top value
                if keep_top > 1:
                    # If keep_top is > 1, use it as absolute number of rows
                    n_keep = int(keep_top)
                else:
                    # If keep_top <= 1, interpret as fraction
                    n_keep = max(1, int(len(sorted_group) * keep_top))
                # Keep top rows
                filtered_trials.extend(sorted_group.head(n_keep).to_dict('records'))

            trials_data = filtered_trials
        if trials_data is not None:
            all_trials.extend(trials_data)
        # Convert to DataFrame
        df = pd.DataFrame(all_trials)
        # Sort DataFrame by tokens column
        df = df.sort_values(
            by=["bout_offset", "chunk_period", "tokens", "return_val", "start_date", "end_date", "minimum_weight", "rule"],
            ascending=[False, True, True, True, True, False, True, True],
            na_position="last",
        )
        # Reorder columns to put specified columns at the front
        column_order = [
            "chunk_period",
            "bout_offset",
            "tokens",
            "rule",
            "return_val", 
            "minimum_weight",
            "start_date",
            "end_date",
            "study_id",
            "trial_number",
            "objective",
            "test_value", 
            "train_value"
        ]
        # Add remaining columns after the specified ones
        column_order.extend([col for col in df.columns if col not in column_order])
        df = df[column_order]
        # Save DataFrame to disk

        df.to_csv(Path(base_dir) / filename, index=False)

    if len(trials_to_analyze) > 0:
        # Filter DataFrame to only include specified trials
        mask = pd.DataFrame(False, index=df.index, columns=["match"])
        for trial_info in trials_to_analyze:
            mask["match"] |= (df["study_id"] == trial_info["study_id"]) & (
                df["trial_number"] == trial_info["trial_number"]
            )
        filtered_df = df[mask["match"]]
        if filtered_df.empty:
            raise ValueError("No matching trials found")
    else:
        filtered_df = df
    print(filtered_df)
    # Create visualizations and analyze results
    simplified_df = analyze_best_trials(
        filtered_df,
        base_dir=base_dir,
        output_string=load_method + "__keeptop_" + str(keep_top) + "_" + str(len(trials_to_analyze)) + "_" + "_".join(tokens),
        do_plots=do_plots,
        convert_daily_to_hourly=convert_daily_to_hourly,
        scale_k_by_frequency=scale_k_by_frequency,
    )

    df = df.merge(
        simplified_df,
        left_on='study_id',
        right_on='Run ID',
        how='left'
    )

    # Then apply the same sorting
    # Convert return_val to categorical with custom order before sorting
    df['return_val'] = pd.Categorical(
        df['return_val'],
        categories=['sharpe', 'calmar', 'ulcer', 'sterling', 'daily_log_sharpe'],
        ordered=True
    )


    base_name = RUN_PERIODS[0]["name"]
    if base_name.endswith("test"):
        base_name = base_name[:-4]
    df = df.sort_values(
        by=[
            "tokens",
            "Returns over HODL train (" + base_name + ")",
            "start_date",
            "end_date",
            "rule",
        ],
        ascending=[True, False, True, False, True],
        na_position="last",
    )
    columns_to_keep = list(simplified_df.columns) + ['study_id']
    columns_to_keep = [
        "tokens",
        "rule",
        "return_val",
        "minimum_weight",
        "start_date",
        "end_date",
        "study_id",
        "trial_number",
        "objective",
        "test_value",
        "train_value",
        "Comments",
        "Params",
        "Returns over HODL train (" + base_name + ")",
        "Returns train (" + base_name + ")",
        "Sharpe train (" + base_name + ")",
        "Annualized Ulcer Index [M] train (" + base_name + ")",
        "Annualized Calmer Ratio [M] train (" + base_name + ")",
        "Returns over HODL test (" + base_name + ")",
        "Returns test (" + base_name + ")",
        "Sharpe test (" + base_name + ")",
        "Annualized Ulcer Index [M] test (" + base_name + ")",
        "Annualized Calmer Ratio [M] test (" + base_name + ")",
        "bout_offset",
        "chunk_period",
        "noise_trader_ratio",
        "analysis_pool_value",
        "analysis_fees",
        "analysis_gas_cost",
        "optimisation_method",
        "sample_method",
        "use_gradient_clipping",
        "clip_norm",
        "optimiser",
        "learning_rate",
        "batch_size",
        "use_plateau_decay",
        "lr_schedule_type",
        "warmup_steps",
        "ste_max_change",
        "ste_min_max_weight",
    ]

    # columns_to_keep.remove('Run ID')  # Remove the duplicate ID column
    df_filtered = df[columns_to_keep]
    # Rename columns containing specific fragments
    df_filtered = rename_date_columns(df_filtered)
    # Reorder columns according to specified order
    column_order = [
        "study_id",
        "trial_number",
        "tokens",
        "rule",
        "Returns over HODL train (" + parse_date_range_column_name(base_name + "train") + ")",
        "Returns over HODL test (" + parse_date_range_column_name(base_name + "test") + ")",
        "Sharpe train (" + parse_date_range_column_name(base_name + "train") + ")",
        "Sharpe test (" + parse_date_range_column_name(base_name + "test") + ")",
        "return_val",
        "objective",
        "train_value",
        "Returns train (" + parse_date_range_column_name(base_name + "train") + ")",
        "Annualized Ulcer Index [M] train (" + parse_date_range_column_name(base_name + "train") + ")",
        "Annualized Calmer Ratio [M] train (" + parse_date_range_column_name(base_name + "train") + ")",
        "test_value",
        "Returns test (" + parse_date_range_column_name(base_name + "test") + ")",
        "Annualized Ulcer Index [M] test (" + parse_date_range_column_name(base_name + "test") + ")",
        "Annualized Calmer Ratio [M] test (" + parse_date_range_column_name(base_name + "test") + ")",
        "bout_offset",
        "chunk_period",
        "Params",
        "minimum_weight",
        "start_date",
        "end_date",
        "Comments",
        "noise_trader_ratio",
        "analysis_pool_value",
        "analysis_fees",
        "analysis_gas_cost",  
        "optimisation_method",
        "sample_method",
        "use_gradient_clipping",
        "clip_norm",
        "optimiser",
        "learning_rate",
        "batch_size",
        "use_plateau_decay",
        "lr_schedule_type",
        "warmup_steps",
        "ste_max_change",
        "ste_min_max_weight",
    ]
    df_filtered = df_filtered[column_order]
    # Add empty rows after each change in minimum_weight
    empty_row = pd.Series([None] * len(df_filtered.columns), index=df_filtered.columns)

    # Create new dataframe with empty rows inserted
    df_with_breaks = pd.DataFrame()

    # Iterate through rows and add empty row after token changes
    for i in range(len(df_filtered)):
        df_with_breaks = pd.concat([df_with_breaks, pd.DataFrame([df_filtered.iloc[i]])])
        if i < len(df_filtered)-1 and df_filtered.iloc[i]['tokens'] != df_filtered.iloc[i+1]['tokens']:
            df_with_breaks = pd.concat([df_with_breaks, pd.DataFrame([empty_row])])

    filename = f"filled_analysis_{load_method}_keeptop_{keep_top}_{len(trials_to_analyze)}_{'_'.join(tokens)}.csv"
    df_with_breaks.to_csv(Path(base_dir) / filename, index=False)


def plot_values(results, tokens, suffix="", plot_start_end=None, plot_white_line=False, white_line_date=None, plot_dir=None, initial_hodl_weights="same"):
    """Plot value over time for all runs on the same graph."""
    suffix = suffix + "_balancer"
    if plot_dir is None:
        plot_dir = "./plots/"
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Create DataFrame for plotting
    df_list = []
    start_date = datetime.strptime(
        next(iter(results.values()))["fingerprint"]["startDateString"],
        "%Y-%m-%d %H:%M:%S",
    )
    if len(results) == 1:
        # For single run case, add a HODL baseline
        if initial_hodl_weights == "same":
            hodl_reserves = next(iter(results.values()))["reserves"][0]
        elif initial_hodl_weights == "uniform":
            initial_hodl_value = next(iter(results.values()))["value"][0].sum()
            initial_hodl_prices = next(iter(results.values()))["prices"][0]
            n_assets = len(initial_hodl_prices)
            hodl_reserves = (1.0/n_assets) * initial_hodl_value / initial_hodl_prices
        prices = next(iter(results.values()))["prices"]
        # hodl_values = np.sum(hodl_reserves * prices, axis=1)
        # results["HODL"] = {
        #     "value": hodl_values,
        #     "fingerprint": next(iter(results.values()))["fingerprint"].copy()
        # }
        # results["HODL"]["fingerprint"]["rule"] = "HODL"
        hodl_fingerprint = next(iter(results.values()))["fingerprint"].copy()
        hodl_fingerprint["rule"] = "HODL"
        hodl_fingerprint["bout_length"] = len(prices) + 1
        hodl_fingerprint["n_assets"] = len(tokens)
        hodl_fingerprint["initial_pool_value"] = float(next(
            iter(results.values())
        )["value"][0].sum())
        # print("initial_balancer_pool_value", hodl_fingerprint["initial_pool_value"])
        # hodl_params = {"initial_weights": jnp.ones(len(tokens)) / len(tokens)}
        # balancer_pool = BalancerPool()
        # hodl_reserves = balancer_pool.calculate_reserves_zero_fees(
        #     params=hodl_params,
        #     run_fingerprint=NestedHashabledict(hodl_fingerprint),
        #     prices=prices,
        #     start_index=jnp.array([0,0]),
        # )
        hodl_values = (hodl_reserves * prices).sum(axis=1)
        results["HODL"] = {
            "value": hodl_values,
            "fingerprint": hodl_fingerprint.copy()
        }
    for run_name, result in results.items():
        values = result["value"]
        dates = pd.date_range(
            start=start_date,
            end=datetime.strptime(
                result["fingerprint"]["endDateString"], "%Y-%m-%d %H:%M:%S"
            ),
            freq="1min",
        )[:-1]

        df_list.extend(
            [
                {
                    "Date": date,
                    "Value": float(value),  # Ensure values are float
                    "Strategy": str(run_name),  # Ensure strategy names are strings
                }
                for date, value in zip(
                    dates[::1440], values[::1440]
                )  # Take every 1440th value (daily)
            ]
        )

    # Create DataFrame with explicit types
    df = pd.DataFrame(df_list)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Value"] = df["Value"].astype(float) / 1e6
    df["Strategy"] = df["Strategy"].astype("category")
    df["Strategy"] = df["Strategy"].apply(name_to_latex_name)
    # Create plot

    if plot_start_end is not None:
        # Filter df to plot_start_end range if provided
        if isinstance(plot_start_end, tuple) and len(plot_start_end) == 2:
            start_str, end_str = plot_start_end
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # sns.set_style("darkgrid")
    # Get default color palette
    default_palette = sns.color_palette()

    # Modify palette to use COLOR for 4th item if needed
    strategies = df["Strategy"].unique()
    n_strategies = len(strategies)
    palette = list(default_palette[:3])
    if n_strategies > 3:
        palette = [COLOR] + palette
        if n_strategies > 4:
            palette.extend(default_palette[4:n_strategies])
    print("n_strategies", n_strategies)
    # Sort strategies to put QuantAMM rules (except QuantAMM Index) first
    # df['Strategy_order'] = df['Strategy'].apply(lambda x:
    #     0 if (x.startswith('QuantAMM') and x != 'QuantAMM Index')
    #     else 1)
    # df = df.sort_values('Strategy_order')
    # Define explicit order for strategies
    strategy_order = [
        "$\\mathrm{QuantAMM\\ Mean-Reversion\\ Channel}$",
        "$\\mathrm{QuantAMM\\ Triple\\ Threat\\ Mean\\ Reversion\\ Channel}$",
        "$\\mathrm{QuantAMM\\ Momentum}$",
        "$\\mathrm{QuantAMM\\ Anti-Momentum}$",
        "$\\mathrm{QuantAMM\\ Power-Channel}$",
        "$\\mathrm{QuantAMM\\ Difference\\ Momentum}$",
        "$\\mathrm{QuantAMM\\ Index}$",
        "$\\mathrm{HODL}$",
        "$\\mathrm{Balancer}$",
        "$\\mathrm{Traditional\\ DEX}$",
        "$\\mathrm{Current\\ Index\\ Product}$",
    ]
    # Filter strategy_order to only include strategies that exist in the data
    strategy_order = [s for s in strategy_order if s in df["Strategy"].unique()]
    sns.lineplot(data=df, x="Date", y="Value", hue="Strategy", linewidth=2, palette=palette, hue_order=strategy_order)

    # plt.title("$\\mathrm{Value\\ Over\\ Time}$", pad=20)
    plt.xlabel("$\\mathrm{Date}$")
    plt.ylabel("$\\mathrm{Value\\ (\\$M\\ USD)}$")

    # Format x-axis
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT_STYLE["COLOR"])
    ax.spines["bottom"].set_color(PLOT_STYLE["COLOR"])
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    if plot_white_line:

        # Add vertical line and shading for train/test split
        plt.axvline(x=pd.Timestamp(white_line_date), color='white', linestyle='--', alpha=0.5)

        # Add "Train" and "Test" labels in LaTeX near the red line
        plt.text(pd.Timestamp(white_line_date) - pd.Timedelta(days=15), plt.ylim()[1]*0.95,
                "$\\mathrm{Train}$", 
                horizontalalignment='right',
                verticalalignment='top',
                fontsize=10,
                color='white',
                alpha=0.6)
        plt.text(pd.Timestamp(white_line_date) + pd.Timedelta(days=15), plt.ylim()[1]*0.95,
                "$\\mathrm{Test}$",
                horizontalalignment='left', 
                verticalalignment='top',
                fontsize=10,
                color='white',
                alpha=0.6)

    # Remove legend title
    # Reorder legend to put QuantAMM rules (except QuantAMM Index) last
    handles, labels = ax.get_legend_handles_labels()
    # Find indices of QuantAMM rules that aren't QuantAMM Index
    quantamm_indices = [i for i, label in enumerate(labels) 
                       if label.startswith("QuantAMM") and label != "QuantAMM Index"]
    if quantamm_indices:
        # Move each QuantAMM rule to the end, preserving their relative order
        for idx in sorted(quantamm_indices, reverse=True):
            handles.append(handles.pop(idx))
            labels.append(labels.pop(idx))
        # Replace legend
        ax.legend(handles, labels)

    ax.get_legend().set_title(None)
    # Save plot
    plt.tight_layout()
    plt.savefig(
        plot_path / (f"pool_values_comparison_{len(results)}_{'-'.join(tokens)}_{suffix}.png"),
        dpi=700,
        bbox_inches="tight",
    )
    return_over_hodl = df[df["Strategy"]!= name_to_latex_name("HODL")]["Value"].iloc[-1] * 1e6 / hodl_values[-1] - 1.0
    plt.close('all')
    del df
    del hodl_values
    del hodl_reserves
    del df_list
    gc.collect()
    gc.collect()
    gc.collect()
    if initial_hodl_weights == "uniform":
        return return_over_hodl

def analyze_best_trials(
    df,
    base_dir="./sgd_studies", 
    output_string = None,
    do_plots=False,
    convert_daily_to_hourly=False,
    scale_k_by_frequency=False,
):
    """Analyze best trials with different parameters and generate detailed performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all trial results
    base_dir : str, default="./sgd_studies"
        Directory containing Optuna study results
    output_string : str, default=None
        String to append to output file name
    do_plots : bool, default=False
        If True, generate plots for each trial
    """
    output_path = Path(base_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parameters to test
    initial_pool_values = [20000.0, 50000.0,100000.0, 1000000.0, 10000000.0]
    fee_values = [0.0, 0.001, 0.003, 0.01]  # 0 to 1%
    gas_costs = [0.0,0.005,0.5]  # USD per trade
    noise_trader_ratios = [0.0, 0.5, 1.0]  # 0 to 1
    # initial_pool_values = [1000000.0]
    fee_values = [0.0, 0.003]
    gas_costs = [0.0, 1.0, 2.0]
    noise_trader_ratios = [0.0, 0.5]

    initial_pool_values = [100000.0]
    noise_trader_ratios = [0.0]  # 0 to 1
    # initial_pool_values = [1000000.0]
    fee_values = [0.0]
    gas_costs = [0.0]
    noise_trader_ratios = [0.0]

    # define run periods
    run_periods = RUN_PERIODS


    # Print number of rows for each rule
    rule_counts = df['rule'].value_counts()
    print("\nNumber of rows per rule:")
    for rule, count in rule_counts.items():
        print(f"{rule}: {count}")

    grouped = df.groupby(["tokens", "initial_pool_value", "end_date", "bout_offset", "chunk_period", "rule"])

    results = []

    base_path = Path(base_dir)

    # Create results directory within base_dir
    results_dir = base_path / "analysis_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = base_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Extract study name from base_dir
    study_name = base_path.name

    # Construct results filename
    results_filename = (
        f"analysis_results"
        f"_{study_name}"
        f"_{output_string}"
        f".json"
    )

    # results_filename = "analysis_unified_results_all_runs_uptodate_pareto_20250405_033520_best.json"

    results_path = results_dir / results_filename
    # Check if analysis file already exists
    if do_plots or os.path.exists(results_path)==False:
        price_data = get_historic_parquet_data(df.iloc[0]["run_fingerprint"]["tokens"], cols=["close"])
        for _, trial in df.iterrows():
            params = trial["params"]
            run_fingerprint = trial["run_fingerprint"]
            run_fingerprint["startTestDateString"] = run_fingerprint["endDateString"]
            # params["raw_exponents"] = jnp.array(
            #     [-0.48537339,  1.44890609, -0.03770628,  0.28857155]
            # )
            # params["raw_exponents"] = jnp.array(
            #     [-0.5935307, 1.49911745, -0.03770628, 0.28857155]
            # )
            for pool_val, fee, gas, noise_trader_ratio in itertools.product(initial_pool_values, fee_values, gas_costs, noise_trader_ratios):
                run_period_results = []
                local_run_periods = run_periods.copy()
                # add run fingerprint defined run period to local run periods
                local_run_periods = [{"name": "period_from_rf", "start_date": run_fingerprint["startDateString"], "end_date": run_fingerprint["endDateString"], "end_test_date": run_fingerprint["endTestDateString"]}] + local_run_periods
                if 'ARB' in run_fingerprint["tokens"]:
                    for i in range(len(local_run_periods)):
                        local_run_periods[i]["start_date"] = local_run_periods[0]["start_date"]
                for run_period in local_run_periods:
                    # run_period["end_test_date"] = "2025-07-12 00:00:00"
                    run_period = run_period.copy()
                    prices = {"train_prices": None, "test_prices": None, "continuous_test_prices": None}
                    print("tokens", run_fingerprint["tokens"])
                    print("study_id", trial["study_id"])
                    print("trial_number", trial["trial_number"])
                    print("rule", trial["rule"])
                    print("pool_val", pool_val)
                    print("fee", fee)
                    print("gas", gas)
                    print("noise_trader_ratio", noise_trader_ratio)

                    # Update run_fingerprint with new values
                    local_fingerprint = run_fingerprint.copy()
                    local_fingerprint["startDateString"] = run_period["start_date"]
                    if "OM" in local_fingerprint["tokens"]:
                        local_fingerprint["startDateString"] = trial["start_date"]
                        run_period["start_date"] = trial["start_date"]
                        if run_period["name"].startswith("March"):
                            run_period["name"] = "May" + run_period["name"][5:]
                    if "PEPE" in local_fingerprint["tokens"]:
                        local_fingerprint["startDateString"] = trial["start_date"]
                        run_period["start_date"] = trial["start_date"]
                        if run_period["name"].startswith("March"):
                            run_period["name"] = "May2023" + run_period["name"][9:]
                            run_period["end_test_date"] = "2025-03-16 00:00:00"
                    print("run_period", run_period["name"])
                    local_fingerprint["endDateString"] = run_period["end_date"]
                    local_fingerprint["startTestDateString"] = run_period["end_date"]
                    local_fingerprint["endTestDateString"] = run_period["end_test_date"]
                    local_fingerprint["initial_pool_value"] = pool_val
                    local_fingerprint["fees"] = fee
                    local_fingerprint["gas_cost"] = gas
                    local_fingerprint["noise_trader_ratio"] = noise_trader_ratio

                    end_date = run_period["end_date"]
                    tokens = trial["tokens"]
                    bout_offset = trial["bout_offset"]
                    chunk_period = trial["chunk_period"]
                    init_pool_val = trial["initial_pool_value"]
                    rule = trial["rule"]
                    # Run simulations
                    if "log_k" in params:
                        params["log_k"] = params["log_k"] * jnp.ones_like(
                            params["initial_weights_logits"]
                        )
                    if "k" in params:
                        params["k"] = params["k"] * jnp.ones_like(
                            params["initial_weights_logits"]
                        )
                    if "logit_lamb" in params:
                        params["logit_lamb"] = params["logit_lamb"] * jnp.ones_like(
                            params["initial_weights_logits"]
                        )
                    if "logit_delta_lamb" in params:
                        params["logit_delta_lamb"] = params[
                            "logit_delta_lamb"
                        ] * jnp.ones_like(params["initial_weights_logits"])

                    # Apply daily-to-hourly conversion if requested
                    if convert_daily_to_hourly:
                        params, local_fingerprint = convert_daily_to_hourly_params(
                            params, local_fingerprint, scale_k_by_frequency
                        )
                        # Update chunk_period for plot prefix after conversion
                        chunk_period = local_fingerprint["chunk_period"]

                    base_plot_prefix = f"{tokens}_{rule}_{pool_val}_{fee}_{gas}_{noise_trader_ratio}_chunk_{chunk_period}_param_{trial['study_id'][4:][:5]}_trial_{trial['trial_number']}_runperiod_{run_period['name']}"

                    # Add suffix to indicate conversion type if applied
                    if convert_daily_to_hourly:
                        conversion_suffix = "_hourly_k_scaled" if scale_k_by_frequency else "_hourly_k_unscaled"
                        base_plot_prefix += conversion_suffix

                    run_filename = f"analysis_{base_plot_prefix}.json"
                    if os.path.exists(output_path / run_filename) and do_plots==False:
                        print(f"Skipping {run_filename} because it already exists")
                        with open(output_path / run_filename, "r") as f:
                            result = json.load(f)
                            results.append(result)
                        continue
                    # else:
                    #     print(f"Skipping {run_filename} because we dont have time")
                    #     continue

                    train_dict, test_dict = do_run_on_historic_data(
                        local_fingerprint,
                        params=params,
                        do_test_period=True,
                        verbose=False,
                        price_data=price_data,
                    )

                    # Run continuous test simulation
                    continuous_fingerprint = local_fingerprint.copy()
                    continuous_fingerprint["endDateString"] = local_fingerprint["endTestDateString"]
                    shifted_end = pd.to_datetime(continuous_fingerprint["endDateString"]) - pd.Timedelta(minutes=1)
                    continuous_fingerprint["endDateString"] = str(shifted_end.strftime("%Y-%m-%d %H:%M:%S"))
                    continuous_dict = do_run_on_historic_data(
                        continuous_fingerprint,
                        params=params,
                        do_test_period=False,
                        verbose=False,
                        price_data=price_data,
                    )
                    # Store prices and remove from result dicts
                    if prices["train_prices"] is None:
                        prices["train_prices"] = train_dict.pop("prices")
                        prices["test_prices"] = test_dict.pop("prices")
                        prices["continuous_test_prices"] = continuous_dict.pop("prices")
                    else:
                        train_dict.pop("prices", None)
                        test_dict.pop("prices", None)
                        continuous_dict.pop("prices", None)

                    # (train_dict["reserves"][0]*train_dict["prices"][-1]).sum()
                    if do_plots:
                        print("top of plots")
                        train_dict["prices"] = prices["train_prices"]
                        plot_weights(
                            train_dict,
                            local_fingerprint,
                            plot_prefix=f"train_{base_plot_prefix}",
                            # plot_dir=plot_dir,
                        )
                        # do_weight_change_as_rebalances_plots(
                        #     train_dict,
                        #     run_fingerprint,
                        #     plot_prefix=f"train_{tokens}_{rule}_{pool_val}_{fee}_{gas}_end_{end_date}_param_{i}",
                        # )
                        train_dict["fingerprint"] = local_fingerprint
                        plot_values(
                            {"Optimized_QuantAMM_pool_"+'-'.join(tokens)+"_rule_"+rule: train_dict},
                            tokens,
                            suffix=f"train_{base_plot_prefix}",
                            plot_dir=plot_dir,
                        )
                        del train_dict["prices"]
                        test_dict["prices"] = prices["test_prices"]
                        test_dict["fingerprint"] = local_fingerprint.copy()
                        test_dict["fingerprint"]["startDateString"] = local_fingerprint["startTestDateString"]
                        test_dict["fingerprint"]["endDateString"] = local_fingerprint["endTestDateString"]
                        plot_values(
                            {
                                "Optimized_QuantAMM_pool_"
                                + "-".join(tokens)
                                + "_rule_"
                                + rule: test_dict
                            },
                            tokens,
                            suffix=f"test_{base_plot_prefix}",
                            plot_dir=plot_dir,
                        )
                        del test_dict["prices"]
                        continuous_dict["prices"] = prices["continuous_test_prices"]
                        continuous_dict["fingerprint"] = continuous_fingerprint
                        plot_values(
                            {
                                "Optimized_QuantAMM_pool_"
                                + "-".join(tokens)
                                + "_rule_"
                                + rule: continuous_dict
                            },
                            tokens,
                            suffix=f"continuous_run_{base_plot_prefix}",
                            plot_white_line=True,
                            white_line_date=test_dict["fingerprint"]["startDateString"],
                            plot_dir=plot_dir,
                        )
                        print("run_period: ", run_period)
                        print("white line date", test_dict["fingerprint"]["startDateString"])
                        del continuous_dict["prices"]
                        continuous_test_results = {
                            "value": continuous_dict["value"][
                                len(train_dict["value"]) : len(train_dict["value"])
                                + len(test_dict["value"])
                            ],
                            "reserves": continuous_dict["reserves"][
                                len(train_dict["reserves"]) : len(
                                    train_dict["reserves"]
                                )
                                + len(test_dict["reserves"])
                            ],
                            "prices": prices["continuous_test_prices"][
                                len(train_dict["value"]) : len(
                                    train_dict["value"]
                                )
                                + len(test_dict["value"])
                            ],
                            "fingerprint": test_dict["fingerprint"],
                        }
                        def calculate_period_returns(results_dict, period_length=30 * 24 * 60):
                            """Calculate returns over different periods and strategies.
                            
                            Args:
                                results_dict: Dictionary containing 'value', 'prices', 'reserves' arrays
                                period_length: Period length in minutes (default 30 days)
                                
                            Returns:
                                List of dictionaries containing return metrics for each period
                            """
                            # Convert inputs to numpy arrays
                            test_values = np.array(results_dict["value"])
                            test_prices = np.array(results_dict["prices"]) 
                            test_reserves = np.array(results_dict["reserves"])

                            period_returns = []
                            n_tokens = len(test_prices[0])
                            uniform_weights = np.ones(n_tokens) / n_tokens

                            # Iterate over each period
                            for period_start in range(0, len(test_values), period_length):
                                period_end = min(period_start + period_length, len(test_values))

                                # print("period_start", period_start)
                                # print("period_end", period_end)
                                # print("period length", period_end - period_start)

                                # Get strategy values for this period
                                start_value = test_values[period_start]
                                end_value = test_values[period_end - 1]
                                strategy_return = end_value / start_value - 1

                                # Calculate initial reserves and prices
                                initial_period_reserves = test_reserves[period_start]
                                start_prices = test_prices[period_start]
                                end_prices = test_prices[period_end - 1]

                                # Calculate weights and price ratios
                                initial_period_weights = (initial_period_reserves * start_prices) / np.sum((initial_period_reserves * start_prices))
                                price_ratios = end_prices / start_prices

                                # Calculate end values for different strategies
                                end_balance_pool_value = start_value * np.prod((price_ratios) ** initial_period_weights)
                                end_uniform_balance_pool_value = start_value * np.prod((price_ratios) ** uniform_weights)
                                hodl_end_value = np.sum(initial_period_reserves * end_prices)

                                # Calculate returns
                                hodl_return = hodl_end_value / start_value - 1
                                returns_over_hodl = (end_value) / (hodl_end_value) - 1
                                returns_over_balancer = (end_value) / (end_balance_pool_value) - 1
                                returns_over_uniform_balancer = (end_value) / (end_uniform_balance_pool_value) - 1

                                # Set returns_over_balancer to 0 if within 10^-12 of 0
                                if abs(returns_over_balancer) < 1e-12:
                                    returns_over_balancer = 0.0
                                period_returns.append({
                                    "period": period_start // period_length,
                                    "strategy_return": strategy_return,
                                    "returns_over_hodl": returns_over_hodl,
                                    "hodl_return": hodl_return,
                                    "returns_over_balancer": returns_over_balancer, 
                                    "returns_over_uniform_balancer": returns_over_uniform_balancer,
                                    "annualised_strategy_return": (1 + strategy_return) ** (365 * 24 * 60 / period_length) - 1,
                                    "annualised_returns_over_hodl": (1 + returns_over_hodl) ** (365 * 24 * 60 / period_length) - 1,
                                    "annualised_returns_over_balancer": (1 + returns_over_balancer) ** (365 * 24 * 60 / period_length) - 1,
                                    "annualised_returns_over_uniform_balancer": (1 + returns_over_uniform_balancer) ** (365 * 24 * 60 / period_length) - 1,
                                })

                            print("\nPeriod Annualised Returns Over HODL:")
                            for period_data in period_returns:
                                print(f"Period {period_data['period']}: {100.0 * period_data['annualised_returns_over_hodl']}")
                            print("--------------------------------")
                            print("\nPeriod Annualised Returns:")
                            for period_data in period_returns:
                                print(
                                    f"Period {period_data['period']}: {100.0 * period_data['annualised_strategy_return']}"
                                )
                            print("--------------------------------")
                            print("\nPeriod Annualised Returns Over Balancer:")
                            for period_data in period_returns:
                                print(f"Period {period_data['period']}: {100.0 * period_data['annualised_returns_over_balancer']}")
                            print("\nPeriod Annualised Returns Over Uniform Balancer:")
                            for period_data in period_returns:
                                print(f"Period {period_data['period']}: {100.0 * period_data['annualised_returns_over_uniform_balancer']}")

                            return period_returns

                        # Calculate monthly returns over HODL (using fixed 30-day periods)
                        print("="*100)
                        print("calculating monthly returns test")
                        calculate_period_returns(continuous_test_results.copy(), period_length=30 * 24 * 60)
                        # print("calculating weekly returns test")
                        # calculate_period_returns(continuous_test_results.copy(), period_length=7 * 24 * 60)
                        print("calculating monthly returns train")
                        train_dict["prices"] = prices["train_prices"]
                        calculate_period_returns(train_dict.copy(), period_length=30 * 24 * 60)
                        # print("calculating weekly returns train")
                        # calculate_period_returns(train_dict.copy(), period_length=7 * 24 * 60)
                        print("="*100)
                        # if "2024" in run_period["name"]:
                        #     raise Exception("Stop here")
                        # Print period returns

                        plot_values(
                            {
                                "Optimized_QuantAMM_pool_"
                                + "-".join(tokens)
                                + "_rule_"
                                + rule: continuous_test_results
                            },
                            tokens,
                            suffix=f"continuous_test_{base_plot_prefix}",
                            plot_dir=plot_dir,
                        )
                        plot_weights(
                            continuous_test_results,
                            continuous_test_results["fingerprint"],
                            plot_prefix=f"continuous_test_{base_plot_prefix}",
                            # plot_dir=plot_dir,
                        )
                        uniform_hodl_return = plot_values(
                                {
                                    "Optimized_QuantAMM_pool_"
                                    + "-".join(tokens)
                                    + "_rule_"
                                    + rule: continuous_test_results
                                },
                                tokens,
                                suffix=f"continuous_test_uniformhodl_{base_plot_prefix}",
                                initial_hodl_weights="uniform",
                                plot_dir=plot_dir,
                            )
                        del continuous_test_results
                    else:
                        # Calculate uniform HODL return for continuous test period
                        initial_value = continuous_dict["value"][
                                len(train_dict["value"]) : len(train_dict["value"])
                                + len(test_dict["value"])
                            ][0]
                        initial_prices = prices["continuous_test_prices"][
                                len(train_dict["value"]) : len(
                                    train_dict["value"]
                                )
                                + len(test_dict["value"])
                            ][0]
                        final_prices = prices["continuous_test_prices"][::1440][-1]
                        # Calculate uniform weights
                        n_tokens = len(tokens)
                        uniform_weights = np.ones(n_tokens) / n_tokens
                        uniform_reserves = initial_value * uniform_weights / initial_prices
                        # Calculate return from uniform HODL strategy
                        price_ratios = final_prices / initial_prices
                        uniform_hodl_return = continuous_dict["value"][::1440][-1] / np.sum(uniform_reserves * final_prices) - 1
                    # Calculate metrics for training period
                    train_metrics = calculate_period_metrics(train_dict, prices["train_prices"])
                    # Prefix each key with "train_"
                    train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                    # Calculate metrics for test period
                    test_metrics = calculate_period_metrics(test_dict, prices["test_prices"])
                    # Prefix each key with "test_"
                    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}

                    # Calculate metrics for continuous test period
                    continuous_test_metrics = calculate_continuous_test_metrics(
                        continuous_dict,
                        len(train_dict["value"]),
                        len(test_dict["value"]),
                        prices["continuous_test_prices"]
                    )
                    continuous_metrics = calculate_period_metrics(continuous_dict, prices["continuous_test_prices"])

                    continuous_metrics = {f"continuous_{k}": v for k, v in continuous_metrics.items()}
                    # Store results
                    result = {
                        "study_id": trial["study_id"],
                        "trial_number": trial["trial_number"],
                        "tokens": tokens,
                        "rule": rule,
                        "run_period": run_period["name"],
                        "original_pool_value": init_pool_val,
                        "original_fees": run_fingerprint["fees"],
                        "original_gas_cost": run_fingerprint["gas_cost"],
                        "original_objective": trial["objective"],
                        "original_return_val": trial["return_val"],
                        "original_train_value": trial["train_value"],
                        "original_test_value": trial["test_value"],
                        "noise_trader_ratio": noise_trader_ratio,
                        "analysis_pool_value": pool_val,
                        "analysis_fees": fee,
                        "analysis_gas_cost": gas,
                        "bout_offset": int(trial["bout_offset"]),
                        "chunk_period": int(trial["chunk_period"]),
                        "run_period_start_date": run_period["start_date"],
                        "start_date": trial.get("start_date"),
                        "end_date": trial.get("end_date"),
                        "end_test_date": trial.get("end_test_date"),
                        "run_period_start_date": run_period["start_date"],
                        "run_period_end_date": run_period["end_date"],
                        "run_period_end_test_date": run_period["end_test_date"],
                        "weight_interpolation_period": trial.get(
                            "weight_interpolation_period"
                        ),
                        "minimum_weight": trial.get("minimum_weight"),
                        **train_metrics,
                        **test_metrics,
                        **continuous_test_metrics,
                        **continuous_metrics,
                        "uniform_continuous_test_hodl_return": float(
                            uniform_hodl_return
                        ),
                        "params": tree_map(
                            lambda x: (x.tolist() if isinstance(x, jnp.ndarray) else x),
                            params,
                        ),
                        "optimisation_method": run_fingerprint["optimisation_settings"][
                            "method"
                        ],
                        "sample_method": run_fingerprint["optimisation_settings"].get(
                            "sample_method"
                        ),
                        "use_gradient_clipping": run_fingerprint["optimisation_settings"].get(
                            "use_gradient_clipping"
                        ),
                        "clip_norm": run_fingerprint["optimisation_settings"].get(
                            "clip_norm"
                        ),
                        "optimiser": run_fingerprint["optimisation_settings"].get(
                            "optimiser"
                        ),
                        "learning_rate": run_fingerprint["optimisation_settings"].get(
                            "base_lr"
                        ),
                        "batch_size": run_fingerprint["optimisation_settings"].get(
                            "batch_size"
                        ),
                        "use_plateau_decay": run_fingerprint["optimisation_settings"].get(
                            "use_plateau_decay"
                        ),
                        "lr_schedule_type": run_fingerprint["optimisation_settings"].get(
                            "lr_schedule_type"
                        ),
                        "warmup_steps": run_fingerprint["optimisation_settings"].get(
                            "warmup_steps"
                        ),
                        "ste_max_change": run_fingerprint.get(
                            "ste_max_change"
                        ),
                        "ste_min_max_weight": run_fingerprint.get(
                            "ste_min_max_weight"
                        ),
                    }
                    print(result)
                    results.append(result)
                    # Save individual result files
                    for filename in [
                        f"analysis_{base_plot_prefix}.json",
                        # f"analysis_{base_plot_prefix}_studyid_{trial['study_id']}_trialno_{trial['trial_number']}_poolval_{pool_val:.0f}_fees_{fee:.4f}_gas_{gas:.1f}_noise_{noise_trader_ratio:.2f}_bout_{trial['bout_offset']}_chunk_{trial['chunk_period']}_end_{trial['end_date']}_param_{0}_usePF_{use_pareto_frontier}.json",
                    ]:
                        with open(output_path / filename, "w") as f:
                            json.dump(result, f, indent=2)
                    gc.collect()

                    del result
                    del train_dict
                    del test_dict
                    del continuous_dict
                    del prices
                    clear_caches()
                    gc.collect()
                gc.collect()
        # Save all results to a single file
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        with open(results_path, "r") as f:
            results = json.load(f)
        print(f"Loading existing results from {results_path}")
    # After collecting all results, but before saving:
    results_df = pd.DataFrame(results)

    # Identify non-varying columns (keys that uniquely identify a trial)
    id_columns = [
        "study_id",
        "trial_number",
        "tokens",
        "rule",
        "start_date",
        "end_date",
        "original_pool_value",
        "original_fees",
        "original_gas_cost",
        "original_objective",
        "original_return_val",
        "original_train_value",
        "original_test_value",
        "noise_trader_ratio",
        "analysis_pool_value",
        "analysis_fees",
        "analysis_gas_cost",
        "bout_offset",
        "chunk_period",
        "weight_interpolation_period",
        "minimum_weight",
        "optimisation_method",
        "sample_method",
        "use_gradient_clipping",
        "clip_norm",
        "optimiser",
        "learning_rate",
        "batch_size",
        "use_plateau_decay",
        "lr_schedule_type",
        "warmup_steps",
        "ste_max_change",
        "ste_min_max_weight",
    ]

    # Create DataFrame with explicit types
    df = pd.DataFrame(results)
    df["study_id"] = df["study_id"].astype(str)
    df["trial_number"] = df["trial_number"].astype(float)
    df["Strategy"] = (
        df["Strategy"].astype("category") if "Strategy" in df.columns else None
    )
    # Get all columns that need period prefix
    metric_columns = [
        col
        for col in results_df.columns
        if col not in id_columns and col != "run_period" and col != "params" and col != "noise_trader_ratio" and col != "analysis_pool_value" and col != "analysis_fees" and col != "analysis_gas_cost"
    ]

    # Create a mapping of (study_id, trial_number) to params
    params_dict = (
        results_df.groupby(["study_id", "trial_number"])["params"].first().to_dict()
    )

    if 'tokens' in results_df.columns:
        results_df['tokens'] = results_df['tokens'].apply(tuple)
    # Set index to id columns for proper pivoting
    results_df = results_df.set_index(id_columns)

    # Pivot all period-specific columns at once
    period_data = []
    for col in metric_columns:
        pivoted = results_df.pivot(columns="run_period", values=col)
        cols = pivoted.columns
        cols = [col[:-4] if col.endswith("test") else col for col in cols]
        pivoted.columns = [f"{period}_{col}" for period in cols]
        period_data.append(pivoted)

    # Combine all pivoted data
    transformed_df = pd.concat(period_data, axis=1)

    # Reset index to get id columns back
    transformed_df = transformed_df.reset_index()

    # Add params column
    transformed_df["params"] = transformed_df.apply(
        lambda row: params_dict[(row["study_id"], row["trial_number"])], axis=1
    )

    # Convert dates to datetime for sorting
    transformed_df["start_date"] = pd.to_datetime(transformed_df["start_date"])
    transformed_df["end_date"] = pd.to_datetime(transformed_df["end_date"])

    # Sort the DataFrame
    transformed_df = transformed_df.sort_values(
        by=[
            "bout_offset",
            "chunk_period",
            "tokens",
            "original_return_val",
            "start_date",
            "end_date",
            "minimum_weight",
            "rule"
        ],
        ascending=[
            False,
            True,
            True,    # tokens ascending
            True,    # return_val ascending
            True,    # start date ascending
            False,   # end date descending
            True,    # minimum weight ascending
            True     # rule alphabetically
        ]
    )

    # Reset index after sorting
    transformed_df = transformed_df.reset_index(drop=True)

    # Add conversion suffix to output filenames if conversion was applied
    conversion_suffix = ""
    if convert_daily_to_hourly:
        conversion_suffix = "_hourly_k_scaled" if scale_k_by_frequency else "_hourly_k_unscaled"

    # Save transformed DataFrame
    filename = f"analysis_results_{Path(base_dir).name}_{output_string}{conversion_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    transformed_df.to_csv(Path(base_dir) / filename, index=False)

    # Create simplified DataFrame matching BTF format
    simplified_df = pd.DataFrame()

    # Map basic columns
    simplified_df["Tokens"] = transformed_df["tokens"]
    simplified_df["Rule"] = transformed_df["rule"].str.replace("_", " ").str.title()
    simplified_df["Objective"] = transformed_df["original_return_val"]
    simplified_df["Min weight"] = transformed_df["minimum_weight"].apply(lambda x: f"{float(x)*100:.0f}%")
    simplified_df["Start date"] = pd.to_datetime(transformed_df["start_date"]).dt.strftime("%Y")
    simplified_df["start test"] = pd.to_datetime(transformed_df["end_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    simplified_df["Run ID"] = transformed_df["study_id"]
    simplified_df["Iteration no"] = transformed_df["trial_number"]
    simplified_df["Best obj"] = transformed_df["original_objective"]
    simplified_df["Test Returns Over HODL"] = transformed_df["original_test_value"]
    simplified_df["Train Returns Over HODL"] = transformed_df["original_train_value"]

    # Add conversion metadata to Comments
    conversion_info = ""
    if convert_daily_to_hourly:
        k_scaling = "with k scaling" if scale_k_by_frequency else "without k scaling"
        conversion_info = f"Converted daily->hourly ({k_scaling})"
    simplified_df["Comments"] = conversion_info

    simplified_df["Params"] = transformed_df["params"]

    # Add metrics for each period
    periods = np.unique(results_df["run_period"]).tolist()
    for period in periods:
        if period.endswith("test"):
            period = period[:-4]
        period_prefix = f"{period}_"
        # simplified_df[f"Returns over HODL test ({period})"] = transformed_df[f"{period_prefix}continuous_test_returns_over_hodl"]
        simplified_df[f"Returns test ({period})"] = transformed_df[f"{period_prefix}continuous_test_return"]
        simplified_df[f"Sharpe test ({period})"] = transformed_df[f"{period_prefix}continuous_test_sharpe"]
        simplified_df[f"Annualized Ulcer Index [M] test ({period})"] = transformed_df[f"{period_prefix}continuous_test_ulcer"]
        simplified_df[f"Annualized Calmer Ratio [M] test ({period})"] = transformed_df[f"{period_prefix}continuous_test_calmar"]
        simplified_df[f"Returns over HODL train ({period})"] = transformed_df[
            f"{period_prefix}train_returns_over_hodl"
        ]
        simplified_df[f"Returns train ({period})"] = transformed_df[
            f"{period_prefix}train_return"
        ]
        simplified_df[f"Sharpe train ({period})"] = transformed_df[
            f"{period_prefix}train_sharpe"
        ]
        simplified_df[f"Annualized Ulcer Index [M] train ({period})"] = transformed_df[
            f"{period_prefix}continuous_test_ulcer"
        ]
        simplified_df[f"Annualized Calmer Ratio [M] train ({period})"] = transformed_df[
            f"{period_prefix}train_calmar"
        ]
        simplified_df[f"Returns over HODL test ({period})"] = transformed_df[
            f"{period_prefix}uniform_continuous_test_hodl_return"
        ]
    simplified_df["noise_trader_ratio"] = transformed_df["noise_trader_ratio"]
    simplified_df["analysis_pool_value"] = transformed_df["analysis_pool_value"]
    simplified_df["analysis_fees"] = transformed_df["analysis_fees"]
    simplified_df["analysis_gas_cost"] = transformed_df["analysis_gas_cost"]
    simplified_df["optimisation_method"] = transformed_df["optimisation_method"]
    simplified_df["sample_method"] = transformed_df["sample_method"]
    simplified_df["use_gradient_clipping"] = transformed_df["use_gradient_clipping"]
    simplified_df["clip_norm"] = transformed_df["clip_norm"]
    simplified_df["optimiser"] = transformed_df["optimiser"]
    simplified_df["learning_rate"] = transformed_df["learning_rate"]
    simplified_df["batch_size"] = transformed_df["batch_size"]
    simplified_df["use_plateau_decay"] = transformed_df["use_plateau_decay"]
    simplified_df["lr_schedule_type"] = transformed_df["lr_schedule_type"]
    simplified_df["warmup_steps"] = transformed_df["warmup_steps"]
    simplified_df["ste_max_change"] = transformed_df["ste_max_change"]
    simplified_df["ste_min_max_weight"] = transformed_df["ste_min_max_weight"]

    # Save simplified CSV
    simplified_filename = f"simplified_analysis_{Path(base_dir).name}_{output_string}{conversion_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    simplified_df.to_csv(Path(base_dir) / simplified_filename, index=False, float_format='%.10f')

    # Continue with existing JSON save for compatibility
    results_path = (
        Path(base_dir)
        / "analysis_results"
        / f"analysis_unified_results_{Path(base_dir).name}_{output_string}{conversion_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return simplified_df


def load_run_fingerprints(
    base_dir,
    trials_to_analyze,
    load_method="best_objective",
    tokens=''
):
    """Analyze specific trials from run files.

    Parameters
    ----------
    base_dir : str
        Directory containing the run files
    trials_to_analyze : list[dict]
        List of dicts containing study_id and trial_number to analyze
        Example: [{"study_id": "run_XXXXX", "trial_number": 42}, ...]
    return_val : str, optional
        Metric to optimize for, by default "sharpe"
    load_method : str, optional
        Method for selecting parameter sets, by default "best_objective"
    use_pareto_frontier : bool, optional
        Whether to use pareto frontier analysis, by default True
    """
    base_path = Path(base_dir)
    all_trials = []

    keep_top = 1.0

    filename = "sgd_analysis_result_" + load_method + "_keeptop_" + str(keep_top) + "_" + "_".join(tokens) + ".csv"

    if (Path(base_dir) / filename).exists():
        df = pd.read_csv(Path(base_dir) / filename)
        # Convert string columns to dicts
        if "params" in df.columns:
            df["params"] = df["params"].str.replace(", dtype=float64", "")
            df["params"] = df["params"].str.replace(",      dtype=float64", "")
            df["params"] = df["params"].str.replace(",\s*dtype=float64", "")
            df["params"] = df["params"].str.replace("Array(", "")
            df["params"] = df["params"].str.replace(")", "")
            # Drop rows where params contains 'nan'
            df = df[~df["params"].astype(str).str.contains('nan')]
            df["params"] = df["params"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            df["params"] = df["params"].apply(lambda x: {k: jnp.array(v) for k, v in x.items()})
        if "tokens" in df.columns:
            df["tokens"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        if "run_fingerprint" in df.columns:
            df["run_fingerprint"] = df["run_fingerprint"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    if len(trials_to_analyze) > 0:
        # Filter DataFrame to only include specified trials
        mask = pd.DataFrame(False, index=df.index, columns=["match"])
        for trial_info in trials_to_analyze:
            mask["match"] |= (df["study_id"] == trial_info["study_id"]) & (
                df["trial_number"] == trial_info["trial_number"]
            )
        filtered_df = df[mask["match"]]
        if filtered_df.empty:
            raise ValueError("No matching trials found")
    else:
        filtered_df = df
    # Write run fingerprints as JSONL file
    output_path = Path(base_dir) / f"run_fingerprints_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in filtered_df.iterrows():
            json.dump(row["run_fingerprint"], f)
            f.write("\n")

    return all_trials

if __name__ == "__main__":
    print("Starting analysis...")
    print("="*100)
    print("Analyzing...")

    
    analyze_specific_trials(
        base_dir="./results", # alther this to point to the relative location of the run files.
        trials_to_analyze=[],
        load_method="best_train_objective",
        tokens=["BTC","ETH", "USDC"], # alter this to be the token selection
    )

    analyze_specific_trials(
        base_dir="./results", # alther this to point to the relative location of the run files.
        trials_to_analyze=[],
        load_method="best_objective",
        tokens=["BTC", "ETH", "USDC"], # alter this to be the token selection
    )
