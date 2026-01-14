import jax.numpy as jnp
from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
)
from quantammsim.core_simulator.windowing_utils import filter_reserves_by_data_indices
from quantammsim.runners.jax_runners import (
    do_run_on_historic_data_with_provided_coarse_weights,
)
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import warnings
import os
import json
import re
from datetime import timezone

warnings.filterwarnings("ignore")

PLOT_DIR = './'

def resample_to_minute_grid(df, timestamp_col='datetime'):
    """
    Resample a randomly sampled dataframe to uniform 1-minute grid using forward fill.
    
    Args:
        df: DataFrame with irregular timestamps
        timestamp_col: Name of the timestamp column
    
    Returns:
        DataFrame resampled to 1-minute intervals
    """
    # Convert timestamp to datetime if not already
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Set timestamp as index
    df_indexed = df.set_index(timestamp_col)

    # Resample to 1-minute intervals with forward fill
    resampled = df_indexed.resample('1T').ffill()

    # Reset index to get timestamp back as column
    return resampled.reset_index()


def resample_to_minute_grid_interpolation(df, timestamp_col="datetime"):
    """
    Resample a randomly sampled dataframe to uniform 1-minute grid using forward fill.

    Args:
        df: DataFrame with irregular timestamps
        timestamp_col: Name of the timestamp column

    Returns:
        DataFrame resampled to 1-minute intervals
    """
    # Convert timestamp to datetime if not already
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Set timestamp as index
    df_indexed = df.set_index(timestamp_col)

    # Resample to 1-minute intervals with forward fill
    resampled = df_indexed.resample("1T").interpolate(method="linear")

    # Reset index to get timestamp back as column
    return resampled.reset_index()


def resample_to_minute_grid_reindex(df, timestamp_col='datetime'):
    """
    Resample using reindex method (similar to your existing code patterns).
    """
    # Convert timestamp to datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df_indexed = df.set_index(timestamp_col)
    
    # Create minute-level index
    minute_range = pd.date_range(
        start=df_indexed.index.min(), 
        end=df_indexed.index.max(), 
        freq='1T'
    )
    
    # Reindex with forward fill
    resampled = df_indexed.reindex(minute_range, method='ffill')
    
    return resampled.reset_index()

def load_scraped_pool_data(data_dir="./sonic_macro"):
    """
    Load and process scraped pool data from CSV files.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the scraped CSV files
        
    Returns
    -------
    dict
        Dictionary containing:
        - coarse_weights: dict with 'weights' and 'unix_values' arrays
        - fees_df: DataFrame with unix timestamps and fee values
        - gas_cost_df: DataFrame with unix timestamps and gas cost values
        - lp_supply_df: DataFrame with unix timestamps and LP supply values
        - tokens: list of token symbols in alphabetical order
    """
    data_path = Path(data_dir)

    # Find all CSV files matching the pattern
    csv_files = list(data_path.glob("reserves_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Sort files by block number (first number in filename)
    def extract_block_number(filename):
        match = re.search(r'reserves_.*_(\d+)_\d+\.csv', filename.name)
        return int(match.group(1)) if match else 0

    csv_files.sort(key=extract_block_number)

    print(f"Found {len(csv_files)} CSV files")

    # Load and concatenate all CSV files
    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # check that the unix column is in milliseconds
            if df['timestamp'].max() < 1000000000000:
                df['timestamp'] = df['timestamp'] * 1000
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue

    if not dfs:
        raise ValueError("No valid CSV files could be loaded")

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    # Sort by timestamp to ensure chronological order
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    # Extract token columns and sort alphabetically
    token_columns = [col for col in combined_df.columns if col.endswith('_balance')]
    tokens = [col.replace('_balance', '') for col in token_columns]
    # Get the indices that would sort the tokens alphabetically
    token_sort_indices = np.argsort(tokens)
    tokens.sort()  # Alphabetical order: BTC, ETH, S, USDC

    print(f"Tokens found: {tokens}")

    # Parse string arrays in balances and weights columns
    def parse_string_array(series):
        """Parse string representations of arrays"""
        parsed_arrays = []
        for val in series:
            try:
                # Remove brackets and split by comma, then convert to float
                if isinstance(val, str):
                    # Remove brackets and split
                    clean_val = val.strip('[]')
                    array_vals = [float(x.strip()) for x in clean_val.split(',')]
                    parsed_arrays.append(array_vals)
                else:
                    parsed_arrays.append([float(val)])
            except Exception as e:
                print(f"Warning: Could not parse array value: {val}, error: {e}")
                parsed_arrays.append([0.0] * len(tokens))  # Default to zeros
        return np.array(parsed_arrays)

    balances_array = parse_string_array(combined_df['balances'])
    # combined_df["balances"] = balances_array
    weights_first_four = parse_string_array(combined_df['weights_first_four'])
    # combined_df["weights_first_four"] = weights_first_four
    weights_second_four = parse_string_array(combined_df['weights_second_four'])
    normalized_weights = parse_string_array(combined_df['normalized_weights'])
    # combined_df["weights_second_four"] = weights_second_four
    token_rates = parse_string_array(combined_df['token_rates'])
    # combined_df["token_rates"] = token_rates
    combined_df["weights_first_four"] = [arr.tolist() for arr in weights_first_four]
    combined_df["weights_second_four"] = [arr.tolist() for arr in weights_second_four]
    combined_df["normalized_weights"] = [arr.tolist() for arr in normalized_weights]
    combined_df["balances"] = [arr.tolist() for arr in balances_array]
    combined_df["token_rates"] = [arr.tolist() for arr in token_rates]
    # Reorder balances and weights to match alphabetical token order
    token_balance_cols = [f"{token}_balance" for token in tokens]
    token_price_cols = [f"{token}_price" for token in tokens]

    # find first entry where total_supply is > 0
    print(f"Loaded {len(combined_df)} total rows")
    first_entry = combined_df[combined_df['total_supply'] > 0].iloc[0]
    print(f"First entry where total_supply is > 0: {first_entry}")
    combined_df = combined_df.iloc[first_entry.name:]

    # Resample to exact minutes after filtering
    print("Resampling data to exact minutes...")

    # Convert timestamp to datetime
    combined_df["datetime"] = pd.to_datetime(
        combined_df["timestamp"], unit="ms"
    )

    # Check data density before resampling
    print(f"Original data range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    print(f"Original data points: {len(combined_df)}")
    print("First 10 timestamps:")
    print(combined_df['datetime'].head(10))
    print("Time gaps between consecutive points:")
    time_diffs = combined_df['datetime'].diff().dropna()
    print(time_diffs.head(10))
    print(f"Min gap: {time_diffs.min()}, Max gap: {time_diffs.max()}")

    # resample this data frame using linear interpolation to get it to regular on-the-minute data
    # combined_df = combined_df.set_index('datetime').resample('1T').asfreq().fillna(method='ffill').reset_index()    # Find first midnight (00:00:00) in the data
    # Create a complete minute-level index that aligns with your data
    # Debug what's in the dataframe before forward fill
    print("Before forward fill:")
    print(combined_df.head())
    print("Any non-null values?")
    print(combined_df.notnull().any())
    combined_df = resample_to_minute_grid(combined_df)[1:]

    # recaculate the unix values
    combined_df["timestamp"] = combined_df["datetime"].astype(int) / 1000000
    midnight_mask = combined_df["datetime"].dt.time == pd.Timestamp("00:00:00").time()
    if not midnight_mask.any():
        print("Warning: No midnight timestamps found, using first and last timestamps")
        first_midnight = combined_df["datetime"].iloc[0]
        last_midnight = combined_df["datetime"].iloc[-1]
    else:
        first_midnight = combined_df[midnight_mask].iloc[0]["datetime"]
        last_midnight = combined_df[midnight_mask].iloc[-1]["datetime"]

    print(f"First midnight: {first_midnight}")
    print(f"Last midnight: {last_midnight}")
    # Filter data to start at first midnight and end at last midnight
    combined_df = combined_df[
        (combined_df["datetime"] >= first_midnight)
        & (combined_df["datetime"] <= last_midnight)
    ].reset_index(drop=True)

    # Drop the temporary datetime column
    combined_df = combined_df.drop('datetime', axis=1)

    print(f"Final data length: {len(combined_df)} rows")

    # Extract balances and prices in alphabetical order
    ordered_balances = np.zeros((len(combined_df), len(tokens)))
    ordered_prices = np.zeros((len(combined_df), len(tokens)))

    for i, token in enumerate(tokens):
        if f"{token}_balance" in combined_df.columns:
            ordered_balances[:, i] = combined_df[f"{token}_balance"].values
        if f"{token}_price" in combined_df.columns:
            ordered_prices[:, i] = combined_df[f"{token}_price"].values

    # Combine weights (assuming first_four and second_four are concatenated)
    weights_array = np.array([np.array(w) for w in combined_df["normalized_weights"]])

    # Create coarse weights (sample every chunk_period minutes)
    chunk_period = 1440  # Daily sampling

    coarse_weights = {
        "weights": jnp.array(weights_array[::chunk_period][:, token_sort_indices]),
        "unix_values": jnp.array(combined_df["timestamp"].iloc[::chunk_period].values),
    }
    # normalise the weights
    coarse_weights["weights"] = coarse_weights["weights"] / np.sum(coarse_weights["weights"], axis=1, keepdims=True)
    # Create fee arrays
    fees_df = pd.DataFrame({
        'unix': combined_df['timestamp'].values,
        'fees': combined_df['fee_data'].values
    })

    # Create gas cost array (using 50th percentile)
    # TODO this is not correct, we should use the actual gas cost of a trade
    # so we need to know the value over time of the gas token and the combine
    # all this with the amount of gas used for a trade.
    gas_cost_df = pd.DataFrame({
        'unix': combined_df['timestamp'].values,
        'trade_gas_cost_usd': combined_df['gas_price_50p_gwei'].values
    })

    # Create LP supply array
    lp_supply_df = pd.DataFrame({
        'unix': combined_df['timestamp'].values,
        'lp_supply': combined_df['total_supply'].values
    })

    # Create arb fees array (using same as fees for now)
    arb_fees_df = pd.DataFrame({
        'unix': combined_df['timestamp'].values,
        'arb_fees': combined_df['fee_data'].values
    })
    return {
        'coarse_weights': coarse_weights,
        'fees_df': fees_df,
        'gas_cost_df': gas_cost_df,
        'lp_supply_df': lp_supply_df,
        'arb_fees_df': arb_fees_df,
        'tokens': tokens,
        'full_data': combined_df,  # Keep full data for debugging
        'ordered_balances': ordered_balances,  # Actual reserves from data
        'ordered_prices': ordered_prices,  # Actual prices from data
    }

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
        return f"$\\mathrm{{QuantAMM\\ {rule}}}$"

    return name_to_latex_name_OG(name)

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

COLOR = "#E6CE97"
sns.set(
    rc={
        "text.color": COLOR,
        "axes.labelcolor": COLOR,
        "xtick.color": COLOR,
        "ytick.color": COLOR,
        "figure.facecolor": "#162536",
        "axes.facecolor": "#162536",
        "text.usetex": True,
        "axes.grid": False
    }
)
def plot_weights(output_dict, run_fingerprint, plot_prefix="weights", plot_dir=None,verbose=True):
    if plot_dir is None:
        plot_dir = "./plots/"
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

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
    df["Date"] = np.tile(date_range, weights.shape[1])

    # Determine time range for appropriate tick formatting
    time_diff = end_date - start_date
    is_monthly = time_diff.days > 7  # More than a week

    # fig, ax = plt.subplots(figsize=(10, 6))
    f = mpl.figure.Figure()

    # Create stacked area plot
    pl = (
        so.Plot(df, "Date", "Weight", color="Token")
        .add(so.Area(alpha=0.7), so.Stack())
        .limit(y=(0, 1))
        .scale(color=sns.color_palette())
        .label(y="$\\mathrm{Weight}$", x="$\\mathrm{Time}$" if not is_monthly else "$\\mathrm{Date}$")
    )

    # Render the plot on our axis
    res = pl.on(f).plot()
    ax = f.axes[0]
    
    # Format x-axis based on time range
    if is_monthly:
        ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=max(1, time_diff.days // 8)))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=max(1, int(time_diff.total_seconds() / 3600) // 6)))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    
    plt.xticks(rotation=45)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Save plot
    pl.save(
        plot_path / (plot_prefix + "_weights_over_time.png"),
        dpi=700,
        bbox_inches="tight"
    )
    plt.close()

def plot_values(results, tokens, suffix="", plot_start_end=None, plot_white_line=False, white_line_date=None, plot_dir=None, initial_hodl_weights="same"):
    """Plot value over time for all runs on the same graph."""

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
        hodl_values = np.sum(hodl_reserves * prices, axis=1)
        results["HODL"] = {
            "value": hodl_values,
            "fingerprint": next(iter(results.values()))["fingerprint"].copy()
        }
        results["HODL"]["fingerprint"]["rule"] = "HODL"
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
        "$\\mathrm{QuantAMM\\ Momentum}$",
        "$\\mathrm{QuantAMM\\ Anti-Momentum}$",
        "$\\mathrm{QuantAMM\\ Power-Channel}$",
        "$\\mathrm{QuantAMM\\ Index}$",
        "$\\mathrm{HODL}$",
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
    ax.spines["left"].set_color(COLOR)
    ax.spines["bottom"].set_color(COLOR)
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


def plot_reserves(result, fingerprint, run_name, actual_reserves_np=None, actual_unix_values=None):
    """Create line plots showing normalized reserves over time for each token."""
    # Extract data and normalize by initial reserves
    reserves = result["reserves"] / result["reserves"][0]
    tokens = sorted(fingerprint["tokens"])

    # # Downsample to hourly data
    # reserves = reserves[::60]

    # Create DataFrame for plotting
    df_list = []
    for i, token in enumerate(tokens):
        df_list.extend(
            [
                {"Time": t, "Reserves": r, "Token": f"${token}$"}  # LaTeX formatting
                for t, r in enumerate(reserves[:, i])
            ]
        )

    df = pd.DataFrame(df_list)

    # Create date range
    start_date = datetime.strptime(fingerprint["startDateString"], "%Y-%m-%d %H:%M:%S")
    end_date = datetime.strptime(fingerprint["endDateString"], "%Y-%m-%d %H:%M:%S")
    date_range = pd.date_range(
        start=start_date, end=end_date, periods=len(df["Time"].unique())
    )

    df["Date"] = np.tile(date_range, reserves.shape[1])

    # Convert types
    df["Reserves"] = df["Reserves"].astype(float)
    df["Token"] = df["Token"].astype("category")
    
    # Determine time range for appropriate tick formatting
    time_diff = end_date - start_date
    is_monthly = time_diff.days > 7  # More than a week
    
    # Create plot
    sns.lineplot(data=df, x="Date", y="Reserves", hue="Token", linewidth=0.5)

    plt.title("$\\mathrm{Reserves\\ Over\\ Time}$", pad=20)
    plt.xlabel("$\\mathrm{Time}$" if not is_monthly else "$\\mathrm{Date}$")
    plt.ylabel("$\\mathrm{Reserves}$")

    # Format axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR)
    ax.spines["bottom"].set_color(COLOR)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    
    # Format x-axis based on time range
    if is_monthly:
        ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=max(1, time_diff.days // 8)))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
    else:
        ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=max(1, int(time_diff.total_seconds() / 3600) // 6)))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    
    plt.xticks(rotation=45)

    # Remove legend title
    ax.get_legend().set_title(None)

    # Save plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, f"reserves_{run_name}.png"), dpi=700, bbox_inches="tight"
    )
    plt.close()

    if actual_reserves_np is not None and actual_unix_values is not None:
        reserve_cols = np.unique(df["Token"].values)
        # Create a plot for each asset
        for i, col in enumerate(reserve_cols):
            plt.figure(figsize=(12, 6))

            # Plot simulated data
            sns.lineplot(
                data=df[df["Token"] == col],
                x="Time",
                y="Reserves",
                label="Simulated",
                color="blue",
                linestyle="-",
            )

            # Plot actual data
            sns.lineplot(
                x=actual_unix_values,
                y=actual_reserves_np[:, i] / actual_reserves_np[0, i],
                label="Actual",
                color="red",
                alpha=0.6,
            )
            # Customize plot
            plt.title(f'Simulated vs Actual Reserves: {col}')
            plt.xlabel("Unix Timestamp")
            plt.ylabel("Reserves")
            plt.grid(True)

            # Show plot
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOT_DIR, f"reserves_vs_real_{run_name}_{col}.png"), dpi=700, bbox_inches="tight"
            )
            plt.close()

# Default fingerprint used as base for all pools
DEFAULT_FINGERPRINT = {
    "startDateString": "2021-01-01 00:00:00",
    "endDateString": "2024-06-01 00:00:00",
    "endTestDateString": "2024-11-30 00:00:00",
    "chunk_period": 60,
    "weight_interpolation_period": 60,
    "fees": 0.0,
    "gas_cost": 0.0,
    "use_alt_lamb": False,
}

def create_config_from_scraped_data(data_dir="./sonic_macro"):
    """
    Create a configuration using scraped pool data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the scraped CSV files
        
    Returns
    -------
    dict
        Configuration dictionary for the simulation
    """
    # Load the scraped data
    scraped_data = load_scraped_pool_data(data_dir)

    # Get the first and last timestamps for date range
    first_timestamp = scraped_data['full_data']['timestamp'].iloc[0]
    last_timestamp = scraped_data['full_data']['timestamp'].iloc[-1]

    # Convert timestamps to date strings
    start_date = datetime.fromtimestamp(first_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    # Get initial reserves from the first row
    initial_weights = scraped_data['coarse_weights']['weights'][0]
    initial_reserves = scraped_data["ordered_balances"][0]
    config = {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": scraped_data['tokens'],
            "rule": "power_channel",
            "startDateString": start_date,
            "endDateString": end_date,
            "initial_pool_value": 1000000.0,  # Default pool value
            "gas_cost": 0.0,  # Will be overridden by gas_cost_df
            "fees": 0.0,  # Will be overridden by fees_df
            "endTestDateString": None,
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "minimum_weight": 0.03,
        },
        "params": {
            "initial_weights": initial_weights,
            "initial_reserves": initial_reserves,
            "log_k": jnp.array([6.0] * len(scraped_data['tokens'])),
            "logit_delta_lamb": jnp.array([0.0] * len(scraped_data['tokens'])),
            "logit_lamb": jnp.array([0.0] * len(scraped_data['tokens'])),
            "raw_exponents": jnp.array([0.0] * len(scraped_data['tokens'])),
        },
        "coarse_weights": scraped_data['coarse_weights'],
        "fees_df": scraped_data['fees_df'],
        "gas_cost_df": scraped_data['gas_cost_df'],
        "lp_supply_df": scraped_data['lp_supply_df'],
        "arb_fees_df": scraped_data['arb_fees_df'],
        "actual_unix_values": scraped_data['full_data']['timestamp'].values,
    }

    return config

def create_config_with_actual_reserves(data_dir="./sonic_macro"):
    """
    Create a configuration using actual reserves from scraped data.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the scraped CSV files
        
    Returns
    -------
    dict
        Configuration dictionary for the simulation using actual reserves
    """
    # Load the scraped data
    scraped_data = load_scraped_pool_data(data_dir)
    
    # Get the first and last timestamps for date range
    first_timestamp = scraped_data['full_data']['timestamp'].iloc[0]
    last_timestamp = scraped_data['full_data']['timestamp'].iloc[-1]
    # Convert timestamps to date strings
    start_date = datetime.fromtimestamp(first_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    # Use actual reserves from the data
    actual_reserves = scraped_data['ordered_balances']
    actual_prices = scraped_data['ordered_prices']
    
    # Calculate initial reserves (first row)
    initial_reserves = actual_reserves[0]
    
    # Create a custom coarse weights that uses actual reserves
    # Sample every chunk_period minutes
    chunk_period = 1440  # Daily sampling
    sampled_indices = np.arange(0, len(actual_reserves), chunk_period)
    
    # Calculate weights from actual reserves and prices
    sampled_reserves = actual_reserves[sampled_indices]
    sampled_prices = actual_prices[sampled_indices]
    total_values = np.sum(sampled_reserves * sampled_prices, axis=1, keepdims=True)
    actual_weights = sampled_reserves * sampled_prices / total_values
    
    coarse_weights = {
        "weights": jnp.array(actual_weights),
        "unix_values": jnp.array(scraped_data['full_data']['timestamp'].iloc[sampled_indices].values)
    }
    
    config = {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": scraped_data['tokens'],
            "rule": "power_channel",
            "startDateString": start_date,
            "endDateString": end_date,
            "initial_pool_value": 1000000.0,  # Default pool value
            "gas_cost": 0.0,  # Will be overridden by gas_cost_df
            "fees": 0.0,  # Will be overridden by fees_df
            "endTestDateString": None,
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "minimum_weight": 0.03,
        },
        "params": {
            "initial_weights": actual_weights[0],
            "initial_reserves": initial_reserves,
            "log_k": jnp.array([6.0] * len(scraped_data['tokens'])),
            "logit_delta_lamb": jnp.array([0.0] * len(scraped_data['tokens'])),
            "logit_lamb": jnp.array([0.0] * len(scraped_data['tokens'])),
            "raw_exponents": jnp.array([0.0] * len(scraped_data['tokens'])),
        },
        "coarse_weights": coarse_weights,
        "fees_df": scraped_data['fees_df'],
        "gas_cost_df": scraped_data['gas_cost_df'],
        "lp_supply_df": scraped_data['lp_supply_df'],
        "arb_fees_df": scraped_data['arb_fees_df'],
        "actual_reserves": actual_reserves,
        "actual_prices": actual_prices,
        "actual_unix_values": scraped_data['full_data']['timestamp'].values,
    }
    
    return config

def example_usage():
    """
    Example of how to use the scraped data loading functions.
    """
    print("Loading scraped pool data...")
    
    # Load the data
    scraped_data = load_scraped_pool_data("./sonic_macro")
    
    print(f"Loaded data for tokens: {scraped_data['tokens']}")
    print(f"Data shape: {scraped_data['full_data'].shape}")
    print(f"Date range: {scraped_data['full_data']['timestamp'].min()} to {scraped_data['full_data']['timestamp'].max()}")
    
    # Create configurations
    config1 = create_config_from_scraped_data("./sonic_macro")
    config2 = create_config_with_actual_reserves("./sonic_macro")
    
    print(f"Configuration 1 tokens: {config1['fingerprint']['tokens']}")
    print(f"Configuration 2 tokens: {config2['fingerprint']['tokens']}")
    
    return config1, config2

EXAMPLE_CONFIGS = {
    "scraped_pool_data": create_config_from_scraped_data(),
    "scraped_pool_data_actual_reserves": create_config_with_actual_reserves(),
}


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np
    from quantammsim.core_simulator.param_utils import (
        generate_params_combinations,
        jax_logit_lamb_to_lamb,
        lamb_to_memory_days,
        lamb_to_memory_days_clipped,
        calc_lamb,
    )
    from quantammsim.core_simulator.windowing_utils import (
        filter_coarse_weights_by_data_indices,
        filter_reserves_by_given_timestamp,
    )
    from quantammsim.utils.data_processing.datetime_utils import (
        datetime_to_unixtimestamp,
    )
    from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
        squareplus,
        inverse_squareplus,
        inverse_squareplus_np,
    )

    # Use the scraped pool data configuration
    # You can choose between:
    # - "scraped_pool_data": Uses calculated weights from balances and prices
    # - "scraped_pool_data_actual_reserves": Uses actual reserves from the data
    name = "scraped_pool_data"
    config = EXAMPLE_CONFIGS[name]

    print(f"\nRunning {name}...")
    start_date = config["fingerprint"]["startDateString"]
    end_date = "2025-09-05 00:00:00"
    list_of_date_variations = [{"startDateString": start_date, "endDateString": end_date}]
    # Generate list of date variations for each day from start to end date
    from datetime import datetime, timedelta
    
    def generate_daily_variations(start_date_str, end_date_str):
        start = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")
        
        variations = []
        current = start
        while current < end:
            next_day = current + timedelta(days=1)
            variations.append({
                "startDateString": current.strftime("%Y-%m-%d %H:%M:%S"),
                "endDateString": next_day.strftime("%Y-%m-%d %H:%M:%S")
            })
            current = next_day
            
        return variations

    list_of_date_variations = list_of_date_variations + generate_daily_variations(start_date, end_date)
    #    {"startDateString": "2025-08-26 00:00:00", "endDateString": "2025-08-27 00:00:00"},
    #    {"startDateString": "2025-08-27 00:00:00", "endDateString": "2025-08-28 00:00:00"}]
    for date_variation in list_of_date_variations:
        config["fingerprint"]["startDateString"] = date_variation["startDateString"]
        config["fingerprint"]["endDateString"] = date_variation["endDateString"]
        config["params"]["initial_reserves"] = filter_reserves_by_given_timestamp(
            EXAMPLE_CONFIGS["scraped_pool_data_actual_reserves"]["actual_reserves"],
            EXAMPLE_CONFIGS["scraped_pool_data_actual_reserves"]["actual_unix_values"],
            datetime_to_unixtimestamp(date_variation["startDateString"], str_format="%Y-%m-%d %H:%M:%S")*1000,
        )
        config["params"]["initial_weights"] = config["coarse_weights"]["weights"][0]
        result = do_run_on_historic_data_with_provided_coarse_weights(
            run_fingerprint=config["fingerprint"],
            coarse_weights=config["coarse_weights"],
            params=config["params"],
            fees_df=config["fees_df"],
            gas_cost_df=config["gas_cost_df"],
            lp_supply_df=config["lp_supply_df"],
            arb_fees_df=config["arb_fees_df"],
        )
        print("-" * 80)
        print(f"Pool Type: {config['fingerprint']['rule']}")
        print(f"Tokens: {', '.join(config['fingerprint']['tokens'])}")
        print(f"Fees: {config['fingerprint'].get('fees', 0.0)}")
        if "arb_quality" in config["fingerprint"]:
            print(f"Arb Quality: {config['fingerprint']['arb_quality']}")
        print(f"Initial Pool Value: ${result['value'][0]:.2f}")
        print(f"Final Pool Value: ${result['final_value']:.2f}")
        print(f"Return: {(result['final_value']/result['value'][0]-1)*100:.2f}%")
        print(
            f"Return over hodl: {(result['final_value']/(result['reserves'][0]*result['prices'][-1]).sum()-1)*100:.2f}%"
        )
        print("-" * 80)
        print("final weights")
        print(f"{jnp.array_str(result['weights'][-1], precision=16, suppress_small=False)}")
        print("-" * 80)
        print("final prices")
        print(f"{jnp.array_str(result['prices'][-1], precision=16, suppress_small=False)}")
        print("=" * 80)

        tokens = config["fingerprint"]["tokens"]
        rule = config["fingerprint"]["rule"]
        pool_val = config["fingerprint"]["initial_pool_value"]
        fee = config["fingerprint"]["fees"]
        gas = config["fingerprint"]["gas_cost"]
        noise_trader_ratio = config["fingerprint"]["noise_trader_ratio"]
        bout_offset = config["fingerprint"]["bout_offset"]
        chunk_period = config["fingerprint"]["chunk_period"]
        start_date = config["fingerprint"]["startDateString"]
        end_date = config["fingerprint"]["endDateString"]
        # Plot value over time

        # Convert timestamps to datetime
        local_reserves = filter_reserves_by_data_indices(
            EXAMPLE_CONFIGS["scraped_pool_data_actual_reserves"]["actual_reserves"],
            EXAMPLE_CONFIGS["scraped_pool_data_actual_reserves"][
                "actual_unix_values"
            ],
            result["data_dict"],
        )[:-1]
        actual_value = (local_reserves * result["prices"]).sum(-1)
        
        # Create datetime array for x-axis
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        datetime_array = pd.date_range(start=start_datetime, end=end_datetime, freq='1min')[:-1]
        
        # Determine time range for appropriate tick formatting
        time_diff = end_datetime - start_datetime
        is_monthly = time_diff.days > 7  # More than a week
        
        plt.figure(figsize=(12, 6))
        plt.plot(datetime_array, result["value"])
        if is_monthly:
            plt.title("$\\mathrm{Pool\\ Value\\ Over\\ Time,\\ Simulated}$")
        else:
            # Extract date from start_date for daily plots
            date_str = start_date.split(' ')[0]  # Get just the date part
            plt.title(f"$\\mathrm{{Pool\\ Value\\ Over\\ Time,\\ Simulated\\ ({date_str})}}$")
        plt.xlabel("$\\mathrm{Time}$" if not is_monthly else "$\\mathrm{Date}$")
        plt.ylabel("$\\mathrm{Value\\ (USD)}$")
        plt.grid(True)
        
        # Format x-axis based on time range
        ax = plt.gca()
        if is_monthly:
            # For monthly data, show dates every few days
            ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=max(1, time_diff.days // 8)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
        else:
            # For daily data, show hours
            ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=max(1, int(time_diff.total_seconds() / 3600) // 6)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{name}_fees_{config['fingerprint']['fees']}_gas_{config['fingerprint']['gas_cost']}_start_{start_date}_end_{end_date}_value.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(datetime_array, actual_value - result["value"])
        if is_monthly:
            plt.title("$\\mathrm{Pool\\ Value\\ Over\\ Time,\\ Real\\ -\\ Simulated}$")
        else:
            # Extract date from start_date for daily plots
            date_str = start_date.split(' ')[0]  # Get just the date part
            plt.title(f"$\\mathrm{{Pool\\ Value\\ Over\\ Time,\\ Real\\ -\\ Simulated\\ ({date_str})}}$")
        plt.xlabel("$\\mathrm{Time}$" if not is_monthly else "$\\mathrm{Date}$")
        plt.ylabel("$\\mathrm{Value\\ difference\\ (USD)}$")
        plt.grid(True)
        
        # Format x-axis based on time range
        ax = plt.gca()
        if is_monthly:
            ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=max(1, time_diff.days // 8)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
        else:
            ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=max(1, int(time_diff.total_seconds() / 3600) // 6)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{name}_fees_{config['fingerprint']['fees']}_gas_{config['fingerprint']['gas_cost']}_start_{start_date}_end_{end_date}_value_difference.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(datetime_array, actual_value/ result["value"])
        if is_monthly:
            plt.title("$\\mathrm{Pool\\ Value\\ Over\\ Time,\\ Real\\ /\\ Simulated}$")
        else:
            # Extract date from start_date for daily plots
            date_str = start_date.split(' ')[0]  # Get just the date part
            plt.title(f"$\\mathrm{{Pool\\ Value\\ Over\\ Time,\\ Real\\ /\\ Simulated\\ ({date_str})}}$")
        plt.xlabel("$\\mathrm{Time}$" if not is_monthly else "$\\mathrm{Date}$")
        plt.ylabel("$\\mathrm{Value\\ ratio}$")
        plt.grid(True)
        
        # Format x-axis based on time range
        ax = plt.gca()
        if is_monthly:
            ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=max(1, time_diff.days // 8)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
        else:
            ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=max(1, int(time_diff.total_seconds() / 3600) // 6)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{name}_fees_{config['fingerprint']['fees']}_gas_{config['fingerprint']['gas_cost']}_start_{start_date}_end_{end_date}_value_ratio.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(datetime_array, result["value"])
        plt.plot(datetime_array, actual_value, color="red")
        if is_monthly:
            plt.title(f"$\\mathrm{{Pool\\ Value\\ Over\\ Time\\, \\ Real\\ vs\\ Simulated}}$")
        else:
            # Extract date from start_date for daily plots
            date_str = start_date.split(' ')[0]  # Get just the date part
            plt.title(f"$\\mathrm{{Pool\\ Value\\ Over\\ Time\\, \\ Real\\ vs\\ Simulated\\ ({date_str})}}$")
        plt.legend(["Simulated", "Real"])
        plt.xlabel("$\\mathrm{Time}$" if not is_monthly else "$\\mathrm{Date}$")
        plt.ylabel("$\\mathrm{Value\\ (USD)}$")
        plt.grid(True)
        
        # Format x-axis based on time range
        ax = plt.gca()
        if is_monthly:
            ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=max(1, time_diff.days // 8)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
        else:
            ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=max(1, int(time_diff.total_seconds() / 3600) // 6)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            f"{name}_fees_{config['fingerprint']['fees']}_gas_{config['fingerprint']['gas_cost']}_start_{start_date}_end_{end_date}_value_actual.png", dpi=300
        )
        plt.close()

        # plot_reserves(
        #     result,
        #     config["fingerprint"],
        #     run_name=f"{name}_fees_{config['fingerprint']['fees']}_gas_{config['fingerprint']['gas_cost']}_start_{start_date}_end_{end_date}",
        # )

        # base_plot_prefix = name + "_" + f"{tokens}_{rule}_{pool_val}_{fee}_{gas}_{noise_trader_ratio}_bout_{bout_offset}_chunk_{chunk_period}_start_{start_date}_end_{end_date}"
        # plot_weights(
        #     result,
        #     config["fingerprint"],
        #     plot_prefix=f"train_{base_plot_prefix}",
        #     plot_dir="./results",
        # )
        # # load up reserves from network
        # simulated_unix_values = np.linspace(
        #     config["coarse_weights"]["unix_values"][0],
        #     config["coarse_weights"]["unix_values"][1],
        #     result["reserves"].shape[0],
        # )
        

        # plot_reserves(
        #     result,
        #     config["fingerprint"],
        #     run_name=f"{name}_fees_{config['fingerprint']['fees']}_gas_{config['fingerprint']['gas_cost']}_start_{start_date}_end_{end_date}_extra",
        #     actual_reserves_np=local_reserves,
        #     actual_unix_values=datetime_array,
        # )
        # raise Exception("Stop here")