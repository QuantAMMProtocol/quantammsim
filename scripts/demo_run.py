import jax.numpy as jnp
from quantammsim.core_simulator.param_utils import (
    memory_days_to_logit_lamb,
)
from quantammsim.runners.jax_runners import do_run_on_historic_data
import debug
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

warnings.filterwarnings("ignore")
from jax import config
config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

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
        # Add gradient shading to training period
        # gradient = np.linspace(0, 0.1, 100)
        # for i in range(len(gradient)):
        #     plt.axvspan(pd.Timestamp("2023-12-21") + pd.Timedelta(days=i/10),
        #                pd.Timestamp("2023-12-21") + pd.Timedelta(days=(i+1)/10),
        #                color='red', alpha=gradient[i])

    # unique_dates = df["Date"].unique()
    # date_indices = np.linspace(0, len(unique_dates)-1, 4, dtype=int)
    # selected_dates = unique_dates[date_indices]
    # date_labels = [f"$$\\mathrm{{{pd.Timestamp(date).strftime('%Y-%m-%d')}}}$$" for date in selected_dates]
    # ax.set_xticks(date_indices, date_labels, rotation=45)
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

EXAMPLE_CONFIGS = {
    "arb_macro_release_candidate": {
        "fingerprint": {
            **DEFAULT_FINGERPRINT,
            "tokens": ["ARB", "BTC", "ETH", "USDC"],
            "rule": "mean_reversion_channel",
            # "startDateString": "2021-03-01 00:00:00",
            # "endDateString": "2025-01-01 00:00:00",
            # "endDateString": "2024-08-01 00:00:00",
            "startDateString": "2023-06-01 00:00:00",
            "endDateString": "2025-10-31 00:00:00",
            "endTestDateString": None,
            "chunk_period": 1440,
            "weight_interpolation_period": 1440,
            "minimum_weight": 0.03,
        },
        "params": {
            "initial_weights_logits": jnp.array([
                0.0,
                0.0,
                0.0,
                0.0
                ]),
            "log_amplitude": jnp.array([
                -25.33861947,
                3.77927732,
                -21.91066149,
                -21.94475514
                ]),
            "log_k": jnp.array([
                -11.21452271,
                21.70660311,
                2.57676706,
                -10.46688991
                ]),
            "logit_delta_lamb": jnp.array([
                0.0,
                0.0,
                0.0,
                0.0
                ]),
            "logit_lamb": jnp.array([
                -15.2434908,
                2.43843401,
                -14.01146173,
                -7.0958629
                ]),
            "raw_exponents": jnp.array([
            -4.20670701,
            10.57392943,
            -10.08946104,
            -14.75863772
            ]),
            "raw_pre_exp_scaling": jnp.array([
            6.1501992,
            15.66947856,
            1.25078714,
            -5.32111837
            ]),
            "raw_width": jnp.array([
            -22.23761534,
            3.65913477,
            -24.92888177,
            7.33192479
            ]),
        },
    },
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
    from quantammsim.pools.G3M.quantamm.update_rule_estimators.estimator_primitives import (
        squareplus,
        inverse_squareplus,
        inverse_squareplus_np,
    )

    for name, config in EXAMPLE_CONFIGS.items():
        print(name)
        if name not in ["arb_macro_test_candidate"]:
            continue
        print(f"\nRunning {name}...")
        result = do_run_on_historic_data(
            run_fingerprint=config["fingerprint"],
            params=config["params"],
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
        # memory_days = lamb_to_memory_days(jax_logit_lamb_to_lamb(config["params"]["logit_lamb"]), config["fingerprint"]["chunk_period"])
        # print("memory days: ", memory_days)
        if "logit_lamb" in config["params"]:
            memory_days = lamb_to_memory_days_clipped(
                calc_lamb(config["params"]),
                chunk_period=config["fingerprint"]["chunk_period"],
                max_memory_days=365,
            )
            print(f"{'memory days':<20} {str(memory_days)}")
            lamb = calc_lamb(config["params"])
            print(
                f"{'lamb':<20} {jnp.array_str(lamb, precision=16, suppress_small=False)}"
            )
            if "log_k" in config["params"]:
                k = 2 ** config["params"]["log_k"] * memory_days
                k_str = " ".join(f"{x:.16e}" for x in k)
                print(f"{'k':<20} [{k_str}]")
                k_per_day_str = " ".join(
                    f"{x:.16e}" for x in 2 ** config["params"]["log_k"]
                )
                print(f"{'k per day':<20} [{k_per_day_str}]")
        if "raw_exponents" in config["params"]:
            exponents = squareplus(config["params"]["raw_exponents"])
            exp_str = " ".join(f"{x:.16f}" for x in exponents)
            print(f"{'exponents':<20} [{exp_str}]")
        if "raw_width" in config["params"]:
            width = 2 ** config["params"]["raw_width"]
            width_str = " ".join(f"{x:.16e}" for x in width)
            print(f"{'width':<20} [{width_str}]")
        if "log_amplitude" in config["params"]:
            memory_days = lamb_to_memory_days_clipped(
                calc_lamb(config["params"]),
                chunk_period=config["fingerprint"]["chunk_period"],
                max_memory_days=365,
            )
            amplitude = (2 ** config["params"]["log_amplitude"]) * memory_days
            amp_str = " ".join(f"{x:.16e}" for x in amplitude)
            print(f"{'amplitude':<20} [{amp_str}]")
        if "logit_pre_exp_scaling" in config["params"]:
            pre_exp_scaling = jnp.exp(config["params"]["logit_pre_exp_scaling"]) / (
                1 + jnp.exp(config["params"]["logit_pre_exp_scaling"])
            )
            pes_str = " ".join(f"{x:.16f}" for x in pre_exp_scaling)
            print(f"{'pre_exp_scaling':<20} [{pes_str}]")
        if "raw_pre_exp_scaling" in config["params"]:
            pre_exp_scaling = 2 ** config["params"]["raw_pre_exp_scaling"]
            pes_str = " ".join(f"{x:.16f}" for x in pre_exp_scaling)
            print(f"{'pre_exp_scaling':<20} [{pes_str}]")

        print("-" * 80)
        print("final readouts")
        for readout in result["readouts"]:
            print(f"{readout}: { jnp.array_str(result['readouts'][readout][-1], precision=16, suppress_small=False)}")
        print("-" * 80)
        print("final weights")
        print(f"{jnp.array_str(result['weights'][-1], precision=16, suppress_small=False)}")
        print("-" * 80)
        print("final prices")
        print(f"{jnp.array_str(result['prices'][-1], precision=16, suppress_small=False)}")
        print("=" * 80)
        # Plot value over time

        # Convert timestamps to datetim

        plt.figure(figsize=(12, 6))
        plt.plot(result["value"][::1440])
        plt.title(f"Pool Value Over Time - {name}")
        plt.xlabel("Date")
        plt.ylabel("Value (USD)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{name}_value.png")
        plt.close()

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
        base_plot_prefix = name + "_" + f"{tokens}_{rule}_{pool_val}_{fee}_{gas}_{noise_trader_ratio}_bout_{bout_offset}_chunk_{chunk_period}_start_{start_date}_end_{end_date}"
        plot_weights(
            result,
            config["fingerprint"],
            plot_prefix=f"train_{base_plot_prefix}",
            plot_dir="./results",
        )
        # do_weight_change_as_rebalances_plots(
        #     train_dict,
        #     run_fingerprint,
        #     plot_prefix=f"train_{tokens}_{rule}_{pool_val}_{fee}_{gas}_end_{end_date}_param_{i}",
        # )
        result["fingerprint"] = config["fingerprint"]
        plot_values(
            {"Optimized_QuantAMM_pool_"+'-'.join(tokens)+"_rule_"+rule: result},
            tokens,
            suffix=f"train_{base_plot_prefix}",
            plot_dir="./results",
        )
