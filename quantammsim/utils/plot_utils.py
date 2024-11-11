import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from datetime import datetime, timezone
import warnings

warnings.filterwarnings("ignore")
sns.set(rc={"text.usetex": True})


def calc_returns_from_values(values_in):
    r"""Calculate returns
    ----------
    arr_in : np.ndarray, float64
        A one-dimenisional numpy array of price data

    Returns
    -------
    np.ndarray
        The returns

    """
    n = values_in.shape[0]
    returns = np.empty((n,), dtype=np.float64)
    returns[0] = 0.0
    for i in range(1, n):
        if values_in[i - 1] == 0.0:
            returns[i] = -1.0
        else:
            returns[i] = values_in[i] / values_in[i - 1] - 1.0
    return returns


def calc_overall_returns_from_values(values_in):
    r"""Calculate returns
    ----------
    arr_in : np.ndarray, float64
        A one-dimenisional numpy array of price data

    Returns
    -------
    float
        The returns

    """
    returns = values_in[-1] / values_in[0] - 1.0
    return returns


def format_date_latex(date_string, abbreviate=False):
    date_obj = datetime.strptime(date_string, "%Y-%m-%d")
    if abbreviate:
        month = date_obj.strftime("%b")
    else:
        month = date_obj.strftime("%B")
    year = date_obj.strftime("%Y")
    return f"\\mathrm{{{month}}}\\,{year}"


def format_date_latex_plus_month(date_string):
    date_obj = datetime.strptime(date_string, "%Y-%m-%d")
    date_obj = date_obj.date().replace(month=date_obj.month + 1)
    month = date_obj.strftime("%B")
    year = date_obj.strftime("%Y")
    return f"\\mathrm{{{month}}}\\,{year}"


def add_date_to_runname(run_name, startDateString, endDateString):
    startDateString = format_date_latex(startDateString[:10])
    endDateString = format_date_latex_plus_month(endDateString[:10])
    return run_name + "\\," + startDateString + "\\,\\mathrm{to}\\," + endDateString


def format_date(date_string):
    date_obj = datetime.strptime(date_string, "%Y-%m-%d")
    return date_obj.strftime("%B %Y")


def calc_pool_returns(
    results_dict, hodl_values, swap_fee, withdrawal_fee, verbose=True, hodl=False
):
    pool_values = results_dict["value"]
    pool_return = (pool_values[-1].sum() / pool_values[0].sum()) - 1.0
    pool_return_against_hodl = (pool_values[-1].sum() / hodl_values[-1].sum()) - 1.0

    # profit from arb trade is sum of profits from weight changes and price changes
    if results_dict.get("profit_weights") is not None:
        arb_profit = np.sum(results_dict["profit_weights"] + results_dict["profit_prices"])
    else:
        arb_profit = 0
    pool_profit_from_arb_swaps = arb_profit * swap_fee

    withdrawal_profit = (pool_values[-1].sum()) * withdrawal_fee

    withdrawal_profit_over_HODL = (
        pool_values[-1].sum() - hodl_values[-1].sum()
    ) * withdrawal_fee

    difference = pool_values.sum(-1) - hodl_values.sum(-1)
    if verbose:
        print("pool_return: ", pool_return)
        print("pool_return against HODL: ", pool_return_against_hodl)
        print("profit from arb trade (m): ", pool_profit_from_arb_swaps / 10**6)
        print("withdrawal profit (m): ", withdrawal_profit / 10**6)
        print("UPLIFT withdrawal profit (m): ", withdrawal_profit_over_HODL / 10**6)
        print(
            "max return over hodl at",
            np.where(difference == np.max(difference))[0][0] / len(difference),
        )
    if hodl:
        return pool_return_against_hodl
    else:
        return pool_return


def plot_pool_returns(
    results_dict,
    hodl_values,
    balancer_values,
    run_string,
    start_date_string,
    end_date_string,
):
    plot_data = [
        results_dict["value"].sum(-1),
        balancer_values.sum(-1),
        hodl_values.sum(-1),
    ]

    names = ["QuantAMM", "Balancer", "HODL"]
    plot_data = np.array(plot_data).T
    df = pd.DataFrame(
        plot_data / plot_data[0],
        columns=["$$\mathrm{" + name + "}$$" for name in names],
    )
    df = df.set_index([pd.Index(np.arange(len(df)) / 30)])
    fig, ax = plt.subplots()
    sns.lineplot(data=df, ax=ax).set_title(
        "$$\mathrm{QuantAMM} \,|\, \mathrm{Balancer}  \,|\, \mathrm{HODL}\,\,\mathrm{value} \,\,\mathrm{over}\,\, \mathrm{Market}\,\,\mathrm{Supercycle}$$"
    )
    plt.legend(prop={"size": 10})
    plt.xlabel("$$\mathrm{Time}\,\mathrm{(/months)}$$")
    plt.ylabel("$$\mathrm{Value}$$")
    ax.set_facecolor("#0F0614")

    raw_ticks = [t for t in ax.get_xticklabels(which="major")]

    start_date_latex = (
        "$$" + format_date_latex(start_date_string[0:10], abbreviate=True) + "$$"
    )
    end_date_latex = (
        "$$" + format_date_latex(end_date_string[0:10], abbreviate=True) + "$$"
    )

    raw_ticks[1].set_text(start_date_latex)
    raw_ticks[-2].set_text(end_date_latex)
    plt.xticks(
        [raw_ticks[1].get_position()[0], 12, 24, len(df) / 30],
        [start_date_latex, "$$12$$", "$$24$$", end_date_latex],
    )
    plt.savefig(
        "./plots/" + run_string + "_returns.png",
        dpi=700,
        bbox_inches="tight",
        facecolor="#0F0614",
    )
    plt.close()


def plot_pool_returns_trad(
    results_dict,
    hodl_values,
    trad_values,
    run_string,
    start_date_string,
    end_date_string,
):
    plot_data = [
        results_dict["value"],
        trad_values,
        # hodl_values,
    ]

    # names = ["QuantAMM", "CEX\,rebalancing", "HODL"]
    names = ["QuantAMM\,Fund", "Fund\,on\,CEX"]

    new_colors_order = sns.color_palette()
    new_colors_order[0], new_colors_order[1] = new_colors_order[1], new_colors_order[0]


    plot_data = np.array(plot_data).T
    plot_data = plot_data
    df = pd.DataFrame(
        100.0*(plot_data / plot_data[0] - 1.0),
        columns=["$$\mathrm{" + name + "}$$" for name in names],
    )
    df = df.set_index([pd.Index(np.arange(len(df))*7 / (30))])
    fig, ax = plt.subplots()
    sns.lineplot(data=df, ax=ax, palette=new_colors_order)
    # .set_title(
        # "$$\mathrm{QuantAMM} \,|\, \mathrm{CEX}  \,\mathrm{value} \,\,\mathrm{over}\,\, \mathrm{Market}\,\,\mathrm{Supercycle}$$"
    # )
    plt.legend(prop={"size": 10})
    plt.xlabel("$$\mathrm{Time}\,\mathrm{(/months)}$$")
    plt.ylabel("$$\mathrm{Return}$$")
    ax.set_facecolor("#0F0614")

    raw_ticks = [t for t in ax.get_xticklabels(which="major")]

    start_date_latex = (
        "$$" + format_date_latex(start_date_string[0:10], abbreviate=True) + "$$"
    )
    end_date_latex = (
        "$$" + format_date_latex(end_date_string[0:10], abbreviate=True) + "$$"
    )

    raw_ticks[1].set_text(start_date_latex)
    raw_ticks[-2].set_text(end_date_latex)
    plt.xticks(
        [raw_ticks[1].get_position()[0], 12.0, 7*len(df) / (30)],
        [start_date_latex, "$$12$$", end_date_latex],
    )
    y_value=['$$+'+'{:,.1f}'.format(x) + '\%$$' for x in ax.get_yticks()]
    y_value[1]='$$0\\%$$'
    ax.set_yticklabels(y_value)
    plt.savefig(
        "./plots/" + run_string + "_returns.png",
        dpi=700,
        bbox_inches="tight",
        facecolor="#0F0614",
    )
    plt.close()


def plot_hist_of_return_differences(
    plot_array, run_string, run_name, bandw=False, batch_location=None
):
    if bandw:
        color_ = "grey"
    else:
        color_ = None

    plt.style.use("default")
    sns.set(rc={"text.usetex": True})
    fig, ax = plt.subplots()

    if run_name is not None:
        sns.histplot(plot_array, color=color_).set(
            title="$\\mathrm{Histogram}\\,\\,\\mathrm{of}\\,\\,\\mathrm{RVR}\\,\\,(\\mathrm{diffences}\\,\\,\\mathrm{in}\\,\\,\\mathrm{returns}),\\mathrm{for}\\,\\, "
            + run_name
            + "$"
        )
    else:
        sns.histplot(plot_array, color=color_)
    ax.set(
        xlabel="$\\mathrm{Total}\\,\\mathrm{\\%}\\,\\mathrm{returns}\\,\\mathrm{difference}$",
        ylabel="$\\mathrm{Count}$",
    )
    plt.savefig(
        "./plots/" + run_string + "_returns_diff_hist.png", dpi=700, bbox_inches="tight"
    )
    plt.close()


def plot_vals(
    plot_values,
    cols,
    run_string,
    run_name,
    window_size_estimate_xaxis,
    window_size_estimate_yaxis,
    log_xaxis,
    log_yaxis,
    crop_heatmap,
    capout=True,
    exp_xaxis=False,
    bandw=False,
    plot_type="alpha",
    litepaper_plot=False
):
    if plot_type == "alpha":
        title_prefix = "$\\alpha"
    elif plot_type == "returns":
        title_prefix = "$\\mathrm{TFMM}\\,\mathrm{returns}"
    elif plot_type == "dp_difference":
        title_prefix = "$\\mathrm{Difference}\\,\\,\\mathrm{in}\\,\\,\\mathrm{returns}"
    elif plot_type == "trad_difference":
        title_prefix = "$\\mathrm{RVR}\\,\\,(\\mathrm{difference}\\,\\,\\mathrm{in}\\,\\,\\mathrm{returns})"
    elif plot_type == "volume":
        title_prefix = "$\\mathrm{Average}\\,\\,\\mathrm{monthly}\\,\\,\\mathrm{volume}\\,\\,(\mathrm{USD})"
    else:
        raise NotImplementedError
    df = pd.DataFrame(plot_values, columns=cols)
    if bandw:
        cmap_ = "binary_r"
        center_ = None
    elif plot_type == "volume":
        cmap_ = None
        center_ = 0
    else:
        # cmap_ = None
        # from  matplotlib.colors import LinearSegmentedColormap
        # c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        # v = [0,.15,.4,.5,0.6,.9,1.]
        # l = list(zip(v,c))
        # cmap_=LinearSegmentedColormap.from_list('rg',l, N=256)
        center_ = 0
        cmap_ = "RdYlGn"

    def get_dfwise(df):
        df_wide = df.pivot(*cols)
        #  clean up window length values
        if window_size_estimate_yaxis:
            window_size_estimate = [str(item) for item in df_wide.index]
            for i in range(len(window_size_estimate)):
                if window_size_estimate[i][-2:] == ".0":
                    window_size_estimate[i] = window_size_estimate[i][:-2]
                df_wide.index = pd.Index(window_size_estimate, name=df_wide.index.name)
            df_wide.index = pd.Index(
                ["$" + str(item) + "$" for item in df_wide.index],
                name=df_wide.index.name,
            )
        rounding_axes_fn = lambda x: int(x) if x%1==0 else x
        if window_size_estimate_xaxis:
            # cut last column of datafram
            if crop_heatmap:
                df_wide = df_wide.iloc[:, :-1]
            window_size_estimate = [str(item) for item in df_wide.columns]
            for i in range(len(window_size_estimate)):
                if window_size_estimate[i][-2:] == ".0":
                    window_size_estimate[i] = window_size_estimate[i][:-2]
                df_wide.columns = pd.Index(
                    window_size_estimate, name=df_wide.columns.name
                )
            df_wide.columns = pd.Index(
                ["$" + str(item) + "$" for item in df_wide.columns],
                name=df_wide.columns.name,
            )

        if log_xaxis:
            df_wide.columns = pd.Index(
                [
                    "$" + str(np.around(np.log2(float(item)), 2)).ljust(3, "0") + "$"
                    for item in df_wide.columns
                ],
                name="$\\log " + df_wide.columns.name[1:],
            )
        elif exp_xaxis:
            df_wide.columns = pd.Index(
                ["$2^{" + str(int(float(item))) + "}$" for item in df_wide.columns],
                name=df_wide.columns.name,
            )
        elif log_xaxis == False and window_size_estimate_xaxis == False:
            x_axis_labels = [item for item in df_wide.columns]
            rounding_degree = int(np.ceil(-np.log10(x_axis_labels[1]-x_axis_labels[0])))+1
            # if window_size_estimate > 2.7:
            #     window_size_estimate = np.around(window_size_estimate, 1)
            # if window_size_estimate > 10:
            #     window_size_estimate = np.around(window_size_estimate, 0)
            rounded_axis_cols = [np.round(float(item), rounding_degree) for item in df_wide.columns]
            rounded_axis_cols = np.array(rounded_axis_cols)
            x_axis_labels = np.array(x_axis_labels)
            rounded_axis_cols[x_axis_labels < 1] = x_axis_labels[x_axis_labels < 1]
            rounded_axis_cols = [rounding_axes_fn(x) for x in rounded_axis_cols]
            df_wide.columns = pd.Index(
                ["$" + str(item) + "$" for item in rounded_axis_cols],
                name=df_wide.columns.name,
            )
        if log_yaxis:
            df_wide.index = pd.Index(
                [
                    "$" + str(np.around(np.log2(float(item)), 2)).ljust(3, "0") + "$"
                    for item in df_wide.index
                ],
                name="$\\log " + df_wide.index.name[1:],
            )
        elif log_yaxis == False and window_size_estimate_yaxis == False:
            y_axis_labels = [item for item in df_wide.index]
            rounding_degree = int(np.ceil(-np.log10(y_axis_labels[1]-y_axis_labels[0])))
            # if window_size_estimate > 2.7:
            #     window_size_estimate = np.around(window_size_estimate, 1)
            # if window_size_estimate > 10:
            #     window_size_estimate = np.around(window_size_estimate, 0)
            rounded_axis_rows = [np.round(float(item), rounding_degree) for item in df_wide.index]
            rounded_axis_rows = np.array(rounded_axis_rows)
            y_axis_labels = np.array(y_axis_labels)
            rounded_axis_rows[y_axis_labels < 1] = y_axis_labels[y_axis_labels < 1]
            rounded_axis_rows = [rounding_axes_fn(x) for x in rounded_axis_rows]
            df_wide.index = pd.Index(
                ["$" + str(item) + "$" for item in rounded_axis_rows],
                name=df_wide.index.name,
            )
            # df_wide.index = pd.Index(
            #     ["$" + str(int(float(item))) + "$" for item in df_wide.index],
            #     name=df_wide.index.name,
            # )
        if crop_heatmap:
            # cut last row of datafram
            df_wide = df_wide.iloc[:-1, :]
            # cut first row of datafram
            df_wide = df_wide.iloc[1:, :]
        return df_wide
    df_wide = get_dfwise(df)

    plt.style.use("default")
    sns.set(rc={"text.usetex": True})
    fig, ax = plt.subplots()

    sns.heatmap(df_wide, center=center_, cmap=cmap_).set(
        title=title_prefix + "\\,\\," + run_name + "$"
    )
    plt.locator_params(axis="both", nbins=11)
    raw_ticks = [t for t in ax.collections[0].colorbar.get_ticks()]
    # print(raw_ticks)
    if min(raw_ticks) >= 0:
        capout = False
    if max(raw_ticks) < 0:
        capout = False
    if max(raw_ticks) > 0 and abs(max(raw_ticks)) > abs(min(raw_ticks)):
        capout = False

    if capout == False:
        ticks = [
            (lambda x: "+" if x > 0 else "")(t) + str(np.around(t, 2))
            for t in ax.collections[0].colorbar.get_ticks()
        ]
        for i in range(len(ticks)):
            if ticks[i][-2:] == ".0":
                ticks[i] = ticks[i][:-2]
            if plot_type != "volume":
                ticks[i] += "\\%"
            else:
                ticks[i] += "\mathrm{M}"
                ticks[i] = ticks[i][1:]
        ax.collections[0].colorbar.set_ticklabels(["$" + str(t) + "$" for t in ticks])
        if exp_xaxis:
            for item in ax.get_xticklabels():
                item.set_rotation(0)
        for item in ax.get_yticklabels():
            item.set_rotation(0)
        if litepaper_plot==True:
            ax.set(yticklabels=[])
            ax.set(xticklabels=[])
            ax.set(xlabel="$\\mathrm{Aggressiveness}\\,\\,\\mathrm{parameter}$", ylabel="$\\mathrm{Time}\\,\\,\\mathrm{parameter}$")
        plt.savefig(
            "./plots/" + plot_type + "_" + run_string + "_light_rg.png",
            dpi=700,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.close()
        # replace very negative values
        df[cols[-1]] = df[cols[-1]].clip(lower=-max(raw_ticks))
        df_wide = get_dfwise(df)
        fig, ax = plt.subplots()
        sns.heatmap(df_wide, center=center_, cmap=cmap_).set(
            title=title_prefix + "\\,\\, " + run_name + "$"
        )
        plt.locator_params(axis="both", nbins=11)
        ticks = [
            (lambda x: "+" if x > 0 else "")(t) + str(np.around(t, 2))
            for t in ax.collections[0].colorbar.get_ticks()
        ]
        for i in range(len(ticks)):
            if ticks[i][-2:] == ".0":
                ticks[i] = ticks[i][:-2]
            if plot_type != "volume":
                ticks[i] += "\\%"
            else:
                ticks[i] += "\mathrm{M}\\,(\mathrm{USD})"
        ticks[0] += "\\,\\mathrm{or}\\,\\mathrm{lower}"
        # print("ticks: ", ticks)
        ax.collections[0].colorbar.set_ticklabels(["$" + str(t) + "$" for t in ticks])
        if exp_xaxis:
            for item in ax.get_xticklabels():
                item.set_rotation(0)
        for item in ax.get_yticklabels():
            item.set_rotation(0)
        if litepaper_plot==True:
            ax.set(yticklabels=[])
            ax.set(xticklabels=[])
            ax.set(xlabel="$\\mathrm{Aggressiveness}\\,\\,\\mathrm{parameter}$", ylabel="$\\mathrm{Time}\\,\\,\\mathrm{parameter}$")
        plt.savefig(
            "./plots/" + plot_type + "_" + run_string + "_light_rg.png",
            dpi=700,
            bbox_inches="tight",
        )
        plt.close()

    # plt.style.use("dark_background")
    sns.set(rc={"text.usetex": True})
    # sns.set(rc={'axes.facecolor': '##232629'})
    # sns.set(rc={'figure.facecolor': '#424242'})

    COLOR = "white"
    sns.set(
        rc={
            "text.color": COLOR,
            "axes.labelcolor": COLOR,
            "xtick.color": COLOR,
            "ytick.color": COLOR,
            "figure.facecolor": "#232629",
        }
    )
    # sns.set(rc={'figure.facecolor': '#424242'})
    fig, ax = plt.subplots()
    sns.heatmap(df_wide, center=center_, cmap=cmap_).set(
        title=title_prefix + "\\,\\," + run_name + "$"
    )
    plt.locator_params(axis="both", nbins=11)
    ax.collections[0].colorbar.set_ticklabels(["$" + str(t) + "$" for t in ticks])
    if exp_xaxis:
        for item in ax.get_xticklabels():
            item.set_rotation(0)
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    if litepaper_plot==True:
        ax.set(yticklabels=[])
        ax.set(xticklabels=[])
        ax.set(xlabel="$\\mathrm{Aggressiveness}\\,\\,\\mathrm{parameter}$", ylabel="$\\mathrm{Time}\\,\\,\\mathrm{parameter}$")
    plt.savefig(
        "./plots/" + plot_type + "_" + run_string + "_dark_rg.png",
        dpi=700,
        bbox_inches="tight",
    )
    plt.close()
    plt.close("all")


def plot_hist_of_returns(
    list_of_results, hodl_values, run_string, run_name, bandw=False, batch_location=None
):
    pool_returns_list = list()
    for i in range(len(list_of_results)):
        # first keep entries from parameter dict that are 'keepcols'
        pool_values = list_of_results[i][2]["value"]
        pool_return = (pool_values[-1].sum() / pool_values[0].sum()) - 1.0
        pool_returns_list.append(pool_return)

    pool_returns = np.array(pool_returns_list)

    if bandw:
        color_ = "grey"
    else:
        color_ = None

    plt.style.use("default")
    sns.set(rc={"text.usetex": True})
    fig, ax = plt.subplots()

    sns.histplot(pool_returns, color=color_).set(
        title="$\\mathrm{Histogram}\\,\\,\\mathrm{of}\\,\\,\\mathrm{returns},\\mathrm{for}\\,\\, "
        + run_name
        + "$"
    )
    ax.set(xlabel="$\\mathrm{Returns}$", ylabel="$\\mathrm{Count}$")
    plt.savefig(
        "./plots/" + run_string + "_returns_hist.png", dpi=700, bbox_inches="tight"
    )
    plt.close()

    pool_returns_list = list()
    for i in range(len(list_of_results)):
        # first keep entries from parameter dict that are 'keepcols'
        pool_values = list_of_results[i][2]["value"]
        pool_return = (pool_values[-1].sum() / hodl_values[-1].sum()) - 1.0
        pool_returns_list.append(pool_return)

    pool_returns = np.array(pool_returns_list)

    max_return = max(pool_returns)

    max_return_idx = np.where(max(pool_returns) == pool_returns)[0][0]
    print(
        "MAX RETURN AGAINST HODL  -- ",
        str(max_return),
        "-----",
        list_of_results[max_return_idx][0],
        list_of_results[max_return_idx][1],
    )

    fig, ax = plt.subplots()

    sns.histplot(
        pool_returns,
        color=color_,
        label="$\\mathrm{Individual}\\,\\mathrm{pool}\\,\\mathrm{returns}$",
    ).set(
        title="$\\mathrm{Histogram}\\,\\,\\mathrm{of}\\,\\,\\mathrm{returns}\\,\\,\\mathrm{against}\\,\\,\\mathrm{HODL},\\mathrm{for}\\,\\, "
        + run_name
        + "$"
    )
    if batch_location is not None:
        plt.axvline(
            x=batch_location,
            ymin=0.00,
            ymax=1.0,
            color="black",
            label="$\\mathrm{Batched}\\,\\mathrm{return}$",
        )
        plt.legend(bbox_to_anchor=(0, 1), loc="upper left")
    # plt.locator_params(axis='both', nbins=11)
    # ax.collections[0].colorbar.set_ticklabels(['$'+str(t)+'$' for t in ticks])
    # if exp_xaxis:
    #     for item in ax.get_xticklabels():
    #         item.set_rotation(0)
    # for item in ax.get_yticklabels():
    #     item.set_rotation(0)
    ax.set(xlabel="$\\mathrm{Returns}$", ylabel="$\\mathrm{Count}$")
    plt.savefig(
        "./plots/" + run_string + "_returns_hist_over_hodl.png",
        dpi=700,
        bbox_inches="tight",
    )
    plt.close()


def calc_jensens_alpha(pool_values, balancer_values, hodl_values, pre_agg=False, annual_risk_free_rate=0.0):
    ## calculates Jensen's alpha of pool returns
    # given hour-level values as input
    # relative to Balancer and HODL with a 0 risk free rate.

    if pre_agg:
        # intially returns are coming in at daily level, we can pre-aggregate to
        # monthly level
        pool_values = pool_values[::30]
        balancer_values = balancer_values[::30]
        hodl_values = hodl_values[::30]

    ## first we need to calculate the returns from the values arrays
    pool_returns = calc_returns_from_values(pool_values.sum(-1)) * 100
    balancer_returns = calc_returns_from_values(balancer_values.sum(-1)) * 100
    hodl_returns = calc_returns_from_values(hodl_values.sum(-1)) * 100

    ## second we need to calculate the overall returns from the values arrays
    pool_overall_returns = calc_overall_returns_from_values(pool_values.sum(-1)) * 100
    balancer_overall_returns = (
        calc_overall_returns_from_values(balancer_values.sum(-1)) * 100
    )
    hodl_overall_returns = calc_overall_returns_from_values(hodl_values.sum(-1)) * 100
    if pre_agg:
        # pre_agg being True means Monthly data
        overall_risk_free_return = (((annual_risk_free_rate + 1.0) ** (len(pool_values)/12.0)) - 1.0) * 100
    else:
        # pre_agg being False means Daily data
        overall_risk_free_return = (((annual_risk_free_rate + 1.0) ** (len(pool_values)/365.25)) - 1.0) * 100

    # now calculate beta, covariance between pool and reference market values
    # divided by reference market variance
    pool_to_bal_cov = np.cov(pool_returns, balancer_returns, rowvar=False)[0, 1]
    pool_to_hodl_cov = np.cov(pool_returns, hodl_returns, rowvar=False)[0, 1]

    bal_var = np.var(balancer_returns)
    hold_var = np.var(hodl_returns)

    pool_bal_beta = pool_to_bal_cov / bal_var
    pool_hodl_beta = pool_to_hodl_cov / hold_var

    # now can calc Jensen's alpha
    pool_bal_alpha = pool_overall_returns - overall_risk_free_return - pool_bal_beta * (balancer_overall_returns - overall_risk_free_return)
    pool_hodl_alpha = pool_overall_returns - overall_risk_free_return - pool_hodl_beta * (hodl_overall_returns - overall_risk_free_return)
    excess_return = pool_overall_returns - hodl_overall_returns

    return pool_bal_alpha, pool_hodl_alpha, excess_return


def calc_values_from_results(
    list_of_results,
    balancer_results_list,
    hodl_values,
    chunk_period,
    keepcols=["k", "lamb"],
    pre_agg=False,
    round_window=True,
    value="alpha",
    annual_risk_free_rate=0.0
):
    results_to_return = list()
    for i in range(len(list_of_results)):
        results_to_return.append([])
        # first keep entries from parameter dict that are 'keepcols'
        if keepcols is not None:
            for col in keepcols:
                results_to_return[-1].append(list_of_results[i][1][col])
        # now extract out the values from the results
        pool_values = list_of_results[i][2]["value"]
        balancer_values = balancer_results_list[0]

        if value == "alpha":
            (
                pool_bal_alpha,
                pool_hodl_alpha,
                excess_returns_over_hodl,
            ) = calc_jensens_alpha(pool_values, balancer_values, hodl_values, pre_agg, annual_risk_free_rate)
            results_to_return[-1].append(pool_bal_alpha)
            results_to_return[-1].append(pool_hodl_alpha)
        elif value == "returns":
            pool_return = (pool_values[-1].sum() / pool_values[0].sum()) - 1.0
            results_to_return[-1].append(pool_return * 100)
            pool_return_over_hodl = (
                pool_values[-1].sum() / hodl_values[-1].sum()
            ) - 1.0
            results_to_return[-1].append(pool_return_over_hodl * 100)
        elif value == "trad_daily":
            pool_return = (pool_values[-1].sum() / pool_values[0].sum()) - 1.0
            trad_values = list_of_results[i][2]["trad_values"][0]
            trad_return = (trad_values.sum() / pool_values[0].sum()) - 1.0
            difference_in_returns = pool_return - trad_return
            results_to_return[-1].append(difference_in_returns * 100)
            results_to_return[-1].append(trad_return * 100)
        elif value == "trad_hourly":
            pool_return = (pool_values[-1].sum() / pool_values[0].sum()) - 1.0
            trad_values = list_of_results[i][2]["trad_values"][1]
            trad_return = (trad_values.sum() / pool_values[0].sum()) - 1.0
            difference_in_returns = pool_return - trad_return
            results_to_return[-1].append(difference_in_returns * 100)
            results_to_return[-1].append(trad_return * 100)
        elif value == "trad_minute":
            pool_return = (pool_values[-1].sum() / pool_values[0].sum()) - 1.0
            trad_values = list_of_results[i][2]["trad_values"][2]
            trad_return = (trad_values.sum() / pool_values[0].sum()) - 1.0
            difference_in_returns = pool_return - trad_return
            results_to_return[-1].append(difference_in_returns * 100)
            results_to_return[-1].append(trad_return * 100)

        # calc estimate window size
        lamb = list_of_results[i][1].get("lamb")
        if lamb is not None:
            window_size_estimate = np.around(
                np.cbrt(6 * lamb / ((1 - lamb) ** 3)) * 2 * chunk_period / 1440, 2
            )
        else:
            window_size_estimate = list_of_results[i][1]["memory_days"]
        if round_window:
            if window_size_estimate > 2.7:
                window_size_estimate = np.around(window_size_estimate, 1)
            if window_size_estimate > 10:
                window_size_estimate = np.around(window_size_estimate, 0)
        results_to_return[-1].append(window_size_estimate)
        if list_of_results[i][1].get("lamb_mixing") is not None:
            lamb = list_of_results[i][1]["lamb_mixing"]
            window_size_estimate = np.around(
                np.cbrt(6 * lamb / ((1 - lamb) ** 3)) * 2 * chunk_period / 1440, 2
            )
            if round_window:
                if window_size_estimate > 2.7:
                    window_size_estimate = np.around(window_size_estimate, 1)
                if window_size_estimate > 10:
                    window_size_estimate = np.around(window_size_estimate, 0)
            results_to_return[-1].append(window_size_estimate)
    return np.array(results_to_return)

def plot_lineplot(x, y, x_name, y_name, title, save_location, label=None, y_percentage=True, y_marks=None, symlog=False):
    fig, ax = plt.subplots()
    if label is not None:
        sns.lineplot(x=x, y=y, ax=ax, label=label).set_title(title)
    else:
        sns.lineplot(x=x, y=y, ax=ax).set_title(title)
    plt.legend(prop={"size": 10})
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if y_marks is not None:
        for y_mark in y_marks:
            plt.axhline(
                y=y_mark["value"],
                color=y_mark["color"],
                label=y_mark["name"],
            )
        plt.legend(bbox_to_anchor=(1, 0.4), loc="upper right")
    else:
        plt.legend([], [], frameon=False)
    if symlog:
        plt.xscale('symlog')
    if y_percentage:
        y_value = ["$$+" + "{:,.1f}".format(x) + "\%$$" for x in ax.get_yticks()]
        y_value = [yv.replace("$$+0\\%$$", "$$0\\%$$") for yv in y_value]
        ax.set_yticklabels(y_value)
    plt.savefig(
        save_location,
        dpi=700,
        bbox_inches="tight")
    plt.close()
