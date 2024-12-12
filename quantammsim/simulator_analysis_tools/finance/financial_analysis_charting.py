import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_line_chart_from_series(
    results_list,
    series_list,
    startDateString,
    target_directory,
    filename,
    title,
    xlabel,
    ylabel,
    train_test_boundary_date=None,
):

    series_dict = {}
    for result, series in zip(results_list, series_list):
        series_dict[series] = result

    dataframe = pd.DataFrame(series_dict)

    # Debugging: print the DataFrame

    plot_line_chart(
        dataframe,
        series_list,
        target_directory,
        filename,
        title,
        xlabel,
        ylabel,
        train_test_boundary_date,
    )


def plot_line_chart_from_results(
    results_list,
    series_list,
    startDateString,
    target_directory,
    filename,
    title,
    xlabel,
    ylabel,
    train_test_boundary_date=None,
):
    series_dict = {}
    for result, series in zip(results_list, series_list):
        result_index = pd.date_range(
            start=startDateString, periods=len(result), freq="T"
        )
        result_series = pd.Series(result, index=result_index)
        series_dict[series] = result_series

    dataframe = pd.DataFrame(series_dict)
    # Debugging: print the DataFrame

    plot_line_chart(
        dataframe,
        series_list,
        target_directory,
        filename,
        title,
        xlabel,
        ylabel,
        train_test_boundary_date,
    )


def plot_line_chart(
    dataframe,
    series_list,
    target_directory,
    filename,
    title,
    xlabel,
    ylabel,
    train_test_boundary_date=None,
):
    """
    Plots a line chart with multiple series from an input DataFrame with dates as the index and saves it as a PNG file.

    Parameters:
    - dataframe: pd.DataFrame, the input DataFrame with dates as the index.
    - series_list: list of str, the names of the series/columns to plot.
    - target_directory: str, the directory where the PNG file will be saved.
    - filename: str, the name of the file to save the chart as.
    - title: str, the title of the chart.
    - xlabel: str, the label for the x-axis.
    - ylabel: str, the label for the y-axis.
    """
    # Reset index to convert the index into a column for seaborn
    df_reset = dataframe.reset_index()

    # Rename the index column to 'Date'
    df_reset.rename(columns={"index": "date"}, inplace=True)

    # Melt the DataFrame to long format for seaborn
    df_melted = df_reset.melt(
        id_vars="date", value_vars=series_list, var_name="Series", value_name="Value"
    )

    plt.rcParams["text.usetex"] = False
    # Plot using seaborn
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_melted, x="date", y="Value", hue="Series")

    if train_test_boundary_date is not None:
        plt.axvline(
            pd.to_datetime(train_test_boundary_date),
            color="red",
            linestyle="--",
            linewidth=1,
        )
        ylim = plt.gca().get_ylim()
        plt.text(
            pd.to_datetime(train_test_boundary_date),
            ylim[0] + (ylim[1] - ylim[0]) * 2 / 3,
            "Train/Test Boundary",
            color="red",
            verticalalignment="top",
        )

    # Set title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Ensure target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Save plot to file
    file_path = os.path.join(target_directory, filename)
    plt.savefig(file_path)
    plt.close()
