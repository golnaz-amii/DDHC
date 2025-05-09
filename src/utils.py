import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import numpy as np

####### Paths #######
dataset_path = "datasets/Labelled_data_v1.xls"
plots_path = "plots"
#####################

####### INFO #######
IDs = [
    "OBJECTID",
    "LINKNO",
    "DSLINKNO",
    "USLINKNO1",
    "USLINKNO2",
    "DSNODEID",
    "PlanUnit",
    # "WSNO",
]

# probably irrelevant
DOUTs = ["DOUT_END", "DOUT_START", "DOUT_MID"]

# incomplete data
incomplete = [
    # "DEM", "CHM"
    "buffer"
]

not_needed = IDs + DOUTs + incomplete

stat_postfix = ["MIN", "MAX", "RANGE", "SUM", "MEDIAN", "PCT90", "MEAN", "STD"]
#####################


def check_column_type(column: pd.Series) -> str:
    """
    Check the type of a DataFrame column.

    Args:
        column (pd.Series): The column to check.

    Returns:
        str: The type of the column ("Discrete", "Continuous", or "Non-Numeric").
    """
    if column.dtype in ["int64", "float64"]:
        if column.dtype == "int64" and column.nunique() < 50:
            return "Discrete"
        else:
            return "Continuous"
    return "Non-Numeric"


def plot_distribution(df: pd.DataFrame, column: str) -> None:
    """
    Plot the distribution of a column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column to plot.

    Returns:
        None
    """
    plt.clf()
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    if check_column_type(df[column]) == "Discrete":
        sns.histplot(df, x=column, kde=False, discrete=True)
        for value, count in df[column].value_counts().items():
            plt.text(float(str(value)), count, str(count), ha="center")
    else:
        sns.histplot(data=df, x=column, kde=True)

    plt.title(f"Histogram of {column}", fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    os.makedirs(f"{plots_path}/all", exist_ok=True)
    plt.savefig(f"{plots_path}/all/{column}_histogram.png")


def plot_category_distribution(df: pd.DataFrame, column: str, category: str) -> None:
    """
    Plot the distribution of a column in the DataFrame, grouped by a category.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column to plot.
        category (str): The category to group by.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    if check_column_type(df[column]) == "Discrete":
        sns.histplot(data=df, x=column, hue=category, discrete=True, kde=False)
        for value, count in df[column].value_counts().items():
            plt.text(float(str(value)), count, f"{count} ({count/len(df)*100:.2f}%)", ha="center")
    else:
        sns.histplot(data=df, x=column, hue=category, multiple="stack", kde=True)

    plt.title(f"Histogram of {column} by {category}", fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel(category, fontsize=14)

    os.makedirs(f"{plots_path}/group", exist_ok=True)
    plt.savefig(f"{plots_path}/group/{column}_by_{category}.png")


def plot_category_distribution_separately(
    df: pd.DataFrame, column: str, category: str
) -> None:
    """
    Plot the distribution of a column in the DataFrame separately for each category.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column to plot.
        category (str): The category to group by.

    Returns:
        None
    """
    unique_categories = df[category].unique()
    num_categories = len(unique_categories)
    for cat in unique_categories:
        subset = df[df[category] == cat]
        plt.figure(figsize=(10, 6))

        if check_column_type(df[column]) == "Discrete":
            sns.histplot(subset[column], kde=False, discrete=True)
            plt.title(
                f"Distribution of {column} for {category} = {cat}, for n = {len(subset)}",
                fontsize=14,
            )
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            for value, count in subset[column].value_counts().items():
                plt.text(value, count, str(count), ha="center")
        else:
            sns.histplot(subset[column], kde=True)
            plt.title(
                f"Distribution of {column} for {category} = {cat} for n = {len(subset)}",
                fontsize=14,
            )
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Density", fontsize=12)

        plt.tight_layout()
        os.makedirs(f"{plots_path}/group_individual_plot", exist_ok=True)
        output_path = f"{plots_path}/group_individual_plot/{column}_by_{category}_{cat}.png"
        plt.show()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")


def clean_columns(df: pd.DataFrame) -> list:
    """
    Clean the columns of the DataFrame by removing not needed columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        list: A list of useful columns.
    """
    not_useful = []
    useful = []
    for column in df.columns:
        if any(substring in column for substring in not_needed):
            not_useful.append(column)
        else:
            useful.append(column)
    useful.remove("WatercourseRank")
    print(f"Number of useful columns: {len(useful)}")
    print(f"Not useful columns: {not_useful}")
    return useful


def plot_confusion_matrix(
    cm, dict_labels, experiment_path="BaseModelBasePreprocessor", normal=False, fmt="d"
) -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["figure.figsize"] = (10, 8)
    # add padding to the figure
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.title("Confusion matrix")
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues")
    dict_labels[0] = "Dry"
    classes_total = len(dict_labels.keys())
     # fade the numbers that are not on the diagonal
   
    plt.xticks(
        ticks=[i + 0.5 for i in range(classes_total)],
        labels=[dict_labels[i] for i in range(classes_total)],
        rotation=45,  # Rotate x-axis labels to avoid overlap
        #ha="right"    # Align labels to the right for better readability
    )
    plt.yticks(
        ticks=[i + 0.5 for i in range(classes_total)],
        labels=[dict_labels[i] for i in range(classes_total)],
        rotation = 0,  # Rotate y-axis labels to avoid overlap
    )
   

    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().xaxis.set_label_position("top")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(experiment_path, exist_ok=True)
    plt.savefig(f"{experiment_path}/confusion_matrix_{normal}.png")
    plt.close()



def set_seed(seed: int = 42):
    """
    Set seed for reproducibility across common libraries.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # Torch not installed, skip

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # TensorFlow not installed, skip