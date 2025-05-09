import numpy as np
from scipy.stats import t
from scipy.stats import ttest_ind_from_stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html
T-test for means of two independent samples from descriptive statistics.
This is a test for the null hypothesis that two independent samples have identical average (expected) values.
Final test perfomed in this file: **Welch's t-test**
"""

table = {
    "XGBoost": {
        "5m": {"accuracy": 67.09, "ci": 0.74},
        "10m": {"accuracy": 67.40, "ci": 0.69},
        "20m": {"accuracy": 60.73, "ci": 1.47},
        "30m": {"accuracy": 67.77, "ci": 0.86},
        "40m": {"accuracy": 68.04, "ci": 0.79},
        "50m": {"accuracy": 67.86, "ci": 0.77},
        "60m": {"accuracy": 67.71, "ci": 0.73},
        "variable": {"accuracy": 74.83, "ci": 0.77},
    },
    "Hier1": {
        "5m": {"accuracy": 68.57, "ci": 0.53},
        "10m": {"accuracy": 68.55, "ci": 0.62},
        "20m": {"accuracy": 63.35, "ci": 1.59},
        "30m": {"accuracy": 68.82, "ci": 0.66},
        "40m": {"accuracy": 68.92, "ci": 0.84},
        "50m": {"accuracy": 69.96, "ci": 0.72},
        "60m": {"accuracy": 69.73, "ci": 0.78},
        "variable": {"accuracy": 75.88, "ci": 0.65},
    },
    "Hier2": {
        "5m": {"accuracy": 69.83, "ci": 0.62},
        "10m": {"accuracy": 69.94, "ci": 0.78},
        "20m": {"accuracy": 63.45, "ci": 1.14},
        "30m": {"accuracy": 69.92, "ci": 0.69},
        "40m": {"accuracy": 69.94, "ci": 0.69},
        "50m": {"accuracy": 70.56, "ci": 0.66},
        "60m": {"accuracy": 70.39, "ci": 0.88},
        "variable": {"accuracy": 76.96, "ci": 0.63},
    },
    "Hier3": {
        "5m": {"accuracy": 68.61, "ci": 0.53},
        "10m": {"accuracy": 69.06, "ci": 0.76},
        "20m": {"accuracy": 62.63, "ci": 1.12},
        "30m": {"accuracy": 69.60, "ci": 0.71},
        "40m": {"accuracy": 69.72, "ci": 0.73},
        "50m": {"accuracy": 69.75, "ci": 0.58},
        "60m": {"accuracy": 69.77, "ci": 0.80},
        "variable": {"accuracy": 76.99, "ci": 0.53},
    },
    "Hier4": {
        "5m": {"accuracy": 69.24, "ci": 0.57},
        "10m": {"accuracy": 69.63, "ci": 0.67},
        "20m": {"accuracy": 64.13, "ci": 1.20},
        "30m": {"accuracy": 69.80, "ci": 0.64},
        "40m": {"accuracy": 70.05, "ci": 0.61},
        "50m": {"accuracy": 70.46, "ci": 0.78},
        "60m": {"accuracy": 71.08, "ci": 0.81},
        "variable": {"accuracy": 76.63, "ci": 0.58},
    },
    "Hier5": {
        "5m": {"accuracy": 69.00, "ci": 0.55},
        "10m": {"accuracy": 69.28, "ci": 0.72},
        "20m": {"accuracy": 62.94, "ci": 1.10},
        "30m": {"accuracy": 69.04, "ci": 0.53},
        "40m": {"accuracy": 69.48, "ci": 0.88},
        "50m": {"accuracy": 69.54, "ci": 0.75},
        "60m": {"accuracy": 69.81, "ci": 0.97},
        "variable": {"accuracy": 77.07, "ci": 0.59},
    },
    "Hier6": {
        "5m": {"accuracy": 68.36, "ci": 0.77},
        "10m": {"accuracy": 67.63, "ci": 0.83},
        "20m": {"accuracy": 62.61, "ci": 1.12},
        "30m": {"accuracy": 68.64, "ci": 0.77},
        "40m": {"accuracy": 68.12, "ci": 0.72},
        "50m": {"accuracy": 68.72, "ci": 0.57},
        "60m": {"accuracy": 68.26, "ci": 0.58},
        "variable": {"accuracy": 74.70, "ci": 0.70},
    },
}

ALPHA = 0.05


def ci_to_std(ci, n):
    # print("coefficient to change ci to std: ",np.sqrt(n)/t.ppf(0.975, n - 1))
    return (ci * np.sqrt(n)) / t.ppf(0.975, n - 1)


def calculate_p_value(mean1, mean2, ci1, ci2, n):
    std1 = ci_to_std(ci1, n)
    std2 = ci_to_std(ci2, n)
    stats = ttest_ind_from_stats(
        mean1=mean1,
        std1=std1,
        nobs1=n,
        mean2=mean2,
        std2=std2,
        nobs2=n,
        equal_var=False,
    )
    return stats.pvalue


def calculate_p_value_table(table, buffer_type):
    models = list(table.keys())
    n = 10
    p_values = np.zeros((len(models), len(models))) + 1
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                mean1 = table[models[i]][buffer_type]["accuracy"]
                mean2 = table[models[j]][buffer_type]["accuracy"]
                ci1 = table[models[i]][buffer_type]["ci"]
                ci2 = table[models[j]][buffer_type]["ci"]
                p_values[i, j] = calculate_p_value(mean1, mean2, ci1, ci2, n)
    return p_values


def find_statisically_significant_models(p_values):
    significant_pairs = []
    for i in range(len(p_values)):
        for j in range(len(p_values)):
            if (
                i != j
                and p_values[i, j] < ALPHA
                and table[list(table.keys())[i]][buffer_type]["accuracy"]
                > table[list(table.keys())[j]][buffer_type]["accuracy"]
            ):
                significant_pairs.append((list(table.keys())[i], list(table.keys())[j]))
    return significant_pairs


def find_statisically_significant_models_vs_XGBoost(p_values):
    significant_pairs = []
    for i in range(len(p_values)):
        if (
            i != 0
            and p_values[i, 0] < ALPHA
            and table[list(table.keys())[i]][buffer_type]["accuracy"]
            > table[list(table.keys())[0]][buffer_type]["accuracy"]
        ):
            significant_pairs.append((list(table.keys())[i], list(table.keys())[0]))
    return significant_pairs


def find_best_model(table, buffer_type, p_values):
    models = list(table.keys())
    wins = {model: 0 for model in models}

    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i != j and p_values[i, j] < ALPHA:
                if (
                    table[model_i][buffer_type]["accuracy"]
                    > table[model_j][buffer_type]["accuracy"]
                ):
                    wins[model_i] += 1

    # Find the model with the most statistically significant wins
    max_wins = max(wins.values())
    candidates = [model for model, count in wins.items() if count == max_wins]

    # Break ties by highest accuracy
    best_model = max(
        candidates, key=lambda model: table[model][buffer_type]["accuracy"]
    )
    return best_model, wins[best_model], table[best_model][buffer_type]["accuracy"]


def plot_p_values(p_values):
    models = list(table.keys())
    # plot the p-values as a heatmap
    sns.heatmap(p_values, annot=True, fmt=".5f", xticklabels=models, yticklabels=models)
    plt.title("P-values for pairwise comparisons")
    plt.savefig("p_values.png")
    plt.close()


if __name__ == "__main__":
    buffer_types = ["5m", "10m", "20m", "30m", "40m", "50m", "60m", "variable"]

    for buffer_type in buffer_types:

        # Calculate p-values for the current buffer type
        p_values = calculate_p_value_table(table, buffer_type)
        print(f"P-values for buffer type '{buffer_type}':\n", p_values)

        # Plot and save the p-values heatmap
        plot_p_values(p_values)

        # Find and display statistically significant pairs
        significant_pairs = find_statisically_significant_models(p_values)
        if significant_pairs:
            print(f"Statistically significant pairs for buffer type '{buffer_type}':")
            for pair in significant_pairs:
                print(f"  {pair[0]} vs {pair[1]}")
        else:
            print(
                f"No statistically significant pairs found for buffer type '{buffer_type}'."
            )

        # Check for significant models compared to XGBoost
        significant_pairs_vs_XGBoost = find_statisically_significant_models_vs_XGBoost(
            p_values
        )
        if significant_pairs_vs_XGBoost:
            print(
                f"Statistically significant pairs vs XGBoost for buffer type '{buffer_type}':"
            )
            for pair in significant_pairs_vs_XGBoost:
                print(f"  {pair[0]} vs {pair[1]}")
        else:
            print(
                f"No statistically significant models found vs XGBoost for buffer type '{buffer_type}'."
            )

        # Identify best model
        best_model, num_wins, accuracy = find_best_model(table, buffer_type, p_values)
        print(
            f"Best model for buffer type '{buffer_type}': {best_model} "
            f"(statistically better than {num_wins} models, accuracy = {accuracy:.2f}%)"
        )
