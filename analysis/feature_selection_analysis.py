import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import torch
import seaborn as sns


def reproduce_exp(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    tf.random.set_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(
    cm, dict_labels, experiment_path="BaseModelBasePreprocessor", normal=False, fmt="d"
) -> None:
    plt.figure(figsize=(10, 7))
    plt.title("Confusion matrix")
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues")
    classes_total = len(dict_labels.keys())
    plt.xticks(
        ticks=[i + 0.5 for i in range(classes_total)],
        labels=[dict_labels[i] for i in range(classes_total)],
    )
    plt.yticks(
        ticks=[i + 0.5 for i in range(classes_total)],
        labels=[dict_labels[i] for i in range(classes_total)],
        rotation=0,
    )
    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().xaxis.set_label_position("top")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(experiment_path, exist_ok=True)
    plt.savefig(f"{experiment_path}/confusion_matrix_{normal}.png")


# Set random seed for reproducibility
RANDOM_SEED = 42
reproduce_exp(RANDOM_SEED)
selection_type = "mutual_info_classif"

dict_labels = {
    0: "No",
    1: "Ephemeral",
    2: "Intermittent",
    3: "Transitional",
    4: "Small",
    5: "Large",
}

perf_array = []
f1_score_array = []
best_n_features = 0
best_perf = 0
best_cmn = []
best_features = []

# Load the data
version = "Variable"
dataset_path = rf'datasets/buffers/Labelled_StreamsV5_{version}.xls'
df = pd.read_excel(dataset_path)

# necessary preprocessing
df = df.dropna()
df.drop(columns=["OBJECTID", "Buffer", "StreamID"], inplace=True)


X = df.drop(columns=["WatercourseRank"])
y = df["WatercourseRank"]

# Split the data into train, eval, and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
X_train, X_eval, y_train, y_eval = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
)


for i in range(1, len(X.columns) + 1):
    # Feature selection
    selector = SelectKBest(mutual_info_classif, k=i)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_eval_selected = selector.transform(X_eval)
    X_test_selected = selector.transform(X_test)

    # Print selected features
    print("Selected features: ", X.columns[selector.get_support()])
    # Print feature importances
    feature_importances = selector.scores_
    feature_importances_sorted_idx = np.argsort(feature_importances)[::-1]
    print("Feature importances (sorted):")
    for idx in feature_importances_sorted_idx:
        print(f"{X.columns[idx]}: {feature_importances[idx]:.4f}")

    # Train model
    model = XGBClassifier(seed=RANDOM_SEED)
    model.fit(X_train_selected, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_selected)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred) * 100
    f1_score_macro = f1_score(y_test, y_pred, average="macro")

    print(f"Balanced accuracy: {balanced_accuracy:.2f}%")
    print(f"F1 score: {f1_score_macro:.2f}")

    perf_array.append(balanced_accuracy)
    f1_score_array.append(f1_score_macro)

    if best_perf < balanced_accuracy:
        best_perf = balanced_accuracy
        best_n_features = i
        best_features = X.columns[selector.get_support()]
        best_cmn = cm

# Save performance arrays
os.makedirs(f"plots/feature_selection_v5/{version}", exist_ok=True)
np.save(f"plots/feature_selection/feature_selection_v5/{version}_acc.npy", perf_array)
np.save(
    f"plots/feature_selection/feature_selection_v5/{version}_f1.npy", f1_score_array
)

# Save best results
with open(f"plots/feature_selection/best_results_v5/{version}.txt", "w") as best_results_file:
    best_results_file.write(f"Best number of features: {best_n_features}\n")
    best_results_file.write(f"Most important features: {best_features}\n")
    best_results_file.write(f"Best performance: {best_perf}\n")

# Print best results
print("Best number of features: ", best_n_features)
print("Most important features: ", best_features)
print("Best performance: ", best_perf)

# Plot performance
plt.clf()
plt.plot(perf_array)
plt.xlabel("Number of features")
plt.ylabel("Balanced accuracy")
plt.title("Balanced accuracy vs Number of features")
plt.savefig(f"plots/feature_selection/{selection_type}_feature_selection_acc_v5_{version}.png")

plt.clf()
plt.plot(f1_score_array)
plt.xlabel("Number of features")
plt.ylabel("F1 score")
plt.title("F1 score vs Number of features")
plt.savefig(
    f"plots/feature_selection/{selection_type}_feature_selection_f1_macro_v5_{version}.png"
)


best_cmn = best_cmn / best_cmn.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(
    best_cmn,
    dict_labels,
    experiment_path="plots/feature_selection",
    normal=True,
    fmt=".2f",
)
