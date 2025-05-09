import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tabpfn import TabPFNClassifier


dict_labels = {
    0: "No",
    1: "Ephemeral",
    2: "Intermittent",
    3: "Transitional",
    4: "Small",
    5: "Large",
}


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


# https://github.com/PriorLabs/TabPFN
def test_tabpfn():
    # fix the random seed
    seed = 42

    # Load the data
    df = pd.read_excel("datasets/Labelled_data_v4.xlsx")

    # necessary preprocessing
    df = df.dropna()

    df.drop(columns=["OBJECTID", "Buffer", "StreamID"], inplace=True)

    X = df.drop(columns=["WatercourseRank"])
    y = df["WatercourseRank"]

    # split the data into train, eval, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    # add any preprocessing here
    selector = SelectKBest(mutual_info_classif, k=58)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    clf = TabPFNClassifier(
        random_state=seed,
        model_path="models/tabpfn-v2-classifier.ckpt",
        memory_saving_mode=True,
    )
    clf.fit(X_train, y_train)

    # Predict labels
    start = time.time()
    predictions = clf.predict(X_test)
    end = time.time()
    print("inference time sec: ", end - start)

    # per class accuracy
    cm_normalized = confusion_matrix(y_test, predictions, normalize="true")
    print(cm_normalized)
    cm_normalized_percentage = cm_normalized * 100
    for i in range(len(cm_normalized_percentage)):
        print(f"Accuracy class {i}: {cm_normalized_percentage[i,i]:.2f}%")
    balanced_accuracy = cm_normalized_percentage.trace() / len(cm_normalized_percentage)
    print(f"Balanced accuracy: {balanced_accuracy:.2f}%")
    print(cm_normalized)

    # plot the confusion matrix
    plt.clf()
    plot_confusion_matrix(cm_normalized, dict_labels, "tabpfn", normal=True, fmt=".2f")


test_tabpfn()
