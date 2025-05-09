import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

dict_labels = {
    0: "No",
    1: "others",
    2: "small",
    3: "large",
    #4: "Small",
   # 5: "Large",

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

def test_xgb_classifier():
    # fix the random seed
    seed = 42

    # Load the data
    datasets = [5, 10, 20, 30, 40, 50, 60, "Variable"]
    path = rf'datasets/Labelled_data_v4.xlsx'
    dataset_path = rf'datasets/buffers/Labelled_StreamsV5_Variable.xls'
    df = pd.read_excel(dataset_path)

    # necessary preprocessing
    df = df.dropna()
    df.drop(columns=["OBJECTID", "Buffer", "StreamID"], inplace=True)
    # merge labels 1,2, and 3 to one class
    df["WatercourseRank"] = df["WatercourseRank"].replace({1: 1, 2: 1, 3: 1, 4:2, 5:3})
#    # Drop features with specific substrings in their names
#     substrings_to_drop = [
#         "COMPAC", "DROUGHT", "EROSION", "EXCESSMOIST",
#         "RUTTING", "SOILTEMP", "VEGCOMP", "WINDTHROW"
#     ]
#     columns_to_drop = [
#         col for col in df.columns 
#         if any(sub.lower() in col.lower() for sub in substrings_to_drop)
#     ]
#     print(len(columns_to_drop))
#     print(columns_to_drop )
#     df.drop(columns=columns_to_drop, inplace=True)

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
    # selector = SelectKBest(mutual_info_classif, k=50)
    # X_train = selector.fit_transform(X_train, y_train)
    # X_test = selector.transform(X_test)
    

    clf = XGBClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    # Predict labels
    predictions = clf.predict(X_test)

    # per class accuracy
    cm_normalized = confusion_matrix(y_test, predictions, normalize="true")
    cm_normalized_percentage = cm_normalized * 100
    for i in range(len(cm_normalized_percentage)):
        print(f"Accuracy class {i}: {cm_normalized_percentage[i,i]:.2f}%")
    balanced_accuracy = cm_normalized_percentage.trace() / len(cm_normalized_percentage)
    print(f"Balanced accuracy: {balanced_accuracy:.2f}%")

    plt.clf()
    plot_confusion_matrix(cm_normalized, dict_labels, "xgb", normal=True, fmt=".2f")

    # Print the predicted probabilities, actual predictions, and actual classes for the first 30 samples
    #print("Predictions for the first 30 samples:")
    # probabilities = clf.predict_proba(X_test[:10])
    # # for i in range(30):
    # #     print(f"Sample {i}: Predicted: {dict_labels[predictions[i]]}, "
    # #           f"Actual: {dict_labels[y_test.iloc[i]]}, "
    # #           f"Probabilities: {np.round(probabilities[i], 3)}")

    # # Plot the predicted probabilities as a heatmap
    # plt.clf()
    # plt.figure(figsize=(8, 8))
    # sns.heatmap(probabilities, fmt=".3f", cmap="YlGnBu", cbar=False, annot=True)
    # plt.title("Predicted Probabilities for the First 10 Samples")
    # plt.xlabel("Classes")
    # plt.ylabel("Samples")
    # plt.xticks(ticks=np.arange(len(dict_labels)) + 0.5, labels=[dict_labels[i] for i in range(len(dict_labels))], rotation=45)
    # plt.tight_layout()
    # plt.savefig("predicted_probabilities_heatmap.png")
    # plt.close()

    # # Identify and print misclassified samples among the first 30
    # misclassified = [
    #     (i, predictions[i], y_test.iloc[i])
    #     for i in range(30) if predictions[i] != y_test.iloc[i]
    # ]
    # print("Misclassified samples:")
    # for i, pred, actual in misclassified:
    #     print(f"Sample {i}: Predicted: {dict_labels[pred]}, Actual: {dict_labels[actual]}")
    
    # Assertions for the test
    # assert (
    #     balanced_accuracy == 58.7749939570729
    # ), "baseline accuracy should be 58.77 (XGB with fixed seed, no preprocessing, no feature selection, no hyperparameter tuning)"


test_xgb_classifier()