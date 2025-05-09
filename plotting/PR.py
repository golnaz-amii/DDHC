import os
import joblib
import json
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
from preprocessors.BasePreprocessor import BasePreprocessor
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

def initialize_plot():
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Paired", len(model_names))

def plot_PR(model, model_name):
    _, _, X_test, _, _, y_test = preprocessor.preprocess_data()
    # draw precision-recall curve (micro average) for multiclass
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    pred_prob = model.predict_proba(X_test)
    print(y_test_binarized.shape, pred_prob.shape)
    precision = {}
    recall = {}
    pr_auc = dict()
    n_class = len(preprocessor.dict_labels)

    # micro average
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test_binarized.ravel(), pred_prob.ravel()
    )

    pr_auc["micro"] = average_precision_score(y_test_binarized, pred_prob, average="micro")
    plt.plot(
        recall["micro"],
        precision["micro"],
        linestyle="--",
        label=f"{model_name} (AP = %0.2f)" % pr_auc["micro"],
    )
    return pr_auc["micro"]

def plot_PR_end():
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")

# Path to the directory containing experiment folders
experiments_dir = "experiments/v3"

dict_labels = {
    0: "No",
    1: "Ephemeral",
    2: "Intermittent",
    3: "Transitional",
    4: "Small",
    5: "Large",
}

preprocessor = BasePreprocessor(
    dataset_path="/Users/Golnaz/Desktop/Golnaz/Carson/datasets/Labelled_Data_v3.xls",
    label_column="WatercourseRank",
    dict_labels=dict_labels,
    seed=42,
)

model_names = []
pr_aucs = []

initialize_plot()
# Iterate through all files in the directory
for experiment_folder in os.listdir(experiments_dir):
    print(experiment_folder)
    experiment_path = os.path.join(experiments_dir, experiment_folder)

    if os.path.isdir(experiment_path):
        config_path = os.path.join(experiment_path, "config.json")
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            model_name = config.get("tag", "Unknown Tag")
            model_names.append(model_name)

        model_path = os.path.join(experiment_path, "model.pkl")
        model = joblib.load(model_path)
        pr_auc = plot_PR(model, model_name)
        pr_aucs.append(pr_auc)

# Sort based on PR AUC
model_names, pr_aucs = zip(*sorted(zip(model_names, pr_aucs), key=lambda x: x[1]))
plot_PR_end()
plt.savefig("plots/PR_3.png")

plt.clf()
plt.barh(model_names, pr_aucs)
plt.xlabel("AP")
plt.ylabel("Model")
plt.savefig("plots/PR_bar_3.png")
