import os
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from preprocessors.BasePreprocessor import BasePreprocessor
from sklearn.preprocessing import label_binarize



def initialize_plot():
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Paired", len(model_names))


def plot_ROC(model, model_name):
    _, _, X_test, _, _, y_test = preprocessor.preprocess_data()
    # draw roc curve (micro average) for multiclass
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    pred_prob = model.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    thresh = dict()
    roc_auc = dict()
    n_class = len(dict_labels)


    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:, i], pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        interp_tpr = np.interp(mean_fpr, fpr[i], tpr[i])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc[i])
        #plt.plot(fpr[i], tpr[i], lw=1, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc[i]:.2f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        label=f"{model_name}",
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        alpha=0.2,
    )

    return mean_auc


def plot_ROC_end():
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

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
roc_aucs = []

initialize_plot()
# Iterate through all files in the directory
for experiment_folder in os.listdir(experiments_dir):
    # if experiment_folder !='test':
    #     continue
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
        roc_auc = plot_ROC(model
                           , model_name)
        roc_aucs.append(roc_auc)

# Sort based on ROC AUC
model_names, roc_aucs = zip(*sorted(zip(model_names, roc_aucs), key=lambda x: x[1]))
plot_ROC_end()
plt.savefig("plots/ROC_3.png")

plt.clf()
plt.barh(model_names, roc_aucs)
plt.xlabel("ROC AUC")
plt.ylabel("Model")
plt.savefig("plots/ROC_bar_3.png")
