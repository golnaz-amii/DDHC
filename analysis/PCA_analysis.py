from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import time


dict_labels = {
    0: "No",
    1: "Ephemeral",
    2: "Intermittent",
    3: "Transitional",
    4: "Small",
    5: "Large",
}

# reproducibility
SEED = 42
np.random.seed(SEED)


def plot_explained_variance_ratio_per_component(X_train, X_eval, X_test):
    # DO PCA
    # advice from https://stackoverflow.com/questions/23838056/what-is-the-difference-between-transform-and-fit-transform-in-sklearn

    scaler = StandardScaler()
    pca = PCA(random_state=SEED)

    X_train_pca = pca.fit_transform(scaler.fit_transform(X_train))
    X_eval_pca = pca.transform(scaler.transform(X_eval))
    X_test_pca = pca.transform(scaler.transform(X_test))
    print("number of components: ", pca.n_components_)

    # plot explained variance ratio per component
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    # mark the optimal number of components
    optimal_n_components_1 = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.95)
    plt.axvline(optimal_n_components_1, color="red", linestyle="--", ymax=0.95)
    plt.axhline(
        0.95,
        color="red",
        linestyle="--",
        xmax=optimal_n_components_1 / len(pca.explained_variance_ratio_),
    )

    optimal_n_components_2 = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.99)
    plt.axhline(
        0.99,
        color="red",
        linestyle="--",
        xmax=optimal_n_components_2 / len(pca.explained_variance_ratio_),
    )
    plt.axvline(optimal_n_components_2, color="red", linestyle="--", ymax=0.99)

    plt.xticks(
        [
            0,
            optimal_n_components_1,
            optimal_n_components_2,
            len(pca.explained_variance_ratio_),
        ]
    )
    plt.title("Explained variance ratio per component")
    plt.savefig("plots/pca_explained_variance_ratio_v4.png")
    plt.clf()
    return pca, optimal_n_components_1, optimal_n_components_2


def feature_selection_analysis(optimal_n_components_2):
    pca = PCA(random_state=SEED, n_components=optimal_n_components_2)
    scaler = StandardScaler()
    pca.fit_transform(scaler.fit_transform(X_train))

    # most important features (columns) in the dataset based on PCA
    n_important_features = 20
    print("PCA components: ", pca.components_.shape)
    # a table where shows how much each feature contributes to the components
    print(pd.DataFrame(pca.components_, columns=X.columns))
    important_features = (
        pd.DataFrame(pca.components_, columns=X.columns)
        .abs()
        .sum(axis=0)
        .nlargest(n_important_features)
        .index
    )
    print(important_features)
    top_feat = (
        pd.DataFrame(pca.components_, columns=X.columns)
        .abs()
        .sum(axis=0)
        .nlargest(n_important_features)
    )
    print(top_feat)
    # save the important features in reports/important_features_pca.txt
    with open("reports/important_features_pca_v4.txt", "w") as f:
        f.write("Important features based on PCA\n")
        for feature in important_features:
            f.write(f"{feature}\n")


def visualize_pca_components(X, y):
    # visualize the components
    pca = PCA(n_components=2, random_state=SEED)
    scaler = StandardScaler()
    X_pca = pca.fit_transform(scaler.fit_transform(X))
    # color the points based on their labels, and contain a legend based on dict_labels
    for i in range(6):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=dict_labels[i])
    plt.legend()
    plt.xlabel("First component")
    plt.ylabel("Second component")
    plt.title("PCA components")
    plt.savefig("plots/pca_components_v4.png")
    plt.clf()


def plot_accuracy_per_n_components(X_train, X_test, y_train, y_test):
    time_start = time.time()
    bal_acc = []
    acc = []
    class_0_acc = []
    class_1_acc = []
    class_2_acc = []
    class_3_acc = []
    class_4_acc = []
    class_5_acc = []

    for i in range(1, X_train.shape[1]):
        # define the pipeline: oversampler, XGboost
        scaler = StandardScaler()
        model = XGBClassifier(random_state=42)
        pca = PCA(n_components=i, random_state=SEED)
        model.fit(pca.fit_transform(scaler.fit_transform(X_train)), y_train)

        # evaluate the model
        y_pred = model.predict(pca.transform(scaler.transform(X_test)))

        bal_acc.append(balanced_accuracy_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        acc.append(cm.diagonal().sum() / cm.sum())
        class_0_acc.append(cm[0, 0] / cm[0].sum())
        class_1_acc.append(cm[1, 1] / cm[1].sum())
        class_2_acc.append(cm[2, 2] / cm[2].sum())
        class_3_acc.append(cm[3, 3] / cm[3].sum())
        class_4_acc.append(cm[4, 4] / cm[4].sum())
        class_5_acc.append(cm[5, 5] / cm[5].sum())

    plt.plot(bal_acc, label="Balanced accuracy")
    plt.plot(class_0_acc, label="No")
    plt.plot(class_1_acc, label="Ephemeral")
    plt.plot(class_2_acc, label="Intermittent")
    plt.plot(class_3_acc, label="Transitional")
    plt.plot(class_4_acc, label="Small")
    plt.plot(class_5_acc, label="Large")
    plt.plot(acc, label="Accuracy")
    plt.legend(loc="lower right")
    plt.xticks([1, 29, 54, len(pca.explained_variance_ratio_)])
    plt.xlabel("Number of components")
    plt.ylabel("Balanced accuracy")
    plt.title("Balanced accuracy per number of components")
    plt.savefig("plots/balanced_accuracy_per_n_components_v4.png")
    plt.clf()
    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.2f} seconds")


if __name__ == "__main__":
    # Load the data
    dataset_path = "datasets/Labelled_data_v4.xlsx"
    df = pd.read_excel(dataset_path)

    # necessary preprocessing
    df = df.dropna()
    df.drop(columns=["OBJECTID", "Buffer", "StreamID"], inplace=True)

    X = df.drop(columns=["WatercourseRank"])
    y = df["WatercourseRank"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    pca, optimal_n_components_1, optimal_n_components_2 = (
        plot_explained_variance_ratio_per_component(X_train, X_eval, X_test)
    )
    feature_selection_analysis(optimal_n_components_2)
    visualize_pca_components(X_train, y_train)
    plot_accuracy_per_n_components(X_train, X_test, y_train, y_test)
