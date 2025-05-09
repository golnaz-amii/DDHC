import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cosine, canberra, mahalanobis
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os
import random
import numpy as np
from src.utils import set_seed


# -------------------- Config --------------------
VERSIONS = ["5m", "10m", "20m", "30m", "40m", "50m", "60m", "Variable"]
VERSION = "Variable"
DATA_PATH = f'datasets/buffers/Labelled_StreamsV5_{VERSION}.xls'
TOP_K_FEATURES =300
LABELS = ["No", "Ephemeral", "Intermittent", "Transitional", "Small", "Large"]
COLORS = ["red", "green", "blue", "purple", "orange", "cyan"]
SEED = 42
METRIC = "mahalanobis"  
EMBEDDING = False
OUTPUT_DIR = f"plots/v5_greedy_hier_clean/{METRIC}/{EMBEDDING}"
# -------------------- Distance Functions --------------------
def pearson_correlation_distance(class_i, class_j):
    class_i = np.mean(class_i, axis=0)
    class_j = np.mean(class_j, axis=0)
    return 1 - pearsonr(class_i.flatten(), class_j.flatten())[0]

    min_len = min(len(class_i), len(class_j))
    class_i, class_j = class_i[:min_len], class_j[:min_len]
    return 1 - pearsonr(class_i.flatten(), class_j.flatten())[0]

def euclidean_distance(class_i, class_j):
    class_i = np.mean(class_i, axis=0)
    class_j = np.mean(class_j, axis=0)
    distance = euclidean(class_i, class_j)
    return distance

    min_len = min(len(class_i), len(class_j))
    class_i, class_j = class_i[:min_len], class_j[:min_len]
    distance = 0
    for i in range(min_len):
        distance += euclidean(class_i[i], class_j[i])
    return distance

def cosine_similarity(class_i, class_j):
    class_i = np.mean(class_i, axis=0)
    class_j = np.mean(class_j, axis=0)
    return cosine(class_i, class_j)

def mahalanobis_distance(class_i, class_j):
    cov = np.cov(np.vstack([class_i, class_j]).T)
    cov += np.eye(cov.shape[0]) * 1e-6  # Add regularization to avoid singular matrix
    inv_cov = np.linalg.pinv(cov)
    mean_diff = np.mean(class_i, axis=0) - np.mean(class_j, axis=0)
    return np.sqrt(mean_diff @ inv_cov @ mean_diff)

def canberra_distance(class_i, class_j):
    class_i = np.mean(class_i, axis=0)
    class_j = np.mean(class_j, axis=0)
    return canberra(class_i, class_j)
    min_len = min(len(class_i), len(class_j))
    class_i, class_j = class_i[:min_len], class_j[:min_len]
    return np.sum(np.abs(class_i - class_j) / (np.abs(class_i) + np.abs(class_j)))


def find_distance_func(metric):
    if metric == "pearson":
        return pearson_correlation_distance
    elif metric == "euclidean":
        return euclidean_distance
    elif metric == "cosine":
        return cosine_similarity
    elif metric == "mahalanobis":
        return mahalanobis_distance
    elif metric == "canberra":
        return canberra_distance
    else:
        raise ValueError(f"Unknown metric: {metric}")


# -------------------- Core Functions --------------------
def load_and_preprocess_data(filepath):
    df = pd.read_excel(filepath, engine="xlrd").dropna()
    df.drop(columns=["OBJECTID", "Buffer", "StreamID"], inplace=True)
    X = df.drop(columns=["WatercourseRank"]).values
    y = df["WatercourseRank"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)

    X = X_train
    y = y_train

    # normalize data #TODO: check if it is necessary, refer to the paper: https://arxiv.org/pdf/2307.00106
    X = StandardScaler().fit_transform(X)
    #selector = SelectKBest(mutual_info_classif, k=TOP_K_FEATURES)
    
   # X_reduced = selector.fit_transform(X, y)
    df_reduced = pd.DataFrame(X)
    df_reduced["WatercourseRank"] = y
   
    return df_reduced

def convert_to_tsne_comps(df, n_components=2, metric=METRIC, random_state=SEED):
    X = df.drop(columns=["WatercourseRank"]).values
    y = df["WatercourseRank"].values

    tsne = TSNE(n_components=n_components, metric=metric, random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    df_tsne = pd.DataFrame(X_tsne, columns=[f"tsne_{i+1}" for i in range(n_components)])
    df_tsne["WatercourseRank"] = y
    return df_tsne


def compute_distance_matrix(df, distance_fn):
    classes = sorted(df["WatercourseRank"].unique())
    matrix = np.zeros((len(classes), len(classes)))
    for i in classes:
        for j in classes:
            if i != j:
                class_i = df[df["WatercourseRank"] == i].drop(columns=["WatercourseRank"]).values
                class_j = df[df["WatercourseRank"] == j].drop(columns=["WatercourseRank"]).values
                matrix[i][j] = round(distance_fn(class_i, class_j), 3)
    matrix /= matrix.max()
    return matrix

def find_min_weight_pairs(matrix):
    pairs = []
    nodes = list(range(matrix.shape[0]))
    while nodes:
        min_val = np.inf
        best_pair = None
        for i in nodes:
            for j in nodes:
                if i != j and matrix[i, j] < min_val:
                    min_val, best_pair = matrix[i, j], (i, j)
        pairs.append((best_pair, min_val))
        nodes.remove(best_pair[0])
        nodes.remove(best_pair[1])
    return pairs

def plot_heatmap(matrix, output_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".4f", cmap="Blues")
    plt.title("Pairwise Distance Heatmap")
    plt.xlabel("Class Index")
    plt.ylabel("Class Index")
    plt.savefig(output_path)
    plt.close()

def plot_tsne_for_pair(df, class_pair, index, output_path, metric):
    i, j = class_pair
    class_i = df[df["WatercourseRank"] == i].drop(columns=["WatercourseRank"]).values
    class_j = df[df["WatercourseRank"] == j].drop(columns=["WatercourseRank"]).values
    X = np.vstack([class_i, class_j])
    y = np.array([0] * len(class_i) + [1] * len(class_j))
    tsne = TSNE(n_components=2, random_state=SEED, metric=metric)
    X_embedded = tsne.fit_transform(X)
    plt.clf()
    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], label=f"Class {i}", color="red")
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], label=f"Class {j}", color="blue")
    plt.legend()
    plt.title(f"TSNE Plot for Classes {i} and {j}")
    plt.savefig(output_path)
    plt.close()

def plot_tsne_all_classes(df, output_path, metric):
    X = df.drop(columns=["WatercourseRank"]).values
    y = df["WatercourseRank"].values
    X_embedded = TSNE(n_components=2, random_state=SEED, metric=metric).fit_transform(X)
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(LABELS):
        plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], label=label, color=COLORS[i])
    plt.legend()
    plt.title("TSNE Plot for All Classes")
    plt.savefig(output_path)
    plt.close()

# -------------------- Main --------------------
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_and_preprocess_data(DATA_PATH)
    if EMBEDDING:
        df = convert_to_tsne_comps(df)
    print(df.shape)
    distance_fn = find_distance_func()  # Replace with other functions to test
    matrix = compute_distance_matrix(df, distance_fn)
    plot_heatmap(matrix, f"{OUTPUT_DIR}/pairwise_distances.png")

    pairs = find_min_weight_pairs(matrix)
    for idx, (pair, weight) in enumerate(pairs[:3]):
        plot_tsne_for_pair(df, pair, idx, f"{OUTPUT_DIR}/tsne_{pair[0]}_{pair[1]}.png", metric=METRIC)

    plot_tsne_all_classes(df, f"{OUTPUT_DIR}/tsne_all_classes.png", metric=METRIC)

    # Log closest pairs
    for couple, weight in pairs:
        print(f"Nodes: {couple}, Weight: {weight}")
    print("3 smallest distances:", np.sort(matrix[np.triu_indices(len(matrix), k=1)])[:3])

if __name__ == "__main__":
    main()
