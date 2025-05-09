import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif as mutual_info
from sklearn.preprocessing import StandardScaler
from src.utils import set_seed


set_seed(0)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 30
plt.rcParams["axes.titlesize"] = 30
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["figure.figsize"] = (10, 8)
# add padding to the figure
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"


VERSIONS = ["5m", "10m", "20m", "30m", "40m", "50m", "60m", "Variable"]
VERSION = "60m"
dataset_path = f"datasets/buffers/Labelled_StreamsV5_{VERSION}.xls"
folder = f"tsne_v5/{VERSION}"
os.makedirs(f"plots/{folder}", exist_ok=True)


df = pd.read_excel(dataset_path, engine="xlrd").dropna()
df.drop(columns=["OBJECTID", "Buffer", "StreamID"], inplace=True)
X = df.drop(columns=["WatercourseRank"]).values
y = df["WatercourseRank"].values

#feature normalization and feature selection
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# selector = SelectKBest(mutual_info, k=20)
# X = selector.fit_transform(X, y)
class_names = ["Dry", "Ephemeral", "Intermittent", "Transitional", "Small", "Large"]
colors = ["red", "green", "blue", "purple", "orange", "cyan"]


# Perform t-SNE for 2D
tsne_2d = TSNE(n_components=2, random_state=42, metric="canberra")
X_tsne_2d = tsne_2d.fit_transform(X)


# Plot the t-SNE results for one class vs all others in 2D
for i, class_name in enumerate(class_names):
    plt.figure(figsize=(10, 8))
    plt.scatter(
        X_tsne_2d[y == i, 0], X_tsne_2d[y == i, 1], label=class_name, color="red"
    )
    plt.scatter(
        X_tsne_2d[y != i, 0],
        X_tsne_2d[y != i, 1],
        label="Others",
        color="blue",
        alpha=0.1,
    )
    #plt.legend()
    plt.title(f"t-SNE of {class_name} vs All Others (2D)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(f"plots/{folder}/tsne_{class_name}_vs_all_2d.png")
    plt.close()

# Plot all classes in one 2D plot
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    plt.scatter(
        X_tsne_2d[y == i, 0], X_tsne_2d[y == i, 1], label=class_name, color=colors[i]
    )
#plt.legend()
# have a seperate plot for the legend
plt.title(f"t-SNE of All Classes (2D) - {VERSION} Buffer")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.savefig(f"plots/{folder}/tsne_all_classes_2d.png")
plt.close()

plt.figure(figsize=(10, 2))
handles = [
    plt.Line2D([0], [0], marker="o", color="w", label=class_name, markerfacecolor=colors[i], markersize=15)
    for i, class_name in enumerate(class_names)
]
plt.legend(
    handles=handles,
    title="Watercourse Rank",
    loc="center",
    bbox_to_anchor=(0.5, 0.5),
    fontsize=20,
    ncol=len(class_names),  # Stack labels horizontally
)
plt.axis("off")
plt.savefig(f"plots/{folder}/legend.png", bbox_inches="tight")
plt.close()
