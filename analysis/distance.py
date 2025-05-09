import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# TODO: check this paper: https://arxiv.org/pdf/2109.05180
def mahalanobis_distance(class_i, class_j):
    # Calculate the covariance matrix
    cov = np.cov(np.vstack([class_i, class_j]).T)
    inv_cov = np.linalg.pinv(cov)
    
    # Calculate the mean of each class
    mean_i = np.mean(class_i, axis=0)
    mean_j = np.mean(class_j, axis=0)
    
    # Calculate the Mahalanobis distance
    distance = np.sqrt((mean_i - mean_j) @ inv_cov @ (mean_i - mean_j))
    return distance

df = pd.read_excel("datasets/Labelled_data_v3.xls", engine="xlrd")
df = df.dropna()

# Calculate pairwise distances
matrices = []
unique_classes = df['WatercourseRank'].unique()
for i in range(6):
    row = []
    for j in range(6):
        
        if i != j:
            class_i = df[df['WatercourseRank'] == i].drop(columns=['WatercourseRank']).values
            class_j = df[df['WatercourseRank'] == j].drop(columns=['WatercourseRank']).values
            distance = mahalanobis_distance(class_i, class_j)
        else:
            distance = 0  # Distance to itself is 0
        print(i,j,distance)  
        row.append(distance)
    matrices.append(row)

# Convert to numpy array and reshape
matrices = np.array(matrices)

# Plot the pairwise distances
sns.set(style="whitegrid")


# Draw the heatmap with the mask
ax = sns.heatmap(matrices, annot=True, fmt=".2f", cmap="Blues")
plt.savefig("pairwise_distances.png")