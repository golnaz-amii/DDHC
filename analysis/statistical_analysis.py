import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Data
xgb = [67.36255392376583, 69.18837720921789, 66.41742209639845, 69.57988053749885, 
    68.95970530820604, 67.68786349909446, 67.84682778200943, 68.2599824269937, 
    66.26372430291401, 67.05409371732465]

hier1 = [70.54159806340824, 71.06745129105326, 67.76457623968605, 70.37003963012053, 
      68.6748314040455, 69.64307879987577, 71.21222003300784, 70.37067827314229, 
      69.78942200391019, 70.2077050195958]

# xgb = [67.63983342420184, 69.26451425076023, 65.34921447212659, 68.3427071272399, 68.34804933716806, 67.41477682528743, 67.93727019276083, 67.97870466265478, 66.6296923240878, 68.22841456459288]
# hier1 = [70.60264106023317, 71.24137356472923, 68.38234685507936, 71.57055688512749, 72.20458291870905, 70.1871644596845, 72.0761386068458, 72.44599374453546, 71.13444516473787, 70.90625730764701]
# # Perform t-test
xgb = np.array(xgb)
hier1 = np.array(hier1)
t_stat, p_value = ttest_ind(xgb, hier1)

# Print t-test results
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("The two groups are statistically different.")
else:
    print("The two groups are not statistically different.")

# Plot distributions
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title("Distribution of Accuracies for XGBoost and Hier1")
sns.kdeplot(xgb, label="XGBoost", color="blue", fill=True, alpha=0.5)
sns.kdeplot(hier1, label="Hier1", color="orange", fill=True, alpha=0.5)

# Add mean and standard deviation lines
plt.axvline(np.mean(xgb), color="blue", linestyle="--", label="XGBoost Mean")
plt.axvline(np.mean(hier1), color="orange", linestyle="--", label="Hier1 Mean")
# plt.axvline(np.mean(xgb) + np.std(xgb), color="blue", linestyle=":", label="XGBoost ± Std")
# plt.axvline(np.mean(xgb) - np.std(xgb), color="blue", linestyle=":")
# plt.axvline(np.mean(hier1) + np.std(hier1), color="orange", linestyle=":", label="Hier1 ± Std")
# plt.axvline(np.mean(hier1) - np.std(hier1), color="orange", linestyle=":")

# Finalize plot
plt.legend()
plt.xlabel("Accuracy")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()

# Save plot
output_dir = "plots/statistical_analysis"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/xgb_vs_hier1.png")

