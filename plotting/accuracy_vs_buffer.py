import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_palette("deep")
list_of_colors = sns.color_palette("deep", 10)

# Data for XGBoost
xgboost_buffers = ["5m", "10m", "20m", "30m", "40m", "50m", "60m", "variable"]
xgboost_accuracies = [67.09, 67.402, 60.73, 67.77, 68.041, 67.86, 67.71, 74.83]
xgboost_errors = [0.74, 0.69, 1.47, 0.861, 0.793, 0.77, 0.73, 0.77]

# Data for Hierarchical 3
hier3_buffers = ["5m", "10m", "20m", "30m", "40m", "50m", "60m", "variable"]
hier3_accuracies = [68.60962668674101, 69.05779066250452, 62.625779545666504, 69.59506792430277, 
                    69.71702419416805, 69.7547104800216, 69.76565025306795, 76.99379489027919]
hier3_errors = [0.527330998912934, 0.7590082111978274, 1.124592654057365, 0.7121126047699913, 
                0.73216128997449, 0.5784489074413164, 0.7979783687518615, 0.5331689069664477]

# Data for Hier6
hier6_buffers = ["5m", "10m", "20m", "30m", "40m", "50m", "60m", "variable"]
hier6_accuracies = [68.35646192485669, 67.62900360195786, 62.612936611580054, 68.63992404229015, 
                    68.12226544152016, 68.72438090206596, 68.26175442005925, 74.69926173842143]
hier6_errors = [0.7666013991414723, 0.8318301270026505, 1.1243070408593447, 0.7666281384381779, 
                0.7200878707027741, 0.5715702028145998, 0.5812224512480562, 0.6995722048475655]

# Plotting
plt.figure(figsize=(10, 6))

# Plot XGBoost
plt.errorbar(xgboost_buffers, xgboost_accuracies, yerr=xgboost_errors, label="XGBoost", fmt='-o', capsize=5, color=list_of_colors[0])

# Plot Hier3
plt.errorbar(hier3_buffers, hier3_accuracies, yerr=hier3_errors, label="Hier3", fmt='-o', capsize=5, color=list_of_colors[3])

# Plot Hier6
plt.errorbar(hier6_buffers, hier6_accuracies, yerr=hier6_errors, label="Hier6", fmt='-o', capsize=5, color=list_of_colors[6])

# Labels and title
plt.xlabel("Buffer Size")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Buffer Size")
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
os.makedirs("plots/buffers", exist_ok=True)
plt.savefig("plots/buffers/accuracy_vs_buffer.png")
print("Plot saved as plots/buffers/accuracy_vs_buffer.png")
plt.close()
