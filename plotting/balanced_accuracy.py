import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Path to the directory containing experiment folders
experiments_dir = "/Users/Golnaz/Desktop/Golnaz/Carson/experiments/v3"


model_names = []
balanced_accuracies = []



for experiment_folder in os.listdir(experiments_dir):
    experiment_path = os.path.join(experiments_dir, experiment_folder)

    if os.path.isdir(experiment_path):
        config_path = os.path.join(experiment_path, "config.json")
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            model_name = config.get("model", {}).get("name", "Unknown Model")
            model_name = config.get("preprocessor", {}).get("name", "Unknown Preprocessor")
            #model name is under "tag", in the json file
            model_name = config.get("tag", "Unknown Tag")
            model_names.append(model_name)

        report_path = os.path.join(experiment_path, "report.txt")
        with open(report_path, "r") as report_file:
            for line in report_file:
                if "balanced accuracy" in line.lower():
                    next_line = next(report_file).strip()

                    next_line = next_line.replace("%", "").strip()
                    print(next_line)
                    balanced_accuracy = float(next_line)
                    balanced_accuracies.append(balanced_accuracy)
                    break


plt.figure(figsize=(10, 6))
print(model_names)
print(balanced_accuracies)
# sort based on balanced accuracy
model_names, balanced_accuracies = zip(*sorted(zip(model_names, balanced_accuracies), key=lambda x: x[1]))
print(model_names)
print(balanced_accuracies)
model_names = np.array(model_names)
balanced_accuracies = np.array(balanced_accuracies)
plt.barh(model_names, balanced_accuracies,
         #color=sns.color_palette("Paired", len(model_names))
         )
plt.xlabel("Balanced Accuracy")
plt.ylabel("Data Preprocessing Method")
plt.title("Balanced Accuracy Comparison of Different Data Preprocessing Methods")
plt.tight_layout()
plt.savefig("plots/balanced_accuracy_comparison_v3.png")
