import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Accuracy and error tracking
train_error_list = {}
val_error_list = {}
test_bal_acc_list = {}
f1_score_list = {}

################ Developer Choices ###################
# number and range of seeds, saving path, confidence constant
# Seeds for reproducibility
seeds = list(range(42, 52))
num_seeds = len(seeds)
print("Number of seeds:", num_seeds)

# saving path
os.makedirs("plots/tests", exist_ok=True)
# confidence constant (choose the confidence level)
confidence_constant = t.ppf(0.975, num_seeds - 1)
######################################################

# Load and preprocess the data
datasets = [5, 10, 20, 30, 40, 50, 60, "Variable"]
path = rf"datasets/Labelled_data_v4.xlsx"
dataset_path = rf"datasets/buffers/Labelled_StreamsV5_Variable.xls"
df = pd.read_excel(dataset_path)

# necessary preprocessing
df = df.dropna()
df.drop(columns=["OBJECTID", "Buffer", "StreamID"], inplace=True)
df["WatercourseRank"] = df["WatercourseRank"].replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 3})

X = df.drop(columns=["WatercourseRank"])
y = df["WatercourseRank"]

# Iterate over seeds
for SEED in seeds:
    clf = XGBClassifier(random_state=SEED)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    # add any necessary preprocessing here
    # e.g Feature selection

    # Train the model
    eval_set = [(X_train, y_train), (X_eval, y_eval)]
    clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Extract training and validation errors
    results = clf.evals_result()
    train_error = results["validation_0"]["mlogloss"]
    val_error = results["validation_1"]["mlogloss"]
    epochs = len(train_error)

    train_error_list[SEED] = train_error
    val_error_list[SEED] = val_error

    print(f"Training error ({SEED}): {train_error[-1]:.2f}")
    print(f"Validation error ({SEED}): {val_error[-1]:.2f}")
    print(f"Number of epochs: {epochs}")

    # Evaluate on the test set
    predictions = clf.predict(X_test)
    test_bal_acc = (
        balanced_accuracy_score(y_test, predictions) * 100
    )  # Convert to percentage
    test_bal_acc_list[SEED] = test_bal_acc

    # macro f1 score per seed
    f1_score_macro = f1_score(y_test, predictions, average="macro")
    f1_score_list[SEED] = f1_score_macro

    print(f"F1 Score (macro) ({SEED}): {f1_score_macro:.2f}")
    print(f"Test balanced accuracy ({SEED}): {test_bal_acc:.2f}%")

# Plot error curves averaged over seeds
plt.figure()
plt.title("XGBoost Learning Curves")
plt.xlabel("Epochs")
plt.ylabel("Error")

# Compute mean and standard deviation of errors
mean_train_error = np.mean([train_error_list[seed] for seed in seeds], axis=0)
mean_val_error = np.mean([val_error_list[seed] for seed in seeds], axis=0)
std_train_error = np.std([train_error_list[seed] for seed in seeds], axis=0)
std_val_error = np.std([val_error_list[seed] for seed in seeds], axis=0)

# Compute confidence intervals
confidence_train_error = confidence_constant * std_train_error / np.sqrt(num_seeds)
confidence_val_error = confidence_constant * std_val_error / np.sqrt(num_seeds)

# Plot mean errors with confidence intervals
plt.plot(range(epochs), mean_train_error, label="Train Error", color="blue")
plt.plot(range(epochs), mean_val_error, label="Validation Error", color="orange")
plt.fill_between(
    range(epochs),
    mean_train_error - confidence_train_error,
    mean_train_error + confidence_train_error,
    color="blue",
    alpha=0.2,
)
plt.fill_between(
    range(epochs),
    mean_val_error - confidence_val_error,
    mean_val_error + confidence_val_error,
    color="orange",
    alpha=0.2,
)
plt.legend()
plt.tight_layout()
plt.grid()

# Save the plot
plt.savefig("plots/tests/xgb_learning_curve.png")

# Compute and display test balanced accuracy statistics
test_bal_acc_values = [test_bal_acc_list[seed] for seed in seeds]
mean_test_bal_acc = np.mean(test_bal_acc_values)
std_test_bal_acc = np.std(test_bal_acc_values)
confidence_interval = confidence_constant * std_test_bal_acc / np.sqrt(num_seeds)
print("#" * 50)
print("Test balanced accuracy statistics:")
print("Mean test balanced accuracy:", mean_test_bal_acc)
print("Confidence interval for test balanced accuracy:", confidence_interval)
print("Standard deviation of test balanced accuracy:", std_test_bal_acc)
print(
    "Standard error of test balanced accuracy:", std_test_bal_acc / np.sqrt(num_seeds)
)
print("#" * 50)

# compute and display f1 score statistics
f1_score_values = [f1_score_list[seed] for seed in seeds]
mean_f1_score = np.mean(f1_score_values)
std_f1_score = np.std(f1_score_values)
confidence_interval_f1 = confidence_constant * std_f1_score / np.sqrt(num_seeds)
print("Test F1 Score statistics:")
print("Mean F1 Score (macro):", mean_f1_score)
print("Confidence interval for F1 Score (macro):", confidence_interval_f1)
print("Standard deviation of F1 Score (macro):", std_f1_score)
print("Standard error of F1 Score (macro):", std_f1_score / np.sqrt(num_seeds))
