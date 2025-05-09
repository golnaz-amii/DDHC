import matplotlib.pyplot as plt

# # comparing the hierarchical models we have tried so far
# model_namees = ["XGB", "size", "size_large","canberra", "pearson", "mah", "custom"]
# accuracies = [58.77, 66.04,65.48, 59.07,62.55, 60.43, 63.75 ]
# # sort the accuracies 
# accuracies, model_namees = zip(*sorted(zip(accuracies, model_namees)))
# plt.bar(model_namees, accuracies)
# plt.xlabel('Model Names')
# plt.ylabel('Accuracy')
# plt.title('Model Accuracy')
# plt.tight_layout()
# plt.savefig("models_accuracy.png")


# comparing the hierarchical models we have tried so far
model_namees = ["XGB", "size", "size_large","canberra", "pearson", "euclidean"]
accuracies = [74.56, 75.31,75.71, 76.55,73.32, 74.72]
# sort the accuracies 
accuracies, model_namees = zip(*sorted(zip(accuracies, model_namees)))
plt.bar(model_namees, accuracies)
plt.xlabel('Model Names')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

# Add accuracy values on top of each bar
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.5, f'{acc:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("models_accuracy.png")