import numpy as np

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, classifier_uncertainty, classifier_margin, entropy_sampling, margin_sampling
from modAL.disagreement import vote_entropy_sampling
# from sklearn.datasets import load_digits

from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Import tqdm for the progress bar

from ExperimentModel import ExperimentModel

import argparse

parser = argparse.ArgumentParser(description='Active Learning with Query Strategies')

parser.add_argument('query_strategy', type=str,
                    choices=['uncertainty_sampling', 'classifier_uncertainty', 'classifier_margin', 'entropy_sampling', 'margin_sampling'],
                    help='Query strategy to use for active learning')
parser.add_argument('n_initial', type=int, help='Number of initial samples')
parser.add_argument('n_queries', type=int, help='Number of queries')

# Parse and access the arguments
args = parser.parse_args()
query_strategy = args.query_strategy
n_initial = args.n_initial
n_queries = args.n_queries

# Initialize ExperimentModel
ExperimentModel = ExperimentModel()
model = ExperimentModel.model
preprocessor = ExperimentModel.preprocessor

# Preprocess data
X_train, X_eval, X_test, y_train, y_eval, y_test = preprocessor.preprocess_data()

# Convert Pandas DataFrames to NumPy arrays
X_train = X_train.to_numpy()
X_eval = X_eval.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_eval = y_eval.to_numpy()
y_test = y_test.to_numpy()

# Create pool
def create_pool(X_train, initial_idx):
    import pandas as pd
    if len(initial_idx) == 0:
        return X_train
    if not isinstance(initial_idx, (list, pd.Index, np.ndarray)):
        try:
            initial_idx = list(initial_idx)
        except TypeError:
            raise TypeError("initial_idx must be a list, array, or Pandas Index")
    num_rows = len(X_train)
    indices_to_keep = np.setdiff1d(np.arange(num_rows), initial_idx)
    if not indices_to_keep.size:
        return pd.DataFrame(columns=X_train.columns)
    pool = X_train[indices_to_keep]
    return pool

initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_initial, y_initial = X_train[initial_idx], y_train[initial_idx]
X_pool = create_pool(X_train, initial_idx)
y_pool = create_pool(y_train, initial_idx)


# uncertainty_sampling, classifier_uncertainty, classifier_uncertainty, classifier_margin, entropy_sampling, margin_sampling

if query_strategy == 'uncertainty_sampling':
    query_strategy = uncertainty_sampling

elif query_strategy == 'classifier_uncertainty':
    query_strategy = classifier_uncertainty

elif query_strategy == 'classifier_margin':
    query_strategy = classifier_margin

elif query_strategy == 'entropy_sampling':
    query_strategy = entropy_sampling

elif query_strategy == 'entropy_sampling':
    query_strategy = vote_entropy_sampling

elif query_strategy == 'margin_sampling':
    query_strategy = margin_sampling

else:
    raise ValueError("Invalid query strategy")

# Initialize ActiveLearner
learner = ActiveLearner(
    estimator=model,
    query_strategy=query_strategy,
    X_training=X_initial, y_training=y_initial
)

# Active learning loop
queries = []
live_plot = False
accuracy_scores = [learner.score(X_test, y_test)]

for i in tqdm(range(n_queries)):  # Wrap the loop with tqdm
    query_idx, query_inst = learner.query(X_pool)
    queries.append(y_pool[query_idx])
    learner.teach(query_inst.reshape(1, -1), y_pool[query_idx])
    X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
    accuracy_scores.append(learner.score(X_test, y_test))

# Print final accuracy
print(f"Final accuracy after {n_queries} queries: {accuracy_scores[-1]}")

sns.set_style("white")
plt.figure(figsize=(10, 5))
plt.title('Accuracy of your model')
plt.plot(range(n_queries + 1), accuracy_scores)
plt.scatter(range(n_queries + 1), accuracy_scores)
plt.xlabel('number of queries')
plt.ylabel('accuracy')
# Save the final figure
if callable(query_strategy):
        query_strategy_name = query_strategy.__name__
else:
    query_strategy_name = query_strategy

plt.savefig(f"../experiments/AL_results/accuracy_plot_{query_strategy_name}_{n_initial}_{n_queries}.png")
plt.close()
