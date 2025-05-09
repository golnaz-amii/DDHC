from ExperimentModel import ExperimentModel
import numpy as np
from scipy.stats import t

if __name__ == "__main__":

    seeds = list(range(42, 52))
    num_seeds = len(seeds)
    print("Number of seeds:", num_seeds)
    confidence_constant = t.ppf(0.975, num_seeds - 1)
    performance_metrics = []  # To store performance metrics for each run

    for seed in seeds:
        print(f"Running experiment with seed: {seed}")

        ExpModel = ExperimentModel(seed=seed)  # Pass seed to the model
        model = ExpModel.model
        preprocessor = ExpModel.preprocessor

        # Preprocess the data
        X_train, X_eval, X_test, y_train, y_eval, y_test = (
            preprocessor.preprocess_data()
        )

        model.set_data(X_train, y_train, X_eval, y_eval, X_test, y_test)

        # Train the model
        model.train(X_train, y_train, X_eval, y_eval)

        # Evaluate the model
        performance = model.evaluate(
            X_test, y_test
        )  # Assume this returns a performance metric (e.g., accuracy)
        performance_metrics.append(performance)

        # Optionally, save the model for each seed
        model.save_model()

    # Calculate mean and standard deviation of performance metrics
    mean_performance = np.mean(performance_metrics)
    std_performance = np.std(performance_metrics)
    print("list of performance metrics:", performance_metrics)
    print(f"Mean Performance: {mean_performance}")
    print(f"confidence interval: {confidence_constant * std_performance / np.sqrt(num_seeds)}")
