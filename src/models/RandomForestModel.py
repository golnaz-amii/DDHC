from .BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import numpy as np

class RandomForestModel(BaseModel):
    def __init__(self, dict_labels,experiment_path, seed=42, **model_params):
        super().__init__(dict_labels, seed, experiment_path, **model_params)
        self.model = RandomForestClassifier(verbose=True, random_state=self.seed)


    def train(self, X_train, y_train, X_eval, y_eval):
        self.model.fit(X_train, y_train)

    def plot_feature_importance(self):
        # Get feature importances
        importances = self.model.feature_importances_

        # Indices of the most to least important features
        indices = importances.argsort()[::-1]
        feature_names = self.X_train.columns

        # Number of most important features to show
        most_important_features = 20
        print(importances[indices[0]])
        print(np.max(importances))

        #print the first 20 most important features
        print("The 20 most important features are:")
        print(feature_names[indices[:most_important_features]])

        # Plot the feature importances of the forest for the first 20 features
        plt.figure(figsize=(10, 7))
        plt.barh(range(most_important_features), importances[indices[:most_important_features]])
        plt.yticks(range(most_important_features), feature_names[indices[:most_important_features]])
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        # Save the plot to the specified path
        plt.savefig(f"{self.experiment_path}/plots/feature_importance.png")
        plt.clf()


