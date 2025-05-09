from .BaseModel import BaseModel
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import os

class XGBoostModel(BaseModel):
    def __init__(
        self, dict_labels, seed=42, experiment_path="XGBoostModelBasePreprocessor", **model_params
    ):
        super().__init__(dict_labels, seed, experiment_path,**model_params ) #TODO: add this and clean other classes as well
        print(self.seed)
        self.model = XGBClassifier(random_state=self.seed)
        self.dict_labels = dict_labels
        self.experiment_path = experiment_path

    def train(self, X_train, y_train, X_eval, y_eval):
        # do hyperparameter tuning here
        # from sklearn.model_selection import GridSearchCV
        # param_grid = {
        #     "n_estimators": [100, 200, 300, 500],
        #     "max_depth": [3, 4, 5, 10],
        #     "learning_rate": [0.1, 0.01, 0.001],

        # }
        # grid_search = GridSearchCV(
        #     self.model, param_grid, cv=3, n_jobs=-1, verbose=2
        # )
        # grid_search.fit(X_train, y_train)
        # print("Best parameters found: ", grid_search.best_params_)
        # self.model = grid_search.best_estimator_

        self.model.fit(X_train,
                       y_train,
                        eval_set=[(X_train,y_train), (X_eval, y_eval)],
                       verbose=True)


        # self.balanced_accuracy_vs_threshold(X_eval, y_eval)



    def hyperparameter_tuning(self, X_train, y_train):
        grid_search = GridSearchCV(
            self.model,
            self.hyperparameters,
            cv=5,
            n_jobs=-1,
            verbose=2,
        )
        grid_search.fit(X_train, y_train)
        print("Best parameters found: ", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        # save the best hyperparameters in the experiment path
        with open(f"{self.experiment_path}/best_hyperparameters.txt", "w") as f:
            f.write(str(grid_search.best_params_))  # write the best hyperparameters to the file


    def plot_learning_curve(self):
        results = self.model.evals_result()
        train_error = results["validation_0"]["mlogloss"]
        validation_error = results["validation_1"]["mlogloss"]

        plt.figure(figsize=(10, 7))
        plt.plot(train_error, label="Train error")
        plt.plot(validation_error, label="Validation error")
        plt.title("Learning curve")
        plt.xlabel("Number of iterations")
        plt.ylabel("Error")
        plt.legend()
        plt.grid()
        os.makedirs(f"{self.experiment_path}/plots/", exist_ok=True)
        plt.savefig(f"{self.experiment_path}/plots/learning_curve.png")
        plt.close()

    def plot_feature_importance(self):
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 20
        plt.rcParams["axes.titlesize"] = 20
        plt.rcParams["axes.labelsize"] = 20
        plt.rcParams["xtick.labelsize"] = 20
        plt.rcParams["ytick.labelsize"] = 20
        plt.rcParams["legend.fontsize"] = 20
        plt.rcParams["figure.figsize"] = (10, 8)
        # add padding to the figure
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"
        plot_importance(self.model, max_num_features=20)
        # print the important features as a list)
        plt.tight_layout()
        plt.savefig(f"{self.experiment_path}/plots/feature_importance.png")
        plt.clf()
