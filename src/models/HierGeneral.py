from .BaseModel import BaseModel
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin

# import svm and random forest
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class CustomEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = None
        self._estimator_type = "classifier"

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        pass


class HierGeneral(BaseModel):
    """
    General hierarchical classifier that can handle any set of tuples for classification.
    """

    def __init__(
        self,
        dict_labels,
        # level_tuples,
        seed=42,
        experiment_path="HierarchicalBasePreprocessor",
        **model_params,
    ):
        super().__init__(dict_labels, seed, experiment_path, **model_params)
        self.model = CustomEstimator()
        self.level_tuples = [(0,5), (1,2), (3, 4)]
        """
        configurations:
        1. intuition and domain knowlege: [(0,1), (2,3), (4,5)]
        (1 is the same as greedy - canberra similarity)
        2. greedy - pearson similarity: [(0, 3), (2, 1), (4, 5)]
        3. greedy - euclidean similarity: [(1,2), (0, 3), (4, 5)]
        3. greedy - mahalanobis similarity: 
        5. greedy - custom metric: ?

        """
        self.level_models = {}
        self.dict_labels = dict_labels
        self.experiment_path = experiment_path

        self.level_1_model = XGBClassifier(random_state=self.seed)
        self.level_2_model_1 = XGBClassifier(random_state=self.seed)
        self.level_2_model_2 = XGBClassifier(random_state=self.seed)
        self.level_2_model_3 =XGBClassifier(random_state=self.seed)

        self.dict_labels = dict_labels
        self.experiment_path = experiment_path

        # set the base estimator's fit, predict, and predict_proba methods
        self.model.fit = self.train
        self.model.predict_proba = self.predict_proba

    def transform_datasset(self, X_train, y_train, X_eval, y_eval, X_test, y_test):
        # no need for validation set
        X_train = X_train._append(X_eval)
        y_train = y_train._append(y_eval)

        # classifier level 1: self.level_tuples
        y_train_level_1 = y_train.apply(
            lambda x: (
                0
                if x in self.level_tuples[0]
                else (1 if x in self.level_tuples[1] else 2)
            )
        )
        y_test_level_1 = y_test.apply(
            lambda x: (
                0
                if x in self.level_tuples[0]
                else (1 if x in self.level_tuples[1] else 2)
            )
        )
        y = y_train_level_1._append(y_test_level_1)
        print("Y (level 1):", y.value_counts())

        # classifier level 2.1: self.level_tuples[0]
        mask_level_2_1_train = y_train_level_1 == 0
        mask_level_2_1_test = y_test_level_1 == 0

        X_train_level_2_1 = X_train[mask_level_2_1_train]
        y_train_level_2_1 = y_train[mask_level_2_1_train].apply(
            lambda x: 0 if x == self.level_tuples[0][0] else 1
        )
        X_test_level_2_1 = X_test[mask_level_2_1_test]
        y_test_level_2_1 = y_test[mask_level_2_1_test].apply(
            lambda x: 0 if x == self.level_tuples[0][0] else 1
        )

        y = y_train_level_2_1._append(y_test_level_2_1)
        print("Y (level 2.1):", y.value_counts())

        # classifier level 2.2: no vs transitional
        mask_level_2_2_train = y_train_level_1 == 1
        mask_level_2_2_test = y_test_level_1 == 1

        X_train_level_2_2 = X_train[mask_level_2_2_train]
        y_train_level_2_2 = y_train[mask_level_2_2_train].apply(
            lambda x: 0 if x == self.level_tuples[1][0] else 1
        )
        X_test_level_2_2 = X_test[mask_level_2_2_test]
        y_test_level_2_2 = y_test[mask_level_2_2_test].apply(
            lambda x: 0 if x == self.level_tuples[1][0] else 1
        )

        y = y_train_level_2_2._append(y_test_level_2_2)
        print("Y (level 2.2):", y.value_counts())

        # classifier level 2.3: small vs large
        mask_level_2_3_train = y_train_level_1 == 2
        mask_level_2_3_test = y_test_level_1 == 2

        X_train_level_2_3 = X_train[mask_level_2_3_train]
        y_train_level_2_3 = y_train[mask_level_2_3_train].apply(
            lambda x: 0 if x == self.level_tuples[2][0] else 1
        )
        X_test_level_2_3 = X_test[mask_level_2_3_test]
        y_test_level_2_3 = y_test[mask_level_2_3_test].apply(
            lambda x: 0 if x == self.level_tuples[2][0] else 1
        )

        y = y_train_level_2_3._append(y_test_level_2_3)
        print("Y (level 2.3):", y.value_counts())

        data_dict = {
            "1": (X_train, y_train_level_1, X_test, y_test_level_1),
            "2.1": (
                X_train_level_2_1,
                y_train_level_2_1,
                X_test_level_2_1,
                y_test_level_2_1,
            ),
            "2.2": (
                X_train_level_2_2,
                y_train_level_2_2,
                X_test_level_2_2,
                y_test_level_2_2,
            ),
            "2.3": (
                X_train_level_2_3,
                y_train_level_2_3,
                X_test_level_2_3,
                y_test_level_2_3,
            ),
        }

        return data_dict

    def train(self, X_train, y_train, X_eval, y_eval):
        X_test = self.X_test
        y_test = self.y_test
        data_dict = self.transform_datasset(
            X_train, y_train, X_eval, y_eval, X_test, y_test
        )
        for level, data in data_dict.items():
            X_train_level, y_train_level, X_test_level, y_test_level = data
            hier_level = level.split(".")[0]
            classifier_id = level.split(".")[1] if len(level.split(".")) > 1 else None
            if classifier_id is None:
                model = getattr(self, f"level_{hier_level}_model")
            else:
                model = getattr(self, f"level_{hier_level}_model_{classifier_id}")
            print(
                "level, classifier_id, data shape:",
                level,
                classifier_id,
                X_train_level.shape,
            )
            model.fit(X_train_level, y_train_level)
            print(
                f"Accuracy of classifier level {level}: ",
                model.score(X_test_level, y_test_level),
            )

    def fit(self, X_train, y_train):
        # update self.model
        self.X_ = X_train
        self.y_ = y_train
        self.X_test = None
        self.y_test = None
        self.train(X_train, y_train, None, None)
        return self.model

    def predict(self, X_test):
        y_pred_level_1 = self.level_1_model.predict(X_test)
        y_pred_level_2_1 = self.level_2_model_1.predict(X_test)
        y_pred_level_2_2 = self.level_2_model_2.predict(X_test)
        y_pred_level_2_3 = self.level_2_model_3.predict(X_test)

        y_pred = []
        for i in range(X_test.shape[0]):
            if y_pred_level_1[i] == 0:
                if y_pred_level_2_1[i] == 0:
                    y_pred.append(self.level_tuples[0][0])
                else:
                    y_pred.append(self.level_tuples[0][1])
            elif y_pred_level_1[i] == 1:
                if y_pred_level_2_2[i] == 0:
                    y_pred.append(self.level_tuples[1][0])
                else:
                    y_pred.append(self.level_tuples[1][1])
            else:
                if y_pred_level_2_3[i] == 0:
                    y_pred.append(self.level_tuples[2][0])
                else:
                    y_pred.append(self.level_tuples[2][1])
        return y_pred

    def predict_proba(self, X_test):
        # Get probabilities for each level
        proba_level_1 = self.level_1_model.predict_proba(X_test)
        self.proba_level_2_1 = self.level_2_model_1.predict_proba(X_test)
        self.proba_level_2_2 = self.level_2_model_2.predict_proba(X_test)
        self.proba_level_2_3 = self.level_2_model_3.predict_proba(X_test)

        n_samples = X_test.shape[0]
        final_proba = np.zeros((n_samples, 6))

        for i in range(n_samples):
            p_tuple_1 = proba_level_1[i, 0]
            p_tuple_2 = proba_level_1[i, 1]
            p_tuple_3 = proba_level_1[i, 2]

            for idx, p in enumerate([p_tuple_1, p_tuple_2, p_tuple_3]):
                next_level_prob = getattr(self, f"proba_level_2_{idx+1}")
                for j in range(2):
                    final_proba[i, self.level_tuples[idx][j]] = (
                        p * next_level_prob[i, j]
                    )

            # normalize final_proba[i] to sum to 1
            final_proba[i] /= np.sum(final_proba[i])

        return final_proba

    def save_model(self):
        with open(f"{self.experiment_path}/model.pkl", "wb") as f:
            pickle.dump(self.level_1_model, f)
            pickle.dump(self.level_2_model_1, f)
            pickle.dump(self.level_2_model_2, f)
            pickle.dump(self.level_2_model_3, f)
