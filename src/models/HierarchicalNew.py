from .BaseModel import BaseModel
from xgboost import XGBClassifier
import numpy as np
import pickle

#import a basic estimator
from sklearn.base import BaseEstimator, ClassifierMixin

base_dict_labels = {
    0: "No",
    1: "Ephemeral",
    2: "Intermittent",
    3: "Transitional",
    4: "Small",
    5: "Large",
}

class CustomEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = None
        self._estimator_type = "classifier"
     

    def fit(self, X, y):
        pass

    

    def predict_proba(self, X):
        pass
    



class HierarchicalNew(BaseModel):
    def __init__(
        self, dict_labels, seed=42, experiment_path="HierarchicalBasePreprocessor", **model_params
    ):
        '''
        '''
        super().__init__(dict_labels, seed, experiment_path, **model_params)
        self.model = CustomEstimator()
        self.level_1_model = XGBClassifier(random_state=self.seed)
        self.level_2_1_model = XGBClassifier(random_state=self.seed)  # Ephemeral vs (Intermittent, Transitional)
        self.level_2_2_model = XGBClassifier(random_state=self.seed)  # Intermittent vs Transitional
        self.dict_labels = dict_labels
        self.experiment_path = experiment_path

        # set the base estimator's fit, predict, and predict_proba methods
        self.model.fit = self.train
        self.model.predict_proba = self.predict_proba

    def transform_datasset(self, X_train, y_train, X_eval, y_eval, X_test, y_test):
        import pandas as pd
        X_train = pd.DataFrame(X_train, columns=self.X_test.columns)
        X_eval = pd.DataFrame(X_eval, columns=self.X_test.columns)
        X_test = pd.DataFrame(X_test, columns=self.X_test.columns)

        X_train = X_train._append(X_eval)
        y_train = y_train._append(y_eval)

        # Level 1: no vs (ephemeral, intermittent, transitional) vs small vs large
        y_train_level_1 = y_train.apply(lambda x: 0 if x == 0 else (1 if x in [1, 2, 3] else (2 if x == 4 else 3)))
        y_test_level_1 = y_test.apply(lambda x: 0 if x == 0 else (1 if x in [1, 2, 3] else (2 if x == 4 else 3)))

        # Level 2: ephemeral vs (intermittent, transitional)
        mask_level_2_1_train = y_train.isin([1, 2, 3])
        mask_level_2_1_test = y_test.isin([1, 2, 3])

        X_train_level_2_1 = X_train[mask_level_2_1_train]
        y_train_level_2_1 = y_train[mask_level_2_1_train].apply(lambda x: 0 if x == 1 else 1)  # Map 1 -> 0, (2, 3) -> 1
        X_test_level_2_1 = X_test[mask_level_2_1_test]
        y_test_level_2_1 = y_test[mask_level_2_1_test].apply(lambda x: 0 if x == 1 else 1)

        # Level 2: intermittent vs transitional
        mask_level_2_2_train = y_train.isin([2, 3])
        mask_level_2_2_test = y_test.isin([2, 3])

        X_train_level_2_2 = X_train[mask_level_2_2_train]
        y_train_level_2_2 = y_train[mask_level_2_2_train].apply(lambda x: x - 2)  # Map 2 -> 0, 3 -> 1
        X_test_level_2_2 = X_test[mask_level_2_2_test]
        y_test_level_2_2 = y_test[mask_level_2_2_test].apply(lambda x: x - 2)

        data_dict = {
            "1": (X_train, y_train_level_1, X_test, y_test_level_1),
            "2_1": (X_train_level_2_1, y_train_level_2_1, X_test_level_2_1, y_test_level_2_1),
            "2_2": (X_train_level_2_2, y_train_level_2_2, X_test_level_2_2, y_test_level_2_2),
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
            model = getattr(self, f"level_{level}_model")
            model.fit(X_train_level, y_train_level)
            print(
                f"Accuracy of classifier level {level}: ",
                model.score(X_test_level, y_test_level),
            )

    def predict(self, X_test):
        y_pred_level_1 = self.level_1_model.predict(X_test)
        y_pred_level_2_1 = self.level_2_1_model.predict(X_test)
        y_pred_level_2_2 = self.level_2_2_model.predict(X_test)

        y_pred = []
        for i in range(len(y_pred_level_1)):
            if y_pred_level_1[i] == 0:
                y_pred.append(0)
            elif y_pred_level_1[i] == 1:
                if y_pred_level_2_1[i] == 0:
                    y_pred.append(1)
                else:
                    y_pred.append(y_pred_level_2_2[i] + 2)
            elif y_pred_level_1[i] == 2:
                y_pred.append(4)
            elif y_pred_level_1[i] == 3:
                y_pred.append(5)

        return y_pred

    def predict_proba(self, X_test):
        proba_level_1 = self.level_1_model.predict_proba(X_test)
        proba_level_2_1 = self.level_2_1_model.predict_proba(X_test)
        proba_level_2_2 = self.level_2_1_model.predict_proba(X_test)

        n_samples = X_test.shape[0]
        final_proba = np.zeros((n_samples, 6))

        for i in range(n_samples):
            p_no = proba_level_1[i, 0]
            p_group = proba_level_1[i, 1]
            p_small = proba_level_1[i, 2]
            p_large = proba_level_1[i, 3]

            p_ephemeral = proba_level_2_1[i, 0] * p_group
            p_inter_trans = proba_level_2_1[i, 1] * p_group

            p_intermittent = proba_level_2_2[i, 0] * p_inter_trans
            p_transitional = proba_level_2_2[i, 1] * p_inter_trans

            final_proba[i, 0] = p_no
            final_proba[i, 1] = p_ephemeral
            final_proba[i, 2] = p_intermittent
            final_proba[i, 3] = p_transitional
            final_proba[i, 4] = p_small
            final_proba[i, 5] = p_large

            final_proba[i] /= np.sum(final_proba[i])

        return final_proba
