from .BaseModel import BaseModel
from xgboost import XGBClassifier
import numpy as np
import pickle

# import a basic estimator
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
        self.model = XGBClassifier()
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class HierSim(BaseModel):
    '''
   
    Graph:
        
     
    '''
    def __init__(
        self,
        dict_labels,
        seed=42,
        experiment_path="HierarchicalBasePreprocessor",
        **model_params,
    ):
        super().__init__(dict_labels, seed, experiment_path, **model_params)
        self.model = CustomEstimator()
        self.level_1_model = XGBClassifier(random_state=self.seed)
        self.level_2_model_1 = XGBClassifier(random_state=self.seed)
        self.level_2_model_2 = XGBClassifier(random_state=self.seed)
        self.level_2_model_3 = XGBClassifier(random_state=self.seed)
        self.dict_labels = dict_labels
        self.experiment_path = experiment_path

        # set the base estimator's fit, predict, and predict_proba methods
        self.model.fit = self.train
        self.model.predict_proba = self.predict_proba

    def transform_datasset(self, X_train, y_train, X_eval, y_eval, X_test, y_test):
        # no need for validation set
        X_train = X_train._append(X_eval)
        y_train = y_train._append(y_eval)

        # classifier level 1: (ephemeral and intermittent) vs (no and transitional) vs (small and large)
        y_train_level_1 = y_train.apply(
            lambda x: 0 if x in [1, 2] else (1 if x in [0, 3] else 2)
        )
        y_test_level_1 = y_test.apply(
            lambda x: 0 if x in [1, 2] else (1 if x in [0, 3] else 2)
        )
        y = y_train_level_1._append(y_test_level_1)
        print("Y (level 1):", y.value_counts())

        # classifier level 2.1: ephemeral vs intermittent
        mask_level_2_1_train = y_train_level_1 == 0
        mask_level_2_1_test = y_test_level_1 == 0

        X_train_level_2_1 = X_train[mask_level_2_1_train]
        y_train_level_2_1 = y_train[mask_level_2_1_train].apply(
            lambda x: 0 if x == 1 else 1
        )
        X_test_level_2_1 = X_test[mask_level_2_1_test]
        y_test_level_2_1 = y_test[mask_level_2_1_test].apply(
            lambda x: 0 if x == 1 else 1
        )

        y = y_train_level_2_1._append(y_test_level_2_1)
        print("Y (level 2.1):", y.value_counts())

        # classifier level 2.2: no vs transitional
        mask_level_2_2_train = y_train_level_1 == 1
        mask_level_2_2_test = y_test_level_1 == 1

        X_train_level_2_2 = X_train[mask_level_2_2_train]
        y_train_level_2_2 = y_train[mask_level_2_2_train].apply(
            lambda x: 0 if x == 0 else 1
        )
        X_test_level_2_2 = X_test[mask_level_2_2_test]
        y_test_level_2_2 = y_test[mask_level_2_2_test].apply(
            lambda x: 0 if x == 0 else 1
        )

        y = y_train_level_2_2._append(y_test_level_2_2)
        print("Y (level 2.2):", y.value_counts())

        # classifier level 2.3: small vs large
        mask_level_2_3_train = y_train_level_1 == 2
        mask_level_2_3_test = y_test_level_1 == 2

        X_train_level_2_3 = X_train[mask_level_2_3_train]
        y_train_level_2_3 = y_train[mask_level_2_3_train].apply(
            lambda x: 0 if x == 4 else 1
        )
        X_test_level_2_3 = X_test[mask_level_2_3_test]
        y_test_level_2_3 = y_test[mask_level_2_3_test].apply(
            lambda x: 0 if x == 4 else 1
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
            print("level, classifier_id, data shape:", level, classifier_id, X_train_level.shape)
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
                    y_pred.append(1)
                else:
                    y_pred.append(2)
            elif y_pred_level_1[i] == 1:
                if y_pred_level_2_2[i] == 0:
                    y_pred.append(0)
                else:
                    y_pred.append(3)
            else:
                if y_pred_level_2_3[i] == 0:
                    y_pred.append(4)
                else:
                    y_pred.append(5)
        return y_pred
       

       

   
    def predict_proba(self, X_test):
        # Get probabilities for each level
        proba_level_1 = self.level_1_model.predict_proba(X_test)
        proba_level_2_1 = self.level_2_model_1.predict_proba(X_test)
        proba_level_2_2 = self.level_2_model_2.predict_proba(X_test)
        proba_level_2_3 = self.level_2_model_3.predict_proba(X_test)

        n_samples = X_test.shape[0]
        final_proba = np.zeros((n_samples, 6))

        for i in range(n_samples):
            p_ephemeral_intermittent = proba_level_1[
                i, 0
            ]  # Probability of class 0 (Ephemeral or Intermittent)
            p_no_transitional = proba_level_1[
                i, 1
            ]  # Probability of class 1 (No or Transitional)
            p_small_large = proba_level_1[
                i, 2
            ]  # Probability of class 2 (Small or Large)

            p_no = proba_level_2_2[i, 0] * p_no_transitional

            p_ephemeral = proba_level_2_1[i, 0] * p_ephemeral_intermittent
            p_intermittent = proba_level_2_1[i, 1] * p_ephemeral_intermittent

            p_transitional = proba_level_2_2[i, 1] * p_no_transitional

            p_small = proba_level_2_3[i, 0] * p_small_large
            p_large = proba_level_2_3[i, 1] * p_small_large

            # Assign probabilities to final classes
            final_proba[i, 0] = p_no
            final_proba[i, 1] = p_ephemeral
            final_proba[i, 2] = p_intermittent
            final_proba[i, 3] = p_transitional
            final_proba[i, 4] = p_small
            final_proba[i, 5] = p_large

            # normalize final_proba[i] to sum to 1
            final_proba[i] /= np.sum(final_proba[i])

        return final_proba

    def save_model(self):
        with open(f"{self.experiment_path}/model.pkl", "wb") as f:
            pickle.dump(self.level_1_model, f)
            pickle.dump(self.level_2_model_1, f)
            pickle.dump(self.level_2_model_2, f)
            pickle.dump(self.level_2_model_3, f)
