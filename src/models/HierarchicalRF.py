from .BaseModel import BaseModel
from sklearn.ensemble import RandomForestClassifier
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
    


class HierarchicalRF(BaseModel):
    def __init__(
        self, dict_labels, seed=42, experiment_path="HierarchicalBasePreprocessor", **model_params
    ):
        super().__init__(dict_labels, seed, experiment_path, **model_params)
        self.model = CustomEstimator()
        self.level_1_model = RandomForestClassifier(verbose=True, random_state=self.seed)
        self.level_2_model = RandomForestClassifier(verbose=True, random_state=self.seed)
        self.level_3_model = RandomForestClassifier(verbose=True, random_state=self.seed)
        self.level_4_model = RandomForestClassifier(verbose=True, random_state=self.seed)
        self.level_5_model = RandomForestClassifier(verbose=True, random_state=self.seed)
        self.dict_labels = dict_labels
        self.experiment_path = experiment_path

        # set the base estimator's fit, predict, and predict_proba methods
        self.model.fit = self.train
        self.model.predict_proba = self.predict_proba



    def transform_datasset(self, X_train, y_train, X_eval, y_eval, X_test, y_test):
        # convert back from numpy to pandas

        # no need for validation set
        X_train = X_train._append(X_eval)
        y_train = y_train._append(y_eval)
        # classifier level 1: no vs rest
        # change the labels of the classes, so that for no class, the label is 1, and for the rest of the classes, the label is 0
        y_train_level_1 = y_train.apply(lambda x: 1 if x == 0 else 0)
        y_test_level_1 = y_test.apply(lambda x: 1 if x == 0 else 0)
        y = y_train_level_1._append(y_test_level_1)
        print("Y (level 1):", y.value_counts())

        # classifier level 2: ephemeral vs rest
        # 1. delete the rows with the label "No"
        # 2. change the labels of the classes, so that for ephemeral class, the label is 1, and for the rest of the classes, the label is 0
        # classifier level 2: ephemeral vs rest
        mask_level_2_train = y_train != 0
        mask_level_2_test = y_test != 0

        X_train_level_2 = X_train[mask_level_2_train]
        y_train_level_2 = y_train[mask_level_2_train].apply(
            lambda x: 1 if x == 1 else 0
        )
        X_test_level_2 = X_test[mask_level_2_test]
        y_test_level_2 = y_test[mask_level_2_test].apply(lambda x: 1 if x == 1 else 0)

        y = y_train_level_2._append(y_test_level_2)
        print("Y (level 2):", y.value_counts())

        # classifier level 3: small vs rest
        mask_level_3_train = (y_train != 0) & (y_train != 1)
        mask_level_3_test = (y_test != 0) & (y_test != 1)

        X_train_level_3 = X_train[mask_level_3_train]
        y_train_level_3 = y_train[mask_level_3_train].apply(
            lambda x: 1 if x == 4 else 0
        )
        X_test_level_3 = X_test[mask_level_3_test]
        y_test_level_3 = y_test[mask_level_3_test].apply(lambda x: 1 if x == 4 else 0)

        y = y_train_level_3._append(y_test_level_3)
        print("Y (level 3):", y.value_counts())

        # classifier level 4: transitional vs rest
        mask_level_4_train = (y_train != 0) & (y_train != 1) & (y_train != 4)
        mask_level_4_test = (y_test != 0) & (y_test != 1) & (y_test != 4)

        X_train_level_4 = X_train[mask_level_4_train]
        y_train_level_4 = y_train[mask_level_4_train].apply(
            lambda x: 1 if x == 3 else 0
        )
        X_test_level_4 = X_test[mask_level_4_test]
        y_test_level_4 = y_test[mask_level_4_test].apply(lambda x: 1 if x == 3 else 0)

        y = y_train_level_4._append(y_test_level_4)
        print("Y (level 4):", y.value_counts())

        # classifier level 5: intermittent vs rest
        mask_level_5_train = (
            (y_train != 0) & (y_train != 1) & (y_train != 4) & (y_train != 3)
        )
        mask_level_5_test = (
            (y_test != 0) & (y_test != 1) & (y_test != 4) & (y_test != 3)
        )

        X_train_level_5 = X_train[mask_level_5_train]
        y_train_level_5 = y_train[mask_level_5_train].apply(
            lambda x: 1 if x == 2 else 0
        )
        X_test_level_5 = X_test[mask_level_5_test]
        y_test_level_5 = y_test[mask_level_5_test].apply(lambda x: 1 if x == 2 else 0)

        y = y_train_level_5._append(y_test_level_5)
        print("Y (level 5):", y.value_counts())

        data_dict = {
            "1": (X_train, y_train_level_1, X_test, y_test_level_1),
            "2": (X_train_level_2, y_train_level_2, X_test_level_2, y_test_level_2),
            "3": (X_train_level_3, y_train_level_3, X_test_level_3, y_test_level_3),
            "4": (X_train_level_4, y_train_level_4, X_test_level_4, y_test_level_4),
            "5": (X_train_level_5, y_train_level_5, X_test_level_5, y_test_level_5),
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
    def fit(self, X_train, y_train):
        # update self.model
        self.X_ = X_train
        self.y_ = y_train
        self.train(X_train, y_train, None, None, None, None)
        return  self.model


    def predict(self, X_test):
        y_pred_level_1 = self.level_1_model.predict(X_test)
        y_pred_level_2 = self.level_2_model.predict(X_test)
        y_pred_level_3 = self.level_3_model.predict(X_test)
        y_pred_level_4 = self.level_4_model.predict(X_test)
        y_pred_level_5 = self.level_5_model.predict(X_test)

        y_pred = []
        for i in range(len(y_pred_level_1)):
            if y_pred_level_1[i] == 1:
                y_pred.append(0)
            elif y_pred_level_2[i] == 1:
                y_pred.append(1)
            elif y_pred_level_3[i] == 1:
                y_pred.append(4)
            elif y_pred_level_4[i] == 1:
                y_pred.append(3)
            elif y_pred_level_5[i] == 1:
                y_pred.append(2)
            else:
                predictions = [
                    y_pred_level_1[i],
                    y_pred_level_2[i],
                    y_pred_level_3[i],
                    y_pred_level_4[i],
                    y_pred_level_5[i],
                ]
                if sum(predictions) > 1:
                    confidences = [
                        self.level_1_model.predict_proba([X_test.iloc[i]])[0][1],
                        self.level_2_model.predict_proba([X_test.iloc[i]])[0][1],
                        self.level_3_model.predict_proba([X_test.iloc[i]])[0][1],
                        self.level_4_model.predict_proba([X_test.iloc[i]])[0][1],
                        self.level_5_model.predict_proba([X_test.iloc[i]])[0][1],
                    ]
                    max_confidence_index = np.argmax(confidences)
                    y_pred.append([0, 1, 4, 3, 2][max_confidence_index])
                else:
                    y_pred.append(5)

        return y_pred

    def predict_proba(self, X_test):
        # Get probabilities for each level
        proba_level_1 = self.level_1_model.predict_proba(
            X_test
        )
        proba_level_2 = self.level_2_model.predict_proba(
            X_test
        )
        proba_level_3 = self.level_3_model.predict_proba(
            X_test
        )
        proba_level_4 = self.level_4_model.predict_proba(
            X_test
        )
        proba_level_5 = self.level_5_model.predict_proba(
            X_test
        )

        n_samples = X_test.shape[0]
        final_proba = np.zeros((n_samples, 6))

        for i in range(n_samples):
            p_no = proba_level_1[i, 1]  # Probability of class 0 (No)
            p_rest = proba_level_1[i, 0]  # Probability of NOT being class 0

            p_ephemeral = proba_level_2[i, 1] * p_rest
            p_rest_2 = proba_level_2[i, 0] * p_rest

            p_small = proba_level_3[i, 1] * p_rest_2
            p_rest_3 = proba_level_3[i, 0] * p_rest_2

            p_transitional = proba_level_4[i, 1] * p_rest_3
            p_rest_4 = proba_level_4[i, 0] * p_rest_3

            p_intermittent = proba_level_5[i, 1] * p_rest_4
            p_large = proba_level_5[i, 0] * p_rest_4

            # Assign probabilities to final classes
            final_proba[i, 0] = p_no
            final_proba[i, 1] = p_ephemeral
            final_proba[i, 4] = p_small
            final_proba[i, 3] = p_transitional
            final_proba[i, 2] = p_intermittent
            final_proba[i, 5] = p_large

            # normalize final_proba[i] to sum to 1
            final_proba[i] /= np.sum(final_proba[i])
            # print(final_proba[i])

        return final_proba

    def save_model(self):
        with open(f"{self.experiment_path}/model.pkl", "wb") as f:
            for level in range(1, 6):
                model = getattr(self, f"level_{level}_model")
                pickle.dump(model, f)
            f.close()
