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


class HierInst(BaseModel):
    '''
    Level 1: no vs (ephemeral and intermittent and transitional and small and large)
    Level 2: ephemeral vs (hyperclass 1: intermittent and transitional) and (hyperclass 2: small and large)
    Level 3:
    Classifier 1: intermittent vs transitional
    Classifier 2: small vs large
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
        self.level_tuples = {
            1: [0, (1, 2, 3, 4, 5)],
            2: [1, (2, 3), (4, 5)],
            3: [(2, 3), (4, 5)]
        }
        self.level_1_model = XGBClassifier(random_state=self.seed)
        self.level_2_model = XGBClassifier(random_state=self.seed)
        self.level_3_model_1 = XGBClassifier(random_state=self.seed)
        self.level_3_model_2 = XGBClassifier(random_state=self.seed)
        self.dict_labels = dict_labels
        self.experiment_path = experiment_path

        # set the base estimator's fit, predict, and predict_proba methods
        self.model.fit = self.train
        self.model.predict_proba = self.predict_proba

    def transform_dataset(self, X_train, y_train, X_eval, y_eval, X_test, y_test):
        data_dict = {}
        X_train = X_train._append(X_eval)
        y_train = y_train._append(y_eval)

        # classifier level 1: no vs (ephemeral and intermittent and transitional and small and large)
        y_train_level_1 = y_train.apply(lambda x: 0 if x == 0 else 1)
        y_test_level_1 = y_test.apply(lambda x: 0 if x == 0 else 1)
        data_dict['1'] = (X_train, y_train_level_1, X_test, y_test_level_1)

        # classifier level 2: ephemeral vs (intermittent and transitional) and (small and large)
        mask_level_2_train = y_train_level_1 == 1
        mask_level_2_test = y_test_level_1 == 1

        X_train_level_2 = X_train[mask_level_2_train]
        y_train_level_2 = y_train[mask_level_2_train].apply(lambda x: 0 if x == 1 else (1 if x in [2, 3] else 2))
        X_test_level_2 = X_test[mask_level_2_test]
        y_test_level_2 = y_test[mask_level_2_test].apply(lambda x: 0 if x == 1 else (1 if x in [2, 3] else 2))
        data_dict['2'] = (X_train_level_2, y_train_level_2, X_test_level_2, y_test_level_2)

        # classifier level 3.1: intermittent vs transitional
        mask_level_3_1_train = y_train_level_2 == 1
        mask_level_3_1_test = y_test_level_2 == 1

        # Align the indices of y_train and mask_level_3_1_train
        mask_level_3_1_train = mask_level_3_1_train.reindex(y_train.index, fill_value=False)
        y_train_level_3_1 = y_train[mask_level_3_1_train].apply(lambda x: 0 if x == 2 else 1)

        # Align the indices of y_test and mask_level_3_1_test
        mask_level_3_1_test = mask_level_3_1_test.reindex(y_test.index, fill_value=False)
        y_test_level_3_1 = y_test[mask_level_3_1_test].apply(lambda x: 0 if x == 2 else 1)

        X_train_level_3_1 = X_train_level_2[mask_level_3_1_train]
        X_test_level_3_1 = X_test_level_2[mask_level_3_1_test]
        data_dict['3.1'] = (X_train_level_3_1, y_train_level_3_1, X_test_level_3_1, y_test_level_3_1)

        # classifier level 3.2: small vs large
        mask_level_3_2_train = y_train_level_2 == 2
        mask_level_3_2_test = y_test_level_2 == 2

        # Align the indices of y_train and mask_level_3_2_train
        mask_level_3_2_train = mask_level_3_2_train.reindex(y_train.index, fill_value=False)
        y_train_level_3_2 = y_train[mask_level_3_2_train].apply(lambda x: 0 if x == 4 else 1)

        # Align the indices of y_test and mask_level_3_2_test
        mask_level_3_2_test = mask_level_3_2_test.reindex(y_test.index, fill_value=False)
        y_test_level_3_2 = y_test[mask_level_3_2_test].apply(lambda x: 0 if x == 4 else 1)

        X_train_level_3_2 = X_train_level_2[mask_level_3_2_train]
        X_test_level_3_2 = X_test_level_2[mask_level_3_2_test]
        data_dict['3.2'] = (X_train_level_3_2, y_train_level_3_2, X_test_level_3_2, y_test_level_3_2)


        return data_dict

    def train(self, X_train, y_train, X_eval, y_eval):
        X_test = self.X_test
        y_test = self.y_test
        data_dict = self.transform_dataset(X_train, y_train, X_eval, y_eval, X_test, y_test)
        for level, data in data_dict.items():
            X_train_level, y_train_level, X_test_level, y_test_level = data
            hier_level = level.split(".")[0]
            model_index = "_"+level.split(".")[1] if "." in level else ""
            model_name = f"level_{hier_level}_model{model_index}"
            model = getattr(self, model_name)
            model.fit(X_train_level, y_train_level)
            setattr(self, model_name, model)
            print(f"Model {model_name} trained")
            print("Accuracy:", model.score(X_test_level, y_test_level))

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
        y_pred_level_2 = self.level_2_model.predict(X_test)
        y_pred_level_3_1 = self.level_3_model_1.predict(X_test)
        y_pred_level_3_2 = self.level_3_model_2.predict(X_test)

        y_pred = []
        for i in range(X_test.shape[0]):
            if y_pred_level_1[i] == 0:
                y_pred.append(0)
            elif y_pred_level_2[i] == 0:
                y_pred.append(1)
            elif y_pred_level_2[i] == 1:
                    if y_pred_level_3_1[i] == 0:
                        y_pred.append(2)
                    elif y_pred_level_3_1[i] == 1:
                        y_pred.append(3)
            elif y_pred_level_2[i] == 2:
                    if y_pred_level_3_2[i] == 0:
                        y_pred.append(4)
                    else:  
                        y_pred.append(5)
            
           
        return y_pred

    def predict_proba(self, X_test):
        # Get probabilities for each level
        proba_level_1 = self.level_1_model.predict_proba(X_test)
        proba_level_2 = self.level_2_model.predict_proba(X_test)
        proba_level_3_1 = self.level_3_model_1.predict_proba(X_test)
        proba_level_3_2 = self.level_3_model_2.predict_proba(X_test)

        n_samples = X_test.shape[0]
        final_proba = np.zeros((n_samples, 6))

        for i in range(n_samples):
            p_no = proba_level_1[i, 0]
            p_ephemeral = proba_level_2[i, 0] * proba_level_1[i, 1]
            p_intermittent = proba_level_3_1[i, 0] * proba_level_2[i, 1] * proba_level_1[i, 1]
            p_transitional = proba_level_3_1[i, 1] * proba_level_2[i, 1] * proba_level_1[i, 1]
            p_small = proba_level_3_2[i, 0] * proba_level_2[i, 2] * proba_level_1[i, 1]
            p_large = proba_level_3_2[i, 1] * proba_level_2[i, 2] * proba_level_1[i, 1]

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
            pickle.dump(self.level_2_model, f)
            pickle.dump(self.level_3_model_1, f)
            pickle.dump(self.level_3_model_2, f)
