from .BaseModel import BaseModel
from xgboost import XGBClassifier


import numpy as np


base_dict_labels = {
    0: "No",
    1: "Ephemeral",
    2: "Intermittent",
    3: "Transitional",
    4: "Small",
    5: "Large",
}

class Hierarchical(BaseModel):
    # using https://github.com/globality-corp/sklearn-hierarchical-classification
    def __init__(
        self, dict_labels, seed=42, experiment_path="HierarchicalBasePreprocessor"
    ):
        self.seed = seed
        self.level_1_model = XGBClassifier(random_state=self.seed)
        self.level_2_model = XGBClassifier(random_state=self.seed)
        self.level_3_model = XGBClassifier(random_state=self.seed)
        self.model = self.level_1_model
        self.class_hierarchy = {
           1:
           {
               0: 0,
               1: 1,
               2: 1,
               3: 1,
               4: 2,
               5: 3,
           },
           2:
           {
               0: 'del',
               1: 0,
               2: 1,
               3: 1,
               4: 'del',
               5: 'del'
           },
           3:
           {
               0: 'del',
               1: 'del',
               2: 0,
               3: 1,
               4: 'del',
               5: 'del'
           }
        }
        self.dict_labels = dict_labels
        self.experiment_path = experiment_path


    def transform_dataset(self,X, y, level):
        for i in range(len(y)):
            mask = self.class_hierarchy[level][y[i]]
            print("mask: ", mask)
            if mask == 'del':
                X.pop(i)
                y.pop(i)
            else:
                y[i] = mask
        print(" X shape: ", len(X))
        print(" y shape: ", len(y))
        print(" y unique: ", np.unique(y))

        return X, y

    def train(self, X_train, y_train, X_eval, y_eval):
        pass






