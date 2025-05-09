from .BaseModel import BaseModel
from tabpfn import TabPFNClassifier
from matplotlib import pyplot as plt
import numpy as np
import faulthandler
faulthandler.enable()

class TabPFNModel(BaseModel):
    def __init__(self, dict_labels,experiment_path, seed=42, **model_params):
        super().__init__(dict_labels, seed, experiment_path, **model_params)
        self.model = TabPFNClassifier(random_state=self.seed, model_path='models/tabpfn-v2-classifier.ckpt', memory_saving_mode=True)
        print("model inititlaized")


    def train(self, X_train, y_train, X_eval, y_eval):
        print(self.model)
        print("what about here")
        print(self.model.fit(X_train, y_train))  # TODO: segmentation error
        self.model.fit(X_train, y_train)
        print("model trained")
