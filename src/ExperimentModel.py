from models.XGBoostModel import XGBoostModel
from models.XGBoostModelSampleWeight import XGBoostModelSampleWeight
from models.RandomForestModel import RandomForestModel
from models.Hierarchical import Hierarchical
from models.HierarchicalBinary import HierarchicalBinary
from models.CustomHier import CustomHier
from models.HierDomain import HierDomain
from models.HierarchicalRF import HierarchicalRF
from models.HierSim import HierSim
from models.HierGeneral import HierGeneral
from models.HierInst import HierInst
from models.TabPFNModel import TabPFNModel
from models.HierarchicalNew import HierarchicalNew

from preprocessors.BasePreprocessor import BasePreprocessor
from preprocessors.RandomUnder import RandomUnder
from preprocessors.RandomOver import RandomOver
from preprocessors.CombinSampler import CombinSampler
from preprocessors.MajorityDel import MajorityDel
from preprocessors.CustomPreprocessor import CustomPreprocessor
from preprocessors.AugmentProcessor import AugmentProcessor
from desc import description
import os
import json
import argparse

dict_labels = {
    0: "No",
    1: "Ephemeral",
    2: "Intermittent",
    3: "Transitional",
    4: "Small",
    5: "Large",
}


class ExperimentModel:
    def __init__(self,seed=None):
     
        self.experiment_name = self.get_experiment_name()
        self.experiment_path = os.path.join(description, self.experiment_name)

        print("address: ", self.experiment_path)

        self.config = self.load_experiment_config()

        if seed is not None:
            self.seed = seed
        else:
            self.seed = self.config["seed"]


        self.dataset_version = self.config["data"]["dataset_version"]
        self.buffer_type = self.config["data"]["buffer_type"]
        self.dataset_path = rf"datasets/buffers/Labelled_StreamsV5_{self.buffer_type}.xls"
        self.label_column = self.config["data"]["label_column"]
        

        self.model = self.initialize_model(self.config["model"])
        self.preprocessor = self.initialize_preprocessor(self.config["preprocessor"])

    def get_experiment_name(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-e", type=str, required=True)
        args = parser.parse_args()
        return args.e

    def load_experiment_config(self):
        config_path = os.path.join(description, self.experiment_name, "config.json")
        with open(config_path, "r") as file:
            config = json.load(file)
        return config

    def initialize_model(self, model_config):
        model_name = model_config["name"]
        model_params = model_config["parameters"]
        print("model params:\n", model_params)

        if model_name == "XGBoostModel":
            return XGBoostModel(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "XGBoostModelSampleWeight":
            return XGBoostModelSampleWeight(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "RandomForestModel":
            return RandomForestModel(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "Hierarchical":
            return Hierarchical(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "HierarchicalBinary":
            return HierarchicalBinary(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "CustomHier":
            return CustomHier(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "HierDomain":
            return HierDomain(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "HierarchicalRF":
            return HierarchicalRF(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "HierSim":
            return HierSim(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "HierGeneral":
            return HierGeneral(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "HierInst":
            return HierInst(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "TabPFNModel":
            return TabPFNModel(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        elif model_name == "HierarchicalNew":
            return HierarchicalNew(
                seed=self.seed,
                experiment_path=self.experiment_path,
                dict_labels=dict_labels,
                **model_params,
            )
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def initialize_preprocessor(self, preprocessor_config):
        preprocessor_name = preprocessor_config["name"]
        preprocessor_params = preprocessor_config["parameters"]
        if preprocessor_name == "BasePreprocessor":
            return BasePreprocessor(
                dataset_path=self.dataset_path,
                label_column=self.label_column,
                seed=self.seed,
                dict_labels=dict_labels,
                **preprocessor_params,
            )
        elif preprocessor_name == "RandomUnder":
            return RandomUnder(
                dataset_path=self.dataset_path,
                label_column=self.label_column,
                seed=self.seed,
                dict_labels=dict_labels,
                **preprocessor_params,
            )
        elif preprocessor_name == "RandomOver":
            return RandomOver(
                dataset_path=self.dataset_path,
                label_column=self.label_column,
                seed=self.seed,
                dict_labels=dict_labels,
                **preprocessor_params,
            )
        elif preprocessor_name == "CombinSampler":
            return CombinSampler(
                dataset_path=self.dataset_path,
                label_column=self.label_column,
                seed=self.seed,
                dict_labels=dict_labels,
                **preprocessor_params,
            )
        elif preprocessor_name == "MajorityDel":
            return MajorityDel(
                dataset_path=self.dataset_path,
                label_column=self.label_column,
                seed=self.seed,
                dict_labels=dict_labels,
                **preprocessor_params,
            )
        elif preprocessor_name == "CustomPreprocessor":
            return CustomPreprocessor(
                dataset_path=self.dataset_path,
                label_column=self.label_column,
                seed=self.seed,
                dict_labels=dict_labels,
                **preprocessor_params,
            )
        elif preprocessor_name == "AugmentProcessor":
            return AugmentProcessor(
                dataset_path=self.dataset_path,
                label_column=self.label_column,
                seed=self.seed,
                dict_labels=dict_labels,
                **preprocessor_params,
            )
        else:
            raise ValueError(f"Invalid preprocessor name: {preprocessor_name}")
