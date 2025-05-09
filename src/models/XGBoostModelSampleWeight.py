from .XGBoostModel import XGBoostModel
from sklearn.utils.class_weight import compute_sample_weight


class XGBoostModelSampleWeight(XGBoostModel):
    def __init__(self, dict_labels, seed=42, experiment_path="XGBoostModelSampleWeightBasePreprocessor"):
        super().__init__(dict_labels, seed, experiment_path)

    def train(self, X_train, y_train, X_eval, y_eval):
        # class_weight = {
        #     0: 1,
        #     1: 5,
        #     2: 5,
        #     3: 20,
        #     4: 1,
        #     5: 1,
        # }
        class_weight = "balanced"
        sample_weights = compute_sample_weight(class_weight=class_weight, y=y_train)
        # assign even more weight to the three unrepresented classes
        # sample_weights[y_train == 1] *= 5
        # sample_weights[y_train == 2] *= 5
        # sample_weights[y_train == 3] *= 20
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_eval, y_eval)],
            verbose=True,
            sample_weight=sample_weights,
        )

