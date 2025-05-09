from .BasePreprocessor import BasePreprocessor
from utils import clean_columns
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler


class AugmentProcessor(BasePreprocessor):
    def __init__(self, dataset_path, label_column, dict_labels, seed):
        super().__init__(dataset_path, label_column, dict_labels, seed)
        np.random.seed(seed)

    def augment(self, X_train, X_eval, y_train, y_eval):
        class_distr = y_train.value_counts(normalize=True)

        X_eval, y_eval = X_eval.align(y_eval, axis=0)
        X_train, y_train = X_train.align(y_train, axis=0)

        for i in range(0, len(self.dict_labels)):
            X_train_class = X_train[y_train == i]
            X_eval_class = X_eval[y_eval == i]

            # calculate the mean and std of each feature for each class
            X_train_mean = X_train_class.mean(axis=0)
            X_train_std = X_train_class.std(axis=0)
            X_eval_mean = X_eval_class.mean(axis=0)
            X_eval_std = X_eval_class.std(axis=0)

            # generate new data from the same distribution for each class
            n_added = int(100*(1-class_distr[i]))
            X_train_new = np.random.normal(
                X_train_mean, X_train_std, (n_added, X_train.shape[1])
            )
            X_eval_new = np.random.normal(
                X_eval_mean, X_eval_std, (n_added, X_eval.shape[1])
            )
            # add the new data to the original data
            X_train = np.vstack((X_train, X_train_new))
            X_eval = np.vstack((X_eval, X_eval_new))
            y_train = np.concatenate((y_train, np.ones(n_added) * i))
            y_eval = np.concatenate((y_eval, np.ones(n_added) * i))

        # change the data to DataFrame
        X_train = pd.DataFrame(X_train, columns=self.features)
        X_eval = pd.DataFrame(X_eval, columns=self.features)
        y_train = pd.Series(y_train, name=self.label_column)
        y_eval = pd.Series(y_eval, name=self.label_column)


        return X_train, X_eval, y_train, y_eval

    def built_in_augment(self, X_train, X_eval, y_train, y_eval):
        ovs = RandomOverSampler(
            random_state=self.seed, sampling_strategy="minority", shrinkage=0.8
        )
        X_train, y_train = ovs.fit_resample(X_train, y_train)
        return X_train, X_eval, y_train, y_eval

    def split_X_y(self):
        self.df = self.df.dropna()
        X = self.df[self.features]
        y = self.df[self.label_column]
        return X, y

    def preprocess_data(self):
        self.initial_analysis()
        X, y = self.split_X_y()
        X_train, X_eval, X_test, y_train, y_eval, y_test = self.split_train_eval_test(
            X, y
        )
        X_train, X_eval, y_train, y_eval = self.augment(
            X_train, X_eval, y_train, y_eval
        )
        self.report_data(X_train, X_eval, X_test, y_train, y_eval, y_test)
        return X_train, X_eval, X_test, y_train, y_eval, y_test
