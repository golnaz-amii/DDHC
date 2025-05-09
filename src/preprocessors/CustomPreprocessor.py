from .BasePreprocessor import BasePreprocessor
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

base_dict_labels = {
    0: "No",
    1: "Ephemeral",
    2: "Intermittent",
    3: "Transitional",
    4: "Small",
    5: "Large",
}

class CustomPreprocessor(BasePreprocessor):
    def __init__(self, dataset_path, label_column, dict_labels, seed):
        self.dataset_path = dataset_path
        self.label_column = label_column
        self.dict_labels = dict_labels
        self.df = self.load_data()
        self.features = self.clean_data()
        self.seed = seed



    def preprocess_data(self):

        # Check for missing values
        missing_columns = [
            col
            for col in self.features + [self.label_column]
            if col not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The following columns are missing from the DataFrame: {missing_columns}"
            )






        #  combine the 0,1, and 2 classes into one class, and label them as 0
        self.df[self.label_column] = self.df[self.label_column].replace({
                                                                        1: 0,
                                                                         2: 0,
                                                                         3: 1,
                                                                         4: 2,
                                                                         5: 3
                                                                         }
                                                                        )





        # Split the data into features and label
        X = self.df[self.features]
        y = self.df[self.label_column]

         # oversample class 3
        oversampler = RandomOverSampler(sampling_strategy={1: 400})
        X, y = oversampler.fit_resample(X,y)





        # Split the data into train, eval, and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.seed
        )

        # Summarize the data
        print("Summary of the data:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_eval shape: {X_eval.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_eval shape: {y_eval.shape}")
        print(f"y_test shape: {y_test.shape}")
        print("y_train distribution:")
        print(y_train.value_counts(normalize=True))
        print("y_eval distribution:")
        print(y_eval.value_counts(normalize=True))
        print("y_test distribution:")
        print(y_test.value_counts(normalize=True))

        return X_train, X_eval, X_test, y_train, y_eval, y_test

