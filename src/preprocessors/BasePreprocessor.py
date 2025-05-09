from utils import clean_columns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

class BasePreprocessor:
    def __init__(self, dataset_path, label_column, dict_labels, seed):
        self.dataset_path = dataset_path
        self.label_column = label_column
        self.dict_labels = dict_labels
        self.df = self.load_data()
        self.features = self.clean_data()
        self.seed = seed

        # self.feature_selection = preprocessor_params.get("feature_selection", False)
        # self.k = preprocessor_params.get("k", 10)



    def load_data(self):
        self.df = pd.read_excel(self.dataset_path)
        print(self.df.info())
         # necessary preprocessing
        self.df = self.df.dropna()
        self.df.drop(columns=["OBJECTID", "Buffer", "StreamID"], inplace=True)

        # # seperate all the rows with label "5"
        # class_large = self.df[self.df["WatercourseRank"] == 5]
        # self.df = self.df[self.df["WatercourseRank"] != 5]
        # # print the size
        # print("large size:",class_large.shape)

        # # randomly subsample 1900 of class large
        # sampled = class_large.sample(n=1949, random_state=42)

        # # add the sampled data to the original data
        # self.df = pd.concat([self.df, sampled])
        # # print the size
        # print("new dataset size",self.df.shape)
        # print(self.df["WatercourseRank"].value_counts())
    


        return self.df

    def clean_data(self):
        self.features = clean_columns(self.df)
        return self.features

    def split_X_y(self):
        #self.df = self.df.dropna()
        X = self.df[self.features]
        y = self.df[self.label_column]
        # convert to numpy arrays
        # X = X.to_numpy() #TODO: make it coherent
        # y = y.to_numpy()
        return X, y

    def split_train_eval_test(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.seed, stratify=y_train
        )

        return X_train, X_eval, X_test, y_train, y_eval, y_test

    def report_data(self, X_train, X_eval, X_test, y_train, y_eval, y_test):
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

    def initial_analysis(self):
        print("Checking for missing values...")
        missing_features = self.df.columns[self.df.isna().any()]
        for feature in missing_features:
            print(f"{feature:<30} {self.df[feature].isna().sum()}")
        print("Total missing values:")
        print(self.df.isna().sum().sum())

        print("Checking for duplicate values...")
        print(self.df.duplicated().sum())
        duplicate_rows = self.df[self.df.duplicated()]
        if not duplicate_rows.empty:
            print("Indexes of duplicated rows:")
            print(duplicate_rows.index.tolist())

        print("Checking for class imbalance...")
        print(self.df[self.label_column].value_counts(normalize=True))
    
    def feature_selection(self, X_train, y_train, X_eval, X_test):
        if True:
            k = 40
            selector = SelectKBest(mutual_info_classif, k=k)
            X_train_index = X_train.index
            X_eval_index = X_eval.index
            X_test_index = X_test.index
            X_train = selector.fit_transform(X_train, y_train)
            X_eval = selector.transform(X_eval)
            X_test = selector.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=self.features[:k], index=X_train_index)
            X_eval = pd.DataFrame(X_eval, columns=self.features[:k], index=X_eval_index)
            X_test = pd.DataFrame(X_test, columns=self.features[:k],  index=X_test_index)
        return X_train, X_eval, X_test

    def preprocess_data(self):
        self.initial_analysis()
        X, y = self.split_X_y()
        X_train, X_eval, X_test, y_train, y_eval, y_test = self.split_train_eval_test(
            X, y
        )
       # X_train, X_eval, X_test = self.feature_selection(X_train, y_train, X_eval, X_test)

        # oversampling
        # ovs = RandomOverSampler( random_state=42)
        # X_train, y_train = ovs.fit_resample(X_train, y_train)

        #undersampling
        # uds = RandomUnderSampler(random_state=42)
        # X_train, y_train = uds.fit_resample(X_train, y_train)

        #smote
        # smote = SMOTE(random_state=42)
        # X_train, y_train = smote.fit_resample(X_train, y_train) 
        


        self.report_data(X_train, X_eval, X_test, y_train, y_eval, y_test)
        print("data preprocessed")
        print(type(X_train), type(X_eval), type(X_test), type(y_train), type(y_eval), type(y_test))
        print(X_train.shape, X_eval.shape, X_test.shape, y_train.shape, y_eval.shape, y_test.shape)
        return X_train, X_eval, X_test, y_train, y_eval, y_test
