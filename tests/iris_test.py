from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
#### try different classifiers for this simple multiclass classification problem ####
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
####################################################################################


def test_iris():
    # fix the random seed
    seed = 42

    # load the data
    df = pd.read_excel("datasets/Labelled_data_v3.xls", engine="xlrd")
    # df = df.dropna()

    X,y = load_iris(return_X_y=True)

    # split the data into train, eval, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )

    # add any preprocessing here 
    

    clf = CatBoostClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    # Predict labels
    predictions = clf.predict(X_test)

    # per class accuracy
    cm_normalized = confusion_matrix(y_test, predictions, normalize="true")
    cm_normalized_percentage = cm_normalized * 100
    for i in range(len(cm_normalized_percentage)):
        print(f"Accuracy class {i}: {cm_normalized_percentage[i,i]:.2f}%")
    balanced_accuracy = cm_normalized_percentage.trace() / len(cm_normalized_percentage)
    print(f"Balanced accuracy: {balanced_accuracy:.2f}%")


   
test_iris()
