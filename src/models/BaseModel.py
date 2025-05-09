from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
)
from utils import plot_confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle




class BaseModel:

    def __init__(
        self,
        dict_labels,
        seed=42,
        experiment_path="BaseModelBasePreprocessor",
        **model_params,
    ):
        self.model = None
        self.seed = seed
        self.dict_labels = dict_labels
        self.experiment_path = experiment_path
        self.model_params = model_params
        self.eval_top_k = self.model_params.get("top_k", False)
        self.hyperparameters = self.model_params.get("hyperparameters", None)
        print(type(self.hyperparameters))
        print(self.hyperparameters)

    def set_data(self, X_train, y_train, X_eval, y_eval, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.X_test = X_test
        self.y_test = y_test

    def train(self, X_train, y_train, X_eval, y_eval):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_eval(self, X):
        if self.eval_top_k:
            y_pred = self.top_k_pred(X, self.y_test, self.X_train, self.y_train)
        else:
            y_pred = self.predict(X)
        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.model.predict(X_test)  # type: ignore
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def evaluate(self, X_test, y_test):
        y_pred = self.predict_eval(X_test)  # type: ignore
        # TODO: fix thisl later
        cm = confusion_matrix(y_test, y_pred)
        print("Evaluation report:")
        print("Classification report:\n", classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
        print("Total accuracy:\n", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("Per class accuracy:")
        for i in range(cm.shape[0]):
            print(
                f"Class {self.dict_labels[i]}: {cm[i][i] / sum(cm[i]) * 100:.2f}%"  # type: ignore
            )
        print(
            "Balanced accuracy:",
            f"{balanced_accuracy_score(y_test, y_pred) * 100:.2f}%",
        )
        plot_confusion_matrix(
            cm,
            experiment_path=self.experiment_path + "/plots/",
            dict_labels=self.dict_labels,
            fmt="d",
        )
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        plot_confusion_matrix(
            cm_normalized,
            experiment_path=self.experiment_path + "/plots/",
            dict_labels=self.dict_labels,
            normal=True,
            fmt=".2f",
        )

        # save everything in the experiment folder, report.txt
        # make sure the folder exists
        os.makedirs(self.experiment_path, exist_ok=True)
        with open(f"{self.experiment_path}/report.txt", "w") as f:
            f.write("Evaluation report:\n")
            f.write("Classification report:\n")
            f.write(str(classification_report(y_test, y_pred)))
            f.write("Confusion matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))
            f.write("\nTotal accuracy:\n")
            f.write(str(f"{accuracy_score(y_test, y_pred) * 100:.2f}%"))
            f.write("\nPer class accuracy:\n")
            for i in range(cm.shape[0]):
                f.write(
                    f"Class {self.dict_labels[i]}: {cm[i][i] / sum(cm[i]) * 100:.2f}%\n"  # type: ignore
                )
            f.write("Balanced accuracy:\n")
            f.write(str(f"{balanced_accuracy_score(y_test, y_pred) * 100:.2f}%"))

            f.close()
    
        # # save the index of all the train and test and validation data for QGIS
        # np.save(f"{self.experiment_path}/train_index.npy", self.X_train.index)
        # np.save(f"{self.experiment_path}/test_index.npy", self.X_test.index)
        # np.save(f"{self.experiment_path}/eval_index.npy", self.X_eval.index)

        # # save all the index of misclassified data for QGIS
        # misclassified_index = X_test[y_test != y_pred].index
        # np.save(f"{self.experiment_path}/misclassified_index.npy", misclassified_index)

        # #save all the actual and predicted values
        # np.save(f"{self.experiment_path}/y_test.npy", y_test)
        # np.save(f"{self.experiment_path}/y_pred.npy", y_pred)

        balanced_accuracy = balanced_accuracy_score(y_test, y_pred) * 100
        #f1_score = f1_score(y_test, y_pred, average="macro", labels=np.unique(y_pred))

        return balanced_accuracy


    def plot_roc_auc_curves(self, X_test, y_test):
        # from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        pred_prob = self.model.predict_proba(X_test)

        # roc curve for classes
        fpr = {}
        tpr = {}
        thresh = {}
        roc_auc = dict()

        n_class = len(self.dict_labels)

        for i in range(n_class):
            fpr[i], tpr[i], thresh[i] = roc_curve(
                y_test_binarized[:, i], pred_prob[:, i]
            )
            roc_auc[i] = auc(fpr[i], tpr[i])

            # plotting
            plt.plot(
                fpr[i],
                tpr[i],
                linestyle="--",
                label="%s vs Rest (AUC=%0.2f)" % (self.dict_labels[i], roc_auc[i]),
            )

        # plot the ROC curve for the multiclass, using the micro average
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test_binarized.ravel(), pred_prob.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            linestyle="--",
            label="micro-average ROC curve (AUC = %0.2f)" % roc_auc["micro"],
        )

        # # plot the ROC curve for the multiclass, using the macro average
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(n_class):
        #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # mean_tpr /= n_class
        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        # plt.plot(
        #     fpr["macro"],
        #     tpr["macro"],
        #     linestyle="--",
        #     label="macro-average ROC curve (AUC = %0.2f)" % roc_auc["macro"],
        # )
        plt.plot([0, 1], [0, 1], "--", color="black", label="Chance - AUC = 0.5")
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive rate")
        plt.legend()
        plt.savefig(f"{self.experiment_path}/plots/roc_auc_curve.png")

    def balanced_accuracy_vs_threshold(self, X_test, y_test):
        # TODO: how do i know current threshold is 0.5?
        plt.clf()
        thresholds = np.linspace(0, 1, 100)
        balanced_accuracies = []
        # handle multiclass
        y_pred = self.model.predict_proba(X_test)
        for threshold in thresholds:
            y_pred_thresholded = np.argmax(y_pred > threshold, axis=1)
            balanced_accuracies.append(
                balanced_accuracy_score(y_test, y_pred_thresholded)
            )
        # choose the maximum balanced accuracy and show it on plot
        max_balanced_accuracy = max(balanced_accuracies)
        max_threshold = thresholds[balanced_accuracies.index(max_balanced_accuracy)]
        plt.axvline(
            x=max_threshold,
            color="r",
            linestyle="--",
            label=f"Max balanced accuracy: {max_balanced_accuracy:.2f}",
        )
        # print the x value
        plt.text(
            max_threshold, max_balanced_accuracy, f"{max_threshold:.2f}", rotation=90
        )
        plt.text(0, max_balanced_accuracy, f"{max_balanced_accuracy:.2f}")
        plt.plot(thresholds, balanced_accuracies)
        plt.xlabel("Threshold")
        plt.ylabel("Balanced accuracy")
        plt.title("Balanced accuracy vs threshold")
        plt.savefig(f"{self.experiment_path}/plots/balanced_accuracy_vs_threshold.png")

    def plot_precision_recall_curve(self, X_test, y_test):
        plt.clf()

        precision = dict()
        recall = dict()
        average_precision = dict()
        _ = dict()

        y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
        pred_prob = self.predict_proba(X_test)
        n_class = len(self.dict_labels)

        for i in range(n_class):
            precision[i], recall[i], _[i] = precision_recall_curve(
                y_test_binarized[:, i], pred_prob[:, i]
            )
            average_precision[i] = average_precision_score(
                y_test_binarized[:, i], pred_prob[:, i]
            )
            plt.plot(
                recall[i],
                precision[i],
                linestyle="--",
                label="%s vs Rest (AP=%0.2f)"
                % (self.dict_labels[i], average_precision[i]),
            )

        # plot the precision-recall curve for the multiclass, using the micro average
        # precision["micro"], recall["micro"], _["micro"] = precision_recall_curve(
        #     y_test_binarized.ravel(), pred_prob.ravel()
        # )
        # average_precision["micro"] = average_precision_score(
        #     y_test_binarized, pred_prob, average="micro"
        #  )
        # plt.plot(
        #     recall["micro"],
        #     precision["micro"],
        #     linestyle="--",
        #     label="micro-average Precision-recall curve (AP = %0.2f)" % average_precision["micro"],
        # )
        # plot chance level
        # average_precision["chance"] = np.mean(y_test_binarized)
        # plt.axhline(
        #     y=np.mean(y_test_binarized),
        #     color="black",
        #     linestyle="--",
        #     label="Chance level",
        # )
        # from sklearn.metrics import PrecisionRecallDisplay
        # from collections import Counter
        # display = PrecisionRecallDisplay(
        #     recall=recall["micro"],
        #     precision=precision["micro"],
        #     average_precision=average_precision["micro"],
        #     prevalence_pos_label=Counter(y_test.ravel())[1] / y_test.size,
        # )
        # display.plot(plot_chance_level=True)
        # _ = display.ax_.set_title("Micro-averaged over all classes")

        # # add standard deviation for the average
        # std_precision = np.std([precision[i] for i in range(n_class)], axis=0)
        # std_recall = np.std([recall[i] for i in range(n_class)], axis=0)
        # averrage_precision = np.mean([average_precision[i] for i in range(n_class)])
        # average_recall = np.mean([recall[i] for i in range(n_class)])
        # plt.fill_between(
        #     recall["micro"],
        #     precision["micro"] - std_precision,
        #     precision["micro"] + std_precision,
        #     alpha=0.2,
        #     color="b",
        # )

        # # find the best threshold
        # best_threshold = _["micro"][np.argmax(precision["micro"] * recall["micro"])]
        # plt.axvline(
        #     x=best_threshold,
        #     color="r",
        #     linestyle="--",
        #     label=f"Best threshold: {best_threshold:.2f}",
        # )
        # print(f"Best threshold: {best_threshold:.2f}")
        # #report multiclass balanced accuracy at best threshold
        # y_pred = self.model.predict_proba(X_test)
        # y_pred_thresholded = np.argmax(y_pred > best_threshold, axis=1)
        # print(
        #     "Balanced accuracy at best threshold:",
        #     f"{balanced_accuracy_score(y_test, y_pred_thresholded) * 100:.2f}%",
        # )
        # print balanced accuracy at 0.5

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-recall curve")
        plt.savefig(f"{self.experiment_path}/plots/precision_recall_curve.png")

    def plot_learning_curve(self):
        pass

    def plot_feature_importance(self):
        pass


    def save_model(self):
        # save the sklearn model
        with open(f"{self.experiment_path}/model.pkl", "wb") as f:
            pickle.dump(self.model, f)
            f.close()

    def top_k_pred(self, X_test, y_test, X_train, y_train):
        print("we are here")
        # Uncalibrated model predictions
        y_pred_prob = self.predict_proba(X_test)
        print("Uncalibrated:", np.round(y_pred_prob, 3)[:10])

        # Calibrate the model
        # does data have null?
        print("null values in X_train:", X_train.isnull().sum().sum())
        print("null values in y_train:", y_train.isnull().sum().sum())
        # calibrated_model = CalibratedClassifierCV(
        #     self.model, method="isotonic", cv=10
        # )  # TODO: hyperparameter tuning
        #TODO: calibraition does not work with hierarchical
        # calibrated_model.fit(X_train, y_train)
        # y_pred_prob = calibrated_model.predict_proba(X_test)
        # print("Calibrated:", np.round(y_pred_prob, 3)[:10])

        # Predictions and actual classes
        y_pred = self.model.predict(X_test)
        print("Predicted classes:", y_pred[:10])
        print("Actual classes:", np.asarray(y_test)[:10])

        # Sort classes by probability
        sorted_indices = np.argsort(y_pred_prob, axis=1)[:, ::-1]
        print("Sorted indices:", sorted_indices[:10])
        print("Most probable class for each row:", sorted_indices[:, 0][:10])

        second_most_probable_class = sorted_indices[:, 1]
        third_most_probable_class = sorted_indices[:, 2]

        # Change the prediction to higher class if the difference in probability is less than 0.3
        count = 0
        # convert to numpy
        y_pred = np.array(y_pred)
        y_pred_prob = np.array(y_pred_prob)
        y_test = np.array(y_test)
        for i in range(len(y_pred)):
            true_class = y_test[i]
            first_pred = y_pred[i]
            first_pred_prob = y_pred_prob[i][first_pred]
            second_pred = second_most_probable_class[i]
            second_pred_prob = y_pred_prob[i][second_pred]
            third_pred = third_most_probable_class[i]
            third_pred_prob = y_pred_prob[i][third_pred]

            difference_in_prob = abs(first_pred_prob - second_pred_prob)

            if difference_in_prob < 0.3 and first_pred != max(first_pred, second_pred):
                y_pred[i] = second_pred
                count += 1

        print(f"Total number of changed probabilities: {count}")

        return y_pred
    
    def calculate_kappa(self, X_test, y_test):
        y_pred = self.predict_eval(X_test)
        kappa = cohen_kappa_score(y_test, y_pred)
        print(f"Cohen's Kappa: {kappa}")
        return kappa
