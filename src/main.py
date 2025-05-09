from ExperimentModel import ExperimentModel

if __name__ == "__main__":

    ExperimentModel = ExperimentModel()
    model = ExperimentModel.model
    preprocessor = ExperimentModel.preprocessor

    # Preprocess the data
    X_train, X_eval, X_test, y_train, y_eval, y_test = preprocessor.preprocess_data()

    model.set_data(X_train, y_train, X_eval, y_eval, X_test, y_test)

    #model.hyperparameter_tuning(X_train, y_train)

    # Train the model
    print("are we here?")
    model.train(X_train, y_train, X_eval, y_eval)

    # Plot the learning curves
    model.plot_learning_curve()

    # Evaluate the model
    model.evaluate(X_test, y_test)

    # Show the relevant information
    model.plot_feature_importance()
   # model.plot_roc_auc_curves(X_test, y_test)
    model.plot_precision_recall_curve(X_test, y_test)
   # model.balanced_accuracy_vs_threshold(X_test, y_test) #TODO: what does it mean on train vs test?
    model.calculate_kappa(X_test, y_test)

    # Save the model
    model.save_model()
