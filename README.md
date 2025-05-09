
# How to run the DDHC repo

1. Make an experiment definition in `experiments\v5\experiment_name\config.json`
2. Make sure the description in `ExpeirmentModel.py` is set to the rigth value
3. Make a virtual environment, and install the requirements, and activate it
4. Run `python src/main.py -e experiment_name`
5. Results will be saved in `experiment_name`: a classification report, all the plots (confusion matrix, ROC curves, precison-recall curve, learning curve, feature importance), the trained model, 
