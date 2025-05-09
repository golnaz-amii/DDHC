
# How to run the Carson repo

1. Make an experiment definition in `experiments\v2\experiment_name\config.json`
2. Make sure the description in `ExpeirmentModel.py` is set to the rigth value
3. Make a virtual environment, and install the requirements, and activate it
4. Run `python src/main.py -e experiment_name`
5. Results will be saved in `experiment_name`: a classification report, all the plots (confusion matrix, ROC curves, precison-recall curve, learning curve, feature importance), the trained model, 



# Amii Base Template

This repository is a template repo that should provide some minimal scripts, organization, and structure to kick-start a python based project.
To use this template, press the big green "Use Template" button in the top right corner of this repository's main page---do not fork this repository.

While this repo will encode some useful procedures for interacting with Amii's resources, make sure to check out the knowledge base for more detailed and up-to-date information!
[awesome-amii](https://github.com/Amii-Industry-Collaboration/awesome-amii)


## First steps

1. Modify `.github/CODEOWNERS` and add the names of your teammates and yourself. This will ensure all members of the project are notified for code review.
1. Create a branch named `release` if one does not already exist. If you are unsure how, see below for details.
1. Decide on what code style is appropriate for the project (e.g. linter rules, type-checking rules, etc.). Edit the corresponding rules in `pyproject.toml` and modify `.github/workflows/style.yml` as needed. When in doubt, prefer to keep the default settings unchanged.
1. Delete the contents of this readme and replace with a description of your project and details of how to run the relevant code.


### Creating a release branch
```bash
# check if "release" already exists
git branch
# if not
git checkout -b release
git push --set-upstream origin release
```

## Contributing to this repo
This repo is open to contributions!
If there are useful scripts that you think most new projects will want to use, changes to the default style guide, missing information in the readme, or other changes you would like to see in this template, please feel free to open a pull request.
