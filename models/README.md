# Models

Here goes the trained models. DVC should use this models in versioning.

# Models Folder

This folder contains the trained machine learning models used in the project. The models have been versioned and tracked using DVC (Data Version Control) to ensure consistency and reproducibility across different environments.

## Model Files

The following models are stored in this folder:

- **`pca.pkl`**: A Principal Components Analysis (PCA) model used for feature enhancement and dimensionality reduction.
  
- **`log_reg`**: A Logistic Regression model, fine-tuned with optimal hyperparameters:
  - `C`: 0.001
  - `max_iter`: 10
  - `solver`: `'lbfgs'`

- **`svc_model`**: A Support Vector Machine (SVM) Classifier, trained with the best hyperparameters:
  - `C`: 0.9643857615941438
  - `degree`: 3
  - `gamma`: `'0.1'`
  - `kernel`: `'rbf'`
  - `tol`: 0.01