# Chronic Kidney Disease Risk Prediction Code

This refactored Python code is designed for predicting risk factors of Chronic Kidney Disease (CKD) using a Logistic Regression model. The code uses a class-based structure to manage data exploration, preprocessing, model training, evaluation, and saving the trained model.

## Key Features

1. **Data Exploration**:  
   The `DataExplorer` class provides static methods to explore and visualize the dataset, including printing summary statistics, plotting histograms for numerical columns, and count plots for categorical columns. It also includes a correlation matrix for numerical values.
   
2. **Preprocessing and Modeling**:  
   The `KidneyRiskModel` class handles data preprocessing, model training, evaluation, and cross-validation. The preprocessing includes encoding categorical features and optional PCA for dimensionality reduction.

3. **Custom PCA Transformer**:  
   The `PCAVarianceThreshold` class is a custom transformer that selects the optimal number of components to reach a specified explained variance threshold. It can be included in the pipeline for feature reduction.

4. **Modular Pipeline**:  
   A scikit-learn pipeline is used for preprocessing, scaling, PCA (optional), and logistic regression. The code can easily be adapted to include different models or transformation steps.

## How It Works

### Main Components:

- **DataExplorer Class**:  
  Contains static methods for exploring and visualizing the dataset.

  - `explore_data`: Displays basic information like head, describe, info, and value counts.
  - `plot_histogram`: Plots histograms for numeric columns.
  - `count_plot`: Plots count plots for categorical columns.
  - `plot_correlation_matrix`: Plots the correlation matrix for numeric columns.

- **KidneyRiskModel Class**:  
  Encapsulates the full pipeline of loading data, preprocessing, training, evaluating, and saving the trained model.

  - `load_data`: Loads and explores the dataset using the `DataExplorer` class.
  - `preprocess_data`: Splits the data into training and test sets, and applies the necessary preprocessing.
  - `train`: Trains the model using the specified pipeline.
  - `evaluate`: Evaluates the model using a confusion matrix and classification report.
  - `cross_validation`: Performs cross-validation to assess model performance.
  - `save_model`: Saves the trained model as a `.pkl` file.

- **PCAVarianceThreshold Class**:  
  A custom transformer to perform PCA and select the minimum number of components to explain a given threshold of variance.