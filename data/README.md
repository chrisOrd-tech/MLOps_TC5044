# Data

Here goes the data for the project, such as training datasets. When using DVC, it should point at this folder.

# Data Folder

This folder contains the data files required for the project, including raw datasets and processed datasets used for training and evaluation. If using DVC (Data Version Control), this folder should be used to track and version the datasets.

## Folder Structure

The data files are organized as follows:

### Raw Data

The raw dataset is located in the `raw` folder:

- `ckd-dataset-v2.csv`: The original dataset containing patient data for Chronic Kidney Disease (CKD) analysis.

### Processed Data

The processed datasets are stored in the `processed` folder, split into training and testing sets:

- `train_X`: Features for the training set.
- `train_y`: Labels for the training set.
- `test_X`: Features for the test set.
- `test_y`: Labels for the test set.