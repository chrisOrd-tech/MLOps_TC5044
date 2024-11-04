import pytest
import pandas as pd
import numpy as np
import yaml
from src.DataExplorer.module import DataExplorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns

import matplotlib.pyplot as plt

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'numeric1': np.random.rand(100),
        'numeric2': np.random.rand(100),
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice(['yes', 'no'], 100)
    })
    return data

@pytest.fixture
def config_file(tmp_path):
    config = {
        'data': {
            'filepath': tmp_path / 'sample_data.csv',
            'header': 0,
            'skiprows': None
        }
    }
    config_path = tmp_path / 'params.yaml'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    return config_path

def test_load_data(config_file, sample_data):
    sample_data.to_csv(config_file.parent / 'sample_data.csv', index=False)
    df = DataExplorer.load_data(config_file)
    assert not df.empty

def test_explore_data(sample_data, capsys):
    DataExplorer.explore_data(sample_data)
    captured = capsys.readouterr()
    assert 'DESCRIBE' in captured.out
    assert 'INFO' in captured.out
    assert 'VALUE COUNTS' in captured.out

def test_plot_histogram(sample_data):
    DataExplorer.plot_histogram(sample_data)
    plt.close('all')

def test_count_plot(sample_data):
    DataExplorer.count_plot(sample_data)
    plt.close('all')

def test_plot_correlation_matrix(sample_data):
    DataExplorer.plot_correlation_matrix(sample_data)
    plt.close('all')

def test_transform_target_column(sample_data):
    transform_dict = {'yes': 1, 'no': 0}
    df = DataExplorer.transform_target_column(sample_data, 'target', transform_dict)
    assert df['target'].isin([0, 1]).all()

def test_standardize_numeric_fit_transform(sample_data):
    scaler = StandardScaler()
    explorer = DataExplorer(scaler=scaler)
    numeric_data = sample_data[['numeric1', 'numeric2']]
    transformed_data = explorer.standardize_numeric_fit_transform(numeric_data, scaler)
    assert transformed_data.shape == numeric_data.shape

def test_standardize_numeric_transform(sample_data):
    scaler = StandardScaler()
    explorer = DataExplorer(scaler=scaler)
    numeric_data = sample_data[['numeric1', 'numeric2']]
    explorer.standardize_numeric_fit_transform(numeric_data, scaler)
    transformed_data = explorer.standardize_numeric_transform(numeric_data)
    assert transformed_data.shape == numeric_data.shape

def test_categorical_encode_fit_transform(sample_data):
    encoder = OneHotEncoder(sparse=False)
    explorer = DataExplorer(encoder=encoder)
    transformed_data = explorer.categorical_encode_fit_transform(sample_data, encoder)
    assert transformed_data.shape[1] > sample_data.shape[1]

def test_categorical_encode_transform(sample_data):
    encoder = OneHotEncoder(sparse=False)
    explorer = DataExplorer(encoder=encoder)
    explorer.categorical_encode_fit_transform(sample_data, encoder)
    transformed_data = explorer.categorical_encode_transform(sample_data)
    assert transformed_data.shape[1] > sample_data.shape[1]