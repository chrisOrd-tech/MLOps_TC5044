import pytest
import pandas as pd
import numpy as np
from src.model.model import train, UnsupportedClassifier



@pytest.fixture
def sample_data():
    df = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    df['target'] = np.random.randint(0, 2, size=100)
    param_grid = {'C': [0.1, 1.0, 10.0], 'max_iter': [100, 200]}
    return df, 'target', param_grid

def test_train_log_reg(sample_data):
    df, target_column, param_grid = sample_data
    model = train(df, target_column, 'log_reg', param_grid, cv=3)
    assert model is not None
    assert hasattr(model, 'best_estimator_')

def test_train_unsupported_classifier(sample_data):
    df, target_column, param_grid = sample_data
    with pytest.raises(UnsupportedClassifier):
        train(df, target_column, 'unsupported_estimator', param_grid, cv=3)