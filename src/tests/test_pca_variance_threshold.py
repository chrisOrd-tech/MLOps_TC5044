import pytest
import pandas as pd
import numpy as np
from src.PCAVarianceThreshold.module import PCAVarianceThreshold

# src/PCAVarianceThreshold/test_module.py


@pytest.fixture
def sample_data():
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, size=100))
    return X, y

def test_init_default():
    pca_vt = PCAVarianceThreshold()
    assert pca_vt.variance_threshold == 0.9
    assert pca_vt.pca is None

def test_init_custom():
    pca_vt = PCAVarianceThreshold(variance_threshold=0.95)
    assert pca_vt.variance_threshold == 0.95
    assert pca_vt.pca is None