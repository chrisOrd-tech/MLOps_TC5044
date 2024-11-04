import pytest
import pandas as pd
import numpy as np
from src.KidneyRiskModel.module import KidneyRiskModel, UnsupportedClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os
import pickle

# src/KidneyRiskModel/test_module.py


@pytest.fixture
def sample_data():
    X_train = pd.DataFrame(np.random.rand(100, 5))
    y_train = pd.Series(np.random.randint(0, 2, size=100))
    X_test = pd.DataFrame(np.random.rand(20, 5))
    y_test = pd.Series(np.random.randint(0, 2, size=20))
    params = {'C': 1.0, 'max_iter': 100}
    return X_train, y_train, X_test, y_test, params

def test_get_supported_estimators():
    estimators = KidneyRiskModel.get_supported_estimators()
    assert 'log_reg' in estimators
    assert 'svc_model' in estimators
    assert 'knn_model' in estimators
    assert estimators['log_reg'] == LogisticRegression
    assert estimators['svc_model'] == SVC
    assert estimators['knn_model'] == KNeighborsClassifier

def test_train(sample_data):
    X_train, y_train, X_test, y_test, params = sample_data
    model = KidneyRiskModel('log_reg', X_train, y_train, X_test, y_test, params)
    model.train()
    assert hasattr(model, 'clf')
    assert isinstance(model.clf, LogisticRegression)

def test_evaluate(sample_data, tmp_path):
    X_train, y_train, X_test, y_test, params = sample_data
    model = KidneyRiskModel('log_reg', X_train, y_train, X_test, y_test, params)
    model.train()
    model_path = tmp_path / "model.pkl"
    model.save_model(model_path)
    cm_path = tmp_path / "confusion_matrix.png"
    model.evaluate(model_path, target_names=['class 0', 'class 1'], cm_path=cm_path)
    assert os.path.exists(cm_path)

def test_cross_validation(sample_data):
    X_train, y_train, X_test, y_test, params = sample_data
    model = KidneyRiskModel('log_reg', X_train, y_train, X_test, y_test, params)
    model.train()
    model.cross_validation(cv=3)

def test_save_model(sample_data, tmp_path):
    X_train, y_train, X_test, y_test, params = sample_data
    model = KidneyRiskModel('log_reg', X_train, y_train, X_test, y_test, params)
    model.train()
    model_path = tmp_path / "model.pkl"
    model.save_model(model_path)
    assert os.path.exists(model_path)

def test_load_model(sample_data, tmp_path):
    X_train, y_train, X_test, y_test, params = sample_data
    model = KidneyRiskModel('log_reg', X_train, y_train, X_test, y_test, params)
    model.train()
    model_path = tmp_path / "model.pkl"
    model.save_model(model_path)
    loaded_model = model.load_model(model_path)
    assert isinstance(loaded_model, LogisticRegression)