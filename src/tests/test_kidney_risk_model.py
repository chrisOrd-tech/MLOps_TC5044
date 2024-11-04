import pytest
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score

from src.KidneyRiskModel.module import KidneyRiskModel, UnsupportedClassifier

class TestModel():
    @pytest.fixture(scope="class")
    def model_setup(self):
        # Load the dataset from the `data/processed` folder
        X_train = pd.read_csv(Path('data/processed/train_X_ckd_dataset_v2.csv'))
        y_train = pd.read_csv(Path('data/processed/train_y_ckd_dataset_v2.csv'))

        X_test = pd.read_csv(Path('data/processed/test_X_ckd_dataset_v2.csv'))
        y_test = pd.read_csv(Path('data/processed/test_y_ckd_dataset_v2.csv'))
        
        estimator = 'log_reg'
        
        params = {
            'log_reg': {'C': 0.01, 'max_iter': 10, 'solver': 'lbfgs'},
            'svc_model': {'C': 0.9643857615941438, 'degree': 5, 'gamma': 0.1, 'kernel': 'rbf', 'tol': 0.01},
            'knn_model': {'n_neighbors': 7, 'algorithm': 'auto', 'leaf_size': 30, 'p': 2, 'weights': 'uniform'}
            }


        model = KidneyRiskModel(estimator_name=estimator,
                                X_train=X_train,
                                y_train=y_train['class'],
                                X_test=X_test,
                                y_test=y_test['class'],
                                params=params[estimator])
        return model
    
    def test_get_supported_estimators(self, model_setup):
        estimators = model_setup.get_supported_estimators()
        assert 'log_reg' in estimators
        assert 'svc_model' in estimators
        assert 'knn_model' in estimators

    def test_unsupported_estimator(self):
        with pytest.raises(UnsupportedClassifier):
            KidneyRiskModel(estimator_name='mlp_model', 
                            X_train=pd.DataFrame([[0, 1], [1, 0]]),
                            y_train=pd.Series([0, 1]),
                            X_test=pd.DataFrame([[0, 1]]),
                            y_test=pd.Series([0]),
                            params={"solver": "liblinear"}).train()
            
    def test_train_model(self, model_setup):
        trained_model = model_setup.train()
        y_hat = trained_model.clf.predict(model_setup.X_test)
        accuracy = accuracy_score(model_setup.y_test, y_hat)
        recall = recall_score(model_setup.y_test, y_hat, average='weighted')

        assert trained_model.clf is not None
        assert accuracy >= 0.8
        assert recall >= 0.8

    def test_save_and_load_model(self, model_setup):
        model_setup.train()
        model_setup.save_model('test_model.pkl')

        loaded_model = model_setup.load_model('test_model.pkl')

        instances = {
            'log_reg': LogisticRegression,
            'svc_model': SVC,
            'knn_model': KNeighborsClassifier
        }
        model_instance = instances[model_setup.estimator_name]

        assert isinstance(loaded_model, model_instance)


    def test_prediction_values(self, model_setup):

        trained_model = model_setup.train()
        y_hat = trained_model.clf.predict(model_setup.X_test)

        # Verify predictions are in the expected range of classes
        assert set(y_hat).issubset(set(model_setup.y_test.unique()))