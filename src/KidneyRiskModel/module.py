import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from typing import Text, Dict

from src.report.visualize import plot_confusion_matrix, print_classification_report

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, make_scorer, ConfusionMatrixDisplay, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
class UnsupportedClassifier(Exception):
    '''
    Class to handle unsupported estimators, inherits from Exception
    '''
    def __init__(self, estimator_name: Text):
        self.msg = f'Unsupported estimator: {estimator_name}'
        super().__init__(estimator_name)

class KidneyRiskModel:
    def __init__(self, estimator_name: Text, X_train: pd.DataFrame, y_train: pd.DataFrame,
                 X_test: pd.DataFrame, y_test: pd.DataFrame, params: Dict) -> None:
        self.estimator_name = estimator_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = params

        mlflow.set_tracking_uri('http://localhost:5000')

    @staticmethod
    def get_supported_estimators() -> Dict:
        '''
        Returns a dictionary with supported estimators

        Args:
            None

        Returns:
            Dict: supported estimators
        '''
        # ToDO Add more estimators in the dictionary
        return {
            'log_reg': LogisticRegression,
            'svc_model': SVC,
            'knn_model': KNeighborsClassifier
        }
    
    def train(self) -> None:
        '''
        Train model

        Args:
            None

        Returns:
            None
        '''
        mlflow.set_experiment(f"ckd-experiment_{self.estimator_name}_train")

        estimators = self.get_supported_estimators()
        if self.estimator_name not in estimators.keys():
            raise UnsupportedClassifier(estimator_name=self.estimator_name)
        
        estimator = estimators[self.estimator_name]

        clf = estimator(**self.params)
        print(type(self.y_train))

        with mlflow.start_run(run_name=f'{self.estimator_name}_train'):
            clf.fit(self.X_train, self.y_train)
            mlflow.log_params(self.params)
            # Calculate metrics
            y_hat = clf.predict(self.X_train)
            acc = accuracy_score(self.y_train, y_hat)
            # prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(self.y_train, y_hat, average='weighted')
            mlflow.log_metrics({"accuracy": acc, "recall": rec})
            mlflow.sklearn.log_model(clf, artifact_path="models")
        
        self.clf = clf
        return self
    
    def evaluate(self, model_path: Text, target_names: Text, cm_path: Text) -> None:
        '''
        Evaluate model

        Args:
            model_path: path to the model to evaluate

        Returns:
            None
        '''
        mlflow.set_experiment(f"ckd-experiment_{self.estimator_name}_evaluate")

        model_ = self.load_model(model_path=model_path)

        with mlflow.start_run(run_name=f'{self.estimator_name}_test'):
            y_hat = model_.predict(self.X_test)
            mlflow.log_params(self.params)
            mlflow.log_metrics({"accuracy": accuracy_score(self.y_test, y_hat), 
                                "recall": recall_score(self.y_test, y_hat, average='binary')})
            mlflow.sklearn.log_model(model_, artifact_path="models")

            cm = confusion_matrix(self.y_test, y_hat)
            
            cm_plot = plot_confusion_matrix(cm=cm, target_names=target_names)
            cm_plot.savefig(cm_path)
            mlflow.log_figure(cm_plot, cm_path)

            print_classification_report(y_true=self.y_test, y_hat=y_hat)
    
    def cross_validation(self, cv: int = 5) -> None:
        scores = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=cv)
        print(f'Average Accuracy with Cross Valdiation: {np.mean(scores) * 100:.2f}%')
        return self
    
    def save_model(self, model_path: Text) -> None:
        '''
        Saves the mode as a pkl file

        Args:
            filename: string with the name of the pkl file

        Returns:
            None
        '''
        with open(model_path, 'wb') as f:
            pickle.dump(self.clf, f)
        return self
    
    def load_model(self, model_path: Text) -> None:
        '''
        Load saved model, given the path

        Args:
            model_path: path to the model

        Returns:
            None
        '''
        with open(model_path, 'rb') as f:
            clf_loaded = pickle.load(f)

        return clf_loaded