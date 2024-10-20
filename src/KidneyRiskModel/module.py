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
            'svc_model': SVC
        }
    
    def train(self) -> None:
        '''
        Train model

        Args:
            None

        Returns:
            None
        '''
        estimators = self.get_supported_estimators()
        if self.estimator_name not in estimators.keys():
            raise UnsupportedClassifier(estimator_name=self.estimator_name)
        
        estimator = estimators[self.estimator_name]

        clf = estimator(**self.params)
        print(type(self.y_train))

        clf.fit(self.X_train, self.y_train)
        
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
        model = self.load_model(model_path=model_path)

        y_hat = model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_hat)
        
        cm_plot = plot_confusion_matrix(cm=cm, target_names=target_names)
        cm_plot.savefig(cm_path)

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
            self.clf = pickle.load(f)

        return self