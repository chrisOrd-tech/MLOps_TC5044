import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import DataExplorer

class KidneyRiskModel:
    def __init__(self, filepath: str, pipeline: Pipeline) -> None:
        self.filepath = filepath
        self.pipeline = pipeline
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self) -> None:
        self.data = pd.read_csv(self.filepath, header=0, skiprows=[1, 2])
        DataExplorer.explore_data(self.data)
        DataExplorer.plot_histogram(self.data)
        DataExplorer.count_plot(self.data)
        DataExplorer.plot_correlation_matrix(self.data)
        return self
    
    def preprocess_data(self, label_column: str) -> None:
        X = self.data.drop(columns=label_column)
        y = self.data[label_column].replace({'ckd':1, 'notckd':0}).infer_objects(copy=False)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        plt.subplot(1,3,1)
        self.y_train.value_counts().plot.pie(y=label_column, title='Proportion of each class for TRAIN set', figsize=(10,6))

        plt.subplot(1,3,3)
        self.y_test.value_counts().plot.pie(y=label_column, title='Proportion of each class for TEST set', figsize=(10,6))

        plt.tight_layout()
        plt.show()

        return self
    
    def train(self) -> None:
        self.pipeline.fit(self.X_train, self.y_train)
        return self
    
    def evaluate(self) -> None:
        y_hat = self.pipeline.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_hat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.y_test))
        disp.plot()
        plt.show()

        report = classification_report(self.y_test, y_hat)
        print('Classification report:')
        print(report)
        return self
    
    def cross_validation(self, cv: int = 5) -> None:
        scores = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=cv)
        print(f'Average Accuracy with Cross Valdiation: {np.mean(scores) * 100:.2f}%')
        return self
    
    def save_model(self, filename: str) -> None:
        '''
        Saves the mode as a pkl file

        Args:
            filename: string with the name of the pkl file

        Returns:
            None
        '''
        file_path = os.path.join('/home/chrisorduna/Repositories/MNA/MLOps_TC5044/models', filename)
        with open(file_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
        return self
    
    def load_model(self, filename: str) -> None:
        file_path = os.path.join('/home/chrisorduna/Repositories/MNA/MLOps_TC5044/models', filename)
        with open(file_path, 'rb') as f:
            self.pipeline = pickle.load(f)

        return self