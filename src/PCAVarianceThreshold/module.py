import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

class PCAVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold: float = 0.9) -> None:
        '''
        Initialize a custom PCA class

        Args:
            variance_threshold: percentage of the data to be explained by the PCA
        
        Returns:
            None
        '''
        self.variance_threshold = variance_threshold
        self.pca = None
    
    def fit(self, X: pd.DataFrame, y:pd.Series) -> None:
        '''
        Fits PCA and get the cumulative explained variance ratio and compare with the threshold to get the min components to reach the threshold

        Args:
            X: DataFrame with the inputs
            y: pandas Series with the target

        Returns:
            None
        '''
        self.pca = PCA().fit(X)
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        num_components = np.argmax(cumulative_variance >= self.variance_threshold)
        self.pca = PCA(n_components=num_components)
        self.pca.fit(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        '''
        PCA transform method

        Args:
            X: DataFrame with the inputs

        Returns:
            None
        '''
        return self.pca.transform(X)