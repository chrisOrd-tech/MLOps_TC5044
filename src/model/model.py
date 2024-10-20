import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, make_scorer
from typing import Dict, Text

class UnsupportedClassifier(Exception):
    '''
    Class to handle unsupported estimators, inherits from Exception
    '''
    def __init__(self, estimator_name: Text):
        self.msg = f'Unsupported estimator: {estimator_name}'
        super().__init__(estimator_name)

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
        'log_reg': LogisticRegression
    }


def train(df: pd.DataFrame, target_column: Text, estimator_name: Text, 
          param_grid: Dict, cv: int):
    '''
    Train model

    Args:
        df {pandas.DataFrame}: dataset
        target_column {Text}: target column name
        estimator_name {Text}: estimator name
        param_grid {Dict}: grid parameters
        cv {int}: cross-validation value

    Returns:
        trained model
    '''
    estimators = get_supported_estimators()

    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name=estimator_name)
    
    estimator = estimators[estimator_name]()
    recall_scorer = make_scorer(recall_score, average='weighted') # We use recall as metric to get the best model

    clf = GridSearchCV(estimator=estimator,
                       param_grid=param_grid,
                       cv=cv,
                       verbose=1,
                       scoring=recall_scorer)
    
    X = df.drop(columns=target_column, axis=1)
    y = df[target_column]

    clf.fit(X=X, y=y)

    return clf