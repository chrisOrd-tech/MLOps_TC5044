import argparse
import yaml
import joblib
import pandas as pd
from typing import Text

from src.KidneyRiskModel.module import KidneyRiskModel

def train_model(config_path: Text) -> None:
    '''
    Train the model.

    Args:
        config_path {Text}: path to params.yaml

    Returns:
        None
    '''
    config = yaml.safe_load(open(config_path))
    estimator_name = config['train']['estimator_name']
    X_train_df = pd.read_csv(config['preprocess']['train_X_path'])
    y_train_df = pd.read_csv(config['preprocess']['train_y_path'])
    params = config['train']['estimators'][estimator_name]['params']
    model_path = config['train']['estimators'][estimator_name]['model_path']

    model = KidneyRiskModel(estimator_name=estimator_name, X_train=X_train_df, y_train=y_train_df['class'], X_test=None, y_test=None, params=params)

    model.train()
    
    model.save_model(model_path=model_path)

    print('Training completed and model has been saved!')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)