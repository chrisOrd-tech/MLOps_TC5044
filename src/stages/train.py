import argparse
import yaml
import joblib
import pandas as pd
from typing import Text

from src.model.model import train

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
    train_df = pd.read_csv(config['data_split']['train_path'])
    target_column = config['feature_engineering']['target_column']
    param_grid = config['train']['estimators'][estimator_name]['param_grid']
    cv = config['train']['cv']
    model_path = config['train']['estimators'][estimator_name]['model_path']

    model = train(df=train_df,
                  target_column=target_column,
                  estimator_name=estimator_name,
                  param_grid=param_grid,
                  cv=cv)
    
    joblib.dump(model, model_path)

    print('Training completed and model has been saved!')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)