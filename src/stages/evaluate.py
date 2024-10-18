import argparse
import yaml
import pandas as pd
from typing import Dict, Text
import os

from src.KidneyRiskModel.module import KidneyRiskModel

def evaluate_model(config_path: Text) -> None:
    '''
    Evaluate the saved model.

    Args:
        config_path {Text}: path to params.yaml

    Returns:
        None
    '''
    config = yaml.safe_load(open(config_path))
    estimator_name = config['train']['estimator_name']
    model_path = config['train']['estimators'][estimator_name]['model_path']
    X_test_path = config['preprocess']['test_X_path']
    y_test_path = config['preprocess']['test_y_path']
    reports_path = config['evaluate']['reports_dir']
    target_names = config['evaluate']['target_names']
    cm_img = config['evaluate']['estimators'][estimator_name]['confusion_matrix_img']

    cm_path = os.path.join(reports_path, cm_img)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    model = KidneyRiskModel(estimator_name=estimator_name, X_train=None, y_train=None, X_test=X_test, y_test=y_test, params={})

    model.evaluate(model_path=model_path, target_names=target_names, cm_path=cm_path)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)