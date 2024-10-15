import argparse
import yaml
import joblib
import json
import pandas as pd
from typing import Dict, Text
import os

from sklearn.metrics import confusion_matrix, recall_score
from src.report.visualize import plot_confusion_matrix, print_classification_report

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
    test_path = config['data_split']['test_path']
    target_column = config['feature_engineering']['target_column']
    reports_path = config['evaluate']['reports_dir']
    target_names = config['evaluate']['target_names']
    metrics_file = config['evaluate']['estimators'][estimator_name]['metrics_file']
    cm_img = config['evaluate']['estimators'][estimator_name]['confusion_matrix_img']

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    predictions = model.predict(X_test)
    recall = recall_score(y_true=y_test, y_pred=predictions, average='macro')

    cm = confusion_matrix(predictions, y_test)

    report = {
        'recall': recall,
        'cm': cm,
        'actual': y_test,
        'predicted': predictions
    }

    metrics_path = os.path.join(reports_path, metrics_file)
    cm_path = os.path.join(reports_path, cm_img)

    json.dump(
        obj={'recall': report['recall']},
        fp=open(metrics_path, 'w')
        )
    
    plt = plot_confusion_matrix(cm=cm,
                                target_names=target_names)
    
    print_classification_report(y_true=y_test, y_hat=predictions)
    
    plt.savefig(cm_path)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)