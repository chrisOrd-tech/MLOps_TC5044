import argparse
import yaml
import pandas as pd
from typing import Text

from sklearn.preprocessing import OrdinalEncoder

def feature_engineering(config_path: Text) -> None:
    '''
    Preprocess the dataset and save the preprocessed data in a csv file.

    Args:
        config_path {Text}: path to params.yaml

    Returns:
        None
    '''
    config = yaml.safe_load(open(config_path))
    cat_columns = config['feature_engineering']['categorical_columns']
    features_path = config['feature_engineering']['features_path']
    target_column = config['feature_engineering']['target_column']
    raw_data = config['data']['filepath']
    header = config['data']['header']
    skiprows = config['data']['skiprows']

    data = pd.read_csv(raw_data, header=header, skiprows=skiprows)
    
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    data[cat_columns] = ordinal_encoder.fit_transform(data[cat_columns])

    pd.set_option('future.no_silent_downcasting', True)
    data[target_column] = data[target_column].replace({'ckd':1, 'notckd':0}).infer_objects(copy=False)

    data.to_csv(features_path, index=False)

    print('Data features saved!')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    feature_engineering(args.config)