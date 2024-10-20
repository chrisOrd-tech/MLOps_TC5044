import argparse
import yaml
import numpy as np
import pandas as pd
from typing import Text

from src.DataExplorer.module import DataExplorer

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(config_path: Text):
    config = yaml.safe_load(open(config_path))
    clean_data = config['data']['clean_file']
    target_column = config['data']['target_column']
    target_dict = config['data']['target_dict']
    test_size = config['preprocess']['test_size']
    random_state = config['base']['random_seed']
    train_X_data = config['preprocess']['train_X_path']
    train_y_data = config['preprocess']['train_y_path']
    test_X_data = config['preprocess']['test_X_path']
    test_y_data = config['preprocess']['test_y_path']

    df = pd.read_csv(clean_data)
    data_df = DataExplorer.transform_target_column(df=df, target_column=target_column, transform_dict=target_dict)

    X = data_df.drop(columns=[target_column])
    y = data_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    data_explorer = DataExplorer(scaler=scaler, encoder=encoder)

    encoded_X_train = data_explorer.categorical_encode_fit_transform(train_df=X_train, encoder=encoder)
    encoded_X_test = data_explorer.categorical_encode_transform(test_df=X_test)

    scaled_X_train = data_explorer.standardize_numeric_fit_transform(train_df=encoded_X_train, scaler=scaler)
    scaled_X_test = data_explorer.standardize_numeric_transform(test_df=encoded_X_test)

    scaled_X_train.to_csv(train_X_data)
    y_train.to_csv(train_y_data)
    
    scaled_X_test.to_csv(test_X_data)
    y_test.to_csv(test_y_data)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    preprocess_data(args.config)
