import argparse
import yaml
import pandas as pd
from typing import Text

from sklearn.model_selection import train_test_split

def data_split(config_path: Text) -> None:
    '''
    Split dataset into train and test datasets and save each in a csv file.

    Args:
        config_path {Text}: path to params.yaml

    Returns:
        None
    '''
    config = yaml.safe_load(open(config_path))
    data = pd.read_csv(config['feature_engineering']['features_path'])
    test_size = config['data_split']['test_size']
    random_state = config['base']['random_seed']
    train_path = config['data_split']['train_path']
    test_path = config['data_split']['test_path']

    train_dataset, test_dataset = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )

    train_dataset.to_csv(train_path)
    test_dataset.to_csv(test_path)

    print('Split data completed!')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)