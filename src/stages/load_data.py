import argparse
import yaml
import pandas as pd
from typing import Text

def load_data(config_path: Text) -> pd.DataFrame:
        '''
        Load raw data.

        Args:
             config_path {Text}: path to params.yaml

        Returns:
                DataFrame with raw data
        '''
        config = yaml.safe_load(open(config_path))
        raw_data = config['data']['filepath']
        header = config['data']['header']
        skiprows = config['data']['skiprows']
        
        print('Data loaded complete!')

        return pd.read_csv(raw_data, header=header, skiprows=skiprows)

if __name__ == '__main__':
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument('--config', dest='config', required=True)
        args = args_parser.parse_args()

        load_data(config_path=args.config)