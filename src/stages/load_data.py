import argparse
import yaml
import pandas as pd
from typing import Text
from src.DataExplorer.module import DataExplorer

def load_data(config_path: Text) -> pd.DataFrame:
        '''
        Load raw data.

        Args:
                config_path {Text}: path to params.yaml
        '''
        config = yaml.safe_load(open(config_path))
        out_file = config['data']['clean_file']
        target_column = config['data']['target_column']
        target_dict = config['data']['target_dict']

        df = DataExplorer.load_data(config_path=config_path)
        numeric_target_df = DataExplorer.transform_target_column(df=df, target_column=target_column, transform_dict=target_dict)

        numeric_target_df.to_csv(out_file)


if __name__ == '__main__':
        args_parser = argparse.ArgumentParser()
        args_parser.add_argument('--config', dest='config', required=True)
        args = args_parser.parse_args()

        load_data(config_path=args.config)