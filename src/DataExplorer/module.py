import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import yaml
from typing import Text, List, Dict


class DataExplorer:
    def __init__(self, scaler = None, encoder = None) -> None:
        self.scaler = scaler
        self.encoder = encoder

    @staticmethod
    def load_data(config_path: Text):
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
        
        print('Data load completed!')

        df = pd.read_csv(raw_data, header=header, skiprows=skiprows)
        
        return df

    @staticmethod
    def explore_data(data: pd.DataFrame) -> None:
        '''
        Print general information of the DataFrame, first 5 rows, describe and info data, count for each column.

        Args:
            data: DataFrame to explore

        Returns:
            None 
        '''
        print(tabulate(data.head(), headers='keys', tablefmt='pretty'))
        print(f'\nDESCRIBE')
        print(tabulate(data.describe(include='all').T, headers='keys', tablefmt='pretty'))
        print(f'\nINFO')
        print(data.info())
        print(f'\nVALUE COUNTS')
        print(data.value_counts())

    @staticmethod
    def plot_histogram(data: pd.DataFrame, bins: int = 10) -> None:
        '''
        Plots the histogram for the numeric columns only.

        Args:
            data: DataFrame to plot

        Returns:
            None
        '''
        quantitative_columns = data.select_dtypes(include=np.number).columns
        ig, axes = plt.subplots(4, 4, figsize=(15,15)) # Creates subplots to show all the columns in a 4x4 matrix
        axes = axes.ravel()

        for column, ax in zip(quantitative_columns, axes):
            ax.set(title=f'{column.upper()} HISTOGRAM', xlabel=None)
            ax.tick_params(axis='y', labelsize=9)
            plot = sns.histplot(x=data[column], ax=ax, kde=True)
            ax.bar_label(ax.containers[0], fontsize=10)

            ax.yaxis.grid(True) # Hide the horizontal gridlines
            ax.xaxis.grid(False) # Show the vertical gridlines

        for i,ax in enumerate(axes): # Removes empty plot in subplots
            if(not ax.get_title()):
                ax.remove()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def count_plot(data: pd.DataFrame) -> None:
        '''
        Plots the count plot for the categorical columns only.

        Args:
            data: DataFrame to plot

        Returns:
            None
        '''
        categorical_columns = data.select_dtypes(include=object).columns
        ig, axes = plt.subplots(4, 4, figsize=(15,15)) # Creates subplots to show all the columns in a 4x4 matrix
        axes = axes.ravel()

        for column, ax in zip(categorical_columns, axes):
            ax.set(title=f'{column.upper()} COUNTPLOT', xlabel=None)
            ax.tick_params(axis='y', labelsize=9)
            ax.tick_params(axis='x', rotation=90)
            plot = sns.countplot(x=data[column], ax=ax)
            ax.bar_label(ax.containers[0], fontsize=10)

            ax.yaxis.grid(True) # Hide the horizontal gridlines
            ax.xaxis.grid(False) # Show the vertical gridlines

        for i,ax in enumerate(axes): # Removes empty plot in subplots
            if(not ax.get_title()):
                ax.remove()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame) -> None:
        '''
        Plots the correlation matrix, this only applies to numeric values.

        Args:
            data: DataFrame to plot

        Returns:
            None
        '''
        quantitative_columns = data.select_dtypes(include=np.number).columns
        plt.figure(figsize=(12,8))
        sns.heatmap(data[quantitative_columns].corr(), annot=True, fmt='.2f')
        plt.show()

    @staticmethod
    def transform_target_column(df: pd.DataFrame, target_column: Text, transform_dict: Dict) -> pd.DataFrame:
        '''
        Transform the target column to the given values in transform_dict

        Args:
            df: DataFrame
            target_column: column to be transformed
            transform_dict: dictionary with the desired values in the target column

        Returns:
            df: transformed DataFrame
        '''
        pd.set_option('future.no_silent_downcasting', True)
        df[target_column] = df[target_column].replace(transform_dict).infer_objects(copy=False)

        return df

    def standardize_numeric_fit_transform(self, train_df: pd.DataFrame, scaler: any) -> pd.DataFrame:
        '''
        Standardize with fit_transform method given the scaler

        Args:
            train_df: train DataFrame
            scaler: scaler to be used

        Returns:
            train_df: DataFrame with numeric columns transformed
        '''
        train_df = pd.DataFrame(self.scaler.fit_transform(train_df), columns=train_df.columns)

        return train_df
    
    def standardize_numeric_transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Standardize with fit given the scaler

        Args:
            test: test DataFrame

        Returns:
            test_df: DataFrame with numeric columns transformed
        '''
        test_df = pd.DataFrame(self.scaler.transform(test_df), columns=test_df.columns)

        return test_df

    def categorical_encode_fit_transform(self, train_df: pd.DataFrame, encoder: any) -> pd.DataFrame:
        '''
        Transform categorical columns from the given DataFrame

        Args:
            df: DataFrame to transform
            encoder: Categorical  encoder

        Returns:
            df: DataFrame with categorical columns transformed
        '''
        cat_columns = train_df.select_dtypes(include=['object']).columns
        self.cat_columns = cat_columns
        train_df[cat_columns] = encoder.fit_transform(train_df[cat_columns])

        return train_df
    
    def categorical_encode_transform(self, test_df: pd.DataFrame) -> pd.DataFrame:
        '''
        Transform categorical columns from the given DataFrame

        Args:
            df: DataFrame to transform
            encoder: Categorical  encoder

        Returns:
            df: DataFrame with categorical columns transformed
        '''
        test_df[self.cat_columns] = self.encoder.transform(test_df[self.cat_columns])

        return test_df