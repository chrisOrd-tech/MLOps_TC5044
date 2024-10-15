import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class DataExplorer:
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