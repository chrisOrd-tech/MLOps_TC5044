
import argparse
import yaml
import joblib
import pandas as pd
from typing import Text
import os
from KidneyRiskModel.module import KidneyRiskModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from KidneyRiskModel.module import KidneyRiskModel
from PCAVarianceThreshold.module import PCAVarianceThreshold
import warnings
   
def main(dataset_path: str, pca: bool = True) -> None:
    categorical_columns = ['sg','al','su','bgr','bu','sod','sc','pot','hemo','pcv','rbcc','wbcc']

    cat_pipeline = Pipeline(steps=[('ordinalEncoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

    preprocessing = ColumnTransformer([
        ('categorical', cat_pipeline, categorical_columns)
    ])

    if pca:
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('standarization', StandardScaler()),
            ('pca', PCAVarianceThreshold(variance_threshold=0.90)),
            ('logisticRegression', LogisticRegression(max_iter=3, random_state=42))
        ])
    else:
        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('standarization', StandardScaler()),
            ('logisticRegression', LogisticRegression(max_iter=10, random_state=41))
        ])

    model = KidneyRiskModel(filepath=dataset_path, pipeline=pipeline)
    (model.load_data()
        .preprocess_data(label_column='class')
        .train()
        .evaluate()
        .cross_validation(cv=5)
        .save_model('lr_model.pkl'))
    
if __name__ == '__main__':
    main(dataset_path='/Users/javi/Code/MLOps_TC5044/data/processed/ckd-dataset-v2.csv')
# model = KidneyRiskModel(filepath='', pipeline=pipeline)
# (model.load_data()
#     .preprocess_data(label_column='class')
#     .train()
#     .evaluate()
#     .cross_validation(cv=5)
#     .save_model('lr_model.pkl'))
