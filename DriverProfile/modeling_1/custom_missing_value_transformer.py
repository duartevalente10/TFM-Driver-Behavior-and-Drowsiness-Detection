"""
modeling.custom_missing_value_transformer
----------------------------

This module provides a custom transformer that fills missing values for a pipeline
"""

# packages
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
# local
from pre_process_1 import (
    read_csv_file,
    delete_missing_values, 
    check_columns_with_nulls, 
    fill_missing_values
)


class CustomMissingValueTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        deleted_df = delete_missing_values(X)
        columns_nan = check_columns_with_nulls(deleted_df)
        return fill_missing_values(X, columns_nan)

