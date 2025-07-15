"""
preprocess.categorical_data
-------

This module provides two aproaches to handle categorical data:
    1 - Label Encoding
    2 - One-Hot Encoding

We apply One-Hot Encoding when:
    - The categorical feature is not ordinal (like the countries above).
    - The number of categorical features is less so one-hot encoding can be
      effectively applied.

We apply Label Encoding when:
    - The categorical feature is ordinal.
    - The number of categories is quite large as one-hot encoding can lead to
      high memory consumption.
"""

# packages
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
# local
from .b_construct_dataset import read_csv_file, store_csv


def label_enconding(trips):
    """
    Apply label encondig:
        - Assign each different light_mode value an int value
            - day = 0 | dusk = 1 | night = 2

    Args:
        trips (pandas.DataFrame): Dataset

    Returns:
        pandas.DataFrame: Dataset updated
    """
    df = trips.copy()
    le = OrdinalEncoder()
    df[['light_mode']] = le.fit_transform(df[['light_mode']])
    return df


def one_hot_encoding(trips):
    """
    Apply one hot encondig:
        - Create a new column for each different light_mode value
          with binary values
            - light_mode_day | light_mode_dusk | light_mode_night

    Args:
        trips (pandas.DataFrame): Dataset

    Returns:
        pandas.DataFrame: Dataset updated
    """
    df = trips.copy()
    one_hot = pd.get_dummies(df['light_mode'], prefix='light_mode')
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop(['light_mode'], axis=1)
    return df


if __name__ == "__main__":

    # read dataset
    trips = read_csv_file('../datasets/missing_values/trips_mv_test')

    df_label_encoding = label_enconding(trips)

    # store_csv(
    #     '../datasets/categorical_data', 'trips_label_encoding_test',
    #     df_label_encoding
    # )

    # df_one_hot_encoding = one_hot_encoding(trips)
    # store_csv(
    #     '../datasets/categorical_data', 'trips_one_hot_encoding',
    #     df_one_hot_encoding
    # )
