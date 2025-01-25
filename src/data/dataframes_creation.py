import pandas as pd
import numpy as np
from pathlib import Path

def data_read(data_dir, dataset_name, dataset):
    # Dataset path
    folder_dir = data_dir / 'processed' / f'{dataset_name}'
    data_path = folder_dir / f'{dataset}'
    df = pd.read_csv(data_path)

    # Identify input and target columns
    col_names = list(df.columns)
    target_names = [col for col in col_names if 'target' in col]
    feature_names = [col for col in col_names if col not in target_names]

    # Separate inputs from targets in different DataFrames
    inputs = df[feature_names]
    targets = df[target_names]

    # Number of instances and target length
    n_instances = len(targets)
    target_length = len(target_names)

    return inputs, targets, n_instances, target_length, target_names, feature_names

def read_data(data_dir, dataset_name, iteration):
    X_train, y_train, _, target_length, target_names, feature_names = data_read(data_dir, dataset_name, f'train_{iteration}')
    X_pool, y_pool, n_pool, target_length, target_names, feature_names = data_read(data_dir, dataset_name, f'pool_{iteration}')
    X_rest, y_rest, _, target_length, target_names, feature_names = data_read(data_dir, dataset_name, f'train+pool_{iteration}')
    X_test, y_test, _, target_length, target_names, feature_names = data_read(data_dir, dataset_name, f'test_{iteration}')
    y_pool_nan = pd.DataFrame(np.nan, index=y_pool.index, columns=y_pool.columns)

    X_pool.index = pd.RangeIndex(start=len(X_train), stop=len(X_train) + len(X_pool), step=1)
    y_pool_nan.index = pd.RangeIndex(start=len(y_train), stop=len(y_train) + len(y_pool), step=1)

    return X_train, y_train, X_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names
