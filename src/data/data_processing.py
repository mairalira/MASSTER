"""
Initialize common imports for the data module.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from numpy.random import RandomState
import sys
import os
from typing import Tuple

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent
# Adding path to sys.path
sys.path.append(str(project_root))

# Import global variables setted on config
from config import *
from utils.data_handling import read_csv

class DataProcessor():
    def __init__(self):
        self.dataset_name = DATASET_NAME
        self.k_folds = K_FOLDS
        self.train_size = TRAIN_SIZE
        self.pool_size = POOL_SIZE
        self.data_path = DATA_PATH
        self.random_state = RANDOM_STATE
        self.data_dir = DATA_DIR

    def fetch_features(self) -> Tuple[pd.DataFrame, int]:
        logging.info(f'Fetching features for {self.dataset_name}...')
        df = read_csv(self.data_path)
        col_names = list(df.columns)
        target_fts_length = 0
        for name in col_names: 
            if 'target' in name:
                target_fts_length += 1
        logging.info(f'{self.dataset_name} has {target_fts_length} targets for regression')
        return df, target_fts_length
    
    def split_datasets(self, df: pd.DataFrame):
        logging.info(f'Splitting {self.dataset_name} in {self.k_folds} splits for cross-validation')
        for i in range(self.k_folds):
            rest = df.sample(frac=(self.train_size+self.pool_size), random_state=self.random_state)
            restcop = rest.copy()
            test = df.loc[~df.index.isin(rest.index)]

            train = rest.sample(frac=(self.train_size/(self.train_size+self.pool_size)), random_state=self.random_state)
            pool = rest = rest.loc[~rest.index.isin(train.index)]

            folder_dir = self.data_dir / 'processed' / f'{self.dataset_name}'
            if not os.path.exists(folder_dir):
                os.makedirs(folder_dir)

            datasets = {'train': train, 'pool': pool, 'test': test, 'train+pool': restcop}

            for dataset in datasets:
                dataset_path = folder_dir / f'{dataset}_{i+1}'
                datasets[dataset].to_csv(dataset_path, index=False)
            
        logging.info(f'Split ended for {self.dataset_name} and data has been saved at {folder_dir}')

        return
    
if __name__ == '__main__':
    data_module = DataProcessor()
    df, target_fts_length = data_module.fetch_features()
    data_module.split_datasets(df)