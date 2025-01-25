import pandas as pd
import sys
from pathlib import Path
from src.data.dataframes_creation import data_read

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
# Adding path to sys.path
sys.path.append(str(project_root))

from config import *

data_dir = DATA_DIR

class ActiveLearning:
    n_trees = N_TREES
    random_state = RANDOM_STATE
    iterations = ITERATIONS

    def __init__(self):
        self.dataset_name = DATASET_NAME

    def read_data(self, iteration):
        X_train, y_train, X_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names = data_read(data_dir, self.dataset_name, iteration)
        return X_train, y_train, X_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names
    
    def target_collect(self, targets, target_length):
        # Collect all the values for specific targets in separate lists
        targets_collected = [[] for _ in range(target_length)]
        for index, row in targets.iterrows():
            for i in range(target_length):
                targets_collected[i].append(row[i])
        return targets_collected
    
    def instances_transfer(self, X_train, X_pool, y_train, y_pool, indices, method):
        # transfer data instances from the unlabelled pool to the training dataset
        # indices must be the real indices values, not positions
        instances_epoch = X_pool.iloc[indices]
        targets_epoch = y_pool.iloc[indices]

        X_train = pd.concat([X_train, instances_epoch])
        y_train = pd.concat([y_train, targets_epoch])
        X_pool = X_pool.drop(indices)
        y_pool = y_pool.drop(indices)

        return instances_epoch, targets_epoch
    

    



    



