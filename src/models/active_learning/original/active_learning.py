""" The class for the general active learning method. """
import pandas as pd
import sys
from pathlib import Path

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
# Adding path to sys.path
sys.path.append(str(project_root))

from config import *
from data.data_processing import *
from utils.data_handling import read_csv

data_dir = DATA_DIR

class activelearning:
    n_trees = N_TREES
    random_state = RANDOM_STATE
    iterations = ITERATIONS

    def __init__(self, dataset_name):
        self.dataset_name = DATASET_NAME
        
    def data_read(self, dataset):
        # split the csv file in the input and target values
        folder_dir = data_dir / 'processed' / f'{self.dataset_name}'
        data_path = folder_dir / f'{dataset}'
        df = pd.read_csv(data_path)

        # obtain the column names
        col_names = list(df.columns)
        target_length = 0

        for name in col_names: 
            if 'target' in name:
                target_length += 1

        target_names = col_names[-target_length:]

        inputs = list()
        targets = list()
        for i in range(len(df)):
            input_val = list()
            target_val = list()
            for col in col_names:
                if col in target_names:
                    target_val.append(df.loc[i, col])
                else:
                    input_val.append(df.loc[i, col])
            inputs.append(input_val)
            targets.append(target_val)

        n_instances = len(targets)
        return inputs, targets, n_instances, target_length
    
    def target_collect(self, targets, target_length):
        # collect all the values for specific targets in separate lists
        targets_collected = list()
        for j, target in enumerate(targets):
            for i in range(target_length):
                # only make the separate lists in the beginning
                if j == 0:
                    targets_collected.append(list())
                targets_collected[i].append(target[i])
        return targets_collected

    def instances_transfer(self, X_train, X_pool, y_train, y_pool, indices, method):
        # transfer data instances from the unlabelled pool to the training dataset
        instances_epoch = list()
        targets_epoch = list()
        
        for index in indices:
            instance = X_pool[index]
            target = y_pool[index]

            X_train.append(instance)
            y_train.append(target)
            X_pool.pop(index)
            y_pool.pop(index)
            instances_epoch.append(instance)
            targets_epoch.append(target)

        instances_epoch.append([])
        targets_epoch.append([])
        return instances_epoch, targets_epoch