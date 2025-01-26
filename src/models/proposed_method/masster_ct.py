"""Proposed method using CoTraining"""
import numpy as np
import time
import sys
import pandas as pd
from pathlib import Path
from statistics import variance
from contextlib import redirect_stdout

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent.parent
# Adding path to sys.path
sys.path.append(str(project_root))

import config
from config import *

from models.single_target_model import SingleTargetRegressor
from data.dataframes_creation import read_data
from models.proposed_method.unlabeled_targets_update import update_y_pool_nan, update_y_pool
from models.active_learning.target_qbc import TargetQBC
from models.semi_supervised_learning.cotraining import TargetCoTraining

data_dir = DATA_DIR
dataset_name = DATASET_NAME

k_folds = K_FOLDS
threshold = THRESHOLD
random_state = RANDOM_STATE
n_trees = N_TREES
batch_percentage = BATCH_PERCENTAGE
iterations = ITERATIONS
threshold = THRESHOLD

# read the first version of the datasets to define the batch size
X_train, y_train, X_pool, y_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names = read_data(data_dir, dataset_name, 1)
batch_size = round((BATCH_PERCENTAGE / 100) * len(X_pool)) 

class MASSTER:
    def __init__(self, ss_model, data_dir, dataset_name, k_folds, random_state, n_trees, batch_size, iterations, threshold):
        self.ss_model = ss_model # string ['cotraining', 'self-learning']
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.k_folds = k_folds
        self.random_state = random_state
        self.n_trees = n_trees
        self.max_iter = 1 #iterations by turn
        self.batch_size = batch_size
        self.iterations = iterations
        self.threshold = threshold
        self.model = SingleTargetRegressor(random_state, n_trees)

        
        self.R2 = np.zeros([self.k_folds, self.iterations+1])
        self.MSE = np.zeros([self.k_folds, self.iterations+1])
        self.MAE = np.zeros([self.k_folds, self.iterations+1])
        self.CA = np.zeros([self.k_folds, self.iterations+1])
        self.ARRMSE = np.zeros([self.k_folds, self.iterations+1])

    def initialize_active_learning(self):
        target_qbc_model = TargetQBC(self.data_dir, self.dataset_name, self.k_folds, self.random_state, self.n_trees, self.max_iter, self.batch_size)
        return target_qbc_model
    
    def initialize_semi_supervised_learning(self):
        if self.ss_model == 'cotraining':
            ss_model = TargetCoTraining(self.data_dir, self.dataset_name, self.k_folds, self.max_iter, self.threshold ,self.random_state, self.n_trees, self.batch_size)
        
        return ss_model
    
    def run_masster(self, fold):
        target_qbc_model = self.initialize_active_learning()
        ss_model = self.initialize_semi_supervised_learning()

        X_train, y_train, X_pool, y_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names = read_data(self.data_dir, self.dataset_name, fold + 1)

        stopping_criterion = False
        iteration = 0
        pairs_per_iteration_active = []

        while not stopping_criterion and iteration < self.iterations:
            # run active learning model by max_iter each turn
            _, _, _, _, _, added_pairs_per_iteration_active, all_pred_selected_pairs_active = target_qbc_model.train_and_evaluate(fold)
            pairs_per_iteration_active.append(added_pairs_per_iteration_active)
            

            # update y_pool_nan to avoid using same pairs on semi-supervised module
            y_pool_nan = update_y_pool_nan(y_pool_nan, all_pred_selected_pairs_active)

            # run semi-supervised learning model by max_iter each turn
            _, _, _, _, _, added_pairs_per_iteration_ss, all_pred_selected_pairs_ss = ss_model.train_and_evaluate(fold)

            # update y_pool to avoid using same pairs on active module 
            y_pool = update_y_pool(y_pool, all_pred_selected_pairs_ss)

            # evaluate model
            model_array = self.model.unique_fit(target_length, y_train, X_train)
            predictions = self.model.unique_predict(model_array, X_test, target_length, y_test.columns)
            r2, mse, mae, ca, arrmse = self.model.unique_evaluate(y_test, predictions)

            # store metrics
            self.R2[fold, iteration] = r2
            self.MSE[fold, iteration] = mse
            self.MAE[fold, iteration] = mae
            self.CA[fold, iteration] = ca
            self.ARRMSE[fold, iteration] = arrmse

            print(f"Iteration {iteration} metrics: R²={r2}, MSE={mse}, MAE={mae}, CA={ca}, ARRMSE={arrmse}")

            stopping_criterion = len(X_pool) == 0  # empty X_pool
            iteration += 1

        self.save_results(fold, pairs_per_iteration_active)

    def save_results(self, fold, pairs_per_iteration_active):

        R2_flat = self.R2[fold, :].flatten()
        MSE_flat = self.MSE[fold, :].flatten()
        MAE_flat = self.MAE[fold, :].flatten()
        CA_flat = self.CA[fold, :].flatten()
        ARRMSE_flat = self.ARRMSE[fold, :].flatten()

        if not pairs_per_iteration_active:
            pairs_per_iteration_active = [0]
        num_entries = max(R2_flat.size, MSE_flat.size, MAE_flat.size, CA_flat.size, ARRMSE_flat.size)

        if len(pairs_per_iteration_active) < num_entries:
            pairs_per_iteration_active.extend([0] * (num_entries - len(pairs_per_iteration_active)))
        if len(R2_flat) < num_entries:
            R2_flat = np.append(R2_flat, [None] * (num_entries - len(R2_flat)))
        if len(MSE_flat) < num_entries:
            MSE_flat = np.append(MSE_flat, [None] * (num_entries - len(MSE_flat)))
        if len(MAE_flat) < num_entries:
            MAE_flat = np.append(MAE_flat, [None] * (num_entries - len(MAE_flat)))
        if len(CA_flat) < num_entries:
            CA_flat = np.append(CA_flat, [None] * (num_entries - len(CA_flat)))
        if len(ARRMSE_flat) < num_entries:
            ARRMSE_flat = np.append(ARRMSE_flat, [None] * (num_entries - len(ARRMSE_flat)))

        results_df = pd.DataFrame({
            'Fold_Index': [fold for _ in range(num_entries)],
            'Iterations': list(range(num_entries)),
            'R2': R2_flat,
            'MSE': MSE_flat,
            'MAE': MAE_flat,
            'CA': CA_flat,
            'ARRMSE': ARRMSE_flat,
            'Added_Pairs': pairs_per_iteration_active
        })

        results_path = Path(f'reports/proposed_method/{self.dataset_name}')
        results_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path / f'masster_{self.ss_model}_fold_{fold}.csv', index=False)
        print('Saved data...')

if __name__ == "__main__":
    data_dir = config.DATA_DIR
    dataset_name = config.DATASET_NAME

    print('Proposed method...')
    X_train, y_labeled, X_pool, y_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length,target_names,feature_names = read_data(data_dir, dataset_name, 1)
    batch_size = round((batch_percentage / 100) * len(X_pool))

    ss_model = 'cotraining'

    masster = MASSTER(ss_model, data_dir, dataset_name, k_folds, random_state, n_trees, batch_size, iterations, threshold)

    for fold in range(k_folds):
        masster.run_masster(fold)
        print(fold)











    

    


