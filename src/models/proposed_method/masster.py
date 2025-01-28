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
from models.semi_supervised_learning.self_learning import TargetSelfLearning

data_dir = DATA_DIR
dataset_name = DATASET_NAME

k_folds = K_FOLDS
threshold = THRESHOLD
random_state = RANDOM_STATE
n_trees = N_TREES
batch_percentage = BATCH_PERCENTAGE
batch_percentage_ssl = BATCH_PERCENTAGE_SSL
iterations = ITERATIONS
threshold = THRESHOLD

# read the first version of the datasets to define the batch size
X_train, y_train, X_pool, y_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names = read_data(data_dir, dataset_name, 1)
batch_size = round((BATCH_PERCENTAGE / 100) * len(X_pool)) 
batch_size_ssl = round((BATCH_PERCENTAGE_SSL / 100) * len(X_pool)) 

class MASSTER:
    def __init__(self, ss_model, data_dir, dataset_name, k_folds, random_state, n_trees, batch_size, batch_size_ssl, iterations, threshold):
        self.ss_model = ss_model # string ['cotraining', 'self-learning']
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.k_folds = k_folds
        self.random_state = random_state
        self.n_trees = n_trees
        self.max_iter = 1 #iterations by turn
        self.batch_size = batch_size
        self.batch_size_ssl = batch_size_ssl
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
        if self.ss_model is None:
            ss_model = None
        if self.ss_model == 'cotraining':
            ss_model = TargetCoTraining(self.data_dir, self.dataset_name, self.k_folds, self.max_iter, self.threshold ,self.random_state, self.n_trees, self.batch_size_ssl)
        if self.ss_model == 'self_learning':
            ss_model = TargetSelfLearning(self.data_dir, self.dataset_name, self.k_folds, self.max_iter, self.threshold, self.random_state, self.n_trees, self.batch_size_ssl)
        return ss_model
    
     
    """ def merge_features(self, X_v1, X_v2, feature_names_v1, feature_names_v2):
        # verify if X_v1 and X_v2 are DataFrames
        if not isinstance(X_v1, pd.DataFrame) or not isinstance(X_v2, pd.DataFrame):
            raise ValueError("X_v1 and X_v2 must be DataFrames.")
        
        # verify if both dataframes have the same number of rows
        if len(X_v1) != len(X_v2):
            raise ValueError("X_v1 and X_v2 must have the same number of rows.")
        
        # metge DataFrames along columns
        X_merged = pd.concat([X_v1, X_v2], axis=1)
        
        # combine feature names
        feature_names_merged = feature_names_v1 + feature_names_v2
        
        return X_merged, feature_names_merged """
    
    def run_masster(self, fold):
        stopping_criterion = False
        iteration = 0
        pairs_per_iteration_active = []
        
        # read data
        X_train, y_train, X_pool, y_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names = read_data(self.data_dir, self.dataset_name, fold + 1)

        while not stopping_criterion and iteration < self.iterations:
            target_qbc_model = self.initialize_active_learning()
            #ss_model = self.initialize_semi_supervised_learning()

            # run active learning model by max_iter each turn
            r2, mse, mae, ca, arrmse, added_pairs_per_iteration_active, all_pred_selected_pairs_active, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length = target_qbc_model.training(
                X_train, X_pool, X_test, y_train, y_pool, y_test, target_length)
            
            print('TENTANDO DEBUGAR')
            print(X_train.shape[0])

            pairs_per_iteration_active.append(added_pairs_per_iteration_active)
            #print('ACTIVE')
            #print(f'X_train len: {len(X_train)}')
            #print(f'y_train len: {len(y_train)}')
            
            """ # update y_pool_nan to avoid using same pairs on semi-supervised module
            y_pool_nan = update_y_pool_nan(y_pool_nan, all_pred_selected_pairs_active)

            # run semi-supervised learning model by max_iter each turn
            if self.ss_model == 'cotraining':
                r2, mse, mae, ca, arrmse, added_pairs_per_iteration_ss, all_pred_selected_pairs_ss, X_train_v1, X_train_v2, X_pool_v1, X_pool_v2, y_train, X_test_v1, X_test_v2, y_test, target_length, feature_names_v1, feature_names_v2, fold_index, y_pool_nan = ss_model.training(
                    X_train, y_train, X_pool, X_test, y_test, target_length, feature_names, fold, y_pool_nan)
                
                X_train, feature_names = self.merge_features(X_train_v1, X_train_v2, feature_names_v1, feature_names_v2)
                X_pool, feature_names = self.merge_features(X_pool_v1, X_pool_v2, feature_names_v1, feature_names_v2)
                X_test, feature_names = self.merge_features(X_test_v1, X_test_v2, feature_names_v1, feature_names_v2)

            if self.ss_model == 'self_learning':
                r2, mse, mae, ca, arrmse, added_pairs_per_iteration_ss, all_pred_selected_pairs_ss, X_train, X_pool, y_train, X_test, y_test, target_length, feature_names, fold_index, y_pool_nan = ss_model.training(
                    X_train, y_train, X_pool, X_test, y_test, target_length, feature_names, fold, y_pool_nan)
                
            #print('COTRAINING')
            #print(f'X_train len: {len(X_train)}')
            #print(f'y_train len: {len(y_train)}')

            # update y_pool to avoid using same pairs on active module 
            y_pool = update_y_pool(y_pool, all_pred_selected_pairs_ss) """

            print('---')
            
            # Ensure metrics are single values
            r2_value = r2[-1,0] if isinstance(r2, (list, np.ndarray)) else r2
            mse_value = mse[-1,0] if isinstance(mse, (list, np.ndarray)) else mse
            mae_value = mae[-1,0] if isinstance(mae, (list, np.ndarray)) else mae
            ca_value = ca[-1,0] if isinstance(ca, (list, np.ndarray)) else ca
            arrmse_value = arrmse[-1,0] if isinstance(arrmse, (list, np.ndarray)) else arrmse

            # store metrics
            self.R2[fold, iteration] = r2_value
            self.MSE[fold, iteration] = mse_value
            self.MAE[fold, iteration] = mae_value
            self.CA[fold, iteration] = ca_value
            self.ARRMSE[fold, iteration] = arrmse_value


            print(f"MASSTER Iteration {iteration} metrics: R²={r2_value}, MSE={mse_value}, MAE={mae_value}, CA={ca_value}, ARRMSE={arrmse_value}")
            print('---')
            print()

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
    #batch_size_ssl = round((batch_percentage_ssl / 100) * len(X_pool))

    #ss_model = 'cotraining'
    #ss_model = 'self_learning'
    #ssl_options = ['cotraining', 'self_learning']

    """ for ss_model in ssl_options:
        print(f'The selected semi_supervised model is: {ss_model}')
        masster = MASSTER(ss_model, data_dir, dataset_name, k_folds, random_state, n_trees, batch_size, batch_size_ssl, iterations, threshold)

        for fold in range(k_folds):
            masster.run_masster(fold)
            print(fold) """
    
    masster = MASSTER(None, data_dir, dataset_name, k_folds, random_state, n_trees, batch_size, batch_size_ssl, iterations, threshold)
    for fold in range(k_folds):
        masster.run_masster(fold)











    

    


