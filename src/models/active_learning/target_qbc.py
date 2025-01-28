"""Target-based QBC method adapted to DataFrames structure"""
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

from models.active_learning.active_learning import *
from models.single_target_model import SingleTargetRegressor
from data.dataframes_creation import data_read, read_data

data_dir = DATA_DIR
dataset = DATASET_NAME

k_folds = K_FOLDS
threshold = THRESHOLD
random_state = RANDOM_STATE
n_trees = N_TREES
n_epochs = N_EPOCHS
batch_percentage = BATCH_PERCENTAGE

# read the first version of the datasets to define the batch size
X_train, y_train, X_pool, y_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names = read_data(data_dir, dataset, 1)
batch_size = round((BATCH_PERCENTAGE / 100) * len(X_pool)) 

class TargetQBC(ActiveLearning):
    def __init__(self, data_dir, dataset_name, k_folds,  random_state, n_trees, n_epochs, batch_size):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.k_folds = k_folds
        self.random_state = random_state
        self.n_trees = n_trees
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = SingleTargetRegressor(random_state, n_trees)
        
        self.R2 = np.zeros([self.k_folds, self.n_epochs+1])
        self.MSE = np.zeros([self.k_folds, self.n_epochs+1])
        self.MAE = np.zeros([self.k_folds, self.n_epochs+1])
        self.CA = np.zeros([self.k_folds, self.n_epochs+1])
        self.ARRMSE = np.zeros([self.k_folds, self.n_epochs+1])

    def calculate_variances(self, model, X_pool, target_length): 
        # Initialize variances and predictions arrays with proper shape
        variances = pd.DataFrame(index=X_pool.index, columns=range(target_length))
        preds = pd.DataFrame(index=X_pool.index, columns=range(target_length))

        for i in range(target_length):
            # Initialize an array for predictions from all trees for the current target
            pool_preds = np.zeros((len(X_pool), self.n_trees))
            
            # Access the random forest model for the current target
            rf_model = model[i]
        
            for j, estimator in enumerate(rf_model.estimators_):
                # Make predictions for the current tree
                predictions = estimator.predict(X_pool.values)  # Ensure no column names passed
                pool_preds[:, j] = predictions

            # Calculate variance for each instance in X_pool
            for idx, row_idx in enumerate(X_pool.index):
                variances.loc[row_idx, i] = variance(pool_preds[idx, :])

        return variances
    
    def max_var(self, variances, target_length, all_pred_selected_pairs):
        # Flatten the variances DataFrame to get pairs of (variance, instance_index, target_index)
        flattened_variances = [(variances.iloc[i, j], variances.index[i], variances.columns[j]) for i in range(variances.shape[0]) for j in range(variances.shape[1])]
        
        # Sort the flattened variances by variance value in descending order
        sorted_variances = sorted(flattened_variances, key=lambda x: x[0], reverse=True)
        
        # Select the top-k variances
        k = self.batch_size * target_length

        top_k_variances = []
        for item in sorted_variances:
            if (item[1], item[2]) not in all_pred_selected_pairs.keys():
                top_k_variances.append(item)
            if len(top_k_variances) == k:
                break
        
        # Create the dictionary with (instance_index, target_index) as key and variance as value
        pred_selected_pairs = {(item[1], item[2]): item[0] for item in top_k_variances}
        
        return pred_selected_pairs
    
    def training(self, X_train, X_pool, X_test, y_train, y_pool, y_test, target_length,
                all_pred_selected_pairs = None,
                added_pairs_per_iteration = None,
                dict_index = None):
        # initialize instances_pool and targets_pool as empty dataframes with target columns and ??? lines
#        print(X_train)
        if all_pred_selected_pairs is None:
            all_pred_selected_pairs = {}
            added_pairs_per_iteration = []
            dict_index = {} # {key:value} -> {index_train:index_pool}

        for epoch in range(self.n_epochs):
            # all_pred_selected_pairs = {}
            # added_pairs_per_iteration = []
            # dict_index = {} # {key:value} -> {index_train:index_pool}
#            print(X_train.dtypes)
 #           print(y_train.dtypes)

            # pd.DataFrame(X_train.to_csv("targetbased_x.csv" + str(epoch)) )            
            # pd.DataFrame(y_train.to_csv("targetbased_y.csv" + str(epoch)))            
            # pd.DataFrame(X_pool.to_csv("targetbased_x_pool.csv" + str(epoch)) )            
            # pd.DataFrame(y_pool.to_csv("targetbased_y_pool.csv" + str(epoch)))            

            print("Epoch {}:".format(epoch+1))
            print("     The training set size: {}".format(len(X_train)))
            print("     The unlabelled pool size: {}".format(len(X_pool)))
#            print(X_train.shape[0])
            # initialize models 
            # NOTE: models_array will containg target_length models, since we are using a local approach to MTR
            models_array = self.model.unique_fit(target_length, y_train, X_train)

            # get initial predictions
            columns = list(y_pool.columns)
            preds = self.model.unique_predict(models_array, X_pool, target_length, columns)

            # compute variances
            variances = self.calculate_variances(models_array, X_pool, target_length)
#            print(variances)
#            self.v = variances

#            pd.DataFrame(variances).to_csv("variance_target.csv" + str(epoch))
            # compute top_k_variances using max_var
            pred_selected_pairs = {}
            pred_selected_pairs = self.max_var(variances, target_length, all_pred_selected_pairs)

            # save oracle predictions to the selected pairs
            oracle_pred_selected_pairs = {(idx, target_idx): y_pool.loc[idx, columns[target_idx]] for idx, target_idx in pred_selected_pairs.keys()}

            # update all_pred_selected_pairs with oracle_pred_selected_pairs
            all_pred_selected_pairs.update(oracle_pred_selected_pairs)
            print(f'cumulative pairs: {len(all_pred_selected_pairs)}')

            if not oracle_pred_selected_pairs:
                print("No input requested to the oracle.")
                break

            else:
                print(f'# requested pairs: {len(oracle_pred_selected_pairs)}')
                

            indices = set()
            for idx_pool, j in oracle_pred_selected_pairs.keys():
                indices.add(idx_pool)

            added_pairs_per_iteration.append(len(oracle_pred_selected_pairs))
            
            print("# of distinct indices "+str(len(indices)))
            count = 0 

            for idx_pool, j in oracle_pred_selected_pairs.keys():
                
                if count == 0 and (idx_pool not in indices):
                    print('initialization...')

                    # updating X_train
                    if idx_pool not in X_train.index:
                        X_train = pd.concat([X_train, X_pool.loc[idx_pool]], ignore_index=False)
                    
                    # columns names
                    columns = list(y_pool.columns)
                    
                    # {train idx:pool idx}
                    idx_train = X_train.index[-1]
                    dict_index[idx_train] = idx_pool

                    # fill y_pool with nan
                    y_pool.loc[idx_pool, columns[j]] = np.nan 

                    # fill y_train
                    y_train.loc[idx_train, columns[j]] = oracle_pred_selected_pairs[(idx_pool,j)]

                    count = count + 1
                    

                else:             
                    # {train idx:pool idx}
                    if idx_pool in dict_index.values():
                        keys = [k for k, v in dict_index.items() if v == idx_pool]
                        
                        y_pool.loc[idx_pool, columns[j]] = np.nan
                        y_train.loc[keys[0], columns[j]] = oracle_pred_selected_pairs[(idx_pool,j)]
                        
                        count = count + 1

                    else:
                        # fill new x_train instance 
                        if idx_pool not in X_train.index:
                            X_train = pd.concat([X_train,X_pool.loc[[idx_pool]]], ignore_index=False)
                        
                        columns = list(y_pool.columns)
                        
                        # create {idx_pool, idx_train} pair
                        last_index = X_train.index[-1]
                        dict_index[last_index] = idx_pool

                        # fill y_pool 
                        y_pool.loc[idx_pool, columns[j]] = np.nan
                        y_train.loc[last_index, columns[j]] = oracle_pred_selected_pairs[(idx_pool,j)]
                        
                        count = count + 1 

            # CONDITION: check for complete lines (instances)
            complete_instances_idx = y_pool[y_pool.isna().all(axis=1)].index
            print('Complete instances')
            print(complete_instances_idx)

            if not complete_instances_idx.empty:
                exclusion_indices = complete_instances_idx[
                    complete_instances_idx.isin(y_pool.index) |
                    complete_instances_idx.isin(X_pool.index)
                ]

                if not exclusion_indices.empty:
                    print('i have already excluded that for you (active)...') 
                    y_pool = y_pool.drop(exclusion_indices, errors='ignore')
                    X_pool = X_pool.drop(exclusion_indices, errors='ignore')   


            # update predictions
            preds = self.model.unique_predict(models_array, X_test, target_length, columns)

            # compute metrics
            r2, mse, mae, ca, arrmse = self.model.unique_evaluate(y_test, preds)
            self.R2[:,epoch] = (r2)
            self.MSE[:,epoch] = (mse)
            self.MAE[:,epoch] = (mae)
            self.CA[:,epoch] = (ca)
            self.ARRMSE[:,epoch] = (arrmse)

        #return r2, mse, mae, ca, arrmse, added_pairs_per_iteration, all_pred_selected_pairs, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length
        return r2, mse, mae, ca, arrmse, added_pairs_per_iteration, all_pred_selected_pairs, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length,  all_pred_selected_pairs, added_pairs_per_iteration, dict_index

#        return r2, mse, mae, ca, arrmse, added_pairs_per_iteration, all_pred_selected_pairs, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length

    def train_and_evaluate(self, fold_index, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length):
        print(f"\nTraining target-based QBC model in fold {fold_index}...")
        
        r2, mse, mae, ca, arrmse, added_pairs_per_iteration, all_pred_selected_pairs, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length, _, _, _ = self.training(X_train, X_pool, X_test, y_train, y_pool, y_test, target_length)

        return self.R2, self.MSE, self.MAE, self.CA, self.ARRMSE, added_pairs_per_iteration, all_pred_selected_pairs, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length

if __name__ == "__main__":
    data_dir = config.DATA_DIR
    dataset_name = config.DATASET_NAME

    print('Target-based QBC...')
    X_train, y_labeled, X_pool, y_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length,target_names,feature_names = read_data(data_dir, dataset_name, 1)
    
    batch_size = round((batch_percentage / 100) * len(X_pool))

    target_active_model = TargetQBC(data_dir, dataset_name, k_folds, random_state, n_trees, n_epochs, batch_size)

    for fold in range(k_folds):
        X_train, y_train, X_pool, y_pool, y_pool_nan, X_rest, y_rest, X_test, y_test, target_length, target_names, feature_names = read_data(data_dir, dataset_name, fold+1)
        R2, MSE, MAE, CA, ARRMSE, added_pairs_per_iteration, all_pred_selected_pairs, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length = target_active_model.train_and_evaluate(fold, X_train, y_train, X_pool, y_pool, X_test, y_test, target_length)
        print(f"Index i: {fold}")
        print(f"Length of added_pairs_per_iteration: {len(added_pairs_per_iteration)}")
        print(f"added_pairs_per_iteration: {added_pairs_per_iteration}")
        # Save results for each fold to DataFrame and CSV
        R2_flat = R2[fold, :].flatten()
        MSE_flat = MSE[fold, :].flatten()
        MAE_flat = MAE[fold, :].flatten()
        CA_flat = CA[fold, :].flatten()
        ARRMSE_flat = ARRMSE[fold, :].flatten()
        if not added_pairs_per_iteration:
            added_pairs_per_iteration = [0]
        # Ensure added_pairs_per_iteration is the same length as the other arrays
        num_entries = max(R2[fold, :].size, MSE[fold, :].size, MAE[fold, :].size, CA[fold, :].size, ARRMSE[fold, :].size)
        if len(added_pairs_per_iteration) < num_entries:
            added_pairs_per_iteration.extend([0] * (num_entries - len(added_pairs_per_iteration)))
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
            'Added_Pairs': added_pairs_per_iteration
        })

        results_path = Path(f'reports/active_learning/{dataset_name}')
        results_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path / f'target_qbc_results_fold_{fold}.csv', index=False)
        print('Saved data...')



    

                    
                           
    
    

    
