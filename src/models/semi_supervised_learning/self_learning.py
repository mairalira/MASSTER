from statistics import variance
import numpy as np
import time
import sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent.parent
print(project_root)
# Adding path to sys.path
sys.path.append(str(project_root))

import config
from config import *

from models.single_target_model import SingleTargetRegressor
from data.dataframes_creation import data_read, read_data
from utils.metrics import custom_accuracy, arrmse_metric


# Main script
data_dir = DATA_DIR
dataset = DATASET_NAME

k_folds = K_FOLDS
iterations = ITERATIONS
threshold = THRESHOLD
random_state = RANDOM_STATE
n_trees = N_TREES
batch_percentage = BATCH_PERCENTAGE_SSL

class SelfLearning:
    def __init__(self, data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.k_folds = k_folds
        self.iterations = iterations
        self.threshold = threshold
        self.random_state = random_state
        self.n_trees = n_trees
        self.model = SingleTargetRegressor(random_state, n_trees)

        self.R2 = np.zeros([self.k_folds, self.iterations+1])
        self.MSE = np.zeros([self.k_folds, self.iterations+1])
        self.MAE = np.zeros([self.k_folds, self.iterations+1])
        self.CA = np.zeros([self.k_folds, self.iterations+1])
        self.ARRMSE = np.zeros([self.k_folds, self.iterations+1])

    def stop_criterion(self, preds):
        return len(preds) == 0 
    
    
    def unique_evaluate_model(self, models_view, X_test, y_test_labeled):
        columns = list(X_test.columns)
        print("Target Self-Learning Making predictions on test data...")
        predictions = pd.DataFrame(np.nan, index=X_test.index, columns=y_test_labeled.columns)
                    
        for i in range(len(models_view)):
            rf_model = models_view[i]
            predictions.iloc[:, i] = rf_model.predict(X_test)  # Atribuindo a predição na coluna correspondente

        y_pred_combined = predictions
                    
        r2 = np.round(r2_score(np.asarray(y_test_labeled), y_pred_combined), 4)

        mse = np.round(mean_squared_error(np.asarray(y_test_labeled), y_pred_combined), 4)

        mae = np.round(mean_absolute_error(np.asarray(y_test_labeled), y_pred_combined), 4)

        ca = np.round(custom_accuracy(y_test_labeled.values, y_pred_combined.values, threshold=CA_THRESHOLD), 4)

        arrmse = np.round(arrmse_metric(np.asarray(y_test_labeled), np.asarray(y_pred_combined)), 4)

        print(f"    Overall: R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}, CA={ca:.3f}, ARRMSE={arrmse:.3f}")
                
        return r2, mse, mae, ca, arrmse

class TargetSelfLearning(SelfLearning):
        def __init__(self, data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees, batch_size):
            super().__init__(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
            self.batch_size = batch_size


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
        
        def select_confident_pairs(self, variances):
            confident_pairs = {}        # {(idx,col) : variance}

            # Iterating over the DataFrame rows and columns using iterrows() for index compatibility
            for idx, row in variances.iterrows():  # iterrows() gives (index, Series)

                for col, value in row.items():  # Iterating over each column in the row
                    if value <= self.threshold:
                        confident_pairs[(int(idx), col)] = value  # Add (index, column) pair to the dictionary
            return confident_pairs

        def training(self, X_train_df, y_train_df, X_pool_df, X_test,  y_test, target_length, feature_names, fold_index, y_pool):

            execution_times = []
            added_pairs_per_iteration = []
            dict_index = {} # {key:value} -> {index_train:index_pool}
            all_pred_selected_pairs = {}
            model = SingleTargetRegressor(self.random_state,self.n_trees)

            for iteration in range(self.iterations):

                print(f"Iteration {iteration + 1}/{self.iterations}")

                start_time = time.time()

                models_view_array = model.unique_fit(target_length, y_train_df, X_train_df)

                columns = list(y_test.columns)
                preds = model.unique_predict(models_view_array, X_pool_df,target_length,columns)
                
                if self.stop_criterion(preds):
                    print(" No more unlabeled examples.")
                    break
                
                variances = self.calculate_variances(models_view_array, X_pool_df, target_length)
                
                confident_pairs= self.select_confident_pairs(variances)

                confident_pairs_combined = {}
                confident_pairs_combined_filtered = {} 
                for pair in set(confident_pairs):
                    if pair in confident_pairs:
                        confident_pairs_combined[pair] = confident_pairs[pair]
                                     
                sorted_confident_pairs_filtered = {}
                

                for pair in confident_pairs_combined.keys():
                    if pair in all_pred_selected_pairs.keys():
                        continue
                    else:
                        sorted_confident_pairs_filtered[pair]= confident_pairs_combined[pair]
                
                sorted_confident_pairs = sorted(sorted_confident_pairs_filtered.items(), key=lambda item: item[1])

                pred_selected_pairs = {}

                for (i,j), _ in sorted_confident_pairs[:self.batch_size * target_length]:
                    pred_selected_pairs[(i, j)]=preds.loc[i, columns[j]]
                    all_pred_selected_pairs[(i, j)]= preds.loc[i, columns[j]]
                
                if not pred_selected_pairs:
                    print("No confident predictions found.")
                    break
                
                print("# of selected pairs: " + str(len(pred_selected_pairs)))
                indices = set()
                for idx_pool, j in pred_selected_pairs.keys():
                    indices.add(idx_pool)
                added_pairs_per_iteration.append(len(pred_selected_pairs))

                print("# of distinct indices "+str(len(indices)))
                count = 0 
                
                for idx_pool, j in pred_selected_pairs.keys():
                    
                    if pd.notna(y_pool.loc[idx_pool, columns[j]]):
                        
                        continue
                    
                    if count == 0 and idx_pool not in indices:
                        print('initialization...')

                        # updating X_trains
                        if idx_pool not in X_train_df.index:
                            X_train_df = pd.concat([X_train_df, X_pool_df.loc[idx_pool]], ignore_index=False)
                            
                        # columns names
                        columns = list(y_pool.columns)
                        
                        # {train idx:pool idx}
                        idx_train = X_train_df.index[-1]
                        dict_index[idx_train] = idx_pool

                        # fill y_pool
                        y_pool.loc[idx_pool, columns[j]] = pred_selected_pairs[(idx_pool,j)]

                        # fill y_train
                        y_train_df.loc[idx_train, columns[j]] = pred_selected_pairs[(idx_pool,j)]

                        count = count + 1
                        
                    else:
                        # {train idx:pool idx}
                        if idx_pool in dict_index.values():
                            keys = [k for k, v in dict_index.items() if v == idx_pool]
                            
                            y_pool.loc[idx_pool, columns[j]] = pred_selected_pairs[(idx_pool,j)]
                            y_train_df.loc[keys[0], columns[j]] = pred_selected_pairs[(idx_pool,j)]
                            
                            count = count + 1

                        else:
                            # fill new x_train instance 
                            if idx_pool not in X_train_df.index:
                                X_train_df = pd.concat([X_train_df,X_pool_df.loc[[idx_pool]]], ignore_index=False)
                                
                            columns = list(y_pool.columns)
                            
                            # create {idx_pool, idx_train} pair
                            last_index = X_train_df.index[-1]
                            dict_index[last_index] = idx_pool

                            # fill y_pool 
                            y_pool.loc[idx_pool, columns[j]] = pred_selected_pairs[(idx_pool,j)]
                            y_train_df.loc[last_index, columns[j]] = pred_selected_pairs[(idx_pool,j)]
                            
                            count = count + 1

                # CONDITION: check for complete lines (instances)
                complete_instances_idx = y_pool[y_pool.notna().all(axis=1)].index
                print(f'Complete instances: {complete_instances_idx}')

                if not complete_instances_idx.empty:
                    exclusion_indices = complete_instances_idx[
                        complete_instances_idx.isin(y_pool.index) |
                        complete_instances_idx.isin(X_pool_df.index) 
                    ]

                    if not exclusion_indices.empty:
                        print('i have already excluded that for you (self-learning)...')
                        y_pool = y_pool.drop(exclusion_indices, errors='ignore')
                        X_pool_df = X_pool_df.drop(exclusion_indices, errors='ignore')

                r2, mse, mae, ca, arrmse = self.unique_evaluate_model(models_view_array, X_test, y_test)
                
                self.R2[fold_index, iteration] = r2
                self.MSE[fold_index, iteration] = mse
                self.MAE[fold_index, iteration] = mae
                self.CA[fold_index, iteration] = ca
                self.ARRMSE[fold_index, iteration] = arrmse

            models_view_array = model.unique_fit(target_length, y_train_df, X_train_df)
            r2, mse, mae, ca, arrmse = self.unique_evaluate_model(models_view_array, X_test, y_test)
            self.R2[fold_index, -1] = r2
            self.MSE[fold_index, -1] = mse
            self.MAE[fold_index, -1] = mae
            self.CA[fold_index, -1] = ca
            self.ARRMSE[fold_index, -1] = arrmse
            
            return r2, mse, mae, ca, arrmse, added_pairs_per_iteration, all_pred_selected_pairs, X_train_df, X_pool_df, y_train_df, X_test, y_test,  target_length, feature_names, fold_index, y_pool
        
        def train_and_evaluate(self, fold_index, X_train, y_train_df, X_pool, X_test, y_test, y_pool, target_length, target_names, feature_names):

            print(f"\nTraining self-learning model in fold {fold_index}...")
           
            r2, mse, mae, ca, arrmse, added_pairs_per_iteration, all_pred_selected_pairs, X_train, X_pool, y_train, X_test, y_test, target_length, feature_names, fold_index, y_pool_nan = self.training(
                X_train, y_train_df, X_pool, X_test, y_test, target_length, feature_names, fold_index, y_pool

            )
            return self.R2, self.MSE, self.MAE, self.CA, self.ARRMSE, added_pairs_per_iteration, all_pred_selected_pairs, X_train, X_pool, y_train_df, X_test, y_test, target_length, feature_names, fold_index, y_pool
        
if __name__ == "__main__":
    data_dir = config.DATA_DIR
    dataset_name = config.DATASET_NAME
    
    print('Target-based Self-Learning...')
    X_train, y_labeled, X_pool, y_pool_labeled, y_pool, X_rest, y_rest, X_test, y_test, target_length,target_names,feature_names = read_data(data_dir, dataset_name, 1)

    batch_size = round((batch_percentage / 100) * len(X_pool))

    target_selflearning_model = TargetSelfLearning(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees, batch_size)

    for i in range(k_folds):
        X_train_labeled, y_labeled, X_pool, y_pool_labeled, y_pool, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length, target_names, feature_names = read_data(data_dir, dataset_name, i+1)

        R2, MSE, MAE, CA, ARRMSE, added_pairs_per_iteration, all_pred_selected_pairs, X_train, X_pool, y_train_df, X_test, y_test, target_length, feature_names, fold_index, y_pool = target_selflearning_model.train_and_evaluate(
            i,  X_train_labeled, y_labeled, X_pool, X_test_labeled, y_test_labeled, y_pool, target_length, target_names, feature_names)
        print(f"Index i: {i}")
        print(f"Length of added_pairs_per_iteration: {len(added_pairs_per_iteration)}")
        print(f"added_pairs_per_iteration: {added_pairs_per_iteration}")
        
        # Save results for each fold to DataFrame and CSV
        R2_flat = R2[i, :].flatten()
        MSE_flat = MSE[i, :].flatten()
        MAE_flat = MAE[i, :].flatten()
        CA_flat = CA[i, :].flatten()
        ARRMSE_flat = ARRMSE[i, :].flatten()
        if not added_pairs_per_iteration:
            added_pairs_per_iteration = [0]
        # Ensure added_pairs_per_iteration is the same length as the other arrays
        num_entries = max(R2[i, :].size, MSE[i, :].size, MAE[i, :].size, CA[i, :].size, ARRMSE[i, :].size)
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
        print(len(R2_flat))
        print(len(added_pairs_per_iteration))
        results_df = pd.DataFrame({
            'Fold_Index': [i for _ in range(num_entries)],
            'Iterations': list(range(num_entries)),
            'R2': R2_flat,
            'MSE': MSE_flat,
            'MAE': MAE_flat,
            'CA': CA_flat,
            'ARRMSE': ARRMSE_flat,
            'Added_Pairs': added_pairs_per_iteration
        })

        results_path = Path(f'reports/semi_supervised_learning/{dataset_name}')
        results_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path / f'target_self_learning_results_fold_{i}.csv', index=False)
        print('Saved data...')
