""" This script contains code of the active learning algorithms for multi-target regression. """ 

# importing the libraries
import csv
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, pairwise_distances_argmin_min, auc
from sklearn.cluster import KMeans
from sklearn import preprocessing 
from pathlib import Path

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent
# Adding path to sys.path   
sys.path.append(str(project_root))

from utils.data_handling import *
from models.active_learning import *
from models.instance_based import *
from models.upperbound import *
from models.randomsampling import *
from models.greedy_sampling import *
from models.qbcrf import *

# Helper function to create directories
def create_directories(base_path, sub_dirs):
    for sub_dir in sub_dirs:
        path = base_path / sub_dir
        if not path.exists():
            path.mkdir(parents=True)

# Helper function to save predictions and transfer instances
def save_predictions_and_transfers(folder_path, prefix, i, Y_pred_df, instances_pool, targets_pool):
    ypred_path = folder_path / 'preds' / f"{prefix}_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_path)
    instances_pool_df = pd.DataFrame(instances_pool)
    targets_pool_df = pd.DataFrame(targets_pool)
    instances_pool_path = folder_path / f'transfer_instances_{prefix}_{i}.csv'
    targets_pool_path = folder_path / f'transfer_targets_{prefix}_{i}.csv'
    instances_pool_df.to_csv(instances_pool_path)
    targets_pool_df.to_csv(targets_pool_path)

# Helper function to plot and save figures
def plot_and_save_figures(epochs, metrics, labels, title, ylabel, fig_path, filename):
    for metric, label in zip(metrics, labels):
        plt.plot(epochs, metric, label=label)
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fig_path / filename)
    plt.show()

# Helper function to read data
def read_data(Method, iteration):
    X_train, y_train, _, _ = Method.data_read(f'train_{iteration}')
    X_pool, y_pool, n_pool, target_length = Method.data_read(f'pool_{iteration}')
    X_rest, y_rest, _, _ = Method.data_read(f'train+pool_{iteration}')
    X_test, y_test, _, _ = Method.data_read(f'test_{iteration}')
    return X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length

# Helper function to calculate mean performances
def calculate_mean_performances(method, submethod):
    mean_R2 = np.mean(getattr(method, f"{submethod}_R2"), axis=0)
    mean_MSE = np.mean(getattr(method, f"{submethod}_MSE"), axis=0)
    mean_MAE = np.mean(getattr(method, f"{submethod}_MAE"), axis=0)
    mean_CA = np.mean(getattr(method, f"{submethod}_CA"), axis=0)
    return mean_R2, mean_MSE, mean_MAE, mean_CA

# Helper function to append mean performances to total performances
def append_mean_performances(total_performances, mean_performances):
    return np.append(total_performances, np.reshape(mean_performances, (1, len(mean_performances))), axis=0)

# Helper function to save total performances to CSV
def save_total_performances_to_csv(folder_path, method_name, total_performances, metrics):
    cols = [f'Epoch {i+1}' for i in range(N_EPOCHS)] + ['AUC']
    rows = [f'Iteration {i+1}' for i in range(Method.iterations)] + ['Average']
    for metric, performance in zip(metrics, total_performances):
        df = pd.DataFrame(performance, index=rows, columns=cols)
        df.to_csv(folder_path / metric / f'{method_name}_{metric}.csv', header=cols)

# Main script
data_dir = DATA_DIR
dataset = DATASET_NAME
Method = activelearning(dataset)

# read the first version of the datasets to define the batch size
X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length = read_data(Method, 1)

batch_size = round((BATCH_PERCENTAGE / 100) * len(X_pool)) 
n_epochs = N_EPOCHS

# define the different methods
proposed_method_instance = instancebased(batch_size, n_epochs)
upperbound_method = upperbound(n_epochs)
lowerbound_method = lowerbound(batch_size, n_epochs)
baseline_method = baseline(batch_size, n_epochs)
qbcrf_method = qbcrf(batch_size, n_epochs)

# make the folders to store the results
folder_path = Path('reports') / 'active_learning' / f'{dataset}'
fig_path = folder_path / 'images'
create_directories(folder_path, ['r2', 'mse', 'mae', 'ca', 'preds'])
create_directories(fig_path, ['r2', 'mse', 'mae', 'ca'])

for i in range(Method.iterations):
    if i > 0:
        X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length = read_data(Method, i+1)

    # Copy the original datasets for the instance based, random and greedy method
    X_train_instance, X_pool_instance, y_train_instance, y_pool_instance = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_random, X_pool_random, y_train_random, y_pool_random = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_baseline, X_pool_baseline, y_train_baseline, y_pool_baseline = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_qbcrf, X_pool_qbcrf, y_train_qbcrf, y_pool_qbcrf = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    
    print("\n" + 50*"-" + f"Iteration {i+1}" + 50*"-" + "\n")

    # QBC-RF method
    print("\n" + 50*"-" + "QBC-RF method" + 50*"-" + "\n")
    R2, MSE, MAE, CA, Y_pred_df, instances_pool_qbcrf, targets_pool_qbcrf = qbcrf_method.training(X_train_qbcrf, X_pool_qbcrf, X_test, y_train_qbcrf, y_pool_qbcrf, y_test, target_length)
    qbcrf_method.qbcrf_R2[i,:] = R2
    qbcrf_method.qbcrf_MSE[i,:] = MSE
    qbcrf_method.qbcrf_MAE[i,:] = MAE
    qbcrf_method.qbcrf_CA[i,:] = CA
    save_predictions_and_transfers(folder_path, 'qbcrf', i, Y_pred_df, instances_pool_qbcrf, targets_pool_qbcrf)

    # Instance based    
    print("\n" + 50*"-" + "Instance based method" + 50*"-" + "\n")
    cols = [f"Target_{i+1}" for epoch in range(n_epochs) for i in range(target_length)]
    R2, MSE, MAE, CA, Y_pred_df, instances_pool_qbc, targets_pool_qbc = proposed_method_instance.training(X_train_instance, X_pool_instance, X_test, y_train_instance, y_pool_instance, y_test, target_length)
    proposed_method_instance.instance_R2[i,:] = R2
    proposed_method_instance.instance_MSE[i,:] = MSE
    proposed_method_instance.instance_MAE[i,:] = MAE
    proposed_method_instance.instance_CA[i,:] = CA
    save_predictions_and_transfers(folder_path, 'instance', i, Y_pred_df, instances_pool_qbc, targets_pool_qbc)

    # Upperbound 
    print("\n" + 50*"-" + "Upperbound method" + 50*"-" + "\n")
    cols = [f"Target_{i+1}" for i in range(target_length)]
    R2, MSE, MAE, CA, Y_pred_df = upperbound_method.training(X_rest, X_test, y_rest, y_test, target_length)
    upperbound_method.upperbound_R2[i,:] = R2
    upperbound_method.upperbound_MSE[i,:] = MSE
    upperbound_method.upperbound_MAE[i,:] = MAE
    upperbound_method.upperbound_CA[i,:] = CA
    ypred_upper_path = folder_path / 'preds' /  f"upperbound_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_upper_path)

    # Random sampling
    print("\n" + 50*"-" + "Lowerbound method" + 50*"-" + "\n")
    cols = [f"Target_{i+1}" for epoch in range(n_epochs) for i in range(target_length)]
    R2, MSE, MAE, CA, Y_pred_df = lowerbound_method.training(X_train_random, X_pool_random, X_test, y_train_random, y_pool_random, y_test, target_length)
    lowerbound_method.random_R2[i,:] = R2
    lowerbound_method.random_MSE[i,:] = MSE
    lowerbound_method.random_MAE[i,:] = MAE
    lowerbound_method.random_CA[i,:] = CA
    ypred_lower_path = folder_path / 'preds' / f"random_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_lower_path)

    # Greedy sampling
    print("\n" + 50*"-" + "Baseline method" + 50*"-" + "\n")
    R2, MSE, MAE, CA, Y_pred_df, instances_pool_baseline, targets_pool_baseline = baseline_method.training(X_train_baseline, X_pool_baseline, X_test, y_train_baseline, y_pool_baseline, y_test, target_length)
    baseline_method.baseline_R2[i,:] = R2
    baseline_method.baseline_MSE[i,:] = MSE
    baseline_method.baseline_MAE[i,:] = MAE
    baseline_method.baseline_CA[i,:] = CA
    save_predictions_and_transfers(folder_path, 'greedy', i, Y_pred_df, instances_pool_baseline, targets_pool_baseline)

    # Plot the results
    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_R2[i,:-1], upperbound_method.upperbound_R2[i,:-1], lowerbound_method.random_R2[i,:-1], baseline_method.baseline_R2[i,:-1], qbcrf_method.qbcrf_R2[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF'], 
                          f'Instance based QBC method R2 performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'R2 score', fig_path / 'r2', f'r2_score_{i+1}')

    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_MSE[i,:-1], upperbound_method.upperbound_MSE[i,:-1], lowerbound_method.random_MSE[i,:-1], baseline_method.baseline_MSE[i,:-1], qbcrf_method.qbcrf_MSE[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF'], 
                          f'Instance based QBC method MSE performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'MSE', fig_path / 'mse', f'mse_{i+1}')

    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_MAE[i,:-1], upperbound_method.upperbound_MAE[i,:-1], lowerbound_method.random_MAE[i,:-1], baseline_method.baseline_MAE[i,:-1], qbcrf_method.qbcrf_MAE[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF'], 
                          f'Instance based QBC method MAE performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'MAE', fig_path / 'mae', f'mae_{i+1}')

    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_CA[i,:-1], upperbound_method.upperbound_CA[i,:-1], lowerbound_method.random_CA[i,:-1], baseline_method.baseline_CA[i,:-1], qbcrf_method.qbcrf_CA[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF'], 
                          f'Instance based QBC method CA performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'CA', fig_path / 'ca', f'ca_{i+1}')

# Calculate and plot mean performances
instance_mean_R2, instance_mean_MSE, instance_mean_MAE, instance_mean_CA = calculate_mean_performances(proposed_method_instance, 'instance')
upperbound_mean_R2, upperbound_mean_MSE, upperbound_mean_MAE, upperbound_mean_CA = calculate_mean_performances(upperbound_method, 'upperbound')
random_mean_R2, random_mean_MSE, random_mean_MAE, random_mean_CA = calculate_mean_performances(lowerbound_method, 'random')
greedy_mean_R2, greedy_mean_MSE, greedy_mean_MAE, greedy_mean_CA = calculate_mean_performances(baseline_method, 'baseline')
qbcrf_mean_R2, qbcrf_mean_MSE, qbcrf_mean_MAE, qbcrf_mean_CA = calculate_mean_performances(qbcrf_method, 'qbcrf')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_R2[:-1], upperbound_mean_R2[:-1], random_mean_R2[:-1], greedy_mean_R2[:-1], qbcrf_mean_R2[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF'], 
                      f'Instance based QBC method R2 performance for {Method.dataset_name} dataset', 
                      'R2 score', fig_path / 'r2', 'r2_all')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_MSE[:-1], upperbound_mean_MSE[:-1], random_mean_MSE[:-1], greedy_mean_MSE[:-1], qbcrf_mean_MSE[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF'], 
                      f'Instance based QBC method MSE performance for {Method.dataset_name} dataset', 
                      'MSE', fig_path / 'mse', 'mse_all')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_MAE[:-1], upperbound_mean_MAE[:-1], random_mean_MAE[:-1], greedy_mean_MAE[:-1], qbcrf_mean_MAE[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF'], 
                      f'Instance based QBC method MAE performance for {Method.dataset_name} dataset', 
                      'MAE', fig_path / 'mae', 'mae_all')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_CA[:-1], upperbound_mean_CA[:-1], random_mean_CA[:-1], greedy_mean_CA[:-1], qbcrf_mean_CA[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF'], 
                      f'Instance based QBC method CA performance for {Method.dataset_name} dataset', 
                      'CA', fig_path / 'ca', 'ca_all')

# Append the average performance to the iteration performances
instance_total_R2 = append_mean_performances(proposed_method_instance.instance_R2, instance_mean_R2)
instance_total_MSE = append_mean_performances(proposed_method_instance.instance_MSE, instance_mean_MSE)
instance_total_MAE = append_mean_performances(proposed_method_instance.instance_MAE, instance_mean_MAE)
instance_total_CA = append_mean_performances(proposed_method_instance.instance_CA, instance_mean_CA)

upperbound_total_R2 = append_mean_performances(upperbound_method.upperbound_R2, upperbound_mean_R2)
upperbound_total_MSE = append_mean_performances(upperbound_method.upperbound_MSE, upperbound_mean_MSE)
upperbound_total_MAE = append_mean_performances(upperbound_method.upperbound_MAE, upperbound_mean_MAE)
upperbound_total_CA = append_mean_performances(upperbound_method.upperbound_CA, upperbound_mean_CA)

random_total_R2 = append_mean_performances(lowerbound_method.random_R2, random_mean_R2)
random_total_MSE = append_mean_performances(lowerbound_method.random_MSE, random_mean_MSE)
random_total_MAE = append_mean_performances(lowerbound_method.random_MAE, random_mean_MAE)
random_total_CA = append_mean_performances(lowerbound_method.random_CA, random_mean_CA)

greedy_total_R2 = append_mean_performances(baseline_method.baseline_R2, greedy_mean_R2)
greedy_total_MSE = append_mean_performances(baseline_method.baseline_MSE, greedy_mean_MSE)
greedy_total_MAE = append_mean_performances(baseline_method.baseline_MAE, greedy_mean_MAE)
greedy_total_CA = append_mean_performances(baseline_method.baseline_CA, greedy_mean_CA)

qbcrf_total_R2 = append_mean_performances(qbcrf_method.qbcrf_R2, qbcrf_mean_R2)
qbcrf_total_MSE = append_mean_performances(qbcrf_method.qbcrf_MSE, qbcrf_mean_MSE)
qbcrf_total_MAE = append_mean_performances(qbcrf_method.qbcrf_MAE, qbcrf_mean_MAE)
qbcrf_total_CA = append_mean_performances(qbcrf_method.qbcrf_CA, qbcrf_mean_CA)

# Save total performances to CSV
save_total_performances_to_csv(folder_path, 'instance', [instance_total_R2, instance_total_MSE, instance_total_MAE, instance_total_CA], ['r2', 'mse', 'mae', 'ca'])
save_total_performances_to_csv(folder_path, 'upperbound', [upperbound_total_R2, upperbound_total_MSE, upperbound_total_MAE, upperbound_total_CA], ['r2', 'mse', 'mae', 'ca'])
save_total_performances_to_csv(folder_path, 'random', [random_total_R2, random_total_MSE, random_total_MAE, random_total_CA], ['r2', 'mse', 'mae', 'ca'])
save_total_performances_to_csv(folder_path, 'greedy', [greedy_total_R2, greedy_total_MSE, greedy_total_MAE, greedy_total_CA], ['r2', 'mse', 'mae', 'ca'])
save_total_performances_to_csv(folder_path, 'qbcrf', [qbcrf_total_R2, qbcrf_total_MSE, qbcrf_total_MAE, qbcrf_total_CA], ['r2', 'mse', 'mae', 'ca'])

print("Instance based results:")
print(instance_total_R2)
print("Upperbound based results:")
print(upperbound_total_R2)
print("Random based results:")
print(random_total_R2)
print("Greedy based results:")
print(greedy_total_R2)
print("QBC-RF based results:")
print(qbcrf_total_R2)