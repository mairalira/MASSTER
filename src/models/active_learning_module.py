""" This script contains code of the active learning algorithms for multi-target regression. """ 

# importing the libraries
import csv
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, pairwise_distances_argmin_min, auc
from sklearn.cluster import KMeans
from sklearn import preprocessing 
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"  # Set the environment variable to avoid memory leak

# Suppress the specific KMeans memory leak warning
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent
# Adding path to sys.path   
sys.path.append(str(project_root))

from utils.data_handling import *
from utils.aux_active import *
from models.active_learning import *
from models.instance_based import *
from models.upperbound import *
from models.randomsampling import *
from models.greedy_sampling import *
from models.qbcrf import *
from models.rtal import *

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
rtal_method = rtal(batch_size, n_epochs)

# make the folders to store the results
folder_path = Path('reports') / 'active_learning' / f'{dataset}'
fig_path = folder_path / 'images'
create_directories(folder_path, ['r2', 'mse', 'mae', 'ca', 'arrmse','preds', 'target_coverage'])
create_directories(fig_path, ['r2', 'mse', 'mae', 'ca', 'arrmse'])

for i in range(Method.iterations):
    if i > 0:
        X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length = read_data(Method, i+1)

    # Copy the original datasets for the instance based, random and greedy method
    X_train_instance, X_pool_instance, y_train_instance, y_pool_instance = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_random, X_pool_random, y_train_random, y_pool_random = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_baseline, X_pool_baseline, y_train_baseline, y_pool_baseline = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_qbcrf, X_pool_qbcrf, y_train_qbcrf, y_pool_qbcrf = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_rtal, X_pool_rtal, y_train_rtal, y_pool_rtal = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    
    print("\n" + 50*"-" + f"Iteration {i+1}" + 50*"-" + "\n")

    # RTAL method
    print("\n" + 50*"-" + "RTAL method" + 50*"-" + "\n")
    R2, MSE, MAE, CA, ARRMSE, Y_pred_df, instances_pool_rtal, targets_pool_rtal, percentage_targets_provided_rtal = rtal_method.training(X_train_rtal, X_pool_rtal, X_test, y_train_rtal, y_pool_rtal, y_test, target_length)
    rtal_method.rtal_R2[i,:] = R2
    rtal_method.rtal_MSE[i,:] = MSE
    rtal_method.rtal_MAE[i,:] = MAE
    rtal_method.rtal_CA[i,:] = CA
    rtal_method.rtal_aRRMSE[i,:] = ARRMSE
    save_predictions_and_transfers(folder_path, 'rtal', i, Y_pred_df, instances_pool_rtal, targets_pool_rtal)

    # Save percentage targets provided by epoch for RTAL method
    percentage_targets_provided_rtal_df = pd.DataFrame(percentage_targets_provided_rtal, columns=['Percentage Targets Provided'])
    percentage_targets_provided_rtal_df.index.name = 'Epoch'
    percentage_targets_provided_rtal_df.to_csv(folder_path / 'target_coverage' / f'percentage_targets_provided_rtal_{i+1}.csv')

    # QBC-RF method
    print("\n" + 50*"-" + "QBC-RF method" + 50*"-" + "\n")
    R2, MSE, MAE, CA, ARRMSE, Y_pred_df, instances_pool_qbcrf, targets_pool_qbcrf, percentage_targets_provided_qbcrf = qbcrf_method.training(X_train_qbcrf, X_pool_qbcrf, X_test, y_train_qbcrf, y_pool_qbcrf, y_test, target_length)
    qbcrf_method.qbcrf_R2[i,:] = R2
    qbcrf_method.qbcrf_MSE[i,:] = MSE
    qbcrf_method.qbcrf_MAE[i,:] = MAE
    qbcrf_method.qbcrf_CA[i,:] = CA
    qbcrf_method.qbcrf_aRRMSE[i,:] = ARRMSE
    save_predictions_and_transfers(folder_path, 'qbcrf', i, Y_pred_df, instances_pool_qbcrf, targets_pool_qbcrf)

    # Save percentage targets provided by epoch for QBC-RF method
    percentage_targets_provided_qbcrf_df = pd.DataFrame(percentage_targets_provided_qbcrf, columns=['Percentage Targets Provided'])
    percentage_targets_provided_qbcrf_df.index.name = 'Epoch'
    percentage_targets_provided_qbcrf_df.to_csv(folder_path / 'target_coverage' / f'percentage_targets_provided_qbcrf_{i+1}.csv')

    # Instance based    
    print("\n" + 50*"-" + "Instance based method" + 50*"-" + "\n")
    cols = [f"Target_{i+1}" for epoch in range(n_epochs) for i in range(target_length)]
    R2, MSE, MAE, CA, ARRMSE, Y_pred_df, instances_pool_qbc, targets_pool_qbc, percentage_targets_provided_instance = proposed_method_instance.training(X_train_instance, X_pool_instance, X_test, y_train_instance, y_pool_instance, y_test, target_length)
    proposed_method_instance.instance_R2[i,:] = R2
    proposed_method_instance.instance_MSE[i,:] = MSE
    proposed_method_instance.instance_MAE[i,:] = MAE
    proposed_method_instance.instance_CA[i,:] = CA
    proposed_method_instance.instance_aRRMSE[i,:] = ARRMSE
    save_predictions_and_transfers(folder_path, 'instance', i, Y_pred_df, instances_pool_qbc, targets_pool_qbc)

    # Save percentage targets provided by epoch for Instance based method
    percentage_targets_provided_instance_df = pd.DataFrame(percentage_targets_provided_instance, columns=['Percentage Targets Provided'])
    percentage_targets_provided_instance_df.index.name = 'Epoch'
    percentage_targets_provided_instance_df.to_csv(folder_path / 'target_coverage' / f'percentage_targets_provided_instance_{i+1}.csv')

    # Upperbound 
    print("\n" + 50*"-" + "Upperbound method" + 50*"-" + "\n")
    cols = [f"Target_{i+1}" for i in range(target_length)]
    R2, MSE, MAE, CA, ARRMSE, Y_pred_df = upperbound_method.training(X_rest, X_test, y_rest, y_test, target_length)
    upperbound_method.upperbound_R2[i,:] = R2
    upperbound_method.upperbound_MSE[i,:] = MSE
    upperbound_method.upperbound_MAE[i,:] = MAE
    upperbound_method.upperbound_CA[i,:] = CA
    upperbound_method.upperbound_aRRMSE[i,:] = ARRMSE
    ypred_upper_path = folder_path / 'preds' /  f"upperbound_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_upper_path)

    # Random sampling
    print("\n" + 50*"-" + "Lowerbound method" + 50*"-" + "\n")
    cols = [f"Target_{i+1}" for epoch in range(n_epochs) for i in range(target_length)]
    R2, MSE, MAE, CA, ARRMSE, Y_pred_df = lowerbound_method.training(X_train_random, X_pool_random, X_test, y_train_random, y_pool_random, y_test, target_length)
    lowerbound_method.random_R2[i,:] = R2
    lowerbound_method.random_MSE[i,:] = MSE
    lowerbound_method.random_MAE[i,:] = MAE
    lowerbound_method.random_CA[i,:] = CA
    lowerbound_method.random_aRRMSE[i,:] = ARRMSE
    ypred_lower_path = folder_path / 'preds' / f"random_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_lower_path)

    # Greedy sampling
    print("\n" + 50*"-" + "Baseline method" + 50*"-" + "\n")
    R2, MSE, MAE, CA, ARRMSE, Y_pred_df, instances_pool_baseline, targets_pool_baseline, percentage_targets_provided_baseline = baseline_method.training(X_train_baseline, X_pool_baseline, X_test, y_train_baseline, y_pool_baseline, y_test, target_length)
    baseline_method.baseline_R2[i,:] = R2
    baseline_method.baseline_MSE[i,:] = MSE
    baseline_method.baseline_MAE[i,:] = MAE
    baseline_method.baseline_CA[i,:] = CA
    baseline_method.baseline_aRRMSE[i,:] = ARRMSE
    save_predictions_and_transfers(folder_path, 'greedy', i, Y_pred_df, instances_pool_baseline, targets_pool_baseline)

    # Save percentage targets provided by epoch for Greedy method
    percentage_targets_provided_baseline_df = pd.DataFrame(percentage_targets_provided_baseline, columns=['Percentage Targets Provided'])
    percentage_targets_provided_baseline_df.index.name = 'Epoch'
    percentage_targets_provided_baseline_df.to_csv(folder_path / 'target_coverage' / f'percentage_targets_provided_baseline_{i+1}.csv')

    # Plot the results
    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_R2[i,:-1], upperbound_method.upperbound_R2[i,:-1], lowerbound_method.random_R2[i,:-1], baseline_method.baseline_R2[i,:-1], qbcrf_method.qbcrf_R2[i,:-1], rtal_method.rtal_R2[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                          f'Instance based QBC method R2 performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'R2 score', fig_path / 'r2', f'r2_score_{i+1}')

    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_MSE[i,:-1], upperbound_method.upperbound_MSE[i,:-1], lowerbound_method.random_MSE[i,:-1], baseline_method.baseline_MSE[i,:-1], qbcrf_method.qbcrf_MSE[i,:-1], rtal_method.rtal_MSE[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                          f'Instance based QBC method MSE performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'MSE', fig_path / 'mse', f'mse_{i+1}')

    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_MAE[i,:-1], upperbound_method.upperbound_MAE[i,:-1], lowerbound_method.random_MAE[i,:-1], baseline_method.baseline_MAE[i,:-1], qbcrf_method.qbcrf_MAE[i,:-1], rtal_method.rtal_MAE[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                          f'Instance based QBC method MAE performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'MAE', fig_path / 'mae', f'mae_{i+1}')

    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_CA[i,:-1], upperbound_method.upperbound_CA[i,:-1], lowerbound_method.random_CA[i,:-1], baseline_method.baseline_CA[i,:-1], qbcrf_method.qbcrf_CA[i,:-1], rtal_method.rtal_CA[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                          f'Instance based QBC method CA performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'CA', fig_path / 'ca', f'ca_{i+1}')
    
    plot_and_save_figures(proposed_method_instance.epochs, 
                          [proposed_method_instance.instance_aRRMSE[i,:-1], upperbound_method.upperbound_aRRMSE[i,:-1], lowerbound_method.random_aRRMSE[i,:-1], baseline_method.baseline_aRRMSE[i,:-1], qbcrf_method.qbcrf_aRRMSE[i,:-1], rtal_method.rtal_aRRMSE[i,:-1]], 
                          ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                          f'Instance based QBC method aRRMSE performance for {Method.dataset_name} dataset: iteration {i+1}', 
                          'aRRMSE', fig_path / 'arrmse', f'arrmse_{i+1}')

# Calculate and plot mean performances
instance_mean_R2, instance_mean_MSE, instance_mean_MAE, instance_mean_CA, instance_mean_ARRMSE = calculate_mean_performances(proposed_method_instance, 'instance')
upperbound_mean_R2, upperbound_mean_MSE, upperbound_mean_MAE, upperbound_mean_CA, upperbound_mean_ARRMSE = calculate_mean_performances(upperbound_method, 'upperbound')
random_mean_R2, random_mean_MSE, random_mean_MAE, random_mean_CA, random_mean_ARRMSE = calculate_mean_performances(lowerbound_method, 'random')
greedy_mean_R2, greedy_mean_MSE, greedy_mean_MAE, greedy_mean_CA, greedy_mean_ARRMSE = calculate_mean_performances(baseline_method, 'baseline')
qbcrf_mean_R2, qbcrf_mean_MSE, qbcrf_mean_MAE, qbcrf_mean_CA, qbcrf_mean_ARRMSE = calculate_mean_performances(qbcrf_method, 'qbcrf')
rtal_mean_R2, rtal_mean_MSE, rtal_mean_MAE, rtal_mean_CA, rtal_mean_ARRMSE = calculate_mean_performances(rtal_method, 'rtal')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_R2[:-1], upperbound_mean_R2[:-1], random_mean_R2[:-1], greedy_mean_R2[:-1], qbcrf_mean_R2[:-1], rtal_mean_R2[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                      f'Instance based QBC method R2 performance for {Method.dataset_name} dataset', 
                      'R2 score', fig_path / 'r2', 'r2_all')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_MSE[:-1], upperbound_mean_MSE[:-1], random_mean_MSE[:-1], greedy_mean_MSE[:-1], qbcrf_mean_MSE[:-1], rtal_mean_MSE[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                      f'Instance based QBC method MSE performance for {Method.dataset_name} dataset', 
                      'MSE', fig_path / 'mse', 'mse_all')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_MAE[:-1], upperbound_mean_MAE[:-1], random_mean_MAE[:-1], greedy_mean_MAE[:-1], qbcrf_mean_MAE[:-1], rtal_mean_MAE[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                      f'Instance based QBC method MAE performance for {Method.dataset_name} dataset', 
                      'MAE', fig_path / 'mae', 'mae_all')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_CA[:-1], upperbound_mean_CA[:-1], random_mean_CA[:-1], greedy_mean_CA[:-1], qbcrf_mean_CA[:-1], rtal_mean_CA[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                      f'Instance based QBC method CA performance for {Method.dataset_name} dataset', 
                      'CA', fig_path / 'ca', 'ca_all')

plot_and_save_figures(proposed_method_instance.epochs, 
                      [instance_mean_ARRMSE[:-1], upperbound_mean_ARRMSE[:-1], random_mean_ARRMSE[:-1], greedy_mean_ARRMSE[:-1], qbcrf_mean_ARRMSE[:-1], rtal_mean_ARRMSE[:-1]], 
                      ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'], 
                      f'Instance based QBC method aRRMSE performance for {Method.dataset_name} dataset', 
                      'aRRMSE', fig_path / 'arrmse', 'arrmse_all')

# Append the average performance to the iteration performances
instance_total_R2 = append_mean_performances(proposed_method_instance.instance_R2, instance_mean_R2)
instance_total_MSE = append_mean_performances(proposed_method_instance.instance_MSE, instance_mean_MSE)
instance_total_MAE = append_mean_performances(proposed_method_instance.instance_MAE, instance_mean_MAE)
instance_total_CA = append_mean_performances(proposed_method_instance.instance_CA, instance_mean_CA)
instance_total_ARRMSE = append_mean_performances(proposed_method_instance.instance_aRRMSE, instance_mean_ARRMSE)

upperbound_total_R2 = append_mean_performances(upperbound_method.upperbound_R2, upperbound_mean_R2)
upperbound_total_MSE = append_mean_performances(upperbound_method.upperbound_MSE, upperbound_mean_MSE)
upperbound_total_MAE = append_mean_performances(upperbound_method.upperbound_MAE, upperbound_mean_MAE)
upperbound_total_CA = append_mean_performances(upperbound_method.upperbound_CA, upperbound_mean_CA)
upperbound_total_ARRMSE = append_mean_performances(upperbound_method.upperbound_aRRMSE, upperbound_mean_ARRMSE)

random_total_R2 = append_mean_performances(lowerbound_method.random_R2, random_mean_R2)
random_total_MSE = append_mean_performances(lowerbound_method.random_MSE, random_mean_MSE)
random_total_MAE = append_mean_performances(lowerbound_method.random_MAE, random_mean_MAE)
random_total_CA = append_mean_performances(lowerbound_method.random_CA, random_mean_CA)
random_total_ARRMSE = append_mean_performances(lowerbound_method.random_aRRMSE, random_mean_ARRMSE)

greedy_total_R2 = append_mean_performances(baseline_method.baseline_R2, greedy_mean_R2)
greedy_total_MSE = append_mean_performances(baseline_method.baseline_MSE, greedy_mean_MSE)
greedy_total_MAE = append_mean_performances(baseline_method.baseline_MAE, greedy_mean_MAE)
greedy_total_CA = append_mean_performances(baseline_method.baseline_CA, greedy_mean_CA)
greedy_total_ARRMSE = append_mean_performances(baseline_method.baseline_aRRMSE, greedy_mean_ARRMSE)

qbcrf_total_R2 = append_mean_performances(qbcrf_method.qbcrf_R2, qbcrf_mean_R2)
qbcrf_total_MSE = append_mean_performances(qbcrf_method.qbcrf_MSE, qbcrf_mean_MSE)
qbcrf_total_MAE = append_mean_performances(qbcrf_method.qbcrf_MAE, qbcrf_mean_MAE)
qbcrf_total_CA = append_mean_performances(qbcrf_method.qbcrf_CA, qbcrf_mean_CA)
qbcrf_total_ARRMSE = append_mean_performances(qbcrf_method.qbcrf_aRRMSE, qbcrf_mean_ARRMSE)

rtal_total_R2 = append_mean_performances(rtal_method.rtal_R2, rtal_mean_R2)
rtal_total_MSE = append_mean_performances(rtal_method.rtal_MSE, rtal_mean_MSE)
rtal_total_MAE = append_mean_performances(rtal_method.rtal_MAE, rtal_mean_MAE)
rtal_total_CA = append_mean_performances(rtal_method.rtal_CA, rtal_mean_CA)
rtal_total_ARRMSE = append_mean_performances(rtal_method.rtal_aRRMSE, rtal_mean_ARRMSE)

# Save total performances to CSV
save_total_performances_to_csv(folder_path, 'instance', [instance_total_R2, instance_total_MSE, instance_total_MAE, instance_total_CA, instance_total_ARRMSE], ['r2', 'mse', 'mae', 'ca', 'arrmse'])
save_total_performances_to_csv(folder_path, 'upperbound', [upperbound_total_R2, upperbound_total_MSE, upperbound_total_MAE, upperbound_total_CA, upperbound_total_ARRMSE], ['r2', 'mse', 'mae', 'ca', 'arrmse'])
save_total_performances_to_csv(folder_path, 'random', [random_total_R2, random_total_MSE, random_total_MAE, random_total_CA, random_total_ARRMSE], ['r2', 'mse', 'mae', 'ca', 'arrmse'])
save_total_performances_to_csv(folder_path, 'greedy', [greedy_total_R2, greedy_total_MSE, greedy_total_MAE, greedy_total_CA, greedy_total_ARRMSE], ['r2', 'mse', 'mae', 'ca', 'arrmse'])
save_total_performances_to_csv(folder_path, 'qbcrf', [qbcrf_total_R2, qbcrf_total_MSE, qbcrf_total_MAE, qbcrf_total_CA, qbcrf_total_ARRMSE], ['r2', 'mse', 'mae', 'ca', 'arrmse'])
save_total_performances_to_csv(folder_path, 'rtal', [rtal_total_R2, rtal_total_MSE, rtal_total_MAE, rtal_total_CA, rtal_total_ARRMSE], ['r2', 'mse', 'mae', 'ca', 'arrmse'])

# Save target coverage to CSV
concatenate_percentage_files(folder_path, 'instance', Method.iterations, 'percentage_targets_instance.csv')
concatenate_percentage_files(folder_path, 'baseline', Method.iterations, 'percentage_targets_baseline.csv')
concatenate_percentage_files(folder_path, 'qbcrf', Method.iterations, 'percentage_targets_qbcrf.csv')
concatenate_percentage_files(folder_path, 'rtal', Method.iterations, 'percentage_targets_rtal.csv')

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
print("RT-AL based results:")
print(rtal_total_R2)