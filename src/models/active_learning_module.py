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

# Ensure the directory structure exists
def ensure_directory_exists(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

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
create_directories(folder_path, ['r2', 'mse', 'mae', 'ca', 'arrmse','preds', 'target_coverage', 'transfer'])
create_directories(fig_path, ['r2', 'mse', 'mae', 'ca', 'arrmse'])

for i in range(Method.iterations):
    if i > 0:
        X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length = read_data(Method, i+1)

    # Copy the original datasets for the instance based, random and greedy method
    X_train_instance, X_pool_instance, y_train_instance, y_pool_instance = copy_datasets(X_train, X_pool, y_train, y_pool)
    X_train_random, X_pool_random, y_train_random, y_pool_random = copy_datasets(X_train, X_pool, y_train, y_pool)
    X_train_baseline, X_pool_baseline, y_train_baseline, y_pool_baseline = copy_datasets(X_train, X_pool, y_train, y_pool)
    X_train_qbcrf, X_pool_qbcrf, y_train_qbcrf, y_pool_qbcrf = copy_datasets(X_train, X_pool, y_train, y_pool)
    X_train_rtal, X_pool_rtal, y_train_rtal, y_pool_rtal = copy_datasets(X_train, X_pool, y_train, y_pool)
    
    print("\n" + 50*"-" + f"Iteration {i+1}" + 50*"-" + "\n")

    # Ensure directories exist before saving files
    required_dirs = ['transfer', 'preds', 'target_coverage']
    for dir_name in required_dirs:
        ensure_directory_exists(folder_path / dir_name)

    # RTAL method
    print("\n" + 50*"-" + "RT-AL method" + 50*"-" + "\n")
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

    # Metrics
    metrics = [
        ('R2', 'r2', proposed_method_instance.instance_R2, upperbound_method.upperbound_R2, lowerbound_method.random_R2, baseline_method.baseline_R2, qbcrf_method.qbcrf_R2, rtal_method.rtal_R2),
        ('MSE', 'mse', proposed_method_instance.instance_MSE, upperbound_method.upperbound_MSE, lowerbound_method.random_MSE, baseline_method.baseline_MSE, qbcrf_method.qbcrf_MSE, rtal_method.rtal_MSE),
        ('MAE', 'mae', proposed_method_instance.instance_MAE, upperbound_method.upperbound_MAE, lowerbound_method.random_MAE, baseline_method.baseline_MAE, qbcrf_method.qbcrf_MAE, rtal_method.rtal_MAE),
        ('CA', 'ca', proposed_method_instance.instance_CA, upperbound_method.upperbound_CA, lowerbound_method.random_CA, baseline_method.baseline_CA, qbcrf_method.qbcrf_CA, rtal_method.rtal_CA),
        ('aRRMSE', 'arrmse', proposed_method_instance.instance_aRRMSE, upperbound_method.upperbound_aRRMSE, lowerbound_method.random_aRRMSE, baseline_method.baseline_aRRMSE, qbcrf_method.qbcrf_aRRMSE, rtal_method.rtal_aRRMSE)
    ]

    for metric_name, metric_short, *metric_values in metrics:
        plot_and_save_figures(
            proposed_method_instance.epochs,
            [values[i, :-1] for values in metric_values],
            ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'],
            f'Instance based QBC method {metric_name} performance for {Method.dataset_name} dataset: iteration {i+1}',
            metric_name,
            fig_path / metric_short,
            f'{metric_short}_score_{i+1}'
        )

# Calculate and plot mean performances
instance_mean_R2, instance_mean_MSE, instance_mean_MAE, instance_mean_CA, instance_mean_ARRMSE = calculate_mean_performances(proposed_method_instance, 'instance')
upperbound_mean_R2, upperbound_mean_MSE, upperbound_mean_MAE, upperbound_mean_CA, upperbound_mean_ARRMSE = calculate_mean_performances(upperbound_method, 'upperbound')
random_mean_R2, random_mean_MSE, random_mean_MAE, random_mean_CA, random_mean_ARRMSE = calculate_mean_performances(lowerbound_method, 'random')
greedy_mean_R2, greedy_mean_MSE, greedy_mean_MAE, greedy_mean_CA, greedy_mean_ARRMSE = calculate_mean_performances(baseline_method, 'baseline')
qbcrf_mean_R2, qbcrf_mean_MSE, qbcrf_mean_MAE, qbcrf_mean_CA, qbcrf_mean_ARRMSE = calculate_mean_performances(qbcrf_method, 'qbcrf')
rtal_mean_R2, rtal_mean_MSE, rtal_mean_MAE, rtal_mean_CA, rtal_mean_ARRMSE = calculate_mean_performances(rtal_method, 'rtal')

metrics = [
    ('R2', 'r2', instance_mean_R2, upperbound_mean_R2, random_mean_R2, greedy_mean_R2, qbcrf_mean_R2, rtal_mean_R2),
    ('MSE', 'mse', instance_mean_MSE, upperbound_mean_MSE, random_mean_MSE, greedy_mean_MSE, qbcrf_mean_MSE, rtal_mean_MSE),
    ('MAE', 'mae', instance_mean_MAE, upperbound_mean_MAE, random_mean_MAE, greedy_mean_MAE, qbcrf_mean_MAE, rtal_mean_MAE),
    ('CA', 'ca', instance_mean_CA, upperbound_mean_CA, random_mean_CA, greedy_mean_CA, qbcrf_mean_CA, rtal_mean_CA),
    ('aRRMSE', 'arrmse', instance_mean_ARRMSE, upperbound_mean_ARRMSE, random_mean_ARRMSE, greedy_mean_ARRMSE, qbcrf_mean_ARRMSE, rtal_mean_ARRMSE)
]

for metric_name, metric_short, *metric_values in metrics:
    plot_and_save_figures(
        proposed_method_instance.epochs,
        [values[:-1] for values in metric_values],
        ['Instance based QBC', 'Upper bound', 'Random sampling', 'Greedy sampling', 'QBC-RF', 'RT-AL'],
        f'{metric_name} performance for {Method.dataset_name} dataset',
        metric_name,
        fig_path / metric_short,
        f'{metric_short}_all'
    )

# Append the average performance to the iteration performances
methods_and_means = [
    (proposed_method_instance, 'instance', instance_mean_R2, instance_mean_MSE, instance_mean_MAE, instance_mean_CA, instance_mean_ARRMSE),
    (upperbound_method, 'upperbound', upperbound_mean_R2, upperbound_mean_MSE, upperbound_mean_MAE, upperbound_mean_CA, upperbound_mean_ARRMSE),
    (lowerbound_method, 'random', random_mean_R2, random_mean_MSE, random_mean_MAE, random_mean_CA, random_mean_ARRMSE),
    (baseline_method, 'baseline', greedy_mean_R2, greedy_mean_MSE, greedy_mean_MAE, greedy_mean_CA, greedy_mean_ARRMSE),
    (qbcrf_method, 'qbcrf', qbcrf_mean_R2, qbcrf_mean_MSE, qbcrf_mean_MAE, qbcrf_mean_CA, qbcrf_mean_ARRMSE),
    (rtal_method, 'rtal', rtal_mean_R2, rtal_mean_MSE, rtal_mean_MAE, rtal_mean_CA, rtal_mean_ARRMSE)
]

total_performances = []

for method, submethod, mean_R2, mean_MSE, mean_MAE, mean_CA, mean_ARRMSE in methods_and_means:
    total_R2 = append_mean_performances(getattr(method, f"{submethod}_R2"), mean_R2)
    total_MSE = append_mean_performances(getattr(method, f"{submethod}_MSE"), mean_MSE)
    total_MAE = append_mean_performances(getattr(method, f"{submethod}_MAE"), mean_MAE)
    total_CA = append_mean_performances(getattr(method, f"{submethod}_CA"), mean_CA)
    total_ARRMSE = append_mean_performances(getattr(method, f"{submethod}_aRRMSE"), mean_ARRMSE)
    total_performances.append((total_R2, total_MSE, total_MAE, total_CA, total_ARRMSE))

# Save total performances to CSV
method_names = ['instance', 'upperbound', 'random', 'greedy', 'qbcrf', 'rtal']
metrics_short = ['r2', 'mse', 'mae', 'ca', 'arrmse']

for method_name, (total_R2, total_MSE, total_MAE, total_CA, total_ARRMSE) in zip(method_names, total_performances):
    save_total_performances_to_csv(folder_path, method_name, [total_R2, total_MSE, total_MAE, total_CA, total_ARRMSE], metrics_short)

# Save target coverage to CSV
concatenate_percentage_files(folder_path, 'instance', Method.iterations, 'percentage_targets_instance.csv')
concatenate_percentage_files(folder_path, 'baseline', Method.iterations, 'percentage_targets_baseline.csv')
concatenate_percentage_files(folder_path, 'qbcrf', Method.iterations, 'percentage_targets_qbcrf.csv')
concatenate_percentage_files(folder_path, 'rtal', Method.iterations, 'percentage_targets_rtal.csv')

print('End')