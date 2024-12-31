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
from models.rtal import *

# Main script
data_dir = DATA_DIR
dataset = DATASET_NAME
Method = activelearning(dataset)


# Helper function to create directories
def create_directories(base_path, sub_dirs):
    for sub_dir in sub_dirs:
        path = base_path / sub_dir
        if not path.exists():
            path.mkdir(parents=True)

# Helper function to save predictions and transfer instances
def save_predictions_and_transfers(folder_path, prefix, i, Y_pred_df, instances_pool, targets_pool):
    try:
        ypred_path = folder_path / 'preds' / f"{prefix}_preds_{i}.csv"
        Y_pred_df.to_csv(ypred_path)
        instances_pool_df = pd.DataFrame(instances_pool)
        targets_pool_df = pd.DataFrame(targets_pool)
        instances_pool_path = folder_path / f'transfer_instances_{prefix}_{i}.csv'
        targets_pool_path = folder_path / f'transfer_targets_{prefix}_{i}.csv'
        instances_pool_df.to_csv(instances_pool_path)
        targets_pool_df.to_csv(targets_pool_path)
    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"An error occurred while saving files: {e}")

# Helper function to plot and save figures
def plot_and_save_figures(epochs, metrics, labels, title, ylabel, fig_path, filename):
    print(f"Plotting and saving figure: {filename}")
    print(f"Figure path: {fig_path / filename}")
    for metric, label in zip(metrics, labels):
        plt.plot(epochs, metric, label=label)
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fig_path / filename)  # Save the figure
    plt.close()  # Close the figure to avoid displaying it

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
    mean_aRRMSE = np.mean(getattr(method, f"{submethod}_aRRMSE"), axis=0)
    return mean_R2, mean_MSE, mean_MAE, mean_CA, mean_aRRMSE

# Helper function to append mean performances to total performances
def append_mean_performances(total_performances, mean_performances):
    return np.append(total_performances, np.reshape(mean_performances, (1, len(mean_performances))), axis=0)

# Helper function to save total performances to CSV
def save_total_performances_to_csv(folder_path, method_name, total_performances, metrics):
    cols = [f'Epoch {i+1}' for i in range(N_EPOCHS)] + ['AUC']
    rows = [f'Iteration {i+1}' for i in range(Method.iterations)] + ['Average']
    for metric, performance in zip(metrics, total_performances):
        df = pd.DataFrame(performance, index=rows, columns=cols)  # Define the DataFrame
        df.to_csv(folder_path / metric / f'{method_name}_{metric}.csv', header=cols)

def concatenate_percentage_files(folder_path, method_name, n_iterations, output_file):
    all_data = []

    for i in range(n_iterations):
        file_path = folder_path / 'target_coverage' / f'percentage_targets_provided_{method_name}_{i+1}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'Epoch' in df.columns:
                df = df.drop(columns=['Epoch'])
            df.columns = [f'Iteration {i+1}']
            all_data.append(df)
        else:
            print(f"File {file_path} does not exist.")

    if all_data:
        concatenated_df = pd.concat(all_data, axis=1)
        concatenated_df.to_csv(folder_path / 'target_coverage' / output_file, index=True)
        print(f"Concatenated file saved as {output_file}")
    else:
        print("No data to concatenate.")