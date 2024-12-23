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

""" The class for the active learning algorithm. """
from models.active_learning import *

""" The subclass for QBC-RF. """
from models.qbcrf import *

""" The subclass for the upper bound method. """
from models.upperbound import *

""" The subclass for random sampling. """
from models.randomsampling import *

""" The subclass for the baseline method. """
from models.greedy_sampling import *

""" The training loop for all the methods. """
data_dir = DATA_DIR
dataset = DATASET_NAME
Method = activelearning(dataset)

# read the first version of the datasets to define the batch size
X_train, y_train, _, _ = Method.data_read('train_{}'.format(1))
X_pool, y_pool, n_pool, target_length = Method.data_read('pool_{}'.format(1))
X_rest, y_rest, _, _ = Method.data_read('train+pool_{}'.format(1))
X_test, y_test, _, _ = Method.data_read('test_{}'.format(1))

batch_percentage = BATCH_PERCENTAGE
batch_size = round((batch_percentage/100) * len(X_pool)) 
n_epochs = N_EPOCHS

# define the different methods
proposed_method_instance = instancebased(batch_size, n_epochs)
upperbound_method = upperbound(n_epochs)
lowerbound_method = lowerbound(batch_size, n_epochs)
baseline_method = baseline(batch_size, n_epochs)

# make the folders to store the results
folder_path = Path('reports') / 'active_learning' / f'{dataset}'
fig_path = folder_path / 'images'

metrics = {'r2', 'mse', 'mae', 'preds'}

for metric in metrics:
    metric_path = folder_path / metric
    fig_metric_path = fig_path / metric
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    if not os.path.exists(fig_metric_path):
        os.makedirs(fig_metric_path)

for i in range(Method.iterations):
    if i > 0:
        X_train, y_train, _, _ = Method.data_read('train_{}'.format(i+1))
        X_pool, y_pool, n_pool, target_length = Method.data_read('pool_{}'.format(i+1))
        X_rest, y_rest, _, _ = Method.data_read('train+pool_{}'.format(i+1))
        X_test, y_test, _, _ = Method.data_read('test_{}'.format(i+1))

    # Copy the original datasets for the instance based, random and greedy method
    X_train_instance, X_pool_instance, y_train_instance, y_pool_instance = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_random, X_pool_random, y_train_random, y_pool_random = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    X_train_baseline, X_pool_baseline, y_train_baseline, y_pool_baseline = X_train.copy(), X_pool.copy(), y_train.copy(), y_pool.copy()
    
    print("\n" + 50*"-" + "Iteration {}".format(i+1) + 50*"-" + "\n")

    # QBC-RF    
    print("\n" + 50*"-" + "Instance based method" + 50*"-" + "\n")
    cols = ["Target_{}".format(i+1) for epoch in range(n_epochs) for i in range(target_length)]
    R2, MSE, MAE, Y_pred_df, instances_pool_qbc, targets_pool_qbc = proposed_method_instance.training(X_train_instance, X_pool_instance, X_test, y_train_instance, y_pool_instance, y_test, target_length)
    proposed_method_instance.instance_R2[i,:] = R2
    proposed_method_instance.instance_MSE[i,:] = MSE
    proposed_method_instance.instance_MAE[i,:] = MAE
    ypred_path = folder_path / 'preds' / f"instance_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_path, header=cols)
    instances_pool_qbc_df = pd.DataFrame(instances_pool_qbc)
    targets_pool_qbc_df = pd.DataFrame(targets_pool_qbc)
    instances_pool_qbc_path = folder_path / f'transfer_instances_qbc_{i}.csv'
    targets_pool_qbc_path = folder_path / f'transfer_targets_qbc_{i}.csv'

    instances_pool_qbc_df.to_csv(instances_pool_qbc_path)
    targets_pool_qbc_df.to_csv(targets_pool_qbc_path)

    # Upperbound 
    print("\n" + 50*"-" + "Upperbound method" + 50*"-" + "\n")
    cols = ["Target_{}".format(i+1) for i in range(target_length)]
    R2, MSE, MAE, Y_pred_df = upperbound_method.training(X_rest, X_test, y_rest, y_test, target_length)
    upperbound_method.upperbound_R2[i,:] = R2
    upperbound_method.upperbound_MSE[i,:] = MSE
    upperbound_method.upperbound_MAE[i,:] = MAE
    ypred_upper_path = folder_path / 'preds' /  f"upperbound_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_upper_path)

    # Random sampling
    print("\n" + 50*"-" + "Lowerbound method" + 50*"-" + "\n")
    cols = ["Target_{}".format(i+1) for epoch in range(n_epochs) for i in range(target_length)]
    R2, MSE, MAE, Y_pred_df = lowerbound_method.training(X_train_random, X_pool_random, X_test, y_train_random, y_pool_random, y_test, target_length)
    lowerbound_method.random_R2[i,:] = R2
    lowerbound_method.random_MSE[i,:] = MSE
    lowerbound_method.random_MAE[i,:] = MAE
    ypred_lower_path = folder_path / 'preds' / f"random_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_lower_path)

    # Greedy sampling
    print("\n" + 50*"-" + "Baseline method" + 50*"-" + "\n")
    R2, MSE, MAE, Y_pred_df, instances_pool_baseline, targets_pool_baseline = baseline_method.training(X_train_baseline, X_pool_baseline, X_test, y_train_baseline, y_pool_baseline, y_test, target_length)
    baseline_method.baseline_R2[i,:] = R2
    baseline_method.baseline_MSE[i,:] = MSE
    baseline_method.baseline_MAE[i,:] = MAE
    ypred_baseline_path = folder_path / 'preds' / f"greedy_preds_{i}.csv"
    Y_pred_df.to_csv(ypred_baseline_path)
    instances_pool_baseline_df = pd.DataFrame(instances_pool_baseline)
    targets_pool_baseline_df = pd.DataFrame(targets_pool_baseline)
    instances_pool_baseline_path = folder_path / f'transfer_instances_greedy_{i}.csv'
    targets_pool_baseline_path = folder_path / f'transfer_targets_greedy_{i}.csv'

    instances_pool_baseline_df.to_csv(instances_pool_baseline_path)
    targets_pool_baseline_df.to_csv(targets_pool_baseline_path)

    # Plot the results

    plt.plot(proposed_method_instance.epochs, proposed_method_instance.instance_R2[i,:-1],'b', label='Instance based QBC')
    plt.plot(proposed_method_instance.epochs, upperbound_method.upperbound_R2[i,:-1],'r', label='Upper bound')
    plt.plot(proposed_method_instance.epochs, lowerbound_method.random_R2[i,:-1],'g', label='Random sampling')
    plt.plot(proposed_method_instance.epochs, baseline_method.baseline_R2[i,:-1],'y', label='Greedy sampling')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('R2 score')
    plt.title('Instance based QBC method R2 performance for {} dataset: iteration {}'.format(Method.dataset_name, i+1))
    r2_fig_path = fig_path / 'r2' / f'r2_score_{i+1}'
    plt.savefig(r2_fig_path)
    plt.show()

    plt.plot(proposed_method_instance.epochs, proposed_method_instance.instance_MSE[i,:-1],'b', label='Instance based QBC')
    plt.plot(proposed_method_instance.epochs, upperbound_method.upperbound_MSE[i,:-1],'r', label='Upper bound')
    plt.plot(proposed_method_instance.epochs, lowerbound_method.random_MSE[i,:-1],'g', label='Random sampling')
    plt.plot(proposed_method_instance.epochs, baseline_method.baseline_MSE[i,:-1],'y', label='Greedy sampling')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('Instance based QBC method MSE performance for {} dataset: iteration {}'.format(Method.dataset_name, i+1))
    mse_fig_path = fig_path / 'mse' / f'mse_{i+1}'
    plt.savefig(mse_fig_path)
    plt.show()

    plt.plot(proposed_method_instance.epochs, proposed_method_instance.instance_MAE[i,:-1],'b', label='Instance based QBC')
    plt.plot(proposed_method_instance.epochs, upperbound_method.upperbound_MAE[i,:-1],'r', label='Upper bound')
    plt.plot(proposed_method_instance.epochs, lowerbound_method.random_MAE[i,:-1],'g', label='Random sampling')
    plt.plot(proposed_method_instance.epochs, baseline_method.baseline_MAE[i,:-1],'y', label='Greedy sampling')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.title('Instance based QBC method MAE performance for {} dataset: iteration {}'.format(Method.dataset_name, i+1))
    mae_fig_path = fig_path / 'mae' / f'mae_{i+1}'
    plt.savefig(mae_fig_path)
    plt.show()

instance_mean_R2, instance_mean_MSE, instance_mean_MAE = np.mean(proposed_method_instance.instance_R2, axis=0), np.mean(proposed_method_instance.instance_MSE, axis=0), np.mean(proposed_method_instance.instance_MAE, axis=0)
upperbound_mean_R2, upperbound_mean_MSE, upperbound_mean_MAE = np.mean(upperbound_method.upperbound_R2, axis=0), np.mean(upperbound_method.upperbound_MSE, axis=0), np.mean(upperbound_method.upperbound_MAE, axis=0)
random_mean_R2, random_mean_MSE, random_mean_MAE = np.mean(lowerbound_method.random_R2, axis=0), np.mean(lowerbound_method.random_MSE, axis=0), np.mean(lowerbound_method.random_MAE, axis=0)
greedy_mean_R2, greedy_mean_MSE, greedy_mean_MAE = np.mean(baseline_method.baseline_R2, axis=0), np.mean(baseline_method.baseline_MSE, axis=0), np.mean(baseline_method.baseline_MAE, axis=0)

plt.plot(proposed_method_instance.epochs, instance_mean_R2[:-1],'b', label='Instance based QBC')
plt.plot(proposed_method_instance.epochs, upperbound_mean_R2[:-1],'r', label='Upper bound')
plt.plot(proposed_method_instance.epochs, random_mean_R2[:-1],'g', label='Random sampling')
plt.plot(proposed_method_instance.epochs, greedy_mean_R2[:-1],'y', label='Greedy sampling')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('R2 score')
plt.title('Instance based QBC method R2 performance for {} dataset'.format(Method.dataset_name))
r2_all_fig_path = fig_path / 'r2' 
plt.savefig(r2_all_fig_path)
plt.show()

plt.plot(proposed_method_instance.epochs, instance_mean_MSE[:-1],'b', label='Instance based QBC')
plt.plot(proposed_method_instance.epochs, upperbound_mean_MSE[:-1],'r', label='Upper bound')
plt.plot(proposed_method_instance.epochs, random_mean_MSE[:-1],'g', label='Random sampling')
plt.plot(proposed_method_instance.epochs, greedy_mean_MSE[:-1],'y', label='Greedy sampling')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.title('Instance based QBC method MSE performance for {} dataset'.format(Method.dataset_name))
mse_all_fig_path = fig_path / 'mse' 
plt.savefig(mse_all_fig_path)
plt.show()

plt.plot(proposed_method_instance.epochs, instance_mean_MAE[:-1],'b', label='Instance based QBC')
plt.plot(proposed_method_instance.epochs, upperbound_mean_MAE[:-1],'r', label='Upper bound')
plt.plot(proposed_method_instance.epochs, random_mean_MAE[:-1],'g', label='Random sampling')
plt.plot(proposed_method_instance.epochs, greedy_mean_MAE[:-1],'y', label='Greedy sampling')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.title('Instance based QBC method MAE performance for {} dataset'.format(Method.dataset_name))
mae_all_fig_path = fig_path / 'mae' 
plt.savefig(mae_all_fig_path)
plt.show()

# Append the average performance to the iteration performances
instance_total_R2, instance_total_MSE, instance_total_MAE  = np.append(proposed_method_instance.instance_R2, np.reshape(instance_mean_R2, (1,n_epochs+1)), axis=0), np.append(proposed_method_instance.instance_MSE, np.reshape(instance_mean_MSE, (1,n_epochs+1)), axis=0), np.append(proposed_method_instance.instance_MAE, np.reshape(instance_mean_MAE, (1,n_epochs+1)), axis=0)
upperbound_total_R2, upperbound_total_MSE, upperbound_total_MAE  = np.append(upperbound_method.upperbound_R2, np.reshape(upperbound_mean_R2, (1,n_epochs+1)), axis=0), np.append(upperbound_method.upperbound_MSE, np.reshape(upperbound_mean_MSE, (1,n_epochs+1)), axis=0), np.append(upperbound_method.upperbound_MAE, np.reshape(upperbound_mean_MAE, (1,n_epochs+1)), axis=0)
random_total_R2, random_total_MSE, random_total_MAE  = np.append(lowerbound_method.random_R2, np.reshape(random_mean_R2, (1,n_epochs+1)), axis=0), np.append(lowerbound_method.random_MSE, np.reshape(random_mean_MSE, (1,n_epochs+1)), axis=0), np.append(lowerbound_method.random_MAE, np.reshape(random_mean_MAE, (1,n_epochs+1)), axis=0)
greedy_total_R2, greedy_total_MSE, greedy_total_MAE  = np.append(baseline_method.baseline_R2, np.reshape(greedy_mean_R2, (1,n_epochs+1)), axis=0), np.append(baseline_method.baseline_MSE, np.reshape(greedy_mean_MSE, (1,n_epochs+1)), axis=0), np.append(baseline_method.baseline_MAE, np.reshape(greedy_mean_MAE, (1,n_epochs+1)), axis=0)

# Put the total performances in a dataframe and store them
cols = ['Epoch {}'.format(i+1) for i in range(n_epochs)].append('AUC')
rows = ['Iteration {}'.format(i+1) for i in range(Method.iterations)].append('Average')
instance_df_R2, instance_df_MSE, instance_df_MAE = pd.DataFrame(instance_total_R2, index=rows, columns=cols), pd.DataFrame(instance_total_MSE, index=rows, columns=cols), pd.DataFrame(instance_total_MAE, index=rows, columns=cols)
upperbound_df_R2, upperbound_df_MSE, upperbound_df_MAE = pd.DataFrame(upperbound_total_R2, index=rows, columns=cols), pd.DataFrame(upperbound_total_MSE, index=rows, columns=cols), pd.DataFrame(upperbound_total_MAE, index=rows, columns=cols)
random_df_R2, random_df_MSE, random_df_MAE = pd.DataFrame(random_total_R2, index=rows, columns=cols), pd.DataFrame(random_total_MSE, index=rows, columns=cols), pd.DataFrame(random_total_MAE, index=rows, columns=cols)
greedy_df_R2, greedy_df_MSE, greedy_df_MAE = pd.DataFrame(greedy_total_R2, index=rows, columns=cols), pd.DataFrame(greedy_total_MSE, index=rows, columns=cols), pd.DataFrame(greedy_total_MAE, index=rows, columns=cols)

instance_df_R2.to_csv(folder_path / 'r2' / 'instance_r2.csv', header=cols) 
instance_df_MSE.to_csv(folder_path / 'mse' / 'instance_mse.csv', header=cols)
instance_df_MAE.to_csv(folder_path / 'mae' / 'instance_mae.csv' , header=cols) 

upperbound_df_R2.to_csv(folder_path / 'r2' / 'upperbound_r2.csv' , header=cols) 
upperbound_df_MSE.to_csv(folder_path / 'mse' / 'upperbound_mse.csv' , header=cols) 
upperbound_df_MAE.to_csv(folder_path / 'mae' / 'upperbound_mae.csv' , header=cols)

random_df_R2.to_csv(folder_path / 'r2' / 'random_r2.csv' , header=cols)
random_df_MSE.to_csv(folder_path / 'mse' / 'random_mse.csv' , header=cols)
random_df_MAE.to_csv(folder_path / 'mae' / 'random_mae.csv' , header=cols)

greedy_df_R2.to_csv(folder_path / 'r2' / 'greedy_r2.csv', header=cols)
greedy_df_MSE.to_csv(folder_path / 'mse' / 'greedy_mse.csv' , header=cols)
greedy_df_MAE.to_csv(folder_path / 'mae' / 'greedy_mae.csv' , header=cols)

print("Instance based results:")
print(instance_total_R2)
print("Upperbound based results:")
print(upperbound_total_R2)
print("Random based results:")
print(random_total_R2)
print("Greedy based results:")
print(greedy_total_R2)