import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import config
import csv
import os


# Função para calcular as métricas
def calculate_metrics(y_true, y_pred):
    """Calcula R2, MSE, MAE, CA e ARRMSE."""
    r2 = r2_score(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    mae = mean_absolute_error(y_true, y_pred)
    ca = custom_accuracy(y_true, y_pred)
    arrmse = np.sqrt(mse / np.mean(y_true ** 2))  # Relative RMSE
    return r2, mse, mae, ca, arrmse

# Função para calcular as métricas médias por método
def calculate_mean_performances(method_results, method_name):
    """Calcula as médias das métricas para um método."""
    r2_list, mse_list, mae_list, ca_list, arrmse_list = [], [], [], [], []

    for y_true, y_pred in method_results:
        r2, mse, mae, ca, arrmse = calculate_metrics(y_true, y_pred)
        r2_list.append(r2)
        mse_list.append(mse)
        mae_list.append(mae)
        ca_list.append(ca)
        arrmse_list.append(arrmse)

    mean_r2 = np.mean(r2_list)
    mean_mse = np.mean(mse_list)
    mean_mae = np.mean(mae_list)
    mean_ca = np.mean(ca_list)
    mean_arrmse = np.mean(arrmse_list)

    print(f"{method_name}: Mean R2={mean_r2:.3f}, Mean MSE={mean_mse:.3f}, "
          f"Mean MAE={mean_mae:.3f}, Mean CA={mean_ca:.3f}, Mean ARRMSE={mean_arrmse:.3f}")
    return mean_r2, mean_mse, mean_mae, mean_ca, mean_arrmse

def data_read(dataset, data_dir,dataset_name):
        # split the csv file in the input and target values
        folder_dir = data_dir / 'processed' / f'{dataset_name}'
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
        for k in range(len(df)):
            input_val = list()
            target_val = list()
            for col in col_names:
                if col in target_names:
                    target_val.append(df.loc[k, col])
                else:
                    input_val.append(df.loc[k, col])
            inputs.append(input_val)
            targets.append(target_val)

        n_instances = len(targets)
        return inputs, targets, n_instances, target_length

# Helper function to read data
def read_data(iteration,data_dir,dataset_name): 
    X_train, y_train, _, _ = data_read(f'train_{iteration}',data_dir,dataset_name)
    X_pool, y_pool, n_pool, target_length = data_read(f'pool_{iteration}',data_dir,dataset_name)
    X_rest, y_rest, _, _ = data_read(f'train+pool_{iteration}',data_dir,dataset_name)
    X_test, y_test, _, _ = data_read(f'test_{iteration}',data_dir,dataset_name)
    return X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length

def get_datasets(k,data_dir,dataset_name):
    X_train_not_missing, Y_train_not_missing, X_unlabeled,Y_train_missing,X_rest, y_rest, X_test_labeled, y_test_labeled, target_length = read_data(k+1,data_dir,dataset_name)
    X_train_not_missing = np.array(X_train_not_missing)
    Y_train_not_missing = np.array(Y_train_not_missing)
    X_unlabeled = np.array(X_unlabeled)
    X_test_labeled = np.array(X_test_labeled)
    y_test_labeled = np.array(y_test_labeled)

    return X_train_not_missing,Y_train_not_missing,X_unlabeled,X_test_labeled,y_test_labeled,Y_train_missing


class PCTNode:
    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.is_leaf = False
        self.prediction = None
        self.depth = depth

class PredictiveClusteringTree:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, Y):
        self.root = self._build_tree(X, Y, depth=0)

    def _build_tree(self, X, Y, depth):
        # Converta X e Y para arrays NumPy, se forem DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(Y, pd.DataFrame):
            Y = Y.to_numpy()

        n_samples, n_features = X.shape

        # Stop conditions: leaf node
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf = PCTNode(depth=depth)
            leaf.is_leaf = True
            leaf.prediction = np.mean(Y, axis=0)
            return leaf

        # Find the best split
        best_split = None
        best_variance = np.inf

        for feature in range(n_features):
            unique_values = np.unique(X[:, feature])  # X é agora um array NumPy
            for value in unique_values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask

                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    Y_left, Y_right = Y[left_mask], Y[right_mask]

                    # Compute variance for this split
                    var_left = np.var(Y_left, axis=0).sum()
                    var_right = np.var(Y_right, axis=0).sum()
                    combined_variance = var_left + var_right

                    if combined_variance < best_variance:
                        best_variance = combined_variance
                        best_split = (feature, value, left_mask, right_mask)

        if best_split is None:
            # If no valid split is found, create a leaf node
            leaf = PCTNode(depth=depth)
            leaf.is_leaf = True
            leaf.prediction = np.mean(Y, axis=0)
            return leaf

        # Create internal node with the best split
        feature, value, left_mask, right_mask = best_split
        node = PCTNode(depth=depth)
        node.split_feature = feature
        node.split_value = value
        node.left = self._build_tree(X[left_mask], Y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], Y[right_mask], depth + 1)

        return node

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.prediction
        if x[node.split_feature] <= node.split_value:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

# Custom Accuracy (CA)
def custom_accuracy(y_true, y_pred, threshold=0.1):
    """Custom accuracy: percentage of predictions within a certain threshold."""
    return np.mean(np.abs(y_true - y_pred) <= threshold)

data_dir = config.DATA_DIR
#dataset_name = config.DATASET_NAME
dataset_names = ['atp7d', 'friedman', 'mp5spec', 'musicOrigin2', 'rf2', 'oes97', 'enb', 'osales', 'wq']
for dataset_name in dataset_names:
    print(dataset_name)
    number_of_pools = 10
    for i in range (number_of_pools):
        print(f"Treinando modelo no pool {i}...")  
        X_train_not_missing, Y_train_not_missing, X_unlabeled, X_test_labeled, y_test_labeled, Y_train_missing = get_datasets(i,data_dir,dataset_name)
        # ------------------ Semissupervisionado ------------------
        start_time = time.time()
        w = 0.5  # Adjust this parameter to control influence (0.0 - only labeled data, 1.0 - equal influence)
        pct_original = PredictiveClusteringTree(max_depth=5, min_samples_split=10)
        pct_original.fit(X_train_not_missing, Y_train_not_missing)
        # Preencher valores faltantes com previsões do modelo original
        Y_train_filled = Y_train_missing.copy()

        for j in range(len(Y_train_filled)):
            # Verifica se a linha contém algum NaN
            if any(pd.isna(Y_train_filled[j])):
                # Obtém os valores da linha correspondente de X_train_missing
                x_missing = X_unlabeled[j, :].reshape(1, -1)

                # Prediz os valores faltantes usando o modelo original
                y_pred = pct_original.predict(x_missing)
                
                # Preenche os valores previstos em Y_train_filled
                Y_train_filled.iloc[j, :] = y_pred

        # Combine os datasets
        if isinstance(X_unlabeled, np.ndarray):
            X_unlabeled = pd.DataFrame(X_unlabeled, columns=[f'feature_{j}' for j in range(X_unlabeled.shape[1])])
        if isinstance(X_train_not_missing, np.ndarray):
            X_train_not_missing = pd.DataFrame(X_train_not_missing, columns=[f'feature_{j}' for j in range(X_train_not_missing.shape[1])])
        
        X_train_combined = pd.concat([X_unlabeled, X_train_not_missing], axis=0)
        X_train_combined.reset_index(drop=True, inplace=True)

        if isinstance(Y_train_filled, np.ndarray):
            Y_train_filled = pd.DataFrame(Y_train_filled, columns=[f'feature_{j}' for j in range(Y_train_filled.shape[1])])
        if isinstance(Y_train_not_missing, np.ndarray):
            Y_train_not_missing = pd.DataFrame(Y_train_not_missing, columns=[f'feature_{j}' for j in range(Y_train_not_missing.shape[1])])
        if isinstance(Y_train_filled, list):
            Y_train_filled = pd.DataFrame(Y_train_filled, columns=[f'feature_{j}' for j in range(len(Y_train_filled[0]))])
        if isinstance(Y_train_missing, list):
            Y_train_missing = pd.DataFrame(Y_train_missing, columns=[f'feature_{j}' for j in range(len(Y_train_missing[0]))])

        Y_train_combined = pd.concat([Y_train_filled, Y_train_not_missing], axis=0)
        Y_train_combined.reset_index(drop=True, inplace=True)

        # Criar uma máscara para instâncias rotuladas (não faltantes)
        labeled_mask = ~Y_train_missing.isna().any(axis=1)

        # Criar o vetor de pesos
        weights_filled = np.full(Y_train_filled.shape[0], w)  # Pesos para instâncias preenchidas
        weights_not_missing = np.ones(np.sum(labeled_mask))  # Pesos para instâncias rotuladas originais
        weights = np.hstack((weights_not_missing, weights_filled))

        # Treinar novo modelo semissupervisionado
        pct_semi_weighted = PredictiveClusteringTree(max_depth=5, min_samples_split=10)
        pct_semi_weighted.fit(X_train_combined, Y_train_combined)

        # Fazer previsões no conjunto de teste
        Y_pred_semi_weighted = pct_semi_weighted.predict(X_test_labeled)

        # Medir o tempo final
        end_time = time.time()
    
        print("Desempenho Semissupervisionado:")
    # Coleta dos resultados para cálculo consolidado
        proposed_method_pct = list(zip(y_test_labeled, Y_pred_semi_weighted))

        # Métricas para o método proposto
        pct_mean_R2, pct_mean_MSE, pct_mean_MAE, pct_mean_CA, pct_mean_ARRMSE = \
            calculate_mean_performances(proposed_method_pct, 'Proposed Method')
            

        # Supondo que as métricas foram calculadas como abaixo:
        metrics = {
            "method": [
                "pct"
            ],
            "mean_R2": [
                pct_mean_R2,
            ],
            "mean_MSE": [
                pct_mean_MSE,
            ],
            "mean_MAE": [
                pct_mean_MAE,
            ],
            "mean_CA": [
                pct_mean_CA,
            ],
            "mean_ARRMSE": [
                pct_mean_ARRMSE,
            ],
        }

        # Caminho para salvar o arquivo CSV

        output_dir = rf"C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\reports\semi_supervised\pct\{dataset_name}"
        output_file = rf"{output_dir}\{i}.csv"

        # Cria o diretório se não existir
        os.makedirs(output_dir, exist_ok=True)

        # Convertendo o dicionário para um DataFrame e salvando no formato CSV
        df_metrics = pd.DataFrame(metrics)
        df_metrics.to_csv(output_file, index=False)

        print(f"Métricas salvas em '{output_file}'.")