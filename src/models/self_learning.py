import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import time
from sklearn.ensemble import RandomForestRegressor
import config

data_dir = config.DATA_DIR
dataset_name = config.DATASET_NAME
def data_read(dataset):
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
        for i in range(len(df)):
            input_val = list()
            target_val = list()
            for col in col_names:
                if col in target_names:
                    target_val.append(df.loc[i, col])
                else:
                    input_val.append(df.loc[i, col])
            inputs.append(input_val)
            targets.append(target_val)

        n_instances = len(targets)
        return inputs, targets, n_instances, target_length

# Helper function to read data
def read_data(iteration):
    X_train, y_train, _, _ = data_read(f'train_{iteration}')
    X_pool, y_pool, n_pool, target_length = data_read(f'pool_{iteration}')
    X_rest, y_rest, _, _ = data_read(f'train+pool_{iteration}')
    X_test, y_test, _, _ = data_read(f'test_{iteration}')
    return X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length

def get_datasets(i):
    X_train_not_missing, Y_train_not_missing, X_unlabeled,Y_train_missing,X_rest, y_rest, X_test_labeled, y_test_labeled, target_length = read_data(i+1)
    X_train_not_missing = np.array(X_train_not_missing)
    Y_train_not_missing = np.array(Y_train_not_missing)
    X_unlabeled = np.array(X_unlabeled)
    X_test_labeled = np.array(X_test_labeled)
    y_test_labeled = np.array(y_test_labeled)

    return X_train_not_missing,Y_train_not_missing,X_unlabeled,X_test_labeled,y_test_labeled,Y_train_missing


# Função de acurácia personalizada
def custom_accuracy(y_true, y_pred, threshold=0.1):
    """Custom accuracy: percentage of predictions within a certain threshold."""
    return np.mean(np.abs(y_true - y_pred) <= threshold)

# Carregar os dados


number_of_pools = 10
for i in range (number_of_pools):
    print(f"Treinando modelo no pool {i}...")  
    X_train_not_missing, Y_train_not_missing, X_unlabeled, X_test_labeled, y_test_labeled, Y_train_missing = get_datasets(i)

# Avaliação do desempenho original com o modelo treinado com os dados completos (X_train_not_missing e Y_train_not_missing)
    print("Desempenho com Dados Originais (sem aprendizado ativo):")
    start_time = time.time()  # Medir o tempo de execução
    regressor = RandomForestRegressor()

    regressor.fit(X_train_not_missing, Y_train_not_missing)
    Y_pred_original = regressor.predict(X_test_labeled)
    for j in range(y_test_labeled.shape[1]):
        r2 = r2_score(y_test_labeled[:, j], Y_pred_original[:, j])
        mae = mean_absolute_error(y_test_labeled[:, j], Y_pred_original[:, j])
        ca = custom_accuracy(y_test_labeled[:, j], Y_pred_original[:, j])
        print(f"Target {j+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
    print(f"Tempo de execução (com dados originais): {time.time() - start_time:.2f} segundos\n")


    print("Desempenho Semissupervisionado")
    # Aprendizado Ativo
    max_iter = 10
    for j in range(max_iter):
        if X_unlabeled.shape[0] == 0:  # Verifica se X_unlabeled está vazio
            print("Conjunto de dados não rotulados esgotado.")
            break  
        print(f"Treinando modelo na epoch {j}...")  
        # modelo de regressão treinado com dados rotulados 
        predictions_unlabeled = regressor.predict(X_unlabeled)
        error_threshold = np.percentile(np.abs(predictions_unlabeled - Y_train_missing), 90)
        confident_idx = np.where(np.abs(predictions_unlabeled - Y_train_missing) < error_threshold)[0]
        
        # Adicionar os dados de alta confiança ao conjunto de treinamento
        X_train_not_missing = np.concatenate([X_train_not_missing, X_unlabeled[confident_idx]], axis=0)
        Y_train_not_missing = np.concatenate([Y_train_not_missing, predictions_unlabeled[confident_idx]], axis=0)
        
        # Atualizar o conjunto de dados não rotulado
        X_unlabeled = np.delete(X_unlabeled, confident_idx, axis=0)
        Y_train_missing = np.delete(Y_train_missing, confident_idx, axis=0)
        regressor.fit(X_train_not_missing, Y_train_not_missing)
    # Avaliação final
    regressor.fit(X_train_not_missing, Y_train_not_missing)
    Y_pred_semi_weighted = regressor.predict(X_test_labeled)
    for j in range(y_test_labeled.shape[1]):
        r2 = r2_score(y_test_labeled[:, j], Y_pred_semi_weighted[:, j])
        mae = mean_absolute_error(y_test_labeled[:, j], Y_pred_semi_weighted[:, j])
        ca = custom_accuracy(y_test_labeled[:, j], Y_pred_semi_weighted[:, j])
        print(f"Target {i+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")

    # Tempo de execução final
    end_time = time.time()
    print(f"Tempo de execução total: {end_time - start_time:.2f} segundos")