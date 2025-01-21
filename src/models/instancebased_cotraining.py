from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, auc
from statistics import variance
import numpy as np
import time
import sys
import pandas as pd
from pathlib import Path

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent
# Adding path to sys.path
sys.path.append(str(project_root))
import config
from config import *
from data.data_processing import *
from utils.metrics import custom_accuracy, arrmse_metric

# Main script
data_dir = DATA_DIR
dataset = DATASET_NAME

k_folds = K_FOLDS
iterations = ITERATIONS
threshold = THRESHOLD
random_state = RANDOM_STATE
n_trees = N_TREES
batch_percentage = BATCH_PERCENTAGE

class CoTraining:
    def __init__(self, data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.k_folds = k_folds
        self.iterations = iterations
        self.threshold = threshold
        self.random_state = random_state
        self.n_trees = n_trees

        self.R2 = np.zeros([self.k_folds, self.iterations+1])
        self.MSE = np.zeros([self.k_folds, self.iterations+1])
        self.MAE = np.zeros([self.k_folds, self.iterations+1])
        self.CA = np.zeros([self.k_folds, self.iterations+1])
        self.ARRMSE = np.zeros([self.k_folds, self.iterations+1])
    def data_read(self, dataset):
        # Caminho do dataset
        folder_dir = data_dir / 'processed' / f'{self.dataset_name}'
        data_path = folder_dir / f'{dataset}'
        df = pd.read_csv(data_path)

        # Identificar colunas de entrada e de alvo
        col_names = list(df.columns)
        target_names = [col for col in col_names if 'target' in col]
        feature_names = [col for col in col_names if col not in target_names]

        # Separar entradas e alvos como DataFrames
        inputs = df[feature_names]
        targets = df[target_names]

        # Número de instâncias e comprimento do alvo
        n_instances = len(targets)
        target_length = len(target_names)

        return inputs, targets, n_instances, target_length, target_names, feature_names
    def read_data(self, iteration):
        X_train, y_train, _, target_length, target_names, feature_names = self.data_read(f'train_{iteration}')
        X_pool, y_pool, n_pool, target_length, target_names, feature_names = self.data_read(f'pool_{iteration}')
        X_rest, y_rest, _, target_length, target_names, feature_names = self.data_read(f'train+pool_{iteration}')
        X_test, y_test, _, target_length, target_names, feature_names = self.data_read(f'test_{iteration}')
        y_pool_nan = pd.DataFrame(np.nan, index=y_pool.index, columns=y_pool.columns)
  
        return X_train, y_train, X_pool, y_pool_nan, X_rest, y_rest, X_test, y_test,target_length, target_names, feature_names

    def train_original_model(self, X_train_labeled, y_train_labeled, X_test_labeled, y_test_labeled):
        start_time = time.time()
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state))
        model.fit(X_train_labeled, y_train_labeled)
        y_pred = model.predict(X_test_labeled)
        execution_time = time.time() - start_time

        r2 = np.round(r2_score(np.asarray(y_test_labeled), y_pred), 4)
        mse = np.round(mean_squared_error(np.asarray(y_test_labeled), y_pred), 4)
        mae = np.round(mean_absolute_error(np.asarray(y_test_labeled), y_pred), 4)
        ca = np.round(custom_accuracy(np.asarray(y_test_labeled), y_pred, self.threshold), 4)
        arrmse = np.round(arrmse_metric(np.asarray(y_test_labeled), y_pred), 4)

        print(f"Original Performance: R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}, CA={ca:.3f}, ARRMSE={arrmse:.3f}")
        print(f"    Execution time (with original data): {execution_time:.2f} seconds\n")
        return model

    def initialize_models(self):
        model_view1 = MultiOutputRegressor(RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state))
        model_view2 = MultiOutputRegressor(RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state))
        return model_view1, model_view2

    def split_features(self, X, feature_names):
        print("type split features:" + str(type(X)))
        
        # Verificar se X já é um DataFrame, se não, convertê-lo
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve ser um DataFrame.")
        
        # Calcular a divisão no meio das colunas
        mid_idx = len(feature_names) // 2
        
        # Separar os dados em duas partes
        X_v1 = X.iloc[:, :mid_idx]  # Primeira metade das colunas
        X_v2 = X.iloc[:, mid_idx:]  # Segunda metade das colunas
        
        # Separar os nomes das features correspondentes
        feature_names_v1 = feature_names[:mid_idx]
        feature_names_v2 = feature_names[mid_idx:]
        
        return X_v1, X_v2, feature_names_v1, feature_names_v2


    def confidence_computation(self, preds1, preds2, X_train_v1, X_train_v2, X_pool_v1, X_pool_v2, y_labeled):
        print(" Calculating prediction confidence...")
        confident_mask1 = np.std(preds1, axis=1) <= self.threshold
        confident_mask2 = np.std(preds2, axis=1) <= self.threshold

        combined_mask = confident_mask1 | confident_mask2

        if not combined_mask.any():
            print(" No confident predictions found.")
            return X_train_v1, X_train_v2, y_labeled, X_pool_v1, X_pool_v2, combined_mask, False

        if confident_mask1.any():
            X_train_v1 = np.vstack([X_train_v1, X_pool_v1[confident_mask1]])
            X_train_v2 = np.vstack([X_train_v2, X_pool_v2[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[confident_mask1]])

        if confident_mask2.any():
            X_train_v1 = np.vstack([X_train_v1, X_pool_v1[confident_mask2]])
            X_train_v2 = np.vstack([X_train_v2, X_pool_v2[confident_mask2]])
            y_labeled = np.vstack([y_labeled, preds2[confident_mask2]])

        X_pool_v1 = X_pool_v1[~combined_mask]
        X_pool_v2 = X_pool_v2[~combined_mask]

        print(f"    {confident_mask1.sum() + confident_mask2.sum()} examples added in this iteration.")
        
        assert X_train_v1.shape[0] == X_train_v2.shape[0], "    Mismatch in sizes of X_train_v1 and X_train_v2"

        return X_train_v1, X_train_v2, y_labeled, X_pool_v1, X_pool_v2, combined_mask, True
    
    def stop_criterion(self, preds1, preds2):
        return len(preds1) == 0 or len(preds2) == 0

    def training(self, model_view1, model_view2, X_train_v1, X_train_v2, X_pool_v1, X_pool_v2, y_labeled, X_test_v1, X_test_v2, y_test_labeled, fold_index):
        execution_times = []
        added_pairs_per_iteration = []
        
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")

            start_time = time.time()

            if len(X_train_v1) == len(y_labeled):
                model_view1.fit(X_train_v1, y_labeled)
            else:
                print(f"    Inconsistent number of samples: {len(X_train_v1)} in X_train_v1, {len(y_labeled)} in y_labeled")
                break

            if len(X_train_v2) == len(y_labeled):
                model_view2.fit(X_train_v2, y_labeled)
            else:
                print(f"    Inconsistent number of samples: {len(X_train_v2)} in X_train_v2, {len(y_labeled)} in y_labeled")
                break

            preds1 = model_view1.predict(X_pool_v1) if len(X_pool_v1) > 0 else np.array([])
            preds2 = model_view2.predict(X_pool_v2) if len(X_pool_v2) > 0 else np.array([])

            if self.stop_criterion(preds1, preds2):
                print(" No more unlabeled examples.")
                break

            X_train_v1, X_train_v2, y_labeled, X_pool_v1, X_pool_v2, combined_mask, continue_training = self.confidence_computation(
                preds1, preds2, X_train_v1, X_train_v2, X_pool_v1, X_pool_v2, y_labeled
            )

            if not continue_training:
                break

            # Track added pairs (original_instance_index, target_number)
            added_pairs = []

            if not selected_pairs:
                print(" No confident predictions found.")
                return model_view1, model_view2, X_train_v1, X_train_v2,X_pool_v1,X_pool_v2, y_labeled, execution_times, added_pairs_per_iteration
                    
            for i, j, pred1, pred2, _ in selected_pairs:
                if (i, j) in selected_pairs_set:
                    continue

                y_labeled_instance = (pred1 + pred2) / 2

                if i < len(y_labeled):  # Verificação para garantir que o índice está dentro do limite de y_labeled
                    mask = np.array(instance_mapping) == i
                    if np.any(mask):
                        idx = np.where(mask)[0][0]
                        y_labeled[idx, j] = y_labeled_instance
                    else:
                        # Adicionar instância a X_train_v1, X_train_v2 e y_labeled
                        new_instance_v1 = X_pool_v1[i].reshape(1, -1)  # Dados de X_train_v1 para i
                        new_instance_v2 = X_pool_v2[i].reshape(1, -1)  # Dados de X_train_v2 para i

                        X_train_v1 = np.vstack([X_train_v1, new_instance_v1])
                        X_train_v2 = np.vstack([X_train_v2, new_instance_v2])

                        new_instance = np.full((1, target_length), np.nan)
                        new_instance[0, j] = y_labeled_instance
                        y_labeled = np.vstack([y_labeled, new_instance])

                        instance_mapping.append(i)

                added_pairs.append((i, j))
                instance_target_count[i] += 1
                selected_pairs_set.add((i, j))

            added_pairs_per_iteration.append(added_pairs)

            confident_indices = [i for i, count in instance_target_count.items() if count == target_length]

            if confident_indices:
                # Criar as máscaras para instâncias confiantes
                confident_mask_v1 = np.isin(np.arange(X_pool_v1.shape[0]), confident_indices)
                confident_mask_v2 = np.isin(np.arange(X_pool_v2.shape[0]), confident_indices)
                
                # Selecionar as instâncias confiantes
                X_confident_v1 = X_pool_v1[confident_mask_v1]
                X_confident_v2 = X_pool_v2[confident_mask_v2]
                y_confident = np.zeros((len(confident_indices), target_length))

                for idx, confident_idx in enumerate(confident_indices):
                    if confident_idx in instance_mapping:
                        idx = instance_mapping.index(confident_idx)
                        confident_values = y_labeled[idx, :].reshape(1, -1)
                    else:
                        print(f"Warning: No data found for confident_idx {confident_idx}")
                        continue
                # Adicionar as instâncias confiantes aos dados de treinamento
                X_train_v1 = np.vstack([X_train_v1, X_confident_v1])
                X_train_v2 = np.vstack([X_train_v2, X_confident_v2])
                y_labeled = np.vstack([y_labeled, y_confident])

                # Remover as instâncias confiantes dos dados não rotulados utilizando máscaras
                print(f"Before removal: {X_pool_v1.shape[0]} unlabeled instances")

                # Remover instâncias confiantes
                X_pool_v1 = X_pool_v1[~confident_mask_v1]
                X_pool_v2 = X_pool_v2[~confident_mask_v2]

                print(f"After removal: {X_pool_v1.shape[0]} unlabeled instances")

                # Atualizar mapeamento e contagens
                instance_mapping = [idx for idx in instance_mapping if idx not in confident_indices]
                instance_target_count = {i: instance_target_count[i] for i in instance_mapping}


            print(f"{len(added_pairs)} (instance, target) pairs added in this iteration.")
            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_v1, X_test_v2, y_test_labeled)
            self.R2[fold_index, j] = r2
            self.MSE[fold_index, j] = mse
            self.MAE[fold_index, j] = mae
            self.CA[fold_index, j] = ca
            self.ARRMSE[fold_index, j] = arrmse

        return model_view1, model_view2, X_train_v1, X_train_v2, y_labeled, execution_times, added_pairs_per_iteration

    def evaluate_model(self, model_view1, model_view2, X_test_v1, X_test_v2, y_test_labeled):
        print(" Making predictions on test data...")
        y_pred_v1 = model_view1.predict(X_test_v1)
        y_pred_v2 = model_view2.predict(X_test_v2)
        y_pred_combined = (y_pred_v1 + y_pred_v2) / 2

        r2 = np.round(r2_score(np.asarray(y_test_labeled), y_pred_combined), 4)
        mse = np.round(mean_squared_error(np.asarray(y_test_labeled), y_pred_combined), 4)
        mae = np.round(mean_absolute_error(np.asarray(y_test_labeled), y_pred_combined), 4)
        ca = np.round(custom_accuracy(np.asarray(y_test_labeled), y_pred_combined, self.threshold), 4)
        arrmse = np.round(arrmse_metric(np.asarray(y_test_labeled), y_pred_combined), 4)

        print(f"    Overall: R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}, CA={ca:.3f}, ARRMSE={arrmse:.3f}")

        return r2, mse, mae, ca, arrmse

    def train_and_evaluate(self, fold_index):
        print(f"\n    Training model in pool {fold_index}...")
        X_train_labeled, y_labeled, X_pool, y_pool, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length,target_names,feature_names = self.read_data(fold_index+1)

        self.train_original_model(X_train_labeled, y_labeled, X_test_labeled, y_test_labeled)

        model_view1, model_view2 = self.initialize_models()
        X_train_v1, X_train_v2,feature_names_v1,feature_names_v2 = self.split_features(X_train_labeled,feature_names)
        X_pool_v1, X_pool_v2,feature_names_v1,feature_names_v2 = self.split_features(X_pool,feature_names)
        X_test_v1, X_test_v2,feature_names_v1,feature_names_v2 = self.split_features(X_test_labeled,feature_names)


        model_view1, model_view2, X_train_v1, X_train_v2, y_labeled, execution_times, added_pairs_per_iteration = self.training(
            model_view1, model_view2, X_train_v1, X_train_v2, X_pool_v1, X_pool_v2, y_labeled, X_test_v1, X_test_v2, y_test_labeled, fold_index
        )

        r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_v1, X_test_v2, y_test_labeled)
        self.R2[fold_index, -1] = r2
        self.MSE[fold_index, -1] = mse
        self.MAE[fold_index, -1] = mae
        self.CA[fold_index, -1] = ca
        self.ARRMSE[fold_index, -1] = arrmse

        return self.R2, self.MSE, self.MAE, self.CA, self.ARRMSE, added_pairs_per_iteration

class InstanceCoTraining(CoTraining):
    def __init__(self, data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees, batch_size):
        super().__init__(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
        self.batch_size = batch_size

    def confidence_computation(self, preds1, preds2, X_train_v1, X_train_v2, X_pool_v1, X_pool_v2, y_labeled):
        print(" Calculating prediction confidence...")
        confident_mask1 = np.std(preds1, axis=1) <= self.threshold
        confident_mask2 = np.std(preds2, axis=1) <= self.threshold

        combined_mask = confident_mask1 | confident_mask2

        if not combined_mask.any():
            print(" No confident predictions found.")
            return X_train_v1, X_train_v2, y_labeled, X_pool_v1, X_pool_v2, combined_mask, False

        confident_indices = np.where(combined_mask)[0]
        if len(confident_indices) > self.batch_size:
            # Select the top batch_size indices based on the highest confidence
            variances = np.std(preds1[confident_indices], axis=1) + np.std(preds2[confident_indices], axis=1)
            top_indices = np.argsort(variances)[:self.batch_size]
            confident_indices = confident_indices[top_indices]

        # Generate top_confident_mask1 and top_confident_mask2
        top_confident_mask1 = np.isin(np.arange(len(confident_mask1)), confident_indices) & confident_mask1
        top_confident_mask2 = np.isin(np.arange(len(confident_mask2)), confident_indices) & confident_mask2

        if top_confident_mask1.any():
            X_train_v1 = np.vstack([X_train_v1, X_pool_v1[top_confident_mask1]])
            X_train_v2 = np.vstack([X_train_v2, X_pool_v2[top_confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[top_confident_mask1]])

        if top_confident_mask2.any():
            X_train_v1 = np.vstack([X_train_v1, X_pool_v1[top_confident_mask2]])
            X_train_v2 = np.vstack([X_train_v2, X_pool_v2[top_confident_mask2]])
            y_labeled = np.vstack([y_labeled, preds2[top_confident_mask2]])

        X_pool_v1 = np.delete(X_pool_v1, confident_indices, axis=0)
        X_pool_v2 = np.delete(X_pool_v2, confident_indices, axis=0)

        top_combined_mask = top_confident_mask1 | top_confident_mask2

        print(f"{len(confident_indices)} examples added in this iteration.")
        
        assert X_train_v1.shape[0] == X_train_v2.shape[0], "Mismatch in sizes of X_train_v1 and X_train_v2"

        return X_train_v1, X_train_v2, y_labeled, X_pool_v1, X_pool_v2, top_combined_mask, True

class TargetCoTraining(CoTraining):
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
            rf_model = model.estimators_[i]
            for j, estimator in enumerate(rf_model.estimators_):
                # Make predictions for the current tree
                predictions = estimator.predict(X_pool.values)  # Ensure no column names passed
                pool_preds[:, j] = predictions

            # Calculate variance for each instance in X_pool
            for idx, row_idx in enumerate(X_pool.index):
                variances.loc[idx, i] = variance(pool_preds[idx, :])

        return variances
    def select_confident_pairs(self, variances):
        confident_pairs = {}
        print('oi confident')
        
        # Iterating over the DataFrame rows and columns using iterrows() for index compatibility
        for idx, row in variances.iterrows():  # iterrows() gives (index, Series)

            for col, value in row.items():  # Iterating over each column in the row
                if value <= self.threshold:
                    confident_pairs[(int(idx), col)] = value  # Add (index, column) pair to the dictionary

        return confident_pairs


    def training(self, model_view1, model_view2, X_train_v1_df, X_train_v2_df, X_pool_v1_df, X_pool_v2_df, y_labeled_df, X_test_v1, X_test_v2, y_test, target_length, fold_index,target_names,feature_names_v1,feature_names_v2,y_pool):
        execution_times = []
        added_pairs_per_iteration = []
        print(y_pool.head())
        print(y_pool.shape)
        print(y_pool.columns)

        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")

            start_time = time.time()

            model_view1.fit(X_train_v1_df, y_labeled_df)
            model_view2.fit(X_train_v2_df, y_labeled_df)

            preds1 = model_view1.predict(X_pool_v1_df)
            preds2 = model_view2.predict(X_pool_v2_df)

            variances1 = self.calculate_variances(model_view1, X_pool_v1_df, target_length)
            variances2 = self.calculate_variances(model_view2, X_pool_v2_df, target_length)
            


            confident_pairs1 = self.select_confident_pairs(variances1)
            confident_pairs2 = self.select_confident_pairs(variances2)
            # o original não usa união
            union_set = set(confident_pairs1.keys()).union(set(confident_pairs2.keys()))
            
            confident_pairs_combined = {}

            for pair in union_set:
                if pair in confident_pairs1 and pair in confident_pairs2:
                    confident_pairs_combined[pair] = (confident_pairs1[pair] + confident_pairs2[pair]) / 2
                elif pair in confident_pairs1:
                    confident_pairs_combined[pair] = confident_pairs1[pair]
                elif pair in confident_pairs2:
                    confident_pairs_combined[pair] = confident_pairs2[pair]
            #print(confident_pairs_combined)
            #dicionario (index, posição y) -> variancia
            sorted_confident_pairs = sorted(confident_pairs_combined.items(), key=lambda item: item[1])

            pred_selected_pairs = {pair: (preds1[pair] + preds2[pair]) / 2 for pair, _ in sorted_confident_pairs[:self.batch_size * target_length]}
            #print(pred_selected_pairs)

            if not pred_selected_pairs:
                print("No confident predictions found.")
                break

            # Tentar subir isso pra pegar o indice ao selecionar a variancia 

            
            print(f"Before inclusion: {X_train_v1_df.shape} labeled instances for X_train_v1")
            print("type X_train_v1"+ str(type(X_train_v1_df)))
            print(f"Before inclusion: {X_train_v2_df.shape} labeled instances for X_train_v2")
            print("type X_train_v2"+ str(type(X_train_v2_df)))
            print(f"Before inclusion: {len(y_labeled_df)} labeled instances for y_labeled")
                        
            print(f"Before inclusion: {X_pool_v1_df.shape} labeled instances for X_pool_v1_df")
            print("type X_pool_v1"+ str(type(X_pool_v1_df)))
            print(f"Before inclusion: {X_pool_v2_df.shape} labeled instances for X_pool_v2_df")
            print("type X_train_v2"+ str(type(X_pool_v2_df)))
            print(f"Before inclusion: {len(y_labeled_df)} labeled instances for y_labeled")
            
            filled_targets_per_instance = {}
            selected_pairs_filled = []
            print(len(pred_selected_pairs))
            indices = set()
            count = 0 
            '''
            for idx, j in pred_selected_pairs.keys():
                if(count == 0):
                else:
                count = count + 1
                indices.add(idx)

                filled_targets_per_instance[idx] = j
                if idx not in selected_pairs_filled:
                        X_train_v1_df = pd.concat([X_train_v1_df,X_pool_v1_df.iloc[[idx]]], ignore_index=False)
                        X_train_v2_df = pd.concat([X_train_v2_df, X_pool_v2_df.iloc[[idx]]], ignore_index=False)
        
                        selected_pairs_filled.append(idx)
            '''
            print("tamanho indices:"+str(len(indices)))
            print(f"After X inclusion: {len(X_train_v1_df)} labeled instances for v1")
            print(X_train_v1_df.head())
            print(f"After X inclusion: {len(X_train_v2_df)} labeled instances for v2")
            print(X_train_v2_df.head())
            #até aqui ok

            #retornar a quantidade de pares que está sendo selecionado 
            #vamos agora pegar os dados das predições pred_selected_pairs e salvar em y_labeled_df
            #salvaria realmente no y_labeled ou em outro vetor inicial?
            for idx, j in pred_selected_pairs.keys():
                y_labeled_df 
            '''

            if len(y_labeled) != len(X_train_v1):
                    # Caso y_labeled tenha menos amostras, vamos expandi-lo com valores padrões (por exemplo, 0 ou -1)
                    # ou você pode inicializá-lo com algum valor adequado para o seu caso
                    new_labels = np.zeros((len(X_train_v1), target_length))  # Ou outro valor padrão
                    y_labeled = np.concatenate([y_labeled, new_labels[len(y_labeled):]])
            print(y_labeled.shape)
            for target in filled_targets_per_instance[idx]:
                    print(pred_selected_pairs[(idx, target)])
                    y_labeled[idx, target] = pred_selected_pairs[(idx, target)]
                    selected_pairs_set.update((idx, target) for target in filled_targets_per_instance[idx])
            
            if len(filled_targets_per_instance[idx]) == target_length:

                    print(filled_targets_per_instance)

                    # Aqui, a instância será removida de X_pool apenas após todos os rótulos estarem preenchidos
                    remaining_indices = [i for i in range(len(X_pool_v1)) if i not in selected_pairs_filled or len(filled_targets_per_instance[i]) < target_length]
                    X_pool_v1 = X_pool_v1[remaining_indices]
                    X_pool_v2 = X_pool_v2[remaining_indices]

            execution_times.append(time.time() - start_time)
            added_pairs_per_iteration.append(len(pred_selected_pairs))
            '''

        return model_view1, model_view2, X_train_v1_df, X_train_v2_df, X_pool_v1_df, X_pool_v2_df, y_labeled, execution_times, added_pairs_per_iteration
    def train_and_evaluate(self, fold_index):
        print(f"\nTraining model in fold {fold_index}...")
        X_train_labeled, y_labeled, X_pool, y_pool, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length,target_names,feature_names = self.read_data(fold_index+1)

        #self.train_original_model(X_train_labeled, y_labeled, X_test_labeled, y_test_labeled)

        model_view1, model_view2 = self.initialize_models()
        X_train_labeled_v1, X_train_labeled_v2,feature_names_v1,feature_names_v2 = self.split_features(X_train_labeled,feature_names)
        X_pool_v1, X_pool_v2,feature_names_v1,feature_names_v2  = self.split_features(X_pool,feature_names)
        X_test_labeled_v1, X_test_labeled_v2,feature_names_v1,feature_names_v2  = self.split_features(X_test_labeled,feature_names)

        model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2,X_pool_v1,X_pool_v2, y_labeled, execution_times, added_pairs_per_iteration = self.training(
            model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, X_pool_v1, X_pool_v2, y_labeled, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled, target_length, fold_index,target_names,feature_names_v1,feature_names_v2,y_pool
        )

        # Avaliação do modelo após o treinamento
        r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled)
        self.R2[fold_index, -1] = r2
        self.MSE[fold_index, -1] = mse
        self.MAE[fold_index, -1] = mae
        self.CA[fold_index, -1] = ca
        self.ARRMSE[fold_index, -1] = arrmse
    
        return self.R2, self.MSE, self.MAE, self.CA, self.ARRMSE, added_pairs_per_iteration
    
if __name__ == "__main__":
    data_dir = config.DATA_DIR
    dataset_name = config.DATASET_NAME

    """ # Original co-training model
    print('Original Co-Training...')
    cotraining_model = CoTraining(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
    
    for i in range(k_folds):
        R2, MSE, MAE, CA, ARRMSE, added_pairs_per_iteration = cotraining_model.train_and_evaluate(i)
        for j, added_pairs in enumerate(added_pairs_per_iteration):
            print(f"        Added pairs in fold {i}, iteration {j}: {added_pairs}")
        
        R2_flat = R2[i, :].flatten()
        MSE_flat = MSE[i, :].flatten()
        MAE_flat = MAE[i, :].flatten()
        CA_flat = CA[i, :].flatten()
        ARRMSE_flat = ARRMSE[i, :].flatten()
        added_pairs_flat = added_pairs_per_iteration
        num_entries = max(R2[i, :].size, MSE[i, :].size, MAE[i, :].size, CA[i, :].size, ARRMSE[i, :].size)

        if len(added_pairs_flat) < num_entries:
            added_pairs_flat.extend([[]] * (num_entries - len(added_pairs_flat)))
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
            'Fold_Index': [i for _ in range(num_entries)],
            'Iterations': list(range(num_entries)),
            'R2': R2_flat,
            'MSE': MSE_flat,
            'MAE': MAE_flat,
            'CA': CA_flat,
            'ARRMSE': ARRMSE_flat,
            'Added_Pairs': added_pairs_flat
        })
        results_path = Path(f'reports/semi_supervised_learning/{dataset_name}')
        results_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path / f'original_cotraining_results_fold_{i}.csv', index=False)

    # Instance-based co-training model
    print('Co-Training with Top-k Confidence...')
    cotraining_model = CoTraining(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
    X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length = cotraining_model.read_data(1)
    batch_size = round((batch_percentage / 100) * len(X_pool))

    instance_cotraining_model = InstanceCoTraining(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees, batch_size)
    
    for i in range(k_folds):
        R2, MSE, MAE, CA, ARRMSE, added_pairs_per_iteration = instance_cotraining_model.train_and_evaluate(i)
        
        # Save results for each fold to DataFrame and CSV
        R2_flat = R2[i, :].flatten()
        MSE_flat = MSE[i, :].flatten()
        MAE_flat = MAE[i, :].flatten()
        CA_flat = CA[i, :].flatten()
        ARRMSE_flat = ARRMSE[i, :].flatten()
        added_pairs_flat = added_pairs_per_iteration
        num_entries = max(R2[i, :].size, MSE[i, :].size, MAE[i, :].size, CA[i, :].size, ARRMSE[i, :].size)

        if len(added_pairs_flat) < num_entries:
            added_pairs_flat.extend([[]] * (num_entries - len(added_pairs_flat)))
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
            'Fold_Index': [i for _ in range(num_entries)],
            'Iterations': list(range(num_entries)),
            'R2': R2_flat,
            'MSE': MSE_flat,
            'MAE': MAE_flat,
            'CA': CA_flat,
            'ARRMSE': ARRMSE_flat,
            'Added_Pairs': added_pairs_flat
        })
        results_path = Path(f'reports/semi_supervised_learning/{dataset_name}')
        results_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path / f'instance_cotraining_results_fold_{i}.csv', index=False) """

    # Target-based co-training model
    print('Target-based Co-Training...')
    cotraining_model = CoTraining(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
    X_train, y_labeled, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length,target_names,feature_names = cotraining_model.read_data(1)

    batch_size = round((batch_percentage / 100) * len(X_pool))

    target_cotraining_model = TargetCoTraining(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees, batch_size)
    
    for i in range(k_folds):
        R2, MSE, MAE, CA, ARRMSE, added_pairs_per_iteration = target_cotraining_model.train_and_evaluate(i)
        
        # Save results for each fold to DataFrame and CSV
        R2_flat = R2[i, :].flatten()
        MSE_flat = MSE[i, :].flatten()
        MAE_flat = MAE[i, :].flatten()
        CA_flat = CA[i, :].flatten()
        ARRMSE_flat = ARRMSE[i, :].flatten()
        added_pairs_flat = added_pairs_per_iteration
        num_entries = max(R2[i, :].size, MSE[i, :].size, MAE[i, :].size, CA[i, :].size, ARRMSE[i, :].size)

        if len(added_pairs_flat) < num_entries:
            added_pairs_flat.extend([[]] * (num_entries - len(added_pairs_flat)))
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
            'Fold_Index': [i for _ in range(num_entries)],
            'Iterations': list(range(num_entries)),
            'R2': R2_flat,
            'MSE': MSE_flat,
            'MAE': MAE_flat,
            'CA': CA_flat,
            'ARRMSE': ARRMSE_flat,
            'Added_Pairs': added_pairs_flat
        })

        results_path = Path(f'reports/semi_supervised_learning/{dataset_name}')
        results_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path / f'target_cotraining_results_fold_{i}.csv', index=False)
        print('Saved data...')
