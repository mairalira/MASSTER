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
project_root = Path(__file__).resolve().parent.parent.parent
# Adding path to sys.path
sys.path.append(str(project_root))

import config
from config import *

from data.data_processing import *
from utils.metrics import custom_accuracy, arrmse_metric
from contextlib import redirect_stdout
with open(r'.\output.txt', 'w') as f:
    with redirect_stdout(f):
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
                self.model = SingleTargetRegressor(random_state, n_trees)

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

                X_pool.index = pd.RangeIndex(start = len(X_train), stop = len(X_train) + len(X_pool), step = 1)
                y_pool_nan.index = pd.RangeIndex(start = len(y_train), stop = len(y_train) + len(y_pool), step = 1)

        
                return X_train, y_train, X_pool, y_pool_nan, X_rest, y_rest, X_test, y_test,target_length, target_names, feature_names

            def train_original_model(self, X_train_labeled, y_train_labeled, X_test_labeled, y_test_labeled):
                start_time = time.time()
                models = self.model.unique_fit(len(y_train_labeled.columns), y_train_labeled, X_train_labeled)
                execution_time = time.time() - start_time

                # Evaluate the model using the evaluate_model method
                r2, mse, mae, ca, arrmse = self.evaluate_model(models, models, X_test_labeled, X_test_labeled, y_test_labeled)

                print(f"Original Performance: R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}, CA={ca:.3f}, ARRMSE={arrmse:.3f}")
                print(f"    Execution time (with original data): {execution_time:.2f} seconds\n")
                return models

            def initialize_models(self):
                return self.model, self.model

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


            def stop_criterion(self, preds1, preds2):
                return len(preds1) == 0 or len(preds2) == 0
                    
        class TargetCoTraining(CoTraining):
                def __init__(self, data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees, batch_size):
                    super().__init__(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
                    self.batch_size = batch_size
                def stop_criterion(self, preds1, preds2):
                    return len(preds1) == 0 or len(preds2) == 0

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
                    confident_pairs = {}        # {(idx,col) : variancia}
                    # Iterating over the DataFrame rows and columns using iterrows() for index compatibility
                    for idx, row in variances.iterrows():  # iterrows() gives (index, Series)

                        for col, value in row.items():  # Iterating over each column in the row
                            if value <= self.threshold:
                                confident_pairs[(int(idx), col)] = value  # Add (index, column) pair to the dictionary
                    return confident_pairs
                
                
                
                

                def training(self, X_train_v1_df, X_train_v2_df, X_pool_v1_df, X_pool_v2_df, y_train_df, X_test_v1, X_test_v2, y_test, target_length, fold_index,target_names,feature_names_v1,feature_names_v2,y_pool):
                    
                            # Agora, qualquer print dentro desse bloco será registrado no arquivo
                            execution_times = []
                            added_pairs_per_iteration = []
                            dict_index = {} # {chave:valor} -> {index_train:index_pool}
                            all_pred_selected_pairs = {}
                            model = SingleTargetRegressor(self.random_state,self.n_trees)
                            for iteration in range(self.iterations):

                                print(f"Iteration {iteration + 1}/{self.iterations}")

                                start_time = time.time()
                                

                                
                                models_view1_array = model.unique_fit(target_length, y_train_df, X_train_v1_df)
                                models_view2_array = model.unique_fit(target_length, y_train_df, X_train_v2_df)

                                columns = list(y_pool.columns)
                                preds1 = model.unique_predict(models_view1_array, X_pool_v1_df,target_length,columns)
                                preds2 = model.unique_predict(models_view2_array, X_pool_v2_df,target_length, columns)    
                                
                                if self.stop_criterion(preds1, preds2):
                                    print(" No more unlabeled examples.")
                                    break
                                
                                variances1 = self.calculate_variances(models_view1_array, X_pool_v1_df, target_length)
                                variances2 = self.calculate_variances(models_view2_array, X_pool_v2_df, target_length)
                                
                                confident_pairs1 = self.select_confident_pairs(variances1)
                                confident_pairs2 = self.select_confident_pairs(variances2)
                                # o original não usa união
                                union_set = set(confident_pairs1.keys()).union(set(confident_pairs2.keys()))
                                
                                confident_pairs_combined = {}
                                confident_pairs_combined_filtered = {} 
                                for pair in union_set:
                                    if pair in confident_pairs1 and pair in confident_pairs2:
                                        confident_pairs_combined[pair] = (confident_pairs1[pair] + confident_pairs2[pair]) / 2
                                    elif pair in confident_pairs1:
                                        confident_pairs_combined[pair] = confident_pairs1[pair]
                                    elif pair in confident_pairs2:
                                        confident_pairs_combined[pair] = confident_pairs2[pair]
                                
                                sorted_confident_pairs_filtered = {}
                                #sorted_confident_pairs_filtered = confident_pairs_combined
                                print(f"{iteration}: Tamanho antes do filtro: {len(confident_pairs_combined)}")

                                for pair in confident_pairs_combined.keys():
                                    
                                    if pair in all_pred_selected_pairs.keys():
                                        print('par'+ str(pair) + "ja apareceu")
                                    else:
                                        sorted_confident_pairs_filtered[pair]= confident_pairs_combined[pair]
                                #ATÉ AQUIII
                                print(f"{iteration}: Tamanho depois do filtro: {len(sorted_confident_pairs_filtered)}")
                                #dicionario (index, posição y) -> variancia
                                sorted_confident_pairs = sorted(sorted_confident_pairs_filtered.items(), key=lambda item: item[1])

                                pred_selected_pairs = {}
                            


                                for (i,j), _ in sorted_confident_pairs[:self.batch_size * target_length]:
                                    #NOTE MODIFICADO
                                    pred_selected_pairs[(i, j)]= (preds1.loc[i, columns[j]] + preds2.loc[i, columns[j]]) / 2
                                    all_pred_selected_pairs[(i, j)]= (preds1.loc[i, columns[j]] + preds2.loc[i, columns[j]]) / 2
                                

                                if not pred_selected_pairs:
                                    print("No confident predictions found.")
                                    break
                                else:
                                    print(pred_selected_pairs)
                                    
                                print(f"Before inclusion: {X_train_v1_df.shape}  X_train_v1")
                                print(f"Before inclusion: {y_train_df.shape}  y_train_df")
                                print(f"Before inclusion: {X_pool_v1_df.shape}  X_pool_v1_df")
                               
                                print()
                                print("Quantidade de pares selecionados: " + str(len(pred_selected_pairs)))
                                indices = set()
                                for idx_pool, j in pred_selected_pairs.keys():
                                    indices.add(idx_pool)
                                added_pairs_per_iteration.append(len(pred_selected_pairs))

                                print("Quantidade de linhas distintas "+str(len(indices)))
                                count = 0 
                                
                                for idx_pool, j in pred_selected_pairs.keys():
                                    
                                    if pd.notna(y_pool.loc[idx_pool, columns[j]]):
                                        print(f"A posicao ({idx_pool}, {j}) nao esta vazia.")
                                        continue
                                    
                                    # NOTE: added idx_pool not in indices conditions to guarantee we are not double stacking a certain X_train idx_pool
                                    if count == 0 and idx_pool not in indices:
                                        print('onde tudo começa...')
                                        print("count")
                                        print(count)
                                        print('par')
                                        print(idx_pool, j)
                                        print()
                                        print(f"Before inclusion: {X_train_v1_df.shape}  X_train_v1")
                                        #print(f"Before inclusion: {X_train_v2_df.shape}  X_train_v2")
                                        print(f"Before inclusion: {y_train_df.shape}  y_train_df")
                                        #print(f"Before inclusion: {X_pool_v1_df.shape}  X_pool_v1_df")
                                        #print(f"Before inclusion: {X_pool_v2_df.shape}  X_pool_v2_df")
                                        #print(f"Before inclusion: {y_pool.shape}  y_pool")
                                        print()

                                        #preenche x_train (adicionar novo) 
                                        #
                                        #x_train_v1, x_train_v2 e y_train_df vão ter o mesmo tamanho 
                                        X_train_v1_df = pd.concat([X_train_v1_df, X_pool_v1_df.loc[idx_pool]], ignore_index=False)
                                        X_train_v2_df = pd.concat([X_train_v2_df, X_pool_v2_df.loc[idx_pool]], ignore_index=False)
                                        
                                        #preenche y_pool (antigo y_labeled) (preencher)
                                        columns = list(y_pool.columns)
                                        
                                        #{índice train:índice pool}
                                        idx_train = X_train_v1_df.index[-1]
                                        dict_index[idx_train] = idx_pool

                                        # como pegar os indices de pool e de labeled direito?
                                        #preenche y_pool 
                                        y_pool.loc[idx_pool, columns[j]] = pred_selected_pairs[(idx_pool,j)]
                                        # preenche y_train
                                        y_train_df.loc[idx_train, columns[j]] = pred_selected_pairs[(idx_pool,j)]

                                        #OBS por enquanto ele não está dando reset no index: talvez seja importante em algum momento 
                                        count = count + 1
                                        #print(y_pool.loc[idx_pool])
                                        print()
                                        print(f"After inclusion: {X_train_v1_df.shape}  X_train_v1")
                                        #print(f"After inclusion: {X_train_v2_df.shape}  X_train_v2")
                                        print(f"After inclusion: {y_train_df.shape}  y_train_df")
                                        #print(f"After inclusion: {X_pool_v1_df.shape}  X_pool_v1_df")
                                        #print(f"After inclusion: {X_pool_v2_df.shape}  X_pool_v2_df")
                                        #print(f"After inclusion: {y_pool.shape}  y_pool")
                                        print()

                                    else:
                                        
                                        print("count")
                                        print(count)
                                        print('par')
                                        print(idx_pool, j)
                                        print(len(y_pool))
                                        print(y_pool.loc[idx_pool])
                                        print()
                                        print(f"Before inclusion: {X_train_v1_df.shape}  X_train_v1")
                                        #print(f"Before inclusion: {X_train_v2_df.shape}  X_train_v2")
                                        print(f"Before inclusion: {y_train_df.shape}  y_train_df")
                                        #print(f"Before inclusion: {X_pool_v1_df.shape}  X_pool_v1_df")
                                        #print(f"Before inclusion: {X_pool_v2_df.shape}  X_pool_v2_df")
                                        #print(f"Before inclusion: {y_pool.shape}  y_pool")
                                        
                                        #{índice train:índice pool}
                                        if idx_pool in dict_index.values():
                                            print('vai campeao')
                                            
                                            keys = [k for k, v in dict_index.items() if v == idx_pool]
                                            
                                            #print(f"O índice {idx_pool} está presente em X_train com id {keys}")
                                            
                                            y_pool.loc[idx_pool, columns[j]] = pred_selected_pairs[(idx_pool,j)]
                                            y_train_df.loc[keys[0], columns[j]] = pred_selected_pairs[(idx_pool,j)]
                                            #print(y_train_df.loc[keys[0]])

                                            #print()
                                            count = count + 1
                                            #print(y_pool.loc[idx_pool])
                                            print()
                                            print(f"After inclusion: {X_train_v1_df.shape}  X_train_v1")
                                            #print(f"After inclusion: {X_train_v2_df.shape}  X_train_v2")
                                            print(f"After inclusion: {y_train_df.shape}  y_train_df")
                                            #print(f"After inclusion: {X_pool_v1_df.shape}  X_pool_v1_df")
                                            #print(f"After inclusion: {X_pool_v2_df.shape}  X_pool_v2_df")
                                            #print(f"After inclusion: {y_pool.shape}  y_pool")
                                            print()


                                        else:
                                            #preenche x_train (adicionar novo) 
                                            
                                            #x_train_v1, x_train_v2 e y_train_df vão ter o mesmo tamanho 
                                            X_train_v1_df = pd.concat([X_train_v1_df,X_pool_v1_df.loc[[idx_pool]]], ignore_index=False)
                                            X_train_v2_df = pd.concat([X_train_v2_df, X_pool_v2_df.loc[[idx_pool]]], ignore_index=False)
                                            #linha_vazia = pd.DataFrame([[np.nan] * y_train_df.shape[1]], columns=y_train_df.columns)
                                            
                                            #y_train_df = pd.concat([y_train_df, linha_vazia], ignore_index=True)                        
                                            #preenche y_pool (antigo y_labeled) (preencher)
                                            columns = list(y_pool.columns)
                                            
                                            #(criar par índice pool e índice train)
                                            ultimo_index = X_train_v1_df.index[-1]
                                            dict_index[ultimo_index] = idx_pool

                                            # como pegar os indices de pool e de labeled direito?
                                            #preenche y_pool 
                                            y_pool.loc[idx_pool, columns[j]] = pred_selected_pairs[(idx_pool,j)]
                                            y_train_df.loc[ultimo_index, columns[j]] = pred_selected_pairs[(idx_pool,j)]
                                            #print(y_train_df.loc[ultimo_index])

                                            #print()
                                            #print(y_pool.loc[idx_pool])
                                            count = count + 1
                                            #print(y_pool.loc[idx_pool])
                                            print()
                                            print(f"After inclusion: {X_train_v1_df.shape}  X_train_v1")
                                            #print(f"After inclusion: {X_train_v2_df.shape}  X_train_v2")
                                            print(f"After inclusion: {y_train_df.shape}  y_train_df")
                                            #print(f"After inclusion: {X_pool_v1_df.shape}  X_pool_v1_df")
                                            #print(f"After inclusion: {X_pool_v2_df.shape}  X_pool_v2_df")
                                            #print(f"After inclusion: {y_pool.shape}  y_pool")
                                            print()

                                        print()
                                        print(f"After inclusion: {X_train_v1_df.shape}  X_train_v1")
                                        print(f"After inclusion: {X_train_v2_df.shape}  X_train_v2")
                                        print(f"After inclusion: {y_train_df.shape}  y_train_df")
                                        print(f"After inclusion: {X_pool_v1_df.shape}  X_pool_v1_df")
                                        print(f"After inclusion: {X_pool_v2_df.shape}  X_pool_v2_df")
                                        print(f"After inclusion: {y_pool.shape}  y_pool")

                                #checar se o y_pool está todo completo 
                                indices_linhas_completas = y_pool[y_pool.notna().all(axis=1)].index
                                print('linhas completas')
                                print(indices_linhas_completas)

                                if not indices_linhas_completas.empty:
                                    
                                    y_pool = y_pool.drop(indices_linhas_completas)
                                    X_pool_v1_df = X_pool_v1_df.drop(indices_linhas_completas)
                                    X_pool_v2_df = X_pool_v2_df.drop(indices_linhas_completas)

                                r2, mse, mae, ca, arrmse = model.unique_evaluate_model(models_view1_array, models_view2_array, X_test_v1, X_test_v2, y_test)
                                print("------------------------------------------------------------------------")
                                print(fold_index)
                                print(iteration)
                                # NOTE: changed from j to iteration. j on the CoTraining code refers only to iteration, but on TargetCoTraining.training, j means target instead

                                self.R2[fold_index, iteration] = r2
                                self.MSE[fold_index, iteration] = mse
                                self.MAE[fold_index, iteration] = mae
                                self.CA[fold_index, iteration] = ca
                                self.ARRMSE[fold_index, iteration] = arrmse
                                print("------------------------------------------------------------------------")

                            models_view1_array = model.unique_fit(target_length, y_train_df, X_train_v1_df)
                            models_view2_array = model.unique_fit(target_length, y_train_df, X_train_v2_df)
                            r2, mse, mae, ca, arrmse = model.unique_evaluate_model(models_view1_array, models_view2_array, X_test_v1, X_test_v2, y_test)
                            self.R2[fold_index, -1] = r2
                            self.MSE[fold_index, -1] = mse
                            self.MAE[fold_index, -1] = mae
                            self.CA[fold_index, -1] = ca
                            self.ARRMSE[fold_index, -1] = arrmse
                            print('saindo do training')
                            return added_pairs_per_iteration
                            #return models_view1_array, models_view2_array, X_train_v1_df, X_train_v2_df, X_pool_v1_df, X_pool_v2_df, y_labeled, execution_times, added_pairs_per_iteration
                        
                def train_and_evaluate(self, fold_index):

                    print(f"\nTraining model in fold {fold_index}...")
                    X_train_labeled, y_labeled, X_pool, y_pool, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length,target_names,feature_names = self.read_data(fold_index+1)

                    #self.train_original_model(X_train_labeled, y_labeled, X_test_labeled, y_test_labeled)

                    X_train_labeled_v1, X_train_labeled_v2,feature_names_v1,feature_names_v2 = self.split_features(X_train_labeled,feature_names)
                    X_pool_v1, X_pool_v2,feature_names_v1,feature_names_v2  = self.split_features(X_pool,feature_names)
                    X_test_labeled_v1, X_test_labeled_v2,feature_names_v1,feature_names_v2  = self.split_features(X_test_labeled,feature_names)
                    
                    added_pairs_per_iteration = self.training(
                        X_train_labeled_v1, X_train_labeled_v2, X_pool_v1, X_pool_v2, y_labeled, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled, target_length, fold_index,target_names,feature_names_v1,feature_names_v2,y_pool
                    )
                    

                    # Avaliação do modelo após o treinamento
                    #r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled)
                    
                
                    return self.R2, self.MSE, self.MAE, self.CA, self.ARRMSE,added_pairs_per_iteration
                    #, added_pairs_per_iteration
                    
        class SingleTargetRegressor():
            def __init__(self, random_state,n_trees):
                    self.random_state = random_state
                    self.n_trees=n_trees
                    
            def unique_evaluate_model(self, models_view1, models_view2, X_test_v1, X_test_v2, y_test_labeled):
                    columns = list(X_test_v1.columns)
                    print("Target Cotraining Making predictions on test data...")
                    predictions_v1 = pd.DataFrame(np.nan, index=X_test_v1.index, columns=y_test_labeled.columns)
                    predictions_v2 = pd.DataFrame(np.nan, index=X_test_v2.index, columns=y_test_labeled.columns)
                    #um modelo por target 
                    for i in range(len(models_view1)):
                        rf_model_v1 = models_view1[i]
                        rf_model_v2 = models_view2[i]

                            # Make predictions for the current tree
                        predictions_v1.iloc[:, i] = rf_model_v1.predict(X_test_v1)  # Atribuindo a predição na coluna correspondente
                        predictions_v2.iloc[:, i] = rf_model_v2.predict(X_test_v2)  # Atribuindo a predição na coluna correspondente

                    y_pred_combined = (predictions_v1 + predictions_v2) / 2
                    
                    r2 = np.round(r2_score(np.asarray(y_test_labeled), y_pred_combined), 4)

                    mse = np.round(mean_squared_error(np.asarray(y_test_labeled), y_pred_combined), 4)

                    mae = np.round(mean_absolute_error(np.asarray(y_test_labeled), y_pred_combined), 4)

                    #ca = np.round(custom_accuracy(np.asarray(y_test_labeled), y_pred_combined, self.threshold), 4)
                    ca = np.round(custom_accuracy(y_test_labeled.values, y_pred_combined.values, threshold=CA_THRESHOLD), 4)

                    # Converter os DataFrames em arrays NumPy antes de passá-los para a função
                    arrmse = np.round(arrmse_metric(np.asarray(y_test_labeled), np.asarray(y_pred_combined)), 4)

                    print(f"    Overall: R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}, CA={ca:.3f}, ARRMSE={arrmse:.3f}")
                
                    return r2, mse, mae, ca, arrmse

            def unique_predict(self, models, X_pool, target_length, columns):

                    # DataFrame para armazenar todas as previsões
                    predictions = pd.DataFrame(
                        data=[[None] * target_length for _ in range(len(X_pool))], 
                        columns=columns,
                        index=X_pool.index
                    )
                    
                    for i, model in enumerate(models):
                        predictions.iloc[:, i] = model.predict(X_pool)
                        
                    return predictions
            
            def unique_fit(self, target_length, y_train_df, X_train):

                    # Array para armazenar os modelos
                    model_array = []
                    columns = list(y_train_df.columns)
                    # Função para instanciar modelo

                    # Loop para cada coluna/target em y_train_df
                    for i in range(target_length):
                        # 1) Avaliar índices válidos para o target (não nulos)
                        valid_indices = ~y_train_df.iloc[:, i].isna()  # Índices válidos para o target atual

                        # Align the boolean Series with the DataFrame index
                        valid_indices = valid_indices.reindex(X_train.index, fill_value=False)

                        # 2) Pegar X_train válido
                        X_train_valid = X_train[valid_indices] 

                        # 3) Combinar os dados (se necessário) e pegar y_train válido para aquela coluna
                        y_train_valid = y_train_df.loc[valid_indices, columns[i]]

                        # 4) Ajustar o modelo para o target
                        model = RandomForestRegressor(random_state=self.random_state, n_estimators=self.n_trees)
                        model.fit(X_train_valid, y_train_valid)

                        # 5) Salvar o modelo no array
                        model_array.append(model)
            
                    # Retornar o array de modelos
                    return model_array


        if __name__ == "__main__":
                data_dir = config.DATA_DIR
                dataset_name = config.DATASET_NAME
                
                print('Target-based Co-Training...')
                cotraining_model = CoTraining(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
                X_train, y_labeled, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length,target_names,feature_names = cotraining_model.read_data(1)

                batch_size = round((batch_percentage / 100) * len(X_pool))

                target_cotraining_model = TargetCoTraining(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees, batch_size)

                for i in range(k_folds):
                    R2, MSE, MAE, CA, ARRMSE, added_pairs_per_iteration = target_cotraining_model.train_and_evaluate(i)
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
                    results_df.to_csv(results_path / f'target_cotraining_results_fold_{i}.csv', index=False)
                    print('Saved data...')

