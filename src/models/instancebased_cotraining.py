from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, auc
import numpy as np
import time
import sys
import pandas as pd
import config
from pathlib import Path

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent
# Adding path to sys.path
sys.path.append(str(project_root))

from config import *
from data.data_processing import *
from utils.aux_active import read_data
from utils.metrics import custom_accuracy

# Main script
data_dir = config.DATA_DIR
dataset = config.DATASET_NAME

k_folds = config.K_FOLDS
iterations = config.ITERATIONS
threshold = config.THRESHOLD
random_state = config.RANDOM_STATE
n_trees = config.N_TREES

class InstanceCoTrainingModel:
    def __init__(self, data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.k_folds = k_folds
        self.iterations = iterations
        self.threshold = threshold
        self.random_state = random_state
        self.n_trees = n_trees

    def train_original_model(self, X_train_labeled, y_train_labeled, X_test_labeled, y_test_labeled):
        print("Desempenho Original:")
        start_time = time.time()
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state))
        model.fit(X_train_labeled, y_train_labeled)
        y_pred = model.predict(X_test_labeled)
        execution_time = time.time() - start_time
        for j in range(len(y_test_labeled[0])):
            r2 = r2_score([row[j] for row in y_test_labeled], [row[j] for row in y_pred])
            mae = mean_absolute_error([row[j] for row in y_test_labeled], [row[j] for row in y_pred])
            ca = custom_accuracy([row[j] for row in y_test_labeled], [row[j] for row in y_pred], self.threshold)
            print(f"Target {j+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
        print(f"Tempo de execução (com dados originais): {execution_time:.2f} segundos\n")
        return model

    def initialize_models(self):
        model_view1 = MultiOutputRegressor(RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state))
        model_view2 = MultiOutputRegressor(RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state))
        return model_view1, model_view2

    def split_features(self, X):
        X = np.array(X)
        X_v1 = X[:, :int(X.shape[1]/2)]
        X_v2 = X[:, int(X.shape[1]/2):]
        return X_v1, X_v2

    def confidence_computation(self, preds1, preds2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled):
        print("Calculando a confiança das previsões...")
        confident_mask1 = np.std(preds1, axis=1) <= self.threshold
        confident_mask2 = np.std(preds2, axis=1) <= self.threshold

        if confident_mask1.any() and confident_mask2.any():
            print(f"Adicionando {confident_mask1.sum()} exemplos confiáveis da visão 1.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_unlabeled_v1[confident_mask1]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_unlabeled_v2[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds2[confident_mask2]])

        if not confident_mask1.any() and not confident_mask2.any():
            print("Nenhuma previsão confiável encontrada.")
            return X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, False

        if confident_mask1.any():
            print(f"Adicionando {confident_mask1.sum()} exemplos confiáveis da visão 1.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_unlabeled_v1[confident_mask1]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_unlabeled_v2[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[confident_mask1]])

        if confident_mask2.any():
            print(f"Adicionando {confident_mask2.sum()} exemplos confiáveis da visão 2.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_unlabeled_v1[confident_mask2]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_unlabeled_v2[confident_mask2]])
            y_labeled = np.vstack([y_labeled, preds2[confident_mask2]])

        print("Removendo exemplos confiáveis do conjunto não rotulado...")
        X_unlabeled_v1 = X_unlabeled_v1[~confident_mask1]
        X_unlabeled_v2 = X_unlabeled_v2[~confident_mask2]

        print(f"{confident_mask1.sum() + confident_mask2.sum()} exemplos adicionados nesta iteração.")
        return X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, True

    def stop_criterion(self, preds1, preds2):
        if len(preds1) == 0 or len(preds2) == 0:
            print("Sem mais exemplos não rotulados.")
            return True
        return False

    def training(self, model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled):
        execution_times = []
        for j in range(self.iterations):
            print(f"Treinando modelo na epoch {j}...")
            start_time = time.time()
            print("Iniciando treinamento com co-training...")

            print("Treinando modelo 1 com visão 1...")
            model_view1.fit(X_train_labeled_v1, y_labeled)
            print("Treinando modelo 2 com visão 2...")
            model_view2.fit(X_train_labeled_v2, y_labeled)

            print("Fazendo previsões para dados não rotulados...")
            preds1 = model_view1.predict(X_unlabeled_v1) if len(X_unlabeled_v1) > 0 else np.array([])
            preds2 = model_view2.predict(X_unlabeled_v2) if len(X_unlabeled_v2) > 0 else np.array([])

            if self.stop_criterion(preds1, preds2):
                break

            X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, continue_training = self.confidence_computation(
                preds1, preds2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled
            )

            if not continue_training:
                break

            execution_time = time.time() - start_time
            execution_times.append(execution_time)

        return model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, y_labeled, execution_times

    def evaluate_model(self, model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled):
        print("Fazendo previsões nos dados de teste...")
        y_pred_v1 = model_view1.predict(X_test_labeled_v1)
        y_pred_v2 = model_view2.predict(X_test_labeled_v2)
        y_pred_combined = (y_pred_v1 + y_pred_v2) / 2

        for j in range(len(y_test_labeled[0])):
            r2 = r2_score([row[j] for row in y_test_labeled], [row[j] for row in y_pred_combined])
            mae = mean_absolute_error([row[j] for row in y_test_labeled], [row[j] for row in y_pred_combined])
            ca = custom_accuracy([row[j] for row in y_test_labeled], [row[j] for row in y_pred_combined], self.threshold)
            print(f"Target {j+1}: R²={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")

    def train_and_evaluate(self):
        for i in range(self.k_folds):
            print(f"Treinando modelo no pool {i}...")
            X_train_not_missing, Y_train_not_missing, X_unlabeled, Y_train_missing, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length = read_data(self, i+1)

            self.train_original_model(X_train_not_missing, Y_train_not_missing, X_test_labeled, y_test_labeled)

            print("Desempenho Semissupervisionado")
            model_view1, model_view2 = self.initialize_models()
            print("Dividindo as features em duas visões...")

            X_train_labeled_v1, X_train_labeled_v2 = self.split_features(X_train_not_missing)
            X_unlabeled_v1, X_unlabeled_v2 = self.split_features(X_unlabeled)
            X_test_labeled_v1, X_test_labeled_v2 = self.split_features(X_test_labeled)

            y_labeled = Y_train_not_missing

            model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, y_labeled, execution_times = self.training(
                model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled
            )

            self.evaluate_model(model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled)

if __name__ == "__main__":
    data_dir = config.DATA_DIR
    dataset_name = config.DATASET_NAME
    cotraining_model = InstanceCoTrainingModel(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
    cotraining_model.train_and_evaluate()
