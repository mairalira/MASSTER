from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, auc
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

class InstanceCoTrainingModel:
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
        # split the csv file in the input and target values
        folder_dir = data_dir / 'processed' / f'{self.dataset_name}'
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
    
    def read_data(self, iteration):
        X_train, y_train, _, _ = self.data_read(f'train_{iteration}')
        X_pool, y_pool, n_pool, target_length = self.data_read(f'pool_{iteration}')
        X_rest, y_rest, _, _ = self.data_read(f'train+pool_{iteration}')
        X_test, y_test, _, _ = self.data_read(f'test_{iteration}')
        return X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length

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
        print(f"Execution time (with original data): {execution_time:.2f} seconds\n")
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
        print("Calculating prediction confidence...")
        confident_mask1 = np.std(preds1, axis=1) <= self.threshold
        confident_mask2 = np.std(preds2, axis=1) <= self.threshold

        combined_mask = confident_mask1 | confident_mask2

        if not combined_mask.any():
            print("No confident predictions found.")
            return X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, False

        if confident_mask1.any():
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_unlabeled_v1[confident_mask1]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_unlabeled_v2[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[confident_mask1]])

        if confident_mask2.any():
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_unlabeled_v1[confident_mask2]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_unlabeled_v2[confident_mask2]])
            y_labeled = np.vstack([y_labeled, preds2[confident_mask2]])

        X_unlabeled_v1 = X_unlabeled_v1[~combined_mask]
        X_unlabeled_v2 = X_unlabeled_v2[~combined_mask]

        print(f"{confident_mask1.sum() + confident_mask2.sum()} examples added in this iteration.")
        
        assert X_train_labeled_v1.shape[0] == X_train_labeled_v2.shape[0], "Mismatch in sizes of X_train_labeled_v1 and X_train_labeled_v2"

        return X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, True

    def stop_criterion(self, preds1, preds2):
        return len(preds1) == 0 or len(preds2) == 0

    def training(self, model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled, fold_index):
        execution_times = []

        for j in range(self.iterations):
            print(f"Training model in epoch {j}...")
            start_time = time.time()

            if len(X_train_labeled_v1) == len(y_labeled):
                model_view1.fit(X_train_labeled_v1, y_labeled)
            else:
                print(f"Inconsistent number of samples: {len(X_train_labeled_v1)} in X_train_labeled_v1, {len(y_labeled)} in y_labeled")
                break

            if len(X_train_labeled_v2) == len(y_labeled):
                model_view2.fit(X_train_labeled_v2, y_labeled)
            else:
                print(f"Inconsistent number of samples: {len(X_train_labeled_v2)} in X_train_labeled_v2, {len(y_labeled)} in y_labeled")
                break

            preds1 = model_view1.predict(X_unlabeled_v1) if len(X_unlabeled_v1) > 0 else np.array([])
            preds2 = model_view2.predict(X_unlabeled_v2) if len(X_unlabeled_v2) > 0 else np.array([])

            if self.stop_criterion(preds1, preds2):
                print("No more unlabeled examples.")
                break

            X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, continue_training = self.confidence_computation(
                preds1, preds2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled
            )

            if not continue_training:
                break

            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled)
            self.R2[fold_index, j] = r2
            self.MSE[fold_index, j] = mse
            self.MAE[fold_index, j] = mae
            self.CA[fold_index, j] = ca
            self.ARRMSE[fold_index, j] = arrmse

        return model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, y_labeled, execution_times

    def evaluate_model(self, model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled):
        print("Making predictions on test data...")
        y_pred_v1 = model_view1.predict(X_test_labeled_v1)
        y_pred_v2 = model_view2.predict(X_test_labeled_v2)
        y_pred_combined = (y_pred_v1 + y_pred_v2) / 2

        r2 = np.round(r2_score(np.asarray(y_test_labeled), y_pred_combined), 4)
        mse = np.round(mean_squared_error(np.asarray(y_test_labeled), y_pred_combined), 4)
        mae = np.round(mean_absolute_error(np.asarray(y_test_labeled), y_pred_combined), 4)
        ca = np.round(custom_accuracy(np.asarray(y_test_labeled), y_pred_combined, self.threshold), 4)
        arrmse = np.round(arrmse_metric(np.asarray(y_test_labeled), y_pred_combined), 4)

        print(f"Overall: R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}, CA={ca:.3f}, ARRMSE={arrmse:.3f}")

        return r2, mse, mae, ca, arrmse

    def train_and_evaluate(self, fold_index):
        print(f"Training model in pool {fold_index}...")
        X_train_not_missing, Y_train_not_missing, X_unlabeled, Y_train_missing, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length = self.read_data(fold_index+1)

        self.train_original_model(X_train_not_missing, Y_train_not_missing, X_test_labeled, y_test_labeled)

        model_view1, model_view2 = self.initialize_models()
        X_train_labeled_v1, X_train_labeled_v2 = self.split_features(X_train_not_missing)
        X_unlabeled_v1, X_unlabeled_v2 = self.split_features(X_unlabeled)
        X_test_labeled_v1, X_test_labeled_v2 = self.split_features(X_test_labeled)

        y_labeled = Y_train_not_missing

        model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, y_labeled, execution_times = self.training(
            model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled, fold_index
        )

        r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled)
        self.R2[fold_index, -1] = r2
        self.MSE[fold_index, -1] = mse
        self.MAE[fold_index, -1] = mae
        self.CA[fold_index, -1] = ca
        self.ARRMSE[fold_index, -1] = arrmse

        return self.R2, self.MSE, self.MAE, self.CA, self.ARRMSE

if __name__ == "__main__":
    data_dir = config.DATA_DIR
    dataset_name = config.DATASET_NAME
    cotraining_model = InstanceCoTrainingModel(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
    
    for i in range(k_folds):
        cotraining_model.train_and_evaluate(i)
    
    R2, MSE, MAE, CA, ARRMSE = cotraining_model.R2, cotraining_model.MSE, cotraining_model.MAE, cotraining_model.CA, cotraining_model.ARRMSE
    print("Final Metrics:")
    print("R2:", R2)
    print("MSE:", MSE)
    print("MAE:", MAE)
    print("CA:", CA)
    print("ARRMSE:", ARRMSE)
