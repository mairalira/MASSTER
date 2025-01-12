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
        # split the csv file into input and target values
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
        print(f"    Execution time (with original data): {execution_time:.2f} seconds\n")
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
        print(" Calculating prediction confidence...")
        confident_mask1 = np.std(preds1, axis=1) <= self.threshold
        confident_mask2 = np.std(preds2, axis=1) <= self.threshold

        combined_mask = confident_mask1 | confident_mask2

        if not combined_mask.any():
            print(" No confident predictions found.")
            return X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, combined_mask, False

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

        print(f"    {confident_mask1.sum() + confident_mask2.sum()} examples added in this iteration.")
        
        assert X_train_labeled_v1.shape[0] == X_train_labeled_v2.shape[0], "    Mismatch in sizes of X_train_labeled_v1 and X_train_labeled_v2"

        return X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, combined_mask, True
    
    def stop_criterion(self, preds1, preds2):
        return len(preds1) == 0 or len(preds2) == 0

    def training(self, model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled, fold_index):
        execution_times = []
        added_pairs_per_iteration = []
        original_indices = np.arange(len(X_unlabeled_v1))  # Track original indices

        for j in range(self.iterations):
            print(f"    Training model in epoch {j}...")
            start_time = time.time()

            if len(X_train_labeled_v1) == len(y_labeled):
                model_view1.fit(X_train_labeled_v1, y_labeled)
            else:
                print(f"    Inconsistent number of samples: {len(X_train_labeled_v1)} in X_train_labeled_v1, {len(y_labeled)} in y_labeled")
                break

            if len(X_train_labeled_v2) == len(y_labeled):
                model_view2.fit(X_train_labeled_v2, y_labeled)
            else:
                print(f"    Inconsistent number of samples: {len(X_train_labeled_v2)} in X_train_labeled_v2, {len(y_labeled)} in y_labeled")
                break

            preds1 = model_view1.predict(X_unlabeled_v1) if len(X_unlabeled_v1) > 0 else np.array([])
            preds2 = model_view2.predict(X_unlabeled_v2) if len(X_unlabeled_v2) > 0 else np.array([])

            if self.stop_criterion(preds1, preds2):
                print(" No more unlabeled examples.")
                break

            X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, combined_mask, continue_training = self.confidence_computation(
                preds1, preds2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled
            )

            if not continue_training:
                break

            # Track added pairs (original_instance_index, target_number)
            added_pairs = []
            confident_indices = np.where(combined_mask)[0]
            for idx in confident_indices:
                original_idx = original_indices[idx]
                for target_idx in range(y_labeled.shape[1]):
                    added_pairs.append((original_idx, target_idx))
            added_pairs_per_iteration.append(added_pairs)

            # Update original indices after removing confident examples
            original_indices = original_indices[~combined_mask]

            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled)
            self.R2[fold_index, j] = r2
            self.MSE[fold_index, j] = mse
            self.MAE[fold_index, j] = mae
            self.CA[fold_index, j] = ca
            self.ARRMSE[fold_index, j] = arrmse

        return model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, y_labeled, execution_times, added_pairs_per_iteration

    def evaluate_model(self, model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled):
        print(" Making predictions on test data...")
        y_pred_v1 = model_view1.predict(X_test_labeled_v1)
        y_pred_v2 = model_view2.predict(X_test_labeled_v2)
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
        X_train_not_missing, Y_train_not_missing, X_unlabeled, Y_train_missing, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length = self.read_data(fold_index+1)

        self.train_original_model(X_train_not_missing, Y_train_not_missing, X_test_labeled, y_test_labeled)

        model_view1, model_view2 = self.initialize_models()
        X_train_labeled_v1, X_train_labeled_v2 = self.split_features(X_train_not_missing)
        X_unlabeled_v1, X_unlabeled_v2 = self.split_features(X_unlabeled)
        X_test_labeled_v1, X_test_labeled_v2 = self.split_features(X_test_labeled)

        y_labeled = Y_train_not_missing

        model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, y_labeled, execution_times, added_pairs_per_iteration = self.training(
            model_view1, model_view2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled, fold_index
        )

        r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_labeled_v1, X_test_labeled_v2, y_test_labeled)
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

    def confidence_computation(self, preds1, preds2, X_train_labeled_v1, X_train_labeled_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled):
        print(" Calculating prediction confidence...")
        confident_mask1 = np.std(preds1, axis=1) <= self.threshold
        confident_mask2 = np.std(preds2, axis=1) <= self.threshold

        combined_mask = confident_mask1 | confident_mask2

        if not combined_mask.any():
            print(" No confident predictions found.")
            return X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, combined_mask, False

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
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_unlabeled_v1[top_confident_mask1]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_unlabeled_v2[top_confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[top_confident_mask1]])

        if top_confident_mask2.any():
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_unlabeled_v1[top_confident_mask2]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_unlabeled_v2[top_confident_mask2]])
            y_labeled = np.vstack([y_labeled, preds2[top_confident_mask2]])

        X_unlabeled_v1 = np.delete(X_unlabeled_v1, confident_indices, axis=0)
        X_unlabeled_v2 = np.delete(X_unlabeled_v2, confident_indices, axis=0)

        top_combined_mask = top_confident_mask1 | top_confident_mask2

        print(f"{len(confident_indices)} examples added in this iteration.")
        
        assert X_train_labeled_v1.shape[0] == X_train_labeled_v2.shape[0], "Mismatch in sizes of X_train_labeled_v1 and X_train_labeled_v2"

        return X_train_labeled_v1, X_train_labeled_v2, y_labeled, X_unlabeled_v1, X_unlabeled_v2, top_combined_mask, True

class TargetCoTraining(CoTraining):
    def __init__(self, data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees, batch_size):
        super().__init__(data_dir, dataset_name, k_folds, iterations, threshold, random_state, n_trees)
        self.batch_size = batch_size

    def calculate_variances(self, model, X_pool, target_length):
        variances = np.zeros((len(X_pool), target_length))
        preds = np.zeros((len(X_pool), target_length))
        for i in range(target_length):
            pool_preds = np.zeros((len(X_pool), len(model.estimators_)))
            for j, estimator in enumerate(model.estimators_):
                pool_preds[:, j] = estimator.predict(X_pool)
            for k, preds_k in enumerate(pool_preds):
                variances[k, i] = variance(preds_k)
                preds[k, i] = np.mean(preds_k)
        return variances, preds
    
    def select_confident_pairs(self, variances, preds1, preds2):
        confident_pairs = []
        for i in range(variances.shape[0]):
            for j in range(variances.shape[1]):
                if variances[i, j] <= self.threshold:
                    confident_pairs.append((i, j, preds1[i, j], preds2[i, j], variances[i, j]))
        return confident_pairs
    
    def training(self, model_view1, model_view2, X_train_v1, X_train_v2, X_unlabeled_v1, X_unlabeled_v2, y_labeled, X_test_v1, X_test_v2, y_test, target_length, fold_index):
        execution_times = []
        added_pairs_per_iteration = []
        original_indices = list(range(len(X_unlabeled_v1)))
        instance_target_count = {i: 0 for i in range(len(X_unlabeled_v1))}
        instance_mapping = {i: original_indices[i] for i in range(len(original_indices))}
        selected_pairs_set = set()

        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")

            start_time = time.time()

            model_view1.fit(X_train_v1, y_labeled)
            model_view2.fit(X_train_v2, y_labeled)

            preds1 = model_view1.predict(X_unlabeled_v1)
            preds2 = model_view2.predict(X_unlabeled_v2)

            variances1, preds1 = self.calculate_variances(model_view1, X_unlabeled_v1, target_length)
            variances2, preds2 = self.calculate_variances(model_view2, X_unlabeled_v2, target_length)

            print('Confident pairs eval')
            confident_pairs1 = self.select_confident_pairs(variances1, preds1, preds2)
            confident_pairs2 = self.select_confident_pairs(variances2, preds2, preds1)

            combined_pairs = list(set(confident_pairs1).union(confident_pairs2))
            combined_pairs = sorted(combined_pairs, key=lambda x: x[4])

            selected_pairs = combined_pairs[:(self.batch_size * target_length)]
            added_pairs = []

            if not selected_pairs:
                print(" No confident predictions found.")
                return model_view1, model_view2, X_train_v1, X_train_v2, y_labeled, execution_times, added_pairs_per_iteration

            top_confident_mask1 = np.zeros(preds1.shape, dtype=bool)
            top_confident_mask2 = np.zeros(preds2.shape, dtype=bool)

            # Save the pairs of (instance, target) along with their variances and predictions
            for i, j, pred1, pred2, _ in selected_pairs:
                if (i, j) in selected_pairs_set:
                    continue

                top_confident_mask1[i, j] = True
                top_confident_mask2[i, j] = True
                X_train_v1 = np.vstack([X_train_v1, X_unlabeled_v1[i]])
                X_train_v2 = np.vstack([X_train_v2, X_unlabeled_v2[i]])

                y_labeled_instance_v1 = model_view1.predict(X_unlabeled_v1[i].reshape(1, -1))
                y_labeled_instance_v2 = model_view2.predict(X_unlabeled_v2[i].reshape(1, -1))

                y_labeled_instance = (y_labeled_instance_v1 + y_labeled_instance_v2) / 2

                if y_labeled_instance.ndim == 1:
                    y_labeled_instance = y_labeled_instance.reshape(1, -1)

                y_labeled = np.vstack([y_labeled, y_labeled_instance])
                added_pairs.append((i, j))
                instance_target_count[i] += 1
                selected_pairs_set.add((i, j))

            added_pairs_per_iteration.append(added_pairs)

            # Remove instances with all targets added to y_labeled
            remove_confident_indices = [i for i, count in instance_target_count.items() if count == target_length]
            X_unlabeled_v1 = np.delete(X_unlabeled_v1, remove_confident_indices, axis=0)
            X_unlabeled_v2 = np.delete(X_unlabeled_v2, remove_confident_indices, axis=0)
            
            # Update instance_mapping and instance_target_count
            instance_mapping = {new_idx: instance_mapping[old_idx] for new_idx, old_idx in enumerate(range(len(X_unlabeled_v1)))}
            instance_target_count = {new_idx: instance_target_count[old_idx] for new_idx, old_idx in enumerate(range(len(X_unlabeled_v1)))}

            print(f"{len(added_pairs)} (instance, target) pairs added in this iteration.")
            execution_time = time.time() - start_time
            execution_times.append(execution_time)

            r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_v1, X_test_v2, y_test)
            self.R2[fold_index, iteration] = r2
            self.MSE[fold_index, iteration] = mse
            self.MAE[fold_index, iteration] = mae
            self.CA[fold_index, iteration] = ca
            self.ARRMSE[fold_index, iteration] = arrmse

            if self.stop_criterion(preds1, preds2):
                break

        return model_view1, model_view2, X_train_v1, X_train_v2, y_labeled, execution_times, added_pairs_per_iteration

    def train_and_evaluate(self, fold_index):
        print(f"\nTraining model in fold {fold_index}...")
        X_train_not_missing, Y_train_not_missing, X_unlabeled, Y_train_missing, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length = self.read_data(fold_index + 1)

        self.train_original_model(X_train_not_missing, Y_train_not_missing, X_test_labeled, y_test_labeled)

        model_view1, model_view2 = self.initialize_models()
        X_train_v1, X_train_v2 = self.split_features(X_train)
        X_pool_v1, X_pool_v2 = self.split_features(X_pool)
        X_test_v1, X_test_v2 = self.split_features(X_test)

        y_labeled = Y_train_not_missing

        model_view1, model_view2, X_train_v1, X_train_v2, y_labeled, execution_times, added_pairs_per_iteration = self.training(
            model_view1, model_view2, X_train_v1, X_train_v2, X_pool_v1, X_pool_v2, y_labeled, X_test_v1, X_test_v2, y_test_labeled, target_length, fold_index
        )

        # Avaliação do modelo após o treinamento
        r2, mse, mae, ca, arrmse = self.evaluate_model(model_view1, model_view2, X_test_v1, X_test_v2, y_test_labeled)
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
    X_train, y_labeled, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length = cotraining_model.read_data(1)
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

