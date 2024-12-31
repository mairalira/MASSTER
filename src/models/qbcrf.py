""" The subclass for QBC-RF. """
from models.active_learning import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statistics import variance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, auc
from utils.metrics import *

class qbcrf(activelearning):
    def __init__(self, batch_size, n_epochs):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epochs = [i for i in range(n_epochs)]
        self.instances_pool_qbcrf = list()
        self.targets_pool_qbcrf = list()
        self.qbcrf_R2 = np.zeros([self.iterations, self.n_epochs+1])
        self.qbcrf_MSE = np.zeros([self.iterations, self.n_epochs+1])
        self.qbcrf_MAE = np.zeros([self.iterations, self.n_epochs+1])
        self.qbcrf_CA = np.zeros([self.iterations, self.n_epochs+1])
        self.qbcrf_aRRMSE = np.zeros([self.iterations, self.n_epochs+1])

    def variances(self, X_train, X_pool, X_test, y_train, y_test, target_length):
        # Convert X_train and y_train to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # calculate the variances and the test set predictions
        variances = np.zeros(shape=(len(X_pool), target_length))
        y_test_preds = np.zeros(shape=(len(X_test), target_length))

        for i in range(target_length):
            var = list()

            # Filter out rows with NaN values in y_train for the current target
            valid_indices = ~np.isnan(y_train[:, i])
            X_train_valid = X_train[valid_indices]
            y_train_valid = y_train[valid_indices, i]

            # first, fit a model to the training set for the selected target
            rf_regressor = RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state)
            rf_regressor.fit(X_train_valid, y_train_valid)
            
            # next, predict the target values of the test set
            test_preds = rf_regressor.predict(X_test)
            y_test_preds[:,i] = np.round(test_preds, 5)

            # at last, calculate the variance of the predictions on the unlabelled pool dataset
            pool_preds = np.zeros(shape=(len(X_pool), len(rf_regressor.estimators_)))
            for j, model in enumerate(rf_regressor.estimators_):
                pool_preds[:,j] = model.predict(X_pool) 
            
            for k, model_preds in enumerate(pool_preds):
                var_value = variance(model_preds)
                var.append(var_value)
            variances[:,i] = var
        
        return variances, y_test, y_test_preds
    
    def max_var(self, variances, k):
        # Flatten the variances array to get pairs of (variance, instance_index, target_index)
        flattened_variances = [(var, i, j) for i, instance_variances in enumerate(variances) for j, var in enumerate(instance_variances)]
        
        # Sort the flattened variances by variance value in descending order
        sorted_variances = sorted(flattened_variances, key=lambda x: x[0], reverse=True)
        
        # Select the top-k variances
        top_k_variances = sorted_variances[:k]
        
        # Extract the instance indices and target indices
        instance_indices = [item[1] for item in top_k_variances]
        target_indices = [item[2] for item in top_k_variances]
        
        return top_k_variances, instance_indices, target_indices
    
    def target_collect(self, targets, target_length):
        # Collect all the values for specific targets in separate lists
        targets_collected = [[] for _ in range(target_length)]
        for target in targets:
            if isinstance(target, (list, np.ndarray)):
                for i in range(target_length):
                    targets_collected[i].append(target[i])
            else:
                for i in range(target_length):
                    targets_collected[i].append(target)
        return targets_collected
    
    def training(self, X_train, X_pool, X_test, y_train, y_pool, y_test, target_length):
        # the instance based active learning training loop
        R2 = np.zeros([1, self.n_epochs+1])
        MSE = np.zeros([1, self.n_epochs+1])
        MAE = np.zeros([1, self.n_epochs+1])
        CA = np.zeros([1, self.n_epochs+1])
        ARRMSE = np.zeros([1, self.n_epochs+1])
        Y_pred = np.zeros([len(X_test), target_length*self.n_epochs])
        instances_pool_qbcrf = list()
        targets_pool_qbcrf = list()
        selected_pairs = set()
        percentage_targets_provided = np.zeros(self.n_epochs)

        # Convert X_pool and y_pool to NumPy arrays
        X_pool = np.array(X_pool)
        y_pool = np.array(y_pool)
        original_indices = list(range(len(X_pool)))

        for i in range(self.n_epochs):
            print("Epoch {}:".format(i+1))
            print("     The training set size: {}".format(len(X_train)))
            print("     The unlabelled pool size: {}".format(len(X_pool)))

            # Ensure y_train_coll has the same number of samples as X_train
            y_train_coll = self.target_collect(y_train, target_length)

            # Convert y_train_coll to a NumPy array
            y_train_coll = np.array(y_train_coll).T

            variances, y_test, y_test_preds = self.variances(X_train, X_pool, X_test, y_train_coll, y_test, target_length)
            Y_pred[:,(i*target_length):((i+1)*target_length)] = y_test_preds
            r2 = (np.round(r2_score(np.asarray(y_test), y_test_preds), 4))
            mse = (np.round(mean_squared_error(np.asarray(y_test), y_test_preds), 4))
            mae = (np.round(mean_absolute_error(np.asarray(y_test), y_test_preds), 4))
            ca = (np.round(custom_accuracy(np.asarray(y_test), y_test_preds), 4))
            arrmse = (np.round(arrmse_metric(np.asarray(y_test), y_test_preds), 4))
            
            R2[:,i] = (r2)
            MSE[:,i] = (mse)
            MAE[:,i] = (mae)
            CA[:,i] = (ca)
            ARRMSE[:,i] = (arrmse)
            
            # Select top-k instances with highest variance
            k = self.batch_size * target_length
            top_k_variances, instance_indices, target_indices = self.max_var(variances, k)
            
            # Map instance indices to original indices
            mapped_indices = [original_indices[idx] for idx in instance_indices]

            # Sort indices in descending order (largest index first)
            sorted_indices = sorted(zip(mapped_indices, target_indices), key=lambda x: x[0], reverse=True)

            # Collect instances and targets to be removed after the epoch
            targets_to_remove = []

            # Reverse iterate through the sorted indices
            for idx, target_idx in sorted_indices:
                
                # Validate the index against current pool size
                if idx >= len(X_pool):
                    print(f"Skipping invalid index: {idx} (current pool size: {len(X_pool)})")
                    continue

                # Check if the pair (instance, target) has already been selected
                if (idx, target_idx) in selected_pairs:
                    continue

                # Append the selected instance and target value to respective lists
                instances_pool_qbcrf.append(X_pool[idx].flatten())
                targets_pool_qbcrf.append((idx, target_idx, y_pool[idx, target_idx]))

                # Collect the target indices for removal
                targets_to_remove.append((idx, target_idx))

                # Mark the pair as selected
                selected_pairs.add((idx, target_idx))

            # Append selected instances to training set
            X_train = np.vstack([X_train, X_pool[list(set([idx for idx, _ in targets_to_remove]))]])
            y_train = np.vstack([y_train, y_pool[list(set([idx for idx, _ in targets_to_remove]))]])

            # Remove the selected targets from the pool after the epoch
            for idx, target_idx in targets_to_remove:
                y_pool[idx, target_idx] = np.nan  # Mark the target as removed

            # Remove rows with all targets removed
            mask = ~np.isnan(y_pool).all(axis=1)
            X_pool = X_pool[mask]
            y_pool = y_pool[mask]

            # Update original_indices after removal
            original_indices = list(range(len(X_pool)))

            # Filter out rows with NaN values in y_train
            valid_indices = ~np.isnan(y_train).any(axis=1)
            X_train = X_train[valid_indices]
            y_train = y_train[valid_indices]

            # Calculate the percentage of targets provided for the current epoch
            total_targets = len(X_pool) * target_length
            provided_targets = len(targets_pool_qbcrf)
            percentage_provided = (provided_targets / total_targets) * 100
            percentage_targets_provided[i] = percentage_provided
            print(f'Percentage of targets in epoch {i}: {percentage_provided}')

        r2_auc = np.round(auc(self.epochs, R2[0,:-1]), 4)
        mse_auc = np.round(auc(self.epochs, MSE[0,:-1]), 4)
        mae_auc = np.round(auc(self.epochs, MAE[0,:-1]), 4) 
        ca_auc = np.round(auc(self.epochs, CA[0,:-1]), 4) 
        arrmse_auc = np.round(auc(self.epochs, ARRMSE[0,:-1]), 4) 

        R2[:,-1] = (r2_auc)
        MSE[:,-1] = (mse_auc)
        MAE[:,-1] = (mae_auc)
        CA[:,-1] = (ca_auc)
        ARRMSE[:,-1] = (arrmse_auc)

        cols = ["Target_{}".format(i+1) for epoch in range(self.n_epochs) for i in range(target_length)]
        Y_pred_df = pd.DataFrame(Y_pred, columns=cols)

        # Create a DataFrame to store the transfer targets
        df = pd.DataFrame(columns=range(target_length))

        # Populate the DataFrame with the transfer targets
        for instance_idx, target_idx, target_value in targets_pool_qbcrf:
            if instance_idx not in df.index:
                df.loc[instance_idx] = [None] * target_length
            df.at[instance_idx, target_idx] = target_value

        transfer_targets_qbcrf_df = df

        return R2, MSE, MAE, CA, ARRMSE, Y_pred_df, instances_pool_qbcrf, transfer_targets_qbcrf_df, percentage_targets_provided