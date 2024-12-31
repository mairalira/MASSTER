""" The subclass for RT-AL. """
from models.active_learning import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances, r2_score, mean_squared_error, mean_absolute_error, auc
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from utils.metrics import *  
from scipy.cluster.vq import vq  
import os
import warnings
# Set the environment variable to avoid memory leak
os.environ["OMP_NUM_THREADS"] = "1"  
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

class rtal(activelearning):
    def __init__(self, batch_size, n_epochs, n_clusters=10):  # Add n_clusters with a default value
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_clusters = n_clusters  # Initialize n_clusters
        self.epochs = [i for i in range(n_epochs)]
        self.instances_pool_rtal = list()
        self.targets_pool_rtal = list()
        self.rtal_R2 = np.zeros([self.iterations, self.n_epochs+1])
        self.rtal_MSE = np.zeros([self.iterations, self.n_epochs+1])
        self.rtal_MAE = np.zeros([self.iterations, self.n_epochs+1])
        self.rtal_CA = np.zeros([self.iterations, self.n_epochs+1])
        self.rtal_aRRMSE = np.zeros([self.iterations, self.n_epochs+1])  

    def get_leaf_indices(self, dt_model, X):
        """Get leaf indices for each instance in X using the trained Decision Tree model."""
        leaf_indices = dt_model.apply(X)
        return leaf_indices

    def calculate_variance(self, dt_model, X):
        """Calculate variance of the predictions for each instance in X using the trained Decision Tree model."""
        leaf_indices = dt_model.apply(X)
        unique_leaves = np.unique(leaf_indices)
        variances = np.zeros(X.shape[0])
        for leaf in unique_leaves:
            leaf_instances = np.where(leaf_indices == leaf)[0]
            if len(leaf_instances) > 1:
                leaf_predictions = dt_model.predict(X[leaf_instances])
                variances[leaf_instances] = np.var(leaf_predictions)
        return variances

    def calculate_probability(self, cluster_labels, n_clusters):
        """Calculate the probability for each instance based on cluster membership."""
        probabilities = np.zeros(len(cluster_labels))
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_size = len(cluster_indices)
            if cluster_size > 0:
                probabilities[cluster_indices] = 1 / cluster_size
        return probabilities

    def calculate_representativeness(self, X_pool, cluster_labels, kmeans):
        """Calculate representativeness for each instance based on cluster membership."""
        representativeness = np.zeros(len(X_pool))
        num_cluster = []
        data_clust = []
        data_clust_index = []
        new_cent = []
        rx = []
        last_leaf_end = 0

        for num in range(len(kmeans.cluster_centers_)):
            num_cluster.append(len(np.where(kmeans.labels_ == num)[0]))
        number_cluster_per_leaf = np.array(num_cluster).astype(int)

        closest = []
        for i in range(len(kmeans.cluster_centers_)):
            data_cluster = []
            for j in range(int(num_cluster[i + last_leaf_end])):
                data_cluster.append(X_pool[np.where(kmeans.labels_ == i)[0][j]])
                data_clust.append(X_pool[np.where(kmeans.labels_ == i)[0][j]])
                data_clust_index.append(np.where(kmeans.labels_ == i)[0][j])

            centroid = kmeans.cluster_centers_[i + last_leaf_end].reshape(1, -1)
            closest_ind, distances = vq(centroid, data_cluster)
            closest.append(np.where(kmeans.labels_ == i)[0][closest_ind])

        for s in closest:
            new_cent.append(X_pool[int(s)])
        new_cent = np.array(new_cent)

        for p in range(len(kmeans.cluster_centers_)):
            rxx = np.zeros(num_cluster[p + last_leaf_end])
            for q in range(num_cluster[p + last_leaf_end]):
                for k in range(num_cluster[p + last_leaf_end]):
                    rxx[q] += np.linalg.norm(X_pool[np.where(kmeans.labels_ == p)[0][k]] - X_pool[np.where(kmeans.labels_ == p)[0][q]])
                if num_cluster[p + last_leaf_end] > 1:
                    rxx[q] = rxx[q] / (num_cluster[p + last_leaf_end] - 1)
                else:
                    rxx[q] = 0  # Avoid division by zero
            rx.append(rxx)

        last_leaf_end = len(num_cluster)
        representativeness[data_clust_index] = np.concatenate(rx)

        return representativeness

    def calculate_diversity(self, X_pool, X_train):
        """Calculate diversity of the instances based on minimum distance to labeled instances."""
        diversity = np.zeros(len(X_pool))
        for i, x in enumerate(X_pool):
            diversity[i] = np.min(np.linalg.norm(X_train - x, axis=1))
        return diversity

    def calculate_combined_score(self, variance, representativeness, diversity, alpha=0.5):
        """Calculate the combined score using variance, representativeness, and diversity."""
        if variance.ndim > 1:
            variance = variance.mean(axis=1)
        if diversity.ndim > 1:
            diversity = diversity.mean(axis=1)
        combined_score = variance - representativeness + alpha * diversity
        return combined_score

    def cluster_instances(self, X_pool, n_clusters):
        """Cluster the instances in the pool using KMeans."""
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)  # Explicitly set n_init
        cluster_labels = kmeans.fit_predict(X_pool)
        return cluster_labels

    def variances(self, X_train, X_pool, X_test, y_train, y_test, target_length):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Ensure X_train and y_train have the same number of rows
        if X_train.shape[0] != y_train.shape[0]:
            min_length = min(X_train.shape[0], y_train.shape[0])
            X_train = X_train[:min_length]
            y_train = y_train[:min_length]

        variances = np.zeros(shape=(len(X_pool), target_length))
        representativenesses = np.zeros(shape=(len(X_pool), target_length))
        diversities = np.zeros(shape=(len(X_pool), target_length))
        y_test_preds = np.zeros(shape=(len(X_test), target_length))

        for i in range(target_length):
            # Filter out rows with NaN values in y_train for the current target
            valid_indices = ~np.isnan(y_train[:, i])
            X_train_valid = X_train[valid_indices]
            y_train_valid = y_train[valid_indices, i]

            # Use DecisionTreeRegressor to compute variance
            dt_regressor = DecisionTreeRegressor(random_state=self.random_state)
            dt_regressor.fit(X_train_valid, y_train_valid)
            leaf_indices = dt_regressor.apply(X_pool)

            # Perform KMeans clustering on each leaf
            cluster_labels = np.zeros(len(X_pool), dtype=int)
            for leaf in np.unique(leaf_indices):
                leaf_instances = np.where(leaf_indices == leaf)[0]
                if len(leaf_instances) > 1:
                    data_leaf = X_pool[leaf_instances]
                    n_clusters = len(leaf_instances)  # Set n_clusters to the number of instances in the leaf
                    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=self.random_state)
                    kmeans.fit(data_leaf)
                    cluster_labels[leaf_instances] = kmeans.labels_

            representativenesses[:, i] = self.calculate_representativeness(X_pool, cluster_labels, kmeans)
            diversities[:, i] = self.calculate_diversity(X_pool, X_train_valid)

            # Use RandomForestRegressor to compute y_test, y_test_preds
            rf_regressor = RandomForestRegressor(n_estimators=N_TREES, random_state=self.random_state)
            rf_regressor.fit(X_train_valid, y_train_valid)
            test_preds = rf_regressor.predict(X_test)
            y_test_preds[:, i] = np.round(test_preds, 5)

        return variances, representativenesses, diversities, y_test, y_test_preds

    def training(self, X_train, X_pool, X_test, y_train, y_pool, y_test, target_length):
        R2 = np.zeros([1, self.n_epochs+1])
        MSE = np.zeros([1, self.n_epochs+1])
        MAE = np.zeros([1, self.n_epochs+1])
        CA = np.zeros([1, self.n_epochs+1])
        ARRMSE = np.zeros([1, self.n_epochs+1])
        Y_pred = np.zeros([len(X_test), target_length*self.n_epochs])
        instances_pool_rtal = list()
        targets_pool_rtal = list()
        selected_pairs = set()
        percentage_targets_provided = np.zeros(self.n_epochs)

        X_pool = np.array(X_pool)
        y_pool = np.array(y_pool)
        original_indices = list(range(len(X_pool)))

        for i in range(self.n_epochs):
            print("Epoch {}:".format(i+1))
            print("     The training set size: {}".format(len(X_train)))
            print("     The unlabelled pool size: {}".format(len(X_pool)))

            y_train_coll = self.target_collect(y_train, target_length)
            y_train_coll = np.array(y_train_coll).T

            variances, representativenesses, diversities, y_test, y_test_preds = self.variances(X_train, X_pool, X_test, y_train_coll, y_test, target_length)
            Y_pred[:, (i*target_length):((i+1)*target_length)] = y_test_preds
            r2 = np.round(r2_score(np.asarray(y_test), y_test_preds), 4)
            mse = np.round(mean_squared_error(np.asarray(y_test), y_test_preds), 4)
            mae = np.round(mean_absolute_error(np.asarray(y_test), y_test_preds), 4)
            ca = np.round(custom_accuracy(np.asarray(y_test), y_test_preds), 4)
            arrmse = np.round(arrmse_metric(np.asarray(y_test), y_test_preds), 4)
            R2[:, i] = r2
            MSE[:, i] = mse
            MAE[:, i] = mae
            CA[:, i] = ca
            ARRMSE[:, i] = arrmse

            # Cluster the instances in the pool
            cluster_labels = self.cluster_instances(X_pool, self.n_clusters)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)  # Initialize KMeans
            kmeans.fit(X_pool)
            representativenesses = self.calculate_representativeness(X_pool, cluster_labels, kmeans)

            combined_scores = self.calculate_combined_score(variances, representativenesses, diversities)

            # Select top-k instances with highest combined score
            k = self.batch_size 
            top_k_indices = np.argsort(combined_scores)[-k:]

            selected_indices = top_k_indices
            mapped_indices = [original_indices[idx] for idx in selected_indices]

            # Verify that mapped_indices are within the range of X_pool and y_pool
            for idx in mapped_indices:
                if idx >= len(X_pool) or idx >= len(y_pool):
                    print(f"Index {idx} is out of range for X_pool or y_pool")

            # Collect instances and targets to be removed after the epoch
            targets_to_remove = []

            for idx in mapped_indices:
                for target_idx in range(target_length):
                    if (idx, target_idx) in selected_pairs:
                        continue

                    instances_pool_rtal.append(X_pool[idx].flatten())
                    targets_pool_rtal.append((idx, target_idx, y_pool[idx, target_idx]))
                    targets_to_remove.append((idx, target_idx))
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
            total_targets = len(original_indices)  
            provided_targets = len(selected_indices) 
            percentage_provided = (provided_targets / total_targets) * 100
            percentage_targets_provided[i] = percentage_provided
            print(f'Percentage of targets in epoch {i}: {percentage_provided}')

        r2_auc = np.round(auc(self.epochs, R2[0, :-1]), 4)
        mse_auc = np.round(auc(self.epochs, MSE[0, :-1]), 4)
        mae_auc = np.round(auc(self.epochs, MAE[0, :-1]), 4)
        ca_auc = np.round(auc(self.epochs, CA[0, :-1]), 4)
        arrmse_auc = np.round(auc(self.epochs, ARRMSE[0, :-1]), 4)

        R2[:, -1] = r2_auc
        MSE[:, -1] = mse_auc
        MAE[:, -1] = mae_auc
        CA[:, -1] = ca_auc
        ARRMSE[:, -1] = arrmse_auc

        cols = ["Target_{}".format(i+1) for epoch in range(self.n_epochs) for i in range(target_length)]
        Y_pred_df = pd.DataFrame(Y_pred, columns=cols)

        # Create a DataFrame to store the transfer targets
        df = pd.DataFrame(columns=range(target_length))

        # Populate the DataFrame with the transfer targets
        for instance_idx, target_idx, target_value in targets_pool_rtal:
            if instance_idx not in df.index:
                df.loc[instance_idx] = [None] * target_length
            df.at[instance_idx, target_idx] = target_value

        transfer_targets_rtal_df = df

        return R2, MSE, MAE, CA, ARRMSE, Y_pred_df, instances_pool_rtal, transfer_targets_rtal_df, percentage_targets_provided