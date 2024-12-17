import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error
import time

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
            unique_values = np.unique(X[:, feature])
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

# Gerar dados simulados
def generate_data():
    np.random.seed(123)
    n, m, t = 1000, 5, 3  # n: samples, m: features, t: targets
    X = np.random.rand(n, m)

    # Simulate multi-target regression outputs
    Y = np.zeros((n, t))
    Y[:, 0] = X[:, 0] * 2.0 + X[:, 1] * 3.0  # Target 1
    Y[:, 1] = X[:, 2] * 4.0 + X[:, 3] * 1.5  # Target 2
    Y[:, 2] = X[:, 1] * 5.0 - X[:, 4] * 2.0  # Target 3

    return train_test_split(X, Y, test_size=0.25, random_state=42)

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = generate_data()

    # Simulate missing data in training set
    missing_mask = np.random.rand(*Y_train.shape) < 0.2
    Y_train_missing = Y_train.copy()
    Y_train_missing[missing_mask] = np.nan

    # ------------------ Modelo Original ------------------
    start_time = time.time()
    not_missing_mask = ~np.isnan(Y_train_missing).any(axis=1)
    X_train_not_missing = X_train[not_missing_mask]
    Y_train_not_missing = Y_train_missing[not_missing_mask]

    pct_original = PredictiveClusteringTree(max_depth=5, min_samples_split=10)
    pct_original.fit(X_train_not_missing, Y_train_not_missing)
    Y_pred_original = pct_original.predict(X_test)
    end_time = time.time()

    print("Desempenho Original:")
    for i in range(Y_test.shape[1]):
        r2 = r2_score(Y_test[:, i], Y_pred_original[:, i])
        mae = mean_absolute_error(Y_test[:, i], Y_pred_original[:, i])
        ca = custom_accuracy(Y_test[:, i], Y_pred_original[:, i])
        print(f"Target {i+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
    print(f"Tempo de execução: {end_time - start_time:.2f} segundos\n")

    # ------------------ Semissupervisionado ------------------
    start_time = time.time()

    # Preencher valores faltantes com previsões do modelo original
    Y_train_filled = Y_train_missing.copy()
    for i in range(Y_train_missing.shape[0]):
        if np.isnan(Y_train_missing[i]).any():
            Y_train_filled[i] = pct_original.predict(X_train[i].reshape(1, -1))

    # Treinar novo modelo semissupervisionado
    pct_semi = PredictiveClusteringTree(max_depth=5, min_samples_split=10)
    pct_semi.fit(X_train, Y_train_filled)
    Y_pred_semi = pct_semi.predict(X_test)
    end_time = time.time()

    print("Desempenho Semissupervisionado:")
    for i in range(Y_test.shape[1]):
        r2 = r2_score(Y_test[:, i], Y_pred_semi[:, i])
        mae = mean_absolute_error(Y_test[:, i], Y_pred_semi[:, i])
        ca = custom_accuracy(Y_test[:, i], Y_pred_semi[:, i])
        print(f"Target {i+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
    print(f"Tempo de execução: {end_time - start_time:.2f} segundos")
