import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

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
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.isnan(Y).any():
            leaf = PCTNode(depth=depth)
            leaf.is_leaf = True
            leaf.prediction = np.nanmean(Y, axis=0)
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
                    var_left = np.nanvar(Y_left, axis=0).sum()
                    var_right = np.nanvar(Y_right, axis=0).sum()
                    combined_variance = var_left + var_right

                    if combined_variance < best_variance:
                        best_variance = combined_variance
                        best_split = (feature, value, left_mask, right_mask)

        if best_split is None:
            # If no valid split is found, create a leaf node
            leaf = PCTNode(depth=depth)
            leaf.is_leaf = True
            leaf.prediction = np.nanmean(Y, axis=0)
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

# Example usage
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

    pct = PredictiveClusteringTree(max_depth=5, min_samples_split=10)
    pct.fit(X_train, Y_train)

    Y_pred = pct.predict(X_test)

    # Evaluate performance
    scores = []
    for i in range(Y_test.shape[1]):
        score = explained_variance_score(Y_test[:, i], Y_pred[:, i])
        scores.append(score)
        print(f"Explained variance for Target {i+1}: {score:.3f}")

    print(f"Average explained variance: {np.mean(scores):.3f}")
