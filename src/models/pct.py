import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error
import time
import pandas as pd

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
        # Converta X e Y para arrays NumPy, se forem DataFrames
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(Y, pd.DataFrame):
            Y = Y.to_numpy()

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
            unique_values = np.unique(X[:, feature])  # X é agora um array NumPy
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

if __name__ == "__main__":
    df_train_not_missing = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\train_1')
    df_train_missing = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\pool_1')
    df_test = pd.read_csv(r'C:\Users\danin\Documents\lu\mestrado\artigo_ricardo\Active-and-Self-Learning-for-Multi-target-Regression\src\processed\atp7d\test_1')

    X_test=df_test.drop(columns = ['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000'])
    Y_test = df_test[['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000']]

    X_train_missing = df_train_missing.drop(columns = ['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000'])
    Y_train_missing = df_train_missing[['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000']]
    Y_train_missing.loc[:, :] = np.nan  # Substitui todos os valores do DataFrame por np.nan
    X_train_not_missing = df_train_not_missing.drop(columns = ['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000'])
    Y_train_not_missing = df_train_not_missing[['target_LBL+ALLminpA+bt7d_000','target_LBL+ALLminp0+bt7d_000','target_LBL+aDLminpA+bt7d_000','target_LBL+aCOminpA+bt7d_000','target_LBL+aFLminpA+bt7d_000','target_LBL+aUAminpA+bt7d_000']]
    
    
    # ------------------ Modelo Original ------------------
    start_time = time.time()
    #not_missing_mask = ~np.isnan(Y_train_missing).any(axis=1)


    pct_original = PredictiveClusteringTree(max_depth=5, min_samples_split=10)
    pct_original.fit(X_train_not_missing.to_numpy(), Y_train_not_missing.to_numpy())
    Y_pred_original = pct_original.predict(X_test.to_numpy())
    end_time = time.time()

    print("Desempenho Original:")
    for i in range(Y_test.shape[1]):
        r2 = r2_score(Y_test.to_numpy()[:, i], Y_pred_original[:, i])
        mae = mean_absolute_error(Y_test.to_numpy()[:, i], Y_pred_original[:, i])
        ca = custom_accuracy(Y_test.to_numpy()[:, i], Y_pred_original[:, i])
        print(f"Target {i+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
    print(f"Tempo de execução: {end_time - start_time:.2f} segundos\n")

    # ------------------ Semissupervisionado ------------------
    start_time = time.time()
    w = 0.5  # Adjust this parameter to control influence (0.0 - only labeled data, 1.0 - equal influence)

    # Preencher valores faltantes com previsões do modelo original
    Y_train_filled = Y_train_missing.copy()

    for i in range(Y_train_filled.shape[0]):
        # Verifica se a linha contém algum NaN
        if Y_train_filled.iloc[i, :].isna().any():
            # Obtém os valores da linha correspondente de X_train_missing
            x_missing = X_train_missing.iloc[i, :].values.reshape(1, -1)
            
            # Prediz os valores faltantes usando o modelo original
            y_pred = pct_original.predict(x_missing)
            
            # Preenche os valores previstos em Y_train_filled
            Y_train_filled.iloc[i, :] = y_pred

    # Combine os datasets
    X_train_combined = pd.concat([X_train_missing, X_train_not_missing], axis=0)
    X_train_combined.reset_index(drop=True, inplace=True)

    Y_train_combined = pd.concat([Y_train_filled, Y_train_not_missing], axis=0)
    Y_train_combined.reset_index(drop=True, inplace=True)

    # Criar uma máscara para instâncias rotuladas (não faltantes)
    labeled_mask = ~Y_train_missing.isna().any(axis=1)

    # Criar o vetor de pesos
    weights_filled = np.full(Y_train_filled.shape[0], w)  # Pesos para instâncias preenchidas
    weights_not_missing = np.ones(np.sum(labeled_mask))  # Pesos para instâncias rotuladas originais
    weights = np.hstack((weights_not_missing, weights_filled))

    # Treinar novo modelo semissupervisionado
    pct_semi_weighted = PredictiveClusteringTree(max_depth=5, min_samples_split=10)
    pct_semi_weighted.fit(X_train_combined.to_numpy(), Y_train_combined.to_numpy())

    # Fazer previsões no conjunto de teste
    Y_pred_semi_weighted = pct_semi_weighted.predict(X_test.to_numpy())

    # Medir o tempo final
    end_time = time.time()

    print("Desempenho Semissupervisionado:")
    for i in range(Y_test.shape[1]):
        r2 = r2_score(Y_test.to_numpy()[:, i], Y_pred_original[:, i])
        mae = mean_absolute_error(Y_test.to_numpy()[:, i], Y_pred_original[:, i])
        ca = custom_accuracy(Y_test.to_numpy()[:, i], Y_pred_original[:, i])
        print(f"Target {i+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
    print(f"Tempo de execução: {end_time - start_time:.2f} segundos\n")
