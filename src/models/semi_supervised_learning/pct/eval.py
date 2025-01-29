from sklearn.metrics import pairwise_distances, r2_score, mean_squared_error, mean_absolute_error, auc
import numpy as np
import pandas as pd
class eval:

    columns = ["Fold_Index", "Iterations", "R2", "MSE", "MAE", "CA", "ARRMSE", "AddedPairs"]
    
    CA_THRESHOLD = 0.1

    n_repetitions = 5
    def __init__(self):
        self.performance_matrix = []
    
        

    def evaluate_fold(self, 
                    y_test, 
                    y_test_preds,
                    fold_nb,
                    ):
        df = pd.DataFrame({
        'Fixed Value': [fold_nb] * self.n_repetitions,
        'Range': range(self.n_repetitions)
        })
        performance = self.evaluate(y_test,
                      y_test_preds)
        performance = pd.DataFrame([performance] * self.n_repetitions).reset_index(drop=True)
        result_df = pd.concat([df, performance], axis=1)
        result_df.columns = self.columns
        return result_df
    def evaluate(self, 
                    y_test, 
                    y_test_preds):

        r2 = np.round(r2_score(np.asarray(y_test), y_test_preds), 4)
        mse = np.round(mean_squared_error(np.asarray(y_test), y_test_preds), 4)
        mae = np.round(mean_absolute_error(np.asarray(y_test), y_test_preds), 4)
        ca = np.round(self.custom_accuracy(np.asarray(y_test), y_test_preds), 4)
        arrmse = np.round(self.arrmse_metric(np.asarray(y_test), y_test_preds), 4)
        return (r2, mse, mae, ca , arrmse, 0) # 0 = added pairs

    def custom_accuracy(self,
                        y_true, 
                        y_pred, 
                        threshold= CA_THRESHOLD):
        """Custom accuracy: percentage of predictions within a certain threshold adapted to multi-target."""
        M = y_true.shape[1]  # Number of targets
        custom_accuracy_value = 0

        for t in range(M):
            y_true_t = y_true[:, t]
            y_pred_t = y_pred[:, t]
            custom_accuracy_t = np.mean(np.abs(y_true_t - y_pred_t) <= threshold)
            custom_accuracy_value += custom_accuracy_t

        custom_accuracy_value /= M
        return custom_accuracy_value

    def arrmse_metric(self,
                      y_true, 
                      y_pred):
        """Calculate the Average Relative Root Mean Squared Error (aRRMSE)."""
        M = y_true.shape[1]  # Number of targets
        N = y_true.shape[0]  # Number of testing instances
        arrmse_value = 0

        for t in range(M):
            y_true_t = y_true[:, t]
            y_pred_t = y_pred[:, t]
            y_mean_t = np.mean(y_true_t)

            numerator = np.sum((y_true_t - y_pred_t) ** 2)
            denominator = np.sum((y_true_t - y_mean_t) ** 2)

            if denominator == 0:
                rrmse_t = 0
            else:
                rrmse_t = np.sqrt(numerator / denominator)

            arrmse_value += rrmse_t

        arrmse_value /= M
        return arrmse_value