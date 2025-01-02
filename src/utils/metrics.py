import numpy as np
from models.active_learning import *
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, pairwise_distances_argmin_min, auc

def custom_accuracy(y_true, y_pred, threshold= CA_THRESHOLD):
    """Custom accuracy: percentage of predictions within a certain threshold."""
    M = y_true.shape[1]  # Number of targets
    custom_accuracy_value = 0

    for t in range(M):
        y_true_t = y_true[:, t]
        y_pred_t = y_pred[:, t]
        custom_accuracy_t = np.mean(np.abs(y_true_t - y_pred_t) <= threshold)
        custom_accuracy_value += custom_accuracy_t

    custom_accuracy_value /= M
    return custom_accuracy_value

def arrmse_metric(y_true, y_pred):
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

