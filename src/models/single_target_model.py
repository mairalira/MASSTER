from statistics import variance
import numpy as np
import time
import sys
import pandas as pd
from pathlib import Path

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent.parent
# Adding path to sys.path
sys.path.append(str(project_root))

import config
from config import *

from data.data_processing import *

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from utils.metrics import custom_accuracy, arrmse_metric

class SingleTargetRegressor:
    def __init__(self, random_state, n_trees):
        self.random_state = random_state
        self.n_trees = n_trees

    def unique_evaluate_model(self, models_view1, models_view2, X_test_v1, X_test_v2, y_test_labeled):
        columns = list(X_test_v1.columns)
        print("Target Cotraining Making predictions on test data...")
        predictions_v1 = pd.DataFrame(np.nan, index=X_test_v1.index, columns=y_test_labeled.columns)
        predictions_v2 = pd.DataFrame(np.nan, index=X_test_v2.index, columns=y_test_labeled.columns)
        for i in range(len(models_view1)):
            rf_model_v1 = models_view1[i]
            rf_model_v2 = models_view2[i]
            predictions_v1.iloc[:, i] = rf_model_v1.predict(X_test_v1)
            predictions_v2.iloc[:, i] = rf_model_v2.predict(X_test_v2)

        y_pred_combined = (predictions_v1 + predictions_v2) / 2
        r2 = np.round(r2_score(np.asarray(y_test_labeled), y_pred_combined), 4)
        mse = np.round(mean_squared_error(np.asarray(y_test_labeled), y_pred_combined), 4)
        mae = np.round(mean_absolute_error(np.asarray(y_test_labeled), y_pred_combined), 4)
        ca = np.round(custom_accuracy(y_test_labeled.values, y_pred_combined.values, threshold=CA_THRESHOLD), 4)
        arrmse = np.round(arrmse_metric(np.asarray(y_test_labeled), np.asarray(y_pred_combined)), 4)

        print(f"    Overall: R²={r2:.3f}, MSE={mse:.3f}, MAE={mae:.3f}, CA={ca:.3f}, ARRMSE={arrmse:.3f}")
        return r2, mse, mae, ca, arrmse

    def unique_predict(self, models, X_pool, target_length, columns):
        predictions = pd.DataFrame(
            data=[[None] * target_length for _ in range(len(X_pool))], 
            columns=columns,
            index=X_pool.index
        )
        for i, model in enumerate(models):
            predictions.iloc[:, i] = model.predict(X_pool)
        return predictions

    def unique_fit(self, target_length, y_train_df, X_train):
        model_array = []
        columns = list(y_train_df.columns)
        for i in range(target_length):
            valid_indices = ~y_train_df.iloc[:, i].isna()
            valid_indices = valid_indices.reindex(X_train.index, fill_value=False)
            X_train_valid = X_train[valid_indices]
            y_train_valid = y_train_df.loc[valid_indices, columns[i]]
            model = RandomForestRegressor(random_state=self.random_state, n_estimators=self.n_trees)
            model.fit(X_train_valid, y_train_valid)
            model_array.append(model)
        return model_array
