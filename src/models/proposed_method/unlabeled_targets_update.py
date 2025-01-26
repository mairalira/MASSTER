import numpy as np
import pandas as pd

def update_y_pool_nan(y_pool_nan, all_pred_selected_pairs):
    """fuction to update y_pool_nan after active learning step"""
    columns = list(y_pool_nan.columns)
    for (idx, target), value in all_pred_selected_pairs.items():
        y_pool_nan.loc[idx,columns[target]] = value
    return y_pool_nan

def update_y_pool(y_pool, all_pred_selected_pairs):
    """fuction to update y_pool after semi-supervised learning step"""
    columns = list(y_pool.columns)
    for (idx, target) in all_pred_selected_pairs.keys():
        y_pool.loc[idx, columns[target]] = np.nan
    return y_pool