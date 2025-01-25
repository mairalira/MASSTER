from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import time
import pandas as pd
import numpy as np
import config
from pathlib import Path

data_dir = config.DATA_DIR
dataset_name = config.DATASET_NAME

def data_read(dataset):
    # split the csv file into input and target values
    folder_dir = data_dir / 'processed' / f'{dataset_name}'
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

# Helper function to read data
def read_data(iteration):
    X_train, y_train, _, _ = data_read(f'train_{iteration}')
    X_pool, y_pool, n_pool, target_length = data_read(f'pool_{iteration}')
    X_rest, y_rest, _, _ = data_read(f'train+pool_{iteration}')
    X_test, y_test, _, _ = data_read(f'test_{iteration}')
    return X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length

# Custom accuracy function
def custom_accuracy(y_true_list, y_pred_list, threshold=0.1):
    # Convert to NumPy arrays
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    """Custom accuracy: percentage of predictions within a certain threshold."""
    return np.mean(np.abs(y_true - y_pred) <= threshold)


number_of_pools = 10
for i in range(number_of_pools):
    print(f"Training model in pool {i}...")
    X_train_not_missing, Y_train_not_missing, X_unlabeled, Y_train_missing, X_rest, y_rest, X_test_labeled, y_test_labeled, target_length = read_data(i+1)
    # X_train, y_train, X_pool, y_pool, X_rest, y_rest, X_test, y_test, target_length
    print("Original Performance:")
    start_time = time.time()

    # Initialize a regression model using the complete labeled data
    model_original = MultiOutputRegressor(RandomForestRegressor(random_state=42))

    model_original.fit(X_train_not_missing, Y_train_not_missing)

    # Make predictions on the test data
    Y_pred_original = model_original.predict(X_test_labeled)
    execution_time = time.time() - start_time
    for j in range(len(y_test_labeled[0])):
        # Access column j of each list manually
        r2 = r2_score([row[j] for row in y_test_labeled], [row[j] for row in Y_pred_original])
        mae = mean_absolute_error([row[j] for row in y_test_labeled], [row[j] for row in Y_pred_original])
        ca = custom_accuracy([row[j] for row in y_test_labeled], [row[j] for row in Y_pred_original])
        print(f"Target {i+1}: R2={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")
    print(f"Execution time (with original data): {time.time() - start_time:.2f} seconds\n")

    print("Semi-supervised Performance")
    
    # Models for multi-target regression
    print("Initializing regression models...")

    model_view1 = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    model_view2 = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    # Split features into two views
    print("Splitting features into two views...")
    # Splitting features correctly

    # Convert the list to a NumPy array
    X_train_not_missing = np.array(X_train_not_missing)
    X_unlabeled = np.array(X_unlabeled)
    X_test_labeled = np.array(X_test_labeled)

    X_train_labeled_v1 = X_train_not_missing[:, :int(X_train_not_missing.shape[1]/2)]  # First half of the features
    X_train_labeled_v2 = X_train_not_missing[:, int(X_train_not_missing.shape[1]/2):]  # Second half of the features
    # Now, apply the mask to filter X_train_not_missing and X_view2
    X_train_unlabeled_v1 = X_unlabeled[:, :int(X_train_not_missing.shape[1]/2)]
    X_train_unlabeled_v2 = X_unlabeled[:, int(X_train_not_missing.shape[1]/2):]

    X_test_labeled_v1 = X_test_labeled[:, :int(X_test_labeled.shape[1] / 2)]  # First half of the features
    X_test_labeled_v2 = X_test_labeled[:, int(X_test_labeled.shape[1] / 2):]  # Second half of the features
        
    y_labeled = Y_train_not_missing
    max_iter = 10
    for j in range(max_iter):
        print(f"Training model in epoch {j}...")  

        # Training with co-training
        start_time = time.time()
        print("Starting training with co-training...")
        
        print("Training model 1 with view 1...")
        model_view1.fit(X_train_labeled_v1, y_labeled)
        print("Training model 2 with view 2...")
        model_view2.fit(X_train_labeled_v2, y_labeled)

        # Make predictions for unlabeled data
        print("Making predictions for unlabeled data...")

        preds1 = model_view1.predict(X_train_unlabeled_v1) if len(X_train_unlabeled_v1) > 0 else np.array([])
        preds2 = model_view2.predict(X_train_unlabeled_v2) if len(X_train_unlabeled_v2) > 0 else np.array([])

        if len(preds1) == 0 or len(preds2) == 0:
            print("No more unlabeled examples.")
            break

        threshold = 0.1

        # Determine confidence of predictions based on variance
        print("Calculating prediction confidence...")
        confident_mask1 = np.std(preds1, axis=1) <= threshold
        confident_mask2 = np.std(preds2, axis=1) <= threshold
        # Add confident examples to the labeled set
        if confident_mask1.any() and confident_mask2.any():
            print(f"Adding {confident_mask1.sum()} confident examples from view 1.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_train_unlabeled_v1[confident_mask1]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_train_unlabeled_v2[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds2[confident_mask2]])

        if not confident_mask1.any() and not confident_mask2.any():
            print("No confident predictions found.")
            break

        # Add confident examples to the labeled set
        if confident_mask1.any():
            print(f"Adding {confident_mask1.sum()} confident examples from view 1.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_train_unlabeled_v1[confident_mask1]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_train_unlabeled_v2[confident_mask1]])
            y_labeled = np.vstack([y_labeled, preds1[confident_mask1]])

        if confident_mask2.any():
            print(f"Adding {confident_mask2.sum()} confident examples from view 2.")
            X_train_labeled_v1 = np.vstack([X_train_labeled_v1, X_train_unlabeled_v1[confident_mask2]])
            X_train_labeled_v2 = np.vstack([X_train_labeled_v2, X_train_unlabeled_v2[confident_mask2]])
            y_labeled = np.vstack([y_labeled, preds2[confident_mask2]])

        # Remove confident examples from the unlabeled set
        print("Removing confident examples from the unlabeled set...")
        # check if it is being updated
        X_train_unlabeled_v1 = X_train_unlabeled_v1[~confident_mask1]
        X_train_unlabeled_v2 = X_train_unlabeled_v2[~confident_mask2]

        print(f"{confident_mask1.sum() + confident_mask2.sum()} examples added in this iteration.")    
        
        # Evaluation on test data
    print("Making predictions on test data...")
    y_pred_v1 = model_view1.predict(X_test_labeled_v1)
    y_pred_v2 = model_view2.predict(X_test_labeled_v2)
    execution_time = time.time() - start_time
    # Combine predictions by averaging
    y_pred_combined = (y_pred_v1 + y_pred_v2) / 2

    # Evaluation of semi-supervised performance
    for j in range(len(y_test_labeled[0])):  # Iterating through each target
        r2 = r2_score([row[j] for row in y_test_labeled], [row[j] for row in y_pred_combined])
        mae = mean_absolute_error([row[j] for row in y_test_labeled], [row[j] for row in y_pred_combined])
        ca = custom_accuracy([row[j] for row in y_test_labeled], [row[j] for row in y_pred_combined])
        print(f"Target {i+1}: R²={r2:.3f}, MAE={mae:.3f}, CA={ca:.3f}")

    print(f"Execution time: {execution_time:.2f} seconds\n")
