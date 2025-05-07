from statistics import variance
import numpy as np
import sys
from pathlib import Path

# Absolute path using Path
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
# Adding path to sys.path
sys.path.append(project_root)

import config
from config import *

sys.path.append(project_root + "/models/semi_supervised_learning/ssl_pct/pyclus/")
from utils import load_dataset
from utils import concat_train_unlabeled
from pyclus.clus.models.regression import RegressionEnsemble
from eval import eval




# Main script
data_dir = DATA_DIR
dataset = DATASET_NAME

k_folds = K_FOLDS
random_state = RANDOM_STATE
n_trees = N_TREES
batch_percentage = BATCH_PERCENTAGE_SSL
e = eval()


if __name__ == "__main__":
    data_dir = str(config.DATA_DIR)
    dataset_name = str(config.DATASET_NAME)
    dataset_path = data_dir  +  "/processed"  +  "/" + dataset_name
    
    results_path = Path(f'reports/semi_supervised_learning/{dataset_name}')
    results_path.mkdir(parents=True, exist_ok=True)
    print("Semi-supervised predictive clustering trees (SSL-PCT)...")

    for fold in range(1, k_folds + 1):
        print("Fold 1")
        path_train = dataset_path  + "/train_" + str(fold)
        path_test = dataset_path  + "/test_"  + str(fold)
        path_unlabeled = dataset_path  + "/pool_"  + str(fold)

        train_x, train_y = load_dataset(path_train)
        test_x, test_y = load_dataset(path_test)
        unlabeled_x, unlabeled_y = load_dataset(path_unlabeled)

        model = RegressionEnsemble(ssl = [], n_trees=n_trees, Ensemble_SelectRandomSubspaces = "SQRT", SemiSupervised_PossibleWeights = [0.5])

        train_unlabeled_x, train_unlabeled_y = concat_train_unlabeled(train_x, train_y, unlabeled_x)
        model.fit(train_unlabeled_x, 
                train_unlabeled_y)
        predictions = np.array(model.predict(test_x)["Forest with " + str(n_trees) + " trees"])

        performance = e.evaluate_fold(test_y, predictions, fold - 1)
        performance.to_csv(str(results_path) + "/sslpct_results_fold_" + str(fold - 1) + ".csv", index = False)
        print('Saved data...')
