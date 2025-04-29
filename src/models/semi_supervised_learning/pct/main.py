import pandas as pd
import numpy as np
import sys
sys.path.append("pyclus/")
from utils import load_dataset
from utils import concat_train_unlabeled
from pyclus.clus.models.regression import RegressionTree, RegressionEnsemble
from ssl_pct import SSLPCT
import pathlib
from eval import eval


path_dataset = sys.argv[1]

n_folds = 10

n_trees = 100
weights = [0.5]

dataset_name = path_dataset.split("/")[-1]
e = eval()

path_results = "reports/semi_supervised_learning/" + dataset_name
pathlib.Path("reports").mkdir(parents=True, exist_ok=True)
pathlib.Path("reports/semi_supervised_learning/").mkdir(parents=True, exist_ok=True)

pathlib.Path(path_results).mkdir(parents=True, exist_ok=True)



for fold in range(1, n_folds + 1):
    path_train = path_dataset  + "/train_" + str(fold)
    path_test = path_dataset  + "/test_"  + str(fold)
    path_unlabeled = path_dataset  + "/pool_"  + str(fold)

    train_x, train_y = load_dataset(path_train)
    test_x, test_y = load_dataset(path_test)
    unlabeled_x, unlabeled_y = load_dataset(path_unlabeled)

    model = RegressionEnsemble(ssl = [], n_trees=n_trees, Ensemble_SelectRandomSubspaces = "SQRT", SemiSupervised_PossibleWeights = [0.5])

    train_unlabeled_x, train_unlabeled_y = concat_train_unlabeled(train_x, train_y, unlabeled_x)
#    print(train_unlabeled_x.shape)
#    print(train_unlabeled_y.shape)
    model.fit(train_unlabeled_x, 
            train_unlabeled_y)
    predictions = np.array(model.predict(test_x)["Forest with 100 trees"])
#    print(predictions)
#    print(np.array(predictions).shape)
    performance = e.evaluate_fold(test_y, predictions, fold - 1)
    performance.to_csv(path_results + "/pct_results_fold_" + str(fold - 1) + ".csv", index = False)