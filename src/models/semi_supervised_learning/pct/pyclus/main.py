import numpy as np
from clus.helpers import Utilities
from clus.models.classification import ClassificationTree, ClassificationEnsemble
from clus.models.regression import RegressionTree, RegressionEnsemble
from clus.models.mlc import MLCTree, MLCEnsemble
from clus.models.hmlc import HMLCTree, HMLCEnsemble, HMLCRelief

from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.model_selection import train_test_split
import logging


logging.basicConfig(level=logging.INFO)


def convert_to_sparse(xs):
    n = len(xs)
    m = len(xs[0])
    rows = []
    cols = []
    vals = []
    for i in range(n):
        for j in range(m):
            rows.append(i)
            cols.append(j)
            vals.append(xs[i][j])
    return rows, cols, vals


def prepare_mlc_data():
    np.random.seed(123)
    n, m = 1000, 5
    xs = np.random.rand(n, m)
    temp = xs[:, 0] * 2.0 + xs[:, 1] * 3.0
    theta1 = np.min(temp) * 0.66 + 0.33 * np.max(temp)
    theta2 = np.min(temp) * 0.33 + 0.66 * np.max(temp)
    y = np.zeros((n, 2), dtype=str)
    y[temp <= theta1, 0] = "0"
    y[temp <= theta1, 1] = "1"
    y[(theta1 < temp) & (temp <= theta2), 0] = "1"
    y[(theta1 < temp) & (temp <= theta2), 1] = "0"
    y[theta2 < temp, 0] = "0"
    y[theta2 < temp, 1] = "1"
    return train_test_split(xs, y, test_size=0.25)


def prepare_hmlc_data(tree=True):
    np.random.seed(123)
    n, m = 1000, 5
    xs = np.random.rand(n, m)
    temp1 = xs[:, 0] * 2.0 + xs[:, 1] * 3.0
    temp2 = xs[:, 2] * 2.0 + xs[:, 3] * 3.0
    theta1 = np.min(temp1) * 0.66 + 0.33 * np.max(temp1)
    theta2 = np.min(temp1) * 0.33 + 0.66 * np.max(temp1)
    theta3 = np.min(temp2) * 0.66 + 0.33 * np.max(temp2)
    theta4 = np.min(temp2) * 0.33 + 0.66 * np.max(temp2)
    y = np.array(["1@1/1@1/1/2" for _ in range(n)])
    if tree:
        #            1
        #     1              2
        #  1     2
        hierarchy = ["1", "1/1", "1/2", "1/1/1", "1/1/2"]
        shape = "Tree"
        y[(temp1 <= theta1) & (temp2 <= theta3)] = "1"
        y[(temp1 <= theta1) & (theta3 < temp2) & (temp2 <= theta4)] = "1@1/1"
        y[(temp1 <= theta1) & (theta4 < temp2)] = "1@1/2"

        y[(temp2 <= theta3) & (theta1 < temp1) & (temp1 <= theta2)] = "1@1/1@1/2"
        y[(temp2 <= theta3) & (theta2 < temp1)] = "1@1/1@1/1/1"
    else:
        # a -> b -> c -> d
        # b -> e -> d
        y[:] = "a@b@c"
        hierarchy = ["a/b", "b/c", "c/d", "b/e", "e/d"]
        shape = "DAG"
        y[(temp1 <= theta1) & (temp2 <= theta3)] = "a@b"
        y[(temp1 <= theta1) & (theta3 < temp2) & (temp2 <= theta4)] = "a@b@e"
        y[(temp1 <= theta1) & (theta4 < temp2)] = "a"

        y[(temp2 <= theta3) & (theta1 < temp1) & (temp1 <= theta2)] = "a@b@e@d"
        y[(temp2 <= theta3) & (theta2 < temp1)] = "a@b@c@d@e"
    return hierarchy, shape, train_test_split(xs, y, test_size=0.25)


def prepare_multi_target_regression_data():
    np.random.seed(123)
    n, m = 1000, 5
    xs = np.random.rand(n, m)
    ys = np.zeros((n, 2))
    ys[:, 0] = xs[:, 0] * 2.0 + xs[:, 1] * 3.0
    ys[:, 1] = xs[:, 0] * 1.0 + xs[:, 1] * 2.0
    return train_test_split(xs, ys, test_size=0.25)


def prepare_single_target_classification_data():
    np.random.seed(123)
    n, m = 1000, 5
    xs = np.random.rand(n, m)
    temp = xs[:, 0] * 2.0 + xs[:, 1] * 3.0
    theta1 = np.min(temp) * 0.66 + 0.33 * np.max(temp)
    theta2 = np.min(temp) * 0.33 + 0.66 * np.max(temp)
    y = np.zeros((n, 2), dtype=str)
    y[temp <= theta1, 0] = "a"
    y[temp <= theta1, 1] = "b"
    y[(theta1 < temp) & (temp <= theta2), 0] = "b"
    y[(theta1 < temp) & (temp <= theta2), 1] = "c"
    y[theta2 < temp, 0] = "c"
    y[theta2 < temp, 1] = "a"
    return train_test_split(xs, y, test_size=0.25)


def prepare_hard_regression_data():
    np.random.seed(123)
    n, m = 1000, 5
    xs = np.random.rand(n, m).tolist()
    # first two nominal
    values = ["ab", "cd"]
    for i, y_vals in enumerate(values):
        for j in range(n):
            val = y_vals[xs[j][i] > 0.5]
            xs[j][i] = val
    y = [int(x[0] == "a") + sum(x[2:]) for x in xs]
    p_missing = 0.05
    for i in range(n):
        for j in range(m):
            if np.random.rand() < p_missing:
                xs[i][j] = "?"
    return train_test_split(xs, y, test_size=0.25)


def test_classification_single_tree(predict=False):
    xs_train, xs_test, y_train, y_test = prepare_single_target_classification_data()
    model = ClassificationTree(verbose=0, is_multi_target=False)  #
    model.fit(xs_train, y_train)
    if predict:
        i = 1
        y_hat_train_all = model.predict(xs_train)["Original"]
        y_hat_test_all = model.predict(xs_test)["Original"]
        y_hat_train = [y[i][0] for y in y_hat_train_all]
        y_hat_test = [y[i][0] for y in y_hat_test_all]
        score_train = accuracy_score([y[i] for y in y_train], y_hat_train)
        score_test = accuracy_score([y[i] for y in y_test], y_hat_test)
        print(f"accuracy of the model\ntrain: {score_train:.3e};test : {score_test:.3e}")


def test_classification_forest(predict=False):
    xs_train, xs_test, y_train, y_test = prepare_single_target_classification_data()
    model = ClassificationEnsemble(
        verbose=0, n_trees=[2, 4], forest=[],
        # Ensemble_Iterations=[2, 4]
    )
    model.fit(xs_train, y_train)
    y_hat_train = model.predict(xs_train)["Forest with 4 trees"]
    y_hat_test = model.predict(xs_test)["Forest with 4 trees"]
    if predict:
        score_train = accuracy_score(y_train, y_hat_train)
        score_test = accuracy_score(y_test, y_hat_test)
        print(f"accuracy of the model\ntrain: {score_train:.3e};test : {score_test:.3e}")


def test_regression_single_tree(predict=False):
    xs_train, xs_test, y_train, y_test = prepare_multi_target_regression_data()
    model = RegressionTree(verbose=0, is_multi_target=True)  #
    model.fit(xs_train, y_train)
    if predict:
        y_hat_train_all = model.predict(xs_train)
        y_hat_test_all = model.predict(xs_test)
        y_hat_train = y_hat_train_all["Original"]
        y_hat_test = y_hat_test_all["Original"]
        score_train = explained_variance_score(y_train, y_hat_train)
        score_test = explained_variance_score(y_test, y_hat_test)
        print(f"explained variance of the model\ntrain: {score_train:.3e};test : {score_test:.3e}")


def test_regression_forest(predict=False):
    xs_train, xs_test, y_train, y_test = prepare_multi_target_regression_data()
    model = RegressionEnsemble(
        verbose=0, forest=[], ssl = [], n_trees=4
    )

    model.fit(xs_train, y_train)
    print(model.feature_importances_)
    if predict:
        y_hat_train = model.predict(xs_train)["Forest with 4 trees"]
        y_hat_test = model.predict(xs_test)["Forest with 4 trees"]
        score_train = explained_variance_score(y_train, y_hat_train)
        score_test = explained_variance_score(y_test, y_hat_test)
        print(f"explained variance of the model\ntrain: {score_train:.3e};test : {score_test:.3e}")


def test_regression_sparse(predict=False):
    xs_train, xs_test, y_train, y_test = prepare_multi_target_regression_data()
    # to sparse
    xs_train = convert_to_sparse(xs_train)
    xs_test = convert_to_sparse(xs_test)
    model = RegressionEnsemble(
        verbose=0, forest=[], n_trees=4
    )
    model.fit(xs_train, y_train)
    print(model.feature_importances_)
    if predict:
        y_hat_train = model.predict(xs_train)["Forest with 4 trees"]
        y_hat_test = model.predict(xs_test)["Forest with 4 trees"]
        score_train = explained_variance_score(y_train, y_hat_train)
        score_test = explained_variance_score(y_test, y_hat_test)
        print(f"explained variance of the model\ntrain: {score_train:.3e};test : {score_test:.3e}")


def test_hard_regression(sparse=False):
    xs_train, xs_test, y_train, y_test = prepare_hard_regression_data()
    if sparse:
        # to sparse
        xs_train = convert_to_sparse(xs_train)
        xs_test = convert_to_sparse(xs_test)
    model = RegressionEnsemble(
        verbose=0, forest=[], n_trees=4
    )
    model.fit(xs_train, y_train)
    y_hat_train = model.predict(xs_train)["Forest with 4 trees"]
    y_hat_test = model.predict(xs_test)["Forest with 4 trees"]
    score_train = explained_variance_score(y_train, y_hat_train)
    score_test = explained_variance_score(y_test, y_hat_test)
    print(f"explained variance of the model\ntrain: {score_train:.3e};test : {score_test:.3e}")


def test_mlc_forest(sparse=False):
    xs_train, xs_test, y_train, y_test = prepare_mlc_data()
    if sparse:
        # to sparse
        xs_train = convert_to_sparse(xs_train)
        xs_test = convert_to_sparse(xs_test)
    model = MLCEnsemble(
        verbose=0, forest=[], n_trees=4
    )
    model.fit(xs_train, y_train)
    i = 1
    y_hat_train_all = model.predict(xs_train)["Forest with 4 trees"]
    y_hat_test_all = model.predict(xs_test)["Forest with 4 trees"]
    y_hat_train = [y[i][0] for y in y_hat_train_all]
    y_hat_test = [y[i][0] for y in y_hat_test_all]
    score_train = accuracy_score([y[i] for y in y_train], y_hat_train)
    score_test = accuracy_score([y[i] for y in y_test], y_hat_test)
    print(f"accuracy of the model\ntrain: {score_train:.3e};test : {score_test:.3e}")


def test_hmlc_forest(tree=True):
    hierarchy, shape, (xs_train, xs_test, y_train, y_test) = prepare_hmlc_data(tree)
    model = HMLCEnsemble(
        verbose=0, forest=[], n_trees=4
    )
    model.fit(xs_train, y_train, hierarchy_shape=shape, hierarchy=hierarchy)
    y_hat_train_all = model.predict(xs_train)
    y_hat_test_all = model.predict(xs_test)
    _ = 21


def test_hmlc_relief(tree=True):
    hierarchy, shape, (xs_train, xs_test, y_train, y_test) = prepare_hmlc_data(tree)
    model = HMLCRelief(
        verbose=0, relief=[], iterations=[0.25, 0.5, 0.75], is_multi_target=False
    )
    model.fit(xs_train, y_train, hierarchy_shape=shape, hierarchy=hierarchy)
    print(model.feature_importances_)


if __name__ == "__main__":
    test_relief = False
    test_r = True
    test_c = True
    test_hard = True
    test_mlc = True
    test_hmlc = True
    if test_relief:
        test_hmlc_relief(True)
        test_hmlc_relief(False)
        exit(12)
    if test_r:
#        test_regression_single_tree(True)
        test_regression_forest(True)
 #       test_regression_sparse(True)
        exit(12)
    if test_c:
        test_classification_single_tree(True)
        test_classification_forest(False)
    if test_hard:
        test_hard_regression(True)
    if test_mlc:
        test_mlc_forest()
    if test_hmlc:
        test_hmlc_forest(True)
        test_hmlc_forest(False)
