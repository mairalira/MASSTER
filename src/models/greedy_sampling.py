""" The subclass for the baseline method. """
from models.active_learning import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, pairwise_distances_argmin_min, auc

class baseline(activelearning):
    def __init__(self, batch_size, n_epochs):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epochs = [i for i in range(n_epochs)]
        self.instances_pool_baseline = list()
        self.targets_pool_baseline = list()
        self.baseline_R2 = np.zeros([self.iterations, self.n_epochs+1])
        self.baseline_MSE = np.zeros([self.iterations, self.n_epochs+1])
        self.baseline_MAE = np.zeros([self.iterations, self.n_epochs+1])

    def distances(self, X_train, X_pool, X_test, y_train, y_test, target_length):
    # here, the instances from the unlabelled pool are chosen based on the distance to the training cluster centroid
        distances = np.zeros(shape=(len(X_pool), 1))
        y_test_preds = np.zeros(shape=(len(X_test), target_length))

        for i in range(target_length):
            # first, fit a model to the training set
            rf_regressor = RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state)  # default value for max features is 1.0 => N features
            rf_regressor.fit(X_train, y_train[i])
            
            # next, predict the target values of the test set
            test_preds = rf_regressor.predict(X_test)
            y_test_preds[:,i] = np.round(test_preds, 5)

        for i, instance in enumerate(X_pool):
            instance = np.array(instance).reshape(1, -1)
            closest, distance = pairwise_distances_argmin_min(instance, X_train)
            distances[i,:] = distance
        return distances.tolist(), y_test, y_test_preds
    
    def max_dist(self, distances):
    # get the max values and indices for the distance 
        sort = sorted(distances)[-self.batch_size:]   # len(distances) is equal to the amount of samples in the pool
        indices = [distances.index(element) for element in sort]
        indices = sorted(indices, reverse=True)
        return sort, indices

    def training(self, X_train, X_pool, X_test, y_train, y_pool, y_test, target_length):
    # the greedy based training loop
        R2 = np.zeros([1, self.n_epochs+1])
        MSE = np.zeros([1, self.n_epochs+1])
        MAE = np.zeros([1, self.n_epochs+1])
        Y_pred = np.zeros([len(X_test), target_length*self.n_epochs])
        instances_pool_baseline = list()
        targets_pool_baseline = list()

        for i in range(self.n_epochs):
            print("Epoch {}:".format(i+1))
            print("     The training set size: {}".format(len(X_train)))
            print("     The unlabelled pool size: {}".format(len(X_pool)))

            y_train_coll = self.target_collect(y_train, target_length)
            distances, y_test, y_test_preds = self.distances(X_train, X_pool, X_test, y_train_coll, y_test, target_length)
            Y_pred[:,(i*target_length):((i+1)*target_length)] = y_test_preds 
            r2, mse, mae = (np.round(r2_score(np.asarray(y_test), y_test_preds), 4)), (np.round(mean_squared_error(np.asarray(y_test), y_test_preds), 4)), (np.round(mean_absolute_error(np.asarray(y_test), y_test_preds), 4))
            R2[:,i] = (r2)
            MSE[:,i] = (mse)
            MAE[:,i] = (mae)

            maxima, indices = self.max_dist(distances) 
            instances_transfer, targets_transfer = self.instances_transfer(X_train, X_pool, y_train, y_pool, indices, "baseline")
            for i in range(len(instances_transfer)):
                instances_pool_baseline.append(instances_transfer[i])
                targets_pool_baseline.append(targets_transfer[i])

        r2_auc, mse_auc, mae_auc = np.round(auc(self.epochs, R2[0,:-1]), 4), np.round(auc(self.epochs, MSE[0,:-1]), 4), np.round(auc(self.epochs, MAE[0,:-1]), 4) 
        R2[:,-1] = (r2_auc)
        MSE[:,-1] = (mse_auc)
        MAE[:,-1] = (mae_auc)

        cols = ["Target_{}".format(i+1) for epoch in range(self.n_epochs) for i in range(target_length)]
        Y_pred_df = pd.DataFrame(Y_pred, columns=cols)

        return R2, MSE, MAE, Y_pred_df, instances_pool_baseline, targets_pool_baseline