""" The subclass for QBC-RF. """
from models.active_learning import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statistics import variance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, auc
from utils.metrics import *

class instancebased(activelearning):
    def __init__(self, batch_size, n_epochs):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epochs = [i for i in range(n_epochs)]
        self.instances_pool_qbc = list()
        self.targets_pool_qbc = list()
        self.instance_R2 = np.zeros([self.iterations, self.n_epochs+1])
        self.instance_MSE = np.zeros([self.iterations, self.n_epochs+1])
        self.instance_MAE = np.zeros([self.iterations, self.n_epochs+1])
        self.instance_CA = np.zeros([self.iterations, self.n_epochs+1])
        self.instance_aRRMSE = np.zeros([self.iterations, self.n_epochs+1])

    def variances(self, X_train, X_pool, X_test, y_train, y_test, target_length):
        # calculate the variances and the test set predictions
        variances = np.zeros(shape=(len(X_pool), target_length))
        y_test_preds = np.zeros(shape=(len(X_test), target_length))

        for i in range(target_length):
            var = list()

            # first, fit a model to the training set
            rf_regressor = RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state)  # default value for max features is 1.0 => N features
            rf_regressor.fit(X_train, y_train[i])
            
            # next, predict the target values of the test set
            test_preds = rf_regressor.predict(X_test)
            y_test_preds[:,i] = np.round(test_preds, 5)

            # at last, calculate the variance of the predicitions on the unlabelled pool dataset
            pool_preds = np.zeros(shape=(len(X_pool), len(rf_regressor.estimators_)))
            for j, model in enumerate(rf_regressor.estimators_):
                pool_preds[:,j] = model.predict(X_pool) 
            
            for k, model_preds in enumerate(pool_preds):
                var_value = variance(model_preds)
                var.append(var_value)
            variances[:,i] = var
        
        return variances, y_test, y_test_preds
    
    def max_var(self, variances):
        # get the max values and indices for the variance 
        sort = sorted(variances)[-self.batch_size:]   # len(variances) is equal to the amount of samples in the pool
        indices = [variances.index(element) for element in sort]
        indices = sorted(indices, reverse=True)
        return sort, indices
    
    def training(self, X_train, X_pool, X_test, y_train, y_pool, y_test, target_length):
        # the instance based active learning training loop
        R2 = np.zeros([1, self.n_epochs+1])
        MSE = np.zeros([1, self.n_epochs+1])
        MAE = np.zeros([1, self.n_epochs+1])
        CA = np.zeros([1, self.n_epochs+1])
        ARRMSE = np.zeros([1, self.n_epochs+1])
        Y_pred = np.zeros([len(X_test), target_length*self.n_epochs])
        instances_pool_qbc = list()
        targets_pool_qbc = list()

        for i in range(self.n_epochs):
            print("Epoch {}:".format(i+1))
            print("     The training set size: {}".format(len(X_train)))
            print("     The unlabelled pool size: {}".format(len(X_pool)))

            y_train_coll = self.target_collect(y_train, target_length)
            variances, y_test, y_test_preds = self.variances(X_train, X_pool, X_test, y_train_coll, y_test, target_length)
            Y_pred[:,(i*target_length):((i+1)*target_length)] = y_test_preds
            r2 = (np.round(r2_score(np.asarray(y_test), y_test_preds), 4))
            mse = (np.round(mean_squared_error(np.asarray(y_test), y_test_preds), 4))
            mae = (np.round(mean_absolute_error(np.asarray(y_test), y_test_preds), 4))
            ca = (np.round(custom_accuracy(np.asarray(y_test), y_test_preds), 4))
            arrmse = (np.round(arrmse_metric(np.asarray(y_test), y_test_preds), 4))

            R2[:,i] = (r2)
            MSE[:,i] = (mse)
            MAE[:,i] = (mae)
            CA[:,i] = (ca)
            ARRMSE[:,i] = (arrmse)

            # sum the variances for the instance based approach
            summed_variances = [sum(values) for values in variances]
            
            maxima, indices = self.max_var(summed_variances) 
            instances_transfer, targets_transfer = self.instances_transfer(X_train, X_pool, y_train, y_pool, indices, "qbc")
            for i in range(len(instances_transfer)):
                instances_pool_qbc.append(instances_transfer[i])
                targets_pool_qbc.append(targets_transfer[i])

        r2_auc = np.round(auc(self.epochs, R2[0,:-1]), 4)
        mse_auc = np.round(auc(self.epochs, MSE[0,:-1]), 4)
        mae_auc = np.round(auc(self.epochs, MAE[0,:-1]), 4) 
        ca_auc = np.round(auc(self.epochs, CA[0,:-1]), 4) 
        arrmse_auc = np.round(auc(self.epochs, ARRMSE[0,:-1]), 4) 


        R2[:,-1] = (r2_auc)
        MSE[:,-1] = (mse_auc)
        MAE[:,-1] = (mae_auc)
        CA[:,-1] = (ca_auc)
        ARRMSE[:,-1] = (arrmse_auc)

        cols = ["Target_{}".format(i+1) for epoch in range(self.n_epochs) for i in range(target_length)]
        Y_pred_df = pd.DataFrame(Y_pred, columns=cols)
        return R2, MSE, MAE, CA, ARRMSE, Y_pred_df, instances_pool_qbc, targets_pool_qbc