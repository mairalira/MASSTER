""" The subclass for the lower bound method. """
from models.active_learning import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, auc
from utils.metrics import *

class lowerbound(activelearning):
    def __init__(self, batch_size, n_epochs):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.epochs = [i for i in range(n_epochs)]
        self.random_R2 = np.zeros([self.iterations, self.n_epochs+1])
        self.random_MSE = np.zeros([self.iterations, self.n_epochs+1])
        self.random_MAE = np.zeros([self.iterations, self.n_epochs+1])
        self.random_CA = np.zeros([self.iterations, self.n_epochs+1])
        self.random_aRRMSE = np.zeros([self.iterations, self.n_epochs+1])
    
    def random(self, X_train, X_pool, X_test, y_train, y_test, target_length):
        # here, the instances from the unlabelled pool are chosen randomly
        y_test_preds = np.zeros(shape=(len(X_test), target_length))

        for i in range(target_length):
            # first, fit a model to the training set
            rf_regressor = RandomForestRegressor(n_estimators=self.n_trees, random_state=self.random_state)  # default value for max features is 1.0 => N features
            rf_regressor.fit(X_train, y_train[i])
            
            # next, predict the target values of the test set
            test_preds = rf_regressor.predict(X_test)
            y_test_preds[:,i] = np.round(test_preds, 5)

        random_indices = sorted(np.random.choice(range(len(X_pool)), size=self.batch_size, replace=False), reverse=True)  # create random indices which are non-repetitive
        return(random_indices, y_test, y_test_preds)
    
    def training(self, X_train, X_pool, X_test, y_train, y_pool, y_test, target_length):
    # the random based training loop    
        R2 = np.zeros([1, self.n_epochs+1])
        MSE = np.zeros([1, self.n_epochs+1])
        MAE = np.zeros([1, self.n_epochs+1])
        CA = np.zeros([1, self.n_epochs+1])
        ARRMSE = np.zeros([1, self.n_epochs+1])
        Y_pred = np.zeros([len(X_test), target_length*self.n_epochs])

        for i in range(self.n_epochs):
            print("Epoch {}:".format(i+1))
            print("     The training set size: {}".format(len(X_train)))
            print("     The unlabelled pool size: {}".format(len(X_pool)))

            y_train_coll = self.target_collect(y_train, target_length)
            indices, y_test, y_test_preds = self.random(X_train, X_pool, X_test, y_train_coll, y_test, target_length)
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
            
            _, _ = self.instances_transfer(X_train, X_pool, y_train, y_pool, indices, "random")

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

        return R2, MSE, MAE, CA, ARRMSE, Y_pred_df