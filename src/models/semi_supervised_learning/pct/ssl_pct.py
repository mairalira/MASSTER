from pyclus.clus.models.regression import RegressionTree
import numpy as np
from utils import concat_train_unlabeled

class SSLPCT:
    weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    verbose = -69
    SemiSupervised_SemiSupervisedMethod = "PCT"
    Ensemble_SelectRandomSubspaces = "SQRT"
    General_RandomSeed = 0
    Clus_Model = "Original" ### clus model name for the unpruned tree
    models = []
    random_state = 0
    np.random.seed(random_state)

    def __init__(self,
                 n_trees = 100,
                 possible_weights = [0.1, 0.3, 0.5, 0.7, 0.9]):
        self.n_trees = n_trees
        self.SemiSupervised_PossibleWeights = possible_weights
    
    
    def fit(self,
            x,
            y,
            unlabeled_x):

        for _ in range(self.n_trees):

            clf = RegressionTree(verbose = self.verbose,
                                General_RandomSeed = self.General_RandomSeed,
                                ssl=[], 
                                SemiSupervised_SemiSupervisedMethod = self.SemiSupervised_SemiSupervisedMethod,
                                SemiSupervised_PossibleWeights = self.SemiSupervised_PossibleWeights,
                                #Ensemble_SelectRandomSubspaces = 1
                                )
            bagged_x, bagged_y = self._bagging_sampling(x,
                                              y)
            train_unlabeled_x, train_unlabeled_y = concat_train_unlabeled(bagged_x,
                                                                       bagged_y,
                                                                       unlabeled_x)
            clf.fit(train_unlabeled_x,
                    train_unlabeled_y,
                    verbose = self.verbose)
#            print(train_unlabeled_x)
#            print(train_unlabeled_y)
            self.models.append(clf)

    def predict(self,
                x):
        predictions = np.mean([m.predict(x)[self.Clus_Model] for m in self.models], axis=0)
        return predictions

    def _bagging_sampling(self,
                            x,
                            y):
        
        indexes = np.random.choice(np.arange(x.shape[0]), size = x.shape[0],  replace = True)
        return x[indexes,:], y[indexes, :]
    