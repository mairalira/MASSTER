import sys
import pandas as pd
#from clus.models.mlc import MLCTree, MLCEnsemble
from clus.models.regression import RegressionTree, RegressionEnsemble

path_dataset = sys.argv[1]

train = pd.read_csv(path_dataset + "/train+pool_1")
test = pd.read_csv(path_dataset + "/test_1")

train_x = train[[c for c in train.columns if "target" not in c]].values
train_y = train[[c for c in train.columns if "target" in c]].values

#print(train_x)
#print(train_y)

test_x = train[[c for c in test.columns if "target" not in c]].values
test_y = train[[c for c in test.columns if "target" in c]].values


#ADICIONAR DADOS NAO ROTULADOS NO POOL, ESTA ERRADO

n_trees = 1

USE SELECT RANDOM SPACES AND SIMULATE RF
model = RegressionEnsemble(verbose = 1, forest=[], ssl=[],  n_trees = 2, Output_WritePerBagModelFile = "Yes", Output_WriteModelFile = "Yes", SemiSupervised_PossibleWeights = 0.5)
#model = RegressionTree(verbose = 0,  ssl=[],  SemiSupervised_PossibleWeights = 0.5)

model.fit(train_x, train_y)

print(model.predict(test_x))