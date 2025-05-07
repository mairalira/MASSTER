import pandas as pd
import numpy as np
def load_dataset(path_dataset):
    dataset = pd.read_csv(path_dataset)
    dataset_x, dataset_y = split_feature_target(dataset)
    return dataset_x, dataset_y


def split_feature_target(dataset):
    x = dataset[[c for c in dataset.columns if "target" not in c]].values
    y = dataset[[c for c in dataset.columns if "target" in c]].values
    return x,y    

def concat_train_unlabeled(train_x,
                        train_y,
                        unlabeled_x
                        ):
    train_unlabeled_x = np.concatenate((train_x, unlabeled_x), axis=0)
    unlabeled_y = np.full((unlabeled_x.shape[0], train_y.shape[1]), "?").astype(object)
    train_unlabeled_y = np.concatenate((train_y, unlabeled_y), axis=0)   
    return train_unlabeled_x, train_unlabeled_y
