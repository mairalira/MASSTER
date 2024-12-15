"""
Initialize common imports for the data module.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing
from numpy.random import RandomState
import sys
import os

# Absolute path using Path
project_root = Path(__file__).resolve().parent.parent

# Adding path to sys.path
sys.path.append(str(project_root))

# Import global variables setted on config
from config import DATA_PATH, DATASET_NAME, TRAIN_SIZE, POOL_SIZE, TEST_SIZE, RANDOM_STATE

print(f"Dataset Name: {DATASET_NAME}")

print(f"Dataset Name: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(df)