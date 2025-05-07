from pathlib import Path
from numpy.random import RandomState
import logging
import os
import argparse

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description='Active Learning Configuration')
parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset')
args = parser.parse_args()

# Project-specific root directory path (for reusability)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory path (relative to project root)
DATA_DIR = PROJECT_ROOT / "data"

# Define variables to run data-preprocessing
DATASET_NAME = args.dataset
K_FOLDS = 10
TRAIN_SIZE = 0.1
POOL_SIZE = 0.7
TEST_SIZE = 0.2
DATA_PATH = DATA_DIR / 'raw' / f'{DATASET_NAME}.csv'
RANDOM_STATE = RandomState(seed=42)
N_EPOCHS = 10 #15   
BATCH_PERCENTAGE = 5
BATCH_PERCENTAGE_SSL = BATCH_PERCENTAGE/2
N_TREES = 100
ITERATIONS = 10
CA_THRESHOLD = 0.1
THRESHOLD = 0.05 #0.1

# Define log structure
LOG_DIR = PROJECT_ROOT / "reports"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = LOG_DIR / 'output_log.log'

logging.basicConfig(
    level=logging.INFO,  # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)