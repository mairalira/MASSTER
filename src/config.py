from pathlib import Path
from numpy.random import RandomState

# Project-specific root directory path (for reusability)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory path (relative to project root)
DATA_DIR = PROJECT_ROOT / "data"

# Define variables to run data-preprocessing
DATASET_NAME = 'edm'
TRAIN_SIZE = 0.1
POOL_SIZE = 0.7
TEST_SIZE = 0.2
DATA_PATH = DATA_DIR / 'raw' / f'{DATASET_NAME}.csv'
RANDOM_STATE = RandomState(seed=42)
